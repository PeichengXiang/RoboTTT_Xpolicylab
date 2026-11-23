"""Serve a compact RoboTTT checkpoint to one local insert_benchmark simulator."""

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

from .data import rgb_tokens
from .isaac_export import CAMERAS
from .policy import PolicyConfig, RoboTTT


def _joint_bounds(benchmark):
    assets = benchmark / "assets/marvin_dual"
    manifest = json.loads((assets / "manifest.json").read_text(encoding="utf-8"))
    bounds = []
    for side in ("left", "right"):
        path = assets / manifest[side]["source"] / "asset_info.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        for name in metadata["arm_joint_names"]:
            joint = metadata["joints"][name]
            bounds.append((joint["lower"], joint["upper"]))
        bounds.append((0.0, 1.0))
    limits = np.asarray(bounds, dtype=np.float32)
    if limits.shape != (16, 2) or not np.isfinite(limits).all() or np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError("Invalid Marvin arm or gripper bounds")
    return limits.T


class IsaacPolicy:
    """Cache the latest frame and commit fast weights once per requested action chunk."""

    def __init__(self, policy, device, *, limits, dimensions, grid, shape, fps, denoise_steps, seed):
        from XPolicyLab.utils.process_data import pack_robot_state, unpack_robot_state

        self._policy = policy
        self._device = device
        self._limits = limits
        self._dimensions = dimensions
        self._grid = grid
        self._shape = shape
        self._fps = fps
        self._denoise_steps = denoise_steps
        self._seed = seed
        self._pack = pack_robot_state
        self._unpack = unpack_robot_state
        self.reset()

    def reset(self):
        self._memory = None
        self._observation = None
        self._chunks = 0

    def on_trial_end(self, _result=None):
        self.reset()

    def update_obs(self, obs):
        if "state" not in obs or "vision" not in obs:
            raise ValueError("Expected an insert_benchmark observation with state and vision")
        rate = obs.get("additional_info", {}).get("frequency", self._fps)
        if rate != self._fps:
            raise ValueError(f"Expected {self._fps} Hz simulator observations, got {rate}")
        self._observation = obs

    def get_action(self):
        if self._observation is None:
            raise RuntimeError("A fresh update_obs is required before each get_action")
        obs = self._observation
        cameras = [np.asarray(obs["vision"][name]["color"]) for name in CAMERAS]
        if any(image.shape != self._shape or image.dtype != np.uint8 for image in cameras):
            raise ValueError(f"Expected {self._shape} uint8 RGB for {CAMERAS}")
        context = torch.from_numpy(rgb_tokens(np.stack(cameras)[None], grid=self._grid)).to(self._device)
        joints = np.asarray(self._pack(obs, "joint", self._dimensions), dtype=np.float32)
        if joints.shape != (16,) or not np.isfinite(joints).all():
            raise ValueError("Expected 16 finite arm and gripper coordinates")
        state = torch.from_numpy(joints[None]).to(self._device)
        generator = torch.Generator(device=self._device).manual_seed(self._seed + self._chunks)
        chunk, memory = self._policy.sample(
            context, state, self._memory, denoise_steps=self._denoise_steps, generator=generator,
        )
        actions = chunk[0].float().cpu().numpy()
        if actions.shape != (self._policy.config.horizon, 16) or not np.isfinite(actions).all():
            raise RuntimeError("Policy generated an invalid joint-action chunk")
        bounded = np.clip(actions, *self._limits)
        result = self._unpack(bounded, "joint", self._dimensions)
        self._memory = memory
        self._chunks += 1
        self._observation = None
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Compact model trained on isaac_export NPZ files")
    parser.add_argument("--benchmark", type=Path, default=os.environ.get("INSERT_BENCHMARK_ROOT"))
    parser.add_argument("--xpolicylab", type=Path, default=os.environ.get("XPOLICYLAB_ROOT"))
    parser.add_argument("--grid", type=int, default=4, help="Same grid as isaac_export")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--denoise-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--port", type=int, default=19000)
    args = parser.parse_args()
    if args.benchmark is None:
        parser.error("Set --benchmark or INSERT_BENCHMARK_ROOT to the insert_benchmark checkout")
    if min(args.grid, args.threads, args.denoise_steps) < 1 or args.seed < 0 or not 1 <= args.port <= 65535:
        parser.error("Use positive grid, threads and denoise steps, a non-negative seed and a valid port")

    benchmark = args.benchmark.expanduser().resolve()
    source = benchmark / "src"
    if not (source / "insert_benchmark/xpolicylab_registration.py").is_file():
        parser.error(f"Cannot find insert_benchmark source at {source}")
    sys.path.insert(0, str(source))
    from insert_benchmark.benchmark.recording import CAMERAS as BENCHMARK_CAMERAS, FPS, HEIGHT, WIDTH
    from insert_benchmark.xpolicylab_registration import DEFAULT_CHECKOUT, DIMENSIONS, activate_workspace

    try:
        width = activate_workspace(args.xpolicylab or DEFAULT_CHECKOUT)
        limits = _joint_bounds(benchmark)
    except (OSError, KeyError, ValueError) as exc:
        parser.error(str(exc))
    if CAMERAS != BENCHMARK_CAMERAS or args.grid > min(HEIGHT, WIDTH):
        parser.error("Camera order or patch grid does not match the supported benchmark")

    from client_server.ws.model_server import PolicyServer, PolicyServerConfig

    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    saved = torch.load(args.checkpoint, map_location=device, weights_only=True)
    config = PolicyConfig(**saved["config"])
    if saved.get("dataset") != "npz" or (config.context_dim, config.state_dim, config.action_dim) != (6, width, width):
        parser.error("Checkpoint must use exported RGB patch tokens and absolute Marvin actions")
    stored_spec = saved.get("data_spec", {})
    if stored_spec.get("tokens_per_obs", len(CAMERAS) * args.grid**2) != len(CAMERAS) * args.grid**2:
        parser.error("The checkpoint token count does not match --grid")
    policy = RoboTTT(config).to(device).eval()
    policy.load_state_dict(saved["model"])
    adapter = IsaacPolicy(
        policy, device, limits=limits, dimensions=DIMENSIONS, grid=args.grid, shape=(HEIGHT, WIDTH, 3),
        fps=FPS, denoise_steps=args.denoise_steps, seed=args.seed,
    )
    host = "127.0.0.1"
    print(f"Serving {args.checkpoint} to one Isaac Sim client at ws://{host}:{args.port}", flush=True)
    asyncio.run(PolicyServer(adapter, PolicyServerConfig(host=host, port=args.port)).serve_forever())


if __name__ == "__main__":
    main()
