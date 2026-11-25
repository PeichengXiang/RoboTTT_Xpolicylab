"""XPolicyLab inference for compact and native GR00T RoboTTT on Marvin."""

import math
import os
from pathlib import Path

import numpy as np
import torch

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import (
    get_batch_size,
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)

from robottt.data import rgb_tokens
from robottt.isaac_export import CAMERAS
from robottt.isaac_serve import _joint_bounds
from robottt.policy import PolicyConfig, RoboTTT


POLICY_DIR = Path(__file__).resolve().parent
SIM_ENV = "marvin_dual_gripper"


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.action_type = model_cfg["action_type"]
        self.env_cfg_type = model_cfg["env_cfg_type"]
        if (self.env_cfg_type, self.action_type) != (SIM_ENV, "joint"):
            raise ValueError(f"RoboTTT supports env_cfg_type={SIM_ENV}, action_type=joint")
        self.dimensions = get_robot_action_dim_info(self.env_cfg_type)
        self.action_dim = sum(self.dimensions["arm_dim"]) + sum(self.dimensions["ee_dim"])
        self.batch_size = get_batch_size(self.env_cfg_type)
        self.device = torch.device(model_cfg["device"] if model_cfg.get("device") else
                                   "cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cpu":
            torch.set_num_threads(int(model_cfg.get("cpu_threads", 2)))

        self.backend = os.environ.get("ROBOTTT_BACKEND") or model_cfg.get("backend", "compact")
        self.denoise_steps = int(model_cfg.get("denoise_steps", 4))
        if self.denoise_steps < 1:
            raise ValueError("denoise_steps must be positive")
        self.seed = int(model_cfg.get("seed") or 0)
        self.reset_memory_each_chunk = bool(model_cfg.get("reset_memory_each_chunk", False))
        self.lower, self.upper = self._action_bounds(model_cfg)

        root = resolve_checkpoint_root(model_cfg, POLICY_DIR / "checkpoints", policy_dir=POLICY_DIR)
        if self.backend == "compact":
            self._load_compact(root, model_cfg)
        elif self.backend == "gr00t_n1_7":
            from .gr00t_backend import Gr00tMarvin

            self.model = Gr00tMarvin(root, model_cfg, self.device, self.action_dim, self.seed)
            self.execution_horizon = self.model.execution_horizon
        else:
            raise ValueError(f"Unsupported RoboTTT backend {self.backend!r}")
        self.reset()

    def _load_compact(self, root, model_cfg):
        checkpoint = root / "latest.pt" if root.is_dir() else root
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.config = PolicyConfig(**saved["config"])
        if (saved.get("dataset") != "npz" or self.config.context_dim != 6
                or (self.config.state_dim, self.config.action_dim) != (self.action_dim, self.action_dim)):
            raise ValueError("Expected a trained RoboTTT checkpoint for Marvin joint and RGB inputs")

        preprocessing = saved.get("preprocessing", {})
        self.task_token = bool(preprocessing.get("task_token", False))
        self.state_mean = self._normalization(preprocessing, "state_mean", 0)
        self.state_scale = self._normalization(preprocessing, "state_scale", 1)
        self.action_mean = self._normalization(preprocessing, "action_mean", 0)
        self.action_scale = self._normalization(preprocessing, "action_scale", 1)
        if np.any(self.state_scale <= 0) or np.any(self.action_scale <= 0):
            raise ValueError("Checkpoint action and state scales must be positive")
        self.task_name = model_cfg.get("task_name")
        token_count = saved.get("data_spec", {}).get("tokens_per_obs")
        inferred_grid = math.isqrt(token_count // len(CAMERAS)) if token_count else 4
        self.grid = int(model_cfg.get("grid") or preprocessing.get("grid") or inferred_grid)
        expected_tokens = len(CAMERAS) * self.grid ** 2 + self.task_token
        if self.grid < 1 or (token_count and token_count != expected_tokens):
            raise ValueError("Configured grid does not match the trained token count")
        self.model = RoboTTT(self.config).to(self.device).eval()
        self.model.load_state_dict(saved["model"])
        self.execution_horizon = self.config.horizon
        print(f"[RoboTTT] loaded {checkpoint} on {self.device} ({self.grid} × {self.grid} RGB patches)", flush=True)

    def _normalization(self, preprocessing, name, default):
        value = np.asarray(preprocessing.get(name, [default] * self.action_dim), dtype=np.float32)
        if value.shape != (self.action_dim,) or not np.isfinite(value).all():
            raise ValueError(f"Invalid checkpoint preprocessing field {name}")
        return value

    def _encode_context(self, frames, obs):
        rgb = np.stack(frames, axis=0)[None]
        if not self.task_token:
            return rgb_tokens(rgb, grid=self.grid)
        from robottt.raw_data import LEVELS, encode_rgb_context, infer_level

        if self.task_name in LEVELS:
            level = self.task_name
        else:
            instruction = obs.get("instruction") or obs.get("instructions") or ""
            level = infer_level(instruction)
        return encode_rgb_context(rgb, self.grid, level)

    def _action_bounds(self, cfg):
        root = cfg.get("benchmark_root") or os.getenv("INSERT_BENCHMARK_ROOT")
        if root:
            return _joint_bounds(Path(root).expanduser())
        lower = np.full(self.action_dim, -np.inf, dtype=np.float32)
        upper = np.full(self.action_dim, np.inf, dtype=np.float32)
        offset = 0
        for arm_width, gripper_width in zip(self.dimensions["arm_dim"], self.dimensions["ee_dim"], strict=True):
            start = offset + arm_width
            offset = start + gripper_width
            lower[start:offset] = 0.0
            upper[start:offset] = 1.0
        return lower, upper

    def reset(self):
        self.observations = {}
        self.memories = {}
        self.chunks = {}

    def on_trial_end(self, result=None):
        self.reset()

    def update_obs(self, obs):
        self.observations[0] = obs

    def update_obs_batch(self, obs_list):
        entries = {int(obs.get("env_idx", index)): obs for index, obs in enumerate(obs_list)}
        if len(entries) != len(obs_list):
            raise ValueError("A batch must provide at most one observation per env_idx")
        self.observations = entries

    def _predict(self, env_idx):
        obs = self.observations.get(env_idx)
        if obs is None:
            raise RuntimeError(f"Missing fresh observation for env_idx={env_idx}")
        rate = obs.get("additional_info", {}).get("frequency")
        if rate is not None and not math.isclose(float(rate), 30.0):
            raise ValueError(f"Marvin training observations are 30 Hz, received {rate}")
        frames = [np.asarray(obs["vision"][camera]["color"]) for camera in CAMERAS]
        if any(frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8 for frame in frames):
            raise ValueError("RoboTTT expects decoded HWC RGB uint8 camera observations")
        state = np.asarray(pack_robot_state(obs, self.action_type, self.dimensions), dtype=np.float32)
        if state.shape != (self.action_dim,) or not np.isfinite(state).all():
            raise ValueError("Expected a finite Marvin joint state")
        chunk_count = self.chunks.get(env_idx, 0)
        generator = torch.Generator(device=self.device).manual_seed(
            self.seed + 100_003 * env_idx + chunk_count
        )
        memory = None if self.reset_memory_each_chunk else self.memories.get(env_idx)
        if self.backend == "gr00t_n1_7":
            instruction = obs.get("instruction") or obs.get("instructions")
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError("GR00T RoboTTT requires a nonempty instruction")
            actions, result_memory = self.model.sample(
                frames, state, instruction, memory,
                denoise_steps=self.denoise_steps, generator=generator,
            )
        else:
            context = self._encode_context(frames, obs)
            state = (state - self.state_mean) / self.state_scale
            output, result_memory = self.model.sample(
                torch.from_numpy(context).to(self.device),
                torch.from_numpy(state[None]).to(self.device),
                memory, denoise_steps=self.denoise_steps, generator=generator,
            )
            actions = output[0].float().cpu().numpy() * self.action_scale + self.action_mean
        if actions.shape != (self.execution_horizon, self.action_dim) or not np.isfinite(actions).all():
            raise RuntimeError("RoboTTT returned an invalid joint chunk")
        actions = unpack_robot_state(np.clip(actions, self.lower, self.upper), self.action_type, self.dimensions)
        return actions, result_memory, chunk_count + 1

    def get_action(self):
        return self.get_action_batch([0])[0]

    def get_action_batch(self, env_idx_list=None):
        indices = list(range(self.batch_size) if env_idx_list is None else env_idx_list)
        if len(indices) != len(set(indices)):
            raise ValueError("env_idx_list must contain unique environment indices")
        results = [self._predict(index) for index in indices]
        for index, (_, memory, chunk) in zip(indices, results, strict=True):
            self.memories[index] = memory
            self.chunks[index] = chunk
            del self.observations[index]
        return [actions for actions, _, _ in results]
