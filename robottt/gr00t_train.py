"""Train the public GR00T N1.7 RoboTTT policy on six Insert Benchmark tasks."""

import argparse
from contextlib import nullcontext
from math import cos, pi
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .ddp_train import StepBatchSampler, _argument_values, _device, _emit, _save, _sync
from .gr00t import Gr00tRoboTTT
from .gr00t_cache import (
    BASE_REVISION, HORIZON, PROCESSOR_REVISION, STRIDE, UPSTREAM_REVISION,
    build_cache, cache_directory, episode_chunks, read_cached_segment,
)
from .gr00t_insert import marvin_statistics
from .policy import corrupt_actions
from .raw_data import manifest, scan_episodes


ADAPTER_CONFIG = {"registers": 16, "ttt_heads": 32, "stabilize_fast_weights": True}


def _flow_loss(velocity, target, action_mask, denominator):
    error = (velocity.float() - target.float()).square()
    return error.masked_fill(~action_mask, 0).sum() / denominator


def train_step(model, optimizer, selections, episodes, cache, args, device, rank, step):
    optimizer.zero_grad(set_to_none=True)
    local_loss = torch.zeros((), device=device)
    lengths = []
    noise_gen = torch.Generator(device=device).manual_seed(args.seed + step * 8192 + rank)
    for index, (episode_id, crop_seed, context) in enumerate(selections):
        episode = episodes[episode_id]
        chunks = episode_chunks(episode)
        length = min(chunks, context)
        first = int(np.random.default_rng(crop_seed).integers(chunks - length + 1))
        lengths.append(length)
        carry = None
        denominator = len(selections) * length * HORIZON * 16
        for offset in range(0, length, args.segment):
            stop = min(length, offset + args.segment)
            sync = index + 1 == len(selections) and stop == length
            sync_scope = model.no_sync() if isinstance(model, DistributedDataParallel) and not sync else nullcontext()
            batch = read_cached_segment(cache, episode, first + offset, first + stop, device)
            tau, noised, target = corrupt_actions(batch["actions"], generator=noise_gen)
            with sync_scope:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    velocity, carry = model(
                        batch["features"], batch["state"], noised, tau, carry,
                        embodiment_id=batch["embodiment_id"], vl_mask=batch["vl_mask"],
                        image_mask=batch["image_mask"], action_mask=batch["action_mask"],
                    )
                    loss = _flow_loss(velocity, target, batch["action_mask"], denominator)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite flow loss at {step=}, {rank=}, {episode.seed=}, {offset=}")
                loss.backward()
            carry = carry.detach()
            local_loss += loss.detach()
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return local_loss, grad.detach(), min(lengths), max(lengths)


def _settings(args):
    fields = ("global_batch", "steps", "context", "context_start", "context_warmup", "segment", "warmup", "seed")
    return {name: getattr(args, name) for name in fields}


def _metadata(args, data_sha256, world, stats):
    return {
        "backend": "gr00t_n1_7", "adapter_config": ADAPTER_CONFIG,
        "tune_marvin_projectors": True, "marvin_statistics": stats,
        "base_revision": BASE_REVISION, "processor_revision": PROCESSOR_REVISION,
        "isaac_gr00t_revision": UPSTREAM_REVISION, "data_sha256": data_sha256,
        "stride": STRIDE, "horizon": HORIZON, "world_size": world, "settings": _settings(args),
    }


def _adapter_state(policy):
    return {name: parameter.detach() for name, parameter in policy.named_parameters() if parameter.requires_grad}


def _load_resume(policy, optimizer, path, metadata):
    saved = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if any(saved.get(key) != value for key, value in metadata.items() if key != "marvin_statistics"):
        raise ValueError("Resume requires identical base, data, rollout and training settings")
    if set(saved["adapter"]) != set(_adapter_state(policy)):
        raise ValueError("Adapter keys differ from the current trainable parameters")
    policy.load_state_dict(saved["adapter"], strict=False)
    optimizer.load_state_dict(saved["optimizer"])
    return int(saved["step"])


def _lr_scale(args, step):
    warmup = min(args.warmup, args.steps - 1)
    if step <= warmup:
        return step / warmup
    decay_start = max(warmup, int(args.steps * 0.8))
    progress = max(0, step - decay_start) / (args.steps - decay_start)
    return 0.1 + 0.45 * (1 + cos(pi * progress))


def _enable_projectors(policy):
    modules = (policy.head.state_encoder, policy.head.action_encoder, policy.head.action_decoder)
    for module in modules:
        module.float().requires_grad_(True)
    return [parameter for module in modules for parameter in module.parameters()]


def train(args, rank, world, device):
    episodes = scan_episodes(args.data, HORIZON, 1, expected_per_task=50)
    data_sha256 = manifest(episodes, episodes, [])["sha256"]
    stats = marvin_statistics(episodes)
    cache = cache_directory(args.cache, data_sha256)
    train_episodes = episodes
    if args.smoke:
        longest = sorted(episodes, key=episode_chunks, reverse=True)
        train_episodes = longest[:args.global_batch]
        if episode_chunks(train_episodes[0]) < args.context or episode_chunks(train_episodes[-1]) <= args.segment:
            raise ValueError("Smoke requires at least one full context and multiple TBPTT segments per rank")
    local_episodes = train_episodes[rank::world]

    def cached(episode, index, total):
        _emit(event="vlm_cache_episode", rank=rank, episode=episode.seed, level=episode.level, done=index, total=total)

    build_cache(local_episodes, cache, args.base, args.processor_assets, stats, device, args.vlm_batch, cached)
    _sync()
    if args.cache_only:
        return
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(args.seed)
    model_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    policy = Gr00tRoboTTT.from_local_checkpoint(
        args.base, embodiment_id=10, device=device, dtype=model_dtype, ttt_dtype=torch.float32,
        **ADAPTER_CONFIG,
    )
    ttt = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    projectors = _enable_projectors(policy)
    optimizer = torch.optim.AdamW([
        {"params": ttt, "lr": 2e-5, "weight_decay": 1e-5},
        {"params": projectors, "lr": 5e-5, "weight_decay": 0.0},
    ], foreach=False)
    metadata = _metadata(args, data_sha256, world, stats)
    start = _load_resume(policy, optimizer, args.resume, metadata) if args.resume else 0
    if not 0 <= start < args.steps:
        raise ValueError(f"Unexpected completed optimizer step {start}")
    model = policy
    if world > 1:
        # TBPTT carries a structured output into the next segment. Let DDP
        # discover the loss graph on the final synchronized forward per rank.
        model = DistributedDataParallel(
            policy, device_ids=[device.index] if device.type == "cuda" else None,
            broadcast_buffers=False, gradient_as_bucket_view=True, find_unused_parameters=True,
        )
    model.train()
    if rank == 0:
        _emit(event="configuration", **_argument_values(args), world_size=world, data_sha256=data_sha256,
              base_revision=BASE_REVISION, horizon=HORIZON, stride=STRIDE, active_episodes=len(train_episodes),
              model_dtype=str(model_dtype), ttt_dtype="torch.float32",
              trainable_parameters=sum(p.numel() for p in ttt + projectors))
    sampling = iter(StepBatchSampler(args, len(train_episodes), rank, world, start))
    args.output.mkdir(parents=True, exist_ok=True)
    for step in range(start + 1, args.steps + 1):
        begun = perf_counter()
        scale = _lr_scale(args, step)
        for group, peak in zip(optimizer.param_groups, (2e-5, 5e-5), strict=True):
            group["lr"] = scale * peak
        selections = [next(sampling)[0] for _ in range(args.global_batch // world)]
        loss, grad, shortest, longest = train_step(model, optimizer, selections, train_episodes, cache, args, device, rank, step)
        if args.smoke and world > 1:
            gates = [torch.empty_like(policy.gates) for _ in range(world)]
            dist.all_gather(gates, policy.gates.detach())
            for value in gates[1:]:
                torch.testing.assert_close(gates[0], value, atol=0, rtol=0)
        if args.smoke or step == 1 or step % args.log_every == 0:
            metrics = torch.tensor([loss.item(), grad.item(), shortest, longest, perf_counter() - begun,
                                    torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0],
                                   device=device)
            peer_metrics = [torch.empty_like(metrics) for _ in range(world)]
            if world > 1:
                dist.all_gather(peer_metrics, metrics)
            else:
                peer_metrics = [metrics]
            if rank == 0:
                rows = torch.stack(peer_metrics).cpu()
                _emit(event="train", step=step, loss=rows[:, 0].mean().item(), grad_norm=grad.item(),
                      context_min=int(rows[:, 2].min().item()), context_max=int(rows[:, 3].max().item()),
                      step_seconds=rows[:, 4].max().item(), peak_gpu_gib=rows[:, 5].max().item())
        if step % args.save_every == 0 or step == args.steps:
            if rank == 0:
                path = args.output / f"step_{step:06d}.pt"
                _save(path, {**metadata, "step": step, "adapter": _adapter_state(policy), "optimizer": optimizer.state_dict()})
                latest = args.output / "latest.pt"
                temporary = args.output / f".latest.{os.getpid()}"
                try:
                    temporary.symlink_to(path.name)
                    temporary.replace(latest)
                finally:
                    temporary.unlink(missing_ok=True)
                _emit(event="checkpoint", step=step, path=str(path))
            _sync()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("data", "cache", "output", "base", "processor-assets"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--global-batch", type=int, default=64)
    parser.add_argument("--context", type=int, default=192)
    parser.add_argument("--context-start", type=int, default=32)
    parser.add_argument("--context-warmup", type=int, default=10000)
    parser.add_argument("--segment", type=int, default=8)
    parser.add_argument("--vlm-batch", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1745)
    parser.add_argument("--save-every", type=int, default=10000)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--require-a100-a800", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.micro_batch = 1
    rank, world = int(os.getenv("RANK", "0")), int(os.getenv("WORLD_SIZE", "1"))
    positive = (args.steps, args.global_batch, args.context, args.context_start, args.segment,
                args.vlm_batch, args.save_every, args.log_every)
    if min(positive) < 1 or min(args.warmup, args.context_warmup, args.seed) < 0:
        parser.error("Invalid nonpositive training setting")
    if args.global_batch % world or args.context_start > args.context:
        parser.error("Need a global batch divisible by world size and a nondecreasing context schedule")
    if args.smoke and (args.context_warmup or args.context_start != args.context or args.steps > 5):
        parser.error("Smoke uses a fixed context and at most five optimizer steps")
    if not args.resume and not args.cache_only and any(args.output.glob("step_*.pt")):
        parser.error("Checkpoint output already contains steps. Specify --resume or use a new output directory")
    device = _device(args, world)
    try:
        train(args, rank, world, device)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
