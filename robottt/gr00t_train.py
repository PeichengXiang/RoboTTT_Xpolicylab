"""Train the public GR00T N1.7 RoboTTT policy on six Insert Benchmark tasks."""

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
from math import cos, pi
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .ddp_train import StepBatchSampler, _argument_values, _device, _emit, _save, _sync
from .gr00t import Gr00tRoboTTT, load_public_backbone
from .gr00t_cache import (
    BASE_REVISION, HORIZON, PROCESSOR_REVISION, STRIDE, UPSTREAM_REVISION,
    build_cache, cache_directory, episode_chunks, load_processor, read_cached_segment, read_raw_segment,
)
from .gr00t_insert import marvin_statistics
from .policy import RolloutState, corrupt_actions
from .ttt import FastWeights
from .raw_data import manifest, scan_episodes


ADAPTER_CONFIG = {"registers": 16, "ttt_heads": 32, "stabilize_fast_weights": True}


def _flow_loss(velocity, target, action_mask, denominator):
    error = (velocity.float() - target.float()).square()
    return error.masked_fill(~action_mask, 0).sum() / denominator


def _trajectory_windows(selections, episodes, stride):
    windows = []
    for episode_id, crop_seed, context in selections:
        episode = episodes[episode_id]
        chunks = episode_chunks(episode, stride)
        length = min(chunks, context)
        first = int(np.random.default_rng(crop_seed).integers(chunks - length + 1))
        windows.append((episode, first, length))
    return windows


def _serial_backward(model, windows, cache, args, device, rank, step, processor, noise_gen):
    local_loss = torch.zeros((), device=device)
    if processor is not None:
        # Native post-training enables dropout. Rebuild its rank-local RNG at
        # every optimizer step so resuming reproduces the uninterrupted run.
        dropout_seed = args.seed + step * 8192 + rank + 2**40
        torch.random.default_generator.manual_seed(dropout_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(dropout_seed)
    for index, window in enumerate(windows):
        episode, first, length = window
        carry = None
        denominator = len(windows) * length * HORIZON * 16
        for offset in range(0, length, args.segment):
            stop = min(length, offset + args.segment)
            sync = index + 1 == len(windows) and stop == length
            sync_scope = model.no_sync() if isinstance(model, DistributedDataParallel) and not sync else nullcontext()
            if processor is None:
                batch = read_cached_segment(cache, episode, first + offset, first + stop, device)
                vision = {"vl_mask": batch["vl_mask"], "image_mask": batch["image_mask"]}
            else:
                batch = read_raw_segment(cache, episode, first + offset, first + stop, device,
                                         stride=args.stride, processor=processor)
                vision = {"raw_vl_inputs": batch["vl_inputs"]}
            tau, noised, target = corrupt_actions(batch["actions"], generator=noise_gen)
            with sync_scope:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    velocity, carry = model(
                        batch.get("features"), batch["state"], noised, tau, carry,
                        embodiment_id=batch["embodiment_id"], action_mask=batch["action_mask"], **vision,
                    )
                    loss = _flow_loss(velocity, target, batch["action_mask"], denominator)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite flow loss at {step=}, {rank=}, {episode.seed=}, {offset=}")
                loss.backward()
            carry = carry.detach()
            local_loss += loss.detach()
    return local_loss


@dataclass
class _PreparedWindow:
    tensors: dict
    length: int
    vl_width: int


def _prepare_windows(windows, cache, device, segment, noise_gen):
    prepared = []
    for episode, first, length in windows:
        batch = read_cached_segment(cache, episode, first, first + length, device)
        # Keep original generator calls and per-trajectory / per-segment shapes,
        # including the short tail, before sorting or padding trajectories.
        noise = [corrupt_actions(chunk, generator=noise_gen) for chunk in batch.pop("actions").split(segment, dim=1)]
        for name, values in zip(("tau", "noised", "target"), zip(*noise, strict=True), strict=True):
            batch[name] = torch.cat(values, dim=1)
        prepared.append(_PreparedWindow(batch, length, batch["features"].shape[2]))
    return sorted(prepared, key=lambda window: window.length, reverse=True)


def _stack_segment(windows, offset, stop):
    # A repeated last observation supplies well-formed attention masks for time
    # padding. Padded queries never update TTT or contribute to the loss.
    width = max(window.vl_width for window in windows)
    parts = {}
    for window in windows:
        end = min(stop, window.length)
        for name, value in window.tensors.items():
            row = value[:, offset:end]
            if end < stop:
                tail = row[:, -1:].expand(-1, stop - end, *row.shape[2:])
                row = torch.cat((row, tail), dim=1)
            if name in ("features", "vl_mask", "image_mask") and window.vl_width < width:
                padding = (0, 0, 0, width - window.vl_width) if name == "features" else (0, width - window.vl_width)
                row = torch.nn.functional.pad(row, padding)
            parts.setdefault(name, []).append(row)
    return {k: torch.cat(rows, dim=0) for k, rows in parts.items()}


def _batched_pretrain_backward(model, windows, cache, args, device, rank, step, noise_gen):
    prepared = _prepare_windows(windows, cache, device, args.segment, noise_gen)
    total_loss = torch.zeros((), device=device)
    width = args.pretrain_micro_batch
    for begin in range(0, len(prepared), width):
        group = prepared[begin:begin + width]
        lengths = torch.tensor([window.length for window in group], device=device)
        longest = group[0].length
        carry = None
        for offset in range(0, longest, args.segment):
            stop = min(offset + args.segment, longest)
            # Finished lanes are at the end because crops are sorted by length.
            remaining = [window for window in group if window.length > offset]
            size = len(remaining)
            if size != len(group):
                group = remaining
                lengths = lengths[:size]
                carry = RolloutState(tuple(FastWeights(s.w1[:size], s.b1[:size], s.w2[:size], s.b2[:size])
                                           for s in carry.fast), carry.index)
            batch = _stack_segment(group, offset, stop)
            valid = torch.arange(offset, stop, device=device)[None, :] < lengths[:, None]
            vl_lengths = tuple(window.vl_width for window in group)
            sync = begin + width >= len(prepared) and stop == longest
            scope = model.no_sync() if isinstance(model, DistributedDataParallel) and not sync else nullcontext()
            with scope:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    velocity, carry = model(
                        batch["features"], batch["state"], batch["noised"], batch["tau"], carry, valid,
                        embodiment_id=batch["embodiment_id"], action_mask=batch["action_mask"],
                        vl_mask=batch["vl_mask"], image_mask=batch["image_mask"], vl_lengths=vl_lengths,
                    )
                    error = (velocity.float() - batch["target"].float()).square()
                    mask = batch["action_mask"] & valid[:, :, None, None]
                    sums = error.masked_fill(~mask, 0).sum(dim=(1, 2, 3))
                    loss = (sums / (len(prepared) * HORIZON * 16 * lengths)).sum()
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite batched pretraining loss at {step=}, {rank=}, {offset=}")
                loss.backward()
            carry = carry.detach()
            total_loss += loss.detach()
    return total_loss


def train_step(model, optimizer, selections, episodes, cache, args, device, rank, step, *, processor=None):
    optimizer.zero_grad(set_to_none=True)
    windows = _trajectory_windows(selections, episodes, args.stride)
    lengths = [length for _, _, length in windows]
    noise_gen = torch.Generator(device=device).manual_seed(args.seed + step * 8192 + rank)
    if getattr(args, "stage", "adapter") == "pretrain" and getattr(args, "pretrain_micro_batch", 1) > 1:
        local_loss = _batched_pretrain_backward(model, windows, cache, args, device, rank, step, noise_gen)
    else:
        local_loss = _serial_backward(model, windows, cache, args, device, rank, step, processor, noise_gen)
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    if processor is not None and (args.smoke or step == args.phase_boundary + 1):
        policy = model.module if isinstance(model, DistributedDataParallel) else model
        prefixes = {"vision": "backbone.model.model.visual.", "language": "backbone.model.model.language_model.",
                    "dit": "head.model.transformer_blocks.", "ttt": "memories."}
        evidence = {}
        for branch, prefix in prefixes.items():
            for name, param in policy.named_parameters():
                if name.startswith(prefix) and param.grad is not None:
                    norm = param.grad.float().norm().item()
                    if norm > 0:
                        evidence[branch] = {"parameter": name, "grad_norm": norm}
                        break
        if evidence.keys() != prefixes.keys():
            raise RuntimeError(f"Missing nonzero full-model gradients at {step=} {rank=}: {evidence}")
        if rank == 0:
            _emit(event="full_model_gradients", step=step, branches=evidence)
    optimizer.step()
    return local_loss, grad.detach(), min(lengths), max(lengths), sum(lengths)


def _settings(args):
    fields = ("global_batch", "steps", "context", "context_start", "context_warmup", "segment", "warmup", "seed")
    return {name: getattr(args, name) for name in fields}


def _metadata(args, data_sha256, world, stats):
    adapter_config = ADAPTER_CONFIG.copy()
    if args.ttt_variant == "reference":
        adapter_config["reference_style"] = True
    metadata = {
        "backend": "gr00t_n1_7", "adapter_config": adapter_config,
        "tune_marvin_projectors": args.stage != "pretrain", "marvin_statistics": stats,
        "base_revision": BASE_REVISION, "processor_revision": PROCESSOR_REVISION,
        "isaac_gr00t_revision": UPSTREAM_REVISION, "data_sha256": data_sha256,
        "stride": args.stride, "horizon": HORIZON, "world_size": world, "settings": _settings(args),
    }
    if args.stage != "adapter":
        metadata.update(model_stage=args.stage, phase_boundary=args.phase_boundary, recipe="paper_two_stage")
    if args.stage == "pretrain" and getattr(args, "pretrain_micro_batch", 1) > 1:
        metadata["pretrain_micro_batch"] = args.pretrain_micro_batch
    return metadata


def _adapter_state(policy):
    return {name: parameter.detach() for name, parameter in policy.named_parameters() if parameter.requires_grad}


def _load_resume(policy, optimizer, path, metadata, stop_at, rank=0):
    saved = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if any(saved.get(key) != value for key, value in metadata.items() if key != "marvin_statistics"):
        raise ValueError("Resume requires identical base, data, rollout and training settings")
    if saved.get("pretrain_micro_batch", 1) != metadata.get("pretrain_micro_batch", 1):
        raise ValueError("Resume requires the original pretraining microbatch")
    completed = int(saved["step"])
    if not 0 <= completed < metadata["settings"]["steps"]:
        raise ValueError(f"Unexpected completed optimizer step {completed}")
    if completed >= stop_at:
        raise ValueError(f"Checkpoint already completed step {completed}. Set --stop-after above that step or omit it")
    if set(saved["adapter"]) != set(_adapter_state(policy)):
        raise ValueError("Adapter keys differ from the current trainable parameters")
    policy.load_state_dict(saved["adapter"], strict=False)
    if "optimizer_shards" in saved:
        shard = torch.load(path.parent / saved["optimizer_shards"][rank], map_location="cpu",
                           mmap=True, weights_only=True)
        if (shard["step"], shard["rank"], shard["world_size"]) != (completed, rank, metadata["world_size"]):
            raise ValueError("Wrong optimizer shard for this checkpoint")
        optimizer.optim.load_state_dict(shard["optimizer"])
    else:
        optimizer.load_state_dict(saved["optimizer"])
    return completed


def _initialize_posttrain(policy, path, metadata):
    saved = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    same = ("backend", "adapter_config", "base_revision", "processor_revision", "isaac_gr00t_revision",
            "data_sha256", "stride", "horizon", "recipe", "phase_boundary")
    if (any(saved.get(key) != metadata[key] for key in same) or saved.get("model_stage") != "pretrain"
            or saved["step"] != metadata["phase_boundary"] or set(saved["adapter"]) != set(_adapter_state(policy))):
        raise ValueError("Post-training requires the matching final TTT-only pretraining checkpoint")
    policy.load_state_dict(saved["adapter"], strict=False)
    return saved["step"]


def _lr_scale(args, step):
    if args.stage == "posttrain":
        # The paper gives a cosine peak and leaves the endpoint unspecified.
        progress = (step - args.phase_boundary - 1) / max(1, args.steps - args.phase_boundary - 1)
        return (1 + cos(pi * progress)) / 2
    duration = args.phase_boundary if args.stage == "pretrain" else args.steps
    warmup = min(args.warmup, duration - 1)
    if step <= warmup:
        return step / warmup
    decay_start = max(warmup, int(duration * 0.8))
    progress = max(0, step - decay_start) / (duration - decay_start)
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
    cache = cache_directory(args.cache, data_sha256, args.stride)
    dense_cache = (
        cache_directory(args.cache, data_sha256, STRIDE)
        if args.stride > STRIDE and args.stride % STRIDE == 0 else None
    )
    train_episodes = episodes
    if args.smoke:
        longest = sorted(episodes, key=lambda episode: episode_chunks(episode, args.stride), reverse=True)
        train_episodes = longest[:args.global_batch]
        if (episode_chunks(train_episodes[0], args.stride) < args.context
                or episode_chunks(train_episodes[-1], args.stride) <= args.segment):
            raise ValueError("Smoke requires at least one full context and multiple TBPTT segments per rank")
    local_episodes = train_episodes[rank::world]

    def cached(episode, index, total):
        _emit(event="vlm_cache_episode", rank=rank, episode=episode.seed, level=episode.level, done=index, total=total)

    counts = build_cache(
        local_episodes, cache, args.base, args.processor_assets, stats, device, args.vlm_batch, cached,
        stride=args.stride, dense_cache=dense_cache,
    )
    _emit(event="vlm_cache_summary", rank=rank, stride=args.stride, episodes=len(local_episodes), **counts)
    _sync()
    if args.cache_only:
        return
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(args.seed)
    full = args.stage == "posttrain"
    model_dtype = torch.bfloat16 if device.type == "cuda" and not full else torch.float32
    metadata = _metadata(args, data_sha256, world, stats)
    policy = Gr00tRoboTTT.from_local_checkpoint(
        args.base, embodiment_id=10, device=device, dtype=model_dtype, ttt_dtype=torch.float32,
        **metadata["adapter_config"],
    )
    start = _initialize_posttrain(policy, args.init_from, metadata) if args.init_from else 0
    processor = None
    if full:
        processor = load_processor(args.base, args.processor_assets, stats)
        backbone = load_public_backbone(args.base, args.processor_assets, device=device, dtype=torch.float32,
                                        trainable=True)
        policy.enable_full_posttrain(backbone)
    parameters = [param for param in policy.parameters() if param.requires_grad]
    groups = [{"params": parameters, "lr": 5e-5 if full else 2e-5, "weight_decay": 1e-5}]
    if args.stage == "adapter":
        groups.append({"params": _enable_projectors(policy), "lr": 5e-5, "weight_decay": 0.0})
    peaks = [group["lr"] for group in groups]
    sharded_optimizer = full and world > 1
    if sharded_optimizer:
        from torch.distributed.optim import ZeroRedundancyOptimizer

        optimizer = ZeroRedundancyOptimizer(groups, optimizer_class=torch.optim.AdamW, foreach=False)
    else:
        optimizer = torch.optim.AdamW(groups, foreach=False)
    stage_end = args.phase_boundary if args.stage == "pretrain" else args.steps
    stop_at = min(args.stop_after, stage_end) if args.stop_after is not None else stage_end
    if args.resume:
        start = _load_resume(policy, optimizer, args.resume, metadata, stop_at, rank)
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
              base_revision=BASE_REVISION, horizon=HORIZON, active_episodes=len(train_episodes),
              model_dtype=str(model_dtype), ttt_dtype="torch.float32",
              trainable_parameters=sum(p.numel() for p in policy.parameters() if p.requires_grad),
              base_trainable_parameters=sum(p.numel() for p in policy.head.parameters() if p.requires_grad),
              vlm_trainable_parameters=sum(p.numel() for p in backbone.parameters() if p.requires_grad) if full else 0)
    sampling = iter(StepBatchSampler(args, len(train_episodes), rank, world, start))
    args.output.mkdir(parents=True, exist_ok=True)
    for step in range(start + 1, stop_at + 1):
        begun = perf_counter()
        scale = _lr_scale(args, step)
        for group, peak in zip(optimizer.param_groups, peaks, strict=True):
            group["lr"] = scale * peak
        selections = [next(sampling)[0] for _ in range(args.global_batch // world)]
        loss, grad, shortest, longest, context_count = train_step(
            model, optimizer, selections, train_episodes, cache, args, device, rank, step, processor=processor,
        )
        if args.smoke and world > 1:
            gates = [torch.empty_like(policy.gates) for _ in range(world)]
            dist.all_gather(gates, policy.gates.detach())
            for value in gates[1:]:
                torch.testing.assert_close(gates[0], value, atol=0, rtol=0)
        if args.smoke or step == 1 or step % args.log_every == 0 or (args.stop_after is not None and step == stop_at):
            metrics = torch.tensor([loss.item(), grad.item(), shortest, longest, context_count, perf_counter() - begun,
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
                      context_total=int(rows[:, 4].sum().item()), context_rank_max=int(rows[:, 4].max().item()),
                      step_seconds=rows[:, 5].max().item(), peak_gpu_gib=rows[:, 6].max().item())
        if step % args.save_every == 0 or step == stop_at:
            path = args.output / f"step_{step:06d}.pt"
            if sharded_optimizer:
                # Publish all optimizer shards before the weight file used for resume.
                shards = [f"{path.stem}.optimizer-rank{i}.pt" for i in range(world)]
                _save(args.output / shards[rank], {"step": step, "rank": rank, "world_size": world,
                                                   "optimizer": optimizer.optim.state_dict()})
                _sync()
            if rank == 0:
                state = {"optimizer_shards": shards} if sharded_optimizer else {"optimizer": optimizer.state_dict()}
                _save(path, {**metadata, "step": step, "adapter": _adapter_state(policy), **state})
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
    parser.add_argument("--stage", choices=("adapter", "pretrain", "posttrain"), default="adapter")
    parser.add_argument("--phase-boundary", type=int, default=30000)
    parser.add_argument("--init-from", type=Path, help="Completed stage-one checkpoint for full post-training")
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--stop-after", type=int,
                        help="Save and stop at this absolute optimizer step while preserving the --steps schedule")
    parser.add_argument("--global-batch", type=int, default=64)
    parser.add_argument("--pretrain-micro-batch", type=int, choices=(1, 2, 4), default=1,
                        help="Trajectories processed together per rank during TTT-only pretraining")
    parser.add_argument("--ttt-variant", choices=("baseline", "reference"), default="baseline",
                        help="Reference adds pre-QKV RMSNorm and shares each chunk's RoPE position across tokens")
    parser.add_argument("--stride", type=int, default=STRIDE, help="Raw-frame interval between policy windows (native horizon: 40)")
    parser.add_argument("--context", type=int, help="Maximum policy contexts (default matches the span of 192 stride-8 contexts)")
    parser.add_argument("--context-start", type=int, help="Initial policy contexts (default matches the span of 32 stride-8 contexts)")
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
    if args.stride < 1:
        parser.error("--stride must be positive")
    if args.context is None:
        args.context = 191 * STRIDE // args.stride + 1
    if args.context_start is None:
        args.context_start = 31 * STRIDE // args.stride + 1
    positive = (args.steps, args.global_batch, args.context, args.context_start, args.segment,
                args.vlm_batch, args.save_every, args.log_every)
    if min(positive) < 1 or min(args.warmup, args.context_warmup, args.seed) < 0:
        parser.error("Invalid nonpositive training setting")
    if args.stop_after is not None and args.stop_after < 1:
        parser.error("--stop-after must be positive")
    if args.stage != "adapter" and not 1 <= args.phase_boundary < args.steps:
        parser.error("Two-stage training requires 0 < --phase-boundary < --steps")
    if args.stage == "posttrain":
        if bool(args.resume) == bool(args.init_from):
            parser.error("Post-training requires exactly one of --init-from and --resume")
        if args.stop_after is not None and args.stop_after <= args.phase_boundary:
            parser.error("Post-training must end after the stage boundary")
    elif args.init_from:
        parser.error("--init-from is only valid for post-training")
    if args.global_batch % world or args.context_start > args.context:
        parser.error("Need a global batch divisible by world size and a nondecreasing context schedule")
    if args.smoke and (args.context_warmup or args.context_start != args.context or args.steps > 5):
        parser.error("Smoke uses a fixed context and at most five optimizer steps")
    existing_steps = list(args.output.glob("step_[0-9][0-9][0-9][0-9][0-9][0-9].pt"))
    if args.init_from:
        existing_steps = [p for p in existing_steps if int(p.stem[5:]) > args.phase_boundary]
    if not args.resume and not args.cache_only and existing_steps:
        parser.error("Checkpoint output already contains steps. Specify --resume or use a new output directory")
    device = _device(args, world)
    try:
        train(args, rank, world, device)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
