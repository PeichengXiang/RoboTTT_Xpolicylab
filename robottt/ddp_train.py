"""Train the compact RoboTTT on all six Insert Benchmark tasks with torchrun."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict
import json
from math import cos, pi
import os
from pathlib import Path
import re
from socket import gethostname
from time import perf_counter

import numpy as np
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from .data import collate_episodes
from .policy import PolicyConfig, RoboTTT, corrupt_actions
from .raw_data import CachedEpisodes, cache_episode, manifest, preprocessing, read_state_actions, scan_episodes, split_episodes
from .training import evaluate_flow_loss


class StepBatchSampler:
    """Stateless global sequence sampling, so resume needs only the optimizer step."""

    def __init__(self, args, sequences, rank, world, completed_steps):
        self.args = args
        self.sequences = sequences
        self.rank = rank
        self.per_rank = args.global_batch // world
        self.completed_steps = completed_steps

    def __iter__(self):
        args = self.args
        for step in range(self.completed_steps, args.steps):
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, step]))
            chosen = rng.choice(self.sequences, args.global_batch, replace=self.sequences < args.global_batch)
            seeds = rng.integers(0, 2**32, size=args.global_batch)
            first = self.rank * self.per_rank
            fraction = min(1.0, step / max(args.context_warmup, 1)) if args.context_warmup else 1.0
            context = round(args.context_start + fraction * (args.context - args.context_start))
            for begin in range(first, first + self.per_rank, args.micro_batch):
                stop = begin + args.micro_batch
                yield [(int(index), int(crop_seed), context)
                       for index, crop_seed in zip(chosen[begin:stop], seeds[begin:stop], strict=True)]


def _emit(**record):
    print(json.dumps(record, ensure_ascii=False), flush=True)


def _sync():
    if dist.is_initialized():
        dist.barrier()


def _local_required_gpu_worker(world, local_rank):
    expected_world = int(os.getenv("ROBOTTT_DLC_GPUS", "4"))
    local_size = int(os.getenv("LOCAL_WORLD_SIZE", str(world)))
    names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    models = [re.search(r"\bA(?:100|800)\b", name) for name in names]
    if (expected_world not in (4, 8) or world != expected_world
            or local_size not in (world, world // 2) or not 0 <= local_rank < local_size
            or len(names) != local_size or not all(models)
            or any("MIG" in name.upper() for name in names)
            or len({model.group() for model in models}) != 1):
        raise RuntimeError(
            f"Expected {expected_world} full A100 or A800 GPUs on one or two equal-size nodes. "
            f"Got {names}, world={world}, local_size={local_size}, local_rank={local_rank}"
        )
    return (os.getenv("GROUP_RANK") or gethostname(), local_size, local_rank, models[0].group())


def _verify_required_gpu_workers(worker, world):
    workers = [None] * world
    dist.all_gather_object(workers, worker)
    _, local_size, _, model = worker
    nodes = {}
    for node, _, local_rank, _ in workers:
        nodes.setdefault(node, []).append(local_rank)
    if ({(size, family) for _, size, _, family in workers} != {(local_size, model)}
            or any(sorted(ranks) != list(range(local_size)) for ranks in nodes.values())):
        raise RuntimeError(f"All ranks must use the same A100 or A800 family and fill each node's local slots: {workers}")


def _device(args, world):
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    worker = None
    if args.device == "cpu" or (args.device == "auto" and not torch.cuda.is_available()):
        device = torch.device("cpu")
        if args.require_a100_a800:
            raise RuntimeError("This training run requires NVIDIA A100 or A800 GPUs")
    else:
        device = torch.device("cuda", local_rank)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested, but PyTorch reports no CUDA")
        if args.require_a100_a800:
            worker = _local_required_gpu_worker(world, local_rank)
        torch.cuda.set_device(device)
        if worker is not None and not torch.cuda.is_bf16_supported(including_emulation=False):
            raise RuntimeError(f"{worker[-1]} at local rank {local_rank} must support native BF16")
        torch.set_float32_matmul_precision("high")
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo", device_id=device if device.type == "cuda" else None)
    if worker is not None:
        try:
            _verify_required_gpu_workers(worker, world)
        except Exception:
            dist.destroy_process_group()
            raise
    return device


def _data_ready(args, episodes, rank, world):
    args.cache.mkdir(parents=True, exist_ok=True)
    mine = episodes[rank::world]
    start = perf_counter()
    with ThreadPoolExecutor(max_workers=args.cache_workers) as workers:
        for number, _ in enumerate(workers.map(lambda episode: cache_episode(args.cache, episode, args.grid), mine), 1):
            if number == 1 or number % 25 == 0 or number == len(mine):
                _emit(event="rgb_cache", rank=rank, episodes=number, local_total=len(mine),
                      elapsed_s=round(perf_counter() - start, 1))
    _sync()


def _normalized_loss(velocity, target, mask, weights):
    # Average over action channels, horizon, and supervised timesteps within
    # each trajectory. Divide across trajectories in the optimizer step.
    error = (velocity.float() - target.float()).square().mean(dim=(-1, -2))
    return ((error * mask).sum(dim=-1) * weights).sum()


def _step(model, optimizer, source, accumulation, device, dtype, segment_size, noise):
    optimizer.zero_grad(set_to_none=True)
    detached_loss = torch.zeros((), device=device)
    for index in range(accumulation):
        batch = next(source)
        valid = batch["valid_mask"]
        mask = batch["loss_mask"] & valid
        weights = mask.sum(dim=1).clamp_min(1).reciprocal().to(device)
        normalizer = weights / (len(weights) * accumulation)
        blocks = valid.shape[1]
        carry = None
        for start in range(0, blocks, segment_size):
            end = min(start + segment_size, blocks)
            section = slice(start, end)
            sync_last = index + 1 == accumulation and end == blocks
            sync = nullcontext() if sync_last or not isinstance(model, DistributedDataParallel) else model.no_sync()
            actions = batch["actions"][:, section].to(device)
            tau, noised, target = corrupt_actions(actions, generator=noise)
            with sync:
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
                    velocity, carry = model(
                        batch["obs"][:, section].to(device), batch["state"][:, section].to(device),
                        noised, tau, state=carry, valid=valid[:, section].to(device),
                    )
                    loss = _normalized_loss(velocity, target, mask[:, section].to(device), normalizer)
                loss.backward()
            detached_loss += loss.detach()
            carry = carry.detach()
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return detached_loss, grad.detach()


def _lr(args, step):
    warmup = min(args.warmup, max(0, args.steps - 1))
    if step <= warmup:
        return args.lr * step / warmup
    progress = (step - warmup) / (args.steps - warmup)
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + cos(pi * progress)))


def _argument_values(args):
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def _save(path, payload):
    temporary = path.with_suffix(f".tmp.{os.getpid()}")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_resume(checkpoint, args, config, data_manifest, world):
    immutable = ("seed", "global_batch", "micro_batch", "context_start", "context", "context_warmup",
                 "segment", "horizon", "grid", "val_per_task", "lr", "warmup", "steps")
    if checkpoint["config"] != asdict(config) or checkpoint["data_sha256"] != data_manifest["sha256"]:
        raise RuntimeError("Cannot resume from a different model or dataset")
    if checkpoint["world_size"] != world or any(checkpoint["train_args"][key] != getattr(args, key) for key in immutable):
        raise RuntimeError("Resume must preserve the original optimizer schedule and data-parallel configuration")


def _train(args, rank, world, device, episodes, train, validation):
    checkpoint_path = args.output / "latest.pt"
    if checkpoint_path.is_file() and args.resume is None:
        raise FileExistsError(f"{checkpoint_path} already exists; pass --resume {checkpoint_path}")
    if args.resume is not None and not args.resume.is_file():
        raise FileNotFoundError(args.resume)
    args.output.mkdir(parents=True, exist_ok=True)
    data_manifest = manifest(episodes, train, validation)
    values = read_state_actions(episodes)
    transform = preprocessing(train, values, args.grid)
    dataset = CachedEpisodes(train, values, args.cache, horizon=args.horizon, transform=transform)
    spec = dataset.spec
    config = PolicyConfig(
        spec.obs_dim, spec.state_dim, spec.action_dim, horizon=args.horizon,
        width=args.width, layers=args.layers, heads=args.heads, use_ttt=not args.no_ttt,
        stabilize_fast_weights=not args.no_ttt,
    )
    if rank == 0:
        _emit(event="configuration", world_size=world, device=str(device), data_sha256=data_manifest["sha256"],
              training_episodes=len(train), validation_episodes=len(validation), config=asdict(config),
              global_batch=args.global_batch, steps=args.steps)
        temporary = args.output / "dataset_manifest.tmp"
        temporary.write_text(json.dumps(data_manifest, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.output / "dataset_manifest.json")

    torch.manual_seed(args.seed)
    base_model = RoboTTT(config).to(device)
    optimizer = torch.optim.AdamW(base_model.parameters(), lr=args.lr, weight_decay=1e-5)
    completed_steps = 0
    if args.resume is not None:
        saved = torch.load(args.resume, map_location=device, weights_only=True)
        _validate_resume(saved, args, config, data_manifest, world)
        base_model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        completed_steps = saved["step"]
        del saved

    model = base_model
    if world > 1:
        model = DistributedDataParallel(
            base_model, device_ids=[device.index] if device.type == "cuda" else None,
            broadcast_buffers=False, find_unused_parameters=config.use_ttt,
        )
    dtype = torch.bfloat16 if device.type == "cuda" and not args.fp32 else None
    if dtype is not None and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is unavailable on this GPU; select --fp32")

    source = iter(DataLoader(
        dataset, batch_sampler=StepBatchSampler(args, len(dataset), rank, world, completed_steps),
        collate_fn=collate_episodes, num_workers=args.workers,
    ))
    if rank == 0 and completed_steps == args.steps:
        _emit(event="complete", step=completed_steps, resumed=True)
    if validation and rank == 0:
        validation_dataset = CachedEpisodes(validation, values, args.cache, horizon=args.horizon, transform=transform)
        validation_loader = DataLoader(
            validation_dataset, sampler=[(index, args.seed, args.context) for index in range(len(validation))],
            batch_size=args.micro_batch, collate_fn=collate_episodes,
        )
    accumulated = args.global_batch // (world * args.micro_batch)
    run_start = perf_counter()
    noise = torch.Generator(device=device)
    for step in range(completed_steps + 1, args.steps + 1):
        for group in optimizer.param_groups:
            group["lr"] = _lr(args, step)
        # Per-step stream is independent of work performed before a restart.
        noise.manual_seed(args.seed + step * world + rank)
        try:
            loss, grad = _step(model, optimizer, source, accumulated, device, dtype, args.segment, noise)
        except RuntimeError:
            if rank == 0:
                nonfinite = [name for name, parameter in base_model.named_parameters()
                             if parameter.grad is not None and not torch.isfinite(parameter.grad).all()]
                _emit(event="optimizer_error", step=step, nonfinite_gradient_parameters=nonfinite[:16])
            raise
        if step == completed_steps + 1 or step % args.log_every == 0 or step == args.steps:
            if world > 1:
                dist.reduce(loss, dst=0)
            if rank == 0:
                elapsed = perf_counter() - run_start
                rate = (step - completed_steps) / elapsed
                _emit(event="train", step=step, steps=args.steps, flow_mse=loss.item() / world,
                      grad_norm=grad.item(), learning_rate=optimizer.param_groups[0]["lr"],
                      steps_per_second=round(rate, 3), sequences_per_second=round(rate * args.global_batch, 2),
                      elapsed_s=round(elapsed, 1), eta_s=round((args.steps - step) / rate, 1))

        if step % args.save_every == 0 or step == args.steps:
            if rank == 0:
                if validation:
                    base_model.eval()
                    metric = evaluate_flow_loss(
                        base_model, validation_loader, device, segment_length=args.segment, seed=args.seed + 10000,
                    )
                    base_model.train()
                    _emit(event="validation", step=step, flow_mse=metric)
                payload = {
                    "model": base_model.state_dict(), "config": asdict(config), "data_spec": asdict(spec),
                    "preprocessing": transform, "dataset": "npz", "input_format": "insert_benchmark_hdf5",
                    "step": step, "optimizer": optimizer.state_dict(), "world_size": world,
                    "data_sha256": data_manifest["sha256"], "train_args": _argument_values(args),
                }
                _save(checkpoint_path, payload)
                if step % args.keep_every == 0 or step == args.steps:
                    _save(args.output / f"step_{step:06d}.pt", payload)
                _emit(event="checkpoint", step=step, checkpoint=str(checkpoint_path))
            _sync()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Root containing the six task trees")
    parser.add_argument("--output", type=Path, help="Checkpoint directory on persistent storage")
    parser.add_argument("--cache", type=Path, default=Path("/tmp/robottt-rgb-cache"))
    parser.add_argument("--check-data", action="store_true", help="Check dataset completeness and exit")
    parser.add_argument("--prepare-data-only", action="store_true", help="Build the local deterministic RGB cache and exit")
    parser.add_argument("--expected-per-task", type=int, default=50, help="Use 0 only for partial dataset smoke tests")
    parser.add_argument("--val-per-task", type=int, default=0, help="Optional trajectory holdout (0 trains on all demos)")
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--global-batch", type=int, default=64, help="Number of trajectory sequences per optimizer step")
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--context", type=int, default=192, help="Maximum policy calls, not individual robot actions")
    parser.add_argument("--context-start", type=int, default=32)
    parser.add_argument("--context-warmup", type=int, default=10_000)
    parser.add_argument("--segment", type=int, default=16)
    parser.add_argument("--grid", type=int, default=16)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cache-workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--keep-every", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--no-ttt", action="store_true")
    parser.add_argument("--require-a100-a800", action="store_true")
    args = parser.parse_args()

    positive = (args.steps, args.global_batch, args.micro_batch, args.horizon, args.context, args.context_start,
                args.segment, args.grid, args.width, args.layers, args.heads, args.lr, args.threads,
                args.cache_workers, args.save_every, args.keep_every, args.log_every)
    nonnegative = (args.context_warmup, args.workers, args.val_per_task, args.expected_per_task, args.warmup, args.seed)
    if min(positive) <= 0 or min(nonnegative) < 0 or args.context_start > args.context:
        parser.error("Training sizes must be positive with 0 < context-start <= context")
    if args.keep_every % args.save_every:
        parser.error("--keep-every must be a multiple of --save-every")
    if not (args.check_data or args.prepare_data_only or args.output):
        parser.error("--output is required for training")

    episodes = scan_episodes(args.data, args.horizon, args.grid, args.expected_per_task)
    train, validation = split_episodes(episodes, args.val_per_task, args.seed)
    rank, world = int(os.getenv("RANK", "0")), int(os.getenv("WORLD_SIZE", "1"))
    if rank == 0:
        _emit(event="dataset_preflight", episodes=len(episodes), counts=dict(Counter(item.level for item in episodes)),
              train=len(train), validation=len(validation))
    if args.check_data:
        return
    if args.global_batch % (world * args.micro_batch):
        parser.error("--global-batch must be divisible by world size × --micro-batch")
    if args.workers and args.device != "cpu":
        parser.error("Use --workers 0 with CUDA to avoid process-fork CUDA initialization")
    torch.set_num_threads(args.threads)
    device = _device(args, world)
    try:
        _data_ready(args, episodes, rank, world)
        if not args.prepare_data_only:
            _train(args, rank, world, device, episodes, train, validation)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
