"""Paired BF16 timing and gradient comparison for GR00T RoboTTT pretraining."""

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
from statistics import median
from time import perf_counter

import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel

from robottt.ddp_train import StepBatchSampler, _device, _emit
from robottt.gr00t import Gr00tRoboTTT
from robottt.gr00t_cache import BASE_REVISION, HORIZON, cache_directory, cache_path
from robottt.gr00t_train import ADAPTER_CONFIG, train_step
from robottt.raw_data import manifest, scan_episodes


def reset_trial(parameters, initial, optimizer):
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(initial[name])
    # Preserve the production AdamW allocation and update kernels, and use the
    # same initial step and moments for both alternatives in every timed pair.
    for state in optimizer.state.values():
        for value in state.values():
            value.zero_()
    optimizer.zero_grad(set_to_none=True)


def timed_step(model, optimizer, selections, episodes, cache, args, rank, world, device, step, size):
    args.pretrain_micro_batch = size
    torch.cuda.synchronize(device)
    dist.barrier()
    torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()
    loss, grad, shortest, longest, count = train_step(
        model, optimizer, selections, episodes, cache, args, device, rank, step,
    )
    torch.cuda.synchronize(device)
    seconds = perf_counter() - started
    row = torch.tensor([loss.item(), grad.item(), shortest, longest, count, seconds,
                        torch.cuda.max_memory_allocated(device) / 1024**3], dtype=torch.float64, device=device)
    rows = [torch.empty_like(row) for _ in range(world)]
    dist.all_gather(rows, row)
    rows = torch.stack(rows).cpu()
    total = int(rows[:, 4].sum().item())
    elapsed = rows[:, 5].max().item()
    return {
        "micro_batch": size, "loss": rows[:, 0].mean().item(), "rank_loss": rows[:, 0].tolist(),
        "rank_grad_norm": rows[:, 1].tolist(), "context_min": int(rows[:, 2].min().item()),
        "context_max": int(rows[:, 3].max().item()), "contexts": total,
        "step_seconds": elapsed, "rank_seconds": rows[:, 5].tolist(), "blocks_per_second": total / elapsed,
        "peak_gpu_gib": rows[:, 6].max().item(),
    }


def gradient_difference(parameters, first, first_size):
    # DDP has already synchronized these optimizer gradients. Stream one tensor
    # at a time so the comparison does not reserve a second full GPU gradient.
    sums = torch.zeros(4, dtype=torch.float64, device=next(iter(parameters.values())).device)
    names = {name for name, p in parameters.items() if p.grad is not None}
    if names != set(first):
        raise RuntimeError(f"Gradient presence mismatch: {sorted(names.symmetric_difference(first))}")
    for name in sorted(names):
        other = first[name].to(sums.device)
        current = parameters[name].grad.detach()
        baseline, candidate = (other, current) if first_size == 1 else (current, other)
        sums[0] += baseline.norm().double().square()
        sums[1] += candidate.norm().double().square()
        sums[2] += (baseline - candidate).norm().double().square()
        sums[3] += (baseline * candidate).sum(dtype=torch.float64)
    n1, n4, diff, dot = sums.tolist()
    if min(n1, n4) <= 0:
        raise RuntimeError("Expected nonzero gradients from both optimizer steps")
    return {"relative_l2": (diff / n1)**0.5, "cosine": dot / (n1*n4)**0.5}


def relative_change(reference, value):
    return abs(reference - value) / max(abs(reference), 1e-12)


def run(args, rank, world, device):
    episodes = scan_episodes(args.data, HORIZON, 1, expected_per_task=50)
    identity = manifest(episodes, episodes, [])["sha256"]
    cache = cache_directory(args.cache, identity, args.stride)
    if not all(cache_path(cache, episode).is_file() for episode in episodes):
        raise FileNotFoundError(f"Expected a complete production feature cache in {cache}")
    torch.manual_seed(args.seed)
    policy = Gr00tRoboTTT.from_local_checkpoint(
        args.base, embodiment_id=10, device=device, dtype=torch.bfloat16,
        ttt_dtype=torch.float32, reference_style=True, **ADAPTER_CONFIG,
    )
    parameters = {name: p for name, p in policy.named_parameters() if p.requires_grad}
    if len(policy.memories) != 32 or any(p.requires_grad for p in policy.head.parameters()):
        raise RuntimeError("Pretraining benchmark requires 32 TTT blocks and a frozen native head")
    model = DistributedDataParallel(policy, device_ids=[device.index], broadcast_buffers=False,
                                    find_unused_parameters=True, gradient_as_bucket_view=True).train()
    optimizer = torch.optim.AdamW(parameters.values(), lr=2e-5, weight_decay=1e-5, foreach=False)
    initial = {name: p.detach().cpu() for name, p in parameters.items()}
    reports = []
    success = True
    for first_step in (2501, 10001):
        sampler = iter(StepBatchSampler(args, len(episodes), rank, world, first_step-1))
        ratios = []
        # First pair warms the allocator, AdamW, CUDA kernels and both data paths.
        # Alternate the order on the four timed pairs to balance NAS page-cache reuse.
        for trial in range(args.pairs + 1):
            step = first_step + trial
            selections = [next(sampler)[0] for _ in range(args.global_batch // world)]
            first_size, second_size = (1, 4) if trial % 2 else (4, 1)
            runs = {}
            for size in (first_size, second_size):
                reset_trial(parameters, initial, optimizer)
                observed = []
                # Warm-up only: check the native head actually received a batch.
                hook = (policy.register_forward_pre_hook(lambda _, inputs: observed.append(inputs[1].shape[0]))
                        if trial == 0 else nullcontext())
                with hook:
                    result = timed_step(model, optimizer, selections, episodes, cache, args, rank,
                                        world, device, step, size)
                if trial == 0:
                    physical = torch.tensor(max(observed), device=device)
                    dist.all_reduce(physical, op=dist.ReduceOp.MAX)
                    if (size == 1 and physical.item() != 1) or (size == 4 and not 1 < physical.item() <= 4):
                        raise RuntimeError(f"Unexpected physical batch {physical.item()} for microbatch {size}")
                    result["max_observed_trajectory_batch"] = int(physical.item())
                runs[size] = result
                if rank == 0:
                    _emit(event="benchmark_trial", warmup=trial == 0, sample_step=step, **result)
                    if size == first_size and trial:
                        first_grad = {name: p.grad.detach().cpu() for name, p in parameters.items()
                                      if p.grad is not None}
            if rank == 0 and trial:
                reference, candidate = runs[1], runs[4]
                match = gradient_difference(parameters, first_grad, first_size)
                del first_grad
                for field in ("rank_loss", "rank_grad_norm"):
                    match[f"max_relative_{field}"] = max(
                        relative_change(a, b) for a, b in zip(reference[field], candidate[field], strict=True)
                    )
                same_context = all(reference[k] == candidate[k] for k in ("contexts", "context_min", "context_max"))
                passed = (same_context and match["max_relative_rank_loss"] <= 0.002
                          and match["max_relative_rank_grad_norm"] <= 0.03
                          and match["relative_l2"] <= 0.05 and match["cosine"] >= 0.999)
                speedup = reference["step_seconds"] / candidate["step_seconds"]
                ratios.append(speedup)
                reports.append({"step": step, "context_cap": selections[0][2], "order": [first_size, second_size],
                                "baseline": reference, "batched": candidate, "gradient": match,
                                "speedup": speedup, "numerically_matched": passed})
                success = success and passed
                _emit(event="benchmark_pair", sample_step=step, context=selections[0][2],
                      speedup=speedup, numerically_matched=passed, **match)
        if rank == 0:
            _emit(event="benchmark_context", context=selections[0][2], median_paired_speedup=median(ratios))
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        report = {"base_revision": BASE_REVISION, "data_sha256": identity, "world_size": world,
                  "device": torch.cuda.get_device_name(device), "batch": args.global_batch,
                  "trainable_parameters": sum(p.numel() for p in parameters.values()), "seed": args.seed,
                  "segment": args.segment, "stride": args.stride, "pairs": reports,
                  "all_numerically_matched": success}
        temporary = args.output / f".benchmark.{os.getpid()}.tmp"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output / "benchmark.json")
    verdict = torch.tensor(int(success), device=device)
    dist.broadcast(verdict, src=0)
    if not verdict.item():
        raise RuntimeError("Paired numerical checks failed. See benchmark.json for per-pair errors.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "cache", "base", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=4)
    args = parser.parse_args()
    if args.pairs < 4 or args.pairs % 2:
        parser.error("--pairs requires an even count of four or more")
    args.stage = "pretrain"
    args.seed = 1745
    args.steps = 50000
    args.context_start, args.context, args.context_warmup = 7, 39, 10000
    args.global_batch, args.micro_batch = 64, 1
    args.stride, args.segment = 40, 8
    args.device, args.require_a100_a800 = "cuda", True
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    device = _device(args, world)
    try:
        run(args, rank, world, device)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
