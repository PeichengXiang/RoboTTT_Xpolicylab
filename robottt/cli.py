"""Local training, closed-loop evaluation, and long-context checks."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import torch
from torch.utils.data import DataLoader

from .policy import PolicyConfig, RoboTTT
from .training import evaluate_flow_loss, train_batch


def _device(args):
    torch.set_num_threads(args.threads)
    return torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)


def _load_model(path, device):
    saved = torch.load(path, map_location=device, weights_only=True)
    policy = RoboTTT(PolicyConfig(**saved["config"])).to(device)
    policy.load_state_dict(saved["model"])
    return policy.eval(), saved.get("dataset", "synthetic")


def _train(args):
    from .data import NPZEpisodes, SyntheticEpisodes, collate_episodes

    device = _device(args)
    torch.manual_seed(args.seed)
    if args.data:
        dataset = NPZEpisodes(args.data, args.horizon, args.context)
    else:
        dataset = SyntheticEpisodes(args.episodes, args.horizon, seed=args.seed)
    if args.data and args.validation_data is None:
        raise SystemExit("--validation-data must point to a separate NPZ directory")
    if args.data and args.data.resolve() == args.validation_data.resolve():
        raise SystemExit("Use separate training and validation directories")
    validation = (
        NPZEpisodes(args.validation_data, args.horizon, args.context, seed=args.seed + 10_000)
        if args.data else SyntheticEpisodes(args.validation_episodes, args.horizon, seed=args.seed + 10_000)
    )
    if dataset.spec != validation.spec:
        raise SystemExit("Training and validation data specs differ")
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_episodes,
        generator=torch.Generator().manual_seed(args.seed),
    )
    valid_loader = DataLoader(validation, batch_size=args.batch_size, collate_fn=collate_episodes)
    spec = dataset.spec
    config = PolicyConfig(
        context_dim=spec.obs_dim, state_dim=spec.state_dim, action_dim=spec.action_dim,
        horizon=spec.chunk_length, width=args.width, layers=args.layers, heads=args.heads,
        registers=args.registers, gate_init=args.gate_init, use_ttt=not args.no_ttt,
    )
    policy = RoboTTT(config).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-5)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    train_log = []
    for epoch in range(1, args.epochs + 1):
        policy.train()
        loss_sum = 0.0
        count_sum = 0
        max_grad = 0.0
        for batch in loader:
            loss, norm = train_batch(policy, optimizer, batch, args.segment, shared_time=args.shared_time)
            count = int((batch["loss_mask"] & batch["valid_mask"]).sum())
            max_grad = max(max_grad, norm)
            loss_sum += loss * count
            count_sum += count
        policy.eval()
        validation_loss = evaluate_flow_loss(
            policy, valid_loader, device, segment_length=args.segment, seed=args.seed + 20_000,
        )
        metrics = {
            "epoch": epoch, "train_flow_mse": loss_sum / count_sum,
            "validation_flow_mse": validation_loss, "max_grad_norm": max_grad,
            "elapsed_s": round(perf_counter() - start, 2),
        }
        print(json.dumps(metrics), flush=True)
        train_log.append(metrics)
        checkpoint = {
            "config": asdict(config), "data_spec": asdict(spec), "model": policy.state_dict(),
            "dataset": "npz" if args.data else "synthetic", "metrics": train_log,
            "seed": args.seed, "train_args": {k: str(v) if isinstance(v, Path) else v
                                            for k, v in vars(args).items() if k != "handler"},
        }
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(args.output)


def _eval(args):
    from .data import NPZEpisodes, collate_episodes
    from .benchmark import evaluate_context_reach

    device = _device(args)
    policy, dataset = _load_model(args.checkpoint, device)
    if args.data:
        episodes = NPZEpisodes(args.data, policy.config.horizon, args.context, seed=args.seed)
        loader = DataLoader(episodes, batch_size=args.batch_size, collate_fn=collate_episodes)
        results = {
            "episodes": len(episodes),
            "flow_mse": evaluate_flow_loss(
                policy, loader, device, segment_length=args.segment, seed=args.seed,
            ),
        }
    elif dataset == "synthetic":
        def begin(prefix_obs, prefix_state):
            memory = None
            generator = torch.Generator(device=device).manual_seed(args.seed)
            obs = torch.as_tensor(prefix_obs, device=device)
            proprio = torch.as_tensor(prefix_state, device=device)
            for step_obs, step_proprio in zip(obs, proprio, strict=True):
                memory = policy.observe(step_obs[None], step_proprio[None], memory, generator=generator)
            return memory

        def act(obs, state, memory, seed, denoise_steps):
            generator = torch.Generator(device=device).manual_seed(seed)
            action, memory = policy.sample(
                torch.as_tensor(obs[None], device=device), torch.as_tensor(state[None], device=device),
                None if args.reset_memory else memory, denoise_steps=denoise_steps, generator=generator,
            )
            return action[0].cpu().numpy(), memory

        results = evaluate_context_reach(
            begin, act, episodes=args.episodes, chunk_length=policy.config.horizon, seed=args.seed,
            denoise_steps=args.denoise_steps,
        )
    else:
        raise SystemExit("For an NPZ checkpoint, use --data with held-out NPZ episodes")
    document = json.dumps(results, ensure_ascii=False, indent=2)
    print(document)
    if args.metrics:
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(document + "\n", encoding="utf-8")


@torch.no_grad()
def _stress(args):
    device = _device(args)
    torch.manual_seed(args.seed)
    config = PolicyConfig(8, 2, 2, horizon=2, width=32, layers=1, heads=4)
    policy = RoboTTT(config).to(device).eval()
    obs = torch.randn(1, 2, 8, device=device)
    proprio = torch.randn(1, 2, device=device)
    incoming = None

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def latency(state):
        for _ in range(8):
            policy.sample(obs, proprio, state, denoise_steps=args.denoise_steps)
        synchronize()
        start = perf_counter()
        for _ in range(32):
            _, result = policy.sample(obs, proprio, state, denoise_steps=args.denoise_steps)
        synchronize()
        return 1000 * (perf_counter() - start) / 32, result

    short_ms, first = latency(incoming)
    start = perf_counter()
    for step in range(0, args.steps, args.segment):
        size = min(args.segment, args.steps - step)
        context = torch.randn(1, size, 2, 8, device=device)
        state = torch.randn(1, size, 2, device=device)
        action = torch.randn(1, size, 2, 2, device=device)
        tau = torch.rand(1, size, device=device)
        velocity, incoming = policy(context, state, action, tau, incoming)
        if not torch.isfinite(velocity).all():
            raise RuntimeError(f"Nonfinite output at context index {incoming.index}")
    synchronize()
    elapsed = perf_counter() - start
    long_ms, final = latency(incoming)
    initial_bytes = _state_bytes(first)
    final_bytes = _state_bytes(final)
    if final.index != args.steps + 1 or initial_bytes != final_bytes:
        raise AssertionError("Streaming state length or storage changed unexpectedly")
    print(json.dumps({
        "context_chunks": incoming.index, "actions_per_chunk": config.horizon,
        "state_bytes_early": initial_bytes, "state_bytes_late": final_bytes,
        "sample_ms_early": short_ms, "sample_ms_late": long_ms,
        "segmented_prefill_s": elapsed, "denoise_steps": args.denoise_steps, "device": str(device),
    }, indent=2))


def _stress_training(args):
    device = _device(args)
    torch.manual_seed(args.seed)
    config = PolicyConfig(8, 2, 2, horizon=2, width=32, layers=1, heads=4)
    policy = RoboTTT(config).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)

    def sample_batch(steps):
        return {
            "obs": torch.randn(1, steps, 2, 8),
            "state": torch.randn(1, steps, 2),
            "actions": torch.randn(1, steps, 2, 2),
            "loss_mask": torch.ones(1, steps, dtype=torch.bool),
            "valid_mask": torch.ones(1, steps, dtype=torch.bool),
        }

    train_batch(policy, optimizer, sample_batch(args.segment), args.segment)
    results = []
    for steps in dict.fromkeys((min(args.steps, args.segment * 8), args.steps)):
        batch = sample_batch(steps)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = perf_counter()
        loss, grad = train_batch(policy, optimizer, batch, args.segment)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        results.append({
            "context_chunks": steps, "segment": args.segment, "flow_mse": loss,
            "grad_norm": grad, "elapsed_s": perf_counter() - start,
            "peak_cuda_allocated_mib": (
                torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
            ),
        })
    print(json.dumps(results, indent=2))


def _state_bytes(state):
    return sum(tensor.numel() * tensor.element_size() for layer in state.fast for tensor in vars(layer).values()
               if isinstance(tensor, torch.Tensor))


def main():
    parser = argparse.ArgumentParser(prog="python -m robottt", description=__doc__)
    commands = parser.add_subparsers(required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--device", default="auto")
    common.add_argument("--threads", type=int, default=2)
    train = commands.add_parser("train", parents=[common])
    train.set_defaults(handler=_train)
    train.add_argument("--data", type=Path)
    train.add_argument("--validation-data", type=Path)
    train.add_argument("--output", type=Path, default=Path("runs/robottt.pt"))
    train.add_argument("--seed", type=int, default=17)
    train.add_argument("--epochs", type=int, default=15)
    train.add_argument("--episodes", type=int, default=256)
    train.add_argument("--validation-episodes", type=int, default=48)
    train.add_argument("--batch-size", type=int, default=24)
    train.add_argument("--context", type=int, default=64)
    train.add_argument("--segment", type=int, default=16)
    train.add_argument("--horizon", type=int, default=2)
    train.add_argument("--width", type=int, default=64)
    train.add_argument("--layers", type=int, default=2)
    train.add_argument("--heads", type=int, default=4)
    train.add_argument("--registers", type=int, default=16)
    train.add_argument("--gate-init", type=float, default=0.001)
    train.add_argument("--lr", type=float, default=0.002)
    train.add_argument("--no-ttt", action="store_true")
    train.add_argument("--shared-time", action="store_true")

    evaluate = commands.add_parser("eval", parents=[common])
    evaluate.set_defaults(handler=_eval)
    evaluate.add_argument("checkpoint", type=Path)
    evaluate.add_argument("--data", type=Path)
    evaluate.add_argument("--metrics", type=Path)
    evaluate.add_argument("--seed", type=int, default=30_000)
    evaluate.add_argument("--episodes", type=int, default=24)
    evaluate.add_argument("--context", type=int, default=64)
    evaluate.add_argument("--segment", type=int, default=16)
    evaluate.add_argument("--batch-size", type=int, default=8)
    evaluate.add_argument("--denoise-steps", type=int, default=4)
    evaluate.add_argument("--reset-memory", action="store_true")

    stress = commands.add_parser("stress", parents=[common])
    stress.set_defaults(handler=lambda args: _stress_training(args) if args.backward else _stress(args))
    stress.add_argument("--steps", type=int, default=8192)
    stress.add_argument("--backward", action="store_true", help="Measure CUDA peak training allocation")
    stress.add_argument("--segment", type=int, default=128)
    stress.add_argument("--seed", type=int, default=42)
    stress.add_argument("--denoise-steps", type=int, default=4)
    args = parser.parse_args()
    if any(getattr(args, key, 1) < 1 for key in ("epochs", "steps", "segment", "threads", "episodes")):
        parser.error("epochs, steps, segment, threads and episodes must be positive")
    args.handler(args)
