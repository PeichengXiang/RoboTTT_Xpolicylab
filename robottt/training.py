"""Sequence loss and bounded-activation-memory optimizer steps."""

import torch

from .policy import corrupt_actions, flow_loss_sum, flow_time


def train_batch(policy, optimizer, batch, segment_length, *, shared_time=False, max_grad_norm=1.0):
    """Carry weights across segments and update slow parameters once per sequence batch."""
    if segment_length < 1:
        raise ValueError("segment_length must be positive")
    context = batch["obs"]
    proprio = batch["state"]
    actions = batch["actions"]
    valid = batch["valid_mask"]
    supervised = batch["loss_mask"] & valid
    horizon = actions.shape[-2]
    supervised_count = int(supervised.sum()) * horizon
    if not supervised_count:
        raise ValueError("Sequence batch contains no supervised action chunks")

    device = next(policy.parameters()).device
    shared_tau = flow_time((actions.shape[0], 1), device) if shared_time else None
    optimizer.zero_grad(set_to_none=True)
    state = None
    total = torch.zeros((), device=device)
    for start in range(0, actions.shape[1], segment_length):
        segment = slice(start, start + segment_length)
        tau, noisy, target = corrupt_actions(actions[:, segment].to(device), tau=shared_tau)
        velocity, state = policy(
            context[:, segment].to(device), proprio[:, segment].to(device), noisy, tau,
            state=state, valid=valid[:, segment].to(device),
        )
        loss = flow_loss_sum(velocity, target, supervised[:, segment, None].to(device))
        (loss / supervised_count).backward()
        total = total + loss.detach()
        state = state.detach()

    norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    return float(total / supervised_count), float(norm)


@torch.no_grad()
def evaluate_flow_loss(policy, loader, device, *, segment_length, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    sum_squared = torch.zeros((), device=device)
    total_count = torch.zeros((), device=device)
    for batch in loader:
        actions = batch["actions"]
        supervised = batch["loss_mask"] & batch["valid_mask"]
        state = None
        for start in range(0, actions.shape[1], segment_length):
            segment = slice(start, start + segment_length)
            tau, noisy, target = corrupt_actions(actions[:, segment].to(device), generator=generator)
            velocity, state = policy(
                batch["obs"][:, segment].to(device), batch["state"][:, segment].to(device),
                noisy, tau, state=state, valid=batch["valid_mask"][:, segment].to(device),
            )
            sum_squared += flow_loss_sum(velocity, target, supervised[:, segment, None].to(device))
        total_count += supervised.sum().to(device) * actions.shape[-2]
    if not float(total_count):
        raise ValueError("Evaluation set contains no supervised action chunks")
    return float(sum_squared / total_count)
