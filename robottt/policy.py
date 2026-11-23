"""Compact RoboTTT DiT and sequence flow matching (paper §§3.1–3.3)."""

from dataclasses import dataclass
from math import log

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class PolicyConfig:
    context_dim: int
    state_dim: int
    action_dim: int
    horizon: int = 4
    width: int = 64
    layers: int = 2
    heads: int = 4
    registers: int = 16
    gate_init: float = 0.001
    use_ttt: bool = True
    stabilize_fast_weights: bool = False


@dataclass(frozen=True)
class RolloutState:
    fast: tuple
    index: int

    def detach(self):
        return RolloutState(tuple(state.detach() if state is not None else None for state in self.fast), self.index)


class StepBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.width
        self.self_norm = nn.LayerNorm(width)
        self.cross_norm = nn.LayerNorm(width)
        self.ff_norm = nn.LayerNorm(width)
        self.self_attention = nn.MultiheadAttention(width, config.heads, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(width, config.heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))
        self.ttt = None
        if config.use_ttt:
            from .ttt import TTTMLP

            self.ttt = TTTMLP(width, heads=config.heads, stabilize=config.stabilize_fast_weights)
            self.gate = nn.Parameter(torch.full((width,), config.gate_init))

    def forward(self, tokens, context, valid, fast, index):
        batch, steps, count, width = tokens.shape
        flattened = tokens.flatten(0, 1)
        context = context.flatten(0, 1)
        x = self.self_norm(flattened)
        x = flattened + self.self_attention(x, x, x, need_weights=False)[0]
        x = x + self.cross_attention(self.cross_norm(x), context, context, need_weights=False)[0]
        x = x.reshape(batch, steps, count, width)
        if self.ttt is not None:
            memory, fast = self.ttt(x, fast, valid=valid, start_step=index)
            x = x + self.gate.tanh() * memory
        x = x + self.ff(self.ff_norm(x))
        return x, fast


class RoboTTT(nn.Module):
    """Reference action head. Context features stand in for the paper's VLM tokens."""

    def __init__(self, config: PolicyConfig):
        super().__init__()
        if min(config.width, config.layers, config.heads, config.horizon, config.registers) < 1:
            raise ValueError("width, layers, heads, horizon and registers must be positive")
        if config.width % (2 * config.heads):
            raise ValueError("width must be divisible by twice the head count for RoPE")
        self.config = config
        width = config.width
        self.context_encoder = nn.Sequential(nn.Linear(config.context_dim, width), nn.LayerNorm(width))
        self.state_encoder = nn.Linear(config.state_dim, width)
        self.action_encoder = nn.Linear(config.action_dim, width)
        self.action_position = nn.Parameter(torch.randn(config.horizon, width) * 0.02)
        self.registers = nn.Parameter(torch.randn(config.registers, width) * 0.02)
        self.register_buffer(
            "time_frequencies",
            torch.exp(torch.linspace(0, -log(10_000), width // 2)),
            persistent=False,
        )
        self.time_encoder = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList(StepBlock(config) for _ in range(config.layers))
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, config.action_dim))

    def forward(
        self,
        context: Tensor,
        proprio: Tensor,
        noised_action: Tensor,
        tau: Tensor,
        state: RolloutState | None = None,
        valid: Tensor | None = None,
    ):
        """Consume [B,T,V,C] context, [B,T,P] states and [B,T,H,A] action chunks."""
        batch, steps = proprio.shape[:2]
        if state is None:
            state = RolloutState((None,) * len(self.blocks), 0)
        if valid is None:
            valid = torch.ones((batch, steps), device=proprio.device, dtype=torch.bool)
        context = self.context_encoder(context)
        registers = self.registers.expand(batch, steps, -1, -1)
        proprio_token = self.state_encoder(proprio).unsqueeze(2)
        action_tokens = self.action_encoder(noised_action) + self.action_position
        phase = tau.unsqueeze(-1) * (1_000 * self.time_frequencies)
        time = self.time_encoder(torch.cat((phase.sin(), phase.cos()), dim=-1))
        tokens = torch.cat((registers, proprio_token, action_tokens), dim=2) + time.unsqueeze(2)
        fast_states = []
        for block, fast in zip(self.blocks, state.fast, strict=True):
            tokens, fast = block(tokens, context, valid, fast, state.index)
            if block.ttt is not None:
                fast_states.append(fast)
        velocity = self.output(tokens[:, :, -self.config.horizon :])
        if not self.config.use_ttt:
            fast_states = [None] * len(self.blocks)
        return velocity, RolloutState(tuple(fast_states), state.index + steps)

    @torch.no_grad()
    def sample(self, context, proprio, state=None, *, denoise_steps=4, generator=None):
        """Euler flow solve. Every flow substep starts from the same incoming fast weights."""
        if denoise_steps < 1:
            raise ValueError("denoise_steps must be positive")
        action = torch.randn(
            (proprio.shape[0], 1, self.config.horizon, self.config.action_dim),
            dtype=proprio.dtype, device=proprio.device, generator=generator,
        )
        for i in range(denoise_steps):
            tau = proprio.new_full((proprio.shape[0], 1), i / denoise_steps)
            velocity, next_state = self(context[:, None], proprio[:, None], action, tau, state)
            action = action + velocity / denoise_steps
        return action[:, 0], next_state

    @torch.no_grad()
    def observe(self, context, proprio, state=None, *, action=None, generator=None):
        """Consume a context observation or a known executed action without emitting a command."""
        if action is None:
            action = proprio.new_zeros(proprio.shape[0], self.config.horizon, self.config.action_dim)
        tau, noisy, _ = corrupt_actions(action[:, None], generator=generator)
        return self(context[:, None], proprio[:, None], noisy, tau, state)[1]


def flow_time(shape, device, generator=None):
    """Sample Eq. 5 via the inverse CDF of Beta(1.5, 1)."""
    u = torch.rand(shape, device=device, generator=generator)
    return 0.999 * (1 - u.pow(2 / 3))


def corrupt_actions(actions, *, generator=None, tau=None):
    """One independent flow level per chunk unless an explicit level is supplied."""
    batch, steps = actions.shape[:2]
    if tau is None:
        tau = flow_time((batch, steps), actions.device, generator)
    tau = tau.to(actions.dtype).expand(batch, steps)
    noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype, generator=generator)
    expanded = tau[:, :, None, None]
    return tau, expanded * actions + (1 - expanded) * noise, actions - noise


def flow_loss_sum(velocity, target, action_mask):
    """Mean over action channels and sum over supervised action tokens."""
    error = (velocity.float() - target.float()).square().mean(dim=-1)
    return (error * action_mask).sum()
