"""Chunk-causal multihead fast MLP for RoboTTT (paper Eq. 1–2)."""

from dataclasses import dataclass, fields
from math import sqrt

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class FastWeights:
    w1: Tensor
    b1: Tensor
    w2: Tensor
    b2: Tensor

    def detach(self):
        return FastWeights(*(getattr(self, field.name).detach() for field in fields(self)))


class TTTMLP(nn.Module):
    """One MSE gradient update per timestep's token block, followed by its read.

    The paper specifies a two-layer GELU MLP and base rate 0.1. This implementation
    uses independent heads, 4x hidden expansion, one learned rate per head
    (2 × base × sigmoid), and RoPE on Q/K at absolute flattened token positions.
    """

    def __init__(self, width, heads=4, expansion=4, base_lr=0.1, rope_theta=10_000):
        super().__init__()
        if min(width, heads, expansion, base_lr, rope_theta) <= 0 or width % (heads * 2):
            raise ValueError("Expected positive dimensions, rates and even head width")
        self.heads = heads
        self.head_dim = width // heads
        self.base_lr = base_lr
        self.rope_theta = rope_theta
        head_width = self.head_dim
        hidden_width = head_width * expansion
        self.qkv = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width)
        self.w1 = nn.Parameter(torch.randn(heads, head_width, hidden_width) / sqrt(head_width))
        self.b1 = nn.Parameter(torch.zeros(heads, 1, hidden_width))
        self.w2 = nn.Parameter(torch.randn(heads, hidden_width, head_width) / sqrt(hidden_width))
        self.b2 = nn.Parameter(torch.zeros(heads, 1, head_width))
        self.lr = nn.Parameter(torch.zeros(heads))

    def initialize(self, batch_size):
        dtype = torch.float64 if self.w1.dtype == torch.float64 else torch.float32
        return FastWeights(*(
            parameter.to(dtype).expand(batch_size, *parameter.shape)
            for parameter in (self.w1, self.b1, self.w2, self.b2)
        ))

    def forward(self, tokens, state=None, *, update=True, valid=None, start_step=0):
        """Accept [B,T,L,D] or a single [B,L,D] chunk and optional validity mask."""
        single = tokens.ndim == 3
        if single:
            tokens = tokens.unsqueeze(1)
            if valid is not None:
                valid = valid.unsqueeze(1)
        batch, steps, count, width = tokens.shape
        if state is None:
            state = self.initialize(batch)
        if valid is None:
            valid = torch.ones(batch, steps, count, dtype=torch.bool, device=tokens.device)
        elif valid.ndim == 2:
            valid = valid.unsqueeze(-1).expand(-1, -1, count)
        projected = self.qkv(tokens.masked_fill(~valid.unsqueeze(-1), 0))
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            memory, state = self._recur(projected.to(state.w1.dtype), state, valid, update, start_step)
        restored = memory.transpose(2, 3).reshape(batch, steps, count, width)
        mapped = self.output(restored.to(self.output.weight.dtype))
        return (mapped[:, 0] if single else mapped), state

    def _recur(self, projected, state, valid, update, start_step):
        batch, steps, count, _ = projected.shape
        query, key, value = projected.reshape(
            batch, steps, count, 3, self.heads, self.head_dim
        ).permute(3, 1, 0, 4, 2, 5).unbind(0)
        rates = self.rope_theta ** (
            -torch.arange(0, self.head_dim, 2, device=projected.device, dtype=projected.dtype) / self.head_dim
        )
        positions = torch.arange(start_step * count, (start_step + steps) * count, device=projected.device)
        angles = (positions[:, None] * rates).unflatten(0, (steps, count))[:, None, None]
        cosine, sine = angles.cos().to(query.dtype), angles.sin().to(query.dtype)
        query = self._rotate(query, cosine, sine)
        key = self._rotate(key, cosine, sine)

        outputs = []
        rate = (2 * self.base_lr * self.lr.to(projected.dtype).sigmoid())[None, :, None, None]
        for query_t, key_t, value_t, mask in zip(query, key, value, valid.transpose(0, 1), strict=True):
            if update:
                state = self._update(state, key_t, value_t, mask, rate)
            outputs.append(F.gelu(query_t @ state.w1 + state.b1) @ state.w2 + state.b2)
        return torch.stack(outputs, dim=1), state

    @staticmethod
    def _rotate(value, cosine, sine):
        even, odd = value[..., ::2], value[..., 1::2]
        return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1).flatten(-2)

    def _update(self, state, key, value, mask, rate):
        # Differentiate the mean squared K→V error analytically. These operations
        # remain differentiable for outer-loop gradients and also run under no_grad.
        pre_activation = key @ state.w1 + state.b1
        hidden = F.gelu(pre_activation)
        valid_count = mask.sum(-1).clamp_min(1).to(value.dtype)[:, None, None, None]
        error = (hidden @ state.w2 + state.b2 - value) * (2 / (self.head_dim * valid_count))
        error = error * mask[:, None, :, None]
        gelu_slope = 0.5 * (1 + torch.erf(pre_activation / sqrt(2)))
        gelu_slope = gelu_slope + pre_activation * (-0.5 * pre_activation.square()).exp() / sqrt(2 * torch.pi)
        hidden_error = (error @ state.w2.transpose(-2, -1)) * gelu_slope
        return FastWeights(
            state.w1 - rate * (key.transpose(-2, -1) @ hidden_error),
            state.b1 - rate * hidden_error.sum(-2, keepdim=True),
            state.w2 - rate * (hidden.transpose(-2, -1) @ error),
            state.b2 - rate * error.sum(-2, keepdim=True),
        )
