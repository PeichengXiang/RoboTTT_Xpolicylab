"""Attach RoboTTT to NVIDIA's public Isaac GR00T N1.7 action head."""

from collections import defaultdict
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .policy import RolloutState, corrupt_actions
from .ttt import TTTMLP


class Gr00tRoboTTT(nn.Module):
    """Sequence wrapper for the public Gr00tN1d7ActionHead.

    Input context is the VLM's raw output before the head's VL projection. Inputs
    use the public model's normalized and padded state/action representation.
    The public N1.7 GA release uses Cosmos-Reason2-2B instead of the paper's Eagle.
    """

    def __init__(
        self, action_head: nn.Module, *, registers: int = 16, ttt_heads: int = 32,
        gate_init: float = 0.001, embodiment_id: int | None = None,
    ):
        super().__init__()
        if registers < 0 or ttt_heads < 1:
            raise ValueError("registers must be nonnegative and ttt_heads must be positive")
        self.head = action_head
        self.default_embodiment_id = embodiment_id
        width = action_head.model.inner_dim
        if width % (ttt_heads * 2):
            raise ValueError("DiT width must be divisible by twice the TTT head count")
        weight = next(action_head.parameters())
        self.registers = nn.Parameter(weight.new_empty((registers, action_head.input_embedding_dim)))
        nn.init.normal_(self.registers, std=0.02)
        self.memories = nn.ModuleList(
            TTTMLP(width, heads=ttt_heads).to(device=weight.device, dtype=weight.dtype)
            for _ in action_head.model.transformer_blocks
        )
        self.gates = nn.Parameter(weight.new_full((len(self.memories), width), gate_init))
        self.set_stage("pretrain")

    @classmethod
    def from_local_checkpoint(cls, checkpoint: str | Path, **kwargs):
        """Load action-head weights from an official local N1.7 safetensors snapshot.

        Install Isaac-GR00T first. No VLM is instantiated, so this can use previously
        extracted compatible embeddings without external model credentials.
        """
        from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
        from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
        from safetensors import safe_open

        checkpoint = Path(checkpoint)
        head = Gr00tN1d7ActionHead(Gr00tN1d7Config.from_pretrained(checkpoint, local_files_only=True))
        prefix = "action_head."
        weights = {}
        index = checkpoint / "model.safetensors.index.json"
        if index.is_file():
            shards = defaultdict(list)
            for name, shard in json.loads(index.read_text())["weight_map"].items():
                if name.startswith(prefix):
                    shards[shard].append(name)
            for shard, names in shards.items():
                with safe_open(checkpoint / shard, framework="pt", device="cpu") as handle:
                    weights.update({name.removeprefix(prefix): handle.get_tensor(name) for name in names})
        else:
            with safe_open(checkpoint / "model.safetensors", framework="pt", device="cpu") as handle:
                weights = {
                    name.removeprefix(prefix): handle.get_tensor(name)
                    for name in handle.keys() if name.startswith(prefix)
                }
        head.load_state_dict(weights, strict=True)
        return cls(head, **kwargs)

    def set_stage(self, stage: str):
        """Freeze the original head for pretraining or unfreeze it for head-only post-training."""
        if stage not in ("pretrain", "posttrain"):
            raise ValueError("stage must be pretrain or posttrain")
        self._freeze_base = stage == "pretrain"
        self.head.requires_grad_(not self._freeze_base)
        self.head.train(self.training and not self._freeze_base)
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        if self._freeze_base:
            self.head.eval()
        return self

    def forward(
        self, context: Tensor, proprio: Tensor, noised_action: Tensor, tau: Tensor,
        state: RolloutState | None = None, valid: Tensor | None = None, *,
        embodiment_id: Tensor | None = None, vl_mask: Tensor | None = None,
        image_mask: Tensor | None = None, action_mask: Tensor | None = None,
    ):
        """Return the flow velocity [B,T,H,A] and one fast-weight state per DiT block.

        Context has shape [B,T,V,D]. Proprio has shape [B,T,S,P] with S state
        observations, or [B,T,P] if S is one. Per-VL-token masks are [B,T,V].
        Embodiment IDs are [B] or [B,T]. A regular GR00T action padding mask
        has shape [B,T,H,A] (or last dimension 1). valid masks padded timesteps.
        """
        head, config = self.head, self.head.config
        model_dtype = next(head.parameters()).dtype
        context = context.to(dtype=model_dtype)
        proprio = proprio.to(dtype=model_dtype)
        noised_action = noised_action.to(dtype=model_dtype)
        batch, steps, vl_tokens, _ = context.shape
        horizon, action_width = noised_action.shape[-2:]
        if proprio.ndim == 3:
            proprio = proprio.unsqueeze(2)
        if proprio.shape != (batch, steps, config.state_history_length, config.max_state_dim):
            raise ValueError("proprio must use the public GR00T state history and padded width")
        if noised_action.shape != (batch, steps, head.action_horizon, head.action_dim):
            raise ValueError("noised_action must use the public GR00T horizon and padded action width")
        if tau.shape != (batch, steps):
            raise ValueError("tau must have shape [B,T]")
        if state is None:
            state = RolloutState((None,) * len(self.memories), 0)
        if valid is None:
            valid = torch.ones(batch, steps, device=context.device, dtype=torch.bool)
        elif valid.shape != (batch, steps):
            raise ValueError("valid must have shape [B,T]")
        else:
            valid = valid.to(device=context.device, dtype=torch.bool)

        if embodiment_id is None:
            if self.default_embodiment_id is None:
                raise ValueError("provide the public GR00T embodiment_id")
            embodiment_id = torch.full(
                (batch,), self.default_embodiment_id, dtype=torch.long, device=context.device,
            )
        if embodiment_id.shape == (batch,):
            embodiment_id = embodiment_id[:, None].expand(batch, steps)
        if embodiment_id.shape != (batch, steps):
            raise ValueError("embodiment_id must have shape [B] or [B,T]")
        ids = embodiment_id.reshape(-1).to(device=context.device, dtype=torch.long)

        vl_shape = (batch, steps, vl_tokens)
        if vl_mask is None:
            vl_mask = torch.ones(vl_shape, device=context.device, dtype=torch.bool)
        if vl_mask.shape != vl_shape:
            raise ValueError("vl_mask must have shape [B,T,V]")
        vl_mask = vl_mask.flatten(0, 1).to(device=context.device, dtype=torch.bool)
        if config.use_alternate_vl_dit:
            if image_mask is None or image_mask.shape != vl_shape:
                raise ValueError("public AlternateVLDiT requires image_mask with shape [B,T,V]")
            image_mask = image_mask.flatten(0, 1).to(device=context.device, dtype=torch.bool)
            image_attention_mask = image_mask & vl_mask
            text_attention_mask = ~image_mask & vl_mask

        if action_mask is not None:
            if action_mask.shape not in (
                (batch, steps, horizon, 1), (batch, steps, horizon, action_width),
            ):
                raise ValueError("action_mask must use the public GR00T padded action shape")
            action_mask = action_mask.flatten(0, 1).to(device=context.device, dtype=torch.bool)
        if config.strict_action_padding_mask and action_mask is None:
            raise ValueError("action_mask is required by this head's strict_action_padding_mask")

        strict_action_mask = action_mask if config.strict_action_padding_mask else None
        backbone = head.process_backbone_output(
            {"backbone_features": context.flatten(0, 1), "backbone_attention_mask": vl_mask}
        )["backbone_features"].contiguous()
        state_tokens = head.state_encoder(proprio.reshape(batch * steps, 1, -1), ids)
        if head.training and head.state_dropout_prob > 0:
            keep = torch.rand(batch * steps, device=state_tokens.device) >= head.state_dropout_prob
            state_tokens = state_tokens * keep[:, None, None]
        tau_ids = (tau.reshape(-1).float() * head.num_timestep_buckets).long()
        actions = noised_action.flatten(0, 1)
        if strict_action_mask is not None:
            actions = actions * strict_action_mask
        action_tokens = head.action_encoder(actions, tau_ids, ids)
        if config.add_pos_embed:
            positions = torch.arange(horizon, device=context.device)
            action_tokens = action_tokens + head.position_embedding(positions)
        registers = self.registers.expand(batch * steps, -1, -1).to(dtype=state_tokens.dtype)
        tokens = torch.cat((registers, state_tokens, action_tokens), dim=1)
        hidden_mask = None
        if strict_action_mask is not None:
            unpadded = torch.ones(registers.shape[:2], device=context.device, dtype=torch.bool)
            hidden_mask = torch.cat((unpadded, unpadded.new_ones((batch * steps, 1)),
                                     strict_action_mask.any(dim=-1)), dim=-1)
            tokens = tokens * hidden_mask.unsqueeze(-1)

        dit = head.model
        temb = dit.timestep_encoder(tau_ids)
        memory_valid = valid
        if hidden_mask is not None:
            memory_valid = hidden_mask.unflatten(0, (batch, steps)) & valid.unsqueeze(-1)
        fast_states = []
        for index, (block, memory, fast, gate) in enumerate(
            zip(dit.transformer_blocks, self.memories, state.fast, self.gates, strict=True)
        ):
            if index % 2 == 1 and dit.config.interleave_self_attention:
                source, mask = None, hidden_mask
            elif config.use_alternate_vl_dit:
                text_layer = index % (2 * dit.attend_text_every_n_blocks) == 0
                source = backbone
                mask = text_attention_mask if text_layer else image_attention_mask
            else:
                source = backbone
                mask = vl_mask if config.strict_vl_padding_mask else None

            after_attention = self._apply_attention(block, tokens, source, mask, temb)
            if hidden_mask is not None:
                after_attention = after_attention * hidden_mask.unsqueeze(-1)
            sequence = after_attention.unflatten(0, (batch, steps))
            recalled, updated = memory(sequence, fast, valid=memory_valid, start_step=state.index)
            fast_states.append(updated)
            tokens = after_attention + recalled.flatten(0, 1).to(after_attention.dtype) * gate.tanh()
            tokens = tokens + block.ff(block.norm3(tokens))
            if hidden_mask is not None:
                tokens = tokens * hidden_mask.unsqueeze(-1)

        shift, scale = dit.proj_out_1(F.silu(temb)).chunk(2, dim=-1)
        tokens = dit.norm_out(tokens) * (1 + scale[:, None]) + shift[:, None]
        velocity = head.action_decoder(dit.proj_out_2(tokens[:, -horizon:]), ids)
        if strict_action_mask is not None:
            velocity = velocity * strict_action_mask
        return velocity.unflatten(0, (batch, steps)), RolloutState(
            tuple(fast_states), state.index + steps,
        )

    @staticmethod
    def _apply_attention(block, tokens, source, mask, temb):
        # Run the public block's attention and FFN separately to insert TTT
        # before the FFN. Attention handling follows its BasicTransformerBlock.
        from gr00t.model.modules.dit import _sdpa_context

        normalized = (
            block.norm1(tokens, temb) if block.norm_type == "ada_norm" else block.norm1(tokens)
        )
        if block.pos_embed is not None:
            normalized = block.pos_embed(normalized)
        with _sdpa_context():
            attended = block.attn1(normalized, encoder_hidden_states=source, attention_mask=mask)
        if block.final_dropout is not None:
            attended = block.final_dropout(attended)
        return tokens + attended

    @torch.no_grad()
    def sample(
        self, context, proprio, state=None, *, denoise_steps=None, generator=None, **kwargs,
    ):
        """Generate one normalized action chunk and commit fast weights once."""
        steps = self.head.num_inference_timesteps if denoise_steps is None else denoise_steps
        if steps < 1:
            raise ValueError("denoise_steps must be positive")
        action = self._noise(context, generator)
        metadata = self._one_step_masks(kwargs)
        for index in range(steps):
            tau = torch.full((context.shape[0], 1), index / steps, device=context.device)
            velocity, next_state = self(context[:, None], proprio[:, None], action, tau, state,
                                        **metadata)
            action = action + velocity / steps
        if self.head.config.strict_action_padding_mask:
            action = action * metadata["action_mask"]
        return action[:, 0], next_state

    @torch.no_grad()
    def observe(self, context, proprio, state=None, *, action=None, generator=None, **kwargs):
        """Commit a context-only step with the training noise schedule.

        Use the observed action when available and a zero placeholder otherwise.
        """
        if action is None:
            action = context.new_zeros(context.shape[0], self.head.action_horizon, self.head.action_dim)
        action = action.to(dtype=next(self.head.parameters()).dtype)
        tau, noisy, _ = corrupt_actions(action[:, None], generator=generator)
        return self(
            context[:, None], proprio[:, None], noisy, tau, state, **self._one_step_masks(kwargs),
        )[1]

    def _noise(self, context, generator):
        return torch.randn(
            (context.shape[0], 1, self.head.action_horizon, self.head.action_dim),
            device=context.device, dtype=next(self.head.parameters()).dtype, generator=generator,
        )

    @staticmethod
    def _one_step_masks(kwargs):
        return {
            name: mask[:, None] if name in {"vl_mask", "image_mask", "action_mask"} else mask
            for name, mask in kwargs.items() if mask is not None
        }
