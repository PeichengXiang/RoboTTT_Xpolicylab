"""Attach RoboTTT to NVIDIA's public Isaac GR00T N1.7 action head."""

from collections import defaultdict
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint as checkpoint_activations, set_checkpoint_early_stop

from .policy import RolloutState, corrupt_actions
from .ttt import TTTMLP


def _checkpoint_tensors(checkpoint: Path, prefix: str):
    from safetensors import safe_open

    index = checkpoint / "model.safetensors.index.json"
    if not index.is_file():
        with safe_open(checkpoint / "model.safetensors", framework="pt", device="cpu") as handle:
            return {
                name.removeprefix(prefix): handle.get_tensor(name)
                for name in handle.keys() if name.startswith(prefix)
            }

    shards = defaultdict(list)
    for name, shard in json.loads(index.read_text())["weight_map"].items():
        if name.startswith(prefix):
            shards[shard].append(name)
    weights = {}
    for shard, names in shards.items():
        with safe_open(checkpoint / shard, framework="pt", device="cpu") as handle:
            weights.update({name.removeprefix(prefix): handle.get_tensor(name) for name in names})
    return weights


def load_public_backbone(checkpoint: str | Path, processor_assets: str | Path, *,
                         device: str | torch.device = "cpu", dtype: torch.dtype | None = None,
                         trainable: bool = False):
    """Restore the native VLM from GR00T's own safetensors snapshot.

    Use the corresponding Cosmos-Reason2-2B config or a verified byte-identical
    public copy. The model parameters all come from the GR00T snapshot.
    """
    from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
    from gr00t.model.modules.qwen3_backbone import Qwen3Backbone
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration

    class SnapshotBackbone(Qwen3Backbone):
        def __init__(self, select_layer, vlm_config):
            nn.Module.__init__(self)
            self.model = Qwen3VLForConditionalGeneration._from_config(
                vlm_config, attn_implementation="sdpa",
            )
            self.select_layer = select_layer
            while len(self.model.language_model.layers) > select_layer:
                self.model.language_model.layers.pop(-1)
            self.set_trainable_parameters(trainable, trainable, 0)

        def posttrain_features(self, inputs):
            # Match the exact hidden state selected by NVIDIA's Qwen3Backbone.
            # A full sequence of vocabulary logits and an inference KV cache
            # are both unused by the action loss.
            return self.model(**inputs, output_hidden_states=True, use_cache=False,
                              logits_to_keep=1).hidden_states[-1]

    checkpoint = Path(checkpoint)
    config = Gr00tN1d7Config.from_pretrained(checkpoint, local_files_only=True)
    vlm_config = AutoConfig.from_pretrained(Path(processor_assets), local_files_only=True)
    with torch.device("meta"):
        backbone = SnapshotBackbone(config.select_layer, vlm_config).eval()
    backbone.load_state_dict(_checkpoint_tensors(checkpoint, "backbone."), strict=True, assign=True)
    # The analytic RoPE buffers are the only VLM state absent from safetensors.
    backbone._reset_rotary_inv_freq()
    backbone.to(device=device, dtype=dtype)
    backbone._reset_rotary_inv_freq()
    backbone._apply_vision_patch_embed_channels_last()
    return backbone.requires_grad_(trainable)


class Gr00tRoboTTT(nn.Module):
    """Sequence wrapper for the public Gr00tN1d7ActionHead.

    Input context is the VLM's raw output before the head's VL projection. Inputs
    use the public model's normalized and padded state/action representation.
    The public N1.7 GA release uses Cosmos-Reason2-2B instead of the paper's Eagle.
    """

    def __init__(
        self, action_head: nn.Module, *, registers: int = 16, ttt_heads: int = 32,
        gate_init: float = 0.001, embodiment_id: int | None = None,
        stabilize_fast_weights: bool = False,
        reference_style: bool = False,
        ttt_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if registers < 0 or ttt_heads < 1:
            raise ValueError("registers must be nonnegative and ttt_heads must be positive")
        self.head = action_head
        self.backbone = None
        self.checkpoint_posttrain = False
        self.default_embodiment_id = embodiment_id
        self.stabilize_fast_weights = stabilize_fast_weights
        width = action_head.model.inner_dim
        if width % (ttt_heads * 2):
            raise ValueError("DiT width must be divisible by twice the TTT head count")
        weight = next(action_head.parameters())
        adapter_dtype = weight.dtype if ttt_dtype is None else ttt_dtype
        self.registers = nn.Parameter(torch.empty(
            registers, action_head.input_embedding_dim, device=weight.device, dtype=adapter_dtype,
        ))
        nn.init.normal_(self.registers, std=0.02)
        self.memories = nn.ModuleList(
            TTTMLP(width, heads=ttt_heads, stabilize=stabilize_fast_weights, reference_style=reference_style).to(
                device=weight.device, dtype=adapter_dtype,
            )
            for _ in action_head.model.transformer_blocks
        )
        self.gates = nn.Parameter(torch.full(
            (len(self.memories), width), gate_init, device=weight.device, dtype=adapter_dtype,
        ))
        self.set_stage("pretrain")

    @classmethod
    def from_local_checkpoint(
        cls, checkpoint: str | Path, *, device: str | torch.device | None = None,
        dtype: torch.dtype | None = None, **kwargs,
    ):
        """Load action-head weights from an official local N1.7 safetensors snapshot.

        Install Isaac-GR00T first. No VLM is instantiated, so this can use previously
        extracted compatible embeddings without external model credentials.
        Set `dtype=torch.bfloat16, ttt_dtype=torch.float32` for a BF16 base and
        FP32 adapter parameters. Move the assembled model by device only to
        preserve that split.
        """
        from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
        from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead

        checkpoint = Path(checkpoint)
        head = Gr00tN1d7ActionHead(Gr00tN1d7Config.from_pretrained(checkpoint, local_files_only=True))
        head.load_state_dict(_checkpoint_tensors(checkpoint, "action_head."), strict=True)
        head.to(device=device, dtype=dtype)
        return cls(head, **kwargs)

    def set_stage(self, stage: str):
        """Freeze the original head for pretraining or unfreeze it for head-only post-training."""
        if stage not in ("pretrain", "posttrain"):
            raise ValueError("stage must be pretrain or posttrain")
        self._freeze_base = stage == "pretrain"
        self.head.requires_grad_(not self._freeze_base)
        self.head.train(self.training and not self._freeze_base)
        if self.backbone is not None:
            self.backbone.set_trainable_parameters(not self._freeze_base, not self._freeze_base, 0)
            self.backbone.requires_grad_(not self._freeze_base).train(self.training and not self._freeze_base)
        return self

    def enable_full_posttrain(self, backbone: nn.Module, *, checkpoint_activations: bool = True):
        """Attach the official VLM and unfreeze the full observation-to-action policy."""
        self.backbone = backbone
        self.checkpoint_posttrain = checkpoint_activations
        self.head.set_trainable_parameters(True, True, True)
        return self.set_stage("posttrain")

    def train(self, mode: bool = True):
        super().train(mode)
        if self._freeze_base:
            self.head.eval()
            if self.backbone is not None:
                self.backbone.eval()
        return self

    def _encode_observations(self, frames, batch, steps):
        if self.backbone is None or len(frames) != steps:
            raise ValueError("Full post-training needs the official VLM and one processor batch per timestep")
        features, vl_masks, image_masks = [], [], []
        for frame in frames:
            if frame["input_ids"].ndim != 2 or frame["input_ids"].shape[0] != batch:
                raise ValueError("Each processor batch must have one token sequence per trajectory")
            if self.training and self.checkpoint_posttrain:
                # Transformers temporarily wraps decoder layers to collect the
                # official hidden state. Complete recomputation restores those hooks.
                with set_checkpoint_early_stop(False):
                    feature = checkpoint_activations(self.backbone.posttrain_features, frame, use_reentrant=False)
            else:
                feature = self.backbone.posttrain_features(frame)
            features.append(feature)
            vl_masks.append(frame["attention_mask"].bool())
            image_masks.append(frame["input_ids"] == self.backbone.model.config.image_token_id)
        width = max(value.shape[1] for value in features)
        features = [F.pad(value, (0, 0, 0, width - value.shape[1])) for value in features]
        vl_masks = [F.pad(value, (0, width - value.shape[1])) for value in vl_masks]
        image_masks = [F.pad(value, (0, width - value.shape[1])) for value in image_masks]
        return torch.stack(features, 1), torch.stack(vl_masks, 1), torch.stack(image_masks, 1)

    def forward(
        self, context: Tensor | None, proprio: Tensor, noised_action: Tensor, tau: Tensor,
        state: RolloutState | None = None, valid: Tensor | None = None, *,
        embodiment_id: Tensor | None = None, vl_mask: Tensor | None = None,
        image_mask: Tensor | None = None, action_mask: Tensor | None = None, raw_vl_inputs=None,
        vl_lengths: tuple[int, ...] | None = None,
    ):
        """Return the flow velocity [B,T,H,A] and one fast-weight state per DiT block.

        Context has shape [B,T,V,D]. Proprio has shape [B,T,S,P] with S state
        observations, or [B,T,P] if S is one. Per-VL-token masks are [B,T,V].
        Embodiment IDs are [B] or [B,T]. A regular GR00T action padding mask
        has shape [B,T,H,A] (or last dimension 1). valid masks padded timesteps.
        vl_lengths records the original visual-language width of each padded trajectory.
        """
        head, config = self.head, self.head.config
        if raw_vl_inputs is not None:
            if context is not None or vl_mask is not None or image_mask is not None or vl_lengths is not None:
                raise ValueError("Raw VLM inputs provide their own context and masks")
            context, vl_mask, image_mask = self._encode_observations(raw_vl_inputs, *noised_action.shape[:2])
        elif context is None or (self.backbone is not None and self.training and not self._freeze_base):
            raise ValueError("Full post-training must encode the current raw observations")
        model_dtype = next(head.parameters()).dtype
        context = context.to(dtype=model_dtype)
        proprio = proprio.to(dtype=model_dtype)
        noised_action = noised_action.to(dtype=model_dtype)
        batch, steps, vl_tokens, _ = context.shape
        if vl_lengths is not None:
            if len(vl_lengths) != batch or min(vl_lengths) < 1 or max(vl_lengths) != vl_tokens:
                raise ValueError("vl_lengths must match the trajectories and padded token width")
            if min(vl_lengths) == vl_tokens:
                vl_lengths = None
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
        cross_padding = None
        if vl_lengths is not None:
            widths = torch.tensor(vl_lengths, device=context.device)
            original_tokens = torch.arange(vl_tokens, device=context.device) < widths[:, None]
            cross_padding = original_tokens.repeat_interleave(steps, dim=0)
            vl_mask = vl_mask & cross_padding
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
        checkpoint_blocks = self.training and self.checkpoint_posttrain and torch.is_grad_enabled()
        if vl_lengths is not None:
            backbone = self._project_variable_widths(context, vl_mask, vl_lengths)
        elif checkpoint_blocks:
            backbone = checkpoint_activations(self._project_context, context.flatten(0, 1), vl_mask,
                                              use_reentrant=False)
        else:
            backbone = self._project_context(context.flatten(0, 1), vl_mask)
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
                mask = vl_mask if config.strict_vl_padding_mask else cross_padding

            block_args = (block, memory, tokens, source, mask, temb, hidden_mask, memory_valid,
                          fast, gate, batch, steps, state.index)
            if checkpoint_blocks:
                tokens, updated = checkpoint_activations(self._apply_ttt_block, *block_args, use_reentrant=False)
            else:
                tokens, updated = self._apply_ttt_block(*block_args)
            fast_states.append(updated)

        shift, scale = dit.proj_out_1(F.silu(temb)).chunk(2, dim=-1)
        tokens = dit.norm_out(tokens) * (1 + scale[:, None]) + shift[:, None]
        velocity = head.action_decoder(dit.proj_out_2(tokens[:, -horizon:]), ids)
        if strict_action_mask is not None:
            velocity = velocity * strict_action_mask
        return velocity.unflatten(0, (batch, steps)), RolloutState(
            tuple(fast_states), state.index + steps,
        )

    def _project_context(self, context, mask):
        return self.head.process_backbone_output(
            {"backbone_features": context, "backbone_attention_mask": mask}
        )["backbone_features"].contiguous()

    def _project_variable_widths(self, context, mask, widths):
        # The public head's visual self-attention can ignore its input mask.
        # Project at each original width, then pad the cross-attention sources.
        batch, steps, padded_width, _ = context.shape
        mask = mask.unflatten(0, (batch, steps))
        by_width = defaultdict(list)
        for lane, width in enumerate(widths):
            by_width[width].append(lane)
        projected = [None] * batch
        for width, lanes in by_width.items():
            values = self._project_context(context[lanes, :, :width].flatten(0, 1),
                                           mask[lanes, :, :width].flatten(0, 1))
            values = F.pad(values, (0, 0, 0, padded_width - width)).unflatten(0, (len(lanes), steps))
            for lane, value in zip(lanes, values.unbind(0), strict=True):
                projected[lane] = value
        return torch.stack(projected).flatten(0, 1)

    def _apply_ttt_block(self, block, memory, tokens, source, mask, temb, hidden_mask, memory_valid,
                         fast, gate, batch, steps, start_step):
        after_attention = self._apply_attention(block, tokens, source, mask, temb)
        if hidden_mask is not None:
            after_attention = after_attention * hidden_mask.unsqueeze(-1)
        sequence = after_attention.unflatten(0, (batch, steps))
        recalled, updated = memory(sequence, fast, valid=memory_valid, start_step=start_step)
        tokens = after_attention + recalled.flatten(0, 1).to(after_attention.dtype) * gate.tanh().to(after_attention.dtype)
        tokens = tokens + block.ff(block.norm3(tokens))
        if hidden_mask is not None:
            tokens = tokens * hidden_mask.unsqueeze(-1)
        return tokens, updated

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
