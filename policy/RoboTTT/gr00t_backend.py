"""Native GR00T N1.7 observation processing and RoboTTT chunk inference."""

import json
import os
from pathlib import Path

import torch

from robottt.gr00t import Gr00tRoboTTT, _checkpoint_tensors, load_public_backbone
from robottt.gr00t_insert import JOINT_KEY, marvin_modalities, marvin_statistics
from robottt.isaac_export import CAMERAS
from robottt.raw_data import scan_episodes


PROJECTORS = ("head.state_encoder.", "head.action_encoder.", "head.action_decoder.")
TTT_CONFIG_FIELDS = ("registers", "ttt_heads", "gate_init", "stabilize_fast_weights", "reference_style")
VL_INPUT_KEYS = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")


def _config_value(config, key):
    return os.environ.get(f"ROBOTTT_{key.upper()}") or config.get(key)


def _path(config, key):
    value = _config_value(config, key)
    return Path(value).expanduser().resolve(strict=True) if value else None


def _allow_untrained(config):
    value = _config_value(config, "allow_untrained_ttt")
    return str(value).lower() in {"1", "true", "yes"}


def _tunes_projectors(saved):
    return saved["adapter_config"].get("tune_marvin_projectors", saved.get("tune_marvin_projectors", False))


def _load_adapter(policy, saved):
    adapter = saved["adapter"]
    if not isinstance(adapter, dict):
        raise ValueError("Native GR00T checkpoint needs an adapter tensor dictionary")
    if type(saved.get("step")) is not int or saved["step"] < 1:
        raise ValueError("Native trained GR00T checkpoint must record a positive optimizer step")
    tune_projectors = _tunes_projectors(saved)
    if saved.get("model_stage") == "posttrain":
        if policy.backbone is None:
            raise ValueError("Full post-training checkpoint requires the native VLM and complete action head")
        expected = {name for name, _ in policy.named_parameters()}
    else:
        expected = {
            name
            for name in policy.state_dict()
            if not name.startswith(("head.", "backbone.")) or (tune_projectors and name.startswith(PROJECTORS))
        }
    missing, unexpected = expected - adapter.keys(), adapter.keys() - expected
    if missing or unexpected:
        raise ValueError(f"GR00T adapter checkpoint key mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    policy.load_state_dict(adapter, strict=False)


class Gr00tMarvin:
    """Run the actual pretrained VLM and action head with per-episode TTT state."""

    def __init__(self, checkpoint_root, config, device, action_dim, seed):
        from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.data.types import MessageType, VLAStepData
        from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
        from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor

        official_base = _path(config, "gr00t_base")
        if official_base is not None and checkpoint_root != official_base:
            checkpoint = checkpoint_root / "latest.pt" if checkpoint_root.is_dir() else checkpoint_root
            saved = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
            if saved.get("backend") != "gr00t_n1_7":
                raise ValueError(f"Expected a trained GR00T N1.7 adapter checkpoint at {checkpoint}")
            if not isinstance(saved.get("marvin_statistics"), dict):
                raise ValueError("A trained GR00T adapter must provide its Marvin normalization statistics")
            stats = saved["marvin_statistics"]
            status = f"trained adapter step {saved.get('step')} from {checkpoint}"
        else:
            if not _allow_untrained(config):
                raise ValueError(
                    "Provide gr00t_base plus a trained adapter checkpoint, "
                    "or explicitly enable allow_untrained_ttt for an integration smoke test"
                )
            official_base = checkpoint_root
            saved = None
            stats = self._smoke_statistics(config)
            status = "newly initialized TTT on the official base (integration smoke only)"

        processor_assets = _path(config, "gr00t_processor")
        if processor_assets is None:
            raise ValueError("Set gr00t_processor to the local Cosmos-compatible processor assets")
        native = Gr00tN1d7Config.from_pretrained(official_base, local_files_only=True)
        self.processor = Gr00tN1d7Processor.from_pretrained(
            official_base,
            modality_configs=marvin_modalities(native.action_horizon),
            model_name=str(processor_assets),
            max_action_horizon=native.action_horizon,
            max_state_dim=native.max_state_dim,
            max_action_dim=native.max_action_dim,
            strict_action_padding_mask=native.strict_action_padding_mask,
            use_relative_action=False,
            local_files_only=True,
            transformers_loading_kwargs={"local_files_only": True, "trust_remote_code": False},
        )
        self.processor.set_statistics(stats, override=True)
        self.processor.eval()
        self.embodiment = EmbodimentTag.NEW_EMBODIMENT
        if self.processor.action_dim[self.embodiment.value] != action_dim:
            raise ValueError("GR00T Marvin statistics do not match the registered joint action dimension")
        self.step_data = VLAStepData
        self.step_type = MessageType.EPISODE_STEP.value
        self.device = device
        self.dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        self.horizon = native.action_horizon
        if saved is not None and saved.get("horizon", self.horizon) != self.horizon:
            raise ValueError("The adapter checkpoint and native GR00T action horizons differ")
        execution_horizon = _config_value(config, "execution_horizon")
        trained_stride = saved.get("stride", 8) if saved is not None else 8
        self.execution_horizon = int(execution_horizon) if execution_horizon is not None else trained_stride
        if not 1 <= self.execution_horizon <= self.horizon:
            raise ValueError("execution_horizon must be inside the native GR00T action horizon")

        with torch.device("meta"):
            head = Gr00tN1d7ActionHead(native).eval()
        head.load_state_dict(_checkpoint_tensors(official_base, "action_head."), strict=True, assign=True)
        head.to(dtype=self.dtype)
        if saved is not None and _tunes_projectors(saved):
            # Keep the trained projectors in their FP32 parameter format and
            # execute their native kernels under BF16 CUDA autocast, as in training.
            for projector in (head.state_encoder, head.action_encoder, head.action_decoder):
                projector.float()
        saved_config = {"stabilize_fast_weights": True} if saved is None else saved["adapter_config"]
        adapter_options = {name: saved_config[name] for name in TTT_CONFIG_FIELDS if name in saved_config}
        if saved is not None:
            adapter_options["ttt_dtype"] = saved["adapter"]["gates"].dtype
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.policy = Gr00tRoboTTT(head, **adapter_options)
        full_posttrain = saved is not None and saved.get("model_stage") == "posttrain"
        self.backbone = load_public_backbone(official_base, processor_assets, device=device, dtype=self.dtype)
        if full_posttrain:
            self.policy.enable_full_posttrain(self.backbone, checkpoint_activations=False)
        if saved is not None:
            _load_adapter(self.policy, saved)
        self.policy.to(device=device).eval()
        print(
            f"[RoboTTT] official GR00T N1.7 {official_base} on {device}, {status}; "
            f"predict {self.horizon}, execute {self.execution_horizon}",
            flush=True,
        )

    @staticmethod
    def _smoke_statistics(config):
        stats_path = _path(config, "gr00t_statistics")
        if stats_path is not None:
            return json.loads(stats_path.read_text(encoding="utf-8"))
        benchmark = config.get("benchmark_root") or os.environ.get("INSERT_BENCHMARK_ROOT")
        if benchmark is None:
            raise ValueError("Initialized smoke needs gr00t_statistics or an Insert Benchmark root")
        episodes = scan_episodes(Path(benchmark) / "benchmark_data", horizon=40, grid=1, expected_per_task=50)
        return marvin_statistics(episodes)

    @torch.inference_mode()
    def sample(self, frames, state, instruction, memory, *, denoise_steps, generator):
        step = self.step_data(
            images={camera: [frame] for camera, frame in zip(CAMERAS, frames, strict=True)},
            states={JOINT_KEY: state[None]},
            actions={},
            text=instruction,
            embodiment=self.embodiment,
        )
        processed = self.processor([{"type": self.step_type, "content": step}])
        batch = self.processor.collator([processed])["inputs"]
        vl_inputs = {
            key: batch[key].to(self.device, dtype=self.dtype if key == "pixel_values" else None)
            for key in VL_INPUT_KEYS
        }
        features = self.backbone(self.backbone.prepare_input(vl_inputs))
        action_mask = batch.get("action_mask")
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            action, memory = self.policy.sample(
                features["backbone_features"],
                batch["state"].to(self.device),
                memory,
                embodiment_id=batch["embodiment_id"].to(self.device),
                vl_mask=features["backbone_attention_mask"],
                image_mask=features["image_mask"],
                action_mask=action_mask.to(self.device) if action_mask is not None else None,
                denoise_steps=denoise_steps,
                generator=generator,
            )
        physical_action = self.processor.decode_action(
            action.float().cpu().numpy(), self.embodiment, {JOINT_KEY: state[None, None]},
        )[JOINT_KEY]
        return physical_action[0, :self.execution_horizon].astype("float32", copy=False), memory
