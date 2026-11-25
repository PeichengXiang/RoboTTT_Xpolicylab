"""Prepare frozen GR00T visual-language features from real demonstrations."""

from hashlib import sha256
from itertools import islice
import json
import os
from pathlib import Path

import torch

from .gr00t import load_public_backbone
from .gr00t_insert import marvin_modalities, read_marvin_steps


BASE_REVISION = "373d34e2723d928870157d07dc216a19a624b5bb"
UPSTREAM_REVISION = "708617f5716c63fc0a327391492ea2b91977ae90"
PROCESSOR_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
HORIZON = 40
STRIDE = 8
CACHE_VERSION = 1


def cache_directory(root, data_sha256):
    identity = [CACHE_VERSION, BASE_REVISION, UPSTREAM_REVISION, PROCESSOR_REVISION,
                HORIZON, STRIDE, data_sha256]
    return root / sha256(json.dumps(identity).encode()).hexdigest()[:24]


def cache_path(directory: Path, episode):
    return directory / f"{episode.level}-{episode.seed}-{episode.signature}.safetensors"


def episode_chunks(episode):
    return (episode.frames - HORIZON) // STRIDE + 1


def load_processor(snapshot, assets, stats):
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor

    processor = Gr00tN1d7Processor.from_pretrained(
        snapshot, modality_configs=marvin_modalities(HORIZON), model_name=str(assets.resolve()),
        use_relative_action=False, local_files_only=True,
        transformers_loading_kwargs={"local_files_only": True, "trust_remote_code": False},
    )
    processor.set_statistics(stats, override=True)
    processor.eval()
    return processor


@torch.inference_mode()
def build_cache(episodes, directory, snapshot, assets, stats, device, batch_size, on_episode):
    from gr00t.data.types import MessageType
    from safetensors.torch import save_file

    directory.mkdir(parents=True, exist_ok=True)
    pending = [episode for episode in episodes if not cache_path(directory, episode).is_file()]
    if not pending:
        return
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    backbone = load_public_backbone(snapshot, assets, device=device, dtype=dtype)
    processor = load_processor(snapshot, assets, stats)
    for index, episode in enumerate(pending, 1):
        parts = {}
        frame_ids = range(0, episode.frames - HORIZON + 1, STRIDE)
        steps = read_marvin_steps(episode, frame_ids, HORIZON)
        while group := list(islice(steps, batch_size)):
            processed = [processor([{"type": MessageType.EPISODE_STEP.value, "content": sample}]) for sample in group]
            batch = processor.collator(processed)["inputs"]
            inputs = {key: batch[key].to(device=device, dtype=dtype if key == "pixel_values" else None)
                      for key in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")}
            result = backbone(backbone.prepare_input(inputs))
            features = result["backbone_features"]
            if not torch.isfinite(features).all():
                raise FloatingPointError(f"Nonfinite VLM features in {episode.path}")
            if not result["backbone_attention_mask"].all():
                raise ValueError(f"Inconsistent visual-language token lengths within {episode.path}")
            fp = {
                "features": features.to(device="cpu", dtype=torch.bfloat16),
                "vl_mask": result["backbone_attention_mask"].cpu(),
                "image_mask": result["image_mask"].cpu(),
                "state": batch["state"], "actions": batch["action"],
                "action_mask": batch["action_mask"].bool(), "embodiment_id": batch["embodiment_id"],
            }
            for name, value in fp.items():
                parts.setdefault(name, []).append(value)
        target = cache_path(directory, episode)
        temporary = target.with_suffix(f".tmp.{os.getpid()}")
        try:
            save_file({name: torch.cat(values).contiguous() for name, values in parts.items()}, temporary)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        on_episode(episode, index, len(pending))


def read_cached_segment(directory, episode, start, end, device):
    from safetensors import safe_open

    with safe_open(cache_path(directory, episode), framework="pt", device="cpu") as handle:
        return {name: handle.get_slice(name)[start:end].unsqueeze(0).to(device)
                for name in handle.keys()}
