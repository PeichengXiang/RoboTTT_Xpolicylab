"""Read normalized Marvin targets, frozen features or raw GR00T vision inputs."""

from hashlib import sha256
from itertools import islice
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch

from .gr00t import load_public_backbone
from .gr00t_insert import marvin_modalities, read_marvin_steps


BASE_REVISION = "373d34e2723d928870157d07dc216a19a624b5bb"
UPSTREAM_REVISION = "708617f5716c63fc0a327391492ea2b91977ae90"
PROCESSOR_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
HORIZON = 40
STRIDE = 8
CACHE_VERSION = 1
CACHE_FIELDS = ("features", "vl_mask", "image_mask", "state", "actions", "action_mask", "embodiment_id")
SUPERVISION_FIELDS = ("state", "actions", "action_mask", "embodiment_id")
VL_INPUT_FIELDS = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")


def cache_directory(root, data_sha256, stride=STRIDE):
    identity = [CACHE_VERSION, BASE_REVISION, UPSTREAM_REVISION, PROCESSOR_REVISION,
                HORIZON, stride, data_sha256]
    return root / sha256(json.dumps(identity).encode()).hexdigest()[:24]


def cache_path(directory: Path, episode):
    return directory / f"{episode.level}-{episode.seed}-{episode.signature}.safetensors"


def episode_chunks(episode, stride=STRIDE):
    return max(0, (episode.frames - HORIZON) // stride + 1)


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


def _write_cache(target, tensors):
    from safetensors.torch import save_file

    with NamedTemporaryFile(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        save_file({name: value.contiguous() for name, value in tensors.items()}, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def _reuse_dense_episode(source, target, episode, stride):
    from safetensors import safe_open

    ratio = stride // STRIDE
    expected = episode_chunks(episode)
    with safe_open(source, framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(CACHE_FIELDS):
            raise ValueError(f"Unexpected tensor fields in the stride-{STRIDE} cache {source}")
        tensors = {}
        for name in CACHE_FIELDS:
            tensor = handle.get_tensor(name)
            if tensor.shape[0] != expected:
                raise ValueError(f"Unexpected frame count for {name} in the stride-{STRIDE} cache {source}")
            tensors[name] = tensor[::ratio].contiguous()
        if tensors["features"].dtype != torch.bfloat16:
            raise ValueError(f"Expected BF16 features in the stride-{STRIDE} cache {source}")
        # Every selected observation and its entire H-step target begin at the
        # same raw frame as a direct pass at the new stride.
        if len(tensors["features"]) != episode_chunks(episode, stride):
            raise ValueError(f"Subsampling {source} did not produce the expected cache length")
        _write_cache(target, tensors)


@torch.inference_mode()
def build_cache(episodes, directory, snapshot, assets, stats, device, batch_size, on_episode, *,
                stride=STRIDE, dense_cache=None):
    if stride < 1 or (dense_cache is not None and (stride <= STRIDE or stride % STRIDE)):
        raise ValueError(f"Stride must be positive, and reuse needs an integer multiple of {STRIDE}")
    directory.mkdir(parents=True, exist_ok=True)
    pending = [episode for episode in episodes if not cache_path(directory, episode).is_file()]
    counts = {"existing": len(episodes) - len(pending), "reused": 0, "encoded": 0}
    if not pending:
        return counts
    uncached = []
    for episode in pending:
        source = cache_path(dense_cache, episode) if dense_cache is not None else None
        if source is None or not source.is_file():
            uncached.append(episode)
            continue
        _reuse_dense_episode(source, cache_path(directory, episode), episode, stride)
        counts["reused"] += 1
        on_episode(episode, counts["reused"], len(pending))
    if not uncached:
        return counts

    from gr00t.data.types import MessageType

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    backbone = load_public_backbone(snapshot, assets, device=device, dtype=dtype)
    processor = load_processor(snapshot, assets, stats)
    for episode in uncached:
        parts = {}
        frame_ids = range(0, episode.frames - HORIZON + 1, stride)
        steps = read_marvin_steps(episode, frame_ids, HORIZON)
        while group := list(islice(steps, batch_size)):
            processed = [processor([{"type": MessageType.EPISODE_STEP.value, "content": sample}]) for sample in group]
            batch = processor.collator(processed)["inputs"]
            inputs = {key: batch[key].to(device=device, dtype=dtype if key == "pixel_values" else None)
                      for key in VL_INPUT_FIELDS}
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
        _write_cache(cache_path(directory, episode), {name: torch.cat(values) for name, values in parts.items()})
        counts["encoded"] += 1
        on_episode(episode, counts["reused"] + counts["encoded"], len(pending))
    return counts


def read_cached_segment(directory, episode, start, end, device, *, fields=CACHE_FIELDS):
    from safetensors import safe_open

    with safe_open(cache_path(directory, episode), framework="pt", device="cpu") as handle:
        return {name: handle.get_slice(name)[start:end].unsqueeze(0).to(device)
                for name in fields}


def _raw_cache_path(directory, episode, stride, dtype):
    if dtype not in (torch.bfloat16, torch.float32) or stride < 1:
        raise ValueError("Official processor input cache requires a positive stride and BF16 or FP32 pixels")
    # The parent is already keyed by data, processor/model revisions and horizon.
    return cache_path(directory / f"processor-inputs-v1-s{stride}-{dtype}", episode)


def _prepare_raw_episode(target, episode, processor, stride, dtype):
    from gr00t.data.types import MessageType

    if processor.training:
        raise ValueError("Pixel caching requires the deterministic eval-mode GR00T processor")
    target.parent.mkdir(parents=True, exist_ok=True)
    frame_indices = range(0, episode.frames - HORIZON + 1, stride)
    shared = {}
    pixels = None
    for index, sample in enumerate(read_marvin_steps(episode, frame_indices, HORIZON)):
        item = processor([{"type": MessageType.EPISODE_STEP.value, "content": sample}])
        inputs = processor.collator([item])["inputs"]
        patches = inputs["pixel_values"]
        metadata = {key: inputs[key] for key in VL_INPUT_FIELDS if key != "pixel_values"}
        if pixels is None:
            shared = metadata
            pixels = torch.empty((len(frame_indices), *patches.shape), dtype=dtype)
        elif (patches.shape != pixels.shape[1:] or
              any(not torch.equal(value, shared[key]) for key, value in metadata.items())):
            raise ValueError(f"Processor token or image layout changed in {episode.path} at frame {frame_indices[index]}")
        pixels[index].copy_(patches)
    if pixels is None:
        raise ValueError(f"Episode has no complete action chunks: {episode.path}")
    _write_cache(target, {**shared, "pixel_values": pixels})


def build_raw_cache(episodes, directory, processor, *, stride, dtype=torch.bfloat16, on_episode=None):
    """Preprocess original RGBs once on CPU for a trainable VLM. No model is loaded."""
    counts = {"existing": 0, "encoded": 0}
    for index, episode in enumerate(episodes, 1):
        target = _raw_cache_path(directory, episode, stride, dtype)
        if target.is_file():
            counts["existing"] += 1
            continue
        _prepare_raw_episode(target, episode, processor, stride, dtype)
        counts["encoded"] += 1
        if on_episode is not None:
            on_episode(episode, index, len(episodes))
    return counts


def read_raw_segment(directory, episode, start, end, device, *, stride, processor):
    """Load supervision and one official VLM input per chronological policy call."""
    from safetensors import safe_open

    if stride < 1 or not 0 <= start < end <= episode_chunks(episode, stride):
        raise ValueError(f"Invalid raw policy segment [{start}, {end}) at stride {stride} in {episode.path}")
    batch = read_cached_segment(directory, episode, start, end, device, fields=SUPERVISION_FIELDS)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    path = _raw_cache_path(directory, episode, stride, dtype)
    if not path.is_file():
        _prepare_raw_episode(path, episode, processor, stride, dtype)
    with safe_open(path, framework="pt", device="cpu") as handle:
        fixed = {key: handle.get_tensor(key).to(device) for key in VL_INPUT_FIELDS if key != "pixel_values"}
        pixels = handle.get_slice("pixel_values")[start:end].to(device)
    batch["vl_inputs"] = [{**fixed, "pixel_values": frame} for frame in pixels.unbind(0)]
    return batch
