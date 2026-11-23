"""Validated Insert Benchmark HDF5 input and a reusable RGB patch cache."""

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from .data import DataSpec
from .isaac_export import CAMERAS


LEVELS = (
    "insert_one", "insert_two", "insert_three", "recover_three", "timed_single", "precision_single",
)
JOINT_CHANNELS = tuple(
    name for side in ("left", "right")
    for name in (*[f"{side}_joint_{index}" for index in range(1, 8)], f"{side}_opening")
)
CACHE_VERSION = 1


@dataclass(frozen=True)
class Episode:
    path: Path
    level: str
    seed: int
    frames: int
    signature: str


def infer_level(instruction):
    """Map one of the benchmark's public instructions to its task token."""
    text = " ".join(str(instruction or "").lower().split())
    if text in LEVELS:
        return text
    if "narrow guide" in text:
        return "precision_single"
    if "five-second clock" in text or "physical clock" in text:
        return "timed_single"
    if "knocked loose" in text:
        return "recover_three"
    if text.startswith("insert 1 optical module"):
        return "insert_one"
    if text.startswith("insert 2 optical modules"):
        return "insert_two"
    if text.startswith("insert 3 optical modules"):
        return "insert_three"
    raise ValueError(f"Cannot infer Insert Benchmark task from instruction {instruction!r}")


def encode_rgb_context(rgb, grid, level):
    """Encode uint8 [F,V,H,W,3] into RGB/XY/camera patches and a task token."""
    if rgb.ndim != 5 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise ValueError("rgb must be uint8 [F,V,H,W,3]")
    frames, views, height, width, _ = rgb.shape
    if views != len(CAMERAS) or not 1 <= grid <= min(height, width):
        raise ValueError(f"Expected {len(CAMERAS)} views and valid square patch grid")
    y_edges = np.linspace(0, height, grid + 1, dtype=int)
    x_edges = np.linspace(0, width, grid + 1, dtype=int)
    area = np.diff(y_edges)[:, None] * np.diff(x_edges)[None, :]
    # Integer sums give exact means at the bounds of each adaptive patch.
    row_sums = np.add.reduceat(rgb, x_edges[:-1], axis=3, dtype=np.uint32)
    sums = np.add.reduceat(row_sums, y_edges[:-1], axis=2, dtype=np.uint32)
    patches = np.empty((frames, views, grid, grid, 6), dtype=np.float32)
    patches[..., :3] = sums / (255 * area[..., None])
    patches[..., 3] = (x_edges[:-1] + x_edges[1:]) / width - 1
    patches[..., 4] = ((y_edges[:-1] + y_edges[1:]) / height - 1)[:, None]
    patches[..., 5] = np.linspace(-1, 1, views)[:, None, None]
    tokens = np.empty((frames, views * grid * grid + 1, 6), dtype=np.float32)
    tokens[:, :-1] = patches.reshape(frames, -1, 6)
    tokens[:, -1] = 0
    tokens[:, -1, LEVELS.index(level)] = 1
    tokens[:, -1, -1] += 3  # Task token marker outside the camera-id range.
    return tokens


def _as_text(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _field_shapes(file, frames, grid):
    if file["action"].shape != (frames, 16) or file["observations/qpos"].shape != (frames, 16):
        raise ValueError("Invalid action/qpos dimensions")
    for camera in CAMERAS:
        image = file[f"observations/images/{camera}"]
        if (image.ndim != 4 or image.shape[0] != frames or image.shape[-1] != 3
            or min(image.shape[1:3]) < grid or image.dtype != np.uint8):
            raise ValueError(f"Invalid RGB dataset {camera}: {image.shape}, {image.dtype}")


def scan_episodes(root, horizon, grid, expected_per_task):
    """Check metadata and count all finalized demonstrations before committing GPUs."""
    import h5py

    paths = sorted(Path(root).rglob("demonstration.hdf5"))
    if not paths:
        raise FileNotFoundError(f"No finalized demonstration.hdf5 files under {root}")
    episodes = []
    for path in paths:
        if any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        try:
            with h5py.File(path, "r") as file:
                attrs = file.attrs
                level = _as_text(attrs["level"])
                seed = int(attrs["seed"])
                frames = int(attrs["steps"])
                if level not in LEVELS or frames < horizon or int(attrs["frequency"]) != 30:
                    raise ValueError("Invalid level, frame count, or non-30 Hz control data")
                if _as_text(attrs["control_label"]) != "same_step_control":
                    raise ValueError("Expected observation aligned with the same-step joint setpoint")
                if "joint_channels" in attrs and tuple(map(_as_text, attrs["joint_channels"])) != JOINT_CHANNELS:
                    raise ValueError("Input channel order does not match Marvin joint setpoints")
                if "gripper_unit" in attrs and _as_text(attrs["gripper_unit"]) != "fraction_0_closed_1_open":
                    raise ValueError("Expected fractional gripper opening with 0 closed and 1 open")
                _field_shapes(file, frames, grid)
            stat = path.stat()
            key = f"{CACHE_VERSION}:{level}:{seed}:{frames}:{stat.st_size}:{stat.st_mtime_ns}"
            episodes.append(Episode(path, level, seed, frames, sha256(key.encode()).hexdigest()[:20]))
        except (OSError, KeyError, ValueError) as err:
            raise ValueError(f"Invalid or incompletely synchronized demonstration {path}: {err}") from err
    counts = Counter(episode.level for episode in episodes)
    keys = {(episode.level, episode.seed) for episode in episodes}
    if len(keys) != len(episodes):
        raise ValueError("Found duplicate (task, seed) demonstrations")
    if expected_per_task and any(counts[level] != expected_per_task for level in LEVELS):
        actual = {level: counts[level] for level in LEVELS}
        raise ValueError(f"Expected {expected_per_task} demos per task, found {actual}")
    if not episodes:
        raise ValueError(f"No usable demonstrations under {root}")
    return sorted(episodes, key=lambda item: (LEVELS.index(item.level), item.seed))


def split_episodes(episodes, val_per_task, seed):
    train, validation = [], []
    rng = np.random.default_rng(seed)
    for level in LEVELS:
        group = [episode for episode in episodes if episode.level == level]
        if not group:
            continue
        if val_per_task >= len(group):
            raise ValueError(f"Need to leave at least one training episode for {level}")
        order = rng.permutation(len(group))
        validation.extend(group[index] for index in order[:val_per_task])
        train.extend(group[index] for index in order[val_per_task:])
    return train, validation


def _feature_path(cache, episode, grid):
    return cache / f"{episode.level}-{episode.seed}-{grid}-{episode.signature}.npy"


def cache_episode(cache, episode, grid):
    """Atomically cache at camera-frame cadence for any future crop/horizon."""
    import h5py

    target = _feature_path(cache, episode, grid)
    shape = (episode.frames, len(CAMERAS) * grid ** 2 + 1, 6)
    try:
        cached = np.load(target, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError):
        cached = None
    if cached is not None and cached.shape == shape and cached.dtype == np.float16:
        return
    temporary = target.with_suffix(f".tmp.{os.getpid()}")
    try:
        features = np.lib.format.open_memmap(temporary, mode="w+", shape=shape, dtype=np.float16)
        with h5py.File(episode.path, "r") as file:
            images = [file[f"observations/images/{camera}"] for camera in CAMERAS]
            for start in range(0, episode.frames, 32):
                selection = slice(start, start + 32)
                rgb = np.stack([image[selection] for image in images], axis=1)
                features[selection] = encode_rgb_context(rgb, grid, episode.level)
        features.flush()
        del features
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def _channel_stats(arrays):
    count = sum(len(array) for array in arrays)
    total = sum(array.sum(axis=0, dtype=np.float64) for array in arrays)
    squares = sum(np.square(array.astype(np.float64)).sum(axis=0) for array in arrays)
    mean = total / count
    scale = np.sqrt(np.maximum(squares / count - mean ** 2, 0.05 ** 2))
    return mean.astype(np.float32).tolist(), scale.astype(np.float32).tolist()


def read_state_actions(episodes):
    """Read inexpensive proprio/action only once per launcher process."""
    import h5py

    values = {}
    for episode in episodes:
        with h5py.File(episode.path, "r") as file:
            states = np.asarray(file["observations/qpos"], dtype=np.float32)
            actions = np.asarray(file["action"], dtype=np.float32)
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError(f"Nonfinite state or action in {episode.path}")
        values[episode] = states, actions
    return values


def preprocessing(train, values, grid):
    states, actions = zip(*(values[episode] for episode in train), strict=True)
    state_mean, state_scale = _channel_stats(states)
    action_mean, action_scale = _channel_stats(actions)
    return dict(
        grid=grid, task_token=True, state_mean=state_mean, state_scale=state_scale,
        action_mean=action_mean, action_scale=action_scale, frequency=30, cameras=list(CAMERAS),
    )


def manifest(episodes, train, validation):
    def rows(group):
        return [
            {"level": item.level, "seed": item.seed, "frames": item.frames, "signature": item.signature}
            for item in group
        ]

    data = {
        "counts": {level: sum(item.level == level for item in episodes) for level in LEVELS},
        "train": rows(train),
        "validation": rows(validation),
    }
    digest = sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    return {**data, "sha256": digest}


class CachedEpisodes(Dataset):
    def __init__(self, episodes, values, cache, *, horizon, transform):
        self.episodes = episodes
        self.horizon = horizon
        self.grid = transform["grid"]
        self.spec = DataSpec(6, 16, 16, len(CAMERAS) * self.grid ** 2 + 1, horizon)
        self.features = []
        self.normalized = []
        state_mean = np.array(transform["state_mean"], dtype=np.float32)
        state_scale = np.array(transform["state_scale"], dtype=np.float32)
        action_mean = np.array(transform["action_mean"], dtype=np.float32)
        action_scale = np.array(transform["action_scale"], dtype=np.float32)
        for episode in episodes:
            self.features.append(np.load(_feature_path(cache, episode, self.grid), mmap_mode="r", allow_pickle=False))
            state, action = values[episode]
            self.normalized.append(((state - state_mean) / state_scale, (action - action_mean) / action_scale))

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, selection):
        index, crop_seed, context_length = selection
        horizon = self.horizon
        episode = self.episodes[index]
        rng = np.random.default_rng(crop_seed)
        # Randomize action-chunk boundaries across repeated draws of a trajectory.
        offset = int(rng.integers(min(horizon, episode.frames - horizon + 1)))
        chunks = (episode.frames - offset) // horizon
        length = min(context_length, chunks)
        start = offset + horizon * int(rng.integers(chunks - length + 1))
        end = start + horizon * length
        features = np.array(self.features[index][start:end:horizon], dtype=np.float32)
        states, actions = self.normalized[index]
        return {"obs": features, "state": states[start:end:horizon],
                "actions": actions[start:end].reshape(length, horizon, -1),
                "loss_mask": np.ones(length, dtype=bool)}
