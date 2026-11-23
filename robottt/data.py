"""Episode inputs for sequence flow matching."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DataSpec:
    obs_dim: int
    state_dim: int
    action_dim: int
    tokens_per_obs: int
    chunk_length: int


def rgb_tokens(rgb: np.ndarray, grid: int = 2) -> np.ndarray:
    """Convert uint8 [F,V,Y,X,3] into [F,V*grid*grid,6] RGB/XY/camera tokens."""
    if rgb.ndim != 5 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise ValueError("rgb must be uint8 [F,V,Y,X,3]")
    frames, views, height, width, _ = rgb.shape
    if grid < 1 or min(height, width) < grid or views == 0:
        raise ValueError("rgb needs at least one camera and grid <= min(height,width)")

    tokens = np.empty((frames, views, grid, grid, 6), dtype=np.float32)
    y_edges = np.linspace(0, height, grid + 1, dtype=int)
    x_edges = np.linspace(0, width, grid + 1, dtype=int)
    camera_ids = np.linspace(-1, 1, views, dtype=np.float32) if views > 1 else np.zeros(1)
    for y, (y0, y1) in enumerate(zip(y_edges[:-1], y_edges[1:])):
        for x, (x0, x1) in enumerate(zip(x_edges[:-1], x_edges[1:])):
            patch = rgb[:, :, y0:y1, x0:x1]
            tokens[:, :, y, x, :3] = patch.mean(axis=(2, 3), dtype=np.float32) / 255
            tokens[:, :, y, x, 3] = (x0 + x1) / width - 1
            tokens[:, :, y, x, 4] = (y0 + y1) / height - 1
            tokens[:, :, y, x, 5] = camera_ids
    return tokens.reshape(frames, views * grid * grid, 6)


class NPZEpisodes(Dataset):
    """Read contiguous windows from one .npz per episode or pre-paired sequence."""

    def __init__(
        self,
        directory: str | Path,
        chunk_length: int,
        context_length: int | None = None,
        *,
        rgb_grid: int = 2,
        seed: int | None = None,
    ) -> None:
        path = Path(directory)
        self.paths = [path] if path.is_file() else sorted(path.rglob("*.npz"))
        if not self.paths:
            raise FileNotFoundError(f"No .npz episode files found at {path}")
        if chunk_length < 1 or (context_length is not None and context_length < 1):
            raise ValueError("chunk_length and context_length must be positive")
        if seed is not None and seed < 0:
            raise ValueError("seed must be non-negative")
        self.chunk_length = chunk_length
        self.context_length = context_length
        self.rgb_grid = rgb_grid
        self.seed = seed
        obs, state, actions, _ = self._load(self.paths[0], crop=False, sample_length=1)
        self.spec = DataSpec(obs.shape[2], state.shape[1], actions.shape[2], obs.shape[1], chunk_length)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        obs, state, actions, loss_mask = self._load(self.paths[index], index=index % len(self.paths))
        actual_spec = DataSpec(
            obs.shape[2], state.shape[1], actions.shape[2], obs.shape[1], actions.shape[1]
        )
        if actual_spec != self.spec:
            raise ValueError(f"{self.paths[index]} has {actual_spec}, expected {self.spec}")
        return {"obs": obs, "state": state, "actions": actions, "loss_mask": loss_mask}

    def _load(
        self, path: Path, *, crop: bool = True, sample_length: int | None = None, index: int = 0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        with np.load(path, allow_pickle=False) as arrays:
            if "state" not in arrays or "actions" not in arrays:
                raise ValueError(f"{path}: state and actions are required")
            states, actions = arrays["state"], arrays["actions"]
            if states.ndim != 2 or actions.ndim not in (2, 3):
                raise ValueError(f"{path}: state must have shape [F,P], actions [F,A] or [T,H,A]")
            if len(states) != len(actions) or not states.shape[-1] or not actions.shape[-1]:
                raise ValueError(f"{path}: state and actions must align and have non-empty channels")
            if actions.ndim == 3 and actions.shape[1] != self.chunk_length:
                raise ValueError(f"{path}: stored action chunk length disagrees with chunk_length")

            stride = self.chunk_length if actions.ndim == 2 else 1
            blocks = len(states) // stride
            if blocks < 1:
                raise ValueError(f"{path}: episode contains no full action chunk")

            length = min(blocks, sample_length or self.context_length or blocks)
            mask = np.ones(blocks, dtype=bool)
            if "loss_mask" in arrays:
                frame_mask = arrays["loss_mask"]
                if frame_mask.shape != (len(states),):
                    raise ValueError(f"{path}: loss_mask must be [F] or [T] and align with state")
                if not np.all((frame_mask == 0) | (frame_mask == 1)):
                    raise ValueError(f"{path}: loss_mask values must be boolean or 0/1")
                # Mixed-control chunks retain context without supervising autonomous actions.
                mask = frame_mask[:blocks * stride].reshape(blocks, stride).all(axis=1)
            if not mask.any():
                raise ValueError(f"{path}: combine context with at least one fully supervised chunk")

            start = 0
            if crop and blocks > length:
                counts = np.concatenate(([0], np.cumsum(mask)))
                eligible = np.flatnonzero(counts[length:] > counts[:-length])
                if self.seed is None:
                    start = int(eligible[torch.randint(len(eligible), ()).item()])
                else:
                    rng = np.random.default_rng(np.random.SeedSequence([self.seed, index]))
                    start = int(rng.choice(eligible))
            ticks = slice(start * stride, (start + length) * stride)
            observations = slice(ticks.start, ticks.stop, stride)

            if "obs" in arrays:
                obs = arrays["obs"]
                if obs.ndim != 3 or len(obs) != len(states) or not all(obs.shape[1:]):
                    raise ValueError(f"{path}: obs must be [F,S,O] and align with state")
                obs = obs[observations]
            elif "rgb" in arrays:
                rgb = arrays["rgb"]
                if rgb.ndim != 5 or len(rgb) != len(states):
                    raise ValueError(f"{path}: rgb must be [F,V,Y,X,3] and align with state")
                obs = rgb_tokens(rgb[observations], grid=self.rgb_grid)
            else:
                raise ValueError(f"{path}: expected obs or rgb")

            obs = np.ascontiguousarray(obs, dtype=np.float32)
            state = np.ascontiguousarray(states[observations], dtype=np.float32)
            chunks = np.ascontiguousarray(
                actions[ticks].reshape(length, self.chunk_length, actions.shape[-1]), dtype=np.float32
            )
            if not all(np.isfinite(array).all() for array in (obs, state, chunks)):
                raise ValueError(f"{path}: obs, state and actions must be finite")
            return obs, state, chunks, mask[start:start + length]


def collate_episodes(examples: list[dict[str, np.ndarray]]) -> dict[str, torch.Tensor]:
    """Right-pad a batch while keeping loss masking distinct from padding."""
    max_length = max(item["state"].shape[0] for item in examples)
    batch_size = len(examples)
    batch = {
        name: torch.zeros((batch_size, max_length, *examples[0][name].shape[1:]), dtype=torch.float32)
        for name in ("obs", "state", "actions")
    }
    batch["loss_mask"] = torch.zeros((batch_size, max_length), dtype=torch.bool)
    batch["valid_mask"] = torch.zeros_like(batch["loss_mask"])
    for index, item in enumerate(examples):
        length = item["state"].shape[0]
        for name in ("obs", "state", "actions", "loss_mask"):
            batch[name][index, :length] = torch.as_tensor(item[name])
        batch["valid_mask"][index, :length] = True
    return batch


class SyntheticEpisodes(Dataset):
    """Deterministic expert episodes for the context-conditioned reach task."""

    def __init__(
        self,
        count: int,
        chunk_length: int = 2,
        seed: int = 0,
        *,
        phase_chunks: int | None = None,
        prefix_chunks: int = 2,
    ) -> None:
        from .benchmark import ContextReach

        if count < 1 or seed < 0:
            raise ValueError("count must be positive and seed non-negative")
        env = ContextReach(seed, chunk_length, phase_chunks=phase_chunks, prefix_chunks=prefix_chunks)
        self.spec = env.spec
        self.count = count
        self.seed = seed
        self.phase_chunks = env.phase_chunks
        self.prefix_chunks = prefix_chunks

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        from .benchmark import ContextReach

        if not 0 <= index < self.count:
            raise IndexError(index)
        env = ContextReach(
            self.seed + index,
            self.spec.chunk_length,
            phase_chunks=self.phase_chunks,
            prefix_chunks=self.prefix_chunks,
        )
        prefix_obs, prefix_state = env.prefix("matched")
        observations, states = list(prefix_obs), list(prefix_state)
        actions = [np.zeros((self.spec.chunk_length, self.spec.action_dim), dtype=np.float32)
                   for _ in range(self.prefix_chunks)]
        while not env.done:
            obs, state = env.observation()
            action = env.expert_action()
            observations.append(obs)
            states.append(state)
            actions.append(action)
            env.step(action)
        mask = np.ones(len(actions), dtype=bool)
        mask[:self.prefix_chunks] = False
        return {"obs": np.stack(observations), "state": np.stack(states),
                "actions": np.stack(actions), "loss_mask": mask}
