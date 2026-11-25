"""Feed native GR00T N1.7 processors from Insert Benchmark demonstrations."""

from collections.abc import Sequence

import numpy as np

from .isaac_export import CAMERAS
from .raw_data import Episode, read_state_actions


JOINT_KEY = "joints"


def marvin_modalities(horizon: int):
    """Configure a new embodiment with the benchmark's 16 absolute joint channels."""
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig

    if horizon < 1:
        raise ValueError("Action horizon must be positive")
    return {
        EmbodimentTag.NEW_EMBODIMENT.value: {
            "video": ModalityConfig(delta_indices=[0], modality_keys=list(CAMERAS)),
            "state": ModalityConfig(delta_indices=[0], modality_keys=[JOINT_KEY]),
            "action": ModalityConfig(
                delta_indices=list(range(horizon)),
                modality_keys=[JOINT_KEY],
                action_configs=[
                    ActionConfig(ActionRepresentation.ABSOLUTE, ActionType.NON_EEF, ActionFormat.DEFAULT),
                ],
            ),
            "language": ModalityConfig(delta_indices=[0], modality_keys=["instruction"]),
        },
    }


def _statistics(values: np.ndarray):
    q01, q99 = np.quantile(values, [0.01, 0.99], axis=0)
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0, dtype=np.float64).tolist(),
        "std": values.std(axis=0, dtype=np.float64).tolist(),
        "q01": q01.tolist(),
        "q99": q99.tolist(),
    }


def marvin_statistics(train: Sequence[Episode]):
    """Compute the official processor's statistics using training demonstrations."""
    from gr00t.data.embodiment_tags import EmbodimentTag

    if not train:
        raise ValueError("At least one training demonstration is required")
    states, actions = zip(*read_state_actions(train).values(), strict=True)
    return {
        EmbodimentTag.NEW_EMBODIMENT.value: {
            "state": {JOINT_KEY: _statistics(np.concatenate(states))},
            "action": {JOINT_KEY: _statistics(np.concatenate(actions))},
        },
    }


def read_marvin_step(episode: Episode, index: int, horizon: int):
    """Return the stored RGBs, original instruction and an aligned action chunk."""
    return next(read_marvin_steps(episode, (index,), horizon))


def read_marvin_steps(episode: Episode, indices: Sequence[int], horizon: int):
    """Read multiple control steps while reusing the open HDF5 file."""
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.types import VLAStepData
    import h5py

    with h5py.File(episode.path, "r") as file:
        instruction = file.attrs["instruction"]
        if isinstance(instruction, bytes):
            instruction = instruction.decode("utf-8")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"Missing instruction text in {episode.path}")
        images = {camera: file[f"observations/images/{camera}"] for camera in CAMERAS}
        states, actions = file["observations/qpos"], file["action"]
        for index in indices:
            if horizon < 1 or index < 0 or index + horizon > episode.frames:
                raise ValueError(f"Incomplete action chunk for {episode.path} at {index=} with {horizon=}")
            yield VLAStepData(
                images={camera: [frames[index]] for camera, frames in images.items()},
                states={JOINT_KEY: np.asarray(states[index:index + 1], dtype=np.float32)},
                actions={JOINT_KEY: np.asarray(actions[index:index + horizon], dtype=np.float32)},
                text=instruction,
                embodiment=EmbodimentTag.NEW_EMBODIMENT,
            )
