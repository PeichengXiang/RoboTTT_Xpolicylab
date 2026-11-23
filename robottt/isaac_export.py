"""Convert local insert_benchmark HDF5 demonstrations to compact RoboTTT NPZ episodes."""

import argparse
from pathlib import Path

import numpy as np

from .data import rgb_tokens


CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")


def export_episode(source, target, grid):
    import h5py

    with h5py.File(source, "r") as hdf5:
        if hdf5.attrs.get("control_label") != "same_step_control":
            raise ValueError(f"{source}: expected same_step_control labels")
        proprio = hdf5["observations/qpos"][:].astype(np.float32)
        action = hdf5["action"][:].astype(np.float32)
        if proprio.shape != action.shape or proprio.ndim != 2 or proprio.shape[1] != 16:
            raise ValueError(f"{source}: expected matching [frames,16] joint arrays")
        images = [hdf5[f"observations/images/{camera}"] for camera in CAMERAS]
        if any(len(camera) != len(action) for camera in images):
            raise ValueError(f"{source}: camera observations are not aligned with actions")
        tokens = []
        # Bound the resident decoded image size regardless of episode length.
        for start in range(0, len(action), 16):
            frames = np.stack([camera[start : start + 16] for camera in images], axis=1)
            tokens.append(rgb_tokens(frames, grid=grid))
        with target.open("xb") as stream:
            np.savez_compressed(stream, obs=np.concatenate(tokens), state=proprio, actions=action)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Directory containing seed_*/demonstration.hdf5")
    parser.add_argument("destination", type=Path)
    parser.add_argument("--grid", type=int, default=4)
    parser.add_argument("--train", type=int, default=12)
    parser.add_argument("--validation", type=int, default=4)
    parser.add_argument("--seed", type=int, default=64)
    args = parser.parse_args()
    if min(args.train, args.validation, args.grid) < 1:
        parser.error("--train, --validation and --grid must be positive")
    paths = sorted(args.source.glob("seed_*/demonstration.hdf5"))
    if len(paths) < args.train + args.validation:
        parser.error(f"Need {args.train + args.validation} episodes under {args.source}, found {len(paths)}")
    order = np.random.default_rng(args.seed).permutation(len(paths))
    for fold, start, count in (("train", 0, args.train), ("validation", args.train, args.validation)):
        directory = args.destination / fold
        directory.mkdir(parents=True, exist_ok=True)
        for index in order[start : start + count]:
            source = paths[index]
            export_episode(source, directory / (source.parent.name + ".npz"), args.grid)
        print(f"{fold}: {count} episodes in {directory}", flush=True)


if __name__ == "__main__":
    main()
