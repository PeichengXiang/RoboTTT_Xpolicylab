"""Closed-loop two-waypoint benchmark for conditioning on a masked history."""

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from .data import DataSpec


class ContextReach:
    """Follow the order of two targets shown only in an observation prefix."""

    command_scale = np.float32(0.24)
    tolerance = 0.18

    def __init__(
        self,
        seed: int,
        chunk_length: int = 2,
        *,
        phase_chunks: int | None = None,
        prefix_chunks: int = 2,
    ) -> None:
        if chunk_length < 1 or prefix_chunks < 1 or (phase_chunks is not None and phase_chunks < 1):
            raise ValueError("chunk_length, phase_chunks and prefix_chunks must be positive")
        rng = np.random.default_rng(seed)
        self.spec = DataSpec(obs_dim=8, state_dim=2, action_dim=2, tokens_per_obs=2, chunk_length=chunk_length)
        self.phase_chunks = phase_chunks if phase_chunks is not None else math.ceil(8 / chunk_length)
        self.prefix_chunks = prefix_chunks
        self.order = 1 if rng.integers(2) else -1
        self.position = rng.uniform(-0.20, 0.20, size=2).astype(np.float32)
        self.chunks = 0
        self.errors: list[float] = []

    @property
    def done(self) -> bool:
        return self.chunks == 2 * self.phase_chunks

    @property
    def score(self) -> float:
        return sum(error <= self.tolerance for error in self.errors) / 2

    @property
    def success(self) -> bool:
        return self.score == 1.0

    @property
    def endpoint_error(self) -> float:
        return float(np.mean(self.errors)) if self.errors else float("nan")

    def _goal(self, phase: int) -> np.ndarray:
        x = self.order * (0.63 if phase == 0 else -0.63)
        y = 0.35 if phase == 0 else -0.35
        return np.array([x, y], dtype=np.float32)

    def prefix(self, mode: str = "matched") -> tuple[np.ndarray, np.ndarray]:
        """The two visual tokens encode the first and second waypoint in order."""
        if mode == "reset":
            return np.empty((0, 2, 8), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
        if mode not in {"matched", "flipped", "blank"}:
            raise ValueError(f"unknown context mode {mode!r}")
        obs = np.zeros((self.prefix_chunks, 2, 8), dtype=np.float32)
        obs[:, :, 0:2] = np.stack([self._goal(0), self._goal(1)])
        obs[:, :, 2] = [-1, 1]
        obs[:, :, 3] = 1
        obs[:, :, 6] = (np.arange(self.prefix_chunks, dtype=np.float32) + 1)[:, None] / self.prefix_chunks
        if mode == "flipped":
            obs[:, :, 0] *= -1
        if mode == "blank":
            obs[:, :, 0] = 0
        return obs, np.zeros((self.prefix_chunks, 2), dtype=np.float32)

    def observation(self) -> tuple[np.ndarray, np.ndarray]:
        if self.done:
            raise RuntimeError("episode is finished")
        phase, progress = divmod(self.chunks, self.phase_chunks)
        obs = np.zeros((2, 8), dtype=np.float32)
        obs[:, 2] = -1 if phase == 0 else 1
        obs[:, 4] = (progress + 1) / self.phase_chunks
        obs[:, 5] = [-1, 1]
        obs[:, 7] = 1
        return obs, self.position.copy()

    def expert_action(self) -> np.ndarray:
        goal = self._goal(self.chunks // self.phase_chunks)
        position = self.position.copy()
        action_chunk = []
        for _ in range(self.spec.chunk_length):
            action = np.clip((goal - position) / self.command_scale, -1, 1)
            position += self.command_scale * action
            action_chunk.append(action)
        return np.stack(action_chunk)

    def step(self, action_chunk: np.ndarray) -> None:
        if self.done:
            raise RuntimeError("episode is finished")
        action_chunk = np.asarray(action_chunk, dtype=np.float32)
        if action_chunk.shape != (self.spec.chunk_length, 2) or not np.isfinite(action_chunk).all():
            raise ValueError(f"action_chunk must be finite with shape [{self.spec.chunk_length},2]")
        for action in np.clip(action_chunk, -1, 1):
            self.position = np.clip(self.position + self.command_scale * action, -1, 1)
        self.chunks += 1
        if self.chunks % self.phase_chunks == 0:
            goal = self._goal(self.chunks // self.phase_chunks - 1)
            self.errors.append(float(np.linalg.norm(self.position - goal)))


def evaluate_context_reach(
    begin: Callable[[np.ndarray, np.ndarray], Any],
    act: Callable[[np.ndarray, np.ndarray, Any, int, int], tuple[np.ndarray, Any]],
    *,
    episodes: int = 24,
    chunk_length: int = 2,
    seed: int = 10000,
    denoise_steps: int = 4,
    modes: Sequence[str] = ("matched", "flipped", "blank"),
    phase_chunks: int | None = None,
    prefix_chunks: int = 2,
) -> dict[str, dict[str, float | int]]:
    """Evaluate identical initial conditions and denoising seeds with different cues."""
    if episodes < 1 or denoise_steps < 1:
        raise ValueError("episodes and denoise_steps must be positive")
    results = {}
    for mode in modes:
        evaluations = []
        for episode_index in range(episodes):
            episode_seed = seed + episode_index
            env = ContextReach(
                episode_seed, chunk_length, phase_chunks=phase_chunks, prefix_chunks=prefix_chunks
            )
            noise = np.random.default_rng(episode_seed + 42_139)
            memory = begin(*env.prefix(mode))
            while not env.done:
                obs, state = env.observation()
                action_seed = int(noise.integers(2**31 - 1))
                action_chunk, memory = act(obs, state, memory, action_seed, denoise_steps)
                env.step(action_chunk)
            evaluations.append(env)
        results[mode] = {
            "episodes": episodes,
            "successes": sum(env.success for env in evaluations),
            "task_completion": float(np.mean([env.score for env in evaluations])),
            "endpoint_error": float(np.mean([env.endpoint_error for env in evaluations])),
        }
    return results
