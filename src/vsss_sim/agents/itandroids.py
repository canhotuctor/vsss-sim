"""ITAndroids strategy as a batched opponent, one ``vss_fast`` agent per process.

``vss_fast`` (built from the itandroids-vss repo, ``fast/``) keeps its strategy
state in process-wide singletons, so N parallel matches need N worker
processes. Workers are *spawned* (never forked) so they do not inherit JAX
runtime state. Nothing here imports ``vss_fast`` in the parent process.

Frames: the agent plays yellow (attacks -x) and receives global-frame state, as
``vss_fast.Agent(yellow=True)`` expects. Its wheel speeds (rad/s) are converted
to the simulator's normalised action with the agent's own wheel radius.
"""
from __future__ import annotations

import functools
import multiprocessing as mp
import sys
from typing import Any, Callable

import numpy as np

from .. import config

AgentFactory = Callable[[], Any]


def vss_fast_factory(
    vss_root: str,
    build_dir: str,
    strategy: str = "EXTREME_OFFENSIVE",
    data_mode: str = "firasim",
    dt: float = config.DT,
    yellow: bool = True,
) -> functools.partial:
    """Picklable factory; ``vss_fast`` is imported only inside the worker."""
    return functools.partial(
        _make_vss_fast_agent, vss_root, build_dir, strategy, data_mode, dt, yellow
    )


def _make_vss_fast_agent(vss_root, build_dir, strategy, data_mode, dt, yellow):
    sys.path.insert(0, build_dir)
    import vss_fast  # noqa: PLC0415 - only importable where it was built

    return vss_fast.Agent(
        vss_root, strategy=strategy, yellow=yellow, data_mode=data_mode, dt=dt
    )


def _worker(conn, factory: AgentFactory) -> None:
    try:
        agent = factory()
        conn.send(("ready", dict(agent.robot_params())))
        while True:
            message = conn.recv()
            command = message[0]
            if command == "close":
                break
            if command == "reset":
                agent.reset()
                conn.send(("ok", None))
            elif command == "step":
                conn.send(("wheels", np.asarray(agent.step(*message[1:]))))
    except BaseException as error:  # report instead of dying silently
        try:
            conn.send(("error", repr(error)))
        except Exception:
            pass
    finally:
        conn.close()


class ITAndroidsPool:
    """``num_envs`` strategy agents stepped in parallel from batched JAX state."""

    def __init__(self, num_envs: int, factory: AgentFactory, timeout: float = 60.0):
        if num_envs < 1:
            raise ValueError("num_envs must be at least 1")
        self.num_envs = num_envs
        self._timeout = timeout
        context = mp.get_context("spawn")
        self._connections = []
        self._processes = []
        for _ in range(num_envs):
            parent, child = context.Pipe()
            process = context.Process(target=_worker, args=(child, factory), daemon=True)
            process.start()
            child.close()
            self._connections.append(parent)
            self._processes.append(process)
        params = [self._receive(i, "ready") for i in range(num_envs)]
        self.robot_params: dict[str, float] = params[0]
        self._wheel_radius = float(self.robot_params["wheel_radius"])
        self._previous_ball = None

    def _receive(self, index: int, expected: str):
        connection = self._connections[index]
        if not connection.poll(self._timeout):
            raise TimeoutError(f"ITAndroids worker {index} gave no reply in {self._timeout}s")
        kind, payload = connection.recv()
        if kind == "error":
            raise RuntimeError(f"ITAndroids worker {index} failed: {payload}")
        if kind != expected:
            raise RuntimeError(f"worker {index}: expected {expected!r}, got {kind!r}")
        return payload

    def reset(self, mask: np.ndarray | None = None) -> None:
        """Reset the agents of the selected envs (all when ``mask`` is None)."""
        indices = range(self.num_envs) if mask is None else np.flatnonzero(mask)
        for i in indices:
            self._connections[i].send(("reset",))
        for i in indices:
            self._receive(i, "ok")

    def step(self, robots: np.ndarray, ball: np.ndarray) -> np.ndarray:
        """Return yellow actions ``(num_envs, N_ROBOTS, 2)`` in [-1, 1].

        ``robots`` is the batched ``SimState.robots`` ``(B, 2, 3, 6)`` and
        ``ball`` the batched ``SimState.ball`` ``(B, 4)``, both global frame.
        """
        robots = np.asarray(robots, dtype=np.float64)
        ball = np.asarray(ball, dtype=np.float64)
        blue, yellow = config.TEAM_BLUE, config.TEAM_YELLOW
        for i, connection in enumerate(self._connections):
            allies, opponents = robots[i, yellow], robots[i, blue]
            connection.send(
                (
                    "step",
                    allies[:, :3].copy(),
                    opponents[:, :3].copy(),
                    ball[i, :2].copy(),
                    allies[:, 3:6].copy(),
                    opponents[:, 3:5].copy(),
                    ball[i, 2:4].copy(),
                )
            )
        wheels = np.stack([self._receive(i, "wheels") for i in range(self.num_envs)])
        linear = wheels * self._wheel_radius
        return np.clip(linear / config.ROBOT_MAX_WHEEL_SPEED, -1.0, 1.0).astype(np.float32)

    def close(self) -> None:
        for connection in self._connections:
            try:
                connection.send(("close",))
            except (BrokenPipeError, OSError):
                pass
        for process in self._processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()

    def __enter__(self) -> "ITAndroidsPool":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
