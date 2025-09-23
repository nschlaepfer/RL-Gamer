"""Real-time viewer for the RL-Gamer runner.

This module renders a live mosaic of Atari observations using OpenCV.
It satisfies the interfaces expected by ``gamer.py`` / ``og.py`` while
providing visual feedback directly on screen. Close the window or press
``q`` to shut the viewer down gracefully.
"""

from __future__ import annotations

import math
import sys
import threading
import time
from typing import Sequence

import numpy as np
import torch

try:
    import cv2  # type: ignore
except ImportError as exc:  # pragma: no cover - runtime dependency
    cv2 = None
    print("[bg_record] OpenCV not available; viewer disabled", file=sys.stderr)


_log_lock = threading.Lock()


def bind_logger(game_id: str, env_idx: int, info_tensor: torch.Tensor) -> None:
    with _log_lock:
        info_tensor[env_idx].zero_()


def log_step(action, obs, reward, terminated, truncated) -> None:
    # Hook for custom logging if desired (e.g., stats aggregation).
    pass


def log_close() -> None:
    pass


def _tile_frames(frames: np.ndarray, games: Sequence[str]) -> np.ndarray:
    num_envs, h, w, c = frames.shape
    cols = int(math.ceil(math.sqrt(num_envs)))
    rows = int(math.ceil(num_envs / cols))

    padded = np.zeros((rows * h, cols * w, c), dtype=frames.dtype)
    for idx in range(num_envs):
        r = idx // cols
        cidx = idx % cols
        padded[r * h : (r + 1) * h, cidx * w : (cidx + 1) * w] = frames[idx]
    return padded


def bg_record_proc(
    obs_tensor: torch.Tensor,
    info_tensor: torch.Tensor,
    shutdown_event: threading.Event,
    games: Sequence[str],
    first_start_at: float,
) -> None:
    if cv2 is None:
        shutdown_event.wait()
        return

    window_name = "RL-Gamer Live"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    last_frame_time = 0.0
    target_interval = 1.0 / 30.0  # display at ~30 FPS irrespective of env FPS

    try:
        while not shutdown_event.is_set():
            now = time.time()
            if now - last_frame_time < target_interval:
                time.sleep(max(0.0, target_interval - (now - last_frame_time)))
                continue
            last_frame_time = now

            frames = obs_tensor.clone().cpu().numpy()
            mosaic = _tile_frames(frames, games)
            mosaic_bgr = cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR)
            cv2.imshow(window_name, mosaic_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q or ESC
                shutdown_event.set()
                break
    finally:
        cv2.destroyWindow(window_name)
