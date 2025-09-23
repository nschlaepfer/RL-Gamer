"""Minimal logging/recording stubs for gamer.py.

These functions satisfy the runner's imports without performing any IO. Replace
with real implementations if you want background video capture or detailed logs.
"""

from __future__ import annotations

import threading


_log_lock = threading.Lock()


def bind_logger(game_id: str, env_idx: int, info_tensor):
    # Populate the info tensor with zeroed stats for the environment slot.
    with _log_lock:
        info_tensor[env_idx].zero_()


def log_step(action, obs, reward, terminated, truncated):
    # Users should replace this with detailed logging if desired.
    pass


def log_close():
    pass


def bg_record_proc(obs_tensor, info_tensor, shutdown_event, games, first_start_at):
    # No-op recorder: waits for shutdown and exits.
    shutdown_event.wait()
