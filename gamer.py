#!/usr/bin/env python3
"""Cross-platform runner for the Atari training loop.

This version keeps all shared buffers on CPU so it works on Windows, Linux,
and macOS, while still letting the agent execute on CUDA or MPS when
available. The overall topology matches the original CUDA runner: one agent
process plus multiple environment worker processes hosting per-game threads.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.multiprocessing as mp
from torch import Tensor

import yaml

from bg_record import bind_logger, bg_record_proc, log_close, log_step

# -----------------------------------------------------------------------------
# Global configuration
# -----------------------------------------------------------------------------

os.environ.setdefault("myseed", "0")
os.environ.setdefault("RUNDURATIONSECONDS", "1800")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

NUM_PROCS = 16
FPS = 60.0
MAX_ACTIONS = 18
MAX_EPISODE_STEPS = int(45 * 60 * FPS)
ACTION_REPEAT = 1
STATS_INTERVAL = 10.0


@dataclass
class EpisodeTracker:
    episodes: int = 0
    total_return: float = 0.0
    best_return: float = float("-inf")
    total_steps: int = 0
    best_steps: int = 0
    last_info: dict[str, Any] = field(default_factory=dict)
    best_info: dict[str, Any] = field(default_factory=dict)

    _priority_keys = (
        "level",
        "stage",
        "world",
        "room",
        "area",
        "phase",
        "lives",
        "ale.lives",
        "ale.frame_number",
        "frame_number",
        "score",
    )

    def record(self, episode_return: float, steps: int, info: Any) -> None:
        filtered = self._filter_info(info)
        self.episodes += 1
        self.total_return += episode_return
        self.total_steps += steps
        self.last_info = filtered

        if episode_return > self.best_return:
            self.best_return = episode_return
            self.best_steps = steps
            self.best_info = filtered

    def episode_log(self, game_id: str, env_idx: int, episode_return: float, steps: int, outcome: str) -> str:
        avg_return = self.total_return / max(self.episodes, 1)
        avg_steps = self.total_steps / max(self.episodes, 1)

        parts = [
            "[episode]",
            f"game_id={game_id}",
            f"env_index={env_idx}",
            f"episodes={self.episodes}",
            f"return={episode_return:.2f}",
            f"avg_return={avg_return:.2f}",
            f"best_return={self.best_return:.2f}",
            f"steps={steps}",
            f"avg_steps={avg_steps:.1f}",
            f"outcome={outcome}",
        ]

        info_desc = self._format_info(self.last_info)
        if info_desc:
            parts.append(f"info[{info_desc}]")

        best_info_desc = self._format_info(self.best_info)
        if self.best_return == episode_return and best_info_desc:
            parts.append(f"new_best[{best_info_desc}]")
        elif best_info_desc:
            parts.append(f"best_info[{best_info_desc}]")

        return " ".join(parts)

    def summary_log(self, game_id: str, env_idx: int) -> Optional[str]:
        if self.episodes == 0:
            return None

        avg_return = self.total_return / self.episodes
        avg_steps = self.total_steps / self.episodes
        best_info_desc = self._format_info(self.best_info)

        parts = [
            "[summary]",
            f"game_id={game_id}",
            f"env_index={env_idx}",
            f"episodes={self.episodes}",
            f"avg_return={avg_return:.2f}",
            f"best_return={self.best_return:.2f}",
            f"avg_steps={avg_steps:.1f}",
            f"best_steps={self.best_steps}",
        ]
        if best_info_desc:
            parts.append(f"best_info[{best_info_desc}]")
        return " ".join(parts)

    def _filter_info(self, info: Any) -> dict[str, Any]:
        if not isinstance(info, Mapping):
            return {}
        filtered: dict[str, Any] = {}
        for key, value in info.items():
            if isinstance(value, (int, float, str)):
                filtered[str(key)] = value
        return filtered

    def _format_info(self, info: Mapping[str, Any]) -> str:
        if not info:
            return ""

        parts: list[str] = []
        added_keys: set[str] = set()
        for key in self._priority_keys:
            if key in info:
                parts.append(f"{key}={info[key]}")
                added_keys.add(key)

        if len(parts) < 4:
            for key in sorted(info.keys()):
                if key in added_keys:
                    continue
                value = info[key]
                if isinstance(value, (int, float, str)):
                    parts.append(f"{key}={value}")
                    added_keys.add(key)
                if len(parts) >= 4:
                    break

        return " ".join(parts)

games = sorted([
    "ALE/Adventure-v5", "ALE/AirRaid-v5", "ALE/Alien-v5", "ALE/Amidar-v5", "ALE/Assault-v5",
    "ALE/Asterix-v5", "ALE/Asteroids-v5", "ALE/Atlantis-v5", "ALE/BankHeist-v5",
    "ALE/BattleZone-v5", "ALE/BeamRider-v5", "ALE/Berzerk-v5", "ALE/Bowling-v5",
    "ALE/Boxing-v5", "ALE/Breakout-v5", "ALE/Carnival-v5", "ALE/Centipede-v5",
    "ALE/ChopperCommand-v5", "ALE/CrazyClimber-v5", "ALE/Defender-v5", "ALE/DemonAttack-v5",
    "ALE/DoubleDunk-v5", "ALE/ElevatorAction-v5", "ALE/Enduro-v5", "ALE/FishingDerby-v5",
    "ALE/Freeway-v5", "ALE/Frostbite-v5", "ALE/Gopher-v5", "ALE/Gravitar-v5", "ALE/Hero-v5",
    "ALE/IceHockey-v5", "ALE/Jamesbond-v5", "ALE/JourneyEscape-v5", "ALE/Kangaroo-v5",
    "ALE/KeystoneKapers-v5", "ALE/KingKong-v5", "ALE/Krull-v5", "ALE/KungFuMaster-v5",
    "ALE/MontezumaRevenge-v5", "ALE/MsPacman-v5", "ALE/NameThisGame-v5", "ALE/Phoenix-v5",
    "ALE/Pitfall-v5", "ALE/Pong-v5", "ALE/Pooyan-v5", "ALE/PrivateEye-v5", "ALE/Qbert-v5",
    "ALE/Riverraid-v5", "ALE/RoadRunner-v5", "ALE/Robotank-v5", "ALE/Seaquest-v5",
    "ALE/Skiing-v5", "ALE/Solaris-v5", "ALE/SpaceInvaders-v5", "ALE/StarGunner-v5",
    "ALE/Tennis-v5", "ALE/TimePilot-v5", "ALE/Tutankham-v5", "ALE/UpNDown-v5",
    "ALE/Venture-v5", "ALE/VideoPinball-v5", "ALE/WizardOfWor-v5", "ALE/YarsRevenge-v5",
    "ALE/Zaxxon-v5"
])
NUM_ENVS = len(games)
print(f"{NUM_ENVS=}")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def refresh_runtime_config() -> None:
    global NUM_PROCS, ACTION_REPEAT, STATS_INTERVAL
    NUM_PROCS = int(os.environ.get("NUM_PROCS", "16"))
    ACTION_REPEAT = max(1, int(os.environ.get("ACTION_REPEAT", "1")))
    STATS_INTERVAL = float(os.environ.get("STATS_INTERVAL", "10"))


def load_config(path: str) -> None:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Config file {path} must contain a mapping at the top level.")

    sections: list[Mapping] = []
    for key in ("env", "runner", "agent"):
        section = data.get(key)
        if section is not None:
            if not isinstance(section, Mapping):
                raise ValueError(f"Config section '{key}' must be a mapping.")
            sections.append(section)

    top_level_scalars = {
        key: value
        for key, value in data.items()
        if not isinstance(value, Mapping)
    }
    if top_level_scalars:
        sections.append(top_level_scalars)

    for section in sections:
        for key, value in section.items():
            if value is None:
                continue
            os.environ[str(key)] = str(value)

    print(f"[config] loaded overrides from {path}")
    refresh_runtime_config()


refresh_runtime_config()

def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed(prefix: str, offset: int) -> None:
    s = int(os.environ["myseed"]) + offset
    print(f"random seed: {prefix}: s={s}")
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# -----------------------------------------------------------------------------
# Environment worker logic
# -----------------------------------------------------------------------------
def env_thread_worker(
    first_start_at: float,
    game_id: str,
    g_idx: int,
    obs_s: Tensor,
    act_s: Tensor,
    info_s: Tensor,
    frame_ctr: Tensor,
    shutdown: mp.Event,
    log_returns: bool,
) -> None:
    import ale_py  # type: ignore  # noqa: F401 (import side-effect for Atari ROMs)

    next_frame_due = first_start_at + 15.0  # give every process time to spin up
    env = gym.make(
        game_id,
        obs_type="rgb",
        frameskip=1,
        repeat_action_probability=0.0,
        full_action_space=True,
        max_episode_steps=MAX_EPISODE_STEPS,
    )
    envseed = g_idx * 100 + int(os.environ["myseed"])
    print(f"{game_id=} {envseed=}")
    obs, _ = env.reset(seed=envseed)
    h, w, _ = obs.shape
    obs_s[g_idx, :h, :w].copy_(torch.from_numpy(obs))
    info_s[g_idx].zero_()
    bind_logger(game_id, g_idx, info_s)

    current_action = 0
    raw_action = 0
    repeat_remaining = 0
    episode_return = 0.0
    episode_steps = 0
    tracker: Optional[EpisodeTracker] = EpisodeTracker() if log_returns else None

    warned_out_of_bounds = False

    while not shutdown.is_set():
        while time.time() > next_frame_due:
            next_frame_due += 1.0 / FPS
        time.sleep(max(0.0, next_frame_due - time.time()))

        if repeat_remaining <= 0:
            raw_action = int(act_s[g_idx].item())
            # Guard against agents returning out-of-range actions.
            current_action = max(0, min(MAX_ACTIONS - 1, raw_action))
            repeat_remaining = ACTION_REPEAT

        # Extra safety: ALE environments expect indices into their action set.
        action_set = getattr(env.unwrapped, "_action_set", None)
        if action_set is not None:
            action_bound = len(action_set)
            if not 0 <= current_action < action_bound:
                if not warned_out_of_bounds:
                    print(
                        f"[warn] clamping actions for {game_id}: agent produced {raw_action}"
                        f" but action set size is {action_bound}",
                    )
                    warned_out_of_bounds = True
                current_action = max(0, min(action_bound - 1, current_action))

        obs, rew, term, trunc, info = env.step(current_action)
        log_step(current_action, obs, rew, term, trunc)
        obs_s[g_idx, :h, :w].copy_(torch.from_numpy(obs))
        frame_ctr[g_idx].add_(1)
        episode_return += float(rew)
        episode_steps += 1

        info_row = info_s[g_idx]
        info_row[0] = float(rew)
        info_row[1] = float(term)
        info_row[2] = float(trunc)
        info_row[3] = float(episode_return)

        repeat_remaining -= 1

        if term or trunc:
            if tracker is not None:
                tracker.record(episode_return, episode_steps, info)
                outcome = "terminated" if term else "truncated"
                print(tracker.episode_log(game_id, g_idx, episode_return, episode_steps, outcome))
            obs, _ = env.reset()
            obs_s[g_idx, :h, :w].copy_(torch.from_numpy(obs))
            frame_ctr[g_idx].add_(1)
            repeat_remaining = 0
            episode_return = 0.0
            episode_steps = 0
            info_s[g_idx].zero_()

    if tracker is not None:
        summary = tracker.summary_log(game_id, g_idx)
        if summary:
            print(summary)

    log_close()


def env_proc(
    first_start_at: float,
    game_chunk: Iterable[str],
    offset: int,
    obs_s: Tensor,
    act_s: Tensor,
    info_s: Tensor,
    frame_ctr: Tensor,
    shutdown: mp.Event,
    log_returns: bool,
) -> None:
    seed("env", offset + 1)
    threads = [
        threading.Thread(
            target=env_thread_worker,
            args=(first_start_at, game, offset + i, obs_s, act_s, info_s, frame_ctr, shutdown, log_returns),
            daemon=True,
        )
        for i, game in enumerate(game_chunk)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


# -----------------------------------------------------------------------------
# Agent process
# -----------------------------------------------------------------------------
def agent_proc(
    obs_s: Tensor,
    act_s: Tensor,
    info_s: Tensor,
    frame_ctr: Tensor,
    shutdown: mp.Event,
    eval_mode: bool = False,
) -> None:
    seed("agent", 0)
    from myagent import Agent

    agent = Agent()

    save_path = "agent.pt"
    if os.path.exists(save_path):
        print(f"loading from {save_path=}")
        agent.load(save_path)
    else:
        print(f"[runner] no checkpoint found at {save_path}, starting from fresh weights")

    if not eval_mode:
        print(f"saving to {save_path=}")
        agent.save(save_path)
        print(f"loading from {save_path=}")
        agent.load(save_path)

    dev = device()
    use_async = dev.type in {"cuda", "mps"}

    print(f"[runner] agent device={dev} (async copies={'yes' if use_async else 'no'})")
    if dev.type == "mps":
        print("[runner] MPS detected: prefer channels-last FP16 inside Agent to maximize throughput.")

    obs_dev = torch.empty((NUM_ENVS, 250, 160, 3), dtype=torch.uint8, device=dev)
    info_dev = torch.empty((NUM_ENVS, 4), dtype=torch.float32, device=dev)
    act_dev = torch.empty((NUM_ENVS,), dtype=torch.int64, device=dev)

    last_seen = torch.zeros(NUM_ENVS, dtype=torch.int32)
    last_save_time = time.time()
    last_report_time = time.time()
    last_report_frames = 0

    while not shutdown.is_set():
        cur_ctr = frame_ctr.clone()
        if torch.equal(cur_ctr, last_seen):
            time.sleep(0.0005)
            continue

        obs_dev.copy_(obs_s, non_blocking=use_async)
        info_dev.copy_(info_s, non_blocking=use_async)

        maybe_actions = agent.act_and_learn(obs_dev, info_dev.clone(), act_dev, train=not eval_mode)
        actions_dev = act_dev if maybe_actions is None else maybe_actions

        act_s.copy_(actions_dev.to(device="cpu", dtype=act_s.dtype))

        if not eval_mode and time.time() - last_save_time > 29 * 60:
            print(f"saving to {save_path=}")
            agent.save(save_path)
            print(f"loading from {save_path=}")
            agent.load(save_path)
            last_save_time = time.time()

        last_seen.copy_(cur_ctr)

        now = time.time()
        if now - last_report_time >= STATS_INTERVAL:
            total_frames = int(cur_ctr.sum().item())
            delta_frames = total_frames - last_report_frames
            delta_t = now - last_report_time
            if delta_t > 0.0:
                fps = delta_frames / delta_t
                print(
                    f"[perf] aggregated_env_fps={fps:.1f} over {delta_frames} frames "
                    f"(interval={delta_t:.1f}s)"
                )
            last_report_time = now
            last_report_frames = total_frames


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the RL-Gamer training loop")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to a YAML file containing environment variable overrides",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run the agent in evaluation-only mode (no learning or checkpoint writes)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.config:
        load_config(args.config)
    else:
        refresh_runtime_config()

    eval_mode = bool(args.eval)

    log_returns = bool(eval_mode)

    first_start_at = time.time()
    mp.set_start_method("spawn", force=True)

    obs_s = torch.empty((NUM_ENVS, 250, 160, 3), dtype=torch.uint8).share_memory_()
    act_s = torch.zeros(NUM_ENVS, dtype=torch.int16).share_memory_()
    info_s = torch.zeros((NUM_ENVS, 4), dtype=torch.float32).share_memory_()
    frame_ctr = torch.zeros(NUM_ENVS, dtype=torch.int32).share_memory_()

    shutdown = mp.Event()

    proc_configs = [{"target": agent_proc, "args": (obs_s, act_s, info_s, frame_ctr, shutdown, eval_mode)}]
    game_chunks = np.array_split(games, min(NUM_PROCS, NUM_ENVS))
    for idx, chunk in enumerate(game_chunks):
        offset = sum(len(c) for c in game_chunks[:idx])
        proc_configs.append(
            {
                "target": env_proc,
                "args": (
                    first_start_at,
                    chunk.tolist(),
                    offset,
                    obs_s,
                    act_s,
                    info_s,
                    frame_ctr,
                    shutdown,
                    log_returns,
                ),
            }
        )

    proc_configs.append(
        {"target": bg_record_proc, "args": (obs_s, info_s, shutdown, games, first_start_at)}
    )

    processes = [mp.Process(**cfg) for cfg in proc_configs]

    for proc in processes:
        proc.start()

    try:
        duration = int(os.environ["RUNDURATIONSECONDS"])
        while time.time() - first_start_at < duration:
            time.sleep(15)
            for proc in processes:
                if not proc.is_alive():
                    print("RIP SOMEONE CRASHED", file=sys.stderr)
                    shutdown.set()
                    raise SystemExit(1)
            sys.stdout.flush()
            sys.stderr.flush()
    except KeyboardInterrupt:
        print("\nShutdown signal received...")
    finally:
        shutdown.set()
        for proc in processes:
            proc.join(timeout=10)
        for proc in processes:
            if proc.is_alive():
                proc.terminate()
        print("All processes terminated.")


if __name__ == "__main__":
    main()
