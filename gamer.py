#!/usr/bin/env python3
"""Cross-platform runner for the Atari training loop.

This version keeps all shared buffers on CPU so it works on Windows, Linux,
and macOS, while still letting the agent execute on CUDA or MPS when
available. The overall topology matches the original CUDA runner: one agent
process plus multiple environment worker processes hosting per-game threads.
"""

from __future__ import annotations

import os
import random
import sys
import threading
import time
from typing import Iterable

import gymnasium as gym
import numpy as np
import torch
import torch.multiprocessing as mp
from torch import Tensor

from bg_record import bind_logger, bg_record_proc, log_close, log_step

# -----------------------------------------------------------------------------
# Global configuration
# -----------------------------------------------------------------------------
NUM_PROCS = int(os.environ.get("NUM_PROCS", "16"))
FPS = 60.0
MAX_ACTIONS = 18
MAX_EPISODE_STEPS = int(45 * 60 * FPS)
ACTION_REPEAT = max(1, int(os.environ.get("ACTION_REPEAT", "1")))
STATS_INTERVAL = float(os.environ.get("STATS_INTERVAL", "10"))

os.environ.setdefault("myseed", "0")
os.environ.setdefault("RUNDURATIONSECONDS", "1800")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

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
                print(
                    f"[warn] action {current_action} out of bounds (size={action_bound})",
                    f"raw={raw_action} env={game_id} idx={g_idx}",
                )
                current_action = max(0, min(action_bound - 1, current_action))

        obs, rew, term, trunc, _ = env.step(current_action)
        log_step(current_action, obs, rew, term, trunc)
        obs_s[g_idx, :h, :w].copy_(torch.from_numpy(obs))
        frame_ctr[g_idx].add_(1)
        episode_return += float(rew)

        info_row = info_s[g_idx]
        info_row[0] = float(rew)
        info_row[1] = float(term)
        info_row[2] = float(trunc)
        info_row[3] = float(episode_return)

        repeat_remaining -= 1

        if term or trunc:
            obs, _ = env.reset()
            obs_s[g_idx, :h, :w].copy_(torch.from_numpy(obs))
            frame_ctr[g_idx].add_(1)
            repeat_remaining = 0
            episode_return = 0.0
            info_s[g_idx].zero_()

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
) -> None:
    seed("env", offset + 1)
    threads = [
        threading.Thread(
            target=env_thread_worker,
            args=(first_start_at, game, offset + i, obs_s, act_s, info_s, frame_ctr, shutdown),
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
def agent_proc(obs_s: Tensor, act_s: Tensor, info_s: Tensor, frame_ctr: Tensor, shutdown: mp.Event) -> None:
    seed("agent", 0)
    from myagent import Agent

    agent = Agent()

    save_path = "agent.pt"
    try:
        print(f"loading from {save_path=}")
        agent.load(save_path)
    except Exception:
        pass
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

        maybe_actions = agent.act_and_learn(obs_dev, info_dev.clone(), act_dev)
        actions_dev = act_dev if maybe_actions is None else maybe_actions

        act_s.copy_(actions_dev.to(device="cpu", dtype=act_s.dtype))

        if time.time() - last_save_time > 29 * 60:
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
def main() -> None:
    first_start_at = time.time()
    mp.set_start_method("spawn", force=True)

    obs_s = torch.empty((NUM_ENVS, 250, 160, 3), dtype=torch.uint8).share_memory_()
    act_s = torch.zeros(NUM_ENVS, dtype=torch.int16).share_memory_()
    info_s = torch.zeros((NUM_ENVS, 4), dtype=torch.float32).share_memory_()
    frame_ctr = torch.zeros(NUM_ENVS, dtype=torch.int32).share_memory_()

    shutdown = mp.Event()

    proc_configs = [{"target": agent_proc, "args": (obs_s, act_s, info_s, frame_ctr, shutdown)}]
    game_chunks = np.array_split(games, min(NUM_PROCS, NUM_ENVS))
    for idx, chunk in enumerate(game_chunks):
        offset = sum(len(c) for c in game_chunks[:idx])
        proc_configs.append(
            {
                "target": env_proc,
                "args": (first_start_at, chunk.tolist(), offset, obs_s, act_s, info_s, frame_ctr, shutdown),
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
