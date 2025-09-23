#!/usr/bin/env python3
import torch, gymnasium as gym, numpy as np, time, sys, threading, os, random
import torch.multiprocessing as mp
from torch import Tensor

from bg_record import log_step, bind_logger, log_close

# torch.set_num_threads(1)

NUM_PROCS = 16
FPS = 60.0
MAX_ACTIONS = 18
MAX_EPISODE_STEPS = int(45 * 60 * FPS)

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
print(f'{NUM_ENVS=}')

def env_thread_worker(first_start_at, game_id, g_idx, obs_s: Tensor, act_s: Tensor, info_s: Tensor, shutdown):
    import ale_py # required for atari
    next_frame_due = first_start_at + 15.0 # let all procs start
    env = gym.make(game_id, obs_type="rgb", frameskip=1, repeat_action_probability=0.0, full_action_space=True, max_episode_steps=MAX_EPISODE_STEPS)
    envseed = g_idx * 100 + int(os.environ['myseed'])
    print(f'{game_id=} {envseed=}')
    obs, _ = env.reset(seed=envseed)
    h, w, _ = obs.shape
    obs_s[g_idx, :h, :w].copy_(torch.from_numpy(obs), non_blocking=True)
    bind_logger(game_id, g_idx, info_s)
    while not shutdown.is_set():
        while time.time() > next_frame_due: next_frame_due += 1.0 / FPS
        time.sleep(max(0, next_frame_due - time.time()))
        action = act_s[g_idx].item()
        obs, rew, term, trunc, _ = env.step(action)
        log_step(action, obs, rew, term, trunc)
        obs_s[g_idx, :h, :w].copy_(torch.from_numpy(obs), non_blocking=True)
        if term or trunc:
            obs, _ = env.reset()
            obs_s[g_idx, :h, :w].copy_(torch.from_numpy(obs), non_blocking=True)
    log_close()
def seed(prefix, offset:int):
    s = int(os.environ['myseed']) + offset
    print(f'random seed: {prefix}: {s=}')
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    # torch.backends.cudnn.deterministic = True # Not worth it!
    # torch.backends.cudnn.benchmark = False # Not worth it!

def env_proc(first_start_at, game_chunk, offset, obs_s, act_s, info_s, shutdown):
    seed('env', offset + 1)
    threads = [threading.Thread(target=env_thread_worker, args=(first_start_at, g, offset+i, obs_s, act_s, info_s, shutdown))
               for i, g in enumerate(game_chunk)]
    for t in threads: t.start()
    for t in threads: t.join()

def agent_proc(obs_s, act_s, info_s, shutdown):
    seed('agent', 0)
    from myagent import Agent
    agent = Agent()

    save_path = "agent.pt"
    try: # only first load attempt allowed to fail
        print(f"loading from {save_path=}")
        agent.load(save_path)
    except Exception: pass
    print(f"saving to {save_path=}")
    agent.save(save_path)
    print(f"loading from {save_path=}")
    agent.load(save_path) # success required

    last_save_time = time.time()
    while not shutdown.is_set():
        # NOTE: THE AGENT IS CALLED IN A LOOP AS FAST AS POSSIBLE. THERE IS NO SLEEP STATEMENT IN THIS BLOCK.
        # (a very fast agent would do multiple passes per frame. a slow agent would take multiple frames to do a pass.)
        # NOTE: THE act_and_learn ARGUMENTS HAVE CHANGED
        # EACH ROW IN info_s IS LIKE (acc_reward, acc_frames, acc_term, acc_trunc)
        agent.act_and_learn(obs_s, info_s.clone(), act_s)
        if time.time() - last_save_time > 29*60:
            print(f"saving to {save_path=}")
            agent.save(save_path)
            print(f"loading from {save_path=}")
            agent.load(save_path)
            last_save_time = time.time()

if __name__ == "__main__":
    first_start_at = time.time()
    mp.set_start_method("forkserver", force=True)
    obs_s = torch.zeros((NUM_ENVS, 250, 160, 3), dtype=torch.uint8, device="cuda").share_memory_()
    act_s = torch.zeros(NUM_ENVS, dtype=torch.int64, device="cuda").share_memory_()
    info_s = torch.zeros((NUM_ENVS, 4), dtype=torch.float32, device="cuda").share_memory_()
    shutdown = mp.Event()

    proc_configs = [{'target': agent_proc, 'args': (obs_s, act_s, info_s, shutdown)}]
    game_chunks = np.array_split(games, NUM_PROCS)
    for i, chunk in enumerate(game_chunks):
        offset = sum(len(c) for c in game_chunks[:i])
        proc_configs.append({'target': env_proc, 'args': (first_start_at, chunk, offset, obs_s, act_s, info_s, shutdown)})

    # bg_record_proc(obs_s, shutdown, out_path="12x6_1080_30.mp4")
    from bg_record import bg_record_proc
    proc_configs.append({'target': bg_record_proc, 'args': (obs_s, info_s, shutdown, games, first_start_at)})

    procs = [mp.Process(**cfg) for cfg in proc_configs]

    for p in procs: p.start()
    try:
        duration = int(os.environ["RUNDURATIONSECONDS"])
        while time.time() - first_start_at < duration:
            time.sleep(15)
            for i, p in enumerate(procs):
                if not p.is_alive():
                    print("RIP SOMEONE CRASHED", file=sys.stderr)
                    sys.exit(1)
            sys.stdout.flush()
            sys.stderr.flush()
    except KeyboardInterrupt:
        print("\nShutdown signal received...")
    finally:
        shutdown.set()
        for p in procs: p.join(timeout=10)
        for p in procs:
            if p.is_alive(): p.terminate()
        print("All processes terminated.")

# agent ranking code has moved but essentially you want your top few episodes to score within 50% of the all-time record on each and every game. especially adventure, pong, pitfall, and skiing. nobody willing to face the pain with skiing... so much pain in that game lol



Let’s take this to the next level 