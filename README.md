# RL-Gamer Runner
<img width="752" height="1147" alt="Screenshot 2025-09-23 at 11 09 11 AM" src="https://github.com/user-attachments/assets/a3be822f-8fe4-4a61-8692-36046fcafae1" />



This project hosts a high-throughput Atari evaluation loop intended for large-scale
reinforcement-learning experiments. The default entry point `gamer.py` mirrors the
original CUDA runner (`og.py`) while staying cross-platform: shared environment
buffers live on CPU, and the agent process automatically selects CUDA, Apple MPS,
or CPU for inference. A modern PPO agent (IMPALA-style encoder + GAE) ships in
`myagent.py`, enabling training out of the box on powerful Apple Silicon machines
(e.g., M3 Max with 128 GB unified memory) or high-end CUDA desktops.

## Features
- Spawns one agent process plus a configurable set of environment worker processes
  hosting per-game threads for the full ALE benchmark suite.
- Keeps observation, action, and info tensors in shared CPU memory, enabling the
  agent to batch-copy to CUDA or MPS once per step.
- Supports optional action repeat and periodic FPS reporting for benchmarking.
- Integrates with the existing background recorder utilities (`bg_record`).

## Requirements
- Python 3.9 or newer (recommended: 3.11 on macOS for wheel availability).
- Recent PyTorch build with CUDA or MPS support (arm64 wheels include MPS).
- `gymnasium`, `ale-py`, `AutoROM`, and any dependencies required by your
  `bg_record` module (e.g., `ffmpeg-python`, `opencv-python-headless`).
- A custom `myagent.py` implementing the training/inference logic described below.

## Quick Start
```bash
# 0) Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# 1) Install runtime dependencies
pip install -r requirements.txt

# 2) Fetch Atari ROMs (accept the license once)
AutoROM --accept-license
```

If `AutoROM` emits `NotOpenSSLWarning` about LibreSSL, the download still
completes successfully on macOS; the warning is harmless.

### Live Viewer (optional)
The bundled `bg_record.py` module displays a real-time mosaic of all running
envs using OpenCV. Because the viewer creates GUI windows, you must install the
full `opencv-python` wheel (already included in `requirements.txt`). Launch the
runner from a macOS session with GUI access to see the window.

## Agent Architecture (`myagent.py`)
`myagent.py` contains a fully functional PPO agent that both `gamer.py` and
`og.py` import. Key traits:

- IMPALA-style convolutional encoder with residual blocks and adaptive pooling.
- PPO with GAE, multi-epoch minibatch updates, entropy/value regularization, and
  gradient clipping.
- AdamW optimizer tuned for high-throughput streaming of 64 ALE environments.
- Automatic device selection (CUDA, MPS, or CPU) with channels-last layout for
  convolutional efficiency.

The agent consumes the runner's shared tensors via `act_and_learn`:

- `obs`: `(N, 250, 160, 3)` uint8 frames batched across all envs.
- `info`: per-env scalars `(reward, terminated, truncated, episode_return)` from
  the most recent step.
- `act_buffer`: shared int tensor where actions are written in-place.

To substitute your own algorithm, keep the class signature (`Agent`) and the
`load/save/act_and_learn` methods, but replace the PPO internals with your model
and learning loop.

## Configuration
`gamer.py` reads several environment variables at launch:

| Variable | Default | Purpose |
|----------|---------|---------|
| `myseed` | `0` | Base seed; env workers add offsets per game. |
| `RUNDURATIONSECONDS` | `1800` | Target wall-clock duration before graceful shutdown. |
| `NUM_PROCS` | `16` | Maximum number of env processes; each hosts a chunk of games. |
| `ACTION_REPEAT` | `1` | Repeat each chosen action this many frames (reduces agent load). |
| `STATS_INTERVAL` | `10` | Seconds between FPS reports logged by the agent process. |
| `PYTORCH_ENABLE_MPS_FALLBACK` | `1` | Allow CPU fallbacks for unsupported MPS ops. |
| `ROLLOUT_STEPS` | `128` | Number of timesteps collected before each PPO update. |
| `RL_LR` | `3e-4` | PPO learning rate (set via AdamW). |
| `RL_PPO_EPOCHS` | `4` | Policy/value passes per update. |
| `RL_MINIBATCH` | `2048` | Minibatch size for PPO updates. |
| `RL_ENTROPY_COEF` | `0.01` | Entropy bonus coefficient. |
| `RL_VALUE_COEF` | `0.5` | Value loss coefficient. |

All shared tensors use CPU memory, so you can safely tweak these values without
worrying about GPU memory exhaustion. When `ACTION_REPEAT > 1`, the env threads
reuse the previous action for the specified frames, enabling the agent to operate
at a lower frequency while maintaining a 60 Hz recording stream.

## Running the Loop
```bash
source .venv/bin/activate
export myseed=0
export RUNDURATIONSECONDS=1800
export NUM_PROCS=12 ACTION_REPEAT=2 STATS_INTERVAL=20  # example benchmarking setup
# PPO tuning knobs (override defaults as needed)
export ROLLOUT_STEPS=256 RL_LR=2.5e-4 RL_PPO_EPOCHS=3
python3 gamer.py
```

Console output highlights:
- `[runner] agent device=...` shows whether CUDA, MPS, or CPU is active.
- `[perf] aggregated_env_fps=...` lines report throughput across all games every
  `STATS_INTERVAL` seconds.
- Env threads print their game IDs and seeds as they initialize.
- `[ppo] update=...` lines summarize loss statistics each time PPO finishes an
  optimization cycle.

Interrupt with `Ctrl+C`; the script sets a shutdown event, joins workers, and
terminates any stragglers after a short timeout. If any child process crashes,
`gamer.py` exits early with `RIP SOMEONE CRASHED`.

## Benchmarking & Tuning
1. Start with `NUM_PROCS` equal to your performance-core count (e.g., 12 on an
   M3 Max). Increase until the `[perf]` FPS stops improving.
2. Adjust `ACTION_REPEAT` (2–4) if the agent cannot keep up with 60 Hz envs.
3. Keep `myseed` fixed for comparable runs. Record `[perf]` metrics after each
   change to track progress.
4. Monitor `[ppo]` logs; if policy loss oscillates wildly, lower `RL_LR` or
   reduce `ROLLOUT_STEPS`. If updates feel sluggish, raise `ROLLOUT_STEPS` or
   increase `NUM_PROCS` to collect experience faster.
5. Profile the agent with `torch.profiler` or macOS Instruments if MPS becomes
   the bottleneck; ensure data format/dtype conversions happen once per batch.

## Background Recording & Live Viewer
`bg_record.py` now implements a live OpenCV viewer. It tiles the shared RGB frames
into a mosaic window (`RL-Gamer Live`) at ~30 FPS and honours the shutdown event.
Close the window or press `q`/`Esc` to stop the session. The same API can be
extended to capture video—swap in an ffmpeg pipeline if you want to record runs.

If you prefer headless operation, replace the module with a no-op variant or add
your own ffmpeg-based recorder. Ensure encoders can consume CPU RGB frames; on
macOS, VideoToolbox (`h264_videotoolbox`) provides hardware acceleration. Install
`ffmpeg` separately if required:
```bash
brew install ffmpeg
```

## Troubleshooting
- **`ModuleNotFoundError: No module named 'myagent'`**: Add a `myagent.py` file
  with the API described above or adjust `sys.path` to locate your agent package.
- **`AutoROM` CLI errors**: Invoke the standalone entry point (`AutoROM --accept-license`) rather than `python -m AutoROM`. LibreSSL warnings are cosmetic.
- **Slow FPS on MPS**: Verify the agent uses channels-last FP16, and consider
  increasing `ACTION_REPEAT` temporarily to isolate model bottlenecks.
- **Crash in a worker**: Check stderr for the first stack trace. The launcher will
  terminate all processes and exit with status 1 to avoid hanging runs.

## Repository Layout
- `gamer.py` – Cross-platform runner with benchmarking hooks.
- `og.py` – Original CUDA-centric runner kept for reference.
- `myagent.py` – PPO agent used by both runners.
- `bg_record.py` – Live OpenCV viewer (replace if you need recording only).
- `requirements.txt` – Python dependencies for quick setup.
- `README.md` – This guide.

## License
No license information is provided. Add a LICENSE file if you plan to distribute
this project.
