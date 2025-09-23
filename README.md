# RL-Gamer Runner

This project hosts a high-throughput Atari evaluation loop intended for large-scale
reinforcement-learning experiments. The default entry point `gamer.py` mirrors the
original CUDA runner (`og.py`) while staying cross-platform: shared environment
buffers live on CPU, and the agent process automatically selects CUDA, Apple MPS,
or CPU for inference. The script also includes built-in performance telemetry so
you can benchmark different process counts and action-repeat settings on powerful
Apple Silicon machines (e.g., M3 Max with 128 GB unified memory) or high-end PCs.

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

# 1) Install PyTorch (CUDA/MPS support included in official wheels)
pip install torch torchvision torchaudio

# 2) Install Gymnasium, ALE interface, ROM fetcher, and recorder deps
pip install gymnasium ale-py AutoROM ffmpeg-python opencv-python-headless

# 3) Download Atari ROMs (accept the license once)
AutoROM --accept-license
```

If `AutoROM` emits `NotOpenSSLWarning` about LibreSSL, the download still
completes successfully on macOS; the warning is harmless.

## Agent Contract (`myagent.py`)
Create a `myagent.py` file alongside `gamer.py` exposing an `Agent` class with:

- `load(path: str) -> None`: Restore weights/optimizer state.
- `save(path: str) -> None`: Persist current state.
- `act_and_learn(obs: Tensor, info: Tensor, act_buffer: Tensor) -> Optional[Tensor]`:
  Consume the latest batch of observations and info, update internal state, and
  either write discrete actions into `act_buffer` (in-place) or return an action
  tensor. Actions must be in `[0, MAX_ACTIONS)`; the runner clamps anything
  outside that range as a last resort but agents should enforce it themselves.

**MPS Optimization Tips**
- Convert the uint8 observations to channels-last FP16 before the model forward:
  ```python
  obs = obs_tensor.permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
  obs = obs.to(device="mps", dtype=torch.float16) / 255.0
  model = model.to(device="mps", dtype=torch.float16, memory_format=torch.channels_last)
  ```
- Run once with `PYTORCH_ENABLE_MPS_FALLBACK=0` to surface unsupported ops, then
  revert to `1` (the default) so PyTorch can fall back to CPU if needed.

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
python3 gamer.py
```

Console output highlights:
- `[runner] agent device=...` shows whether CUDA, MPS, or CPU is active.
- `[perf] aggregated_env_fps=...` lines report throughput across all games every
  `STATS_INTERVAL` seconds.
- Env threads print their game IDs and seeds as they initialize.

Interrupt with `Ctrl+C`; the script sets a shutdown event, joins workers, and
terminates any stragglers after a short timeout. If any child process crashes,
`gamer.py` exits early with `RIP SOMEONE CRASHED`.

## Benchmarking & Tuning
1. Start with `NUM_PROCS` equal to your performance-core count (e.g., 12 on an
   M3 Max). Increase until the `[perf]` FPS stops improving.
2. Adjust `ACTION_REPEAT` (2–4) if the agent cannot keep up with 60 Hz envs.
3. Keep `myseed` fixed for comparable runs. Record `[perf]` metrics after each
   change to track progress.
4. Profile the agent with `torch.profiler` or macOS Instruments if MPS becomes
   the bottleneck; ensure data format/dtype conversions happen once per batch.

## Background Recording
The repository now includes a minimal `bg_record.py` stub that satisfies the
runner's imports without writing video or logs. It zeroes the shared info tensor
and exits when the shutdown event fires. If you want full recording support,
replace the stub with your own implementation that provides:

- `bind_logger(game_id, env_idx, info_tensor)`
- `log_step(action, obs, reward, terminated, truncated)`
- `log_close()`
- `bg_record_proc(obs_tensor, info_tensor, shutdown_event, games, first_start_at)`

Ensure your recorder can consume CPU RGB frames. On macOS, prefer VideoToolbox
encoders (e.g., `h264_videotoolbox`) for hardware acceleration. Install `ffmpeg`
separately if required:
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
- `README.md` – This guide.
- (expected) `bg_record.py`, `myagent.py`, and any supporting modules.

## License
No license information is provided. Add a LICENSE file if you plan to distribute
this project.
