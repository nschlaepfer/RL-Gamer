#!/usr/bin/env python3
"""Reference Agent module shared by ``gamer.py`` and ``og.py``.

This implementation provides a scalable on-policy learner based on PPO with
IMPALA-style convolutional feature extraction. It is designed to exploit modern
hardware (CUDA or Apple MPS) while remaining compatible with the runner's
multiprocess layout. The agent learns directly from the batched Atari frames the
runner streams in real time and performs gradient updates once a configurable
rollout horizon is collected.

To plug in your own algorithm, replace the network or the ``PPOAgent`` class,
but the provided implementation is fully functional and ready for large-scale
runs on machines with ample unified memory (e.g., 128 GB Apple Silicon).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

MAX_ACTIONS = 18  # keep in sync with gamer.py


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class AgentConfig:
    num_actions: int = MAX_ACTIONS
    obs_size: int = int(os.environ.get("AGENT_OBS_SIZE", "128"))
    rollout_length: int = int(os.environ.get("ROLLOUT_STEPS", "128"))
    gamma: float = float(os.environ.get("RL_GAMMA", "0.99"))
    gae_lambda: float = float(os.environ.get("RL_GAE_LAMBDA", "0.95"))
    learning_rate: float = float(os.environ.get("RL_LR", "3e-4"))
    eps: float = float(os.environ.get("RL_ADAM_EPS", "1e-5"))
    ppo_clip: float = float(os.environ.get("RL_PPO_CLIP", "0.2"))
    ppo_epochs: int = int(os.environ.get("RL_PPO_EPOCHS", "4"))
    minibatch_size: int = int(os.environ.get("RL_MINIBATCH", "2048"))
    entropy_coef: float = float(os.environ.get("RL_ENTROPY_COEF", "0.01"))
    value_coef: float = float(os.environ.get("RL_VALUE_COEF", "0.5"))
    max_grad_norm: float = float(os.environ.get("RL_MAX_GRAD_NORM", "0.5"))
    channels_last: bool = True
    storage_dtype: torch.dtype = torch.float16


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.activation(self.conv1(x))
        x = self.conv2(x)
        return self.activation(x + residual)


class ImpalaBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.conv(x))
        x = F.avg_pool2d(x, kernel_size=2)
        x = self.res1(x)
        x = self.res2(x)
        return x


class ImpalaEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, depths: Optional[List[int]] = None) -> None:
        super().__init__()
        depths = depths or [32, 64, 64]
        layers: List[nn.Module] = []
        prev = in_channels
        for depth in depths:
            layers.append(ImpalaBlock(prev, depth))
            prev = depth
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ActorCritic(nn.Module):
    def __init__(self, num_actions: int, input_shape: torch.Size, hidden_dim: int = 512) -> None:
        super().__init__()
        self.encoder = ImpalaEncoder(in_channels=input_shape[0])
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            encoded = self.encoder(dummy)
            flat_dim = encoded.numel()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.policy = nn.Linear(hidden_dim, num_actions)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.encoder(x)
        feats = self.fc(feats)
        logits = self.policy(feats)
        value = self.value(feats)
        return logits, value


class RolloutBuffer:
    def __init__(self, storage_dtype: torch.dtype = torch.float16) -> None:
        self.storage_dtype = storage_dtype
        self.obs: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.logprobs: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.dones: List[torch.Tensor] = []

    def add(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        logprobs: torch.Tensor,
        values: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        self.obs.append(obs.detach().to("cpu", dtype=self.storage_dtype))
        self.actions.append(actions.detach().to("cpu", dtype=torch.long))
        self.logprobs.append(logprobs.detach().to("cpu", dtype=torch.float32))
        self.values.append(values.detach().to("cpu", dtype=torch.float32))
        self.rewards.append(rewards.detach().to("cpu", dtype=torch.float32))
        self.dones.append(dones.detach().to("cpu", dtype=torch.float32))

    def clear(self) -> None:
        self.obs.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()

    def __len__(self) -> int:
        return len(self.rewards)


class PPOAgent:
    def __init__(self, config: Optional[AgentConfig] = None) -> None:
        self.cfg = config or AgentConfig()
        self.device = _select_device()
        self.obs_shape = (3, self.cfg.obs_size, self.cfg.obs_size)

        self.model = ActorCritic(self.cfg.num_actions, torch.Size(self.obs_shape))
        self.model.to(self.device, dtype=torch.float32)
        if self.cfg.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
        self.model.eval()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.learning_rate,
            eps=self.cfg.eps,
            weight_decay=0.0,
        )

        self.rollout = RolloutBuffer(storage_dtype=self.cfg.storage_dtype)

        self.last_obs: Optional[torch.Tensor] = None
        self.last_action: Optional[torch.Tensor] = None
        self.last_logprob: Optional[torch.Tensor] = None
        self.last_value: Optional[torch.Tensor] = None
        self.update_step = 0
        self.training_start_time = time.time()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def load(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location=self.device)
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload.get("optimizer", {}))
        self.update_step = int(payload.get("update_step", 0))

    def save(self, path: str) -> None:
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update_step": self.update_step,
        }
        torch.save(payload, path)

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------
    def _prepare_obs(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.to(device=self.device, dtype=torch.float32)
        obs = obs.permute(0, 3, 1, 2) / 255.0
        obs = F.interpolate(obs, size=(self.cfg.obs_size, self.cfg.obs_size), mode="bilinear", align_corners=False)
        if self.cfg.channels_last:
            obs = obs.contiguous(memory_format=torch.channels_last)
        return obs

    def _maybe_update(self, bootstrap_value: torch.Tensor) -> None:
        rollout_len = len(self.rollout)
        if rollout_len < self.cfg.rollout_length:
            return

        obs_tensor = torch.stack(self.rollout.obs, dim=0).to(self.device, dtype=torch.float32)
        obs_tensor = obs_tensor.contiguous()
        actions_tensor = torch.stack(self.rollout.actions, dim=0).to(self.device)
        logprob_tensor = torch.stack(self.rollout.logprobs, dim=0).to(self.device)
        value_tensor = torch.stack(self.rollout.values, dim=0).to(self.device, dtype=torch.float32)
        reward_tensor = torch.stack(self.rollout.rewards, dim=0).to(self.device, dtype=torch.float32)
        done_tensor = torch.stack(self.rollout.dones, dim=0).to(self.device, dtype=torch.float32)
        bootstrap_value = bootstrap_value.detach().to(self.device, dtype=torch.float32)

        T, N = reward_tensor.shape

        advantages = torch.zeros((T, N), device=self.device, dtype=torch.float32)
        gae = torch.zeros(N, device=self.device, dtype=torch.float32)
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = bootstrap_value
            else:
                next_value = value_tensor[t + 1]
            mask = 1.0 - done_tensor[t]
            delta = reward_tensor[t] + self.cfg.gamma * next_value * mask - value_tensor[t]
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * mask * gae
            advantages[t] = gae
        returns = advantages + value_tensor

        obs_flat = obs_tensor.view(T * N, *self.obs_shape)
        if self.cfg.channels_last:
            obs_flat = obs_flat.contiguous(memory_format=torch.channels_last)
        actions_flat = actions_tensor.view(T * N)
        logprob_flat = logprob_tensor.view(T * N)
        advantages_flat = advantages.view(T * N)
        returns_flat = returns.view(T * N)
        values_flat = value_tensor.view(T * N)

        advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std(unbiased=False) + 1e-8)

        self.model.train()
        num_samples = obs_flat.shape[0]
        minibatch = self.cfg.minibatch_size
        if minibatch <= 0 or minibatch > num_samples:
            minibatch = num_samples

        loss_summaries: Dict[str, float] = {"policy": 0.0, "value": 0.0, "entropy": 0.0}
        batch_count = 0

        for epoch in range(self.cfg.ppo_epochs):
            indices = torch.randperm(num_samples, device=self.device)
            for start in range(0, num_samples, minibatch):
                idx = indices[start:start + minibatch]
                batch_obs = obs_flat[idx]
                batch_actions = actions_flat[idx]
                batch_old_logprob = logprob_flat[idx]
                batch_adv = advantages_flat[idx]
                batch_ret = returns_flat[idx]

                logits, values_pred = self.model(batch_obs)
                dist = Categorical(logits=logits.float())
                logprob = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                ratio = (logprob - batch_old_logprob).exp()
                surrogate1 = ratio * batch_adv
                surrogate2 = torch.clamp(ratio, 1.0 - self.cfg.ppo_clip, 1.0 + self.cfg.ppo_clip) * batch_adv
                policy_loss = -torch.min(surrogate1, surrogate2).mean()

                value_error = batch_ret - values_pred.squeeze(-1)
                value_loss = 0.5 * (value_error ** 2).mean()

                loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                loss_summaries["policy"] += policy_loss.item()
                loss_summaries["value"] += value_loss.item()
                loss_summaries["entropy"] += entropy.item()
                batch_count += 1

        self.rollout.clear()
        self.update_step += 1
        elapsed = time.time() - self.training_start_time
        avg_policy = loss_summaries["policy"] / max(batch_count, 1)
        avg_value = loss_summaries["value"] / max(batch_count, 1)
        avg_entropy = loss_summaries["entropy"] / max(batch_count, 1)
        print(
            f"[ppo] update={self.update_step} elapsed={elapsed/60:.1f}m policy={avg_policy:.4f} "
            f"value={avg_value:.4f} entropy={avg_entropy:.4f}"
        )
        self.model.eval()

    @torch.no_grad()
    def act_and_infer(self, obs_proc: torch.Tensor) -> torch.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.model(obs_proc)
        dist = Categorical(logits=logits.float())
        actions = dist.sample()
        logprob = dist.log_prob(actions)
        return actions, logprob, value.squeeze(-1)

    def act_and_learn(
        self,
        obs: torch.Tensor,
        info: torch.Tensor,
        act_buffer: torch.Tensor,
        *,
        train: bool = True,
    ) -> Optional[torch.Tensor]:
        obs_proc = self._prepare_obs(obs)

        if train:
            rewards = info[:, 0].to(self.device, dtype=torch.float32)
            dones = (info[:, 1] + info[:, 2]).to(self.device, dtype=torch.float32).clamp(0.0, 1.0)

        actions, logprob, value = self.act_and_infer(obs_proc)

        if train:
            if self.last_obs is not None:
                self.rollout.add(
                    self.last_obs,
                    self.last_action,
                    self.last_logprob,
                    self.last_value,
                    rewards,
                    dones,
                )
                self._maybe_update(value)

            self.last_obs = obs_proc.detach()
            self.last_action = actions.detach()
            self.last_logprob = logprob.detach()
            self.last_value = value.detach()
        else:
            # Drop any partial rollout state to avoid mixing eval frames
            self.rollout.clear()
            self.last_obs = None
            self.last_action = None
            self.last_logprob = None
            self.last_value = None

        if act_buffer.device != self.device:
            act_out = actions.to(device=act_buffer.device, dtype=act_buffer.dtype)
        else:
            act_out = actions.to(dtype=act_buffer.dtype)
        act_buffer.copy_(act_out)
        return None


# Alias for the runner import
Agent = PPOAgent
