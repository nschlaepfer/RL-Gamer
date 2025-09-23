#!/usr/bin/env python3
"""Reference Agent stub for RL-Gamer.

This implementation is intentionally simple: it samples actions from a small
convolutional policy network without performing any learning. The goal is to
provide a drop-in module that satisfies the runner's API so the end-to-end
pipeline can be profiled before wiring up a real reinforcement-learning agent.

Replace the policy definition, optimizer logic, and `act_and_learn` body with
your actual algorithm once you're ready.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


MAX_ACTIONS = 18  # keep in sync with gamer.py
DEFAULT_DTYPE = torch.float16


@dataclass
class AgentConfig:
    num_actions: int = MAX_ACTIONS
    epsilon: float = float(os.environ.get("AGENT_EPSILON", "0.05"))
    dtype_pref: torch.dtype = DEFAULT_DTYPE
    lr: float = 1e-4
    channels_last: bool = True


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TinyPolicy(nn.Module):
    """Small convolutional policy producing action logits."""

    def __init__(self, num_actions: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 27 * 16, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        return self.head(x)


class Agent:
    """Minimal agent compatible with gamer.py's expectations."""

    def __init__(self, config: Optional[AgentConfig] = None) -> None:
        self.cfg = config or AgentConfig()
        self.device = _select_device()

        dtype = self.cfg.dtype_pref if self.device.type in {"cuda", "mps"} else torch.float32
        memory_format = torch.channels_last if self.cfg.channels_last else torch.contiguous_format

        self.policy = TinyPolicy(self.cfg.num_actions)
        self.policy = self.policy.to(device=self.device, dtype=dtype, memory_format=memory_format)

        # Optimizer stub (unused until a real loss is implemented)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.cfg.lr)

        self.dtype = dtype
        self.step = 0

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def load(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        state = torch.load(path, map_location=self.device)
        policy_state = state.get("policy")
        if policy_state is not None:
            self.policy.load_state_dict(policy_state)
        opt_state = state.get("optimizer")
        if opt_state:
            try:
                self.optimizer.load_state_dict(opt_state)
            except ValueError:
                # Size mismatch is acceptable for stub upgrades; ignore optimizer state.
                pass
        self.step = int(state.get("step", 0))

    def save(self, path: str) -> None:
        state = {
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.step,
        }
        torch.save(state, path)

    # ------------------------------------------------------------------
    # Core API expected by gamer.py
    # ------------------------------------------------------------------
    @torch.no_grad()
    def act_and_learn(
        self,
        obs: torch.Tensor,
        info: torch.Tensor,
        act_buffer: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Produce an action for each environment instance.

        Args:
            obs: uint8 tensor shaped (N, 250, 160, 3) on the runner's device.
            info: float32 tensor shaped (N, 4) with aggregated stats (unused here).
            act_buffer: int64 tensor on the same device where actions can be written.
        Returns:
            If actions are written into `act_buffer`, returns ``None``; otherwise,
            returns an action tensor. This stub writes in-place and returns ``None``.
        """

        self.policy.eval()

        # Ensure tensors reside on the agent device.
        if obs.device != self.device:
            obs = obs.to(device=self.device, non_blocking=True)
        if info.device != self.device:
            info = info.to(device=self.device, non_blocking=True)

        # Prepare observations for the policy network
        obs_norm = obs.permute(0, 3, 1, 2)
        if self.cfg.channels_last:
            obs_norm = obs_norm.contiguous(memory_format=torch.channels_last)
        else:
            obs_norm = obs_norm.contiguous()
        obs_norm = obs_norm.to(dtype=self.dtype) / 255.0

        logits = self.policy(obs_norm)

        # Simple epsilon-greedy sampling on the policy logits
        if self.cfg.epsilon > 0.0:
            probs = F.softmax(logits.float(), dim=-1)
            greedy_actions = torch.argmax(probs, dim=-1)
            random_actions = torch.randint(0, self.cfg.num_actions, greedy_actions.shape, device=obs.device)
            choose_random = torch.rand_like(greedy_actions.float()) < self.cfg.epsilon
            actions = torch.where(choose_random, random_actions, greedy_actions)
        else:
            actions = torch.argmax(logits, dim=-1)

        actions = actions.clamp_(0, self.cfg.num_actions - 1)

        if act_buffer.device != self.device:
            act_out = actions.to(device=act_buffer.device, dtype=act_buffer.dtype)
        else:
            act_out = actions.to(dtype=act_buffer.dtype)

        act_buffer.copy_(act_out)
        self.step += 1
        return None
