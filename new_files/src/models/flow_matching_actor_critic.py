"""
actor_critic.py の方策フローマッチング版。

- 行動をnum_actions次元のone-hotベクトルとして扱い、ノイズからそのベクトルへの
  速度場を予測するフローマッチングモデルを方策として使う(Categorical分布は使わない)。
- フローマッチングは厳密なlog_probを簡単には計算できないため、通常のREINFORCE損失
  (-log_prob(act) * advantage)ではなく、Advantage-Weighted Regression(AWR)方式で学習する:
  実際に取った行動をターゲットに回帰(MSE)しつつ、アドバンテージが高いサンプルほど
  重みを大きくする。
- エントロピー正則化はフローマッチングでは同じ形で計算できないため、今回は省略する。
- 価値関数(critic)・状態エンコーダ・LSTMは元のActorCriticと共通のものを再利用する。
"""

from collections import namedtuple
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from .actor_critic import ActorCriticEncoder, compute_lambda_returns
from .blocks import FourierFeatures
from coroutines.flow_matching_env_loop import make_flow_matching_env_loop
from envs import TorchEnv, WorldModelEnv
from utils import init_lstm, LossAndLogs

FlowActorCriticOutput = namedtuple("FlowActorCriticOutput", "act val hidden hx_cx")


@dataclass
class FlowMatchingActorCriticConfig:
    lstm_dim: int
    img_channels: int
    img_size: int
    channels: List[int]
    down: List[int]
    num_flow_steps: int = 10
    awr_temperature: float = 1.0
    awr_max_weight: float = 20.0
    num_actions: Optional[int] = None


class FlowMatchingActorCritic(nn.Module):
    def __init__(self, cfg: FlowMatchingActorCriticConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = ActorCriticEncoder(cfg)
        self.lstm_dim = cfg.lstm_dim
        input_dim_lstm = cfg.channels[-1] * (cfg.img_size // 2 ** (sum(cfg.down))) ** 2
        self.lstm = nn.LSTMCell(input_dim_lstm, cfg.lstm_dim)
        self.critic_linear = nn.Linear(cfg.lstm_dim, 1)

        time_dim = 32
        self.time_emb = FourierFeatures(time_dim)
        self.flow_head = nn.Sequential(
            nn.Linear(cfg.num_actions + time_dim + cfg.lstm_dim, cfg.lstm_dim),
            nn.SiLU(),
            nn.Linear(cfg.lstm_dim, cfg.lstm_dim),
            nn.SiLU(),
            nn.Linear(cfg.lstm_dim, cfg.num_actions),
        )

        self.critic_linear.weight.data.fill_(0)
        self.critic_linear.bias.data.fill_(0)
        init_lstm(self.lstm)

        self.env_loop = None
        self.loss_cfg = None

    @property
    def device(self) -> torch.device:
        return self.lstm.weight_hh.device

    def setup_training(self, rl_env: Union[TorchEnv, WorldModelEnv], loss_cfg) -> None:
        assert self.env_loop is None and self.loss_cfg is None
        self.env_loop = make_flow_matching_env_loop(rl_env, self)
        self.loss_cfg = loss_cfg

    def compute_velocity(self, x_t: Tensor, t: Tensor, hidden: Tensor) -> Tensor:
        te = self.time_emb(t)
        return self.flow_head(torch.cat([x_t, te, hidden], dim=-1))

    @torch.no_grad()
    def sample_action_vector(self, hidden: Tensor) -> Tensor:
        b = hidden.size(0)
        x = torch.randn(b, self.cfg.num_actions, device=hidden.device)
        dt = 1.0 / self.cfg.num_flow_steps
        for step in range(self.cfg.num_flow_steps):
            t = torch.full((b,), step * dt, device=hidden.device)
            v = self.compute_velocity(x, t, hidden)
            x = x + v * dt
        return x

    def predict_act_value(self, obs: Tensor, hx_cx: Tuple[Tensor, Tensor]) -> FlowActorCriticOutput:
        assert obs.ndim == 4
        x = self.encoder(obs)
        x = x.flatten(start_dim=1)
        hx, cx = self.lstm(x, hx_cx)
        val = self.critic_linear(hx).squeeze(dim=1)
        action_vector = self.sample_action_vector(hx)
        act = action_vector.argmax(dim=-1)
        return FlowActorCriticOutput(act, val, hx, (hx, cx))

    def forward(self) -> LossAndLogs:
        c = self.loss_cfg
        _, act, rew, end, trunc, hidden, val, val_bootstrap, _ = self.env_loop.send(c.backup_every)

        lambda_returns = compute_lambda_returns(rew, end, trunc, val_bootstrap, c.gamma, c.lambda_)
        advantage = (lambda_returns - val).detach()

        b, t_len = act.shape
        hidden_flat = hidden.reshape(b * t_len, -1)
        act_flat = act.reshape(b * t_len)
        adv_flat = advantage.reshape(b * t_len)

        x1 = F.one_hot(act_flat, num_classes=self.cfg.num_actions).float()
        x0 = torch.randn_like(x1)
        t = torch.rand(b * t_len, device=self.device)
        t_ = t.unsqueeze(-1)
        x_t = (1 - t_) * x0 + t_ * x1
        target_v = x1 - x0

        pred_v = self.compute_velocity(x_t, t, hidden_flat)

        # Advantage-Weighted Regression: アドバンテージが高い行動ほど強く模倣する
        weight = torch.exp(adv_flat / self.cfg.awr_temperature).clamp(max=self.cfg.awr_max_weight)
        per_sample_loss = F.mse_loss(pred_v, target_v, reduction="none").mean(dim=-1)
        loss_actions = (weight * per_sample_loss).mean()

        loss_values = c.weight_value_loss * F.mse_loss(val, lambda_returns)
        loss = loss_actions + loss_values

        metrics = {
            "loss_actions": loss_actions.detach(),
            "loss_values": loss_values.detach(),
            "awr_weight_mean": weight.mean().detach(),
            "loss_total": loss.detach(),
        }
        return loss, metrics
