"""
env_loop.py の方策フローマッチング版。

通常版との違い: Categorical(logits).sample() で行動を選ぶ代わりに、
model.predict_act_value() が既にフロー(ノイズ→行動ベクトル)から
行動をサンプリング済みで返してくるので、それをそのまま使う。
また、ロールアウトバッファには logits の代わりに LSTM隠れ状態(hidden)を
保存しておき、後で損失計算時に速度場を再計算できるようにする。
"""

import random
from typing import Generator, Tuple, Union

import torch
import torch.nn as nn

from . import coroutine
from envs import TorchEnv, WorldModelEnv


@coroutine
def make_flow_matching_env_loop(
    env: Union[TorchEnv, WorldModelEnv], model: nn.Module, epsilon: float = 0.0
) -> Generator[Tuple[torch.Tensor, ...], int, None]:
    num_steps = yield

    hx = torch.zeros(env.num_envs, model.lstm_dim, device=model.device)
    cx = torch.zeros(env.num_envs, model.lstm_dim, device=model.device)

    seed = random.randint(0, 2**31 - 1)
    obs, _ = env.reset(seed=[seed + i for i in range(env.num_envs)])

    while True:
        hx, cx = hx.detach(), cx.detach()
        all_ = []
        infos = []
        n = 0

        while n < num_steps:
            act, val, hidden, (hx, cx) = model.predict_act_value(obs, (hx, cx))

            if random.random() < epsilon:
                act = torch.randint(low=0, high=env.num_actions, size=(obs.size(0),), device=obs.device)

            next_obs, rew, end, trunc, info = env.step(act)

            if n > 0:
                val_bootstrap = val.detach().clone()
                if dead.any():
                    val_bootstrap[dead] = val_final_obs
                all_[-1][-1] = val_bootstrap

            dead = torch.logical_or(end, trunc)

            if dead.any():
                with torch.no_grad():
                    _, val_final_obs, _, _ = model.predict_act_value(info["final_observation"], (hx[dead], cx[dead]))
                reset_gate = 1 - dead.float().unsqueeze(1)
                hx = hx * reset_gate
                cx = cx * reset_gate
                if "burnin_obs" in info:
                    burnin_obs = info["burnin_obs"]
                    for i in range(burnin_obs.size(1)):
                        _, _, _, (hx[dead], cx[dead]) = model.predict_act_value(burnin_obs[:, i], (hx[dead], cx[dead]))

            all_.append([obs, act, rew, end, trunc, hidden, val, None])
            infos.append(info)

            obs = next_obs
            n += 1

        with torch.no_grad():
            _, val_bootstrap, _, _ = model.predict_act_value(next_obs, (hx, cx))  # do not update hx/cx

        if dead.any():
            val_bootstrap[dead] = val_final_obs

        all_[-1][-1] = val_bootstrap

        all_obs, act, rew, end, trunc, hidden, val, val_bootstrap = (torch.stack(x, dim=1) for x in zip(*all_))

        num_steps = yield all_obs, act, rew, end, trunc, hidden, val, val_bootstrap, infos
