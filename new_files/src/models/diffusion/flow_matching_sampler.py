"""
FlowMatchingモデル用のサンプラー。DiffusionSamplerと同じ
`sample(prev_obs, prev_act) -> (x, trajectory)` インターフェースを持つ。

拡散モデルのように多段階(数十〜数百ステップ)の複雑なEuler/Heun法は使わず、
ノイズx0から本物画像x1へ向かう速度場vを、t=0から1まで単純なEuler積分するだけ。
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from .flow_matching import FlowMatching


@dataclass
class FlowMatchingSamplerConfig:
    num_steps: int = 10


class FlowMatchingSampler:
    def __init__(self, denoiser: FlowMatching, cfg: FlowMatchingSamplerConfig) -> None:
        # DiffusionSamplerとの互換性のため、属性名は`denoiser`のままにする
        # (WorldModelEnv.deviceがsampler.denoiser.deviceを参照するため)
        self.denoiser = denoiser
        self.cfg = cfg

    @torch.no_grad()
    def sample(self, prev_obs: Tensor, prev_act: Optional[Tensor]) -> Tuple[Tensor, List[Tensor]]:
        device = prev_obs.device
        b, t, c, h, w = prev_obs.size()
        obs_flat = prev_obs.reshape(b, t * c, h, w)

        x = torch.randn(b, c, h, w, device=device)
        trajectory = [x]
        dt = 1.0 / self.cfg.num_steps
        for step in range(self.cfg.num_steps):
            t_val = torch.full((b,), step * dt, device=device)
            v = self.denoiser.compute_velocity(x, t_val, obs_flat, prev_act)
            x = x + v * dt
            trajectory.append(x)

        x = x.clamp(-1, 1)
        return x, trajectory
