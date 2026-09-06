"""
DIAMONDのDenoiser(EDMスタイル拡散モデル)を、フローマッチング(Rectified Flow)で
置き換えた実験的な実装。

- denoiser.pyとの違い:
  - ノイズレベルsigmaに応じたc_in/c_out/c_skipのような前処理係数は使わない
  - ノイズx0と本物画像x1を直線で結んだ経路 x_t = (1-t)*x0 + t*x1 上の
    「速度場」v = x1 - x0 を直接予測するように学習する(損失はMSEのみ)
  - InnerModel(U-Net本体)はdenoiser.pyと完全に共通のものを再利用する
"""

from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from data import Batch
from .denoiser import add_dims
from .inner_model import InnerModel, InnerModelConfig
from utils import LossAndLogs


@dataclass
class FlowMatchingConfig:
    inner_model: InnerModelConfig


class FlowMatching(nn.Module):
    def __init__(self, cfg: FlowMatchingConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.inner_model = InnerModel(cfg.inner_model)

    @property
    def device(self) -> torch.device:
        return self.inner_model.noise_emb.weight.device

    def setup_training(self, *args, **kwargs) -> None:
        # 拡散モデル版(Denoiser)と違い、sigma分布のセットアップは不要なのでno-op
        pass

    def compute_velocity(self, x_t: Tensor, t: Tensor, obs: Tensor, act: Tensor) -> Tensor:
        c_noise = add_dims(t, 1)
        return self.inner_model(x_t, c_noise, obs, act)

    def forward(self, batch: Batch) -> LossAndLogs:
        n = self.cfg.inner_model.num_steps_conditioning
        seq_length = batch.obs.size(1) - n

        all_obs = batch.obs.clone()
        loss = 0

        for i in range(seq_length):
            obs = all_obs[:, i : n + i]
            next_obs = all_obs[:, n + i]  # x1: 本物の次フレーム ([-1, 1]想定)
            act = batch.act[:, i : n + i]
            mask = batch.mask_padding[:, n + i]

            b, t_dim, c, h, w = obs.shape
            obs_flat = obs.reshape(b, t_dim * c, h, w)

            x0 = torch.randn_like(next_obs)
            t = torch.rand(b, device=self.device)
            t_ = add_dims(t, next_obs.ndim)
            x_t = (1 - t_) * x0 + t_ * next_obs
            target_v = next_obs - x0

            pred_v = self.compute_velocity(x_t, t, obs_flat, act)
            loss += F.mse_loss(pred_v[mask], target_v[mask])

        loss /= seq_length
        return loss, {"loss_flow_matching": loss.detach()}
