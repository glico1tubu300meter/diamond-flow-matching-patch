"""
DIAMOND本家のアップサンプラー(拡散モデルで低解像度→高解像度)と同じ考え方を、
既存のFlowMatching/InnerModelをそのまま流用してフローマッチング版で軽く試す。

- 「直前のゲーム画面(複数フレーム)」を条件にする代わりに、
  「バイキュービックで粗く高解像度サイズまで拡大した低解像度画像」を条件にする
- num_steps_conditioning=1(条件フレームは1枚)、num_actions=1(行動条件は使わないのでダミー)
  にするだけで、既存のFlowMatching/InnerModelがそのまま超解像モデルとして使える
- 実際のBreakoutを高解像度(128x128)でプレイして集めたフレームを正解(x1)とし、
  それをバイキュービックで64x64に一度落として128x128に戻したもの(ぼやけた画像)を
  条件入力にする、というペアデータで学習する
"""

import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

REPO_SRC = r"I:\マイドライブ\Claude2\output\2609\06-021-diamond-wm\repo\src"
sys.path.insert(0, REPO_SRC)

from models.diffusion.flow_matching import FlowMatching, FlowMatchingConfig  # noqa: E402
from models.diffusion.inner_model import InnerModelConfig  # noqa: E402
from envs import make_atari_env  # noqa: E402

HIGH_RES = 128
LOW_RES = 64
OUT_DIR = r"I:\マイドライブ\Claude2\output\2609\06-021-diamond-wm\repo\outputs\upsampler_demo"


def collect_high_res_frames(n_frames: int, device: torch.device) -> torch.Tensor:
    env = make_atari_env(
        id="BreakoutNoFrameskip-v4", num_envs=1, device=device,
        done_on_life_loss=False, size=HIGH_RES, max_episode_steps=None,
    )
    frames = []
    obs, _ = env.reset()
    frames.append(obs[0].clone())
    for _ in range(n_frames - 1):
        act = torch.randint(0, env.num_actions, (1,), device=device)
        obs, rew, end, trunc, info = env.step(act)
        frames.append(obs[0].clone())
        if (end + trunc).any():
            obs, _ = env.reset()
    env.close()
    return torch.stack(frames)  # (n, c, H, W), 値域[-1, 1]


def make_low_res_conditioning(high_res: torch.Tensor) -> torch.Tensor:
    low = F.interpolate(high_res, size=(LOW_RES, LOW_RES), mode="bicubic", align_corners=False)
    low_upsampled = F.interpolate(low, size=(HIGH_RES, HIGH_RES), mode="bicubic", align_corners=False)
    return low_upsampled.clamp(-1, 1)


def to_bgr(img: torch.Tensor) -> np.ndarray:
    arr = img.add(1).div(2).mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    print("collecting real Breakout frames at high resolution...")
    high_res_frames = collect_high_res_frames(600, device)
    low_res_cond = make_low_res_conditioning(high_res_frames)
    print("dataset:", high_res_frames.shape)

    n = high_res_frames.shape[0]
    n_train = int(n * 0.9)

    inner_cfg = InnerModelConfig(
        img_channels=3,
        num_steps_conditioning=1,
        cond_channels=64,
        depths=[1, 1, 1],
        channels=[32, 32, 64],
        attn_depths=[0, 0, 0],
        num_actions=1,
    )
    model = FlowMatching(FlowMatchingConfig(inner_model=inner_cfg)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)

    dummy_act = torch.zeros(1, 1, dtype=torch.long, device=device)

    batch_size = 16
    n_steps = 600
    for step in range(n_steps):
        idx = torch.randint(0, n_train, (batch_size,))
        x1 = high_res_frames[idx]
        cond = low_res_cond[idx]
        x0 = torch.randn_like(x1)
        t = torch.rand(batch_size, device=device)
        t_ = t.view(-1, 1, 1, 1)
        x_t = (1 - t_) * x0 + t_ * x1
        target_v = x1 - x0

        act = dummy_act.expand(batch_size, 1)
        pred_v = model.compute_velocity(x_t, t, cond, act)
        loss = F.mse_loss(pred_v, target_v)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 50 == 0 or step == n_steps - 1:
            print(f"step={step} loss={loss.item():.4f}")

    # held-outデータで生成(超解像)を試す
    model.eval()
    idx = n_train  # 学習に使っていないフレーム
    cond = low_res_cond[idx : idx + 1]
    gt = high_res_frames[idx : idx + 1]

    with torch.no_grad():
        x = torch.randn_like(gt)
        num_flow_steps = 10
        dt = 1.0 / num_flow_steps
        act = dummy_act
        for s in range(num_flow_steps):
            t = torch.full((1,), s * dt, device=device)
            v = model.compute_velocity(x, t, cond, act)
            x = x + v * dt
        generated = x.clamp(-1, 1)

    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    panel = np.concatenate([
        to_bgr(cond[0]),
        to_bgr(generated[0]),
        to_bgr(gt[0]),
    ], axis=1)
    out_path = os.path.join(OUT_DIR, "upsampler_comparison.png")
    ok, buf = cv2.imencode(".png", panel)
    assert ok
    with open(out_path, "wb") as f:
        f.write(buf.tobytes())
    print("saved comparison image (left=bicubic input / middle=flow-matching output / right=ground truth) to:", out_path)


if __name__ == "__main__":
    main()
