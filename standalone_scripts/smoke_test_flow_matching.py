"""
DIAMONDリポジトリに追加したFlowMatching(flow_matching.py / flow_matching_sampler.py)が
正しく動くかを検証する、実データ不要のスタンドアロン・スモークテスト。

- ランダムな合成データでBatchを組み立てforward()を呼び、lossが計算でき、
  backward()で勾配がInnerModelのパラメータに伝わることを確認する
- FlowMatchingSampler.sample()が正しい形状の画像を生成できることを確認する
"""

import sys

import torch

REPO_SRC = r"I:\マイドライブ\Claude2\output\2609\06-021-diamond-wm\repo\src"
sys.path.insert(0, REPO_SRC)

from data import Batch  # noqa: E402
from models.diffusion.flow_matching import FlowMatching, FlowMatchingConfig  # noqa: E402
from models.diffusion.flow_matching_sampler import FlowMatchingSampler, FlowMatchingSamplerConfig  # noqa: E402
from models.diffusion.inner_model import InnerModelConfig  # noqa: E402


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    img_channels = 3
    num_steps_conditioning = 4
    num_actions = 4
    h = w = 32  # 本番は64だが、スモークテストなので軽量化

    inner_cfg = InnerModelConfig(
        img_channels=img_channels,
        num_steps_conditioning=num_steps_conditioning,
        cond_channels=64,
        depths=[1, 1],
        channels=[16, 16],
        attn_depths=[0, 0],
        num_actions=num_actions,
    )
    fm_cfg = FlowMatchingConfig(inner_model=inner_cfg)
    model = FlowMatching(fm_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch_size = 4
    seq_len = num_steps_conditioning + 3  # 3ステップ分のロールアウトを学習

    def make_batch():
        obs = torch.rand(batch_size, seq_len, img_channels, h, w, device=device) * 2 - 1  # [-1, 1]
        act = torch.randint(0, num_actions, (batch_size, seq_len), device=device)
        mask_padding = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        return Batch(
            obs=obs, act=act,
            rew=torch.zeros(batch_size, seq_len, device=device),
            end=torch.zeros(batch_size, seq_len, dtype=torch.long, device=device),
            trunc=torch.zeros(batch_size, seq_len, dtype=torch.long, device=device),
            mask_padding=mask_padding, info=[{} for _ in range(batch_size)], segment_ids=[],
        )

    print("--- training smoke test (does the loss go down?) ---")
    for step in range(30):
        batch = make_batch()
        loss, logs = model(batch)
        optimizer.zero_grad()
        loss.backward()
        grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
        optimizer.step()
        if step % 5 == 0 or step == 29:
            print(f"step={step:2d} loss={loss.item():.4f} grad_norm={grad_norm:.4f}")

    print("--- sampler smoke test (does it produce the right shape?) ---")
    sampler = FlowMatchingSampler(model, FlowMatchingSamplerConfig(num_steps=5))
    prev_obs = torch.rand(batch_size, num_steps_conditioning, img_channels, h, w, device=device) * 2 - 1
    prev_act = torch.randint(0, num_actions, (batch_size, num_steps_conditioning), device=device)
    x, trajectory = sampler.sample(prev_obs, prev_act)
    print("sampled image shape:", tuple(x.shape), "value range:", x.min().item(), x.max().item())
    print("trajectory length:", len(trajectory))
    print("OK")


if __name__ == "__main__":
    main()
