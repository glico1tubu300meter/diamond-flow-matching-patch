"""
学習済みのFlowMatching世界モデルの中で、ランダムな行動を取り続けた様子をmp4動画として保存する。
play.pyのprepare_play_modeと同じ要領で、実環境から少しだけ初期フレームを集めてから
WorldModelEnv(想像上のゲーム世界)をリセットし、そこでランダム行動を繰り返す。

使い方(このrunフォルダの中で実行する想定):
    python random_action_video.py <run_dir> <num_steps>
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

REPO_SRC = r"I:\マイドライブ\Claude2\output\2609\06-021-diamond-wm\repo\src"
sys.path.insert(0, REPO_SRC)

OmegaConf.register_new_resolver("eval", eval)

from agent import Agent  # noqa: E402
from coroutines.collector import make_collector, NumToCollect  # noqa: E402
from data import BatchSampler, collate_segments_to_batch, Dataset  # noqa: E402
from envs import make_atari_env, WorldModelEnv  # noqa: E402
from utils import get_path_agent_ckpt  # noqa: E402


def tensor_to_bgr(obs: torch.Tensor) -> np.ndarray:
    # obs: (1, c, h, w), 値域[-1, 1] -> uint8 BGR画像
    img = obs[0].add(1).div(2).mul(255).clamp(0, 255).byte()
    img = img.permute(1, 2, 0).cpu().numpy()  # (h, w, c) RGB
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def main():
    run_dir = Path(sys.argv[1])
    num_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 150

    os.chdir(run_dir)  # play.pyと同様、checkpoints/datasetの相対パスをrunフォルダ基準にする

    with initialize_config_dir(version_base="1.3", config_dir=str(Path(REPO_SRC).parent / "config")):
        cfg = compose(config_name="trainer", overrides=["agent=flow_matching", "world_model_env=flow_matching"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_env = make_atari_env(num_envs=1, device=device, **cfg.env.test)

    agent = Agent(instantiate(cfg.agent, num_actions=test_env.num_actions)).to(device).eval()
    path_ckpt = get_path_agent_ckpt("checkpoints", epoch=-1)
    print("loading checkpoint:", path_ckpt)
    agent.load(path_ckpt)

    # 世界モデルの「想像」を始めるための初期フレームを、実環境からランダム行動で少し集める
    n_burnin = 200
    dataset = Dataset(Path(f"dataset/random_video_burnin_{n_burnin}"))
    dataset.load_from_default_path()
    if len(dataset) == 0:
        print(f"Collecting {n_burnin} steps in real environment for world model initialization.")
        collector = make_collector(test_env, agent.actor_critic, dataset, epsilon=1.0)  # epsilon=1.0で完全ランダム行動
        collector.send(NumToCollect(steps=n_burnin))
        dataset.save_to_default_path()

    bs = BatchSampler(dataset, 0, 1, 1, cfg.agent.denoiser.inner_model.num_steps_conditioning, None, False)
    dl = DataLoader(dataset, batch_sampler=bs, collate_fn=collate_segments_to_batch)
    wm_env_cfg = instantiate(cfg.world_model_env, num_batches_to_preload=1)
    wm_env = WorldModelEnv(agent.denoiser, agent.rew_end_model, dl, wm_env_cfg)

    video_path = run_dir / "random_action_flow_matching.mp4"
    with torch.no_grad():
        obs, _ = wm_env.reset()
        h, w = obs.shape[-2:]
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h))
        writer.write(tensor_to_bgr(obs.unsqueeze(0) if obs.ndim == 3 else obs))

        for step in range(num_steps):
            act = torch.randint(0, test_env.num_actions, (1,), device=device)
            obs, rew, end, trunc, info = wm_env.step(act)
            writer.write(tensor_to_bgr(obs.unsqueeze(0) if obs.ndim == 3 else obs))
            if step % 20 == 0:
                print(f"step={step} reward={rew.item():.2f} end={bool(end.item())}")

    writer.release()
    print("saved video to:", video_path)


if __name__ == "__main__":
    main()
