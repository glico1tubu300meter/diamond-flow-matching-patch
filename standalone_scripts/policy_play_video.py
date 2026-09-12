"""
学習済みのFlowMatching方策(actor_critic)に、実際のBreakout環境をプレイさせて
その様子をmp4動画として保存する(想像上の世界モデルではなく本物のゲーム画面)。

使い方(runフォルダを指定して実行する):
    python policy_play_video.py <run_dir> <num_steps>
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

REPO_SRC = r"/path/to/diamond/src"
sys.path.insert(0, REPO_SRC)

OmegaConf.register_new_resolver("eval", eval)

from agent import Agent  # noqa: E402
from envs import make_atari_env  # noqa: E402
from utils import get_path_agent_ckpt  # noqa: E402


def tensor_to_bgr(obs: torch.Tensor) -> np.ndarray:
    img = obs[0].add(1).div(2).mul(255).clamp(0, 255).byte()
    img = img.permute(1, 2, 0).cpu().numpy()
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def main():
    run_dir = Path(sys.argv[1])
    num_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 300

    os.chdir(run_dir)

    with initialize_config_dir(version_base="1.3", config_dir=str(Path(REPO_SRC).parent / "config")):
        cfg = compose(config_name="trainer", overrides=["agent=flow_matching_policy", "world_model_env=flow_matching"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_env = make_atari_env(num_envs=1, device=device, **cfg.env.test)

    agent = Agent(instantiate(cfg.agent, num_actions=test_env.num_actions)).to(device).eval()
    path_ckpt = get_path_agent_ckpt("checkpoints", epoch=-1)
    print("loading checkpoint:", path_ckpt)
    agent.load(path_ckpt)

    video_path = run_dir / "policy_play_flow_matching.mp4"

    with torch.no_grad():
        obs, _ = test_env.reset()
        h, w = obs.shape[-2:]
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h))
        writer.write(tensor_to_bgr(obs))

        hx = torch.zeros(1, agent.actor_critic.lstm_dim, device=device)
        cx = torch.zeros(1, agent.actor_critic.lstm_dim, device=device)

        total_reward = 0.0
        for step in range(num_steps):
            act, val, hidden, (hx, cx) = agent.actor_critic.predict_act_value(obs, (hx, cx))
            obs, rew, end, trunc, info = test_env.step(act)
            total_reward += rew.item()
            writer.write(tensor_to_bgr(obs))
            if step % 30 == 0:
                print(f"step={step} action={act.item()} reward_so_far={total_reward:.1f}")
            if (end + trunc).any():
                print(f"episode ended at step {step}, total_reward={total_reward:.1f}")
                obs, _ = test_env.reset()
                hx.zero_()
                cx.zero_()

    writer.release()
    print("saved video to:", video_path)
    print("total_reward:", total_reward)


if __name__ == "__main__":
    main()
