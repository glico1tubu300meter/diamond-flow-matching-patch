# DIAMOND + Flow Matching (patch-only release)

This repository contains **only the code we wrote ourselves**: a set of flow
matching variants for [DIAMOND](https://github.com/eloialonso/diamond)
(*Diffusion for World Modeling: Visual Details Matter in Atari*, NeurIPS
2024, [arXiv:2405.12399](https://arxiv.org/abs/2405.12399)). It does **not**
include DIAMOND's own source code — apply the patch on top of your own
checkout of the upstream repo (MIT licensed).

## What this adds

DIAMOND has three neural components. This patch adds a flow-matching
alternative for each one, reusing DIAMOND's existing `InnerModel` (U-Net)
backbone wherever possible:

| Component | Upstream (DIAMOND) | This patch |
|---|---|---|
| World model | EDM-style diffusion `Denoiser` / `DiffusionSampler` | `FlowMatching` / `FlowMatchingSampler` — predicts a velocity field along the straight path from noise to the next frame, sampled with a handful of Euler steps |
| Policy | Categorical distribution + REINFORCE-style actor-critic | `FlowMatchingActorCritic` — flow from noise to a one-hot action vector, trained via Advantage-Weighted Regression (log-prob is intractable for flow matching, so the usual `-log_prob(act) * advantage` loss doesn't apply) |
| Super-resolution upsampler (proof of concept) | Diffusion upsampler (CSGO branch of DIAMOND) | Same `InnerModel`, repurposed with `num_steps_conditioning=1` so the "conditioning frame" is a bicubic-upsampled low-res image instead of previous game frames — see `standalone_scripts/train_flow_matching_upsampler.py` |

All of this is **additive**: DIAMOND's original diffusion pipeline is left
untouched, and the new behavior is opt-in via Hydra config overrides
(`agent=flow_matching`, `agent=flow_matching_policy`,
`world_model_env=flow_matching`).

## How to apply

```bash
git clone https://github.com/eloialonso/diamond.git
cd diamond
git checkout 5bcd1599755b4f2fae8e5e079e02f0728e174965   # main, as of this patch
git apply /path/to/patches/modified_files.patch
cp -r /path/to/new_files/* .
```

Then train/play with, e.g.:

```bash
python src/main.py agent=flow_matching world_model_env=flow_matching env.train.id=BreakoutNoFrameskip-v4
python src/main.py agent=flow_matching_policy world_model_env=flow_matching env.train.id=BreakoutNoFrameskip-v4
```

## Layout

- `patches/modified_files.patch` — unified diff of the small edits made to
  existing DIAMOND files (mostly `isinstance` branches so the new classes
  can be selected via config; also a Windows-only bugfix for episode saving,
  where `Path.rename` cannot overwrite on Windows).
- `new_files/` — new files, at the same relative paths as in DIAMOND's `src/`
  and `config/` trees.
- `standalone_scripts/` — scripts used to sanity-check things outside the
  main training loop (a synthetic-data smoke test, videos of the
  world model / policy, and the super-resolution proof of concept). These
  hardcode local paths as examples — adjust them for your own checkout.
- `assets/upsampler_comparison.png` — bicubic input vs. flow-matching output
  vs. ground truth, from a very short (600-step) training run.

## Status / limitations

Everything here was verified with **short, small-scale runs** on
`BreakoutNoFrameskip-v4` to confirm the pipeline is wired correctly end to
end (data collection → world-model training → policy training in
imagination → real-environment evaluation) — not at the paper's scale.
Concretely:

- World-model denoising loss decreases from ~1.7 to ~0.01–0.04 over a few
  thousand real environment steps.
- The flow-matching policy completes real episodes with nonzero reward.
- The upsampler recovers visibly sharper detail (score digits, block edges)
  than the bicubic input after 600 training steps on 600 collected frames.

None of this has been trained long enough to demonstrate strong Atari 100k
scores, and the policy's Advantage-Weighted Regression objective has no
entropy-regularization equivalent, unlike the original REINFORCE loss.

## License

DIAMOND is MIT licensed (Copyright (c) 2024 Eloi Alonso). This patch is
provided under the same terms.
