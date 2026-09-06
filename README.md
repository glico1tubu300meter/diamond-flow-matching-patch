# DIAMOND + Flow Matching(差分パッチのみ)

このリポジトリには**私たちが独自に書いたコードだけ**が入っています。[DIAMOND](https://github.com/eloialonso/diamond)
(*Diffusion for World Modeling: Visual Details Matter in Atari*, NeurIPS
2024, [arXiv:2405.12399](https://arxiv.org/abs/2405.12399)) に対する、フローマッチング版の追加実装一式です。DIAMOND本体のソースコードは**含みません** — 本家(MITライセンス)を各自でcloneし、このパッチを適用して使ってください。

## 何を追加したか

DIAMONDには3つのニューラルネットがあります。このパッチでは、それぞれについてフローマッチング版の代替実装を追加しています(可能な限り、DIAMOND既存の`InnerModel`(U-Net)をそのまま再利用)。

| コンポーネント | 本家(DIAMOND) | このパッチ |
|---|---|---|
| 世界モデル | EDMスタイルの拡散モデル `Denoiser` / `DiffusionSampler` | `FlowMatching` / `FlowMatchingSampler` — ノイズから次フレームへの直線経路上の「速度場」を予測し、少数回のEuler積分でサンプリング |
| 方策(行動選択) | カテゴリカル分布 + REINFORCE系のactor-critic | `FlowMatchingActorCritic` — ノイズからone-hot行動ベクトルへのフローを、Advantage-Weighted Regression(AWR)で学習(フローマッチングは厳密なlog_probが計算できないため、通常の`-log_prob(act) * advantage`損失は使えない) |
| 超解像アップサンプラー(実証実験) | 拡散モデル版アップサンプラー(DIAMONDのCSGOブランチ) | 同じ`InnerModel`を、`num_steps_conditioning=1`にして「条件フレーム」を過去のゲーム画面ではなく「バイキュービックで粗く拡大した低解像度画像」に差し替えて流用 — `standalone_scripts/train_flow_matching_upsampler.py`参照 |

これらはすべて**追加的な変更**です。DIAMOND本来の拡散モデルのパイプラインには一切手を加えておらず、新しい機能はHydraの設定上書き(`agent=flow_matching`、`agent=flow_matching_policy`、`world_model_env=flow_matching`)で選択的に有効化される形になっています。

## 適用方法

```bash
git clone https://github.com/eloialonso/diamond.git
cd diamond
git checkout 5bcd1599755b4f2fae8e5e079e02f0728e174965   # このパッチのベースにしたmainのコミット
git apply /path/to/patches/modified_files.patch
cp -r /path/to/new_files/* .
```

学習・プレイの実行例:

```bash
python src/main.py agent=flow_matching world_model_env=flow_matching env.train.id=BreakoutNoFrameskip-v4
python src/main.py agent=flow_matching_policy world_model_env=flow_matching env.train.id=BreakoutNoFrameskip-v4
```

## 構成

- `patches/modified_files.patch` — 既存のDIAMONDファイルへの小さな変更のunified diff(主に`isinstance`による分岐で、設定次第で新しいクラスを選べるようにしたもの。加えてWindows特有のバグ修正 — `Path.rename`はWindowsで上書きできないため、エピソード保存処理を修正済み)
- `new_files/` — 新規追加ファイル。DIAMONDの`src/`・`config/`ツリーと同じ相対パスに配置
- `standalone_scripts/` — メインの学習ループとは別に動作確認に使ったスクリプト群(合成データでのスモークテスト、世界モデル/方策のプレイ動画生成、超解像の実証実験)。ローカルパスがハードコードされた例になっているので、各自の環境に合わせて書き換えてください
- `assets/upsampler_comparison.png` — バイキュービック(入力) / フローマッチングの出力 / 正解画像 の比較(600stepのごく短い学習での結果)

## 現状・制約

ここでの検証はすべて`BreakoutNoFrameskip-v4`での**短時間・小規模な実行**によるもので、パイプライン全体(データ収集→世界モデル学習→世界モデル内での方策学習→実環境での評価)が正しく配線されているかを確認する目的です。論文と同規模の学習は行っていません。具体的には:

- 世界モデルの denoising loss は、数千回の実環境ステップで約1.7→約0.01〜0.04まで低下
- フローマッチング方策は、実環境で報酬を獲得しながらエピソードを完走できる
- アップサンプラーは、600フレーム・600stepの学習後、バイキュービック入力よりも明らかにくっきりした detail(スコア表示の文字やブロックのエッジ)を復元できている

いずれも、Atari 100kベンチマークで良いスコアを出せるほどの学習量ではありません。また方策側のAdvantage-Weighted Regression損失には、元のREINFORCE損失にあったエントロピー正則化に相当する仕組みがありません。

## ライセンス

DIAMOND本体はMITライセンス(Copyright (c) 2024 Eloi Alonso)です。本パッチも同条件で提供します。
