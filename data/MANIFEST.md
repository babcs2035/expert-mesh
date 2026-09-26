<!-- data/・models/ は .gitignore 対象のため，再現性を担保する目的で成果物のハッシュと生成コマンドのみここに記録する（docs/d0003 F5）．内容を再生成した場合は本ファイルも更新すること． -->

# data/ MANIFEST — 再現性のためのハッシュと生成コマンド

`data/`・`models/` はローカルディスク容量の都合で `.gitignore` されているため，実験結果の再現性は
このファイルに記録するハッシュと生成コマンドのみに依存する．各ファイルを再生成した際は，
`sha256sum <path>` を再実行し，値とコミットハッシュを更新すること．

## 評価データセット・分類器訓練データ

生成コマンド（`mise run setup` から呼ばれる／単体でも実行可能．build_dataset.py の docstring 参照）:

```
uv run python build_dataset.py \
    --output data/dataset.jsonl \
    --classifier-train-output data/classifier_train.jsonl
```

`_JMMLU_SAMPLE_SEED=20260726`（評価用）と `_CLASSIFIER_TRAIN_SAMPLE_SEED=20260727`（分類器訓練用）で
サンプリングシードを分離し，さらに質問本文単位でも重複排除している（docs/d0002 §2-1・§6-E で
本文重複 0 件を実測確認済み）．

**2026-07-30 更新（research_frontier 項目2 / d0003 X4）**: 複合ドメイン設問（`_COMPOUND_QUESTIONS`）を
20問→100問（43組み合わせ，10ドメイン全体をカバー）へ拡充した．JMMLU由来の単一ドメイン設問（1500問，
サンプリングシード・タスクマップとも無変更）と分類器訓練データは影響を受けず，
`data/classifier_train.jsonl` のハッシュは変更前と完全一致することを確認済み．

**2026-09-26 更新（Iteration 78, `compound_eval_set_expansion=llm_generated_separate_generator`）**:
LLM 生成 + 独立検証済みの複合設問 315 行（45 ドメインペア × 7 件，`compound-101`〜`compound-415`）を
`data/compound_questions_generated.jsonl` として純追加し，評価データセットを 1600 行→1915 行へ拡張した．
既存 100 行の複合設問（`compound-001`〜`compound-100`）と単一ドメイン 1500 行はビット単位で無変更
（`build_dataset.py:_build_rows()` の `_COMPOUND_QUESTIONS` ループの**後**に append するため）．
分類器訓練データ（`data/classifier_train.jsonl`）も無変更（ハッシュ一致，上表参照）．

生成コマンド（`scripts/generate_compound_eval_questions.py` の docstring 参照．生成器は
`judge_model` とは別系統の `qwen3.5:9b`，検証器は `judge_model`＝Swallow 8B．**wafl-ctrl5 限定**
（config.yml 絶対条件 B））:

```
# 第1パス（45ペア x 12件生成 -> 検証 -> 各ペア7件に切り詰め；16ペアが目標未達で270/315行）
uv run python -m scripts.generate_compound_eval_questions \
    --generator-model qwen3.5:9b \
    --verifier-model schroneko/llama-3.1-swallow-8b-instruct-v0.1:q4_k_m \
    --ollama-host 127.0.0.1 --ollama-port 11499 \
    --per-pair 12 --target-per-pair 7 \
    --output data/compound_questions_generated.jsonl \
    --multidomain-dedup-file data/classifier_train_multidomain.jsonl \
    --multidomain-dedup-file data/classifier_train_multidomain_iter61.jsonl \
    --multidomain-dedup-file data/classifier_train_multidomain_iter64.jsonl \
    --multidomain-dedup-file data/classifier_train_multidomain_iter65.jsonl

# 第2パス（不足16ペアだけを対象に、generate_all_rows()の同一実装をペア単体で再呼び出し。
# コード変更なし。近重複参照に第1パスの採択済み315件を含める。journal.md Iter78「実装・実験」節参照）
# -> 45行を追加生成し、第1パスの270行とマージして315行を確定。

uv run python build_dataset.py \
    --output data/dataset.jsonl \
    --classifier-train-output data/classifier_train.jsonl
```

| ファイル | sha256 | 行数 |
|---|---|---|
| `data/dataset.jsonl` | `2d4397542e67d71cec1d447ceb8f9636f67af254e885bd7e0eafd309a81d2f0c` | 1915 |
| `data/compound_questions_generated.jsonl` | `1bfb5add6a5f4028c3ceb52531987aaea595c409bfaca1c98bbf48c80901d3c6` | 315 |
| `data/classifier_train.jsonl` | `eb89bf7b0ad6303d41f2b668549f85362988de1eaee7b4faf98b3d3f5edcd9ef` | 1427（無変更） |

出典: JMMLU（`nlp-waseda/JMMLU`, commit `3637b25e444ccfdcde4d23a783cbe8e674faa01b`）．ライセンス CC BY-NC-ND 4.0．
複合設問の LLM 生成部分（`compound-101` 以降）は本リポジトリの生成・独立検証パイプラインの出力であり，
JMMLU 由来ではない．

## E6 教師あり分類器

生成コマンド（実機の ollama ノードが必要．scripts/train_domain_classifier.py の docstring 参照．
2026-09-27 以降は Iteration 80 の G0 ゲート不合格（後述）により，`embedding_model_replacement` の
値は計画時点の `qwen3_embedding_4b` ではなく `bge_m3` へ自動切替され（backlog B126），
埋め込みモデルが `bge-m3` へ変更されている．**wafl-ctrl5 限定**（config.yml 絶対条件 B））:

```
uv run python -m scripts.train_domain_classifier \
    --train-data data/classifier_train.jsonl \
    --embedding-model bge-m3 \
    --ollama-host 127.0.0.1 --ollama-port 11499 \
    --output models/domain_classifier.joblib
```

| ファイル | sha256 |
|---|---|
| `models/domain_classifier.joblib`（現行，bge-m3，1024次元） | `37d71b6339ba3a16d6ce6ff0183fd92b9349776a154dfbb8431b5d6026f29b12` |
| `models/domain_classifier_pre_iter80_qwen3_0.6b.joblib`（Iter80 直前の退避，qwen3-embedding:0.6b，1024次元） | `21e16ec63db89f4c8435264e98e6a704382aec723548359f6a74f44753f6ca19` |
| `models/domain_classifier_pre_iter79_nomic.joblib`（Iter79 直前の退避，nomic-embed-text，768次元） | `02caf2b8e7a85972ff47867f05c8145e7d985e2000a2eb55662c6285db408905` |

オフライン性能（旧・nomic-embed-text，docs/d0002 §6-E）: 訓練 100.00%（1427/1427），評価 59.87%（898/1500，
1500 行版データセット当時の実測）．過学習の傾向が残る．

オフライン性能（qwen3-embedding:0.6b，Iteration 79 実装フェーズ実測）: 訓練 84.09%（1200/1427），
評価（argmax `selected_domain` が `expected_domains` に含まれる比率，dispatch/aggregation を経ない
分類器単体の値，dataset.jsonl 全 1915 行）75.35%（1443/1915）．旧モデルより訓練精度が下がっている
一方，評価精度は大きく上回っており，過学習が緩和された可能性がある（判定は分析・考察フェーズに委ねる）．

オフライン性能（新・bge-m3，Iteration 80 実装フェーズ実測）: 訓練 100.00%（1427/1427，p≫n による
完全適合），評価（同定義，dataset.jsonl 全 1915 行）72.48%（1388/1915）．qwen3-embedding:0.6b の
75.35% を下回る着地点予測（判定には用いない．本走の実測は分析・考察フェーズで確定する）．

G0（Iteration 80 VRAM 実測ゲート，wafl-ctrl5，`ollama ps`／`nvidia-smi`）: `qwen3-embedding:4b` を
`docker exec ollama-ctrl ollama stop qwen3-embedding:0.6b` 等で枠を空けたうえで pull・ロードし，
常駐サイズ 4.4GB・`100% GPU` を実測．依頼者ノード wafl500 のピーク予算（12288 MiB -
`light_model 3.1GB` - `expert_model 5.3GB` ≈ 3.4GB，合格条件 X+8.4GB≤11.5GB）を 1.3GB 超過し
**G0 不合格**．代替候補 `bge-m3` は常駐サイズ 0.664GB・`100% GPU` で合格（9.06GB ≤ 11.5GB）．
詳細と選定の経緯は backlog B126 参照．

G1（Iteration 80 事前スクリーニング，`data/classifier_train.jsonl` のみの 5-fold StratifiedKFold CV，
`scripts/screen_embedding_models.py`，3 候補同時評価）: `qwen3-embedding:0.6b`（基準線）
cv_accuracy=0.7561（macro-F1 0.7562），`qwen3-embedding:4b` cv_accuracy=0.7722（macro-F1 0.7718，
3 候補中最良だが G0 不合格のため選定対象外），`bge-m3` cv_accuracy=0.7120（macro-F1 0.7140）．
選定規則（G0 を通過した非現行候補のうち CV accuracy 最大）により `bge-m3` を選定．

G2（旧 qwen3-embedding:0.6b／新 bge-m3 artifact の argmax replay，dataset.jsonl 全 1915 行，
conformal 無し，`scripts.evaluate_classifier_calibration`）: discordant 行数 n_d=469
（必要偏り率 `1.96/sqrt(469)`=0.0905）．合格ライン n_d≥30 を満たす．

参考（Iteration 79 の G1/G2，埋め込みモデル nomic-embed-text→qwen3-embedding:0.6b）:
G1: `nomic-embed-text` cv_accuracy=0.5711（macro-F1 0.5710），`qwen3-embedding:0.6b`
cv_accuracy=0.7561（macro-F1 0.7562）．G2: discordant 行数 n_d=787（必要偏り率
`1.96/sqrt(787)`=0.0699）。

## E10 ドメイン別 LoRA アダプタ

生成は3段階（各スクリプトの docstring 参照）:

```
# 1. JMMLU から評価データセットと分離した instruction-tuning データを作成
uv run python scripts/prepare_lora_training_data.py \
    --domains <domain> \
    --output-dir data/lora_train \
    --eval-dataset data/dataset.jsonl \
    --jmmlu-zip /path/to/JMMLU.zip

# 2. ドメインごとに LoRA アダプタを訓練（10 ノード並列で wall-clock 2〜4 時間，docs/d0002 §3-3 Iter18）
uv run python scripts/train_domain_lora.py \
    --model schroneko/llama-3.1-swallow-8b-instruct-v0.1 \
    --data data/lora_train/<domain>.jsonl \
    --output models/lora_adapters/<domain>/ \
    --lora-r 16 --lora-alpha 32 --epochs 3 --batch-size 2

# 3. Ollama へアダプタを登録（ノードごとに実行）
uv run python scripts/create_lora_model.py \
    --base schroneko/llama-3.1-swallow-8b-instruct-v0.1:q4_k_m \
    --adapter models/lora_adapters/<domain>/ \
    --name expert-mesh-<domain>-lora \
    --ollama-host <node_ip> --ollama-port 11434
```

`models/lora_adapters/<domain>/adapter.gguf`（Ollama の `ADAPTER` ディレクティブが実際に参照するファイル）の sha256:

| domain | sha256（adapter.gguf） |
|---|---|
| general | `c5f16bc4c4a93cf0ada78b5ba21405e724722cd946b13ca9073453f861cdf9e1` |
| education | `f3f43b93b0f56da95441782fcba38ac69f3853cbcb1468c4087140ceba82fdb5` |
| legal | `e75c92b0313a103e5464871c9d62375f0fc99dae2254c9c50ed8b5e7d8716a33` |
| medical | `6b5184fd08fdf8ea34f264a80e36fc629b58a448146805c2dad67caa8e2797b1` |
| business_economics | `6e94180f5a0b6554e604f7de65bacfcc76543e071aefdb7713d796f70f220bdb` |
| computer_science | `16b79141ece66a21ff8c65ad953c218b790474ff757fd9049ffc4b0d6da73a37` |
| natural_science | `4b8ea3b30ce481b8efedf7c828b7c646f99077e32f747f1e7b6e98366a9eca7d` |
| mathematics | `958e406025dd05537dab390cb6fd44d5f4815e804bac69436674d286f61d46bc` |
| history_culture | `631742d429a153e78d34d11b7e93cb2785f65c7ff5f3b8a6f91f6884b1f622c7` |
| social_science | `cdaffed7515fe4ffcec110d88da95cca2a30b4fc193e9f987cff16c4a8bc0ff1` |

2026-07-29 時点で，wafl500〜509 の Ollama に上記10種が全て登録済みであることを実機（`docker compose exec
ollama ollama list`）で確認済み．

## 記録日

2026-07-29．`git rev-parse HEAD` = `30e3627020c986dfd24a3b0a4c0cdd26d1136b85`（本ファイル作成時点）．
以降にこれらのファイルを再生成した場合は，このセクションと各ハッシュを更新すること．
