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

| ファイル | sha256 | 行数 |
|---|---|---|
| `data/dataset.jsonl` | `485a85f522bbf304f8abf28d4955315d175475d5a68b3c8e8007f6571f1d40e9` | 1600 |
| `data/classifier_train.jsonl` | `eb89bf7b0ad6303d41f2b668549f85362988de1eaee7b4faf98b3d3f5edcd9ef` | 1427（無変更） |

出典: JMMLU（`nlp-waseda/JMMLU`, commit `3637b25e444ccfdcde4d23a783cbe8e674faa01b`）．ライセンス CC BY-NC-ND 4.0．

## E6 教師あり分類器

生成コマンド（実機の ollama ノードが必要．scripts/train_domain_classifier.py の docstring 参照）:

```
uv run python -m scripts.train_domain_classifier \
    --train-data data/classifier_train.jsonl \
    --embedding-model nomic-embed-text \
    --ollama-host 192.168.15.100 \
    --output models/domain_classifier.joblib
```

| ファイル | sha256 |
|---|---|
| `models/domain_classifier.joblib` | `3a5610aa88d70b9e94af4620d2747b313c52b834a9dbaa5e872ed45c3520dcb0` |

オフライン性能（docs/d0002 §6-E）: 訓練 100.00%（1427/1427），評価 59.87%（898/1500）．過学習の傾向が残る．

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
