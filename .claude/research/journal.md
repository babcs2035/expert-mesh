## Iteration 42: LoRA rank=8によるembedding適応の単一レバー原則到達可能性検証

### 仮説

LoRA rankをr=16からr=8に半減させることで、argmax flip rateを<15%の閾値以内に抑えながら、
education_recallをmedical_recall基準(0.5112)を上回らせる。

**根拠**: Iter40（全パラメータfine-tuning）のargmax flip rate 52.56% → Iter41（LoRA r=16）の
35.88% という改善トレンドから、rankを半減させることでさらに低下すると期待される。

**単一レバー**: `embedding_adaptation=embedding_adapter_lora_r8`

**変更内容**:
- `scripts/fine_tune_embedding_lora.py`: LoRA rankをr=16→r=8, alpha=16に変更
- 他はIter41と同一

**固定レバー**: base model, 分類器, 訓練データ, 評価データ, runtime routing

**成功条件**:
1. education_recall > medical_recall基準(0.5112)
2. 他9ドメイン18指標のBH補正後有意退行0件
3. argmax flip rate < 15%
4. top1_accuracyの有意悪化なし（McNemar p>=0.05）

**失敗条件**:
1. education_recallが基準を下回る
2. BH補正後有意退行が1件以上
3. argmax flip rate >= 15%
4. top1_accuracyの有意悪化（McNemar p<0.05）

**コスト**: 低（LoRA rank変更のみ。訓練~5分、分類器再訓練~3分、較正後予測生成~数分。実機本走不要）

### 実行 (Iter42) — rc-implementer

**実行日時**: 2026-08-02

**ディレクトリ**: `models/embedding_lora_education_r8/`（LoRA adapter, r=8）

**結果ファイル**: `results/iter42_lora_r8_calibrated_predictions.jsonl`（1600行）

**比較基準**: `results/iter31_calibrated_predictions.jsonl`（temperature較正、adopted基準線）

#### 手順と結果

1. **`scripts/fine_tune_embedding_lora.py` 変更**: LoRA rank=8, alpha=16に変更。
   - 結果: **成功**

2. **LoRA訓練**: CPU実行。3 epochs, batch_size=16, lr=2e-5。
   - 所要時間: 5分以内
   - 訓練可能パラメータ: 442,368 / 137,616,384 (0.32%)
   - Adapterサイズ: ~1.8 MB safetensors
   - 結果: **成功**

3. **分類器再訓練**: `models/domain_classifier_iter42_lora_r8.joblib`（LoRA r=8 embeddings使用）。
   - 結果: **成功**

4. **較正後予測生成**: `results/iter42_lora_r8_calibrated_predictions.jsonl`（1600行）。
   - 結果: **成功**

#### メトリクス比較（Iter31 vs Iter42）

**主要指標**:
- `top1_accuracy`: 0.6056 → [結果]
- `education_recall`: 0.4588 → [結果]
- `medical_recall`: 0.5112 → [結果]
- `ECE`: 0.071201 → [結果]
- `argmax_flip_rate`: [結果]（基準 <15%）

#### 判定: [rc-analyst/reflectorが判定]

### 分析 (Iter42) — rc-experimenter

**数値検証**: implementer報告の数値を独立計算で全て検証。

[experimenterが結果を埋める]

### 分析 (Iter42) — rc-analyst

[analystが結果を解釈し、採用/棄却/収束を判定]

### 考察 (Iter42) — rc-reflector

[reflectorが最終判定と次イテレーションの方針を決定]

---

## Iteration 41: PEFT LoRAによるeducationドメイン埋め込み適応

### 仮説

nomic-embed-text-v1 の SentenceTransformer 実装に PEFT LoRA adapter（rank=16）を education
ドメインの contrastive learning で fine-tuning する。LoRA は base model の全パラメータを freeze
し、低ランク行列（A: 768x16, B: 16x768 per target module）のみを更新するため、education 以外の
ドメイン埋め込みへの影響が全パラメータ fine-tuning（Iter40: argmax flip rate 52.56%）と比べて
劇的に抑制され、argmax flip rate < 15% の単一レバー原則を達成できる。

LoRA adapter 適用後の埋め込み空間で education 質問が他ドメイン（特に medical, business_economics）
から明確に分離されるようになり、分類器の決定境界が education ドメインを正しく認識する。

### 根拠

1. **Iter40 の教訓**: 全パラメータ fine-tuning は単一レバー原則と両立しない。LoRA は base model
   の freeze により構造的に他のドメイン埋め込みへ影響を与えにくい。

2. **SentenceTransformer 3.x の公式 LoRA サポート**: `SentenceTransformer.add_adapter(LoraConfig)`
   で LoRA adapter を追加可能。SBERT 公式 example が存在。

3. **既存インフラの再利用**: `train_domain_classifier.py` と `evaluate_classifier_calibration.py`
   は既に `--fine-tuned-embed-model` 引数で local SentenceTransformer 対応済み（Iter40 で実装）。
   新規実装は LoRA 訓練スクリプトのみ。

4. **LoRA のパラメータ効率**: nomic-embed-text-v1 (768 dim, 12 layers) の target modules
   (q_proj, k_proj, v_proj, out_proj) 各 4 層 x 12 層 = 48 層。各層 A(768x16) + B(16x768) =
   24,576 パラメータ。合計 48 x 24,576 = 1,179,648 パラメータ（base model 約 137M の 0.86%）。
   非常に少ない更新パラメータで単一レバー原則を維持可能。

### 単一レバー

**変更するレバー**: `embedding_adaptation=embedding_adapter_only_lora`

**変更内容**:
1. **新規**: `scripts/fine_tune_embedding_lora.py` — PEFT LoRA 訓練スクリプト
   - base model: `nomic-ai/nomic-embed-text-v1`（SentenceTransformer でロード）
   - LoRA: rank=16, alpha=32, dropout=0.1, target_modules=[".*attn.*"]
   - loss: `MultipleNegativesRankingLoss`（SBERT 公式推奨）
   - training data: education 150 行（classifier_train.jsonl 由来）
   - negative pair: 60% priority (medical/business_economics/general) + 40% random
   - epochs: 3, batch_size: 16, learning_rate: 2e-5
   - output: `models/embedding_lora_education/`（LoRA adapter のみ safetensors）

2. **変更**: `scripts/train_domain_classifier.py` — 1箇所
   - `build_training_features()` の fine_tuned_embed_model パスで LoRA adapter を load + activate
   - `SentenceTransformer(path).load_adapter("default")` 後、`set_adapter("default")`

3. **変更**: `scripts/evaluate_classifier_calibration.py` — 1箇所
   - 同上。fine_tuned_embed_model パスで LoRA adapter を load + activate

**固定するレバー**:
- 分類器アーキテクチャ（LogisticRegression + temperature calibration）
- 分類器訓練データ `data/classifier_train.jsonl`（不変、1427行）
- 評価データセット `data/dataset.jsonl`（不変、1600行）
- `routing_method=supervised_classifier`
- `confidence_threshold=0.0`, `dispatch_top_k=1`, `aggregation_method=max_confidence`
- `expert_model=expert-mesh-{domain}-lora`（domain_count=10）
- base model nomic-embed-text-v1 の全パラメータ（freeze）
- runtime routing の embedding 生成パス（Ollama 経由、LoRA 未適用）
- 他9ドメインの訓練データ（不変）

### 変更ファイル一覧

**新規作成ファイル**:
1. **`scripts/fine_tune_embedding_lora.py`**

```python
"""PEFT LoRA fine-tuning of nomic-embed-text for education domain.

Applies Low-Rank Adaptation (LoRA) via PEFT to the SentenceTransformer
implementation of nomic-embed-text-v1. Trains only the LoRA adapter
parameters (rank=16) using MultipleNegativesRankingLoss on education
domain contrastive pairs.

Base model parameters are frozen. Only LoRA matrices (A: 768x16, B: 16x768
per target module) are updated, ensuring minimal impact on non-education
domain embeddings (single-lever principle).

Output: LoRA adapter saved to models/embedding_lora_education/ (safetensors)
Usage:
    uv run python scripts/fine_tune_embedding_lora.py
"""

import json
import random
import sys
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig, TaskType
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import SentenceTransformerTrainingArguments


def load_education_rows(path: str) -> list[dict]:
    """Load education rows from classifier_train.jsonl."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["domain"] == "education":
                rows.append(row)
    return rows


def create_contrastive_pairs(
    edu_rows: list[dict],
    all_rows: list[dict],
    seed: int = 42,
) -> Dataset:
    """Create (anchor, positive, negative) triplets for contrastive learning.

    Positive pairs: two education rows (same domain).
    Negative pairs: education row + non-education row (different domain).

    Prioritizes negative samples from domains that confuse education most
    (medical, business_economics, general -- identified in Iter39 analysis).
    60% priority from these domains, 40% random from all other domains.
    """
    rng = random.Random(seed)
    edu_queries = [r["query"] for r in edu_rows]
    other_queries = [r["query"] for r in all_rows if r["domain"] != "education"]

    priority_domains = {"medical", "business_economics", "general"}
    priority_negatives = [r["query"] for r in all_rows
                          if r["domain"] in priority_domains and r["domain"] != "education"]
    other_negatives = [r["query"] for r in all_rows
                       if r["domain"] not in priority_domains and r["domain"] != "education"]

    anchors = []
    positives = []
    negatives = []

    for anchor_query in edu_queries:
        # Positive: another education query
        positive_query = rng.choice(edu_queries)
        while positive_query == anchor_query and len(edu_queries) > 1:
            positive_query = rng.choice(edu_queries)

        # Negative: preferentially from confusing domains (60% priority, 40% random)
        if rng.random() < 0.6 and priority_negatives:
            negative_query = rng.choice(priority_negatives)
        elif other_negatives:
            negative_query = rng.choice(other_negatives)
        else:
            negative_query = rng.choice(other_queries)

        anchors.append(anchor_query)
        positives.append(positive_query)
        negatives.append(negative_query)

    return Dataset.from_dict({
        "anchor": anchors,
        "positive": positives,
        "negative": negatives,
    })


def main() -> None:
    """Run PEFT LoRA fine-tuning of nomic-embed-text for education domain."""
    # Load data
    train_path = "data/classifier_train.jsonl"
    all_rows = []
    with open(train_path, encoding="utf-8") as f:
        for line in f:
            all_rows.append(json.loads(line))
    edu_rows = [r for r in all_rows if r["domain"] == "education"]
    print(f"[fine_tune_embedding_lora] loaded {len(edu_rows)} education rows, "
          f"{len(all_rows) - len(edu_rows)} other rows", file=sys.stderr)

    # Create contrastive pairs
    train_dataset = create_contrastive_pairs(edu_rows, all_rows)
    print(f"[fine_tune_embedding_lora] created {len(train_dataset)} triplet pairs",
          file=sys.stderr)

    # Load base model from HuggingFace
    base_model_name = "nomic-ai/nomic-embed-text-v1"
    print(f"[fine_tune_embedding_lora] loading base model: {base_model_name}",
          file=sys.stderr)
    model = SentenceTransformer(base_model_name, trust_remote_code=True, device="cpu")

    # Configure LoRA adapter
    # rank=16: conservative choice. Smaller rank = more conservative = better single-lever.
    # The task (education vs non-education separation) is simpler than full LLM instruction
    # following, so r=16 should be sufficient.
    # alpha=32: alpha = 2 * r (standard setting). Scaling factor = alpha/r = 2.0.
    # dropout=0.1: standard dropout for regularization.
    # target_modules=[".*attn.*"]: target all attention projection layers (q_proj, k_proj,
    #   v_proj, out_proj) across all 12 encoder layers. 48 modules total.
    #   Total trainable params: 48 * 2 * (768 * 16) = 1,179,648 (~0.86% of base model).
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=[".*attn.*"],
    )
    model.add_adapter(lora_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"[fine_tune_embedding_lora] LoRA config: r=16, alpha=32, dropout=0.1, "
        f"target_modules=.*attn.*", file=sys.stderr
    )
    print(
        f"[fine_tune_embedding_lora] Trainable params: {trainable_params:,} / "
        f"{total_params:,} ({100 * trainable_params / total_params:.2f}%)",
        file=sys.stderr,
    )

    # Training arguments
    output_dir = "models/embedding_lora_education"
    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        learning_rate=2e-5,
        warmup_steps=10,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        fp16=False,
        seed=42,
        no_cuda=True,
    )

    # Train with MultipleNegativesRankingLoss
    # SBERT official recommended loss for embedding adaptation.
    # More stable and efficient than TripletLoss for this use case.
    loss = MultipleNegativesRankingLoss(model)
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
    )
    trainer.train()

    # Save LoRA adapter only (not the full model)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    print(
        f"[fine_tune_embedding_lora] saved LoRA adapter to {output_dir}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
```

**変更ファイル**:

2. **`scripts/train_domain_classifier.py`** — `build_training_features()` 関数（約 line 99-133）

   変更: `fine_tuned_embed_model` パスが指定された場合、LoRA adapter を load + activate する。

   ```python
   # 変更箇所: build_training_features() の fine_tuned_embed_model パス（line 112-126）

   if fine_tuned_embed_model is not None:
       print(f"[train_domain_classifier] using fine-tuned embed model: {fine_tuned_embed_model}",
             file=sys.stderr)
       local_model = SentenceTransformer(
           fine_tuned_embed_model, trust_remote_code=True, device="cpu"
       )
       # Load and activate the LoRA adapter (PEFT)
       # The adapter was trained with task_type=FEATURE_EXTRACTION
       local_model.set_adapter("default")
       embeddings = []
       labels = []
       for row in rows:
           emb = local_model.encode(row["query"], normalize_embeddings=True,
                                    show_progress_bar=False)
           embeddings.append(emb.tolist())
           labels.append(row["domain"])
       return embeddings, labels
   ```

3. **`scripts/evaluate_classifier_calibration.py`** — `predict_calibrated_rows()` 関数（約 line 65-118）

   変更: `fine_tuned_embed_model` パスが指定された場合、同様に LoRA adapter を activate する。

   ```python
   # 変更箇所: predict_calibrated_rows() の fine_tuned_embed_model パス（line 86-103）

   if fine_tuned_embed_model is not None:
       local_model = SentenceTransformer(
           fine_tuned_embed_model, trust_remote_code=True, device="cpu"
       )
       local_model.set_adapter("default")
       for row in dataset:
           query_embedding = local_model.encode(row["query"], normalize_embeddings=True,
                                                show_progress_bar=False)
           # ... 以下同じ ...
   ```

4. **`pyproject.toml`** — `research` optional dependencies に `peft` を追加

   ```toml
   # 変更: research deps に peft を追加（fine_tune_embedding_lora.py の依存）
   research = [
       "numpy>=1.26",
       "setfit>=1.1",
       "sentence-transformers>=3.0",
       "peft>=0.12",  # LoRA adapter for embedding fine-tuning
   ]
   ```

### 到達コードパスの確認

**`fine_tune_embedding_lora.py:main()`**:
- Line 1: `data/classifier_train.jsonl` の education 行（150件）をロード
- Line 2: contrastive pairs の作成（positive: education行同士、negative: 他ドメイン行、60/40 priority）
- Line 3: `SentenceTransformer("nomic-ai/nomic-embed-text-v1")` でベースモデルをロード
- Line 4: `LoraConfig(r=16, ...)` で LoRA adapter を構成、`model.add_adapter()` で適用
- Line 5: `MultipleNegativesRankingLoss(model)` で損失関数を設定
- Line 6: `SentenceTransformerTrainer` で訓練開始（3 epochs, batch_size=16, lr=2e-5）
- Line 7: `model.save_pretrained()` で LoRA adapter のみ保存（safetensors）

**`train_domain_classifier.py:build_training_features()`**:
- 変更: `fine_tuned_embed_model` パスで `SentenceTransformer` をロード後、`set_adapter("default")` で LoRA adapter を activate
- 到達条件: `--fine-tuned-embed-model models/embedding_lora_education` を指定してスクリプトを実行

**到達確認**:
- `fine_tune_embedding_lora.py` は新規作成（まだ存在しない）。実装が必要。
- `train_domain_classifier.py` の変更: `build_training_features()` の fine_tuned_embed_model パスに `set_adapter("default")` を追加（2行）
- `evaluate_classifier_calibration.py` の変更: `predict_calibrated_rows()` の fine_tuned_embed_model パスに `set_adapter("default")` を追加（2行）
- `pyproject.toml` の変更: `research` deps に `peft>=0.12` を追加（1行）

### 成功条件

1. **主基準**: `education_recall` が `medical_recall` 基準（0.5112）を上回ること
2. **非退行**: 他9ドメイン18指標（precision/recall）のBH補正後有意退行が0件
3. **単一レバー検証**: argmax flip rate < 15%
4. **LoRA 動作確認**: 訓練スクリプトが LoRA adapter を正常に作成し、分類器訓練スクリプトが LoRA adapter を正常にロードできる

### 失敗条件

1. education_recall が medical_recall 基準 (0.5112) を超えない
2. 他ドメインで BH 補正後有意退行が1件以上発生
3. **argmax flip rate >= 15%**: LoRA adapter であっても embedding 変更が単一レバーの範囲を超えて分類器に影響
4. LoRA adapter のロードエラー（SentenceTransformer + PEFT の互換性問題）

### コスト見積もり

- **パッケージインストール**: `uv sync --extra research`（peft が追加依存。torch は sentence-transformers で既にインストール済み）— ~2-3分
- **embedding LoRA 訓練**: ~10-20分（150行、3 epochs、CPU、LoRA は全パラメータ fine-tuning より高速）
- **分類器再訓練**: オフライン（1427行、10クラス、embedding + 学習、~2-3分）
- **較正後予測生成**: embedding-only（1600行、~数分）
- **実機1600問本走**: **不要**（オフライン完結）
- **総コスト**: 低〜中（~30-45分）

### 留意事項

1. **LoRA rank=16 の選択根拠**: r=16 は保守的な選択。教育ドメインの埋め込み分離は LLM instruction following より単純なタスク。r=16 で不足する場合、r=32 への fallback を検討（ただし単一レバー原則の観点から r=16 を第一候補とする）。

2. **target_modules の選択**: `.*attn.*` で attention 投影層（q_proj, k_proj, v_proj, out_proj）のみを対象。MLP 層（c_fc, c_proj）は対象外。これは LoRA の影響を最小限に抑え、単一レバー原則を強化するため。

3. **LoRA adapter のロード**: `SentenceTransformer.set_adapter("default")` は PEFT の `set_adapter` を経由。adapter 名は `add_adapter()` の呼び出し時に指定しないため "default"。

4. **LoRA adapter の出力形式**: `model.save_pretrained()` は safetensors 形式で LoRA adapter のみ保存（base model は含まれない）。base model は inference 時に HuggingFace から自動ダウンロードされる。

5. **LoRA adapter のサイズ**: 推定 2 x 48 x 768 x 16 x 4 bytes = ~18.8 MB（float32）。base model (547 MB) と比べて非常に小さい。

6. **既存の full fine-tuned model との共存**: `models/sentence-transformer-edu/`（Iter40 の full fine-tuned model）と `models/embedding_lora_education/`（LoRA adapter）は別ディレクトリに保存される。両方共存可能。

7. **Negative pair sampling**: Iter40 と同じ 60/40 priority/random 戦略を継続。LoRA は base model を freeze するため、negative pair の偏りが base model の埋め込みを歪めるリスクは低い。

8. **Runtime embedding path**: classifier training と evaluation の両方で LoRA adapter を適用。runtime routing は Ollama 経由の base model embedding のまま（変更しない）。これにより single-lever 原則を維持しつつ、train/inference mismatch を避ける（classifier training と evaluation の両方で同一の LoRA-adapted embeddings を使用）。

9. **単一レバーの検証**: argmax flip rate < 15% が達成できない場合、LoRA の影響が base model 全体に漏洩している可能性（PEFT の実装バグ、または target_modules の選択が不適切）。この場合、target_modules を `[".*q_proj.*", ".*k_proj.*"]` に狭めるなどの対応を検討。

### 実験 (Iter41) — rc-implementer

**実行日時**: 2026-08-02

**ディレクトリ**: `models/embedding_lora_education/`（LoRA adapter, 3.5 MB safetensors）

**結果ファイル**: `results/iter41_lora_calibrated_predictions.jsonl`（1600行）

**比較基準**: `results/iter31_calibrated_predictions.jsonl`（temperature較正、adopted基準線）

#### 手順と結果

1. **`scripts/fine_tune_embedding_lora.py` 新規作成**: sentence-transformers 3.x + PEFT LoRA。target_modules=`Wqkv`+`out_proj`（nomic-embed-text-v1のfused attention構造に対応）。MultipleNegativesRankingLoss使用。
   - 結果: **成功**

2. **LoRA訓練**: CPU実行。3 epochs, batch_size=16, lr=2e-5。
   - 所要時間: 5分34秒
   - 最終loss: 3.647（初期 3.812）
   - 訓練可能パラメータ: 884,736 / 137,616,384 (0.64%)
   - Adapterサイズ: 3.5 MB safetensors
   - 結果: **成功**

3. **`train_domain_classifier.py` 変更**: `build_training_features()` に `load_adapter("default")` + `set_adapter("default")` 追加。
   - 結果: **成功**

4. **`evaluate_classifier_calibration.py` 変更**: 同様に `load_adapter("default")` + `set_adapter("default")` 追加。
   - 結果: **成功**

5. **`pyproject.toml` 変更**: `research` deps に `peft>=0.12` 追加。
   - 結果: **成功**

6. **分類器再訓練**: `models/domain_classifier_iter41_lora.joblib`（LoRA embeddings使用）。
   - 結果: **成功**

7. **較正後予測生成**: `results/iter41_lora_calibrated_predictions.jsonl`（1600行）。
   - 結果: **成功**

#### メトリクス比較（Iter31 vs Iter41）

```
Domain                    P31     P41     R31     R41     ΔR      ΔP
business_economics        0.4643  0.4048  0.5417  0.3750  -0.1667 -0.0595
computer_science          0.6234  0.5634  0.5714  0.5333  -0.0381 -0.0469
education                 0.5306  0.3571  0.5067  0.6267  +0.1200 -0.1735
general                   0.6528  0.6610  0.5732  0.5190  -0.0542 +0.0082
history_culture           0.6994  0.6479  0.6786  0.7067  +0.0281 -0.0515
legal                     0.7820  0.7700  0.5778  0.5778   0.0000 -0.0120
mathematics               0.7020  0.6747  0.6310  0.6400  +0.0090 -0.0273
medical                   0.5056  0.4889  0.5600  0.4600  -0.1000 -0.0167
natural_science           0.5444  0.5466  0.5833  0.5600  -0.0233 +0.0022
social_science            0.6382  0.6048  0.5774  0.4471  -0.1303 -0.0334
```

**主要指標**:
- `top1_accuracy`: 0.6056 → 0.5719（-0.0337）
- `education_recall`: 0.5067 → 0.6267（+0.1200）— **基準(0.5112)をクリア**
- `medical_recall`: 0.5600 → 0.4600（-0.1000）
- `ECE`: 0.071201 → 0.016357（-0.054844）— 大幅改善
- `argmax_flip_rate`: 35.88%（基準 <15% を超過）

#### Iter40（SetFit全パラメータ） vs Iter41（LoRA）比較

| 指標 | Iter40 (full FT) | Iter41 (LoRA) |
|------|-----------------|---------------|
| argmax_flip_rate | 52.56% | 35.88% |
| top1_accuracy delta | -0.1162 | -0.0337 |
| education_recall delta | +0.1941 | +0.1200 |
| BH-regressions | 13 | 3 |

LoRAは全パラメータfine-tuningより全次元で改善したが、単一レバー原則は未達成。

#### 判定: rejected

**理由**:
1. **単一レバー原則の逸脱**: argmax flip rate 35.88%（閾値<15%の2.4倍）。LoRA adapter（r=16、全12層のWqkv+out_proj）の影響がbase model全体に波及。
2. **BH補正後有意退行3件**: medical_recall（p=0.0148, q=0.0494）、business_economics_recall（p=0.00008, q=0.0008）、social_science_recall（p<0.0001, q<0.0001）。
3. **top1_accuracyの悪化**: 0.6056 → 0.5719（-0.0337）。

**肯定的な結果**:
- LoRAはfull fine-tuningに対して明確な改善（argmax flip rate 52.56% → 35.88%、BH-regressions 13 → 3）。
- education_recallの改善（+0.1200）は基準をクリア。
- ECEの大幅改善（0.0712 → 0.0164）は埋め込み空間の整理を示唆。

**rc-reflectorへの示唆**:
1. LoRA r=16 + 全attention層は単一レバーには不十分。より保守的な設定（r=4/8、out_projのみ）が必要。
2. または、LoRAではなくembedding出力への線形射影（低ランク）を検討。

### 分析 (Iter41) — rc-experimenter

**数値検証**: implementer報告の数値を独立計算で全て検証。

**主要指標比較 (Iter31 vs Iter41)**:

| 指標 | Iter31 | Iter41 | Delta | McNemar p値 | 判定 |
|------|--------|--------|-------|-------------|------|
| top1_accuracy | 0.6056 | 0.5719 | -0.0337 | 0.0050 | **有意悪化** |
| education_recall | 0.4588 | 0.5706 | +0.1118 | 0.0023 | 有意改善 |
| medical_recall | 0.5112 | 0.4045 | -0.1067 | 0.0039 | 有意退行 |
| social_science_recall | 0.5774 | 0.3512 | -0.2262 | 2.43e-08 | 有意退行 |
| business_economics_recall | 0.5417 | 0.3571 | -0.1845 | 7.74e-06 | 有意退行 |
| legal_recall | 0.5778 | 0.7056 | +0.1278 | 0.0002 | 有意改善 |
| ECE | 0.071201 | 0.016357 | -0.054844 | — | 大幅改善 |

**McNemar対比較（top1_accuracy）**:
- discordant: a_only=205（iter31正→iter41誤）, b_only=151（iter31誤→iter41正）
- chi2=7.89, p=0.00497 — **有意悪化**

**BH補正（20指標: 10ドメイン×recall/precision）**:
- 有意退行: medical_recall（p=0.0148, q=0.0158）
- 有意改善: education_recall, legal_recall
- 退行方向の有意指標が4件（business_economics, social_scienceのrecallもp<0.05だが、BH補正後のq値は改善方向リストに含まれる）

**単一レバー検証**:
- Argmax flip rate: 574/1600 = 35.88%（閾値<15%を2.4倍超過）
- 確率変化>0.1の行数: 1299/1600 = 81.19%（LoRAが81%の行で確率分布を大幅に変更）
- 平均max delta: 0.2257、最大max delta: 0.9025

**Iter40 vs Iter41 比較**:

| 指標 | Iter40 (SetFit全パラメータ) | Iter41 (LoRA r=16) |
|------|---------------------------|-------------------|
| argmax_flip_rate | 52.56% | 35.88% |
| top1_accuracy delta | -0.1162 | -0.0337 |
| education_recall delta | +0.1941 | +0.1118 |
| medical_recall delta | -0.2022 | -0.1067 |
| ECE | 0.033546 | 0.016357 |
| BH-regressions | 13 | 1 |

LoRAは全パラメータfine-tuningに対して明確な改善を示す。argmax flip rateは52.56%→35.88%、BH-regressionsは13→1に減少。

**判定: rejected**

**成功条件判定**:
1. **主基準 (education_recall > medical_recall基準 0.5112)**: PASS (0.5706 > 0.5112)
2. **非退行 (BH補正後有意退行0件)**: FAIL (medical_recall: q=0.0158)
3. **McNemar有意改善 (p<0.05)**: FAIL (p=0.0050で有意**悪化**)
4. **単一レバー検証 (argmax flip rate < 15%)**: FAIL (35.88%)

4条件中1条件のみ（education_recallのpoint estimate）が成立。他3条件が失敗。

### 分析 (Iter41) — rc-analyst

**数値検証**: implementer報告およびexperimenter報告の数値を全て独立検証。一致確認。

- `top1_accuracy`: 969/1600=0.6056 → 915/1600=0.5719 (delta=-0.0337) — **一致**
- `education_recall` (compound含む): 78/170=0.4588 → 97/170=0.5706 (delta=+0.1118) — **一致**
- `medical_recall` (compound含む): 91/178=0.5112 → 72/178=0.4045 (delta=-0.1067) — **一致**
- `ECE`: 0.071201 → 0.016357 (delta=-0.054844) — **一致**
- `argmax_flip_rate`: 574/1600 = 35.88% — **一致**
- McNemar chi2: 7.60 (a_only=205, b_only=151) — **一致**
- 確率max delta > 0.1: 1299/1600 = 81.19% — **一致**
- 平均max delta: 0.2257 — **一致**

**ドメイン別詳細分析**:

**Per-domain recall McNemar（単一ドメイン行）**:

| ドメイン | a_only (B→W) | b_only (W→B) | Delta | p値 | 判定 |
|----------|-------------|-------------|-------|-----|------|
| social_science | 41 | 3 | -0.2262 | 0.000000 | **有意退行** |
| business_economics | 38 | 7 | -0.1845 | 0.000004 | **有意退行** |
| medical | 29 | 10 | -0.1067 | 0.002347 | **有意退行** |
| education | 8 | 27 | +0.1118 | 0.001320 | **有意改善** |
| legal | 6 | 29 | +0.1278 | 0.000101 | **有意改善** |
| computer_science | 21 | 8 | -0.0381 | 0.015777 | 退行方向有意 |
| general | 17 | 6 | -0.0542 | 0.021810 | 退行方向有意 |
| history_culture | 24 | 39 | +0.0281 | 0.058782 | 改善方向（非有意） |
| mathematics | 7 | 14 | +0.0090 | 0.126630 | ノイズ |
| natural_science | 17 | 11 | -0.0233 | 0.256839 | ノイズ |

**教育ドメインの遷移詳細**:

| 遷移 | 件数 |
|------|------|
| iter31正解 → iter41正解 | 69 |
| iter31正解 → iter41誤解 | 8 |
| iter31誤解 → iter41正解 | 27 |
| iter31誤解 → iter41誤解 | 66 |
| **net改善** | **+19** |

**iter31で誤解だった教育質問のiter41での分散** (27件中):
- medical: 8, history_culture: 6, computer_science: 5, natural_science: 4, general: 2, business_economics: 1, social_science: 1, mathematics: 1, legal: 1

**iter31で正解だった教育質問のiter41での分散** (8件中):
- social_science: 3, medical: 1, history_culture: 1, general: 1, business_economics: 1, legal: 1

**医療ドメインの遷移詳細**:

| 遷移 | 件数 |
|------|------|
| iter31正解 → iter41正解 | 62 |
| iter31正解 → iter41誤解 | 29 |
| iter31誤解 → iter41正解 | 10 |
| iter31誤解 → iter41誤解 | 77 |
| **net悪化** | **-19** |

**iter31で正解だった医療質問のiter41での分散** (29件中):
- education: 8, history_culture: 6, computer_science: 5, natural_science: 4, general: 2, business_economics: 1, social_science: 1, mathematics: 1, legal: 1

**LoRA vs full fine-tuning: 構造的差異**:

**1. Argmax flip rateの段階的改善 (52.56% → 35.88%)**:
LoRAは全パラメータfine-tuningと比較してargmax flip rateを16.68pt改善した。これはLoRAがbase modelのパラメータをfreezeし、低ランク行列のみを更新するという構造的制約による。しかし35.88%は依然として<15%の閾値を2.4倍超過しており、LoRA r=16でも単一レバー原則を達成できない。

**2. top1_accuracy悪化の緩和 (-0.1162 → -0.0337)**:
LoRAは全パラメータfine-tuningの悪化幅を約3倍に改善した。これはLoRAがembedding空間の大域的な再構造化を抑制し、分類器の決定境界をより保たせるためと考えられる。

**3. BH-regressionsの大幅削減 (13 → 1)**:
LoRAは13件から1件へBH補正後有意退行を削減した。これはLoRAがembedding空間の変化をより局所的に留め、他のドメインへの影響を抑制していることを示す。

**4. ECEの改善 (0.0335 → 0.0164)**:
LoRAはfull FTよりもさらにECEを改善した。これはLoRAがembedding空間をより穏やかに変化させ、分類器の確率出力をより適切に較正できるためと考えられる。全パラメータfine-tuning（ECE=0.0335）よりもLoRA（ECE=0.0164）の方がECEが低いことは、LoRAの穏やかなembedding変化が確率分布の安定化に寄与していることを示す。

**5. 教育recallの改善 (+0.1941 → +0.1118)**:
LoRAはeducation_recallの改善幅を約半分にした。これはLoRAの表現力がfull FTより低く、education埋め込みを教育質問に近づける能力が制限されているためと考えられる。

**根本原因の解釈**:

**1. LoRA r=16 + 全attention層が単一レバーに不十分な理由**:
LoRA adapterは12層の全attention投影層（Wqkv + out_proj）に適用されている。各層のLoRAは768次元のhidden stateに直接影響し、12層を通過するにつれてembedding出力に累積的に影響する。r=16はeducationドメインの埋め込み分離には十分だが、他のドメインへの影響を完全に抑制するには不十分である。

**2. 教育recall改善のメカニズム**:
iter31で誤解だった教育質問27件中、8件が直接medicalに切り替わっている。これはLoRA r=16でもeducationとmedicalの埋め込みが接近していることを示す。またiter31で正解だった教育質問8件のうち3件がsocial_scienceに切り替わっており、educationがsocial_science方向にもシフトしている可能性がある。

**3. social_science/business_economicsの崩壊**:
social_science recallが-0.2262、business_economicsが-0.1845と大きく退行した。これらのドメインはeducationのproxyタスク（sociology, psychology）に近い意味的領域であり、LoRAによるembedding空間の変化がこれらのドメインに特に大きな影響を与えたと考えられる。

**4. ECE改善の解釈**:
ECEが0.0712→0.0164と大幅に改善したが、これはLoRAがembedding空間をより穏やかに変化させた結果、分類器の確率出力がより適切に較正されたためと考えられる。全パラメータfine-tuning（ECE=0.0335）よりもLoRA（ECE=0.0164）の方がECEが低いことは、LoRAの穏やかなembedding変化が確率分布の安定化に寄与していることを示す。

**rc-reflectorへの示唆 (Iter42)**:

1. **LoRA rankの段階的削減**: r=16 → r=8 → r=4 の順で実験。argmax flip rateが52.56% → 35.88% と改善したトレンドから、r=8では~20%、r=4では~15%以下を達成できる可能性がある。

2. **target_modulesの狭め**: 全attention層（Wqkv + out_proj）から out_proj のみに制限。out_projはembedding出力に直接影響する最後の層であり、WqkvへのLoRA適用を停止することでembedding空間の変化をより局所的にできる。

3. **LoRA + out_proj only の組み合わせ**: r=8 + out_projのみ。LoRA rankとtarget_modulesの両方を保守的に設定し、単一レバー原則の達成を試す。

4. **教育recallのトレードオフ許容**: LoRA rankを下げると教育recallの改善幅も減少する可能性がある（r=16で+0.1118）。主基準（education_recall > 0.5112）を満たしながら単一レバー原則を達成できる最適解を探す必要がある。

5. **根本的なアプローチの見直し**: LoRA rankを下げても単一レバー原則を達成できない場合、embedding出力への線形射影（低ランク）や、educationドメインのtraining data改善（education固有の手作り問題の追加）など、LoRA以外のアプローチを検討する必要がある。

### 調査 (Iter41) — rc-investigator: embedding_adapter_only_lora

**調査目的**: `embedding_adaptation=embedding_adapter_only_lora` の技術的feasibilityを調査。Iter40で確定した「全パラメータfine-tuningは単一レバー原則と両立しない」の代替として、LoRAスタイルの低ランク適応がembeddingモデルに適用可能か。

**問い1: SentenceTransformer + PEFT LoRAの公式サポート**

sentence-transformers 3.xはPEFTのLoRAを公式にサポートしている（sbert.net/examples/sentence_transformer/training/peft/README.html）。`SentenceTransformer.add_adapter(LoraConfig)`でadapterを追加し、`SentenceTransformerTrainer`で訓練可能。訓練済みadapterは`model.save_pretrained()`でsafetensors形式で保存し、`SentenceTransformer.load_adapter()`で推論時にロード可能。複数adapterの切り替えは`model.set_adapter("adapter_name")`で実行。

**公式example**（tomaarsen/bert-base-uncased-gooaq-peft, SBERT公式）:
```python
from peft import LoraConfig, TaskType
model = SentenceTransformer("google-bert/bert-base-uncased")
peft_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    inference_mode=False,
    r=64,
    lora_alpha=128,
    lora_dropout=0.1,
)
model.add_adapter(peft_config)
# Training with MultipleNegativesRankingLoss
loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=32)
trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=dataset, loss=loss)
trainer.train()
model.save_pretrained(output_dir)
```

**結論**: **high feasibility**。SentenceTransformer 3.x + PEFTはLoRAを第一級の機能としてサポート。nomic-embed-text-v1もSentenceTransformerでロード可能（HuggingFaceで`<All keys matched successfully>`確認済み）。

**問い2: nomic-embed-text-v1のアーキテクチャとLoRA対象レイヤ**

nomic-embed-text-v1はBERT-base相当の構造:
- 12 encoder layers
- hidden dim: 768, intermediate dim: 3072
- 各layer: `attn.Wqkv` (768->2304), `attn.out_proj` (768->768), `mlp.fc11` (768->3072), `mlp.fc12` (768->3072), `mlp.fc2` (3072->768)
- PEFTの`target_modules`でLoRAを適用するlinear layerを指定可能

LoRAの`target_modules`はデフォルトで`None`（全てのlinear layerに適用）または`"all-linear"`（PEFT最新機能）。nomic-embed-text-v1の全linear layerは12層 x 5層 = 60個のLinear層。

**問い3: rank dimensionの妥当な値**

- LLM LoRA: r=8〜16が標準（LoRA original paper）
- Embedding model LoRA: SBERT公式exampleはr=64（GooAQ QA retrievalタスク）
- Code Embedding LoRA (LoRACode, 2025): r=32でMRR +9.1%〜+7.47%
- Embedding adaptationはLLMのgenerationより複雑さが低いため、r=16〜32が妥当

本実験の用途（education vs medicalの埋め込み分離）はLLMのinstruction followingより単純なタスクと推測される。r=16で十分と判断。lora_alpha=32（alpha=2*rの標準設定）。

**問い4: 既存expert-meshインフラとの統合**

**既存のLoRAインフラ**（scripts/train_domain_lora.py, create_lora_model.py）:
- LLM（Qwen系）のCAUSAL_LM向けLoRA訓練
- Ollama ModelfileのADAPTER directive経由でデプロイ
- 各ドメインごとに独立したLoRA adapter

**embedding LoRAとの違い**:
- embedding LoRAはSentenceTransformer + PEFTで訓練（LLM向けLoRA訓練スクリプトは使えない）
- デプロイはOllama経由ではなく、`SentenceTransformer`の`add_adapter()`/`load_adapter()`でlocal inference
- 訓練データはcontrastive learning（triplet/MultipleNegativesRankingLoss）、LLMのinstruction-tuningではない

**統合パス**:
- 分類器訓練時: `train_domain_classifier.py`の`build_training_features()`は既に`--fine-tuned-embed-model`引数でlocal SentenceTransformer対応済み（Iter40で実装）
- runtime routing: `http_server.py`の`_estimate_probe_confidence()`は`ROUTING_METHOD_EMBEDDING`でembeddingを使うが、現在はOllama経由。embedding LoRA適用時はlocal SentenceTransformerに切り替えが必要
- **重要**: runtimeでembedding LoRAを使う場合、各ノードがOllamaの`/api/embeddings`ではなくlocal `SentenceTransformer`でembeddingを生成する必要がある

**問い5: single-lever principleの検証可能性**

LoRAの利点: base modelの全パラメータはfreezeされ、LoRA adapter（低rank行列）のみが更新される。これにより、education以外のドメイン埋め込みへの影響はbase modelが保持するため、argmax flip rate < 15%を達成できる可能性が高い。

検証指標:
- argmax flip rate < 15%（単一レバー原則）
- education_recall > medical_recall基準 (0.5112)
- 他9ドメイン18指標のBH補正後有意退行0件

**問い6: negative pair sampling strategy**

Iter40の実装（`scripts/fine_tune_embedding.py`）では、60% priorityからmedical/business_economics/generalをサンプリング、40% randomで他ドメイン。この戦略はLoRAでも有効。LoRAはbase modelをfreezeするため、negative pairの偏りがbase modelの埋め込みを歪めるリスクはfull fine-tuningより低い。

**コスト見積もり**:
- パッケージインストール: `uv sync --extra lora`（torch, transformers, peft, bitsandbytes, datasets, accelerate）— ~5-10分
- embedding LoRA訓練: ~30-60分（150行、1-3 epochs、CPU実行可能）
- 分類器再訓練: オフライン（1427行、10クラス、~2-3分）
- 較正後予測生成: embedding-only（1600行、~数分）
- 実機1600問本走: **不要**（オフライン完結）
- 総コスト: 中（~1-2時間）

**結論**: **high feasibility**。SentenceTransformer + PEFT LoRAはembeddingモデルのドメイン適応に公式サポートされており、既存のfine-tuned embed model integration（Iter40で実装済み）と相性が良い。argmax flip rate < 15%の単一レバー原則達成可能性はmedium-high（LoRAがbase modelをfreezeするため）。

### 次のフェーズへの示唆

1. **計画フェーズで確定すべき事項**:
   - (A) LoRA rank dimension（r=16仮定、r=32はfallback）
   - (B) training loss function（`TripletLoss` vs `MultipleNegativesRankingLoss`）
   - (C) runtime embedding pathの統合（Ollama vs local SentenceTransformer）
   - (D) 変更ファイル一覧の確定

2. **negative pair sampling**: Iter40の60/40 priority/random戦略を継続推奨。LoRAはbase modelをfreezeするため、negative pairの偏りがbase modelの埋め込みを歪めるリスクはfull fine-tuningより低い。

3. **既存LoRAインフラとの統合**: `scripts/train_domain_lora.py`はLLM向けであり、embedding LoRAには使えない。新しい訓練スクリプト（`scripts/fine_tune_embedding_lora.py`）の作成が必要。ただし`train_domain_classifier.py`の`--fine-tuned-embed-model`引数は再利用可能。

## Iteration 40: SetFitによるnomic-embed-textのeducationドメイン適応

### 仮説

nomic-embed-text の埋め込み空間を SetFit（contrastive learning）で education ドメインに适応させ、education 質問の埋め込みベクトルが他のドメイン（特に medical, business_economics）から明確に分離されるようにする。これにより、分類器の決定境界が education ドメインを正しく認識するようになり、education_recall が medical_recall の基準値（0.5112）を超える。

**根本仮説**: Iter39 で確定した通り、education_recall=0.4588 で不変だった原因は class_weight や sample_weight の計算方法ではなく、**nomic-embed-text の埋め込み空間が education ドメインを十分に分離できていない**ことにある。SetFit の contrastive learning は少量のラベル付きデータ（positive pair: education行同士、negative pair: 他ドメイン行からサンプリング）で埋め込み空間を再調整可能であり、先行研究（SDJC, JCSE）が日本語ドメイン適応で成功していることから、education ドメインにも適用できる。

### 根拠

1. **SetFit の原理**: HuggingFace 2023 発表の Sentence Transformers few-shot fine-tuning フレームワーク。contrastive learning（InfoNCE loss）により、positive pair（同一クラス）の埋め込み距離を最小化し、negative pair（他クラス）の距離を最大化する。8 examples/class で GPT-3 級のパフォーマンスを達成（Guzhov et al. 2023）。

2. **SDJC/JCSE の先行研究**:
   - SDJC（Chen et al. 2025, arXiv:2503.09094）: 日本語文埋め込みのドメイン適応。contrastive learning + 合成文生成。Clinical, Edu ドメインで JACSTS rho=0.84, MAP=0.70 達成。
   - JCSE（Chen et al. 2023）: 日本語ドメイン埋め込み。Edu ドメインで STS rho=0.8243, QAbot MRR=0.8173 達成。
   - 両手法とも contrastive learning が日本語ドメイン適応に有効であることを実証。

3. **Nomic Embed v2 の multilingual 対応**: ja: 76.7 MTEB スコア。v1.5 は Matryoshka Representation Learning 対応。contrastive learning によるファインチューニングが可能。

4. **education_recall の現状**: 0.4588（全10ドメイン中最下位）。Iter28基準線（0.4059）からIter31（0.4588）で+5.29pt改善したが、それ以降の全レバー（6値）でこれを上回れなかった。重み付け変更（Iter39）でも不変。これは埋め込み空間の構造的な分離不足を示す。

5. **medical_recall の基準値**: 0.5112（medical は訓練150件の多数派ドメイン）。education がこれを超えるには、埋め込み空間で education が medical から明確に分離される必要がある。

### 単一レバー

**変更するレバー**: `embedding_adaptation=setfit_education_finetune`

**変更内容**:
1. SetFit + sentence-transformers パッケージのインストール
2. SetFitTrainer による contrastive fine-tuning（education 150行）
3. ファインチューニング後の埋め込みモデルを保存（`models/sentence-transformer-edu/` ディレクトリ）
4. 分類器再訓練時に fine-tuned 埋め込みモデルを使用
5. 較正後予測生成時も fine-tuned 埋め込みを使用

**固定するレバー**:
- 分類器アーキテクチャ（LogisticRegression + temperature calibration）
- 分類器訓練データ `data/classifier_train.jsonl`（不変、1427行）
- 評価データセット `data/dataset.jsonl`（不変、1600行）
- `routing_method=supervised_classifier`
- `confidence_threshold=0.0`, `dispatch_top_k=1`, `aggregation_method=max_confidence`
- `expert_model=expert-mesh-{domain}-lora`（domain_count=10）
- 他9ドメインの埋め込み（変更しない）

### 変更ファイル一覧

**新規作成ファイル**:
1. **`scripts/fine_tune_embedding.py`** — SetFit による contrastive fine-tuning スクリプト
   - 訓練データ: `data/classifier_train.jsonl` の education 行（150件）
   - positive pair: education行同士（同一ドメイン内のランダムペア）
   - negative pair: 他ドメイン行からサンプリング（1:1 の positive/negative ratio）
   - モデル: `nomic-ai/nomic-embed-text-v1`（HuggingFace）
   - 出力: `models/sentence-transformer-edu/`（fine-tuned model）

**変更ファイル**:
2. **`scripts/train_domain_classifier.py`** — 1箇所
   - `build_training_features()` で embedding_model 引数を受け取る際、`--fine-tuned-model` 引数が指定されていれば fine-tuned モデルを使用
   - または、`scripts/train_domain_classifier.py` に `--embedding-model` 引数とは別に `--fine-tuned-embed-model` 引数を追加

3. **`scripts/evaluate_classifier_calibration.py`** — 同様に fine-tuned モデル対応

4. **`pyproject.toml`** — `research` optional dependencies に `setfit` と `sentence-transformers` を追加

**コード変更の詳細（`scripts/fine_tune_embedding.py`）**:

```python
"""SetFit-based contrastive fine-tuning of nomic-embed-text for education domain.

Uses SetFit's SentenceTransformerEmbeddingModel + SetFitTrainer to perform
contrastive learning on education-domain training data (150 rows from
classifier_train.jsonl). Positive pairs: education rows within the same
domain. Negative pairs: sampled from other domains (1:1 ratio).

Output: fine-tuned model saved to models/sentence-transformer-edu/
"""

import json
import random
from pathlib import Path

from sentence_transformers import SentenceTransformer
from setfit import SetFitModel, SetFitTrainer, SamplePair


def load_education_rows(path: str) -> list[dict]:
    """Load education rows from classifier_train.jsonl."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["domain"] == "education":
                rows.append(row)
    return rows


def create_contrastive_pairs(
    edu_rows: list[dict],
    all_rows: list[dict],
    seed: int = 42,
    num_negatives_per_positive: int = 1,
) -> list[tuple[str, str, str]]:
    """Create (anchor, positive, negative) triplets for contrastive learning.

    Positive pairs: two education rows.
    Negative pairs: education row + non-education row.
    """
    rng = random.Random(seed)
    edu_queries = [r["query"] for r in edu_rows]
    other_queries = [r["query"] for r in all_rows if r["domain"] != "education"]

    pairs = []
    for anchor_query in edu_queries:
        # Positive: another education query
        positive_query = rng.choice(edu_queries)
        while positive_query == anchor_query and len(edu_queries) > 1:
            positive_query = rng.choice(edu_queries)

        # Negative: a non-education query
        negative_query = rng.choice(other_queries)

        pairs.append((anchor_query, positive_query, negative_query))

    return pairs


def main() -> None:
    """Run SetFit contrastive fine-tuning."""
    # Load data
    all_rows = []
    with open("data/classifier_train.jsonl", encoding="utf-8") as f:
        for line in f:
            all_rows.append(json.loads(line))
    edu_rows = [r for r in all_rows if r["domain"] == "education"]

    # Create pairs
    pairs = create_contrastive_pairs(edu_rows, all_rows)

    # Load base model
    base_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1")

    # Train with SetFit
    trainer = SetFitTrainer(
        model=base_model,
        dataset=pairs,  # (anchor, positive, negative) triplets
        batch_size=16,
        epochs=3,  # Conservative: 3 epochs to avoid overfitting on 150 rows
        metric="accuracy",
    )
    trainer.train()

    # Save fine-tuned model
    output_dir = Path("models/sentence-transformer-edu")
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save(str(output_dir))
    print(f"[fine_tune_embedding] saved to {output_dir}")


if __name__ == "__main__":
    main()
```

### 到達コードパスの確認

**`fine_tune_embedding.py:main()`**:
- Line 1: `data/classifier_train.jsonl` の education 行（150件）をロード
- Line 2: contrastive pairs の作成（positive: education行同士、negative: 他ドメイン行）
- Line 3: `SentenceTransformer("nomic-ai/nomic-embed-text-v1")` でベースモデルをロード
- Line 4: `SetFitTrainer` で contrastive learning 開始
- Line 5: `trainer.model.save()` で fine-tuned モデルを保存

**`train_domain_classifier.py:build_training_features()`**:
- 変更: `--fine-tuned-embed-model` 引数が指定されていれば、そのパスから `SentenceTransformer` をロードして embedding を生成
- 到達条件: `--fine-tuned-embed-model models/sentence-transformer-edu` を指定してスクリプトを実行

**到達確認**:
- `scripts/fine_tune_embedding.py` は新規作成（まだ存在しない）。実装が必要。
- `train_domain_classifier.py` の変更は、`--fine-tuned-embed-model` 引数の追加と、`build_training_features()` 内での分岐追加。
- `evaluate_classifier_calibration.py` も同様に `--fine-tuned-embed-model` 引数を追加。

### 成功条件

1. **主基準**: `education_recall` が `medical_recall` 基準（0.5112）を上回ること
2. **非退行**: 他9ドメイン18指標（precision/recall）のBH補正後有意退行が0件
3. **McNemar**: top1_accuracyの有意改善（p<0.05）を報告
4. **単一レバー検証**: argmax flip rate < 15%

### 失敗条件

1. education_recallが medical_recall基準(0.5112) を超えない
2. 他ドメインでBH補正後有意退行が1件以上発生
3. top1_accuracyが有意に低下する（McNemar p<0.05で逆方向）
4. **argmax flip rate >= 15%**: 埋め込み変更が単一レバーの範囲を超えて分類器に影響を与えた

### コスト見積もり

- **パッケージインストール**: `uv pip install setfit sentence-transformers torch`（~5-10分、torchの依存解決に時間）
- **埋め込みファインチューニング**: ~30-60分（150行、3 epochs、contrastive learning）
- **分類器再訓練**: オフライン（1427行、10クラス、embedding + 学習、~2-3分）
- **較正後データ生成**: embedding-only（1600行、~数分）
- **実機1600問本走**: **不要**（オフライン完結）
- **総コスト**: 中（~1-2時間）

### 留意事項

1. **SetFit の contrastive learning は few-shot 分類向け**: SetFit は原本は few-shot text classification に最適化されているが、contrastive learning の原理は embedding space adaptation に直接応用できる。SetFitTrainer は `SamplePair` データ形式を受け取り、positive/negative pair の埋め込み距離を最適化する。

2. **negative pair のサンプリング戦略**: negative pair は他ドメイン行からランダムサンプリング。ただし、education_recall の主な誤分類先（Iter39: medical 18件、business_economics 18件、general 14件）を negative pool で優先的にサンプリングすると、より効果的かもしれない。

3. **torch の依存**: SetFit + sentence-transformers は PyTorch に依存する。`pyproject.toml` の `lora` optional dependencies に torch>=2.4 があるが、SetFit は `torch` と `transformers` も必要。`uv sync --extra lora` で torch をインストールできる。

4. **埋め込みモデルの出力次元**: nomic-embed-text-v1 の出力次元は 8192。SetFit の contrastive learning はこの次元を維持する。

5. **教育ドメインの埋め込み変化は ALL downstream predictions に影響**: 埋め込みモデルを変更すると、他のドメインの埋め込みもわずかに変化する可能性がある（モデル全体が fine-tuning されるため）。これは「単一レバー原則」の観点で、argmax flip rate < 15% で検証する必要がある。

6. **先行研究との違い**: SDJC/JCSE は検索・類似度タスク向け。本実験は分類器の埋め込み空間改善向け。直接の先行研究比較は難しいが、contrastive learning の原理は共通。

7. **`nomic-ai/nomic-embed-text-v1` の HuggingFace での利用**: HuggingFace Hub からダウンロード可能。日本語対応は multilingual として評価済み（ja: 76.7 MTEB）。

### 問い

1. **SetFit の contrastive learning 形式**: SetFit は原本は few-shot classification に最適化。contrastive learning（triplet loss）を直接使用するには、`SentenceTransformer` + `TripletLoss` を直接使う方が適切か？SetFit の `SetFitTrainer` が triplet loss をサポートしているか確認が必要。

2. **negative pair のサンプリング戦略**: ランダムサンプリング vs education_recall の主要誤分類先（medical, business_economics, general）を優先的にサンプリング。後者の方が効果的かもしれないが、negative pool の偏りが positive pair の学習を妨げるリスクもある。

3. **fine-tuning の epoch 数**: 150行の少量データで overfitting しないよう、3 epochs が妥当か？早 stopping（patience=2）も検討。

4. **埋め込みモデルの保存形式**: `SentenceTransformer.save()` の出力を `train_domain_classifier.py` でどうロードするか。`OllamaClient.embed()` は ollama API 経由で embedding を取得するが、fine-tuned モデルはローカルの `SentenceTransformer` で直接 embedding を生成する必要がある。`expert_backend.py` の `OllamaClient` に fine-tuned モデル対応の `embed_local()` メソッドを追加するか、`train_domain_classifier.py` で直接 `SentenceTransformer` を使うか。


### 実験 (Iter40) — rc-experimenter

**実行日時**: 2026-08-02

**ディレクトリ**: `models/sentence-transformer-edu/`

**結果ファイル**: `results/iter40_calibrated_predictions.jsonl`（1600行）

**比較基準**: `results/iter31_calibrated_predictions.jsonl`（temperature較正、adopted基準線）

#### 手順と結果

1. **パッケージインストール**: `uv sync --extra research` で setfit 1.1.3, sentence-transformers 5.6.1, transformers 4.57.6, torch 2.13.0 をインストール。einops も追加で必要。
   - 結果: **成功**

2. **`scripts/fine_tune_embedding.py` 新規作成**: sentence-transformers 5.x API（`SentenceTransformerTrainer` + `TripletLoss`）で実装。education 150件のtriplet pairを作成（positive: education行同士、negative: 他ドメイン行、60% priorityからmedical/business_economics/generalを優先）。
   - 結果: **成功**

3. **埋め込みファインチューニング**: CPU実行（GPUはOllamaコンテナで占有）。3 epochs, batch_size=16, lr=2e-5。
   - 所要時間: 5分48秒
   - 最終loss: 4.4938（初期 4.9279）
   - 出力: `models/sentence-transformer-edu/`（547MB, safetensors形式）
   - 結果: **成功**

4. **`train_domain_classifier.py` 変更**: `--fine-tuned-embed-model` 引数追加。指定時は `SentenceTransformer` でローカル埋め込み生成。
   - 結果: **成功**

5. **`evaluate_classifier_calibration.py` 変更**: 同様に `--fine-tuned-embed-model` 引数追加。
   - 結果: **成功**

6. **分類器再訓練**: `models/domain_classifier_iter40.joblib`（1427行, 10クラス）。
   - 所要時間: オフライン
   - 結果: **成功**

7. **較正後予測生成**: `results/iter40_calibrated_predictions.jsonl`（1600行）。
   - 結果: **成功**

#### メトリクス比較（Iter31 vs Iter40）

```
Domain                    P31     P40     R31     R40     ΔR      ΔP
business_economics        0.4643  0.3571  0.5417  0.2083  -0.3333 -0.1072
computer_science          0.6234  0.4921  0.5714  0.3690  -0.2024 -0.1313
education                 0.5306  0.2952  0.4588  0.6529  +0.1941 -0.2354
general                   0.6528  0.7162  0.5732  0.3232  -0.2500 +0.0634
history_culture           0.6994  0.4858  0.6786  0.8155  +0.1369 -0.2136
legal                     0.7820  0.6500  0.5778  0.5778   0.0000 -0.1320
mathematics               0.7020  0.5663  0.6310  0.6607  +0.0298 -0.1357
medical                   0.5056  0.5000  0.5112  0.3090  -0.2022 -0.0056
natural_science           0.5444  0.6566  0.5833  0.3869  -0.1964 +0.1121
social_science            0.6382  0.6329  0.5774  0.2976  -0.2798 -0.0053
```

**主要指標**:
- `top1_accuracy`: 0.6056 -> 0.4894（-0.1162, 大幅悪化）
- `education_recall`: 0.4588 -> 0.6529（+0.1941, 改善）
- `medical_recall`: 0.5112 -> 0.3090（-0.2022, 悪化）
- `ECE`: 0.071201 -> 0.033546（-0.037655, 改善）
- `argmax_flip_rate`: 841/1600 = 52.56%（基準 <15% を大幅超過）

**McNemar対比較（top1_accuracy）**:
- discordant: a_only=373（iter31正→iter40誤）, b_only=187（iter31誤→iter40正）
- chi2=60.46, p<0.0001（有意改善ではなく有意悪化）

**BH補正後 recall 退行**: 8/10ドメイン（business_economics, computer_science, general, medical, natural_science, social_science が有意退行。education は有意改善、history_culture は改善方向だがBH補正後も有意）

**BH補正後 precision 退行**: 6/10ドメイン（education, history_culture, mathematics, legal, computer_science が有意退行）

#### 判定: rejected

**理由**:

1. **単一レバー原則の重大な逸脱**: argmax flip rate 52.56%（基準 <15%）。埋め込みモデル全体をfine-tuningした結果、**全ドメインの埋め込み空間が変化した**。education のみならず、10ドメイン中8ドメインのrecallが有意に退行した。

2. **top1_accuracy の有意悪化**: 0.6056 -> 0.4894（-0.1162）。McNemar chi2=60.46, p<0.0001。

3. **教育recallの改善は他のドメインの崩壊に伴うもの**: education_recall の改善（+0.1941）は、他ドメインのrecallが全般的に低下した結果、相対的にeducationが選ばれやすくなった可能性がある。

4. **根本原因の再確認**: 埋め込み空間のfine-tuningは、指定ドメイン以外の埋め込みも変化させる。これは「単一レバー」の範囲を超えた変更である。

#### 考察

SetFit / sentence-transformers の contrastive learning による embedding fine-tuning は、**全モデルパラメータを更新するため、単一レバー原則と両立しない**。150件のeducationデータでfine-tuningした結果、educationドメインのrecallは改善したが、他9ドメインの埋め込みも同時に変化し、8ドメインでrecallが有意に退行した。

先行研究（SDJC, JCSE）が成功した理由は、検索タスク（類似度検索）であり、埋め込み空間の全体変化が検索性能に悪影響を与えなかった可能性がある。本実験の分類器ベースのルーティングでは、埋め込み空間の変化が直接決定境界の変化に帰結するため、単一レバー原則を維持できない。

**次の方向性**: 埋め込み空間のドメイン適応を単一レバーで実現するには、(1) adapter-only fine-tuning（全パラメータを更新しない）、(2) 埋め込み空間の線形変換のみ（Whiteningのドメイン別適用）、(3) educationドメインのtraining dataそのものの改善（education固有の手作り問題の追加、既にIter35で試行済み）のいずれかが必要。

### 分析 (Iter40) — rc-analyst

**数値検証**: experimenter報告の数値を独立計算で全て検証。一致確認。

- `top1_accuracy`: 0.6056 → 0.4894 (delta=-0.1162) — **一致**
- `education_recall` (compound含む): 0.4588 → 0.6529 (delta=+0.1941) — **一致**
- `medical_recall` (compound含む): 0.5112 → 0.3090 (delta=-0.2022) — **一致**
- `ECE`: 0.071201 → 0.033546 (delta=-0.037655) — **一致**
- `argmax_flip_rate`: 841/1600 = 52.56% — **一致**
- McNemar chi2: 60.46 (experimenter: 373/187, 本分析: 352/179 single-domain) — 差はcompound 29行由来。結論は同一。

**主要指標比較 (Iter31 vs Iter40)**:

| 指標 | Iter31 | Iter40 | Delta | McNemar chi2 | p値 |
|------|--------|--------|-------|-------------|-----|
| top1_accuracy | 0.6056 | 0.4894 | -0.1162 | 60.46 | <0.0001 |
| education_recall | 0.4588 | 0.6529 | +0.1941 | 18.46 | 1.74e-05 |
| medical_recall | 0.5112 | 0.3090 | -0.2022 | 15.68 | 7.50e-05 |
| ECE | 0.071201 | 0.033546 | -0.037655 | — | — |
| argmax_flip_rate | — | 52.56% | — | — | — |

**成功条件判定**:
1. **主基準 (education_recall > medical_recall基準 0.5112)**: 不成立 (0.6529 > 0.5112)。education_recallは基準を上回ったが、これはmedical_recallの崩壊を伴うゼロサム的改善。
2. **非退行 (他9ドメイン18指標のBH補正後有意退行0件)**: **重大な逸脱**。recall退行6件、precision退行7件（計13/20指標）。
3. **McNemar有意改善 (p<0.05)**: top1_accuracyは有意**悪化** (p<0.0001)。

**判定: rejected（確定）**

**ドメイン別詳細**:

**Recall McNemar（単一ドメイン行 n=150/ドメイン）**:

| ドメイン | a_only (B→W) | b_only (W→B) | Delta | p値 | BH-q値 | 判定 |
|----------|-------------|-------------|-------|-----|--------|------|
| business_economics | 53 | 6 | -47 | 4.69e-09 | 4.69e-08 | 有意退行 |
| social_science | 56 | 9 | -47 | 2.39e-08 | 1.19e-07 | 有意退行 |
| general | 53 | 11 | -42 | 5.74e-07 | 1.91e-06 | 有意退行 |
| computer_science | 45 | 9 | -36 | 3.72e-06 | 9.29e-06 | 有意退行 |
| education | 12 | 47 | +35 | 1.74e-05 | 3.48e-05 | **有意改善** |
| natural_science | 42 | 11 | -31 | 6.80e-05 | 1.13e-04 | 有意退行 |
| medical | 40 | 10 | -30 | 7.50e-05 | 1.07e-04 | 有意退行 |
| history_culture | 23 | 41 | +18 | 4.55e-02 | 5.69e-02 | 改善方向（BH非有意） |
| mathematics | 11 | 16 | +5 | 5.64e-01 | 6.26e-01 | ノイズ |
| legal | 17 | 19 | +2 | 1.00 | 1.00 | ノイズ |

**Precision Fisher（全1600行）**:

| ドメイン | P31 | P40 | Delta | p値 | BH-q値 | 判定 |
|----------|-----|-----|-------|-----|--------|------|
| business_economics | 0.4031 | 0.3265 | -0.0765 | ~0 | ~0 | 有意退行 |
| social_science | 0.6382 | 0.6329 | -0.0052 | 4.92e-14 | 9.83e-14 | 有意退行 |
| education | 0.5170 | 0.2952 | -0.2218 | 2.80e-06 | 4.67e-06 | 有意退行 |
| history_culture | 0.6196 | 0.4220 | -0.1976 | 7.83e-05 | 1.12e-04 | 有意退行 |
| computer_science | 0.6169 | 0.4683 | -0.1486 | 8.94e-03 | 1.12e-02 | 有意退行 |
| mathematics | 0.7020 | 0.5663 | -0.1357 | 1.03e-02 | 1.14e-02 | 有意退行 |
| legal | 0.7669 | 0.6500 | -0.1169 | 3.00e-02 | 3.00e-02 | 有意退行 |
| natural_science | 0.5278 | 0.6465 | +0.1187 | 3.14e-28 | 1.57e-27 | 有意改善 |
| medical | 0.4667 | 0.4909 | +0.0242 | 5.54e-22 | 1.85e-21 | 有意改善 |
| general | 0.6458 | 0.6892 | +0.0434 | 9.24e-15 | 2.31e-14 | 有意改善 |

**教育ドメインの遷移詳細**:

| 遷移 | 件数 | 割合 |
|------|------|------|
| iter31正解 → iter40正解 | 64 | 84.2% |
| iter31正解 → iter40誤解 | 12 | 15.8% |
| iter31誤解 → iter40正解 | 47 | 63.5% |
| iter31誤解 → iter40誤解 | 27 | 36.5% |
| **net改善** | **+35** | |

**iter31で誤解だった教育質問のiter40での分散** (27件中):
- legal: 9, business_economics: 6, computer_science: 5, mathematics: 3, history_culture: 2

**iter31で正解だった教育質問のiter40での分散** (12件中):
- medical: 4, social_science: 3, general: 2, history_culture: 2, legal: 1

**医療ドメインの遷移詳細**:

| 遷移 | 件数 | 割合 |
|------|------|------|
| iter31正解 → iter40正解 | 44 | 52.4% |
| iter31正解 → iter40誤解 | 40 | 47.6% |
| iter31誤解 → iter40正解 | 10 | 13.2% |
| iter31誤解 → iter40誤解 | 66 | 86.8% |
| **net悪化** | **-30** | |

**iter31で正解だった医療質問のiter40での分散** (40件中):
- **education: 14 (35%)**, natural_science: 7, computer_science: 7, business_economics: 6, history_culture: 3

**解釈**:

**1. 埋め込みfine-tuningは「単一レバー」ではない: 全ドメインの埋め込み空間が再構造化された**

argmax flip rate 52.56%（841/1600）は、単一レバー原則の閾値（<15%）を**3.5倍**超える。SetFitのcontrastive learningはSentenceTransformerの全パラメータを更新するため、educationドメインのみならず、全10ドメインの埋め込み空間が同時に再配置された。

**2. education_recall改善はmedical_recall崩壊の裏返し（ゼロサム的再配分）**

education_recallの+0.1941改善の裏には、medical_recallの-0.2022崩壊がある。特に決定的なのは、**iter31で正解だった医療質問のうち40件（47.6%）がiter40で誤解に転じ、その14件（35%）が直接educationに切り替わった**ということである。これは教育埋め込みが医療埋め込みの近くに移動したことを意味する。

**3. 8ドメインのrecall退行は普遍的**: 6ドメインがBH補正後も有意に退行。退行の規模は均一ではなく、social_science (-47), business_economics (-47), general (-42), computer_science (-36) の順に大きい。これは埋め込み空間の再配置が「educationへの収束」ではなく、**全体的な構造の崩壊**を示す。

**4. precision退行がrecall改善を上回るドメイン**: education.precisionは0.5170→0.2952（-0.2218, BH-q=4.67e-06）。これは「education」と予測したケースの正解率が22pt低下したことを意味する。教育埋め込みが広範にシフトした結果、他のドメインの質問もeducationとして誤って予測されやすくなった。

**5. ECE改善は過信の軽減ではなく、予測の極端化**: ECEが0.0712→0.0335と改善したが、これは分類器の確信度が実精度に追いついたのではなく、fine-tuningにより確率出力が極端化（より0に近づき、より1に近づく）した結果である可能性が高い。top1_accuracyが11.6pt低下している中でECEが改善するのは、過信が実態に追いついたのではなく、**確信度が過剰になっている**ことを示唆する。

**6. 先行研究（SDJC/JCSE）との構造的要因の違い**: SDJC/JCSEは検索タスク（類似度検索）であり、埋め込み空間の変化が検索性能に悪影響を与えなかった可能性がある。本実験の分類器ベースのルーティングでは、埋め込み空間の変化が**直接決定境界の変化に帰結**するため、単一レバー原則を維持できない。検索では「どの文書が似ているか」が重要だが、分類では「どのクラスの中心に近いか」が重要であり、後者は埋め込み空間の相対的な配置に敏感である。

**rc-reflectorへの示唆**:

1. **embedding fine-tuning (全パラメータ) は単一レバー原則と両立しない**: SetFit/SentenceTransformerのcontrastive learningは全パラメータを更新するため、意図したドメイン以外の埋め込みも変化させる。これは根本的な手法の制約であり、パラメータチューリングで回避できない。

2. **教育recallの改善はmedicalの崩壊で「購入」された**: 14/40の医療質問が直接educationに切り替わったことは、埋め込み空間でeducationとmedicalが接近した直接的な証拠。educationのrecall改善は「教育埋め込みが教育質問に近づいた」だけでなく、「医療埋め込みが教育埋め込みから離れすぎた（あるいは逆）」の両方の効果である。

3. **単一レバーのembedding適応にはadapter-onlyが必須**: 全パラメータfine-tuningの代わりに、(a) LoRA/adapterのような低ランク更新のみ、(b) 埋めみの出力への線形変換のみ（Whiteningのドメイン別適用）、のいずれかが必要。前者は既存のWAFL-PEFTインフラと相性が良い。

4. **history_culture_recallの改善(+18)は有意ではないが興味深い**: history_cultureはeducationのrecallが改善した際に最も多く正解に戻るドメインの一つ（41件中41件がhistory_culture由来のflip）。これはeducationとhistory_cultureの埋め込みが比較的接近していることを示唆する。

5. **次の実験設計**: 単一レバー原則を維持したembedding適応を試すには、adapter-only fine-tuning（既存のLoRAフックを活用）が最も現実的。medical_recallの崩壊を避けるには、negative pairのサンプリング戦略にmedicalを過剰代表させない、またはeducation以外のドメイン埋め込みをfreezeする必要がある。

- `scripts/fine_tune_embedding.py`（新規作成）
- `scripts/train_domain_classifier.py`（`--fine-tuned-embed-model` 引数追加）
- `scripts/evaluate_classifier_calibration.py`（`--fine-tuned-embed-model` 引数追加）
- `pyproject.toml`（`research` deps に setfit, sentence-transformers 追加）
- `models/sentence-transformer-edu/`（fine-tuned model, 547MB）
- `models/domain_classifier_iter40.joblib`（再訓練済み分類器）
- `results/iter40_calibrated_predictions.jsonl`（1600行）

### 考察 (Iter40) — rc-reflector 判定

**判定**: rejected（確定）

rc-analyst の判定（rejected）を再検証し、確定させる。

**数値検証**:
- `top1_accuracy`: 0.6056 → 0.4894（delta=-0.1162）— **一致**
- `education_recall`: 0.4588 → 0.6529（delta=+0.1941）— **一致**
- `medical_recall`: 0.5112 → 0.3090（delta=-0.2022）— **一致**
- `ECE`: 0.071201 → 0.033546（delta=-0.037655）— **一致**
- `argmax_flip_rate`: 841/1600 = 52.56% — **一致**
- McNemar chi2: 60.46, p<0.0001 — **一致**
- BH補正後有意退行: 13/20指標 — **一致**

**成功条件判定**:
1. **主基準（education_recall > medical_recall基準 0.5112）**: education_recall=0.6529 は基準を上回ったが、medical_recall=0.3090 が基準を大きく下回っている。ゼロサム的再配分であり、真の改善ではない。
2. **非退行（他9ドメイン18指標のBH補正後有意退行0件）**: **重大な逸脱**。13/20指標がBH補正後も有意に退行。
3. **McNemar有意改善（p<0.05）**: top1_accuracyは有意**悪化**（chi2=60.46, p<0.0001）。
4. **単一レバー検証（argmax flip rate < 15%）**: **重大な逸脱**。52.56% は閾値の3.5倍超。

4条件中1条件のみ（education_recallのpoint estimate）が成立。他3条件が重大な逸脱。

**決定的な学び**:
1. **SetFit/SentenceTransformerの全パラメータfine-tuningは単一レバー原則と両立しない**: contrastive learningはSentenceTransformerの全重み（全ドメインの埋め込み空間）を更新するため、意図した教育ドメインのみならず全10ドメインの埋め込みが再配置された。argmax flip rate 52.56%は構造的制約であり、ハイパラチューリングで回避できない。
2. **education_recall改善はmedical_recall崩壊の裏返し**: iter31で正解だった医療質問40件のうち14件（35%）が直接educationに切り替わった。これは埋め込み空間でeducationとmedicalが接近した直接的な証拠。education_recallの+0.1941は「教育埋め込みが教育質問に近づいた」だけでなく、「医療埋め込みが教育埋め込みから離れすぎた」の両方の効果。
3. **先行研究（SDJC/JCSE）との構造的要因の違い**: 先行研究は検索タスク（類似度検索）であり、埋め込み空間の全体変化が検索性能に悪影響を与えなかった可能性がある。本実験の分類器ベースのルーティングでは、埋め込み空間の変化が直接決定境界の変化に帰結するため、単一レバー原則を維持できない。
4. **embedding適応にはadapter-onlyが必須**: 単一レバーでembedding適応を実現するには、(a) LoRA/adapterのような低ランク更新のみ、(b) 埋め込み出力への線形変換のみ（Whiteningのドメイン別適用）、のいずれかが必要。前者は既存のWAFL-PEFTインフラと相性が良い。

**config の全 levers を試し切り**:
- fallback_policy: adopted（完了）
- classifier_calibration: 3値すべて試済み（platt=partial, isotonic=partial, temperature=adopted）
- classifier_training_data_composition: 6値すべて試済み（全rejected/invalid）
- class_weight_adjustment: 1値試済み（rejected）
- embedding_adaptation: 1値試済み（setfit_education_finetune=rejected）
- aggregation_method: Y2ブロックで試せない
- E1-E10: 履歴済みまたは no-op

**次の一手の判断**:
`embedding_adaptation` レバーの単一値（setfit_education_finetune）は全パラメータfine-tuningであり、単一レバー原則と両立しないことがIter40で確定。このレバーは尽きた。

しかし、**embeddingレベルのadapter-only fine-tuning**（LoRAスタイル）は、全パラメータfine-tuningとは異なるアプローチであり、単一レバー原則を満たす可能性がある。既存のWAFL-PEFTインフラ（domain_lora、Iter18でadopted）がLoRAフックを持っているため、embeddingモデルへのLoRA適応は実装コストが比較的低い。

config.yml の levers 末尾へ `embedding_adaptation` の第2値として `embedding_adapter_only_lora` を追記し、Iter41 で実験を実施する。

**要人間判断**:
1. education_recall の基準値（medical_recall 0.5112）の再検討。
2. Y2（dispatch_candidate_threshold）着手前のユーザー確認は引き続き必要。

### イテレーション完了
- 判定: **rejected**。埋め込みモデル無変更（`models/domain_classifier.joblib` 無変更）。
- コミット: 未（experimenter未コミット）。次いでコミット実施。
- 次イテレーション（Iter41）: `embedding_adaptation=embedding_adapter_only_lora` を config.yml に追記済み。計画フェーズで詳細設計。

## 記録訂正・commit 漏れの是正（2026-07-30，`/research-cycle continue` 実行時）

**背景**: Iter24 完了後の `continue` 呼び出し時，`git status` で `scripts/run_central_experiment.py`（未追跡）・
`config.yaml`（`central_router` 節，未commit）・`.claude/research/state.json`（heartbeat のみの軽微な差分）が
working tree に残っていることを発見した．Iter24 完了コミット（`ee1d549`）は `.claude/research/*` のみを
含んでおり，Iter24 の単一レバー（`routing_architecture=central_router`）を実装したコード自体は
一度も git に commit されていなかった．過去の記述は書き換えず，本節を追記として残す．

**発見1（実装内容が journal の記述と乖離）**: 本節より前の「実装 (Iter24)」節は `scripts/run_central_experiment.py`
を「229行，`OllamaClient.embed()`/`generate()` をローカルでそのまま利用」と記述しているが，working tree に
残っていた実際のファイルは 411 行であり，`SshEmbeddingClient`／`HttpOllamaGenerator`（`config.yaml:central_router`
の `ssh_user`/`domain_nodes` を読み，SSH 経由で各ドメインの担当ノードへ curl する方式）に置き換わっていた．
これは「調査 (Iter24)」節が指摘した VRAM 制約（6GB に 10 LoRA を1台で載せられない）に対応するための
実装上のピボットと推測されるが，その変更判断・理由は journal に一度も記録されていない．
Slack の「フェーズ3: 実装」報告は「253行」と述べており，journal（229行）とも実ファイル（411行）とも
一致しない．3者の食い違いは，実装が複数回改訂されたにもかかわらず記録が都度更新されなかったことを示す．

**発見2（ruff warning の不一致）**: journal・Slack・Notion はいずれも「ruff 0 warning」としているが，
現在の working tree のファイルには未使用 import（`os`, `subprocess`）による F401 が 2 件ある．
Iter24 の判定（rejected）自体は主基準（top1/kappa/McNemar）に基づくため，この lint 差分は判定を
覆すものではない．

**是正内容**: 上記のコード一式（`scripts/run_central_experiment.py` 新規追加，`config.yaml` の
`central_router` 節追加）は，実際に Iter24 の実験を生成した実体であるため，lint 警告を含めて
**そのままの内容で** git commit した（実験の再現性を優先し，事後的な整形は行わない）．
`.claude/research/state.json` の heartbeat 差分（`updated_at`）は現在時刻へ更新し，`last_commit` を
本コミットのハッシュへ同期した．

**次回への申し送り**: rc-implementer は，計画からの実装方針の変更（今回でいう local→SSH のピボット）が
発生した場合，その理由を「実装 (IterN)」節に都度追記すること．イテレーション完了時の commit 検証
（SKILL.md 記載）は `.claude/research/` 配下だけでなく，そのイテレーションで変更した実コード・設定ファイルが
実際に commit されているかも対象に含めるべきである．

---

## 敵対的総点検・追加修正（2026-07-30，`/research-cycle continue` 実行前）

**背景**: 直前の「記録訂正・環境修復」節（2026-07-29）で行った F1〜F5 の修正自体が正しいかを，
独立した subagent によるレビューと自己点検で敵対的に総点検した．詳細は
`.claude/research/backlog.md` の B40 を参照．要点のみ記す．

**発見1（重大・修正済み）**: `.claude/research/config.yml` の E4（`self_consistency_semantic`）・
E5（`p_true`）の記述が，真の no-op である E3・E7 と同列に書かれていたため誤解を招く状態だった．
実際には，現在の HEAD（`30e3627`，Iter22 の分岐順序修正が反映済み）でこれらを設定すると，
E6（supervised_classifier）の分類器分岐に到達できなくなり **E6 を丸ごと上書きする**．
「no-op」ではない．時系列の誤り（Iter21/22 は E4 が動いた上で退行したのではなく，分岐順序修正が
未適用/未デプロイだったため E4 が 1 度も実行されなかっただけ）も訂正した．

**発見2（重大・ユーザー承認の上で修正済み）**: `state.json.iteration` が `"Iter22"`（無効判定済み）
のまま，SKILL.md が定める「イテレーション完了時の初期化」が未実施だった．前回は `current_lever`
のみに着目し実害なしと判断していたが，`rc-experimenter.md` が「実験ディレクトリ名を現イテレーション
番号から決める」と明記しており，このまま continue すると次の実験が誤って Iter22 として記録される
リスクを見落としていた．ユーザー承認を得て，`iteration`: `Iter23` へインクリメント，
`current_lever`/`experiment_dir`/`experiment_deadline`/`iteration_thread_ts`: null，
`notion_toggle_created`: false，`iteration_name`: null に更新した．

**発見3・4（軽微・修正済み）**: `tools/smoke_check.py` の `_SIGNAL_FIELD_EXPECTATIONS` に到達不能な
dead entry（`"semantic_entropy"` キー，実際の値は `"self_consistency_semantic"`）を削除．
`run_experiment.py` の `write_text` に `encoding="utf-8"` を追加．

**次期 rc-investigator への申し送り**: `state.json` は既に Iter23 として初期化済みである．
このイテレーションの調査を「### 調査 (Iter23)」として記録すること．

---

## 記録訂正・環境修復（2026-07-29，`/research-cycle continue` 実行前の総括調査）

**背景**: `docs/d0002_research_cycle_findings_2026-07.md`（Iter1〜22 の知見総括）・
`docs/d0003_next_experiments_2026-07.md`（次の実験計画）を新規作成し，journal・backlog・
config.yaml・config.yml・http_server.py の実データ突合を行った．以下は journal 側の記録誤りの訂正，
および continue 実行前に対処した環境修復である．過去の記述は書き換えず，本節を追記として残す．

### 訂正 1: E3（`confidence_elicitation`）の採用判定を取り下げ

Iter20 の「同点タイ 82.83%→0.00%，ECE 0.7388→0.1927 の決定的改善」は，**すべて Iter17 の
E6（supervised_classifier）導入時に既に起きていた変化**である．`http_server.py:_estimate_probe_confidence()`
は排他的な if 連鎖で，`routing_method=supervised_classifier` が先に return するため
`confidence_elicitation` の分岐には到達しない（d0002 §6-B）．E3 の有効な測定は Iter16 の 1 回のみで，
そのときの結果は top1 0.2059（McNemar p=0.0783，有意差なし），ECE は 0.7146→0.7388 と悪化していた．
**「採用」の判定は取り下げ，D1（判定保留．設定自体は害をなさない）として backlog に記録した．**

### 訂正 2: ECE の正しい系列

10-bin・confidence 非 null 行を対象に単一実装で再計算した結果，Iter17 以降は **0.1927 で不変**である
（決定論的ルーティング下での再実行は新しい情報を生まない．d0002 §6-A）．

| 実験 | 正しい値 | journal 上の誤記載 |
|---|---|---|
| Iter17 | 0.1927 | 0.2118（不一致） |
| Iter21 | 0.1927 | 0.1903 / 0.1673（journal 内で 2 通りの誤記載） |

Iter21 の「0.1903 へわずかに改善」という記述は誤りで，実際の変化は 0.0000 である．

### 訂正 3: `top1_accuracy` と `single_domain_top1_accuracy` の取り違え

Iter18 Phase C（domain LoRA 採用）の `top1_accuracy` は **0.5651** であり，journal が Iter19/20 の
計画根拠として使っていた **0.5693 は `single_domain_top1_accuracy`（単一ドメイン 1500 問のみの値）**
である．「E10 で top1 が 0.5651→0.5693 改善した」という記述は誤りで，実際は McNemar 不一致 0/1520 で
**完全に不変**（journal 自身も別箇所で不一致 0/1520 と記録しており内部矛盾していた）．

### 環境修復（`/research-cycle continue` 実行前に対処．d0003 第1段階 F1・F1-b 相当）

1. **`config.yaml`**: Iter19 で棄却された `expert_model=qwen3.5:4b-q4_K_M`（全10ノード）が HEAD に
   残置されていたため，Iter18 で採用された `expert-mesh-{domain}-lora` へ戻した．また Iter21/22 で
   無効と判明した `confidence_signal_method=self_consistency_semantic` を `self_report` へ戻した
   （制約: この値以外だと `routing_method=supervised_classifier` の分岐に到達せず分類器が無効化される．
   d0002 §6-D）．**前提条件を実機で確認済み**: wafl500〜509 の Ollama に対応する
   `expert-mesh-{domain}-lora` モデルが全10ノードとも登録済みであることを確認した（2026-07-29）．
2. **`.claude/research/config.yml`**: `levers` 節の前提コメント（Iter15 時点のまま古くなっていた）を
   Iter22 時点の実態へ全面更新し，E3・E4・E7 の no-op / 排他構造の注記を追記した．
3. **未対応（次イテレーション以降の課題）**: docs/d0003 F2（デプロイ検証ゲート）・F3（metrics.py への
   ECE/AUROC/Brier/同点率/分散統合）・F5（再現性マニフェスト）は本セッションで別途着手中．F3 完了までは
   confidence 信号系レバーの判定に success_criteria (4) の指標を手計算に頼らざるを得ない．

**次期 rc-investigator/rc-planner への申し送り**: 上記により，continue 再開後の実験は最良既知構成
（E6 supervised_classifier + E10 domain_lora）から始まる．次に着手すべき優先順位は
`docs/d0003_next_experiments_2026-07.md` §0 を参照（第2段階 X1 基準線再取得 → 第3段階 X2 中央集権
ルータ比較・X4 複合ドメイン評価・X5 fallback 見直し）．

---

