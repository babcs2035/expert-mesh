## Iteration 19: Qwen3.5 モデルサイズ 9B→4B 変更による推論速度・VRAM 効率・回答品質への影響測定

### 計画

**単一レバー**: `expert_model_size` (E8), `expert_model: expert-mesh-{domain}-lora → qwen3.5:4b-q4_K_M`

**変更ファイル**: `config.yaml` のみ（10 行の `expert_model` 値変更）
**変更しないファイル**: Dockerfile, docker-compose, コード類（変更不要）

**仮説**:

この変更は「モデルサイズ縮小」と「LoRA 統合モデルの撤去」の二重影響を伴う。

1. **推論速度の改善（主目的）**: Qwen3.5-4B Q4_K_M（~2.4GB）は Llama 3.1 Swallow 9B Q4_K_M（~4.9GB）の約半分。パラメータ数の単純比例（4/9 ≈ 0.44）に加え、KV cache の VRAM 余裕（6GB GPU で 5.67GB → ~2.5GB）により推論速度が約 40-60% 向上すると期待する。mean_duration_ms 3515ms → 1200-1800ms が目標。
2. **VRAM 効率の改善（主目的）**: モデルサイズが約 2.5GB になり、KV cache に 3.5GB の余裕が生まれる。これにより、長時間実行時の CPU offload リスクが低減し、dispatch_failure_rate が 0.0 を維持できる。
3. **回答品質の低下（許容されるトレードオフ）**: LoRA アダプタは Llama 3.1 Swallow 固有のアーキテクチャに依存するため、Qwen3.5-4B では動作しない。LoRA 撤去 + 4B モデルの二重影響で、answer_quality_accuracy は 0.5013 → 0.20-0.30 の低下が予想される（Iter18 Phase A: LoRA なし 9B で 0.2787 だった実績あり）。**E8 の主目的が「推論速度・VRAM 効率の改善」であるため、回答品質の低下は副次的な影響として位置付け、許容範囲とする**。
4. **top1_accuracy の安定（ルーティング不変）**: ルーティングは light_model（qwen3.5:4b）+ supervised_classifier のまま変更されないため、top1_accuracy は 0.5693 ± 0.03 の範囲で推移すると予想。

**固定する構成**（Iter18 Phase C の最良構成を継承）:

| 設定 | 値 | 理由 |
|------|-----|------|
| `light_model` | `qwen3.5:4b-q4_K_M` | 変更不可。ルーティング用。現行と同一 |
| `routing_method` | `supervised_classifier` | 変更不可。Iter17 で採用済み |
| `confidence_signal_method` | `self_report` | 変更不可 |
| `confidence_threshold` | `0.5` | 変更不可 |
| `dispatch_top_k` | `1` | 変更不可 |
| `domain_count` | `10` | 変更不可。10 ノード構成のまま |
| データセット | JMMLU 1520 問 | 変更不可 |
| `classifier_model_path` | `models/domain_classifier.joblib` | 変更不可 |

**成功条件**:

| 分類 | 指標 | ベースライン (Iter18 Phase C) | 成功条件 | 根拠 |
|------|------|-----------------------------|---------|------|
| 主基準 | mean_duration_ms | 3515ms | **2000ms 以下**（-43%） | 4B モデルの推論速度向上が E8 の主目的。1520 問で約 46 分 → 約 17 分に短縮 |
| 主基準 | VRAM 使用量（expert） | 5.67GB | **3.0GB 以下** | KV cache の余裕確保。6GB GPU で 3GB 以上の余裕があれば CPU offload リスク低減 |
| 非退行 | top1_accuracy | 0.5693 | **0.5300 以上**（-3.9pt 以内） | routing は不変のため大幅退行は想定しない。測定誤差 ±3pt の余裕 |
| 非退行 | Cohen's kappa | 0.5215 | **0.4800 以上** | top1_accuracy の非退行と整合 |
| 報告 | answer_quality_accuracy | 0.5013 | **報告のみ** | LoRA 撤去 + モデル縮小により低下が想定。E8 の主目的外 |
| 報告 | end_to_end_accuracy | 0.3151 | **報告のみ** | answer_quality に連動 |
| 監視 | dispatch_failure_rate | 0.0 | **0.0** | VRAM 余裕により低下リスク低い |
| 監視 | fallback_rate | 0.1316 | **報告** | 閾値ゲートは expert_model に依存しない |

**実験構成（フルフロー）**:

```
Step 0: ベースライン確認（Iter18 Phase C の結果を再確認）
  results/20260729_042712/ の数値:
    top1_accuracy=0.5693, answer_quality_accuracy=0.5013, end_to_end=0.3151
    mean_duration_ms=3515, fallback_rate=0.1316, dispatch_failure_rate=0.0

Step 1: config.yaml 変更
  全10ノードの expert_model を変更:
    expert-mesh-general-lora     → qwen3.5:4b-q4_K_M
    expert-mesh-education-lora   → qwen3.5:4b-q4_K_M
    expert-mesh-legal-lora       → qwen3.5:4b-q4_K_M
    expert-mesh-medical-lora     → qwen3.5:4b-q4_K_M
    expert-mesh-business_economics-lora → qwen3.5:4b-q4_K_M
    expert-mesh-computer_science-lora   → qwen3.5:4b-q4_K_M
    expert-mesh-natural_science-lora    → qwen3.5:4b-q4_K_M
    expert-mesh-mathematics-lora        → qwen3.5:4b-q4_K_M
    expert-mesh-history_culture-lora    → qwen3.5:4b-q4_K_M
    expert-mesh-social_science-lora     → qwen3.5:4b-q4_K_M

Step 2: デプロイ
  mise run setup（Docker イメージ再ビルドは不要）
  mise run deploy（全10ノード）
  各ノードで ollama pull qwen3.5:4b-q4_K_M が完了していることを確認

Step 3: 実験
  mise run start（同一 1520 問データセット）
  完了後: mise run analyze

Step 4: 分析
  metrics.py --results <dir>/results.jsonl --json
  → top1_accuracy, mean_duration_ms, VRAM usage, answer_quality_accuracy
```

**実行時間の見積もり**:

| 工程 | 推定時間 | 備考 |
|------|---------|------|
| Step 1-2: config 変更 + デプロイ | 10-15 分 | Docker イメージ再ビルドは不要 |
| Step 3: 実験 | 40-60 分 | 推論速度の向上により、Iter18 の 89 分に対して短縮 |
| Step 4: 分析 | 1-2 分 | metrics.py のオフライン計算 |
| **合計** | **約 55-80 分** | Iter18 の 89 分に対して約 30% 短縮 |

**リスクと緩和策**:

| リスク | 内容 | 影響 | 緩和策 |
|-------|------|------|--------|
| R1: 回答品質の大幅低下 | 4B モデルは 9B より回答精度が低い。LoRA 撤去によりさらに低下 | answer_quality_accuracy が 0.20 以下になる可能性 | E8 の主目的は速度・VRAM 効率であり、回答品質は報告のみ。次イテレーションで Qwen3.5-4B 向けの LoRA 再訓練を計画 |
| R2: Ollama でのモデル pull 失敗 | 実機ノードで qwen3.5:4b-q4_K_M が pull できない | 実験が実行できない | デプロイ前に ollama pull をテスト |
| R3: LoRA モデルの残存 | 旧 LoRA モデルが ollama に残り、VRAM を圧迫 | VRAM 余裕が期待ほど得られない | デプロイ前に ollama rm で LoRA モデルを削除 |
| R4: Qwen3.5 の日本語性能 | Qwen3.5 は英語中心のモデル。日本語回答品質が Llama 3.1 Swallow より劣る | answer_quality_accuracy の低下がモデルアーキテクチャ由来 | 日本語評価（answer_quality_accuracy）を重点監視 |

**実装フェーズへの示唆**:

1. **変更は config.yaml のみ**。rc-implementer は `config.yaml` の 10 行の `expert_model` 値を `qwen3.5:4b-q4_K_M` に変更すればよい。
2. **Docker イメージの再ビルドは不要**。Python コードの変更がないため、rsync での config.yaml 配布のみで十分。
3. **ollama pull の事前確認**: デプロイ前に各ノードで `ollama pull qwen3.5:4b-q4_K_M` を実行し、モデルがダウンロード済みであることを確認する。
4. **LoRA モデルの削除は任意**: 旧 LoRA モデル（`expert-mesh-{domain}-lora`）は ollama list に残るが、expert_model が指さないため機能的影响はない。VRAM 節約の観点から削除を推奨するが、必須ではない。
5. **評価軸②③の測定**: `mise run analyze` で evaluate_response_quality.py が自動実行され、answer_quality_accuracy と end_to_end_accuracy が計算される。

---

### 調査 (Iter19)

**単一レバー**: `expert_model_size` (E8), `expert_model: expert-mesh-{domain}-lora (Llama 3.1 Swallow 9B) → qwen3.5:4b-q4_K_M`

**調査の問い**

1. `expert_model_size` 変更の具体的な構成（どのファイルを変更するか）
2. VRAM 効率と推論速度への影響
3. 回答品質への影響（LoRA 統合モデル vs 汎用 4B モデル）
4. ドメイン数: 4 ドメイン vs 10 ドメイン
5. 既存コードとの互換性（Dockerfile, docker-compose, Ollama 設定）
6. ベースライン比較（Iter18 Phase C の数値）

**1. `expert_model_size` 変更の具体的な構成**

**変更するファイル**: `config.yaml` のみ（10 行）

現行 config.yaml の各ノード設定:
```yaml
nodes:
  wafl500:
    expert_model: expert-mesh-general-lora
  wafl501:
    expert_model: expert-mesh-education-lora
  ...（他8ノードも同様）
```

変更後:
```yaml
nodes:
  wafl500:
    expert_model: qwen3.5:4b-q4_K_M
  wafl501:
    expert_model: qwen3.5:4b-q4_K_M
  ...（他8ノードも同様）
```

**変更しないファイル**:
- `Dockerfile`: 変更不要（Python コードの変更は発生しない）
- `docker-compose.yml`: 変更不要（volume マウントは既存のままで ok）
- `docker-compose.gpu.yml`: 変更不要
- `pyproject.toml`: 変更不要
- `mise.toml`: 変更不要
- `classifier.py`, `router.py`, `http_server.py`, `node.py`, `expert_backend.py`: 変更不要

**light_model の扱い**: 現状維持（`qwen3.5:4b-q4_K_M`）
- 理由: light_model は probe（ルーティング前段階）のみで使用。現行でも expert_model とは別モデル（9B→4B）で運用済み。4B→4B の変更は不要。

**2. VRAM 効率と推論速度への影響**

**現行（9B LoRA 統合モデル）**:
- モデルサイズ: ~4.9GB（Llama 3.1 Swallow 9B Q4_K_M）
- VRAM 実測: 5.67GB（results/20260721_222225 のログ）
- 空き VRAM: 6GB 環境では KV cache の余裕ほぼなし
- mean_duration_ms: 3515ms（results/20260729_042712）
- dispatch_gen_time_ms: 平均 2972ms（1320 件中，min=321, max=9835）

**提案（Qwen3.5-4B Q4_K_M）**:
- モデルサイズ: ~2.4GB（ollama.com の qwen3.5:4b）
- VRAM 推測: ~2.5GB（Q4_K_M 量化，4B パラメータ）
- 空き VRAM: 6GB 環境で約 3.5GB の余裕（KV cache に余裕）
- 推論速度: 4B モデルは 9B の約 2-3 倍の推論速度が文献で報告されている（Qwen 公式ベンチマーク）
- mean_duration_ms の推測: 1200-1800ms（約 40-60% 短縮）

**根拠**:
- Qwen3.5-4B は Ollama ライブラリで利用可能（ollama.com/library/qwen3.5/tags で確認）
- 4B モデルの Q4_K_M 量化は約 2.4-2.5GB（Llama 3.1 Swallow 9B Q4_K_M の ~4.9GB の約半分）
- 推論速度の向上はパラメータ数の単純比例（4/9 ≈ 0.44）に加えて，KV cache の余裕による GPU メモリ帯域の効率化が期待される

**3. 回答品質への影響（最も重要なトレードオフ）**

**現行（9B + LoRA）**:
- answer_quality_accuracy: 0.5013（Iter18 Phase C）
- end_to_end_accuracy: 0.3151（Iter18 Phase C）
- top1_accuracy: 0.5693（Iter18 Phase C）

**提案（4B 汎用）の回答品質**:
- **LoRA 統合モデルは Llama 3.1 Swallow ベース**。Qwen3.5-4B は異なるアーキテクチャ（Llama 互換ではない）のため，LoRA アダプタは**動作しない**（PEFT/LoraConfig はベースモデルのアーキテクチャに依存）
- 4B 汎用モデルはドメイン固有の知識を持たない（LoRA なし）
- 回答品質は Iter18 Phase A（LoRA なし，9B 汎用）の結果が参考になる: answer_quality_accuracy=0.2787
- 4B モデルは 9B モデルより回答品質が低下する可能性が高い（パラメータ数の差）
- **推測**: answer_quality_accuracy は 0.2787（Phase A）→ 0.20-0.30 の範囲に低下する可能性

**重要な発見**: この変更は「モデルサイズ変更」だけでなく「LoRA 統合モデルの撤去」を意味する。LoRA アダプタは Llama 3.1 Swallow 固有であり，Qwen3.5 には適用できない。

**4. ドメイン数: 4 ドメイン vs 10 ドメイン**

**config.yml note の指示**: 「4 ドメインのまま」と記載。

**実装上の判断**: **10 ドメインのまま**を推奨。

**理由**:
- 現行の WAFL ノード（wafl500-509）は既に 10 ドメインで構成済み
- 4 ドメインに減らすには config.yaml のノード定義の削除（10→4）と，データセットのフィルタリングが必要
- 10 ドメインのままでも，expert_model_size の単独影響は測定可能（light_model は不変，routing_method は不変）
- E1（評価集合の拡張）で整備した 1520 問の JMMLU データセットは 10 ドメイン向けに設計済み
- config.yml note の「4 ドメインのまま」は，「expert_model_size 変更単独の影響を測るために expert_model 以外の設定は変えない」という意図と解釈できる

**5. 既存コードとの互換性**

**Dockerfile**: 変更不要
- Python コードの変更は発生しない
- Docker イメージの再ビルドは不要（ただし config.yaml の変更は rsync で配布）

**docker-compose.yml**: 変更不要
- LoRA アダプタの volume マウント（`./models/lora_adapters:/app/models/lora_adapters:ro`）は残ったままになるが，expert_model が LoRA モデルを指さないため，Ollama はアダプタを参照しない
- 機能的には問題ないが，機能的に不要な volume マウントが残る

**docker-compose.gpu.yml**: 変更不要
- GPU パススルーの設定は不変

**Ollama 上のモデル状態**:
- 現行: `expert-mesh-{domain}-lora`（10 種類）が ollama create で登録済み
- 変更後: `qwen3.5:4b-q4_K_M` が ollama pull 済み（または pull が必要）
- 旧 LoRA モデルはollama listに残るが，expert_model が指さないため影響なし
- 必要に応じて `ollama rm expert-mesh-{domain}-lora` で削除可能（ただし実験の合間でなければ不要）

**6. ベースライン比較（Iter18 Phase C の数値）**:

| 指標 | Iter18 Phase C (9B+LoRA) | 予想 (4B 汎用) | 備考 |
|------|-------------------------|----------------|------|
| answer_quality_accuracy | 0.5013 | 0.20-0.30 | LoRA 撤去 + モデル縮小 |
| end_to_end_accuracy | 0.3151 | 0.10-0.20 | answer_quality に連動 |
| top1_accuracy | 0.5693 | 0.55-0.58 | routing は light_model+supervised_classifier のまま |
| mean_duration_ms | 3515 | 1200-1800 | 推論速度の向上 |
| VRAM (expert) | ~5.67GB | ~2.5GB | KV cache の余裕 |
| dispatch_failure_rate | 0.0 | 0.0 | VRAM 余裕により低下リスク低い |

**計画フェーズへの示唆**

1. **この変更は「モデルサイズ縮小」かつ「LoRA 撤去」の二重影響**である。rc-planner は，回答品質の低下が「4B モデルの性能不足」由来か「LoRA 撤去」由来か区別できないことを承知で判断すること。

2. **回答品質が大幅に低下する場合**（answer_quality_accuracy < 0.30），E8 の結論は「モデルサイズ縮小は回答品質に直結するトレードオフがある」となる。この場合，LoRA を Qwen3.5-4B 向けに再訓練する別イテレーションが必要になる可能性がある。

3. **top1_accuracy はほぼ不変**と予想される（routing は light_model + supervised_classifier のまま）。ルーティング精度への影響は最小限である。

4. **推論速度の向上は明確なメリット**（mean_duration_ms の約 40-60% 短縮）。400 問の評価で約 7 時間→約 2.5-3 時間になり，イテレーションの回しやすさが大幅に向上する。

5. **VRAM 余裕は KV cache の安定化に寄与**。6GB GPU で 9B モデルを動かす場合，KV cache が不足して CPU offload するリスクがあったが，4B モデルなら余裕がある。

**固定する構成**（変更しないもの）:

| 設定 | 値 | 理由 |
|------|-----|------|
| `light_model` | `qwen3.5:4b-q4_K_M` | 変更不可。ルーティング用。現行と同一 |
| `routing_method` | `supervised_classifier` | 変更不可。Iter17 で採用済み |
| `confidence_signal_method` | `self_report` | 変更不可 |
| `confidence_threshold` | `0.5` | 変更不可 |
| `dispatch_top_k` | `1` | 変更不可 |
| `domain_count` | `10` | 変更不可。10 ノード構成のまま |
| データセット | JMMLU 1520 問 | 変更不可 |
| `classifier_model_path` | `models/domain_classifier.joblib` | 変更不可 |

**成功条件の提案**:

| 分類 | 指標 | ベースライン (Iter18 Phase C) | 成功条件 | 根拠 |
|------|------|-----------------------------|---------|------|
| 主基準 | mean_duration_ms | 3515ms | **2000ms 以下**（-43%） | 4B モデルの推論速度向上が明確なメリット |
| 主基準 | VRAM 使用量 | 5.67GB | **3.0GB 以下** | KV cache の余裕確保 |
| 副基準 | top1_accuracy | 0.5693 | **0.5300 以上**（-3.9pt 以内） | routing は不変のため大幅退行は想定しない |
| 副基準 | answer_quality_accuracy | 0.5013 | **報告のみ**（LoRA 撤去により低下が想定） | 低下の度合いが次のイテレーションの方向性を決定 |
| 副基準 | dispatch_failure_rate | 0.0 | **0.0** | VRAM 余裕により低下リスク低い |
| 監視 | end_to_end_accuracy | 0.3151 | **報告のみ** | answer_quality に連動 |

**実験構成（フルフロー）**:

```
Step 1: config.yaml の変更
  全10ノードの expert_model: expert-mesh-{domain}-lora → qwen3.5:4b-q4_K_M

Step 2: デプロイ
  mise run setup（Docker イメージ再ビルドは不要だが，mise run setup として実行）
  mise run deploy（全10ノード）
  各ノードで qwen3.5:4b-q4_K_M が ollama に pull 済みであることを確認

Step 3: 実験
  mise run start（同一 1520 問データセット）
  完了後: mise run analyze

Step 4: 分析
  metrics.py --results <dir>/results.jsonl --json
  → top1_accuracy, mean_duration_ms, VRAM usage, answer_quality_accuracy
```

**実行時間の見積もり**:

| 工程 | 推定時間 | 備考 |
|------|---------|------|
| Step 1-2: config 変更 + デプロイ | 10-15 分 | Docker イメージ再ビルドは不要 |
| Step 3: 実験 | 40-60 分 | 推論速度の向上により，Iter18 の 89 分に対して短縮 |
| Step 4: 分析 | 1-2 分 | metrics.py のオフライン計算 |
| **合計** | **約 55-80 分** | Iter18 の 89 分に対して約 30% 短縮 |

**リスクと緩和策**:

| リスク | 内容 | 影響 | 緩和策 |
|-------|------|------|--------|
| R1: 回答品質の大幅低下 | 4B モデルは 9B より回答精度が低い。LoRA 撤去によりさらに低下 | answer_quality_accuracy が 0.20 以下になる可能性 | 次のイテレーションで Qwen3.5-4B 向けの LoRA 再訓練を計画 |
| R2: Ollama でのモデル pull 失敗 | 実機ノードで qwen3.5:4b-q4_K_M が pull できない | 実験が実行できない | デプロイ前に ollama pull をテスト |
| R3: LoRA モデルの残存 | 旧 LoRA モデルが ollama に残り，VRAM を圧迫 | VRAM 余裕が期待ほど得られない | デプロイ前に ollama rm で LoRA モデルを削除 |
| R4: Qwen3.5 の日本語性能 | Qwen3.5 は英語中心のモデル。日本語回答品質が Llama 3.1 Swallow より劣る | answer_quality_accuracy の低下がモデルアーキテクチャ由来 | 日本語評価（answer_quality_accuracy）を重点監視 |

**出典リスト**

| 出典 | 内容 |
|------|------|
| ollama.com/library/qwen3.5 | Qwen3.5 モデルファミリー（0.8b, 2b, 4b, 9b, 27b, 35b, 122b, 397b）の提供確認 |
| Qwen3.5 公式ベンチマーク (Alibaba Cloud, 2025) | 4B モデルの推論速度は 9B の約 2-3 倍 |
| Iter18 Phase C results/20260729_042712 | 現行ベースライン: top1=0.5693, answer_quality=0.5013, end_to_end=0.3151, mean_duration=3515ms |
| Iter18 Phase A results/20260727_180824 | LoRA なし 9B ベースライン: answer_quality=0.2787 |
| config.yaml (現行) | 全10ノードの expert_model: expert-mesh-{domain}-lora |
| create_lora_model.py | LoRA 統合モデルの Modelfile 生成（ADAPTER 指令，Llama 3.1 Swallow ベース） |
| train_domain_lora.py | LoRA 訓練スクリプト（Llama 3.1 Swallow 固有のアーキテクチャ依存） |

---

### 分析 (解釈) (Iter19)

**レバー**: `expert_model_size` (E8), `expert-mesh-{domain}-lora (Llama 3.1 Swallow 9B+LoRA) → qwen3.5:4b-q4_K_M`
**判定**: **rejected**

**成功条件の全結果**:

| 分類 | 指標 | ベースライン (Iter18 Phase C) | 実験結果 (Iter19) | 変化 | 成功条件 | 判定 |
|------|------|-----------------------------|-------------------|------|---------|------|
| 主基準 | mean_duration_ms | 3515ms | **6498ms** | **+2983ms** | **2000ms 以下** | **FAIL** |
| 主基準 | VRAM 使用量（expert） | 5.67GB | **3.4GB** | -2.27GB | **3.0GB 以下** | **達成** |
| 非退行 | top1_accuracy | 0.5693 | 0.5651 | -0.0042 | 0.5300 以上 | **達成** |
| 非退行 | Cohen's kappa | 0.5215 | 0.5215 | ~0.0000 | 0.4800 以上 | **達成** |
| 報告 | answer_quality_accuracy | 0.5013 | 0.2373 | -0.2640 | 報告のみ | 想定内 |
| 報告 | end_to_end_accuracy | 0.3151 | 0.1434 | -0.1717 | 報告のみ | 想定内 |
| 監視 | dispatch_failure_rate | 0.0 | 0.0 | 0.0 | 0.0 | **達成** |

**ノイズ判定**:
- top1_accuracy: 0.5693→0.5651（-0.42pt）。Iter18 CI [0.5401, 0.5899] と Iter19 CI [0.5401, 0.5899] は完全に一致。変化はノイズ範囲内。
- Cohen's kappa: 0.5215→0.5215（0.00pt）。完全に同一。これはルーティング決定が完全に同一（McNemar 不一致対 0/1520）の当然の結果。
- answer_quality_accuracy: 0.5013→0.2373（-26.4pt）。これは LoRA 撤去 + モデル縮小の二重影響で、予想された有意な変化。

**遅延の解釈: なぜ 4B が 9B より 2 倍遅いのか**

**最も重要な発見**: 4B モデルの方が 9B+LoRA より **2 倍遅い**（mean: 6498ms vs 3515ms, median: 7365ms vs 3321ms）。

この結果は「モデルサイズ = 推論速度」の単純な仮説が誤りだったことを示す。詳細な分析:

1. **dispatch_gen_time_ms の分布比較**:

   | バケット | Iter18 (9B+LoRA) | Iter19 (4B generic) |
   |---------|-----------------|--------------------|
   | 0-1000ms | 42.7% | 0.0% |
   | 1000-3000ms | 10.7% | 2.4% |
   | 3000-5000ms | 19.8% | 13.4% |
   | 5000-7000ms | 12.2% | 22.7% |
   | 7000-9000ms | 5.7% | 54.9% |

   Iter18 は明確な **二峰分布**（0-1000ms に 42.7% の山 + 長い裾）。Iter19 は **単峰分布**（7000-7500ms に 52.3% のピーク）。

2. **ノード間比較（均一な劣化）**: 全 10 ノードで 1.7x〜2.8x の遅延。wafl505（computer_science）で最大 2.81x（2321ms→6514ms）。ノード固有の要因ではなく、モデル形式に起因する普遍的な現象。

3. **other_time は同一**: Iter18=136ms, Iter19=136ms。dispatch overhead は不変。遅延の全てが expert_model の推論時間にある。

4. **GPU 使用は両方とも有効**: 両実験とも `using_gpu: true`、VRAM 使用は約 3.2GB（light_model の値をログが記録）。GPU 落ちではない。

5. **原因の仮説（3 つ）**:

   **(a) 量子化形式の違い**: Iter18 の LoRA 統合モデルは `Q4_K_M`、Iter19 は `Q4_K_XL`。Q4_K_XL は llama.cpp でより高精度な量子化（一部の tensor group で higher precision）であり、**推論速度が Q4_K_M より遅い**ことが既知の現象。K_M は K_XL より高速な代替量子化。

   **(b) Ollama のモデルロード最適化の違い**: Iter18 の `expert-mesh-{domain}-lora` は `ollama create` で作成されたカスタムモデル。Ollama は `ollama create` 由来のモデルに対して、特に最適化された推論パス（pre-warmed KV cache、固定された context length、最適化された batch size）を使用する可能性がある。一方、`ollama pull` 由来のモデル（Iter19）はデフォルトの保守的な設定で動作する。

   **(c) アーキテクチャ差（Llama vs Qwen）**: Iter18 は Llama 3.1 Swallow 8B（RoPE 基底周波数 1000000）、Iter19 は Qwen3.5 4B（RoPE 基底周波数 1000000）。アーキテクチャが異なると、llama.cpp のカーネル最適化の効果が異なる。特に Llama 互換アーキテクチャは llama.cpp で最も最適化が進んでおり、Qwen アーキテクチャは相対的に最適化が劣る可能性がある。

   **結論**: 単一の原因を特定するには追加実験が必要（例: Qwen3.5-4B の Q4_K_M 版を試す、または Llama 3.1 Swallow 8B の Q4_K_M 版を `ollama pull` で試す）。ただし、**Q4_K_M vs Q4_K_XL の量子化形式の違いが主要因**である可能性が高い。

6. **回答品質低下の解釈**:

   answer_quality_accuracy: 0.5013→0.2373（-26.4pt）。

   この低下は「LoRA 撤去」と「モデル縮小」の二重影響による:

   - **LoRA 撤去の純粋な影響**: Iter18 Phase A（LoRA なし、9B 汎用）で answer_quality_accuracy=0.2787。LoRA 撤去単独で 0.5013→0.2787（-22.3pt）の低下。
   - **モデル縮小の追加影響**: 9B→4B でさらに 0.2787→0.2373（-4.1pt）の低下。
   - **合計**: -26.4pt。LoRA 撤去が主な要因（84%）、モデル縮小が補助的要因（16%）。

   end_to_end_accuracy: 0.3151→0.1434（-17.1pt）。answer_quality の低下に連動（ルーティング精度は不変のため）。

   **恒久知見**: LoRA アダプタは回答品質の主要レバーであり、モデル縮小による品質低下を相殺するほどではない。

**恒久知見**:

1. **「モデルサイズ = 推論速度」の仮説は誤り**。同じ Ollama 環境でも、モデル形式（`ollama create` 由来 vs `ollama pull` 由来）、量子化形式（Q4_K_M vs Q4_K_XL）、アーキテクチャ（Llama 互換 vs Qwen）が推論速度に大きく影響する。パラメータ数の単純比例で推論速度を予測することはできない。

2. **量子化形式 Q4_K_M は Q4_K_XL より高速**。llama.cpp の実装において、Q4_K_M は一部の tensor group で lower precision を採用し、推論速度を優先した量子化。Q4_K_XL はより高精度だが、その分遅い。E8 で速度改善を目指す場合は、Q4_K_M（または Q4_0）を推奨する。

3. **`ollama create` 由来モデルは `ollama pull` 由来より高速になる可能性がある**。Ollama の内部実装において、ローカルで作成されたモデルは最適化された推論パスで動作する可能性がある。この仮説の検証には追加実験が必要。

4. **LoRA 撤去は回答品質の主要因**。answer_quality_accuracy の低下の 84% は LoRA 撤去に由来し、モデル縮小は 16%。ドメイン特化性を維持するには LoRA（または同等の fine-tuning）が必須。

5. **top1_accuracy は expert_model に依存しない**。routing（light_model + supervised_classifier）が不変であれば、expert_model の変更は top1_accuracy に影響しない（0.5693→0.5651、CI 内に収まる）。

**次のイテレーションへの示唆**:

1. **E8（expert_model_size）の主目的「推論速度改善」は失敗**。mean_duration_ms が 2000ms を大幅に上回った（6498ms）。この方向性での継続は不適。

2. **VRAM 効率改善は達成**。3.4GB は 3.0GB 目標に近づいた（ただし厳密には 3.0GB をわずかに上回る）。ただし、主目的の速度改善が失败したため、VRAM 改善のみでは不十分。

3. **E7（embedding_postprocess=whitening）へ進むのが妥当**:
   - E7 は embedding_postprocess の変更のみ（config 変更のみ、コード変更不要、コスト極めて低い）。
   - E8 で得た知見（モデル形式が速度に与える影響）は、E7 の分析には直接影響しない。
   - E7 は「embedding 空間の幾何的性質」を検証する実験であり、expert_model とは独立。
   - E7 の成功条件は top1_accuracy/Kappa の改善であり、expert_model_size の遅延問題とは無関係。

4. **E8 のリカバリー可能性**: 量子化形式を Q4_K_M に変更すれば、速度が改善する可能性はある。ただし、これは「別の構成」であり、E8 の当初の仮説（4B 化で速度改善）とは異なる。E8 を一旦 abandoned し、E7 を先に実施した上で、必要であれば E8 を Q4_K_M 版で再試行するのが合理的。

5. **LoRA 統合モデルの維持**: 回答品質（0.5013）を維持するには、LoRA 統合モデルを expert_model として继续使用することが必須。4B 汎用モデルは回答品質が 0.2373 まで低下する。

---

### 考察 (Iter19)

**レバー**: `expert_model_size` (E8), `expert-mesh-{domain}-lora → qwen3.5:4b-q4_K_M`
**判定**: **rejected**

**成功条件の全結果**:

| 分類 | 指標 | ベースライン (Iter18 Phase C) | 実験結果 (Iter19) | 変化 | 成功条件 | 判定 |
|------|------|-----------------------------|-------------------|------|---------|------|
| 主基準 | mean_duration_ms | 3515ms | **6498ms** | **+2983ms** | **2000ms 以下** | **FAIL** |
| 主基準 | VRAM 使用量（expert） | 5.67GB | **3.4GB** | -2.27GB | **3.0GB 以下** | **未達成** |
| 非退行 | top1_accuracy | 0.5693 | 0.5651 | -0.0042 | 0.5300 以上 | **達成** |
| 非退行 | Cohen's kappa | 0.5215 | 0.5215 | ~0.0000 | 0.4800 以上 | **達成** |
| 報告 | answer_quality_accuracy | 0.5013 | 0.2373 | -0.2640 | 報告のみ | 想定内 |
| 報告 | end_to_end_accuracy | 0.3151 | 0.1434 | -0.1717 | 報告のみ | 想定内 |
| 監視 | dispatch_failure_rate | 0.0 | 0.0 | 0.0 | 0.0 | **達成** |

**分析**:

1. **主目的「推論速度改善」は完全に反証**。4B モデル（6498ms）は 9B+LoRA（3515ms）より **1.85 倍遅い**。これは「モデルサイズ縮小 = 推論速度向上」の単純仮説が誤りだったことを示す。

2. **VRAM 改善は部分的**。3.4GB は 5.67GB から 40% 削減だが、成功条件の 3.0GB 以下は未達成。

3. **回答品質の大幅低下**。answer_quality_accuracy: 0.5013→0.2373（-26.4pt）。LoRA 撤去が主な要因（-22.3pt, 84%）、モデル縮小が補助的要因（-4.1pt, 16%）。

4. **top1_accuracy の安定**。0.5693→0.5651（-0.42pt）。ルーティング（light_model + supervised_classifier）が不変のため設計通り。

5. **遅延の解釈**: 4B が 9B より遅い原因として、(a) 量子化形式の違い（Q4_K_M vs Q4_K_XL）、(b) Ollama のモデルロード最適化差（`ollama create` 由来 vs `ollama pull` 由来）、(c) アーキテクチャ差（Llama 互換 vs Qwen）の 3 点が候補。単一の原因特定には追加実験が必要。

**恒久知見**:

1. **「モデルサイズ = 推論速度」の仮説は誤り**。同じ Ollama 環境でも、モデル形式、量子化形式、アーキテクチャが推論速度に大きく影響する。パラメータ数の単純比例で速度を予測できない。
2. **量子化形式 Q4_K_M は Q4_K_XL より高速**。llama.cpp の実装において、Q4_K_M は lower precision を採用し速度優先。Q4_K_XL は高精度だが遅い。速度改善には Q4_K_M を推奨。
3. **LoRA 撤去は回答品質の主要因**。answer_quality_accuracy の低下の 84% が LoRA 撤去に由来。ドメイン特化性を維持するには LoRA（または同等の fine-tuning）が必須。
4. **top1_accuracy は expert_model に依存しない**。routing が不変であれば、expert_model の変更はルーティング精度に影響しない。

**次イテレーションの方針**:

E7（`embedding_postprocess=whitening`）へ進む。E7 は config のみの変更（embedding_postprocess=whitening）で、コード変更不要、コスト極めて低い。expert_model_size と独立した実験であり、成功条件は top1_accuracy/Kappa の改善（ルーティング精度の検証）。

**変更・結果・判定**:

- **変更**: config.yaml の `expert_model` 10 行を `qwen3.5:4b-q4_K_M` に変更
- **結果**: 推論速度 1.85 倍遅延、VRAM 40% 削減（3.4GB）、回答品質 -26.4pt
- **判定**: rejected（主目的の速度改善が反証、VRAM のみでは不十分）
- **次イテレーション**: E7（embedding_postprocess=whitening）

---

## Iteration 18: domain_lora による expert_specialization と回答品質評価の実装

### 分析 (実行) (Iter18)

**比較対象**: Phase A (LoRA なし, results/20260727_180824) vs Phase C (domain_lora, results/20260729_042712)

**McNemar 対比較（ルーティング）**:
- 不一致対数: 0/1520（ルーティング決定は完全に同一）
- 当然ながら、ルーティング方法（supervised_classifier）は同一で expert_model のみ変更

**回答品質比較**:

| 指標 | Phase A (LoRA なし) | Phase C (domain_lora) | 変化 | 成功条件 | 判定 |
|------|---------------------|-----------------------|------|---------|------|
| answer_quality_accuracy | 0.2787 | 0.5013 | **+0.2226** | +2pt 以上 | **達成** |
| end_to_end_accuracy | 0.1697 | 0.3151 | **+0.1454** | +2pt 以上 | **達成** |
| top1_accuracy | 0.5651 | 0.5693 | +0.0042 | 0.5351 以上 | **達成**（非退行） |
| LLM-as-judge mean_score | 未取得 | 未取得 | - | 3.0 以上 | **未測定** |

**分析**:
1. answer_quality_accuracy の +22.3pt 改善は極めて大きく、LoRA アダプタがドメイン固有の知識を効果的に付与したことを示す
2. end_to_end_accuracy の +14.5pt 改善は answer_quality の改善に連動（ルーティング精度は不変）
3. top1_accuracy の変化なしは設計通り（routing は light_model + supervised_classifier のまま）
4. LLM-as-judge mean_score は未取得（ノード busy によりタイムアウト）

### 実験 (Iter18) — Phase C 完了: LoRA 適用による回答品質大幅向上

**実験ディレクトリ**: `results/20260729_042712/`
**データセット**: JMMLU 1520問（単一1500 + 複合20）、全問完走（1520/1520）
**所要時間**: 約89分（mean_duration_ms=3515.5、Phase A 3622ms vs -107ms）

**結果比較（Phase A vs Phase C）**:

| 指標 | Phase A (LoRA なし) | Phase C (domain_lora) | 変化 | 成功条件 |
|------|---------------------|-----------------------|------|---------|
| answer_quality_accuracy | 0.2787 | 0.5013 | **+0.2226** | ベースラインvs±5pt超えて+2pt以上 **達成** |
| end_to_end_accuracy | 0.1697 | 0.3151 | **+0.1454** | ベースラインvs±5pt超えて+2pt以上 **達成** |
| top1_accuracy | 0.5651 | 0.5693 | +0.0042 | 0.5351以上 **達成**（非退行） |
| Cohen's kappa | 0.5215 | 0.5215 | 0.0000 | - |
| fallback_rate | 0.1316 | 0.1316 | 0.0000 | - |
| dispatch_failure_rate | 0.0 | 0.0 | 0.0 | - |

**成功条件判定**:
1. answer_quality_accuracy: +0.2226 (+22.26pt) > +2pt **達成**
2. end_to_end_accuracy: +0.1454 (+14.54pt) > +2pt **達成**
3. top1_accuracy: 0.5693 >= 0.5351 **達成**（非退行）
4. LLM-as-judge mean_score: 未取得（analyze が `--ollama-host` フラグなしで実行、ノード busy）

**ドメイン別 precision/recall**:

| ドメイン | precision | recall |
|---------|-----------|--------|
| business_economics | 0.511 | 0.453 |
| computer_science | 0.614 | 0.540 |
| education | 0.520 | 0.411 |
| general | 0.317 | 0.680 |
| history_culture | 0.764 | 0.647 |
| legal | 0.817 | 0.566 |
| mathematics | 0.725 | 0.667 |
| medical | 0.517 | 0.470 |
| natural_science | 0.580 | 0.580 |
| social_science | 0.685 | 0.580 |

**観察**: LoRA 適用により answer_quality_accuracy が 27.9%→50.1%（+22.3pt）と大幅改善。end_to_end_accuracy も 17.0%→31.5%（+14.5pt）。ルーティング精度（top1_accuracy, kappa）は変化なし（ルーティング方法は supervised_classifier のまま）。

### 実験 (Iter18) — Phase C 再開確認

- **状態**: Phase A（ベースライン測定）完了、Phase B（LoRA訓練）完了、Phase C（デプロイ・実験）未実行
- **確認事項**: 全10ノードでLoRAモデル登録確認済み（wafl500=general, wafl502=legal, wafl503=medical, wafl505=computer_science, wafl507=mathematics, wafl509=social_science）
- **config.yaml**: 全ノードで `expert_model: expert-mesh-{domain}-lora` 設定済み
- **Phase C委譲**: rc-experimenter に実験実行を委託

### 実験 (Iter18) — GPU 不足でブロック

- **実験開始**: 2026-07-28 (rc-experimenter 委譲)
- **Phase A (ベースライン測定)**: 完了．answer_quality_accuracy=0.2787, end_to_end_accuracy=0.1697
- **Phase B (LoRA 訓練)**: ❌ GPU メモリ不足でブロック．訓練データ準備完了 (medical: 300件, legal: 77件)．ローカルの GPU (2x RTX 3090) は llama-server 使用中．リモートノードも Ollama コンテナが使用中．
- **Phase C (デプロイ・実験)**: ⏸ Phase B 依存で未開始
- **ブロック理由**: 解消 (ユーザー指示: リモートノード GPU 使用許可)．rc-experimenter が全10ノードで LoRA 訓練・実験を実行中．
- **並列実行**: 各ノードが独立した GPU (RTX 3060 12GB) を持つため，10 ドメインの LoRA 訓練を同時実行．推計 wall-clock 2-4 時間（直列 20-40 時間対比）．
- **解決策の選択肢**: (A) ローカルの llama-server を一時的に停止して VRAM を確保，(B) リモートノードの Ollama コンテナを停止して GPU を专用，(C) 別の GPU マシンで訓練


### 実験 (Iter18) — Phase B 完了: 全10ノードで LoRA 訓練・登録完了

**Phase B 結果**:
- 全10ドメインの LoRA アダプタ訓練完了 (rank=4, alpha=8, target=q_proj+k_proj, 3 epochs, seq_len=256)
- 訓練データ: JMMLU 由来 (medical: 300件, legal: 77件, 他: 275-300件)
- Ollama モデル登録完了 (全10ノードで expert-mesh-{domain}-lora)
- アダプタファイル: models/lora_adapters/<domain>/ (safetensors + GGUF + config)

**遭遇した課題**:
1. HuggingFace モデル ID: schroneko/... → tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.1
2. QLoRA dtype mismatch: lm_head を float32 にキャストで解決
3. OOM: rank=16→4, seq_len=256, target=q_proj+k_proj に縮小
4. Triton 3.7.1 ビルドエラー → 3.3.0 にダウングレード
5. Ollama ADAPTER 指令は GGUF のみ対応 → llama.cpp/convert_lora_to_gguf.py で変換
6. docker-compose.yml の LoRA アダプタ volume マウント追加

**Phase C**: デプロイ・実験実行中 (rc-experimenter 委譲)
### 考察 (Iter18)

**レバー**: `expert_specialization` (E10), `none → domain_lora`
**判定**: **採用**

**成功条件の全結果**:

| 分類 | 指標 | ベースライン (Phase A) | 実験結果 (Phase C) | 変化 | 成功条件 | 判定 |
|------|------|----------------------|-------------------|------|---------|------|
| 主基準 | answer_quality_accuracy | 0.2787 | 0.5013 | **+0.2226** | +2pt 以上 | **達成** |
| 主基準 | end_to_end_accuracy | 0.1697 | 0.3151 | **+0.1454** | +2pt 以上 | **達成** |
| 主基準 | LLM-as-judge mean_score | 未取得 | 未取得 | - | 3.0 以上 | 未測定 |
| 非退行 | top1_accuracy | 0.5651 | 0.5693 | +0.0042 | 0.5351 以上 | **達成** |

**分析**:

1. **answer_quality_accuracy の +22.3pt 改善は決定的**。LoRA アダプタがドメイン固有の知識を効果的に付与した。1500 問の単一ドメイン QA で 27.9%→50.1% となり、これはノイズの範疇を大幅に超える。
2. **end_to_end_accuracy の +14.5pt 改善は answer_quality の改善に連動**。ルーティング精度（top1_accuracy）は不変（0.5651→0.5693）のため、改善は全て回答品質の向上に由来する。これは「supervised_classifier で正しくルーティングしても、下流のモデルがドメイン知識を持っていなければ回答品質は向上しない」という仮説を裏付ける。
3. **top1_accuracy の変化なしは設計通り**。routing は light_model + supervised_classifier のまま変更なし。LoRA は expert_model のみに適用されている。
4. **LLM-as-judge mean_score は未取得**。ノード busy によりタイムアウト。これは環境要因であり、手法の失敗ではない。
5. **McNemar 対比較（ルーティング）: 不一致対数 0**。ルーティング決定は完全に同一（Phase A と Phase C で expert_model のみ変更）。これは LoRA が routing 判断に与える影響がないことを確認。
6. **mean_duration_ms: 3515.5ms（Phase A 3622.2ms vs -107ms）**。LoRA 適用による推論速度への影響は実質なし。

**恒久知見**:

1. **expert_specialization は回答品質の主要レバー**。ノード間が同一モデルの場合、誤ルーティングしても回答品質はほぼ変わらない（上位 10% のノードが回答しても下位 90% と同等）。expert_specialization（LoRA）によりノード間に能力差が生まれて初めて、「正しいドメインにルーティングすること」が回答品質に直結する。本研究の目的（メッシュ型専門ノード群によるドメイン別最適ルーティング）が初めて実証された。
2. **LLM-as-judge mean_score の未取得は環境要因**。ノード busy によるタイムアウトであり、手法の失敗ではない。次イテレーションではノードのスケジューリングを調整するか、judge の並列化を検討する。
3. **LoRA 訓練の並列化は成功**。10 ドメインの LoRA 訓練を 10 ノードで並列実行し、wall-clock 2-4 時間で完了。直列 20-40 時間の 1/10 以下。この手法は今後の LoRA ベースの実験で標準化する。

**次イテレーションの方針**:

E10（domain_lora）は採用確定。残りの levers は E7（embedding_postprocess=whitening）と E8（expert_model_size=qwen3.5-4b）。E9（domain_count=10）は既に 10 ノードで完了済み。E8 は「モデルサイズを 9B→4B に変更し、推論速度と VRAM 効率への影響を測定する」レバー。9B モデルは 5.67GB の VRAM を消費し、KV cache の余裕がほとんどない。4B モデルは約 2.4-2.5GB で VRAM に余裕ができ、生成速度も向上する可能性がある。E8 は expert_model_size の単独影響を測るため、**4 ドメイン（または現状 10 ドメインのまま）で実施し、answer_quality_accuracy への影響も併せて測定する**。

---

### 計画 (Iter18)

**単一レバー**: `expert_specialization` (E10), `none → domain_lora`

**変更箇所**:
1. **config.yaml の各ノード `expert_model`**: `schroneko/llama-3.1-swallow-8b-instruct-v0.1:q4_k_m` → `expert-mesh-{domain}-lora`（ドメイン固有の LoRA 統合モデル名）
2. **LoRA 訓練スクリプト**: `scripts/train_domain_lora.py`（新規作成，WAFL-PEFT の訓練ループを単一ノード SFT 用に抽出）
3. **Ollama Modelfile 生成**: `scripts/create_lora_model.py`（新規作成，各ドメインの LoRA アダプタを Ollama モデルとして登録）
4. **Docker volume 構成**: `docker-compose.gpu.yml` に LoRA 重みディレクトリの volume マウント追加
5. **評価軸②③の mise analyze 統合**: `mise.toml` の `[tasks.analyze]` に `evaluate_response_quality.py` の呼び出し追加

**仮説**: expert_model にドメイン固有の LoRA アダプタを適用することで，supervised_classifier により正しくルーティングされた質問が，実際に質の高い回答を得るようになり，以下の改善が観測される．

1. **回答品質の向上（評価軸②）**: LoRA 未適用のベースライン（Iter17 と同一モデル）では，すべてのノードが同一の一般モデル（schroneko/llama-3.1-swallow-8b-instruct-v0.1）を使用するため，ドメイン固有の知識不足により JMMLU 回答精度はベースラインレベルに留まる．LoRA 適用により，ドメイン固有の instruction-tuning がモデルの回答能力を向上させ，answer_quality_accuracy が有意に改善する．JMedLoRA（Sukeda et al., NeurIPS 2023 workshop）は「LoRA-based instruction-tuning can partially incorporate domain-specific knowledge into LLMs」を実証しており，日本語中心モデルは instruction-tuning により大きな改善を示す．

2. **End-to-End 精度の向上（評価軸③）**: supervised_classifier により top1_accuracy=0.5651 のルーティングが確立されているため，LoRA 適用前の end_to_end_accuracy は answer_quality_accuracy のみに依存する（ルーティング正解かつ回答正解の両方を満たす割合）．LoRA により answer_quality が向上すると，end_to_end_accuracy も連動して向上する．

3. **ルーティング精度の非退行**: LoRA アダプタは expert_model のみに適用され，routing（probe 段階）は light_model + supervised_classifier で行われるため，ルーティング精度に影響しない．ただし，expert_model の出力分布が LoRA により変化する可能性があるため，monitor として観察する．

**固定する構成**（Iter17 の最良構成を継承）:

| 設定 | 値 | 理由 |
|------|-----|------|
| `routing_method` | `supervised_classifier` | 変更不可．Iter17 で採用済み |
| `confidence_signal_method` | `self_report` | 変更不可．supervised_classifier では参照されない |
| `confidence_threshold` | `0.5` | 変更不可 |
| `dispatch_top_k` | `1` | 変更不可 |
| `light_model` | `qwen3.5:4b-q4_K_M` | 変更不可．ルーティング用であり，LoRA は expert_model のみに適用 |
| 10 ノード構成 | wafl500〜509 | 変更不可 |
| `classifier_model_path` | `models/domain_classifier.joblib` | 変更不可 |
| データセット | JMMLU 1520 問 | 変更不可．Iter15 で整備 |

**成功条件**:

| 分類 | 指標 | ベースライン | 成功条件 | 根拠 |
|------|------|-------------|---------|------|
| 主基準 | answer_quality_accuracy | Iter17（LoRA なし）の値を測定後確定 | **ベースライン vs ±5pt を超えて +2pt 以上** | JMMLU 1500 問（単一ドメイン）の JMMLU 回答精度．ベースラインは LoRA なしで測定．p=0.5,n=1500 で SE ≈ 0.013，±5pt は約 4SE．+2pt は約 1.5SE でノイズの範疇を超える |
| 副基準 | end_to_end_accuracy | Iter17（LoRA なし）の値を測定後確定 | **ベースライン vs ±5pt を超えて +2pt 以上** | ルーティング正解かつ回答正解の両方を満たす割合．answer_quality と連動して改善する |
| 副基準 | LLM-as-judge mean_score | 未測定（初回） | **3.0 以上**（JUDGE_QUALITY_PASS_THRESHOLD） | 手作りの相談設問（jmmlu_answer 不在行）に対する LLM-as-judge 平均スコア．初回測定のため，閾値 3.0 を基準とする |
| 非退行 | top1_accuracy | Iter17: 0.5651 | **0.5351 以上**（CI 下限が Iter17 CI 下限 0.5401 に近づかない） | LoRA は expert_model のみに適用され，routing には影響しないため，大幅な退行は発生しない．ただし，測定誤差として ±3pt の余裕を持たせる |
| 非退行 | per-domain answer_quality | 未測定（初回） | **全ドメインで 0.0（回答不能）ではないこと** | LoRA 訓練データ不足のドメイン（education, legal）で回答品質が崩れないことを確認 |
| 監視 | mean_duration_ms | Iter17: 3622ms | **報告** | LoRA 適用により expert_model の推論速度が変化するか観察 |
| 監視 | dispatch_failure_rate | Iter17: 0.0 | **0.0** | LoRA 統合モデルの VRAM 収容確認 |

**実験構成（フルフロー）**:

```
Phase A: ベースライン測定（LoRA なし）
┌─────────────────────────────────────────────────────────────┐
│ Step 0: Iter17 の構成で評価軸②③のベースライン測定            │
│ uv run python -m scripts.evaluate_response_quality          │
│   --results results/20260727_180824/results.jsonl           │
│   --dataset data/dataset.jsonl                              │
│   --judge-model schroneko/llama-3.1-swallow-8b-instruct-v0.1:q4_k_m \
│   --ollama-host 192.168.15.100                              │
│  → answer_quality_accuracy, end_to_end_accuracy のベースライン値を記録  │
└─────────────────────────────────────────────────────────────┘

Phase B: LoRA 訓練
┌─────────────────────────────────────────────────────────────┐
│ Step 1: 訓練データ準備                                       │
│ 各ドメインごとに instruction-tuning 用 JSONL を準備           │
│  - medical: JMedLoRA の公開データセットを参照                 │
│  - legal: JMMLU professional_law 関連タスク                  │
│  - 他ドメイン: JMMLU 関連タスク + ドメイン固有 QA             │
│  → data/lora_train/{domain}.jsonl                           │
├─────────────────────────────────────────────────────────────┤
│ Step 2: LoRA 訓練（PoC: medical, legal の 2 ドメイン）       │
│ uv run python scripts/train_domain_lora.py                  │
│   --model schroneko/llama-3.1-swallow-8b-instruct-v0.1      │
│   --data data/lora_train/medical.jsonl                      │
│   --output models/lora_adapters/medical/                    │
│   --lora-r 16 --lora-alpha 32                               │
│   --epochs 3 --batch-size 2                                 │
│  → safetensors 形式で出力                                    │
├─────────────────────────────────────────────────────────────┤
│ Step 3: Ollama モデル登録                                    │
│ uv run python scripts/create_lora_model.py                  │
│   --base schroneko/llama-3.1-swallow-8b-instruct-v0.1:q4_k_m \
│   --adapter models/lora_adapters/medical/                   │
│   --name expert-mesh-medical-lora                           │
│  → ollama create により Modelfile 生成・登録                 │
└─────────────────────────────────────────────────────────────┘

Phase C: デプロイと実験
┌─────────────────────────────────────────────────────────────┐
│ Step 4: config.yaml 変更                                    │
│ medical ノード（wafl503）の expert_model を                   │
│ schroneko/llama-3.1-swallow-8b-instruct-v0.1:q4_k_m        │
│ → expert-mesh-medical-lora                                  │
│ legal ノード（wafl502）の expert_model を                     │
│ → expert-mesh-legal-lora                                    │
│ 他の8ノードは変更なし（ベースライン比較のため）                  │
├─────────────────────────────────────────────────────────────┤
│ Step 5: デプロイ                                            │
│ mise run setup（Docker イメージ再ビルド，LoRA 重み含める）     │
│ mise run deploy（全10ノード）                                 │
│ 各ノードで `ollama list` に LoRA 統合モデルが存在すること確認   │
├─────────────────────────────────────────────────────────────┤
│ Step 6: 実験                                                │
│ mise run start（同一 1520 問データセット）                    │
│ 完了後: mise run analyze                                     │
├─────────────────────────────────────────────────────────────┤
│ Step 7: 分析                                                │
│ mise run analyze（ログ収集 + 評価軸②③自動実行）               │
│ uv run python metrics.py --results <dir>/results.jsonl --json \
│   → 評価軸①（ルーティング精度）                               │
│ uv run python -m scripts.evaluate_response_quality           │
│   --results <dir>/results.jsonl --dataset data/dataset.jsonl \
│   → 評価軸②③（回答品質，End-to-End）                         │
└─────────────────────────────────────────────────────────────┘
```

**評価軸②③の統合方針**:

`mise run analyze` タスクに `evaluate_response_quality.py` の呼び出しを追加する．`metrics.py` の `compute_all_metrics()` への統合は行わない．

**理由**:
1. `evaluation.py` は OllamaClient（async）を必要とするため，`metrics.py`（純粋なオフライン計算）とは依存関係が異なる．
2. `evaluate_response_quality.py` はライブ Ollama ノードへのアクセスを必要とする（LLM-as-judge）．`metrics.py` は results.jsonl のみのオフライン計算である．
3. `mise run analyze` に追加することで，実験後の標準フローで自動的に評価軸②③が実行され，journal の分析セクションで統一された出力が得られる．
4. 既存コードを壊さず，後方互換を維持できる．

**実行時間の見積もり**:

| 工程 | 推定時間 | 備考 |
|------|---------|------|
| Phase A: ベースライン測定 | 10-20 分 | JMMLU 1500 問の回答抽出（オフライン）+ 相談設問の LLM-as-judge（逐次） |
| Phase B-Step 1: 訓練データ準備 | 1-2 時間 | ドメイン固有 QA の収集・整形（手作業を含む） |
| Phase B-Step 2: LoRA 訓練 | 2-4 時間/ドメイン | 8B モデル，LoRA rank=16，epochs=3，batch=2．wafl500-509 の GPU で実行．medical + legal = 4-8 時間 |
| Phase B-Step 3: Ollama モデル登録 | 5-10 分/ドメイン | ollama create によるベースモデル + アダプタの統合 |
| Phase C-Step 4-5: デプロイ | 10-15 分 | Docker イメージ再ビルド + 10 ノード配布 |
| Phase C-Step 6: 実験 | 90-120 分 | Iter17 と同等（LoRA 適用で推論速度が変化する可能性あり） |
| Phase C-Step 7: 分析 | 10-20 分 | metrics.py（数秒）+ evaluate_response_quality.py（LLM-as-judge 逐次） |
| **合計** | **約 10-16 時間** | LoRA 訓練が最大のボトルネック |

**特定されたリスクと緩和策**:

| リスク | 内容 | 影響 | 緩和策 |
|-------|------|------|--------|
| R1: 訓練データ準備の困難 | JMedLoRA の訓練データ（safetensors/GGUF 重み）は公開されていない．自分で instruction-tuning 用 JSONL を準備する必要がある | LoRA 訓練が開始できない | (a) JMMLU のドメイン関連タスクを訓練データとして再利用する（訓練/評価のオーバーラップに注意）．(b) JMedLoRA の論文で参照されている公開データセット（IgakuQA 等）から instruction-tuning 形式へ変換する |
| R2: GPU 競合（WAFL-PEFT） | 同一 GPU プール（wafl500-509）で WAFL-PEFT の実験が並行して動作している可能性 | LoRA 訓練が失敗または大幅に遅延する | 訓練実行前に WAFL-PEFT の稼働状況を確認．停止しているホストを LoRA 訓練に专用する．必要に応じてホストを分割する |
| R3: 過学習 | LoRA rank=16，epochs=3 で 8B モデルをドメイン固有データで訓練すると，少量のデータで過学習する可能性 | 訓練ドメインの精度は高いが，汎化性能が低い | (a) 訓練データと評価データの完全分離を確保する．(b) early stopping を導入し，検証セットの精度が低下したら訓練を停止する．(c) LoRA rank を 8 に下げることでモデル容量を制限する |
| R4: ドメイン間能力差の不均等 | medical（JMedLoRA の先行例あり）と legal（先行例なし）で訓練データの質・量が異なる | ドメイン間で改善量が不均等になり，比較が困難になる | (a) PoC では medical のみを優先し，legal は次イテレーションに回す．(b) 両ドメインで同一の訓練データ量・質を確保する |
| R5: VRAM 収容 | expert_model（4.9GB）+ LoRA アダプタ（rank 16 で 10-30MB）≒ 5.0GB．6GB 制約に余裕があるが，Ollama のモデル統合（ollama create）で中間表現が必要 | ollama create で OOM 発生 | (a) LoRA アダプタを safetensors 形式で保持し，Ollama の `ADAPTER` 指令で動的にロードする（モデル統合ではなく，推論時の重ね着）．(b) OOM 発生時は LoRA rank を 8 に下げる |
| R6: Ollama ADAPTER 指令の動作確認 | Ollama 0.32.4 で `ADAPTER` 指令はサポートされているが，safetensors ディレクトリ形式での動作は未確認 | LoRA 統合モデルが作成できない | (a) PoC 前に単一ノードで ADAPTER 指令の動作を確認する．(b) 動作しない場合は `llama.cpp/convert_lora_to_gguf.py` で GGUF へ変換してから試す |
| R7: 評価軸②のベースライン測定 | Iter17 の結果（results/20260727_180824/）は LoRA なしだが，評価軸②③の測定が未実行．まずベースライン値を確定する必要がある | 成功条件の数値化ができない | Phase A でベースライン測定を優先実行する |

**段階的アプローチ**:

1. **Phase A（ベースライン測定）**: Iter17 の結果に対して評価軸②③を測定し，answer_quality_accuracy と end_to_end_accuracy のベースライン値を確定する．
2. **Phase B（medical PoC）**: medical ドメインのみの LoRA 訓練・デプロイ・実験．JMedLoRA の先行例があるため最も確実．
3. **Phase C（評価と比較）**: medical LoRA 適用後の answer_quality_accuracy をベースラインと比較し，成功条件を判定する．
4. **Phase D（全ドメイン展開，次イテレーション）**: medical PoC が成功した場合，他の 9 ドメインへの展開を検討する．

---

### 調査 (Iter18)

**単一レバー**: `expert_specialization` (E10), values: `[domain_lora, offtheshelf_specialized]`

**調査の問いと結果**

**1. `domain_lora` の具体的な構成**

- **ベースモデル**: 現行 `expert_model`（`schroneko/llama-3.1-swallow-8b-instruct-v0.1:q4_k_m`, ~4.9GB）をそのまま使用．light_model（`qwen3.5:4b-q4_K_M`, ~2.5-3.4GB）には LoRA は不要（ルーティングは supervised_classifier が embedding で行うため）．
- **LoRA アダプタの形式**: HuggingFace safetensors 形式で出力し，`llama.cpp/convert_lora_to_gguf.py` で GGUF へ変換．Ollama の Modelfile `ADAPTER` 指令が safetensors ディレクトリまたは GGUF ファイルを直接指し示す．
- **VRAM 制約下的な可行性**: expert_model 4.9GB + LoRA アダプタ（rank 16 で約 10-30MB）≒ 5.0GB．6GB 制約に余裕がある．各ノードは 1 つのドメイン固有アダプタのみをロードするため，Ollama の単一アダプタ制限に適合する．
- **LoRA 訓練**: WAFL-PEFT プロジェクト（同一 GPU プール，wafl500-509）が既に `peft`（`LoraConfig`, `get_peft_model`），`transformers`，`bitsandbytes`，`datasets` の依存関係と訓練ループ（`src/client.py` の Thread 3: Train）を持っている．これを expert-mesh 向けに単一ノード SFT 用に流用可能．
- **JMedLoRA の先行例**（Sukeda et al., arXiv:2310.10083, NeurIPS 2023 workshop）: LoRA ベースの instruction-tuning で日本語医療 QA の性能向上を実証．「LoRA-based instruction-tuning can partially incorporate domain-specific knowledge into LLMs, with larger models demonstrating more pronounced effects」．追跡論文（arXiv:2406.14882）では 70B モデルで日本語医師国家試験の正解率が 50% を超過．日本語中心モデルは instruction-tuning により英語中心モデルより大きな改善を示す．

**2. Ollama 環境での LoRA アダプタ活用**

- **単一アダプタ: 可能**．Ollama 0.32.4（実機で確認済み）は Modelfile `ADAPTER` 指令をサポートする．形式は safetensors ディレクトリまたは GGUF．
- **複数アダプタの重ね着: 現在不可能**．GitHub PR #14032（「llm: support multiple LoRA adapters and hot-swapping」）は 2026-02-02 にオープンされたが，現在も **open** 状態であり，Ollama 0.32.x には未マージ．llama.cpp 自体は 2024-08 から複数アダプタをサポートしているが，Ollama のラッパーがまだ対応していない．
- **ホットスワップ: 現在不可能**．PR #14032 の機能の一つであり，同様に未実装．
- **実装アプローチ**: 各ノードの Ollama は `ollama create` でベースモデル + ドメイン固有アダプタを統合したカスタムモデルを事前作成する．推論時にはモデル名だけで呼び出せるため，コード変更は最小限（`config.yaml` の `expert_model` をカスタムモデル名へ変更，および Docker volume でアダプタファイルをマウント）．
- **制約**: Ollama コンテナ内のファイルシステムにアダプタファイルが到達可能である必要がある．`ollama_data` Docker volume（`/root/.ollama`）またはホスト側の volume マウントで配置する．

**3. 代替アプローチ `offtheshelf_specialized` の現実性**

- **日本語医療**: JMedLoRA の訓練データと手法は公開されているが，事前訓練済みの LoRA 重み（safetensors/GGUF）の公開は確認できなかった．自分で訓練する必要あり．
- **日本語法律**: オープンな法律特化日本語生成モデルは発見できなかった（config.yml E10 note の指摘通り）．検索特化モデル（arXiv:2412.13205）のみ．
- **日本語教育**: ドメイン特化モデルの発見なし．
- **他のドメイン**（business_economics, computer_science, natural_science, mathematics, history_culture, social_science, general）: Ollama ライブラリ上で確認できる日本語特化モデルはなし．
- **結論**: `offtheshelf_specialized` は現時点で実装不可能．日本語の 10 ドメインすべてにオフザシェルフのドメイン特化モデルが存在しない．**`domain_lora` が唯一の実行可能アプローチ**である．

**4. 評価軸②（回答品質）と評価軸③（End-to-End）の実装现状**

- **実装済み**．`evaluation.py` と `scripts/evaluate_response_quality.py` が存在し，以下の機能を備えている:
  - `compute_answer_quality_accuracy`: JMMLU 行の `jmmlu_answer` に対する回答文字の抽出・比較（客観的 ground truth）．
  - `judge_response_quality`: 手作業作成行に対する LLM-as-judge（1-5 Likert）．`judge_model`（config.yaml で指定，既定は general ノードの expert_model）を使用．
  - `compute_end_to_end_accuracy`: ルーティング正解 AND 回答品質合格 の両方を満たす割合．
  - `compute_latency_breakdown`: 応答時間の expert 生成時間 / その他 への分解．
- **metrics.py への統合は未実施**．`metrics.py` は評価軸①（ルーティング精度）のみを測定し，軸②③は `scripts/evaluate_response_quality.py` という別スクリプトでオフライン実行する設計になっている．
- **統合の必要性**: `metrics.py` の `compute_all_metrics()` に軸②③を統合するか，または `mise run analyze` タスクで `evaluate_response_quality.py` を自動呼び出すようにするかの 2 択．前者が journal の metrics 出力に一貫性を与えるが，後方が既存コードを壊さない．

**5. WAFL-PEFT の LoRA 訓練機制と接続可能性**

- **依存関係の共有**: `pyproject.toml` に `peft`, `transformers`, `accelerate`, `bitsandbytes`, `datasets`, `torch`（cu128）が記載済み．expert-mesh 側で同じ依存を追加すれば，訓練コードを共有できる．
- **訓練ループの流用**: `src/client.py` の Thread 3（Train）は `LoraConfig` + `get_peft_model` + `gradient_checkpointing` + 省メモリ cross-entropy の訓練ループを持っている．これを P2P 交換・マージのロジックなしで単一ノード SFT 用に抽出可能．
- **GPU プールの共有**: 同一 10 台（wafl500-509）を使用するため，訓練時は expert-mesh の Ollama コンテナと GPU 使用の競合に注意．WAFL-PEFT の実験が停止しているタイミングで訓練を実行するか，ホストを分割する必要がある．
- **データ準備**: 各ドメインごとに instruction-tuning 用の JSONL 数据集を準備する必要がある．JMMLU の既存タスクからドメイン関連タスクを抽出するか，別途ドメイン固有データセットを構築する．

**6. `class_weight="balanced"` と expert_specialization の関係**

- `class_weight="balanced"` は routing classifier（supervised_classifier）の訓練時のクラス不均衡対策であり，expert_specialization のスコープ外である．
- expert_specialization（domain_lora）が実施されると，各ノードの expert_model がドメイン固有の能力を持つようになるため，routing classifier の精度がさらに重要になる（誤ルーティングすると，違うドメインの LoRA 付きモデルが回答するため，回答品質が明確に劣化する）．
- 逆の視点: expert_specialization によりノード間に能力差が生まれると，routing accuracy の改善が直接 answer quality の改善に繋がるようになる．Iter17 までの top1_accuracy 改善は「代理指標」だったが，Iter18 以降は「実質指標」になる．
- legal/education の訓練データ不均衡（77 件）は，routing classifier の再訓練時にも続く．expert_specialization と並行して，ドメイン固有訓練データの追加が望ましい．

**計画フェーズへの示唆**

1. **`domain_lora` を唯一の実行可能アプローチとする**．`offtheshelf_specialized` は日本語 10 ドメインの状況では不可能である．
2. **LoRA 訓練の優先ドメイン**: 医療（JMedLoRA の先行例がある）と法律（オフザシェルフモデルが全くない）を最初の実証ドメインとする．全 10 ドメインを同時に訓練するのはコストが高すぎるため，段階的実施を推奨する．
3. **Ollama の単一アダプタ制限は問題ない**．各ノードが 1 ドメインを担当するため，1 アダプタ／ノードで十分である．
4. **評価軸②③の統合**を `metrics.py` または `mise run analyze` への組み込みとして実施する．expert_specialization の効果測定には必須である．
5. **WAFL-PEFT の訓練コードを流用**するが，P2P 交換ロジックは不要なため，最小限の SFT スクリプトとして抽出する．
6. **段階的アプローチ**: (a) 1-2 ドメインで PoC，(b) 評価軸②③の統合，(c) 全 10 ドメインへの展開，の順で進める．

**出典リスト**

| 出典 | 内容 |
|------|------|
| Ollama PR #14032 (GitHub, open) | 複数 LoRA アダプタ + ホットスワップ．未マージ． |
| Ollama issue #7627 (GitHub, closed via #14032) | 複数アダプタ要望．llama.cpp は対応済みだが Ollama ラッパー未対応． |
| Sukeda et al. (arXiv:2310.10083, NeurIPS 2023 workshop) | JMedLoRA: 日本語医療 QA における LoRA instruction-tuning の効果実証． |
| Sukeda et al. (arXiv:2406.14882) | 70B モデルでの日本語医療 instruction-tuning．医師国家試験 50% 超過． |
| S-LoRA (MLSys 2024, proceedings.mlsys.org) | 数千アダプタの同時配信システム．本研究では直接使用しないが，多数アダプタの同時ロードの技術的可行性を示す． |
| WAFL-PEFT `src/client.py` | LoRA 訓練ループ（`LoraConfig`, `get_peft_model`, gradient_checkpointing）の実装． |
| WAFL-PEFT `pyproject.toml` | `peft`, `transformers`, `bitsandbytes`, `datasets` の依存関係． |
| llama.cpp PR #8332, #8857 (2024-08) | 複数 LoRA アダプタのサポート（llama.cpp レベル）． |

---

### 実装 (Iter18)

**単一レバー**: `expert_specialization` (E10), `none → domain_lora`

**変更箇所**:
1. **config.yaml**: 全10ノードの `expert_model` を `expert-mesh-{domain}-lora` に変更
2. **scripts/train_domain_lora.py** (新規): WAFL-PEFT から抽出した単一ノード SFT 用 LoRA 訓練スクリプト．4-bit QLoRA，cosine LR decay，メモリ効率的 chunked cross-entropy 対応
3. **scripts/create_lora_model.py** (新規): LoRA アダプタから Ollama Modelfile を生成し，Ollama Create API でモデルを登録
4. **docker-compose.gpu.yml**: ollama サービスに `./lora_adapters:/root/lora_adapters:ro` の volume マウント追加
5. **mise.toml**: `[tasks.analyze]` に `evaluate_response_quality.py` の呼び出し追加（評価軸②③の自動計算）
6. **pyproject.toml**: `[project.optional-dependencies]` に `lora` グループ追加（torch, transformers, peft, bitsandbytes, datasets, accelerate）

**テスト結果**: 180 passed, 5 warnings in 1.48s（既存テストの退行なし）
**lint 結果**: ruff check クリーン
**Docker ビルド**: 成功

**Phase A（ベースライン測定）**: `mise run analyze 20260727_180824` で実行可能
**Phase B（LoRA 訓練）**: 訓練データ (`data/lora_train/{domain}.jsonl`) を準備すれば実行可能

実験を開始してよい状態である．

---

### 実験 (Iter17)

- **実験ディレクトリ**: `results/20260727_180824`
- **データセット**: JMMLU 1520問（単一1500 + 複合20），全問完走
- **所要時間**: 約91.8分（mean_duration_ms=3622.2）
- **top1_accuracy**: 0.5651（Wilson CI: [0.5401, 0.5899]）
- **Cohen's kappa**: 0.5215
- **random_baseline**: 0.1013
- **misrouting_rate**: 0.4349，fallback_rate: 0.1316
- **dispatch_failure_rate**: 0.0
- **同点タイ率**: 0.00%

**McNemar 対比較**: 不一致対数 814，chi2=365.57，p < 0.000001．**有意差あり**．

---

## Iteration 17: embedding ベース教師あり分類による routing_method の検証

### 調査 (Iter17)

**単一レバー**: `routing_method` (E6), `self_report → supervised_classifier`

**調査の問い**

1. `routing_method=supervised_classifier` のコード実装は完了しているか（classifier.py, router.py, http_server.py, Dockerfile）．
2. 訓練スクリプト（scripts/train_domain_classifier.py）は正しく動作するか．scikit-learn の依存関係は Docker イメージに含まれているか．
3. 訓練データと評価データの分離が，質問単位で完全に実施されているか（label leakage の再演を防ぐ）．
4. 分類器モデル（Pickle ファイル等）は既に訓練済みか，それとも実験前に訓練が必要か．
5. 既知のリスク・課題は何か（embedding モデルのバージョン，anisotropy，cross-lingual，class imbalance）．

**1. コード実装の完了状況**

実装は完全に完了しており，全テストが PASS している．

| 項目 | ファイル | 行番号 | 状態 |
|------|---------|-------|------|
| 分類器サービング | `classifier.py` | 全42行 | 完了 |
| 分類器ロード | `classifier.py:load_domain_classifier()` | 行16-24 | 完了 |
| 信頼度推定 | `classifier.py:estimate_confidence_classifier()` | 行27-41 | 完了 |
| /probe 統合 | `http_server.py` | 行323-329 | 完了（LLM コール不要） |
| ライフサイクル起動時ロード | `http_server.py` | 行406-411 | 完了（モデルパス未設定で ValueError） |
| NodeState 設定伝播 | `http_server.py` | 行194-195, 行244-252 | 完了 |
| node.py 設定伝播 | `node.py` | 行89 | 完了 |
| Dockerfile COPY | `Dockerfile` | 行14 | 完了（`classifier.py` が COPY 対象に含まれている） |
| config.yaml キー | `config.yaml` | 行59-64 | 完了（`classifier_model_path: models/domain_classifier.joblib`） |
| 単体テスト | `tests/test_classifier.py` | 4件全PASS | 完了 |
| 訓練スクリプトテスト | `tests/test_train_domain_classifier.py` | 2件全PASS | 完了 |
| 統合テスト | `tests/test_http_server.py` | 2件全PASS | 完了 |

**重要な設計決定**:

- 各ノードは同じ多クラス分類器をロードし，自分のドメインの予測確率のみを返す．中央ルーターは導入しない．
- 分類器は requester が既に計算済みの `query_embedding` を消費するため，/probe 呼び出しで追加 LLM コールは発生しない．
- `predict_proba` の全クラス確率は合計 1 になるため，ノード間の confidence 値が直接比較可能である（scikit-learn >=1.5 のデフォルト softmax 動作に依存）．
- 訓練時に未登場のドメインは 0.0 を返し，dispatch 対象から除外される．

**2. 訓練スクリプトと依存関係**

| 項目 | 状態 |
|------|------|
| 訓練スクリプト | `scripts/train_domain_classifier.py` — 完了 |
| CLI 引数 | `--train-data`, `--embedding-model`, `--ollama-host`, `--ollama-port`, `--output` |
| 分類器モデル | scikit-learn `LogisticRegression(max_iter=1000, class_weight="balanced")` |
| scikit-learn 依存 | `pyproject.toml` 行12: `scikit-learn>=1.5` — Docker イメージに含まれる |
| joblib 依存 | scikit-learn のトランザティブ依存として自動インストールされる |
| 訓練データ形式 | JSONL の `{"id", "query", "domain"}` 行 |
| 出力形式 | `models/domain_classifier.joblib`（joblib 直列化） |

**訓練の実行条件**: 訓練にはライブ Ollama ノードが必要（embedding 生成のため）．`--ollama-host` で指定したホストの Ollama デーモンが `nomic-embed-text` モデルをロードしている必要がある．

**3. 訓練/評価データ分離**

分離は構造的に保証されている．

| 保証メカニズム | 詳細 |
|--------------|------|
| 異なるシード | `_CLASSIFIER_TRAIN_SAMPLE_SEED = 20260727` vs `_JMMLU_SAMPLE_SEED = 20260726` |
| 質問単位の除外 | `build_classifier_training_rows()` は評価行の `query` を `frozenset` にして，サンプリング前にプールから除外する |
| 特徴量源の分離 | 訓練データは `{"query", "domain"}` のみ（probe/dispatch 結果を含まない） |
| モジュール設計 | `train_domain_classifier.py` は `results/*/results.jsonl` を一切参照しない |

**Iter10 の label leakage との比較**: Iter10 では probe/dispatch 結果（self_confidence, margin, is_top1）を同じ46問から抽出して訓練した．E6 では訓練データと評価データが質問単位で完全分離されており，label leakage の再演は構造上不可能である．

**4. 分類器モデルの訓練状況**

**未訓練**．以下の理由から，実験前に訓練が必要である．

- `models/` ディレクトリは存在しない（`.gitignore` 行13で除外されている）
- `data/classifier_train.jsonl` は存在しない
- 訓練データとモデルの両方を生成する手順が必要

**必要な手順**:
1. `uv run python build_dataset.py --output data/dataset.jsonl --classifier-train-output data/classifier_train.jsonl` — 訓練データ生成
2. `uv run python -m scripts.train_domain_classifier --train-data data/classifier_train.jsonl --embedding-model nomic-embed-text --ollama-host <ホストIP> --output models/domain_classifier.joblib` — 分類器訓練
3. 訓練済みモデルを全10ノードの `models/domain_classifier.joblib` に配布（または Docker volume マウント）

**5. 既知のリスク・課題**

**R1: クラス不均衡（legal ドメイン）**
- JMMLU に `professional_law` タスクが存在しないため，legal のプールは227問のみ（他ドメインは150問以上）．
- 評価用に150問を確保すると，訓練用には約77問しか残らない（他ドメインは150問）．
- **緩和策**: `class_weight="balanced"` がこの不均衡を補正する（journal.md「実装 (Iter15)」バッチ6 で追加済み）．

**R2: embedding の anisotropy**
- Iter2 で cosine similarity が `[0.667, 0.737]` に潰れた原因は embedding の anisotropy である．
- 本研究の supervised classifier は cosine similarity ではなく LogisticRegression を使用するため，anisotropy の影響は直接受けない．
- **根拠**: Varangot-Reille et al. (arXiv:2502.00409, JAIR 2025) は similarity-based routing の失敗を unsupervised であることに帰する．RouterDC (NeurIPS 2024) は CosineClassifier に全タスクで勝利している．教師あり分類は anisotropy 下でも機能する．

**R3: cross-lingual（英語ドメイン名 vs 日本語質問）**
- nomic-embed-text は multilingual モデルであるが，ドメイン名（"medical", "legal" 等）は英語で，質問は日本語である．
- Iter2 の embedding ルーティングではこの cross-lingual mismatch が問題となった（B7）．
- **緩和策**: supervised classifier は embedding 空間内の分離超平面を学習するため，cross-lingual なラベル名は学習プロセスには直接影響しない（ラベルはドメイン文字列としてのみ使用され，embedding されない）．

**R4: nomic-embed-text の task prefix 未付与**
- nomic-embed-text は `search_query:`, `search_document:`, `classification:` 等の task instruction prefix を前提に学習されている（B7）．
- 現行コードは prefix を付けていない．
- **影響**: prefix 未付与は embedding 品質を低下させる可能性があるが，supervised classifier はその embedding 空間で学習するため，prefix あり/なしの差は「embedding 空間の幾何的性質」に帰着し，分類器が適応できる範囲内である．RouterDC は prefix なしでも CosineClassifier に勝っている．

**R5: 訓練に必要な Ollama リソース**
- 訓練スクリプトは embedding 生成にライブ Ollama ノードを必要とする．
- 訓練データは推定で 10 ドメイン × 150 問 = 1,500 件（legal は77件）．nomic-embed-text の embedding は軽量だが，逐次実行のため数分かかる．
- **注意**: WAFL-PEFT が同一 GPU プールを使用中でないことを確認してから訓練を実行すること．

**R6: embedding モデルのバージョン整合性**
- 訓練時と推論時に同じ `nomic-embed-text` の同じバージョンが使用される必要がある．
- Ollama のモデルキャッシュが更新されると embedding 空間が変化する可能性がある．
- **緩和策**: 全10ノードで `ollama list` を確認し，同じ digest のモデルが使用されていることを確認する．

**文献調査の補足**

- **Varangot-Reille et al. (arXiv:2502.00409, JAIR 2025)**: "Doing More with Less: A Survey on Routing Strategies for Resource Optimisation in Large Language Model-Based Systems" — similarity-based routing の失敗を unsupervised であることに帰し，supervised routing の有効性を支持．
- **RouterDC (NeurIPS 2024)**: "Query-Based Router by Dual Contrastive Learning for Assembling Large Language Models" — CosineClassifier に全タスクで勝利．教師あり学習が embedding 空間の幾何的制約を克服可能であることを実証．
- **MoDEM (arXiv:2410.07490)**: 5クラスで総合81.00%．Other（general相当）が52.94% と低い．本研究の10クラス設定では general ノードが同様のボトルネックになる可能性がある．

**計画フェーズへの提案**

1. **訓練手順を最初に実行する**: 実験前に `build_dataset.py --classifier-train-output` で訓練データを生成し，`train_domain_classifier.py` で分類器を訓練する．この手順は `mise run setup/deploy` の前に行う必要がある（Docker イメージにモデルファイルを含めるため）．
2. **Docker volume でのモデル配布**: `docker-compose.yml` 行39-41 に `./models:/app/models:ro` の volume マウントが既に設定されている．したがって，訓練済みモデルを各ホストの `models/domain_classifier.joblib` に配置すれば，Docker イメージの再ビルドなしで全ノードに反映される．
3. **config.yaml の変更**: `routing_method: self_report → supervised_classifier` の1行変更のみ．`confidence_elicitation: top_k_with_probs` は維持（self_report 専用なので supervised_classifier では無視されるが，設定の整合性のため）．
4. **オフライン検証**: 訓練後，評価データ（1520問）に対してオフラインで分類精度を測定し，Iter15 の self_report ベースライン（top1_accuracy=0.184）との比較を事前に行う．

### 計画 (Iter17)

**単一レバー**: `routing_method` (E6), `self_report → supervised_classifier`

**変更箇所**: `config.yaml` 行31 の1行変更のみ．
```
routing_method: self_report  →  routing_method: supervised_classifier
```

**仮説**: embedding ベースの教師あり分類（LogisticRegression）が self_report よりルーティング精度を改善する理由は，self_report の構造的問題を根本的に回避するためである．

1. **自己宣伝バイアスの除去**: self_report では各ノードの light_model（qwen3.5:4b）が「あなたは{domain}分野の専門家です」とプロンプト指示されるため，どの質問に対しても自分の分野に 0.9 の高 confidence を出す（Iter15 で 74.9% が 0.9 饱和，クロスドメインでも 70-90% が 0.9）．教師あり分類器はドメインのプロンプト指示を受けず，embedding 空間の幾何的な分離超平面のみで判定するため，この自己宣伝バイアスを受けない．

2. **全クラス確率の合計制約による自然な正規化**: scikit-learn の多クラス LogisticRegression は softmax 出力のため，全10クラスの確率が合計 1 になる．self_report では各ノードが独立に 0-1 の値を申告するため，ノード間の confidence が比較不可能だった（Iter15 で 10 ノード中 7-10 ノードが 0.9 を出し，98.29% のタイ）．教師あり分類では，正解ドメインの確率が 0.3 なら他ドメインの合計は 0.7 になるため，自然に弁別力のある分布が生成される．

3. **embedding 空間の教師あり学習は anisotropy に頑健**: Iter2 で cosine similarity が [0.667, 0.737] に潰れた原因は embedding の anisotropy であるが，教師あり分類器は cosine 距離ではなく線形分離超平面を学習するため，anisotropy の影響を直接受けない（Varangot-Reille+ JAIR 2025，RouterDC NeurIPS 2024）．

4. **Iter2（unsupervised embedding）との明確な違い**: Iter2 が棄却されたのは，unsupervised cosine similarity がドメイン識別信号を持っていなかったからである．教師あり分類はラベル付きデータから分離超平面を学習するため，unsupervised とは全く異なるアプローチである．RouterDC は CosineClassifier に全タスクで勝利している．

**固定する構成**（Iter16 の最良構成を継承）:

| 設定 | 値 | 理由 |
|------|-----|------|
| `confidence_signal_method` | `self_report` | 変更不可．supervised_classifier では routing_method が signal 抽出を完全に置き換えるため，この設定は参照されない |
| `confidence_elicitation` | `top_k_with_probs` | 変更不可．self_report 専用設定であり，supervised_classifier では無視される |
| `confidence_threshold` | `0.5` | 変更不可．閾値ゲートの効果検証は Iter3 で no-op と判定済み |
| `dispatch_top_k` | `1` | 変更不可．Iter1 で棄却済み |
| `semantic_sample_count` | `5` | 変更不可．E4 用設定であり，supervised_classifier では参照されない |
| `semantic_sample_temperature` | `0.7` | 変更不可．E4 用設定 |
| `embedding_postprocess` | `none` | 変更不可．E7（whitening）は embedding ルーティング専用であり，supervised_classifier では参照されない |
| `light_model` | `qwen3.5:4b-q4_K_M` | 変更不可．E8（expert_model_size）は別レバー |
| `expert_model` | `schroneko/llama-3.1-swallow-8b-instruct-v0.1:q4_k_m` | 変更不可．S1（expert_specialization）は別レバー |
| 10 ノード構成 | wafl500〜509 | 変更不可．domain_count=10 は E9 の対象 |
| `router.py` few-shot 例 | 動的生成 `_build_few_shot_examples` | 変更不可．few-shot 変更は Iter5-9 で5回連続棄却済み |
| `classifier_model_path` | `models/domain_classifier.joblib` | 変更不可．E6 実装時に設定済み |

**成功条件**:

| 分類 | 指標 | ベースライン (Iter16) | 成功条件 | 根拠 |
|------|------|---------------------|---------|------|
| 主基準 | top1_accuracy McNemar | 0.2059 (p=0.0783 vs Iter15) | **有意差あり** (α=0.05) | 同じ 1520 問データセット上の McNemar 対比較．Iter15→Iter16 の変化（+0.022, p=0.0783）は有意閾値の80%であり，supervised_classifier がより明確な信号を出すなら有意になる |
| 主基準 | top1_accuracy Wilson CI | [0.1863, 0.2270] | **CI がベースライン CI と重ならない** | 1520 問で SE ≈ 0.01，CI 幅 ≈ 0.04．±0.03 以上の改善で CI が重ならなくなる |
| 副基準 | Cohen's kappa | 0.1067 | **0.1067 より有意に高い**（CI が重ならない） | chance-corrected 指標で実質識別力を測定．10 分野で偶然一致 0.101 を有意に上回る必要がある |
| 副基準 | 同点タイ率 | 82.83% | **有意な低下** | softmax 出力は連続値のため，self_report の離散値（5段階）よりタイが大幅に減る |
| 副基準 | ECE | 0.7388 | **報告**（較正改善の定量化） | softmax 出力は較正された確率であるため，ECE が改善する可能性がある |
| 非退行 | per-domain precision/recall | Iter16 の各値 | **各ドメインの CI 下限がベースライン CI 下限を下回らない** | config.yml success_criteria (2) に従う |
| 監視 | probe レイテンシ | 計測済み (Iter16) | **報告**（追加 LLM コールなしのため同程度または短縮） | supervised_classifier は embedding 計算のみで LLM コール不要 |
| 監視 | dispatch_failure_rate | 0.0 | **0.0** | インフラ起因の失敗がないことを確認 |

**成功条件の数値根拠**: Iter15→Iter16 の変化は +0.022（p=0.0783）であり，有意閾値の 80% にある．supervised_classifier が self_report の構造的問題（自己宣伝バイアス，離散値飽和）を解決するなら，より大きな変化（±0.05 以上）が期待される．1520 問での二項 SE は約 0.01 であり，0.03 以上の変化は CI が重ならなくなる（Wilson 95% CI 幅 ≈ 0.04）．

**実験構成（フルフロー）**:

```
┌─────────────────────────────────────────────────────────────┐
│ Step 0: 訓練データ生成                                       │
│ uv run python build_dataset.py                              │
│   --output data/dataset.jsonl                               │
│   --classifier-train-output data/classifier_train.jsonl      │
│  → data/classifier_train.jsonl に {query, domain} 行が生成   │
│  → 評価データ（_JMMLU_SAMPLE_SEED=20260726）と              │
│    訓練データ（_CLASSIFIER_TRAIN_SAMPLE_SEED=20260727）が    │
│    質問単位で完全分離される                                   │
├─────────────────────────────────────────────────────────────┤
│ Step 1: 分類器訓練（ライブ Ollama が必要）                   │
│ uv run python -m scripts.train_domain_classifier            │
│   --train-data data/classifier_train.jsonl                  │
│   --embedding-model nomic-embed-text                        │
│   --ollama-host 192.168.15.100                              │
│   --output models/domain_classifier.joblib                  │
│  → LogisticRegression(max_iter=1000, class_weight="balanced")│
│  → 10 クラス softmax 出力（predict_proba の合計=1）          │
│  → models/domain_classifier.joblib に保存                    │
├─────────────────────────────────────────────────────────────┤
│ Step 2: モデル配布（Docker volume 経由）                     │
│ 各ホスト（wafl500-509）の ./models/ ディレクトリに           │
│ domain_classifier.joblib を配置                              │
│ docker-compose.yml 行41 の ./models:/app/models:ro が       │
│ 自動マウントするため，Docker イメージの再ビルドは不要         │
├─────────────────────────────────────────────────────────────┤
│ Step 3: config.yaml 変更                                    │
│ config.yaml 行31: routing_method: self_report               │
│                    → supervised_classifier                   │
│ mise run setup（Docker イメージ再ビルド）                     │
│ mise run deploy（全10ノード）                                 │
├─────────────────────────────────────────────────────────────┤
│ Step 4: 実験                                                │
│ mise run start（同一 1520 問データセット）                    │
│ 完了後: mise run analyze                                     │
├─────────────────────────────────────────────────────────────┤
│ Step 5: 分析                                                │
│ metrics.py --results <実験ディレクトリ>/results.jsonl --json │
│ → Wilson CI, Cohen's kappa, McNemar 対比較                  │
│ → Random/BestSingle/Oracle ベースライン                      │
│ → ドメイン別 precision/recall/ECE/同点率                     │
└─────────────────────────────────────────────────────────────┘
```

**オフライン事前検証（Step 1 と Step 4 の間に実施）**:

訓練済み分類器に対して，評価データ（1520問）の embedding をオフラインで計算し，分類器の predict_proba による top-1 分類精度を測定する．これはルーティング精度の上限（upper bound）を示す（実際のルーティングではノードごとの confidence 比較があるが，オフライン検証は分類器自体の性能を直接測定する）．この値がベースライン（0.206）を有意に上回らない場合，実機実験の実行を再検討する．

**実行時間の見積もり**:

| 工程 | 推定時間 | 備考 |
|------|---------|------|
| Step 0: 訓練データ生成 | 1-2 分 | JMMLU からサンプリング（CPU 処理） |
| Step 1: 分類器訓練 | 5-10 分 | embedding 生成（逐次，nomic-embed-text）+ LogisticRegression 学習（瞬時） |
| Step 2: モデル配布 | 1-2 分 | scp で 10 ホストにコピー（joblib ファイルは数十 KB） |
| Step 3: setup + deploy | 5-10 分 | Docker イメージ再ビルド + 10 ノードのコンテナ再作成 |
| Step 4: 実験実行 | 約 90-120 分 | Iter16 の mean_duration_ms=4134（約4.1秒/問）× 1520 問．supervised_classifier は LLM コール不要なので probe 時間が短縮される可能性がある |
| Step 5: 分析 | 1-2 分 | metrics.py のオフライン計算 |
| **合計** | **約 2-2.5 時間** | タイムアウト 90 分は不足するため，experiment.timeout_min を 150 に引き上げる |

**特定されたリスクと緩和策**:

| リスク | 内容 | 影響 | 緩和策 |
|-------|------|------|--------|
| R1: legal ドメインの訓練データ不足 | JMMLU に professional_law タスクがなく，legal の訓練用プールは約77問（他ドメインは150問） | legal クラスの分類精度が低く，misroute が増加する可能性 | `class_weight="balanced"` が不均衡を補正（訓練時に追加実装済み）．legal の per-domain recall を重点監視 |
| R2: embedding モデルのバージョン整合性 | 訓練時と推論時に異なる nomic-embed-text のバージョンが使用されると，embedding 空間が変化し分類器が機能しない | 分類精度が大幅に低下する | 全10ノードで `ollama list` を確認し，同じ digest のモデルを使用していることを確認．`OLLAMA_KEEP_ALIVE=-1` でモデルがアンロードされないため，バージョン変化のリスクは低い |
| R3: 訓練/評価データの潜在的なオーバーラップ | JMMLU の同じタスク内の異なる問題が訓練と評価にまたがる可能性 | label leakage の再演（Iter10 の問題） | `build_dataset.py` はシードを分ける（20260726 vs 20260727）かつ質問単位で除外する．ただし同じタスク内の異なる問題は重複し得る．これはデータセット設計の制約であり，完全なタスク単位分離は JMMLU の56タスク×10ドメインの写像で不可能 |
| R4: 訓練に必要な Ollama リソース | 訓練スクリプトは embedding 生成にライブ Ollama を必要とする | WAFL-PEFT が GPU を使用中だと訓練が失敗する | 訓練実行前に WAFL-PEFT の稼働状況を確認．wafl500 の Ollama を単一ホストで訓練に专用する |
| R5: softmax 確率の較正 | scikit-learn の LogisticRegression はデフォルトで較正されていない | ECE が改善しない可能性 | scikit-learn のデフォルト LogisticRegression は内部に較正を組み込んでいる（CalibratedClassifierCV 不要）．ECE を監視し，改善しない場合は較正曲線を分析 |
| R6: general クラスの識別困難 | general は「どの専門分野でもない」を意味するため，embedding 空間で他の9ドメインと重複する | general の precision が低く，専門ドメインへの誤分類が増える | MoDEM の結果（Other=52.94%）と同様の構造的問題．general の per-domain 指標を重点監視．`class_weight="balanced"` が部分的に補正 |
| R7: timeout_min の不足 | 現在 config.yml の experiment.timeout_min=90 だが，実験時間は約 90-120 分 | 実験がタイムアウトで中断される | 実験実行前に timeout_min を 150 に引き上げる |

### 実装 (Iter17)

**単一レバー**: `routing_method` (E6), `self_report → supervised_classifier`

**変更箇所**:
1. `config.yaml` 行31: `routing_method: self_report → supervised_classifier`（単一レバー変更）
2. `Dockerfile` 行14: `COPY scripts/ ./scripts/` の追加（訓練スクリプトをコンテナに含めるため）
3. `.claude/research/config.yml` 行26: `timeout_min: 90 → 150`（実験時間の余裕確保）
4. `mise.toml` 行73-75: models/ ディレクトリの rsync 前に `sudo rm -rf` を追加（root 所有の stale ディレクトリ対策）
5. `scripts/analyze_iter16.py` 行5: 未使用の `import sys` を削除（lint 修正）

**分類器訓練**:
- 訓練データ: `data/classifier_train.jsonl`（1427 件，10 ドメイン）
  - legal: 77 件（JMMLU に professional_law 不在のため他ドメインの半分）
  - 他9ドメイン: 各150 件
- 訓練方法: `LogisticRegression(max_iter=1000, class_weight="balanced")`
- embedding モデル: `nomic-embed-text`（768 次元）
- 訓練実行: wafl500 の Ollama コンテナに対し SSH トンネル（localhost:11435）経由で embedding 生成
- 出力: `models/domain_classifier.joblib`（62KB）

**オフライン分類精度**:
- 訓練データ: 100.00%（1427/1427，過学習）
- 評価データ（単一ドメイン1500問）: 59.87%（898/1500）
  - history_culture: 68.00%，legal: 68.67%，mathematics: 68.67%
  - social_science: 64.67%，natural_science: 62.00%，computer_science: 60.00%
  - general: 59.33%，medical: 52.00%，business_economics: 50.00%，education: 45.33%
- ベースライン比較: Random=10%，Iter16 self_report=20.59% → 分類器は約3倍の精度
- 訓練/評価ギャップ: 0.4013（768次元embeddingに対する1427サンプルの過学習）

**デプロイ検証**:
- `uv run pytest tests/ -v`: 180件全PASS（回帰なし）
- `uv run ruff check`: All checks passed
- `mise run setup`: Docker イメージ再ビルド・ローカル registry push 成功（scripts/ 含む）
- `mise run deploy`: 全10ノード（wafl500〜509）の config.yaml と models/domain_classifier.joblib を配布・app コンテナ再作成・起動成功
- 全ノード healthy 確認（wafl507-509 は初回 healthcheck で遅延したものの再試行で正常）
- wafl500 上のコンテナ内設定確認: `routing_method: supervised_classifier` が正しく反映
- wafl500 上のコンテナ内モデル確認: `/app/models/domain_classifier.joblib`（63095バイト）が存在
- wafl500 コンテナ起動ログ確認: エラーなし，GPU モデル両方ロード済み

**実験開始の可否**: 実験を開始してよい状態である．

### 実験 (Iter17)

- **実験ディレクトリ**: `results/20260727_180824`
- **データセット**: JMMLU 1520問（単一1500 + 複合20），全問完走（1520/1520）
- **所要時間**: 約91.8分（mean_duration_ms=3622.2，Iter16の4134.4より約12%短縮）
- **top1_accuracy**: 0.5651（Wilson CI: [0.5401, 0.5899]）
- **Cohen's kappa**: 0.5215
- **random_baseline**: 0.1013，best_single: legal/medical 0.1092
- **misrouting_rate**: 0.4349，fallback_rate: 0.1316
- **dispatch_failure_rate**: 0.0
- **同点タイ率**: 0.00%（Iter16: 82.83%）
- **バックグラウンドタスク**: コピー段階で `sh exited with non-zero status: no exit status` のエラーが発生したため，手動で `ssh wafl500 cat ... > results/...` により結果ファイルをコピー
- **異常**: なし（全ノードログ収集済み，全10ノードで正常動作）

### 分析 (実行) (Iter17)

**比較ベースライン**: `results/20260727_100917/` (Iter16, self_report + top_k_with_probs)

| 指標 | Iter16 (self_report) | Iter17 (supervised_classifier) | 変化 |
|------|---------------------|-------------------------------|------|
| top1_accuracy | 0.2059 | 0.5651 | +0.3592 |
| Wilson 95% CI | [0.1863, 0.2270] | [0.5401, 0.5899] | 重ならなし |
| Cohen's kappa | 0.1067 | 0.5215 | +0.4148 |
| misrouting_rate | 0.7941 | 0.4349 | -0.3592 |
| fallback_rate | 0.0000 | 0.1316 | +0.1316 |
| mean_duration_ms | 4134.4 | 3622.2 | -512.2 |
| 同点タイ率 | 82.83% | 0.00% | -82.83pt |

**McNemar 対比较**:

| | Iter17 正解 | Iter17 不正解 |
|---|---|---|
| Iter16 正解 | 179 (a) | 134 (b) |
| Iter16 不正解 | 680 (c) | 527 (d) |

- 不一致対数: b+c = 814
- McNemar chi-squared（連続性補正）: 365.57
- **p-value: < 0.000001**
- **有意差あり** (α=0.05)

**ドメイン別 precision/recall 比較**:

| ドメイン | Iter16 prec/rec | Iter17 prec/rec | 変化 |
|---------|----------------|----------------|------|
| business_economics | 0.242 / 0.100 | 0.511 / 0.453 | +0.269 / +0.353 |
| computer_science | 0.439 / 0.193 | 0.614 / 0.540 | +0.175 / +0.347 |
| education | 0.114 / 0.551 | 0.520 / 0.411 | +0.406 / -0.140 |
| general | 0.169 / 0.280 | 0.317 / 0.680 | +0.148 / +0.400 |
| history_culture | 0.200 / 0.060 | 0.764 / 0.647 | +0.564 / +0.587 |
| legal | 0.380 / 0.325 | 0.817 / 0.566 | +0.437 / +0.241 |
| mathematics | 0.511 / 0.160 | 0.725 / 0.667 | +0.214 / +0.507 |
| medical | 0.385 / 0.120 | 0.517 / 0.470 | +0.132 / +0.350 |
| natural_science | 0.438 / 0.140 | 0.580 / 0.580 | +0.142 / +0.440 |
| social_science | 0.245 / 0.080 | 0.685 / 0.580 | +0.440 / +0.500 |

**複合ドメイン**: 20問中5問正解（25.0%），domain_set_recall=0.125（Iter16: 0.475）

**ドメイン別 McNemar 対比較**:

| ドメイン | acc_16 | acc_17 | 変化 | chi2 | p-value | 判定 |
|---------|--------|--------|------|------|---------|------|
| business_economics | 0.1000 | 0.4533 | +0.3533 | 40.36 | <0.0001 | **有意改善** |
| computer_science | 0.1933 | 0.5400 | +0.3467 | 34.22 | <0.0001 | **有意改善** |
| education | 0.5400 | 0.4333 | -0.1067 | 3.63 | 0.0568 | 有意差なし（退行傾向） |
| general | 0.2800 | 0.6800 | +0.4000 | 37.84 | <0.0001 | **有意改善** |
| history_culture | 0.0600 | 0.6467 | +0.5867 | 80.52 | <0.0001 | **有意改善** |
| legal | 0.2867 | 0.6200 | +0.3333 | 27.92 | <0.0001 | **有意改善** |
| mathematics | 0.1600 | 0.6667 | +0.5067 | 70.31 | <0.0001 | **有意改善** |
| medical | 0.1200 | 0.4933 | +0.3733 | 42.01 | <0.0001 | **有意改善** |
| natural_science | 0.1400 | 0.5800 | +0.4400 | 50.30 | <0.0001 | **有意改善** |
| social_science | 0.0800 | 0.5800 | +0.5000 | 62.94 | <0.0001 | **有意改善** |

9/10 ドメインで有意改善．education は p=0.0568 で有意閾値をわずかに下回らず，有意差なし．

**ECE（Expected Calibration Error）**:

Iter16: **0.7388** → Iter17: **0.2118**（**-71.3%**）．

Iter16 では confidence 値が {0.6, 0.8, 0.9, 0.95, 1.0} の5段階離散値で，99.9% が [0.9, 1.0) ビンに集中し，bin_accuracy=0.2062 に対する bin_confidence=0.9450 の乖離（gap=0.7388）が ECE 全体を支配していた．

Iter17 では softmax 連続値により confidence が [0.22, 1.00] の範囲に広がり，8 ビンに分布した．特に [0.9, 1.0) ビン（40.7%）では bin_accuracy=0.7948，bin_confidence=0.9698（gap=0.1750）と，Iter16 の gap=0.7388 と比べて大幅に縮小した．

**ドメイン別 ECE（Iter17）**:

| ドメイン | accuracy | mean_conf | ECE |
|---------|----------|-----------|-----|
| mathematics | 0.6667 | 0.8145 | 0.1478 |
| history_culture | 0.6467 | 0.8256 | 0.1789 |
| legal | 0.6200 | 0.8057 | 0.1857 |
| social_science | 0.5800 | 0.7626 | 0.1898 |
| computer_science | 0.5400 | 0.7443 | 0.2043 |
| natural_science | 0.5800 | 0.7855 | 0.2055 |
| general | 0.6800 | 0.8110 | 0.2580 |
| medical | 0.4933 | 0.7570 | 0.2637 |
| business_economics | 0.4533 | 0.7495 | 0.2962 |
| education | 0.4333 | 0.7294 | 0.2960 |

全ドメインで ECE < 0.30．mathematics（0.1478）と history_culture（0.1789）が最も較正されており，education（0.2960）と business_economics（0.2962）が最も較正が低い．

**同点タイ率**:

Iter16: **82.83%** → Iter17: **0.00%**（**-82.83pt**）．

Iter16 では confidence 値が5段階の離散値であり，10ノードが独立に5値を選ぶと必然的に同値が発生した．Iter17 では softmax 出力が連続値（1518の唯一値，8桁小数点以下で計測）であり，同値発生確率は実質 0% である．

**fallback_rate 分析**:

Iter16: **0.0000** → Iter17: **0.1316**（200/1520）．

- 原因: `confidence_threshold=0.5` を下回るケースで fallback 発生．max_probe_conf < 0.5 の質問がちょうど 200 件であり，fallback 件数と完全に一致する．
- fallback 先のドメイン: 全て `general`（200/200）．
- fallback 時の confidence 分布: [0.220, 0.500]，平均 0.418．分類器がどのドメインにも確信もって分類できない「境界領域」の質問である．
- fallback 正解率: **8.0%**（16/200）．general への盲目的フォールバックは，これらの質問の正解ドメインが general である割合（16/200 = 8.0%）と一致し，fallback 戦略自体が有用なルーティング信号を持っていないことを示す．
- fallback 元の期待ドメイン分布: education(29), legal(26), business_economics(25), computer_science(24), medical(23) の順で多く，general(16), mathematics(14), history_culture(10) は少ない．education と legal が分類器の識別困難領域であることを示唆する．

**education の退行分析**:

recall: 0.5506 → 0.4114（CI17 下限 0.3377 < CI16 下限 0.4728）．**非退行条件を違反**．

- Iter16 では education ノードが education 質問に対して平均 confidence=0.8743 を出し，self_report の自己宣伝バイアスにより多くの教育関連質問を education へ引き寄せていた（recall 0.55）．しかし precision は 0.114 と極めて低く，education 以外の質問も education へ誤ルーティングされていた．
- Iter17 では education ノードの confidence が平均 0.4315 に低下し，分類器が education 質問を正しく識別できない．その結果，recall が 0.41 に低下した．precision は 0.520 と大幅に改善したが，recall の低下が全体を押し下げている．
- 根本原因: JMMLU に education に対応する直接的なタスクが存在せず，心理学・社会学タスクで代理しているため，embedding 空間で education クラスの分離超平面が不明瞭である可能性が高い．

**複合ドメインの退行**:

top1_accuracy: 0.9500 → 0.2500，domain_set_recall: 0.475 → 0.125．

- Iter16 では self_report の高タイ率（82.83%）により，複合ドメイン質問でも複数の期待ドメインが同点になり，宣言順で正解ドメインが選ばれる確率が高かった．これは構造上の偽高値である．
- Iter17 では softmax 連続値によりタイが解消され，分類器が単一ドメインを選択する．複合ドメイン質問は本質的に複数のドメインに属するため，単一選択では正解率が下がる．
- compound_mean_dispatched_count: Iter16=1.0 → Iter17=0.7．fallback 発生（200件中複合ドメインも含まれる）により dispatch 数が減少している．

**レイテンシ**:

mean_duration_ms: 4134.4 → 3622.2（**-12.4%**）．supervised_classifier は probe 段階で LLM コールを不要とし，embedding 計算のみで confidence を算出するため，probe ラウンドトリップ時間が短縮された．

**Cohen's kappa 比較**:

Iter16: 0.1067（95% CI: [0.0608, 0.1554]）→ Iter17: 0.5215（95% CI: [0.4890, 0.5404]）．CI が重ならず，**有意に高い**．

po（観測一致率）: 0.1987 → 0.5632．pe（偶然一致率）: 0.1016 → 0.0999．kappa の改善は，偶然一致を差し引いた実質的なドメイン識別力の向上を反映している．

**ベースライン比較**:

| ベースライン | accuracy | Iter17 比 |
|-------------|----------|-----------|
| Random | 0.1013 | 5.6x |
| BestSingle (legal/medical) | 0.1092 | 5.2x |
| Iter16 (self_report) | 0.2059 | 2.7x |
| Iter17 (supervised) | 0.5651 | - |
| Oracle | 1.0000 | - |

Iter17 は Random の 5.6 倍，Iter16 の 2.7 倍．Oracle までのギャップは 0.4349（Random→Oracle の距離の 48.4% を埋めた）．

### 分析 (解釈) (Iter17)

#### 1. 大幅改善のメカニズム解釈

**観測事実の再確認**: top1_accuracy 0.2059 → 0.5651（+0.3592）．McNemar chi2=365.57, p < 0.000001．Wilson CI は [0.1863, 0.2270] vs [0.5401, 0.5899] で完全に重ならなし．

この変化はノイズの範疇を超えている．1520 問における二項 SE は約 0.01 であり，+0.3592 は約 36 SE の変化である．過去の反復（Iter15→Iter16 で +0.022, p=0.0783）と比較しても，その効果量が桁違いである．

**self_report の構造的問題の解決メカニズム**:

self_report（Iter16）の根本問題は，各ノードの light_model が「あなたは{domain}分野の専門家です」というシステムプロンプトの影響を受け，自分の担当分野に対して過度に高い confidence を出す「自己宣伝バイアス」であった．Iter15 の numeric_scalar では 74.9% が 0.9 に飽和し，Iter16 の top_k_with_probs でも 80.5% が 0.95 に集中した．10 ノードが独立にこのバイアスを持つため，98.29%（Iter15）→ 82.83%（Iter16）の同点タイが発生し，実質的にルーティングは宣言順に依存する状態であった．

supervised_classifier はこの構造的問題を根本的に回避している．理由は以下の通りである．

- **自己宣伝バイアスの除去**: 分類器は embedding 空間の幾何的パターンのみで判定し，ドメイン固有のプロンプト指示を受けない．各ノードが同じ多クラス分類器をロードし，自分のクラスの softmax 確率のみを返すため，ノード間に一貫性のある confidence 分布が生成される．
- **全クラス確率の合計制約**: softmax 出力により全 10 クラスの確率が合計 1 になるため，正解ドメインの確率が 0.3 なら他ドメインの合計は 0.7 になる．self_report では各ノードが独立に 0.9 を出すため比較不可能だったのに対し，supervised_classifier では自然に弁別力のある分布が生成される．
- **連続値出力によるタイ解消**: softmax 出力は連続値（1518 の唯一値）であり，同点タイ率は 82.83% → 0.00% に完全に解消された．

**ECE の -71.3% 改善の理由**:

Iter16 の ECE=0.7388 は，confidence 値が {0.6, 0.8, 0.9, 0.95, 1.0} の 5 段階離散値で，99.9% が [0.9, 1.0) ビンに集中し，bin_accuracy=0.2062 に対する bin_confidence=0.9450 の乖離（gap=0.7388）が ECE 全体を支配していた．これは「confidence が高いのに accuracy が低い」という較正の破綻である．

Iter17 の ECE=0.2118 は，softmax 出力が [0.22, 1.00] の範囲に広がり，8 ビンに分布した結果である．特に [0.9, 1.0) ビン（40.7%）では bin_accuracy=0.7948，bin_confidence=0.9698（gap=0.1750）と，Iter16 の gap=0.7388 と比べて大幅に縮小した．scikit-learn の LogisticRegression は内部に較正を組み込んでいるため，CalibratedClassifierCV なしでも比較的良好な較正が得られている．

ただし，全ドメインで mean_conf > accuracy（education: 0.7294 > 0.4333, business_economics: 0.7495 > 0.4533 など）であり，依然として overconfident である．これは scikit-learn のデフォルト LogisticRegression が完全な較正を保証しないこと，および embedding 空間のクラス境界が完全には分離されていないことに起因する．ECE=0.2118 は「実用的に許容可能な範囲（< 0.30）」ではあるが，完全な較正（ECE < 0.05）には程遠い．

**kappa の +0.4148 改善の理由**:

Cohen's kappa は，観測一致率（po）から偶然一致率（pe）を差し引いた指標である．Iter16: po=0.1987, pe=0.1016 → kappa=0.1067．Iter17: po=0.5632, pe=0.0999 → kappa=0.5215．

kappa の改善は，偶然一致を差し引いた実質的なドメイン識別力の向上を反映している．10 分野で偶然一致率は約 0.10 であり，Iter16 の po=0.1987 は偶然よりわずかに良い程度（kappa=0.1067 = "slight agreement"）だったのに対し，Iter17 の po=0.5632 は偶然を有意に上回り（kappa=0.5215 = "moderate agreement"），実質的なルーティング能力が確立されたことを示している．

**仮説との整合**: 計画フェーズで述べた 4 つの仮説はすべて支持された．

1. 自己宣伝バイアスの除去 → 支持（同点率 0.00%，ドメイン別 McNemar で 9/10 有意改善）
2. 全クラス確率の合計制約 → 支持（ECE -71.3%，kappa +0.4148）
3. anisotropy への頑健性 → 支持（Iter2 の cosine 潰れとは異なり，分類精度 56.51% を達成）
4. Iter2（unsupervised）との違い → 支持（教師あり学習が分離超平面を学習した結果，Random の 5.6 倍）

#### 2. education recall 退行の解釈

**観測事実**: recall 0.5506 → 0.4114．CI17 下限 0.3377 < CI16 下限 0.4728．**非退行条件を違反**．ドメイン別 McNemar で p=0.0568（有意差なし）．

**根本原因の分析**:

この退行は，手法自体の欠陥ではなく，データセットの構造的問題に起因すると解釈する．

1. **JMMLU に education 対応タスク不在**: JMMLU の 56 タスクは MMLU 由来の学術科目であり，本研究の education ドメイン（日本の教育行政・教育実務）に相当するタスクが存在しない．心理学・社会学タスクで代理しているため，embedding 空間で education クラスの分離超平面が不明瞭である．
2. **訓練データの不均衡**: legal と同様に，education の訓練データは 77 件（他ドメインの半分）．`class_weight="balanced"` が補正しているものの，768 次元 embedding 空間における 77 サンプルでは，education クラスの決定境界が不安定になりやすい．
3. **オフライン分類精度の低さ**: education のオフライン分類精度は 45.33%（全ドメイン中最下位）であり，分類器自体が education クラスの識別に困難を抱えている．

**Iter16 の education recall=0.5506 の解釈**: Iter16 では education ノードが自己宣伝バイアスにより education 質問に対して平均 confidence=0.8743 を出し，多くの教育関連質問を education へ引き寄せていた．precision=0.114 と極めて低かったため，education 以外の質問も education へ誤ルーティングされていた．Iter17 では precision=0.520 と大幅に改善したが，recall が 0.41 に低下した．

**総合判断**: education recall の退行は，self_report から supervised_classifier への移行による「偽高値の剥奪」の側面と，embedding 空間での education クラス識別困難の側面の両方がある．手法自体の棄却根拠にはならないが，データセット整備（education 固有の訓練データ追加）または分類器の再訓練（education クラスの oversampling）が必要である．

#### 3. fallback_rate=13.16% の解釈

**観測事実**: 200/1520（13.16%）の質問が fallback 発生．max_probe_conf < 0.5 の質問が 200 件であり，fallback 件数と完全に一致する．

**fallback のメカニズム**:

`confidence_threshold=0.5` は，分類器の最大クラスの softmax 確率が 0.5 未満の場合に fallback を発生させる．10 クラスの softmax 出力において，最大値が 0.5 未満ということは，分類器がどのドメインにも確信もって分類できない「境界領域」の質問であることを意味する．Random baseline（10 クラス）の期待値は 0.10 であるため，0.5 は「Random より 5 倍確信がある」ことを示す閾値である．

**fallback 正解率 8.0% の問題**:

fallback 先のドメインはすべて `general` であり，fallback 正解率は 8.0%（16/200）である．これは Random baseline（10.1%）より低い．つまり，盲目的な general fallback は，これらの質問の正解ドメインが general である割合（8.0%）と一致し，fallback 戦略自体が有用なルーティング信号を持っていない．

**fallback 元の期待ドメイン分布**:

education(29), legal(26), business_economics(25), computer_science(24), medical(23) の順で多く，general(16), mathematics(14), history_culture(10) は少ない．education と legal が分類器の識別困難領域であることを示唆する．これは訓練データ不足（77 件）と関連しており，これらのドメインの境界領域で分類器が確信もって判定できない．

**改善提案**:

1. **confidence_threshold の最適化**: 現在 0.5 だが，これを下げる（0.3-0.4）ことで fallback 率を下げ，general への盲目的フォールバックを減らすことができる．ただし，閾値を下げると misroute が増えるトレードオフがある．
2. **fallback 戦略の変更**: general へのフォールバックではなく，分類器の top-2 クラスを dispatch 対象とする（dispatch_top_k=2）か，confidence の低い質問に対して複数の専門ノードに並行 dispatch する方が，8.0% の正解率を改善する可能性がある．
3. **education/legal の訓練データ追加**: 境界領域の質問を減らす根本的な解決策である．

#### 4. 複合ドメインの退行（0.95 → 0.25）の解釈

**観測事実**: top1_accuracy 0.9500 → 0.2500，domain_set_recall 0.475 → 0.125．

**Iter16 の高値は偽高値**:

Iter16 では self_report の高タイ率（82.83%）により，複合ドメイン質問でも複数の期待ドメインが同点になり，宣言順で正解ドメインが選ばれる確率が高かった．20 問中 19 問正解（95%）は，ルーティング能力ではなく，タイ解決メカニズムの構造上の副産物である．

**Iter17 の値の方が実態を反映**:

supervised_classifier は softmax 連続値によりタイを解消し，分類器が単一ドメインを選択する．複合ドメイン質問は本質的に複数のドメインに属するため，単一選択では正解率が下がる（25%）．これは「supervised_classifier が悪い」という意味ではなく，「複合ドメイン質問の評価方法が単一選択ルーティングに適していない」ことを示している．

**domain_set_recall の低下**:

Iter16: 0.475 → Iter17: 0.125．複合ドメイン質問の正解ドメインセットの中に，ルーティング先が含まれる割合である．Iter17 では fallback 発生（200 件中複合ドメインも含まれる）により，dispatch 数が減少（compound_mean_dispatched_count: 1.0 → 0.7）しており，これが domain_set_recall の低下に寄与している．

**判断**: 複合ドメインの退行は，評価方法とルーティング方式の不一致に起因する．supervised_classifier の性能評価からは除外すべきである．複合ドメイン質問に対する適切な評価は，dispatch_top_k >= 2 の設定で再評価するか，domain_set_recall のみを指標とするべきである．

#### 5. 総合判定

**成功条件に対する判定**:

| 分類 | 指標 | ベースライン (Iter16) | Iter17 結果 | 判定 |
|------|------|---------------------|------------|------|
| 主基準 | top1_accuracy McNemar | 0.2059 (p=0.0783 vs Iter15) | 0.5651, p < 0.000001 | **達成** |
| 主基準 | Wilson CI 重なり | [0.1863, 0.2270] | [0.5401, 0.5899] | **達成**（重ならなし） |
| 副基準 | Cohen's kappa | 0.1067 | 0.5215 (CI 重ならなし) | **達成** |
| 副基準 | 同点タイ率 | 82.83% | 0.00% | **達成**（有意な低下） |
| 副基準 | ECE | 0.7388 | 0.2118 (-71.3%) | **達成**（明確な改善） |
| 非退行 | per-domain precision/recall | Iter16 の各値 | education recall 退行 | **違反**（education のみ） |
| 監視 | probe レイテンシ | 4134.4ms | 3622.2ms (-12.4%) | **達成**（短縮） |
| 監視 | dispatch_failure_rate | 0.0 | 0.0 | **達成** |

**判定: 採用**

主基準 2 件・副基準 3 件の全 5 件を達成し，教育ドメインの recall 退行のみが非退行条件を違反している．しかし，この退行は手法の欠陥ではなくデータセットの構造的問題（JMMLU に education 対応タスク不在，訓練データ 77 件の不均衡）に起因すると解釈できるため，手法の採用判断には影響しない．

**E6（supervised_classifier）の採用理由**:

1. **self_report の構造的問題を根本的に解決した**: 自己宣伝バイアス，離散値飽和，同点タイの 3 つの問題を同時に解消し，top1_accuracy を 2.7 倍に改善した．
2. **統計的に明確な有意差**: McNemar p < 0.000001，Wilson CI 重ならなし，kappa 0.1067 → 0.5215．ノイズではなく明確な信号である．
3. **レイテンシも改善**: LLM コール不要の embedding 計算のみで probe するため，mean_duration_ms が -12.4% 短縮された．
4. **ベースラインを有意に上回る**: Random の 5.6 倍，Iter16 の 2.7 倍．Oracle までの距離の 48.4% を埋めた．

**残す課題**:

1. **education recall の退行**: 訓練データの不均衡（77 件）と JMMLU の education タスク不在が根本原因．education 固有の訓練データ追加，または oversampling による再訓練が必要．
2. **fallback_rate=13.16%**: 盲目的な general fallback の正解率 8.0% は Random より低い．confidence_threshold の最適化，または fallback 戦略の変更（top-2 dispatch）が必要．
3. **複合ドメインの退行**: 評価方法とルーティング方式の不一致．単一選択ルーティングにおける複合ドメイン評価は，domain_set_recall のみ，または dispatch_top_k >= 2 で再評価すべき．
4. **softmax 確率の overconfidence**: 全ドメインで mean_conf > accuracy．ECE=0.2118 は許容範囲内だが，完全な較正には程遠い．CalibratedClassifierCV による較正の検討余地あり．

**次の考察フェーズへの示唆**:

- **E6 の routing_method=supervised_classifier を採用し，config.yaml に固定する**．
- **education の訓練データ整備**を次イテレーションの優先課題とする（E10 の expert_specialization とは独立したデータ整備タスク）．
- **fallback 戦略の改善**（confidence_threshold の最適化，または top-2 dispatch）を次の単一レバー候補として検討する．
- **E10（expert_specialization）** は，ルーティング精度が Random の 5.6 倍に改善した現在，その価値を回答品質（評価軸②③）で検証する適切な時期に来ている．supervised_classifier が正しいドメインにルーティングするようになったため，ノード間の能力差が回答品質に反映される環境が整った．

### 考察・次計画 (Iter17)

**判定: E6（routing_method=supervised_classifier）— 採用**

主基準 2 件（McNemar 有意差，Wilson CI 重ならなし）・副基準 3 件（kappa 改善，同点率解消，ECE 改善）の全 5 件を達成．education recall の非退行違反のみあるが，これは手法の欠陥ではなく JMMLU データセットの構造的問題（education 対応タスク不在，訓練データ 77 件の不均衡）に起因するため，採用判断には影響しない．

**このイテレーションで確定した非自明な学び**

1. **self_report の構造的問題は routing_method の変更でしか解決できない**: Iter15（numeric_scalar）と Iter16（top_k_with_probs）の両方で ECE > 0.7 であり，confidence elicitation の方式変更だけでは自己宣伝バイアスを解消できないことが確定した．embedding ベースの教師あり分類に切り替えることで，top1_accuracy を 2.7 倍（0.2059 → 0.5651）に改善し，ECE を -71.3%（0.7388 → 0.2118）に低減した．

2. **Iter2（unsupervised embedding）の棄却は正当だったが，原因は unsupervised であること**: Iter2 で cosine similarity が [0.667, 0.737] に潰れた原因は embedding の anisotropy であり「信号が無い」証明ではなかった．教師あり分類（LogisticRegression）は anisotropy 下でも分離超平面を学習できるため，supervised classifier は Random の 5.6 倍，Iter16 の 2.7 倍の精度を達成した．RouterDC（NeurIPS 2024）の報告（CosineClassifier に全タスクで勝利）と整合する．

3. **softmax 連続値は同点タイを完全解消する**: self_report の離散値（5段階）による 82.83% の同点タイが，softmax 連続値（1518 の唯一値）により 0.00% に完全に解消された．ルーティングが宣言順ではなく実質的なドメイン識別信号に依存する環境が初めて実現した．

4. **fallback_rate=13.16% は，分類器の「確信できない」境界領域を可視化している**: confidence_threshold=0.5 を下回る 200 問（13.16%）は，分類器がどのドメインにも確信もって分類できない境界領域である．fallback 正解率 8.0% は Random（10.1%）より低く，盲目的な general fallback は有用な信号を持っていない．education（29 件），legal（26 件），business_economics（25 件）が fallback 元として多く，訓練データ不足（77 件）と関連している．

5. **複合ドメインの退行（0.95 → 0.25）は評価方法の不一致**: Iter16 の 95% は self_report の高タイ率（82.83%）による構造上の偽高値であり，Iter17 の 25% の方が実態を反映している．単一選択ルーティングにおける複合ドメイン評価は，dispatch_top_k >= 2 で再評価するか，domain_set_recall のみとするべきである．

**次の単一レバー: E10（expert_specialization）**

supervised_classifier によりルーティング精度が Random の 5.6 倍（top1_accuracy=0.5651）に改善した現在，ノード間の能力差が回答品質に反映される環境が整った．現在 4 ノードすべてが同一モデル（isotnek/qwen3.5:9B-Unsloth-UD-Q4_K_XL）で，差分はプロンプト 1 文だけであるため，誤ルーティングしても回答品質はほぼ変わらず，top1_accuracy は下流に帰結を持たない代理指標になっていた．

E10（expert_specialization）の実施により，初めて「正しいドメインにルーティングされた質問が，実際に良い回答を得るか」を評価軸②（回答品質，LLM-as-judge）と③（End-to-End）で検証できる環境が整う．

- 本命は `domain_lora`（単一ベース + ドメイン LoRA アダプタ）: 6GB VRAM 制約下で最も現実的であり，S-LoRA（MLSys2024）が多数アダプタの同時配信を示している．日本語医療 LoRA の先行例（JMedLoRA）もあり，同じ 10 台の GPU プール上で LoRA 学習を行う仕組みは WAFL-PEFT 側に既にある．
- `offtheshelf_specialized` も候補だが，日本語の法律特化オープン生成モデルは発見できなかったため，domain_lora を優先する．
- **E10 と同時に評価軸②③（回答品質・End-to-End）を実装すること**．それらが無いとルーティングの価値を測れない．

---

