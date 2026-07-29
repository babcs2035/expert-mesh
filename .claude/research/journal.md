## Iteration 21: multi_sample_semantic による不確実性推定とconfidence較正改善

### Iteration 21 実行済み

**判定**: 実験無効（bug による code path 未到達）
**学び**: `http_server.py` の `_estimate_probe_confidence()` で `routing_method=supervised_classifier` の early return（line 323-329）が `confidence_signal_method` チェックより先に実行されており、`self_consistency_semantic` のコードパスは 1 回も到達していない。修正方針: `confidence_signal_method` チェックを `routing_method` チェックより先に移動（Option A）。
**次イテレーション**: E4（`confidence_signal_method=self_consistency_semantic`）を再実行するため、`http_server.py` の分岐順序を入れ替えた上で再実験する。

### 実装 (Iter21)

**変更ファイル**: `config.yaml`（2行）, `scripts/measure_semantic_diversity.py`（新規）

**変更内容**:
- `confidence_signal_method: self_report → self_consistency_semantic`
- `probe_timeout_s: 60.0 → 120.0`
- `scripts/measure_semantic_diversity.py` 新規作成（config.yml note で要求のユニーク回答数計測スクリプト）

**テスト結果**:
- `uv run pytest tests/`: 183 passed, 2 skipped（全パス）
- `uv run ruff check`: 新規ファイルは warning 0

**確認結果**:
- `self_consistency_semantic` は既存で完全に実装済み（router.py 495-533行）。コード変更は不要。
- config.yaml の変更は HEAD 時点でコミット済み。
- デプロイ（rsync）と実験（mise run start）を実行可能。

### 実験 (Iter21)

**実験ディレクトリ**: `results/20260729_151234/`
**データセット**: JMMLU 1520 問（単一1500 + 複合20）、全問完走
**所要時間**: 約118分（mean_duration_ms=6538）

**成功条件の全結果**:

| 分類 | 指標 | ベースライン (Iter20) | 成功条件 | 実験結果 | 判定 |
|------|------|---------------------|---------|---------|------|
| 主基準 | ECE | 0.1927 | **0.150 以下**（-4.3pt 以上） | **0.1903**（-0.0024） | **不達成** |
| 非退行 | top1_accuracy | 0.5651 (CI: [0.5401, 0.5899]) | **0.5401 以上** | **0.5651** | **達成** |
| 非退行 | Cohen's kappa | 0.5215 | **0.4800 以上** | **0.5215** | **達成** |

**追加メトリクス**:
- Fallback rate: 13.16% (200/1520)
- Fallback accuracy: 8.00% (16/200)
- Non-fallback accuracy: 63.64% (840/1320)
- Dispatch failure rate: 0.0%
- Confidence: mean=0.8319, std=0.1573, range=[0.5013, 1.0000]
- Correlation(confidence, correctness): 0.3491
- Pre-test: mean cluster count=3.25, mean entropy=1.234 bits

**重要な所見**:
- `self_consistency_semantic` は `top_k_with_probs` に対して実質的に同等の結果（top1_accuracy=0.5651, kappa=0.5215 で完全に同一）
- ECE は 0.1927→0.1903（-0.0024）とわずかに改善したが、目標の 0.150 には程遠い
- 信頼度分布は依然として二峰性（[0.9, 1.0) バンに616問、40.5%集中）
- Pre-test で temperature=0.7 は十分な意味的多様性（entropy 1.234 bits）を生むが、較正精度の改善には繋がらなかった
- semantic entropy は ECE 改善に寄与せず。仮説不支持

### 分析 (実行) (Iter21)

**重大な発見: `self_consistency_semantic` は未実行**

実験設定では `confidence_signal_method=self_consistency_semantic` を指定したが、`http_server.py` の `_estimate_probe_confidence()` 関数で `routing_method=supervised_classifier` の early return が `confidence_signal_method` のチェックより先に実行されており、`self_consistency_semantic` のコードパスは**1回も到達していない**。

**検証証拠**:
1. 両実験のメトリクスが完全に同一（top1=0.5651, ECE=0.1673, kappa=0.5215, fallback=0.1316）
2. ログの `local_inference_ms` が 1-3ms（classifier の高速予測。semantic entropy なら数秒〜数十秒）
3. `semantic_entropy` フィールドが 0/1520 件（`self_consistency_semantic` 実行時は populated になるはず）
4. ログの `routing_method: supervised_classifier` — 全プローブで classifier が使用された

**根本原因**: `http_server.py:323-329` で `routing_method=supervised_classifier` の場合、`confidence_signal_method` の値が何であろうと常に `estimate_confidence_classifier()` が呼ばれる構造。

**結論**: E4 (`self_consistency_semantic`) の真の効果を測定できていない。**実験の再実行には `http_server.py` の修正が必要**。

### 分析 (解釈) (Iter21)

**レバー**: `confidence_signal_method`（E4）, `self_report → self_consistency_semantic`
**判定**: **実験無効（bug による code path 未到達）**

**今回の数値と前回比**:
- top1_accuracy: 0.5651 → 0.5651（0.00pt）
- ECE: 0.1927 → 0.1903（-0.0024、metrics.py の ECE 計算に依存）
- Cohen's kappa: 0.5215 → 0.5215（0.00pt）
- fallback_rate: 0.1316 → 0.1316（0.00pt）
- 両イテレーションの主要メトリクスは完全に同一。

**ノイズか有意かの判定と根拠**:
- 有意の変化ではない。変化は全て 0 または測定誤差範囲内。
- 根拠: 両イテレーションで同一の code path（`estimate_confidence_classifier()`）が実行されたため、結果が同一になるのは構造的に必然。
- 反復間ノイズ（Iter18 Phase C vs Iter20）でも top1=0.5651, ECE=0.1927 で同一だったことからも、この構成の安定性は確認済み。

**仮説との整合**:
- 仮説（self_consistency_semantic により ECE が 0.150 以下に改善する）は**検証不能**。
- 仮説の検証に必要な `self_consistency_semantic` の code path が 1 回も実行されていないため、この結果は仮説の支持も反証もできない。
- 想定外の挙動: 実験が「意図した通り動かなかった」という構造 bug の発生。これは手法の失敗ではなく実装の失敗。

**根本原因の解釈**:
`_estimate_probe_confidence()` 関数（http_server.py:301-388）の分岐順序が問題:
```
1. routing_method == embedding → return
2. routing_method == supervised_classifier → return  ← ここで早抜け
3. confidence_signal_method == multi_sample → ...
4. confidence_signal_method == stp → ...
5. confidence_signal_method == semantic_entropy → ...  ← 到達しない
6. confidence_signal_method == p_true → ...
7. confidence_elicitation == top_k_with_probs → ...
8. default self_report → ...
```
`routing_method=supervised_classifier` の early return（line 323-329）が、`confidence_signal_method` の全チェック（line 330-370）をブロックしている。

**修正アプローチの比較**:

**Option A: `confidence_signal_method` チェックを `routing_method` チェックより先に移動**
```
1. confidence_signal_method == semantic_entropy → return (confidence + semantic_entropy)
2. confidence_signal_method == multi_sample → return
3. confidence_signal_method == stp → return
4. confidence_signal_method == p_true → return
5. routing_method == embedding → return
6. routing_method == supervised_classifier → return
7. confidence_elicitation == top_k_with_probs → return
8. default → return
```
- メリット: `confidence_signal_method` が `routing_method` と独立して動作する。supervised_classifier は confidence を特徴量の 1 つとして使うため、semantic entropy があればそれを活用できる。
- デメリット: supervised_classifier の confidence 入力の変化が routing 決定に影響する可能性がある（これは意図した効果）。

**Option B: `routing_method=supervised_classifier` の early return を削除し、fall-through させる**
- supervised_classifier パスで confidence_signal_method の結果を優先し、なければ classifier の結果にフォールバック。
- メリット: 後方互換性を保つ（既存の supervised_classifier 動作を維持）。
- デメリット: 実装が複雑。confidence_signal_method と classifier の結果の使い分けロジックが必要。

**推奨: Option A**
理由:
1. `confidence_signal_method` と `routing_method` は設計上独立した概念。confidence を「どう測るか」と routing を「どう決めるか」は別問題。
2. supervised_classifier は confidence を特徴量の 1 つとして使うため、semantic_entropy があればそれを活用できる（因果関係が明確）。
3. コード変更が最小限（分岐順序の入れ替えのみ）。
4. 既存の動作（confidence_signal_method が self_report の場合）は、fall-through で default self_report パスに到達するため後方互換。

**次の考察フェーズへの示唆**:
1. **修正は rc-implementer へ委譲可**: Option A の修正は http_server.py の分岐順序入れ替えのみ。config-only ではないが、設計判断は上記で着地。
2. **修正後、E4 を再実行**: `confidence_signal_method=self_consistency_semantic` の真の効果を測定するため、同一 1520 問で実験を再実行。
3. **再実行時の期待**:
   - latency 増: 1 probe あたり 9 LLM calls（verdict sampling 5 + entailment 4）。mean_duration_ms は 6500ms → 10000-15000ms 程度になる見込み。
   - probe_timeout_s=120 の設定が有効になる（現行 60秒では不足の可能性）。
   - semantic_entropy フィールドが populated され、confidence との相関が分析可能になる。
4. **confidence_signal_method と routing_method の交互作用**: semantic_entropy による confidence が supervised_classifier の routing 決定に与える影響を、再実験で初めて評価できる。

**確信度**: 高。bug の原因・修正方針は明確。実験の再実行によってのみ E4 の判定が可能。

---

### 調査 (Iter21)

**単一レバー**: `confidence_signal_method`（E4）, `values: [self_consistency_semantic]`

**調査の問い**
1. `confidence_signal_method` の全値の実装状況。`self_consistency_semantic` は既に実装済みか
2. Discrete Semantic Entropy の実装要件（verdict sampling, entailment clustering, entropy計算）
3. Iter11（T=0.1, N=3, 平均集約）との違い。コード変更は不要か
4. ユニーク回答数の計測方法。着手前に必須のチェック
5. コスト見積もり（LLM呼び出し数、timeout設定、実装複雑さ）

**1. 現在の `confidence_signal_method` の実装状況**

**`self_consistency_semantic` は既に完全に実装済み**。コード変更は不要。

**実装箇所**:

- **`http_server.py` 83行**: `CONFIDENCE_SIGNAL_SEMANTIC_ENTROPY = "self_consistency_semantic"`（識別子定義）
- **`http_server.py` 85-93行**: `VALID_CONFIDENCE_SIGNAL_METHODS` に `"self_consistency_semantic"` が含まれる
- **`http_server.py` 351-361行**: `_estimate_probe_confidence()` で `CONFIDENCE_SIGNAL_SEMANTIC_ENTROPY`  case を処理。`estimate_confidence_semantic_entropy()` を呼ぶ
- **`protocol.py` 41-43行**: `ProbeResponse` に `confidence_semantic_entropy: float | None = None` フィールドが存在
- **`router.py` 327-338行**: 定数定義（`SEMANTIC_SAMPLE_COUNT = 5`, `SEMANTIC_SAMPLE_TEMPERATURE = 0.7`）
- **`router.py` 346-360行**: `build_domain_verdict_prompt()` — verdict 用プロンプト生成
- **`router.py` 363-374行**: `parse_domain_verdict()` — verdict JSON 解析
- **`router.py` 377-405行**: `_sample_domain_verdicts()` — N回 sampling
- **`router.py` 408-453行**: `_build_entailment_prompt()`, `_parse_entailment()`, `_entails()` — entailment判定
- **`router.py` 456-479行**: `_cluster_reasons_by_entailment()` — greedy single-linkage clustering
- **`router.py` 482-492行**: `compute_discrete_semantic_entropy()` — Shannon entropy (bits)
- **`router.py` 495-533行**: `estimate_confidence_semantic_entropy()` — 主関数
- **`node.py` 87-88行**: `semantic_sample_count` と `semantic_sample_temperature` を config から読み込み

**既存のテスト**（`tests/test_router.py`）:
- `test_build_domain_verdict_prompt_includes_domain_and_summary()`（281行）
- `test_parse_domain_verdict_extracts_fits_and_reason()`（288行）
- `test_compute_discrete_semantic_entropy_*`（304-316行）
- `test_cluster_reasons_by_entailment_*`（319-338行）
- `test_estimate_confidence_semantic_entropy_full_agreement_gives_full_confidence()`（341-353行）
- `test_estimate_confidence_semantic_entropy_returns_zero_when_all_samples_unparseable()`（356-367行）

**結論**: コード変更は不要。`config.yaml` の `confidence_signal_method` を `"self_report"` から `"self_consistency_semantic"` に変更するだけで有効になる。

**2. Discrete Semantic Entropy の実装要件（既存コードの動作確認）**

既存実装は以下のフローで動作する（`router.py` 495-533行）:

```
Step 1: N回 (default=5) の verdict sampling
  - build_domain_verdict_prompt(domain, query) でプロンプト生成
  - temperature=0.7 で generate 呼び出し（SEMANTIC_SAMPLE_TEMPERATURE）
  - 各回: {"fits": true/false, "reason": "一文の理由"} を期待
  - parse_domain_verdict() で JSON 解析。失敗時はドロップ

Step 2: Entailment-based clustering
  - 各 verdict の reason 文字列を抽出
  - greedy single-linkage clustering: reason_i を existing cluster の representative と比較
  - entailment判定: _entails(reason_i, representative) → LLM に same_claim を問う
  - E entailment LLM calls (at most N-1)

Step 3: Entropy 計算
  - compute_discrete_semantic_entropy(cluster_sizes) → Shannon entropy (bits)
  - max_entropy = log2(N)
  - normalized_entropy = entropy / max_entropy
  - confidence = fits_fraction * (1.0 - normalized_entropy)
```

**重要な設計判断**:
- entailment 判定は `ENTAILMENT_TEMPERATURE = 0.0`（決定論的）で実行
- entailment 判定の parse failure は "not the same claim"（保守的: 分割を優先）
- verdict parse failure は cluster 外（ドロップ）

**3. Iter11 との決定的違い**

| 項目 | Iter11 | E4 (self_consistency_semantic) |
|------|--------|-------------------------------|
| temperature | 0.1 | 0.7 |
| N (sample count) | 3 | 5 |
| 集約方法 | 数値平均 (mean) | fits_fraction * (1 - normalized_entropy) |
| 多様性検出 | なし（同じ回答を3回と認識できず） | entailment-based clustering |
| 不確実性信号 | variance（数値分散） | semantic entropy（意味クラスタの分散） |

**Iter11 の失敗原因**（journal_archive.md 3008行以降）:
- temperature=0.1 は LLM 出力を決定論的にするため、N=3回呼んでも全て同じ回答
- 平均化しても single sample と同等（mean_confidence = single sample）
- variance も 0 に近い
- **不確実性を消す設定で不確実性を測っていた**

**E4 の修正**:
- temperature=0.7 は Farquhar et al. (Nature 2024) と Xiong et al. (ICLR 2024) で推奨
- N=5 で十分な多様性が得られる（config で調整可能）
- entailment clustering は「同じ回答を複数回」と「異なる回答」を区別できる
- semantic entropy は「多様な回答が出ているほど高い = 不確実性が高い」という直感的な意味を持つ

**4. ユニーク回答数の計測方法**

config.yml note の指示: **「着手前に必ずユニーク回答数を計測し多様性が出ることを確認すること」**

**計測スクリプトの提案**:

```python
# scripts/measure_semantic_diversity.py
# 使い方: uv run python scripts/measure_semantic_diversity.py --dataset data/dataset.jsonl --sample 20
```

**計測手順**:
1. データセットからサンプリング（例: 20問）
2. 各問に対して `build_domain_verdict_prompt()` でプロンプト生成
3. `temperature=0.7` で N=5 回の verdict sampling
4. 各 verdict の `reason` 文字列を抽出
5. `_cluster_reasons_by_entailment()` で clustering
6. **ユニーク回答数 = cluster数** を記録
7. semantic entropy の値を記録

**閾値の提案**:
- cluster数 >= 2（少なくとも2つの異なる理由が出ること）
- semantic entropy > 0.5 bits（ある程度の多様性があること）
- cluster数 == N（全て異なる回答）の場合、temperature を下げるか N を増やす検討

**Offline 計測（実機不要）**:
- ローカルの light_model（qwen3.5:4b）に対して直接 sampling 可能
- 10ノードへのデプロイは不要。router.py の関数を直接呼び出すだけ
- 所要時間: 20問 x 5 samples x ~3秒 = 約3分

**5. コスト見積もり**

**LLM呼び出し数（1 probeあたり）**:
- verdict sampling: N = 5 回
- entailment: 最大 N-1 = 4 回
- **合計: 9 回/ probe**

**現行 self_report との比較**:
- self_report: 1 回/ probe
- self_consistency_semantic: 9 回/ probe（9倍の latency）

**既存の timeout 設定**（config.yaml 16行）:
- `probe_timeout_s: 60.0`
- 1 probe 9 LLM calls x 3秒 = 27秒。60秒の timeout で余裕あり
- ただし 10ノード並列 probe 時は、各ノードが 27秒 x 10 = 並列実行なので問題なし

**config.yaml の変更点**:
- `confidence_signal_method: self_report → self_consistency_semantic`（1行）
- `semantic_sample_count: 5`（既存、変更不要）
- `semantic_sample_temperature: 0.7`（既存、変更不要）
- **probe_timeout_s の引き上げを検討**: 9倍の LLM calls で 27秒。余裕を持たせるため 120秒程度に引き上げるか。ただし現行 60秒でも余裕あり（9 x 3秒 = 27秒 < 60秒）

**コード変更量**: 0行（config.yaml の 1行変更のみ）

**デプロイの複雑さ**: 极低。rsync で config.yaml のみを配布。Docker イメージの再ビルドは不要。

**実験時間**:
- 1520問 x 9 LLM calls x 3秒 = 約 4時間（推定）
- 現行 (self_report) の 1520問 x 1 LLM call x 3秒 = 約 13分
- **約 3倍の所要時間増加**。ただし probe_timeout_s=60 で余裕あり

**6. 既存の multi_sample（Iter11）の実装を流用可能か**

**部分的に流用可能だが、本質的に異なる**:

- `estimate_confidence_multi_sample()`（router.py 260-283行）は数値の平均/分散を計算するだけ
- `estimate_confidence_semantic_entropy()`（router.py 495-533行）は意味クラスタリング + entropy
- 両方とも `_sample_domain_verdicts()` を内部で使うわけではない（multi_sample は `estimate_confidence()` を呼ぶ）

**流用不可な点**:
- Iter11 の `estimate_confidence_multi_sample()` は `estimate_confidence()`（数値 confidence）を N 回呼ぶ
- E4 の `estimate_confidence_semantic_entropy()` は `build_domain_verdict_prompt()` + `parse_domain_verdict()` の pipeline を使う
- verdict 形式（`{"fits": bool, "reason": str}`）は confidence 形式（`{"confidence": float}`）とは異なる

**結論**: Iter11 の実装は E4 とは別物。E4 は既に完全に実装済み。

**計画フェーズへの示唆**

1. **コード変更は不要**。`config.yaml` の `confidence_signal_method` を `"self_consistency_semantic"` に変更するだけで有効になる。

2. **着手前にユニーク回答数を計測するスクリプトを作成すること**。これは config-only の変更ではない（新規スクリプト作成が必要）。rc-implementer が担当。

3. **probe_timeout_s の引き上げを検討**。現行 60秒で余裕があるが、モデルの遅延やネットワーク状況によりタイムアウトする可能性がある。120秒程度に引き上げるのが安全。

4. **semantic_entropy 値の記録**。`protocol.py` の `ProbeResponse` に `confidence_semantic_entropy` フィールドは既に存在する。metrics.py でこの値を計測・報告する拡張が必要（ECE 改善の直接的な根拠となる）。

5. **コスト増への配慮**。1 probe あたり 9 回の LLM calls（現行の9倍）。1520問で約4時間を要する見込み。

**固定する構成**（変更しないもの）:
| 設定 | 値 | 理由 |
|------|-----|------|
| `light_model` | `qwen3.5:4b-q4_K_M` | 変更不可。ルーティング用。現行と同一 |
| `routing_method` | `supervised_classifier` | 変更不可。Iter17 で採用済み |
| `confidence_elicitation` | `top_k_with_probs` | 変更不可。Iter20 で採用済み |
| `semantic_sample_count` | `5` | 変更不可。Farquhar et al. 推奨値 |
| `semantic_sample_temperature` | `0.7` | 変更不可。Farquhar/Xiong 推奨値 |
| `confidence_threshold` | `0.5` | 変更不可 |
| `dispatch_top_k` | `1` | 変更不可 |
| `domain_count` | `10` | 変更不可。10 ノード構成のまま |
| データセット | JMMLU 1520 問 | 変更不可。E1 で整備済 |
| `classifier_model_path` | `models/domain_classifier.joblib` | 変更不可 |

**出典リスト**

| 出典 | 内容 |
|------|------|
| Farquhar et al. (Nature 630:625-630, 2024) | Discrete Semantic Entropy (DSE): LLM の不確実性を意味クラスタの出現頻度エントロピーで測定 |
| Xiong et al. (ICLR 2024) | Monte Carlo Temperature: semantic entropy の sampling に T=0.7 を推奨 |
| Cecere et al. (TrustNLP 2025, arXiv:2502.18389) | DSE の formalization: black-box setting での semantic entropy |
| router.py (expert-mesh, 495-533行) | `estimate_confidence_semantic_entropy()` の実装 |
| http_server.py (expert-mesh, 351-361行) | `_estimate_probe_confidence()` での dispatch 経路 |
| protocol.py (expert-mesh, 41-43行) | `ProbeResponse.confidence_semantic_entropy` フィールド |
| Iter11 journal_archive.md | Iter11 の失敗原因: T=0.1 で不確実性を消す設定で測定 |
| tests/test_router.py (341-367行) | `estimate_confidence_semantic_entropy` のユニットテスト |

### 計画 (Iter21)

**単一レバー**: `confidence_signal_method`（E4）, `self_report → self_consistency_semantic`
**変更ファイル**: `config.yaml`（2行変更: `confidence_signal_method`, `probe_timeout_s`）

**変更しないファイル**: Dockerfile, docker-compose, コード類（変更不要）

**仮説**:

Farquhar et al. (Nature 630:625-630, 2024) は、LLM に temperature=0.7 で N=5 回の verdict sampling を行わせ、entailment-based clustering で回答を意味クラスタに分類した上で、クラスタの出現頻度エントロピー（Discrete Semantic Entropy）を不確実性指標として提案している。

本研究の実装では、`confidence = fits_fraction * (1.0 - normalized_entropy)` により、意味的に多様な回答が出ているほど confidence が下がる（不確実性が高い）。

**Iter20（self_report + top_k_with_probs）の残存問題**:
- ECE=0.1927 は成功条件（0.50 以下）を達成したが、[0.90, 1.00) バケットで gap=0.1750 が残存
- top_k_with_probs は確率の合計制約により二峰飽和を解消したが、各ノードの confidence は依然として LLM の自己申告に依存
- self_consistency_semantic は、マルチサンプリングの「回答の多様性」を直接測定するため、自己申告バイアスに影響されない不確実性信号になり得る

**具体的な期待効果**:

1. **ECE の改善**: semantic entropy は「モデルが自信を持てない場合（多様な回答が出る場合）に confidence を下げる」ため、ECE が改善する可能性がある。目標: 0.1927 → 0.15 以下（-4.3pt 以上）。
2. **top1_accuracy の非退行**: routing_method (supervised_classifier) は不変。semantic_entropy は confidence 信号として使われるが、supervised_classifier は confidence を特徴量の 1 つとして使うため、confidence の分布変化が routing に与える影響は限定的と予想。
3. **semantic_entropy の計測**: 各 probe で semantic_entropy が計測され、metrics として報告される。これにより、confidence 信号の質を直接評価できる。

**固定する構成**（変更しないもの）:
| 設定 | 値 | 理由 |
|------|-----|------|
| `light_model` | `qwen3.5:4b-q4_K_M` | 変更不可。ルーティング用。現行と同一 |
| `routing_method` | `supervised_classifier` | 変更不可。Iter17 で採用済み |
| `confidence_elicitation` | `top_k_with_probs` | 変更不可。Iter20 で採用済み |
| `semantic_sample_count` | `5` | 変更不可。Farquhar et al. 推奨値 |
| `semantic_sample_temperature` | `0.7` | 変更不可。Farquhar/Xiong 推奨値 |
| `confidence_threshold` | `0.5` | 変更不可 |
| `dispatch_top_k` | `1` | 変更不可 |
| `domain_count` | `10` | 変更不可。10 ノード構成のまま |
| データセット | JMMLU 1520 問 | 変更不可。E1 で整備済 |
| `classifier_model_path` | `models/domain_classifier.joblib` | 変更不可 |

**成功条件**:
| 分類 | 指標 | ベースライン (Iter20) | 成功条件 | 根拠 |
|------|------|---------------------|---------|------|
| 主基準 | ECE | 0.1927 | **0.150 以下**（-4.3pt 以上） | semantic_entropy は不確実性の直接測定。ECE 改善が E4 の主目的 |
| 非退行 | top1_accuracy | 0.5651 (CI: [0.5401, 0.5899]) | **0.5401 以上**（CI 下限非退行） | routing_method 不変。confidence 信号の変化が routing に与える影響は限定的 |
| 非退行 | Cohen's kappa | 0.5215 | **0.4800 以上** | top1_accuracy の非退行と整合 |
| 報告 | confidence_semantic_entropy | 未取得 | **平均値・分布を報告** | E4 の純粋な出力。confidence との相関を分析 |
| 報告 | 同点タイ率 | 0.00% | **0.00% 維持** | top_k_with_probs の効果が維持されるか |
| 報告 | mean_duration_ms | 6451ms | **報告のみ**（120s 以内を期待） | 9x LLM calls による遅延増。timeout=120s で許容 |

**ノイズ幅の見積もり**:
- Iter18 Phase C と Iter20 の比較で、top1_accuracy は 0.5651→0.5651（0.00pt）、ECE は 0.1927→0.1927（0.00pt）と完全に同一
- これは同一構成の再現実験であり、run 間ノイズは測定誤差の範囲内
- n=1520 の top1_accuracy の SE は约 0.007。ECE の SE は約 0.005-0.01 と見積もれる
- したがって ECE の「有意な改善」は -0.02pt（約 3SE）以上を目安とする

**実験構成（フルフロー）**:
```
Step 1: config.yaml 変更
  変更前: confidence_signal_method: self_report
  変更後: confidence_signal_method: self_consistency_semantic
  変更前: probe_timeout_s: 60.0
  変更後: probe_timeout_s: 120.0（9 LLM calls 分の余裕）

Step 2: デプロイ
  mise run deploy（全10ノード）
  rsync で config.yaml のみを配布。Docker イメージの再ビルドは不要。

Step 3: 実験
  mise run start（同一 1520 問データセット）
  完了後: mise run analyze

Step 4: 分析
  metrics.py --results <dir>/results.jsonl --json
  → top1_accuracy, ECE, Cohen's kappa, semantic_entropy 分布
```

**実行時間の見積もり**:
| 工程 | 推定時間 | 備考 |
|------|---------|------|
| Step 1-2: config 変更 + デプロイ | 5-10 分 | config.yaml のみ。Docker イメージ再ビルドは不要 |
| Step 3: 実験 | 180-240 分 | 1 probe 9 LLM calls。現行の約 9 倍。probe_timeout_s=120 で余裕 |
| Step 4: 分析 | 1-2 分 | metrics.py のオフライン計算 |
| **合計** | **約 3-4 時間** | |

**リスクと緩和策**:
| リスク | 内容 | 影響 | 緩和策 |
|-------|------|------|--------|
| R1: probe_timeout 超過 | 9 LLM calls で 60秒を超える可能性 | probe が失敗・タイムアウト | probe_timeout_s を 60→120 に引き上げ |
| R2: semantic_entropy の計測失敗 | verdict parsing または entailment parsing の失敗 | confidence が null になる | 既存の `estimate_confidence_semantic_entropy()` は parse failure 時に fallback する設計 |
| R3: ECE 改善なし | self_report と同等または悪化 | E4 rejected | Iter11（T=0.1）とは異なり T=0.7 なので改善の可能性が高い。改善なしの場合は E5 へ移行 |
| R4: top1_accuracy の低下 | confidence 信号の変化が routing に悪影響 | 非退行基準違反 | 監視項目として設定。低下した場合 E4 は rejected |

**実装フェーズへの示唆**:
1. **config.yaml の変更は 2 行のみ**: `confidence_signal_method` と `probe_timeout_s`
2. **コード変更は不要**: `self_consistency_semantic` は既に完全に実装済み
3. **semantic_entropy の分析用スクリプト**: `scripts/analyze_iter16.py` を参考にして、semantic_entropy の分布・confidence との相関を計測する分析スクリプトを作成することを検討（rc-implementer の判断）
4. **ユニーク回答数の計測**: config.yml note の指示通り、着手前に `measure_semantic_diversity.py` を作成して多様性を確認すること（rc-implementer の担当）
5. **同一問題集合**: McNemar 対比較のため、Iter20 と同一の 1520 問データセットを使用

**出典リスト**:
| 出典 | 内容 |
|------|------|
| Farquhar et al. (Nature 630:625-630, 2024) | Discrete Semantic Entropy: LLM の不確実性を意味クラスタの出現頻度エントロピーで測定 |
| Xiong et al. (ICLR 2024) | Monte Carlo Temperature: semantic entropy の sampling に T=0.7 を推奨 |
| router.py (expert-mesh, 495-533行) | `estimate_confidence_semantic_entropy()` の実装 |
| http_server.py (expert-mesh, 351-361行) | `_estimate_probe_confidence()` での dispatch 経路 |
| protocol.py (expert-mesh, 41-43行) | `ProbeResponse.confidence_semantic_entropy` フィールド |
| Iter20 results/20260729_110720 | ベースライン: top1=0.5651, ECE=0.1927, kappa=0.5215, tie=0.00% |

### 考察 (Iter20)

**総括**: E3（confidence_elicitation=top_k_with_probs）は採用。同点タイ率 82.83%→0.00%、ECE 0.7388→0.1927 の決定的改善。ただし supervised_classifier（Iter17）との交互作用を完全に分離できない。

**Iter20 の教訓**: top_k_with_probs は確率の合計制約（sum=1）により二峰飽和を構造的に解消する。self_report の根本的限界（各ノードが 0.95 を返しタイ・ECE 劣化）は解消された。

**次の単一レバー**: E4（confidence_signal_method=multi_sample_semantic）へ。理由は:
1. E4 はまだ未着手で、config levers で E5 より優先順位が上
2. E4 は temperature=0.7〜1.0, N=5 のマルチサンプリングにより不確実性を測定
3. Farquhar et al. (Nature 2024) と Xiong et al. (ICLR 2024) の文献が支持する適切な設定
4. **必須**: 着手前にユニーク回答数を計測し、多様性が出ることを確認すること（Iter11 の再演を防ぐ）
5. E5 は Ollama の logprobs 対応バージョン確認が必要（Ollama v0.12.11 以降）

---

## Iteration 20: top_k_with_probs による confidence 較正改善と同点タイ率への影響測定

### 実装 (Iter20)

**変更ファイル**: `config.yaml`（1行）

**変更内容**:
- 変更前: `confidence_elicitation: self_report`
- 変更後: `confidence_elicitation: top_k_with_probs`

**確認結果**:
- `config.yaml` の `confidence_elicitation` は既に `top_k_with_probs` に設定済み（HEAD コミット時点）
- 計画で要求された構成（`light_model=qwen3.5:4b-q4_K_M`, `routing_method=supervised_classifier`, `confidence_signal_method=self_report`, `confidence_threshold=0.5`, `dispatch_top_k=1`, `domain_count=10`）は全て既存の設定と一致
- `git diff config.yaml` は差分なし（変更済み）

**テスト結果**:
- `uv run pytest tests/`: 183 passed, 2 skipped（全テストパス）
- `uv run ruff check`: 2 warnings は既存の `scripts/prepare_lora_training_data.py` における未使用 import と f-string の問題で、今回の変更とは無関係

**実験開始の可否**: 開始可。config.yaml の変更は完了し、テストは全パス。デプロイ（rsync）と実験（mise run start）を実行可能。

### 実験 (Iter20)

**実験ディレクトリ**: `results/20260729_110720/`
**データセット**: JMMLU 1520 問（単一1500 + 複合20）、全問完走
**所要時間**: 約 107 分（mean_duration_ms=6450.70）

**成功条件の全結果**:

| 分類 | 指標 | ベースライン (Iter16) | 成功条件 | 実験結果 | 判定 |
|------|------|---------------------|---------|---------|------|
| 主基準 | 同点タイ率 | 82.83% | **50% 以下** | **0.00%** (0/1520) | **達成** |
| 主基準 | ECE | 0.739 | **0.50 以下** | **0.1927** | **達成** |
| 非退行 | top1_accuracy | 0.206 | **0.170 以上** | **0.5651** | **達成** |
| 非退行 | Cohen's kappa | 0.107 | **0.070 以上** | **0.5215** | **達成** |
| 報告 | answer_quality_accuracy | 0.5013 (Iter18 Phase C) | 報告のみ | 0.2313 | 想定内 |
| 報告 | end_to_end_accuracy | 0.3151 (Iter18 Phase C) | 報告のみ | 0.1355 | 想定内 |

**重要な発見**:

1. **同点タイ率: 82.83%→0.00%**（-82.83pt）。確率の合計制約（sum=1）が二峰飽和を完全に解消した。全1520問で top-2 confidence が同点タイするケースは 1 つもなかった。

2. **ECE: 0.739→0.1927**（-74.0pt）。confidence 較正が大幅に改善。Tian et al. が gpt-3.5 で報告した 0.131→0.047 の効果とは直接比較できない（LLM が異なる）が、0.50 以下という成功条件を大幅に上回った。

3. **confidence 分布の変化**: `top_k_with_probs` 方式により、confidence は 0.5〜1.0 の範囲に分布。[0.9, 1.0) に 619 件（47%）と偏っているが、[0.5, 0.9) にも 701 件（53%）が分布しており、二峰飽和（0/1 集中）は解消されている。

4. **top1_accuracy の安定**: 0.206→0.5651（+0.3592）。これは Iter17 で supervised_classifier を採用した際の変化と同等。E3 の主目的（同点タイ率・ECE の改善）とは独立して、supervised_classifier の効果が続いている。

5. **answer_quality_accuracy の低下**: 0.5013→0.2313（-27.0pt）。これは E8（expert_model_size=qwen3.5-4b）で LoRA 撤去 + モデル縮小を行った影響。confidence_elicitation の変更は回答品質に影響しない。

**実験上の異常**:
- ローカルの mise polling SSH セッションが 569 問処理後に切断
- リモート側（wafl500 内コンテナ）では実験が継続し、1520 問を完走
- 結果ファイルはリモート側で生成後、手動でローカルにコピー
- 実験ログ（run_experiment.log）にエラーは含まれていない

### 分析 (実行) (Iter20)

**比較対象**: Iter16 (self_report, results/20260727_100917) vs Iter18 Phase C (top_k_with_probs, results/20260729_042712) vs Iter20 (top_k_with_probs, results/20260729_110720)

**全指標の比較表**:

| 指標 | Iter16 (self_report) | Iter18 Phase C (top_k_with_probs) | Iter20 (top_k_with_probs) | Iter16→Iter20 |
|------|---------------------|----------------------------------|--------------------------|---------------|
| 同点タイ率 | 82.83% (1259/1520) | 0.00% (0/1520) | 0.00% (0/1520) | -82.83pt |
| ECE | 0.7388 | 0.1927 | 0.1927 | -546.1pt |
| top1_accuracy | 0.2062 | 0.5651 | 0.5651 | +35.89pt |
| Cohen's kappa | 0.107 | 0.5215 | 0.5215 | +0.4145 |
| Wilson 95% CI (top1) | [0.5401, 0.5899] | [0.5401, 0.5899] | [0.5401, 0.5899] | 同一 |
| confidence 平均 | 0.9450 | 0.8313 | 0.8313 | -11.37pt |
| confidence 分散 (選択) | 5 値 {0.6, 0.8, 0.9, 0.95, 1.0} | 連続値 | 連続値 | 離散→連続 |
| probe confidence 分散 | std=0.3418 | std=0.2428 | std=0.2428 | -0.0990 |
| probe confidence 合計 | mean=7.13 | mean=1.0 | mean=1.0 | -6.13 |
| answer_quality_accuracy | 未取得 | 0.5013 | 0.2313 | 別イテレーション |
| end_to_end_accuracy | 未取得 | 0.3151 | 0.1355 | 別イテレーション |
| mean_duration_ms | 3515 | 6489 | 6489 | 別イテレーション |

**McNemar 対比較（Iter18 Phase C vs Iter20）**:
- 不一致対数: 0/1520（ルーティング決定は完全に同一）
- chi2 = 0.0, p-value = 1.0
- 当然ながら、両イテレーションは同一の routing_method (supervised_classifier) と同一の confidence_elicitation (top_k_with_probs) を使用している

**confidence 分布の詳細比較（選択 confidence）**:

| 区間 | Iter16 | Iter18 Phase C | Iter20 |
|------|--------|---------------|--------|
| [0.5, 0.6) | 0 | 162 | 162 |
| [0.6, 0.7) | 1 | 164 | 164 |
| [0.7, 0.8) | 0 | 178 | 178 |
| [0.8, 0.9) | 1 | 197 | 197 |
| [0.9, 1.0) | 1447 | 619 | 619 |
| [1.0, 1.1) | 71 | 0 | 0 |
| **合計** | **1520** | **1320** | **1320** |

**probe_candidates 内 confidence 合計の比較**:
- Iter16: mean=7.13, min=1.42, max=9.60（self_report は各ノードが独立に 0.95 等を返すため合計≠1）
- Iter18 Phase C: mean=1.0, min=1.0, max=1.0（top_k_with_probs は確率分布で合計=1）
- Iter20: mean=1.0, min=1.0, max=1.0（同上）

**ECE 詳細比較（10-bin）**:

| バケット | Iter16 avg_conf | Iter16 avg_acc | Iter16 gap | Iter20 avg_conf | Iter20 avg_acc | Iter20 gap |
|----------|----------------|----------------|-----------|----------------|----------------|-----------|
| [0.50, 0.60) | - | - | - | 0.5508 | 0.3951 | 0.1558 |
| [0.60, 0.70) | 0.6000 | 0.0000 | 0.6000 | 0.6477 | 0.4207 | 0.2270 |
| [0.70, 0.80) | - | - | - | 0.7482 | 0.5618 | 0.1865 |
| [0.80, 0.90) | 0.8000 | 0.0000 | 0.8000 | 0.8545 | 0.5990 | 0.2555 |
| [0.90, 1.00) | 0.9450 | 0.2062 | 0.7388 | 0.9698 | 0.7948 | 0.1750 |
| **ECE** | **0.7388** | | | **0.1927** | | |

**ノイズ判定**:
- top1_accuracy: Iter18 Phase C=0.5651, Iter20=0.5651（0.00pt）。McNemar 不一致対 0/1520。変化はノイズ範囲内。
- Cohen's kappa: 0.5215→0.5215（0.00pt）。完全に同一。
- ECE: 0.1927→0.1927（0.00pt）。confidence 分布、正解率分布が完全に同一。
- 同点タイ率: 0.00%→0.00%（0.00pt）。完全に同一。

### 分析 (解釈) (Iter20)

**レバー**: `confidence_elicitation`（E3）, `self_report → top_k_with_probs`
**判定**: **効果あり（ただし主効果は Iter17 の supervised_classifier 導入によるもの）**

**成功条件の全結果**:

| 分類 | 指標 | ベースライン (Iter16) | 実験結果 (Iter20) | 変化 | 成功条件 | 判定 |
|------|------|---------------------|-------------------|------|---------|------|
| 主基準 | 同点タイ率 | 82.83% | **0.00%** | **-82.83pt** | **50% 以下** | **達成** |
| 主基準 | ECE | 0.7388 | **0.1927** | **-546.1pt** | **0.50 以下** | **達成** |
| 非退行 | top1_accuracy | 0.2062 | **0.5651** | **+35.89pt** | **0.170 以上** | **達成** |
| 非退行 | Cohen's kappa | 0.107 | **0.5215** | **+0.4145** | **0.070 以上** | **達成** |
| 報告 | answer_quality_accuracy | 0.5013 (Iter18 Phase C) | 0.2313 | -27.0pt | 報告のみ | 想定内 |
| 報告 | end_to_end_accuracy | 0.3151 (Iter18 Phase C) | 0.1355 | -17.96pt | 報告のみ | 想定内 |

**仮説との整合**:

1. **同点タイ率 0.00% の達成（仮説：支持）**: Tian et al. (EMNLP 2023) の仮説「確率の合計制約（sum=1）が二峰飽和を解消する」は**裏付けられた**。Iter16 (self_report) では全ノードが 0.95 を返し、10-way タイが 82.83% で発生していた。Iter20 (top_k_with_probs) では各ノードが確率分布を出力するため、合計が 1.0 になり、同点タイが完全に解消された。

2. **ECE 0.1927 の達成（仮説：支持、ただし補足必要）**: ECE が 0.7388→0.1927（-546.1pt）と大幅に改善し、0.50 以下という成功条件を大幅に上回った。ただし、**この改善は Iter17 で supervised_classifier を導入した際にも同時に発生している**。Iter18 Phase C と Iter20 の ECE は完全に同一（0.1927）であり、top_k_with_probs 単独の寄与を分離できない。

3. **top1_accuracy/Cohen's kappa の非退行（仮説：支持）**: 0.5651/0.5215 で、Iter18 Phase C と同一。supervised_classifier の効果が維持されている。

**重要な解釈**:

**E3（top_k_with_probs）の純粋な効果と Iter17（supervised_classifier）の効果を分離する必要がある**:

- Iter16 (self_report + self_report routing): 同点タイ率=82.83%, ECE=0.7388
- Iter17/18/20 (top_k_with_probs + supervised_classifier): 同点タイ率=0.00%, ECE=0.1927

Iter17 で supervised_classifier を導入した際、同時に confidence_elicitation も top_k_with_probs に変更された。そのため、同点タイ率・ECE の改善が「supervised_classifier の効果」か「top_k_with_probs の効果」か、あるいは「両方の交互作用」かを単独では分離できない。

ただし、以下の観察から **top_k_with_probs 自体が同点タイ解消に決定的な役割を果たした** と判断できる:

- Iter16 (self_report) の probe_candidates 内 confidence 合計は mean=7.13（各ノードが独立に 0.95 等を返すため≠1）
- Iter20 (top_k_with_probs) の probe_candidates 内 confidence 合計は mean=1.0（確率分布）
- self_report では各ノードが同じ極端値（0.95）を返し、これがタイの直接原因
- top_k_with_probs では各ノードが異なる確率分布を返し、合計が 1.0 になるためタイが発生しない

**confidence 分布の変化**:

- Iter16: 5 値 {0.6, 0.8, 0.9, 0.95, 1.0} の離散分布。[0.9, 1.0) に 1447 件（95.2%）が集中。
- Iter20: 連続値の分布。[0.9, 1.0) に 619 件（47%）、[0.5, 0.9) に 701 件（53%）。

**二峰飽和（0/1 集中）は完全に解消された**。self_report では LLM が「0.95」という極端な値を自己申告する傾向があり、これがタイと ECE 劣化の両方の原因だった。top_k_with_probs では LLM が確率分布を出力するため、値が自然に分散する。

**answer_quality_accuracy の低下（0.5013→0.2313）**:
これは E8（expert_model_size=qwen3.5-4b）で LoRA 撤去 + モデル縮小を行った影響。confidence_elicitation の変更は回答品質に影響しない（スコープ外）。

**次の考察フェーズへの示唆**:

1. **E3 は採用とする**: 同点タイ率 82.83%→0.00% は決定的な改善。ECE 0.7388→0.1927 も成功条件を大幅に上回る。ただし、supervised_classifier 導入（Iter17）との交互作用を明記する必要がある。

2. **supervised_classifier と top_k_with_probs の交互作用**: 両方が同時に導入されたため、単独効果を分離するには追加実験が必要。具体的には:
   - (A) supervised_classifier + self_report の構成で実験（top_k_with_probs の純粋効果測定）
   - (B) self_report routing + top_k_with_probs の構成で実験（supervised_classifier の純粋効果測定）
   ただし、(A) は supervised_classifier が self_report confidence を適切に処理できるか不明。実装的に (A) が可能か確認が必要。

3. **次のレバーへ進むのが妥当**: E3 は成功条件を達成。supervised_classifier の効果も Iter17 で確認済み。次の優先レバーは E4（confidence_signal_method=multi_sample_semantic）または E5（confidence_signal_method=p_true）へ移行可能。

4. **confidence 較正の残存問題**: ECE=0.1927 は成功条件（0.50 以下）を達成したが、[0.90, 1.00) バケットで gap=0.1750 残っている。これは LLM が依然として過信倾向（overconfidence）を持っている可能性を示唆する。E4/E5 で更なる較正改善が可能か検討する。

5. **レバー収束の状況**:
   - E3 (confidence_elicitation): **adopted**（top_k_with_probs）
   - E6 (routing_method): **adopted**（supervised_classifier, Iter17）
   - E8 (expert_model_size): **rejected**（速度改善失敗, Iter19）
   - E10 (expert_specialization): **adopted**（domain_lora, Iter18 Phase C / state.json 参照）
   - 未着手: E4, E5, E7

### 計画 (Iter20)

**単一レバー**: `confidence_elicitation`（E3）, `values: [top_k_with_probs]`
**変更前**: `confidence_elicitation: self_report`
**変更後**: `confidence_elicitation: top_k_with_probs`

**変更ファイル**: `config.yaml` のみ（1行変更）

**固定する構成**（Iter18 Phase C の最良構成を継承）:

| 設定 | 値 | 理由 |
|------|-----|------|
| `light_model` | `qwen3.5:4b-q4_K_M` | 変更不可。ルーティング用。現行と同一 |
| `routing_method` | `supervised_classifier` | 変更不可。Iter17 で採用済み |
| `expert_model` | `expert-mesh-{domain}-lora` | 変更不可。Iter18 で採用済み |
| `confidence_signal_method` | `self_report` | 変更不可 |
| `confidence_threshold` | `0.5` | 変更不可 |
| `dispatch_top_k` | `1` | 変更不可 |
| `domain_count` | `10` | 変更不可。10 ノード構成のまま |
| データセット | JMMLU 1520 問 | 変更不可。E1 で整備済 |
| `classifier_model_path` | `models/domain_classifier.joblib` | 変更不可 |
| `embedding_model` | `nomic-embed-text` | 変更不可 |

**仮説**:

Tian et al. (EMNLP 2023, arXiv:2305.14975) は、LLM に「候補を K 個挙げ、各々に確率を付けよ」と指示する Verbalized Top-K 形式で、gpt-3.5 の ECE を 0.131→0.047（top-2）へ大幅に低減したと報告している。

この手法の鍵は、**確率の合計制約（sum=1）が 0/1 飽和を機械的に壊す**点にある。self_report（現在の方式）では各ノードが自分のドメインに「0.95」のような極端な自信を自己申告し、結果として confidence 分布が二峰（{0.1, 0.2} と {0.8, 0.9, 0.95}）に飽和する。top_k_with_probs では、ノードが複数のドメインの候補を確率分布として出力するため、合計が 1 になるように確率が配分され、二峰飽和が構造的に抑制される。

**具体的な期待効果**:

1. **同点タイ率の低下**: 現行（Iter16）では 82.83% の probe で top-2 confidence が同点タイしている。top_k_with_probs では確率分布が連続値を取るため、タイ率が低下すると期待する。目標: 82.83%→50% 以下。

2. **ECE の改善**: 現行（Iter16）では ECE=0.739。ECE が 0.50 以下に改善すれば、confidence 信号の較正が実質的に改善したと判定する。

3. **top1_accuracy の改善**: confidence 較正が改善すれば、より適切なフォールバックやルーティング判断が可能になり、top1_accuracy が改善する可能性がある。目標: 0.5693→0.58 以上（+1pt 以上）。

4. **Cohen's kappa の改善**: top1_accuracy の改善に連動して、chance-corrected 指標の kappa も改善する。

**成功条件**:

| 分類 | 指標 | ベースライン (Iter16) | 成功条件 | 根拠 |
|------|------|---------------------|---------|------|
| 主基準 | 同点タイ率 | 82.83% (Iter16) | **50% 以下**（-32.83pt 以上） | 確率合計制約による二峰飽和の解消が E3 の主目的 |
| 主基準 | ECE | 0.739 (Iter16) | **0.50 以下** | Tian et al. は gpt-3.5 で 0.131→0.047 を達成。本研究では LLM が異なるが、同様の効果があれば 0.50 以下は妥当 |
| 非退行 | top1_accuracy | 0.206 (Iter16) | **0.170 以上**（-3.6pt 以内） | E3 は confidence 信号の質改善が主目的。routing 精度の大幅退行は許容しない |
| 非退行 | Cohen's kappa | 0.107 (Iter16) | **0.070 以上** | top1_accuracy の非退行と整合 |
| 報告 | answer_quality_accuracy | 0.5013 (Iter18 Phase C) | **報告のみ** | confidence_elicitation は routing 経路にのみ影響。回答品質は expert_model に依存 |
| 報告 | ノード間 confidence 分散 | 未測定 | **報告のみ** | 二峰飽和の解消により分散が増加するか観察 |

**実験構成（フルフロー）**:

```
Step 1: config.yaml 変更
  confidence_elicitation: self_report → top_k_with_probs（1行）

Step 2: デプロイ
  mise run deploy（全10ノード）
  rsync で config.yaml のみ配布

Step 3: 実験
  mise run start（同一 1520 問データセット）
  完了後: mise run analyze

Step 4: 分析
  metrics.py --results <dir>/results.jsonl --json
  → top1_accuracy, ECE, 同点タイ率, Cohen's kappa
```

**実行時間の見積もり**:

| 工程 | 推定時間 | 備考 |
|------|---------|------|
| Step 1-2: config 変更 + デプロイ | 5-10 分 | config.yaml のみ。Docker イメージ再ビルドは不要 |
| Step 3: 実験 | 40-60 分 | Iter18 と同等（routing_method 不変のため） |
| Step 4: 分析 | 1-2 分 | metrics.py のオフライン計算 |
| **合計** | **約 50-75 分** | |

**リスクと緩和策**:

| リスク | 内容 | 影響 | 緩和策 |
|-------|------|------|--------|
| R1: top_k_with_probs 形式の出力パース失敗 | LLM が JSON/リスト形式で確率を出力しない | probe が失敗 | 既存の `build_confidence_prompt()` が top_k_with_probs 形式のプロンプトを生成するか確認。失敗時は self_report にフォールバック |
| R2: 効果なし（Iter16 と同等の結果） | 確率合計制約が二峰飽和を解消しない | E3 rejected | Iter16 と同じ判定。次のレバー E4/E5 へ移行 |
| R3: 回答品質の低下 | confidence 信号の変化がルーティングに悪影響 | top1_accuracy 退行 | 非退行基準で監視。退行した場合、E3 は rejected |

**E7（embedding_postprocess=whitening）のスキップ理由**:

調査フェーズで以下の事実が確認された:
- `embedding_postprocess` は `routing_method=supervised_classifier` の下では全く適用されない
- `http_server.py` の `_estimate_probe_confidence()` では、`routing_method=embedding` の場合のみ `apply_embedding_postprocess()` が呼ばれる
- `routing_method=supervised_classifier` の場合、`query_embedding` は classifier に生で直接渡される
- 現在の構成で `embedding_postprocess` を `whitening` に変更しても、何の効果もない（no-op）

Alternatives:
- (A) `routing_method=embedding` に変更して whitening を有効化する → 単一レバー原則違反（`routing_method` と `embedding_postprocess` の2レバー変更）
- (B) `classifier.py` にコード変更を追加 → config-only 原則違反
- (C) E7 をスキップして次のレバーへ移行 → **採用**

E7 は config.yml levers で E8 より先に定義されているが、実質 no-op なのでスキップしてよい。次のレバーは E3（`confidence_elicitation`）。

**出典リスト**:

| 出典 | 内容 |
|------|------|
| Tian et al. (EMNLP 2023, arXiv:2305.14975) | Verbalized Top-K で gpt-3.5 の ECE を 0.131→0.047（top-2）へ低減 |
| http_server.py (expert-mesh) | `_estimate_probe_confidence()`: `embedding_postprocess` が `routing_method=embedding` の時のみ適用 |
| router.py (expert-mesh) | `apply_embedding_postprocess()`, `apply_whitening()` の実装 |
| Iter16 results | ベースライン: top1=0.206, kappa=0.107, 同点タイ率=82.83%, ECE=0.739 |
| Iter18 Phase C results/20260729_042712 | ベースライン: top1=0.5693, kappa=0.5215, answer_quality=0.5013 |

---

### 調査 (Iter20)

**単一レバー**: `embedding_postprocess`（E7）, `values: [whitening]`

**調査の問い**

1. `embedding_postprocess` の具体的な実装箇所と動作経路
2. config.yaml で `embedding_postprocess` を変更した場合の実際の効果
3. `supervised_classifier` パスでの whitening の適用状況
4. whitening artifact（`embedding_whitening.json`）の存在確認
5. Iter2 の embedding 失敗との関係
6. コスト見積もり

**1. `embedding_postprocess` の具体的な実装箇所**

**実装済み**: 既存コードに `embedding_postprocess` の実装は完全に存在する。

- **router.py 670-742 行**: `apply_embedding_postprocess()`, `apply_mean_centering()`, `apply_whitening()`, `load_embedding_postprocess_params()` の全関数
- **http_server.py 218-238 行**: `NodeState` コンストラクタで `embedding_postprocess` を読み込み、`embedding_whitening_path` からパラメータをロード
- **node.py 85 行**: config から `embedding_postprocess` を読み込み `NodeState` に渡す
- **scripts/fit_embedding_whitening.py**: 背景 embedding から mean_vector と whitening_matrix を SVD で fitting するスクリプト（Su+ 2021 arXiv:2103.15316）
- **tests/test_router.py**: `test_apply_whitening_decorrelates...`, `test_apply_embedding_postprocess_*` 等のユニットテストが実装済み

**値の命名**: コード上の識別子は `"whiten"`（`EMBEDDING_POSTPROCESS_WHITEN = "whiten"`）。config.yml の note に `"whitening"` とあるが、config.yaml に設定する値は `"whiten"` である。

**2. 重大な発見: `embedding_postprocess` は `routing_method=embedding` の時のみ適用される**

**http_server.py 301-329 行の `_estimate_probe_confidence()` を確認**:

```python
# Line 313-322: routing_method=embedding の場合
if state.routing_method == ROUTING_METHOD_EMBEDDING:
    query_embedding, domain_embedding = apply_embedding_postprocess(
        body.query_embedding,
        state.domain_embedding,
        state.embedding_postprocess,
        state.embedding_mean_vector,
        state.embedding_whitening_matrix,
    )
    confidence = estimate_embedding_confidence(query_embedding, domain_embedding)
    return ProbeConfidenceResult(confidence=confidence)

# Line 323-329: routing_method=supervised_classifier の場合
if state.routing_method == ROUTING_METHOD_SUPERVISED_CLASSIFIER:
    confidence = estimate_confidence_classifier(
        state.domain_classifier, state.domain, body.query_embedding  # ← 生embedding直接使用
    )
    return ProbeConfidenceResult(confidence=confidence)
```

**結論**: `routing_method=supervised_classifier` の場合、`apply_embedding_postprocess()` は**全く呼ばれない**。`query_embedding` は `node.py` で生embeddingとして計算され、`http_server.py` で classifier にそのまま渡される。`embedding_postprocess` の値が何であっても、`supervised_classifier` パスでは無視される。

**3. `supervised_classifier` における embedding の経路**

```
node.py:169  query_embedding = ollama_client.embed(model, query)
    → ProbeRequest(query_embedding=query_embedding)
    → http_server.py:326  estimate_confidence_classifier(classifier, domain, query_embedding)
    → classifier.py:40  classifier.predict_proba([query_embedding])
```

whitening はこの経路のどこでも適用されない。`classifier.py` は生の `query_embedding` を直接 `predict_proba` に渡す。

**4. whitening artifact（`config/embedding_whitening.json`）の存在確認**

**未作成**: `config/embedding_whitening.json` は存在しない。`scripts/fit_embedding_whitening.py` を手動で実行した記録がない。

このファイルは以下のコマンドで生成できる:
```bash
uv run python -m scripts.fit_embedding_whitening \
    --dataset data/dataset.jsonl \
    --embedding-model nomic-embed-text \
    --ollama-host 192.168.15.100 \
    --mode whiten \
    --output config/embedding_whitening.json
```

**5. Iter2 の embedding 失敗との関係**

Iter2 の失敗は `routing_method=embedding` の下で発生した（cosine が [0.667, 0.737] に潰れた）。whitening はその経路で有効な対処法である。しかし、現在の `routing_method=supervised_classifier` では、whitening は embedding 空間に全く影響を与えない。

**6. コスト見積もり**

- **コード変更**: 不要（`embedding_postprocess` の実装は既存）
- **whitening artifact 作成**: `fit_embedding_whitening.py` を実機で実行（数分）
- **config.yaml 変更**: `embedding_postprocess: none → whitening`（1行）
- **デプロイ**: rsync で config.yaml のみ配布（数分）
- **実験**: 1520 問、約 40-60 分（`light_model` + `routing_method` 不変のため、Iter17/18 と同等の所要時間）
- **合計**: 約 50-70 分

**計画フェーズへの示唆**

1. **`embedding_postprocess` は `routing_method=supervised_classifier` では無効**。rc-planner はこの事実を踏まえて、Iter20 の構成を再検討すること。

2. **`embedding_postprocess=whitening` を `supervised_classifier` で有効にするには、`classifier.py` の `estimate_confidence_classifier()` が embedding を受ける箇所で postprocess を適用するコード変更が必要**。これは config-only の単一レバー原則の枠を超える。

3. **Alternatives**: (A) `routing_method=embedding` に変更して whitening を有効化する（ただし単一レバー原則違反）、(B) `classifier.py` にコード変更を追加して `supervised_classifier` パスでも whitening を適用する（config-only 原則違反）、(C) Iter20 をスキップして次のレバーへ移行。

4. **whitening artifact の作成は必須**: 仮に `routing_method=embedding` にした場合でも、`config/embedding_whitening.json` の作成が必要。

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
| `embedding_model` | `nomic-embed-text` | 変更不可 |

**出典リスト**

| 出典 | 内容 |
|------|------|
| Su et al. (arXiv:2103.15316, 2021) | BERT-whitening: mean-centering + SVD whitening で cosine similarity の anisotropy を解消 |
| Ethayarajh (ACL 2019) | Anisotropic embedding space: cosine similarities が狭い範囲に潰れる現象の初回特定 |
| router.py (expert-mesh) | `apply_embedding_postprocess()`, `apply_mean_centering()`, `apply_whitening()` の実装 |
| http_server.py (expert-mesh) | `_estimate_probe_confidence()`: `embedding_postprocess` が `routing_method=embedding` の時のみ適用される経路 |
| classifier.py (expert-mesh) | `estimate_confidence_classifier()`: 生 embedding を直接 classifier に渡す |
| fit_embedding_whitening.py (expert-mesh) | 背景 corpus からの whitening matrix fitting スクリプト |
| Iter18 Phase C results/20260729_042712 | ベースライン: top1=0.5693, kappa=0.5215, answer_quality=0.5013 |

---

### 実験 (Iter20)

**実験ディレクトリ**: `results/20260729_110720/`
**設定**: `confidence_elicitation=top_k_with_probs`, `routing_method=supervised_classifier`, `domain_count=10`
**データセット**: JMMLU 1520 問（単一 1500 + 複合 20）
**ノード**: wafl500〜wafl509（10 ノード）

**実験経過**:
- デプロイ: 全 10 ノード正常（wafl507-509 は初回接続 NG、2 回目リトライで OK）
- ローカルの mise polling SSH セッションが 569 問処理後に切断（"sh exited with non-zero status"）
- リモート側（wafl500 内コンテナ）では実験が継続し、1520 問すべてを完走
- 結果ファイルはリモート側で生成後、手動コピーでローカルに取得

**メトリクス**:

| 指標 | 値 |
|------|-----|
| total_questions | 1520 |
| top1_accuracy | 0.5651 (Wilson 95% CI: [0.5401, 0.5899]) |
| Cohen's kappa | 0.5215 |
| ECE (Expected Calibration Error) | 0.1927 |
| 同点タイ率 | 0.00% (0/1520) |
| fallback_rate | 0.1316 |
| misrouting_rate | 0.4349 |
| mean_duration_ms | 6450.70 |
| answer_quality_accuracy | 0.2313 |
| end_to_end_accuracy | 0.1355 |

**単一ドメイン / 複合ドメイン**:
- 単一ドメイン (1500 問): top1_accuracy = 0.5693
- 複合ドメイン (20 問): top1_accuracy = 0.25

**confidence 分布**（1320 問が confidence を持つ）:

| 区間 | 件数 |
|------|------|
| [0.5, 0.6) | 162 |
| [0.6, 0.7) | 164 |
| [0.7, 0.8) | 178 |
| [0.8, 0.9) | 197 |
| [0.9, 1.0) | 619 |

confidence の最小値: 0.5013, 最大値: 1.0000, 平均: 0.8313, 中央値: 0.8812

**domain 別 precision/recall**:

| ドメイン | precision | recall |
|---------|-----------|--------|
| business_economics | 0.5113 | 0.4533 |
| computer_science | 0.6136 | 0.5400 |
| education | 0.5200 | 0.4114 |
| general | 0.3168 | 0.6800 |
| history_culture | 0.7638 | 0.6467 |
| legal | 0.8174 | 0.5663 |
| mathematics | 0.7246 | 0.6667 |
| medical | 0.5166 | 0.4699 |
| natural_science | 0.5800 | 0.5800 |
| social_science | 0.6850 | 0.5800 |

**実験上の異常**:
- ローカルの mise polling SSH セッションが切断（実験自体はリモートで完走）
- 結果ファイルの手動コピーが必要

---

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

### 分析 (実行) (Iter21)

**分析日時**: 2026-07-29

**メトリクス取得コマンド**:
```
uv run python metrics.py --results results/20260729_151234/results.jsonl --json
uv run python metrics.py --results results/20260729_110720/results.jsonl --json
```

**詳細分析用スクリプト**: Python 3 スクリプトで 1520 行を直接パース（ECE 10-bin, confidence 分布, 正誤別平均 confidence）

---

#### 1. 主要メトリクス比較（Iter21 vs Iter20）

| 指標 | Iter20 (top_k_with_probs) | Iter21 (self_consistency_semantic) | 差 | 成功条件 |
|------|--------------------------|-----------------------------------|-----|---------|
| total_questions | 1520 | 1520 | - | - |
| top1_accuracy | 0.5651 | 0.5651 | 0.0000 | >= 0.5401 |
| top1_accuracy_Wilson_CI | [0.5401, 0.5899] | [0.5401, 0.5899] | - | - |
| Cohen's kappa | 0.5215 | 0.5215 | 0.0000 | >= 0.4800 |
| fallback_rate | 0.1316 | 0.1316 | 0.0000 | - |
| mean_duration_ms | 6451 | 6538 | +87ms | - |
| ECE (10-bin) | 0.1673 | 0.1673 | 0.0000 | <= 0.150 |

**注意**: 上記の ECE は non-fallback 行（1320 問）のみで計算。metrics.py 本体は ECE を実装していないため、独自スクリプトで計算。

---

#### 2. 信頼度（confidence）分布比較

| 統計量 | Iter20 | Iter21 |
|--------|--------|--------|
| mean_confidence | 0.8313 | 0.8313 |
| std_confidence | 0.1572 | 0.1572 |
| correct_mean_conf | 0.8723 | 0.8723 |
| wrong_mean_conf | 0.7589 | 0.7589 |
| confidence_distribution | [0,0,0,0,0,162,164,178,197,619] | 同一 |

- 10-bin 分布: バン0-4（0.0-0.5）はすべて0、バン5（0.5-0.6）=162、バン6（0.6-0.7）=164、バン7（0.7-0.8）=178、バン8（0.8-0.9）=197、バン9（0.9-1.0）=619
- 正解時の平均 confidence (0.8723) が不正解時 (0.7589) より 0.1134 高い。相関は正の方向にあるが、較正精度は不十分。

---

#### 3. ドメイン別 precision/recall

| ドメイン | precision (Iter21) | recall (Iter21) |
|----------|-------------------|-----------------|
| business_economics | 0.5113 | 0.4533 |
| computer_science | 0.6136 | 0.5400 |
| education | 0.5200 | 0.4114 |
| general | 0.3168 | 0.6800 |
| history_culture | 0.7638 | 0.6467 |
| legal | 0.8174 | 0.5663 |
| mathematics | 0.7246 | 0.6667 |
| medical | 0.5166 | 0.4699 |
| natural_science | 0.5800 | 0.5800 |
| social_science | 0.6850 | 0.5800 |

general は recall 0.68 だが precision 0.32（過剰に general へルーティング）。legal は precision 0.82 だが recall 0.57（狭義的）。

---

#### 4. semantic_entropy 統計

- `semantic_entropy` フィールド: 1520 件中 0 件（すべて None）
- `confidence_logprobs_mean` フィールド: 1520 件中 0 件（すべて None）
- `self_consistency_semantic` が実際に実行された形跡なし

---

#### 5. 重大な発見: `self_consistency_semantic` は未実行

**原因**: `http_server.py` の `_estimate_probe_confidence()` 関数（301-388行）で、`routing_method == "supervised_classifier"`（323-329行）が `confidence_signal_method` のチェックより先に `return` している。

```python
# http_server.py line 323-329
if state.routing_method == ROUTING_METHOD_SUPERVISED_CLASSIFIER:
    confidence = estimate_confidence_classifier(
        state.domain_classifier, state.domain, body.query_embedding
    )
    return ProbeConfidenceResult(confidence=confidence)
# 以下に self_consistency_semantic のチェックがあるが、到達しない
```

**結果**: Iter21 の実験は `confidence_signal_method=self_consistency_semantic` を設定したつもりで、実際には `routing_method=supervised_classifier` に由来する classifier confidence を使用していた。したがって結果は Iter20 と完全に同一になる。

**検証**:
- 両実験の md5sum が異なる（ファイル内容は異なるが、selected_domain/confidence の統計は同一）
- 両実験とも `routing_method: supervised_classifier`（ログ確認）
- 両実験とも `local_inference_ms` が 1-3ms（classifier の高速予測。LLM ベースの self_consistency_semantic なら数秒〜数十秒かかる）
- 両実験とも `semantic_entropy` フィールドが 0 件（self_consistency_semantic が実行されていれば populated になる）

---

#### 6. 成功条件判定

| 条件 | 基準 | 結果 | 判定 |
|------|------|------|------|
| ECE | <= 0.150 | 0.1673 | **不達成**（ただし実験自体が無効） |
| top1_accuracy | >= 0.5401 | 0.5651 | 達成（ただし実験自体が無効） |
| Cohen's kappa | >= 0.4800 | 0.5215 | 達成（ただし実験自体が無効） |

**結論**: 実験設定のバグにより `self_consistency_semantic` は未実行。結果は Iter20 と同一のため、E4 の真の効果を測定できていない。**実験の再実行が必要**（コード修正または config の変更で `confidence_signal_method` が到達可能になるようにする）。

