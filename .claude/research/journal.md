
## Iteration 46: aggregation_method変更による複数ノードdispatch時の集約方式比較

### 仮説

`aggregation_method` を `max_confidence` から `majority_vote` または `llm_judge` に変更することで、`compound_domain_set_recall` が有意に改善する。`dispatch_top_k=2` の下で、2つの専門家が異なる回答を生成した場合に、多数決またはLLM判定が正解にたどり着く確率が、単にconfidenceが高い方を選ぶより高くなる。

### 根拠

1. **Iter27 の教訓**: 2位ノードの適格件数は `dispatch_candidate_threshold=0.3` で 230 件（14.4%）。2つのノードが異なるJMMLUの4択回答を出したケースでは、max_confidenceは1つだけを選ぶが、多数決は合意形成により正解を特定できる可能性がある。

2. **compound_domain_set_recall の構造的上限**: `dispatch_top_k=1` では構造的上限 0.500。`dispatch_top_k=2` では上限 1.000。現状 Iter28 基準線（fallback廃止後）の compound_domain_set_recall=0.165 は、top_k=1 の限界によるもの。

3. **majority_vote の理論的利点**: 2つのノードが異なる回答を出した場合（例: 1つがA、1つがB）、max_confidenceはconfidenceの高い方だけを選ぶ。多数決は両方の回答を考慮し、もし両方が同じ回答（A,A）ならそれを採用する。これは2ノードが合意したケースのrecallを最大化する。

4. **llm_judge の理論的利点**: 専門家の回答をLLM判定で比較し、最も正確な回答を選ぶ。これは2つの回答のうち正解に近い方を選ぶ可能性があり、max_confidence（confidenceはルーティング精度の指標であらず回答品質の指標ではない）より優れる可能性がある。

### 単一レバー

**変更するレバー**: `aggregation_method` の値変更
- `max_confidence`（現行）→ `majority_vote`（第一候補）

**前提変更（レバーではない固定設定）**: `dispatch_top_k` を 1 → 2 に変更
- これは aggregation_method が意味を持つための前提条件。aggregation_method 比較の両側で同一（top_k=2）のため、単一レバー原則を逸脱しない。
- `dispatch_candidate_threshold` は Y2 で 0.0 に設定済み（変更なし）。

**固定レバー**:
- `routing_method=supervised_classifier`
- `confidence_threshold=0.0`（fallback 廃止済み、Iter28 adopted）
- `classifier_calibration=temperature`（Iter31 adopted）
- `classifier_head_adaptation=education_boundary_tuning (intercept_delta=+0.7)`（Iter44 adopted）
- `dispatch_candidate_threshold=0.0`（Y2 で新設）
- 分類器訓練データ、評価データセット、embedding model
- 他9ドメインの訓練データ

### 変更ファイル一覧

**変更ファイル**:
1. `config.yaml` — 2箇所変更
   - line 57: `dispatch_top_k: 1` → `dispatch_top_k: 2`
   - line 68: `aggregation_method: max_confidence` → `aggregation_method: majority_vote`

**新規作成ファイル**: なし

### 成功条件

1. **主基準**: `compound_domain_set_recall` が `majority_vote` において `max_confidence`（dispatch_top_k=2）ベースラインを **5pt 以上上回る**こと。
   - 根拠: Iter27 の answer_quality_accuracy の 3SD=2.6pt を踏まえ、測定ノイズの2倍以上の効果量を要求。
   - ベースライン: `dispatch_top_k=2, aggregation_method=max_confidence` の結果（同一イテレーション内で取得）
2. **非退行**: `top1_accuracy` の McNemar p >= 0.05（有意悪化なし）
3. **単一レバー検証**: `dispatched_domains` の長さが 2 以上になる件数が > 0 であること（aggregation_method が実際に発火した証拠）

### 失敗条件

1. `majority_vote` が `max_confidence` と比較して `compound_domain_set_recall` で 5pt 以上の改善を示さない
2. `top1_accuracy` が有意に悪化する（McNemar p < 0.05）
3. `dispatched_domains` の長さが 2 以上になる件数が 0 である（`dispatch_candidate_threshold` の設定誤り、またはコードパス到達エラー）

### ハイパラ値

- **dispatch_top_k**: 1 → 2
- **aggregation_method**: `max_confidence` → `majority_vote`（第一候補）
  - 第二候補: `llm_judge`（judge_model が必要、latency 増）
- **dispatch_candidate_threshold**: 0.0（変更なし。Y2 で設定済み）

### コスト見積もり

- **実装コスト**: 低（~5分）。`config.yaml` の 2 値変更のみ。コード変更不要。
- **実行コスト**: 中（~90-100分）。1600 問の実機本走 x 1 回（majority_vote）。max_confidence ベースラインは既存（Iter28）の結果が使えるが、厳密な対比のため同一環境で再実行を推奨。
- **オフライン完結**: いいえ（実機1600問本走が必要）

### 到達コードパスの確認

**`aggregation_method` が実際に読まれるコードパス**:

1. **`node.py:195`**: `aggregation_method = config.get("aggregation_method", AGGREGATION_METHOD_MAX_CONFIDENCE)`
   - 到達条件: `run_ask_flow()` が呼ばれる（`run_experiment.py:49` または `node.py:253`）

2. **`node.py:196`**: `validate_aggregation_method(aggregation_method)`
   - 到達条件: 同上。`majority_vote` は `VALID_AGGREGATION_METHODS` に含まれるので ValueError にならない。

3. **`node.py:238`**: `_dispatch_to_targets(..., aggregation_method, ...)`
   - 到達条件: `select_dispatch_targets()` が空でないリストを返す（targets が空でない）

4. **`node.py:141-143`**: `if aggregation_method == AGGREGATION_METHOD_MAJORITY_VOTE:`
   - 到達条件: `aggregation_method=majority_vote` で設定されていること
   - **発火条件**: `dispatch_top_k >= 2` かつ `dispatch_candidate_threshold` が十分低い

5. **`aggregator.py:98-125`**: `select_best_dispatch_response_majority_vote(dispatch_responses)`
   - 到達条件: 上記の分岐を通ること
   - **内部ロジック**:
     - `extract_answer_letter()` で各回答から A/B/C/D を抽出
     - 各文字の出現数をカウント
     - 最大票数 >= 2 の場合、その文字を持つ回答群から confidence が高いものを選択
     - 最大票数 < 2 の場合（全員別回答）、`select_best_dispatch_response()` にフォールバック（max_confidence 同等）

**`dispatch_top_k` が読まれるコードパス**:

6. **`node.py:217`**: `top_k=config.get("dispatch_top_k", 1)`
   - 到達条件: 同上
   - **変更点**: 1 → 2。`select_dispatch_targets()` が 2 件以上の候補を返すようになる。

7. **`aggregator.py:67`**: `return candidates[:top_k]`
   - 到達条件: `top_k=2` で設定されていること
   - **発火条件**: 2 位ノードの confidence >= `dispatch_candidate_threshold`（0.0）

**no-op にならないことの確認**:
- `dispatch_top_k=2` + `dispatch_candidate_threshold=0.0` の組み合わせにより、2位ノードの confidence は常に 0.0 以上（confidence は確率で負にならない）ため、2位ノードは必ずqualified。
- したがって `dispatched_domains` の長さは常に 2 以上になり、`aggregation_method` の分岐（node.py:141-143）は必ず発火する。
- **これは Iter27 の失敗（confidence_threshold=0.5 で2位がqualifiedにならなかった）とは異なり、今回の設定では確実に発火する。**

### 固定レバー

- `routing_method=supervised_classifier`
- `confidence_threshold=0.0`（fallback 廃止済み）
- `classifier_calibration=temperature`（Iter31 adopted）
- `classifier_head_adaptation=education_boundary_tuning (intercept_delta=+0.7)`（Iter44 adopted）
- `dispatch_candidate_threshold=0.0`（Y2 で新設）
- 分類器訓練データ、評価データセット、embedding model
- 他9ドメインの訓練データ
- `judge_model=schroneko/llama-3.1-swallow-8b-instruct-v0.1:q4_k_m`（llm_judge使用時のみ読まれるが、今回は majority_vote のため変更なし）

### 備考: llm_judge の比較について

`llm_judge` は `majority_vote` より高コスト（各dispatchでjudge_modelの追加LLM呼び出しが必要）だが、理論的にはより高性能な可能性がある。本次第では `majority_vote` を第一候補とし、結果が期待以上であれば `llm_judge` も比較対象とする。`llm_judge` を試す場合は追加のイテレーションを要する（~100-120分/回、judge_modelの生成時間による）。
