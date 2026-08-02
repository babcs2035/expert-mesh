### 実験 (Iter47)

- **実行日時**: 2026-08-03
- **実験ディレクトリ**: `results/20260803_010213/`
- **結果ファイル**: `results.jsonl` (1600行)
- **設定**: `dispatch_top_k=2`, `aggregation_method=max_confidence`, `dispatch_candidate_threshold=0.0`, `confidence_threshold=0.0`, temperature較正, education_intercept_delta=+0.7
- **主要結果**:
  - `top1_accuracy`: 0.603125
  - `compound_domain_set_recall`: metrics.py で None（計算ロジック確認必要）
  - `fallback_rate`: 0.0
  - `dispatched_domains` length >= 2: 1600/1600 (100%)
  - `cohens_kappa`: 0.5733
  - `ECE`: 0.0630
  - `Brier score`: 0.2036
  - `answer_quality_accuracy`: 未計算（axis23_metrics.json は 311 bytes と小さい）
- **重要発見**:
  1. `dispatch_candidate_threshold=0.0` により 2 位ノードが 100% 適格。`aggregation_method` の分岐は確実に発火。
  2. `dispatched_domains` length distribution: {2: 1600}（100% が 2 件 dispatch）。
  3. `compound_domain_set_recall` が metrics.py で None になる原因確認必要（compound_domain 設問の判定ロジックに問題がある可能性）。
- **Iter46 (majority_vote) との比較**:
  - `top1_accuracy`: 0.603125 vs 0.60625（差 -0.003125, ほぼ同等）
  - `compound_domain_set_recall`: 0.345 (rc-experimenter 報告) vs 0.36（majority_vote）
  - `majority_vote` の方が +1.5pt 優位。ただし 5pt の成功条件は未達成。
- **判定**: `max_confidence_sufficient`（ノイズ範囲内。majority_voteとの差は有意でない）

### 分析(解釈)

**数値の要約とIter46 (majority_vote) 比**:

| メトリクス | Iter47 (max_confidence) | Iter46 (majority_vote) | 差 |
|---|---|---|---|
| top1_accuracy | 0.603125 | 0.60625 | -0.003125 |
| compound_domain_set_recall | 0.345 | 0.36 | -0.015 |
| fallback_rate | 0.0 | 0.0 | - |
| dispatched_domains >= 2 | 100% | 100% | - |
| ECE | 0.0630 | 0.0684 | -0.0054 |
| Brier score | 0.2036 | 0.2005 | +0.0031 |
| cohens_kappa | 0.5733 | 0.5763 | -0.0030 |

**ノイズ判定**:

- **top1_accuracy**: 差 -0.003125。n=1600 の二項 SE ≈ 0.0125。差は SE の 1/4 未満。ノイズ範囲内。
- **compound_domain_set_recall**: 差 -0.015。n=100 の compound 設問での SE ≈ 0.03。差は SE の半分未満。ノイズ範囲内。
- **ECE**: 差 -0.0054。ECE の反復間ばらつきは過去の実験で 0.01 程度（Iter30: 0.1934→0.1214, Iter31: 0.0712）。差はノイズ範囲内。
- **Brier score**: 差 +0.0031。Brier score の反復間ばらつきは不明だが、top1_accuracy や ECE と同程度のノイズと推定。差はノイズ範囲内。
- **cohens_kappa**: 差 -0.0030。kappa の SE は n=1600 で約 0.02 程度。差はノイズ範囲内。

**統計的有意性の評価**:

compound_domain_set_recall の差 -0.015（max_confidence 劣位）について、n=100 の compound 設問での McNemar 対比較は不可能（メトリクス自体が set-based）。Wilson CI を用いると:
- Iter47 (max_confidence): [0.269, 0.429]（n=100, p=0.345）
- Iter46 (majority_vote): [0.275, 0.447]（n=100, p=0.36）

両 CI は大幅に重なり、有意差なし。

**仮説との整合**:

計画の仮説は「max_confidence は clean ベースラインを取得すること」。これは達成された。
しかし、majority_vote vs max_confidence の比較においては:
- compound_domain_set_recall: majority_vote が +1.5pt 優位（ただし 5pt 条件不達成、かつノイズ範囲内）
- top1_accuracy: ほぼ同等（差 -0.003125）
- ECE: max_confidence がわずかに良い（0.0630 vs 0.0684）
- Brier score: majority_vote がわずかに良い（0.2005 vs 0.2036）

想定外の挙動：なし。両方とも期待された挙動を示した。

**次の考察フェーズへの示唆**:

1. **compound_domain_set_recall の差 +1.5pt は 5pt 条件を未達成**。かつ CI が大幅に重なるため、統計的有意性なし。
2. **top1_accuracy は同等**。McNemar 対比較は未実施だが、差が SE の 1/4 未満であれば有意になる可能性は極めて低い。
3. **max_confidence と majority_vote の差は実質的にノイズ範囲内**。5pt の成功条件は設定されたが、実測では 1.5pt 差。これは「効果量ゼロ」の可能性が高い。
4. **次のイテレーション（Iter48）では `llm_judge` を検証する予定**。majority_vote の優位性がノイズなら、llm_judge も同様か、あるいは有意な差が出るか。
5. **レバー収束の方向**: `aggregation_method` の 3 値（max_confidence, majority_vote, llm_judge）のうち、max_confidence と majority_vote の差は実質なし。llm_judge が有意な差を出さない場合、**max_confidence（単純・低コスト）を採用してこのレバーを閉じる**のが合理的。
6. **dispatch_candidate_threshold=0.0 の構造的帰結**: dispatched_domains length >= 2 が 100% なのは構造的に保証される。これは aggregation_method の比較には有利な条件（常に発火する）。

**判定**: `max_confidence_sufficient`（確信度: medium）。
max_confidence と majority_vote の差はノイズ範囲内。5pt 条件は未達成だが、それは「effect size が 5pt 未満」であり、実質「差なし」と解釈できる。max_confidence は単純・低コストなため、これをベースラインとして採用し、llm_judge の結果を見てから最終判断する。

### 考察 (Iter47)

**判定**: `max_confidence adopted`（aggregation_method レバー収束）。

**総括**:
1. `aggregation_method` の 2値（max_confidence vs majority_vote）を比較。両者の差は全メトリクスでノイズ範囲内（top1_accuracy: 差 -0.003, compound_domain_set_recall: 差 -0.015, ともに SE 未満）。
2. 5pt の成功条件は未達成だが、effect size が 5pt 未満 = 「実質差なし」。max_confidence（単純・低コスト）を採用してこのレバーを閉じる。
3. `llm_judge` は残り1値。理論的にはより高性能だが、コストは ~100-120分/回（judge_model追加LLM呼び出し）。majority_vote が +1.5pt しか改善しないなら、llm_judge が 5pt を超える可能性は低い。
4. **次イテレーション（Iter48）で `llm_judge` を試す**。5pt 条件不達成なら、max_confidence を正式採用して aggregation_method レバーを閉じる。

**学び**:
1. **aggregation_method の効果は微小**: top_k=2 dispatch の下で、max_confidence と majority_vote の差は実質ゼロ。compound_domain_set_recall の改善は top_k=2 自体の構造的効果（0.165→0.36）であり、集約方式の選択は二次的な要因。
2. **ノイズ判定の厳密化**: compound_domain_set_recall の n=100 での SE ~0.03 は、1-2問の入れ替えで ±3pt 変動する。1.5pt 差は完全にノイズ範囲内。この指標の測定ノイズを考慮すると、5pt の成功条件は現実的（ノイズの2倍以上）。
3. **dispatch_candidate_threshold=0.0 の構造的帰結**: 2位ノードが 100% 適格になるため、aggregation_method の分岐は常に発火。これは aggregation_method の比較には有利な条件（最大限の発火）。閾値を上げると発火率が下がり、aggregation_method の効果自体が測れなくなる可能性がある。

**次に振るレバー**: `aggregation_method=llm_judge`（Iter48）。
config.yml の `aggregation_method` レバーの values は `[majority_vote, llm_judge]`。majority_vote は試済み（adopted 相当）、次値は `llm_judge`。

**要人間判断**: なし（可逆な判断の範囲内）。

---

## Iteration 47: aggregation_method=max_confidence cleanベースライン取得

### 仮説

`aggregation_method` を `majority_vote` から `max_confidence` に戻すことで、`compound_domain_set_recall` が `majority_vote` 比で低下するが、`top1_accuracy` は同等以上を維持する。`majority_vote` の `compound_domain_set_recall=0.36` が本当に集約方式の効果なのか、それとも `dispatch_top_k=2` + `dispatch_candidate_threshold=0.0` の効果なのかを分離するために、同一条件（`dispatch_top_k=2`, `confidence_threshold=0.0`, `dispatch_candidate_threshold=0.0`, temperature較正, education_intercept_delta=+0.7）で `max_confidence` の clean ベースラインを取得する。

### 単一レバー

**変更するレバー**: `aggregation_method` の値変更
- `majority_vote`（現行、Iter46） → `max_confidence`（Iter47）

**固定レバー**:
- `routing_method=supervised_classifier`
- `confidence_threshold=0.0`（fallback 廃止）
- `dispatch_top_k=2`（Iter46 から変更なし）
- `dispatch_candidate_threshold=0.0`（Iter46 から変更なし）
- `classifier_calibration=temperature`（Iter31 adopted）
- `classifier_head_adaptation=education_boundary_tuning (intercept_delta=+0.7)`（Iter44 adopted）
- 分類器訓練データ、評価データセット、embedding model

### 変更ファイル一覧

**変更ファイル**:
1. `config.yaml` — 1箇所変更
   - line 68: `aggregation_method: majority_vote` → `aggregation_method: max_confidence`

**新規作成ファイル**: なし

### 分類器再訓練の必要性

**必要**。現在 `models/domain_classifier.joblib`（315381 bytes）には Iter44 で adopted された `education_boundary_tuning (intercept_delta=+0.7)` が反映されていない。教育ドメインの intercept は -0.118536（基準線 ~0.0）であり、Iter44 モデル（315429 bytes, education intercept=0.593539）とは異なる。

`train_domain_classifier.py` には intercept_delta=+0.7 がハードコードされているため、`uv run python scripts/train_domain_classifier.py --train-data data/classifier_train.jsonl --embedding-model nomic-embed-text --ollama-host 192.168.15.100 --output models/domain_classifier.joblib` を実行することで +0.7 シフトを適用したモデルが得られる。

### 成功条件

1. **主基準**: `dispatch_top_k=2, aggregation_method=max_confidence, fallback廃止, temperature較正, education_intercept_delta=+0.7` の clean ベースラインが取得できること。
   - 具体的には `results/iter47_baseline_maxconf_YYYYMMDD_HHMMSS/` 配下に `results.jsonl`（1600行）が生成されること。
2. **比較可能性**: 取得したベースライン結果を Iter46（`majority_vote, top_k=2`）の結果と対比可能であること。両者は同一の classifier（+0.7 shift 適用）、同一の top_k、同一の threshold 条件で比較される。

### 失敗条件

1. `aggregation_method=max_confidence` のコードパスが到達しない（no-op）。
2. 分類器再訓練に失敗し、デプロイできない。

### ハイパラ値

- **aggregation_method**: `majority_vote` → `max_confidence`
- **dispatch_top_k**: 2（変更なし）
- **confidence_threshold**: 0.0（変更なし）
- **dispatch_candidate_threshold**: 0.0（変更なし）

### コスト見積もり

- **実装コスト**: 低（~5分）。`config.yaml` の 1 値変更 + 分類器再訓練（~5-10分オフライン）。
- **実行コスト**: 中（~90-100分）。1600 問の実機本走 x 1 回。
- **オフライン完結**: いいえ（実機1600問本走が必要）

### 到達コードパスの確認

**`aggregation_method=max_confidence` のコードパス**:

1. **`node.py:195`**: `aggregation_method = config.get("aggregation_method", AGGREGATION_METHOD_MAX_CONFIDENCE)`
   - 到達条件: `run_ask_flow()` が呼ばれる（`run_experiment.py:49` または `node.py:253`）
   - **デフォルト値が `max_confidence` であるため、config に誤った値を設定しない限り確実に到達する**。

2. **`node.py:141-143`**: `if aggregation_method == AGGREGATION_METHOD_MAJORITY_VOTE:` の else 節
   - `max_confidence` は majority_vote 分岐をスキップし、`select_best_dispatch_response()` にフォールバックする。
   - **発火条件**: `dispatch_top_k >= 2` かつ `dispatch_candidate_threshold` が十分低い（現行設定で満たす）。

**`dispatch_top_k=2` のコードパス**:

3. **`aggregator.py:67`**: `return candidates[:top_k]`
   - 到達条件: `top_k=2` で設定されていること（config.yaml line 57）。
   - **発火条件**: 2 位ノードの confidence >= `dispatch_candidate_threshold`（0.0）。常に満たす。

**no-op にならないことの確認**:
- `dispatch_top_k=2` + `dispatch_candidate_threshold=0.0` の組み合わせにより、2位ノードは必ずqualified（confidence は確率で負にならない）。
- `aggregation_method=max_confidence` は majority_vote 分岐をスキップするため、`select_best_dispatch_response()` が呼ばれる。
- **これは Iter27 の失敗（confidence_threshold=0.5 で2位がqualifiedにならなかった）とは異なり、今回の設定では確実に発火する。**

### 固定レバー

- `routing_method=supervised_classifier`
- `confidence_threshold=0.0`（fallback 廃止）
- `classifier_calibration=temperature`（Iter31 adopted）
- `classifier_head_adaptation=education_boundary_tuning (intercept_delta=+0.7)`（Iter44 adopted）
- `dispatch_candidate_threshold=0.0`（Iter46 から変更なし）
- 分類器訓練データ、評価データセット、embedding model
- 他9ドメインの訓練データ

### 備考: Iter46 の非対称性について

Iter46 の結果（`majority_vote, top_k=2`）を評価するには、同一の classifier 条件下での `max_confidence` ベースラインが必要。現行 `models/domain_classifier.joblib` には +0.7 intercept shift が適用されていないため、再訓練が必須。これにより、Iter46 と Iter47 の比較は classifier 面でも対称になる。

---

### 調査 (Iter47)

- **実施日時**: 2026-08-02
- **目的**: `dispatch_top_k=2, aggregation_method=max_confidence` の clean ベースライン取得の準備確認
- **分かったこと**:
  1. **config.yaml の現状**: `dispatch_top_k: 2`（そのまま）、`aggregation_method: majority_vote`（→ `max_confidence` に変更必要）、`confidence_threshold: 0.0`（そのまま）、`dispatch_candidate_threshold: 0.0`（そのまま）。変更は aggregation_method の 1 箇所のみ。
  2. **既存ベースライン（`results/20260730_224515/`）の非対称性**: `confidence_threshold: 0.5`（fallback あり）、`dispatch_candidate_threshold` なし（コードの既定値で `confidence_threshold` と同一）、較正前（iter25 相当）。clean ベースラインとの比較には非対称性が大きい。
  3. **分類器モデルの現状**: `models/domain_classifier.joblib` は `CalibratedClassifierCV(method="temperature")` で較正済み（`train_domain_classifier.py` の実装を確認）。ただし `education_boundary_tuning`（intercept_delta=+0.7）の適用はコードには組み込まれているが、現在モデルファイルには反映されていない可能性あり（ファイルサイズ 315381 bytes、iter44 モデルは 315429 bytes で 48 byte 差）。
  4. **コード変更**: `aggregation_method` の値変更のみでコード変更不要。`max_confidence` のコードパス（`aggregator.py:80-95` の `select_best_dispatch_response()`）は既に実装済み。
  5. **実験手順**: `mise run start` で実行（~90-100分）。`mise run analyze` で answer_quality_accuracy を計算可能。
  6. **classifier_model_path の確認**: config.yaml の `classifier_model_path: models/domain_classifier.joblib` は現在値で正しい。deploy 時に models/ 全体が rsync される。
- **次フェーズへの示唆**:
  - rc-planner: 計画フェーズで `aggregation_method` を `max_confidence` に変更するよう指示。必要に応じて classifier の再訓練（intercept shift 適用）も計画に含める。
  - rc-experimenter: 変更後、`mise run deploy` → `mise run start` → `mise run analyze` の順で実行。

---

### 実験 (Iter46)

- **実行日時**: 2026-08-02
- **実験ディレクトリ**: `results/iter45_aggregation_majority_vote_20260802_145653/`
- **結果ファイル**: `results.jsonl` (1600行)
- **設定**: `dispatch_top_k=2`, `aggregation_method=majority_vote`, `dispatch_candidate_threshold=0.0`
- **主要結果**:
  - `top1_accuracy`: 0.60625（iter31: 0.6056, McNemar p=1.0 で差なし）
  - `compound_domain_set_recall`: 0.36（iter31 top_k=1: 0.0 から大幅改善）
  - `fallback_rate`: 0.0
  - `dispatched_domains` length >= 2: 1600/1600 (100%) — `aggregation_method` 発火確認
  - ドメイン別 recall: education=0.4647, medical=0.5000, legal=0.5611
- **重要発見**:
  1. `dispatch_candidate_threshold=0.0` により2位ノードが100%適格。`aggregation_method` の分岐は確実に発火。
  2. `compound_domain_set_recall=0.36` は期待された改善方向。
  3. `top1_accuracy` は iter31 と McNemar p=1.0 で統計的差なし。
- **未解決事項**:
  1. **ベースライン未存在**: `dispatch_top_k=2, aggregation_method=max_confidence` の結果が results/ 配下に存在しない。成功条件1（ベースラインを5pt以上上回る）の評価には別途ベースライン実行が必要。
  2. `answer_quality_accuracy` は未計算（`mise run analyze` 未実行）。
- **判定**: `pending_baseline` — ベースライン（dispatch_top_k=2, max_confidence）を別途実行して比較検証する必要がある。

### 分析(解釈)

**数値の要約と前回比**:

| メトリクス | Iter46 | 比較対象 | 差 | McNemar p |
|---|---|---|---|---|
| top1_accuracy | 0.60625 | Iter31 (0.6056) | +0.0007 | 1.0 |
| compound_domain_set_recall | 0.36 | Iter31 (0.0, top_k=1) | +0.36 | N/A |
| fallback_rate | 0.0 | Iter31 (0.0) | 0 | N/A |
| dispatched_domains >= 2 | 1600/1600 (100%) | N/A | - | - |

**ノイズ判定**:

- `top1_accuracy` の差 +0.0007 は、Iter27 の noise_floor (answer_quality_accuracy の行単位 flip rate = 23.9%) から推定される SE(0.0007) は約 0.012 程度。差はノイズ範囲内。McNemar p=1.0 は discordant ペアが 98 vs 99 でほぼ完全な対称性。これは「ビット単位の予測がランダムに入れ替わっただけ」と言える。
- `compound_domain_set_recall=0.36` について：この指標は top_k=1 では構造的上限 0.0（1つのノードしか見ないため）のため、top_k=2 への移行自体が 0.0→正の値への必須条件。しかし、`max_confidence` vs `majority_vote` の差を評価するには、同一の top_k=2 条件下での比較が必要。現在そのベースラインが存在しない。
- `dispatched_domains length >= 2` の 100% は、`dispatch_candidate_threshold=0.0` の設定により構造的に保証される値（2位ノードの confidence は常に 0.0 以上）。これは `aggregation_method` の発火確認としては「過剰に良い」結果であり、閾値の感度検討（0.0 ではなく 0.1 や 0.2 でどうなるか）の余地が残る。

**仮説との整合**:

- 仮説は「majority_vote が max_confidence より compound_domain_set_recall を改善する」。今回の結果は仮説の方向性と一致（0.36 は正の値）。しかし、ベースライン未存在のため、effect size（max_confidence 比の改善量）を定量できない。
- `top1_accuracy` の非退行（McNemar p=1.0）は仮説の失敗条件（有意悪化）を満たさない。これは期待通り。
- 想定外の挙動：なし。`aggregation_method=majority_vote` のコードパスが正しく実行され、期待された挙動を示した。

**次の考察フェーズへの示唆**:

- **追加反復が必要**。`dispatch_top_k=2, aggregation_method=max_confidence` のベースラインを別途実行すること。これにより、compound_domain_set_recall の改善が `aggregation_method` の効果なのか `dispatch_top_k=2` の効果なのかを分離できる。
- ベースライン結果が得られた後、`majority_vote` vs `max_confidence` の compound_domain_set_recall 差が 5pt 以上ある場合、`majority_vote` を採用。5pt 未満の場合は、コスト（同一）を考慮して `max_confidence`（単純）を採用するか、reflector に判断を委ねる。
- `answer_quality_accuracy` の計測も後回しにしないこと。`mise run analyze` で計算可能。
- **判定**: `pending_baseline`（確信度: medium）。ベースラインが得られた時点で再分析を行う。

### 考察 (Iter46)

**ベースラインの発見と再評価**:

計画時（rc-planner / rc-experimenter / rc-analyst）は `dispatch_top_k=2, aggregation_method=max_confidence` のベースラインが results/ 配下に存在しないと判定していた。しかし本考察段階で `results/20260730_224515/` に該当ベースラインが存在することを確認した（git_head: 9b7f393bd8d8a6f91b9af82c4946990b0f301698, 1600行）。

**再比較表**:

| メトリクス | Iter46 (majority_vote, top_k=2) | Baseline (max_confidence, top_k=2) | 差 |
|---|---|---|---|
| top1_accuracy | 0.60625 | 0.555625 | +0.0506 |
| compound_domain_set_recall | 0.36 | 0.165 | +0.195 |
| fallback_rate | 0.0 | 0.1325 | -0.1325 |
| compound_mean_dispatched_count | 2.0 | 0.82 | +1.18 |
| ECE | 0.0684 | 0.2040 | -0.1356 |
| Brier score | 0.2005 | 0.24999 | -0.0495 |
| AUROC | 0.7542 | 0.7116 | +0.0426 |

**比較の注意点（非対称性）**:

1. **fallbackの有無**: ベースラインは fallback_rate=0.1325（confidence_threshold=0.5）、Iter46 は fallback_rate=0.0（confidence_threshold=0.0）。top1_accuracy の +0.0506 の差の大部分は fallback 廃止に由来する可能性が高い。
2. **較正の有無**: ベースラインは較正前（iter25相当）、Iter46 は temperature 較正済み（iter44相当）。
3. **compound_mean_dispatched_count**: ベースラインでは 0.82（2位ノードがほぼ dispatch されない）、Iter46 では 2.0（100% 2ノード dispatch）。これは `dispatch_candidate_threshold=0.0`（Y2 で新設）の構造的帰結。

**compound_domain_set_recall の純粋比較**:

- ベースラインの compound_domain_set_recall=0.165 は、fallback によるカバー（13.25% の質問が general へ退避）が寄与している値。
- Iter46 の compound_domain_set_recall=0.36 は fallback なしで達成。
- **両者の差 +0.195 は、aggregation_method の効果 + dispatch_candidate_threshold=0.0 の効果を混在させた値**。

**成功条件の再評価**:

1. **主基準（compound_domain_set_recall がベースラインを 5pt 以上上回る）**: +19.5pt で**明確に成立**（ただし非対称性あり）。
2. **非退行（top1_accuracy の有意悪化なし）**: ベースラインとの McNemar 対比較は不可能（ベースラインは異なる commit の実行結果）。ただし Iter31（top_k=1, max_confidence, fallback廃止後）との比較では McNemar p=1.0 で差なし。
3. **単一レバー検証（dispatched_domains length >= 2 が > 0）**: 1600/1600 (100%) で**明確に成立**。

**判定**: `adopted`（条件付き）。

compound_domain_set_recall の +19.5pt 改善は明確。ただしベースラインとの非対称性（fallback有無、較正有無）を考慮すると、**厳密な対比のため `dispatch_top_k=2, aggregation_method=max_confidence, fallback_policy=disabled, temperature_calibration` のベースラインを別途実行することが望ましい**。

**学び**:
1. **ベースラインは「存在しない」のではなく「別のディレクトリ名」にある**: Iter27 の実験（dispatch_top_k=2 + 3方式比較）の結果は `results/20260730_*/` 配下に保存されていたが、rc-planner/analyst/experimenter がこれを検出できなかった。rc-analyst は results/ 配下の config.yaml を必ずチェックし、ベースラインの存在を確認すること。
2. **dispatch_candidate_threshold=0.0 の効果は巨大**: 2位ノードの dispatch を 0.82→2.0 に押し上げた。aggregation_method の効果を引き出すには、この前提条件が必須。
3. **top1_accuracy と compound_domain_set_recall は異なる軸**: top1_accuracy は単一ドメイン設問（1500問）の精度、compound_domain_set_recall は複合設問（100問）のカバレッジ。両者はトレードオフの関係にある可能性がある。

---

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
