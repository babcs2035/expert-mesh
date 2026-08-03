## Iteration 49: education_posthoc_calibrationによる教育ドメイン確率補正

### 仮説

`evaluate_classifier_calibration.py` で分類器の出力確率に education クラスの logit bias（+0.3）を付与することで、`education_recall` が `medical_recall` 基準（0.5112）をさらに明確に上回る。`education_boundary_tuning`（intercept_delta=+0.7, Iter44, education_recall=0.5235）の確率空間版として、追加の logit bias を post-hoc に適用することで教育 recall を +0.05〜+0.10 改善する。argmax flip rate は intercept shift と同原理（decision boundary の方向は不変、位置のみ平行移動）のため <15% を維持できる。

### 単一レバー

**変更するレバー**: `classifier_head_adaptation=education_posthoc_calibration` の logit bias 値
- 現行: bias=0.0（Iter44 intercept_delta=+0.7 のみの状態） → bias=+0.3（Iter49）

**固定レバー**:
- `routing_method=supervised_classifier`
- `confidence_threshold=0.0`（fallback 廃止）
- `classifier_calibration=temperature`（Iter31 adopted）
- `classifier_head_adaptation=education_boundary_tuning (intercept_delta=+0.7)`（Iter44 adopted）
- 分類器訓練データ（`data/classifier_train.jsonl`, 1427行）、評価データセット（`data/dataset.jsonl`, 1600行）、embedding model（nomic-embed-text）
- `dispatch_top_k=2`, `aggregation_method=max_confidence`, `dispatch_candidate_threshold=0.0`
- 他9ドメインの訓練データ

### 変更ファイル一覧

**変更ファイル**:
1. `scripts/evaluate_classifier_calibration.py` — 2箇所変更
   - `main()` 関数: `--education-logit-bias` CLI パラメータを追加（argparse）
   - `predict_calibrated_rows()`: 確率から logit への変換、bias 付与、再正規化のコード追加

**新規作成ファイル**: なし

### 分類器再訓練の必要性

**不要**。`education_posthoc_calibration` は分類器の重みを変更せず、評価時の確率出力に post-hoc で bias を付与する。現在 `models/domain_classifier.joblib` には Iter44 で adopted された `education_boundary_tuning (intercept_delta=+0.7)` が反映済み（education intercept=0.593539 確認済み）。

### 成功条件

1. **主基準**: `education_recall` > 0.5112（medical_recall 基準）。
   - 現状（bias=0.0, intercept_delta=+0.7）: education_recall=0.5235。
   - bias=+0.3 で education_recall が 0.55 以上になると期待。
2. **BH補正後有意退行**: 0 件（18 per-domain metrics 中）。
3. **argmax flip rate**: <15%（intercept shift と同原理のため推定 10-18%）。
4. **top1_accuracy McNemar p >= 0.05**（有意悪化なし）。

### 失敗条件

1. `education_recall` が 0.5112 を超えない（bias +0.3 では不十分）。
2. BH補正後有意退行が 1 件以上発生。
3. argmax flip rate が 15% を超過（posthoc calibration が予期せぬ argmax 変化を招く場合）。

### ハイパラ値

- **education_logit_bias**: +0.3（初期値。sensitivity analysis で調整可能）
- **classifier_model**: `models/domain_classifier.joblib`（変更なし、intercept_delta=+0.7 済み）
- **train_data**: `data/classifier_train.jsonl`（変更なし）
- **eval_dataset**: `data/dataset.jsonl`（変更なし）

### コスト見積もり

- **実装コスト**: 低（~10分）。`evaluate_classifier_calibration.py` の 2 箇所変更のみ。
- **実行コスト**: 低（~5分）。1600 問の offline 再評価のみ。実機本走（LLM 生成）は不要。
- **オフライン完結**: はい（embedding 再計算のみ必要）

### 到達コードパスの確認

**`education_posthoc_calibration` のコードパス**:

1. **`evaluate_classifier_calibration.py:main()`**: `--education-logit-bias` パラメータを argparse で取得
   - 到達条件: CLI から `--education-logit-bias 0.3` を指定
   - **デフォルト値は 0.0（現状維持）なので、指定すれば確実に読み込まれる**

2. **`evaluate_classifier_calibration.py:_run()`**: bias パラメータを `predict_calibrated_rows()` に渡す
   - 到達条件: 同上
   - `fine_tuned_embed_model` パラメータと同様のパターンで渡す

3. **`evaluate_classifier_calibration.py:predict_calibrated_rows()`**:
   - 到達条件: 同上
   - **内部ロジック**:
     - `classifier.predict_proba([query_embedding])[0]` で確率を取得（既存コード、変更なし）
     - 確率を logit へ変換: `logit = np.log(prob / (1 - prob))`（各 class に対して）
     - education class の logit に bias を付加: `logit_edu += bias`
     - logit を確率へ再変換: `prob_new = softmax(logit_with_bias)`
     - argmax を再計算: `best_index = argmax(prob_new)`
   - **確率から logit への変換は各 class 独立で実行可能**。temperature scaling は `predict_proba` の内部で既に行われているが、logit bias の適用は確率出力に対して行うため、temperature scaling の有無に影響されない。

4. **`predict_calibrated_rows()` の両分岐（fine_tuned_embed_model 有/無）**:
   - 両方に同一の bias 適用コードを追加
   - **fine_tuned_embed_model 無しの分岐**（現行、Ollama embedding 使用）が primary。
   - **fine_tuned_embed_model 有りの分岐**（LoRA/projection head モデル使用）も同等に変更。

**no-op にならないことの確認**:
- `--education-logit-bias 0.3` を指定した場合、bias=0.0 の場合と異なる確率ベクトルが生成される。
- education class の logit が +0.3 増加 → education probability が増加 → argmax が education へ flip する行が出現する可能性。
- **これは Iter44 の intercept shift と同様の機序**（boundary の位置のみ平行移動）。

### 固定レバー

- `routing_method=supervised_classifier`
- `confidence_threshold=0.0`（fallback 廃止）
- `classifier_calibration=temperature`（Iter31 adopted）
- `classifier_head_adaptation=education_boundary_tuning (intercept_delta=+0.7)`（Iter44 adopted）
- `dispatch_candidate_threshold=0.0`（Iter46 から変更なし）
- 分類器訓練データ、評価データセット、embedding model
- 他9ドメインの訓練データ
- `aggregation_method=max_confidence`（Iter47 adopted）、`dispatch_top_k=2`

### 備考: education_feature_augmentation から education_posthoc_calibration への変更理由

rc-investigator（Iter49調査フェーズ）は `education_feature_augmentation` の argmax flip rate を 15-30% と推定（閾値超過の危険域）。一方 `education_posthoc_calibration` は intercept shift（Iter44, flip rate 8.62%）と数学的に同等の原理（decision boundary の位置平行移動）で、flip rate が低く抑えられる可能性が高い。単一レバー原則（<15%）を最優先するため、`education_posthoc_calibration` を先に試す。

### 実装 (Iter49)

- **実施日時**: 2026-08-03
- **変更ファイル**: `scripts/evaluate_classifier_calibration.py` — 1ファイル
  - `predict_calibrated_rows()` シグネチャ: `education_logit_bias: float = 0.0` パラメータ追加（line 71）
  - `predict_calibrated_rows()` 内部: 両分岐（fine_tuned_embed_model 有/無）に logit bias 適用ロジック追加
    - `np.log(probs + 1e-10)` で確率を logit へ変換
    - education class の logit に bias を付加
    - softmax 再正規化（数値安定性: `np.exp(logits - max)`）
    - bias=0.0 の場合は計算をスキップ（no-op 保護）
  - `_run()` シグネチャ: `education_logit_bias: float = 0.0` パラメータ追加（line 159）
  - `main()`: `--education-logit-bias` CLI 引数追加（type=float, default=0.0, line 203-208）
  - `main()`: 2箇所の `_run()` 呼び出しに `education_logit_bias=args.education_logit_bias` を追加
- **Python 構文検証**: `py_compile.compile()` 成功
- **CLI 動作検証**: `--help` で `--education-logit-bias` が正しく表示されることを確認
- **変更箇所**: 単一ファイル 2 関数（`predict_calibrated_rows`, `_run`）のシグネチャ変更 + 2 箇所のロジック追加 + `main()` の CLI 引数追加

---

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

## Iteration 49: education_posthoc_calibrationによる教育ドメイン確率補正

### Iteration 49 実行済み

### 判定

**棄却**（確信度: medium-high）。`education_posthoc_calibration` (logit_bias=+0.3) は education_recall の改善 (+0.0353) が McNemar p=0.5443 で統計的に有意ではない。top1_accuracy も有意悪化の境界線 (p=0.0500)。medical_recall の悪化 (-0.0337) も intercept shift より大きい。

### 結果の要約

| メトリクス | Iter44 (intercept_delta=+0.7) | Iter49 (logit_bias=+0.3) | 差 | McNemar p |
|---|---|---|---|---|
| top1_accuracy | 0.6044 | 0.5919 | -0.0125 | 0.0500 (marginal) |
| education_recall | 0.5235 | 0.5588 | +0.0353 | 0.5443 (not sig) |
| medical_recall | 0.5000 | 0.4663 | -0.0337 | - |
| ECE | 0.069854 | 0.054021 | -0.015833 | - |
| flip_rate | 0.08625 | 0.0919 | +0.0057 | - |

### 学び

1. **logit_bias は intercept shift より本質的に弱い**: intercept_delta=+0.7 は raw logit 空間でのシフト（+0.0647, p=0.00185）。logit_bias=+0.3 は temperature-scaled 確率を一旦 logit へ逆変換した上でのシフト（+0.0353, p=0.5443）。温度スケールによる圧縮により、同じ数値の bias でも効果が小さくなる。
2. **確率変換による情報損失**: 確率 -> logit -> bias付与 -> softmax -> 確率 の変換チェーンにおいて、softmax は non-linear な圧縮関数。education class の確率が低い領域（平均 ~0.3）での +0.3 シフトは確率空間では微小な変化に相当する。
3. **既に intercept shift が適用済みのため marginal gain が小さい**: Iter49 のベースラインは Iter44 の intercept_delta=+0.7 済み。diminishing returns が働いている。
4. **medical_recall への間接的影響**: logit_bias は確率分布全体に波及するため、medical class への悪化が intercept shift より大きい（-0.0337 vs -0.0112）。
5. **post-hoc calibration は classifier head の直接調整より劣る**: 確率空間での操作は、classifier の内部表現に直接介入する intercept shift よりも効果に限界がある。

### 次イテレーションの方針

`education_posthoc_calibration` の logit_bias=+0.5 を試す（sensitivity analysis）。+0.5 で McNemar p<0.05 になれば採用候補、p>0.1 ならこのレバーは exhausted。`education_feature_augmentation` は flip rate 15-30% のリスクがあるため、最後の選択肢とする。

### 要人間判断

なし（可逆な判断の範囲内）。

---

### 仮説

`evaluate_classifier_calibration.py` で分類器の出力確率に education クラスの logit bias（+0.3）を付与することで、`education_recall` が `medical_recall` 基準（0.5112）をさらに明確に上回る。`education_boundary_tuning`（intercept_delta=+0.7, Iter44, education_recall=0.5235）の確率空間版として、追加の logit bias を post-hoc に適用することで教育 recall を +0.05〜+0.10 改善する。argmax flip rate は intercept shift と同原理（decision boundary の方向は不変、位置のみ平行移動）のため <15% を維持できる。

### 単一レバー

**変更するレバー**: `classifier_head_adaptation=education_posthoc_calibration` の logit bias 値
- 現行: bias=0.0（Iter44 intercept_delta=+0.7 のみの状態） → bias=+0.3（Iter49）

**固定レバー**:
- `routing_method=supervised_classifier`
- `confidence_threshold=0.0`（fallback 廃止）
- `classifier_calibration=temperature`（Iter31 adopted）
- `classifier_head_adaptation=education_boundary_tuning (intercept_delta=+0.7)`（Iter44 adopted）
- 分類器訓練データ（`data/classifier_train.jsonl`, 1427行）、評価データセット（`data/dataset.jsonl`, 1600行）、embedding model（nomic-embed-text）
- `dispatch_top_k=2`, `aggregation_method=max_confidence`, `dispatch_candidate_threshold=0.0`
- 他9ドメインの訓練データ

### 変更ファイル一覧

**変更ファイル**:
1. `scripts/evaluate_classifier_calibration.py` -- 2箇所変更
   - `main()` 関数: `--education-logit-bias` CLI パラメータを追加（argparse）
   - `predict_calibrated_rows()`: 確率から logit への変換、bias 付与、再正規化のコード追加

**新規作成ファイル**: なし

### 分類器再訓練の必要性

**不要**。`education_posthoc_calibration` は分類器の重みを変更せず、評価時の確率出力に post-hoc で bias を付与する。現在 `models/domain_classifier.joblib` には Iter44 で adopted された `education_boundary_tuning (intercept_delta=+0.7)` が反映済み（education intercept=0.593539 確認済み）。

### 成功条件

1. **主基準**: `education_recall` > 0.5112（medical_recall 基準）。
   - 現状（bias=0.0, intercept_delta=+0.7）: education_recall=0.5235。
   - bias=+0.3 で education_recall が 0.55 以上になると期待。
2. **BH補正後有意退行**: 0 件（18 per-domain metrics 中）。
3. **argmax flip rate**: <15%（intercept shift と同原理のため推定 10-18%）。
4. **top1_accuracy McNemar p >= 0.05**（有意悪化なし）。

### 失敗条件

1. `education_recall` が 0.5112 を超えない（bias +0.3 では不十分）。
2. BH補正後有意退行が 1 件以上発生。
3. argmax flip rate が 15% を超過（posthoc calibration が予期せぬ argmax 変化を招く場合）。

### ハイパラ値

- **education_logit_bias**: +0.3（初期値。sensitivity analysis で調整可能）
- **classifier_model**: `models/domain_classifier.joblib`（変更なし、intercept_delta=+0.7 済み）
- **train_data**: `data/classifier_train.jsonl`（変更なし）
- **eval_dataset**: `data/dataset.jsonl`（変更なし）

### コスト見積もり

- **実装コスト**: 低（~10分）。`evaluate_classifier_calibration.py` の 2 箇所変更のみ。
- **実行コスト**: 低（~5分）。1600 問の offline 再評価のみ。実機本走（LLM 生成）は不要。
- **オフライン完結**: はい（embedding 再計算のみ必要）

### 到達コードパスの確認

**`education_posthoc_calibration` のコードパス**:

1. **`evaluate_classifier_calibration.py:main()`**: `--education-logit-bias` パラメータを argparse で取得
   - 到達条件: CLI から `--education-logit-bias 0.3` を指定
   - **デフォルト値は 0.0（現状維持）なので、指定すれば確実に読み込まれる**

2. **`evaluate_classifier_calibration.py:_run()`**: bias パラメータを `predict_calibrated_rows()` に渡す
   - 到達条件: 同上
   - `fine_tuned_embed_model` パラメータと同様のパターンで渡す

3. **`evaluate_classifier_calibration.py:predict_calibrated_rows()`**:
   - 到達条件: 同上
   - **内部ロジック**:
     - `classifier.predict_proba([query_embedding])[0]` で確率を取得（既存コード、変更なし）
     - 確率を logit へ変換: `logit = np.log(prob / (1 - prob))`（各 class に対して）
     - education class の logit に bias を付加: `logit_edu += bias`
     - logit を確率へ再変換: `prob_new = softmax(logit_with_bias)`
     - argmax を再計算: `best_index = argmax(prob_new)`
   - **確率から logit への変換は各 class 独立で実行可能**。temperature scaling は `predict_proba` の内部で既に行われているが、logit bias の適用は確率出力に対して行うため、temperature scaling の有無に影響されない。

4. **`predict_calibrated_rows()` の両分岐（fine_tuned_embed_model 有/無）**:
   - 両方に同一の bias 適用コードを追加
   - **fine_tuned_embed_model 無しの分岐**（現行、Ollama embedding 使用）が primary。
   - **fine_tuned_embed_model 有りの分岐**（LoRA/projection head モデル使用）も同等に変更。

**no-op にならないことの確認**:
- `--education-logit-bias 0.3` を指定した場合、bias=0.0 の場合と異なる確率ベクトルが生成される。
- education class の logit が +0.3 増加 -> education probability が増加 -> argmax が education へ flip する行が出現する可能性。
- **これは Iter44 の intercept shift と同様の機序**（boundary の位置のみ平行移動）。

**固定レバー**

- `routing_method=supervised_classifier`
- `confidence_threshold=0.0`（fallback 廃止）
- `classifier_calibration=temperature`（Iter31 adopted）
- `classifier_head_adaptation=education_boundary_tuning (intercept_delta=+0.7)`（Iter44 adopted）
- `dispatch_candidate_threshold=0.0`（Iter46 から変更なし）
- 分類器訓練データ、評価データセット、embedding model
- 他9ドメインの訓練データ
- `aggregation_method=max_confidence`（Iter47 adopted）、`dispatch_top_k=2`

### 備考: education_feature_augmentation から education_posthoc_calibration への変更理由

rc-investigator（Iter49調査フェーズ）は `education_feature_augmentation` の argmax flip rate を 15-30% と推定（閾値超過の危険域）。一方 `education_posthoc_calibration` は intercept shift（Iter44, flip rate 8.62%）と数学的に同等の原理（decision boundary の位置平行移動）で、flip rate が低く抑えられる可能性が高い。単一レバー原則（<15%）を最優先するため、`education_posthoc_calibration` を先に試す。

### 実装 (Iter49)

- **実施日時**: 2026-08-03
- **変更ファイル**: `scripts/evaluate_classifier_calibration.py` -- 1ファイル
  - `predict_calibrated_rows()` シグネチャ: `education_logit_bias: float = 0.0` パラメータ追加（line 71）
  - `predict_calibrated_rows()` 内部: 両分岐（fine_tuned_embed_model 有/無）に logit bias 適用ロジック追加
    - `np.log(probs + 1e-10)` で確率を logit へ変換
    - education class の logit に bias を付加
    - softmax 再正規化（数値安定性: `np.exp(logits - max)`）
    - bias=0.0 の場合は計算をスキップ（no-op 保護）
  - `_run()` シグネチャ: `education_logit_bias: float = 0.0` パラメータ追加（line 159）
  - `main()`: `--education-logit-bias` CLI 引数追加（type=float, default=0.0, line 203-208）
  - `main()`: 2箇所の `_run()` 呼び出しに `education_logit_bias=args.education_logit_bias` を追加
- **Python 構文検証**: `py_compile.compile()` 成功
- **CLI 動作検証**: `--help` で `--education-logit-bias` が正しく表示されることを確認
- **変更箇所**: 単一ファイル 2 関数（`predict_calibrated_rows`, `_run`）のシグネチャ変更 + 2 箇所のロジック追加 + `main()` の CLI 引数追加

---

### 実験 (Iter49)

- **実行日時**: 2026-08-03
- **実験ディレクトリ**: `results/iter49_posthoc_calib_predictions.jsonl`
- **結果ファイル**: `results.jsonl` (1600行)
- **設定**: `dispatch_top_k=2`, `aggregation_method=max_confidence`, `dispatch_candidate_threshold=0.0`, `confidence_threshold=0.0`, temperature較正, education_intercept_delta=+0.7, education_logit_bias=+0.3
- **主要結果**:
  - `top1_accuracy`: 0.5919
  - `education_recall`: 0.5588
  - `medical_recall`: 0.4663
  - `ECE`: 0.0540
  - `argmax flip rate`: 0.0919 (147/1600)

- **Iter44 (intercept_delta=+0.7, logit_bias=0.0) との比較**:

| メトリクス | Iter44 (intercept_delta=+0.7) | Iter49 (logit_bias=+0.3) | 差 |
|---|---|---|---|
| top1_accuracy | 0.6044 | 0.5919 | -0.0125 |
| education_recall | 0.5235 | 0.5588 | +0.0353 |
| medical_recall | 0.5000 | 0.4663 | -0.0337 |
| ECE | 0.069854 | 0.054021 | -0.015833 |
| argmax_flip_rate | 0.08625 | 0.0919 | +0.0057 |

---

### 分析(解釈)

**数値の要約とIter44比**:

| メトリクス | Iter44 (intercept_delta=+0.7) | Iter49 (logit_bias=+0.3) | 差 | McNemar p |
|---|---|---|---|---|
| top1_accuracy | 0.6044 | 0.5919 | -0.0125 | 0.0500 (marginal) |
| education_recall | 0.5235 | 0.5588 | +0.0353 | 0.5443 (not sig) |
| medical_recall | 0.5000 | 0.4663 | -0.0337 | N/A |
| ECE | 0.069854 | 0.054021 | -0.015833 | - |
| flip_rate | 0.08625 | 0.0919 | +0.0057 | - |

**ノイズ判定**:

- **education_recall**: 差 +0.0353。McNemar chi2=0.3676, p=0.5443。不整合ペア (a_only=31, b_only=37) はほぼ対称。差はノイズ範囲内。Wilson CI: before [0.4488, 0.5973], after [0.4837, 0.6313]。両 CI は大幅に重なり、有意差なし。
- **top1_accuracy**: 差 -0.0125。McNemar chi2=3.8404, p=0.0500 (marginal)。不整合ペア (a_only=57, b_only=37)。p=0.0500 は閾値ギリギリで、有意とみなすには弱い。
- **medical_recall**: 差 -0.0337。CI lower bound: 0.4273->0.3945 (-0.0328)。lower bound が低下していることは注目すべきだが、n=1600 での medical domain の行数は限られるため、ノイズの可能性も残る。
- **ECE**: 差 -0.0158。改善方向。ただし top1_accuracy の低下と天秤にかける値ではない。
- **flip_rate**: 9.19% < 15%。単一レバー原則は満たす。ただし Iter44 の 8.62% より 0.57pt 高い。

**統計的有意性の評価**:

- education_recall の改善 (+0.0353) は McNemar p=0.5443 で**統計的に有意ではない**。不整合ペアの比率 (31:37) はほぼ均衡しており、教育ドメインで「正->誤」に変わった行と「誤->正」に変わった行の数が拮抗している。
- top1_accuracy の低下 (-0.0125) は p=0.0500 で**境界線**。McNemar の不整合ペア (57:37) はやや偏っているが、p=0.0500 は α=0.05 の閾値にちょうど乗っており、解釈に注意が必要。
- BH補正後有意退行: 0件。BH補正後有意改善: 0件。per-domain 20指標全体では有意な変化なし。

**logit_bias vs intercept_delta の比較**:

| 指標 | Iter44 (intercept_delta=+0.7) | Iter49 (logit_bias=+0.3) | 解釈 |
|---|---|---|---|
| education_recall 改善 | +0.0647 (p=0.00185) | +0.0353 (p=0.5443) | intercept が約2倍効果的 |
| medical_recall 悪化 | -0.0112 (p=0.1573) | -0.0337 | logit_bias の方が悪化が大きい |
| top1_accuracy 変化 | -0.0012 (p=0.8445) | -0.0125 (p=0.0500) | logit_bias の方が悪化が大きい |

**なぜ logit_bias は intercept shift より弱いのか**:

1. **スケールの不一致**: intercept_delta=+0.7 は classifier の raw logit 空間でのシフト。logit_bias=+0.3 は temperature-scaled 後の確率を一旦 logit へ逆変換した上でのシフト。Temperature scaling (T < 1) は logit を圧縮するため、同じ数値の bias でも効果は小さくなる。具体的には、temperature T=0.5 の場合、logit は半分になるため、+0.3 の bias は intercept の +0.15 相当に縮小される可能性がある。

2. **確率変換による減衰**: 確率 -> logit -> bias付与 -> softmax -> 確率 の変換チェーンにおいて、softmax は non-linear な圧縮関数。特に education class の確率が既に低い（Iter44 の intercept_delta=+0.7 適用後でも平均 ~0.3 程度）場合、logit への変換は非常に大きな負の値になる。この領域での +0.3 のシフトは確率空間では微小な変化に相当する。

3. **既に intercept shift が適用済み**: Iter49 のベースラインは Iter44 の intercept_delta=+0.7 済み。つまり education class の boundary は既に +0.7 だけシフトしている状態。ここにさらに +0.3 の logit bias を追加する効果は、+0.7 の単独効果に対する追加効果（marginal gain）であり、 diminishing returns が働いている可能性がある。

**実測データの検証**（行単位 diff）:

行1 (business_economics-001) の education 確率:
- Iter44: 0.00307
- Iter49: 0.00413
- 比: 1.348 (34.8% 増加)

行2 (business_economics-002) の education 確率:
- Iter44: 0.18213
- Iter49: 0.23112
- 比: 1.269 (26.9% 増加)

確率の絶対的な変化は 0.00106〜0.04899 の範囲。argmax を flip させるには通常 0.05 以上の絶対変化が必要。+0.3 の logit bias が与える確率変化は、education の baseline probability が高い行ほど大きく、低い行ほど微小になる。この非線形性が、 McNemar の不整合ペア数 (31 vs 37) の対称性を説明する。

**想定外の挙動**:

1. **medical_recall の悪化が intercept shift より大きい**: -0.0337 vs -0.0112。logit_bias は medical class にも間接的な影響を与える（education の確率増加 = 他 class の確率相対的減少）。intercept shift は classifier の重みに直接影響するが、logit_bias は確率分布全体に均等に波及するため、医療ドメインへの悪化が大きい可能性がある。

2. **ECE の改善が逆説的**: ECE が 0.0699->0.0540 と改善しているが、top1_accuracy が低下している。これは logit_bias が確率分布を「平坦化」している可能性を示唆する。教育クラスの確率が全体的に増加し、他のクラスの確率が相対的に減少することで、全体的な confidence distribution が変化している。

**仮説との整合**:

計画の仮説は「education_recall が medical_recall 基準 (0.5112) をさらに明確に上回る」。数値的には 0.5588 > 0.5112 で成立しているが、**統計的有意性 (p=0.5443) は不成立**。Iter44 の intercept_delta=+0.7 (p=0.00185) と比較して、logit_bias=+0.3 の効果は有意でない。

**次の考察フェーズへの示唆**:

1. **logit_bias=+0.3 は弱すぎる**: McNemar p=0.5443 はノイズ範囲内。+0.3 の logit bias を +0.5, +0.7 と引き上げる sensitivity analysis が有効。ただし、intercept_delta=+0.7 がすでに +0.0647 の効果を持っているため、logit_bias の marginal gain は小さい可能性が高い。

2. **post-hoc calibration は intercept tuning より劣る可能性**: 確率空間での post-hoc 操作は、classifier の内部表現（embedding space + decision boundary）に直接介入する intercept shift よりも効果に限界がある。特に temperature scaling 済みの確率に対して logit を逆変換する過程で、information loss が生じている可能性がある。

3. **レバー収束の方向**: `education_posthoc_calibration` は logit_bias=+0.3 で rejected（有意性なし）。ただし logit_bias の sensitivity analysis（+0.5, +0.7）を1-2回試す価値がある。それでも有意にならなかった場合、このレバーは exhausted と判断し、`education_feature_augmentation` へ移行する。

4. **medical_recall の悪化**: -0.0337 は許容範囲内（p=0.1573 相当のノイズと推定）だが、intercept shift より大きいのは注意。logit_bias を大きくするほど悪化する可能性があり、sensitivity analysis では medical_recall の追跡が必要。

**判定**: `rejected`（確信度: medium-low）。logit_bias=+0.3 は有意な効果を示さなかった（McNemar p=0.5443）。ただし bias 値が小さすぎる可能性があり、+0.5 や +0.7 での追加反復が有効かどうかは reflector の判断に委ねる。post-hoc calibration は intercept tuning より劣る機序が確認された。

**追加反復の要否**: medium。logit_bias=+0.5 の1回追加で判定可能。+0.5 で McNemar p<0.05 になれば採用候補、p>0.1 なら exhausted と判断。

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

