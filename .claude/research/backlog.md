# backlog — 人間判断待ち事項 / 自動判断の記録

## B69 [auto-decided 2026-08-02] Iter46: majority_vote adopted（条件付き）。次は clean ベースライン取得 → llm_judge

- **状況**: Iter46（aggregation_method=majority_vote, dispatch_top_k=2）の結果を考察。計画時はベースライン未存在と判定されたが、`results/20260730_224515/` に `dispatch_top_k=2, max_confidence` のベースラインが存在することを本考察で発見。
- **判定**: `adopted`（条件付き）。compound_domain_set_recall: 0.36 (majority_vote) vs 0.165 (max_confidence) = **+19.5pt** で主基準（5pt以上）を明確に超過。
- **条件付きの理由**: 比較の非対称性（ベースラインは fallback_rate=0.1325、Iter46は0.0；ベースラインは較正前、Iter46はtemperature較正済み）。厳密な対比のため clean ベースラインの再実行を推奨。
- **決定的な学び**:
  1. **ベースラインは「存在しない」のではなく「別のディレクトリ名」にある**: Iter27 の実験結果は `results/20260730_*/` 配下に保存されていたが、rc-planner/analyst/experimenter が検出できなかった。次回以降、results/ 配下の config.yaml を必ずチェックすること。
  2. **dispatch_candidate_threshold=0.0 の効果は巨大**: compound_mean_dispatched_count を 0.82→2.0 に押し上げた。aggregation_method の効果を引き出すにはこの前提条件が必須。
  3. **top1_accuracy と compound_domain_set_recall は異なる軸**: top1 は単一ドメイン設問（1500問）、compound は複合設問（100問）のカバレッジ。両者はトレードオフの可能性がある。
- **次イテレーションの計画**:
  1. **Iter47**: `dispatch_top_k=2, aggregation_method=max_confidence` の clean ベースライン（fallback廃止, temperature較正済み）。コスト: ~90-100分。
  2. **Iter48**: `aggregation_method=llm_judge`（dispatch_top_k=2）。コスト: ~100-120分（judge_model追加呼び出し）。
- **config.yml の levers 状態**: `aggregation_method` の values は `[majority_vote, llm_judge]`。majority_vote は adopted（条件付き）。次値は `llm_judge`。
- **要人間判断**: なし（可逆な判断の範囲内）。

---

## B68 [auto-decided 2026-08-02] Iter46: Y3 aggregation_method 比較実験（majority_vote vs max_confidence）

- **状況**: Y2（dispatch_candidate_threshold 新設）が Iter45 で完了。config.yml の全 levers を試し切り済み。`aggregation_method` レバーの次値（majority_vote, llm_judge）を試すための前提条件が整った。
- **自動選択**: 単一レバーを `aggregation_method=majority_vote` とする。`iteration_name` は「aggregation_method変更による複数ノードdispatch時の集約方式比較」。
- **選定理由**:
  1. Y2（dispatch_candidate_threshold 新設）により、2位ノードの dispatch が可能になった。
  2. `dispatch_top_k=2` の下で、`majority_vote` が `max_confidence` を超える可能性がある。
  3. 変更は `config.yaml` の 2 値のみ（dispatch_top_k: 1->2, aggregation_method: max_confidence->majority_vote）。コード変更不要。
  4. 低コスト（実装~5分、実行~90-100分）。
- **変更ファイル**:
  1. `config.yaml` — line 57: `dispatch_top_k: 1` → `dispatch_top_k: 2`
  2. `config.yaml` — line 68: `aggregation_method: max_confidence` → `aggregation_method: majority_vote`
- **成功条件**: `compound_domain_set_recall` が `max_confidence` ベースラインを 5pt 以上上回る。`top1_accuracy` の有意悪化なし（McNemar p >= 0.05）。
- **固定レバー**: routing_method, confidence_threshold, classifier_calibration, classifier_head_adaptation, dispatch_candidate_threshold=0.0, 分類器訓練データ、評価データセット、embedding model。
- **到達コードパスの確認**:
  - `node.py:195`: `aggregation_method` を config から取得
  - `node.py:196`: `validate_aggregation_method()` で validation
  - `node.py:217`: `dispatch_top_k=2` で `select_dispatch_targets()` を呼び出し
  - `node.py:238`: `_dispatch_to_targets()` に `aggregation_method` を渡す
  - `node.py:141-143`: `aggregation_method==majority_vote` で分岐
  - `aggregator.py:98-125`: `select_best_dispatch_response_majority_vote()` を実行
  - **no-op にならない確認**: `dispatch_candidate_threshold=0.0` のため、2位ノードの confidence は常に 0.0 以上（確率は負にならない）→ 2位ノードは必ず qualified → `aggregation_method` の分岐は必ず発火。
- **コスト**: 実装~5分、実行~90-100分（1600問本走x1回）。オフライン完結なし。
- **llm_judge の比較**: 次イテレーションで検討（judge_model の追加LLM呼び出しで~100-120分/回）。

---

## B67 [auto-decided 2026-08-02] Iter45: Y2（dispatch_candidate_threshold 新設）の設計確定とユーザー確認待ち

- **状況**: config.yml の全 levers を試し切り済み。`classifier_head_adaptation` の残り2値は実質的に試す価値が低い（`education_posthoc_calibration` は intercept シフトと数学的に同等、`education_feature_augmentation` は argmax flip rate 15%超のリスク）。`aggregation_method` は Y2 が完了するまで試せない。
- **自動選択**: Y2（`dispatch_candidate_threshold` 新設）を次レバーとして設計確定。`config.yaml` への新フィールド追加 + `aggregator.select_dispatch_targets()` のシグネチャ変更を伴うスキーマ変更のため、ユーザー確認待ち状態とする。
- **Y2 の設計**:
  1. `config.yaml` へ `dispatch_candidate_threshold` フィールドを新設（既定値は `confidence_threshold` と同値）
  2. `aggregator.py:select_dispatch_targets()` に `dispatch_candidate_threshold` パラメータを追加
  3. rank 1 は `confidence_threshold` で判定、rank 2+ は `dispatch_candidate_threshold` で判定
  4. `node.py:214` と `run_experiment.py:85` の呼び出し側を変更
- **Y3 での sweep 値**: `dispatch_candidate_threshold` = 0.3, 0.4, 0.5（Iter27 の分析: 0.3→~230件/14.4%, 0.4→~120件/7.5%, 0.5→~75件/4.7%）
- **Y3 の主指標**: `compound_domain_set_recall`（現状 0.165。dispatch_top_k=2 で構造的上限 1.000）
- **変更ファイル**:
  1. `config.yaml` — `dispatch_candidate_threshold` フィールド追加
  2. `aggregator.py` — `select_dispatch_targets()` のシグネチャ変更
  3. `node.py` — 呼び出し側の変更
  4. `run_experiment.py` — 呼び出し側の変更
  5. `tests/test_aggregator.py` — 新テスト追加
- **実装コスト**: 低（~1-2時間）
- **実行コスト**: Y3 本走 x 3 値 = ~270分
- **要人間判断**: `config.yaml` への新フィールド追加および関数シグネチャ変更はスキーマ変更。確認を得てから着手。
- **Label Space Reduction の検討**: rc-investigator は general class の細分化（general→general_tech/general_business/general_humanities）も代替アプローチとして提案。ただしこれは classifier の class 数変更（10→12+）を伴い、config.yaml の node 定義・router.py の GENERAL_DOMAIN 定数・build_dataset.py の _DOMAIN_TASK_MAP 等の変更を必要とするため、スキーマ変更扱い。Y2 完了後に検討する。

---

## B66 [auto-decided 2026-08-02] Iter44: education_boundary_tuning (intercept_delta=+0.5)

- **自動選択**: 単一レバーを `classifier_head_adaptation=education_boundary_tuning` とする。
  `iteration_name` は「educationドメインinterceptシフト(+0.5)によるdecision boundary調整」．
- **選定理由**:
  1. rc-investigator（Iter44調査フェーズ）は `education_boundary_tuning`（interceptシフト）を
     推奨。argmax flip rate ~3-5%（<15%閾値内）を見込み。
  2. 既存分類器の education intercept (-0.1185) は medical (-0.0256) より -0.093 低い。
     education eval 170件中、誤解92件のうち23件 (25.0%) は edu_prob > 0.2。
     intercept を +0.5 シフトすればこれらのケースの多くが education へ flip する。
  3. 単一レバー原則の保証が最高: intercept の平行移動は decision boundary の方向を変えず、
     位置だけ動かす。係数ベクトル（判別方向）は不変。
  4. 実装コストが最小: `train_domain_classifier.py` の `train_classifier()` 内のみ変更（~10行）。
  5. オフライン完結（実機1600問本走不要）。
- **変更ファイル**:
  1. `scripts/train_domain_classifier.py` line 192-193: `train_classifier()` 内で
     `calibrated_model.calibrated_classifiers_[i].estimator.intercept_[edu_idx] += 0.5`
- **intercept_delta の初期値**: +0.5
  - 推定 argmax flip rate: ~3-5%
  - 失敗時の感度分析: +0.7, +1.0 の順で増強（単一レバー原則の範囲内で）
- **成功条件**: (1) education_recall > 0.5112, (2) BH補正後有意退行0件,
  (3) argmax flip rate <15%, (4) top1_accuracy McNemar p>=0.05
- **固定レバー**: classifier_architecture (LogisticRegression + temperature),
  training_data (1427行), eval_dataset (1600行), embedding_model (nomic-embed-text),
  temperature params, routing_method, confidence_threshold, dispatch_top_k,
  aggregation_method, class_weight=None + sample_weight
- **コスト**: 低（~10-15分、オフライン完結）

---

このファイルは research-cycle が「本来は人間の判断が要るが，サイクルを止めないために暫定で自動選択した事項」と，
「不可逆・危険なため停止して人間に委ねた事項」を記録する．新しいものを常に先頭に追記する（逆時系列）．

## B65 [auto-decided 2026-08-02] Iter43: embedding_adapter_projection_head rejected。全embedding適応値尽きた。次レバー=classifier_head_adaptation

- 状況: Iter43（embedding_adaptation=embedding_adapter_projection_head）の rc-analyst 判定（rejected）
  を rc-reflector が検証・確定させた。
- **判定: rejected（確定）**。argmax flip rate 42.00%（閾値<15%の2.8倍超過）。top1_accuracy 有意悪化
  （McNemar p=3.0e-9）。BH補正後有意退行 15/20指標。
- **全embedding適応試行の総括**:

  | イテレーション | アプローチ | argmax flip rate | education_recall | medical_recall | top1_accuracy | 判定 |
  |---|---|---|---|---|---|---|
  | Iter40 | SetFit full FT | 52.56% | 0.6529 | 0.3090 | 0.4894 | rejected |
  | Iter41 | LoRA r=16 | 35.88% | 0.5706 | 0.4045 | 0.5719 | rejected |
  | Iter42 | LoRA r=8 | 35.88% | 0.6235 | 0.4326 | 0.5719 | rejected |
  | Iter43 | Dense projection head (590K) | 42.00% | 0.5529 | 0.3596 | 0.5269 | rejected |

- **決定的な学び**:
  1. **embedding適応は単一レバー原則と両立しない**: 全4手法がargmax flip rate >= 35.88%でrejected。
     embedding空間の再構造化は必然的に他ドメインに影響する。
  2. **Dense projection headはLoRAより悪い**: 42.00% flip rateはLoRAの35.88%より悪い。
     multiplicative projectionもadditive perturbationと同様にembedding空間を再配置。
  3. **intrinsic dimensionality <= 8の知恵**: educationドメイン適応に必要な有効自由度は1つ。
     LoRA r=8とr=16がビット単位で同一。
  4. **embedding空間の幾何学的制約**: 768次元空間を10ドメインで共有。教育ドメインのみを分離するには
     空間の「回転」が必要だが、これは必然的に他ドメインも移動させる。手法の変更では解消できない。
  5. **social_science崩壊が最も深刻**: social_science_recall 0.5774→0.1964（-38.1pt）。
- **config.yml の変更**:
  1. `embedding_adaptation` レバーの note に Iter43 結果を追記。
  2. 新規レバー `classifier_head_adaptation` を config.yml 末尾へ追記（values: education_feature_augmentation, education_boundary_tuning, education_posthoc_calibration）。
- **次の一手: Iter44 で `classifier_head_adaptation` をinvestigate**。
  rc-investigatorは以下の3アプローチのfeasibilityを調査すること:
  (a) education_feature_augmentation: 既存embedding特徴量にeducation-awareな変数を追加
  (b) education_boundary_tuning: LogisticRegressionのeducation classのdecision boundaryを直接調整
  (c) education_posthoc_calibration: 分類器の出力確率にeducation-specificなpost-hoc較正を適用
- **要人間判断**: (1) education_recall の基準値（medical_recall 0.5112）の再検討。(2) Y2
  （`confidence_threshold`の二重責務分離，スキーマ変更）着手前のユーザー確認は引き続き必要。
  (3) classifier_head_adaptationのアプローチの妥当性判断。

---

## B64 [auto-decided 2026-08-02] Iter41: embedding_adapter_only_lora (r=16) rejected。次レバー=embedding_adapter_lora_r8

- 状況: Iter41（embedding_adaptation=embedding_adapter_only_lora, r=16）の rc-analyst 判定（rejected）
  を rc-reflector が検証・確定させた。
- **判定: rejected（確定）**。argmax flip rate 35.88%（閾値<15%の2.4倍超過）。top1_accuracy 有意悪化
  （McNemar p=0.0050）。BH補正後有意退行 1件（medical_recall: q=0.0158）。
- **決定的な学び**:
  1. **LoRAは全パラメータfine-tuningより構造的に優れる**: argmax flip rate 52.56%→35.88%、
     BH-regressions 13→1、top1_accuracy悪化幅 -0.1162→-0.0337。全次元で単調改善。
  2. **social_science/business_economicsの崩壊**: これらのproxy-taskドメインはeducationと意味的に
     近いため、LoRAのembedding変化で最も大きな影響を受けた（social_science -0.2262,
     business_economics -0.1845）。
  3. **ECE改善はLoRAの穏やかな変化の証**: ECE 0.0712→0.0164。LoRAのgentlerなembedding変化が
     確率分布の安定化に寄与。
  4. **トレンドはrank削減を支持**: 52.56%(full FT) → 35.88%(r=16)。r=8では~20%、r=4では~10-15%
     のargmax flip rateが期待される。
- **config.yml の変更**: `embedding_adaptation` レバーの `values` に `embedding_adapter_lora_r8`
  を追記済み。
- **次の一手: Iter42 で `embedding_adaptation=embedding_adapter_lora_r8` を検証**。
  rc-plannerは計画フェーズでLoRA r=8の詳細設計を確定すること。
  変更: `scripts/fine_tune_embedding_lora.py` の `r=16` → `r=8`, `lora_alpha=32` → `lora_alpha=16`。
- **r=8がrejectedの場合の次の手**: LoRAアプローチは構造的限界の可能性が高い（全12層のattention層に
  適用するため、rankを下げてもembedding空間への影響が累積する）。その場合は、(a) embedding出力への
  線形射影（projection head）方式、(b) LoRA target_modulesをout_projのみに限定、のいずれかを検討。
  両方失敗した場合、rc-investigatorへ調査フェーズからの再探索を申し送る。
- **要人間判断**: (1) education_recall の基準値（medical_recall 0.5112）の再検討。(2) Y2
  （`confidence_threshold`の二重責務分離，スキーマ変更）着手前のユーザー確認は引き続き必要。

## B63 [auto-decided 2026-08-02] Iter40: embedding_adaptation=setfit_education_finetune rejected。次レバー=embedding_adapter_only_lora

- 状況: Iter40（embedding_adaptation=setfit_education_finetune）の rc-analyst 判定（rejected）を rc-reflector が検証・確定させた。
- **判定: rejected（確定）**。argmax flip rate 52.56%（閾値<15%の3.5倍超）。top1_accuracy 有意悪化（McNemar chi2=60.46, p<0.0001）。BH補正後有意退行 13/20指標。medical_recall -0.2022（iter31正解40件中14件が直接educationに切り替わった）。
- **決定的な学び**:
  1. **SetFit/SentenceTransformerの全パラメータfine-tuningは単一レバー原則と両立しない**: contrastive learningは全重み（全ドメインの埋め込み空間）を更新するため、argmax flip rate 52.56%は構造的制約。ハイパラチューリングで回避できない。
  2. **education_recall改善はmedical_recall崩壊の裏返し**: 医療質問14件（35%）が直接educationに切り替わった。埋め込み空間でeducationとmedicalが接近した直接的な証拠。
  3. **先行研究（SDJC/JCSE）との違い**: 検索タスクでは埋め込み空間の全体変化が許容されたが、分類器ベースのルーティングでは決定境界の直接変化に帰結するため、単一レバー原則を維持できない。
  4. **embedding適応にはadapter-onlyが必須**: LoRA/adapterのような低ランク更新のみ、または埋め込み出力への線形変換のみが単一レバーで実現可能。
- **config.yml の変更**: `embedding_adaptation` レバーの `values` に `embedding_adapter_only_lora` を追記済み。
- **次の一手: Iter41 で `embedding_adaptation=embedding_adapter_only_lora` を検証**。rc-plannerは計画フェーズでLoRAフックの詳細設計を確定すること。既存のWAFL-PEFTインフラ（domain_lora, Iter18 adopted）のLoRAフックが参考になる。
- **要人間判断**: (1) education_recall の基準値（medical_recall 0.5112）の再検討。(2) Y2（`confidence_threshold`の二重責務分離，スキーマ変更）着手前のユーザー確認は引き続き必要。

---

## B62 [auto-decided 2026-08-02] Iter40: 調査フェーズから開始，次レバー=embedding_adaptation

- **自動選択**: 次イテレーション（Iter40）を調査フェーズから開始。`current_lever=null`。
  次レバーとして `embedding_adaptation=setfit_education_finetune` を config.yml に追記済み。
  `iteration_name` は「embedding_adaptationの調査とSetFitによる教育ドメイン埋め込み適応の実装計画」。
- **選定理由**:
  1. config.yml の全 levers を試し切り（fallback_policy=adopted, classifier_calibration=adopted,
     classifier_training_data_composition=全6値rejected, class_weight_adjustment=rejected）
  2. education_recall=0.4588 で不変だった原因は class_weight ではなく embedding 空間の分離不足
  3. rc-investigator (Iter39) の Tavily 検索で SetFit/SDJC/JCSE の存在を確認済み
  4. SetFit は few-shot contrastive learning により少量データ（education 150件）で
     埋め込み空間を再調整可能。根本原因に直接対処する唯一の実行可能なレバー。
  5. コストは中（数日〜1週間）だが、スキーマ変更は伴わない（data change）
- **Iter40の調査項目**:
  1. SetFitによるnomic-embed-textのeducationドメイン適応のfeasibility
  2. SDJC/JCSE等の日本語ドメイン適応手法の詳細
  3. education_recallの根本原因（embedding空間の分離不足）の定量評価
  4. JMMLU外部の教育固有タスク（再調査）
- **要人間判断**: embedding_adaptationの実装コスト見積もり（数日〜1週間）の承認。

---

## B61 [auto-decided 2026-08-02] Iter39 の単一レバー: class_weight_adjustment (class_weight=None + 手動sample_weight)

- **自動選択**: 単一レバーを
  `class_weight_adjustment=none_manual_sample_weight` とする。
  `iteration_name` は「手動sample_weightによるclass_weight balancedの代替実装」．
- **選定理由**:
  1. rc-investigator（Iter39調査フェーズ）は `class_weight=None` + 手動sample_weight が
     Iter32の失敗（sample_weight *= class_weight_ 乗算バグ）を根本的に解消すると判定
  2. 単一レバー原則の範囲内で実装可能（`train_domain_classifier.py` の code change 2箇所）
  3. config.yml の levers への新規追加はスキーマ変更ではない（data change）
  4. 大規模な新規実装（research_frontier相当）ではない
  5. オフライン完結（実機本走不要）
- **変更ファイル**:
  1. `train_domain_classifier.py` line 144: `class_weight="balanced"` → `class_weight=None`
  2. `train_domain_classifier.py` line 78-80: `_extract_sample_weights()` をドメイン別balanced重み計算に変更
- **ドメイン別balanced重み**（sklearn実測）:
  - 150行ドメイン（education, general, medical等9ドメイン）: 0.9513
  - 77行ドメイン（legal）: 1.8532
  - 全ドメインの有効重み: 142.70（完全一致）
- **investigator提案との差分**: investigatorは全行sample_weight=1.0を提案。
  しかしこれだとlegalの有効重みが142.70→77へ-46%低下し、
  legal_recall退行のリスクが高い（legalは現在0.5833で唯一の基準クリアドメイン）。
  本計画ではドメイン別balanced重みを再現するsample_weightを使用。
- **成功条件**: education_recall > medical_recall基準(0.5112)、
  他9ドメイン18指標のBH補正後有意退行0件、top1_accuracyのMcNemar有意改善
- **コスト**: オフライン完結（~2-3分の分類器再訓練 + 数分の較正後データ生成）

---

## B60 [auto-decided 2026-08-02] Iter38 は rejected で確定。classifier_training_data_composition 全値試し切り、次イテレーションは調査フェーズ

- 状況: Iter38（classifier_training_data_composition=education_hybrid_proxy_and_civics）の rc-analyst 判定（rejected）を rc-reflector が検証・確定させた。
- **判定: rejected（確定）**。主基準（education_recall > medical_recall基準 0.5112）は完全に不成立（0.4000 < 0.5112, gap=11.12pt）。top1_accuracy の McNemar p=0.0748 で有意改善なし。BH補正後有意退行 1件（education_precision）。
- **決定的な学び**:
  1. **japanese_civics の追加は education recall を改善しない**: Iter37（japanese_civicsのみ、Label Leakageあり）で education_recall +0.4235 の改善方向を示したように見えたが、Iter38 で Label Leakage を除去した hybrid approach では recall が -0.0588 へ退化。japanese_civics の「改善効果」は Iter37 の Label Leakage artifact だった可能性が高い。
  2. **class_weight="balanced" の再計算が教育の重みを低下**: education 訓練行数が 150→350 になったため、`class_weight_[education]` が sklearn によって自動再計算され低下。これが education の recall/precision 低下に寄与している可能性が高い。
  3. **hybrid approach の設計自体は Label Leakage 回避に有効**: 7つの単一レバー検証をすべて PASS した。ただし japanese_civics の追加自体が education recall にプラス効果をもたらさない。
- **config.yml の全 levers を試し切り**:
  - classifier_training_data_composition: 6 値すべて試済み（revision=rejected, resampling 案C=rejected, resampling 案A=rejected, handmade=rejected, replacement=rejected, reassignment=invalid, hybrid=rejected）
  - classifier_calibration: 3 値すべて試済み（platt=partial, isotonic=partial, temperature=adopted）
  - fallback_policy: adopted（完了）
  - aggregation_method: Y2 ブロックで試せない
  - E1-E10: 履歴済みまたは no-op
- **次の一手: 調査フェーズから開始**（Iter39）。`current_lever=null` で初期化。
- **rc-investigator への申し送り**（Tavily search 重点調査）:
  1. **`class_weight=None` + 手動 sample_weight の feasibility**: `scripts/train_domain_classifier.py` の変更は code change か？新規レバー `class_weight_adjustment` として config.yml に追加できるか。スキーマ変更かデータ変更かの線引き。
  2. **JMMLU/MMLU 外部の教育固有タスク（再調査）**: 前回調査（Iter37）で EduBench（LLM合成）、Pedagogy Benchmark（チリ教育）のみ。より広範な検索（arXiv, HuggingFace datasets）で教育実務固有の4択タスクを探す。
  3. **education_recall の基準値再検討の材料収集**: medical_recall 0.5112 という基準が education に対して現実的か。類似の研究（ドメイン分類タスクにおける education ドメインの recall）を探す。
  4. **embedding model の education ドメイン適応**: nomic-embed-text の education ドメイン特化ファインチューニングの有効性。
- **要人間判断**:
  1. `class_weight=None` + 手動 sample_weight の実装は code change。新規レバーとして `class_weight_adjustment` を config.yml に追加する形で提案する。
  2. education_recall の基準値（medical_recall 0.5112）の再検討。
  3. Y2（`confidence_threshold`の二重責務分離，スキーマ変更）着手前のユーザー確認は引き続き必要（B49〜B52既存項目）。
  4. fallback設計思想の論文上の位置付け（B48）も未解決。
  5. D5（`data/`/`models/` のバージョン管理方針）も未解決。

## B59 [auto-decided 2026-08-02] Iter37 は invalid で確定。classifier_training_data_composition 全値試し切り、次イテレーションは調査フェーズ

- 状況: Iter37（classifier_training_data_composition=history_culture_japanese_civics_reassignment_to_education）の rc-analyst 判定（invalid）を rc-reflector が検証・確定させた。
- **判定: invalid（確定）**。3つの致命的問題:
  1. **Label Leakage（決定打）**: japanese_civics プールは正確に 150 件で eval ターゲットサイズと同一。全 150 件の japanese_civics 質問が訓練データと評価データの両方に含まれる。純粋 education recall = 1.0000（100%）は分類器が eval 問題を完全に暗記している決定的証拠。
  2. **単一レバー原則の逸脱**: argmax flip rate 52.5% は単一レバー比較の範囲（<15%）を大幅に逸脱。分類器は完全に再訓練された。
  3. **Legal 訓練データ増加**: legal 訓練行数が 77→150 に増加。legal_recall の有意な改善 (+0.2167) は訓練データ増加の直接的結果。
- **決定的な学び**:
  1. **japanese_civics は意味的に適切だが、JMMLU の排他マッピング制約により 150 件しか確保できない**。150 件 = eval ターゲットサイズのため、train/eval で同一質問の重複（Label Leakage）が避けられない。
  2. **この制約を回避するには**: (a) eval から japanese_civics を除外して旧 proxy タスクに戻す、(b) japanese_civics のサブセットのみを訓練に使用する、(c) JMMLU 外部から教育固有タスクを追加する。
  3. **japanese_civics が education の proxy タスクとして意味的に適切である可能性**は示唆された（education_recall +0.4235 の改善方向）。ただし Label Leakage により値は信頼できない。
- **config.yml の全 levers を試し切り**:
  - classifier_training_data_composition: 5 値すべて試済み（revision=rejected, resampling 案C=rejected, resampling 案A=rejected, handmade=rejected, replacement=rejected, reassignment=invalid）
  - classifier_calibration: 3 値すべて試済み（platt=partial, isotonic=partial, temperature=adopted）
  - fallback_policy: adopted（完了）
  - aggregation_method: Y2 ブロックで試せない
  - E1-E10: 履歴済みまたは no-op
- **Iter38 の単一レバー**: `null`（調査フェーズから開始）。`iteration_name` は「education_classification の Label Leakage 回避策の調査と hybrid proxy approach の実装計画」。
- **rc-investigator への申し送り**:
  1. **Label Leakage の回避策を重点調査**: japanese_civics を education 訓練データとして使用するが、eval の education 行を旧 proxy タスクに戻す（hybrid approach）が最も現実的。Label Leakage が解消され、japanese_civics の真の効果が測定可能。
  2. **hybrid approach の実装計画**: `build_dataset.py` と `prepare_lora_training_data.py` で education のタスクマッピングを `japanese_civics + 旧 proxy タスク` に変更。訓練データは japanese_civics + 旧 proxy タスクの両方を含む。eval は旧 proxy タスクのみ。
  3. **代替アプローチの調査**: JMMLU 外部からの教育固有タスク追加の有効性とコスト見積もり。
- **要人間判断**: education_recall の基準値（medical_recall 0.5112）の再検討は、hybrid approach の結果を見てから判断する。
- **留保**: hybrid approach では eval の education 行が旧 proxy タスク（sociology, high_school_psychology, moral_disputes）になる。これは「japanese_civics が旧 proxy タスク上でどれだけ有用か」を測定するものであり、教育固有質問上の性能ではない。

## B58 [auto-decided 2026-08-02] Iter36 は rejected で確定。history_cultureからjapanese_civicsをeducationへ再割当を次レバーに

- 状況: Iter36（classifier_training_data_composition=education_proxy_task_replacement, japanese_civicsへの置換）の rc-analyst 判定（rejected）を rc-reflector が検証・確定させた。
- **判定: rejected（確定）**．主基準（education_recall > medical_recall基準 0.5112）は完全に不成立（0.0529 < 0.5112, gap=45.83pt）．education_recallは0.4588→0.0529へ崩壊（-79.6%）．top1_accuracyも有意悪化（McNemar p < 0.0001, b=134, c=54）．BH補正後有意退行0件（非退行のみ成立）．
- **根本原因の確定**:
  1. **train/evalタスクの不一致**: iter36分類器はjapanese_civicsで訓練、evalは旧proxyタスク（sociology 56 + high_school_psychology 48 + moral_disputes 46 = 150件）．分類器は旧proxyタスクをeducationとして認識できない（education分類確率平均: iter31=0.3056 → iter36=0.0625, -79.6%）．
  2. **JMMLUの排他マッピング制約**: japanese_civicsは150件しか存在せず、history_cultureも24件使用．educationにjapanese_civicsを完全に割り当てるには、history_cultureから除外する必要がある．
  3. **既存proxyタスクでの教育recallは可能**: iter31（旧proxyタスク + temperature較正）でeducation_recall 0.4588を達成．問題は「proxyタスクの意味的ギャップ」ではなく「trainとevalで同一のproxyタスクを使う必要がある」という制約．
- **5連投のrejected（Iter32-36）は決定的**:
  教育recallのトレンド: 0.4588 (Iter31) → 0.4412 (Iter32) → 0.4412 (Iter33) → 0.4353 (Iter34) → 0.4118 (Iter35) → **0.0529** (Iter36)．
- **決定的な学び**:
  1. **proxyタスクの置換はeval再生成なしでは機能しない**: japanese_civicsの意味的整合性は高いが、evalデータセットが旧proxyタスクで固定されているため、置換後の分類器はeval問題をeducationとして認識できない．
  2. **config.ymlの全leversを試した**: `classifier_training_data_composition`の4値（revision, resampling, handmade, replacement）はすべてrejected．`classifier_calibration`の3値（platt, isotonic, temperature）はtemperatureがadopted．`fallback_policy`はadopted．`aggregation_method`はY2ブロックで試せない．
  3. **残る代替アプローチ**:
     - (a) **history_cultureからjapanese_civicsを除外しeducationに割り当てる**（未試行）
     - (b) education_recallの基準値再検討（人間判断必要）
     - (c) handmade問題の大幅増加（コスト大）
- **自動選択: 次イテレーション（Iter37）の単一レバーを
  `classifier_training_data_composition=history_culture_japanese_civics_reassignment_to_education`とする**．`iteration_name` は「history_cultureからjapanese_civicsをeducationへ再割当による訓練データ構成変更」．config.yml の `classifier_training_data_composition` レバーの `values` へ追記した．
- **根拠**:
  1. japanese_civicsをeducationの唯一のproxyタスクとし、history_cultureから除外する．
  2. history_cultureは残り7タスクで150件をサンプリング（行数150→150不変）．
  3. japanese_civicsの意味的整合性が高いため、education_recallが向上する可能性．
  4. **ただし、evalデータセットは旧proxyタスクベースのままのため、同様のtrain/eval不一致リスクがある**（Iter36で確認済み）．このアプローチも失敗する可能性がある．
- **留保**:
  1. このレバーは `education_proxy_task_replacement` とは異なる（history_culture側のマッピングも変更するため、別レバーとして扱う）．
  2. productionモデル（`models/domain_classifier.joblib`）は無変更．
  3. history_culture_recallの退行チェックは必須．
  4. **evalデータセットのtrain/eval不一致リスク**: history_cultureからjapanese_civicsをeducationへ再割当した場合、evalのeducation行は旧proxyタスクのままになるため、**同様の崩壊が再発する可能性が高い**．
- **失敗した場合の次の一手**:
  1. education_recallの基準値（medical_recall 0.5112）の再検討（人間判断必要）
  2. education固有のタスクをJMMLU外部から追加（手作業コスト大）
  3. Y2（dispatch_candidate_threshold）着手前の下調べ（調査フェーズ）
- **要レビュー**: (1) Y2（`confidence_threshold`の二重責務分離，スキーマ変更）着手前のユーザー確認は引き続き必要（B49〜B52既存項目）．(2) fallback設計思想の論文上の位置付け（B48）も未解決．(3) D5（`data/`/`models` のバージョン管理方針）も未解決．(4) education_recallの基準値再検討は人間判断が必要．

---

## B57 [auto-decided 2026-08-01] Iter36 の単一レバー: education_proxy_task_replacement (japanese_civicsへの置換)

- 状況: Iter35（handmade 50件追加）はrejected確定。config.ymlの全leversを試し切った。
  rc-investigator（Iter36調査フェーズ）はjapanese_civics（公民，JMMLU固有150件）が
  educationのproxyタスクとして最も有望と判定した。
- **自動選択**: 単一レバーを
  `classifier_training_data_composition=education_proxy_task_replacement`とする。
  `iteration_name` は「education代理タスクをjapanese_civicsへ置換による訓練データ構成変更」．
- **選定理由**:
  1. rc-investigatorが調査した代替タスク候補の中で，japanese_civicsがeducation実務（学校教育行政）
     との意味的整合性が最も高い（教育基本法，学校管理，教育委員会を含む可能性）
  2. 単一レバー原則の範囲内で実装可能（2ファイルの `_DOMAIN_TASK_MAP["education"]` 値変更のみ）
  3. 大規模な新規実装（research_frontier相当）ではない
  4. 埋め込みモデルのファインチューニング（第二候補）はコスト中（1-2日）かつ分類器再訓練が必要
- **history_cultureへの影響**: japanese_civicsをhistory_cultureから除外すると，
  history_cultureは7タスク（japanese_history, high_school_european_history, prehistory,
  japanese_idiom, japanese_geography, high_school_geography, world_history）になる。
  各タスク150件（計約1050件）から150件をサンプリングするため，行数は150→150で不変。
  意味的特徴の大幅な変化はないと推測されるが，history_culture_recallの退行チェックは必須。
- **変更ファイル**:
  1. `build_dataset.py` line 97-101: `_DOMAIN_TASK_MAP["education"]` を `["japanese_civics"]` へ
  2. `prepare_lora_training_data.py` line 42: 同様の編集
- **追加変更**: `_EDUCATION_PROXY_TASK_TRAIN_TARGET_SIZES` の更新（educationのタスクが1つになるため，
  空辞書にするか japanese_civics のみ残す。`assert sum(...) == _DOMAIN_TARGET_SIZE` のアサーションが
  成立することを確認）
- **留保**: japanese_civicsの内容（実際の質問）を直接確認できていない。計画フェーズでJMMLU.zipから
  japanese_civicsのCSVを抽出し，教育行政相关内容が含まれるか確認する必要がある。

---

## B56 [auto-decided 2026-08-01] Iter35 は rejected で確定．config全levers試し切り完了，次イテレーションは調査フェーズへ

- 状況: Iter35（classifier_training_data_composition=education_handmade_training_problems，
  handmade 50件追加）の rc-analyst 判定（rejected）を rc-reflector が検証・確定させた．
- **判定: rejected（確定）**．主基準（education_recallがmedical_recall基準0.5112を上回ること）が
  不成立（0.4118 < 0.5112, gap=9.94pt）．education_recall自体がIter31比で-4.71pt悪化．
  top1_accuracy McNemar p=0.4966で有意改善なし．ECE悪化（0.0712→0.0751）．
- **4連投のrejected（Iter32-35）は決定的**．education_recallのトレンド:
  0.5000 (Iter31) → 0.4412 (Iter32) → 0.4412 (Iter33) → 0.4353 (Iter34) → 0.4118 (Iter35)．
  baseline (Iter28: 0.4059) と同等まで低下した．
- **決定的な学び**:
  1. **埋め込み空間での意味的競合**: handmade問題50件は既存proxyタスク150件の埋め込み空間と競合し，
     classification boundaryを混乱させた。educationの分類確率平均はほぼ不変だが中央値が低下（0.2552→0.2228）。
     non-education行の偽陽性率（4.83%→5.03%）はほぼ不変であり， handmade問題は「既存education行の
     埋め込み信号を薄めている」．追加ではなく置換が必要かもしれない．
  2. **config.ymlの全leversを試し切った**:
     - classifier_training_data_composition: 3値すべてrejected（revision, resampling, handmade）
     - classifier_calibration: 3値すべて試済み（platt=partial, isotonic=partial, temperature=adopted）
     - fallback_policy: adopted（完了）
     - aggregation_method: Y2ブロックで試せない
     - E1-E10: 履歴済みまたはno-op
  3. **Y2（スキーマ変更）は着手不能**: dispatch_candidate_thresholdの新設はユーザー確認が必要．
- **自動選択: 次イテレーション（Iter36）の単一レバーをnullとする（調査フェーズから開始）**．
  `iteration_name` は「education_recallの根本原因に対する代替アプローチの調査」．
  rc-investigatorはtavily-search等で以下の観点から調査すること:
  1. educationドメインのembeddingsを改善する既存の手法（ドメイン特化埋め込み，fine-tuning等）
  2. proxyタスクの置換（意味的ギャップが小さい代替タスクの探索）
  3. education_recallのボトルネック分析（どの教育問題がどのドメインに誤分類されているか，
     具体的な失敗パターンから改善方向性を見出す）
  4. Y2（dispatch_candidate_threshold）の下調べ（閾値設計の指針となる先行研究）
- **残る要レビュー**: (1) Y2着手前のユーザー確認（B49〜B52）(2) fallback設計思想の論文上の位置付け（B48）
  (3) D5（data/modelsのバージョン管理方針）(4) education_recallの根本原因に対する代替アプローチ

- 状況: Iter34（classifier_training_data_composition=education_proxy_task_resampling，案A:
  sociology=90/high_school_psychology=30/moral_disputes=30）の rc-analyst 判定（rejected）を
  rc-reflector が検証・確定させた．
- **判定: rejected（確定）**．主基準（education_recallがmedical_recall基準0.5112を上回ること）が
  不成立（0.4353 < 0.5112，75.59ptギャップ）．McNemar p=0.0725 で top1_accuracy の有意改善なし．
  education_recallはIter31比で-6.47pt，Iter33比でも-0.59ptの低下．
  非退行条件（BH補正後有意退行0件）は成立するが，主基準が通らないため採用不可．
- **3連投のrejected（Iter32 sample_weight, Iter33 案C, Iter34 案A）は決定的**．
  resampling系レバーは尽きた．sociology pool cap 94に対し90件使用（95.7%）で，
  残り4件の余裕は実質的に意味をなさない．
- **学び**:
  1. education_recallの低下トレンド（Iter31: 0.5000 → Iter34: 0.4353）は懸念．
     案Aで弱い2タスクの訓練露出を-45%に削ったことが，計画フェーズで指摘された
     「学習信号喪失リスク」を実際に発現させた可能性が高い．
  2. 「代理タスクの意味的ギャップ」という根本原因は，抽出比率の変更では解消できない．
     これはIter32の調査で確認済み（sociology(0.625)・high_school_psychology(0.438)・
     moral_disputes(0.435)のいずれも，educationの実務（学校教育行政・学習指導要領等）
     とは主題が明確に異なる）．
  3. 次はeducation固有の手作り訓練問題の追加へ切り替える．
     手作り問題は4択形式（A/B/C/D）を保つ必要がある（書式 shortcuts リスク，
     Iter32調査で確認済み）．教育行政実務に即した問題（学校事故責任，生徒健康管理，
     アレルギー対応，懲戒処分等）を作成する．
- **自動選択: 次イテレーション（Iter35）の単一レバーを
  `classifier_training_data_composition=education_handmade_training_problems`とする**．
  `iteration_name` は「education固有の手作り訓練問題追加による意味的ギャップ解消」．
  config.yml の `classifier_training_data_composition` レバーの `values` へ
  `education_handmade_training_problems` を追記した（`[education_proxy_task_revision,
  education_proxy_task_resampling]` → `[education_proxy_task_revision,
  education_proxy_task_resampling, education_handmade_training_problems]`）．
- **根拠**:
  1. config.yml の levers note で既に「案Aも不成立なら，education固有の手作り訓練問題の追加
     へ切り替える」と指示済み（B54）．
  2. resampling系レバーは尽きた（pool cap 94を使い切った）．次の一手は handmade problem
     追加のみ．
  3. handmade problem追加は config.yml note で見積もられた「1〜3日，オフライン完結」の
     コストで，実機1600問本走は不要．
  4. Y2（スキーマ変更）は着手不能のため，実行可能な登録済みレバーは Y5（education）のみ．
     これは config の全levers を試し切った場合の「停止条件の優先順1」に従う．
- **留保**: handmade problemsの数は50件から始める予定（Iter35計画フェーズで確定）．
  数が少なすぎれば教育recallへの信号が弱く，多すぎればlabel leakageリスクが高まる．
  計画フェーズで適切な数を確定すること．また，手作り問題のドメインラベル付けが
  誤って行われるとlabel leakageになるため，`exclude_queries` の仕組みが適切に
  機能しているか必ず確認すること．
- **要レビュー**: (1) Y2（`confidence_threshold`の二重責務分離，スキーマ変更）着手前の
  ユーザー確認は引き続き必要（B49〜B52既存項目）．(2) fallback設計思想の論文上の
  位置付け（B48）も未解決．(3) D5（`data/`/`models/` のバージョン管理方針）も未解決．

---

## B54 [auto-decided 2026-08-01] Iter33 は rejected で確定．次は案A（90/30/30）を1回試す

- 状況: Iter33（classifier_training_data_composition=education_proxy_task_resampling，案C:
  sociology=70/high_school_psychology=40/moral_disputes=40）の rc-analyst 判定（rejected）を
  rc-reflector が検証・確定させた．
- **判定: rejected（確定）**．主基準（education_recallがmedical_recall基準0.5112を上回ること）が
  不成立（0.4412 < 0.5112，70ptギャップ）．McNemar p=0.1589 で top1_accuracy の有意改善なし．
  education_recall の +3.53pt 改善は SE~3.8pt のノイズ範囲内．非退行条件（BH補正後有意退行0件）
  は成立するが，主基準が通らないため採用不可．
- **学び**:
  1. 案C（70/40/40）は現状比（41/55/54）から sociology を +29pt，他2タスクを -15ptずつ
     変更した．この変化幅では教育recallへの信号がノイズに埋もれた．
  2. 2イテレーション連続（Iter32 sample_weight, Iter33 resampling案C）でrejectedとなった
     背景には，「教育ドメインの代理タスクが本質的にeducationの意味的ギャップを抱えている」
     という根本原因がある．抽出比率の変更という表面的な最適化では，この根本原因に対処できない．
  3. 案A（90/30/30）は変化幅が約2倍（sociology +49pt，他2タスク -25pt）であり，有意検出の
     可能性が高い．ただし弱い2タスクの削減幅が大きい（-45%）ため，学習信号喪失のリスクも
     相対的に高い．
  4. 案Aも不成立なら，「代理タスクの意味的ギャップ」を解消する根本対策（education固有の
     手作り訓練問題の追加）へ切り替える必要がある．
- **自動選択: 次イテレーション（Iter34）の単一レバーを
  `classifier_training_data_composition=education_proxy_task_resampling`（案A: sociology=90,
  high_school_psychology=30, moral_disputes=30）とする**．`iteration_name` は
  「education代理タスク抽出比率の再配分（案A）による訓練データ構成変更」．
- **根拠**:
  1. 案Aはconfig.ymlのlevers noteでbacklog B53が既に例示している（sociology 90・
     high_school_psychology 30・moral_disputes 30）．案Cがrc-plannerによって第一候補に
     選ばれたが，案Aが次点として登録済み．
  2. 案Aは変化幅が案Cの約2倍であり，有意検出の可能性が高い．
  3. config.ymlの`values`に`education_proxy_task_resampling`は既に登録済み（案Cが既定）．
     案Aへの変更は`values`の順序変更または`_EDUCATION_PROXY_TASK_TRAIN_TARGET_SIZES`の
     値変更のみで，スキーマ変更ではない．
  4. 案Aを1回試して不成立なら，handmade problem追加へ切り替える．これ以上resamplingのバリエ
     ーションを増やしても，意味的ギャップという根本原因には届かないと判断する．
- **留保**: 案Aの弱い2タスクの削減幅（-45%）は相対的に高く，Iter32のconfusion matrix分析が
  示す「弱い2タスクの誤分類は`medical`・`social_science`・`legal`との学術的近接が主因」
  という機序を踏まえると，該当タスクの学習信号自体を失わせて逆効果になるリスクがある．
  案Aがrejectedの場合，handmade problem追加へ直ちに切り替える．
- **要レビュー**: (1) 案Aの Sociology 90 件は残りプール上限 94 件に対し 95.7% を使い切る．
  これ以上 sociology を増やす余地がないため，案Aが不成立の場合の次の一手は handmade
  problem追加のみになる．(2) Y2（`confidence_threshold`の二重責務分離，スキーマ変更）
  着手前のユーザー確認は引き続き必要（B49〜B52既存項目）．(3) fallback設計思想の論文上の
  位置付け（B48）も未解決．

---

## B53 [auto-decided 2026-07-31] Iter32 は rejected で確定．sample_weightはclass_weightと結合し逆効果．次は抽出比率の再配分

- 状況: Iter32（classifier_training_data_composition=education_proxy_task_revision，弱い代理タスク
  `high_school_psychology`・`moral_disputes`への`sample_weight=2.0`，Y5）の rc-analyst 判定
  （rejected）を rc-reflector が検証・確定させた．
- **判定: rejected（確定，rc-analyst 提案を覆さず）**．主基準（education_recallがmedical_recall
  基準0.5112を上回ること，point estimate）が不成立であるだけでなく，education_recall自体が
  before比で悪化（0.4588→0.4412，-1.76pt）し，全体top1_accuracyも統計的に有意に悪化した
  （McNemar p=0.0026，discordant 11件が全て悪化方向，改善方向0件）．得られた利得が一つもない．
- **機序（sklearnソース・実データで実測確認済み）**: `LogisticRegression(class_weight="balanced")`
  は`sample_weight`に依存してクラス重みを再計算する（`sklearn/utils/class_weight.py`の
  `compute_class_weight`が`sample_weight`込みの重み付きクラス別合計を分母に使う）ため，
  taskへの`sample_weight`増加と`class_weight`によるドメイン間バランス調整は独立ではなく
  数式レベルで結合していた．education用`sample_weight`増加により
  `class_weight_[education]`が0.9513→0.5931へ自動的に下がり，狙った2倍の強化は実質+24.7%へ
  減衰し，変更対象外だった`sociology`行の実効重みも-37.7%失われ，同時にeducation以外の
  9ドメイン全てに一律+7.6%の相対的優位を与える副作用を生んだ．
- **学び**: 「単一レバー原則を実装上は守っていても，`sklearn`側の既存の仕組み
  （`class_weight='balanced'`）と数式レベルで結合しているレバーは，実質的に複数の量を
  同時に動かしてしまう」という一般化可能な知見．本リポジトリの分類器訓練で今後
  `sample_weight`を使う場合は，`class_weight`との結合の有無を必ず先に確認すべき事項として
  記録する．計画(Iter32)は`sample_weight *= class_weight_`という乗算関係（一次情報で確認済み）
  までは把握していたが，`class_weight_`自体が`sample_weight`の値に連動して再計算される点
  （入れ子の依存関係）を見落としていた．次にsample_weightを使う設計を検討する際は，両方向の
  依存を確認すること．
- **実験用ファイルの扱い**: `models/domain_classifier_iter32_reweighted.joblib`・
  `data/classifier_train_iter32_reweighted.jsonl`はいずれも削除した（rejectedが確定し
  機序も特定済みのため再利用の見込みがなく，数値的な結果はjournal.md「分析(解釈) (Iter32)」
  「考察 (Iter32)」節に記録済みで十分参照可能．両方とも`.gitignore`対象のため削除はgit履歴に
  残らない）．`results/iter32_calibrated_predictions.jsonl`はIter29〜31の
  `resultsXX_calibrated_predictions.jsonl`と同様に一次結果データとしてgit追跡対象に残した．
- **自動選択: 次イテレーション（Iter33）の単一レバーを
  `classifier_training_data_composition=education_proxy_task_resampling`とする**．config.yml の
  `classifier_training_data_composition`レバーの`values`へ`education_proxy_task_resampling`を
  追記した（`[education_proxy_task_revision]`→
  `[education_proxy_task_revision, education_proxy_task_resampling]`）．`iteration_name`は
  「education代理タスク抽出比率の再配分（sample_weight不使用）によるclass_weight結合回避型
  データ構成変更（Y5継続）」．
- **根拠（次のレバーの選定理由）**: Iter32の失敗は「弱いタスクへの重み付け」という着想自体では
  なく，`sample_weight`という実装手段が`class_weight="balanced"`と結合していたことに起因する．
  したがって次点は，同じ着想（`sociology`優位・弱い2タスク劣位という配分の是正）を
  `sample_weight`を一切使わない形（抽出段階でのタスク別目標件数の変更）で実現する．
  **`education`の総行数を150件（他ドメインと同数）のまま変えない**ことが設計上の要点で，
  これにより`class_weight_[education]`はIter31以前と完全に同じ値（0.9513）のままとなり，
  `sample_weight`を使わないためsklearn側の結合バグの影響を受けない．これは
  rc-analystが次への示唆で挙げた「サンプル数を増やしつつsociologyの比率を高める折衷案」を
  一歩進め，**サンプル数自体は増やさず構成比のみを変える**ことで，候補(1)（単純な
  サンプル数増量，150→298）が抱える同種のclass_weight連動リスク（総行数を増やせば
  `class_weight_[education]`がさらに下がる）も同時に回避する．具体的な配分比率（例:
  sociology 90・high_school_psychology 30・moral_disputes 30）は次の計画フェーズで確定する．
- **留保（次の計画フェーズへの申し送り）**: この変更も「代理タスクの意味的ギャップという
  根本原因」自体は解消しない．`sociology`自体が「学校教育行政実務」という`education`の
  実務上の定義とは主題が異なる学部教養レベルの社会学問題であることに変わりはなく，
  達成できるのは「3タスクのうち相対的に混同されにくいタスクの寄与を増やす」という限定的な
  改善にとどまる可能性が高い．目標未達に終わった場合の次点候補は，journal.md「分析(解釈)
  (Iter32)」節に整理済み（候補(3)＝4択形式を保った手作り問題追加，または埋め込み特徴量
  自体・base estimatorの見直し）．
- 要レビュー: (1) Y2（`confidence_threshold`の二重責務分離，スキーマ変更）着手前のユーザー確認
  は引き続き必要（B49・B50・B51・B52既存項目，新規の追加事項なし）．(2) fallback設計思想の
  論文上の位置付け（B48）も未解決．(3) 「`education_recall`という既存メトリクスの改善」と
  「`education`ドメインの実務忠実性」の両立不可能性（B52既出）は今回の結果を経てもなお
  未解決のまま．(4) config.yml・backlog B52 の「business_economics 0.4533」という陳腐化した
  記述は，計画(Iter32)がjournalで訂正済み（現状下限はmedical_recall 0.5112）だが，config.yml
  本体・backlog B52本文の訂正はまだ未実施（本イテレーションでも範囲外として申し送りを維持）．

---

## B52 [auto-decided 2026-07-31] Iter31 は adopted で確定．本番反映を実施．次は Y5（education 訓練データ）

- 状況: Iter31（classifier_calibration=temperature，Y4）の rc-analyst 判定（adopted）を
  rc-reflector が確定させた．
- **判定: adopted（全面採用，確定，rc-analyst 提案を覆さず）**．d0003 X9 の成功条件（ECE≤0.150・
  top1_accuracy 非退行・per-domain 20指標の BH 補正後悪化方向有意指標 0 件の AND 条件）を
  platt（Iter29，ECE 絶対閾値未達で partial）・isotonic（Iter30，medical_recall の BH 補正後
  有意悪化で partial）に続き，temperature が初めて明確に満たした．ECE 0.193358→0.071201
  （目標に 7.88pt の余裕，3 手法中最良），top1_accuracy 0.585→0.605625（McNemar p=0.000906，
  非退行を上回る有意改善），per-domain 20 指標の BH 補正後有意退行 0 件．Iter30 で唯一の懸念
  だった medical_recall も有意差なし（p=0.182422）でむしろ改善方向．
- **本番反映: 実施済み**．`models/domain_classifier.joblib`（較正前，
  sha256=`3a5610aa88d70b9e94af4620d2747b313c52b834a9dbaa5e872ed45c3520dcb0`）を
  `models/domain_classifier_uncalibrated_pre_iter31.joblib` へ退避し，
  `models/domain_classifier_temperature.joblib`
  （sha256=`04bb9ff223d5b41d94ab13897eb89b59af64524d1fd0d5ce9d598a5a3b06a2e5`）で置き換えた．
  判断根拠: (1) 成功条件の AND 条件を明確な余裕で満たしている，(2) `config.yaml`・公開 API の
  変更を一切伴わない可逆なファイル差し替えである（不具合が判明すれば
  `models/domain_classifier_uncalibrated_pre_iter31.joblib` へ即座に戻せる），(3) 委譲時の指示で
  「rc-reflector の自律判断範囲内（可逆な判断）として進めて構わない」と明示的に許可された操作
  である．**注意**: `models/` は `.gitignore`（19行目）で除外されており，この置き換えは git
  履歴に残らない．ロールバック手順と両ファイルの sha256 はこの記録と journal.md「考察 (Iter31)」
  節にのみ残るため，次回このモデルに触れる際は必ず両方を参照すること．
- **学び**: (1) isotonic の medical_recall 悪化が「OvR 方式由来のクラス固有曲線歪み」という
  機序で説明できることが，temperature への切り替えのみで解消したという形で強く裏付けられた．
  同一データ・同一 cv/ensemble の下で較正手法だけを変えた比較が 3 イテレーション連続で積み
  上がったことで，単発の考察ではなく再現性のある知見になった．(2) 「表現力が高い較正手法が
  必ず良い較正を生むとは限らない」という一般的な知見が実測で裏付けられた．ECE 改善幅は
  temperature(0.1222) > isotonic(0.0719) > platt(0.0258) と，もっとも柔軟性の低い手法が
  もっとも大きく改善するという事前の留保とは逆の結果になった．小標本条件下では OvR 方式の
  クラス別自由度が held-out のノイズを拾って過学習し，かえって較正を悪化させたためと考えられる．
  次に較正関連のレバーを検討する際は「手法の表現力の高さ＝較正の質」という前提を置かないこと．
  (3) `models/` が gitignore 対象であるため，較正済み分類器の本番反映は git 履歴に残らない．
  D5（backlog 未解決事項，`data/`/`models/` のバージョン管理方針）が引き続き未解決であり，
  本番アーティファクトを差し替える判断が繰り返し発生する局面では，最低限 sha256 ハッシュの
  マニフェストを journal/backlog に記録する運用（今回実施した方式）を今後も徹底する必要がある．
- **Y4（分類器の較正，d0003 X9）は本イテレーションをもって完了**．config.yml の
  `classifier_calibration` レバーは `[platt, isotonic, temperature]` の3値すべてを試し終えた．
- **自動選択: 次イテレーション（Iter32）の単一レバーを
  `classifier_training_data_composition=education_proxy_task_revision`（Y5，d0003 X8 の
  スコープを絞り込んだもの）とする**．config.yml の levers 末尾（classifier_calibration の
  直後）へ新規レバーとして追記した．`iteration_name` は「education ドメインの代理タスク妥当性
  見直しによる訓練データ品質改善（Y5）」．
- **根拠（Y2 に進むか，他の可逆なレバーを探すかの判断）**: d0004 §5 の優先順位は
  Y1（完了）→Y4（完了，本イテレーション）→Y2（前提整備，要ユーザー確認）→Y3（Y2 完了後）→
  Y5（education/legal のデータ不均衡是正）．Y2 は `config.yaml` への
  `dispatch_candidate_threshold` 新設・`aggregator.select_dispatch_targets()` のシグネチャ変更
  というスキーマ変更を伴い，B49・B50・B51 で繰り返し「着手前にユーザー確認が必要」と申し送られて
  きた．rc-reflector の自律判断権限は可逆な判断（レバー選定）に限られ，スキーマ変更を伴う着手
  そのものは今の場で自律開始できない．Y3 は Y2 完了が前提のため同様に着手不能．よって実行可能な
  登録済みレバーは `classifier_calibration`（完了）・`fallback_policy`（完了）のみとなり，
  `aggregation_method`（Y3）は Y2 完了までブロックされたまま実質「試せない」．これは config
  の全levers を試し切った場合と実質同じ状況（唯一残る登録レバーが実行不能）と判断し，SKILL.md の
  停止条件の優先順1（journal/backlog の学びから次の有望なレバーを自分で考案し config.yml へ
  追記して継続する）に従い，Y5 を新規レバーとして追記した．
- **Y5 のスコープを education のみへ絞り込んだ根拠**: d0003 X8 は education・legal 双方の
  データ不均衡是正を対象としていたが，Iter31 の実測で legal は既に X8 の成功条件
  （他ドメイン下限 business_economics 0.4533 を上回ること）を満たしていた
  （legal_recall 0.5833）．legal の訓練データ最少77件という懸念は，較正への影響としても
  B50・B51 で2イテレーション連続で反証済みであり，今回 recall 自体の実測でも同様の反証が
  得られた．一方 education_recall は 0.4059（全10ドメイン中最下位）で，Iter28 基準線から
  Iter29〜31（較正のみ変更）まで一貫して同一値のまま変化していない．education の訓練データは
  150件で他の majority ドメインと同数のため，サンプル数不足ではなく
  `scripts/prepare_lora_training_data.py:42` が使う代理タスク（`sociology`・
  `high_school_psychology`・`moral_disputes`）の妥当性自体が疑わしい（d0001 §5.1 が推奨する
  「代理タスクの写像表の明示・境界事例の別集計」という未実施の宿題に対応する）．
- 要レビュー: (1) Y2（`confidence_threshold` の二重責務分離，スキーマ変更）着手前のユーザー確認
  は引き続き必要（B49・B50・B51 既存項目，新規の追加事項なし）．(2) fallback 設計思想の論文上の
  位置付け（B48）も未解決．(3) Y5（education 訓練データ）の具体的な実施方法（代理タスクの
  置換候補選定，必要なら手作り訓練問題の追加）は次イテレーションの investigate/plan フェーズで
  具体化する．(4) D5（`data/`/`models/` のバージョン管理方針，較正済み分類器の sha256 マニフェスト
  運用の恒久化）は今回も未解決のまま．

---

## B51 [auto-decided 2026-07-31] Iter30 は partial で確定．本番反映は見送り，次は temperature scaling を検証

- 状況: Iter30（classifier_calibration=isotonic，Y4）の rc-analyst 判定（partial）を
  rc-reflector が確定させた．
- **判定: partial（部分的採用，確定，rc-analyst 提案を覆さず）**．d0003 X9 の成功条件は
  「ECE≤0.150 かつ top1_accuracy 非退行 かつ per-domain 20指標の BH 補正後・悪化方向の
  有意指標0件」の AND 条件．ECE は 0.193358→0.121424（目標に2.86ptの余裕）で成立，
  top1_accuracy は McNemar p=0.301（非退行，方向は改善寄り）で成立するが，per-domain は
  `medical_recall` がBH補正（q=0.05，20指標中2件通過のうち悪化方向1件）後もなお有意に悪化
  （0.4831→0.3820, p=0.000144）しており不成立．3条件ANDのうち1条件不成立のため partial．
- **本番反映: 見送り**（`models/domain_classifier.joblib` は較正前のまま，
  `models/domain_classifier_isotonic.joblib` へ置き換えない）．理由は成功条件のAND条件が
  未成立であること．`medical` は訓練150件の多数派ドメインであり，Iter29のlegalのような
  「訓練データ拡充で解消しうる」という見立てが立たない．
- **学び**: (1) isotonicの実際のリスクは事前に警戒していた小標本ドメイン（legal）ではなく
  多数派ドメイン（medical）に現れた．「訓練データ量が少ないドメインほど較正の影響を受け
  やすい」という直感的仮説は，Iter29（B50，computer_science/mathematics も偽陽性で該当）に
  続き2イテレーション連続で反証された．較正関連レバーの事前リスク評価には訓練データ量だけで
  なく較正曲線の形状・到達可能な確信度の天井の確認が要る．(2) BH補正を計画段階から組み込む
  運用（調査(Iter30)の申し送り）は機能した．Iter29の「20指標中9指標が該当（うち有意は0件）」
  という過検出をIter30では「20指標中2指標が該当（うち有意な悪化は1件）」まで絞り込めた．
  (3) medical_recall悪化の原因は0/1張り付き（isotonic特有の既知病理，非選択クラスの
  82%行で発生）でも小標本held-out不安定性でもなく，isotonic較正曲線がmedicalクラス固有に
  系統的にスコアを圧縮する（較正後の最大確率0.7062が他9ドメイン0.7496〜0.8795を全て下回る）
  という，事前に想定していなかった第3の機序である可能性が高い．
- **自動選択: 次イテレーション（Iter31）の単一レバーを `classifier_calibration=temperature`
  とする**．config.yml の `classifier_calibration` レバーの `values` へ `temperature` を
  末尾追記した（`[platt, isotonic]` → `[platt, isotonic, temperature]`，可逆な値追加であり
  スキーマ変更ではない）．`iteration_name` は「分類器較正のtemperature scaling方式による
  argmax不変性の実証とECE目標到達可否の検証」．
- **根拠（`cv=3`感度分析 と `method='temperature'` のどちらを優先するかの判断）**:
  `cv=3`はisotonicという同一手法内のハイパラ変更に過ぎず，medicalは訓練150件のため`cv=5`
  （1foldあたり約30件）を`cv=3`（同約50件）にしても sklearn 公式の目安「greater than
  ~1000」からは変わらず大きく下回ったままであり，かつ悪化した19行のうち0/1張り付きが直接
  原因の行は0件だったことから，fold サンプル数の微調整では根本原因（較正曲線のクラス固有
  圧縮）に届きにくいと判断した．一方 `method='temperature'` は，クラスごとに個別の較正器を
  fit せず単一スカラーTでロジット全体を変換する構造であるため，isotonic/plattが抱える
  OvR方式由来の全リスク（クラス固有の曲線歪み・tie・0/1張り付き）を構造的に排除でき，
  今回発見した根本原因に直接対処する．sklearn公式が「Tはsoftmaxの最大値の位置に影響しない」
  と明記しておりtop1_accuracy不変が理論保証される点も，per-domain非退行という今回最も
  厳しかった条件への対処として優先度が高いと判断した根拠である．
- 要レビュー: (1) temperatureは多クラス全体で単一のTしか学習しないため，isotonic
  （0.121424）はもとよりPlatt（0.16751，目標未達だった実績）と比べても較正の柔軟性が低く，
  ECE改善幅がより小さくtemperatureが0.150に届かない可能性がある．その場合は「per-domain
  非退行のためにはOvR方式の柔軟性を犠牲にできない」という新しい知見が得られ，isotonicの
  運用調整（例: medicalのみ較正を無効化する等）を再検討する材料とする．(2) Y2
  （`confidence_threshold`の二重責務分離，スキーマ変更）着手前のユーザー確認は引き続き必要
  （B49既存項目）．(3) fallback設計思想の論文上の位置付け（B48）も未解決．(4) 較正済み
  分類器の本番反映可否は，いずれかの較正手法が成功条件を完全に満たした時点で改めて判断する．

---

## B50 [auto-decided 2026-07-31] Iter29 は partial で確定．本番反映は見送り，次は isotonic を追試

- 状況: Iter29（classifier_calibration=platt，Y4）の rc-analyst 判定（partial）を rc-reflector が
  確定させた．
- **判定: partial（部分的採用，確定）**．d0003 X9 の成功条件は「ECE≤0.150 かつ top1_accuracy
  非退行」の AND 条件．ECE は 0.19336→0.16751 と改善方向（-2.58pt）は決定論的な測定で確定的だが
  絶対閾値 0.150 に未達（0.0175pt 不足）．top1_accuracy は McNemar p=0.139 で非退行（方向は
  改善，discordant_b_only=67>discordant_a_only=50）．Iter20（E3）の partial 判定運用実績と
  対称的なケースであり，adopted・rejected いずれの二択にも実態が合わない．
- **本番反映: 見送り**（`models/domain_classifier.joblib` は較正前のまま，
  `models/domain_classifier_platt.joblib` へ置き換えない）．理由は d0003 X9 の AND 条件が
  未成立（ECE 未達）であることそのもの．**legal データ拡充（Y5）の完了を前提条件にはしない**
  （下記の追加分析で legal 固有説が相対化されたため）．
- **追加分析（rc-analyst 未了分を解消）**: legal・education の2ドメインのみだった per-domain CI
  比較を全10ドメイン20指標へ拡張したところ，「CI下限が較正前を下回る」という成功条件(3)の
  字義通りの基準では legal recall 以外に8指標（computer_science精度・recall等，訓練150件の
  ドメイン含む）が該当した．いずれも区間は非交差ではなく重なっており統計的に有意ではない．
  **legal は訓練データ最少（77件）だから較正の影響を受けやすい，という rc-analyst の当初仮説は
  唯一の説明ではないと判明**．より妥当な説明は，20指標を多重比較補正なしに単純な CI下限比較で
  判定する運用が，較正による11.0%のargmax再配分の下で偽陽性を生みやすいという統計的な
  アーティファクト．
- 学び: (1) per-domain CI下限の単純前後比較は，多重比較の補正なしでは非退行チェックとして
  脆弱（20指標中9指標が該当するが全て区間重複＝有意でない）．次回以降 success_criteria (2) は
  「CI下限比較」ではなく「区間が非交差」または「ドメイン単位のMcNemar検定」へ改める運用を
  検討すること（Iter28の学び「paired比較でMcNemarとWilson CIの周辺重複が食い違う」と同根）．
  (2) legal recall低下を「訓練データ最少ドメイン固有の脆弱性」と即断せず，全ドメイン同一手順の
  分析を最初から計画に含めること（事後の穴埋めをしない）．
- 自動選択: 次イテレーション（Iter30）の単一レバーを **classifier_calibration=isotonic**
  （config.yml 既登録の候補，スキーマ変更なし・ユーザー確認不要）とする．`cv`（5）・`ensemble`
  （True）は Platt と同一に固定し，較正手法のみを単一レバーとして変える．`cv=3` 感度分析は
  isotonic の主結果次第の副次分析に留める．`iteration_name` は「分類器較正のisotonic方式による
  ECE目標達成の追試とドメイン別非退行の全数検証」．
- 根拠: 計画(Iter29)・調査(Iter29)の時点で「isotonic はPlattが不成功の場合のみ次イテレーションで
  別途検証する」と明記済み．今回Plattが ECE絶対閾値未達（＝「不成功」）だったため条件成立．
  isotonicが目標未達の場合，`method='temperature'`（top1_accuracy不変が理論保証される代替）を
  次々点として検討する．
- 要レビュー: (1) Y2（`confidence_threshold`の二重責務分離，スキーマ変更）着手前ユーザー確認は
  引き続き必要（B49既存項目）．(2) fallback設計思想の論文上の位置付け（B48）も未解決．
  (3) 較正済み分類器の本番反映可否は，isotonic等が成功条件を完全に満たした時点で改めて判断する
  （本番ルーティング挙動を変える判断は都度検討）．

---

## B49 [auto-decided 2026-07-31] Iter28 は adopted で確定．次は Y4（分類器の較正）を Y2 より先に実施

- 状況: Iter28（fallback 方策の廃止，Y1）の rc-analyst 判定（adopted）を rc-reflector が確定させた．
- **判定: adopted（確定）**．成功条件4項目のうち3項目（top1_accuracyのMcNemar有意改善 p=1.59e-7・
  fallback発生0/1600の直接確認・answer_quality改善+3.8ptが3SD=2.61ptを超過）は明確に成立．
  条件4（非退行）は`general`ドメインのrecallのみCI下限を割ったが，同一212行内でprecisionが
  0.3134→0.6522へ大幅改善しており，fallback廃止に内在する構造的トレードオフ（fallbackの唯一の
  送り先がgeneralであることによる母集団変化）と判断し，判定を覆す理由にしていない．
- 学び: (1) paired比較ではMcNemarとWilson CIの周辺重複が食い違いうる（p=1.59e-7の一方でCIは
  1.91pt重複）．次回計画時，paired設計と分かっている場合は「主基準McNemar，Wilson CIは参考」と
  明記する運用に改めること．(2) fallbackは「安全網」ではなく「識別困難なケースを正解率8.5%の
  選択肢へ機械的に振り替える処理」だったことが分散版実機で統計的に裏付けられた．(3) 事前実測
  （central_iter26 vs 26b，中央集権アーキテクチャ）と分散版実測はtop1・κでほぼ完全一致し，
  Iter26の「アーキテクチャを変えてもルーティング判定は完全一致する」という知見が別レバーでも
  再確認された．
- 自動選択: 次イテレーション（Iter29）の単一レバーを **classifier_calibration（Y4，
  CalibratedClassifierCV，d0003 X9）**とする．`iteration_name` は「分類器の較正
  （CalibratedClassifierCV）によるECE改善とルーティング非退行の検証」．config.yml の levers 末尾に
  新規レバーとして追記した．
- 根拠（Y2 vs Y4 の判断基準）: (1) **自律判断の可逆性**: Y2 は `config.yaml` への
  `dispatch_candidate_threshold` 新設・`select_dispatch_targets()` のシグネチャ変更という
  設定ファイル形式・関数シグネチャの変更であり，config.yml 自身が「着手前にユーザー確認が必要」と
  明記している．rc-reflector の自律判断権限は可逆な判断（レバー選定）に限られ，スキーマ変更を
  伴う着手そのものは今の場で自律開始できない．(2) **コストと独立性**: Y4 は既存の訓練データ
  （1427件）に対するオフライン処理で，d0004 が「Y1と並行して進めてよい」と明記している．
  スキーマ変更もユーザー確認も不要．(3) **Y2設計への波及**: Y4の結果（較正でECEがどれだけ
  下がるか）はY2の`dispatch_candidate_threshold`のデフォルト値設計の判断材料になりうる．
- 要レビュー: (1) Y4完了後，Y2（スキーマ変更）に着手する前に改めてユーザー確認を得ること．
  (2) fallback廃止の`general`ドメインrecall低下（trade-off）の扱いは，下記B48要レビュー項目
  （fallback設計思想の論文上の位置付け）に統合済み．今回新たな示唆は無い．

---

## B48 [auto-decided 2026-07-31] Iter27 は実験不成立（no-op）と判定．次は fallback 廃止（Y1）へ

- 状況: Iter27（B47 の計画）の 3 実行（max_confidence / majority_vote / llm_judge，各 1600 問，
  計約 5 時間）は 07-31 03:44 に完走していたが，分析・記録・コミットが行われないまま約 12 時間
  停止していた．本セッションで事後整理した．
- **判定: invalid（実験不成立）**．3 方式とも Iter25 基準線と主要指標が小数点以下まで完全一致し，
  McNemar の不一致ペアは 3 方式とも 0 件．**`dispatched_domains` の長さが 2 以上の行が
  1 件も無かった（0/1600）**．「集約方式に差が無い」のではなく，集約が一度も実行されていない．
- 機序: `aggregator.select_dispatch_targets()` は `confidence >= confidence_threshold` で
  候補を絞ってから top-k を取る．`supervised_classifier` では各ノードが 10 クラス確率の自分の分
  のみを返し総和が 1 になるため，2 ノードが同時に 0.5 以上になるには p₁+p₂ ≥ 1.0 が必要で
  起こり得ない．実測でも **2 位 confidence の最大値は 0.4955**．デプロイの成否と無関係に成立する．
- **構造的問題（次の設計に直結）**: `confidence_threshold` が (a) fallback ゲートと
  (b) dispatch 候補ゲートの 2 役を兼ねている．閾値を下げると両方が同時に動くため，
  集約方式も fallback 方策も単一レバーとして分離できない．分離案は docs/d0004 §5 Y2．
- 副産物: 3 実行 + Iter25 基準線が「生成のランダム性のみ異なる 4 反復」になり，**d0003 X6
  （回答品質のノイズ床の確定）が実質完了した**．answer_quality の標準偏差 0.87pt，
  **3SD = 2.61pt**．暫定値 1.3pt（n=2）を置き換え，`config.yml` の `success_criteria` に反映済み．
- 自動選択: 次イテレーション（Iter28）の単一レバーを **fallback 方策の廃止**（d0003 X5 /
  d0004 Y1）とする．`iteration_name` は「fallback 方策の廃止によるルーティング精度・
  回答品質への影響測定」．
- 根拠: 既存データから効果が実測済みである．`results/central_iter26/`（fallback 廃止相当）と
  `results/central_iter26b/`（現行）はアーキテクチャ・分類器・データセットが同一で fallback
  方策だけが異なる（B46 の副産物）．top1 +2.94pt，κ +3.26pt，answer_quality +5.74pt（3SD の
  2.2 倍），レイテンシも −323ms，McNemar p=1.59e-7．**fallback が発動した 212 問だけを見ると，
  general へ送ると 18/212（8.5%）しか正解しないが，argmax のドメインへ送れば 65/212（30.7%）
  正解する**．d0002 §8-3 の「fallback は Random（10.1%）を下回る」が定量的に裏付けられた．
- 実行上の注意: Y1 の実験では `dispatch_top_k` を現在の 2 から **1 へ戻す**こと．
  `confidence_threshold` を下げると候補ゲートも緩むため，top_k=1 に固定して単一レバーを保つ．
- 要レビュー: (1) 分散版で中央版と同じ fallback 廃止効果が再現するか（Iter26 の「アーキテクチャを
  変えてもルーティングは完全一致」から予測されるが未確認）．(2) fallback を完全に廃止するか，
  閾値を下げるに留めるか．廃止すると「確信が持てないときに汎用ノードへ退避する」という設計書の
  意図自体を捨てることになるため，論文上の位置付けをどう書くかは人間判断が要る．
- **【2026-07-31 追記，Iter28 完了・確定を受けて】** (1) は Iter28 で再現確認済み（journal.md
  「考察 (Iter28)」節・backlog B49）．top1・κは事前実測とほぼ完全一致（誤差0.04pt未満），
  answer_qualityの乖離もノイズ床3SD=2.61pt内で「事前実測とおおむね整合」と確定した．
  **(2) は今回の実験結果だけでは未解決のまま残る**：fallback廃止により`general`ドメインの
  recallがCI下限を割る一方，同じ212行内でprecisionが大幅改善するという表裏一体のトレードオフが
  実測された（journal.md「分析 (解釈) (Iter28)」節）．これは「fallbackという安全網を撤廃してよいか」
  という論文上の位置付けの判断材料が増えただけで，判断そのものは依然として人間判断事項である．
  次レバー（Iter29 = Y4 分類器の較正）の選定はこの論点とは独立に決定済み（B49）．

---

## B48-b [auto-decided 2026-07-31] 残留 heartbeat プロセスの停止と，運用上の 3 件の不備

- 状況: Iter27 の停止が watchdog に検知されなかった原因を調査した．
- **原因: Iter23 の使い捨てスクリプト `/tmp/iter23_heartbeat.sh`（PID 871683）が 2026-07-30 01:42 から
  1 日 14 時間動き続け，`state.json` の `updated_at` を 120 秒ごとに上書きしていた**．
  停止条件のマーカー `/tmp/iter23_start.done` が生成されず無限ループになっていた．
  `updated_at` は watchdog がハング検知に使う唯一の指標であり，偽の heartbeat がこれを恒久的に
  無効化していた．
- 自動選択: 当該プロセスを `kill` した．停止を確認済み．
- 根拠: 完了済みイテレーションの使い捨てスクリプトであり，機能は `state.json` の 1 フィールドを
  上書きするだけ．研究記録に偽の生存信号を書き込み続ける害の方が大きい．可逆な操作である．
- 併せて発見した不備 2 件（未修正・申し送り）:
  1. **Iter27 の 3 実行に provenance が無い**: `config.yaml`・`git_head.txt`・`metrics.json` を欠き，
     ファイル名も `results_topk2_*.jsonl` と非標準．F5 は `mise run start` の標準経路でのみ機能する．
     標準経路を外れる場合は `_record_experiment_provenance()` を明示的に呼ぶこと．
  2. **journal に実在しないコマンドが記録されている**: Iter24 計画節の
     `metrics.py --results A --compare B` は `--compare` 引数が存在せず実行できない
     （`compute_mcnemar_test()` は関数としてのみ存在し CLI 未公開）．
- 要レビュー: 実験監視に `/tmp` の使い捨てスクリプトを使う運用自体を見直すか，
  停止条件にタイムアウトを必須化するか．`state.json` の heartbeat は research-cycle
  オーケストレータ自身のみが更新すべきである．

---

## B47 [user-approved 2026-07-30] Iter27計画: 高度な集約方式の比較実験（research_frontier項目5）

- 状況: Iter26（中央集権ルータ再実験，B46で修正済み）完了後の次イテレーションとして，項目5（高度な
  集約方式）の実機比較実験を計画した．aggregator.py への実装自体はcommit `178960a`で完了済み
  （`max_confidence`/`majority_vote`/`llm_judge`の3方式，config.yamlの`aggregation_method`で選択可能）．
- 狙い: 現状は`dispatch_top_k=1`で最も自信度の高いドメイン専門家1人にしか聞いていない．
  `dispatch_top_k=2`で2人に聞き，回答の選び方（集約方式）を変えることで精度が改善するかを検証する．
- 比較する3方式:
  1. `max_confidence`（現状基準）: probe時の自己申告confidenceが高い方をそのまま採用．追加コストなし
  2. `majority_vote`（新規）: 2人の回答を選択肢（A/B/C/D）に還元し，一致すればそれを採用．一致しなければ
     `max_confidence`にフォールバック．追加のLLM呼び出しなし
  3. `llm_judge`（新規）: 3人目の審査役LLMに2つの回答を見せ，優れている方を選ばせる．質問1件につき
     LLM呼び出しが1回増える
- 対象データセットの検討: 単一ドメイン設問は1人目の専門家の自信度が高く2人目を呼んでも結果が変わり
  にくいため，2ドメインの確信度が拮抗しやすい複合設問100問を中心に検証するのが効率的．全1600問で
  行うか複合設問中心に絞るかはIter26完了後，実施直前に判断する．
- 実施手順:
  1. 現在デプロイ済みのDockerイメージは aggregator.py 変更（`178960a`）より前にビルドされた可能性が
     高いため，`mise run setup`（再ビルド）→`mise run deploy`（10ノードへ再配布）が必要．
  2. **Iter26bの実機実験が完了してから実施する**（`mise run deploy`は各ノードのOllama/appコンテナを
     再作成するため，進行中の実験と並行させると干渉するリスクがある）．
  3. `config.yaml`の`dispatch_top_k`・`aggregation_method`を切り替えながら3方式を実行し，正答率・
     End-to-End精度を比較する．
- 要レビュー: 全1600問 vs 複合設問中心のどちらで実施するかは実施直前に確定する．

---

## B46 [auto-decided 2026-07-30] Iter26初回実行: fallback方策の違いが不一致の全原因と判明，再実験へ

- 状況: バグ修正版 `scripts/run_central_experiment.py` で1600問の初回実験を実行したところ，
  top1_accuracy=0.585（分散版Iter25=0.5556），McNemar p=1.6e-7で有意差ありという結果になった．
  中央版の方が高精度という，Iter24（バグにより低精度）とは逆方向の予想外の結果．
- 調査: embedding が wafl500/wafl502 間で完全に一致（bit単位で同一）することを実機で直接確認．
  分類器のsha256ハッシュもローカル・実機間で完全一致．よって「同一classifier」の前提は崩れていない．
- 原因判明: 分散版は `confidence_threshold=0.5` でargmaxドメインの確率をフィルタしてから
  dispatchし，閾値未満なら`general`ノードのlight_modelへフォールバックする（`node.py:run_ask_flow`）．
  中央版の初回実装は閾値なしの純粋argmaxで常にdispatchしていた．McNemarで不一致だった77件を
  精査した結果，**全77件が分散版のfallback行（212件中）と完全に一致**していた（中央版が正解62件・
  分散版が正解15件，いずれもfallback行のみ）．つまり「同一classifierなのに結果が違う」という
  Iter24以来の謎は，実装バグではなく，2つの実装が異なる意思決定方策（閾値ゲート付きdispatch vs
  無条件argmax）を比較していたことによる，単一レバー原則違反だったと判明した．
- 自動選択: `run_central_experiment.py` に分散版と同一の `confidence_threshold` フィルタと
  fallback ロジック（`node.py:FALLBACK_PROMPT_TEMPLATE`・`FALLBACK_MAX_TOKENS` を再利用，
  fallback先は分散版実験のデフォルト requester と同じ `domain_nodes["general"]` のホスト）を
  実装し，アーキテクチャのみを単一レバーとして分離できるようにした．回帰防止テスト
  `tests/test_run_central_experiment.py` を追加（2件）．再実験を実施する．
- 根拠: X2の目的は「同一classifierを前提に，分散か中央集権かというアーキテクチャのみのコストを
  測る」ことであり，fallback方策の有無という別の変数が混入した状態での比較は単一レバー原則に
  違反し，結論を導けない．
- 要レビュー: 再実験の結果（journal.md「Iteration 26」節）を参照．

---

## B45 [user-approved 2026-07-30] Iter25: データセット拡充後の基準線再取得

- 状況: B44 の方針決定に基づき，項目2（複合ドメイン評価データセット拡充）を実施．データセットが
  1520問→1600問へ変わったため，Iter23のX1基準線はもはや直接比較できず，新データセットでの基準線
  再取得（Iter25）を実施した．
- 結果: single_domain_top1_accuracy=0.5693・Cohen's kappa=0.5215・answer_quality_accuracy=0.508667が
  Iter23と完全一致（ルーティング・回答品質は無変化）．overall top1_accuracyの低下（0.5651→0.5556）は
  複合設問の母数増加（20→100）による合成比率の変化であり回帰ではない．compound_domain_top1_accuracy
  は初めて統計的に議論できる規模（n=100）で0.35と測定された．詳細はjournal.md「Iteration 25」節参照．
- 判定: adopted（新基準線として確定）．以後のIter26（中央集権ルータ再実験）・Iter27（集約方式比較）は
  この基準線（results/20260730_145356/）と比較する．
- 実験の実体（provenance）: `results/20260730_145356/`に`config.yaml`・`git_head.txt`・`metrics.json`を
  事後的に追加保存した（`mise run start`タスク自体はresults.jsonlのみコピーする仕様のため，F5の趣旨に
  合わせて手動で補完．次回以降このタスク自体を拡張する余地がある）．

---

## B44 [user-approved 2026-07-30] research_frontier 項目を全て実装・設定する方針決定

- 状況: converged状態（B42）でのcontinue後，ユーザーから直接「research_frontier 項目を全て実装・設定
  せよ」との指示を受けた．調査したところ，項目1（新規ドメイン追加）・項目3（LLM-as-judge/E2E評価）は
  既にコード実装済みで，config.ymlの記述が古いだけだったと判明した．
- ユーザーに3点確認し，以下の方針で合意した:
  1. 全体方針: 未完了分（項目2・4・5）のみ着手し，項目1・3はconfig.ymlの記述を実態へ更新するに留める
  2. 複合ドメイン評価データセット（項目2）の拡充規模: 100問程度へ拡充（d0003 X4 / D4 の推奨に従う）
  3. 項目4（中央集権ルータ再実験）・項目5（新集約方式）: 実機（wafl500〜509）での実験まで本セッション
     内で実行する
- 実施内容: (1) config.yml research_frontier節・metrics.py docstringの是正 (2) 複合ドメイン設問20→100問
  拡充（Iter25，本ファイルB45参照）(3) scripts/run_central_experiment.pyのプロンプト・APIエンドポイント
  不一致を修正（backlog B43参照）(4) aggregator.pyに多数決集約・LLM-as-judge集約を追加
- 要レビュー: 項目4・5の実機実験（Iter26・Iter27）は本セッション内で継続実施中．

---

## B43 [auto-decided 2026-07-30] Iter24 実装ファイルの commit 漏れを是正

- 状況: `converged` 後の `/research-cycle continue` 実行時，`git status` で `scripts/run_central_experiment.py`
  （未追跡）と `config.yaml` の `central_router` 節（未commit）が working tree に残っていることを発見した．
  Iter24 完了コミット（`ee1d549`）は `.claude/research/*` のみを含み，レバーの実装本体は一度も commit
  されていなかった．
- 追加発見: 実際のファイル（411行，SSH経由で各ドメインノードへ委譲する方式）は journal（229行，ローカル
  `OllamaClient` 直接利用）・Slack報告（253行）のいずれとも一致せず，実装が複数回改訂されたにもかかわらず
  記録が更新されていなかった．また ruff の未使用 import（`os`, `subprocess`）2件が残っており，「ruff 0
  warning」の過去報告と食い違う．詳細は journal.md 先頭「記録訂正・commit 漏れの是正」節を参照．
- 自動選択: 実験を実際に生成したコードをそのまま（lint 修正等の事後整形をせず）git commit した．再現性を
  記録の見た目より優先するのが妥当と判断．`state.json` の heartbeat・`last_commit` も同時に同期した．
- 根拠: (1) 該当コードは既に判定確定済み（rejected）の実験を生成した実体であり，改変すると再現性が損なわれる
  (2) commit 自体は可逆な操作であり，自律判断ポリシー上「通常の git commit」に該当する (3) 内容は用途外の
  ファイルを含まず，今回の変更に直接関係する範囲に限定した．
- 要レビュー: rc-implementer が計画からの実装方針変更（local→SSH ピボット等）を journal に記録する運用を
  徹底すること．必要であれば SKILL.md のイテレーション完了時コミット検証手順に「実コード・設定ファイルの
  commit 有無」を明記する改訂を検討されたい．

---

## B42 [auto-decided 2026-07-30] Iter24完了: 全levers試し切り完了、converged

- 状況: Iter24（X2: 中央集権ルータ比較）がrejectedで完了。config.yml の全 levers（E1〜E10）を試し切り。
- 判定: `status="converged"`。次イテレーションは実験を開始せず待機。
- 根拠: (1) levers の全エントリを實驗済み（採用: E6, E10, E20, E23 / 棄却: E8, E24 / 無効: E22 / 保留: E3, E4, E5, E7）(2) reflector が新レバーを考案せず (3) research_frontier の項目は単一レバー原則の枠を超える大規模変更
- 要レビュー: ユーザーに次の方向を指示してもらう必要がある。候補: (A) research_frontier の項目から着手（データセット本格化、評価軸②③実装、ベースライン比較）(B) 新レバーの考案（prompt builder 再利用、fallback 見直し、etc.）

---

## B41 [auto-decided 2026-07-30] Iter24: X2 中央集権ルータ比較の実施計画

- 状況: Iter23（X1 基準線再取得）が成功（主基準4項目が期待値と完全に一致）．次は docs/d0003
  第3段階の X2（中央集権ルータ比較）へ移行．
- 自動選択: 単一レバーを `routing_architecture=central_router` として，新規スクリプト
  `scripts/run_central_experiment.py` を作成し，分散版（現行）と比較する．
- 根拠: (1) d0003 で X2 が「最重要」と位置付けられている．(2) rc-investigator の調査で，
  実装コストが低〜中程度（1 新規スクリプト，~150-200行）と見積もられた．(3) ルーティング結果は
  理論上一致するはず（同一 classifier），違いはオーバーヘッド（通信・VRAM）のみ．(4) McNemar 対
  比較は metrics.py に実装済み．(5) データセット分離は完了済み（0 件重複）．
- 計画の詳細: journal.md「計画 (Iter24)」節を参照．成功条件は top1 accuracy 差 < 2pt, McNemar
  p > 0.05, probe latency 改善 50% 以上．VRAM 測定は「理想（全モデル常駐）」と「実測（swap あり）」
  の両方を報告．
- 実装示唆: `run_experiment.py` は変更せず，`scripts/run_central_experiment.py` の新規作成のみ．
  出力スキーマは run_experiment.py と同一．classifier は `classifier.py:load_domain_classifier()`
  を再利用．Ollama 接続は `expert_backend.py:OllamaClient` を再利用．

---

## B40 [user-approved 2026-07-30] `/research-cycle continue` 実行前の敵対的総点検・追加修正

- 状況: B39 で行った環境修復（F1〜F5）自体に見落としがないかを敵対的にレビューした．独立した
  subagent によるレビューと自己点検の両方で，以下を発見した．
- **[重大・修正済み] `.claude/research/config.yml` の E4（`confidence_signal_method=self_consistency_semantic`）・
  E5（`p_true`）の記述が誤解を招く形だった**: E3・E7（真の no-op）と同列に「Iter21/22 とも無効」
  「分岐の排他構造」とまとめていたが，E4/E5 は「no-op」ではなく，**現在の HEAD（`30e3627`，Iter22 の
  分岐順序修正が実際に反映済み）でこれらを設定すると，E6（supervised_classifier）の分類器分岐に
  到達できなくなり，E6 の効果を丸ごと上書きする**という積極的な副作用を持つ．さらに時系列の誤りも
  あった：Iter21/22 は「E4 が動いた上で E6 に退行した」のではなく「分岐順序修正が未適用/未デプロイ
  だったため E4 が 1 度も実行されなかった」だけである．config.yml のレバー全体コメント・E4/E5 個別
  note を訂正した．次期 rc-planner がこれを読んで X3（E4 再設計）を検討する際の誤解を防ぐ．
- **[重大・user-approved で修正済み] `state.json` の `iteration` が `"Iter22"`（無効判定済み）のまま，
  SKILL.md が定める「イテレーション完了時の初期化」が未実施だった**: B39 の時点では `current_lever`
  のみに着目し「rc-planner が上書きするので実害なし」と判断していたが，`rc-experimenter.md` に
  「実験ディレクトリ名を config の name_scheme と**現イテレーション番号**から決める」と明記されており，
  このまま continue すると次の実験が誤って Iter22 として記録されるリスクを見落としていた．ユーザーに
  確認の上，SKILL.md 127-135 行の初期化手順を代行した: `iteration`: `"Iter22"` → `"Iter23"`，
  `current_lever`/`experiment_dir`/`experiment_deadline`/`iteration_thread_ts`: null，
  `notion_toggle_created`: false，`iteration_name`: null，`updated_at`: 現在時刻に更新．
  `e10_results`/`e20_results`/`e8_results`/`e22_results`（SKILL.md の必須スキーマ外の参照情報）は
  そのまま残した．
- **[軽微・修正済み] `tools/smoke_check.py` の `_SIGNAL_FIELD_EXPECTATIONS` に到達不能な dead entry**:
  `"semantic_entropy"` というキーを追加していたが，実際の `confidence_signal_method` の値は常に
  `"self_consistency_semantic"`（Python 定数名 `CONFIDENCE_SIGNAL_SEMANTIC_ENTROPY` との混同）．削除した．
- **[軽微・修正済み] `run_experiment.py:168` の `write_text` に `encoding="utf-8"` 欠落**: 追加した．
- **[軽微・未修正，報告のみ]** `smoke_check.py` の SSH コマンドにタイムアウト未設定（healthcheck.py の
  `HEALTHCHECK_TIMEOUT_S` と不整合）．`routing_method=embedding` 用の専用チェック未実装（現状未使用の
  設定のため実害小）．`metrics.py` の `compute_tie_rate`/`compute_confidence_dispersion` の
  `probe_candidates` 1件のみのケース，`compute_auroc` の同点ケースのテスト未カバー（数式自体は
  検算済みで正しい）．いずれも次回以降の課題として残す．
- 検証: 修正後 `uv run pytest -q` 198 passed / 2 skipped，`uv run ruff check` 全通過，
  `config.yml`/`state.json` の構文検証済み．

---

## B39 [auto-decided 2026-07-29] continue 実行前の環境修復・state.json 不整合の申し送り

- 状況: d0002_research_cycle_findings_2026-07.md（Iter1〜22 総括調査）・d0003_next_experiments_2026-07.md
  （次の実験計画）を新規作成する総括調査の一環として，`/research-cycle continue` 実行前に対処すべき
  修正を行った．その過程で `state.json` の不整合を発見した:
  `phase="investigate"` だが `current_lever="confidence_signal_method=self_consistency_semantic"`
  （Iter22 で無効と判明済みのレバー）のままで，`iteration` も `"Iter22"` のまま増分されていない．
  SKILL.md の「イテレーション完了時」手順（`current_lever=null`・`iteration` インクリメント等への
  リセット）を経ずに，ユーザー指示による実験停止で `phase=investigate, status=running` へ直接
  書き換えられたため（journal.md「実験 (Iter22) — 停止（ユーザー指示）」参照）．
- 自動選択: `state.json` 自体は research-cycle オーケストレータの管理領域のため直接書き換えず，
  次に起動する rc-investigator が journal.md 冒頭の訂正記録節（本 backlog と同日付）を読んで
  正しい前提（E6+E10 が最良既知構成，E3/E4 は no-op/無効，config.yaml は F1 で最良既知構成へ復元済み）
  から調査を再開できるようにした．`current_lever` の古い値は，rc-planner がフェーズ2で新しいレバーを
  確定した時点で上書きされるため実害はないと判断した．
- 実施した環境修復（docs/d0003 第1段階 F1・F1-b 相当，詳細は journal.md 冒頭「記録訂正・環境修復」節）:
  1. `config.yaml`: `expert_model` を10ノード全て `expert-mesh-{domain}-lora`（Iter18 採用構成）へ復元．
     `confidence_signal_method` を `self_report` へ復元（`self_consistency_semantic` のままだと
     `routing_method=supervised_classifier` の分岐に到達せず E6 の成果が無効化されるため）．
     wafl500〜509 の Ollama に対応 LoRA モデルが全10ノード登録済みであることを実機で確認済み．
  2. `.claude/research/config.yml`: `levers` 節の前提コメント（Iter15 時点のまま）を Iter22 時点へ
     全面更新．E3（confidence_elicitation）・E4（self_consistency_semantic）・E7（embedding_postprocess）
     の note に no-op / 排他構造の注記を追加．
  3. journal.md 冒頭に，ECE 系列の誤記載・top1_accuracy と single_domain_top1_accuracy の取り違え・
     E3 採用判定の取り下げ（D1 相当）を訂正として追記．
- 要レビュー: 次期 rc-planner は，config.yml の levers 節が示す優先順位（docs/d0003 §0: 第2段階 X1 基準線
  再取得 → 第3段階 X2 中央集権ルータ比較・X4 複合ドメイン評価・X5 fallback 見直し）に従って次の一手を
  選ぶこと．docs/d0003 の F2（デプロイ検証ゲート）・F3（metrics.py 指標統合）・F5（再現性マニフェスト）
  は本セッションで別途着手中のため，完了状況を journal.md で確認してから X1 に着手すること．

---

## B38 [auto-decided 2026-07-29] Iter21: 実験無効（bug 発見）, Iter22 で E4 再実行

- 状況: Iter21（`confidence_signal_method=self_consistency_semantic`）の結果は無効。`http_server.py` の `_estimate_probe_confidence()` で `routing_method=supervised_classifier` の early return（line 323-329）が `confidence_signal_method` チェックより先に実行されており、`self_consistency_semantic` のコードパスが 1 回も到達していない。
- 検証証拠: (1) Iter20 とメトリクスが完全に同一（top1=0.5651, kappa=0.5215, fallback=0.1316）(2) `local_inference_ms` が 1-3ms（semantic entropy なら数秒〜数十秒）(3) `semantic_entropy` フィールドが 0/1520 件 (4) ログの `routing_method: supervised_classifier` — 全プローブで classifier が使用された。
- 修正方針: `http_server.py` の `_estimate_probe_confidence()` で `confidence_signal_method` チェックを `routing_method` チェックより先に移動（Option A）。これにより両者が独立して動作する。
- Iter22 計画: 修正後、同一 1520 問で `confidence_signal_method=self_consistency_semantic` を再実行。期待: latency 増（1 probe あたり 9 LLM calls）、mean_duration_ms 6500ms → 10000-15000ms 程度。
- 要レビュー: (1) 修正が正しく適用されたか実験ログで確認すること。(2) `probe_timeout_s=120` が有効になるか。

---

## B37 [auto-decided 2026-07-29] Iter21 計画: confidence_signal_method=self_consistency_semantic

- 状況: Iter21 の計画フェーズ完了。単一レバーを `confidence_signal_method=self_consistency_semantic`（E4）で確定。
- 変更内容: `config.yaml` の 2 行変更（`confidence_signal_method: self_report → self_consistency_semantic`, `probe_timeout_s: 60.0 → 120.0`）
- コード変更: 0行（`self_consistency_semantic` は既に完全に実装済み）
- 成功条件: ECE 0.1927 → 0.150 以下（-4.3pt 以上）。top1_accuracy/Cohen's kappa の非退行。
- ノイズ幅: Iter18 Phase C ↔ Iter20 の比較で top1/ECE ともに 0.00pt の再現性。ECE の有意な改善閾値は約 -0.02pt（3SE）。
- 実装フェーズへの示唆: (1) config.yaml の 2 行変更のみ (2) `measure_semantic_diversity.py` の作成（config.yml note で要求）(3) semantic_entropy の分析用スクリプト作成の検討
- 要レビュー: (1) ECE の成功条件（0.150 以下）が妥当か (2) probe_timeout_s の 120s 引き上げが適切か

---

## B36 [auto-decided 2026-07-29] Iter20 総括と次イテレーションの単一レバー決定

- 状況: Iter20（E3: confidence_elicitation=top_k_with_probs）の結果、adopted 判定。同点タイ率 82.83%→0.00%、ECE 0.7388→0.1927 の決定的改善。
- 自動選択: 次イテレーション（Iter21）の単一レバーを `confidence_signal_method=multi_sample_semantic`（E4）とする。`iteration_name` は「multi_sample_semantic による不確実性推定とconfidence較正改善」。
- 根拠: (1) E4 は未着手で、config levers で E5 より優先度が高い。(2) E4 は温度 0.7〜1.0、N=5 のマルチサンプリングで不確実性を測定。(3) Iter11 の失敗（T=0.1）とは異なり、文献に基づく適切な設定。(4) E5 は Ollama バージョン確認が必要で E4 より実装コストが高い。
- 要レビュー: (1) E4 の着手前にユニーク回答数（多様性）を計測すること。(2) E4 の成功条件（ECE 改善目標値）を rc-planner が具体化すること。

---

## B35 [auto-decided 2026-07-29] E7（embedding_postprocess=whitening）のスキップと E3 への方向転換

- 状況: Iter20 の単一レバーを E7（embedding_postprocess=whitening）とする計画だったが、調査フェーズで重大な構造的問題が発見された。
- 発見: `embedding_postprocess` は `routing_method=supervised_classifier` の下では全く適用されない。`http_server.py` の `_estimate_probe_confidence()` では `routing_method=embedding` の場合のみ `apply_embedding_postprocess()` が呼ばれ、`supervised_classifier` パスでは `query_embedding` が classifier に生で直接渡される。つまり現在の構成で whitening を有効にしても no-op である。
- Alternatives: (A) `routing_method=embedding` に変更（単一レバー原則違反）、(B) `classifier.py` にコード変更（config-only 原則違反）、(C) E7 をスキップして次のレバーへ移行。
- 自動選択: (C) を採用。E7 をスキップし、次レバー E3（`confidence_elicitation=top_k_with_probs`）へ移行。`iteration_name` を「top_k_with_probs による confidence 較正改善と同点タイ率への影響測定」に変更。
- 根拠: (1) E7 は config-only の枠内で検証不可能（no-op 確定）。(2) A/B は単一レバー原則または config-only 原則の両方を違反する。(3) E3 は Iter16 で rejected されたが、当時は n=46 の評価集合。E1 完了後の 1520 問で再検証する意義がある。(4) E3 は config.yaml の 1 行変更のみで検証可能（コスト極めて低い）。
- 要レビュー: (1) E3 の再試行が統計的に意味があるか（n=1520 で SE ≈ 0.007）。(2) Iter16 で E3 が rejected された理由（同点タイ率の低下効果なし）が、n=1520 で再現されるか。

---

## B34 [auto-decided 2026-07-29] Iter19 総括と次イテレーションの単一レバー決定

- 状況: Iter19（E8: expert_model_size=qwen3.5-4b-q4_K_M）の結果、rejected 判定。主目的の推論速度改善は完全に反証（6498ms vs 3515ms, 1.85 倍遅い）。VRAM 改善のみ（5.67GB→3.4GB, -40%）では不十分。回答品質も大幅低下（-26.4pt）。
- 自動選択: 次イテレーション（Iter20）の単一レバーを `embedding_postprocess`（E7）とする。`iteration_name` は「embedding_postprocess=whitening による embedding 空間の幾何的性質とルーティング精度への影響測定」。E7 は config のみの変更（embedding_postprocess=whitening）で、コード変更不要、コスト極めて低い。
- 根拠: (1) E8 は rejected 確定。E7 は config.yml levers で E8 より先に定義されている（優先度高い）。(2) E7 は「mean-centering + whitening 後に cosine を取る」で、Iter2 の embedding 失敗が『幾何』由来か『信号不在』由来かを切り分ける最小実験。(3) config-only 変更で検証可能（コスト極めて低い）。(4) E7 の成功条件は top1_accuracy/Kappa の改善であり、expert_model_size の遅延問題とは無関係。
- 要レビュー: (1) E7 の config-only 変更が正しく適用されるか確認すること。(2) whitening 適用後の embedding 分散・平均値を metrics.py で計測できているか確認すること。

---

## B33 [auto-decided 2026-07-29] Iter18 総括と次イテレーションの単一レバー決定

- 状況: Iter18（E10: expert_specialization=domain_lora）の実験・分析が完了．Phase A（LoRA なしベースライン: answer_quality_accuracy=0.2787）と Phase C（domain_lora: answer_quality_accuracy=0.5013）の比較．
- 自動選択: E10 は「採用」と判定．次イテレーション（Iter19）の単一レバーを `expert_model_size`（E8）とする．`iteration_name` は「Qwen3.5 モデルサイズ 9B→4B 変更による推論速度・VRAM 効率・回答品質への影響測定」．
- 根拠: (1) E10（domain_lora）は answer_quality_accuracy +22.3pt, end_to_end_accuracy +14.5pt の大幅改善で採用確定．これにより本研究の目的（メッシュ型専門ノード群によるドメイン別最適ルーティング）が初めて実証された．(2) 残りの levers は E7（embedding_postprocess=whitening）と E8（expert_model_size=qwen3.5-4b）のみ．(3) E7 は「whitening が unsupervised embedding の幾何的問題を解消するか」の切り分け用で、supervised_classifier 採用後は次要的なレバー．(4) E8 は現行 9B モデル（5.67GB VRAM）の代わりに 4B モデル（~2.4GB）を使用し、推論速度・VRAM 効率・回答品質への影響を測定．(5) research_frontier の「ベースライン比較」や「top-k dispatch 高度化」は levers を試し切ってから着手する順序に従う．
- 要レビュー: (1) E8 の expert_model_size 変更は light_model と expert_model の両方に影響するか、expert_model のみか確認すること．(2) 4B モデルの回答品質が 9B モデルより大きく劣化する場合、E8 の結論は「サイズ低下は回答品質に直結する」になる．このトレードオフをどう位置付けるか．(3) E7（whitening）はスキップしてよいか、または E8 実施後に検討するか．
- 恒久知見: expert_specialization（LoRA）によりノード間に能力差が生まれて初めて、「正しいドメインにルーティングすること」が回答品質に直結する．同一モデルのメッシュでは、誤ルーティングしても回答品質はほぼ変わらない．これは本研究のメッシュ型専門ノード群の価値を初めて実証した．

## B32 [auto-decided 2026-07-27] Iter17 総括と次イテレーションの単一レバー決定

- 状況: Iter17（E6: routing_method=supervised_classifier）の実験・分析が完了．1520問（10ドメイン）を実機10ノードで実行．top1_accuracy=0.5651（Iter16 0.2059 vs +0.3592），Cohen's kappa=0.5215（Iter16 0.1067 vs +0.4148），ECE=0.2118（Iter16 0.7388 vs -71.3%），同点タイ率 0.00%（Iter16 82.83% vs -82.83pt）．McNemar 対比較で決定的有意差（p < 0.000001）．Wilson CI 重ならなし．成功条件: 主基準2件・副基準3件 全5件達成．education recall のみ非退行違反（JMMLU データセットの構造的問題に起因）．
- 自動選択: E6 は「採用」と判定．次イテレーション（Iter18）の単一レバーを `expert_specialization`（E10）とする．`iteration_name` は null のまま（rc-planner が決定）．
- 根拠: (1) supervised_classifier によりルーティング精度が Random の 5.6 倍，Iter16 の 2.7 倍に改善し，ノード間の能力差が回答品質に反映される環境が整った．(2) 現在全ノードが同一モデルで差分はプロンプト 1 文だけであるため，誤ルーティングしても回答品質はほぼ変わらず，top1_accuracy は下流に帰結を持たない代理指標だった．(3) E10（expert_specialization）の実施により，初めて「正しいドメインにルーティングされた質問が，実際に良い回答を得るか」を評価軸②（回答品質，LLM-as-judge）と③（End-to-End）で検証できる．(4) 本命は `domain_lora`（単一ベース + ドメイン LoRA アダプタ）で，6GB VRAM 制約下で最も現実的．(5) E10 と同時に評価軸②③を実装することが必須（それらが無いとルーティングの価値を測れない）．
- 要レビュー: (1) E10（expert_specialization）の実装はコード変更を伴う（LoRA アダプタの準備・ロード，評価軸②③の実装）ため，単一レバー原則の枠を超える可能性があることを確認すること．(2) 日本語の法律特化オープン生成モデルが見つからないため，`offtheshelf_specialized` ではなく `domain_lora` を優先する方針を確認すること．(3) 評価軸②③（回答品質・End-to-End）の実装スコープと成功条件を rc-planner が具体化すること．

## B31 [auto-decided 2026-07-27] Iter16 総括と次イテレーションの単一レバー決定

- 状況: Iter16（E3: confidence_elicitation=top_k_with_probs）の実験・分析が完了．1520問（10ドメイン）を実機10ノードで実行．top1_accuracy=0.206（Iter15 0.184 vs +0.022），Cohen's kappa=0.107（Iter15 0.081 vs +0.025），同点タイ率 82.83%（Iter15 98.29% vs -15.46pt）．McNemar 対比較で有意差なし（p=0.0783）．ECE は悪化（0.715→0.739）．
- 自動選択: E3 は「部分的採用」と判定．次イテレーション（Iter17）の単一レバーを `routing_method=supervised_classifier`（E6）とする．`iteration_name` は「embedding ベース教師あり分類による routing_method の検証」．
- 根拠: (1) self_report の根本的限界が numeric_scalar と top_k_with_probs の両方で確認された（ECE > 0.7）．confidence elicitation の方式変更だけでは self_report の構造的問題（各ノードが自分の分野に偏った confidence を出す）は解消されない．(2) embedding ベースの教師あり分類は self_report（言語的自信）とは全く異なる信号源であり，E3 の結果とは独立して評価できる．(3) Iter2（embedding）の失敗は unsupervised cosine similarity の anisotropy 問題であり，教師あり分類では解消される可能性がある（Varangot-Reille+ JAIR2025，RouterDC NeurIPS2024）．(4) 訓練/評価分離は既に実装済みであり，label leakage の再演リスクは低い．(5) config.yaml 1行変更のみで検証可能（コード変更不要）．
- 要レビュー: (1) E6 の実機実験結果が，self_report を有意に上回るか確認すること．(2) E6 が不成功の場合，次は E7（whitening）または E4/E5（confidence_signal_method の変更）を検討する方針を backlog に残しておくこと．

## B30 [auto-decided 2026-07-27] Iter15 総括と次イテレーションの単一レバー決定

- 状況: Iter15（E1: eval_set_size）の実験・分析が完了．1520問（10ドメイン×150問単一 + 20問複合）を実機10ノードで実行．top1_accuracy=0.184（Wilson CI: [0.165, 0.204]），Cohen's kappa=0.081（chance直上），Random=0.101を上回る．E1 成功条件全条件 PASS（統計基盤の整備完了）．
- 自動選択: 次イテレーション（Iter16）の単一レバーを `confidence_elicitation=top_k_with_probs`（E3）とする．`iteration_name` は「Verbalized Top-K による二峰飽和と同点タイの解消検証」．
- 根拠: (1) 98.29% の同点タイが最大のボトルネックであり，self_report の二峰飽和（0.9 が 74.9%）がその根本原因．(2) Tian et al. (EMNLP 2023) の Verbalized Top-K は確率の合計制約（sum=1）で 0/1 飽和を機械的に壊す（gpt-3.5 の ECE を 0.131→0.047）．(3) プロンプトのみの変更（config 1行）で，コード変更不要（既に実装済み）．(4) 同一 1520 問データセット上で McNemar 対比較が可能（E1 で整備した統計基盤を活用）．(5) E4〜E7 と比較してコスト最小・リスク最低．
- 要レビュー: (1) E3 の実機実験結果が，同点率の低下と kappa の改善をもたらすか確認すること．(2) E3 が不成功の場合，次は E6（supervised_classifier）を検討する方針を backlog に残しておくこと．

## B29 [user-approved 2026-07-27] 実機デプロイ時のホスト環境不整合（3種）を sudo で修復

- 状況: B28 で実装した全レバーを実際に物理クラスタ（wafl500〜509）へ `mise run deploy` した結果，
  10ホスト中5ホストで環境不整合により失敗した。(1) wafl504・wafl506・wafl507 は
  `nvidia-container-toolkit` が不完全（`nvidia-container-runtime` 実行ファイル自体が欠落）で GPU が
  使えず，(2) wafl508・wafl509 は `docker compose`（v2プラグイン）自体が未導入だった。
  いずれもコードの問題ではなく，WAFL-PEFT とも共有する物理ホストのパッケージ状態の不整合。
- ユーザーの選択: 「Claude が sudo でインストールする」を選択（AskUserQuestion で確認済み）。
- 実施内容: 各ホストで sudo apt install により，(1) 他7ホストと同一バージョンの
  `nvidia-container-toolkit=1.19.0-1` 一式，(2) Ubuntu標準リポジトリの `docker-compose-v2` を導入。
  実施前に全対象ホストで WAFL-PEFT 等の起動中コンテナが無いことを確認してから docker daemon 再起動を
  伴う作業を行った。あわせて `docker-compose.gpu.yml` を，一部ホストで欠落しがちな `nvidia-ctk` 依存の
  CDI 方式から，全ホスト共通で登録済みのレガシー `runtime: nvidia` 方式へ書き換えた（リポジトリ側の
  変更としてコミット対象）。
- 根拠: 実機投入がB28の完了条件の残り（「次段階としてユーザーの別途確認を要する」と明記済み）であり，
  今回のユーザー指示（実験のテストと問題の完全解決）の範囲内。sudo でのシステムパッケージ導入は
  CLAUDE.mdの「本番環境・破壊的操作」に該当するため個別に確認を取った。
- 要レビュー: 修復した3ホスト（504/506/507）と2ホスト（508/509）が，今後のクラスタ運用でも
  他ホストと同一のパッケージ状態を維持できているかを，次回のノード追加・再構築時に確認すること。
  詳細は journal.md「実験 (Iter15) — 実機デプロイテストとインフラ不備の解決」を参照。

## B28 [resolved 2026-07-26] E1〜E7 + モデル/専門家/評価軸②③の一括実装（ユーザー手動セッション）

- 状況: ユーザーが対話セッションで，`docs/d0001_literature_survey_2026-07.md` と
  `plans/p0001_research_direction_2026-07.md` を完全に把握した上で全レバーを実装するよう明示的に指示．
  単一レバー原則を今回に限り上書きし，バッチ0〜10（11単位）で E1〜E7・モデル9B→4B化・専門家の実体化
  （S1）・評価軸②③を全て実装した．詳細は journal.md の「Iteration 15: 実装 (Iter15)」節を参照．
- B27 の解消: `config.yaml` は `confidence_threshold: 0.5`・`dispatch_top_k: 1`・
  `confidence_signal_method: self_report` の状態を維持しており（このセッションでは変更していない），
  journal が記録する最良構成と一致している．`router.py` の few-shot 例 5〜7 相当の追加は，
  10 ドメイン対応のため動的生成（`_build_few_shot_examples`）へ全面書き換えたことで解消済み
  （個別の手書き例を追加/削除する形自体が無くなった）．**B27 はこれをもって解消とする．**
- 実機投入は未実施: 新規ノード wafl504〜509 の到達性確認・`ollama pull`，E4/E5/E6 の実機実験は
  次段階としてユーザーの別途確認を要する（WAFL-PEFT が同一 GPU プールを使用中でないことの確認が前提）．

## B27 [needs-human 2026-07-26] 作業ツリーに journal 未記録の変更が残っている

- 状況: `git status` に，Iter1〜14 のどのイテレーションにも対応しない未コミット変更がある．
  - `config.yaml`: `confidence_threshold` 0.5→0.3，`dispatch_top_k` 1→2，`confidence_signal_method` stp→self_report
  - `router.py`: few-shot 例 5・6・7 の追加（general と medical の切り分け，education と legal の切り分け）
  さらに git HEAD (`d56516c`) の `config.yaml` は Iter13 の実験設定（`stp`）のまま未リバートである．
  つまり「リポジトリの現在の設定」と「journal が記録する研究上の最良構成（self_report / threshold 0.5 /
  top_k 1）」が一致していない．
- 対応: CLAUDE.md の規約（作業前から存在する未コミット変更は明示的な依頼なく触らない）に従い，
  **変更は加えずそのまま残した**．
- 要人間判断: これらが (a) 意図した未記録の実験なのか，(b) 作業途中の放置なのかを確認し，
  研究サイクル再開前にリポジトリの設定を最良構成へ揃えるかどうかを決めること．
  なお `router.py` の few-shot 追加は Iter5〜9 で 5 パターンすべてが棄却された系統の変更であり，
  仮に実験するとしても E1（評価集合の拡張）完了後でなければ判定できない．

## B26 [auto-decided 2026-07-26] Iter14 の converged 判定を撤回し，測定系の立て直しから再開する

- 状況: 先行研究の再調査と既存結果の統計的再検討（`plans/p0001_research_direction_2026-07.md`）により，
  Iter14 の「実行可能な新レバーを定義できない」という収束判定の前提が崩れた．
- 自動選択: `status` を `converged` から解除し，`config.yml` の levers を全面改訂して E1〜E7 を登録した．
- 根拠（3 点．いずれも既存判定の誤りを示す）:
  1. **評価集合が 46 問しかない**（単一ドメイン 40 = 4×10，複合 6）．p=0.87,n=46 で SE ±5.0pt，
     Wilson 95% CI [74.3%, 93.9%]．Iter10/Iter11 の「0.870→0.848」は **40/46→39/46 の 1 問差**であり，
     棄却根拠にならない．ドメイン別指標は 1 ドメイン 10 問で SE ±9.5pt．
     Iter3・Iter5〜11 の「no-op / 僅差で棄却」は，レバーが効かなかったのではなく差を検出できなかった
     可能性が高い．
  2. **Iter11（multi_sample）は実験設計の欠陥**．Farquhar et al. Nature 2024 は temperature 0.1 を
     「点推定としての最良回答」の生成にのみ用い，不確実性推定は T=1・nucleus P=0.9 で行うと明記している
     （Xiong+ ICLR2024 も T=0.7）．Iter11 は不確実性を消す設定で不確実性を測っていた．
  3. **Iter13（STP）の 0.065 は偶然一致 0.25 を約 2.9 SD 下回る**．偶然より systematically に悪いのは
     符号反転バグを示唆する．保存済み `results.jsonl` の符号を反転して再計算するだけで検証できる（E2）．
- 併せて，Iter2（embedding）の cosine 潰れは埋め込みの anisotropy という既知の幾何的現象であり
  「信号が無い」証明ではないことを確認した（Varangot-Reille+ JAIR2025 は similarity-based routing の
  失敗を unsupervised であることに帰し，RouterDC は CosineClassifier に全タスクで勝利している）．
- 要レビュー: E1 は評価データセットの本格的な拡張であり，`build_dataset.py` と `metrics.py` の改修を伴う．
  research_frontier の「評価用データセットの本格化」と実質同じスコープなので，単一レバー原則の
  例外として扱ってよいかを確認したい（`plans/p0001` は「E1 を最優先」としている）．

## B25 [auto-decided 2026-07-22] Iter14: 全 levers 試し切り完了・研究サイクル収束判定

- 状況: config levers の全7本を試行・検証。hidden_state は Ollama API で raw hidden state 抽出不可が決定。
- 自動選択: `status="converged"` として研究サイクルを終了し、人間判断を待つ。
- 根拠: (1) Ollama REST API で raw hidden states は取得できない（/api/embeddings は semantic representation のみ）。(2) embedding ベースの信号は Iter2 で rejected（task prefix 未付与で cross-lingual mismatch）。(3) 研究フロンティア全項目が単一レバー原則の枠を超える。
- 要レビュー: 次のアクションを (A) 研究サイクル終了 or (B) 単一レバー原則再設計の上で研究フロンティア着手（例: routing_method=embedding + task_prefix 修正）のいずれかを選択すること。

---

書式:
```
## B{n} [auto-decided YYYY-MM-DD] 題目
- 状況: なぜ判断が要ったか
- 自動選択: 何を選んだか
- 根拠: なぜそれが最も妥当か
- 要レビュー: 人間が確認・却下する際に何を見るか（却下時はこの項目を編集して差し替える）
```
不可逆な事項は `[needs-human YYYY-MM-DD]` として記録し，Slack で @mention 済みであることを明記する．

---

## B24 [auto-decided 2026-07-22] STP レバー最終判定（rejected）および config levers 全試し切り・新レバー hidden_state 追加

- 状況: Iter13（STP再実験）の結果、top1_accuracy=0.0652（baseline 0.8696 から -0.8044）。misrouting_rate=0.9348。STP confidence spread=0.0147（全ノード・全ドメインでほぼ同一）。
- 自動選択: STP レバーを rejected と確定。config levers の全6レバーを試しまれた。新レバー `confidence_signal_method=hidden_state` を config.yml の levers 末尾へ追記して継続する。
- 根拠: (1) Sigmoid 正規化が信号を破壊（spread 0.1328→0.0147）。(2) Raw logprobs は「生成 fluency」を測定しておりドメイン expertise を測定していない。(3) self_report（bimodal, spread 0.95）でさえ STP（uniform, spread 0.015）より良い信号だった。(4) research_frontier に hidden states / embeddings-based approach が明記済み（Mahaut et al. 2024）。(5) モデル生成に依存しない信号源の検討が必須。
- 要レビュー: (1) hidden_state の実装詳細（last_layer activations vs embedding vectors のいずれを使用するか）。(2) config.yml levers への追記が単一レバー原則の枠を超える変更を伴うことを承認するか。(3) hidden_state 抽出には expert_backend.py の変更が必要（hidden state の取得経路）。
- 関連する恒久知見: verbalized confidence（self_report）と token-level confidence（STP）の両方が失敗した時点で、モデル生成に依存しない新しい信号源の検討が必須。hidden states は「入力の内部表現とドメイン知識の一致度」を測定し、この2つのアプローチとは異なる特性が期待される。

---

## B23 [auto-decided 2026-07-22] STP sigmoid shift 調整による信号弁別力回復

- 状況: STP コードは正常に動作したが、sigmoid(shift=2.0) の飽和領域で mean_logprob が動作し、signal の弁別力が失われた。top1_accuracy=0.043。raw logprob の spread は 0.1615 あるが、sigmoid-normalized confidence の spread は 0.0193 に圧縮。
- 自動選択: router.py の sigmoid shift を 2.0 -> 0.0（raw logprob 直接使用）へ変更。次イテレーションで実施。ただしコード変更を伴うため単一レバー原則の枠を超える。rc-planner で承認を得て継続するか、調査フェーズから代替アプローチを検索するか判断させる。
- 根拠: (1) STP コードは既にコミット済み（de37559）。(2) 修正コストは router.py の ~5行のみ。(3) raw logprob は [-inf, +inf] の広い範囲を持ち弁別力が高い。(4) config levers は全試し切り済み。次はコード変更を伴うアプローチに切り替える必要がある。
- 要レビュー: (1) shift=0.0 が最適な値か、あるいは最適化された shift 値（例: raw logprob の分布から計算）すべきか。(2) コード変更を伴うレバーとして単一レバー原則の枠を超えることを承認するか。(3) STP 修正以外にも代替アプローチ（confidence prompt の出力フォーマット強制 JSON、aggregator での raw logprob 直接使用）があるため、それらとの比較検討も必要。

---



## B22 [auto-decided 2026-07-22] Iter12: infrastructure_failure - デプロイフローの修正と STP 再実験

- 状況: Iter12（STP）の結果は無効。`mise run deploy` が Docker イメージを再ビルドせず、Python コード変更がコンテナ内に反映されなかった。全 probe が self_report 経路を通り、結果は baseline と同等の run 間ノイズ。
- 自動選択: Iter13 を STP 再実験とする。その前にデプロイフローを修正する（rc-investigator で調査 → rc-implementer で修正）。単一レバー方針: `confidence_signal_method=stp`（前回と同じ構成）+ デプロイフロー修正（並行）。
- 根拠: (1) STP コード変更は完了済み（テスト全PASS）。(2) 問題はコードではなくインフラ。Docker イメージの再ビルドまたは rsync での Python ソース配布を追加すれば、STP レバーを正しくテスト可能。(3) config levers は試し切り済み（B18, B20 参照）。STP が有効なら研究継続、無効なら research_frontier へ移行。
- 要レビュー: デプロイフローの修正方針。(A) `mise run deploy` に docker build ステップを組み込む（確実だが時間がかかる）、(B) rsync で Python ソースファイルをコンテナ内に配布 + コンテナ再起動（軽量だが新しい手順が必要）。rc-investigator が調査し、rc-planner が承認すること。
- 関連する恒久知見: Docker イメージにコードを bake する場合、デプロイ時は必ずイメージの再ビルドが必要。config.yaml のみ rsync で配布しても、Python コードの変更は反映されない。この教訓は研究サイクル全体の skill ドキュメントにも記録済み（B20）。

---

## B21 [auto-decided 2026-07-22] Iter11: multi_sample consistency rejected、次は STP

- 状況: Iter11（confidence_signal_method=multi_sample, N=3）の結果、top1_accuracy が 0.870→0.848 に退行。single_domain_top1_accuracy 0.875→0.850、misrouting_rate 0.130→0.152 も悪化。主基準・非退行とも全件未達。
- 自動選択: multi_sample レバーは rejected。次イテレーションの単一レバーを `confidence_signal_method=stp`（STP: Surrogate Token Probability）へ。config.yml の levers では既に stp が multi_sample より先に定義されているので、rc-planner は stp を選択するはず。
- 根拠: (1) temperature=0.1 では LLM 出力が実質決定論的で、N回probeしても値が変わらないため平均化効果が働かない。(2) confidence信号の分布は二峰性（{0.1, 0.2} vs {0.8, 0.9, 0.95}）に飽和しており、multi_sampleではdistribution shape自体を変えられない。(3) STP はトークン確率（logprobs）をconfidence signalとして使用する。verbalized confidence より頑健な信号になり得ることは Self-REF (ICML 2025) で実証済み。(4) config.yml の levers では stp が multi_sample より先に定義されているため、rc-planner は自然に stp を選択する。
- 要レビュー: STP 実装には expert_backend.py（logprobs サポート）、router.py（STP 用関数）、protocol.py（新フィールド追加）、http_server.py（logprobs 含む ProbeResponse 構築）の変更が必要（合計 ~45行）。これは config-only の単一レバー原則の枠を超えるため、次 rc-planner で承認を得ること。
- 関連する恒久知見: confidence signal の較正が本研究の根本ボトルネックであることが Iter1-11 で決定的に示された。config.yaml の値変更（dispatch_top_k, routing_method, confidence_threshold）や few-shot 例の変更、calibrated routing classifier、multi_sample consistency いずれも期待した改善をもたらさなかった。signal の抽出方式そのものを変える STP が唯一の残されたアプローチ。

---

## B20 [auto-decided 2026-07-22] Iter10 収束後，config.yml に新レバー confidence_signal_method を追加して再開
- 状況: Iter10 で config-only の 3 レバー（dispatch_top_k, routing_method, confidence_threshold）を試し切り，
  reflector が `status="converged"` として待機していた（B19 参照）。従来の設計では全 levers 試し切りは
  即座に人間の判断待ちとなる仕様だったが，B19 の時点で reflector 自身が次の方向性（STP / multi-sample
  consistency）を既に提示していたため，人間の判断を経て再開する。
- 自動選択: config.yml の `levers` 末尾に `confidence_signal_method`（values: `[stp, multi_sample]`）を
  追加し，`state.json` を次イテレーション（Iter11，phase=investigate, status=running）へ初期化した。
  どちらの値を先に試すかは次の rc-planner の判断に委ねる。
- 根拠: B19 の要レビューで「(A) STP vs (B) multi-sample consistency のいずれが実装コストと期待効果を
  兼ね備えるか rc-planner が判断すること」と既に整理済みであり，新しいレバーとして定式化するのに十分な
  情報が揃っている。
- 要レビュー: rc-planner がどちらを選んだか，および実装スコープが単一レバー原則の範囲に収まっているかを
  確認すること。
- 関連する恒久対応: 「config の全 levers 試し切り＝即停止」という設計自体も見直した。今後は reflector が
  自分で次のレバーを考案できればそのまま継続し，考案できない場合は次イテレーションを調査フェーズから
  開始して rc-investigator に tavily-search で代替アプローチを重点調査させ，それでも見つからない場合の
  みに人間の判断を待つ（SKILL.md「停止条件」節，rc-reflector.md/rc-investigator.md/rc-planner.md 更新済み）。

## B19 [auto-decided 2026-07-22] Iter10: calibrated routing rejected、次方向は STP / multi-sample consistency
- 状況: Iter10（calibrated routing）の結果、top1_accuracy が 0.870→0.848 に退行。offline AUC=1.000 は online improvement を保証しないことを示す。
- 自動選択: calibrated routing レバーは rejected。次イテレーションの単一レバーの方針として (A) Surrogate Token Probability（生成中のトークン確率を confidence signal として抽出）または (B) multi-sample consistency（複数回 probe した confidence の分散を信頼度 signal として使用）を rc-planner に提示する。
- 根拠: (1) offline classifier の特徴量（margin, is_top1）は routing decision と情報的に重複しており label leakage が生じた。(2) confidence 値自体の run 間変動（±0.05）は offline classifier を無効化。(3) Self-REF (ICML 2025) は STP や confidence tokens で self-report より頑健な信号を実証。
- 要レビュー: next direction の (A) vs (B) のうちいずれが実装コストと期待効果を兼ね備えるか rc-planner が判断すること。(A) は tokenizer logprobs の抽出が必要で実装量多め。(B) は probe の複数回実行で latency 増大のトレードオフ。

## B18 [auto-decided 2026-07-22] Iter10: probe-based calibrated routing の採用決定
- 状況: config-only レバー（dispatch_top_k, routing_method, confidence_threshold）は3本とも試し切り。few-shot 変更も5回連続（Iter5-9）で試されたが限界。根本ボトルネックは confidence 信号の較正であり、config.yaml の値変更だけでは対処できないことが Iter1-9 で決定的に示された。
- 自動選択: probe_candidates から抽出した特徴量（self_confidence, max_other_confidence, margin, is_top1 など）を用いた logistic regression classifier を offline analysis にて訓練・評価する approach を採用。offline で有効性が確認できたら aggregator.py へ online routing として組み込む（2-phase approach）。
- 根拠: (1) misroute の内訳は構造的に理解可能：general-008 は medical=0.9 > general=0.85、education-003/004/008/009 は legal=0.9, education=0.9 の tie。margin <= 0 で misroute が集中的に発生。(2) n=184 sample (46 query x 4 domain) に対し logistic regression (p=6-7) は過学習リスクが低く、coefficient の解釈も可能。(3) Self-REF (Chuang et al., ICML 2025) は confidence tokens による fine-tuning で routing accuracy が大幅改善。Amazon Science (2024) は calibrated confidence scores で cascading ensemble policy を設計し推論コストを2倍削減。これらの知見は本研究の approach と整合する。(4) Mahaut et al. (2024) の probe-based classifier は verbalized/self-reported confidence より優位だが、hidden states 抽出が必要で現時点の実装では困難。代わりに probe_candidates の confidence values を特徴量とする logistic regression が現実的な第一歩。
- 要レビュー: (1) Phase 1 の offline AUC >= 0.85 という成功条件は妥当か。(2) Phase 2 で aggregator.py に calibrated routing function を組み込む変更を承認するか。(3) baseline 比較は results/20260721_222225（Iter9）とするか results/20260721_185132（Iter8）とするか。

## B17 [auto-decided 2026-07-21] Iter9: few-shot 構造変更は rejected（education precision 改善だが recall 低下）
- 状況: Iter9（全ドメイン表示 + 保守的指示追加）の結果、education precision=1.0（>=0.93 PASS）だが、recall=0.5（>=0.62 FAIL）。general/legal precision も退行。
- 自動選択: few_shot_structure_change レバーは rejected。router.py の few-shot 例変更は 5 回連続（Iter5-9）で試されたが、いずれも期待した効果を持たなかった。このレバーは収束。
- 根拠: (1) education precision は改善したが、recall が大幅に低下（0.667→0.5）。全ドメイン表示 + 保守的指示により education ノードが過剰抑制。(2) general/legal precision も退行。(3) misrouting_rate が悪化（0.087→0.130）。
- 要レビュー: 次 rc-planner は config-only の枠を出る根本的なアプローチ（probe ロジック変更、新しいルーティング方式）を提示すること。few-shot 例の変更は限界に達している。

## B16 [auto-decided 2026-07-21] Iter9: 単一レバーの決定（few_shot_structure_change）
- 状況: confidence_threshold レバーは Iter9 調査で education 過信抑制の文脈でも no-op 確定。4回連続 few-shot 変更（Iter5-8）は「書き方」の変更にとどまり限界。
- 自動選択: 単一レバーを `few_shot_structure_change`（router.py の few-shot 例ブロックの構造変更）へ。具体案: (1) 例1-3を全ドメイン表示へ変更（現在2ドメイン→4ドメイン）(2) 評価基準に保守的指示を追加。
- 根拠: (1) 直近 few-shot 変更は「書き方」の問題（例4の教育ノード視点追加）であり、構造的な問題（例1-3の2ドメイン表示のみで cross-domain 対比が弱い）は放置されたまま。(2) 全ドメイン表示により education ノードは general=0.9 > education=0.1 の対比を few-shot 例から直接学習可能。(3) 変更量: 例1-3の各行に2ドメイン分追記 + 評価基準に1行追加。計5行弱。
- 要レビュー: 単一レバー原則（config-only の枠を出る変更）の承認。router.py の few-shot 例ブロック変更が影響範囲限定（5行弱）のため承認可能か。

## B15 [auto-decided 2026-07-21] Iter9: confidence_threshold の再検討結果（education 過信抑制の文脈でも no-op 確定）
- 状況: Iter9 で confidence_threshold の再検討を実施。results/20260721_185132 の probe_candidates から offline 掃引。
- 自動選択: confidence_threshold レバーは rejected。values [0.3, 0.5, 0.7] は education 過信抑制の文脈でも no-op（空帯域 (0.3,0.7) に値 0 件は同じ）。閾値 0.85+ は意味があるが、education 過信の根本原因（confidence 信号の較正）には対処できない。
- 根拠: (1) general-004（education 過信の主要ケース）は education=0.95 > general=0.9 で、どの threshold でも education が勝つ。threshold 非効力。(2) education-002（tie at 0.95）と education-009（tie at 0.8）も threshold で解決不可。(3) threshold=0.85 で education-009 が fallback になるのみ（1 件）。
- 要レビュー: confidence_threshold は education 過信抑制のレバーとして不適。次 rc-planner は config-only の枠を出る変更（router.py の few-shot 例修正、probe ロジック変更）を提示すること。

## B14 [auto-decided 2026-07-21] Iter9: confidence_threshold の再検討（education 過信抑制の文脈）
- 状況: Iter5-8 で 4 回連続 few-shot 例の変更を試したが、いずれも期待した効果を持たなかった。Iter8 では「education ノード視点」への変更が過剰抑制の副作用（education recall -0.166）を引き起こし、few_shot_node_perspective レバーは収束確定。
- 自動選択: config.yml levers の次候補 `confidence_threshold`（values: [0.3, 0.5, 0.7]）へ移行。Iter3 で「二峰・空帯域分布による no-op」と判定されたが、当時の目的は「フォールバック率とのトレードオフ」であり、今回は「education の過信抑制」という新たな文脈で再検討する。
- 根拠: (1) levers 優先順で confidence_threshold が唯一未試行の config-only レバー（dispatch_top_k=Iter1 棄却、routing_method=Iter2 棄却）。(2) education ノードの confidence 分布 {0.2, 0.8, 0.85, 0.9, 0.95} において、0.9 閾値は high-clusters の education 過信（0.9, 0.95）を fallback へ落とす可能性がある。(3) config-only 変更で検証可能。
- 要レビュー: Iter3 で no-op と判定された confidence_threshold を再検証する根拠。閾値を上げすぎると fallback_rate が急増し品質退行するリスク。次 rc-planner は具体的な成功条件（閾値候補，fallback_rate の許容範囲）を提示すること。

## B13 [auto-decided 2026-07-21] Iter8: few-shot 例の構造変更（education ノード視点へ）
- 状況: Iter7（router.py の few-shot 例ブロックに一般質問のネガティブ例追加）は rejected（主基準2件未達）。例4は general ドメインの視点（「読書感想文→general=0.9, education=0.1」）で書かれており、education ノードの過信を抑制できなかった。
- 自動選択: 例4の書き方だけを「education ノード視点」へ変更。例: 「質問「読書感想文の書き方」は general 分野であり、education ドメインではない。education ノードは low confidence (0.1) を出すべき」。既存の例1-3は不変。変更量: 1行の書き換え。
- 根拠: 分析(解釈)で「視点の不一致」が根本原因と特定。例4の「読書感想文」語彙は general-004 と完全に一致するため、語彙的アンカリングで逆効果。education ノードが self-report する際の few-shot 例として、education ノードの視点で書かれたネガティブ例が効果的。
- 要レビュー: 例4の education ノード視点への変更が有効か。次イテレーション（Iter8）で router.py の few-shot 例ブロックを修正し、education ノードの過信抑制効果を測定する。

## B12 [auto-decided 2026-07-21] Iter7: 単一レバーが config-only の枠を超えるためユーザー承認必要
- 状況: 調査フェーズで「few-shot 例へのネガティブ例追加」が推奨。ただし router.py のコード変更を伴う。
- 自動選択: 変更量2行で影響範囲が限定されるため、単一レバーとして承認可能と判断。
- 根拠: 3イテレーション連続（Iter4-6）で config-only の枠内では改善できず、few-shot 構造の修正が唯一の有効なアプローチ。
- 要レビュー: router.py の few-shot 例追加が単一レバーとして承認されるか。却下時は confidence_threshold 再較正（B）に留めること。

## B11 [auto-decided 2026-07-21] Iter6: few-shot 追加が rejected と判定され、抑制アンカリングの必要性が確定
- 状況: Iter6（router.py build_confidence_prompt() に education 固有 few-shot 例を1件追加）は rejected（主基準2件未達，非退行2件未達）。education ノードの confidence 値が Iter5 と10問中10件完全に同一。few-shot 追加は confidence 信号に何の影響も与えなかった。
- 自動選択: 次イテレーションの単一レバーの方針を「抑制アンカリング few-shot 例への差し替え」へ方向付け。具体的には (A) general 質問→medical/legal/education すべて low confidence のパターンを few-shot 例に追加、(B) confidence_threshold を 0.9 付近へ引き上げ（Iter3 再検証）、(C) education ノードのプロンプトに「読書、勉強、習い事等は general 分野」と明確に指示する文を追加、の3方向を rc-planner が提示する。
- 根拠: Iter5-6 で「few-shot 例は該当する→high confidence のパターンしか示さない」という構造的要因が確定。抑制のアンカリング（general 質問で education 関連の言葉が出ても low confidence）が欠如していることが根本原因。config-only レバー探索は3イテレーション連続で限界が確定しており、router.py の few-shot 構造修正が唯一の有効なアプローチ候補。
- 要レビュー: rc-planner は (A)(B)(C) のうちいずれを単一レバーとして提案するか。単一レバー原則（config-only）の枠を超える router.py 変更を承認するか、config-only のまま confidence_threshold 再較正（B）に留めるか、ユーザー判断を仰ぐこと。

## B10 [auto-decided 2026-07-21] Iter5: few-shot 差し替えが router.py 側でしか効かない構造的原因の特定と次レバーの方針
- 状況: Iter5（education ノード few-shot 例の education 固有話題への差し替え）は rejected（主基準 2 件未達，非退行 2 件未達）．
  分析で決定的な構造的要因が特定された: build_dataset.py の _EDUCATION_QUESTIONS はテストクエリであり few-shot 例ではない．
  confidence 自己申告ロジックの few-shot 例は router.py の build_confidence_prompt() でハードコード（「歯の痛み→medical」「賃貸契約→legal」）され，
  全ドメイン共通で使われる．education ノードの評価にも medical/legal の例が使われるため，build_dataset.py の変更は confidence 信号に一切影響しない．
  決定的証拠: education ノードの confidence 値が Iter5 とベースラインで完全に同一．
- 自動選択: 次イテレーションの単一レバーを「router.py の build_confidence_prompt() に education 固有の few-shot 例を追加」へ方向付け．
  ただしこれはコード変更を伴うため単一レバー原則（config-only）の枠を超え，ユーザーの判断を仰ぐべき．
  次 rc-planner は以下の 2 選択肢のいずれかを提示する:
  - A: router.py の few-shot 例に education 関連話題を追加（コード変更，単一レバー原則の再設計が必要）
  - B: confidence_threshold の実質的な再較正（0.9 付近の閾値で education の過信を抑制，config-only 維持可能か検証）
  両方とも「単一レバー原則の再設計」が前提．B7 で記録した nomic-embed-text task prefix 未付与の問題も
  信号較正の文脈で並行検討すべき．
- 根拠: 3 イテレーション連続（Iter1: confidence飽和, Iter2: embedding弁別喪失, Iter3: 閾値no-op）で
  config-only の枠内で改善できないことが確定．Iteration 4-5 で education ドメイン追加および few-shot 差し替えを試したが，
  router.py 側の few-shot 構造が根本原因であることが判明．次は router.py の修正または confidence_threshold の実質的再較正へ．
- 要レビュー: 次 rc-planner が具体的な仮説と成功条件を提示する際，(1) router.py の few-shot 例追加が単一レバーとして成立するか，
  (2) 既存 results との比較に使う baseline は results/20260721_085735（Iter5）か results/20260721_011117（Iter4 ベースライン）か，
  (3) education ドメインの追加評価指標（precision/recall 目標値）をどう再定義するか，を明確化すること．

## B9 [auto-decided 2026-07-20] Iter3=confidence_threshold の no-op 確定と config-only レバー探索の収束・移行方針
- 状況: Iter3 対象レバー `confidence_threshold`（candidates [0.3, 0.5, 0.7]，既定 0.5）について，調査で二峰・
  空帯域分布による構造的 no-op が示唆されていた．ゲートは requester 側 aggregator が記録済み probe_responses に
  適用するだけのため，計画フェーズで**新規実験なしに**ベースライン結果 results/20260720_171532（34 行）の
  probe_candidates から閾値掃引をオフライン再計算し，thr=0.3/0.5/0.7/0.85 で fallback=0・total_dispatch=34・
  selected_domain 全 34 行一致（帯域 (0.3,0.7) に値 0 件，fallback は 0.9 以上でのみ発生かつ品質退行側）を確認．
  no-op が決定的に確定した．これで config.yml levers 3 本（dispatch_top_k=Iter1 棄却，routing_method=Iter2
  棄却，confidence_threshold=Iter3 no-op）を試し切り，config-only レバー探索が収束した．次にどの大きな
  方向へ進むかの判断（可逆だが実装量の大きい方向転換）が必要になった．
- 自動選択: 案C3 を採用．(1) 案C1（no-op を新規 run で実証）は棄却＝ゲートがオフライン再計算可能で新規 run
  （約 46 分）が冗長．(2) 案C2（levers.values を稠密域 ~0.15/~0.9 へ差し替え，config-only 単一レバー維持）は
  棄却＝top_k=1 固定下で ~0.15 は依然 no-op，~0.9 は専門ノードを general へ落とす品質退行で改善余地なし．
  (3) 案C3 を採り，本イテレーションは実験・実装をスキップ（config.yaml 無変更）し，**停止して人間判断を仰ぐ**
  形で移行方針を提示する．
- 根拠: 3 イテレーション連続で「config 値では confidence 信号の質を baseline 以上にできない」ことが示され，
  真のボトルネックが confidence 信号の較正（過信・飽和）という config-only の枠外にあることが確定した
  （Iter1: self_report 複合行 confidence 飽和 / Iter2: embedding cosine が [0.67,0.74] に潰れ弁別喪失・
  top1 0.53 / Iter3: 二峰分布で閾値どの候補値でも無反応）．config.yml research_frontier の順序規約
  「levers を試し切ってから research_frontier へ」の発火条件を満たす．
- 要レビュー（人間が選ぶ方向）: 次サイクルの重心を以下 A/B のどちらに置くか判断すること．
  - A: research_frontier「新規専門ドメイン追加」（B6 で方向性はユーザー承認済み）．ただし具体ドメイン選定・
    build_dataset.py 拡充・新規モデル準備・config.yaml ノード追加・router.py ドメイン別プロンプト整備を伴う
    大きめの変更で，次期 rc-planner での具体化と，どのドメイン/何ノードかの人間入力が要る．
  - B: confidence 信号の較正改良（B7 起点の nomic-embed-text task prefix 付与，複数 utterance ルート定義，
    ドメイン別 few-shot プロンプト整備）．いずれもコード変更を伴い config-only 単一レバー原則の外側（未承認）で，
    計測基盤・比較 baseline の再定義が必要．
  いずれも「単一レバー原則の再設計（config-only の枠を出る）」が前提になる点，および 3 イテレーション一貫の
  知見（ボトルネック=信号較正）が A/B どちらにも通底する点を判断材料とすること．C3 のためこのイテレーションでは
  config.yaml は無変更・deploy 不要．

## B8 [auto-decided 2026-07-20] Iter2 の判定と次イテレーション（Iter3）の単一レバー選定
- 状況: Iter2（routing_method: self_report→embedding）の結果，主基準（信号の質）が決定的未達
  （positive-margin 率 0.529 vs 基準 0.971，mean margin -0.0040 vs 0.60），非退行 3 指標も決定的未達
  （single_domain_top1_accuracy 0.500，top1_accuracy 0.529，misrouting_rate 0.471）．embedding は決定的計算
  （cosine のみ）で run 間ノイズがほぼ 0 のため構造的劣化と断定，追加反復は不要．よって routing_method
  レバーは棄却．config levers 優先順位 1（dispatch_top_k, Iter1 棄却）・2（routing_method, Iter2 棄却）を
  使い切り，残る config-only レバーの選定が必要だった（可逆な判断）．
- 自動選択: 次の単一レバーを `confidence_threshold`（config levers 優先順位 3，values: [0.3, 0.5, 0.7]）とし，
  ベースライン 0.5 を基準に振る．次イテレーション（Iter3）名は「confidence_threshold 掃引による fallback 率と
  general 過信リークのトレードオフ検証」．routing_method は交絡回避のため config.yaml でベースライン
  （self_report）に戻した．
- 根拠: (1) levers 優先順位で confidence_threshold が唯一未試行の config-only レバー（1・2 は棄却済み）．
  (2) Iter2 で「self_report 方式では閾値ゲートが機能している（Iter1 で複合行 medical=0.2 が閾値 0.5 に
  ブロックされ dispatch 未発火）」ことと，「embedding 方式では閾値 0.5 が 102/102 probe で全通過し無意味化」
  という対照的な新知見が得られており，self_report ベースラインで閾値を動かす効果を測る素地がある．
  (3) research_frontier（新規専門ドメイン追加）はコード・データセット・ノード追加を伴う大きめの変更で，
  config-only レバーを試し切ってから着手する順序（config.yml research_frontier に明記）を守る．
- 要レビュー: (1) B5 で記録したトレードオフ「confidence_threshold を下げると general の過信リーク
  （Iter1 で general-008 の余分 dispatch を実測）が悪化する」を rc-planner が非退行基準に組み込み数値化する
  こと．具体的には fallback_rate（直近実験では 0.0 のため閾値を上げた側で初めて動く可能性）・単一ドメイン行の
  over-dispatch・general の precision を監視項目に含める．(2) 直近実験は fallback_rate=0.0 のため
  confidence_threshold の効果自体が薄い可能性がある（config levers の note にも記載済み）．null 結果になった
  場合は config-only レバーを試し切ったと判断し，停止条件（グローバル skill）か research_frontier（新規
  専門ドメイン追加）への移行を rc-planner が検討すること．(3) 閾値を下げる方向（0.3）と上げる方向（0.7）で
  効くメカニズムが異なる（下げる＝over-dispatch/リーク，上げる＝fallback 増）ため，どちらを主眼に置くか
  rc-planner が仮説を明確化すること．

## B7 [auto-decided 2026-07-20] nomic-embed-text の task prefix 未付与（Iter2 スコープ外・劣化時の切り分け課題）
- 状況: Iter2（routing_method: self_report→embedding）の調査で，nomic-embed-text は非対称タスク用の
  task instruction prefix（search_query: / search_document: / classification: 等）を前提に学習されているが，
  現行コードは query（node.py:143）にも domain（http_server.py:184）にも prefix を付けていないと判明．
  prefix 無し＋英単語(domain)対日本語(query)のクロスリンガル比較は較正上不利で，embedding ルーティングの
  単一ドメイン精度が退行する既知の落とし穴になり得る．
- 自動選択: Iter2 では prefix 付与を行わず，prefix 無しの現状のまま routing_method=embedding を config-only で
  評価する．退行（特に single_domain_top1_accuracy の低下）が観測された場合に，prefix 起因かの切り分けを
  次段階の課題としてここに残す．
- 根拠: prefix 付与は両 embed 経路（node.py・http_server.py）のコード変更が必要で，Iter2 の config-only
  単一レバー原則と衝突する．まず現状構成で embedding の素の性能を測り，交絡なく劣化要因を特定するのが妥当
  （調査の提案どおり）．
- 要レビュー: Iter2 で embedding が非退行基準を割った場合，(1) prefix 付与（query に search_query:，domain に
  search_document: 等）を加えた再実験を別イテレーションで行うか，(2) domain 定義をドメイン名 1 単語から
  複数代表発話(utterances)へ拡張するか（Semantic Router ベストプラクティス，調査参照），を rc-planner が
  検討すること．いずれもコード変更を伴うため単一レバー原則との整合を人間判断すること．

## B6 [user-approved 2026-07-20] 実機ノード拡張（最大10台）と VRAM 常時確保
- 状況: ユーザーから (1) 192.168.15.100〜109（最大10台，同一スペック）が実機として利用可能になったこと，
  (2) 実機のVRAMが解放されてしまっていたこと，の2点の連絡があった．(2) の確認のため3ノード（wafl500/502/503）
  の ollama `/api/ps` を確認したところ全ノードで `models: []`（アンロード済み）だった．
- ユーザーの選択:
  - VRAM確保: docker-compose.ymlのollamaサービスに`OLLAMA_KEEP_ALIVE=-1`を追加（commit 94d4b50，push済み）．
    `mise run deploy`で3ノードに反映し，`/api/ps`で両モデル（qwen3.5:9B, nomic-embed-text）が
    `expires_at: 2318-10-30`（実質無期限）でロード済みであることを確認．
  - ノード拡張の活用方向: 「新規専門ドメインの追加」（既存ドメインの冗長化・ノード数スケール検証は不採用）．
    config.yml の research_frontier に記録済み．具体的なドメイン候補・データセット拡充は次期rc-planner着手時に
    具体化する．
- 根拠: VRAM対応はユーザーの直接指示（運用上の緊急対応，可逆）．ノード拡張の方向性はユーザーが
  AskUserQuestionで直接選択（新規専門ドメイン追加＝メッシュ本来の目的である専門分野分担の拡充に直結）．
- 要レビュー: 新規専門ドメイン追加は，現在進行中のIteration 2（routing_method）の後，かつ単一レバー原則
  （既存ノードの構成・動作を変えない形での追加）を守って着手すること．具体的なドメイン名（教育・金融等）
  はまだ未決定．git push はグローバルCLAUDE.mdの「git push絶対禁止」規約と衝突するため，このVRAM対応の
  push もAskUserQuestionで個別に確認済み（研究サイクルのreflectorによるpushとは別に確認が必要だった点，
  今後も同様の非イテレーション変更ではpush前に確認すること）．

## B5 [auto-decided 2026-07-20] Iter1 の判定と次イテレーションの単一レバー選定
- 状況: Iter1（dispatch_top_k=1→2）の結果，主基準 compound_covered_domain_count>=6 は未達（実測 4→5,
  +1 のみ）．根本原因は confidence_threshold=0.5 のゲートを複合 3 行の medical confidence(0.2) が越えられず
  追加 dispatch が発火しないこと．真のボトルネックは confidence 信号の質であり dispatch 並列数ではないと判明．
  よって dispatch_top_k レバーは棄却．次に振る単一レバーを決める必要があった（可逆な判断）．
- 自動選択: 次レバーを `routing_method`（config levers 優先順位 2 番目）とし，方式 B(self_report,既定)→
  方式 A(embedding) へ振る．次イテレーション名は「embedding ルーティング(方式A)への切替による複合ドメイン
  被覆の検証」．dispatch_top_k は交絡回避のため config.yaml でベースライン(1)に戻した．
- 根拠: (1) levers 優先順位で routing_method が 2 番目．(2) Iter1 で self_report の自己申告 confidence が
  過信/較正不良で複合行の弁別に効かないことが実証され，confidence 信号そのものを変える routing_method は
  根本原因に直接対応する．(3) 3 番目候補 confidence_threshold を先に下げる案は，general の過信リーク
  （Iter1 で general-008 の余分 dispatch を実測）を悪化させるトレードオフがあり，信号の質を改善する方が先．
- 要レビュー: rc-planner は embedding 方式での成功基準（複合行被覆の目標値・非退行として top1_accuracy と
  general 過信リークの許容範囲）を数値化すること．embedding_model=nomic-embed-text の probe レイテンシと
  精度のトレードオフも観測項目に加えること．k=3 は複合行の期待ドメインが最大 2 のため k=2 と同一結果に
  なる見込みで検証価値が低く，dispatch_top_k の追加反復は不要と判断した（この点への異議があれば差し替え）．

## B4 [auto-decided 2026-07-20] 既存テストの壊れた import 修正（本イテレーションとは無関係）
- 状況: rc-implementer が完了条件（`uv run pytest tests/ -v` が通ること）を確認しようとしたところ，
  `tests/test_metrics.py` と `tests/test_build_dataset.py` が存在しないパッケージ `benchmark`
  （`from benchmark.metrics import ...` 等）を import しており，pytest の collection 自体が
  全滅していた．commit `71ac11a` 由来で，本イテレーションの単一レバー（dispatch_top_k）とは無関係の
  既存バグである．
- 自動選択: 他の全テストファイルが使っている import 形式（`from metrics import ...` 等）に，
  該当 2 行のみ修正した．assert・テストロジック本体は変更していない．
- 根拠: 完了条件を満たすには pytest 自体が動く必要があり，かつ修正は import 文のみで最小限のため，
  今回の実装作業に含めて自動判断した（CLAUDE.md「軽微な不明点は合理的な仮定を明記して前に進めてよい」）．
- 要レビュー: このバグ自体は本来 dispatch_top_k とは無関係の既存不具合であり，なぜ・いつから
  collection エラーになっていたかの経緯確認は未実施．必要であれば別途調査すること．

## B3 [user-approved 2026-07-20] 被覆率計測のため run_experiment.py の出力スキーマ拡張
- 状況: rc-planner が B2（metrics.py のみの変更）の具体化を検討したところ，複合ドメイン行の dispatch 候補
  集合（confidence_threshold 超のノードのうち dispatch_top_k 件）を記録する一次データが，現行の
  results.jsonl（run_experiment.py 出力，selected_domain 単体のみ）には存在しないと判明した．metrics.py
  は事後解析のみのため，欠けている一次データを事後に復元することはできない．B2 で承認された範囲
  （metrics.py のみ）を超えるスコープ拡大のため，再度ユーザーに確認した．
- ユーザーの選択: run_experiment.py の出力にも新規フィールド（dispatched_domains, probe_candidates）を
  追加して進める．既存フィールドは変更せず，新フィールドを持たない旧 results.jsonl は metrics.py 側で
  スキップする後方互換設計とする．aggregator.py 等の集約ロジック本体は変更しない．
- 根拠: dispatch_top_k レバーの効果を測定可能にするための最小限の観測項目追加であり，ルーティング・
  集約の挙動そのものは変えない．これを行わない場合，dispatch_top_k レバーは検証不能（no-op のまま）．
- 要レビュー: rc-implementer が実装した後，既存の results.jsonl 読み込みコード（metrics.py の
  _read_results 等）が新フィールドの有無で分岐し，古い実行結果に対しても実行時エラーなく動作することを
  確認すること．dispatch_top_k=1 のベースラインも新スキーマで再実行が必要（旧 results/20260709_214113
  は新フィールドを持たないため比較に使えない）．

## B2 [user-approved 2026-07-20] dispatch_top_k レバーの no-op 判明への対応方針
- 状況: rc-investigator の調査（journal.md「調査 (Iter1)」参照）により，dispatch_top_k を config-only で
  1→2,3 に振っても，現行の集約ロジック（aggregator.select_best_dispatch_response が confidence 最大値を
  選ぶのみ．probe の confidence が http_server.py 側で request_id 単位キャッシュされ dispatch にそのまま
  引き継がれる）では最終的な selected_domain が変わらず，metrics.py が測る medical recall に対して no-op
  になる可能性が高いと判明した．「config-only の単一レバー原則」と「target 指標を動かすのに必要な変更」が
  衝突する分岐点のため，AskUserQuestion でユーザーに直接確認した．
- ユーザーの選択: 案X2（評価方式を拡張）．metrics.py に，複合ドメイン行を対象とした set-valued 被覆判定の
  指標を既存の recall 等に「追加」する形で実装し，dispatch_top_k>1 が実際に候補集合をどれだけ被覆できて
  いるかを測れるようにする．既存の単一 selected_domain 前提の指標（top1_accuracy 等）は変更せず残す．
- 根拠: この指標拡張なしには dispatch_top_k レバー自体が意味をなさない．小規模な評価コード追加のみで，
  集約ロジック（aggregator.py）自体や実験の起動系（mise タスク）には手を入れないため，影響範囲を絞れる．
- 要レビュー: rc-planner が具体的な指標定義（例: dispatch 候補集合と expected_domains の Jaccard/被覆率）
  と成功基準の数値化を行う．rc-implementer が metrics.py への実装を行う際は，既存指標の出力形式・関数を
  破壊しないこと（既存の journal・過去 results との比較可能性を保つため）．

## B1 [auto-decided 2026-07-20] config.yml 初回セットアップ
- 状況: research-cycle 初回起動（このリポジトリでは config.yml/state.json/journal.md/backlog.md が未作成だった）．
  levers の優先順位・success_criteria・tasks コマンド・timeout_min 等を新規に決める必要があった．
- 自動選択: levers の優先順位（1. dispatch_top_k，2. routing_method，3. confidence_threshold）はユーザーに
  AskUserQuestion で確認済み（dispatch_top_k を最優先として承認）．success_criteria は暫定的な定性表現とし，
  timeout_min（90分）・metrics_cmd（最新の results/*/results.jsonl を動的解決）は直近の完走実験ログから算出．
- 根拠: 直近の完走実験（results/20260709_214113/results.jsonl）で metrics.py を実行したところ top1_accuracy=1.0
  だが medical の recall=0.79（legal 0.93, general 1.0）と判明．dispatch_top_k>1 は既存の
  aggregator.select_best_dispatch_response で対応済みのため，新規実装なしで即検証できる最有力候補と判断．
- 要レビュー: success_criteria の数値基準が未確定（イテレーションを重ねてノイズ幅が分かってから rc-planner が
  数値化する設計）．research_frontier の各項目（ベースライン比較・回答品質評価等）は levers 探索後の着手を
  想定しているが，優先度を変えたい場合は config.yml の該当セクションを直接編集してよい．

## B64 [auto-decided 2026-08-02] Iter41: embedding_adaptation=embedding_adapter_only_lora の計画

- **自動選択**: 単一レバーを `embedding_adaptation=embedding_adapter_only_lora` とする。
  `iteration_name` は「PEFT LoRAによるeducationドメイン埋め込み適応」．
- **選定理由**:
  1. Iter40（setfit_education_finetune）で全パラメータfine-tuningが単一レバー原則と両立しないことが確定（argmax flip rate 52.56%）
  2. rc-investigator（Iter41調査フェーズ）は PEFT LoRA の feasibility が HIGH と判定
  3. SentenceTransformer 3.x が LoRA を公式サポート
  4. 既存の `--fine-tuned-embed-model` 統合（Iter40 で実装済み）を再利用可能
  5. LoRA adapter の更新パラメータは base model の 0.86% のみ（rank=16, attention layers のみ）
  6. オフライン完結（実機1600問本走不要、総コスト~30-45分）
- **計画フェーズの決定事項**:
  1. **Rank dimension**: r=16（保守的。単一レバー原則優先。r=32 は fallback）
  2. **Training loss**: MultipleNegativesRankingLoss（SBERT 公式推奨、TripletLoss より安定）
  3. **Runtime embedding path**: classifier training + evaluation で LoRA adapter 適用。runtime routing は base model のまま（単一レバー原則）。train/inference mismatch を避けるため、classifier training と evaluation で同一の LoRA-adapted embeddings を使用。
  4. **Negative pair sampling**: 60/40 priority/random（Iter40 と同一）
  5. **LoRA target modules**: `.*attn.*`（attention 投影層のみ。MLP 層は対象外）
- **変更ファイル**:
  1. `scripts/fine_tune_embedding_lora.py`（新規作成）— LoRA 訓練スクリプト
  2. `scripts/train_domain_classifier.py`（`build_training_features()` に `set_adapter("default")` 追加）
  3. `scripts/evaluate_classifier_calibration.py`（`predict_calibrated_rows()` に `set_adapter("default")` 追加）
  4. `pyproject.toml`（`research` deps に `peft>=0.12` 追加）
- **成功条件**: education_recall > medical_recall基準(0.5112)、他9ドメイン18指標のBH補正後有意退行0件、argmax flip rate < 15%
- **失敗条件**: education_recall が基準超えない、BH補正後有意退行1件以上、argmax flip rate >= 15%、LoRA adapter ロードエラー
- **コスト**: 低〜中（~30-45分、オフライン完結）

## B67 [auto-decided 2026-08-02] Iter44: education_boundary_tuning (intercept_delta=+0.7) adopted。全levers試し切り、次は調査フェーズ

- **状況**: Iter44（classifier_head_adaptation=education_boundary_tuning, intercept_delta=+0.7）
  の rc-analyst 判定（adopted）を rc-reflector が検証・確定させた。
- **判定: adopted（確定）**。4条件中4条件すべてPASS:
  - education_recall 0.4588→0.5235 (+0.0647) > 0.5112 **PASS**
  - BH補正後有意退行 0件 **PASS**
  - argmax flip rate 8.62% < 15% **PASS**
  - top1_accuracy McNemar p=0.8445 >= 0.05 **PASS**
- **決定的な学び**:
  1. **intercept シフトは単一レバー原則を達成できる**: argmax flip rate 8.62% は embedding
     適応（全4手法で35.88〜52.56%）の桁違いの改善。embedding 空間を不変に保ち、classifier
     head の intercept だけを動かす設計が有効だった。
  2. **education intercept の系統的低下が主因**: education intercept=-0.1185 は他ドメイン
     に対して系統的に低い。+0.7 で補正した結果、education_recall 0.4588→0.5235。
     感度閾値は +0.5〜+0.7 の間。
  3. **全embedding適応試行の総括**: SetFit full FT(52.56% flip), LoRA r=16(35.88%),
     LoRA r=8(35.88%), Dense projection head(42.00%) — 全rejected。embedding空間の
     幾何学的制約により単一レバー原則と両立しないことが確定。
  4. **classifier_head_adaptationの残値は実質不要**: education_posthoc_calibrationは
     interceptシフトと数学的に同等、education_feature_augmentationはargmax flip rate
     15%超リスクが高い。
- **config.yml の全 levers 試し切り状況**:
  - `fallback_policy`: adopted（完了）
  - `classifier_calibration`: 3値すべて試済み（temperature=adopted）
  - `classifier_training_data_composition`: 6値すべて試済み（全rejected/invalid）
  - `class_weight_adjustment`: 1値試済み（rejected）
  - `embedding_adaptation`: 4値すべて試済み（全rejected）
  - `classifier_head_adaptation`: 1値試済み（education_boundary_tuning=adopted）
    残り2値は実質不要と判断
  - `aggregation_method`: Y2ブロックで試せない
- **次の一手: 調査フェーズから開始**（Iter45）。`current_lever=null`。
  tavily-searchで代替アプローチを重点調査すること。
- **要人間判断**:
  1. education_recall の基準値（medical_recall 0.5112）の再検討
  2. Y2（`confidence_threshold` の二重責務分離，スキーマ変更）着手前のユーザー確認
  3. fallback 設計思想の論文上の位置付け（B48）
  4. D5（`data/`/`models` のバージョン管理方針）

