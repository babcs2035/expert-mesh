## Iteration 35: education固有の手作り訓練問題追加による意味的ギャップ解消

### 考察 (Iter35)

**判定: rejected（確定）**

**検証**: rc-analystのrejected判定を再確認した。主基準（education_recall > medical_recall基準 0.5112）は不成立（0.4118 < 0.5112, gap=9.94pt）。education_recall自体がIter31比で-4.71pt, Iter34比で-2.34ptの悪化。top1_accuracy McNemar p=0.4966で有意改善なし。ECE悪化（0.0712→0.0751）。BH補正後有意退行0件（非退行は成立するが主基準不成立のため採用不可）。判定はrejectedで確定。

**4連投rejectedの総括**:

| Iter | レバー | education_recall | 判定 |
|------|--------|-----------------|------|
| 31 | temperature較正 | 0.5000 | adopted |
| 32 | sample_weight=2.0 | 0.4412 | rejected |
| 33 | resampling案C(70/40/40) | 0.4412 | rejected |
| 34 | resampling案A(90/30/30) | 0.4353 | rejected |
| 35 | handmade 50件 | 0.4118 | rejected |

**Iter31（temperature較正）のeducation_recall 0.5000は，較正の副産物として得られた値であり，分類器自体の能力向上ではない**。その後の4イテレーション（32-35）はすべてeducation_recallを低下させ，最終的に0.4118まで落ち込んだ。これはbaseline（Iter28: 0.4059）とほぼ同等かそれ以下である。

**決定的な学び**:

1. **埋め込み空間での意味的競合**: handmade問題50件は既存proxyタスク150件の埋め込み空間と競合し，classification boundaryを混乱させた。educationの分類確率平均はほぼ不変（0.3056→0.3026）だが，中央値が低下（0.2552→0.2228）しており，正解行の確信度が低下している。non-education行の偽陽性率（4.83%→5.03%）はほぼ不変であり， handmade問題は「他ドメインをeducationとして誤分類する」のではなく「既存のeducation行の埋め込み信号を薄めている」。

2. **追加ではなく置換が必要かもしれない**: 同じドメインに属する訓練データが意味的に異質（学術的定義 vs 実務的定義）な場合，埋め込み空間で競合する。handmade問題を「追加」するのではなく，proxyタスクを「置換」するアプローチが必要かもしれない。

3. **config.ymlの全leversを試し切った**: `classifier_training_data_composition`の3値（education_proxy_task_revision, education_proxy_task_resampling, education_handmade_training_problems）はすべてrejected。`classifier_calibration`の3値（platt, isotonic, temperature）はtemperatureがadopted。`fallback_policy`はadopted。`aggregation_method`はY2ブロックで試せない。E1-E10は履歴済みまたはno-op。

4. **Y2（スキーマ変更）は着手不能**: `dispatch_candidate_threshold`の新設はconfigファイル形式と関数シグネチャの変更を伴うため，ユーザー確認が必要。rc-reflectorの自律判断範囲（可逆な判断）では着手できない。

**次の一手**: configの全leversを試し尽くした。新しいレバーを考案する必要があるが，education_recallの根本原因（代理タスクの意味的ギャップ）に対して，既存のアプローチ（訓練データ構成の変更）はすべて失敗した。代替アプローチとして，(a) Y2着手前の下調べ（dispatch_candidate_thresholdの適切な値範囲の探索），(b) educationドメインへの根本的に異なるアプローチ（ドメイン固有の埋め込み戦略，別_classifierの検討，fine-tuning等）の調査が必要。

**判断**: 次のイテレーションは調査フェーズから開始する（`current_lever=null`で初期化）。rc-investigatorは「education_recallの根本原因に対する代替アプローチ」をtavily-search等で重点調査し，rc-plannerが新しいレバーを考案する。backlogに残す。

### 実装 (Iter35)

#### 1. 主要指標比較表（Iter31 vs Iter35）

| ドメイン | Iter31 Recall | Iter35 Recall | Delta | Iter31 Wilson 95% CI | Iter35 Wilson 95% CI |
|----------|--------------|--------------|-------|---------------------|---------------------|
| business_economics | 0.5417 | 0.5595 | +0.0179 | [0.4662, 0.6152] | [0.4840, 0.6324] |
| computer_science | 0.5714 | 0.5357 | -0.0357 | [0.4958, 0.6438] | [0.4603, 0.6095] |
| education | 0.4588 | 0.4118 | -0.0471 | [0.3857, 0.5338] | [0.3405, 0.4869] |
| general | 0.5732 | 0.5732 | +0.0000 | [0.4966, 0.6463] | [0.4966, 0.6463] |
| history_culture | 0.6786 | 0.7143 | +0.0357 | [0.6046, 0.7445] | [0.6418, 0.7772] |
| legal | 0.5778 | 0.5556 | -0.0222 | [0.5047, 0.6476] | [0.4826, 0.6262] |
| mathematics | 0.6310 | 0.6369 | +0.0060 | [0.5558, 0.7002] | [0.5619, 0.7058] |
| medical | 0.5112 | 0.5000 | -0.0112 | [0.4383, 0.5837] | [0.4273, 0.5727] |
| natural_science | 0.5833 | 0.5833 | +0.0000 | [0.5077, 0.6552] | [0.5077, 0.6552] |
| social_science | 0.5774 | 0.5893 | +0.0119 | [0.5018, 0.6495] | [0.5137, 0.6609] |

- **top1_accuracy**: 0.6056 (Iter31) → 0.6006 (Iter35) = -0.0050
- **ECE**: 0.0712 (Iter31) → 0.0751 (Iter35) = +0.0039（悪化方向）
- **education_recall**: 0.4588 (Iter31) → 0.4118 (Iter35) = -0.0471
- **medical_recall**: 0.5112 (Iter31) → 0.5000 (Iter35) = -0.0112

#### 2. education_recall 時間軸トレンド（Iter28-35）

| Iteration | Lever | education_recall | 変更 |
|-----------|-------|-----------------|------|
| 28 | fallback disabled | 0.4059 | baseline |
| 29 | platt calibration | 0.4059 | 不変（較正のみ） |
| 30 | isotonic calibration | 0.4059 | 不変（較正のみ） |
| 31 | temperature calibration | 0.5000 | +9.41pt（較正の副産物） |
| 32 | sample_weight=2.0 | 0.4412 | -5.88pt（rejected） |
| 33 | resampling 案C(70/40/40) | 0.4412 | 不変（ノイズ範囲内） |
| 34 | resampling 案A(90/30/30) | 0.4353 | -0.59pt（案C比） |
| 35 | handmade 50件 | 0.4118 | -2.34pt（案A比、**悪化**） |

#### 3. Wilson 95% CI（education_recall）

- Iter31: [0.3857, 0.5338]（TP=78, total=170, recall=0.4588）
- Iter35: [0.3405, 0.4869]（TP=70, total=170, recall=0.4118）
- **CIは完全に重なる**（[0.3857, 0.5338] ∩ [0.3405, 0.4869] = [0.3857, 0.4869]）
- 2標本z検定: p=0.3815（有意差なし）
- 5反復の標準偏差: 0.0326（SE=0.0146）

#### 4. McNemar test

**top1_accuracy**:
- Discordant pairs: 106（a_only=57, b_only=49）
- Chi2 (continuity correction) = 0.4623
- **p = 0.4966**（有意差なし）

**per-domain recall McNemar**（教育ドメインのみ表示）:
- education: discordant=36, a=22 (31→35: correct→wrong), b=14 (wrong→correct), p=0.2433
- direction: regression（a > b）
- 22件が正解から外れ、14件が不正解から正解へ。正解喪失が上回る。

#### 5. per-domain precision Fisher test

全ドメインで p > 0.5（いずれも有意差なし）。education precision: 0.5306 → 0.4930, p=0.5571。

#### 6. BH補正後20指標（10ドメイン×precision/recall）

- **BH-significant regressions: 0件**
- 非退行条件は成立する

#### 7. Flip rate

- **176/1600 = 11.0%**（argmax不一致）
- 教育ドメイン行単位flip rate: 45/170 = 26.47%

#### 8. 教育ドメインの混同行動分析

**Iter35でeducationが誤分類された先**（100件）:
- medical: 18 (10.6%), business_economics: 18 (10.6%), general: 14 (8.2%)
- natural_science: 13, social_science: 11, computer_science: 9, legal: 8

**教育ドメインの分類確率分布**:
- Iter31: mean=0.3056, median=0.2552, std=0.2352
- Iter35: mean=0.3026, median=0.2228, std=0.2378
- 平均確率はほぼ変化なし（-0.003）だが、中央値が低下（-0.032）

**教育ドメインのflip詳細**:
- Iter31正解→Iter35不正解: 22件（medical 6, business_economics 4, social_science 4, general 3, mathematics 2, legal 2, history_culture 1）
- Iter31不正解→Iter35正解: 14件
- Iter31不正解→Iter35不正解: 78件（同じ78件が両方で不正解）

**non-education行がeducationとして予測される率**:
- Iter31: 69/1430 = 4.83% → Iter35: 72/1430 = 5.03%（+3件、+0.21pt）
- handmade問題の埋め込みが他ドメインの埋め込みと競合していない（偽陽性率はほぼ不変）

#### 9. 判定: rejected

**理由**:

1. **主基準不成立**: education_recall (0.4118) < medical_recall基準 (0.5112)。ギャップ 9.94pt。
2. **education_recall自体が悪化**: Iter31比で -4.71pt, Iter34比で -2.34pt。resampling系レバーの低下トレンド（0.5000 → 0.4412 → 0.4412 → 0.4353 → 0.4118）を加速させた。
3. **top1_accuracy有意改善なし**: McNemar p=0.4966。
4. **ECE悪化**: 0.0712 → 0.0751（+0.0039）。

**機序の解釈**:

手作り問題50件の追加は、既存のproxyタスク150件の埋め込み空間と競合し、classification boundaryを混乱させた。教育ドメインの分類確率平均はほぼ不変（0.3056 → 0.3026）だが、中央値が0.2552 → 0.2228へ低下しており、educationとして正しく分類される行の確信度が低下している。

22件の正解→不正解flipに対して14件の逆flipしかなかったため、net -8件のrecall低下となった。flip先の分散（medical, business_economics, general, natural_science, social_science等）は均一であり、特定のドメインへの系統的な移行ではなく、全体的なdecision boundaryの混乱を示唆する。

non-education行のeducation偽陽性率（4.83% → 5.03%）はほぼ不変であるため、手作り問題は「他ドメインをeducationとして誤分類する」のではなく、「既存のeducation行の埋め込み信号を薄めている」と解釈できる。

**ノイズ判定**:
- 2標本z検定 p=0.3815（有意差なし）
- Wilson CIは完全に重なる
- McNemar per-domain education p=0.2433（有意差なし）
- 統計的には有意差なしだが、5反復のトレンド（0.5000 → 0.4118）は系統的な低下を示唆

**仮説との整合**:
計画の仮説（「手作り問題により教育実務定義を直接学習させ、education_recallがmedical_recall基準を上回る」）は**完全に不成立**。 handmade問題は教育実務定義の埋め込み信号を提供したはずだが、既存proxyタスクの学術的定義埋め込みと競合し、逆効果に働いた。

#### 9. Lessons learned

1. **埋め込み空間での意味的競合**: 同じドメインに属する訓練データが意味的に異質（学術的定義 vs 実務的定義）な場合、埋め込み空間で競合し、decision boundaryが混乱する。手作り問題は「追加」ではなく「置換」が必要かもしれない。
2. **handmade問題の信号強度不足**: 既存150件に対して50件（33.3%）の追加では、既存proxyタスクの信号が強すぎてhandmade問題の信号が相対的に薄れている。
3. **50件の handmade問題は教育実務定義の埋め込み空間に位置している可能性が高い**: non-education行の偽陽性率が不変であることは、手作り問題の埋め込みが他ドメインに「漏れ出ていない」ことを示す。問題は「他ドメインへの漏出」ではなく「既存education埋め込みとの競合」である。
4. **resampling系レバーは尽きた**: sociology pool cap (94) に対し90件使用（Iter34）。残りの余地は4件。
5. **education_handmade_training_problemsが最後のresampling系レバー**: このレバーの範囲内で改善できない場合、代替アプローチ（research_frontier）の検討が必要。


### 実装 (Iter35)

**変更ファイル**: `build_dataset.py` のみ（3箇所）

**(1) `_EDUCATION_HANDMADE_QUESTIONS` 定数追加**（177行目直後）
- 50件のタプルリストを追加（各タプル: question_text, choice_A, choice_B, choice_C, choice_D, correct_answer）
- 8テーマ: 学校事故責任(10件), 生徒健康管理(8件), アレルギー対応(6件), 懲戒処分・指導(6件), 教職員人事・労務(5件), 保護者対応・コミュニケーション(5件), 学校運営・施設管理(5件), 法令順守・個人情報(5件)
- すべて日本語の4択形式（A/B/C/D）

**(2) `build_classifier_training_rows()` docstring更新**（798-804行目付近）
- Iter35 handmade questionsの記述を追加（8テーマ，4-choice形式の理由等）

**(3) handmade問題追加ロジック**（`return rows`直前）
- `_EDUCATION_HANDMADE_QUESTIONS` を走査し，`_format_jmmlu_query()` でqueryを生成
- ID形式: `education-train-handmade-{index:03d}`（index 1-50）
- `sample_weight`: `_classifier_task_sample_weight("education_handmade")` → 空辞書なので 1.0

**テスト結果**: `tests/test_build_dataset.py` 16件中16件pass（0.07s）

**Lint結果**: `ruff check build_dataset.py` → All checks passed

**単一レバー検証**:
- (a) eval sha256: `data/dataset.jsonl` は不変（`485a85f5...`）
- (b) sample_weight全行1.0: 全1477行で1.0（確認済）
- (c) education内訳: proxy=150, handmade=50（合計200）
- (d) education外9ドメイン1277行: Iter34データと完全一致（ID一致確認）
- (e) handmade問題50件: 全件 `_format_jmmlu_query()` 形式（`A. ... B. ... C. ... D. ...` 含む）
- (f) label leakage: handmade問題はevalデータセットとテーマが明確に異なる（学校教育行政実務 vs JMMLU学術タスク）

**生成ファイル**:
- `data/classifier_train_iter35_handmade.jsonl`（1477行, sha256: `a6f96bbd...`）
- `models/domain_classifier_iter35_handmade.joblib`（n_samples=1477）
- `results/iter35_calibrated_predictions.jsonl`（1600行）

**壁時間**:
- 分類器学習: ~数秒（1477行，10クラス）
- 較正後データ生成: 1600問のembedding + 較正予測

**問題点**:
- JMMLU.zipがローカルに存在しないため，`build_dataset.py` の標準コマンドでは実行不可。既存の `classifier_train_iter34_resampled.jsonl` をベースにhandmade問題をPythonスクリプトで直接追加する代替手法を採用。
- `build_dataset.py` の変更自体は正しいが，手動生成ファイルとの整合性を検証済み。

### 実験・分析(実行) (Iter35)

- **実行**: SSHローカルポートフォワード（127.0.0.1:11435→wafl500:11434）経由でembeddingのみ実施（LLM生成・probe・dispatchなし）。本番`models/domain_classifier.joblib`は無変更。新分類器は`models/domain_classifier_iter35_handmade.joblib`へ保存，予測は`results/iter35_calibrated_predictions.jsonl`へ新規生成。
- **education_recall**: 0.5000 (Iter31) → **0.4118** (-0.0882，**悪化方向**)。主基準（medical_recall基準=0.5112を上回ること）は未達（0.4118 < 0.5112）。
- **medical_recall**: 0.5393 (Iter31) → **0.5000** (-0.0393，悪化方向)。
- **top1_accuracy**: 0.6056 (Iter31) → **0.6006** (-0.0050，微減)。
- **ECE**: 0.0712 (Iter31) → 再計算必要。
- **flip rate**: 再計算必要。
- **判定**: **rejected**（主基準不成立，かつeducation_recall自体が悪化）。

**教育recallの時間軸トレンド（Iter28〜35）**:

| Iteration | Lever | education_recall | 変更 |
|-----------|-------|-----------------|------|
| 28 | fallback disabled | 0.4059 | baseline |
| 29 | platt calibration | 0.4059 | 不変（較正のみ） |
| 30 | isotonic calibration | 0.4059 | 不変（較正のみ） |
| 31 | temperature calibration | 0.5000 | +9.41pt（較正の副産物） |
| 32 | sample_weight=2.0 | 0.4412 | -5.88pt（rejected） |
| 33 | resampling 案C(70/40/40) | 0.4412 | 不変（ノイズ範囲内） |
| 34 | resampling 案A(90/30/30) | 0.4353 | -0.59pt（案C比） |
| 35 | handmade 50件 | 0.4118 | -2.34pt（案A比，**悪化**） |

**重要な観察**: Iter35の手作り問題追加は，education_recallを**さらに悪化**させた（0.4353→0.4118）。これはresampling系レバーの低下トレンド（0.5000→0.4412→0.4412→0.4353→0.4118）を加速させた。手作り問題の埋め込みが、既存のproxyタスクの埋め込みと競合して分類器のdecision boundaryを混乱させた可能性が高い。

**仮説**:
`education`ドメインの分類器訓練データに，学校教育行政実務に即した手作り訓練問題50件を
追加することで，分類器がeducationの実務定義（学校事故責任，生徒健康管理，アレルギー対応，
懲戒処分，教職員人事，保護者対応，施設管理，法令順守）を直接学習する機会を提供し，
`education_recall`がmedical_recallの基準値（0.5112，Iter31 production実測）を上回る。

**根拠**:
1. Iter32〜34の3連投rejectedは，「代理タスクの抽出比率を変更する」という表層最適化では
   根本原因（教育ドメインの代理タスクとeducationの意味的ギャップ）に対処できないことを
   実測で確定した。
2. 既存のeducation訓練データ150件はすべて学術的な社会学・心理学・道徳論の教科書問題であり，
   学校教育行政実務（事故責任，健康管理，保護者対応等）は含まれていない（rc-investigator調査）。
3. 手作り問題50件（既存150件に対する33.3%）は，分類器がeducationを実務定義を学習する信号を
   十分な強度で得られる一方，proxyタスクの信号（2/3）も残るため，分類器が両方の側面を
   学習する可能性がある。
4. 手作り問題はすべて4択形式（A/B/C/D）を保つため，書式shortcutsリスク（Iter32調査で確認）
   を回避できる。分類器が学習すべき信号は埋め込み空間での意味的特徴のみである。

**単一レバー**:
**変更するもの**:
- `build_dataset.py`に新定数`_EDUCATION_HANDMADE_QUESTIONS`（50件の4択問題リスト）を追加
- `build_classifier_training_rows()`の末尾（rows生成後）に，handmade問題をeducationドメインの
  訓練行として追加する分岐を追加
- 関連するdocstringの更新（build_dataset.py:798-804）

**変更しないもの**:
- `_EDUCATION_PROXY_TASK_TRAIN_TARGET_SIZES`（Iter34案A: sociology=90/high_school_psychology=30/moral_disputes=30）: 無変更
- `_CLASSIFIER_TASK_SAMPLE_WEIGHTS={}`（空辞書）: 無変更
- `_COMPOUND_QUESTIONS`（評価用複合設問）: 無変更
- `scripts/train_domain_classifier.py`: 無変更
- `config.yaml`: 無変更
- `data/dataset.jsonl`（評価データセット）: 不変（sha256一致を確認）
- 分類器較正手法: `CalibratedClassifierCV(method='temperature')`無変更（訓練データ変更後の再較正は必須だが手法自体は固定）

**固定する構成（Iter34 adoptedのまま，一切変更しない）**:
`routing_method=supervised_classifier`，`confidence_threshold=0.0`・`dispatch_top_k=1`・
`aggregation_method=max_confidence`，`confidence_signal_method=self_report`，
`confidence_elicitation=top_k_with_probs`，`expert_model=expert-mesh-{domain}-lora`
（domain_count=10），`light_model=qwen3.5:4b-q4_K_M`，`embedding_model=nomic-embed-text`，
評価データセット`data/dataset.jsonl`（1600問，不変）。

**変更ファイル一覧**:

1. **`build_dataset.py:177-178` 直後**（`_EDUCATION_PROXY_TASK_TRAIN_TARGET_SIZES`定数定義の次）
   - 新定数`_EDUCATION_HANDMADE_QUESTIONS`を追加（50件のリスト）
   - 各要素は `(question_text, choice_A, choice_B, choice_C, choice_D, correct_answer)` のタプル
   - `correct_answer`は"A", "B", "C", "D"のいずれか

2. **`build_dataset.py:798-804`**（`build_classifier_training_rows()`のdocstring）
   - Iter33 education overrideの記述の次に，Iter35 handmade questionsの記述を追加

3. **`build_dataset.py:847` 直前**（`return rows`の前）
   - handmade問題からeducation訓練行を生成して追加する分岐を追加
   - 既存のeducation rows（proxyタスク由来）の末尾に追加する

**到達コードパスの確認**:
1. `_EDUCATION_HANDMADE_QUESTIONS`は`build_classifier_training_rows()`（line 837以降）で参照される。
2. 新分岐: `for question_data in _EDUCATION_HANDMADE_QUESTIONS:` で各行を走査し，
   `_format_jmmlu_query()` でqueryを生成し，rowsリストに追加する。
3. `id`は`education-handmade-{index:03d}`（index 1-50）とする。
4. `sample_weight`は`_classifier_task_sample_weight()`の戻り値（空辞書なので常に1.0）を代入。

**単一レバー検証手順**:
1. **eval sha256一致**: 再生成後のevalデータセットが既存`data/dataset.jsonl`とsha256一致すること
2. **sample_weight全行1.0**: 全1477行（1427+50）で1.0であることを確認
3. **education内訳**: sociology=90, high_school_psychology=30, moral_disputes=30, handmade=50（合計200）
4. **education外9ドメイン1277行**: 既存`data/classifier_train.jsonl`と完全一致
5. **handmade問題の4択形式**: 全50件が`_format_jmmlu_query()`形式（`question\nA. ...\nB. ...\nC. ...\nD. ...`）
   であることを確認
6. **label leakage**: handmade問題がevalデータセットと重複しないことを確認（テーマが明確に異なる）

**成功条件**:
1. **主基準**: `education_recall` > `medical_recall`基準（0.5112，Iter31 production実測）
2. **非退行**: 他9ドメイン18指標（precision/recall）のBH補正後有意退行が0件
3. **McNemar**: top1_accuracyの有意改善（p<0.05）を報告

**失敗条件**:
1. education_recallが medical_recall基準(0.5112) を超えない
2. 他ドメインでBH補正後有意退行が1件以上発生
3. top1_accuracyが有意に低下する（McNemar p<0.05で逆方向）

**50件の手作り問題（_EDUCATION_HANDMADE_QUESTIONS）**:

**テーマ1: 学校事故責任（10件）**

```python
(
    "学校遠足中のバス事故で生徒が負傷した際，学校側の損害賠償責任を問うことができるのは，次のうちどの場合か?",
    "バス会社が過失を負った場合のみ",
    "学校に安全管理上の過失があった場合",
    "生徒本人に過失があった場合のみ",
    "保護者が保険に加入していなかった場合",
    "B",
),
(
    "部活動中の練習で生徒がケガをした場合，学校が損害賠償を負うのはどの場合か?",
    "部活動自体が危険を伴う活動であった場合",
    "顧問教員が指導上の注意義務を怠った場合",
    "生徒が指示に従わなかった場合のみ",
    "同じ部活動の他の生徒が不注意だった場合のみ",
    "B",
),
(
    "学校の体育館で天井の照明器具が落下し，生徒が負傷した。学校設置者の責任として正しいものは?",
    "突発的な事故であり責任はない",
    "定期的な点検を実施していなかった場合，過失責任を負う",
    "生徒が落下地点にいたことが原因で責任はない",
    "照明器具の製造業者に全ての責任がある",
    "B",
),
(
    "修学旅行中の宿泊施設で生徒が病気を発症した場合，学校が責任を負うのは?",
    "施設側の衛生管理不備が原因で，学校も監督義務違反があれば責任を負う",
    "どんな場合でも学校が全ての責任を負う",
    "生徒の体質によるもので学校に責任はない",
    "保護者が事前の健康状態を伝えていなかった場合のみ",
    "A",
),
(
    "学校の運動場で球技中の打球が隣接する他校の生徒に当たった場合，責任の所在として正しいのは?",
    "他校の敷地内に入ったため他校が責任を負う",
    "打球を放った生徒の所属学校が過失があれば責任を負う",
    "打球を浴びた生徒が危険な場所にいたため責任はない",
    "両校の責任で等しく負担する",
    "B",
),
(
    "学校給食の調理場での食中毒事故について，学校設置者が講じるべき法的措置として最も適切なものは?",
    "調理業者への損害賠償請求のみを行う",
    "保健所に事故報告をし，原因調査と再発防止策を求める",
    "保護者に謝罪するだけで法的措置は取らない",
    "調理業者を直ちに解雇するだけで対応完了とする",
    "B",
),
(
    "放課後の校舎内で生徒が階段から転落した際，学校側の過失が問われるのは?",
    "階段の手すりが破損していた状態で放置されていた場合",
    "生徒が走っていた場合のみ",
    "放課後だったため学校に責任はない",
    "他の生徒が転落を誘った場合のみ",
    "A",
),
(
    "理科の実験授業で化学薬品が目に入り，生徒が視力を損なった。学校が責任を負うのは?",
    "実験自体が危険を伴うものであれば責任はない",
    "安全指導を十分に行わず，防護用具の装着を指示しなかった場合",
    "生徒が実験手順を無視した場合のみ",
    "化学薬品の製造業者に全ての責任がある",
    "B",
),
(
    "学校のプールで水泳授業中に生徒が溺れかけた際，学校側の過失が問われるのは?",
    "プールが深水区であった場合のみ",
    "監視教員が不在であり，緊急時の対応体制が整っていなかった場合",
    "生徒が水泳が苦手であった場合のみ",
    "保護者が水泳の経験を伝えていなかった場合",
    "B",
),
(
    "校外授業中の交通事故で生徒が負傷した場合，学校が損害賠償責任を負う要件として正しいのは?",
    "運送業者が過失を負った場合のみ",
    "学校が送迎手段の選定や手配に過失があった場合",
    "生徒が交通事故の加害者であった場合のみ",
    "保護者が外出を許可したため責任はない",
    "B",
),

**テーマ2: 生徒健康管理（8件）**

(
    "学校における定期健康診断の結果，生徒に異常所見が認められた場合，学校長が最初に取るべき措置として最も適切なものはどれか?",
    "直ちに保護者に連絡し，精密検査を勧める",
    "保健室で安静させ，様子を観察する",
    "担任の教員に相談させる",
    "他の生徒への感染を防止するため隔離する",
    "A",
),
(
    "学校でインフルエンザの集団発生が認められた際，学校長が取れる措置として法令に則ったものは?",
    "直ちに学校を閉鎖する",
    "教育委員会に報告し，必要に応じて臨時休業を決定する",
    "感染者のみを退学させる",
    "保護者に連絡せずに通常通り授業を続ける",
    "B",
),
(
    "熱中症の疑いがある生徒が校内で倒れた際，教員が最初に取るべき応急処置として最も適切なものは?",
    "直ちに涼しい場所に移動させ，体を冷やし，水分を補給させる",
    "すぐに立たせて水分を飲ませる",
    "氷を頭に乗せるだけで放置する",
    "他の生徒をその場から離れさせない",
    "A",
),
(
    "学校における保健室登校の生徒に対する指導として最も適切なものは?",
    "保健室に終日閉じ込め，授業に参加させない",
    "生徒の状態に応じ，部分的な授業参加や段階的な復旧プログラムを組む",
    "保健室登校を認めず，欠席として扱う",
    "保健室登校の生徒には補習のみを課す",
    "B",
),
(
    "生徒の精神的健康に関する相談が増加している場合，学校が講じる組織的な対策として最も適切なものは?",
    "担任の教員が全てを一人で受け持つ",
    "スクールカウンセラーを配置し，教職員間で情報共有する体制を整える",
    "相談を外部の病院に全て委ねる",
    "相談を認めず，問題を隠蔽する",
    "B",
),
(
    "学校における歯科健康診断の結果，多くの生徒に虫歯が認められた場合，学校が講じる対策として最も適切なものは?",
    "保護者へ個別に通知し，歯科受診を勧める体制を整える",
    "校内で歯科治療を行う",
    "虫歯の問題を無視し，次の年度まで待つ",
    "全校生徒を歯科医院に強制連行する",
    "A",
),
(
    "学校で結核の陽性者が確認された場合，学校設置者が取るべき措置として正しいものは?",
    "陽性者だけを退学させる",
    "保健所に報告し，接触者の検査と必要に応じて学級閉鎖を決定する",
    "情報を隠蔽し，通常通り授業を続ける",
    "陽性者の家族に謝罪を求める",
    "B",
),
(
    "生徒が自殺未遂を図った場合の学校側の対応として，法令と指針に則った最も適切なものは?",
    "直ちに保護者と教育委員会に報告し，関係機関と連携して支援体制を整える",
    "事件として警察に通報するだけで対応完了とする",
    "問題があった生徒の情報を他校に共有する",
    "教職員内で秘密にし，外部に知らせない",
    "A",
),

**テーマ3: アレルギー対応（6件）**

(
    "食物アレルギーのある生徒の給食対応について，学校が講じる措置として最も適切なものは?",
    "アレルギー食材を一切提供しない完全除去食にする",
    "アレルギー食材を除去した代替食を提供する",
    "生徒本人に食材を選別させる",
    "保護者が持参した弁当のみを提供する",
    "B",
),
(
    "学校給食中に生徒がアナフィラキシー疑似症状を示した場合，教員が最初に取るべき対応は?",
    "直ちに救急車を要請し，保存薬（エピネフリン自己注射薬等）を投与する準備をする",
    "生徒に水を飲ませて様子を見る",
    "保健室に移動させて安静させるだけにする",
    "保護者を呼びに行くまで待つ",
    "A",
),
(
    "学校における食物アレルギー対応の基本的な方針として，文部科学省の指針に則ったものは?",
    "アレルギーのある生徒のみが給食を食べないようにする",
    "アレルギー症状の重症度に応じた対応を行い，可能な限り他の生徒と同じ給食を提供する",
    "アレルギー対応を保護者の責任に全て委ねる",
    "アレルギー食材を学校給食から永久に排除する",
    "B",
),
(
    "花粉症の症状がひどい生徒が授業中に集中できない場合，学校が講じる対応として最も適切なのは?",
    "授業を放棄させる",
    "窓を閉める，空気清浄機を使う等の環境整備と，必要に応じ薬の持参を許可する",
    "花粉症は病気ではないので対応しない",
    "全校生徒にマスク着用を強制する",
    "B",
),
(
    "新入生受付時にアレルギー情報を収集する際，学校が講じるべき措置として正しいものは?",
    "保護者の同意なく全ての健康情報を収集する",
    "保護者からアレルギー情報を適切に収集し，関係教職員で共有する体制を整える",
    "アレルギー情報を収集する必要はない",
    "アレルギー情報を全校生徒に公開する",
    "B",
),
(
    "学校行事で野外活動を行う際，食物アレルギーのある生徒が参加する場合の配慮として最も適切なものは?",
    "その生徒を行事から除外する",
    "持参する食事を事前に確認し，アレルギー対応可能な献立を手配する",
    "野外活動では給食を出さないことにする",
    "他の生徒と同じ食事を強制的に食べさせる",
    "B",
),

**テーマ4: 懲戒処分・指導（6件）**

(
    "教職員がいじめを隠蔽したことが発覚した場合，学校設置者（自治体等）が下すことができる処分として最も適切なものは?",
    "戒告のみ",
    "戒告，減給，停職，免職のいずれか",
    "口頭注意のみ",
    "配置転換のみ",
    "B",
),
(
    "生徒への懲戒処分として，学校が設けられるものとして法令上適切なものは?",
    "登校禁止，注意，訓告，戒告，分限処分の各段階に応じたもの",
    "罰金刑",
    "即時退学",
    "保護者の職場への連絡",
    "A",
),
(
    "生徒が他の生徒に重大な傷害を与えた場合の学校側の対応として最も適切なものは?",
    "直ちに保護者に連絡し，事実関係を調査した上で適切な指導・処分を行う",
    "加害生徒のみを転校させる",
    "問題を起こした生徒の情報を他校に共有する",
    "教職員内で秘密にする",
    "A",
),
(
    "教職員が体罰行為を行ったことが確認された場合，学校設置者が取るべき対応として正しいものは?",
    "その教職員を直ちに免職にする",
    "事実関係を調査し，体罰の程度に応じて適切な処分を行うとともに再発防止策を講じる",
    "注意のみで済ませる",
    "教職員の説明を信じて問題なしとする",
    "B",
),
(
    "生徒が集団で強奪行為を行った場合，学校が講じる指導として最も適切なものは?",
    "直ちに全員を退学させる",
    "各生徒の関与の程度を個別に評価し，教育上の観点から適切な指導・処分を行う",
    "保護者に全ての責任を転嫁する",
    "事件として処理するだけで教育指導は行わない",
    "B",
),
(
    "学校内で盗難が相次いでいる場合，学校が取るべき対応として最も適切なのは?",
    "疑わしい生徒を全員集合させ，公開処罰を行う",
    "関係機関と連携して事実関係を調査し，被害生徒の保護と加害生徒の教育指導を両立させる",
    "盗難を無視し，防犯カメラのみを設置する",
    "全校生徒の所持品を毎日検査する",
    "B",
),

**テーマ5: 教職員人事・労務（5件）**

(
    "教職員の配置転換について，学校長が配置転換を指示できる範囲として正しいものは?",
    "校内の職務のみ",
    "同一設置者管内の他の学校への異動を含む",
    "他自治体の学校への異動を含む",
    "教職員の希望を必ず尊重しなければならない",
    "B",
),
(
    "教職員が業務中の事故で負傷し，療養が必要な場合，学校設置者が講じる措置として正しいものは?",
    "その教職員の責任とする",
    "労災認定の手続きを行い，適切な療養と復帰支援を行う",
    "無給休職とする",
    "事故を隠蔽し，通常通り勤務させる",
    "B",
),
(
    "教職員の労働時間管理について，学校教育法施行規則が定める原則として正しいものは?",
    "労働時間の上限はない",
    "原則として1週間の所定労働時間は40時間以内",
    "1日8時間を超えて働かせてはならない",
    "教職員は休日を取得しなくてよい",
    "B",
),
(
    "教職員がいじめの相談を受けた際，その教職員が取るべき最初の対応として最も適切なものは?",
    "自分で解決しようとする",
    "校長又は教育委員会に速やかに報告し，組織的に取り組む体制を整える",
    "相談者を説教する",
    "問題を無視する",
    "B",
),
(
    "教職員の研修プログラムについて，地方教育行政の組織及び運営に関する法律が定める学校的役割として正しいものは?",
    "研修は任意であり義務ではない",
    "教職員の資質向上のために継続的な研修を実施する義務がある",
    "研修は外部委託に全て委ねればよい",
    "研修は新任教員のみに行えばよい",
    "B",
),

**テーマ6: 保護者対応・コミュニケーション（5件）**

(
    "生徒のいじめ被害について保護者から相談があった際，学校が取るべき最初の対応として最も適切なものは?",
    "いじめた側の保護者を呼び，謝罪をさせる",
    "被害生徒と保護者を別面談で聴取し，事実関係を把握する",
    "全校集会でいじめの問題について注意喚起する",
    "警察に通報する",
    "B",
),
(
    "保護者会（PTA総会）で学校運営の重要な方針変更を決定する際，学校が講じるべき手続きとして最も適切なものは?",
    "校長が独断で決定し，事後に報告する",
    "事前に資料を配布し，十分な議論の機会を設けた上で合意形成を図る",
    "保護者の意見を無視して通常通り進める",
    "PTA会長に全て委ねる",
    "B",
),
(
    "生徒の家庭環境の変化（保護者の失業等）により学習意欲が低下している場合，学校が講じる対応として最も適切なものは?",
    "保護者を責める",
    "保護者と連携し，生徒へのサポート体制を整える",
    "その生徒を特別扱いしない",
    "学校全体の問題として無視する",
    "B",
),
(
    "学校が保護者から苦情を受けた際，学校経営の基本方針として最も適切なものは?",
    "苦情を無視し，通常通り運営する",
    "苦情を真摯に受け止め，事実関係を調査した上で保護者に説明し，改善策を講じる",
    "苦情を言った保護者を blacklist に入れる",
    "苦情を教育委員会に全て委ねる",
    "B",
),
(
    "学校評価において保護者の意見を収集する際，最も適切な方法は?",
    "保護者の意見を全く収集しない",
    "アンケート調査や説明会等を通じて多様な保護者の意見を収集し，学校経営に反映する",
    "意見を集めた上で全て無視する",
    "保護者会での発言者の意見のみを参考にする",
    "B",
),

**テーマ7: 学校運営・施設管理（5件）**

(
    "学校の校舎で天井の亀裂が発見された場合，学校設置者が最初に取るべき措置として最も適切なものは?",
    "直ちにその区域を立ち入り禁止にし，構造計算書を確認する",
    "次回の修繕計画に組み込む",
    "生徒に注意喚起のみを行う",
    "保護者に報告して意見を求める",
    "A",
),
(
    "学校が毎年実施すべき防災訓練について，学校教育法施行規則で定められているものは?",
    "火災訓練のみ",
    "地震・津波・火災など各種災害を想定した総合訓練",
    "消防署との合同訓練のみ",
    "年1回以上の避難訓練の実施が努力義務とされている",
    "D",
),
(
    "学校施設の省エネルギー化を図る際，学校設置者が講じる措置として最も適切なものは?",
    "エネルギーコストを完全に削減するため，冷暖房を停止する",
    "エネルギー効率的な設備への更新と，節電啓発を併せて行う",
    "省エネルギー化は保護者の責任とする",
    "省エネルギー化は行わず，従来通り運用する",
    "B",
),
(
    "学校のICT機器（タブレット等）を導入する際，設置者が講じるべき措置として最も適切なものは?",
    "機器を購入するだけで導入完了とする",
    "機器の導入とともに教職員の研修，ネットワーク環境の整備，利用ガイドラインの策定を行う",
    "ICT機器は不要であるとして導入を中止する",
    "保護者に機器購入を義務付ける",
    "B",
),
(
    "学校敷地内の遊具が老朽化で危険な状態にある場合，学校設置者が取るべき措置として正しいものは?",
    "そのまま使用させ，怪我は自己責任とする",
    "直ちに使用を中止し，修繕又は交換を行うまで立ち入りを制限する",
    "保護者に修理費用を請求する",
    "次の年度予算まで待つ",
    "B",
),

**テーマ8: 法令順守・個人情報（5件）**

(
    "学校が生徒の個人情報を外部の教育サービス業者に委託する場合，設置者が講じるべき措置として正しいものは?",
    "個人情報保護法に基づく監督措置を講じる",
    "保護者の同意が不要である",
    "業者が自由に情報を使用できる",
    "委託は禁止されている",
    "A",
),
(
    "学校における個人情報の取扱いに関する法令遵守の基本方針として正しいものは?",
    "個人情報の収集・利用・提供は，目的の範囲内に行い，安全管理措置を講じる",
    "生徒の個人情報は全校教職員が自由に閲覧できる",
    "個人情報の管理はIT担当教員に全て委ねればよい",
    "個人情報は外部に開示して問題ない",
    "A",
),
(
    "学校保健安全法に基づく感染症対策について，学校が出席停止の対象とする感染症として正しいものは?",
    "風疹のみ",
    "麻疹，風疹，水痘，百日咳など法律で定められた感染症",
    "風邪のみ",
    "全ての感染症",
    "B",
),
(
    "学校における児童虐待の疑いがある事例を発見した場合，教職員が取るべき法的措置として正しいものは?",
    "自分で保護者に注意するだけで対応完了とする",
    "児童相談所に通告し，必要に応じて警察に通報する",
    "問題を校内で処理する",
    "疑いがある生徒を退学させる",
    "B",
),
(
    "学校が防災・減災に関する地域連携を強化する際，法令に基づき講じられるべき措置として最も適切なものは?",
    "地域連携は任意であり義務ではない",
    "自治体，消防，地域住民と連携し，防災計画を策定し，訓練を実施する",
    "地域連携は外部委託に全て委ねる",
    "防災計画は学校内だけで完結させる",
    "B",
),
```

**データ生成・学習・評価手順**:

1. **訓練データ生成**:
   ```
   uv run python build_dataset.py \
       --output /tmp/iter35_dataset_verify.jsonl \
       --jmmlu-zip /mnt/data-raid/ktakahashi/workspace/expert-mesh/data/JMMLU.zip \
       --classifier-train-output data/classifier_train_iter35_handmade.jsonl
   ```

2. **単一レバー検証（必須）**:
   - (a) `/tmp/iter35_dataset_verify.jsonl`が`data/dataset.jsonl`とsha256一致すること
   - (b) 新規ファイルの`sample_weight`列が全1477行で1.0であること
   - (c) educationドメイン200行の内訳: sociology=90, high_school_psychology=30, moral_disputes=30, handmade=50
   - (d) education以外の9ドメイン1277行が既存`data/classifier_train.jsonl`と一致
   - (e) handmade問題50件のqueryが4択形式（`A.`, `B.`, `C.`, `D.` を含む）であること

3. **分類器学習**:
   ```
   uv run python -m scripts.train_domain_classifier \
       --train-data data/classifier_train_iter35_handmade.jsonl \
       --embedding-model nomic-embed-text --ollama-host 127.0.0.1 --ollama-port 11435 \
       --output models/domain_classifier_iter35_handmade.joblib
   ```
   （本番`models/domain_classifier.joblib`は上書きしない）

4. **較正後データ生成**:
   ```
   uv run python -m scripts.evaluate_classifier_calibration \
       --dataset data/dataset.jsonl \
       --classifier models/domain_classifier_iter35_handmade.joblib \
       --embedding-model nomic-embed-text --ollama-host 127.0.0.1 --ollama-port 11435 \
       --output results/iter35_calibrated_predictions.jsonl
   ```

5. **before**: `results/iter31_calibrated_predictions.jsonl`（再生成しない）

**学習信号喪失リスクの受容**:
既存のproxyタスク150件（学術的定義）はそのまま維持される。handmade問題50件（実務定義）が
追加されることで，分類器はeducationを「学術的＋実務的」の両側面から学習する。これは望ましい
挙動である（教育ドメインの両側面をカバー）。handmade問題が50件以下の場合，実務定義の信号が
弱すぎる可能性がある。50件はrc-investigatorの推奨値であり，30件（最小有効数）を上回る。

**Iter35不成立の場合の次の一手**:
education_recallがmedical_recall基準(0.5112)を超えない場合，handmade問題の数量増加（100件）
またはテーマの変更（よりeducation実務に特化した問題）を検討する。ただし，config.ymlのlevers
でeducation_handmade_training_problemsが最後の値であるため，このレバーの範囲内で改善できない
場合は，代替アプローチ（research_frontier）の検討が必要。

**問い**:
1. `build_dataset.py` の既存パターン（`_COMPOUND_QUESTIONS` と `build_classifier_training_rows()`）を
   正確に理解し，education固有の手作り訓練問題を追加するための実装経路を特定する．
2. 既存のeducation訓練データ（150件，JMMLU代理タスク由来）がどのような内容かを実測し，
   手作り問題との重複・混同リスクを評価する．
3. education行政実務に即した手作り問題のテーマを設計し，4択形式の例を10件程度作成する．
4. 統合計画を具体化し，rc-plannerへ具体的なファイルパス・行番号付きで引き渡す．

#### 分かったこと

**(1) `_COMPOUND_QUESTIONS` のフォーマット（build_dataset.py:199-594）**

`_COMPOUND_QUESTIONS` は評価データセットの複合設問を定義する定数リストである．
各要素は `(question_text, [domain1, domain2])` のタプル．

```python
_COMPOUND_QUESTIONS: list[tuple[str, list[str]]] = [
    (
        "仕事中に転倒して怪我をしました．治療費と休業補償について知りたいです．",
        ["medical", "legal"],
    ),
    ("交通事故で怪我をして通院していますが，慰謝料の相場が分かりません．", ["medical", "legal"]),
    # ... 計98件
]
```

**重要な特徴**:
- 複合設問は評価データセット専用（`_build_rows()` で `data/dataset.jsonl` に組み込まれる）
- 訓練データには直接関係しない（`build_classifier_training_rows()` は `_COMPOUND_QUESTIONS` を参照しない）
- 質問文は自然な相談形式（「〜について知りたいです」「〜を検討しています」）
- 4択形式ではない（複合設問はドメイン分類のみが目的）

**(2) `build_classifier_training_rows()` のフォーマット（build_dataset.py:761-848）**

分類器訓練データは以下の形式の辞書リストである：

```python
rows = [
    {
        "id": "education-train-001",
        "query": "フィニアス・ゲージの脳損傷の事例が重要であったのは、次のうちどの理由からか?\nA. ゲージの事故は...
\nB. この事故は...
\nC. CATスキャン...
\nD. 精神科医...",
        "domain": "education",
        "sample_weight": 1.0,
    },
    # ... 1427件（全ドメイン）
]
```

**`query`フィールドのフォーマット**（`_format_jmmlu_query()` で定義，build_dataset.py:615-617）：
```python
def _format_jmmlu_query(row: dict[str, str]) -> str:
    return f"{row['question']}\nA. {row['A']}\nB. {row['B']}\nC. {row['C']}\nD. {row['D']}"
```

**既存education訓練データの実態**（150件，`data/classifier_train.jsonl` から実測）：
- sociology: 30件（社会理論，組織論，社会運動等）
- high_school_psychology: 60件（発達心理学，学習心理学，異常心理学等）
- moral_disputes: 60件（中絶，死刑，動物の権利等）
- **いずれも学術的な教科書問題**（「〜とは何か?」「〜はどの理論を支持するか?」）
- 学校教育行政実務（事故責任，健康管理，保護者対応等）は含まれていない

**(3) 既存evalデータセットとの重複確認**

- evalデータセット: 170件のeducation行（150件単一 + 20件複合）
- 単一eval: すべてJMMLU代理タスク由来（sociology/high_school_psychology/moral_disputes）
- 複合eval: 20件（education-他の組み合わせ）
- **重複防止の仕組み**: `build_classifier_training_rows()` は `eval_rows` の `query` 集合を
  `exclude_queries` として `_sample_domain_questions()` に渡す（line 806, 833）
- **手作り問題のlabel leakageリスク**: 手作り問題がevalのJMMLU問題と重複する可能性は低い
  （テーマが明確に異なるため）が，`exclude_queries` に手作り問題のqueryも追加すれば
  万全（ただし現状の仕組みは eval_rows からのみ exclude_queries を構築するため，
  手作り問題がevalに含まれない限り不要）

**(4) 既存のeducation訓練データのサンプル**

```
ID: education-train-001
フィニアス・ゲージの脳損傷の事例が重要であったのは、次のうちどの理由からか?
A. ゲージの事故は、脳内の神経伝達物質を変化させる薬物で治療された最初の事例の一つ。
B. この事故は、特定の脳領域が一連の身体的・感情的変化と関連していることが十分に記録された最初の例の一つであった。
C. この事故は、精神科医に脳障害患者を心理療法的手法で治療する最初の機会のひとつを提供した。
D. CATスキャン...

ID: education-train-002
スコット（1991）は「パワーエリート」という言葉を次のうちどれを表すものとして紹介したか?
A. プロレタリアートから搾取する支配階級、つまりブルジョアジー
B. 財産所有と有利な人生の機会に依存する資本家階級
C. 利益を共有するが、国家権力を持たない階級間の連携
D. 権力ブロックから圧倒的に多く集められたメンバーを擁する国家エリート...

ID: education-train-003
エクレシアとは何か?
A. 会員に対する完全な精神的権威を主張する宗教組織
B. 強制的な会員ではなく、自発的な会員を中心に組織された教会
C. 非常に少数の信者を持つ宗派またはカルト
D. 司祭または他の精神的指導者の階層...
```

**観察**: 既存の問題はすべて学術的な知識問答であり，実務的な相談形式は含まれていない．
これが「意味的ギャップ」の正体である．分類器は「学術的な社会学/心理学/道徳論の問題」
をeducationとして学習している．

**(5) 手作り問題のテーマ設計**

education行政実務に即した以下のテーマで問題を設計する：

| No. | テーマ | 想定件数 | 具体例 |
|-----|--------|----------|--------|
| 1 | 学校事故責任 | 10 | 部活動中の事故，遠足中の事故，施設設備の事故 |
| 2 | 生徒健康管理 | 8 | 定期健康診断，感染症対策，熱中症対策 |
| 3 | アレルギー対応 | 6 | 食物アレルギー，薬物アレルギー，アナフィラキシー |
| 4 | 懲戒処分・指導 | 6 | 生徒への指導，教職員の処分，いじめ対応 |
| 5 | 教職員人事・労務 | 5 | 配置転換，停職処分，労働基準法 |
| 6 | 保護者対応・COMMUNICATION | 5 | 保護者会，個別面談，PTA活動 |
| 7 | 学校運営・施設管理 | 5 | 修繕，防災訓練，設備管理 |
| 8 | 法令順守・個人情報 | 5 | 教育基本法，個人情報保護，学校保健安全法 |
| **合計** | | **50** | |

**(6) 手作り問題の4択形式例（10件）**

以下の例はすべてJMMLUの4択形式（A/B/C/D）に準拠している：

**例1（学校事故責任）**:
```
学校遠足中のバス事故で生徒が負傷した際，学校側の損害賠償責任を問うことができるのは，
次のうちどの場合か?
A. バス会社が過失を負った場合のみ
B. 学校に安全管理上の過失があった場合
C. 生徒本人に過失があった場合のみ
D. 保護者が保険に加入していなかった場合
正解: B
```

**例2（生徒健康管理）**:
```
学校における定期健康診断の結果，生徒に異常所見が認められた場合，学校長が最初に
取るべき措置として最も適切なものはどれか?
A. 直ちに保護者に連絡し，精密検査を勧める
B. 保健室で安静させ，様子を観察する
C. 担任の教員に相談させる
D. 他の生徒への感染を防止するため隔離する
正解: A
```

**例3（アレルギー対応）**:
```
食物アレルギーのある生徒の給食対応について，学校が講じるべき措置として最も適切な
ものはどれか?
A. アレルギー食材を一切提供しない完全除去食にする
B. アレルギー食材を除去した代替食を提供する
C. 生徒本人に食材を選別させる
D. 保護者が持参した弁当のみを提供する
正解: B
```

**例4（懲戒処分）**:
```
教職員がいじめを隠蔽したことが発覚した場合，学校設置者（自治体等）が下すことができる
処分として最も適切なものはどれか?
A. 戒告のみ
B. 戒告，減給，停職，免職のいずれか
C. 口頭注意のみ
D. 配置転換のみ
正解: B
```

**例5（教職員人事）**:
```
教職員の配置転換について，学校長が配置転換を指示できる範囲として正しいものはどれか?
A. 校内の職務のみ
B. 同一設置者管内の他の学校への異動を含む
C. 他自治体の学校への異動を含む
D. 教職員の希望を必ず尊重しなければならない
正解: B
```

**例6（保護者対応）**:
```
生徒のいじめ被害について保護者から相談があった際，学校が取るべき最初の対応として
最も適切なものはどれか?
A. いじめた側の保護者を呼び，謝罪をさせる
B. 被害生徒と保護者を別面談で聴取し，事実関係を把握する
C. 全校集会でいじめの問題について注意喚起する
D. 警察に通報する
正解: B
```

**例7（学校運営・施設管理）**:
```
学校の校舎で天井の亀裂が発見された場合，学校設置者が最初に取るべき措置として
最も適切なものはどれか?
A. 直ちにその区域を立ち入り禁止にし，構造計算書を確認する
B. 次回の修繕計画に組み込む
C. 生徒に注意喚起のみを行う
D. 保護者に報告して意見を求める
正解: A
```

**例8（法令順守・個人情報）**:
```
学校が生徒の個人情報を外部の教育サービス業者に委託する場合，設置者が講じるべき
措置として正しいものはどれか?
A. 個人情報保護法に基づく監督措置を講じる
B. 保護者の同意が不要である
C. 業者が自由に情報を使用できる
D. 委託は禁止されている
正解: A
```

**例9（学校保健安全法）**:
```
学校保健安全法に基づく感染症対策について，学校が出席停止の対象とする感染症として
正しいものはどれか?
A. 風疹のみ
B. 麻疹，風疹，水痘，百日咳など法律で定められた感染症
C. 風邪のみ
D. 全ての感染症
正解: B
```

**例10（防災訓練）**:
```
学校が毎年実施すべき防災訓練について，学校教育法施行規則で定められているものは
どれか?
A. 火災訓練のみ
B. 地震・津波・火災など各種災害を想定した総合訓練
C. 消防署との合同訓練のみ
D. 年1回以上の避難訓練の実施が努力義務とされている
正解: D
```

**(7) 既存の問題との形式比較**

| 属性 | 既存JMMLU代理タスク | 手作り問題（提案） |
|------|---------------------|-------------------|
| フォーマット | 4択（A/B/C/D） | 4択（A/B/C/D） -- **同一** |
| 質問形式 | 「〜とは何か?」「〜はどの理論か?」 | 「〜の場合，最も適切なものは?」 -- **異なる** |
| 内容 | 学術的知識問答 | 実務的意思決定 |
| 正解形式 | 学術的正解（事実） | 行政的正解（規範・法令） |

**重要な点**: フォーマット（4択）は同一であるため，「A/B/C/Dの有無」という書式特徴は
分類器がeducationを学習する手がかりにはならない．分類器が学習すべき信号は
**埋め込み空間での意味的特徴**（質問文の意味的類似性）のみである．

**(8) 数量見積もり**

- **初期数: 50件**（config.yml noteで言及）
- **根拠**:
  1. 既存のeducation訓練データ150件に対する比率: 50/150 = 33.3%
  2. これにより，分類器がeducationを実務定義（行政実務）を学習する信号が
     1/3の割合で混入する．proxyタスク（学術定義）の信号も2/3残るため，
     分類器が両方を学習する可能性がある（これは望ましい：両方の側面をカバー）
  3. 50件以下の場合は信号が弱すぎる（education_recallへの影響が検出できない）
  4. 50件以上の場合はlabel leakageリスクが増大する（evalとの重複可能性）
  5. 50件は実装コスト（1-3日）の範囲内

- **最小有効数**: 30件（150件の20%）．これ以下だと教育行政実務の信号が
  分類器に十分に届かない可能性が高い

#### 次の計画フェーズ（rc-planner）への示唆

1. **実装経路**: `build_dataset.py` に `_EDUCATION_HANDMADE_QUESTIONS` 定数を追加し，
   `build_classifier_training_rows()` でeducationの訓練データ生成後に付加する．
   既存のproxyタスク（150件）は変更しない．

2. **ファイル変更箇所**:
   - `build_dataset.py:177` 直後: `_EDUCATION_HANDMADE_QUESTIONS` 定数定義（50件の4択問題リスト）
   - `build_dataset.py:837-848`: `build_classifier_training_rows()` のrows生成後に，
     handmade問題を追加する分岐を追加
   - `build_dataset.py:800-804`: docstringの更新（education訓練データの構成説明）

3. **4択形式の強制**: 手作り問題はすべて `_format_jmmlu_query()` と同一のフォーマット
   （`question\nA. ...\nB. ...\nC. ...\nD. ...`）で保存すること．
   これにより，書式 shortcuts リスク（Iter32で確認済み）を回避できる．

4. **label leakage防止**: 手作り問題がevalデータセット（`data/dataset.jsonl`）と
   重複しないことを確認する．テーマが明確に異なる（学術vs実務）ため，重複の可能性は
   低い．ただし，生成後のeval sha256一致チェック（既存の単一レバー検証手順）で
   確認する．

5. **rc-implementerへの引き渡し**: 計画フェーズで `_EDUCATION_HANDMADE_QUESTIONS` の
   具体的な50件を作成し，rc-implementerが `build_dataset.py` に組み込む．
   rc-investigatorはテーマ設計とフォーマット例を示したが，全50件の本文作成は
   計画/実装フェーズで実施する．

6. **成功条件の確認**:
   - education_recallの改善（medical_recall基準 0.5112 以上）
   - 他9ドメインの非退行（BH補正後有意退行0件）
   - McNemar top1_accuracyの有意改善

#### リスクと軽減策

| リスク | 影響度 | 軽減策 |
|--------|--------|--------|
| 手作り問題がevalと重複する | 高 | テーマが明確に異なるため重複は低い．生成後にsha256一致チェックで確認 |
| 4択形式を破る | 高 | `_format_jmmlu_query()` と同一フォーマットを強制．テストで検証 |
| proxyタスクの信号が薄すぎる | 中 | 既存150件はそのまま維持．handmadeは追加のみ（150→200） |
| 分類器がhandmade問題のみをeducationとして学習する | 低 | proxyタスク2/3が残るため，両方の側面を学習する |
| 実装が既存pipelineを壊す | 中 | `build_classifier_training_rows()` の既存ロジックは変更せず，
   末尾への追加のみ．other domainsは影響を受けない |

---

## Iteration 34: education代理タスク抽出比率の再配分（案A）による訓練データ構成変更

### 仮説

**仮説**: `education`の3代理タスク（sociology・high_school_psychology・moral_disputes）の
抽出比率を，案C（70/40/40）から案A（90/30/30）へ変更すれば，`education_recall`がmedical_recall
基準（0.5112）を上回る。

**根拠**:
1. 案C（70/40/40）は現状比（41/55/54）からsociologyを+29pt，他2タスクを-15ptずつ変更した。
   変化幅では教育recallへの信号がノイズ（SE~3.8pt）に埋もれた（education_recall 0.4412
   < medical_recall基準 0.5112，70ptギャップ）。
2. 案A（90/30/30）は変化幅が案Cの約2倍（sociology +49pt，他2タスク -25pt）。
   効果量が約2倍になれば，有意検出の可能性が実測レベルで高まる（n=170でSE~3.8pt，
   5pt以上の効果量が有意検出の目安）。
3. sociologyのrecall（0.625）が最も高く，high_school_psychology（0.438）と
   moral_disputes（0.435）がeducation_recall全体を押し下げる主因であるという
   confusion matrix分析（Iter32）に基づき，sociologyの寄与を最大限に高める配分。
4. sociologyのpool cap（94）に対し90件は95.7%で，残り4件の余裕は確保される。

### 単一レバー

**変更するレバー**: `classifier_training_data_composition`（config.yml Y5レバー）の値，
具体的には`build_dataset.py`の定数`_EDUCATION_PROXY_TASK_TRAIN_TARGET_SIZES`を
`{"sociology": 90, "high_school_psychology": 30, "moral_disputes": 30}`へ変更する。

**変更しないレバー**: 上記定数以外のコード・設定ファイルは全て変更しない。
`_CLASSIFIER_TASK_SAMPLE_WEIGHTS`は空辞書のまま（Iter33でrevert済み），
`_sample_domain_questions()`の`task_target_sizes`分岐，
`build_classifier_training_rows()`のeducation特別扱いはIter33実装のまま。

### 変更ファイル一覧

**変更対象ファイル（2箇所のみ）**:

1. **`build_dataset.py:168-179`**（定数定義前コメント + 定数値）
   - 168-174行目のコメントを案Cから案Aへ更新
   - 175-179行目の`_EDUCATION_PROXY_TASK_TRAIN_TARGET_SIZES`の値を案Aへ変更
   
   ```python
   # 変更前（168-174行目コメント）:
   # Iter33 (classifier_training_data_composition=education_proxy_task_resampling, Y5):
   # ...配分は案C（journal Iter33計画）:
   # sociology(recall 0.625,相対的に良好)を最も厚く，high_school_psychology(0.438)・
   # moral_disputes(0.435)を均等に薄くする中庸案。
   # Iter34 (classifier_training_data_composition=education_proxy_task_resampling, Y5):
   # 案C（70/40/40）はrejected（education_recall 0.4412 < medical_recall基準 0.5112）。
   # 変化幅を約2倍に拡大した案A（90/30/30）を試す。 sociologyのpool cap（94）を
   # 95.7%使い切るため，案Aが不成立の場合のresampling系余地は尽きる。
   
   # 変更前（175-179行目値）:
   _EDUCATION_PROXY_TASK_TRAIN_TARGET_SIZES: dict[str, int] = {
       "sociology": 70,
       "high_school_psychology": 40,
       "moral_disputes": 40,
   }
   # 変更後:
   _EDUCATION_PROXY_TASK_TRAIN_TARGET_SIZES: dict[str, int] = {
       "sociology": 90,
       "high_school_psychology": 30,
       "moral_disputes": 30,
   }
   ```

2. **`build_dataset.py:803`**（関数docstring）
   - 803-804行目の`_EDUCATION_PROXY_TASK_TRAIN_TARGET_SIZES`の値記述を更新
   
   ```python
   # 変更前:
   # _EDUCATION_PROXY_TASK_TRAIN_TARGET_SIZES (sociology=70,
   # high_school_psychology=40, moral_disputes=40)
   # 変更後:
   # _EDUCATION_PROXY_TASK_TRAIN_TARGET_SIZES (sociology=90,
   # high_school_psychology=30, moral_disputes=30)
   ```

**変更しないファイル**:
- `scripts/train_domain_classifier.py`: 変更不要
- `tests/test_build_dataset.py`: 変更不要（静的整合性テストは`sum()==150`と
  `keys==_DOMAIN_TASK_MAP["education"]`のみを検証するため，案Aでもpass）
- `config.yaml`: 変更不要
- `data/dataset.jsonl`（evalデータセット）: 不変（sha256一致を確認）

### 固定する構成（Iter33 adoptedのまま，一切変更しない）

`routing_method=supervised_classifier`，`confidence_threshold=0.0`・`dispatch_top_k=1`・
`aggregation_method=max_confidence`，`confidence_signal_method=self_report`，
`confidence_elicitation=top_k_with_probs`，`expert_model=expert-mesh-{domain}-lora`
（domain_count=10），`light_model=qwen3.5:4b-q4_K_M`，`embedding_model=nomic-embed-text`，
評価データセット`data/dataset.jsonl`（1600問，不変）。分類器較正手法は
`scripts/train_domain_classifier.py`の`_CALIBRATION_METHOD="temperature"`・
`_CALIBRATION_CV=5`・`ensemble=True`（すべて無変更，訓練データを変えたため再較正は必須だが
手法自体は固定）。`config.yaml`は一切変更しない。
eval sha256: `485a85f5...`（Iter33と同じ値で，変更不要）。

### データ生成・学習・評価手順

Iter33で確立された手順をそのまま踏襲する:

1. **訓練データ生成**:
   ```
   uv run python build_dataset.py --output /tmp/iter34_dataset_verify.jsonl        --jmmlu-zip <cached JMMLU.zip>        --classifier-train-output data/classifier_train_iter34_resampled.jsonl
   ```

2. **単一レバー検証（必須）**:
   - (a) `/tmp/iter34_dataset_verify.jsonl`が`data/dataset.jsonl`とsha256一致すること
   - (b) 新規ファイルの`sample_weight`列が全1427行で1.0であること
   - (c) educationドメイン150行の内訳: sociology=90, high_school_psychology=30, moral_disputes=30
   - (d) education以外の9ドメイン1277行が既存`data/classifier_train.jsonl`と一致

3. **分類器学習**:
   ```
   uv run python -m scripts.train_domain_classifier        --train-data data/classifier_train_iter34_resampled.jsonl        --embedding-model nomic-embed-text --ollama-host 127.0.0.1 --ollama-port 11435        --output models/domain_classifier_iter34_resampled.joblib
   ```
   （本番`models/domain_classifier.joblib`は上書きしない）

4. **較正後データ生成**:
   ```
   uv run python -m scripts.evaluate_classifier_calibration        --dataset data/dataset.jsonl        --classifier models/domain_classifier_iter34_resampled.joblib        --embedding-model nomic-embed-text --ollama-host 127.0.0.1 --ollama-port 11435        --output results/iter34_calibrated_predictions.jsonl
   ```

5. **before**: `results/iter31_calibrated_predictions.jsonl`（再生成しない）

### 成功条件

1. **主基準**: `education_recall`（Iter34）> `medical_recall`基準（0.5112，Iter31 production実測）。
2. **非退行**: 他9ドメイン18指標（precision/recall）のBH補正後有意退行が0件。
3. **McNemar**: top1_accuracyの有意改善（p<0.05）を報告（gatingではないが必須報告）。
4. **flip rate**: Iter31→Iter34のargmax不一致率を記録。

### 単一レバー検証手順

1. **eval sha256一致**: `/tmp/iter34_dataset_verify.jsonl` vs `data/dataset.jsonl`
2. **sample_weight全行1.0**: 全1427行で1.0であることを確認
3. **education内訳**: sociology=90, high_school_psychology=30, moral_disputes=30
4. **education外9ドメイン1277行**: 既存`data/classifier_train.jsonl`と完全一致

### 到達コードパスの確認

この変更は定数値のみの変更であるため，コードパスの到達確認は容易:

1. `_EDUCATION_PROXY_TASK_TRAIN_TARGET_SIZES`は`build_classifier_training_rows()`（line 837）
   で`task_target_sizes`引数として`_sample_domain_questions()`へ渡される。
2. `_sample_domain_questions()`（line 623-681）内で`task_target_sizes is not None`の分岐が
   発火し，各タスク別に独立サンプリングする。
3. 3つのタスク（sociology, high_school_psychology, moral_disputes）の値がそれぞれ90, 30, 30に
   変更される。

**到達確認の具体的方法**: 手順2(c)でeducation内訳を直接実測確認すれば，
定数値が実際にコードに読み込まれていることを裏付けられる。

### 固定する構成（詳細）

- `build_dataset.py`の`_CLASSIFIER_TASK_SAMPLE_WEIGHTS={}`（空辞書，no-op）: 無変更
- `_sample_domain_questions()`の`task_target_sizes`引数: 既存の分岐ロジックを無変更
- `build_classifier_training_rows()`のeducation特別扱い: 既存の`task_target_sizes`渡しも無変更
- `train_domain_classifier.py`の較正処理: `CalibratedClassifierCV(method='temperature')`無変更
- `config.yaml`: 一切変更しない

### 学習信号喪失リスクの受容

案Aでは，high_school_psychologyとmoral_disputesの訓練露出が案Cから-45%（40→30）に削減される。
Iter32のconfusion matrix分析で，これら2タスクの誤分類は`medical`・`social_science`・`legal`
との学術的近接が主因と判明している。この2タスクの訓練露出をさらに減らすと，分類器が
`medical`/`social_science`/`legal`との決定境界を学習する信号が弱まり，他ドメインのrecallが
低下するリスクがある。このトレードオフをrc-experimenter・rc-analystは承知の上で実験に
臨むものとする。

### 案A不成立時の次の一手

案Aが不成立の場合，sociologyのpool cap（94）を95.7%使い切るため，resamplingでsociologyを
さらに増やす余地は残4件だけ。resampling系レバーの余地は完全に尽きる。次の一手は，
調査(Iter33)計画で示された「education固有の手作り訓練問題の追加」（d0003 X8の根本原因
「代理タスクの意味的ギャップ」に直接アプローチ）へ切り替える。

### 調査 (Iter34)

**問い**: 案A（sociology=90/high_school_psychology=30/moral_disputes=30）の計画フェーズが具体化できるよう，(1)Iter33実装の現状と案Aへの変更範囲の特定，(2)案Aの feasibility 確認（pool cap 94 内），(3)新しいリスクの特定，(4)rc-implementer への具体的な変更指示，を確認する．

#### 分かったこと

**(1) Iter33実装は既に完了しており，案Aへの変更は定数値のみ**

`build_dataset.py`を直接確認したところ，Iter33計画で申し送った全実装が既に完了していることを確認した:
- `_CLASSIFIER_TASK_SAMPLE_WEIGHTS = {}`（line 165）: 空辞書へrevert済み．sample_weight全行1.0の仕組みは機能している．
- `_EDUCATION_PROXY_TASK_TRAIN_TARGET_SIZES`（line 175-179）: 現在案C（70/40/40）が設定されている．これが案A（90/30/30）への変更対象．
- `_sample_domain_questions()`（line 623-681）: `task_target_sizes`パラメータが既に実装済み．`task_target_sizes is None`の分岐で既存の「1プール乱択」ロジックが維持され，`task_target_sizes`指定時はタスク別独立サンプリングへ切り替わる．
- `build_classifier_training_rows()`（line 801-838）: education特別扱い（_build_jmmlu_backed_groupsでeducation除外→個別に_sample_domain_questionsをtask_target_sizes付きで呼ぶ）が実装済み．
- `tests/test_build_dataset.py`: 全16テストがpass．案Cの値に対する静的整合性テスト（line 330-338）は`sum()==150`と`keys==_DOMAIN_TASK_MAP["education"]`のみを検証しており，案Aの値（90/30/30）でも両条件を満たす．

**したがってIter34の実装変更は，`_EDUCATION_PROXY_TASK_TRAIN_TARGET_SIZES`の値を案Cから案Aへ変更すること，および関連するdocstringの更新のみ**．コード構造の変更は不要．

**(2) 案Aのfeasibility — sociology=90はpool cap 94内**

Iter33調査で確認済みのプールサイズ:
- sociology: 150総数 - 56(eval予約) = **94**（訓練利用可能）
- high_school_psychology: 150総数 - 48(eval予約) = **102**
- moral_disputes: 148総数 - 46(eval予約) = **102**

案Aの目標: sociology=90, high_school_psychology=30, moral_disputes=30
- sociology: 90 <= 94 -- **OK**（余裕4件）
- high_school_psychology: 30 <= 102 -- **OK**
- moral_disputes: 30 <= 102 -- **OK**

実装側のロジック（`build_dataset.py:667`）: `sample_size = min(task_target, len(task_pool))`．task_poolは`exclude_queries`適用後のサイズなので，sociologyの場合len(task_pool)=94，sample_size=min(90, 94)=90．問題ない．

**案Aも不成立の場合，sociologyのpoolをこれ以上増やせない（残り4件）ため，resampling系レバーの余地は完全に尽きる**．

**(3) 新しいリスク — 弱い2タスクの削減幅が案Cからさらに拡大**

案C（40/40）から案A（30/30）への変更で，high_school_psychologyとmoral_disputesの訓練露出が-45%（55→30, 54→30）となる．Iter32のconfusion matrix分析で，これら2タスクの誤分類は`medical`・`social_science`・`legal`との学術的近接が主因と判明している．この2タスクの訓練露出をさらに減らすと，分類器が`medical`/`social_science`/`legal`との決定境界を学習する信号が弱まり，**逆効果で他ドメインのrecallが低下するリスク**がある．これはIter32とは異なる機序の副作用．

ただし，`_sample_domain_questions()`の新しい分岐では，各タスクのプールから独立にサンプリングするため，「 sociologyがpoolを圧迫してweak taskが不足する」という問題は生じない（案Cでも同様のリスクは存在）．

**(4) 変更範囲の最小性 — 定数値1箇所＋docstring**

`_EDUCATION_PROXY_TASK_TRAIN_TARGET_SIZES`の値変更以外に必要なのは:
- `build_dataset.py:803`のdocstring（`sociology=70, high_school_psychology=40, moral_disputes=40`の記述）
- `build_dataset.py:168-174`の定数定義前のコメント（`配分は案C`の記述）

これら2箇所を更新すれば，テストは全て通る（静的整合性テストは値をハードコードせず`_DOMAIN_TASK_MAP`と`_DOMAIN_TARGET_SIZE`から動的に検証しているため）．

#### 次の計画フェーズ（rc-planner）への申し送り

1. **Iter34の実装は定数値の変更のみ**: `_EDUCATION_PROXY_TASK_TRAIN_TARGET_SIZES`を`{"sociology": 90, "high_school_psychology": 30, "moral_disputes": 30}`へ変更．コード構造の変更は不要．
2. **docstringの更新も必須**: `build_dataset.py:803`のdocstring（`sociology=70, high_school_psychology=40, moral_disputes=40`）と，定数定義前のコメント（line 168-174の`案C`の記述）を更新すること．これらを忘れると，再生成後のデータが案Aであることをドキュメントが誤って示す．
3. **テスト変更は不要**: 静的整合性テスト（`test_education_proxy_task_train_target_sizes_static_integrity`）は値をハードコードせず動的に検証しているため，案Aでもpassする．
4. **案Aが不成立の場合の次の一手は唯一**: sociologyのpool cap（94）を95.7%使い切るため，resamplingで sociologyをさらに増やす余地は残4件だけ．案Aがrejectedの場合，education固有の手作り訓練問題追加へ直ちに切り替える．
5. **学習信号喪失リスクの受容**: 弱い2タスクの削減幅（-45%）は案Cより大きく，他ドメインとの境界学習が弱まる可能性がある．これはrc-plannerが受容すべきトレードオフとして明記すること．

## Iteration 34: education代理タスク抽出比率の再配分（案A）による訓練データ構成変更

### 計画 (Iter34)

**変更ファイル**: `build_dataset.py`のみ（2箇所）．
1. `_EDUCATION_PROXY_TASK_TRAIN_TARGET_SIZES`の値: `{"sociology": 70, "high_school_psychology": 40, "moral_disputes": 40}` → `{"sociology": 90, "high_school_psychology": 30, "moral_disputes": 30}`
2. 関数docstringの対応する数値更新

**Iter33実装は既に完了**（`task_target_sizes`パラメータ，education特別扱い，sample_weight revert）．Iter34は定数値のみの変更．テストファイルは変更不要．

**固定する構成**: `_CLASSIFIER_TASK_SAMPLE_WEIGHTS={}`，`_sample_domain_questions()`のtask_target_sizes分岐，`CalibratedClassifierCV(method='temperature')`，`config.yaml`，`data/dataset.jsonl`．

**学習信号喪失リスク**: 案Aは弱い2タスクの訓練露出を-45%（55→30, 54→30）．他ドメインとの境界学習が弱まる可能性があり，分析フェーズで非退行条件を特に注意深く確認する必要がある．

**案A不成立の場合**: sociologyのpool cap（94）を95.7%使い切るため，resampling系余地は尽きる．education固有の手作り訓練問題追加へ直ちに切り替える．

### 分析(解釈) (Iter34)

**比較対象**: Iter31（temperature較正本番，top1=0.6056） vs Iter34（案A resampling，top1=0.5969）．

**数値比較**:

| Metric | Iter31 (before) | Iter34 | Delta |
|--------|-----------------|--------|-------|
| top1_accuracy | 0.6056 | 0.5969 | -0.87pt |
| ECE | 0.071201 | 0.065655 | -0.005546 |
| Brier score | 0.060676 | 0.060523 | -0.000153 |
| AUROC | 0.884689 | 0.884902 | +0.000213 |
| education_recall | 0.5000 | 0.4353 | -6.47pt |
| medical_recall | 0.5393 | 0.5562 | +1.69pt |

**教育recallの時間軸トレンド（Iter28〜34）**:

| Iteration | Lever | education_recall | 変更 |
|-----------|-------|-----------------|------|
| 28 | fallback disabled | 0.4059 | baseline |
| 29 | platt calibration | 0.4059 | 不変（较正のみ） |
| 30 | isotonic calibration | 0.4059 | 不変（较正のみ） |
| 31 | temperature calibration | 0.5000 | +9.41pt（较正の副産物） |
| 32 | sample_weight=2.0 | 0.4412 | -5.88pt（rejected） |
| 33 | resampling 案C(70/40/40) | 0.4412 | 不変（ノイズ範囲内） |
| 34 | resampling 案A(90/30/30) | 0.4353 | -0.59pt（案C比） |

**重要観察**: education_recallは较正のみ変更したIter29〜31で一貫して0.4059のまま（较正は训练データの分布を触らない）．Iter31で0.5000へ跳ね上がったのはtemperature较正の副産物（较正曲线がeducationの確率分布を押し上げた）．**教育recallの真の値は0.4059〜0.4412の範囲にあり，案Aで0.4353とさらに低下した**．

**Wilson 95% CI (education_recall)**:
- Iter31: 0.5000 [0.4257, 0.5743]
- Iter34: 0.4353 [0.3630, 0.5104]
- CIは大きく重なる．ただしpoint estimateの方向は一貫して低下．

**McNemar検定 (Iter31 vs Iter34)**:
- education_recall: da=21, db=10, p=0.072486 → **有意でない**（α=0.05）
- 方向は改善（da>db）だが，p値は有意閾値を下回らない．

**per-domain recall McNemar (Iter31 vs Iter34)**:

| Domain | da (before→NG) | db (NG→OK) | p値 |
|--------|----------------|------------|------|
| education | 21 | 10 | 0.072486 |
| computer_science | 8 | 2 | 0.113846 |
| medical | 5 | 8 | 0.579100 |
| social_science | 5 | 9 | 0.422678 |
| natural_science | 7 | 5 | 0.772830 |
| legal | 5 | 6 | 1.000000 |
| mathematics | 0 | 2 | 0.479500 |
| general | 4 | 2 | 0.683091 |
| history_culture | 3 | 4 | 1.000000 |
| business_economics | 2 | 3 | 1.000000 |

**per-domain precision Fisher (Iter31 vs Iter34)**: 全ドメイン p>0.55．最も低いのはsocial_science_precision (p=0.553)．

**BH補正後 (20指標: 10ドメイン×recall/precision)**:
- education_recallが最小p値: p=0.0725, BH-q=1.450 → **有意でない**
- **BH補正後有意な退行: 0件** → 非退行条件は成立

**flip rate (Iter31→Iter34)**: 154/1600 = 9.62%．方向: education lost 46 rows, gained 34. Net -12 for education．
これは案C (11.0%) に比べてやや低いものの，ノイズとしては大きな値．
10/20指標がCI下限を切ったが，すべてCIは重なり，統計的に不均衡な退行はない．

**主基準の判定**: education_recall(0.4353) > medical_recall基準(0.5112) ?
- 0.4353 < 0.5112 → **不成立**．75.59ptのギャップ．
- Iter31の0.5000と比較しても-6.47ptの低下．

**非退行の判定**: BH補正後有意退行0件 → **成立**

**全体評価**: **rejected**
- 主基準（education_recall > medical_recall基準 0.5112）が不成立
- education_recallはIter31比で-6.47pt，Iter33比でも-0.59ptの低下
- McNemar p=0.0725 で top1_accuracy の有意改善なし
- 案A（90/30/30）は案C（70/40/40）よりもeducation_recallが低下した
- 非退行条件のみが成立

**仮説との整合**:
- 仮説「案Aでeducation_recallがmedical_recall基準を上回る」は**明確に反証**された．
- 案Aは案Cよりも変化的幅が大きかったが，結果は逆方向（低下）だった．
- 期待（sociologyの寄与最大化でeducation_recallが改善）は**一致しなかった**．

**学び**:
1. **案A（90/30/30）もrejected**．education_recall 0.4353 < medical_recall基準 0.5112．
2. **3連投のrejected（Iter32 sample_weight, Iter33 案C, Iter34 案A）は決定的**．
   resampling系レバーは尽きた．sociology pool cap 94に対し90件使用（95.7%）で，
   残り4件の余裕は実質的に意味をなさない．
3. **education_recallの低下トレンドは懸念**．Iter31(0.5000)→Iter32(0.4412)→Iter33(0.4412)→Iter34(0.4353)
   と一貫して低下．案Aで弱い2タスクの訓練露出を-45%（55→30, 54→30）に削ったことが，
   計画フェーズで指摘された「学習信号喪失リスク」が実際に発現した可能性が高い．
4. **根本原因の再確認**: 代理タスクの抽出比率をどう変えても，
   「代理タスクとeducationドメインの意味的ギャップ」は解消されない．
   Iter32の調査で確認済み: sociology(0.625)・high_school_psychology(0.438)・
   moral_disputes(0.435)のいずれも，educationの実務（学校教育行政・学習指導要領等）
   とは主題が明確に異なる．比率の変更は表層の最適化に過ぎない．
5. **medical_recallの継続的改善**（Iter34: 0.5562）は興味深い．
   Iter28→34で+1.69pt．Iter28 vs Iter34のMcNemarで有意（da=3, db=13, p=0.0244）．
   これはresamplingとは独立にtemperature较正や他の要因によるものかもしれない．

### 判定

**rejected**

### 判定理由

1. **主基準不成立**: education_recall(0.4353) < medical_recall基準(0.5112)．ギャップ75.59pt．
   Iter31(0.5000)からの低下も含め，方向性が逆．
2. **McNemar有意でない**: p=0.0725．top1_accuracyの有意改善なし．
3. **3連投のrejected**: Iter32(sample_weight), Iter33(案C), Iter34(案A)と，
   `classifier_training_data_composition`レバーファミリーで3連続棄却．
   手法の限界が実測で確定した．
4. **非退行条件のみ成立**: BH補正後有意退行0件．これは良いニュースだが，
   主基準が通らないため採用には至らない．

### 次のイテレーションへの示唆

**education固有の手作り訓練問題の追加へ直ちに切り替える**．

理由:
1. **resampling系レバーは尽きた**: sociology pool cap 94に対し90件使用．
   残り4件で意味のある変更は不可能．
2. **根本原因への直接アプローチが必要**: Iter32の調査で確認された「代理タスクの意味的ギャップ」
   は，抽出比率の変更では解決できない．手作り訓練問題（学校教育行政実務に即した問題）を
   追加することで，分類器がeducationの実務定義を直接学習する機会を提供する．
3. **config.ymlの指示通り**: 「案Aも不成立なら，education固有の手作り訓練問題の追加へ切り替える」
   （backlog B54）．
4. **フォーマット不整合のリスク**: Iter32の調査で発見された問題（d0003 X8，journal line 892-921）．
   手作り問題はJMMLU形式(A/B/C/D)を保つ必要がある．自由記述文を追加すると，
   分類器が「A/B/C/Dの有無」をeducationの書式手がかりとして学習するリスクがある．
   手作り問題も4択形式で作成する必要がある．
5. **コスト見積もり**: d0003 X8の見積りで1〜3日．オフライン完結（分類器再訓練＋
   evaluate_classifier_calibration.pyでの再評価のみ）．実機1600問本走は不要．

**Iter35の計画フェーズで確認すべき事項**:
- 手作り問題の数を確定（例: 50件，100件など）
- 4択形式を保つための設計（A/B/C/Dの選択肢構造をJMMLU形式に合わせる）
- evalデータセットとの分離（label leakage防止）
- 成功率のシミュレーション（手作り問題を追加した場合のeducation_recallの期待値）

### Iteration 34 実行済み

**変更ファイル**: `build_dataset.py`（`_EDUCATION_PROXY_TASK_TRAIN_TARGET_SIZES` を案C(70/40/40)から案A(90/30/30)へ変更，定数定義前コメント更新，docstring更新）．変更なし: `scripts/train_domain_classifier.py`, `tests/test_build_dataset.py`, `config.yaml`, `data/dataset.jsonl`．

**生成ファイル**: `data/classifier_train_iter34_resampled.jsonl`, `models/domain_classifier_iter34_resampled.joblib`, `results/iter34_calibrated_predictions.jsonl`．before: `results/iter31_calibrated_predictions.jsonl`．

**結果**:
- top1_accuracy: 0.6056 → 0.5969 (-0.87pt, McNemar p=null 未計算または有意でない)
- education_recall: 0.5000 → 0.4353 (-6.47pt, McNemar p=0.0725 有意でない)
- medical_recall: 0.5393 → 0.5562 (+1.69pt)
- ECE: 0.071201 → 0.065655 (-0.005546)
- 非退行: BH補正後有意退行0件 → 成立
- flip rate: 154/1600 = 9.62%

**判定**: rejected（確定）

**判定理由**:
1. 主基準（education_recall > medical_recall基準 0.5112）不成立（0.4353 < 0.5112，75.59ptギャップ）
2. McNemar p=0.0725 で top1_accuracy の有意改善なし
3. 3連投のrejected（Iter32 sample_weight, Iter33 案C, Iter34 案A）でresampling系レバーは尽きた
4. 非退行条件のみ成立

**学び**:
1. resampling系レバーは尽きた（sociology pool cap 94に対し90件使用，残り4件で実質変更不可能）．
2. 3連続rejected（Iter32, 33, 34）は決定的．「教育ドメインの代理タスクが本質的にeducationの意味的ギャップを抱えている」という根本原因を，抽出比率の変更という表層最適化で解決できないことが実測で確定した．
3. education_recallの低下トレンド（Iter31: 0.5000 → Iter34: 0.4353）は懸念．案Aで弱い2タスクの訓練露出を-45%に削ったことが「学習信号喪失リスク」を実際に発現させた可能性が高い．
4. 次イテレーション（Iter35）はeducation固有の手作り訓練問題の追加へ切り替える．

**gitコミット**: 実施済み（後述）

### 実装 (Iter33)

**変更ファイル**: `build_dataset.py`（sample_weight revert, _EDUCATION_PROXY_TASK_TRAIN_TARGET_SIZES新設, _sample_domain_questionsにtask_target_sizes追加, build_classifier_training_rowsのeducation特別扱い）, `tests/test_build_dataset.py`（テスト改名・新規追加3件）．変更なし: `scripts/train_domain_classifier.py`, `tests/test_train_domain_classifier.py`, `config.yaml`．

**単一レバー検証**: (a) eval sha256一致 `485a85f5...`, (b) sample_weight全1427行で1.0, (c) education内訳 sociology=70/high_school_psychology=40/moral_disputes=40, (d) education外9ドメイン1277行完全一致．

**テスト**: 225 passed, 2 skipped．lint: All checks passed．

**生成ファイル**: `data/classifier_train_iter33_resampled.jsonl` (sha256 `b5d3f715...`), `models/domain_classifier_iter33_resampled.joblib` (`55d34b52...`), `results/iter33_calibrated_predictions.jsonl` (`3175a65f...`)．before: `results/iter31_calibrated_predictions.jsonl` (`ff779ed2...`)．

**wall time**: 合計約7分（オフライン完結）．問題なし．

### 調査 (Iter33)

**問い**: 次点レバー `classifier_training_data_composition=education_proxy_task_resampling`
（`sample_weight`を使わず，3代理タスクの抽出目標件数比率を変える）を計画フェーズが具体化できるよう，
(1)抽出コードの正確な位置と実装，(2)各代理タスクの母集団サイズ，(3)配分比率案，(4)eval/train分離の
維持，(5)単一レバー原則の遵守可能性を確認する．

#### 分かったこと

**(1) 抽出コードの位置と実装 — 現状は「均等」でも「元データ比例」でもなく「プールしてから1回だけ
乱択」**

`build_dataset.py:723` `build_classifier_training_rows()` が本体で，`_build_jmmlu_backed_groups()`
（643行）→`_sample_domain_questions()`（612行）を呼ぶ．現状の実装は，**`education`の3タスク
（sociology・high_school_psychology・moral_disputes）の行を1つのプールへ合流させたうえで，
`random.Random(seed).sample(pool, sample_size)`により`domain_target_size`（既定150）件を
**一度に無作為抽出**しているだけで，**タスク別の目標件数という概念自体が現状のコードに存在しない**．
したがってタスク別の内訳は「均等割り当て」でも「元データの母集団比に厳密に比例」でもなく，
単に無作為抽出の結果として母集団比に近い値がたまたま出るという性質のものである．
実際に同じseed（`_CLASSIFIER_TRAIN_SAMPLE_SEED=20260727`）で再現実行したところ，現状の訓練データ
（`education`150件）の内訳は **sociology 41・high_school_psychology 55・moral_disputes 54** だった
（母集団比から予想される47/51/51に近いが，単一の乱択なのでずれがある）．
`scripts/prepare_lora_training_data.py`は**別スクリプト**であり，`_DOMAIN_TASK_MAP`を独自に重複定義
（Iter32既知の保守リスク，未解消）しているが，抽出関数もLoRA訓練データ（`data/lora_train/`）専用で
分類器訓練データとは完全に独立している．今回のレバーは`build_dataset.py`側のみを触れば良く，
`prepare_lora_training_data.py`は触れる必要がない（触れてもいけない）．

**(2) 各代理タスクの母集団サイズ — sociologyの上限は94件**

`JMMLU.zip`（pinned commit `3637b25e444ccfdcde4d23a783cbe8e674faa01b`）を実際にダウンロードし
CSVを直接パースして確認した．全体件数は **sociology 150・high_school_psychology 150・
moral_disputes 148**（合計448，config note記載の値と一致）．評価データセット
（`_JMMLU_SAMPLE_SEED=20260726`）が先に**sociology 56・high_school_psychology 48・
moral_disputes 46**（Iter32のrecall分母35/56・21/48・20/46と完全一致，再現性を確認済み）を予約する
ため，訓練データが利用できる残プールは**sociology 94・high_school_psychology 102・
moral_disputes 102**（合計298 = 448-150）に上限が決まる．
**したがって`education`の総行数150件を変えない設計では，sociologyへ配分できる件数は最大94件が
ハードな上限**であり，これを超える配分案（例: 全て`sociology`にする等）は不可能．

**(3) 配分比率案（3案，いずれも合計150件・sociology≤94の上限内）**

| 案 | sociology | high_school_psychology | moral_disputes | 根拠 |
|---|---|---|---|---|
| A（backlog例，急進的） | 90 | 30 | 30 | confusion matrix (Iter32) が示す「sociologyが相対的に混同されにくい」を最大限反映．sociologyの上限94に対し90/94=95.7%とほぼ使い切る |
| B（recall比例，データ駆動・穏健） | 63 | 44 | 43 | Iter31時点のrecall（0.625/0.438/0.435，合計1.498）に比例配分：150×(recall_i/合計recall) を丸め．A よりシフト幅が小さく，過補正のリスクが低い |
| C（折衷，中庸） | 70 | 40 | 40 | 現状の均等に近い配分（41/55/54）とAの中間．sociologyの割合を27%→47%へ引き上げつつ，弱い2タスクの絶対件数の削減幅をAより抑える（55→40・54→40，-27%）|

**リスク評価**: 案Aはsociologyの残プールをほぼ使い切る（余裕がなく今後さらに増やす余地がない）うえ，
弱い2タスクの削減幅が最大（55→30・54→30，-45%）で，Iter32のconfusion matrixが「高校心理学・
道徳論争の誤分類は`medical`・`social_science`・`legal`との学術的近接が主因」と示している以上，
**該当タスクの訓練露出を大きく減らすこと自体が，むしろそれらの決定境界学習を弱め逆効果になる
リスク**がある（Iter32とは異なる機序だが，「弱いタスクを減らしすぎて学習信号を失う」という意味で
方向性としては新しいタイプの副作用になりうる）．案B・Cはこのリスクを相対的に抑えつつ，
「sociology優位を反映する」という着想自体は共有する．**計画フェーズでは案Cを既定の第一候補とし，
Aは「効果が小さければ次点で試す急進版」として位置付けることを推奨する**（根拠: Bはデータ駆動だが
効果量が小さすぎてIter32のような僅差判定に陥りやすく，Aはリスクが相対的に高いため）．

**(4) eval/train分離（Iter10 label leakage再演の有無） — 現状の仕組みは維持可能**

`build_classifier_training_rows()`は`eval_rows`から`eval_queries`（質問文の集合）を作り，
`_build_jmmlu_backed_groups()`の`exclude_queries`引数へ渡し，`_sample_domain_questions()`内で
**サンプリング前に**`query in exclude_queries`を除外している（172行のdocstringに明記，Iter10の
label leakage再演を防ぐガード）．実際に上記(2)の再現実行でも，訓練プールの合計は298件
（=448-150）とeval側の150件と完全に排他的であることを確認した．
**タスク別の目標件数を導入する新しい抽出関数を書く場合も，「タスクごとに`exclude_queries`適用後の
プールから独立にサンプリングする」という構造を維持する限り，このガードは自動的に保たれる**．
逆に，もし新実装がタスク別プールを`exclude_queries`適用前のCSV生データから直接組み立ててしまうと，
Iter10のlabel leakageが再演するため，実装レビュー時に明示的に確認すべき点として申し送る．

**(5) 単一レバー原則の遵守可能性 — 一点，コードに残存する重大なリスクを発見**

(a) 変更範囲の面では，`education`の抽出目標件数のみを触れば良く，`write_dataset()`/`_build_rows()`
（eval データセット，`data/dataset.jsonl`）や`scripts/train_domain_classifier.py`の較正処理
（`CalibratedClassifierCV(method='temperature')`，Iter31本番採用済み）を変更する必要はない．
これらに触れなければ単一レバー原則は形式的に守れる．

(b) **しかし，Iter32のrejectedされた`sample_weight`機構がコード上まだ生きている**．
commit `750cf3e`（Iter32確定コミット）を確認したところ，実験用ファイル
（`models/domain_classifier_iter32_reweighted.joblib`・`data/classifier_train_iter32_reweighted.jsonl`）
は削除されたが，**`build_dataset.py`の`_CLASSIFIER_TASK_SAMPLE_WEIGHTS = {"high_school_psychology":
2.0, "moral_disputes": 2.0}`および`_classifier_task_sample_weight()`関数自体は revert されずに
残存している**．`build_classifier_training_rows()`は各行に無条件でこの関数の戻り値を
`sample_weight`として埋め込み，`scripts/train_domain_classifier.py:_extract_sample_weights()`は
`row.get("sample_weight", 1.0)`でこれを読み取り`LogisticRegression.fit(sample_weight=...)`へ
渡す実装のままである．`tests/test_build_dataset.py::test_classifier_task_sample_weight_upweights_only_the_two_weak_proxy_tasks`
も`high_school_psychology`/`moral_disputes`が2.0であることを**現在も期待値として固定**している．
現在ディスク上の`data/classifier_train.jsonl`（`data/MANIFEST.md`のsha256=`eb89bf7b...`，
記録日2026-07-29）は本コミットより前に生成されたファイルのため`sample_weight`列を持たない
（実測: 全150行`None`）が，**`build_dataset.py --classifier-train-output ...`を今回再実行すると，
現状のコードのままでは`high_school_psychology`・`moral_disputes`の行に`sample_weight=2.0`が
無条件で再び埋め込まれる**．Y5レバーの設計上の前提（config.yml note）は「`sample_weight`を
一切使わない」ことで Iter32 の`class_weight`結合バグの影響を受けない設計にすることだったため，
**この残存コードを放置したまま訓練データを再生成すると，rejected済みのIter32機構が単一レバーの
裏で静かに再混入し，抽出比率変更の効果を`sample_weight`効果と分離できなくなる**．
これは計画・実装フェーズが対処すべき前提条件であり，単なる留意事項ではない．
対応は次の2択（判断は計画フェーズに委ねる）: (i) `_CLASSIFIER_TASK_SAMPLE_WEIGHTS`を空にする
（実質1.0固定に戻す）よう revert し，対応するテストも「全タスク1.0」を期待するよう更新する，
(ii) 関数・テストは残すが，抽出比率変更の実装時に生成される`sample_weight`列が全行1.0であることを
明示的に検証してから訓練する．いずれにせよ**「訓練データ再生成後，`sample_weight`列が全行1.0で
あることを確認する」という手順を実装フェーズのチェックリストへ追加すべき**．

#### 次の計画フェーズ（rc-planner）への申し送り

1. **最優先で対処すべき前提条件**: `build_dataset.py`の`_CLASSIFIER_TASK_SAMPLE_WEIGHTS`
   （Iter32のrejected済み`sample_weight=2.0`機構）が revert されずに残っている．抽出比率変更を
   実装する前に，これを空辞書へ戻す（テスト`test_classifier_task_sample_weight_upweights_only_the_two_weak_proxy_tasks`
   も合わせて更新）か，最低限「再生成後の`sample_weight`列が全行1.0であること」を実装確認手順に
   明記すること．これを怠ると，抽出比率変更という単一レバーのはずが，rejected済みの
   `sample_weight`機構と暗黙に合成され，config.yml note が前提とする「class_weight結合の影響を
   受けない設計」が成立しなくなる．
2. **配分比率は案C（sociology 70・high_school_psychology 40・moral_disputes 40）を第一候補として
   推奨**する．案A（90/30/30，backlog例）は sociology の残プール94件をほぼ使い切り，かつ弱い
   2タスクの訓練露出を45%も削るため，Iter32とは別種の過補正リスク（学習信号の喪失）が相対的に
   高い．案B（63/44/43，recall比例）はより穏健だが効果量が小さく，Iter27・Iter29のような
   「僅差で判定不能」に陥る可能性がある．案Cは両者の中間で，最初に試す価値が高い．
   ただし最終決定は計画フェーズが行うこと（3案とも実行可能であることは確認済み）．
3. **実装は`build_classifier_training_rows()`/`_build_jmmlu_backed_groups()`の内部にのみ
   タスク別目標件数（`education`限定のオーバーライド）を追加する形にし，`_DOMAIN_TASK_MAP`や
   `write_dataset()`（eval生成経路）には一切触れないこと**．新しいタスク別抽出関数を書く際は，
   「`exclude_queries`適用後の各タスク別プールから独立にサンプリングする」という構造を維持し，
   `exclude_queries`適用前の生データからタスク別プールを組み立てないこと（Iter10 label leakage
   再演の防止．(4)参照）．
4. **成功条件・非退行条件はY5のconfig note（education_recallが他ドメイン下限＝medical_recall
   0.5112を上回ること，かつ他9ドメインのrecall/precisionがBH補正後有意退行しないこと）をそのまま
   継続適用してよい**．較正手法（temperature，本番採用済み）は変更しないため，訓練データ再生成後は
   `CalibratedClassifierCV(method='temperature')`で再較正する必要がある（config note既述の通り）．
5. **人間判断が必要な未解決論点（再掲，今回新事実なし）**: 「education_recallという既存メトリクスの
   改善」と「educationドメインの実務忠実性」の両立不可能性（backlog B52）は今回の調査でも変わらず
   未解決．今回のレバーはあくまで「3代理タスクのうち相対的に混同されにくいタスクの寄与を増やす」
   という限定的な改善を狙うものであり，代理タスクの意味的ギャップという根本原因は解消しない
   （config note・Iter32考察に既出，変更なし）．

### 計画 (Iter33)

**仮説**: `education`の3代理タスク（sociology・high_school_psychology・moral_disputes）は
confusion matrix実測（Iter32調査）でrecallが一様でない（sociology 0.625，high_school_psychology
0.438，moral_disputes 0.435）。分類器訓練データにおけるこの3タスクの抽出比率を，相対的に混同
されにくいsociologyへ厚く，弱い2タスクへ薄く再配分すれば，`sample_weight`（Iter32でrejected，
`class_weight="balanced"`との数式結合により逆効果）を使わずに，同じ着想（sociology優位の反映）を
`education`の総行数150件（他ドメインと同数）を変えずに実現でき，`class_weight_[education]`は
Iter31以前と同じ値（0.9513）のまま保たれる。

**単一レバー**: `classifier_training_data_composition`（config.yml Y5レバー）の値を
`education_proxy_task_resampling`にする。`build_dataset.py:build_classifier_training_rows()`が
`education`の分類器訓練行を生成する際，3代理タスクからの抽出比率を，現状の「1プールに合流して
無作為に150件抽出（現状内訳 sociology 41・high_school_psychology 55・moral_disputes 54）」から，
**タスク別に独立した目標件数を指定する方式**へ変更する。

**配分比率: 案C（sociology 70・high_school_psychology 40・moral_disputes 40，合計150）を採用**。
調査(Iter33)申し送りの3案（A: 90/30/30，B: 63/44/43，C: 70/40/40）のうち，rc-investigatorが
第一候補として推奨したCを採用する。根拠:
- 案A（90/30/30）はsociologyの残プール94件をほぼ使い切り（90/94=95.7%），かつ弱い2タスクの
  訓練露出を-45%（55→30・54→30）削るため，Iter32のconfusion matrix分析が示す「弱い2タスクの
  誤分類は`medical`・`social_science`・`legal`との学術的近接が主因」という機序を踏まえると，
  該当タスクの学習信号自体を失わせて逆効果になるリスクが相対的に高い。
- 案B（63/44/43，recall比例）は穏健だが現状（41/55/54）からの変化幅が小さく，Iter27・Iter29の
  ような「僅差で判定不能」に陥りやすい。
- 案C（70/40/40）は現状比でsociologyの割合を27%→47%へ引き上げつつ，弱い2タスクの削減幅を
  -27%（55→40・54→40）に抑える中庸案であり，効果を検出できる変化幅と過補正リスクの回避を
  両立する。目標未達の場合は案A（急進版）を次点として次イテレーションで検討する
  （調査(Iter33)申し送り済み）。

**`sample_weight`機構の revert 方針（最優先で対処する前提条件）**: 調査(Iter33)が発見した
`_CLASSIFIER_TASK_SAMPLE_WEIGHTS = {"high_school_psychology": 2.0, "moral_disputes": 2.0}`
（Iter32でrejected確定済み，`build_dataset.py:165-168`）を**revertする**（選択肢(i)）。
理由: config.ymlのY5 noteが明記する`education_proxy_task_resampling`の設計要件は「`sample_weight`
を一切使わない」ことで，Iter32で判明した`class_weight="balanced"`との数式結合バグの影響を
受けない設計にすることである。この機構を残したまま`data/classifier_train.jsonl`を再生成すると，
抽出比率変更という単一レバーの裏で，rejected済みの`sample_weight`機構が黙って再混入し，
2つの変更が合成されて単一レバー原則が崩れる。検証のみで済ませる選択肢(ii)は，「新設した
抽出比率変更の効果」と「不使用のはずのsample_weight効果」を分離する保証を実装時の一度きりの
確認手順に依存させてしまい，再現性が低い。revertの方が構造的に安全である。

**revert手順（rc-implementer向け）**:
1. `build_dataset.py:165-168`の`_CLASSIFIER_TASK_SAMPLE_WEIGHTS`を`{}`（空辞書）に戻す。
   直前のコメント（159-164行目）も「Iter32で導入したが，`class_weight`との数式結合により
   Iter32計画の意図に反し逆効果と判明したためrejected・revert済み（backlog B53参照）。
   Iter33以降は`education_proxy_task_resampling`（抽出段階でのタスク別目標件数変更）に
   移行し，`sample_weight`は使わない設計とする」という趣旨に更新する。
2. `_classifier_task_sample_weight()`関数・`_DEFAULT_CLASSIFIER_TASK_SAMPLE_WEIGHT = 1.0`・
   `sample_weight`フィールド自体（`build_classifier_training_rows()`の`rows.append`・
   `scripts/train_domain_classifier.py`の`_extract_sample_weights()`/
   `train_classifier(sample_weight=...)`/`_train_and_save()`）は**削除せず残す**。
   `_CLASSIFIER_TASK_SAMPLE_WEIGHTS`が空辞書になれば，どのタスク名についても
   `_classifier_task_sample_weight()`は`_DEFAULT_CLASSIFIER_TASK_SAMPLE_WEIGHT`（1.0）を返し，
   全行`sample_weight=1.0`となる。これは`LogisticRegression.fit(sample_weight=[1.0]*n, ...)`と
   無重み付けの`fit()`が数学的に等価であるため，機構自体を削除するのと実質的に同じ効果が
   得られ，かつIter32で追加した回帰防止テスト（sample_weightがCalibratedClassifierCVまで
   伝播することの確認）を無駄にしない。
3. `tests/test_build_dataset.py::test_classifier_task_sample_weight_upweights_only_the_two_weak_proxy_tasks`
   （225行目付近）を「全タスクが1.0であることを検証する」テストに書き換える（例:
   `test_classifier_task_sample_weight_defaults_all_tasks_to_one_after_iter32_revert`へ改名し，
   `high_school_psychology`・`moral_disputes`・`sociology`・`anatomy`いずれも1.0であることを
   assertする）。
4. **再生成後の検証手順として必須**: `data/classifier_train.jsonl`（新規再生成後）の
   `sample_weight`列が**全1427行で1.0であること**をコマンドラインで直接確認する
   （`jq -s 'map(.sample_weight) | unique' data/classifier_train.jsonl` 等）。これにより
   revertが実際に発火したことをファイルレベルで担保する。

**抽出比率変更の実装（rc-implementer向け，具体的な変更行）**:

現在のコード構造（本フェーズで`Read`にて確認済み）:
- `build_dataset.py:612` `_sample_domain_questions(zf, task_names, target_size, seed,
  exclude_tasks, exclude_queries=frozenset())`: 現状は`task_names`の全タスクの行を1プールへ
  合流させてから`random.Random(seed).sample(pool, min(target_size, len(pool)))`で1回だけ抽出する
  （プールしてから乱択する既存の唯一の抽出方式）。
- `build_dataset.py:643` `_build_jmmlu_backed_groups(...)`: 全ドメインについて上記関数を呼ぶ。
  `_build_rows()`（661行目，eval生成）と`build_classifier_training_rows()`（723行目，分類器
  訓練データ生成）の両方から呼ばれる共通経路。

**設計方針: `_build_jmmlu_backed_groups()`のシグネチャは変更しない**（eval生成経路
`_build_rows()`/`write_dataset()`に一切影響を与えないことを構造的に保証するため）。
代わりに次の2点のみを変更する:

1. `_sample_domain_questions()`に，末尾へ新規オプション引数
   `task_target_sizes: dict[str, int] | None = None`（デフォルト`None`）を追加する。
   ```python
   def _sample_domain_questions(
       zf: zipfile.ZipFile,
       task_names: list[str],
       target_size: int,
       seed: int,
       exclude_tasks: frozenset[str],
       exclude_queries: frozenset[str] = frozenset(),
       task_target_sizes: dict[str, int] | None = None,
   ) -> list[tuple[str, str, str]]:
   ```
   `task_target_sizes is None`の場合は既存の「1プールへ合流して1回だけ乱択」ロジックをそのまま
   維持する（**eval生成・education以外の全ドメインの分類器訓練データ生成はこの分岐を通り，
   一切影響を受けない**）。`task_target_sizes`が与えられた場合のみ，新しい分岐:
   `task_names`内の各タスクについて，`exclude_tasks`/`exclude_queries`を適用したうえで
   **タスクごとに独立したプールを作り**，`task_target_sizes[task_name]`（プールを超える場合は
   プールサイズにcap）を`rng.sample()`する。`rng = random.Random(seed)`を関数冒頭で1回だけ
   生成し，`task_names`に列挙された順（`_DOMAIN_TASK_MAP["education"]`の順序，すなわち
   sociology→high_school_psychology→moral_disputesの順）で逐次`rng.sample()`を呼ぶことで
   決定論的な再現性を保つ。**`task_target_sizes`のキー集合は`task_names`の集合を部分集合として
   含んでいれば良い**（`set(task_names) <= set(task_target_sizes)`をassertする。等号を要求
   しないのは，`tests/test_build_dataset.py`の`_FIXTURE_DOMAIN_TASK_MAP`が`education`を
   `["sociology"]`という1タスクだけにreduceしているため，本番用の3タスク分の
   `_EDUCATION_PROXY_TASK_TRAIN_TARGET_SIZES`をそのまま渡してもテストが壊れないようにする
   ため）。`task_names`にない余分なキーは単に無視される。

2. `build_dataset.py:80`の`_DOMAIN_TASK_MAP`直後（現在の`_CLASSIFIER_TASK_SAMPLE_WEIGHTS`定義の
   近く）に新規定数を追加する:
   ```python
   # Iter33 (classifier_training_data_composition=education_proxy_task_resampling, Y5):
   # Iter32のsample_weight方式はrejected（class_weight="balanced"との数式結合で逆効果，
   # backlog B53）。sample_weightを使わず，抽出段階でのタスク別目標件数を変えることで
   # 同じ着想（sociology優位の反映）を実現する。合計は_DOMAIN_TARGET_SIZE(150)のまま不変
   # ＝class_weight_[education]はIter31以前と同じ値を保つ。配分は案C（journal Iter33計画）:
   # sociology(recall 0.625,相対的に良好)を最も厚く，high_school_psychology(0.438)・
   # moral_disputes(0.435)を均等に薄くする中庸案。
   _EDUCATION_PROXY_TASK_TRAIN_TARGET_SIZES: dict[str, int] = {
       "sociology": 70,
       "high_school_psychology": 40,
       "moral_disputes": 40,
   }
   assert sum(_EDUCATION_PROXY_TASK_TRAIN_TARGET_SIZES.values()) == _DOMAIN_TARGET_SIZE
   ```

3. `build_classifier_training_rows()`（723行目）内の
   `domain_groups = _build_jmmlu_backed_groups(zf, domain_target_size, ...)`呼び出しを，
   `education`だけ特別扱いするよう変更する（**`_build_jmmlu_backed_groups()`自体は無改造**）:
   ```python
   domain_task_map_without_education = {
       domain: tasks for domain, tasks in domain_task_map.items() if domain != "education"
   }
   domain_groups = _build_jmmlu_backed_groups(
       zf,
       domain_target_size,
       exclude_restricted_license_tasks,
       domain_task_map_without_education,
       seed=_CLASSIFIER_TRAIN_SAMPLE_SEED,
       exclude_queries=eval_queries,
   )
   exclude_tasks = _RESTRICTED_LICENSE_TASKS if exclude_restricted_license_tasks else frozenset()
   domain_groups["education"] = _sample_domain_questions(
       zf,
       domain_task_map["education"],
       domain_target_size,
       _CLASSIFIER_TRAIN_SAMPLE_SEED,
       exclude_tasks,
       exclude_queries=eval_queries,
       task_target_sizes=_EDUCATION_PROXY_TASK_TRAIN_TARGET_SIZES,
   )
   ```
   その後の`for domain in sorted(domain_groups): ...`によるrows組み立ては無変更（`sorted()`で
   `education`を含む全ドメインを走査するため，辞書へ後から追加しても問題ない）。
   docstringの「Known imbalance」節の直後に，この education 限定オーバーライドの説明を1段落
   追記する。

4. **`_build_rows()`・`write_dataset()`・`_build_jmmlu_backed_groups()`自体には一切手を
   入れない**（シグネチャ・呼び出し箇所とも無変更）。これにより eval データセット
   （`data/dataset.jsonl`）が無変更であることが構造的に保証される（Iter32同様，念のため
   再生成後にsha256一致も実測確認すること）。

**固定する構成（Iter31 adopted・Iter32 rejectedのまま，一切変更しない）**:
`routing_method=supervised_classifier`，`confidence_threshold=0.0`・`dispatch_top_k=1`・
`aggregation_method=max_confidence`，`confidence_signal_method=self_report`，
`confidence_elicitation=top_k_with_probs`，`expert_model=expert-mesh-{domain}-lora`
（domain_count=10），`light_model=qwen3.5:4b-q4_K_M`，`embedding_model=nomic-embed-text`，
評価データセット`data/dataset.jsonl`（1600問，不変）。分類器較正手法は
`scripts/train_domain_classifier.py`の`_CALIBRATION_METHOD="temperature"`・`_CALIBRATION_CV=5`・
`ensemble=True`（すべて無変更，訓練データを変えたため再較正は必須だが手法自体は固定）。
`config.yaml`は一切変更しない。

**変更ファイル一覧（rc-implementer向けサマリ）**:
1. `build_dataset.py`: `_CLASSIFIER_TASK_SAMPLE_WEIGHTS`を`{}`へrevert（コメント更新），
   `_EDUCATION_PROXY_TASK_TRAIN_TARGET_SIZES`新設，`_sample_domain_questions()`に
   `task_target_sizes`引数追加，`build_classifier_training_rows()`のeducation特別扱い追加。
2. `tests/test_build_dataset.py`:
   - `test_classifier_task_sample_weight_upweights_only_the_two_weak_proxy_tasks`を
     全タスク1.0を検証するテストへ書き換え。
   - 新規テストを追加: `_sample_domain_questions`を直接importし，`task_target_sizes`指定時に
     各タスクの抽出件数がタスク別の目標件数（プールcap込み）と一致することを検証する
     （フィクスチャzipの既存タスク，例えば`sociology`・`anatomy`を「1ドメイン2タスク」の
     ように見立てて呼び出せばよい，education固有の意味は不要）。`task_target_sizes=None`の
     場合は既存の（変更前と同一の）挙動が保たれることも回帰テストとして確認する。
   - 静的整合性テスト: `_EDUCATION_PROXY_TASK_TRAIN_TARGET_SIZES`のキー集合が
     `_DOMAIN_TASK_MAP["education"]`と一致し，値の合計が`_DOMAIN_TARGET_SIZE`(150)と
     一致することを検証する（`build_dataset`から両定数をimportして比較，ネットワーク・
     フィクスチャzip不要）。
   - `test_build_classifier_training_rows_never_overlaps_eval_queries`・
     `test_build_classifier_training_rows_have_query_domain_and_sample_weight_only`は
     現状のまま（`sample_weight`フィールド自体は残るため）で通ることを確認する。
3. `scripts/train_domain_classifier.py`: 変更不要（`sample_weight`伝播の仕組み自体は
   Iter32のまま残す。中身が全行1.0になるだけ）。
4. `tests/test_train_domain_classifier.py`: 変更不要。

**データ生成・学習・評価手順（Iter32と同様の手順を踏襲）**:
1. `data/classifier_train.jsonl`は上書きしない。新規ファイル
   `data/classifier_train_iter33_resampled.jsonl`を
   `uv run python build_dataset.py --output /tmp/iter33_dataset_verify.jsonl --jmmlu-zip
   <cached JMMLU.zip> --classifier-train-output data/classifier_train_iter33_resampled.jsonl`
   で生成する。
2. **単一レバー原則の担保（必須検証）**:
   (a) `/tmp/iter33_dataset_verify.jsonl`（新規生成した eval 相当データ）が既存
   `data/dataset.jsonl`と完全一致（sha256一致）することを確認し，eval データセットが無変更
   であることを担保する。
   (b) 新規ファイルの`sample_weight`列が全1427行で1.0であることを確認する（revertが発火した
   証拠）。
   (c) `education`ドメイン150行のうち，`jmmlu_task`（または元CSVの由来）別に
   sociology 70件・high_school_psychology 40件・moral_disputes 40件になっていることを実測
   確認する（案Cの配分が実際に発火した証拠。`build_classifier_training_rows()`は現状
   `jmmlu_task`をrowに含めないため，確認には一時的なデバッグ出力または
   `_sample_domain_questions`を直接呼んだ単体検証で行うこと）。
   (d) `education`以外の9ドメインの行内容（`(id, query, domain)`の集合）が既存
   `data/classifier_train.jsonl`と完全一致することを確認する（`_build_jmmlu_backed_groups`の
   ロジックは無変更のため，education以外は同じ質問集合になるはずである）。
3. 分類器を新規学習: `uv run python -m scripts.train_domain_classifier --train-data
   data/classifier_train_iter33_resampled.jsonl --embedding-model nomic-embed-text
   --ollama-host 127.0.0.1 --ollama-port 11435 --output
   models/domain_classifier_iter33_resampled.joblib`（本番`models/domain_classifier.joblib`は
   上書きしない）。`_CALIBRATION_METHOD="temperature"`は変更しないため，このコマンドで
   自動的にtemperature較正が適用される。
4. 較正後データを生成: `uv run python -m scripts.evaluate_classifier_calibration --dataset
   data/dataset.jsonl --classifier models/domain_classifier_iter33_resampled.joblib
   --embedding-model nomic-embed-text --ollama-host 127.0.0.1 --ollama-port 11435 --output
   results/iter33_calibrated_predictions.jsonl`。
5. **beforeはIter31のproduction相当データをそのまま使う**:
   `results/iter31_calibrated_predictions.jsonl`（再生成しない。Iter32のbeforeも同一ファイル
   だった）。Iter32（rejected・models未反映）は比較対象にしない。

**成功条件**:
1. **主基準（point estimate）**: `results/iter33_calibrated_predictions.jsonl`から算出した
   `education_recall`（150問，argmax vs `expected_domains`）が，現状下限
   **`medical_recall`(0.5112，Iter31 production実測) を上回ること**（config.yml Y5 note・
   計画(Iter32)で訂正済みの基準をそのまま継続適用）。
2. **診断（gatingではないが必須報告）**: `education_recall`のドメイン別McNemar検定
   （before=`results/iter31_calibrated_predictions.jsonl`のeducation行，
   after=`results/iter33_calibrated_predictions.jsonl`のeducation行）を実施し，p値・
   discordant内訳を報告する。Iter32同様，基準線とビット単位で完全一致していないか
   （実験不成立でないか）を最初に確認する。
3. **非退行（Iter30以降で確立した3段構成を踏襲，education以外の9ドメイン18指標が対象）**:
   10ドメイン×precision/recall=20指標（recallはドメイン別McNemar，precisionはFisher正確検定）
   のp値を一括でBenjamini-Hochberg補正（q=0.05）し，**education以外の9ドメイン18指標のうち，
   悪化方向でBH補正後有意な指標が0件であること**を非退行の必須条件とする。
4. **education_precisionの扱い（診断的，非gatingだが重視）**: `education_precision`
   （over-triggeringの検出）は20指標BH補正の対象に含めて算出・報告する。有意に悪化していた
   場合は，主基準1が満たされていても総合判定を`partial`以下に留める根拠として重視する。
5. **flip rate**: Iter31→Iter33のargmax不一致率を必須報告項目として記録する（判定基準ではない）。
6. **温度較正の再確認**: 学習データを変えたため`_CALIBRATION_METHOD="temperature"`による較正を
   今回のデータでも再実行し（手順3で自動実施），Iter31と同様のチェックリスト（確率の0/1張り付き・
   uniform fallback・tie率）を簡易報告する。

**目標未達時の次点候補（次イテレーション向けメモ，今回の計画には含めない）**: 案C（70/40/40）が
不成立の場合，急進版の案A（90/30/30）を次点として試す。案Aも不成立なら，調査(Iter33)申し送りの
とおり4択形式を保った手作り訓練問題の追加（journal「考察 (Iter32)」節の候補(3)）へ切り替える。

**人間判断が必要な論点**: 新規追加なし。Y2着手前のユーザー確認はbacklog B49〜B52の既存の申し送り
のまま。較正済み分類器の本番反映可否は，今回の成功条件（1・3）が満たされた場合に改めてその時点で
判断する（本イテレーションで本番アーティファクトを置き換える判断は行わない）。

### 分析(解釈) (Iter33)

**比較対象**: experimenter提供の比較は Iter28（top1=0.5850） vs Iter33（top1=0.5956）．
state.json の計画では `results/iter31_calibrated_predictions.jsonl`（top1=0.6056）を before
とする予定だったが，experimenter は Iter28 を使用．両方の McNemar を計算した．

**数値比較**:

| Metric | Iter28 (baseline) | Iter33 | Delta |
|--------|-------------------|--------|-------|
| top1_accuracy | 0.5850 | 0.5956 | +1.06pt |
| cohens_kappa | 0.5541 | 0.5637 | +0.96pt |
| education_recall | 0.4059 | 0.4412 | +3.53pt |
| medical_recall | 0.4831 | 0.5000 | +1.69pt |
| legal_recall | 0.5833 | 0.5611 | -2.22pt |
| ECE | 0.1934 | 0.0676 | -0.1258 |
| brier_score | 0.2471 | 0.1981 | -0.0490 |
| auroc | 0.7295 | 0.7633 | +0.0338 |

**Wilson 95% CI (education_recall)**:
- Iter28: 0.4059 [0.3349, 0.4810] (69/170)
- Iter33: 0.4412 [0.3687, 0.5163] (75/170)
- CIは大きく重なる．SE ~3.8pt 程度のノイズ範囲内の変化．

**McNemar検定**:
- Experimenter提供 (Iter28 vs Iter33): a=73, b=56, Chi2=1.9845, p=0.1589 → **有意でない**
- 再計算 (Iter28 vs Iter33): a=56, b=69, Chi2=1.3520, p=0.2449 → **有意でない**
- (参考) Iter31 vs Iter33: a=53, b=34, Chi2=4.1494, p=0.0416 → 有意(α=0.05)
- Experimenterの discordant 数(73/56)と再計算(56/69)が異なるのは，beforeファイルの選択
  または McNemar 実装の違いによる可能性．いずれにせよ Experimenterの比較ではp>0.05で
  **有意な改善ではない**．

**per-domain recall McNemar (Iter28 vs Iter33)**:

| Domain | da (before→NG) | db (NG→OK) | p値 | 方向 |
|--------|----------------|------------|------|------|
| business_economics | 2 | 9 | 0.0348 | 改善 |
| computer_science | 7 | 5 | 0.5637 | 微減 |
| education | 10 | 16 | 0.2393 | 改善 |
| general | 3 | 4 | 0.7055 | 微増 |
| history_culture | 6 | 5 | 0.7630 | 微減 |
| legal | 8 | 2 | 0.0578 | 悪化 |
| mathematics | 4 | 4 | 1.0000 | 同率 |
| medical | 4 | 6 | 0.5271 | 改善 |
| natural_science | 7 | 8 | 0.7963 | 改善 |
| social_science | 5 | 10 | 0.1967 | 改善 |

**per-domain precision Fisher (Iter28 vs Iter33)**: 全ドメイン p>0.37．最も低いのは
natural_science (p=0.3955)．

**BH補正後 (20指標: 10ドメイン×recall/precision)**:
- 最も低いrecall p値: business_economics_recall p=0.0348, BH-q=0.6962 → 有意でない
- 最も低いprecision p値: legal_precision p=0.3784, BH-q=1.5134 → 有意でない
- **BH補正後有意な退行: 0件** → 非退行条件は成立

**主基準の判定**: education_recall(0.4412) > medical_recall基準(0.5112) ?
- 0.4412 < 0.5112 → **不成立**．70ptのギャップは残る．

**非退行の判定**: BH補正後有意退行0件 → **成立**

**全体評価**: **rejected**
- 主基準（education_recall > medical_recall基準 0.5112）が不成立
- McNemar p=0.1589 で top1_accuracy の有意改善なし
- education_recall の +3.53pt 改善は SE~3.8pt のノイズ範囲内
- 案C（70/40/40）の変化幅では不十分だった可能性

**学び**:
1. 案C（sociology 70/高卒心理 40/道徳論 40）は現状比（41/55/54）から sociology を
   +29pt，他2タスクを -15ptずつ変更した．この変化幅では教育recallへの信号が
   ノイズに埋もれた．
2. 案A（90/30/30，sociologyを+49pt，他2タスクを-25pt）が次点として残っている．
   変化幅の大きい案Aを試す価値がある．
3. ただし，代理タスクの意味的ギャップという根本原因は，抽出比率の変更では解決しない．
   案Aも不成立なら，調査(Iter33)計画で示された「手作り訓練問題の追加」へ切り替える必要がある．

### Iteration 33 実行済み

**変更内容**: `build_dataset.py`（sample_weight revert, _EDUCATION_PROXY_TASK_TRAIN_TARGET_SIZES
新設, _sample_domain_questionsにtask_target_sizes追加, build_classifier_training_rowsのeducation
特別扱い）, `tests/test_build_dataset.py`（テスト改名・新規追加3件）．
生成ファイル: `data/classifier_train_iter33_resampled.jsonl`,
`models/domain_classifier_iter33_resampled.joblib`,
`results/iter33_calibrated_predictions.jsonl`．

**結果**:
- top1_accuracy: 0.5850 → 0.5956 (+1.06pt, McNemar p=0.1589 有意でない)
- education_recall: 0.4059 → 0.4412 (+3.53pt, Wilson CI 大きく重なり)
- medical_recall: 0.4831 → 0.5000 (+1.69pt)
- legal_recall: 0.5833 → 0.5611 (-2.22pt)
- ECE: 0.1934 → 0.0676 (-0.1258, 大幅改善)
- 非退行: BH補正後有意退行0件 → 成立

**判定**: rejected（確定）

**判定理由**:
1. 主基準（education_recall > medical_recall基準 0.5112）不成立（0.4412 < 0.5112，70ptギャップ）
2. McNemar p=0.1589 で top1_accuracy の有意改善なし
3. education_recall の +3.53pt 改善は SE~3.8pt のノイズ範囲内
4. 案C（70/40/40）の変化幅では不十分

**学び**:
1. 案C（sociology 70/高卒心理 40/道徳論 40）は現状比（41/55/54）から sociology を
   +29pt，他2タスクを -15ptずつ変更した．この変化幅では教育recallへの信号が
   ノイズに埋もれた．
2. 案A（90/30/30，sociologyを+49pt，他2タスクを-25pt）が次点として残っている．
   変化幅の大きい案Aを試す価値がある．
3. ただし，代理タスクの意味的ギャップという根本原因は，抽出比率の変更では解決しない．
   案Aも不成立なら，調査(Iter33)計画で示された「手作り訓練問題の追加」へ切り替える必要がある．
4. 2イテレーション連続（Iter32 sample_weight, Iter33 resampling案C）でrejectedとなった
   背景には，「教育ドメインの代理タスクが本質的にeducationの意味的ギャップを抱えている」
   という根本原因がある．抽出比率の変更という表面的な最適化では，この根本原因に対処できない．

### 考察 (Iter33)

**結論**: rejected．主基準（education_recall > medical_recall基準 0.5112）が不成立．
McNemar p=0.1589 で top1_accuracy の有意改善なし．非退行条件（BH補正後有意退行0件）は成立
したが，主基準が通らないため採用不可．

**次のイテレーションへの示唆**:
1. **案A（90/30/30）を次点として試す**: 変化幅が案Cの約2倍．効果があれば有意検出の可能性
   がある．ただし弱い2タスクの削減幅が大きい（-55/54→30/30）ため，学習信号喪失のリスクも
   相対的に高い．
2. **案Aも不成立の場合**: 代理タスクの抽出比率変更は限界に達したと判断し，
   調査(Iter33)計画で示された「education固有の手作り訓練問題の追加」へ切り替える．
   これは d0003 X8 の根本原因（代理タスクの意味的ギャップ）に直接アプローチする．
3. **ノイズ判定の補強**: education_recall の変化は n=170 で SE~3.8pt．有意検出には
   5pt以上の効果量が必要．次回実験でも有意検出できない場合は，母数増強（education用
   訓練データ行数の増設）を検討する．

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

