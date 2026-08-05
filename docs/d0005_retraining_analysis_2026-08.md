# d0005: classifier retraining への移行分析（2026-08-03）

**この文書の役割**: Iter53 で全 levers 試し切り完了後、post-hoc 手法の天花板（education_recall ~0.60）を突破するための classifier retraining 移行の分析文書。

**関連文書**:
- `docs/d0004_research_status_and_direction_2026-08.md`: 研究の現況（Iter27 時点）
- `.claude/research/journal.md`: イテレーション単位の生記録（一次情報）
- `.claude/research/backlog.md`: 自動判断の記録・要人間判断事項

---

## 1. 現状の把握

### 1.1 post-hoc 天井の定量値

| イテレーション | 設定 | education_recall | 累積改善 |
|---|---|---|---|
| Iter31 | threshold=0.0, intercept=0.0 | 0.4588 | ベースライン |
| Iter44 | intercept_delta=+0.7 | 0.5588 | +0.1000 |
| Iter53 | threshold=+0.05 | 0.6000 | +0.1412 |

**education_recall = 0.6000 が post-hoc 天井**。これ以上は decision boundary の回転（classifier retraining）が必要。

### 1.2 全レバー試し切り状態（最終）

| レバー | 状況 |
|---|---|
| fallback_policy | adopted (完了) |
| classifier_calibration | 全 3 値試済み (temperature adopted) |
| classifier_training_data_composition | 全 6 値 rejected |
| class_weight_adjustment | rejected |
| embedding_adaptation | 全 4 値 rejected |
| classifier_head_adaptation | 2 adopted, 1 exhausted, 1 skip (**CLOSED**) |
| aggregation_method | 全 3 値試済み (max_confidence adopted) |

---

## 2. 根本原因の分析

### 2.1 education_recall が低い理由

- education は JMMLU に直接対応するタスクがない
- 現状の proxy tasks（sociology, high_school_psychology, moral_disputes）は教育実務（学校教育行政・学習指導要領等）と意味的に乖離している
- sociology の recall は 0.625 だが、high_school_psychology は 0.438、moral_disputes は 0.435
- これらの proxy tasks の埋め込み空間は、real education questions と十分に重ならない

### 2.2 なぜ post-hoc 手法では天花板があるのか

- intercept shift と threshold addition は、decision boundary の**位置**を平行移動するだけで、**方向**は変えない
- boundary を越えない教育質問の誤分類は解消できない
- 天花板を突破するには、**decision boundary の回転**（係数ベクトルの変更）が必要
- これは分類器の再訓練を伴う

### 2.3 既知のアプローチ全試行済み

**classifier_training_data_composition**（6 値、全 rejected）:
- Iter32: sample_weight → 0.4412（悪化、sklearn の class_weight 結合バグ）
- Iter33: resampling 案 C（70/40/40）→ 0.4412
- Iter34: resampling 案 A（90/30/30）→ 0.4353
- Iter35: handmade 50 件追加 → 0.4118（悪化、埋め込み空間競合）
- Iter36: japanese_civics 置換 → 0.0529（崩壊、train/eval 不一致）
- Iter37: japanese_civics 再割当 → invalid（label leakage）
- Iter38: hybrid approach → 0.4000（japanese_civics 追加が recall を悪化）

**embedding_adaptation**（4 値、全 rejected）:
- Iter40: SetFit full FT → flip_rate 52.56%
- Iter41: LoRA r=16 → flip_rate 35.88%
- Iter42: LoRA r=8 → flip_rate 35.88%（r=16 と同一、intrinsic dimensionality <=8）
- Iter43: Dense projection head → flip_rate 42.00%

---

## 3. retraining が難しい理由

### 3.1 単一レバー原則との両立が困難

- retraining = training data 変更 = boundary shift
- argmax flip rate <15% を保証できない
- 既存 adopted 設定（iter44, iter52）との互換性を変更することに同意する必要がある

### 3.2 埋め込み空間の制約

- embedding model（nomic-embed-text）は freeze 必須
- embedding space を回転させられない限り限界
- classifier の weight vectors だけでは、embedding space で分離できない質問は誤分類される

### 3.3 label leakage リスク

- japanese_civics（150 件）は eval ターゲットサイズと同一
- 訓練データに含めると label leakage になる（Iter37 で確認）

---

## 4. 検討すべきアプローチ

### 4.1 アプローチ A: 教育固有訓練データ追加（大規模）

- handmade 50→200-300 件増強
- より多様な教育実務トピックをカバー
- **リスク**: Iter35 で 50 件追加で recall が -0.0471 悪化。200 件で同様の競合が起きるか？
- **flip_rate 推定**: 訓練行数 1427→1627-1727（+14-20%）。flip_rate は 15% 超のリスクが高い

### 4.2 アプローチ B: 訓練データ構成の根本変更

- japanese_civics を education training data に追加（eval では除外）
- Iter36 で確認: japanese_civics のみだと train/eval 不一致で崩壊
- 改良案: japanese_civics + 旧 proxy tasks の hybrid（Iter38）→ 0.4000（悪化）
- **結論**: japanese_civics 追加自体が recall を悪化させる要因になっている可能性

### 4.3 アプローチ C: feature engineering（embedding freeze）

- 既存 embedding に education-aware features を追加（例: education class の mean embedding との cosine similarity）
- LogisticRegression の入力次元を増やし、education をより分離可能に
- **リスク**: 入力次元の変更は classifier retraining を伴う。flip_rate は 15-30% のリスク（過去推定）

### 4.4 アプローチ D: 別 embedding model への切り替え

- nomic-embed-text の代わりに、日本語教育ドメインで fine-tuning 済みの embedding model を使用
- **制約**: WAFL-PEFT のインフラ変更を伴う。research_frontier 相当の大規模変更

---

## 5. 推奨

### 5.1 retraining 移行の条件

1. **embedding model は freeze する**（nomic-embed-text 維持）
2. **training data の変更のみ**（build_dataset.py, prepare_lora_training_data.py の変更）
3. **flip_rate の計測と評価**（単一レバー原則の範囲内かどうかを厳密に検証）
4. **human judgment による承認**（retraining = boundary shift = 既存 adopted 設定の変更）

### 5.2 次の一手

> **【この節は Iter54 の結果で覆っている】** ここで提案した `classifier_training_data_composition` の
> 新しい値（`education_soft_label_distillation`）は Iter54 の計画フェーズで検討されたが，
> **single_lever_compatibility が低いと評価され，実験を実行せずに棄却された**．研究は
> `status="converged"` へ移行している．経緯は `docs/d0006_research_summary_iter28-54_2026-08.md` §2.6
> を参照すること．§5.3 の要人間判断 3 項目は，2026-08-05 に backlog B84 で全て「現状維持」として
> 確定した（`docs/d0006` §7 参照）．

- **Iter54+**: `classifier_training_data_composition` の新しい値を計画フェーズで検討
- **重点調査**: より高品質な education training data の設計（handmade 問題の増強、または新しい proxy task の探索）
- **評価基準**: flip_rate <15% を厳密に検証。超える場合は retraining 承認の見送り

### 5.3 要人間判断

1. **classifier retraining の承認**: training data の変更は decision boundary の移動を伴う。既存 adopted 設定（iter44, iter52）との互換性を変更することに同意するか？
2. **flip_rate 許容範囲の定義**: <15% を厳守するか、<20% まで許容するか？
3. **education_recall 基準値の再定義**: medical_recall 0.5112 は education に不公平な基準。再定義するか？

---

## 6. 参考文献

- `docs/d0001_literature_survey_2026-07.md`: 先行研究の文献調査
- `docs/d0002_research_cycle_findings_2026-07.md`: Iter1〜22 の総括
- `docs/d0003_next_experiments_2026-07.md`: 次の実験計画（Iter22 時点）
- `docs/d0004_research_status_and_direction_2026-08.md`: Iter15〜27 の到達点（Iter27 で更新停止）
- `docs/d0006_research_summary_iter28-54_2026-08.md`: **Iter28〜54 の総括（研究収束時点の現況）**
- `.claude/research/journal.md`: イテレーション単位の生記録（一次情報）
- `.claude/research/backlog.md`: 自動判断の記録・要人間判断事項

---

**作成日**: 2026-08-03
**更新日**: 2026-08-03
**著者**: research-cycle（rc-reflector 分析ベース）
