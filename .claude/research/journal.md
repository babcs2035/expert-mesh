## Iteration 31: 分類器較正のtemperature scaling方式によるargmax不変性の実証とECE目標到達可否の検証

### 調査 (Iter31)

**問い**:
1. 本リポジトリの base estimator（`LogisticRegression(max_iter=1000, class_weight='balanced')`）に
   対して `CalibratedClassifierCV(method='temperature')` を使うと，sklearn は
   `decision_function`（ロジット）を経由するのか，`predict_proba` の対数近似を使うのか．
2. `class_weight='balanced'` は temperature の「argmax 不変」という理論保証を壊す余地があるか．
3. Iter30 で確立した非退行チェック手順（BH補正 q=0.05・recallはドメイン別McNemar・precisionは
   Fisher正確検定・20指標）は temperature でもそのまま再利用可能か．
4. `cv`・`ensemble` パラメータは Platt/isotonic（`cv=5, ensemble=True`）と同じ設定を踏襲すべきか．

#### 分かったこと

**(1) `decision_function` 経路が使われることをソースコードで直接確認 — 一次情報**

本リポジトリの実行環境（`.venv/lib/python3.12/site-packages/sklearn/calibration.py`，
`scikit-learn==1.9.0`）を `Read` で直接確認した。

- `_fit_calibrator()`（687-749行）は `method == "temperature"` の分岐で
  `calibrator = _TemperatureScaling(); calibrator.fit(predictions, y, sample_weight)` を呼ぶ。
  この `predictions` は `CalibratedClassifierCV.fit()` 内で
  `_get_response_values(estimator, X, response_method=["decision_function", "predict_proba"])`
  （`decision_function` を優先するリスト順）から得られており，`LogisticRegression` は
  `decision_function` を実装しているため，**実際に使われるのは
  ロジット（`w^T x + b`，多クラスなので `(n_samples, n_classes)` 形状）そのものであり，
  `predict_proba` の対数近似は使われない**。
- `_TemperatureScaling.fit()`（1077-1181行）の docstring に「If the input appears to be
  probabilities (i.e., values between 0 and 1 that sum to 1 across classes), it will be
  converted to logits using `np.log(p + eps)`」と明記されている。`_convert_to_logits()`
  （954-983行）はこの判定を行うヘルパーだが，`LogisticRegression.decision_function()`
  の出力は確率ではなくロジットそのもの（0-1に収まらず合計も1にならない）ため，この
  変換は発火せず，ロジットがそのまま `raw_prediction = exp(log_beta) * logits` として
  multinomial loss（`HalfMultinomialLoss`）の最小化に使われる（`log_beta` を
  `scipy.optimize.minimize_scalar`で`bounds=(-10.0, 10.0)`の範囲で最適化）。
  `beta_ = exp(log_beta*)` が「逆温度」であり `T = 1/beta_`。

**(2) `class_weight='balanced'` はロジット自体の値には影響するが，temperature の
argmax 不変性を壊す経路にはならない**

`class_weight='balanced'` は `LogisticRegression.fit()` 内の損失関数の重み付けにのみ影響し，
学習後の `decision_function()` は単なる固定の線形写像 `w^T x + b` である。temperature
scaling は，この**固定されたロジットベクトル全体**を単一スカラー `1/T` 倍してから softmax を
取るだけの変換であり，`class_weight` がどうロジットの値そのものを決めたかとは独立に，
「全クラスに同一の正の定数を掛けて softmax を取る操作は argmax を変えない」という数学的事実
（`softmax` は単調変換に対して順序不変）がそのまま成立する。実測でも確認した（下記(3)）。

**(3) 実測検証（`uv run python` での合成データ実験，1427件・legal 77件/他150件という
本リポジトリの訓練データ規模を模した設定）— 新たな重要な留保を発見**

`sklearn.linear_model.LogisticRegression` と `sklearn.calibration.CalibratedClassifierCV` を
実際に合成データ（10ドメイン，legal 77件・他9ドメイン各150件，32次元の埋め込みを模した
乱数特徴量）で fit させ，argmax の一致率を直接計測した。

- **fold内（同一 `(estimator, T)` ペア）での argmax 保持は理論通り厳密に 100%**。
  `ensemble=True` の各 fold について，その fold の base estimator 自身の
  `decision_function` の argmax と，その fold の temperature 較正後 `predict_proba`
  の argmax を比較したところ，5 fold 全てで一致率 1.0（0/1427 不一致）だった。
  sklearn 公式の「T は softmax の argmax の位置に影響しない」という保証は，この
  「単一の (estimator, T) ペア内」という意味で寸分違わず成立している。
- **しかし `ensemble=True` では，本番推論時に使われるのは 5 つの異なる fold
  （80% サブセットで学習した 5 つの異なる `LogisticRegression`，かつ 5 つの異なる T）の
  予測確率の平均であり，全データで学習した単一モデルとの比較では非ゼロの flip が生じる**。
  合成データでの実測: `method='temperature', ensemble=True` で
  全データ学習の単一 base estimator との argmax 不一致 16/1427（1.12%）。
  同条件で `method='sigmoid'` は 55/1427（3.85%），`method='isotonic'` は 60/1427（4.20%）。
  **この非ゼロ flip の原因は temperature の較正曲線の歪みではなく，
  `ensemble=True` 自体が持つバギング的な平均化効果（5 つの異なるサブセットで学習した
  分類器の予測を平均する）であり，sigmoid/isotonic にも共通する構造である**。ただし
  temperature は per-class の曲線歪みという追加の誤差源を持たないため，同じ
  `ensemble=True` 条件でも isotonic/sigmoid よりこの合成データで一貫して小さい。
- **`ensemble=False` にすると，temperature は理論通り厳密に 0% の flip（0/1427）を
  達成した**。この設定では本番推論に使われる base estimator は全データで学習した単一
  モデルであり（サブセット学習の平均化がない），T も CV による out-of-fold 予測から
  1 つだけ学習されて，その単一モデルの固定ロジットに適用される。一方，同じ
  `ensemble=False` でも `sigmoid`（2.87%不一致）・`isotonic`（5.19%不一致）は依然として
  非ゼロだった（OvR 事後正規化由来の歪みは `ensemble` の設定と無関係に残る）。
- sklearn 公式のリリースノート（1.8, `tavily-extract`で直接取得）が示す temperature
  scaling の使用例は `CalibratedClassifierCV(clf, method="temperature", ensemble=False)`
  であり，`ensemble=False` を使うサンプルコードになっている。ただし本リポジトリの
  Iter29（Platt）・Iter30（isotonic）はいずれも `cv=5, ensemble=True` で実施済みであり，
  config.yml のレバー note は「temperatureも同条件を踏襲する」ことを既定の想定として書いている。

**(4) Iter30 の非退行チェック手順（BH補正・McNemar・Fisher）はそのまま再利用可能。
isotonic特有チェックリストは temperature には構造的に該当しない**

- BH補正付き 20 指標非退行チェック（recall=ドメイン別McNemar，precision=Fisher，
  計 20 個の p 値へ BH q=0.05）は手法に依存しない一般的な統計手続きであり，
  temperature でも変更なくそのまま使える。
- Iter30 のisotonic特有チェックリスト（(a) 確率の厳密な0/1張り付き，(b) 全クラス0.1の
  uniform fallback，(c) tie率）は，temperature の実装構造上そもそも発生しない。
  `_CalibratedClassifier.predict_proba()`（833-843行）の `method == "temperature"` 分岐は
  `proba = self.calibrators[0].predict(predictions)` で softmax の出力をそのまま使うため，
  isotonic/sigmoid のような「クラスごとの計算結果を後から正規化し，分母が0ならuniform
  fallbackする」という経路（810-832行）を一切通らない。softmax は有限のロジット入力に対し
  厳密に0や1にはならず（`exp` の値は常に正），tie も理論上は浮動小数点の偶然の一致でしか
  起こらない。したがって temperature ではこの3チェックは「必ず該当なし」になる見込みが高く，
  報告項目として残す価値は低い（形だけ算出して0件であることを確認する程度で十分）。
- Iter30 で判明した `medical_recall` 系統的圧縮（isotonic較正曲線がmedicalクラス固有に
  確率の天井を下げていた）は，temperature が単一スカラーで全ドメイン共通の変換しかしない
  構造上，**再現しないと理論的に予想される**が，逆に言えば temperature は medical だけを
  選択的に補正することもできない。ある特定ドメインの較正だけがずれている場合，temperature
  はそのドメインを狙って直すことはできず，全体の log loss を最小化する単一の T に丸め込む
  （config.yml note の留保どおり）。

**出典**:
- ローカル実行環境の直接確認: `.venv/lib/python3.12/site-packages/sklearn/calibration.py`
  （`_fit_calibrator`687-749行，`_CalibratedClassifier.predict_proba`781-847行，
  `_TemperatureScaling.fit/predict`1068-1230行，`_convert_to_logits`954-983行）を
  `Read`・`grep` で直接確認（2026-07-31実施，一次ソース）。
- `uv run python` での合成データ実測（10ドメイン，legal 77件/他150件を模した規模，
  32次元乱数特徴量，`method in {sigmoid, isotonic, temperature}` × `ensemble in {True, False}`
  の argmax 一致率比較）。本セッションで実施，再現可能。
- https://scikit-learn.org/stable/auto_examples/release_highlights/plot_release_highlights_1_8_0.html
  （`tavily-extract`で直接取得。「Temperature scaling in CalibratedClassifierCV」節，
  `ensemble=False`を使うサンプルコード，「particularly well suited for multiclass problems
  because it provides (better) calibrated probabilities with a single free parameter」の原文）
- https://scikit-learn.org/stable/whats_new/v1.8.html （`tavily-search`。
  「Added temperature scaling method in calibration.CalibratedClassifierCV」の変更履歴，
  Array API対応の追加も1.8で行われたことの確認）
- https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html
  （`tavily-search`。1.9.0時点のAPIリファレンス，`.. versionchanged:: 1.8 Added option
  'temperature'`の再確認）
- journal.md「調査 (Iter30)」節（`method='temperature'`の存在確認・sklearn公式ドキュメントの
  「Tはsoftmaxのargmaxの位置に影響しない」引用は既出のため本イテレーションでは再掲のみ）

#### rc-planner への申し送り

1. **問い1・2は確定的に解消**: `decision_function`（ロジット）経路が使われることをソースコード
   直接確認・実測の両方で確定した。`class_weight='balanced'`はロジットの値を決めるだけで，
   temperatureのargmax不変性の理論保証を壊す経路は存在しない。
2. **問い4（cv/ensemble）に，計画フェーズで判断すべき新しい論点が生じた**。
   `ensemble=True`（Iter29/30と同一設定）を維持すると，temperatureでも
   ensemble平均化に起因する非ゼロのflip（合成データで1.12%，isotonic4.20%・sigmoid3.85%より
   小さいが0ではない）が生じることが判明した。これは「T はsoftmaxのargmaxを変えない」という
   sklearn公式の理論保証が，**個々の(estimator, T)ペア内では厳密に成立するが，
   `ensemble=True`によって生成される最終的な推論モデル（5ペアの平均）全体には及ばない**
   ことを意味する。したがって「temperatureはtop1_accuracy不変が理論的に保証される」という
   config.yml note・backlog B51の記述は，`ensemble=True`を維持する場合は**厳密には
   正確ではなく**，「isotonic/sigmoidより小さいが，ゼロではないflipが生じうる」と
   修正して計画に反映すべきである。
   - 選択肢 A（推奨）: `cv=5, ensemble=True`をIter29/30と完全に同一のまま維持する。
     単一レバー原則を「較正手法のみ」に厳密に限定でき，Iter29/30との直接比較可能性が
     最大になる。ただし成功条件（top1_accuracy非退行・per-domain非退行）の判定基準文言に
     「temperatureの理論的argmax不変性は個々の fold ペア内の話であり，ensemble平均化に
     起因する小さな非ゼロflipは想定内である」旨を明記し，isotonicのような「クラス固有の
     曲線歪み」由来の系統的退行（medical_recallのような）とは区別して解釈する必要がある。
   - 選択肢 B: `ensemble=False`に変更する（sklearn公式のリリースノートのサンプルコードが
     使う設定）。この場合argmax不変性が文字通り厳密に成立する（合成データで実測0%）。
     ただしIter29/30とは`ensemble`パラメータ自体が異なるため，較正手法とensembleの
     2変数が同時に変わることになり，単一レバー原則の厳密な適用としては説明が要る
     （「temperatureは1パラメータのみの学習で過学習リスクが本質的に低いため，
     isotonic/plattで必要だったensemble平均化によるロバスト化が不要」という理屈は立つが，
     計画書で明示的に正当化すること）。
   いずれを選ぶにせよ，計画(Iter31)節に「今回のcv/ensemble設定と，Iter29/30との比較可能性
   への影響」を明記すること。
3. **非退行チェック手順はIter30のBH補正付き20指標チェック（recall=McNemar，precision=Fisher，
   BH q=0.05）をそのまま流用してよい**。isotonic特有の3項目チェックリスト（0/1張り付き・
   uniform fallback・tie率）はtemperatureでは構造的に該当なしと予想されるため，
   簡略化して「該当0件であることの確認」程度に留めてよい（実験時間の節約になる）。
4. **medical_recall問題がtemperatureで再現するかどうかは，Y4全体の結論に関わる重要な
   観察点である**。isotonicのmedical系統的圧縮が「OvR方式のクラス固有曲線歪み」に起因する
   という Iter30 の結論が正しければ，単一Tしか使わないtemperatureではこの種の系統的圧縮は
   原理上起こらないはずである。もしtemperatureでも同様の非退行違反が起きた場合，
   「OvR方式由来ではない別の根本原因（例えばmedicalクラス自体の埋め込み分離の弱さ）」を
   疑う材料になる。
5. **ECE改善幅がplatt（0.16751）・isotonic（0.121424）に届かない可能性は，config.yml note
   どおり留保として残る**。単一Tでは表現力がisotonic/plattより低いため，目標0.150に届かない
   （platt同様partial）シナリオも十分あり得る。この場合は「per-domain非退行のためには
   OvR方式の柔軟性を犠牲にできない」という新知見が得られ，次の一手（isotonicの運用調整，
   例えばmedicalドメインのみ較正を無効化する等）を検討する材料になる（backlog B51要レビュー
   (1)がすでに示唆済み）。

---

### 計画 (Iter31)

**仮説**: `scripts/train_domain_classifier.py:train_classifier()` の較正手法を `method="isotonic"`
（Iter30，partial：ECE目標達成もmedical_recallがBH補正後有意悪化）から `method="temperature"`
へ切り替えると，単一スカラーTでロジット全体を変換する構造上，isotonic/plattのOvR方式由来の
クラス固有曲線歪み（medical_recall悪化の疑わしい原因，Iter30考察）が構造的に排除され，
per-domain非退行が成立する。一方，temperatureは表現力がisotonic/plattより低いため，ECE改善幅が
isotonic（0.121424）はもとよりplatt（0.16751，目標未達実績）にも届かず，目標0.150未達となる
可能性が留保として残る（config.yml note・backlog B51要レビュー(1)）。

**単一レバー**: `classifier_calibration`（`.claude/research/config.yml` のレバー，150行目）。
今回試す値は `values: [platt, isotonic, temperature]` のうち **`temperature` のみ**
（backlog B51の自動選択）。これで config.yml 登録済みの3値をすべて試したことになる。

**cv/ensemble設定の決定（調査(Iter31)申し送り2の選択肢A/Bのいずれかを選ぶ）**:

**選択肢A（`cv=5, ensemble=True`，Iter29/30と完全同一）を採用する**。理由:

1. 単一レバー原則を「較正手法のみ」に厳密に限定できる。`cv`・`ensemble`を較正手法と同時に
   変えると2変数が同時に動き，ECE・flip rate・per-domain結果の変化が較正手法の違い由来か
   ensemble設定の違い由来か切り分けられなくなる。今回のY4の核心的な問いは「isotonic/plattとの
   直接比較の下でtemperatureがmedical_recall問題を回避しつつECE目標に届くか」であり，
   Iter29・Iter30との比較可能性の維持そのものがこのイテレーションの価値の大部分を占める。
2. 調査(Iter31)の実測で明らかになった`ensemble=True`由来の非ゼロflip（合成データで1.12%）は，
   5 fold間のバギング的平均化という手法非依存の一般的機序に起因し，isotonic/plattが抱える
   「OvR方式のクラス固有曲線歪み」（medical_recall悪化の疑わしい原因）とは異なる機序である。
   したがってこの程度のflipは，medical_recallのような系統的・ドメイン固有の退行の温床には
   ならないと考えられ，「temperatureはOvR由来のクラス固有歪みを構造的に持たない」という
   本イテレーションが検証したい理論的主張の意義を損なわない。
3. sklearn公式リリースノートの`ensemble=False`サンプルコードは「temperatureの使い方の一例」に
   過ぎず，本リポジトリがIter29・Iter30で確立した比較条件を犠牲にしてまで踏襲すべき規範とは
   判断しない。
4. 選択肢B（`ensemble=False`）を取らない理由: 較正手法とensembleの2変数が同時に変わり，
   「temperatureが優れているのか，ensemble平均化を止めたことが効いたのか」を切り分けられなく
   なる。仮にtemperatureがper-domain非退行を達成しても，isotonic/plattでも`ensemble=False`に
   すれば同様に改善した可能性を排除できず，Y4全体の結論（較正手法としてのtemperatureの優位性）
   が弱まる。

**成功条件の解釈への反映（調査(Iter31)申し送り2が要求した文言）**: temperatureの理論的argmax
不変性（sklearn公式が保証する「Tはsoftmaxのargmaxの位置に影響しない」）は，個々の
`(estimator, T)` ペア内で厳密に成立する事実であり，`ensemble=True`による5fold平均化に起因する
小さな非ゼロflip（合成データ実測1.12%，isotonic4.20%・sigmoid3.85%より小さいが0ではない）は，
この理論保証が主張する範囲の外側にある，想定内の挙動として扱う。したがって今回の実測で
flip_rateが完全に0%でないこと自体は失敗ではない。成功条件2・3（下記）の判定はあくまで
統計的検定（McNemar／Fisher／BH補正）の結果で行い，「flipが0でないから理論違反」という
短絡的な解釈はしない。

**固定する構成（Iter29/30と完全に同一，`config.yaml`は一切変更しない）**:
`routing_method=supervised_classifier`，`confidence_threshold=0.0`・`dispatch_top_k=1`・
`aggregation_method=max_confidence`（Iter28 adopted構成），`confidence_signal_method=self_report`，
`confidence_elicitation=top_k_with_probs`，`expert_model=expert-mesh-{domain}-lora`
（domain_count=10），`light_model=qwen3.5:4b-q4_K_M`，`embedding_model=nomic-embed-text`，
評価データセットはIter25以降固定の1600問（`data/dataset.jsonl`）。訓練データも
`data/classifier_train.jsonl`（1427件，legal 77件・他9ドメイン各150件）でIter29/30と同一。
`CalibratedClassifierCV`の`cv=5`・`ensemble=True`もIter29/30と同一（上記選択肢A）。
**今回変更するのは`train_classifier()`内の較正手法（`_CALIBRATION_METHOD`定数の値）のみであり，
`config.yaml`のキーは1つも変えない。**

**変更ファイル・行**:

1. `scripts/train_domain_classifier.py`
   - `_CALIBRATION_METHOD = "isotonic"`（64行）→ `"temperature"`に変更。`_CALIBRATION_CV = 5`
     （65行）は無変更。`ensemble=True`（`train_classifier()`117-120行の
     `CalibratedClassifierCV(...)`呼び出しにハードコード）は無変更のまま維持する（上記
     選択肢Aの実装上の反映箇所，これ自体は変更しない）。
   - `_CALIBRATION_METHOD`直上のコメント（45-63行付近，isotonicを選んだ理由の説明）を，
     temperatureを選ぶ理由（Iter30のisotonicがmedical_recallのBH補正後有意悪化でpartial判定・
     backlog B51の自動選択）と，調査(Iter31)が確認した構造上の利点（単一スカラーTでロジット
     全体を変換するためクラスごとの個別較正器を持たず，OvR方式由来のクラス固有曲線歪みを
     構造的に排除する）に更新する。cv/ensembleを維持する理由（比較可能性優先，上記選択肢Aの
     要約）も一言追記する。
   - モジュール冒頭docstring（1-5行）の`Iter30, classifier_calibration=isotonic`を
     `Iter31, classifier_calibration=temperature`に更新する。
   - `train_classifier()`のdocstring（95-116行付近）を`method="temperature"`の説明に更新する。
     isotonic特有の注意点（tie・0/1張り付き・uniform fallback）への言及は，調査(Iter31)
     分かったこと(4)よりtemperatureの実装構造上「該当しない」旨に置き換える。
   - 出力アーティファクト名は`models/domain_classifier_temperature.joblib`（isotonic版
     `models/domain_classifier_isotonic.joblib`・platt版
     `models/domain_classifier_platt.joblib`とは別名で新規生成，本番
     `models/domain_classifier.joblib`は上書きしない）。
   - `tests/test_train_domain_classifier.py`はisotonic特有のアサーションを含んでおらず
     （確認済み，`grep`でisotonic/`_CALIBRATION_METHOD`への直接参照なし），`method`の変更が
     `StratifiedKFold`の分割条件に影響しないため無変更で通る見込み（実装フェーズで実行して
     確認する）。

2. `scripts/evaluate_classifier_calibration.py`: **変更不要**。`probabilities`フィールド
   （10ドメイン全ての確率）はIter30で既に追加済みであり，temperature特有チェックリスト
   （下記手順7）にもそのまま使える。

3. `metrics.py`: **変更不要**。`compute_domain_recall_mcnemar_test`・
   `compute_domain_precision_fisher_test`・`apply_benjamini_hochberg`はIter30で実装済みで，
   手法非依存の統計手続きのためそのまま再利用する。

4. `tests/test_metrics.py`: 変更不要（Iter30で追加したテストは手法非依存のため，そのまま
   有効）。

**評価手順（Iter30の手順1-8をそのまま踏襲し，モデル名・出力ファイル名のみ変更）**:

1. 新分類器の学習: `uv run python -m scripts.train_domain_classifier --train-data
   data/classifier_train.jsonl --embedding-model nomic-embed-text --ollama-host <live node>
   --output models/domain_classifier_temperature.joblib`（Iter29/30と同じくライブなollama
   ノード1台へのembeddingのみ）。
2. 「較正前」データはIter29/30と同一の`results/20260731_162722/results.jsonl`（Iter28実測，
   fallback 0/1600）をそのまま使う。**再実行しない**（3イテレーションを同じ較正前基準で
   揃えて比較可能にするため）。
3. 「較正後」データは`scripts/evaluate_classifier_calibration.py`で1600問を再embeddingし，
   `--classifier models/domain_classifier_temperature.joblib --output
   results/iter31_calibrated_predictions.jsonl`として生成する。
4. `metrics.py:compute_ece(n_bins=10)`を較正前・較正後の両方に同一のbin設定で適用し，ECEを
   比較する（較正前基準0.19336はIter29/30から流用，再計算しない）。
5. top1_accuracyを較正前・較正後で算出し，新旧の正誤ペアで`compute_mcnemar_test`（全体，
   α=0.05）を行う（Iter29/30の手順5と同一）。
6. **per-domain非退行チェック（Iter30で確立した3段構成をそのまま踏襲）**: 全10ドメインに
   ついて，(a) recallは`compute_domain_recall_mcnemar_test`（計10検定），(b) precisionは
   `compute_domain_precision_fisher_test`（計10検定）を実施し，計20個のp値を集めて
   `apply_benjamini_hochberg(p_values, q=0.05)`を一括適用する。adjusted有意かつ方向が悪化
   （較正後の点推定<較正前の点推定）である指標のみを「統計的に有意な退行」と判定する。
7. **temperature特有の実装確認（調査(Iter31)申し送り3により簡略化）**: 較正後の1600行に
   ついて，(a)確率のいずれかが厳密に`0.0`または`1.0`になっている行数，(b)10クラス全てが
   `0.1`に近いuniform fallback行数，(c)tie率の3点を，Iter30と同じ定義で算出するが，
   構造上いずれも「該当0件」になると予想されるため，legalドメイン個別集計などの詳細内訳は
   省略し，3点とも「該当0件であることの確認」に留める簡易報告とする（実験時間の節約）。
   もし予想に反して非ゼロの値が出た場合は，簡易報告に留めず詳細を追加報告すること。
8. 新旧classifierのargmax不一致件数（flip rate）をIter29/30と同じ定義で報告し，
   Iter29（platt，ensemble=True，11.0%）・Iter30（isotonic，ensemble=True，14.3125%）と比較する
   （必須報告項目，判定基準ではない）。

**成功条件（d0003 X9．AND条件．cv/ensemble選択に応じた解釈の但し書きを追加）**:

1. ECE（手順2・4，較正前基準0.19336に対する較正後の値，`n_bins=10`）が**0.150以下**であること。
2. top1_accuracy（手順5）が旧分類器（Iter28実測0.585）に対しMcNemar検定で有意に悪化していない
   （p>=0.05，または新側が改善方向）こと。
3. **per-domain非退行（手順6，Iter30と同一の3段構成）**: 20指標（10ドメイン×precision/recall）
   のp値へBH補正（q=0.05）を適用した結果，adjusted有意かつ悪化方向の指標が**0件**であること。
4. **【但し書き，調査(Iter31)申し送り2】** 条件2・3の判定において，`ensemble=True`に起因する
   合成データ実測1.12%程度の非ゼロargmax flipは，それ自体を理由に条件2・3を不成立とはしない。
   これは個々の`(estimator, T)`ペア内で厳密に成立する理論的argmax不変性が主張する範囲の外側
   （5fold平均化という別の機序）であり，判定はあくまで統計的検定（McNemar／Fisher／BH補正）の
   結果に基づく。もし条件2・3が実際に不成立になった場合，分析(解釈)フェーズでその原因が
   「ensemble平均化由来の偶発的再配分」なのか「temperature特有の別の機序（単一Tへの丸め込み
   によるドメイン固有のトレードオフ）」なのかを切り分けて報告すること。
5. 手順7のtemperature特有チェックリスト（簡易報告）とflip rate（手順8）は，成功・失敗の
   判定基準ではなく必須報告項目として記録する。

**目標未達時の次点候補（次イテレーション向けメモ，今回の計画には含めない）**: config.ymlの
`classifier_calibration`レバーは`platt`・`isotonic`・`temperature`の登録済み3値を今回で
使い切ることになる。仮に3手法いずれもd0003 X9のAND条件を満たせない場合，次の一手は
較正手法そのものの追加候補ではなく，運用的な対処（例：medicalドメインに限定して較正を
無効化する，ドメイン別に異なる較正手法を組み合わせる等）になる可能性が高い（backlog B51
要レビュー(1)がすでに示唆済み）。この判断は本計画の範囲外とし，次イテレーションのrc-reflector
に委ねる。

**人間判断が必要な論点**: 新規追加なし。Y2（`confidence_threshold`の二重責務分離，スキーマ
変更）着手前のユーザー確認はbacklog B49・B50・B51の既存の申し送りのまま。較正済み分類器の
本番反映可否も，temperatureが成功条件（本計画の1-3すべて）を満たした場合に改めてその時点で
判断する（今回のイテレーションで本番アーティファクトを置き換える判断は行わない）。

---

### 実装 (Iter31)

計画どおり単一レバー（`classifier_calibration=temperature`）のみを実装した．`config.yaml` は
変更していない（`git diff --stat -- config.yaml` が空であることを確認済み）。

**変更ファイル**:

1. `scripts/train_domain_classifier.py`
   - `_CALIBRATION_METHOD = "isotonic"` → `"temperature"` に変更。`_CALIBRATION_CV = 5` は無変更。
     `ensemble=True`（`train_classifier()`内`CalibratedClassifierCV(...)`呼び出しにハードコード）
     も計画どおり無変更で維持した（選択肢A，比較可能性優先）。
   - `_CALIBRATION_METHOD`直上のコメントを，temperatureを選ぶ理由（Iter30のisotonicが
     medical_recallのBH補正後有意悪化でpartial判定・backlog B51の自動選択）と，
     調査(Iter31)が確認した構造上の利点（単一スカラーTでロジット全体を変換するため
     クラスごとの個別較正器を持たず，isotonic/PlattのOvR方式由来のクラス固有曲線歪みを
     構造的に排除する）に更新し，cv/ensembleを維持する理由（較正手法のみを単一レバーとして
     切り分けるための比較可能性優先，Iter29/30と同一条件）も追記した。
   - モジュール冒頭docstringの`Iter30, classifier_calibration=isotonic`を
     `Iter31, classifier_calibration=temperature`に更新した。
   - `train_classifier()`のdocstringを`method="temperature"`の説明に更新し，isotonic特有の
     注意点（tie・0/1張り付き・uniform fallback）への言及を，「temperatureの実装構造上
     該当しない」旨に置き換えた。呼び出し側は引き続き手順7でチェックするが，リスクとしてでは
     なく「0件であることの確認」として扱う旨を明記した。
   - 出力アーティファクト名（`--output`の既定値）は変更していない
     （`models/domain_classifier.joblib`のまま）。Iter29/30と同じパターンで，実験フェーズでの
     実行時に`--output models/domain_classifier_temperature.joblib`をCLI引数で明示指定する
     ことで本番アーティファクトを上書きしない運用とする（スクリプト側の既定値変更は不要）。
2. `scripts/evaluate_classifier_calibration.py`: 計画どおり変更不要と確認した。
   `predict_calibrated_rows()`が返す`probabilities`フィールド（Iter30で追加済み，10ドメイン
   全ての確率）はtemperatureの実装確認（手順7）にもそのまま使えることをコード読解で確認した。
3. `metrics.py`: 計画どおり変更不要と確認した。`compute_domain_recall_mcnemar_test`
   （282行）・`compute_domain_precision_fisher_test`（318行）・`apply_benjamini_hochberg`
   （367行）がIter30で実装済みであることを`grep`で確認した。手法非依存の統計手続きのため
   そのまま再利用する。
4. `tests/test_train_domain_classifier.py`: isotonicや`_CALIBRATION_METHOD`への直接参照が
   ないことを`grep`で確認した上で無変更のまま実行し，pass することを確認した。

**テスト結果**: `uv run pytest -q` → 218 passed, 2 skipped（Iter30時点と同数，既存のスキップ
2件は本変更と無関係）。

**lint**: `uv run ruff check .` → 2件のエラー（`scripts/prepare_lora_training_data.py`のF541・
未使用import）が残るが，これはIter29から既知の本変更と無関係な既存差分であることを
`uv run ruff check scripts/train_domain_classifier.py`単体で"All checks passed!"となることで
確認した（単一レバー原則に従い今回も触っていない）。

**config.yaml の確認**: `git diff --stat -- config.yaml`が空であることを確認し，一切変更して
いないことを確認した。

**実験を開始してよい状態か**: はい。コード変更は完了し，テスト・lintとも整合。フェーズ4では，
(1) `scripts/train_domain_classifier.py`で`models/domain_classifier_temperature.joblib`を
1台のライブollamaノードへのembedding呼び出しで新規生成（本番`models/domain_classifier.joblib`
は上書きしない），(2) `scripts/evaluate_classifier_calibration.py`で1600問を再embeddingして
較正後の予測JSONL（`probabilities`フィールド付き）を`results/iter31_calibrated_predictions.jsonl`
として生成，(3) `metrics.py`の既存関数群で較正前（`results/20260731_162722/results.jsonl`，
再実行不要）と較正後を比較し，成功条件1-3（ECE≤0.150・McNemar非退行・per-domain 20指標への
BH補正非退行）と必須報告項目（temperature特有チェックリスト・flip rate）を実測すればよい。

---

### 実験・分析(実行) (Iter31)

計画どおり実機1600問本走は行わず，既存のSSHローカルポートフォワード（`127.0.0.1:11435 ->
wafl500:11434`，`ssh -fNT -L 11435:localhost:11434 wafl500`，Iter29/30から起動済みのプロセスを
そのまま流用．事前に`curl http://127.0.0.1:11435/api/tags`で疎通確認済み）経由のembedding呼び出し
のみで較正前後の比較データを揃えた．LLM生成・probe・dispatchは一切発生していない．

**手順1: 新分類器の学習**

```
uv run python -m scripts.train_domain_classifier \
  --train-data data/classifier_train.jsonl \
  --embedding-model nomic-embed-text \
  --ollama-host 127.0.0.1 --ollama-port 11435 \
  --output models/domain_classifier_temperature.joblib
```

標準出力: `[train_domain_classifier] wrote models/domain_classifier_temperature.joblib
(n_samples=1427, classes=[...10ドメイン...])`．実行時間124.55秒（`time`実測，Iter29のPlatt
124.09秒・Iter30のisotonic126.51秒とほぼ同水準）．`models/domain_classifier_temperature.joblib`
を新規生成し，本番`models/domain_classifier.joblib`のタイムスタンプ（Jul 27 16:08）が今回の実行後
も変化していないこと（＝上書きされていないこと）をファイルシステム上で確認した．

**手順3: 較正後データ生成**

```
uv run python -m scripts.evaluate_classifier_calibration \
  --dataset data/dataset.jsonl \
  --classifier models/domain_classifier_temperature.joblib \
  --embedding-model nomic-embed-text \
  --ollama-host 127.0.0.1 --ollama-port 11435 \
  --output results/iter31_calibrated_predictions.jsonl
```

標準出力: `[evaluate_classifier_calibration] wrote 1600 rows
(classifier=models/domain_classifier_temperature.joblib)`．実行時間141.56秒．出力JSONLは
計画どおり`probabilities`フィールド（10ドメイン全ての確率）付きで1600行生成された．

**手順2**: 較正前データは計画どおり`results/20260731_162722/results.jsonl`（Iter28実測，fallback
0/1600）を再実行せずそのまま使用．新旧2ファイルの`id`集合が完全一致することを確認済み
（`{r["id"] for r in before} == {r["id"] for r in after}`が`True`）．

**異常の有無**: なし．両スクリプトとも例外・タイムアウト・リトライなく正常終了した．実機呼び出し
はwafl500（192.168.15.100:11434）へのembeddingのみ（`nomic-embed-text`，計3027回：1427+1600），
LLM生成・probe・dispatchは一切発生していない．総所要時間は266.11秒（約4.4分，`timeout_min:150`
に対し十分余裕あり）．

`metrics.py`の既存関数（`compute_ece`／`compute_top1_accuracy`／`compute_mcnemar_test`／
`compute_precision_recall_per_domain`）と既存3関数（`compute_domain_recall_mcnemar_test`／
`compute_domain_precision_fisher_test`／`apply_benjamini_hochberg`，いずれもIter30で実装済みで
変更なし）を呼ぶ一時スクリプト（`/tmp/iter31_analysis.py`，非永続）で較正前
（`results/20260731_162722/results.jsonl`）と較正後（`results/iter31_calibrated_predictions.jsonl`，
各1600行）を比較した（判定はここでは行わず，数値のみを機械的に集計する）．

**手順4: ECE（`n_bins=10`で統一）**

- 較正前: **0.193357556998477**（`state.json`の`e29_results.ece_before`／`e30_results.ece_before`と
  同一値，再計算不要のところ実測でも一致することを確認）
- 較正後: **0.07120101725284995**
- 改善幅: 0.122157（較正前→較正後で減少，改善方向）
- 0.150との比較: 較正後0.0712 < 0.150（platt 0.16751・isotonic 0.12142より大幅に低い．目標
  0.150に対し余裕7.88pt，isotonicの2.86ptを大きく上回る）

**手順5: top1_accuracy（1600問，`expected_domains`との一致率）**

- 較正前: 0.585000（Iter28実測と同一値）
- 較正後: 0.605625
- 差分: +0.020625（較正後が高い，Iter29 platt +0.010625・Iter30 isotonic +0.008750より改善幅が大きい）

**手順5: McNemar検定（全体，対応のある2条件比較，較正前=A・較正後=B，連続性補正あり）**

- discordant_a_only（較正前のみ正解）: 30
- discordant_b_only（較正後のみ正解）: 63
- discordant_pairs（合計）: 93
- chi2_statistic: 11.010752688172044
- p_value: **0.0009058485425290641**（α=0.05で有意．較正後が正解に転じた行(63)が誤りに転じた
  行(30)を上回り，方向は改善で統計的に有意．platt(p=0.139)・isotonic(p=0.301)はいずれも有意
  差なしだったのに対し，今回は有意な改善という異なる結果）

**手順8: flip rate（argmaxが変わった行の割合，`id`で対応付け，Iter29/30と同じ定義）**

- **137/1600 = 0.085625（8.5625%）**．Iter29（platt，11.0%）・Iter30（isotonic，14.3125%）
  いずれよりも低い．調査(Iter31)の合成データ実測（`ensemble=True`下でtemperatureは
  isotonic/plattより小さいflipになる，1.12% vs 4.20%/3.85%）と定性的に整合する方向（実データでの
  絶対値は合成データの規模・分離度と異なるため単純比較はできないが，3手法中もっとも低いという
  順序は一致）．

**手順6: per-domain非退行チェック（Iter30で確立した3段構成：recall=ドメイン別McNemar・
precision=Fisher正確検定・20指標へBH補正q=0.05）**

全20指標の点推定（較正前→較正後）と個別検定のp値，BH補正後の有意フラグ：

| domain | metric | before | after | p_value | BH有意 | 方向 |
|---|---|---|---|---|---|---|
| business_economics | recall | 0.5179 | 0.5417 | 0.220671 | 否 | 改善 |
| business_economics | precision | 0.4328 | 0.4643 | 0.546043 | 否 | 改善 |
| computer_science | recall | 0.5417 | 0.5714 | 0.227800 | 否 | 改善 |
| computer_science | precision | 0.5987 | 0.6234 | 0.725154 | 否 | 改善 |
| education | recall | 0.4059 | 0.4588 | 0.015861 | 否 | 改善 |
| education | precision | 0.4631 | 0.5306 | 0.295424 | 否 | 改善 |
| general | recall | 0.5488 | 0.5732 | 0.220671 | 否 | 改善 |
| general | precision | 0.6522 | 0.6528 | 1.000000 | 否 | 改善 |
| history_culture | recall | 0.6667 | 0.6786 | 0.723674 | 否 | 改善 |
| history_culture | precision | 0.7320 | 0.6994 | 0.535412 | 否 | 悪化 |
| legal | recall | 0.5833 | 0.5778 | 1.000000 | 否 | 悪化 |
| legal | precision | 0.7500 | 0.7820 | 0.569458 | 否 | 改善 |
| mathematics | recall | 0.6190 | 0.6310 | 0.723674 | 否 | 改善 |
| mathematics | precision | 0.7075 | 0.7020 | 1.000000 | 否 | 悪化 |
| medical | recall | 0.4831 | 0.5112 | 0.182422 | 否 | 改善 |
| medical | precision | 0.4725 | 0.5056 | 0.599143 | 否 | 改善 |
| natural_science | recall | 0.5655 | 0.5833 | 0.605577 | 否 | 改善 |
| natural_science | precision | 0.5135 | 0.5444 | 0.600359 | 否 | 改善 |
| social_science | recall | 0.5774 | 0.5774 | 0.751830 | 否 | 改善 |
| social_science | precision | 0.6340 | 0.6382 | 1.000000 | 否 | 改善 |

BH（q=0.05）通過（adjusted有意）は20指標中**0件**．悪化方向の指標（history_culture_precision・
legal_recall・mathematics_precision）はいずれもp値が0.53-1.00と大きく，統計的な退行の根拠はない．

**medical_recallの内訳（Iter30でBH補正後有意に悪化していた指標，今回の再現有無を確認）**:
discordant_a_only=2（較正前のみ正解）・discordant_b_only=7（較正後のみ正解）・discordant_pairs=9・
chi2=1.7778・p=0.182422．**Iter30（isotonic，discordant_a_only=19・discordant_b_only=1・
p=0.000144・有意に悪化）とは対照的に，temperatureではmedical_recallはむしろ改善方向
（0.4831→0.5112）であり，統計的に有意な変化もない**．調査(Iter31)の理論的予想（単一Tはクラス
固有のOvR曲線歪みを構造的に持たないため，isotonicのmedical系統的圧縮は再現しないはず）と実測が
一致した．

**手順7: temperature特有の実装確認チェックリスト（`probabilities`フィールドを使用，1600行対象，
調査(Iter31)申し送り3により簡易報告）**

- (a) 確率のいずれかが厳密に`0.0`または`1.0`になっている行数: **0/1600**
- (b) 10クラス全てが`0.1`に近い（`math.isclose(p, 0.1, abs_tol=1e-9)`）uniform fallback行数:
  **0/1600**
- (c) 選択ドメインのconfidenceと同一の値を持つ他ドメインが存在する行の割合（tie率，厳密な
  浮動小数点一致で判定）: **0/1600（0.0000%）**

3点とも予想どおり該当0件だった．softmaxの出力は有限のロジット入力に対し厳密に0や1にはならず，
tieも理論上は浮動小数点の偶然の一致でしか起こらないという調査(Iter31)の実装読解（分かったこと(4)）
と実測が一致した．非ゼロの値は観測されなかったため詳細報告は不要と判断した．

**使用データ**:

- 訓練データ: `data/classifier_train.jsonl`（1427件，legal 77件・他9ドメイン各150件，
  Iter29/30と同一）
- 評価データセット（再embedding対象）: `data/dataset.jsonl`（1600件）
- 較正前の実行結果: `results/20260731_162722/results.jsonl`（Iter28実測，1600行，再実行なし）
- 較正後の実行結果（新規生成）: `results/iter31_calibrated_predictions.jsonl`（1600行，
  `probabilities`フィールド付き）
- 新規モデルアーティファクト: `models/domain_classifier_temperature.joblib`（本番
  `models/domain_classifier.joblib`は無変更のまま，タイムスタンプで確認済み）

**実行時間・実機呼び出しの有無**:

- `train_domain_classifier.py`: 124.55秒（1427回のembedding呼び出し）
- `evaluate_classifier_calibration.py`: 141.56秒（1600回のembedding呼び出し）
- 実機呼び出しはwafl500（192.168.15.100:11434）へのembeddingのみ（`nomic-embed-text`），
  計3027回．LLM生成・probe・dispatchは一切発生していない．
- 接続経路はIter29/30と同一の既存SSHローカルポートフォワード（`127.0.0.1:11435` ←
  `wafl500:11434`）をそのまま流用．新規に張り直す必要はなく，実行中のログ・エラーに異常なし
  （例外・タイムアウト・リトライなし，両スクリプトとも正常終了メッセージを出力）．

**state.json更新**: `status: waiting_experiment`（開始時，`experiment_dir:
results/iter31_calibrated_predictions.jsonl`・`experiment_deadline`設定）→`running`（完了時，
`experiment_dir`/`experiment_deadline`を`null`に戻した）．`e31_results`への数値記録・`judgment`
確定はフェーズ5b（rc-analyst）に委ねる（本フェーズでは数値の良否判定は行わない）．

---

### 分析(解釈) (Iter31)

**成功条件（d0003 X9，計画(Iter31)節，AND条件）との照合**

1. **ECE ≤ 0.150**: 明確に成立。較正後 0.071201 は較正前 0.193358 から −12.22pt（相対63.2%減）
   であり，3手法中もっとも大きい改善幅である。目標0.150に対する余裕は 7.88pt（isotonicの
   2.86pt・plattの未達）を大きく上回り，platt・isotonicいずれと比べても大きな差で目標を
   上回っている。ルーティングは決定論的（config.yml success_criteria (5)）であり，同一1600問・
   同一embeddingモデルに対し較正手法のみを変えた比較のため，この差分はノイズではなく較正手法の
   変更そのものが生んだ実測値と判断してよい（Iter29・Iter30と同じ根拠）。
2. **top1_accuracy 非退行**: 成立するだけでなく，**成功条件が想定した「非退行」の範囲を
   超えて有意な改善**である。計画(Iter31)の条件2は「p>=0.05，または改善方向」を非退行の
   基準としていたが，実測はMcNemar p=0.000906（α=0.05で有意）かつdiscordant_b_only
   （較正後のみ正解，63）がdiscordant_a_only（較正前のみ正解，30）を倍以上上回る改善方向
   であり，**統計的に有意な改善**と言うべき水準にある。top1_accuracyの絶対値も0.585→0.605625
   （+2.06pt）と3手法中もっとも大きい伸びであり，platt（p=0.139，非退行だが有意差なし）・
   isotonic（p=0.301，同）とは質的に異なる結果である。較正が単に「悪化していない」だけでなく
   ルーティング精度そのものを引き上げた可能性を示している。
3. **per-domain 20指標のBH補正後・悪化方向有意指標0件**: 成立。20指標（10ドメイン×
   precision/recall）のうちBH（q=0.05）通過は**0件**であり，isotonicの`medical_recall`
   （p=0.000144，通過1件）やIter29 platt（字義通り基準で9件該当，事後分析で全て非有意と
   判明）と異なり，最初から厳格な多重比較補正の下で悪化方向の有意指標が皆無だった。悪化方向
   の3指標（history_culture_precision, legal_recall, mathematics_precision）もp値は
   0.53〜1.00と大きく，統計的な退行の根拠はない。3手法のうちこの条件を単独で満たしたのは
   temperatureのみである。
4. **【但し書き，調査(Iter31)申し送り2・計画(Iter31)条件4】ensemble由来の非ゼロflip
   （8.5625%＝137/1600）の解釈**: この値は理論保証（sklearn公式の「Tはsoftmaxのargmaxの
   位置に影響しない」）が主張する範囲の**外側**（個々の(estimator, T)ペア内ではなく，
   `ensemble=True`による5fold平均化という別の機序）で生じており，それ自体を理由に条件2・3を
   不成立とはしない，という計画(Iter31)の事前合意どおりに扱った。実際，条件2・3の判定は
   統計的検定（McNemar・Fisher・BH補正）の結果のみに基づいており，flip率が非ゼロであること
   自体はここまでの分析で不利に働いていない。むしろ8.5625%はisotonic（14.3125%）・
   platt（11.0%）より低く，調査(Iter31)の合成データ実測（temperatureのensemble由来flipは
   isotonic/plattより一貫して小さい）と実データでも順序が一致した。この事実は，「ensemble
   平均化由来の偶発的再配分」以上の何かがtop1_accuracyを押し上げているという解釈（条件2）
   を弱めるものではなく，むしろ較正手法自体が生む再配分の総量が少ないままaccuracyが伸びた
   ことを示しており，isotonic/plattのような「大きな再配分の副産物として一部ドメインが
   犠牲になる」構造とは異なる結果である。

**isotonic（Iter30）との対比 — medical_recall問題の再現有無**

Iter30ではmedical_recallがBH補正後も有意に悪化（0.4831→0.3820, p=0.000144，
discordant a_only=19:b_only=1）し，較正後の最大確率0.7062が他9ドメイン（0.7496〜0.8795）
を全て下回るという，isotonic較正曲線のmedicalクラス固有の系統的圧縮が疑われていた。
今回のtemperatureでは，同じmedical_recallがdiscordant a_only=2:b_only=7・p=0.182422
（有意差なし）で，**点推定はむしろ改善方向（0.4831→0.5112）**である。

これは調査(Iter31)の理論的予想——「temperatureは単一スカラーTでロジット全体を変換する
構造上，クラスごとの個別較正器を持たず，isotonic/PlattのOvR方式由来のクラス固有曲線歪みを
構造的に持たない」——を**強く支持する証拠**と評価してよい。理由は次の3点である。

1. 同一の訓練データ（`data/classifier_train.jsonl`，medical 150件）・同一の評価データ
   （1600問）・同一のbase estimator（`LogisticRegression(class_weight='balanced')`）・
   同一の`cv=5, ensemble=True`という条件下で，較正手法だけを変えた比較になっている。
   単一レバー原則が厳密に保たれているため，medical_recallの挙動の違いは較正手法の構造差に
   帰責してよい。
2. isotonicのときに観測された「medicalだけ較正後の最大確率が体系的に低い」という現象は，
   OvR方式（各クラスを独立に二値較正してから正規化する）に固有の自由度の高さ（区分定数
   フィットが特定クラスのheld-outデータで不安定に歪みうる）に起因すると解釈されていた。
   temperatureは全クラスに同一の逆温度を掛けるだけで，どのクラスかによらず変換が対称的
   であるため，「特定の1クラスだけ確率の天井が下がる」という現象が構造的に起こり得ない。
   今回の実測（medical・legal含め全10ドメインで悪化方向の有意指標が0件）はこの構造的な
   予測と整合する。
3. ただし，これは「1回の実験（n=1）による整合」であることに留意が必要である。medical
   クラスの埋め込み分離が本来弱いという可能性自体を否定する証拠ではなく，あくまで
   「OvR方式由来の較正曲線歪みという機序」が今回不在だったことを示すに留まる。それでも
   Iter30が示した唯一の懸念（medical_recall）が，理論から予想された通りの手法変更
   （temperatureへの切り替え）だけで解消したことは，偶然の一致にしては機序の説明が具体的
   （単一スカラー変換とOvR個別較正器という明確な構造差）であり，強い状況証拠と判断する。

**platt/isotonicとの比較でtemperatureが3手法中もっとも成功条件を満たしている理由の考察**

事前の留保（config.yml note・backlog B51・調査(Iter31)分かったこと(4)）は「temperatureは
表現力が低いためECE改善幅がisotonic・plattより小さくなる可能性」を懸念していたが，実測は
その逆で，ECE改善幅は temperature(0.1222) > isotonic(0.0719) > platt(0.0258) という
**もっとも深い**結果になった（数値は較正前0.193358からの絶対改善幅）。この逆転は次のように
解釈できる。

- 較正の訓練データは1427件を10クラスに分割し（medical/legal以外は各150件，legalのみ77件），
  `cv=5`ではさらに1foldあたり数十件規模まで細分される。isotonic・plattはこの少量データ上で
  **クラスごとに個別の較正関数**を学習するため，held-outデータのノイズに対して過学習しやすい
  （isotonicは特に自由度が高い区分定数フィットで，Iter30のmedical系統的圧縮はこの過学習の
  症状と整合する）。temperatureは全クラス共通の**単一スカラーパラメータ**しか学習しないため，
  1427件全体（実質的に多クラスのmultinomial loss全体）から1つのTを推定でき，個々のクラスの
  小標本性に脆弱ではない。つまり，このデータ規模では「表現力の低さ」がむしろ分散を抑え，
  過学習を防いだと考えられる——古典的なバイアス・分散トレードオフで，パラメータ数が少ない
  ほうが小標本の較正タスクでは汎化しやすかった，という説明である。
- もう一つの見立ては，このLogisticRegression分類器の較正誤差が，そもそも「クラスごとに
  異なる歪み方をする」構造ではなく，「全クラス一律に過信（over-confident）している」という
  **大域的な過信バイアス**が支配的だった可能性である。もしそうであれば，単一Tによる大域的
  スケーリングだけで大部分の誤差を解消でき，OvR方式のクラス固有補正は，本来存在しない
  クラス間の歪みの違いを学習データのノイズから読み取ってしまい，かえって較正を悪化させる
  （isotonicのmedical系統的圧縮，plattのECE絶対閾値未達）方向に作用したと考えられる。
- 前者・後者いずれの説明も「表現力が高い手法が必ず良い較正を生むとは限らない」という一般的な
  較正手法選択の知見（少数クラス・小標本条件下での過学習リスク）と整合しており，本リポジトリ
  の訓練データ規模（1427件・10クラス）が，isotonic/plattの柔軟性を活かすには小さすぎた
  可能性を示唆する。ECE改善幅の逆転という結果自体は今回のn=1測定だが，isotonic（Iter30）で
  観測された系統的圧縮という具体的な機序と符合しており，単なる偶然の逆転とは考えにくい。

**本番反映（`models/domain_classifier.joblib`をtemperature版に置き換えるか）についての見解
（提案，確定はrc-reflector）**

**採用（adopted）し，本番反映を進めることを提案する**。判断基準:

- 成功条件1〜3のAND条件をすべて満たした較正手法は，platt・isotonic・temperatureの3手法中
  temperatureのみである。isotonic（Iter30）はmedical_recall悪化で条件3不成立，
  platt（Iter29）はECE絶対閾値未達で条件1不成立と，いずれもpartialで確定している。
  temperatureはこの2つの懸念をいずれも回避しており，「AND条件を字義通り満たす」という
  意味で今回初めて明確なadopted相当の結果が得られている。
- 条件2（top1_accuracy）は非退行を超えて有意な改善（p=0.000906）であり，条件1（ECE）も
  目標に対し7.88ptの余裕がある。isotonic・plattのように「AND条件の一部だけ危うい」形では
  なく，3条件のいずれにも明確な余裕がある。
- 可逆性の観点でも問題は小さい。`models/domain_classifier_temperature.joblib`は既に
  別名で生成済みであり，本番`models/domain_classifier.joblib`との入れ替えはファイルの
  差し替えのみで完結し，何らかの不具合が判明した場合は較正前のjoblibへ即座に戻せる
  （config.yaml自体は一切変更していないため，ロールバックにコード変更は不要）。
- 一方で，確信度を完全な最終確定ではなく「提案」に留めるべき留保点が2つある。(a) 本判定は
  n=1（ルーティングは決定論的だが，1600問という単一の評価セット・単一の訓練データ分割に
  基づく）測定であり，Iter28・Iter29・Iter30と同じ制約を共有している。(b) isotonic
  （Iter30）のmedical_recall悪化が「OvR由来の機序」の実例として片付けられるかどうかは，
  今回の1イテレーションの整合的な結果からの推論であり，直接に反証実験を行ったわけではない
  （例えば，temperatureをさらに複数回・複数の訓練データ分割で再現するような追加検証は
  今回行っていない）。判断の主要な数値（ECE・McNemar・BH補正）自体は確定的であり追加反復を
  要するとは考えないが，「なぜtemperatureがisotonic/plattより優れていたか」という機序の
  説明は今回の考察であり，本番反映後もECE・per-domain指標の定期的なモニタリングを継続する
  ことを勧める。

**確信度と追加反復の要否**: 成功条件1〜3の判定そのものの確信度は高い（決定論的ルーティング・
BH補正済み・3手法全てで同一手順を適用した比較のため）。追加反復（同一実験の再実行）は
不要と考える——ルーティングが決定論的である以上，再実行しても数値は変わらない。ただし
上記のとおり「temperatureが優れていた理由」の機序面の説明はn=1の考察に留まるため，
本番反映後の運用モニタリング（例えば次に大きな訓練データ更新が入った際にECE・per-domain
指標を再確認する）は推奨事項として申し送る。

**総合判断（rc-analyst提案）: adopted（全面採用）**。config.ymlの`classifier_calibration`
レバーは`[platt, isotonic, temperature]`の3値を全て試し終えており，platt=partial・
isotonic=partial・temperature=今回の提案どおりadoptedとなれば，Y4（分類器の較正）は
temperatureの採用をもって完了とすることを提案する。最終的な採否確定と，
`models/domain_classifier.joblib`の実際の置き換え作業はrc-reflectorに委ねる。

---

### 考察 (Iter31)

**判定: adopted（全面採用，rc-analyst提案を覆さず確定）**。d0003 X9 の成功条件（ECE≤0.150・
top1_accuracy非退行・per-domain 20指標のBH補正後悪化方向有意指標0件のAND条件）を，`platt`
（Iter29，ECE絶対閾値未達でpartial）・`isotonic`（Iter30，medical_recallのBH補正後有意悪化で
partial）に続き3手法目の`temperature`が初めて明確に満たした。ECEは0.193358→0.071201（目標に
7.88ptの余裕，3手法中もっとも大きい改善幅），top1_accuracyは0.585→0.605625でMcNemar
p=0.000906の**有意な改善**（非退行を上回る），per-domain 20指標のBH補正後有意指標は0件。
Iter30で唯一の懸念だったmedical_recallも，temperatureでは有意差なし（p=0.182422）でむしろ
改善方向（0.4831→0.5112）と，isotonicの系統的圧縮が再現しなかった。rc-analystの分析（機序：
temperatureは単一スカラーTでロジット全体を変換するためisotonic/plattのOvR方式由来のクラス
固有曲線歪みを構造的に持たない）は，同一訓練データ・同一評価データ・同一cv/ensemble設定という
単一レバー原則が厳密に保たれた比較の下で得られた結果であり，覆す理由を見いだせなかったため
確定させる。

**本番反映: 実施済み**。`models/domain_classifier.joblib`（旧・較正なし，
sha256=`3a5610a...`）を`models/domain_classifier_uncalibrated_pre_iter31.joblib`へ退避のうえ，
`models/domain_classifier_temperature.joblib`（sha256=`04bb9ff...`）で置き換えた。判断根拠:
(1) 成功条件のAND条件を明確な余裕（ECE 7.88pt・top1有意改善・BH補正後有意退行0件）で満たして
いる，(2) `config.yaml`・公開APIの変更を一切伴わない可逆なファイル差し替えである（不具合が
判明すれば`models/domain_classifier_uncalibrated_pre_iter31.joblib`へ即座に戻せる），
(3) これは委譲時の指示で明示的に「rc-reflectorの自律判断範囲内（可逆な判断）として進めて
構わない」とされた操作である。**注意**: `models/`はリポジトリの`.gitignore`（19行目）で除外
されており，この置き換えはgit管理下にない。ロールバック手順と両ファイルのsha256はこの節と
上記に記録した以外に残らないため，次回このモデルに触れる際は本節を参照すること。

**学び**:

1. **isotonicのmedical_recall悪化は「OvR方式由来のクラス固有曲線歪み」という機序で
   説明できることが，temperatureへの切り替えのみで解消したという形で強く裏付けられた**。
   同一データ・同一cv/ensembleの下で較正手法だけを変えた比較が3イテレーション連続で
   積み上がったことで，この機序の特定は単発の考察ではなく再現性のある知見になった。
2. **「表現力が高い較正手法が必ず良い較正を生むとは限らない」という一般的な較正手法選択の
   知見が，本リポジトリの訓練データ規模（1427件・10クラス，legalのみ77件）で実測として
   裏付けられた**。ECE改善幅はtemperature(0.1222) > isotonic(0.0719) > platt(0.0258)と，
   もっとも柔軟性の低い手法がもっとも大きく改善するという事前の留保（config.yml note・
   backlog B51）とは逆の結果になった。小標本条件下ではOvR方式のクラス別自由度がheld-outの
   ノイズを拾って過学習し，かえって較正を悪化させるためと考えられる。次に較正関連のレバーを
   検討する際は，「手法の表現力の高さ＝較正の質」という前提を置かないこと。
3. **`ensemble=True`由来の非ゼロflip（合成データ実測1.12%，実データ8.5625%）は，
   sklearn公式の「Tはsoftmaxのargmaxを変えない」という理論保証の範囲外（個々の
   (estimator, T)ペア内の話であり，5fold平均化という別の機構）であるという整理は，
   今後isotonic/platt/temperatureいずれについても「flipが非ゼロ＝理論違反」という
   短絡的解釈を避けるために有効だった。次回較正手法を検討する際も踏襲すること。
4. **`models/`がgitignore対象であるため，較正済み分類器の本番反映はgit履歴に残らない**。
   D5（backlog未解決事項，`data/`/`models/`のバージョン管理方針）が引き続き未解決であり，
   今回のように本番アーティファクトを差し替える判断が何度も発生する局面では，最低限
   sha256ハッシュのマニフェストをjournal/backlogに記録する運用（今回実施した方式）を
   今後も徹底する必要がある。

**Y4（分類器の較正，d0003 X9）は本イテレーションをもって完了**。config.ymlの
`classifier_calibration`レバーは`[platt, isotonic, temperature]`の3値すべてを試し終えた。

**次イテレーション（Iter32）の単一レバー決定**: d0004 §5の優先順位はY1（完了）→Y4（完了，
本イテレーション）→Y2（前提整備，スキーマ変更を伴い着手前にユーザー確認が必要）→Y3（Y2完了後）
→Y5（education/legalのデータ不均衡是正）である。Y2は`config.yaml`への
`dispatch_candidate_threshold`新設・`aggregator.select_dispatch_targets()`のシグネチャ変更を
伴い，backlog B49・B50・B51で繰り返し「着手前にユーザー確認が必要」と申し送られてきた
不可逆側の判断であり，rc-reflectorの自律判断権限（可逆な判断に限る）では着手を開始できない。
一方，Y3はY2完了が前提のため同様に着手不能。したがって実行可能な登録済みレバーは
`classifier_calibration`（完了）・`fallback_policy`（完了）のみとなり，`aggregation_method`
（Y3）はY2完了までブロックされたまま実質「試せない」状態にある。

これは「config の全 levers を試し切った」場合と実質的に同じ状況（唯一残る登録レバーが
ブロックされていて実行不能）と判断し，SKILL.mdが定める停止条件の優先順1（journal/backlogの
学びから次の有望なレバーを自分で考案し，config.ymlのlevers末尾へ追記して継続する）に従い，
**Y5（education/legalのデータ不均衡是正，d0003 X8）を新規レバーとしてconfig.ymlへ追記し，
Iter32の単一レバーとする**。理由と選定過程はbacklog.md B52に記録する（下記参照）。Y2は
自律着手不能なままのため，backlogの「要レビュー」として引き続き申し送る（新規の追加事項はない）。

---

## Iteration 30: 分類器較正のisotonic方式によるECE目標達成の追試とドメイン別非退行の全数検証

### 調査 (Iter30)

**問い**:
1. 1427件・legal 77件という規模で`CalibratedClassifierCV(method='isotonic')`を使う具体的リスクは何か．
   Iter29が確認した「≪1000件で過学習」という sklearn 公式の目安を，本イテレーションで独立に裏取りできるか．
2. 20指標（10ドメイン×precision/recall）の非退行チェックにおいて，Iter29の学び1（CI下限の単純前後比較は
   多重比較補正なしでは脆弱）を受け，どう改めるべきか．Bonferroni／Benjamini-Hochberg（BH）／区間の
   非交差／ドメイン単位McNemar検定のうち，実装コストと妥当性のバランスが良い方法を1つ推奨する．
3. isotonicはplattより表現力が高い分，過学習時の argmax flip がplattより大きくなりうるか．
   実装上の落とし穴（単調性の破れ・確率の0/1張り付き等）を整理する．
4. `method='temperature'`はsklearnの`CalibratedClassifierCV`に実在するか．Iter29 reflectorの申し送り
   （sklearn>=1.8で利用可能）の前提が正しいかを確認する．

#### 分かったこと

**(1) isotonic較正の技術的妥当性 — Iter29の裏取りに加え，新たな具体的懸念点を確認**

本リポジトリの実行環境（`.venv`，`uv.lock`固定）で `scikit-learn==1.9.0` がインストール済みであることを
`uv run python` から直接確認した．インストール済みパッケージのソース
（`sklearn/calibration.py`，`CalibratedClassifierCV`のdocstring）には
「Isotonic calibration is not recommended when the number of calibration samples is too low
``(≪1000)`` since it then tends to overfit」という文言が verbatim で存在し，Iter29が引用した
sklearn公式ドキュメント（`calibration.html`）の記述と完全に一致することを一次ソース（インストール
済みパッケージそのもの）で再確認した．さらに `tavily-extract` で `calibration.html` を直接取得し，
「Overall, 'isotonic' will perform as well as or better than 'sigmoid' when there is enough data
(**greater than ~ 1000 samples**)」という定量的な閾値の原文を確認した．また同じ文言
（`<<1000`／Platt推奨）が sklearn 0.18（2016年当時）の過去ドキュメントにも既に存在していたことを
web検索で確認しており（`vighneshbirodkar.github.io`のアーカイブ），この目安は最近の変更ではなく
10年近く sklearn が一貫して明記してきた安定した経験則である．

本データでの実測（Iter29既出，本イテレーションで再確認）: `cv=5`・`ensemble=True`の下では
1 fold あたりの較正サンプル数は9ドメインで約30件，legalで約15件．これは「≪1000」を大きく下回るのは
もちろん，isotonic回帰それ自体の性質（ノンパラメトリックで自由度が事実上サンプル数に等しい）から
言えば，1000件どころか一般的な「数百件」規準（emergentmind.comの「200件未満で過学習し得る」という
目安，Iter29既出）にも legal は届かない．**追加確認**: `IsotonicRegression`は`out_of_bounds="clip"`
で運用されており，較正用の held-out データに含まれない極端なスコアはヒストグラムの両端の値へ
クリップされる．該当ドメインの held-out データが少ないほど，この「両端の値」自体が0や1に近い
不安定な推定値になりやすい．

**(2) 多重比較への対処 — Benjamini-Hochberg（BH）法を，指標の対応構造に応じた2種類の検定と
組み合わせて用いることを推奨**

一般的なガイドライン（LaunchDarkly社の実験ドキュメント，2026年時点で確認）は「比較数が3以下なら
Bonferroni，それを超えるとBHの方が検出力とのバランスが良い」と明記している．20指標（10ドメイン×
precision/recall）はこの目安を大きく超えるため，Bonferroni（α=0.05/20=0.0025）は過度に保守的で
真の退行を見逃すリスクが高く，「区間の非交差」を基準にする案（config.ymlの申し送りにある選択肢の
一つ）はBonferroniよりさらに保守的な基準になりがちで感度が低い．

**推奨: BH法（FDR制御，q=0.05）を第一候補とする．ただし適用する検定は，指標ごとの対応構造に応じて
使い分けるべきである**．

- **recall**（分母＝真のドメインがXである行の集合．較正前後で分母の行集合は不変＝対応データ）には，
  既存の`metrics.py:compute_mcnemar_test`をドメイン別にサブセット適用する（=10検定）．これは
  Iter29の学び1が示唆する「ドメイン単位のMcNemar検定」をそのまま使える構造である．
- **precision**（分母＝分類器がXと予測した行の集合．較正で argmax が変われば分母の行集合自体が
  変わる＝非対応データ）は，McNemarの前提（同一対象への対の観測）を満たさないため，2標本比率の
  差の検定（Fisher正確検定または$\chi^2$検定，非対応）を用いる（=10検定）．
- 得られた計20個のp値に対しBH法を一括適用し，adjusted p<0.05のもののみを「統計的に有意な退行」と
  判定する．実装コストは低い（既存の`compute_mcnemar_test`のドメイン別ラッパー関数＋
  `scipy.stats.fisher_exact`または`chi2_contingency`の呼び出し＋BH補正（p値をソートして
  `p_(i) * m / i` を取るだけの数行）で完結し，外部ライブラリの新規追加は不要）．

**(3) isotonic特有の非退行確認の注意点 — sklearn公式ドキュメント・ソースコードで3点を具体的に確認**

- **ties（同値化）による ranking の粗視化**: sklearn公式ドキュメント（`calibration.html`
  1.16.3.3節脚注）が明記：「isotonic regression introduces ties in the predicted probabilities」
  であり，「It is generally expected that calibration does not affect ranking metrics such as
  ROC-AUC. However, these metrics might differ after calibration when using
  `method="isotonic"`」．一方 sigmoid は「a strictly monotonic transformation and thus keeps
  the ranking」と明記されている．本タスクのargmax選択は本質的にランキング操作であるため，
  isotonicはplattよりtie（複数ドメインが同一の較正後確率を持つ状態）を生みやすく，僅差の候補間で
  argmaxが不安定化するリスクがplattより高いと考えられる．cv fold あたりのサンプルが最少のlegal
  （約15件）で最も起きやすい．
- **確率の0/1張り付き（exact zeros）**: sklearn公式ソース（`_CalibratedClassifier.predict_proba`
  のdocstring）に「The predicted probabilities. Can be exact zeros.」と明記されている．
  `IsotonicRegression(out_of_bounds="clip")`は較正用データの範囲外のスコアを最も近い観測値へ
  クリップするため，その観測値自体が0や1（小標本のheld-outデータでは十分あり得る）であれば，
  較正後の確率がそのまま0または1に張り付く．これはIter16で問題視された「verbalized confidence
  の0/1飽和」と同種の病理を，較正という「飽和を直す」はずの処理が別の経路（isotonicの区分定数性）
  で再導入しうることを意味し，ECEの見かけ上の改善と裏腹に個々の予測の信頼性を損なう可能性がある．
- **全クラスが0になった場合のuniform fallback**: sklearn公式ソース（`_fit_calibrator`直後の
  `predict_proba`実装，コメント「In the edge case where for each class calibrator returns a
  zero probability for a given sample, use the uniform distribution instead」）が明記する
  実装上のフォールバック．10クラス全てのOvR較正器が0を返すサンプルが発生すると，較正後確率は
  10クラス均等（各0.1）に置き換わり，argmaxは分類器本来のランキングと無関係な（実装依存の）
  tie-breakで決まる．発生頻度は不明だが，該当した場合は「較正が改善させた」のではなく
  「較正が情報を破壊した」ケースであり，flip rateの数値だけでは区別できない．**実験時は
  `predict_proba`の行和が学習データ内で0.1×10=1.0のuniform行になっていないか（例えば
  `np.allclose`で0.1の一様分布との一致を検出）を追加でチェックすることを推奨する**．

**(4) `method='temperature'`は実在する — Iter29 reflectorの申し送りは正確**

課題文は「sklearnにあるのはsigmoidとisotonicの2値のみのはず」という疑いを提示していたが，
本リポジトリの実行環境で直接確認した結果，**Iter29の申し送りは正確であり，疑いは誤りだった**．

- `uv run python -c "from sklearn.calibration import CalibratedClassifierCV; help(...)"`で，
  `method`パラメータの型注釈が `{'sigmoid', 'isotonic', 'temperature'}` であることを確認．
  docstringに `.. versionchanged:: 1.8 Added option 'temperature'.` と明記されている．
  本リポジトリの`uv.lock`は`scikit-learn==1.9.0`を固定しており，1.8以降のバージョンなので
  `temperature`は現に利用可能である．
- sklearn公式ドキュメント（`calibration.html` 1.16.3.4節，`tavily-extract`で直接取得）は
  temperature scalingについて次のように明記している：「temperature scaling naturally supports
  multiclass predictions by working with logits and finally applying the softmax function」
  （sigmoid/isotonicのようなOvR分解＋事後正規化が不要）．「The parameter T is learned by
  minimizing log_loss ... on a hold-out (calibration) set. Note that T does not affect the
  location of the maximum in the softmax output. Therefore, temperature scaling does not alter
  the accuracy of the calibrating estimator.」——ロジット（`decision_function`の出力，または
  `predict_proba`の対数）全体を単一のスカラーTで割るだけの変換であるため，クラス間の大小関係
  （argmax）が理論的に不変であることが公式に保証されている．sklearnソース
  （`_fit_calibrator`）でも，`method="temperature"`の場合はsigmoid/isotonicのようにクラスごとに
  個別の較正器を作らず，**単一の`_TemperatureScaling`インスタンスのみを fit する**実装になって
  おり，OvR方式に起因するargmax入れ替わりのリスク（Iter29が指摘した多クラス較正の主要懸念）は
  構造的に排除されている．
- 使用中のbase estimator（`LogisticRegression`）は`decision_function`を持つため，temperature
  scalingはロジットを直接使う経路（`predict_proba`の対数を取る近似ではなく）で動作する．

#### rc-planner への申し送り

1. **isotonicの技術的リスクはIter29の想定どおり，むしろ具体化された**．legalドメイン
   （較正fold内約15件）はsklearn公式の「≪1000」「~1000件超で互角以上」のどちらの目安からも
   大きく外れており，`cv=5`のまま実施する場合はplatt以上に慎重な監視が要る．
2. **per-domain非退行チェックの運用を今回から変更することを強く推奨する**：
   `success_criteria (2)`の「CI下限の単純比較」をそのまま使い続けると，Iter29で実際に起きたように
   20指標中9指標が偽陽性で該当してしまう．今回のisotonic実験では**最初から**（事後の穴埋めでなく）
   (a) recallはドメイン別McNemar検定，(b) precisionは2標本比率検定（Fisher正確検定），
   (c) 計20個のp値へBH法（q=0.05）を適用，という3段構成で判定することを計画に含めるべきである．
   これは既存の`compute_mcnemar_test`／`compute_wilson_confidence_interval`の関数群を活かしつつ
   数十行の追加で実装できる．
3. **isotonic特有の実装確認項目を計画・実験段階でチェックリスト化すること**: (a) 較正後
   `predict_proba`の値が厳密に0または1になっている行がないか，(b) 10クラス全て0.1（uniform
   fallback）になっている行がないか，(c) 較正後の同一confidence値を持つ行（tie）の割合，
   の3点をIter29のflip rate報告に加えて算出する．特にlegalドメインの行を優先的に確認する．
4. **isotonicがECE目標（0.150以下）に届かない場合の次点候補は`method='temperature'`で確定できる**．
   Iter29の申し送りは正確であり，本リポジトリの`scikit-learn==1.9.0`で実際に利用可能である．
   temperature scalingはtop1_accuracy不変が理論的・実装的（単一の`_TemperatureScaling`インスタンス
   のみをfitする構造）に保証されるため，「ECE改善とルーティング非退行」というY4の目的に対し，
   sigmoid/isotonicのOvR方式が抱える構造的リスク（argmax入れ替わり，tie，0/1張り付き）を
   そもそも持たない代替である．ただし，temperatureは「多クラス全体で単一のTを学習する」ため，
   ドメインごとの較正の柔軟性はsigmoid/isotonicより低く，legalのように較正のずれ方が
   ドメイン固有の場合には改善幅が小さい可能性がある点は留保として記録する．
5. 今回のisotonic実験の計画では，Iter29の考察で確定した手順（全10ドメインのCIを較正前後で
   同一手順・最初から算出する）に加え，上記2・3の追加チェックを組み込むこと．

**出典**:
- ローカル実行環境の直接確認: `uv run python -c "import sklearn; print(sklearn.__version__)"`
  → `1.9.0`，および`sklearn.calibration.CalibratedClassifierCV`のdocstring・ソース
  （`.venv/lib/python3.12/site-packages/sklearn/calibration.py`）を`help()`・`grep`・`Read`で
  直接確認（2026-07-31実施，一次ソース）．
- https://scikit-learn.org/stable/modules/calibration.html （`tavily-extract`で直接取得，
  1.16.3.3 Multiclass support・1.16.3.4 Temperature Scaling・isotonic過学習閾値・
  ties/ranking注記，2026-07-31時点のstable版）
- http://vighneshbirodkar.github.io/scikit-learn.github.io/dev/modules/generated/sklearn.calibration.CalibratedClassifierCV.html
  （sklearn 0.18時代の同一文言のアーカイブ，`tavily-search`で発見．「≪1000」目安が10年近く
  一貫していることの裏付け）
- https://notes.cs307.org/classifier-calibration.html ，
  https://medium.com/data-science-at-microsoft/model-calibration-for-classification-tasks-using-python-1a7093b57a46
  （isotonicが区分定数関数でsigmoidより過学習しやすいという解説の補強，`tavily-search`）
- https://stats.stackexchange.com/questions/493393/ （isotonicがties経由でROC-AUC等のranking指標に
  影響するというコメント，`tavily-search`）
- https://launchdarkly.com/docs/guides/statistical-methodology/mcc （Bonferroni対BHの使い分け目安
  「3件以下ならBonferroni，それ以上ならBH」，`tavily-search`）
- https://docs.statsig.com/statsig-warehouse-native/features/statistics/methodologies/benjamini-hochberg-procedure
  （BH法の定義，FWER対FDRの違い，`tavily-search`）
- journal.md「調査 (Iter29)」節（本調査の裏取り元，sklearn issue #18709・#34312・
  emergentmind.comの引用は Iter29 で既出のため本イテレーションでは再掲のみ）

---

### 計画 (Iter30)

**仮説**: `scripts/train_domain_classifier.py:train_classifier()` の較正手法を
`method="sigmoid"`（Platt，Iter29 で partial 判定）から `method="isotonic"` へ切り替えると，
isotonic のノンパラメトリックな柔軟性により ECE が Platt（0.16751）よりさらに改善し，
目標の 0.150 以下へ到達する．一方，legal ドメイン（cv fold あたり較正サンプル約 15 件）では
isotonic 特有の過学習・tie・0/1 張り付きにより，per-domain の非退行が Platt 以上に脅かされる
リスクがある．この 2 つのトレードオフを，調査(Iter30) が申し送った多重比較補正済みの統計的
判定手順で最初から検証する．

**単一レバー**: `classifier_calibration`（`.claude/research/config.yml` のレバー名，150-170行）．
今回試す値は `values: [platt, isotonic]` のうち **`isotonic` のみ**（backlog B50 の自動選択）．
`cv=5`・`ensemble=True` は Iter29（Platt）と完全に同一のまま固定し，較正手法のみを変える．
`cv=3` 等の感度分析は，isotonic の主結果（`cv=5`）で per-domain 非退行が崩れた場合にのみ
副次分析として検討し，今回の主比較には含めない（backlog B50 の申し送りどおり）．

**固定する構成（Iter29 と完全に同一，`config.yaml` は一切変更しない）**:
`routing_method=supervised_classifier`，`confidence_threshold=0.0`・`dispatch_top_k=1`・
`aggregation_method=max_confidence`（Iter28 adopted 構成），`confidence_signal_method=self_report`，
`confidence_elicitation=top_k_with_probs`，`expert_model=expert-mesh-{domain}-lora`
（domain_count=10），`light_model=qwen3.5:4b-q4_K_M`，`embedding_model=nomic-embed-text`，
評価データセットは Iter25 以降固定の 1600 問（`data/dataset.jsonl`）．
`CalibratedClassifierCV` の `cv=5`・`ensemble=True` も Iter29 と同一．**今回変更するのは
`train_classifier()` 内の較正手法（`_CALIBRATION_METHOD` 定数の値）のみであり，`config.yaml`
のキーは 1 つも変えない．**

**変更ファイル・行**:

1. `scripts/train_domain_classifier.py`
   - `_CALIBRATION_METHOD = "sigmoid"`（55行）→ `"isotonic"` に変更。`_CALIBRATION_CV = 5`
     （56行）は無変更。
   - 45-54行のコメント（sigmoid を選んだ理由の説明）を，isotonic を選ぶ理由（Iter29 の Platt が
     ECE 絶対閾値未達で partial 判定・backlog B50 の自動選択）と，調査(Iter30) が確認した
     追加リスク（isotonic はノンパラメトリックで自由度が事実上サンプル数に等しく，sigmoid より
     過学習しやすい／`out_of_bounds="clip"` により legal の held-out データが極端値に張り付き
     やすい）に更新する。
   - モジュール冒頭 docstring（1-5行）の `Iter29, classifier_calibration=platt` を
     `Iter30, classifier_calibration=isotonic` に更新。
   - `train_classifier()` の docstring（95-97行）の `method="sigmoid"=Platt` を
     `method="isotonic"` に更新し，isotonic 特有の注意点（ties・0/1 張り付き・uniform
     fallback，調査(Iter30) 分かったこと(3)）を一言追記する。
   - 出力アーティファクト名は `models/domain_classifier_isotonic.joblib`（Platt 版
     `models/domain_classifier_platt.joblib` とは別名で新規生成，本番
     `models/domain_classifier.joblib` は上書きしない）。
   - `tests/test_train_domain_classifier.py` は Iter29 で `cv=5` 実行に必要な最小データ量
     （各クラス5件）へ既に拡張済みであり，`method` を変えても `StratifiedKFold` の分割条件は
     変わらないため無変更で通る見込み（実装フェーズで実行して確認する）。

2. `scripts/evaluate_classifier_calibration.py`
   - `predict_calibrated_rows()`（55-82行）が返す各行の辞書に，`"probabilities"`
     フィールド（`{domain: float(p) for domain, p in zip(classes, probabilities)}`，10 ドメイン
     全ての確率）を追加する。Iter29 では選択ドメインの confidence のみで十分だったが，
     isotonic 特有のチェックリスト（0/1 張り付き・uniform fallback・tie 検出）には全クラスの
     確率ベクトルが必要なため（調査(Iter30) 申し送り3）。既存フィールド（`id`／
     `expected_domains`／`selected_domain`／`confidence`）は変更しない（`metrics.py` の
     既存関数がそのまま読める後方互換を保つ）。
   - モジュール冒頭 docstring（1-26行）に，Iter30 で `probabilities` フィールドを追加した
     理由を追記する。CLI 引数（`--dataset`／`--classifier`／`--output` 等）は変更不要。

3. `metrics.py`（新規関数を追加，既存関数は無変更）
   - `compute_mcnemar_test`（226-261行）と本質的に同じ discordant-pair の χ²／p 値計算を
     `_mcnemar_from_correctness(correct_a: dict[str, bool], correct_b: dict[str, bool]) ->
     dict[str, float]` として切り出す（DRY，重複コード回避が目的の小さな抽出であり目的外の
     大規模リファクタリングではない）。`compute_mcnemar_test` はこのヘルパーを呼ぶよう変更。
   - 新規: `compute_domain_recall_mcnemar_test(results_a: list[dict], results_b: list[dict],
     domain: str) -> dict[str, float]`。`id` が一致する行のうち `domain in
     expected_domains` の行だけをサブセットし，正誤を `selected_domain == domain`
     （recall の定義そのもの）で定義して `_mcnemar_from_correctness` に渡す。id 集合の不一致は
     `compute_mcnemar_test` と同様に `ValueError`。
   - 新規: `compute_domain_precision_fisher_test(results_a: list[dict], results_b: list[dict],
     domain: str) -> dict[str, float]`。precision は分母（`selected_domain == domain` の行集合）
     自体が較正前後で変わる非対応データのため，2×2 分割表
     `[[tp_a, selected_a - tp_a], [tp_b, selected_b - tp_b]]`（`tp` = `selected_domain ==
     domain and domain in expected_domains`）を作り `scipy.stats.fisher_exact`
     （両側検定）で `p_value`・`odds_ratio` を返す。分母 0 件（そのドメインへの選択が
     一方の側で 0 件）の場合は `ValueError`（Wilson CI 同様サイレントに 0 除算しない）。
   - 新規: `apply_benjamini_hochberg(p_values: list[float], q: float = 0.05) ->
     list[bool]`。標準的な BH step-up 手順（p 値を昇順ソートし，最大の `i` で
     `p_(i) <= (i/m)*q` を満たすものを見つけ，それ以下の順位を有意とする）。入力順序を
     保った `bool` のリストを返す。空リストは空リストを返す。
   - `import` 追加: `from scipy.stats import fisher_exact`。
   - `pyproject.toml`: `dependencies`（6-14行付近）に `"scipy>=1.18"` を追加する。現状
     scipy は scikit-learn 経由の間接依存でしか入っておらず（`uv run python -c "import
     scipy"` は通るが `pyproject.toml` に宣言がない），`metrics.py` が直接 import する以上
     明示的な直接依存として宣言すべきである。`uv add scipy` を実行して `uv.lock` を更新する
     （既にインストール済みの 1.18.0 がそのまま解決される見込みで，大きな依存変更は
     発生しないはずだが，実装フェーズで `uv.lock` の diff を確認すること）。

4. `tests/test_metrics.py`
   - `compute_domain_recall_mcnemar_test`：小さなトイデータ（3〜4行，domain 該当行のみ）で
     discordant 件数・p 値が手計算と一致することを確認するテスト，および
     `compute_mcnemar_test` と同様の id 不一致 `ValueError` テストを追加する。
   - `compute_domain_precision_fisher_test`：2×2 のトイデータで `scipy.stats.fisher_exact`
     を直接呼んだ場合と同じ p 値になることを確認するテスト，および分母 0 件時の
     `ValueError` テストを追加する。
   - `apply_benjamini_hochberg`：教科書的な既知の例（例: p値 `[0.01, 0.02, 0.03, 0.04, 0.20]`，
     `q=0.05` で先頭 何件が有意になるか）で結果が一致することを確認するテスト，全て非有意な
     ケース，空リストのテストを追加する。
   - 既存の `test_compute_mcnemar_test_*` 系テストは，`_mcnemar_from_correctness` への
     抽出後も `compute_mcnemar_test` の外部インターフェースは変わらないため無変更で通る見込み
     （実装フェーズで実行して確認する）。

**評価手順**:

1. 新分類器の学習: `uv run python -m scripts.train_domain_classifier
   --train-data data/classifier_train.jsonl --embedding-model nomic-embed-text
   --ollama-host <live node> --output models/domain_classifier_isotonic.joblib`
   （Iter29 と同じくライブな ollama ノード 1 台への embedding のみ）。
2. 「較正前」データは Iter29 と同一の `results/20260731_162722/results.jsonl`
   （Iter28 実測，fallback 0/1600）をそのまま使う。**再実行しない**（Iter29・Iter30 を
   同じ較正前基準で揃えて比較可能にするため）。
3. 「較正後」データは `scripts/evaluate_classifier_calibration.py` で 1600 問を再 embedding し，
   `--classifier models/domain_classifier_isotonic.joblib --output
   results/iter30_calibrated_predictions.jsonl` として生成する。
4. `metrics.py:compute_ece(n_bins=10)` を較正前・較正後の両方に同一の bin 設定で適用し，
   ECE を比較する（Iter29 の較正前基準 0.19336 を流用し，再計算しない）。
5. top1_accuracy を較正前・較正後で算出し，新旧の正誤ペアで `compute_mcnemar_test`
   （全体，α=0.05）を行う（Iter29 の手順5と同一）。
6. **per-domain 非退行チェック（今回から運用変更，調査(Iter30) 申し送り2）**: 全10ドメイン
   について，(a) recall は `compute_domain_recall_mcnemar_test` （計10検定），(b) precision は
   `compute_domain_precision_fisher_test` （計10検定）を実施し，計20個の p 値を集めて
   `apply_benjamini_hochberg(p_values, q=0.05)` を一括適用する。adjusted 有意（BH 通過）かつ
   方向が悪化（較正後の点推定 < 較正前の点推定）である指標のみを「統計的に有意な退行」と
   判定する（有意だが改善方向のものは退行ではない）。全10ドメイン・20指標を**最初から**
   算出し，Iter29 のように事後で穴埋めしない。
7. **isotonic 特有の実装確認チェックリスト（調査(Iter30) 申し送り3，`probabilities`
   フィールドを使って算出）**: 較正後の 1600 行について，(a) `probabilities` の値のいずれかが
   厳密に `0.0` または `1.0` になっている行数，(b) 10 クラス全てが `0.1` に近い
   （`math.isclose(p, 0.1, abs_tol=1e-9)` 相当）uniform fallback 行数，(c) 選択ドメインの
   confidence と同一の値を持つ他ドメインが存在する行の割合（tie 率）。特に legal ドメインの
   行を優先して個別集計する。これらは判定基準ではなく必須報告項目。
8. 新旧 classifier の argmax 不一致件数（flip rate）を Iter29 と同じ定義で報告する（必須報告
   項目，判定基準ではない）。

**成功条件（d0003 X9．AND 条件）**:

1. ECE（手順2・4，較正前基準 0.19336 に対する較正後の値，`n_bins=10`）が **0.150 以下**
   であること。
2. top1_accuracy（手順5）が旧分類器（Iter28 実測 0.585）に対し McNemar 検定で有意に悪化
   していない（p>=0.05，または新側が改善方向）こと。Iter29 と同じく理論的仮定ではなく
   実測比較で判定する。
3. **per-domain 非退行（手順6，3段構成）**: 20指標（10ドメイン×precision/recall）の p 値へ
   BH 補正（q=0.05）を適用した結果，adjusted 有意かつ悪化方向の指標が **0 件**であること。
   （Iter29 で用いた「CI 下限の単純比較」は多重比較補正なしで 20 指標中 9 指標が偽陽性に
   なることが判明済みのため，今回はこの基準を使わない．CI そのものは参考情報として引き続き
   算出・報告する。）
4. 手順7のisotonic特有チェックリスト（0/1張り付き・uniform fallback・tie率）とflip rate
   （手順8）は，成功・失敗の判定基準ではなく必須報告項目として全件記録する。

**目標未達時の次点候補（次イテレーション向けメモ，今回の計画には含めない）**: 調査(Iter30)
申し送り4のとおり，isotonicがECE 0.150以下に届かない場合，`method='temperature'`
（sklearn>=1.8，本リポジトリの`scikit-learn==1.9.0`で利用可能，top1_accuracy不変が理論的に
保証される）を次点候補として検討する。ただしtemperatureは多クラス全体で単一のTを学習するため
ドメインごとの較正の柔軟性はsigmoid/isotonicより低い点は留保として記録しておく。

**人間判断が必要な論点**: 新規追加なし。Y2（`confidence_threshold`の二重責務分離，スキーマ
変更）着手前のユーザー確認は backlog B49・B50 の既存の申し送りのまま。較正済み分類器の本番
反映可否も，isotonicが成功条件（本計画の1-3すべて）を満たした場合に改めてその時点で判断する
（今回のイテレーションで本番アーティファクトを置き換える判断は行わない）。

---

### 実装 (Iter30)

計画どおり単一レバー（`classifier_calibration=isotonic`）のみを実装した．`config.yaml` は
変更していない（`git diff config.yaml` が空であることを確認済み）。

**変更ファイル**:

1. `scripts/train_domain_classifier.py`
   - `_CALIBRATION_METHOD = "sigmoid"` → `"isotonic"` に変更。`_CALIBRATION_CV = 5` は無変更。
   - `_CALIBRATION_METHOD` 直上のコメントを，isotonic を選ぶ理由（Iter29 の Platt が
     ECE 絶対閾値未達で partial 判定・config.yml の `classifier_calibration` レバーが
     isotonic を次点候補として登録済み・backlog B50）と，調査(Iter30) が確認した追加リスク
     （isotonic はノンパラメトリックで自由度が事実上サンプル数に等しく sigmoid より
     過学習しやすい／`out_of_bounds="clip"` により legal の held-out データが極端値に
     張り付きやすい）に更新。
   - モジュール冒頭 docstring の `Iter29, classifier_calibration=platt` を
     `Iter30, classifier_calibration=isotonic` に更新。
   - `train_classifier()` の docstring を `method="isotonic"` の説明に更新し，isotonic
     特有の注意点（tie・0/1 張り付き・uniform fallback，調査(Iter30) 分かったこと(3)）を
     追記。
   - 出力アーティファクト名（`--output` の既定値）は変更していない
     （`models/domain_classifier.joblib` のまま）。計画どおり，実験フェーズでの実行時に
     `--output models/domain_classifier_isotonic.joblib` を明示指定することで本番
     アーティファクトを上書きしない運用とする（CLI 引数のみで対応可能なため，スクリプト
     側の既定値変更は不要と判断）。
   - `tests/test_train_domain_classifier.py` は無変更で実行し，pass することを確認した
     （`method` を変えても `StratifiedKFold` の分割条件は変わらないため）。
2. `scripts/evaluate_classifier_calibration.py`
   - `predict_calibrated_rows()` が返す各行の辞書に `"probabilities"`
     フィールド（`{domain: float(p) for domain, p in zip(classes, probabilities)}`，10
     ドメイン全ての確率）を追加。既存フィールド（`id`／`expected_domains`／
     `selected_domain`／`confidence`）は無変更。
   - モジュール冒頭 docstring に，isotonic 特有のチェックリスト（0/1 張り付き・uniform
     fallback・tie 検出）に全クラスの確率ベクトルが必要なため `probabilities` を追加した，
     という理由を追記。CLI 引数は無変更。
3. `metrics.py`
   - `compute_mcnemar_test`（226-261行相当）から discordant-pair の χ²／p 値計算を
     `_mcnemar_from_correctness(correct_a: dict[str, bool], correct_b: dict[str, bool]) ->
     dict[str, float]` として切り出し，`compute_mcnemar_test` はこのヘルパーを呼ぶよう変更
     （外部インターフェースは無変更）。
   - 新規 `compute_domain_recall_mcnemar_test(results_a, results_b, domain) ->
     dict[str, float]`：`id` が一致する行のうち `domain in expected_domains` の行だけを
     サブセットし，`selected_domain == domain` を正誤として `_mcnemar_from_correctness`
     に渡す。id 集合の不一致は `ValueError`。
   - 新規 `compute_domain_precision_fisher_test(results_a, results_b, domain) ->
     dict[str, float]`：2×2 分割表 `[[tp_a, selected_a - tp_a], [tp_b, selected_b - tp_b]]`
     （`tp` = `selected_domain == domain and domain in expected_domains`）を作り
     `scipy.stats.fisher_exact`（両側）で `p_value`・`odds_ratio` を返す。片側の選択数が
     0 件の場合は `ValueError`。
   - 新規 `apply_benjamini_hochberg(p_values: list[float], q: float = 0.05) ->
     list[bool]`：標準的な BH step-up 手順。入力順序を保った `bool` のリストを返す。
     空リストは空リストを返す。
   - `import` 追加: `from scipy.stats import fisher_exact`。
   - `pyproject.toml` の `dependencies` に `"scipy>=1.18"` を追加し，`uv add "scipy>=1.18"`
     で `uv.lock` を更新した。`uv.lock` の diff を確認したところ，`scipy` パッケージの
     エントリ追加自体は想定どおり小さいが，`lora` extra 配下の nvidia/cuda 系パッケージの
     プラットフォームマーカーが再解決の副作用で一部変化していた（バージョン変更は一切なし，
     `win32`/`AMD64` 条件が一部エントリから外れる形の書き換えのみ）。`git stash` で
     `pyproject.toml` を元に戻した状態で `uv lock --check` を実行し，変更前の `uv.lock` が
     既に最新状態であったこと（＝この差分が scipy 追加以前からの潜在的なズレではなく，
     今回の relock で新たに解決された結果であること）を確認済み。`uv add` でも手動編集＋
     `uv lock` でも同一の差分になることを確認しており，`lora` extra は既定ではインストール
     されない（`uv sync --extra lora` 時のみ関与）ため，本プロジェクトの通常の依存関係
     解決には影響しない。
4. `tests/test_metrics.py`
   - `compute_domain_recall_mcnemar_test`：既存の
     `test_compute_mcnemar_test_matches_known_chi_square_critical_values` と同じ discordant
     カウント（29／15）を `domain="legal"` のサブセットに対して再現するトイデータ（加えて
     サブセット対象外のノイズ行2件が結果に影響しないことも確認），および id 不一致
     `ValueError` テストを追加。
   - `compute_domain_precision_fisher_test`：2×2 トイデータ（`[[6, 4], [2, 6]]`）で
     `scipy.stats.fisher_exact` を直接呼んだ場合と `odds_ratio`／`p_value` が一致することを
     確認するテスト，および分母 0 件（片側でドメインが一度も選択されない）時の `ValueError`
     テストを追加。
   - `apply_benjamini_hochberg`：教科書的な既知の例（p値 `[0.01, 0.02, 0.03, 0.04, 0.20]`，
     `q=0.05` で先頭4件が有意）のテスト，全て非有意なケース，空リストのケースを追加。
   - 既存の `test_compute_mcnemar_test_*` 系テストは無変更で実行し，pass することを確認した。

**テスト結果**: `uv run pytest -q` → 218 passed, 2 skipped（既存のスキップ2件は本変更と
無関係）。新規追加した8件のテストを含め全て pass。

**lint/format**: `uv run ruff check metrics.py scripts/train_domain_classifier.py
scripts/evaluate_classifier_calibration.py tests/test_metrics.py
tests/test_train_domain_classifier.py pyproject.toml` → All checks passed。
`uv run ruff format --check` は `metrics.py`／`tests/test_metrics.py` が未整形と報告されたが，
これは Iter30 の変更前から repository 全体で `ruff format` 規約に沿っていなかった既存差分
であることを `git stash` で変更前の状態に戻して確認済み（単一レバー原則に従い，本イテレーション
の変更範囲外として触っていない）。新規追加した `scripts/evaluate_classifier_calibration.py`
の1行のみ未整形だったため，その箇所だけ手動で1行に整形し直し，整形済みであることを再確認した。
リポジトリ全体の `ruff check .` に残る2件（`scripts/prepare_lora_training_data.py`）は
Iter29 から既知の，本変更と無関係な既存差分であり，単一レバー原則に従い今回も触っていない。

**config.yaml の確認**: `git diff --stat -- config.yaml` が空であることを確認し，一切
変更していないことを確認した。

**実験を開始してよい状態か**: はい。コード変更は完了し，型注釈・テスト・lint とも整合。
フェーズ4では，(1) `scripts/train_domain_classifier.py` で
`models/domain_classifier_isotonic.joblib` を1台のライブ ollama ノードへの embedding
呼び出しで新規生成（本番 `models/domain_classifier.joblib` は上書きしない），(2)
`scripts/evaluate_classifier_calibration.py` で 1600 問を再 embedding して較正後の予測
JSONL（`probabilities` フィールド付き）を生成，(3) `metrics.py` の既存関数群＋新規3関数で
較正前（`results/20260731_162722/results.jsonl`，再実行不要）と較正後を比較し，成功条件
1-4（ECE≤0.150・McNemar 非退行・per-domain 20指標への BH 補正非退行・isotonic 特有チェック
リストと flip rate の報告）を実測すればよい。

---

### 実験 (Iter30)

計画どおり実機 1600 問本走は行わず，Iter29 と同一の SSH ローカルポートフォワード
（`127.0.0.1:11435 -> wafl500:11434`，`ssh -fNT -L 11435:localhost:11434 wafl500`，
既存プロセスが起動済みで新規に張り直す必要はなかった。事前に `curl` で
`http://127.0.0.1:11435/api/tags` が疎通することを確認済み）経由の embedding 呼び出しのみで
較正前後の比較データを揃えた。

1. 新分類器の学習:
   ```
   uv run python -m scripts.train_domain_classifier \
     --train-data data/classifier_train.jsonl \
     --embedding-model nomic-embed-text \
     --ollama-host 127.0.0.1 --ollama-port 11435 \
     --output models/domain_classifier_isotonic.joblib
   ```
   標準出力: `[train_domain_classifier] wrote models/domain_classifier_isotonic.joblib
   (n_samples=1427, classes=[...10ドメイン...])`。実行時間 126.51 秒（実測，Iter29 の Platt
   124.09 秒とほぼ同水準）。`models/domain_classifier_isotonic.joblib` を新規生成し，
   本番 `models/domain_classifier.joblib` のタイムスタンプ（Jul 27 16:08）が今回の実行後も
   変化していないこと（＝上書きされていないこと）をファイルシステム上で確認した。
2. 較正後データ生成:
   ```
   uv run python -m scripts.evaluate_classifier_calibration \
     --dataset data/dataset.jsonl \
     --classifier models/domain_classifier_isotonic.joblib \
     --embedding-model nomic-embed-text \
     --ollama-host 127.0.0.1 --ollama-port 11435 \
     --output results/iter30_calibrated_predictions.jsonl
   ```
   標準出力: `[evaluate_classifier_calibration] wrote 1600 rows
   (classifier=models/domain_classifier_isotonic.joblib)`。実行時間 136.74 秒。出力
   JSONL は計画どおり `probabilities` フィールド（10 ドメイン全ての確率）付きで 1600 行生成された。
3. 較正前データは計画どおり `results/20260731_162722/results.jsonl`（Iter28 実測，fallback
   0/1600）を再実行せずそのまま使用。新旧2ファイルの `id` 集合が完全一致することを確認済み
   （`{r["id"] for r in before} == {r["id"] for r in after}` が `True`）。

**異常の有無**: なし。両スクリプトとも例外・タイムアウト・リトライなく正常終了した。実機呼び出し
は wafl500（192.168.15.100:11434）への embedding のみ（`nomic-embed-text`，計 3027 回：
1427+1600），LLM 生成・probe・dispatch は一切発生していない。

---

### 分析(実行) (Iter30)

`metrics.py` の既存関数（`compute_ece`／`compute_top1_accuracy`／`compute_mcnemar_test`／
`compute_precision_recall_per_domain`）と新規3関数（`compute_domain_recall_mcnemar_test`／
`compute_domain_precision_fisher_test`／`apply_benjamini_hochberg`）を呼ぶ一時スクリプトで
較正前（`results/20260731_162722/results.jsonl`）と較正後（`results/iter30_calibrated_predictions.jsonl`，
各 1600 行）を比較した（判定はここでは行わず，数値のみを機械的に集計する）。

**手順4: ECE（`n_bins=10` で統一）**

- 較正前: **0.193357556998477**（`state.json` の `e29_results.ece_before` と同一値，再計算不要の
  ところ実測でも一致することを確認）
- 較正後: **0.1214241251658703**
- 改善幅: 0.071933（較正前→較正後で減少，改善方向）
- 0.150 との比較: 較正後 0.1214 < 0.150

**手順5: top1_accuracy（1600問，`expected_domains` との一致率）**

- 較正前: 0.585000（Iter28 実測と同一値）
- 較正後: 0.593750
- 差分: +0.008750（較正後が高い）

**手順5: McNemar 検定（全体，対応のある2条件比較，較正前=A・較正後=B，連続性補正あり）**

- discordant_a_only（較正前のみ正解）: 72
- discordant_b_only（較正後のみ正解）: 86
- discordant_pairs（合計）: 158
- chi2_statistic: 1.0696202531645569
- p_value: **0.30103123736220994**（α=0.05 で有意差なし。較正後が正解に転じた行(86)が誤りに
  転じた行(72)を上回り，方向としては改善寄り）

**手順8: flip rate（argmax が変わった行の割合，`id` で対応付け，Iter29 と同じ定義）**

- **229/1600 = 0.143125**（14.3125%）。Iter29（Platt，11.0%）より高い（isotonic の方が
  柔軟な分だけ argmax の入れ替わりが多いという調査(Iter30) の事前予想と整合）。

**手順6: per-domain 非退行チェック（10ドメイン×recall/precision＝20指標，BH補正 q=0.05）**

全20指標の点推定（較正前→較正後）と個別検定のp値，BH補正後の有意フラグ：

| domain | metric | before | after | p_value | BH有意 | 方向 |
|---|---|---|---|---|---|---|
| business_economics | recall | 0.5179 | 0.5357 | 0.546494 | 否 | 改善 |
| business_economics | precision | 0.4328 | 0.4688 | 0.479833 | 否 | 改善 |
| computer_science | recall | 0.5417 | 0.5595 | 0.627626 | 否 | 改善 |
| computer_science | precision | 0.5987 | 0.5529 | 0.430737 | 否 | 悪化 |
| education | recall | 0.4059 | 0.5000 | 0.000796 | **有** | 改善 |
| education | precision | 0.4631 | 0.4315 | 0.585896 | 否 | 悪化 |
| general | recall | 0.5488 | 0.5427 | 1.000000 | 否 | 悪化 |
| general | precision | 0.6522 | 0.6899 | 0.518329 | 否 | 改善 |
| history_culture | recall | 0.6667 | 0.7024 | 0.211300 | 否 | 改善 |
| history_culture | precision | 0.7320 | 0.6705 | 0.231070 | 否 | 悪化 |
| legal | recall | 0.5833 | 0.5889 | 1.000000 | 否 | 改善 |
| legal | precision | 0.7500 | 0.7852 | 0.568393 | 否 | 改善 |
| mathematics | recall | 0.6190 | 0.6786 | 0.009375 | 否 | 改善 |
| mathematics | precision | 0.7075 | 0.6867 | 0.713168 | 否 | 悪化 |
| medical | recall | 0.4831 | 0.3820 | **0.000144** | **有** | **悪化** |
| medical | precision | 0.4725 | 0.5231 | 0.421841 | 否 | 改善 |
| natural_science | recall | 0.5655 | 0.5833 | 0.662521 | 否 | 改善 |
| natural_science | precision | 0.5135 | 0.5475 | 0.530141 | 否 | 改善 |
| social_science | recall | 0.5774 | 0.5238 | 0.052345 | 否 | 悪化 |
| social_science | precision | 0.6340 | 0.6984 | 0.308685 | 否 | 改善 |

BH（q=0.05）通過（adjusted 有意）は20指標中2件: `education_recall`（p=0.000796，改善方向）・
`medical_recall`（p=0.000144，**悪化方向**）。`medical_recall` の内訳:
discordant_a_only=19（較正前のみ正解）・discordant_b_only=1（較正後のみ正解）・
discordant_pairs=20・chi2=14.45。BH 補正後も有意かつ悪化方向の指標は **1件**（`medical_recall`）。

**手順7: isotonic 特有の実装確認チェックリスト（`probabilities` フィールドを使用，1600行対象）**

- (a) 確率のいずれかが厳密に `0.0` または `1.0` になっている行数: **1311/1600**（うち厳密に
  `1.0` を含む行は **0 件**，厳密に `0.0` を含む値の総数は全行合計で **2123 個**）。
- (b) 10クラス全てが `0.1` に近い（`math.isclose(p, 0.1, abs_tol=1e-9)`）uniform fallback 行数:
  **0/1600**。
- (c) 選択ドメインの confidence と同一の値を持つ他ドメインが存在する行の割合（tie率，
  厳密な浮動小数点一致で判定）: **0/1600（0.0000%）**。

legal ドメインの個別集計（優先報告）:
- `legal` が `expected_domains` に含まれる行（180行）のうち，確率に厳密な `0.0`/`1.0` を含む
  行数: **158/180**。
- `legal` が `selected_domain` の行（135行）のうち，同条件: **121/135**。
- legal の uniform fallback 行数: **0/180**。tie 行数: **0/180（0.0000%）**。

**使用データ**:

- 訓練データ: `data/classifier_train.jsonl`（1427件，legal 77件・他9ドメイン各150件，Iter29と同一）
- 評価データセット（再embedding対象）: `data/dataset.jsonl`（1600件）
- 較正前の実行結果: `results/20260731_162722/results.jsonl`（Iter28実測，1600行，再実行なし）
- 較正後の実行結果（新規生成）: `results/iter30_calibrated_predictions.jsonl`（1600行，
  `probabilities`フィールド付き）
- 新規モデルアーティファクト: `models/domain_classifier_isotonic.joblib`（本番
  `models/domain_classifier.joblib`は無変更のまま，タイムスタンプで確認済み）

**実行時間・実機呼び出しの有無**:

- `train_domain_classifier.py`: 126.51秒（1427回のembedding呼び出し）
- `evaluate_classifier_calibration.py`: 136.74秒（1600回のembedding呼び出し）
- 実機呼び出しはwafl500（192.168.15.100:11434）へのembeddingのみ（`nomic-embed-text`），
  計3027回。LLM生成・probe・dispatchは一切発生していない。
- 接続経路はIter29と同一の既存SSHローカルポートフォワード（`127.0.0.1:11435` ←
  `wafl500:11434`）をそのまま流用。新規に張り直す必要はなく，実行中のログ・エラーに異常
  なし（例外・タイムアウト・リトライなし，両スクリプトとも正常終了メッセージを出力）。

**state.json更新**: `status: waiting_experiment`（開始時，`experiment_dir:
results/iter30_calibrated_predictions.jsonl`・`experiment_deadline`設定）→`running`
（完了時，`experiment_dir`/`experiment_deadline`を`null`に戻した）。`e30_results`への数値
記録・`judgment`確定はフェーズ5b（rc-analyst）に委ねる（本フェーズでは数値の良否判定は行わない）。

---

### 分析(解釈) (Iter30)

**成功条件（d0003 X9，AND条件，計画(Iter30)節）との照合**

1. **ECE ≤ 0.150**: 成立。較正後 0.121424 は較正前 0.193358 から −7.19pt（相対37.2%減）であり，
   Iter29（Platt，0.16751）より 4.6pt 深く改善し，目標にも 2.86pt の余裕をもって到達している。
   ルーティングは決定論的（config.yml success_criteria (5)）であり，同一 1600 問・同一
   embedding モデルに対し分類器のみを変えた比較のため，この差分はノイズではなく較正手法の
   変更そのものが生んだ実測値と判断してよい（Iter29 と同じ根拠）。
2. **top1_accuracy 非退行**: 成立。McNemar p=0.301031（α=0.05 で有意差なし）であり，
   discordant_b_only（較正後のみ正解，86）が discordant_a_only（較正前のみ正解，72）を
   上回っているため方向としては改善寄りである。Iter29（p=0.139，b_only=67>a_only=50）と
   同種の非退行パターンが再現している。
3. **per-domain 20指標のBH補正後・悪化方向の有意指標0件**: **不成立**。BH（q=0.05）通過は
   `education_recall`（p=0.000796，改善方向）と`medical_recall`（p=0.000144，悪化方向，
   0.4831→0.3820）の2件で，悪化方向で通過したのは`medical_recall`の1件。discordant内訳は
   a_only（較正前のみ正解）=19・b_only（較正後のみ正解）=1・discordant_pairs=20・chi2=14.45
   であり，19:1という非対称性は補正後もなお際立って大きい。
4. isotonic特有チェックリスト（0/1張り付き1311/1600・uniform fallback 0件・tie率0%）と
   flip rate（229/1600=14.3125%，Plattの11.0%より高い）は報告事項として確認した（詳細は下記）。

**medical_recall悪化（BH通過）の解釈 — Iter28・Iter29との異同**

まず事実確認として`data/classifier_train.jsonl`を実際に確認した結果，**medicalの訓練データは
150件であり，legal（77件）のような少数派ドメインではなく，他8ドメインと同数の多数派ドメインで
ある**（`business_economics`〜`social_science`まで全て150件，`legal`のみ77件）。これは
Iter29の申し送り（「computer_science/mathematicsは150件の多数派ドメインなのに偽陽性で
引っかかった」）が示唆したとおり，訓練データ量の多寡だけではmedical_recallの悪化を説明できない
ことを裏付ける事実である。

次に，Iter29までとの決定的な違いは**検定の厳格さ**にある。Iter29の per-domain 非退行チェックは
「較正前後のCI下限の単純比較」という多重比較補正なしの基準であり，事後の追加分析（B50）で
20指標中9指標が該当したものの全て区間重複で統計的に非有意な偽陽性だったと判明した。今回は
Iter29の教訓を踏まえ，(a) recallはドメイン別McNemar検定，(b) precisionは2標本Fisher正確検定，
(c) 計20個のp値へBH法を**最初から**適用するという，より厳格な手順で臨んだ。その結果として
残った`medical_recall`1件は，Iter29の9件のような「緩い基準でしか引っかからない偽陽性」とは
性質が異なり，**多重比較を補正してもなお統計的に有意な，再現性のある効果**である。BH法は
20検定という規模で偶然生じる誤検出（FDR）を5%以下に抑えるよう設計されており，それでも
生き残った1件は，Iter29の legal recall 低下（追加分析で相対化された）よりも判定上の重みが
大きいと考えるべきである。

precisionは同時に0.4725→0.5231へ改善しており，表面上はIter28のgeneralドメイン
（recall低下・precision大幅改善が同一212行内で表裏一体）と類似する。しかし規模を比較すると
性質が異なる。Iter28のgeneral precisionは0.3134→0.6522（+33.9pt）という recall 低下を
大きく上回る改善であり，かつ「fallbackの送り先が常にgeneralだった」というレバー変更に
数学的に内在する構造（fallback廃止で流入経路が変わるのは必然）が機序として明確だった。
今回のmedicalはprecision改善が+5.06pt（0.4725→0.5231）にとどまり，recall悪化の−10.11pt
（0.4831→0.3820）の半分程度に過ぎない。かつdiscordantの非対称性（a_only=19 : b_only=1）は
Iter28のgeneral（fallback対象212行内の再配分という機構が既知）のような「レバー自体が
生む必然的な流入経路変化」では説明できず，isotonic較正曲線がmedicalクラス固有にどう
振る舞ったかを調べる必要がある。

**追加検証（数値再計算，本フェーズで実施）**: `results/20260731_162722/results.jsonl`
（較正前）と`results/iter30_calibrated_predictions.jsonl`（較正後，`probabilities`
フィールド付き）から，medical_recallが悪化した19行（discordant a_only）を個別に確認した。

- 19行のうち，較正後の`probabilities`でmedicalクラスの値が厳密に`0.0`になっている行は
  **0件**であり，0/1張り付き（isotonic特有チェックリストの(a)）が直接の原因ではない。
  むしろ19行の多くは較正後もmedicalが2位相当の確率（0.21〜0.39）を保持しており，僅差
  （margin 0.003〜0.13）で他ドメイン（`computer_science`5件・`natural_science`4件・
  `education`3件・`history_culture`3件・`business_economics`2件・`mathematics`1件・
  `social_science`0件他）に argmax を奪われている。
- 19行中，較正前の`confidence`（medicalの確信度）が0.87〜0.98という高い値だった行が3件
  含まれており，較正前は明確にmedicalが最有力だったにもかかわらず，較正後は0.35〜0.39まで
  値が圧縮されて argmax を失っている。
- **1600行全体でmedicalクラスの較正後確率の最大値は0.7062であり，他9ドメインの最大値
  （0.7496〜0.8795）を全て下回る**。medicalが較正後に到達しうる確信度の「天井」自体が，
  他ドメインより体系的に低く抑えられている（`business_economics`最大0.8238，
  `history_culture`最大0.8795 など）。selectedとして選ばれた回数も較正前182件→較正後130件
  （−28.6%）へ減少しており，このドメインだけisotonic較正曲線がクラス全体で系統的に
  スコアを下方へ圧縮している疑いが強い。
- 一方，選択された行の`confidence`自体（ECEの算出対象）に厳密な0.0/1.0は1件もなく
  （全1600行で確認済み），isotonic特有チェックリストの「uniform fallback」も0件であるため，
  ECE 0.121424の改善はmedicalの0/1張り付きのような病理によって水増しされたものではない。

以上から，medical_recallの悪化は「isotonicの0/1張り付き」や「legalのような小標本held-out
較正の不安定性」という調査(Iter30)が事前に警戒していた2つの機序のいずれでもなく，
**medicalクラスの isotonic 較正曲線がcv=5較正foldにおいて系統的にスコアを圧縮し，
他ドメインとの僅差の argmax 競争で構造的に不利になる**という，訓練データ量では説明できない
第3の機序である可能性が高い。この機序は事前の投資フェーズでは想定されておらず，**次回
isotonicを継続検討する場合は，legalだけでなくmedicalのように多数派ドメインでも同種の
較正曲線圧縮が起こりうることを踏まえ，cv=3等の感度分析やドメイン単位の較正曲線可視化を
対象ドメインを限定せず行うべき**という新たな示唆を得た。

なお，legalドメイン（Iter29でrecall低下が唯一の懸念だった小標本ドメイン）は今回
recall 0.5833→0.5889へ**改善**しており，isotonicが小標本ドメインで一律に悪化を招くという
調査(Iter30)の事前予想（sigmoidより過学習しやすい，legalが最も影響を受けやすい）はむしろ
反証された。isotonicの実際のリスクは事前に警戒していたlegalではなく，多数派ドメインの
medicalという想定外の箇所に現れており，この点は仮説と実測の不一致として明示しておく。

**isotonic特有チェックリスト（0/1張り付き1311/1600＝82%）の判定への反映**

sklearn公式ドキュメントが警告する「isotonicはties/0-1張り付きを生みやすい」という調査(Iter30)
の申し送りは，非選択クラスの確率に関しては実測でも裏付けられた（1311/1600行で少なくとも
1クラスが厳密0.0，legalは158/180行と特に高率）。ただし上記の追加検証で確認したとおり，
**この0/1張り付きは主に非選択（劣勢）クラスに生じており，ECEの算出対象である選択ドメインの
confidence自体には1件も及んでいない**（厳密0.0/1.0の選択行は0/1600）。したがって，
「ECEの見かけ上の改善が個々の予測の信頼性を代償にしている」という懸念は，少なくとも
ECEの数値そのものについては支持されない。一方で，非選択クラスの0/1張り付きが82%という
高率で生じている事実自体は，isotonic較正曲線の区分定数性・ノンパラメトリックな自由度の高さ
（調査(Iter30)分かったこと(1)）を裏付ける実装上の懸念として記録に値し，medical_recall悪化の
根本原因（較正曲線の系統的圧縮）と同根の現象（held-out較正データが少ない状態でのisotonic
回帰の不安定な区分定数フィット）である可能性が高い。判定上は「ECEの数値を歪める」形では
現れていないが，「特定ドメインの較正曲線が予測不能に歪みうる」という構造的リスクの実例として
medical_recall悪化の解釈に反映させる。

**Iter20（E3）precedentに関する留保**: config.ymlの申し送りが参照する「Iter20 partial運用実績」
は，本journal内の訂正1（環境修復セクション）で，Iter20当時の判定（「効果あり」）自体が
Iter17（supervised_classifier導入）との交絡により事後的に取り下げられ，D1（判定保留）へ
再分類されている経緯がある。したがって「主基準改善・副基準悪化ならpartial」という運用実績の
参照先としては，交絡のないIter29（同一AND条件構造・同一レバー系列）の方がIter30との対称性が
直接的であり，本判定はIter29を主たる比較対象とし，Iter20は参考情報にとどめる。

**総合判断（rc-analyst 提案，確定は rc-reflector）: partial（部分的採用）**

根拠:

1. 成功条件1・2は明確に成立し，特にECEはIter29のPlattを大きく上回る改善で目標に十分な余裕を
   もって到達している。この点はisotonicへの切り替えが「ECE改善」という当初目的に対し
   Platt以上に有効だったことを裏付ける。
2. 成功条件3は字義通り不成立である。BH補正という，Iter29の教訓を踏まえて最初から導入した
   厳格な多重比較補正の下でもなお生き残った`medical_recall`の悪化は，Iter29のlegal recall
   低下（事後分析で多重比較アーティファクトと判明）と同列には扱えない。訓練データ量では
   説明がつかず（medicalは150件の多数派），かつ0/1張り付きという既知のisotonic病理でも
   直接説明できず（19行中0件），較正曲線のクラス固有の系統的圧縮という，事前に想定していな
   かった機序で生じている。discordantの非対称性（19:1）とprecision改善幅（+5.06pt）が
   recall悪化幅（−10.11pt）の半分程度に留まることを踏まえると，Iter28のgeneralドメインの
   ような「レバーに内在する必然的トレードオフ」として判定を覆さない扱いにするのは根拠が
   弱い。
3. 以上を総合すると，「ECE目標達成」という主目的は明確に成立し，「per-domain非退行」という
   副次条件は統計的に確認された1件の悪化により不成立という，Iter29と同型（AND条件の一部が
   未達）だが**逆方向**の未達パターンである。Iter29はECE（主目的側）が未達でtop1・
   per-domain（当時は非有意）が成立していたのに対し，今回はECE・top1（主目的側）が成立し
   per-domain（副次条件）が１件のみ有意に未達という非対称な関係にある。いずれの場合も
   「AND条件の一部未達」を理由に，明確な改善方向にある指標の価値を無視して即rejectedとする
   のは実態を捉えず，かつ未解決の懸念（medical_recall）を残したまま本番へ即時反映する
   adoptedも時期尚早である。**partial（部分的採用）を提案する**。

**本番反映（`models/domain_classifier.joblib`の置き換え）についての見解**

**現時点では見送りを推奨する**（最終決定はrc-reflectorとユーザー確認事項）。判断基準:

- 成功条件のAND条件が字義通り未成立（医療ドメインrecallの統計的に有意な悪化）である以上，
  「採用して本番へ反映する」ための閾値をこの一回の実験だけでは満たしていない。
- 単一レバー原則・可逆性の観点では，本番アーティファクトを据え置く（`models/domain_classifier.joblib`
  は変更しない）方が取り消しコストが低い可逆な選択である。今回のisotonic版は
  `models/domain_classifier_isotonic.joblib`として別名生成済みであり，本番を上書きしていない。
- medical_recallの悪化は，Iter29のlegal recallのように「訓練データ拡充（Y5）で解消しうる」
  という見立てが立ちにくい（medicalは既に150件の多数派ドメインであるため）。原因はisotonic
  較正曲線のクラス固有の圧縮という，追加データではなく較正手法・パラメータ側の対処
  （例: `cv=3`感度分析でmedicalの較正foldサンプル数を増やす，または調査(Iter30)申し送り4の
  `method='temperature'`で全クラス共通の単一スカラー変換に切り替えargmax不変を理論的に
  保証する）が必要と考えられる。
- 一方で，ECE目標達成というY4の主目的自体は今回明確に成立しており，isotonicという手法選択
  そのものを棄却する根拠はない。次イテレーションでmedical_recall悪化の原因を狭く切り分ける
  追加検証（`cv=3`感度分析，またはmethod='temperature'との比較）を行い，その結果を踏まえて
  改めて本番反映を判断することを推奨する。

**確信度と追加反復の要否**: 判定の確信度は中程度以上と考える。ECE・top1_accuracyの2条件は
実測・検定とも明確であり追加反復は不要。medical_recallの悪化はBH補正済みで統計的には
確定的（p=0.000144）だが，**その原因（較正曲線のクラス固有圧縮）を裏付ける機序面の追加検証
（cv=3感度分析，較正曲線そのものの可視化）が次回に要る**という点は明記しておく。1回の本走
（n=1）に基づく判定である点はIter28・Iter29と同じ制約であり，ルーティングが決定論的である
以上，再実行によって数値自体が変わることはない。

---

### 考察 (Iter30)

**単一レバーの判定: 部分的採用（partial）を確定**．rc-analyst の「分析(解釈)」節の総合判定
（partial）をそのまま確定させる（覆さない）．判断基準は3点．

1. `classifier_calibration` レバーの成功条件（d0003 X9，計画(Iter30)節）は「ECE≤0.150 **かつ**
   top1_accuracy 非退行 **かつ** per-domain 20指標の BH 補正後の悪化方向有意指標 0 件」の
   AND 条件である．条件1（ECE 0.121424，目標に2.86pt の余裕）・条件2（McNemar p=0.301031，
   方向は改善寄り）は明確に成立するが，条件3は `medical_recall`（p=0.000144，BH 補正後も有意，
   0.4831→0.3820）が1件残っており字義通り不成立である．3条件AND のうち1条件が不成立である以上，
   無条件の adopted は成立しない．
2. 一方，`medical_recall` 以外の19指標はBH補正を通過しておらず，かつECE・top1_accuracyという
   主目的側の2条件は今回のイテレーションの本来の狙い（Iter29のPlattがECE絶対閾値未達だったため
   isotonicで追試する）に対し明確に達成している．「per-domain 1件の統計的に有意な悪化」のみを
   理由に，2条件の明確な達成を無視して rejected とするのは実態を捉えない．
3. rc-analyst が指摘するとおり，今回の `medical_recall` 悪化は Iter29 の legal recall 低下
   （事後の全ドメイン拡張分析で多重比較アーティファクトと判明，backlog B50）とは性質が異なる．
   Iter30 では調査(Iter30) の申し送りに従い，計画段階から BH 補正・ドメイン別 McNemar／Fisher
   検定を組み込んだ厳格な手順で臨んでおり，その手順を通過してなお残った1件は，Iter29 のような
   「緩い基準でしか引っかからない偽陽性」とは重みが異なる．Iter28（E1，fallback廃止，
   backlog B49）の `general` ドメイン recall 低下がレバーに内在する構造的トレードオフ
   （fallback の送り先が常に general という機構）として明確に説明できたのに対し，今回の
   `medical` は precision 改善（+5.06pt）が recall 悪化（−10.11pt）の半分程度にとどまり，
   `discordant` の非対称性（19:1）も「レバーに内在する必然的再配分」では説明できない．
   したがって Iter28 のような「判定を覆さない扱い」の類推は成り立たず，条件3の不成立を
   額面どおり受け止めて partial とすることが妥当である．

以上，rc-analyst の提案どおり **partial（部分的採用）** で確定する．

**本番反映の判断: 見送り（`models/domain_classifier.joblib` は isotonic 版へ置き換えない）**．
rc-analyst の見解をそのまま採用する．

- 成功条件のAND条件が字義通り未成立（`medical_recall` の統計的に有意な悪化）である以上，
  本番へ反映するための閾値をこの一回の実験だけでは満たしていない．
- 単一レバー原則・可逆性の観点では，本番アーティファクトを据え置く方が取り消しコストの低い
  可逆な選択である．今回のisotonic版は `models/domain_classifier_isotonic.joblib` として
  別名生成済みで，本番（`models/domain_classifier.joblib`，タイムスタンプ Jul 27 16:08 のまま
  変化なしを確認済み）を上書きしていない．
- `medical_recall` の悪化は，Iter29 の legal recall 低下と異なり「訓練データ拡充（Y5）で
  解消しうる」という見立てが立ちにくい（medical は既に150件の多数派ドメイン）．原因は
  訓練データ量ではなく較正手法・パラメータ側にあると考えられ，次回以降の追加検証で切り分ける
  べき問題として残す．

**得られた学び（次回以降に活きる非自明な点）**:

1. **isotonic の実際のリスクは，事前に警戒していた小標本ドメイン（legal）ではなく，多数派
   ドメイン（medical）に現れた**．調査(Iter30) の事前予想（「≪1000件で過学習しやすい」＝
   held-out データが最少の legal が最も影響を受けるはず）は，実測で明確に反証された（legal
   recall はむしろ改善 0.5833→0.5889）．訓練データ量という一次元の指標だけでは isotonic の
   ドメイン別リスクを予測できないことが，Iter29（B50，computer_science・mathematics という
   150件ドメインも偽陽性で該当）に続き2イテレーション連続で確認された．**「小標本ドメインが
   最も脆弱」という直感的な仮説は，較正手法の非退行リスクを評価する際の判断材料として単独では
   信頼できない**．次回以降，較正関連のレバーで事前リスクを予測する際は，訓練データ量だけでなく
   （分析(解釈)で行ったような）較正曲線そのものの形状・到達可能な確信度の天井を確認する必要が
   ある．
2. **BH補正という多重比較への厳格な対処は，Iter29の教訓（9件の偽陽性）を実際に解消した**．
   今回は20指標中2件のみがBH通過（悪化方向1件・改善方向1件）で，Iter29の「20指標中9指標が
   該当（うち1件のみ有意）」という状況から大きく改善した．計画段階から検定手順を組み込む
   （事後の穴埋めをしない）運用が機能したことを確認できた．この運用は今後の per-domain
   非退行チェックの標準手順として定着させてよい．
3. **isotonic の 0/1 張り付き（非選択クラスで82%の行に発生）は，ECE の数値そのものを歪めては
   いなかった**（選択ドメインの confidence に厳密な0/1は0件）が，`medical_recall` 悪化の
   根本原因（較正曲線のクラス固有の系統的圧縮）と同根の現象である可能性が高いと分析(解釈)で
   整理された．isotonic のノンパラメトリックな区分定数フィットが，held-out データの少なさと
   組み合わさると，どのドメインが影響を受けるか事前に予測しにくい形で歪みうるという構造的
   リスクを実証したことは，`method='temperature'`（クラスごとの個別較正器を持たず単一スカラー
   のみで変換するため，この種のクラス固有の歪みが構造的に発生しない）を次に検証する強い動機に
   なる．

**次に振る単一レバーの選定: `classifier_calibration=temperature`**

判断基準（`cv=3` 感度分析 と `method='temperature'` のいずれを優先するか）:

- **`cv=3` 感度分析は今回の `medical_recall` 悪化の根本原因に届きにくいと判断した**．
  分析(解釈)で確認したとおり，`medical` は訓練150件の多数派ドメインであり，`cv=5` でも
  `cv=3` でも1foldあたりの較正サンプル数はおよそ30件→50件程度の違いにとどまり，
  sklearn公式が目安とする「greater than ~1000」からは`cv`を3に変えても依然として大きく
  下回ったままである．かつ，19行の悪化事例のうち0/1張り付きが直接の原因だった行は0件で，
  「較正曲線がクラス全体で系統的にスコアを圧縮する」という機序（分析(解釈)节）は fold
  サンプル数の微調整では解消しない構造的な問題である可能性が高い．`cv=3` は同一手法
  （isotonic の OvR 個別較正）内のハイパラ変更にすぎず，今回発見した「OvR 較正がクラス固有に
  予測不能な歪みを生みうる」という根本の懸念には対処しない．
- **`method='temperature'` は，今回発見した根本原因に構造的に対処する**．調査(Iter30) が
  確認したとおり，temperature scaling はクラスごとに個別の較正器を fit せず，単一の
  `_TemperatureScaling` インスタンスのみでロジット全体を単一スカラー T で割る変換であり，
  argmax（top1_accuracy）が理論的に不変であることが sklearn 公式に保証されている．
  これは isotonic／platt が抱える OvR 方式由来の全リスク（クラス固有の曲線歪み・tie・0/1
  張り付き）を構造的に排除する代替であり，今回 `medical` で顕在化した「事前に予測できない
  ドメイン固有の較正曲線圧縮」という新たな懸念に直接応える．
- 留保（調査(Iter30) 申し送り4，分析(解釈) で既出）: temperature は多クラス全体で単一の T
  しか学習しないため，isotonic（0.121424）は元より Platt（0.16751）と比べても較正の柔軟性は
  低く，ECE 改善幅がより小さい可能性がある．Platt でさえ ECE 絶対閾値（0.150）に届かなかった
  経緯があるため，temperature がECE条件を満たせない可能性は相応にある．しかし，それ自体が
  次回イテレーションで検証すべき有益な情報である．仮に temperature が ECE 目標未達であれば，
  「per-domain 非退行のためには OvR 方式の柔軟性を犠牲にできない」という新しい知見が得られ，
  isotonic の運用（例: medical のみ較正を無効化する，較正曲線を平滑化する等）を再検討する
  材料になる．
- **可逆性・独立性**: `classifier_calibration` は既に config.yml の levers に登録済みだが，
  値は `[platt, isotonic]` のみで `temperature` は未登録のため，本フェーズで
  `values: [platt, isotonic, temperature]` へ末尾追記する（可逆な自動判断，スキーマ変更では
  なく既存レバーへの値追加）．`cv`（既定5）・`ensemble`（既定True）は temperature スケーリング
  自体には適用されない sklearn の実装（分析(解釈)出典の `_fit_calibrator` 参照）だが，同じ
  `CalibratedClassifierCV` API 経由で呼ぶため，実装フェーズで挙動を確認すること．

**iteration_name（Iter31）**: 「分類器較正のtemperature scaling方式によるargmax不変性の実証と
ECE目標到達可否の検証」

**要人間判断として残す論点（新規追加なし）**: Y2（`confidence_threshold` の二重責務分離，
スキーマ変更）の着手前ユーザー確認は backlog B49・B50 の既存の申し送りのまま．fallback
設計思想の論文上の位置付け（backlog B48）も未解決のまま据え置く．較正済み分類器の本番反映
可否も，今回は「見送り」という可逆な既定選択を自律判断で行ったのみで，将来いずれかの較正手法が
成功条件を完全に満たした場合の本番反映という判断（本番運用中のルーティング挙動を変える）自体は，
改めてその時点で検討する．

---

## Iteration 29: 分類器の較正（CalibratedClassifierCV）によるECE改善とルーティング非退行の検証

### 計画 (Iter29)

**仮説**: `scripts/train_domain_classifier.py:train_classifier()` が返す
`LogisticRegression` を `sklearn.calibration.CalibratedClassifierCV`（`method="sigmoid"`＝Platt，
`cv=5`，`ensemble=True`，いずれも既定値）でラップして較正すると，ECE が改善し（目標
0.150 以下），かつ 1600 問評価セットでの top1_accuracy（`selected_domain` と `expected_domains`
の一致率）が有意に悪化しない．

**単一レバー**: `classifier_calibration`（`.claude/research/config.yml` のレバー名，
150-170行）．今回試す値は `values: [platt, isotonic]` のうち **`platt` のみ**．調査(Iter29)の
結論どおり，isotonic は 1427 件・legal 77 件という規模では sklearn 公式が「≪1000 件で過学習」と
明言する水準を大きく下回るため，第一候補である Platt 単独をこのイテレーションで検証し，
isotonic は Platt が成功条件を満たせなかった場合のみ次イテレーションで別途検証する
（同一イテレーションに混ぜると単一レバー原則が崩れる）．

**固定する構成（Iter28 で確定した最良構成をすべて維持，`config.yaml` は一切変更しない）**:
`routing_method=supervised_classifier`，`confidence_threshold=0.0`・`dispatch_top_k=1`・
`aggregation_method=max_confidence`（Iter28 adopted の fallback 廃止構成），
`confidence_signal_method=self_report`，`confidence_elicitation=top_k_with_probs`，
`expert_model=expert-mesh-{domain}-lora`（domain_count=10），`light_model=qwen3.5:4b-q4_K_M`，
`embedding_model=nomic-embed-text`，評価データセットは Iter25 以降固定の 1600 問
（`data/dataset.jsonl`）．**今回変更するのはモデルアーティファクト（`models/*.joblib`）を
生成する学習方法のみであり，`config.yaml` のキーは 1 つも変えない．**

**変更ファイル・行**:

1. `scripts/train_domain_classifier.py`
   - import 追加: `from sklearn.calibration import CalibratedClassifierCV`（既存の
     `from sklearn.linear_model import LogisticRegression` は base estimator 用に残す）。
   - `train_classifier()`（62-75行）を変更: `LogisticRegression(max_iter=_MAX_ITER,
     class_weight="balanced")` を base estimator とし，
     `CalibratedClassifierCV(base, method="sigmoid", cv=cv, ensemble=True)` でラップして
     `fit()` し，返す。テスト側で小さい `cv` を指定できるよう `cv: int = 5` を新規引数として
     追加する（本番呼び出しはデフォルトの 5 のまま，`_train_and_save()` の呼び出しは変更不要）。
   - 返り値の型注釈: `LogisticRegression` → `CalibratedClassifierCV`。
   - モジュール冒頭コメント（31-36行）・`train_classifier()` の docstring を更新し，
     `method="sigmoid"` を選んだ理由（isotonic は本データ規模では過学習リスクが高いという
     sklearn 公式見解，調査(Iter29)参照）を明記する。
2. `classifier.py`
   - import 変更: `from sklearn.linear_model import LogisticRegression` →
     `from sklearn.calibration import CalibratedClassifierCV`。
   - `load_domain_classifier()`（16行）・`estimate_confidence_classifier()`（27行）の
     型注釈を `CalibratedClassifierCV` へ変更。
   - docstring に，`.classes_`／`.predict_proba()` は較正前後で同じインターフェースのまま
     動作すること（duck typing，調査(Iter29)で sklearn 公式ドキュメントにより確認済み），
     および多クラス確率の再正規化は sklearn 側が自動で行うことを明記する。
3. `tests/test_train_domain_classifier.py`
   - `test_train_classifier_fits_a_model_that_predicts_seen_labels()` の既存トイデータ
     （各クラス2件）は，`CalibratedClassifierCV` の既定 `cv=5`（`StratifiedKFold`）が
     「`n_splits(5)` が各クラスのサンプル数(2)を超える」ため失敗する。各クラスのサンプル数を
     5 件以上に増やして本番の `cv=5` の挙動に近づける（`cv` を引数で下げて回避する方法は
     本番と異なる分岐を通ってしまうため採らない）。
   - `tests/test_classifier.py` は `LogisticRegression` を直接構築して
     `estimate_confidence_classifier`/`load_domain_classifier` を呼ぶ既存の単体テストであり，
     両関数は duck typing で動くため変更不要（型注釈は実行時に強制されない．リポジトリは
     mypy 等の静的型チェックを CI で実行していないことを確認済み）。
4. （新規）較正前後の比較を行うオフライン検証スクリプト
   `scripts/evaluate_classifier_calibration.py`（仮称，rc-implementer が命名してよい）:
   - 入力: 新分類器 `models/domain_classifier_platt.joblib`（下記手順で新規生成，
     本番の `models/domain_classifier.joblib` は上書きしない），1600問評価セット
     `data/dataset.jsonl`，Iter28 実測 `results/20260731_162722/results.jsonl`。
   - 処理: 1600 問の `query` を `config.yaml` の `embedding_model`（`nomic-embed-text`）で
     再 embedding し（ライブな ollama ノード 1 台への embedding のみ．LLM 生成・probe・
     dispatch は一切発生しない），新分類器の `predict_proba` で argmax ドメインと confidence
     を求める。
   - 出力: `metrics.py:compute_ece()` と同じ行形式（`id`／`confidence`／`selected_domain`／
     `expected_domains`）の JSONL を新分類器分だけ生成する（旧分類器分は
     `results/20260731_162722/results.jsonl` をそのまま使い，再計算しない）。

**評価手順**:

1. 新分類器の学習: `uv run python -m scripts.train_domain_classifier
   --train-data data/classifier_train.jsonl --embedding-model nomic-embed-text
   --ollama-host 192.168.15.100 --output models/domain_classifier_platt.joblib`
   （wafl500．`OLLAMA_KEEP_ALIVE=-1` で常時起動済みのノードなら任意の1台でよい）。
2. 「較正前」データは `results/20260731_162722/results.jsonl`（Iter28 実測，fallback 0/1600 で
   全1600行に confidence あり）をそのまま使う。**再実行しない**。
3. 「較正後」データは 3-4 節の検証スクリプトで 1600 問を再 embedding し，新分類器の
   `predict_proba` から argmax・confidence を求めて作る。
4. `metrics.py:compute_ece(n_bins=10)` を較正前・較正後の両方に**同一の bin 設定**で適用し，
   ECE を比較する（調査(Iter29)の申し送り5）。
5. top1_accuracy（`expected_domains` との一致率）を較正前・較正後それぞれで算出し，
   新旧の正誤ペアで McNemar 検定（α=0.05）を行う。
6. ドメイン別 precision/recall を Wilson 95% CI とともに較正前・較正後で比較する
   （`legal`・`education` は訓練データが少なく Y5 で既知の弱点ドメインのため個別に確認する）。
7. 新旧 classifier の argmax 不一致件数（flip rate）を必ず報告する（成功・失敗に関わらず，
   較正が実際に argmax へ与えた影響量を定量化する）。

**ECE 基準値についての注意（今回の計画で判明した訂正）**: `config.yml` の note にある
「現状 ECE=0.204」は Iter25 の測定値であり，Iter25 は fallback 発生 212/1600 件を除いた
1388 行が母集団だった（`compute_ece` は `confidence=None` の行を除外する仕様）。Iter28
（fallback 廃止済み，現在の最良構成）は fallback 0 件で 1600 行全てに confidence があるため
母集団が異なる。**したがって「較正前」の基準値として 0.204 をそのまま流用せず，
手順2で `results/20260731_162722/results.jsonl` から改めて算出した値を Iter29 の正式な
較正前基準とする**。目標は「今回算出した較正前基準に対し較正後が改善方向であること」と
「較正後の絶対値が 0.150 以下であること」の両方を満たすこととする。

**期待効果**: sklearn 公式が述べる ensemble=True の分散抑制効果・sigmoid のモノトニック性から
ECE の改善が見込まれるが，具体的な改善幅の事前データはなく未知数である。改善幅そのものが
今回の主要な観測対象になる。

**成功条件（d0003 X9．非退行確認は理論的な形式確認ではなく実測必須と明記する）**:

1. ECE（手順2-4で算出した較正前基準に対する較正後の値，`n_bins=10` で統一）が **0.150 以下**
   であること。
2. top1_accuracy（新分類器の argmax vs `expected_domains`，1600問）が旧分類器（Iter28 実測，
   0.585）に対し McNemar 検定で有意に悪化していない（p>=0.05，または新側が改善方向）こと。
   **この判定は「predict_proba の値だけが変わり argmax はほぼ不変」という理論的仮定に基づく
   形式確認ではなく，新旧 classifier の実際の predict_proba 出力を 1600 問全件で再計算した
   実測比較として行う**（調査(Iter29)が sklearn 公式ドキュメントより「10 クラス One-vs-Rest
   較正はクラス割当を変えうる」ことを確認したため，成功条件2は必ず実測結果に基づいて判定し，
   理論的にほぼ不変であることを根拠に省略してはならない）。
3. per-domain precision/recall の CI 下限が旧分類器（Iter28 基準）の CI 下限を下回らないこと
   （success_criteria (2)）。特に `legal`（訓練77件，最小標本）・`education`（Y5 で既知の弱点
   ドメイン）の非退行を個別に確認する。
4. 新旧 classifier の argmax 不一致率（flip rate）を必ず報告する（成功・失敗の判定条件では
   ないが，較正が argmax に与えた実際の影響量を透明にするため必須）。

**「オフライン完結」という前提の検証（今回の計画で明らかになった訂正）**:
`config.yml` の note は「オフラインで完結する（実機1600問本走は不要）」としているが，
厳密には成立しない。理由: (a) 新分類器の学習自体，1427件の訓練クエリを nomic-embed-text で
embedding する必要があり，ライブな ollama ノード（embedding エンドポイントのみ）が1台要る。
(b) 較正後の argmax 非退行確認にも，1600問評価セットのクエリを同じく再 embedding する必要が
ある（`query_embedding` は `results.jsonl` に保存されていないため）。ただしいずれも
「embedding のみ」の軽い呼び出しであり，10 ノードへの probe/dispatch/LLM 生成を伴う
「実機1600問本走」（実測約90〜101分）とは負荷が全く異なる（単一ノードへの計 3027 回
（1427+1600）の embedding 呼び出しのみで，目安は数分程度）。**「実機1600問本走は不要」
という較正前後比較の主旨自体は成立するが，「ゼロ通信で完結する」という字義は不正確であり，
次回以降の申し送りとして訂正する**。

**人間判断が必要な論点**: 新規追加なし（Y2 着手前のユーザー確認が要る点は backlog B49 の
既存の申し送りのまま）。

---

### 調査 (Iter29)

**問い**: (1) 1427件・10クラス（うち legal は 77件と最少）という規模で，
`CalibratedClassifierCV` の `method='sigmoid'`（Platt）と `method='isotonic'` のどちらが技術的に
妥当か．`cv`・`ensemble` パラメータの実装上の注意点は何か．(2) 10クラスの one-vs-rest 較正では
argmax（top1）がどの程度変わりうるか，確率の再正規化は自動か手動実装が要るか．(3) ECE の
測定方法自体の問題（既出のため今回は較正手法選定に焦点を当て，深入りしない）．

#### 分かったこと

**(1) sigmoid（Platt）を第一候補とすべき — isotonic は本データ規模では過学習リスクが高い**

sklearn 公式ドキュメント（stable, 1.9.0）は明確に断定している:
「Isotonic calibration is not recommended when the number of calibration samples is too low
(≪1000) since it then tends to overfit」，および「isotonic will perform as well as or better
than sigmoid when there is enough data (greater than ~1000 samples)」
（https://scikit-learn.org/stable/modules/calibration.html）．本データは全体で 1427 件，
`legal` ドメインが最少 77 件（`data/classifier_train.jsonl` を実測，
他 9 ドメインは各 150 件均等）．`cv`（既定値 `None`=5-fold の `StratifiedKFold`，多クラスのため）と
`ensemble=True`（既定）の下では，各 fold の較正は held-out 側（約 20%）だけで行われるため，
1 fold あたりの較正サンプル数はドメインごとに **9 ドメインで約 30 件，legal で約 15 件**にしかならず，
「≪1000」はもちろん，別ソース（emergentmind.com のサーベイ）が挙げる「200 件未満で isotonic は
過学習し得る」という目安すら大きく下回る．**sigmoid（Platt）を第一候補とし，isotonic は
（実施するとしても）過学習前提のセカンダリ候補として扱うべき**．特に legal ドメインでは isotonic の
較正曲線が不安定になりやすいと予想される（B49/Y5 で既出の legal データ不足問題と同根）．

`cv`: 既定の 5-fold のままで legal（77件）は 1 fold あたり最低 5 件以上を満たすため実行は可能だが
余裕は小さい．`cv=3` にすると legal の 1 fold あたりの較正サンプルが約 25 件へ増える一方，
分類器自体の学習データが減る（この trade-off は emergentmind.com にも「K が大きいほど較正データは
増えるが計算コストが増す」と一般論として記載）．今回は既定 `cv=5` を主候補とし，
`cv=3` は余力があれば感度分析として試す程度でよい．

`ensemble`: 既定は `"auto"`（`FrozenEstimator` でなければ実質 `True`）．`ensemble=True` は
k 個の (classifier, calibrator) 組の predict_proba を平均する，バギングに近い効果があり，
小標本レジームでは分散を抑える方向に働く（sklearn 公式: 「the resulting ensemble should both be
well calibrated and slightly more accurate than with ensemble=False」）．**既定の
`ensemble=True` を維持することを推奨**．`ensemble=False`（`cross_val_predict` で unbiased
predictions を作り単一の較正器を fit）は計算コスト重視の選択で，今回のオフライン処理には
メリットが薄い．

**(2) 確率の再正規化は sklearn 内部で自動処理されるが，argmax（top1）不変という前提は
過大評価であり実機確認が必須**

sklearn 公式ドキュメント 1.16.3.3「Multiclass support」に明記: 多クラスの場合
`CalibratedClassifierCV` は `OneVsRestClassifier` 方式でクラスごとに独立して較正し，
「As those probabilities do not necessarily sum to one, a postprocessing is performed to
normalize them」．つまり `predict_proba()` の出力が 10 ドメイン間で合計 1 になる性質
（`scripts/train_domain_classifier.py` 冒頭のコメントが `estimate_confidence_classifier` の
前提として明記しているもの）は，**追加の実装なしに sklearn 側が自動的に保つ**．rc-implementer が
手動で再正規化コードを書く必要はない．

一方，config.yml の note にある「`predict_proba`の値だけが変わりargmaxの順位は理論上ほぼ不変」
という前提は，**単一の二値較正器内では真だが（同一関数によるモノトニック変換なので順位不変），
本件のように 10 個の独立した較正器（各ドメインが別々の sigmoid/isotonic パラメータを持つ）を
比較する場合には成立しないことが sklearn 公式サンプルで明示されている**．sklearn 公式の
3-class 較正サンプル（plot_calibration_multiclass.html）は次のように述べている:
「some arrows seem to cross class assignment boundaries which is not necessarily what one
would expect from a calibration map as it means that some predicted classes will change
after calibration. All in all, the One-vs-Rest multiclass-calibration strategy implemented
in CalibratedClassifierCV should not be trusted blindly.」（クラス割当が較正前後で入れ替わる
ケースがあり得ることを sklearn 自身が図示・警告している）．さらに isotonic は「introduces ties
in the predicted probabilities」（ランキング指標に影響しうる）とも明記されており，タイブレークが
argmax を不安定にする追加要因になる．sklearn issue #18709（scikit-learn/scikit-learn）でも，
メンテナ自身が多クラス較正のテストが乱数シードに対して脆弱（brittle）だったと述べている．
**結論: 「top1_accuracy は理論上ほぼ不変」という想定は過大評価であり，計画済みの軽量な実機/
オフライン再計算による非退行確認は形式的な確認ではなく必須の検証として扱うべき**．sigmoid・
isotonic の両候補について実施すること．

**参考（sklearn>=1.8 の新機能，今回のレバー値には含まれないが記録に値する）**: 本リポジトリは
`uv.lock` で scikit-learn 1.9.0 を固定しており，1.8 で追加された `method='temperature'`
（temperature scaling）が既に利用可能である．これは softmax ベースでロジットに単一スカラー
パラメータ `T` を掛けるだけの較正で，sklearn 公式が「T does not affect the location of the
maximum in the softmax output. Therefore, temperature scaling does not alter the accuracy of
the calibrating estimator」と明記する通り，**top1_accuracy 不変が理論的に保証される**（sigmoid・
isotonic の OvR 方式にはこの保証がない）．config.yml の `classifier_calibration` の
`values: [platt, isotonic]` は d0003 X9 の定義通りであり本イテレーションの変更は提案しないが，
**もし sigmoid/isotonic のいずれも非退行条件を満たせない場合，安価な追加候補として
`temperature` を検討する価値がある**ことを申し送る．

**(3) ECE 測定方法自体の注意点（簡潔に）**: `metrics.py:compute_ece()` は固定幅 10 bin
（`n_bins=10`）で ECE を計算している．binning 手法自体の妥当性（adaptive binning 等）は
過去のイテレーションで既出のため深入りしないが，**較正前後の ECE を比較する際は
bin 数・binning 方式を変更しないこと**（変更すると改善が較正の効果か binning の変更由来かを
区別できなくなる）．文献側では bin 数依存性・discretization bias は既知の問題として広く指摘されている
（Kumar et al. 2018, Nixon et al. 2019 ほか，arXiv:2501.19047 のサーベイに整理あり）が，
今回の判断には影響しない．

#### rc-planner への申し送り

1. **第一候補は `method='sigmoid'`（Platt）**．isotonic は理論・実装両面（sklearn 公式の
   「≪1000」基準，legal 77件という実データ）で過学習リスクが高く，実施する場合も
   「過学習前提のセカンダリ比較」として扱うこと．
2. `cv` は既定の 5-fold（`StratifiedKFold`）を主候補とし，`cv=3` は余力があれば感度分析として
   追加する程度でよい．`ensemble` は既定の `True`（`"auto"`）を維持すること．
3. **実装時の型注釈修正が必要**: `classifier.py:load_domain_classifier()`・
   `estimate_confidence_classifier()` の型注釈は現在 `LogisticRegression` 固定だが，
   較正後は `joblib.load()` が返すオブジェクトが `CalibratedClassifierCV` になる．
   `.classes_`・`.predict_proba()` は両方とも `CalibratedClassifierCV` に存在し実行時の挙動は
   変わらないが，型注釈とdocstring（`train_domain_classifier.py`冒頭コメント含む）の更新が要る．
4. **確率の再正規化は sklearn が内部で自動処理する**（追加実装不要）．ただし
   **「argmax（top1）はほぼ不変」という config.yml の前提は sklearn 公式が明示的に否定している
   （較正前後でクラス割当が入れ替わり得る）ため，計画済みの top1_accuracy 非退行確認は
   形式的なものではなく，sigmoid・isotonic 双方について必ず実施すること**．
5. ECE 比較時は `metrics.py:compute_ece(n_bins=10)` の bin 設定を較正前後で統一すること．
6. （任意・スコープ外の可能性あり）sigmoid/isotonic がいずれも非退行条件を満たせない場合，
   `sklearn>=1.8`（本リポジトリは1.9.0を使用）の `method='temperature'` が top1_accuracy 不変を
   理論的に保証する代替候補になりうる．今回の config.yml のレバー値定義（`[platt, isotonic]`，
   d0003 X9）を変更する提案ではなく，あくまでフォールバック案として記録する．

**出典**:
- https://scikit-learn.org/stable/modules/calibration.html （多クラス較正・isotonic過学習基準・
  temperature scaling・ensemble説明，全て 2026-07-31 時点の stable 版）
- https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html
  （`cv`・`ensemble` パラメータ定義）
- https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_multiclass.html
  （OvR較正でクラス割当が入れ替わる図示と警告文）
- https://github.com/scikit-learn/scikit-learn/issues/18709 （多クラス較正テストの脆弱性，
  Zadrozny & Elkan 2002 との実装差異の議論）
- https://github.com/scikit-learn/scikit-learn/issues/34312 （2025年，sigmoid較正にロジットを
  使うべきという未マージの改善提案，CalArena ベンチマーク論文への言及）
- https://www.emergentmind.com/topics/calibratedclassifiercv （200件未満でisotonic過学習という
  目安，cv fold数のtrade-off一般論）
- arXiv:2501.19047「Understanding Model Calibration」（ECEのbinning依存性サーベイ，参考程度）

---

### 実装 (Iter29)

計画どおり単一レバー（`classifier_calibration=platt`）のみを実装した．`config.yaml` は変更していない
（`git diff config.yaml` が空であることを確認済み）．

**変更ファイル**:

1. `scripts/train_domain_classifier.py`
   - import 追加: `from sklearn.calibration import CalibratedClassifierCV`（既存の
     `LogisticRegression` import は base estimator 用に維持）。
   - `train_classifier(embeddings, labels, cv: int = 5) -> CalibratedClassifierCV`
     （旧: `-> LogisticRegression`）。`LogisticRegression(max_iter=1000,
     class_weight="balanced")` を base estimator とし，
     `CalibratedClassifierCV(base, method="sigmoid", cv=cv, ensemble=True)` でラップして
     `fit()` するよう変更。`cv` は本番呼び出し（`_train_and_save()`）では未指定＝既定値
     5 のままで，テスト側だけが同じ引数を明示的に渡せる。
   - モジュール冒頭コメント・`_MAX_ITER` 直下のコメント・`train_classifier()` の
     docstring を更新し，sigmoid（Platt）を選んだ理由（調査(Iter29)の sklearn 公式見解，
     legal 77 件という規模）を明記。新規定数 `_CALIBRATION_METHOD = "sigmoid"`・
     `_CALIBRATION_CV = 5` を追加（マジックナンバー回避）。
2. `classifier.py`
   - import 変更: `from sklearn.linear_model import LogisticRegression` →
     `from sklearn.calibration import CalibratedClassifierCV`。
   - `load_domain_classifier()`・`estimate_confidence_classifier()` の型注釈を
     `CalibratedClassifierCV` に変更。関数本体（`.classes_`／`.predict_proba()` の
     duck typing 呼び出し）は無変更。
   - モジュール冒頭 docstring に，較正後も predict_proba がドメイン間で合計 1 になること
     （sklearn 側が one-vs-rest 較正の後処理として自動的に再正規化する）を追記。
3. `http_server.py`（計画の想定範囲外だが同一の型注釈修正として実施）
   - `NodeState.__init__` の `domain_classifier` 引数とインスタンス属性の型注釈を
     `LogisticRegression | None` → `CalibratedClassifierCV | None` に変更（import も
     `sklearn.calibration.CalibratedClassifierCV` に切替）。理由: `classifier.py` の
     2 関数と同じ較正済みアーティファクトを保持する変数であり，型注釈を放置すると
     実行時の実体と乖離した誤った型が残るため（CLAUDE.md「型を明示する」に整合）。
     `tests/test_http_server.py` は `LogisticRegression` を直接注入する既存テストのままで
     duck typing により無変更で通る（未修正）。
4. `tests/test_train_domain_classifier.py`
   - `test_train_classifier_fits_a_model_that_predicts_seen_labels()` のトイデータを
     各クラス 2 件 → 5 件（medical・legal 各 5 点，2 クラスタに揺らぎを加えた分離可能な
     2 次元点）に拡張。`CalibratedClassifierCV` の既定 `cv=5`（`StratifiedKFold`）が
     「n_splits がクラスの最小サンプル数を超える」ため不可能だった問題を回避しつつ，
     本番と同じ `cv=5` の経路を通すようにした（計画どおり `cv` を引数で下げる回避策は
     採らなかった）。
   - `tests/test_classifier.py` は計画どおり無変更（duck typing により
     `LogisticRegression` を直接構築するテストのままで通る．本リポジトリに mypy 等の
     静的型チェックは無いことを確認済み）。
5. （新規）`scripts/evaluate_classifier_calibration.py`
   - 計画の 3-4 節で規定された範囲に厳密に絞った: 1600 問評価データセットの `query` を
     ライブな ollama ノードへ再 embedding し，新分類器（`CalibratedClassifierCV`）の
     `predict_proba` から argmax ドメイン（`selected_domain`）と `confidence`
     （選択ドメインの確率）を求め，`id`／`expected_domains`／`selected_domain`／
     `confidence` の JSONL（`metrics.py` の各 `compute_*` 関数がそのまま食える行形式）を
     出力するだけに留めた。ECE・McNemar・per-domain CI・flip rate の**比較計算自体は
     実装していない**（実験フェーズで `metrics.py` の既存関数
     `compute_ece`／`compute_mcnemar_test`／`compute_precision_recall_per_domain`／
     `compute_wilson_confidence_interval` を呼び出して行う想定，本フェーズのタスク範囲
     「実験は行わない」に従った）。
   - `--dataset`／`--classifier`／`--embedding-model`／`--ollama-host`／`--ollama-port`／
     `--output`（省略時 stdout）の CLI。`_run_and_save()` 相当の非同期ループは
     `train_domain_classifier.py:build_training_features` と同様に逐次呼び出し
     （同一ノードへの同時多重呼び出しを避ける既存方針を踏襲）。

**接続方法（申し送り事項への回答）**: 較正の学習・評価は，SSH ではなく既存スクリプト群
（`train_domain_classifier.py`／`evaluate_response_quality.py`）と同じ方式，すなわち
ollama の HTTP API（`http://<ollama-host>:11434`，既定ポート）に `--ollama-host` で
直接接続する方式で実行可能である．リポジトリ内に SSH 経由でのオンライン呼び出しの前例は
なく（`remote_dir`／`mise run deploy` は Docker デプロイ用であり embedding 呼び出しには
使わない），`config.yaml` の `nodes` セクションの host（例 `192.168.15.100`＝wafl500）へ
直接 HTTP 接続すればよい。実機呼び出し自体（学習・評価スクリプトの実行）はこのフェーズ
では行っていない（フェーズ4の担当）。

**テスト結果**: `uv run pytest -q` → 211 passed, 2 skipped（既存のスキップ2件は本変更と
無関係）。全既存テストが通過。

**lint/format**: `uv run ruff check <変更ファイル>` → All checks passed。
`uv run ruff format --check <変更ファイル>` → 新規ファイル
`scripts/evaluate_classifier_calibration.py` のみ未整形だったため `ruff format` を適用し
再確認して整形済みとした。リポジトリ全体の `ruff check .` には
`scripts/prepare_lora_training_data.py` の既存 2 件（本変更と無関係，未変更ファイル）が
残っているが，これは本イテレーション以前からの既存の lint 差分であり，単一レバー原則に
従い今回は触っていない。

**オフライン fit/save/load の動作確認**: ダミー embedding（乱数8次元ベクトル）を用いて
`train_classifier()` → `joblib.dump()` → `classifier.load_domain_classifier()` →
`estimate_confidence_classifier()` の一連の経路を手動実行し，`CalibratedClassifierCV` の
round-trip・`legal`（77件）での `cv=5` 実行が例外なく成功することを確認した（実データ・
実 embedding での学習・評価は未実施，フェーズ4で行う）。

**実験を開始してよい状態か**: はい。コード変更は完了し，型注釈・テスト・lint とも整合。
フェーズ4では，(1) `scripts/train_domain_classifier.py` で
`models/domain_classifier_platt.joblib` を1台のライブ ollama ノードへの embedding 呼び出しで
新規生成（本番 `models/domain_classifier.joblib` は上書きしない），(2)
`scripts/evaluate_classifier_calibration.py` で 1600 問を再 embedding して較正後の予測
JSONL を生成，(3) `metrics.py` の既存関数群で較正前
（`results/20260731_162722/results.jsonl`，再実行不要）と較正後を比較し，成功条件
1-4（ECE≤0.150・McNemar 非退行・per-domain CI 非退行・flip rate 報告）を実測すればよい。

---

### 実験・分析(実行) (Iter29)

**実施内容**: 計画どおり実機 1600 問本走は行わず，オフラインの embedding 呼び出しのみで
較正前後の比較データを揃えた。

1. `uv run python -m scripts.train_domain_classifier --train-data data/classifier_train.jsonl
   --embedding-model nomic-embed-text --ollama-host 127.0.0.1 --ollama-port 11435
   --output models/domain_classifier_platt.joblib`（1427 件，wafl500 へ既存の SSH ローカル
   ポートフォワード `127.0.0.1:11435 -> wafl500:11434` 経由で接続。このサンドボックスから
   `192.168.15.100:11434` へ直接 TCP 接続はできない＝`curl` が 10 秒でタイムアウトすることを
   確認済みで，SSH 越しの接続が必須と判明。既存の同種トンネルが起動済みだったため流用した）。
   実行時間 124.09 秒。`models/domain_classifier_platt.joblib` を新規生成（本番の
   `models/domain_classifier.joblib` は無変更）。
2. `uv run python -m scripts.evaluate_classifier_calibration --dataset data/dataset.jsonl
   --classifier models/domain_classifier_platt.joblib --embedding-model nomic-embed-text
   --ollama-host 127.0.0.1 --ollama-port 11435 --output
   results/iter29_calibrated_predictions.jsonl`（1600 問，同トンネル経由）。実行時間 143.25 秒。
   1600 行すべてに `confidence` あり（fallback 相当なし，`predict_proba` は必ず値を返すため）。
3. 較正前データは計画どおり `results/20260731_162722/results.jsonl`（Iter28 実測）を
   再実行せずそのまま使用。同ファイルの `confidence` は
   `evaluate_classifier_calibration.py` の docstring が明記するとおり，
   `routing_method=supervised_classifier` の下では各ノードの probe confidence が
   分類器自身の `predict_proba`（そのノードのドメインの確率）と一致するため，
   較正後との比較は同一の量（分類器の predict_proba）同士の比較になっている。
4. `metrics.py` の既存関数（`compute_ece`／`compute_top1_accuracy`／
   `compute_mcnemar_test`／`compute_precision_recall_per_domain`／
   `compute_wilson_confidence_interval`）を呼び出す一時スクリプトで両ファイル（各 1600 行，
   `id` 集合が完全一致することを確認済み）を比較した。

**ECE（`n_bins=10` で統一，計画の申し送りどおり較正前基準を今回新規算出）**:

- 較正前: **0.19336**（1600 行，Iter25 の 0.204／Iter27 の 0.204 とは異なる母集団
  ＝fallback 0 件で 1600 行全件が母集団の Iter28 実測値。今回新規算出した正式な較正前基準）
- 較正後: **0.16751**（1600 行）
- 改善幅: **0.02584**（較正前→較正後で減少，改善方向）
- 目標値 0.150 以下との比較: **未達（0.16751 > 0.150）**

**top1_accuracy（1600 問，`expected_domains` との一致率）**:

- 較正前: 0.585000（Iter28 実測と同一値，`results/20260731_162722/results.jsonl` そのまま）
- 較正後: 0.595625
- 差分: **+0.010625**（較正後が高い）

**flip rate（argmax が変わった行の割合，`id` で対応付け）**:

- **176/1600 = 0.1100**（11.0%）が較正前後で `selected_domain` の argmax が変化。
  調査(Iter29)の sklearn 公式警告（10 クラス One-vs-Rest 較正で argmax が入れ替わり得る）が
  実測でも裏付けられた（flip 例: `medical-148` natural_science→medical，
  `legal-144` medical→education，`education-043` general→business_economics，
  `mathematics-134` computer_science→mathematics 等）。

**McNemar 検定（対応のある2条件比較，較正前=A・較正後=B，連続性補正あり）**:

- discordant_a_only（較正前のみ正解）: 50
- discordant_b_only（較正後のみ正解）: 67
- discordant_pairs（合計）: 117
- chi2_statistic: 2.18803
- **p_value: 0.13909**（α=0.05 で有意差なし。较正後が正解に転じた行(67)が誤りに転じた行(50)を
  上回っており，方向としては改善寄りだが統計的に有意ではない＝非退行）

**per-domain precision/recall（legal・education を個別確認，Wilson 95% CI 併記）**:

`legal`（訓練 77 件，最小標本）:

| 指標 | 較正前 | 較正後 | 差分 | 較正前 CI | 較正後 CI |
|---|---|---|---|---|---|
| precision | 0.7500 (105/140) | 0.8151 (97/119) | +0.0651 | [0.6722, 0.8144] | [0.7359, 0.8746] |
| recall | 0.5833 (105/180) | 0.5389 (97/180) | -0.0444 | [0.5103, 0.6529] | [0.4660, 0.6101] |

precision は改善（CI 下限も上昇）。recall は較正後に低下し，**CI 下限が較正前の CI 下限
（0.5103）を下回った（較正後 0.4660）**——per-domain CI 非退行条件との関係では legal の
recall のみが唯一，較正前 CI 下限割れとなった実測結果である。

`education`（Y5 既知の弱点ドメイン）:

| 指標 | 較正前 | 較正後 | 差分 | 較正前 CI | 較正後 CI |
|---|---|---|---|---|---|
| precision | 0.4631 (69/149) | 0.4633 (82/177) | +0.0002 | [0.3850, 0.5431] | [0.3914, 0.5367] |
| recall | 0.4059 (69/170) | 0.4824 (82/170) | +0.0765 | [0.3349, 0.4810] | [0.4085, 0.5570] |

education は precision がほぼ同水準，recall が大きく改善し CI 下限も上昇（下回りなし）。

**他 8 ドメインの precision/recall（較正前→較正後，参考）**:

| domain | precision | recall |
|---|---|---|
| business_economics | 0.4328→0.4439 | 0.5179→0.5417 |
| computer_science | 0.5987→0.5817 | 0.5417→0.5298 |
| general | 0.6522→0.6218 | 0.5488→0.5915 |
| history_culture | 0.7320→0.6798 | 0.6667→0.7202 |
| mathematics | 0.7075→0.6728 | 0.6190→0.6488 |
| medical | 0.4725→0.5532 | 0.4831→0.4382 |
| natural_science | 0.5135→0.5529 | 0.5655→0.5595 |
| social_science | 0.6340→0.6835 | 0.5774→0.5655 |

**使用データ**:

- 訓練データ: `data/classifier_train.jsonl`（1427 件，legal 77 件・他 9 ドメイン各 150 件）
- 評価データセット（再 embedding 対象）: `data/dataset.jsonl`（1600 件）
- 較正前の実行結果: `results/20260731_162722/results.jsonl`（Iter28 実測，1600 行，再実行なし）
- 較正後の実行結果（新規生成）: `results/iter29_calibrated_predictions.jsonl`（1600 行）
- 新規モデルアーティファクト: `models/domain_classifier_platt.joblib`（本番
  `models/domain_classifier.joblib` は無変更のまま）

**実行時間・実機呼び出しの有無**:

- `train_domain_classifier.py`: 124.09 秒（1427 回の embedding 呼び出し）
- `evaluate_classifier_calibration.py`: 143.25 秒（1600 回の embedding 呼び出し）
- 実機呼び出しは wafl500（192.168.15.100:11434）への embedding のみ（`nomic-embed-text`），
  計 3027 回。LLM 生成・probe・dispatch は一切発生していない（1600 問本走は不要という計画の
  前提どおり）。接続経路は SSH ローカルポートフォワード（`127.0.0.1:11435` ←
  `wafl500:11434`）で，このサンドボックス環境から `192.168.15.x` への直接 TCP 到達性はない
  ことを実測で確認した上でトンネル経由に切り替えた（申し送り: 次回以降オフライン検証で
  同様のスクリプトを使う際は SSH ローカルフォワードが必須である旨を明記しておく）。
- 実行中のログ・エラーに異常なし（例外・タイムアウト・リトライなし，両スクリプトとも
  正常終了メッセージを出力）。

**state.json 更新**: `status: waiting_experiment`（開始時）→`running`（完了時），
`experiment_dir`／`experiment_deadline` はジョブ開始時に設定し完了時に `null` へ戻した。
`e29_results` に上記数値一式を記録済み（`judgment` は `pending_rc_analyst_review` とし，
採否判断はフェーズ5b（rc-analyst）に委ねる）。

---

### 分析(解釈) (Iter29)

**成功条件（d0003 X9）との照合**

1. **ECE ≤ 0.150**: 未達（0.16751 > 0.150，目標まで 0.0175pt 不足）。ただし較正前 0.19336 から
   -0.02584pt（相対 13.4% 減）という改善方向自体は明確である。ルーティングは決定論的
   （config.yml success_criteria (5)）であり，同一の 1600 問・同一の embedding モデルに対し
   分類器だけを変えた比較なので，この差分はノイズではなく較正処理そのものが生んだ実測値と
   判断してよい。
2. **top1_accuracy 非退行**: 満たす。McNemar p=0.139（α=0.05 で有意差なし＝非退行）であり，
   discordant_b_only（較正後のみ正解，67件）が discordant_a_only（較正前のみ正解，50件）を
   上回っているため，方向としてはむしろ改善寄りである。
3. **per-domain CI 非退行**: legal の recall の CI 下限のみが較正前を下回った
   （0.5103→0.4660）。education は precision がほぼ横ばい・recall が改善（CI 下限も改善）で
   条件を満たす。他 8 ドメインは journal・state.json に CI 値が記録されておらず，厳密な
   非退行確認は今回未了である（結果ファイルは残っているため，必要なら追加で算出できる点を
   申し送る）。
4. **flip rate**: 176/1600（11.0%）を報告済み。判定基準ではなく報告義務だが，満たしている。

**flip rate 11.0% の解釈**

計画時点の config.yml の想定（「predict_proba の値だけが変わり argmax はほぼ不変」）は，
調査(Iter29)が sklearn 公式ドキュメント（10クラス One-vs-Rest 較正はクラス割当を変えうる，
`plot_calibration_multiclass.html` の警告）により事前に否定しており，実測でも 11.0% という
無視できない比率で argmax が変化した。この flip 自体は調査の警告どおり実際に起きたが，
「悪い」かどうかは下流の top1_accuracy で判断すべきである。McNemar の discordant 内訳
（a_only=50 < b_only=67，合計117）は，較正前後で不一致になった 117 行のうち，較正後に
正解へ転じた行（67）が誤りへ転じた行（50）より多いことを直接示している。つまり，flip rate
11.0% は「較正が argmax を大きく揺らした」という事実そのものは調査の警告どおりだが，その揺れは
統計的に有意ではないにせよ正誤の観点では改善方向に偏っており，flip rate の大きさ自体を
「悪影響の証拠」として扱うべきではない。

**legal ドメインの recall 低下の扱い（Iter28 の一般ドメイン低下との異同）**

Iter28（fallback 廃止）で general ドメインの recall 低下を「構造的トレードオフ」として判定を
覆さなかった前例（backlog B49）と，今回の legal recall 低下は機序が異なると判断する。

- Iter28 の general 低下は，fallback の送り先が常に general だったという**レバー変更そのものに
  内在する構造**（送り先を廃止すれば general への流入経路が変わるのは必然）に起因していた。
  レバーの種類（confidence 閾値・dispatch 経路）を問わず起こりうる，設計上不可避の副作用である。
- 対して今回の legal recall 低下は，(a) legal の precision は同時に明確に改善しており
  （0.7500→0.8151，CI 下限も 0.6722→0.7359 へ上昇），較正が legal への閾値をより保守的に
  動かし，境界事例の一部を他ドメインへ逃した結果と解釈できる非対称な動きである，(b) 調査(Iter29)
  が事前に指摘した「1 fold あたりの較正サンプル数が legal で最少（held-out 20%×77件≈15件）」
  という小標本レジームでの較正不安定性が，isotonic だけでなく sigmoid でも（程度は軽いにせよ）
  顕在化した可能性が高い。すなわちこれは「較正という手法一般に内在するトレードオフ」ではなく，
  「legal の訓練データが 77 件と全ドメイン中最少である」という既知の弱点（Y5，backlog B49 の
  要レビュー項目）と較正処理が相互作用した結果，と解釈するのがより妥当である。同じく弱点
  ドメインとされていた education は recall が改善（CI 下限も改善）しており precision もほぼ
  横ばいであるため，「較正は小標本ドメインに一律に悪影響を及ぼす」という単純な説明も成立しない。
  legal 固有の非対称な動き（precision 改善・recall 悪化）を踏まえると，legal の recall 低下は
  Y5（legal データ拡充）が未着手のまま較正を導入したことによる副作用であり，Y5 実施後に
  再検証する価値がある，というより限定的な解釈にとどめる。

**総合判断（rc-analyst 提案，確定は rc-reflector）: partial（部分的採用）**

根拠:

1. ECE は目標未達だが，方向性は明確な改善（-2.58pt，決定論的な測定でノイズの影響を受けない）。
   Iter20（E3, confidence_elicitation=top_k_with_probs）の「部分的採用」判定は，主基準
   （同点率）が明確に改善しつつ副基準（ECE）が悪化したケースだった。今回はこれと対称的に，
   主基準（top1_accuracy）が非退行（むしろ改善方向）で，目的としていた副基準（ECE）は
   改善したものの絶対閾値には届かなかったケースである。いずれも「単純な rejected／adopted
   の二択では実態を捉えられない」という点で共通しており，partial 判定の運用実績
   （Iter20）に整合する。
2. top1_accuracy は非退行（McNemar p=0.139，方向は改善）であり，較正導入によってルーティング
   精度が損なわれたという証拠はない。
3. legal の recall 低下（CI 下限割れ）は，成功条件(3)の唯一の違反である。ただし上記のとおり
   機序は Iter28 の general の場合のように較正という手法自体に内在する構造的トレードオフとは
   言い切れず，legal のデータ不足（Y5 未着手）と較正処理が相互作用した，より限定的で対処可能な
   副作用と考えられる。legal の precision は明確に改善しており，legal 全体への downstream の
   影響は一方向的な悪化ではない。
4. 以上を総合すると，「ECE の絶対閾値未達」のみを理由に rejected とするのは top1_accuracy の
   非退行（むしろ改善方向）という事実を過小評価することになり，一方で legal recall 低下という
   未解決の懸念を残したまま本番の `models/domain_classifier.joblib` を即座に platt 較正版へ
   置き換える adopted も時期尚早である。**partial（部分的採用）**を提案する。具体的には，
   (a) ECE 改善という方向性・top1_accuracy 非退行という事実は次イテレーション以降の判断材料
   として確定的に記録し，(b) 本番アーティファクトの置換可否は legal recall 低下の原因切り分け
   （Y5 のデータ拡充後の再検証，または `cv=3` 等への感度分析）を経てから判断する，という留保
   付きの判定が実態に即している。

**isotonic を次イテレーションで試す価値について**

調査(Iter29)は，isotonic が本データ規模（1427件，legal 77件）では sklearn 公式が明言する
「≪1000件で過学習」を大きく下回るため，sigmoid よりリスクが高いと事前に指摘していた。今回，
相対的に安全なはずの sigmoid でも legal recall の低下が観測された事実は，この事前予測
（小標本レジームでの較正不安定性）と整合する。isotonic はより柔軟な非パラメトリック較正であり
ECE を 0.150 以下まで押し下げられる可能性はあるが，legal のようにサンプル数が最少のドメインでは
較正曲線がさらに不安定になり，recall 低下が拡大するリスクが高い。**isotonic を次に試すこと自体
は妥当だが，legal ドメインの per-domain CI を今回以上に注意深く監視すること，および `cv=3`
（held-out 較正サンプルを 1/5→1/3 へ増やし legal の calibration fold を約15件→約25件に
増やす）などの感度分析を併せて行うことを条件とすべきである**。あるいは，調査(Iter29)が
申し送った `method='temperature'`（sklearn>=1.8，top1_accuracy 不変が理論的に保証される
代替）を legal 側の安全策として比較対象に加える案も検討に値する。

---

### 考察 (Iter29)

**単一レバーの判定: 部分的採用（partial）を確定**．rc-analyst の「分析 (解釈)」節の総合判定
（partial）をそのまま確定させる．判断基準は 2 点．

1. `classifier_calibration` レバーの note が明記する成功条件（d0003 X9）は「ECE≤0.150 **かつ**
   top1_accuracy 非退行」であり，AND 条件である．ECE は 0.19336→0.16751 と改善方向は明確だが
   絶対閾値 0.150 に届いておらず（0.0175pt 不足），AND 条件の一方が未達である以上，無条件の
   adopted は成立しない．
2. 一方，top1_accuracy は McNemar p=0.139 で非退行（方向はむしろ改善，discordant_b_only=67 >
   discordant_a_only=50）であり，「ECE 絶対閾値未達」のみを理由に rejected とするのは，
   非退行という事実および ECE 改善という方向性を過小評価することになる．Iter20（E3,
   confidence_elicitation=top_k_with_probs）で確立した「主基準は明確改善だが副基準が絶対閾値・
   非退行条件を満たさない場合は partial とする」運用実績と対称的に整合するケースであり，
   partial 判定を維持する．

**本番反映の判断: 見送り（`models/domain_classifier.joblib` は Platt 版へ置き換えない）**．

判断基準:

- 成功条件（d0003 X9）の AND 条件が未成立（ECE 未達）である以上，「採用して本番へ反映する」
  ための閾値をこの一回の実験だけでは満たしていない．
- 単一レバー原則・可逆性の観点では，「本番アーティファクトを置き換えない」方が常に取り消し
  コストが低い可逆な選択である．一方，仮に今回置き換えた場合，legal recall 低下という未解決の
  懸念（下記の追加分析でむしろ相対化されたが，解消されたわけではない）を抱えたまま本番へ反映
  することになり，取り消しには再度較正前の joblib を復元する手順が要る．改善の方向性は
  明確だが目標未達という「部分的採用」の性質上，最も慎重な既定選択（現状維持）を取る．
- **legal データ拡充（Y5）の完了を本番反映の前提条件にはしない**．下記の追加分析により，
  legal の recall 低下は「legal 固有の訓練データ不足（77件）が較正と相互作用した結果」という
  rc-analyst の当初仮説ほど特異な現象ではないと判明したため（後述）．したがって本番反映見送りの
  理由は「legal 固有の弱点」ではなく，**ECE 絶対閾値が未達であるという d0003 X9 の成功条件
  そのもの**に一本化する．

**追加分析: 較正前後 CI 比較を全10ドメインへ拡張（rc-analyst 未了分の解消）**

rc-analyst は分析(解釈)フェーズで legal・education の2ドメインのみ per-domain CI を確認し，
他8ドメインは「今回未了」と申し送っていた．本フェーズで `metrics.py` 既存関数
（`compute_wilson_confidence_interval`）を用い，較正前（`results/20260731_162722/results.jsonl`）・
較正後（`results/iter29_calibrated_predictions.jsonl`）の全10ドメイン・precision/recall
（計20指標）の Wilson 95% CI を追加算出した．

結果，「較正後の CI 下限が較正前の CI 下限を下回る」という成功条件(3)の字義どおりの判定基準で
見ると，**legal の recall 以外にも 8 件が該当した**（computer_science: precision・recall 両方，
general: precision，history_culture: precision，mathematics: precision，medical: recall，
natural_science: recall，social_science: recall）．20 指標中 9 指標（45%）が該当し，legal は
その中の 1 件にすぎない．

しかし，**該当した9件はいずれも較正前後の CI が大きく重なっており（区間が非交差），統計的に
有意な差とは言えない**（例: computer_science precision 較正前 [0.5193,0.6732]・較正後
[0.5025,0.6569]，mathematics precision 較正前 [0.6294,0.7750]・較正後 [0.5973,0.7404]，
legal recall 較正前 [0.5103,0.6529]・較正後 [0.4660,0.6101]）．また該当ドメインには
computer_science・mathematics・history_culture という訓練データが legal（77件）と同じく
最少ではなく他ドメインと同じ 150 件のドメインが含まれる．**これは「legal は訓練データが
最少だから較正の影響を受けやすい」という rc-analyst の仮説（分析(解釈)節）が，唯一の説明では
ないことを示す**．より妥当な解釈は，「CI 下限の単純な大小比較」という成功条件(3)の字義通りの
運用が，10 ドメイン×2 指標＝20 個の周辺検定を補正なしに行っていることに等しく，較正が
argmax の 11.0%（176/1600）を無作為に近い形で再配分すれば，どのドメインの点推定も上下に
ブレて当然，約半数の指標で「たまたま CI 下限が下がる」という偽陽性が生じる，という統計的な
アーティファクトである可能性が高い．

**得られた学び（次回以降に活きる非自明な点）**:

1. **per-domain CI 下限の単純比較は，ドメイン数×指標数が多い場合，多重比較の補正なしに
   「非退行チェック」として運用すると簡単に偽陽性を出す**．今回 20 指標中 9 指標が該当基準を
   満たしたが，いずれも区間は非交差ではなく重なっており，統計的に有意な退行ではない．
   Iter28 の考察（backlog B49 学び1，「paired 比較で McNemar と Wilson CI の周辺重複が食い違う」）
   と同根の問題であり，**次回以降 success_criteria (2) を運用するときは，「CI 下限の単純な
   前後比較」ではなく「区間が非交差（overlap しない）」を退行の閾値にするか，可能ならドメイン単位の
   McNemar 検定に切り替えるべき**である（config.yml success_criteria の見直し候補として記録．
   次回 rc-planner・rc-analyst が判断する）．
2. **legal recall 低下を「訓練データ最少ドメイン固有の脆弱性」と断定するのは早計だった**．
   同種の（統計的に有意でない）変動は訓練150件の複数ドメインでも観測されており，較正が
   小標本ドメインに選択的に悪影響を与えるという仮説は，今回の追加分析だけでは支持されない．
   ただし否定もされていない（isotonic でより顕著化する可能性は依然残る，調査(Iter29)の事前
   予測どおり）．次回 isotonic を試す際は，legal だけでなく **全10ドメインの CI を同一手順で
   算出する運用を標準化する**（今回のように事後追加で穴埋めしない）．
3. **ECE 改善幅（-2.58pt，相対13.4%減）は決定論的な測定でノイズの影響を受けないため確定的な
   事実として扱ってよいが，絶対閾値0.150への到達には Platt 単独では届かない**．較正手法の
   選択そのもの（sigmoid vs isotonic vs temperature）の効果差が，見かけ上のドメイン別ノイズより
   小さい可能性があり，次回は isotonic で追加の改善余地があるかを確認する価値がある．

**次に振る単一レバーの選定: `classifier_calibration=isotonic`**

判断基準:

- 計画(Iter29)・調査(Iter29)の時点で「isotonic は Platt が不成功の場合のみ次イテレーションで
  別途検証する」と明記済みであり（単一レバー原則を守るため同一イテレーションに混ぜなかった），
  今回 Platt が ECE 絶対閾値に未達（＝d0003 X9 の意味で「不成功」）だったため，この条件が
  成立した．
- **可逆性・独立性**: `classifier_calibration` は config.yml の levers に既に
  `values: [platt, isotonic]` として登録済みの候補であり，スキーマ変更・関数シグネチャ変更を
  伴わない．ユーザー確認は不要（Y4 全体が「オフライン・低コスト・スキーマ変更不要」として
  既に承認済みの方針の範囲内）．
- `cv`（既定5）・`ensemble`（既定True）は Platt と同一に固定し，較正手法（sigmoid→isotonic）
  のみを単一レバーとして変える．調査(Iter29)が申し送った `cv=3` 感度分析は，isotonic の主結果
  （`cv=5`）で legal 等の recall がさらに悪化する場合にのみ追加で実施する副次分析とし，
  同一イテレーションの主比較には含めない（単一レバー原則を守るため）．
- 今回標準化した「全10ドメインの CI を較正前後で同一手順で算出する」分析を isotonic でも
  最初から行い，事後の穴埋めが不要な計画にすること．
- ECE が isotonic でも 0.150 に届かない場合，調査(Iter29)が申し送った `method='temperature'`
  （sklearn>=1.8，top1_accuracy 不変が理論的に保証される）を次々点の代替候補として検討する．

**iteration_name（Iter30）**: 「分類器較正のisotonic方式によるECE目標達成の追試とドメイン別
非退行の全数検証」

**要人間判断として残す論点（新規追加なし）**: Y2（`confidence_threshold` の二重責務分離，
スキーマ変更）の着手前ユーザー確認は backlog B49 の既存の申し送りのまま．fallback 設計思想の
論文上の位置付け（backlog B48）も未解決のまま据え置く．較正済み分類器の本番反映可否も，
今回は「見送り」という可逆な既定選択を自律判断で行ったのみで，将来 isotonic 等が成功条件を
完全に満たした場合の本番反映という不可逆に近い判断（本番運用中のルーティング挙動を変える）
自体は，改めてその時点で検討する．

---

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

