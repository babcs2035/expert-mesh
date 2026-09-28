## Iteration 98: 分類器の多クラス分解を誤り訂正出力符号へ変える

### 調査 (Iter98)

本フェーズでも**実験ノード wafl500〜509 には一切触れていない**（2026-09-23 絶対条件 (B)）．
開発ホスト上で行ったのは，リポジトリのコード読み取りと，**合成データによる sklearn の API 実現性の
確認（CPU のみ・数十秒）**だけである．`models/`・`data/`・`config.yaml`・`results/` への書き込みは 0 件．

**(1) 問い**（B163 (f) の申し送りを受けて設定した）

- Q1: ECOC は Iter97 で退行の機序と読んだ 2 点，すなわち **(a) 対ごと分解の標本分割（p≫n の悪化）**と
  **(b) 非適格分類器の票**を，構造的に回避するか．回避するなら OvO の反証は「決定層の分解一般」ではなく
  「OvO 固有の弱点」に帰属でき，軸を閉じる根拠になる．
- Q2: ECOC 固有の落とし穴（符号長というハイパラ，復号方式，離散性）は何か．Iter97 の P4（gap 分布の
  収縮）と同型の問題を持ち込まないか．
- Q3: 現行の較正済みパイプライン（`CalibratedClassifierCV(method="temperature")` ＋
  `sample_weight = n/(K*n_d)`）と，`classifier.py:estimate_confidence_classifier()` の
  「`predict_proba` が 10 ドメインで和 1」という前提を壊さずに実装できるか．

**(2) 文献調査（tvly search / extract）**

- **Dietterich & Bakiri (1995), *Solving Multiclass Learning Problems via Error-Correcting Output
  Codes*, JAIR 2:263–286** —— ECOC の原典．k クラスに長さ L の符号語を割り当て，各列が
  **クラスを 2 つのメタクラスに分ける二値問題**になる．行間の Hamming 距離が d なら
  ⌊(d−1)/2⌋ ビットまでの二値分類器の誤りを復号で訂正できる．
- **Allwein, Schapire & Singer (2000), *Reducing Multiclass to Binary: A Unifying Approach for
  Margin Classifiers*, JMLR 1:113–141**,
  <https://www.jmlr.org/papers/volume1/allwein00a/allwein00a.pdf> ——
  本反復の設計はこの論文に従う．PDF から直接引用して確認した事実は 3 点．
  (i) **dense random code の長さは ⌈10·log₂(k)⌉**（sparse は ⌈15·log₂(k)⌉）で，
  **10,000 本の候補からから行間最小 Hamming 距離が最大のものを選ぶ**（「examining 10,000 random codes
  and choosing the code that had the largest ... and did not have any identical columns」）．
  **k=10 なら L = ⌈33.22⌉ = 34**．符号長を掃引せず単一値を事前登録せよという B163 (f)(1) の要件に，
  **文献由来の既定値**で答えられる．
  (ii) **loss-based decoding は Hamming decoding より優れる**（「These bounds indicate that
  **loss-based decoding is superior to Hamming decoding**」）．さらに
  「the loss-based decoding method for **log-loss is the well known and widely used
  maximum-likelihood decoding**」とあり，log 損失での復号＝最尤復号である．
  (iii) 基底学習器が SVM のとき「it is clear that **the widely used one-against-all code is inferior
  to all the other codes we tested**」．ただし「there is **no clear winner** among the four other
  output codes」であり，AdaBoost では「none of the codes is persistently better than the other」．
  すなわち**符号の種類による差は保証されていない**．
- **反証側: Rifkin & Klautau (2004), *In Defense of One-Vs-All Classification*, JMLR 5:101–141**,
  <https://www.jmlr.org/papers/volume5/rifkin04a/rifkin04a.pdf> ——
  「a simple one-vs-all scheme is **as accurate as any other approach**, assuming that the underlying
  binary classifiers are **well-tuned regularized classifiers**」．同論文は，異なる分解方式が
  同じ点で同じ誤りを犯す（AGREE 列が誤り率より大きい）ことを示し，
  「points become errors **not because of deficiencies in the method of combining binary
  classifiers**」と述べる．**Iter97 の実測はこの側と整合した**ので，Iter98 も同じ側に落ちる確率は高い．
  本反復を「改良案」ではなく**軸を閉じるための確認実験**と位置づける根拠である（B163 (f)）．
- **Ghani (2000), *Using Error-Correcting Codes for Text Classification*, ICML** および
  Berger (1999), *Error-correcting output coding for text classification*, IJCAI ——
  テキスト分類（高次元疎特徴・多クラス）で ECOC が有効だった先行例．ただし両者の基底学習器は
  Naive Bayes であり，**弱い基底学習器のとき ECOC の利得が大きい**という Rifkin らの説明と矛盾しない．
  本研究の基底学習器は正則化つき線形 LR であり，条件は Ghani らより Rifkin ら側に近い．
- **Escalera, Pujol & Radeva (2010), *On the Decoding Process in Ternary ECOC*, IEEE TPAMI
  32(1):120–134** —— 復号方式の比較（Hamming / Euclidean / loss-based）．本反復は 3 値符号を使わないので
  0 要素の扱いは論点にならない．

**(3) ECOC は Iter97 の 2 つの機序を構造的に回避するか（Q1 への回答）**

| Iter97 で読んだ退行の機序 | OvO | ECOC（dense random, L=34） |
|---|---|---|
| (a) 標本分割（各二値問題が約 500 行しか見ない） | 該当．2,275 → 約 500 行（1/4.5） | **非該当**．各列が**全 2,275 行**を使う |
| (b) 非適格分類器の票（真クラス d を知らない 36/45 本が投票） | 該当 | **非該当**．全列が 10 クラス全部を ±1 に割り当てる |
| P4 のスコア離散性（得票数 0〜9 の階段） | 該当（`gap<0.36` が 20.1%→8.3%） | **非該当**．log 尤度の和は連続量（合成データで 600 行すべて相異なる値を実測） |

したがって ECOC は (d) の読み（B163 (d)）の**直接の対照実験**として機能する．
**ただし ECOC 固有のリスクがある**: 各列は「5 クラス対 5 クラス」のようなメタクラス分割であり，
意味的にまとまらない多峰なクラス塊を 1 本の線形境界で分けることになる．
n=2,275・p=5,120（p≫n）では**任意のラベル付けが訓練データ上ほぼ必ず線形分離可能**なので，
訓練誤差は 0 に近づく一方で汎化しない恐れがある（Dietterich & Bakiri が要求する「基底学習器が
任意の二分割を学習できること」は満たすが，「汎化できること」は別問題である）．
この懸念は**列ごとの二値 CV 正解率**として直接測れるので，診断出力に含める（下記 P5）．

**(4) 実現性の実測（開発ホスト，CPU のみ，合成データ．Iter97 の落とし穴に相当するものを事前に潰した）**

- **最大の落とし穴（実測）**: **`sklearn.multiclass.OutputCodeClassifier`（sklearn 1.9.0）は
  `predict` しか持たない**（`predict_proba` も `decision_function` も無い）．そのため
  `CalibratedClassifierCV(OutputCodeClassifier(...))` は
  `InvalidParameterError: The 'estimator' parameter of CalibratedClassifierCV must be an object
  implementing 'fit' and 'predict_proba', an object implementing 'fit' and 'decision_function' or
  None` で**そもそも fit できない**．すなわち Iter97 のような「base estimator を差し替えるだけ」の
  変更は**不可能**である．`classifier.py` が要求する「`predict_proba` が 10 ドメインで和 1」を
  満たすには，**復号スコアを返す薄いラッパを自前で書く**必要がある．
- そこで **`EcocLogLossClassifier`**（dense random code ＋ **log 損失による loss-based decoding
  ＝最尤復号**）を約 60 行で試作した．`decision_function(X)[i, d] = Σ_j log σ(M[d,j]·f_j(x_i))`．
  これを既存の `CalibratedClassifierCV(method="temperature", cv=5, ensemble=True)` にそのまま渡せる．
  実測（合成データ）:
  - `CalibratedClassifierCV(EcocLogLossClassifier(...), method="temperature", cv=5, ensemble=True)`
    は fit でき，**`predict_proba` の行和は全行 1.0**（`classifier.py` の前提を満たす）．
  - **`sample_weight` は警告 0 件で 34 本すべての二値 LR へ到達する**．`fit(self, X, y, sample_weight=None)`
    と明示的な引数を持たせたので，sklearn の `has_fit_parameter` 判定が通り，
    Iter97 で必要だった `enable_metadata_routing` は**不要**である．到達の実証として，
    クラス 0 の重みを 20 倍にした場合と等重みの場合で二値 LR の係数が
    `max|Δcoef| = 0.0935`（較正ラッパ経由でも 0.0738）と**有意に動く**ことを確認した
    （0 なら未到達．Iter97 の学び 2 に対応する機械的な検査）．
  - 符号行列は L=34，**行間最小 Hamming 距離 14**（⌊13/2⌋=6 ビットまで訂正可能），重複列 0．
  - **スコアは連続**（合成 600 行で上位 2 位差の相異なる値が 600/600）．Iter97 の P4（離散性による
    gap 収縮）は原理的に起こらない．
  - 計算量（n=1,820 × p=5,120 の合成データ）: 単一 softmax 0.20 秒，5 対 5 の二値 1 本 0.07 秒，
    ECOC 34 本 **1.01 秒**，temperature 較正込み（cv=5）**5.1 秒**．
    3 seed × 5-fold でも **wafl-ctrl5 の CPU で数分〜十数分**に収まる（再埋め込み 0 回・GPU 不要）．
- **デプロイ側の到達条件（足切り通過時のみ問題になる）**: joblib はクラスを**モジュールパス参照**で
  直列化するため，`EcocLogLossClassifier` を `scripts/train_domain_classifier.py` の中で定義して
  スクリプト実行すると `__module__` が `__main__` になり，ノード側で**復元できない**．
  そのため実装時は**リポジトリ直下に `ecoc_head.py` を新設**して同クラスを置き，
  `scripts/train_domain_classifier.py` から import し，`Dockerfile` の `COPY` 行（現行 L14）へ
  `ecoc_head.py` を追加する．スクリーニング段階では直列化しないのでこの作業は不要である．

### 仮説 (Iter98)

**Iter97 の OvO の退行（Δ=−0.806pt）は「決定層を分解したこと」自体ではなく，OvO 固有の 2 点
（各二値問題が全体の約 1/4.5 の行しか見ない標本分割，および真クラスを知らない 36/45 本の票）に
由来する．** 全行を使い非適格分類器を持たない ECOC（dense random code, L=34, 最尤復号）に替えれば，
少なくとも**近傍クラス（social_science・history_culture）の有意退行は再現しない**はずである．
そのうえで CV top1 が上がるかどうかは両論あり（Allwein et al. は OvA 劣位を報告，
Rifkin & Klautau は同等と主張），**Iter97 と同じく両論のまま検定する**．
対抗仮説は「5,120 次元・2,275 行で任意の 5 対 5 分割を線形に学習させても汎化しない」であり，
これが正しければ列ごとの二値 CV 正解率が低く（偶然水準に近く），ECOC も退行する．

### 単一レバー (Iter98)

- **レバー**: `classifier_multiclass_decomposition` = **`error_correcting_output_codes`**
  （config.yml に事前登録済みの未試行 value．新レバーの追加は不要）
- **何を何から何へ**: `scripts/train_domain_classifier.py:train_classifier()` の **L210**
  `base_estimator = LogisticRegression(max_iter=_MAX_ITER, class_weight=None)` を，
  **`EcocLogLossClassifier(estimator=LogisticRegression(max_iter=_MAX_ITER, class_weight=None),
  code_length=34, random_state=98)`**（新モジュール `ecoc_head.py`）へ差し替える．
  L211-214 の `CalibratedClassifierCV(base_estimator, method=_CALIBRATION_METHOD, cv=cv,
  ensemble=True)` と `fit(..., sample_weight=sample_weight)` は**一字も変えない**．
- **事前登録する設計値（掃引しない．B163 (f)(1)）**:
  - **符号長 L = 34 = ⌈10·log₂(10)⌉**（Allwein et al. 2000 の dense random code の既定）．
  - **符号行列**: 候補 10,000 本から**行間最小 Hamming 距離が最大**のものを選ぶ（同論文の手続き）．
    定数列・重複行は排除する．`random_state = 98`（イテレーション番号．固定）．
  - **復号**: log 損失の loss-based decoding（＝最尤復号）．Hamming 復号は使わない（同論文の
    理論的優位に従う）．**代替値の比較は一切行わない**（行えば単一レバー原則が濁る）．
- **レバーを読むコード行と到達条件（d0004 §4 の恒久対策）**:
  - スクリーニング段階: `scripts/screen_classifier_multiclass_decomposition.py` に**腕 C** を追加する
    （`_train_ecoc_classifier()`）．腕 A は既存の `train_classifier()` を無変更で import して呼び，
    **CV top1 = 0.802637 の再現**をもって測定系の健全性を確認する（Iter97 と同じ手順）．
  - 発火の証拠（1 つでも欠けたら実験不成立として中断）:
    (a) 学習済み腕 C の `code_book_.shape == (10, 34)`，(b) 行間最小 Hamming 距離 ≥ 10 を記録，
    (c) `sample_weight` 到達の実証（等重み版との `max|Δcoef| > 0`），
    (d) `fit` 中の警告 0 件（`warnings.catch_warnings(record=True)` ガード．Iter97 の実装を流用），
    (e) `predict_proba` の行和が 1.0 ± 1e-6．
  - 本走段階（足切り通過時のみ）: `models/domain_classifier.joblib` の差し替えとデプロイが唯一の
    到達条件である．旧 artifact を `models/domain_classifier_pre_iter98_ecoc.joblib` へ退避し，
    新 artifact の sha256 が旧と異なること・`ecoc_head` がコンテナ内で import 可能なこと
    （`docker compose exec ... python -c "import ecoc_head"`）・各ノードの joblib の sha256 一致・
    `predict_proba` の行和 1.0 を確認する．
- **固定する構成（1 つも動かさない）**: 訓練集合 `data/classifier_train_iter94_dedup.jsonl`（2,275 行）・
  `sample_weight = n/(K*n_d)`・較正 `temperature`（`cv=5, ensemble=True`）・`C=1.0`・`max_iter=1000`・
  埋め込みモデル `qwen3-embedding:4b`・`embedding_instruction`（Iter81 の P1 文言）・
  `embedding_view_concat: true`（5,120 次元）・評価集合 `data/dataset.jsonl`（3,750 行，読むだけ）・
  `dispatch_gap_threshold: 0.36`・`dispatch_gap_max_k: 4`・`dispatch_top_k: 2`・
  `aggregation_method: max_confidence`・expert/light モデルとノード割り当て．

### 事前スクリーニング設計（事前登録 / Iter98）

Iter95〜97 の型をそのまま踏襲する．実施場所は **wafl-ctrl5**（絶対条件 (B)．`~/expert-mesh-iter95/` の
uv 3.12 環境と `data/embcache_iter96_scaled.npy` の先頭 2,275 行を流用．**新規の埋め込み計算は 0 回**）．
**足切りを通った場合は本走を省略しない**（2026-09-23 絶対条件 (A)）．

- **主評価**: 既存 2,275 行の層化 5-fold（`StratifiedKFold`，seed = 96 / 196 / 296，併合 n = 6,825）．
  テスト fold は両腕で完全に同一．**腕 A = 現行 softmax**（CV top1 = 0.802637 の再現を確認），
  **腕 C = ECOC（L=34, 最尤復号）**．腕 B（OvO）は再実行しない（反証済み．B163 (a)）が，
  Iter97 の実測値を表に併記して 3 方式を比較できるようにする．
- **統計**: `metrics.py` の `compute_mcnemar_test` / `compute_domain_recall_mcnemar_test` /
  `apply_benjamini_hochberg` を流用．**seed 併合値を主とし，discordant を seed 数で割った保守値を
  必ず併記する**（B163 (b)・Iter97 学び 4）．
- **最初から実装する診断出力（B163 (f)(4)．事後の再実行を避ける）**:
  1. 両腕の **10×10 混同行列**．
  2. **34 列それぞれの二値 CV 正解率**（メタクラス分割が学習できているかの直接の測定）と，
     符号行列の行間最小／平均 Hamming 距離．
  3. 較正後の **ECE・Brier**（両腕）と，**`gap ≥ 0.36` バケットの正解率**（Iter97 学び 5 への対応）．
  4. **gap 分布**と `gap < 0.36` の行割合．
  5. 腕 C の**復号スコアの上位 1 位と 2 位の差**の分布（OvO の票差に対応する量．離散性の確認）．

### 反証可能な予測（事前登録 / Iter98）

- **P1（足切りと同値・主予測）**: CV top1 Δ (C−A) **≥ +1.0pt**．
- **P2（機序の切り分け．本反復の中心）**: Iter97 で BH 有意退行だった 2 クラスの recall Δ の合計が
  **≥ −1.0pt**（Iter97 の OvO は social_science −4.53 ＋ history_culture −2.86 = **−7.39pt**）．
  - P2 が通り P1 が外れる → 退行は **OvO 固有**だったが，分解方式では精度は動かない．
    **決定層の軸は打ち止め**にしてよい（B163 (g)(h)）．
  - P2 も外れる（ECOC でも同じ 2 クラスが大きく退行する） → 標本分割・非適格分類器という
    Iter97 の機序の読み（B163 (d)）は**誤り**であり，「10 クラスを二値へ分解すること自体」が
    この特徴空間で損をしている，という別の読みに差し替える必要がある．
- **P3（確率の健全性）**: `predict_proba` の行和が全行 1.0 ± 1e-6 で，かつ `gap < 0.36` の行割合が
  腕 A（20.1%）から **±5pt 以内**（Iter97 の OvO は −11.8pt 逸脱）．連続スコアなので通る見込み．
- **P4（較正の健全性）**: 腕 C の ECE が腕 A の ECE を**上回らない**．
  外れる場合は「鋭さの改善」を性能の証拠として読まない（Iter97 学び 5）．
- **P5（対抗仮説の直接検定）**: 34 列の二値 CV 正解率の**中央値が 0.70 以上**．
  0.6 を下回るなら「p≫n でメタクラス分割は分離できても汎化しない」が支持され，
  ECOC の失敗はこれで説明される（この場合，符号長を伸ばしても救えない）．

### 成功条件・非退行条件（事前登録 / Iter98）

**基準線（本走）**: `results/20260928_160921/` の top1 = **0.833067**（3,750 問，
Wilson 95%CI [0.8208, 0.8447]）．再現性の床は ±0.25pt．
**基準線（CV）**: 腕 A = 0.802637．CV の Δ の SE ≒ **0.22pt**（Iter95/96 から見積り．
足切り +1.0pt は約 4.5 SE）．

- **スクリーニングの足切り（両方を満たさなければ本走せず `closed`．Iter97 と同一水準に据え置く）**:
  1. **CV top1 Δ (C−A) ≥ +1.0pt**．
  2. **per-domain recall の BH 補正後（q=0.05，10 指標）有意な退行が 0 件**
     （seed 併合値を主，保守値を併記）．
  - 外した場合は `models/`・`config.yaml`・`data/dataset.jsonl`・`scripts/train_domain_classifier.py` を
    **1 バイトも変更せず**，スクリーニングスクリプトと結果 JSON だけを残してフェーズ 3 へ渡す．
- **本走（足切り通過時のみ）**: `models/domain_classifier.joblib` を ECOC 版で再訓練し，
  `mise run deploy` と sha256 確認のうえ **wafl500〜509 で 3,750 問フルスペック**を 1 回実行する．
- **本走の判定**: Δ ≥ +0.5pt かつ McNemar p < 0.05 で `adopted`，Δ ≥ +0.5pt だが p ≥ 0.05 なら
  `adopted_small`（要再現），|Δ| < 0.25pt なら `negligible`，Δ ≤ −0.5pt なら `rejected`．
- **必須の非退行条件（1 つでも破れたら artifact をロールバック）**:
  - C1: per-domain precision/recall 計 20 指標の BH 補正後の有意退行が **0 件**．
  - C2: `used_fallback` 率・`dispatch_failed` 件数が基準線から増えない．
  - C3: レバー発火の証拠（上記 (a)〜(e) と，ノード上の joblib の sha256 一致・`ecoc_head` の import 可否）．
  - C4（参考値）: Random / BestSingle / Oracle を併記し BestSingle 超過を明示（success_criteria (3)）．
  - C5: 複合設問 730 行の top1（基準 0.7904）が −1.0pt を超えて下がらないこと．
  - C6: `mean_duration_ms` が基準線の +20% 以内（推論は `predict_proba` 1 回のままで実質不変のはず）．

### 実行フェーズへの申し送り（Iter98）

- **`OutputCodeClassifier` をそのまま使ってはならない**（`predict_proba` も `decision_function` も
  持たず `CalibratedClassifierCV` に渡せない）．`ecoc_head.py` の `EcocLogLossClassifier` を使うこと．
  試作は `/tmp/ecoc_probe.py` にあるが，本実装は責務コメント・docstring・型注釈を付けて書き直すこと．
- **符号長 34・最尤復号・`random_state=98` は事前登録値であり掃引禁止**．
  「符号長を変えたら上がるかもしれない」という誘惑に乗ると単一レバー原則が壊れる（B163 (f)(1)）．
- **`sample_weight` 到達の機械的な検査を実装に埋め込むこと**（警告ガード＋`max|Δcoef| > 0` の実証）．
  Iter97 の学び 2 の型を踏襲する．
- 診断出力（混同行列・列ごと二値 CV 正解率・ECE/Brier・gap 分布・復号スコア差分布）は
  **最初から**出すこと．足切りを外した場合でも機序が読めるようにするためである．
- `dispatch_gap_threshold` の再較正は別レバーであり，本反復では**絶対に触らない**．
- 実機ノード wafl500〜509 は**本走時のみ**．スクリーニングは wafl-ctrl5 で完結する．

### 実装・実験 (Iter98)

**結論: 事前スクリーニングの足切り 2 条件をいずれも不通過．本走は行っていない．**
実機ノード wafl500〜509 には一切触れておらず，`models/`・`data/dataset.jsonl`・`config.yaml`・
`scripts/train_domain_classifier.py`・`Dockerfile` は 1 バイトも変更していない
（`git diff --stat` で実測確認済み）．

**(1) 実装**

- **`ecoc_head.py`（リポジトリ直下に新規）**: `EcocLogLossClassifier`．dense random code，
  L = ⌈10·log₂(K)⌉ = 34，候補 10,000 本から行間最小 Hamming 距離最大を選択，
  `decision_function` は log 損失の loss-based decoding（＝最尤復号）．リポジトリ直下に置いたのは
  joblib がクラスをモジュールパス参照で直列化するためである（`scripts/` 配下や `__main__` で
  定義するとノード側で復元できない）．**足切り不通過のため本番経路へは配線していない**．
- **`scripts/screen_classifier_multiclass_decomposition.py`（既存 298 行を書き直し）**:
  腕 A（既存 softmax，import は無変更）と腕 C（`_train_ecoc_classifier()`）の 3 seed × 固定 5-fold 比較．
  腕 B（OvO）は再実行せず Iter97 実測値を `_ITER97_ARM_B_RECORDED` 定数として表に併記した．
  診断出力（10×10 混同行列・34 列の二値 CV 正解率・符号行列の Hamming 距離・ECE/Brier・gap 分布・
  較正前の生復号スコアの上位 1-2 位差分布）を最初から実装している（B163 (f)(4)）．
  保守側 McNemar は discordant を seed 数 3 で割ったうえで `metrics._mcnemar_from_correctness` の
  既存 χ²/p 式を再利用して算出した（χ² 式を自前で再導出していない点が重要．B163 (b)）．
- **`results/iter98_screening/screening_result.json`（新規）**: 生の実行結果全件．
- 検証: `ruff check` 全通過，`py_compile` OK，`pytest tests/test_metrics.py` 53 passed
  （`metrics.py` は無変更で回帰なし）．`pytest tests/test_train_domain_classifier.py` は
  4 passed / 1 failed だが，失敗は `ModuleNotFoundError: sentence_transformers` であり
  開発ホストの venv に当該パッケージが無いという環境要因．`train_domain_classifier.py` の diff は 0 で
  本反復の変更とは無関係である．
- 合成データによる単体検証: `predict_proba` 行和 1.0，符号行列 (10, 34)・最小行間距離 14，
  `sample_weight` 到達（等重み比 `max|Δcoef| = 0.123`，警告 0 件）．

**(2) 事前スクリーニングの結果**（wafl-ctrl5，CPU のみ，n = 2,275，3 seed × 固定 5-fold，
併合 n = 6,825，所要 約 48 分）

| 腕 | 分解方式 | CV top1 |
|---|---|---|
| A | 単一 softmax（現行） | **0.802637** |
| B | OvO（Iter97 実測の併記．本反復では再実行せず） | 0.794579 |
| C | ECOC（L=34，最尤復号） | **0.788425** |

- **CV Δ (C−A) = −1.421pt** → 足切り 1（≥ +1.0pt）に**不通過．符号も逆**である．
- 全体 McNemar 併合値: discordant 179/82，χ² = 35.31，**p = 2.81e-9**．
  保守値（discordant ÷ 3）: 60/27，χ² = 11.77，**p = 0.00060**．
  **併合・保守のいずれでも有意に悪化**しており，B163 (b) で懸念した判定の反転は起きていない．
- per-domain recall Δ（併合 p・BH 判定 / 保守 p・保守 BH 判定）:
  - education: **−7.00pt**，p=1.23e-8 有意 / 保守 p=0.00137 有意
  - social_science: **−6.40pt**，p=1.60e-10 有意 / 保守 p=4.07e-4 有意
  - general: −2.67pt，p=5.10e-5 有意 / 保守 p=0.0233 非有意
  - business_economics: **+1.87pt**，p=0.0108 有意（改善方向） / 保守 p=0.182 非有意
  - history_culture: −0.16pt，p=1.0 非有意（**Iter97 で BH 有意退行だった域が今回は動いていない**）
  - legal・medical・mathematics・computer_science・natural_science: |Δ| ≤ 1.3pt，いずれも非有意
- **BH 後の有意退行 = 併合 3 件（education・general・social_science）／保守 2 件（education・
  social_science）** → 足切り 2（0 件）に**不通過**．

**(3) 事前登録した予測 P1〜P5 の当落**

- **P1（CV Δ ≥ +1.0pt）: 落選**（−1.421pt）．
- **P2（Iter97 で BH 有意退行だった 2 クラスの recall Δ 合計 ≥ −1.0pt）: 落選**（合計 −6.559pt）．
  内訳が重要で，**history_culture は −0.16pt でほぼ不変＝Iter97 の当該退行は再現しなかった**一方，
  **social_science は −6.40pt と Iter97 の −4.53pt よりさらに悪化**した．
  事前登録どおり「P2 も外れる」ケースに該当し，B163 (d) の機序の読み（標本分割・非適格分類器）は
  差し替えを要する．
- **P3（`predict_proba` 行和 1.0 かつ gap<0.36 割合が腕 A ±5pt 以内）: 当選**
  （行和 1.0±1e-6，gap<0.36 割合 Δ = +0.396pt）．Iter97 の P4（gap の人為的収縮）は ECOC では
  予告どおり起きておらず，復号スコアの連続性という構造的差異は実測で確認された．
- **P4（腕 C の ECE が腕 A を上回らない）: 当選**（ECE 0.01650 ≤ 0.01971）．
  Brier は腕 C がわずかに悪い（0.12796 vs 0.12337）が，P4 の判定基準は事前登録どおり ECE のみ．
- **P5（34 列の二値 CV 正解率の中央値 ≥ 0.70）: 当選＝対抗仮説は棄却**
  （中央値 0.893，最小 0.845）．**「5 対 5 のメタクラス分割は p≫n で訓練上分離できても汎化しない」
  という対抗仮説は成り立たない**．各列の二値問題は十分に汎化しているのに，それらを最尤復号で
  束ねると全体の top1 が下がる．

**(4) 判定**

足切り 1・2 とも不通過のため**本走（wafl500〜509，3,750 問）は実施していない**．
判定語の付与と機序の解釈は分析・考察フェーズの担当とする．
なお wafl-ctrl5 の `~/expert-mesh-iter95/` に `ecoc_head.py` と更新版スクリーニングスクリプトを
配置済みで，再現実行が必要ならそのまま使える．

### Iteration 98 実行済み —— 判定 `closed`（足切り不通過・本走なし．value は「反証済み」）

**変更（このイテレーションで実際に触ったもの）**

- 新規: `ecoc_head.py`（`EcocLogLossClassifier`，本番経路へは**未配線**），
  `scripts/screen_classifier_multiclass_decomposition.py`（書き直し．腕 A/腕 C の 3 seed × 固定 5-fold），
  `results/iter98_screening/screening_result.json`（生データ全件）．
- **無変更**: `models/`・`data/dataset.jsonl`・`config.yaml`・`scripts/train_domain_classifier.py`・
  `Dockerfile`・実機ノード wafl500〜509（`git diff --stat` で実測確認済み）．

**結果（要点のみ．詳細は上記「実装・実験 (Iter98)」）**

- CV top1: 腕 A 0.802637（Iter95/96/97 と厳密一致＝測定系は健全）／腕 C 0.788425，**Δ = −1.421pt**．
  Δ の SE ≒ 0.22pt に対し **6.5 SE 相当**で，符号も足切りと逆．全体 McNemar は併合 p=2.81e-9，
  保守 p=0.00060 で**いずれも有意に悪化**．ノイズではない．
- BH 後の有意退行は併合 3 件／保守 2 件（education −7.00pt，social_science −6.40pt，general −2.67pt）．
- 予測の当落: P1 落選・P2 落選・P3 当選・P4 当選・**P5 当選（＝対抗仮説の棄却）**．

**判定: `closed`（`rejected` ではない）．ただし value `error_correcting_output_codes` は「反証済み」**

- `rejected` は本走の 2 軸表（Δtop1 × McNemar p）に基づく判定語であり，本走をしていない今回は
  どのセルにも到達していない（B159 (A)・B161 (a)・B163 (a) と同一論点．判定語集合の拡張は
  記録スキーマの破壊的変更なので自動決定しない）．
- 一方で value の扱いは Iter95/96 型（「Δ は正だが足切りに届かない＝未通過・再試行の余地あり」）ではなく，
  **Iter97 型（符号ごと逆・保守側でも有意＝反証済み，再試行しない）**とする．
  さらに今回は下記 M6 により**失敗の機序まで説明が付いている**ので，符号長 L や復号方式を変えた
  再試行にも見込みが無い（M6 の予測 (3)）．

**機序の再定式化 M6: 「分解の失敗」ではなく「分解で新しい情報が増えない」**
（B163 (d) の読み＝標本分割・非適格分類器は **P2 落選により棄却**．以下で差し替える）

事前登録した診断出力だけで 3 手順の検証が完結した（開発ホスト，CPU 数秒．実機不使用）．

1. **独立誤りの仮定を置いた場合の予測値と実測の乖離**．34 列の二値 CV 正解率（平均 0.8903）の誤り率で
   **ビット誤りが列間で独立**に起きるとした Monte Carlo（20 万行，`random_state=98` の符号行列を
   再生成．行間最小 Hamming 距離 14 を再現）では **top1 = 0.9997**，1 行あたり平均ビット誤り 3.73 で
   訂正能力 6 ビットに十分収まる．**実測は 0.7884**．
   → 予測どおりなら ECOC は圧勝するはずで，**列間のビット誤りは独立でない**．
2. **ビット誤りは誤答行に全量集中している（恒等式による検証）**．
   「誤答行では観測符号語が競合クラスの符号語へ**丸ごと**移り，正答行のビット誤りは 0」と仮定すると，
   期待される列誤り率 = (ECOC の誤答率 0.21158) × (平均行間 Hamming 距離 17.378 / 34) = **0.1081**．
   実測の平均列誤り率は **0.1097**（乖離 1.5% 相対，1 行あたり 0.055 ビット分）．
   → 34 本の二値分類器は「独立にときどき間違える弱学習器」ではなく，**同じ行で足並みを揃えて
   丸ごと隣のクラスへ倒れている**．符号の冗長性（6 ビットまで訂正）は名目値で，実際に起きる誤りは
   7 ビット以上（典型 17 ビット）の符号語置換なのでこの符号長では原理的に訂正できない．
3. **腕 A と腕 C の誤答行は 9 割方同じ行である（AGREE の実測）**．併合 n=6,825 で腕 A の誤答 1,347 行・
   腕 C の誤答 1,444 行，共通 1,265 行．**腕 C の誤答の 87.6%・腕 A の誤答の 93.9% が共通**．
   → Rifkin & Klautau (2004) の「points become errors **not because of deficiencies in the method of
   combining binary classifiers**」（AGREE が誤り率を上回る）が本データで**そのまま再現**した．

**M6 の言明**: 5,120 次元の埋め込み上では，誤りは**行（＝入力表現）の性質**であって決定則の性質ではない．
どの分解方式も同じ行で同じ向きに倒れるため，**符号の冗長性に注入できる独立な情報が存在しない**．
その状態で分解すると，softmax が学習していた**クラスごとの重み付き読み出し**が，34 列を等重みで足す
**固定の読み出し**に置き換わる分だけ情報が減る．これが Δ = −1.42pt の中身である．

**M6 の系（既存データでの裏づけ）**

- 損失は**事後分布が平坦なクラス**に集中する．腕 A の recall と Δ の Spearman = **+0.596（p=0.069，n=10）**
  で，高 recall 側（mathematics 0.948 → +0.40pt，computer_science 0.920 → +0.40pt）は動かず，
  低 recall 側（education 0.535 → −7.00pt，social_science 0.788 → −6.40pt，general 0.799 → −2.67pt）が
  落ちる．正側合計 +3.97pt に対し負側合計 −16.36pt で**正味の損失**（零和の再配分ではない）．
- 失った質量の**行き先が意味的隣接クラスでない**．education −50 行の行き先は
  business_economics +17・legal +17・history_culture +6・natural_science +6 と分散し，
  social_science −48 行は legal +18・business_economics +17・general +9・**computer_science +8** と
  分散する．意味的近傍への流出（Iter97 で読んだ像）ではなく，**平坦な事後分布を任意の固定読み出しで
  割り振った結果の拡散**である．
- P4（ECE 0.01650 ≤ 0.01971）が当選しているのに top1 が下がる点も M6 と整合する．復号スコアは連続で
  較正も壊れていない（P3 も当選）．**壊れているのは確率の形ではなく，どの行をどのクラスへ割るかという
  情報そのもの**である．

**M6 が生む反証可能な予測（今後この軸を再訪するなら，これらを外した証拠が必要）**

1. `random_state` を変えて別の dense random code を引くと，**退行するドメインの顔ぶれは入れ替わるが
   全体 Δ は −1.4pt 前後のまま**（本反復で退行した education/social_science と，Iter97 の OvO で
   退行した history_culture が入れ替わった事実の一般化）．
2. 符号長を L=34 → 68 に伸ばしても **top1 の変化は ±0.3pt 以内**（ビット誤りが独立でない以上，
   冗長性を足しても訂正できる誤りが増えないため）．
3. OvA・OvO・ECOC のどれを使っても，softmax との誤答行の AGREE は **85% 以上**のまま．
   （以上は**実行しない**．実行する価値が無いことを示すのが M6 の役割である．）

**P2 の内訳の非対称性（history_culture 復帰・education/social_science 悪化）の読み**

Iter97 と Iter98 で退行するドメインが入れ替わったこと自体が M6 の予測 1 の直接の観測である．
OvO では 45 対の組み方，ECOC では 34 列の符号がそれぞれ「平坦な事後分布の行をどこへ倒すか」を決める
**任意の固定読み出し**であり，どのクラスが割を食うかは分解の実装詳細に依存する．
したがって「history_culture の退行が ECOC で解消した」ことを**改善の証拠として読んではならない**
（同時に education が新規に −7.00pt 落ちている）．**共通しているのは「低 recall クラスが損をする」
という一点だけ**であり，これは M6 の系そのものである．

**決定層という軸の総括: 打ち止め（closed）**

- 事前登録では「P2 通過かつ P1 不通過なら打ち止め」としていたが，実際は **P2 も落選**した．
  事前登録の分岐は「軸を閉じる唯一の経路は『退行は OvO 固有だった』と示すこと」という前提で書かれており，
  この前提が誤っていた．**実測はより強い経路で軸を閉じる**: 分解方式に依らず誤答行が共通（AGREE 87.6%）で，
  ビット誤りに独立成分が無い（検証 1・2）以上，**どの分解方式を持ってきても softmax を上回れない**．
  「OvO 固有か否か」は，軸を閉じる根拠として不要になった．
- Rifkin & Klautau (2004) の主張（基底学習器が正則化されていれば分解方式は等価）との整合:
  **本データは同論文の側に 2 反復連続で落ちた**．ただし厳密には「等価」ではなく **softmax が 0.8〜1.4pt
  優る**（OvO −0.806pt，ECOC −1.421pt）．同論文の主張は「OvA が他に劣らない」であって「分解が単一
  多項モデルに劣らない」ではないので，矛盾ではない．本研究の条件（p=5,120 ≫ n=2,275，L2 正則化 C=1.0）
  では，単一の多項 softmax が持つ**クラス間で結合した正規化**が，分解では復元できない情報を担っている，
  と読むのが実測に最も忠実である．
- 以上より **`classifier_multiclass_decomposition` は closed**．同レバーの残 value は無く，再訪もしない．

**学び**

1. **「部分問題が解けている」ことは「分解が有効である」ことの証拠にならない**．P5（列ごと二値 CV 正解率
   中央値 0.893）は事前には ECOC 成功の前提条件として登録したが，実際には**分解が無意味であることの
   証拠**だった．二値精度 0.89 は，softmax の top1 0.803 から機械的に導かれる値（誤答 19.7% のうち
   競合クラスが分割の反対側に来る確率が約半分 → 約 0.90）とほぼ一致する．つまり 34 本の二値分類器は
   新しい識別面を 1 つも作っておらず，**同じ 10 値決定を 34 ビットに再符号化していただけ**である．
   今後「分解して部分問題の精度を見る」型の診断では，**部分問題精度の期待値を基準線から先に計算して
   おく**こと（これをやっていれば計画時点で ECOC の見込みの薄さを数値で示せた）．
2. **誤り訂正符号の訂正能力は，誤りの独立性の仮定の上でしか意味を持たない**．最小 Hamming 距離 14・
   6 ビット訂正という設計値は，本データでは一度も働いていない（実際の誤りは典型 17 ビットの符号語置換）．
   ECOC を検討する場面では，**符号長や最小距離ではなく「列間の誤り相関」を先に測る**．測り方は本反復の
   検証 2 の恒等式（列誤り率 ≒ 行誤り率 × 平均行間 Hamming 距離 / L）で，**追加実験なしに既存の
   診断出力だけで判定できる**．この恒等式が成り立ってしまったら訂正能力は幻である．
3. **AGREE（腕間の誤答行の重なり）は，軸を閉じる/開くの判断に直接使える指標である**．今回 87.6% /
   93.9% という値が，「決定層をこれ以上いじっても無駄で，残る説明変数は入力表現である」という結論を
   1 つの数字で支えた．**McNemar の discordant 数から追加計算 0 で求まる**（共通誤答 = 腕 C の誤答数
   − discordant_a_only）ので，今後は腕比較のたびに既定で併記する．
4. （運用）事前登録した分岐（「P2 通過かつ P1 不通過なら打ち止め」）が，実測が示した経路と食い違うことが
   起こりうる．そのときは**事前登録の分岐に機械的に従うのではなく，前提が崩れた事実を明記したうえで
   実測に即した結論を書く**．事前登録は結論を縛るためではなく，事後の都合の良い解釈を防ぐためにある．

**次イテレーション（Iter99）の方針**

`classifier_multiclass_decomposition` が closed となり，config.yml の既存 levers に未試行 value は無い．
M6 の結論（誤りは決定則ではなく**入力表現**の性質）と，B115(3) のユーザー指示による優先順位
（複合評価集合の拡充＝Iter78 で完了・730 行へ拡大済み ＞ **埋め込みモデルの差し替え** ＞ 全ドメイン共通
ルールでの訓練データ拡充）が同じ方向を指す．そこで `embedding_model_replacement` に新 value
**`japanese_specialized_ruri_v3_310m`** を追記し，Iter99 のレバーとする（B165）．
`iteration_name` = **「埋め込みを日本語特化モデル ruri-v3-310m へ差し替える」**．
根拠は Iter79 の調査で記録済みの JMTEB Classification（`cl-nagoya/ruri-v3-310m` **78.66** vs
`Qwen3-Embedding-0.6B` 66.09，出典 hotchpotch 2025-06-11）．当時は「instruction prefix が必須で
2 レバー目になる」ため候補外としたが，**Iter81 で prefix 機構が，Iter82 で 2 view 連結が既に本番に
入っている**ので，その除外理由は現在では成立しない．
**必須の申し送り**: (i) 現行の埋め込みは各ノードの Ollama `/api/embeddings` 経由（`config.yaml:4`）であり，
ruri-v3-310m が Ollama で取得可能かは未確認である．Iter79/89 と同型の **G0（実現性ゲート）を計画段階に
必ず置く**こと．(ii) G0 不合格なら事前登録した代替 `multilingual_e5_large`（JMTEB Classification 72.89，
`query: ` prefix 必須だが prefix 機構は既存）へ値を切り替える（この 2 値のみ `values` へ追記済み）．
(iii) 次元が変われば `embedding_view_concat` の連結次元も変わるため，`train_domain_classifier.py` の
再訓練は差し替えに構造的に付随する作業であり別レバーではない（Iter79/89 で確立した型）．

## Iteration 97: 分類器の多クラス分解を softmax から一対一へ変える

### 調査 (Iter97)

本フェーズでも**実験ノード wafl500〜509 には一切触れていない**（2026-09-23 絶対条件 (B)）．
開発ホスト上で行ったのは，リポジトリのコード・JSONL の読み取りと，**合成データによる
sklearn の API 実現性の確認（CPU のみ・数秒）**だけである．`models/`・`data/dataset.jsonl`・
`results/` への書き込みは 0 件．

**(1) 問い**（B161 (g) の申し送りを受けて設定した）

- Q1: 「10 クラスが 1 つの softmax と 1 組の線形境界を共有する」という現在の定式化の**外側**に，
  10 ドメイン一律の規則として実装できる代替の定式化はあるか．
- Q2: その代替は，Iter95/96 で観測された**零和的な再配分**（機序 M5）を構造的に回避できるか．
- Q3: 現行の較正済みパイプライン（`CalibratedClassifierCV(method="temperature")` ＋
  `sample_weight = n/(K*n_d)`）と，推論経路 `classifier.py:estimate_confidence_classifier()` の
  「`predict_proba` が 10 ドメインで和 1」という前提を壊さずに実装できるか．

**(2) 文献調査（tvly search / extract）**

- **Fürnkranz (2002), *Round Robin Classification*, JMLR 2:721–747**,
  <https://www.jmlr.org/papers/volume2/fuernkranz02a/html/node3.html> ——
  c クラス問題を c(c−1)/2 個の対ごと二値問題へ分解する方式．引用すると「in the round robin case,
  the base classifier uses fewer examples and thus has **more freedom for fitting a decision
  boundary** between the two classes」「**pairwise decision boundaries can be considerably
  simpler** than those originating from unordered binarization」であり，実例として
  **Knerr et al. (1992)** の数字認識（クラスは対ごとには線形分離可能だが one-against-all は
  単層ネットで解けなかった）を挙げる．さらに **Hsu & Lin (2002)** が
  「**線形カーネル SVM でこそ非線形カーネルより大きな OvO の優位**を得た」ことを引き，
  その理由を対ごと境界の単純さに帰している．**本研究の分類器は線形ヘッドであり条件が一致する**．
- **Galar, Fernández, Barrenechea, Bustince & Herrera (2011), *An overview of ensemble methods
  for binary classifiers in multi-class problems: Experimental study on one-vs-one and one-vs-all
  schemes*, Pattern Recognition 44(8):1761–1776**, <https://sci2s.ugr.es/ovo-ova> ——
  SVM・決定木・kNN 等の基底学習器を横断した実験比較で，OvO が OvA を上回る傾向を報告する．
  同グループのチュートリアル（<https://sci2s.ugr.es/sites/default/files/files/TutorialsAndPlenaryTalks/
  SSTiC-Trends%20in-Classification-Imbalanced-data-sets.pdf>）は，**多クラス不均衡への対処として
  pairwise learning を明示的に位置づけている**（各二値問題が 2 クラスだけを見るため，
  多数クラス全体を相手にする OvA より不均衡が緩む）．本研究は legal 77 行 対 250 行 ×8 という
  多クラス不均衡を抱えるので，この論点も該当する．
- **反証側: Rifkin & Klautau (2004), *In Defense of One-Vs-All Classification*, JMLR 5:101–141**,
  <https://www.jmlr.org/papers/volume5/rifkin04a/rifkin04a.pdf> ——
  「a simple one-vs-all scheme is **as accurate as any other approach**, assuming that the
  underlying binary classifiers are well-tuned」と主張する．**効果が出ない可能性も文献上
  同程度に支持されている**ことを明記しておく．本レバーは片側に寄った改良案ではなく，
  両論ある仮説の検定である．
- **Wu, Lin & Weng (2004), *Probability Estimates for Multi-class Classification by Pairwise
  Coupling*, JMLR 5:975–1005**（ <https://www.jmlr.org/papers/volume5/wu04a/wu04a.pdf> ，
  書誌は <https://stat.nccu.edu.tw/en/members/journalpaper/T-F-Wu-C-J-Lin-R-C-Weng-2004-Probability-Estimates-for-Multi-class-Classification-by-Pairwise-Coupling-Journal-of-Machine-Learning-Research-Vol-5-pp-975-1005-SCIE-10278537> ）
  —— 対ごと二値出力から多クラス確率を作る標準的な手続き（libsvm の実装根拠）．
  **本反復ではこれを自前実装せず**，sklearn の `OneVsOneClassifier.decision_function` を
  既存の temperature 較正に通す（下記 (3) で和 1 を実測確認済み）．自前実装が必要になるのは
  較正後の確率が退化していた場合だけで，その判断はスクリーニングのデータで行う．
- 本リポジトリでの既往: Iter59 前後で **OvR（binary relevance）を rank_2 以降の並べ替え用
  スコアとして**使った例はある（journal_archive の該当節）が，**argmax を決める多クラス分解
  そのものを softmax 以外にした実験は 1 度も無い**（`OneVsOne` の grep ヒット 0 件）．

**(3) 実現性の実測（開発ホスト，CPU のみ，合成データ）**

- `CalibratedClassifierCV(OneVsOneClassifier(LogisticRegression(max_iter=1000)),
  method="temperature", cv=5, ensemble=True)` は sklearn 1.9.0 で fit でき，
  `predict_proba` は (n, 10) で**各行の和が 1.0**（`classifier.py` の前提を満たす）．
- **重大な落とし穴（実測）**: 上記へ `sample_weight` を渡すと
  `UserWarning: Since OneVsOneClassifier does not appear to accept sample_weight, sample weights
  will only be used for the calibration itself.` が出て，**base estimator への重みが黙って
  捨てられる**．そのまま走らせると『分解方式の変更』と『クラス均衡重み（B60）の喪失』を同時に
  変えることになり，単一レバー原則が壊れる．
  `sklearn.config_context(enable_metadata_routing=True)` の下で
  `LogisticRegression(...).set_fit_request(sample_weight=True)` を付けると，45 本すべての
  対ごと LR へ重みが流れ**警告は出なくなる**ことを確認した．
- 計算量: 二値 LR 1 本（400 行 × 5,120 次元）の fit は **0.06 秒**，単一 softmax
  （1,820 行 × 5,120 次元）は **1.0 秒**．45 本でも同オーダーで，スクリーニング全体が
  wafl-ctrl5 の CPU で完結する（**再埋め込み 0 回・GPU 不要**）．
- なお対ごとの重みは，クラス d の行が `n/(K*n_d)` を持つため**どの対でも両クラスの総重みが
  n/K で等しく**なる．すなわち OvO へ移しても『クラス均衡』の意味は保たれ，変わるのは
  「境界を 1 組の共有ベクトルで張るか，対ごとに独立に張るか」だけである．

### 仮説 (Iter97)

**education（CV recall 0.5350）・medical（0.6733）の誤りが減らず，かつドメイン間で零和的に
再配分されるのは，訓練データの量や重みの問題ではなく，10 クラスが 1 つの softmax 正規化と
1 組の線形境界を共有していることに由来する（M5 の構造的な言い換え）．**
対ごとに独立な二値問題へ分解すれば，education 対 medical のような紛らわしい対の境界は
他 8 クラスの事情から解放され，Fürnkranz (2002)・Hsu & Lin (2002) の言う「線形でこそ効く
対ごと境界の単純さ」が得られるはずである．予測される観測は，**弱いクラス（education・medical）の
recall が上がり，かつ強いクラスの退行が Iter95/96 のようには生じない**ことである
（零和なら total は動かないが，分解が効くなら total が動く）．

### 単一レバー (Iter97)

- **レバー**: `classifier_multiclass_decomposition` = `one_vs_one_pairwise_coupling`
  （config.yml の `levers` 末尾に本フェーズで追記．B162 で `[auto-decided]`）
- **何を何から何へ**: `scripts/train_domain_classifier.py:train_classifier()`（L210 付近）の
  base estimator を
  **`LogisticRegression(max_iter=1000, class_weight=None)`（単一 softmax）から
  `OneVsOneClassifier(LogisticRegression(max_iter=1000, class_weight=None))`（45 本の対ごと二値 LR）**へ．
  `CalibratedClassifierCV(method="temperature", ensemble=True, cv=5)` でラップする点，
  `sample_weight` の式，訓練行はそのまま．**変えるのはこの 1 箇所だけ**．
- **レバーを読むコード行と到達条件（d0004 §4 の恒久対策）**:
  - 変更箇所は `scripts/train_domain_classifier.py:train_classifier()` の base estimator 生成行
    （現行 `base_estimator = LogisticRegression(max_iter=_MAX_ITER, class_weight=None)`）．
    この関数は再訓練時に必ず通る（スクリーニングの
    `scripts/screen_composition_preserving_volume_expansion.py` も同関数を import している）．
  - 実行時経路 `classifier.py:estimate_confidence_classifier()` は
    `models/domain_classifier.joblib` を読んで `predict_proba` を呼ぶだけで，**コード変更は不要**．
    したがって「設定が読まれない no-op」は原理的に起こりえない代わりに，
    **artifact の差し替えとデプロイが唯一の到達条件**である．
    本走前に (a) 新 artifact の sha256 が旧と異なること，(b) `n_features_in_` == 5120，
    (c) 10 ドメインの `predict_proba` の行和が 1.0，(d) 各ノード上の
    `models/domain_classifier.joblib` の sha256 が一致すること，を必ず確認する．
- **固定する構成（1 つも動かさない）**: 訓練集合 `data/classifier_train_iter94_dedup.jsonl`
  （2,275 行）・`sample_weight = n/(K*n_d)`・較正 `temperature`・`C=1.0`・`max_iter=1000`・
  埋め込みモデル `qwen3-embedding:4b`・`embedding_instruction`（Iter81 の P1 文言）・
  `embedding_view_concat: true`（5,120 次元）・評価集合 `data/dataset.jsonl`
  （sha256 `2114e048...`，3,750 行，読むだけ）・`dispatch_gap_threshold: 0.36`・
  `dispatch_gap_max_k: 4`・`dispatch_top_k: 2`・`aggregation_method: max_confidence`・
  expert/light モデルとノード割り当て．

### 事前スクリーニング設計（事前登録 / Iter97）

Iter95/96 で確立した型をそのまま踏襲する．**足切りを通った場合は本走を省略しない**
（2026-09-23 絶対条件 (A)）．実施場所は **wafl-ctrl5**（絶対条件 (B)．
`~/expert-mesh-iter95/` の uv 3.12 環境と `data/embcache_iter96_scaled.npy` の先頭 2,275 行を流用．
**新規の埋め込み計算は 0 回**）．

- **主評価**: 既存 2,275 行の層化 5-fold（`StratifiedKFold`，seed = 96 / 196 / 296 の 3 seed，
  併合 n = 6,825）．テスト fold は両腕で完全に同一．
  **腕 A = 現行 softmax ヘッド（Iter96 実測 CV top1 = 0.802637 と一致することを確認すること）**，
  **腕 B = OvO ヘッド**．訓練行・重み・埋め込みは両腕で同一で，違いは base estimator だけ．
  スクリプトは `scripts/screen_composition_preserving_volume_expansion.py` を雛形にし，
  「追加行」の概念を落として base estimator を切り替える形に書き換える
  （統計は `metrics.py` の既存実装 `compute_mcnemar_test` /
  `compute_domain_recall_mcnemar_test` / `apply_benjamini_hochberg` をそのまま使う）．
- **併記する診断値（判定には使わないが必ず出す）**: 両腕の
  (a) per-domain recall（腕 A の実測値は下表），(b) top-2 確率差 `gap` の分布と
  **`gap < 0.36` の行の割合**（dispatch 候補数が動くかの事前把握），
  (c) `sample_weight` 警告の有無（1 件でも出たら**実験不成立として中断**）．

腕 A の per-domain CV recall（Iter96 スクリーニング実測．`results/iter96_screening/screening_result.json`）:
mathematics 0.9480 / computer_science 0.9200 / natural_science 0.8760 / history_culture 0.8524 /
legal 0.8398 / business_economics 0.8160 / general 0.7987 / social_science 0.7880 /
medical 0.6733 / **education 0.5350**．

### 反証可能な予測（事前登録 / Iter97）

- **P1（仮説の核心）**: 弱い 2 クラス（education・medical）の CV recall Δ の合計が **+3.0pt 以上**．
  これが 0 付近なら「共有境界が律速」という読みは誤りで，分解では触れない領域の問題である．
- **P2（零和性の破れ）**: CV top1 の Δ が **+1.0pt 以上**．Iter95（+0.366pt）・Iter96（+0.322pt）は
  いずれも零和で相殺された結果なので，分解が効くならここが初めて動く．
- **P3（不均衡の緩和）**: legal（77 行）の recall が退行しない（Δ ≥ −1.0pt）．
  Galar らの言う pairwise learning の不均衡緩和が効くなら，むしろ上がる側に出る．
- **P4（確率の健全性）**: 較正後 `predict_proba` の行和が全行 1.0 ± 1e-6 で，
  `gap < 0.36` の行の割合が腕 A から **±5pt 以内**に収まる．
  これを外れる場合は dispatch 候補数が変わるため，本走の複合設問指標の解釈に注釈を付ける．

### 成功条件・非退行条件（事前登録 / Iter97）

**基準線（本走）**: `results/20260928_160921/` の top1 = **0.833067**（3,750 問，
Wilson 95%CI [0.8208, 0.8447]）．再現性の床は ±0.25pt．
**基準線（CV）**: 腕 A = 0.802637．CV の Δ のノイズ幅は，Iter95（Δ=+0.366pt, p=0.1595）・
Iter96（Δ=+0.322pt, p=0.1415）から **SE ≒ 0.22pt** と見積もる（足切り +1.0pt は約 4.5 SE）．

- **スクリーニングの足切り（両方を満たさなければ本走せず `closed`）**:
  1. **CV top1 Δ (B−A) ≥ +1.0pt**（Iter95・Iter96 と同一水準を据え置く．
     水準そのものの一般則は B154 (A3) として未回答のまま）．
  2. **per-domain recall の BH 補正後（q=0.05，10 指標）有意な退行が 0 件**．
  - 外した場合は `models/`・`config.yaml`・`data/dataset.jsonl` を**1 バイトも変更せず**
    結果だけ記録してフェーズ 3 へ渡す（Iter95・Iter96 と同じ運用．絶対条件 (A) 非抵触）．
- **本走（足切り通過時のみ）**: `models/domain_classifier.joblib` を OvO 版で再訓練
  （旧 artifact は `models/domain_classifier_pre_iter97_ovo.joblib` へ退避），`mise run deploy` と
  sha256 確認のうえ，**wafl500〜509 で 3,750 問フルスペック**を 1 回実行する．
- **本走の判定**: 全体 top1 が **Δ ≥ +0.5pt かつ McNemar p < 0.05** で `adopted`，
  Δ ≥ +0.5pt だが p ≥ 0.05 なら `adopted_small`（要再現），|Δ| < 0.25pt なら `negligible`，
  Δ ≤ −0.5pt なら `rejected`．
- **必須の非退行条件（1 つでも破れたら artifact をロールバック）**:
  - C1: per-domain precision/recall 計 20 指標の BH 補正後の有意退行が **0 件**．
  - C2: `used_fallback` 率・`dispatch_failed` 件数が基準線から増えない．
  - C3: レバー発火の証拠（新 artifact の sha256 が旧と異なる，`n_features_in_`=5120，
    ノード上の sha256 一致，`predict_proba` の行和 1.0）．
  - C4（参考値）: Random / BestSingle / Oracle を併記し BestSingle 超過を明示（success_criteria (3)）．
  - C5: 複合設問 730 行の top1（基準 0.7904）が −1.0pt を超えて下がらないこと，
    および `compound_domain_set_recall`・`compound_mean_dispatched_count` の併記．
  - C6: `mean_duration_ms` が基準線の +20% 以内（推論は 1 回の `predict_proba` のままなので
    実質不変のはずだが，確認する）．

### 実行フェーズへの申し送り（Iter97）

- **`sample_weight` の metadata routing を必ず入れること**（上記 (3)）．警告が出た時点で中断．
  これを怠ると「分解方式」と「クラス重み」の 2 レバーを同時に動かした実験になる．
- スクリーニングは wafl-ctrl5 で完結する（再埋め込み 0 回・GPU 不要）．
  wafl500〜509 は**本走以外で一切使わない**（絶対条件 (B)）．
- `dispatch_gap_threshold` の再較正は別レバーであり，本反復では**絶対に触らない**．
  gap 分布が動いた場合も，記録に留めて次イテレーションの候補とする．
- 却下した代替案と理由は backlog B162 に記録した（top-2 だけを対ごとに並べ替える 2 段階案・
  非線形ヘッド・LLM による入力表現の拡張・ラベル定義の変更・複合評価集合の拡充）．

### 実験 (Iter97)

**本走は実施していない**．事前登録した足切り 2 条件をいずれも満たさなかったため，停止規則どおり
本番コード（`scripts/train_domain_classifier.py`・`config.yaml`・`data/dataset.jsonl`・`models/`）を
1 バイトも変更せずフェーズ 3 へ引き渡した（Iter93・Iter95・Iter96 と同一運用）．
実機ノード wafl500〜509 は不使用．スクリーニングは wafl-ctrl5 の `~/expert-mesh-iter95/` 環境で完結し，
再埋め込み 0 回・GPU 不要で実行した．

**単一レバー原則の担保（最重要の事前条件）**: 腕 B は
`sklearn.config_context(enable_metadata_routing=True)` ＋
`LogisticRegression.set_fit_request(sample_weight=True)` を入れたうえで，全ての `fit()` 呼び出しを
warning ガードで包み「警告が 1 件でも出たら `SampleWeightDroppedError` を送出して中断」する実装とした．
**警告は一切発生せず**，45 本すべての対ごと LR に `sample_weight` が到達したことを確認した．
すなわち本実験は「クラス均衡重み（B60）の喪失」との 2 レバー同時変更にはなっていない．

**設計**: 3 seed（96/196/296）× 固定 5-fold，n=2,275（併合 6,825）．
腕 A は `train_classifier()` を無変更のまま import して呼び出し，腕 B は `_train_ovo_classifier()`
（`CalibratedClassifierCV(OneVsOneClassifier(LogisticRegression(...)), method="temperature", cv=5,
ensemble=True)`）とした．生データは `results/iter97_screening/screening_result.json`．

**結果**

| 指標 | 値 | 事前登録した足切り | 判定 |
|---|---|---|---|
| 腕 A CV top1 | 0.802637 | —（Iter96 実測と厳密一致） | 測定系は健全 |
| 腕 B CV top1 | 0.794579 | — | — |
| **CV Δ (B−A)** | **−0.806pt** | ① Δ ≥ +1.0pt | **不通過** |
| 全体 McNemar | discordant 138 / 83，χ²=13.19，**p=0.00028** | — | **B が有意に悪い** |
| BH 後の有意退行 | **2 件** | ② 0 件 | **不通過** |

per-domain recall Δ: history_culture −2.86pt（p=0.000144，BH 有意退行），
social_science −4.53pt（p=3.06e-6，BH 有意退行），education +2.24pt（p=0.0648，非有意），
medical −1.87pt（p=0.0303，BH 補正後は非有意），legal ±0.0pt．

**事前登録した予測の照合**: P1（education+medical の recall Δ 合計 ≥ +3.0pt）は実測 **+0.374pt で反証**．
P3（legal recall Δ ≥ −1.0pt）は ±0.0pt で通過．P4（`gap<0.36` の行割合が ±5pt 以内）は
腕 A 20.1% → 腕 B 8.3%（**Δ = −11.8pt**）で**逸脱**した．なお `predict_proba` の行和は両腕とも 1.0 で，
確率化そのものは健全である（Wu-Lin-Weng の自前実装を避け sklearn の temperature 較正で代替した判断は妥当だった）．

**検証**: `ruff check` 通過，`py_compile` OK，`git diff --stat scripts/train_domain_classifier.py` 差分なし，
`git status --short config.yaml models/ data/dataset.jsonl` 差分なし（無変更を実測確認）．

**分析フェーズへの申し送り**: (i) 文献の反証側 Rifkin & Klautau (2004) *In Defense of One-Vs-All* が
今回の実測と整合した．両論ある仮説の検定として事前に反証側を明記していたため，この結果は解釈可能である．
(ii) history_culture・social_science という**強クラス側**で有意退行が出た機序（対ごと分解が強クラス間の
境界をむしろ不安定化した可能性，および P4 の gap 分布の大幅な収縮との関係）は考察対象として残る．
(iii) 予備値 `error_correcting_output_codes`（B162 (C)）を次点として引くかはフェーズ 3 の判断とする．

### Iteration 97 実行済み

**変更したもの**: 事前スクリーニングのみ．`scripts/screen_classifier_multiclass_decomposition.py`（新規）と
`results/iter97_screening/screening_result.json`（新規）の 2 件だけである．
事前登録した足切り 2 条件をいずれも満たさなかったため停止規則どおり本走を行わず，
`scripts/train_domain_classifier.py`・`config.yaml`・`data/dataset.jsonl`・`models/` は 1 バイトも
変更していない．基準線 top1 = 0.833067 は不変．実機ノード wafl500〜509 は不使用．

**結果（再掲）**: 腕 A（現行 softmax）CV top1 = 0.802637（Iter96 実測と厳密一致＝測定系は健全），
腕 B（OvO）= 0.794579，**Δ = −0.806pt**．全体 McNemar discordant 138/83，χ²=13.19，p=0.00028．
BH 後の有意退行 2 件（history_culture −2.86pt，social_science −4.53pt）．
P1 反証（+0.374pt < +3.0pt），P2 不通過，P3 通過，P4 逸脱（`gap<0.36` の行割合 20.1%→8.3%）．
`sample_weight` の警告 0 件で，45 本全ての対ごと LR に重みが到達した（単一レバー原則は担保）．

#### ノイズか信号か

Δ = −0.806pt は，Iter95（+0.366pt, p=0.1595）・Iter96（+0.322pt, p=0.1415）から見積もった
**CV の Δ の SE ≒ 0.22pt の約 3.7 倍**であり，かつ同一 fold の対比較（McNemar p=0.00028）でも
有意に悪い．**ノイズではなく，負方向の信号である**と判断する．

ただし**検定の楽観性を明記しておく**．3 seed × 5-fold の併合 n=6,825 は，実体としては同じ 2,275 行を
3 回数えたものであり，行の独立性を仮定した McNemar は反保守的である．discordant を seed 数で割った
保守側の再計算では，全体 46/28 で χ²=4.56・**p=0.033**（依然有意），social_science は p=0.0055，
history_culture は p=0.020 となる．すなわち**「腕 B が悪い」という全体の結論は保守側でも保たれる**が，
**足切り 2（BH 後の有意退行 0 件）の当落は保守側では入れ替わりうる**（BH q=0.05・10 指標では
最小 p=0.0055 が閾値 0.005 をわずかに超えるため有意 0 件になる）．
今回は**足切り 1 が符号ごと逆方向に外れている**ため，この揺らぎは判定に影響しない．
なお同じ楽観性は Iter95・Iter96 の足切り 2 の評価にも及ぶので，**次回以降は seed 併合 McNemar を
主，seed 除算した保守値を併記**すること（学び 4）．

#### 判定: `closed`（ただし当該 value は再試行不可）

`rejected` は**本走の 2 軸表（Δtop1 × McNemar p）に基づく判定語**であり，本走が無い以上どのセルにも
到達していない．したがって Iter93・Iter95・Iter96（B159 (A)・B161 (a)）と同じく **`closed`** とする．
新しい判定語は設けない（判定語の集合を増やすことは記録スキーマの破壊的変更であり，1 イテレーションの
都合で自動決定すべきでない．必要なら人間判断で導入する — B163 (A)）．

その代わり，**同じ `closed` でも今回は先行 3 例と性質が異なる**ことを記録に残す．
Iter95・Iter96 は「Δ は正だが足切りに届かない」であり，検出力を上げれば再試行の余地があった．
今回は **Δ が負で，かつ有意**である．したがって value `one_vs_one_pairwise_coupling` は
**単なる未通過ではなく反証されたものとして扱い，再試行しない**（config.yml の当該 note に注記した）．

#### 考察 (i): なぜ強クラス側（history_culture・social_science）が退行したのか

退行は「強いクラスが一律に損をした」のではない．腕 A の recall と Δ の**単調性は完全に消えている**
（Spearman ρ = **+0.018, p = 0.96**．Iter96 は ρ = −0.717, p = 0.020 だった）．
natural_science（0.876）は +0.67pt，mathematics（0.948）は −0.53pt にすぎず，
損失は **social_science −4.53 / history_culture −2.86 / medical −1.87** の 3 クラスに集中している．
これらは**意味的な近傍を多く持つクラス**（social_science は general・education・business_economics と，
history_culture は general・social_science と重なる）であり，mathematics・computer_science のような
孤立したクラスは動いていない．

OvO でこの分布になる機序として，文献上よく知られた次の 2 つが本件の条件と噛み合う．

- **(a) 対ごと二値問題の分散増大（p≫n の悪化）**: 各対ごと LR は 2 クラス分の行（約 250+250=500 行）
  しか見ないのに次元は 5,120 のままである．単一 softmax の `w_d` が 2,275 行すべてから推定されるのに対し，
  対ごと境界は**約 1/4.5 の標本で推定される**．近傍クラス間の境界ほどこの分散の影響が大きい．
- **(b) 非適格分類器（non-competent classifier）の票**: ある行の真クラスが d のとき，
  45 本のうち d を含むのは 9 本だけで，残る 36 本は d を知らないまま票を投じる．
  票数で argmax を決める以上，近傍クラスに 1 本負けただけで順位が入れ替わりうる．
  Galar らが OvO に対して動的分類器選択を併用する動機がこれである（本反復では素の OvO を使った）．

**既存データで検証可能な手順（再埋め込み 0 回・CPU のみ）**:
`scripts/screen_classifier_multiclass_decomposition.py` に診断出力を足して再実行する．
1. 両腕の **10×10 混同行列**を出す．予測: 腕 B で失われた social_science の 42 行は
   general・education・business_economics へ偏って流れる（一様には散らない）．
2. **45 対それぞれの二値 CV 正解率**を出す．予測: social_science–general，
   social_science–education，history_culture–general の 3 対が最下位群に来る．
3. 腕 B の**勝者と次点の票差の分布**を出す．予測: 腕 A で正解し腕 B で誤った 138 行のうち
   多数派が**票差 1**（＝ 1 本の対ごと分類器の誤りで覆った）である．
   これが (b) の直接証拠になり，票差が 2 以上ばかりなら (a) 側の説明が優勢になる．

#### 考察 (ii): P4 の gap 分布の収縮（20.1% → 8.3%）は何を意味するか

`OneVsOneClassifier.decision_function` は**整数の得票数**（0〜9）に微小な決定値を足したものであり，
本質的に**離散で分解能が低い**．勝者はしばしば 9 勝，次点は 8 勝といった値を取るので，
スコア差は連続量ではなく粗い階段状になる．これに **単一スカラーの temperature 較正**を掛けると，
階段差がそのまま確率差へ写り，**上位 2 クラスの差が人為的に広がる**．
実測の `gap < 0.36` の行割合 20.1% → 8.3%（−11.8pt）はこの形で説明できる．

**確信度は上がったのに正解率は下がった**（Δ top1 = −0.806pt）という組み合わせなので，
これは「識別が鋭くなった」のではなく**過信（較正の劣化）**である可能性が高い．
運用上の含意は 2 つある．
- **(1)** 仮に本走していれば，`dispatch_gap_threshold=0.36` の下で複数専門家へ送出される行が
  20.1% → 8.3% へ半減し，`compound_mean_dispatched_count` と `compound_domain_set_recall` が
  CV top1 の −0.81pt を超えて悪化した公算が高い．**足切りで止めた判断は結果的に妥当だった**．
- **(2)** `dispatch_gap_threshold` は**分解方式を跨いで移植できない**．
  archive の学び「特徴量を変えるレバーの後には必ず gap 閾値の未較正が残る」（Iter81→82→83 で 3 回観測）は，
  **特徴量だけでなく決定層の形を変えた場合にも成り立つ**．今回は本走していないので再較正は不要であり，
  `dispatch_gap_threshold_recalibration` は adopted・収束のまま触らない（本反復の禁止事項どおり）．

**既存データで検証可能な手順**: 両腕の**行ごとの max-prob と gap をダンプ**し，
(a) ECE と Brier スコア，(b) `gap ≥ 0.36` の行に限った正解率，を腕ごとに比較する．
過信であれば **腕 B の ECE が腕 A より大きく，かつ `gap ≥ 0.36` バケットの正解率が腕 A より低い**．
逆に腕 B の高 gap バケットの正解率が腕 A と同等以上なら，「鋭くなったが総数を落とした」という
別の読みになる．併せて較正前の `decision_function` のヒストグラムを見れば，離散性（数個のモードへの
集中）は目視で確認できる．

#### 考察 (iii): 機序 M5 との整合／不整合

M5 は「クラス総重み一定（`sample_weight = n/(K·n_d)`）の下では，行追加は情報の追加ではなく
**クラス間の決定境界の再配分**である」という読みで，Iter95/96 の**零和性**と
**腕 A の recall に対する Δ の単調性**を根拠にしていた．Iter97 はこの M5 が
「1 つの softmax 正規化と 1 組の共有境界」に由来するという構造的解釈を検定した実験であり，
結果は**部分的に不整合**である．

- **不整合 1（零和性が再現しない）**: per-domain recall Δ の内訳は
  **正側 +3.04pt 対 負側 −10.72pt，合計 −7.68pt**（クラスがほぼ均等なので非重み平均 −0.768pt は
  全体 Δ −0.806pt とよく一致する）．Iter95/96 のような**差し引きゼロの再配分ではなく，正味の損失**である．
  すなわち OvO への変更は「境界の配り直し」ではなく**情報を失う操作**だった（考察 (i) の (a)(b)）．
- **不整合 2（単調性が再現しない）**: Spearman ρ = +0.018（p=0.96）で，Iter96 の −0.717 は消える．
  **強さに対する単調性は M5 の普遍的な性質ではなく，「行を足す」という操作に固有の署名だった**．
- **整合している点**: 弱いクラス（education）だけが改善し（+2.24pt），強いクラスが損をするという
  **符号の向き自体**は残っている．ただし今回は改善幅が小さく非有意（p=0.065，BH 後も非有意）で，
  education の recall は 0.535 → 0.557 と**依然として壊滅的**である．

したがって M5 の**構造的解釈（共有 softmax 正規化が零和性の原因）は支持されなかった**．
決定層の分解方式を根本から変えても education の誤りはほぼそのまま残ったのだから，
**律速は決定層ではなく，5,120 次元の埋め込み表現そのもの（あるいはラベルの張り方）にある**という
読みが最も素直である．M5 自体は「行追加操作に関する経験則」としては生き残るが，
**その原因を softmax 正規化に帰する説明は取り下げる**．

**既存データで検証可能な手順**: 腕 A のクラス別 `‖w_d‖` と，腕 B の対ごと重みベクトルから合成した
実効的なクラス別ノルムを比較する．M5 の構造的解釈が正しければ腕 B では総重み一定の制約が消えて
ノルムの分散が広がるはずで，広がっていなければ「制約は softmax ではなく標本サイズ由来」という
上の読みが補強される．

#### 学び

1. **両論ある仮説を事前に両論のまま登録しておくと，負の結果が解釈可能な知識になる**．
   本反復は Fürnkranz (2002)・Galar et al. (2011)（OvO 優位）と Rifkin & Klautau (2004)
   *In Defense of One-Vs-All*（OvA で十分）を**事前に**並記して検定と位置づけた．
   実測は後者と整合し，「小標本・高次元・クラス数 10」という本研究の条件では
   **対ごと分解の『境界の単純さ』の利得より，標本分割による分散増大の損失が勝つ**ことが分かった．
   Hsu & Lin (2002) の線形カーネル SVM での OvO 優位が本件に外挿できなかったのは，
   本研究の n/p 比（2,275 行 / 5,120 次元）が文献の設定と桁違いに厳しいためと考える．
2. **sklearn のメタ推定器は，サポートしないメタデータを警告 1 行で黙って捨てる**．
   `CalibratedClassifierCV(OneVsOneClassifier(...))` に `sample_weight` を渡すと
   base estimator 側の重みが落ち，気づかなければ「分解方式」と「クラス均衡重み」の
   2 レバー同時変更になっていた．`enable_metadata_routing=True` ＋ `set_fit_request` で明示配線し，
   **「警告が 1 件でも出たら実験不成立として中断する」ガードをコードに埋め込んだ**のが有効だった．
   この型（単一レバー性をコードで機械的に強制する）は今後のスクリーニングでも踏襲する．
3. **決定層（多クラス分解・較正・重み・訓練行）を一巡した結果，CV は 0.79〜0.82 に張り付いたままである**．
   Iter88〜96（データ量・重み・粒度）に加え Iter97（分解方式）まで動かないのだから，
   残る説明変数は**入力表現とラベルの張り方**である．次の探索軸はここに限定してよい．
4. **seed 併合の McNemar は反保守的である**（同じ行を seed 数だけ重複計上している）．
   本反復では結論は変わらなかったが，足切り 2 の当落は保守側で入れ替わりうることを確認した．
   今後は**保守値（discordant を seed 数で割った再計算）を必ず併記**する．
5. **確信度が鋭くなったことを性能改善の証拠として読んではいけない**．腕 B は
   `gap<0.36` の行を半減させながら top1 を下げた．**gap 分布の変化は，正解率と切り離して
   単独では解釈できない**（ECE と高 gap バケットの正解率を必ず併記する）．

**次の一手**: `classifier_multiclass_decomposition` の未試行 value
**`error_correcting_output_codes`** を引く（backlog B163）．これは決定層の軸を**閉じるための
確認実験**であり，OvO 固有の弱点（対ごとの標本分割・得票の離散性・非適格分類器）と
「決定層の分解一般」を切り分ける．既存のスクリーニングスクリプトへ第 3 腕を足すだけで済み，
追加費用はほぼゼロである．ここも通らなければ**決定層の軸は打ち止め**とし，
Iter99 は調査フェーズから入力表現／ラベルの軸を探索する．

## Iteration 96: タスク構成比を保ったまま訓練行を増やし n の効果だけを測る

### 調査 (Iter96)

本フェーズでは**実験ノード wafl500〜509 に一切触れていない**（2026-09-23 絶対条件 (B)）．
行った計算は，開発ホスト上での JSONL / JMMLU.zip の読み取りと行数の数え上げ（純粋な CPU・GPU 不要・
埋め込み計算 0 回）だけである．`data/dataset.jsonl`・`models/`・`results/` への書き込みは 0 件．

**(1) 問い**: (a) 「各 JMMLU タスクの訓練行を m 倍にする」規則で，タスク構成比を実質的に崩さずに
達成できる m の最大値はいくつか，(b) その m で n はいくつになり，学習曲線 err ∝ n^(-0.127) の
外挿ではどれだけの CV 改善が見込めるか，(c) Iter95 の機序 (M1)（CV のテスト fold が旧訓練分布の
ままなので訓練分布を動かした腕が不利になる）は文献上どう位置づけられるか．

**(2) per-task のプール残量の実測（現行訓練集合 `data/classifier_train_iter94_dedup.jsonl` 2,275 行）**

56 タスクすべてについて「現行訓練行数 cur」と「評価集合とも現行訓練集合とも素なプール残量 pool」を
数えた．タスク間で設問文が重複する行（Iter95 実装で natural_science に 5 件の二重登録が生じた原因）は，
`_DOMAIN_TASK_MAP` の順で最初に現れたタスクへ一意に割り当てて 1 回だけ数えた．
主な結果（ドメイン合計）: medical cur 250 / pool 810，natural_science 250 / 479（重複除去後），
history_culture 210 / 479，business_economics 250 / 405，mathematics 250 / 48，education 238 / 9，
legal 77 / 0，computer_science 250 / 0，social_science 250 / 0，general 250 / 0．

**m を変えたときの per-domain タスク構成比のずれ（追加前後の全変動距離 TV）と n**

| m | 追加行 | n | TV（最大のドメイン） | プール潤沢 4 ドメインの TV | 頭打ちタスク |
|---|---|---|---|---|---|
| 1.5 | +534 | 2,809 | 0.026 (math) | ≈ 0 | 23/56 |
| **2** | **+1,015** | **3,290** | **0.026 (math)** | **0.000〜0.004** | 24/56（うち 20 は pool=0） |
| 2.5 | +1,452 | 3,727 | 0.052 | 0.02〜0.05 | 30/56 |
| 3 | +1,749 | 4,024 | 0.065 (bus_econ) | 0.002〜0.065 | 36/56 |
| 4 | +2,045 | 4,320 | 0.105 (hist_cult) | 0.05〜0.11 | 46/56 |

**m=2 は「プール残量のある 4 ドメイン（medical・natural_science・history_culture・
business_economics）で per-task の倍増がほぼ正確に達成できる最大の整数」である．**
m=2 で頭打ちになるのは `business_ethics`（cur 28 / pool 26，−2 行）だけで，この 4 ドメインの
構成比 TV は 0.000〜0.0036 に収まる．m=3 以上では natural_science・history_culture・
business_economics で多数のタスクが頭打ちになり，構成比 TV が 0.043〜0.065 へ跳ね上がる
（Iter95 の訓練 vs 評価 TV 最大 0.189 ほどではないが，本レバーの主旨である「構成比を保つ」が崩れる）．
mathematics（TV 0.026）と education（0.008）は m によらずプールが薄く，追加は 48 行・9 行に留まる．

**(3) 学習曲線の外挿（Iter95 計画フェーズの実測 err ∝ n^(-0.127) を流用．新規計算なし）**

hold-out を差し引いた後の n = 3,271（下記 (ii)）は現行の **1.438 倍**で，
Iter95 スクリーニングの基準腕 CV 0.80322 に当てると **予測 CV Δ = +0.89pt**
（伝達率 43% を当てた e2e 予測は +0.38pt）．**これは事前登録の足切り +1.0pt を下回る**
（判断は B160 に記録）．Iter95（n 1.98 倍）の外挿値は +1.73pt で実測 +0.366pt だったので，
本反復の外挿値も上限の目安として扱う．

**(4) 文献調査（tvly search）**

- Sugiyama, Krauledat & Müller (2007), *Covariate Shift Adaptation by Importance Weighted
  Cross Validation*, JMLR 8:985–1005, <https://www.jmlr.org/papers/v8/sugiyama07a.html> ——
  訓練入力分布とテスト入力分布が異なるとき，通常の交差検証は**モデル選択の基準として不偏でなくなる**．
  是正手段は (a) 重要度重み付き CV を使うか，(b) そもそも訓練入力分布を動かさないか，の 2 つ．
  Iter95 の機序 (M1) はこの (a) を怠った状態そのものであり，**本反復は (b) を選ぶ設計**である．
  （本研究では真の密度比を推定できないので (a) は採らない．）
- Ye et al. (2025), *Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Model
  Performance*, ICLR 2025, arXiv:2403.16952, <https://arxiv.org/abs/2403.16952> ——
  性能は**データ量とは別に混合比の関数として定量的に動く**．量と比を同時に動かした Iter95 では
  両者の寄与が分離できない．比を固定して量だけ動かす本反復の設計は，この分離を意図している．
- Viering & Loog (2024), *The Shape of Learning Curves: A Review*, *Machine Learning*,
  doi:10.1007/s10994-024-06619-7 —— 逆べき乗則による外挿（Iter95 で実測した α=0.127）の根拠．
  同論文は外挿が外れる形状（plateau・ill-behaved curve）も整理しており，Iter95 の実測
  +0.366pt（外挿 +1.73pt の 1/5）はその可能性も残す．
- Byrd & Lipton (2019), ICML, <http://proceedings.mlr.press/v97/byrd19a/byrd19a.pdf> ——
  再重み付け・再サンプリングによる分布合わせの効果は容量の大きいモデルで漸近的に消える．
  本反復が「構成比を変えない」側を選ぶことと整合する（構成比を動かす操作に期待しない）．

### 仮説 (Iter96)

**Iter95 で n の効果が見えなかったのは n が効かないからではなく，n の増加と per-domain の
タスク構成比の移動が交絡し，かつ CV のテスト fold が旧構成のままだったからである（機序 M1）．**
タスクごとの構成比を保ったまま行数だけを m=2 倍に揃えれば，Sugiyama et al. (2007) の意味で
CV は妥当な比較基準になり，学習曲線どおりの向き（正）の Δ が観測されるはずである．

### 単一レバー (Iter96)

- **レバー**: `composition_preserving_volume_expansion` = `per_task_proportional_scale`
  （config.yml の `levers` 末尾に Iter95 分析フェーズが追記済み．B159 (d) で `[auto-decided]`）
- **何を何から何へ**: 分類器の訓練集合を `data/classifier_train_iter94_dedup.jsonl`（**2,275 行**）から
  `data/classifier_train_iter96_scaled.jsonl`（**3,271 行**，+996 行）へ．
  規則は「**各 JMMLU タスクについて，hold-out を除いたプールから，現行訓練行数の m=2 倍になるまで
  評価集合と素な行を足す（プール残量で頭打ち）**」の 1 つだけ．**m=2 は全 56 タスク・全 10 ドメイン
  共通の単一定数として本フェーズで事前登録する**（根拠は上記 (2)）．既存 2,275 行は 1 行も削らない．
- **事前登録した内訳（実測．executor はこの数値と一致することを確認すること）**

| ドメイン | 現行 | 追加 | 追加後 | 備考 |
|---|---|---|---|---|
| medical | 250 | +250 | 500 | 全 10 タスクで正確に 2 倍 |
| natural_science | 250 | +250 | 500 | 全 8 タスクで正確に 2 倍 |
| history_culture | 210 | +210 | 420 | 全 7 タスクで正確に 2 倍 |
| business_economics | 250 | +237 | 487 | `business_ethics`・`management` のみ頭打ち |
| mathematics | 250 | +40 | 290 | 5 タスクとも頭打ち（プール 6〜15 行） |
| education | 238 | +9 | 247 | 3 タスクで数行のみ |
| legal / computer_science / social_science / general | 77 / 250 / 250 / 250 | +0 | 同左 | プール枯渇 |
| **合計** | **2,275** | **+996** | **3,271** | d/n は 2.25 → **1.57** |

- **固定する構成（1 つも動かさない）**: 評価集合 `data/dataset.jsonl`（sha256 `2114e048...`，3,750 行，
  読むだけ）・埋め込みモデル `qwen3-embedding:4b`・`embedding_instruction`（Iter81 の P1 文言）・
  `embedding_view_concat: true`（5,120 次元）・`CalibratedClassifierCV`・`C=1.0`・
  重み `n/(K*n_d)`・`classifier.py` の推論経路・`dispatch_gap_threshold: 0.36`・
  `dispatch_gap_max_k: 4`・`dispatch_top_k: 2`・`aggregation_method: max_confidence`・
  expert/light モデルとノード割り当て．

### 事前スクリーニング設計（事前登録 / Iter96）

config.yml の当該 note (i)(ii) を両方実施する．**足切りを通った場合は本走を省略しない**
（2026-09-23 恒久ルール．事前シミュレーションで着地点を言語化しても本走は必須）．

- **(i) 主評価 —— Iter95 と同一の固定 5-fold CV．** 既存 2,275 行のみを層化 5 分割（3 seed）し，
  テスト fold は両腕で完全に同一．腕 A = 無追加（2,275 行），腕 B = m=2 追加（3,271 行）．
  追加行は常に訓練側だけに入れる．較正込みの本番パイプライン（`train_domain_classifier.train_classifier()`）
  をそのまま呼び，統計は `metrics.py` の既存実装（`compute_top1_accuracy`・`compute_mcnemar_test`・
  `compute_domain_recall_mcnemar_test`・`apply_benjamini_hochberg`）を使う．
  本レバーは訓練のタスク構成比を（プール潤沢 4 ドメインで TV ≤ 0.004 に）保つので，
  Iter95 で問題になった「テスト fold の分布ずれによる系統的不利」は生じない．
- **(ii) 副次評価 —— プール hold-out を足したテスト集合．** プールから **per-task で 20%（切り捨て）**を
  hold-out として取り分け，**どちらの腕でも訓練に使わない**（合計 426 行．
  medical 158 / natural_science 90 / history_culture 92 / business_economics 78 / mathematics 8 /
  education 0 / 他 0）．選択規則は「各タスクのプール行を JMMLU CSV 順に並べ，添字 i % 5 == 0 を
  hold-out」とし，乱数シードを持たない．各 fold のテスト集合に hold-out 426 行を足した accuracy も
  報告し，**(i) と Δ の符号が一致するか**を確認する．一致しなければ (i) の結論は保留扱いにする．
  hold-out はプール残量のある 5 ドメインにしか存在しないので，**これは全体の不偏推定ではなく
  符号の整合性チェックである**（限界として journal に明記済み）．
- **評価集合 `data/dataset.jsonl` はスクリーニングでは一切参照しない**（除外集合の計算のみに読む）．

**足切り（両方を満たしたときだけ本走へ進む．停止規則）**

1. **CV top1 Δ (B−A) ≥ +1.0pt**（Iter95 と同一．B158 (d) の引き下げを踏襲）．
2. **per-domain recall の BH 補正後（q=0.05，10 指標）有意な退行が 0 件**．
   1 件でもあれば本走しない．
- どちらかを外したら，本番コード（`models/`・`config.yaml`・`data/dataset.jsonl`）を
  **1 バイトも変更せずに**結果だけ記録してフェーズ 3 へ渡す（Iter95 と同じ運用．絶対条件 (A) 非抵触）．
- 通った場合のみ `models/domain_classifier.joblib` を再訓練（旧 artifact は
  `models/domain_classifier_pre_iter96_scaled.joblib` へ退避），`mise run deploy` と sha256 確認のうえ，
  **wafl500〜509 で 3,750 問フルスペックの本走**を行う．

### 反証可能な予測（事前登録 / Iter96）

- **P1（機序 M1 の核心）**: Iter95 で BH 後有意に退行した `history_culture`（−5.08pt）・
  `natural_science`（−5.20pt）の退行が，構成比を保つ本反復では**消える**（recall Δ ≥ −1.0pt）．
  同程度の退行が再現するなら M1 は誤りで，「行を増やすこと自体が害」と読み替える．
- **P2（C5 の再検証．Iter95 では反証された）**: プール残量のある 5 ドメイン
  （medical / natural_science / history_culture / business_economics / mathematics）の recall Δ 合計が，
  枯渇 5 ドメイン（legal / computer_science / social_science / general / education）の Δ 合計を上回る．
  枯渇側はほぼ不変（各 |Δ| ≤ 1.0pt）であるはず．
- **P3（量の効果）**: CV Δ の点推定が **+0.4pt 〜 +1.4pt**（学習曲線外挿 +0.89pt を中心とした帯）に入る．
  Iter95 と同水準（+0.4pt 未満）なら，n を 1.44 倍にした程度では動かないことの追加証拠になる．
- **P4（設計の健全性）**: (ii) の hold-out 込み評価の Δ の符号が (i) と一致する．

### 成功条件・非退行条件（事前登録 / Iter96）

**基準線**: `results/20260928_160921/` の top1 = **0.833067**（3,750 問，Wilson 95%CI [0.8208, 0.8447]）．
再現性の床は ±0.25pt．

- **スクリーニング段階**: 上記の足切り 1・2 を両方満たすこと（満たさなければ `closed`，本走なし）．
- **本走に到達した場合の採択条件**: 全体 top1 が **0.838 以上（Δ ≥ +0.5pt）** かつ
  McNemar **p < 0.05**なら `adopted`．Δ ≥ +0.5pt だが p ≥ 0.05 なら `adopted_small`（要再現）．
  |Δ| < 0.25pt（再現性の床以下）なら `negligible`．Δ ≤ −0.5pt なら `rejected`．
- **非退行条件（本走時）**: per-domain recall に BH 補正後（q=0.05，10 指標）有意な退行が 0 件．
  特に `legal`（訓練 77 行のまま，全体に占める比率が 3.4% → 2.4% へ低下）と `education`（238 → 247）は
  クラス重み `n/(K*n_d)` で補正されるとはいえ実効的な不均衡が進むため，退行の有無を必ず確認する．
  複合設問 730 行の top1（基準 0.7904）が −1.0pt を超えて下がらないこと．

### 実行フェーズへの申し送り（Iter96）

- `scripts/expand_classifier_train_pool.py` を流用し，(a) per-task の上限計算（m=2），
  (b) タスク間で重複する設問文の一意化，(c) hold-out 20% の取り分け，の 3 点だけを足すこと
  （新規スクリプトにする場合も既存の `build_dataset.py` 内部関数の再利用は崩さない）．
  `build_dataset.py` の CLI は**絶対に呼ばない**（`--domain-target-size` が評価集合と共用のため）．
- 追加行・hold-out 行はいずれも Iter95 の 4,507 行の部分集合なので，
  wafl-ctrl5 上の `data/embcache_iter95_fullpool.npy`（4,507 × 5,120）を query 文字列で引き当てれば
  **再埋め込みは不要**（新規の `/api/embeddings` 呼び出しは 0 回が目標）．
  引き当てに失敗した行が出た場合のみ wafl-ctrl5 で追加埋め込みを行う（絶対条件 (B)）．
- スクリーニングは `scripts/screen_classifier_training_volume_expansion.py` を雛形にし，
  hold-out 込みの副次評価を足す．実行は wafl-ctrl5 の `~/expert-mesh-iter95/`（uv 3.12 環境）を流用する．

### 実装・実験 (Iter96 executor)

**実施場所の申告**: 生成・埋め込み引き当て・スクリーニングはすべて実験ノード wafl500〜509 に一切
触れていない．生成は開発ホスト（CPU のみ，`/tmp/expert-mesh-cache/JMMLU.zip` の読み取りと
`data/dataset.jsonl`（読むだけ．sha256 `2114e048...` 不変を確認済み）の読み取りのみ），埋め込み引き当て
とスクリーニングは wafl-ctrl5 の `~/expert-mesh-iter95/`（uv 3.12 環境）．**gate が不通過だったため，
`models/`・`config.yaml`・`data/dataset.jsonl` は 1 バイトも変更せず，wafl500〜509 の本走は行っていない．**

**(a) データ生成**: `scripts/expand_classifier_train_pool_proportional.py`（新規．
`scripts/expand_classifier_train_pool.py` の関数群（`_DOMAIN_TASK_MAP`・`_format_jmmlu_query`・
`_load_jmmlu_zip_bytes`・`_parse_jmmlu_task_csv`）をそのまま import して流用し，(a) per-task 上限計算
（`cur_task * (m-1)` をプール残量で頭打ち），(b) タスク間の設問文一意化（`_DOMAIN_TASK_MAP` 辞書順・
タスク順で最初に現れたタスクへ 1 回だけ割り当て），(c) hold-out 20% 切り捨て取り分け，の 3 点だけを
追加）を実行し，`data/classifier_train_iter96_scaled.jsonl`（**3,271 行，既存 2,275 行 + 新規 996 行**）
と `data/classifier_train_iter96_holdout.jsonl`（**426 行**）を生成した．
**事前登録した内訳表と完全一致**（ドメインごとの追加行数: medical 250 / natural_science 250 /
history_culture 210 / business_economics 237 / mathematics 40 / education 9，
hold-out: medical 158 / natural_science 90 / history_culture 92 / business_economics 78 /
mathematics 8，他 5 ドメインは 0）．一意性検証: 訓練 3,271 行はクエリ文字列で重複 0，hold-out 426 行と
訓練行・評価集合との重複 0，訓練行と評価集合との重複 0．

**実装中に発見して修正したバグ 1 件**: hold-out の切り出しルール（「タスク内 CSV 順で添字
i % 5 == 0」）の解釈を，最初は「タスク単位でカウンタをリセットして i % 5 == 0 の個数をそのまま採る」
（= 各タスクで `ceil(L/5)` 行）で実装したところ，タスク数が多いドメイン（medical 10 タスク・
natural_science 8 タスク）で丸めが過大に効いて hold-out が計 461〜448 行（事前登録 426 行から
+22〜+35）になった．事前登録文言の「20%（切り捨て）」と整合させるため，
`floor(L/5)` 行（インデックス 0, 5, 10, ... のうち先頭 `L//5` 個，端数グループは切り捨て）に直したところ
医療 158・natural_science 90・history_culture 92・business_economics 78・mathematics 8，
合計 426 行で事前登録表と完全一致した．
さらに，JMMLU の CSV 自体にタスク内重複行が存在すること（`college_physics` に 10 件，
`conceptual_physics` に 1 件の verbatim 重複問題．Iter95 で見つかったのはタスク**間**の重複 5 件で，
これはタスク**内**の重複であり別種）を実行中に発見し，タスク内一意化（`seen_in_task` セット）を追加して
1 件の重複出力（`natural_science-pool096-062` と `-067` が同一クエリ）を解消した．
（この 2 点は事前登録した「3 点だけ足す」の範囲内の実装詳細の確定であり，レバーの内容や足切り基準は
一切変更していない．）

**(b) 埋め込み引き当て**: `scripts/build_iter96_embcache_from_iter95_pool.py`（新規）で，
wafl-ctrl5 上の `data/embcache_iter95_fullpool.npy`（4,507 × 5,120，Iter95 で作成済み）を
`data/classifier_train_iter95_fullpool.jsonl`（同じ 4,507 行，行順が embcache と一致）とのクエリ文字列
結合で引き当てた．**訓練 3,271/3,271 行・hold-out 426/426 行がすべて引き当て成功（引き当て率 100%）．
新規の `/api/embeddings` 呼び出しは 0 回**（目標どおり）．

**(c) スクリーニング**: `scripts/screen_composition_preserving_volume_expansion.py`（新規．
`scripts/screen_classifier_training_volume_expansion.py` の固定 5-fold CV 構造をそのまま流用し，
(ii) hold-out 込み副次評価を追加）を wafl-ctrl5 で実行（3 seed × 5 fold = 15 fold，各 fold で腕 A・腕 B
とも `train_domain_classifier.train_classifier()`／`_extract_sample_weights()` を無改変で呼び出し，
統計は `metrics.py` の `compute_top1_accuracy`・`compute_mcnemar_test`・
`compute_domain_recall_mcnemar_test`・`apply_benjamini_hochberg` をそのまま使用．実行時間 約 13 分）．

**スクリーニング結果（数値そのまま，評価は分析フェーズに委ねる）**

| 指標 | 値 |
|---|---|
| CV top1（腕 A，無追加） | 0.802637 |
| CV top1（腕 B，m=2 追加） | 0.805861 |
| **CV top1 Δ (B−A)** | **+0.322pt** |
| 全体 McNemar p | 0.1415（discordant a-only 91 / b-only 113） |
| **足切り 1（CV Δ ≥ +1.0pt）** | **不通過**（+0.322pt） |
| BH 補正後（q=0.05，10 指標）有意な per-domain recall 変化 | 4 件: education +4.06pt(p=8.8e-05)・medical +4.40pt(p=3.0e-06)・history_culture **−2.38pt**(p=0.00225)・natural_science **−4.13pt**(p<1e-6)．**有意な退行 2 件**（history_culture・natural_science） |
| **足切り 2（有意な退行 0 件）** | **不通過**（2 件） |
| **足切り総合判定** | **不通過**（両方満たさず） |
| (ii) hold-out 込み副次評価: accuracy（腕 A / 腕 B） | 0.852062 / 0.862807 |
| (ii) hold-out 込み Δ | **+1.075pt**（符号は (i) の +0.322pt と一致） |
| C5: プール潤沢 5 ドメイン（medical/natural_science/history_culture/business_economics/mathematics）recall Δ 合計 | **−2.78pt** |
| C5: プール枯渇 5 ドメイン（legal/computer_science/social_science/general/education）recall Δ 合計 | **+5.83pt** |

**足切りの通過可否**: **不通過**（両方の停止規則を満たさず）．
**規定どおり，本番コード（`models/`・`config.yaml`・`data/dataset.jsonl`）は 1 バイトも変更せず，
wafl500〜509 の本走は実施していない．** スクリーニング数値のみを本節に記録し，分析・考察フェーズへ
引き渡す．

**事前登録した予測 P1〜P4 の実測結果**

- **P1（機序 M1 の核心．反証**）: 構成比を保ったにもかかわらず，Iter95 で有意退行した
  history_culture・natural_science は本反復でも有意に退行した（history_culture −2.38pt，
  natural_science −4.13pt；Iter95 は −5.08pt／−5.20pt）．「退行が消える」という予測は外れ，
  P1 の反証条件（同程度の退行が再現）に該当する．
- **P2（C5 の再検証．反証**）: プール潤沢 5 ドメインの recall Δ 合計（−2.78pt）は，
  プール枯渇 5 ドメインの合計（+5.83pt）を**下回った**（予測は「上回る」）．枯渇側も
  |Δ| ≤ 1.0pt という予測に反し，education（+4.06pt）・（legal/computer_science/social_science/general
  は |Δ| ≤ 0.7pt で小さい）で大きく動いた．
- **P3（量の効果．範囲外）**: CV Δ 点推定は **+0.322pt** で，予測帯 +0.4pt〜+1.4pt の**下限を下回った**．
  Iter95（+0.366pt）とほぼ同水準で，学習曲線外挿（+0.89pt）の 1/3 弱に留まった．
- **P4（設計の健全性．成立）**: (ii) hold-out 込み評価の Δ（+1.075pt）は (i) の Δ（+0.322pt）と
  符号が一致した．

### Iteration 96 実行済み —— 判定 `closed`（事前スクリーニング足切り不通過・本走なし）

**変更（本番への適用は 0 件）**: 新規スクリプト 3 本
（`scripts/expand_classifier_train_pool_proportional.py`・
`scripts/build_iter96_embcache_from_iter95_pool.py`・
`scripts/screen_composition_preserving_volume_expansion.py`）と
`results/iter96_screening/screening_result.json`，および `data/classifier_train_iter96_{scaled,holdout}.jsonl`
（`.gitignore` 対象・どのコードからも読まれない）のみ．
`models/`・`config.yaml`・`data/dataset.jsonl` は 1 バイトも変更していない．基準線
`results/20260928_160921/` top1 = 0.833067 は不変．wafl500〜509 不使用（実行は wafl-ctrl5）．

**結果（固定 5-fold CV，3 seed 併合，n=6,825）**: 腕 A 2,275 行 0.802637 → 腕 B 3,271 行 0.805861，
**Δ = +0.322pt（McNemar p = 0.1415，discordant 91/113）**．足切り 1（+1.0pt）不通過．
per-domain recall の BH 後有意は 4 件（education +4.06 / medical +4.40 の改善，
**history_culture −2.38 / natural_science −4.13 の退行**）で，足切り 2（有意退行 0 件）も不通過．
副次評価（hold-out 426 行込み）は 0.852062 → 0.862807，Δ = +1.075pt（符号は主評価と一致）．

#### 判定

- **`closed`**（Iter93 `classifier_label_granularity`・Iter95 `classifier_training_volume_expansion` と
  同じ扱い）．**`rejected` ではない**: `rejected` は本走の 2 軸表（Δtop1 × McNemar p）に基づく判定語で
  あり，事前登録した停止規則により本走を行っていない以上どのセルにも到達していない．
  `composition_preserving_volume_expansion` は値 `per_task_proportional_scale` をもって closed とし，
  config.yml の当該 note に実測値を注記した（同じ案を再度引かないため）．

#### 分析 1 —— 機序 M1（n と構成比の交絡）は支持されなかった

Iter95 の分析が最有力とした **(M1)「n の増加とドメイン内タスク構成比の移動が交絡しており，
CV のテスト fold が旧構成のままなので追加が多いドメインが系統的に不利になる」は，
構成比を per-task で固定する（プール潤沢 4 ドメインで TV ≤ 0.004）という直接的な操作で反証された．**
history_culture・natural_science は今回も BH 後有意に退行した（−2.38 / −4.13pt．Iter95 は −5.08 / −5.20pt）．
退行幅は縮んだが，事前登録した反証条件（recall Δ ≥ −1.0pt で「消える」）を満たさない．
構成比の移動は退行の**主因ではなく，せいぜい寄与の一部**である．

同時に，「行を足すほど退行する」という単純な読み替えも**成立しない**．
同じ相対用量（×2 倍）を受けた 4 ドメインで符号が割れているためである．

| ドメイン | 追加行（倍率） | 腕 A recall | recall Δ (pt) | BH |
|---|---|---|---|---|
| medical | +250 (×2.0) | 0.673 | **+4.40** | 有意（改善） |
| business_economics | +237 (×1.95) | 0.816 | −0.93 | 非有意 |
| history_culture | +210 (×2.0) | 0.852 | **−2.38** | 有意（退行） |
| natural_science | +250 (×2.0) | 0.876 | **−4.13** | 有意（退行） |
| mathematics | +40 (×1.16) | 0.948 | +0.27 | 非有意 |
| education | +9 (×1.04) | 0.535 | **+4.06** | 有意（改善） |
| legal / computer_science / social_science / general | +0 | 0.840/0.920/0.788/0.799 | +0.43/+0.27/+0.40/+0.67 | 全て非有意 |

**同用量 4 ドメインの Δ は腕 A の recall について完全に単調（0.673→+4.40，0.816→−0.93，
0.852→−2.38，0.876→−4.13）**であり，10 ドメイン全体でも Δ と腕 A recall の相関は
Spearman −0.717（p = 0.020）である．**符号を決めているのは追加量ではなく，そのクラスが
元々どれだけ被覆できていなかったか**である．

#### 分析 2 —— 有力な機序 (M5): 一定のクラス総重みの下での「行数増加＝決定境界の再配分」

現行の重み付けは `class_weight=None` ＋ 手動 `sample_weight = n/(K·n_d)`（Iter39 / B60）であり，
**どのクラスも総重みは n/K で一定**である．したがって，あるクラス d に行を足す操作は
「クラス d の証拠を増やす」ことではなく，**固定された総重みをより多様な点へ薄く配り直す**ことである．
その帰結は 2 つに分かれる．

- クラス d の discriminant は平滑化され（peak が鈍り），境界付近の点を取りに行く力が弱まる
  → **周囲のクラスの recall が上がる**．
- クラス d 自身の recall は，「被覆が増える利得」と「peak が鈍る損失」の差で決まる．
  被覆不足の多峰クラス（medical 0.673・education 0.535）では前者が勝ち，
  既に飽和した compact なクラス（natural_science 0.876・history_culture 0.852）では後者が勝つ．

この 1 つの機序で今回の観測 4 点がすべて説明できる: (i) 同用量下で Δ が腕 A recall に単調，
(ii) 追加 0 の 4 ドメインが揃って小さく改善（+0.27〜+0.67pt），(iii) **education が +9 行しか
足していないのに +4.06pt 改善**（education は最も拡散したクラスで，境界の取り合いに最も負けていた．
自分の行が増えたからではなく，**競合クラスが退いたから**上がった），(iv) 全体 top1 がほぼ動かない
（ドメイン間で ±4pt を再配分するだけの零和）．

**Iter95 と Iter96 を跨いだ定量的な規則性**（M5 の最も強い傍証）: 追加 0（枯渇 5）側の recall Δ 合計は
Iter95 +11.48pt / 2,232 行 = **0.00514 pt/追加行**，Iter96 +5.83pt / 996 行 = **0.00585 pt/追加行**で，
用量が 2.2 倍違うのに **1 行あたりの効果がほぼ一致**する（education 単独でも 0.00383 対 0.00408）．
**枯渇側の改善は「自分に足された行数」ではなく「他クラスに足された総行数」に比例している．**
すなわち本研究が「n を増やす」と呼んできた操作は，実質的に
**クラス条件つきの決定境界（事前分布）の再配分**であり，情報量の追加ではない．

**M5 以外の候補と，既存データでの検証手順（いずれも wafl-ctrl5 の CV 枠内で完結し，本走を要さない）**

1. **(M5) 決定境界の再配分**: 腕 A / 腕 B の学習済み係数について **クラス別の ‖w_d‖ と
   クラス別の平均 max-prob** を比べる．M5 が正しければ，行を足したクラスほど ‖w_d‖ と
   自クラス平均確信度が下がり，枯渇クラスは相対的に上がる．
   さらに **Δrecall を「自クラス追加行数」と「他クラス追加総行数」に回帰**すれば，
   上記 0.0055 pt/行 の係数が両反復で共通かを直接検定できる．
2. **(M2) 較正の非対称**: `CalibratedClassifierCV` の内部 fold に入る行数がクラス間で 250 対 500 に
   開く．腕 A / 腕 B のクラス別較正曲線（reliability diagram）と Brier を比較すれば分離できる．
   較正を外した素の LR で同じ CV を回し，per-domain Δ のパターンが消えるかを見るのが最も安い．
3. **(M6) 実効正則化の変化**: `C=1.0` 固定のまま総重み（= n）が 1.44 倍になるため，
   腕 B は実効的に正則化が弱い．Iter91 の `classifier_regularization_strength` が `no_effect` だった
   ことから寄与は小さいと見込まれるが，腕 B を `C=1.0/1.438` で回した第 3 の腕を足せば直接分離できる．
4. **(M7) 追加行の情報量の低さ**: 追加行は同一タスク・同一書式の残りプールであり，
   JMMLU CSV にはタスク内の verbatim 重複すら存在する（本反復で college_physics 10 件・
   conceptual_physics 1 件を発見）．埋め込み空間で「追加行の最近傍（既存訓練行）との cos 類似度」の
   分布を出し，実効的な n の増分が名目の 1.44 倍より小さいことを定量化できる（学習曲線の外挿ずれの説明）．
5. **(M3) タスク単位の粒度**: per-domain ではなく **JMMLU タスク単位の recall Δ** を出す．
   M5 なら退行はドメイン内の特定タスクに偏らず一様に薄く出るはずで，M3（特定タスクの峰が
   境界を引っ張る）ならタスク間で偏るはずである．両者はこれで識別できる．

#### 分析 3 —— P4（hold-out 込み評価の Δ が大きい）をどう読むか

副次評価の Δ = +1.075pt は主評価の +0.322pt より大きいが，**これを「本当に追加が効いている証拠」と
読むことはできない**．

- 副次評価の内訳を分解すると，hold-out 426 行そのものの accuracy は **腕 A 0.9049 → 腕 B 0.9236，
  Δ = +1.88pt**，すなわち **426 行中わずか 8.0 行**の差である．executor はこの差に対する検定を
  報告していない．discordant がこの規模なら McNemar は有意にならない見込みで，
  **8 行の差はノイズと区別できない**．
- 「hold-out の構成がプール偏重だから」という見かけの説明も**単独では成り立たない**．
  hold-out の構成（medical 158 / natural_science 90 / history_culture 92 / business_economics 78 /
  mathematics 8）に主評価の per-domain Δ を当てた合成値は **+0.079pt** にすぎず，観測の +1.88pt を
  説明しない．つまり「構成の偏り」でも「主評価と同じ効き方」でもない量が出ている．
- 残る説明は 2 つで，どちらも肯定的な証拠にならない．(a) **8 行規模の偶然**（最も単純），
  (b) **hold-out 行と追加訓練行が同一タスクの同一プールから来ているための近接**（executor は
  verbatim 重複 0 を確認済みだが，設問幹を共有する準重複は未確認．JMMLU にタスク内重複が実在する
  以上，準重複の存在は十分あり得る）．
- なお `build_dataset.py` の `_sample_domain_questions` は `random.Random(seed)` による無作為抽出で
  あることを確認した（CSV 先頭からの切り出しではない）ので，「既存行＝CSV 前半・プール行＝後半」
  というタスク内の系統差は機序として否定できる．

**判定への影響**: P4 は「符号の整合性チェック」として事前登録されたものであり，
成立しても足切り 1・2 を代替しない．上記のとおり Δ の**大小関係**は統計的裏付けを欠くため，
判定は `closed` のまま動かない．次に同型の副次評価を置くときは，**hold-out 側にも McNemar を課し，
準重複（埋め込み cos 類似度の閾値）の監査を必須にする**こと．

#### 分析 4 —— 学習曲線 err ∝ n^(-0.127) の外挿は今後は使わない

| 反復 | 名目 n 倍率 | 外挿の予測 CV Δ | 実測 CV Δ | 実測/予測 |
|---|---|---|---|---|
| Iter95 | 1.98 | +1.73pt | +0.366pt | 0.21 |
| Iter96 | 1.44 | +0.89pt | +0.322pt | 0.36 |

2 反復とも外挿は **3〜5 倍の過大**であり，しかも外れ方の向きが一致している．理由は 3 つ考えられる．
(a) α = 0.127 は **既存 2,275 行の部分抽出**（n=451〜1804）で測ったもので，各追加行が既存行と
i.i.d. であることを前提にしている．実際の追加行は同一 56 タスクの残りプールで，書式・出題範囲が
既存行と重複しており **実効的な情報量が小さい**．(b) JMMLU にはタスク内 verbatim 重複が実在する
（本反復で発見）ため，名目 n の増分は実効 n の増分を上回る．(c) そもそも M5 が正しいなら，
総重み一定の下での行追加は学習曲線が想定する「サンプル数の増加」とは別の操作であり，
外挿の適用条件を満たしていない．

**結論: 名目 n を入れた学習曲線の外挿を，足切り水準の設定や着手可否の根拠に使うことを止める．**
どうしても事前見積りが要る場合は，実測 2 点から得た経験的な割引率（実測/予測 = 0.21〜0.36）を
明示的に掛けた値を使い，「上限の目安」としてのみ扱う．より良いのは，外挿ではなく
**実効 n（追加行の最近傍類似度で重み付けした行数）**を測ってから判断することである．

#### 分析 5 —— B160 (c)（足切り +1.0pt が厳しすぎたのではないか）の事後評価

B160 (c) は「実現可能な n が 1.44 倍に留まるため，学習曲線どおりでも足切り +1.0pt を通らない」と
指摘していた．実測を見た今，この懸念は**本反復の結論には効いていない**．

- 実測 CV Δ は +0.322pt で，外挿値 +0.89pt にすら遠く届かない．足切りを「外挿値の 60%」
  （= +0.53pt）へ緩めても不通過である．
- さらに**足切り 2（BH 後の有意退行 0 件）が 2 件で独立に不通過**であり，足切り 1 をいくら緩めても
  停止規則は発火する．すなわち **足切り水準の設定は今回の判定を左右していない．**
- したがって B160 (c) は「今回は空振りだったが，一般則としては未解決」という位置づけで
  B154 (A3)（足切り +2.3pt・効果量閾値 1.0pt の一般則の見直し）と束ねて人間へ回す．

#### 学び（次の自分が読んで分かる形）

1. **`sample_weight = n/(K·n_d)`（クラス総重み一定）の下では，訓練行の追加は「情報の追加」ではなく
   「クラス間の決定境界の再配分」である．** 行を足したクラスの discriminant は平滑化されて
   境界付近を取りに行かなくなり，**追加 0 のクラスの recall が，他クラスへの総追加行数に比例して
   上がる（実測 2 反復で 0.0051・0.0059 pt/追加行と一致）**．「n を増やす」系のレバーを引く前に，
   重み付けとの結合を必ず確認すること．単に行を足すだけでは全体 top1 は零和で動かない．
2. **同じ相対用量を与えても，Δrecall の符号はそのクラスの被覆不足度（腕 A の recall）で決まる**
   （同用量 4 ドメインで完全単調，全 10 ドメインで Spearman −0.717, p=0.020）．
   改善が要るのは education 0.535・medical 0.673 だが，**プール残量があるのは medical 側だけで，
   最も弱い education は残り 9 行**．つまり「行を足す」手はこの研究では原理的に頭打ちである
   （Iter95 の学び 3 の一般化）．
3. **Iter95 の (M1)（測定側の交絡）は，直接操作による検証で支持されなかった．**
   機序仮説を「最有力」と書いた次の反復でそれを直接検証する設計は有効だったが，
   同時に「最有力」の当たり率は高くないことを前提に，代替機序（今回の M2・M5・M6・M7）を
   最初から検証可能な形で並べておくべきである．
4. **副次評価（hold-out）は行数が小さいと簡単に過大に見える．** 426 行での +1.88pt は 8 行の差に
   すぎない．副次評価にも必ず検定を課し，「主評価より Δ が大きい」を成果として読まないこと．
5. **JMMLU の CSV にはタスク内 verbatim 重複が存在する**（college_physics 10 件・
   conceptual_physics 1 件）．Iter95 で見つかったタスク**間**重複とは別種で，
   プールを使う実装では per-task の一意化が要る．

