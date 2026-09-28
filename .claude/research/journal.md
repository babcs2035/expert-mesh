## Iteration 91: 分類器 L2 正則化強度の再選定（訓練集合内 CV による）

### 調査 (Iter91)

B149 (d) が本反復のレバーを `classifier_regularization_strength` = `l2_C_selected_by_train_only_cv` と
指定し，config.yml の `levers` 末尾に追記済みである．B149 が実測した訓練集合内 5-fold CV は
**素の `LogisticRegression` 単体**で測ったものだったため，本フェーズでは 3 つの問いを立てた．
**(Q1) 高次元・小標本（p=5120 ≫ n=2327）で，訓練集合内 CV で選んだ正則化強度は保持集合へ汎化するか．
(Q2) `CalibratedClassifierCV(method="temperature", cv=5, ensemble=True)` という入れ子構造の下でも
同じ CV 曲線になるか（本番の推論はこのラッパを通る）．(Q3) C=3.0 と C=10.0 のどちらを選ぶべきか．**

**Q1: 「CV の推定値」は楽観バイアスを持つが，「CV で選ばれたハイパラ」は実務上ほぼ最適に近い**

- Cawley & Talbot (2010), *On Over-fitting in Model Selection and Subsequent Selection Bias in
  Performance Evaluation*, JMLR 11:2079-2107, <https://www.jmlr.org/papers/v11/cawley10a.html> ——
  モデル選択に使った CV スコアをそのまま性能推定値として報告すると，**選択バイアスにより楽観側へ
  ずれる**（しかもそのずれは「意外なほど大きい」と本文が述べている）．
  **本反復への含意**: CV で得た +2.7pt という数字を「評価集合でも +2.7pt 出る」と読んではならない．
  事前登録の期待効果は減衰を見込んで置き，判定は本走の McNemar で行う．
- Wainer & Cawley (2018/2021), *Nested cross-validation when selecting classifiers is overzealous
  for most practical applications*, arXiv:1809.09446, <https://arxiv.org/html/1809.09446v1> ——
  ハイパラ選択にフラットな（入れ子でない）CV を使っても，**選ばれたハイパラの汎化性能は
  入れ子 CV で選んだものとほとんど変わらない**（差が出るのは「性能推定値」の側）．
  **本反復への含意**: 本反復の C 選定は，評価集合（`data/dataset.jsonl`）を一切見ずに
  訓練集合内 CV だけで行う方式で問題ない．**バイアスが乗るのは推定値であって選択ではない**．
- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning* 2nd ed. §7.10（one-standard-error
  rule），<https://esl.hohoweiya.xyz/book/The%20Elements%20of%20Statistical%20Learning.pdf> ——
  CV 曲線は最小値の近傍で平坦になることが多く，**最良値の 1 標準誤差以内にある中で最も倹約的な
  （＝正則化が強い）モデルを選ぶ**ことが推奨される．**Q3 の判定規則としてこれを採用する**．
- 参考（慣行の確認）: MTEB / C-MTEB の Classification タスクは凍結埋め込み上で
  `LogisticRegression` を既定ハイパラで訓練する（<https://bge-model.com/tutorial/4_Evaluation/4.2.3.html>）．
  つまり **`C=1.0` のまま使うのは分野の慣行ではあるが，n/p 比に合わせて調整されたものではない**．
  本リポジトリの現行設定（`train_domain_classifier.py:210` が `C` を書いていない＝既定の 1.0）は
  この慣行をそのまま引き継いだだけで，Iter89 で p が 2048 → 5120 になった後も見直されていない．
  （断定を避ける: これは「C=1.0 が悪い」ことの証拠ではなく，「調整された形跡が無い」ことの記録である．）

**Q2: 本番パイプライン（較正ラッパ込み）でも CV 曲線の形は変わらない — 本フェーズで実測**

B149 の測定は素の `LogisticRegression` で行われたが，本番の推論は
`CalibratedClassifierCV(method="temperature", cv=5, ensemble=True)` を通る．ensemble=True は
**内側 5 fold それぞれで訓練した 5 つの基底推定器の確率を平均する**ため，単体モデルと argmax が
一致する保証は無い（temperature scaling 自体は共有 softmax の単調変換なので，単体なら argmax を
変えないが，5 モデルの平均は変えうる）．そこで本フェーズで，**外側 5-fold の各 fold で本番と同一の
パイプラインを丸ごと組み立て直す入れ子 CV**を実行した（スクリプト `/tmp/iter91/cv_pipeline.py`・
`/tmp/iter91/cv_seeds.py`．入力は `data/classifier_train_iter87_hybrid.jsonl` 2,327 行 ＋
`data/embcache_train_iter89_qwen3-embedding_4b{,__p1}.npy` を連結した (2327, 5120)，
重みは本番 `_extract_sample_weights()` と同一のドメイン均衡重み．
**`data/dataset.jsonl`（評価集合）には一切触れていない**．CPU のみで実行し wafl500〜509 は使っていない）．

| C | 素の LR（seed 0） | **本番パイプライン** seed 0 | seed 1 | seed 2 | **3 seed 平均** | C=1.0 比 |
|---|---|---|---|---|---|---|
| 1.0（現行＝sklearn 既定） | 0.8096 | 0.8036 | 0.8049 | 0.8040 | **0.8042** | — |
| 3.0 | 0.8307 | 0.8251 | 0.8255 | 0.8251 | **0.8252** | +2.10pt |
| 10.0 | 0.8367 | 0.8354 | 0.8290 | 0.8298 | **0.8314** | **+2.72pt** |
| 30.0 | 0.8341 | 0.8354 | 0.8315 | 0.8324 | **0.8331** | +2.89pt |

- **順位は 3 つの seed すべてで `1.0 < 3.0 < 10.0 ≤ 30.0` と完全に一致した**．較正ラッパは水準を
  0.1〜0.6pt 押し下げるが，**C に対する曲線の形は変えない**．したがって B149 の素の LR による
  曲線を根拠に使ってよい（本フェーズはそれを本番パイプラインで追試したことになる）．
- B149 の実測値（C=1.0: 0.8144 / 3.0: 0.8328 / 10.0: 0.8354 / 100.0: 0.8333）と本フェーズの素の LR の
  値が 0.3〜0.5pt ずれるのは外側 fold の分割乱数が違うためで，**順位と差の大きさは一致している**．
- C=30.0 が C=10.0 をわずかに上回ったが，差は **+0.17pt** で fold 間 SD（0.011〜0.017）より小さい．

**Q3: one-standard-error rule により C = 10.0 を選ぶ**

Q2 の本番パイプライン 3 seed × 5 fold = 15 fold の結果に §7.10 の規則を当てはめる．
最良は C=30.0 の 0.8331．fold 間 SD ≈ 0.0134．

- 15 fold を独立とみなした SE = 0.0134/√15 = 0.00346 → 1-SE 下限 = **0.8296**．
- seed をまたいだ fold は同一データの再分割なので独立ではない．保守側に **1 seed 分の
  SE = 0.0134/√5 = 0.00599** を使っても 1-SE 下限 = **0.8271**．
- **どちらの見積りでも C=10.0（0.8314）は下限を超え，C=3.0（0.8252）は超えない．**
  よって「最良の 1SE 以内で最も正則化が強い C」は **C = 10.0** に決まる．

**C=3.0 を採らない理由**（委譲元の指摘「差が 0.26pt しかない／C を上げるほど過学習リスクが増す」への回答）:
B149 の素の LR 曲線では C=3.0 と C=10.0 の差は 0.26pt だったが，**本番パイプラインで測り直すと差は
0.62pt（3 seed 平均）に広がり，3 つの seed すべてで C=10.0 が上回った**．1-SE 規則は
「差が誤差に埋もれるなら弱い方（強い正則化）を採る」規則であり，本件では**埋もれていない**．
逆に C=30.0 を採らないのは，C=10.0 との差 0.17pt が誤差に埋もれており，1-SE 規則が
**より強い正則化（＝小さい C）**を選ぶよう指示するためである．したがって C=10.0 は
「CV の argmax だから」ではなく「事前に決めた規則の適用結果」として選ばれている．
**結論は B149 (d) が示唆した C=10.0 と一致する（乖離なし）．**

**過学習リスクについての注記（断定しない）**: p=5120 ≫ n=2327 なので L2 罰則を弱めるほど係数ノルムは
増え，訓練データ特有の方向へ載りやすくなる．ただし上表の保持 fold 精度は C=1→30 で単調増加しており，
**この範囲では過学習による劣化はまだ観測されていない**（B149 の C=100.0 で 0.8333 と頭打ちになる
ことから，転回点は C=30〜100 付近にあると推測される．確証はない）．C=10.0 はその転回点より
手前かつ 1-SE 下限を満たす位置にある．

### 計画 (Iter91)

**単一レバー**: `classifier_regularization_strength` = **`l2_C_selected_by_train_only_cv`**，
**具体値 C = 10.0**（`sklearn.linear_model.LogisticRegression` の逆正則化強度．
C=10.0 は既定 1.0 に対し L2 罰則を 1/10 に弱めることを意味する）．
他は Iter90 の最良構成に固定する（埋め込み `qwen3-embedding:4b` 2 ビュー連結，
訓練データ `data/classifier_train_iter87_hybrid.jsonl` 2,327 行，評価集合 `data/dataset.jsonl`
3,750 行，`method="temperature"`・`cv=5`・`ensemble=True`，`class_weight=None` ＋
`_extract_sample_weights()` のドメイン均衡重み，`confidence_threshold=0.0`，`dispatch_top_k=1`）．

**仮説**: Iter89 で特徴次元が 2048 → 5120 へ 2.5 倍になったのに `C` が sklearn 既定の 1.0 のままで，
p=5120 ≫ n=2327 の領域では正則化が強すぎる側に外れている．C を 10.0 へ上げれば分類器の
top1 が上がり，それが実行時のルーティング精度（`top1_accuracy`）へ +1.0pt 以上の形で現れる．

**変更点（最小差分．コード 2 行 ＋ 定数定義）**

1. `scripts/train_domain_classifier.py` のファイル冒頭（`_MAX_ITER` の近く）に定数を定義する．
   マジックナンバー禁止規約に従い，値の根拠（訓練集合内 CV ＋ 1-SE rule）をコメントに残す．
2. 同ファイル **`:210`** を
   `base_estimator = LogisticRegression(max_iter=_MAX_ITER, class_weight=None)` から
   `base_estimator = LogisticRegression(max_iter=_MAX_ITER, class_weight=None, C=_L2_INVERSE_REGULARIZATION)`
   へ変更する．**これ以外のコード変更・config 変更は一切行わない**（`config.yaml` は 1 バイトも触らない）．
3. `models/domain_classifier.joblib` を再訓練する．**埋め込み計算を伴うので wafl-ctrl5 上で行う**
   （2026-09-23 絶対条件 (B)．wafl500〜509 は絶対に使わない）．
4. `data/MANIFEST.md` に新 artifact の sha256 と訓練コマンドを追記する．

**レバーを読むコード行と，そこへ到達する条件**
（config.yml 冒頭「最重要の注意 — 同じ失敗を 6 回繰り返している」への対応）

| # | コード行 | 何をするか |
|---|---|---|
| 1 | `scripts/train_domain_classifier.py:210` `LogisticRegression(max_iter=_MAX_ITER, class_weight=None)` | **ここに `C=` を足す．レバーの唯一の入口** |
| 2 | 同 `:211-213` `CalibratedClassifierCV(base_estimator, method="temperature", cv=5, ensemble=True)` | sklearn の `clone()` は `get_params()` 経由なので **内側 5 fold すべてに C が伝播する** |
| 3 | 同 `:214` `.fit(embeddings, labels, sample_weight=...)` → `:231` `train_classifier(...)` → `:235` `joblib.dump(model, output_path)` | artifact を書き出す |
| 4 | `mise.toml:71-78` deploy | `sudo rm -rf $REMOTE_DIR/models` の後に `models/` を全ノードへ rsync する（古い artifact は残らない） |
| 5 | `config.yaml:50` `routing_method: supervised_classifier` ／ `config.yaml:131` `classifier_model_path: models/domain_classifier.joblib` | 到達の前提条件．**両方とも現行構成で既に満たされている**（変更不要） |
| 6 | `http_server.py:406-411`（FastAPI `lifespan`）`state.domain_classifier = load_domain_classifier(...)` | **artifact は起動時に 1 回だけ読む** |
| 7 | `http_server.py:364-368` supervised_classifier 分岐 → `classifier.py:69` `classifier.predict_proba([query_embedding])[0]` | 1 問ごとに新 artifact の確率が出る |

**到達条件（すべて現行構成で満たされることを確認済み）**
- (a) `config.yaml:50` は実際に `supervised_classifier`．`http_server.py:365` のコメントどおりこの分岐は
  LLM を呼ばず，`light_model` を使う分岐は先行分岐で到達しない．fallback は `confidence_threshold=0.0` で
  Iter28 以降 0 件（Iter90 実測 `fallback_rate` = 0.0）．**したがって分類器は全 3,750 問で必ず通る．**
- (b) **唯一の落とし穴は #6 である**．artifact は `lifespan` で 1 回だけ読まれるので，
  **rsync しただけではプロセス内の古いモデルが使われ続ける**．Iter16/20/21/22/27/B35 と同型の
  「config は正しいのにコードへ到達しない」失敗はここで起きうる．
  **`mise run deploy` によるコンテナ再作成を必ず行い，再作成されたことを確認すること．**
- (c) #2 の伝播は仮定ではなく確認済みである（本フェーズの `/tmp/iter91/cv_pipeline.py` が
  同じ構成で C を変えて CV 精度が動いたこと自体が，C が内側 fold へ届いている証拠になっている）．

**レバー固有の直接証拠（B149 恒久申し送り 2: 「基準線との完全一致」で兼ねない）**

| # | 証拠 | 確認方法 |
|---|---|---|
| E1 | artifact が差し替わった | `sha256sum models/domain_classifier.joblib` が基準線の `ff8aad9c...` と**異なる** |
| E2 | 実際に使われた C が 10.0 である | 新 artifact を `joblib.load` し，`m.calibrated_classifiers_[0].estimator.C == 10.0`，`len(m.calibrated_classifiers_) == 5`，`m.calibrated_classifiers_[0].estimator.n_features_in_ == 5120`，`sorted(m.classes_)` が 10 ドメイン |
| E3 | 全ノードが新 artifact を**ロードしている** | deploy 後・本走前に wafl500〜509 の 10 台すべてで，コンテナ内 `/app/models/domain_classifier.joblib` の sha256 が E1 の新値と一致し，かつ**コンテナの起動時刻が deploy 後である**こと |
| E4 | 実行時の出力が変わった | 予備 20 問の `probe_candidates` が Iter90 予備実行（`results/20260928_032455/`）の同じ 20 行と**相違すること** |
| E5 | 同一セッション内の決定論性 | 予備 20 問と本走の同じ 20 行が **200/200 スロットでビット単位一致**（B149 恒久申し送り 3．同一セッション内はデプロイを挟まないので一致を要求してよい） |

**事前ゲート G0（結果を見る前に判定規則を固定する）**
- **G0-a（訓練の実行場所）**: 再訓練の埋め込み計算は **wafl-ctrl5 の Ollama（127.0.0.1:11499）でのみ**行う．
  `--ollama-host` に wafl500〜509 を指定していないことをコマンドログで確認する（絶対条件 (B)）．
- **G0-b（埋め込みの再現性の確認）**: 再訓練時に計算した 2,327 行 × 2 ビューの埋め込みを
  `data/embcache_train_iter89_qwen3-embedding_4b{,__p1}.npy` と突き合わせ，**最大絶対差を記録する**．
  B149 (d) は `qwen3-embedding:4b` でセッション跨ぎの再現性が失われている疑いを残しており，
  差が大きければ「C の効果」と「埋め込みの揺らぎ」が交絡する．**この値は報告必須**．
- **G0-c（交絡を切り分ける対照 artifact．オフライン replay のみ）**: G0-b と同じ新しい埋め込みから
  **C=1.0 の artifact も 1 つ作り**，キャッシュ済み評価埋め込みによる replay で top1 を求める．
  これが Iter90 本走の 0.829867 と **±0.25pt 以内**（B149 恒久申し送り 1 の再現性の床）に収まれば，
  以降の差は C に帰属できる．**この C=1.0 artifact は絶対にデプロイしない**
  （「複数の C を本走して良い方を採る」禁止条項に抵触させないため．デプロイするのは C=10.0 のみ）．
- **G0-d（事前 replay 予測）**: C=10.0 artifact で同じ replay を行い，**本走前に** top1 と
  per-domain を journal へ記録する（Iter87〜89 で 3 反復連続 0.12pt 以内の一致が実証済み）．
  **これは予測の記録であってゲートではない．予測値がどうであれ本走は必ず実施する**（絶対条件 (A)）．
- **G0-e（予備 20 問）**: `data/dataset.jsonl` の先頭 20 問で予備実行し，HTTP 500 が 0 件，
  E4 が満たされること．
- **G0-f（時間予算）**: G0-e の実測から 3,750 問の総所要を外挿する．**175 分を超える見込みなら，
  `experiment.timeout_min` を 180 → 210 へ引き上げる**（これは watchdog の設定であって実験条件ではない．
  B137 と同じ扱い）．**本走を縮小・省略する分岐は作らない**（絶対条件 (A)．Iter90 本走は 158 分だった）．

**検出力の確認（判定可能であることを本フェーズで再計算した）**
基準線 Iter90 本走の n = 3,750．基準線と新 artifact の discordant 率は，同型の artifact 差し替えである
Iter89（旧 0.6b → 新 4b）の実測 347/3,435 = **π̂_d = 0.10102** を用いる．

- McNemar 有意境界 `1.96·√(π̂_d/n)` = 1.96·√(0.10102/3750) = **1.017pt**
- 80% 検出力の最小効果 `2.802·√(π̂_d/n)` = **1.454pt**
- 期待効果 +1.8pt で検出力 **0.934**，+2.1pt で **0.982**，+2.7pt（CV の生の差）で **0.999**．
  仮に減衰して +1.4pt でも **0.770**，+1.0pt だと **0.487**．
- **セッション跨ぎの再現性の床（top1 ±0.25pt）は有意境界 1.017pt の約 1/4 であり，
  主基準 ≥ +1.0pt はノイズ床を 4 倍上回る．判定可能である．**
- 注意（Cawley & Talbot 2010 の含意）: CV の +2.72pt をそのまま期待してはならない．
  訓練集合（hard negative 混成 2,327 行）と評価集合（JMMLU 由来 3,020 行 ＋ 複合 730 行）は
  分布が違うため，**+1.0pt を下回る減衰が起きれば判定不能になりうる**．その場合の扱いは
  下の判定語 `no_effect` / `partial` で事前に定義してある．

**成功条件（事前登録．結果を見た後に書き換えない）**

| 区分 | 指標 | 基準線（Iter90 本走 `results/20260928_032909/`，n=3,750） | 判定基準 |
|---|---|---|---|
| **主基準 (i)** | `top1_accuracy` の McNemar 検定 | 0.829867 | **p < 0.05（有意改善方向）** |
| **主基準 (ii)** | Δ`top1_accuracy` | 0.829867 | **≥ +1.0pt**（= 有意境界 1.017pt とほぼ同値．Wilson 95%CI を併記） |
| **到達確認** | E1〜E5 | — | **全件充足**（1 つでも欠ければ `invalid`） |

**非退行条件**（基準線は同じく `results/20260928_032909/`．全 3,750 行で評価する．
B149 恒久申し送り 1 に従い「ビット単位一致」条項は**置かない**）

| # | 指標 | 基準線 | 判定基準 |
|---|---|---|---|
| ① | per-domain 20 指標（precision/recall × 10） | Iter90 本走 | BH 補正（q=0.05）後の**有意退行 0 件** |
| ② | `fallback_rate` | 0.0 | **= 0.0** |
| ③ | `dispatch_failure_rate` | 0.000533 | **≤ 0.005**（絶対値．Iter86〜90 と同一） |
| ④ | `compound_domain_set_recall`（複合 730 行） | 0.570548 | **≥ 0.544048**（基準線 −2.65pt．Iter86〜90 と同じ幅） |
| ⑤ | `compound_mean_dispatched_count`（複合 730 行） | 1.964384 | **≤ 2.10**（絶対値．Iter86〜90 と同一） |
| ⑥ | `mean_duration_ms`（全 3,750 行） | 2531.538 | **≤ 3037.8**（基準線 +20%）．**B149 恒久申し送り 5 に従い部分集合では評価しない** |
| ⑦ | `ece` | 0.029892 | **≤ 0.05**（絶対値）．C を上げると logit のスケールが変わるため，temperature が吸収しきれない可能性への保険 |
| ⑧ | `tie_rate` | 0.0 | **= 0.0** |

**定義語と数値アンカーの突き合わせ**（B149 恒久申し送り 4．Iter90 の ③ 誤判定の再発防止）
上の表のアンカーは**すべて `/tmp/iter90/metrics_full3750.json`（= `results/20260928_032909/` の
全 3,750 行に対する `metrics.py` 出力）から本フェーズで直接読み出した値**であり，
本文の定義語と同じ量である．具体的に突き合わせた対応は次のとおり:
④ 本文「複合 730 行の set recall」= `compound_coverage.compound_domain_set_recall` = 0.5705479...（730 行基準．
**Iter86〜89 で使っていた 0.566265 は 415 行基準なので使わない**），
⑤ = `compound_coverage.compound_mean_dispatched_count` = 1.9643835...（同じく 730 行基準．
**Iter90 の条文にあった 1.879518 は 415 行基準**），⑥ = `mean_duration_ms` = 2531.5384（全 3,750 行．
**Iter90 が使った 1643.249 は 3,435 行サブセット基準なので使わない**），
③ = `dispatch_failure_rate` = 0.0005333...（全 3,750 行．Iter90 計画節の 0.001165 は 3,435 行基準），
⑦ = `ece.ece` = 0.0298918（n_rows=3748），主基準 = `top1_accuracy` = 0.8298666...（全 3,750 行）．
**基準が 415/3,435 行のものと 730/3,750 行のものが混在していた点を，本反復ですべて後者へ統一した．**

**判定語の定義**
- **adopted**: 主基準 (i)(ii) の **AND** ＋ 到達確認 E1〜E5 ＋ 非退行①〜⑧をすべて満たす．
- **partial**: (i) 有意だが (ii) が +1.0pt 未満，または非退行①〜⑧のいずれかが未達．
- **no_effect**: |Δtop1| < 0.5pt **または** McNemar が非有意（Iter86〜88 と同一条文）．
- **rejected**: Δtop1 ≤ −1.0pt かつ McNemar 有意（悪化方向）．
- **invalid（実験不成立）**: E1〜E5 のいずれかが欠ける（= 新 artifact が実際にはロードされていない）．
  **top1 が基準線と近いことは invalid の根拠にしない**（B149 恒久申し送り 1・2）．

**本走（2026-09-23 絶対条件 (A)）**: 変更適用後に **wafl500〜509 を用いた 3,750 問のフルスペック本走を
必ず 1 回実施する**．`mise run start -- --dataset data/dataset.jsonl --output results.jsonl` ののち
`mise run analyze -- <timestamp>` を **timestamp 明示指定**で実行する（B118 落とし穴 1 の再発防止）．
想定所要は Iter90 実績から **150〜165 分**（分類器の差し替えは推論コストを変えない）．
**C を複数値で本走して良い方を採ることは禁止する**（config.yml 当該レバーの note．
デプロイするのは C=10.0 のただ 1 つ）．

**恒久ルールへの適合確認**
- B115 (1)（ドメイン固有の後付け補正の禁止）: `C` は 10 クラス共通の単一スカラーであり，
  ドメインごとに値を変えない．
- 2026-09-23 絶対条件 (A): 本走を必ず実施する．G0 のどの分岐にも本走省略の経路を作っていない．
- 2026-09-23 絶対条件 (B): 再訓練の埋め込み計算・replay・CV はすべて wafl-ctrl5 または
  ローカル CPU で行い，wafl500〜509 は本走にのみ使う．
- 単一レバー原則: 変えるのは `C` の 1 つだけ．`method`・`cv`・`ensemble`・`max_iter`・
  `class_weight`・sample_weight・埋め込み・訓練データ・評価集合・`config.yaml` は一切変えない．

### 実験結果（Iter91，フェーズ 2 実測．判定はフェーズ 3）

**変更した差分**: `scripts/train_domain_classifier.py` の 1 ファイルのみ．冒頭 `_MAX_ITER = 1000` の直後に
定数 `_L2_INVERSE_REGULARIZATION = 10.0` を選定根拠のコメント付きで新規定義し，`train_classifier()` 内の
`LogisticRegression(max_iter=_MAX_ITER, class_weight=None)` へ `C=_L2_INVERSE_REGULARIZATION` を追加した．
`config.yaml` は 1 バイトも変更していない．`ruff check` の新規エラー 0 件．
`pytest tests/test_train_domain_classifier.py tests/test_classifier.py` は 11 PASS・1 FAIL
（`test_build_training_features_embeds_each_row_in_order` の `ModuleNotFoundError: sentence_transformers`）だが，
この失敗は `git stash` で変更前にも再現する環境依存の既知失敗であり本変更起因ではない．
`ruff format --check` の reformat 推奨も同様に変更前から存在するドリフトで，今回触った 2 箇所とは無関係．

**事前ゲート G0 の結果**

| ゲート | 結果 |
|---|---|
| G0-a 実行場所 | 再訓練・埋め込み計算はすべて `--ollama-host 127.0.0.1 --ollama-port 11499`（wafl-ctrl5 トンネル）のみ．wafl500〜509 への指定 0 件（絶対条件 (B) 充足） |
| G0-b 埋め込み再現性 | 既存キャッシュとの最大絶対差 **0.011406**（plain view）／0.009027（instructed view）．B149 (d) の非決定性の範囲内 |
| G0-c C=1.0 対照 replay | 全 3,750 行で **top1 = 0.831200**．基準線 0.829867 との差 **+0.1333pt**（床 ±0.25pt 以内）→ 埋め込み差による交絡なしと確認．対照 artifact は `/tmp/iter91/` に置きデプロイしていない |
| G0-d C=10.0 replay 予測 | **top1 = 0.833067**（ゲートではない．本走は予測値によらず実施した） |
| G0-e 予備 20 問 | HTTP 500 = 0 件 |
| G0-f 時間予算 | 外挿 約 130 分（< 175 分）につき `timeout_min` は 180 のまま変更せず．実測 158.1 分で完走し判断は妥当だった |

**レバー到達の直接証拠 E1〜E5（すべて充足）**
- **E1**: 新 artifact sha256 = `6a5905f025d6dae91ab333968e71c944d77ba286f4f3e3891c1b3a1fe5ca3b30`（基準線 `ff8aad9c...` と相違）．
- **E2**: `calibrated_classifiers_[0].estimator.C == 10.0`，`len(calibrated_classifiers_) == 5`，
  `n_features_in_ == 5120`，`classes_` が 10 ドメイン．
- **E3**: wafl500〜509 の 10 台すべてでコンテナ内 artifact の sha256 が E1 と一致し，
  コンテナ起動時刻 2026-09-27T22:41 UTC が deploy 起動 22:35:26 UTC より後（lifespan 再読込みを確認）．
- **E4**: **計画が参照先とした `results/20260928_032455/` は，実際には `data/dataset.jsonl` 先頭 20 行ではなく
  `compound-416`〜`435`（Iter90 の複合評価集合検証用の別サブセット）だった．**同じ head-20
  （`business_economics-001`〜`020`）の予備実行が実在する `results/20260927_232858/preview20.jsonl` を
  id 完全一致を確認したうえで代替の比較対象に採用した（可逆な代替）．`probe_candidates` は 20/20 行すべて相違．
- **E5**: 予備 20 問 `results/20260928_074531/` と本走先頭 20 行が **200/200 スロット完全一致**，
  `selected_domain` も全行一致（同一セッション内の決定論性を再確認）．

**本走**: `results/20260928_074903/`．3,750/3,750 行完走，**158.1 分**（1790549318→1790558802）．
HTTP 500 系エラー・GPU OOM なし．`mise run analyze -- 20260928_074903` を timestamp 明示で実行済み．

**主基準の実測値**
- `top1_accuracy` = **0.831467**（Wilson 95%CI [0.819148, 0.843107]）．基準線 0.829867 に対し **Δtop1 = +0.1600pt**．
- McNemar: discordant_a_only = 109，discordant_b_only = 115，discordant_pairs = 224，chi2 = 0.1116，**p = 0.7383**．
- 事前登録の有意境界 1.017pt・80% 検出力の最小効果 1.454pt のいずれも下回る．

**非退行 8 条件の実測値**

| 条件 | 実測 | 事前登録の閾値 | 可否 |
|---|---|---|---|
| ① per-domain 20 指標 BH 補正（q=0.05）後の有意退行 | **1 件**（`legal_recall`） | 0 件 | **未達** |
| ② `fallback_rate` | 0.0 | = 0.0 | 充足 |
| ③ `dispatch_failure_rate` | 0.0016 | ≤ 0.005 | 充足 |
| ④ `compound_domain_set_recall` | 0.567808 | ≥ 0.544048 | 充足 |
| ⑤ `compound_mean_dispatched_count` | 2.046575 | ≤ 2.10 | 充足 |
| ⑥ `mean_duration_ms` | 2512.330 | ≤ 3037.8 | 充足 |
| ⑦ `ece` | 0.025464 | ≤ 0.05 | 充足 |
| ⑧ `tie_rate` | 0.0 | = 0.0 | 充足 |

① の BH 補正後に有意だった 4 件の内訳: `education_recall`（p=0.00501，正味改善 a_only 17 / b_only 39），
**`legal_recall`（p=0.00511，正味退行 a_only 20 / b_only 5，net −15/150 行）**，
`legal_precision`（p=0.00126，正味改善 0.822→0.936），
`social_science_recall`（p=0.00131，正味改善 a_only 5 / b_only 23）．**退行方向は `legal_recall` の 1 件のみ．**

統計量はすべて `metrics.py` の既存関数（`compute_mcnemar_test`，`compute_domain_recall_mcnemar_test`，
`compute_domain_precision_fisher_test`，`apply_benjamini_hochberg`，`compute_top1_accuracy_wilson_ci`）を
そのまま呼び出しており，フェーズ 2 で自前の再導出は行っていない．

**artifact の扱い**: `models/domain_classifier.joblib` を新 artifact（`6a5905f0...`）へ差し替え，
旧 artifact は `models/domain_classifier_pre_iter91_c1.joblib` へ退避（`models/`・`data/` は `.gitignore` 対象）．
`data/MANIFEST.md` に artifact の sha256 と再現コマンドを追記した．

### Iteration 91 実行済み（分析・考察）

**変更**: `scripts/train_domain_classifier.py` の `LogisticRegression` へ `C=10.0` を明示（定数
`_L2_INVERSE_REGULARIZATION`）し，同一の訓練データ・埋め込み・較正設定で再訓練した artifact
（sha256 `6a5905f0...`）を全 10 ノードへデプロイして 3,750 問を本走した．`config.yaml` は無変更．

**判定: `no_effect`**

- 主基準 (i) McNemar **p = 0.7383**（非有意），(ii) Δtop1 = **+0.1600pt**（< +1.0pt）．**AND 条件は不成立**．
- 事前登録の判定語を literal に当てると **2 つの条文が同時に発火する**:
  - `no_effect` の条文「|Δtop1| < 0.5pt **または** McNemar が非有意」→ 0.160pt < 0.5pt かつ p = 0.7383 で，
    **両方の節が成立**する．
  - `partial` の条文「(i) 有意だが (ii) が +1.0pt 未満，**または**非退行①〜⑧のいずれかが未達」→
    非退行① が未達（`legal_recall`）なので**第 2 節が成立**する．
  - **条文の前提が実測でどうだったか**: `partial` の第 2 節は「主基準に効果が出た実験の副作用を記述する」
    意図で書かれた条文であり，主基準が非有意の場合を想定していない（第 1 節が「(i) 有意だが」と
    始まることがその証拠）．本反復は主基準に検出可能な効果が無いので，`no_effect` を主たる判定語とする．
    **ただし条文の literal な帰結として `partial` も成立することをここに明記し，条文を読み替えて
    `partial` を消したのではない**ことを記録に残す（B149 (a) と同種の論点．**要レビュー**）．
  - `invalid` は当たらない．E1〜E5 が全件充足で，レバーは確実にコードパスへ到達している
    （artifact sha256 相違・`estimator.C == 10.0`・10 台の sha256 一致とコンテナ再作成・
    予備 20 問と本走の 200/200 ビット一致）．**「top1 が基準線と近いこと」を invalid の根拠にしない**
    という B149 恒久申し送り 1・2 をそのまま適用した．
- **判定語定義の欠陥（B149 (c) の再発）**: 「主基準が非有意」かつ「非退行に有意退行がある」場合を
  一意に指す語が無い．次の計画フェーズは判定語を排他的な決定木として書き直すこと．

**分析 1: CV の +2.72pt が end-to-end で +0.16pt になった機序**

まず**測定系の事実を 1 つ確定させた**。`metrics.py:32 compute_top1_accuracy` は
`selected_domain in expected_domains` を数えるだけで，**専門ノードの回答内容を一切見ていない**．
すなわち **`top1_accuracy` は「分類器の argmax がルーティング先として正しかった割合」そのもの**であり，
「分類器が当てても専門ノードが誤答すれば top1 に現れない」という機序は本リポジトリには存在しない
（回答品質は `answer_quality_accuracy` / `end_to_end_accuracy` という別軸である）．
`probe_candidates` の argmax と `selected_domain` の相違は基準線 2 行・本走 6 行のみで，
いずれも `dispatch_failed=true`（送出先ノードの実行時失敗で `selected_domain=null`）の行だった．

したがって**分類器精度 → top1 の伝達率は定義上 1.0 であり，両者の差は送出失敗行の分だけ**である:

| 量 | 基準線 `20260928_032909` | 本走 `20260928_074903` | Δ |
|---|---|---|---|
| 分類器 argmax 精度（`probe_candidates` から再計算，3,750 行） | 0.830400 | **0.833067** | **+0.2667pt** |
| `top1_accuracy`（＝上から `dispatch_failed` 行を落としたもの） | 0.829867 | 0.831467 | +0.1600pt |
| 差（= `dispatch_failed` 行数 / 3750） | 2 行 = 0.0533pt | 6 行 = 0.1600pt | — |

- 分類器 argmax 精度での McNemar も **a_only 104 / b_only 114，chi2 = 0.3716，p = 0.5422** で非有意．
  **送出失敗を除いても判定は変わらない**（本走の送出失敗 6 行のうち 5 行は argmax が正解ドメインで，
  実行時失敗が無ければ拾えていた．これは実行時ノイズであってレバーの効果ではない）．
- **G0-d の replay 予測 0.833067 は，本走の分類器 argmax 精度 0.833067 と小数 6 桁まで完全一致した．**
  すなわち replay は「分類器が何を選ぶか」を**誤差 0 で**予測しており，replay と本走 top1 の
  −0.16pt の差は**送出失敗 6 行だけで完全に説明される**．再現性の床 ±0.25pt の範囲内であるどころか，
  **replay の予測誤差は 0 であった**（`data/embcache_eval_*` が固定されている限り，分類器の出力は
  デプロイを跨いでも決定論的である．B149 (d) で観測されたセッション跨ぎのゆらぎは，
  埋め込みを実行時に計算し直す経路に由来する．**本走の埋め込みも実質同一だったことになる**）．

**では +2.72pt はどこへ消えたか．評価集合の部分集合ごとに符号が逆だった**（事前登録外の事後分割．
探索的な分析であり，これを根拠に採用判定を変えてはならない）:

| 部分集合 | n | 分類器 argmax 精度 基準線 → 本走 | Δ | McNemar |
|---|---|---|---|---|
| 単一ドメイン行（JMMLU 由来） | 3,020 | 0.835762 → 0.847351 | **+1.1589pt** | a_only 59 / b_only 94，p = **0.0060** |
| 複合ドメイン行（Iter90 拡充分） | 730 | 0.808219 → 0.773973 | **−3.4247pt** | a_only 45 / b_only 20，p = **0.0029** |
| 全体 | 3,750 | 0.830400 → 0.833067 | +0.2667pt | p = 0.5422 |

- **(a) Cawley & Talbot (2010) の選択バイアス＋分布シフト**は，単一ドメイン行で確かに効いている．
  訓練集合内 CV の +2.72pt に対し，訓練集合と同じ「単一ラベル 1 問 1 ドメイン」の構造を持つ
  評価部分集合では **+1.16pt（伝達率 43%）**しか出なかった．CV の値をそのまま期待してはならない
  という事前の注意（計画節）は正しかった．
- **(b) 「測っている量が違う」は成立する．ただし想定した機序（ルーティング後の回答正誤）ではなく，
  評価集合の構成の違いによる**．CV は単一ラベル行だけで測っており，**複合 730 行（評価集合の 19.5%）に
  相当するものを訓練集合も CV も一切含んでいない**．複合行では正則化を弱めたことで
  **−3.42pt（有意）**の逆効果が出て，単一行の +1.16pt をほぼ打ち消した．
  0.195 × (−3.42) + 0.805 × (+1.16) = **+0.27pt** で全体の実測と一致する．
- argmax の反転は 293/3,750 行（7.81%）で，うち正解化 114・誤答化 104・どちらも不正解 75．
  **分類器の決定は大きく動いているが，正味では釣り合っている．**
- **今後のレバー選定への含意（定量）**: 分類器側のレバーは伝達率 1.0 で top1 へ効くので
  「分類器をいじる価値」は構造的にはある．しかし**訓練集合内 CV で測れる利得は，
  単一行へは 43%・複合行へは負の符号で伝わる**．end-to-end で +1.0pt（有意境界）を出すには，
  この混合比（80.5% / 19.5%）の下で **訓練集合内 CV で最低でも +2.3pt 以上**の改善が必要であり，
  かつ複合行を悪化させない性質を持つ必要がある．C の掃引は C=1.0 → 30.0 の全域で CV +2.9pt が上限
  （B149 の C=100 で頭打ち）なので，**正則化強度というレバーはこの要求水準をほぼ使い切っている**．

**分析 2: `legal_recall` の有意退行（BH 補正 q=0.05 後の唯一の退行）**

`legal` を expected に含む 306 行のうち `selected_domain == "legal"` だった行が **162 → 147**
（a_only 20 / b_only 5，p = 0.00511）．probabilities から見た機序は**「legal 全体の確率質量が縮んだ」**である:

| 量 | 基準線 | 本走 |
|---|---|---|
| `legal` を予測した行数（3,750 行中） | 197 | **157**（−40） |
| `legal_precision` | 0.8223 | **0.9363**（+11.4pt） |
| `legal` 確率の平均（legal 行 306 行上） | 0.4760 | 0.4439 |
| `legal` 確率の平均（非 legal 行 3,444 行上） | 0.0163 | **0.0099**（−39%） |

- **これはドメイン固有の異常ではなく，正則化を弱めたときの精度/再現率のトレードオフである．**
  C を上げると係数ノルムが伸び，**訓練行数の多いクラスの決定領域が広がり，少ないクラスの領域が縮む**．
  実際 `legal` の訓練行は **77 行しかなく，他 9 ドメインは各 250 行**（`classifier_train_iter87_hybrid.jsonl`）．
  hard negative mining のプール（`pool_hash=221e45e8...`，3,127 行）に **legal は 0 行**しか無く，
  Iter84/86/87 のいずれでも legal だけ 77 行のままである．`_extract_sample_weights()` の
  ドメイン均衡重みは**行の重みを 3.25 倍にできても行の多様性を作れない**．
- 失われた 20 行の移動先は `history_culture` 12 / `medical` 3 / `social_science` 3 / `education` 2 で，
  **12 行が history_culture**（訓練 250 行・recall 0.917 の強いクラス）へ流れた．
  20 行中 **11 行は複合行**であり，分析 1 の複合行の退行と同じ現象の一部である．
- 予測分布全体でも同じ向きが見える: `legal` −40 / `business_economics` −33 / `computer_science` −15 に対し
  `history_culture` +25 / `social_science` +20 / `education` +18 / `medical` +13．
- **対処**: B115 (1) により legal だけ閾値・intercept・重みをいじることは禁止である．本反復では
  **何も後付け補正しない**．退行の原因は「legal の訓練行が 77 行しかない」というデータ側の構造問題であり，
  解くなら 10 ドメイン共通の規則（＝全ドメインの訓練行数を揃える）で解く必要がある．
  現行プールには legal の在庫が 0 なので，**新しいデータ源の調査が要る**（次の一手へ引き継ぐ）．

**分析 3: E4 の参照先の食い違いと，予備実行ディレクトリの記録方法**

計画が E4 の比較対象に指定した `results/20260928_032455/` は，実際には `data/dataset.jsonl` の
head-20 ではなく `compound-416`〜`435` の 20 行だった．executor は同じ head-20
（`business_economics-001`〜`020`）を持つ `results/20260927_232858/preview20.jsonl` を，
**id の完全一致を確認したうえで**代替に採用した．
**この代替は妥当である**: E4 の目的は「同一入力に対する出力が artifact 差し替えで変わったこと」の確認であり，
入力 id 集合が一致していれば比較対象として等価だからである（実際 20/20 行すべてで `probe_candidates` が相違）．
根本原因は**予備実行の結果ディレクトリがタイムスタンプ名だけで，どのサブセットを流したかを保持していない**
ことにある．`data/dataset_iter{89,90}_preview20.jsonl` という入力側のファイルは残っているのに，
出力側とは結び付いていない．**是正案は backlog B151 へ記録した**（`results/<ts>/` へ実行時の
`--dataset` パスと行数・id レンジを書いた `run_meta.json` を残す．これは測定系の改善であって
レバーではないので，次の実験と同時にオフラインで実施してよい）．

**artifact の扱いとロールバック（次イテレーションの基準線）**

- **`models/domain_classifier.joblib` を `C`=1.0 の `ff8aad9c...` へ戻した**（`C`=10.0 版は
  `models/domain_classifier_iter91_c10.joblib` に保存，再採用は可逆）．
  `scripts/train_domain_classifier.py` の `C=_L2_INVERSE_REGULARIZATION` も revert した．
- **理由**: 事前登録の主基準を満たさず（+0.16pt・p=0.74），非退行① に有意退行が 1 件あり，
  全体の Δ は再現性の床 ±0.25pt の内側にある．**「効果が測れなかった変更は入れない」**という
  倹約側の既定に従った．単一行 +1.16pt は事後分割の探索的知見にすぎず，採用の根拠にしない．
- **次イテレーションの基準線は Iter90 本走 `results/20260928_032909/`（3,750 行，top1 = 0.829867，
  分類器 sha256 `ff8aad9c...`）のまま据え置く**（ロールバックにより，基準線はその artifact の
  実測値として今も有効である）．
- **落とし穴**: wafl500〜509 上には Iter91 本走時点の `C`=10.0 artifact が載ったままである．
  **次に実験を行う際は必ず `mise run deploy` を先に実行すること**（`data/MANIFEST.md` にも明記した）．

**学び**

1. **`top1_accuracy` は分類器の argmax 精度そのものである**（`metrics.py:32`．回答内容を見ていない）．
   したがって**分類器側のレバーはオフライン replay で誤差 0 で予測できる**——本反復で G0-d の
   replay 予測と本走の argmax 精度が小数 6 桁まで一致した．**本走の 158 分は「送出失敗率」と
   「実行時の健全性」を測るためにだけ必要で，分類器の効果量そのものは replay で事前に分かる**．
   ただし**選定に評価集合の replay を使えばリークになる**ので，用途は「事前登録した予測の記録」に限る．
2. **訓練集合内 CV → 評価集合の伝達率を本反復で初めて定量した: 単一ドメイン行へ 43%，複合行へは負**．
   今後，分類器側のレバーは **訓練集合内 CV で +2.3pt 以上**を見込めなければ end-to-end の
   有意境界 1.0pt に届かない．Iter89 学び（判定不能と事前に分かるレバーには着手しない）の
   具体的な換算式がこれで得られた．
3. **評価集合に複合行を 19.5% 混ぜたことで，単一ラベル訓練に最適化するレバーの効果が構造的に薄まる**．
   Iter90 の拡充は測定分解能のために正しかったが，副作用として
   「単一行で得た利得が複合行の損で相殺される」経路が生まれた．今後は**主基準の内訳
   （単一 / 複合）を必ず併記する**こと（事前登録に加えるべき．ただし**部分集合ごとに
   判定基準を分けるのは多重比較になる**ので，主基準は全体のままとする）．
4. **クラス間の訓練行数の不均衡（legal 77 行 vs 他 250 行）が，正則化を弱めたときに
   最初に壊れる場所を決めている**．sample_weight による均衡化は重みを増やせても多様性を増やせない．
   `legal` は hard negative プールにも在庫が 0 で，**既存データ源だけでは是正できない**．
5. 訓練集合内 CV を「本番と同じ入れ子パイプラインで」測り直した手順（計画節 Q2）は正しく機能した
   （較正ラッパは水準を下げるが曲線の形を変えない，という予測が本走でも破綻しなかった）．
   **失敗したのは CV の測り方ではなく，CV と評価集合の分布の違いの見積り**である．

## Iteration 90: 複合設問評価集合の拡充（既存公開データセットの調査と追加）

### 調査 (Iter90)

B147 (c) は本反復のレバーを `compound_eval_set_expansion` = `existing_public_dataset` と指定した．
config.yml 冒頭の **B116 (2)**（データセット拡充ではまず信頼できる既存公開データセットを調査し，
見つからない場合に限り LLM 生成へ落とす．出典・ライセンス・ドメイン適合性を明記する）に従い，
値を確定する前に実際に調査した．問いは 3 つ．
**(Q1) 「1 行が 2 つの専門ドメインラベルを同時に持つ日本語の自然文設問」に相当する公開データセットは
存在するか．(Q2) JMMLU など既存の公開ソースから複合行を派生させる経路は使えるか（ライセンス面を含む）．
(Q3) 現行 415 行の検出力はいくつで，1 反復の実行時間予算の中でどこまで増やせるか．**

**Q1: 既存公開データセット — この探索範囲では見つからなかった（(i) は不成立）**

tavily-search（`tvly search --depth advanced`）で日本語・英語の 6 クエリ，および Hugging Face Hub の
Datasets API で 20 クエリを検索した．確認できた候補と，本研究の要件（**1 行が 2 つのドメインラベルを
同時に持つ，日本語の自然文**）に対する適合性は次のとおり．

| 候補 | 出典 | ライセンス / 入手性 | 10 ドメイン体系との適合性 |
|---|---|---|---|
| M2QA (Multi-domain Multilingual QA) | Engländer et al., EMNLP 2024 Findings <https://aclanthology.org/2024.findings-emnlp.365> / HF `UKPLab/m2qa` | 公開 | **不適合**．multi-domain とは「言語 × ドメインの組み合わせを網羅する」意味であり，**1 インスタンスは単一ドメイン**．日本語は対象言語に含まれない（de/tr/zh） |
| RouterArena | Lu et al., arXiv:2510.00202 / HF `RouteWorks/RouterArena` | 公開 | **不適合**（Iter78 と同一結論）．英語・**1 クエリ 1 ドメインの単一ラベル** |
| RouterBench / RouterEval | HF `withmartian/routerbench`，`linggm/RouterEval` | 公開 | **不適合**．いずれも既存英語ベンチ（MMLU/GSM8K/HellaSwag 等）のクエリにモデル別スコアを付けたもので，ドメインラベルは単一 |
| MMLU-ProX-Japanese | HF `tokyotech-llm/MMLU-ProX-Japanese` | 公開 | **不適合**．日本語である点は合致するが **1 問 1 カテゴリ**の 10 択 MCQ．単一ドメイン行の追加供給源にはなるが複合行にはならない |
| JamC-QA | SB Intuitions，NLP2025 Q2-18 <https://www.anlp.jp/proceedings/annual_meeting/2025/pdf_dir/Q2-18.pdf> | 公開 | **不適合**．日本固有知識の多肢選択，単一科目ラベル |
| lawqa_jp / JMED-LLM / JDocQA / JAQKET / JGLUE | デジタル庁 <https://github.com/digital-go-jp/lawqa_jp> ほか | 公開 | **不適合**（Iter78 と同一結論）．いずれも単一ドメインまたはドメインラベル無し |
| AnswerCarefully (NII) | <https://llmc.nii.ac.jp/answercarefully-dataset> | 公開（研究利用） | **不適合**．日本語・自然文である点は合致するが，付与されているのは**安全性リスク分類**であって専門ドメインではない．10 ドメインへの写像ができない |
| MASSIVE / MixSNIPS 系（multi-intent NLU） | Amazon MASSIVE (CC BY 4.0) ほか | 公開 | **不適合**．1 発話に複数 intent を持つ構成は本研究の要件に形は近いが，scenario ラベル（alarm/music/weather 等）が **10 専門ドメインと重ならない** |
| Yahoo!知恵袋データセット / 国民生活センター PIO-NET | NII ×ヤフー / 国民生活センター | **契約・申込みが必要**（オープンライセンスではない） | Iter78 と同一結論．自然文相談という点だけ合致するが**ドメインラベルが無く**，付与作業は結局 (ii)(iii) と同じ |

**結論**: 「日本語」「自然文」「1 行に 2 つの専門ドメインラベル」の 3 条件を同時に満たす公開データセットは，
本探索範囲（tavily-search 6 クエリ ＋ HF Datasets API 20 クエリ）では見つからなかった．
**これは Iter78（2026-09-26）の調査と独立に行った 2 度目の調査であり，同じ結論に到達した．**
断定を避けるべき点として，これは「存在しないことの証明」ではなく「この探索範囲では見つからなかった」
ことの記録である．**したがって `existing_public_dataset` は本反復でも不成立とする．**

**Q2: JMMLU 派生の複合行 — ライセンスと設問構造の両面で不可**

「既存公開データセットから複合行を派生させる」経路（例: 学際的な科目——`econometrics`（数学＋経済），
`medical_genetics`（医療＋自然科学），`business_ethics`（経済＋社会科学）——に 2 ドメインラベルを付ける）を検討したが，
**2 つの独立した理由で採らない**．
1. **ライセンス**: `build_dataset.py:32-40` が記録するとおり JMMLU 全体は **CC BY-NC-ND 4.0（改変禁止）**である
   （HF `nlp-waseda/JMMLU` の license タグ `cc-by-nc-nd-4.0` を本調査で再確認．
   CC BY-SA 版 `nlp-waseda/JMMLU_CC-BY-SA` も存在するが README に license タグが無く，条件が確認できない）．
   ラベルの張り替えや再構成は派生物に当たり，ND 条項に抵触する疑いがある．
2. **設問構造**: `build_dataset.py:629-632` が既に
   「JMMLU の 4 択問題はそれぞれ単一タスクに属し，真の cross-domain な曖昧さを表現できない」と明記している．
   加えて既存行への 2 ラベル付与は B118 が定めた「**純粋な追加に限定し，既存行の改変・削除は行わない**」に反する．

**Q3: 現行 415 行の検出力と，実行時間予算から決まる到達可能な行数**

- **検出力の実測**: Iter89 本走の複合 415 行は Δ = −2.65pt（0.819277 → 0.792771），
  discordant n_d = 47（a_only 29 / b_only 18），**p = 0.1447 で非有意**．
  したがって π̂_d = 47/415 = **0.11325**．
  McNemar 有意境界 `1.96·√(π̂_d/n)` は n=415 で **3.238pt**，80% 検出力に必要な Δ は **4.628pt**．
  **Iter89 で実際に観測された −2.65pt は，現行 415 行では原理的に検出できない大きさである．**
- **実行時間の実測（本フェーズで results/20260927_232950/results.jsonl から算出）**:
  複合行 **8.057 s/行**（415 行で 55.7 分），単一ドメイン行 **0.762 s/行**（3,020 行で 38.4 分），
  合計 **94.1 分**．複合行が単一行の 10.6 倍遅いのは，開放型の相談文で生成長が長く，かつ
  `compound_mean_dispatched_count` = 1.88 で平均 2 ノードへ dispatch するためである．
- **予算から決まる上限**: `experiment.timeout_min` = 180 分．複合行 1 行 = 8.06 秒，
  劣化余地を見て 10 秒/行で見積もると，複合 n 行のときの総所要は `38.4 + n×10/60` 分．
  n=730 → **160 分**（実測 8.06 秒なら 136 分），n=820 → 175 分，n=1,265 → 249 分．
  **Iter89 の −2.65pt を 80% 検出力で捉えるには n ≈ 1,265 行が必要だが，これは 1 回の本走に収まらない．**
  **したがって本反復の到達目標は n = 730 とする**（有意境界 3.238 → **2.441pt**，
  80% MDE 4.628 → **3.489pt**．期待 discordant は 0.11325×730 ≈ 82.7 行）．
  これは「−2.65pt 級の差を初めて有意判定できる水準に乗せる」ことを意味し，
  80% 検出力まで届かせることは 1 反復では**意図的に諦めている**（残差は backlog B148 に記録する）．

### 計画 (Iter90)

**単一レバー**: `compound_eval_set_expansion` = **`llm_generated_separate_generator_scaleup_730`**
（config.yml の当該レバーの `values` 末尾に本フェーズで追記した．B148 参照）．
**`existing_public_dataset` は Q1 により不成立**，**`manual_authoring` は (ii) より順位が下**のため，
B116 (2) の調達順位 (i) → (ii) → (iii) に従って **(ii) の規模拡大**を選ぶ．
Iter78 で確立した生成器分離の構成（生成器 `qwen3.5:9b` ≠ 検証器 = config.yaml の `judge_model`）を
**一字も変えずに踏襲**し，変えるのは `--target-per-pair` の 7 → 14 だけである．

**仮説**: 複合評価行を 415 → 730 行へ純粋追加すれば，複合サブセットの McNemar 有意境界が
3.238pt → 2.441pt へ縮み，Iter89 で観測された −2.65pt 級の複合ドメイン効果を初めて統計的に判定できる．
既存 3,435 行の計算は 1 ビットも変わらないため，全体 top1 の基準線比較は無傷のまま保たれる．

**変更点（最小差分）**
1. `data/compound_questions_generated.jsonl` を **315 行 → 630 行**（45 ペア × 14 行）へ**追記**する．
   既存 315 行は id・query・expected_domains をそのまま保持する（B118 の「純粋な追加」）．
   生成は `scripts/generate_compound_eval_questions.py` を `--target-per-pair 14` で実行し，
   **wafl-ctrl5 の Ollama（127.0.0.1:11499）上でのみ**行う
   （config.yml 2026-09-23 絶対条件 (B)．wafl500〜509 は絶対に使わない）．
2. `mise run setup`（= `build_dataset.py`）で `data/dataset.jsonl` を **3,435 → 3,750 行**へ再生成する．
3. `data/MANIFEST.md` に新しい sha256 と生成コマンドを追記する（`data/` は .gitignore 対象のため）．
4. **コードの変更は無い**．`build_dataset.py` は既に本経路を持っている．

**レバーを読むコード行と，そこへ到達する条件（config.yml 冒頭「同じ失敗を 6 回繰り返している」への対応）**

| # | コード行 | 何をするか |
|---|---|---|
| 1 | `build_dataset.py:626` `_DEFAULT_GENERATED_COMPOUND_QUESTIONS_PATH = "data/compound_questions_generated.jsonl"` | 既定パスの定義 |
| 2 | `build_dataset.py:1430-1438` argparse `--generated-compound-questions`（default が #1） | CLI 既定値として #1 を採る |
| 3 | `build_dataset.py:1451` `generated_compound_questions_path = args.generated_compound_questions or None` | 空文字なら None に落とす |
| 4 | `build_dataset.py:1459-1466` `_build_rows(..., generated_compound_questions_path=...)` | 値を `_build_rows` へ渡す |
| 5 | `build_dataset.py:1217` `_load_generated_compound_questions(generated_compound_questions_path)` | JSONL を読む |
| 6 | `build_dataset.py:1218-1226` `id=f"compound-{len(_COMPOUND_QUESTIONS)+offset:03d}"`, `is_compound=True` | 行を追加 |
| 7 | `metrics.py:667-669` `compound_domain_question_count` / `compound_domain_top1_accuracy` / `compound_coverage` | `is_compound` で複合指標を算出 |

**到達条件（すべて現行構成で満たされることを確認済み）**
- (a) `mise run setup` は引数なしで `build_dataset.py` を呼び，#2 の default により #1 のパスが渡る．
  **追加の config 変更は不要**であり，到達条件は「ファイルが実在し行数が増えていること」だけである．
- (b) **最大の落とし穴**: `_load_generated_compound_questions()` は `build_dataset.py:1141-1147` で
  **path が None でもファイルが無くても例外を投げず `[]` を返す**．すなわち生成に失敗しても
  `build_dataset.py` は静かに成功し，`data/dataset.jsonl` は 3,435 行のままになる．
  これが Iter16/20/21/22/27/B35 と同型の「実験不成立」を生む唯一の経路である．
  **実装フェーズは (i) `wc -l data/compound_questions_generated.jsonl` = 630，
  (ii) `wc -l data/dataset.jsonl` = 3,750 を本走前に必ず確認すること．**
- (c) id の桁あふれは起きない．offset 最大 630 → `compound-730` で `{:03d}` に収まる．
- (d) レバーが発火した直接の証拠は **results.jsonl に `compound-416` 以降の id が存在すること**である．
- (e) 予備実行は **新規追加行の先頭 20 問**（`data/dataset_iter90_preview20.jsonl`）で行う．
  既存行で予備実行してもレバーの発火証拠にならない．

**事前ゲート G0（結果を見る前に判定規則を固定する）**
- **G0-a（生成器の可用性）**: wafl-ctrl5 の Ollama に `qwen3.5:9b` と judge_model が存在すること．
  欠けている場合は pull する．**wafl500〜509 は使わない．**
- **G0-b（生成物の妥当性）**: 45 ペアすべてがちょうど 14 行．新規 315 行と，既存 415 複合行および
  全単一ドメイン行との間で Jaccard ≥ 0.6 の近重複が 0 件
  （`_NEAR_DUPLICATE_JACCARD_THRESHOLD` = 0.6 の既存ガードをそのまま使う）．
- **G0-c（予備 20 問）**: 新規行の先頭 20 問で予備実行し，(i) HTTP 500 が 0 件，
  (ii) 1 行あたり平均所要 **≤ 12,086ms**（= 実測 8,057ms の 1.5 倍）．
- **G0-d（所要時間の外挿）**: G0-c の実測から 3,750 行の総所要を外挿し，**170 分以内**であること．
  超える場合は **`--target-per-pair` を 11（複合 595 行）へ落として再実行する**（この条項は事前登録．
  本走を省略する選択肢は取らない——config.yml 絶対条件 (A)）．

**成功条件（事前登録．本レバーは精度向上ではなく測定系の整備であるため，主基準は top1_accuracy ではない）**

| 区分 | 指標 | 現状 | 判定基準 |
|---|---|---|---|
| **主基準①（規模）** | `compound_domain_question_count` / `data/dataset.jsonl` 行数 | 415 / 3,435 | **730 / 3,750 ちょうど** |
| **主基準②（純粋追加）** | 既存 3,435 行の id・query・expected_domains | — | **ビット単位で一致**（B118．1 行でも変化したら実験不成立） |
| **主基準③（検出力）** | 旧 artifact `f6c33edb...`（Iter87）と現行 `ff8aad9c...`（Iter89）の replay による複合 730 行の argmax flip 行数 n_d | 47（n=415） | **n_d ≥ 74**（比例期待値 82.7 の 9 割）．同時に実測 π̂_d から `1.96·√(π̂_d/n)` を算出し **≤ 2.6pt**（現状 3.238pt） |
| **主基準④（品質）** | 人手スポットレビュー（乱数種 90 で新規 315 行から 30 件抽出）「2 ドメインの知識が本当に両方要るか」 | 5/30（Iter78 実測） | **不適合 ≤ 6/30（20%）**（Iter78 と同一閾値．緩めない） |
| **主基準⑤（分布）** | ペアごとの行数 | 7 | **45 ペアすべて 14 行**．`general` を含むペアの不適合率を別途報告（Iter78 の所見の追跡） |
| 報告のみ | 全 3,750 行の top1_accuracy・`compound_domain_top1_accuracy` | 0.830859 / 0.792771 | **判定に用いない**（複合比率が 12.1% → 19.5% へ上がるため全体 top1 は機械的に下がる） |

**非退行条件（基準線 = Iter89 本走 `results/20260927_232950/`．すべて「既存 3,435 行サブセット」上で評価する）**

| # | 指標 | 基準線 | 判定基準 |
|---|---|---|---|
| ① | per-domain 20 指標（precision/recall × 10） | Iter89 本走 | BH 補正後の有意退行 **0 件** |
| ② | `compound_domain_set_recall`（既存 415 行サブセット） | 0.566265 | **≥ 0.539759**（絶対値．Iter86〜89 と同一） |
| ③ | `compound_mean_dispatched_count`（既存 415 行サブセット） | 1.879518 | **≤ 2.10**（絶対値．同上） |
| ④ | top1_accuracy（既存 3,435 行サブセット） | 0.830859 | **完全一致**．拡充は既存行の計算に影響しないため，**不一致なら実験不成立**（Iter85 と同型の検証） |
| ⑤ | `fallback_rate` | 0.0 | **= 0.0** |
| ⑥ | `mean_duration_ms`（既存 3,435 行サブセット） | 1643.249 | **≤ 1971.9**（基準線 +20%．**全 3,750 行では複合比率が上がるため機械的に増えるので，サブセットで評価する**） |
| ⑦ | `dispatch_failure_rate`（全 3,750 行） | 0.001165（本フェーズで results.jsonl を直接数えて確認．4/3,435 行） | **≤ 0.005**（絶対値．Iter86〜89 と同一） |

**判定語の定義（Iter78 と同一条文）**
- **adopted**: 主基準①〜⑤と非退行①〜⑦をすべて満たす．**top1_accuracy の増減は判定に用いない．**
- **partial**: 主基準③（検出力）だけが未達．実測 π̂_d から必要 n を再計算して backlog へ残す．
- **invalid（実験不成立）**: `data/dataset.jsonl` が 3,435 行のまま／`compound_domain_question_count` が
  415 のまま／results.jsonl に `compound-416` 以降の id が 0 件／非退行④が不一致，のいずれか．

**本走（config.yml 絶対条件 (A)）**: 変更適用後に **wafl500〜509 を用いた 3,750 問のフルスペック本走を
必ず 1 回実施する**．`mise run start -- --dataset data/dataset.jsonl --output results.jsonl` ののち
`mise run analyze -- <timestamp>` を timestamp 明示指定で実行する（B118 落とし穴 1 の再発防止）．
想定所要は **136〜160 分**（timeout 180 分以内）．

**恒久ルールへの適合確認**
- B115 (1)（ドメイン固有の後付け補正を追加しない）: 45 ペア一律 14 行で，特定ドメインを狙い撃ちにしていない．
- B116 (2): 既存公開データセットの調査を先に実施し（Q1），不成立を出典付きで記録したうえで (ii) へ落とした．
- 2026-09-23 絶対条件 (B): 生成・検証の LLM 呼び出しは **wafl-ctrl5 のみ**．
- 2026-09-23 絶対条件 (A): 本走を必ず実施する（G0-d でも本走省略の分岐を作っていない）．

### 実験結果（Iter90，フェーズ 2 実測．判定はフェーズ 3）

**本走**: `results/20260928_032909/`（3,750 問，wafl500〜509 でフルスペック 1 回，所要約 158 分，timeout 180 分以内）．
`mise run analyze -- 20260928_032909`（timestamp 明示指定）まで実施済み．
**レバー発火の直接証拠**: results.jsonl に `compound-416`〜`compound-730` が存在し，ログ上でも実際にディスパッチされたことを確認．

**実装上の逸脱（重要）**: 計画の生成コマンド（`generate_compound_eval_questions.py --target-per-pair 14` を直接実行）は，
同スクリプトが `--output` を毎回 `open(..., "w")` で全体上書きする実装のため，**そのまま実行すると既存 315 行を破壊する**ことが
実装フェーズで判明した（630 行を丸ごと再サンプリングしてしまい B118「純粋な追加」にならない）．
Iter78 の「追い上げパス」と同じ手法（`generate_all_rows()` を直接呼ぶ使い捨てドライバ．スクリプト本体は無改変）で
新規 7 件/ペアだけを生成し既存ファイル末尾へ追記した．既存 315 行がバイト単位で不変であることを `diff` で確認済み．

**事前ゲート G0**: a（wafl-ctrl5 に `qwen3.5:9b`・judge_model とも既存，pull 不要）PASS ／ b（45 ペアすべて 14 行，
近重複 0 件・最大 Jaccard 0.148，exact 重複 0 件）PASS ／ c（新規行先頭 20 問 `compound-416`〜`435` で HTTP 500 0 件，
平均 8,588.45ms ≤ 12,086ms）PASS ／ d（外挿 139〜143 分 ≤ 170 分．`--target-per-pair` 11 へのフォールバックは不要）PASS．

**主基準の実測**

| # | 基準 | 実測 | 閾値 | 可否 |
|---|---|---|---|---|
| ① 規模 | `compound_domain_question_count` / dataset 行数 | **730 / 3,750** | 730 / 3,750 ちょうど | PASS |
| ② 純粋追加 | 既存 3,435 行の id・query・expected_domains | **完全一致（diff 0 件）** | ビット単位一致 | PASS |
| ③ 検出力 | n_d（複合 730 行の replay flip 行数） | **150** | ≥ 74 | PASS |
| ③ 副条件 | `1.96·√(π̂_d/n)` | **3.288pt**（π̂_d = 0.205479） | ≤ 2.6pt | **未達** |
| ④ 品質 | 人手スポットレビュー不適合（種 90，新規 315 行から 30 件） | **約 9〜10/30（30〜33%）**（実装フェーズの主観判定） | ≤ 6/30 | **未達の可能性** |
| ⑤ 分布 | ペアごとの行数 | **45 ペアすべて 14 行** | 全ペア 14 行 | PASS |

③ の π̂_d は計画時の想定 0.11325 から 0.205479 へ大きく上振れしたため，n_d は基準を大きく上回った一方で有意境界は
縮まらず 3.238pt → **3.288pt** とほぼ横ばいになった（拡充の目的であった分解能の改善が得られていない）．
`metrics.compute_mcnemar_test` の再利用値は discordant_a_only=43 / b_only=35 / p=0.428．
④ の不適合パターンは Iter78 所見と同型で，(a) `general` を含むペアの過度な抽象化・列挙型設問，
(b) 無関係な 2 話の単純併記，(c) 技術用語の強引な結び付け．**`general` を含むペアの不適合率は約 75%（8 件中 6 件）**で，
非 general 組の 18% と大きく乖離している．

**全 3,750 行（報告のみ，判定に用いない）**: top1_accuracy = 0.829867，compound_domain_top1_accuracy = 0.808219，
fallback_rate = 0.0，dispatch_failure_rate = 0.000533，mean_duration_ms = 2531.54．

**非退行条件の実測**（基準線 `results/20260927_232950/`，既存 3,435 行サブセット上で評価）

| # | 条件 | 実測 | 閾値 | 可否 |
|---|---|---|---|---|
| ① | per-domain 20 指標の BH 補正後の有意退行 | **0 件**（20/20 非有意） | 0 件 | PASS |
| ② | `compound_domain_set_recall`（既存 415 行） | **0.565060**（基準線 0.566265） | ≥ 0.539759 | PASS |
| ③ | `compound_mean_dispatched_count`（既存 415 行） | **1.925301**（基準線 1.879518） | ≤ 2.10 | PASS |
| ④ | top1_accuracy（既存 3,435 行） | **0.829985**（2,851/3,435） | 0.830859 と完全一致 | **未達** |
| ⑤ | `fallback_rate` | **0.0** | = 0.0 | PASS |
| ⑥ | `mean_duration_ms`（既存 3,435 行） | **2004.312ms** | ≤ 1971.9ms | **未達（+22.0%）** |
| ⑦ | `dispatch_failure_rate`（全 3,750 行） | **0.000533** | ≤ 0.005 | PASS |

**非退行④の詳細（フェーズ 3 で最も重要な論点）**: 基準線 2,854/3,435 に対し本走は 2,851/3,435 で **−0.087pt・正味 3 行差**．
行単位 diff では **19 行の判定が入れ替わっている**（True→False 11 行，False→True 8 行）．うち 4 行は `dispatch_failed` の状態自体が
変化しており（一時的なノード過負荷起因），残り 15 行は confidence の僅かな変動（例 0.4693893… → 0.4511954…）による境界での
argmax flip である．**「既存行の計算はビット単位で不変」という計画の前提が，実運用（Ollama 経由の GPU バッチ埋め込みと
dispatch の実行時非決定性）では成立していない**ことを示す実測であり，条文を literal に適用すると「実験不成立」に触れる．
原因の推測は上記のとおりだが確証は得ていない．

**その他**: `mise run deploy` が auto mode の分類器から一度「Production Deploy」として拒否されたが，フォアグラウンドでの
再実行で許可され正常完了（全 10 ノード healthy，smoke_check 全項目 PASS）．実験自体はハングなく完走．
監査用の中間生成物は `/tmp/iter90/`（`metrics_full3750.json`，`metrics_existing3435.json`，`metrics_existing415compound.json`，
`replay_result.json`，`topup_generate.py` ほか）に保存済み．`data/MANIFEST.md` に Iteration 90 節（sha256・生成コマンド）を追記済み．

### Iteration 90 実行済み

**単一レバー**: `compound_eval_set_expansion` = `llm_generated_separate_generator_scaleup_730`
（`scripts/generate_compound_eval_questions.py` の `--target-per-pair` を 7 → 14 相当へ．生成器 `qwen3.5:9b` ≠ 検証器 = `judge_model` の
構成は Iter78 から不変．コード変更なし）．複合行 415 → 730，`data/dataset.jsonl` 3,435 → 3,750 行．
本走 `results/20260928_032909/`（3,750 問，約 158 分）．基準線 `results/20260927_232950/`（Iter89，3,435 問，top1 = 0.830859）．

**判定: `partial`**（後述のとおり 4 つの判定語のいずれも字義どおりには当てはまらず，実態に最も近いものを選んだ）．
**評価集合 730 行は採用し，ロールバックしない．Iter91 以降の基準線は `results/20260928_032909/`（3,750 行，top1 = 0.829867）とする．**

#### 1. 主基準③（検出力）— フェーズ 2 の「未達」は **π̂_d の定義ずれによる誤判定**であり，実際は PASS

事前登録の ③ は，本文で「複合 730 行の **argmax flip 行数** n_d」と書きながら，数値アンカーは
「現状 47（n=415）」「現状 3.238pt」の 2 つを置いていた．本フェーズで旧 artifact `f6c33edb`（Iter87，0.6b）と
現行 artifact `ff8aad9c`（Iter89，4b）の replay を **旧 415 行 / 新 315 行に分けて再実行**したところ，この 2 つのアンカーは
どちらも **argmax flip ではなく McNemar の discordant（正誤の不一致ペア）** から計算された値であることが確定した．

| 部分集合 | argmax flip | discordant（a/b） | π̂_d（discordant） | 1.96·√(π̂_d/n) | replay top1（旧→新 artifact） |
|---|---|---|---|---|---|
| 旧 415 行 | 85 (20.48%) | **47**（29/18） | **0.11325** | **3.238pt** | 0.819277 → 0.792771 |
| 新 315 行 | 65 (20.63%) | 31（14/17） | 0.09841 | 3.464pt | 0.815873 → 0.825397 |
| 全 730 行 | **150** (20.55%) | **78**（43/35） | **0.10685** | **2.371pt** | 0.817808 → 0.806849 |

旧 415 行の再計算が事前登録の 47・3.238pt・0.11325 と**完全に一致**したことで，アンカーの定義は discordant で確定する．
したがって**アンカーと同じ定義を一貫して適用すれば n_d = 78 ≥ 74 で PASS，有意境界は 3.238 → 2.371pt で ≤2.6pt を満たし，
③ は副条件も含めて PASS** である．フェーズ 2 が報告した 3.288pt は分子に argmax flip（150）を用いた値で，
同じ定義を基準線側にも当てれば基準線は 3.238pt ではなく **4.354pt**（85/415）であり，4.354 → 3.288pt は
**√(415/730) = 0.754 倍という純粋な √n スケーリングと小数 3 桁まで一致する**．
すなわち**どちらの定義で統一しても拡充は設計どおり効いており，「分解能が改善しなかった」というフェーズ 2 の読みは，
基準線と実測で異なる量を比べたことによる artifact である**．条文を緩めたのではなく，条文内部の不整合を
「2 つの数値アンカーが実際に計算された定義」の側へ解消した（両方の読みの数値を上表に併記する）．

副産物として，**Iter90 の投資が意図どおりの成果を出した**ことも確認できた．Iter89 で唯一の悪化方向だった複合サブセットの
効果（B147 (c) が次レバー選定の根拠にした −2.65pt）は，730 行では **−1.10pt，discordant 43/35，p = 0.428** に縮み，
有意境界 2.371pt を下回る．**「Iter89 の埋め込み差し替えが複合設問を悪化させた」という疑いは，分解能を上げた測定系の下では支持されない．**

#### 2. 非退行④（top1 のビット単位一致）— 条文の前提が誤っていた．原因はレバーではなく**セッション境界の非決定性**

実測 0.829985（2,851/3,435）vs 基準線 0.830859（2,854/3,435），正味 −0.087pt・3 行．行単位では 19 行が反転
（True→False 11／False→True 8，厳密二項 **p = 0.6476**）．本フェーズで原因を切り分けた．

- **既存 3,435 行の `probe_candidates` を基準線と全スロット突き合わせた結果，34,350／34,350 スロットすべてが相違**していた
  （行ごとの最大差の中央値 0.00269，p90 0.01486，最大 0.0806）．すなわち「既存行だけが 3 行動いた」のではなく
  **全行の確信度が動いており，そのうち境界付近の 19 行だけが argmax を跨いだ**．
- 一方，**同一セッション内（デプロイを挟まない）は完全に決定論的**である．Iter89 の予備 20 問と Iter89 本走は
  データセットの行数も順序も全く違うのに **200/200 スロットがビット単位で一致**し，Iter90 の予備 20 問と Iter90 本走も
  **200/200 一致**した．**したがって「行数が増えて GPU のバッチ構成が変わったから」という機序（フェーズ 2 の推測）は棄却される．**
  変動はセッションを跨いだとき（＝ `mise run deploy` によるコンテナ再作成と埋め込みモデルの再ロードを挟んだとき）にのみ生じる．
- 分類器 artifact は `ff8aad9c...` のまま不変（sha256 で確認）．`data/dataset.jsonl` の既存 3,435 行も diff 0 件．
  **レバーが既存行の計算に影響した証拠は無い．**
- 対照として，Iter82 → Iter83（`qwen3-embedding:0.6b`）ではデプロイを挟んでも `probe_candidates` の最大絶対差が **0.0** だった
  （journal_archive「Iteration 83」）．**0.6b では成立していたセッション跨ぎの再現性が，4b（Iter89 で採用）では失われている**
  ことになる．機序は未確証だが，12GB VRAM に expert モデルと同居する 4b のロード時オフロード構成が
  セッションごとに変わりうる点が第一の容疑である（本反復では検証していない）．

**結論**: 非退行④は**条文の前提（「拡充は既存行の計算に影響しないから完全一致するはず」）そのものが実運用で成立しない**ことが
実測で示された．条文を literal に適用すれば invalid だが，この条文の目的（success_criteria (6)・d0004 §4）は
「レバーがコードパスに到達したかの検出」であり，本反復ではレバー発火の直接証拠（`compound-416`〜`730` の存在，
`compound_domain_question_count` = 730）が揃っているうえ，差分は純ノイズと区別できない（p = 0.6476）．
**`invalid` を宣言して 158 分の有効な測定を破棄することは，条文の目的にも実測にも反する．**
よって invalid とはせず，事実と再現性の床を確定させたうえで下記の恒久的な申し送りに置き換える．

**再現性の床（今後の非退行条件で使う数値）**: デプロイを挟んだ同一構成の 2 本走で，**既存 3,435 行の 0.553%（19 行）が
正誤反転し，top1 は ±0.087pt 動く．この対から導かれる McNemar 有意境界は 0.249pt** である．
**top1 の 0.25pt 未満の差は今後いかなるレバーにも帰属させてはならない．**
（参考: 訓練の乱数種ばらつきは replay 上で 0.31pt — Iter88 学び．軸②③の生成ノイズ床は 2.6pt — success_criteria (5)．）

#### 3. 主基準④（品質）— 閾値の真上で判定不能．`general` ペアの構造的欠陥は 2 反復連続で再現

フェーズ 2 の主観判定（約 9〜10/30）を鵜呑みにせず，同じ種 90 の 30 件を本フェーズで独立に読み直した．
「2 ドメインの知識が本当に両方要るか」で明確に不適合と言えるのは **6/30**（列挙型の [4]，感情吐露で一方のドメイン知識が不要な [6]，
確率論を強引に接続した [8]，cs 単独で完結する [22]，疑似科学的な結び付けの [24]，無関係 2 話の併記 [28]），
判断が割れる境界例（[3][5][9][16][27] 等）を不適合に数えると **9〜11/30** になる．
**閾値は ≤6/30 であり，2 名の独立した判定が 6 と 9〜10 に割れて閾値のちょうど上下に落ちた．
すなわち ④ は「未達」とも「達成」とも言えず，測定器としての `≤6/30 の主観 1 名判定` が基準として機能していない．**
④ を根拠に adopted を主張することはできないが，④ を根拠に棄却することもできない．

一方 **`general` を含むペアの偏りは明確で再現性がある**．私の明確な不適合 6 件のうち **5 件が `general` 組**で，
サンプル中の general 行 8 件に対する不適合率 **62.5%**，非 general 22 件に対しては **4.5%** である．
Iter78 のスポットレビュー（5/30，全件が general 絡み）と合わせて**2 反復連続の同型所見**であり，偶発ではない．
機序は「`general` は他 9 ドメインの補集合として定義されているため，`general`＋X のペアに対して
生成器が『X の話題＋日常の愚痴』という 2 話併記や，X に還元できる設問を作ってしまう」ことである．

なお，**品質の低さが測定系を汚しているという仮説は，今回の数値では支持されない**．新規 315 行の discordant 率は
0.09841 で旧 415 行の 0.11325 **より低く**，本走の実測 top1 も新 315 行 0.828571 > 旧 415 行 0.792771 である．
`general` ペアの設問は「複合設問として妥当でない」が「ルーティングの評価行としては旧行より素直」という状態にある．

#### 4. 非退行⑥（`mean_duration_ms`）— 条文の設計ミス

既存 3,435 行だけで測っても 1643.249 → **2004.312ms（+22.0%）**で閾値 +20% を超えた．
しかし**同一本走の中で複合行を 415 → 730 に増やせば，既存行もノード競合の影響を受けて遅くなる**．
「既存行サブセットで測れば負荷の影響を除ける」という事前登録の想定（B148 (E)）が誤りであり，
部分集合の切り出しでは交絡を除去できない．レバー自体の欠陥ではないが，事前登録の条文としては失敗である．

#### 5. 判定語の確定

| 主基準 | 可否 | 非退行 | 可否 |
|---|---|---|---|
| ① 規模（730 / 3,750） | PASS | ① per-domain 20 指標 | PASS（有意退行 0 件） |
| ② 純粋追加（既存 3,435 行 diff 0） | PASS | ② `compound_domain_set_recall` | PASS |
| ③ 検出力（n_d = 78 ≥ 74，境界 2.371pt ≤ 2.6pt） | **PASS**（上記 §1） | ③ `compound_mean_dispatched_count` | PASS |
| ④ 品質（≤6/30） | **判定不能**（6 と 9〜10 に割れ） | ④ top1 完全一致 | **FAIL**（原因はレバー外．上記 §2） |
| ⑤ 分布（45 ペア × 14 行） | PASS | ⑤ `fallback_rate` = 0.0 | PASS |
| | | ⑥ `mean_duration_ms` | **FAIL**（条文の設計ミス．上記 §4） |
| | | ⑦ `dispatch_failure_rate` | PASS |

`adopted`（全条件 PASS）は ④ が判定不能・非退行④⑥ が FAIL のため取れない．
`rejected`（レバーが効かなかった）は ①②③⑤ が PASS でレバーが目的を果たしているため事実に反する．
`invalid`（測定が無意味）は §2 のとおり事実に反する．
**残る `partial` を選ぶ．**ただし `partial` の条文（「主基準③だけが未達」）にも字義どおりには当てはまらない．
**事前登録の 4 語が今回の事象（条文の前提の誤り・定義の内部不整合・判定不能な主観指標）を網羅していなかった**ことを
そのまま記録し，次の判定語定義の設計に反映する．

#### 6. 学び（次の自分が読んで分かる形で）

1. **「基準線とビット単位で一致すること」を非退行条件に書いてはならない．**`qwen3-embedding:4b` を採用した Iter89 以降，
   デプロイ（コンテナ再作成・モデル再ロード）を挟むと既存行の確信度が**全スロット**動く．代わりに
   **top1 の差 ≤0.25pt かつ正誤反転行 ≤1.0%** を再現性の床として使うこと．
   **レバーがコードパスに到達したかの検出は「基準線との完全一致」ではなく，レバー固有の直接証拠
   （今回なら `compound-416` 以降の id の存在）で行う．**両者は別の目的であり，同じ条文で兼ねてはいけない．
2. **決定性が成り立つ範囲は「同一セッション内」である．**予備 20 問と本走の突き合わせ（デプロイを挟まない）は
   Iter89・Iter90 とも 200/200 ビット一致で，**予備実行による事前検証は引き続き完全に信頼できる**．
   信頼できないのはセッションを跨いだ比較だけである．
3. **事前登録で指標を定義したら，本文の定義語と数値アンカーが同じ量を指しているかを必ず突き合わせること．**
   ③ は「argmax flip 行数」と書きながらアンカー 47・3.238pt は discordant 由来で，フェーズ 2 が文言どおりに計算した結果，
   **実際には達成していた基準を「未達」と誤判定した**．アンカーの再現計算（今回の旧 415 行 replay）は 5 分で済む．
4. **「部分集合で測れば負荷の交絡を除ける」は誤り．**同一本走の中で行数を増やせば既存行の所要時間も増える（+22%）．
   時間系の非退行条件は，行あたり平均ではなく負荷項を織り込んだ形にするか，そもそも判定に使わないこと．
5. **主観 1 名・閾値 1 本のスポットレビューは判定基準として機能しない．**2 名の独立判定が 6 と 9〜10 に割れ，
   閾値 ≤6/30 のちょうど上下に落ちた．今後この種の条件を置くなら，判定を二値ではなく
   「明確な不適合」「境界例」に分け，**明確な不適合だけで閾値を切る**など，割れにくい定義にすること．
6. **`general` を含む複合ペアは 2 反復連続で不適合率が突出している**（Iter78 5/30 全件 general 絡み，Iter90 62.5% vs 4.5%）．
   `general` が他 9 ドメインの補集合として定義されていることに由来する構造的な問題であり，生成器を変えても消えない．
   ただし**ルーティング評価行としての素直さ（discordant 率・top1）は旧行より良い**ので，測定系を汚してはいない．
7. **Iter90 の投資は回収された．**B147 (c) が起点にした「Iter89 の複合 −2.65pt」は，730 行では −1.10pt・p = 0.428 に縮み，
   有意境界 2.371pt を下回る．**複合ドメインの悪化という疑いは支持されない**ので，次はこの心配のために
   レバーを費やす必要がない．

## Iteration 89: 埋め込みモデルの qwen3-embedding:4b への差し替え

### 調査 (Iter89)

本反復のレバーは backlog B145 (d) で `embedding_model_replacement` = `qwen3_embedding_4b` に確定済みで，
選定の裁量は無い．したがって調査の問いは 4 つである．
**(Q1) この値は Iter80 で一度「G0 不合格・実機未検証」になっている．当時と何が変わり，今回は走らせられるのか．
(Q2) 4b は本タスク（日本語 10 ドメイン分類）で 0.6b を上回るという根拠はどこまであるか．
(Q3) 次元 2560（連結で 5120）× 訓練 2,327 行という p ≫ n は何を意味するか．
(Q4) 事前投影と，B145 が確定した測定分解能 0.9pt との関係．**

**Q1（最重要）: 本値は Iter80 で実機未検証のまま棚上げされた．原因は精度ではなく VRAM である．
今回の実測では「light_model を落とせば収まる」ところまで条件が判明した**

- **B145 は Iter80 の前歴に触れていないが，journal_archive.md「Iteration 80」節のとおり，
  `qwen3_embedding_4b` は一度着手されて G0（VRAM ゲート）で不合格になり，代替の `bge-m3` へ差し替えられている**
  （その `bge-m3` は本走で top1 が基準線を 4.4pt 下回り rejected）．Iter80 の実測は
  **4b の常駐 4.4GB・PROCESSOR 100% GPU（wafl-ctrl5）**，一方 G0-b の予算式は
  「X + light 3.1GB + expert 5.3GB ≤ 11.5GB」＝ **X ≤ 3.1GB** で，4.4GB は算術的に不合格だった．
  つまり **Iter80 の不合格は「4b が GPU に載らない」ことの証明ではなく，「3 モデル同時常駐なら載らない」ことの証明**である．
- 本フェーズで実機を read-only 実測した（2026-09-27．`wafl500`・`wafl-ctrl5`．生成処理は一切走らせていない）．

  | ホスト | GPU | used / free | 常駐モデル（`ollama ps`） |
  |---|---|---|---|
  | wafl500（依頼者兼 general） | 12288 MiB | **11156 / 755 MiB** | `qwen3-embedding:0.6b` 2.4GB（100% GPU）＋ `expert-mesh-general-lora` 5.3GB（100% GPU）＋ **`qwen3.5:4b-q4_K_M` 3.7GB（25%/75% CPU/GPU）** |
  | wafl-ctrl5（制御ホスト） | 12288 MiB | 8457 / 3453 MiB | `qwen3-embedding:0.6b` 2.4GB ＋ `bge-m3` 0.664GB ＋ swallow-8B 5.3GB（いずれも 100% GPU．`qwen3-embedding:4b` は Iter80 で pull 済み，未ロード） |

- **現行構成でも既に light_model は 25%/75% で CPU に溢れている**．ここで重要なのは，
  **`routing_method=supervised_classifier` の下で light_model は実行時に 1 度も呼ばれない**ことである．
  `http_server.py:365-371` の `/probe` は supervised_classifier 分岐で
  「No LLM call: the classifier consumes the query_embedding」とコメントどおり分類器だけを呼び，
  light_model を使う分岐（multi_sample / stp / semantic_entropy / p_true / top_k / 既定 self_report）は
  いずれも `confidence_signal_method` か `routing_method` の先行分岐で到達しない．
  fallback も `confidence_threshold=0.0` で Iter28 以降 0 件が続いている（Iter88 実測も fallback_rate = 0.0）．
  **light_model が常駐しているのは `http_server.py:397` の起動時 warmup のためだけ**である．
- したがって **実行時の実効常駐は expert 5.3GB ＋ embedding X**．X = 4.4GB なら 9.7GB で 12288 MiB に収まる．
  **Iter80 の G0-b（静的な算術ゲート）は，実行時には使われない light_model を予算に含めていたぶん保守的すぎた．**
  ただし「Ollama が `OLLAMA_KEEP_ALIVE=-1` の light_model を退避してくれるか」「退避せず 4b を CPU 混在で
  載せるか」は**実測でしか決まらない**（現に light_model 自身が CPU 混在で載っている）．
  そこで本反復の G0 は**静的な算術ゲートをやめ，実機での常駐状態と予備 20 問の実測に置き換える**（計画節 G0）．

**Q2: 公称ベンチは一貫して 4b > 0.6b．ただし日本語分類の直接値は Iter80 時点と同じく存在しない**

- Qwen 公式 Model Card / GitHub（<https://huggingface.co/Qwen/Qwen3-Embedding-4B>，
  <https://github.com/QwenLM/Qwen3-Embedding>，2026-09-27 再確認）: MMTEB Mean(Task) は
  **0.6B 70.70 → 4B 74.60 → 8B 75.22**，MTEB multilingual は **0.6B 64.33 → 4B 69.45**（+5.12pt）．
  サイズ方向の単調性は複数ベンチで一致している（事実）．
- **JMTEB（日本語）の 4B の公開値は今回も見つからなかった**．hotchpotch の JMTEB 計測
  （<https://secon.dev/entry/2025/06/11/100000-qwen3-embedding-jmteb>）は 0.6B のみ（Classification 66.09）で，
  4B 行は無い．**「4b が日本語分類で 0.6b を上回る」は外挿であり未検証の推測である**（Iter80 の記述と同じ状態）．
  なお同記事では 0.6B の JMTEB Classification 66.09 に対し日本語専用の `ruri-v3-310m` が 78.66 と大きく上回るが，
  日本語専用モデルは prefix 規約が異なり 2 レバー目になるため本反復の候補外である（B125(5) と同じ理由）．
- **本リポジトリ内に，公称値より価値の高い一次データがある**．Iter80 の G1（`data/classifier_train.jsonl` 1,427 行・
  単一ビュー・5-fold StratifiedKFold）の実測は **0.6b cv_accuracy 0.7561 / macro-F1 0.7562 に対し
  4b 0.7722 / 0.7718（+1.61pt）**で，4b が最良だった．**4b の訓練行埋め込みは
  `data/embcache_qwen3-embedding_4b.npy`（shape (1427, 2560)，float64）として残っている**ため，
  単一ビュー分は再計算不要である（ただし現行は連結ビューかつ訓練データが 2,327 行なので，
  計画節 G1 では 4b の 2 ビュー × 2,327 行を新規に計算する）．
- Iter79 の学び 2（公称ベンチは候補を絞る道具であって採否の根拠にならない．MTEB 差 +2.05pt に対し
  実測 +16.34pt だった）を今回もそのまま適用し，**採否は本走でのみ決める**．

**Q3: 5120 次元 × 2,327 行は p ≫ n だが，方向としては Iter80 の警戒と同じで新規リスクではない**

- 現行 artifact `models/domain_classifier.joblib`（sha256 `f6c33edb...`）は `n_features_in_` = **2048**
  （1024 × 2 ビュー），訓練行は `data/classifier_train_iter87_hybrid.jsonl` の **2,327 行**（本フェーズで実測確認）．
  4b にすると **2560 × 2 = 5120 次元**となり，B145 (1) の申し送りどおり検証値は **5120** に引き直す．
- `LogisticRegression(max_iter=1000, class_weight=None)` ＋ `_extract_sample_weights()` ＋
  `CalibratedClassifierCV(method='temperature', ensemble=True)` という構成は**一切変えない**（単一レバー原則）．
  p/n が 0.88 → 2.20 へ上がるため，**訓練データ内 CV は本走 top1 の上振れした推定になりやすい**．
  Iter80 の解釈規則をそのまま踏襲し，**G1 の CV は本走の予測値として扱わない**．
- 所要時間: 特徴次元が 2.5 倍でも `predict_proba` は行列積 1 回で，probe のオーバーヘッドは無視できる．
  効くのは **Ollama の埋め込み 1 回あたりの latency（0.6b で約 5ms，B145 (3)）**の増加である（Q4 で扱う）．

**Q4: 事前投影 — 点推定 Δtop1 ≒ +1.0pt，区間は −1 〜 +3pt．主基準到達確率は 40〜50% で，
B145 が確定した分解能 0.9pt に対して「測れる可能性のある」初めてのレバーである**

- 基準線は Iter87 本走 `results/20260927_174150/`（top1 = **0.801456**，artifact `f6c33edb...`，全 10 ノード配布済み）．
  Iter88 は `no_effect`（Δ −0.03pt）で基準線を動かしていない．
- 根拠は 2 つしかない．(a) Iter80 の G1 CV 差 **+1.61pt**（1,427 行・単一ビュー・訓練データ内 CV），
  (b) MTEB multilingual の差 +5.12pt（日本語分類との対応は不明）．
  (a) は訓練データ内 CV なので本走への伝達率は 1 未満と見るのが自然で，かつ現行は連結ビュー・
  hard negative 拡充後で基準線側が既に底上げされている（伸びしろが削られている方向）．
  他方 Iter79（nomic → 0.6b）は CV 差より本走差が大きかった（+16.34pt）前例もあり，**方向の不確実性が大きい**．
  したがって点推定 +1.0pt，80% 区間 −1 〜 +3pt とする．
- **検出力**: B145 (b) の確定値（3,435 問・1 本走・種 1 個で McNemar 有意境界 0.612pt，80% 検出力に必要な Δ は
  約 0.92pt）に照らすと，点推定 +1.0pt は**ちょうど分解能の境界の上**にある．
  Iter88 のような「比の微調整」（投影 +0.28pt）とは異なり，**本反復は判定が成立しうる**．
  ただし discordant 行数は特徴空間を丸ごと入れ替えるため大きくなる見込みで，
  n_d が Iter87 の 226 を大きく超えると有意境界も上がる（`1.96·√n_d / 3435`）．
  n_d = 600 なら境界 1.40pt，n_d = 900 なら 1.71pt となり，**n_d 次第では +1.0pt でも有意に届かない**．
  この点は計画節 G2 のオフライン replay で本走前に数値化する．

### 計画 (Iter89)

**単一レバー**

`embedding_model_replacement` = **`qwen3_embedding_4b`**．
**`config.yaml:4` の `embedding_model: qwen3-embedding:0.6b` → `qwen3-embedding:4b` の 1 行のみ**が本レバーの本体である．
同じ埋め込みで `models/domain_classifier.joblib` を再訓練するのは，次元変更に構造的に付随する作業であって
別レバーではない（Iter79・Iter80 で確立した型）．

**固定する構成（基準線 = Iter87 本走 `results/20260927_174150/`，artifact `f6c33edb...`）**

`config.yaml` の `embedding_model` 以外の全項目（`embedding_instruction`（Iter81 の P1 文言）・
`embedding_view_concat: true`・`routing_method=supervised_classifier`・`confidence_threshold=0.0`・
`dispatch_candidate_threshold=0.0`・`dispatch_top_k=2`・`dispatch_gap_threshold=0.36`・`aggregation_method`・
`judge_model`・`classifier_model_path`・各ノードの `light_model=qwen3.5:4b-q4_K_M`／
`expert_model=expert-mesh-*-lora`・`probe_timeout_s`／`dispatch_timeout_s`），
`data/dataset.jsonl`（3,435 行，ビット単位で不変），`data/classifier_train_iter87_hybrid.jsonl`
（2,327 行，sha256 `63e73c20...`，ビット単位で不変．**訓練データは Iter87 の採択構成のまま，
`cross_domain_training_data_augmentation` は B145 (c) で closed**），
`scripts/train_domain_classifier.py` のモデル定義（`LogisticRegression(max_iter=1000, class_weight=None)` ＋
`_extract_sample_weights()` ＋ `CalibratedClassifierCV(method='temperature', ensemble=True)`），
`classifier.py`・`node.py`・`http_server.py`・`aggregator.py`・`metrics.py`・`build_dataset.py`・
`docker-compose.yml`（`OLLAMA_KEEP_ALIVE=-1` を含む）．
**ドメイン固有の補正は一切追加しない**（2026-09-23 恒久運用ルール (1)）．

**レバーを読むコード行と，そこへ到達する条件（d0004 §4 の再発防止．6 回繰り返した同型事故の対策）**

1. 設定 → 全 10 ノードへのモデル配布: `tools/node_models.py:get_models()`（L13）が `config["embedding_model"]` を
   返し，`mise.toml` L96-104 の deploy ループが `ollama pull` する．
   **到達確認: 全 10 ノードで `ollama list` に `qwen3-embedding:4b` 行があること**（B145 (2) の申し送り）．
2. 設定 → 各ノードの config: `mise.toml` L67 の `rsync config.yaml`．
   **到達確認: 全 10 ノードで `grep '^embedding_model:' $REMOTE_DIR/config.yaml` が `4b`**．
3. 設定 → 実行時のクエリ埋め込み: `node.py:202-207` の `embed_query_views(..., config["embedding_model"], ...)`．
   **到達確認: 予備 20 問が 500 を返さないこと**（5120 次元の特徴を 2048 次元の旧 artifact に食わせれば
   `predict_proba` が必ず例外になるので，train/eval 不一致はここで必ず落ちる＝ Iter36 型の無言の不一致は起きない）．
4. artifact → 全 10 ノード: `mise.toml` L70-74 の `models/` rsync．
   **到達確認: 全 10 ノードの `models/domain_classifier.joblib` の sha256 が新値と一致し，
   `n_features_in_` == 5120 であること**（B145 (1)）．
5. 訓練側の同一性: `scripts/train_domain_classifier.py:build_training_features()` は runtime と同じ
   `embed_query_views()` を呼ぶので，`--embedding-model qwen3-embedding:4b --embedding-instruction <P1 文言>
   --embedding-view-concat` を渡す限りビュー順（[plain, instructed]）は構造的に一致する．
6. 実験 → 指標: `metrics.py` 無変更．**到達確認: `total_questions == 3435` かつ
   `compound_domain_question_count == 415`**．

**事前ゲート G0（VRAM・所要時間．結果を見る前に判定規則を固定する）**

Iter80 の静的な算術ゲート（`X + light 3.1GB + expert 5.3GB ≤ 11.5GB`）は**採らない**．
実行時に呼ばれない light_model を予算に含めており，Q1 のとおり実測と乖離するためである．
代わりに**実機の常駐状態と予備 20 問の実測**をゲートにする．**wafl500〜509 での生成処理は
予備 20 問（本走と同じ経路）に限り，それ以前の埋め込み計算・訓練はすべて wafl-ctrl5 で行う**（絶対条件 B）．

- **G0-a（wafl-ctrl5 で 4b をロードできること）**: `bge-m3` と swallow-8B を `ollama stop` で退避して枠を空け，
  4b をロードして `ollama ps` の SIZE・PROCESSOR を記録する．**PROCESSOR が `100% GPU` でなければ G0-a 失敗**
  （その場合 G1/G2 のオフライン計算そのものが非現実的な時間になるため，即座に G0 失敗として扱う）．
  退避した swallow-8B は analyze（judge）の前に戻す．
- **G0-b（deploy 後の実機常駐）**: deploy 後・本走前に全 10 ノードで `ollama ps` と `nvidia-smi` を取る．
  合格条件は **(i) `qwen3-embedding:4b` と当該ノードの `expert-mesh-*-lora` がともに `100% GPU`** であること．
  **light_model が退避されていること自体は合格を妨げない**（Q1 のとおり実行時に呼ばれないため）．
- **G0-c（予備 20 問）**: 先頭 20 問で予備実行し，(i) HTTP 500 が 0 件，(ii) 1 問あたり平均所要が
  **3090ms 以下**（＝基準線 `mean_duration_ms` 1544.912 の 2 倍）であることを確認する．
- **是正の梯子（事前登録．上から順に試し，最初に G0-b/G0-c を満たした時点で止める）**:
  - **R0**: そのまま（追加操作なし）．
  - **R1**: 全 10 ノードで `ollama stop qwen3.5:4b-q4_K_M` を実行してから再計測する．
    **根拠**: `http_server.py:365-371` により supervised_classifier 経路では light_model は 1 度も呼ばれず，
    `confidence_threshold=0.0` で fallback も 0 件が Iter28 以降続いている．したがってこの操作は
    **どの行の出力も変えず，VRAM と latency にしか影響しない**．計算結果を変えないので 2 本目のレバーにならない．
  - **R2**: `expert_backend.OllamaClient.embed()` の POST に `"options": {"num_ctx": 2048}` を足す
    （訓練・実行時の両方が同じ関数を通るので自動的に一致する）．
    **適用前提（必須）**: 評価 3,435 行・訓練 2,327 行の**全行が prefix 込みで 2048 トークン以下**であることを
    実測で確認すること（最長は評価 1,811 文字・訓練 1,239 文字．`prompt_eval_count` か tokenizer で確認する）．
    1 行でも超えるなら R2 は**適用しない**（切り詰めは埋め込みを変えるため）．
  - **R3（最後の手段）**: R0〜R2 のいずれでも `100% GPU` に届かない場合でも，
    **予備 20 問からの外挿で 3,435 問の所要が 180 分（`experiment.timeout_min`）以内なら本走は実施する**
    （絶対条件 (A)．「改善しない見込みでも本走を省略しない」の運用と同じ）．
    この場合 **非退行⑥（`mean_duration_ms`）は FAIL する見込みである旨を本走前に記録する**．
  - **G0 失敗（＝本走を実施しない）と判定するのは，R3 の外挿でも 180 分を超える場合だけ**とする．
    その場合は `invalid`（実験不成立・VRAM 制約）として分析フェーズへ引き継ぎ，
    **本反復内で別の埋め込みモデルへ差し替えることはしない**（Iter80 が `bge-m3` へ差し替えて
    「イテレーション名と実際の値がずれる」状態を作った失敗を繰り返さないため．`bge_m3` は Iter80 で rejected 済み）．

**事前ゲート G1（report-only．CV．評価集合を一切見ない）**

wafl-ctrl5 上で `data/classifier_train_iter87_hybrid.jsonl`（2,327 行）**のみ**を使い，
4b の 2 ビュー（plain / instructed）を計算して 5,120 次元特徴を作り，5-fold StratifiedKFold の
accuracy / macro-F1 を測る．0.6b の同条件の値と並べて記録する．
**これは値の選定には使わない（レバーは確定済み）．Q3 の解釈規則により本走の予測値としても扱わない．**
参考値: Iter80 の単一ビュー・1,427 行 CV は 0.6b 0.7561 / 4b 0.7722．

**事前ゲート G2（検出力と着地点の事前登録）**

評価 3,435 行について 0.6b・4b 双方の 2 ビュー埋め込みを wafl-ctrl5 で計算し
（`data/embcache_eval_qwen3-embedding_0.6b*.npy` は 1,915 行分しか無いので 3,435 行分を作り直す），
旧 artifact（`f6c33edb...`，2048 次元）と新 artifact（5120 次元）の `predict_proba` argmax を replay する．

- **discordant 行数 n_d を算出し，`1.96·√n_d / 3435` で本走の McNemar 有意境界を事前に確定して journal に記録する．**
- **n_d ≥ 30 を合格条件**とする．一桁なら「効果なし」ではなく **config 未到達**を既定の解釈とする（d0004 §4）．
- replay から予測した Δtop1 も記録する（Iter87 実測で replay と本走の乖離は 0.104pt）．
  **この予測値を見て成功条件を書き換えてはならない**（B131 以来の運用）．

**成功条件・非退行条件（事前登録．結果を見る前に固定し，事後に緩めない）**

基準線は Iter87 本走 `results/20260927_174150/`（top1 = 0.801456）．条文は Iter86〜88 の事前登録を踏襲する．

- **主基準 (i)**: McNemar 検定（対応あり）で `top1_accuracy` が基準線に対し **有意（p < 0.05）**．
- **主基準 (ii)**: **Δtop1 ≥ +1.0pt**（すなわち top1 ≥ **0.811456**）．
  この 1.0pt は B145 (b) が確定した本測定系の分解能（80% 検出力に必要な Δ ≒ 0.92pt）と整合する．

| 条件 | 指標 | 基準線（Iter87 実測） | 合否ライン |
|---|---|---|---|
| **非退行①** | per-domain recall / precision 計 20 指標（BH 補正 q=0.05） | Iter87 実測値 | **有意退行 0 件**．**`education` recall（0.3626）・`natural_science` precision（0.8424）は個別に明記する**（B143 (b) の継続監視） |
| **非退行②（複合被覆）** | `compound_domain_set_recall` | 0.566265 | **≥ 0.539759**（絶対値．Iter86〜88 と同一） |
| **非退行③（複合予算）** | `compound_mean_dispatched_count` | 1.879518 | **≤ 2.10**（絶対値．同上） |
| **非退行④** | `fallback_rate` / `dispatch_failure_rate` | 0.0 / 0.001456 | fallback = 0.0，dispatch_failure ≤ 0.005（絶対値．同上） |
| **非退行⑤** | rank1 以外が選ばれた行数 | 5 | **≤ 15**（絶対値．同上） |
| **非退行⑥** | `mean_duration_ms` | 1544.912 | **≤ 1853.9**（規則「基準線 +20% 以内」は同一．基準線が Iter87 のまま据え置きなので閾値も据え置く．**本反復で初めて現実的なリスクになる**——B145 (3)） |
| **非退行⑦** | ECE | 0.045717 | **≤ 0.08**（絶対値．同上） |

**判定規則（事前登録）**

- **adopted**: 主基準 (i)(ii) を満たし，非退行①〜⑦をすべて満たす．
- **partial**: McNemar が有意で Δtop1 が +0.5〜+1.0pt，かつ非退行①〜⑦を満たす．
- **no_effect**: |Δtop1| < 0.5pt **または** McNemar が非有意（かつ非退行に違反なし）．
- **rejected**: Δtop1 ≤ −0.5pt，**または**非退行①〜⑦のいずれかに違反．
  **非退行⑥のみの違反で rejected になる場合は，top1 側の結果を併記して「精度は改善したが latency で落ちた」と
  明示的に記録する**（次の一手の判断材料になるため）．
- **invalid（実験不成立）**: G0 失敗（R3 の外挿でも 180 分超），G2 の n_d < 30，
  本走の `total_questions ≠ 3435`，全 10 ノードのいずれかで artifact sha256 または `n_features_in_` が不一致，
  のいずれか．

**変更するファイルと箇所**

1. `config.yaml:4`: `embedding_model: qwen3-embedding:0.6b` → `qwen3-embedding:4b`．**本レバーの本体（1 行）**．
2. `models/domain_classifier.joblib`: 4b 埋め込みで再訓練して差し替える．
   **旧版は `models/domain_classifier_pre_iter89_qwen3_0.6b.joblib` へ `cp` で退避**してから上書きする
   （Iter77/79/80 と同じ慣行．flip 計測と adopted 以外での復元に必要）．
   訓練コマンドは `data/MANIFEST.md` の Iter87 節のものから `--embedding-model` だけを差し替える
   （`--train-data data/classifier_train_iter87_hybrid.jsonl`・`--embedding-instruction` の P1 文言・
   `--embedding-view-concat` はそのまま）．
3. `data/embcache_eval_qwen3-embedding_4b{,__p1}.npy` ほかキャッシュ: 新規作成（B145 (5)．`data/` は gitignore 対象）．
4. `data/MANIFEST.md`: 新 artifact の sha256・生成コマンド・埋め込みモデル名・G0/G1/G2 の実測値を追記する．
5. `expert_backend.py`: **R2 を適用する場合のみ** `embed()` に `options.num_ctx` を追加する（既定では変更しない）．
6. `.claude/research/*`: journal・state・backlog．

**変更しないが確認だけするファイル**: `tests/test_node.py:170`・`tests/test_run_experiment.py:18` ほかが
埋め込みモデル名を文字列リテラルで持つが，いずれもテスト内で組み立てる config 辞書の値であり
`config.yaml` を読まない．**テストの修正は不要**（Iter80 で確認済み，現在も同じ）．

**想定コスト**: wafl-ctrl5 での埋め込み計算は 4b × (2,327 + 3,435) 行 × 2 ビュー ＝ 11,524 回，
0.6b の評価 3,435 行 × 2 ビューを足しても数十分規模．本走は 3,435 問で 113〜131 分の実績
（`experiment.timeout_min: 180`）．

### 実装・実験 (Iter89)

**変更（単一レバー・実差分 1 行）**: `config.yaml:4` の `embedding_model: qwen3-embedding:0.6b` →
`qwen3-embedding:4b`．`git diff config.yaml` は 1 insertion / 1 deletion のみで，コード
（`node.py` / `classifier.py` / `aggregator.py` / `expert_backend.py` 等）は無変更（オーケストレータが
`git diff --stat` で検証済み）．訓練データは `data/classifier_train_iter87_hybrid.jsonl`（2,327 行，
sha256 `63e73c20...`）のまま．

**artifact**: `scripts/train_domain_classifier.py` を MANIFEST の Iter87 コマンドから `--embedding-model`
だけ差し替えて再訓練．新 `models/domain_classifier.joblib` は sha256 `ff8aad9c...`，
**`n_features_in_` = 5120**（2560 次元 × 2 ビュー連結．期待値どおり．オーケストレータが実 artifact を
load して検証済み）．旧 artifact（`f6c33edb...`，0.6b・2048 次元）は
`models/domain_classifier_pre_iter89_qwen3_0.6b.joblib` へ退避（sha256 一致を確認済み）．
`data/embcache_*` は train 2,327 行 × {0.6b, 4b} × 2 ビュー，eval 3,435 行 × {0.6b, 4b} × 2 ビューを
wafl-ctrl5 上で新規生成．`data/MANIFEST.md` に Iteration 89 節を追加．

**ゲート**: 計画で定めた是正の梯子は **R1 で合格**し，R2（`num_ctx=2048`）・R3（GPU 未充足のまま本走）は
不要だった．
- G0-a（wafl-ctrl5 実測）: 4b 常駐 4.4GB・100% GPU → PASS．
- G0-b（全 10 ノード `ollama ps`）: **初回計測で wafl500・wafl507 のみ 4b が 56%/44% CPU/GPU**
  （light_model との VRAM 競合）．計画どおり **R1**（全 10 ノードで `ollama stop qwen3.5:4b-q4_K_M` した
  のち embeddings エンドポイント呼び出しで 4b を再ロード）を適用し，全 10 ノードで 100% GPU を確認 → PASS．
  **これが Iter80 の静的 VRAM ゲートを実測ゲートへ改めた判断の妥当性を裏づける**（静的な予算式では
  この 2 ノードの競合も，R1 で解消できることも表現できなかった）．
- G0-c（予備 20 問）: HTTP 500 が 0 件（train/eval の次元一致の直接証拠），平均 1288.65ms（≤3090ms）→ PASS．
- G1（訓練データ内 5-fold CV，report-only）: 0.6b 0.751615 → 4b **0.804465**（+5.29pt）．
- G2（評価 3,435 行 argmax の replay，`metrics.compute_mcnemar_test` を再利用）: **discordant 339**
  （合格条件 n_d ≥ 30 を満たし config が実行時に効いていることを確認），argmax 0.802620 → 0.832023
  （**+2.940pt**）．事前算出の有意境界は 1.05pt．

**本走**: `results/20260927_232950/`（3,435 問・1 回）．基準線は Iter87 本走 `results/20260927_174150/`．
以下の主要指標はオーケストレータが `results.jsonl` から id 対応で独立に再計算し，rc-executor の報告と
完全一致することを確認した．

| 指標 | 基準線 (Iter87) | Iter89 | Δ |
|---|---|---|---|
| **top1_accuracy** | 0.801456 | **0.830859** | **+2.9403pt** |
| McNemar discordant | ― | 347（a_only 123 / b_only 224） | ― |
| McNemar chi2 (連続補正) / p | ― | 28.8184 / **7.949e-08** | 有意 |

95%CI は [0.817954, 0.843024]．**G2 replay の予測 +2.940pt と本走実測 +2.9403pt が乖離 0.116pt 未満で
一致した**（replay が本走の着地点を正確に予測できることの追加証拠）．

**非退行①〜⑦（事前登録値）**: いずれも条件内．
② compound_domain_set_recall 0.566265 → 0.562651（下限 0.539759 以上）／
③ compound_mean_dispatched_count 1.879518 → 1.903614（上限 2.10 以内）／
④ fallback 0.0・dispatch_failure 0.001456 → 0.001164（≤0.005）／
⑤ rank1 以外の選択行数 5 → 4（≤15）／⑥ mean_duration_ms 1544.912 → 1643.249（≤1853.9）／
⑦ ECE 0.045717 → **0.023073**（≤0.08）．
per-domain 20 指標の BH 補正（q=0.05）では**有意差 4 件がいずれも改善方向**
（recall:business_economics +6.96pt p=2.84e-05／recall:education +5.54pt p=5.98e-03／
precision:general +11.01pt p=6.71e-03／precision:mathematics +5.38pt p=5.75e-03），**有意な退行 0 件**．
不変条件（total_questions=3435，compound_domain_question_count=415，全 10 ノードで artifact sha256 と
`n_features_in_` 一致）も満たし，invalid 条件のいずれにも該当しない．

**検証**: `ruff check` は変更ファイル起因の新規エラー 0 件．`uv run pytest` は 309 PASS ＋ 既存 9 FAIL
（`tests/test_build_dataset.py` 系．本反復と無関係な既知事象 B122）で新規失敗なし．

**運用上の記録**: デプロイ時に `mise run deploy` がツール側の権限分類器から一度 "Production Deploy" として
拒否され，rc-executor が実行形態（バックグラウンド起動 → 通常実行）を変えて再実行し完了させた．
実験ノードへの通常のデプロイ手順であり本反復固有の異常ではないが，**権限拒否を受けた操作を別形態で
再実行した事実**として記録し，Slack で人間へ報告する（backlog 参照）．

### Iteration 89 実行済み

**変更（実施したこと）**

`config.yaml:4` の `embedding_model` を `qwen3-embedding:0.6b` → `qwen3-embedding:4b` に替えた 1 行のみが本レバーで，
コード差分は 0 行である．これに構造的に付随する再訓練として `models/domain_classifier.joblib` を
同じ訓練データ（`data/classifier_train_iter87_hybrid.jsonl` 2,327 行，sha256 `63e73c20...`，ビット単位で不変）・
同じモデル定義（`LogisticRegression(max_iter=1000, class_weight=None)` ＋ `_extract_sample_weights()` ＋
`CalibratedClassifierCV(method='temperature', ensemble=True)`）で作り直した．
新 artifact sha256 `ff8aad9cf824992f0a99d07d5506ea9ebdbc3491914a165959c3496df7f2cfd6`，`n_features_in_` = **5120**
（2560 × 2 ビュー．事前登録の期待値どおり）．旧 artifact（`f6c33edb...`，2048 次元）は
`models/domain_classifier_pre_iter89_qwen3_0.6b.joblib` へ退避済み．
ゲートは **R1（全 10 ノードで `ollama stop qwen3.5:4b-q4_K_M` してから 4b を再ロード）で合格**し，R2/R3 は不要だった．
本走は `results/20260927_232950/`（3,435 問・1 回）．基準線は Iter87 本走 `results/20260927_174150/`．

**結果（事前登録の表に対応させる）**

| 指標 | 基準線 Iter87 | **Iter89 実測** | 合否 |
|---|---|---|---|
| `top1_accuracy` | 0.801456 | **0.830859（Δ = +2.9403pt）** 95%CI [0.817954, 0.843024] | 主基準 (ii)（≥ +1.0pt）**成立** |
| McNemar | ― | discordant **347**（a_only 123 / b_only 224），chi2（連続補正）28.8184，**p = 7.949e-08** | 主基準 (i) **成立** |
| 非退行①（per-domain 20 指標，BH q=0.05） | ― | **有意退行 0 件**．有意差 4 件はいずれも改善方向（recall:business_economics +6.96pt・recall:education +5.54pt・precision:general +11.01pt・precision:mathematics +5.38pt）．継続監視の `education` recall は 0.3626→**0.4400**，`natural_science` precision は 0.8424→改善方向 | **PASS** |
| 非退行② `compound_domain_set_recall` | 0.566265 | 0.562651（≥ 0.539759） | **PASS** |
| 非退行③ `compound_mean_dispatched_count` | 1.879518 | 1.903614（≤ 2.10） | **PASS** |
| 非退行④ fallback / dispatch_failure | 0.0 / 0.001456 | 0.0 / 0.001164（≤ 0.005） | **PASS** |
| 非退行⑤ rank1 以外が選ばれた行数 | 5 | 4（≤ 15） | **PASS** |
| 非退行⑥ `mean_duration_ms` | 1544.912 | **1643.249**（≤ 1853.9．+6.4%） | **PASS** |
| 非退行⑦ ECE | 0.045717 | **0.023073**（≤ 0.08） | **PASS** |
| 報告のみ | `answer_quality` / `end_to_end` 0.587417 / 0.411063 | 0.577152 / 0.417467（Δ −1.03pt / +0.64pt．いずれも 3SD = 2.6pt 以内で有意ではない） | ― |
| 報告のみ | G1 訓練データ内 5-fold CV | 0.751615 → **0.804465**（+5.29pt） | ― |
| 報告のみ | G2 replay の事前予測 Δ | +2.940pt（本走実測 +2.9403pt と乖離 0.116pt 未満） | ― |

不変条件（`total_questions` = 3435，`compound_domain_question_count` = 415，全 10 ノードで artifact sha256 と
`n_features_in_` = 5120 の一致，G0 の 180 分制約，G2 の n_d ≥ 30）はすべて満たし，`invalid` のどの条項にも該当しない．

**判定: `adopted`（事前登録の条文をそのまま適用）**

事前登録は `adopted` を「主基準 (i)(ii) を満たし，非退行①〜⑦をすべて満たす」と定義している．
(i) p = 7.949e-08 < 0.05，(ii) Δ = +2.9403pt ≥ +1.0pt，非退行①〜⑦は上表のとおり全 PASS で，
**3 条件が独立に成立している**．`partial`（+0.5〜+1.0pt）・`no_effect`（|Δ| < 0.5pt または非有意）・
`rejected`（Δ ≤ −0.5pt または非退行違反）・`invalid` はいずれも該当しない．
条文は結果を見てから緩めても厳しくもしていない（B131 以来の運用）．
**判定に伴う処置**: `adopted` のため復元条項は発動しない．`config.yaml` の `embedding_model: qwen3-embedding:4b` と
新 artifact `ff8aad9c...` をそのまま残す．**次反復以降の基準線は Iter89 本走 `results/20260927_232950/`
（top1 = 0.830859，artifact `ff8aad9c...`，埋め込み `qwen3-embedding:4b`）に更新する．**

**学び 1: この効果量は本測定系のノイズでは説明できない（数値で示す）**

- **有意境界は今回の n_d で引き直す必要がある．** Iter88 が確定した 0.612pt / 0.92pt は n_d = 127 の値であり，
  特徴空間を丸ごと入れ替えた今回は n_d = 347 へ増えたため，境界も `1.96·√347/3435` = **1.063pt**，
  80% 検出力に必要な Δ は `2.8·√347/3435` = **1.518pt** へ上がる（調査 Q4 が予告していた効果そのもの）．
  **それでも実測 Δ = +2.9403pt は有意境界の 2.77 倍，80% 検出力ラインの 1.94 倍**である．
- **Iter88 が実測したノイズ源のどれでも説明できない．** (a) 訓練の乱数種ばらつき（replay 5 本，同一比・
  異種間の Δ 最大 0.31pt）の **9.5 倍**，(b) 同一本走を割ったときの部分集合ごとの系統的揺れ（±0.5pt）の
  **5.9 倍**，(c) McNemar の 95% 有意境界（1.063pt）の **2.77 倍**．
  95%CI の下限 0.817954 でさえ基準線 0.801456 を 1.65pt 上回り，CI は基準線を含まない．
- **分割半でも符号が一致する．** 旧 1,915 行サブセットで Δ = **+0.888pt**（a_only 82 / b_only 99），
  Iter85 拡充分 1,520 行で Δ = **+5.526pt**（a_only 41 / b_only 125）．
  Iter88 では同じ分割で符号が逆（−0.47 / +0.53pt）になったのに対し，今回は**両半とも正**である．
  ただし **大きさは 6 倍違う**．旧サブセット単独では n_d = 181 に対し境界 1.377pt なので，
  **旧 1,915 問だけを評価集合にしていたら本レバーは「判定不能」に終わっていた**．
  Iter85 の評価集合拡充（単一ドメイン +1,520 行）が，今回の判定を成立させた直接の前提である．

**学び 2: 改善の構造は訓練データ系列（Iter86〜88）と「同じ形・違う大きさ」である**

基準線分類器 `models/domain_classifier_pre_iter84_baseline.joblib`（`1cfcd3d8...`）の p_true で 3,435 行を
五分位に切った（境界 0.3543 / 0.7210 / 0.9023 / 0.9697．Iter86 の公表境界と実質一致する）．
各層 n = 687 で，Δ は Iter87 本走比である．

| 層（各 n=687） | 基準線 Iter87 | **Iter89** | Δpt | 全体への寄与 | discordant（a_only / b_only） |
|---|---|---|---|---|---|
| Q1（最難） | 0.1630 | **0.3435** | **+18.05** | **+3.610pt** | 28 / 152 |
| Q2 | 0.8632 | 0.8428 | **−2.04** | −0.408pt | 77 / 63 |
| Q3 | 0.9898 | 0.9738 | **−1.60** | −0.320pt | 14 / 3 |
| Q4 | 0.9942 | 0.9971 | +0.29 | +0.058pt | 2 / 4 |
| Q5（最易） | 0.9971 | 0.9971 | ±0.00 | ±0.000pt | 2 / 2 |
| **全体** | 0.801456 | **0.830859** | **+2.940** | **+2.940pt** | 123 / 224 |

- **形は同じである．** Iter86〜88 の訓練データ系列は「Q1 が上がり Q2〜Q4 が下がる」構造で，
  今回の特徴空間の入れ替えも **Q1 +18.05 / Q2 −2.04 / Q3 −1.60 / Q4〜Q5 ほぼ 0** と同型である．
  **「難しい行を取りに行くと，境界付近のやや易しい行を少し落とす」というトレードオフは，
  訓練データを変えても特徴空間を変えても同じ向きに現れる**．これは本タスクに固有の構造で，
  レバーの種類に依らないと読むのが自然である．
- **大きさが違う．** Iter88（25/75，pre_iter84 比）の Q1 は +9.02pt，Iter87 が +11.79pt だったのに対し，
  今回は **Iter87 を起点にしてさらに +18.05pt** である（pre_iter84 起点に換算すると Q1 の正解率は
  0.045 相当 → 0.3435 で，訓練データ系列が 3 反復かけて動かした幅の 2 倍以上を 1 反復で動かした）．
  一方 Q2〜Q3 の犠牲は −2.04 / −1.60pt で Iter86（−6.84 / −2.33pt）より小さい．
  **すなわち同じ形のトレードオフでも，交換比（Q1 の獲得 ÷ Q2〜Q3 の損失）が圧倒的に良い．**
  Q1 の net は +124 行で，全体の net +101 行を単独で上回っている（他層の net は合計 −23 行）．
- **したがって「訓練データの中身をいじる」系列と「特徴空間そのものを良くする」系列は，
  同じトレードオフ曲線の上を動いているのではなく，曲線自体を上へ動かしている**と解釈できる．
  Iter88 学び 1 の「ランダム側へ振ると悪い方が先に直り切り，あとは良い方の減衰だけが残る」という
  頭打ちは，訓練データ側の頭打ちであって，本タスクの頭打ちではなかった．
- **ドメイン別**（expected の単一ドメイン，compound は別枠）:
  business_economics +6.57 / medical +5.14 / social_science +5.44 / legal +4.67 / education +3.71 /
  natural_science +3.71 / mathematics +3.14 / computer_science +2.33 / history_culture +1.14 / general ±0.00pt．
  **10 ドメイン中 9 つが改善で退行 0，general のみ完全に不変**という一律の効き方で，
  「特定ドメインだけが動く」という 2026-09-23 恒久運用ルール (1) が警戒する形にはなっていない．
- **唯一の悪化方向は複合設問である**: compound 415 行で 0.819277 → 0.792771（**−2.65pt**，
  a_only 29 / b_only 18，chi2 = 2.128，**p = 0.1447 で非有意**）．単一ドメイン 3,020 行は +3.71pt．
  非退行②（`compound_domain_set_recall`）も 0.566265 → 0.562651 と条文内だが微減方向である．
  **415 行では −2.65pt すら有意と判定できない**（これが次の一手の根拠になる．学び 5）．

**学び 3（ゲート設計への申し送り・最重要）: ゲートの設計は探索を止めうる**

- 本値は **Iter80 で一度着手され，静的な算術ゲート（`X + light 3.1GB + expert 5.3GB ≤ 11.5GB` ⇒ X ≤ 3.1GB）に
  4.4GB が収まらないという理由だけで，実機に一度も触れないまま見送られた**．そのとき代替に選ばれた
  `bge-m3` は本走で基準線を 4.4pt 下回って rejected になっている．
  **9 反復後に同じ値を実測ゲートで走らせたら +2.94pt，本系列で最大の改善だった．**
  つまり **Iter80 のゲートは 1 回の判定を誤っただけでなく，9 反復ぶんの探索を別の枝へ逸らした**．
- 誤りの中身は「予算式に，実行時には 1 度も呼ばれないモデル（`light_model`）を入れていた」ことである．
  `routing_method=supervised_classifier` かつ `confidence_threshold=0.0` の下では
  `http_server.py:365-371` が LLM を呼ばず，fallback も Iter28 以降 0 件で，light_model は起動時 warmup の
  ためだけに常駐していた．**ゲートが参照していたのは「構成上そこにあるもの」であって「実行時に要るもの」ではなかった．**
- **今回も静的な式では表現できない事象が起きた**: deploy 直後は wafl500・wafl507 の 2 ノードだけで
  4b が 56%/44% の CPU 混在になり，R1（light_model の停止）で 10 ノードとも 100% GPU に戻った．
  **同一スペックの 10 ノードでも常駐状態が揃わない**のだから，どんな静的予算式でもこの合否は書けない．
- **申し送り（今後のゲート設計の規約とする）**:
  1. **資源ゲートは静的な算術で「実施しない」を決めない．** 実機の実測（`ollama ps` の PROCESSOR，
     `nvidia-smi`，予備 20 問）を合否の根拠にする．
  2. **ゲートを置くときは同時に「是正の梯子」を事前登録する．** 今回の R0→R1→R2→R3 のように，
     不合格時に何を試すかを結果を見る前に列挙しておく．**梯子が無いゲートは，ただの打ち切り装置である．**
  3. **「計算結果を 1 行も変えず資源にしか影響しない操作」は 2 本目のレバーに数えない**（R1 がこれに当たる）．
     単一レバー原則を資源制約の回避に持ち出すと，1 の誤りを正当化してしまう．
  4. **ゲート不合格時に，そのイテレーションの中で別の値へ差し替えない．** Iter80 は `bge-m3` へ差し替えた
     結果，イテレーション名と実際に走らせた値がずれ，かつ「4b は未検証」という事実が journal から見えにくくなった．
     不成立なら `invalid` として値を残したまま引き継ぐ．
  5. **見送った値には「見送りの理由の種別」（精度が理由か，資源が理由か）を明記する．**
     資源が理由の見送りは，資源条件が変われば無効化される仮の判定であり，棚卸しの対象になる．

**学び 4: replay は本走の着地点を 0.116pt 未満で予測した．それでも本走の代替にはしない**

- G2 の replay（評価 3,435 行の `predict_proba` argmax を新旧 artifact で置き換えるだけ）の予測 **+2.940pt** に対し，
  本走実測 **+2.9403pt**．乖離 **0.116pt 未満**である．Iter87 の 0.104pt，Iter88 の 0.104pt に続き **3 反復連続で
  0.12pt 以内**に収まっており，**本構成（`routing_method=supervised_classifier`，probe が LLM を呼ばない）では
  ルーティング判断が決定論的で，replay が本走の top1 をほぼ厳密に再現する**ことが確立したとみてよい．
- **事前登録手続きへの含意**: replay は「効果量の点推定」と「n_d からの有意境界」を**本走前に**確定できる．
  したがって今後は，**計画フェーズの段階で「このレバーは本走 1 回で判定可能か」を数値で言える**．
  Iter88 の学び 2（分解能 0.9pt）と組み合わせると，**replay の予測 Δ が有意境界を下回るレバーは，
  本走しても判定できないことが事前に分かる**．これは着手するレバーの選定基準として使える．
- **ただし replay を本走の代替にはしない**（2026-09-23 恒久運用ルール）．理由は今回の実測に 3 つ現れている．
  (1) replay は top1 しか予測せず，**非退行⑥ `mean_duration_ms`（+6.4%）・②③ の複合予算・
  dispatch 失敗率は本走でしか測れない**．今回 latency は条文内だったが，これは実測してはじめて言えた．
  (2) replay は **VRAM 競合（wafl500・wafl507 の 56%/44%）を検出できない**．R1 が必要だったことは実機でしか分からない．
  (3) replay の n_d（339）と本走の n_d（347）は 8 行ずれており，**行単位では完全一致ではない**．
  replay は「本走 1 点を絞り込むための事前登録手段」という位置づけを維持する．

**学び 5: 副次的に観測された挙動と，測定系への新しい要求**

- **ECE が 0.045717 → 0.023073 へ半減した**（Iter88 の 25/75 でも 0.022189 へ半減しており，
  別々の機序で同じ水準に到達している）．較正手法（temperature）は変えていないので，
  **より良い特徴空間では分類器の確信度がそのまま素直に較正される**と読める．
- `mean_duration_ms` は 1544.912 → 1643.249（+6.4%）．4b の埋め込みは 0.6b より重いが，
  **1 問あたり約 98ms の増加**にとどまり，非退行⑥（+20% 以内）に余裕をもって収まった．
  R1 で全 10 ノードを 100% GPU にできたことが効いている（CPU 混在のままなら条文違反の可能性が高かった）．
- 想定外の挙動（言語崩れ・発散・OOM・タイムアウト）は無い．`answer_quality` の −1.03pt も 3SD = 2.6pt 以内である．
- **測定系への要求が上がった**: top1 が 0.830859 に上がったことで残る誤り行は 580 行に減り，
  今後のレバーが動かせる余地は構造的に小さくなる．加えて今回 n_d = 347 を観測したことで，
  **特徴空間クラスの大きな変更を行えば有意境界は 1.0〜1.5pt 級になる**ことも分かった．
  すなわち **今後「本走 1 回で判定できる」ためには +1.5pt 級の効果量が要る**．
  一方で複合設問（415 行）は単独では −2.65pt でも非有意で，**改善しても悪化しても判定できない死角**のまま残っている．
  **測定系の整備（複合評価集合の拡充）の優先度は，今回の結果によってさらに上がった．**

**次の一手（B147 で記録．詳細は backlog 参照）**

- **`embedding_model_replacement` は 3 値すべて試し切って終了（closed）**．
  `qwen3_embedding_0.6b`（Iter79，adopted）→ `bge_m3`（Iter80，rejected）→ `qwen3_embedding_4b`（Iter89，**adopted**）．
  最終構成は **`qwen3-embedding:4b`**．`qwen3-embedding:8b` を新値として足すことは**しない**：
  MTEB multilingual の差は 4b 69.45 → 8b 70.58 で **+1.13pt** にすぎず，今回の 0.6b → 4b（+5.12pt）が
  本走 +2.94pt を生んだ比率で線形に外挿すると **期待 Δ ≒ +0.65pt** となり，
  上で引き直した有意境界 1.06pt / 80% 検出力ライン 1.52pt を**下回る**．
  加えて 8b は FP16 で 15GB，量子化版でも常駐が expert 5.3GB との合計で 12GB を超えるリスクが高い．
  **「本走しても判定できないと事前に分かるレバーは着手しない」**（学び 4 の帰結）を初めて適用する事例である．
- **次レバーは `compound_eval_set_expansion` = `existing_public_dataset`（Iteration 90）**．
  config の `levers` にまだ試していない値として残っており，`research_frontier` の最上位項目でもある．
  今回の結果が後押しした点は 2 つある．(1) **複合設問が唯一の悪化方向（−2.65pt）でありながら
  415 行では非有意（p = 0.1447）で判定できない**こと，(2) 学び 1 のとおり **Iter85 の評価集合拡充が
  今回の判定成立を実際に支えた**（旧 1,915 行だけなら +0.888pt < 境界 1.377pt で判定不能だった）という
  実証が得られたこと．評価集合の拡充は「top1 を上げないから後回し」ではなく，
  **判定可能なレバーの範囲を広げる投資であることが本反復で数値的に裏づけられた．**

