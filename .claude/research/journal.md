## Iteration 72: conformal校正集合を評価データの層化ホールドアウトへ変更し交換可能性を回復して被覆を再測定

### 調査 (Iter72)

**問い**

- Q1: split conformal の被覆保証が要求する交換可能性とは具体的に何であり，本実装の校正経路（`classifier_train.jsonl` の OOF・素の LogisticRegression）はそのどこを破っているのか．
- Q2: 校正集合を**評価データ自身の層化ホールドアウト**から取る構成は，conformal の手続きとして正当か（校正データがモデル適合に使われていないと言えるか）．
- Q3: その構成にしたとき名目 0.90 に対する実測被覆はどこへ着地するか（本実行前に着地点を言語化する，Iter71 で有効だった手順）．

**Q1: 交換可能性の要件と本実装の破れ（出典付き）**

- split conformal の有限標本被覆 `P(Y ∈ C(X)) ≥ 1-α` は，校正点 n 個とテスト点 1 個の**非適合スコアが交換可能**であることのみから従う（Angelopoulos & Bates, "A Gentle Introduction to Conformal Prediction", arXiv:2107.07511, 2021）．スコアは**同一の固定されたモデル**から計算され，かつそのモデルの学習にどちらの点も使われていないことが前提である．
- 交換可能性が破れると「校正集合の適合スコア分布がテスト分布と一致しなくなる」ことが被覆逸脱の直接の原因になる（"Ensuring Calibration Robustness in Split Conformal Prediction Under Adversarial Attacks", arXiv:2511.18562, 2025，Introduction．同論文は Barber et al. 2023 の beyond-exchangeability 系の議論を引く）．
- 本実装（`scripts/evaluate_classifier_calibration.py:predict_calibrated_rows()` L226-292）は，校正スコアを `classifier.estimator`（素の `LogisticRegression`）を `StratifiedKFold(5)` で**再学習**した fold モデルの `predict_proba`（OOF）から作る一方，評価スコアは `models/domain_classifier.joblib`（`CalibratedClassifierCV`，全データ学習・temperature 較正済み）の `predict_proba` から作る．**スコア関数そのものが校正側と評価側で別物**であり，上記の前提が明確に破れている（Iter71 考察 §2 で実測とともに確定済み．q_hat=0.000980）．これは「分布シフト」以前の，スコア関数の不一致という基本的な破れである．

**Q2: 評価データの層化ホールドアウトを校正集合にすることの正当性**

- split conformal が禁じるのは「**モデルの学習に使った**データでスコアを作ること」であり，校正データが評価データと同じプールから取られること自体は禁じられていない．むしろ校正半と評価半を同一プールからランダムに割れば，交換可能性は構成上ほぼ自明に成立する．
- 本リポジトリでは分類器 `models/domain_classifier.joblib` は `data/classifier_train.jsonl`（1,427 行）のみで学習されており，`data/dataset.jsonl`（1,600 行）は学習に一切使われていない．したがって dataset.jsonl の任意の部分集合は分類器にとって held-out であり，校正集合として適格である．
- 留保（事実として記録）: `classifier_train.jsonl` と `dataset.jsonl` は query 文字列で **72 件重複**する（Iter71 考察時に実測）．この 72 件は校正半・評価半へほぼ同率で散るため両半の交換可能性は壊さないが，「分類器が既見の行を含む」点は結果の解釈に付記する．

**Q3: 非ランダム化 APS の過被覆と，事前シミュレーションによる着地点の予測**

- Romano et al. (2020) の APS はスコアに一様乱数 U を混ぜることで被覆を名目値へ厳密に一致させるが，**本実装は非ランダム化版**であり，閾値をまたいだクラスを丸ごと集合に含めるため構造的に名目を上回る（過被覆する）．APS 系が大きな集合を生むことは Angelopoulos et al., "Uncertainty Sets for Image Classifiers using Conformal Prediction"（RAPS，arXiv:2009.14193, 2021）でも報告されており，同論文はホールドアウトで閾値を選び直す運用（例: α=10% で 93% の推定確率質量を使う）を APS と呼んでいる．本イテレーションの構成はまさにこの運用に対応する．なお「非ランダム化版が過被覆する」という因果の説明部分は一次資料の記述そのものではなく，打ち切り規則からの当方の導出である．
- **事前シミュレーション（本実行前に実施．`results/20260919_215923/Iter71_qhat_alpha_lower.jsonl` の `probabilities` 1,600 行を使用）**: 本レバーは `probabilities` を一切変えないため，被覆は既存出力から**ほぼ厳密に再現計算できる**（Iter70 の掃引が評価集合自身で閾値を選んだ「楽観値」だったのとは性質が異なる）．`expected_domains[0]` で層化した `StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)` の第 1 返値を校正半・第 2 返値を評価半とし，校正半の真クラス補数スコアの `⌊(n+1)α⌋/n` 分位点（n=800, α=0.10 ⟹ 80/800 ＝昇順 80 番目）を q_hat とした結果:

| seed | 役割 | q_hat | coverage（評価半 n=800） | mean_set_size |
|---|---|---|---|---|
| **42** | **本実験の構成** | **0.051766** | **0.9400** | **5.521** |
| 42 | cross-fit（役割入替） | 0.052441 | 0.9487 | 5.532 |
| 1 | A / cross | 0.052567 / 0.051766 | 0.9513 / 0.9375 | 5.548 / 5.501 |
| 7 | A / cross | 0.054988 / 0.050475 | 0.9400 / 0.9513 | 5.430 / 5.586 |
| 2026 | A / cross | 0.056042 / 0.049520 | 0.9425 / 0.9463 | 5.322 / 5.702 |
| 123 | A / cross | 0.059741 / 0.046794 | 0.9387 / 0.9537 | 5.291 / 5.725 |

  q_hat は Iter71 の 0.000980 から **0.05 前後**へ 1〜2 桁移動し，Iter71 計画節の掃引表が示した帯（0.04 ≤ q_hat ≤ 0.145）の内側に入る．**10 分割中 coverage が帯 0.88–0.95 に収まるのは 7/10，上限 0.95 を超えるのは 3/10** であり，**本構成（seed=42）の予測値は coverage=0.9400・mean_set_size≈5.52 で合格側**だが，帯上限まで 1.0pt（二項 SE 0.0084 の 1.2 倍）しか余裕がなく**分割の引き次第で判定が反転しうる**．この事実を判定前に事前登録しておく（seed は 42 に固定し，結果が帯外でも seed を振り直して合格を探すことはしない）．
- 副次的に確認した事実: 全 1,600 行を校正に使った場合の q_hat は 0.052250 で，半分にしても中心値はほぼ変わらない（分割半減の影響は q_hat の**ばらつき** 0.0468–0.0597 に現れる）．

### 計画 (Iter72)

**仮説**

Iter71 の過被覆（coverage=0.996875）の原因は実装の第 4 の欠陥ではなく，**校正スコアと評価スコアが別の確率モデル（生 LR の OOF vs 較正済み CalibratedClassifierCV）から出ていること**である（調査 Q1）．校正集合を評価データ自身の層化ホールドアウトに置き換え，校正半のスコアを**評価と同一の `models/domain_classifier.joblib` の `predict_proba`** で計算すれば，学習データ量・確率較正の有無・元データという 3 つの差が同時に消えて交換可能性が回復し，q_hat は 0.000980 から 0.05 前後へ移動して coverage は名目 0.90 の近傍（非ランダム化 APS の過被覆分を含め 0.93〜0.95）へ着地する．

**単一レバー**

`conformal_calibration_exchangeability`: 校正集合の取り方を `oof_train`（現行既定．`data/classifier_train.jsonl` 1,427 行を生 LR の 5-fold OOF で採点）→ `eval_holdout`（`data/dataset.jsonl` 1,600 行を `expected_domains[0]` で層化 50/50 分割し，校正半 800 行を評価と同一の分類器で採点）へ変更する．動かすのはこの 1 点のみ．

**固定する構成（直近の最良構成に固定）**

- 分位点方向: `--qhat-quantile-direction alpha_lower`（Iter71 で実装確定．レバーではなく固定値として使う）．
- 集合構成: `--set-construction corrected_aps`（Iter70 で修正・維持と決定）．
- q_hat の母集団: `--qhat-source true_class`（Iter69 で確定）．
- 名目水準: `--confidence-level 0.90`（**レバーに含めない**．名目を振った曲線は付随報告）．
- 評価データ `data/dataset.jsonl`（1,600 行），分類器 `models/domain_classifier.joblib`，埋め込み `nomic-embed-text:latest`（wafl-ctrl5 の `127.0.0.1:11435`），`--education-logit-bias 0.0` / `--education-threshold 0.0`，`--fine-tuned-embed-model` は指定しない（ollama 分岐を通す）．
- `config.yaml`・`http_server.py`・`classifier.py`・`aggregator.py`・`mise.toml` は変更しない．実機ノード wafl500〜509 は使用しない．

**変更箇所（`scripts/evaluate_classifier_calibration.py` 1 ファイル＋テスト．行番号は 2026-09-23 計画時点，全 582 行）**

1. CLI に 2 引数を追加する（`main()` L521-530 の `--qhat-quantile-direction` の直後）:
   - `--calibration-source`（`choices=["oof_train", "eval_holdout"]`, `default="oof_train"` ＝現行挙動を温存）
   - `--holdout-seed`（`type=int`, `default=42`）
2. `main()` の `--output` 有無による**2 分岐の両方**へ渡す．**Iter69・70・71 と 3 回連続で警告されている伝播漏れ箇所であり，今回もチェックリストとして実装完了時に明示確認すること**:
   - [ ] stdout 側 `if args.output is None:`（**L538-557** の `_run(...)`）
   - [ ] ファイル出力側 `with open(args.output, "w", ...)`（**L558-578** の `_run(...)`）※本実験が通るのはこちら
3. `_run()`（L407-444）のシグネチャへ `calibration_source: str = "oof_train"` / `holdout_seed: int = 42` を追加し，`predict_calibrated_rows()` 呼び出し（L427-438）へ伝播する．
4. `predict_calibrated_rows()`（L187-404）を `calibration_source` で分岐させる．**現行の `oof_train` 経路（L226-292）は一切書き換えず，`if calibration_source == "oof_train":` の下へそのまま置く**（後方互換の md5 一致を壊さないため）．`eval_holdout` 経路は次の順序で実装する:
   1. 評価データ全 1,600 行の埋め込みを先に計算し（ollama 分岐．`local_model` 側も同じ構造で通す），`classifier.predict_proba` で確率行列を得る．**校正 1,427 行の埋め込み計算は不要になるためコストはむしろ下がる**．
   2. `labels = [classes.index(r["expected_domains"][0]) for r in dataset]`（複合設問 100 行も先頭ドメインを真クラスとする．coverage の定義と一致させる）で層化し，`StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=holdout_seed)` を 1 回 `split` する．**第 1 返値（train index）を校正半，第 2 返値（test index）を評価半とする**（この対応を逆にすると seed=42 の予測値 0.9400 が cross-fit 側 0.9487 に変わるため，実装時に必ず確認する）．
   3. 校正半 800 行について真クラス補数スコア `1 - cumsum`（確率降順で真クラスに到達した時点）を計算し，`cp_data = {"true_class_scores": ...}` を作る（`all_scores` は `true_class` 固定のため不要だが，キーの欠落で `qhat_source="all"` が落ちないよう `all_scores` も同形で埋めるか，`eval_holdout` では `qhat_source="all"` を `ValueError` で拒否するかを実装時に決め，どちらにしたか journal に記す）．
   4. 各行の出力 dict に `"split": "cal" | "eval"` を付与する．予測集合は全 1,600 行について計算してよいが，**集計は eval 半のみで行う**．
5. 既存の stderr 診断 print（L309-315）へ `calibration_source` と `n_cal` を追加する（`eval_holdout` では `n_cal=800`，`q_hat≈0.05` が出るはず＝**レバー発火の証拠**．`oof_train` のままなら `n_cal=1427`，`q_hat=0.000980`）．
6. `tests/test_evaluate_classifier_calibration.py` へ既存の `test_qhat_source_*` / `test_set_construction_*` / `test_qhat_quantile_direction_*` に倣い追加する:
   (a) 層化分割が 800/800 かつ両半のドメイン構成比が一致すること，(b) 同一 seed で分割が再現すること，(c) 未知の `calibration_source` で `ValueError`，(d) 既定値 `oof_train` の回帰テスト（現行経路を通ること）．

**到達コードパス**

CLI `--conformal-prediction --calibration-source eval_holdout --holdout-seed 42 --qhat-quantile-direction alpha_lower --set-construction corrected_aps --qhat-source true_class --confidence-level 0.90 --output ...`
→ `main()` L558（ファイル出力分岐）→ `_run()` L407 → `predict_calibrated_rows()` L187 → **新設の `eval_holdout` 校正ブロック（q_hat の値が変わる唯一の地点）** → 評価 1,600 行ループ（ollama 分岐 L368-403）→ `_compute_prediction_set()` L387 → 出力 jsonl の `prediction_set` / `set_size` / `split` → eval 半 800 行で coverage・mean_set_size を集計．`config.yaml` を経由しないため「デプロイ漏れ」型の失敗は構造上起こらない．唯一のリスクは変更箇所 2 の CLI 2 分岐伝播漏れであり，予備実行（先頭 20〜40 行）の stderr で `calibration_source=eval_holdout n_cal=800 q_hat≈0.05` を目視確認して潰す．

**成功条件（事前登録）**

名目 0.90，評価半 n=800．比較対象は `results/20260919_215923/Iter71_qhat_alpha_lower.jsonl`（Iter71 実測 coverage=0.996875）．

| 指標 | 定義 | Iter71 実測 | 合格条件 |
|---|---|---|---|
| coverage（主基準） | eval 半で `mean(expected_domains[0] in prediction_set)` | 0.996875（n=1600） | **0.88 ≤ coverage ≤ 0.95** |
| mean_set_size（副基準） | eval 半の `mean(set_size)` | 9.66375（n=1600，実測再計算） | 報告のみ（判定に用いない） |
| ECE | eval 半で `metrics.py:compute_ece()` | 0.062998（n=1600） | 報告のみ（母集団が変わるため閾値判定に用いない） |

- **adopted**: coverage が 0.88–0.95 に入り，かつ下記の非退行条件を全て満たすこと．
- **rejected**: coverage が帯外であること．外れ方の向きで解釈を分ける（事前登録）:
  - `coverage > 0.95`（過被覆）: 交換可能性は回復したが非ランダム化 APS の過被覆が支配的，という解釈になる．次の候補は randomized APS（Romano et al. 2020 の U 項）だが，これは**別レバー**であり本イテレーションでは扱わない．
  - `coverage < 0.88`（過少被覆）: 交換可能性以外の要因が残っていることになり，config note のとおり「本データ・本分類器では有効な動作点が存在しない」として conformal 系列を閉じる判断を人間に諮る．
- **非退行条件**（いずれも eval 半 800 行について）:
  1. `selected_domain` が Iter71 出力の同 id 行と完全一致すること．
  2. `confidence`・`probabilities` が同 id 行と一致すること（許容差 1e-9）．
  3. `set_size` が 1 以上 10 以下で，`prediction_set` に重複がないこと．
  4. 後方互換: `--calibration-source oof_train` での再実行が Iter71 出力（`Iter71_qhat_alpha_lower.jsonl`）と **md5 一致**すること．
  5. 予備実行の stderr に `calibration_source=eval_holdout` と `n_cal=800` が出ること（発火証拠）．
- **ノイズ幅**: n=800・p≈0.94 で二項 SE≈0.0084．帯端からの逸脱は SE の何倍かを必ず併記する．事前シミュレーションでは分割由来のばらつきが coverage で 0.9375〜0.9537（10 分割）あり，**SE と同程度以上の分割ノイズが乗る**ことを前提に解釈する．
- **付随報告（判定に用いない）**: (i) 役割を入れ替えた cross-fit の coverage（予測値 0.9487），(ii) 名目水準を 0.70〜0.95 で振った coverage/mean_set_size 曲線，(iii) 実測 q_hat と校正半スコア分布の分位点，(iv) `classifier_train.jsonl` と重複する 72 件が校正半・評価半それぞれに何件入ったか．

**期待効果**

Iter56 以来 4 反復続いた conformal 系列について，「実装欠陥の列挙」ではなく「交換可能性という統計的前提」で決着をつける．成立すれば conformal prediction は adopted で系列を閉じ（ただし mean_set_size≈5.5 は top-2 dispatch への流用には大きすぎるため，実行時経路への配線は別途判断），不成立なら系列を閉じる判断を人間に諮る．

**コスト**: 1 実行 10〜30 分（埋め込み 1,600 行のみ）．GPU 実機占有なし，分類器再訓練なし．

### 実装 (Iter72)

変更したファイルは 2 つのみ（`config.yaml`・`http_server.py`・`classifier.py`・`aggregator.py`・`mise.toml` は不変．実機ノード wafl500〜509 は不使用）．

1. `scripts/evaluate_classifier_calibration.py`
   - CLI に `--calibration-source {oof_train,eval_holdout}`（既定 `oof_train`）と `--holdout-seed`（既定 42）を追加．
   - `main()` の **stdout 分岐・ファイル出力分岐の両方**へ伝播（Iter69〜71 で 3 回連続して警告されていた箇所．計画のチェックリストどおり実装完了時に両分岐を目視確認した）．
   - `_run()` → `predict_calibrated_rows()` へ伝播．
   - `predict_calibrated_rows()` を `calibration_source` で分岐．**`oof_train` 経路は 1 行も書き換えず** `if conformal_prediction and calibration_source == "oof_train":` の下へそのまま残した（後方互換 md5 一致のため）．`eval_holdout` 経路は「全 1,600 行を評価と同一の `classifier.predict_proba` で採点 → `expected_domains[0]` で層化 `StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)` → 第 1 返値を校正半・第 2 返値を評価半 → 校正半 800 行の真クラス補数スコアで q_hat」の順で構成した．
   - 計画 4-3 で保留していた `qhat_source="all"` の扱いは，**`ValueError` で明示的に拒否する**方を選んだ（`all_scores` をダミーで埋めると「動いたが意味のない q_hat」を生む危険があるため）．
   - 各行に `"split": "cal" | "eval"` を付与．集計は eval 半のみ．
   - stderr 診断 print に `calibration_source` と `n_cal` を追加．
2. `tests/test_evaluate_classifier_calibration.py`: 層化分割の 800/800・両半のドメイン構成比一致・同一 seed での再現・未知値の `ValueError`・既定値 `oof_train` の回帰，の 5 件を追加（既存 13 件と合わせ 18 件 PASS）．

**実装中に発見・修正した既存バグ**: `eval_holdout` 経路を追加した際，評価ループ側が `oof_train` 経路でのみ定義される局所変数を参照しており `UnboundLocalError` が出た．既存テストはこの分岐を一度も通らないため検出できていなかった（**新経路を足すときに「既存経路でしか初期化されない局所変数」を洗い出す**という手順が要る，という学び）．

### 実験・分析(実行) (Iter72)

結果ディレクトリ `results/20260923_132431/`．2 実行を行った（GPU 実機占有なし・分類器再訓練なし・所要は 2 実行合計で約 6 分）．

- A（後方互換アンカー）: `--calibration-source oof_train` → `Iter72_oof_train_backcompat.jsonl`．stderr は `calibration_source=oof_train ... q_hat=0.0010 n_cal=1427`．
- B（本実行）: `--calibration-source eval_holdout --holdout-seed 42 --qhat-quantile-direction alpha_lower --set-construction corrected_aps --qhat-source true_class --confidence-level 0.90` → `Iter72_eval_holdout.jsonl`．stderr は `calibration_source=eval_holdout ... q_hat=0.0518 n_cal=800`（**レバー発火の証拠**）．

| 指標 | Iter71 実測（n=1600） | Iter72 実測（eval 半 n=800） | 合格条件 | 判定 |
|---|---|---|---|---|
| coverage（主基準） | 0.996875 | **0.940000** | 0.88 ≤ coverage ≤ 0.95 | **PASS** |
| mean_set_size（副基準） | 9.66375 | 5.52125 | 報告のみ | — |
| ECE | 0.062998 | 0.076460 | 報告のみ（母集団が変わる） | — |
| q_hat | 0.000980 | 0.051766 | — | — |

- set_size 範囲 1〜9（上限 10 以内），分布 `{1:9, 2:35, 3:54, 4:97, 5:175, 6:179, 7:176, 8:68, 9:7}`．
- 非退行条件 1〜5 は全て充足: (1) `selected_domain` が Iter71 出力の同 id 行と完全一致，(2) `probabilities` の最大差 9.99e-16（許容 1e-9），(3) set_size 1〜9・`prediction_set` に重複なし，(4) A の md5 = `575594c7a973fe615202485de7fe0a8f` が Iter71 出力と一致，(5) stderr の発火証拠あり．
- テスト 18 passed / `uv run ruff check scripts/evaluate_classifier_calibration.py tests/test_evaluate_classifier_calibration.py` All checks passed（リポジトリ全体の ruff 23 件は `scripts/analyze_iter43.py` 等の既存指摘で本変更と無関係）．

**付随報告（判定に用いない．いずれも本実行 jsonl の `probabilities` からの再計算）**

- cross-fit（校正半と評価半の役割を入替）: q_hat=0.052441, coverage=0.94875, mean_set_size=5.5325（事前登録の予測値 0.9487 と一致）．
- 名目水準を振った曲線（校正半で q_hat を取り直し評価半で測定）: α=0.05→coverage 0.97375 / mss 6.90，α=0.10→**0.94000 / 5.52**，α=0.20→0.91500 / 4.27，α=0.30→0.86500 / 3.40．
- 校正半（in-sample 相当）の coverage=0.94875．
- `classifier_train.jsonl` と重複する 72 件の内訳: 校正半 40 件・評価半 32 件（ほぼ均等に散っており，どちらかの半へ偏ってはいない）．

### 分析(解釈) (Iter72)

**1. 主基準は PASS．ただし帯上限側の余裕は薄い（事前登録どおり）**

coverage=0.940000 は帯 0.88–0.95 の内側．二項 SE = √(0.94·0.06/800) = 0.008396 として，**帯下限 0.88 からは +7.15 SE，帯上限 0.95 までは -1.19 SE**．Wilson 95%CI は [0.92135, 0.95445] で上限がわずかに 0.95 を超える．すなわち「有意に過少被覆ではない」ことは強く言えるが，「有意に 0.95 以下」とまでは言えない．事前シミュレーション（10 分割で 0.9375〜0.9537，7/10 が帯内）が予告していた**分割由来のばらつきが SE と同程度以上**という状況が実測でもそのまま再現している．事前登録で seed=42 に固定し「帯外でも振り直さない」と宣言していたため，この 1 点で判定する．

**2. 実測値が事前シミュレーションと完全一致した — 実装は仕様どおり動いている**

計画節の表が予告した seed=42 の予測値は coverage=0.9400・mean_set_size≈5.521・q_hat=0.051766 であり，本実行の実測は coverage=0.940000・mean_set_size=5.52125・q_hat=0.051766 で**すべて一致**した．cross-fit も予測 0.9487 に対し実測 0.94875 と一致する．本レバーは `probabilities` を一切変えないため被覆が既存出力から厳密に再現計算できるという計画の前提が正しく，かつ実装が意図した分割・分位点・集合構成をそのまま実現していることが裏付けられた．Iter16/20/21/22/27 系列の「設定は変えたのにコードへ到達しない」型の失敗ではないことは，stderr の `n_cal=800 q_hat=0.0518` と，A 実行の md5 一致（現行経路を壊していない）の両方で二重に確認できている．

**3. 仮説は支持された — 真因は交換可能性の破れだった**

q_hat は Iter71 の 0.000980 から 0.051766 へ **52.8 倍**移動し，Iter71 計画節の掃引表が「帯に入る動作点」として示していた区間 0.04 ≤ q_hat ≤ 0.145 の内側へ着地した．coverage は 0.996875 → 0.940000（-5.69pt），mean_set_size は 9.66 → 5.52（-4.14 クラス）．計画の仮説「校正スコアと評価スコアが別の確率モデルから出ていることが過被覆の原因」は，予測した方向・予測した大きさの両方で一致した．

決定的なのは**名目水準を振ったときの応答**である．Iter71 では α を 0.30 まで上げても coverage=0.9875 で動かなかった（＝校正分布が評価分布と噛み合っておらず，α が被覆を制御できていなかった）．Iter72 では α=0.05/0.10/0.20/0.30 に対し coverage が 0.974/0.940/0.915/0.865 と**単調に応答する**．conformal の制御レバーとしての α が初めて機能した，というのがこの反復の実質的な成果である．

**4. 残る +4.0pt の過被覆は非ランダム化 APS の離散化で説明でき，交換可能性の残存破れではない**

名目 0.90 に対し実測 0.940（+4.0pt）だが，これを「まだ交換可能性が破れている」と読むのは誤りである．根拠は 2 つ．(a) 校正半自身での coverage も 0.94875 であり，評価半（0.94000）とほぼ同じ量だけ名目を上回る．交換可能性の破れなら両半で乖離が出るはずだが出ていない．(b) 過被覆量は α とともに拡大する（α=0.10 で +4.0pt，0.20 で +11.5pt，0.30 で +16.5pt）．これは「閾値をまたいだクラスを丸ごと集合へ入れる」という非ランダム化 APS の打ち切り規則から予想される振る舞いそのもので（調査 Q3，Romano et al. 2020 の U 項を持たない版），α が大きいほど 1 クラスあたりの確率質量が効いて過剰分が増える．したがって残差の帰属先は**集合構成の離散化**であり，校正集合の取り方ではない．

**5. 実用上の限界は被覆ではなく集合サイズに移った**

mean_set_size=5.52 は 10 ドメイン中 5.5 個を「可能性あり」と返すことを意味し，dispatch 先の絞り込みとしてはほぼ情報がない（無情報な全集合は 10，ランダム 5.5 個選択と同等の粒度）．set_size 分布も最頻値が 6 で，set_size ≤ 2 はわずか 44/800（5.5%）にとどまる．複合設問は 2 ドメインなので top-2 dispatch へ流用するには set_size ≈ 2 が要る．**被覆保証は得られたが，得られた集合は現状の運用目的には大きすぎる**．これが次の一手を規定する．

**6. 留保（結果の解釈に付記する事実）**

- n が 1,600 → 800 へ半減したため SE は 0.006 → 0.0084 に拡大している．Iter71 以前の coverage と同じ土俵で比較する際は注意が要る．ECE=0.076460 も母集団が異なるため Iter71 の 0.062998 との差（+0.013）を「悪化」と読んではならない（判定に用いない旨を事前登録済み）．
- `classifier_train.jsonl` と query が重複する 72 件は校正半 40・評価半 32 とほぼ均等で，どちらかの半に偏っていない（両半の交換可能性は壊れていない）．ただし「分類器が既見の行を含む」こと自体は解消していない．
- 評価半 800 行のうち複合設問は 46 行しかない．複合ドメインに関する結論をこの実験から導くことはできない（B104 A1 の検出力問題は未解消のまま）．

### 考察 (Iter72)

**判定: adopted**．主基準 coverage=0.940000 が事前登録の帯 0.88–0.95 に入り，非退行条件 1〜5 を全て充足した．`conformal_calibration_exchangeability` は単一値 `eval_holdout` を試し切ったのでレバーをクローズする．実装は **revert せず維持**する（既定 `oof_train` が Iter71 出力を md5 単位で再現する後方互換設計であり，以降のイテレーションは `eval_holdout` を固定値として使う）．

**この反復で確定した知見**

1. Iter56 以来 5 反復続いた conformal 系列の過被覆・過少被覆は，**分位点方向（Iter71）でも集合構成（Iter70）でも q_hat の母集団（Iter69）でもなく，校正スコアと評価スコアが別モデルから出ていたこと**が支配的な原因だった．Iter69〜71 の 3 反復は「式の細部」を順に潰したが，実際の効き幅（q_hat 52.8 倍）はこの 1 点が桁違いに大きい．**split conformal を実装したら，まず『校正スコアと評価スコアを生んだモデルが同一の固定モデルか』を確認する**のが最短経路である．この確認は 1 行の stderr（`n_cal` と `q_hat` の桁）で可能だった．
2. 「α を振っても coverage が動かない」は交換可能性の破れの**診断シグナル**として使える．Iter71 の付随報告（α=0.30 でも 0.9875）は当時「名目の選び方では説明できない」と記録されていたが，これを真因特定の手がかりとして読み切れていなかった．今後 conformal 系の実験では，本走の前に α 掃引曲線が単調応答するかを見ることで交換可能性の破れを安価に検出できる．
3. 評価データ自身を層化 50/50 に割る構成は正当だが（分類器は `dataset.jsonl` を学習に使っていない），**coverage の n が半減し分割由来のばらつきが二項 SE と同程度乗る**．本実験でも帯上限までの余裕は 1.19 SE しかなく，seed を引き直せば判定が反転しうる範囲だった．事前にシミュレーションで着地点と分割ばらつきを出し，seed を固定して事前登録するという Iter71 由来の手順が，結果の恣意的な選択を防いだ．
4. 新しい分岐を既存関数へ足すとき，**既存分岐でのみ初期化される局所変数**が `UnboundLocalError` の温床になる．既存テストは新分岐を通らないので検出できない．今後は分岐追加時にその関数内で定義される全ローカル変数の初期化位置を確認する．

**次の一手（停止条件 1 を適用）**

config の levers はこれで再び全て試行済みになるが，本反復の学び 5（集合サイズが実用上の限界に移った）から**次の有望なレバーを具体的に考案できる**ため，SKILL.md の停止条件 1 に従い新レバーを config へ追記して継続する．

- 新レバー **`conformal_set_size_reduction: [randomized_aps, raps_penalty]`**．Iter73 の単一レバーは **`randomized_aps`**．
- 根拠: 分析 4 で残る +4.0pt の過被覆は非ランダム化 APS の打ち切り規則に帰属することが（校正半・評価半の一致と α 依存性から）特定できている．Romano et al. (2020) の APS は打ち切り時のクラスを確率 U で含める／含めないと決めることでこの離散化を消し，被覆を名目へ厳密に一致させる．被覆が 0.94 → 0.90 へ下がる分だけ集合も小さくなるはずで，**単一の U 項の追加という最小変更**で確認できる．先に RAPS（サイズ正則化項 k_reg・λ）を試さない理由は，RAPS がハイパラ 2 個を持ち単一レバー原則の粒度として粗いためで，まず無償で得られる縮小分を取り切ってから残差に対して RAPS を当てる順序が正しい．
- Iter73 の想定成功条件（planner が確定する）: coverage が 0.88–0.95 の帯に留まったまま mean_set_size が 5.52 から有意に減少すること．非退行は本反復と同型（`selected_domain` 一致・`probabilities` 一致・既定値での md5 一致）．

**人間判断を要する事項（新規に確定させない）**

- conformal prediction を**実行時経路（`http_server.py` / `classifier.py`）へ配線するか**は B104 A2 として未回答のまま維持する．本反復で被覆保証は得られたが mean_set_size=5.52 では dispatch の絞り込みに使えないため，配線の是非は集合サイズ縮小（Iter73 以降）の結果を見てから諮るのが妥当である．今回新たに @mention はしない．
- B104 A1（複合評価集合 n=100 の検出力．評価半では 46 行）も未回答のまま維持する．

---

## Iteration 71: conformal予測のq_hat分位点方向を補数スコアのα分位点へ修正して被覆保証を検証

### 計画 (Iter71)

**仮説**

Iter70 で集合構成（`corrected_aps`）は正したが coverage=0.7638 と名目 0.90 を大きく下回った．
調査 (Iter71) Q1/Q2 のとおり，本実装の非適合スコアは `score = 1 - cumsum`（標準 APS スコアの**補数**）
であるため，q_hat は補数スコアの **α 分位点（下側，10th percentile）**で取らねばならないのに，現行は
**(1-α) 分位点（上側，90th percentile）**を取っている．この方向を α 側へ直せば，実効の累積確率閾値
`1 - q_hat` が 0.6135 から 0.95 前後へ引き上がり，coverage は名目 0.90 近傍へ到達するはずである．
本イテレーションの目的は Iter56 以来の conformal prediction 系列を，**実装が 2 つとも正しい状態での
確定的判定**へ置き換えることである（backlog B106）．

**単一レバー**

`conformal_qhat_quantile_direction`: `upper`（現行．補数スコアの (1-α) 分位点，
`target=min(1,(1-α)(1+1/n))` × `method="higher"`）→ `alpha_lower`（補数スコアの α 分位点，
`target=α(1+1/n)` × `method="lower"`．順位統計量としては `⌊(n+1)α⌋/n = 142/1427 ≈ 0.099510`）．
動かすのはこの分位点方向 1 点のみである．

**固定する構成（直近の最良構成に固定）**

- 集合構成: `--set-construction corrected_aps`（Iter70 で修正・維持と決定）．
- q_hat の母集団: `--qhat-source true_class`（`data/classifier_train.jsonl` の OOF 真クラス
  非適合スコア 1,427 件．Iter69 で確定）．
- `--confidence-level 0.90`（名目水準はレバーに**含めない**），評価データ `data/dataset.jsonl`（1,600 行），
  分類器 `models/domain_classifier.joblib`，埋め込み `nomic-embed-text:latest`（`127.0.0.1:11435`），
  `--education-logit-bias 0.0` / `--education-threshold 0.0`（既定），`--fine-tuned-embed-model` は
  指定しない（ollama 分岐を通す）．
- `config.yaml`・`http_server.py`・`classifier.py`・`aggregator.py`・`mise.toml` は変更しない．
- 入力データの同一性（計画時に確認済み）: `data/classifier_train.jsonl`（mtime 2026-07-30 14:44）・
  `data/dataset.jsonl`（2026-09-19 00:49）・`models/domain_classifier.joblib`（2026-08-02 23:41）は
  いずれも Iter70 本実行（2026-09-19 21:17）より前から変化していない．

**変更箇所（`scripts/evaluate_classifier_calibration.py` 1 ファイル＋テスト．行番号は 2026-09-19 計画時点）**

1. `_compute_prediction_set()`（L66-72 のシグネチャ）へ `qhat_quantile_direction: str = "upper"` を追加し，
   L111-116 の検証ブロックへ `if qhat_quantile_direction not in ("upper", "alpha_lower"): raise ValueError(...)`
   を既存 2 引数と同じ書き方で追加する．docstring に両方向の定義と「スコアが補数であるため α 側が
   正しい」理由（Angelopoulos & Bates 2021 / Barber et al. 2021 の `q̂-_{n,α}{v}=⌊(n+1)α⌋/n`）を記す．
2. q_hat 計算（L125-128）を分岐させる:
   - `"upper"`（既定・現行挙動を温存）: `target = min(1.0, (1.0 - alpha) * (1.0 + 1.0 / n))`,
     `np.quantile(flat_scores, target, method="higher")`．
   - `"alpha_lower"`: `target = alpha * (1.0 + 1.0 / n)`, `np.quantile(flat_scores, target, method="lower")`．
     （α<0.5 なので `min(1.0, ...)` のクランプは不要．上側実装と対称な `⌊(n+1)α⌋/n` 順位統計量に到達する．）
   分位点の水準と `method` 以外は変更しない（`np.quantile` の使い方自体は標準的な実装パターンであり
   作り替えない）．
3. 診断 print 用の重複ロジック（L272-273 `_diag_target` / `_diag_q_hat`）へ同じ分岐を適用し，
   L274-279 の print へ `qhat_quantile_direction={...}` を追加する（Iter70 で `set_construction` を
   print へ足した前例に倣う．発火証拠の恒久化）．**この重複箇所の更新漏れは診断だけが旧値を出す
   紛らわしい失敗になるため，実装時に L127-128 と L272-273 を必ず対で確認する．**
4. `predict_calibrated_rows()`（L157-170）・`_run()`（L369-383）のシグネチャへ引数を追加し，
   `_compute_prediction_set()` 呼び出し 2 箇所（L314-317 fine-tuned 分岐，L350-353 ollama 分岐）と
   `_run()` 内の `predict_calibrated_rows()` 呼び出し（L397 付近）へ伝播する．
5. CLI に `--qhat-quantile-direction`（`choices=["upper", "alpha_lower"]`, `default="upper"`）を追加し，
   **`main()` の `--output` 有無による 2 分岐の両方へ渡す**．
   - [ ] stdout 側: `if args.output is None:`（**L488**）配下の `_run(...)` 呼び出し
   - [ ] ファイル出力側: `with open(args.output, "w", ...)`（**L508**）配下の `_run(...)` 呼び出し
   本実験が通るのは**ファイル出力側**である．Iter69 実装フェーズでこの伝播漏れを実際に起こしており，
   Iter69・Iter70・Iter71 と 3 回連続で注意喚起されている箇所であるため，上のチェックボックス 2 つを
   実装完了時に明示的に確認し，さらに予備実行（下記手順 2）の stderr で実効値を目視確認する．
6. `tests/test_evaluate_classifier_calibration.py` へ既存の `test_qhat_source_*` / `test_set_construction_*`
   に倣い 4 件追加する:
   (a) 同一の `flat_scores` に対し `"upper"` と `"alpha_lower"` で q_hat が異なり，
       `q_hat(alpha_lower) < q_hat(upper)` であること，
   (b) 小さな合成配列で `"alpha_lower"` の q_hat が `⌊(n+1)α⌋/n` 順位統計量（ソート済み配列の
       `⌊(n+1)α⌋` 番目）と厳密一致すること，
   (c) 既定値 `"upper"` の回帰テスト（現行 q_hat と一致），
   (d) 未知の値で `ValueError`．

**到達コードパス**

CLI `--qhat-quantile-direction alpha_lower --set-construction corrected_aps --qhat-source true_class`
→ `main()` L508（ファイル出力分岐）→ `_run()` → `predict_calibrated_rows()` → cp_data 構築
（校正 1,427 行の OOF スコア，L230-260）→ 評価 1,600 行ループ → `_compute_prediction_set()` L350
（ollama 分岐．**分位点方向の切替が実際に効く唯一の地点**）→ 出力 jsonl の `prediction_set` /
`set_size` → coverage・mean_set_size 集計．`config.yaml` を経由しないため「デプロイ漏れ」型の失敗は
構造上起こらない．唯一のリスクは変更箇所 5 の CLI 2 分岐伝播漏れであり，予備実行で潰す．

**事前シミュレーション（計画時に実施．本実行の予測値）**

Iter70 本実行 B の出力 `results/20260919_211708/Iter70_corrected_aps.jsonl`（1,600 行の
`probabilities`）に対し，q_hat を掃引して `corrected_aps` 構成をオフライン適用した結果:

| q_hat | 実効閾値 `1-q_hat` | coverage | mean_set_size |
|---|---|---|---|
| 0.0200 | 0.9800 | 0.9775 | 7.08 |
| 0.0400 | 0.9600 | 0.9550 | 5.99 |
| 0.0522 | 0.9478 | 0.9444 | 5.52 |
| 0.0800 | 0.9200 | 0.9300 | 4.75 |
| 0.1000 | 0.9000 | 0.9181 | 4.34 |
| 0.1200 | 0.8800 | 0.9012 | 4.01 |
| 0.1500 | 0.8500 | 0.8781 | 3.58 |
| 0.3865（Iter70 実測） | 0.6135 | 0.7638 | 1.94 |

すなわち **coverage が帯 0.88–0.95 に収まる q_hat の範囲は概ね 0.04 ≤ q_hat ≤ 0.145** である．
問題は「校正集合（1,427 件 OOF）の真クラス補数スコアの α 分位点」が実際にどこへ落ちるかであり，
これは埋め込みの再計算が必要なため計画時には確定できない．参考として**評価集合側**の真クラス補数
スコア（Iter70 出力から事後計算，n=1600）の分位点は `⌊(n+1)α⌋/n` 位置で **0.05225**，
上側 (1-α) 位置で 0.59568 である．一方 Iter69/70 実測の**校正集合**の上側 q_hat は 0.3865 であり，
校正集合のスコア分布は評価集合より低い側に寄っている（0.3865 / 0.5957 ≈ 0.65）．同じ比で縮むと
仮定すると校正集合の α 分位点は **0.03 前後**と見積もられ，その場合 coverage ≈ 0.96・
mean_set_size ≈ 6.3 となり **帯の上限 0.95 を超える（過被覆）可能性がある**．
したがって本実行の事前予測は「**coverage は 0.93〜0.97，mean_set_size は 5〜7**」であり，
帯の下限は確実に超えるが上限で外れる目があることを，実測前に明記しておく．

**成功条件（事前登録．config.yml `conformal_qhat_quantile_direction` note の案をそのまま採用）**

名目信頼水準 `--confidence-level 0.90`，評価 1,600 行．比較対象は
`results/20260919_211708/Iter70_corrected_aps.jsonl`（md5=`ee6fbf7d0e4dfd9408c76b2ff9fd1cdf`）．

| 指標 | 定義 | 基準線（Iter70 実測） | 合格条件 |
|---|---|---|---|
| coverage（主基準） | `mean(expected_domains[0] in prediction_set)` | 0.7638 | **0.88 ≤ coverage ≤ 0.95** |
| mean_set_size（副基準） | `mean(set_size)` | 1.9438 | 報告のみ（判定に用いない） |
| ECE（同一性アンカー） | `metrics.py:compute_ece()` | 0.062998 | **ECE ≤ 0.0680** |

- **adopted**: coverage が 0.88–0.95 に入り，かつ ECE ≤ 0.0680 と下記非退行条件 1〜4 を全て満たすこと．
- **rejected**: coverage が帯外であること．ただし**外れ方の向きで解釈を分ける**（これも事前登録する）:
  - `coverage < 0.88`: 分位点方向の修正でも過小被覆が残る．実装または校正手続きに第 3 の欠陥がある
    可能性として扱い，本系列を閉じる前に原因を journal に記録する．
  - `coverage > 0.95`: **過被覆**．分位点方向の修正自体は機能している（0.7638 → 0.95 超は
    名目 0.90 を跨ぐ大幅な移動）が，校正集合（OOF スコア）と評価集合のスコア分布のずれにより
    q_hat が小さすぎる，という**別の原因**に切り分けられる．この場合は本レバーを rejected とし，
    次レバー候補として「校正集合の交換可能性（OOF vs ホールドアウト）」を backlog へ起票する．
    **判定基準そのものは緩めない**（事後の帯の拡大はしない）．
- **invalid**（判定保留）: 予備実行で `qhat_quantile_direction` の実効値 print が `alpha_lower` に
  ならない，q_hat が Iter70 の 0.3865 から変化しない，または `set_size` 分布が Iter70 と同一である場合．
  いずれも分岐未到達を意味するため，本実行に進まず実装をやり直す．
- **ノイズ幅**: パイプラインは `StratifiedKFold(random_state=42)` で決定的であり，変動源は ollama
  埋め込みの数値再現性のみ．coverage は n=1600・p≈0.94 の二項標本誤差で SE≈0.006，実質的な差は
  ±1.2pt 超とする．帯端からの逸脱は SE の何倍かを併記して判定する．

**非退行条件**（比較対象は `results/20260919_211708/Iter70_corrected_aps.jsonl`）

1. `selected_domain` が全 1,600 行で一致（不一致 0 件）．q_hat は argmax に影響しないため，
   不一致があれば実装ミスである．
2. `confidence` および `probabilities` が一致（許容差 1e-9．埋め込み再計算由来の微差が出た場合は
   その最大値を journal に記録する）．
3. `top1_accuracy` が 0.603125 から不変（条件 1 の系）．
4. 既定値 `--qhat-quantile-direction upper`（＋ `corrected_aps` / `true_class`）での再実行が Iter70 の
   `Iter70_corrected_aps.jsonl` と **md5=`ee6fbf7d0e4dfd9408c76b2ff9fd1cdf`** で一致すること（後方互換）．
5. 予測集合の rank_2 候補源への流用は本イテレーションでは行わない（スコープ外）．

**実行手順**（オフライン完結．実機 10 ノードの LLM 生成・dispatch・probe は一切行わない．
埋め込みのみ `127.0.0.1:11435` の ollama を使う）

1. **実装**: 上記 1〜6 を実施し，`uv run pytest tests/test_evaluate_classifier_calibration.py` と
   既存テスト・lint・型検査を通す．CLI 2 分岐のチェックボックスを確認する．
2. **予備実行（発火確認・実測 q_hat の取得）**: `data/dataset.jsonl` の先頭 20 行を
   `/tmp/iter71_head20.jsonl` に切り出し，`--qhat-source true_class --set-construction corrected_aps`
   固定で `--qhat-quantile-direction` を `upper` と `alpha_lower` の 2 通り実行する（校正 1,427 件の
   埋め込みは両方で必要なため 1 回あたり十数分）．
   - 期待: stderr の print が `qhat_quantile_direction=upper q_hat=0.3865` と
     `qhat_quantile_direction=alpha_lower q_hat=<新値>` をそれぞれ示し，母集団サイズは両方 1427，
     `set_size` の分布が後者で明確に大きくなる．
   - **ここで得た実測 q_hat を上の掃引表に当てはめ，本実行の coverage 予測値を journal に書き留めて
     から手順 4 へ進む**（事前予測と実測の突き合わせを判定の一部とするため）．
   - print が `upper` のままである・q_hat が変化しない場合は CLI 伝播漏れ（変更箇所 5）を疑い，
     本実行に進まず実装をやり直す．
3. **本実行 A（後方互換の確認）**: 全 1,600 行を `--qhat-quantile-direction upper --set-construction
   corrected_aps --qhat-source true_class` で実行し，md5 が `ee6fbf7d0e4dfd9408c76b2ff9fd1cdf` と
   一致することを確認する．不一致なら原因（ollama バージョン・digest 差）を特定し journal へ記録してから進む．
4. **本実行 B（レバー）**: 同一コマンドの `--qhat-quantile-direction alpha_lower` のみを変えて実行する．
5. **分析**: B の出力から coverage・mean_set_size・`set_size` 分布（1〜10 のヒストグラム）を集計し，
   ECE は `metrics.py:compute_ece()` を流用する．手順 2 で立てた予測値との一致を確認し，
   Iter70 出力との突き合わせで非退行条件 1〜4 を検証したうえで adopted / rejected / invalid を判定する．
6. **付随報告（判定には用いない）**: `--confidence-level` を 0.70 / 0.80 / 0.95 に振ったときの
   coverage・mean_set_size を B の `probabilities` からオフライン再計算し（埋め込み再実行は不要），
   「どの名目水準なら実用的な集合サイズに収まるか」を曲線として記録する．次レバーの材料とする．

- **コスト**: 埋め込み 1,427（校正）＋1,600（評価）件の逐次計算で本実行 1 回あたり 10〜30 分．
  予備実行 2 回を含め計 1〜1.5 時間程度．GPU の実機占有・LLM 生成は不要．
- **出力先**: `results/<timestamp>/Iter71_qhat_upper.jsonl`，`results/<timestamp>/Iter71_qhat_alpha_lower.jsonl`．

---

### 調査 (Iter71)

**前提確認**: `state.json`（iteration=71, phase=investigate, current_lever=conformal_qhat_quantile_direction）・
`config.yml` の `conformal_qhat_quantile_direction: [alpha_lower_quantile]`（Iter70 reflector 新設 /
backlog B106）・backlog B106・journal Iter70 の「調査 (Iter70)」Q1〜Q3 を確認した．単一レバーは
分位点方向 1 点のみ，集合構成は `corrected_aps` に固定（Iter70 で維持），q_hat の母集団は
`true_class`（校正集合 1,427 件）に固定，分類器・埋め込み・argmax・confidence・`config.yaml` は変更しない．

**Q1: 現行コードの q_hat 算出ロジックを読解し，config note の主張（分位点方向が逆）を数式で検証**

`scripts/evaluate_classifier_calibration.py`（2026-09-19 Iter71 調査時点の行番号）:

- `_compute_prediction_set()` 内 L127-128:
  ```python
  target = min(1.0, (1.0 - alpha) * (1.0 + 1.0 / len(flat_scores)))
  q_hat = float(np.quantile(flat_scores, target, method="higher"))
  ```
  `alpha = 1.0 - confidence_level`（confidence_level=0.90 なら alpha=0.10）．`target ≈ 0.9006`
  （n=1427 のとき），すなわち**補数スコア（`score = 1 - cumsum`）の (1-α) 分位点＝90th percentile**
  を q_hat としている．同一ロジックが診断 print 用に L272-273 にも重複している（`_diag_target` /
  `_diag_q_hat`）ため，修正はこの 2 箇所の両方に対して行う必要がある．
- 集合構成（`corrected_aps`，Iter70 で修正済み・L134-140）は「確率降順に走査しながら
  `cumsum` を加算 → 先に `pred_set.append` → `score = 1 - cumsum` が `q_hat` 以下になった回で
  `break`」という順序．すなわち**打ち切り条件は `1 - cumsum <= q_hat` ⟺ `cumsum >= 1 - q_hat`**であり，
  これは「累積確率が `1 - q_hat` という質量に達するまでクラスを追加する」という標準 APS の構成と
  数式的に同じ形をしている．
- 問題は「この `1 - q_hat` が正しい被覆質量（目標 `1 - α = 0.90`）に対応する値になっているか」である．
  標準 APS では，校正セットの**標準スコア**（`S = cumsum`，真クラスまでの累積確率，単調増加）の
  `(1-α)` 分位点 `q_hat_std` を求め，集合構成の打ち切り条件は `cumsum >= q_hat_std` である．
  本実装のスコアは `score = 1 - cumsum`（標準スコアの補数）なので，`q_hat_std = 1 - q_hat_complement`
  が成り立つべきだが，これは「補数スコアの分位点」と「標準スコアの分位点」が**単調減少変換で
  写り合う**ことを意味する．一般に `Y = 1 - X` のとき，`X` の `p` 分位点 `x_p` に対応する `Y` の
  分位点は `1 - x_p`（＝`Y` の `1-p` 分位点）である．すなわち `q_hat_std`（`X` の `(1-α)` 分位点）に
  対応する `q_hat_complement`（`Y=1-X` 側の分位点）は，`X` の `(1-α)` 分位点の位置を `Y` 側に
  写した **`Y` の `α` 分位点**でなければならない．
  **現行コードは `Y`（補数スコア）の `(1-α)` 分位点（90th percentile）を取っており，
  正しくは `α` 分位点（10th percentile）を取るべきである**．config note（backlog B106）の
  主張と一致する．
- 直感的な確認: `q_hat` が大きいほど打ち切り条件 `cumsum >= 1 - q_hat` の右辺（必要な累積確率）は
  **小さく**なり，集合は小さく・被覆は低くなる．現行実装は補数スコアの 90th percentile
  （＝相対的に**大きい**値）を q_hat に採用しているため，必要な累積質量 `1 - q_hat` が過小になり，
  構造的に過小被覆（under-coverage）を招く．Iter70 実測の coverage=0.7638（目標 0.87-0.95 未達）は
  この機序と整合する．

**Q2: 標準 conformal prediction（split conformal / APS）における q_hat の分位点方向と
有限標本補正式を一次資料で確認**（tavily-search）

- Angelopoulos & Bates, "A Gentle Introduction to Conformal Prediction and Distribution-Free
  Uncertainty Quantification" (arXiv:2107.07511, 2021・NeurIPS 2020 の Romano et al. APS 論文の
  標準的な解説として広く引用される)．同論文の手順（`arxiv.org/html/2107.07511v6` より直接引用）:
  「Compute `q̂` as the `⌈(n+1)(1-α)⌉/n` quantile of the calibration scores」．
  これは非適合スコア `S`（大きいほど不適合＝標準スコアと同じ向き）に対する**上側**分位点
  （`(1-α)` 側）であることを確認した．
- `ConformalPrediction.jl` の公式ドキュメント（`taija.org/ConformalPrediction.jl/dev/explanation/
  finite_sample_correction`）は Angelopoulos & Bates (2021) と Barber et al. (2021,
  "Predictive Inference with the Jackknife+", Annals of Statistics) の記法を引用し，**上側分位点と
  下側分位点の両方を式で定義**している:
  ```
  q̂+_{n,α}{v} = ⌈(n+1)(1-α)⌉ / n            （上側，標準スコアに対して使う分位点）
  q̂-_{n,α}{v} = ⌊(n+1)α⌋ / n = -q̂+_{n,α}{-v}  （下側，符号反転したスコアに対して使う分位点）
  ```
  この第 2 式 `q̂-_{n,α}{v} = -q̂+_{n,α}{-v}` は，まさに Q1 で導出した「`Y=1-X` の分位点は `X` の
  分位点の写像」という関係の一般形であり，**`α` 分位点（下側）を `⌊(n+1)α⌋/n` の位置で取る**という
  補正式が一次資料で裏付けられた．
- 補足（medium.com の実装解説記事）は現行コードと同型の実装トリック
  （`q_level = ceil((n+1)*(1-alpha))/n; q_hat = np.quantile(scores, q_level, method='higher')`）を
  示しており，本リポジトリの `_compute_prediction_set()` の書き方（`np.quantile(..., method="higher")`）
  が標準的な実装パターンに沿っていることも確認した．したがって修正すべきは**分位点の水準と
  補間方向（`method`）のみ**であり，`np.quantile` の使い方自体を作り替える必要はない．

**Q3: n=1,427 に対する正しい有限標本補正式の確定**

校正集合サイズ `n_cal = 1427`（`data/classifier_train.jsonl`，Iter69 で確定．journal Iter69/70 に
既出）．`confidence_level=0.90` ⟹ `alpha=0.10`．

- **現行（upper，(1-α) 側）**: `target = (1-α)(1+1/n) ≈ 0.9 × 1.0007009 ≈ 0.900631`，
  `np.quantile(scores, target, method="higher")`．これは `⌈(n+1)(1-α)⌉/n = ⌈1428×0.9⌉/n
  = ⌈1285.2⌉/n = 1286/1427 ≈ 0.901192` の近似実装（`method="higher"` の丸め上げでほぼ同じ
  順位统計量に到達する設計）．
- **修正後（`alpha_lower_quantile`，α 側）**: `⌊(n+1)α⌋/n = ⌊1428×0.10⌋/n = ⌊142.8⌋/n
  = 142/1427 ≈ 0.099510`．実装は現行の `method="higher"` を上側専用の丸めトリックとして使っているのに
  対応させ，下側は `target = α(1+1/n) ≈ 0.10 × 1.0007009 ≈ 0.100070`，
  `np.quantile(scores, target, method="lower")`（丸め下げ）とすることで，上側実装と対称な
  `⌊(n+1)α⌋/n` 順位統計量に到達する設計が，一次資料の式・現行コードの実装パターンの両方と整合する．
  **`min(1.0, ...)` のクランプは下側では不要**（`α(1+1/n)` は α<0.5 なら 1.0 を超えない．
  ただし `max(0.0, ...)` は理論上不要だが防御的に残してもよい）．
- **具体値の実測は本イテレーションでは行っていない**（校正集合 1,427 件の埋め込み・OOF 予測の
  再計算が必要で，これは実装・実験フェーズのコストに属する）．config note が挙げる
  「評価集合の補数スコア q0.10=0.0524 / q0.90=0.5956」は**評価集合（1,600 行）の argmax
  確率から事後計算した補数スコアの分位点**であり，**校正集合（1,427 行，真クラススコアのみ）から
  計算される実際の q_hat とは異なる母集団**である点に注意．両者を混同すると q_hat の値を
  誤って見積もるため，計画・実装フェーズでは校正集合の `true_class_scores` から直接
  `np.quantile(scores, 0.100070, method="lower")` を計算し，実測値を journal に記録すること．

**Q4: レバーを実際に発火させるための到達コードパスと変更箇所**

`grep -n "conformal_qhat_quantile_direction"` は `.claude/research/config.yml` と本 journal 以外に
0 件（`config.yaml`・`http_server.py`・`classifier.py`・`aggregator.py`・`mise.toml` のいずれにも
未出現）．Iter69/70 と同型の CLI 直接起動オフライン評価であり，`config.yaml` を経由しない．

変更が必要な箇所（すべて `scripts/evaluate_classifier_calibration.py`，Iter71 調査時点の行番号）:

1. `_compute_prediction_set()` のシグネチャ（現 L66-72）に新引数
   （例: `qhat_quantile_direction: str = "upper"`）を追加し，L127-128 の分位点計算を分岐させる:
   - `"upper"`（既定・現行挙動）: `target = min(1.0, (1-alpha)*(1+1/n))`, `method="higher"`．
   - `"alpha_lower"`（新値）: `target = alpha*(1+1/n)`, `method="lower"`．
   未知の値は `ValueError`（既存の `qhat_source`・`set_construction` の検証パターン，
   L111-116 相当，に揃える）．
2. 診断 print 用の重複ロジック（L272-273 `_diag_target` / `_diag_q_hat`）にも同じ分岐を適用する
   （Iter70 で `set_construction` を print へ追加した前例に倣い，`qhat_quantile_direction` の
   実効値と結果 q_hat も print へ追加し，発火の証拠を残す）．
3. `predict_calibrated_rows()`（L157-169 のシグネチャ）・`_run()`（L369-383 のシグネチャ）へ
   引数を追加し，`_compute_prediction_set()` 呼び出し 2 箇所（L313 fine-tuned embedding 分岐，
   L349 ollama 分岐）へ伝播する．**本実験で通るのは ollama 分岐（L349 相当）**．
4. CLI に `--qhat-quantile-direction`（`choices=["upper", "alpha_lower"]`, `default="upper"`）を
   追加し，`main()` の **`--output` 有無による 2 分岐（L488 `if args.output is None:` の stdout 側，
   L508 `with open(args.output, ...)` のファイル出力側）の両方**へ渡す．
   Iter69 実装フェーズでこの伝播漏れを一度起こしており（journal Iter69/70），Iter70 計画節も
   同じ注意を明記している．**本イテレーションでも同じ落とし穴が存在する**ため，計画フェーズは
   この 2 箇所を明示的にチェックリスト化すること．実験で使うのはファイル出力側（L508 相当）．
5. `tests/test_evaluate_classifier_calibration.py` に，既存の `qhat_source`/`set_construction` の
   テストパターン（`test_qhat_source_*`, `test_set_construction_*`）に倣い，
   (a) 同一の `flat_scores` に対し `"upper"` と `"alpha_lower"` で異なる q_hat が出ること，
   (b) `"alpha_lower"` の q_hat が `⌊(n+1)α⌋/n` 順位統計量と一致すること（小さな合成配列で検証），
   (c) 既定値 `"upper"` の回帰テスト，(d) 未知の値で `ValueError`，を追加する．

**到達コードパス（まとめ）**: CLI `--qhat-quantile-direction alpha_lower --set-construction
corrected_aps --qhat-source true_class` → `main()` ファイル出力分岐（L508 相当）→ `_run()` →
`predict_calibrated_rows()` → 評価 1,600 行ループ → `_compute_prediction_set()`（ollama 分岐，
L349 相当，**分位点方向の切替が実際に効く唯一の地点**）→ 出力 jsonl の `prediction_set` /
`set_size` → coverage・mean_set_size 集計．`config.yaml` を経由しないため「デプロイ漏れで実行時に
読まれない」型の失敗は構造上起こらない．唯一のリスクは Q4-4 の CLI 2 分岐伝播漏れであり，
Iter69/70 と同様に予備実行（先頭 20 行）で `set_construction` と `qhat_quantile_direction` の
実効値 print・q_hat の値・`set_size` 分布の変化を必ず確認すること．

**次フェーズへの示唆**

- 単一レバーは分位点方向 1 点のみ．集合構成 `corrected_aps`・母集団 `true_class` は固定のまま，
  `--qhat-quantile-direction alpha_lower` を追加する設計で config note の意図と整合する．
- 有限標本補正式は `⌊(n+1)α⌋/n`（n=1427, α=0.10 のとき 142/1427≈0.09951）で確定．
  実装は `target = alpha*(1+1/n)` と `np.quantile(..., method="lower")` の組で，現行の上側実装と
  対称な形にできる．
- 実測 q_hat（校正集合の `true_class_scores` に対する `alpha_lower` 分位点）は計画・実装フェーズで
  必ず算出し，config note の楽観的見積り（q_hat≈0.05）と一致するか確認すること。評価集合の分位点
  （q0.10=0.0524等）は校正集合とは異なる母集団であり参考値に留まる。
- CLI の 2 分岐（stdout / ファイル出力）への伝播が Iter69 以来 3 回連続で注意喚起されている箇所
  であり，予備実行での発火確認（stderr 診断 print の実効値・set_size 分布の変化）を計画に明記すること。
- 成功条件は config note 記載のとおり（coverage 0.88-0.95，ECE≤0.0680 同一性アンカー，
  `selected_domain`/`confidence`/`probabilities` の Iter70 出力との一致）を踏襲すればよく，
  本調査で新たな懸念は見つからなかった。

---

### 実装 (Iter71)

**変更ファイル**: `scripts/evaluate_classifier_calibration.py`（1 ファイル）＋
`tests/test_evaluate_classifier_calibration.py`（テスト追加）．計画どおり `config.yaml` は変更していない．

**変更内容（行番号は実装後）**:

1. `_compute_prediction_set()` シグネチャ（L66-73）に `qhat_quantile_direction: str = "upper"` を追加．
   docstring（L110-127）へ `q-hat+_{n,alpha}` / `q-hat-_{n,alpha}` の定義（Angelopoulos & Bates 2021
   arXiv:2107.07511，Barber et al. 2021 Annals of Statistics）と，本実装のスコアが標準 APS スコアの
   補数であるため `alpha_lower` が正しい理由を記載．検証ブロック（L137-141）へ
   `qhat_source`/`set_construction` と同型の `ValueError` 分岐を追加．
2. q_hat 計算（L150-158）を分岐化: `"alpha_lower"` は `target = alpha*(1+1/n)` を
   `np.quantile(flat_scores, target, method="lower")` で評価，`"upper"`（既定）は現行式のまま温存．
3. 診断 print 用の重複ロジック（旧 L272-273 相当，現 `predict_calibrated_rows()` 内）にも同一分岐を適用し，
   print 文へ `qhat_quantile_direction={...}` を追加（L296-314 付近）．計画で指摘された「対で確認」を実施済み．
4. `predict_calibrated_rows()`・`_run()` のシグネチャへ引数を追加し，`_compute_prediction_set()` 呼び出し
   2 箇所（fine-tuned 分岐・ollama 分岐）と `_run()` 内 `predict_calibrated_rows()` 呼び出しへ伝播．
5. CLI に `--qhat-quantile-direction`（`choices=["upper","alpha_lower"]`, `default="upper"`）を追加し，
   `main()` の `--output` 有無 2 分岐（stdout 側 L529-548，ファイル出力側 L550-568 相当）**両方**へ
   `qhat_quantile_direction=args.qhat_quantile_direction` を渡した．Edit の `replace_all` は
   インデント差（16 スペース vs 20 スペース）のため 1 回で両方には反映されず，ファイル出力側は個別の
   Edit で追加漏れがないことを確認した（計画で警告されていた 3 回連続の伝播漏れパターンを本イテレーションで再現しかけたが，
   実装中に検知・修正済み）．
6. `tests/test_evaluate_classifier_calibration.py` へ 4 件追加:
   (a) `test_qhat_quantile_direction_upper_and_alpha_lower_yield_different_q_hat`: 既存 10 クラス
       toy データ（n_cal=12）で `upper`（q_hat=0.95, set_size=1）と `alpha_lower`（q_hat=0.60,
       set_size=2）が分岐し，`alpha_lower` の方が大きい集合になることを検証．
   (b) `test_qhat_quantile_direction_alpha_lower_matches_finite_sample_rank_statistic`: n=19 の等間隔
       合成スコア配列で `⌊(n+1)α⌋/n`（floor(20*0.1)=2 番目＝0.10）が厳密に一致することを
       `np.quantile` 直接呼び出しで確認したうえで，浮動小数点境界を避けるため余裕を持たせた
       確率配列（score 0.15→0.05 で交差）で `_compute_prediction_set()` 経由の集合サイズが
       一致することを検証．
   (c) `test_qhat_quantile_direction_default_is_upper_for_backward_compatibility`: 既定値回帰テスト．
   (d) `test_qhat_quantile_direction_rejects_unknown_value`: 未知値で `ValueError`．

**テスト結果**: `uv run pytest tests/test_evaluate_classifier_calibration.py -v` で全 13 件
（既存 9 件＋新規 4 件）PASS．`uv run ruff check scripts/evaluate_classifier_calibration.py
tests/test_evaluate_classifier_calibration.py` は "All checks passed"．
`uv run pytest -q`（全体）は 275 passed / 12 failed だが，failed はいずれも
`tests/test_build_dataset.py`・`tests/test_train_domain_classifier.py`（`scripts/train_domain_classifier.py`
の `AttributeError`）であり，`git stash` で本イテレーションの変更を退避した状態でも同じ 12 件が
失敗することを確認済み（本変更前から存在する既存不具合であり非退行）．型検査ツール（mypy 等）は
`pyproject.toml` に設定がなく，リポジトリに lint/type task が mise に無いため実行していない
（ruff のみ実行）．

**予備実行（発火確認）**: `data/dataset.jsonl` 先頭 20 行を切り出し，`--qhat-source true_class
--set-construction corrected_aps` 固定で `--qhat-quantile-direction` を `upper`／`alpha_lower` の
2 通り実行（`--ollama-host 127.0.0.1 --ollama-port 11435`，wafl-ctrl5 の ollama 経由，実機ノード不使用）．

- `upper`: stderr `qhat_source=true_class q_hat=0.3865 population_size=1427
  set_construction=corrected_aps qhat_quantile_direction=upper`．q_hat=0.3865 は Iter69/70 実測値と
  一致（後方互換の傍証）．`set_size` 分布（20 行）: `[2,3,4,2,4,2,4,2,2,2,3,4,2,3,3,4,2,3,2,2]`．
- `alpha_lower`: stderr `qhat_source=true_class q_hat=0.0010 population_size=1427
  set_construction=corrected_aps qhat_quantile_direction=alpha_lower`．`set_size` 分布（20 行）:
  `[9,10,10,10,10,10,10,10,10,10,10,10,9,10,10,10,10,10,9,9]`．
- 判定: `population_size` は両方とも 1427 で一致，`qhat_quantile_direction` の実効値表示が
  それぞれのモードで正しく切り替わり，q_hat（0.3865→0.0010）・`set_size` 分布（2-4 個→9-10 個）が
  明確に分岐している．計画の invalid 条件（q_hat 不変・set_size 分布が Iter70 と同一）には該当せず，
  実装は発火していると判断する。
- **実測 q_hat の計画予測との乖離を明記**: 計画の事前見積り（校正集合の α 分位点 ≈0.03 前後，
  掃引表の 0.04〜0.145 帯）に対し，実測 q_hat=0.0010 は 1 桁以上小さい．校正集合（1,427 件の OOF
  真クラス補数スコア）は，計画時の「評価集合との比 0.65 で縮小」という粗い外挿より大きく 0 側に
  偏っている（多くの校正サンプルで分類器が真クラスにほぼ確信的＝`cumsum` がほぼ 1 に近く，
  `score=1-cumsum` がほぼ 0 に近いサンプルが 10% 分位点を大きく下回るほど存在する，という解釈）．
  この値自体は本フェーズで実装ミスによるものではない（`population_size=1427` の一致，ユニットテスト
  (a)(b) での手計算との整合，`upper` 側が Iter70 実測と完全一致することから，計算経路は正しいと判断）．
  ただし本実行 B（1,600 行）で `1-q_hat≈0.999` の質量を要求すると，予備実行の `set_size` 分布
  （9〜10/10 クラス）から推定して **coverage は 0.95 の合格上限を大きく超える可能性が高い**
  （over-coverage，10 クラス中 9〜10 個を毎回集合に含める状態）．この見立ては判定には用いず，
  次の実験フェーズで 1,600 行の本実行により確定させる．

**実験を開始してよい状態か**: はい．コード・テストとも計画どおり実装済み，回帰なし，CLI 2 分岐の
伝播も確認済み．ただし上記のとおり予備実行の実測 q_hat は計画の想定レンジ外であり，本実行 B は
「rejected（coverage>0.95，過被覆）」に着地する可能性が高いことを実験・分析フェーズへ申し送る．

---

### 実験・分析(実行) (Iter71)

**実行環境**: オフライン完結．実機ノード wafl500〜509 は不使用．埋め込み計算のみ
`127.0.0.1:11435`（SSH ローカルフォワード先，wafl-ctrl5 の ollama，`nomic-embed-text:latest` 在中）を
使用．LLM 生成・probe・dispatch トラフィックは発生していない．入力（`data/dataset.jsonl`
mtime 2026-09-19 00:49，`data/classifier_train.jsonl` mtime 2026-07-30 14:44，
`models/domain_classifier.joblib` mtime 2026-08-02 23:41）はいずれも計画・実装フェーズの記録と一致し，
実行直前に再確認した（変化なし）．

**実行コマンド**（A・B とも `--qhat-quantile-direction` のみ変更，他は計画節固定パラメータのとおり）:

```
uv run python -m scripts.evaluate_classifier_calibration \
  --dataset data/dataset.jsonl \
  --classifier models/domain_classifier.joblib \
  --embedding-model nomic-embed-text \
  --ollama-host 127.0.0.1 --ollama-port 11435 \
  --conformal-prediction --confidence-level 0.90 \
  --calibration-dataset data/classifier_train.jsonl \
  --qhat-source true_class \
  --set-construction corrected_aps \
  --qhat-quantile-direction [upper|alpha_lower] \
  --output results/20260919_215923/Iter71_qhat_[upper|alpha_lower].jsonl
```

`--education-logit-bias`/`--education-threshold` は既定値 0.0，`--fine-tuned-embed-model` は指定なし
（ollama 分岐）で，計画節の固定パラメータと一致する．

**本実行 A（`--qhat-quantile-direction upper`，後方互換確認）**

- 出力: `results/20260919_215923/Iter71_qhat_upper.jsonl`（1,600 行）
- stderr 診断: `qhat_source=true_class q_hat=0.3865 population_size=1427
  set_construction=corrected_aps qhat_quantile_direction=upper`
- md5=`ee6fbf7d0e4dfd9408c76b2ff9fd1cdf`。比較対象
  `results/20260919_211708/Iter70_corrected_aps.jsonl`（同 md5）と**バイト単位で完全一致**した。
  手順 1（後方互換確認）は成功。原因調査は不要だった。

**本実行 B（`--qhat-quantile-direction alpha_lower`，レバー）**

- 出力: `results/20260919_215923/Iter71_qhat_alpha_lower.jsonl`（1,600 行，md5=`575594c7a973fe615202485de7fe0a8f`）
- stderr 診断: `qhat_source=true_class q_hat=0.0010 population_size=1427
  set_construction=corrected_aps qhat_quantile_direction=alpha_lower`。予備実行（先頭 20 行）の
  実測値 q_hat=0.0010 と一致し，1,600 行の本実行でも変化しなかった（母集団は校正集合 1,427 件で
  評価データ数に依存しないため，これは想定どおり）。

**実測値**（`metrics.py:compute_ece()` を流用。coverage の定義は
`expected_domains[0] in prediction_set` の平均値。判定は行わず数値のみ記録する。
分析コード: `/tmp/iter71_analyze.py`，作業用の一時ファイルでリポジトリには含めていない）:

| 指標 | A（upper） | B（alpha_lower） |
|---|---|---|
| coverage | 0.763750（1222/1600） | **0.996875（1595/1600）** |
| mean_set_size | 1.943750 | 9.663750 |
| set_size ヒストグラム（1〜10） | {1:539, 2:671, 3:332, 4:57, 5:1, 6:0, 7:0, 8:0, 9:0, 10:0} | {1:0, 2:0, 3:0, 4:0, 5:0, 6:5, 7:23, 8:65, 9:319, 10:1188} |
| ECE | 0.062998 | 0.062998 |
| top1_accuracy | 0.603125 | 0.603125 |

A の数値は Iter70 の B（`corrected_aps`, `upper`）実測値（coverage=0.763750,
mean_set_size=1.943750, ECE=0.062998）と完全一致し，md5 一致（本実行 A）とあわせて整合的である。
ECE・top1_accuracy が A・B で不変なのは，`qhat_quantile_direction` が `confidence`/`selected_domain`
（argmax）に影響しない設計どおりである。

**非退行条件の検証**（比較対象: `results/20260919_211708/Iter70_corrected_aps.jsonl`，
1,600 行を `id` で突き合わせ）:

1. `selected_domain` 不一致件数 = **0**。
2. `confidence` 不一致件数（許容差 1e-9 超）= **0**，実測最大差 = **0.000e+00**。
3. `probabilities`（10 クラス×1,600 行 = 16,000 要素）不一致件数（許容差 1e-9 超）= **0**，
   実測最大差 = **0.000e+00**。
4. `top1_accuracy` は A・B とも **0.603125** で不変（条件 1 の系）。
5. 既定値 `--qhat-quantile-direction upper`（本実行 A）は Iter70 出力（`--set-construction
   corrected_aps` 側）と **md5 完全一致**（`ee6fbf7d0e4dfd9408c76b2ff9fd1cdf`）。

非退行条件 1〜5 はすべて満たされた。分位点方向の切替（B）が `probabilities`・`confidence`・
`selected_domain`・`top1_accuracy` に一切影響しないことが実測でも確認された。

**手順 2（予備実行）で立てた予測との突き合わせ**: 予備実行時点で「coverage は 0.95 の合格上限を
大きく超える可能性が高い」と申し送っていたとおり，本実行 B の実測 coverage=0.996875 は帯上限 0.95 を
大きく超えた（過被覆）。計画の事前シミュレーション表（q_hat 掃引，0.04≤q_hat≤0.145 で coverage
0.88–0.95）とは異なり，実測 q_hat=0.0010 は掃引表の最小値 0.0200 よりさらに 1 桁小さく，
対応する `1-q_hat≈0.999` という極端に高い累積質量要求により，mean_set_size が 9.66/10（ほぼ全クラスを
含む集合）まで拡大した。

**付随報告（判定には用いない）: 名目水準振り**

計画手順 6 のとおり，本実行 B の `probabilities`（評価データ側，再計算不要）はそのまま流用し，
校正集合（1,427 件）の true-class 補数スコア配列のみを 1 回計算（`/tmp/iter71_qhat_sweep.py`，
評価データ側の埋め込み・分類器推論の再実行はなし）して，`--confidence-level` を 0.70/0.80/0.90/0.95
に振ったときの `alpha_lower` 方向 q_hat・coverage・mean_set_size をオフライン再計算した：

| confidence_level | alpha | q_hat(alpha_lower) | coverage | mean_set_size | set_size ヒストグラム（1〜10） |
|---|---|---|---|---|---|
| 0.70 | 0.30 | 0.009681 | 0.987500 | 8.025625 | {1:1, 2:4, 3:9, 4:25, 5:56, 6:110, 7:246, 8:443, 9:561, 10:145} |
| 0.80 | 0.20 | 0.003876 | 0.993750 | 8.910000 | {1:0, 2:0, 3:2, 4:5, 5:14, 6:41, 7:107, 8:262, 9:621, 10:548} |
| 0.90 | 0.10 | 0.000980 | 0.996875 | 9.663750 | {1:0, 2:0, 3:0, 4:0, 5:0, 6:5, 7:23, 8:65, 9:319, 10:1188} |
| 0.95 | 0.05 | 0.000353 | 1.000000 | 9.883125 | {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:2, 8:24, 9:133, 10:1441} |

`confidence_level=0.90` 行の q_hat=0.000980 は，本実行 B の stderr 診断値 `q_hat=0.0010`（小数第 4 位
丸め表示）と一致し，sweep スクリプトの計算経路（本体実装 `_compute_prediction_set()` の
`target=alpha*(1+1/n)`, `method="lower"` を再現）が本体実装と整合することの傍証になる。
名目水準を 0.70 まで下げても coverage は 0.9875，mean_set_size は 8.0/10 に留まり，本データ・
本分類器・本校正集合の組み合わせでは，いずれの名目水準でも実用的な集合サイズ（例えば
mean_set_size≤3 程度）には到達しないことが判明した。

**出力ファイル**: `results/20260919_215923/Iter71_qhat_upper.jsonl`，
`results/20260919_215923/Iter71_qhat_alpha_lower.jsonl`，`results/20260919_215923/run_A.log`，
`results/20260919_215923/run_B.log`。分析コード（作業用一時ファイル，リポジトリ未収録）:
`/tmp/iter71_analyze.py`（coverage・mean_set_size・ヒストグラム・ECE・非退行条件），
`/tmp/iter71_qhat_sweep.py`（名目水準振り）。

---

### 分析(解釈) (Iter71)

**判定: rejected（過被覆．事前登録の `coverage > 0.95` パターンに該当）**

#### 1. 数値の独立再検証

実験フェーズの報告値を，出力 jsonl から独立に再集計して確認した（検証スクリプト
`/tmp/iter71_verify.py`，一時ファイル）．

| 指標 | A（upper） | B（alpha_lower） | Iter70 基準線 |
|---|---|---|---|
| coverage | 0.763750（1222/1600） | **0.996875（1595/1600）** | 0.763750 |
| mean_set_size | 1.943750 | 9.663750 | 1.943750 |
| top1_accuracy | 0.603125 | 0.603125 | 0.603125 |
| ECE | 0.062998 | 0.062998 | 0.062998 |

- `set_size` と `len(prediction_set)` の不一致は A・B とも 0 件（出力の自己整合性を確認）．
- B の `set_size` ヒストグラム {6:5, 7:23, 8:65, 9:319, 10:1188} を再現．10 クラス中
  9〜10 個を含む行が 1,507/1,600（94.2%）を占める．
- B で被覆を外した 5 行は `education-020`・`compound-053`・`compound-064`・`compound-074`・
  `compound-096`．複合設問側の全ラベル被覆（`all(expected_domains ⊆ prediction_set)`）も
  0.996250 とほぼ同値であり，主基準の定義（`expected_domains[0]`）の取り方に依存する結論ではない．

#### 2. ノイズか有意か — 帯上限からの逸脱幅

- **実行系のノイズはこのイテレーションでは実測 0 である**．本実行 A は Iter70 出力と
  md5 単位で完全一致（`ee6fbf7d0e4dfd9408c76b2ff9fd1cdf`），B と Iter70 の
  `confidence`・`probabilities` の実測最大差も `0.000e+00`（1e-9 の許容差を使うまでもない）．
  すなわち「ollama 埋め込みの数値再現性」という唯一の変動源も今回は発現しておらず，
  A/B の差はレバーの効果のみに帰着する．
- 残る不確実性は評価集合 1,600 問の標本誤差のみ．事前登録のノイズ幅 SE≈0.006（n=1600, p≈0.94）で
  測ると，実測 coverage=0.996875 は帯上限 0.95 から **+0.046875 ＝ SE の 7.81 倍**離れている．
  実測比率での SE（√(p̂(1-p̂)/n)=0.00140）で測れば **33.6 倍**であり，どちらの取り方でも
  ノイズでは説明できない．
- 件数で見ても，coverage ≤ 0.95 に収まるには被覆外れが 80 件必要なところ，実測は 5 件である．
  Wilson 95% 信頼区間は **[0.99271, 0.99866]** で帯 0.88–0.95 と全く重ならない．
- よって「帯の上を有意に外れた（過被覆）」という判定はノイズ由来ではなく信号である．

#### 3. 非退行条件の再確認（比較対象 `results/20260919_211708/Iter70_corrected_aps.jsonl`）

分析フェーズで独立に再計算した結果，実験フェーズの報告どおり全て充足していた．

1. `selected_domain` 不一致 **0 件**（1,600 行を `id` で突き合わせ，id 欠落も 0）．
2. `confidence` 実測最大差 **0.000e+00**（許容差 1e-9 以内），
   `probabilities`（16,000 要素）実測最大差 **0.000e+00**．
3. `top1_accuracy` は A・B とも **0.603125** で Iter70 から不変．
4. 既定値 `upper` での再実行（本実行 A）が Iter70 出力と **md5 完全一致**（後方互換 OK）．
5. rank_2 候補源への流用は行っていない（スコープ外の約束を遵守）．

同一性アンカーの ECE=0.062998 も合格条件 ≤0.0680 を満たす．**すなわち adopted の 4 条件のうち，
非退行条件と ECE は全て満たし，主基準 coverage のみが不成立である**．

#### 4. 仮説との整合

- **合っていた部分**: 「分位点方向を α 側へ直せば coverage は大きく上がる」という仮説の向きは
  正しかった（0.763750 → 0.996875，+23.3pt．名目 0.90 を跨いで上方へ移動）．実装の発火も
  予備実行・本実行の診断 print（`qhat_quantile_direction=alpha_lower`, `population_size=1427`）と
  `set_size` 分布の激変（1〜5 個 → 6〜10 個）で確認済みであり，invalid 条件には該当しない．
  Iter16 以降くり返してきた「設定は変えたが実行が到達しない」型の失敗ではない．
- **外れた部分**: 大きさが合わなかった．計画の事前見積り（校正集合の α 分位点 ≈0.03，掃引表で
  帯に入るのは 0.04≤q_hat≤0.145）に対し，実測 q_hat=0.000980 は **1〜2 桁小さい**．
  計画時の外挿（校正集合は評価集合の 0.65 倍に縮む）が成り立たず，校正集合の真クラス補数スコアは
  0 側に極端に偏っていた（`cumsum≈1`，すなわち OOF 予測が真クラスにほぼ確信的な校正サンプルが
  下位 10% を埋め尽くしている）．結果として実効閾値 `1-q_hat≈0.999` を要求し，10 クラス中
  9〜10 個を毎回含める自明な集合になった．計画節が実測前に明記していた「上限で外れる目がある」
  という留保が，予想より極端な形で的中したことになる．
- 名目水準振り（付随報告，判定外）はこの解釈を補強する．`confidence_level` を 0.70 まで下げても
  coverage=0.9875・mean_set_size=8.03 であり，**名目水準をどう選んでも実測被覆が名目を大きく
  上回る**（0.70→0.9875, 0.80→0.9938, 0.90→0.9969, 0.95→1.0000）．これは「α の選び方の問題」
  ではなく，**校正集合のスコア分布が評価集合のそれと系統的にずれている**こと，すなわち
  交換可能性（exchangeability）の前提が破れていることを示す形になっている．
  OOF スコアは訓練データ上の交差検証値であり，未見データである評価集合より真クラスへ
  確信的になりやすい，という機序と整合する．

#### 5. 事前登録ルールの適用

事前登録（`### 計画 (Iter71)` 成功条件）では，`coverage > 0.95` の場合は

> **過被覆**．分位点方向の修正自体は機能している（…）が，校正集合（OOF スコア）と評価集合の
> スコア分布のずれにより q_hat が小さすぎる，という**別の原因**に切り分けられる．この場合は
> 本レバーを rejected とし，（…）**判定基準そのものは緩めない**（事後の帯の拡大はしない）．

と定めていた．実測 coverage=0.996875 はこれに該当するため，**rejected** とする．
帯（0.88–0.95）の事後的な拡大・主基準の差し替え・`mean_set_size` による救済はいずれも行わない．
同時に，事前登録どおり「分位点方向の修正自体は機能しているが，校正集合の交換可能性の破れにより
q_hat が小さすぎる」という解釈を採用し，次レバー候補は
**「校正集合の交換可能性（OOF vs ホールドアウト）」**とする（起票は reflector の担当）．

なお本判定は `coverage < 0.88` 側の分岐（「第 3 の実装欠陥の疑い」）には該当しない．
今回の実装は上側・下側の両方向が一次資料の式（`q̂+=⌈(n+1)(1-α)⌉/n`, `q̂-=⌊(n+1)α⌋/n`）と
ユニットテストで突き合わされており，A が Iter70 と md5 一致する後方互換も取れている．
**「実装が 2 つとも正しい状態での確定的判定」という本イテレーションの目的自体は達成された**．

#### 6. 確信度と追加反復の要否

- **追加反復は不要**．パイプラインが決定的（`StratifiedKFold(random_state=42)`，md5 一致で実証）
  であり，同一条件の再実行は同一値を返す．判定を覆すには 5 件の外れが 80 件へ 16 倍に増える必要が
  あり，Wilson CI が帯と重ならないことから標本誤差でも到達しない．確信度は高い．
- 単一レバー（分位点方向）の効果として因果的に言えるのは「coverage を 0.7638 → 0.9969 へ動かした」
  ところまでである．「conformal prediction 自体が本タスクに不適」までは**一般化しない**：
  掃引表が示すとおり q_hat が 0.04〜0.145 の範囲にあれば帯に入る構成は存在し，今回の不成立は
  q_hat の**値**（校正集合の分布）に起因する．したがって系列を閉じる前に，校正集合の取り方を
  変える 1 レバーを試す余地が残っている．

---

### 考察 (Iter71)

**判定の確定: rejected（過被覆）**．`conformal_qhat_quantile_direction` レバー（`values:
[alpha_lower_quantile]` の単一値）はこれでクローズとする．事前登録の帯（0.88 ≤ coverage ≤ 0.95）は
事後に緩めない．実装（`scripts/evaluate_classifier_calibration.py` の `--qhat-quantile-direction`）は
**revert せず維持する**（既定値 `upper` が Iter69/70 出力を md5 単位で再現する後方互換設計であり，
次レバーで `alpha_lower` を固定値として使う必要があるため）．

**このイテレーションで確定した学び**

1. **conformal 系列の実装欠陥は 2 つとも潰れた**．集合構成（Iter70）・分位点方向（Iter71）の両方が
   一次資料（Angelopoulos & Bates 2021 の `q̂+=⌈(n+1)(1-α)⌉/n`，Barber et al. 2021 の
   `q̂-=⌊(n+1)α⌋/n`）と一致し，ユニットテストと後方互換 md5 で裏が取れている．
   Iter56→69→70 と 3 度続いた「原因帰属の誤り」は，少なくとも**実装側では打ち止め**である．
   今回の不成立は実装の第 4 の欠陥ではなく，**q_hat の値＝校正集合の分布**に起因する．
2. **非自明な学び: 校正スコアと評価スコアが別の確率モデルから出ている**．
   `predict_calibrated_rows()` の校正パス（L264-292）は，`classifier.estimator`
   （素の `LogisticRegression`）を 5-fold で**再学習**した fold モデルの `predict_proba` から OOF
   スコアを作る．一方，評価パス（L330 以降）は `models/domain_classifier.joblib`
   （`CalibratedClassifierCV`，全データ学習済み）の `predict_proba` を使う．
   すなわち校正と評価で (a) 学習データ量（4/5 vs 全体），(b) **確率較正の有無**（生 LR vs Platt/isotonic
   較正済み），(c) 元データ（`classifier_train.jsonl` vs `dataset.jsonl`）の 3 点が同時に違う．
   split conformal の被覆保証が要求する交換可能性はここで破れており，較正されていない LR は真クラスへ
   過信的（`cumsum≈1` ⟹ 補数スコア `≈0`）になるため，校正スコアの下側 10% 分位点が
   **q_hat=0.000980** という極端に小さい値へ落ちた．名目水準を 0.70 まで下げても coverage が 0.9875 に
   留まる（実測被覆が名目を常に大きく上回る）という付随報告の形は，この「校正側だけがスコア 0 側に
   偏っている」という機序でしか説明できない．
3. **したがって次の一手は「校正集合の交換可能性」である**．q_hat 掃引表（計画節）が示すとおり
   0.04 ≤ q_hat ≤ 0.145 なら帯に入る動作点は同一データ上に実在する．校正スコアを評価と同一の
   確率モデル・同一分布から取れば q_hat はこのレンジへ移動しうる．
4. **運用上の学び**: 予備実行（先頭 20 行）で実測 q_hat を取り，本実行前に着地点を予測して journal に
   書き留める手順は今回機能した（「rejected に着地する可能性が高い」という申し送りが的中）．
   1〜2 時間の本実行に入る前に着地点を言語化させる手順は今後も維持する．

**次イテレーション（Iter72）の方針**

config の levers は再び全て試行済みになった．SKILL.md 停止条件 1（学びから新レバーを考案できる）を
適用し，新レバー **`conformal_calibration_exchangeability: [eval_holdout_split]`** を config.yml の
`conformal_qhat_quantile_direction` 直下へ追加した（backlog B107）．
`data/dataset.jsonl`（1,600 行）を層化 50/50 分割し，**校正半（800 行）のスコアを評価と同一の
`models/domain_classifier.joblib` で計算**して q_hat を取り，**評価半（800 行）**で coverage を測る．
分位点方向は `alpha_lower`，集合構成は `corrected_aps` に固定する（動かすのは校正集合の取り方 1 点のみ）．
これは上記学び 2 の 3 つの差（学習量・較正の有無・元データ）を**同時に消す**唯一の構成であり，
交換可能性が真因かどうかを 1 反復で決着させられる．成立すれば conformal 系列は adopted で閉じ，
不成立なら「本実装系の欠陥ではなくデータ・分類器側の限界」として系列を閉じる判断を人間に諮る．

**人間判断を要する申し送り（非ブロッキング）**: config の `conformal_qhat_quantile_direction` note の
「不成立の場合」節は，分位点方向を正しても被覆保証が成立しなければ「MAPIE 等の標準ライブラリで追試」か
「conformal 系列を閉じる」かを人間に諮る，と定めていた．本イテレーションは形式的にこれに該当するが，
Iter71 の**計画節（事前登録）**がより具体的に「`coverage > 0.95` の場合は次レバー候補を
『校正集合の交換可能性（OOF vs ホールドアウト）』とする」と定めており，後発かつ具体的なこちらを
優先して Iter72 を自律着手する．人間が「ここで conformal 系列を閉じる」または「MAPIE 追試へ切り替える」
と判断する場合は Iter72 着手前に指示されたい（backlog B107 の要レビュー項目）．

---

## Iteration 70: conformal予測集合の構成規則をAPS標準へ修正して被覆を再測定

### 計画 (Iter70)

**仮説**

Iter56/69 で conformal prediction が rejected になった主因は，`_compute_prediction_set()` の集合構成が
`1 - p_max <= q_hat` という二値ゲートに縮退していた実装欠陥である（調査 (Iter70) Q1 で実データ
`set_size ∈ {1:1061, 10:539}` として確認済み）．`append` を `cumsum` 更新・`break` 判定より前に移し，
標準 APS（Romano et al., NeurIPS 2020）の「確率降順に累積確率が閾値へ達するまでクラスを加える」構成に
直せば，集合サイズは 1〜10 の中間値を取るようになり，被覆と集合サイズのトレードオフを初めて正しく
測定できる．**ただし本イテレーションの目的は採択の獲得ではなく，Iter29 以来 conformal prediction に
関して journal に積み上がってきた記録（Iter56 の invalid，Iter69 の rejected）を，実装が正しい状態での
確定的な判定へ置き換えることである**（backlog B105 の趣旨）．

**単一レバー**

`conformal_set_construction`: `broken`（現行．`append` を `break` 判定の後に行う退化した実装）→
`corrected_aps`（標準 APS．`append` を `cumsum` 更新・`break` 判定より前に移す）．
動かすのはこの集合構成規則 1 点のみ．

**固定する構成（直近の最良構成に固定）**

- q_hat の母集団: Iter69 で確定した `--qhat-source true_class`（`data/classifier_train.jsonl` の
  真クラス非適合スコア 1,427 件，有限標本補正込み 90th percentile，実測 **q_hat=0.3865**）．
  q_hat 算出ロジック（L94-107）には手を入れないため，本実行でも同じ 0.3865 が出るはずである．
- `--confidence-level 0.90`，評価データ `data/dataset.jsonl`（1,600 行），
  分類器 `models/domain_classifier.joblib`，埋め込み `nomic-embed-text:latest`（`127.0.0.1:11435`），
  `--education-logit-bias 0.0` / `--education-threshold 0.0`（既定），
  `--fine-tuned-embed-model` は指定しない（ollama 分岐を通す）．
- `config.yaml`・`http_server.py`・`classifier.py`・`aggregator.py`・`mise.toml` は変更しない．
- 入力データの同一性: `data/classifier_train.jsonl`（mtime 2026-07-30 14:44:31）・
  `data/dataset.jsonl`（2026-09-19 00:49:11）・`models/domain_classifier.joblib`（2026-08-02 23:41:27）
  はいずれも Iter69 本実行（2026-09-19 20:27）より前から変化していないことを計画時に確認した．

**変更箇所（`scripts/evaluate_classifier_calibration.py` 1 ファイル＋テスト）**

1. `_compute_prediction_set()`（L66-126）に引数 `set_construction: str = "broken"` を追加し，
   L113-120 のループを分岐させる．`corrected_aps` の側は次の順序とする:

   ```
   cumsum = 0.0
   for idx in sorted_indices:          # 確率降順
       cumsum += probabilities[idx]
       pred_set.append(int(idx))       # 先に追加
       if 1.0 - cumsum <= q_hat:       # 閾値へ達した回で打ち切る
           break
   ```

   既定値 `"broken"` は現行の挙動（`append` を `break` 判定の後に置く）をそのまま温存し，
   Iter56/69 出力のバイト単位再現性を壊さない．未知の値は `ValueError` を送出する
   （既存の `qhat_source` 検証 L100-101 と同じ書き方に揃える）．
2. docstring の誤記（「the top class gets the SMALLEST score」．実際は `score = 1 - cumsum` が単調
   **減少**するため先頭クラスが最大値を取る．調査 (Iter70) Q1）を修正し，両モードの定義を記す．
3. 呼び出し側 2 箇所（L285 の fine-tuned embedding 分岐，L320 の ollama 分岐）へ `set_construction` を
   伝播．本実験が通るのは **ollama 分岐**である．
4. CLI に `--set-construction`（`choices=["broken", "corrected_aps"]`, `default="broken"`）を追加し，
   `main()` → `_run()` → `predict_calibrated_rows()` → `_compute_prediction_set()` へ受け渡す．
   **`main()` の `--output` 有無による 2 分岐（L443-460 の stdout 側と L461-479 のファイル出力側）の
   両方へ渡すこと．** Iter69 実装フェーズでここの伝播漏れを実際に一度起こしている（調査 (Iter70) Q3）．
   実験で使うのはファイル出力側である．
5. stderr の診断 print に `set_construction` の実効値を追加する（発火証拠の恒久化．既存の
   q_hat・母集団サイズの print はそのまま残す）．
6. `tests/test_evaluate_classifier_calibration.py`（既存）へ単体テストを追加する:
   (a) 同一の `probabilities` / `cp_data` に対し `set_construction="broken"` は 1 か n_classes しか
   返さないのに対し `"corrected_aps"` は中間サイズを返すこと，(b) `corrected_aps` では先頭クラス
   （argmax）が常に集合に含まれること，(c) `corrected_aps` の集合の累積確率が初めて `1-q_hat` 以上に
   なる地点で止まること（1 つ手前のクラスまででは `1-q_hat` に届かないこと），を小さな合成配列で検証する．
   既定値 `"broken"` の回帰テストも 1 件残す．

**到達コードパス**

CLI `--set-construction corrected_aps` → `main()` L461-479（ファイル出力分岐）→ `_run()` L337 →
`predict_calibrated_rows()` L129 → 評価 1,600 行のループ → `_compute_prediction_set()` L320（ollama 分岐．
**集合構成の切替が実際に効く唯一の地点**）→ 出力 jsonl の `prediction_set` / `set_size` → 集計．
`config.yaml` を経由しないため「デプロイ漏れで実行時に読まれない」型の失敗は構造上起こらない．
唯一のリスクは「既定値 `broken` のまま走る」ことであり，下記の予備実行と set_size 分布で潰す．

**事前シミュレーション（計画時に実施．本実行の予測値）**

Iter69 の本実行 B 出力 `results/20260919_202700/Iter69_conformal_qhat_true_class.jsonl` の
`probabilities` 1,600 行に対し，q_hat=0.3865 のまま上記 `corrected_aps` の構成規則をオフラインで
適用したところ:

| 指標 | 予測値 |
|---|---|
| coverage（`expected_domains[0] in prediction_set`） | **0.7638** |
| mean_set_size | **1.9438** |
| set_size 分布 | {1: 539, 2: 671, 3: 332, 4: 57, 5: 1}（6 以上は 0 件） |

q_hat の算出ロジックは本レバーで変更しないため，本実行でもこの値がほぼそのまま再現するはずである．
すなわち **事前登録した成功条件に照らすと coverage 0.7638 は下限 0.87 に届かず，rejected になる公算が
高い**．それでもこの実験を実行する理由は，(a) 予測を事前に書き留めたうえで実測と突き合わせることが
実装の正しさの最も強い検証になること，(b) Iter56 以来「10 クラス APS は方法的限界」とされてきた記録が
バグ由来かどうかの決着が，正しい実装での実測なしには付かないこと，の 2 点である．
なお Iter69 §5 の楽観値（累積確率閾値 0.80 で coverage 0.8544）と矛盾はしない: q_hat=0.3865 は
累積確率閾値 0.6135 に相当し，0.80 より緩いため被覆も低く出る．

**成功条件（事前登録．config.yml `conformal_set_construction` note と Iter69 と同一帯）**

名目信頼水準 `--confidence-level 0.90`，評価 1,600 行．

| 指標 | 定義 | 基準線（Iter69 実測） | 合格条件 |
|---|---|---|---|
| coverage | `mean(expected_domains[0] in prediction_set)` | 0.5988 | **0.87 ≤ coverage ≤ 0.93** |
| mean_set_size | `mean(set_size)` | 4.03（size1=1061 / size10=539 より） | **1.5 ≤ mean_set_size ≤ 4.0** |
| ECE | `metrics.py:compute_ece()` | 0.0630 | **ECE ≤ 0.0680**（同一性アンカー） |

- **adopted**: 上記 3 つを**すべて**満たすこと．
- **rejected**: coverage または mean_set_size が範囲外であること．このとき「実装を正した APS でも
  本分類器（10 クラス，OOF accuracy 57.32%）では被覆と集合サイズを両立できない」という結論が
  初めて正当な実装の上で確定する．
- **invalid**（判定保留）: 実測の coverage / mean_set_size / set_size 分布が上の事前シミュレーション値
  （coverage 0.7638±0.01，mean_set_size 1.9438±0.05）から外れた場合，あるいは set_size が再び
  1 と 10 の 2 値のみになった場合．前者は入力の混入，後者は分岐未到達を意味するため，原因を特定する
  までは adopted / rejected を確定させない．
- **ノイズ幅**: パイプラインは `StratifiedKFold(random_state=42)` で決定的であり，変動源は ollama 埋め込みの
  数値再現性のみ．coverage は n=1600・p≈0.76 の二項標本誤差で SE≈0.011，実質的な差は ±2pt 超とする．
  mean_set_size は ±0.1 を目安とする．
- **ECE の位置づけ**: 本レバーは `confidence`（argmax 確率）を一切変えないため ECE は 0.0630 で不変の
  はずであり，改善を期待する指標ではなくパイプラインの他部分が動いていないことの同一性アンカーである．

**非退行条件**（比較対象は `results/20260919_202700/Iter69_conformal_qhat_true_class.jsonl`）

1. `selected_domain` が全 1,600 行で一致（不一致 0 件）．集合構成は argmax に影響しないため，
   不一致があれば実装ミスである．
2. `confidence` および `probabilities` が一致（許容差 1e-9．埋め込み再計算由来の微差が出た場合は
   その最大値を journal に記録する）．
3. `top1_accuracy` が 0.603125 から不変（条件 1 の系）．
4. 既定値 `--set-construction broken`（＋ `--qhat-source true_class`）での再実行が Iter69 出力と
   md5 一致すること（後方互換．実行手順 3）．
5. 予測集合の rank_2 候補源への流用は本イテレーションでは行わない（スコープ外）．

**実行手順**（オフライン完結．実機 10 ノードの LLM 生成・dispatch・probe は一切行わない．
埋め込みのみ `127.0.0.1:11435` の ollama を使う）

1. **実装**: 上記 1〜6 を実施し，`uv run pytest tests/test_evaluate_classifier_calibration.py` と
   既存テスト・lint・型検査を通す．
2. **予備実行（発火確認）**: `data/dataset.jsonl` の先頭 20 行を `/tmp/iter70_head20.jsonl` に切り出し，
   `--qhat-source true_class` 固定で `--set-construction broken` と `corrected_aps` の 2 通りを実行する．
   - 期待: 両者とも stderr の q_hat=0.3865・母集団 1427 が一致し，`set_construction` の print が
     それぞれの値を示し，**`set_size` の分布のみが変わる**（`broken` は 1 と 10 のみ，`corrected_aps` は
     中間値を含む）．分布が同一なら分岐未到達であり，本実行に進まず実装をやり直す．
3. **本実行 A（後方互換の確認）**: 全 1,600 行を `--qhat-source true_class --set-construction broken` で
   実行し，Iter69 出力（md5=`c9f6b37ef463d91d0402f72fc60ccb96`）と md5 一致することを確認する．
   一致しない場合は差分の原因（ollama バージョン・digest 差）を特定し journal に記録してから進む．
4. **本実行 B（レバー）**: 同一コマンドの `--set-construction corrected_aps` のみを変えて実行する．
5. **分析**: B の出力から coverage・mean_set_size・set_size 分布（1〜10 のヒストグラム）を集計し，
   ECE は `metrics.py:compute_ece()` を流用する．事前シミュレーション値との一致を確認し，
   Iter69 出力との突き合わせで非退行条件 1〜4 を検証したうえで adopted / rejected / invalid を判定する．
6. **補助分析（判定には用いない探索的診断）**: B の `probabilities` を使い，累積確率閾値を
   0.60〜0.98 で掃引したときの coverage と mean_set_size の曲線を算出し，「被覆 0.87 を得るために
   必要な集合サイズ」を記録する．これは次レバー（名目水準の設定など）の材料であり，本イテレーションの
   成功条件には**含めない**．

- **コスト**: 埋め込み 1,427（校正）＋1,600（評価）件の逐次計算で 1 実行あたり 10〜30 分．
  予備実行を含め計 1 時間程度．GPU の実機占有・LLM 生成は不要．
- **出力先**: `results/<timestamp>/Iter70_aps_broken.jsonl`，`results/<timestamp>/Iter70_aps_corrected.jsonl`．

---

### 調査 (Iter70)

**前提確認**: `state.json`（iteration=70, phase=investigate, current_lever=conformal_set_construction）・
`config.yml` の `conformal_set_construction: [corrected_aps]`（785 行目付近，Iter69 reflector 新設・
backlog B105）・journal Iter69 の「分析(解釈) §4」「考察」・backlog B105 を確認した．単一レバーは
`conformal_set_construction=corrected_aps` のみで，q_hat の母集団は Iter69 で確定した true_class 版
（真クラス非適合スコア 1,427 件の 90th percentile，実測 q_hat=0.3865）に固定し，分類器・埋め込み・
argmax・confidence・`config.yaml` は変更しない．成功条件は Iter69 と同一帯
（coverage 0.87–0.93 ∧ 1.5 ≤ mean_set_size ≤ 4.0 ∧ ECE ≤ 0.0680）．

**Q1: 現行実装のバグを自分でコード読解により確認**（`scripts/evaluate_classifier_calibration.py`
`_compute_prediction_set()` L66-126，2026-09-19 時点の行番号．Iter69 が挙げた L66-110 から，同関数への
`qhat_source` 引数追加により行数が伸びている）

- L111 `sorted_indices = np.argsort(-probabilities)` で確率降順にソートし，L113-120 のループで
  `cumsum += probabilities[idx]`（L115）→ `score = 1.0 - cumsum`（L116）→
  `if score <= q_hat: pred_set.append(idx)`／`else: break`（L117-120）という順で処理している．
- **`cumsum` はループが進むにつれ単調増加するため，`score = 1 - cumsum` は単調減少する．**
  したがって `score` が最大値（＝先頭クラスの `1 - p_max`）を取るのは**ループの 1 回目**であり，
  以降のクラスに進むほど `score` は小さくなる一方である．
- ここから帰結する 2 通りの退化した挙動:
  1. 1 回目（先頭クラス）で `score = 1 - p_max <= q_hat` が成立する場合，`score` は以降ずっと
     単調に小さくなり続けるため条件は**二度と偽にならず break が発火しない**．結果，10 クラス
     全てが `pred_set` に入る（`set_size=10`）．
  2. 1 回目で `score = 1 - p_max <= q_hat` が不成立の場合，即座に `break` し `pred_set` は空のまま
     ループを抜け，L122-124 の fallback（`np.argmax(probabilities)` の 1 クラスのみ採用）に落ちる
     （`set_size=1`）．
  - すなわち判定は実質的に **`1 - p_max <= q_hat` という単一の二値ゲート**に縮退しており，
    2〜9 の中間サイズは構造的に出現しない．docstring（L89-90 相当）の「the top class gets the
    SMALLEST score」という説明も，`score` が単調減少なので先頭クラスが**最大値**を取るという
    実装と矛盾している（記述自体が誤り）．
  - **実データでの直接確認**: Iter69 の本実行 B 出力
    `results/20260919_202700/Iter69_conformal_qhat_true_class.jsonl`（1,600 行）を Python で
    再集計したところ `set_size` の分布は `{1: 1061, 10: 539}` の 2 値のみで，2〜9 は 0 件だった．
    journal Iter69「分析(解釈) §4」の記載（A: size1=1510/size10=90，B: size1=1061/size10=539）と
    一致し，バグは journal の主張どおり実データで再現していることを自分で確認した．

**Q2: 標準 APS（Romano, Sesia, Candès, NeurIPS 2020, "Classification with Valid and Adaptive
Coverage"）の正しい集合構成規則**（tavily-search で一次情報・複数解説を確認）

- 出典: NeurIPS 2020 論文本体・補足資料（`proceedings.neurips.cc/paper_files/paper/2020/file/
  244edd7e85dc81602b7615cd705545f5-Supplemental.pdf`），および実装解説（Stanford CS224W course
  blog, "Conformal Prediction for GNNs", `medium.com/stanford-cs224w/...`）．後者は Angelopoulos ら
  の一般的な APS 実装（`aps()` 関数）を要約している．
- **校正時のスコア定義**: クラスを確率降順に並べたとき，真クラスの順位までの**累積確率**
  （cumulative sum，単調増加）をそのサンプルのスコアとする．すなわち標準的な記法では
  スコアは「大きいほど確信度が低い」方向に単調増加する量である．
- **q_hat**: 校正セットのスコアの `⌈(n+1)(1-alpha)⌉/n` 分位点（有限標本補正込み）．
- **テスト時の集合構成**: 確率降順に走査しながら累積確率を足し込み，**累積確率が q_hat 以上に
  達した時点で，その時点までのクラス（達した回のクラスを含む）を集合に加えて打ち切る**．
  「先頭クラスから始めて，必要な被覆質量に達するまで貪欲にクラスを追加する」という説明
  （"include top-rank classes until the cumulative sum of their probabilities meets the desired
  coverage"）が一致して確認できた．
- **現行コードの変数系への翻訳**: 本リポジトリは `score = 1 - cumsum`（標準スコアの補数）という
  定義を採用しており，これ自体は等価な変形で問題ない．しかし「クラスを集合に加えるかどうかの
  判定」と「ループを打ち切るタイミング」を取り違えている．正しい実装は，
  「**現在のクラスを先に `pred_set` に加えてから `cumsum` を更新し，`score <= q_hat` になった
  その回で break する**」という順序でなければならない．擬似コードにすると:

  ```
  cumsum = 0.0
  for idx in sorted_indices:               # 確率降順
      cumsum += probabilities[idx]
      pred_set.append(idx)                 # 先に追加
      score = 1.0 - cumsum
      if score <= q_hat:                   # 閾値に達した回で打ち切る
          break
  ```

  現行コードとの差分は「`append` と `break` 判定の位置を入れ替える」だけであり，`q_hat` の算出
  ロジック（L94-107）・呼び出し側・CLI 引数・fallback（L122-124，理論上は先頭クラスが必ず含まれる
  ため到達しなくなるはずだが安全装置として残してよい）には手を入れる必要がない．この 1 箇所の
  入れ替えで，先頭クラスは常に含まれ（1 回目のループで必ず `append` される），以降は累積確率が
  閾値に達するまでクラスを追加し続けて中間サイズ（2〜9）が出現するようになる見込みが高い．

**Q3: 到達コードパス（config.yaml 非経由であることの再確認）**

- Iter69 の到達コードパス確認（journal 調査(Iter69) Q4）から変化はない．`conformal_set_construction`
  も `config.yaml`・`http_server.py`・`classifier.py`・`aggregator.py`・`mise.toml` のいずれにも
  出現しない（`grep -rn "conformal_set_construction"` は `.claude/research/config.yml` と本 journal
  以外に 0 件）．実行は CLI 直接起動のオフライン評価のみで，実機ノード・LLM 生成・dispatch は
  発生しない．
- CLI 引数の呼び出し経路は Iter69 実装フェーズで確認済みの構造と同型: `main()` L373 以降で
  `argparse` を構築 → `args.qhat_source` 等を `_run()`（L337）→ `predict_calibrated_rows()`
  （L129）→ `_compute_prediction_set()`（L285 の fine-tuned embedding 分岐，L320 の ollama 分岐）
  へ伝播．`main()` の CLI 分岐は **`--output` の有無で `_run()` 呼び出しが 2 箇所（L443-460
  「stdout 出力」／L461-479「ファイル出力」）に分かれており**，Iter69 実装フェーズが「新規 CLI
  引数の伝播漏れ」を実際にここで一度やらかして修正した経緯（journal 実装(Iter69)「実装中に発見し，
  その場で修正した不整合」）がある．本イテレーションで集合構成を切り替える新規 CLI フラグ
  （例 `--set-construction`）を追加する場合も，**この 2 箇所（L443-460 と L461-479）の両方へ
  伝播させることを実装フェーズで必ず確認すること**．実験で実際に使うのはファイル出力側
  （L461-479）である．

**Q4: 再現性・過去データの状況（本イテレーションで再利用可能なもの）**

- `results/20260919_202700/` は現存し，以下 4 ファイルを確認した:
  `Iter69_conformal_qhat_all.jsonl`（933,688 バイト，md5=`2e0a1533a754e5f853d429ef57836616`，
  Iter56 出力とバイト単位一致），
  `Iter69_conformal_qhat_true_class.jsonl`（997,681 バイト，md5=`c9f6b37ef463d91d0402f72fc60ccb96`，
  q_hat=0.3865／population_size=1427，本レバーが固定すべき q_hat の実測値そのもの），
  `run_A.log`／`run_B.log`（stderr 診断ログ）．
- `Iter69_conformal_qhat_true_class.jsonl` を Python で直接読み，`prediction_set`/`set_size`/
  `probabilities`/`confidence`/`selected_domain`/`expected_domains` の全フィールドが揃っている
  ことを確認した．**本レバーの非退行条件（`selected_domain`・`confidence`・`probabilities` が
  1,600 行すべてでこのファイルと一致すること）は，このファイルをそのまま突き合わせ対象として
  使える**．`_compute_prediction_set()` の入力（`probabilities` と `cp_data`）は q_hat の母集団
  選択（Iter69 で固定済み）にのみ依存し集合構成規則には依存しないため，本レバーで
  `probabilities`・`confidence`・`selected_domain` が変化する経路はコード構造上存在しない．
- `data/classifier_train.jsonl`（1,427 行，校正データ）・`data/dataset.jsonl`（1,600 行，評価
  データ）・`models/domain_classifier.joblib` はいずれも Iter69 時点から更新されていないことを
  Iter69 journal で確認済みであり，本イテレーション開始前に再確認は不要（mtime に変化があれば
  計画・実験フェーズで再確認すること）．

**次フェーズ（計画）への示唆**

- 修正は `_compute_prediction_set()`（L66-126）内の「`pred_set.append(idx)` を `cumsum` 更新・
  `break` 判定より前に移す」という 1 行相当の入れ替えに限定できる見込みが高い．`q_hat` 算出
  ロジック（L94-107，Iter69 で確定済みの true_class 母集団）や呼び出し側の引数構造は変更不要．
  Iter69 と同様，既定値では旧（退化した）挙動を温存し，`--set-construction`（仮称，
  `choices=["broken", "corrected_aps"]` 等）のような CLI フラグ 1 つで切り替える設計が単一レバー
  原則・後方互換テストの両方と相性が良い．
- 実装フェーズは，CLI 引数追加時に `main()` の 2 箇所（stdout 出力 L443-460／ファイル出力
  L461-479）双方への伝播を必ず確認すること（Iter69 で実際に踏んだ落とし穴と同型）．
- 実験フェーズは，本レバーの基準線（旧実装）として `results/20260919_202700/
  Iter69_conformal_qhat_true_class.jsonl` をそのまま再利用でき，校正・埋め込み・分類器の再計算は
  不要（`probabilities`/`confidence`/`selected_domain` は不変のはずなので，新規実行では
  `prediction_set`/`set_size` のみを再計算し，このファイルの該当列と突き合わせて非退行条件を
  検証すればよい）．ただし出力ファイルとして独立した jsonl を新規に生成し直すか，
  既存ファイルの再利用可否は rc-planner／rc-experimenter の判断に委ねる．
- 修正後の期待値としては，Iter69 config note が事前根拠として挙げた評価集合自身での楽観的閾値
  掃引（coverage 0.87→size≈3.3，0.90→size≈4.0）が参考になるが，これは校正データではなく評価
  データ自身で閾値を選んだ楽観値である点に留意（実際の OOF 校正では動作点が集合の大きい側へ
  ずれる可能性が高いという config note の留保はそのまま有効）．

---

### 実装 (Iter70)

計画どおり `scripts/evaluate_classifier_calibration.py` 1 ファイルとテストのみを変更した．
`config.yaml`・`state.json`・`config.yml`・分類器・埋め込み・q_hat 算出ロジック（L94-107 相当）は
一切変更していない．

**変更差分の要点**

1. `_compute_prediction_set()` に `set_construction: str = "broken"` を追加．`corrected_aps` の
   分岐では `pred_set.append(int(idx))` を `cumsum` 更新の直後・`score <= q_hat` 判定の**前**に
   置き，計画の擬似コードどおり「先に追加してから閾値到達回で break」の順序にした．既定値
   `"broken"` 分岐は旧コードの行を一切変更せずそのまま残し（`append` は `score <= q_hat` 判定の
   後），バイト単位再現性を壊さないようにした．未知の値は `ValueError`（`qhat_source` の既存検証
   と同じ書き方）．
2. docstring の誤記（「the top class gets the SMALLEST score」）を修正し，`broken`／`corrected_aps`
   両モードの構成規則と，`broken` がなぜ二値ゲートに縮退するかを明記した．
   なお `predict_calibrated_rows()` 内の校正データ構築コメント（L244 付近，`all_scores` を計算する
   ループの直前コメント）にも同型の「top class gets the SMALLEST score」という誤記が別途存在する
   ことに気づいたが，これは計画が変更対象として挙げた `_compute_prediction_set()`（L66-126）の
   docstring とは別の関数内の独立したコメントであり，単一レバー原則（ついでの修正をしない）に
   従い今回は変更していない．後日の別提案として記録する（S1: `predict_calibrated_rows()` 内の
   `all_scores` 計算コメントの「SMALLEST」を「HIGHEST」に修正する，挙動に影響しない純粋なコメント
   修正）．
3. `predict_calibrated_rows()` に `set_construction: str = "broken"` を追加し，fine-tuned embedding
   分岐（旧 L285）・ollama 分岐（旧 L320）の両方の `_compute_prediction_set()` 呼び出しへ伝播．
   2 箇所は完全に同一のコード片だったため `replace_all` で一括置換した．
4. `_run()` に同名引数を追加し `predict_calibrated_rows()` へ伝播．
5. CLI に `--set-construction`（`choices=["broken", "corrected_aps"]`, `default="broken"`）を追加し，
   `main()` の `--output` 有無による 2 分岐（stdout 側／ファイル出力側）**両方**に
   `set_construction=args.set_construction` を伝播したことを個別に確認した（Iter69 で実際に一方の
   分岐への伝播を漏らした経緯があるため，2 箇所とも grep で最終確認済み）．
6. stderr の診断 print に `set_construction={set_construction}` を追記（既存の `qhat_source`・
   `q_hat`・`population_size` の print はそのまま）．

**テスト**

`tests/test_evaluate_classifier_calibration.py` に Iter70 用のテストを 5 件追加した
（`_ITER70_PROBABILITIES = [0.5, 0.3, 0.1, 0.06, 0.04]`，`true_class_scores` を全件 `0.15` に
揃えることで q_hat を厳密に 0.15 に固定する小さな合成配列を使用）:

- `test_set_construction_broken_and_corrected_aps_yield_different_set_sizes`: `broken` は
  `[0]`（size=1，フォールバック経由）に縮退し，`corrected_aps` は `[0, 1, 2]`（size=3）という
  中間サイズを返すことを検証．
- `test_set_construction_corrected_aps_always_includes_top_class`: `corrected_aps` の集合の先頭が
  常に argmax クラスであることを検証．
- `test_set_construction_corrected_aps_stops_at_first_cumulative_crossing`: rank-2 までの累積確率
  だけでは `1-q_hat` に届かず（`1-cumsum(rank2)=0.20 > 0.15`），rank-3 で初めて届く
  （`1-cumsum(rank3)=0.10 <= 0.15`）ことを直接検証し，打ち切り位置がちょうど 1 つずれていないかを
  確認．
- `test_set_construction_default_is_broken_for_backward_compatibility`: 既定値省略時の出力が
  `set_construction="broken"` 明示時と一致することを検証．
- `test_set_construction_rejects_unknown_value`: 未知の値で `ValueError` を検証．

`uv run pytest tests/test_evaluate_classifier_calibration.py -v` は新規 5 件を含む全 9 件が
pass．`uv run ruff check scripts/evaluate_classifier_calibration.py
tests/test_evaluate_classifier_calibration.py` も pass．リポジトリ全体の `uv run pytest tests/`
では `tests/test_build_dataset.py`・`tests/test_train_domain_classifier.py` 計 12 件が失敗するが，
`git stash` で本変更を退避した状態でも同じ 12 件が同じ理由（`train_domain_classifier.py:201` の
`AttributeError`）で失敗することを確認済みであり，本イテレーションの変更とは無関係な既存の失敗
である．

**発火確認（予備実行，計画の実行手順 2）**

`data/dataset.jsonl` の先頭 20 行を `/tmp/iter70_head20.jsonl` に切り出し，`--qhat-source
true_class` 固定で `--set-construction broken` と `corrected_aps` を実行した
（`--ollama-host 127.0.0.1 --ollama-port 11435`，wafl-ctrl5 の Ollama を使用．実機ノード
wafl500〜509 へのアクセスは発生していない）．

- 両モードとも stderr で `q_hat=0.3865 population_size=1427` が一致（Iter69 実測値と同一，計画の
  予測どおり q_hat 算出ロジックは無変更）．`set_construction` の print はそれぞれ `broken` /
  `corrected_aps` と正しく分岐した．
- `set_size` 分布: `broken` は `{1: 20}`（20 行全て size=1．今回のサンプルでは size=10 側は出現
  しなかったが，二値ゲートに縮退している既知の挙動と整合），`corrected_aps` は `{2: 10, 3: 5,
  4: 5}`（中間サイズのみ，分布が一致していないことを確認．計画の「分布が同一なら分岐未到達」
  という失敗条件には該当しない）．
- 非退行条件 1〜3 の先行確認: 20 行全てで `selected_domain` と `confidence` が両モード間で完全一致
  （`abs(confidence_broken - confidence_corrected) <= 1e-9` を含め不一致 0 件）．`probabilities` は
  両モードで同一の `predict_proba` 呼び出し結果を使うため構造上一致する．

**実験を開始してよい状態か**

上記のとおり，コード変更・単体テスト・発火確認のいずれも計画どおりの結果が得られており，実験
フェーズ（本実行 A: 後方互換確認，本実行 B: レバー，全 1,600 行）を開始してよい状態である．

---

### 実験・分析(実行) (Iter70)

**実行環境**: オフライン完結．実機ノード wafl500〜509 は不使用．埋め込み計算のみ
`127.0.0.1:11435`（SSH ローカルフォワード先，wafl-ctrl5 の ollama，`nomic-embed-text:latest` 在中）を
使用．LLM 生成・probe・dispatch トラフィックは発生していない．入力（`data/dataset.jsonl`
mtime 2026-09-19 00:49，`data/classifier_train.jsonl` mtime 2026-07-30 14:44，
`models/domain_classifier.joblib` mtime 2026-08-02 23:41）はいずれも計画時の記録と一致し，
実行直前に再確認した．

**実行コマンド**（A・B とも `--set-construction` のみ変更）:

```
uv run python -m scripts.evaluate_classifier_calibration \
  --dataset data/dataset.jsonl \
  --classifier models/domain_classifier.joblib \
  --embedding-model nomic-embed-text \
  --ollama-host 127.0.0.1 --ollama-port 11435 \
  --conformal-prediction --confidence-level 0.90 \
  --calibration-dataset data/classifier_train.jsonl \
  --qhat-source true_class \
  --set-construction [broken|corrected_aps] \
  --output results/20260919_211708/Iter70_[broken|corrected_aps].jsonl
```

**本実行 A（`--set-construction broken`，後方互換確認）**

- 出力: `results/20260919_211708/Iter70_broken.jsonl`（1,600 行）
- stderr 診断: `q_hat=0.3865 population_size=1427 set_construction=broken`
- md5=`c9f6b37ef463d91d0402f72fc60ccb96`。基準線
  `results/20260919_202700/Iter69_conformal_qhat_true_class.jsonl`（同 md5）と
  **バイト単位で完全一致**した。非退行条件 4（既定値 `broken` の後方互換）を満たす。

**本実行 B（`--set-construction corrected_aps`，レバー）**

- 出力: `results/20260919_211708/Iter70_corrected_aps.jsonl`（1,600 行）
- stderr 診断: `q_hat=0.3865 population_size=1427 set_construction=corrected_aps`
  （q_hat・母集団サイズは A と同一で，計画どおり集合構成規則のみ変化）。

**実測値**（`compute_ece`（`metrics.py`）を流用して算出。coverage の定義は
`expected_domains[0] in prediction_set` の平均値。判定は行わず数値のみ記録する）:

| 指標 | A（broken） | B（corrected_aps） |
|---|---|---|
| coverage | 0.655625（1049/1600） | 0.763750（1222/1600） |
| mean_set_size | 4.031875 | 1.943750 |
| set_size ヒストグラム（1〜10） | {1:1061, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:539} | {1:539, 2:671, 3:332, 4:57, 5:1, 6:0, 7:0, 8:0, 9:0, 10:0} |
| ECE | 0.062998 | 0.062998 |
| top1_accuracy | 0.603125 | 0.603125 |

B の実測値は計画（`### 計画 (Iter70)` 事前シミュレーション）が予測した
coverage=0.7638・mean_set_size=1.9438・分布 {1:539, 2:671, 3:332, 4:57, 5:1} と**小数第4位まで一致**した。
A の実測値（coverage=0.655625, mean_set_size=4.031875）は，Iter69 本実行 B（`qhat_source=true_class`,
旧 `broken` 実装，journal「分析(実行) (Iter69)」§コード修正後の再集計値 coverage=0.6556,
mean_set_size=4.0319）と一致し，md5 一致（本実行 A）とあわせて整合的である。

**非退行条件の検証**（比較対象: `results/20260919_202700/Iter69_conformal_qhat_true_class.jsonl`，
1,600 行を `id` で突き合わせ）:

1. `selected_domain` 不一致件数 = **0**。
2. `confidence` 不一致件数（許容差 1e-9 超）= **0**，実測最大差 = **0.000e+00**。
3. `probabilities`（10 クラス×1,600 行 = 16,000 要素）不一致件数（許容差 1e-9 超）= **0**，
   実測最大差 = **0.000e+00**。
4. `top1_accuracy` は A・B とも **0.603125** で不変（条件 1 の系）。
5. 既定値 `--set-construction broken` の全 1,600 行出力（本実行 A）は Iter69 出力と
   **md5 完全一致**（`c9f6b37ef463d91d0402f72fc60ccb96`）。

非退行条件 1〜5 はすべて満たされた。集合構成規則の切替（B）が `probabilities`・`confidence`・
`selected_domain`・`top1_accuracy` に一切影響しないことが，予測どおり実測でも確認された。

**分析コード**: `/tmp/iter70_analyze.py`（`metrics.py:compute_ece()` を import して流用，
coverage・mean_set_size・ヒストグラム・非退行条件をこのファイル内で算出。作業用の一時ファイルであり
リポジトリには含めていない）。

**出力ファイル**: `results/20260919_211708/Iter70_broken.jsonl`，
`results/20260919_211708/Iter70_corrected_aps.jsonl`，`results/20260919_211708/run_A.log`，
`results/20260919_211708/run_B.log`。

---

### 分析(解釈) (Iter70)

本節は `results/20260919_211708/Iter70_corrected_aps.jsonl`（1,600 行）を一次データとして
直接再集計した結果に基づく（診断スクリプト `/tmp/iter70_interpret.py`，読み取り専用の一時ファイル）．
採否の確定・config.yml への記録・次レバー選定は次フェーズ（rc-reflector）の仕事であり，
本節は「何が起きたか・なぜ起きたか・ノイズか有意か」の解釈に限る．

#### 1. 事前登録した成功条件との機械的対比 → **AND 不成立**

| 指標 | 合格条件 | 本実行 B 実測 | 判定 | 条件境界からの距離 |
|---|---|---|---|---|
| coverage | 0.87 ≤ x ≤ 0.93 | 0.763750 | **不合格** | 下限に対し **-10.63pt** |
| mean_set_size | 1.5 ≤ x ≤ 4.0 | 1.943750 | 合格 | 上限まで 2.06 の余裕 |
| ECE | ≤ 0.0680 | 0.062998 | 合格 | 0.0050 の余裕 |

3 条件の AND が成立条件であるため，**判定は rejected**．未達は coverage の 1 点のみであり，
Iter69（coverage 0.6556 かつ mean_set_size 4.0319 で 2 指標不合格）からは前進しているが，
事前登録の帯には入っていない．非退行条件 1〜5 はすべて充足（実験フェーズ記載のとおり
`selected_domain`・`confidence`・`probabilities` の不一致 0 件・最大差 0.000e+00，
`top1_accuracy`=0.603125 不変，既定値 `broken` の md5 完全一致）．

#### 2. ノイズか有意か → **ノイズ外．境界事例ではない**

- coverage の未達幅 -10.63pt は，n=1600・p=0.7638 の二項標本誤差 **SE=0.01062 の 10.01 倍**である．
  95% CI=[0.7429, 0.7846] の上端でも下限 0.87 に **8.5pt** 届かない．Iter69 で事前登録した
  ノイズ幅（coverage SE≈0.010，±2pt 超を実質的な差とする）に照らしても，
  10.63pt は明確にノイズ外であり，追加反復で判定が反転する余地はない．
- さらに本実験には実質的な測定ノイズが存在しない．(a) 本実行 A が Iter69 出力と **md5 完全一致**
  （パイプラインの決定性を実測で確認），(b) 本実行 B の実測 3 値（coverage 0.7638・
  mean_set_size 1.9438・分布 {1:539, 2:671, 3:332, 4:57, 5:1}）が計画時の事前シミュレーションと
  **小数第 4 位まで一致**．したがって 0.7638 は「1 回の測定のばらつき」ではなく，
  この q_hat・この構成規則の下での決定的な値である．
- **レバーの発火は完全**: `set_size` が A と B で変化した行は **1,600/1,600 行**．被覆の対応付けでは
  discordant が「B のみ被覆」269 件・「A のみ被覆」96 件で McNemar 正確検定 **p=4.22e-20**．
  d0004 §4 の「実験不成立」には該当しない有効な測定である．
  なお「A のみ被覆」が 96 件出るのは，`broken` 版で `set_size=10`（＝全クラス）に跳ねた行が
  真クラスを自明に含んでいたのに対し，`corrected_aps` では小さな集合に絞られて外れるためであり，
  退行ではなく退化した規則の消滅に伴う当然の帰結である．

#### 3. 仮説との整合 → 「集合が適応的になる」は的中，「帯に入る」は不成立

- **的中した部分（実装バグの修正としては成功）**: 計画の仮説「`append` と `break` の順序を正せば
  集合サイズは中間値を取る」は実測で確認された．`set_size` は Iter69 の `{1:1061, 10:539}`（2 値に
  縮退，中間 0 件）から `{1:539, 2:671, 3:332, 4:57, 5:1}` へ変わり，**中間サイズが 1,061 行（66.3%）**
  を占める．APS の眼目である「確信度に応じて集合サイズが適応する」自由度が初めて機能した．
  同時に mean_set_size が 4.0319 → 1.9438 と下がって合格帯に入り，coverage は 0.6556 → 0.7638
  （+10.81pt）と上がった．**被覆を上げながら集合を半分以下に縮めた**点は，退化した二値ゲートが
  被覆効率として極端に悪かったこと（Iter69 §4 の「被覆の伸びは size=10 に跳ねた行だけが担っていた」）
  の裏返しであり，実装バグの修正それ自体の独立した価値として記録に値する．
- **不成立の部分**: 計画が事前シミュレーションで明言していたとおり（「rejected になる公算が高い」），
  coverage は名目水準 0.90 に対し 0.7638 で 13.6pt 低い．計画はこの結果を実行前に予測しており，
  実測はその予測と完全に一致した．すなわち**想定外の挙動（言語崩れ・発散・OOM・分岐未到達など）は
  一切なく**，予測どおりの失敗である．

#### 4. 被覆不足の原因 → **分類器性能の限界ではなく，q_hat の分位点方向という第 2 の独立したバグ**

「10 クラス問題で OOF accuracy 57.32% だから被覆が出ない」という説明は，一次データに照らすと
**主因ではない**．本実行 B の `probabilities` 1,600 行から，真クラスの降順順位と累積確率を直接算出して
確認した（すべて本節の診断スクリプトでの再集計）．

- **真クラスの順位分布**: rank1=953, 2=239, 3=126, 4=95, 5=62, 6=43, 7=32, 8=15, 9=23, 10=12．
  累積すると top1=0.5956 / top2=0.7450 / top3=0.8237 / top4=0.8831 / top5=0.9219．
  すなわち **coverage 0.87 に必要な集合サイズの下限は 3〜4 程度**で，事前登録した上限 4.0 の内側である．
  分類器性能は帯の成立を原理的に妨げていない．
- **本実装の実効閾値**: `corrected_aps` の規則は「累積確率が `1 - q_hat` に達した回で打ち切る」であり，
  q_hat=0.3865 は**累積確率閾値 0.6135** に相当する．同一データで累積確率閾値を掃引すると:

  | 累積確率閾値 | coverage | mean_set_size |
  |---|---|---|
  | 0.6135（＝本実行 B の実効値） | **0.7638** | **1.9438** |
  | 0.75 | 0.8269 | 2.6331 |
  | 0.83 | 0.8688 | 3.3481 |
  | 0.85 | 0.8781 | 3.5806 |
  | 0.88 | 0.9012 | 4.0081 |
  | 0.90 | 0.9181 | 4.3444 |
  | 0.9476 | 0.9444 | 5.5175 |

  閾値 0.6135 の行が実測（0.7638 / 1.9438）と完全に一致することで，掃引の妥当性を確認している．
  **coverage 0.87〜0.93 かつ mean_set_size ≤ 4.0 を満たす閾値帯は [0.832, 0.880] 程度で実在する**．
  つまり今回帯を外したのは，閾値（q_hat）の値が帯に対して大きく緩すぎた（0.6135 ≪ 0.832）ことに尽きる．
- **なぜ閾値が緩すぎるのか（第 2 のバグ）**: 本リポジトリの非適合スコアは
  `score = 1 - cumsum`（累積確率の**補数**）であり，標準 APS のスコア（cumsum，大きいほど悪い）とは
  **符号が反転している**．コードの打ち切り規則から被覆条件を導くと，真クラスが rank r のとき
  「被覆される ⟺ `1 - cumsum_{r-1} > q_hat` ⟺ `score_r + p_r > q_hat`」である．したがって
  P(被覆) ≥ 1-α を保証する q_hat は**補数スコアの α 分位点（= 10th percentile）**でなければならない．
  ところが実装（L94-107，Iter69 で確定させた `true_class` 母集団）は
  **(1-α) 分位点（= 90th percentile）**を取っており，「良さ」を表す量の上側分位点を閾値にしている．
  結果として q_hat が過大（＝閾値 `1-q_hat` が過小）になり，被覆が名目水準を構造的に下回る．
  評価集合上の補数スコアの分位点は q0.10=0.0524・q0.90=0.5956 であり，
  **正しい方向なら q_hat≈0.05 前後（閾値 0.9476）で coverage 0.944 になる**のに対し，
  現実装は q_hat=0.3865（閾値 0.6135）で coverage 0.7638 に留まる．この 18pt 相当の差が，
  今回の未達 10.63pt をそのまま説明する．
- **付随して判明したこと**: Iter56 が「真クラス 90th percentile = 0.5956」と記録した値は，
  **評価集合の補数スコアの 90th percentile そのもの**（本節の実測 0.5956）である．
  Iter69 考察「学び 1」が誤記として整理した数値の出自がこれで確定した（校正集合 OOF での
  同じ量が 0.3865，評価集合でのそれが 0.5956 であり，両者は別集合上の同一定義の統計量である）．

#### 5. 「conformal prediction 自体がこの分類器に不適」かどうか → **まだそうは言えない**

Iter69 考察「学び 2」は「棄却理由は手法ではなく実装欠陥」と結論したが，本イテレーションの結果は
**その結論を維持したまま，欠陥が 1 つではなく 2 つ（集合構成の順序 ＋ q_hat の分位点方向）
だったことを示している**．今回修正したのは前者のみである（単一レバー原則の下では正しい進め方だが，
結果として「正しい実装での確定判定」という当初目的は **半分しか達成できていない**）．
§4 の掃引が示すとおり，分類器性能の側には帯 [0.87,0.93]×[≤4.0] を満たす動作点が実在する．

ただし次レバーへ引き継ぐべき**留保**が 2 点ある．

1. §4 の掃引は評価集合自身で閾値を選んだ**楽観値**である（Iter69 §5 と同じ留保）．実際の校正は
   `data/classifier_train.jsonl` の OOF で行われ，その補数スコア分布は評価集合と異なる
   （上側 10% 分位点が 0.3865 vs 0.5956 で，校正集合のほうが裾が軽い）．正しい方向の分位点
   （下側 10%）が校正集合でどの値になるかは本イテレーションのデータでは測れていない．
2. **非ランダム化 APS は過被覆する**．打ち切り回のクラスを含めるため，名目 α=0.10 で校正しても
   実測 coverage は 0.944（§4 の閾値 0.9476 行）となり，そのとき mean_set_size は 5.52 で
   **上限 4.0 を超える**．すなわち「分位点方向を正すだけ」では，名目 0.90 のままだと
   今度は mean_set_size 側で rejected になる可能性が高い．帯に入る動作点
   （閾値 0.832〜0.880）は名目水準で言えば **0.78〜0.83 相当**である．

#### 6. 判定の確信度と，次フェーズ（考察）への示唆

- **確信度は高い．追加反復は不要**．根拠: (a) パイプラインは決定的で本実行 A が md5 一致，
  (b) 実測が事前シミュレーションと小数第 4 位まで一致，(c) 未達幅が二項 SE の 10 倍，
  (d) 閾値掃引により「この q_hat では帯に入らない」ことを同一データ上で示せており
  単発測定に依存しない．
- 次フェーズへの示唆は **rejected（レバー `conformal_set_construction=corrected_aps` は帯未達）
  かつレバークローズ**（`values: [corrected_aps]` は単一値であり，これで試行済みとなる）．
  ただし棄却の記録には次の 3 点を必ず残すこと．そうしないと Iter56→Iter69→Iter70 と
  2 度繰り返した「誤った原因帰属」が 3 度目を迎える．
  1. **集合構成の修正自体は成功しており，既存バグの修正として独立の価値がある**
     （`set_size` の中間値が 66.3% の行で出現，mean_set_size が合格帯に入り，coverage も +10.81pt）．
     この修正は棄却されたレバーの一部だが，コードとしては維持すべきである（後戻りさせない）．
  2. **未達の原因は `_compute_prediction_set()` の q_hat 分位点方向という第 2 の独立したバグ**
     （補数スコアに対して (1-α) 分位点ではなく α 分位点を取らねばならない．§4 の導出）．
     これは Iter69 で確定させた「母集団の選択（true_class）」とは直交する別の欠陥である．
  3. **次レバーを立てるなら，分位点方向の修正と名目水準の設定を同時に考える必要がある**．
     §5 の留保 2 のとおり，方向だけ正して名目 0.90 に置くと mean_set_size≈5.5 で今度は
     上限 4.0 を外す見込みである．事前登録の帯（coverage 0.87-0.93 ∧ size 1.5-4.0）は
     この分類器では**名目 0.78〜0.83 相当の動作点に対応する帯**であり，名目 0.90 とは両立しない．
     成功条件を「名目水準の妥当性検証（coverage ≈ 名目 ± 2pt）」に組み替えるか，
     名目水準自体をレバーに含めるかの判断は rc-reflector に委ねる．
- **d0004 §4 の再発防止への追記候補**: Iter69 考察が「レバーが作用する内部量の分布を必ず出す」と
  記録した教訓は今回機能し（`set_size` ヒストグラムを事前登録し，中間値の出現を確認できた），
  同型のバグの再発は防げた．一方で今回見落とされていたのは「閾値（q_hat）が想定した実効値に
  なっているか」であり，**内部量の分布に加えて『閾値・ハイパラの実効値が理論値と整合するか』も
  事前登録に含める**べきだった（今回は q_hat=0.3865 が「累積確率閾値 0.6135 に相当する」という
  換算を計画時に一度書いていながら，それが名目 0.90 に必要な 0.95 と乖離していることを
  照合していなかった．計画節の脚注「q_hat=0.3865 は累積確率閾値 0.6135 に相当し，0.80 より緩い」が
  まさにその手前まで来ていた）．

---

### 考察 (Iter70)

**判定: rejected（レバー `conformal_set_construction = corrected_aps`）．本レバーはこれでクローズ．**

事前登録した 3 条件の AND が不成立である．coverage=0.763750 が合格帯の下限 0.87 に **-10.63pt** 届かず，
その未達幅は二項標本誤差 SE=0.01062 の 10.01 倍で，境界事例でもノイズでもない（95% CI の上端 0.7846 でも
下限に 8.5pt 届かない）．mean_set_size=1.943750（帯 1.5-4.0）・ECE=0.062998（≤0.0680）は合格だが，
AND 条件のため判定は rejected で確定する．非退行条件 1〜5 はすべて充足（`selected_domain`・`confidence`・
`probabilities` の不一致 0 件・最大差 0.000e+00，`top1_accuracy`=0.603125 不変，既定値 `broken` での
全 1,600 行出力が Iter69 と md5 完全一致 `c9f6b37ef463d91d0402f72fc60ccb96`）．
追加反復は行わない（パイプラインが決定的で md5 一致が取れており，実測が計画時の事前シミュレーションと
小数第 4 位まで一致しているため，再実行しても同じ値が出る）．

`config.yml` の `conformal_set_construction` は `values: [corrected_aps]` の単一値であり，これで
全値を試行済みとなる．**レバークローズ**．

**コードは後戻りさせない（棄却されたレバーだが実装は維持する）**

レバーとしては棄却だが，`_compute_prediction_set()` の集合構成修正それ自体は**独立したバグ修正として
維持する**．`set_size` の分布は `{1:1061, 10:539}`（2 値に縮退・中間 0 件）から
`{1:539, 2:671, 3:332, 4:57, 5:1}`（中間サイズが 1,061 行 ＝ 66.3%）へ変わり，同時に
mean_set_size 4.0319→1.9438・coverage 0.6556→0.7638（+10.81pt）と，**集合を半分以下に縮めながら
被覆を上げた**．退化した二値ゲートが被覆効率として極端に悪かったことの裏返しであり，
この修正を revert する理由はない．CLI 既定値は `broken` のままなので Iter56/69 の出力は
バイト単位で再現でき，過去の記録も無効化されない．

**学び 1: 棄却の真因は分類器性能ではなく，`q_hat` の分位点方向という第 2 の独立したバグである**

「10 クラスで OOF accuracy 57.32% だから被覆が出ない」という Iter56 以来の説明は一次データに反する．
真クラスの降順順位の累積は top3=0.8237 / top4=0.8831 であり，**coverage 0.87 に必要な集合サイズの
下限は 3〜4 で，事前登録の上限 4.0 の内側**にある．分類器性能は帯の成立を原理的に妨げていない．
帯を外した理由は閾値が緩すぎたこと 1 点に尽きる（実効の累積確率閾値 0.6135 に対し，帯を満たす
閾値帯は [0.832, 0.880] で実在する）．

なぜ緩すぎたかは実装から導ける．本リポジトリの非適合スコアは `score = 1 - cumsum`（標準 APS の
スコアの**補数**）であり，符号が反転している．打ち切り規則から被覆条件を書き下すと，真クラスが
rank r のとき「被覆される ⟺ `score_r + p_r > q_hat`」であるから，P(被覆) ≥ 1-α を保証する q_hat は
**補数スコアの α 分位点（10th percentile）**でなければならない．ところが実装（L94-107）は
**(1-α) 分位点（90th percentile）**を取っており，「良さ」を表す量の上側分位点を閾値にしている．
結果 q_hat が過大（閾値 `1-q_hat` が過小）になり，被覆が名目水準を構造的に下回る．
評価集合の補数スコアは q0.10=0.0524・q0.90=0.5956 であり，方向が正しければ q_hat≈0.05（閾値 0.9476）で
coverage 0.944 に届くところを，現実装は q_hat=0.3865（閾値 0.6135）で 0.7638 に留まっていた．
この 18pt 相当の差が今回の未達 10.63pt をそのまま説明する．
これは Iter69 で確定させた「母集団の選択（`true_class`）」とは**直交する別の欠陥**である．

**学び 2: 「原因帰属の誤り」を 2 度繰り返した．3 度目を避けるための記録**

Iter56 は「10 クラス APS は方法的限界」と帰属し，Iter69 は「q_hat の母集団選択」と帰属し，Iter70 は
「集合構成の順序」と帰属した．いずれも部分的に正しかったが，どれも被覆不足の主因ではなかった．
**この系列で `conformal prediction は本分類器に不適』という結論を出してはならない**（§5 のとおり
帯を満たす動作点が同一データ上に実在する）．結論を確定させてよいのは，分位点方向まで正した実装で
測定した後である．

**学び 3: 事前登録には「内部量の分布」に加えて「閾値・ハイパラの実効値が理論値と整合するか」を含める**

Iter69 考察の教訓（レバーが作用する内部量の分布を必ず出す）は今回機能し，`set_size` ヒストグラムを
事前登録したことで集合構成バグの再発は防げた．一方で見落としたのは閾値側で，計画節に
「q_hat=0.3865 は累積確率閾値 0.6135 に相当し，0.80 より緩い」と**自分で書いていながら**，
名目 0.90 に必要な閾値 ≈0.95 との乖離を照合していなかった．`d0004 §4` の再発防止チェックへ
「ハイパラの実効値を理論値と突き合わせる」を追記する候補とする．

**学び 4: 非ランダム化 APS の過被覆により，帯と名目水準は両立しない**

打ち切り回のクラスを含める非ランダム化 APS は過被覆する．同一データの掃引では閾値 0.9476 で
coverage 0.944・mean_set_size 5.52 であり，**上限 4.0 を超える**．すなわち「分位点方向を正すだけ」で
名目 0.90 に置くと，今度は mean_set_size 側で棄却される見込みが高い．事前登録してきた帯
（coverage 0.87-0.93 ∧ size 1.5-4.0）は，この分類器では**名目 0.78〜0.83 相当の動作点に対応する帯**である．
次レバーの成功条件は「名目水準を固定した帯」ではなく「**被覆保証の妥当性（coverage ≈ 名目 ± 2pt）**」を
主基準に据えるべきである．なお掃引はいずれも評価集合自身で閾値を選んだ楽観値であり，実際の校正は
`data/classifier_train.jsonl` の OOF で行う（補数スコアの下側 10% 分位点が校正集合でいくつになるかは
本イテレーションのデータでは測れていない）という留保が残る．

**次の一手（新レバーを考案して継続．`status` は `running` を維持）**

`conformal_set_construction` のクローズにより config の levers は再び全て試行済みになったが，
SKILL.md 停止条件の優先順位 1（学びから次の有望なレバーを考案できる）に該当するため converged にはしない．
学び 1 が特定した欠陥は具体的・局所的（`_compute_prediction_set()` L94-107 の分位点方向 1 箇所）で，
オフライン完結・分類器再訓練不要・`config.yaml` スキーマ変更なしで自律着手できる．
新レバー **`conformal_qhat_quantile_direction: [alpha_lower_quantile]`** を config.yml の
`conformal_set_construction` 直下へ追加し，Iter71 の単一レバーとする（backlog B106）．
**単一レバー原則の守り方**: 動かすのは「補数スコアに対して (1-α) 分位点ではなく α 分位点を取る」という
分位点方向 1 点のみ．集合構成は Iter70 で維持と決めた `corrected_aps` に固定し，母集団は
`true_class` に固定する．名目水準 `--confidence-level` はレバーではなく**成功条件の側の変数**として扱い，
名目 0.90 を主判定（coverage ≈ 0.90 ± 2pt の妥当性検証）とし，名目を 0.70〜0.95 で振った
coverage/mean_set_size 曲線は「どの名目水準なら帯に入るか」を示す付随報告として記録する
（曲線は判定に用いない）．詳細は `config.yml` の同レバー note と backlog B106 を参照．

---

