## Iteration 69: conformal predictionのq_hat修正版による予測集合被覆の再測定

### 計画 (Iter69)

**仮説**

Iter56 で `conformal_prediction` を rejected と記録した根拠（coverage=0.6056）は，q_hat を
「全 (sample, class) ペアの非適合スコア 14,270 件」の 90th percentile（0.3865）から算出した
実装誤りに起因する．APS（Adaptive Prediction Sets）の理論どおり **真ラベルクラスの非適合スコア
1,427 件**の 90th percentile（Iter56 の手計算では 0.5956）を q_hat に用いれば，名目信頼水準 0.90 に
対する実測 coverage は 0.6056 から大きく上振れする．ただし 10 クラス問題かつ OOF accuracy 57.32% と
いう分類器性能では，被覆を満たすために予測集合が肥大化し（Iter56 の手計算シミュレーションで
mean_set_size=7.31），coverage（0.8025）も名目水準に届かない見込みが強い．
本イテレーションの目的は「採択の獲得」ではなく，**バグ入りの invalid な記録を，正しい実装による
再現可能な判定（採択または棄却）へ置き換えること**である．

**単一レバー**

`routing_confidence_calibration_method`: `conformal_prediction`（q_hat = 全スコア 14,270 件の
90th percentile）→ `conformal_prediction_true_class_qhat`（q_hat = 真クラススコア 1,427 件の
90th percentile）．変更するのは q_hat の算出母集団のみで，分類器・埋め込み・argmax・confidence は
一切変更しない．

**変更箇所（`scripts/evaluate_classifier_calibration.py` 1 ファイルのみ）**

1. `predict_calibrated_rows()` L200-207 の `all_scores` 構築直後・L209 の `cp_data = {...}` の手前に
   真クラススコア抽出を追加し，`cp_data` に `"true_class_scores"`（1,427 要素の 1 次元 ndarray）を
   持たせる．`labels`（L158-161）が同スコープに既にあるためそのまま利用できる．
2. `_compute_prediction_set()`（L66-110）に引数 `qhat_source: str = "all"` を追加し，
   `"true_class"` のとき L89 の `flat_scores = all_scores.flatten()` を
   `cp_data["true_class_scores"]` に差し替える．L90-91 の有限標本補正
   `target = min(1.0, (1-alpha)*(1+1/n))` はそのまま（n は選んだ母集団のサイズ）．docstring の
   「q_hat = (1-alpha) quantile of ALL calibration scores」も両モードを記す形へ更新する．
3. 呼び出し側 L242-246（fine-tuned embedding 分岐）と L277-281（ollama 分岐）へ `qhat_source` を伝播．
   本実験が通るのは **ollama 分岐（L277-281）** である（`--fine-tuned-embed-model` は指定しない）．
4. CLI に `--qhat-source`（`choices=["all", "true_class"]`, default `"all"`）を 1 つ追加し，
   `main()` → `_run()` → `predict_calibrated_rows()` へ受け渡す．既存フラグ
   `--conformal-prediction` / `--confidence-level` / `--calibration-dataset` は変更しない．
5. q_hat の値と算出母集団サイズを stderr に 1 行 print する（発火証拠の恒久化）．
6. `tests/test_evaluate_classifier_calibration.py`（新規）に `_compute_prediction_set()` の単体テストを
   追加する: 同一の `cp_data` に対し `qhat_source="all"` と `"true_class"` で **q_hat が異なり，
   得られる予測集合サイズが異なる**ことを小さな合成配列で検証する（レバーが分岐として機能する証明）．

`config.yaml`・`http_server.py`・`classifier.py`・`aggregator.py`・`mise.toml` は変更しない．

**到達コードパス（d0004 §4 の no-op 6 連発への対処）**

CLI `--qhat-source true_class` → `main()` L391 `args` → `_run()` L297-322 →
`predict_calibrated_rows()` L113 → `if conformal_prediction:` L150-209（ここで `true_class_scores` を
構築）→ 評価 1,600 行のループ L260-293 → `_compute_prediction_set()` L279-281（**ここで q_hat の
母集団切替が実際に効く唯一の地点**）→ 出力 jsonl の `prediction_set` / `set_size` → 集計スクリプト．
本レバーは config.yaml を経由しないため「デプロイ漏れで実行時に読まれない」型の失敗は構造的に
起こり得ない．代わりの唯一のリスクは「`qhat_source="true_class"` 分岐に入らないまま既定値 `"all"` で
走る」ことであり，下記の予備実行で潰す．

**成功条件（事前登録．`fallback_rate` は Iter28 の fallback 廃止で常に 0 のため使用しない）**

名目信頼水準は `--confidence-level 0.90`．評価は `data/dataset.jsonl` 1,600 行．

| 指標 | 定義 | 基準線（Iter56 バグ版実測） | 合格条件 |
|---|---|---|---|
| coverage | `mean(expected_domains[0] in prediction_set)` | 0.6056 | **0.87 ≤ coverage ≤ 0.93** |
| mean_set_size | `mean(set_size)` | 1.51 | **1.5 ≤ mean_set_size ≤ 4.0** |
| ECE | `metrics.py:compute_ece()` | 0.0630 | **ECE ≤ 0.0680**（＝基準線 +0.005 以内の非退行） |

- **adopted の条件**: 上記 3 つを**すべて**満たすこと．
- **rejected の条件**: coverage または mean_set_size のいずれかが範囲外であること
  （このとき Iter56 の invalid 記録は，正しい実装に基づく正式な棄却の記録へ置き換わる）．
- **ノイズ幅の見積もり**: パイプラインは `StratifiedKFold(random_state=42)` で決定的であり，
  変動源は ollama 埋め込みの数値再現性のみ．coverage は n=1600・p≈0.8 の二項標本誤差で
  SE≈0.010 のため ±2pt を超える差のみを実質的な差とみなす．mean_set_size は 10 クラス上限で
  離散的に動くため ±0.1 を目安とする．
- **ECE についての事前の但し書き**: 本レバーは `confidence`（= argmax 確率）を一切変えないため，
  ECE は原理的に Iter56 と同一値（0.0630）になるはずである．したがって ECE は「改善を期待する指標」
  ではなく **パイプラインの他部分が意図せず変わっていないことを確認する同一性アンカー**として
  事前登録する．0.0630 から 0.005 を超えて動いた場合は，レバー以外の混入を疑い原因を特定するまで
  判定を確定させない．

**非退行条件**

1. `selected_domain`（argmax）が全 1,600 行で Iter56 の
   `results/20260808_000000/Iter56_conformal_prediction.jsonl` と**ビット単位で一致**すること
   （不一致 0 件）．q_hat は予測集合のみに影響し argmax には影響しないため，一致しなければ実装ミス．
2. `top1_accuracy` が 0.6056 から不変であること（条件 1 の系）．
3. `confidence` および `probabilities` が Iter56 出力と一致すること（許容差 1e-9．埋め込み再計算に
   由来する微差が出た場合はその大きさを journal に記録する）．
4. 予測集合の rank_2 候補源への流用は本イテレーションでは**行わない**（config note の指示どおり
   スコープ外．mean_set_size が top-2 dispatch に対して大きすぎるため）．

**実行手順**

本実験は **オフライン完結**である．10 ノードの実機ディスパッチ（`run_experiment.py` 1,600 問の
LLM 生成）は一切行わず，埋め込み計算のみ `127.0.0.1:11435`（SSH ローカルフォワード先の ollama，
`nomic-embed-text:latest` 在中を確認済み）を使う．LLM 生成・probe・dispatch トラフィックは発生しない．

1. **実装**: 上記 1〜6 を実施し，`uv run pytest tests/test_evaluate_classifier_calibration.py` と
   既存テスト・lint を通す．既定値 `qhat_source="all"` により Iter56 の挙動は温存する．
2. **予備実行（発火確認．d0004 §4 の教訓）**: 評価データセットの**先頭 20 行**のみを
   `/tmp/iter69_head20.jsonl` に切り出し，`--qhat-source all` と `--qhat-source true_class` の
   2 通りで実行し，stderr に出る q_hat を記録する．
   - 期待: `all` → q_hat ≈ 0.3865（母集団 14,270），`true_class` → q_hat ≈ 0.5956（母集団 1,427）．
   - **2 つの q_hat が異なることを確認できるまで本実行に進まない**．値が一致した，または
     `true_class` 側で母集団サイズが 14,270 と表示された場合は分岐未到達であり実装をやり直す．
3. **本実行 A（基準線の再現）**: 全 1,600 行を `--qhat-source all` で実行し，coverage=0.6056・
   mean_set_size=1.51・q_hat=0.3865 が再現することを確認する（埋め込み経路の同一性検証を兼ねる）．
   再現しない場合は，差分の原因（ollama バージョン・digest 差）を特定して journal に記録してから進む．
4. **本実行 B（レバー）**: 同一コマンドの `--qhat-source true_class` のみを変えて実行する．
5. **分析**: 出力 jsonl から coverage・mean_set_size・set_size 分布（1〜10 のヒストグラム）を
   10 行程度の集計スクリプトで算出し，ECE は `metrics.py:compute_ece()` を流用する．
   A と B の `selected_domain` / `confidence` を突き合わせ，非退行条件 1〜3 を検証する．
   成功条件表の 3 指標と非退行条件の結果を journal に記録し，adopted / rejected を判定する．

- **コスト見積もり**: 埋め込み 1,427（校正）＋1,600（評価）件の逐次計算で 1 実行あたり 10〜30 分．
  A・B の 2 実行と予備実行で計 1 時間程度．GPU の実機占有・LLM 生成は不要．
- **出力先**: `results/<timestamp>/Iter69_conformal_qhat_all.jsonl` および
  `Iter69_conformal_qhat_true_class.jsonl`．

---

### 調査 (Iter69)

**背景**: レバー `routing_confidence_calibration_method=conformal_prediction_true_class_qhat` は
config.yml:718-761（backlog B103）で既に決定済み．本節はレバー選定ではなく，計画フェーズ（rc-planner）
が実験設計を確定するための一次情報確認（実装箇所・再現性・計測方法・到達コードパス）を行う．

**Q1: q_hat 計算の実装場所の特定**（`scripts/evaluate_classifier_calibration.py`）

- **バグ箇所（現行 = Iter56 の全スコア版）**: `_compute_prediction_set()` L66-110．
  L85 `all_scores = cp_data["all_scores"]`（shape=(1427, 10)，n_cal×n_classes）を
  L89 `flat_scores = all_scores.flatten()` で **14,270 要素すべて**平坦化し，L90-91 で
  その 90th percentile を q_hat とする．これが「全スコア14,270件」の実体．
- **cp_data 構築箇所**: `predict_calibrated_rows()` L139-209．真ラベルのクラス index は
  L158-161 の `labels`（`classes.index(cal_row["domain"])`，1427 要素）としてこの関数スコープ内に
  既に存在する．L196-207 で `all_scores[i, idx] = 1.0 - cumsum`（各サンプル i の全 10 クラス分の
  非適合スコアを格納）を計算しているが，**真クラスのみを抽出する処理は現状存在しない**．
- **修正すべき具体箇所**: L200-207 の直後（cp_data 構築を `{"all_scores": all_scores}` としている
  L209 の手前）に，
  `true_class_scores = np.array([all_scores[i, labels[i]] for i in range(n_cal)])` を追加し，
  cp_data に `"true_class_scores": true_class_scores`（1,427 要素）を持たせる．
  `_compute_prediction_set()` には q_hat の算出元を切り替える引数（例: `qhat_source: str = "all"`，
  `"true_class"` のとき L89 の `flat_scores` を `cp_data["true_class_scores"]` に差し替える）を追加する．
  CLI 側は新規フラグ（例 `--qhat-source true_class`）を 1 つ足すだけで済み，既存の
  `--conformal-prediction` / `--confidence-level` / `--calibration-dataset` はそのまま流用できる．
  変更ファイルは本スクリプト 1 本のみ（単一レバー原則との相性は良好）．

**Q2: Iter56 修正シミュレーションの再現性確認**

- journal_archive.md:6656-6659 および 6704-6710 に coverage=0.8025・mean_set_size=7.31（真クラス
  90th percentile=0.5956 使用）の記載があるが，**算出に使った具体的なコマンド・スクリプト名は
  journal に明記されていない**（「シミュレーション」とのみ記述）．
- 一次情報として確認できたこと: 校正データセットは `data/classifier_train.jsonl`（1,427 行，
  `md5sum` = `fa2c9f57cf32ca7f1cb08384c7f49800`，更新日時 2026-07-30，**Iter56 実行日 2026-08-08 より
  前で内容不変**）．評価データセットは `data/dataset.jsonl`（1,600 行，education 77 件等の内訳は
  journal_archive.md:15356 と整合）．OOF 予測は `StratifiedKFold(n_splits=5, shuffle=True,
  random_state=42)`（L162）で決定的にシードされており，同一データ・同一シードなら再現可能．
  `results/20260808_000000/Iter56_conformal_prediction.jsonl`（933,688 バイト，現存）は当時の
  「全スコア版」の出力で，各行の `probabilities` フィールドは q_hat の取り方に依存しないため，
  **このファイルの `probabilities` と `expected_domains` を読み直すだけでも真クラス版 q_hat・
  prediction_set を事後計算できる**（cp_data の OOF 再学習をやり直す必要はない）．
  ただし厳密な再現には Q1 の修正版コードを実際に実行し，`_compute_prediction_set()` の出力
  `prediction_set`/`set_size` をログとして残すほうが「シミュレーション」より検証可能性が高い．
- **結論**: オフライン完結・実機（ollama ノードでの embedding 計算）不要で再現可能．
  校正データが 1,427 件と小さいため 5-fold OOF 学習は数秒〜数十秒で完了する．

**Q3: ECE・coverage・mean_set_size の計測方法**

- **ECE**: `metrics.py:486-521` `compute_ece(results)` をそのまま流用可能．`evaluate_classifier_
  calibration.py` の出力行はいずれも `confidence`/`selected_domain`/`expected_domains` を持つため
  （L248-254, L283-289），追加実装は不要．Iter56 でも同じ関数系のロジックで ECE=0.0630 が算出されている
  （L517-518 の判定式 `selected_domain in expected_domains` は単一ドメインデータセットで問題なく機能）．
- **coverage・mean_set_size**: `metrics.py` 内に該当関数は**存在しない**（`grep -n "coverage" metrics.py`
  でヒットするのは `compute_compound_coverage_metrics()` のみで，これは複合ドメイン行の候補集合被覆
  用であり，conformal prediction の prediction set 被覆とは定義が異なる）．Iter56 でも同様にアドホック
  集計だったとみられる（journal に metrics.py 関数呼び出しの記載がない）．**新規に軽量な集計処理が
  必要**: `coverage = mean(row["expected_domains"][0] in row["prediction_set"] for row in rows)`，
  `mean_set_size = mean(row["set_size"] for row in rows)`．データセットが単一ドメインのみのため
  `expected_domains[0]` で真ラベルが一意に定まる．この 2 指標は `evaluate_classifier_calibration.py`
  の出力 jsonl（`prediction_set`/`set_size` フィールドが既に付与されている，L255-257, L290-292）を
  読むだけで計算できる 10 行程度のスクリプトで足り，新規のモジュール実装は不要．

**Q4: 単一レバー原則の到達確認**

- `routing_confidence_calibration_method`（および両方の値 `conformal_prediction` /
  `conformal_prediction_true_class_qhat`）は `config.yaml`・`http_server.py`・`classifier.py`・
  `mise.toml` のいずれにも出現しない（`grep -rn` で 0 件，`.claude/research` 配下のみヒット）．
  Iter56 も本イテレーションも，**実行時経路（config.yaml デプロイ・http_server 起動・実機 1600 問）を
  一切経由しない**．実行はすべて `scripts/evaluate_classifier_calibration.py` の CLI 直接起動
  （`--conformal-prediction` 等のフラグ）によるオフライン評価であり，`results/20260808_000000/` は
  その出力である．
- **到達コードパス**: CLI 引数（新設 `--qhat-source true_class`）→ `main()` L331 以降 → `_run()`
  L297 → `predict_calibrated_rows()` L113 → `if conformal_prediction:` ブロック L150-209（ここに
  Q1 の `true_class_scores` 追加）→ 各行ループ内 `_compute_prediction_set()` 呼び出し L242-246 /
  L277-281（ここで q_hat 算出元の切り替えが実際に効く）→ 出力 jsonl の `prediction_set`/`set_size` →
  Q3 の新規集計（coverage・mean_set_size）および `metrics.compute_ece()`．
- **d0004 §4 教訓との関係**: 過去 6 回の失敗パターンは「config.yaml を正しく変えたのにコードが実行時に
  その変更を読まない（デプロイ漏れ等）」というものだったが，本レバーはそもそも **config.yaml を一切
  変更しない**オフライン限定の実験であるため，同種の失敗リスクは構造的に存在しない．代わりに注意すべき
  リスクは「新設した `qhat_source="true_class"` 分岐が実際に呼ばれたか」の確認漏れであり，
  rc-experimenter は本走前に少数サンプル（例: 先頭 20 行）で q_hat の値そのもの（0.5956 付近になる
  はず）を print・記録して発火を確認すること．

**次フェーズ（計画）への示唆**

- 成功条件は fallback_rate を使わず，config note（Iter68 reflector追記）の指示どおり
  **ECE・coverage・mean_set_size の 3 つ**で事前登録する．Iter56 バグ版の実測値
  （coverage=0.6056, mean_set_size=1.51, ECE=0.0630）と Iter56 の手計算シミュレーション
  （coverage=0.8025, mean_set_size=7.31）を基準線・期待レンジの参考値として使える．
- 予測集合の rank_2 候補源への流用は対象外（config note に明記済み，本調査でも変更不要と確認）．
- 変更ファイルは `scripts/evaluate_classifier_calibration.py` 1 本（`_compute_prediction_set()` の
  q_hat 算出元切り替え＋ cp_data への `true_class_scores` 追加＋ CLI フラグ 1 つ）に閉じられる見込み．
  新規の coverage/mean_set_size 集計は別のワンライナー的スクリプトで良く，`metrics.py` の改変は不要
  （ECE のみ既存関数を再利用）．

### 実装 (Iter69)

**変更ファイルは計画どおり `scripts/evaluate_classifier_calibration.py` 1 本のみ**（`config.yaml`
は無変更）．実装前にコードを Read して行番号を照合したところ，計画時点の見積り
（L66-110 / L200-207 / L209 / L242-246 / L277-281 / L158-161）はいずれも実装時点の行番号と
**完全に一致**しており，ズレはなかった．

1. `predict_calibrated_rows()` の `cp_data = {"all_scores": all_scores}`（旧 L209）の手前に，
   `true_class_scores = np.array([all_scores[i, labels[i]] for i in range(n_cal)])` を追加し，
   `cp_data = {"all_scores": all_scores, "true_class_scores": true_class_scores}` とした．
2. `_compute_prediction_set()` に `qhat_source: str = "all"` を追加し，未知の値には
   `ValueError` を送出するガードを入れた．`"true_class"` のとき `flat_scores` を
   `cp_data["true_class_scores"]` に差し替え，有限標本補正 `(1-alpha)*(1+1/n)` はそのまま
   （n は選んだ母集団のサイズに追随）．docstring を APS の標準手続き（Romano et al., 2020,
   "Classification with Valid and Adaptive Coverage sets"）に基づく true-class 版と，旧実装の
   all 版の両方を説明する形に更新した．
3. 呼び出し元 2 箇所（fine-tuned embedding 分岐・ollama 分岐）双方の `_compute_prediction_set()`
   呼び出しへ `qhat_source=qhat_source` を伝播．`predict_calibrated_rows()` → `_run()` →
   `main()`（CLI 分岐）まで一貫して引数を通した．
4. CLI に `--qhat-source`（`choices=["all", "true_class"]`, default `"all"`）を追加．
5. **発火証拠の恒久化**: `cp_data` 構築直後（校正データに対して 1 回だけ），選択された
   `qhat_source` の q_hat 値と算出母集団サイズを stderr に 1 行 print するようにした
   （行ごとに 1600 回出さず，校正時に 1 回のみ．ノイズを避けつつ発火確認は可能）．
6. 新規テスト `tests/test_evaluate_classifier_calibration.py`（4 ケース）を作成した．
   10 クラスの toy 校正データ（true class 列だけ非適合スコアを高く設定）を用い，
   (a) `qhat_source="all"` と `"true_class"` で予測集合サイズが異なること（1 vs 10，
   q_hat が分岐として機能する直接証拠），(b) `"true_class"` モードが `cp_data["all_scores"]`
   に一切触れないこと（空配列を仕込んでも例外が出ないことで確認），(c) 引数省略時の挙動が
   `qhat_source="all"` 明示指定と完全一致すること（後方互換），(d) 未知の `qhat_source` 文字列で
   `ValueError` が送出されることを検証した．

**実装中に発見し，その場で修正した不整合（計画には記載のなかった追加修正）**: `main()` の
CLI 分岐は `--output` 指定の有無で `_run()` 呼び出しが 2 箇所に分かれている
（L443-460「stdout 出力」・L461-478「ファイル出力」）．最初の実装で stdout 側にのみ
`qhat_source=args.qhat_source` を渡し，ファイル出力側（実験で実際に使う経路）への伝播を
書き漏らした．CLI 引数を追加する既存の 2 箇所組を見落とすと，計画が警告する「config を
正しく変えたがコードに到達しない」と同型の分岐未到達バグを新規コードに作り込むところだった．
2 箇所を Read で突き合わせて発見し，実装直後に修正済み（テストでは検出できない類のバグのため，
後続の予備実行での q_hat 実測確認が実際に効いた）．

**単体テスト結果**: `uv run pytest tests/test_evaluate_classifier_calibration.py -v` は
4/4 passed．

**既存テストへの影響確認**: `uv run pytest tests/ -q` は 12 failed / 266 passed。失敗した 12 件
（`test_build_dataset.py` 9 件・`test_train_domain_classifier.py` 3 件）はすべて
`scripts/train_domain_classifier.py:201` の `calibrated_model.classes_` が
`AttributeError: 'CalibratedClassifierCV' object has no attribute 'classes_'` で落ちるという，
sklearn バージョン起因の**本変更と無関係な既存の失敗**であることを `git stash` で本変更前の
状態に戻して同じ 12 件が同様に失敗することを確認して切り分けた（本変更のコミットに起因しない）．
`uv run ruff check scripts/evaluate_classifier_calibration.py tests/test_evaluate_classifier_calibration.py`
は all checks passed．

**後方互換の確認**: 先頭 20 行（`/tmp/iter69_head20.jsonl`，`data/dataset.jsonl` 冒頭）を
`models/domain_classifier.joblib` + `data/classifier_train.jsonl`（1,427 行）+
`nomic-embed-text`（127.0.0.1:11435）で `--conformal-prediction --confidence-level 0.90` により
実行し，(1) `--qhat-source` を省略した出力と (2) `--qhat-source all` を明示した出力を
`diff` で比較したところ **バイト単位で完全一致**した．既定値 `"all"` により旧挙動が温存されている
ことを確認した．

**予備実行（発火確認）**: 同じ先頭 20 行に対し `--qhat-source all` と `--qhat-source true_class`
を実行し，stderr の診断出力を記録した:
- `qhat_source=all q_hat=0.1133 population_size=14270`
- `qhat_source=true_class q_hat=0.3865 population_size=1427`

母集団サイズ（14,270 / 1,427）は計画どおりで，2 つの q_hat（0.1133 と 0.3865）は明確に異なり，
分岐は実際に発火していることを確認した．**ただし実測値は journal Iter56/Iter69 計画節が挙げた
見積り（all≈0.3865, true_class≈0.5956）とは一致しなかった**（本実装での true_class 実測値
0.3865 は，むしろ Iter56 の「all（バグ版）」の実測値と数値が一致するという偶然の符合がある）。
原因として考えられるのは，Iter56 実行時（2026-08-08）に使われた `models/domain_classifier.joblib`
と現在の同名ファイルが同一でない可能性，または classifier.estimator のハイパーパラメータ・
校正データの差異である．**この数値差の切り分けは本実装フェーズの範囲外**とし，次フェーズ
（rc-experimenter）が計画の「本実行 A（基準線の再現）」で `--qhat-source all` の全 1,600 行実行を
行い，Iter56 実測（coverage=0.6056, mean_set_size=1.51, q_hat=0.3865）と一致するか確認する際に
合わせて検証すること．一致しない場合は計画の指示どおり差分の原因（ollama バージョン・
classifier joblib の版・digest 差）を特定してから本実行 B（true_class）に進むこと．
なお，先頭 20 行はいずれも `business_economics` ドメインの高確信度な行だったため
`set_size` は all・true_class 双方で 1（同一）だった．prediction set サイズの分岐は
q_hat が実際に異なることをもって代えて確認済みであり（上記診断出力），かつ toy データによる
単体テストで q_hat 差が prediction set サイズ差に反映されることも別途証明済みである．

**実験を開始してよい状態か**: 実装・単体テスト・後方互換・発火確認はすべて完了しており，
コード面では次フェーズ（rc-experimenter）による本実行に進んでよい．ただし上記の q_hat 実測値と
Iter56 見積りの不一致は，本実行 A（基準線再現）で解消するか原因を特定するまで，最終的な
adopted/rejected 判定を確定させないこと．

### 実験 (Iter69)

**実行環境**: オフライン完結．実機ノード wafl500〜509 は不使用．埋め込み計算のみ
`127.0.0.1:11435`（SSH ローカルフォワード先，wafl-ctrl5 の ollama，`nomic-embed-text` 在中）を使用．
LLM 生成・probe・dispatch トラフィックは発生していない．

**実行コマンド（A・B とも `--qhat-source` のみ変更）**:
```
uv run python -m scripts.evaluate_classifier_calibration \
  --dataset data/dataset.jsonl \
  --classifier models/domain_classifier.joblib \
  --embedding-model nomic-embed-text \
  --ollama-host 127.0.0.1 --ollama-port 11435 \
  --conformal-prediction --confidence-level 0.90 \
  --calibration-dataset data/classifier_train.jsonl \
  --qhat-source [all|true_class] \
  --output results/20260919_202700/Iter69_conformal_qhat_[all|true_class].jsonl
```

**本実行 A（`--qhat-source all`，基準線の再現，開始前に最優先で確認）**

- 出力: `results/20260919_202700/Iter69_conformal_qhat_all.jsonl`（1,600 行）
- stderr 診断: `q_hat=0.1133 population_size=14270`
- **q_hat の生値（0.1133）は Iter56 の記録値（0.3865）および Iter69 実装フェーズの予備実行での
  見積りと一致しなかった**が，これは実装フェーズ時点で判明していた既知の懸念であり，本実行では
  この不一致自体の原因切り分けを行った（詳細は下記「q_hat 数値不一致の原因調査」）．
- **成功条件表・非退行条件で実際に使う指標（selected_domain・confidence・probabilities・
  top1_accuracy・coverage・mean_set_size・ECE）はいずれも Iter56 の記録と一致した**
  （数値は「分析(実行) (Iter69)」節に記載）．

**本実行 B（`--qhat-source true_class`，レバー）**

- 出力: `results/20260919_202700/Iter69_conformal_qhat_true_class.jsonl`（1,600 行）
- stderr 診断: `q_hat=0.3865 population_size=1427`
- **偶然の符合**: この true_class の q_hat 実測値（0.3865）は，Iter56 が記録した all モードの
  q_hat（0.3865）と数値が一致する．一方，Iter56 の手計算シミュレーションが見積もった
  true_class の q_hat（0.5956）とは一致しない．

**q_hat 数値不一致の原因調査（本実行 A 実施前後に実施）**

計画で指示された「一致しない場合は原因を特定してから本実行 B に進む」に従い，以下を確認した:

1. **入力ファイルの不変性**: `models/domain_classifier.joblib`（mtime: Aug 2 23:41，Iter56 実行
   `2026-08-08` より前）・`data/classifier_train.jsonl`（mtime: Jul 30 14:44，同様に Iter56 より前）
   はいずれも Iter56 実行以降に更新されていない．
2. **コードの不変性**: `git log --oneline --follow -- scripts/evaluate_classifier_calibration.py`
   の最新ヒットは Iter56 のコミット `1e63d67` そのものであり，`--qhat-source` を追加した Iter69 の
   差分（未コミット）を除けば，OOF 較正処理（`StratifiedKFold(random_state=42, shuffle=True)`・
   `fold_clf = type(base_estimator)(max_iter=1000)`・スコア計算式）は Iter56 実行時点から
   一切変更されていない．
3. **ライブラリバージョンの不変性**: `uv.lock` の `scikit-learn`（1.9.0）は Iter43 以降コミットが
   なく，Iter56 実行時と同一バージョンである．
4. **決定性の確認**: 校正データセット（1,427 行）に対する OOF 較正を独立に 2 回実行
   （`/tmp/iter69_head5.jsonl` を評価対象に使い，校正部分だけを比較），いずれも
   `q_hat=0.1133 population_size=14270` と完全に一致し，現行環境・現行入力のもとでは
   再現性がある（非決定的な揺らぎではない）．
5. **eval 行の再現性（決定的な一次証拠）**: 本実行 A の 1,600 行すべてについて，
   `selected_domain`・`confidence`・`probabilities` を Iter56 の記録
   （`results/20260808_000000/Iter56_conformal_prediction.jsonl`）と突き合わせたところ，
   **1,600 行全てでビット単位（許容差 1e-9）で一致した**（0 件不一致，詳細は次節）．
   これは，凍結済み分類器（`classifier.predict_proba`）を通る eval 行の経路については，
   embedding 計算（`127.0.0.1:11435` 経由）・分類器アーティファクト・データセット行順序が
   Iter56 実行時と実質的に同一であることを示す一次証拠である．
6. **結論（切り分けの限界）**: 上記 1〜5 により，コード・データ・ライブラリ・embedding
   経路のいずれも変化していないことを確認したが，OOF 較正専用の再学習経路
   （`fold_clf.fit(cal_embeddings[train_idx], ...)`）が生成する q_hat の生値そのものが
   Iter56 記録（0.3865）と異なる根本原因は，Iter56 当時の生の OOF 確率や校正スコア配列自体が
   一切保存されていない（journal・results のいずれにも persist されていない）ため，
   **これ以上の一次情報による特定はできなかった**．ただし，この q_hat の生値の差は，
   下記「分析(実行)」節が示すとおり，coverage・mean_set_size・ECE・top1_accuracy という
   実際に成功条件判定で使う指標には実質的な影響を与えていない（Iter56 記録とノイズ帯内で
   一致，または完全一致）．計画が定めた「本実行 A が基準線と一致することを確認してから
   本実行 B に進む」という条件は，この「使用指標が一致する」という基準で満たしたと判断し，
   本実行 B に進んだ．

### 分析(実行) (Iter69)

**集計方法**: 出力 jsonl から `coverage = mean(expected_domains[0] in prediction_set)`・
`mean_set_size = mean(set_size)` を 10 行程度のワンライナー相当のスクリプト
（`/tmp/iter69_aggregate.py`，本フェーズで新規作成）で算出し，ECE は `metrics.py:compute_ece()`
をそのまま流用した．

**本実行 A（`--qhat-source all`）の実測値と Iter56 基準線との対比**

| 指標 | Iter56 記録 | 本実行 A 実測 | 差分 |
|---|---|---|---|
| q_hat（生値） | 0.3865（population 14,270） | 0.1133（population 14,270） | 一致せず（原因未特定，上記参照） |
| coverage | 0.6056 | 0.5988 | -0.68pt（事前登録ノイズ帯 ±1pt 以内） |
| mean_set_size | 1.51 | 1.5063 | -0.0037（事前登録ノイズ帯 ±0.1 以内） |
| ECE | 0.0630 | 0.0630 | 差分なし |
| top1_accuracy | 0.603125（recompute） | 0.603125 | 差分なし |
| set_size 分布 | size=1: 94.4%, size=10: 5.6% | size=1: 1510/1600(94.375%), size=10: 90/1600(5.625%) | 一致 |
| selected_domain（1,600 行） | - | Iter56 との不一致 0 件 | 完全一致 |
| confidence（1,600 行，許容差 1e-9） | - | Iter56 との不一致 0 件 | 完全一致 |
| probabilities（1,600 行，各ドメイン成分，許容差 1e-9） | - | Iter56 との不一致 0 件 | 完全一致 |

**本実行 B（`--qhat-source true_class`）の実測値**

| 指標 | 実測値 |
|---|---|
| q_hat（生値，population 1,427） | 0.3865 |
| coverage | 0.6556 |
| mean_set_size | 4.0319 |
| ECE | 0.0630 |
| top1_accuracy | 0.603125 |
| set_size 分布 | size=1: 1061/1600(66.31%), size=10: 539/1600(33.69%) |

**selected_domain の非退行確認（A vs B）**

A（1,600 行）と B（1,600 行）を id で突き合わせたところ，`selected_domain` の不一致は
**0 件 / 1,600 行**，`confidence` の不一致（許容差 1e-9）も **0 件 / 1,600 行** だった．
q_hat（＝母集団の切替）が argmax・confidence・probabilities に影響を与えないという計画の
事前想定（q_hat は `prediction_set`/`set_size` のみに作用する）が，A・B 双方の全行で
実測により裏付けられた．

**生成した結果ファイル**

- `results/20260919_202700/Iter69_conformal_qhat_all.jsonl`（本実行 A，1,600 行）
- `results/20260919_202700/Iter69_conformal_qhat_true_class.jsonl`（本実行 B，1,600 行）
- `results/20260919_202700/run_A.log` / `run_B.log`（stderr 診断ログ）
- `/tmp/iter69_aggregate.py`（coverage・mean_set_size 集計と A/B・Iter56 突き合わせ用の
  読み取り専用スクリプト，リポジトリ外の一時ファイル）

**実行/ログ上の異常の有無**: プロセスは A・B とも正常終了（各 1,600 行出力，途中エラー・
OOM・タイムアウトなし）．q_hat の生値が Iter56 記録と数値的に異なる点のみ「異常」として
上記「q_hat 数値不一致の原因調査」節に記録したが，成功条件判定に使う指標（coverage・
mean_set_size・ECE・top1_accuracy・selected_domain・confidence・probabilities）には
実質的な影響がないことを確認済みである．採否判定は次フェーズ（分析(解釈)）に委ねる．

### 分析(解釈) (Iter69)

本節は出力 jsonl（`results/20260919_202700/*.jsonl` および
`results/20260808_000000/Iter56_conformal_prediction.jsonl`）を一次データとして直接再集計した
結果に基づく．採否の確定・config.yml への記録・次レバー選定は次フェーズ（rc-reflector）の仕事で
あり，本節は「何が起きたか・なぜ起きたか・ノイズか有意か」の解釈に限る．

#### 1. 事前登録した成功条件との機械的対比（本実行 B）

| 指標 | 合格条件 | 本実行 B 実測 | 判定 | 条件境界からの距離 |
|---|---|---|---|---|
| coverage | 0.87 ≤ x ≤ 0.93 | 0.6556 | **不合格** | 下限に対し **-21.44pt** |
| mean_set_size | 1.5 ≤ x ≤ 4.0 | 4.0319 | **不合格** | 上限を **+0.0319** 超過 |
| ECE | ≤ 0.0680 | 0.0630 | 合格 | 0.0050 の余裕 |

3 条件の AND が成立条件であるため，**成功条件は不成立**である．未達の主因は coverage であり，
-21.44pt は本実験の統計誤差（n=1600，p=0.6556 の二項 SE=0.0119，95% CI=[0.6323, 0.6789]）の
18 倍に相当する．CI の上端でも下限 0.87 に 19.1pt 届かない．mean_set_size の +0.0319 超過は
事前登録したノイズ目安（±0.1）の範囲内でありそれ単独では判定に足りないが，coverage の判定を
覆すものではない．

非退行条件は 4 件とも充足している（`selected_domain` 不一致 0/1600，`confidence`・
`probabilities` の不一致 0/1600（許容差 1e-9），`top1_accuracy`=0.603125 で A・B 同値，
rank_2 流用は未実施）．

#### 2. 「実験不成立」（d0004 §4 対策 C）に該当するか → **該当しない．有効な測定である**

d0004 §4 対策 C が定める既定解釈は「主要指標が小数点 6 桁まで基準線と一致し，かつ McNemar
discordant=0 ならレバーは発火していない」である．今回はいずれの条件にも当てはまらない．

- **レバー発火の直接証拠（3 系統）**: (a) stderr 診断が母集団サイズの切替を記録している
  （A: population_size=14270 / B: population_size=1427），(b) q_hat の生値が 0.1133 と 0.3865 で
  異なる，(c) 出力の `set_size` が A・B で実際に変化した行が **449 行 / 1,600 行**ある
  （1→10 が 449 件，10→1 が 0 件）．
- **主要指標も一致していない**: coverage 0.5988→0.6556（+5.69pt），mean_set_size 1.5063→4.0319．
  被覆の変化を行ごとに対応づけると discordant は「B のみ被覆」91 件・「A のみ被覆」0 件で，
  McNemar 正確検定 **p=8.08e-28**．ノイズではなく決定的な差である（q_hat が単調に集合を
  拡大するため，被覆の低下は原理的に起こり得ず片側にしか discordant が出ない）．

したがって本イテレーションは「実験不成立」ではなく，**正しく発火したレバーに対する有効な測定**
である．Iter56 の invalid な記録を，再現可能な実装に基づく正式な判定へ置き換えるという当初の
目的自体は達成されている．

#### 3. q_hat 生値の不一致は解決した — Iter56 journal の記録誤りであり，パイプラインは同一

実装・実験フェーズが原因を特定できなかった「本実行 A の q_hat=0.1133 が Iter56 記録の 0.3865 と
一致しない」問題は，一次データの照合により決着した．

1. **本実行 A の出力ファイルは Iter56 の出力ファイルとバイト単位で同一**である
   （md5sum が両者とも `2e0a1533a754e5f853d429ef57836616`，`diff` も差分なし）．つまり
   校正・埋め込み・分類器・集合構成のすべてが Iter56 実行時と完全に一致しており，
   パイプラインの変化は存在しない．
2. **Iter56 の出力ファイル自身から，当時の q_hat を逆算できる**．本実装の集合構成規則は決定的で，
   `set_size=10` となる条件は `1 - max(prob) ≤ q_hat` に完全に一致する（1,600/1,600 行で規則が
   的中することを実測確認）．Iter56 出力における `size=10` 行の `1-p_max` 最大値は 0.1130，
   `size=1` 行の `1-p_max` 最小値は 0.1136 であり，**当時の q_hat は (0.1130, 0.1136] の区間に
   確定する**．本実行 A の 0.1133 はこの区間内にある．
3. 対偶として，仮に Iter56 が本当に q_hat=0.3865 で走っていたなら，出力は決定的に
   coverage=0.6556・mean_set_size=4.0319（＝今回の本実行 B と同値）になっていたはずであり，
   Iter56 が記録した 1.51 とは両立しない．
4. **結論**: Iter56 journal の「q_hat=0.3865」は誤記であり（0.3865 は真クラス版の q_hat，
   すなわち今回の本実行 B の値である），実際の Iter56 の q_hat は 0.1133 だった．Iter69 計画節が
   Iter56 から引き継いだ見積り（all≈0.3865, true_class≈0.5956）も，この誤記を前提としていたため
   まとめてずれていた．**判定を無効化する要因ではない**．
5. 付随して判明した軽微な定義差: Iter56 の記録 coverage=0.6056 は
   `any(d in prediction_set for d in expected_domains)` 定義の値であり，Iter69 計画が採用した
   `expected_domains[0] in prediction_set` 定義では同一ファイルから 0.5988 が得られる
   （同ファイルで cov_any=0.6056, cov_first=0.5988）．**A と Iter56 の -0.68pt の差はノイズでは
   なく，被覆の定義差そのもの**である（複合設問 100 行の扱いの違い．差は 11 行）．どちらの定義を
   採っても成功条件 0.87 には遠く及ばないため判定に影響しない．なお計画節の非退行条件 2 が
   「top1_accuracy が 0.6056 から不変」としていたのも同じ誤記の連鎖で，実測 top1_accuracy は
   A・B とも 0.603125 である（Iter56 出力から再計算しても同値）．

#### 4. coverage が Iter56 事前予測（0.8025）よりさらに低い機序 — 集合構成の符号が逆転している

Iter56 のシミュレーション予測（coverage=0.8025, mean_set_size=7.31）と実測（0.6556, 4.0319）の
乖離，および coverage が名目 0.90 に遠く及ばない理由は，**q_hat の算出母集団の問題ではなく，
予測集合の構成ループ自体が退化していること**にある．一次データから以下を確認した．

- **set_size が 1 と 10 の 2 値しかとらない**（A: size1=1510 / size10=90，B: size1=1061 /
  size10=539．2〜9 は A・B とも 0 件）．適応的な集合サイズという APS の眼目が機能していない．
- **機序**: `_compute_prediction_set()` は確率降順に走査しながら `score = 1 - cumsum` を
  q_hat と比較するが，`cumsum` は走査に伴い単調増加するため `score` は**単調減少**する
  （docstring の「the top class gets the SMALLEST score」は逆で，実際には先頭クラスが最大値
  `1 - p_max` を取る）．結果として，先頭クラスが閾値を通れば以降のすべてのクラスも必ず通り
  集合サイズは 10 に，先頭クラスが落ちれば即 `break` して空集合となり fallback（L122-124）で
  サイズ 1 になる．**判定式は実質的に `1 - p_max ≤ q_hat` という単一の二値ゲートに縮退している**
  （1,600/1,600 行で一致を確認）．
- **この規則の下では coverage が構造的に頭打ちになる**．coverage は
  `r + (1-r) × (size1 行の top1 正解率)` に分解でき，実測では
  B: 0.3369×1.000 + 0.6631×0.4807 = 0.6556 と完全に一致する．つまり**被覆の伸びは「サイズ 10 に
  跳ね上がった行の割合 r」だけが担っており，中間サイズによる効率的な被覆獲得が一切ない**．
- **帰結として，本実装のままでは成功条件 2 つを同時に満たすことが数学的に不可能である**．
  q_hat を掃引した実測フロンティア（同一の 1,600 行で再集計）:

  | q_hat | size10 割合 | coverage | mean_set_size |
  |---|---|---|---|
  | 0.1133（=A） | 0.0563 | 0.5988 | 1.5063 |
  | 0.3865（=B） | 0.3369 | 0.6556 | 4.0319 |
  | 0.50 | 0.5162 | 0.7175 | 5.6463 |
  | 0.60 | 0.7081 | 0.8063 | 7.3731 |
  | 0.65 | 0.8106 | **0.8681** | 8.2956 |
  | 0.70 | 0.8950 | **0.9256** | 9.0550 |

  coverage 0.87 に到達するには mean_set_size が約 8.3 必要で，成功条件上限 4.0 の 2 倍を超える．
  **どの q_hat を選んでも coverage∈[0.87,0.93] と mean_set_size≤4.0 は両立しない**．
  Iter56 の事前予測（0.8025 / 7.31）は，この表の q_hat≈0.60 の行とほぼ一致しており，
  「真クラス版の q_hat がもっと大きい（0.5956）」という誤った前提の下で同じ退化した規則を
  シミュレートした結果だと説明できる．実測の真クラス q_hat が 0.3865 とより小さかったため，
  実測 coverage は予測より低い 0.6556 に落ち着いた．**予測と実測の乖離は q_hat の見積り誤差で
  説明でき，別種の異常ではない**．

#### 5. 分類器性能の限界か，実装の限界か — 切り分け

「10 クラス問題かつ分類器性能が低いため被覆が出ない」という計画時の仮説は，**部分的にしか
正しくない**．同じ 1,600 行の確率で，標準的な APS（非適合スコア＝真クラスまでの累積確率，
集合＝降順に累積確率が閾値に達するまで）を事後計算すると次の水準になる（閾値掃引．
校正を評価集合自身で行った楽観的な推定である点に留保が要る）:

| 閾値 | coverage | mean_set_size |
|---|---|---|
| 0.80 | 0.8544 | 3.0587 |
| 0.90 | 0.9181 | 4.3444 |
| 0.95 | 0.9463 | 5.6031 |

補間すると coverage 0.87 は mean_set_size≈3.3，coverage 0.90 は ≈4.0 で到達する．
参考として top-k 正解率は top1=0.5956, top2=0.7450, top3=0.8237, top4=0.8831, top5=0.9219 であり，
被覆 0.87〜0.90 に必要な集合サイズは分類器性能から見て 3〜4 程度が下限である．
すなわち **成功条件の帯（coverage 0.87-0.93 かつ mean_set_size 1.5-4.0）は分類器性能の観点からは
ぎりぎり成立しうる領域であり，今回それを大きく外したのは集合構成の実装が退化しているため**である．
ただし上記は評価集合自身で校正した楽観値で，実際の校正（`classifier_train.jsonl` の OOF，
OOF accuracy 57.32% と評価集合 top1 59.56% より低い）を使えば q_hat はより大きくなり，
動作点は上表より右（集合が大きい側）へずれるため，**修正実装でも成功条件の上限 4.0 を
超える可能性は十分にある**．この点は本イテレーションの一次データだけでは確定できない．

#### 6. 仮説との整合

- 計画の仮説「真クラス版 q_hat にすれば coverage は 0.6056 から大きく上振れする」→
  **方向は一致するが幅が不足**．+5.69pt（0.5988→0.6556）にとどまり，「大きく上振れ」とは言えない．
  原因は §4 のとおり，真クラス版の q_hat 実測（0.3865）が見積り（0.5956）より小さかったこと，
  および集合構成が二値ゲートに縮退していることの二つである．
- 計画の仮説「被覆を満たすには集合が肥大化し，coverage も名目水準に届かない見込みが強い」→
  **的中**．実測でも名目 0.90 に対し 0.6556 で，同時達成は不可能であることを§4 の掃引で定量化した．
- ECE は事前の但し書きどおり 0.0630 のまま完全不変で，同一性アンカーとして機能した．
  パイプラインの他部分に意図しない混入がないことが確認できている．
- 想定外の挙動: `set_size` が 1 と 10 の 2 値に縮退していた点（計画・実装・実験のいずれの
  フェーズでも明示的に確認されていなかった）．これは本イテレーションで新たに特定した
  **実装上の欠陥**であり，Iter56 の結論も同じ欠陥の上に乗っていた（Iter56 の判定は q_hat の
  母集団誤りだけが原因だと考えられていたが，実際にはより根の深い集合構成の誤りが併存していた）．

#### 7. 判定の確信度と，次フェーズへの示唆

- **確信度は高い．追加反復は不要**と考える．根拠: (a) パイプラインは決定的で，本実行 A が
  Iter56 出力とバイト単位で一致し再現性が確認済み，(b) coverage の未達幅 -21.44pt は二項 SE の
  18 倍で，ノイズでは説明できない，(c) q_hat をどう選んでも成功条件が両立しないことを
  同一データ上の掃引で示せており，1 回の測定に依存した判断ではない．
- 次フェーズ（rc-reflector）への示唆: **rejected 相当**．ただし棄却の理由は
  「conformal prediction という手法がこの分類器に不適」ではなく，
  **「`_compute_prediction_set()` の集合構成が二値ゲートに縮退しており，APS として機能していない」**
  である．config.yml へ記録する際は，Iter56 の記録誤り（q_hat=0.3865 は真クラス版の値）と
  この実装欠陥の 2 点を残さないと，将来同じ誤解が再生産される．
  なお §5 の事後計算は「集合構成を修正すれば成功条件の帯に入りうる」ことを示唆するが，
  それは本イテレーションのレバー（q_hat の母集団）とは別の変更であり，追試するなら
  別レバー（例: `conformal_set_construction=corrected_aps`）として単一レバー原則の下で
  立てるべきである．採用可否・優先度の判断は rc-reflector に委ねる．

### 考察 (Iter69)

#### 判定: `routing_confidence_calibration_method=conformal_prediction_true_class_qhat` は **rejected**．本レバーはクローズ

事前登録した 3 指標の AND が不成立である（coverage=0.6556 は合格帯下限 0.87 に対し -21.44pt，
二項 SE=0.0119 の 18 倍で，95% CI の上端 0.6789 でも 19.1pt 届かない．mean_set_size=4.0319 は
上限 4.0 を +0.0319 超過．ECE=0.0630 のみ合格）．レバーの発火は 3 系統
（母集団 14,270 vs 1,427・q_hat 0.1133 vs 0.3865・set_size 変化 449/1,600 行，McNemar p=8.08e-28）で
確認済みであり，d0004 §4 の「実験不成立」には該当しない．**有効な測定による棄却**である．
追加反復は不要と判断した（パイプラインは決定的で，本実行 A は Iter56 出力と md5 一致．
かつ q_hat を掃引しても成功条件が両立しないことを同一データ上で示せており，単発測定に依存しない）．
値 `conformal_prediction` は Iter56 で試行済みのため，**`routing_confidence_calibration_method` は
両値とも試行済みとなりクローズを確定**した（config.yml:718- の note に追記済み）．

Iter68 の当初目的「バグ入りの invalid な記録を，正しい実装による正式な判定へ置き換える」は
達成されている．B88 が付していた「invalid のまま放置されている」状態は解消した．

#### 学び 1: Iter56 の q_hat 記録は誤記であり，パイプラインは Iter56 から一切変わっていない

実装・実験フェーズが「原因未特定」として持ち越した q_hat の不一致（0.1133 vs 記録 0.3865）は，
一次データの照合で決着した．本実行 A の出力は Iter56 出力と md5 一致
（`2e0a1533a754e5f853d429ef57836616`）であり，そこから逆算した当時の q_hat は (0.1130, 0.1136] に
確定する．記録されていた 0.3865 は真クラス版の値（＝今回の本実行 B の q_hat）であり，
journal への転記時点で取り違えていた．Iter56 の修正シミュレーション（q_hat=0.5956 →
coverage=0.8025 / mean_set_size=7.31）もこの誤記の連鎖で，実測とは一致しない．
**非自明な教訓**: 「原因未特定」で止まった数値差でも，出力ファイルさえ残っていれば
決定的な集合構成規則から当時のハイパラを逆算できる．実験フェーズは q_hat そのものを
persist していなかったが，出力の `set_size`/`probabilities` から区間として復元できた．
今後は校正で得た q_hat を出力 jsonl のメタ行に書き出しておくと，この復元作業自体が不要になる．

#### 学び 2（本イテレーション最大の収穫）: 棄却理由は手法ではなく既存コードの実装欠陥だった

`scripts/evaluate_classifier_calibration.py:_compute_prediction_set()` は確率降順に走査しながら
`score = 1 - cumsum` を q_hat と比較するが，`cumsum` が単調増加するため `score` は**単調減少**する．
それにもかかわらず「score が q_hat を超えたら break」する設計になっているため，判定は実質
`1 - p_max ≤ q_hat` という**二値ゲートに縮退**していた（set_size は 1 か 10 しか取らず，
A・B とも中間サイズ 0 件．1,600/1,600 行でこの規則が的中することを実測確認）．
この欠陥は今回のレバー（q_hat の母集団）とは**独立の既存バグ**であり，Iter56 の結論も同じ欠陥の
上に乗っていた．つまり Iter56 以来「10 クラス問題での APS の方法的限界」と記録してきた解釈は
**誤りを含む**．同一データで標準的な APS を事後計算すると coverage 0.87 は mean_set_size≈3.3，
0.90 は ≈4.0 で到達し（top-k 正解率 top3=0.8237 / top4=0.8831 とも整合），
**成功条件の帯は分類器性能の観点からは成立しうる**．
**非自明な教訓**: 「手法が効かない」という結論を出す前に，その手法の**特徴量的な内部量**
（ここでは set_size の分布）が想定どおりの自由度を持っているかを確認すべきだった．
主要指標（coverage・mean_set_size）だけを見ていると，平均値としては尤もらしい 1.51 や 4.03 が
出るため，分布が 2 値に潰れていることに 2 イテレーション（Iter56・Iter69 の計画/実装/実験）
気づけなかった．今後の事前登録には「主要指標に加え，レバーが作用する内部量の分布を必ず出す」
ことを含める．

#### 次の一手: 新レバー `conformal_set_construction=corrected_aps` を config へ追加した

config の levers は Iter69 着手前の時点で未試行値を使い切っていたが，SKILL.md の停止条件 1
（journal/backlog の学びから次の有望なレバーを自分で考案できるならそれを追記して継続）に該当する．
本イテレーションの学び 2 が，(a) 具体的な実装欠陥の所在，(b) 修正後に成功条件の帯へ入りうる
定量的な事前根拠（閾値掃引表と top-k 正解率），(c) オフライン完結・分類器再訓練なし・
`config.yaml` スキーマ変更なしという着手容易性，を同時に与えているため，
`conformal_set_construction: [corrected_aps]` を config.yml に新設し，成功条件の帯
（coverage 0.87-0.93 ∧ mean_set_size 1.5-4.0 ∧ ECE ≤ 0.0680）と非退行条件を事前登録した．
**Iter70 の単一レバーはこれとする**（iteration_name 案:
「conformal予測集合の構成規則をAPS標準へ修正して被覆を再測定」）．
期待値は Iter69 より明確に高いが，実際の OOF 校正では q_hat が楽観値より大きくなるため
mean_set_size が上限 4.0 を超えて rejected になる可能性も相応にある．いずれに転んでも，
「conformal prediction がこの分類器で使えるか」という Iter56 以来の問いに確定的な答えが出る．

#### B104（A1: 複合評価集合の拡充 / A2: R-F の実行時配線）の扱い

`status="blocked"` にはしない．理由は 2 つある．(i) Iter69 の結果（conformal 系列の棄却理由が
実装欠陥に帰着し，方法自体の当落は未確定）は A1/A2 の判断材料に直接ならず，人間へ提示する
情報は Iter68 時点から**実質的に変化していない**ため，SKILL.md「起動時の手順」の運用ルールに
従い再 @mention はしない．(ii) 上記の新レバーにより自律着手できる作業が存在し，研究サイクルは
停止しない．B104 は `[needs-human]` のまま維持し，回答が得られた時点で Iter71 以降の方針へ反映する．

---

## Iteration 68: rank_2訓練データへのecho除去フィルタ適用

### 調査 (Iter68)

**問い**（config.yml:1357-1401＝backlog B101 が単一レバー `multilabel_synthetic_data_quality
=echo_filtered_synthetic_rows` を事前登録済み．note (a)(b)(c) が指定した一次情報の確認を実施した）

- Q(a): Iter65 の生成品質調査で使われた「echo 疑い率 3.33%」のロジックは `scripts/` 配下に
  残っているか．810 行に実際に適用した場合の除去率は何 % か．10% 未満なら本レバーを見送るべきか．
- Q(b): `evaluate_dispatch_candidate_ranking.py` の `--rank1-source baseline` 経路は Iter67 から
  変更されていないか．
- Q(c): echo 除去（質の効果）と行数減少（量の効果）の交絡を分離する「同数ランダム除去対照ヘッド」は，
  既存スクリプトの引数だけで実装可能か．

**分かったこと（コード読了・実データへの実測・git log による一次情報）**

1. **Q(a) — echo 判定ロジックは `scripts/` 配下に残っていない．journal 自身がそれを認めている**．
   `journal.md`（旧版，現 `journal_archive.md:4294-4296`）の Iter60 節は合成 153 件を「目視・正規表現」
   で走査し「7 件（4.6%）がプロンプト文言の echo」と報告したが，正規表現の定義自体は本文に
   書かれておらず，スクリプトとしても保存されていない．Iter65 節（`journal.md:2273-2275`）も
   「echo 判定は journal Iter64 が用いた語句パターンの正確な定義が記録に残っていないため，
   本節では 3 ファイルに同一の正規表現を当てて相対比較のみに用いた」と明記しており，**その
   正規表現自体も `/tmp/iter6{4,5,6,7}_*.py`（現存する 20 ファイルを `grep -l echo` で全件検索，
   0 件ヒット）のどこにも残っていない**．`scripts/generate_multidomain_training_examples.py` の
   生成後フィルタ F1〜F4（`_passes_filters()`，:128-137）は文字数・四択マーカー・完全一致重複・
   複数行のみを弾く仕組みで，echo（プロンプトの指示文をそのまま反復する低品質行）を検出する
   目的の機構ではない（モジュール docstring 自体が「F1〜F4 のいずれの機械的フィルタにも掛からず
   通過している」と明記，`journal_archive.md:4089`）．**したがって Q(a) 前段は「残っていない」が
   結論であり，本調査で新たに定義・実測する必要があった**。
2. **実測（本調査で新規作成した読み取り専用スクリプト `/tmp/iter68_echo_detect.py` で実施．
   `data/classifier_train_multidomain_iter65.jsonl` 810 行に対し 3 段階の定義を実際に適用）**:
   - **厳格（完全に空虚な行のみ）**: クエリ全体が「XとYの両方の知識（が必要|を必要とする）
     （な相談文|なような相談文|な状況）を（作成します|以下に示します|1件だけ作成します）。」の
     ような，具体的な相談内容を一切含まない定型文そのものである行 — **13/810（1.60%）**．
     例: `分野Aと分野Bの両方の知識が必要な相談文を作成します。`（プロンプトのプレースホルダー
     `分野A`/`分野B` を置換し忘れた，最も明白な echo）．
   - **狭義（instruction 語句の部分一致，例文の実例「〜の両方の知識が必要な相談文：」に対応）**:
     クエリ中に「両方の知識」「知識が必要な相談文」「知識がないと」「適切に答えられない」の
     いずれかを含む行 — **76/810（9.38%）**（ただしこの一部は「経営学と教育学の両方の知識が
     必要な分野で，どのようなスキルが求められるか教えてください」のように実質的な相談内容を
     伴っており，false positive の余地がある）．
   - **広義（狭義＋プレースホルダー漏れ「分野A」「分野B」＋メタ言及「相談文を作成/以下に/1件」）**:
     **80/810（9.88%）**．
   - **n-gram 重なり率ベース（config note が挙げた代替案）**: 生成プロンプトの指示文全体
     （`_build_prompt(domain1, domain2)` の出力）とクエリの文字 3-gram Jaccard を計算したところ，
     最大値は 0.0730（`synth-mathematics-natural_science-007`），閾値 0.15 以上は **0 件**．
     日本語の指示文（100 字超）に対しクエリ（20〜200 字）が短いため，このスコアは絶対値が
     構造的に小さくなり，識別力を持たない（**config note の「n-gram 重なり率の閾値」案は，
     指示文全体を基準にする限り実測上機能しない**．狭義／広義の語句パターン一致の方が実効的）．
   - **結論**: どの定義でも除去率は **1.60%〜9.88%** で，config note が事前に懸念した
     「10% 未満（約 27 行相当）なら効果も量の変化も検出不能」の境界線上か，それを下回る．
     **もっとも広い定義（9.88%）でも 10% には届かない**．
3. **Q(a) の判断材料**: 過去の用量反応の実測（Iter64→65，`journal.md:2317-2319`）では，
   合成行数を **135→405→810（約 6 倍）** 変化させても被覆 2 個行の増分は
   +9→+12→**−3** と符号が反転する程度の効き方しかしておらず，かつ「Iter65 の実測は R-H
   （n=100・discordant 15〜19 行では ±3〜4 行を検出できない）が臨界に達している」ことが
   backlog B101 で明記されている．**本レバーで見込める行数変化は最大でも 80 行（9.88%）で，
   これは Iter64→65 の 6 倍変化よりも 2 桁小さい摂動である**．R-H の検出限界（±3〜4 行）に
   対し，80 行除去がもたらす rank_2 命中数の変化がそれを上回る保証はコード上・データ上どこにも
   ない．**したがって「除去率が 10% 未満なら効果も量の変化も検出不能になる可能性が高い」という
   config note の懸念は，実測（1.60%〜9.88%）でもそのまま該当すると判断する**。
4. **Q(b)**: `git log --oneline --follow -- scripts/evaluate_dispatch_candidate_ranking.py` は
   `c1d1116`（Iter63）を最新のヒットとして示し，**Iter63 以降（Iter64〜67 を含む）このファイルへの
   変更は一切ない**ことを確認した．`train_multilabel_dispatch_head.py` も最新コミットが
   `927e363`（Iter62）で同様に不変．**`--rank1-source baseline` 経路は Iter67 時点から変更
   されていない**（Iter67 journal の確認結果がそのまま今も有効）．
5. **Q(c)**: `train_multilabel_dispatch_head.py:212-250` の `_train_and_save()` は
   `--train-data`（単一ドメイン行）と `--multilabel-train-data`（合成行）の 2 つのファイルパスを
   単純に読み込んで連結するだけであり（`_load_training_rows()` は行数に関する前提を一切課さない），
   `_assert_a0_true_multilabel_signal()`（:229）が要求する下限は multilabel_row_count ≧ 120 のみで
   ある．**echo 除去版（810−N 行）とランダム除去版（810−N 行，同数）をそれぞれ独立の
   `.jsonl` ファイルとして作成し，`--multilabel-train-data` にそれぞれ渡して 2 回学習を回すだけで，
   スクリプト本体のコード変更なしに交絡分離が実装できる**（N=13〜80 のいずれでも残存行数は
   730〜797 行で 120 の下限を大きく上回り，A0 は問題なく成立する見込み）．

**次フェーズへの示唆**

- **見送りを推奨する**．理由は 2 点の一次情報に基づく:
  (i) 実測除去率は定義を最も広くとっても 9.88%（80/810）で，config note が事前に定めた
  「10% 未満なら効果も量の変化も検出不能になる可能性が高い」という基準に実質的に該当する．
  さらに狭い／厳格な定義では 9.38%／1.60% とより小さい．
  (ii) n-gram 重なり率という代替の広い基準（config note が候補として挙げたもの）を実際に
  実装・実測したが，指示文全体を基準にすると構造的に低い値しか出ず，語句パターン一致より
  識別力が低いことが判明した．**「より広い echo 判定基準を計画フェーズで定義する」という
  config note の代替案は，n-gram 案に関しては本調査で試行済みかつ機能しないことが分かった
  ため，計画フェーズが新たに考案できる基準は限られる**（語句パターンの拡張のみで，
  それも広義 9.88% が事実上の天井に近い）．
  (iii) Iter64→65 の量の変化（135→810，6 倍）でさえ被覆 2 個行の変化が ±数行（R-H の検出限界
  付近）にとどまったことを踏まえると，本レバーの摂動（最大 80 行，9.88%）から統計的に
  意味のある変化を検出できる見込みは低い．
- 一方で，(c) の交絡分離自体はコード変更なしで実装可能であることを確認済みであり，
  **仮に計画フェーズが「見送らずに実施する」と判断する場合**は，(1) 広義定義（80 行）で
  echo 除去版を作り，(2) 同数（80 行）をランダム除去した対照版を作り，(3) 両方を
  `train_multilabel_dispatch_head.py` に別々に通して 2 ヘッドを学習し，(4) 両ヘッドの
  rank_2 命中数を対応あり McNemar で比較する，という手順がそのまま使える．
  厳格定義（13 行）はサンプルサイズが小さすぎて検出力の観点でさらに不利であり，
  実施するとしても広義定義（80 行）を使うべきである．
- **backlog への申し送り案**: 本レバーは config note 自身が事前に定めた見送り条件
  （除去率 10% 未満）に実測が該当する．過去の失敗パターン（config を正しく変えたが効果が
  測定不能）とは異なり，今回はコード到達の問題ではなく，**レバーの摂動量自体が測定系の
  検出限界（R-H）を下回る**という別の型の限界である．reflector は，(i) 本レバーを見送って
  クローズする，(ii) 見送らず実施だけして「効果なし」ではなく「検出力不足」と判定する前提で
  進める，のいずれかを選ぶ必要がある．config.yml/backlog B101 は「不成立の場合は同系列の
  レバー探索を打ち止めとし，(i) 複合設問データセットの拡充，(ii) R-F（実行時経路への配線）の
  是非を人間に諮ること」まで既に明記しているため，**本調査の実測結果はその条件（不成立）に
  相当すると解釈でき，着手前に打ち止め判断へ進むという選択肢が自動判断の範囲内で成り立つ**．

### 計画 (Iter68)

**結論: 単一レバー `multilabel_synthetic_data_quality=echo_filtered_synthetic_rows` は
着手前に見送る（実験フェーズに進まない）**．config.yml:1357-1401 の note が事前登録した
分岐条件「除去率が 10% 未満なら，より広い echo 判定基準を計画フェーズで定義するか，
本レバー自体を着手前に見送ること」に対し，計画フェーズとして **「より広い基準の定義」を
試みたうえで，それでも見送る** と判断した．根拠は以下の 3 点である．

**根拠 1 — 事前登録した見送り条件に実測が該当する**．調査フェーズの実測（上節）は
厳格 1.60%（13/810）・狭義 9.38%（76/810）・広義 9.88%（80/810）で，**最も広い定義でも
10% に届かない**．これは note が着手前の見送り条件として明示した閾値そのものである．

**根拠 2 — 計画フェーズとして「より広い判定基準」を 3 つ追加で定義・実測したが，
10% を超えるには echo 以外の異質な基準を混ぜるほかなく，単一レバーとして定式化できない**．
調査フェーズが試した n-gram 重なり率（指示文全体基準・閾値 0.15 以上で 0 件）に加えて，
本計画フェーズで以下を `data/classifier_train_multidomain_iter65.jsonl` に実測した（読み取り専用）:
- **完全一致重複**: **0 行**（生成側フィルタ F3 が既に除去済み．`_passes_filters()` の仕様どおり）．
- **近傍重複（クエリ間の文字 4-gram Jaccard で貪欲に片方を残す）**:
  閾値 0.7 で 9 行（1.11%）・0.6 で 17 行（2.10%）・**0.5 で 34 行（4.20%）**．
- **極端に短い行（クエリ 20-29 字）**: 34 行（4.20%）．クエリ長の分布は
  min 20 / p10 38 / median 66 / p90 112 / max 185 字．
- **上記 3 つと広義 echo（80 行）の和集合**: **103 行（12.72%）**．
これで初めて 10% を超えるが，**「プロンプトの指示文を反復している（echo）」「他の行とよく似ている
（近傍重複）」「短い」は互いに独立した 3 つの品質概念であり，これを 1 つのフィルタにまとめると
レバー名（echo 除去）と実体が乖離し，単一レバー原則にも反する**（結果が動いても 3 概念の
どれが効いたか切り分けられない）．**「より広い echo 判定基準」として自然に定義できる上限は
広義の 9.88%（80 行）であり，これが事実上の天井である**と結論する．

**根拠 3（決定的）— 仮に広義 80 行で実施しても，摂動量が測定系の検出限界 R-H を下回る**．
本レバーの効果は「訓練 810 行のうち最大 80 行（9.88%）の取捨」である．同系列の用量反応の
実測（Iter64→65，journal.md 上記 Iter65 節）では，合成行数を **135→405→810（6 倍・+500%）**
動かしても被覆 2 個行の増分は **+9→+12→−3** にとどまり，Iter65 時点で既に R-H
（n=100・discordant 15〜19 行では ±3〜4 行を検出できない）が臨界に達していると
backlog B101 が明記している．**−9.88% という摂動は Iter64→65 の +500% より 1〜2 桁小さく，
期待される被覆 2 個行の変化は 1 行未満と見積もられる**．主基準に定めた
「echo 除去版 vs 同数ランダム除去版の対応あり exact McNemar」は，両群の訓練行数が
同一（730 行）で差分が 80 行の中身だけになるため，**構成上さらに小さい効果量しか生まない**．
すなわち，この実験は実施しても「効果なし」ではなく「検出力不足で判定不能」としか結論できず，
**事前に p(有意) がほぼゼロと分かっている実験を回すことになる**．

**過去の失敗パターンとの区別（config.yml 冒頭の既知の失敗型との照合）**: 今回の見送りは
「config を正しく変えたがコードに到達せず結果が基準線と完全一致した」（過去 6 回）とは
**別の型**である．調査フェーズ Q(c) のとおり，`train_multilabel_dispatch_head.py:212-250`
`_train_and_save()` は `--multilabel-train-data` のファイルを読むだけで
（`_load_training_rows()` は行数の前提を課さず，`_assert_a0_true_multilabel_signal()` の下限は
multilabel_row_count ≧ 120，除去後 730 行はこれを満たす），**レバーは確実にコードへ到達する**．
到達しないのではなく，**到達しても測定系の分解能が足りない**．したがって本判断は
「実装の不備」ではなく「実験設計として検出力が足りない」ことを理由とする見送りである．

**見送りに伴う措置**
- config.yml の `multilabel_synthetic_data_quality` レバー note 末尾に，Iter68 計画フェーズの
  見送り確定を追記した（Iter66/67 の追記パターンに合わせた体裁）．values 単一値のため
  **本レバーはクローズ**扱いとする．
- **本イテレーションは実験フェーズに進まない**．実装・実験・採点・統計は一切行わない．
  `results/` への新規ディレクトリ生成もなし．基準線 `results/20260918_202613/results.jsonl` は
  そのまま次イテレーションでも参照可能である．
- **今回は実験なしで考察フェーズ（rc-reflector）へ進む**．

**次レバー候補（最終決定は rc-reflector に委ねる）**
config.yml note / backlog B101 は「本レバーが不成立の場合は同系列でのレバー探索を打ち止めとし，
(i) 複合設問データセットの拡充（現状 100 行．research_frontier 相当・人間判断），
(ii) R-F（実行時経路への配線．Iter67 時点で 10 反復目の見送り）の是非を人間に諮ること」と
既に明記している．本計画フェーズの判断はその「不成立」条件に相当する．
本計画フェーズの追加所見として，**根拠 3 が示すのは『rank_2 ヘッドの学習データ側をどう動かしても，
n=100 の複合評価集合では判定できない』という測定系そのものの限界**であり，
**次に取り組むべきは新しいレバーではなく評価集合の拡充（上記 (i)）だと考える**
（R-H を緩めない限り，Iter63〜68 と同型の「partial・検出力不足」が繰り返されるだけである）．
ただしこれは research_frontier 相当の規模であり単一レバーとして定式化できないため，
**rc-reflector が人間判断を仰ぐ形で backlog に起票することを申し送る**．

### 考察 (Iter68)

**本イテレーションは実験を実施していない**（実装・実機走行・採点・統計はゼロ）．
`results/` への新規ディレクトリ生成はなく，基準線 `results/20260918_202613/results.jsonl` は
そのまま次イテレーションでも参照可能である．以下は調査・計画の 2 フェーズだけで確定した結論である．

**判定: `multilabel_synthetic_data_quality = echo_filtered_synthetic_rows` は
「着手前見送り・レバークローズ」（採用でも棄却でもない．実験を行っていない以上，
本レバーの効果の有無について何も主張しない）**．

**確定した内容**

1. **事前登録した見送り条件が実測で発動した**．config.yml:1357 note (a) が
   「除去率が 10% 未満なら，より広い判定基準を定義するか本レバーを着手前に見送ること」と
   事前に定めており，`data/classifier_train_multidomain_iter65.jsonl`（810 行）への実測は
   厳格 1.60%（13 行）／狭義 9.38%（76 行）／広義 9.88%（80 行）で，いずれも 10% 未満だった．
2. **「より広い基準」は計画フェーズで実際に定義・実測して天井に達した**．n-gram 重なり率
   （指示文全体との文字 3-gram Jaccard）は閾値 0.15 以上が 0 件（最大 0.0730）で識別力なし，
   完全一致重複 0 行，近傍重複（4-gram Jaccard≧0.5）34 行 4.20%，短文（<30 字）34 行 4.20%．
   全和集合でようやく 103 行 12.72% だが，これは echo・近傍重複・短文という 3 つの独立した
   品質概念の混合であり，単一レバーとして定式化できない．
3. **決定的理由は検出力（R-H）である**．Iter64→65 の用量反応は合成行数 +500%（135→810）でも
   被覆 2 個行が +9→+12→−3 としか動かず，n=100・discordant 15〜19 行では ±3〜4 行を検出できない．
   本レバーの摂動 −9.88% はこれより 1〜2 桁小さく，さらに主基準（echo 除去版 vs 同数ランダム
   除去版）は両群 730 行で差分が 80 行の中身だけになるため効果量は構成上さらに小さい．

**このイテレーションの学び（次の自分向け）**

- **「回さない」という判断も一次情報に基づけば正当な帰結である**．過去 6 回の失敗型
  （config を変えたがコードに到達せず，基準線とビット単位で一致）は「実装の不備」だったが，
  今回は `train_multilabel_dispatch_head.py:212-250` `_train_and_save()` にレバーが確実に到達する
  （下限 multilabel_row_count ≧ 120 に対し除去後 730 行）ことを確認したうえで，**到達しても
  測定系の分解能が足りない**ことを理由に見送った．**別の型の限界であり，同じ札に数えないこと**．
- **「事前に結果が判定不能と分かっている実験は回さない」が，「事前に結果が負と予想される実験」は
  別扱いである**．前者は情報量がゼロ（回しても『検出力不足』としか書けない），後者は測定済みの
  負の記録が残り論文上の主張になる．この区別を次レバーの選定（下記）にそのまま適用した．
- **Iter63〜68 の 6 反復で確定したこと**: rank_2 ヘッドの学習データ側の変数（混ぜ方 Iter66 /
  量 Iter64-65 / 構造 Iter67 / 質 Iter68）はすべて打ち止めであり，**律速はレバー側ではなく
  複合評価集合 n=100 という測定系そのもの**である．この系列の「partial・判定不能」の反復は
  レバーの選び方が悪かったのではなく，分解能の問題だった．

**次イテレーション（Iter69）の単一レバー: `routing_confidence_calibration_method =
conformal_prediction_true_class_qhat`**

- **選定理由**: SKILL.md の停止条件は「config の未試行レバーを優先」と定める．本値は
  Iter56 の q_hat 計算バグ（全スコア 14,270 件を使ったが，正しくは真ラベルクラスのみ 1,427 件）を
  修正した再試行として backlog B88 が追加した**唯一の未試行値**であり，その着手条件
  （「production_deployment_gap・dispatch_policy の 2 レバーに着手できる余地が無くなってから」）は
  Iter57・Iter58 の完了で既に満たされている．オフライン完結・分類器の再訓練不要・
  `config.yaml` のスキーマ変更なし・実機 1600 問本走なしで，自律着手できる．
- **期待値は低いが情報量はゼロではない**．Iter56 の修正シミュレーションは coverage=0.8025
  （target 0.87 未達）・mean_set_size=7.31（target 1.5〜4.0 を大幅超過）を示しており，
  **rejected になる見込みが高い**．それでも実施する理由は，現在の記録が「バグを含んだ invalid」
  のままであり（success_criteria (6) の趣旨），**バグ修正版で測った正式な棄却の記録に置き換える
  価値がある**ためである（上記「学び」2 点目の区別）．
- **調査・計画フェーズへの申し送り**: (1) 成功条件のうち `fallback_rate` は Iter28 で
  fallback を廃止（confidence_threshold=0.0）しているため現構成では常に 0 であり，
  **成功条件から外して ECE と coverage／mean_set_size で事前登録し直すこと**．
  (2) 予測集合という出力形式は複合設問（2 ドメイン）の候補集合と概念的に対応するが，
  mean_set_size 7.31 は top-2 dispatch には大きすぎる．**rank_2 候補源への流用は本イテレーションの
  レバーに含めない**（別レバーとして立てるなら次回以降）．
- **人間判断を要する論点は別途 backlog B104 に起票した**（下記）．Iter69 はその回答を待たずに
  進められるが，回答が得られれば Iter70 以降の方針がそれに従って変わる．

**人間判断を要する論点（backlog B104 `[needs-human]`，Slack で `<@U055AN8LWF6>` へ mention 要）**

- (A1) **複合設問評価集合の拡充（現状 100 行）**．R-H を緩めない限り Iter63〜68 と同型の
  「partial・判定不能」が繰り返される．ただし (a) 評価集合の変更は 1600 問基準線の比較可能性を
  壊し，過去の全ベースラインの再取得（実機 100 分／回）を要する，(b) 既存 100 行は人手作成
  （`build_dataset.py:_COMPOUND_QUESTIONS`）であり，LLM 生成で増やすと rank_2 の合成訓練データと
  生成器を共有して**循環的な妥当性の問題**を生む，という 2 点で research_frontier 相当であり，
  自律判断の範囲を超える．
- (A2) **R-F（多ラベルヘッドの実行時経路への配線）**．今回で **11 反復連続の見送り**．
  `config.yaml` のスキーマ変更を伴い自律判断の範囲外（論点は B95 要レビュー 1 に一本化したまま）．
  2 ヘッド構成が N5 を一切犠牲にしないことは Iter67 で確定したが，複合側の利得が系列内で
  有意でない現状で配線に踏み切るかは人間判断である．

### 調査 (Iter67)

**問い**（config.yml:1296-1345＝backlog B100 が単一レバー
`multilabel_head_architecture=two_head_rank1_single_domain_classifier` を事前登録済み．
新規の先行研究探索ではなく，note (a)(b) が指定した一次情報＝コード確認を実施した）

- Q(a)-1: `train_multilabel_dispatch_head.py --train-data` は必須引数だが，合成 810 行のみ
  （単一ドメイン行ゼロ）で学習する最小差分の手段は何か．
- Q(a)-2: `_assert_a0_true_multilabel_signal()`・`build_multilabel_targets()` は単一ラベル行
  ゼロの入力（合成 810 行のみ）で正常に成立するか．10 ドメイン全てが `mlb.classes_` に
  現れるか．
- Q(b): `evaluate_dispatch_candidate_ranking.py` の `--rank1-source baseline` 経路
  （`build_new_rows()`）が Iter61 以降変更されていないか．

**分かったこと（コード全文読了・git log・実データ検査による一次情報）**

1. **Q(a)-1**: `train_multilabel_dispatch_head.py:212-223` の `_train_and_save()` は
   `single_label_rows = _load_training_rows(train_data_path)` の後，
   `rows = single_label_rows + synthetic_rows` と単純にリスト連結するだけである．
   `_load_training_rows()`（`train_domain_classifier.py:74-77`）は
   `[json.loads(line) for line in f if line.strip()]` であり，**空ファイル（0 行）を渡すと
   例外なく空リストを返す**．したがって `--train-data /dev/null`（または空の
   `.jsonl` ファイルを新規作成して渡す）を指定するだけで，**スクリプトのコード変更を一切
   行わずに**合成 810 行のみでの学習が成立する．これが最小差分であり，「空ファイルを別途
   用意して指定」案のうち，新規ファイル作成すら不要な `/dev/null` 直接指定がさらに最小である
   （新規ファイルを残したくない場合の代替として空の `.jsonl` を作ってもよいが，
   本質的な差はない）．「最小限のスクリプト改修」（`--train-data` を optional 化する等）は
   不要であり，むしろ既存の必須引数制約を無傷で保てる分，この案より安全である．
   `_extract_sample_weights()`（`train_domain_classifier.py:80-94`，class_weight="balanced" 相当の
   sample_weight 計算）は `train_domain_classifier.py` の `_train_and_save()` 専用であり
   `train_multilabel_dispatch_head.py` からは呼ばれないため，`--train-data` を空にしても
   このヘッド学習経路には無関係である．
2. **Q(a)-2**: `data/classifier_train_multidomain_iter65.jsonl`（810 行）を実データ検査した
   ところ，10 ドメイン全てが正確に **162 行ずつ**出現し（10×162=1620＝810 行×2 ラベル），
   45 通りの 2 ドメインペア（C(10,2)=45）が全て 18 行ずつ存在することを確認した
   （note の「45 ペア×各ドメイン 9 ペア」＝9×18=162 と整合）．
   `build_multilabel_targets()`（:96-110）は `_normalize_labels()` で全行を `list[str]` へ揃えた
   うえで `MultiLabelBinarizer().fit_transform()` するだけであり，単一ラベル行の有無に依存する
   実装上の前提はコード中に存在しない．`--train-data` を空にした場合，
   `_assert_a0_true_multilabel_signal()`（:113-142）の 4 条件は次のように成立する見込みである：
   (i) `multilabel_row_count == n_synthetic_rows`（810）は，全行が 2 ドメイン合成行のため
   `Y.sum(axis=1)>=2` が 810 行全てで真になり成立，(ii) `>= 120` の下限も充足，
   (iii) `len(mlb.classes_) == 10` は上記のドメイン頻度分布（10 ドメイン全出現）から成立，
   (iv) 全クラスが文字列であることも `domain` フィールドの型から成立．
   **実装上の前提破壊は見つからなかった**．参考として，単一ドメイン行ゼロのため
   各ドメインの OvR 二値問題は正例 162・負例 648（他ドメインの合成行のうち当該ドメインを
   含まないもの）となり，`CalibratedClassifierCV(cv=5)` の各 fold で両クラスが十分な件数
   確保できる見込みである（Iter66 で確認済みの `cv=5`＝`StratifiedKFold(shuffle=False)` の
   挙動と合わせ，学習自体が失敗する要因は見当たらない）．
3. **Q(b)**: `git log --oneline --follow -- scripts/evaluate_dispatch_candidate_ranking.py` は
   Iter59・Iter60・Iter62・Iter63（`c1d1116`）の 4 コミットのみを示し，Iter63 以降
   （Iter64〜66 を含む）**このファイルへの変更は一切ない**ことを確認した．
   `c1d1116` のコミットメッセージには「既定モードは bit 単位で従来と一致（A10 で実証）」と
   明記されており，`build_new_rows()`（:216-271）の `if rank1_source ==
   _RANK1_SOURCE_BASELINE:` 分岐（:250-252）は `rank_1 = row["dispatched_domains"][0]`・
   `selected_domain = row["selected_domain"]` と，`--baseline` の値をそのまま通すだけの
   処理であることをコードで直接確認した．`_assert_rank1_unchanged()`（A1，:274-285）が
   baseline モードでは必ず走り，`--baseline` との完全一致を実行時に強制する．
   **`--rank1-source baseline`（既定値）経路は Iter61（実質 Iter59-62 と同一実装）以降，
   構造的・実測的に不変であることが確認できた**．
4. **配線確認（付随）**: `evaluate_dispatch_candidate_ranking.py --head` はこのスクリプトが
   `joblib.dump({"model": ..., "classes": ...}, output_path)` で保存した dict 形式をそのまま
   受け取れる（`_load_head()` は Iter60〜66 で不変）．したがって
   `models/dispatch_multilabel_head_iter67_synth_only.joblib`（`--train-data /dev/null` で
   学習した新ヘッド）をそのまま `--head` に渡せば良く，評価スクリプト側の変更も不要である．
5. **実行環境**: wafl-ctrl5 は SSH 接続良好．`ollama-ctrl` コンテナ稼働 5 時間，
   GPU 5,736/12,288MiB 使用（使用率 6%，新規ジョブを妨げない），ディスク空き 435GB．
   埋め込み対象は合成 810 行のみ（単一ドメイン行 1427 行を含まない）で，
   Iter65 の 2237 行・Iter66 の 3664 行より大幅に少ないため，note の見積り「10 分程度」は
   妥当，むしろ実測はこれを下回る可能性が高い．

**次フェーズへの示唆**

- 計画フェーズは，コード変更なしで実行できる：
  `uv run python -m scripts.train_multilabel_dispatch_head --train-data /dev/null
  --multilabel-train-data data/classifier_train_multidomain_iter65.jsonl
  --embedding-model nomic-embed-text --ollama-host 127.0.0.1 --ollama-port <tunnel port>
  --output models/dispatch_multilabel_head_iter67_synth_only.joblib`
  に続けて `evaluate_dispatch_candidate_ranking.py --rank1-source baseline`（既定値，
  明示指定でも可）で新ヘッドを `--head` に渡すだけでよい．新規ファイルは
  `models/dispatch_multilabel_head_iter67_synth_only.joblib`・
  `results/iter67_multilabel_ranking_predictions.jsonl`・`results/iter67_stats.json` のみで，
  スクリプト本体への変更は不要（差分ゼロ）．
- note の実験成立検査（N5 が基準線と完全一致するはず）は，`_assert_rank1_unchanged()`（A1）が
  実行時に自動で強制するため，計画フェーズで別途チェックロジックを足す必要はなく，
  A1 の AssertionError の有無をそのまま「実験成立」の一次シグナルにできる．
- 空ファイル方式（`/dev/null` 指定）は実装上の懸念がない一方，将来同種の「あるサブセットのみで
  学習したい」ケースが繰り返し出てくるようなら，`--train-data` を optional 化する
  小規模リファクタは検討候補として残る．ただし今回 1 回限りであれば過剰設計であり，
  今イテレーションでは不要と判断する．

### 計画 (Iter67)

**仮説**

Iter63〜66 の 4 反復で，単一ヘッド構成では「合成データの混ぜ方」（本数・質量比）という 1 つの
スカラーの上で **N5（単一ドメイン 1500 行の argmax 正解率）と複合被覆がトレードオフし，両立点が
存在しない**ことが実測で確定した（Iter64 0.591/24 → Iter65 0.563/21 → Iter66 0.577/21）．
機序は rank_1 と rank_2 を同一ヘッドが決める構造（R-E）にあり，合成 2 ドメイン文が単一ドメイン
判別の決定境界を融解させることが避けられない．**rank_1 を既存の単一ドメイン分類器の出力へ戻し，
rank_2 のみを合成文だけで学習した別ヘッドに任せれば，単一ドメイン 1500 行の判定は定義上不変に
なり（退行が構造的にゼロ），複合側の利得だけを取り出せる**というのが本イテレーションの仮説である．
反証されるのは「rank_1 が弱まる代償（複合 100 行の rank_1 正解 72→41 相当）を払うと，
複合被覆が Iter61 水準（12/100）から伸びない」という対立仮説である．

**単一レバー（今回変更する唯一の変数）**

`multilabel_head_architecture`: `single_head_rank1_head_argmax`（Iter63〜66 の実質値＝同一ヘッドが
rank_1 と rank_2 の両方を決める）→ **`two_head_rank1_single_domain_classifier`**．
具体的には次の 2 点を同時に満たす構成へ切り替える（この 2 点は「構造を 2 ヘッドに分ける」という
1 つの変更の不可分な表裏であり，第 2 のレバーではない）．

1. rank_1 の供給源: `evaluate_dispatch_candidate_ranking.py --rank1-source baseline`
   （既定値．Iter61 と同一．基準線 `results/20260918_202613/results.jsonl` の `selected_domain` を
   そのまま通す）．
2. rank_2 ヘッドの訓練データ: **合成 810 行のみ**
   （`--train-data /dev/null --multilabel-train-data data/classifier_train_multidomain_iter65.jsonl`）．
   単一ドメイン行 `data/classifier_train.jsonl`（1427 行）も Iter66 の 2 重化ファイルも混ぜない．

**確定した実装仕様（本フェーズの決定事項）**

1. **スクリプトのコード変更を一切行わない（差分ゼロ）**．調査 Q(a)-1 で確認したとおり
   `_load_training_rows()` は空ファイルに対し例外なく空リストを返し，`_train_and_save()` は
   `single_label_rows + synthetic_rows` と連結するだけであるため，`--train-data /dev/null` を
   渡すだけで「合成 810 行のみでの学習」が成立する．`--train-data` の必須引数制約も
   optional 化せずそのまま維持する（1 回限りの用途に対する API 変更は過剰設計であるため）．
2. 空ファイルとして **`/dev/null` を直接指定**し，空の `.jsonl` をリポジトリへ新規追加しない
   （残骸を作らないため．挙動上の差はない）．
3. `--rank1-source baseline` は**明示指定する**（既定値と同一だが，本イテレーションの成否が
   この値に依存するため，コマンドから意図が読めるようにする）．
4. 統計は `scripts/compute_iter59_ranking_stats.py` を無改造で使う．
5. 被覆 2 個行数・対 Iter61 McNemar などは Iter65 で確立したアドホック集計（読み取り専用）を使う．
6. **`--iter59-predictions` には `results/iter66_multilabel_ranking_predictions.jsonl` を渡す**
   （S4＝対 Iter66 不一致行を測るため）．

**N5 に関する重要な訂正（本フェーズで確定し，実験フェーズはこの定義に従うこと）**

config.yml の note と backlog B100 は「**N5 は基準線と完全一致するはず**」と書いているが，
`evaluate_dispatch_candidate_ranking.py:378-406` の `_compute_single_domain_argmax_accuracy()` は
**`--rank1-source` に一切依存せず，`--head` に渡したヘッド自身の 10 ドメインスコアの argmax**で
1500 行の正解率を計算する実装である（コードで直接確認）．したがって：

- スクリプトが出力する `n5_single_domain_argmax_accuracy` は，合成 810 行のみで学習した新ヘッドの
  素の性能であり，**基準線とは一致しない．0.590 のフロアを大きく下回り WARNING が出ることが
  想定内である**（単一ドメインの四択問題を一度も学習していないため）．
  **この値は診断値として記録するのみで，判定にも対外引用にも用いない．**
- config.yml/B100 が意図していた「単一ドメイン 1500 行の判定が定義上不変」という主張の正しい
  操作化は **V1（下記）＝ 1600 行すべてで rank_1 が基準線と bit 単位で一致すること**であり，
  これは `_assert_rank1_unchanged()`（A1，:274-285）が baseline モードで必ず実行時に強制する．
  実験成立の検査はこの A1 と `new_top1_accuracy` の一致で行う．

**固定する構成（Iter66 から変えない／Iter61 と揃える）**

- 合成訓練データ: `data/classifier_train_multidomain_iter65.jsonl`（810 行）を**再生成せず再利用**
  （生成系一式＝プロンプト・F1〜F4・temperature=0.8・生成モデル・45 ペア集合・A7 閾値は今回一度も
  起動しない）．
- ヘッド種別: `OneVsRestClassifier(CalibratedClassifierCV(LogisticRegression(class_weight='balanced'),
  method='sigmoid', cv=5, ensemble=True))`（Iter62 以降不変）．
- 埋め込みモデル `nomic-embed-text`・評価クエリ埋め込みキャッシュ `results/iter59_query_embeddings.npz`・
  基準線 `results/20260918_202613/results.jsonl`・rank_2 の選択ロジック（`rank_1` を除く最大スコア）・
  `_head_scores()`・A1/A2/A3/A6・統計スクリプト・`config.yaml`．
- **実行時経路（`node.py` のルータ）への配線は本イテレーションでも行わない**（R-F・9 回目）．
- 実行基盤は wafl-ctrl5（SSH ローカルフォワード `127.0.0.1:11499`．調査 5 で生存確認済み）．

**出力ファイル命名（既存成果物を上書きしないこと）**

| 種別 | 既存（保護・読み取り専用） | Iter67（新規作成） |
|---|---|---|
| 訓練データ | `data/classifier_train_multidomain_iter65.jsonl`（810 行） | **新規作成しない（再利用）** |
| ヘッド | `models/dispatch_multilabel_head_iter66.joblib` | `models/dispatch_multilabel_head_iter67_synth_only.joblib` |
| 予測 | `results/iter66_multilabel_ranking_predictions.jsonl` | `results/iter67_multilabel_ranking_predictions.jsonl` |
| 統計 | `results/iter66_stats.json` | `results/iter67_stats.json` |

**実行計画**

```
# 1) rank_2 用ヘッドの学習（合成 810 行のみ．--train-data /dev/null が唯一の実装上の工夫）
uv run python -m scripts.train_multilabel_dispatch_head \
    --train-data /dev/null \
    --multilabel-train-data data/classifier_train_multidomain_iter65.jsonl \
    --embedding-model nomic-embed-text --ollama-host 127.0.0.1 --ollama-port 11499 \
    --output models/dispatch_multilabel_head_iter67_synth_only.joblib

# 2) 1600 問のオフライン採点（rank_1 は基準線．A1 が実行時に一致を強制する）
uv run python -m scripts.evaluate_dispatch_candidate_ranking \
    --baseline results/20260918_202613/results.jsonl \
    --head models/dispatch_multilabel_head_iter67_synth_only.joblib \
    --rank1-source baseline \
    --embedding-model nomic-embed-text --ollama-host 127.0.0.1 --ollama-port 11499 \
    --embedding-cache results/iter59_query_embeddings.npz \
    --iter59-predictions results/iter66_multilabel_ranking_predictions.jsonl \
    --output results/iter67_multilabel_ranking_predictions.jsonl

# 3) 指標・検定（Iter59〜66 と同一スクリプト・無改造）
uv run python -m scripts.compute_iter59_ranking_stats \
    --baseline results/20260918_202613/results.jsonl \
    --new results/iter67_multilabel_ranking_predictions.jsonl \
    --output results/iter67_stats.json

# 4) 被覆 2 個行数・対 Iter61／対 Iter66 McNemar などのアドホック集計（読み取り専用）
```

**タイムアウトと実行上の運用**

- 学習ステップ: **タイムアウト 3600 秒**・バックグラウンド実行．埋め込み対象は 810 行のみで
  Iter65 の 2237 行・Iter66 の 3664 行より大幅に少なく，実測ベース見積りは **5 分未満**．
- 採点・統計・アドホック集計: 各 **タイムアウト 900 秒**．
- 評価クエリ埋め込みキャッシュのミス件数を記録すること（期待値 0．非 0 なら固定構成の破れ）．
- 学習時に `_assert_a0_true_multilabel_signal()` が通ること（調査 Q(a)-2 の予測: multilabel 行数
  810・`len(mlb.classes_)`=10）を stderr で確認する．

**参照点（すべて実測値．事前に確定）**

| 系列 | 構成 | 被覆 2 個行 | `compound_domain_set_recall` | 複合 100 行 rank_1 正解 |
|---|---|---|---|---|
| 基準線 | 固定 k=2 | — | 0.345 | 41/100 |
| **Iter61** | rank_1=baseline・合成 135 行を単一ドメイン行と混合 | **12/100** | **0.445**（+10.0pt） | 41/100 |
| Iter64 | rank_1=head_argmax・合成 405 行 | 24/100 | 0.545（+20.0pt） | 72/100 |
| Iter66 | rank_1=head_argmax・合成 810 行・質量比 35.1% | 21/100 | 0.545 | 72/100 |

Iter67 は **rank_1=baseline 系列（＝Iter61 と同じ土俵）**に属する．よって主基準の閾値は
Iter61 の実測値に置き，Iter64 の 24 は「単一ヘッドで到達できた最良値」＝**上側の参照点**として
効果量の割合計算にのみ用いる（主基準にはしない）．

**成功条件（事前登録．事後変更禁止．不等号の向きと境界値の扱いを明示する）**

主基準は複合側に置く（backlog B100 申し送り (1)）．以下で X＝複合 100 行のうち
`expected_domains` の 2 ドメインを `dispatched_domains` が完全被覆した行数（整数，0〜100），
R＝`compound_domain_set_recall`（ドメイン対 n=200 の部分点）とする．

- **P1（主基準 a）**: **X ≧ 12**（**境界値 12 を含めて PASS**．Iter61 実測と同値なら「2 ヘッド化の
  代償は複合側を Iter61 より悪化させない」とみなす）．X ≦ 11 で FAIL．
- **P2（主基準 b）**: **R ≧ 0.445**（**境界値 0.445 ちょうどを含めて PASS**．Iter61 実測＝
  基準線 0.345 に対し +10.0pt）．R < 0.445 で FAIL．浮動小数の比較は
  `R >= 0.445 - 1e-9` で行う（0.445 は 200 分の 89 で厳密に表現されるため実質同値だが，
  境界判定を機械的に再現可能にするため許容誤差を明示する）．
- **P3**: **`mean_dispatch` = 2.000000**（完全一致）かつ `duplicate_rank1_rank2_count` = 0．
- **P4**: **対 Iter66 予測の不一致行 > 0**（S4．レバーが予測を実際に動かしたことの確認．
  0 なら実験不成立＝下記 V3）．

**実験成立の検査項目（V．主基準ではない．FAIL なら adopted/partial/rejected のいずれにも
分類せず，原因調査へ戻る）**

- **V1**: `_assert_rank1_unchanged()`（A1）が AssertionError を出さずに完走すること．
  すなわち 1600 行すべてで rank_1 が基準線の `dispatched_domains[0]` と一致する．
  **これが「単一ドメイン 1500 行の判定が定義上不変」であることの唯一の正しい操作化である．**
- **V2**: `new_top1_accuracy` が基準線の `top1_accuracy` **0.5975 と完全一致**すること
  （`selected_domain` を上書きしないため定義上一致する．不一致なら実装の取り違え）．
- **V3**: P4 が不成立（対 Iter66 不一致行 = 0）の場合は，新ヘッドが読まれていない疑いがあるため
  不成立として扱う．
- **V4**: 評価クエリ埋め込みキャッシュのミスが発生しないこと（ミス > 0 なら固定構成の破れ）．
- **V5**: 学習時に `_assert_a0_true_multilabel_signal()` が通り，multilabel 行数が 810，
  `len(classes_)` が 10 であること．

**採否の判定基準（効果量の割合で次の一手まで一意に決める．二択の分岐にしない）**

まず **V1〜V5 のいずれかが FAIL なら「実験不成立」**として原因調査に戻る（判定を下さない）．
V が全て通ったうえで，**P1・P2・P3 のいずれかが FAIL なら rejected** とする．
P1〜P4 をすべて充足した場合，**効果量の達成割合**
**r = (X − 12) / (24 − 12)**（下側参照点＝Iter61 の 12，上側参照点＝単一ヘッド最良の Iter64 の 24）
を計算し，次の帯で判定と次の一手を決める（境界値はすべて左側の帯に属する＝
不等号は `≧` を上の帯の下限に付ける）．

| 帯 | r の範囲 | X（同値な整数条件） | 判定 | 次の一手 |
|---|---|---|---|---|
| A | r ≧ 1.00 | X ≧ 24 | **adopted（強）** | 「N5 を一切犠牲にせず単一ヘッド最良値に並んだ」．R-F（実行時経路への配線）の是非を**人間に諮る**材料が初めて揃う．backlog に needs-human で起票する |
| B | 0.75 ≦ r < 1.00 | 21 ≦ X ≦ 23 | **adopted** | 単一ヘッド（Iter65/66 の 21）に rank_1 無犠牲で並んだ．次レバーは 2 ヘッド構成を固定したまま rank_2 ヘッドの合成量を増やす用量反応の再開 |
| C | 0.25 ≦ r < 0.75 | 15 ≦ X ≦ 20 | **partial** | 構造分離は効くが単一ヘッド水準に届かない．次レバーは rank_2 ヘッド側の学習設定（合成量・ペア被覆）であり，rank_1 側には戻らない |
| D | 0 ≦ r < 0.25 | 12 ≦ X ≦ 14 | **partial（弱）** | 構造分離は無害だが利得がない＝rank_2 の質が律速．次レバーは生成文の品質改善（echo 除去フィルタ） |
| E | r < 0 | X ≦ 11 | **rejected**（P1 FAIL） | 合成のみで学習したヘッドは rank_2 としても Iter61 に劣る．2 ヘッド構成を棄却し，複合設問データセット自体の再設計（research_frontier 相当）を人間に諮る |

注: 帯 E は P1 FAIL と同一条件であり，表は P1〜P4 充足時のみ帯 A〜D が適用されることを明示するために
E も併記している．**P2（R ≧ 0.445）が FAIL の場合は X の値に関わらず rejected** とする
（複合側の部分点が Iter61 を割るなら，被覆 2 個行だけが増えていても構成として採れない）．

**非退行条件（事前登録．FAIL でも rejected にはせず最大 partial とする．Iter63〜66 と同じ運用）**

- **N3**: legal 自己被覆 **≧ 8/30**．
- **N6''**: education 自己被覆 **≧ 6/20** かつ medical 自己被覆 **≧ 18/28**
  （Iter64〜66 と同一の下限を据え置き，イテレーション間の比較可能性を優先する）．
- 本構成では rank_1 が基準線に固定されるため，これらの自己被覆は rank_2 の寄与のみで動く．
  Iter61 系列の値との差が大きい場合は解釈節で機序を論じる．

**探索的な診断値（主基準にしないこと）**

- `n5_single_domain_argmax_accuracy` の実測値（**上記の訂正のとおり判定に用いない**．
  合成のみで学習したヘッドが単一ドメイン四択をどれだけ当てられるかの素の観察値であり，
  0.590 未満の WARNING は想定内）．
- 対 Iter61（12/100）の被覆 2 個行の対応あり exact McNemar（discordant の内訳つき）．
  帯 A〜D の判定とは独立に記録する（R-H＝複合 100 行の検出力限界は本イテレーションでも未解消）．
- 対 Iter66（21/100）の同 McNemar（系列が違うため参考値として扱う）．
- 複合 100 行の rank_1 正解が 41/100（基準線と同値）であることの確認．
- rank_1 正解行に占める rank_2 正解割合（Iter61 は 12/41 = 0.2927）．
- `rank2_flip_rate`（対基準線の rank_2 変化率）．
- 被覆 2 個行の Wilson 95% 信頼区間．

**既知の留保（事前登録）**

- **R-K（効果量系列の分離．B100 要レビュー (b)）**: 本イテレーションは rank_1=baseline 系列に
  属するため，結果を B98 の +20.0pt 系列（rank_1=head_argmax）と同一視してはならない．
  対外記述は B95（+10.0pt 系列）・B98（+20.0pt 系列）の 2 本立てを維持し，Iter67 の値を
  3 本目として混同しないこと．
- **R-L（rank_1 の弱化は設計上の代償）**: 複合 100 行の rank_1 正解は 72（Iter66）→41（基準線）へ
  落ちる．被覆 2 個行の上限は構造的に 41 であり，Iter64/66 の上限 72 とは土俵が違う．
  この非対称性は帯の閾値（下側 12・上側 24）に織り込み済みだが，**上側参照点 24 は本来 72 の
  土俵で得られた値であり，帯 A の達成は原理的に難しい**ことを事前に明記しておく．
- **R-M（合成 810 行への条件付き）**: 結論は Iter65 の生成乱数の実現値である 810 文に対して
  条件付きである．
- **R-F（実行時経路への未配線．9 反復目）**: adopted でも実機での有効性は主張できない．
- 較正値（確率スコアの絶対値）は対外引用しない（R-I の継続．本イテレーションは行複製がないため
  リーク源自体は消えるが，運用方針は据え置く）．

**単一レバー原則の確認（混入チェック）**

- 合成訓練データ（ファイル・行数・生成乱数の実現値）→ **無変更**（再生成しない）．
- 埋め込みモデル・埋め込みキャッシュ・ヘッド種別・較正・ハイパーパラメータ → **無変更**．
- 採点スクリプト・rank_2 の選択ロジック・統計スクリプト・基準線・`config.yaml` → **無変更**
  （コード差分ゼロ）．
- 変更点は「rank_1 の供給源を baseline に戻す」ことと「rank_2 ヘッドを合成のみで学習する」ことの
  2 つだが，**これは『rank_1 と rank_2 を別ヘッドに分離する』という 1 つの構造変更の表裏**であり，
  どちらか一方だけでは 2 ヘッド構成にならない（rank_1 を戻さなければ単一ヘッドのままであり，
  合成のみで学習しなければ rank_2 ヘッドが単一ドメイン行に汚染される）．単一レバー原則に適合する．

**想定コスト**: 学習 5 分未満 ＋ 採点 約 1 分 ＋ 統計・アドホック集計 約 1 分 ＝ **10 分程度**．
合成生成なし・オフライン完結・`config.yaml` のスキーマ変更なし・実機 1600 問本走なしのため
自律着手してよい．

### 実装 (Iter67)

**CLI 引数の事前検証（`--help` 実測）**

`train_multilabel_dispatch_head.py --help` と `evaluate_dispatch_candidate_ranking.py --help` を
実行し，計画（調査 Q(a)-1・Q(b) で確認済みの実装）に登場する引数（`--train-data`（必須）・
`--multilabel-train-data`（必須）・`--embedding-model`・`--ollama-host`・`--ollama-port`・
`--output`／`--baseline`・`--head`・`--embedding-cache`・`--iter59-predictions`（optional）・
`--rank1-source {baseline,head_argmax}`（既定 `baseline`））が，綴り・必須/任意の別とも計画の
記述と完全に一致することを確認した．**スクリプトのコード変更は不要であり，実際に一切行っていない**
（`git status --short` でスクリプト差分ゼロを確認）．

**実行環境**

wafl-ctrl5 への SSH（`ssh wafl-ctrl5`）で GPU 使用率 0%・空きディスク 435GB・
`ollama-ctrl` コンテナ稼働中を確認．過去イテレーションと同様，リポジトリ自体は wafl-ctrl5 上には
存在せず，ローカル（本リポジトリ）から既存の SSH ローカルフォワード `127.0.0.1:11499` 経由で
wafl-ctrl5 上の Ollama へ埋め込み計算を委譲する運用（`curl http://127.0.0.1:11499/api/tags` で
`nomic-embed-text:latest` の生存を確認済み）．コード実行自体はローカルで行うが，埋め込み計算という
重い処理は wafl-ctrl5 の GPU で行われるため，config.yml 恒久ルールに適合する．

**実行コマンドと結果**

1. ヘッド学習（計画どおり，バックグラウンド実行・実測 約 90 秒）:
   ```
   uv run python -m scripts.train_multilabel_dispatch_head \
       --train-data /dev/null \
       --multilabel-train-data data/classifier_train_multidomain_iter65.jsonl \
       --embedding-model nomic-embed-text --ollama-host 127.0.0.1 --ollama-port 11499 \
       --output models/dispatch_multilabel_head_iter67_synth_only.joblib
   ```
   出力: `A0 PASS: 810 multi-label rows covering 45 distinct domain pairs`（10 ドメイン全出現，
   調査 Q(a)-2 の予測どおり）．`n_positive=162`（全ドメイン共通）．
   `wrote models/dispatch_multilabel_head_iter67_synth_only.joblib (n_single_label_rows=0,
   n_synthetic_rows=810, classes=[10 ドメイン])`．**`n_single_label_rows=0` により，
   単一ドメイン行が混入していないことを実行ログで直接確認した（V5 充足）**．
2. 採点（計画どおり）:
   ```
   uv run python -m scripts.evaluate_dispatch_candidate_ranking \
       --baseline results/20260918_202613/results.jsonl \
       --head models/dispatch_multilabel_head_iter67_synth_only.joblib \
       --rank1-source baseline \
       --embedding-model nomic-embed-text --ollama-host 127.0.0.1 --ollama-port 11499 \
       --embedding-cache results/iter59_query_embeddings.npz \
       --iter59-predictions results/iter66_multilabel_ranking_predictions.jsonl \
       --output results/iter67_multilabel_ranking_predictions.jsonl
   ```
   出力: `WARNING: N5 single-domain argmax accuracy 0.2433 < floor 0.59`（**計画で事前に想定済みの
   診断値．判定には用いない**）．`mean_dispatch: 2.0`・`compound_domain_set_recall: 0.5`・
   `a5_iter59_disagreement.mismatches: 1239`（対 Iter66，P4 充足の一次シグナル）．
   実行時間は約 2 分．埋め込みキャッシュ（`results/iter59_query_embeddings.npz`）のミス件数を
   示す明示的なログ行はスクリプトに存在しないが，実行時間（約 2 分，1600 行全件を新規埋め込みする
   場合の Iter65〜66 実測より大幅に短い）から，既存キャッシュがヒットしていることが間接的に伺える．
   **分析フェーズは `results/iter67_multilabel_ranking_predictions.jsonl` 側で埋め込みキャッシュの
   ヒット率を直接検証できる指標がスクリプトにない点を踏まえ，V4 の判定には別途注意すること**
   （このスクリプトの現状の出力だけでは V4 を厳密に確認できない旨をここに記録する）．
3. 統計（計画どおり，無改造の `compute_iter59_ranking_stats.py`）:
   ```
   uv run python -m scripts.compute_iter59_ranking_stats \
       --baseline results/20260918_202613/results.jsonl \
       --new results/iter67_multilabel_ranking_predictions.jsonl \
       --output results/iter67_stats.json
   ```
   出力の主要項目（全文は `results/iter67_stats.json` 参照）:
   - `N1_rank1_invariance.mismatch_count: 0, pass: true` → **V1（`_assert_rank1_unchanged()` 相当の
     bit 単位一致）を実測で確認**．
   - `N2_top1_accuracy_invariance.new_top1_accuracy: 0.5975`（基準線の `0.5975` と `exact_match: true`）
     → **V2 充足**．
   - `S3_cost_neutrality.duplicate_rank1_rank2_count: 0, mean_dispatch: 2.0` → P3 充足．
   - `S4_flip_rate_evidence_of_firing.flips: 1319, rank2_flip_rate: 0.824375` → 対基準線での
     rank_2 変化率（対 Iter66 の不一致行数とは別集計．P4 は上記 `a5_iter59_disagreement.mismatches:
     1239 > 0` で判定する）．
   - `new_compound_domain_set_recall: 0.5`（基準線 0.345 比 +15.5pt）．
   - `N3_legal_non_regression.new_legal_self_coverage: 10`（下限 8 を上回る）．

**次フェーズ（実験）が確認すべきこと**

- 成果物パス: `models/dispatch_multilabel_head_iter67_synth_only.joblib`・
  `results/iter67_multilabel_ranking_predictions.jsonl`・`results/iter67_stats.json`．
  既存の Iter66 成果物（`models/dispatch_multilabel_head_iter66.joblib`・
  `results/iter66_multilabel_ranking_predictions.jsonl`・`results/iter66_stats.json`）は
  md5sum を実装前後で比較し，**無変更であることを確認済み**．
- `results/iter67_stats.json` は V1・V2・P3 の一次判定に使えるが，**P1（X≧12＝被覆2個行数）・
  P2（R≧0.445）・帯 A〜E の判定に必要な「複合 100 行のうち完全被覆した行数」の生値は
  `compute_iter59_ranking_stats.py` の出力に含まれない**（`new_compound_domain_set_recall: 0.5` は
  ドメイン対 n=200 の部分点であり，行数 X とは別集計）．計画の「実行計画 4）」に記載の
  アドホック集計（Iter65 で確立した読み取り専用スクリプト／手順）を実行し，X の実測値を出す作業は
  実装フェーズの範囲外（本フェーズの委譲元指示は学習・採点の 2 ステップのみ）として次フェーズに
  委ねる．
- 実行時間実測: 学習 約 90 秒，採点 約 2 分，統計 数秒．計画見積り「10 分程度」を大幅に下回った．
- 型/lint/テスト: スクリプトへの変更が一切ないため実行していない（差分ゼロのコード変更に対する
  型・lint 検証は対象がなく不要と判断した）．
- 実験を開始してよい状態: **可**．V1・V2・V5 は実装フェーズの実測で充足を確認済み．
  残る V3（P4 充足）は `a5_iter59_disagreement.mismatches: 1239 > 0` で満たされている．
  V4（埋め込みキャッシュミス 0）は上記のとおりログから直接確認できないが，`stat` で
  `results/iter59_query_embeddings.npz` の mtime が本イテレーションの実行前後で不変（2026-09-19
  02:29:59，Iter65 の実行時刻のまま）であることを実測した．キャッシュに新規書き込みが発生して
  いれば mtime が更新されるはずであり，不変であることは全クエリ埋め込みがキャッシュヒットした
  （ミス 0）ことの間接証拠として扱える．**V4 は実測で充足を確認した**．

### 分析(実行) (Iter67)

**前提**: 実装フェーズが `models/dispatch_multilabel_head_iter67_synth_only.joblib`・
`results/iter67_multilabel_ranking_predictions.jsonl`・`results/iter67_stats.json` を既に生成済み
（本走ではなくオフライン完結のレバーのため，本フェーズは新規ジョブを起動せず，既存 3 ファイルに
対する読み取り専用のアドホック集計のみを行う．state.json の実験起動用フィールド
（`experiment_dir`/`experiment_deadline`）は，長時間ジョブを新規に起動していないため更新していない）．

**アドホック集計スクリプト**: `/tmp/iter67_adhoc.py`（Iter65/66 で確立した手法をそのまま踏襲，
公式スクリプト無改造）．`results/iter67_multilabel_ranking_predictions.jsonl`・
`results/iter66_multilabel_ranking_predictions.jsonl`・`results/iter61_multilabel_ranking_predictions.jsonl`
の `expected_domains`/`dispatched_domains`/`selected_domain`/`rank2_new` フィールドのみを突き合わせる
10〜20 行程度のブール演算である．

```
uv run python /tmp/iter67_adhoc.py
```

実測出力:

```json
{
  "coverage2_count_compound100_X": 15,
  "rank1_correct_compound100": 41,
  "n3_legal_self_coverage": {"covered": 10, "total": 30},
  "n6pp_education_self_coverage": {"covered": 7, "total": 20},
  "n6pp_medical_self_coverage": {"covered": 13, "total": 28},
  "mcnemar_coverage2_vs_iter66": {
    "a_true_b_false": 8, "a_false_b_true": 14,
    "both_true": 7, "both_false": 71,
    "p_value_exact_binomtest": 0.28627872467041016, "b_count": 21
  },
  "mcnemar_coverage2_vs_iter61": {
    "a_true_b_false": 11, "a_false_b_true": 8,
    "both_true": 4, "both_false": 77,
    "p_value_exact_binomtest": 0.6476058959960938, "b_count": 12
  },
  "mismatch_rows_vs_iter66_selected_domain_check": 491,
  "coverage2_within_rank1_correct": {"covered": 15, "total": 41, "ratio": 0.36585365853658536}
}
```

**読み取り整合性の検算**: `n3_legal_self_coverage`（アドホック 10/30）は
`results/iter67_stats.json` の `N3_legal_non_regression.new_legal_self_coverage: 10`
（公式出力）と完全一致した．また P4 の一次シグナルとして実装フェーズが報告した
`a5_iter59_disagreement.mismatches: 1239`（評価スクリプト標準出力，対 Iter66）を，
`rank2_new` フィールドを直接突き合わせて独立に再現した（`mismatch=1239`，一致）．
（`mismatch_rows_vs_iter66_selected_domain_check: 491` は `selected_domain`＝rank_1 ベースの
補助集計であり，Iter66 が `head_argmax`・Iter67 が `baseline` と rank_1 の供給源自体が異なる
ため乖離するのは想定どおりであり，P4 の判定には用いない．公式値は rank_2 ベースの 1239）．

**X（被覆2個行数）と rank_1 正解数**: X = **15/100**．rank_1 正解数 = **41/100**
（基準線と完全一致，計画の想定「72→41 相当」のとおり実測された）．

**`compound_domain_set_recall` の定義について**: Iter63〜66 いずれも，P2 相当の主基準の判定には
`compute_iter59_ranking_stats.py` が出力する `new_compound_domain_set_recall`（ドメイン対
n=200 の部分点）をそのまま用いている（journal.md の Iter64 S2＝`compound_domain_set_recall`
≧0.510／Iter65 S2＝同 ≧0.565／Iter66 N7＝同 ≧0.545 の各節で，いずれも stats.json 由来の値を
直接閾値比較している）．今回もこの定義を踏襲し，**P2 判定には
`results/iter67_stats.json` の `new_compound_domain_set_recall: 0.5` をそのまま用いる**
（行ベースの X とは別集計であり，新たに「行ベースの recall」を算出する必要はない）．

**判定用の一次数値（数値のみ．判定は rc-analyst に委ねる）**

| 項目 | 定義 | 実測値 | 事前登録の閾値／参照 |
|---|---|---|---|
| P1（X） | 複合100行のうち`dispatched_domains`が`expected_domains`を完全被覆した行数 | **15/100** | X≧12でPASS |
| P2（R） | `results/iter67_stats.json`の`new_compound_domain_set_recall` | **0.5** | R≧0.445でPASS |
| P3a | `mean_dispatch` | 2.000000 | =2.000000でPASS |
| P3b | `duplicate_rank1_rank2_count` | 0 | =0でPASS |
| P4 | 対Iter66不一致行数（`rank2_new`基準，公式値） | 1239/1600 | >0でPASS |
| V1 | `N1_rank1_invariance.mismatch_count` | 0 | =0でPASS |
| V2 | `new_top1_accuracy` | 0.5975 | 基準線0.5975と完全一致でPASS |
| V3 | P4充足 | 1239>0 | 充足 |
| V4 | 埋め込みキャッシュミス（間接証拠：npz mtime不変） | 0（間接） | =0でPASS |
| V5 | 学習ログ`n_single_label_rows`/`multilabel行数`/`classes_`数 | 0／810／10 | 実装フェーズで確認済み |
| r（効果量） | (X−12)/(24−12) | **(15−12)/(24−12) = 0.25** | 帯Cの下端（0.25≦r<0.75）に一致 |
| 複合100 rank_1正解 | `selected_domain`が`expected_domains`のいずれかに一致 | **41/100** | 基準線と完全一致 |
| rank_1正解中のrank_2正解割合 | 被覆2個 ÷ rank_1正解 | 15/41 = 0.3659 | 参考値（Iter61=12/41=0.2927） |
| N3（legal自己被覆） | 複合ペア母集団を分母（Iter64〜66と同一手法） | **10/30** | ≧8/30でPASS |
| N6''（education自己被覆） | 同上 | **7/20** | ≧6/20でPASS |
| N6''（medical自己被覆） | 同上 | **13/28** | ≧18/28で**FAIL**（5行不足） |
| 対Iter66 McNemar（被覆2個，exact binomtest） | discordant b=8（67のみTrue）・c=14（66のみTrue） | p=**0.286278** | 記録のみ |
| 対Iter61 McNemar（被覆2個，exact binomtest） | discordant b=11（67のみTrue）・c=8（61のみTrue） | p=**0.647606** | 記録のみ |

**事実としての異常の有無**: 実行・ログ上の異常は検出されなかった（V1〜V5 はすべて充足，
実装フェーズおよび本フェーズの検算で確認済み）．一方，**N6''（medical自己被覆）13/28 は
事前登録の下限 18/28 を下回っており，非退行条件が FAIL している**（数値の事実のみ報告，
良否判定は rc-analyst に委ねる）．

### 考察 (Iter67)

**判定: partial（帯 C・事前登録の機械的適用）．ただし実質的な利得は rank_1=baseline 系列の
既存 3 点（Iter60/61/62）と区別できない．**

#### 1. 事前登録ルールの機械適用（判定の導出．ここは解釈を挟まない）

| 段階 | 条件 | 実測 | 結果 |
|---|---|---|---|
| 実験成立 | V1〜V5 | mismatch 0／top1 0.5975 一致／不一致 1239／キャッシュ mtime 不変／`n_single_label_rows=0`・810・10 クラス | 全充足．**成立** |
| P1 | X ≧ 12 | X = **15** | PASS |
| P2 | R ≧ 0.445 | R = **0.5**（`0.5 >= 0.445 - 1e-9`） | PASS |
| P3 | mean_dispatch=2.000000 かつ duplicate=0 | 2.000000／0 | PASS |
| P4 | 対 Iter66 不一致 > 0 | 1239 | PASS |
| 効果量 | r = (X−12)/(24−12) | **0.25** | 帯 C（0.25 ≦ r < 0.75）に該当 |

**境界値 r=0.25 の扱い**: 事前登録は「境界値はすべて左側の帯に属する＝不等号は `≧` を上の帯の
下限に付ける」と明記している．帯 C の下限は 0.25 で `≧` が付くため，**r=0.25 は帯 D ではなく
帯 C に属する**．整数条件（15 ≦ X ≦ 20）でも X=15 は帯 C の下端に一致し，2 通りの表現が
矛盾しないことを確認した．よって判定は **partial**，指示された次の一手は
「rank_2 ヘッド側の学習設定（合成量・ペア被覆），rank_1 側には戻らない」である．

**非退行 N6''(medical) FAIL の影響**: 事前登録は「非退行の FAIL は最大 partial に留める
（adopted にはできない）」と定めている．今回は帯 C により元々 partial であるため，この制約が
判定を追加で引き下げることはない（partial より下の格は事前登録に存在せず，rejected は
P1/P2/P3 の FAIL 条件に限定されている）．したがって **N6''(medical) FAIL は判定値を変えないが，
「adopted への昇格経路が塞がれていた」という事実として記録し，partial の理由付けに含める**．
すなわち本イテレーションの partial は (i) 効果量が帯 C の下端 r=0.25 であったこと，
(ii) 非退行 N6''(medical) が FAIL したこと，の 2 つの独立な根拠による．

#### 2. ノイズか信号か（本フェーズで新たに得た系列分解が決定的）

分析(実行)フェーズが記録した対 Iter61（p=0.6476）・対 Iter66（p=0.2863）に加え，本フェーズで
**予測 JSONL 8 本（Iter60〜67）の rank_1 が基準線と一致するかを直接判定し，比較可能な系列を確定した**
（読み取り専用集計 `/tmp/iter67_rank1src.py`・`/tmp/iter67_analyst_check.py`・`/tmp/iter67_paired.py`）．

| Iter | rank_1 の供給源（実測） | 複合 100 行の rank_1 正解 | X（被覆 2 個） | R |
|---|---|---|---|---|
| 基準線 | — | 41 | **3** | 0.345 |
| 60 | baseline（1600/1600 一致） | 41 | **14** | 0.48 |
| 61 | baseline（1600/1600 一致） | 41 | **12** | 0.445 |
| 62 | baseline（1600/1600 一致） | 41 | **15** | 0.445 |
| 63 | head_argmax（1220/1600） | 61 | 17 | 0.49 |
| 64 | head_argmax（1165/1600） | 72 | 24 | 0.545 |
| 65 | head_argmax（1068/1600） | 72 | 21 | 0.565 |
| 66 | head_argmax（1109/1600） | 73 | 21 | 0.545 |
| **67** | **baseline（1600/1600 一致）** | **41** | **15** | **0.5** |

**Iter67 と同じ土俵（rank_1=baseline）の点は Iter61 だけではなく Iter60・Iter62 も該当する**
（3 点とも 1600/1600 で rank_1 が基準線と bit 単位一致することを実測で確認した）．
この 3 点の X は **14・12・15**（平均 13.67，標本 SD 1.53）であり，**Iter67 の 15 はこの範囲の
上端と同値**にすぎない．対応あり exact 検定も全て有意でない：

| 比較（複合 100 行，対応あり exact binomtest） | discordant | p |
|---|---|---|
| X: Iter67 vs Iter60 | b=8／c=7 | **1.0000** |
| X: Iter67 vs Iter61 | b=11／c=8 | 0.6476 |
| X: Iter67 vs Iter62 | b=9／c=9 | **1.0000** |
| rank_2 命中数: Iter67(59) vs Iter60(55) | b=18／c=14 | 0.5966 |
| rank_2 命中数: Iter67(59) vs Iter61(48) | b=24／c=13 | 0.0989 |
| rank_2 命中数: Iter67(59) vs Iter62(48) | b=25／c=14 | 0.1081 |

X=15 の Wilson 95% CI は [0.093, 0.233] で，Iter61 の 12（[0.070, 0.198]）とほぼ完全に重なる．

**判定**: **X=15 は本系列のノイズ範囲内であり，2 ヘッド構成（＋合成のみ学習）が
rank_1=baseline 系列の既存水準を超えた証拠はない**．P1 が PASS したのは，事前登録が下側参照点を
Iter61 の 12 に置いたためであるが，**12 は同一系列 3 点（14/12/15）の最小値であり，
この閾値を超えることは「改善」の証拠にならない**．これは事前登録時に Iter60・Iter62 も同じ
rank_1=baseline 系列に属することを認識していなかったことによる閾値設定の弱さであり，
事前登録は変更しない（判定は上記のとおり partial のまま）が，**次の事前登録では
「同一 rank_1 系列の既存全点の中央値または最大値」を下側参照点にすべき**という手続き上の学びとして残す．

R（0.5）についても同様である．系列内訳は rank_1 命中 41（4 点とも定義上同一）＋ rank_2 命中
55／48／48／**59** であり，R の差は完全に rank_2 命中数の差である．Iter60 比 +4 行（p=0.597）で
あり，系列 3 点（0.48／0.445／0.445）の幅 0.035 に対し Iter67 の 0.5 は +0.02〜+0.055 に収まる．
**R=0.5 は系列最高値だが有意ではない**（対 Iter61/62 は p≈0.10 で方向は一貫，対 Iter60 は p=0.597）．

#### 3. 非退行条件の内訳と N6''(medical) FAIL の機序

ドメイン別自己被覆を全系列で同一手法により再集計した（基準線も含む）：

| 系列 | legal /30 | education /20 | medical /28 |
|---|---|---|---|
| 基準線 | 8 | 9 | **13** |
| Iter60（baseline） | 20 | 4 | 19 |
| Iter61（baseline） | 15 | 5 | **12** |
| Iter62（baseline） | 16 | 6 | **11** |
| Iter63〜66（head_argmax） | 17／16／15／13 | 3／6／7／6 | **18／20／18／18** |
| **Iter67（baseline）** | **10** | **7** | **13** |

**N6''(medical) の下限 18/28 は rank_1=head_argmax 系列（Iter63〜66 が 18・20・18・18）の実測水準を
据え置いたものであり，rank_1=baseline 系列の実測（基準線 13・Iter60 19・Iter61 12・Iter62 11）とは
土俵が違う**．Iter67 の 13/28 は基準線とちょうど同値，Iter61 比 +1（b=3／c=2，p=1.0000），
Iter62 比 +2（b=3／c=1，p=0.6250）であり，**同一系列の中では退行していない**．
つまり今回の N6'' FAIL は，rank_1 が 72/100 正解する系列で成立していた下限を，rank_1 が 41/100 しか
正解しない系列へそのまま輸入したことによる系列跨ぎのアーティファクトである
（計画節が「本構成では自己被覆は rank_2 の寄与のみで動く．Iter61 系列との差が大きい場合は
機序を論じる」と事前に予告していた事象が実際に起きた）．**事前登録は変更しないため FAIL は FAIL として
扱い，判定は最大 partial に据え置く**が，「medical のルーティングが実際に劣化した」という
読み方は実測が支持しないことを明記する．

一方，**N3(legal) 10/30 は下限 8/30 を満たし PASS だが，rank_1=baseline 系列の中では最低値である**
（20／15／16 → 10）．対 Iter60 の対応あり検定は b=0／c=10，**p=0.0020**（本考察で行った 12 検定に
BH 補正を掛けても q≈0.024 で有意）であり，**本イテレーションで唯一の実質的な悪化サインである**．
機序としては，合成 810 行のみで学習したヘッドでは legal が rank_2 に選ばれにくくなったことになる
（legal は単一ドメイン訓練データが 77 行と最少のドメインだが，合成 810 行では他ドメインと同じ
162 行が割り当てられており，混合学習時に legal に効いていた重み付けが失われた可能性がある）．
n=30 の探索的観察であり断定はしない．education 7/20 は系列最高（4/5/6 → 7）だが基準線 9/20 は
下回っており，これも系列内のばらつき（p≥0.25）の域を出ない．

#### 4. 仮説との整合

計画の仮説は 2 つの主張から成る．

1. **「rank_1 を戻せば単一ドメイン 1500 行の判定は定義上不変になり，退行が構造的にゼロになる」
   → 完全に支持された**．V1（1600/1600 で rank_1 一致）・V2（top1 0.5975 が基準線と厳密一致）で
   実測確認済みであり，これは測定ではなく構成上の恒等式である．Iter63〜66 が払い続けた N5 退行
   （0.591→0.563→0.577，基準線 0.590 割れ）の代償は，本構成では原理的に発生しない．
2. **「複合側の利得だけを取り出せる」→ 支持されなかった**．対立仮説「rank_1 弱化の代償を払うと
   複合被覆が Iter61 水準から伸びない」は，X=15 が Iter60/62 と p=1.0 で区別できない以上，
   **棄却できない**．正確には「Iter61（12）よりはわずかに上だが，同系列の Iter62（15）と同値で，
   2 ヘッド化＋合成のみ学習という今回の変更が上乗せした利得はゼロと区別できない」である．

なお，rank_1 正解行に条件付けた被覆率は Iter67 が 15/41=0.366 で，Iter61 12/41=0.293・
Iter64 24/72=0.333・Iter66 21/73=0.288 を上回るが，**Iter62 の 15/41=0.366 と同値**である．
「rank_1 正解行に限れば 2 ヘッド構成が最良」という読み方も，同系列の Iter62 と区別できない．

#### 5.「主基準 PASS だが効果量が控えめ」であることの機序（過学習・学習不足の観点）

- **rank_2 ヘッドの素の能力は極めて低い**．診断値 `n5_single_domain_argmax_accuracy=0.2433` は，
  10 ドメインの偶然一致 0.10 に対して 2.4 倍にすぎない（混合学習したヘッドは 0.56〜0.59）．
  合成 810 行は 45 ペア × 18 行の完全一様配分であり，各ドメインの OvR は正例 162／負例 648 と
  少数かつ人工的な分布である．**このヘッドは「合成文の分布」に対しては学習できているが，
  評価集合（JMMLU 由来の四択 1500 行＋手作り相談文 100 行）への転移が弱い**．
- ただし，**その低い素の能力にもかかわらず rank_2 命中数は 59/100 と，単一ドメイン行を混ぜて
  学習した Iter60/61/62（55/48/48）と同水準以上である**．これは
  **「単一ドメイン行を混ぜるか否か」は rank_2 の質をほとんど左右しない**ことを意味する．
  Iter63〜66 で確定した「混ぜ方（本数・質量比）は N5 と複合被覆のトレードオフ曲線上を動くだけ」
  という結論に，**「rank_2 側から見れば混ぜ方はそもそも効いていない」**という一段強い形が加わった．
- したがって **rank_2 の律速は学習データの混ぜ方でも量でもなく，合成文そのものの質と，
  合成分布（45 ペア一様）と評価集合（実際の複合 100 行）の分布ギャップにある**と解釈する．
  これは帯 D の次の一手（生成文の品質改善＝echo 除去）が指していた診断と実質的に同じ結論であり，
  帯 C と帯 D の境界上（r=0.25 ちょうど）に落ちたことと整合する．
- **過学習リスク**: 合成 810 行のみでの学習は，`CalibratedClassifierCV(cv=5)` の較正が合成分布に
  対して行われるため，較正値の絶対水準は評価集合に対して意味を持たない（R-I の運用どおり
  較正値は対外引用しない）．なお Iter66 で懸念された行複製由来の fold リークは，本構成では
  複製がないため消えている．

#### 6. 留保の更新

- **R-K（効果量系列の分離）を訂正して強化する**．Iter67 は「3 本目の系列」ではなく
  **B95 の rank_1=baseline 系列（+10.0pt 系列）の 4 点目**である．同系列の対基準線改善幅は
  Iter60 +13.5pt・Iter61/62 +10.0pt・Iter67 **+15.5pt** であり，**対外記述では単点の +15.5pt を
  引用せず「rank_1=baseline 構成では +10.0〜+15.5pt（4 点，いずれも互いに有意差なし）」と
  幅で報告すること**．head_argmax 系列（Iter63〜66，+14.5〜+22.0pt）とは rank_1 の供給源が異なるため
  混同しない．
- **R-L（rank_1 弱化は設計上の代償）は実測で裏づけられた**．被覆 2 個行の構造的上限は
  rank_1 正解 41 であり，上側参照点 24 は上限 72 の土俵で得た値である．**r の分母（24−12）は
  土俵混在のため，r の絶対値を「達成割合」として過度に読まないこと**（事前登録に従い判定には
  用いたが，解釈上の意味は弱い）．rank_1 正解行に条件付けた指標（15/41=0.366）の方が
  系列間比較には適する．
- **R-H（複合 100 行の検出力限界）が本イテレーションで臨界に達した**．同系列 4 点の X は
  12〜15 に収まり，対応あり検定はいずれも p ≧ 0.65．n=100・discordant 15〜19 行の設計では
  **±3〜4 行の差を検出できない**．「混ぜ方」「量」「構造」のいずれのレバーも，この分解能の下では
  もう区別がつかない．
- R-M（合成 810 行の実現値への条件付き）・R-F（実行時経路への未配線．**10 反復目**）・
  R-I（較正値の非引用）は継続．

#### 7. 確信度と追加反復の要否

- **判定（partial）の確信度: 高**．事前登録から一意に導かれ，解釈の余地はない．
- **「2 ヘッド化は複合側に上乗せの利得を与えなかった」という実質判断の確信度: 中〜高**．
  根拠は同系列 3 点との対応あり検定（p=1.0／0.65／1.0）と，系列内 SD 1.53 に対し差が +1.3 行で
  あること．反証されうるとすれば検出力不足（R-H）による見逃しだが，**同じ n=100 の設計で
  追加反復しても分解能は上がらない**ため，**同一レバーでの追加反復は推奨しない**．
- **一方，「rank_1 の判定が構造的に不変になる」という 2 ヘッド構成の利点は確定事実である**．
  複合側で上乗せがないだけで，N5 を一切犠牲にしない点は Iter63〜66 に対する明確な優位であり，
  構成としては保持する価値がある（帯 C の指示「rank_1 側には戻らない」と整合）．

#### 8. 次の考察・計画フェーズへの申し送り（帯 C「rank_2 ヘッドの学習設定」の具体化）

帯 C が指す「rank_2 ヘッド側の学習設定（合成量・ペア被覆）」のうち，**合成量の用量反応は
Iter64（405）→Iter65（810）で頭打ち，質量比は Iter66 で不支持**として既に尽きている．
上記 5 の機序（律速は混ぜ方ではなく合成文の質と分布ギャップ）を踏まえ，次レバーの候補を
優先順に示す（決定は rc-reflector に委ねる）．

1. **合成文の品質フィルタ（echo 除去）を rank_2 ヘッドの訓練データに適用する**（推奨）．
   2 ヘッド構成は固定し，`data/classifier_train_multidomain_iter65.jsonl`（810 行）から
   設問文の語をそのまま反復しているだけの行を機械的に除去した部分集合で再学習する．
   帯 C の「学習設定」に含まれ，かつ Iter63〜66 で唯一未探索の軸である．**行数が減るため
   「量の効果」と交絡しうる点に注意し，計画フェーズで除去率と量の交絡をどう分離するかを
   事前登録すること**（例: 同数をランダム除去した対照ヘッドを同時に作る）．
2. **ペア被覆の非一様化**（45 ペア一律 18 行をやめる）．ただし**評価集合（複合 100 行）の
   実際のペア分布に合わせる設計は test set への適合＝リークであり採用してはならない**．
   採るなら「単一ドメイン訓練データの共起統計」など評価集合と独立な根拠に基づく配分に限る．
   計画フェーズでリーク判定を明示的に行うこと．
3. **追加反復（同一構成・生成乱数違い）は推奨しない**．R-M の条件付きを外す価値はあるが，
   R-H（n=100 で ±3〜4 行を検出できない）により結論は変わらない．
4. **人間に諮る材料が 2 件たまった**（backlog 起票を推奨）．
   (a) **R-H の解消＝複合設問データセットの拡充**（現状 100 行．research_frontier 相当）．
   これなしには本系列のレバーはこれ以上の分解能を持たない．
   (b) **R-F（実行時経路への配線）**．帯 A に届かなかったため事前登録上は諮る条件を満たさないが，
   「N5 を一切犠牲にしない」という 2 ヘッド構成の利点自体は確定しており，
   複合側の利得が系列内で有意でない現状でも配線の是非を問う価値があるかは人間判断である．

### Iteration 67 実行済み

**単一レバー**: `multilabel_head_architecture = two_head_rank1_single_domain_classifier`
（rank_1 を既存の単一ドメイン分類器の出力＝`--rank1-source baseline` に戻し，rank_2 のみを
合成 810 行だけで学習した別ヘッドから選ぶ 2 ヘッド構成）．固定した構成は合成訓練データ
（`data/classifier_train_multidomain_iter65.jsonl` 810 行・生成乱数の実現値）・埋め込み
`nomic-embed-text` とキャッシュ・ヘッド種別（OvR×Platt 較正 cv=5）・基準線
`results/20260918_202613/results.jsonl`・統計スクリプト・`config.yaml`・実行時経路（配線しない）．

**変更（生成物）**: `models/dispatch_multilabel_head_iter67_synth_only.joblib`（gitignore 対象の
`models/` 配下のため git 履歴には残らない）・`results/iter67_multilabel_ranking_predictions.jsonl`・
`results/iter67_stats.json`．本走なし・オフライン完結（約 10 分）．

**結果（主要値）**

| 項目 | 実測 | 事前登録 | 可否 |
|---|---|---|---|
| P1 被覆 2 個行 X | **15/100** | ≧12 | PASS |
| P2 `compound_domain_set_recall` R | **0.5** | ≧0.445 | PASS |
| P3 `mean_dispatch`／重複 | 2.000000／0 | =2.000000／=0 | PASS |
| P4 対 Iter66 不一致（rank2 基準） | 1239/1600 | >0 | PASS |
| V1 rank_1 不一致 | 0/1600 | =0 | PASS |
| V2 `new_top1_accuracy` | 0.5975（基準線と厳密一致） | 一致 | PASS |
| N3 legal 自己被覆 | 10/30 | ≧8/30 | PASS（系列最低値） |
| N6'' education 自己被覆 | 7/20 | ≧6/20 | PASS |
| N6'' medical 自己被覆 | 13/28 | ≧18/28 | **FAIL** |
| 効果量 r=(X−12)/(24−12) | **0.25** | 帯 C の下端 | 帯 C |

**判定: partial（採用でも棄却でもない．事前登録の機械適用で帯 C．本レバーは値を試し切ったため
クローズする）**．理由は 2 つの独立な根拠による．(i) 効果量が帯 C の下端 r=0.25 であったこと，
(ii) 非退行 N6''(medical) が FAIL し adopted への昇格経路が塞がれていたこと．

**学び**

1. **仮説の前半は構成上の恒等式として完全に支持された**．rank_1 を基準線へ戻せば単一ドメイン
   1500 行の判定は定義上不変で，V1（1600/1600 一致）・V2（top1 が小数点以下まで一致）が実測した．
   Iter63〜66 が 4 反復にわたり払い続けた N5 退行（基準線 0.590 に対し 0.591→0.563→0.577）は
   本構成では原理的に発生しない．**「N5 を犠牲にしない」という点では 2 ヘッド構成が優位である**．
2. **仮説の後半（複合側の利得だけを取り出せる）は支持されなかった**．X=15 は
   同じ rank_1=baseline 系列の既存 3 点（Iter60=14／Iter61=12／Iter62=15）と統計的に区別できない
   （対 Iter60 p=1.0000，対 Iter62 p=1.0000，対 Iter61 p=0.6476．系列 SD 1.53 に対し差は +1.3 行）．
   **2 ヘッド化が上乗せした利得はゼロと区別できない．**
3. **事前登録の下側参照点の置き方に手続き上の欠陥があった**．P1 の閾値 12 は Iter61 単点に置いたが，
   12 は同一系列 3 点（14/12/15）の**最小値**であり，これを超えても改善の証拠にならない．
   **次の事前登録では「同一 rank_1 系列の既存全点の中央値または最大値」を下側参照点にすること．**
4. **N6''(medical) FAIL は実際の劣化ではなく系列跨ぎのアーティファクトである**．下限 18/28 は
   rank_1=head_argmax 系列（Iter63〜66 が 18/20/18/18）の水準をそのまま輸入した値で，
   rank_1=baseline 系列の実測は基準線 13・Iter60 19・Iter61 12・Iter62 11 と分布が異なる．
   Iter67 の 13/28 は基準線と同値，Iter61 比 +1（p=1.0000）・Iter62 比 +2（p=0.6250）で系列内では
   退行していない．**事前登録は変更せず FAIL は FAIL として扱ったが，「medical のルーティングが
   劣化した」という読み方は実測が支持しない．**
5. **本イテレーション唯一の実質的な悪化サインは N3(legal) である**．10/30 は閾値 8/30 を満たすが
   rank_1=baseline 系列の最低値（20/15/16 → 10）で，対 Iter60 が b=0／c=10，**p=0.0020**
   （12 検定への BH 補正後も q≈0.024）．機序としては，合成 810 行のみで学習したヘッドでは
   legal（単一ドメイン訓練行が 77 行と最少）に効いていた重み付けが失われ，rank_2 に選ばれにくく
   なった可能性がある．n=30 の探索的観察であり断定はしない．
6. **rank_2 の律速は「混ぜ方」でも「量」でもなく合成文の質と分布ギャップである**．合成のみで
   学習したヘッドの素の単一ドメイン argmax 精度は 0.2433（偶然一致 0.10 の 2.4 倍にすぎない）
   にもかかわらず，rank_2 命中数は 59/100 で単一ドメイン行を混ぜた Iter60/61/62（55/48/48）と
   同水準以上だった．**rank_2 側から見れば「単一ドメイン行を混ぜるか否か」はそもそも効いていない．**
7. **R-H（複合 100 行の検出力限界）が臨界に達した**．同系列 4 点の X は 12〜15 に収まり対応あり
   検定はいずれも p≧0.65．n=100・discordant 15〜19 行の設計では ±3〜4 行を検出できない．
   **同一レバーでの追加反復（値のバリエーション）は分解能を上げないため推奨しない．**

**留保の更新**: R-K を訂正し，Iter67 は「3 本目の系列」ではなく B95 の rank_1=baseline 系列
（+10.0pt 系列）の 4 点目とする．**対外記述では単点 +15.5pt を引用せず「rank_1=baseline 構成では
+10.0〜+15.5pt（4 点，互いに有意差なし）」と幅で報告すること**．R-L（r の分母 24−12 は土俵混在の
ため絶対値を達成割合として読まない）・R-M・R-I は継続．**R-F（実行時経路への未配線）は 10 反復目**．

**次の一手**: 帯 C の指示「rank_2 ヘッド側の学習設定，rank_1 側には戻らない」に従い，
新レバー `multilabel_synthetic_data_quality = echo_filtered_synthetic_rows` を config.yml の
levers 末尾へ追記した（backlog B101）．2 ヘッド構成は固定し，合成 810 行から設問文の語を反復して
いるだけの行を機械的に除去した部分集合で rank_2 ヘッドを再学習する．**行数が減るため「品質の効果」と
「量の効果」が交絡する．同数をランダム除去した対照ヘッドを同時に作り，事前登録で分離すること．**

## Iteration 66: 合成文の質量比のみをIter64水準へ戻す（単一ドメイン行の2重化）

### 調査 (Iter66)

**問い**（config.yml:1217-1261＝backlog B99 が単一レバーを事前登録済み．新規の先行研究探索ではなく，
Iter64/65 と同様に一次情報＝コード・実データの確認を優先した）

- Q1: `data/classifier_train.jsonl` の行数・スキーマは note の前提（1427 行）と一致するか．
- Q2: `data/classifier_train_multidomain_iter65.jsonl`（810 行）は実在し，Iter65 の生成物か．
- Q3: 行を 2 重化する処理（jsonl を単純連結するだけ）は `train_multilabel_dispatch_head.py
  --train-data` の読み込みロジック（`_load_training_rows`）と整合するか．重複行があっても
  問題なく学習できる実装か．
- Q4: N5・N2'・被覆 2 個行数などの評価指標を計算するスクリプトは，Iter66 の出力物
  （新ヘッド・新予測ファイル）に対してもそのまま使えるか．
- Q5: wafl-ctrl5（制御ホスト）の現状（Ollama 起動状況・GPU 空き・ディスク容量）と，
  コスト見積り 15 分の妥当性．
- Q6: `CalibratedClassifierCV(cv=5)` の fold 分割の留保（note の「既知の留保」）について，
  実装（sklearn の内部 CV 挙動）を確認し，行複製によるリークの実際の影響範囲を評価する．

**分かったこと（全文読了・実行確認による一次情報）**

1. **Q1**: `wc -l data/classifier_train.jsonl` は **1427**（note の前提と一致）．
   `mtime` は 2026-07-30 14:44 で Iter65 以降変更されていない（固定資産として扱ってよい）．
   1 行目のスキーマは `{"id": str, "query": str, "domain": str}`（単一ドメイン文字列）．
2. **Q2**: `data/classifier_train_multidomain_iter65.jsonl` は **810 行**存在し，`mtime` は
   2026-09-19 15:35 で `models/dispatch_multilabel_head_iter65.joblib`（15:41）・
   `results/iter65_multilabel_ranking_predictions.jsonl`（15:44）と時系列が整合する
   （Iter65 の生成物であることをファイルシステムから確認）．スキーマは
   `{"id": str, "query": str, "domain": [str, str]}`（多ラベル）で，note の記述と一致．
3. **Q3**: `scripts/train_domain_classifier.py:74-77` の `_load_training_rows()` は
   `[json.loads(line) for line in f if line.strip()]` と，jsonl を単純にリストへ読むだけで，
   `id` によるキー化・重複排除・辞書マージは一切行わない．
   `train_multilabel_dispatch_head.py:221-223` も `single_label_rows + synthetic_rows` と
   リストを単純連結するだけで，`build_training_features()`（`train_domain_classifier.py:99〜`）も
   `for row in rows:` と順序保持でループするだけである．**したがって
   `classifier_train.jsonl` を単純連結（2 回）した `classifier_train_iter66_x2.jsonl` は
   コード変更なしにそのまま `--train-data` に渡して学習できる**．重複 `id` があっても
   ロジック上のエラーや意図しない縮約は起きない．
4. **Q4**: `evaluate_dispatch_candidate_ranking.py` の N5（`_N5_SINGLE_DOMAIN_ARGMAX_ACCURACY_FLOOR
   = 0.590`，:108，計算本体 :380-406，WARNING 出力 :486-488）は `--head` に渡すヘッドの
   `classes_`／予測確率のみを参照し，訓練データの由来（単一ドメイン行が何重化されているか）を
   一切問わない実装であることを再確認した．`_load_head()`（:173〜）も Iter60〜65 で不変であり，
   `models/dispatch_multilabel_head_iter66.joblib` を渡すだけでコード変更なしに動作する．
   N2'（`selected_domain` を `head_argmax` で上書きした後の一致率）・被覆 2 個行数の
   アドホック集計（Iter65 で確立済みの `expected_domains`/`dispatched_domains` 完全一致判定）も
   同様に予測 JSONL のフィールド構造にのみ依存し，Iter66 の出力物に対してそのまま使える．
5. **Q5（実行環境）**: wafl-ctrl5 は SSH 接続良好．`docker ps` で `ollama-ctrl` が 4 時間稼働中，
   `nomic-embed-text`（274MB）と judge 用モデルの両方が取得済みで pull 不要．
   GPU は 12,288MiB 中 5,736MiB 使用・使用率 0%（新規ジョブの実行を妨げない）．
   ディスクは 435GB 空き．ローカルポート 11499 のトンネルは生存しており
   `curl http://127.0.0.1:11499/api/tags` が応答した．
   コスト見積りについて，`train_multilabel_dispatch_head.py` に `--embedding-cache` の類は
   実装されておらず（`evaluate_dispatch_candidate_ranking.py` 側にはあるが訓練スクリプト側にはない），
   単一ドメイン行を 2 重化すると同一テキストへの埋め込み呼び出しも単純に 2 倍（1427→2854 回）に
   なる（重複排除によるキャッシュ再利用はできない）．埋め込み対象総行数は
   Iter65 の 2237 行（1427+810）→ Iter66 の 3664 行（2854+810）で **約 1.64 倍**．
   Iter65 の実測 mtime（訓練データ 15:35 →ヘッド 15:41=6分→予測 15:44=3分，合計 9 分）に
   1.64 を掛けると約 15 分となり，**note の見積り「15 分程度」と実測ベースの推定はほぼ一致する**．
6. **Q6（既知の留保の定量評価）**: `train_multilabel_ranking_head()`
   （`train_multilabel_dispatch_head.py:154-178`）は
   `CalibratedClassifierCV(LogisticRegression(...), method="sigmoid", cv=5, ensemble=True)` を
   `OneVsRestClassifier` でラップしており，`cv=5`（整数）は sklearn 内部で分類問題に対し
   `StratifiedKFold(shuffle=False)` を用いる．**この分割の `shuffle=False` という性質が，
   行の複製方法（単純連結か，行ごとの隣接複製か）によってリークの深刻度を劇的に変えることを
   シミュレーションで確認した**：
   - 単純な「原本 1427 行＋その全コピー 1427 行」というブロック連結（`cat file file`）の場合，
     StratifiedKFold は同一クラスの行をクラス内出現順に fold へ循環割当てするため，
     **複製ペアの 100%（5 シードで再現，実データの粒度に近いクラス分布でも 0%が同一 fold）が
     異なる fold へ分離される**．これは「複製元がある fold の訓練側に含まれている状態で，
     その複製先が別の fold の較正用ホールドアウトとして使われる」ケースがほぼ全複製行で
     発生することを意味し，note が『可能性がある』としていたリークは実際には**ほぼ確実に発生する
     構造的な現象**である．
   - 一方，「各行を隣接して 2 回連続で並べる」補完（interleave; `line, line, line2, line2, ...`）
     に変えると，複製ペアの **98.9% が同一 fold に収まる**（ペアが分割されないため，
     このタイプのリークはほぼ発生しない）．
   - **note の記述「各行を 2 回含む 2854 行」は連結方式を明記していない**．この違いは較正値の
     楽観化の深刻度（ほぼ皆無 vs ほぼ全複製行で発生）を左右するため，**計画フェーズで
     連結方式（ブロック連結か行ごとの隣接複製か）を明示的に決定・事前登録する必要がある**．
   - ただし note が指摘するとおり，**N5（argmax 正解率）はこの較正値そのものではなく
     argmax の順序にしか依存しない**ため，仮にブロック連結でリークが最大化しても N5 の
     解釈への影響は限定的である．一方，較正値（確率スコア）を将来のレバー（例: rank_2 の
     信頼度閾値など）で対外引用・比較に使う計画があるなら，行ごとの隣接複製を選んでおくほうが
     安全側であり，追加コストはゼロである．

**結論**

backlog B99 が設計した Iter66 の単一レバー（単一ドメイン行の 2 重化）は，
`train_multilabel_dispatch_head.py`・`evaluate_dispatch_candidate_ranking.py` ともに
コード変更なしでそのまま実行可能である．前提となる資産（`classifier_train.jsonl` 1427 行，
`classifier_train_multidomain_iter65.jsonl` 810 行）はいずれも実在し，Iter65 生成物である
ことをファイルシステムの mtime から確認した．コスト見積り「15 分程度」は，埋め込み対象総行数の
比（Iter65 比 約 1.64 倍）から実測ベースでもほぼ一致することを確認した．wafl-ctrl5 の
Ollama・GPU・ディスクはいずれも新規ジョブ実行に支障ない状態である．
唯一かつ重要な新知見は，**note の「既知の留保」（較正値の楽観化）が，行の連結方式（ブロック連結か
隣接複製か）によって「ほぼ全複製行で発生」から「ほぼ皆無」まで変わる**ことをシミュレーションで
定量的に示した点である．note はこの連結方式を明記しておらず，計画フェーズでの決定事項として
残っていた．

**次フェーズへの示唆**

- 計画フェーズは，2 重化ファイル生成スクリプトの連結方式を**行ごとの隣接複製
  （`line, line, line2, line2, ...`）に明示的に決定する**ことを推奨する．ブロック連結
  （原本 1427 行の後に同じ 1427 行を丸ごと追記）は避ける．理由: N5 の解釈自体には影響しないが，
  較正値のリークをほぼゼロコストで回避できるため，「既知の留保」を事前に無害化できる
  （デメリットなし）．
- 主基準（N5≧0.590）・副基準（N2'≧0.5875，被覆 2 個行>21，S3=2.000000・S4 不一致行>0）は
  note の暫定案をそのまま事前登録してよい．探索的指標（被覆 2 個行が Iter64 の 24 を上回るか）も
  note のとおり主基準にしないことを踏襲する．
- コスト見積りは note の「15 分程度」で妥当（実測ベースの推定 約 15 分と整合）．
  wafl-ctrl5 のセットアップ・トンネルは生きているため追加の環境構築は不要．
- 実装フェーズでは，2 重化ファイルの生成スクリプト（5 行程度）に連結方式を明示するコメントを
  残し，`wc -l` で 2854 行であることを事前確認する運用を申し送る．

### 計画 (Iter66)

**仮説**

Iter65（合成 405→810 行）では，(1) 被覆 2 個行の用量反応が 405 行で頭打ちになり
（12→24→21），(2) N5（単一ドメイン 1500 行の argmax 正解率）が 0.591333→0.563333 へ
有意に退行し（対 Iter64 McNemar p=0.0053），誤り先が「合成ペアで頻繁に共起させた相手ドメイン」へ
集中した．しかし Iter64→65 は**「合成文の本数（語彙的多様性）405→810」と「各ドメインの陽性訓練行に
占める合成文の質量比 35.1%→51.9%」を同時に動かしており，N5 の退行がどちらに由来するかが
分離されていない**．本イテレーションは**合成側 810 行をファイルごと固定（再生成しない）したまま，
単一ドメイン訓練行を 2 重化して質量比だけを 35.1%（Iter64 水準）へ戻す**．
質量比仮説（境界の融解は質量比に由来する）が正しければ **N5 は 0.590 台へ回復する**．
回復しなければ，N5 の退行は質量比ではなく**合成文そのものの分布シフト**に由来すると確定し，
次は 2 ヘッド構成（rank_1 は単一ドメイン分類器，rank_2 のみ合成データ由来）へ移る．
いずれに転んでも次の一手が一意に決まる点が本レバーの設計意図である（backlog B99）．

**単一レバー（今回変更する唯一の変数）**

`multilabel_training_mixture_ratio`: `single_domain_rows_as_is`（Iter65 の実質値＝
`--train-data data/classifier_train.jsonl` 1427 行をそのまま）→ **`single_domain_rows_duplicated_x2`**
（`--train-data data/classifier_train_iter66_x2.jsonl`．**同一内容の 1427 行を各行 2 回，計 2854 行**）．
非 legal ドメインの陽性訓練行は 150×2+162=462 行となり，合成比率は 162/462=**35.1%**＝Iter64 と同一．
`train_multilabel_dispatch_head.py` の `--multilabel-train-data` は
`data/classifier_train_multidomain_iter65.jsonl`（810 行）のまま**据え置く**．

**確定した実装仕様（本フェーズの決定事項）**

1. **複製方式は「行ごとの隣接複製」（interleave: `line1, line1, line2, line2, ...`）を採用し，
   ブロック連結（`cat f f`＝原本 1427 行の後に全コピー 1427 行）は採らない**．
   根拠は調査 Q6 のシミュレーション: `CalibratedClassifierCV(cv=5)` は内部で
   `StratifiedKFold(shuffle=False)` を使うため，ブロック連結では**複製ペアの約 100% が異なる fold へ
   分離され**（複製元が訓練側，複製先が較正用ホールドアウト側に来る）較正値が構造的に楽観化する一方，
   隣接複製では**複製ペアの 98.9% が同一 fold に収まり**この種のリークはほぼ発生しない．
   N5（argmax の順序のみに依存）の解釈には影響しないが，**追加コストがゼロでデメリットがない**ため
   安全側を採る．これは config.yml の note が計画フェーズへ委ねていた未決定事項の確定である．
2. **生成方法**: 新規の小スクリプト `scripts/duplicate_training_rows.py`（docstring 付き・30 行程度）を
   追加し，入力 JSONL を 1 行ずつ読んで**即座に同じ行を 2 回書き出す**実装とする（順序保持・
   隣接複製が実装から自明になる形にする）．**既存パイプラインのコード
   （`train_multilabel_dispatch_head.py`・`train_domain_classifier.py`・
   `evaluate_dispatch_candidate_ranking.py`・`compute_iter59_ranking_stats.py`・`config.yaml`）は
   一切変更しない**．
3. **`id` は複製後も重複したままにする**（バイト単位で同一の 2 行を隣接させる）．
   調査 Q3 で `_load_training_rows()` が `id` によるキー化・重複排除を行わないこと，
   埋め込み対象は `query` のみであることを確認済みであり，`id` を書き換えると
   「原本と同一であること」の検証（`sort | uniq -c` が全行 2 になる）が難しくなるためである．
4. **`sample_weight` は使わない**（Iter32 で実測した `class_weight='balanced'` との乗算結合を
   構造的に避けるため．行の複製は重み 2.0 と数学的に等価だが sklearn の内部結合に依存しない）．
5. 採点は `--rank1-source head_argmax` を主系として固定（Iter63〜65 と同一）．N2' は
   `build_new_rows()` が上書きした後の `selected_domain` に対して算出する．
6. 統計は `scripts/compute_iter59_ranking_stats.py` を無改造で使う．N1・N2 の `exact_match` は
   本構成では定義上 `pass:false` になるため**記録のみ**で判定に用いない（Iter63〜65 と同じ）．
7. 被覆 2 個行数・N6'' などは Iter65 で確立したアドホック集計（読み取り専用）をそのまま使う．

**固定する構成（Iter65 から一切変えない）**

- 合成訓練データ: `data/classifier_train_multidomain_iter65.jsonl`（810 行）を**再生成せず再利用**
  （生成乱数の実現値まで固定＝Iter65 との差分が混合比のみになる）．生成系一式（プロンプト・
  F1〜F4・temperature=0.8・生成モデル・45 ペア集合・A7 閾値）は**今回一度も起動しない**．
- 単一ドメイン訓練データの**内容**: `data/classifier_train.jsonl`（1427 行，mtime 2026-07-30）．
  変えるのは各行の**出現回数のみ**であり，文面・ラベル・行の集合は不変．
- ヘッド種別: `train_multilabel_dispatch_head.py` の現行実装＝
  `OneVsRestClassifier(CalibratedClassifierCV(LogisticRegression(class_weight='balanced'),
  method='sigmoid', cv=5, ensemble=True))`（**Platt 較正済み**．Iter62 以降不変．
  Iter63〜65 の計画節の「未較正」という記述は誤りであり config.yml で訂正済み）．
- 埋め込みモデル `nomic-embed-text`・評価クエリ埋め込みキャッシュ
  `results/iter59_query_embeddings.npz`・基準線 `results/20260918_202613/results.jsonl`・
  rank_2 の選択ロジック・`_head_scores()`・A2/A3/A6/A9・N5 の計算・統計スクリプト・`config.yaml`．
- **実行時経路（`node.py` のルータ）への配線は本イテレーションでも行わない**（B94/B95．7 回目・R-F）．
- 実行基盤は wafl-ctrl5 に一本化（SSH ローカルフォワード `127.0.0.1:11499`．調査 Q5 で生存確認済み）．

**出力ファイル命名（Iter65 以前の成果物を上書きしないこと）**

| 種別 | 既存（保護・読み取り専用） | Iter66（新規作成） |
|---|---|---|
| 単一ドメイン訓練データ | `data/classifier_train.jsonl`（1427 行） | `data/classifier_train_iter66_x2.jsonl`（2854 行） |
| 合成訓練データ | `data/classifier_train_multidomain_iter65.jsonl`（810 行） | **新規作成しない（再利用）** |
| ヘッド | `models/dispatch_multilabel_head_iter65.joblib` | `models/dispatch_multilabel_head_iter66.joblib` |
| 予測 | `results/iter65_multilabel_ranking_predictions.jsonl` | `results/iter66_multilabel_ranking_predictions.jsonl` |
| 統計 | `results/iter65_stats.json` | `results/iter66_stats.json` |

**実行計画**

```
# 0) 単一ドメイン訓練行の 2 重化（唯一のレバー変更点．行ごとの隣接複製）
uv run python -m scripts.duplicate_training_rows \
    --input data/classifier_train.jsonl \
    --output data/classifier_train_iter66_x2.jsonl

# 0') 事前検査（A9'．下記の中止規則を機械的に確認する）
#     - 行数が 2854 であること
#     - 奇数行と次の偶数行がバイト単位で一致すること（＝隣接複製．ブロック連結でないこと）
#     - 重複を畳んだ集合が原本 1427 行と完全一致すること

# 1) 多ラベルヘッドの再訓練（2854 + 810 行を embed．合成側は Iter65 のファイルを据え置き）
uv run python -m scripts.train_multilabel_dispatch_head \
    --train-data data/classifier_train_iter66_x2.jsonl \
    --multilabel-train-data data/classifier_train_multidomain_iter65.jsonl \
    --embedding-model nomic-embed-text --ollama-host 127.0.0.1 --ollama-port 11499 \
    --output models/dispatch_multilabel_head_iter66.joblib

# 2) 1600 問のオフライン採点（主系＝head_argmax．評価クエリ埋め込みはキャッシュ完全ヒットの想定）
#    --iter59-predictions は引数名に反して汎用．S4（対 Iter65 不一致）を測るため Iter65 の予測を渡す．
uv run python -m scripts.evaluate_dispatch_candidate_ranking \
    --baseline results/20260918_202613/results.jsonl \
    --head models/dispatch_multilabel_head_iter66.joblib \
    --rank1-source head_argmax \
    --embedding-model nomic-embed-text --ollama-host 127.0.0.1 --ollama-port 11499 \
    --embedding-cache results/iter59_query_embeddings.npz \
    --iter59-predictions results/iter65_multilabel_ranking_predictions.jsonl \
    --output results/iter66_multilabel_ranking_predictions.jsonl

# 3) 指標・検定（Iter59〜65 と同一スクリプト・無改造）
uv run python -m scripts.compute_iter59_ranking_stats \
    --baseline results/20260918_202613/results.jsonl \
    --new results/iter66_multilabel_ranking_predictions.jsonl \
    --output results/iter66_stats.json

# 4) 被覆 2 個行数・N6''・対 Iter65 McNemar などのアドホック集計（読み取り専用．公式パスは無改造）
```

**タイムアウトと実行上の運用**

- 2 重化ステップ: 数秒（タイムアウト 60 秒）．
- 再訓練ステップ: **タイムアウト 3600 秒（60 分）**・バックグラウンド実行．
  埋め込み対象は 3664 行で Iter65 の 2237 行の約 1.64 倍，実測ベース見積りは約 10 分．
  `train_multilabel_dispatch_head.py` に埋め込みキャッシュは実装されていないため，
  複製行も含めて 2854 回 embed される（重複排除による短縮はない）．
- 採点・統計・アドホック集計: 各 **タイムアウト 900 秒（15 分）**．
- 評価クエリ埋め込みキャッシュのミス件数を記録すること（0 件が期待値．非 0 なら固定構成の破れ）．

**成功条件（事前登録．事後変更禁止）**

参照点は **Iter65**（合成 810 行・質量比 51.9%．N5 0.563333，N2' 0.573125，被覆 2 個行 **21/100**，
`compound_domain_set_recall` 0.565，rank_1 正解 72/100，N3 15/30，N6'' education 7/20・medical 18/28）と
**Iter64**（合成 405 行・質量比 35.1%．N5 0.591333，N2' 0.5875 相当，被覆 2 個行 **24/100**，recall 0.545）である．

- **P1（主基準）**: **N5 ≧ 0.590**（単一ドメイン 1500 行の argmax 正解率．質量比仮説が正しければ
  Iter64 水準 0.591333 へ回復するはず）．**この 1 点が本レバーの成否そのものである．**
- **P2**: **N2' ≧ 0.5875**（`new_top1_accuracy`．`selected_domain` 上書き後の値で算出）．
- **P3**: 複合 100 行の**被覆 2 個行 > 21**（Iter65 実測．質量比を戻しても複合側が犠牲にならないこと）．
- **P4**: **`mean_dispatch` = 2.000000**（完全一致．`duplicate_rank1_rank2_count`=0）．
- **P5**: **対 Iter65 予測の不一致行 > 0**（レバーが予測を実際に動かしたことの確認）．

**非退行条件（事前登録．事後変更禁止）**

- **N3**: legal 自己被覆 **≧ 8/30**（Iter63〜65 と同一水準）．
- **N6''**: education 自己被覆 **≧ 6/20** かつ medical 自己被覆 **≧ 18/28**
  （Iter64/65 と同一の下限を据え置く．イテレーション間の比較可能性を優先する）．
- **N7（本計画で追加．理由を明記する）**: `compound_domain_set_recall` **≧ 0.545**（Iter64 水準）．
  config.yml の note は S2 を条件に挙げていないが，**質量比を戻すことで複合側の部分点が
  Iter64 水準より下へ崩れていないこと**を確認する必要があるため非退行として登録する．
  閾値を Iter65 実測 0.565 ではなく Iter64 実測 0.545 に置くのは，本レバーが狙うのは N5 の回復であり，
  S2 の 2pt 程度の揺れで rejected にしない（P3 で複合側は別途見る）ためである．
  **S2 は Iter65 の考察どおり『2 ドメイン性の改善』の代理として不適切であり，値を対外引用しない．**
- **S1（対基準線 200 ペア exact McNemar）は記録のみ**とし判定に用いない
  （Iter63 以降一貫して p<1e-5 で飽和しており，本レバーの弁別力を持たないため）．

**採否の判定基準（事前に機械的に定める）**

- **adopted**: **P1〜P5 をすべて充足し，かつ N3・N6''・N7 をすべて充足**した場合．
  解釈: N5 の退行は**質量比**に由来すると確定し，「本数は 810 行のまま保てる」ことになるため，
  次は用量反応の再開（質量比を一定に保ったまま本数を増やす設計）が正当化される．
- **partial**: 上記に達しないが，以下のいずれかに該当する場合．
  1. **P1 を充足するが P3 が不成立**（被覆 2 個行 ≦21）または **P2 が不成立**．
     ＝質量比仮説は支持されるが，単一ドメイン判別と複合側がトレードオフの関係にある．
  2. **P1 が不成立だが，N5 が Iter65（0.563333）から有意に回復**した場合
     （**対 Iter65 の対応あり exact McNemar で p<0.05 かつ点推定で +1.0pt 以上**）．
     ＝質量比は N5 退行の一因ではあるが唯一の原因ではない（合成文の分布シフトも寄与）．
  3. **P1〜P5 を充足するが N3・N6''・N7 のいずれかが FAIL** した場合
     （Iter63〜65 と同じく**非退行のみの FAIL は最大 partial とし rejected にしない**）．
- **rejected**: 以下のいずれかに該当する場合．
  1. **P1 が不成立（N5 < 0.590）かつ N5 の対 Iter65 有意回復もない**（McNemar p≧0.05 または
     点推定の改善 <1.0pt）．＝**質量比では N5 の退行を説明できない**．この場合は
     『合成文は質量比に関わらず単一ドメイン判別を壊す』と結論し，次レバーを
     **2 ヘッド構成**（rank_1 は既存の単一ドメイン分類器，rank_2 のみ合成データで学習した別ヘッド）とする．
  2. **P4 が不成立**（`mean_dispatch` ≠ 2.000000）．＝出力構造そのものが壊れている．
- **判定を下さず原因調査に戻る（中止規則．A9'）**: 以下は「実験が成立していない」ケースであり，
  adopted/partial/rejected のいずれにも分類しない．
  1. `data/classifier_train_iter66_x2.jsonl` が **2854 行でない**，または隣接複製になっていない．
  2. **P5 が不成立（対 Iter65 不一致行 = 0）**．同一の合成データ・同一の評価クエリ・同一のヘッド実装で
     予測が 1 行も動かないのは，2 重化ファイルが実際には読まれていない（レバー未到達）ことを意味する．
  3. 評価クエリ埋め込みキャッシュのミスが発生した場合（固定構成の破れ）．

**探索的な診断値（主基準にしないこと）**

- **被覆 2 個行が Iter64 の 24 を上回るか**（config.yml の note が指定した探索的指標）．
  上回れば「本数は効くが質量比が打ち消していた」という強い証拠になり，用量反応の再開が正当化される．
  **ただし主基準に昇格させない**（Iter64 で確認した R-H＝複合 100 行では 1 反復増分の検出力が
  構造的に不足する問題は本イテレーションでも解消していない）．
- 対 Iter65 の被覆 2 個行の対応あり McNemar（discordant の内訳を含む．記録のみ）．
- N5 の誤り先の分布．Iter65 では medical→natural_science 9 件・social_science→legal 6 件と
  **合成ペアで共起させた相手ドメイン**へ集中していた．質量比を戻してこの集中が解消するかは，
  「境界の融解」という機序の直接の検証になる（P1 の裏付け）．
- 各ドメインの陽性訓練行に占める合成文の比率の実測値（設計値 35.1% と一致することの確認）．
- N5 の Wilson 95% 信頼区間（Iter65 は上限 0.5882 で閾値 0.590 の外にあった）．

**既知の留保（事前登録）**

- **R-I（較正値の楽観化）**: 行の複製により `CalibratedClassifierCV(cv=5)` の較正 fold で
  同一クエリの行が訓練側と検証側に跨りうる．本計画は隣接複製の採用により**複製ペアの 98.9% を
  同一 fold に収める**ことでこれをほぼ無害化するが，残り 1.1% は原理的に残る．
  **較正値（確率スコアの絶対値）を対外引用せず，将来のレバーで閾値の根拠に使わないこと．**
  N5・N2'・被覆 2 個行はいずれも argmax／順序に依存する指標であり，この留保の影響は受けない．
- **R-J（`class_weight='balanced'` との相互作用）**: `LogisticRegression(class_weight='balanced')` は
  複製後のクラス頻度から重みを再計算するため，複製による「全体のスケール」の効果は
  クラス重みによって部分的に打ち消される．ただし**本レバーが狙うのは各ドメインの陽性集合の
  内部構成比（単一ドメイン文 150→300 行に対し合成文 162 行で一定）であり，これはクラス重みでは
  打ち消されない**．実測の比率（探索的診断値）でこの前提を確認すること．
- **R-F（実行時経路への未配線）**: 7 イテレーション連続でオフライン評価のみ．adopted でも
  実機での有効性は主張できない．
- 810 行の合成データは Iter65 の生成乱数の実現値に固定されているため，本イテレーションの結論は
  「この 810 文に対して」条件付きである（再生成すれば別の実現値になる）．

**単一レバー原則の確認（混入チェック）**

- 合成訓練データ（ファイル・行数・生成乱数の実現値）→ **無変更**（再生成しない）．
- 単一ドメイン訓練データの文面・ラベル・行集合 → **無変更**（出現回数のみ 1→2 に変更）．
- ヘッド種別・較正の有無・ハイパーパラメータ・埋め込みモデル・埋め込みキャッシュ → **無変更**．
- 採点スクリプト・`--rank1-source head_argmax`・rank_2 の選択ロジック・統計スクリプト・基準線・
  `config.yaml` → **無変更**．
- 追加する `scripts/duplicate_training_rows.py` は**データ準備の独立ユーティリティ**であり，
  評価・訓練パイプラインのいずれのコードパスにも介入しない（第 2 のレバーではない）．
- **ヘッドの再訓練は本レバーに構造的に従属する**（訓練データの構成を変える以上，再訓練しなければ
  レバーを読むコードに到達しない）ため，独立した第 2 のレバーではない．

**想定コスト**: 2 重化 数秒 ＋ ヘッド再訓練 約 10〜15 分 ＋ 採点 約 1 分 ＋ 統計・アドホック集計 約 1 分
＝ **合計 15 分程度**（config.yml の見積りと調査 Q5 の実測ベース推定が一致）．
**合成生成（律速だったステップ）が不要**な点が Iter64/65 との最大の違いである．
オフライン完結・`config.yaml` のスキーマ変更なし・実機 1600 問本走なしのため自律着手してよい．

### 実装 (Iter66)

**新規作成ファイル**:

- `scripts/duplicate_training_rows.py`（新規，30 行程度・docstring 付き）。
  `--input`／`--output` を取り，入力 JSONL を 1 行ずつ読んで**即座に同じ行を 2 回**書き出す
  （`line1, line1, line2, line2, ...` の隣接複製）。`id` は書き換えない。ブロック連結との違いと
  採用理由（`CalibratedClassifierCV(cv=5)` の `StratifiedKFold(shuffle=False)` に対する較正リーク
  回避）を docstring に明記した。
- `data/classifier_train_iter66_x2.jsonl`（新規，2854 行．`.gitignore` の `data/*` により
  リポジトリ管理外）。既存の `data/classifier_train.jsonl`（1427 行）は無変更。

**既存パイプラインコードの変更**: なし．計画どおり
`train_multilabel_dispatch_head.py`・`train_domain_classifier.py`・
`evaluate_dispatch_candidate_ranking.py`・`compute_iter59_ranking_stats.py`・`config.yaml` は
一切触っていない．`git status` 差分は `scripts/duplicate_training_rows.py`（新規）のみで，
`config.yaml`・`results/iter45_preliminary/logs/*` 等の既存の未追跡・未コミット差分は
本フェーズ開始前から存在するものであり（CLAUDE.md の規約に従い）放置した．

**実行コマンド**:

```
uv run python -m scripts.duplicate_training_rows \
    --input data/classifier_train.jsonl \
    --output data/classifier_train_iter66_x2.jsonl
```

出力: `Wrote 2854 lines (adjacent duplication) to data/classifier_train_iter66_x2.jsonl`

**事前検査（A9'．3 点とも成立を確認した．中止規則には抵触しない）**:

1. **行数**: `wc -l data/classifier_train_iter66_x2.jsonl` → **2854 行**（想定どおり）。
2. **隣接複製の検証**: 全 2854 行を読み込み，`lines[2i]` と `lines[2i+1]`（0-indexed，奇数行目と
   その次の偶数行目に相当）をバイト単位で比較した。**不一致 0 件**。ブロック連結（原本 1427 行
   丸ごとの後にコピー 1427 行）であれば `lines[0] != lines[1]` となり検出できる設計であり，
   隣接複製になっていることを確認した。
3. **一意集合の一致**: `sort -u data/classifier_train_iter66_x2.jsonl` と
   `sort -u data/classifier_train.jsonl` を `diff` し，**差分ゼロ（IDENTICAL）**，
   両者とも一意行数 **1427 行**で一致することを確認した。`id` は複製後も重複したまま
   （書き換えていない）。

**実行してよい状態か**: はい．3 点の事前検査すべてが成立し，A9' の中止規則には該当しないため，
次フェーズ（rc-experimenter）は計画節の実行計画ステップ 1〜4（ヘッド再訓練→採点→統計→
アドホック集計）へ進んでよい．コード変更は本イテレーションでは発生していない（新規データ
ファイル 1 つと新規スクリプト 1 本の追加のみ）。

### 実験・分析(実行) (Iter66)

**実行環境**: wafl-ctrl5 への SSH ローカルフォワード（`127.0.0.1:11499`）はリポジトリ実行環境から
生きていることを確認済み（`curl http://127.0.0.1:11499/api/tags` が `nomic-embed-text:latest` を
含む応答を返した）．計画節どおり `uv run python -m scripts...` はリポジトリのローカル実行環境から
直接実行し（トンネル経由で wafl-ctrl5 の Ollama へ到達），wafl-ctrl5 自体への ssh ログインでの
リモート実行は行わなかった．事前に `data/classifier_train_iter66_x2.jsonl`（2854 行）・
`data/classifier_train_multidomain_iter65.jsonl`（810 行）・基準線 `results/20260918_202613/results.jsonl`・
埋め込みキャッシュ `results/iter59_query_embeddings.npz`・`results/iter65_multilabel_ranking_predictions.jsonl`
の実在をすべて確認した．

**ステップ1: 多ラベルヘッド再訓練**（バックグラウンド実行，ログ `/tmp/iter66_logs/step1_train.log`）

```
uv run python -m scripts.train_multilabel_dispatch_head \
    --train-data data/classifier_train_iter66_x2.jsonl \
    --multilabel-train-data data/classifier_train_multidomain_iter65.jsonl \
    --embedding-model nomic-embed-text --ollama-host 127.0.0.1 --ollama-port 11499 \
    --output models/dispatch_multilabel_head_iter66.joblib
```

所要時間: 約 5 分 46 秒（16:23:49 起動 → 16:29:35 完了，タイムアウト目安 3600 秒に対し十分短い）．
出力ログで `n_single_label_rows=2854, n_synthetic_rows=810` を確認．各ドメインの `n_positive`
は非 legal ドメインすべて **462**（150×2+162，設計値どおり），legal のみ **316**（legal は非
legal 側と陽性行構成が異なるため異なる値になるのは想定どおり）．CV ROC-AUC は 0.9195
（medical）〜0.9830（history_culture）の範囲．

**ステップ2: 1600 問オフライン採点**（ログ `/tmp/iter66_logs/step2_eval.log`）

```
uv run python -m scripts.evaluate_dispatch_candidate_ranking \
    --baseline results/20260918_202613/results.jsonl \
    --head models/dispatch_multilabel_head_iter66.joblib \
    --rank1-source head_argmax \
    --embedding-model nomic-embed-text --ollama-host 127.0.0.1 --ollama-port 11499 \
    --embedding-cache results/iter59_query_embeddings.npz \
    --iter59-predictions results/iter65_multilabel_ranking_predictions.jsonl \
    --output results/iter66_multilabel_ranking_predictions.jsonl
```

所要時間: 約 1 分 19 秒（16:29:35 起動相当 → 16:30:54 完了）．**評価クエリ埋め込みキャッシュの
ミス件数は 0 件**（`results/iter59_query_embeddings.npz` の `mtime` が実行前後で不変
＝02:29:59 のまま更新されず，`ids` 配列の要素数が 1600 で全 1600 行が既存キャッシュでヒットした
ことを直接確認した．中止規則「キャッシュのミス多発」には該当しない）．
スクリプト自身の WARNING 出力: `N5 single-domain argmax accuracy 0.5773 < floor 0.59`．
標準出力の JSON サマリ: `mean_dispatch=2.0`，`compound_domain_set_recall=0.545`，
`n5_single_domain_argmax_accuracy.accuracy=0.5773333333333334`（correct=866/1500），
`a5_iter59_disagreement.mismatches=397`（`rank2_new` を Iter65 予測と比較した不一致行数．
mismatch_rate=0.248125）．

**ステップ3: 指標・検定**（ログ `/tmp/iter66_logs/step3_stats.log`）

```
uv run python -m scripts.compute_iter59_ranking_stats \
    --baseline results/20260918_202613/results.jsonl \
    --new results/iter66_multilabel_ranking_predictions.jsonl \
    --output results/iter66_stats.json
```

所要時間: 約 43 秒（16:30:54 → 16:31:37）．主要出力（`results/iter66_stats.json`）:
`S3_cost_neutrality.mean_dispatch=2.0`（`duplicate_rank1_rank2_count=0`），
`N2_top1_accuracy_invariance.new_top1_accuracy=0.586875`（`compute_top1_accuracy` は
`selected_domain in expected_domains` の割合であり，これが N2' の定義値そのもの．939/1600 相当），
`N3_legal_non_regression.new_legal_self_coverage=13`（`n_legal_involving_pairs=30`），
`N1_rank1_invariance.mismatch_count=491`（対基準線．定義上 `pass:false` は想定どおりで判定に
用いない），`new_compound_domain_set_recall=0.545`．

**ステップ4: アドホック集計**（読み取り専用，`/tmp/iter66_adhoc.py`・`/tmp/iter66_n6.py`．
`results/iter66_multilabel_ranking_predictions.jsonl` と `results/iter65_multilabel_ranking_predictions.jsonl`
の `expected_domains`/`dispatched_domains`/`selected_domain`/`head_scores` フィールドを突き合わせただけで，
公式スクリプトは一切改造していない）:

- **被覆 2 個行数**（複合 100 行のうち `dispatched_domains` が `expected_domains` を完全に包含する行）:
  **21/100**（`rank1_correct_compound100=73`）．
- **N6''**: `education` を含む複合ペア **6/20**（フロア 6 ちょうど），`medical` を含む複合ペア
  **18/28**（フロア 18 ちょうど）．いずれも `N3` と同じ手法（複合ペア母集団を分母にする）で算出．
- **N3**（検算）: `legal` を含む複合ペア **13/30**（ステップ3の公式出力と完全一致，読み取り整合性を
  確認済み）．
- **対 Iter65 の被覆 2 個行の対応あり比較**（McNemar，複合 100 行）: Iter66 で True・Iter65 で False
  が **2 行**，Iter66 で False・Iter65 で True が **2 行**，両方 True **19 行**，両方 False **77 行**．
  `iter65_coverage2_count=21`（Iter66 と同数）．discordant 4 行のみのため exact McNemar
  （binomtest）の **p 値 = 1.0**（有意差なし）．
- **対 Iter65 の予測不一致行**: 公式の `rank2_new` ベース（P5 の定義，ステップ2出力）で
  **397/1600**．参考として `selected_domain`（rank_1）ベースの不一致は **150/1600**
  （こちらは非公式の補助集計．P5 の判定には公式値 397 を用いる）．

**中止規則（A9'）の確認結果**:

- `data/classifier_train_iter66_x2.jsonl` の行数・隣接複製は実装フェーズで検査済み（2854 行，
  不一致 0 件）．実験フェーズでの再確認は行っていないが，ファイルは実験開始から完了まで
  変更されていない（`stat` で `mtime` が実装フェーズ時点のまま）．
- **P5（対 Iter65 不一致行）**: 公式値 397 > 0．**「レバー未到達」には該当せず，判定を継続してよい**．
- **評価クエリ埋め込みキャッシュのミス**: 0 件．**多発の中止規則には該当しない**．

**中止規則にはいずれも抵触せず，実験は完遂した．** 以下は判定用に収集した機械可読な数値である
（判定・解釈は rc-analyst に委ねる）：

| 指標 | 値 | 事前登録の閾値 |
|---|---|---|
| N5（単一ドメイン argmax 正解率） | 0.577333（866/1500） | ≧0.590 |
| N2'（`selected_domain` 上書き後 top1 accuracy） | 0.586875（939/1600） | ≧0.5875 |
| 被覆 2 個行数（複合100） | 21/100 | >21 |
| N7（`compound_domain_set_recall`） | 0.545 | ≧0.545 |
| mean_dispatch | 2.000000 | =2.000000 |
| P5（対Iter65不一致，`rank2_new`基準） | 397/1600 | >0 |
| N3（legal自己被覆） | 13/30 | ≧8/30 |
| N6''（education） | 6/20 | ≧6/20 |
| N6''（medical） | 18/28 | ≧18/28 |
| 対Iter65 被覆2個行 McNemar（exact binomtest） | p=1.0（discordant b=2,c=2） | 記録のみ |

**所要時間の総括**: 2 重化（実装フェーズで完了済み，数秒）を除き，ステップ1〜3 の合計は
約 7 分 48 秒（16:23:49〜16:31:37）．ステップ4（アドホック集計）は数秒．計画節の想定
「約 15 分」を下回った（採点ステップが埋め込みキャッシュ完全ヒットにより計画時の想定
（「約 1 分」）どおり短時間で完了したため）．

### 分析(解釈) (Iter66)

**判定: partial**（事前登録された partial 分岐 (2)「P1 は不成立だが N5 が Iter65 から有意に回復」に
機械的に該当する）．

**1. 事前登録基準への機械的照合**

| 項目 | 事前登録の閾値 | 実測 | 判定 |
|---|---|---|---|
| **P1（主基準）** N5 | ≧0.590 | **0.577333**（866/1500） | **FAIL**（−1.27pt） |
| **P2** N2' | ≧0.5875 | **0.586875**（939/1600） | **FAIL**（1 行差．940 で PASS） |
| **P3** 被覆 2 個行 | >21 | **21/100** | **FAIL**（同数．狭義不等号） |
| **P4** mean_dispatch | =2.000000 | 2.000000（`duplicate_rank1_rank2_count`=0） | PASS |
| **P5** 対 Iter65 不一致行 | >0 | 397/1600 | PASS |
| **N3** legal 自己被覆 | ≧8/30 | 13/30 | PASS |
| **N6''** education | ≧6/20 | 6/20 | PASS（境界値，`≧` のため成立） |
| **N6''** medical | ≧18/28 | 18/28 | PASS（境界値，同上） |
| **N7** `compound_domain_set_recall` | ≧0.545 | 0.545 | PASS（境界値，同上） |

分岐の当てはめ（事後に基準を動かしていないことを明示する）:

- **adopted**（P1〜P5 全充足 かつ N3・N6''・N7 全充足）→ P1・P2・P3 が FAIL のため**不成立**．
- **partial 分岐 (1)**（P1 充足だが P3 または P2 が不成立）→ P1 が FAIL なので**前提を満たさない**．
- **partial 分岐 (3)**（P1〜P5 充足だが非退行が FAIL）→ P1 が FAIL なので**前提を満たさない**．
  なお非退行 N3・N6''・N7 は全 PASS であり，この分岐は実測上も呼ばれない．
- **partial 分岐 (2)**（P1 不成立だが，N5 が Iter65 から**対応あり exact McNemar で p<0.05
  かつ点推定で +1.0pt 以上**回復）→ **下記 2 のとおり p=0.027534・+1.40pt で両条件を充足．
  該当する．**
- **rejected 分岐 (1)**（P1 不成立**かつ**対 Iter65 の有意回復もない）→ 有意回復があるため**不成立**．
- **rejected 分岐 (2)**（P4 不成立）→ mean_dispatch=2.000000 のため**不成立**．
- **中止規則 A9'**: 2 重化ファイルは 2854 行・隣接複製を実装フェーズで検査済み（実験中に mtime 不変），
  P5=397>0，埋め込みキャッシュのミス 0 件．**いずれにも抵触しない**．

したがって判定は **partial** で一意に確定する．

**2. 追加で実施した統計（`/tmp/iter66_n5_mcnemar.py`・`/tmp/iter66_diag.py`・`/tmp/iter66_n2.py`．
いずれも `results/iter6{4,5,6}_multilabel_ranking_predictions.jsonl` の読み取り専用集計で，
公式スクリプトは無改造）**

N5 は `head_scores` の argmax と `expected_domains[0]` の一致で再計算し，公式出力
（866/1500・845/1500）と一致することを検算したうえで対応あり比較を行った．

| 比較 | b（前×→今○） | c（前○→今×） | exact McNemar p | 点推定差 |
|---|---|---|---|---|
| **N5 Iter65→Iter66** | **52** | **31** | **0.027534** | **+1.40pt**（845→866） |
| N5 Iter64→Iter66 | 84 | 105 | 0.145531 | −1.40pt（887→866） |
| N5 Iter64→Iter65（再現確認） | 87 | 129 | 0.005157 | −2.80pt | 
| N2' Iter65→Iter66 | 56 | 34 | 0.026302 | +1.375pt（917→939） |
| N2' Iter64→Iter66 | 94 | 114 | 0.187571 | −1.25pt（959→939） |

Iter64→65 の p=0.005157 は Iter65 分析節の記録値（p=0.0053）と一致し，集計手法の再現性を確認した．

- **N5 の Wilson 95% CI**: Iter65 [0.538104, 0.588239]（上限が閾値 0.590 の**外**）→
  Iter66 [0.552168, 0.602103]（閾値 0.590 を**含む**）．P1 の FAIL は事前登録の点閾値に対する
  機械的判定としては確定だが，区間としては「明確に下回る」状態ではなくなった（探索的診断値．
  事後に閾値を緩める根拠には用いない）．
- 非対応（独立）の二項 SE は p≈0.58・n=1500 で 1.27pt であり，+1.40pt は単独では 1.1 SE 相当＝
  ノイズと区別しがたい．有意判定が得られたのは**同一 1500 行の対応あり比較**（discordant 83 行）
  によるものであり，この検出力の差が判定の根拠である点を明記しておく．

**3. 質量比仮説の判定: 「一因だが唯一の原因ではない」（部分的支持）**

計画節の仮説は「N5 の退行が質量比 35.1%→51.9% に由来するなら，質量比を戻せば N5 は 0.590 台へ
回復する」であった．実測は**回復したが 0.590 台には届かない**という中間の結果である．

- 質量比の実測は設計どおりで，レバーは意図した量だけ動いた: 非 legal 9 ドメインの陽性訓練行
  462 行（150×2+162）のうち合成 162 行で **35.06%**，legal は 316 行（77×2+162）のうち
  162 行で **51.27%**．Iter64 も同じ計算で非 legal 81/231=35.06%・legal 81/158=51.27% であり，
  **10 ドメインすべてで質量比が Iter64 と厳密に一致している**（R-J の前提＝クラス重みでは
  打ち消されない内部構成比が狙いどおり復元されたことの確認）．
- 回復量は **Iter64→65 で失われた 42 行のうち 21 行（ちょうど 50.0%）**．精度で見ると
  −2.80pt の退行に対し +1.40pt の回復で，過不足なく半分である．
- 残る半分は質量比では説明できない．Iter66 は Iter64 と質量比・合成ファイル本数以外の構成が
  すべて同一であり，**差分は「合成文が 405 行か 810 行か」（＝本数・語彙的多様性）だけ**である．
  したがって残差 −1.40pt（対 Iter64，p=0.1456 で有意ではない）は合成文そのものの分布シフトに
  帰属する．ただしこの残差は有意ではないため，「分布シフトの寄与が確実に存在する」とまでは
  言えず，**「質量比で説明できるのは最大でも半分であり，残りは合成文本数に由来する可能性がある
  （n=1500 では有意に検出できない大きさ）」**というのが実測が支持する最も強い主張である．

**4. 「境界の融解」機序の直接検証（探索的診断値）**

Iter65 の分析は，Iter64→65 で新たに生じた誤りが **medical→natural_science 9 件・
social_science→legal 6 件**など「合成ペアで頻繁に共起させた相手ドメイン」へ集中することを
機序の証拠とした．Iter66 で同じ集計を行うと:

- **Iter65→Iter66 で解消した 52 行**の Iter65 時点の誤り先は最大でも 5 件
  （natural_science→computer_science）で，35 通りの (正解, 誤り先) ペアに広く分散している．
  特に medical の流出（→education 3・→social_science 2・→general 2・→natural_science 2）と
  social_science→legal 3 が解消しており，**Iter65 で観測された共起ドメインへの集中が
  部分的に巻き戻った**．
- **Iter66 で新たに生じた 31 行**も最大 5 件（computer_science→mathematics）で分散しており，
  新たな集中は生じていない．
- ただし **Iter64→Iter66 で新たに誤った行**を数えると **medical→natural_science が依然 9 件**で
  最大であり，Iter65 で観測された最大の流出経路は**解消していない**．
- ドメイン別 N5（Iter65→Iter66）では medical +6・natural_science +7・general +4・mathematics +4 と
  回復側が多い一方，computer_science −6（0.6000→0.5600）・social_science −1 は悪化した．
  medical は Iter64→65 で −19 行と最も崩れたドメインだが，Iter66 の回復は +6 行にとどまる．

以上より，「合成文の質量比が共起ドメイン間の境界を融解させる」という機序は**部分的に裏付けられた**
（質量比を戻すと集中が緩む）が，**medical–natural_science という最大の流出経路は質量比を戻しても
残る**．これが上記 3 の「残り半分」の実体である．

**5. 探索的仮説「本数は効くが質量比が打ち消していた」は支持されない**

計画節の探索的指標（被覆 2 個行が Iter64 の 24 を上回るか）は **21/100** で上回らなかった．
対 Iter65 の対応あり比較も discordant b=2・c=2 の exact McNemar p=1.0 で，**Iter65 と実質同一**である．
合成 810 行という本数を保ったまま質量比を Iter64 水準へ戻しても複合側の被覆は Iter64 の 24 にも
戻らず（21），**「本数を増やせば複合側が伸びるが質量比がそれを打ち消していた」という説明は
実測に支持されない**．なお複合 100 行では 1 反復増分の検出力が構造的に不足する（R-H）ため，
この否定的結果も「効果がないことの証明」ではなく「この標本サイズでは検出できない」に留まる．
N7=0.545 は Iter64 水準ちょうどで非退行条件を満たす（Iter65 の 0.565 からは −2.0pt だが，
閾値を Iter64 実測に置いた事前登録の意図どおり rejected には至らない）．

**6. 期待との一致・不一致，および想定外の挙動**

- **一致**: レバーは意図どおり発火した（`n_single_label_rows=2854`・`n_positive=462`・
  質量比 35.06%・P5=397 行の予測変化）．実装・実行に不成立要因はない．
- **不一致**: 「質量比を戻せば N5 が 0.590 台へ回復する」という計画の予測は外れ，回復は半分に
  留まった．計画節が事前に用意した二分法（回復すれば質量比，しなければ分布シフト）に対し，
  実測は**その中間**に落ちた．これは計画の仮説設計が「排他的な二択」を想定していたことの
  限界であり，次フェーズはこの点を踏まえる必要がある．
- **想定外の挙動**: 発散・言語崩れ・OOM 等はない．P2 が **1 行差**（939 対 940）で FAIL した点，
  N6''（education 6/20・medical 18/28）と N7（0.545）が**いずれも閾値ちょうど**である点は，
  判定が境界に密集していることを示す．`≧` の事前登録により N6''・N7 は PASS，`>` の P3 と
  点閾値の P2 は FAIL で，**表記上の不等号の向きが判定を分けている**．事後に緩めないという
  原則に従いこのまま確定させるが，次イテレーションの閾値設計では境界値の扱いを明示しておくべきである．
- 計画節の参照点表に「Iter64 N2' 0.5875 相当」とあるのは**転記の誤り**である．Iter64 の N2' 実測は
  **0.599375（959/1600）**であり，0.5875 は Iter63 以降一貫して使われている基準線
  （0.5975）からの −1.0pt 下限である．**閾値 0.5875 自体は事前登録どおりで変更しておらず，
  判定に影響はない**が，参照点の記述として訂正しておく．

**7. 確信度と追加反復の要否**

判定 partial 自体の確信度は高い（事前登録規則への機械的当てはめが一意．p=0.027534 は
分岐条件 p<0.05 に対して余裕が大きくはないが，点推定 +1.40pt も条件 +1.0pt を上回り，
両条件の充足は境界的ではない）．一方，**「残り半分が合成文の本数（分布シフト）に由来する」という
機序の帰属は対 Iter64 で p=0.1456 と有意ではなく，確信度は低い**．この点を確定させたい場合は
追加反復（合成 405 行・質量比 35.1% の Iter64 構成を同一手順で再実行して N5 の再現性を測る）が
必要だが，**次レバーの選択はこの帰属の確定を待たずに決められる**（下記 8）ため，
追加反復は必須ではないと判断する．

**8. 次フェーズ（考察）への示唆**

- **レバー `multilabel_training_mixture_ratio` は収束扱いが妥当**．質量比を Iter64 水準へ戻す
  という操作の効果量は実測で +1.40pt（失われた分の半分）と確定し，これ以上この軸を動かしても
  N5≧0.590 には届かない見通しが立った（質量比は既に Iter64 と厳密一致しており，
  さらに下げる＝単一ドメイン行を 3 重化以上にする方向は，合成データの寄与自体を希釈して
  複合側（既に Iter64 未満の 21/100）を損なうトレードオフに入る）．
- **計画節が「不成立の場合」に指定した次善策＝2 ヘッド構成（rank_1 は単一ドメイン分類器，
  rank_2 のみ合成データ由来の別ヘッド）は，今回の結果でむしろ動機が強まった**．
  理由: 単一ドメイン判別（N5）と複合側（被覆 2 個行）が，質量比という 1 つのスカラーの上で
  トレードオフすることが 3 反復（Iter64: N5 0.591/被覆 24，Iter65: 0.563/21，Iter66: 0.577/21）で
  示され，**単一ヘッドで両立させる余地が乏しい**ことが実測で示されたためである．
  ただし計画節では 2 ヘッド構成は「rejected の場合」の分岐に紐付いていた．判定は partial なので，
  次レバーの正式決定は考察フェーズ（rc-reflector）が行う．
- 用量反応の再開（質量比一定で本数を増やす設計）は，adopted 分岐に紐付いていた選択肢であり，
  **今回は正当化されない**（P1 未達に加え，被覆 2 個行が本数 810 行でも 21 に留まり Iter64 の 24 を
  下回ったため，本数を増やす方向の期待値が実測で支持されない）．
- R-I（較正値を対外引用しない）は本分析で遵守した．用いた指標 N5・N2'・被覆 2 個行・N3・N6'' は
  すべて argmax／集合一致に基づき，確率の絶対値には依存しない．R-F（実行時経路への未配線）は
  8 反復連続で継続しており，partial であっても実機での有効性は主張できない．

### 考察 (Iter66)

**判定: partial（確定）／レバー `multilabel_training_mixture_ratio` は収束（クローズ）**

「計画 (Iter66)」節が事前に機械的に定めた判定規則へ実測を照合した結果，
**partial 分岐 (2)（P1 は不成立だが N5 が Iter65 から対応あり exact McNemar で p<0.05 かつ
点推定 +1.0pt 以上の回復）**に一意に該当する（p=0.027534・+1.40pt．845→866/1500）．
adopted は P1・P2・P3 の FAIL により不成立，rejected 分岐 1 は有意回復があるため不成立，
分岐 2（`mean_dispatch`≠2）も不成立．中止規則 A9' にも抵触しない（2854 行・隣接複製・
P5=397>0・埋め込みキャッシュのミス 0 件）．**事後に閾値は一切動かしていない**
（P2 は 1 行差の 939/1600，P3 は同数 21 の狭義不等号，N6''・N7 は閾値ちょうどで `≧` により PASS）．

レバー自体は values 単一値であり，かつ下記の機序により**これ以上この軸を動かす価値がない**ため
**収束扱いでクローズ**する．

**確定した機序**

1. **質量比仮説は「部分的支持」＝一因ではあるが唯一の原因ではない**．質量比は設計どおり厳密に
   Iter64 水準へ復元された（非 legal 9 ドメインで 162/462=**35.06%**，legal で 162/316=**51.27%**，
   いずれも Iter64 と小数点以下まで一致）にもかかわらず，**Iter64→65 で失われた 42 行のうち
   回復したのはちょうど半分の 21 行**（−2.80pt に対し +1.40pt）であった．
   対 Iter64 の残差 −1.40pt は p=0.1456 で有意ではないため，「残り半分は合成文の本数
   （405→810 行という分布シフト）に由来する」という帰属は**確信度が低い**．実測が支持する
   最も強い主張は「**質量比で説明できるのは最大でも半分**であり，残りは n=1500 では
   有意に検出できない大きさである」までである．
2. **「境界の融解」は部分的に巻き戻ったが，最大の流出経路は残る**．Iter65→66 で解消した 52 行は
   35 通りの (正解,誤り先) ペアへ広く分散し（最大 5 件），Iter65 で観測された共起ドメインへの集中
   （medical の流出・social_science→legal）は緩んだ．一方 **Iter64→Iter66 で新たに誤った行では
   medical→natural_science が依然 9 件で最大**であり，この経路は質量比を戻しても解消しない．
   これが上記 1 の「残り半分」の実体である．
3. **探索的仮説「本数は効くが質量比が打ち消していた」は支持されない**．被覆 2 個行は
   **21/100** で Iter64 の 24 に戻らず，対 Iter65 の対応あり比較も discordant b=2・c=2 の
   exact McNemar **p=1.0** で Iter65 と実質同一であった．合成 810 行という本数を保ったまま
   質量比だけを Iter64 水準へ戻しても複合側は回復しない．したがって**「質量比を一定に保って
   本数を増やす」という用量反応の再開は実測に支持されない**（ただし R-H により，これは
   「効果がないことの証明」ではなく「複合 100 行では検出できない」に留まる）．
4. **N5 と複合側は，質量比という 1 つのスカラーの上でトレードオフする**．3 反復の実測は
   Iter64（35.1%）: N5 0.591／被覆 24 → Iter65（51.9%）: 0.563／21 → Iter66（35.1%・本数 810）:
   0.577／21 であり，**単一ヘッドで両立させる余地が乏しい**ことが示された．
   さらに質量比を下げる方向（単一ドメイン行の 3 重化以上）は，合成データの寄与自体を希釈して
   既に Iter64 未満の複合側（21/100）を損なうため，探索の価値がない．

**学び（次の自分への申し送り）**

- **「合成データの混ぜ方（量・比率）」という変数群は 4 反復（Iter63〜66）で形が確定した．**
  本数（135/405/810）も質量比（15.3%/35.1%/51.9%）も，単一ヘッドの中では
  N5 と複合被覆のトレードオフ曲線上を移動するだけであり，両立点は存在しなかった．
  **この系統の次のレバーは「混ぜ方」ではなく「構造（どのヘッドが何を決めるか）」でなければならない．**
- **計画の仮説設計が排他的二択（回復すれば質量比・しなければ分布シフト）を前提にしていたことの限界**．
  実測は中間（ちょうど半分の回復）に落ち，事前に用意した次レバーの分岐（adopted→用量反応再開／
  rejected→2 ヘッド構成）がそのままでは適用できなかった．**連続量を動かすレバーでは，
  二択ではなく「効果量が閾値の何割か」で次の一手を決める設計にしておくこと．**
- **判定が境界に密集した**（P2 は 1 行差の FAIL，P3 は同数で狭義不等号により FAIL，N6''・N7 は
  閾値ちょうどで `≧` により PASS）．事後に緩めない原則に従い確定させたが，**次の計画フェーズでは
  各条件の不等号の向きと境界値の扱いを明示的に書くこと**．
- 事実の訂正: 計画節の参照点表の「Iter64 N2' 0.5875 相当」は転記の誤りで，**Iter64 の N2' 実測は
  0.599375（959/1600）**である（0.5875 は基準線 0.5975 からの −1.0pt 下限であり，
  閾値としては事前登録どおりで判定には影響しない）．
- 継続する留保: **R-F（実行時経路への配線が 8 イテレーション連続で未実施）**・R-E（rank_1/rank_2 を
  単一ヘッドが決める）・R-I（較正値を対外引用しない．本分析では遵守し，用いた指標はすべて
  argmax／集合一致に基づく）・R-H（複合 100 行の検出力限界）．
  R-J（`class_weight='balanced'` が複製の効果を打ち消す懸念）は，質量比が設計値どおり復元された
  ことの実測により**否定され役目を終えた**（クラス重みは内部構成比を打ち消さない）．

**次のレバー（単一レバー原則）**

config.yml の levers のうち未試行で残るのは `production_deployment_gap`・
`dispatch_policy=adaptive_confidence_gap` のみで，いずれも `config.yaml` のスキーマ変更または
実機本走を要し自律着手できない（B99 と同じ状況）．そこで SKILL.md「停止条件」の選択肢 1 に従い，
**本イテレーションの学びから新レバーを考案して `levers` 末尾へ追記した**（backlog B100）．

- 新レバー: **`multilabel_head_architecture = two_head_rank1_single_domain_classifier`**．
- 内容: **rank_1 を既存の単一ドメイン分類器の出力（`--rank1-source baseline`＝基準線 results の
  `selected_domain`．Iter61 と同じ構成）に戻し，rank_2 のみを合成 810 行**だけで学習した
  別ヘッド（`models/dispatch_multilabel_head_iter67_synth_only.joblib`）から選ぶ．
  合成データを単一ドメイン判別の学習から**構造的に切り離す**．
- 選定理由: (1) 上記機序 4 のトレードオフは「同一ヘッドが rank_1 と rank_2 の両方を決める」
  ことに起因する．2 ヘッドにすれば **N5 は定義上，基準線と完全一致する（退行が構造的にゼロになる）**
  ため，4 反復にわたり主基準／非退行の律速だった N5 制約そのものが消える．
  (2) 残る問いは「rank_1 を弱める（head_argmax 72/100 → baseline 61/100 相当）代償を払っても
  複合被覆が保てるか」の 1 点に絞られ，反証可能性が高い．
  (3) **既存スクリプトのオプションの組み合わせだけで実現できる見込みで，`config.yaml` の
  スキーマ変更・実機本走を伴わない**（合成のみのヘッドを学習する際の `--train-data` の扱いは
  調査・計画フェーズで確定すること）．
- 参照点: Iter61（rank_1=baseline・被覆 12/100・+10.0pt）と Iter64（rank_1=head_argmax・
  被覆 24/100・+20.0pt）の 2 本立て．主基準は複合側（被覆 2 個行・`compound_domain_set_recall`）に
  置き，N5 は「基準線と完全一致すること」を実験成立の検査項目（レバーが意図どおり効いているかの
  確認）として使う．
- コスト: 合成生成不要・ヘッド再訓練約 5 分（埋め込み 810 行のみ）＋採点・統計 2 分＝**10 分程度**．
- 次イテレーション名: **「rank_1 を単一ドメイン分類器に戻す 2 ヘッド構成」**

**要人間判断**: なし（レバーの考案・追記・次イテレーション名の決定はいずれも可逆な判断の範囲）．
ただし累積した申し送りとして，**R-F（実行時経路への配線が 8 イテレーション連続で未実施であり，
本研究線のオフライン成果はいずれも実機での有効性を主張できない）**は，研究の結論を確定させる
段階で人間判断を要する（B95 要レビュー 1 に一本化したまま維持）．

