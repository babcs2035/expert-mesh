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

