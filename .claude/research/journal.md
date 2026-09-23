## Iteration 74: conformal予測集合へRAPSのサイズ正則化を導入し被覆を保ったまま集合サイズを2近傍へ縮める

### 調査 (Iter74)

**問い**

- Q1: RAPS（Angelopoulos et al.）のサイズ正則化は，非適合スコアのどこに，どの形で入るのか（一次資料と参照実装の両方で確定する）．
- Q2: 本実装は Romano のスコアの**補数**（`S = 1 - cumsum + u·π`）で書かれている．RAPS のペナルティ項をこの補数規約へ写すとどうなるか．打ち切り探索（先頭から走査して最初の不成立で break）は正しいままか．
- Q3: k_reg=2 固定で λ を掃引したとき，coverage と mean_set_size はどこへ着地するか（B109 制約 (2)．本実行前に着地点を数値で言語化する手順は Iter71・72・73 で 3 反復連続して実測と一致している）．

**Q1: RAPS の定義（出典付き）**

- Angelopoulos, Bates, Malik & Jordan, "Uncertainty Sets for Image Classifiers using Conformal Prediction", ICLR 2021 Spotlight, arXiv:2009.14193（<https://arxiv.org/abs/2009.14193>）．APS（Romano et al. 2020）が裾の重い巨大な集合を出す問題に対し，非適合スコアへサイズ正則化項を加えて集合を明示的に小さくする手法である．
- 参照実装 `aangelopoulos/conformal_classification` の `conformal.py` で，ペナルティの入り方を行レベルで確認した（<https://raw.githubusercontent.com/aangelopoulos/conformal_classification/master/conformal.py>）．
  - `self.penalties = np.zeros((1, num_classes)); self.penalties[:, kreg:] += lamda`（L30-31 / L107-109）— 0-index で `kreg` 以降のクラス，すなわち **1-index の順位 o > k_reg のクラス 1 個ごとに λ** を割り当てる．
  - 集合構成 `gcq()`（L176-178）は `penalties_cumsum = np.cumsum(penalties)` を使い `(cumsum + penalties_cumsum) <= tau` で判定する．校正スコア `get_tau()`（L221）も `U*ordered[idx] + cumsum[idx-1] + penalty[0:idx+1].sum()` である．
  - つまりペナルティは**累積**であり，順位 o のクラスのスコアは `E(x,y,u) = ρ + u·π_(o) + λ·(o - k_reg)^+`（ρ は上位クラスの確率質量，π_(o) は自分の確率）．校正・評価の両側に同じ形で入る．
- 補足: `pick_kreg()`（L236）は校正データの真クラス順位の (1-α) 分位点で k_reg を選ぶが，本イテレーションでは B109 制約 (1) に従い**ルーティングの要求（top-2 dispatch）から k_reg=2 に先に固定**し，λ の 1 次元だけを扱う．

**Q2: 本実装（補数規約）への写像**

本リポジトリの `_compute_prediction_set()` は Romano スコアの補数で書かれている．Iter73 で入れたランダム化 APS のスコアは `S = 1 - cumsum_incl + u·π = 1 - (ρ + (1-u)·π)` であり，`v := 1-u ~ U(0,1)` と置けば `S = 1 - E_Romano(v)` で一致する（校正・評価で同じ u を使う限り分布も一致する）．ここへ RAPS のペナルティを写すと

`S_raps = (1 - cumsum_incl + u·π_(o)) - λ·max(0, o - k_reg)`，包含条件は `S_raps >= q_hat`（q_hat は校正半の真クラス `S_raps` の α 下側分位点）

となる．ペナルティは順位 o について単調非減少なので `S_raps` は順位について単調減少であり，**現行の「先頭から走査し最初に条件を満たさなくなった時点で break」という探索はそのまま正しい**（Iter70 で `corrected_aps` について確認した単調性の議論がそのまま通る）．λ=0 のとき `randomized_aps` と厳密に一致するので，既存分岐の回帰テストがそのまま新分岐の縮退テストになる．

**Q3: 事前シミュレーション（本実行前に実施．B109 制約 (2)）**

本レバーも `probabilities` を一切変えないため，Iter73 の出力 `results/20260923_135653/Iter73_randomized_aps.jsonl`（1,600 行，`split` 付き）から coverage・mean_set_size を厳密に再現計算できる．u は `np.random.default_rng(42).random(1600)` を dataset 順に割り当て，q_hat は校正半 800 行の真クラス `S_raps` の α 下側分位点（有限標本補正 `floor((n+1)α)/n`）とした．

まず λ=0 で現行実装の再現を確認した:

| 規則 | q_hat | coverage（eval 半 n=800） | mean_set_size | サイズ分布 |
|---|---|---|---|---|
| シミュレータ λ=0, k_reg=2 | 0.093879 | 0.90375 | 4.00125 | [52, 92, 154, 182, 178, 111, 29, 2, 0, 0] |
| Iter73 実測 | 0.0939 | 0.90375 | 4.00125 | {1:52, 2:92, 3:154, 4:182, 5:178, 6:111, 7:29, 8:2} |

小数点以下まで一致するので，シミュレータは本実装と同じものを計算している．その上で k_reg=2 のまま λ を掃引した:

| λ | q_hat | coverage | mean_set_size | P(size<=2) | 複合46行の2ドメイン同時被覆 | サイズ分布（1〜7） |
|---|---|---|---|---|---|---|
| 0（＝Iter73） | 0.09388 | 0.90375 | **4.0012** | 0.1800 | 0.2826 | [52, 92, 154, 182, 178, 111, 29] |
| 0.001 | 0.09106 | 0.90500 | 4.0025 | 0.1750 | 0.3043 | [49, 91, 155, 188, 179, 108, 29] |
| 0.005 | 0.07647 | 0.91125 | 4.0825 | 0.1487 | 0.3261 | [36, 83, 145, 208, 195, 109, 24] |
| 0.01 | 0.06137 | 0.91000 | 4.1175 | 0.1200 | 0.3261 | [25, 71, 152, 224, 209, 103, 16] |
| **0.02（本実験に事前登録）** | **0.03233** | **0.90750** | **4.1638** | **0.0775** | **0.2826** | **[5, 57, 154, 259, 241, 79, 5]** |
| 0.05 | -0.04264 | 0.90125 | 4.1575 | 0.0000 | 0.3043 | [0, 0, 154, 391, 230, 25, 0] |
| 0.1 | -0.18764 | 0.90625 | 4.2838 | 0.0000 | 0.3478 | [0, 0, 16, 542, 241, 1, 0] |
| 0.2〜0.5 | — | 0.90625 | 4.3025 | 0.0000 | 0.3478 | [0, 0, 0, 558, 242, 0, 0]（飽和） |

**この掃引の結論は明確である: k_reg=2・任意の λ で mean_set_size は Iter73 の 4.0012 を下回らない．** λ を上げると 4.16〜4.30 へ**増加**して飽和し，P(size<=2) は単調に 0 へ落ちる．k_reg を 1・3・4 に変えても同じで（k_reg=1: 4.015〜4.30，k_reg=3: 4.013〜4.30，k_reg=4: 4.006〜4.30），いずれも λ→0 が最小である．

**機序（掃引から読み取れること）**: RAPS のペナルティは校正スコアと評価スコアの**両方**に同じ形で入るため，λ を上げると q_hat がほぼ同量だけ下がって相殺する．残るのは「サイズ分布の圧縮」であって「平均の縮小」ではない．λ=0.2 以上では全 800 行が size 4 か 5 の 2 値に潰れる．RAPS が原論文で平均サイズを下げたのは ImageNet（1,000 クラス）で APS が裾の重い巨大集合を出していたからであり，**10 クラス・最大サイズ 8 の本問題には切るべき裾が無い**．

**追加調査: そもそも被覆 0.90 でサイズ 2 は到達可能か（サイズ最適な構成での床の測定）**

RAPS が効かない理由が「本問題のサイズは構成法ではなく分類器の鋭さで決まっているから」であるなら，どんな構成法でも 2 には届かないはずである．これを確かめるため，平均集合サイズが**証明付きで最小**である LAC / THR（Sadinle, Lei & Wasserman, "Least Ambiguous Set-Valued Classifiers with Bounded Error Levels", JASA 114(525):223-234, 2019, arXiv:1609.00451．与えられた被覆の下で期待集合サイズを最小化する集合値分類器が条件付きクラス確率の閾値化であることを示した論文）を同じ校正/評価半でシミュレートした:

| 名目水準 | q_hat | 確率閾値 | coverage | mean_set_size | P(size<=2) | 複合の同時被覆 |
|---|---|---|---|---|---|---|
| 0.80 | 0.88173 | 0.11827 | 0.79625 | **2.2437** | 0.6312 | 0.0652 |
| 0.85 | 0.92058 | 0.07942 | 0.84125 | 2.8687 | 0.4075 | 0.1304 |
| **0.90** | 0.95073 | 0.04927 | 0.89000 | **3.8250** | 0.2087 | 0.2609 |
| 0.95 | 0.97513 | 0.02487 | 0.94375 | 5.2625 | 0.0825 | 0.5000 |

- 名目 0.90 でのサイズ最適構成が 3.825（実測被覆 0.89000）である．Iter73 のランダム化 APS の 4.0012（被覆 0.90375）は，**被覆をそろえれば最適値との差が 0.1〜0.2 程度しかない**．
- したがって「集合サイズを 2 近傍へ」は，**構成法の選択では達成できず，被覆水準を 0.80 前後まで落とすことでしか達成できない**（LAC でも size 2.24 に要る被覆は 0.796）．これは本分類器の確率分布の鋭さ（top1 精度 0.60 前後）が決めている量である．
- この観測は Iter73 分析 5（集合縮小と複合設問の同時被覆のトレードオフ）とも整合する．LAC でも被覆 0.80 まで落とすと複合の同時被覆は 0.065 まで崩れる．

**この調査で分かったことの要約**

1. RAPS の定義と本実装の補数規約への写し方は確定した（Q1・Q2）．実装可能で，探索の単調性も保たれる．
2. しかし k_reg=2 の下で λ をどう選んでも mean_set_size は Iter73 を下回らない．これは B109 が想定した「残差へサイズ正則化を当てる」という見込みに対する**反証**である（Q3）．
3. さらにサイズ最適な LAC でも名目 0.90 では 3.83 が床であり，size≈2 は被覆 0.80 前後を受け入れない限り到達しない．**「予測集合を top-2 dispatch へ流用する」という構想の障害は conformal の構成法ではなく分類器の確率分布そのものにある**．

### 計画 (Iter74)

**仮説（反証形で事前登録する）**

B109 が事前登録した見込み「RAPS のサイズ正則化を当てれば mean_set_size が 4.00 から 2 近傍へ縮む」は**成り立たない**．RAPS のペナルティは校正スコアと評価スコアへ同形に入るため q_hat がほぼ同量だけ移動して相殺し，効果はサイズ分布の圧縮（size 4〜5 への集中）に留まり，平均はむしろ +0.16 増える．k_reg=2・λ=0.02 の実測は事前シミュレーションと一致し（coverage=0.90750，mean_set_size=4.16375），主基準 2（サイズの有意な減少）は不成立となる．

この仮説を採る理由は，シミュレータが Iter73 実測を小数点以下まで再現しており（Q3 の再現表），かつ 4 反復連続で事前予測が実測と一致してきたためである．**本実行は，この予測を実装で確認して `raps_penalty` を根拠をもって閉じるための反証実験である**．合格を探して λ を振り直すことはしない（掃引は既に全域で終えており，どの λ も主基準を満たさない）．

**単一レバー**

`conformal_set_size_reduction`: 予測集合の構成を `randomized_aps`（Iter73 で adopted）→ **`raps_penalty`（k_reg=2 固定，λ=0.02）** へ変更する．動かすのはこの 1 点のみ．**k_reg はルーティング要求から 2 に先に固定し，振るのは λ の 1 次元だけ．さらにその λ も事前シミュレーションで掃引済みのため，本実行は 0.02 の 1 点に事前登録する**（B109 制約 (1)(2)）．

**λ=0.02 を選んだ理由**: 掃引上どの λ も主基準を満たさないので「最も有利な λ」は存在しない．λ→0 は `randomized_aps` への縮退でレバーとして無意味なため，**正則化が分布を実際に動かしていることが目視できる最小の水準**（size 1 が 52→5 行へ潰れ，size 4〜5 が 360→500 行へ集中する）を選んだ．発火証拠としての q_hat も 0.0939→0.0323 と明確に変わる．

**固定する構成（Iter73 の最良構成に固定）**

`--calibration-source eval_holdout` / `--holdout-seed 42` / `--randomization-seed 42` / `--qhat-quantile-direction alpha_lower` / `--qhat-source true_class` / `--confidence-level 0.90`．評価データ `data/dataset.jsonl`（1,600 行），分類器 `models/domain_classifier.joblib`，埋め込み `nomic-embed-text:latest`（wafl-ctrl5 の `127.0.0.1:11435`），`--education-logit-bias 0.0` / `--education-threshold 0.0`，`--fine-tuned-embed-model` は指定しない．`config.yaml`・`http_server.py`・`classifier.py`・`aggregator.py`・`mise.toml` は変更しない．

**変更箇所（`scripts/evaluate_classifier_calibration.py` 1 ファイル＋テスト）**

1. `_compute_prediction_set()`:
   - 引数へ `raps_lambda: float | None = None`・`raps_k_reg: int | None = None` を追加．
   - `set_construction` の許容値へ `"raps_penalty"` を追加（`ValueError` メッセージも更新）．`raps_penalty` のとき `randomization_u is None` / `raps_lambda is None` / `raps_k_reg is None` のいずれかなら `ValueError`（無言で非正則化へ落ちない．Iter69 の教訓）．
   - 新分岐: 降順走査で 1-index の順位 `rank` を持ち，`score = 1.0 - cumsum + randomization_u * probabilities[idx] - raps_lambda * max(0, rank - raps_k_reg)`．`score >= q_hat` の間だけ append し，最初の不成立で break（`randomized_aps` と同じ打ち切り規則）．既存 3 分岐（`broken` / `corrected_aps` / `randomized_aps`）は 1 行も書き換えない．
   - 空集合 fallback（top クラスを入れる）は共通のまま残す．
2. `predict_calibrated_rows()`:
   - 引数へ `raps_lambda: float = 0.0`・`raps_k_reg: int = 2` を追加．
   - `raps_penalty` は `randomized_aps` と同じ前提（`calibration_source == "eval_holdout"` 以外は `ValueError`．u を両側で揃えられないため）．
   - `eval_holdout` 分岐の校正スコア計算を `raps_penalty` のとき `1.0 - cumsum + u_all[i] * probs[idx] - raps_lambda * max(0, rank - raps_k_reg)` に変える（`rank` は真クラスの 1-index 順位．**現行ループは `idx == eval_labels[i]` で break しているので，その位置のループ回数がそのまま rank になる**）．
   - 評価ループ **2 箇所**（fine-tuned 分岐・ollama 分岐）の `_compute_prediction_set()` 呼び出しへ `raps_lambda` / `raps_k_reg` を渡す．**この 2 箇所はどちらも通す必要がある**．
   - stderr 診断へ `raps_lambda` / `raps_k_reg` を追記し，q_hat の診断計算も同じペナルティ付きスコアで行う（**レバー発火の証拠**．`raps_penalty` なら `q_hat≈0.0323`，`randomized_aps` なら `0.0939` が出るはず）．
3. `main()`:
   - `--set-construction` の `choices` へ `raps_penalty` を追加．`--raps-lambda`（`type=float`, `default=0.0`）・`--raps-k-reg`（`type=int`, `default=2`）を新設．
   - **CLI の `--output` 有無 2 分岐の両方へ伝播する．Iter69〜73 で 5 回連続して警告されている箇所であり，実装完了時にチェックリストとして目視確認すること**:
     - [ ] stdout 側 `if args.output is None:` の `_run(...)`
     - [ ] ファイル出力側 `with open(args.output, "w", ...)` の `_run(...)` ※本実験が通るのはこちら
   - `_run()` のシグネチャと `predict_calibrated_rows()` 呼び出しへも追加．
4. 新分岐を足す前に，既存分岐でのみ初期化される局所変数（`u_all`・`precomputed_eval_holdout`・`holdout_split`・`cp_data`・`n_cal`）を洗い出す（Iter72 の `UnboundLocalError` の教訓）．
5. `tests/test_evaluate_classifier_calibration.py` へ追加:
   (a) `raps_penalty` かつ `raps_lambda=0.0` のとき `randomized_aps` と同一の集合を返すこと（縮退の同値性），
   (b) λ を上げると集合内の低順位クラスが減るか等しいこと（単調性），
   (c) `raps_penalty` かつ `raps_lambda is None` / `raps_k_reg is None` / `randomization_u is None` でそれぞれ `ValueError`，
   (d) `raps_penalty` かつ `calibration_source="oof_train"` で `ValueError`，
   (e) 既定値（`corrected_aps`）の回帰テストが通ること．

**到達コードパス**

CLI `--conformal-prediction --calibration-source eval_holdout --holdout-seed 42 --set-construction raps_penalty --raps-lambda 0.02 --raps-k-reg 2 --randomization-seed 42 --qhat-quantile-direction alpha_lower --qhat-source true_class --confidence-level 0.90 --output ...`
→ `main()` のファイル出力分岐 → `_run()` → `predict_calibrated_rows()` → `eval_holdout` 分岐の校正スコア計算（**ペナルティが入る第 1 の地点＝q_hat が 0.0939→0.0323 へ変わる**）→ 評価 1,600 行ループ（ollama 分岐）→ `_compute_prediction_set(..., raps_lambda=0.02, raps_k_reg=2)`（**第 2 の地点＝集合の中身が変わる**）→ 出力 jsonl の `prediction_set` / `set_size` / `split` → eval 半 800 行で集計．`config.yaml` を経由しないため「デプロイ漏れ」型の失敗は構造上起こらない．唯一のリスクは CLI 2 分岐の伝播漏れと評価ループ 2 箇所のうち片方だけへの伝播漏れであり，予備実行（先頭 20〜40 行）の stderr で `set_construction=raps_penalty raps_lambda=0.02 raps_k_reg=2 q_hat=0.0323` を目視確認して潰す．

**成功条件（事前登録）**

名目 0.90，評価半 n=800．比較対象は `results/20260923_135653/Iter73_randomized_aps.jsonl`（coverage=0.90375，mean_set_size=4.00125）．

| 指標 | 定義 | Iter73 実測 | 事前予測（λ=0.02, k_reg=2） | 合格条件 |
|---|---|---|---|---|
| coverage（主基準 1） | eval 半で `mean(expected_domains[0] in prediction_set)` | 0.90375 | 0.90750 | **0.88 <= coverage <= 0.95** |
| mean_set_size（主基準 2） | eval 半の `mean(set_size)` | 4.00125 | 4.16375 | **4.00125 から有意に減少**（対応あり 800 行の差分平均が 0 より有意に小，両側 t 検定 p<0.01，かつ減少幅 >= 0.5） |
| P(set_size<=2) | eval 半で `mean(set_size<=2)` | 0.1800 | 0.0775 | 報告のみ |
| 複合46行の2ドメイン同時被覆 | 複合設問のみ．両ドメインが集合に入る割合 | 0.2826 | 0.2826 | 報告のみ（B109 制約 (3)） |

- **adopted**: 主基準 1・2 の両方を満たし，非退行条件を全て満たすこと．
- **rejected**: どちらかが不成立．**事前予測は「主基準 2 が不成立（実際には mean_set_size が +0.16 増える）」である**．予測どおり rejected になった場合の解釈も事前に登録する:
  - `mean_set_size` が予測 4.16375 の ±0.02 以内であれば，**実装は正しく発火したうえで手法が効かなかった**と判定する（実装不成立ではない）．根拠は掃引全域（k_reg∈{1,2,3,4}×λ∈[0.001,0.5]）で最小値が λ→0 だったこと，および LAC（サイズ最適）でも名目 0.90 で 3.83 が床だったこと．
  - この場合 `conformal_set_size_reduction` は全 2 値試行済みでクローズとなり，**「集合サイズを構成法で 2 近傍へ縮める」路線は閉じる**．次の判断（B104 A2 の実行時配線に進むか，被覆水準を 0.80 前後へ下げる設計変更を人間に諮るか，conformal 系列を閉じるか）は考察フェーズで backlog へ起票する．
  - 実測が予測から ±0.02 を超えて外れた場合は，まず**実装不成立**（伝播漏れ・rank の 0/1-index 取り違え・校正側だけペナルティ未適用）を疑う．これが既定の解釈である．
- **非退行条件**（いずれも eval 半 800 行について）:
  1. `selected_domain` が `Iter73_randomized_aps.jsonl` の同 id 行と完全一致すること．
  2. `confidence`・`probabilities` が同 id 行と一致すること（許容差 1e-9）．
  3. `split` の割り当て（cal/eval）が Iter73 と完全一致すること（`--holdout-seed 42` 固定）．
  4. `set_size` が 1 以上 10 以下で，`prediction_set` に重複がないこと．
  5. 後方互換: `--set-construction randomized_aps --randomization-seed 42`（他は同条件）での再実行が Iter73 出力と **md5 一致**すること．
  6. 予備実行の stderr に `set_construction=raps_penalty`・`raps_lambda=0.02`・`raps_k_reg=2`・`q_hat≈0.0323` が出ること（発火証拠）．
- **ノイズ幅**: n=800・p≈0.90 で二項 SE≈0.0106．mean_set_size は対応あり比較で評価する（予測される差分は +0.1625，対応あり SE=0.0216，t=7.51，減少 86 行・同数 507 行・増加 207 行）．coverage の予測値 0.90750 は帯下限 0.88 から +2.59 SE，帯上限 0.95 まで -4.01 SE の位置にある．

**付随報告（判定に用いない）**

(i) 実測 q_hat（予測 0.032331）と校正半のペナルティ付きスコア分布の分位点，(ii) 集合サイズのヒストグラム（予測 `[5, 57, 154, 259, 241, 79, 5, 0, 0, 0]`），(iii) 空集合 fallback の発火件数（予測 0 件），(iv) **複合設問 46 行の 2 ドメイン同時被覆（予測 0.2826）・どちらか 1 つ被覆（予測 0.8913）・複合行の mean_set_size**（B109 制約 (3)．Iter73 で 0.478→0.283 へ落ちた量の追跡），(v) λ∈{0, 0.005, 0.01, 0.02, 0.05, 0.1} の掃引を本実装で再現し調査 Q3 の表と一致するかの確認，(vi) LAC（`1 - p_true` 閾値，名目 0.90）の coverage / mean_set_size をオフラインで再測定し「サイズ最適構成の床＝3.83」を本実装系で裏付ける．

**実機フルスペック本走（2026-09-23 ユーザー指示．レバーではない）**

ユーザーから「wafl500〜509 を用いたフルスペックの本実験を積極的に実行する方針を貫け」との指示があったため，本イテレーションでは上記のオフライン実験と並行して**実機 10 ノードでの 1,600 問本走を実施する**．**これは単一レバーではなく，構成を一切変えない基準線の再取得・検証である**（`config.yaml`・分類器・モデルはすべて現行 HEAD のまま）．

- 手順: `mise run deploy` → `mise run start`（`data/dataset.jsonl` 1,600 問）→ `mise run analyze`．所要 90〜150 分（config の `timeout_min: 150`）．
- 実施理由 1（基準線の鮮度）: 直近の実機本走は `results/20260919_005727/`（top1=0.596875，kappa=0.565958，misrouting=0.403125，fallback=0.0，dispatch_failure=0.000625，mean_duration_ms=1502.16，single_domain_top1=0.609333，compound_top1=0.41）であり，Iter56 以降の 18 反復はすべてオフラインだった．現行 HEAD の end-to-end 指標を再確認する．
- 実施理由 2（conformal 系列との接続確認）: `results.jsonl` の `probe_candidates` は 10 ノード分の `confidence` を保持しており，**実行時ルーティングが実際に使っている確率ベクトルそのもの**である．これを `models/domain_classifier.joblib` の `predict_proba`（オフライン conformal 系列の入力）と 1,600 行全件で突き合わせ，許容差 1e-9 で一致するかを確認する．Iter56〜74 の conformal の議論が実機の分布に対して有効かどうかは，この 18 反復で一度も検証されていない（1 行のスポット確認では `business_economics-001` で 0.4884300235443738 vs 0.4884300235443737 と一致している）．この repo が 6 回繰り返した「設定は変えたのにコードへ到達しない」型の失敗と同じ系統の未検証事項である．
- 非退行の目安（判定には用いない．構成を変えていないので一致するはず）: top1_accuracy が 0.596875 から二項 SE=0.0123 の 2 倍（±2.5pt）以内，fallback_rate=0.0，dispatch_failure_rate <= 0.001．これを外れた場合は実機側の状態異常（モデル未 pull・ノード欠落等）を疑い，オフライン実験の判定とは切り離して報告する．

**期待効果**

`raps_penalty` を実装のうえで 1 点実行し，`conformal_set_size_reduction` レバーを 2 値とも試し切って閉じる．事前シミュレーションが示すとおり主基準は不成立となる公算が高いが，その場合でも得られるのは「サイズ縮小は構成法では達成できず，分類器の確率分布の鋭さと被覆水準が決めている」という，LAC の最適性（Sadinle et al. 2019）に裏打ちされた**否定的だが確定的な知見**である．これは conformal を本線のルーティングへ接続するか否かの判断材料そのものになる．併せて実機 1,600 問本走で end-to-end 基準線を 19 反復ぶりに再取得し，オフライン conformal 系列の入力が実行時経路と同一であることを初めて検証する．

**コスト**: オフライン実験は 1 実行 10〜30 分（埋め込み 1,600 行のみ，wafl-ctrl5 の ollama `127.0.0.1:11435`）＋後方互換アンカー 1 実行．実機本走は 90〜150 分（wafl500〜509 を占有）．分類器の再訓練は無し．

### Iteration 74 実行済み

**変更（2 ファイルのみ．`config.yaml`・`http_server.py`・`classifier.py`・`aggregator.py`・`mise.toml` は不変）**

1. `scripts/evaluate_classifier_calibration.py`: `_compute_prediction_set()` へ `raps_lambda: float | None = None`・`raps_k_reg: int | None = None` を追加し `set_construction="raps_penalty"` 分岐を新設（降順走査で 1-index の `rank` を持ち `score = 1 - cumsum + u·p_(rank) - λ·max(0, rank-k_reg)`，`score >= q_hat` の間だけ append し最初の不成立で break）．いずれかの引数が `None` なら `ValueError`（無言で `randomized_aps` へ落ちない）．`predict_calibrated_rows()` の `eval_holdout` 分岐の校正スコア（真クラスの 1-index 順位を `rank` として同じペナルティを適用）と評価ループ 2 箇所（fine-tuned/ollama）の両方へ `raps_lambda`/`raps_k_reg` を伝播．stderr 診断へ `raps_lambda`/`raps_k_reg` を追記．`main()` に `--raps-lambda`（既定 0.0）・`--raps-k-reg`（既定 2）を新設し，**`--output` 有無の 2 分岐両方へ伝播**（チェックリストで目視確認．漏れなし）．
2. `tests/test_evaluate_classifier_calibration.py`: 計画 5 の (a)〜(e) 相当 6 件（λ=0 での `randomized_aps` への縮退，λ増加に対するサイズの単調非増加，`raps_lambda`/`raps_k_reg`/`randomization_u` 欠落時の `ValueError`，`oof_train` との非互換 `ValueError`，`broken` 既定値の回帰）を追加．既存 24 件と合わせ **30 件 PASS**，`ruff check` PASS．

**実験（`results/20260923_142619/`，オフライン 3 実行＋実機 1 本走）**

- A（後方互換アンカー）: `--set-construction randomized_aps --randomization-seed 42`（他は Iter73 と同一）→ `Iter74_randomized_aps_backcompat.jsonl`．**`results/20260923_135653/Iter73_randomized_aps.jsonl` と md5 完全一致**（`afde650eea9ac611a43f0bb20703c2e3`）．
- B（本実行）: `--set-construction raps_penalty --raps-lambda 0.02 --raps-k-reg 2 --randomization-seed 42 --calibration-source eval_holdout --holdout-seed 42 --qhat-quantile-direction alpha_lower --qhat-source true_class --confidence-level 0.90` → `Iter74_raps_penalty.jsonl`．stderr は `set_construction=raps_penalty randomization_seed=42 q_hat=0.0323 n_cal=800 raps_lambda=0.02 raps_k_reg=2`（**レバー発火の証拠**．事前予測 q_hat≈0.032331 と一致）．
- C（実機フルスペック本走，レバーではなく基準線再取得＋接続確認）: `mise run deploy` → `mise run start`（`data/dataset.jsonl` 1,600 問，wafl500〜509）→ `mise run analyze 20260923_150540`．`results/20260923_150540/`．**`mise run analyze`（引数なし）はディレクトリ名をアルファベット順 `sort` で選ぶため `results/iter45_preliminary/`（`i` > `2`）を誤って選択する落とし穴があり，`analyze 20260923_150540` と明示して回避した**（新規の未報告事項として付随報告に記載）．

| 指標 | 定義 | Iter73 実測 | 事前予測（λ=0.02, k_reg=2） | Iter74 実測（eval 半 n=800） | 合格条件 | 判定 |
|---|---|---|---|---|---|---|
| coverage（主基準 1） | eval 半で `mean(expected_domains[0] in prediction_set)` | 0.90375 | 0.90750 | **0.90750** | 0.88 ≤ coverage ≤ 0.95 | **PASS** |
| mean_set_size（主基準 2） | eval 半の `mean(set_size)` | 4.00125 | 4.16375 | **4.16375** | 4.00125 から有意に減少・減少幅 ≥ 0.5 | **FAIL**（+0.1625 の有意な**増加**，`ttest_rel` t=-7.515, p=1.53e-13） |
| q_hat | — | 0.0939 | 0.032331 | **0.0323** | — | — |
| P(set_size≤2) | eval 半 | 0.1800 | 0.0775 | **0.0775** | 報告のみ | — |
| 複合 46 行の 2 ドメイン同時被覆 | 複合設問のみ | 0.2826 | 0.2826 | **0.2826** | 報告のみ | — |

- 対応あり 800 行の set_size 差分（Iter73 − Iter74）: **減少（73>74）86 行・同数 507 行・増加（73<74）207 行**．事前予測と完全一致．
- 集合サイズ分布（eval 半）: `{1:5, 2:57, 3:154, 4:259, 5:241, 6:79, 7:5}`．事前予測 `[5, 57, 154, 259, 241, 79, 5, 0, 0, 0]` と完全一致．
- 非退行 6 項目すべて充足: `selected_domain` 完全一致（0 件不一致，1600 行中） / `probabilities` 最大差 0.0（浮動小数点誤差ですら発生せず） / `split` 割当一致（1600 行中 0 件不一致） / `set_size` 1〜7 かつ重複なし / A の再実行が Iter73 出力と **md5 完全一致** / stderr 発火証拠あり．
- 付随（複合設問，B109 制約 (3)）: 2 ドメイン同時被覆 0.2826（46 行中 13 行），どちらか 1 つ被覆 0.8913，複合行の mean_set_size 3.9565．いずれも事前予測と一致．

**実機フルスペック本走（C）の結果**

| 指標 | `results/20260919_005727/`（直近基準線） | `results/20260923_150540/`（本反復） | 差 |
|---|---|---|---|
| top1_accuracy | 0.596875 | **0.595625** | -0.00125（二項 SE=0.0123 の 0.1 SE） |
| cohens_kappa | 0.565958 | **0.564477** | -0.00148 |
| misrouting_rate | 0.403125 | **0.404375** | +0.00125 |
| fallback_rate | 0.0 | **0.0** | 0 |
| dispatch_failure_rate | 0.000625 | **0.000625** | 0 |
| mean_duration_ms | 1502.16 | **1499.76** | -2.40 |
| single_domain_top1 | 0.609333 | **0.608** | -0.00133 |
| compound_top1 | 0.41 | **0.41** | 0 |
| answer_quality_accuracy | — | 0.562667 | — |
| end_to_end_accuracy | — | 0.330625 | — |
| ece | — | 0.055085（n=1599） | — |

構成を一切変えていない基準線の再取得として，非退行の目安（±2.5pt 以内・fallback=0.0・dispatch_failure≤0.001）をすべて満たした．19 反復ぶりの実機本走で end-to-end 指標が Iter56 直前の水準から動いていないことを確認した．

**実施理由 2（`probe_candidates` と `predict_proba` の突き合わせ）の結果**: `results/20260923_150540/results.jsonl` の `probe_candidates`（10 ノード分の `confidence`）と，同一評価データに対する `models/domain_classifier.joblib.predict_proba()` のオフライン再計算（`--education-threshold 0.0` 指定，本実行 B と同条件）を 1,600 行×10 ドメイン全件で突き合わせたところ，**`education` ドメインのみ全 1,600 行で厳密に +0.05 の系統的な差**が見つかり，他 9 ドメインは最大絶対差 1e-15 台（浮動小数点誤差のみ）で一致した．原因を `classifier.py` を読んで特定した: **Iter52/53 で adopted・Iter57 前後で実行時経路 `classifier.py:estimate_confidence_classifier()` へ配線された `education_per_class_threshold=0.05`（`production_deployment_gap` レバー，config.yml 冒頭節参照）が実行時には効いているが，本イテレーションの conformal オフライン評価は計画どおり `--education-threshold 0.0` に固定しているため，この差分が生じる**．`--education-threshold 0.05` を指定して同じ突き合わせをやり直すと，**1,600 行×10 ドメイン全件で最大絶対差 0.0（ビット単位で完全一致）** になることを確認した（`results/20260923_142619/Iter74_probs_only_edu005.jsonl` で検証）．結論: **実行時ルーティングが使う確率ベクトルは，`education` の +0.05 補正を含めれば `models/domain_classifier.joblib` の `predict_proba` とビット単位で一致する**．Iter56〜74 の conformal 系列は，この実行時分布に対して有効な議論をしてきたことが初めて裏付けられた．一方で，conformal 系列は一貫して `--education-threshold 0.0` を使ってきており（本イテレーションもそれを踏襲），これは実行時経路とは異なる分布を評価してきたことを意味する．この差の影響（education 分類の被覆・集合サイズへの影響）は未評価であり，次回以降の検討事項として考察節に記録する．

### 分析(解釈) (Iter74)

**1. 実測は事前シミュレーションと完全一致し，実装は正しく発火した**

coverage=0.90750（予測 0.90750）・mean_set_size=4.16375（予測 4.16375）・q_hat=0.0323（予測 0.032331）・サイズ分布・対応あり差分の内訳（86/507/207）まで小数点以下含めて一致した．非退行 6 項目もすべて充足し，`probabilities` は浮動小数点誤差の範囲でさえ動いていない（最大差 0.0）．Iter16/20/21/22/27 系列の「設定は変えたのにコードへ到達しない」型の失敗ではないことは，stderr の `q_hat=0.0323`（λ=0 なら 0.0939 が出るはずの箇所）と A 実行の md5 一致で二重に確認できている．

**2. 主基準 2（サイズの有意な減少）は事前登録どおり不成立**

mean_set_size は 4.00125 → 4.16375 で **+0.1625 の有意な増加**（`ttest_rel` t=-7.515, p=1.53e-13）．増加方向であり「減少」という主基準の定義自体を満たさない．事前登録した解釈基準（実測が予測 4.16375 の ±0.02 以内なら実装は正しく発火したうえで手法が効かなかったと判定）に照らすと，**実測は予測と完全一致（差 0.0）** であり，「実装不成立」ではなく「手法が効かない」という判定になる．

**3. 機序は調査 Q3 の事前シミュレーションで特定済みの内容がそのまま再現された**

RAPS のペナルティは校正側の真クラススコアと評価側の各クラススコアの両方に同形で入るため，λ を上げると q_hat がほぼ同量だけ下がって相殺し，残るのはサイズ分布の圧縮（size 1 が 52→5 行，size 4〜5 に 360→500 行が集中）であって平均の縮小ではない．10 クラス・最大サイズ 8 の本問題には ImageNet（RAPS の原論文の実験対象，1,000 クラス）のような「切るべき裾」が無いという調査時点の考察が実測でも裏付けられた．

**4. 実機フルスペック本走は非退行を確認し，conformal 系列の前提を初めて検証した**

19 反復ぶりの実機 1,600 問本走は，直近基準線（`results/20260919_005727/`）と非退行の目安を満たす水準（top1 差 -0.00125，fallback/dispatch_failure 不変）で再現した．より重要な収穫は `probe_candidates` と `models/domain_classifier.joblib.predict_proba` の突き合わせで，**`education` の `+0.05` 補正を揃えればビット単位で一致する**ことが判明した点である．これにより，Iter56〜74 の conformal の議論（校正・被覆・集合構成）が実際の実行時分布に対して有効であったことが初めて裏付けられた一方，conformal 系列自体は一貫して `--education-threshold 0.0` を使ってきたため，10 ドメイン中 1 つ（education）については実行時分布とは異なる分布を評価してきたことも同時に判明した．この差の影響は本反復では評価していない．

### 考察 (Iter74)

**判定: `raps_penalty` は rejected（確定）**．事前登録の解釈基準どおり，実測が予測の ±0.02 以内で一致したため「実装は正しく発火したうえで手法が効かなかった」と判定する．実装は revert せず維持する（既定 `broken` が後方互換 md5 を保つ設計であり，`raps_penalty` 分岐はコードベースに残すが今後の固定値としては採用しない）．

**`conformal_set_size_reduction` レバーは 2 値（`randomized_aps` adopted, `raps_penalty` rejected）を試し切り，クローズする**．

**この反復で確定した知見**

1. RAPS のサイズ正則化は，校正・評価の両側に同形で入るペナルティ付き分位点法である限り，q_hat の自己補正によって「平均サイズの縮小」ではなく「サイズ分布の圧縮」にしか効かない．これは調査 Q3 の事前シミュレーション（k_reg∈{1,2,3,4}×λ∈[0.001,0.5] の全域探索）と LAC（サイズ最適構成）による床の確認（名目 0.90 で 3.83）の両方から導かれ，本実装でも寸分違わず再現した．
2. 「予測集合を top-2 dispatch へ流用する」という Iter72 以来の構想は，**構成法の選択では到達できない**．到達するには被覆水準を 0.80 前後まで下げるほかなく，それは複合設問の同時被覆（Iter73 分析 5）をさらに崩す．`conformal_set_size_reduction` レバーはこれで打ち切りが妥当である．
3. **実機本走で初めて，オフライン conformal 系列の入力が実行時ルーティングの確率分布と（education の +0.05 補正を除いて）ビット単位で一致することを確認できた**．これは Iter56〜74 の 19 反復にわたる conformal の議論の妥当性を裏付ける一次情報であり，同時に「conformal 系列は education について実行時と異なる分布を評価してきた」という新しい留保も明らかにした．

**次の一手（人間判断を要する事項を含む）**

- `conformal_set_size_reduction` はクローズ．次に残る conformal 関連の論点は config.yml 冒頭に記録済みの **B104 A2（実行時経路への配線）** と **B104 A1（複合評価集合の検出力，eval 半 46 行）**．本反復では新たに確定させない．
- **新規発見（要 backlog 起票）**: conformal オフライン評価が一貫して `--education-threshold 0.0` を使ってきたことによる，実行時分布との education 側の乖離．被覆・集合サイズへの影響は未評価．次に conformal 系列へ戻る際は `--education-threshold 0.05` を既定に含めるかを検討する必要がある．
- **新規発見（要 backlog 起票，運用上の落とし穴）**: `mise run analyze`（datetime 引数省略時）はアルファベット順 `sort` でディレクトリを選ぶため，`results/iter45_preliminary/` のような非タイムスタンプ形式のディレクトリが最新のタイムスタンプディレクトリより後にソートされ，誤って選択される．本反復では `analyze <datetime>` と明示して回避したが，スクリプト側の修正（タイムスタンプ形式のみを対象にする，または最終更新時刻でソートする）を検討する価値がある．

### 反省・判定確定 (Iter74, reflector)

**数値の独立再検証（出力 jsonl から reflector 自身が再計算した）**

実行フェーズの報告値を鵜呑みにせず，`results/20260923_142619/Iter74_raps_penalty.jsonl` と `results/20260923_135653/Iter73_randomized_aps.jsonl` を直接読み直して再集計した．eval 半 n=800 で coverage=0.90750・mean_set_size=4.16375・P(size<=2)=0.0775・サイズ分布 `{1:5, 2:57, 3:154, 4:259, 5:241, 6:79, 7:5}`・複合 46 行の 2 ドメイン同時被覆 0.2826・対応あり差分（減少 86／同数 507／増加 207）・`ttest_rel` t=-7.514964855512641, p=1.5276e-13 をすべて再現した．非退行も再検証し，`probabilities` の最大絶対差 0.0（ドメイン辞書の全キー・全 1,600 行），`selected_domain` 不一致 0 件，`split` 不一致 0 件，`set_size` は 1〜7 で `prediction_set` に重複なし，後方互換アンカーの md5 `afde650eea9ac611a43f0bb20703c2e3` は Iter73 出力と一致した．

**ノイズか信号かの切り分け**

- `mean_set_size` の +0.1625 は対応あり 800 行の SE=0.1625/7.515=0.02163 に対し **7.5 SE** であり，ノイズではなく信号である．しかも方向が主基準（減少）の**逆**で，合格条件（減少幅 >= 0.5）とは 0.66 の隔たりがある．
- `coverage` の 0.90375 → 0.90750（+0.00375）は二項 SE=0.0106 の 0.35 倍で**ノイズ幅の中**．帯 0.88-0.95 の内側（下限から +2.59 SE，上限まで -4.01 SE）であり，主基準 1 は満たす．
- 実測と事前予測の差は `mean_set_size` で **0.0**（事前登録した許容幅 ±0.02 の中心）．coverage・q_hat・サイズ分布・差分内訳も含め予測と乖離が無いので，Iter16/20/21/22/27 型の「設定を変えたのにコードへ到達しない」失敗ではないと確定できる．

**判定（確定）: `raps_penalty` は rejected．ただし「実験不成立」ではなく「実装は正しく発火したうえで手法が効かない」**

事前登録した解釈規則（実測が予測 4.16375 の ±0.02 以内なら実装成立・手法無効と読む）をそのまま適用した．差が 0.0 であることに加え，stderr の発火証拠（`q_hat=0.0323`，λ=0 なら 0.0939 が出る箇所）と後方互換 md5 一致という 2 つの独立した証拠が揃っている．因果として言えるのはここまでで，「RAPS は一般に無効」ではなく **「校正・評価の両側へ同形のペナルティを入れる分位点法は，クラス数が少なく裾の軽い本問題では q_hat の自己補正により平均サイズを縮めない」** という条件付きの主張に留める．

**`conformal_set_size_reduction` レバーは全 2 値試行済みでクローズ．「集合サイズを構成法で 2 近傍へ縮める」路線も閉じる**

閉じる根拠は本反復 1 点の結果ではない．(a) k_reg∈{1,2,3,4}×λ∈[0.001,0.5] の全域掃引で最小が λ→0（＝`randomized_aps`）だったこと，(b) サイズ最適性が証明されている LAC/THR（Sadinle et al., JASA 2019）でも名目 0.90 で 3.825 が床であること，(c) 本反復でその掃引の 1 点が実装上も寸分違わず再現されたこと，の 3 点である．size≈2 には被覆水準を 0.80 前後まで落とすほかなく，それは複合設問の 2 ドメイン同時被覆を 0.26→0.065 へ崩す（Iter73 分析 5 と整合）．B109 の要レビュー事項「予測集合を top-k dispatch へ流用する構想は conformal の保証（第 1 ドメインの周辺被覆）と目的（2 ドメイン同時網羅）がずれている」は，本反復で**定量的に裏付けられた**．

**次の単一レバー（Iter75）: 新レバー `conformal_runtime_distribution_alignment = education_threshold_0.05`**

`conformal_set_size_reduction` のクローズにより config の levers は再び全て試行済みになったため，skill の停止条件を順に適用し，**停止条件 1（学びから新レバーを考案）** を採った．本反復の実機本走で得た新事実——実行時ルーティングの確率分布は `education` に +0.05 された分布であり，Iter56〜74 の 19 反復の conformal はその補正を外した分布の上で被覆を議論してきた——が，そのまま次に振るべき 1 変数を与えている．`--education-threshold 0.0 → 0.05` の 1 点だけを動かし，Iter73 の adopted 構成（`randomized_aps` / seed 42 / `eval_holdout` / `alpha_lower` / `true_class` / 0.90）は固定する．既存 CLI に実装済みのため新規実装は不要，オフライン完結で実機ノードも使わない．

このレバーを選んだ理由は 2 つある．第 1 に，**B104 A2（conformal を実行時経路へ配線するか）を人間に諮る前に答えておくべき前提条件**だからである．オフラインで測った被覆が実行時分布でも成立するかが未確認のまま配線の是非を問うても，判断材料が欠けている．第 2 に，これは新手法の検証ではなく既存 19 反復の結論の外的妥当性の確認であり，`production_deployment_gap` レバー（Iter57 以降）と同じ性格の，自律着手可能で結論が確定する作業だからである．主基準は「実行時と同一の分布でも 0.88 <= coverage <= 0.95 が保たれること」，副基準として education を正解とする行に限った被覆・集合サイズを報告する（+0.05 が効くのはこの部分集合であり，ここが動かなければ 19 反復の議論は実行時分布でもそのまま有効と言える）．

**この反復で得た非自明な学び（次の自分向け）**

1. **事前シミュレーションは「合格を探す道具」ではなく「反証実験を 1 点に絞る道具」として機能した**．Iter71 以降 4 反復連続で予測が実測と一致しており，本反復では掃引全域で主基準を満たす点が無いことを事前に知ったうえで，あえて 1 点を実行して手法を根拠付きで閉じた．λ を振り直して合格を探さないと事前に宣言しておいたことが，rejected を「失敗」ではなく「確定した知見」に変えている．
2. **rejected の解釈規則を実行前に数値で登録しておくと，実装バグと手法の無効を事後に切り分けられる**（本反復では ±0.02）．この repo が 6 回繰り返した「実験不成立を効果なしと誤読する」失敗への，事前登録という形の対策として機能した．
3. **19 反復にわたるオフライン系列の入力が，実行時経路の入力と同一であることは誰も確認していなかった**．ユーザー指示で実施した実機本走（レバーではない基準線再取得）の副産物として初めて突き合わせ，education のみ +0.05 ずれていることが分かった．オフラインで長く回す系列は，入力分布が実行時と一致しているかを定期的に一次データで確認する必要がある．

---

## Iteration 73: conformal予測集合にランダム化APSを導入し被覆を保ったまま集合サイズを縮小する

### 調査 (Iter73)

**問い**

- Q1: Romano et al. (2020) の APS における「ランダム化」とは，非適合スコアと予測集合の構成のどこに，どういう形で入るものか（一次資料の定義）．
- Q2: 本実装（`_compute_prediction_set()` の `corrected_aps` + `predict_calibrated_rows()` の `eval_holdout` 校正）は，その定義のどこからずれているのか．Iter72 で残った +4.0pt の過被覆は，そのずれで定量的に説明できるか．
- Q3: ランダム化版に直したとき，coverage と mean_set_size はどこへ着地するか（本実行前に着地点を数値で言語化する．Iter71・Iter72 で有効だった手順）．

**Q1: ランダム化 APS の定義（出典付き）**

- Romano, Sesia & Candès, "Classification with Valid and Adaptive Coverage", NeurIPS 33 (2020), arXiv:2006.02544（<https://arxiv.org/abs/2006.02544>）．同論文の中核は "a novel conformity score"（generalized inverse quantile conformity score）であり，クラスを確率降順に並べたとき，真クラス y の順位を r，その上位クラスの確率質量を ρ = Σ_{j<r} π_(j)，真クラス自身の確率を π_(r) として

  `E(x, y, u) = ρ + u · π_(r)`,  `u ~ Uniform(0,1)`

  をスコアとする．校正集合の E の (1-α) 分位点を τ とし，テスト点では同じ規則で `C(x, u) = { y : ρ_y + u · π_y ≤ τ }` を出す．**同一の u を校正側とテスト側の両方で使う**点，および**打ち切り位置のクラスを確率 (τ - ρ_L)/π_L で含める／含めない**点が「ランダム化」の実体である．u を混ぜることで E の分布が連続になり，被覆が名目値へ（離散化による上振れなしに）一致する．
- 非ランダム化版が保守側（過被覆）になることは実装側の一次情報でも明示されている．Angelopoulos らの参照実装 `aangelopoulos/conformal_classification` の README は `ConformalModel` の `randomized` フラグについて "This will lead to conservative coverage, but deterministic behavior" と記す（<https://github.com/aangelopoulos/conformal_classification>）．なお README の当該文は `randomized=True` と書かれており True/False が入れ替わっているように読めるが，「非ランダム化＝保守的（過被覆）・決定的」という対応自体は同リポジトリの `conformal.py` の実装および Romano et al. の定義と整合する（この読み替えは当方の判断である）．
- RAPS（次点レバー）の出典は Angelopoulos, Bates, Malik & Jordan, "Uncertainty Sets for Image Classifiers using Conformal Prediction", arXiv:2009.14193．APS が大きな集合を生む問題にサイズ正則化（k_reg, λ）で対処する系列であり，本イテレーションでは扱わない．

**Q2: 本実装のずれ（コードを読んで特定）**

`scripts/evaluate_classifier_calibration.py`（全 733 行）の現状は，スコアの複素成分 `S = 1 - cumsum`（Romano の E の補数）で書かれている．校正側とテスト側で U が食い違っている．

- 校正側（L375-384，`eval_holdout` 分岐）: `true_class_scores[pos] = 1.0 - cumsum`（真クラスまでの**inclusive** な cumsum）．これは `S = 1 - (ρ + π_r)`，すなわち Romano の `E` で **u = 0** に固定した場合に一致する．
- テスト側（L164-170，`corrected_aps`）: 確率降順に**先に append してから** `1 - cumsum <= q_hat` で break する．すなわちクラス k が集合に入る条件は `1 - cumsum_{k-1} > q_hat`，これを k 自身の量で書き直すと `1 - cumsum_k + π_k > q_hat` であり，**u = 1** に固定した規則と同値である．
- したがって現行は「校正で u=0，評価で u=1」という**不整合な組み合わせ**であり，評価側だけが 1 クラス分だけ寛容になる．これが保守側にずれる（過被覆する）機序であり，Iter72 の考察が「非ランダム化 APS の離散化に帰属」と述べた +4.0pt の正体を，コードの行レベルまで具体化したものである．
- 併せて確認した事実: `_compute_prediction_set()` は `cp_data` から毎行 q_hat を再計算する構造（L143-158）であり，行ごとに異なる u を渡すには引数を 1 つ足すだけでよい．`predict_calibrated_rows()` の `eval_holdout` 分岐は評価 1,600 行を一括で採点済み（L360 `all_probs`）なので，u のベクトルも行 index で一括生成できる．

**Q3: 事前シミュレーション（本実行前に実施）**

本レバーは `probabilities` を一切変えないため，Iter72 の出力 `results/20260923_132431/Iter72_eval_holdout.jsonl`（1,600 行，`split` 付き）から coverage・mean_set_size を**厳密に再現計算できる**（Iter72 で実測と完全一致した手法をそのまま踏襲）．u は `np.random.default_rng(seed).random(1600)` を dataset 順に割り当て，校正半・評価半で同じ u を使う．q_hat は現行と同じ `alpha_lower`（`floor((n+1)α)/n`，n=800）．

まず現行実装の再現を確認した（規則の同値性の検証）:

| 規則 | q_hat | coverage（eval 半 n=800） | mean_set_size |
|---|---|---|---|
| 校正 u=0 / 評価 u=1（＝**現行 Iter72**） | 0.051766 | 0.9400 | 5.5213 |
| Iter72 の実測値 | 0.051766 | 0.940000 | 5.52125 |

小数点以下まで一致するので，シミュレータは本実装と同じものを計算している．その上でランダム化 APS（両側で同一 u，包含条件 `1 - cumsum_k + u·π_k >= q_hat`）を評価した:

| seed | q_hat | coverage | mean_set_size | P(set_size<=2) | 空集合率 |
|---|---|---|---|---|---|
| **42（本実験）** | **0.093879** | **0.9038** | **4.0012** | **0.1800** | 0.0000 |
| 1 | 0.085317 | 0.9025 | 4.1837 | 0.1550 | 0.0025 |
| 7 | 0.094661 | 0.8950 | 3.9975 | 0.1862 | 0.0037 |
| 2026 | 0.093322 | 0.8962 | 4.0475 | 0.1812 | 0.0000 |
| 123 | 0.091981 | 0.8912 | 4.0075 | 0.1837 | 0.0000 |
| 200 seed 平均 | 0.091735 | 0.8991 ± 0.0052 | 4.0540 ± 0.0791 | 0.1751 | 0.0012 |

- coverage は名目 0.90 へ**厳密に寄る**（200 seed 平均 0.8991，範囲 0.8850-0.9113）．Iter72 の +4.0pt の過被覆はほぼ消える．
- mean_set_size は 5.52125 → 4.0012（seed=42）で **-1.52**．対応あり 800 行の差分は 759 行で減少・41 行で同数・**増加は 0 行**，対応あり SE=0.02398（t=63.4）．
- set_size<=2 の割合は 0.0550 → 0.1800（44 行 → 144 行）．集合サイズ分布は seed=42 で `[52, 92, 154, 182, 178, 111, 29, 2, 0, 0]`（size 1〜10），Iter72 は `[9, 35, 54, 97, 175, 179, 176, 68, 7, 0]`．
- **正直に記録すべき分解**: 上の縮小分は「乱数を入れたこと」そのものよりも，**校正側と評価側で u を揃えたこと**（現行の u=0/u=1 不整合の解消）が大半を占める．参考として u を両側 1 に固定した決定的な変種は coverage=0.8963・mean_set_size=3.9688 であり，ランダム化版とほぼ同じ水準に着く．ランダム化の固有の寄与は「被覆を名目へ厳密に一致させる（保守性を残さない）」ことであって，サイズ縮小の主因ではない．この点は結果の解釈で誇張しない．
- 留保: 空集合が低確率（200 seed 平均 0.12%）で発生しうる．既存の fallback（`_compute_prediction_set()` L180-182 で top クラスを入れる）をそのまま残すため，出力上は必ず 1 以上になる．この fallback は厳密には被覆を保守側へわずかにずらすが，影響は 0.1% 台で，dispatch 用途では空集合の方が無意味なので残す判断とする．発生件数は報告する．

### 計画 (Iter73)

**仮説**

Iter72 に残った +4.0pt の過被覆（coverage=0.940 vs 名目 0.90）と mean_set_size=5.52 は，`_compute_prediction_set()` が校正側で u=0・評価側で u=1 に相当する非整合な APS を実装していることに起因する．Romano et al. (2020) の定義どおり，**校正・評価の両側で同一の一様乱数 u を用いる**ランダム化 APS に直せば，coverage は名目 0.90 近傍（0.89-0.91）へ下がり，その分 mean_set_size は 5.52 から 4.0 前後へ縮む．

**単一レバー**

`conformal_set_size_reduction`: 予測集合の構成を `corrected_aps`（Iter70 以降の非ランダム化版）→ **`randomized_aps`** へ変更する．動かすのはこの 1 点のみ．

**固定する構成（Iter72 の最良構成に固定）**

`--calibration-source eval_holdout` / `--holdout-seed 42` / `--qhat-quantile-direction alpha_lower` / `--qhat-source true_class` / `--confidence-level 0.90`．評価データ `data/dataset.jsonl`（1,600 行），分類器 `models/domain_classifier.joblib`，埋め込み `nomic-embed-text:latest`（wafl-ctrl5 の `127.0.0.1:11435`），`--education-logit-bias 0.0` / `--education-threshold 0.0`，`--fine-tuned-embed-model` は指定しない．`config.yaml`・`http_server.py`・`classifier.py`・`aggregator.py`・`mise.toml` は変更しない．実機ノード wafl500〜509 は不使用．**乱数 seed は 42 に固定し，結果が帯外でも振り直して合格を探さない**．

**変更箇所（`scripts/evaluate_classifier_calibration.py` 1 ファイル＋テスト．行番号は 2026-09-23 計画時点，全 733 行）**

1. `_compute_prediction_set()`（L66-184）:
   - 引数へ `randomization_u: float | None = None` を追加．
   - `set_construction` の許容値へ `"randomized_aps"` を追加（L133-136 の `ValueError` も更新）．`randomized_aps` のとき `randomization_u is None` なら `ValueError`（無言で非ランダム化へ落ちない．Iter69 の「no-op 失敗が事後再計算でしか気付けなかった」教訓）．
   - 集合構成の新分岐: 確率降順に走査し，**`1.0 - cumsum + randomization_u * probabilities[idx] >= q_hat` が成り立つ間だけ append し，成り立たなくなった時点で break（その最後のクラスは append しない）**．`corrected_aps` の「append してから break」とはここが違う．既存 2 分岐（`corrected_aps` / `broken`）は 1 行も書き換えない．
   - 空集合 fallback（L180-182）は共通のまま残す．
2. `predict_calibrated_rows()`（L187-528）:
   - 引数へ `randomization_seed: int = 42` を追加．
   - `set_construction == "randomized_aps"` かつ `calibration_source != "eval_holdout"` は `ValueError` で拒否する（`oof_train` は校正行が別データセットで行 index が共有できず，u を両側で揃えられないため．黙って不整合な u を使うより落とす）．
   - `eval_holdout` 分岐（L314-403）で `u_all = np.random.default_rng(randomization_seed).random(len(dataset))` を dataset 順に生成し，校正スコア（L375-384）を `1.0 - cumsum + u_all[i] * probs[idx]` に変える（`randomized_aps` のときのみ．`corrected_aps` のときは現行式のまま）．
   - 評価ループ 2 箇所（L466-471 の fine-tuned 分岐，L508-513 の ollama 分岐）の `_compute_prediction_set()` 呼び出しへ `randomization_u=u_all[row_idx]` を渡す（`randomized_aps` 以外では `None`）．**この 2 箇所はどちらも通す必要がある**．
   - stderr 診断（L422-429）へ `randomization_seed` を追記し，q_hat の診断計算も同じ u 付きスコアで行う（**レバー発火の証拠**．`randomized_aps` なら `q_hat≈0.0939`，`corrected_aps` なら `q_hat=0.0518` が出るはず）．
3. `main()`:
   - `--set-construction`（L639）の `choices` へ `randomized_aps` を追加．
   - `--randomization-seed`（`type=int`, `default=42`）を新設．
   - **CLI の `--output` 有無 2 分岐の両方へ伝播する．Iter69〜72 で 4 回連続して警告されている箇所であり，実装完了時にチェックリストとして目視確認すること**:
     - [ ] stdout 側 `if args.output is None:`（**L685-706** の `_run(...)`）
     - [ ] ファイル出力側 `with open(args.output, "w", ...)`（**L708-731** の `_run(...)`）※本実験が通るのはこちら
   - `_run()`（L531-573）のシグネチャと `predict_calibrated_rows()` 呼び出しへも追加．
4. **Iter72 で `UnboundLocalError` を出した教訓への対処**: 新分岐を足す前に，既存分岐でのみ初期化される局所変数（`u_all`・`precomputed_eval_holdout`・`holdout_split`・`cp_data`・`n_cal` など）を洗い出し，関数先頭で `None` 初期化されているかを確認してから書く．
5. `tests/test_evaluate_classifier_calibration.py`（全 479 行）へ既存の `test_set_construction_*` / `test_calibration_source_*` に倣って追加:
   (a) `randomized_aps` で `randomization_u=1.0` としたとき，同じ q_hat の下で `corrected_aps` と同一の集合を返すこと（u=1 同値性．Q2 の主張のユニットテスト化），
   (b) `randomization_u=0.0` では u=1 のときより集合が小さいか等しいこと（単調性），
   (c) `randomized_aps` かつ `randomization_u=None` で `ValueError`，
   (d) `randomized_aps` かつ `calibration_source="oof_train"` で `ValueError`，
   (e) 同一 `randomization_seed` で 2 回実行して出力が一致すること（再現性），
   (f) 既定値（`corrected_aps`）の回帰テストが通ること．

**到達コードパス**

CLI `--conformal-prediction --calibration-source eval_holdout --holdout-seed 42 --set-construction randomized_aps --randomization-seed 42 --qhat-quantile-direction alpha_lower --qhat-source true_class --confidence-level 0.90 --output ...`
→ `main()` L708（ファイル出力分岐）→ `_run()` L531 → `predict_calibrated_rows()` L187 → `eval_holdout` 分岐の校正スコア計算（**u が入る第 1 の地点＝q_hat の値が変わる**）→ 評価 1,600 行ループ（ollama 分岐 L487-527）→ `_compute_prediction_set(..., randomization_u=u_all[row_idx])`（**u が入る第 2 の地点＝集合の大きさが変わる**）→ 出力 jsonl の `prediction_set` / `set_size` / `split` → eval 半 800 行で集計．`config.yaml` を経由しないため「デプロイ漏れ」型の失敗は構造上起こらない．唯一のリスクは CLI 2 分岐の伝播漏れと，評価ループ 2 箇所のうち片方だけへの `randomization_u` 伝播漏れであり，予備実行（先頭 20〜40 行）の stderr で `set_construction=randomized_aps randomization_seed=42 q_hat=0.0939` を目視確認して潰す．

**成功条件（事前登録）**

名目 0.90，評価半 n=800．比較対象は `results/20260923_132431/Iter72_eval_holdout.jsonl`（Iter72 実測 coverage=0.940000，mean_set_size=5.52125）．

| 指標 | 定義 | Iter72 実測 | 合格条件 |
|---|---|---|---|
| coverage（主基準 1） | eval 半で `mean(expected_domains[0] in prediction_set)` | 0.940000 | **0.88 <= coverage <= 0.95**（帯内に留まること） |
| mean_set_size（主基準 2） | eval 半の `mean(set_size)` | 5.52125 | **5.52125 から有意に減少**（対応あり 800 行の差分平均が 0 より有意に大．両側 t 検定 p<0.01，かつ減少幅 >= 0.5） |
| P(set_size<=2) | eval 半で `mean(set_size<=2)` | 0.0550 | 報告のみ（判定には用いない．top-2 dispatch への流用可否の目安） |
| ECE | eval 半で `metrics.py:compute_ece()` | 0.076460 | 報告のみ（`probabilities` が不変なので変化しないはず） |

- **adopted**: 主基準 1・2 の両方を満たし，かつ下記の非退行条件を全て満たすこと．
- **rejected**: どちらかが不成立．向きで解釈を分ける（事前登録）:
  - `coverage < 0.88`（過少被覆）: ランダム化により保守性が抜けすぎた，または校正半 800 行の q_hat の分割ノイズが支配的．raps_penalty へ進む前に名目水準の再設定を検討する材料になる．
  - `mean_set_size` が縮まない: 実装が発火していない可能性がまず疑わしい（stderr の q_hat を確認）．真に縮まないなら「本分類器の確率分布では名目 0.90 を保ったままルーティングに使える集合は作れない」という結論に近づき，raps_penalty へ進むか conformal 系列を閉じるかを人間に諮る．
- **非退行条件**（いずれも eval 半 800 行について）:
  1. `selected_domain` が `Iter72_eval_holdout.jsonl` の同 id 行と完全一致すること．
  2. `confidence`・`probabilities` が同 id 行と一致すること（許容差 1e-9）．
  3. `split` の割り当て（cal/eval）が Iter72 と完全一致すること（`--holdout-seed 42` 固定なので変わってはならない）．
  4. `set_size` が 1 以上 10 以下で，`prediction_set` に重複がないこと．
  5. 後方互換: `--set-construction corrected_aps`（他は同条件）での再実行が Iter72 出力と **md5 一致**すること．
  6. 予備実行の stderr に `set_construction=randomized_aps` と `randomization_seed=42` が出ること（発火証拠）．
- **ノイズ幅**: n=800・p≈0.90 で二項 SE≈0.0106．事前シミュレーションでは乱数由来のばらつきが coverage で 0.8850-0.9113（200 seed，sd=0.0052）・mean_set_size で 4.84-5.26（sd=0.079）あり，**乱数 seed 由来のばらつきは二項 SE の半分程度**である．seed=42 の予測値 coverage=0.9038 は帯下限 0.88 から +2.24 SE（二項 SE 基準），帯上限 0.95 まで -4.36 SE の位置にあり，Iter72（上限まで -1.19 SE）より余裕がある．
- **付随報告（判定に用いない）**: (i) 実測 q_hat（予測 0.093879）と校正半の u 付きスコア分布の分位点，(ii) 集合サイズのヒストグラム（予測 `[52, 92, 154, 182, 178, 111, 29, 2, 0, 0]`），(iii) 空集合 fallback の発火件数（予測 0 件），(iv) u を両側 1 に固定した決定的変種の coverage / mean_set_size（予測 0.8963 / 3.9688）との比較＝「ランダム化そのものの寄与」の分解，(v) 名目水準を 0.70-0.95 で振った coverage / mean_set_size 曲線，(vi) 複合設問（eval 半に 46 行）の set 内 2 ドメイン被覆率．

**期待効果**

被覆保証（Iter72 で達成）を帯内に保ったまま mean_set_size を 5.52 → 4.0 前後へ下げ，set_size<=2 の割合を 5.5% → 18% へ広げる．これは top-2 dispatch への流用にはまだ届かないが，届かなかった場合に「残差へ RAPS のサイズ正則化を当てる」か「conformal 系列を閉じる」かを判断するための，正しい実装の上での基準点になる．Iter56 以来の conformal 系列で初めて，実装欠陥の修正ではなく**手法本来の性能を測る**イテレーションである．

**コスト**: 1 実行 10-30 分（埋め込み 1,600 行のみ）．GPU 実機占有なし，分類器再訓練なし，オフライン完結．

### Iteration 73 実行済み

**変更（2 ファイルのみ．`config.yaml`・`http_server.py`・`classifier.py`・`aggregator.py`・`mise.toml` は不変．実機ノード wafl500〜509 は不使用．実行ホストは wafl-ctrl5）**

1. `scripts/evaluate_classifier_calibration.py`: `_compute_prediction_set()` へ `randomization_u` を追加し `set_construction="randomized_aps"` 分岐を新設（`randomization_u is None` なら `ValueError`．無言で非ランダム化へ落ちない）．`predict_calibrated_rows()` へ `randomization_seed`（既定 42）を追加し，`eval_holdout` 分岐で `u_all` を dataset 順に生成して校正スコア（`1 - cumsum + u·p_true`）と評価側の打ち切り判定の**両方**に同じ u を使う．`randomized_aps` × `calibration_source != eval_holdout` は `ValueError` で拒否．CLI に `--set-construction randomized_aps` / `--randomization-seed` を追加し，`--output` 有無の 2 分岐**両方**へ伝播（Iter69〜72 で 4 回警告された箇所．今回は伝播漏れなし）．
2. `tests/test_evaluate_classifier_calibration.py`: 計画 5 の (a)〜(f) 6 件を追加．既存 18 件と合わせ **24 件 PASS**，`ruff` PASS．

**実験（`results/20260923_135653/`，2 実行）**

- A（後方互換アンカー）: `--set-construction corrected_aps` → `Iter73_corrected_aps_backcompat.jsonl`．
- B（本実行）: `--set-construction randomized_aps --randomization-seed 42 --calibration-source eval_holdout --holdout-seed 42 --qhat-quantile-direction alpha_lower --qhat-source true_class --confidence-level 0.90` → `Iter73_randomized_aps.jsonl`．stderr は `set_construction=randomized_aps randomization_seed=42 q_hat=0.0939 n_cal=800`（**レバー発火の証拠**．`corrected_aps` なら 0.0518 が出るはずの箇所）．

| 指標 | Iter72 実測 | 事前予測（seed=42） | Iter73 実測（eval 半 n=800） | 合格条件 | 判定 |
|---|---|---|---|---|---|
| coverage（主基準 1） | 0.940000 | 0.9038 | **0.90375** | 0.88 ≤ coverage ≤ 0.95 | **PASS** |
| mean_set_size（主基準 2） | 5.52125 | 4.0012 | **4.00125** | 有意に減少・減少幅 ≥ 0.5 | **PASS**（-1.52，`ttest_rel` t=63.376, p≈0） |
| q_hat | 0.051766 | 0.093879 | 0.0939 | — | — |
| P(set_size≤2) | 0.0550 | 0.1800 | 0.1800 | 報告のみ | — |
| ECE | 0.076460 | 不変のはず | 0.076456 | 報告のみ | — |

- 対応あり 800 行の set_size 差分（Iter72 − Iter73）: **減少 759 行・同数 41 行・増加 0 行**．集合が大きくなった行は 1 つもない．
- 集合サイズ分布（eval 半）: `{1:52, 2:92, 3:154, 4:182, 5:178, 6:111, 7:29, 8:2}`．事前予測 `[52, 92, 154, 182, 178, 111, 29, 2, 0, 0]` と完全一致．
- 非退行 6 項目すべて充足: `selected_domain` 完全一致 / `probabilities` 1e-9 一致 / `split` 割当一致 / `set_size` 1〜8 かつ重複なし / A の再実行が Iter72 出力 `results/20260923_132431/Iter72_eval_holdout.jsonl` と **md5 一致** / stderr 発火証拠あり．
- 付随（reflector が本実行 jsonl から再計算）: 空集合 fallback は発火 0 件と推定（`selected_domain` は全 800 行で `prediction_set` に含まれ，size 1 の 52 行も規則由来）．複合設問 46 行の**2 ドメインとも被覆**は 0.478 → **0.283**，どちらか 1 つ被覆は 0.913 → 0.870，複合行の mean_set_size は 5.196 → 3.826．
- 未実施の付随報告: 計画 (iv) u を両側 1 に固定した決定的変種の本実行での再測定，(v) 名目水準 0.70-0.95 の掃引曲線．いずれも判定に用いない項目であり，(iv) は調査 Q3 の事前シミュレーション（coverage=0.8963 / mss=3.9688）で代替できる．

### 分析(解釈) (Iter73)

**1. 主基準は両方 PASS．しかも実測が事前シミュレーションと小数点以下まで一致した**

coverage=0.90375（予測 0.9038）・mean_set_size=4.00125（予測 4.0012）・q_hat=0.0939（予測 0.093879）・サイズ分布まで一致した．本レバーが `probabilities` を変えないため Iter72 出力から厳密に再現計算できるという計画の前提が正しく，実装が意図した規則をそのまま実現している．Iter16/20/21/22/27 系列の「設定は変えたのにコードへ到達しない」型の失敗でないことは，stderr の `q_hat=0.0939`（`corrected_aps` なら 0.0518）と A 実行の md5 一致（既存経路を壊していない）の両方で二重に確認できている．

**2. coverage の改善は「ノイズ」ではない**

coverage 0.940000 → 0.90375（-3.63pt）は二項 SE=0.0105 の **3.5 SE** に相当し，seed 由来のばらつき（200 seed で sd=0.0052）の 7 倍である．名目 0.90 との差は +0.375pt = 0.36 SE で，**統計的に名目と区別できない**．Iter72 に残っていた +4.0pt の過被覆はほぼ完全に消えた．mean_set_size の減少は対応あり比較で増加 0 行・t=63.4 であり，ノイズ幅の議論の余地がない．

**3. 仮説は支持されたが，機序の帰属は計画どおり限定的に述べる**

Iter72 考察が「残る過被覆は非ランダム化 APS の離散化に帰属」と述べた推定は，本反復でコードの行レベル（校正側 u=0 / 評価側 u=1 の不整合）まで特定され，修正で予測どおりの量が消えた．ただし**調査 Q3 で事前に明記したとおり，サイズ縮小 -1.52 の大半は「乱数を入れたこと」ではなく「校正側と評価側で u を揃えたこと」で説明される**．u を両側 1 に固定した決定的変種は coverage=0.8963 / mss=3.9688 とほぼ同水準に着く（事前シミュレーション）．ランダム化固有の寄与は「被覆を名目へ厳密に一致させ保守性を残さない」ことであって，サイズ縮小の主因ではない．この点を誇張しない．

**4. 実用上の到達点: 4.00 は 5.52 より明確に良いが，top-2 dispatch にはまだ届かない**

10 ドメイン中 4.0 個を返す集合は「絞り込み」としてまだ弱い．set_size ≤ 2 は 5.5% → 18.0% へ 3.3 倍になったが，dispatch へ流用するには過半が ≤2 である必要がある．最頻値も 4〜5 のままである．

**5. 被覆の適正化は複合設問の 2 ドメイン被覆を犠牲にしている（本反復で新たに見えた事実）**

複合設問 46 行で「2 ドメインとも集合に入る」割合は 0.478 → 0.283 へ下がった．集合が小さくなれば当然だが，**conformal の被覆保証は `expected_domains[0]`（第 1 ドメイン）に対する周辺被覆であり，多ラベルの同時被覆は一切保証していない**．「集合サイズを縮めて top-2 dispatch に使う」という当初の狙いは，この方向の縮小では複合設問の網羅性と直接トレードオフになる．なお n=46 なので単独では結論にならない（B104 A1 の検出力問題は未解消）．

**6. 留保**

- 判定は seed=42 の 1 点である（事前登録どおり振り直していない）．200 seed 平均は coverage 0.8991±0.0052 / mss 4.0540±0.0791 で，seed=42 は平均近傍にある．
- 空集合が低確率（200 seed 平均 0.12%）で発生しうる仕様は残る．seed=42 では 0 件だった．
- eval 半 800 行・`classifier_train.jsonl` との重複 32 件という Iter72 由来の条件は変わっていない．

### 考察 (Iter73)

**判定: adopted**．事前登録の主基準 2 つと非退行 6 項目をすべて満たした．実装は **revert せず維持**する（既定は `corrected_aps` のままで md5 後方互換を保つ設計であり，以降は `--set-construction randomized_aps --randomization-seed 42` を固定値として使う）．Iter56 以来の conformal 系列で初めて，実装欠陥の修正ではなく**手法本来の性能を測れた**反復である．

**この反復で確定した知見**

1. split conformal では**校正側と評価側で非適合スコアの定義が一致しているか**が，分位点方向や集合構成の細部よりも先に効く．本件の不整合（校正 u=0 / 評価 u=1）は「両方とも APS」と書かれた 2 つの分岐に分かれて存在し，片方ずつ読む限り誤りに見えなかった．**校正スコアの式と評価側の包含条件を並べて同じ記号で書き下す**という手順が唯一の検出法だった（Iter72 の学び 1「同じモデルから出ているか」の，スコア式版の系）．
2. ランダム化 APS の効果は「被覆を名目へ厳密に合わせる」ことであって「集合を小さくする」ことではない．実測の縮小 -1.52 の大半は不整合解消分で，決定的な u=1 両側固定でもほぼ同じ所に着く．**論文の主張（randomization）と，自分の実装で実際に効いた要因（整合性）を混同しない**．
3. conformal の被覆保証は第 1 ドメインに対する周辺被覆であり，複合設問の 2 ドメイン同時被覆は保証しない．集合縮小は同時被覆（0.478 → 0.283）と直接トレードオフする．**予測集合を top-k dispatch へ流用する構想は，そもそも保証している量が違う**という点を次の設計判断で明示的に扱う必要がある．
4. 事前に「本実行前に着地点を数値で言語化する」手順（Iter71 で導入・Iter72・Iter73 で有効）は 3 反復連続で実測と一致した．`probabilities` を変えないレバーでは，この事前シミュレーションが事実上の実装検証になる．

**次の一手**

- レバー `conformal_set_size_reduction` の未試行値 **`raps_penalty`** が残っているため，Iter74 の単一レバーはこれとする（levers 使い切りではないので新レバー考案も再探索も不要）．
- 根拠: 無償で得られる縮小分（不整合解消＋ランダム化）は本反復で取り切った．mean_set_size=4.00 は Iter72 の 5.52 より明確に良いが，dispatch への流用に要る ≤2 には届かない．残差に対して明示的なサイズ正則化（RAPS の k_reg・λ）を当てる順序は，Iter72 reflector が config の note に事前登録した方針そのままである．
- 粒度の注意: RAPS はハイパラが 2 個ある．**k_reg はルーティングの要求から 2 に先に固定し，λ の 1 次元だけを振る**（config の note に記載済みの制約）．λ の値は事前シミュレーション（Iter72/73 出力の `probabilities` から再現計算できる）で先に掃引し，本実行は 1 点に絞って事前登録する．
- 事前に想定される限界: 分析 5 より，RAPS で mean_set_size を 2 近傍まで落とすと複合設問の 2 ドメイン同時被覆はさらに下がる公算が高い．Iter74 では**複合 46 行の同時被覆を付随報告に必ず含める**こと．

**人間判断を要する事項（今回新たに確定させない）**

- B104 A2（conformal を実行時経路 `http_server.py` / `classifier.py` へ配線するか）は未回答のまま維持する．mean_set_size=4.00 では依然として dispatch の絞り込みに使えないため，raps_penalty の結果を見てから諮るのが妥当．今回 @mention は不要．
- B104 A1（複合評価集合の検出力．eval 半で 46 行）も未回答のまま維持する．分析 5 の所見はこの n では確定させない．

---

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

