## Iteration 107: 送出集合の確率の和を複合行の確信度として設計する

### 経緯（B198〜B200）

- B198（人間の決定）: 方向 (i)（複合行に合った確信度．例: 送った上位 k のドメインの確率の和）を承認．ECE は基準線 `results/20261004_173104/results.jsonl` から新旧の両方の定義で計算し，過去と比べられるようにする．あわせて replay と本走の selected_domain の不一致（共通行で 16 行）を調べる．
- B199: 調査・計画フェーズを「第 1 段の診断 → 第 2 段のレバー設計」に分けて委譲する．
- B200（第 1 段の結果）: 不一致は入力の埋め込みの違い（本走は実行時に Ollama で埋め込む）だけから生じ，決定経路の差は無い（3,744 行のすべてで本番の `select_dispatch_targets` が再現）．16 行は k=2 の近傍の入れ替わり 8 行と，上位 2 つの僅差による argmax の反転 8 行．第 2 段の条件: (1) 効果は同じ eval キャッシュ上の「変種 − replay の none」の対応のある比較で測る，(2) 選択 16 行・top1 正味 ±2 行・複合行 ECE ±0.3pt をノイズの床とする，(3) 僅差 49 行と kNN の 2 位・3 位の接近行を除いた部分集合でも効果を確かめる．

### 第 2 段（レバー設計）

#### 調査

- 旧来の ECE（`metrics.py:486-521`）は，確信度を `confidence`（rank 1 の確率．`run_experiment.py:143`）とし，正誤を `selected_domain ∈ expected_domains` とする top-label の ECE である（Guo ら，ICML 2017，arXiv:1706.04599）．複合行では送出集合が複数のドメインを持つのに，確信度は rank 1 の確率だけなので，過小確信（ギャップ −0.112）になる．
- Mortier ら "On the Calibration of Probabilistic Classifier Sets"（AISTATS 2023，PMLR v206）は，確信度に基づく ECE の拡張を扱う．集合の確率の和を「集合が正解を含む確率」と読む解釈は，抄録の範囲から推定したもので，本文では確かめていない．
- "Set Learning for Accurate and Calibrated Models"（ICLR 2024）は，ECE は予測器の小さな摂動で大きく揺れうると指摘している（Kakade & Foster 2004，Foster & Hart 2018 を引用）．ノイズの床を置く根拠の一つにした．
- 本走の `probe_candidates` は全行で 10 ドメインの確率を持ち（和は 1±4e-16，最大値は `confidence` と一致），基準線の results.jsonl からコードを変えずに新しい ECE を事後計算できる．

#### 仮説

送出集合の確率の和を確信度とし，正誤を「送出集合 ∩ 正解 ≠ ∅」とすれば，確信度と正誤の母集団が一致する．その結果，複合行の過小確信が縮み，複合行の ECE が下がる．単一行は大半が k=1 なので，ほとんど動かない（推定．送出数の分布は未確認）．

#### 単一レバー

- `dispatched_set_confidence`: 報告用の確信度を，`confidence`（p_(1)）だけの状態から，`dispatched_confidence`（Σ_{d∈dispatched} p_d）を加えた状態へ変える．決定は変えない．
- 変更箇所: `run_experiment.py:143` の直後に `dispatched_confidence` を加える（fallback と dispatch_failed の行は null）．`metrics.py` に集合の正誤で ECE を計算する関数と `ece_dispatched_set`（Brier と AUROC の集合版も）を加え（`metrics.py:670` の付近），`print_summary`（`metrics.py:723-724`）に 1 行加える．
- 既存の `confidence`，`ece`，`brier_score`，`auroc` と，API のレスポンスは変えない．

#### 固定する構成

`models/domain_classifier.joblib`（MD5 `c7172ad37c10e1082a42481553ae25b0`，k=2，λ=0.3，T=0.9426），`dispatch_gap_threshold` 0.36，`dispatch_gap_max_k` 4，max_confidence．

#### replay による予測

`.claude/research/_iter107_replay_setconf.py`（出力は `_iter107_replay_setconf.json`，要点は `_iter107_design_notes.md`）．ECE は旧 → 新で，括弧内は「新 − 旧」の行を対にした bootstrap（2000 回）の 95% CI である．top1 は none 変種と同じ（決定を変えないので差は 0）．

| 範囲 | 行数 | replay | 基準線の本走からの事後計算 |
|---|---|---|---|
| 共通行 | 3,744 | 0.0370 → 0.0289（−0.0081 [−0.0154, −0.0025]） | 0.0373 → 0.0288（−0.0085） |
| 単一行 | 3,014 | 0.0199 → 0.0189（−0.0010 [−0.0089, +0.0025]） | 0.0210 → 0.0196（−0.0014） |
| 複合行 | 730 | 0.1146 → 0.0745（−0.0401 [−0.0624, −0.0218]） | 0.1122 → 0.0694（−0.0429 [−0.0639, −0.0231]） |
| 接近行を除いた複合行 | 609 | 0.1226 → 0.0719（−0.0507 [−0.0747, −0.0302]） | 0.1224 → 0.0707（−0.0516） |
| OOF（単一ラベル） | 2,275 | 0.0198 → 0.0144（5 fold すべてで新 < 旧） | — |

- 複合行の符号付きギャップは −0.1146 → −0.0745（replay）で，過小確信は残る．
- 接近行は，上位 2 つの差が 0.02 未満の 49 行と，kNN の 2 位と 3 位の類似度の差が 1e-3 未満の 376 行である．
- replay − 本走（複合行）は旧 +0.0024，新 +0.0052 で，新の定義では B200 の床（±0.3pt）を超える．床を ±0.6pt と見積もり直した（1 回の比較に基づく）．予測される効果 −4.0pt はその約 7 倍である．
- OOF は訓練集合が単一ラベルだけなので，選択には使わず，単一行の非退行の確認にだけ使った．

#### 成功条件（事前登録．config.yml の success_criteria (7)）

- 主基準: 本走の共通行の複合行で，新の ECE ≤ 旧の ECE − 0.02，かつ「新 − 旧」の 95% CI の上限 < 0．
- 非退行: 共通行の全体で新 ≤ 旧，単一行で新 ≤ 旧 + 0.005，旧の定義の ECE（共通行）≤ 0.0373 + 0.006 = 0.0433．C1〜C6 は Iter105 の値のまま．
- 決定の不変: \|Δtop1\| ≤ 0.1pt，かつ McNemar が有意でない．(1) の McNemar を主基準とする規定と (6) の invalid の規定は適用しない．埋め込みの取り直しで selected の不一致が 16 行程度は出る．
- C7，ECE の判定帯，AUROC，Brier は新旧の両方を併記し，どちらの定義かを明記する．

#### 実験の計画

- `run_experiment.py` と `metrics.py` は `Dockerfile:23` でイメージに入り，本走は app コンテナの中で実行される（`mise.toml:219`）．このため，イメージの再ビルドと deploy が要る．
- G0: 短い動作確認（数十問）で，`dispatched_confidence` が results.jsonl に出ることと，`ece_dispatched_set` が metrics.json に出ることを確かめる．あわせて，`dispatched_confidence` が `probe_candidates` からの事後計算と一致することも確かめる．`tools/smoke_check.py:54` の `DEPLOYED_FILES` に 2 つのファイルが無いので，イメージの中の MD5 を手で照合する．
- 本走: 3,750 行で行い，基準線 `results/20261004_173104` と (7) で比べる．基準線の事後計算は既に (7) の主基準を満たしているので，本走はこれが新しいフィールドの実装でも再現するかを確かめる位置づけである．

### Iteration 107 実装・実験（2026-10-04〜05）

**再開の経緯**: 前のセッションは本走を起動した後，記録を残す前に中断した．23:19 の `[start] ERROR sh exited with non-zero status` は，ローカルの待機ループの SSH が切れただけで，本走（wafl500 の PID 787025，`docker compose exec -d`）は継続していた．二重には起動していない．この節は，rc-executor の報告をもとにオーケストレータが追記した（rc-executor は追記の前にツールを使えなくなった）．

**変更**（未コミット）: `run_experiment.py`（+13），`metrics.py`（+41），`tests/test_metrics.py`（+58），`tests/test_run_experiment.py`（+5）．既存の `confidence`，`ece`，`brier_score`，`auroc` は変えていない．`pytest` は 63 件が通り，`ruff check` も通った．`ruff format --check` は 3 つのファイルを指摘したが，HEAD の時点から未整形だった行だけなので整形していない．

**イメージの照合**: コンテナの中の MD5 はローカルと一致した（`run_experiment.py` `3fee3ec39dfaeae2ccfdb4d116038867`，`metrics.py` `34de20337e6335d4dbbd7e78ad0c3b4f`）．10 ノードのイメージの digest は `sha256:c9f5224177257cf488b4fe66b948823d73dbbea693b2333d5fe6b34a7352e072` で揃っている．`git_head.txt` は `a3e9801-dirty-iter107`，`data/dataset.jsonl` の MD5 は `769f2a58dc807e23e1b73e44b97e97b8`（ビルドの前後で同じ）．

**G0**（`results/20261004_225309/preview20.jsonl`，20 行）: 20 行のすべてに `dispatched_confidence` があり，`probe_candidates` からの事後計算との不一致は 0 行，`ece_dispatched_set` も出力された．20 行とも単一行で送出先が 1 つだったので，複合の送出の照合は本走の 740 行で行った（不一致 0 行，null の規則の違反 0 行）．

**本走**: `results/20261004_225553`（22:55:53 開始，01:14 に `results.jsonl.done`，3,750 行．期限の前に完了）．`mise run analyze`，`metrics.py --json`，`mise run stop` はいずれも exit 0．集計は `.claude/research/_iter107_main_summary.{py,json,out}`．

共通行（両方の実行で `confidence` が非 null の 3,744 行）での新旧の ECE（CI は行を対にした bootstrap 2000 回，seed 107 の「新 − 旧」の 95% 区間）:

| 範囲 | n | ECE 旧 | ECE 新 | 新 − 旧 [95% CI] | 符号付きギャップ 旧 → 新 |
|---|---|---|---|---|---|
| 共通行 | 3,744 | 0.037281 | 0.028756 | −0.008525 [−0.015309, −0.002286] | −0.0365 → −0.0288 |
| 単一行 | 3,014 | 0.020953 | 0.019582 | −0.001370 [−0.009046, +0.003058] | −0.0182 → −0.0189 |
| 複合行 | 730 | 0.112223 | 0.069354 | −0.042869 [−0.064626, −0.023781] | −0.1122 → −0.0694 |
| 接近行を除いた複合行 | 609 | 0.122354 | 0.070718 | −0.051637 [−0.073291, −0.028434] | −0.1224 → −0.0707 |

| 範囲 | Brier 旧 → 新 | AUROC 旧 → 新 | top1（本走 / 基準線） | McNemar（b_only / a_only，p） | selected の不一致 |
|---|---|---|---|---|---|
| 共通行 | 0.100591 → 0.057541 | 0.84924 → 0.81890 | 0.849626 / 0.849626 | 0 / 0，p=1.0 | 0 |
| 単一行 | 0.090644 → 0.054863 | 0.86295 → 0.83962 | 0.860319 / 0.860319 | 0 / 0，p=1.0 | 0 |
| 複合行 | 0.141659 → 0.068602 | 0.80827 → 0.74047 | 0.805479 / 0.805479 | 0 / 0，p=1.0 | 0 |
| 接近行を除いた複合行 | 0.138649 → 0.069238 | 0.80339 → 0.72123 | 0.822660 / 0.822660 | 0 / 0，p=1.0 | 0 |

全行（`metrics.json`）: top1 0.849333 [0.83753, 0.86042]（基準線 0.848267），ECE 旧 0.037401 / 新 0.028884（3,748 行），Brier 旧 0.100528 / 新 0.057525，AUROC 旧 0.849260 / 新 0.818719，dispatch_failure_rate 0.000533（2 行．基準線は 6 行），fallback_rate 0，compound_mean_dispatched_count 1.980822（基準線と同じ）．軸 2 / 3 は answer_quality_accuracy 0.58609（3,020 行），end_to_end_accuracy 0.40587．

**異常と未確認の点**:

- ノードのログに traceback，OOM，CUDA error は無い．`dispatch_model_not_ready`（Ollama の 500）が wafl503（23:11:49）と wafl506（23:13:31）で 1 件ずつあった．本走の dispatch_failed の 2 行に当たると推定するが，request_id は照合していない．
- **共通行で selected_domain の不一致が 0 行，旧の ECE が基準線と 6 桁まで一致した**．B200 の見込み（16 行程度）とは違う．B200 の 16 行は「replay と本走」の差であり，「本走と本走」の差ではないので，同じハードウェアの Ollama の埋め込みが決定的なら説明はつく（推定）．新しい実行である根拠は，`dispatched_confidence` の有無，dispatch_failed の行数（2 と 6），`mean_duration_ms`，`git_head` の違いである．一方，`request_id` と `probe_candidates` の確率値の行ごとの照合は行っていない．rc-evaluator はこれを判定の前に確かめること．

### Iteration 107 実行済み（2026-10-05）

この節は，rc-evaluator の報告をもとにオーケストレータが追記した（rc-evaluator は書き込みの前にツールを使えなくなった）．照合のスクリプトは `.claude/research/_iter107_verify.py` と `_iter107_auroc.py`（読み取りだけ．コミットしない）．

**照合（本走が今回の実行の出力か）**:

- `request_id` は 3,750 行のすべてで基準線と異なる．`dispatched_confidence` は本走の 3,750 行にあり，基準線には無い．`answer_text` の一致は 2,347 / 3,750 行で，生成は非決定的である．以上から，本走は基準線の写しではない（オーケストレータも `request_id` と `dispatched_confidence` を再確認した）．
- 一方，共通 3,744 行の `confidence` と `probe_candidates` の 10 ドメインの確信度は，基準線と完全に一致した（最大差 0.0）．同じイメージ，同じ artifact，同じハードウェアでは，Ollama の埋め込みと分類器の出力が決定的だと推定する．B200 の 16 行は replay と本走の差であり，本走どうしの差ではなかった．success_criteria (7) の「selected の不一致は 16 行程度は出る」という見込みは誤りだった．
- dispatch_failed の 2 行は，ログの `dispatch_model_not_ready` と `request_id` で対応した（`medical-110` は wafl503 の 23:11:49，`natural_science-079` は wafl506 の 23:13:31）．基準線で失敗した 6 行のうち 4 行は，本走では成功して正答だった．

**判定: adopted**（success_criteria (7)．ECE は共通行の値）:

| 条件 | 基準 | 実測 | 判定 |
|---|---|---|---|
| 主基準 | 複合行で新 ≤ 旧 − 0.02 = 0.092223，かつ CI の上限 < 0 | 0.069354，CI [−0.064626, −0.023781] | PASS |
| 非退行（全体） | 新 ≤ 旧 | 0.028756 ≤ 0.037281 | PASS |
| 非退行（単一行） | 新 ≤ 旧 + 0.005 = 0.025953 | 0.019582 | PASS |
| 非退行（旧の定義） | ≤ 0.0433 | 0.037281 | PASS |
| 決定の不変（共通行） | \|Δtop1\| ≤ 0.1pt，McNemar が有意でない | Δ 0，0 / 0，p = 1.0 | PASS |
| C1 | BH 補正後の有意な退行 0 件 | 不一致は本走だけが正答の 4 行だけで，退行の向きの差は無い | PASS（推論．関数での再計算はしていない） |
| C2 | fallback 0，dispatch_failure_rate ≤ 0.005 | 0，0.000533 | PASS |
| C3 | G0 の記録と MD5 | `_iter107_g0_*` があり，ローカルの MD5 は一致する．イメージの中の artifact の MD5 の記録は無いが，probe の出力 37,500 値が基準線とビット単位で一致する | PASS（注記つき） |
| C4 | 複合行の top1 ≥ 0.780411 | 0.805479 | PASS |
| C5 | set_recall ≥ 0.54，mean_dispatched ≤ 2.10 | 0.573288，1.980822 | PASS |
| C6 | mean_duration_ms ≤ 3041.7 | 2223.79 | PASS |
| C7 | ECE ≤ 0.08（新旧の併記） | 旧 0.037401，新 0.028884（metrics.json，3,748 行） | PASS |

- 根拠: 複合行の ECE は 0.112 → 0.069（−4.29pt）で，replay の予測（−4.01pt）と合う．揺らぎの床（±0.6pt）の約 7 倍である．決定は全行で不変である．
- 注記: 全行の top1 は 0.848267 → 0.849333（+0.107pt）で，帯の 0.1pt をわずかに超える．差の 4 行はすべて基準線だけで送出に失敗した行で，レバーとは関係が無い（全行の McNemar は 4 / 0，連続補正つきで p ≈ 0.13）．(7) の主基準が共通行を指定しているので，決定の不変も共通行で判定した．
- 採用の意味: `dispatched_confidence` と `ece_dispatched_set` を報告に加える．既存の `confidence` と `ece` は変えていないので，過去の結果とは旧の定義で比べられる．

**学び**:

1. 同じ構成の本走どうしでは，probe の確率まで一致した（推定: 決定的）．決定を変えない報告用のレバーでは，基準線の results.jsonl からの事後計算が本走の値と一致した．同じ型のレバーで本走（約 2.3 時間）を事後計算で置き換えるかは，手順の変更なので B203 で人間に尋ねる．
2. 単一行の送出数（失敗を除く 3,018 行）は，k=1 が 2,538 行（84.1%），k=2 が 95 行，k=4 が 385 行で，k=3 は 0 行である．B201 の推定「単一行の大半は k=1」は確かめられた．k≥2 の 480 行では，確信度の平均が 0.509 → 0.963，正答率が 0.527 → 0.985 と大きく動くが，両方が一緒に動くので，単一行の ECE はほとんど変わらない．k=3 が出ない理由は確かめていない．
3. 複合行の送出数（730 行）は，k=1 が 470 行，k=2 が 32 行，k=4 が 228 行である．k=4 の行の正答率は，旧の定義で 0.570，新の定義で 0.969 である．
4. 新の定義では AUROC が下がる（共通行 0.849 → 0.819，複合行 0.808 → 0.740）．新の定義の誤り 251 行（旧は 563 行）のうち 233 行は k=1 で，「1 つに自信を持って送って外した行」に誤りが集まる．k≥2 の行は確率の和が 0.95 前後に詰まるので，k の内側の AUROC も低い（共通行で k=2 が 0.57，k=4 が 0.73，k=1 が 0.81）．正例の率が 85% から 93% へ変わるので，新旧の AUROC と Brier は同じ尺度で比べられない．新の定義は送出数を増やすほど確信度も正誤も甘くなるので，送出数を変えるレバーとは独立に読めない．今後も新旧を併記する．
5. 送出の失敗（Ollama の 500，`dispatch_model_not_ready`）は 4 回目の再発である（Iter102，103，105，107）．

**次の一手**: config.yml の levers は再び使い切りになった．Iter106 では再探索で新しいレバーを定義できず，人間が方向を与えた．このため，次のレバーは自分では確定せず，Iter108 は調査・計画フェーズから始める（B202）．評価の手順に関わる 2 つの判断は B203 で人間に尋ねる．

## Iteration 106: kNN 類似度の中心化と ABTT を replay で検証する

**実施場所の申告**: 本フェーズで行ったのは，開発ホストでのリポジトリの読み取り，tavily（`tvly search`）による外部調査，開発ホストの CPU での replay（`uv run`．計算済みの埋め込みキャッシュを読むだけ）である．GPU，LLM，wafl500〜509 は使っていない．本走と deploy は行っていない．

**位置づけ**: config.yml の levers は Iter105 で使い切った（B195）．オーケストレータは方向 (i)（複数の正解を許す確信度．ECE の定義が変わる）を保留し，方向 (ii)（rank 1 の決定を変えて top1 を上げる）を採ると自動で判断した（B196）．調査・計画フェーズは 3 回委譲した．1 回目と 2 回目は，subagent が途中でシェルのコマンドを実行できなくなり，ファイルへの書き込みの前に終わった．3 回目で replay を完了した．

**確かめた事実**:

1. 過去の打ち止めとは重ならない．E7 の whitening（`router.py:670-742`）は `routing_method=embedding` の経路でしか読まれず，kNN 成分には一度も当てられていない（d0004:302，journal_archive.md 36553〜36678 行目）．Iter99 の PCA は LR の入力に対する操作である．k は Iter104 で {1, 2, 3, 5, 7} を掃引済みである．
2. kNN 成分の類似度の計算は `knn_interpolated_head.py:122-123` だけである（L2 正規化と内積）．融合埋め込みは (2275, 6656) で，各ブロックのノルムは 1，全体は 2 である．訓練行どうしの cosine は平均 0.618（p5 0.558，p95 0.705）で，狭い帯に詰まっている．

**外部調査（tavily，2026-10-04）**:

- Wang, Chao, Weinberger, van der Maaten, "SimpleShot: Revisiting Nearest-Neighbor Classification for Few-Shot Learning", 2019, arXiv:1911.04623（https://arxiv.org/abs/1911.04623 ）．CL2N（中心化と L2 正規化）の出典である．論文には，L2 正規化の後の中心化は精度をそれ以上は上げない，という趣旨の記述がある．
- Mu, Viswanath, "All-but-the-Top: Simple and Effective Postprocessing for Word Representations", ICLR 2018（https://openreview.net/ ）．平均と上位の主成分を取り除く後処理（ABTT）である．
- "Centering versus Scaling for Hubness Reduction"（ofai.at），"An Evaluation of Hubness Reduction Methods for Entity Alignment"（dbs.uni-leipzig.de）．中心化を hubness の低減策として扱う．

**replay**（`.claude/research/_iter106_replay_center.py`，結果は `_iter106_replay_result.json`）:

- 固定した構成は LR の artifact，k=2，λ=0.3，T=0.9426，外側 5-fold（seed 104）である．平均と主成分は fold の訓練部分だけから求めた．
- 再現の確認: 現行の OOF CV top1 は 0.826813（Iter105 の 0.8268 と一致）．eval の `predict_proba` は現 artifact との最大の絶対差が 0.0 である．
- 選択規則（事前登録）: CV top1 が現行を上回り，かつ 5 fold のうち 4 fold 以上で上回ること．主候補は cl2n，ほかの 4 つは参考候補である．

| 変種 | CV top1 | 上回った fold 数 | 規則 | eval の N_2 の歪度 / 最大出現回数 | eval 共通 3,744 行の top1 | eval の ECE |
|---|---|---|---|---|---|---|
| none（現行） | 0.826813 | — | — | 3.29 / 39 | 0.850160 | 0.0371 |
| cl2n（主候補） | 0.822857 | 1 | 不適合 | 3.68 / 53 | 0.849092 | 0.0353 |
| block_center | 0.822418 | 1 | 不適合 | 3.48 / 49 | 0.847756 | 0.0351 |
| abtt1 | 0.828571 | 2 | 不適合 | 4.30 / 60 | 0.848825 | 0.0347 |
| abtt3 | 0.829011 | 2 | 不適合 | 4.72 / 64 | 0.849359 | 0.0376 |
| abtt10 | 0.829451 | 3 | 不適合 | 6.13 / 78 | 0.848558 | 0.0452 |

- 基準線の本走 `results/20261004_173104` の共通 3,744 行の top1 は 0.849626 である．none の再現に対する McNemar は，どの変種でも p ≥ 0.29 で，C1（20 指標，BH q=0.05）の有意はどの変種でも 0 件である．
- 中心化で hubness が下がるという前提は成り立たなかった．N_2 の歪度と最大出現回数は，どの変種でも現行より大きい．
- 未解決: none の再現と本走とで selected_domain が共通行で 16 行（全行で 22 行）異なる．`predict_proba` は一致するので，差は分類器の外（同順位の並べ方，送出の失敗など）で生じている．原因は調べていない．

**判定**: 事前登録の規則を満たす候補が無いので，新しいレバーを定義しない（要人間判断: 実行可能な新レバーを定義できない）．skill の早期終了の手順により，フェーズ 2 と 3 は行わない．config.yml の levers は変えていない．採用構成は Iter105（`models/domain_classifier.joblib`，MD5 `c7172ad37c10e1082a42481553ae25b0`，T=0.9426）のままである．次の方向は B197 で人間の判断を仰ぐ．

## Iteration 105: kNN 補間分布の温度を再較正する

### 調査 (Iter105)

**実施場所の申告**: 本フェーズで行ったのは，開発ホストでのリポジトリの読み取り，tavily（`tvly search`）による外部調査，開発ホストの CPU での replay（`uv run`．計算済みの埋め込みキャッシュを読むだけ）である．GPU，LLM，wafl500〜509 は使っていない．

**位置づけ**: B193 と C7 の規定（ECE が 0.05 を超えたら，次の反復を校正側に立てる）による．Iter104 の本走 `results/20261004_132607` の ECE は 0.050104 だった．温度は LR の出力（`CalibratedClassifierCV(method="temperature")`）にしか当たっておらず，kNN 分布を 0.3 混ぜた後の分布には当て直していない（Iter104 の学び 4）．

**確かめた事実**:

1. **送出集合が確信度で決まる経路**（`aggregator.py:28-91` を Read で確認）．
   - `node.py:232-239` が `select_dispatch_targets()` を呼び，`confidence_threshold`（`config.yaml:33`，0.0），`dispatch_candidate_threshold`（`:38`，0.0），`gap_threshold`（`:131`，0.36），`gap_max_k`（`:132`，4）を渡す．
   - `aggregator.py:62` が確信度の降順で安定ソートする．`:71` と `:75-77` の閾値はどちらも 0.0 なので，10 ノードがすべて候補に残る．`:87-91` は，隣り合う順位の差が 0.36 未満の間，k を 4 まで増やす．
   - 各ノードの確信度は `classifier.py:69` の `predict_proba` の自ドメイン列で，実体は `knn_interpolated_head.py:99-107` である．温度を変えると argmax は変わらないが，隣り合う順位の差が変わるので，送出集合は直接動く．
   - `results.jsonl` の `confidence` は rank 1 の probe 確信度で，`metrics.py:486-521` の ECE はこれを読む．判定は `selected_domain in expected_domains`（複数の正解を許す）である．
   - replay の `simulate_rows()` は `:87-91` の連鎖を再現している．違いは同順位の扱いだけで，replay はクラスの順，本番は peers.yaml の順に並べる．
2. **単一ラベルの温度と，複数の正解を許す ECE の食い違い**（journal_archive.md 8973 行目）．複合行では，正解が 2 つあるため構造的に過小確信になる．Iter104 の本走では，単一行の符号付きギャップ（平均確信度 − 平均正答）が −0.031，複合行が −0.129 である．
3. **過去の本走の ECE**（`results/*/metrics.json`）: 20260923_150540 0.0551，20260926_221822 0.0327，20260929_192157（Iter101）0.0396，20261002_105743（Iter103）0.0385，20261004_132607（Iter104）0.0501．同じ LR の artifact で走った Iter101 と Iter103 の差は 0.0011 である．
4. **commit b1f5cb0**（2026-10-04）: 実験の後に `mise run stop` でコンテナを止め，deploy の最後に `docker image prune -a -f` を実行する．このため本走はモデルの再ロードから始まり，先頭の数十問の所要時間が伸びる見込みである（推測）．

**外部調査（tavily，2026-10-04）**:

- Guo, Pleiss, Sun, Weinberger, "On Calibration of Modern Neural Networks", ICML 2017, arXiv:1706.04599（https://arxiv.org/abs/1706.04599 ，https://proceedings.mlr.press/）．温度 T は検証集合の NLL を最小にして当てる．予測の順位を保つので accuracy は変わらない．T>1 で分布が平らになる．
- Minderer ほか, "Revisiting the Calibration of Modern Neural Networks", NeurIPS 2021（proceedings.neurips.cc）．温度スケーリングを，追加の費用が小さい基準の校正法として扱っている．
- Khandelwal ほか, kNN-LM, ICLR 2020, arXiv:1911.00172．補間 \(p = \lambda p_{kNN} + (1-\lambda) p_{LM}\) の出典である（Iter104 と同じ）．補間した後の分布を校正し直す手順は，検索の結果からは確かめられなかった．
- 限定: Guo らは，過信する（T>1 が要る）深層ネットを主な対象にしている．本件の補間分布は，OOF でも eval でも過小確信の向きにある（下記）．そのため T<1（分布を鋭くする向き）になる．温度スケーリングの式はどちらの向きにも使える．

**replay**（`/tmp/iter105/replay_temp.py`，写しは `.claude/research/_iter105_replay_temp.py`．結果は `.claude/research/_iter105_replay_result.json`．開発ホストの CPU で約 53 秒，EXIT=0）:

- 方法: 温度は \(p_T \propto p^{1/T}\)（log p を logit とみなす）で当てる．OOF は訓練集合 2,275 行の外側 5-fold（seed 104）の補間分布である．k=2，λ=0.3 で，kNN の近傍は fold の訓練部分だけから引く．eval の指標は，基準線の各行の `selected_domain`，`dispatched_domains`，`confidence` だけを差し替えて，`metrics.py` の関数で求めた．
- **T の選択（事前登録した唯一の規則．OOF の単一ラベル NLL 最小）**: T* = **0.9426**（探索範囲 [0.05, 20] の端ではない）．fold ごとの T は 0.869 / 0.952 / 0.899 / 1.010 / 0.972 である．OOF の NLL は 0.52657 → 0.52519，OOF の ECE は 0.0241 → 0.0198，符号付きギャップは −0.0195 → −0.0063．OOF の CV top1 は 0.8268 で，Iter104 の値と一致した．
- **T=1 で基準線の本走をどこまで再現できるか**: selected の一致率は 99.573%（不一致 16 行），送出集合の一致率は 98.213%（不一致 67 行）．確信度の差は平均 0.0045，最大 0.226 である．top1 は 0.8504（本走は 0.849867），ECE は 0.050269（本走は 0.050104），`compound_mean_dispatched_count` は 2.068493（本走は 2.064384）．

| 指標 | 本走（基準線） | replay T=1 | replay T*=0.9426 | 条件 |
|---|---|---|---|---|
| top1 | 0.849867 | 0.8504 | **0.8504**（T=1 との不一致 0） | \|Δ\| < 0.25pt |
| McNemar（対 本走） | — | — | 8 対 6，p=0.789 | 参考 |
| 複合行の top1 | 0.805479 | 0.808219 | 0.808219 | C4 ≥ 0.780411 |
| ECE | 0.050104 | 0.050269 | **0.037124** | 主基準 |
| ECE（単一 / 複合） | 0.0326 / 0.1292 | 0.0331 / 0.1316 | 0.0201 / 0.1146 | 参考 |
| 符号付きギャップ（全体） | −0.0501 | −0.0503 | −0.0368 | 参考 |
| `compound_domain_set_recall` | 0.589041 | 0.590411 | **0.574658** | C5 ≥ 0.5400 |
| `compound_mean_dispatched_count` | 2.064384 | 2.068493 | **1.975342** | C5 ≤ 2.10 |
| 全行の平均送出数 | 1.5696 | 1.5765 | 1.5245 | 参考 |
| 複合の k 分布 | {1:451, 2:30, 4:249} | {1:450, 2:30, 4:250} | {1:472, 2:31, 4:227} | 参考 |
| 単一の k 分布 | {1:2507, 2:90, 4:423} | {1:2502, 2:86, 4:432} | {1:2541, 2:91, 4:388} | 参考 |

- **C1（20 指標，BH q=0.05）**: T* と本走の比較で有意は 0 件（最小の p は 0.48）．T* と T=1 の比較でも 0 件．argmax が変わらないので，C1 が動く経路は送出の失敗と埋め込みの微小な差だけである．
- **参考値（選択には使わない）**:
  - eval で ECE が最小になる T は 0.80（ECE 0.0150，mean_dispatched 1.8137，set_recall 0.5521）．
  - eval で C5 の両方を満たす T の範囲は [0.75, 1.02]．
  - OOF で ECE が最小になる T は 0.94．
  - A2（複数の正解を許す判定に合わせた目的関数）: 訓練行はすべて単一ラベルなので，OOF の上では単一ラベルの NLL と一致し，別の値は出ない．eval の上で当てはめた値は選択に使えないので，近いものとして eval の ECE 最小（T=0.80）を併記するに留める．
  - eval の過小確信（ギャップ −0.050）は OOF（−0.020）より大きい．主な原因は複合行の構造的な過小確信（−0.13）であり，単一ラベルの規則ではこれを吸収しきれない（推測）．
- **replay の途中で起きた事故**: 初回のバッチで `replay_temp.py` への Write が「先に Read していない」として失敗した．続く Bash は写しただけの旧スクリプトを実行し，`/tmp/iter104/result_v2.json` を 16:40:15 に上書きした．CV の部分は乱数の種と入力が同じなので元と同じ値である．一方，eval の「基準 artifact の再現」の列は，いまの kNN 補間の artifact で求めた値に変わった．Iter104 の数値は journal の Iter104 節に残っている．このファイルは G0-a の照合元としては使えない（B194 に記録）．

### 計画 (Iter105)

**判断**: T* で ECE は 0.0130 下がる（no-op ではない）．C5 は mean_dispatched 1.975 で上限の内側にある．C1〜C7 を満たす見込みなので，このレバーを採る．代替案の調査（手順 3 の後段）は行わない．

**仮説**: Iter104 の補間分布は，訓練集合の OOF でも eval でも過小確信の向きにある（平均確信度が平均正答を下回る）．OOF の単一ラベル NLL で当てはめた温度 T*=0.9426 で分布を鋭くすると，rank 1 の確信度が上がり，ECE は 0.050 から 0.037 前後へ下がる．argmax は変わらないので top1 は動かない．隣り合う順位の差が広がるので，gap 方式の送出数は減る（mean_dispatched 2.06 → 1.98）．その分 set_recall は約 1.4pt 下がる．

**単一レバー**: `knn_interpolation_temperature`．`KnnInterpolatedClassifier.predict_proba()` の出力を，補間分布 \(p\)（T=1 に相当）から \(p_T \propto p^{1/T}\)，T=0.9426 へ変える．

**固定する構成（Iter104 の採用構成）**: base の LR の artifact（再訓練しない），k=2，λ=0.3，近傍の母集団 `data/classifier_train_iter94_dedup.jsonl` と `data/embcache_train_iter100_fused.npy`，融合表現，`dispatch_gap_threshold` 0.36，`dispatch_gap_max_k` 4，`aggregation_method: max_confidence`，`confidence_threshold` と `dispatch_candidate_threshold` 0.0，評価集合 3,750 行（MD5 `769f2a58dc807e23e1b73e44b97e97b8`），`config.yaml` のすべて．

**レバーを読むコード行と到達条件**:

- 読み込み: `http_server.py:428` → `classifier.py:48`（`joblib.load`）．
- 推定: `http_server.py:379` → `classifier.py:69` → `knn_interpolated_head.py:99-107`（温度を当てる場所）．
- 送出: probe の確信度 → `node.py:232-239` → `aggregator.py:62`（ソート），`:87-91`（gap の連鎖）．rank 1 の確信度は `results.jsonl` の `confidence` に入り，`metrics.py:486-521` の ECE が読む．
- 到達条件: `routing_method: supervised_classifier`（`config.yaml:75`），`confidence_signal_method: self_report`（`:74`），ノードのドメインが `classes_` に含まれること，`dispatch_gap_threshold` が null でないこと．**現行構成は 4 つとも満たす**．

**実装の仕様（フェーズ 2）**:

1. `knn_interpolated_head.py`: `KnnInterpolatedClassifier.__init__` に `temperature: float = 1.0` を加え，T ≤ 0 なら ValueError にする．`predict_proba()` は補間した後に \(p_T \propto \exp(\log(\max(p, 10^{-12}))/T)\) を行ごとに正規化して返す（最大値を引いてから exp を取る）．T=1 のときは現在の出力をそのまま返す．λ=0 の早期 return は，T=1 のときに限る．
   - Iter104 の artifact（`temperature` 属性を持たない pickle）を読めるよう，`__setstate__` で既定値 1.0 を補う．
   - 冒頭の docstring に温度の式と出典（Guo ら，2017）を加える．
2. `scripts/build_knn_interpolated_classifier.py`: `--temperature`（既定 1.0）を加え，ラッパーに渡す．値の導出は `.claude/research/_iter105_replay_temp.py` で再現できる．
3. artifact: 現在の `models/domain_classifier.joblib`（MD5 `c3888f66d90f4172cd7415e5c105328a`）を `models/domain_classifier_pre_iter105_knn.joblib` へ退避する．`--base-classifier models/domain_classifier_pre_iter104_lr.joblib --k 2 --interpolation-lambda 0.3 --temperature 0.9426` で作り直して新しい `models/domain_classifier.joblib` に置き，MD5 を journal に記録する．
4. `knn_interpolated_head.py` はイメージに焼き込まれるので，イメージを再ビルドする（B192 と同じく `mise.toml` の `docker build` と `docker push` だけ．`GIT_HEAD=<HEAD>-dirty-iter105`）．b1f5cb0 により deploy の最後に prune が走るので，前のイメージ `b5b717d1fc7a` には，レジストリの digest `sha256:0a499d0e…` で戻る．
5. テスト（`tests/test_knn_interpolated_head.py` に追加する）: T=1 で従来の出力と一致すること，T≠1 でも行和が 1 になること，T≠1 でも argmax が変わらないこと，T ≤ 0 で ValueError になること，`temperature` 属性を持たない pickle が T=1 で読めること，joblib で往復しても出力が変わらないこと．

**G0（本走の前に，すべての合格を条件とする）**:

- **G0-a**: 新しい artifact で eval キャッシュの `predict_proba` を求める．argmax が現在の artifact と 3,750/3,750 で一致すること．基準線の行の決定を差し替えた ECE が 0.0371 ± 0.0005，`compound_mean_dispatched_count` が 1.9753 ± 0.0030 であること（T を 4 桁に丸めた分の差を幅に含めた）．
- **G0-b**: 全 10 ノードで artifact と `knn_interpolated_head.py` の MD5 がローカルと一致すること．コンテナの中で `joblib.load` した結果が `KnnInterpolatedClassifier` で，`temperature` が 0.9426 であること．
- **G0-c**: 予備 20 問で，`probe_candidates` の確信度の argmax がオフラインの値（T を当てた後）と 20/20 で一致すること．確信度の差は最大 0.02 以内とする（Iter104 の G0-c の最大差は 0.0117）．`mean_duration_ms` は 3,041.7ms 以下であること．

**成功条件と非退行条件（事前登録．以後は変えない）**:

- **基準線**: Iter104 の採用本走 `results/20261004_132607`．top1 0.849867，複合行の top1 0.805479，`compound_domain_set_recall` 0.589041，`compound_mean_dispatched_count` 2.064384，ECE 0.050104，`mean_duration_ms` 2253.224．
- **主基準**: argmax が変わらないレバーなので，ECE を主に見る．
  - `adopted`: ECE ≤ 0.040，かつ |Δtop1| < 0.25pt，かつ C1〜C7 をすべて満たす．
  - `partial`: 0.040 < ECE ≤ 0.045，かつ |Δtop1| < 0.25pt，かつ C1〜C7 を満たす．
  - `negligible`: ECE > 0.045（下げ幅が 0.005 未満）で，C1〜C7 を満たす．
  - `rejected`: Δtop1 ≤ −0.25pt，または C1〜C7 のどれかに違反．
  - `invalid`: G0 に合格しない．または，本走の確信度の分布が基準線と変わらない（ECE の差が 0.001 未満で，かつ複合と単一の k 分布が基準線と同じ）．後者の場合は到達経路を先に調べる（success_criteria (6)）．
- **閾値の根拠**: replay の予測は 0.0371 である．T=1 の replay と本走の ECE の差は 0.00017，同じ artifact で走った本走 2 本（Iter101 と Iter103）の差は 0.0011 である．0.040 は予測から 0.0029 上で，観測した実行間の差の約 2.6 倍の余裕がある．0.040 は Iter101 の水準（0.0396）にあたり，C7 の再較正の基準 0.05 とも十分に離れている．|Δtop1| の帯 0.25pt は，Iter104 の `negligible` の帯と同じ値にした．
- **非退行条件**: Iter104 の C1〜C7 を数値もそのまま使う．
  - **C1**: precision と recall の計 20 指標を BH 補正（q=0.05）し，有意な退行が 0 件．
  - **C2**: `fallback_rate` 0.0，`dispatch_failure_rate` ≤ 0.005．
  - **C3**: G0-a〜G0-c の記録と artifact の MD5 がすべて残っていること．
  - **C4**: 複合行の top1 ≥ 0.780411．
  - **C5**: `compound_domain_set_recall` ≥ 0.5400，`compound_mean_dispatched_count` ≤ 2.10．
  - **C6**: `mean_duration_ms` ≤ 3041.7（絶対値の判定は変えない）．b1f5cb0 により本走がモデルの再ロードから始まるので，参考として，先頭 50 問を除いた平均と，単一層の中央値と p95，複合層の p25 と中央値を併記する．
  - **C7**: ECE ≤ 0.08．

**事前登録する予測**:

- **P1（主予測）**: ECE は 0.035〜0.040（点推定 0.037）で，`adopted` になる．
- **P2**: |Δtop1| < 0.1pt．argmax は変わらず，差は送出の失敗と埋め込みの微小な差だけから生じる（replay では本走と 8 対 6）．
- **P3**: `compound_mean_dispatched_count` は 1.96〜1.99，`compound_domain_set_recall` は 0.570〜0.580．set_recall は約 1.4pt 下がるが，C5 の内側にとどまる．
- **P4**: C1 の有意は 0 件．
- **P5**: 全行の平均送出数が約 3% 減るので，先頭 50 問を除いた `mean_duration_ms` は基準線から −5%〜+2% に入る．再ロードを含めた全体の平均も C6 の内側に入る．
- **P6**: 複合行の ECE は 0.11 前後に残る．単一ラベルで当てた温度では，複数の正解による過小確信は吸収しきれない．

**フェーズ 2 への申し送り**:

- コードの変更は，上の「実装の仕様」の 1，2，5 と，Dockerfile は変えずにイメージを再ビルドすることに限る．`config.yaml` は変えない．
- G0-a は開発ホストの CPU で行える（`uv run`）．`/tmp/iter104/result_v2.json` は上書きされたので照合には使わない．照合元は `.claude/research/_iter105_replay_result.json` である．
- deploy の後，本走の前に G0-b と G0-c を行う．本走は 3,750 問のフルスペックで行う．b1f5cb0 の手順により，analyze の後に `mise run stop` が走る．
- 判定の比較相手は `results/20261004_132607`．C1 は `metrics.py` の関数で計算する．ECE には単一行と複合行の内訳と，符号付きギャップを併記する．
- T を変えた場合の set_recall と所要時間のトレードオフ（参考格子）は `_iter105_replay_result.json` の `reference_not_for_selection` にある．T を選び直すことはしない．

### Iteration 105 実装・実験（2026-10-04）

**再ビルド前のイメージ（ロールバック用．deploy の末尾の prune で手元から消えるため，ビルドの前に記録した）**: `localhost:5001/expert-mesh:latest` の image ID は `sha256:b5b717d1fc7aeac6ff38cf67996d6526304db151a9913fedf6ea3e2a93a99459`（Created 2026-10-04T12:54:34+09:00，`GIT_HEAD=7c0b703-dirty-iter104`）である．RepoDigest とレジストリの v2 manifest の digest は `sha256:0a499d0e172319c7af21a09b64d46203a2872981e2dab9e1a23b1fc17f505868` である．戻すときは `localhost:5001/expert-mesh@sha256:0a499d0e…` を pull する．

**変更したファイル**（`config.yaml` と Dockerfile は変えていない）:

- `knn_interpolated_head.py`（MD5 `76e0d089a512f5b9c0c80be8f9a4a60c`）:
  - `apply_temperature_to_proba()` を加えた．式は \(p_T \propto \exp(\log(\max(p, 10^{-12}))/T)\) で，行の最大値を引いてから exp を取り，行ごとに正規化する．replay の `apply_temperature()` と同じ式である．
  - `KnnInterpolatedClassifier.__init__` に `temperature: float = 1.0` を加えた（T ≤ 0 なら ValueError）．`predict_proba()` は補間の後に温度を当て，T=1 のときは補間の結果をそのまま返す．
  - `__setstate__` で，`temperature` を持たない Iter104 の pickle に 1.0 を補う．冒頭の docstring に式と出典（Guo ら，2017）を加えた．
- `scripts/build_knn_interpolated_classifier.py`: `--temperature`（既定 1.0）を加え，ラッパーに渡す．
- `tests/test_knn_interpolated_head.py`: 6 関数（parametrize を含めて 7 ケース）を加えた．T=1 で従来の出力と一致すること，T≠1 で行和が 1 になること，T≠1 で argmax が変わらないこと，T ≤ 0 で ValueError になること，`temperature` 属性の無い pickle が T=1 で読めること，joblib の往復で出力が変わらないことを確かめる．

**テストと lint**: `uv run pytest -q tests/test_knn_interpolated_head.py tests/test_classifier.py` は 23 passed だった．warning は joblib の NumPy 2.5 の DeprecationWarning だけである．変更した 3 ファイルの `uv run ruff check` は All checks passed だった．

**artifact**:

- 退避: `models/domain_classifier_pre_iter105_knn.joblib`．MD5 `c3888f66d90f4172cd7415e5c105328a` で，元の artifact と一致した．
- 新しい `models/domain_classifier.joblib`: MD5 `c7172ad37c10e1082a42481553ae25b0`．`--base-classifier models/domain_classifier_pre_iter104_lr.joblib --k 2 --interpolation-lambda 0.3 --temperature 0.9426` で作った（n_train=2275，dim=6656）．

**イメージ**: `mise.toml` の `docker build`（`--build-arg GIT_HEAD=b1f5cb0-dirty-iter105`）と `docker push` だけを実行した．`mise run setup` は実行していない．ログは `.claude/research/_iter105_build.log` にある．

- 新イメージの image ID は `sha256:b4a3373f69436b4ab3900d7273cc3adb156b029244fd5750507a966ef52d05c8`，push の digest は `sha256:a328caf4860b925c53cb394004cd6be92879cb325d1095663107c2b441a062e9` である．
- イメージの中の `/app/knn_interpolated_head.py` の MD5 はローカルと一致した．
- `data/dataset.jsonl` の MD5 は，ビルドの前後とも `769f2a58dc807e23e1b73e44b97e97b8` だった．

**deploy**: `mise run deploy` は終了コード 0 で終わった（`.claude/research/_iter105_deploy.log`）．healthcheck と smoke_check（git-status，hashes，probe．probe の latency は 12ms）は通り，末尾の prune は全ノードと手元で実行された．

**G0**:

- **G0-a: 合格**（`.claude/research/_iter105_g0a.json`．スクリプトは `/tmp/iter105/g0a_check.py` で，replay の `simulate_rows()` と `summarize()` を再利用した）．
  - argmax は旧 artifact と 3750/3750 で一致した．
  - ECE は 0.037119（許容 0.0371±0.0005），`compound_mean_dispatched_count` は 1.975342（許容 1.9753±0.0030）だった．
  - replay の式で旧 artifact に T を当てた値との差の最大は 0.0 だった．
- **G0-b: 合格**（`.claude/research/_iter105_g0b.txt`）．
  - wafl500〜509 の全 10 ノードで，`/app/models/domain_classifier.joblib`（`c7172ad3…`）と `/app/knn_interpolated_head.py`（`76e0d089…`）の MD5 がローカルと一致した．
  - コンテナの中の `GIT_HEAD` は `b1f5cb0-dirty-iter105` だった．`joblib.load` の結果は `knn_interpolated_head.KnnInterpolatedClassifier` で，temperature=0.9426，k=2，λ=0.3 だった．
  - `docker inspect` の `Image` の表示は，wafl500〜507 が digest（`a328caf4…`），wafl508〜509 が image ID（`b4a3373f…`）だった．どちらも今回のビルドを指す．
- **G0-c: 合格**（`.claude/research/_iter105_g0c.txt` と `_iter105_g0c.json`．予備実行は `results/20261004_173015/preview20.jsonl`，`data/dataset_20.jsonl`，requester は wafl500）．
  - オンラインとオフライン（T=0.9426 の後）の argmax は 20/20 で一致した．確信度の差は最大 0.0111（business_economics-014）だった．
  - dispatch の失敗と fallback は 0 件，`mean_duration_ms` は 581.1 だった．

**本走**: `results/20261004_173104`（3,750 問，`mise run start -- --dataset data/dataset.jsonl --output results.jsonl`，requester は wafl500）．17:31:04 に起動し，19:50 頃に完了した（EXIT=0．ログは `.claude/research/_iter105_mainrun_start.log`）．

- `results.jsonl` は 3,750 行で，MD5 `d96dad0550d4750e34e39bc9dc63a3bf` はローカルと wafl500 側で一致した．`git_head.txt` は `b1f5cb0-dirty-iter105` である．
- `mise run analyze -- 20261004_173104` は終了コード 0 で終わった（`.claude/research/_iter105_analyze.log`．`answer_quality_accuracy` 0.583113，`end_to_end_accuracy` 0.401067）．
- `uv run python metrics.py --results results/20261004_173104/results.jsonl --json` の出力は `results/20261004_173104/metrics.json` に書いた．
- その後，`mise run stop` で全ノードのコンテナを止めた（終了コード 0．削除はしていない）．

**主要指標**（`metrics.json`，および replay の `summarize()` で同じ関数を両方の本走に当てた値．集計の記録は `.claude/research/_iter105_main_summary.json`）:

| 指標 | Iter105 本走 | 基準線 20261004_132607 |
|---|---|---|
| top1 | 0.848267 | 0.849867 |
| 複合行の top1 | 0.805479 | 0.805479 |
| `compound_domain_set_recall` | 0.573288 | 0.589041 |
| `compound_mean_dispatched_count` | 1.980822 | 2.064384 |
| ECE | 0.037281（n=3,744） | 0.050104（n=3,750） |
| ECE（単一 / 複合） | 0.02095 / 0.11222 | 0.03262 / 0.12915 |
| 符号付きギャップ（全体 / 単一 / 複合） | −0.0365 / −0.0182 / −0.1122 | −0.0501 / −0.0310 / −0.1292 |
| Brier | 0.100591（n=3,744） | 0.102122（n=3,750） |
| `mean_duration_ms` | 2233.403 | 2253.224 |
| 参考: 先頭 50 問を除いた平均 | 2254.678 | 2273.579 |
| 参考: 先頭 50 問の平均 | 659.08 | 746.98 |
| 参考: 単一層の中央値 / p95 | 535.5 / 3089.2 | 539.0 / 3252.4 |
| 参考: 複合層の p25 / 中央値 | 7668.25 / 8785.0 | 7583.5 / 8795.5 |
| `fallback_rate` | 0.0 | 0.0 |
| `dispatch_failure_rate` | 0.0016（6 行） | 0.0 |
| 参考: 全行の平均送出数 | 1.5243 | 1.5696 |
| 参考: 複合の k 分布 | {1:470, 2:32, 4:228} | {1:451, 2:30, 4:249} |
| 参考: 単一の k 分布 | {1:2540, 2:95, 4:385} | {1:2507, 2:90, 4:423} |

- ECE と Brier の n が 3,744 なのは，`confidence` が null の行が 6 行あるためである．この 6 行は `dispatch_failed` の 6 行と同じ件数である．
- C1（BH 補正の 20 指標）と McNemar は，このフェーズでは計算していない．フェーズ 3 で `metrics.py` の関数を使って計算する．
- 実行上の異常: dispatch の失敗が 6 件あった（`medical-110`，`natural_science-002`，`natural_science-079`，`natural_science-109`，`medical-exp085-140`，`education-exp085-044`．いずれも単一行で，送出先は 1 件（`dispatched_domains` は medical / natural_science / education）．原因はこのフェーズでは調べていない）．fallback は 0 件で，ハング，OOM，エラーの終了は無かった．ポーリングでは複合行の区間（17:55〜19:31 頃）で進み方が約 7.5 行/分に落ちたが，wafl500 側の `results.jsonl` は更新され続けていた．

### Iteration 105 実行済み

#### 計算の方法

- スクリプトは `.claude/research/_iter105_post_analysis.py` で，出力は `.claude/research/_iter105_post_analysis.json` に保存した．`uv run` で実行し，EXIT=0 だった．手順は `scripts/_iter101_post_analysis.py` に合わせた．
- 比較の相手は基準線 `results/20261004_132607/results.jsonl` である．id は 3,750 行で完全に一致する．
- top1 の対の比較には `metrics.compute_mcnemar_test`（連続補正つき．a=本走，b=基準線）を使った．C1 には `compute_domain_recall_mcnemar_test`，`compute_domain_precision_fisher_test`，`apply_benjamini_hochberg`（q=0.05）を使った．
- 参考として，送出に失敗した 6 行を両方の run から除いた共通の 3,744 行で，`compute_ece` と `compute_brier_score` を当て直した．主の判定には事前登録どおり `metrics.json` の値を使う．
- **一時スクリプトの退避**: `/tmp/iter105/{g0a_check.py,g0c_check.py,main_summary.py}` は再起動で消えうるので，`.claude/research/_iter105_g0a_check.py`，`_iter105_g0c_check.py`，`_iter105_main_summary.py` へ写した．写しの MD5 は元と一致した（`9ed511ab…`，`33c9a562…`，`873641cc…`）．G0-a の記録と主要指標の表は，これらで再現できる．`.claude/research/_iter10x_*` は過去の反復でも追跡していないので，写した 3 本と分析スクリプトもコミットせず，作業ツリーに残す（backlog B195）．
- テストと lint は分析の時点で再実行した．`uv run pytest -q tests/test_knn_interpolated_head.py tests/test_classifier.py` は 23 passed，変更した 3 ファイルと分析スクリプトの `ruff check` は clean だった．作業ツリーの `knn_interpolated_head.py` の MD5 は `76e0d089a512f5b9c0c80be8f9a4a60c` で，G0-b でイメージの中から取った値と一致する．

#### 結果（本走 `results/20261004_173104` 対 基準線 `results/20261004_132607`）

| 指標 | 基準線 | 本走 | Δ | 事前登録の予測 |
|---|---|---|---|---|
| ECE（`metrics.json`．主基準） | 0.050104（n=3,750） | **0.037281**（n=3,744） | −0.0128 | P1: 0.035〜0.040．的中 |
| 参考: ECE（共通の 3,744 行） | 0.049996 | 0.037281 | −0.0127 | — |
| ECE（単一 / 複合．共通の行） | 0.03245 / 0.12915 | 0.02095 / 0.11222 | −0.0115 / −0.0169 | P6: 複合は 0.11 前後．的中 |
| 符号付きギャップ（全体 / 単一 / 複合） | −0.0501 / −0.0310 / −0.1292 | −0.0365 / −0.0182 / −0.1122 | — | — |
| Brier（`metrics.json`） | 0.102122（n=3,750） | 0.100591（n=3,744） | −0.0015 | — |
| 参考: Brier（共通の 3,744 行） | 0.102235 | 0.100591 | −0.0016 | — |
| top1 | 0.849867 | **0.848267** | **−0.16pt（−6 行）** | P2: \|Δ\| < 0.1pt．**外れ** |
| McNemar（連続補正） | — | 本走のみ正答 0，基準線のみ正答 6，chi2 4.1667，p = 0.0412 | — | — |
| 参考: top1（共通の 3,744 行） | 0.849626 | 0.849626 | 0 | — |
| 複合行の top1 | 0.805479 | 0.805479 | 0 | — |
| `compound_domain_set_recall` | 0.589041 | 0.573288 | −1.58pt | P3: 0.570〜0.580．的中 |
| `compound_mean_dispatched_count` | 2.064384 | 1.980822 | −0.0836 | P3: 1.96〜1.99．的中 |
| 全行の平均送出数 | 1.5696 | 1.5243 | −2.9% | P5 の前提（約 3% 減）と一致 |
| `mean_duration_ms` | 2253.224 | 2233.403 | −0.88% | — |
| 先頭 50 問を除いた平均 | 2273.579 | 2254.678 | −0.83% | P5: −5%〜+2%．的中 |

- **top1 の −6 行の内訳**: 基準線のみ正答の 6 行は，`education-exp085-044`，`medical-110`，`medical-exp085-140`，`natural_science-002`，`natural_science-079`，`natural_science-109` で，本走で `dispatch_failed` になった 6 行と id で完全に一致した．本走のみ正答の行は 0 行である．失敗した 6 行を除いた共通の 3,744 行では，top1 は両方とも 0.849626 で同じ値になる．温度で正誤が変わった行は 1 行も無い．

#### 事前登録条件の照合（C1〜C7，Iter104 と同じ閾値）

| 条件 | 基準 | 実測 | 判定 |
|---|---|---|---|
| 主基準 | ECE ≤ 0.040 かつ \|Δtop1\| < 0.25pt | 0.037281，0.16pt | PASS |
| C1 | 20 指標の BH 補正（q=0.05）後の有意な退行が 0 件 | 0 件（20 指標とも生の p ≥ 0.248） | PASS |
| C2 | `fallback_rate` = 0.0，`dispatch_failure_rate` ≤ 0.005 | 0.0，0.0016 | PASS |
| C3 | G0-a〜G0-c の記録と artifact の MD5 | `_iter105_g0a.json`，`_iter105_g0b.txt`，`_iter105_g0c.{txt,json}` がある．artifact の MD5 は `c7172ad37c10e1082a42481553ae25b0`（全 10 ノードで一致） | PASS |
| C4 | 複合行の top1 ≥ 0.780411 | 0.805479 | PASS |
| C5 | set_recall ≥ 0.5400 かつ mean_dispatched ≤ 2.10 | 0.573288 / 1.980822 | PASS（上限まで残り 0.119） |
| C6 | `mean_duration_ms` ≤ 3041.7 | 2233.403 | PASS |
| C7 | ECE ≤ 0.08 | 0.037281 | PASS．0.05 を下回ったので，校正側に立てる規定は解除される |

- **C1 の内訳**: recall で差が出たのは `natural_science_recall`（本走のみ正答 0 対 基準線のみ正答 3，p = 0.248），`medical_recall`（0 対 2，p = 0.480），`education_recall`（0 対 1，p = 1.0）の 3 指標だけである．この 6 行は，送出に失敗した 6 行と同じである．ほかの 7 ドメインの recall は不一致 0 で，p = 1.0 である．precision の 10 指標はすべて p = 1.0 である（真陽性と選択数の差は，失敗した行の分の 1〜3 行だけ）．
- **C6 の分位点**（括弧内は基準線）: 単一層の中央値 535.5ms（539.0），p95 3,089.2ms（3,252.4）．複合層の p25 7,668.25ms（7,583.5），中央値 8,785.0ms（8,795.5）．先頭 50 問の平均は 659.08ms（746.98）で，再ロードによる遅れは平均を押し上げていない．

#### 事前登録の予測との照合

- **P1（主予測）: 一致**．ECE は 0.037281 で，予測区間 0.035〜0.040 に入り，点推定 0.037 とほぼ同じ値である．replay の 0.037124 との差は 0.00016，G0-a の 0.037119 との差も 0.00016 である．
- **P2: 不一致**．\|Δtop1\| は 0.16pt で，予測の 0.1pt 未満を超えた．argmax が変わらないこと自体は予測どおりで，差はすべて送出の失敗 6 行から生じた．予測は「送出の失敗と埋め込みの微小な差」を差の出どころに挙げていたが，失敗の件数を基準線の水準（0 行）で見込んでいた．このため，区間の幅が足りなかった．
- **P3: 一致**．mean_dispatched は 1.980822，set_recall は 0.573288 で，どちらも区間の内側にある．
- **P4: 一致**．C1 の有意は 0 件である．
- **P5: 一致**．先頭 50 問を除いた平均は −0.83% で，全体の平均も C6 の内側にある．
- **P6: 一致**．複合行の ECE は 0.112 で，0.11 前後に残った．

#### 判定（分析フェーズ，2026-10-04）

- **判定: `adopted`**．事前登録の判定規則（「計画 (Iter105)」）をそのまま当てはめた．閾値は緩めても締めてもいない．ECE は 0.037281 ≤ 0.040，\|Δtop1\| は 0.16pt < 0.25pt で，C1〜C7 をすべて満たす．G0 はすべて合格し，k の分布も基準線から動いているので，`invalid` には当たらない．
- **採用構成**: `models/domain_classifier.joblib`（`KnnInterpolatedClassifier`，k=2，λ=0.3，T=0.9426，MD5 `c7172ad37c10e1082a42481553ae25b0`）を以後の基準とする．T=1 の artifact は `models/domain_classifier_pre_iter105_knn.joblib`（MD5 `c3888f66d90f4172cd7415e5c105328a`）として残す．新しい基準線は本走 `results/20261004_173104` とする．
- **雑音と信号の切り分け**:
  - ECE の −0.0128 は，同じ artifact で走った本走 2 本（Iter101 の 0.0396，Iter103 の 0.0385）の差 0.0011 の約 12 倍ある．replay の予測とは 0.00016 の差で一致した．したがって，雑音ではなく温度の効果とみる．共通の 3,744 行に揃えても −0.0127 で，n の違いは結論に効かない．
  - 送出数の減少（set_recall −1.58pt，mean_dispatched −0.084）も replay の予測（0.5747，1.9753）に近く，温度の効果である．
  - top1 の McNemar は p = 0.041 で，名目上は 0.05 を下回る．ただし不一致の 6 行は，すべて送出に失敗した行である．共通の行では top1 は完全に同じである．したがって，この差はレバーの効果ではない．温度 T は分類器の確率にしか効かないので，送出の失敗はインフラ側の事象と推定している（推定であり，ollama のログが無いので原因は確かめていない）．判定規則は McNemar ではなく \|Δtop1\| の帯で事前登録しているので，判定は変わらない．
- **計画の仮説との一致**: 「補間分布は過小確信の向きにあり，T*=0.9426 で鋭くすると ECE が下がり，argmax は変わらず，送出数が減って set_recall が約 1.4pt 下がる」は，すべて支持された（set_recall の下げ幅は 1.58pt）．ECE の改善は主に単一行で出た（0.0326 → 0.0210）．複合行は 0.129 → 0.112 で，ギャップ −0.112 が残る．
- **想定外の挙動**: 言語崩れ，発散，OOM は無い．送出の失敗が 6 行あった（基準線は 0 行）．送出先のノードの ollama が `/api/chat` に `500 Internal Server Error` を返し，app のログに `dispatch_model_not_ready` が出ていた（wafl503 で 2 件，wafl506 で 3 件，wafl501 で 1 件．ほかに wafl509 で，複合行 `social_science-121` の 4 送出のうち 1 件．これはほかの応答で補われた）．失敗は 17:47〜17:49 と 19:33〜19:35 の 2 つの時間帯に固まっていた．probe は正常で，振り分けも正しかった．
- **後始末**: config.yaml は変えていないので，戻す設定は無い．`mise run stop` でコンテナは止めてある（削除はしていない）．

#### 学び (Iter105)

1. **argmax を変えない校正のレバーは，キャッシュの replay で本走の値まで予測できる**．ECE の差は 0.00016，set_recall と mean_dispatched も予測に近かった．argmax が変わらないので，top1 の差の出どころは送出の失敗だけになる．同じ型のレバーでは，本走は「インフラの揺れの中で予測が保たれるか」の確認になる．
2. **単一ラベルで当てた温度では，複合行の過小確信は消えない**．複合行のギャップは −0.129 → −0.112 で，ほとんど残った．原因は，複数の正解を許す判定（`selected_domain in expected_domains`）と，rank 1 の単一ドメインの確信度との食い違いである．これを ECE から除くには，確信度の定義（たとえば上位の確率の和）か判定の側を変える必要があり，温度の再調整では届かない（推測）．
3. **送出の失敗は，argmax を変えないレバーでも top1 の McNemar を「有意」にする**．今回は 0 対 6 で p = 0.041 だった．判定を \|Δtop1\| の帯で事前登録していたので，判定は揺れなかった．今後の対の比較では，`dispatch_failed` の行を除いた共通の行の値を必ず併記する．
4. **ollama の 500 による送出の失敗は，原因が確かめられたものとしては Iter102 に続いて 2 回目である**（Iter103 の失敗 4 行については，journal に原因の記録が無い）．今回は 7 件が 2 つの時間帯に固まっていた．`mise run analyze` は app のログしか集めないので，ollama のログが残らず，原因を追えなかった．analyze で ollama のログも集める提案を B195 に残した．
5. **levers は使い切った**．Iter106 は `current_lever=null` として調査・計画フェーズから始める（backlog B195．tavily-search で重点調査する）．
6. **一時スクリプトは，結果を記録した時点で `.claude/research/` へ写す**．`/tmp` は再起動で消えうる．今回は 3 本を分析の時点で写した．G0 や集計のスクリプトは，実行フェーズのうちに写しておくと，記録の再現性が切れない．

