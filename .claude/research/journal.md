## Iteration 108: 信頼学習による訓練行の除去を事前 CV で足切りする

このブロックは，rc-researcher の報告をもとにオーケストレータが挿入した（rc-researcher は 3 回の委譲のどれでも，記録の前にツールを使えなくなった．起動モードは tenbin）．

### 調査・計画

#### 現状（Iter107 までの採用構成）

- 採用 artifact は Iter105（MD5 `c7172ad37c10e1082a42481553ae25b0`，k=2，λ=0.3，T=0.9426）に Iter107 の報告用フィールドを加えたもの．本走 `results/20261004_225553`: 共通行 top1 0.849626，ECE（旧）0.037281，複合行 top1 0.805479，set_recall 0.573288，mean_dispatched 1.980822．

#### 誤りの内訳（新の定義，251 行）

- 送出数別: k=1 が 233 行，k=4 が 13 行，k=2 が 5 行．k=1 の誤りの確信度は中央値 0.747 で，0.8 以上が 83 行ある．
- 混同の組: education→business_economics 26，education→history_culture 14，education→medical 10，education→legal 7（education が正解の誤りは少なくとも 71 行で，全体の約 3 割）．natural_science→medical 13，medical→natural_science 9．history_culture と social_science の間で 8 と 6．

#### 過去の試行との重なり

- cleanlab，confident learning，label noise，datastore は backlog，journal_archive，config に記述が無い（未着手）．
- 訓練行数を増やす系は Iter95・96 で打ち止め（B161，機序 M5）．OvO・ECOC は Iter97・98 で反証（機序 M6: 誤りの原因は決定則ではなく入力の行の側にある）．kNN 類似度の前処理は Iter106 で打ち止め．

#### 出典

- Northcutt, Jiang, Chuang, "Confident Learning: Estimating Uncertainty in Dataset Labels"，arXiv:1911.00068，JAIR 2021．
- cleanlab, "Handling Label Errors in Text Classification Datasets"．評価側にもノイズがあると，訓練側を除いても評価上の効果が出ない場合があると述べている．
- Shao ら, "Scaling Retrieval-Based Language Models with a Trillion-Token Datastore"（B2 の根拠）．

#### 候補の比較

| ID | 内容 | 利点 | 欠点・リスク |
|---|---|---|---|
| B1（推奨） | 信頼学習で訓練行のラベル問題を検出し，除去して再訓練する | 10 ドメインに共通で，ドメイン固有の補正にならない．未着手．M6 と整合する | 評価集合も同じタスク由来の曖昧さを持つなら，除去が逆効果になりうる |
| B2 | kNN の datastore だけに，評価集合と素な JMMLU プール行を追加する | 分類器の訓練は変えずに近傍の質を変えられる | education のプール行が 9 行しか無く偏る．B161 との関係の解釈が要る |
| B3 | `dispatch_model_not_ready` への再試行を加える | 実装が小さい | 影響は 2〜6 行で，ノイズの床（top1 正味 ±2 行）と同程度 |

レバーは，データのタスク名の有無（2a）と，config の `classifier_train_label_map_consistency`・`classifier_label_granularity` との重なり（2b）を確かめた後に確定する．

#### 2a: タスク名の有無と，誤りの出どころ

- 訓練ファイル `data/classifier_train_iter94_dedup.jsonl`（2,275 行）は `domain`，`id`，`query` しか持たず，タスク名が無い．id の接頭辞の内訳は，`*-train` が 7 ドメインで各 150 行，history_culture 124 行，education 104 行，legal 77 行，`*-hardneg` が 9 ドメインで各 100 行，`education-civics` が 20 行（Iter92 で追加）．
- Iter93 では JMMLU.zip への逆引きで，訓練行はすべて 56 タスクのどれかに一致し，`_DOMAIN_TASK_MAP` との食い違いも 0 行だった（`journal_archive.md:3974-3977`）．zip のキャッシュは消えているので，行ごとのタスクを数え直すには `build_dataset.py:62-64` の URL から取り直す必要がある．
- 評価集合の単一行 3,020 行は `jmmlu_task` を持つ．本走 `results/20261004_225553` の単一行の誤り（fallback と dispatch_failed を除き，`selected_domain ∉ expected`）は 421 行である．上の 251 行は送出集合で判定した数なので，数え方が異なる．

| タスク（正解ドメイン） | 誤り / 行数 | 誤り率 |
|---|---|---|
| japanese_civics（education） | 61 / 116 | 0.526 |
| miscellaneous（general） | 41 / 69 | 0.594 |
| professional_psychology（medical） | 18 / 38 | 0.474 |
| sociology（education） | 36 / 85 | 0.424 |
| high_school_psychology（education） | 28 / 73 | 0.384 |
| moral_disputes（education） | 23 / 76 | 0.303 |
| philosophy（social_science） | 20 / 77 | 0.260 |

- 主な組は，education → business_economics（japanese_civics）37 行，education → medical（high_school_psychology）16 行，education → history_culture（sociology）16 行，medical → education（professional_psychology）11 行である．
- 解釈（推測）: 誤りは，タスクからドメインへの写像そのものが曖昧なタスクに集まる（心理学が medical と education に，哲学と倫理が social_science と education に分かれている）．訓練ラベルは写像から決定的に付くので，行単位の注釈のノイズは原理的に無い．信頼学習が検出するのは写像の曖昧さになる見込みが高く，評価ラベルも同じ写像で作られているので，除くと評価の定義から離れる危険がある．

#### 2b: 過去の試行との重なり

- `classifier_train_label_map_consistency`（Iter92，B153 で partial）: 訓練の japanese_civics 行を写像に合わせ，recall が 0.0086 → 0.2328 に上がった．正しいラベルの行を「誤り」として除けば，逆向きに動きうる．
- `classifier_label_granularity`（Iter93）: 56 クラスへの細分化は CV の足切り（+2.3pt）に届かずクローズした．B1 はラベルの粒度を変えないので，この案の再試行には当たらない．
- Iter94 で訓練と評価の重複 64 行を除いたものが，いまの訓練ファイルである．
- 結論: 「直せる誤ラベルが残っている」という前提は成り立たない見込みが高い．ただし，「タスクの境界で他のドメインに埋もれる行を除くと境界が整う」という別の機序は否定されておらず，CPU の CV で安く確かめられる．

#### 単一レバー

- `confident_learning_train_row_pruning`: 分類器 artifact の訓練集合（LR と kNN の datastore の両方）から，信頼学習で検出した行を除いて作り直す．
- OOF 確率: いまの構成（LR の temperature 較正，kNN k=2，λ=0.3，T=0.9426，StratifiedKFold 5 fold，seed 104）．`_iter105_replay_temp.py` の `run_outer_fold` と `apply_temperature` を流用する．
- 除く行の規則（10 ドメインに一様）: cleanlab の `find_label_issues(filter_by="prune_by_noise_rate")` を既定の引数で使い，順位づけは `self_confidence` とする．上限は 227 行（10%）．T は 0.9426 に固定する．
- cleanlab は `uv run --with cleanlab` で一時的に使い，依存には入れない．計算は CPU だけで行う．
- B2 と B3 は採らない．B2 は education のプール行が 9 行しか無く，B161 との関係を整理できていない．B3 は効果がノイズの床と同程度である．

#### 仮説

訓練集合のうち，与えられたラベルとの整合が低い行を除くと，ドメイン間の境界が整い，単一行の top1 が上がる．2a と 2b から，この仮説は弱いと見込む．そのため，次の Step 0 で先に確かめる．

#### Step 0: 入れ子 CV による足切り（eval は使わない）

- 外側 fold の訓練部分だけで，内側の OOF と除去を行う．評価は外側の検証部分の全行（除去しない）で行う．
- 進む条件: CV top1 が none を +1.0pt 以上上回り，かつ 5 fold のうち 4 fold 以上で上回ること．届かなければ artifact を変えずにクローズする（Iter93 と同じ型．本走は行わない）．
- 除いた行のドメイン別の内訳と，JMMLU.zip を取り直せればタスク別の内訳も記録する．
- 出力: `.claude/research/_iter108_replay_cl.{py,json}`．

#### 到達条件（rc-researcher の報告．行番号は実行フェーズで照合する）

- `http_server.py:428` → `classifier.py:48` で artifact を読み，`http_server.py:379` → `classifier.py:66-69` → `knn_interpolated_head.py:135-149` の `predict_proba` を呼ぶ．`knn_label_distribution`（120-133 行）が `train_label_indices` と `train_weights` を読む．
- 前提: `routing_method: supervised_classifier`（config.yaml:75），`confidence_signal_method: self_report`，ノードのドメインが `classes_` に含まれること，`dispatch_gap_threshold` が null でないこと．
- 発火の証拠: 新しい artifact の MD5 が `c7172ad37c10e1082a42481553ae25b0` と異なること，`len(train_label_indices) = 2275 − 除いた行数`，deploy 後の全ノードのコンテナ内で MD5 が一致すること．

#### 成功条件（事前登録．config.yml の success_criteria (8)）

- replay: 同じ eval キャッシュ上の「変種 − none」の対応のある比較．ノイズの床は B200 に従い，接近行を除いた部分集合も併記する．
- 主基準（本走）: 基準線 `results/20261004_225553` に対し，共通行で Δtop1 ≥ +0.5pt（約 19 行）かつ McNemar 両側 p < 0.05．
- 非退行: Iter107 の C1〜C7．上の 4 タスクの recall の差を併記する．ECE は新旧を併記し，判定は旧の定義で行う（B203 の A1-1）．

#### 未確認の点

- 到達条件の行番号，B115/B116・B153・B161 の本文，cleanlab の導入とライセンス（AGPL-3.0 と認識しているが未確認），replay と入れ子 CV の結果．
- 足切りに届かない可能性が高い（推測）．その場合，levers は再び使い切りになる．根本の対処が評価ラベルの写像の見直しになるなら，評価集合の定義を変えるので，人間の判断を仰ぐ．

### Iteration 108 実装・実験（2026-10-05）

この節は，rc-executor の報告をもとにオーケストレータが追記した（rc-executor は追記の前にツールを使えなくなった）．

- **到達条件の照合**: `http_server.py:428`，`378-380`，`classifier.py:48`，`66-69`，`knn_interpolated_head.py:120-133`，`135-149` はすべて一致した．artifact の MD5 は `c7172ad37c10e1082a42481553ae25b0` で一致した．datastore（2,275 行）は `data/embcache_train_iter100_fused.npy` と `data/classifier_train_iter94_dedup.jsonl` から作り直したものと一致した．`scripts/build_knn_interpolated_classifier.py` はベース LR（`models/domain_classifier_pre_iter104_lr.joblib`）を再学習しないので，行を除くレバーを本番に入れるには LR の再学習も要る．
- **訂正**: cleanlab 2.9.0 のライセンスは，パッケージのメタデータ上 Apache License 2.0 である．上の計画節，config.yml の lever の note，B204 にある「AGPL-3.0 と認識しているが未確認」は誤りである．また，735KB あるのは config.yml（339,768 バイト）ではなく backlog.md（756,039 バイト）である．
- **Step 0**（入れ子 CV．`.claude/research/_iter108_replay_cl.{py,json}`，ログは `_iter108_step0.log`．gpu2 の CPU で実行し，wafl500〜509 は使っていない）:

| fold | 検出 | 除去 | 上限の適用 | none | 変種 | 差 | 行数換算 |
|---|---|---|---|---|---|---|---|
| 0 | 184 | 182 | あり | 0.835165 | 0.815385 | −0.019780 | −9 |
| 1 | 203 | 182 | あり | 0.819780 | 0.780220 | −0.039560 | −18 |
| 2 | 170 | 170 | なし | 0.841758 | 0.839560 | −0.002198 | −1 |
| 3 | 187 | 182 | あり | 0.826374 | 0.795604 | −0.030769 | −14 |
| 4 | 158 | 158 | なし | 0.810989 | 0.806593 | −0.004396 | −2 |
| 平均 | — | — | — | 0.826813 | 0.807472 | −0.019341 | — |

- 除いた行は 5 fold の合計で 874 行で，内訳は medical 192，education 189，social_science 94，general 91，natural_science 83，business_economics 66，history_culture 54，computer_science 44，legal 38，mathematics 23 である．medical と education で全体の 43% を占め，誤りが集まるタスク写像の曖昧なドメインと重なる．
- **判定**: 進む条件（+1.0pt 以上，かつ 4 fold 以上で上回る）を満たさないので，Step 0 で不通過である．artifact，config.yaml，コードは変えず，replay，deploy，G0，本走は行っていない（success_criteria (8) と Iter93 と同じ扱い）．state.json は `experiment_dir=null` のままである．
- **未確認**: 除いた行のタスク別の内訳は，JMMLU.zip を取り直していないので記録していない．

### Iteration 108 実行済み（2026-10-05）

この節は，rc-evaluator の報告をもとにオーケストレータが追記した（rc-evaluator は最初の書き込みの時点でツールを使えなくなった）．統計値はオーケストレータが fold ごとの差から計算し直して確かめた．

**判定: rejected（Step 0 で不通過．本走しない）**

| 項目 | 値 |
|---|---|
| CV top1（none / pruned） | 0.826813 / 0.807473 |
| 平均差 | −1.934pt（進む条件は +1.0pt 以上） |
| 上回った fold | 0 / 5（進む条件は 4 以上） |
| fold ごとの差の標準偏差 / 平均の標準誤差 | 1.63pt / 0.73pt |
| t(4)，両側 p，平均差の 95% CI | −2.66，≈ 0.056，[−3.95pt, +0.08pt] |

- ノイズとの切り分け: 5 fold すべてで下がった．Iter106 の 5 変種の CV の差（−0.44〜+0.26pt）の幅も超える．改善の向きの信号は無い．p ≈ 0.056 なので，悪化が有意とまでは言わない．
- 仮説との照合: 「整合の低い行を除くと境界が整い，top1 が上がる」は支持されなかった．計画の 2a と 2b の見込み（信頼学習が拾うのは写像の曖昧さである）とは一致する．
- 採用構成は変えない: Iter105 の artifact（MD5 `c7172ad37c10e1082a42481553ae25b0`）に Iter107 の報告用フィールドを加えたもの．基準線は `results/20261004_225553` のままとする．

**学び**:

1. 訓練ラベルがタスクの写像から決定的に付くデータでは，信頼学習の「ラベル問題」は写像の境界にあるタスクの行になる．これを除くと，同じ写像で付けた検証側の行を当てる手がかりが減る．誤りの根は訓練行の注釈ではなく，写像の曖昧さにある見込みが強まった（推定）．
2. 検出した行数が多い fold ほど大きく下がった（検出 158 行で −0.44pt，203 行で −3.96pt．n=5 なので参考値）．上限が掛からなかった fold 2 と 4 でも none を上回っていないので，閾値を緩めても届く見込みは無い．閾値や上限の振り直しは多重性の問題があるので行わない．
3. CPU だけの Step 0 で，GPU と本走の約 2.3 時間を使わずに棄却できた．Iter93 と同じ型の足切りは，訓練データを除くか作り変えるレバーで引き続き有効である．
4. cleanlab 2.9.0 のライセンスは Apache License 2.0 である（実行フェーズで確認）．

**次の一手**: levers は再び使い切りになった．Iter109 は調査・計画フェーズから始める（B205）．評価ラベルの写像を見直す方向は，評価集合の定義を変えるので，人間の判断を仰ぐ（B206）．

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

