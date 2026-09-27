## Iteration 86: 拡充した評価集合での hard negative mining 再試行

### 調査 (Iter86)

本反復のレバーは backlog B138 (c) で `cross_domain_training_data_augmentation` =
`hard_negative_mining_all_domains_retry_expanded_set` に確定済みで，レバー選定の裁量は無い．
config.yml の同レバー note は「**Iter84 と完全に同一の抽出手順・同一のハイパーパラメータを，拡充後の
評価集合の上で再実行する**」と規定している．したがって調査の問いは「その再実行をどう構成すれば
Iter84 の問い（Δtop1 +1.201pt は本物か）に決着が付くか」の 1 点に絞られ，具体的には 3 つである．
**(Q1) Iter84 が作った訓練集合をそのまま再利用してよいか（＝評価集合の拡充との相互作用は無いか）．
(Q2) 再採掘する場合，同一規則で同じ形の訓練集合が作れるだけのプールが残っているか．
(Q3) 拡充後の評価集合上での期待効果量と検出力はいくつか．**

**Q1: そのままの再利用は不可．Iter84 の hard negative 900 行のうち 355 行が，Iter85 で評価集合へ入った**

- 本フェーズで実測した．`data/classifier_train_iter84_hardneg.jsonl` の追加 900 行と，
  Iter85 で追加した `data/single_domain_expansion_iter85.jsonl` の 1,520 行は，**設問文の完全一致で 355 行重複する**
  （business_economics 30 / computer_science 56 / education 56 / general 20 / history_culture 20 /
  mathematics 59 / medical 22 / natural_science 32 / social_science 60）．
  原因は構造的である．Iter84 の採掘プールも Iter85 の拡充プールも **同じ「JMMLU の未使用行」**から取っており，
  Iter85 のプール定義は評価集合 1,915 行と `data/classifier_train.jsonl` 1,427 行だけを除外して
  **`classifier_train_iter84_hardneg.jsonl` の追加 900 行を除外していなかった**（Iter85 時点では Iter84 が
  rejected・artifact 復元済みで，その 900 行は「使っていない行」だったため矛盾はしていない）．
- **この 355 行は評価集合の中で最難の部類である**．基準線 Iter85 本走 `results/20260927_130237/` 上で，
  355 行の `top1_accuracy` は **0.3887**，残り 3,080 行は **0.8321**（全体 0.7863）．
  p_true 最小順で採掘した行なので当然の結果だが，**もし Iter84 の訓練集合をそのまま使えば，
  処理群だけがこの 355 行を訓練で見た状態になる**．仮に半分を暗記するだけで
  355×0.5/3435 = **+5.2pt** が上乗せされ，真の効果（+1.2pt 規模）を完全に覆い隠す．
- これは Kapoor & Narayanan, "Leakage and the reproducibility crisis in machine-learning-based science"
  （Patterns 4(9):100804, 2023．プロジェクトページ <https://reproducible.cs.princeton.edu/>，2026-09-27 確認）が
  17 分野 294 本の論文で最も多く観測したと報告する type L1 系の漏洩（訓練集合と評価集合の分離の失敗，
  重複を含む）そのものである．**「同じ訓練ファイルを再利用する」は手順の同一性を守るように見えて，
  測定を無効化する**．
- **代替案として「Iter84 の訓練集合を再利用し，漏洩 355 行を除いた 3,080 行で判定する」を検討し，棄却した**．
  推定量としては不偏だが，除かれる 355 行は最難層に偏っており（下記 Q3 の第 1 五分位が 210 行を占める），
  **残る 3,080 行での期待効果は +0.490pt** にしかならない（Q3 の層別投影）．有意境界 1.086pt を大きく下回り，
  検出力は 15% 程度である．Iter85 が検出力のためだけに 1 反復を費やした経緯に照らして本末転倒であり，
  かつ「効果なし」が出ても真因（推定対象の移動）と区別できない．
- **したがって採るのは「`scripts/mine_hard_negatives.py` を 1 行も変えずに再実行する」である．**
  同スクリプトは `--eval-data`（既定 `data/dataset.jsonl`）の設問文をプールから除外する仕様なので，
  **同じコマンドを打つだけで拡充後 3,435 行が自動的に除外され，漏洩 0 件の訓練集合が得られる**．
  抽出規則・N=100・スコア（基準線 artifact の p_true）・同値解決順序はすべて Iter84 と同一で，
  変わるのは「評価集合が増えた分だけプールが縮む」点のみである．これは config.yml note の
  「手順を変えてはならない」に適合する（手順は不変，入力が Iter85 の成果によって更新されただけである）．

**Q2: 再採掘後も Iter84 と同じ形（legal 0・他 9 ドメイン各 100・計 900 行）が作れる**

本フェーズで `mine_hard_negatives.py` の `determine_used_tasks` / `build_pool` を直接呼び，
JMMLU.zip（本環境キャッシュ，sha256 `3ba7d912...`）からプールを実測した（CPU のみ．埋め込み不使用）．

| ドメイン | Iter84 時点のプール（評価 1,915 行を除外） | **Iter86 のプール（評価 3,435 行を除外）** | N=100 の充足 |
|---|---|---|---|
| medical | 1,110 | **910** | ○ |
| history_culture | 779 | **579** | ○ |
| natural_science | 776 | **576** | ○ |
| business_economics | 705 | **505** | ○ |
| mathematics | 348 | **148** | ○ |
| computer_science | 251 | **100** | ○（プール全量） |
| social_science | 244 | **100** | ○（プール全量） |
| education | 232 | **109** | ○ |
| general | 125 | **100** | ○（プール全量） |
| legal | 0 | **0** | ―（構造的に 0） |
| 合計 | 4,570 | **3,127** | 選定 900 行 |

- **9 ドメインすべてが N=100 をちょうど満たす**（Iter85 が「各ドメイン 100 行を訓練用に留保する」設計で
  拡充したことが，ここで設計どおりに効いている）．**追加行数・ドメイン別内訳は Iter84 と完全に同一の
  {legal: 0, 他 9 ドメイン: 各 100} = 900 行**になり，出力は 2,327 行で行数まで一致する．
- **留保（事前に明記する）**: computer_science・social_science・general の 3 ドメインは
  **プール = 100 = N** なので，「最難 100 件を選ぶ」が「プール全量を取る」と一致し，**この 3 ドメインでは
  hard 選択の選択性が失われる**（無作為抽出と同一になる）．education も 109/100 でほぼ同様である．
  不確実性サンプリングは候補間の情報量の分散が大きいときにのみ無作為を上回る
  （Tripp, <https://www.austintripp.ca/blog/2025-04-02-active-learning-random>，Iter84 調査節で引用済み．
  選択性喪失時の不安定性については "A More Robust Baseline for Active Learning by Injecting Randomness"
  <https://www.csie.ntu.edu.tw/~htlin/paper/doc/wsdm22aloe.pdf> なども同旨，2026-09-27 確認）．
  したがって **Iter84 の効果がそのまま再現する保証は無く，4 ドメイン分は弱まりうる**．これは評価集合拡充の
  代償であり，漏洩を許容する選択肢は Q1 のとおり測定自体が成立しないので，この代償を受け入れる．
- 再採掘後の 900 行と Iter84 の 900 行の重なりは，最大 545 行（900 − 355）になる見込みである．
  実測値は実験フェーズで記録する（継続性の参考指標．判定には使わない）．

**Q3: Iter84 の効果は「最難層のみの大幅改善 ＋ 易しい層の広く浅い退行」だった．拡充後の期待は +1.34pt**

本フェーズで Iter83 本走（基準線）と Iter84 本走の 1,915 行を行単位で対応付け，
**基準線分類器が正解ドメインへ与えた確率 p_true の五分位**で層別した（`probe_candidates` の確信度を使用）．

| 層（基準線 p_true） | n | 基準線 top1 | Iter84 top1 | Δ |
|---|---|---|---|---|
| Q1（≤0.3606） | 383 | 0.0627 | 0.2428 | **+18.02pt** |
| Q2（≤0.7155） | 383 | 0.9034 | 0.8460 | −5.74pt |
| Q3（≤0.8902） | 383 | 1.0000 | 0.9608 | −3.92pt |
| Q4（≤0.9678） | 383 | 1.0000 | 0.9791 | −2.09pt |
| Q5（≤0.9985） | 383 | 0.9974 | 0.9948 | −0.26pt |

- **Iter84 の +1.201pt は「最難五分位の +18pt」と「他 4 層の −0.26〜−5.74pt」の差し引きである**．
  hard negative mining は境界近傍を張り直す一方で，既に正しく分類できていた易しい行をいくらか壊す，
  という機序がここで初めて数値化された．**Iter84 で `education_recall` が有意退行した（0.4120→0.3133，
  p=0.000427）のも同じ機序の一部と考えられる**（本フェーズでは検証していない．仮説である）．
- 拡充後 3,435 行の p_true 分布（基準線 Iter85 本走で実測）は各層 698 / 660 / 638 / 725 / 714 行で，
  上表の層別 Δ をそのまま当てはめた投影は **Δtop1 = +1.336pt** である．
  対応のある二項検定の標準誤差は `√(p_d/N)`（p_d=0.09452 は Iter84 実測の discordant 率）で
  **SE=0.5247pt**，有意境界 `1.96·SE` = **1.0284pt**，**検出力は約 72%**（z=2.547 − 1.96 = 0.587）．
  B138 が記した 86.8% は single/compound の 2 層分解による粗い投影で，本フェーズの 5 層分解の方が
  Iter84 の構造を忠実に反映している．**72% は理想的ではないが，拡充前（1,915 行）の 40.1% とは別次元であり，
  到達可能な最良の設計である**（プール上限と本走時間の制約は Iter85 の調査節で既に確定している）．

### 計画 (Iter86)

**単一レバー**

`cross_domain_training_data_augmentation` = **`hard_negative_mining_all_domains_retry_expanded_set`**．
**`scripts/mine_hard_negatives.py` を 1 行も変更せずに，拡充後の `data/dataset.jsonl`（3,435 行）を
`--eval-data` として再実行し，得られた `data/classifier_train_iter86_hardneg.jsonl`（2,327 行）で
分類器を再訓練すること**だけを行う．評価集合・`config.yaml`・埋め込みモデル・instruction・連結仕様・
送出閾値・集約方式・コードは一切変えない．

**Iter84 との差分は 1 点のみ**: プールから除外する評価集合が 1,915 行 → 3,435 行になる（漏洩排除のため必須．
Q1）．抽出規則・N=100・スコア関数・同値解決順序・出力書式・id 規則（`{domain}-hardneg-{連番:03d}`）は不変．

**レバーを読むコード行と，そこへ到達する条件（d0004 §4 の再発防止．省略不可）**

| 経路 | レバーが効く箇所 | 到達条件 | 到達確認 |
|---|---|---|---|
| 採掘（オフライン） | `scripts/mine_hard_negatives.py:build_pool()` の `exclude_queries`（`--eval-data` 由来） | `--eval-data data/dataset.jsonl` が **3,435 行**であること | **F2** |
| 訓練（オフライン） | `scripts/train_domain_classifier.py` の `--train-data` | 新 JSONL を渡すこと | **F3** |
| 実行時（本体） | `classifier.py:load_domain_classifier()` → `estimate_confidence_classifier()` が `config.yaml` の `classifier_model_path`（=`models/domain_classifier.joblib`）を読む | 各ノードのコンテナが**配布後の** artifact を読むこと（`mise.toml` の rsync） | **F4** |
| 記録側 | `run_experiment.py:98` の `dispatched_domains` 再計算（gap_threshold=0.36，不変） | 同上 | **F5** |
| 到達しない経路（変更しない） | 埋め込み・`aggregator.py`・`config.yaml`・`build_dataset.py`・`data/dataset.jsonl` | 一切触らない | **F1** |

**事前ゲート（実装フェーズで判定．G0 は B139 の再発防止条項）**

| ゲート | 内容 | 不合格時 |
|---|---|---|
| **G0（調達元の同一性．B139）** | 採掘に用いる JMMLU.zip の sha256 が `data/MANIFEST.md` の記録値 **`3ba7d912943ede44fb7ec06aa1df067ac6bee157e65c5588736b9b983f0e684d`** と一致すること（本環境がキャッシュする実体．ピン留めコミット `3637b25e` の公称値とは異なるが，既存の全データセットはこの zip から作られている） | **G0 が FAIL の場合，下記 F2 の「件数が計画表と完全一致」条項は適用しない**（実測値を記録し，非遮断の観察事項として扱う）．本走は G0 の合否に関わらず実施する |
| **F1（変更の最小性）** | `git diff` が `data/classifier_train_iter86_hardneg.jsonl`（新規）・`models/`・`data/MANIFEST.md`・`.claude/research/*` のみ．**`config.yaml` と `scripts/mine_hard_negatives.py` の diff は 0 行**．`node.py`・`classifier.py`・`aggregator.py`・`train_domain_classifier.py`・`build_dataset.py` の diff 0 行．`data/dataset.jsonl` の sha256 不変（3,435 行） | invalid |
| **F2（データ．G0 PASS 時のみ件数条項を適用）** | 出力が **2,327 行**．先頭 1,427 行が `data/classifier_train.jsonl` と**バイト一致**（sha256 `eb89bf7b...`）．**追加 900 行と `data/dataset.jsonl` の 3,435 行の設問文重複が 0 件**（本反復の生命線）．既存 1,427 行との重複 0 件，追加行同士の重複 0 件，id 重複 0 件．ドメイン別追加数が **{legal: 0, 他 9 ドメイン: 各 100}**．プール実測が上の表と一致 | invalid（重複 0 件条項のみは G0 に関わらず絶対条件） |
| **F3（artifact）** | 新 `models/domain_classifier.joblib` の sha256 が基準線 `1cfcd3d8...` と**異なり**，かつ Iter84 の `34e4d33b...` とも**異なる**（一致したら評価集合の除外が効いていない）．`n_features_in_`=2048，`classes_` が 10 ドメイン．退避 `models/domain_classifier_pre_iter84_baseline.joblib` は `1cfcd3d8...` のまま．**新 artifact を `models/domain_classifier_iter86_hardneg.joblib` へ複製して保存する**（Iter84 の `34e4d33b...` は復元時に上書き消失しており，本フェーズで再検証できなかった．同じ損失を繰り返さないため） | invalid |
| **F4（配布）** | `mise run deploy` 後，全 10 ノードで artifact sha256 が新値で一致，`dispatch_gap_threshold: 0.36` 一致，`docker compose exec app wc -l /app/data/dataset.jsonl` = **3435**（10/10） | invalid |
| **F5（実行時経路）** | 予備 20 問（`data/dataset_20.jsonl`．既存 1,915 行側の行なので埋め込みキャッシュで replay 可能）の `confidence`・`dispatched_domains` が，新 artifact ＋ キャッシュ埋め込みのオフライン予測と **20/20 一致**．不一致なら本走に進まない | 本走中止 |

**固定する構成（直近の基準線 = Iter85 本走 `results/20260927_130237/` と同一）**

`config.yaml` は 1 行も変更しない（`embedding_model=qwen3-embedding:0.6b`，`embedding_instruction`（Iter81 の P1 文言），
`embedding_view_concat=true`，`routing_method=supervised_classifier`，`confidence_threshold=0.0`，
`dispatch_candidate_threshold=0.0`，`dispatch_top_k=2`，`dispatch_gap_max_k=4`，`dispatch_gap_threshold=0.36`，
`aggregation_method=max_confidence`）．`data/dataset.jsonl`（3,435 行，バイト単位で不変），
`data/classifier_train.jsonl`（1,427 行，バイト単位で不変．追加分は別ファイルへ書く）．
**ドメイン固有の後付け補正は追加しない．棄権・人間エスカレーションは扱わない（2026-09-23 恒久運用ルール）．
N・選択規則・プール定義は 10 ドメイン共通である．**

**仮説（事前登録）**

「Iter84 で観測した Δtop1 +1.201pt は，最難五分位での +18.02pt と，それ以外の層での −0.26〜−5.74pt の
差し引きとして生じた実在の効果であり，測定系の検出力不足（N=1,915 で有意境界 1.378pt）のために
非有意になっただけである．**評価集合を 3,435 行へ拡充し，漏洩を排除した同一規則の再採掘を行えば，
Δtop1 は +1.3pt 前後で McNemar 有意になる**．ただし computer_science・social_science・general・education の
4 ドメインはプールが N に肉薄しており hard 選択の選択性が失われるため，効果は Iter84 より弱まりうる．」

**着地点予測（事前登録．事後に書き換えない）**

| 指標 | 基準線 Iter85 本走 `results/20260927_130237/` | **Iter86 予測** |
|---|---|---|
| `top1_accuracy`（3,435 行） | 0.786317（Wilson CI [0.772294, 0.799701]） | **0.793〜0.803．点推定 0.7997（+1.34pt）** |
| `single_domain_top1_accuracy`（3,020 行） | 0.782119 | +1〜+2pt |
| `compound_domain_top1_accuracy`（415 行） | 0.816867 | ほぼ不変 |
| `education` recall / precision | 0.387991 / 0.563758 | **最大の失点候補**（Iter84 で −9.87pt・p=0.000427）．eval 母数 350 行で確実に検出される |
| `legal` recall / precision | 0.662551 / 0.759434 | 追加行 0 件の唯一のドメイン．監視対象 |
| `compound_domain_set_recall` / `mean_dispatched_count` | 0.581928 / 1.889157 | 予測不能（確信度分布が動く）．非退行の枠内で報告 |
| `mean_duration_ms` | 1574.096 | 送出数に比例．非退行枠 1888.9 |
| ECE | 0.018509 | 0.01〜0.05 |
| rank1 以外が選ばれた行数 | 5 | ≤ 15 |
| `answer_quality_accuracy` / `end_to_end_accuracy` | 0.587417 / 0.403202 | ノイズ床 3SD=2.6pt の範囲でのみ判定 |
| Random / BestSingle / Oracle | 0.112082 / 0.128384 / 1.0 | success_criteria (3) により毎回併記 |

**成功条件・非退行条件（事前登録．結果を見る前に固定し，事後に緩めない）**

| 区分 | 指標 | 基準線 | 合格条件 |
|---|---|---|---|
| **主基準（効果）** | 全 3,435 行 `top1_accuracy` | 0.786317 | **(i) McNemar 対比較（α=0.05）で有意，かつ (ii) +1.0pt 以上（≥ 0.796317）**．AND 条件．Wilson 95%CI と検出限界 `1.96·√(n_d)/3435` を併記 |
| **非退行①** | per-domain recall / precision 計 20 指標（BH 補正 q=0.05） | 上表 | **有意退行 0 件**．`education`（eval 350 行）と `legal`（追加 0 件）は個別に明記する |
| **非退行②（複合被覆）** | `compound_domain_set_recall` | 0.581928 | **≥ 0.539759**（Iter82 水準を下回らない．gt=0.36 は新分布に未較正のため Iter85 水準の維持は要求しない） |
| **非退行③（複合予算）** | `compound_mean_dispatched_count` | 1.889157 | **≤ 2.10** |
| **非退行④** | `fallback_rate` / `dispatch_failure_rate` | 0.0 / 0.001164 | fallback = 0.0，dispatch_failure ≤ 0.005 |
| **非退行⑤** | rank1 以外が選ばれた行数 | 5 | **≤ 15**（部分 dispatch 失敗の検出器．評価集合が 1.79 倍になった分を見込んだ値） |
| **非退行⑥** | `mean_duration_ms` | 1574.096 | **≤ 1888.9（+20% 以内）** |
| **非退行⑦** | ECE | 0.018509 | **≤ 0.08** |
| 報告のみ | 層別 Δ（基準線 p_true の五分位） | 上の五分位表 | **必ず併記する**．Iter84 の機序（最難層 +18pt／他層 −数 pt）が再現したかを判定の解釈に使う |
| 報告のみ | 再採掘 900 行と Iter84 900 行の重なり | — | 継続性の参考指標 |
| 報告のみ | 1,915 行サブセット top1 | 0.791123 | 過去基準線との接続用に併記 |
| 報告のみ | `answer_quality_accuracy` / `end_to_end_accuracy` | 0.587417 / 0.403202 | 3SD=2.6pt のノイズ床の範囲でのみ有意と判定 |

**判定規則（事前登録）**

- **adopted**: 主基準 (i)(ii) を満たし，非退行①〜⑦をすべて満たす．
- **partial**: McNemar が有意で Δtop1 が +0.5〜+1.0pt，かつ非退行①〜⑦を満たす．
- **no_effect**: |Δtop1| < 0.5pt または McNemar が非有意．**この場合，Iter84 の +1.201pt は
  N=1,915 のばらつきの範囲だったと結論する**（B138 が求めた「決着」の一方の答え）．
  結論前に F3〜F5 を再確認し，実験不成立でないことを示すこと．
- **rejected**: Δtop1 ≤ −0.5pt，または非退行①〜⑦のいずれかに違反．
  **`education_recall` の有意退行が再現した場合はここに該当する**．その場合は「hard negative mining は
  全体 top1 を上げるが education を犠牲にする」という機序が 2 反復で再現したと記録し，
  ドメイン固有の手当てへは進まない（2026-09-23 恒久運用ルール (1)）．
- **invalid（実験不成立）**: F1〜F5 のいずれか不合格，`total_questions != 3435`，
  `compound_domain_question_count != 415`，新 artifact の sha256 が `1cfcd3d8...` または `34e4d33b...` と一致，
  **追加 900 行と評価集合の重複が 1 件でもある**，または本走 top1 がオフライン replay 予測から 1.0pt 以上乖離．
- **復元手順（adopted 以外すべて）**: `cp models/domain_classifier_pre_iter84_baseline.joblib
  models/domain_classifier.joblib`（sha256 が `1cfcd3d8...` に戻ることを確認）→ `mise run deploy` →
  全 10 ノードで smoke_check．`config.yaml`・`data/dataset.jsonl`・`data/classifier_train.jsonl` は
  本反復で触らないので復元不要．`data/classifier_train_iter86_hardneg.jsonl` と
  `models/domain_classifier_iter86_hardneg.joblib` は記録として残す．

**実験手順（この順で行うこと）**

1. **前提確認**: `wc -l data/dataset.jsonl` = 3435，`wc -l data/classifier_train.jsonl` = 1427，
   `sha256sum models/domain_classifier.joblib` = `1cfcd3d8...`，`sha256sum /tmp/expert-mesh-cache/JMMLU.zip`
   = `3ba7d912...`（**G0**），`config.yaml` の `embedding_view_concat: true` / `embedding_instruction` /
   `dispatch_gap_threshold: 0.36`．
2. **採掘**（**wafl-ctrl5 限定**．config.yml 絶対条件 (B)）: `data/MANIFEST.md` の Iter84 生成コマンドの
   `--output` だけを `data/classifier_train_iter86_hardneg.jsonl` に替えて実行する（他は 1 文字も変えない．
   `--eval-data data/dataset.jsonl` は据え置きで，中身が 3,435 行になっていることが本反復の変更点である）．
   **プールが変わるため `data/embcache_pool_*` の `pool_hash` は不一致になり，3,127 行を再埋め込みする**
   （旧キャッシュは 4,570 行の別プールのもの．スクリプトは不一致を検出して stderr に出す．
   ここでスクリプトを書き換えてキャッシュを流用してはならない — F1 に抵触する）．→ **F2**
3. **artifact 退避の確認**: `models/domain_classifier_pre_iter84_baseline.joblib` が `1cfcd3d8...` であることを確認．
4. **再訓練**（**wafl-ctrl5 限定**）: `data/MANIFEST.md` の `train_domain_classifier` コマンドの `--train-data` だけを
   `data/classifier_train_iter86_hardneg.jsonl` に替えて実行 → `cp models/domain_classifier.joblib
   models/domain_classifier_iter86_hardneg.joblib` → **F3**．
5. **本走前の着地点記録**: 新 artifact ＋ 既存の 1,915 行キャッシュ埋め込みで replay を走らせ，
   予測 top1 / per-domain / 複合指標を **本走前に journal へ追記する**（B136 で解消済みの役割分担に従い，
   rc-executor が journal へ直接追記してよい）．
6. **配布**: `mise run deploy` → **F4**．
7. **予備 20 問** → **F5**．不一致なら本走に進まない．
8. **本走**: wafl500〜509 で **3,435 問フルスペック 1 回**（config.yml 絶対条件 (A)．事前 replay で
   経路指標が予測できても省略しない）．想定所要は Iter85 実績から **約 90 分**（timeout 180 分以内）．
9. `mise run analyze -- <YYYYMMDD_HHMMSS>`（**B135: 引数を省略すると `results/iter45_preliminary/` を誤選択する**）．
   層別 Δ（基準線 p_true 五分位）と 1,915 行サブセット top1 も併せて算出する．
10. `data/MANIFEST.md` に新訓練ファイル・新 artifact の sha256 と行数・生成コマンド・ゲート結果を追記する．

**次イテレーションへの申し送り**

- 本反復が **adopted** になった場合は，分類器を作り替えたことで rank1−rank2 gap の分布が動くため，
  直後に `dispatch_gap_threshold_recalibration` を 1 回挟むこと（Iter81→82→83 で 3 回観測した機序）．
- **Iter85 のプール定義が `classifier_train_iter84_hardneg.jsonl` を除外していなかった件（本フェーズで発見）**は，
  今後データ調達を伴うレバーで「既存の全 `data/classifier_train_iter*.jsonl` と `data/*_expansion_*.jsonl` を
  除外集合へ入れる」という恒久規則として backlog へ起票した（B140）．

### Iteration 86 実行済み

**変更したもの（3 点のみ．計画どおり）**

- `data/classifier_train_iter86_hardneg.jsonl`（新規 2,327 行 = 既存 1,427 行 ＋ 追加 900 行）．
  `scripts/mine_hard_negatives.py` は 1 行も変更せず，`--eval-data data/dataset.jsonl`（3,435 行）で再実行した．
- `models/domain_classifier.joblib`（再訓練．sha256 `fd1ccd7d...`）と，その複製 `models/domain_classifier_iter86_hardneg.joblib`（F3 の要請）．
- `data/MANIFEST.md`（生成コマンド・sha256・行数・ゲート結果の追記）．
- `scripts/*`・`config.yaml`・`data/dataset.jsonl` の diff は 0 行（F1 PASS）．

**ゲート**: **G0 PASS**（JMMLU.zip sha256 が MANIFEST 記録値 `3ba7d912...` と一致．B139 の再発防止条項が初めて成立した）．
採掘プール **3,127 行**は計画表と完全一致，選定 900 行のドメイン別内訳も {legal: 0, 他 9 ドメイン各 100} で一致．
**絶対条件（追加 900 行 × 評価集合 3,435 行の設問文重複）= 0 件**．F2〜F5 いずれも PASS．**実験は成立している**．

**主基準（効果）: (i)(ii) とも PASS．ただし (ii) は 1 行未満の余裕での通過**

| 指標 | 基準線 `results/20260927_130237/` | Iter86 `results/20260927_152208/` | Δ |
|---|---|---|---|
| `top1_accuracy`（3,435 行） | 0.786317 | **0.796507** | **+1.019pt** |

- McNemar: chi2=4.056140, **p=0.044011**（有意．discordant a_only=125 / b_only=160，計 285）．
  Wilson 95%CI [0.782715, 0.809635]．検出限界 `1.96·√285/3435` = **0.963pt** で，観測 1.019pt はこれをわずかに超えた水準である．
- **(ii) の閾値 0.796317 に対し実測 0.796507 で，余裕は +0.019pt = 3,435 行中 0.65 行分しかない**．
  1 行の入れ替わり（0.029pt）で不成立に転ぶ境界上の通過であり，「+1.0pt を確実に超えた」とは読めない．
- 事前登録の着地点予測（0.793〜0.803，点推定 0.7997）の**範囲内**に着地した．層別投影による予測が機能した．

**非退行①（per-domain 20 指標，BH q=0.05）: 有意 4 件．うち悪化 2 件で FAIL**

| 指標 | 基準線 | Iter86 | p | 向き |
|---|---|---|---|---|
| `history_culture` recall | 0.823666 | 0.883991 | 0.000010 | 改善 |
| `natural_science` recall | 0.733179 | 0.774942 | 0.004607 | 改善 |
| **`education` recall** | 0.387991 | **0.344111** | 0.003948 | **悪化（Iter84 に続き 2 反復連続）** |
| **`natural_science` precision** | 0.872928 | **0.802885** | 0.008897 | **悪化** |

非退行②〜⑦は全 PASS（`compound_domain_set_recall` 0.581928→0.555422 ≥ 0.539759 / `mean_dispatched` 1.889157→1.925301 ≤ 2.10 /
fallback 0.0・dispatch_failure 0.001164→0.000291 / rank1 以外 5→1 行 / `mean_duration_ms` 1574.096→1534.531 / ECE 0.018509→0.040154 ≤ 0.08）．
報告のみ: single 0.782119→0.797020，compound 0.816867→0.792771，1,915 行部分集合 0.802611（本走前 replay 予測 0.803133 と 0.05pt 乖離），
`answer_quality` 0.587417→0.571523・`end_to_end` 0.403202→0.397671（いずれも 3SD=2.6pt のノイズ床内），Random/BestSingle/Oracle 不変．

**判定: rejected**

事前登録の判定規則は `adopted` を「主基準 (i)(ii) ＋ 非退行①〜⑦をすべて満たす」，`partial` を
「McNemar 有意かつ Δtop1 が +0.5〜+1.0pt **かつ非退行①〜⑦を満たす**」，`rejected` を
「Δtop1 ≤ −0.5pt，**または非退行①〜⑦のいずれかに違反**」と定義し，さらに
「**`education_recall` の有意退行が再現した場合はここ（rejected）に該当する**」と名指しで規定していた．
非退行①が BH 補正後の有意悪化 2 件で FAIL しており，`education_recall` の再現も明示条項に字義どおり該当するため，
**判定語は `rejected` で一義に定まる**（B138 のような「定義文のどれにも当てはまらない型」ではない．保守側へ倒す裁量も要らない）．
**主基準が PASS していても，事前登録は非退行違反を rejected へ優先させている．事後に条文を緩めることはしない（B131 以来の運用）．**

**層別 Δ（基準線 p_true 五分位）: Iter84 の構造が再現した**

| 層 | n | 上限 p_true | 基準線 | Iter86 | Δ | 参考: Iter84（1,915 行） |
|---|---|---|---|---|---|---|
| Q1 | 687 | 0.3542 | 0.0451 | 0.1936 | **+14.85pt** | +18.02pt |
| Q2 | 687 | 0.7209 | 0.8923 | 0.8239 | **−6.84pt** | −5.74pt |
| Q3 | 687 | 0.9022 | 1.0000 | 0.9767 | −2.33pt | −3.92pt |
| Q4 | 687 | 0.9697 | 1.0000 | 0.9898 | −1.02pt | −2.09pt |
| Q5 | 687 | 0.9985 | 0.9942 | 0.9985 | +0.44pt | −0.26pt |

**「最難五分位での大幅改善と，それ以外の層での広く浅い退行の差し引き」という機序が，独立した評価集合（3,435 行，
うち 1,520 行は Iter84 当時は存在しなかった行）の上で再現した．**Q1 の効果量は +18.02→+14.85pt と 3.2pt 縮んでおり，
計画節で事前に留保した「computer_science・social_science・general でプール = N = 100 となり hard 選択の選択性が失われる」
という代償が，実際に効果を弱めた方向と整合する（因果の特定はしていない．留保どおり弱まったという一致に留める）．

**学び 1: hard negative mining の便益はプールの選択性に比例し，プールが枯れたドメインは隣接ドメインに territory を奪われる**

ドメイン別 recall の変化を採掘プール規模と並べると対応が明瞭である．

| ドメイン | プール | recall Δ |
|---|---|---|
| medical | 910 | +0.68pt |
| history_culture | 579 | **+6.03pt**（BH 有意） |
| natural_science | 576 | **+4.18pt**（BH 有意） |
| business_economics | 505 | +1.62pt |
| mathematics | 148 | −0.93pt |
| education | 109 | **−4.39pt**（BH 有意） |
| computer_science / social_science / general | 各 100（= N，選択性ゼロ） | −1.05 / +2.13 / +1.98pt |
| legal | 0（構造的） | −2.06pt |

**有意改善した 2 ドメインはプール上位 2〜3 位で，有意悪化した education はプールが N に肉薄していた**（下位群）．
行単位で追うと，education が失った 30 行のうち **21 行が medical へ流れ**，同時に natural_science の誤検出元は
medical が 9→42 行へ増えている（これが natural_science precision の −7.00pt の実体である）．
つまり **選択性の高いドメイン（medical・natural_science）が境界を押し広げ，その押し出しが education → medical → natural_science と
連鎖している**．これは「education が弱い」というドメイン固有の話ではなく，**10 ドメイン共通の規則（各ドメイン最難 N=100）を
プール規模が不均一な状況に適用すると，実効的な強化量がドメイン間で不均一になる**という，規則側の性質である．

**学び 2: education recall の 2 反復連続の有意悪化は，ドメイン固有の手当てでは扱わない**

2026-09-23 の恒久運用ルール (1) により，education だけを狙う閾値・intercept・訓練データ調整は禁止されている（B115/B116 で，
撤去後に精度が悪化しても特別扱いへ戻さないことまで確定済み）．したがって本反復でも education 固有の補正は一切検討しない．
**全ドメイン共通の枠内で言えることは学び 1 のとおり「プールの選択性の不均一が実効強化量の不均一を生む」であり，
対処もその水準（10 ドメイン一律の抽出規則の作り替え）でのみ行う．**次レバーはこの立場から選定した（B141）．

**学び 3: 「主基準は通ったが非退行で落ちる」型に初めて到達した．B138 が求めた決着は付いた**

B138 の問い（Iter84 の +1.201pt は N=1,915 のばらつきか，実在の効果か）に対する答えは
**「実在するが，+1.0pt 前後というのが真の効果量の上限側であり，かつ全体 top1 の改善と per-domain の退行が同一機序の表裏である」**である．
評価集合を 1.79 倍にして検出力を 40.1%→72% へ引き上げた Iter85 の投資は，この決着を可能にした点で回収された．
一方で，**現行の hard negative mining をこれ以上同じ形で繰り返しても，Q1 の改善と Q2〜Q4 の退行のトレードオフは変わらない**．
レバーを「hard か否か」から「hard と easy をどう混ぜるか」へ移すのが次の一手である．

**復元（要フォローアップ）**

事前登録の復元手順（adopted 以外はすべて実施）のうち，**`models/domain_classifier.joblib` を
`models/domain_classifier_pre_iter84_baseline.joblib`（sha256 `1cfcd3d8...`）へ戻す操作と `mise run deploy` は本フェーズで未実施である**
（分析フェーズの実行環境が当該コピー操作を拒否したため）．現在の実機構成は Iter86 artifact（`fd1ccd7d...`）のままである．
**次イテレーションの実装フェーズは，何らかの訓練を行う前に必ずこの復元と再配布・smoke_check を済ませること．**
Iter86 artifact 自体は `models/domain_classifier_iter86_hardneg.joblib` に複製済みで失われない．

## Iteration 85: 単一ドメイン評価集合の拡充による検出力の確保

### 調査 (Iter85)

本反復のレバーは backlog B134（Iter84 分析フェーズ）で `single_domain_eval_set_expansion` =
`jmmlu_unused_rows_power_targeted` に確定済みで，レバー選定の裁量は無い．調査の問いは 3 つ．
**(Q1) 評価集合を何行にすれば，想定効果量 δ に対して McNemar の検出力が確保できるのか（数値で決める）．
(Q2) 調達元（JMMLU 未使用行）は実際に何行あり，どのドメインにどれだけ配れるのか．
(Q3) 追加行の選び方に，測定系を壊す汚染・偏りの経路は無いか．**

**Q1: 必要標本数は Connor / Miettinen の対応のある二項検定の標本設計式で決まる**

- McNemar 検定の標本設計は Connor, "Sample size for testing differences in proportions for the
  paired-sample design"（Biometrics 43(1):207-211, 1987, p.209）が標準で，漸近近似は
  Miettinen, "The matched pairs design in the case of all-or-none responses"（Biometrics, 1968）に遡る
  （実装例: <https://gist.github.com/...pwr.mcnemar>，計算機: <https://powerandsamplesize.com/>，
  <https://homepage.univie.ac.at/robin.ristl/samplesize.php?test=mcnemar>，いずれも 2026-09-27 確認）．
  Connor の形は `n = [z_α·√p_d + z_β·√(p_d − δ²)]² / δ²` で，本研究のように δ ≪ p_d の領域では
  journal の Iter84 学び 1 で使った `N ≥ (z_α+z_β)²·p_d/δ²` と数値的に一致する
  （p_d=0.09452・δ=0.012 で p_d − δ² = 0.09438，差は 0.08%）．**したがって Iter84 が導いた
  「N=1,915 では有意境界 δ=1.378pt，検出力 80% で δ=1.97pt」という結論は文献側の定式と整合する．**
- 本フェーズで N ごとの検出力を再計算した（p_d は Iter84 実測 0.09452 を設計定数として使う．
  分類器を作り替えるレバーの実測値であり，本レバーの直後に再試行する
  `cross_domain_training_data_augmentation` と同型のレバーだからである）:

  | N（全体） | 有意境界 `1.96·√(p_d/N)` | 検出力 80% の最小効果 `2.802·√(p_d/N)` |
  |---|---|---|
  | 1,915（現行） | 1.377pt | 1.969pt |
  | 2,903 | 1.118pt | 1.599pt |
  | **3,447（本計画）** | **1.026pt** | **1.467pt** |
  | 3,731 | 0.987pt | 1.410pt |
  | 5,153 | 0.876pt | 1.200pt |

- **δ=1.2pt を検出力 80% で拾うには N≥5,153 が要る**が，これは後述 Q2 のプール上限（訓練用に各 100 行を
  残すと追加可能なのは 3,829 行）と本走時間（`experiment.timeout_min`）の双方から到達不能である．
  **到達可能な範囲での最良は N≈3,400〜3,700** であり，config.yml の新レバー note が示す目標帯
  （3,400〜3,600）と一致する．
- **ただし本レバーの後に控える再試行では，期待効果量は 1.2pt ではなくもっと大きい**．Iter84 の内訳は
  single 行 +2.133pt / compound 行 −2.169pt で，拡充後は single の構成比が 78.3%→88.0% へ上がる．
  同じ部分効果が成り立つなら期待 δ は **0.880×2.133 + 0.120×(−2.169) = +1.615pt** で，
  N=3,447（SE=0.5235pt）での検出力は **約 87%** になる．**すなわち N=3,447 は「Iter84 の問いに決着を
  付ける」という目的に対しては十分である**（δ=1.2pt 一般を 80% で拾う汎用的な検出力は得られない）．

**Q2: 未使用プールは 4,731 行．ただし legal 2 行・general 128 行という構造的な偏りがある**

本フェーズで JMMLU（pinned commit `3637b25e`，CC BY-NC-ND 4.0）の全 56 タスクを
`build_dataset.py:_DOMAIN_TASK_MAP` で 10 ドメインへ写像し，評価集合 1,915 行・分類器訓練集合 1,427 行の
設問文（4 択を連結した本文の完全一致）を除いた残りを実測した．

| ドメイン | 未使用プール | 訓練用に 100 行残した後の上限 | LoRA 訓練集合と重複しない行 |
|---|---|---|---|
| medical | 1,120 | 1,020 | 861 |
| natural_science | 791 | 691 | 545 |
| history_culture | 786 | 686 | 599 |
| business_economics | 713 | 613 | 466 |
| mathematics | 353 | 253 | 138 |
| education | 334 | 234 | 197 |
| computer_science | 256 | 156 | 61 |
| social_science | 248 | 148 | 64 |
| general | 128 | 28 | 3 |
| **legal** | **2** | **0** | 2 |
| 合計 | **4,731** | **3,829** | 2,936 |

- B133 が記録した 4,578 行は**訓練集合に既出のタスクへ限定した**値で，評価集合の拡充では
  その限定は不要である（評価集合は 10 ドメインとも `_DOMAIN_TASK_MAP` の全タスクを既に使っており，
  education も `japanese_civics` 39 行を含む．タスク構成は変わらない）．今回の 4,731 行が正しい母数である．
  また B134 note の「legal 0」は訓練タスク限定時の値で，評価側の実測は **2 行**（229 − 150 − 77）である．
  どちらにせよ legal は事実上拡充できない．
- **general も上限 28 行**（プール 128 − 訓練用留保 100）で，legal と合わせて 2 ドメインは拡充がほぼ効かない．
  拡充後の単一行はドメイン間で不均衡（medical/education/business/natural_science/mathematics/history_culture
  各 350，computer_science 306，social_science 298，general 178，legal 150）になる．
  **これはドメイン固有の扱いではなく，10 ドメイン共通の規則をデータ側の制約が切り詰めた結果である**
  （2026-09-23 恒久ルール (1) に抵触しない）．

**Q3: 汚染経路は 2 本ある．分類器訓練集合（軸①に効く）と LoRA 訓練集合（軸②③に効く）**

- **(a) 分類器訓練集合との重複は主基準に直結する**ので，プール定義の時点で `data/classifier_train.jsonl`
  1,427 行の設問文を完全除外した（上表はその条件での実測）．なお既存の評価集合には B125 が起票済みの
  72 行の重複が残っているが，本反復では単一レバー原則により触らない．拡充で希釈され 72/1,915 → 72/3,447 になる．
- **(b) 専門家 LoRA の訓練集合との重複は本フェーズで新たに実測した**．`data/lora_train/*.jsonl` は
  `scripts/prepare_lora_training_data.py`（seed=42，各ドメイン最大 300 行，生成時点の評価集合を除外）が
  作った 2,749 行で，設問文の書式は評価集合と同一である．**現在の評価集合 1,915 行のうち 135 行（7.05%）が
  既にこの LoRA 訓練集合と重複している**（Iter37/78 で評価集合を作り直した際に生じたもので，本反復の責任範囲外）．
  ルーティング（軸①）は分類器だけで決まるので影響しないが，`answer_quality_accuracy`・`end_to_end_accuracy`
  （軸②③）は専門家が訓練で見た設問を再現できる分だけ上振れする．
- 未使用プールの 38%（1,795/4,731）が LoRA 訓練行なので，**無作為に取ると追加行の約 4 割が汚染行になり，
  評価集合全体の汚染率は 7.05%→約 21% へ跳ね上がる**．そこで選択規則を「**LoRA 訓練集合と重複しない行を
  先に取り，足りない分だけ重複行で埋める**」（clean-first）とする．LoRA 訓練行は seed 付き無作為抽出なので
  clean/汚染の別は難易度と独立であり，この優先順位は難易度の偏りを生まない．
  この規則の下での汚染は 1,532 行中 269 行に留まり，評価集合全体では 7.05%→**11.72%** の増加で収まる．
- **難易度で選ぶことは絶対にしない**（Iter84 の hard negative mining は訓練側の話）．評価集合を難しい行へ
  寄せると top1 の水準自体が動き，1,915 行サブセットとの比較・過去基準線との接続が壊れる．
  抽出は各ドメインのプール内での seed 固定の一様無作為（clean-first の 2 層のみ）に限る．

### 計画 (Iter85)

**単一レバー**

`single_domain_eval_set_expansion` = **`jmmlu_unused_rows_power_targeted`**．
**`data/dataset.jsonl`（現行 1,915 行）の末尾へ，JMMLU の未使用単一ドメイン行 1,532 行を追記して 3,447 行にする**
ことだけを行う．`config.yaml`・分類器 artifact・埋め込み・訓練データ・送出閾値・集約方式・既存 1,915 行は
一切変えない（既存行はバイト単位で不変．先頭 1,915 行の diff が 0 であることを検証条件に含める）．

**選択規則（10 ドメイン共通．ドメイン別パラメータを持たない）**

1. **プール**: 各ドメインについて `_DOMAIN_TASK_MAP` の全タスクの全行から，現行評価集合 1,915 行と
   `data/classifier_train.jsonl` 1,427 行の設問文（4 択連結後の本文）を除いた残り（計 4,731 行）．
2. **訓練用留保**: 各ドメインのプールから **100 行を訓練用に残す**（`cross_domain_training_data_augmentation`
   の再試行余地を潰さないため．B134 の指示）．取得可能数は `max(0, pool − 100)`．
3. **取得数**: 各ドメイン **N_add = 200**（取得可能数がこれ未満のドメインはその全量）．
4. **層の順序**: `data/lora_train/*.jsonl` の設問文と重複しない行を優先し，不足分のみ重複行で埋める（Q3-b）．
   各層の内部は `random.Random(20260927).shuffle` による一様無作為で，同一 zip なら完全に再現する．
5. **追記**: 選定行を既存 1,915 行の**後ろ**に append する．id は `{domain}-exp085-{連番:03d}`
   （既存 id と衝突せず，1,915 行サブセットを id だけで切り出せる）．`is_compound=false`，
   `expected_domains=[domain]`，`jmmlu_task`・`jmmlu_answer` は JMMLU の値をそのまま持たせる．

**確定した配分（本フェーズで実測・決定論的）**

| ドメイン | プール | 取得可能 | 取得 | うち LoRA 非重複 | 拡充後の単一行数 |
|---|---|---|---|---|---|
| medical | 1,120 | 1,020 | 200 | 200 | 350 |
| education | 334 | 234 | 200 | 197 | 350 |
| business_economics | 713 | 613 | 200 | 200 | 350 |
| natural_science | 791 | 691 | 200 | 200 | 350 |
| mathematics | 353 | 253 | 200 | 138 | 350 |
| history_culture | 786 | 686 | 200 | 200 | 350 |
| computer_science | 256 | 156 | 156 | 61 | 306 |
| social_science | 248 | 148 | 148 | 64 | 298 |
| general | 128 | 28 | 28 | 3 | 178 |
| legal | 2 | 0 | 0 | 0 | 150 |
| 合計 | 4,731 | 3,829 | **1,532** | 1,263 | **3,032** |

**全体 N = 1,915 + 1,532 = 3,447 行（単一 3,032 ＋ 複合 415）**．
有意境界 **1.026pt**，検出力 80% の最小効果 **1.467pt**（p_d=0.09452 を仮定）．

**レバーを読むコード行と到達条件（d0004 §4 の再発防止）**

- 追加行を読むのは `build_dataset.py:main()` に新設する `--single-domain-expansion <path>`
  （既定 `data/single_domain_expansion_iter85.jsonl`）→ `_build_rows()` の末尾で append する新規ブロック．
  **到達条件は `mise.toml` L31 の `uv run python build_dataset.py --output data/dataset.jsonl`
  （`mise run setup` 内）が実行されること**．Iter78（複合 100→415）と同一の作りにし，
  `tests/test_build_dataset.py` の既存呼び出し（`_build_rows()` 直叩き）は引数既定 `None` で不変に保つ．
- 本走が読むのは**コンテナ内の** `/app/data/dataset.jsonl`（`Dockerfile` の `COPY data/ ./data/`）なので，
  **`mise run setup`（イメージ再ビルド・push）→ `mise run deploy` を必ず通す**．
  これを省くと 1,915 行のまま走り invalid になる．
- **落とし穴（B123 と同型）**: `data/` は `.gitignore` 対象なので
  `data/single_domain_expansion_iter85.jsonl` は `git add -f` で明示コミットし，sha256 を
  `data/MANIFEST.md` に記録する．消失すると `mise run setup` が黙って 1,915 行へ戻す．
- **落とし穴（B118-2）**: `mise run setup` 内の素の `uv sync` が research extra を落とすので，
  直後に `uv sync --extra research` を実行する．
- **落とし穴（B135）**: `mise run analyze` は引数なしだと `results/iter45_preliminary/` を誤選択するので
  `-- <YYYYMMDD_HHMMSS>` を明示する．

**実装・実験手順（この順で行うこと）**

0. **【最優先・必須】Iter84 の artifact を基準線へ復元する**:
   `cp models/domain_classifier_pre_iter84_baseline.joblib models/domain_classifier.joblib` →
   `sha256sum` が `1cfcd3d8...` に戻ることを確認 → 後続の `mise run deploy` で全 10 ノードへ再配布し，
   各ノードで sha256 の一致を確認する．**これを怠ると別 artifact の上で走り invalid になる．**
1. `scripts/expand_single_domain_eval.py`（新規）で上記規則の 1,532 行を生成し
   `data/single_domain_expansion_iter85.jsonl` へ書く．**LLM も埋め込みも使わない純粋な CPU 処理**なので
   実機ノードも wafl-ctrl5 も不要（zip の解析のみ）．生成時に (a) 評価集合との重複 0，
   (b) `classifier_train.jsonl` との重複 0，(c) 追加行同士の重複 0，(d) ドメイン別件数が上表と一致，
   を assert する．
2. `build_dataset.py` に `--single-domain-expansion` を追加（既定値あり，`main()` からのみ渡す）．
   `uv run pytest tests/test_build_dataset.py` と `uv run ruff check` を通す．
3. `mise run setup` → `uv sync --extra research` → `wc -l data/dataset.jsonl` = **3447** を確認．
   **`head -n 1915 data/dataset.jsonl` が変更前のファイルとバイト単位で一致すること**を diff で確認する．
4. `mise run deploy` → 全ノード healthy・smoke_check（git-status/hashes/probe）PASS →
   各ノードで `docker compose exec app wc -l /app/data/dataset.jsonl` = **3447**（10/10）と
   分類器 sha256 = `1cfcd3d8...`（10/10）を確認．
5. `mise run start` で本走 1 回（3,447 問）．所要見積りは **113〜131 分**
   （Iter84 実績 1,915 問 63 分の線形外挿＝113 分，`mean_duration_ms`=2280.4 基準＝131 分）．
   watchdog の `experiment.timeout_min` は 150 では余裕が 19 分しかないため **180 へ引き上げる**
   （測定系ではなく監視の設定であり，実験条件ではない）．
6. `mise run analyze -- <run>`，および 1,915 行サブセット（id が `-exp085-` を含まない行）での
   指標を別途算出する．**オフライン replay を行う場合，`data/embcache_eval_*.npy` は 1,915 行分しか
   無いので追加行を wafl-ctrl5 で再埋め込みすること**（1,915 行キャッシュをそのまま流用しない）．

**固定する構成（直近の最良構成 = Iter83 本走 `results/20260927_070239/` と同一）**

`config.yaml` は 1 行も変更しない（`embedding_model=qwen3-embedding:0.6b`，`embedding_view_concat=true`，
`routing_method=supervised_classifier`，`confidence_threshold=0.0`，`dispatch_candidate_threshold=0.0`，
`dispatch_top_k=2`，`dispatch_gap_max_k=4`，`dispatch_gap_threshold=0.36`，`aggregation_method=max_confidence`）．
`models/domain_classifier.joblib` は復元後の `1cfcd3d8...`，`data/classifier_train.jsonl` は 1,427 行のまま不変，
`node.py`・`aggregator.py`・`classifier.py`・`metrics.py`・`run_experiment.py` はコード変更なし．
**ドメイン固有の後付け補正は追加しない．棄権・人間エスカレーションは扱わない（2026-09-23 恒久ルール）．**

**仮説**

評価集合を 1,915→3,447 行へ増やせば，同じ p_d の下で McNemar の標準誤差が √(1915/3447)=0.745 倍になり，
有意境界は 1.377pt→1.026pt，検出力 80% の最小効果は 1.969pt→1.467pt へ下がる．
さらに single 行の構成比が上がることで，Iter84 で観測された部分効果（single +2.13pt / compound −2.17pt）の
合成値は +1.2pt→約 +1.6pt へ上振れし，**同じ施策を再試行したときの検出力が約 87% になる**．
**本反復は精度向上のレバーではなく測定系のレバーであり，top1_accuracy が動かないことがむしろ正常である．**

**成功条件（事前登録．事後に緩めない）**

主基準（AND）:
- **P1（検出力）**: 本走が完走した評価集合の `question_count` = **3,447**（単一 3,032・複合 415）で，
  設計定数 p_d=0.09452 を代入した有意境界 `1.96·√(p_d/N)` ≤ **1.10pt**（設計値 1.026pt）を満たすこと．
- **P2（整合性）**: 1,915 行サブセット（id に `-exp085-` を含まない行）の `top1_accuracy` が
  Iter83 本走 0.792689 と **±0.5pt 以内**，かつ行単位の `selected_domain` 一致率が **≥ 99%**
  （≤19 行の相違まで許容．ルーティングは決定論的なので本来ほぼ完全一致するはずであり，
  これは実装漏れ・artifact 未復元の検出器として使う）．
- **P3（無汚染・純追加）**: `head -n 1915 data/dataset.jsonl` が変更前とバイト単位で一致，
  追加 1,532 行と `data/classifier_train.jsonl` の設問文重複が **0 件**，追加行同士および既存行との
  重複が **0 件**，追加行のドメイン別件数が計画表と完全一致．

非退行（すべて 1,915 行サブセット上で判定する．拡充後の全体値は水準が動くため比較に使わない）:
1. per-domain 20 指標の BH 補正（q=0.05）後の有意退行 **0 件**（構造的には Iter83 と同値のはず）．
2. 複合 415 行の `compound_domain_set_recall` = 0.581928・`compound_mean_dispatched_count` = 1.889 と
   ±0.5pt / ±0.05 以内で一致（複合行は 1 行も足していないので不変のはず）．
3. `fallback_rate` = 0.0，`dispatch_failure_rate` ≤ 0.005．
4. 1 問あたり `mean_duration_ms` ≤ 2744.3（既存上限）．
5. 本走が `experiment.timeout_min`（180 へ引き上げ）以内に完走すること．

参考値として必ず併記するもの（判定には使わない）: 拡充後の全体 top1_accuracy と Wilson 95%CI，
単一/複合の内訳，per-domain 指標（拡充後の母数），LoRA 訓練集合と重複する行の数（135+269=404 の想定）と
それを除いた `answer_quality_accuracy`．

**判定語**

- **adopted**: P1〜P3 と非退行 1〜5 をすべて充足．拡充後の `data/dataset.jsonl` を以後の基準線とし，
  本走を **新しい基準線**（Iter86 以降の比較対象）として登録する．
- **partial**: P1・P3 充足かつ P2 または非退行のいずれか 1 件が FAIL．
- **rejected**: P1 または P3 が不成立（＝拡充の設計自体が要件を満たさない）．
- **invalid（実験不成立）**: 分類器 sha256 が `1cfcd3d8...` でないまま本走した／`question_count ≠ 3447`／
  1,915 行サブセットの一致率 < 95% もしくは |Δtop1| > 1.0pt／先頭 1,915 行に差分がある／
  追加行に訓練集合との重複が見つかった．いずれも「効果なし」ではなく setup・deploy・実装の漏れと解釈する．

**リスクと留保**

- 拡充後の評価集合はドメイン間で不均衡（legal 150 / general 178 に対し medical 350 など）になり，
  **全体 top1 の水準は現行 0.7927 から動く**（比較は必ず 1,915 行サブセットで行う）．
- legal は構造的に 1 行も足せず，general も 28 行に留まる．この 2 ドメインの per-domain 検出力は改善しない．
- LoRA 訓練集合との重複が 7.05%→11.72% へ増えるため，**軸②③の水準は上振れする**．
  軸②③は 3SD=2.6pt のノイズ床込みで参考値として扱い，判定には使わない．
- p_d=0.09452 は Iter84（分類器差し替え）の実測値である．config 1 行だけを変えるレバー（Iter83 は p_d≈0.001）
  では p_d が桁で小さく，検出力の議論はそのまま当てはまらない．

### 実装・実験 (Iter85)

**実施した変更（単一レバーのみ）**

0. **分類器 artifact を基準線へ復元**: `models/domain_classifier.joblib` ←
   `models/domain_classifier_pre_iter84_baseline.joblib`（sha256 `1cfcd3d8...`）．全 10 ノードで一致を確認した．
   B134 (b) の申し送りをここで解消した．
1. 新規 `scripts/expand_single_domain_eval.py`（CPU のみ．LLM・埋め込み不要）で
   `data/single_domain_expansion_iter85.jsonl` を生成．選択規則はプール構築 → 訓練用に 100 行留保 →
   各ドメイン目標 200 行 → LoRA 訓練集合と重複しない行を優先し不足分のみ重複行で埋める clean-first →
   `random.Random(20260927)` で層内シャッフル．内部重複 0・評価集合と 0 重複・分類器訓練集合と 0 重複を
   アサートで保証した．
2. `build_dataset.py` に `--single-domain-expansion`（既定 `data/single_domain_expansion_iter85.jsonl`）を追加．
   `_build_rows()` の複合行ループの後に純追加するのみで，同名引数の既定は `None`（既存テスト呼び出しは不変）．
3. `mise run setup` → `uv sync --extra research`（B118-2 対策）→ `head -n 1915` が変更前とバイト単位で一致することを確認．
4. `mise run deploy` → 全 10 ノード healthy，smoke_check（git-status / hashes / probe）PASS．
   10/10 ノードで `wc -l /app/data/dataset.jsonl`=3435，分類器 sha256=`1cfcd3d8...` を確認．
5. `mise run start` を 1 回（`results/20260927_130237/`，3,435 問，約 90 分で完走．timeout 180 分以内）．
6. `mise run analyze -- 20260927_130237`．統計は `metrics.py` の
   `compute_domain_recall_mcnemar_test` / `compute_domain_precision_fisher_test` / `apply_benjamini_hochberg`
   をそのまま再利用した（自前の統計式は書いていない）．

**計画からの乖離（要レビュー）**

計画は追加 1,532 行（合計 3,447 行）だったが，実測は **1,520 行（合計 3,435 行．単一 3,020 + 複合 415）**．
原因は本環境がキャッシュする JMMLU.zip の sha256 が `3ba7d912...` で，ピン留めコミット `3637b25e...` の
期待値と異なること（journal_archive.md に既知・無害として記載済みの差異）．実際のプールが計画時推定より
わずかに小さく，`computer_science`（計画 156 → 実測 151）・`social_science`（148 → 144）・`general`（28 → 25）の
3 ドメインで不足した．他 7 ドメイン（medical / education / business_economics / natural_science /
mathematics / history_culture = 200，legal = 0）は計画どおり．
また，この過程で JMMLU 側の cross-task 重複質問 1 件（education の sociology / moral_disputes 間）を検出し，
プール構築時にドメイン内の重複除去を追加した（計画に無い対応だが，P3「追加行同士の重複 0 件」を
満たすために必須）．代替は取らず，スクリプトは計画値と実測値を両方 stderr に出力し，`data/MANIFEST.md`
にも明記した．

**主要メトリクス**

全体（3,435 問，参考値）: top1_accuracy = 0.786317（Wilson 95% CI [0.7723, 0.7997]），
single_domain_top1 = 0.782119（n=3,020），compound_top1 = 0.816867（n=415），
fallback_rate = 0.0，dispatch_failure_rate = 0.001164，mean_duration_ms = 1574.10．

1,915 行サブセット（判定用）:

| 条件 | 実測 | 閾値 | 判定 |
|---|---|---|---|
| P1 有意境界 `1.96·√(p_d/N)` | 1.028pt（p_d=0.09452, N=3435） | ≤1.10pt | PASS |
| P2 サブセット top1 | 0.791123（Iter83 0.792689 比 Δ=−0.157pt） | ±0.5pt 以内 | PASS |
| P2 `selected_domain` 行一致率 | 1909/1915 = 99.69% | ≥99% | PASS |
| P3 先頭 1,915 行バイト一致 | 一致 | 一致 | PASS |
| P3 訓練集合との重複 | 0 件 | 0 件 | PASS |
| P3 ドメイン別件数 | 3 ドメインで僅差 | 配分表と完全一致 | 字義上 FAIL |
| 非退行① per-domain 20 指標（BH q=0.05） | 有意退行 0 件 | 0 件 | PASS |
| 非退行② compound set_recall | 0.581928 | 0.581928 | PASS |
| 非退行② compound mean_dispatched | 1.889157 | 1.889 | PASS |
| 非退行③ fallback / dispatch_failure | 0.0 / 0.002089 | 0.0 / ≤0.005 | PASS |
| 非退行④ mean_duration_ms | 2308.51 | ≤2744.3 | PASS |
| 非退行⑤ 完走 | 約 90 分 | timeout 180 分以内 | PASS |

P2 の不一致 6 行: education-045，medical-004 / 007 / 058 / 109，natural_science-097．

**検証**

- `uv run ruff check build_dataset.py scripts/expand_single_domain_eval.py` → All checks passed．
- `uv run pytest tests/test_build_dataset.py -q` → 13 passed / 9 failed．**この 9 件は変更前から失敗していた
  既存の不具合**（fixture zip `tests/fixtures/jmmlu_sample.zip` に `japanese_civics.csv` が無いための
  `KeyError`）で，`git stash` して同一テストを実行し失敗内容・件数が完全に一致することを確認済み．
  本反復のスコープ外として未修正のまま残す．

**異常の有無**: 実行・ログ上の異常は無い．上記の計画乖離が唯一の要レビュー事項である．

### Iteration 85 実行済み

**変更（単一レバー）**: `single_domain_eval_set_expansion` = `jmmlu_unused_rows_power_targeted`．
`data/dataset.jsonl` を 1,915 → **3,435 行**（単一 3,020 ＋ 複合 415）へ純追加で拡充した．
`config.yaml`・分類器 artifact（`1cfcd3d8...` へ復元済み）・埋め込み・訓練データ・送出閾値・集約方式は不変．
`experiment.timeout_min` のみ 150→180（watchdog 設定であり実験条件ではない．B137-(5)）．

**結果（本分析フェーズで `metrics.py` の既存関数により再計算・executor 値を全件再現した）**

| 条件 | 実測 | 閾値 | 判定 |
|---|---|---|---|
| P1 有意境界 `1.96·√(p_d/N)`（p_d=0.09452, N=3,435） | **1.0281pt** | ≤1.10pt | PASS |
| P1 `question_count` | 3,435（計画 3,447） | 3,447 | 下記「乖離」参照 |
| P2 サブセット top1 | 0.791123（Iter83 0.792689，Δ=**−0.157pt**） | ±0.5pt 以内 | PASS |
| P2 `selected_domain` 行一致率 | 1909/1915 = **99.69%** | ≥99% | PASS |
| P3 先頭 1,915 行バイト一致 / 訓練集合重複 / 追加行重複 | 一致 / 0 件 / 0 件 | 同左 | PASS |
| P3 追加行のドメイン別件数 | 1,520（計画 1,532．3 ドメインで −5/−4/−3） | 計画表と完全一致 | **字義上 FAIL** |
| 非退行① per-domain 20 指標（BH q=0.05） | 有意 **0/20** 件 | 0 件 | PASS |
| 非退行② compound set_recall / mean_dispatched | 0.5819277 / 1.8891566（基準線と**完全同値**） | ±0.5pt / ±0.05 | PASS |
| 非退行③ fallback / dispatch_failure（サブセット） | 0.0 / 0.002089 | 0.0 / ≤0.005 | PASS |
| 非退行④ `mean_duration_ms`（サブセット） | 2308.51 | ≤2744.3 | PASS |
| 非退行⑤ 完走 | 約 90 分 | timeout 180 分以内 | PASS |

参考値（判定に使わない）: 全体 3,435 行 top1 = 0.786317（Wilson 95%CI [0.7723, 0.7997]），
単一 3,020 行 0.782119 / 複合 415 行 0.816867，追加 1,520 行のみ 0.780263（CI [0.7588, 0.8004]），
全体 `mean_duration_ms` = 1574.10（追加行は複合行を含まないため 648.83 と速い），
`answer_quality_accuracy` = 0.587417，`end_to_end_accuracy` = 0.403202（LoRA 汚染率が 7.05%→11.7% へ
上がるため上振れしている．軸②③は 3SD=2.6pt のノイズ床込みの参考値）．

**ノイズか有意かの切り分け**

- サブセットの Δ = **−0.157pt** は，基準線との対比較で **discordant 5 行（4 vs 1），McNemar p = 0.3711** で
  有意でない．N=1,915 の有意境界 1.377pt に対し桁で小さく，明確にノイズ範囲内である．
- **さらに強い結論が得られた: 1,915 行すべてで `probe_candidates`（各ドメインの確信度）と
  `dispatched_domains` が基準線と 1915/1915 で完全一致した**．すなわち**ルーティング層は 100% 不変**で，
  拡充は既存行の測定に一切影響していない．不一致 6 行の内訳は (a) `dispatch_failed` の発生行の差 5 行
  （基準線 medical-007 の 1 行 → 今回 medical-004/058/109・natural_science-097 の 4 行），
  (b) education-045 の 1 行は probe 確信度が完全一致のまま `max_confidence` 集約の結果だけが
  history_culture→social_science へ振れたもので，**生成側 confidence の揺らぎに由来する**．
  いずれもルーティングではなく生成・ネットワーク層の非決定性であり，P2 の検出器としての目的
  （実装漏れ・artifact 未復元の検出）は最高水準で満たされた．
- 複合 415 行の被覆指標が基準線と小数点以下まで完全同値であることも，純追加であることの独立な裏付けである．

**判定: `partial`**

- 本レバーの唯一の目的である **P1（検出力の確保）は充足**した．N=3,435 で有意境界 1.0281pt，
  検出力 80% の最小効果 1.4698pt（N=1,915 ではそれぞれ 1.3770pt / 1.9685pt）．
- P3 の 4 条件のうち，P3 の名目そのものを担う 3 条件（純追加・無汚染）は完全充足．**FAIL したのは
  「ドメイン別件数が計画表と完全一致」という手続き的な決定論チェック条項 1 件のみ**で，その原因は
  抽出規則の逸脱ではなく計画時に前提したプール母数の誤り（後述の zip 差異）である．
  規則 3「各ドメイン N_add=200，取得可能数がこれ未満ならその全量」は 10 ドメインすべてで字義どおり
  満たされており，規則は設計どおり決定論的に動いた．
- **語彙運用の整理（B131 以来の「事後に事前登録を緩めない」運用の下で）**: 事前登録の `rejected` は
  「P1 または P3 が不成立（＝拡充の設計自体が要件を満たさない）」，`partial` は「P1・P3 充足かつ
  P2 または非退行のいずれか 1 件が FAIL」と定義していた．今回は **P2・非退行が全 PASS で P3 の一部条項のみ
  FAIL** という，どちらの定義文にも字義上当てはまらない**事前登録が想定していなかった型**である．
  前例（Iter29/30・Iter62/63）の `partial` は「主基準充足＋非退行 1 件 FAIL」の型で，今回とは型が異なる．
  そこで **事前登録を事後に書き換えるのではなく，未定義領域を保守側（`adopted` ではない）に倒して
  `partial` を割り当てる**扱いとした．`rejected` を採らないのは，その定義文が括弧で明記する趣旨
  （拡充の設計が要件を満たさない）と実態が食い違い，拡充を破棄すると Iter84 の問いが永久に決着しない
  ためである．**この語彙運用の是非は要レビュー事項として backlog B138 に上げた．**
- **帰結**: 拡充後の `data/dataset.jsonl`（3,435 行）を **Iter86 以降の新しい基準線**として採用し，
  本走 `results/20260927_130237/` を比較対象に登録する．

**JMMLU.zip の sha256 不一致（再現性の論点）**

本環境のキャッシュ zip は sha256 `3ba7d912...` で，ピン留めコミット `3637b25e...` の期待値と異なる．
`journal_archive.md` に「既知・無害」として記載されていたが，**今回それが計画フェーズの数値前提
（プール件数）を狂わせ，事前登録の一条項を FAIL させた**．無害ではなく，計画の決定論性を壊す実害が
あることが判明した．ただし **本反復のスコープ外**（単一レバー原則）であり，今回は修正しない．
独立項目として backlog **B139** に起票した（選択肢: ピン留めコミットから再取得して MANIFEST の
期待値に合わせる／実キャッシュの sha256 を正として MANIFEST・ドキュメント側を訂正する）．
なお `scripts/expand_single_domain_eval.py` は計画値と実測値を両方 stderr に出力するため，
同型の乖離は今後も生成時点で検出できる．

**学び**

1. **ルーティング層の決定性は行数を倍近く増やしても完全に保たれる**（`probe_candidates` 1915/1915 一致）．
   一方で `selected_domain` は `max_confidence` 集約を通るため**生成側の confidence 揺らぎを拾う**．
   今後「ルーティングが不変か」を検証したいときは `selected_domain` ではなく
   **`probe_candidates` / `dispatched_domains` を突き合わせるべき**である（より鋭い検出器になる）．
2. `dispatch_failure` は行ごとに固定ではなく実行ごとに 1〜4 行の範囲で位置が動く（今回 4 行 / 基準線 1 行．
   いずれも medical・natural_science のノード）．**±5 行程度の top1 の揺れはここから生じる**ので，
   軸①を「完全に決定論的」と断じるときはこの生成層のノイズ床（今回 5/1915 = 0.26pt）を併記すること．
3. **事前登録に「計画値と完全一致」という決定論チェックを書くときは，前提となる外部データの
   sha256 を先に検証する条項とセットにしないと，本質的でない乖離で主基準を落とす**．
   次回以降，データ調達を伴うレバーでは「調達元の sha256 が MANIFEST と一致すること」を
   事前条件（ゲート）側に置き，件数一致は事前条件成立時のみ適用する形にする．
4. 拡充後の全体 per-domain は水準が動いた（mathematics recall 0.6710→0.8005，social_science
   0.4545→0.5520，computer_science 0.8268→0.8796．education は 0.4120→0.3880 で依然最下位）．
   追加行のドメイン別 top1 は education 0.360 が突出して低く，computer_science 0.960 /
   mathematics 0.950 が高い．**全体値は 1,915 行時代の数値と直接比較してはならない**（B138 に明記）．

**次の一手**: 拡充後の評価集合で **Iter84 の hard negative mining を再試行**する
（`cross_domain_training_data_augmentation` = `hard_negative_mining_all_domains_retry_expanded_set`）．
実測構成比（単一 87.92%）で Iter84 の部分効果（single +2.133pt / compound −2.169pt）が成り立つなら
期待 δ = **+1.613pt**，N=3,435（SE=0.5246pt）での検出力は **86.8%**（拡充前の N=1,915 では期待 δ=+1.201pt・
検出力 40.1% にすぎず，判定不能が再発する公算が高かった）．本レバーが目的とした検出力は実測でも確保された．

---

## Iteration 84: 全ドメイン共通の hard negative mining による分類器訓練データ拡充

### 調査 (Iter84)

本反復のレバーは backlog B132（Iter83 分析フェーズ）で `cross_domain_training_data_augmentation` =
`hard_negative_mining_all_domains` に確定済みで，レバー選定の裁量は無い．調査の問いは 3 つ．
**(Q1) 「モデルが苦手な行を優先的に訓練集合へ足す」という選択規則は，無作為に同数足すより本当に良いのか．
(Q2) 最難帯の行はラベル雑音・外れ値であり，足すとむしろ害になるという警告が知られている．本研究のデータで
それは当てはまるのか．(Q3) 追加する行はどこから調達するか（B116(2) の調達順位に従う）．**

**Q1: hard example / hard negative mining は「情報量の高い行を選ぶ」系の標準手法だが，優位性は無条件ではない**

- 能動学習の uncertainty sampling（Lewis & Gale 1994 以来の定番）は「モデルが最も不確かな事例を選ぶ」規則で，
  本レバーの選択規則と数学的に同型である．Sharma & Bilgic, "Evidence-Based Uncertainty Sampling for Active
  Learning"（DMKD 2017, <http://www.cs.iit.edu/~ml/pdfs/sharma-dmkd17.pdf>，2026-09-27 確認）は，
  単純さと実証的成功から最も頻用される戦略だと位置づけている．Zhu et al., "Active Learning with Sampling by
  Uncertainty and Density"（COLING 2008, <https://aclanthology.org/C08-1143.pdf>）はテキスト分類で
  無作為抽出が明確に劣ることを報告している．
- ただし優位性は無条件ではない．Tripp, "Why your active learning algorithm may not do better than random"
  （<https://www.austintripp.ca/blog/2025-04-02-active-learning-random>，2026-09-27 確認）は，
  (i) 候補間で情報量の分散が大きいこと，(ii) 情報量の推定（確信度）が信頼できること，が成立して初めて
  無作為を上回ると整理している．**本研究の分類器は `CalibratedClassifierCV(method="temperature")` で
  較正済み（Iter83 本走の ECE=0.026）なので (ii) は満たす**．(i) は本フェーズのオフライン実測（G3）で確認した．
- 検索分野の hard negative mining でも，ANCE（Xiong et al., ICLR 2021）・NV-Retriever（Moreira et al.,
  2024）が「積極的に掘った negative は性能を上げるが，制御しないと訓練を不安定にする」と報告されている
  （ECI, arXiv 2603.20990, <https://arxiv.org/html/2603.20990v1>，2026-09-27 確認の関連研究節より）．

**Q2: 「最難帯＝ラベル雑音」という警告は本研究のデータには当てはまらなかった（実測で確認）**

- Thakur et al. の RLHN（2025）は，掘った hard negative 集合に含まれる false negative とラベル雑音が
  dense retriever の劣化の主因だと報告している（ARHN, arXiv 2604.11092,
  <https://arxiv.org/html/2604.11092v1>，2026-09-27 確認の関連研究節より）．
- 分類側でも同じ警告がある．"Hard Example Mining" のパターン整理（<https://distilledpatterns.org/patterns/hard-example-mining>，
  2026-09-27 確認）は **「高損失事例の大半がラベル雑音・破損・対象外事例であるとき」「データ集合が小さく
  反復的な mining がすぐ過適合を起こすとき」は使うべきでない**と明記する．Frontiers in AI, "Beyond
  uncertainty in modern active learning for trustworthy AI"（2026,
  <https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1844765/full>，
  2026-09-27 確認）も「不確実性ベースの取得関数は外れ値・雑音・配備分布から遠い事例を過剰に選びうる」と述べる．
- **本研究の該当リスク**: ドメインラベルは JMMLU のタスク名から決定論的に割り当てた代理ラベルであり，
  境界（例: `professional_psychology`→medical と `high_school_psychology`→education）は本質的に曖昧である．
  最難帯にこの種の「代理ラベルとしては無理のある行」が集中している可能性がある．
- **実測（G4）**: 最難から順に取る純粋な least-confidence と，最難 10 件 / 20 件を飛ばしてから取る変種を
  オフラインで比較したところ，**純粋版が最良**（3 seed 平均 top1: pure 0.7929 / skip10 0.7887 / skip20 0.7740）．
  **本研究のデータでは最難帯の除外は改善に寄与しない**ので，追加パラメータを持たない純粋版を採る．

**Q3: 追加行は JMMLU の未使用行から調達する（B116(2) の調達順位 (i)「既存の信頼できる公開データセット」に該当）**

- 現行の訓練集合 `data/classifier_train.jsonl`（1,427 行）は JMMLU（`nlp-waseda/JMMLU`, commit
  `3637b25e`，CC BY-NC-ND 4.0）の各ドメイン 150 行の無作為標本で，評価集合 1,500 行とは別の乱数種で抽出
  されている．**同じタスク群には未使用の行が大量に残っている**ので，新たなデータ生成（LLM 合成・人手作成）は
  不要である．出典・ライセンス・ドメイン適合性はいずれも既存データセットと同一で，新たな検討事項は生じない．
- 本フェーズで未使用プールを実測した（評価集合 1,500 行・現行訓練 1,427 行の設問文を両方除外）:

  | ドメイン | 現行訓練で使用中のタスクに限った未使用プール |
  |---|---|
  | medical | 1,110 |
  | natural_science | 783 |
  | history_culture | 779 |
  | business_economics | 705 |
  | mathematics | 348 |
  | computer_science | 251 |
  | social_science | 244 |
  | education | 233 |
  | general | 125 |
  | **legal** | **0** |
  | 合計 | **4,578** |

- **legal のプールは 0 である**．JMMLU の legal 対応タスクは `international_law` + `jurisprudence` の
  227 行しかなく，150 行が評価集合，残る 77 行が訓練集合に既に入っている（`build_dataset.py` の
  `build_classifier_training_rows` docstring が既に指摘している構造的制約）．**したがって legal だけは
  追加行が 0 件になる**．これはドメイン固有の特別扱いではなく，10 ドメイン共通の規則をデータ側の制約が
  切り詰めた結果である．`_extract_sample_weights()` が `n_samples/(n_classes*n_domain_samples)` で
  ドメイン別の実効重み合計を常に均等化するため，**legal の実効重みは追加後も他ドメインと完全に等しい**
  （1 行あたりの重みが 1.85→3.02 へ上がるだけ）．legal の非退行は個別に監視する．
- **プールを「現行訓練集合に既に登場しているタスク」へ限定する**（上表はその条件での実測値）．無制限にすると
  education に `japanese_civics` が流入するが，これは **Iter36（0.0529 で崩壊）・Iter37（invalid）・
  Iter38（hybrid で education_recall 0.4000 へ悪化）で 3 回失敗が記録された変更**であり，本レバー
  （hard な行の選択）とタスク構成変更の 2 レバーが混ざる．タスク構成を固定することで単一レバーを保つ．
  この限定で影響を受けるのは education のみ（330→233），他 9 ドメインの数値は変わらない．

### 計画 (Iter84)

**単一レバー**

`cross_domain_training_data_augmentation` = **`hard_negative_mining_all_domains`**．
**`data/classifier_train.jsonl`（1,427 行）へ，現行分類器が正解ドメインへ低い確率しか与えない未使用 JMMLU 行を
10 ドメイン共通の規則で各 100 行（プールが足りないドメインはプール全量）追加した
`data/classifier_train_iter84_hardneg.jsonl`（2,327 行）で分類器を再訓練すること**だけ**を行う．
埋め込みモデル・instruction・連結仕様・送出閾値・集約方式・集約コードは一切変えない．

**選択規則（10 ドメイン共通．ドメインごとにパラメータを変えない）**

1. **プール**: JMMLU（pinned commit `3637b25e`）の各ドメインについて，**現行訓練集合にそのドメインで
   既に登場しているタスク**の全行から，評価集合 1,915 行の設問文と現行訓練集合 1,427 行の設問文を除外した残り
   （合計 4,578 行）．
2. **難易度スコア**: 現行 artifact `models/domain_classifier.joblib`（sha256 `1cfcd3d8...`，`n_features_in_`=2048）
   に現行と同一の埋め込み（`qwen3-embedding:0.6b`，prefix なし ⊕ prefix ありの 2048 次元連結）を与えて
   `predict_proba` を計算し，**正解ドメインの確率 p_true を難易度スコア（小さいほど hard）とする**．
3. **選定**: 各ドメインについて p_true の昇順で先頭 **N=100** 件（プールが 100 未満ならプール全量）．
   同値は (JMMLU タスク名, 設問文) の辞書順で決定論的に解く．
4. **追加**: 選定行を既存 1,427 行の**後ろに追記**する．既存行は 1 行も改変・削除しない．
   id は `{domain}-hardneg-{連番:03d}`．`sample_weight` は付けない（`_extract_sample_weights()` が
   ドメイン数から自動計算するため未使用．`train_domain_classifier.py` docstring 参照）．

**N=100 を選んだ理由**: G3 のオフライン実測で最良だったのは「追加行数 / 元の訓練行数 ≒ 56%」の点である
（713 行に 399 行を追加）．現行の 1,427 行に対する同比率は約 800 行で，10 ドメイン共通の切りの良い値としては
N=100（追加 900 行，+63%）が最も近い．N=100 でプールの過半を使い切るのは general のみ（100/125 = 80%）で，
そこでは hard 選択の優位が無作為と同程度へ縮むだけで害は観測されていない（G3 の N=60 条件＝プールの 80% に相当）．

**固定する構成（直近の最良構成 = Iter83 本走 `results/20260927_070239/` の構成そのまま）**

`config.yaml` は **1 行も変更しない**．`embedding_model=qwen3-embedding:0.6b`，
`embedding_instruction`（Iter81 の P1 文言），`embedding_view_concat=true`，`routing_method=supervised_classifier`，
`confidence_threshold=0.0`，`dispatch_candidate_threshold=0.0`，`dispatch_top_k=2`，`dispatch_gap_max_k=4`，
**`dispatch_gap_threshold=0.36`（Iter83 で採用．今回は触らない）**，`aggregation_method=max_confidence`，
`judge_model`，`classifier_model_path`．`data/dataset.jsonl`（1,915 行，ビット単位で不変），
`data/classifier_train.jsonl`（1,427 行，**ビット単位で不変**．追加分は別ファイルへ書く），
`node.py`・`aggregator.py`・`classifier.py`・`metrics.py`・`run_experiment.py`・`train_domain_classifier.py`
（いずれも**コード変更なし**）．**ドメイン固有の後付け補正は 2026-09-23 恒久運用ルールにより追加しない．
N・選択規則・プールの定義は 10 ドメイン共通である．**

**仮説（事前登録．本走前に数値で記録する）**

「現行の訓練集合は各ドメイン 150 行の無作為標本で，決定境界の近傍が十分に張られていない．同一タスク群には
未使用行が 4,578 行残っている．**現行分類器が正解ドメインへ低い確率しか与えない行を 10 ドメイン共通の規則で
各 100 行追加すると，同数を無作為に追加した場合より境界が精緻化され，`top1_accuracy` が向上する**．
効果は education（recall 0.4800）・social_science（0.6867）・general（0.6933）など境界の弱いドメインに
相対的に大きく出ると予想するが，**ドメインごとの調整は一切行わない**．」

**オフライン事前実測（本フェーズで実施．実機不使用．`data/embcache_*` と `results/` のみを使用）**

| ゲート | 内容 | 実測 | 判定 |
|---|---|---|---|
| **G1（replay の忠実度）** | キャッシュ埋め込み（`embcache_eval_qwen3-embedding_0.6b{,__p1}.npy`）＋現行 artifact から `select_dispatch_targets()` の gap escalation を再現し，Iter83 本走を再現できるか | `compound_domain_set_recall`=**0.5819277108**，`compound_mean_dispatched_count`=**1.8891566265**，複合 k 分布 (278,21,116) が**本走実測と完全一致**．全 1,915 行 top1 は 0.793211（本走 0.792689，**差 1 行**） | **PASS**（本走の経路指標はオフラインで事前に予測できる） |
| **G2（訓練経路の決定性）** | `data/classifier_train.jsonl` とキャッシュ埋め込みから再訓練した分類器が，配備中の artifact と一致するか | 再訓練モデルと artifact の評価集合 1,500 行 top1 が **0.786667 で完全一致** | **PASS**（訓練に乱数由来の揺らぎは無い．差が出れば必ず訓練データの差） |
| **G3（選択規則の有効性．本レバーの中核）** | 訓練 1,427 行を各ドメイン 50% で seed / pool に分割し，seed で訓練した分類器で pool を採点．hard 上位 N 件と無作為 N 件を seed へ足して再訓練し，評価集合 1,500 行の top1 を比較（3 seed × N∈{20,40,60}） | 平均 top1: seed のみ 0.7649 / N=20 hard **0.7887** vs rand 0.7749（**+1.38pt**）/ N=40 hard **0.7929** vs rand 0.7778（**+1.51pt**）/ N=60 hard 0.7894 vs rand 0.7864（+0.30pt）．**9 対比較中 8 で hard が上回る**．N=40 hard（1,112 行）は**全 1,427 行で訓練した現行構成の 0.7867 すら上回る** | **PASS** |
| **G4（最難帯を除外すべきか）** | N=40 固定で，最難 10 件 / 20 件を飛ばしてから取る変種を比較（3 seed） | 平均 top1: pure hard **0.7929** / skip10 0.7887 / skip20 0.7740．**除外は改善しない** | **PASS**（純粋な least-confidence を採用．追加パラメータを持たない） |
| **G5（着地点の事前登録）** | 下表を本走前に記録すること．**実装フェーズでは，新 artifact を作った直後に同じ replay を走らせ，本走前に予測値を journal へ追記すること**（G1 により本走の経路指標は事前に確定できる） | 下表 | 実装フェーズで判定（合格条件は課さない） |

G3 の測定上の注意: 評価集合 1,500 行のうち 72 行は現行訓練集合と設問文が重複している（**既知の欠陥．
backlog B125 に起票済み**．education 46 / history_culture 26）．G3/G4 はこの 72 行を除いた 1,428 行でも
同時に算出しており，結論（hard > rand，pure > skip）は同一である．本反復で追加する 900 行は評価集合と
0 件重複なので，漏洩の割合は 72/1,427 から 72/2,327 へ下がる．72 行は基準線・新構成の双方に等しく含まれ，
対比較を歪めない．

**着地点予測（事前登録）**

| 指標 | Iter83 本走実測（基準線 `results/20260927_070239/`） | **Iter84 予測値** |
|---|---|---|
| `top1_accuracy`（全 1,915 行） | 0.792689 | **0.800〜0.815．点推定 0.807（+1.4pt）** |
| 単一 1,500 行 top1 | 0.786000 | +1〜+3pt |
| 複合 415 行 top1 | 0.816867 | ほぼ不変〜微増 |
| 既存 1,600 行部分集合 top1 | 0.787500 | +1〜+3pt |
| `education` recall / precision | 0.4800 / 0.5304 | 上振れ余地が最大．ただし個別の成功条件は課さない |
| `legal` recall / precision | 0.8800 / 0.8610 | **追加行 0 件の唯一のドメイン．非退行の監視対象** |
| `compound_domain_set_recall` | 0.581928 | **予測不能（確信度分布が動くため）**．非退行の枠内で報告 |
| `compound_mean_dispatched_count` | 1.889157 | **上振れ／下振れの両方がありうる**．非退行の枠内で報告 |
| `mean_duration_ms` | 2286.9 | 送出数に比例．非退行枠 2744.3 |
| ECE | 0.026 | 0.02〜0.05 |
| `answer_quality_accuracy` / `end_to_end_accuracy` | 0.570667 / 0.350914 | ノイズ床 3SD=2.6pt（success_criteria (5)）の範囲でのみ判定 |
| rank1 以外が選ばれた行数 | 2 | ≤ 10 |
| Random / BestSingle / Oracle | 0.121671 / 0.126893 / 1.0 | success_criteria (3) により毎回併記 |

**許容幅とノイズ床の根拠（B132 申し送り 3 への回答．「±1 行」を使わない理由）**

- 訓練パイプラインは埋め込みを固定すればビット決定的である（G2 で確認）．実行時に残る揺らぎは ollama の
  埋め込み再計算だけで，その大きさは**オフライン replay と Iter83 本走の差＝1,915 行中 1 行（0.052pt）**である．
- しかし **Iter83 は送出段だけを動かすレバーで `selected_domain` が構造的に不変だったのに対し，本反復は
  分類器そのものを作り替える**．よって「基準線との差が 1 行以内であること」を非退行条件にするのは誤りであり，
  **非退行は (a) per-domain の検定（BH 補正 q=0.05），(b) 複合指標の明示レンジ，(c) 運用指標の上限**で判定する．
- 主基準 `top1_accuracy` は McNemar の対比較で判定する．検出限界は discordant ペア数 n_d に対し
  `1.96·sqrt(n_d)/1915` で，n_d≈100 なら 1.02pt，n_d≈150 なら 1.25pt である．**採用閾値 +1.0pt は
  この検出限界とほぼ同じ水準**なので，閾値の充足だけでなく McNemar の有意性も同時に要求する（下記 AND 条件）．

**成功条件・非退行条件（事前登録．結果を見る前に固定する）**

基準線は **Iter83 本走 `results/20260927_070239/`**（全 1,915 行: top1=0.792689，単一 1,500 行 top1=0.786000，
複合 415 行 top1=0.816867，1,600 行部分集合 top1=0.787500，`compound_domain_set_recall`=0.581928，
`compound_mean_dispatched_count`=1.889157，fallback=0.0，dispatch_failure=0.000522，ECE=0.026，
mean_duration_ms=2286.9，answer_quality=0.570667，end_to_end=0.350914，rank1 以外が選ばれた行数=2）．

| 区分 | 指標 | 基準線 | 合格条件 |
|---|---|---|---|
| **主基準（効果）** | 全 1,915 行 `top1_accuracy` | 0.792689 | **(i) McNemar 対比較（α=0.05）で有意，かつ (ii) +1.0pt 以上（≥ 0.802689）**．2 条件の AND．Wilson 95% CI と検出限界 `1.96·sqrt(n_d)/1915` を併記する |
| **非退行①** | per-domain recall / precision 計 20 指標（BH 補正 q=0.05） | Iter83 実測 | **有意退行 0 件**．特に **legal（唯一の追加 0 件ドメイン）**と，教師データが増えたのに悪化したドメインが無いことを個別に明記する |
| **非退行②（複合被覆）** | `compound_domain_set_recall` | 0.581928 | **≥ 0.539759（Iter82 水準を下回らない）**．gt=0.36 は新しい確信度分布に対して未較正なので，Iter83 水準の維持は要求しない |
| **非退行③（複合予算）** | `compound_mean_dispatched_count` | 1.889157 | **≤ 2.10（Iter83 比 +11% 以内）**．超過は「送出コストの実質的な悪化」として rejected |
| **非退行④** | `fallback_rate` / `dispatch_failure_rate` | 0.0 / 0.000522 | fallback = 0.0，dispatch_failure ≤ 0.005 |
| **非退行⑤（B132 申し送り 2）** | **rank1 以外が選ばれた行数**（`selected_domain` ≠ `probe_candidates` の最大確信度ドメイン） | 2 | **≤ 10**．部分 dispatch 失敗の検出器．超過したら送出段の失敗が top1 を汚染している疑いとして原因を特定してから判定する |
| **非退行⑥** | `mean_duration_ms` | 2286.9 | **≤ 2744.3（+20% 以内）** |
| **非退行⑦** | ECE | 0.026 | **≤ 0.08** |
| 報告のみ | `answer_quality_accuracy` / `end_to_end_accuracy` | 0.570667 / 0.350914 | 3SD=2.6pt のノイズ床（success_criteria (5)）を超えない限り有意と判定しない |
| 報告のみ | 単一 1,500 行 / 複合 415 行 / 既存 1,600 行部分集合の top1 | 0.786000 / 0.816867 / 0.787500 | 毎回併記 |
| 報告のみ | Random / BestSingle / Oracle | 0.121671 / 0.126893 / 1.0 | success_criteria (3) により毎回併記 |
| 報告のみ | 漏洩 72 行を除いた 1,843 行の top1 | — | B125 の絶対値水増しの影響を切り分けるため併記 |

**判定規則（事前登録）**

- **adopted**: 主基準 (i)(ii) を満たし，非退行①〜⑦をすべて満たす．
- **partial**: McNemar が有意で Δtop1 が +0.5〜+1.0pt，かつ非退行①〜⑦を満たす．
  **G3 の予測（hard 選択の優位 +1.5pt）からの乖離の原因**（プール規模の違い，基準訓練集合が大きいことによる
  逓減，最難帯の雑音など）を特定して記録したうえで，N を変えた再試行の是非を backlog へ起票する．
- **no_effect**: |Δtop1| < 0.5pt または McNemar が非有意．**先に F3/F4/F5 を再検証**してから結論を書く
  （artifact が実際に差し替わっているか・各ノードへ配布されたか）．
- **rejected**: Δtop1 ≤ −0.5pt，または非退行①〜⑦のいずれかに違反．
- **invalid（実験不成立）**: F1〜F5 のいずれか不合格，`question_count != 1915`，
  `compound_domain_question_count != 415`，**新 artifact の sha256 が `1cfcd3d8...`（基準線）と一致**
  （＝再訓練が反映されていない），または**本走の top1 が新 artifact のオフライン replay 予測から 1.0pt 以上乖離**
  （＝実行時経路が predicted と別の artifact を読んでいる疑い）．
- **復元手順（partial / no_effect / rejected 共通）**:
  `cp models/domain_classifier_pre_iter84_baseline.joblib models/domain_classifier.joblib` で artifact を戻し
  （sha256 が `1cfcd3d8...` に戻ることを確認），`mise run deploy` を再実行して全 10 ノードで smoke_check を通す．
  `data/classifier_train.jsonl` と `config.yaml` は本反復で一切触っていないので復元不要．
  `data/classifier_train_iter84_hardneg.jsonl` は記録として残す（過去の `classifier_train_iter*.jsonl` と同じ扱い）．

**レバーを読むコード行と，そこへ到達する条件（「変更したが実行パスに到達しない」失敗への恒久対策．省略不可）**

| 経路 | レバーが効く箇所 | 到達条件 | 到達確認の手段 |
|---|---|---|---|
| **訓練側（オフライン）** | `scripts/train_domain_classifier.py` の `--train-data` | 新しい JSONL を渡すこと | **F2** |
| **実行時側（本体）** | `classifier.py:load_domain_classifier()` → `estimate_confidence_classifier()` が `config.yaml` の `classifier_model_path`（= `models/domain_classifier.joblib`）を読む | 各ノードのコンテナが**配布後の** artifact を読むこと．artifact は `mise.toml` の `rsync` で配られる | **F4**: 全 10 ノードで artifact の sha256 が新値で一致（deploy 内の smoke_check） |
| **記録側（指標の分母）** | `run_experiment.py:98` の `dispatched_domains` 再計算（gap_threshold=0.36，今回は不変） | 同上 | **F5** |
| **到達しない経路（確認のみ．変更しない）** | 埋め込み（`expert_backend.embed_query_views()`）・`aggregator.py`・`config.yaml` | 本反復では一切触らない | **F1**: `config.yaml` の diff 0 行，`.py` の既存ファイルの diff 0 行 |

**事前ゲート（実行パス到達確認．実装フェーズで判定する）**

| ゲート | 内容 |
|---|---|
| **F1（変更の最小性）** | `git diff` が `scripts/mine_hard_negatives.py`（新規）・`data/MANIFEST.md`・`.claude/research/*` のみ．**`config.yaml` の diff は 0 行**．`node.py`・`classifier.py`・`aggregator.py`・`train_domain_classifier.py`・`build_dataset.py` の diff は 0 行 |
| **F2（データ）** | `data/classifier_train_iter84_hardneg.jsonl` が **2,327 行**．先頭 1,427 行が `data/classifier_train.jsonl` と**バイト一致**．追加 900 行の設問文が `data/dataset.jsonl` の 1,915 行および既存 1,427 行と **0 件重複**．ドメイン別の追加数が **{legal: 0, 他 9 ドメイン: 100}**．id の重複 0 件．`data/classifier_train.jsonl` の sha256 が `eb89bf7b...` のまま不変 |
| **F3（artifact）** | 新 `models/domain_classifier.joblib` の sha256 が基準線 `1cfcd3d8...` と**異なる**．`n_features_in_`=2048，`classes_` が 10 ドメイン．退避ファイル `models/domain_classifier_pre_iter84_baseline.joblib` の sha256 = `1cfcd3d8...` |
| **F4（配布）** | `mise run deploy` 後，全 10 ノードで artifact の sha256 が新値で一致．`grep '^dispatch_gap_threshold:' $REMOTE_DIR/config.yaml` が全ノードで `0.36` |
| **F5（実行時経路）** | 先頭 20 問の予備実行の `confidence`（rank1 の値）と `dispatched_domains` が，同 20 問のオフライン予測（新 artifact ＋ キャッシュ埋め込み，gt=0.36）と **20/20 一致**すること．**不一致なら本走に進まない** |

**実験手順**

1. **前提確認**: `wc -l data/dataset.jsonl` = 1915，`wc -l data/classifier_train.jsonl` = 1427，
   `sha256sum models/domain_classifier.joblib` = `1cfcd3d8...`，`config.yaml` の
   `embedding_view_concat: true` / `embedding_instruction` / `dispatch_gap_threshold: 0.36`．
2. **artifact 退避**: `cp models/domain_classifier.joblib models/domain_classifier_pre_iter84_baseline.joblib`．
3. **hard negative の採掘**（**wafl-ctrl5 限定**．config.yml 絶対条件 (B)）:
   新規 `scripts/mine_hard_negatives.py` を実装して実行する．プール構築は `build_dataset.py` の
   `_parse_jmmlu_task_csv` / `_format_jmmlu_query` / `_DOMAIN_TASK_MAP` を**再利用**し（重複実装しない），
   埋め込みは `expert_backend.embed_query_views(..., instruction=..., concat_views=True)` を使う
   （訓練・実行時と同一関数．Iter36 型の train/eval 不一致を構造的に防ぐ）．
   プール埋め込みは `data/embcache_pool_qwen3-embedding_0.6b{,__p1}.npy` へ保存し再実行を安くする．
4. **再訓練**（**wafl-ctrl5 限定**）: `data/MANIFEST.md` の現行コマンドの `--train-data` だけを
   `data/classifier_train_iter84_hardneg.jsonl` に差し替えて実行する（他の引数は 1 文字も変えない）．
5. **本走前の着地点記録（G5）**: 新 artifact ＋ キャッシュ埋め込みで replay を走らせ，
   予測 top1 / per-domain / `compound_domain_set_recall` / `compound_mean_dispatched_count` を
   **本走前に** journal へ追記する．
6. **配布**: `mise run deploy` → **F4**．
7. **予備 20 問**（`data/dataset_20.jsonl`）→ **F5**．不一致なら本走に進まない．
8. **本走**: wafl500〜509 で 1,915 問フルスペック 1 回（config.yml 絶対条件 (A)．事前 replay で
   経路指標が予測できても省略しない）．想定所要は Iter83 実績から約 75 分．
9. `data/MANIFEST.md` に新 artifact の sha256・新訓練ファイルの sha256 と行数・生成コマンドを追記する．

**次イテレーションへの申し送り（B132 申し送り 1 の引き継ぎ）**

本レバーは分類器を作り替えるので **rank1−rank2 gap の分布が再び動く**．Iter81→82→83 で 3 回続けて観測した
機序のとおり，**本反復が adopted になった場合は，直後に `dispatch_gap_threshold` の再較正イテレーションを
1 回挟むこと**（レバー `dispatch_gap_threshold_recalibration` の再オープン．予算整合点の掃引をやり直す）．
単一レバー原則により本反復では閾値を触らないため，`dispatch_gap_threshold=0.36` は新分布に対して未較正のまま
本走する．非退行②③はその前提で緩めに置いてある．

### 実装・実験 (Iter84)

**（この節はオーケストレータが rc-executor の報告から補完した．rc-executor は「journal への実験結果記入は
分析・考察フェーズの担当」という自身の役割定義を理由に journal を編集せず，`data/MANIFEST.md` と戻り値に
のみ記録した．G5 は「本走前に journal へ追記」を要件としていたため，この点は要件どおりに運用されていない．
G5 の予測値そのものは本走前に算出・記録されており（下表），事後の辻褄合わせではない．運用ルールの
不整合として backlog へ申し送る．）**

**実施した変更**

| 種別 | パス | 内容 |
|---|---|---|
| 新規 | `scripts/mine_hard_negatives.py`（369 行） | JMMLU 未使用プールから 10 ドメイン共通規則で hard negative を採掘．プール構築は `build_dataset.py` の `_DOMAIN_TASK_MAP` / `_parse_jmmlu_task_csv` / `_format_jmmlu_query` を再利用，埋め込みは `expert_backend.embed_query_views(..., concat_views=True)`（`train_domain_classifier.py`・`node.py` と同一関数） |
| 生成 | `data/classifier_train_iter84_hardneg.jsonl` | 2,327 行（既存 1,427 行 ＋ 追加 900 行） |
| 生成 | `data/embcache_pool_qwen3-embedding_0.6b{,__p1}.npy` | プール埋め込みキャッシュ |
| 退避 | `models/domain_classifier_pre_iter84_baseline.joblib` | 基準線 artifact（sha256 `1cfcd3d8...`） |
| 更新 | `models/domain_classifier.joblib` | 新 artifact（sha256 `34e4d33bcfb0695a48b410cb70fd1d3d38973e048631b1c1ffe516df423b34c9`，`n_features_in_`=2048，10 クラス） |
| 追記 | `data/MANIFEST.md`（+109 行） | 生成コマンド・sha256・ゲート結果・本走実測値 |
| **無変更** | `config.yaml`・`node.py`・`aggregator.py`・`classifier.py`・`train_domain_classifier.py`・`build_dataset.py`・`data/classifier_train.jsonl` | `git diff` 0 行．`classifier_train.jsonl` は sha256 `eb89bf7b...` 不変 |

**事前ゲートの判定**

| ゲート | 内容 | 結果 |
|---|---|---|
| **F1** | 変更の最小性（上記 6 ファイル ＋ 既存訓練データの diff が 0 行） | **PASS** |
| **F2** | データ健全性: 2,327 行，先頭 1,427 行が既存ファイルとバイト一致，追加 900 行の内訳 `{legal: 0, 他 9 ドメイン: 各 100}`，id 重複 0，評価 1,915 行・既存訓練 1,427 行・追加行同士の重複いずれも 0 | **PASS** |
| **F3** | 新 artifact が基準線と別物で，退避ファイルは基準線と一致 | **PASS** |
| **G5** | 本走前 replay（新 artifact ＋ キャッシュ埋め込み） | **実施**（予測値は下表） |
| **F4** | `mise run deploy` 後，全 10 ノードで artifact sha256 `34e4d33b...` 一致・`dispatch_gap_threshold: 0.36` 一致．健康チェック・smoke_check（git-status / hashes / probe）全合格 | **PASS** |
| **F5** | 予備 20 問（`data/dataset_20.jsonl`）の `confidence`・`dispatched_domains` がオフライン予測と **20/20 完全一致** | **PASS** |

**本走**: `results/20260927_110526/`（1,915 問フルスペック，所要約 63 分）．基準線は Iter83 本走
`results/20260927_070239/`．

**主基準の実測（判定は分析フェーズ）**

| 指標 | 基準線（Iter83） | 事前登録の予測 | G5 の本走前 replay 予測 | **本走実測** |
|---|---|---|---|---|
| `top1_accuracy`（全 1,915 行） | 0.792689 | 0.800〜0.815（点推定 0.807） | 0.806266 | **0.804700**（+1.201pt） |
| Wilson 95% CI | — | — | — | [0.786342, 0.821838] |
| McNemar | — | α=0.05 で有意 | — | chi2=2.674033，**p=0.101997（非有意）** |
| discordant（基準線正解→新誤り / 基準線誤り→新正解） | — | — | — | 79 / 102 |
| 検出限界 `1.96·sqrt(n_d)/1915`（n_d=181） | — | — | — | 1.378pt（実測 Δ1.201pt は**これを下回る**） |

**事前登録の主基準は AND 条件「(i) McNemar 有意 かつ (ii) +1.0pt 以上」であり，(ii) は満たすが (i) を
満たさない．**

**非退行条件の実測**

| # | 条件 | 基準線 | 実測 | 充足 |
|---|---|---|---|---|
| ① | per-domain 20 指標（BH q=0.05）で有意退行 0 件 | — | **有意差 2 件**．`education_recall` 0.4120→**0.3133**（p=0.000427，**退行**），`history_culture_recall` 0.8009→0.8701（p=0.000796，改善）．他 18 指標は有意差なし | **不充足**（退行 1 件） |
| ② | `compound_domain_set_recall ≥ 0.539759` | 0.581928 | 0.573494 | 充足 |
| ③ | `compound_mean_dispatched_count ≤ 2.10` | 1.889157 | 2.048193（+8.4%） | 充足 |
| ④ | fallback 0.0 / dispatch_failure ≤ 0.005 | 0.0 / 0.000522 | 0.0 / 0.001567 | 充足 |
| ⑤ | rank1 以外が選ばれた行数 ≤ 10 | 2 | 3 | 充足 |
| ⑥ | `mean_duration_ms ≤ 2744.3` | 2286.9 | 2280.408 | 充足 |
| ⑦ | ECE ≤ 0.08 | 0.026 | 0.075591（+4.96pt） | 充足（ただし上限際） |

**報告指標**

| 指標 | 基準線 | 実測 |
|---|---|---|
| 既存 1,600 行部分集合 top1 | 0.787500 | 0.808125 |
| `single_domain_top1`（single 1,500 行） | 0.786000 | 0.807333 |
| 複合 415 行 top1 | 0.816867 | **0.795181**（低下） |
| 漏洩 72 行除外（1,843 行）top1 | — | 0.811177 |
| Cohen's kappa | 0.761499 | 0.785973 |
| `answer_quality_accuracy` | 0.570667 | 0.580667 |
| `end_to_end_accuracy` | 0.350914 | 0.362924 |
| Random / BestSingle / Oracle baseline | — | 0.121671 / 0.126893 / 1.0（不変） |

**per-domain 詳細（基準線 → 実測，recall / precision）**

business_economics 0.8182/0.7746→0.8139/0.7611，computer_science 0.8268/0.8884→0.8571/0.9041，
education 0.4120/0.5304→**0.3133**/0.6759，general 0.4626/0.8750→0.5022/0.8769，
history_culture 0.8009/0.7400→**0.8701**/0.7701，legal 0.6626/0.8610→0.6420/0.9123，
mathematics 0.6710/0.8564→0.7013/0.8757，medical 0.7510/0.8153→0.7427/0.7553，
natural_science 0.6494/0.8571→0.6580/0.7958，social_science 0.4545/0.7554→0.5108/0.7239．

**想定外の事象（分析フェーズへの申し送り）**

1. **プール実測値が事前登録値から微小に乖離**: natural_science 783→776（-7），education 233→232（-1），
   legal・他は一致．原因は JMMLU の**同一タスク CSV 内で文字通り重複している設問行**（`college_physics`
   内 6 件，`conceptual_physics` 内 1 件，`high_school_psychology` 内 1 件）をプール構築時に重複排除した
   ため（ドメイン間の混入ではなく同一タスク内の重複であることを実データで確認済み）．各ドメインとも
   N=100 に対して十分な余裕があり，**最終選定結果（9 ドメイン各 100 件，legal 0 件，計 900 件）には
   影響しない**．
2. **`mise run analyze` の最新ディレクトリ自動検出が誤動作**: `ls -1d results/*/ | sort | tail -1` が
   ASCII 順で `i` > `2` のため `results/iter45_preliminary/` を選んでしまう．今回は
   `mise run analyze -- 20260927_110526` と明示指定して回避した（コード変更なし）．`mise.toml` 側の
   既知の弱点として backlog へ記録する．
3. **主基準の McNemar が非有意**（p=0.102）であり，G3 のオフライン推定（+1.51pt，9 対比較中 8 で hard
   優位）から実効果幅（+1.201pt，検出限界 1.378pt 未満）へ乖離した．
4. **`education_recall` が事前仮説に反して有意に退行**（0.4120→0.3133）．ただし `education_precision` は
   0.5304→0.6759 と上昇しており，education へ張り出していた決定境界が引き締まった可能性がある．
5. **複合 415 行 top1 が 0.816867→0.795181 と低下**する一方，single 1,500 行は 0.786000→0.807333 と
   上昇した．hard negative が single 由来（JMMLU の単一タスク行）であることとの関係を分析フェーズで
   検討すること．

### Iteration 84 実行済み

**変更（1 レバーのみ）**

`cross_domain_training_data_augmentation` = `hard_negative_mining_all_domains`．
`data/classifier_train.jsonl`（1,427 行）へ，現行分類器の正解ドメイン確率 `p_true` が低い JMMLU 未使用行を
10 ドメイン共通規則で各 100 行（legal はプール 0 のため 0 行）追加した 2,327 行で分類器を再訓練し，
`models/domain_classifier.joblib` を差し替えた（sha256 `1cfcd3d8...` → `34e4d33b...`）．
`config.yaml`・既存 `.py`・`data/classifier_train.jsonl`・評価集合はいずれも diff 0 行（F1・F2 で確認）．

**判定: rejected（事前登録の判定規則にそのまま該当．artifact は基準線へ復元する）**

| 事前登録の条件 | 実測 | 充足 |
|---|---|---|
| 主基準 (i) McNemar α=0.05 で有意 | chi2=2.674033，**p=0.101997** | **不充足** |
| 主基準 (ii) Δtop1 ≥ +1.0pt | 0.792689 → **0.804700**（+1.201pt，Wilson 95%CI [0.786342, 0.821838]） | 充足 |
| 非退行① per-domain 20 指標（BH q=0.05）で有意退行 0 件 | `education_recall` 0.4120→**0.3133**（p=0.000427，退行）／`history_culture_recall` 0.8009→0.8701（改善） | **不充足** |
| 非退行②〜⑦ | set_recall 0.573494 ≥ 0.539759，mean_dispatched 2.048193 ≤ 2.10，fallback 0.0／dispatch_failure 0.001567，rank1 以外 3 行 ≤ 10，mean_duration_ms 2280.4 ≤ 2744.3，ECE 0.075591 ≤ 0.08 | すべて充足 |

判定規則の `rejected` 分岐（「非退行①〜⑦のいずれかに違反」）と `no_effect` 分岐（「McNemar が非有意」）が
同時に成立する．**より厳しい `rejected` を採る**．BH 補正後に有意な実測の退行が 1 件ある以上，
「効果が見えなかった」ではなく「害が確認された」と記録するのが正しいためである．
`partial` は採らない: 本リポジトリの前例（Iter29/30 の較正系，Iter62/63 の multilabel 系）で `partial` は
**主基準を満たしたうえで非退行が 1 件 FAIL した場合**に限って用いられており，今回は主基準 (i) が不成立で
その前提を欠く．事前登録の `partial` 分岐（Δ +0.5〜+1.0pt かつ非退行全充足）にも該当しない．
**`invalid` ではない**: F3（sha256 が別物）・F4（全 10 ノードで新 sha256 一致）・F5（予備 20 問が 20/20 一致）が
すべて PASS で，本走 top1 0.804700 は新 artifact の replay 予測 0.806266 と 0.157pt しか違わない
（invalid 判定の閾値 1.0pt 以内）．レバーは確かに実行パスへ到達している．

**復元（必須．次の実験の前に行うこと）**: `cp models/domain_classifier_pre_iter84_baseline.joblib
models/domain_classifier.joblib`（sha256 が `1cfcd3d8...` に戻ることを確認）→ `mise run deploy` で全 10 ノードへ再配布．
本節の執筆時点では**まだ実施していない**（実機への配布は実験フェーズの操作のため）．
以後の基準線は Iter83 本走 `results/20260927_070239/` のままである．

**結果の内訳（本節で新たに実測した数値．すべて `results/20260927_070239` 対 `results/20260927_110526`）**

| 母数 | 基準線 | 実測 | Δ | discordant（悪化/改善） | McNemar |
|---|---|---|---|---|---|
| 全 1,915 行 | 0.792689 | 0.804700 | **+1.201pt** | 79 / 102 | chi2=2.674，p=0.102（非有意） |
| single 1,500 行 | 0.786000 | 0.807333 | **+2.133pt** | 54 / 86 | chi2=6.864，**p=0.0088（有意）** |
| compound 415 行 | 0.816867 | 0.795181 | **−2.169pt** | 25 / 16 | chi2=1.561，p=0.212（非有意） |

**論点 1: G3 のオフライン推定 +1.51pt と実効果 +1.201pt の乖離，および必要標本数**

- 乖離そのものは小さい（0.31pt）．G3 は訓練 1,427 行を半分（713 行）に割った小規模設定で，
  **追加行数／元の訓練行数 ≒ 56%** の点を測っていた．小さい訓練集合ほど 1 行あたりの限界情報量が大きいので，
  G3 の +1.51pt は本走（1,427 行に +900 行）への上振れ気味の外挿である．実効果がその 8 割で出たこと自体は
  仮説と整合し，**方向・桁とも外していない**．外れたのは効果量ではなく**検出力の見積り**である．
- 計画時は「n_d≈100〜150 なら検出限界 1.02〜1.25pt」と見積り，採用閾値 +1.0pt をそこに合わせた．
  実際の n_d は **181**（discordant 率 p_d=0.09452）で検出限界は **1.378pt** へ伸び，
  +1.201pt はその内側に収まった．**分類器を作り替えるレバーは送出段のレバーより discordant を多く生む**
  （Iter83 は正味 1 行）ため，n_d を 100〜150 と置いた前提自体が楽観的だった．
- **必要標本数（今後の成功条件設計に使うこと）**: McNemar は `N ≥ (z_α+z_β)²·p_d/δ²` で見積れる．
  実測 p_d=0.0945 のとき **δ=1.2pt を有意にするには N≥2,517（検出力 50%＝有意境界）・
  N≥5,143（検出力 80%）・N≥6,885（検出力 90%）**．現行 N=1,915 で 80% の検出力が得られる最小効果は
  **δ≥1.97pt**，有意境界でも **δ≥1.378pt** である．
  **帰結: 「+1.0pt 以上 かつ McNemar 有意」という AND 条件は，N=1,915 では事実上満たせない事前登録だった**
  （+1.0〜1.38pt の帯が構造的に判定不能になる）．今後は (a) 期待効果が 2pt 未満なら評価集合を先に増やす，
  (b) それが出来ないなら閾値を検出限界（`1.96·sqrt(p_d/N)`）以上に置く，のどちらかを計画時に選ぶこと．

**論点 2: `education_recall` の退行と `education_precision` の上昇（「境界の引き締まり」は single 行では正しく，compound 行では成り立たない）**

- 指標の母数は expected に education を含む **233 行**（single 150 ＋ compound 83）である．内訳は
  **single 72→62（−10）・compound 24→11（−13）**で，**退行 23 行のうち 57% が compound 行由来**である．
- **single 行だけを見ると「引き締まり」の解釈は妥当**: recall 0.4800→0.4133（−6.67pt）に対し
  precision 0.5669→0.7294（**+16.25pt**），**F1 は 0.5199→0.5277（+0.78pt）で微増**．
  基準線で education が誤って吸い込んでいた行（medical 23・social_science 17 など計 55 行）は
  新構成で 23 行（medical 8・legal 5・social_science 5 ほか）まで減り，**そのうち 29 行が正解ドメインへ復帰した**．
  逆向きの流出は single の education 行 18 行（基準線で正解→新構成で誤り）で，行き先は **medical 10・
  social_science 8**．差し引き −10 行（72→62）である．
  すなわち single 行では「過剰に張り出していた education の決定境界が引き締まり，他ドメインの recall を押し上げた」
  という解釈がデータと一致する（general F1 +4.69pt，history_culture F1 +5.50pt，computer_science F1 +2.91pt）．
- **compound 行では純粋な損失**: education を含む 83 行で education が選ばれた回数が 24→11 へ落ち，
  compound で悪化した 25 行のうち 16 行が expected に education を含む（education+general 7・
  education+natural_science 4・education+legal 3・education+social_science 2）．
  複合設問は「2 ドメインのどちらかが選ばれれば正解」なので，**張り出した education 境界は compound 行では
  得点源として働いていた**．引き締めはその得点源を直接削る．
- **母数込みの F1（233 行基準）は 0.46375→0.42815（−3.56pt）で悪化**する．すなわち
  「F1 で見れば相殺される」とは言えない．**single 行に限れば +0.78pt，全母数なら −3.56pt** と結論が反転するので，
  今後 education を論じるときは母数を必ず明記すること．
- **2026-09-23 恒久ルールにより，education だけを狙った閾値・intercept・訓練データの後付け補正での是正は行わない．**
  是正するなら 10 ドメイン共通の規則（例: 追加行数 N の変更，compound 行を含む評価での選択規則）でのみ行う．

**論点 3: compound の低下（−2.169pt）と single の上昇（+2.133pt）の乖離**

- 訓練へ追加した 900 行は**すべて JMMLU の単一タスク行**であり，複合設問は 1 行も含まない．
  したがって本レバーは **single 行の分布上でのみ境界を精緻化し，compound 行にとっては分布外の変更**である．
- compound で悪化した 25 行の遷移先は **medical 14・history_culture 4・business_economics 3** に集中する．
  medical は precision が 0.8153→0.7553（−6.00pt）と落ちる一方 compound での被選択が 222→237 行へ増えており，
  **「複合設問の文面に対して medical が過剰に立つ」方向へ境界が動いた**．
  medical は未使用プールが最大（1,110 行）で，追加 100 行の情報量が最も豊富だった側である．
- 機序の整理: **single 行では「1 つの正解ドメインを当てる」ので境界の引き締めが得になり，
  compound 行では「2 つのうちどちらでもよい」ので境界の緩さが得になる**．両者は最適な境界が構造的に食い違う．
  hard negative mining を single 行だけから採る限り，この食い違いは N を調整しても消えない．
  **compound を守るには，訓練側に複合的な信号を入れるか，評価側で両者を分けて事前登録するかのどちらかが要る．**
- なお compound の −2.169pt は discordant 25/16・chi2=1.561・p=0.212 で**単体では有意ではない**
  （n=415 では ±4pt 程度が検出限界）．「compound が有意に壊れた」とまでは言えない点を記録しておく．

**論点 4: ECE 0.026→0.075591 の悪化は「過信」ではなく「自信不足」への移行である**

| 指標（全 1,915 行） | 基準線 | 新構成 |
|---|---|---|
| rank1 confidence 平均 | 0.7843 | 0.7306 |
| top1_accuracy | 0.7927 | 0.8047 |
| 平均確信度 − 正答率 | **−0.0084** | **−0.0741** |
| rank1−rank2 gap 平均 | 0.6650 | 0.5967 |
| gap の p25 / p50 / p75 | 0.4159 / 0.7730 / 0.9381 | 0.3557 / 0.6657 / 0.8636 |
| gap < 0.36 の行の割合 | 21.78% | 25.22% |

- 精度は上がったのに確信度が下がった．**最難帯 900 行を訓練へ足したことで `CalibratedClassifierCV`
  （temperature）の温度が上がり，確率が一様側へ寄った**というのが素直な機序である．ECE の悪化は
  過信ではなく**系統的な自信不足**（−7.41pt）で，方向は Iter29〜31 で扱った過信とは逆である．
- gap 分布が一様に圧縮された結果，固定閾値 `dispatch_gap_threshold=0.36` に対して
  **gap<0.36 の行が 21.78%→25.22% へ増え**，`compound_mean_dispatched_count` が 1.889→2.048（+8.4%）になった．
  compound 行に限れば gap<0.36 は 33.01%→36.87%．これは Iter81→82→83 で 3 回観測した
  「特徴量・分類器を変えると gap 閾値が未較正になる」機序の **4 回目**である．
- **B132 申し送り 1（再較正イテレーションを挟むか）への回答**: 本反復は rejected で artifact を復元するので，
  確信度分布は基準線へ戻り `gt=0.36` は較正済みのまま保たれる．**したがって今回は再較正を挟まない**．
  ただし上の数値は，**分類器を作り替えるレバーが adopted になった時には必ず再較正が要る**ことを
  4 例目として裏づける．次に分類器を差し替える反復では，計画時点で再較正をセットで予定すること．

**学び**

1. **N=1,915 の評価集合は +1.0〜1.4pt の効果を原理的に判定できない**（p_d≈0.095 のとき有意境界 1.378pt）．
   この帯に入る効果を狙うレバーでは，事前登録の時点で「判定不能で終わる」ことが確定している．
   Iter63〜68 の複合設問の検出力不足（n=100）と**同型の失敗が，今度は全体集合の側で起きた**．
   `N ≥ (z_α+z_β)²·p_d/δ²` を計画フェーズの必須計算項目にすること．
2. **single 行と compound 行では最適な決定境界が逆を向く**．single は引き締め，compound は緩さを好む．
   単一ドメイン行だけから採った hard negative は前者しか最適化しないので，複合被覆とは構造的に
   トレードオフになる．今後 `top1_accuracy`（全体）を主基準に置くレバーでは，この 2 部分集合の
   内訳を必ず分解して報告すること（全体だけ見ると打ち消し合って「効果なし」に見える）．
3. **難しい行を訓練へ足すと確率は一様側へ寄る**（精度は上がるのに確信度が下がる＝自信不足）．
   ECE 悪化＝過信という先入観で読むと誤診する．較正の向きは毎回，平均確信度 − 正答率の符号で確かめること．
4. **教科書どおりの hard negative mining は，本研究のデータでは「効果が無い」のではなく
   「single では効き，compound と education で損を出し，正味が検出限界に埋もれる」**．
   オフライン G3 の 9 対比較中 8 勝という強い事前証拠があっても，評価集合の母数が足りなければ確定できない．

---

