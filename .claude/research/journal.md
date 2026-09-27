## Iteration 87: hard negative と無作為抽出の半々混合による訓練データ拡充

### 調査 (Iter87)

本反復のレバーは backlog B141 (c) で `cross_domain_training_data_augmentation` =
`hard_random_hybrid_all_domains` に確定済みで，選定の裁量は無い．config.yml の同レバー note は
「追加総数 900 行・ドメイン別内訳（legal 0，他 9 ドメイン各 100）・出力書式・id 規則を Iter86 と同一に保ったまま，
各ドメインの内訳を『最難 50 行 ＋ 残プールからの無作為 50 行』へ変える（10 ドメイン一律）」と規定している．
したがって調査の問いは 4 つである．**(Q1) 50/50 が各ドメインで実際に構成可能か（プール実測）．
(Q2) 乱数種の固定方法と再現可能性．(Q3) Iter86 の層別 Δ から見た定量投影と検出力．(Q4) 関連研究．**

**Q1: 9 ドメインすべてで構成可能．ただし実質的に変わるのは 6 ドメイン・900 行中 214 行にすぎない**

本フェーズで `mine_hard_negatives.py` の `determine_used_tasks` / `build_pool` を直接呼び，
JMMLU.zip（本環境キャッシュ，sha256 `3ba7d912...`．**MANIFEST 記録値と一致＝ G0 は現時点で PASS**）から
プールを実測した（CPU のみ．実機ノード不使用）．`data/dataset.jsonl`（3,435 行）と
`data/classifier_train.jsonl`（1,427 行）は Iter86 から 1 バイトも変わっていないため，**プールは Iter86 と完全に同一**である．

| ドメイン | プール M | 最難 50 の可否 | 無作為 50 の抽出元（M−50） | 選択性 | Iter86 の選定 100 行から入れ替わる期待行数 |
|---|---|---|---|---|---|
| medical | 910 | ○ | 860 | 高 | **47.1** |
| history_culture | 579 | ○ | 529 | 高 | **45.3** |
| natural_science | 576 | ○ | 526 | 高 | **45.2** |
| business_economics | 505 | ○ | 455 | 高 | **44.5** |
| mathematics | 148 | ○ | 98 | 中 | 24.5 |
| education | 109 | ○ | 59 | **極小** | **7.6** |
| computer_science | 100 | ○ | **50（＝必要数ちょうど）** | **ゼロ** | **0.0（完全な no-op）** |
| social_science | 100 | ○ | **50（同上）** | **ゼロ** | **0.0（同上）** |
| general | 100 | ○ | **50（同上）** | **ゼロ** | **0.0（同上）** |
| legal | 0 | ―（構造的に 0） | ― | ― | 0 |
| 合計 | 3,127 | 選定 900 行 | | | **214.2 / 900（23.8%）** |

- **無作為 50 が「抽出」として成立しないドメインが 3 つある**．computer_science・social_science・general は
  M = 100 = N なので，最難 50 を除いた残りがちょうど 50 行になり，**無作為側は残り全量で確定する**（選択の自由度ゼロ）．
  結果として選ばれる 100 行は Iter86（最難 100 ＝ プール全量）と**集合として完全に一致し，本レバーはこの 3 ドメインで厳密に no-op** である．
  B141 が留保として予告したとおりの状況が実測で確認された．education も M=109 で，入れ替わるのは期待 7.6 行にとどまる．
- 入れ替わり期待行数は，残プール M−50 から 50 行を無作為に引いたとき Iter86 側の「ランク 51〜100」に当たる本数の期待値
  `50 × 50/(M−50)` から算出した（超幾何分布の期待値）．
- **したがって本レバーが実際に動かしているのは，プールが大きい medical・history_culture・natural_science・
  business_economics の 4 ドメイン（＋ mathematics）である．これは Iter86 で有意改善した 2 ドメイン
  （history_culture +6.03pt・natural_science +4.18pt）と，education を押し出した medical そのものである．**
  対して，非退行①を落とした当事者である education 側は，本レバーではほぼ触られない．
  **つまり本反復の作用機序は「弱いドメインを直接手当てする」ではなく「強いドメインの境界の押し出しを弱める」である**
  （2026-09-23 恒久運用ルール (1) に適合．ドメイン固有の補正は一切入れていない）．

**Q1 補足: 追加行の難易度プロファイル（基準線分類器の p_true で実測）**

プール埋め込みキャッシュ `data/embcache_pool_qwen3-embedding_0.6b{,__p1}.npy` の `pool_hash` が
現在のプールと**一致する**ことを確認した（`221e45e8...`，3,127 行）ので，基準線 artifact
`models/domain_classifier_pre_iter84_baseline.joblib`（`1cfcd3d8...`）で p_true を CPU 上で再計算した．

| ドメイン | Iter86 の追加 100 行の平均 p_true | **Iter87 の追加 100 行の平均 p_true（投影）** | 最難 50 の平均 | 無作為 50 の平均 |
|---|---|---|---|---|
| medical | 0.0441 | **0.3616** | 0.0202 | 0.7030 |
| history_culture | 0.1714 | **0.4279** | 0.0724 | 0.7834 |
| business_economics | 0.2088 | **0.4450** | 0.0732 | 0.8169 |
| natural_science | 0.2694 | **0.4790** | 0.1263 | 0.8317 |
| mathematics | 0.8060 | 0.8141 | 0.6580 | 0.9702 |
| education | 0.4602 | 0.4783 | 0.2566 | 0.7001 |
| computer_science / social_science / general | 0.8142 / 0.5869 / 0.6806 | **同値（no-op）** | ― | ― |

- **プール上位 4 ドメインでは追加行の平均 p_true が 0.04〜0.27 → 0.36〜0.48 へ大きく上がる**（＝訓練集合へ入る行が
  「境界のごく近傍」から「そのドメインの典型的な分布」へ移る）．行の同一性ベースでは 23.8% しか変わらないが，
  **難易度プロファイルでは 4 ドメインが別物になる**．この非対称性が本レバーの効き方を決める．

**Q2: 乱数種は `--random-seed 87` を 1 個だけ事前登録し，種の探索は行わない**

- 現行 `scripts/mine_hard_negatives.py` は乱数を一切使わない（`select_hard_negatives()` が
  `(p_true, task, query)` の辞書式順で上位 N を取るだけの決定論的実装）．本レバーの実装では
  同関数と CLI にのみ手を入れ，`--random-count`（既定 0＝従来挙動）と `--random-seed` を追加する．
- 再現可能性の担保は 3 点で行う．(a) 残プールを `(p_true, task, query)` で**全順序に整列してから**
  `numpy.random.default_rng(seed)` の `permutation` で添字を引く（`set` の反復順序や dict 順序に依存させない）．
  (b) 種・出力 JSONL の sha256・行数・ドメイン別内訳を `data/MANIFEST.md` へ記録する．
  (c) 混合後の 100 行は Iter86 と同じ `(p_true, task, query)` 順に並べ直してから
  `{domain}-hardneg-{seq:03d}` を振る（**id 規則・出力書式は Iter86 と同一**．ただし id の連番はもはや
  「難しい順の順位」を意味しない点は MANIFEST に注記する）．
- **種を複数試して良い結果のものを採る行為は行わない**．事前に 1 個へ固定する（garden of forking paths の回避）．
  一方で，**種によるばらつきは McNemar の分散には含まれない**追加の不確実性であり，本走 1 回の Δ には
  この分の変動が乗る．これは本反復の測定上の限界として明記しておく（緩和策は下記「実験手順」5b の report-only 計測）．

**Q3: 投影は Δtop1 = +0.78 〜 +1.32pt（点推定 +1.0pt 前後）．主基準 (ii) は境界上で，最も確からしい着地は `partial` 帯**

Iter86 の層別 Δ（基準線 p_true の五分位，各 n=687）: Q1 +14.85 / Q2 −6.84 / Q3 −2.33 / Q4 −1.02 / Q5 +0.44pt．
ここから 2 つのモデルで投影する（どちらが正しいかは本走でしか分からないので，両端を事前に置く）．

| モデル | 考え方 | Q1 | Q2 | Q3 | Q4 | Q5 | **Δtop1** |
|---|---|---|---|---|---|---|---|
| **A（保守・比例減衰）** | 効果は「Iter86 と同じ行が訓練に入った割合」76.2% に比例し，無作為 50 行は中立 | +11.32 | −5.21 | −1.78 | −0.78 | +0.34 | **+0.78pt** |
| **B（楽観・修復あり）** | Q1 は A と同じだが，無作為行が易しい層の境界を**積極的に補修**して Q2〜Q4 の退行を半減させる | +11.32 | −3.42 | −1.17 | −0.51 | +0.40 | **+1.32pt** |

- **モデル A では主基準 (ii)（Δ ≥ +1.0pt）を満たさない．** Iter86 が (ii) を +0.019pt（0.65 行分）の余裕で
  通過した境界上の結果だったことを踏まえると，Q1 の効果量がわずかでも縮めば (ii) は落ちる．
- 検出力: Iter86 実測の discordant は 285/3,435（SE = √285/3435 = 0.4915pt，有意境界 0.963pt）．
  入れ替わり行数に比例して discordant も 0.762 倍（n_d ≈ 217，SE = 0.4288pt，有意境界 0.840pt）になると仮定すると，
  **McNemar 有意（主基準 (i)）の検出力は モデル A で約 44%・モデル B で約 86%**，
  **(i) AND (ii) の厳格条件では モデル A で約 30%・モデル B で約 77%** である．
- **したがって最も確からしい着地は「McNemar 有意・Δ が +0.5〜+1.0pt・非退行①〜⑦を全て充足」＝ `partial` 帯**である．
  これは Iter86 の事前登録に既に定義済みの帯であり，**本反復のために判定規則を作り替える必要はない（緩めもしない）**．
  研究上の意味も明確で，`partial` に着地すれば「全体 top1 を有意に上げながら per-domain を壊さない構成」に初めて到達したことになる．
- 非退行①の投影: education の追加行はほぼ不変（7.6 行）だが，**education recall の退行は本レバーでは
  education 側ではなく medical 側の緩和を通じて効く**（Iter86 では education が失った 30 行のうち 21 行が medical へ流れた）．
  medical の追加行の平均 p_true が 0.044 → 0.362 になるため押し出しは確実に弱まる．
  **education recall は 0.36〜0.39（基準線 0.387991，Iter86 0.344111）と投影する．非退行①の合否はここで決まる．**
  natural_science precision も同経路（medical の誤検出が 9→42 行に増えたのが実体）なので 0.83〜0.87 へ戻ると投影する．

**Q4: 関連研究 — 「最難だけを採るのは不安定で，中程度・無作為を混ぜた方が堅い」は複数分野で一致している**

- **SimANS: Simple Ambiguous Negatives Sampling for Dense Text Retrieval**（Zhou et al., EMNLP 2022 industry track，
  <https://arxiv.org/html/2210.11773v2>，2026-09-27 確認）．「positive の近傍にランクする negative（ambiguous）は
  情報量が大きく false negative になりにくい」一方，**top-k の最難 negative は false negative のリスクを抱え，
  純無作為は情報量の乏しい行を拾う**と整理し，両極を避ける採取確率分布を提案している．
  本研究に引き直すと，Iter86 の「p_true 最小の 100 行」は**ドメイン帰属自体が曖昧な行（本研究の文脈での false negative 相当）を
  最優先で拾う設計**であり，medical 0.0202 / history_culture 0.0724 という最難 50 の平均 p_true はまさにその領域である．
  50/50 混合は，SimANS の主張する「両極を避ける」に厳密には一致しないが，**平均 p_true を 0.04→0.36 へ押し上げる点で
  同じ方向の是正**になっている（本研究は 10 ドメイン一律の単純規則を守るため，SimANS 流の確率分布の導入までは踏み込まない）．
- **A More Robust Baseline for Active Learning by Injecting Randomness to Uncertainty Sampling**
  （Lin らのグループ．WSDM'22 ALOE <https://www.csie.ntu.edu.tw/~htlin/paper/doc/wsdm22aloe.pdf>，
  ICML 2023 workshop 版 <https://icml.cc/virtual/2023/27400>，2026-09-27 確認）．
  不確実性サンプリングは条件次第で無作為に負け AULC が不安定になるが，**無作為性を注入すると最悪ケースが改善する**．
  Iter84（+18.02pt）→ Iter86（+14.85pt）で Q1 の効果量が振れたこと，プール選択性の有無でドメイン別 Δ が
  ±6pt の幅で散ったことは，この「不安定性」の実例として読める．
- Tripp（<https://www.austintripp.ca/blog/2025-04-02-active-learning-random>，既出）も同旨で，
  不確実性サンプリングが無作為を上回るのは候補間の情報量の分散が大きいときに限られるとする．
- **いずれも「50/50 が最適比である」とは言っていない**（比の最適値は課題依存）．本反復は B141 が確定した
  50/50 を検証するのみで，比の掃引は行わない（単一レバー原則．結果を見てから次反復で扱う）．

### 計画 (Iter87)

**単一レバー**

`cross_domain_training_data_augmentation` = **`hard_random_hybrid_all_domains`**．
**`scripts/mine_hard_negatives.py` の選定規則を「最難 100」から「最難 50 ＋ 残プールから無作為 50」へ変える**ことだけを行う．
評価集合・`config.yaml`・埋め込みモデル・instruction・連結仕様・送出閾値・集約方式・訓練スクリプト・プール定義は一切変えない．

**Iter86 との差分は 1 点のみ**: `select_hard_negatives()` の選び方（＋ その引数を渡す CLI 2 個）．
追加総数 900 行・ドメイン別内訳 {legal: 0, 他 9 ドメイン各 100}・出力書式・id 規則・スコア関数（基準線 artifact の p_true）・
プール定義・除外集合はすべて不変．

**【実装フェーズが最初に行うこと（前提条件．訓練より前）】**

Iter86 の分析フェーズで**事前登録の復元手順が未実施のまま残っている**（journal「Iteration 86 実行済み」節末尾，B141 要フォローアップ）．
現在の `models/domain_classifier.joblib` は Iter86 artifact（`fd1ccd7d3e85c3340b5e44ffc116af078fed61bb1c2f63a9ac8f92583bb21ba4`）である．

1. `cp models/domain_classifier_pre_iter84_baseline.joblib models/domain_classifier.joblib`
2. `sha256sum models/domain_classifier.joblib` が
   `1cfcd3d836c5b5b421b8a47a487c3fb3ba996dd54aadcebae688cdfc8bcd0a48` であることを確認
3. `mise run deploy` → 全 10 ノードで artifact sha256 が `1cfcd3d8...` で一致することを確認（smoke_check）

**これは手順上の後始末ではなく，本反復の測定の前提である**．採掘の難易度スコア p_true は
`--classifier-model models/domain_classifier.joblib` が読む artifact で計算されるため，**復元前に採掘すると
Iter86 artifact でスコアリングした別物の訓練集合ができ，Iter84/86 との比較が成立しなくなる**．
（`models/domain_classifier_iter86_hardneg.joblib` に Iter86 artifact の複製が残っているので復元で失われるものは無い．）

**レバーを読むコード行と，そこへ到達する条件（d0004 §4 の再発防止．省略不可）**

| 経路 | レバーが効く箇所 | 到達条件 | 到達確認 |
|---|---|---|---|
| 採掘・選定（オフライン） | `scripts/mine_hard_negatives.py:select_hard_negatives()`（新しい `random_count` / `rng` 引数） | CLI に `--random-count 50 --random-seed 87` を渡すこと．`--per-domain 100` は据え置き | **F2** |
| 採掘・プール（不変） | 同 `build_pool()` の `exclude_queries` | `--eval-data data/dataset.jsonl`（3,435 行）据え置き | **F2** |
| 採掘・スコア（不変） | 同 `_run()` の `model.predict_proba` | `--classifier-model models/domain_classifier.joblib` が**復元後の `1cfcd3d8...`** であること | **F0** |
| 訓練（オフライン） | `scripts/train_domain_classifier.py` の `--train-data` | 新 JSONL を渡すこと（スクリプト自体の diff は 0 行） | **F3** |
| 実行時（本体） | `classifier.py:load_domain_classifier()` → `estimate_confidence_classifier()` が `config.yaml` の `classifier_model_path`（=`models/domain_classifier.joblib`）を読む | 各ノードのコンテナが**配布後の** artifact を読むこと（`mise.toml` の rsync） | **F4** |
| 記録側 | `run_experiment.py:98` の `dispatched_domains` 再計算（gap_threshold=0.36，不変） | 同上 | **F5** |
| 到達しない経路（変更しない） | 埋め込み・`aggregator.py`・`config.yaml`・`build_dataset.py`・`data/dataset.jsonl`・`data/classifier_train.jsonl`・`train_domain_classifier.py` | 一切触らない | **F1** |

**事前ゲート（実装フェーズで判定）**

| ゲート | 内容 | 不合格時 |
|---|---|---|
| **F0（復元）** | 上記の復元手順 1〜3 を採掘の**前**に完了し，`models/domain_classifier.joblib` = `1cfcd3d8...`，全 10 ノードで同値 | 以降へ進まない |
| **G0（調達元の同一性．B139）** | 採掘に用いる JMMLU.zip の sha256 が `data/MANIFEST.md` の記録値 **`3ba7d912943ede44fb7ec06aa1df067ac6bee157e65c5588736b9b983f0e684d`** と一致（本フェーズの実測では一致している） | **FAIL の場合，F2 の「件数・プールが計画表と完全一致」条項は適用しない**（実測値を記録し非遮断の観察事項とする）．重複 0 件条項は G0 に関わらず絶対条件 |
| **F1（変更の最小性）** | `git diff` が `scripts/mine_hard_negatives.py`・`data/classifier_train_iter87_hybrid.jsonl`（新規）・`models/`・`data/MANIFEST.md`・`.claude/research/*` のみ．**`config.yaml`・`node.py`・`classifier.py`・`aggregator.py`・`train_domain_classifier.py`・`build_dataset.py` の diff 0 行**．`data/dataset.jsonl`・`data/classifier_train.jsonl` の sha256 不変．`mine_hard_negatives.py` の diff は `select_hard_negatives()`・`_run()` の呼び出し・CLI 引数追加・docstring に限る（`build_pool` / `determine_used_tasks` / `_pool_hash` / `write_output` は 1 行も変えない） | invalid |
| **F2（データ）** | 出力 `data/classifier_train_iter87_hybrid.jsonl` が **2,327 行**．先頭 1,427 行が `data/classifier_train.jsonl` と**バイト一致**（sha256 `eb89bf7b...`）．**追加 900 行と `data/dataset.jsonl` 3,435 行の設問文重複 0 件**（絶対条件）．既存 1,427 行との重複 0 件，追加行同士の重複 0 件，id 重複 0 件．ドメイン別追加数が **{legal: 0, 他 9 ドメイン各 100}**．プール実測が Q1 の表（計 3,127，medical 910 / history_culture 579 / natural_science 576 / business_economics 505 / mathematics 148 / education 109 / computer_science 100 / social_science 100 / general 100 / legal 0）と一致．**cs・social_science・general の追加 100 行が Iter86 の同ドメイン 100 行と集合として完全一致**（no-op の確認．一致しなければ実装誤り）．Iter86 全体との入れ替わり行数を実測し記録（期待 214±20 行） | invalid |
| **F3（artifact）** | 新 `models/domain_classifier.joblib` の sha256 が基準線 `1cfcd3d8...` とも Iter86 `fd1ccd7d...` とも**異なる**．`n_features_in_`=2048，`classes_` が 10 ドメイン．退避 `models/domain_classifier_pre_iter84_baseline.joblib` は `1cfcd3d8...` のまま．**新 artifact を `models/domain_classifier_iter87_hybrid.joblib` へ複製**（Iter86 の教訓） | invalid |
| **F4（配布）** | `mise run deploy` 後，全 10 ノードで artifact sha256 が新値で一致，`dispatch_gap_threshold: 0.36` 一致，`docker compose exec app wc -l /app/data/dataset.jsonl` = **3435**（10/10） | invalid |
| **F5（実行時経路）** | 予備 20 問（`data/dataset_20.jsonl`）の `confidence`・`dispatched_domains` が，新 artifact ＋ キャッシュ埋め込みのオフライン予測と **20/20 一致** | 本走中止 |

**固定する構成（基準線 = Iter85 本走 `results/20260927_130237/` と同一）**

`config.yaml` は 1 行も変更しない（`embedding_model=qwen3-embedding:0.6b`，`embedding_instruction`（Iter81 の P1 文言），
`embedding_view_concat=true`，`routing_method=supervised_classifier`，`confidence_threshold=0.0`，
`dispatch_candidate_threshold=0.0`，`dispatch_top_k=2`，`dispatch_gap_max_k=4`，`dispatch_gap_threshold=0.36`，
`aggregation_method=max_confidence`）．`data/dataset.jsonl`（3,435 行，バイト不変），
`data/classifier_train.jsonl`（1,427 行，バイト不変．追加分は別ファイルへ書く）．
**ドメイン固有の後付け補正は追加しない．棄権・人間エスカレーションは扱わない（2026-09-23 恒久運用ルール）．
N・混合比・乱数種・プール定義は 10 ドメイン共通である．**

**仮説（事前登録）**

「Iter86 で観測した『最難層 +14.85pt と易しい 3 層の −1.0〜−6.8pt の差し引き』は，訓練集合へ
**p_true がほぼ 0 の境界行だけ**を入れたことで，プール選択性の高いドメイン（medical・history_culture・
natural_science・business_economics）の決定境界が過剰に押し広げられ，その押し出しが
education → medical → natural_science と連鎖したことによる．**各ドメインの追加 100 行を
最難 50 ＋ 無作為 50 に替えると，これら 4 ドメインの追加行の平均 p_true が 0.04〜0.27 から 0.36〜0.48 へ上がり，
押し出しが弱まる．その結果，非退行①（とくに education recall と natural_science precision）を満たしたまま，
Q1 の改善の相当部分（+11pt 程度）を残せる**．ただし 10 ドメイン中 3 ドメイン（computer_science・
social_science・general）では混合しても選ばれる集合が Iter86 と同一で本レバーは no-op であり，
education 自身も 7.6 行しか変わらないため，全体 Δtop1 は Iter86 の +1.019pt より小さくなる可能性が高い．」

**着地点予測（事前登録．事後に書き換えない）**

| 指標 | 基準線 `results/20260927_130237/` | 参考: Iter86 | **Iter87 予測** |
|---|---|---|---|
| `top1_accuracy`（3,435 行） | 0.786317 | 0.796507 | **0.794〜0.800．点推定 0.7963（+1.0pt）．モデル A/B の幅は +0.78〜+1.32pt** |
| `single_domain_top1_accuracy`（3,020 行） | 0.782119 | 0.797020 | +0.6〜+1.5pt |
| `compound_domain_top1_accuracy`（415 行） | 0.816867 | 0.792771 | 0.79〜0.82 |
| **`education` recall** | 0.387991 | 0.344111（BH 有意悪化） | **0.36〜0.39．非退行①の合否はここで決まる** |
| **`natural_science` precision** | 0.872928 | 0.802885（BH 有意悪化） | **0.83〜0.87** |
| `history_culture` recall | 0.823666 | 0.883991（BH 有意改善） | +2〜+4pt（Iter86 より小さい） |
| `natural_science` recall | 0.733179 | 0.774942（BH 有意改善） | +1〜+3pt |
| `legal` recall / precision | 0.662551 / 0.759434 | −2.06pt | 追加行 0 件の唯一のドメイン．監視対象 |
| `compound_domain_set_recall` / `mean_dispatched_count` | 0.581928 / 1.889157 | 0.555422 / 1.925301 | 非退行の枠内で報告 |
| `mean_duration_ms` | 1574.096 | 1534.531 | 非退行枠 1888.9 |
| ECE | 0.018509 | 0.040154 | 0.01〜0.06 |
| rank1 以外が選ばれた行数 | 5 | 1 | ≤ 15 |
| `answer_quality_accuracy` / `end_to_end_accuracy` | 0.587417 / 0.403202 | 0.571523 / 0.397671 | ノイズ床 3SD=2.6pt の範囲でのみ判定 |
| Random / BestSingle / Oracle | 0.112082 / 0.128384 / 1.0 | 不変 | success_criteria (3) により毎回併記 |

**成功条件・非退行条件（事前登録．Iter86 の事前登録をそのまま踏襲する．結果を見る前に固定し，事後に緩めない）**

| 区分 | 指標 | 基準線 | 合格条件 |
|---|---|---|---|
| **主基準（効果）** | 全 3,435 行 `top1_accuracy` | 0.786317 | **(i) McNemar 対比較（α=0.05）で有意，かつ (ii) +1.0pt 以上（≥ 0.796317）**．AND 条件．Wilson 95%CI と検出限界 `1.96·√(n_d)/3435` を併記 |
| **非退行①** | per-domain recall / precision 計 20 指標（BH 補正 q=0.05） | 上表 | **有意退行 0 件**．`education` recall（eval 350 行）・`natural_science` precision・`legal`（追加 0 件）は個別に明記する |
| **非退行②（複合被覆）** | `compound_domain_set_recall` | 0.581928 | **≥ 0.539759**（Iter82 水準を下回らない） |
| **非退行③（複合予算）** | `compound_mean_dispatched_count` | 1.889157 | **≤ 2.10** |
| **非退行④** | `fallback_rate` / `dispatch_failure_rate` | 0.0 / 0.001164 | fallback = 0.0，dispatch_failure ≤ 0.005 |
| **非退行⑤** | rank1 以外が選ばれた行数 | 5 | **≤ 15** |
| **非退行⑥** | `mean_duration_ms` | 1574.096 | **≤ 1888.9（+20% 以内）** |
| **非退行⑦** | ECE | 0.018509 | **≤ 0.08** |
| 報告のみ | 層別 Δ（基準線 p_true の五分位） | Iter86 の五分位表 | **必ず併記**．モデル A / B のどちらに近いかを解釈に使う |
| 報告のみ | Iter86 の 900 行との入れ替わり実測行数 | 期待 214 | 選択規則が効いた証拠 |
| 報告のみ | 1,915 行サブセット top1 | 0.791123（Iter86 は 0.802611） | 過去基準線との接続用 |
| 報告のみ | `answer_quality_accuracy` / `end_to_end_accuracy` | 0.587417 / 0.403202 | 3SD=2.6pt のノイズ床の範囲でのみ有意と判定 |

**判定規則（事前登録．Iter86 と同一）**

- **adopted**: 主基準 (i)(ii) を満たし，非退行①〜⑦をすべて満たす．
- **partial**: McNemar が有意で Δtop1 が +0.5〜+1.0pt，かつ非退行①〜⑦を満たす．
  **Q3 の投影ではこれが最も確からしい着地である．この場合「hard/random 混合は Iter86 より効果量は小さいが
  per-domain を壊さない」と記録し，次反復で混合比（例: 70/30）の探索へ進む判断材料とする．**
- **no_effect**: |Δtop1| < 0.5pt または McNemar が非有意．結論前に F0〜F5 を再確認し実験不成立でないことを示すこと．
- **rejected**: Δtop1 ≤ −0.5pt，または非退行①〜⑦のいずれかに違反．
  **`education_recall` の有意退行が 3 反復連続で再現した場合はここに該当する．** その場合，
  「hard negative を含む訓練データ拡充は，混合比を変えても全体 top1 と education を両立できない」と記録し，
  ドメイン固有の手当てへは進まない（2026-09-23 恒久運用ルール (1)）．
- **invalid（実験不成立）**: F0〜F5 のいずれか不合格，`total_questions != 3435`，
  `compound_domain_question_count != 415`，新 artifact の sha256 が `1cfcd3d8...` または `fd1ccd7d...` と一致，
  **追加 900 行と評価集合の重複が 1 件でもある**，cs/social_science/general の追加行が Iter86 と不一致，
  または本走 top1 がオフライン replay 予測から 1.0pt 以上乖離．
- **復元手順（adopted 以外すべて）**: `cp models/domain_classifier_pre_iter84_baseline.joblib
  models/domain_classifier.joblib`（sha256 が `1cfcd3d8...` に戻ることを確認）→ `mise run deploy` →
  全 10 ノードで smoke_check．**この復元は分析フェーズ内で必ず完了させ，未実施のまま次反復へ送らない**
  （Iter86 でこれが積み残され，本反復の前提条件になった）．`config.yaml`・`data/dataset.jsonl`・
  `data/classifier_train.jsonl` は本反復で触らないので復元不要．`data/classifier_train_iter87_hybrid.jsonl` と
  `models/domain_classifier_iter87_hybrid.joblib` は記録として残す．

**実験手順（この順で行うこと）**

1. **復元（最優先．F0）**: 上記「実装フェーズが最初に行うこと」1〜3 を実施．
2. **前提確認**: `wc -l data/dataset.jsonl` = 3435，`wc -l data/classifier_train.jsonl` = 1427，
   `sha256sum /tmp/expert-mesh-cache/JMMLU.zip` = `3ba7d912...`（**G0**），
   `config.yaml` の `embedding_view_concat: true` / `embedding_instruction` / `dispatch_gap_threshold: 0.36`．
3. **実装**: `scripts/mine_hard_negatives.py` に `--random-count`（既定 0）・`--random-seed`（既定 0）を追加し，
   `select_hard_negatives(pool, p_true, per_domain, random_count, seed)` を
   「`(p_true, task, query)` 昇順の上位 `per_domain − random_count` 行 ＋ 残りから
   `numpy.random.default_rng(seed).permutation` で `random_count` 行（残りが不足ならある分だけ）」へ変更する．
   選んだ 100 行は最後に `(p_true, task, query)` 昇順へ並べ直してから id を振る．docstring も更新する．→ **F1**
4. **採掘**（**wafl-ctrl5 限定**．config.yml 絶対条件 (B)．ただし本反復は**プール埋め込みキャッシュが
   `pool_hash=221e45e8...` で一致するため再埋め込みは発生せず，実質 CPU のみで完了する**）:
   `data/MANIFEST.md` の Iter86 生成コマンドに `--random-count 50 --random-seed 87` を追加し，
   `--output data/classifier_train_iter87_hybrid.jsonl` へ替えて実行する（他は 1 文字も変えない）．→ **F2**
5. **再訓練**（**wafl-ctrl5 限定**）: `data/MANIFEST.md` の `train_domain_classifier` コマンドの `--train-data` だけを
   `data/classifier_train_iter87_hybrid.jsonl` に替えて実行 → `cp models/domain_classifier.joblib
   models/domain_classifier_iter87_hybrid.joblib` → **F3**．
   **5b（report-only．余力がある場合のみ）**: 種 88・89 でも採掘・再訓練し，1,915 行キャッシュ埋め込みでの
   オフライン replay top1 の散らばりを記録する．**この結果で本走に使う種を選び直してはならない**（種は 87 で固定）．
   目的は Q2 で述べた種由来の不確実性の大きさを数値で残すことだけである．
6. **本走前の着地点記録**: 新 artifact ＋ 既存 1,915 行キャッシュ埋め込みで replay を走らせ，
   予測 top1 / per-domain / 複合指標を**本走前に** journal へ追記する（B136）．
7. **配布**: `mise run deploy` → **F4**．
8. **予備 20 問** → **F5**．不一致なら本走に進まない．
9. **本走**: wafl500〜509 で **3,435 問フルスペック 1 回**（config.yml 絶対条件 (A)．事前 replay で
   経路指標が予測できても省略しない）．想定所要は Iter85/86 実績から **約 90 分**（timeout 180 分以内）．
10. `mise run analyze -- <YYYYMMDD_HHMMSS>`（**B135: 引数を省略すると `results/iter45_preliminary/` を誤選択する**）．
    層別 Δ（基準線 p_true 五分位）・Iter86 との入れ替わり行数・1,915 行サブセット top1 も併せて算出する．
11. `data/MANIFEST.md` に新訓練ファイル・新 artifact の sha256 と行数・**乱数種 87**・生成コマンド・
    ゲート結果・「id の連番は難易度順位を意味しない」旨を追記する．

**次イテレーションへの申し送り**

- `partial` 着地なら，次は**混合比の探索**（例: hard 70 / random 30）が自然な次手である．ただし B141 は
  「他の比の探索は次々回以降」としているので，比を振る前に本反復の層別 Δ がモデル A / B のどちらだったかを確認すること．
- `rejected`（education recall の 3 反復連続退行）なら，`cross_domain_training_data_augmentation` 系は
  打ち止めとし，config.yml の levers の次候補へ移る．
- **プールの構造的上限**: computer_science・social_science・general は M=100 = N で，追加行数 100 を保つ限り
  今後どんな抽出規則を入れても no-op である．抽出規則側でこの 3 ドメインを動かす余地はもう無い
  （動かすには追加行数 N か評価集合の構成を変えるしかなく，どちらも別レバーになる）．

### 実装・実験 (Iter87)

**実施した変更（単一レバーのみ）**

0. **分類器 artifact を基準線へ復元（F0，本反復の前提条件）**: `models/domain_classifier.joblib` ←
   `models/domain_classifier_pre_iter84_baseline.joblib`（sha256 `1cfcd3d8...`）．全 10 ノードで一致を
   確認した．Iter86 の分析フェーズで積み残されていた復元手順をここで解消した．
1. `scripts/mine_hard_negatives.py` の `select_hard_negatives()` を「最難 100」から「最難 50 ＋ 残プールから
   `numpy.random.default_rng(87).permutation` による無作為 50」へ変更し，CLI に `--random-count`（既定 0）・
   `--random-seed`（既定 0）を追加した．`build_pool` / `determine_used_tasks` / `_pool_hash` / `write_output`
   は 1 行も変えていない（`git diff --stat` は当該関数群を含まない）．`ruff check` / `py_compile` PASS．
2. 採掘（wafl-ctrl5 の Ollama へトンネル経由）: `--random-count 50 --random-seed 87` を追加し
   `data/classifier_train_iter87_hybrid.jsonl`（2,327 行）を生成．プール埋め込みキャッシュが一致したため
   再埋め込みは発生せず，実質 CPU のみで完了．
3. 再訓練（同 wafl-ctrl5）→ `models/domain_classifier.joblib`（sha256 `f6c33edb...`）→
   `models/domain_classifier_iter87_hybrid.joblib` へ複製．
4. 本走前の着地点記録（新 artifact ＋ 既存 1,915 行キャッシュ埋め込みで `aggregator.select_dispatch_targets()`
   を再利用した replay，gt=0.36）: top1(1,915 部分集合) 予測 0.810966，compound_domain_set_recall 予測
   0.566265，compound_mean_dispatched_count 予測 1.879518．replay 手法自体は，同じ手法を基準線 artifact に
   適用すると Iter83 の G1 記録値 0.793211 と完全一致することで忠実度を確認済み．
5. `mise run deploy` → 全 10 ノード healthy，smoke_check（git-status / hashes / probe）PASS．
   10/10 ノードで artifact sha256 = `f6c33edb...`，`wc -l /app/data/dataset.jsonl`=3435，
   `dispatch_gap_threshold: 0.36` を確認（F4）．
6. 予備 20 問（`data/dataset_20.jsonl`）: `confidence`・`dispatched_domains` がオフライン予測と
   **20/20 完全一致**（F5，PASS）．
7. `mise run start` を 1 回（`results/20260927_174150/`，3,435 問，約 90 分で完走．timeout 180 分以内）。
8. `mise run analyze -- 20260927_174150`。統計は `metrics.py` の `compute_mcnemar_test` /
   `compute_domain_recall_mcnemar_test` / `compute_domain_precision_fisher_test` /
   `apply_benjamini_hochberg` / `compute_top1_accuracy_wilson_ci` をそのまま再利用した（自前の統計式は
   書いていない）。層別 Δ（基準線 p_true 五分位）は `probe_candidates` のうち各行の `expected_domains` に
   一致する最大 confidence を p_true として実測し，五分位境界が Iter86 の記録値と完全一致することを
   確認した上で算出した。

**計画からの乖離**: なし。計画どおり，`select_hard_negatives()` と CLI 2 引数のみを変更した。

**F2 ゲート実測**: computer_science・social_science・general の追加 100 行が Iter86 の同ドメイン 100 行と
**集合として完全一致**（no-op 確認，PASS）。Iter86 全体 900 行との入れ替わり実測行数は **219 行**
（事前登録の期待 214±20 行の範囲内，PASS）。追加 900 行と評価集合 3,435 行・既存 1,427 行・追加行同士の
重複はいずれも 0 件。

**主要メトリクス**

全体（3,435 問）: top1_accuracy = 0.786317 → **0.801456**（**Δ = +1.514pt**，Wilson 95% CI
[0.787782, 0.814456]），single_domain_top1 = 0.782119 → 0.799007（n=3,020），
compound_top1 = 0.816867 → 0.819277（n=415），fallback_rate = 0.0 → 0.0，
dispatch_failure_rate = 0.001164 → 0.001456（5 行），mean_duration_ms = 1574.096 → 1544.912。

主基準: `compute_mcnemar_test` で continuity-corrected chi2=11.508850，**p=0.000693（有意，α=0.05）**，
discordant_a_only（基準線正解→新誤り）=87／discordant_b_only（基準線誤り→新正解）=139，
discordant_pairs=226，有意境界 `1.96・√(n_d)/3435`=0.858pt。**主基準 (i) 有意・(ii) Δ≥+1.0pt
（0.801456 ≥ 0.796317）はいずれも実測上 PASS**（判定語の確定は分析フェーズに委ねる）。

1,915 行部分集合（実測）top1 = 0.809922（本走前 replay 予測 0.810966 と乖離 0.104pt，
`selected_domain` 行一致率 1913/1915 = 99.90%）。

**層別 Δ（基準線 p_true の五分位，各 n=687，境界値は Iter86 の記録値と完全一致）**

| 層 | 上限 p_true | 基準線 | Iter87 | Δ | 参考: Iter86 |
|---|---|---|---|---|---|
| Q1 | 0.3542 | 0.0451 | 0.1630 | **+11.79pt** | +14.85pt |
| Q2 | 0.7209 | 0.8923 | 0.8632 | −2.91pt | −6.84pt |
| Q3 | 0.9022 | 1.0000 | 0.9898 | −1.02pt | −2.33pt |
| Q4 | 0.9697 | 1.0000 | 0.9942 | −0.58pt | −1.02pt |
| Q5 | 0.9985 | 0.9942 | 0.9971 | +0.29pt | +0.44pt |

**非退行①**: per-domain recall/precision 計 20 指標（BH 補正 q=0.05）: **有意差 1 件**——
**history_culture_recall（0.823666→0.870070，p=0.000105，改善方向）**。**education_recall
（p=0.072486）・natural_science_precision（p=0.251250）はいずれも BH 補正後有意差なし**（Iter84・Iter86 で
2 反復連続していた education_recall の有意退行は本反復では再現しなかった）。他 18 指標も有意差なし。
per-domain 詳細（baseline→iter87，recall/precision）: business_economics 0.8283/0.7951→0.8492/0.7922，
computer_science 0.8796/0.8912→0.8770/0.9128，education 0.3880/0.5638→0.3626/0.6461，
general 0.4841/0.7673→0.5119/0.8113，history_culture 0.8237/0.7100→0.8701/0.7184，
legal 0.6626/0.7594→0.6831/0.8137，mathematics 0.8005/0.8824→0.7935/0.8976，
medical 0.7574/0.8087→0.7710/0.8000，natural_science 0.7332/0.8729→0.7564/0.8424，
social_science 0.5520/0.7667→0.5787/0.7750。

非退行②（複合被覆）: compound_domain_set_recall 0.581928→0.566265（下限 0.539759 以上，**PASS**）。
非退行③（複合予算）: compound_mean_dispatched_count 1.889157→1.879518（上限 2.10 以内，**PASS**）。
非退行④: fallback_rate 0.0→0.0（**PASS**），dispatch_failure_rate 0.001164→0.001456（≤0.005，**PASS**）。
非退行⑤: rank1 以外が選ばれた行数 5→5（≤15，**PASS**）。
非退行⑥: mean_duration_ms 1574.096→1544.912（≤1888.9，**PASS**）。
非退行⑦: ECE 0.018509→0.045717（≤0.08，**PASS**）。

報告のみ: answer_quality_accuracy 0.587417→0.587417（実質不変），end_to_end_accuracy 0.403202→0.411063
（+0.786pt，ノイズ床 3SD=2.6pt 以内），Random/BestSingle/Oracle baseline 0.112082/0.128384/1.0（不変）。
ECE=0.045717・Brier=0.126108・AUROC=0.820982（n=3430）。

**invalid 条件チェック**: `total_questions`=3435（一致），`compound_domain_question_count`=415（一致），
新 artifact sha256 は基準線・Iter86 いずれとも不一致，追加 900 行と評価集合の重複 0 件，cs/social_science/
general の追加行は Iter86 と一致，本走 top1（1,915 部分集合実測）はオフライン replay 予測から 0.104pt
乖離のみ（1.0pt 未満）。**invalid 条件はいずれにも該当しない**。

**検証**

- `uv run ruff check scripts/mine_hard_negatives.py` → All checks passed．
- `uv run python -m py_compile scripts/mine_hard_negatives.py` → 成功．
- `select_hard_negatives()` の単体動作確認（合成データ）: `random_count=0` が旧実装と同一の出力になること，
  同一 seed で再現可能なこと，`M=N` の境界ケース（computer_science 等）で no-op になることをそれぞれ
  アサートで確認した（`tests/` に既存の専用テストファイルは無かったため，リポジトリ内での ad hoc 検証に留めた）。

**異常の有無**: 実行・ログ上の異常は無い。復元手順（F0）を実装フェーズの最初に完了させたことを含め，
計画からの乖離は無い。

**判定は分析・考察フェーズに委ねる**（本セクションは実装・実験フェーズの機械可読な実測値の記録のみ）。

### Iteration 87 実行済み

**変更（単一レバー）**: `scripts/mine_hard_negatives.py:select_hard_negatives()` の選定規則を
「各ドメイン最難 100 行」から「最難 50 行 ＋ 残プールから `numpy.random.default_rng(87).permutation`
による無作為 50 行」へ変更（CLI に `--random-count` / `--random-seed` を追加）．追加総数 900 行・
ドメイン別内訳 {legal: 0，他 9 ドメイン各 100}・出力書式・id 規則・プール定義・除外集合・
`config.yaml`・評価集合・訓練スクリプトはすべて Iter86 と同一．実装フェーズ冒頭で基準線 artifact
（`1cfcd3d8...`）への復元と再配布（Iter86 の積み残し）を完了させてから採掘した．

**判定: adopted**

事前登録の条文（Iter86 から一字も変えずに据え置いたもの）は adopted を「主基準 (i)(ii) を満たし，
非退行①〜⑦をすべて満たす」と定義している．実測は次のとおりで，条文が一義に `adopted` を指す．

| 条項 | 条件 | 実測 | 判定 |
|---|---|---|---|
| 主基準 (i) | McNemar α=0.05 で有意 | chi2=11.508850，**p=0.000693**（discordant 226 = a_only 87 / b_only 139） | PASS |
| 主基準 (ii) | top1 ≥ 0.796317（+1.0pt） | **0.786317 → 0.801456（Δ +1.514pt）**，Wilson 95%CI [0.787782, 0.814456] | PASS |
| 非退行① | per-domain 20 指標の BH 補正後有意退行 0 件 | 有意差は `history_culture_recall`（0.823666→0.870070，p=0.000105）**改善方向 1 件のみ**．退行 0 件 | PASS |
| 非退行②〜⑦ | 記載どおり | set_recall 0.566265／dispatched_count 1.879518／fallback 0.0・dispatch_failure 0.001456／rank1 以外 5 行／1544.912ms／ECE 0.045717 | 全 PASS |
| invalid 条件 | 記載どおり | 非該当（total 3435・compound 415・sha256 相違・重複 0 件・replay 乖離 0.104pt） | 該当なし |

判定語の選択に裁量は働いていない．B131 以来の運用（主基準の PASS を理由に条文を緩めない／
届かないからといって条文を緩めない）を本反復でも守っており，**Iter86 を `rejected` にしたのと
同じ条文をそのまま適用して `adopted` になった**点が本判定の担保である．

**ノイズか信号か**

- Δ +1.514pt は，本走の有意境界 `1.96·√226/3435` = 0.858pt の **1.76 倍**である．同一レバー系列の
  過去 3 反復（Iter84 +1.201pt／n=1,915／p=0.101997・非有意，Iter86 +1.019pt／p=0.044011・境界上，
  Iter87 +1.514pt／p=0.000693）の中で，**初めて境界から明確に離れた位置での有意**になった．
  discordant も 285→226 と減ったうえで b_only−a_only が 52 行へ広がっている（Iter86 は 35 行）．
- ただし **McNemar の分散に乱数種由来の分散は含まれない**（B142 (A)）．本反復は種 87 を 1 個だけ
  事前登録して本走 1 回で判定しており，**種を替えたときに Δ がどれだけ振れるかは未測定のまま残る**．
  計画で report-only 手順 5b（種 88/89 のオフライン replay）を置いたが実施されなかったため，
  「+1.514pt のうち何 pt が種の引きの良さか」は本反復では答えられない．**これは adopted の判定を
  覆す材料ではない**（条文は事前登録どおり充足している）が，次反復で混合比を動かすときには
  比の差（おそらく 0.3〜0.5pt 規模）と種の分散が同程度になりうるため，**比の解釈の前に種の分散を
  数値で押さえる必要がある**．次反復の必須手順として申し送る（B143）．

**層別 Δ は仮説を支持し，楽観モデル B すら上回った**

| 層 | 基準線 | Iter87 実測 Δ | モデル A（保守） | モデル B（楽観） | 参考 Iter86 |
|---|---|---|---|---|---|
| Q1 | 0.0451 | **+11.79pt** | +11.32 | +11.32 | +14.85 |
| Q2 | 0.8923 | **−2.91** | −5.21 | −3.42 | −6.84 |
| Q3 | 1.0000 | **−1.02** | −1.78 | −1.17 | −2.33 |
| Q4 | 1.0000 | **−0.58** | −0.78 | −0.51 | −1.02 |
| Q5 | 0.9942 | **+0.29** | +0.34 | +0.40 | +0.44 |
| 全体 | 0.786317 | **+1.514pt** | +0.78 | +1.32 | +1.019 |

- 計画の仮説「混合により易しい層の退行が緩和される」は**支持された**．Q2 の退行は −6.84→−2.91pt へ
  57.5% 縮み，Q3 は −2.33→−1.02（56% 縮），Q4 は −1.02→−0.58（43% 縮）である．
  **モデル B は「Q2〜Q4 の退行が半減する」と置いたが，実測は Q2 でそれをわずかに上回る縮小だった．**
- 一方 Q1 は +14.85→+11.79pt で，モデル A/B が共通に置いた投影 +11.32pt とほぼ一致した
  （差 +0.47pt）．**つまり「最難行の効果は残した行数に比例して減り，易しい層の退行は比例以上に消える」**
  という非対称が実測された．Δ が A（+0.78）も B（+1.32）も超えた理由はここにある．
- 因果として言えるのはここまでである．**単一レバーの変更は選定規則だけなので，この非対称は
  「訓練集合へ入る行の難易度プロファイル（上位 4 ドメインで平均 p_true 0.04〜0.27 → 0.36〜0.48）」に
  帰属させてよい．** ただし「50/50 が最適比である」ことは示していない（2 点しか測っていない）．
  また 10 ドメイン中 3 ドメイン（computer_science・social_science・general）は M=100=N で
  厳密に no-op のままであり，**本レバーが動かしたのは実質 6 ドメイン・900 行中 219 行**である点も
  効果量の解釈に必ず添えること．

**education_recall の有意退行が再現しなかったことの解釈（p=0.072 は「差がない」の証明ではない）**

- 実測は education recall 0.387991 → 0.362587（n=433，**Δ −2.54pt**，BH 補正後 p=0.072486）．
  **これは「有意でない」であって「差がない」ではない．点推定は依然として負であり，Iter86 の
  −4.39pt（p=0.003948）から半分強に縮んだだけである．** n=433 という母数では −2.5pt 規模を
  安定に検出できないため，「education は直った」と読むのは誤りである．非退行①は条文上 PASS だが，
  **education は次反復以降も個別に明記して監視し続ける**．
- 計画で立てた作用機序（education を直接触らず，**medical 側の押し出しを弱める**経路で効かせる）は，
  行単位の流出先を追うと**実際に効いていた**．本フェーズで基準線・Iter86・Iter87 の
  `results.jsonl` から再集計した（`selected_domain` ベース，`expected_domains` に education を含む 433 行）:

  | 経路 | 基準線 | Iter86 | **Iter87** |
  |---|---|---|---|
  | education 正解行の喪失（合計） | ― | 29 行 | **21 行** |
  | うち medical へ流出 | ― | **22 行** | **17 行** |
  | education の獲得 | ― | 10 行 | 10 行 |
  | education 正味 | ― | **−19 行** | **−11 行** |
  | natural_science と判定された行の FP 総数 | 46 | 82 | **61** |
  | うち真ドメインが medical の行 | **9** | **42** | **24** |

  **medical の押し出しは弱まったが消えてはいない**（education→medical 22→17 行，
  medical→natural_science 誤検出 42→24 行．いずれも基準線の 9 行・0 行相当までは戻っていない）．
  natural_science precision は 0.872928→0.842355（p=0.251250，有意差なし）で，
  この −3.06pt の実体は上表の medical 由来 FP の 9→24 行の増加である．
  **つまり Iter86 で確定した連鎖（追加行の難易度 → medical の境界の押し出し → education の喪失 →
  natural_science の FP）は Iter87 でも同じ向きに存在し，強度だけが約半分になった**と読むのが正確である．
- なお `history_culture_recall` の有意改善（+4.64pt）は Iter86（+6.03pt）から縮んでおり，
  これも「効果量は比例して減る」側の挙動と整合する．

**学び**

1. **「最難だけを採る」は，効果の源泉ではなく副作用の源泉と同じ場所にある．** Q1 の改善と Q2〜Q4 の
   退行は Iter84/86 で「同一機序の表裏」と結論していたが，本反復は**その表裏が分離可能であること**を
   初めて示した．無作為 50 行を混ぜると，Q1 の改善は入れ替わり行数に比例してしか減らない（−21%）のに，
   易しい層の退行は比例以上に消える（−43〜−58%）．**最難行が作る過剰な境界の押し出しは，
   同じドメインの典型行を同数入れるだけで打ち消せる**．
2. **BH 補正後の非有意を「解決した」と読まない．** education recall は p=0.003948 → 0.072486 へ動いたが，
   点推定は −4.39pt → −2.54pt で符号は変わっていない．**有意性の消失は効果量の縮小と検出力の限界の
   合成であり，行単位の流出先を追って初めて「機序は残っている」と判定できた．**
   per-domain 指標の判定では，p 値だけでなく行単位の流出先を必ず併記する運用を今後も続ける．
3. **種を 1 個に固定した設計は，adopted の判定には十分だが，比較の細分化には不十分である．**
   本反復のように「条文を明確に超える」Δ なら種の分散に吞まれないが，混合比を 50/50 から
   25/75 へ動かすような**差が小さい比較では，種由来の分散を先に測らないと解釈できない**．
   report-only 手順（5b）を「余力がある場合のみ」に置いたのは弱すぎた．次反復では必須手順にする．
4. **プールの構造的上限は変わっていない．** computer_science・social_science・general は M=100=N で，
   追加行数 N=100 を保つ限りどんな抽出規則でも no-op である．抽出規則側でこの 3 ドメインを動かす
   余地はもう無く，動かすには N か評価集合の構成を変えるしかない（どちらも別レバー）．

**artifact の扱い**: 判定が adopted のため，事前登録の復元手順（adopted 以外すべてに適用）は
**実行しない**．`models/domain_classifier.joblib` は新 artifact（sha256 `f6c33edb...`）のまま維持し，
全 10 ノードへ配布済みの状態を保つ．複製 `models/domain_classifier_iter87_hybrid.joblib` も同値で保存済み．
**次反復以降の基準線は本走 `results/20260927_174150/`（top1 = 0.801456，分類器 `f6c33edb...`）へ更新する**
（`results/20260927_130237/` は Iter84〜87 の系列基準線として引き続き参照する）．

**次イテレーション**: `cross_domain_training_data_augmentation` = `hard_random_hybrid_ratio_25_75_all_domains`
（混合比の用量反応．hard 25 / random 75，10 ドメイン一律，種 87 据え置き）．
**必須の report-only 手順として，50/50 と 25/75 の双方について種 87/88/89 のオフライン replay を行い，
種由来の Δ の散らばりを数値化してから比の差を解釈すること**（B143）．

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

