## Iteration 88: hard negative 混合比の用量反応（25/75）と乱数種ばらつきの計測

### 調査 (Iter88)

本反復のレバーは backlog B143 (d) で `cross_domain_training_data_augmentation` =
`hard_random_hybrid_ratio_25_75_all_domains` に確定済みで，選定の裁量は無い．各ドメインの追加 100 行を
「最難 25 ＋ 残プールから無作為 75」へ変え，追加総数 900 行・ドメイン別内訳（legal 0，他 9 ドメイン各 100）・
出力書式・id 規則・乱数種 87 は Iter87 と同一に保つ．したがって調査の問いは 5 つである．
**(Q1) 実装差分は本当に CLI 引数だけで済むか．(Q2) 採掘スコアに使う artifact をどれにするか．
(Q3) 25/75 のプール実測（入れ替わり行数・難易度プロファイル）．(Q4) 定量的な事前投影と検出力．(Q5) 関連研究．**

**Q1: コード変更は 0 行．`--random-count 50` を `75` に替えるだけで足りる（実コードを Read して確認）**

`scripts/mine_hard_negatives.py:select_hard_negatives()`（L253-300）は Iter87 で既に
`random_count` / `seed` を受け取る実装になっており，`hard_count = max(per_domain - random_count, 0)`，
残プールを `(p_true, task, query)` の全順序へ整列してから `numpy.random.default_rng(seed).permutation` で
添字を引く．CLI にも `--random-count`（既定 0）・`--random-seed`（既定 0）が既にある（L371-381）．
**したがって Iter88 の変更は `data/MANIFEST.md` の Iter87 生成コマンドの `--random-count 50` を `75` に，
`--output` を `data/classifier_train_iter88_hybrid2575.jsonl` に替えるだけであり，リポジトリのソース差分は 0 行になる**
（`mine_hard_negatives.py` を含め，`scripts/` 配下の `git diff` は 0 行であることを F1 ゲートで要求する）．
`rng` はドメイン名の昇順に 1 回ずつ消費されるため，同じ seed でも `random_count` が変われば引かれる行は変わる（意図どおり）．

**Q2: 採掘のスコア artifact は基準線 `models/domain_classifier_pre_iter84_baseline.joblib`（`1cfcd3d8...`）を
CLI で明示指定する．実機に配布済みの Iter87 artifact（`f6c33edb...`）は触らない**

用量反応（hard 100% / 50% / 25%）を比較するには，**3 点すべてで p_true のランキングが同一**でなければならない．
Iter86（100/0）・Iter87（50/50）はいずれも `1cfcd3d8...` でスコアリングしている（Iter87 は実装フェーズの冒頭で
`models/domain_classifier.joblib` をこれへ復元してから採掘した）．Iter88 も同じ artifact を使う必要がある．
一方 B143 (c) は「`models/domain_classifier.joblib` = `f6c33edb...` を全 10 ノードへ配布済みの状態を維持する」と
定めており，これが本反復の**基準線側の実行時 artifact** である．
**この 2 つは `--classifier-model models/domain_classifier_pre_iter84_baseline.joblib` を明示指定することで両立する**
（Iter87 のように `models/domain_classifier.joblib` を上書き復元する必要はなく，復元→再配布→再復元の往復も不要になる）．
`_run()`（L347）は `joblib.load(args.classifier_model)` を読むだけで，他の経路からは参照しない．

**この選択の妥当性は本フェーズで実測検証した**（下記 Q3 のプローブ）．`1cfcd3d8...` で p_true を再計算し
`--random-count 0` で選定すると `data/classifier_train_iter86_hardneg.jsonl` の追加 900 行と**差分 0 行**，
`--random-count 50 --random-seed 87` で選定すると `data/classifier_train_iter87_hybrid.jsonl` の追加 900 行と
**差分 0 行**で再現した．採掘系が決定論的であること，および本フェーズのプローブが実装と一致していることが裏付けられた．

**Q3: プール実測（CPU のみ，実機ノード不使用）— 入れ替わりは Iter87 比 309 行，ただし 3 ドメインは今回も厳密に no-op**

`determine_used_tasks` / `build_pool` / `select_hard_negatives` を直接呼び，JMMLU.zip
（sha256 `3ba7d912943ede44fb7ec06aa1df067ac6bee157e65c5588736b9b983f0e684d`，**MANIFEST 記録値と一致＝ G0 は現時点で PASS**）と
プール埋め込みキャッシュ（`pool_hash=221e45e8...`，3,127 行，一致）から実測した．
`data/dataset.jsonl`（3,435 行）・`data/classifier_train.jsonl`（1,427 行）は Iter86/87 から 1 バイトも変わっていないため，
**プールは Iter86/87 と完全に同一（計 3,127 行）**である．

| ドメイン | プール M | 最難 25 の平均 p_true | 無作為 75 の平均 p_true | 追加 100 行の平均 p_true（Iter87 → **Iter88**） | Iter86 比 入れ替わり | 期待値 `75(M−100)/(M−25)` | **Iter87 比 入れ替わり** |
|---|---|---|---|---|---|---|---|
| medical | 910 | 0.0101 | 0.7038 | 0.3616 → **0.5304** | 70 | 68.6 | **71** |
| history_culture | 579 | 0.0407 | 0.7618 | 0.4279 → **0.5815** | 67 | 64.8 | **68** |
| natural_science | 576 | 0.0531 | 0.8102 | 0.4790 → **0.6209** | 67 | 64.8 | **65** |
| business_economics | 505 | 0.0268 | 0.7608 | 0.4450 → **0.5773** | 64 | 63.3 | **66** |
| mathematics | 148 | 0.4581 | 0.9453 | 0.8141 → **0.8235** | 25 | 29.3 | **31** |
| education | 109 | 0.1171 | 0.6028 | 0.4783 → **0.4814** | 7 | 8.0 | **8** |
| computer_science | 100 | 0.4684 | 0.9295 | 0.8142 → **0.8142（不変）** | **0** | **0.0** | **0** |
| social_science | 100 | 0.1826 | 0.7216 | 0.5869 → **0.5869（不変）** | **0** | **0.0** | **0** |
| general | 100 | 0.1202 | 0.8674 | 0.6806 → **0.6806（不変）** | **0** | **0.0** | **0** |
| legal | 0 | ― | ― | ―（追加 0 件） | 0 | ― | 0 |
| **合計** | **3,127** | | | **0.5606 → 0.6330** | **300** | **298.9** | **309** |

- **computer_science・social_science・general は M = 100 = N のため，今回も自由度がゼロで厳密に no-op である**．
  最難 25 を除いた残り 75 行が必要数ちょうどなので `draw = min(75, 75) = 75` となり，選ばれる 100 行は
  Iter86・Iter87 と**集合として完全に一致する**（実測の入れ替わり 0 行）．
  **比をどう振ってもこの 3 ドメイン（900 行中 300 行，33.3%）は今後一切動かない**．
  education も M=109 で入れ替わりは 8 行にとどまる．Iter87 の申し送り「プールの構造的上限」がそのまま該当する．
- 入れ替わり期待値は超幾何分布による．Iter88 の選定 100 行のうち，残プール（M−25 行）から引く 75 行に対し，
  Iter86 の top-100 のうち残プールに属するのは 75 行（ランク 26〜100）なので，
  期待重複 = 25 + 75·75/(M−25)，期待入れ替わり = 75(M−100)/(M−25)．**Iter87 の選定 100 行も残プール内に
  ちょうど 75 行（ランク 26〜50 の 25 行 ＋ 無作為 50 行）を持つため，Iter87 比の期待入れ替わりも同じ式になる**．
  実測（300 / 309 行）はいずれも期待値 298.9 の近傍にあり，整合する．
- **難易度の「用量」は追加 900 行の平均 p_true で測ると 0.4491（100/0）→ 0.5606（50/50）→ 0.6330（25/75）で，
  刻みは +11.15pt → +7.24pt と劣線形になる**（残プールを深く引くほど中央値に近づくため）．
  つまり 50→25 への移動は，100→50 への移動より**用量の増分が 0.65 倍しかない**．これは Q4 の投影の前提になる．

**Q4: 事前投影 — Δtop1（Iter87 基準線比）は −0.6 〜 +0.5pt，点推定 +0.2〜+0.3pt．主基準到達確率は 5% 程度**

基準線は Iter87 本走 `results/20260927_174150/`（top1 = 0.801456）だが，層別 Δ の実測は Iter86・Iter87 とも
**pre_iter84 基準線（0.786317）比**で記録されているので，まず pre_iter84 比で投影し，最後に 1.514pt を引いて
Iter87 比へ変換する．五分位は基準線 p_true で切った Iter86 の境界値（Q1 ≤0.3542 / Q2 ≤0.7209 / Q3 ≤0.9022 /
Q4 ≤0.9697），各 n=687．

| 層 | Iter86（hard 100%） | Iter87（hard 50%） | **A: 飽和（下振れ）** | **B: 用量線形（点推定）** | **C: 退行消失（上振れ）** |
|---|---|---|---|---|---|
| Q1 | +14.85 | +11.79 | +8.73 | +9.8 〜 +10.3 | +10.26 |
| Q2 | −6.84 | −2.91 | −2.91 | −0.4 〜 −0.9 | 0.00 |
| Q3 | −2.33 | −1.02 | −1.02 | −0.2 〜 −0.4 | −0.20 |
| Q4 | −1.02 | −0.58 | −0.58 | −0.3 〜 −0.4 | −0.20 |
| Q5 | +0.44 | +0.29 | +0.29 | +0.2 | +0.22 |
| **Δtop1（pre_iter84 比）** | **+1.019** | **+1.514** | **+0.90** | **+1.76 〜 +1.84** | **+2.02** |
| **Δtop1（Iter87 基準線比）** | ― | 0（基準線） | **−0.61pt** | **+0.25 〜 +0.32pt** | **+0.50pt** |

- モデル A は「Q1 の改善は hard 行数の対数に比例して減衰する（14.85 → 11.79 の減り方を h の対数で外挿）が，
  Q2〜Q4 の退行はこれ以上は縮まない（Iter87 で床に達した）」とする保守側の端点．
- モデル B は 2 実測点の線形外挿で，hard 比 h で外挿した場合（+1.76）と Q3 の平均 p_true 用量で外挿した場合（+1.84）の
  両方を併記した．両者が近い値に収束することが点推定の根拠である．
- モデル C は Q2〜Q4 の退行が 25/75 でほぼ消えるとする楽観側の端点．
- **どのモデルでも主基準 (ii)（Iter87 基準線比 +1.0pt 以上，すなわち top1 ≥ 0.811456）には届かない．**
  用量反応が 50/50 で頭打ちに近いこと自体が本反復の測る対象であり，投影が届かないからといって
  **判定規則は緩めない**（B131 以来の運用．Iter86→87 でも同じ条文を据え置いた）．

**検出力**: Iter87 vs 基準線の discordant は 226/3,435 だった．Iter88 vs Iter87 は入れ替わり行数こそ多い（309 行）が
用量差は小さいので，discordant を **n_d ≈ 200〜280（中心 240）** と見積もる．
SE = √n_d/3435 = **0.451pt**，有意境界 1.96·SE = **0.884pt**．

| 真の Δ | z = Δ/SE | **McNemar 有意（主基準 i）の検出力** | **(i) AND (ii) の到達確率** |
|---|---|---|---|
| A: −0.61pt | −1.35 | 約 27%（退行方向） | 0% |
| B: +0.28pt | +0.62 | **約 10%** | **約 5%** |
| C: +0.50pt | +1.11 | 約 20% | 約 6% |

- **したがって最も確からしい着地は `no_effect`（|Δtop1| < 0.5pt または McNemar 非有意）である．**
  これは Iter86 の事前登録に既に定義済みの帯であり，新しい条文を要しない．
  `no_effect` に着地した場合の研究上の意味は明確で，**「hard/random 混合比の用量反応は 50/50 付近で頭打ちであり，
  これ以上ランダム側へ振っても top1 は伸びない」**と記録でき，B143 (根拠) が予告した次の一手
  （純無作為 0/100 の対照実験）へ進む判断材料になる．
- **本反復の主たる情報価値は「比の差を検出すること」ではなく，(a) 用量反応曲線の 3 点目を測ること，
  (b) 下記 5b で種由来の Δ の散らばりを数値化し，そもそも 0.3〜0.5pt 規模の比の差が測定可能かを確定することにある．**
- 非退行①の投影: education の追加行は 8 行しか変わらず平均 p_true も 0.4783 → 0.4814 でほぼ不変なので，
  education 側から新たな退行が生じる理由は無い．medical の追加行の平均 p_true は 0.3616 → 0.5304 へさらに上がるため，
  Iter86 で観測された「medical の押し出しが education を削り natural_science の FP を増やす」経路はさらに弱まる方向である．
  **education_recall は 0.35〜0.38（Iter87 基準線 0.3626），natural_science_precision は 0.83〜0.86（同 0.8424）と投影する．**
  ただし追加行の平均 p_true が 0.63 まで上がると，今度は**訓練集合がプールの典型分布に寄って境界情報が薄まる**方向の
  リスクがあり，Q1 層（最難層）の取りこぼしが増えれば全ドメインで薄く recall が下がりうる．非退行①の合否はそこで決まる．

**Q5: 関連研究 — 混合比の用量反応は「p>0 でさえあれば比の値には鈍感」というのが実測報告の一致した所見**

- **Understanding Hard Negatives in Noise Contrastive Estimation**（Wenzheng Zhang, Karl Stratos, NAACL 2021．
  <https://arxiv.org/abs/2104.06245> / <https://karlstratos.com/papers/naacl21hard.pdf>，2026-09-27 確認）．
  Appendix A に**まさに本反復と同じ形の用量反応表**がある．負例のうち hard の割合 p を 0/25/50/75/100% と振ったときの
  top-64 validation recall は，DUAL 91.08/92.18/91.75/92.24/92.05，MULTI-8 91.13/92.74/92.76/93.41/93.27，
  SOM 92.51/94.13/94.66/94.37/94.54．**p=0（純無作為）からの改善は明確（+1.0〜+2.0pt）だが，p=25 と p=50 の差は
  3 設定で +0.43 / −0.02 / −0.53pt と符号が揃わず，平均は −0.04pt でほぼゼロ**である．
  著者自身が「the exact choice of p > 0 is not as important」「performance is not sensitive to the value of p」と述べ，
  p=50 を選んだ理由も「無作為を少し混ぜると小さいが一貫した改善が出るから」という程度の根拠に留めている．
  **本研究の hard 50% → 25% は，この表の p=50 → p=25 に対応する．文献側の実測も差はノイズ規模であり，Q4 の投影
  （+0.2〜+0.3pt，検出力 10%）と整合する．** 同時に，この非単調性（p=25 が p=50 を上回る設定と下回る設定が混在する）は
  **比の差が実行ごとのばらつきに埋もれる規模であることの直接の証拠**であり，5b の種ばらつき計測を必須手順に置く根拠になる．
- **false negative / 過度に難しい負例による劣化**: RocketQA（Qu et al., NAACL 2021）が MSMARCO の
  top-retrieved 未ラベル passage を人手確認して 70% が実は正例だったと報告して以来，
  「hard negative を無選別に採ると mislabeled 行を優先的に拾って劣化する」という知見が定着している
  （Mitigating the Impact of False Negatives in Dense Retrieval with Contrastive Confidence Regularization,
  AAAI 2024, <https://arxiv.org/html/2401.00165v2>；SyNeg: LLM-Driven Synthetic Hard-Negatives for Dense Retrieval,
  Li et al. 2024, <https://arxiv.org/abs/2412.17250>「existing hard negative sampling methods are prone to false
  negatives, resulting in performance degradation and training instability」．いずれも 2026-09-27 確認）．
  本研究の文脈では，最難 25 行の平均 p_true が medical 0.0101・history_culture 0.0407 と極端に低く，
  **ドメイン帰属自体が曖昧な行（本研究における false negative 相当）**である．25/75 はこの領域を 100 行から 25 行へ
  絞るので，劣化要因を減らす方向ではある．ただし上記 NAACL 2021 の表が示すとおり，**減らしすぎると今度は
  情報量のある境界行を失う**ので，一方向に良くなり続けるとは想定しない（Q4 のモデル A が対応する）．
- SimANS（Zhou et al., EMNLP 2022）・WSDM'22 ALOE・Tripp (2025) は Iter86/87 の調査節で既に引用済みで，
  「両極（最難のみ・純無作為のみ）を避けた方が堅い」という主張は本反復でも変わらず有効である．
  **いずれの文献も「25/75 が 50/50 より良い」とは言っていない．**

### 計画 (Iter88)

**単一レバー**

`cross_domain_training_data_augmentation` = **`hard_random_hybrid_ratio_25_75_all_domains`**．
**採掘コマンドの `--random-count` を `50` から `75` へ変えることだけを行う**（`--per-domain 100`・`--random-seed 87` は据え置き）．
評価集合・`config.yaml`・埋め込みモデル・instruction・連結仕様・送出閾値・集約方式・訓練スクリプト・プール定義・
除外集合・出力書式・id 規則・ドメイン別内訳（legal 0，他 9 ドメイン各 100）はすべて不変．**ソースコードの差分は 0 行**．

**レバーを読むコード行と，そこへ到達する条件（d0004 §4 の再発防止．省略不可）**

| 経路 | レバーが効く箇所 | 到達条件 | 到達確認 |
|---|---|---|---|
| 採掘・選定（オフライン） | `scripts/mine_hard_negatives.py:select_hard_negatives()` L281-289（`hard_count = max(per_domain - random_count, 0)` と `rng.permutation`） | CLI に **`--random-count 75 --random-seed 87`** を渡す．`--per-domain 100` 据え置き | **F2** |
| 採掘・スコア（不変） | 同 `_run()` L347 `joblib.load(args.classifier_model)` → `predict_proba` | **`--classifier-model models/domain_classifier_pre_iter84_baseline.joblib`**（`1cfcd3d8...`）を明示指定 | **F0** |
| 採掘・プール（不変） | 同 `build_pool()` の `exclude_queries` | `--eval-data data/dataset.jsonl`（3,435 行）・`--train-data data/classifier_train.jsonl`（1,427 行）据え置き | **F2** |
| 訓練（オフライン） | `scripts/train_domain_classifier.py` の `--train-data` | 新 JSONL を渡すこと（スクリプト自体の diff は 0 行） | **F3** |
| 実行時（本体） | `classifier.py:load_domain_classifier()` → `estimate_confidence_classifier()` が `config.yaml` の `classifier_model_path`（=`models/domain_classifier.joblib`）を読む | 各ノードのコンテナが**配布後の**新 artifact を読むこと（`mise.toml` の rsync） | **F4** |
| 記録側 | `run_experiment.py:98` の `dispatched_domains` 再計算（gap_threshold=0.36，不変） | 同上 | **F5** |
| 到達しない経路（変更しない） | 埋め込み・`aggregator.py`・`config.yaml`・`build_dataset.py`・`data/dataset.jsonl`・`data/classifier_train.jsonl`・`train_domain_classifier.py`・`mine_hard_negatives.py` | 一切触らない | **F1** |

**事前ゲート（実装フェーズで判定）**

| ゲート | 内容 | 不合格時 |
|---|---|---|
| **F0（スコア artifact）** | 採掘の直前に `sha256sum models/domain_classifier_pre_iter84_baseline.joblib` = `1cfcd3d836c5b5b421b8a47a487c3fb3ba996dd54aadcebae688cdfc8bcd0a48` を確認．**同時に `models/domain_classifier.joblib` が `f6c33edb1b80a43dbd4d7153b9088b97d220d4557958d0149fe043f0c47594d2`（Iter87 artifact＝本反復の基準線）であり，全 10 ノードにも同値が配布済みであることを確認する**．Iter87 と違い，採掘のために `models/domain_classifier.joblib` を上書き復元してはならない | 以降へ進まない |
| **G0（調達元の同一性．B139）** | `sha256sum /tmp/expert-mesh-cache/JMMLU.zip` = `3ba7d912943ede44fb7ec06aa1df067ac6bee157e65c5588736b9b983f0e684d`（本フェーズの実測では一致） | **FAIL の場合，F2 の「件数・入れ替わり行数が計画表と完全一致」条項は適用しない**（実測値を記録し非遮断の観察事項とする）．重複 0 件条項は G0 に関わらず絶対条件 |
| **F1（変更の最小性）** | `git diff` が `data/classifier_train_iter88_hybrid2575.jsonl`（新規）・`models/`・`data/MANIFEST.md`・`.claude/research/*` のみ．**`scripts/` 配下の diff は 0 行**（`mine_hard_negatives.py` を含む）．`config.yaml`・`node.py`・`classifier.py`・`aggregator.py` も 0 行．`data/dataset.jsonl`（3435 行）・`data/classifier_train.jsonl`（sha256 `eb89bf7b...`，1427 行）不変 | invalid |
| **F2（データ）** | 出力 `data/classifier_train_iter88_hybrid2575.jsonl` が **2,327 行**．先頭 1,427 行が `data/classifier_train.jsonl` と**バイト一致**．**追加 900 行と `data/dataset.jsonl` 3,435 行の設問文重複 0 件（絶対条件）**，既存 1,427 行との重複 0 件，追加行同士の重複 0 件，id 重複 0 件．ドメイン別追加数が **{legal: 0, 他 9 ドメイン各 100}**．プール実測が Q3 の表（計 3,127，medical 910 / history_culture 579 / natural_science 576 / business_economics 505 / mathematics 148 / education 109 / computer_science 100 / social_science 100 / general 100 / legal 0）と一致．**cs・social_science・general の追加 100 行が Iter86・Iter87 の同ドメイン 100 行と集合として完全一致（no-op の確認）**．**Iter87 の追加 900 行との入れ替わり実測が 309 行，Iter86 の追加 900 行との入れ替わり実測が 300 行と完全一致**（決定論的な予測値なので ±0 で一致すること．不一致なら seed 消費順か artifact 指定の誤り） | invalid |
| **F3（artifact）** | 新 `models/domain_classifier.joblib` の sha256 が `1cfcd3d8...`・`fd1ccd7d...`（Iter86）・`f6c33edb...`（Iter87）の**いずれとも異なる**．`n_features_in_`=2048，`classes_` が 10 ドメイン．`models/domain_classifier_iter87_hybrid.joblib` が `f6c33edb...` のまま残っている（復元元）．**新 artifact を `models/domain_classifier_iter88_hybrid2575.joblib` へ複製** | invalid |
| **F4（配布）** | `mise run deploy` 後，全 10 ノードで artifact sha256 が新値で一致，`dispatch_gap_threshold: 0.36` 一致，`docker compose exec app wc -l /app/data/dataset.jsonl` = **3435**（10/10） | invalid |
| **F5（実行時経路）** | 予備 20 問（`data/dataset_20.jsonl`）の `confidence`・`dispatched_domains` が，新 artifact ＋ キャッシュ埋め込みのオフライン予測と **20/20 一致** | 本走中止 |

**固定する構成（基準線 = Iter87 本走 `results/20260927_174150/`）**

`config.yaml` は 1 行も変更しない（`embedding_model=qwen3-embedding:0.6b`，`embedding_instruction`（Iter81 の P1 文言），
`embedding_view_concat=true`，`routing_method=supervised_classifier`，`confidence_threshold=0.0`，
`dispatch_candidate_threshold=0.0`，`dispatch_top_k=2`，`dispatch_gap_max_k=4`，`dispatch_gap_threshold=0.36`，
`aggregation_method=max_confidence`）．`data/dataset.jsonl`（3,435 行，バイト不変），
`data/classifier_train.jsonl`（1,427 行，バイト不変．追加分は別ファイルへ書く）．
**ドメイン固有の後付け補正は追加しない．棄権・人間エスカレーションは扱わない（2026-09-23 恒久運用ルール (1)(2)）．
本体実験（wafl500〜509 の 3,435 問本走）以外の処理（採掘・埋め込み・再訓練・replay）はすべて wafl-ctrl5 で行う（同ルール (B)）．
N・混合比・乱数種・プール定義は 10 ドメイン共通である．**

**仮説（事前登録）**

「Iter86（hard 100%）→ Iter87（hard 50%）で観測された『Q1 の改善は入れ替わり行数に比例してしか減らない（+14.85→+11.79pt，−21%）
のに，Q2〜Q4 の退行は比例以上に消える（−43〜−58%）』という非対称が，hard 25% でも同じ向きに続くなら，
Δtop1 は Iter87 基準線比で +0.2〜+0.5pt 伸びる．逆に，Q2〜Q4 の退行が Iter87 で既に床に達していて
Q1 の改善だけが減衰するなら Δtop1 は −0.6pt 程度になる．**どちらであっても効果量は McNemar の有意境界 0.88pt を下回る見込みで，
最も確からしい着地は `no_effect` である．**本反復の目的は，用量反応曲線の 3 点目を測ることと，
下記 5b で種由来の Δ の散らばりを数値化して『比の差を測定できる分解能が本測定系にあるか』を確定することにある．
なお追加 900 行のうち 300 行（computer_science・social_science・general）はプール M=100=N のため比をどう振っても動かず，
education も 8 行しか動かないので，実際に効くのは medical・history_culture・natural_science・business_economics・mathematics の 5 ドメインである．」

**着地点予測（事前登録．事後に書き換えない）**

| 指標 | 基準線 `results/20260927_174150/`（Iter87） | 参考: Iter86 / pre_iter84 | **Iter88 予測** |
|---|---|---|---|
| `top1_accuracy`（3,435 行） | 0.801456 | 0.796507 / 0.786317 | **0.795〜0.807．点推定 0.8043（+0.28pt）．モデル A〜C の幅は −0.61 〜 +0.50pt** |
| `single_domain_top1_accuracy`（3,020 行） | 0.799007 | 0.797020 | −0.7 〜 +0.6pt |
| `compound_domain_top1_accuracy`（415 行） | 0.819277 | 0.792771 | 0.80〜0.83 |
| **`education` recall** | **0.3626** | 0.3441 / 0.3880 | **0.35〜0.38．非退行①の個別監視項目（B143 (b)）** |
| **`natural_science` precision** | **0.8424** | 0.8029 / 0.8729 | **0.83〜0.86．非退行①の個別監視項目（B143 (b)）** |
| `history_culture` recall | 0.8701 | 0.8840 / 0.8237 | 0.85〜0.88 |
| `medical` recall / precision | 0.7710 / 0.8000 | ― | 追加行の平均 p_true が最も大きく動くドメイン（0.362→0.530）．監視 |
| `legal` recall / precision | 0.6831 / 0.8137 | 0.6420 / ― | 追加行 0 件の唯一のドメイン．監視対象 |
| `compound_domain_set_recall` / `compound_mean_dispatched_count` | 0.566265 / 1.879518 | ― | 非退行②③の枠内で報告 |
| `mean_duration_ms` | 1544.912 | 1534.531 | 非退行⑥ 1853.9 以内 |
| ECE | 0.045717 | 0.040154 | 0.02〜0.08 |
| rank1 以外が選ばれた行数 | 5 | 1 | ≤ 15 |
| `answer_quality_accuracy` / `end_to_end_accuracy` | 0.587417 / 0.411063 | ― | ノイズ床 3SD=2.6pt の範囲でのみ判定 |
| Random / BestSingle / Oracle | 0.112082 / 0.128384 / 1.0 | 不変 | success_criteria (3) により毎回併記 |

**成功条件・非退行条件（事前登録．Iter86/87 の事前登録を一字も変えずに据え置く．結果を見る前に固定し，事後に緩めない）**

| 区分 | 指標 | 基準線（Iter87） | 合格条件 |
|---|---|---|---|
| **主基準（効果）** | 全 3,435 行 `top1_accuracy` | 0.801456 | **(i) McNemar 対比較（α=0.05）で有意，かつ (ii) +1.0pt 以上（≥ 0.811456）**．AND 条件．Wilson 95%CI と検出限界 `1.96·√(n_d)/3435` を併記 |
| **非退行①** | per-domain recall / precision 計 20 指標（BH 補正 q=0.05） | 上表（Iter87 実測値） | **有意退行 0 件**．**`education` recall（0.3626）・`natural_science` precision（0.8424）は個別に明記する（B143 (b)）**．`legal`（追加 0 件）も明記 |
| **非退行②（複合被覆）** | `compound_domain_set_recall` | 0.566265 | **≥ 0.539759**（Iter82 水準を下回らない．絶対値の床であり Iter86/87 と同一） |
| **非退行③（複合予算）** | `compound_mean_dispatched_count` | 1.879518 | **≤ 2.10**（絶対値．Iter86/87 と同一） |
| **非退行④** | `fallback_rate` / `dispatch_failure_rate` | 0.0 / 0.001456 | fallback = 0.0，dispatch_failure ≤ 0.005（絶対値．Iter86/87 と同一） |
| **非退行⑤** | rank1 以外が選ばれた行数 | 5 | **≤ 15**（絶対値．Iter86/87 と同一） |
| **非退行⑥** | `mean_duration_ms` | 1544.912 | **≤ 1853.9**（規則は Iter86/87 と同一の「基準線 +20% 以内」．基準線が 1574.096→1544.912 へ更新されたので閾値は 1888.9→1853.9 と**わずかに厳しく**なる．緩めていない） |
| **非退行⑦** | ECE | 0.045717 | **≤ 0.08**（絶対値．Iter86/87 と同一） |
| 報告のみ | 層別 Δ（基準線 p_true の五分位） | Iter86/87 の五分位表（境界値は同一） | **必ず併記**．モデル A / B / C のどれに近いかを解釈に使う．**pre_iter84 基準線比の値も併記して 100/50/25 の用量反応曲線 3 点を 1 表にまとめること** |
| 報告のみ | Iter87 の 900 行との入れ替わり実測行数 | 予測 309（Iter86 比は 300） | 選択規則が効いた証拠 |
| 報告のみ | **種 87/88/89 × 比 50/50・25/75 の replay Δ の散らばり** | 未測定 | **下記 5b．必須手順** |
| 報告のみ | 1,915 行サブセット top1 | 0.809922 | 過去基準線との接続用 |
| 報告のみ | `education` recall の pre_iter84 基準線（0.387991）からの累積ドリフト | −2.54pt（Iter87 時点） | B143 (b) の申し送り．有意性ではなく点推定の符号と大きさを記録する |
| 報告のみ | `answer_quality_accuracy` / `end_to_end_accuracy` | 0.587417 / 0.411063 | 3SD=2.6pt のノイズ床の範囲でのみ有意と判定 |

**判定規則（事前登録．Iter86/87 と同一．一字も変えていない）**

- **adopted**: 主基準 (i)(ii) を満たし，非退行①〜⑦をすべて満たす．
- **partial**: McNemar が有意で Δtop1 が +0.5〜+1.0pt，かつ非退行①〜⑦を満たす．
- **no_effect**: |Δtop1| < 0.5pt または McNemar が非有意．結論前に F0〜F5 を再確認し実験不成立でないことを示すこと．
  **Q4 の投影ではこれが最も確からしい着地である．この場合「hard/random 混合比の用量反応は 50/50 付近で頭打ちで，
  25/75 へ振っても top1 は動かない」と記録し，B143 が予告した純無作為 0/100 の対照実験へ進むか，
  `cross_domain_training_data_augmentation` を打ち止めにして config.yml の levers の次候補へ移るかを分析フェーズで決める．**
- **rejected**: Δtop1 ≤ −0.5pt，または非退行①〜⑦のいずれかに違反．
  **`education_recall` の有意退行が再現した場合はここに該当する．** その場合，「hard negative を含む訓練データ拡充は
  混合比を変えても頭打ちで，ランダム側へ振りすぎると境界情報が薄まって退行する」と記録し，
  ドメイン固有の手当てへは進まない（2026-09-23 恒久運用ルール (1)）．
- **invalid（実験不成立）**: F0〜F5 のいずれか不合格，`total_questions != 3435`，`compound_domain_question_count != 415`，
  新 artifact の sha256 が `1cfcd3d8...` / `fd1ccd7d...` / `f6c33edb...` のいずれかと一致，
  **追加 900 行と評価集合の重複が 1 件でもある**，cs/social_science/general の追加行が Iter86・Iter87 と不一致，
  Iter87 比の入れ替わり実測が 309 行と不一致（G0 PASS 時），または本走 top1 がオフライン replay 予測から 1.0pt 以上乖離．
- **復元手順（adopted 以外すべて）**: `cp models/domain_classifier_iter87_hybrid.joblib models/domain_classifier.joblib`
  （sha256 が **`f6c33edb...`**（Iter87＝現基準線）に戻ることを確認）→ `mise run deploy` → 全 10 ノードで smoke_check．
  **`1cfcd3d8...`（pre_iter84）へは戻さない**（基準線は B143 (c) で Iter87 へ更新済みのため）．
  **この復元は分析フェーズ内で必ず完了させ，未実施のまま次反復へ送らない**（Iter86 でこれを積み残した前例がある）．
  `config.yaml`・`data/dataset.jsonl`・`data/classifier_train.jsonl` は本反復で触らないので復元不要．
  `data/classifier_train_iter88_hybrid2575.jsonl` と `models/domain_classifier_iter88_hybrid2575.joblib` は記録として残す．

**実験手順（この順で行うこと）**

1. **前提確認（F0・G0）**: `sha256sum models/domain_classifier_pre_iter84_baseline.joblib` = `1cfcd3d8...`，
   `sha256sum models/domain_classifier.joblib` = `f6c33edb...`，全 10 ノードでも `f6c33edb...`（smoke_check），
   `wc -l data/dataset.jsonl` = 3435，`wc -l data/classifier_train.jsonl` = 1427，
   `sha256sum /tmp/expert-mesh-cache/JMMLU.zip` = `3ba7d912...`，
   `config.yaml` の `embedding_view_concat: true` / `embedding_instruction` / `dispatch_gap_threshold: 0.36`．
   **`models/domain_classifier.joblib` を上書きする操作はこの段階では行わない．**
2. **採掘**（**wafl-ctrl5 限定**．プール埋め込みキャッシュ `pool_hash=221e45e8...` が一致するため再埋め込みは発生せず，実質 CPU のみ）:
   `data/MANIFEST.md` の Iter87 生成コマンドから **`--random-count 50` → `75`**，
   **`--classifier-model models/domain_classifier.joblib` → `models/domain_classifier_pre_iter84_baseline.joblib`**，
   **`--output` → `data/classifier_train_iter88_hybrid2575.jsonl`** の 3 箇所だけを替えて実行する（他は 1 文字も変えない）．→ **F2**
3. **F2 の決定論的検証**: 同じコマンドの `--random-count` を `0` / `50` に替えて一時ファイルへ出力し，
   それぞれ `data/classifier_train_iter86_hardneg.jsonl` / `data/classifier_train_iter87_hybrid.jsonl` と
   **バイト一致**することを確認する（採掘系の決定論性と artifact 指定の正しさを同時に担保する．本フェーズで実測確認済み）．
   確認後は一時ファイルを削除する．
4. **再訓練**（**wafl-ctrl5 限定**）: `data/MANIFEST.md` の `train_domain_classifier` コマンドの `--train-data` だけを
   `data/classifier_train_iter88_hybrid2575.jsonl` に替えて実行 →
   `cp models/domain_classifier.joblib models/domain_classifier_iter88_hybrid2575.joblib` → **F3**．
   `train_domain_classifier.py` は埋め込みキャッシュを持たないため 2,327 行 ×2 view を毎回埋め込み直す点に注意（時間見積りの根拠）．
5. **【必須】5b: 乱数種ばらつきの report-only 計測（B143 が必須と定めた手順．「余力があれば」ではない）**
   （**wafl-ctrl5 限定**）．Iter87 ではこれを余力条件に置いた結果，実施されず未測定のまま残った．本反復では必須とする．
   - **対象は 6 構成**: 比 {50/50, 25/75} × 種 {87, 88, 89}．うち 2 構成（50/50 種 87 = `models/domain_classifier_iter87_hybrid.joblib`，
     25/75 種 87 = 手順 4 の新 artifact）は既に存在するので，**追加で採掘・再訓練するのは 4 構成**（50/50 種 88・89，25/75 種 88・89）．
   - 各構成について，`data/embcache_eval_qwen3-embedding_0.6b{,__p1}.npy`（**1,915 行**，本フェーズで shape 実測確認済み）を
     特徴量として `predict_proba` し，`aggregator.select_dispatch_targets()` を再利用して
     `gap_threshold=0.36` / `gap_max_k=4` / 閾値 0.0 の下で `dispatched_domains` を再現する
     （Iter87 と同一手法．基準線 artifact に適用すると Iter83 の G1 記録値 0.793211 と完全一致することで忠実度確認済み．
     Iter87 実測では replay 予測 0.810966 対 本走実測 0.809922 で乖離 0.104pt）．
   - **記録するもの**: 6 構成の replay top1（1,915 行），比ごとの 3 種の平均・標準偏差・レンジ，
     **および「同一比・異種間の Δ の絶対値の最大値」**．これが**比の差（投影 0.3〜0.5pt）と同程度かそれ以上であれば，
     本測定系では比の差を 1 回の本走で判定できないと結論し，その旨を分析フェーズの結論に明記する**．
   - **種の選び直しには絶対に使わない．** 本走に使う種は 87 で事前固定済みであり，
     5b の結果がどうであろうと手順 4 の artifact を差し替えてはならない（garden of forking paths の回避）．
     この計測は「Δ の不確実性の大きさを知る」ためだけのものである．
6. **本走前の着地点記録（B136）**: 手順 4 の新 artifact ＋ 既存 1,915 行キャッシュ埋め込みで replay を走らせ，
   予測 top1 / per-domain / 複合指標を**本走前に** journal へ追記する．
7. **配布**: `mise run deploy` → **F4**（ここで初めて `models/domain_classifier.joblib` が新 artifact になる）．
8. **予備 20 問**（`data/dataset_20.jsonl`）→ **F5**．不一致なら本走に進まない．
9. **本走**: wafl500〜509 で **3,435 問フルスペック 1 回**（config.yml 絶対条件 (A)．事前 replay で経路指標が予測できても省略しない）．
   想定所要は Iter85/86/87 実績から **約 90 分**（timeout 180 分以内）．
10. `mise run analyze -- <YYYYMMDD_HHMMSS>`（**B135: 引数を省略すると `results/iter45_preliminary/` を誤選択する**）．
    層別 Δ（基準線 p_true 五分位．**pre_iter84 比も併記して 100/50/25 の 3 点曲線を作る**）・Iter87/Iter86 との入れ替わり行数・
    1,915 行サブセット top1 も併せて算出する．統計は `metrics.py` の既存関数
    （`compute_mcnemar_test` / `compute_domain_recall_mcnemar_test` / `compute_domain_precision_fisher_test` /
    `apply_benjamini_hochberg` / `compute_top1_accuracy_wilson_ci`）をそのまま再利用し，自前の統計式は書かない．
11. `data/MANIFEST.md` に新訓練ファイル・新 artifact の sha256 と行数・**乱数種 87・`--random-count 75`**・生成コマンド・
    採掘に用いたスコア artifact（`1cfcd3d8...`）・ゲート結果・5b の 6 構成の replay 値を追記する．

**次イテレーションへの申し送り**

- `no_effect`（最も確からしい着地）なら，用量反応は 50/50 付近で頭打ちと確定する．次の選択肢は
  (α) 純無作為 0/100 の対照（B143 が予告済み．「hard negative mining が無作為に勝つか」の決着がつく）か，
  (β) `cross_domain_training_data_augmentation` を打ち止めにして config.yml の levers の次候補へ移るか．
  **5b で測った種ばらつきが比の差と同程度だった場合は (α) も 1 回の本走では判定できないので，(β) を推す．**
- `rejected`（退行方向）なら，50/50 が用量反応の最適点であると記録し，本レバーは打ち止めにする．
- `adopted` / `partial` なら，さらにランダム側（例: 10/90）へ振る余地があるが，
  **cs・social_science・general の 300 行と education の 92 行は比をどう振っても動かない**という構造的上限は変わらない．
  この上限を外すには追加行数 N か評価集合の構成を変えるしかなく，どちらも別レバーになる．

### Iteration 88 実行済み

**変更（実施したこと）**

採掘コマンドの `--random-count` を `50` → `75` に替えただけである（`--per-domain 100`・`--random-seed 87` 据え置き）．
`--classifier-model` は B144 (A) の判断どおり `models/domain_classifier_pre_iter84_baseline.joblib`（`1cfcd3d8...`）を
明示指定し，基準線側の実行時 artifact `f6c33edb...` は採掘のために上書きしていない．**`scripts/` 配下を含めソース差分は 0 行**で，
実質の差分は `data/MANIFEST.md`・`.claude/research/*`・新規データ/artifact のみ．新訓練データ
`data/classifier_train_iter88_hybrid2575.jsonl`（2,327 行，sha256 `c341baef...`），新 artifact
`models/domain_classifier_iter88_hybrid2575.joblib`（sha256 `51b9ced5...`）を全 10 ノードへ配布．
F0〜F5・G0 すべて PASS．入れ替わり行数は Iter87 比 **309 行**・Iter86 比 **300 行**で事前登録の決定論的予測値に ±0 で一致．
cs・social_science・general の追加 100 行は Iter86/87 と集合として完全一致（no-op）．
採掘の決定論性は `--random-count 0` / `50` の再現出力が Iter86/87 の訓練ファイルとバイト一致することで確認済み．
本走は `results/20260927_202256/`（3,435 問，約 91 分）．

**結果（事前登録の表に対応させる）**

| 指標 | 基準線 Iter87 `results/20260927_174150/` | **Iter88 実測** | 合否 |
|---|---|---|---|
| `top1_accuracy`（3,435 行） | 0.801456 | **0.801164（Δ = −0.03pt）** Wilson 95%CI [0.787484, 0.814172] | 主基準 (i)(ii) **不成立** |
| McNemar | ― | **chi2 = 0.0，p = 1.0（非有意）**，discordant 127（a_only 64 / b_only 63），検出限界 0.612pt | ― |
| `single_domain_top1` / `compound_top1` | 0.799007 / 0.819277 | 0.798344 / 0.821687 | 予測帯内 |
| 非退行①（per-domain 20 指標，BH q=0.05） | ― | **有意退行 0 件**．`education_recall` 0.3626→**0.3811**（+1.85pt，生 p=0.098960，非有意），`natural_science_precision` 0.8424→**0.8541**（+1.17pt，生 p=0.687276，非有意）．legal_recall 0.6337（生 p=0.003283）・mathematics_recall（生 p=0.013328）も BH 閾値を上回らず非有意 | **PASS** |
| 非退行② `compound_domain_set_recall` | 0.566265 | 0.553012（≥ 0.539759） | **PASS** |
| 非退行③ `compound_mean_dispatched_count` | 1.879518 | 1.783133（≤ 2.10） | **PASS** |
| 非退行④ fallback / dispatch_failure | 0.0 / 0.001456 | 0.0 / 0.000291（1 行） | **PASS** |
| 非退行⑤ rank1 以外が選ばれた行数 | 5 | **1**（≤ 15） | **PASS** |
| 非退行⑥ `mean_duration_ms` | 1544.912 | 1546.784（≤ 1853.9） | **PASS** |
| 非退行⑦ ECE | 0.045717 | **0.022189**（≤ 0.08） | **PASS** |
| 報告のみ | `answer_quality` / `end_to_end` 0.587417 / 0.411063 | 0.585762 / 0.407860（3SD=2.6pt 以内） | ― |
| 報告のみ | 1,915 行サブセット | 0.805222（本走前 replay 予測 0.806266 から乖離 0.104pt，行一致 1913/1915） | ― |

invalid 条件（`total_questions`=3435・`compound_domain_question_count`=415・sha256 の非一致・
追加 900 行と評価集合の重複 0 件・入れ替わり 309 行一致・replay 乖離 < 1.0pt）はいずれにも該当しない．
**すなわち実験は成立しており，「効果が無かった」を素直に読んでよい**（d0004 §4 の「基準線と完全一致＝実験不成立」型とは異なる．
今回は本走の行単位の結果が 127 行入れ替わっており，レバーが発火した証拠は十分にある）．

**判定: `no_effect`（事前登録の条文をそのまま適用）**

事前登録は `no_effect` を「|Δtop1| < 0.5pt **または** McNemar が非有意」と定義している．
実測は |Δ| = 0.03pt < 0.5pt（第 1 項）かつ p = 1.0（第 2 項）で，**両方を独立に満たす**．
`adopted`（主基準 (i)(ii) の AND）・`partial`（有意かつ +0.5〜+1.0pt）は (i) が不成立のため該当しない．
`rejected`（Δ ≤ −0.5pt または非退行違反）も，Δ = −0.03pt > −0.5pt かつ非退行①〜⑦全 PASS のため該当しない．
判定語は事前登録どおり **`no_effect`** で確定する．規則は結果に合わせて緩めても厳しくもしていない（B131 以来の運用）．

**学び 1: 相殺の構造 — Q1 の改善減衰と Q2〜Q4 の退行消失が，ちょうど打ち消し合った**

| 層（各 n=687） | pre_iter84 比 Iter86（100/0） | Iter87（50/50） | **Iter88（25/75）** | Iter87→88 の寄与（×0.2） |
|---|---|---|---|---|
| Q1（最難） | +14.85 | +11.79 | **+9.02** | **−0.554pt** |
| Q2 | −6.84 | −2.91 | **−1.75** | +0.232pt |
| Q3 | −2.33 | −1.02 | **−0.15** | +0.174pt |
| Q4 | −1.02 | −0.58 | **−0.15** | +0.086pt |
| Q5（最易） | +0.44 | +0.29 | **+0.44** | +0.030pt |
| **全体 Δtop1（pre_iter84 比）** | **+1.019** | **+1.514** | **+1.485** | **−0.03pt** |

- **仮説の両半分がどちらも当たった上で，合計だけがゼロになった．** 事前登録の仮説は
  「Q2〜Q4 の退行消失が続くなら +0.2〜+0.5pt，Q1 の減衰だけが進むなら −0.6pt」という二者択一だったが，
  実測は**両方が同時に単調に進行**した．Q1 は −2.77pt（寄与 −0.554pt），Q2〜Q5 は合計 +2.62pt（寄与 +0.524pt）で，
  差し引き −0.03pt．モデル A（−0.61pt）とモデル C（+0.50pt）の**ほぼ中点**であり，
  投影の点推定 +0.28pt（モデル B）とも 0.31pt しか違わない．投影の幅取り自体は妥当だった．
- **Q1 の減衰は hard 行数の対数にほぼ線形**（100→50 で −3.06pt，50→25 で −2.77pt）．
  一方 Q2〜Q4 の退行は減衰が急で，Q3・Q4 は −0.15pt まで来てほぼ床に達した．
  **つまり「ランダム側へ振ると悪い方が先に直り切り，そのあとは良い方の減衰だけが残る」**構造である．
  Q2〜Q4 の回復余地がもう 0.5pt 分しか残っていない以上，さらにランダム側（10/90 等）へ振れば
  Q1 の減衰が露出して **Δ は必ず負に転じる**．用量反応の向きは 25/75 で決着した．
- **50/50 は「交差点」か**: 3 点を log2(hard 比) の二次で近似すると頂点は **hard 比 36.7%・peak +1.566pt**．
  ただし **頂点と 50/50（+1.514pt）の差はわずか 0.05pt** であり，これは下記の種ばらつき 0.31pt の 1/6 である．
  **3 点では「50/50 と 25/75 の間のどこかに幅の広い平坦な頂上がある」までしか言えず，
  頂点位置を種ばらつきより細かく特定することはできない．** 過剰に「36.7% が最適」と読んではならない．
- **構造的制約**: cs・social_science・general は M=100=N で追加 900 行のうち 300 行（33.3%）が今回も厳密に no-op，
  education も 8 行しか動かない．**比の刻みで動かせるのは実質 5 ドメインの 600 行弱**であり，
  比を振り続けても効果量の上限がここで頭打ちになる．この上限を外すには N か評価集合の構成を変えるしかなく，別レバーになる．

**学び 2（本反復の最大の成果）: 5b の実測により，「比の差は本測定系の分解能を下回る」ことが確定した**

| 構成 | replay top1（1,915 行） | 比ごとの平均 / 標本 SD / レンジ |
|---|---|---|
| 50/50 種 87 / 88 / 89 | 0.810966 / 0.813055 / 0.809922 | 0.811314 / 0.001595 / **0.003133（0.31pt）** |
| 25/75 種 87 / 88 / 89 | 0.806266 / 0.809399 / 0.807833 | 0.807833 / 0.001567 / **0.003133（0.31pt）** |

- **同一比・異種間の Δ の絶対値の最大値は両比とも 0.31pt．** 一方，比の効果は
  事前投影で +0.28pt，replay の種対応差で −0.35pt（種別 −0.47 / −0.37 / −0.21pt）である．
  **比の効果と種由来のノイズは完全に同オーダーであり，前者が後者に埋もれている．**
- **本走 1 回の分解能を数値で確定する**: discordant n_d = 127 のとき McNemar の有意境界は
  `1.96·√127/3435` = **0.612pt**，80% 検出力に必要な Δ は `2.8·√127/3435` ≈ **0.92pt**．
  **したがって，0.3〜0.5pt 規模の比の差は本走 1 回では原理的に判定できない．**
  これは本反復の事後解釈ではなく，事前登録（検出力表：真の Δ = +0.28pt での検出力 10%）が予告していたとおりである．
- **さらに決定的な証拠（分割半信頼性）**: 同じ 1 回の本走を 2 つに割ると，
  **1,915 行の旧サブセットでは Δ = −0.47pt，残る 1,520 行（Iter85 拡充分）では Δ = +0.53pt** と
  **符号が逆で大きさが同程度**になり，合計してはじめて −0.03pt になる．
  **同一実行の中でさえ ±0.5pt の系統的な揺れが部分集合ごとに生じる**以上，
  0.3〜0.5pt の差を根拠に比の優劣を論じることはできない．
- **結論（今後の実験設計を規定する）**: **本測定系（3,435 問・1 本走・種 1 個）で判定できるのは
  おおむね 0.9pt 以上の効果量に限られる．これを下回る効果量のレバーは，同系列で刻み続けても結論が出ない．**
  今後のレバー選定は「効果量が 0.9pt を明確に上回る見込みがあるか」を事前に見積もってから行う．
  0.3pt 級を測りたければ (a) 評価集合をさらに数倍にする，(b) 種を複数（≥3）平均する，のいずれかが必須で，
  どちらも本走コスト（1 回 90 分）を数倍にする．**現時点でそのコストを払う価値のある問いは無い．**

**学び 3: 副次的に観測された挙動**

- **education_recall は +1.85pt（0.3626→0.3811）で初めて明確に改善方向**．pre_iter84（0.387991）からの累積ドリフトは
  −2.54pt → **−0.69pt** まで縮んだ．`natural_science_precision` も +1.17pt（0.8541）で pre_iter84（0.8729）に接近した．
  B143 (b) が監視対象としていた「medical の押し出しが education を削り natural_science の FP を増やす」経路は，
  追加行の平均 p_true が上がるにつれ**単調に弱まっており，25/75 でほぼ解消した**．ただし両指標とも生 p は
  0.098960 / 0.687276 で**非有意であり「直った」とは記録しない**（B143 (b) と同じ読み方を維持する）．
- **ECE が 0.045717 → 0.022189 へ半減した**．追加 900 行の平均 p_true が 0.5606 → 0.6330 と
  プールの典型分布に寄ったことで，訓練分布と評価分布の乖離が縮み確信度較正が改善したと解釈できる．
  非退行⑦は絶対値 0.08 以下の条文なので判定には影響しないが，**「難しい行ばかり足すと過信が増える」という
  副作用が比で制御できる**ことを示す観測である．
- **rank1 以外が選ばれた行数が 5 → 1 へ**，dispatch_failure も 5 行相当 → 1 行へ減った（いずれも絶対値の床は満たす）．
- 想定外の挙動（言語崩れ・発散・OOM・タイムアウト）は無い．本走は約 91 分で完走した．

**判定に伴う処置（事前登録の復元手順を条文どおり実行済み）**

判定が `adopted` 以外のため復元条項に該当する．本分析フェーズ内で以下を完了した（次反復へ積み残さない）．
`cp models/domain_classifier_iter87_hybrid.joblib models/domain_classifier.joblib` →
sha256 が **`f6c33edb1b80a43dbd4d7153b9088b97d220d4557958d0149fe043f0c47594d2`** に戻ることを確認 →
`mise run deploy` → **全 10 ノード（wafl500〜509）で `models/domain_classifier.joblib` の sha256 = `f6c33edb...` を実機確認**，
`smoke_check` の git-status / hashes / probe すべて PASS（probe latency 5ms，LLM 呼び出し無し）．
`1cfcd3d8...`（pre_iter84）へは戻していない（基準線は B143 (c) で Iter87 へ更新済み）．
`config.yaml`・`data/dataset.jsonl`・`data/classifier_train.jsonl` は本反復で触っていないため復元不要．
`data/classifier_train_iter88_hybrid2575.jsonl` と `models/domain_classifier_iter88_hybrid2575.joblib` は記録として残す．
**次反復の基準線は引き続き Iter87 本走 `results/20260927_174150/`（top1 = 0.801456，artifact `f6c33edb...`）である．**

**次の一手（B145 で記録．詳細は backlog 参照）**

- **`cross_domain_training_data_augmentation` は打ち止め（closed）**．事前登録の申し送りは
  「5b で測った種ばらつきが比の差と同程度だった場合は (α) 純無作為 0/100 の対照も 1 回の本走では判定できないので (β) を推す」
  と定めており，実測（種間 Δ 最大 0.31pt ＝ 比の効果と同オーダー）はこの条件に該当する．
  **よって (α) は実施せず，(β)＝ config.yml の levers の次候補へ移る．** 最終構成は Iter87 の 50/50 のままとする．
- **次レバーは `embedding_model_replacement` = `qwen3_embedding_4b`**（Iteration 89）．
  選定理由は学び 2 の帰結そのもので，**効果量が測定分解能 0.9pt を明確に上回る見込みがある唯一の近接候補**だからである
  （MTEB multilingual は 0.6b の 64.33 に対し 4b が 69.45 で +5.12pt．0.6b への差し替え自体が Iter79 系列で
  複数 pt の変化を生んだ実績がある）．B116 (3) によりユーザー事前承認済みで，着手前の追加確認は不要．
  `research_frontier` 最上位の複合評価集合拡充（`compound_eval_set_expansion`）を先に採らない理由は，
  それが top1 を上げるレバーではなく測定系の整備であり，全基準線の再取得（1 回 90〜150 分 × 複数）を伴うためである
  （ただし学び 2 により，将来この拡充の優先度は上がった．B145 に申し送る）．

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

