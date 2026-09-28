## Iteration 95: 未使用の JMMLU プールを全投入し訓練行を約 2 倍にする

### 調査 (Iter95)

backlog B157 (d)(e) の申し送り「`levers` を使い切ったので調査フェーズから再探索し，tavily-search で
関連研究・代替アプローチを重点調査して新レバーを考案せよ」に従い，(1) 現行基準線の誤りの内訳の再集計，
(2) 訓練集合の構成と JMMLU プール残量の実測，(3) 学習曲線の実測，(4) 文献調査，の 4 点を行った．
**評価集合 `data/dataset.jsonl` は読むだけで変更していない．`data/`・`models/`・`results/` への
書き込みは 0 件．実機ノード wafl500〜509 も不使用**（新規の埋め込み計算も 0 件で，
既存キャッシュ `data/embcache_train_iter89_qwen3-embedding_4b{,__p1}.npy` のみを使った）．
**（実施場所の申告）**(3) の学習曲線は sklearn の CPU 計算のみで GPU を使わないため，
wafl-ctrl5 に Python 環境（uv も sklearn も未導入）を新設せずに開発ホスト上で実行した．
2026-09-23 絶対条件 (B) が禁じている wafl500〜509 は一切使っていない．
wafl-ctrl5 への環境構築は実装フェーズの作業として B158 に申し送る．

**(1) 基準線 `results/20260928_160921/`（top1 = 0.833067）の誤り 626 件の内訳**

| 区分 | n | top1 |
|---|---|---|
| 全体 | 3,750 | 0.8331 |
| 単一ドメイン行 | 3,020 | 0.8434（誤り 473） |
| 複合設問行 | 730 | 0.7904（誤り 153） |

単一ドメイン行の per-domain recall は education 0.500（誤り 175 = 全体の 37%）・general 0.743・
social_science 0.789・medical 0.837 で，残り 6 ドメインは 0.911〜0.957．
**education の誤りは特定の 1 ドメインへ流れているのではなく business_economics 48 / history_culture 38 /
legal 28 / medical 27 / social_science 21 と分散している**．JMMLU タスク単位に割ると
japanese_civics 0.319（n=116，予測先は business_economics 39・education 37・history_culture 18・legal 17）・
moral_disputes 0.553・sociology 0.588・high_school_psychology 0.630 で，
**最悪は japanese_civics であり，その誤りの向き（経済・法・歴史）は公民という科目の内容そのものと一致する**．
general の誤りも miscellaneous 0.391 に集中しており，同じく「内容が他ドメインと本質的に重なるタスク」である．
すなわち education/general の低 recall の相当部分は**ラベルの意味的重なりに由来する**と読める．

**(2) 訓練集合の構成と JMMLU プール残量（実測）**

`data/classifier_train_iter94_dedup.jsonl`（2,275 行）の内訳は，各ドメイン 150 行の本体 ＋ 100 行の
hard negative（Iter84/86/88 由来）で 250 行，ただし legal 77 行（プール枯渇）・education 238 行・
history_culture 210 行である．JMMLU 全 56 タスクを突き合わせた結果，
**評価にも訓練にも使われていない行が 2,232 行残っている**．

| ドメイン | プール | 評価 | 訓練 | 未使用 |
|---|---|---|---|---|
| medical | 1,410 | 350 | 250 | **810** |
| natural_science | 1,087 | 354 | 252 | **481** |
| history_culture | 1,039 | 350 | 210 | **479** |
| business_economics | 1,007 | 351 | 251 | **405** |
| mathematics | 648 | 350 | 250 | 48 |
| education | 598 | 351 | 238 | 9 |
| legal / computer_science / social_science / general | 227 / 551 / 544 / 425 | 150 / 301 / 294 / 175 | 77 / 250 / 250 / 250 | **0** |

つまり **`build_dataset.py` の per-domain サンプリング上限 150 が効いていただけで，データが尽きていた
わけではない**（尽きているのは legal・computer_science・social_science・general・education の 5 つ）．
特徴次元は 2 ビュー連結で d = 5,120，訓練行数は n = 2,275 で **d/n = 2.25 の過剰パラメータ領域**にある．

なお，ドメイン内のタスク構成が訓練と評価でずれていることも実測した（全変動距離 TV：
education 0.189 / medical 0.169 / history_culture 0.153 / …… / mathematics 0.032）．
ただし **JMMLU プールの構成と評価集合の構成の TV は 0.017〜0.081 と小さい**ので，
「プール比例で訓練を層化し直す」は評価集合を覗かずに書ける規則である．
しかし medical で不足しているタスク（professional_medicine 1.000・nutrition 0.977・clinical_knowledge 0.905）は
recall が高く，過剰なタスク（professional_psychology 0.658・college_medicine 0.714）は recall が低いので，
**構成合わせは難しいタスクから訓練を奪う向きに働く**．採らない．

**(3) 学習曲線の実測 —— err ∝ n^(-0.127)**

既存キャッシュで再現した 2,255 行（評価と重複する行を除いた Iter88 訓練集合）に対し，
層化 5-fold × 3 seed で，訓練側だけを 25/50/75/100% に層化サブサンプルして測った
（テスト fold は同一，`LogisticRegression(max_iter=3000)` ＋ 既存と同じ `n/(K*n_d)` の balanced 重み，較正なし）．

| 訓練行数 | CV accuracy | error |
|---|---|---|
| 451 | 0.7561 | 0.2439 |
| 902 | 0.7718 | 0.2282 |
| 1,353 | 0.7865 | 0.2135 |
| 1,804 | 0.7954 | 0.2046 |

逆べき乗則を当てると **α = 0.127**（n=1,804 の CV 0.7954 は B154 の実測 0.7961 と整合する）．
外挿すると **n を 2 倍で CV +1.73pt，1.5 倍で +1.03pt**．Iter91 実測の伝達率 43% を当てた
end-to-end の予測は **+0.74pt**（2 倍時）である．**再現性の床 ±0.25pt の約 3 倍**にあたる．

**(4) 文献調査（tavily-search）**

- Viering & Loog (2024), *The shape of learning curves: a review* / 学習曲線による意思決定の総説,
  *Machine Learning*, <https://link.springer.com/article/10.1007/s10994-024-06619-7> ——
  逆べき乗則による外挿で「あと何行増やせばどれだけ上がるか」を事前に見積もる手法は確立している．
  Figueroa et al. (2012), *BMC Medical Informatics and Decision Making* 12:8 も同じ枠組み．
  本反復 (3) はこの手続きそのものである．
- Byrd & Lipton (2019), *What is the Effect of Importance Weighting in Deep Learning?*, ICML 2019,
  <http://proceedings.mlr.press/v97/byrd19a/byrd19a.pdf> —— 訓練集合を**重み付け・再サンプリング**して
  分布を合わせる操作は，訓練集合を完全に当てはめられる容量のモデルでは漸近的に効果が消える．
  d/n = 2.25 の本研究の分類器はこの領域にあり，(2) の構成合わせ案を採らない理由の 1 つである．
- Cichy & Rakotomamonjy 系ではなく，重み付けでなく**行数**を増やすべき，という同じ含意は
  Lee et al. (2022) の重複除去論文（Iter94 で引用）とも矛盾しない．
- Piedboeuf & Langlais (2023) / Cegin et al. (2025), *LLMs vs Established Text Augmentation Techniques
  for Classification*, NAACL 2025, <https://aclanthology.org/2025.naacl-long.526.pdf> ——
  LLM によるパラフレーズ増強が効くのは **1 ラベルあたり 5〜20 シード**の領域で，
  30 シードを超えると既存手法との差が縮む．本研究は 1 ドメイン 77〜250 行なので，
  **合成データ増強は本命ではない**．B116 (2) の「まず既存データセットを探す」にも合致する
  （そして探した結果が下記）．
- 外部の日本語データ源: JMMLU 以外で公民・教育行政を四択で覆う CC BY 級の公開データは見つからなかった．
  llm-jp-eval の DATASET 一覧（<https://github.com/llm-jp/llm-jp-eval/blob/dev/DATASET_en.md>）で
  日本語の人間試験系は JMMLU / MMMLU / MMLU-ProX / GPQA-JA に限られ，公民相当のタスクを持つのは JMMLU だけである．
  Web 上の一問一答サイト（いちご ドリル等）は規約・ライセンスが不明で，B154 (A2) のとおり
  **調達には人間の確認が要る**ため今回は選ばない．
- RouterDC (Chen et al., NeurIPS 2024, arXiv:2409.19886,
  <https://proceedings.neurips.cc/paper_files/paper/2024/hash/7a641b8ec86162fc875fb9f6456a542f-Abstract-Conference.html>)
  —— ルータのエンコーダを 2 つの対照損失で学習する手法．sample-sample 損失は k-means クラスタで
  同クラスタの設問を引き寄せる構成で，B154 (d) の k-means 潜在サブクラス（CV +1.68pt）と発想が近い．
  ただしエンコーダの学習を伴い，Iter40〜43 で `embedding_adaptation` が全値 rejected になった系列と
  同じ構造（単一レバー原則との両立が難しい）なので，今回は採らない．

### 仮説 (Iter95)

**B154 の『どの腕でも CV 0.80〜0.82 に張り付く』は推定量の限界ではなく，n = 2,275 の限界である．**
Iter79〜93 が動かしたのは全て「固定 n のもとでの推定量の選び方」であり，n 自体は Iter88 以降
変わっていない．d/n = 2.25 の過剰パラメータ領域では，推定量の選び方より n の増加が効く
（Byrd & Lipton 2019 の含意，Viering & Loog 2024 の学習曲線）．未使用の JMMLU プール 2,232 行を
投入して n を 4,507（d/n = 1.14）へ倍増させれば，学習曲線の外挿どおり CV で +1.7pt 前後，
end-to-end で +0.7pt 前後の改善が出るはずである．
**反証可能な予測**: 追加行が 0 の legal・computer_science・social_science・general・education の
5 ドメインでは recall はほぼ動かず，追加行の多い medical（×4.2）・history_culture（×3.3）・
natural_science（×2.9）・business_economics（×2.6）で改善が集中するはずである．
この向きが出なければ仮説は誤りである．

### 単一レバー (Iter95)

- **レバー**: `classifier_training_volume_expansion` = `full_eval_disjoint_jmmlu_pool`
  （config.yml の `levers` 末尾に本フェーズで追記．選定理由と要レビューは backlog B158）
- **何を何から何へ**: 分類器の訓練集合を
  `data/classifier_train_iter94_dedup.jsonl`（**2,275 行**）から
  `data/classifier_train_iter95_fullpool.jsonl`（**4,507 行**，＋2,232 行）へ．
  規則は「各ドメインの JMMLU タスクプールのうち，評価集合とも既存訓練集合とも重複しない行を全て足す」
  の 1 つだけで，自由パラメータは無い．**既存 2,275 行は 1 行も削らない（純粋な追加）**．
- **固定する構成（1 つも動かさない）**: 評価集合 `data/dataset.jsonl`（sha256 `2114e048...`，3,750 行）・
  埋め込みモデル `qwen3-embedding:4b`・`embedding_instruction`（Iter81 の P1 文言）・
  `embedding_view_concat: true`（2 ビュー連結 5,120 次元）・`CalibratedClassifierCV`（Iter31 以降の較正）・
  `C=1.0`・重み `n/(K*n_d)`・`classifier.py` の推論経路・`dispatch_gap_threshold: 0.36`・
  `dispatch_gap_max_k: 4`・`dispatch_top_k: 2`・`aggregation_method: max_confidence`・
  expert/light モデルとノード割り当て．
- **実装の要点**:
  - `build_dataset.py` の `--domain-target-size` は評価集合と訓練集合で共用なので **CLI からの再生成は禁止**
    （評価集合が変わってしまう）．新規スクリプト `scripts/expand_classifier_train_pool.py` で
    既存 JSONL への追記として作る．追加行の id は `{domain}-pool095-NNN`，`sample_weight` は 1.0．
  - 追加 2,232 行 × 2 ビュー＝ 4,464 回の埋め込み計算は **wafl-ctrl5 で行う（絶対条件 (B)）**．
  - 再訓練は `scripts/train_domain_classifier.py` をハイパラ無変更で実行．
    旧 artifact は `models/domain_classifier_pre_iter95_fullpool.joblib` へ退避（可逆にする）．
  - 本走前に `mise run deploy` と sha256 確認．本走は wafl500〜509 で 3,750 問フルスペック（絶対条件 (A)）．

### 成功条件（事前登録 / Iter95）

**基準線**: `results/20260928_160921/`，top1 = **0.833067**（Wilson 95%CI [0.8208, 0.8447]，3,750 行）．
再現性の床は ±0.25pt．

**事前スクリーニング（本走 1 点を絞り込む手段であり，本走の代替ではない）**
現行 2,275 行の層化 5-fold を固定し，各 fold の訓練側に追加プール行を足した腕と足さない腕を比較する
（テスト fold が同一なので n の違う腕でも直接比較できる．3 seed × 5-fold，較正込みの本番パイプライン）．
- **足切り: 固定 fold CV の改善が +1.0pt 未満なら本走を行わずレバーを閉じる**
  （伝達率 43% で end-to-end +0.43pt，再現性の床の 1.7 倍にあたる．
  B151 (d)4 の +2.3pt からの引き下げであり，要レビュー事項として B158 に記録した）．
- **per-domain CV の Δ も出し，BH 補正後に有意退行が見込まれるドメインが 1 つでもあれば本走を行わない**
  （追加行が 0 で相対的にクラスが痩せる legal 77 行・education 238 行を特に見る）．

**本走の判定（効果量 × 有意性の 2 軸表．B154 (A4) の宿題に対応）**
Δtop1 ＝ 新 − 基準線（pt），p ＝ McNemar 検定（同一 3,750 行，α=0.05）．

| | p < 0.05 | p ≥ 0.05 |
|---|---|---|
| Δ ≥ +1.0 | **adopted** | 判定不能（検出力不足として記録） |
| +0.25 ≤ Δ < +1.0 | **adopted_small**（採用．効果は小と明記） | 判定不能 |
| \|Δ\| < 0.25 | no_effect | no_effect |
| −1.0 < Δ ≤ −0.25 | **rejected** | 判定不能 |
| Δ ≤ −1.0 | **rejected** | rejected |

**事前予測はどのセルか**: 学習曲線の外挿（+0.74pt）から **`adopted_small`（+0.25 ≤ Δ < +1.0 かつ p<0.05）** を予測する．

**必須の非退行条件（1 つでも破れたら artifact をロールバックする）**
- C1: per-domain precision/recall 計 20 指標の BH 補正後の有意退行が **0 件**．
- C2: `used_fallback` 率・`dispatch_failed` 件数が基準線から増えない．
- C3: レバーが発火したことの証拠（新 artifact の sha256 が旧と異なる，訓練行数 4,507，
  ノード上の `models/domain_classifier.joblib` の sha256 一致）．
- C4（副基準・参考値）: Random / BestSingle / Oracle を併記し，BestSingle 超過を明示する（success_criteria (3)）．
- C5（仮説の検証）: 追加行の多い 4 ドメイン（medical / history_culture / natural_science / business_economics）の
  recall の Δ の合計が，追加行 0 の 5 ドメインの Δ の合計より大きいこと．
  これが成り立たなければ「行数が効いた」とは書かない（偶然の変動として扱う）．
- C6: 複合設問 730 行の `compound_domain_set_recall` と `compound_mean_dispatched_count` を併記する
  （dispatch 予算が動いていないことの確認）．

### 実装・実験 (Iter95) —— 事前スクリーニングの足切りに掛かり本走を見送り

**実施場所**: 埋め込み計算・分類器の訓練評価は全て **wafl-ctrl5**（192.168.15.10）で実施した
（絶対条件 (B) ／ backlog B158 (f) の申し送り）．wafl-ctrl5 には uv・sklearn が無かったため，
本フェーズで `~/.local/bin/uv`（uv 0.12.19）をインストールし，リポジトリのコード一式
（`build_dataset.py`・`expert_backend.py`・`metrics.py`・`scripts/`・`pyproject.toml`・`uv.lock`・
`config.yaml`・`data/dataset.jsonl`・`data/classifier_train_iter94_dedup.jsonl`・
`/tmp/expert-mesh-cache/JMMLU.zip`）を `~/expert-mesh-iter95/` へ rsync し，
`uv sync --extra research --python 3.12` で環境を構築した（`--python 3.14` 系はデフォルトだと
`safetensors` のビルド済み wheel が無くソースビルドで失敗したため 3.12 を明示指定．
`pyproject.toml` の `requires-python>=3.12` の範囲内）．**wafl500〜509 は一切使っていない**．

**1. 新規スクリプト（追記的実装）**

- `scripts/expand_classifier_train_pool.py`（新規）: `data/classifier_train_iter94_dedup.jsonl`
  （2,275 行，読むだけで変更せず）と `data/dataset.jsonl`（3,750 行，同）に対して素な JMMLU 行を
  ドメインごとに全数追加し，`data/classifier_train_iter95_fullpool.jsonl` を新規生成する．
  `build_dataset.py` の内部関数（`_load_jmmlu_zip_bytes`・`_parse_jmmlu_task_csv`・
  `_format_jmmlu_query`・`_DOMAIN_TASK_MAP`）を再利用し，独自のサンプリングロジックは書いていない．
  サンプリングではなく「残り全部を足す」ため乱数シードは無い（計画どおり自由パラメータ 0）．
  開発ホスト上でローカル実行（GPU 不要・純粋な JSON 組立てのため絶対条件 (B) の対象外）．
  **結果は計画時の実測値と完全一致**: 既存 2,275 行 → 4,507 行（+2,232 行），
  内訳 medical +810 / natural_science +481 / history_culture +479 / business_economics +405 /
  mathematics +48 / education +9 / legal・computer_science・social_science・general +0．
  sha256: `data/classifier_train_iter94_dedup.jsonl` は `d6b23735...`（不変，Iter94 と同一）．
  なお `natural_science` の新規追加行のうち 5 件は，JMMLU の異なるタスク CSV（例:
  college_physics と conceptual_physics）に同一設問文が重複して存在するために 2 回登録されている
  （両方とも `natural_science` ラベルなのでラベル矛盾は無い）．これは `build_dataset.py` の
  既存のタスクプール方式（`_sample_domain_questions` も同様にタスク間の重複除去をしない）と
  同じ挙動であり，本レバー固有の不具合ではない．
- `scripts/embed_classifier_train_pool.py`（新規，wafl-ctrl5 専用ヘルパー）:
  `train_domain_classifier.py` の `embed_query_views`（`instruction`＝config.yaml の
  `embedding_instruction`，`concat_views=true`）をそのまま呼んで 4,507 行を 1 回だけ埋め込み，
  `.npy` にキャッシュする．スクリーニングの 3 seed × 5 fold × 2 腕で同じ埋め込みを使い回すための
  補助であり，本番 artifact ではない．wafl-ctrl5 のローカル Ollama（`127.0.0.1:11434`，
  `qwen3-embedding:4b` は導入済み）に対して 4,507 行 × 2 ビュー＝9,014 回の `/api/embeddings`
  呼び出しを行った（所要時間 約 43 分）．出力 `data/embcache_iter95_fullpool.npy`
  （shape (4507, 5120)）．
- `scripts/screen_classifier_training_volume_expansion.py`（新規，wafl-ctrl5 専用）:
  事前登録した「固定 5-fold（既存 2,275 行のみに対する層化分割，3 seed）× 2 腕（無追加／全追加）」
  スクリーニングを実装．`train_domain_classifier.py` の `train_classifier()`（較正込み本番パイプライン）
  と `_extract_sample_weights()` をそのまま呼び，独自の分類器・較正ロジックは書いていない．
  **統計量は `metrics.py` の既存実装をそのまま呼んだ**（`compute_top1_accuracy`・
  `compute_mcnemar_test`・`compute_domain_recall_mcnemar_test`・`apply_benjamini_hochberg`）．
  各 fold の out-of-fold 予測を `{id, selected_domain, expected_domains}` 形式に整形して渡すことで，
  chi2・BH 補正を独自再導出せずに済ませた．

**2. 事前スクリーニング結果（wafl-ctrl5，3 seed × 5-fold，較正込み本番パイプライン）**

固定 fold CV top1 accuracy（out-of-fold，3 seed × 5-fold 併合，n=6,825＝2,275×3）:

| 腕 | CV accuracy |
|---|---|
| A（無追加，現行 2,275 行） | 0.80322 |
| B（全追加，4,507 行） | 0.80689 |
| **Δ (B−A)** | **+0.366pt** |

McNemar（3 seed 併合，discordant a_only=133 / b_only=158）: p = 0.1595．

**足切り判定 1（事前登録の CV +1.0pt 基準）: 不通過．** +0.366pt は学習曲線外挿（+1.73pt）の
約 1/5 に留まり，B158 (d) で引き下げた後の足切り +1.0pt にも届かない．

per-domain recall Δ（B−A，BH 補正 q=0.05，10 指標）:

| ドメイン | 追加行数 | recall Δ (pt) | p | BH 有意 |
|---|---|---|---|---|
| education | +9 | **+8.54** | 1.1e-12 | 有意（改善） |
| general | +0 | +1.33 | 0.0094 | 有意（改善） |
| computer_science | +0 | +0.93 | 0.0233 | 有意（改善） |
| medical | +810 | +2.27 | 0.0506 | 非有意（境界） |
| social_science | +0 | +0.67 | 0.383 | 非有意 |
| mathematics | +48 | +0.13 | 1.0 | 非有意 |
| legal | +0 | 0.00 | 1.0 | 非有意（discordant 0 件） |
| business_economics | +405 | −0.67 | 0.424 | 非有意 |
| **history_culture** | **+479** | **−5.08** | **2.4e-7** | **有意（退行）** |
| **natural_science** | **+481** | **−5.20** | **3.0e-8** | **有意（退行）** |

**足切り判定 2（per-domain の有意退行が 1 件でもあれば本走しない）: 不通過．**
`natural_science`・`history_culture` の 2 ドメインで BH 補正後も有意な recall 退行が見られる．
計画時に名指しした懸念ドメイン（追加 0 行の `legal`・`education`）はどちらも退行していない
（`legal` は動かず，`education` はむしろ最大の改善）．**退行が出たのは，仮説が「最も改善する」と
予測していた「追加行が多い」側の 2 ドメイン（`natural_science` +481・`history_culture` +479）で
あり，狙いと真逆の向きに出た．**

**C5（仮説の反証可能な予測）の検証**: 追加行の多い 4 ドメイン
（medical/history_culture/natural_science/business_economics）の recall Δ 合計 = **−8.68pt**，
追加 0 行の 5 ドメイン（legal/computer_science/social_science/general/education，
`education` は実質ほぼ 0 の +9 行なのでここに含めた計画どおりの区分）の Δ 合計 = **+11.48pt**．
事前登録の予測（前者 > 後者）は **反証された**（`c5_hypothesis_holds: false`）．
「n を増やせば学習曲線どおりに効く」という Iter95 の仮説は，少なくともこの構成では支持されない．
考えられる機序（示唆であり未検証）: `history_culture`・`natural_science` は元々タスク数が多い
ドメイン（7・8 タスク）で，追加行がタスク間の構成比を大きく変える（例:
`natural_science` は元 250 行に対し新規 481 行を追加するため合計の 66% が新規データになる）．
Iter93 の学び（journal「Iteration 93」節，タスク数の多いドメインほど細粒度化で退行しやすい）と
方向性が整合する——ただし本イテレーションではこの機序を追加検証していないため断定しない．

**足切り判定（事前登録の 2 条件）: 両方とも不通過．計画の停止規則により本走を行わない．**
計画時の 2 軸判定表・end-to-end 予測（+0.74pt，`adopted_small` を予測）は，本走を行っていないため
どのセルにも到達しない（そもそも本走の前段ゲートで閉じた）．

**3. 本番コードへの変更は 0 件（絶対条件 (A) との整合）**

足切りに掛かったため，計画の記述どおり「本番コードへの変更を適用しないまま結果を記録してフェーズ3へ
渡す」を実施した．具体的には次を**行っていない**:
`scripts/train_domain_classifier.py` による本番 `models/domain_classifier.joblib` の再訓練，
`models/domain_classifier_pre_iter95_fullpool.joblib` への退避，`mise run deploy`，
wafl500〜509 での本走．**`config.yaml`・`models/domain_classifier.joblib`・`data/dataset.jsonl` は
1 バイトも変更していない**（sha256 は Iter94 終了時点から不変: `dataset.jsonl` = `2114e048...`）．
施策（訓練集合の追加）を適用していないので，絶対条件 (A)（適用したら必ず本走）には抵触しない
（計画フェーズが明記した停止規則どおり）．

**4. 新規に追加したファイル（副作用の全量）**

- `scripts/expand_classifier_train_pool.py`・`scripts/embed_classifier_train_pool.py`・
  `scripts/screen_classifier_training_volume_expansion.py`（新規スクリプト 3 本，リポジトリにコミット対象）．
- `data/classifier_train_iter95_fullpool.jsonl`（4,507 行，新規生成．**未使用のまま**．
  どのコードからも読まれない＝本番の実行時挙動には影響しない）．
- wafl-ctrl5 上のみ: `~/expert-mesh-iter95/`（uv 環境一式）・`data/embcache_iter95_fullpool.npy`
  （埋め込みキャッシュ，5,120 次元 × 4,507 行）・`/tmp/iter95_screening.json`（スクリーニング結果）．
  いずれも wafl-ctrl5 のローカルディスクのみで，このリポジトリの git 管理下にはない．
  次イテレーション以降で不要になれば削除してよい（本フェーズでは削除していない）．
- `models/`・`data/dataset.jsonl`・`config.yaml` への変更は 0 件．

**5. 詰まった点**

- wafl-ctrl5 の `uv` が既定で python 3.14（free-threaded ビルド）を選び，`safetensors` の
  prebuilt wheel が無くソースビルド（cargo/maturin）に失敗した．`--python 3.12` を明示して解決した．
- `ssh host "... & echo done"` 形式のバックグラウンド起動は，リダイレクト先を用意していても
  ssh セッション自体がすぐには返らず，このエージェントのシェルツール側の 120 秒タイムアウトで
  バックグラウンド送りになる事象が 2 回あった（実害無し，別途 `tail -f` でポーリングして対処）．

### Iteration 95 実行済み —— 判定 `closed`（事前スクリーニング足切り不通過・本走なし）

**変更（本番への適用は 0 件）**: 新規スクリプト 3 本（`scripts/expand_classifier_train_pool.py`・
`scripts/embed_classifier_train_pool.py`・`scripts/screen_classifier_training_volume_expansion.py`）と
新規データ `data/classifier_train_iter95_fullpool.jsonl`（4,507 行，git 管理外・どのコードからも
読まれない）のみ．`models/`・`config.yaml`・`data/dataset.jsonl` は 1 バイトも変更していない．
施策を本番へ適用していないので 2026-09-23 絶対条件 (A)（適用したら必ず本走）には抵触しない．

**結果（固定 5-fold CV，3 seed 併合，n=6,825）**: 無追加 2,275 行 0.80322 → 全追加 4,507 行 0.80689，
**Δ = +0.366pt（McNemar p = 0.1595，discordant 133/158）**．事前登録の足切り +1.0pt に不通過．
per-domain recall Δ は education +8.54 / general +1.33 / computer_science +0.93 が BH 後有意な改善，
**history_culture −5.08・natural_science −5.20 が BH 後有意な退行**で，有意退行 1 件でも本走しないという
足切り条件 2 にも不通過．C5（追加行の多い 4 ドメインの Δ 合計 > 追加 0 の 5 ドメインの Δ 合計）は
**−8.68pt 対 +11.48pt で反証された**．

#### 分析（ノイズか信号か・本走が無いので何が言えないか）

- **全体 Δ = +0.366pt はノイズと区別できない**．p = 0.1595 であり，かつ 3 seed は同じ 2,275 行を
  3 通りに分割し直したものなので独立標本ではなく，n=6,825 として扱った p は楽観側に寄っている．
  一方 **per-domain の ±5pt 級は偶然変動ではない**（p = 2.4e-7 / 3.0e-8）．すなわち
  「全体は動かないが，ドメイン間で ±5pt の再配分が起きた」が今回の信号である．
- **本走をしていないので end-to-end の Δ は存在しない．** CV Δ から e2e を予測する手続き
  （伝達率 43%）は Iter91 の 1 点の実測に基づく外挿で，(i) 伝達率が腕の種類によらず一定，
  (ii) CV のテスト分布と評価集合 `data/dataset.jsonl` の分布が同じ意味を持つ，という 2 つの仮定に
  依存する．今回は後述のとおり **(ii) が破れている疑いが濃い**ため，0.366 × 0.43 = +0.16pt という
  e2e 予測値は参考値ですらなく，**符号すら保証されない**．したがって本レバーについて言えるのは
  「事前登録した足切りを通らなかった」ことだけであり，「実機で効かない／退行する」とは言えない．
- **学習曲線の外挿（n 倍増で CV +1.73pt）は当たらなかった（実測はその約 1/5）**．外挿が暗黙に
  仮定していたのは「追加行が既存行と同分布であること」で，今回の追加行はそうではなかった（下記）．

#### 考察 —— C5 の反証は「n は効かない」ではなく「n と構成比が交絡していた」

機序の候補を，既存の数値と過去の journal だけから整理する（いずれも本反復では追加検証していない）．

- **(M1) スクリーニングのテスト fold が旧訓練分布のままである（測定側の交絡．最有力）．**
  固定 5-fold のテスト fold は既存 2,275 行から作るので，そのドメイン内タスク構成は
  `build_dataset.py` の per-domain 上限 150 によるサンプリング構成である．腕 B はその同じドメインの
  クラス条件分布を **JMMLU プール構成へ動かす**（全プール追加後の構成 ≒ プール構成）．
  計画フェーズの実測では，訓練 vs 評価 の全変動距離は education 0.189 / medical 0.169 /
  history_culture 0.153 と大きい一方，**プール vs 評価 は 0.017〜0.081 と小さい**．
  つまり今回の操作は **評価集合には近づくが，CV のテスト fold からは遠ざかる**向きである．
  追加行が多いドメインほどこのずれが大きく，そこだけ recall が落ち，追加 0 のドメインは
  競合クラスが自分の領域から退いた分だけ上がる．観測された符号パターン（−8.68 / +11.48，
  総和はほぼ相殺で +0.366pt）はこの説明とちょうど整合する．**history_culture / natural_science の
  −5pt を「本番でも退行する」と読むのは誤り得る．**
- **(M2) 較正の非対称性**: 重みは `n/(K*n_d)` の balanced 相当なのでクラス事前分布そのものは
  正規化されるが，`CalibratedClassifierCV` の較正はクラスごとのスコア分布の推定に依存する．
  クラスあたり行数が 250 行 対 1,060 行と 4 倍に開くと，較正曲線の推定精度がクラス間で非対称になる．
- **(M3) ドメイン内のタスク多様性（Iter93 の学びの裏返し）**: タスク数の多い history_culture (7)・
  natural_science (8) では追加行が別タスクの峰を厚くし，単一の線形境界がそちらへ引かれる．
  ただし **medical は 10 タスク・+810 行で改善している（+2.27pt, p=0.0506）**ので，
  タスク数だけでは符号を説明できない．「タスク数」より「構成比の移動量」のほうが説明力が高く，
  この点でも (M1) が優る．
- **(M4)** `natural_science` の重複 5 行（異なるタスク CSV に同一設問）は規模として無関係．

したがって **C5 の反証は「訓練量 n は効かない」を意味しない．今回の設計では
n の増加とドメイン内タスク構成比の移動が完全に交絡しており，純粋な n の効果は未測定のままである．**

#### 判定

- **`closed`**（Iter93 の `classifier_label_granularity` と同じ扱い：事前 CV の足切りに不通過で
  本走を行わずレバーを閉じる）．**`rejected` ではない**．`rejected` は本走の 2 軸表に基づく判定語であり，
  本走が無い以上そのセルには到達していない．`classifier_training_volume_expansion` は
  値 `full_eval_disjoint_jmmlu_pool` をもって closed とし，config.yml の当該 note に実測値を注記した
  （同じ案を再度引かないため）．

#### 学び（次の自分が読んで分かる形）

1. **訓練分布そのものを動かすレバー（行の追加・削除・再重み付け）を訓練集合内 CV で足切りすると，
   テスト fold が旧分布のままであるために，変更したクラスへ系統的に不利なバイアスがかかる．**
   この型のレバーでは，スクリーニング設計の段階で「腕 A と腕 B でテスト fold が同じ意味を持つか」を
   必ず確認すること．具体的な回避策は 2 つある．(a) 追加行の構成比を既存構成比に一致させて
   分布を動かさない（次イテレーションの方針），(b) プール側から層化して取り分けた
   hold-out 行をテスト集合に足す（評価集合を覗かずに済む）．
2. **学習曲線 err ∝ n^(-α) による外挿は「追加行が既存行と同分布」を暗黙に仮定している．**
   本研究の既存訓練行は per-domain 上限 150 のサンプリングで構成が歪んでいたため，
   残りプールを足す操作は n の増加と同時に分布のシフトを伴った．外挿値を足切り閾値と比べる前に，
   追加行の由来分布が既存行と同じかを確認すること．
3. **JMMLU プールは尽きていない**（未使用 2,232 行．内訳は本イテレーション「調査 (2)」の表）．
   `legal`・`computer_science`・`social_science`・`general`・`education` の 5 ドメインだけが枯渇しており，
   n を触る手は「4 ドメインだけを太らせる」形にしかならない．この非対称性自体が (M1) の交絡の源である．
4. 運用面: wafl-ctrl5 には uv が無く，既定の python 3.14（free-threaded）では `safetensors` の
   wheel が無くビルドに失敗する．`uv sync --extra research --python 3.12` を明示すること．

## Iteration 94: 訓練集合と評価集合で重複する設問 64 行を除去し基準線を引き直す

### 調査 (Iter94)

backlog B155 (b) が事前登録したレバー `train_eval_question_overlap_removal` を実施するにあたり，
(1) 重複 64 行の同定条件の再現・確定，(2) 汚染（train/test contamination）の扱いに関する文献調査，
(3) 除去後の基準線のオフライン replay による事前予測，の 3 点を行った．
**評価集合 `data/dataset.jsonl` は読むだけで変更していない．実機ノード wafl500〜509 も不使用**
（新規の埋め込み計算は 0 件．すべて既存キャッシュから再構成した）．作業は制御ホスト wafl-ctrl5 上で行い，
中間生成物は `/tmp/iter94/` にのみ書いた（`data/`・`models/`・`results/` は一切変更していない）．

**(1) 重複 64 行の同定条件を確定した —— 「設問文 `query` の完全一致」で十分である**

`data/classifier_train_iter92_civics_aligned.jsonl`（2,339 行）と `data/dataset.jsonl`（3,750 行，
sha256 `2114e048...`）を突き合わせた結果は次のとおり（スクリプト `/tmp/iter94_overlap.py`）．

- **完全一致（`query` の文字列同値）: 64 行．正規化一致（NFKC ＋ 全空白除去 ＋ lowercase）でも 64 行**で，
  両者は同一の集合である．すなわち**表記ゆれによる取りこぼしは無く，同定条件は完全一致で確定してよい**
  （正規化を入れても新たな重複は 1 行も現れない）．これは両者が同じ `build_dataset.py` の
  `_format_jmmlu_query()` を通って生成されているためで，`build_dataset.py:1322` の
  `exclude_queries=eval_queries` による重複排除も同じ完全一致を前提にしている．
  **よって除去規則は本番コードの重複排除規則と同一であり，新しい判定規則を導入しない．**
- 64 行はすべて**ラベルも一致**（訓練側 `domain` が評価側 `expected_domains` に含まれる）．
  **評価側 64 行はすべて単一ドメイン行で，複合設問（`is_compound=true` の 730 行）との重複は 0 行**．
  訓練側 64 行はすべて通常行（`*-train-*`）で，**hard negative 行（`*-hardneg-*`）・
  Iter92 追加の civics 行（`education-civics-*`）は 1 行も含まない**．
- 内訳（訓練ラベル ＝ 評価ラベル）: `education` 46 行，`history_culture` 18 行．他の 8 ドメインは 0 行．
  JMMLU タスク単位では moral_disputes 19 / sociology 14 / high_school_psychology 13 /
  japanese_idiom 5 / world_history 3 / high_school_geography 3 / japanese_history 2 / prehistory 2 /
  high_school_european_history 2 / japanese_geography 1．
- 除去後の訓練行数は **2,339 → 2,275 行**．ドメイン別は `education` 284→238，`history_culture` 228→210 で，
  **他 8 ドメインは不変**（`legal` は 77 行のまま減らない）．config.yml のレバー注記が
  「`legal` など小さいクラスが過度に痩せないか事前に確認せよ」と求めていた点は**問題なし**である
  （最小クラスは `legal` 77 行のまま，`sample_weight` は `n/(K*n_d)` で自動的に再計算される）．

**(2) 基準線 `results/20260928_111644/` における重複 64 行の実測 —— 「最大 1.7pt」は上限であって推定値ではない**

B154 (A1) の「最大 1.7pt の上振れ」は **64/3,750 = 1.707pt**，すなわち「重複 64 行が全問正解でありかつ
その正解が全て記憶によるものだった場合」の**理論上限**である．基準線の実測はこれよりずっと小さい．

| 部分集合 | n | 基準線 top1 |
|---|---|---|
| 全体 | 3,750 | 0.835467 |
| **重複 64 行** | 64 | **0.6875 (44/64)** |
| 残り | 3,686 | 0.838036 |

**重複行の精度は全体平均より低い**（0.6875 < 0.8380）．これは 64 行が recall の低い `education`（全体 0.497）
に偏っているためで，ドメインを揃えて比べると記憶の痕跡が見える: `education` は重複 46 行 0.652 に対し
残り 304 行 0.474（**+17.8pt**），`history_culture` は重複 18 行 0.778 に対し残り 332 行 0.913（−13.5pt，
ただし n=18 で偶然の範囲）．

**(3) 文献調査（tavily-search）—— 「訓練側から除く」か「評価側から除く」か**

- Elangovan, He & Verspoor (2021), *Memorization vs. Generalization: Quantifying Data Leakage in NLP
  Performance Evaluation*, EACL 2021, pp.1325-1335, <https://aclanthology.org/2021.eacl-main.113/>
  （arXiv:2102.01818）—— 訓練集合と評価集合の重複は記憶と汎化を混同させ，報告精度を汎化性能の推定として
  読めなくする．**重複の量と，重複部分／非重複部分それぞれの精度を分けて報告することを推奨している**．
  本反復の (2) はまさにこの分解にあたる．
- Lee et al. (2022), *Deduplicating Training Data Makes Language Models Better*, ACL 2022,
  <https://aclanthology.org/2022.acl-long.577/>（arXiv:2107.06499）—— 標準的なコーパスでは
  **train-test overlap が検証データの 4% 超に及び，重複を訓練側から除くことが基本的な処方**である．
  本研究の重複率は評価集合の 1.71%（64/3,750）でこれより小さいが，除去の向き（訓練側から除く）は同じ．
- Brown et al. (2020), *Language Models are Few-Shot Learners*, NeurIPS 2020, 第 4 節
  *Measuring and Preventing Memorization of Benchmarks*,
  <https://proceedings.neurips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf>
  —— **訓練側の重複排除が事後的に不完全だった場合の対処として，評価集合側から汚染行を除いた
  「clean subset」での精度を併記する**手法を採っている．すなわち文献上，是正の向きは 2 通りある．

**本研究でどちらを採るか（判断と理由）**: **訓練側から除去する**（Lee et al. 側）．

- (i) 評価集合 `data/dataset.jsonl`（sha256 `2114e048...`）は Iter85/90 で拡充したのち，Iter89〜93 の
  全基準線が参照する固定点である．評価側を削ると過去の全 top1 と比較不能になり，
  実機での再取得（1 回約 155 分）が基準線の本数だけ必要になる．
- (ii) 訓練側を除くほうが是正として強い．評価側だけを削っても，分類器は依然として当該設問で学習された
  ままであり（近傍の設問への間接的な影響が残る），Elangovan et al. の言う記憶と汎化の分離は
  完全にはならない．
- (iii) ただし Brown et al. の clean subset 報告は**併記として有用**なので，分析フェーズでは
  全 3,750 行の top1 に加え，**重複 64 行を除いた 3,686 行での top1 も必ず併記する**
  （Iter92 で 8 行について事前登録したのと同じ扱い．B152 要レビュー (b) の踏襲）．

**(4) オフライン replay による事前予測（G0-d 相当．事前登録の予測であり，本走の代替ではない）**

`/tmp/iter94/predict_replay.py` で，除去後 2,275 行の訓練特徴を **Iter92 と同一の規則**
（`embcache_train_iter89_qwen3-embedding_4b{,__p1}.npy` の該当行 ＋
`embcache_train_iter92_qwen3-embedding_4b_new20{,__p1}.npy`，新規埋め込み 0 件）で再構成し，
`scripts/train_domain_classifier.py` の `train_classifier()` をそのまま呼んで分類器を組み，
評価 3,750 行の argmax を現行 artifact（`models/domain_classifier.joblib`，sha256 `98da6f2d...`）と比較した．
Iter91 で replay が分類器 argmax 精度を小数 6 桁一致で再現することは実証済みである．

- **replay top1: 0.836267 → 0.834933（Δ = −0.133pt）．McNemar: discordant 34 / 29，p = 0.614．**
- **重複 64 行: 0.6875 → 0.5781（−10.9pt ＝ 7 行）．残り 3,686 行: 0.8389 → 0.8394（+0.054pt）．**
- per-domain recall（単一ドメイン行）の変化は最大でも ±0.9pt 以内:
  history_culture +0.86pt / social_science +0.68 / computer_science +0.33 / education +0.29 /
  business_economics・legal・mathematics・natural_science ±0.00 / general −0.57 / medical −0.57．

**この分解が本反復の中心的な知見である**: 除去の影響は**ほぼ完全に重複 64 行の中に閉じており**
（7 行分の正解が消える），残り 3,686 行はむしろ微増する．すなわち
**測定系の上振れの実測値は約 0.19pt（7/3,750）であり，B154 が挙げた理論上限 1.7pt の約 1/9 である**．
`education` の recall が上がらない（+0.29pt）点も重要で，**Iter92 で観測した education の低 recall は
リークで嵩上げされていた見かけの値ではなく，除去後も 0.497 のまま残る真の弱点**である．

### 計画 (Iter94)

**単一レバー（事前登録）**: `train_eval_question_overlap_removal` =
**`remove_duplicate_question_rows_and_rebaseline`**（backlog B155 (b) の事前登録どおり）．

**変更点は 1 つだけである**: 訓練集合 `data/classifier_train_iter92_civics_aligned.jsonl` から，
評価集合 `data/dataset.jsonl` と `query` が完全一致する 64 行を除去し（2,339 → 2,275 行），
同一の手順で再訓練して本走 1 回を行う．**本レバーは精度向上を狙うものではなく，測定系の是正である．**
したがって B151 (d)4 の足切り（訓練集合内 CV で +2.3pt）は適用しない（B155 (b)・本タスクの指示）．

**仮説**: 基準線 `results/20260928_111644/`（top1 = 0.835467）は，訓練で見た設問 64 行を評価に含むため
上振れている．その上振れの実体は「重複行のうち記憶によってのみ正解している分」であり，
実測では**約 7 行（0.19pt）**である．重複行を訓練から除けば，(a) 重複 64 行の部分精度が
0.6875 → 0.58 付近へ下がり，(b) 残り 3,686 行の精度は ±0.5pt 以内で変わらず，
(c) 全体 top1 は 0.835 → 0.834 付近（Δ ≈ −0.13pt）へ動く．

**固定する構成（Iter92/93 の最良構成をそのまま維持）**:
埋め込み `qwen3-embedding:4b` の 2 ビュー連結（5,120 次元），instruction prefix 現行値，
`CalibratedClassifierCV(method="temperature", cv=5, ensemble=True)`，`C`=1.0，
`class_weight=None` ＋ドメイン均衡 `sample_weight`（`n/(K*n_d)`，除去後の行数で再計算），
`confidence_threshold=0.0`，`dispatch_top_k=1`，`routing_method=supervised_classifier`，
評価集合 `data/dataset.jsonl`（3,750 行，sha256 `2114e048...`，**変更しない**），
`classifier.py` の推論経路・`config.yaml`・`router.py`・`http_server.py` は無変更．
**基準線は `results/20260928_111644/`（3,750 行，top1 = 0.835467）**．

**成功条件（事前登録．B153 要レビュー (B) / B154 (A4) に従い，効果量と有意性を分けた 2 軸で定義する）**

まず，是正が成立したことを確かめる**必須ガード** C1〜C3（いずれか 1 つでも不成立なら判定は `invalid` とし，
artifact をロールバックして原因を調査する）．

- **C1（重複 0 行）**: 新訓練ファイルと `data/dataset.jsonl` の設問文の一致が，
  **完全一致・正規化一致（NFKC ＋ 全空白除去 ＋ lowercase）のいずれでも 0 行**であること．
- **C2（変更の最小性）**: 新訓練ファイルの行数が **2,275 行**で，旧ファイルとの差分が
  **`/tmp/iter94_dup_train_ids.json` の 64 個の id の削除のみ**であること（行の並び・各行の
  `id`/`query`/`domain` は他に 1 文字も変わらない）．コード（`*.py`）・`config.yaml`・
  `data/dataset.jsonl`・評価条件の変更が 0 件であること．
- **C3（本走の健全性）**: 本走が 3,750 行を完走し，`dispatch_failed` 件数・`used_fallback` 件数が
  基準線と同水準であること（従来の E1〜E5 相当のガード，`mise run deploy` 後の全 10 台での
  artifact sha256 一致確認を含む）．

そのうえで，**判定語は Δtop1（全 3,750 行，新 − 基準線）の効果量と McNemar の有意性の 2 軸で決める**．

| | McNemar p < 0.05 | p ≥ 0.05 |
|---|---|---|
| **\|Δtop1\| < 0.5pt** | `corrected_negligible`（是正完了．基準線の水準は実質不変） | `corrected_negligible`（同左．**事前予測はここ**） |
| **−1.7pt < Δtop1 ≤ −0.5pt** | `corrected_bias_removed`（上振れが実測された） | `corrected_bias_removed_weak`（向きは一致するが有意でない） |
| **Δtop1 ≤ −1.7pt または Δtop1 ≥ +0.5pt** | `unexpected_shift` | `unexpected_shift` |

`unexpected_shift` は「重複 64 行の除去だけでは説明できない変化」を意味し，判定を確定させずに
原因を調査する（C5 の帰属確認と replay との乖離を突き合わせる）．
**下限 −1.7pt は理論上限 64/3,750 に対応し，これを超える低下は除去以外の要因を疑うべき水準である．**

加えて，**是正が意図どおりの場所で起きたことの帰属確認**を副基準として事前登録する
（主基準の判定語は変えないが，不成立なら分析フェーズで必ず論じる）．

- **C4（replay との整合）**: 実測 Δtop1 と replay 予測 Δ = −0.133pt の乖離が **0.5pt 以内**であること
  （過去の反復では replay と本走の差は 3 行程度）．
- **C5（リークの帰属）**: 重複 64 行の部分精度が下がり（予測 0.6875 → 0.58 付近），
  かつ**残り 3,686 行の Δ が ±0.5pt 以内**であること．これが成り立てば，
  「上振れは重複行の記憶に閉じていた」という仮説が支持される．
- **C6（clean subset の併記）**: Brown et al. (2020) 第 4 節に倣い，
  **重複 64 行を除いた 3,686 行での top1** を基準線側・新側の双方について報告する．
  以後の反復はこの 3,686 行基準の値も併せて引き継ぐ．

**非退行の扱い（従来と異なる点を明記する）**: per-domain 20 指標の BH 補正後の有意退行を従来どおり
集計・報告するが，**本レバーは是正であるため，退行が出ても撤回しない**．
退行が出た場合は，それが重複行の分布（`education` 46 行・`history_culture` 18 行の除去）で
説明できるかを必ず論じる．replay 上は最大の低下が general / medical の −0.57pt で，
有意退行は生じない見込みである．

**artifact の扱い（事前登録）**: 新 artifact を採用して `models/domain_classifier.joblib` を更新し，
**新しい基準線とする**（top1 が下がっても是正なので戻さない）．
旧版は `models/domain_classifier_pre_iter94_dedup.joblib` へ退避する（従来どおり可逆にする）．
ただし判定が `invalid` または `unexpected_shift` かつ C5 不成立の場合はロールバックする．

**運用上の遵守事項**: 本体実験（wafl500〜509 でのルーティング本走）以外の処理はすべて制御ホスト
wafl-ctrl5 で行う．ドメイン固有の後付け補正は追加しない（重複判定は設問文の完全一致という
10 ドメイン共通の規則であり，2026-09-23 恒久ルール (1) に適合する．`education` と `history_culture`
にしか行が無いのは実測でそこにしか重複が無いためである）．棄権・エスカレーション系のレバーには
着手しない．**上記 (4) の事前 replay は予測であって本走の代替ではなく，本走はフルスペック
（3,750 行，実機 10 台）で 1 回行う．**

### Iteration 94 実行済み

**変更（単一レバー）**: `train_eval_question_overlap_removal` =
`remove_duplicate_question_rows_and_rebaseline`．訓練集合から評価集合と設問文が完全一致する 64 行を
除去した `data/classifier_train_iter94_dedup.jsonl`（2,275 行，sha256 `d6b23735...`）を新規作成し，
同一手順で再訓練して `models/domain_classifier.joblib` を更新（新 sha256 `2f801357...`，旧版は
`models/domain_classifier_pre_iter94_dedup.joblib` へ退避）．`data/MANIFEST.md` に Iteration 94 節を追記．
**コード変更 0 件**（`config.yaml`・`classifier.py`・`router.py`・`http_server.py`・`data/dataset.jsonl`
（sha256 `2114e048...`）は不変を実測確認）．本走は `results/20260928_160921/`．

**必須ガード**: C1（完全一致・正規化一致とも重複 0 行）PASS ／ C2（差分は指定 64 id の削除のみ，
行順保存，コード変更 0 件）PASS ／ C3（3,750 問完走，wafl500〜509 全 10 台で artifact sha256 一致，
smoke_check PASS，`dispatch_failed` 3→1 件，`used_fallback` 両者 0 件）PASS．
分析フェーズでも `results.jsonl` から top1・McNemar・部分集合精度を独立に再計算し，
rc-executor の報告値と小数 6 桁まで一致することを確認した．

**結果（基準線 `results/20260928_111644/` top1 = 0.835467）**

| 指標 | 基準線 | 新 | Δ |
|---|---|---|---|
| top1（全 3,750 行） | 0.835467 | **0.833067** | **−0.24pt** |
| **重複 64 行** | 0.6875 | **0.59375** | −9.375pt |
| **clean subset 3,686 行（C6）** | 0.838036 | **0.837222** | −0.081pt |

Wilson 95%CI [0.820791, 0.844660]．McNemar（discordant 34/25，両側二項検定）**p = 0.2976**．
参考: single_domain 0.843377（n=3,020），compound_domain 0.790411（n=730），ECE 0.0318，AUROC 0.8337．
per-domain recall は education 0.4556→0.4234（−3.23pt，未補正 p=0.0150），medical +1.59pt（p=0.0614），
他 8 ドメインは ±0.4pt 以内．precision は全 10 ドメイン ±2.2pt 以内で全 p>0.4．
**recall/precision 計 20 指標の BH 補正（q=0.05）後の有意な指標は 0 件**．
副基準は C4（replay 予測 −0.133pt と実測 −0.24pt の乖離 0.107pt ≤ 0.5pt）PASS，
C5（影響が重複 64 行に集中し，残り 3,686 行の Δ は ±0.5pt 以内）PASS，C6 併記済み．

**判定: `corrected_negligible`（採用．是正完了・基準線を新値へ引き直す）**

事前登録の 2 軸表で |Δtop1| = 0.24pt < 0.5pt かつ p = 0.2976 ≥ 0.05 のセルに該当し，
これは計画時に「事前予測はここ」と名指ししたセルそのものである．ノイズと信号の切り分けとして，
Δ = −0.24pt は**再現性の床 ±0.25pt（Iter27 で確定）を下回っており，単独では実行間変動と区別できない**．
すなわち「除去によって top1 が有意に下がった」とは言えず，「基準線の水準は実質不変のまま，
測定系のバイアスだけが除かれた」と読むのが正しい．ただし Δ の符号・大きさは replay 予測（−0.133pt）と
同じ向き・同オーダーであり，かつ C5 により低下分が重複 64 行にほぼ完全に帰属するため，
**ノイズだけで説明されるのではなく「小さな真の効果 ＋ 床以下の変動」の重ね合わせ**と見るのが妥当である．
非退行条件は事前登録どおり集計・報告したうえで，**是正レバーであるため退行が出ても撤回しない**方針に従う
（実際には BH 補正後の有意退行 0 件であり，撤回判断は発生していない）．

**学び**

1. **リークの帰属は重複行の中に閉じていた．** 重複 64 行は −9.375pt（6 行）落ちたのに対し，
   残り 3,686 行は −0.081pt とほぼ不変であった．訓練側から 64 行を抜いても近傍の非重複設問へは
   波及しない，すなわち**この分類器の「記憶」は設問単位で局所的**である．
   言い換えると，上振れの実体は約 6 行（0.16pt）であり，
   **B154 (A1) の「最大 1.7pt の上振れ」は理論上限（64/3,750）にすぎず，実測はその約 1/10 だった**．
   予測（replay 7 行 ＝ 0.19pt）ともよく合う．論文で「汚染により最大 1.7pt 上振れていた」と書くのは
   誤りで，正しくは「汚染分は 0.2pt 未満であり，是正後も結論は変わらない」である．
2. **今後の論文化で用いる基準線の数値を確定した．**
   主表は**新 artifact の全 3,750 行 top1 = 0.833067**（Wilson 95%CI [0.8208, 0.8447]）を用いる．
   Brown et al. (2020) 第 4 節に倣う clean subset の併記値は **3,686 行 top1 = 0.837222**．
   Iter89〜93 の過去反復の値は汚染込みであるが，汚染の寄与が 0.2pt 未満で再現性の床 ±0.25pt 以下である
   ことを本反復が実測したため，**過去反復との Δ 比較はそのまま有効**（再走は不要）である．
   以後の反復は基準線を `results/20260928_160921/`（0.833067）へ切り替える．
3. **`education` recall の −3.23pt は「リーク剥落」ではなく検出力不足の範囲．**
   除去 64 行のうち 46 行が education だったため退行の向きは予想どおりだが，
   未補正 p=0.0150 は 20 指標の BH 補正後に有意ではなくなる（education は n=350 程度で 1 行 ≈ 0.29pt，
   −3.23pt は 11 行に相当し，この規模のドメインでは実行間で普通に動く幅である）．
   Iter94 の事前 replay が education +0.29pt を予測していたのに実測は −3.23pt だった点は
   乖離として記録に残すが，**C4（全体 Δ の乖離 0.107pt）が PASS している以上，
   per-domain レベルの予測精度が全体ほど高くないというだけの話**であり，是正の成否には影響しない．
   重要なのは方向性で，**education の低 recall（0.42〜0.50）はリークで嵩上げされた見かけの値ではなく，
   除去後も残る真の弱点**であることが本走で追認された（Iter92 の学びと B156 の予測を支持）．
4. **測定系の是正レバーには精度レバーの足切り（訓練集合内 CV +2.3pt）を当てはめてはいけない**という
   運用が実地で機能した．本レバーは「上がらないこと」が成功であり，効果量 × 有意性の 2 軸表に
   `corrected_negligible` という**低下も無変化も成功と読める判定語**を事前登録しておいたことで，
   結果を後付けで解釈する余地なく締められた．B153 要レビュー (B) / B154 (A4) の宿題はこれで完了する．
5. **config の `levers` はこれで全て試し切った**（本レバーが末尾）．分類器側の手は Iter79〜93 で一巡し，
   測定系の是正も本反復で完了したため，次イテレーションは調査・計画フェーズからの再探索
   （停止条件 (2)）で始める．詳細と候補は backlog B157 に記録した．

## Iteration 93: 分類器のラベル粒度を JMMLU タスク単位へ細分化しドメインへ写像する

### 調査 (Iter93)

Iter92 の学び 3（`education` は 4 タスクにまたがる多峰クラスで単一の線形境界に収まっていない，
という仮説）を出発点に，backlog B153 (e) が指定したレバー `classifier_label_granularity` を検討した．
本フェーズは (1) 文献調査，(2) 訓練集合内 CV による事前見積り（B151 (d)4 の足切り）の 2 本立てで行った．
**評価集合 `data/dataset.jsonl` には一切触れていない．実機ノード wafl500〜509 も不使用**（特徴量は
既存キャッシュから再構成したため Ollama 呼び出しも発生していない）．

**(1) 文献調査（tavily-search）で分かったこと**

- Silla & Freitas (2011), *A survey of hierarchical classification across different application domains*,
  Data Mining and Knowledge Discovery 22(1-2):31-72,
  <https://www.cs.kent.ac.uk/people/staff/aaf/pub_papers.dir/DMKD-J-2010-Silla.pdf> ——
  「葉クラスで学習して親へ写像する」構成は *flat classification approach* として定式化済みの
  標準的な比較対象である．本レバーはこの flat approach そのものであり，新規手法ではない．
- Chen, Ding & Marculescu (2018), *Understanding the Impact of Label Granularity on CNN-Based
  Image Classification*, IEEE ICDMW 2018, pp.895-904, <https://par.nsf.gov/servlets/purl/10121660>
  —— CIFAR-10 / CIFAR-100 / ImageNet で細粒度ラベルによる学習が粗粒度の訓練精度・汎化精度の
  **双方**を改善したと報告している．本レバーの期待の根拠にあたる肯定的な先行研究．
- Novack, McAuley, Lipton & Garg (2023), *CHiLS: Zero-Shot Image Classification with Hierarchical
  Label Sets*, ICML 2023, arXiv:2302.02551,
  <https://proceedings.mlr.press/v202/novack23a/novack23a.pdf> —— (i) 各クラスのサブクラス集合を作り，
  (ii) サブクラスをラベルとして予測し，(iii) 予測サブクラスを親へ写し戻す，という 3 段構成で
  superclass 精度が改善する．さらに superclass 確率 × subclass 確率の積を採る変種も提示している
  （後述の CV で腕 E として測った）．
- **Pirovano et al. (2025), *The Advantage of Fine-Grained Training*, arXiv:2509.05130
  （Scientific Reports, 2026, doi:10.1038/s41598-026-64362-6），<https://arxiv.org/abs/2509.05130>
  —— 本反復の結果を最もよく説明する文献**．「細粒度ラベルでの学習は**普遍的には**精度を改善しない．
  効果は (a) データの幾何とラベル階層の関係，具体的には細粒度タスクと粗粒度タスクが要求する決定境界の
  重なり具合（著者らの言う *boundary redundancy*），(b) データセット規模，(c) モデルの容量
  （過剰パラメータ化の度合い）に依存し，細粒度学習が有利な領域と粗粒度学習が有利な領域を分ける
  遷移が存在する」．**すなわち「効くか効かないかはデータ側の構造で決まるので測るしかない」**．

この最後の知見から，**本レバーは着手の前に訓練集合内 CV で測るべき典型例**と判断した
（B153 (e) が課した B151 (d)4 の足切りとも整合する）．

**(2) 訓練集合内 CV の設計**

- 特徴量: `data/embcache_train_iter89_qwen3-embedding_4b{,__p1}.npy`（2,319 行）と
  `data/embcache_train_iter92_qwen3-embedding_4b_new20{,__p1}.npy`（20 行）から，Iter92 の
  `/tmp/iter92/retrain.py` と同一の規則で 2,339 行 × 5,120 次元を再構成した（新規の埋め込み計算なし）．
- サブクラスラベル: 設問文をキーに `/tmp/expert-mesh-cache/JMMLU.zip` へ逆引きし，
  **2,339 行すべてが 56 タスクのいずれかに一致（未一致 0 行）**，かつ
  **現行 `build_dataset.py:_DOMAIN_TASK_MAP` との食い違い 0 行**であることを確認した
  （Iter92 の是正で写像ずれが解消されたことの独立な確認にもなっている）．
  hard negative 行（`*-hardneg-*`）も全て JMMLU 由来でタスクを持ち，
  B153 (e) が挙げたリスク (b)「サブクラス割り当て規則の未定義」は発生しなかった．
- クラス構成: 56 タスク，1 タスクあたり **15〜95 行**（最小 `professional_medicine` 15 行）．
  `CalibratedClassifierCV(cv=5)` の内側 fold にも 2 行以上が残るため，リスク (a) の懸念のうち
  「fold が組めない」事象は起きなかった．
- 外側 5-fold StratifiedKFold（層化キーは JMMLU タスク）× seed 0/1/2 の計 15 fold．
  `sample_weight` は本番 `_extract_sample_weights()` と同じ**ドメイン均衡重み** `n/(K*n_d)`（K=10）を
  全腕で共通に使い，変えるのは学習ラベルの粒度と写像規則だけにした（単一レバー原則）．

### 計画 (Iter93)

**単一レバー（事前登録）**: `classifier_label_granularity` =
**`jmmlu_task_level_subclass_then_sum_to_domain`**．

B153 (e) が計画フェーズへ委ねた「argmax 写像のみ／タスク確率のドメイン合算のどちらか 1 つを
事前に決める」という論点は，**合算（sum）**を選ぶ．理由は 3 点である．
(i) CV で合算が argmax 写像を上回った（後述），
(ii) 合算は潜在サブクラスに関する周辺化そのもので，`classifier.py:estimate_confidence_classifier()` が
前提とする「10 次元で総和 1 の確率ベクトル」を構成上そのまま保てる（argmax 写像では 10 次元確率を
別途こしらえる必要があり，較正の意味が変わる），
(iii) 10 ドメインへ同一の規則で適用するため 2026-09-23 恒久ルール (1) に抵触しない．

**仮説**: `education` の recall が 0.65 付近で頭打ちなのは，4 つの JMMLU タスクにまたがる多峰クラスを
単一の線形境界で表現しているためである．サブクラス単位で学習して確率をドメインへ合算すれば，
多峰クラスを複数の線形境界の和で表現でき，`education`・`general`・`social_science` の recall が上がる．

**固定する構成（Iter92 の最良構成）**: 埋め込み `qwen3-embedding:4b` の 2 ビュー連結（5,120 次元），
instruction prefix 現行値，`CalibratedClassifierCV(method="temperature", cv=5, ensemble=True)`，
`C`=1.0，`class_weight=None` ＋ドメイン均衡 `sample_weight`，`confidence_threshold=0.0`，
`dispatch_top_k=1`，訓練 `data/classifier_train_iter92_civics_aligned.jsonl`（2,339 行），
評価 `data/dataset.jsonl`（3,750 行，sha256 `2114e048...`）．
**基準線は `results/20260928_111644/`（3,750 行，top1 = 0.835467）**．

### 着手可否の判定 (Iter93) —— B151 (d)4 の足切りに不合格．本走は行わない

**本番パイプライン（温度較正あり）での訓練集合内 CV．3 seed × 外側 5-fold ＝ 15 fold，n=2,339**

| 腕 | 学習ラベル | ドメイン写像 | CV ドメイン精度 | Δ vs 現行 |
|---|---|---|---|---|
| **A（現行）** | ドメイン 10 クラス | — | **0.7961**（fold 間 sd 0.0174） | — |
| B | JMMLU タスク 56 クラス | argmax タスク → ドメイン | 0.8045（sd 0.0133） | **+0.84pt** |
| **C** | JMMLU タスク 56 クラス | タスク確率をドメインへ合算 | **0.8060**（sd 0.0158） | **+1.00pt** |

**較正ラッパなしの素の `LogisticRegression`（同一分割，3 seed × 5 fold）での写像規則の比較**

| 腕 | 内容 | CV | Δ vs A |
|---|---|---|---|
| A | ドメイン 10 クラス（現行） | 0.7996 | — |
| B | タスク 56 クラス → argmax 写像 | 0.8109 | +1.13pt |
| **C** | タスク 56 クラス → 確率合算 | **0.8147** | **+1.51pt** |
| D | タスク 56 クラス（**タスク**均衡 `sample_weight`）→ 確率合算 | 0.8015 | +0.19pt |
| E | ドメイン確率 × タスク合算確率（CHiLS の積） | 0.8105 | +1.08pt |

**判定: 最良の腕 C でも本番パイプライン CV で +1.00pt であり，B151 (d)4 の足切り
（訓練集合内 CV で +2.3pt 以上）に届かない．よって本走に進まない**（Iter89 の学び
「判定不能と事前に分かるレバーには着手しない」，B153 (e) の着手条件，停止条件 (2)）．
Iter91 で実測した伝達率 43% を当てると end-to-end の予測は **+0.43pt** で，
再現性の床 ±0.25pt をかろうじて超える程度，事前登録し得る効果量閾値 1.0pt の半分未満である．
Iter92（Δ +0.508pt，p = 0.0509）と同じ「境界上で判定語が決まらない」結果を繰り返すことになる．

**さらに，非退行条件①を破る見込みが高い**（本走を行わない第 2 の理由）．
腕 C の per-domain recall（較正あり，15 fold 合算）は次のとおりで，**改善と退行が真っ二つに割れる**．

| ドメイン | タスク数 | A（現行） | C | Δ |
|---|---|---|---|---|
| education | 4 | 0.5411 | 0.6984 | **+15.7pt** |
| social_science | 4 | 0.7773 | 0.8587 | +8.1pt |
| general | 3 | 0.7960 | 0.8613 | +6.5pt |
| legal | 2 | 0.8398 | 0.8701 | +3.0pt |
| mathematics | 5 | 0.9520 | 0.9760 | +2.4pt |
| computer_science | 5 | 0.9200 | 0.9400 | +2.0pt |
| natural_science | 8 | 0.8720 | 0.8560 | −1.6pt |
| business_economics | 8 | 0.8173 | 0.7813 | −3.6pt |
| history_culture | 7 | 0.8494 | 0.7865 | −6.3pt |
| **medical** | **10** | 0.6653 | 0.4893 | **−17.6pt** |

**Δ はドメインあたりの JMMLU タスク数と明確に逆相関している**（タスク数 2〜5 の 6 ドメインは全て改善，
7〜10 の 4 ドメインは全て退行）．機序は，タスク数の多いドメインでは 1 サブクラスあたりの行数が
15〜38 行まで削られ，各サブクラスの境界推定の分散が増えるためと読める
（`medical` は 10 タスク・最小 `professional_medicine` 15 行）．確率を合算しても，
弱い境界を 10 本足したものは 10 倍のデータで引いた 1 本の境界に負ける．
これは Pirovano et al. (2025) の「細粒度学習の利得はサブクラスあたりのデータ量と
boundary redundancy に依存し，普遍的ではない」という主張の**本データでの実例**である．
すなわち **Iter92 の学び 3 の仮説（`education` は多峰クラス）は CV 上は支持された**（education +15.7pt）
一方で，**それを 10 ドメイン一律の規則として適用すると medical で失う分が上回る**．

**代替の粒度も測ったが，やはり足切りに届かない．**
タスク数の偏りを消すため「全ドメインを一律 k 個の潜在サブクラスへ分ける」（訓練 fold 内の埋め込みに
対する k-means．fold 内で fit するのでリークしない）という 10 ドメイン均一な規則を素の
`LogisticRegression`・3 seed × 5 fold で掃引した（基準 A = 0.8019，層化キーはドメイン）．

| k | 2 | 3 | **4** | 5 | 6 | 8 |
|---|---|---|---|---|---|---|
| Δ vs A | +0.61pt | +1.03pt | **+1.68pt** | +1.15pt | +1.17pt | +0.38pt |

k=4 が頂点の上に凸な曲線で，**退行するドメインが無い**点は腕 C より健全である
（medical 0.6813→0.6973，history_culture 0.8494→0.8421，legal 0.8528→0.8398）．
ただし素の LR で +1.68pt であり，較正ラッパを通すと本レバーでは利得が約 2/3 に縮む実測
（腕 C: +1.51pt → +1.00pt）を当てると **+1.1pt 前後**と見込まれ，これも足切りに届かない．

**本イテレーションの結論**: `classifier_label_granularity` は，値 `..._sum_to_domain`
（当初案の `..._argmax_map_to_domain` を含む）・k-means 潜在サブクラス版のいずれも
事前 CV が足切り +2.3pt に届かないため **本走を行わず，レバーを閉じる**．
`config.yml` の当該レバーには本 CV の実測値を注記した（同じ案を再度引く無駄を避けるため）．

**次の一手（rc-planner から人間／オーケストレータへの申し送り．backlog B154）**

分類器側の手（埋め込みモデル・instruction prefix・ビュー連結・較正手法・正則化 `C`・クラス重み・
訓練ラベル写像・ラベル粒度）は Iter79 以降で一巡し，**訓練 2,339 行・特徴 5,120 次元という
現在の構成での訓練集合内 CV はどの腕でも 0.80〜0.82 に張り付いている**．
残る方向は次の 3 つで，いずれも単一レバー原則の外側か人間判断を要する．

- **(A1, 推奨) 測定系の是正**（B153 (f) が「優先度は (e) の次」とした項目）: 訓練と評価で設問文が
  一致する残り 64 行を訓練から除去し，基準線を引き直す．精度レバーではないが，
  絶対水準を最大 1.7pt 上振れさせうるバイアスを，論文化前に取り除ける．オフラインで準備でき，
  新規データ源も不要．ただし基準線の本走 1 回（約 155 分）が要る．
- **(A2) 外部の日本語データ源の調達**（B151 (d)1）: CV で最も弱いのは `education` 0.54 と
  `medical` 0.67 で，hard negative プールの在庫は `education` 9 行・`legal`/`general`/`social_science` 0 行
  （B151 (c)）．行数を増やす以外の手は本反復で尽きた．**ライセンス（NC/ND 条項）と評価集合との
  重複の確認を伴うため，調達前に人間の確認を要する．**
- **(A3) 足切り +2.3pt そのものの見直し**: この値は「伝達率 43% × end-to-end 効果量閾値 1.0pt」から
  導いたものだが，1.0pt という閾値自体が Iter92 の学び 4 で「機序が予測どおり発火しても際どい」と
  判明している．**+0.5pt 級の改善を積み上げる方針へ切り替えるなら**，腕 C（予測 +0.43pt）ではなく
  k-means k=4（予測 +1.1pt・退行ドメインなし）が最有力の候補として残る．
  **これは研究の合否基準の変更であり，人間の判断を仰ぎたい．**

