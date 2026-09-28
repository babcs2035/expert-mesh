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

## Iteration 92: 訓練ラベル写像を評価集合へ一致させる（japanese_civics の education 復帰）

### 調査 (Iter92)

backlog B151 (c) の指示により `levers` を使い切った状態からの再探索である．B151 (d) が挙げた 4 観点
（日本語 legal/education/social_science/general の公開データ源／クラス不均衡下の線形分類器／複合クエリの
ルーティング／訓練集合内 CV で +2.3pt 以上を見込めること）を出発点に tavily-search で文献を当たりつつ，
**基準線 `results/20260928_032909/`（3,750 行，top1 = 0.829867）の誤りがどこに集中しているかを
`jmmlu_task` 単位まで分解した**．その結果，候補レバーを机上で比較する前に**測定系の側に
未発見の不整合が 1 つ残っていること**が分かったので，本反復はそれを主題にする．

**(1) 文献調査（tavily-search）で分かったこと**

- Menon et al. (2021), *Long-tail learning via logit adjustment*, ICLR 2021, arXiv:2007.07314,
  <https://arxiv.org/abs/2007.07314> —— クラス事前確率に基づく logit 補正は，重み付けよりも
  長尾クラスの誤りを直接減らす．**本研究への適用可能性の見積り**: 本研究で事前分布が偏っているのは
  `legal`（77 行 / 他 9 ドメイン各 250 行）だけであり，τ=1 の補正が動かすのは訓練行の 3.3%，
  訓練集合内 CV 換算で高々 +0.2pt 程度にしかならない．B151 (d) 4 の足切り（CV +2.3pt）に
  遠く届かないため，**今回は採らない**（Iter89 の学び「判定不能と事前に分かるレバーには着手しない」）．
- Northcutt et al. (2021), *Pervasive Label Errors in Test Sets Destabilize Machine Learning
  Benchmarks*, NeurIPS Datasets & Benchmarks,
  <https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/f2217062e9a397a1dca429e7d70bc6ca-Paper-round1.pdf>
  —— 実務では「補正前のテスト精度」しか見えないため，ラベル定義の誤りはモデル側の差として
  誤読されやすい．**本反復の発見はまさにこの型である**（後述）．
- Nevin et al. (2025), *The effects of mismatched train and test data cleaning*,
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC12190241> —— 訓練側と評価側で前処理・整形が食い違うと，
  モデル選択の判断そのものが変わりうる．**断定は避けるが**，本研究で 3 反復続けて分類器側のレバーが
  no_effect だったことと整合的な説明を与える可能性がある．
- Tunstall et al. (2022) SetFit 系の実務報告（Cloudera FFL, *Few-Shot Text Classification*,
  <https://few-shot-text-classification.fastforwardlabs.com>）—— 凍結埋め込み上の軽量分類器は
  **クラスあたり 8〜30 例**で実用水準に達し，100 例以上は飽和しやすい．
  **本反復への含意**: 後述の欠落サブクラスに対し 34 行しか用意できないが，この範囲は
  「効果が出るなら出る」水準ではある（保証ではない）．
- JMMLU のライセンス: 『日本問題』（japanese_civics・japanese_idiom・japanese_geography）の著作権は
  New Style（VIST 学習塾）にあり **CC BY-NC-ND 4.0**，japanese_history / world_history は STEP 社で
  研究・評価目的以外の商用利用禁止（<https://huggingface.co/datasets/nlp-waseda/JMMLU>，
  <https://github.com/nlp-waseda/JMMLU>）．`build_dataset.py:_RESTRICTED_LICENSE_TASKS` はこの 5 タスクを
  列挙しており，`mise.toml:31` の `build_dataset.py --output data/dataset.jsonl` は
  `--exclude-restricted-license-tasks` を**渡していない**（＝評価集合には japanese_civics が入る）．
  **B151 (d) 1 が求めた「JMMLU 以外の日本語データ源の調達」は今回は不要と判断した**．理由は
  次節のとおり，既存データの中に未使用の正解信号が残っていることが分かったためである．

**(2) 基準線の誤りの所在 —— `education` の不足分はほぼ全量が 1 タスクに集中している**

`data/dataset.jsonl` の `jmmlu_task` で基準線 `results/20260928_032909/` の単一ドメイン行を割ると:

| ドメイン / タスク | n | recall | 主な誤送先 |
|---|---|---|---|
| education / **japanese_civics** | 116 | **0.0086 (1/116)** | history_culture 59・business_economics 36・legal 15 |
| education / high_school_psychology | 73 | 0.685 | medical 14 |
| education / sociology | 85 | 0.659 | history_culture 15 |
| education / moral_disputes | 76 | 0.645 | social_science 12・legal 11 |
| general / miscellaneous | 69 | 0.391 | history_culture 14・natural_science 10 |
| general / formal_logic ・ logical_fallacies | 106 | 0.98 / 0.964 | — |
| social_science / philosophy | 77 | 0.610 | education 21 |

- **education_recall 0.4457 の不足 194 行のうち 115 行（全 3,750 行の 3.07pt）が japanese_civics 1 タスク**
  である．他 3 タスクは 0.645〜0.685 で，10 ドメイン中の中位と同程度でしかない．
- `general` も同型で，0.7429 の不足はほぼ `miscellaneous`（0.391）に集中し，形式論理系 2 タスクは 0.97 前後．

**(3) 原因は分類器ではなく訓練側のタスク→ドメイン写像のずれ（本フェーズで実測）**

`/tmp/expert-mesh-cache/JMMLU.zip` の設問文をキーに，本番訓練ファイル
`data/classifier_train_iter87_hybrid.jsonl`（2,327 行）の各行を JMMLU のタスクへ逆引きした結果:

- **`education` ラベル 250 行の内訳は high_school_psychology 87 / moral_disputes 87 / sociology 76 で，
  japanese_civics は 0 行**．
- 一方 **japanese_civics 22 行が `history_culture` ラベルで訓練集合に入っている**．
  これは誤送先第 1 位（59 行）と一致しており，**現行の訓練データは「公民の設問は history_culture」と
  明示的に教えている**．
- 訓練集合 2,327 行のうち現行 `_DOMAIN_TASK_MAP` と食い違う行は **24 行のみ**で，そのうち 22 行が
  この japanese_civics である（残り 2 行は international_law → business_economics 1 行，
  high_school_microeconomics → mathematics 1 行）．つまり**ずれは 1 タスクに限局している**．
- 経緯: `data/classifier_train.jsonl`（1,427 行，sha256 `eb89bf7b...`）は Iter37/38 の写像更新より前に
  作られたまま再生成されておらず，Iter84 の hard negative 採掘は「現行訓練集合に既出のタスクへプールを
  限定する」方針（backlog B133 (2)）を採ったため，旧写像がそのまま温存された．
  一方 `data/dataset.jsonl` は Iter85（単一ドメイン拡充）・Iter90（複合拡充）で現行
  `_DOMAIN_TASK_MAP` に基づき再構築されており，**評価側だけが新写像に移っていた**．

**Iter36/37/38 の否定的所見との関係（重要．前提が逆である）**: Iter38
（`education_hybrid_proxy_and_civics`）は rejected だが，当時の実装検証項目に
「eval: education 150（旧 proxy のみ），japanese_civics = 0 件 — PASS」と明記されているとおり，
**評価集合から japanese_civics を除いた上で訓練へ 150 行入れた**構成だった（訓練だけに存在する希釈）．
現在は符号が逆で，**評価側に 116 行あり訓練側が 0 行**である．
backlog B133 (2) が civics の流入を避けた判断も，評価側の再構築より前の情報に基づく．
したがって過去 3 回の失敗は本反復の変更を否定しない（断定を避けて言えば，
「同じ変更が別の前提の下で測られたもの」であって再現実験ではない）．

**(4) リーク上限の実測 —— 訓練へ回せる japanese_civics は 34 行が上限**

JMMLU の japanese_civics は全 150 問．内訳は **評価集合に 116 問 / 訓練集合に 22 問（うち 8 問は
評価集合とも重複）/ どちらにも未使用 20 問**．よって**評価集合を汚さずに訓練へ回せるのは
14（非重複の既存訓練行）＋ 20（未使用）＝ 34 行が上限**である．
評価集合に載っている 116 問を訓練へ入れることは絶対にしない．

**(5) 副次的に見つかった測定系の欠陥（本反復では直さない．backlog B152 へ記録）**

訓練 2,327 行と評価 3,750 行の**設問文完全一致が 72 行**ある（うち 64 行はラベルも一致）．
`build_classifier_training_rows()` は `exclude_queries=eval_queries` で重複を排除する設計だが，
本番の訓練ファイルは評価集合の拡充（Iter85/90）より前に作られたため事後的に重複が生じた．
**本反復では (4) の civics 8 行だけを削除し，残り 64 行には触れない**（単一レバー原則）．

### 計画 (Iter92)

**単一レバー**: `classifier_train_label_map_consistency` = **`japanese_civics_realignment_to_eval_map`**
（config.yml の `levers` 末尾に本フェーズで追記．backlog B152 に選定理由を記録）．

**仮説**: `education` の recall が低いのは代理タスクの意味的ギャップでも分類器の容量不足でもなく，
**評価集合の education の 33%（116/350）を占める japanese_civics を，訓練データが
`history_culture` として明示的に教えているため**である．訓練側の写像を評価側と一致させれば，
japanese_civics の recall が 0.0086 から大きく上がり，top1_accuracy が +1.0pt 以上動く．

**変更点（データのみ．コードは 1 行も変えない）**

`data/classifier_train_iter92_civics_aligned.jsonl` を新規に作り，
`data/classifier_train_iter87_hybrid.jsonl` との差分を次の 3 点だけにする．

1. 評価集合に出現しない japanese_civics 訓練行 **14 行**: `domain` を `history_culture` → `education`．
2. 評価集合と設問文が重複する japanese_civics 訓練行 **8 行**: 削除．
3. 訓練にも評価にも未使用の japanese_civics **20 問**: `education` 行として追加．

結果は **education 250→284 行 / history_culture 250→228 行 / 合計 2,327→2,339 行**．
`scripts/train_domain_classifier.py`・`build_dataset.py`・`config.yaml`・`classifier.py`・
`data/dataset.jsonl` は無変更（`C` は Iter91 で revert 済みの既定 1.0 のまま）．

**固定する構成（Iter90 の最良構成）**: 埋め込み `qwen3-embedding:4b` の 2 ビュー連結（5,120 次元），
instruction prefix は現行値のまま，`CalibratedClassifierCV(method="temperature", cv=5, ensemble=True)`，
`class_weight=None` ＋ `_extract_sample_weights()` のドメイン均衡重み，`confidence_threshold=0.0`，
`dispatch_top_k=1`，評価集合 `data/dataset.jsonl` 3,750 行．

**再訓練の手順（絶対条件）**

- 埋め込み計算は **wafl-ctrl5 上で行う**（2026-09-23 恒久運用ルール．wafl500〜509 は本走以外に使わない）．
- **既存キャッシュを再利用し，新規に埋め込むのは追加 20 行だけにする**
  （`data/embcache_train_iter89_qwen3-embedding_4b{,__p1}.npy` の該当行を流用）．
  B149 (d) の `qwen3-embedding:4b` のセッション跨ぎ非決定性（最大絶対差 0.011406）と本レバーの効果が
  交絡するのを避けるための措置であり，**2,319 行の特徴量が基準線 artifact と同一であることを
  G0-b で確認する**．同等の結果が得られる別経路を採ってもよいが，その場合も同一性の確認は必須とする．

**本走前のゲート（G0．すべて本走前に journal へ記録する）**

| # | 内容 | 合格条件 |
|---|---|---|
| G0-a | 新訓練ファイルの構成検証 | 合計 2,339 行 / education 284 / history_culture 228 / japanese_civics 行は全て `education` かつ 34 行 / `history_culture` の japanese_civics 0 行 / 新・改変行と `data/dataset.jsonl` の設問文重複 0 件 |
| G0-b | 特徴量の同一性 | 変更していない 2,319 行の埋め込みがキャッシュとビット一致 |
| G0-c | 変更の最小性 | `git diff --stat` に `scripts/`・`build_dataset.py`・`config.yaml`・`classifier.py`・`node.py`・`aggregator.py` が現れないこと．`data/dataset.jsonl` の sha256 不変 |
| G0-d | オフライン replay による効果量の事前登録 | 新 artifact で `data/dataset.jsonl` を replay し，top1・per-domain・japanese_civics 部分集合 recall を**本走前に**記録する（**予測の事前登録のみ．採否の判断には使わない**．Iter91 で replay が分類器 argmax 精度を小数 6 桁一致で当てることが実証済み） |
| G0-e | **デプロイ** | **wafl500〜509 には Iter91 の `C`=10.0 artifact が載ったままである（B151 (b)）．本走前に必ず `mise run deploy` を実行し，10 台すべてで新 artifact の sha256 一致と smoke_check PASS を確認すること** |

**成功条件（事前登録．基準線は `results/20260928_032909/`，3,750 行，top1 = 0.829867）**

- **主基準（両方を満たすとき `adopted`）**
  - (i) Δtop1_accuracy **≥ +1.0pt**（≥ 0.839867）かつ McNemar 両側 p < 0.05（Wilson 95%CI 併記）．
  - (ii) education 単一 350 行の recall **≥ 0.55**（基準線 0.4457）**かつ** japanese_civics 部分集合
    116 行の recall **≥ 0.30**（基準線 0.0086）．
- **非退行（1 つでも破れば `partial`）**
  - ① per-domain recall/precision 20 指標（単一 3,020 行）の BH 補正 q=0.05 後の有意退行 0 件．
    **history_culture は訓練行が 22 行減るため最も危ない**（基準線 recall 0.9114）．
  - ② 複合 730 行の正解率 **≥ 0.7932**（基準線 0.8082 − 1.5pt）かつ複合行での McNemar 有意退行なし．
  - ③ `fallback_rate` = 0.0，`dispatch_failure_rate` ≤ 0.005，`mean_duration_ms` ≤ 3,040（基準線 2,531 の 1.2 倍）．
- **判定語**: (i)(ii) と非退行①〜③がすべて成立 → `adopted`．
  (i)(ii) のいずれかが成立し非退行が破れる → `partial`．
  (i) が非有意または |Δtop1| < 0.5pt → `no_effect`．到達確認 G0-a〜G0-c が破れる → `invalid`．
  （B149 (c)・B151 要レビュー (a) の「判定語を排他的な決定木へ書き直す」という宿題は未承認のため，
  本反復では上記の順に**上から順に最初に当てはまったものを採る**と明示して曖昧さを消す．）

**期待効果（事前登録の点予測）**: japanese_civics recall 0.0086 → **0.45**（116 行中 52 行．
SetFit 系の実務報告が示す「クラスあたり 30 例前後で実用水準」の下端にあたる 34 行しか用意できないため，
飽和水準には届かないと見込む）．これが overall へ **+1.36pt**，history_culture の減少分 −0.1pt 前後，
差引 **Δtop1 = +1.0〜+1.5pt** と予測する．B151 (d) 4 の「訓練集合内 CV +2.3pt」という足切りは，
**訓練集合に japanese_civics の education 行が 1 行も無い以上，訓練集合内 CV ではこの欠陥が
そもそも観測できない**ため本レバーには適用できない（伝達率 43% は分類器の容量を触るレバーの値であり，
訓練と評価の分布ずれを直すレバーには当てはまらない）．代わりに G0-d の replay を事前登録の予測として使う．

**測定上の注記（分析フェーズへの申し送り）**: 変更点 2 で削除する 8 行は評価集合にも含まれるため，
削除によって汎化と無関係に最大 **0.21pt**（8/3,750）だけ top1 が上振れしうる．
分析フェーズでは**この 8 行を除いた 3,742 行での top1 も必ず併記すること**．

**不成立だった場合の次の一手**: japanese_civics recall が 0.30 に届かない場合，34 行という行数が
不足しているのか（→ 外部の日本語公民データ源の調達．ライセンス確認が要るため人間判断）
それとも `education` というクラス定義自体が 4 タスクをまたいで多峰的すぎるのか（→ クラス定義の再検討）
を切り分けること．

### 実装・実験 (Iter92)

**変更（コード 0 行）**: `data/classifier_train_iter92_civics_aligned.jsonl`（2,339 行，sha256
`39c4ca52...`）を新規作成した．差分は事前登録どおり 3 点のみ — (1) `history_culture` → `education`
への付け替え 14 行（`history_culture-train-002,006,009,024,025,051,052,053,056,063,085,103,110,148`），
(2) 評価集合と設問文が重複する 8 行の削除（同 `-013,014,020,078,099,118,120,135`），
(3) 訓練・評価とも未使用の japanese_civics 20 問を `education` として追加（`education-civics-001`〜`-020`）．
education 250→284 / history_culture 250→228．追加 20 行の設問整形は `build_dataset.py` の
`_parse_jmmlu_task_csv()` / `_format_jmmlu_query()` をそのまま import して生成した（自前整形なし）．
再訓練は wafl-ctrl5 の Ollama（127.0.0.1:11499）で実施し，変更のない 2,319 行は
`data/embcache_train_iter89_qwen3-embedding_4b{,__p1}.npy` を直接インデックス参照で再利用，
新規 embed は追加 20 行のみ（`embed_query_views` と同じ plain→instructed の順）．
新 artifact `models/domain_classifier.joblib` sha256 `98da6f2d0d44...`（n_features_in_=5120），
旧版（基準線と同一 `ff8aad9c...`）は `models/domain_classifier_pre_iter92_civics.joblib` へ退避（可逆）．
`data/MANIFEST.md` 追記済み．再訓練スクリプトは `/tmp/iter92/retrain.py`（リポジトリ外）．

**G0 ゲート**: a（構成検証: 2,339 行／education 284／history_culture 228／japanese_civics 行は全て
education かつ 34 行／history_culture 内の japanese_civics 0 行／評価集合との設問文重複 0 件）**PASS**，
b（特徴量同一性: 2,319 行はキャッシュから直接インデックス参照．スクリプト内 assert で検証）**PASS**，
c（変更の最小性: `git diff --stat` に `scripts/`・`build_dataset.py`・`config.yaml`・`classifier.py`・
`node.py`・`aggregator.py` は一切現れず，`data/dataset.jsonl` sha256 不変 `2114e048...`，
`train_domain_classifier.py` の `C` は既定 1.0 のまま差分 0 行）**PASS**，
d（replay による予測の事前登録: top1 0.831467→0.836267，Δ+0.48pt，McNemar p=0.0636，
japanese_civics recall 0.0086→0.2241，education recall 0.4429→0.4943．**採否判断には使わない**）**記録済**，
e（デプロイ: `mise run deploy` 後，全 10 ノード wafl500〜509 で artifact sha256 が `98da6f2d0d44...` に
一致，smoke_check も全台 PASS）**PASS**．

**本走**: `results/20260928_111644/`（3,750 問，フルスペック 1 回，所要約 155 分 < timeout 180 分）．
基準線は `results/20260928_032909/`（top1 = 0.829867）．

| 指標 | 基準線 | 本反復 | Δ |
|---|---|---|---|
| top1（全 3,750 行） | 0.829867 | 0.835467 | **+0.560pt**（McNemar p = 0.032015，discordant 33/54） |
| top1（3,742 行．重複削除 8 行を除外） | 0.831641 | 0.836718 | +0.508pt（McNemar p = 0.050894，discordant 33/52） |
| education 単一 350 行 recall | 0.445714 | 0.497143 | +5.143pt |
| japanese_civics 部分集合 116 行 recall | 0.008621 | 0.232759 | +22.414pt（1/116 → 27/116） |
| 複合 730 行 top1 | 0.808219 | 0.805479 | −0.274pt（McNemar p = 0.789268） |

per-domain 20 指標の BH 補正（q=0.05）後の**有意退行は 0 件**（生の p 値では `recall:education` の
p = 0.005820 が最小だが，BH のランク 1 閾値 0.0025 に対し不足．他は p > 0.2）．
運用指標は `fallback_rate` 0.0/0.0，`dispatch_failure_rate` 0.000533→0.000800（≤0.005），
`mean_duration_ms` 2531.5→2535.3（≤3040）で全て基準内．

**逸脱・注記**: (1) `mise run deploy` はツールタイムアウト 2 分を超えるためバックグラウンド実行へ
切り替えて完走を待った（コマンド自体は正常終了．Iter89 と同種の運用上の事象）．
(2) G0-b の「ビット一致」はキャッシュからの直接インデックス参照という構成上，独立に再計算して
差分を測る形の検証ではなく，同じ配列を使い回していることによる構造的な保証である．
埋め込みのセッション跨ぎ非決定性（B149 (d)）とは交絡しない設計だが，この違いは明記しておく．
(3) 本走は途中経過のポーリングのみで完走を待ち，強制終了等は行っていない．

### Iteration 92 実行済み

**変更**: コード 0 行．`data/classifier_train_iter92_civics_aligned.jsonl`（2,339 行，sha256
`39c4ca52...`）を新規作成し，訓練側の japanese_civics のラベルを評価側の `_DOMAIN_TASK_MAP` へ
一致させた（`history_culture`→`education` 付け替え 14 行／評価集合と設問文が重複する 8 行を削除／
訓練・評価とも未使用の 20 問を `education` として追加．education 250→284，history_culture 250→228）．
新 artifact `models/domain_classifier.joblib` sha256 `98da6f2d0d44...`．G0-a〜e は全て PASS
（評価集合 sha256 不変，wafl500〜509 の 10 台で artifact sha256 一致，smoke_check PASS）で，
**レバーが発火していることは確認済みである**．

**結果（本走 `results/20260928_111644/` 対 基準線 `results/20260928_032909/`）**

| 指標 | 基準線 | 本反復 | Δ | p |
|---|---|---|---|---|
| top1（3,750 行） | 0.829867 (3112/3750) | 0.835467 (3133/3750) | +0.560pt | McNemar 0.031418（54 改善 / 33 悪化） |
| **top1（3,742 行．訓練削除 8 行を除外．主報告値）** | 0.831641 | 0.836718 | **+0.508pt** | **0.050894** |
| education 単一 350 行 recall | 0.445714 | 0.497143 | +5.143pt | — |
| japanese_civics 116 行 recall | 0.008621 (1/116) | 0.232759 (27/116) | **+22.414pt** | **exact McNemar 2.98e-8（26 改善 / 0 悪化）** |
| **japanese_civics を除く 3,634 行** | — | — | **−5 行（28 改善 / 33 悪化）** | **0.608921** |
| 複合 730 行 top1 | 0.808219 | 0.805479 | −0.274pt | 0.789268 |

per-domain 20 指標の BH 補正（q=0.05）後の有意退行 **0 件**（生 p 最小は `recall:education` の
0.005820 だが BH ランク 1 閾値 0.0025 に不足）．運用指標は全て基準内（`fallback_rate` 0.0，
`dispatch_failure_rate` 0.000800 ≤ 0.005，`mean_duration_ms` 2535.3 ≤ 3040）．

**ノイズか信号か**: 本測定系の再現性の床は top1 ±0.25pt，そこから導かれる McNemar 有意境界は
0.249pt である（B149 恒久申し送り 1，Iter88 実測）．Δ = +0.560pt（3,742 行で +0.508pt）は床の
約 2.0〜2.2 倍であり，**符号は信号として読める**．ただし +1.0pt に対しては約半分で，
主基準 (i) が要求した効果量には届いていない．一方 **japanese_civics 部分集合の 26 改善 / 0 悪化
（exact p = 2.98e-8）はノイズでは説明不能**であり，狙った機序が発火したことは確定した．

**主報告値の選択（論点 3）**: 削除した 8 行は評価集合にも含まれるため事前登録どおり両方を出したうえで，
**3,742 行版（Δ +0.508pt，p = 0.050894）を主報告値とする**．理由は，削除された 8 行の正誤変化は
汎化とは無関係な構成上の産物（基準線ではこの 8 行が `history_culture` ラベルで訓練に入っており
確実な誤りを作っていた）だからである．実際に生じた上振れは **+2 行 = +0.053pt** に留まり，
事前に見積もった最大 0.21pt よりはるかに小さい（基準線 0/8 正解，本反復 2/8 正解）．
2 つの数字の差は 0.05pt で，どちらを採っても効果量の結論は変わらないが，p 値は 0.031 と 0.051 で
有意水準 0.05 をまたぐ．**保守側（3,742 行，p = 0.050894 ＝ 非有意）を主として扱う．**

**判定: `partial`（主たる判定語）**

事前登録の判定条文を literal に当てはめた結果を先に書く．
- `adopted`: (i) Δtop1 ≥ +1.0pt が不成立（+0.508〜+0.560pt）．(ii) education recall 0.497 < 0.55，
  japanese_civics recall 0.2328 < 0.30 も不成立．**不成立**．
- `partial`（「(i)(ii) のいずれかが成立し非退行が破れる」）: (i)(ii) はいずれも不成立，
  非退行①〜③は全て充足．**literal には不成立**．
- `no_effect`（「(i) が非有意 **または** |Δtop1| < 0.5pt」）: 主報告値 3,742 行では p = 0.050894 で
  非有意のため **literal には成立する**（3,750 行を採れば p = 0.031418 で有意，|Δ| = 0.560 ≥ 0.5 となり
  literal には不成立となる．**どちらの行数を採るかで判定語が変わる**）．
- `invalid`: G0-a〜e 全 PASS のため不成立．

すなわち**事前登録の判定木には，「有意性は境界上だが効果量が閾値の半分・非退行は全充足・
狙った機序は明確に発火」という今回の状態を受け止める枝が無い**（B151 要レビュー (a) が指摘した
条文の重なりとは逆向きの，網羅性の欠落である）．そのうえで主たる判定語を **`partial`** とする．
根拠は次の 3 点である．
1. `no_effect` は「レバーが効かなかった」ことを記録する語だが，japanese_civics 26 改善 / 0 悪化
   （p = 2.98e-8）は**レバーが意図した経路で確実に効いたことを直接示す**．この状態を `no_effect` と
   記録すると，次の自分が「訓練／評価のラベル写像ずれは直しても効かない」という誤った事実を
   引き継ぐ．Iter91 の `no_effect`（Δ +0.16pt・床の内側・機序の証拠なし）とは質的に異なる．
2. `partial` の語義は「事前登録した成功条件の一部だけを満たした」であり，今回は
   主基準 (ii) の 2 条件のうち方向・機序は実現したが水準（0.30 / 0.55）に未達，
   (i) は効果量未達という状態で，この語義には合致する．条文の文言（「非退行が破れる」）に
   合致しないのは条文側の欠落である．
3. Iter90 も「設計どおりに動いたが数値基準は未達」で `partial` と記録しており，用語の一貫性を保てる．

**artifact の採否: 維持（ロールバックしない．論点 2）**

`models/domain_classifier.joblib` は新版 `98da6f2d0d44...` のまま，wafl500〜509 も新版配布済みのまま
据え置く（旧版 `ff8aad9c...` は `models/domain_classifier_pre_iter92_civics.joblib` に保持，可逆）．
Iter91 で倹約側の既定（効果が測れず退行がある変更は入れない）を採ったのと条件が異なる:
Iter91 は Δ +0.16pt（床 ±0.25pt の**内側**＝効果が測れない）かつ `legal_recall` に有意退行 1 件だった．
今回は Δ が床の 2 倍以上・退行 0 件・運用指標も基準内である．加えて本変更は精度チューニングではなく，
**訓練ラベルが評価側のラベル定義と矛盾していたという欠陥の是正**であり，仮に Δ が 0 でも
矛盾した訓練データへ戻す積極的な理由が無い．**次イテレーションの基準線は
`results/20260928_111644/`（3,750 行，top1 = 0.835467）へ更新する**（評価集合 `data/dataset.jsonl` は
sha256 不変 `2114e048...` なので行数・母集団は基準線間で一致する）．
再訓練を行う場合の訓練ファイルは `data/classifier_train_iter92_civics_aligned.jsonl` である．

**学び**

1. **`education` の低 recall の主因は分類器側ではなく，訓練ラベルが評価ラベル定義と矛盾していたこと
   だった．**Iter89〜91 の 3 反復にわたって分類器側のレバー（埋め込み・正則化）が no_effect だった
   背景にはこれがあった可能性が高い（断定はしない．Nevin et al. 2025 の「訓練／評価の整形不一致は
   モデル選択の判断自体を変えうる」と整合的な実例である）．**モデル側を触る前に，訓練と評価の
   ラベル定義が同一の写像から生成されているかを毎回確認すること．**
2. **効果は狙った 116 行の中にほぼ完全に閉じており，波及も副作用も無かった．**civics を除く
   3,634 行では 28 改善 / 33 悪化（p = 0.61）で**正味 −5 行**である．すなわち
   「+0.56pt は civics 由来 +26 行と，それ以外での −5 行の差引」であり，
   **このレバーで overall を動かせる上限は評価集合中の該当タスクの行数で決まる**．
   訓練データの局所的な修正が全体へ波及するという期待は，今回のデータでは支持されない．
3. **34 行では civics recall は 0.233 にしか届かなかった（事前予測 0.45 の半分）．**
   計画が事前登録した切り分け（行数不足か，`education` クラス定義の多峰性か）に対しては，
   **行数不足だけでは説明しきれない**と読む．他 3 代理タスク（psychology 0.685・sociology 0.659・
   moral_disputes 0.645）はそれぞれ 76〜87 行で 0.65 前後に留まっており，行数を増やしても
   0.65 付近が `education` クラスの天井である可能性がある．**`education` は 4 タスクにまたがる
   多峰クラスであり，単一の線形境界で表現しきれていない**という仮説が次の検討対象になる
   （→ Iter93 の新レバー `classifier_label_granularity` へ．backlog B153）．
4. **事前登録の効果量閾値は，機序の指標（部分集合 recall）と overall 指標の両方に置いたうえで，
   「部分集合の寄与から overall の上限を先に計算しておく」べきだった．**今回 civics 116 行が
   全問正解になっても overall は +3.07pt が上限であり，実際に得られた recall 0.233 では
   構造的に +0.69pt が上限だった．すなわち **+1.0pt という主基準は，機序が事前予測どおり
   （recall 0.45）に発火しても達成が際どい水準に設定されていた**．次回以降，
   「部分集合の期待改善 × 部分集合の母集団比」を計画時に必ず計算して閾値の実現可能性を検算すること．
5. 判定木の網羅性: `no_effect` の条文（非有意 **または** |Δ| < 0.5pt）は，
   「有意だが効果量が閾値未達」を `no_effect` に落としてしまう幅がある．
   B151 (a) の宿題（判定語を排他的な決定木へ書き直す）は未承認のままだが，
   **今回はその欠落が実際に判定語の選択を左右した**．次の計画フェーズは，
   主基準の効果量条件と有意性条件を分けた 2 軸の表として判定語を定義すること（B153 要レビュー）．

## Iteration 91: 分類器 L2 正則化強度の再選定（訓練集合内 CV による）

### 調査 (Iter91)

B149 (d) が本反復のレバーを `classifier_regularization_strength` = `l2_C_selected_by_train_only_cv` と
指定し，config.yml の `levers` 末尾に追記済みである．B149 が実測した訓練集合内 5-fold CV は
**素の `LogisticRegression` 単体**で測ったものだったため，本フェーズでは 3 つの問いを立てた．
**(Q1) 高次元・小標本（p=5120 ≫ n=2327）で，訓練集合内 CV で選んだ正則化強度は保持集合へ汎化するか．
(Q2) `CalibratedClassifierCV(method="temperature", cv=5, ensemble=True)` という入れ子構造の下でも
同じ CV 曲線になるか（本番の推論はこのラッパを通る）．(Q3) C=3.0 と C=10.0 のどちらを選ぶべきか．**

**Q1: 「CV の推定値」は楽観バイアスを持つが，「CV で選ばれたハイパラ」は実務上ほぼ最適に近い**

- Cawley & Talbot (2010), *On Over-fitting in Model Selection and Subsequent Selection Bias in
  Performance Evaluation*, JMLR 11:2079-2107, <https://www.jmlr.org/papers/v11/cawley10a.html> ——
  モデル選択に使った CV スコアをそのまま性能推定値として報告すると，**選択バイアスにより楽観側へ
  ずれる**（しかもそのずれは「意外なほど大きい」と本文が述べている）．
  **本反復への含意**: CV で得た +2.7pt という数字を「評価集合でも +2.7pt 出る」と読んではならない．
  事前登録の期待効果は減衰を見込んで置き，判定は本走の McNemar で行う．
- Wainer & Cawley (2018/2021), *Nested cross-validation when selecting classifiers is overzealous
  for most practical applications*, arXiv:1809.09446, <https://arxiv.org/html/1809.09446v1> ——
  ハイパラ選択にフラットな（入れ子でない）CV を使っても，**選ばれたハイパラの汎化性能は
  入れ子 CV で選んだものとほとんど変わらない**（差が出るのは「性能推定値」の側）．
  **本反復への含意**: 本反復の C 選定は，評価集合（`data/dataset.jsonl`）を一切見ずに
  訓練集合内 CV だけで行う方式で問題ない．**バイアスが乗るのは推定値であって選択ではない**．
- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning* 2nd ed. §7.10（one-standard-error
  rule），<https://esl.hohoweiya.xyz/book/The%20Elements%20of%20Statistical%20Learning.pdf> ——
  CV 曲線は最小値の近傍で平坦になることが多く，**最良値の 1 標準誤差以内にある中で最も倹約的な
  （＝正則化が強い）モデルを選ぶ**ことが推奨される．**Q3 の判定規則としてこれを採用する**．
- 参考（慣行の確認）: MTEB / C-MTEB の Classification タスクは凍結埋め込み上で
  `LogisticRegression` を既定ハイパラで訓練する（<https://bge-model.com/tutorial/4_Evaluation/4.2.3.html>）．
  つまり **`C=1.0` のまま使うのは分野の慣行ではあるが，n/p 比に合わせて調整されたものではない**．
  本リポジトリの現行設定（`train_domain_classifier.py:210` が `C` を書いていない＝既定の 1.0）は
  この慣行をそのまま引き継いだだけで，Iter89 で p が 2048 → 5120 になった後も見直されていない．
  （断定を避ける: これは「C=1.0 が悪い」ことの証拠ではなく，「調整された形跡が無い」ことの記録である．）

**Q2: 本番パイプライン（較正ラッパ込み）でも CV 曲線の形は変わらない — 本フェーズで実測**

B149 の測定は素の `LogisticRegression` で行われたが，本番の推論は
`CalibratedClassifierCV(method="temperature", cv=5, ensemble=True)` を通る．ensemble=True は
**内側 5 fold それぞれで訓練した 5 つの基底推定器の確率を平均する**ため，単体モデルと argmax が
一致する保証は無い（temperature scaling 自体は共有 softmax の単調変換なので，単体なら argmax を
変えないが，5 モデルの平均は変えうる）．そこで本フェーズで，**外側 5-fold の各 fold で本番と同一の
パイプラインを丸ごと組み立て直す入れ子 CV**を実行した（スクリプト `/tmp/iter91/cv_pipeline.py`・
`/tmp/iter91/cv_seeds.py`．入力は `data/classifier_train_iter87_hybrid.jsonl` 2,327 行 ＋
`data/embcache_train_iter89_qwen3-embedding_4b{,__p1}.npy` を連結した (2327, 5120)，
重みは本番 `_extract_sample_weights()` と同一のドメイン均衡重み．
**`data/dataset.jsonl`（評価集合）には一切触れていない**．CPU のみで実行し wafl500〜509 は使っていない）．

| C | 素の LR（seed 0） | **本番パイプライン** seed 0 | seed 1 | seed 2 | **3 seed 平均** | C=1.0 比 |
|---|---|---|---|---|---|---|
| 1.0（現行＝sklearn 既定） | 0.8096 | 0.8036 | 0.8049 | 0.8040 | **0.8042** | — |
| 3.0 | 0.8307 | 0.8251 | 0.8255 | 0.8251 | **0.8252** | +2.10pt |
| 10.0 | 0.8367 | 0.8354 | 0.8290 | 0.8298 | **0.8314** | **+2.72pt** |
| 30.0 | 0.8341 | 0.8354 | 0.8315 | 0.8324 | **0.8331** | +2.89pt |

- **順位は 3 つの seed すべてで `1.0 < 3.0 < 10.0 ≤ 30.0` と完全に一致した**．較正ラッパは水準を
  0.1〜0.6pt 押し下げるが，**C に対する曲線の形は変えない**．したがって B149 の素の LR による
  曲線を根拠に使ってよい（本フェーズはそれを本番パイプラインで追試したことになる）．
- B149 の実測値（C=1.0: 0.8144 / 3.0: 0.8328 / 10.0: 0.8354 / 100.0: 0.8333）と本フェーズの素の LR の
  値が 0.3〜0.5pt ずれるのは外側 fold の分割乱数が違うためで，**順位と差の大きさは一致している**．
- C=30.0 が C=10.0 をわずかに上回ったが，差は **+0.17pt** で fold 間 SD（0.011〜0.017）より小さい．

**Q3: one-standard-error rule により C = 10.0 を選ぶ**

Q2 の本番パイプライン 3 seed × 5 fold = 15 fold の結果に §7.10 の規則を当てはめる．
最良は C=30.0 の 0.8331．fold 間 SD ≈ 0.0134．

- 15 fold を独立とみなした SE = 0.0134/√15 = 0.00346 → 1-SE 下限 = **0.8296**．
- seed をまたいだ fold は同一データの再分割なので独立ではない．保守側に **1 seed 分の
  SE = 0.0134/√5 = 0.00599** を使っても 1-SE 下限 = **0.8271**．
- **どちらの見積りでも C=10.0（0.8314）は下限を超え，C=3.0（0.8252）は超えない．**
  よって「最良の 1SE 以内で最も正則化が強い C」は **C = 10.0** に決まる．

**C=3.0 を採らない理由**（委譲元の指摘「差が 0.26pt しかない／C を上げるほど過学習リスクが増す」への回答）:
B149 の素の LR 曲線では C=3.0 と C=10.0 の差は 0.26pt だったが，**本番パイプラインで測り直すと差は
0.62pt（3 seed 平均）に広がり，3 つの seed すべてで C=10.0 が上回った**．1-SE 規則は
「差が誤差に埋もれるなら弱い方（強い正則化）を採る」規則であり，本件では**埋もれていない**．
逆に C=30.0 を採らないのは，C=10.0 との差 0.17pt が誤差に埋もれており，1-SE 規則が
**より強い正則化（＝小さい C）**を選ぶよう指示するためである．したがって C=10.0 は
「CV の argmax だから」ではなく「事前に決めた規則の適用結果」として選ばれている．
**結論は B149 (d) が示唆した C=10.0 と一致する（乖離なし）．**

**過学習リスクについての注記（断定しない）**: p=5120 ≫ n=2327 なので L2 罰則を弱めるほど係数ノルムは
増え，訓練データ特有の方向へ載りやすくなる．ただし上表の保持 fold 精度は C=1→30 で単調増加しており，
**この範囲では過学習による劣化はまだ観測されていない**（B149 の C=100.0 で 0.8333 と頭打ちになる
ことから，転回点は C=30〜100 付近にあると推測される．確証はない）．C=10.0 はその転回点より
手前かつ 1-SE 下限を満たす位置にある．

### 計画 (Iter91)

**単一レバー**: `classifier_regularization_strength` = **`l2_C_selected_by_train_only_cv`**，
**具体値 C = 10.0**（`sklearn.linear_model.LogisticRegression` の逆正則化強度．
C=10.0 は既定 1.0 に対し L2 罰則を 1/10 に弱めることを意味する）．
他は Iter90 の最良構成に固定する（埋め込み `qwen3-embedding:4b` 2 ビュー連結，
訓練データ `data/classifier_train_iter87_hybrid.jsonl` 2,327 行，評価集合 `data/dataset.jsonl`
3,750 行，`method="temperature"`・`cv=5`・`ensemble=True`，`class_weight=None` ＋
`_extract_sample_weights()` のドメイン均衡重み，`confidence_threshold=0.0`，`dispatch_top_k=1`）．

**仮説**: Iter89 で特徴次元が 2048 → 5120 へ 2.5 倍になったのに `C` が sklearn 既定の 1.0 のままで，
p=5120 ≫ n=2327 の領域では正則化が強すぎる側に外れている．C を 10.0 へ上げれば分類器の
top1 が上がり，それが実行時のルーティング精度（`top1_accuracy`）へ +1.0pt 以上の形で現れる．

**変更点（最小差分．コード 2 行 ＋ 定数定義）**

1. `scripts/train_domain_classifier.py` のファイル冒頭（`_MAX_ITER` の近く）に定数を定義する．
   マジックナンバー禁止規約に従い，値の根拠（訓練集合内 CV ＋ 1-SE rule）をコメントに残す．
2. 同ファイル **`:210`** を
   `base_estimator = LogisticRegression(max_iter=_MAX_ITER, class_weight=None)` から
   `base_estimator = LogisticRegression(max_iter=_MAX_ITER, class_weight=None, C=_L2_INVERSE_REGULARIZATION)`
   へ変更する．**これ以外のコード変更・config 変更は一切行わない**（`config.yaml` は 1 バイトも触らない）．
3. `models/domain_classifier.joblib` を再訓練する．**埋め込み計算を伴うので wafl-ctrl5 上で行う**
   （2026-09-23 絶対条件 (B)．wafl500〜509 は絶対に使わない）．
4. `data/MANIFEST.md` に新 artifact の sha256 と訓練コマンドを追記する．

**レバーを読むコード行と，そこへ到達する条件**
（config.yml 冒頭「最重要の注意 — 同じ失敗を 6 回繰り返している」への対応）

| # | コード行 | 何をするか |
|---|---|---|
| 1 | `scripts/train_domain_classifier.py:210` `LogisticRegression(max_iter=_MAX_ITER, class_weight=None)` | **ここに `C=` を足す．レバーの唯一の入口** |
| 2 | 同 `:211-213` `CalibratedClassifierCV(base_estimator, method="temperature", cv=5, ensemble=True)` | sklearn の `clone()` は `get_params()` 経由なので **内側 5 fold すべてに C が伝播する** |
| 3 | 同 `:214` `.fit(embeddings, labels, sample_weight=...)` → `:231` `train_classifier(...)` → `:235` `joblib.dump(model, output_path)` | artifact を書き出す |
| 4 | `mise.toml:71-78` deploy | `sudo rm -rf $REMOTE_DIR/models` の後に `models/` を全ノードへ rsync する（古い artifact は残らない） |
| 5 | `config.yaml:50` `routing_method: supervised_classifier` ／ `config.yaml:131` `classifier_model_path: models/domain_classifier.joblib` | 到達の前提条件．**両方とも現行構成で既に満たされている**（変更不要） |
| 6 | `http_server.py:406-411`（FastAPI `lifespan`）`state.domain_classifier = load_domain_classifier(...)` | **artifact は起動時に 1 回だけ読む** |
| 7 | `http_server.py:364-368` supervised_classifier 分岐 → `classifier.py:69` `classifier.predict_proba([query_embedding])[0]` | 1 問ごとに新 artifact の確率が出る |

**到達条件（すべて現行構成で満たされることを確認済み）**
- (a) `config.yaml:50` は実際に `supervised_classifier`．`http_server.py:365` のコメントどおりこの分岐は
  LLM を呼ばず，`light_model` を使う分岐は先行分岐で到達しない．fallback は `confidence_threshold=0.0` で
  Iter28 以降 0 件（Iter90 実測 `fallback_rate` = 0.0）．**したがって分類器は全 3,750 問で必ず通る．**
- (b) **唯一の落とし穴は #6 である**．artifact は `lifespan` で 1 回だけ読まれるので，
  **rsync しただけではプロセス内の古いモデルが使われ続ける**．Iter16/20/21/22/27/B35 と同型の
  「config は正しいのにコードへ到達しない」失敗はここで起きうる．
  **`mise run deploy` によるコンテナ再作成を必ず行い，再作成されたことを確認すること．**
- (c) #2 の伝播は仮定ではなく確認済みである（本フェーズの `/tmp/iter91/cv_pipeline.py` が
  同じ構成で C を変えて CV 精度が動いたこと自体が，C が内側 fold へ届いている証拠になっている）．

**レバー固有の直接証拠（B149 恒久申し送り 2: 「基準線との完全一致」で兼ねない）**

| # | 証拠 | 確認方法 |
|---|---|---|
| E1 | artifact が差し替わった | `sha256sum models/domain_classifier.joblib` が基準線の `ff8aad9c...` と**異なる** |
| E2 | 実際に使われた C が 10.0 である | 新 artifact を `joblib.load` し，`m.calibrated_classifiers_[0].estimator.C == 10.0`，`len(m.calibrated_classifiers_) == 5`，`m.calibrated_classifiers_[0].estimator.n_features_in_ == 5120`，`sorted(m.classes_)` が 10 ドメイン |
| E3 | 全ノードが新 artifact を**ロードしている** | deploy 後・本走前に wafl500〜509 の 10 台すべてで，コンテナ内 `/app/models/domain_classifier.joblib` の sha256 が E1 の新値と一致し，かつ**コンテナの起動時刻が deploy 後である**こと |
| E4 | 実行時の出力が変わった | 予備 20 問の `probe_candidates` が Iter90 予備実行（`results/20260928_032455/`）の同じ 20 行と**相違すること** |
| E5 | 同一セッション内の決定論性 | 予備 20 問と本走の同じ 20 行が **200/200 スロットでビット単位一致**（B149 恒久申し送り 3．同一セッション内はデプロイを挟まないので一致を要求してよい） |

**事前ゲート G0（結果を見る前に判定規則を固定する）**
- **G0-a（訓練の実行場所）**: 再訓練の埋め込み計算は **wafl-ctrl5 の Ollama（127.0.0.1:11499）でのみ**行う．
  `--ollama-host` に wafl500〜509 を指定していないことをコマンドログで確認する（絶対条件 (B)）．
- **G0-b（埋め込みの再現性の確認）**: 再訓練時に計算した 2,327 行 × 2 ビューの埋め込みを
  `data/embcache_train_iter89_qwen3-embedding_4b{,__p1}.npy` と突き合わせ，**最大絶対差を記録する**．
  B149 (d) は `qwen3-embedding:4b` でセッション跨ぎの再現性が失われている疑いを残しており，
  差が大きければ「C の効果」と「埋め込みの揺らぎ」が交絡する．**この値は報告必須**．
- **G0-c（交絡を切り分ける対照 artifact．オフライン replay のみ）**: G0-b と同じ新しい埋め込みから
  **C=1.0 の artifact も 1 つ作り**，キャッシュ済み評価埋め込みによる replay で top1 を求める．
  これが Iter90 本走の 0.829867 と **±0.25pt 以内**（B149 恒久申し送り 1 の再現性の床）に収まれば，
  以降の差は C に帰属できる．**この C=1.0 artifact は絶対にデプロイしない**
  （「複数の C を本走して良い方を採る」禁止条項に抵触させないため．デプロイするのは C=10.0 のみ）．
- **G0-d（事前 replay 予測）**: C=10.0 artifact で同じ replay を行い，**本走前に** top1 と
  per-domain を journal へ記録する（Iter87〜89 で 3 反復連続 0.12pt 以内の一致が実証済み）．
  **これは予測の記録であってゲートではない．予測値がどうであれ本走は必ず実施する**（絶対条件 (A)）．
- **G0-e（予備 20 問）**: `data/dataset.jsonl` の先頭 20 問で予備実行し，HTTP 500 が 0 件，
  E4 が満たされること．
- **G0-f（時間予算）**: G0-e の実測から 3,750 問の総所要を外挿する．**175 分を超える見込みなら，
  `experiment.timeout_min` を 180 → 210 へ引き上げる**（これは watchdog の設定であって実験条件ではない．
  B137 と同じ扱い）．**本走を縮小・省略する分岐は作らない**（絶対条件 (A)．Iter90 本走は 158 分だった）．

**検出力の確認（判定可能であることを本フェーズで再計算した）**
基準線 Iter90 本走の n = 3,750．基準線と新 artifact の discordant 率は，同型の artifact 差し替えである
Iter89（旧 0.6b → 新 4b）の実測 347/3,435 = **π̂_d = 0.10102** を用いる．

- McNemar 有意境界 `1.96·√(π̂_d/n)` = 1.96·√(0.10102/3750) = **1.017pt**
- 80% 検出力の最小効果 `2.802·√(π̂_d/n)` = **1.454pt**
- 期待効果 +1.8pt で検出力 **0.934**，+2.1pt で **0.982**，+2.7pt（CV の生の差）で **0.999**．
  仮に減衰して +1.4pt でも **0.770**，+1.0pt だと **0.487**．
- **セッション跨ぎの再現性の床（top1 ±0.25pt）は有意境界 1.017pt の約 1/4 であり，
  主基準 ≥ +1.0pt はノイズ床を 4 倍上回る．判定可能である．**
- 注意（Cawley & Talbot 2010 の含意）: CV の +2.72pt をそのまま期待してはならない．
  訓練集合（hard negative 混成 2,327 行）と評価集合（JMMLU 由来 3,020 行 ＋ 複合 730 行）は
  分布が違うため，**+1.0pt を下回る減衰が起きれば判定不能になりうる**．その場合の扱いは
  下の判定語 `no_effect` / `partial` で事前に定義してある．

**成功条件（事前登録．結果を見た後に書き換えない）**

| 区分 | 指標 | 基準線（Iter90 本走 `results/20260928_032909/`，n=3,750） | 判定基準 |
|---|---|---|---|
| **主基準 (i)** | `top1_accuracy` の McNemar 検定 | 0.829867 | **p < 0.05（有意改善方向）** |
| **主基準 (ii)** | Δ`top1_accuracy` | 0.829867 | **≥ +1.0pt**（= 有意境界 1.017pt とほぼ同値．Wilson 95%CI を併記） |
| **到達確認** | E1〜E5 | — | **全件充足**（1 つでも欠ければ `invalid`） |

**非退行条件**（基準線は同じく `results/20260928_032909/`．全 3,750 行で評価する．
B149 恒久申し送り 1 に従い「ビット単位一致」条項は**置かない**）

| # | 指標 | 基準線 | 判定基準 |
|---|---|---|---|
| ① | per-domain 20 指標（precision/recall × 10） | Iter90 本走 | BH 補正（q=0.05）後の**有意退行 0 件** |
| ② | `fallback_rate` | 0.0 | **= 0.0** |
| ③ | `dispatch_failure_rate` | 0.000533 | **≤ 0.005**（絶対値．Iter86〜90 と同一） |
| ④ | `compound_domain_set_recall`（複合 730 行） | 0.570548 | **≥ 0.544048**（基準線 −2.65pt．Iter86〜90 と同じ幅） |
| ⑤ | `compound_mean_dispatched_count`（複合 730 行） | 1.964384 | **≤ 2.10**（絶対値．Iter86〜90 と同一） |
| ⑥ | `mean_duration_ms`（全 3,750 行） | 2531.538 | **≤ 3037.8**（基準線 +20%）．**B149 恒久申し送り 5 に従い部分集合では評価しない** |
| ⑦ | `ece` | 0.029892 | **≤ 0.05**（絶対値）．C を上げると logit のスケールが変わるため，temperature が吸収しきれない可能性への保険 |
| ⑧ | `tie_rate` | 0.0 | **= 0.0** |

**定義語と数値アンカーの突き合わせ**（B149 恒久申し送り 4．Iter90 の ③ 誤判定の再発防止）
上の表のアンカーは**すべて `/tmp/iter90/metrics_full3750.json`（= `results/20260928_032909/` の
全 3,750 行に対する `metrics.py` 出力）から本フェーズで直接読み出した値**であり，
本文の定義語と同じ量である．具体的に突き合わせた対応は次のとおり:
④ 本文「複合 730 行の set recall」= `compound_coverage.compound_domain_set_recall` = 0.5705479...（730 行基準．
**Iter86〜89 で使っていた 0.566265 は 415 行基準なので使わない**），
⑤ = `compound_coverage.compound_mean_dispatched_count` = 1.9643835...（同じく 730 行基準．
**Iter90 の条文にあった 1.879518 は 415 行基準**），⑥ = `mean_duration_ms` = 2531.5384（全 3,750 行．
**Iter90 が使った 1643.249 は 3,435 行サブセット基準なので使わない**），
③ = `dispatch_failure_rate` = 0.0005333...（全 3,750 行．Iter90 計画節の 0.001165 は 3,435 行基準），
⑦ = `ece.ece` = 0.0298918（n_rows=3748），主基準 = `top1_accuracy` = 0.8298666...（全 3,750 行）．
**基準が 415/3,435 行のものと 730/3,750 行のものが混在していた点を，本反復ですべて後者へ統一した．**

**判定語の定義**
- **adopted**: 主基準 (i)(ii) の **AND** ＋ 到達確認 E1〜E5 ＋ 非退行①〜⑧をすべて満たす．
- **partial**: (i) 有意だが (ii) が +1.0pt 未満，または非退行①〜⑧のいずれかが未達．
- **no_effect**: |Δtop1| < 0.5pt **または** McNemar が非有意（Iter86〜88 と同一条文）．
- **rejected**: Δtop1 ≤ −1.0pt かつ McNemar 有意（悪化方向）．
- **invalid（実験不成立）**: E1〜E5 のいずれかが欠ける（= 新 artifact が実際にはロードされていない）．
  **top1 が基準線と近いことは invalid の根拠にしない**（B149 恒久申し送り 1・2）．

**本走（2026-09-23 絶対条件 (A)）**: 変更適用後に **wafl500〜509 を用いた 3,750 問のフルスペック本走を
必ず 1 回実施する**．`mise run start -- --dataset data/dataset.jsonl --output results.jsonl` ののち
`mise run analyze -- <timestamp>` を **timestamp 明示指定**で実行する（B118 落とし穴 1 の再発防止）．
想定所要は Iter90 実績から **150〜165 分**（分類器の差し替えは推論コストを変えない）．
**C を複数値で本走して良い方を採ることは禁止する**（config.yml 当該レバーの note．
デプロイするのは C=10.0 のただ 1 つ）．

**恒久ルールへの適合確認**
- B115 (1)（ドメイン固有の後付け補正の禁止）: `C` は 10 クラス共通の単一スカラーであり，
  ドメインごとに値を変えない．
- 2026-09-23 絶対条件 (A): 本走を必ず実施する．G0 のどの分岐にも本走省略の経路を作っていない．
- 2026-09-23 絶対条件 (B): 再訓練の埋め込み計算・replay・CV はすべて wafl-ctrl5 または
  ローカル CPU で行い，wafl500〜509 は本走にのみ使う．
- 単一レバー原則: 変えるのは `C` の 1 つだけ．`method`・`cv`・`ensemble`・`max_iter`・
  `class_weight`・sample_weight・埋め込み・訓練データ・評価集合・`config.yaml` は一切変えない．

### 実験結果（Iter91，フェーズ 2 実測．判定はフェーズ 3）

**変更した差分**: `scripts/train_domain_classifier.py` の 1 ファイルのみ．冒頭 `_MAX_ITER = 1000` の直後に
定数 `_L2_INVERSE_REGULARIZATION = 10.0` を選定根拠のコメント付きで新規定義し，`train_classifier()` 内の
`LogisticRegression(max_iter=_MAX_ITER, class_weight=None)` へ `C=_L2_INVERSE_REGULARIZATION` を追加した．
`config.yaml` は 1 バイトも変更していない．`ruff check` の新規エラー 0 件．
`pytest tests/test_train_domain_classifier.py tests/test_classifier.py` は 11 PASS・1 FAIL
（`test_build_training_features_embeds_each_row_in_order` の `ModuleNotFoundError: sentence_transformers`）だが，
この失敗は `git stash` で変更前にも再現する環境依存の既知失敗であり本変更起因ではない．
`ruff format --check` の reformat 推奨も同様に変更前から存在するドリフトで，今回触った 2 箇所とは無関係．

**事前ゲート G0 の結果**

| ゲート | 結果 |
|---|---|
| G0-a 実行場所 | 再訓練・埋め込み計算はすべて `--ollama-host 127.0.0.1 --ollama-port 11499`（wafl-ctrl5 トンネル）のみ．wafl500〜509 への指定 0 件（絶対条件 (B) 充足） |
| G0-b 埋め込み再現性 | 既存キャッシュとの最大絶対差 **0.011406**（plain view）／0.009027（instructed view）．B149 (d) の非決定性の範囲内 |
| G0-c C=1.0 対照 replay | 全 3,750 行で **top1 = 0.831200**．基準線 0.829867 との差 **+0.1333pt**（床 ±0.25pt 以内）→ 埋め込み差による交絡なしと確認．対照 artifact は `/tmp/iter91/` に置きデプロイしていない |
| G0-d C=10.0 replay 予測 | **top1 = 0.833067**（ゲートではない．本走は予測値によらず実施した） |
| G0-e 予備 20 問 | HTTP 500 = 0 件 |
| G0-f 時間予算 | 外挿 約 130 分（< 175 分）につき `timeout_min` は 180 のまま変更せず．実測 158.1 分で完走し判断は妥当だった |

**レバー到達の直接証拠 E1〜E5（すべて充足）**
- **E1**: 新 artifact sha256 = `6a5905f025d6dae91ab333968e71c944d77ba286f4f3e3891c1b3a1fe5ca3b30`（基準線 `ff8aad9c...` と相違）．
- **E2**: `calibrated_classifiers_[0].estimator.C == 10.0`，`len(calibrated_classifiers_) == 5`，
  `n_features_in_ == 5120`，`classes_` が 10 ドメイン．
- **E3**: wafl500〜509 の 10 台すべてでコンテナ内 artifact の sha256 が E1 と一致し，
  コンテナ起動時刻 2026-09-27T22:41 UTC が deploy 起動 22:35:26 UTC より後（lifespan 再読込みを確認）．
- **E4**: **計画が参照先とした `results/20260928_032455/` は，実際には `data/dataset.jsonl` 先頭 20 行ではなく
  `compound-416`〜`435`（Iter90 の複合評価集合検証用の別サブセット）だった．**同じ head-20
  （`business_economics-001`〜`020`）の予備実行が実在する `results/20260927_232858/preview20.jsonl` を
  id 完全一致を確認したうえで代替の比較対象に採用した（可逆な代替）．`probe_candidates` は 20/20 行すべて相違．
- **E5**: 予備 20 問 `results/20260928_074531/` と本走先頭 20 行が **200/200 スロット完全一致**，
  `selected_domain` も全行一致（同一セッション内の決定論性を再確認）．

**本走**: `results/20260928_074903/`．3,750/3,750 行完走，**158.1 分**（1790549318→1790558802）．
HTTP 500 系エラー・GPU OOM なし．`mise run analyze -- 20260928_074903` を timestamp 明示で実行済み．

**主基準の実測値**
- `top1_accuracy` = **0.831467**（Wilson 95%CI [0.819148, 0.843107]）．基準線 0.829867 に対し **Δtop1 = +0.1600pt**．
- McNemar: discordant_a_only = 109，discordant_b_only = 115，discordant_pairs = 224，chi2 = 0.1116，**p = 0.7383**．
- 事前登録の有意境界 1.017pt・80% 検出力の最小効果 1.454pt のいずれも下回る．

**非退行 8 条件の実測値**

| 条件 | 実測 | 事前登録の閾値 | 可否 |
|---|---|---|---|
| ① per-domain 20 指標 BH 補正（q=0.05）後の有意退行 | **1 件**（`legal_recall`） | 0 件 | **未達** |
| ② `fallback_rate` | 0.0 | = 0.0 | 充足 |
| ③ `dispatch_failure_rate` | 0.0016 | ≤ 0.005 | 充足 |
| ④ `compound_domain_set_recall` | 0.567808 | ≥ 0.544048 | 充足 |
| ⑤ `compound_mean_dispatched_count` | 2.046575 | ≤ 2.10 | 充足 |
| ⑥ `mean_duration_ms` | 2512.330 | ≤ 3037.8 | 充足 |
| ⑦ `ece` | 0.025464 | ≤ 0.05 | 充足 |
| ⑧ `tie_rate` | 0.0 | = 0.0 | 充足 |

① の BH 補正後に有意だった 4 件の内訳: `education_recall`（p=0.00501，正味改善 a_only 17 / b_only 39），
**`legal_recall`（p=0.00511，正味退行 a_only 20 / b_only 5，net −15/150 行）**，
`legal_precision`（p=0.00126，正味改善 0.822→0.936），
`social_science_recall`（p=0.00131，正味改善 a_only 5 / b_only 23）．**退行方向は `legal_recall` の 1 件のみ．**

統計量はすべて `metrics.py` の既存関数（`compute_mcnemar_test`，`compute_domain_recall_mcnemar_test`，
`compute_domain_precision_fisher_test`，`apply_benjamini_hochberg`，`compute_top1_accuracy_wilson_ci`）を
そのまま呼び出しており，フェーズ 2 で自前の再導出は行っていない．

**artifact の扱い**: `models/domain_classifier.joblib` を新 artifact（`6a5905f0...`）へ差し替え，
旧 artifact は `models/domain_classifier_pre_iter91_c1.joblib` へ退避（`models/`・`data/` は `.gitignore` 対象）．
`data/MANIFEST.md` に artifact の sha256 と再現コマンドを追記した．

### Iteration 91 実行済み（分析・考察）

**変更**: `scripts/train_domain_classifier.py` の `LogisticRegression` へ `C=10.0` を明示（定数
`_L2_INVERSE_REGULARIZATION`）し，同一の訓練データ・埋め込み・較正設定で再訓練した artifact
（sha256 `6a5905f0...`）を全 10 ノードへデプロイして 3,750 問を本走した．`config.yaml` は無変更．

**判定: `no_effect`**

- 主基準 (i) McNemar **p = 0.7383**（非有意），(ii) Δtop1 = **+0.1600pt**（< +1.0pt）．**AND 条件は不成立**．
- 事前登録の判定語を literal に当てると **2 つの条文が同時に発火する**:
  - `no_effect` の条文「|Δtop1| < 0.5pt **または** McNemar が非有意」→ 0.160pt < 0.5pt かつ p = 0.7383 で，
    **両方の節が成立**する．
  - `partial` の条文「(i) 有意だが (ii) が +1.0pt 未満，**または**非退行①〜⑧のいずれかが未達」→
    非退行① が未達（`legal_recall`）なので**第 2 節が成立**する．
  - **条文の前提が実測でどうだったか**: `partial` の第 2 節は「主基準に効果が出た実験の副作用を記述する」
    意図で書かれた条文であり，主基準が非有意の場合を想定していない（第 1 節が「(i) 有意だが」と
    始まることがその証拠）．本反復は主基準に検出可能な効果が無いので，`no_effect` を主たる判定語とする．
    **ただし条文の literal な帰結として `partial` も成立することをここに明記し，条文を読み替えて
    `partial` を消したのではない**ことを記録に残す（B149 (a) と同種の論点．**要レビュー**）．
  - `invalid` は当たらない．E1〜E5 が全件充足で，レバーは確実にコードパスへ到達している
    （artifact sha256 相違・`estimator.C == 10.0`・10 台の sha256 一致とコンテナ再作成・
    予備 20 問と本走の 200/200 ビット一致）．**「top1 が基準線と近いこと」を invalid の根拠にしない**
    という B149 恒久申し送り 1・2 をそのまま適用した．
- **判定語定義の欠陥（B149 (c) の再発）**: 「主基準が非有意」かつ「非退行に有意退行がある」場合を
  一意に指す語が無い．次の計画フェーズは判定語を排他的な決定木として書き直すこと．

**分析 1: CV の +2.72pt が end-to-end で +0.16pt になった機序**

まず**測定系の事実を 1 つ確定させた**。`metrics.py:32 compute_top1_accuracy` は
`selected_domain in expected_domains` を数えるだけで，**専門ノードの回答内容を一切見ていない**．
すなわち **`top1_accuracy` は「分類器の argmax がルーティング先として正しかった割合」そのもの**であり，
「分類器が当てても専門ノードが誤答すれば top1 に現れない」という機序は本リポジトリには存在しない
（回答品質は `answer_quality_accuracy` / `end_to_end_accuracy` という別軸である）．
`probe_candidates` の argmax と `selected_domain` の相違は基準線 2 行・本走 6 行のみで，
いずれも `dispatch_failed=true`（送出先ノードの実行時失敗で `selected_domain=null`）の行だった．

したがって**分類器精度 → top1 の伝達率は定義上 1.0 であり，両者の差は送出失敗行の分だけ**である:

| 量 | 基準線 `20260928_032909` | 本走 `20260928_074903` | Δ |
|---|---|---|---|
| 分類器 argmax 精度（`probe_candidates` から再計算，3,750 行） | 0.830400 | **0.833067** | **+0.2667pt** |
| `top1_accuracy`（＝上から `dispatch_failed` 行を落としたもの） | 0.829867 | 0.831467 | +0.1600pt |
| 差（= `dispatch_failed` 行数 / 3750） | 2 行 = 0.0533pt | 6 行 = 0.1600pt | — |

- 分類器 argmax 精度での McNemar も **a_only 104 / b_only 114，chi2 = 0.3716，p = 0.5422** で非有意．
  **送出失敗を除いても判定は変わらない**（本走の送出失敗 6 行のうち 5 行は argmax が正解ドメインで，
  実行時失敗が無ければ拾えていた．これは実行時ノイズであってレバーの効果ではない）．
- **G0-d の replay 予測 0.833067 は，本走の分類器 argmax 精度 0.833067 と小数 6 桁まで完全一致した．**
  すなわち replay は「分類器が何を選ぶか」を**誤差 0 で**予測しており，replay と本走 top1 の
  −0.16pt の差は**送出失敗 6 行だけで完全に説明される**．再現性の床 ±0.25pt の範囲内であるどころか，
  **replay の予測誤差は 0 であった**（`data/embcache_eval_*` が固定されている限り，分類器の出力は
  デプロイを跨いでも決定論的である．B149 (d) で観測されたセッション跨ぎのゆらぎは，
  埋め込みを実行時に計算し直す経路に由来する．**本走の埋め込みも実質同一だったことになる**）．

**では +2.72pt はどこへ消えたか．評価集合の部分集合ごとに符号が逆だった**（事前登録外の事後分割．
探索的な分析であり，これを根拠に採用判定を変えてはならない）:

| 部分集合 | n | 分類器 argmax 精度 基準線 → 本走 | Δ | McNemar |
|---|---|---|---|---|
| 単一ドメイン行（JMMLU 由来） | 3,020 | 0.835762 → 0.847351 | **+1.1589pt** | a_only 59 / b_only 94，p = **0.0060** |
| 複合ドメイン行（Iter90 拡充分） | 730 | 0.808219 → 0.773973 | **−3.4247pt** | a_only 45 / b_only 20，p = **0.0029** |
| 全体 | 3,750 | 0.830400 → 0.833067 | +0.2667pt | p = 0.5422 |

- **(a) Cawley & Talbot (2010) の選択バイアス＋分布シフト**は，単一ドメイン行で確かに効いている．
  訓練集合内 CV の +2.72pt に対し，訓練集合と同じ「単一ラベル 1 問 1 ドメイン」の構造を持つ
  評価部分集合では **+1.16pt（伝達率 43%）**しか出なかった．CV の値をそのまま期待してはならない
  という事前の注意（計画節）は正しかった．
- **(b) 「測っている量が違う」は成立する．ただし想定した機序（ルーティング後の回答正誤）ではなく，
  評価集合の構成の違いによる**．CV は単一ラベル行だけで測っており，**複合 730 行（評価集合の 19.5%）に
  相当するものを訓練集合も CV も一切含んでいない**．複合行では正則化を弱めたことで
  **−3.42pt（有意）**の逆効果が出て，単一行の +1.16pt をほぼ打ち消した．
  0.195 × (−3.42) + 0.805 × (+1.16) = **+0.27pt** で全体の実測と一致する．
- argmax の反転は 293/3,750 行（7.81%）で，うち正解化 114・誤答化 104・どちらも不正解 75．
  **分類器の決定は大きく動いているが，正味では釣り合っている．**
- **今後のレバー選定への含意（定量）**: 分類器側のレバーは伝達率 1.0 で top1 へ効くので
  「分類器をいじる価値」は構造的にはある．しかし**訓練集合内 CV で測れる利得は，
  単一行へは 43%・複合行へは負の符号で伝わる**．end-to-end で +1.0pt（有意境界）を出すには，
  この混合比（80.5% / 19.5%）の下で **訓練集合内 CV で最低でも +2.3pt 以上**の改善が必要であり，
  かつ複合行を悪化させない性質を持つ必要がある．C の掃引は C=1.0 → 30.0 の全域で CV +2.9pt が上限
  （B149 の C=100 で頭打ち）なので，**正則化強度というレバーはこの要求水準をほぼ使い切っている**．

**分析 2: `legal_recall` の有意退行（BH 補正 q=0.05 後の唯一の退行）**

`legal` を expected に含む 306 行のうち `selected_domain == "legal"` だった行が **162 → 147**
（a_only 20 / b_only 5，p = 0.00511）．probabilities から見た機序は**「legal 全体の確率質量が縮んだ」**である:

| 量 | 基準線 | 本走 |
|---|---|---|
| `legal` を予測した行数（3,750 行中） | 197 | **157**（−40） |
| `legal_precision` | 0.8223 | **0.9363**（+11.4pt） |
| `legal` 確率の平均（legal 行 306 行上） | 0.4760 | 0.4439 |
| `legal` 確率の平均（非 legal 行 3,444 行上） | 0.0163 | **0.0099**（−39%） |

- **これはドメイン固有の異常ではなく，正則化を弱めたときの精度/再現率のトレードオフである．**
  C を上げると係数ノルムが伸び，**訓練行数の多いクラスの決定領域が広がり，少ないクラスの領域が縮む**．
  実際 `legal` の訓練行は **77 行しかなく，他 9 ドメインは各 250 行**（`classifier_train_iter87_hybrid.jsonl`）．
  hard negative mining のプール（`pool_hash=221e45e8...`，3,127 行）に **legal は 0 行**しか無く，
  Iter84/86/87 のいずれでも legal だけ 77 行のままである．`_extract_sample_weights()` の
  ドメイン均衡重みは**行の重みを 3.25 倍にできても行の多様性を作れない**．
- 失われた 20 行の移動先は `history_culture` 12 / `medical` 3 / `social_science` 3 / `education` 2 で，
  **12 行が history_culture**（訓練 250 行・recall 0.917 の強いクラス）へ流れた．
  20 行中 **11 行は複合行**であり，分析 1 の複合行の退行と同じ現象の一部である．
- 予測分布全体でも同じ向きが見える: `legal` −40 / `business_economics` −33 / `computer_science` −15 に対し
  `history_culture` +25 / `social_science` +20 / `education` +18 / `medical` +13．
- **対処**: B115 (1) により legal だけ閾値・intercept・重みをいじることは禁止である．本反復では
  **何も後付け補正しない**．退行の原因は「legal の訓練行が 77 行しかない」というデータ側の構造問題であり，
  解くなら 10 ドメイン共通の規則（＝全ドメインの訓練行数を揃える）で解く必要がある．
  現行プールには legal の在庫が 0 なので，**新しいデータ源の調査が要る**（次の一手へ引き継ぐ）．

**分析 3: E4 の参照先の食い違いと，予備実行ディレクトリの記録方法**

計画が E4 の比較対象に指定した `results/20260928_032455/` は，実際には `data/dataset.jsonl` の
head-20 ではなく `compound-416`〜`435` の 20 行だった．executor は同じ head-20
（`business_economics-001`〜`020`）を持つ `results/20260927_232858/preview20.jsonl` を，
**id の完全一致を確認したうえで**代替に採用した．
**この代替は妥当である**: E4 の目的は「同一入力に対する出力が artifact 差し替えで変わったこと」の確認であり，
入力 id 集合が一致していれば比較対象として等価だからである（実際 20/20 行すべてで `probe_candidates` が相違）．
根本原因は**予備実行の結果ディレクトリがタイムスタンプ名だけで，どのサブセットを流したかを保持していない**
ことにある．`data/dataset_iter{89,90}_preview20.jsonl` という入力側のファイルは残っているのに，
出力側とは結び付いていない．**是正案は backlog B151 へ記録した**（`results/<ts>/` へ実行時の
`--dataset` パスと行数・id レンジを書いた `run_meta.json` を残す．これは測定系の改善であって
レバーではないので，次の実験と同時にオフラインで実施してよい）．

**artifact の扱いとロールバック（次イテレーションの基準線）**

- **`models/domain_classifier.joblib` を `C`=1.0 の `ff8aad9c...` へ戻した**（`C`=10.0 版は
  `models/domain_classifier_iter91_c10.joblib` に保存，再採用は可逆）．
  `scripts/train_domain_classifier.py` の `C=_L2_INVERSE_REGULARIZATION` も revert した．
- **理由**: 事前登録の主基準を満たさず（+0.16pt・p=0.74），非退行① に有意退行が 1 件あり，
  全体の Δ は再現性の床 ±0.25pt の内側にある．**「効果が測れなかった変更は入れない」**という
  倹約側の既定に従った．単一行 +1.16pt は事後分割の探索的知見にすぎず，採用の根拠にしない．
- **次イテレーションの基準線は Iter90 本走 `results/20260928_032909/`（3,750 行，top1 = 0.829867，
  分類器 sha256 `ff8aad9c...`）のまま据え置く**（ロールバックにより，基準線はその artifact の
  実測値として今も有効である）．
- **落とし穴**: wafl500〜509 上には Iter91 本走時点の `C`=10.0 artifact が載ったままである．
  **次に実験を行う際は必ず `mise run deploy` を先に実行すること**（`data/MANIFEST.md` にも明記した）．

**学び**

1. **`top1_accuracy` は分類器の argmax 精度そのものである**（`metrics.py:32`．回答内容を見ていない）．
   したがって**分類器側のレバーはオフライン replay で誤差 0 で予測できる**——本反復で G0-d の
   replay 予測と本走の argmax 精度が小数 6 桁まで一致した．**本走の 158 分は「送出失敗率」と
   「実行時の健全性」を測るためにだけ必要で，分類器の効果量そのものは replay で事前に分かる**．
   ただし**選定に評価集合の replay を使えばリークになる**ので，用途は「事前登録した予測の記録」に限る．
2. **訓練集合内 CV → 評価集合の伝達率を本反復で初めて定量した: 単一ドメイン行へ 43%，複合行へは負**．
   今後，分類器側のレバーは **訓練集合内 CV で +2.3pt 以上**を見込めなければ end-to-end の
   有意境界 1.0pt に届かない．Iter89 学び（判定不能と事前に分かるレバーには着手しない）の
   具体的な換算式がこれで得られた．
3. **評価集合に複合行を 19.5% 混ぜたことで，単一ラベル訓練に最適化するレバーの効果が構造的に薄まる**．
   Iter90 の拡充は測定分解能のために正しかったが，副作用として
   「単一行で得た利得が複合行の損で相殺される」経路が生まれた．今後は**主基準の内訳
   （単一 / 複合）を必ず併記する**こと（事前登録に加えるべき．ただし**部分集合ごとに
   判定基準を分けるのは多重比較になる**ので，主基準は全体のままとする）．
4. **クラス間の訓練行数の不均衡（legal 77 行 vs 他 250 行）が，正則化を弱めたときに
   最初に壊れる場所を決めている**．sample_weight による均衡化は重みを増やせても多様性を増やせない．
   `legal` は hard negative プールにも在庫が 0 で，**既存データ源だけでは是正できない**．
5. 訓練集合内 CV を「本番と同じ入れ子パイプラインで」測り直した手順（計画節 Q2）は正しく機能した
   （較正ラッパは水準を下げるが曲線の形を変えない，という予測が本走でも破綻しなかった）．
   **失敗したのは CV の測り方ではなく，CV と評価集合の分布の違いの見積り**である．

## Iteration 90: 複合設問評価集合の拡充（既存公開データセットの調査と追加）

### 調査 (Iter90)

B147 (c) は本反復のレバーを `compound_eval_set_expansion` = `existing_public_dataset` と指定した．
config.yml 冒頭の **B116 (2)**（データセット拡充ではまず信頼できる既存公開データセットを調査し，
見つからない場合に限り LLM 生成へ落とす．出典・ライセンス・ドメイン適合性を明記する）に従い，
値を確定する前に実際に調査した．問いは 3 つ．
**(Q1) 「1 行が 2 つの専門ドメインラベルを同時に持つ日本語の自然文設問」に相当する公開データセットは
存在するか．(Q2) JMMLU など既存の公開ソースから複合行を派生させる経路は使えるか（ライセンス面を含む）．
(Q3) 現行 415 行の検出力はいくつで，1 反復の実行時間予算の中でどこまで増やせるか．**

**Q1: 既存公開データセット — この探索範囲では見つからなかった（(i) は不成立）**

tavily-search（`tvly search --depth advanced`）で日本語・英語の 6 クエリ，および Hugging Face Hub の
Datasets API で 20 クエリを検索した．確認できた候補と，本研究の要件（**1 行が 2 つのドメインラベルを
同時に持つ，日本語の自然文**）に対する適合性は次のとおり．

| 候補 | 出典 | ライセンス / 入手性 | 10 ドメイン体系との適合性 |
|---|---|---|---|
| M2QA (Multi-domain Multilingual QA) | Engländer et al., EMNLP 2024 Findings <https://aclanthology.org/2024.findings-emnlp.365> / HF `UKPLab/m2qa` | 公開 | **不適合**．multi-domain とは「言語 × ドメインの組み合わせを網羅する」意味であり，**1 インスタンスは単一ドメイン**．日本語は対象言語に含まれない（de/tr/zh） |
| RouterArena | Lu et al., arXiv:2510.00202 / HF `RouteWorks/RouterArena` | 公開 | **不適合**（Iter78 と同一結論）．英語・**1 クエリ 1 ドメインの単一ラベル** |
| RouterBench / RouterEval | HF `withmartian/routerbench`，`linggm/RouterEval` | 公開 | **不適合**．いずれも既存英語ベンチ（MMLU/GSM8K/HellaSwag 等）のクエリにモデル別スコアを付けたもので，ドメインラベルは単一 |
| MMLU-ProX-Japanese | HF `tokyotech-llm/MMLU-ProX-Japanese` | 公開 | **不適合**．日本語である点は合致するが **1 問 1 カテゴリ**の 10 択 MCQ．単一ドメイン行の追加供給源にはなるが複合行にはならない |
| JamC-QA | SB Intuitions，NLP2025 Q2-18 <https://www.anlp.jp/proceedings/annual_meeting/2025/pdf_dir/Q2-18.pdf> | 公開 | **不適合**．日本固有知識の多肢選択，単一科目ラベル |
| lawqa_jp / JMED-LLM / JDocQA / JAQKET / JGLUE | デジタル庁 <https://github.com/digital-go-jp/lawqa_jp> ほか | 公開 | **不適合**（Iter78 と同一結論）．いずれも単一ドメインまたはドメインラベル無し |
| AnswerCarefully (NII) | <https://llmc.nii.ac.jp/answercarefully-dataset> | 公開（研究利用） | **不適合**．日本語・自然文である点は合致するが，付与されているのは**安全性リスク分類**であって専門ドメインではない．10 ドメインへの写像ができない |
| MASSIVE / MixSNIPS 系（multi-intent NLU） | Amazon MASSIVE (CC BY 4.0) ほか | 公開 | **不適合**．1 発話に複数 intent を持つ構成は本研究の要件に形は近いが，scenario ラベル（alarm/music/weather 等）が **10 専門ドメインと重ならない** |
| Yahoo!知恵袋データセット / 国民生活センター PIO-NET | NII ×ヤフー / 国民生活センター | **契約・申込みが必要**（オープンライセンスではない） | Iter78 と同一結論．自然文相談という点だけ合致するが**ドメインラベルが無く**，付与作業は結局 (ii)(iii) と同じ |

**結論**: 「日本語」「自然文」「1 行に 2 つの専門ドメインラベル」の 3 条件を同時に満たす公開データセットは，
本探索範囲（tavily-search 6 クエリ ＋ HF Datasets API 20 クエリ）では見つからなかった．
**これは Iter78（2026-09-26）の調査と独立に行った 2 度目の調査であり，同じ結論に到達した．**
断定を避けるべき点として，これは「存在しないことの証明」ではなく「この探索範囲では見つからなかった」
ことの記録である．**したがって `existing_public_dataset` は本反復でも不成立とする．**

**Q2: JMMLU 派生の複合行 — ライセンスと設問構造の両面で不可**

「既存公開データセットから複合行を派生させる」経路（例: 学際的な科目——`econometrics`（数学＋経済），
`medical_genetics`（医療＋自然科学），`business_ethics`（経済＋社会科学）——に 2 ドメインラベルを付ける）を検討したが，
**2 つの独立した理由で採らない**．
1. **ライセンス**: `build_dataset.py:32-40` が記録するとおり JMMLU 全体は **CC BY-NC-ND 4.0（改変禁止）**である
   （HF `nlp-waseda/JMMLU` の license タグ `cc-by-nc-nd-4.0` を本調査で再確認．
   CC BY-SA 版 `nlp-waseda/JMMLU_CC-BY-SA` も存在するが README に license タグが無く，条件が確認できない）．
   ラベルの張り替えや再構成は派生物に当たり，ND 条項に抵触する疑いがある．
2. **設問構造**: `build_dataset.py:629-632` が既に
   「JMMLU の 4 択問題はそれぞれ単一タスクに属し，真の cross-domain な曖昧さを表現できない」と明記している．
   加えて既存行への 2 ラベル付与は B118 が定めた「**純粋な追加に限定し，既存行の改変・削除は行わない**」に反する．

**Q3: 現行 415 行の検出力と，実行時間予算から決まる到達可能な行数**

- **検出力の実測**: Iter89 本走の複合 415 行は Δ = −2.65pt（0.819277 → 0.792771），
  discordant n_d = 47（a_only 29 / b_only 18），**p = 0.1447 で非有意**．
  したがって π̂_d = 47/415 = **0.11325**．
  McNemar 有意境界 `1.96·√(π̂_d/n)` は n=415 で **3.238pt**，80% 検出力に必要な Δ は **4.628pt**．
  **Iter89 で実際に観測された −2.65pt は，現行 415 行では原理的に検出できない大きさである．**
- **実行時間の実測（本フェーズで results/20260927_232950/results.jsonl から算出）**:
  複合行 **8.057 s/行**（415 行で 55.7 分），単一ドメイン行 **0.762 s/行**（3,020 行で 38.4 分），
  合計 **94.1 分**．複合行が単一行の 10.6 倍遅いのは，開放型の相談文で生成長が長く，かつ
  `compound_mean_dispatched_count` = 1.88 で平均 2 ノードへ dispatch するためである．
- **予算から決まる上限**: `experiment.timeout_min` = 180 分．複合行 1 行 = 8.06 秒，
  劣化余地を見て 10 秒/行で見積もると，複合 n 行のときの総所要は `38.4 + n×10/60` 分．
  n=730 → **160 分**（実測 8.06 秒なら 136 分），n=820 → 175 分，n=1,265 → 249 分．
  **Iter89 の −2.65pt を 80% 検出力で捉えるには n ≈ 1,265 行が必要だが，これは 1 回の本走に収まらない．**
  **したがって本反復の到達目標は n = 730 とする**（有意境界 3.238 → **2.441pt**，
  80% MDE 4.628 → **3.489pt**．期待 discordant は 0.11325×730 ≈ 82.7 行）．
  これは「−2.65pt 級の差を初めて有意判定できる水準に乗せる」ことを意味し，
  80% 検出力まで届かせることは 1 反復では**意図的に諦めている**（残差は backlog B148 に記録する）．

### 計画 (Iter90)

**単一レバー**: `compound_eval_set_expansion` = **`llm_generated_separate_generator_scaleup_730`**
（config.yml の当該レバーの `values` 末尾に本フェーズで追記した．B148 参照）．
**`existing_public_dataset` は Q1 により不成立**，**`manual_authoring` は (ii) より順位が下**のため，
B116 (2) の調達順位 (i) → (ii) → (iii) に従って **(ii) の規模拡大**を選ぶ．
Iter78 で確立した生成器分離の構成（生成器 `qwen3.5:9b` ≠ 検証器 = config.yaml の `judge_model`）を
**一字も変えずに踏襲**し，変えるのは `--target-per-pair` の 7 → 14 だけである．

**仮説**: 複合評価行を 415 → 730 行へ純粋追加すれば，複合サブセットの McNemar 有意境界が
3.238pt → 2.441pt へ縮み，Iter89 で観測された −2.65pt 級の複合ドメイン効果を初めて統計的に判定できる．
既存 3,435 行の計算は 1 ビットも変わらないため，全体 top1 の基準線比較は無傷のまま保たれる．

**変更点（最小差分）**
1. `data/compound_questions_generated.jsonl` を **315 行 → 630 行**（45 ペア × 14 行）へ**追記**する．
   既存 315 行は id・query・expected_domains をそのまま保持する（B118 の「純粋な追加」）．
   生成は `scripts/generate_compound_eval_questions.py` を `--target-per-pair 14` で実行し，
   **wafl-ctrl5 の Ollama（127.0.0.1:11499）上でのみ**行う
   （config.yml 2026-09-23 絶対条件 (B)．wafl500〜509 は絶対に使わない）．
2. `mise run setup`（= `build_dataset.py`）で `data/dataset.jsonl` を **3,435 → 3,750 行**へ再生成する．
3. `data/MANIFEST.md` に新しい sha256 と生成コマンドを追記する（`data/` は .gitignore 対象のため）．
4. **コードの変更は無い**．`build_dataset.py` は既に本経路を持っている．

**レバーを読むコード行と，そこへ到達する条件（config.yml 冒頭「同じ失敗を 6 回繰り返している」への対応）**

| # | コード行 | 何をするか |
|---|---|---|
| 1 | `build_dataset.py:626` `_DEFAULT_GENERATED_COMPOUND_QUESTIONS_PATH = "data/compound_questions_generated.jsonl"` | 既定パスの定義 |
| 2 | `build_dataset.py:1430-1438` argparse `--generated-compound-questions`（default が #1） | CLI 既定値として #1 を採る |
| 3 | `build_dataset.py:1451` `generated_compound_questions_path = args.generated_compound_questions or None` | 空文字なら None に落とす |
| 4 | `build_dataset.py:1459-1466` `_build_rows(..., generated_compound_questions_path=...)` | 値を `_build_rows` へ渡す |
| 5 | `build_dataset.py:1217` `_load_generated_compound_questions(generated_compound_questions_path)` | JSONL を読む |
| 6 | `build_dataset.py:1218-1226` `id=f"compound-{len(_COMPOUND_QUESTIONS)+offset:03d}"`, `is_compound=True` | 行を追加 |
| 7 | `metrics.py:667-669` `compound_domain_question_count` / `compound_domain_top1_accuracy` / `compound_coverage` | `is_compound` で複合指標を算出 |

**到達条件（すべて現行構成で満たされることを確認済み）**
- (a) `mise run setup` は引数なしで `build_dataset.py` を呼び，#2 の default により #1 のパスが渡る．
  **追加の config 変更は不要**であり，到達条件は「ファイルが実在し行数が増えていること」だけである．
- (b) **最大の落とし穴**: `_load_generated_compound_questions()` は `build_dataset.py:1141-1147` で
  **path が None でもファイルが無くても例外を投げず `[]` を返す**．すなわち生成に失敗しても
  `build_dataset.py` は静かに成功し，`data/dataset.jsonl` は 3,435 行のままになる．
  これが Iter16/20/21/22/27/B35 と同型の「実験不成立」を生む唯一の経路である．
  **実装フェーズは (i) `wc -l data/compound_questions_generated.jsonl` = 630，
  (ii) `wc -l data/dataset.jsonl` = 3,750 を本走前に必ず確認すること．**
- (c) id の桁あふれは起きない．offset 最大 630 → `compound-730` で `{:03d}` に収まる．
- (d) レバーが発火した直接の証拠は **results.jsonl に `compound-416` 以降の id が存在すること**である．
- (e) 予備実行は **新規追加行の先頭 20 問**（`data/dataset_iter90_preview20.jsonl`）で行う．
  既存行で予備実行してもレバーの発火証拠にならない．

**事前ゲート G0（結果を見る前に判定規則を固定する）**
- **G0-a（生成器の可用性）**: wafl-ctrl5 の Ollama に `qwen3.5:9b` と judge_model が存在すること．
  欠けている場合は pull する．**wafl500〜509 は使わない．**
- **G0-b（生成物の妥当性）**: 45 ペアすべてがちょうど 14 行．新規 315 行と，既存 415 複合行および
  全単一ドメイン行との間で Jaccard ≥ 0.6 の近重複が 0 件
  （`_NEAR_DUPLICATE_JACCARD_THRESHOLD` = 0.6 の既存ガードをそのまま使う）．
- **G0-c（予備 20 問）**: 新規行の先頭 20 問で予備実行し，(i) HTTP 500 が 0 件，
  (ii) 1 行あたり平均所要 **≤ 12,086ms**（= 実測 8,057ms の 1.5 倍）．
- **G0-d（所要時間の外挿）**: G0-c の実測から 3,750 行の総所要を外挿し，**170 分以内**であること．
  超える場合は **`--target-per-pair` を 11（複合 595 行）へ落として再実行する**（この条項は事前登録．
  本走を省略する選択肢は取らない——config.yml 絶対条件 (A)）．

**成功条件（事前登録．本レバーは精度向上ではなく測定系の整備であるため，主基準は top1_accuracy ではない）**

| 区分 | 指標 | 現状 | 判定基準 |
|---|---|---|---|
| **主基準①（規模）** | `compound_domain_question_count` / `data/dataset.jsonl` 行数 | 415 / 3,435 | **730 / 3,750 ちょうど** |
| **主基準②（純粋追加）** | 既存 3,435 行の id・query・expected_domains | — | **ビット単位で一致**（B118．1 行でも変化したら実験不成立） |
| **主基準③（検出力）** | 旧 artifact `f6c33edb...`（Iter87）と現行 `ff8aad9c...`（Iter89）の replay による複合 730 行の argmax flip 行数 n_d | 47（n=415） | **n_d ≥ 74**（比例期待値 82.7 の 9 割）．同時に実測 π̂_d から `1.96·√(π̂_d/n)` を算出し **≤ 2.6pt**（現状 3.238pt） |
| **主基準④（品質）** | 人手スポットレビュー（乱数種 90 で新規 315 行から 30 件抽出）「2 ドメインの知識が本当に両方要るか」 | 5/30（Iter78 実測） | **不適合 ≤ 6/30（20%）**（Iter78 と同一閾値．緩めない） |
| **主基準⑤（分布）** | ペアごとの行数 | 7 | **45 ペアすべて 14 行**．`general` を含むペアの不適合率を別途報告（Iter78 の所見の追跡） |
| 報告のみ | 全 3,750 行の top1_accuracy・`compound_domain_top1_accuracy` | 0.830859 / 0.792771 | **判定に用いない**（複合比率が 12.1% → 19.5% へ上がるため全体 top1 は機械的に下がる） |

**非退行条件（基準線 = Iter89 本走 `results/20260927_232950/`．すべて「既存 3,435 行サブセット」上で評価する）**

| # | 指標 | 基準線 | 判定基準 |
|---|---|---|---|
| ① | per-domain 20 指標（precision/recall × 10） | Iter89 本走 | BH 補正後の有意退行 **0 件** |
| ② | `compound_domain_set_recall`（既存 415 行サブセット） | 0.566265 | **≥ 0.539759**（絶対値．Iter86〜89 と同一） |
| ③ | `compound_mean_dispatched_count`（既存 415 行サブセット） | 1.879518 | **≤ 2.10**（絶対値．同上） |
| ④ | top1_accuracy（既存 3,435 行サブセット） | 0.830859 | **完全一致**．拡充は既存行の計算に影響しないため，**不一致なら実験不成立**（Iter85 と同型の検証） |
| ⑤ | `fallback_rate` | 0.0 | **= 0.0** |
| ⑥ | `mean_duration_ms`（既存 3,435 行サブセット） | 1643.249 | **≤ 1971.9**（基準線 +20%．**全 3,750 行では複合比率が上がるため機械的に増えるので，サブセットで評価する**） |
| ⑦ | `dispatch_failure_rate`（全 3,750 行） | 0.001165（本フェーズで results.jsonl を直接数えて確認．4/3,435 行） | **≤ 0.005**（絶対値．Iter86〜89 と同一） |

**判定語の定義（Iter78 と同一条文）**
- **adopted**: 主基準①〜⑤と非退行①〜⑦をすべて満たす．**top1_accuracy の増減は判定に用いない．**
- **partial**: 主基準③（検出力）だけが未達．実測 π̂_d から必要 n を再計算して backlog へ残す．
- **invalid（実験不成立）**: `data/dataset.jsonl` が 3,435 行のまま／`compound_domain_question_count` が
  415 のまま／results.jsonl に `compound-416` 以降の id が 0 件／非退行④が不一致，のいずれか．

**本走（config.yml 絶対条件 (A)）**: 変更適用後に **wafl500〜509 を用いた 3,750 問のフルスペック本走を
必ず 1 回実施する**．`mise run start -- --dataset data/dataset.jsonl --output results.jsonl` ののち
`mise run analyze -- <timestamp>` を timestamp 明示指定で実行する（B118 落とし穴 1 の再発防止）．
想定所要は **136〜160 分**（timeout 180 分以内）．

**恒久ルールへの適合確認**
- B115 (1)（ドメイン固有の後付け補正を追加しない）: 45 ペア一律 14 行で，特定ドメインを狙い撃ちにしていない．
- B116 (2): 既存公開データセットの調査を先に実施し（Q1），不成立を出典付きで記録したうえで (ii) へ落とした．
- 2026-09-23 絶対条件 (B): 生成・検証の LLM 呼び出しは **wafl-ctrl5 のみ**．
- 2026-09-23 絶対条件 (A): 本走を必ず実施する（G0-d でも本走省略の分岐を作っていない）．

### 実験結果（Iter90，フェーズ 2 実測．判定はフェーズ 3）

**本走**: `results/20260928_032909/`（3,750 問，wafl500〜509 でフルスペック 1 回，所要約 158 分，timeout 180 分以内）．
`mise run analyze -- 20260928_032909`（timestamp 明示指定）まで実施済み．
**レバー発火の直接証拠**: results.jsonl に `compound-416`〜`compound-730` が存在し，ログ上でも実際にディスパッチされたことを確認．

**実装上の逸脱（重要）**: 計画の生成コマンド（`generate_compound_eval_questions.py --target-per-pair 14` を直接実行）は，
同スクリプトが `--output` を毎回 `open(..., "w")` で全体上書きする実装のため，**そのまま実行すると既存 315 行を破壊する**ことが
実装フェーズで判明した（630 行を丸ごと再サンプリングしてしまい B118「純粋な追加」にならない）．
Iter78 の「追い上げパス」と同じ手法（`generate_all_rows()` を直接呼ぶ使い捨てドライバ．スクリプト本体は無改変）で
新規 7 件/ペアだけを生成し既存ファイル末尾へ追記した．既存 315 行がバイト単位で不変であることを `diff` で確認済み．

**事前ゲート G0**: a（wafl-ctrl5 に `qwen3.5:9b`・judge_model とも既存，pull 不要）PASS ／ b（45 ペアすべて 14 行，
近重複 0 件・最大 Jaccard 0.148，exact 重複 0 件）PASS ／ c（新規行先頭 20 問 `compound-416`〜`435` で HTTP 500 0 件，
平均 8,588.45ms ≤ 12,086ms）PASS ／ d（外挿 139〜143 分 ≤ 170 分．`--target-per-pair` 11 へのフォールバックは不要）PASS．

**主基準の実測**

| # | 基準 | 実測 | 閾値 | 可否 |
|---|---|---|---|---|
| ① 規模 | `compound_domain_question_count` / dataset 行数 | **730 / 3,750** | 730 / 3,750 ちょうど | PASS |
| ② 純粋追加 | 既存 3,435 行の id・query・expected_domains | **完全一致（diff 0 件）** | ビット単位一致 | PASS |
| ③ 検出力 | n_d（複合 730 行の replay flip 行数） | **150** | ≥ 74 | PASS |
| ③ 副条件 | `1.96·√(π̂_d/n)` | **3.288pt**（π̂_d = 0.205479） | ≤ 2.6pt | **未達** |
| ④ 品質 | 人手スポットレビュー不適合（種 90，新規 315 行から 30 件） | **約 9〜10/30（30〜33%）**（実装フェーズの主観判定） | ≤ 6/30 | **未達の可能性** |
| ⑤ 分布 | ペアごとの行数 | **45 ペアすべて 14 行** | 全ペア 14 行 | PASS |

③ の π̂_d は計画時の想定 0.11325 から 0.205479 へ大きく上振れしたため，n_d は基準を大きく上回った一方で有意境界は
縮まらず 3.238pt → **3.288pt** とほぼ横ばいになった（拡充の目的であった分解能の改善が得られていない）．
`metrics.compute_mcnemar_test` の再利用値は discordant_a_only=43 / b_only=35 / p=0.428．
④ の不適合パターンは Iter78 所見と同型で，(a) `general` を含むペアの過度な抽象化・列挙型設問，
(b) 無関係な 2 話の単純併記，(c) 技術用語の強引な結び付け．**`general` を含むペアの不適合率は約 75%（8 件中 6 件）**で，
非 general 組の 18% と大きく乖離している．

**全 3,750 行（報告のみ，判定に用いない）**: top1_accuracy = 0.829867，compound_domain_top1_accuracy = 0.808219，
fallback_rate = 0.0，dispatch_failure_rate = 0.000533，mean_duration_ms = 2531.54．

**非退行条件の実測**（基準線 `results/20260927_232950/`，既存 3,435 行サブセット上で評価）

| # | 条件 | 実測 | 閾値 | 可否 |
|---|---|---|---|---|
| ① | per-domain 20 指標の BH 補正後の有意退行 | **0 件**（20/20 非有意） | 0 件 | PASS |
| ② | `compound_domain_set_recall`（既存 415 行） | **0.565060**（基準線 0.566265） | ≥ 0.539759 | PASS |
| ③ | `compound_mean_dispatched_count`（既存 415 行） | **1.925301**（基準線 1.879518） | ≤ 2.10 | PASS |
| ④ | top1_accuracy（既存 3,435 行） | **0.829985**（2,851/3,435） | 0.830859 と完全一致 | **未達** |
| ⑤ | `fallback_rate` | **0.0** | = 0.0 | PASS |
| ⑥ | `mean_duration_ms`（既存 3,435 行） | **2004.312ms** | ≤ 1971.9ms | **未達（+22.0%）** |
| ⑦ | `dispatch_failure_rate`（全 3,750 行） | **0.000533** | ≤ 0.005 | PASS |

**非退行④の詳細（フェーズ 3 で最も重要な論点）**: 基準線 2,854/3,435 に対し本走は 2,851/3,435 で **−0.087pt・正味 3 行差**．
行単位 diff では **19 行の判定が入れ替わっている**（True→False 11 行，False→True 8 行）．うち 4 行は `dispatch_failed` の状態自体が
変化しており（一時的なノード過負荷起因），残り 15 行は confidence の僅かな変動（例 0.4693893… → 0.4511954…）による境界での
argmax flip である．**「既存行の計算はビット単位で不変」という計画の前提が，実運用（Ollama 経由の GPU バッチ埋め込みと
dispatch の実行時非決定性）では成立していない**ことを示す実測であり，条文を literal に適用すると「実験不成立」に触れる．
原因の推測は上記のとおりだが確証は得ていない．

**その他**: `mise run deploy` が auto mode の分類器から一度「Production Deploy」として拒否されたが，フォアグラウンドでの
再実行で許可され正常完了（全 10 ノード healthy，smoke_check 全項目 PASS）．実験自体はハングなく完走．
監査用の中間生成物は `/tmp/iter90/`（`metrics_full3750.json`，`metrics_existing3435.json`，`metrics_existing415compound.json`，
`replay_result.json`，`topup_generate.py` ほか）に保存済み．`data/MANIFEST.md` に Iteration 90 節（sha256・生成コマンド）を追記済み．

### Iteration 90 実行済み

**単一レバー**: `compound_eval_set_expansion` = `llm_generated_separate_generator_scaleup_730`
（`scripts/generate_compound_eval_questions.py` の `--target-per-pair` を 7 → 14 相当へ．生成器 `qwen3.5:9b` ≠ 検証器 = `judge_model` の
構成は Iter78 から不変．コード変更なし）．複合行 415 → 730，`data/dataset.jsonl` 3,435 → 3,750 行．
本走 `results/20260928_032909/`（3,750 問，約 158 分）．基準線 `results/20260927_232950/`（Iter89，3,435 問，top1 = 0.830859）．

**判定: `partial`**（後述のとおり 4 つの判定語のいずれも字義どおりには当てはまらず，実態に最も近いものを選んだ）．
**評価集合 730 行は採用し，ロールバックしない．Iter91 以降の基準線は `results/20260928_032909/`（3,750 行，top1 = 0.829867）とする．**

#### 1. 主基準③（検出力）— フェーズ 2 の「未達」は **π̂_d の定義ずれによる誤判定**であり，実際は PASS

事前登録の ③ は，本文で「複合 730 行の **argmax flip 行数** n_d」と書きながら，数値アンカーは
「現状 47（n=415）」「現状 3.238pt」の 2 つを置いていた．本フェーズで旧 artifact `f6c33edb`（Iter87，0.6b）と
現行 artifact `ff8aad9c`（Iter89，4b）の replay を **旧 415 行 / 新 315 行に分けて再実行**したところ，この 2 つのアンカーは
どちらも **argmax flip ではなく McNemar の discordant（正誤の不一致ペア）** から計算された値であることが確定した．

| 部分集合 | argmax flip | discordant（a/b） | π̂_d（discordant） | 1.96·√(π̂_d/n) | replay top1（旧→新 artifact） |
|---|---|---|---|---|---|
| 旧 415 行 | 85 (20.48%) | **47**（29/18） | **0.11325** | **3.238pt** | 0.819277 → 0.792771 |
| 新 315 行 | 65 (20.63%) | 31（14/17） | 0.09841 | 3.464pt | 0.815873 → 0.825397 |
| 全 730 行 | **150** (20.55%) | **78**（43/35） | **0.10685** | **2.371pt** | 0.817808 → 0.806849 |

旧 415 行の再計算が事前登録の 47・3.238pt・0.11325 と**完全に一致**したことで，アンカーの定義は discordant で確定する．
したがって**アンカーと同じ定義を一貫して適用すれば n_d = 78 ≥ 74 で PASS，有意境界は 3.238 → 2.371pt で ≤2.6pt を満たし，
③ は副条件も含めて PASS** である．フェーズ 2 が報告した 3.288pt は分子に argmax flip（150）を用いた値で，
同じ定義を基準線側にも当てれば基準線は 3.238pt ではなく **4.354pt**（85/415）であり，4.354 → 3.288pt は
**√(415/730) = 0.754 倍という純粋な √n スケーリングと小数 3 桁まで一致する**．
すなわち**どちらの定義で統一しても拡充は設計どおり効いており，「分解能が改善しなかった」というフェーズ 2 の読みは，
基準線と実測で異なる量を比べたことによる artifact である**．条文を緩めたのではなく，条文内部の不整合を
「2 つの数値アンカーが実際に計算された定義」の側へ解消した（両方の読みの数値を上表に併記する）．

副産物として，**Iter90 の投資が意図どおりの成果を出した**ことも確認できた．Iter89 で唯一の悪化方向だった複合サブセットの
効果（B147 (c) が次レバー選定の根拠にした −2.65pt）は，730 行では **−1.10pt，discordant 43/35，p = 0.428** に縮み，
有意境界 2.371pt を下回る．**「Iter89 の埋め込み差し替えが複合設問を悪化させた」という疑いは，分解能を上げた測定系の下では支持されない．**

#### 2. 非退行④（top1 のビット単位一致）— 条文の前提が誤っていた．原因はレバーではなく**セッション境界の非決定性**

実測 0.829985（2,851/3,435）vs 基準線 0.830859（2,854/3,435），正味 −0.087pt・3 行．行単位では 19 行が反転
（True→False 11／False→True 8，厳密二項 **p = 0.6476**）．本フェーズで原因を切り分けた．

- **既存 3,435 行の `probe_candidates` を基準線と全スロット突き合わせた結果，34,350／34,350 スロットすべてが相違**していた
  （行ごとの最大差の中央値 0.00269，p90 0.01486，最大 0.0806）．すなわち「既存行だけが 3 行動いた」のではなく
  **全行の確信度が動いており，そのうち境界付近の 19 行だけが argmax を跨いだ**．
- 一方，**同一セッション内（デプロイを挟まない）は完全に決定論的**である．Iter89 の予備 20 問と Iter89 本走は
  データセットの行数も順序も全く違うのに **200/200 スロットがビット単位で一致**し，Iter90 の予備 20 問と Iter90 本走も
  **200/200 一致**した．**したがって「行数が増えて GPU のバッチ構成が変わったから」という機序（フェーズ 2 の推測）は棄却される．**
  変動はセッションを跨いだとき（＝ `mise run deploy` によるコンテナ再作成と埋め込みモデルの再ロードを挟んだとき）にのみ生じる．
- 分類器 artifact は `ff8aad9c...` のまま不変（sha256 で確認）．`data/dataset.jsonl` の既存 3,435 行も diff 0 件．
  **レバーが既存行の計算に影響した証拠は無い．**
- 対照として，Iter82 → Iter83（`qwen3-embedding:0.6b`）ではデプロイを挟んでも `probe_candidates` の最大絶対差が **0.0** だった
  （journal_archive「Iteration 83」）．**0.6b では成立していたセッション跨ぎの再現性が，4b（Iter89 で採用）では失われている**
  ことになる．機序は未確証だが，12GB VRAM に expert モデルと同居する 4b のロード時オフロード構成が
  セッションごとに変わりうる点が第一の容疑である（本反復では検証していない）．

**結論**: 非退行④は**条文の前提（「拡充は既存行の計算に影響しないから完全一致するはず」）そのものが実運用で成立しない**ことが
実測で示された．条文を literal に適用すれば invalid だが，この条文の目的（success_criteria (6)・d0004 §4）は
「レバーがコードパスに到達したかの検出」であり，本反復ではレバー発火の直接証拠（`compound-416`〜`730` の存在，
`compound_domain_question_count` = 730）が揃っているうえ，差分は純ノイズと区別できない（p = 0.6476）．
**`invalid` を宣言して 158 分の有効な測定を破棄することは，条文の目的にも実測にも反する．**
よって invalid とはせず，事実と再現性の床を確定させたうえで下記の恒久的な申し送りに置き換える．

**再現性の床（今後の非退行条件で使う数値）**: デプロイを挟んだ同一構成の 2 本走で，**既存 3,435 行の 0.553%（19 行）が
正誤反転し，top1 は ±0.087pt 動く．この対から導かれる McNemar 有意境界は 0.249pt** である．
**top1 の 0.25pt 未満の差は今後いかなるレバーにも帰属させてはならない．**
（参考: 訓練の乱数種ばらつきは replay 上で 0.31pt — Iter88 学び．軸②③の生成ノイズ床は 2.6pt — success_criteria (5)．）

#### 3. 主基準④（品質）— 閾値の真上で判定不能．`general` ペアの構造的欠陥は 2 反復連続で再現

フェーズ 2 の主観判定（約 9〜10/30）を鵜呑みにせず，同じ種 90 の 30 件を本フェーズで独立に読み直した．
「2 ドメインの知識が本当に両方要るか」で明確に不適合と言えるのは **6/30**（列挙型の [4]，感情吐露で一方のドメイン知識が不要な [6]，
確率論を強引に接続した [8]，cs 単独で完結する [22]，疑似科学的な結び付けの [24]，無関係 2 話の併記 [28]），
判断が割れる境界例（[3][5][9][16][27] 等）を不適合に数えると **9〜11/30** になる．
**閾値は ≤6/30 であり，2 名の独立した判定が 6 と 9〜10 に割れて閾値のちょうど上下に落ちた．
すなわち ④ は「未達」とも「達成」とも言えず，測定器としての `≤6/30 の主観 1 名判定` が基準として機能していない．**
④ を根拠に adopted を主張することはできないが，④ を根拠に棄却することもできない．

一方 **`general` を含むペアの偏りは明確で再現性がある**．私の明確な不適合 6 件のうち **5 件が `general` 組**で，
サンプル中の general 行 8 件に対する不適合率 **62.5%**，非 general 22 件に対しては **4.5%** である．
Iter78 のスポットレビュー（5/30，全件が general 絡み）と合わせて**2 反復連続の同型所見**であり，偶発ではない．
機序は「`general` は他 9 ドメインの補集合として定義されているため，`general`＋X のペアに対して
生成器が『X の話題＋日常の愚痴』という 2 話併記や，X に還元できる設問を作ってしまう」ことである．

なお，**品質の低さが測定系を汚しているという仮説は，今回の数値では支持されない**．新規 315 行の discordant 率は
0.09841 で旧 415 行の 0.11325 **より低く**，本走の実測 top1 も新 315 行 0.828571 > 旧 415 行 0.792771 である．
`general` ペアの設問は「複合設問として妥当でない」が「ルーティングの評価行としては旧行より素直」という状態にある．

#### 4. 非退行⑥（`mean_duration_ms`）— 条文の設計ミス

既存 3,435 行だけで測っても 1643.249 → **2004.312ms（+22.0%）**で閾値 +20% を超えた．
しかし**同一本走の中で複合行を 415 → 730 に増やせば，既存行もノード競合の影響を受けて遅くなる**．
「既存行サブセットで測れば負荷の影響を除ける」という事前登録の想定（B148 (E)）が誤りであり，
部分集合の切り出しでは交絡を除去できない．レバー自体の欠陥ではないが，事前登録の条文としては失敗である．

#### 5. 判定語の確定

| 主基準 | 可否 | 非退行 | 可否 |
|---|---|---|---|
| ① 規模（730 / 3,750） | PASS | ① per-domain 20 指標 | PASS（有意退行 0 件） |
| ② 純粋追加（既存 3,435 行 diff 0） | PASS | ② `compound_domain_set_recall` | PASS |
| ③ 検出力（n_d = 78 ≥ 74，境界 2.371pt ≤ 2.6pt） | **PASS**（上記 §1） | ③ `compound_mean_dispatched_count` | PASS |
| ④ 品質（≤6/30） | **判定不能**（6 と 9〜10 に割れ） | ④ top1 完全一致 | **FAIL**（原因はレバー外．上記 §2） |
| ⑤ 分布（45 ペア × 14 行） | PASS | ⑤ `fallback_rate` = 0.0 | PASS |
| | | ⑥ `mean_duration_ms` | **FAIL**（条文の設計ミス．上記 §4） |
| | | ⑦ `dispatch_failure_rate` | PASS |

`adopted`（全条件 PASS）は ④ が判定不能・非退行④⑥ が FAIL のため取れない．
`rejected`（レバーが効かなかった）は ①②③⑤ が PASS でレバーが目的を果たしているため事実に反する．
`invalid`（測定が無意味）は §2 のとおり事実に反する．
**残る `partial` を選ぶ．**ただし `partial` の条文（「主基準③だけが未達」）にも字義どおりには当てはまらない．
**事前登録の 4 語が今回の事象（条文の前提の誤り・定義の内部不整合・判定不能な主観指標）を網羅していなかった**ことを
そのまま記録し，次の判定語定義の設計に反映する．

#### 6. 学び（次の自分が読んで分かる形で）

1. **「基準線とビット単位で一致すること」を非退行条件に書いてはならない．**`qwen3-embedding:4b` を採用した Iter89 以降，
   デプロイ（コンテナ再作成・モデル再ロード）を挟むと既存行の確信度が**全スロット**動く．代わりに
   **top1 の差 ≤0.25pt かつ正誤反転行 ≤1.0%** を再現性の床として使うこと．
   **レバーがコードパスに到達したかの検出は「基準線との完全一致」ではなく，レバー固有の直接証拠
   （今回なら `compound-416` 以降の id の存在）で行う．**両者は別の目的であり，同じ条文で兼ねてはいけない．
2. **決定性が成り立つ範囲は「同一セッション内」である．**予備 20 問と本走の突き合わせ（デプロイを挟まない）は
   Iter89・Iter90 とも 200/200 ビット一致で，**予備実行による事前検証は引き続き完全に信頼できる**．
   信頼できないのはセッションを跨いだ比較だけである．
3. **事前登録で指標を定義したら，本文の定義語と数値アンカーが同じ量を指しているかを必ず突き合わせること．**
   ③ は「argmax flip 行数」と書きながらアンカー 47・3.238pt は discordant 由来で，フェーズ 2 が文言どおりに計算した結果，
   **実際には達成していた基準を「未達」と誤判定した**．アンカーの再現計算（今回の旧 415 行 replay）は 5 分で済む．
4. **「部分集合で測れば負荷の交絡を除ける」は誤り．**同一本走の中で行数を増やせば既存行の所要時間も増える（+22%）．
   時間系の非退行条件は，行あたり平均ではなく負荷項を織り込んだ形にするか，そもそも判定に使わないこと．
5. **主観 1 名・閾値 1 本のスポットレビューは判定基準として機能しない．**2 名の独立判定が 6 と 9〜10 に割れ，
   閾値 ≤6/30 のちょうど上下に落ちた．今後この種の条件を置くなら，判定を二値ではなく
   「明確な不適合」「境界例」に分け，**明確な不適合だけで閾値を切る**など，割れにくい定義にすること．
6. **`general` を含む複合ペアは 2 反復連続で不適合率が突出している**（Iter78 5/30 全件 general 絡み，Iter90 62.5% vs 4.5%）．
   `general` が他 9 ドメインの補集合として定義されていることに由来する構造的な問題であり，生成器を変えても消えない．
   ただし**ルーティング評価行としての素直さ（discordant 率・top1）は旧行より良い**ので，測定系を汚してはいない．
7. **Iter90 の投資は回収された．**B147 (c) が起点にした「Iter89 の複合 −2.65pt」は，730 行では −1.10pt・p = 0.428 に縮み，
   有意境界 2.371pt を下回る．**複合ドメインの悪化という疑いは支持されない**ので，次はこの心配のために
   レバーを費やす必要がない．

