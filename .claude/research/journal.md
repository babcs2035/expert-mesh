## Iteration 111: 写像を固定した 6 系統の外で代替のレバーを探す

### Iteration 111 調査・計画（2026-10-05）

- 申し送り: B212（人間の回答は無いので C3 の扱い）．levers は使い切り．打ち止めの 6 系統（決定層 M6，訓練行の除去 Iter108，訓練行の追加 B161，kNN 類似度の前処理 Iter106，事前分布の補正 Iter109，別の入力の特徴 Iter110）の外で，実行可能な単一レバーを探す．B206（写像の見直し）と B207（評価のクエリから π_t を推定すること）に当たる変更は始めない．
- 過去の試行との照合（grep．対象は `journal_archive.md`，`backlog.md`，`config.yml`）:
  - 記録があったもの: SetFit（archive 51 件．Iter39〜40 の `embedding_adaptation` で `setfit_education_finetune` は rejected．argmax の反転率が大きく，単一レバー原則と両立しないとした．`config.yml:422-456`，`1968`），instruction prefix（Iter81 で本番に入った．`config.yml:2222-2241`），埋め込みモデルの交換と融合（Iter79，Iter80，Iter99〜100．`config.yml:1995-2009`），soft label distillation（Iter54 の候補．archive 23111 行目以降），LLM による集約 `llm_judge`（Iter48 で −16.8pt．backlog B71），階層分類と CHiLS（Iter92 の文献として触れただけ．archive 4398，4407 行目），cross-encoder（Iter59 の文献例．archive 20569 行目）．
  - 記録が無かったもの: test-time augmentation（0 件），label smoothing（0 件），mixup（0 件），reranker（0 件），LLM による候補の並べ直し（`LLM.*rerank` 0 件）．zero-shot は archive に 1 件だけで，分類の手段としては試していない．
- 調査の記録（検索ごとに追記する）:
  - 検索 1（問い (a): 埋め込みの分類器の上位候補を LLM で選び直すと，取り違えの多いクラスで正解率が上がるか．`tvly search`，advanced，8 件）: 直接に関係するのは「Confusion-Aware Retrieval and Knowledge Injection for ...」（arXiv:2609.01564，2026，著者と題の後半は抜粋に無い，https://arxiv.org/html/2609.01564v1 ）だけである．抜粋は「埋め込みの類似度で上位 K 個のラベル候補を取り，LLM に選ばせる」手法を一般的な方式として挙げる．ほかは RAG の reranker の解説記事と，LLM によるデータ拡張の論文（EDM 2025，MDPI Electronics 13(13):2535）で，本件の問いには当たらない．Medium の記事（Youness Mansar，「Retrieval Augmented Classification」）は kNN と LLM の組み合わせで +9pt と書くが，個人の記事で条件が確かめられないので根拠には使わない．
  - 抽出 1（arXiv:2609.01564 の本文．「From Confusion to Clarity: Confusion-Aware Retrieval and Knowledge Injection for Text Classification」，2026，著者は抽出した本文に無い．`tvly extract` で HTML 版 https://arxiv.org/html/2609.01564v1 を取得した．75,598 文字）．候補 D1（取り違えの多い対に限り，訓練の誤答から作った判別の規則を LLM に与え，上位 2 候補から選び直させる）の根拠になるかを，次の 3 点で確かめた．
    - (i) fine-tune した教師ありの分類器との比較（5.3 節の表 5，付録 A.3 の表 7 と表 9）: 比べている．ModernBERT-base と RoBERTa-base を fine-tune した．きれいで欠けの無い訓練データでは，最も細かい階層の Macro F1 で教師ありの分類器が上回る．Flipkart L3 は 80.0 / 81.7 に対して提案法 76.7，WOS L2 は 72.9 / 80.4 に対して 57.7，LEDGAR は 75.4 / 78.1 に対して 58.9 である（提案法は Qwen3-32B）．提案法が上回るのは，ラベルの雑音（10〜50% を入れ替え），訓練データの不足（10% = 3 行/クラス，25% = 9 行/クラス），テスト時の摂動の条件だけである．50%（18 行/クラス）で両者は並ぶ（69.8 / 70.8 対 69.8）．主表（表 1）の比較の相手は，分類器を固定して推論時の文脈だけを変える方式（zero-shot，few-shot，埋め込みで候補を取る retrieval，MIPROv2，GEPA）である．
    - (ii) データの規模とクラス数（4.1 節の表 2）: LEDGAR は 100 ラベル（訓練 60k，テスト 10k，法律），WOS は 134 ラベル・2 階層（32.8k / 9.5k，学術），Flipkart は 351 ラベル・3 階層（12.3k / 3.8k，EC）である．本件（10 クラス，訓練 2,275 行，約 227 行/クラス）よりクラスがずっと多く，候補の絞り込みが要る条件である．
    - (iii) 短文と多言語: 3 つとも英語である．Limitations の節は「多言語の入力，ずっと大きいタクソノミー，時間的なずれでの釣り合いは試していない」と書く．短文かどうかは本文に明記が無い（Flipkart は商品の掲載文）．
    - 残る誤りの分析（付録 A.13，Flipkart L3 の 737 行）: 20.8% は「同じ概念のラベルが別の親の下にある」タクソノミーの不整合で，著者は「商品の文面だけからは，どの分類器でも解きにくい」と書く．31.2% は本当に意味が重なる対（Western Wear / Fusion Wear など）で，規則は Western→Fusion の誤りを retrieval だけの場合の 112 行から 71 行に減らした．ただしこの減少は LLM の基準線に対する値であり，教師ありの分類器に対する値ではない．
    - D1 にとっての読み: 本件の訓練データはラベルがきれいで（Iter93 で写像との食い違いは 0 行），クラスあたりの行数は表 9 で両者が並んだ 18 行/クラスの 10 倍以上ある．評価集合も固定である．論文の結果からは，この条件では LLM が選ぶ方式が教師ありの分類器を下回ると見込む（推定．本件のいまの分類器は fine-tune した BERT ではなく埋め込みの上の LR と kNN の補間だが，教師ありである点は同じ）．本件の誤りの根である写像の曖昧さは，付録 A.13 の「タクソノミーの不整合」に当たり，論文の手法でも残った種類の誤りである．
- 判断: **要人間判断（実行可能な新レバーを定義できない）**．D1 は levers に追記しない．config.yml と state.json は変えない（B213）．
  - 根拠 1（比較の相手）: arXiv:2609.01564 の主な利得（Macro F1 で最大 +10.0pp）は，LLM が選ぶ基準線（retrieval，zero-shot，few-shot，MIPROv2，GEPA）に対する値である．fine-tune した教師ありの分類器と比べた節（5.3 節の表 5，付録 A.3 の表 7）では，きれいで欠けの無い訓練データの条件で，3 つのデータセットのすべてで教師ありの分類器が上回った（−3.3〜−22.7pt）．
  - 根拠 2（本件の条件）: 提案法が上回ったのは，ラベルの雑音，訓練データの不足（9 行/クラス以下），テスト時の摂動の条件だけで，18 行/クラスで並んだ（付録 A.3 の表 9）．本件はラベルの食い違いが 0 行（Iter93），約 227 行/クラスで，評価集合も固定なので，論文が教師ありの分類器に有利と示した条件に当たる．
  - 根拠 3（誤りの種類）: 本件の誤りの根である写像の曖昧さ（Iter108 の 2a）は，付録 A.13 の「タクソノミーの不整合」（残る誤りの 20.8%）に当たる．著者はこれを「文面だけからは，どの分類器でも解きにくい」としている．
  - 根拠 4（過去の実測）: LLM に選ばせる方向の過去の実測は負である（Iter48 の `llm_judge` は −16.8pt，backlog B71．対象は回答の比較で D1 とは違う）．英語以外での検証は無い（Limitations の節）．
  - Step 0a の条件（LLM 単体の正解率がいまの分類器以上）を置いて安く足切りする道もある．しかし上の根拠 1〜3 から通過の見込みは低い．しかも Step 0a の段階で，LLM の推論の経路（wafl-ctrl5 の GPU，規則の生成，訓練の OOF 2,275 行の再判定）を新しく作る必要がある．過去の CPU だけの Step 0 より費用が大きいので，根拠の無いまま自動では立てない．
- 未確認の点: 教師ありの埋め込みと LR の分類器（fine-tune した BERT ではない）に対して D1 がどうなるかは，論文では検証されていない．日本語の短い設問での LLM 単体の正解率（Step 0a の前提）は実測していない．test-time augmentation，label smoothing，mixup は grep の記録が 0 件だったが，文献は調べていない（label smoothing と mixup は訓練の変更で，B161 や Iter108 の系統に近い見込み）．

## Iteration 110: 取り違えの多いクラス対を文字 n-gram で並べ直す

このブロックの前半（「判断」の行まで）は，rc-researcher の報告をもとにオーケストレータが挿入した（rc-researcher はツールの呼び出しの形式を誤り，記録の前にツールを使えなくなった．起動モードは tenbin）．「文献調査」以降は，再委譲した rc-researcher が書いた．見出しは iteration_name の案に合わせて書き換えた（state.json への反映はオーケストレータが行う）．

### Iteration 110 調査・計画（2026-10-05）

- 調査: tavily-search は実行できなかった（skill の読み込みまでは成功した）．外部の出典は無い．
- 確かめた事実: M6（`journal_archive.md` 2628〜2702 行目）により，同じ埋め込みの上で対ごとに分類器を作っても新しい情報は増えない（OvO と ECOC の誤答行の重なり AGREE 87.6% / 93.9%）．訓練ファイル `data/classifier_train_iter94_dedup.jsonl` は `query` を持つので，語彙特徴は写像を変えずに作れる．
- 過去の試行との照合（オーケストレータが grep で確かめた．キーワード: tf-idf，n-gram，bm25，lexical，語彙特徴，char_wb，bag of words）: 語彙特徴を分類の入力として実装した記録は無い．Iter54 の調査で「education 関連キーワードの TF-IDF」が候補に挙がったが，rc-reflector が argmax の反転率 15〜30% のおそれを指摘し，実装していない（`journal_archive.md:23129`）．Iter59 の BM25 は retrieve-then-rerank の文献例として触れただけである（`journal_archive.md:20432`）．n-gram の重なり率は echo の除去（データの掃除）で使ったもので，分類の特徴ではない．
- 候補 C1（未確定）: `confusable_pair_lexical_rerank`．訓練の OOF の混同行列で決めた対に限り，上位 2 候補を文字 n-gram の TF-IDF の LR で並べ直す．対象の候補は Iter108 の 2a の education → business_economics（japanese_civics）37 行，education → medical 16 行，education → history_culture 16 行，medical → education 11 行．
- Step 0 の案（CPU だけで，本番を変えない．評価のラベルとクエリは 0c まで使わない）: 0a は訓練の外側 5-fold（seed 104）の OOF で語彙特徴の LR と現行の分類器の誤答行の重なりを測り，AGREE が 85% 以上ならクローズする．0b は入れ子 CV で OOF top1 が none を +0.5pt 以上かつ 4/5 fold 以上で上回ること．0c は replay の共通行 3,748 行で Δtop1 が +0.5pt 以上かつ正味 +3 行以上（B200 の床 ±2 行を超える）．McNemar を併記する．
- 未確認: 外部の文献の裏付け．写像が曖昧な対（心理学や哲学など）では語彙も 2 クラスで共有されている見込みがあり，その場合は語彙特徴でも新しい情報は増えない（推定）．
- 判断: 文献の裏付けが無いので，levers には追記していない（B209）．
- 文献調査（rc-researcher の再委譲，tavily-search，検索ごとに追記する）:
  - 検索 1（問い (a)，埋め込みと TF-IDF の併用）: 査読つきの対照実験は見つからなかった．見つかったのは解説記事「How to Combine LLM Embeddings + TF-IDF + Metadata in One Scikit-learn Pipeline」（Iván Palomares Carrascosa，MachineLearningMastery.com，2026，https://machinelearningmastery.com/how-to-combine-llm-embeddings-tf-idf-metadata-in-one-scikit-learn-pipeline ）と，プレプリント「A Hybrid TF–IDF and SBERT Approach for Enhanced Text Classification Performance」（preprints.org 202510.2427，2025，著者は抜粋に無い，https://www.preprints.org/manuscript/202510.2427 ）である．いずれも併用の手順を示すが，効果の大きさは抜粋からは確かめられない．短文や日本語の事例は無い．
  - 検索 2（問い (a) の補足，疎と密の特徴の相補性）: 上のプレプリント（preprints.org 202510.2427）の表 2 では，TF–IDF と SBERT を併用した線形 SVM の accuracy が 0.892 である．ただし本文は ablation の数値を「同種の研究と整合する現実的な改善」と書いており，実測値かどうかを抜粋からは判断できないので，根拠には使わない．「Comparison and Combination of Sentence Embeddings Derived from Different Supervision Signals」（*SEM 2022，著者は抜粋に無い，https://aclanthology.org/2022.starsem-1.12.pdf ）は，STS を語の重なりの比で分けると，SBERT は表層が似た文対の意味の差をよく捉えると報告する．語彙の重なりと埋め込みの判断が食い違う領域があることの傍証にとどまり，分類での利得は示さない．
  - 検索 3（問い (b)，混同行列で決めた対に特化した分類器）: 結果の多くは混同行列の解説だった．関連するのは「Training Highly Multiclass Classifiers」（Gupta, Bengio, Weston，JMLR 15，2014，https://jmlr.org/papers/volume15/gupta14a/gupta14a.pdf ）だけである．一度訓練した分類器から経験的なクラス混同確率の行列を求め，それを使って再訓練する逐次の手順を述べる．混同行列を使って取り違えの多い対に資源を寄せる考え方の前例になるが，同じ特徴の上での再訓練なので，M6 の制約（新しい情報が増えない）は解かない．
  - 検索 4（問い (b) の補足，specialist model）: 「Distilling the Knowledge in a Neural Network」（Hinton, Vinyals, Dean，arXiv:1503.02531，2015，https://arxiv.org/abs/1503.02531 ）は，generalist が取り違えるクラスの集合（generalist の共分散行列の列を K-means でまとめる）ごとに specialist を作り，generalist の上位 n 候補に関わる specialist だけを使って top1 を決める 2 段の手順を示す．C1 の「混同で決めた対に限って上位候補を並べ直す」形の直接の前例である．ただし specialist は generalist の重みから再訓練するので表現そのものが変わる．C1 は表現を変えず，埋め込みとは別の入力（文字 n-gram）を足す点が異なる．Hinton の講演（YouTube「Distilling the Knowledge in a Neural Network - Geoffrey Hinton」）では，generalist が候補を外した場合は specialist でも回復しないと述べている．C1 で言えば，正解が上位 2 候補に入っていない行は直らない．
  - 検索 5（問い (c)，ラベルの定義が重なるクラス）: 「Scaling and Disagreements: Bias, Noise, and Ambiguity」（Uma ら，Frontiers in Artificial Intelligence 2022，PMC9012579，https://pmc.ncbi.nlm.nih.gov/articles/PMC9012579 ）は，カテゴリが互いに排他でなく重なる注釈の方式では，1 つを選ばされた注釈者の選択はほぼ無作為になると報告する（Uma ら 2021b の分析を引く）．この場合の誤りはラベルの側にあり，入力の特徴をどう増やしても減らない（Bayes 誤差の側）．本件で言えば，写像が曖昧なタスク（japanese_civics など）の行は同じクエリの文面に 2 クラスの手がかりが両方あるので，語彙特徴でも分けられない見込みが高い（推定．文献は分類器の特徴の追加を直接は検証していない）．
  - 検索 6（問い (a) の日本語の事例）: 「Integrated ensemble of BERT- and features-based models for authorship attribution in Japanese literary works」（arXiv:2504.08527，2025，著者は抜粋に無い，https://arxiv.org/html/2504.08527v1 ）は，日本語の文書で BERT と文体の特徴量のモデルを統合する．関連研究として，Fabien らが BERT の出力と文体・文構造の特徴を LR でスタッキングしたところ，著者あたりの訓練例が 100 以下の場合を含む 4 データセットで，ほとんどのスコアが BERT 単体を下回ったと引いている．訓練行の少ないクラス（本件の legal 77 行など）でスタッキングの学習器が過学習すると，併用は逆効果になりうる．統合の利得の大きさは抜粋からは確かめられない．
  - 調査のまとめ: C1 の形（generalist の上位候補に限り，取り違えの多いクラスの集合に特化したモデルで並べ直す）は Hinton ら 2015 に直接の前例がある．埋め込みと語彙特徴の併用の効果は，査読つきの短文・日本語の対照実験では確かめられなかった（プレプリントと解説記事だけ）．小さい訓練集合ではスタッキングが単体を下回る報告がある（arXiv:2504.08527 が引く Fabien ら）．ラベルの定義が重なる行では，特徴を増やしても誤りは減らない見込みが高い（Uma ら 2022）．文献だけでは効果の有無を決められないので，判断は Step 0a（誤答行の重なり）の実測に委ねる．
- 到達経路の確認（コードを読んで確かめた）: いまの分類器の入口は埋め込みだけを受け取る（`classifier.py:51-70` の `estimate_confidence_classifier`，`knn_interpolated_head.py:135` の `predict_proba(X)`）．プローブのリクエストは `query_summary` を持ち（`protocol.py:26`），中身はクエリの先頭 200 文字である（`node.py:34` の `QUERY_SUMMARY_MAX_LENGTH = 200`，`node.py:211`）．`http_server.py:378-380` は現在 `body.query_embedding` だけを渡している．訓練のクエリの長さは中央値 103 文字，90 パーセンタイル 238 文字，最大 1,239 文字なので，訓練と replay でも語彙特徴は `query[:200]` から作り，本番と揃える．プロトコル（API のスキーマ）は変えずに済む．
- 判断: C1 を採る（B210）．config.yml の levers 末尾に `confusable_pair_lexical_rerank` を追記し，success_criteria に (10) を新設した．
- 仮説: 取り違えの多い対では，埋め込みが 2 クラスを近くに置くので，同じ埋め込みの上で決定層を変えても情報は増えない（M6）．文字 n-gram は埋め込みとは別の経路の情報（固有の用語，表記）を持つので，対の中の並びを一部直せる．ただし，写像が曖昧で文面に 2 クラスの手がかりが両方ある行（japanese_civics など）は直らない（Uma ら 2022 からの推定）．
- 単一レバー: 分類器の出力を，いまの補間と温度の後の確率 p から，次の並べ直しを加えた p' へ変える．
  - 対の選び方（事前登録．評価のラベルとクエリは使わない）: 訓練の外側 5-fold（seed 104）の OOF で，いまの構成の誤答を順序なしの対に集計し（両方向の和），行数の多い順に最大 4 対を取る．ただし 5 行未満の対は取らない．
  - 対ごとの語彙モデル: その 2 クラスの訓練行だけで，`TfidfVectorizer(analyzer="char", ngram_range=(1, 3), sublinear_tf=True)` を `query[:200]` に当て，`LogisticRegression(C=1.0, class_weight="balanced")` を訓練する．C は選び直さない．
  - 発火の条件: p の上位 2 クラスが選んだ対に一致する行だけ．
  - 並べ直し: 2 クラスの質量 m = p_a + p_b は保ち，対の中の比を s_a = 0.5 · p_a / m + 0.5 · r_a に変える（r_a は語彙モデルの確率）．p'_a = m · s_a，p'_b = m · (1 − s_a)，他のクラスは変えない．重みを学習しないのは，訓練行の少ない条件でスタッキングの学習器が単体を下回った報告（arXiv:2504.08527 が引く Fabien ら）があるためである．
- 固定する構成: `models/domain_classifier.joblib` の中身（MD5 c7172ad37c10e1082a42481553ae25b0，k=2，λ=0.3，T=0.9426．新しい artifact はこれを内側に包む），dispatch_gap_threshold 0.36，dispatch_gap_max_k 4，max_confidence，埋め込み，`config.yaml`，訓練ファイル `data/classifier_train_iter94_dedup.jsonl`．
- 期待効果: 上限は「正解が上位 2 候補に入っていて，対が選ばれた行」に限られる（Hinton の講演の指摘のとおり，上位 2 候補を外した行は直らない）．Iter108 の 2a の候補（合計 80 行）が上限の目安で，そのうち写像の曖昧な行は直らない見込みなので，Δtop1 は +0.5pt 前後が上限と見る（推定）．
- 成功条件: success_criteria (10) を参照．Step 0a（OOF で語彙モデルと現行の誤答行の重なり AGREE が 85% 以上ならクローズ），0b（入れ子 CV の OOF top1 が none を +0.5pt 以上，かつ 4/5 fold 以上で上回る），0c（replay の共通行 3,748 行で Δtop1 ≥ +0.5pt かつ正味 +3 行以上，McNemar を併記）の 3 段を満たしたときだけ本走へ進む．本走の主基準は基準線 results/20261004_225553 に対し Δtop1 ≥ +0.5pt かつ McNemar の両側 p < 0.05．
- Step 0a の AGREE の定義（測る前にオーケストレータが確定した．success_criteria (10) は分母を書いていないため）: 対象の行は，外側 5-fold（seed 104）の OOF で発火の条件（温度後の p の上位 2 クラスが選んだ対に一致）を満たし，かつ正解がその対に含まれる行とする．判定に使う AGREE は「いまの分類器の誤答行のうち，語彙モデル（同じ fold の訓練部分の 2 クラスの行だけで学習）も誤る行の割合」とする．語彙モデルで直せる行の上限が 1 − AGREE になるので，この向きを判定に使う．逆向き（語彙モデルの誤答のうち，いまの分類器も誤る割合）と，対ごとの内訳は併記するだけにする．0a の対の選択は非入れ子の OOF で行い，入れ子にするのは 0b からである．

### Iteration 110 実装・実験（2026-10-05）

- 実行者についての注記: Step 0a と 0b のスクリプトの作成と実行は，rc-executor に委譲せずオーケストレータが行った（SKILL.md「1 イテレーションの進め方」の手順違反．B211 に記録）．開発ホストの CPU だけで実行し，artifact，`config.yaml`，本番のコードは変えていない．`_iter105_replay_temp.py` の `load_cache`，`run_outer_fold`，`apply_temperature` を流用した．
- Step 0a の結果（`_iter110_step0a.py`，`_iter110_step0a.json`，CPU のみ）: OOF top1 0.8268．選ばれた対は education|medical（誤答 51 行），medical|natural_science（43），education|social_science（31），business_economics|education（29）．発火 697 行，対象 644 行（正解が対の外の発火 53 行は対象外）．いまの分類器の誤答 122 行，語彙モデルの誤答 142 行，両方の誤答 59 行．**AGREE = 59/122 = 48.4%**（逆向き 41.5%）で閉じる条件（≥ 85%）を満たさないので，0b へ進む．直せる行の上限は 63 行．
  - 読み: AGREE が低いことは，語彙モデルが独立な情報を持つ証拠とは限らない．対象の行での語彙モデルの正解率は 71.7〜85.5% で，medical|natural_science では 73.8%（いまの分類器は 82.9%）とむしろ弱い．弱い語彙モデルの誤りは，雑音であってもいまの分類器の誤りと重ならないので，AGREE を下げる．M6 の OvO/ECOC は同じ埋め込みの上の分類器同士で正解率も近かったので，AGREE の尺度をそのまま比べられない点に注意する．
  - 診断（判定に使わない．対の選択に検証行が漏れている）: 並べ直した OOF top1 は 0.8286（+0.18pt，wrong→correct 10 行，correct→wrong 6 行）．漏れがある条件でも 0b の基準 +0.5pt に届いておらず，0b は通らない見込み（推定）．
- Step 0b の結果（`_iter110_step0b.py`，`_iter110_step0b.json`，CPU のみ．入れ子 CV で，対の選択と語彙モデルの学習は外側の訓練部分だけで行った）: 外側 top1 の平均は none 0.8268 → 並べ直し 0.8290（**+0.22pt**）．fold ごとの差は 0.00 / +0.66 / −0.22 / +0.66 / 0.00pt で，**上回ったのは 2/5 fold**．正味は合計 +5 行（wrong→correct 12 行，correct→wrong 7 行．rc-evaluator が `_iter110_step0b.json` の fold ごとと対ごとの値から数え直して訂正した．当初は 16 行と 11 行と書いていた）．進む条件（平均 +0.5pt 以上かつ 4/5 fold 以上）を両方とも満たさないので**不通過**．0c と本走は行わず，artifact を変えずにクローズする（success_criteria (10)）．
  - 選ばれた対は fold で変わった（education|medical と medical|natural_science は全 fold，business_economics|education は 4 fold，education|legal と education|social_science は 3 fold と 2 fold）．
  - 対ごとの遷移（5 fold の合計）: education|medical 発火 176 行で +7/−4，medical|natural_science 217 行で +0/−1，business_economics|education 110 行で +3/−0，education|legal 45 行で +2/−1，education|social_science 94 行で +0/−1．
  - 読み: 0a の「直せる行の上限 63 行」のうち実際に直ったのは 12 行で，7 行の正答が崩れた（rc-evaluator が訂正した．当初は 16 行と 11 行）．語彙モデルの対象の行での正解率（71.7〜85.5%）がいまの分類器と同程度かそれ以下なので，混合の重み 0.5 では argmax を動かすことが少なく，動かした行も両方向に割れる．0a の AGREE 48.4% は独立な情報ではなく，主に語彙モデルの誤りの雑音を測っていたと読むのが実測に整合する（推定）．文字 n-gram の側からも，取り違えの多い対を分ける情報は Step 0 の閾値を超えるほどには増えない．
- 暫定の読み（判定は rc-evaluator が `### Iteration 110 実行済み` で行う）: success_criteria (10) の規則どおりなら，`confusable_pair_lexical_rerank` は Step 0b で棄却となる（Iter93，Iter108，Iter109 と同じ扱い）．levers は再び使い切った状態に戻る．B209 の要レビューと 0a の申し送りに従い，status=converged を要人間判断として扱う（backlog B211）．

### Iteration 110 実行済み（2026-10-05）

**判定: rejected（success_criteria (10) の Step 0b で不通過．0c と本走は行わない）**

| 項目 | 値 |
|---|---|
| 0a の AGREE（いまの分類器の誤答のうち語彙モデルも誤る割合） | 59/122 = 48.4%（逆向き 41.5%）．閉じる条件 ≥ 85% を満たさず通過 |
| 0b の外側 top1（none / 並べ直し） | 0.8268 / 0.8290（+0.22pt．進む条件は +0.5pt 以上） |
| 0b の fold ごとの差 | 0.00 / +0.66 / −0.22 / +0.66 / 0.00pt（上回った fold は 2/5．進む条件は 4/5 以上） |
| 0b の遷移（5 fold の合計） | 誤→正 12 行，正→誤 7 行，正味 +5 行．二項の正確検定の両側 p = 0.36 |

- 数値の訂正: 実装・実験の節と B211 は 0b の遷移を「+16/−11」と書いていたが，`_iter110_step0b.json` の fold ごとの値（1+3+1+3+4 / 1+0+2+0+4）と対ごとの値（7+0+3+2+0 / 4+1+0+1+1）のどちらでも 12 / 7 になる．正味 +5 行と判定は変わらない．
- ノイズとの切り分け: fold の差の標準偏差は 0.41pt，標準誤差は 0.18pt で，平均 +0.22pt は標準誤差の約 1.2 倍にとどまる．fold あたり 455 行なので，1 行が 0.22pt にあたり，fold の差は −1〜+3 行の範囲で動いただけである．遷移 12 対 7 も二項の揺らぎの範囲である（p = 0.36）．ノイズの範囲内と判定する．
- B200 の床（正味 ±2 行）との関係: B200 の床は，replay と本走の比較（3,748 行，埋め込みの取り直しによる揺らぎ）から得た尺度で，同じキャッシュの上で決定論的に計算する CV には当てはまらない．CV では fold の間の分散と，不一致の行の二項の揺らぎがノイズの尺度になる．0c に進んでいれば B200 の床を使う設計だった．
- 仮説との照合: 「文字 n-gram は埋め込みとは別の経路の情報を持ち，対の中の並びを一部直せる」は，直る向きの行（12 行）が出た点では否定されないが，Step 0b の閾値を超える大きさでは支持されなかった．計画の「Δtop1 は +0.5pt 前後が上限」という見込みに対し，実測はその半分以下だった．
- 想定外の挙動: medical|natural_science は全 fold で選ばれ，217 行で発火したが，直った行は 0 行，崩れた行は 1 行だった．この対では語彙モデルの正解率（73.8%）がいまの分類器（82.9%）を下回る．

**解釈（事実と推定を分ける）**:

- (a) 事実: 対象の行での語彙モデルの正解率は 71.7〜85.5% で，いまの分類器（74.7〜84.7%）と同程度かそれ以下だった．
- (b) 推定: 0a の AGREE 48.4% は，語彙モデルが独立な情報を持つことよりも，弱いモデルの誤りが雑音として散ったことを主に測っていた．AGREE は，正解率の近い 2 つの分類器（M6 の OvO/ECOC）どうしで比べたときだけ，情報の重なりの尺度になる．
- (c) 推定: 写像の曖昧な行では文面に 2 クラスの手がかりが両方あり（Uma ら 2022），語彙でも分けられない．education を含む対で直った行が多かった（education|medical +7/−4，business_economics|education +3/−0）ことは，一部の行に固有の用語の手がかりがあることと整合するが，量は小さい．

**採用構成は変えない**: Iter105 の artifact（MD5 `c7172ad37c10e1082a42481553ae25b0`）に Iter107 の報告用フィールドを加えたもの．基準線は `results/20261004_225553` のままとする．

**学び**:

1. AGREE（誤答行の重なり）は，比べる 2 つの分類器の正解率が近いときだけ，新しい情報の有無の尺度になる．正解率の低いモデルと比べると，雑音で AGREE が下がり，0a を素通りする．今後この型の足切りを置くときは，語彙モデル単体の正解率がいまの分類器以上であることを条件に加える．
2. 取り違えの多い対に限っても，文字 n-gram の TF-IDF の LR は埋め込みの分類器と同程度かそれ以下の正解率しか出さない．固有の用語の手がかりは一部の行にしか無い．
3. 写像を固定した条件では，決定層（M6），訓練行の除去（Iter108），訓練行の追加（B161），kNN 類似度の前処理（Iter106），事前分布の補正（Iter109），別の入力の特徴（Iter110）の 6 系統が，すべて Step 0 の閾値に届かなかった．
4. 記録の数値は，一次資料の JSON から数え直してから書く（今回，遷移の行数を 16 / 11 と誤記していた）．

**次の一手**: 調査・計画フェーズからの再探索（Iter108〜110）が 3 回続けて Step 0 で棄却されたので，手順 7-3 の条件に当たると判断する．ただし converged は研究の結論の確定に近いので自動では設定せず，要人間判断とする（B212）．回答があるまでは，次のイテレーションを調査・計画フェーズから始める（`current_lever=null`）．B206（写像の見直し）と B207（π_t の推定の扱い）は回答待ちのままである．

## Iteration 109: 訓練と評価のドメイン事前分布の差を EM で補正する

このブロックは，rc-researcher の報告をもとにオーケストレータが挿入した（rc-researcher はツールの呼び出しの形式を誤り，記録の前にツールを使えなくなった．起動モードは tenbin）．ドメインの分布，Iter92 の記録の位置，文献の書誌は，オーケストレータが確かめ直した．

### 調査・計画

#### 調査

- Iter92 で logit adjustment（Menon ら，ICLR 2021，arXiv:2007.07314）を検討した（`journal_archive.md:4397`，`9750`）．訓練の実効重みは 10 ドメインで揃えてあるので，原理的な補正量はほぼ 0 とした．このときは訓練集合の中の偏りだけを扱い，訓練と評価の分布の差は扱っていない．
- ドメインの分布（評価は `results/20261004_225553/results.jsonl` の `expected_domains` が 1 つの行，訓練は `data/classifier_train_iter94_dedup.jsonl`．オーケストレータが数え直した）:

| ドメイン | 評価の単一行 | 割合 | 訓練 | 割合 |
|---|---|---|---|---|
| business_economics，mathematics，medical，natural_science | 各 350 | 各 11.6% | 各 250 | 各 11.0% |
| education | 350 | 11.6% | 238 | 10.5% |
| history_culture | 350 | 11.6% | 210 | 9.2% |
| computer_science | 301 | 10.0% | 250 | 11.0% |
| social_science | 294 | 9.7% | 250 | 11.0% |
| general | 175 | 5.8% | 250 | 11.0% |
| legal | 150 | 5.0% | 77 | 3.4% |
| 計 | 3,020 | — | 2,275 | — |

- 評価の残りの 730 行は複合行である．
- 実効の事前分布: LR は balanced な重みで学習するのでほぼ一様で，kNN の datastore は訓練行数の比を持つ（legal 3.4%）．補間後の分布の実効的な事前分布は，どちらとも一致しないと見込む（推定．Step 0 で OOF の平均確率として実測する）．
- Saerens, Latinne, Decaestecker, "Adjusting the Outputs of a Classifier to New a Priori Probabilities: A Simple Procedure", Neural Computation 14(1):21–41, 2002（DOI 10.1162/089976602753284446）．事後確率を出す分類器について，ラベルなしの新しいデータから事前分布を EM で推定し，出力を補正し直す手順である（書誌は tavily-search で確認した．本文は読んでいない）．
- Alexandari, Kundaje, Shrikumar, "Maximum Likelihood with Bias-Corrected Calibration is Hard-To-Beat at Label Shift Adaptation", ICML 2020, PMLR 119:222–232．事後確率を bias 項つきの温度スケーリングなどで較正してから EM（最尤推定）を使うと，BBSE などの代替法に対して強い基準線になる．ICML の発表スライドは，尤度が凹なので EM は大域最適に収束すると述べる（tavily-search の抜粋で確認した）．
- 注意: Pisa 大学の発表資料（pesaresi_molinari.pdf）には，SLD（Saerens らの EM）が期待外れの結果を出すことがあるという記述がある（抜粋だけを確認した）．較正が不十分な事後確率では，推定が崩れうる．本リポジトリの分類器は温度 T=0.9426 だけで較正しており，bias 項は無い．

#### 仮説

補間後の分布の事前分布を，ラベルなしの評価のクエリから EM で推定した事前分布に合わせると，general を過剰に選ぶ誤りが減り，単一行の top1 が上がる．主な誤り（education → business_economics など）は，評価で同じ割合（11.6%）を持つクラスの間で起きているので，効果は小さいと見込む（推測）．Step 0 で落ちる可能性が高い．

#### 単一レバー

- `test_time_label_shift_prior_correction`: `saerens_em_on_unlabeled_queries`．
- 補間と温度を適用した後の確率に w_y = π_t(y) / π_s(y) を掛け，正規化し直す．π_s は訓練の OOF の平均確率，π_t はラベルなしの評価のクエリ（単一行と複合行の 3,750 行）に対する EM の推定値とする．EM は収束まで回し，調整する値は持たない．
- 変更箇所: `knn_interpolated_head.py` の `predict_proba`（属性 `class_prior_weights` を足す．既定値は無補正），`scripts/build_knn_interpolated_classifier.py`（重みを artifact に書き込む）．
- 固定する構成: Iter105 の artifact（MD5 `c7172ad37c10e1082a42481553ae25b0`，k=2，λ=0.3，T=0.9426）と Iter107 の報告用フィールド，`dispatch_gap_threshold` 0.36，`dispatch_gap_max_k` 4，max_confidence．`_DOMAIN_TASK_MAP` と評価集合の定義は変えない（B206 の A1-1）．

#### Step 0（replay による足切り）

- `_iter105_replay_temp.py` の関数（`run_outer_fold`，`apply_temperature`，`simulate_rows`，`summarize`，`c1_tests`）を流用し，同じ評価キャッシュの上で「補正 − 無補正」を比べる．出力は `.claude/research/_iter109_replay_prior.{py,json}` とする．
- 進む条件: Δtop1 が +0.5pt 以上で，ノイズの床（B200: top1 の正味 ±2 行）を超えること．満たさなければ artifact を変えずにクローズする（Iter93，Iter108 と同じ扱い）．
- 記録する値: π_s，π_t，ドメインごとの重み，EM の反復回数と収束，argmax が変わった行数とその向き，複合行の送出集合の変化．π_t と評価の実際の分布との差は診断のためだけに記録し，選択には使わない．

#### 到達条件

- `http_server.py:428` → `classifier.py:48` で artifact を読み，`http_server.py:378-380` → `classifier.py:66-69` → `knn_interpolated_head.py:135-149` の `predict_proba` が重みを掛ける（行番号は Iter108 の実行フェーズで照合済み．変更後に確かめ直す）．
- 発火の証拠: 新しい artifact の MD5 が `c7172ad37c10e1082a42481553ae25b0` と異なること，`class_prior_weights` がすべて 1 ではないこと，deploy 後の全ノードのコンテナの中で MD5 が一致すること．

#### 成功条件（事前登録．config.yml の success_criteria (9)）

- 主基準（本走）: 基準線 `results/20261004_225553` の共通行（dispatch_failed を除く）で，Δtop1 ≥ +0.5pt かつ McNemar の両側 p < 0.05．
- 非退行: Iter107 の C1〜C7．legal と general の recall の差を併記する．ECE は新旧を併記し，判定は旧の定義で行う（B203 の A1-1）．決定を変えるレバーなので本走を行う（B203 の A2-2）．

#### 未確認の点

- 2 本の文献の本文（書誌と抜粋だけを確かめた）．
- ラベルなしの評価のクエリから事前分布を推定することの扱い（B207 の要レビュー）．
- 複合行 730 行の送出集合への影響．

### Iteration 109 実装・実験（2026-10-05）

#### Step 0（replay による足切り）: 不通過

- スクリプトは `.claude/research/_iter109_replay_prior.py`，出力は `_iter109_replay_prior.json`，ログは `_iter109_step0.log` である．開発ホストの CPU だけで実行した（wafl500〜509 は使っていない）．`_iter105_replay_temp.py` の `run_outer_fold`，`apply_temperature`，`simulate_rows`，`summarize`，`c1_tests` を流用した．統計量は `metrics.py` の `compute_mcnemar_test`，`compute_precision_recall_per_domain` を呼んだ．
- π_s は，外側 5-fold（seed 104）の OOF 補間分布に artifact の温度 T=0.9426 を当てた確率の平均である（OOF top1 0.826813）．π_t は，artifact の `predict_proba` の出力（補間と温度を適用済み）の 3,750 行に Saerens らの EM を当てて求めた．初期値は π_s で，停止条件は各成分の変化が 1e-10 未満になることとした．45 回の反復で収束した．
- 比較の範囲は，基準線 `results/20261004_225553` の共通行（confidence が非 null．`_iter107_main_summary.py` と同じ条件）の 3,748 行である．無補正の replay と本走の一致は，selected が 0.995731（16 行が不一致），送出集合が 0.984258（59 行が不一致）だった．

| ドメイン | π_s | π_t（EM） | w = π_t/π_s | 補正前の eval の平均予測 | eval の実際（診断用．複合行は等分） |
|---|---|---|---|---|---|
| business_economics | 0.1120 | 0.1437 | 1.2831 | 0.1357 | 0.1125 |
| computer_science | 0.1099 | 0.1045 | 0.9514 | 0.1051 | 0.0995 |
| education | 0.0976 | 0.0793 | 0.8126 | 0.0974 | 0.1128 |
| general | 0.0999 | 0.0375 | 0.3749 | 0.0501 | 0.0653 |
| history_culture | 0.0971 | 0.1416 | 1.4587 | 0.1261 | 0.1125 |
| legal | 0.0429 | 0.0597 | 1.3921 | 0.0531 | 0.0608 |
| mathematics | 0.1132 | 0.1033 | 0.9127 | 0.1058 | 0.1125 |
| medical | 0.1069 | 0.1547 | 1.4470 | 0.1353 | 0.1139 |
| natural_science | 0.1160 | 0.1009 | 0.8697 | 0.1062 | 0.1125 |
| social_science | 0.1045 | 0.0747 | 0.7150 | 0.0853 | 0.0976 |

- 実際の分布の列は診断のためだけに記録し，選択には使っていない．

| 指標（共通行 3,748） | 無補正 | 補正 | 差 |
|---|---|---|---|
| top1（全体） | 0.850320 | 0.841515 | −0.8805pt（正味 −33 行） |
| top1（単一行 3,018） | 0.860504 | 0.854539 | −0.5964pt（正味 −18 行） |
| top1（複合行 730） | 0.808219 | 0.787671 | −2.0548pt（正味 −15 行） |
| compound_domain_set_recall | 0.574658 | 0.554110 | −2.0548pt |
| compound_mean_dispatched_count | 1.975342 | 1.934247 | −0.041096 |
| ECE（`summarize` の値） | 0.037125 | 0.018875 | −0.018250 |

- argmax が変わった行は 186 行である．向きは，誤→正が 50，正→誤が 83，誤→誤が 42，正→正（複合行）が 11 だった．
- 単一行で多い変化（[正解] 旧→新）: [education] education→medical 12，[medical] education→medical 11，[education] education→history_culture 8，[education] education→legal 7，[education] education→business_economics 6，[social_science] social_science→legal 6，[medical] natural_science→medical 5．
- 複合行の送出集合は 145 行で変わった．送出数の変化は，4→4 が 45，4→1 が 43，1→4 が 34，2→1 が 8，2→4 が 8，4→2 が 6，1→2 が 1 である．
- recall の差（補正 − 無補正）: legal +0.98pt（0.5588→0.5686），general −3.49pt（0.4159→0.3810）．ほかに education −8.87pt，medical +6.36pt，social_science −2.97pt，history_culture +2.23pt，natural_science −1.83pt．
- McNemar（補正 対 無補正）: 全体は 50 対 83 で p=0.00552，単一行は 44 対 62 で p=0.0987，複合行は 6 対 21 で p=0.00705．C1 の BH 補正で有意になったのは 7 指標（education_recall，general_recall，general_precision，history_culture_recall，medical_recall，natural_science_recall，social_science_recall）である．
- 進む条件（Δtop1 ≥ +0.5pt かつ正味の行数がノイズの床 ±2 行を超える）を満たさない．計画どおり artifact，`config.yaml`，`knn_interpolated_head.py`，`scripts/build_knn_interpolated_classifier.py` は変えず，deploy，G0，本走は行わなかった．experiment_dir は null のままである．

### Iteration 109 実行済み（2026-10-05）

**判定: rejected（success_criteria (9) の Step 0 で不通過．本走しない）**

| 項目 | 値 |
|---|---|
| top1（共通行 3,748．無補正 / 補正） | 0.850320 / 0.841515 |
| Δtop1 | −0.88pt，正味 −33 行（進む条件は +0.5pt 以上かつ正味 +3 行以上） |
| McNemar（誤→正 / 正→誤） | 全体 50 / 83，p=0.0055（二項の正確検定でも 0.0053）．単一行 44 / 62，p=0.099．複合行 6 / 21，p=0.0071 |
| C1（BH，q=0.05） | 有意な変化 7 指標．education_recall −8.87pt，general_recall −3.49pt，social_science_recall −2.97pt，natural_science_recall −1.83pt が退行の向き |
| legal / general の recall | +0.98pt / −3.49pt |
| ECE（`summarize` の値） | 0.0371 → 0.0189（判定には使わない） |

- ノイズとの切り分け: replay は決定論的なので，無補正と補正の差はレバーだけから生じる．B200 の床（top1 の正味 ±2 行）の 16 倍以上の差である．replay と本走の selected の不一致 16 行がすべて補正の側に有利に働いたと仮定しても，正味 −17 行で下がる向きは変わらない．全体の McNemar は悪化の向きで有意である．単一行だけでは p=0.099 なので，単一行の悪化は有意とまでは言わない．
- 仮説との照合: 「general を過剰に選ぶ誤りが減り，単一行の top1 が上がる」は支持されなかった．general の重みは 0.375 まで下がったが，general の recall も下がり，単一行の top1 は −0.60pt だった．計画の「効果は小さく，Step 0 で落ちる可能性が高い」という見込みとは，落ちる点で一致し，効果の大きさ（悪化が有意になること）で一致しない．

**解釈（事実と推定を分ける）**:

- (a) π_t は評価の実際の分布から離れた（事実）．実際の分布（診断用．複合行は等分）との全変動距離は，π_s が 0.056，補正前の eval の平均予測が 0.064，EM の π_t が 0.106 で，EM が最も遠い．10 クラスのすべてで「π_t − 平均予測」の符号は「平均予測 − π_s」の符号と一致した．EM は平均予測のずれを打ち消さずに，同じ向きへ広げた．仮に実際の分布を使った場合の比 π_actual/π_s と EM の重みを比べると，legal（1.418 / 1.392）だけが近く，general（0.654 / 0.375）と history_culture（1.159 / 1.459），medical（1.065 / 1.447）は行き過ぎ，education（1.156 / 0.813）は向きが逆である（この比は診断のためだけに計算し，選択には使っていない）．
- (b) 較正と前提（推定）．EM の不動点は「補正後の平均事後確率 = π_t」なので，平均予測のずれがすべて事前分布の差から来るという前提（label shift．p(x|y) が訓練と評価で同じ）に依存する．ここでの平均予測のずれは，クラスの割合の差よりも，クラスごとの取り違え（education の行が medical や business_economics へ流れる）から来ている見込みが強い．温度だけの較正にはクラスごとの bias 項が無いので，この取り違えによるクラス別の偏りが較正の後も残り，EM がそれを事前分布の差として増幅した，と考えるのが (a) の事実と整合する．bias 項つきの較正（Alexandari らの BCTS）で崩れが止まるかは確かめていない．取り違えが写像の曖昧さ（Iter108 の学び 1）に由来するなら，較正を変えても label shift の前提は満たされない見込みである（推定）．
- (c) education が下がった理由．事実: 補正前の分類器は eval で education に平均 0.0974 の確率しか置かず，実際の割合（0.1128）を下回る．education の recall は 0.50 と全クラスで最も低い．EM は education の π_t を 0.0793 へ下げ，重みは 0.813 になった．education→medical の argmax の変化は全体で 46 行あり，単一行では正解 education の行が education から medical，history_culture，legal，business_economics へ計 33 行移った（正→誤）．一方，正解 medical の行が education→medical で 11 行直った．推定: education の行の多くは他クラスとの差が小さく（僅差の行），1 を下回る重みで容易に順位が入れ替わる．分類器が education を当てにくいこと自体が，EM には「評価に education が少ない」と見え，さらに当てにくくする方向へ働いた．
- ECE が 0.0371 から 0.0189 へ下がったのは，平均の確信度が平均の正解率へ寄った（gap −0.037 → −0.019）ためで，top1 の悪化とは別の現象である．ECE は (9) の判定に使わない．

**採用構成は変えない**: Iter105 の artifact（MD5 `c7172ad37c10e1082a42481553ae25b0`）に Iter107 の報告用フィールドを加えたもの．基準線は `results/20261004_225553` のままとする．

**学び**:

1. 取り違えに偏りのある分類器では，Saerens らの EM は平均予測のずれを事前分布の差として増幅する．本リポジトリでは π_t が実際の分布から π_s よりも遠くへ離れた（全変動距離 0.106 対 0.056）．誤りの根がクラスの割合ではなくタスクの写像の曖昧さにあるという Iter108 の見立てと整合する．
2. 当てにくいクラス（education，recall 0.50）ほど EM で重みが下がり，さらに当てにくくなる．事前分布の補正は，recall の低いクラスを持つ分類器では退行の向きに働きうる．
3. 補正後の ECE の低下は top1 の改善を意味しない．Step 0 を top1 と正味の行数で事前登録していたので，ECE に引きずられずに判定できた．CPU だけの replay で，deploy と本走を使わずに棄却できた（Iter93，Iter108 と同じ型）．

**次の一手**: levers は再び使い切りになった．bias 項つきの較正と EM の組み合わせは，(b) の推定から label shift の前提そのものが崩れている見込みがあり，B207 の要レビュー（評価のクエリから π_t を推定してよいか）も未回答なので，確信を持って選べない．Iter110 は調査・計画フェーズから始める（B208）．B206（写像の見直し）と B207（π_t の推定の扱い）は人間の回答待ちのままである．

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

