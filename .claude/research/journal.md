## Iteration 59: multi-labelヘッドによるdispatch候補順位付けのオフライン検証

### 調査 (Iter59)

**問い**
- Q1: 既存の argmax 単一ラベル分類器の上に，多ラベルランキング（binary relevance / OvR sigmoid ヘッド）を
  後付けする設計の先行研究・実践知見（較正上の注意点，多段設計の先行例，少量データでの過学習対策）。
- Q2（最優先）: 訓練スクリプト・データスキーマ・実行時経路・`results.jsonl` のフィールド・
  `metrics.py` の関連関数・`select_dispatch_targets()` 系の全呼び出し元を正確に特定する。
- Q3: Q1・Q2 を踏まえた，次の計画フェーズが実装計画を書けるレベルの具体的な手順。

**分かったこと（Q1: 先行研究）**

- **OvR/binary relevance の較正が抱える構造的問題は，このリポジトリのコード自身のコメントと
  scikit-learn 公式ドキュメントの両方で一致して確認できた**。`scikit-learn.org/stable/modules/
  calibration.html` §1.16.3.3 は「`CalibratedClassifierCV` はマルチクラスに対して各クラスを
  OvR 方式で個別に較正する．各クラスの較正済み確率は独立に予測されるため合計が 1 にならず，
  事後的な正規化（renormalization）が行われる」と明記している（出典:
  https://scikit-learn.org/stable/modules/calibration.html ，
  補足の Stack Overflow スレッド https://stackoverflow.com/questions/60110209 も同旨）．
  本リポジトリの `scripts/train_domain_classifier.py`（L39-48）・`classifier.py`（L11-19）の
  コメントは，この事実を根拠に「temperature scaling は単一スカラーで logits 全体を再スケールする
  ため softmax 出力が合計 1 を保つが，isotonic/platt は per-class 較正器を持つため合計が 1 に
  ならない」という Iter31 の較正手法選定理由を説明しており，**本レバーで新設する OvR sigmoid ヘッドは
  この「合計が1にならない」問題をまさに抱える方式であることが，過去の自分自身の較正実験
  （Iter29 platt・Iter30 isotonic）の失敗機序（medical_recall の BH 補正後有意悪化）と地続きである**．
  ただし今回はこの非正規化スコアを「確率」としてではなく「rank_2 以降を並べ替えるためのランキング
  スコア」としてのみ使うため（softmax と比較して argmax を取るわけではない），較正の破綻が
  argmax の決定に影響しない設計である点が Iter29/30 との重要な違いである．
  （MetricGate の整理記事 https://metricgate.com/blogs/one-vs-rest-vs-one-vs-one-classifier も
  「OvR スコアは per-binary-problem で較正されているだけで，jointly には較正されていない．
  確率ベクトルが欲しければ softmax か Platt+再正規化が必要」と同旨を述べている）。
- **1位は既存の強いモデルに任せ，2位以降だけ別モデルで並べ替える，という設計そのものの先行例は
  検索範囲内では見つからなかった**。関連が近いのは検索・推薦分野の「retrieve-then-rerank」
  2 段階設計（例: BM25 で候補生成→cross-encoder で re-rank，futureagi.com の LTR 用語集
  https://futureagi.com/glossary/learning-to-rank ，MDPI の MultiLTR
  https://www.mdpi.com/2078-2489/16/4/308）だが，これらは通常**候補集合全体**を再ランキングし，
  「1位だけ固定して残りだけ動かす」という非対称な設計ではない。したがって本レバーの設計
  （rank_1 argmax 固定＋ rank_2+ のみ OvR で並べ替え）は，文献上の直接の先例を持たない独自設計
  であり，その安全性（argmax flip rate 構造的0%）は文献の裏付けではなく本リポジトリのコード
  ロジック自体（`select_dispatch_targets()` を全く変えず，並べ替えは候補リスト構築より前で完結
  させる）から導かれる点を計画書に明記すべきである。
- **少量データでの binary relevance 過学習**: Zhang & Zhou, "Binary Relevance for Multi-Label
  Learning: An Overview" (Frontiers of Computer Science 2017) は，binary relevance の各ラベルの
  二値問題が「クラス不均衡（正例が少数派）」を本質的に抱えることを指摘し，対策として
  クラス重み付け・閾値調整・ラベル依存性の活用（classifier chain）を挙げている
  （出典: http://palm.seu.edu.cn/zhangml/files/FCS'17.pdf ）。
  Wikipedia の Multi-label classification 項目も同旨（binary relevance はラベル間の依存関係を
  一切見ない設計であることが弱点，と明記，https://en.wikipedia.org/wiki/Multi-label_classification ）。
  **本リポジトリのデータでは legal が 77 件（他ドメインの約半分）で最小**であり，かつ
  compound 100 行のうち legal が絡む行が 30 件（全ドメイン中最多，2 番目は medical の 28 件）
  という調査結果（下記 Q2 の実データ確認）と合わせると，**legal の二値分類器の質が本レバーの
  compound recall 改善効果を左右しやすい構造**にあることに注意が必要．
  **対策**: (a) sklearn `LogisticRegression(class_weight="balanced")` を各二値問題に**独立に**
  適用する（OvR の各二値分類器は互いに独立なので，本体分類器で起きた Iter32 の
  `sample_weight × class_weight_` 結合バグは構造的に起こり得ない．各ラベルの `class_weight_`
  はそのラベルの正例/負例比のみに依存する）。(b) `CalibratedClassifierCV` の 5-fold CV を
  ヘッド訓練にもそのまま踏襲し，legal のような少数クラスでの単純な訓練セット丸暗記を避ける。
- **OneVsRestClassifier vs MultiOutputClassifier の実装選択**: scikit-learn 公式ドキュメント
  （https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html）
  によれば，`OneVsRestClassifier` はマルチラベルの場合そのまま binary relevance を実装し，
  `MultiOutputClassifier` は「マルチラベル拡張の別の方法」として同義的に使える。本リポジトリの
  データは各行が単一ドメインラベルのみを持つ（`classifier_train.jsonl` は 1 行 1 ドメイン）ため，
  多ラベル目的変数を作るには `MultiLabelBinarizer` は不要で，**既存の単一ラベル配列をそのまま
  `OneVsRestClassifier(LogisticRegression(...))` に渡せばよい**（各ラベルについて「そのドメイン
  かそれ以外か」の二値問題が自動的に構成される）。

**分かったこと（Q2: コードベース調査，最優先）**

1. **`scripts/train_domain_classifier.py` の構造**（L1-223 通読）:
   - `_load_training_rows()` で `classifier_train.jsonl` を読み，`_extract_sample_weights()` で
     `n_samples / (n_classes * n_domain_samples)` のドメイン別重み（`class_weight='balanced'` と
     数式的に同値，Iter39 で確定した設計）を計算する。
   - `build_training_features()` は各行の `query` を `ollama_client.embed(embedding_model, ...)`
     で埋め込む（**逐次実行，並列化なし**）。**embedding はどこにもキャッシュされない**
     （joblib で保存されるのは学習済み分類器のみで，埋め込みベクトル自体はディスクに残らない）。
     したがって新しい OvR ヘッドを訓練するには，1427 行を**再度 embed し直す必要がある**
     （embedding_model は `nomic-embed-text` で固定，`config.yaml` 参照）。
   - `train_classifier()` は `LogisticRegression(max_iter=1000, class_weight=None)` を
     `CalibratedClassifierCV(method="temperature", cv=5, ensemble=True)` でラップし，
     `education` クラスの `intercept_` に `+0.7`（`intercept_delta`，Iter45 採用）を較正後の
     各 fold の estimator に直接加算する（L200-204）。これは**現行の rank_1 分類器の訓練ロジック**
     であり，本レバーでは一切変更しない（新設する OvR ヘッドは全く別のクラス・別のファイルとして
     並存させる）。
   - **本レバー実装への示唆**: 新スクリプト（例: `scripts/train_dispatch_candidate_ranking_head.py`）
     は `_load_training_rows()`・`build_training_features()` をそのまま import して再利用し，
     `train_classifier()` の代わりに `OneVsRestClassifier(LogisticRegression(max_iter=1000,
     class_weight="balanced"))` を fit する関数を新設するのが最小差分（embed のコードパスは
     完全に共用でき，1427 行の embed 計算だけがコスト＝オフラインで数分）。

2. **`data/classifier_train.jsonl` のスキーマ**: 1427 行，フィールドは `{"id", "query", "domain"}`
   の3つのみ（`sample_weight` フィールドは現状のデータには存在せず，コード側にオプション対応が
   残っているだけ）。ドメイン別行数を実測: `business_economics/computer_science/education/general/
   history_culture/mathematics/medical/natural_science/social_science` が各150件，**`legal` のみ
   77件**（合計 150×9+77=1427 と一致）。

3. **`models/domain_classifier.joblib` の入出力形式**: `classifier.py:load_domain_classifier()`
   で読み込む `CalibratedClassifierCV` オブジェクト。`.classes_`（ドメイン名の配列）と
   `.predict_proba([embedding])[0]`（10 要素の確率配列，合計1）の2つのAPIのみに依存する
   （duck typing，L14-19 のコメント）。**この分類器はレバー実装で一切変更しない**（rank_1 の
   argmax を担保する唯一の情報源として維持する）。

4. **`classifier.py:estimate_confidence_classifier()`（Iter57 で education 加算が追加された実行時
   経路）**: `domain not in classifier.classes_` なら 0.0，そうでなければ
   `predict_proba([query_embedding])[0][domain_index]` を返し，`domain=="education"` のときのみ
   `EDUCATION_THRESHOLD(0.05)` を**正規化なしで**加算する（L59-78）。**呼び出し元は
   `http_server.py:367` の1箇所のみ**（`/probe` エンドポイント，各ノードが自分のドメイン分だけ
   呼ぶ）。`results.jsonl` の `probe_candidates` フィールドは，1行あたり全10ノード分の
   `{"node_id", "domain", "confidence", "confidence_logprobs_mean"}` のリストであり，**この
   `confidence` が `estimate_confidence_classifier()` の返り値そのもの**（実測で確認，
   `results/20260919_005727/results.jsonl` の1行目を確認）。つまり `probe_candidates` は
   「10ドメイン全ての現行分類器 confidence（rank_1〜rank_10 相当）」を既に完全に保持しており，
   **rank_1（argmax）の再現には `results.jsonl` の再生だけで十分**（embedding の再計算は不要）。
   ただし**query_embedding 自体は `results.jsonl` に保存されていない**ため，新設する OvR ヘッドの
   スコアを得るには 1600 問の embedding をあらためて計算し直す必要がある（`node.py:202`
   `ollama_client.embed(config["embedding_model"], query)` が実行時に一度だけ計算し捨てている）。
   この「embedding は非永続」というパターンは `scripts/evaluate_classifier_calibration.py`
   （Iter29 以降，較正手法比較のたびに使われてきたオフライン評価テンプレート）の docstring にも
   明記されている（「query_embedding is not persisted in results.jsonl, so it must be recomputed；
   ただし LLM 生成・probe・dispatch トラフィックは一切発生しない」）。**この既存スクリプトが
   本レバーのオフライン採点スクリプトの直接のテンプレートになる**。

5. **`metrics.py` の関連関数**:
   - `compute_compound_coverage_metrics()`（L127-）: compound 行（`len(expected_domains)>1`）のみを
     対象に，`dispatched_domains` と `expected_domains` の集合演算で
     `compound_domain_set_recall = covered_domain_count / expected_domain_total` を算出する
     （`r.get("dispatched_domains")` で古い形式は自動スキップ，後方互換）。
   - `_mcnemar_from_correctness()`（L228-）: 連続性補正付き McNemar（2値の対応マップ2つを受け取る
     汎用関数）。`compute_mcnemar_test()`（top1 用）・`compute_domain_recall_mcnemar_test()`
     （ドメイン別 recall 用）はいずれもこれを呼ぶだけの薄いラッパー。**Iter58 の主基準検定
     （ドメイン単位 n=200 のペア比較）は，この関数を `(row_id, expected_domain)` ペアの
     `dict[str, bool]`（baseline が被覆したか／new が被覆したか）に直接投入することで実現していた**
     （journal Iter58「実験」節）。本レバーの主基準検定も同一パターンを再利用すればよい。
   - `apply_benjamini_hochberg(p_values, q=0.05)`（L367-）: per-domain 20指標の多重比較補正に使用。

6. **直近の本走結果ファイル**: `ls -1dt results/*/ | head` の結果，最新は
   `results/20260919_005727/`（Iter58 本走，gap方式 T=0.29/max_k=4）。`results.jsonl` の各行は
   `probe_candidates` フィールドを持ち，全10ノードの `{node_id, domain, confidence,
   confidence_logprobs_mean}` を保持していることを実データで確認した（上記4参照）。**オフライン
   採点の rank_1 情報源としてこのファイルをそのまま使える**。ただし Iter58 の gap 方式（k∈{1,2,4}）
   の下で記録された `dispatched_domains` は「固定 k=2」の基準線（compound_domain_set_recall=0.345）
   とは異なる母集団なので，**本レバーの成功条件が参照する 0.345 の基準線には，固定 `dispatch_top_k=2`
   時代の `results/20260918_202613/results.jsonl`（Iter57，gap方式導入前）を使う必要がある**
   （`results/20260919_005727/` を使うと，比較対象が「gap方式 vs OvR方式」になってしまい，
   config.yml note が定義した比較（「固定k=2 vs 固定k=2+OvR並べ替え」）にならない）。
   `results/20260918_202613/results.jsonl` は Iter58 調査・実験フェーズで既に compound_domain_set_recall
   =0.345 と検証済み（`scripts/replay_dispatch_gap_policy.py` の再生値と実機値が完全一致）。

7. **`select_dispatch_targets()` の全呼び出し元**（d0004 §4 の教訓を踏まえ機械的に確認，
   `grep -rn "select_dispatch_targets(" --include="*.py" . | grep -v "/tests/"` を実行）:
   - `aggregator.py:28`（定義そのもの）
   - `node.py:214`（`run_ask_flow()`，実際の dispatch/回答生成に使う唯一の実行時経路）
   - `run_experiment.py:93`（`_run_one()`，metrics 記録用の**2箇所目の独立呼び出し**．
     Iter58 で「gap 引数を渡し忘れる」という第2の no-op バグがここで発生し `ea4f680` で修正済み。
     現在のコードは `gap_threshold`/`gap_max_k` を `node.py` と同一に渡している）。
   - `scripts/replay_dispatch_gap_policy.py`（docstring 内の言及のみ，実際には
     `select_dispatch_targets()` を呼ばず同等ロジックを Python で再実装している独立スクリプト）。
   - **呼び出し元は実質2箇所（`node.py` と `run_experiment.py`）で全数一致**。本レバーは
     第1イテレーションでは**この関数自体を全く呼ばず**（オフライン採点は
     `replay_dispatch_gap_policy.py` と同じ「独立再実装」パターンを踏襲する），実行時経路への
     配線は次々イテレーション（スキーマ変更・ユーザー確認後）に先送りされる設計（config.yml note
     に既に明記済み）なので，**今回は d0004 §4 型の「レバーを読むコードに到達しない」リスク自体が
     存在しない**（そもそも実行時コードを経由しないオフライン検証だから）。ただし，**将来の配線
     段階で `node.py` と `run_experiment.py` の**両方**を同時に変更しないと，Iter58 と全く同じ
     「記録される `dispatched_domains` だけが旧方式のまま」という第2種の no-op を再演するリスクが
     ある**点を，次々イテレーションの計画に申し送る必要がある。
   - `estimate_confidence_classifier()` の呼び出し元は `http_server.py:367` の1箇所のみ
     （grep で確認済み，テスト除く）。本レバーはこの関数も変更しない。

**分かったこと（Q3: 実験設計への示唆）**

1. **OvR ヘッドの具体的な訓練方法**: `sklearn.multiclass.OneVsRestClassifier(
   sklearn.linear_model.LogisticRegression(max_iter=1000, class_weight="balanced"))` を
   `classifier_train.jsonl` の 1427 行（embedding は `scripts/train_domain_classifier.py` の
   `build_training_features()` を再利用して再計算，nomic-embed-text，同一 embedding_model）で fit する。
   `class_weight="balanced"` は**各ラベルの二値問題ごとに独立**に計算されるため（`OneVsRestClassifier`
   は内部で `n_labels` 個の独立な estimator を fit する），Iter32 で判明した
   「`sample_weight` と `class_weight_` の乗算結合」バグは構造的に発生しない（各ラベルの重みが
   他ラベルの行数分布に一切依存しないため）。**embedding は既存の `domain_classifier.joblib` が
   使うものと全く同じ生成経路（同一 `embedding_model`）だが，モデル自体は再利用できない**
   （`domain_classifier.joblib` は `CalibratedClassifierCV(LogisticRegression)` の単一マルチクラス
   分類器であり，`OneVsRestClassifier` の内部構造とは別物のため，学習済みパラメータの流用は不可能．
   **再訓練対象は「新規ヘッドのみ」で，embedding 計算そのものは1427件について1回再実行が必要**）。
2. **オフライン採点の具体的な手順**（新規ファイル2つを想定）:
   - `scripts/train_dispatch_candidate_ranking_head.py`: 上記1のとおり訓練し，
     `models/dispatch_candidate_ranking_head.joblib` として保存する（**`domain_classifier.joblib`
     とは別ファイル**．rank_1 用モデルへの上書き・混同を防ぐため命名を明確に区別する）。
   - `scripts/evaluate_dispatch_candidate_ranking.py`: `scripts/evaluate_classifier_calibration.py`
     と同じ構造（`--dataset data/dataset.jsonl --ollama-host ... --output results/....jsonl`）を
     踏襲し，1600 問について (a) query を embed（1回のみ，OvR ヘッドのスコア計算用），
     (b) 新設 OvR ヘッドの `predict_proba` で10ドメイン分の sigmoid スコアを得る，
     (c) **rank_1 は `results/20260918_202613/results.jsonl` の該当行の `probe_candidates` から
     argmax を取得**（既存分類器を再度呼ばず，確定済みの記録値をそのまま使うことで「rank_1 が
     文字どおり不変である」ことを実装上も保証する），(d) rank_1 以外の9ドメインを OvR sigmoid
     スコア降順に並べ，最上位を rank_2_new とする，(e) `dispatched_domains_new = {rank_1, rank_2_new}`
     を出力に書き込む（固定 k=2，コスト中立）。
3. **統計検定の具体的な手続き**（`metrics.py` 既存関数のみ再利用，Iter58 と同一パターン）:
   - 主基準: `(row_id, expected_domain)` の 200 ペアについて，baseline
     （`results/20260918_202613/results.jsonl` の `dispatched_domains`）と new（上記4の出力）
     それぞれで被覆されたかを `dict[str, bool]` にし，`metrics._mcnemar_from_correctness()` へ
     直接投入する（Iter58 実験フェーズと全く同じ呼び出し方，コード追記不要）。
   - `compute_compound_coverage_metrics()` を new 側の合成 `results` リスト
     （baseline の `expected_domains` 等はそのまま，`dispatched_domains` だけ new に差し替え）に対して
     呼び，`compound_domain_set_recall` を直接得る。
   - 非退行: rank_1 は定義上不変なので `top1_accuracy` の再計算・McNemar は不要（Iter58 の申し送り
     「rank_1 不変が構造的帰結」と同じ論理）。**ただし実装ミスの検出のため，一度だけ
     `sum(new_row["dispatched_domains"][0] == baseline_row["dispatched_domains"][0]
     for id) == 1600`（rank_1 が全行で一致すること）を assert 的に確認するチェックを
     採点スクリプトに含めるべき**（もし不一致があれば「rank_1 を再計算してしまっている」実装バグ）。
4. **レバー発火の証拠フィールド（d0004 §4 対策，過去の no-op 反復への警戒）**: 本レバーは
   argmax（rank_1）を変えないため，「発火した証拠」は**rank_2 の選出ドメインが新旧で何%異なるか**
   （rank2_flip_rate）でしか観測できない。旧方式の rank_2 は「同じ確率分布内で2番目に高い
   softmax 確率のドメイン」，新方式の rank_2 は「OvR ヘッドの独立 sigmoid スコアで最大のドメイン」
   であり，学習アルゴリズムも訓練目的関数も異なるため，**rank2_flip_rate が 0% に近い場合は
   実装が旧ロジックへフォールバックしている（発火していない）ことを疑うべき**というしきい値を
   計画フェーズで明記しておくとよい（目安: Iter58 の「発火の証拠は dispatched_domains 長の分散」
   と同じ発想で，今回は「rank_2 ドメインの分布が変わったか」を確認する）。
   `results/20260918_202613/results.jsonl` の2位 confidence 分布（Iter58 調査節で実測済み，
   平均0.291・中央値0.200の gap）を踏まえると，rank2_flip_rate は 0%〜100% のどこに落ちても
   ありうるため，事前の期待値を置かず，**実測してそのまま報告する**方針が妥当。
5. **legal ドメインへの追加の留保**: compound 100 行のうち legal を含む行が 30 件と最多
   （medical 28，education 20 が続く）である一方，legal の訓練データは 77 件と全ドメイン最小。
   OvR ヘッドの legal 二値分類器の質が本レバーの改善幅を左右しやすいため，考察フェーズでは
   「compound recall の改善が legal 絡みの行に偏っていないか」をドメインペア単位で内訳確認する
   ことを推奨する。

**次の計画フェーズへの示唆**

1. **単一レバー・スキーマ変更なしで着手可能**: 新規ファイル2つ（訓練・採点）の追加のみで，
   既存の `aggregator.py`・`node.py`・`run_experiment.py`・`config.yaml` は一切変更しない。
   `select_dispatch_targets()` の呼び出し元（`node.py`・`run_experiment.py` の2箇所）も触らない
   ため，Iter58 で発生した「2箇所目の呼び出しへの引数追加漏れ」型の no-op は今回は構造的に
   起こり得ない（実行時経路自体を通らないため）。
2. **基準線データは `results/20260918_202613/results.jsonl`（固定 `dispatch_top_k=2`，Iter57）を
   使うこと**。最新の `results/20260919_005727/`（Iter58 gap方式）を誤って基準線に使うと，
   比較対象がずれる（gap方式 vs OvR方式になってしまう）ので要注意。
3. **実施コストは埋め込み計算のみ**（訓練1427件＋評価1600件，合計3027件の embed 呼び出し，
   `scripts/evaluate_classifier_calibration.py` の実績から数分〜十数分で完了見込み，実機の
   probe/dispatch/LLM生成トラフィックは一切発生しない）。
4. **rank_1 が全行で完全一致することの機械的な確認**を採点スクリプトの必須ステップとして計画に
   明記すること（発火の誤検出・過検出どちらも防ぐため）。
5. **legal ドメイン（訓練77件，compound内訳最多30件）の内訳確認**を考察フェーズの必須項目として
   計画時点で申し送ること。

**出典一覧**
- scikit-learn, "1.16. Probability calibration" §1.16.3.3 / §1.16.3.4:
  https://scikit-learn.org/stable/modules/calibration.html
- Stack Overflow, "Multiclass classification: probabilities and calibration":
  https://stackoverflow.com/questions/60110209/multiclass-classification-probabilities-and-calibration
- MetricGate, "One-vs-Rest vs One-vs-One Classifier":
  https://metricgate.com/blogs/one-vs-rest-vs-one-vs-one-classifier
- futureagi.com, "What Is Learning to Rank? Definition & Methods (2026)":
  https://futureagi.com/glossary/learning-to-rank
- MDPI, "MultiLTR: Text Ranking with a Multi-Stage Learning-to-Rank Approach" (2025):
  https://www.mdpi.com/2078-2489/16/4/308
- Zhang & Zhou, "Binary Relevance for Multi-Label Learning: An Overview", Frontiers of Computer
  Science (2017): http://palm.seu.edu.cn/zhangml/files/FCS'17.pdf
- Wikipedia, "Multi-label classification": https://en.wikipedia.org/wiki/Multi-label_classification
- scikit-learn, `OneVsRestClassifier` API reference:
  https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html

### 計画 (Iter59)

**仮説**

compound 設問で2つ目の正解ドメインが上位に来ないのは，各ノードの confidence が
「10クラス softmax の自分のクラスの確率」＝単一ラベル構成の副産物であり，2位以降の順位が
多ラベル的な関連度を表していないためである（Iter58 実測: 2つの正解が上位2位に両方入るのは
3/100 行のみ，2つ目の正解のランク中央値5〜6位）．frozen embedding の上に各ドメイン独立の
sigmoid（OvR / binary relevance）ヘッドを置き，**rank_2 以降の順位付けだけ**をそのスコアで
差し替えれば，dispatch コストを一切増やさない（固定 k=2，mean dispatch 2.0）まま
`compound_domain_set_recall` が基準線 0.345 から改善する．

**単一レバー**

`dispatch_candidate_ranking`: `softmax_second_highest`（現行＝既存分類器の softmax 確率の降順で
2位を選ぶ）→ `multilabel_binary_relevance_head`（既存分類器の argmax を rank_1 に固定したまま，
残り9ドメインを新設 OvR sigmoid ヘッドのスコア降順に並べ，その最上位を rank_2 とする）．

固定する構成（直近最良から変更しない）:
- embedding: `nomic-embed-text`（frozen，再訓練なし）
- rank_1 分類器: `models/domain_classifier.joblib`（`CalibratedClassifierCV(temperature)` ＋
  education `intercept_delta=+0.7`）を**一切変更しない**
- `aggregator.py` / `node.py` / `run_experiment.py` / `classifier.py` / `config.yaml`: 無変更
- dispatch コスト: 固定 k=2（本イテレーションでは gap 方式を使わない．基準線も固定 k=2 の
  `results/20260918_202613/results.jsonl`）

**変更するファイルと箇所（最小差分・新規2ファイルのみ）**

1. `scripts/train_dispatch_candidate_ranking_head.py`（新規）
   - `scripts/train_domain_classifier.py` の `_load_training_rows()` と
     `build_training_features()` を import して再利用（embed 経路を完全共用．
     `_extract_sample_weights()` は**使わない**＝各二値問題に独立な
     `class_weight="balanced"` を使うため．Iter32 の `sample_weight × class_weight_` 結合バグは
     OvR では構造的に起こらない）
   - `OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight="balanced"))` を
     1427 行で fit し `models/dispatch_candidate_ranking_head.joblib` へ保存
     （**`models/domain_classifier.joblib` は上書きしない**．別ファイル名で並存させる）
   - CLI は `train_domain_classifier.py` と同じ体裁（`--train-data`, `--ollama-host`,
     `--ollama-port`, `--embedding-model`, `--output`）
   - 訓練時に 5-fold CV の per-domain ROC-AUC / average precision を stderr に出力する
     （legal 77件の過学習診断用．訓練セット丸暗記の検出）
2. `scripts/evaluate_dispatch_candidate_ranking.py`（新規）
   - テンプレート: `scripts/evaluate_classifier_calibration.py`（Iter29 以降のオフライン採点の定型．
     LLM生成・probe・dispatch トラフィックは発生させず embedding だけ実機ノードへ問い合わせる）
   - 入力: `--baseline results/20260918_202613/results.jsonl`（1600行，全行 k=2 を実測確認済み），
     `--head models/dispatch_candidate_ranking_head.joblib`，`--ollama-host`，
     `--embedding-cache results/iter59_query_embeddings.npz`（あれば再利用，無ければ計算して保存．
     再採点のたびに 1600 回 embed し直さないためのキャッシュ）
   - 手順: (a) 各行の query を embed，(b) ヘッドの `predict_proba` で10ドメインの sigmoid スコア，
     (c) **rank_1 は baseline 行の `dispatched_domains[0]` をそのまま採用（既存分類器を再計算しない）**，
     (d) rank_1 を除く9ドメインをヘッドスコア降順に並べ最上位を rank_2_new，
     (e) `dispatched_domains = [rank_1, rank_2_new]` の新 results リストを出力
   - 出力: `results/iter59_ovr_ranking_predictions.jsonl`（baseline 行の `id` /
     `expected_domains` / `selected_domain` はそのまま引き継ぎ，`dispatched_domains` のみ差し替え，
     加えて `head_scores`（10ドメイン分）と `rank2_baseline` / `rank2_new` を診断用に保持）

**レバーが読まれるコード行と到達条件（d0004 §4 対策）**

本イテレーションは実行時経路（`node.py:214` / `run_experiment.py:93` の
`select_dispatch_targets()` 呼び出し2箇所）を**一切通らない**オフライン検証であるため，
Iter16/20/21/22/27/58 型の「レバーを読むコードに到達しない」no-op は構造的に起こり得ない．
代わりに，採点スクリプトが「正しく発火し，かつ rank_1 を壊していない」ことを機械的に保証する
以下3点を採点スクリプト内の必須アサーションとして実装する:

- A1（rank_1 不変）: `sum(new[i]["dispatched_domains"][0] == base[i]["dispatched_domains"][0]) == 1600`
  を assert する．1件でも不一致なら「rank_1 を再計算してしまっている」実装バグとして中断する．
- A2（コスト中立）: 全1600行で `len(dispatched_domains) == 2` かつ rank_1 ≠ rank_2（重複なし）を
  assert し，mean dispatch = 2.000000 を出力に記録する．
- A3（発火の証拠）: `rank2_flip_rate = mean(rank2_new != rank2_baseline)` を算出して報告する．
  **0.0% の場合はヘッドが旧ロジックへ退化している（例: ヘッドのスコアが softmax と同順）疑いが
  強く，実験を成立させず実装を見直す**．事前の期待値は置かず実測値をそのまま報告する
  （baseline の rank_2 分布は education 421 / business_economics 188 / … / legal 73 と偏っており，
  新旧で分布が変わるかも併記する）．

**実施方法（コマンド手順）**

```
# 1) OvR ヘッドの訓練（1427 件 embed，数分．LLM 生成なし）
uv run python -m scripts.train_dispatch_candidate_ranking_head \
    --train-data data/classifier_train.jsonl \
    --embedding-model nomic-embed-text \
    --ollama-host 192.168.15.100 \
    --output models/dispatch_candidate_ranking_head.joblib

# 2) 1600 問のオフライン採点（embed のみ，probe/dispatch/LLM なし）
uv run python -m scripts.evaluate_dispatch_candidate_ranking \
    --baseline results/20260918_202613/results.jsonl \
    --head models/dispatch_candidate_ranking_head.joblib \
    --embedding-model nomic-embed-text \
    --ollama-host 192.168.15.100 \
    --embedding-cache results/iter59_query_embeddings.npz \
    --output results/iter59_ovr_ranking_predictions.jsonl

# 3) 指標・検定（metrics.py の既存関数のみ使用．コード追加なし）
#    - metrics.compute_compound_coverage_metrics(new_rows) → compound_domain_set_recall
#    - (row_id, expected_domain) の 200 ペアの被覆 bool 2 本を作り
#      scipy.stats.binomtest(b, b+c, 0.5) で exact McNemar（Iter58 主基準と同一手続き）
#      併せて metrics._mcnemar_from_correctness() の連続性補正版も参考値として併記
```

**成功条件（事前登録，判定は全て αレベル固定・事後変更しない）**

S1（主基準）: `compound_domain_set_recall` が基準線 **0.345（69/200，実測で再検算済み）** から
  上昇し，ドメイン単位 n=200 の **exact McNemar（two-sided binomtest，α=0.05）で p < 0.05**．
S2（効果量の下限）: 点推定の上昇が **+0.04 以上（＝被覆ドメイン数 69 → 77 以上）**．
  微小だが有意，という判定を採用に倒さないための下限．
S3（コスト中立）: 全1600行で k=2，mean dispatch = 2.000（A2 の assert が通ること）．
  基準線と同一コストなので Iter58 のような「同コスト参照ポリシー」の追加走行は不要
  （k を伸ばす変種を併せて評価する場合に限り同コスト参照を必須とする，という config.yml の
  条件には該当しない）．
S4（発火の証拠）: `rank2_flip_rate > 0`．0 なら実験不成立として実装を見直す．

S1〜S4 全てを満たせば「採用（次々イテレーションで `config.yaml` へ配線，ただしスキーマ変更のため
ユーザー確認）」，S1 のみ不成立で S2 相当の上昇が見える場合は partial，S1・S2 とも不成立なら棄却．

**非退行条件（事前登録）**

N1（rank_1 完全不変）: A1 の assert（1600/1600 一致）が通ること．これが破れた実験結果は
  単一レバー原則違反として無効とする．
N2（top1_accuracy 不変）: N1 の帰結として定義上不変．念のため new 側 rows に対し
  `metrics.compute_top1_accuracy()` を再計算し，baseline と**完全一致（小数点以下すべて）**する
  ことを確認する（不一致なら実装バグ）．
N3（legal 非退行・過学習チェック）: legal は訓練データ最少（77件，他ドメインは各150件）で，
  compound 100 行のうち legal を含む行が 30 件と最多という不均衡構造にある．基準線の実測値は
  **legal 絡み行の被覆 20/60 ドメイン単位，うち legal 自身の被覆 8/30**．新方式でこの
  「legal 自身の被覆 8/30」が**下回らない**こと（≧8）を非退行条件とする．また，ヘッドの
  legal 二値分類器のスコアが定数的に退化していないこと（1600 行での legal スコアの標準偏差 > 0，
  かつ 5-fold CV の legal ROC-AUC を報告）を確認する．
N4（全体の恩恵の偏り確認）: 改善がある場合，その内訳をドメインペア単位（legal 絡み / medical
  絡み / その他）で分解して報告する．legal 絡みのみに改善が集中している場合は，
  「compound データの 30% が legal であることに依存した結果」として主張の強度を落とす．

**留保（考察フェーズへの申し送り）**

R1: OvR ヘッドのスコアは per-binary で較正されるだけで合計1にならない（scikit-learn 公式・
  Iter29/30 で確認済みの構造）．本設計では argmax 決定に使わずランキングにのみ使うため
  較正の破綻は決定に影響しないが，**得られたスコアを「確率」として対外記述しないこと**．
R2: 1600 問は Iter58 までと同一の in-sample データであり，ヘッドの訓練データ
  （`classifier_train.jsonl` 1427件）とは別集合だが，評価集合上でのハイパラ選択は行わない
  （`LogisticRegression(max_iter=1000, class_weight="balanced")` 固定，探索しない）ことで
  Iter58 の「T を in-sample 選定」問題を繰り返さない．
R3: 本イテレーションはオフライン完結であり，実機の dispatch 挙動・回答品質・レイテンシは
  測定しない．採用となった場合でも，実行時経路への配線（`node.py:214` と
  `run_experiment.py:93` の**両方**を同時に変更しないと Iter58 と同型の「記録される
  `dispatched_domains` だけ旧方式」という no-op を再演する）と `config.yaml` のスキーマ変更
  （`dispatch_ranking_model` 等）はユーザー確認を経て次々イテレーションで行う．
R4: 「1位は既存モデル固定，2位以降だけ別モデルで並べ替える」という非対称設計は，調査フェーズの
  検索範囲では文献上の直接の先例が無い（近いのは retrieve-then-rerank だが候補集合全体を
  再ランキングする）．安全性の根拠は文献ではなく本リポジトリのコードロジック（A1 の assert）に
  ある点を対外記述で明示する．
- Medium (John I. Douglas Marangon), "Multilabel and Multiclass Classification Made Simple with
  scikit-learn": https://medium.com/@johnidouglasmarangon/multilabel-and-multiclass-classification-made-simple-with-scikit-learn-6fa485b55b59

### 実装 (Iter59)

**変更したファイル（新規2件のみ，既存ファイルは無編集）**

1. `scripts/train_dispatch_candidate_ranking_head.py`（新規）: `scripts/train_domain_classifier.py`
   の `_load_training_rows()` / `build_training_features()` を import して再利用し，
   `data/classifier_train.jsonl`（1427件）を embed した上で
   `OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight="balanced"))` を fit する。
   `_extract_sample_weights()`（計画どおり不使用）。5-fold `StratifiedKFold` の out-of-fold
   `decision_function` から per-domain ROC-AUC / average precision を stderr に出力する診断機能を
   実装（`_print_per_domain_cv_diagnostics()`）。保存先 `models/dispatch_candidate_ranking_head.joblib`。
2. `scripts/evaluate_dispatch_candidate_ranking.py`（新規）: `--baseline` の各行の `query` フィールド
   （`results.jsonl` 自体が `query` を保持しているため，計画にあった `data/dataset.jsonl` への
   突合は不要と判明し省略——単一レバー原則を守りつつ計画より単純な実装で同じ入力を得られる差分）
   を embed し，OvR ヘッドの `decision_function()` にシグモイドを適用した非正規化スコア
   （`OneVsRestClassifier.predict_proba()` はマルチクラス単一ラベルの場合合計1に再正規化してしまう
   ため，調査フェーズが指摘した「合計が1にならない独立スコア」を保つには `decision_function` 経由が
   必須と判明——実装中に発見した計画との差分）で rank_2 を再計算する。rank_1 は `--baseline` 行の
   `dispatched_domains[0]` を**計算せずそのままコピー**（A1 を構造的に保証）。埋め込みキャッシュ
   `results/iter59_query_embeddings.npz`（id キー）を実装。`metrics.compute_compound_coverage_metrics()`
   を import して診断用に `compound_domain_set_recall` を stderr へ出力する（計画どおり，最終的な
   McNemar 検定・BH 補正は次の実験・分析フェーズに委ねる）。

**実行環境上の対応（実装作業中に判明，コードには影響なし）**

- このサンドボックスから `192.168.15.100`（wafl500）への直接到達性は無い（`ping`/直接 TCP ともに
  タイムアウト，`ip route` にも当該サブネットへの経路なし）。過去イテレーション（Iter29〜Iter53）
  と同一の確立済み手順（`ssh -fNT -L 11435:localhost:11434 wafl500`，`~/.ssh/config` の
  `ProxyJump wafl` 経由）でローカルポートフォワードを新規に張り直し，`--ollama-host 127.0.0.1
  --ollama-port 11435` で疎通を確認した上で実行した（`curl http://127.0.0.1:11435/api/tags` で
  200 を確認済み）。実機の probe/dispatch/LLM 生成トラフィックは一切発生させていない
  （embedding のみ）。トンネルは次フェーズでの再利用のため起動したまま維持している
  （PID 2086401，`ssh -fNT -L 11435:localhost:11434 wafl500`）。
- `build_training_features()`（`train_domain_classifier.py`，既存・無編集）が
  `fine_tuned_embed_model=None` でも無条件に `from sentence_transformers import SentenceTransformer`
  を実行する既存の実装上，`sentence-transformers` が仮想環境に入っていないと import エラーになる
  ことが判明したため，`uv sync --extra research`（`pyproject.toml` 既定義の optional dependency
  group）を実行して依存関係を追加した。コード変更ではなく，既存の `pyproject.toml` に定義済みの
  extra を有効化しただけである。

**実行結果**

1. 訓練（`uv run python -m scripts.train_dispatch_candidate_ranking_head ...`，1427件 embed）:
   5-fold CV per-domain ROC-AUC/average precision（stderr出力，抜粋）:
   `business_economics 0.8556/0.5191`，`computer_science 0.9130/0.6252`，
   `education 0.8663/0.4445`，`general 0.8981/0.6452`，`history_culture 0.9351/0.7328`，
   **`legal 0.9202/0.6219`（n=77，過学習・退化の兆候なし，むしろ全ドメイン中2番目に高いROC-AUC）**，
   `mathematics 0.9498/0.8053`，`medical 0.7645/0.3406`（全ドメイン中最低），
   `natural_science 0.8902/0.5653`，`social_science 0.8948/0.6161`。
   `models/dispatch_candidate_ranking_head.joblib` を新規保存（`models/domain_classifier.joblib`
   の mtime・md5（`b360ef8...`）は実行前後で不変であることを確認済み，上書きなし）。
2. 採点（`uv run python -m scripts.evaluate_dispatch_candidate_ranking ...`，1600行，
   embedding cache `results/iter59_query_embeddings.npz` 新規生成）:
   `results/iter59_ovr_ranking_predictions.jsonl`（1600行）を生成。

**A1〜A3 実測値（スクリプト内蔵アサーションに加え，独立コードで再検算し二重確認済み）**

- **A1（rank_1不変）**: 1600/1600 行で baseline の `dispatched_domains[0]` と完全一致（不一致0件）。
  スクリプト内の `_assert_rank1_unchanged()` は例外を投げず正常終了し，別途 Python で再検算した
  結果も mismatch=0 で一致。
- **A2（コスト中立）**: 全1600行で `len(dispatched_domains)==2`（重複長は`{2}`のみ），
  rank_1≠rank_2（重複0件），mean dispatch = **2.000000**。
- **A3（発火の証拠）**: `rank2_flip_rate = 0.356875`（1600行中571行で旧方式と新方式のrank_2が
  異なる）。0%ではないため警告は発火せず，実装が発火していることを確認した。

**診断用に得られた副次的な値（正式な統計検定は次フェーズの担当，事前登録された成功条件の判定は
行っていない）**: `compound_domain_set_recall = 0.35`（70/200，基準線0.345=69/200から+0.005，
`metrics.compute_compound_coverage_metrics()` をそのまま呼び出して算出，独自の再計算式は書いていない）。

**型/lint/テスト**

- `uv run ruff check scripts/train_dispatch_candidate_ranking_head.py
  scripts/evaluate_dispatch_candidate_ranking.py` → All checks passed。
- `uv run pytest -q`: 227 passed, 12 failed, 5 warnings（実装前に `git stash` で確認済みの
  **既存の13件の失敗**——`tests/test_build_dataset.py`（JMMLUアーカイブ内に
  `japanese_civics.csv` が無い環境依存のKeyError）と `tests/test_train_domain_classifier.py`
  （`CalibratedClassifierCV` オブジェクトに `classes_` 属性が無い，sklearn バージョン起因の
  既存不具合）——のうち12件がそのまま残存（本イテレーションの変更に起因せず，`uv sync --extra
  research` の副作用で `test_build_training_features_embeds_each_row_in_order` が1件追加で通る
  ようになった）。**新規2ファイルはどちらのテストファイルにも import されておらず，今回の変更が
  これらの失敗の原因でないことを `git stash` での再現テストで確認済み**。新規ファイルに対する
  ユニットテストは追加していない（`scripts/evaluate_classifier_calibration.py` や
  `scripts/replay_dispatch_gap_policy.py` など同種のオフライン採点用テンプレートにも既存テストは
  無く，本リポジトリの慣行と整合的な判断）。

**実験開始可否**: 実験フェーズが開始してよい状態である。`results/iter59_ovr_ranking_predictions.jsonl`
と診断用サマリ（stderr JSON: `{"n_rows": 1600, "mean_dispatch": 2.0, "rank2_flip_rate": 0.356875,
"compound_domain_set_recall": 0.35, "compound_rows_evaluated": 100}`）が揃っており，次フェーズは
`metrics._mcnemar_from_correctness()` を用いた主基準検定（S1）・効果量下限（S2）・legal非退行
（N3）等の事前登録済み判定に直接進める。ただし診断値の compound_domain_set_recall=0.35 は
S2 の効果量下限（+0.04，77/200以上）に遠く届いていない点は，実験・分析フェーズが検定前に
留意すべき事実として申し送る。

### 実験・分析(実行) (Iter59)

本フェーズは新規の実機トラフィック（probe/dispatch/LLM生成）を一切発生させていない。実装フェーズが
既に生成済みの `results/iter59_ovr_ranking_predictions.jsonl`（1600行）と基準線
`results/20260918_202613/results.jsonl`（1600行，固定 `dispatch_top_k=2`）に対し，`metrics.py` の
既存関数（`compute_compound_coverage_metrics`・`_mcnemar_from_correctness`・
`compute_top1_accuracy`）のみを import して事前登録済みの S1〜S4・N1〜N4 を機械的に判定した。
新規に自作した統計ロジックは，`(row_id, expected_domain)` の 200 ペアを作るペアリング処理と，
`scipy.stats.binomtest` による exact McNemar のクロスチェックのみ（Iter58 と同一パターン）。
新規スクリプト `scripts/compute_iter59_ranking_stats.py` を作成・実行し，結果を機械可読な
`results/iter59_stats.json` に保存した（実行コマンド:
`uv run python -m scripts.compute_iter59_ranking_stats --baseline
results/20260918_202613/results.jsonl --new results/iter59_ovr_ranking_predictions.jsonl
--output results/iter59_stats.json`）。**解釈・採否判断はこのフェーズでは行わない**（次の
analyst/reflector フェーズに委ねる）。

**前提の再確認（検定前）**: `compute_compound_coverage_metrics()` を基準線に対して独立に
再実行し，`compound_domain_set_recall = 0.345`（69/200），`compound_rows_evaluated = 100`
（→ 200 ペア）であることを確認した（事前登録値と完全一致，スクリプト内で assert 済み）。

**S1（主基準，ドメイン単位 n=200 のペア比較）**: `metrics._mcnemar_from_correctness()` に
baseline/new 双方の被覆 bool マップを渡した結果，改善（baseline非被覆→new被覆）= **10件**，
悪化（baseline被覆→new非被覆）= **9件**，discordant = 19，連続性補正chi2 = 0.0，
**p = 1.0**（連続性補正版）。`scipy.stats.binomtest(k=9, n=19, p=0.5, alternative="two-sided")`
による exact 版も **p = 1.0** で完全一致。**α=0.05 に対し p<0.05 を満たさず，S1 は FAIL**。

**S2（効果量下限）**: `new_compound_domain_set_recall = 0.35`（70/200，実装フェーズの診断値と
完全一致），基準線 0.345 からの差 **+0.005pt**。事前登録の下限 **+0.04pt（77/200以上）に届かず，
S2 は FAIL**。

**S3（コスト中立）**: 1600行全てで `len(dispatched_domains)==2`（分布 `{2: 1600}`），
rank_1≠rank_2（重複0件），`mean_dispatch = 2.000000`。**PASS**（独立再計算，実装フェーズの
assertion と一致）。

**S4（発火の証拠）**: `rank2_flip_rate = 0.356875`（1600行中571行）。独立再計算した値が
実装フェーズの速報値（0.356875）と完全一致。0%ではないため**PASS**。

**N1（rank_1完全不変）**: 1600/1600行で baseline の `dispatched_domains[0]` と完全一致
（不一致0件）。**PASS**。

**N2（top1_accuracy不変）**: `metrics.compute_top1_accuracy()` を new_rows に対し再計算した結果
`0.5975`（baseline と小数点以下完全一致）。**PASS**。ただし機械的に確認した重要な留保:
`evaluate_dispatch_candidate_ranking.py` の `build_new_rows()` は `selected_domain` フィールドを
baseline行から**そのままコピー**しており，新しい `dispatched_domains` から
`aggregator.select_best_dispatch_response()` を再実行して再導出したものではない。実機では
`select_best_dispatch_response()` は実際に dispatch されたノードの生成結果に依存するため，
この一致は「オフライン採点スクリプトがフィールドを保存しているだけ」であることを示すに留まり，
「rank_2 が変わっても実機の集約結果が不変である」ことまでは示さない（それを確認するには
オンライン再実行が必要で，本イテレーションの設計では意図的に行っていない）。

**N3（legal非退行）**: legal を含む compound ペアは30件（基準線の legal 自身の被覆 = **8/30**，
事前登録値と一致）。new 側の legal 自身の被覆 = **9/30**。8を下回っておらず**PASS**。

**N4（改善・悪化の内訳，ドメイン別分解，判定なし・報告のみ）**: compound 100行の内訳は
legal絡み30行（60ペア，legal∩medical重複12行を含む），medical絡み（legal除く）16行（32ペア），
その他54行（108ペア）。ペア単位の改善/悪化件数:

| カテゴリ | 対象ペア数 | 改善 | 悪化 | 不変 |
|---|---|---|---|---|
| legal絡み | 60 | 5 | 3 | 52 |
| medical絡み（legal除く） | 32 | 1 | 1 | 30 |
| その他 | 108 | 4 | 5 | 99 |
| 合計 | 200 | 10 | 9 | 181 |

（内訳合計は S1 の改善10件・悪化9件と一致，独立検算OK）。改善・悪化とも全カテゴリに分散しており，
legal絡みのみに改善が集中している様子は見られない。

**PASS/FAIL集計**: S1=FAIL，S2=FAIL，S3=PASS，S4=PASS，N1=PASS，N2=PASS，N3=PASS
（N4は判定なし・報告のみ）。**成功条件2/2がFAIL，非退行条件3/3・コスト条件1/1・発火条件1/1が
PASS**。内容面の解釈・採否判断（partial／棄却の別など）は次の analyst/reflector フェーズに委ねる。

**成果物**: `results/iter59_stats.json`（S1〜S4・N1〜N4 の実測値，機械可読），
`scripts/compute_iter59_ranking_stats.py`（新規，`metrics.py` 既存関数のみ再利用，
`uv run ruff check` all pass）。

### 分析(解釈) (Iter59)

**1. 独立検算の結果（`metrics.py` 既存関数のみ使用，不一致0件）**

`results/20260918_202613/results.jsonl`（基準線）と `results/iter59_ovr_ranking_predictions.jsonl`
（新方式）を読み直し，`results/iter59_stats.json` と journal 記載値を独立に再計算した．
**全項目が完全一致し，不一致は1件も無い**．

| 項目 | 独立再計算値 | stats.json |
|---|---|---|
| compound_domain_set_recall（基準線） | 0.345（69/200，rows=100） | 一致 |
| compound_domain_set_recall（新） | 0.350（70/200） | 一致 |
| compound_domain_jaccard_mean | 0.2400 → 0.2500 | （未記録，本フェーズで追加算出） |
| 主基準ペア比較（n=200） | 改善10 / 悪化9 / discordant 19 / chi2=0.0 / p=1.0 | 一致 |
| exact binomtest | p=1.0 | 一致 |
| rank2_flip_rate | 0.356875（571/1600） | 一致 |
| dispatched_domains 長分布 | `{2: 1600}`，rank_1≠rank_2 重複0 | 一致 |
| rank_1 不一致件数 | 0/1600 | 一致 |
| top1_accuracy | 0.5975 → 0.5975 | 一致 |

**2. S1・S2 FAIL は「検出できなかった」のではなく「効果が無い」——より高検出力の直接検証**

S1 の discordant はわずか19件で，これだけでは「弱い効果を検出できなかった（検出力不足）」
可能性を排除できない．実際，ペア差の近似 SE は sqrt(10+9)/200 = **2.18pt**，
95% CI は **+0.005 ± 0.043 = [-3.8pt, +4.8pt]** であり，事前登録の効果量下限 +4.0pt は
CI 上端の内側に入る（＝ +4pt の真の効果があっても本設計では検出しきれない）．
そこで，レバーの機序そのものを直接測る**より高検出力の検定**を追加で行った．

- **指標**: compound 100行それぞれについて，「rank_1 と異なる正解ドメイン」（計 **159 個**）が，
  rank_1 を除く 9 ドメインの中で**何位に来るか**を，旧方式（probe softmax confidence 降順）と
  新方式（OvR sigmoid スコア降順）で対応づけて比較した．
- **結果**: 平均順位 旧 **4.201** vs 新 **4.258**（差 +0.057，**新の方がわずかに悪い方向**），
  中央値はどちらも **4位**．1位的中は 旧28 / 新29，3位以内は 旧74 / 新75．
  順位が動いた対象は 89/159 で，**新が改善46 / 悪化43**．
  **Wilcoxon 符号順位検定 p=0.914**（n=159，非ゼロ差89）．
- **解釈**: 主基準の discordant 19 件よりはるかに多い 89 件の非ゼロ差をもってしても，
  順位の分布は新旧で区別がつかない．すなわち **S1/S2 の FAIL は検出力不足の帰結ではなく，
  「OvR ヘッドは2つ目の正解ドメインについて既存 softmax と同等以上の情報を持たない」という
  実体を反映している**と判断する．

**3. 「発火しているのに効かない」ことの機序 — ヘッドは別物だが同じだけしか知らない**

- **発火は本物**: rank2_flip_rate 35.7%（compound 行に限れば 31/100，単一ドメイン行 36.0%）で，
  rank_2 の**分布そのものが大きく変わっている**（education 421→142，legal 73→115，
  medical 150→213，business_economics 188→223）．実装上のフォールバックではない．
- **しかし精度は同等**: 単一ドメイン1500行で，OvR ヘッドの argmax 正解率は **0.610**，
  既存分類器（probe confidence の argmax）も **0.610** で一致する．
  ヘッドは既存分類器より弱くも強くもなく，**同等の判別力で誤り方だけが脱相関している**．
  その結果，rank_2 の入れ替えは「当たりを外れに変える」と「外れを当たりに変える」を
  ほぼ同数（改善10 / 悪化9，flip した compound 31行でも 旧9正解 / 新10正解）発生させ，
  正味の効果がゼロ付近に収束した．
- **根本原因（訓練データに多ラベル情報が存在しない）**: OvR ヘッドの訓練データは
  `data/classifier_train.jsonl`（1427件）で，**各行は単一ドメインラベルしか持たない**
  （調査フェーズ Q2-2 で確認済み）．binary relevance は「ラベル間の依存を見ない」ことが
  弱点だと文献（Zhang & Zhou 2017）が指摘するが，本件はそれ以前の問題で，
  **「ある質問が2つのドメインに同時に関連する」という事例を1件も学習していない**．
  したがって OvR ヘッドが学べるのは結局「この質問はどのドメインか」という単一ラベル信号の
  別分解にすぎず，2位以降の順位に新しい情報が入る経路が構造的に無い．
  訓練時 CV の ROC-AUC（legal 0.9202 等）が高いことは，この**単一ラベル判別**が
  うまく学習できている証拠であって，**「2つ目の正解を当てられる」ことの証拠ではない**
  （実測の2つ目の正解の中央順位は4位のまま）．Iter58 調査の「gap の compound 判別力は弱い
  （AUC 0.576）」という知見と同じく，**単一ラベル訓練由来の信号はどう加工しても
  多ラベル性を持たない**，というのが両イテレーションを貫く共通の構図である．
- **上限との距離**: 固定 k=2・rank_1 固定という条件下での recall の**オラクル上限は 0.705**
  （rank_1 が正解している compound 行は 41/100 で，残り1枠を必ず当てた場合 141/200）．
  実測 0.345→0.350 はこの上限の半分以下であり，**改善余地は大きいのに本レバーはその
  1/70 しか動かしていない**．なお rank_1 自体が compound 行で正解しているのは 41/100 に
  すぎず，**より大きなボトルネックは本レバーが意図的に凍結した rank_1 側にある**．

**4. N4 内訳にパターンは無い（偏りの証拠なし）**

legal絡み60ペア（改善5/悪化3），medical絡み32ペア（改善1/悪化1），その他108ペア（改善4/悪化5）．
**全カテゴリで改善と悪化がほぼ同数**であり，legal 絡みに改善が集中する懸念（計画 N4，
legal は訓練77件・compound 内訳最多30件）は実測されなかった．
N3（legal 自身の被覆 8/30→9/30）も +1件で，これは discordant 全体が19件しかないことを踏まえれば
ノイズ帯の変動である（legal 単独で有意性を主張できる母数ではない）．
**逆に言えば「legal のデータ不足が効果を潰した」という弁明も成り立たない**——
legal の CV ROC-AUC は 0.9202（全ドメイン中2位）で退化しておらず，
効果が出なかった原因を legal の少数性に帰することはできない．

**5. N2 の留保の評価 — 今回の判定を覆さない．さらに構造的に解消できる**

rc-experimenter の留保（`build_new_rows()` が `selected_domain` を baseline からコピーしており，
`aggregator.select_best_dispatch_response()` を再実行していない）は手続きとしては正しい指摘だが，
**本イテレーションの判定に対する影響は無い**．理由は3点．

1. **判定の論理上**: S1（主基準）と S2（効果量）は `dispatched_domains` のみから計算され，
   `selected_domain` に一切依存しない．N2 は非退行条件であり，仮に N2 が破れても
   FAIL 判定を PASS に転じさせることはできない．
2. **機序上，実機でも top1 は構造的に不変**: `aggregator.select_best_dispatch_response()`
   （aggregator.py:104-119）は `max(dispatch_responses, key=lambda r: r.confidence)` であり，
   ここでの confidence は **/probe 時に確定した既存分類器の confidence**（生成結果に依存しない）．
   rank_1 は定義上その argmax なので，**rank_2 に何を選んでも rank_1 が必ず勝つ**．
   実データでも基準線1600行すべてで `selected_domain == dispatched_domains[0]`（1600/1600），
   `selected_domain == dispatched_domains[1]` は0件であった．
   したがって「実機で再走すれば top1 が変わりうる」という懸念は，
   現行の `aggregation_method=max_confidence`（Iter47/48 で採用・レバークローズ済み）の下では
   成立しない．留保は「max_confidence 以外の集約（majority_vote / llm_judge）へ戻した場合に限り
   有効」と限定できる．
3. **例外条件も実データで空**: rank_1 の dispatch が失敗した場合のみ rank_2 が採用されうるが，
   基準線の `dispatch_failed` は **0/1600**，`used_fallback` も 0/1600 である．
   （ただし Iter58 では dispatch_failed が1件発生しており，将来0である保証は無い．
   実機配線時にはこの経路が唯一の top1 変動源になる．）

**結論: N2 の留保は軽微**．ただし，`answer_quality` / `end_to_end` / レイテンシは
本イテレーションでは一切測定していない（オフライン完結，計画 R3 のとおり）ため，
「実機で何も変わらないことを確認した」とは書けない．正確には
**「rank_2 の入れ替えは，現行の max_confidence 集約の下では実機でも採用回答を変えない」**
という機序上の帰結が示せる，という表現にとどめるべきである．

**6. 仮説との整合 — 仮説は明確に反証された**

計画の仮説は「2位以降の順位が多ラベル的関連度を表していないことが原因であり，
OvR ヘッドで rank_2 を差し替えればコスト中立のまま compound_domain_set_recall が改善する」だった．

- **整合した部分**: 「現行の2位以降の順位が多ラベル関連度を表していない」という**診断**は
  正しい（2つ目の正解の中央順位は4位のまま，オラクル上限 0.705 に対し実測 0.345）．
  また，設計の安全性（rank_1 完全不変・コスト完全中立）は事前の主張どおり構造的に達成された
  （N1 1600/1600，mean dispatch 2.000000）．
- **反証された部分**: 「OvR ヘッドならその多ラベル関連度を表せる」という**処方**は誤りだった．
  同一の単一ラベル訓練データから作った別分解は，順位付けを変えはするが情報を増やさない
  （Wilcoxon p=0.914）．

**7. 判定の確信度と，追加反復の要否**

- **確信度: 高**．理由は，(i) 決定論的なオフライン比較でランダム実行間ノイズが存在せず，
  独立検算で不一致0件，(ii) 主基準（discordant 19）だけでなく，より高検出力な順位レベルの
  検定（非ゼロ差89，Wilcoxon p=0.914）でも効果ゼロが支持され，(iii) 「ヘッドが壊れている／
  弱い」という代替説明が，単一ドメイン argmax 正解率の一致（0.610 vs 0.610）と
  legal ROC-AUC 0.9202 によって排除されるため．
- **同一設計での追加反復は不要**（決定論的なので同じ値が再現するだけ．Iter58 の学び2と同じ）．
- **仮に追加検証するなら**，確認すべきは以下のいずれかであり，いずれも本レバーの
  「同じ訓練データの別分解」という枠組みを**出る**必要がある:
  (a) **多ラベル訓練データの新規作成**（compound 相当の2ドメイン付き訓練事例）．
      これが無い限り，どんなヘッド構造でも2位の情報は増えない（本イテレーションの中心的学び）．
  (b) **rank_1 側の改善**（compound 行での rank_1 正解率 41/100 がより大きなボトルネック）．
  (c) k を増やす方向（Iter58 で検証済み．コスト増に見合う固有寄与は示せていない）．

**次フェーズ（rc-reflector）への申し送り**

- **棄却（rejected）が妥当と考えられる理由**: 事前登録の成功条件 S1・S2 がともに FAIL であり，
  計画文が定めた判定規則（「S1・S2 とも不成立なら棄却」）に機械的に該当する．
  加えて本フェーズの追加検証により，FAIL が検出力不足ではなく実効果ゼロであることが
  裏付けられた（Wilcoxon p=0.914）．partial に倒す材料（方向性のある効果量，特定ドメインでの
  一貫した改善）は見当たらない（+0.005pt，全カテゴリで改善≒悪化）．
- **ただし実験そのものは成立している**: S3・S4・N1〜N3 は全 PASS で，d0004 §4 型の no-op
  （レバーを読むコードに到達しない）ではない．**「実験不成立」ではなく「有効な陰性結果」**
  として記録すべきである．
- **`config.yaml` への配線（スキーマ変更・ユーザー確認）は行わないこと**．
  計画では「オフラインで所定の改善が確認できた場合にのみ次々イテレーションで配線」と
  条件付けており，その条件は満たされていない．
  `models/dispatch_candidate_ranking_head.joblib` は本番経路から参照されていない
  （`models/domain_classifier.joblib` は md5 不変・上書きなしを実装フェーズで確認済み）ため，
  ロールバック作業は不要である．
- **次レバー選定への最大の示唆**: 「単一ラベル訓練データ（1行1ドメイン）の加工では
  多ラベル性は生まれない」という制約が，Iter58（gap 信号，AUC 0.576）と Iter59（OvR ヘッド，
  Wilcoxon p=0.914）の2回連続で確認された．compound_domain_set_recall を本質的に動かすには，
  **多ラベル教師信号そのものを作る**（例: `build_dataset.py` の手作り複合設問と同じ手法で
  2ドメインラベル付き訓練事例を追加し，ヘッドを真の multi-label 問題として訓練する）か，
  **rank_1 側（compound 行で 41/100）を改善する**かのどちらかが必要である．

### 考察 (Iter59)

**判定: 棄却（rejected）．レバー `dispatch_candidate_ranking` はクローズ（収束）**

事前登録した成功条件 S1（主基準・exact McNemar p<0.05）・S2（効果量 +0.04pt 以上）がともに FAIL であり
（p=1.0，実測 +0.005pt），計画文が定めた判定規則「S1・S2 とも不成立なら棄却」に機械的に該当する．
journal の実験・分析節の数値を自分でも突き合わせて確認した（S1: 改善10/悪化9/discordant 19，
exact binomtest p=1.0，S2: 0.345→0.350）．`dispatch_candidate_ranking` は values が
`multilabel_binary_relevance_head` の単一値のみのため，本イテレーションでクローズとする．

**「実験不成立（invalid）」ではなく「有効な陰性結果」である根拠**

config.yml success_criteria (6) は「主要指標が基準線と完全一致し McNemar の不一致ペアが 0 件なら
invalid と判定せよ」と定めるが，本件は discordant 19 件・rank2_flip_rate 35.7%（571/1600）で
レバーは確実に発火している．コスト条件 S3（mean dispatch 2.000000）・発火条件 S4・
非退行条件 N1（rank_1 1600/1600 一致）・N2（top1 0.5975 完全一致）・N3（legal 8/30→9/30）は全 PASS．
d0004 §4 型の no-op ではない．

**棄却の機序（本イテレーションの中心的な学び）**

OvR ヘッドは既存分類器と「同等の判別力で誤り方だけが脱相関している」——単一ドメイン1500行での
argmax 正解率が 0.610 と 0.610 で完全一致し，2つ目の正解ドメインの順位比較（n=159，非ゼロ差89）でも
Wilcoxon p=0.914，平均順位 4.201→4.258 とむしろ微悪化方向だった．discordant 19 件だけでは
検出力不足を疑う余地が残るが，はるかに高検出力のこの順位レベルの検定でも効果ゼロが支持されるため，
**FAIL は検出力不足ではなく実効果ゼロ**と確定できる．
根本原因は `data/classifier_train.jsonl` が 1 行 1 ドメインで複合事例を 1 件も含まないことにあり，
「ある質問が 2 つのドメインに同時に関連する」という事例を学習していない以上，OvR という別分解を
与えても 2 位以降に新しい情報が入る経路が構造的に無い．訓練時 CV の ROC-AUC が高いこと
（legal 0.9202 等）は単一ラベル判別の学習成功を示すだけで，2つ目の正解を当てられる証拠ではない．

**Iter58 との共通構図 — 2回連続で同じ壁に当たった**

- Iter58: 既存 confidence の gap（1位-2位差）で compound を判別しようとした → AUC 0.576．
- Iter59: 既存 embedding ＋ 単一ラベル訓練データの OvR 再分解 → Wilcoxon p=0.914．

**「単一ラベル訓練データ（1行1ドメイン）の加工・後処理では多ラベル性は生まれない」**ことが
異なる 2 つの機序で独立に確認された．同じ枠組み（既存の単一ラベル教師信号を使い回す）での
3 度目の再挑戦は同じ壁に当たる公算が高く，行わない．

**ボトルネックの所在（次の設計の前提）**

固定 k=2・rank_1 固定という条件下での compound_domain_set_recall のオラクル上限は 0.705
（rank_1 が正解している compound 行が 41/100，残り1枠を必ず当てた場合 141/200）．
実測 0.345→0.350 はその半分以下であり，2 位枠の改善余地は大きい．一方で rank_1 自体が
compound 行で正解しているのは 41/100 に過ぎず，**本レバーが意図的に凍結した rank_1 側にも
同規模のボトルネックがある**．

**留保**

- N2（top1_accuracy 一致）は，採点スクリプトが `selected_domain` を基準線からコピーしているために
  生じた一致であり，実機での再導出ではない．ただし現行の `aggregation_method=max_confidence`
  （Iter47/48 でクローズ）の下では `select_best_dispatch_response()` が probe 時点の confidence の
  argmax を取るため rank_1 が構造的に必ず勝ち（基準線 1600/1600 で
  `selected_domain == dispatched_domains[0]`，`dispatch_failed` 0/1600），判定を覆さない．
  この留保は majority_vote / llm_judge へ戻した場合にのみ有効になる．
- 本イテレーションはオフライン完結であり，`answer_quality` / `end_to_end` / レイテンシは未測定．
  「実機で何も変わらないことを確認した」とは書けない．
- `config.yaml` への配線は行わない（計画が条件付けた改善が得られていない）．
  `models/dispatch_candidate_ranking_head.joblib` は本番経路から参照されず，
  `models/domain_classifier.joblib` は md5 不変（上書きなし）なのでロールバック作業は不要．

**次レバー: `multilabel_training_signal = synthetic_two_domain_training_examples`（新設，config.yml 末尾へ追記）**

skill の停止条件 1（journal/backlog の学びから次の有望なレバーを考案できるなら追記して継続）に従う．
既存 levers は実質試し切り（残る `conformal_prediction_true_class_qhat` は B88 で失敗見込み確定，
`post_hoc_langdetect_retry` は Iter55 で langdetect ja=100/100 のため改善余地なし）である．

Iter58・Iter59 が共通して指し示す欠落は「多ラベル教師信号そのものが存在しない」ことなので，
次は**教師信号を作る側**へ移る．評価用の複合設問 100 問（`build_dataset.py` の `_COMPOUND_QUESTIONS`）は
テストセットであり訓練に流用しない（リーク禁止）．代わりに，既存の単一ドメイン訓練行から
2 ドメインにまたがる訓練事例を新規生成し，**真の multi-label 問題として**ヘッドを訓練する．
Iter59 のインフラ（`scripts/train_dispatch_candidate_ranking_head.py`・
`scripts/evaluate_dispatch_candidate_ranking.py`・埋め込みキャッシュ）はそのまま再利用でき，
差分は訓練データの生成部のみ．基準線・成功条件・非退行条件も Iter59 と完全に揃えられるため，
**「ヘッド構造は同じで教師信号だけが違う」という Iter59 との直接対比が成立する**（これが
今回の陰性結果を，次の実験の対照群として活かす最も情報量の多い設計である）．
本レバーが不成立なら，compound 方向の改善余地は「rank_1 側の改善」か
「複合設問データセット自体の再設計（research_frontier）」に限られることになる．

次イテレーション名: 「2ドメイン訓練事例の新規生成による多ラベルヘッドの再訓練」．

## Iteration 58: adaptive_confidence_gapによる複合ドメインdispatchの動的化

### 調査 (Iter58)

**問い**
- Q1: adaptive gating（Huang et al. 想定，実際は Li et al., EMNLP 2023）・Expert Choice Routing
  （Zhou et al. 2022，Google Research）は gap 閾値 / top-k 動的化をどう設計しているか。
  固定値かデータ駆動の校正か。
- Q2: `aggregator.py` の `select_dispatch_targets()` 実装・呼び出し経路・`config.yaml` の現行
  `dispatch_top_k` の位置づけを踏まえ，gap 閾値をどこに・どのスキーマで追加するのが自然か。
- Q3: 本リポジトリの実データ（既存 `results.jsonl` の `probe_candidates`）を使って，gap 閾値による
  動的化が compound_domain_set_recall・単一ドメイン設問の dispatch コストに実際どう効くかを
  オフラインで再生（replay）し，T の妥当な探索範囲と副作用を定量化する。

**分かったこと（Q1: 先行研究の設計）**
- **Li et al., "Adaptive Gating in Mixture-of-Experts based Language Models", EMNLP 2023**
  （ACL Anthology 2023.emnlp-main.217，著者は Li/Su/Yang/Jiang/Wang/Xu — config.yml の
  「Huang et al.」表記は著者名の取り違えの可能性が高い．次イテレーションでの引用時は
  著者名を修正すること）。機序はトークン単位で「デフォルトは top-1 gating，正規化した
  top-1 ゲート値が `1 - T` を下回る（＝ top-1 に確信が集中していない）場合のみ top-2 gating
  へ昇格する」という**二値のエスカレーション**（3 位以降への拡張はしない）。
  閾値 T はタスクごとに ablation（4.6 節，Table 5）で決めており，**固定値の理論的最適解はなく，
  検証セット上でのグリッドサーチ**（QA タスクでは T=0.2 が最良，他タスクでは T=0.1 台）
  で選んでいる。T を上げるほど top-2 適用率が上がり FLOPs も増える trade-off を明記。
  出典: https://aclanthology.org/2023.emnlp-main.217 ,
  https://henryhxu.github.io/share/jiamin-emnlp23.pdf
- **別系統として PMC/Frontiers の survey が言及する「Huang et al. 2024」の閾値付き動的ルーティング**
  は，top-k をソートした活性化確率の**累積和が閾値 p を超える最小集合**を選ぶ方式（nucleus/top-p
  的な累積質量閾値）で，3 位以降への拡張を自然に許す点が Li et al. の二値方式と異なる。
  出典: https://pmc.ncbi.nlm.nih.gov/articles/PMC12558867 ,
  https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2025.1590994/pdf
  （**この区別が Q3 の結論に直結する**．config.yml の note は両方式を「Huang et al., EMNLP 2023」
  として一括りにしているが，実際は別の 2 論文の別機序である）。
- **Expert Choice Routing（Zhou et al. 2022, Google Research，arXiv:2202.09368）**は per-instance の
  confidence gap 閾値を使わない．経路が根本的に逆（expert が token を選ぶ）で，「capacity factor
  c（1 トークンあたり平均何 expert に届くか）」というグローバルなハイパーパラメータで k を
  間接的に制御する．学習時のみ有効な方式で，推論時に個別クエリごとの gap を見て k を決める
  今回の用途とは設計思想が異なる（k はトークンごとに emergent に 1〜4+ に分布するが，これは
  容量制約からの副産物であり，個々の confidence gap を明示的に閾値判定しているわけではない）。
  出典: https://research.google/blog/mixture-of-experts-with-expert-choice-routing ,
  https://arxiv.org/pdf/2202.09368
- **示唆**: T は先行研究でも「理論的に導出される値」ではなく，**検証データ上でのグリッドサーチ
  で選ぶもの**という点は一貫している。本プロジェクトには独立した検証セットがないため
  （1600 問が唯一のデータセット），T の選定は既存の 1600 行データ上でのオフライン再生
  （下記 Q3）で行い，本走で使う前に妥当性を確認するのが筋が良い。

**分かったこと（Q2: 実装箇所）**
- `select_dispatch_targets()`（`aggregator.py:28-67`）は `rank_1` を `confidence_threshold` で，
  `rank_2+` を `dispatch_candidate_threshold` でゲートしたのち `candidates[:top_k]` で固定 k を
  切り出すだけの純粋関数．gap ロジックを入れるならこの関数のシグネチャに `top_k` の代わりに
  （または加えて）`gap_threshold: float | None` を追加し，`rank_1.confidence - rest[0].confidence
  < gap_threshold` を満たすときだけ `top_k` を動的に決める分岐を関数内に追加するのが最小差分。
- **呼び出し経路は `http_server.py` ではなく `node.py:214-219`（`run_ask_flow()`）のみ**．
  `http_server.py` は `/probe`・`/dispatch` エンドポイント（受信側）を実装するだけで
  `select_dispatch_targets` を呼ばない．依頼元タスク文の想定（http_server.py 経由）は誤りであり，
  実際に変更すべき呼び出し元は `node.py` の 1 箇所のみ（`run_experiment.py` も `run_ask_flow` 経由で
  同じ関数を使うため，二重実装の心配はない）。
- `config.yaml` の現行 `dispatch_top_k: 2`（L57）は Iter47/48 で `aggregation_method=max_confidence`
  採用時に固定された値で，`confidence_threshold: 0.0`・`dispatch_candidate_threshold: 0.0`（L5, L10）
  と合わせて**現状は毎回無条件で上位 2 ノードへ dispatch している**（gap 判定は一切していない）。
  つまり現行本番設定は「常に k=2」であり，「常に k=1」ではない点に注意（後述 Q3 の解釈に直結）。
- **スキーマ変更案（複数）**:
  1. **`dispatch_gap_threshold: float | null` を新設**（推奨）。`null`（既定）なら現行どおり
     `dispatch_top_k` を固定値として使う後方互換パスを残し，値を入れたときだけ gap 判定で
     `top_k` を動的決定する分岐を有効化する。`dispatch_candidate_threshold` の導入パターン
     （Y2，既定値で後方互換）を踏襲でき，レビューしやすい。
  2. `dispatch_top_k` を `int` から `{base: int, gap_threshold: float, max_k: int}` のような
     構造体に変更する案もあるが，既存の `config.get("dispatch_top_k", 1)`（`node.py:217`）の
     単純な `int` 読み取り箇所を構造化パースへ書き換える必要があり，変更点が増える割に
     利点が薄い（非推奨）。
  3. 案1をベースに `dispatch_gap_max_k: int`（既定 2，現行と同じ上限）を併設し，gap が小さい
     ときにどこまで k を伸ばすかを明示的に制御できるようにする（Q3 の結論を踏まえると，
     この `max_k` の設計が成果を左右するため必須に近い）。
- いずれの案でも `select_dispatch_targets()` は純粋関数のまま保て，`node.py` 側で
  `config.get("dispatch_gap_threshold")` を読んで分岐を渡すだけで済む．分類器の再訓練は不要
  （依頼元の前提どおり）．

**分かったこと（Q3: 実データでのオフライン再生 — 本調査の主要な発見）**
- `results/20260918_202613/results.jsonl`（Iter57 本走，1600 行，`probe_candidates` に全 10 ノードの
  confidence を保持）を使い，`select_dispatch_targets` 相当のロジックを Python で再現し，
  gap 閾値方式を**追加の実機実験なしに**オフライン検証した（この検証手法自体を次回以降の
  T 探索の標準手順として推奨する）。
- **前提の再確認（固定 top_k のスイープ）**: `top_k=1` → compound_domain_set_recall
  **0.205**（構造的下限，note の 0.500 は「両方カバーできた行の割合」ベースの別集計；本指標
  `covered/total_expected` の定義（`metrics.py:190`）では 0.205），`top_k=2` → **0.345**
  （既知の報告値と完全一致，再現確認済み），`top_k=3` → 0.465，`top_k=4` → 0.575，
  `top_k=5` → 0.645，`top_k=10`（全ノード）→ 1.000（理論上限）。
- **決定的な発見1**: compound 100 行のうち，**2 つの正解ドメインが両方とも confidence 上位 2 位
  以内に収まっている行はわずか 3/100**。残り 97 行は「1 つは上位（多くは 1 位）に来るが，
  もう 1 つの正解ドメインは 3 位以降（中央値ランク 5〜6，最大 10 位）」という分布
  （expected domain の順位分布: 1位=41, 2位=28, 3位=24, 4位=22, 5位=14, 6位=23, 7位=14, 8位=8,
  9位=19, 10位=7）。**つまり現行 `dispatch_top_k=2` が compound_domain_set_recall=0.345 を
  達成できているのは，ほぼ全て「rank_1 が正解ドメインの片方に一致する」ことの寄与であり，
  rank_2 が 2 つ目の正解ドメインに一致するのは稀（100 行中せいぜい 十数〜数十行）**。
- **決定的な発見2（gap の compound 予測力は弱い）**: compound 行の gap（confidence 差）分布
  （平均 0.291，中央値 0.200）と単一ドメイン行の gap 分布（平均 0.355，中央値 0.306）は
  重なりが大きく，**gap を compound/single の分類スコアとして見た AUC は 0.576**
  （ランダム 0.5 に近い．rank_2 の生 confidence 値を使っても AUC 0.580 と同程度）。
  依頼元タスク文にある「2位confidenceの最大値0.4955，0.4→75件(4.7%)」等の分布は，
  compound 率（6.25%）と近い割合になる点で一見有望に見えるが，実際に「rank_2 confidence が
  高い行」と「compound な行」の対応は弱く，**多くの false positive（実は単一ドメインだが
  rank_2 が高い行）を含む**（precision は T をどこに置いても 7〜8% 程度で頭打ち）。
- **決定的な発見3（依頼元想定に近い二値エスカレーション方式のシミュレーション）**: Li et al.
  型の「既定 k=1，gap(1位,2位) < T のときだけ k=2 へ昇格」という方式を T=0.05〜0.40 で
  スイープしたところ，**どの T でも compound_domain_set_recall は現行の固定 k=2 基準線
  0.345 に届かない**（最良は T=0.35〜0.40 で 0.295，かつこの時点で単一ドメイン行の
  60.3% が既に k=2 へ昇格しており，現行の「常に k=2」とほぼ同じコストに近づいているのに
  性能は下回る）。T=0.05 では compound_set_recall=0.225（k=2 昇格率は compound/single とも
  約 12〜13%）と，現行基準線を大きく下回る。
  **原因**: 現行本番設定は「常に k=2」であり，二値エスカレーション方式は「基本 k=1，まれに
  k=2」に変えるものなので，gap が compound を正しく見分けられない限り，昇格率を上げるほど
  現行に近づくだけで超えられない（k=2 に固定した方が強い）。
- **示唆**: 依頼元の成功条件「compound_domain_set_recall が 0.345 から有意に改善」を，
  Li et al. 型の**二値**エスカレーション（k∈{1,2}）で満たすのは，本データでは**構造的に困難**
  （gap という信号自体の compound 判別力が弱いため）。改善の余地があるとすれば，
  Huang et al. 2024 型の**累積閾値／連鎖的エスカレーション**（gap が小さい限り k=3,4,...と
  伸ばし続ける）だが，これは同時に単一ドメイン行の平均 dispatch 数も押し上げる
  （T=0.15 で compound 側 mean_k=3.35 まで改善（recall 0.435）に対し，単一ドメイン側も
  mean_k=2.76 まで増加し，現行の一律 k=2（mean_k=2.0）より**コストが高くなる**）。
  つまり「compound recall 改善」と「単一ドメインのコスト削減」は，この gap 信号を使う限り
  **同時には成立しにくい**（依頼元の項目3が懸念する副作用は，まさにこのトレードオフとして
  データで裏付けられた）。

**次の計画フェーズへの示唆**
1. **T の探索範囲**: 文献（Li et al. EMNLP2023）は T=0.1〜0.2 を報告しているが，本リポジトリの
   gap 分布（compound 中央値 0.20，single 中央値 0.31）に照らすと，二値エスカレーション方式
   では T=0.15〜0.35 の範囲でグリッドサーチしても現行基準線 0.345 を超えないことが上記
   オフライン再生で判明済み。**T の探索自体は `results/20260918_202613/results.jsonl` の
   `probe_candidates` を使ってオフラインで再現可能**（追加の実機実験は不要）なので，rc-planner
   は本走を組む前にこの offline replay を「(a) オフライン検証」ステップとして計画に組み込み，
   期待される改善幅を事前に確認することを推奨する。
2. **単純な二値ゲート（k=1/2）では success_criteria「0.345 から有意に改善」を満たせない見込みが
   高い**（上記シミュレーション根拠）。選択肢:
   (a) 累積閾値／連鎖的エスカレーション（k を 3 以上まで動的に伸ばす，スキーマ案3の
   `dispatch_gap_max_k` を 2 より大きく設定）を採用しつつ，単一ドメイン側のコスト増を
   許容範囲として明示的に success_criteria に組み込む（例: 単一ドメイン平均 dispatch 数の
   上限を明記し，それを超えない範囲で compound recall 改善を狙う）。
   (b) 依頼元の成功条件を「compound_domain_set_recall の有意な改善」から「同等以上の
   compound_domain_set_recall を維持しつつ，単一ドメイン行の平均 dispatch 数を有意に削減
   （コスト最適化）」へ再定義する（gap 信号は compound 判別には弱いが，「rank_1 が圧倒的に
   確信を持っている単一ドメイン行」を見分ける程度の弱い分離能力（AUC 0.58 は compound
   flagging には弱いが，逆に「gap が大きい行は高確率で単一ドメイン」という片側の主張には
   使える可能性があり，この観点は未検証．必要なら次イテレーションで
   `single 側の gap が大きい行を k=1 に落とす際の false-negative率`（＝実は compound なのに
   k=1 にされてしまう行の割合）を追加検証すること）。
   どちらを採るかはユーザー判断が要る可能性があるため，rc-planner は A1/A2 のような形で
   選択肢を明示し，必要なら backlog へ登録すること。
3. **実装は `aggregator.select_dispatch_targets()` への `gap_threshold`（＋必要なら `max_k`）引数
   追加＋ `config.yaml` への `dispatch_gap_threshold`（＋ `dispatch_gap_max_k`）新設が最小差分**。
   呼び出し元は `node.py:214-219` の 1 箇所のみ（http_server.py は無関係）。
4. **config.yml の note の引用は次回修正が必要**: 「Huang et al., EMNLP 2023」は著者名の誤り
   （正しくは Li et al.）であり，かつ「累積閾値方式（Huang et al. 2024 系）」と「二値
   エスカレーション方式（Li et al. EMNLP2023）」は別機序なので，どちらを実装するかを
   rc-planner の計画時に明記すること。

### 計画 (Iter58)

**仮説**
現行の固定 `dispatch_top_k=2`（常に上位 2 ノードへ dispatch）を，**隣接ランク間の confidence gap による
連鎖的エスカレーション**（Huang et al. 2024 系の累積閾値型．Li et al. EMNLP2023 の二値エスカレーション
ではない）へ置き換えると，単一ドメイン設問では k を 1 に落として節約し，confidence が拮抗する複合ドメイン
設問でのみ k を 3 以上へ伸ばせるため，dispatch コストの増加を +20% 以内に抑えたまま
`compound_domain_set_recall` を 0.345 から有意に改善できる．

**単一レバー（何を何から何へ）**
- レバー: `dispatch_policy` = `adaptive_confidence_gap`（config.yml levers 記載，B90 でユーザー承認済み）
- 変更前: `config.yaml:dispatch_top_k: 2` による固定 k=2
- 変更後: `config.yaml` に **`dispatch_gap_threshold: float | null`（既定 null）** と
  **`dispatch_gap_max_k: int`（既定 2）** を新設し，`dispatch_gap_threshold` が非 null のときのみ
  下記の連鎖的エスカレーションで k を動的決定する（null なら従来どおり `dispatch_top_k` の固定値を使う
  完全な後方互換パス．`dispatch_candidate_threshold` 導入時（Y2）と同じパターン）．
- k の決定規則（`aggregator.select_dispatch_targets()` 内，confidence 降順ソート済み候補 `cs` に対して）:
  ```
  k = 1
  while k < max_k and (cs[k-1].confidence - cs[k].confidence) < T:
      k += 1
  ```
  **隣接ランク間の差**を見る点が肝である（rank_1 との差を見る変種は同一コスト帯で compound recall が
  一貫して劣ることをオフライン再生で確認済み．例: single mean_k≈1.93 のとき隣接差型 0.355 に対し
  rank_1 差型 0.32）．既存の `confidence_threshold` / `dispatch_candidate_threshold` によるゲートは
  変更せず，ゲート通過後の候補列に対して上式を適用する．

**変更するファイルと箇所（最小差分）**
1. `aggregator.py:28-67 select_dispatch_targets()` — 引数に `gap_threshold: float | None = None`,
   `gap_max_k: int = 2` を追加し，`candidates[:top_k]` の直前で上記ループにより `top_k` を動的決定する．
   純粋関数のまま保つ（テスト容易性のため）．
2. `node.py:214-219 run_ask_flow()` — `select_dispatch_targets()` 呼び出しに
   `gap_threshold=config.get("dispatch_gap_threshold")`,
   `gap_max_k=config.get("dispatch_gap_max_k", 2)` を追加する．**呼び出し元はここ 1 箇所のみ**
   （`http_server.py` は `select_dispatch_targets` を呼ばない．`run_experiment.py` も `run_ask_flow` 経由）．
3. `config.yaml` — `dispatch_top_k: 2`（L57）の直下に `dispatch_gap_threshold` と `dispatch_gap_max_k`
   を追記する（値は下記(a)で確定）．
4. `tests/` — `select_dispatch_targets` の既存テストに，(i) `gap_threshold=None` で従来と同一の結果に
   なること（後方互換），(ii) gap が T 未満のとき k が伸びること，(iii) `gap_max_k` で頭打ちになること，
   の 3 ケースを追加する．

**レバーが読まれるコード行と到達条件（d0004 §4 の反復失敗への対策，必須記載）**
- 読まれる行: `node.py:217-218` で `config.get("dispatch_gap_threshold")` を読み，
  `aggregator.py` の上記ループへ渡る．
- 到達条件: `run_ask_flow()` は fallback 判定より前に必ず通る経路であり，現行設定
  `confidence_threshold=0.0` / `dispatch_candidate_threshold=0.0` では rank_1 は常に適格・
  候補は常に 10 件あるため，**1600 問すべてでこのループに到達する**（Iter27 のような候補ゲートによる
  no-op は起き得ない）．
- **発火の証拠フィールド**: `results.jsonl` の `dispatched_domains` の**長さの分布**．現行基準線では
  全 1600 行が長さ 2 で固定である．発火していれば長さが 1〜max_k に分散する．
  rc-experimenter は本走前の予備実行（先頭 20 問）でこの分布を直接確認すること．
- **重要（analyst への申し送り）**: 集約は `max_confidence` であり，`select_best_dispatch_response()` は
  probe 時の confidence が最大の応答＝ rank_1 を返す．したがって **`selected_domain` / `top1_accuracy` は
  本レバーでは構造的に不変**（rank_1 の dispatch が失敗した行を除く）．
  **top1_accuracy が基準線と完全一致しても success_criteria (6) の「実験不成立(invalid)」とは判定しないこと**．
  invalid の判定は `dispatched_domains` 長が全行 2 で固定だった場合に限る．

**実施方法**
- (a) **オフライン再生による (T, max_k) の確定（実機不要，数分）**: `results/20260918_202613/results.jsonl`
  （Iter57 本走 1600 行，`probe_candidates` に全 10 ノードの confidence を保持）に対して上記 k 決定規則を
  再生し，`T ∈ {0.01,…,0.60}` × `max_k ∈ {3,4,5,6}` のグリッドで
  `compound_domain_set_recall`・単一ドメイン行の mean dispatch 数・全体 mean dispatch 数を算出する．
  **事前登録した選定規則**: 「単一ドメイン行の mean dispatch 数 ≤ 2.40 かつ全体 mean dispatch 数 ≤ 2.45」
  を満たす組のうち `compound_domain_set_recall` が最大のものを選ぶ（同値なら mean dispatch 数が小さい方）．
  計画フェーズでの予備再生では **`max_k=4, T=0.29〜0.30`** が該当し，
  compound_domain_set_recall 0.425〜0.430（基準線 0.345），単一ドメイン mean_k 2.37〜2.42，
  ドメイン単位ペア比較（n=200）で改善 28 件・悪化 11〜12 件，McNemar 正確検定 p=0.0095〜0.0166 が
  得られる見込みである．この値を再現できることを (a) の合格条件とする．
  再生スクリプトは `scripts/` 配下に `replay_dispatch_gap_policy.py` として残し，後続の T 再探索に使えるようにする．
- (b) 実装（上記 1〜4）．
- (c) 予備実行（先頭 20 問）で `dispatched_domains` 長の分散を確認．
- (d) 実機 1600 問を 1 回実行（約 100 分）し，`mise run analyze` で全指標を取得する．

**成功条件（事前登録）**
1. **主基準**: `compound_domain_set_recall` が基準線 0.345（Iter47 以降 `dispatch_top_k=2` 固定時の値）から
   改善し，ドメイン単位（n=200）のペア比較で McNemar 正確検定 p < 0.05 であること．
   目標値は (a) で確定した再生値（見込み 0.425 前後）．
2. **コスト条件**: 単一ドメイン行（1500 行）の mean dispatch 数 ≤ 2.40（基準線 2.0 に対し +20% 以内），
   かつ全体 mean dispatch 数 ≤ 2.45．
3. **再現性条件（実装検証）**: 実機実測の `compound_domain_set_recall` と
   `compound_mean_dispatched_count` が (a) のオフライン再生値と一致すること（probe の confidence は
   分類器出力で決定論的なため，本来は完全一致するはず）．**不一致が 1pt を超える場合は実装バグを疑い，
   採否判定より先に原因を特定する**．

**非退行条件**
1. `top1_accuracy` が非退行（McNemar p ≥ 0.05）．上記のとおり構造的に不変が期待値であり，
   **有意な変化が出た場合はむしろ実装の副作用を疑う**．
2. per-domain precision/recall 20 指標の BH 補正後，有意に悪化する指標が 0 件．
3. `answer_quality_accuracy`・`end_to_end_accuracy` の変化が 3SD = 2.6pt 以内
   （success_criteria (5)．max_confidence 集約のため本来ほぼ不変のはず）．
4. `mean_duration_ms` の増加が +25% 以内（dispatch は `asyncio.gather` で並列のため，k 増加は
   帯域・ノード負荷の増加であって直列な遅延増ではない．基準線 Iter57 の実測値と比較する）．

**留保（考察フェーズで必ず言及すること）**
- T の選定は独立した検証セットではなく評価集合 1600 問そのもの（Iter57 の probe 出力）の上で行う
  in-sample なグリッドサーチである．これは先行研究（Li et al. EMNLP2023 の ablation）と同じ手続きだが，
  汎化性能の主張はできない．
- gap の compound 判別力自体は弱い（AUC 0.576，調査節）．本方式が効くのは「gap が小さい行で k を伸ばす」
  弱い相関の積み上げによるものであり，**コスト中立（単一ドメイン mean_k ≤ 2.0）の範囲では
  compound_domain_set_recall は 0.37〜0.39 止まりで有意差に届かない**（予備再生で確認済み．
  p ≈ 0.175）．今回 +20% のコスト増を許容した判断の是非は backlog B91 に記録した．

### 実装 (Iter58)

**変更したファイル（計画どおり最小差分，目的外の変更なし）**
1. `aggregator.py` の `select_dispatch_targets()`（旧 L28-67）: 引数に
   `gap_threshold: float | None = None`, `gap_max_k: int = 2` を追加．既存のゲート処理
   （`confidence_threshold` / `dispatch_candidate_threshold`）はそのまま，`candidates = [rank_1] +
   qualified_rest` の直後に分岐を追加した．`gap_threshold is None` なら従来どおり
   `candidates[:top_k]`（変更前と完全に同一の戻り値）．非 None のときのみ計画どおりの
   隣接ランク差ループ（`k=1` から `k < min(gap_max_k, len(candidates))` かつ
   `candidates[k-1].confidence - candidates[k].confidence < gap_threshold` の間 `k += 1`）で
   `k` を決め `candidates[:k]` を返す．`gap_max_k` を候補数でクランプしているのは，ゲート通過後の
   候補が `gap_max_k` 未満しかない場合に `IndexError` を避けるため（計画書の擬似コードには
   明記されていなかった実装上の補完．純粋関数の性質・シグネチャの意味は変えていない）．
2. `node.py` の `run_ask_flow()`（`select_dispatch_targets()` 呼び出し，旧 L214-219）:
   `gap_threshold=config.get("dispatch_gap_threshold")`,
   `gap_max_k=config.get("dispatch_gap_max_k", 2)` の2キーワード引数を追加．他の引数・
   呼び出し順は変更なし．
3. `config.yaml`: `dispatch_top_k: 2`（L57）の直下に `dispatch_gap_threshold: null` と
   `dispatch_gap_max_k: 2` を追記．**計画書の指示どおり T・max_k の具体値はここでは書き込んで
   いない**（既定値は null のまま，後方互換パスを維持）．既存の無関係な未コミット差分
   （`central_router.embed_node_host: wafl502 → wafl-ctrl5`）には触れていない．
4. `tests/test_aggregator.py`: 計画の3ケースを `select_best_dispatch_response_returns_none...`
   の直前に追加した．
   - `test_select_dispatch_targets_gap_threshold_none_matches_fixed_top_k`: gap が僅差
     （0.9 vs 0.89）でも `gap_threshold=None` なら `top_k=1` の従来どおり1件のみ返ることを確認
     （後方互換）．
   - `test_select_dispatch_targets_gap_threshold_escalates_k_when_gap_small`: 隣接差
     0.05 < T=0.1 のとき k=1→2 へ伸び，次の隣接差 0.45 ≥ T で止まることを確認．
   - `test_select_dispatch_targets_gap_threshold_capped_by_gap_max_k`: 全隣接差が T 未満でも
     `gap_max_k=2` で頭打ちになることを確認．

**検証結果**
- `uv run pytest tests/test_aggregator.py -q`: 25 passed（既存22＋新規3）．
- `uv run pytest -q`（全体）: 223 passed, 2 skipped, 13 failed．失敗13件は全て
  `tests/test_build_dataset.py`（9件）と `tests/test_train_domain_classifier.py`（4件）に限られ，
  依頼元が事前に無関係と明記した既知の失敗（JMMLUデータ欠落・`CalibratedClassifierCV`に
  `classes_`属性が無い sklearn API不整合）と一致することを確認した．今回変更した
  `aggregator.py`・`node.py`・`config.yaml`・`tests/test_aggregator.py` に起因する失敗はない．
- `uv run ruff check aggregator.py node.py tests/test_aggregator.py`: All checks passed（`ruff`
  は YAML を Python として構文解析しようとするため `config.yaml` は対象外，`uv run python -c
  "yaml.safe_load(...)"` で構文の妥当性のみ別途確認済み．`dispatch_gap_threshold`/
  `dispatch_gap_max_k`/既存の `embed_node_host: wafl-ctrl5` とも意図どおり読める）．

**実験開始可否**: 実装は完了し既存テスト・新規テストとも green．**ただしこのまま実機本走へは
進めない**．計画フェーズが事前登録した (T, max_k) の確定手続きが未実施のため，次フェーズ
（rc-experimenter）は以下を先に行うこと．
- **(a) オフライン再生を先に実施すること**: `scripts/replay_dispatch_gap_policy.py` を新規作成し，
  `results/20260918_202613/results.jsonl` の `probe_candidates` を使って，計画フェーズの
  事前登録手続き（`T ∈ {0.01,…,0.60}` × `max_k ∈ {3,4,5,6}` のグリッド，選定規則「単一ドメイン行
  mean dispatch 数 ≤ 2.40 かつ全体 mean dispatch 数 ≤ 2.45 を満たす組のうち
  compound_domain_set_recall 最大」）に従って (T, max_k) を確定させる．計画フェーズの予備再生
  （`max_k=4, T≈0.29〜0.30` で compound_domain_set_recall 0.425〜0.430 見込み）を再現できるかを
  (a) の合格条件とする．**実機実験は不要**（このスクリプトは今回作成しておらず，次フェーズの
  最初のタスクとして残っている）．
- (a) で確定した T・max_k を `config.yaml` の `dispatch_gap_threshold` / `dispatch_gap_max_k` に
  反映してから，予備実行（先頭20問で `dispatched_domains` 長の分散を確認）→ 実機1600問本走へ
  進むこと．

### 実験 (Iter58)

**(a) オフライン再生による (T, max_k) の確定**

- 新規作成した `scripts/replay_dispatch_gap_policy.py` で，`results/20260918_202613/results.jsonl`
  （Iter57 本走 1600 行）の `probe_candidates` に対し，計画フェーズ事前登録の決定規則
  （`k=1` から `k < max_k` かつ隣接ランク差 `candidates[k-1].confidence - candidates[k].confidence
  < T` の間 `k += 1`）を再生した．集計は `metrics.py:compute_compound_coverage_metrics()` と同一定義
  （`covered_domain_count / expected_domain_total`，compound 行 = `len(expected_domains) > 1`）に
  合わせ，加えて単一ドメイン行（1500行）・全体（1600行）の mean dispatch 数を算出した．
  実行コマンド:
  ```
  uv run python scripts/replay_dispatch_gap_policy.py \
      --results results/20260918_202613/results.jsonl \
      --max-k-values 3,4,5,6
  ```
  （`--t-min/--t-max/--t-step`・`--*-cost-limit` は既定値のまま＝計画どおり
  `T∈{0.01,…,0.60}`(0.01刻み)，選定規則の閾値は単一ドメイン≤2.40・全体≤2.45）．
- **グリッド全体**: `max_k∈{3,4,5,6}` × `T`60点＝240組．選定規則（両コスト条件を満たす組のうち
  `compound_domain_set_recall` 最大，同値なら mean dispatch 数最小）を満たす組は **120/240**．
  `max_k` 別の（コスト条件下での）最良点は以下のとおりで，`max_k=4` が全 `max_k` の中で最良
  （他の `max_k` の最良点をいずれも上回る）:
  - `max_k=3`: `T=0.50` → recall 0.395，single_mean 2.3947，overall_mean 2.4025
  - **`max_k=4`: `T=0.29` → recall 0.425，single_mean 2.366，overall_mean 2.399375（グローバル最良）**
  - `max_k=5`: `T=0.22` → recall 0.415，single_mean 2.3367，overall_mean 2.370625
  - `max_k=6`: `T=0.19` → recall 0.42，single_mean 2.400，overall_mean 2.430625
  - 参考として `max_k=4` 近傍の T 感度（`T=0.25`〜`0.33`）:
    `T=0.25`→recall 0.395/single 2.180，`T=0.28`→0.410/2.313，**`T=0.29`→0.425/2.366（選定点）**，
    `T=0.30`→0.430/2.420（single_mean 2.420 > 2.40 の上限を超過し不適格），`T=0.33`→0.445/2.567
    （overall_mean 2.596 でさらに超過）．**`T=0.30` が僅差でコスト条件を超過するため，境界の
    `T=0.29` が選定される**（計画フェーズの「`T≈0.29〜0.30`」という幅はこの境界のことを指す）．
- **選定 (T, max_k) = (0.29, 4)**．根拠: 上記事前登録の選定規則（両コスト条件下で
  `compound_domain_set_recall` 最大）を機械的に適用した結果，`max_k=4, T=0.29` が
  `eligible_count=120` 組の中でグローバル最良（recall 0.425，かつ他の `max_k` の最良点
  0.395/0.415/0.420 をいずれも上回る）だった．
- **(a) の合格条件（計画フェーズの予備再生 `max_k=4, T≈0.29〜0.30` で recall 0.425〜0.430，
  単一ドメイン mean_k 2.37〜2.42 を再現できること）との対比**: recall 0.425 は範囲内で一致，
  単一ドメイン mean_k 2.366 は範囲下限 2.37 よりわずかに小さい（-0.004pt）が，計画フェーズの
  値が「予備再生」（概算）であるのに対し今回はグリッド全点を機械的に再計算した確定値であり，
  乖離幅も無視できる小ささのため実装バグの兆候とは判断しない．**(a) は合格**とみなし，
  T・max_k の確定手続きを完了した．
- 全グリッド生データ（TSV，241行）は本メッセージには含めない（`scripts/replay_dispatch_gap_policy.py`
  を同一引数で再実行すればいつでも再現可能，決定論的）．再現コマンドは上記のとおり．

**(b)〜(d) 実装コミット・config反映・予備実行，および予備実行で発見した第2のno-opバグ**

- コミット `b9df0b8`（🔀 Iter58: dispatch_policy=adaptive_confidence_gap実装，T=0.29/max_k=4を
  オフライン再生で確定）: `aggregator.py`・`node.py`・`tests/test_aggregator.py` の計画どおりの
  差分と，`config.yaml` への `dispatch_gap_threshold: 0.29` / `dispatch_gap_max_k: 4`（(a)の確定値）
  の反映，`scripts/replay_dispatch_gap_policy.py` の新規追加を含む．
  既存の無関係な未コミット差分（`central_router.embed_node_host: wafl502→wafl-ctrl5`，
  `results/iter45_preliminary/logs/` 配下）は意図的に除外（`git add -p` で該当hunkのみ選択）．
- `mise run setup`（image digest `sha256:0d5dbb77...`，git HEAD=`b9df0b8`）→ `mise run deploy`
  （全10ノードhealthy，smoke check git-status/hashes/probe全pass）．
- **(d) 予備実行（先頭20問，`results/20260919_004600/results_prelim20.jsonl`）で
  `dispatched_domains` 長の分布を確認したところ，20/20行すべてが長さ2で固定**（分散していない）．
  計画フェーズの到達条件チェック「発火していれば長さが1〜max_kに分散する」に抵触したため，
  本走前に原因を特定した．
- **原因（第2のno-opバグ，config到達性ではなくコード重複由来）**: `select_dispatch_targets()`
  の呼び出し箇所は `node.py:214`（`run_ask_flow()`，実際のdispatchに使われる．gap引数を正しく渡す）
  だけでなく，**`run_experiment.py:85`（`_run_one()`）にも独立した2箇所目の呼び出しがあった**
  （調査フェーズの「呼び出し経路はnode.py 1箇所のみ」という記述は誤りだったと判明．
  `run_experiment.py` は `run_ask_flow()` を呼んで実際のdispatch/回答生成は行うが，
  `dispatched_domains`/`probe_candidates`（metrics.py が読む集約用フィールド）は
  同じ `probe_responses` から**別途もう一度** `select_dispatch_targets()` を呼んで再計算しており，
  この2箇所目の呼び出しが `gap_threshold`/`gap_max_k` を渡していなかったため，実際のdispatchは
  gap方式で動いているのに，記録される `dispatched_domains` だけが旧来の固定 `dispatch_top_k=2`
  にフォールバックしていた（gap12=0.34・0.306・0.428・0.299（いずれもT=0.29超）の行でも
  長さ2が記録されていたことから特定，`business_economics-006/008/010/020` 等）．
  **回答生成（`selected_domain`/`confidence`/`answer_text`）自体は正しくgap方式で行われており
  影響を受けていない．影響を受けるのは `dispatched_domains` から導出される
  `compound_domain_set_recall`・mean dispatch数などmetrics.py側の集計のみ**．
- **修正**: `run_experiment.py:85` の `select_dispatch_targets()` 呼び出しに
  `gap_threshold=config.get("dispatch_gap_threshold")`, `gap_max_k=config.get("dispatch_gap_max_k",
  2)` を追加（`node.py` と同一の2引数．純粋関数のシグネチャ・ロジックは無変更）．
  検証: `uv run pytest tests/test_run_experiment.py tests/test_aggregator.py -q`
  （31 passed）・`uv run ruff check run_experiment.py`（all pass）．
- **教訓（次回のrc-plannerへの申し送り）**: 「`select_dispatch_targets` の呼び出し元は1箇所」
  という調査フェーズの結論は，grepの対象を運用コードパス（`node.py`）だけに絞ったために
  ベンチマーク実行スクリプト（`run_experiment.py`）内の**メトリクス記録専用の重複呼び出し**を
  見落としたことが原因．今後同種のレバーを扱う際は `grep -rn "関数名("` をテストディレクトリ以外
  の全 `.py` に対して行い，呼び出し元の数を機械的に確認すること．
- 修正後，`mise run setup`（image digest `sha256:9ab0c4c3...`，git HEAD=`ea4f680`）→
  `mise run deploy`（全10ノードhealthy，1回のリトライ後にhealthy化，smoke check全pass）で
  再デプロイし，**同一20問（`results/20260919_005614/results_prelim20b.jsonl`）で予備実行を
  再実行**したところ，`dispatched_domains` 長は `{1: 5件, 2: 1件, 4: 14件}`（mean 3.15，
  business_economicsドメイン20問という偏った小標本のため，全体基準の単一ドメイン
  mean 2.366より高いが，business_economicsは既知の低confidence分離ドメインであるため
  方向として妥当）と**1〜max_k(4)に分散し，修正前の全行長さ2固定から明確に変化**．
  隣接ランク差から手計算した期待値（例: business_economics-004 gap12=0.2947>T=0.29→k=1，
  business_economics-006 gap12=0.34>T=0.29→k=1，business_economics-001 gap12=0.1177<T→k=2かつ
  gap23=0.3176≥T→k=2で停止）と実際の出力が全て一致することを個別に確認した．**(c)(d)の
  合格条件（発火の証拠＝長さの分散）を修正後に達成**．本走へ進む．

**(e) 本走（1600問）とメトリクス取得**

- 実行コマンド: `mise run start --dataset data/dataset.jsonl --output results.jsonl`
  （`mise run analyze` は既知の「最新ディレクトリ」辞書順解決バグ（Iter57で既報）のため
  `mise run analyze 20260919_005727` と明示指定で実行）．
  結果: `results/20260919_005727/results.jsonl`（1600行，実行時間 約44分，開始00:57:27〜完了
  01:41付近）．git HEAD=`ea4f680`（run_experiment.py修正後），image digest `sha256:9ab0c4c3...`．
- **`dispatched_domains` 長分布（1600行全体）**: `{1: 815件, 2: 58件, 4: 727件, 3: 0件}`．
  mean = (815×1+58×2+727×4)/1600 = **2.399375**．k=3がゼロ件だったのは，一度隣接gapがTを
  下回り始めると（gap分布の性質上）後続の隣接gapも連続して小さいままになりやすく，
  途中で止まらず`max_k=4`まで到達する行が多いためと考えられる（考察フェーズで検討要）．
  `dispatch_failed`は1件（`social_science-100`，`dispatched_domains=['social_science']`の
  単一ターゲットへの`/dispatch`呼び出し自体が失敗．レバーのロジックとは無関係な
  ノード側の一過性障害）．`used_fallback`は0件．
- **`mise run analyze`実行中に第2のバグを発見**: `scripts/evaluate_response_quality.py`
  （`tasks.analyze`が呼ぶ）が`dispatch_failed`行（`answer_text=None`）で
  `TypeError: expected string or bytes-like object, got 'NoneType'`をraiseしクラッシュした．
  原因は`evaluation.py:compute_answer_quality_accuracy()`の
  `extract_answer_letter(result.get("answer_text", ""))`が`or ""`ガードを欠いており，
  `.get(key, default)`はキー自体が無い場合のみdefaultを返す（値が`None`のときは`None`を
  そのまま返す）という基本的な誤り．同じモジュール内の唯一のもう1つの呼び出し元
  （`scripts/evaluate_response_quality.py`の`_run()`）は既に`or ""`で正しくガードしていた
  ため非対称だった．**レバーとは無関係の既存バグ**（`dispatch_failed`行が実質存在しなかった
  過去の全実行では踏まれなかった経路）．
  修正: `evaluation.py:73`に`or ""`を追加．回帰テスト
  `test_compute_answer_quality_accuracy_treats_none_answer_text_as_incorrect`を追加．
  検証: `uv run pytest tests/test_evaluation.py -q`（20 passed）・`uv run ruff check
  evaluation.py tests/test_evaluation.py`（all pass）．この修正はローカルのみで完結する
  スクリプト（`scripts/evaluate_response_quality.py`はコンテナではなくホストで実行）のため
  再デプロイ不要．修正後に`uv run python -m scripts.evaluate_response_quality --results
  results/20260919_005727/results.jsonl --dataset data/dataset.jsonl`を再実行して取得．
- **主要メトリクス（`uv run python metrics.py --results results/20260919_005727/results.jsonl
  --json`，および axis23 は上記コマンド）**:
  | 指標 | 基準線（Iter57, `results/20260918_202613/`, dispatch_top_k=2固定） | 本走（Iter58） |
  |---|---|---|
  | top1_accuracy | 0.5975 (956/1600) | 0.596875 (955/1600) |
  | compound_domain_set_recall | 0.345 (69/200) | **0.425 (85/200)** |
  | compound_mean_dispatched_count | 2.0 | 2.9 |
  | single_domain_mean_dispatch（1500行，`dispatched_domains`長平均） | 2.0 | 2.366 |
  | overall_mean_dispatch（1600行） | 2.0 | 2.399375 |
  | ECE | 0.0549 | 0.054626 |
  | answer_quality_accuracy | 0.569333 | 0.558 |
  | end_to_end_accuracy | 0.335 | 0.326875 |
  | mean_duration_ms | 1491.9 | 1502.156875（axis23側1501.68，`rows_with_dispatch_timing`
    1600→1599の差はdispatch_failed行の欠測による） |
  | dispatch_failure_rate | 0.0 | 0.000625 (1/1600) |
  | fallback_rate | 0.0 | 0.0 |
  - compound domain別内訳（`compound_coverage`）: covered=85/expected=200
    （baseline covered=69/200，`metrics.py`の`compute_compound_coverage_metrics`を
    `results/20260918_202613/results.jsonl`に対して再実行して確認）．
- **(a)オフライン再生との再現性照合（成功条件3）**: `compound_domain_set_recall`=0.425，
  `single_domain_mean_dispatch`=2.366，`overall_mean_dispatch`=2.399375は
  いずれも(a)の再生値と**完全一致（小数点以下差分0）**．**成功条件3（再現性）は合格**．
- **統計的検定（`metrics.py`の既存関数のみ使用，continuity-corrected McNemar／Fisher正確検定，
  独立再計算可能）**:
  1. **主基準: compound_domain_set_recallのドメイン単位ペア比較（n=200，
     `(row_id, expected_domain)`ペアごとに「baseline/newそれぞれの`dispatched_domains`に
     含まれるか」を対応のある2値として`metrics._mcnemar_from_correctness()`へ渡した）**:
     改善（baselineで非被覆→newで被覆）= **28件**，悪化（baselineで被覆→newで非被覆）=
     **12件**，discordant=40，chi2=5.625，**p=0.017706**．計画フェーズの事前予測
     （改善28件・悪化11〜12件，p=0.0095〜0.0166）と改善/悪化件数は完全一致，pは範囲より
     わずかに大きいが同オーダー．**α=0.05でp<0.05のため主基準は合格**．
  2. top1_accuracy McNemar（`compute_mcnemar_test`）: discordant_a_only=1（baseline正解→new
     不正解，`social_science-100`のdispatch_failedによるもの），discordant_b_only=0，chi2=0.0，
     **p=1.0**．非退行条件1（p≥0.05）**合格**．構造的に不変という事前予想どおり．
  3. per-domain 20指標（10ドメイン×recall/precision，`compute_domain_recall_mcnemar_test`／
     `compute_domain_precision_fisher_test`）＋BH補正（`apply_benjamini_hochberg`, q=0.05）:
     **BH補正後有意 0/20**．非零のdiscordantはsocial_science recall/precisionのみ（同じ
     `social_science-100`由来，p=1.0）．非退行条件2**合格**．
  4. answer_quality_accuracy 0.569333→0.558（**-1.13pt**），end_to_end_accuracy
     0.335→0.326875（**-0.81pt**）．いずれも3SD=2.6pt以内．非退行条件3**合格**．
  5. mean_duration_ms 1491.9→1502.156875（**+0.688%**）．+25%以内．非退行条件4**合格**．
- **コミット**: `b9df0b8`（実装＋(a)確定値のconfig反映）→`ea4f680`（run_experiment.pyの
  gap引数欠落no-op修正）→`735eb25`（予備実行分散確認の記録，journal.mdのみ）．
  本メッセージ確定後，evaluation.pyのNoneガード修正・axis23再取得・本走メトリクスの
  journal追記をまとめて次コミットで記録する．
- **すべて機械的な事前登録済み基準に対する結果**: 主基準1件・コスト条件2件・再現性条件1件
  （成功条件，計4項目）と非退行条件4項目の**計8項目すべてPASS**．内容面の解釈・採否判断は
  次フェーズ（analyst/reflector）に委ねる．

### 分析(解釈) (Iter58)

**1. 独立検算の結果（`metrics.py`・`evaluation.py`の既存関数のみ使用，`results/20260918_202613/`
vs `results/20260919_005727/`）**

- 実験フェーズの報告値・`results/20260919_005727/metrics.json`・`axis23_metrics.json` と
  **全項目が完全一致**．**不一致は1件も無い**．検算した値:
  `compute_compound_coverage_metrics`: covered 69/200→85/200，recall 0.345→0.425，
  compound_mean_dispatched_count 2.0→2.9，jaccard_mean 0.2400→0.2317．
  `dispatched_domains` 長分布 1600行 `{2:1600}` → `{1:815, 2:58, 4:727}`（k=3 は 0 件），
  single_mean 2.0→2.366（n=1500），overall_mean 2.0→2.399375．
  `compute_top1_accuracy` 0.5975→0.596875，`compute_mcnemar_test` a_only=1/b_only=0/chi2=0.0/p=1.0．
  `compute_mean_duration_ms` 1491.9→1502.156875（+0.688%）．
  `compute_domain_recall_mcnemar_test`×10 ＋ `compute_domain_precision_fisher_test`×10 ＋
  `apply_benjamini_hochberg(q=0.05)`: **BH補正後有意 0/20**（非零discordantは
  social_science recall/precision のみ，いずれも `social_science-100` の dispatch_failed 由来で p=1.0）．
  `compute_answer_quality_accuracy` 0.569333→0.558（-1.13pt．gradable 1500行分母を再確認: 854/1500→837/1500）．
- 主基準のドメイン単位ペア比較も独立に再構成（`(row_id, expected_domain)` 200ペアを
  `metrics._mcnemar_from_correctness()` へ投入）し，**改善28件・悪化12件・discordant 40・
  chi2=5.625・p=0.017706** を再現．
- **成功条件3（再現性）の独立再確認**: `scripts/replay_dispatch_gap_policy.py` を同一引数で再実行し，
  `eligible_count=120 / grid_size=240`，選定 `(T, max_k)=(0.29, 4)`，
  recall 0.425・single 2.366・overall 2.399375 を再現．**実機実測3値と小数点以下まで完全一致**．
  さらに本走の `probe_candidates` から固定 top_k の recall を再計算したところ
  k=1..5 で 0.205/0.345/0.465/0.575/0.645 と調査フェーズ（Iter57データ由来）の値に完全一致した．
  これは**probe分類器出力が2実行間で完全に決定論的**であることの直接的証拠であり，
  実装の正しさの強い保証になる．**成功条件3 PASS**．

**2. 事前登録基準の判定（独立検算後）**

| # | 条件 | 基準 | 実測 | 判定 | 余裕 |
|---|---|---|---|---|---|
| 成功1 | compound_domain_set_recall 改善＋McNemar p<0.05 | p<0.05 | 0.345→0.425，exact p=0.016589（連続性補正版 p=0.017706） | **PASS** | 中（後述の脆弱性あり） |
| 成功2a | 単一ドメイン mean dispatch | ≤2.40 | 2.366 | **PASS** | 0.034 |
| 成功2b | 全体 mean dispatch | ≤2.45 | 2.399375 | **PASS** | 0.051 |
| 成功3 | 実機＝オフライン再生 | 一致（差1pt以内） | 差0（完全一致） | **PASS** | 最大 |
| 非退行1 | top1_accuracy | McNemar p≥0.05 | p=1.0（discordant 1件） | **PASS** | 最大 |
| 非退行2 | per-domain 20指標 BH補正後悪化 | 0件 | 0/20 | **PASS** | 最大 |
| 非退行3 | answer_quality / end_to_end 変化 | ≤2.6pt (3SD) | -1.13pt / -0.81pt | **PASS** | 1.5pt / 1.8pt |
| 非退行4 | mean_duration_ms 増加 | ≤+25% | +0.688% | **PASS** | 大 |

**8項目すべて PASS（独立検算でも同一結論）**．

**3. 主基準 p=0.0177 と計画フェーズ事前予測 p=0.0095〜0.0166 の乖離の原因 — 構造的差ではなく検定の変種違い**

- 事前予測の範囲 0.0095〜0.0166 は，**exact binomial McNemar** における (改善28,悪化11)=0.009475 と
  (28,12)=0.016589 の両端に一致する（手計算で確認）．実測は (28,12) であり，
  **exact p=0.016589 は予測レンジの上端と完全一致**．実験フェーズが報告した 0.017706 は
  `metrics._mcnemar_from_correctness()` の**連続性補正つき正規近似**の値であり，
  同じデータに対する別の検定統計量にすぎない（差 +0.0011）．
- したがって「pが予測よりわずかに大きい」のは**ノイズでも構造的差でもなく，
  exact と連続性補正近似の使い分けの差**である．発見された2件のバグ修正の影響でもない
  （後述4のとおり，両バグとも `dispatched_domains` の値自体には影響していない）．
  なお事前登録文の表記は「McNemar 正確検定」だが，実装は連続性補正版である．
  **どちらの値でも α=0.05 を下回るため判定は変わらない**が，今後の事前予測では
  どちらの変種で書くかを統一すべきである（backlog候補）．
- **有意性の脆弱性**: discordant 40 件のうち exact p が 0.05 を割るのは (28,12)→0.0166，
  (27,13)→0.0385 まで．**(26,14) で p=0.0807 となり非有意化する**．
  すなわち**ドメインペア2件が反転すれば有意性は失われる**．n=200 かつ効果量+8.0pt に対して
  この脆弱性は小さくないため，「有意」という結論は**境界的**と位置づけるのが妥当である．
- **ペアの行内相関の懸念は実測上ない**: 200ペアは100行由来なので独立性が疑われるが，
  行単位（n=100）で被覆数の増減を数えると **改善28行・悪化12行・同数60行**とペア単位と
  完全に同数で，「1行で2ペアとも動いた」ケースは0件だった．行単位の符号検定でも
  exact p=0.016589 と同値であり，**相関によるp値の過小評価は起きていない**．

**4. 発見された2件のバグの扱い（いずれもレバー効果の解釈を汚染しない）**

- **バグ1（`run_experiment.py:85` の `select_dispatch_targets()` 重複呼び出しにgap引数欠落）**:
  本走（`ea4f680`）より前に修正済みで，本走データは影響を受けていない．
  **基準線（Iter57, `20260918_202613`）への影響も無い**: 当時 `dispatch_gap_threshold` 自体が
  存在せず，重複呼び出しも `node.py` も同じ固定 `dispatch_top_k=2` パスを通るため，
  記録値と実挙動は一致していた（全1600行が長さ2で整合）．**前後比較の妥当性は損なわれていない**．
- **バグ2（`evaluation.py:73` の `or ""` 欠落）**: `answer_text=None` となる `dispatch_failed` 行でのみ
  発現する既存バグ．基準線には該当行が0件のため基準線値は不変，本走は1行（`social_science-100`）が
  不正解として計上される．影響は最大 1/1500 = **0.067pt** であり，answer_quality の -1.13pt の
  うち説明できるのは 6% 未満．**採否判定を左右しない**．
- 両バグともレバー（gap方式のk決定）そのものとは独立であり，**8項目の判定には影響しない**．

**5. answer_quality -1.13pt / end_to_end -0.81pt はノイズと判定（構造的副作用ではない）**

- **根拠(a) 経路上ほぼ不変**: 集約は `max_confidence` で rank_1 が選ばれるため，
  `selected_domain` は **1600行中1行しか変化していない**（その1行も dispatch_failed 由来）．
  一方 `answer_text` は **494/1600 行で異なる**．同一ノード・同一プロンプトで文面だけが変わっており，
  これは**LLM生成の非決定性**そのものである．「k増加で複数ノードの回答が競合し集約結果が変わる」
  という機序は，selected_domain がほぼ完全に不変である以上**成立していない**．
- **根拠(b) k別の層別で差が偏っていない**: answer_quality の前後差を実測 k で層別すると
  k=1（n=815）-0.98pt，k=2（n=58）-1.72pt，k=4（n=727）-1.10pt と**ほぼ一様**．
  もし k 増加が原因なら k=4 層に偏るはずだが，k を減らした k=1 層でも同程度下がっている．
- **根拠(c) 検定**: answer_quality を行単位でペア化した McNemar は
  discordant 155（baseline のみ正解86 / new のみ正解69），**exact p=0.199**．有意でない．
- **根拠(d) 過去実行のばらつき**: LoRA適用後の比較可能な直近7実行の answer_quality は
  0.5467/0.5500/0.5680/0.5553/0.5607/0.5693/0.5580 で **SD=0.85pt**（レンジ2.27pt），
  end_to_end は直近6実行で **SD=0.74pt**（レンジ2.00pt）．
  今回の -1.13pt は **1.3SD**，-0.81pt は **1.1SD** に相当し，明確にノイズ帯である
  （事前登録の 3SD=2.6pt という見積りも実測SD 0.85pt と整合しており，妥当だった）．
- 結論: **両指標の低下はノイズ**．考察フェーズで「回答品質が下がった」と読むべきではない．

**6. 仮説との整合性 — 半分整合，半分は反証**

計画フェーズの仮説は「**単一ドメイン設問では k を 1 に落として節約し**，confidence が拮抗する
複合ドメイン設問でのみ k を 3 以上へ伸ばせるため，**dispatch コスト増を +20% 以内に抑えたまま**
compound_domain_set_recall を有意に改善できる」であった．

- **整合した部分**:
  - compound_domain_set_recall の有意改善（0.345→0.425，exact p=0.0166）は達成．
  - dispatch 総呼び出し数 3200→3839 = **+19.97%** で，仮説の「+20%以内」を満たした
    （ただし**余裕は0.03%しかなく，実質的に上限ぎりぎり**）．
  - gap 信号は弱いながら compound を識別している: k=4 へエスカレートした割合は
    **compound行 62.0% vs 単一ドメイン行 44.3%**（比 1.40）．調査フェーズの AUC 0.576 と整合する
    弱い分離であり，「効いてはいるが弱い」という事前の見立てどおり．
- **反証された部分**:
  - 「単一ドメインでは k=1 に落として節約」は**成立していない**．単一ドメイン1500行の内訳は
    k=1:781 / k=2:54 / **k=4:665** で，平均は 2.0→**2.366（+18.3%）**と**増加**した．
    781行の節約を665行のk=4昇格が上回っている．コストは「節約」ではなく「compound側へ
    再配分しつつ全体で純増」した．なお事前登録のコスト条件（≤2.40）はこの増加を
    織り込んだ上限だったため基準判定には影響しない．
  - 「k を 3 以上へ伸ばす連鎖的エスカレーション」も，**k=3 は1600行中0件**で実体がない．
    機序を検証したところ，gap12<T かつ gap23<T を満たす727行において
    **gap34 の最大値が 0.2357 で T=0.29 を一度も超えない**（平均 0.054，p95 0.147）．
    confidence の裾は一度平坦域に入ると隣接差が T まで戻らないため，
    **T=0.29 のもとで本方式は事実上 k∈{1,4} の二値ポリシー**（k=2 は gap12<T かつ gap23≥T の
    58行のみ）に退化している．**max_k が主要な制御変数，T は分割比率の制御変数**という
    理解が実態に合う．

**7. 追加分析（事前登録外）— 改善はどこまで「gap信号の手柄」か**

- 固定 top_k の性能・コスト境界（本走 `probe_candidates` で再計算）は
  k=1:0.205 / k=2:0.345 / k=3:0.465 / k=4:0.575 / k=5:0.645．
- 本方式の全体コスト 2.399375 に**コストを揃えた**参照ポリシー（各行をランダムに 39.94% の確率で
  k=3，残りを k=2 とする混合）を2000シードでシミュレートすると，
  recall の平均は **0.3929（SD 0.0119）**．本方式の 0.425 はこれを **+3.21pt** 上回る．
- ただしこの +3.21pt を本方式とペア比較すると，exact McNemar の **p の中央値は 0.405**，
  **p<0.05 で本方式が勝つシードは 0.1%** にとどまる．
- **解釈**: 主基準で得られた有意な改善（0.345→0.425）の大部分は「dispatch を約20%多く投げた」
  ことによるものであり，**gap 信号そのものの寄与（+3.2pt）は方向としては正だが
  本サンプルサイズでは有意に示せない**．「同コストなら固定kの混合より良い」とは
  現時点のデータでは主張できない．これは採否判定の中心論点になる．
- 補助的所見: `compound_domain_jaccard_mean` は **0.2400→0.2317 とわずかに低下**した．
  recall が上がったのは集合を広げたためで，dispatch 集合の的中の質（集合一致度）は
  改善していない．

**8. 留保（計画フェーズの留保の再確認と追加）**

- **in-sample 選定**: (T, max_k) は評価集合1600問そのもの（Iter57のprobe出力）上の
  240点グリッドサーチで選ばれ，本走も**同一1600問**で実施された．したがって本走は
  「独立した確証実験」ではなく，実質的には**実装の再現性検証**である
  （事実，3指標が小数点以下まで一致した）．主基準の p 値はグリッド探索の多重性に対して
  補正されていない．**汎化性能の主張はできない**．
  緩和材料として，recall はコスト（T）に対して単調・滑らかに増加しており
  （max_k=4 で T=0.25→0.395, 0.28→0.410, 0.29→0.425, 0.30→0.430, 0.33→0.445），
  ノイズの尖りを拾った選定ではなく「コスト上限内で最大のTを選んだ」だけであることは確認した．
- **判定の確信度**: 主基準の有意性は境界的（2ペア反転で非有意化），かつ効果の大半が
  コスト増で説明できる．追加反復を行う場合は**同一設定の再実行では意味がなく**
  （決定論的で同じ値が出る），ホールドアウト分割か新規評価問題の追加が必要である．

**次フェーズ（rc-reflector）への示唆**

- 事前登録基準は**8/8 PASS**で，機械的判定としては「採用可」．独立検算での不一致は0件．
- ただし採否の実質的論点は次の3つである．
  1. 有意性が**境界的**（2ペア反転で失効）で，かつ**同一データ上の in-sample 選定＋同一データでの
     確認**なので，統計的主張の強度は「探索的知見」相当にとどまる．
  2. 改善の大半は**コスト+19.97%の純増**で説明でき，gap信号固有の寄与（+3.2pt）は有意でない．
     「固定 `dispatch_top_k=3`（recall 0.465, コスト+50%）」という単純な代替に対する
     優位性も，コスト制約を外せば明確ではない．採否は「compound recall +8pt に
     dispatch 呼び出し +20% を払う価値があるか」という**運用上のトレードオフ判断**になる．
  3. 実装が T=0.29 で **k∈{1,4} の二値に退化**している（k=3 が0件）ことは，
     「連鎖的エスカレーション」という設計意図と実態の乖離である．採用する場合でも
     この事実を仕様として明記すべきで，より小さい max_k（=3）での再探索や，
     gap ではなく累積確率質量（top-p型）を使う変種は次レバーの候補になり得る．
- 非退行は**すべて実質的に問題なし**（top1は構造的不変，per-domain 0/20，品質2指標は
  1.1〜1.3SDのノイズ，レイテンシ+0.7%）．品質低下を採否の減点材料にしないこと．
- 記録すべきバグ修正2件はいずれもレバーと独立で，結果解釈を汚染していない．

### 考察 (Iter58)

**判定: partial（条件付き採用，確信度 中）．レバー `dispatch_policy` は収束（クローズ）．**

- **採用した範囲**: 本番設定として `config.yaml` の `dispatch_gap_threshold: 0.29` /
  `dispatch_gap_max_k: 4` を**維持する**（ロールバックしない）．根拠は次の3点．
  1. 事前登録した8項目（主基準1・コスト条件2・再現性条件1・非退行4）が独立検算でも
     全PASSで，不一致0件．非退行は実質的にも問題がない（top1は構造的不変，per-domain 0/20，
     answer_quality -1.13pt・end_to_end -0.81pt は過去7実行のSD 0.85pt / 0.74pt に対し
     1.1〜1.3SD のノイズ帯，レイテンシ +0.688%）．
  2. **コスト-性能フロンティア上で下回っていない**．固定 k=2（recall 0.345，コスト2.0）と
     固定 k=3（0.465，コスト3.0）を線形補間すると，本方式のコスト2.399 では 0.3929 相当
     （実際に同コストのランダムk混合を2000シードでシミュレートした平均値と一致）．
     本方式の 0.425 はこれを +3.2pt 上回る．有意ではないが，**下回るという証拠もない**ため，
     同コストで劣る設定を本番に置くことにはならない．
  3. 後方互換パス（`dispatch_gap_threshold: null`）が実装済みで，判断は完全に可逆である．
     採用を維持するコストは低く，棄却して戻すべき積極的理由（退行・不安定性）が無い．
- **「条件付き」とした理由（採用を強い主張にしてはならない3点）**:
  1. **有意性が境界的**．exact McNemar (28,12) で p=0.0166，discordant 40 件のうち
     **2ペア反転（26,14）で p=0.0807 となり非有意化する**．n=200・効果量+8.0pt に対して
     この脆弱性は小さくない．
  2. **改善の大半はコスト純増で説明できる**．dispatch 総呼び出しは 3200→3839（**+19.97%**，
     事前登録上限 +20% に対し余裕 0.03%）．同コスト条件下での gap 信号固有の寄与は +3.2pt
     にとどまり，ペア比較での p の中央値は 0.405（本方式が有意に勝つシードは 0.1%）．
     **「gap 信号は同コストの固定k混合より良い」とは現時点のデータでは主張できない**．
  3. **in-sample 選定**．(T, max_k) は評価集合1600問そのもの上の240点グリッドで選ばれ，
     本走も同一1600問で行われた．本走は独立確証実験ではなく**実装の再現性検証**に相当し
     （3指標が小数点以下まで一致した），主基準の p はグリッド探索の多重性に未補正である．
     **汎化性能の主張はできない**．
- **したがって論文・対外記述では**，(i) compound_domain_set_recall 0.345→0.425 は
  「dispatch コスト +20% を伴う探索的知見」であること，(ii) 同コストの参照値 0.3929 を必ず併記
  すること，(iii) T の選定が in-sample であることを明記すること．この3点を欠いた記述は
  データが支えていない．

**総括（仮説の照合）**

- 仮説「単一ドメインでは k=1 に落として節約し，compound でのみ k を伸ばす」は**半分反証された**．
  単一ドメイン1500行は k=1:781 / k=2:54 / **k=4:665** で平均 2.0→2.366（+18.3%）と**増加**した．
  コストは「節約」ではなく「compound へ再配分しつつ全体で純増」した．
- 「連鎖的エスカレーション（k を 3 以上へ段階的に伸ばす）」も実体が無い．**k=3 は1600行中0件**で，
  T=0.29 の下では事実上 **k∈{1,4} の二値ポリシー**に退化している（k=2 は58行のみ）．
  機序は明確で，gap12<T かつ gap23<T を満たす727行では **gap34 の最大値が 0.2357** で T に一度も
  届かない（平均0.054）——confidence の裾は一度平坦域に入ると隣接差が戻らない．
  **max_k が主要な制御変数，T は k=1 群と k=max_k 群の分割比率の制御変数**というのが実態である．
- gap 信号が compound を識別している程度は，k=4 へのエスカレート率 compound 62.0% vs
  単一 44.3%（比1.40）で，調査フェーズの AUC 0.576 と整合する「弱いが正の」分離にとどまる．
  `compound_domain_jaccard_mean` は 0.2400→**0.2317 と微減**しており，recall が上がったのは
  集合を広げた効果で，dispatch 集合の的中の質は改善していない．

**学び（次の自分が読んで分かる形で）**

1. **「弱い信号 × コスト増」で得た改善は，同コスト参照ポリシーと比較しない限り解釈できない**．
   本イテレーションで最も価値のある分析は事前登録外の§7（同コストのランダムk混合との比較）
   だった．今後，k やリトライ回数など**コストを動かすレバー**を扱う場合は，
   **等コスト参照ポリシーとの比較を事前登録の成功条件に含める**こと．主基準を
   「基準線からの改善」だけで書くと，レバー固有の寄与と単なる予算増を分離できない．
2. **同一データで選定した閾値を同一データで確認しても新情報は増えない**（決定論的なので
   3指標が小数点以下まで一致した）．閾値系レバーでは，選定用と確認用のデータ分割
   （ホールドアウト）を計画段階で必ず設けること．再実行による追加反復は本レバーでは無意味．
3. **呼び出し元の数え漏れによる no-op は再発した**（Iter27 に続き2回目，今回は
   `run_experiment.py:85` のメトリクス記録専用の重複呼び出しが gap 引数を受け取らず，
   実 dispatch は新方式・記録は旧方式という**部分 no-op**になっていた）．
   予備20問での「発火の証拠フィールド」確認が唯一の検出手段として機能した．
   **今後は `grep -rn "関数名(" --include=*.py` をテスト以外の全ファイルに対して機械的に実行し，
   呼び出し元の件数を計画書に明記する**こと．
4. **`.get(key, default)` は値が `None` のときに default を返さない**（`evaluation.py:73` の
   `or ""` 欠落によるクラッシュ）．`dispatch_failed` 行が初めて出た今回まで踏まれなかった
   既存バグで，修正済み（回帰テスト追加済み）．欠測が起こり得るフィールドの取り出しは
   `.get(k) or default` を既定の書き方とする．
5. **検定の変種を事前登録文と実装で揃える**．事前予測 p=0.0095〜0.0166（exact）と実測報告
   0.0177（連続性補正近似）の乖離は，データ差ではなく検定変種の違いだった．
   `metrics._mcnemar_from_correctness()` は連続性補正版である．

**次レバー（自動判断，詳細は backlog B92）**

- `dispatch_policy` は values が単一値のため**本イテレーションでクローズ**．同時に，config.yml の
  levers は `conformal_prediction_true_class_qhat`（B88 で優先度低・シミュレーション時点で
  coverage 0.8025 < target 0.87，mean_set_size 7.31 > 4.0 と失敗見込みが確定済み）と
  `post_hoc_langdetect_retry`（Iter55 で langdetect ja=100/100 のため改善余地が無い）しか
  残っておらず，**実質的に試し切りの状態**にある．
- そこで skill の停止条件 1（journal/backlog の学びから新レバーを考案して継続）に従い，
  新レバー **`dispatch_candidate_ranking = multilabel_binary_relevance_head`** を config.yml の
  levers 末尾に追記した．狙いは本イテレーションで露呈した**根本原因**への直撃である:
  compound 100行のうち2つの正解ドメインが上位2位に収まるのは 3/100 のみで，2つ目の正解の
  ランク中央値は5〜6位．これは 10クラス softmax（各ノードが自分のクラスの確率のみ返す単一ラベル
  構成）を多ラベル問題に流用していることの構造的帰結であり，**k を増やす側の工夫では
  コストを払う以上のことはできない**ことが今回のデータで示された．rank_1 の argmax は既存分類器の
  ままとし，**rank_2 以降の順位付けのみ**を OvR sigmoid（binary relevance）ヘッドに差し替えれば，
  argmax flip rate は構造的に 0% で単一レバー原則を自動的に満たす．
  **第1イテレーションはオフライン完結**（既存1427件で OvR ヘッドを訓練し，1600問をオフライン採点して
  固定 k=2・コスト中立条件下の compound_domain_set_recall を測る）とし，`config.yaml` の
  スキーマ変更（実行時経路への配線）は，オフラインで所定の改善が確認できた場合にのみ
  次々イテレーションでユーザー確認のうえ行う．
- 次イテレーション名: **「multi-labelヘッドによるdispatch候補順位付けのオフライン検証」**．
- 今回の分析が挙げた他の候補（`max_k=3` での再探索，累積確率質量(top-p)型の変種）は，
  いずれも**同一データ上の同一信号の再探索**にとどまり学び2に反するため，次の一手には採らない
  （backlog に候補として記録のみ）．

## Iteration 57: education_threshold=0.05の実行時経路反映と実機検証

### 実験 (Iter57)

- **実施日時**: 2026-09-18
- **(a1) オフライン再評価（Phase A）**: wafl500 の Docker ollama（0.34.2，nomic-embed-text
  digest `0a109f42...`，Iter55 と同一重み）へ SSH ポートフォワードで 1600 問を
  threshold=0/0.05 の 2 回再評価．**ゲート全パス**: threshold=0 で
  education_recall = 87/170 = 0.5118（Iter55 実走と一致），argmax 不一致 0 行
  （embedding bit 一致）→ (a1) 成功，(a2) フォールバック不要．threshold=0.05:
  flip 38 件（全て education へ，TP 寄与 4 件），education_recall = 91/170 = 0.5353，
  confidence > 1.0 は 0 件．
  成果物: `results/iter57_a1_threshold0_predictions.jsonl` /
  `results/iter57_a1_threshold0.05_predictions.jsonl`
- **環境異常と復旧**: 全 10 ノードでコンテナ・イメージが消失（ollama_data ボリューム生存）．
  前半は制御機 gpu2 の docker 権限欠如で Phase B ブロックされ，ユーザーが権限付与後に継続した．
- **デプロイ（Phase B）**: `mise run setup`（git HEAD=1eaa7c0，image digest
  `sha256:7aaa3974...`）→ `mise run deploy`（全 10 ノード healthy，モデル pull 全 skip
  ＝モデルデータ生存を確認）．smoke check: git-status / hashes（全 10 ノード × 4 ファイル，
  classifier.py 含む）/ probe（6ms）全 pass．二重安全: wafl501 のデプロイ済み
  classifier.py で EDUCATION_THRESHOLD を grep 確認．
- **予備実行（Phase C，先頭 20 問）**: `results/20260918_201933/`．**発火証拠確認済み**:
  education の +0.05 厳密一致 20/20（bit 一致），他 9 ドメイン bit 一致 20/20，
  selected_domain 変化 0．
- **本走（Phase D）**: `results/20260918_202613/`（1600 行，約 40 分 — 過去実測 96-101 分より
  大幅に短い．全ノードで GPU pass-through が有効だったためと推定）．
  confidence > 1.0 の行数: 0（top-level・probe_candidates とも）．
  - 主要指標（metrics.json）: top1_accuracy = **0.5975**（Iter55: 0.6031），
    education recall = **91/170 = 0.5353**（Iter55: 87/170 = 0.5118），
    medical recall = 0.4775（Iter55: 0.5000），ECE = 0.0549（Iter55: 0.0630），
    answer_quality_accuracy = 0.5693（Iter55: 0.5607），fallback/dispatch_failure = 0．
- **shadow 検証（Phase E）**: 本走 flip（Iter55 → 本走）38 件 ＝ オフライン (a1) flip 38 件，
  **共通 38（100% 一致）**，本走 vs オフラインの selected_domain 不一致 0/1600
  （オフライン flip 集合が実走の完全な shadow）．flip rate = 38/1600 = 2.375%．
- **ツールバグ発見（reflector へ申し送り）**:
  1. setup の registry 判定バグ: `docker ps --filter name=... | grep -q .` がコンテナ 0 件でも
     ヘッダー行にマッチし「already running」と誤判定 → push が connection refused で失敗．
     wipe 後に初めて顕在化．手動で registry 起動（setup と同一コマンド）して回避．
     mise.toml の `docker ps -q --filter ...` 修正を推奨．
  2. analyze の「最新」解決バグ: `ls -1d results/*/ | sort | tail -1` は辞書順のため
     `results/iter45_preliminary/`（0 行）が選択される．明示 datetime 指定で回避
     （副作用: 初回実行で全ノードのログが iter45_preliminary/logs/ に混入）．
     metrics_cmd と同じ `ls -1dt` 方式への修正を推奨．

### 分析 (Iter57)

- **独立再計算（突合）**: 主要数値を `metrics.py` の既存関数
  （`compute_precision_recall_per_domain` / `compute_mcnemar_test`（continuity-corrected）/
  `compute_domain_recall_mcnemar_test` / `compute_domain_precision_fisher_test` /
  `apply_benjamini_hochberg` / `compute_ece` / `compute_wilson_confidence_interval`）で
  独立に再計算した結果，実験フェーズの報告値と `metrics.json` とは**全項目一致**（不一致 0）．
  - top1: 本走 956/1600 = 0.5975（Wilson [0.57326, 0.62127]），Iter55 965/1600 = 0.6031
  - education recall: 本走 91/170 = 0.5353（precision 0.3745），Iter55 87/170 = 0.5118
  - medical recall: 本走 **85/178 = 0.4775**（precision 0.53125），Iter55 89/178 = 0.5000
  - 他ドメイン recall（本走）: business_economics 0.5119，computer_science 0.5357，
    general 0.5305，history_culture 0.7679，legal 0.5333，mathematics 0.6310，
    natural_science 0.5833，social_science 0.5238（報告値と一致）
  - ECE 0.0549（Iter55 0.0630），brier 0.2026，AUROC 0.7450，tie_rate 0.0，
    mean_confidence_sum 1.05，fallback/dispatch_failure 0，compound_set_recall 0.345
    （Iter55 と同一），single/compound top1 = 0.61/0.41
  - answer_quality: 0.5693 vs 0.5607（+0.87pt，3SD = 2.6pt のノイズ床より小さいため有意でない）

- **統計的判定（McNemar は continuity-corrected，`metrics.py` の関数のみ使用）**:
  - top1: discordant a_only = 13（Iter55 正解 → 本走不正解），b_only = 4，
    chi2 = 3.765，**p = 0.0523** → α = 0.05 で有意でない（非退行成立，ただし境界値）
  - education recall: a_only = 0，b_only = 4（新規正解: education-024/087/122/125，
    全て単一ドメイン行），chi2 = 2.25，**p = 0.1336** → 改善は有意でないが方向は一方向（4/0）
  - medical recall: a_only = 4（新規不正解: medical-010/071/083/137），b_only = 0，
    chi2 = 2.25，**p = 0.1336** → 有意でない
  - 他 9 ドメイン 18 指標（recall は McNemar，precision は Fisher の 2 標本検定）:
    **BH 補正後有意 0 件，有意退行 0 件**（最小 p は social_science_recall 0.248）

- **ノイズ判定**: 軸①（ルーティング系）は決定論的であり（success_criteria (5)），
  反復間ノイズ床は適用しない．top1 の -0.0056（9 行）と medical の -0.0225（4 行）は
  生成ノイズではなく，38 件の flip の決定論的帰結である．根拠: (i) 同一構成の実走は
  過去に bit 一致で再現している（Iter47 top1 = 0.6031 ＝ Iter55 top1 = 0.6031，
  本走のオフライン threshold=0 再評価も Iter55 と selected_domain 不一致 0/1600），
  (ii) 本走の全 38 flip は education へのみであり，top1 の新規不正解 13 行と
  medical の新規不正解 4 行は全てこの 38 行に含まれる．両 McNemar の p が 0.05 以上のため，
  有意な退行とは判定しない．

- **仮説との整合**: 計画の仮説「flip 約 2.56%，education_recall が 0.55〜0.56 前後へ上昇，
  他 9 ドメイン 18 指標・top1_accuracy に有意退行なし」は**機序は整合，
  上昇幅 0.55〜0.56 は不整合（実測 0.5353）**．計画自体が予見した通り，0.5647 は
  旧オフライン経路（11435 系）の値であり，実走経路（11434 系）での正値は 0.5353
  （オフライン shadow も 0.5353）である．flip 2.375%（38/1600）はオフライン実績
  2.56%（41/1600）とほぼ一致．「有意退行なし」の予測は整合（BH 0 件，top1 p = 0.0523，
  medical p = 0.1336）．

- **shadow 検証（独立再確認）**: オフライン `iter57_a1_*_predictions.jsonl` と実走
  `results.jsonl` を直接突合し，実験フェーズの報告を独立に確認した:
  1. offline threshold=0 vs Iter55: selected_domain 不一致 **0/1600**（bit 一致）
  2. offline threshold=0.05 の flip = 38 件，全て education へ
  3. offline 0.05 vs 本走: selected_domain 不一致 **0/1600**
  4. offline 0.05 の education_recall = 91/170 = 0.5353（本走と同一）
  5. 本走 flip 集合 ＝ offline flip 集合（38 = 38，100% 一致）
  6. 本走の education ノード報告 confidence ＝ offline 生確率 + 0.05 が **1600/1600 行で
     bit 一致**（発火証拠の独立確認）
  → オフライン +0.05 予測が実走の完全な shadow であり，本イテレーションの目的である
  「オフライン-実行時の一貫性」が成立した．production_deployment_gap は解消された．

- **その他記録**: confidence > 1.0 の行は 0 件（top-level・probe_candidates とも．
  オフライン iter44 経路では 1 件存在した）→ `compute_ece` の [0,1] bin 外漏れなしで
  ECE は信頼できる．mean_confidence_sum = 1.05 は再正規化なし +0.05 加算の语义
  （報告確率和 1.0 + 0.05）と整合．medical の -0.0225 は flip 機序の副次効果
  （medical 正解行 4 件が education に取られたもの）で，オフライン iter52 の
  threshold=0.05（旧経路）でも med_recall = 0.4775 と同値が出ており再現性がある．

- **成功条件 (d) の判定**:
  | # | 条件 | 実測 | 判定 |
  |---|---|---|---|
  | 1 | education_recall > 0.5112（medical 基準，公式 multi-label，n=170） | 0.5353（91/170） | **PASS** |
  | 2a | 他 9 ドメイン 18 指標の BH 補正後有意退行 0 件 | 0 件 | **PASS** |
  | 2b | argmax flip rate < 15% | 2.375% | **PASS** |
  | 2c | top1_accuracy 非退行（continuity-corrected McNemar，p >= 0.05） | p = 0.0523 | **PASS**（境界値，追記観察推奨） |
  | 3 | 全指標を公式 multi-label recall に固定 | 本分析で実施 | **PASS** |

- **次の考察フェーズへの示唆**: 全成功条件 PASS．`production_deployment_gap =
  apply_education_threshold_to_runtime` は**採用**が妥当（Iter52/53 の adopted 決定が
  実行時経路でも成立したことを実機で確認済み）．単一値レバーであり採用で収束する．
  追加反復は不要（軸①は決定論的で，shadow 100% 一致は反復より強い保証を与える）．
  留意点: top1 p = 0.0523 が境界値であること，および medical_recall が 0.5000 → 0.4775
  と medical 基準（0.5112）を割り込んでいる点は，次のレバー
  （dispatch_policy=adaptive_confidence_gap，要ユーザー確認）の計画材料として残す．

### Iteration 57 実行済み

- **変更**: `classifier.py` に `EDUCATION_THRESHOLD = 0.05` を定義（education のみ，
  再正規化なし，確率和 1.05 の语义）＋単体テスト 3 件＋`tools/smoke_check.py` の
  `DEPLOYED_FILES` に `classifier.py` を追加（コミット `1eaa7c0`）．
- **結果**: 成功条件 (d) 全項目 PASS．
  - education_recall 0.5118 → **0.5353**（91/170）> 0.5112（medical 基準，公式 multi-label）
  - top1_accuracy 0.6031 → **0.5975**（McNemar continuity-corrected p = 0.0523，
    a_only=13 / b_only=4，非退行成立だが境界値）
  - 他 9 ドメイン 18 指標の BH 補正後有意退行 **0 件**（最小 p = 0.248）
  - argmax flip rate **2.375%**（38/1600）< 15%
  - shadow 検証: オフライン (a1) flip 38 件 ＝ 実走 flip 38 件（**100% 一致**），
    selected_domain 不一致 0/1600，education confidence = 生確率 + 0.05 が
    1600/1600 行で bit 一致 → 「オフライン-実行時の一貫性」が成立
  - 前回比: ECE 0.0630 → 0.0549，answer_quality 0.5607 → 0.5693
    （+0.87pt < 3SD = 2.6pt で有意でない），medical_recall 0.5000 → 0.4775
- **判定**: **採用（adopted）**．Iter52/53 の adopted 決定が実行時経路でも成立したことを
  実機で確認したことで **deployment gap は解消された**．単一値レバーであり採用で収束，
  追加反復は不要（軸①は決定論的であり，shadow 100% 一致は反復より強い保証）．
- **学び**:
  1. **決定論的レバー（軸①）では shadow 検証が反復より強い保証を与える**:
     オフライン flip 集合 ＝ 実走 flip 集合（38 = 38，100% 一致）を確認できれば，
     同一構成の実走反復で得られる情報よりも強い．反復は生成ノイズの推定にしか
     使えないが，shadow 一致は「オフライン予測が実行時を完全再現する」ことの
     直接証拠になる．今後の決定論的レバーでは，オフライン shadow 予測との
     bit 単位突合を標準手順にすべき．
  2. **オフライン-実行時の embedding 経路乖離は事前検出可能だった**:
     オフライン再評価（11435 系）と実走（11434 系）の embedding 経路差により，
     旧経路では 0.5647 だった education_recall が実走経路では 0.5353 になる
     （閾値効果ではなく経路差）．実走と同じ経路での (a1) ゲート
     （threshold=0 で基準線と bit 一致確認）により，本走前に乖離の存在を検出
     できた．**計画フェーズで「期待値 0.55〜0.56」を引用する際は，その値が
     どの経路で出た数値かを必ず明記すること**（Iter57 計画の 0.55〜0.56 予見が
     実測 0.5353 と不整合だった原因はこれ）．
  3. **ツールバグ 2 件**（backlog B89 へ申し送り）: (a) setup の registry 判定が
     `docker ps --filter name=... | grep -q .` のヘッダー行マッチでコンテナ 0 件でも
     「already running」と誤判定し push が connection refused になる．
     `docker ps -q --filter ...` への修正を推奨．(b) analyze の「最新」解決が
     `ls -1d results/*/ | sort | tail -1`（辞書順）のため `results/iter45_preliminary/`
     が選択される．metrics_cmd と同じ `ls -1dt` 方式への修正を推奨．
  4. **留意点（次レバー計画の材料）**: (a) top1 の McNemar p = 0.0523 が境界値
     （a_only=13 / b_only=4）であること．(b) medical_recall 0.4775 が medical 基準
     （0.5112）を割っていること（オフライン iter52 旧経路でも同値で再現性あり，
     flip 機序の副次効果で有意退行ではないが，dispatch_policy の計画時に
     medical 側の影響を評価する材料とする）．

### 実装 (Iter57)

- **変更ファイル**（3 ファイル，最小差分，コミット `1eaa7c0`）:
  - `classifier.py`: 定数 `EDUCATION_THRESHOLD = 0.05` を定義（Iter52/53 の
    adopted 決定・再正規化なしで確率和 1.05 の语义・Iter51 の 0.3 失敗は加算量過大
    が原因，をコメントに明記）．`estimate_confidence_classifier()` の末尾で
    `domain == "education"` の場合のみ `float(probabilities[domain_index] +
    EDUCATION_THRESHOLD)` を返す（他ドメインは生値不変，再正規化なし）．
    classes に無い domain が 0.0 を返す現行挙動は不変．ファイル冒頭・関数の
    docstring を更新．
  - `tests/test_classifier.py`: education を含む 4 ドメインの toy classifier
    fixture（実 fit の LogisticRegression，モック不使用）と単体テスト 3 件を追加
    （education は生確率 + 0.05，medical は生確率のまま不変，未知 domain は 0.0）．
  - `tools/smoke_check.py`: `DEPLOYED_FILES` に `classifier.py` を追加（計画承認済み
    hygiene．hash check が本イテレーションで変更するファイルをカバーし，Iter12/Iter22
    型のデプロイ漏れ検出ギャップを恒久是正）．
- **検証**: `uv run pytest tests/test_classifier.py -v` → 7 passed（既存 4 + 新規 3）．
  `uv run ruff check`（3 ファイル）→ All checks passed．全スイート実行時の失敗 12 件は
  `test_build_dataset.py` / `test_train_domain_classifier.py` の既存失敗
  （JMMLU データ欠落・sklearn API 不整合）で本変更とは無関係（import 経路が独立）．
- **実験への申し送り**: コミット `1eaa7c0` が git HEAD に含まれる状態で
  `mise run setup`（git HEAD から image build）→ `mise run deploy` の順を厳守すること
  （実行チェックリスト 1）．deploy 後の smoke hash check は classifier.py をカバー
  するため，チェックリスト 2 の手動 grep 確認は二重安全として併用する．

### 計画 (Iter57)

- **目的**: Iter52/53 で adopted 確定した `education_per_class_threshold=0.05`（公式 multi-label
  education_recall 0.5647，オフライン）を実行時経路へ反映する（実装漏れの是正，新規手法の検証では
  ない）．実機で education_recall が medical 基準（0.5112，公式定義 91/178）を上回るかを最終確認する．
- **単一レバー**: `production_deployment_gap = apply_education_threshold_to_runtime`．
  変更は `classifier.py:estimate_confidence_classifier()` への education 領域のみ `+0.05` 加算
  （再正規化なし）の 1 箇所のみ．他はすべて直近の最良構成に固定する．
- **仮説**: 実行時に education の報告 confidence に +0.05 加算すると，argmax flip は約 2.56%
  （オフライン Iter52b/53 実績 41/1600）にとどまり，education_recall は Iter55 実機の 0.5118
  （87/170）から 0.55〜0.56 前後へ上昇し，他 9 ドメイン 18 指標・top1_accuracy に有意退行は出ない．
  ただしオフラインと実走の embedding 経路が乖離している（argmax 不一致 96/1600，Iter57 調査）
  ため，実走値がオフラインの 0.5647 に到達する保証はなく，判定は > 0.5112 基準で行う．
- **固定する構成**（変更しない）: `routing_method=supervised_classifier`，
  `confidence_threshold=0.0`，`dispatch_candidate_threshold=0.0`，`dispatch_top_k=2`，
  `aggregation_method=max_confidence`，`expert_model=expert-mesh-{domain}-lora`，
  `light_model=qwen3.5:4b-q4_K_M`，分類器は現行本番 `models/domain_classifier.joblib`
  （temperature 較正 + intercept_delta=0.7 焼き込み，再訓練しない），評価集合 1600 問不変．
- **実装設計**:
  - `classifier.py`: 利用箇所の直上（`estimate_confidence_classifier()` の直前）に定数
    `EDUCATION_THRESHOLD = 0.05` を定義（マジックナンバー禁止規約．`train_domain_classifier.py`
    の `intercept_delta = 0.7` ハードコード（L200 付近）と同一の既存パターン）．
  - `estimate_confidence_classifier()`（L47-61）の末尾で，`domain == "education"` の場合のみ
    `float(probabilities[domain_index] + EDUCATION_THRESHOLD)` を返す（それ以外は生値を返す
    現行 L61 のまま）．再正規化は行わない（確率和 1.05 が adopted 決定 Iter52/53 と同じ语义．
    Iter51 の threshold=0.3 失敗は加算量過大が原因）．
  - 等価性: 全ノードが同一分類器を共有し各ノードが自ドメイン成分のみ報告するため，
    education ノードが `p+0.05` を報告することは，オフラインの「確率ベクトルの education 成分に
    +0.05 加算して argmax」と数学的に等価（aggregator は報告値の max を取る）．
  - `config.yaml` のスキーマ変更なし，`mise.toml` 変更なし（ユーザー確認不要）．
  - `tests/test_classifier.py` に単体テスト追加: education は p+0.05 を返す，他ドメインは不変，
    classes に無い domain は 0.0 を返す（現行挙動の回帰防止）．
  - 任意の hygiene（レバーではない）: `tools/smoke_check.py` の `DEPLOYED_FILES` に
    `classifier.py` を追加することを推奨する（現状の hash smoke check は
    http_server.py/router.py/config.yaml のみで，本イテレーションで変更するファイルが
    カバー外である）．
- **成功条件の再定義 — (a1) を選択する**:
  - 元のレバー note の (a)「オフライン再計算で実機 0.5118 と一致」は既存オフライン成果物では
    不成立（iter44 予測ファイルで 89/170 = 0.5235．原因は embedding 経路の乖離: オフラインは
    制御ノード ollama 11435，実走はノード Docker ollama 11434．分類器係数は bit 一致）．
  - **選択: (a1)（実走と同じ embedding 経路でオフライン再評価をやり直す）**．理由:
    (i) 本イテレーションの目的はオフライン-実行時の一貫性検証そのものであり，(a1) は発見された
    ギャップに直接対処する（(a2) は乖離を容認するのみ）．(ii) (a1) によりオフライン +0.05 の
    flip 集合が実走 flip 集合のほぼ厳密な shadow 予測になる（分類器は決定論的，embedding が
    同一）ため，実走結果とオフライン予測 0.5647 の差を embedding 乖離の影響として定量化できる．
    (iii) コストは modest（~30-60 分）で，(d) の最終判定に依存しない．
  - (a1) の実施: requester ノード（wafl500）の Docker ollama へ SSH ローカルポートフォワードを
    張り，`evaluate_classifier_calibration.py` を `--ollama-host/--ollama-port` で当該経路を指して
    1600 問再評価する．実行前に `docker compose exec -T ollama ollama list` で nomic-embed-text
    の digest と ollama バージョンを確認し記録する（調査フェーズの未了事項）．
  - (a1) の成功: threshold=0 の再評価で Iter55 実走と education_recall = 87/170 = 0.5118 に一致し，
    argmax 不一致 0 行（embedding が bit 一致なら厳密一致が期待される．不一致 > 0 なら原因を
    特定してから本走に進む）．
  - **フォールバック (a2)**: SSH 不安定やバージョン不一致等でノード ollama 経由の再計算が実行
    不能な場合，オフライン基準を 0.5235 と認め，embedding 乖離を既知の制約として journal に明記し，
    最終判定を (d) に委ねる．(d) が最終判定である点は (a1)/(a2) いずれでも不変．
- **最終判定 (d)**: 実機 1600 問 1 回実行し，
  - 主基準: education_recall（公式 multi-label recall，`metrics.py:compute_precision_recall_per_domain`，
    母数 n=170）が 0.5112（medical 基準，公式定義）を超えること．
  - 非退行: 他 9 ドメイン 18 指標の BH 補正後有意退行 0 件，argmax flip rate < 15%
    （オフライン実績 2.56%），top1_accuracy 非退行（McNemar は continuity-corrected，
    `metrics.py` の関数のみで計算，p >= 0.05）．
  - 全指標を公式 multi-label recall に固定し，行レベル正解率（0.6000 系）を混用しない．
  - 比較基準線: Iter55 実走 `results/20260808_194131/`（現行本番構成，top1=0.6031，
    edu_recall=0.5118，med_recall=0.5000，answer_quality=0.5607）．
- **実行チェックリスト（d0004 §4 の教訓，no-op 6 連発の再発防止）**:
  1. デプロイ順序を厳守: commit（git HEAD に classifier.py 変更を含む）→ `mise run setup`
     （**イメージは setup が git HEAD から build + push する**．deploy は pull するだけなので
     setup を飛ばすと Iter12 と同一のデプロイ漏れになる）→ `mise run deploy`
     （pull + restart + smoke check + healthcheck）→ 予備実行 → 本走 → `mise run analyze`．
  2. `classifier.py` は smoke hash check の `DEPLOYED_FILES` に含まれないため，deploy 後に
     手動でデプロイ済みコピーに新コードが含まれることを直接確認すること
     （例: `ssh <node> "cd <remote_dir> && docker compose exec -T app grep EDUCATION_THRESHOLD classifier.py"`）．
  3. 先頭 20 問の予備実行で発火証拠を確認: 同一質問について education ノードの報告 confidence が
     Iter55 実走の confidence に +0.05 加算した値と bit 一致し，他 9 ドメインの confidence が
     Iter55 と bit 一致すること（実走の embedding 経路は Iter55 と同一のため差分は加算のみの
     はずである）．
  4. 実走後: confidence > 1.0 の行が 0 件であることを確認する（オフライン iter44 ファイルでは
     p_edu > 0.95 の行が 1 件存在．1 件以上出た場合は journal に明記すること —
     `metrics.py:compute_ece`（L505-513）の [0,1] bin は > 1.0 の行を bin 外漏れさせ ECE を
     静かに下振れさせる）．
  5. `mise run analyze` を必ず実行し answer_quality_accuracy を計測する（B88 hygiene の計測
     復活．noise floor 3SD = 2.6pt を超えない限り有意と判定しない）．
  6. shadow 検証: 実走の flip 集合（Iter55 → 本走）をオフライン (a1) の +0.05 flip 集合と比較し，
     一致度（件数・education 寄与）を journal に記録する．
- **コスト見積もり**: 実装 + 単体テスト ~30 分．(a1) オフライン再評価 ~30-60 分
  （SSH フォワード設定 + 1600 件の embedding 10-30 分 + threshold=0/0.05 の 2 評価）．
  `mise run setup`（image build + push）~10 分．`mise run deploy` ~15 分．予備実行（20 問）
  ~10 分．本走 1600 問 ~96-101 分（config timeout 150 分）．`mise run analyze` ~15-30 分．
  合計 wall-clock ~3-4 時間 + 実装．
- **reflector への申し送り**: 「post-hoc 手法の天井 0.6000」（B81/B83/B84）は行レベル正解率の値
  であり，公式 multi-label 定義では 0.5647（iter52b/53，threshold=0.05）である．Iter44 は
  0.5235（公式）/ 0.5588（行レベル）．journal/backlog の該当記述は定義付きで書き直すこと．
  0.5647 > 0.5112 なので adopted 判定自体は不変．
- **不成立の場合**: 実走 education_recall が 0.5112 を下回った場合，B84 の最終結論
  （研究 converged）を撤回し，「post-hoc 手法の天井」という結論自体を再検討する
  （B88 により要人間判断）．

### 調査 (Iter57, 補足)

- **問い**: (1) 実行時集約経路（`node.py:run_ask_flow` / `aggregator.py`）において，オフラインの
  `probabilities[edu_idx] += 0.05` → argmax を厳密に再現する実装は 2 案（(a) education ノードが
  `p+0.05` を報告／(b) 集約側で +0.05 加算）のうちどちらか．(2) 現行実機構成での education_recall を
  オフライン再計算し，Iter55 実機 0.5118 と一致するか．(3) B88 の hygiene task 3 件（recall 母数／
  McNemar chi2／answer_quality 未計測）の原因確定．(4) per-class threshold のオフライン評価と
  実行時デプロイの一致性に関する pitfall の先行調査．

#### ギャップ検証（コード読解，オーケストレータ報告との突合）

- オーケストレータ報告の事実を全て一次確認できた：`classifier.py:47-61`（生 `predict_proba()` の
  自ドメイン成分のみを float 返却，threshold 加算なし）／`http_server.py:364-370`
  （`ROUTING_METHOD_SUPERVISED_CLASSIFIER` 分岐で `estimate_confidence_classifier()` 呼び出し）／
  `http_server.py:406-411`（lifespan で `load_domain_classifier(state.classifier_model_path)`）／
  `scripts/evaluate_classifier_calibration.py:237-241`（fine-tuned embedding 経路）と `272-276`
  （ollama 経路）の `probabilities[edu_idx] += education_threshold`（再正規化なし）／`mise.toml`
  通読で education/threshold 関連パラメータの受け渡しなし（deploy は config.yaml と models/ の
  rsync のみ，start は `--node-id/--dataset/--output` のみ）．**ギャップの存在は確定**．
- 現行 `config.yaml` の到達条件: `routing_method: supervised_classifier`（L36）＋
  `classifier_model_path: models/domain_classifier.joblib`（L81）＋ `confidence_threshold: 0.0`
  （L5）＋ `dispatch_candidate_threshold: 0.0`（L10）＋ `dispatch_top_k: 2`（L57）＋
  `aggregation_method: max_confidence`（L68）．`http_server.py:367` は現行構成で必ず到達
  （Iter55 実走が証拠）．

#### 集約経路の分析（実行時実装の等価 2 案の比較）

- 実行時の最終選択経路: `node.py:202`（requester が query_embedding を 1 回計算）→ `node.py:211-213`
  （`probe_all`，各ノードが自ドメインの確率のみ報告）→ `node.py:214-219`
  （`select_dispatch_targets`，rank1 は `confidence_threshold=0.0`，rank2+ は
  `dispatch_candidate_threshold=0.0` で常に適格，`top_k=2`）→ `node.py:232-241`
  （`_dispatch_to_targets`）→ `aggregator.py:80-95`（`select_best_dispatch_response` =
  dispatch された候補中の最大 confidence）．
- **実測確認**: Iter55 実走 `results/20260808_194131/results.jsonl` の 1600 行全てで
  `argmax(probe_candidates の 10 ドメイン confidence) == selected_domain`（不一致 0 件）．
  実行時の最終選択は報告 confidence の argmax と厳密に一致する（top_k=2 の 2 位 dispatch は
  選択に影響せず，`dispatched_domains`（複合指標）と生成コストのみに影響）．
- **案 (a)（education ノードが `p_edu + 0.05` を報告，`classifier.py` 1 関数）**: 報告ベクトルが
  オフラインの threshold 適用後ベクトルと一致するため，argmax・記録 confidence
  （`run_experiment.py:63` が選択ノードの probe confidence を記録）の両方がオフライン
  `evaluate_classifier_calibration.py:252`（`"confidence": float(probabilities[best_index])`，
  加算後）と bit 一致．**オフラインと厳密に等価**．副次効果: 2 位 dispatch 対象が変化しうる
  （`dispatched_domains` → `compound_domain_set_recall` に影響．オフライン評価には 2 位が
  モデル化されていないため，この指標のオフライン比較は成立しない点に注意）．
- **案 (b)（集約側で education の confidence に +0.05）**: 加算が `select_dispatch_targets`
  **以前**（probe_responses の confidence 値レベル）で行われる場合のみ (a) と等価．
  `select_best_dispatch_response`（最終回答選択）の段階でのみ加算すると**非等価**：
  生確率で 3 位以下だが +0.05 で 1 位になる education は dispatch 対象外のため勝利できない
  （オフラインでは argmax で選ばれる）．
- **推奨は (a)**: 変更が `classifier.py` の 1 関数に閉じ，オフライン语义（加算のみ・再正規化なし・
  記録 confidence も加算後）と bit 一致，config.yaml スキーマ変更なし（`train_domain_classifier.py`
  の `intercept_delta = 0.7` ハードコード（L200 付近）と同一の既存パターン）．(b) は等価性を保つ
  ため加算位置が probe_responses レベルでなければならないという制約と，requester 側とノード側の
  二重管理という欠点がある．
- **落とし穴（confidence > 1.0）**: `metrics.py:505-513`（`compute_ece`）は [0,1] の等幅 bin
  （最終 bin 上限 1.0 包含）で，confidence > 1.0 の行はどの bin にも入らず `total`
  （L506）の分母のみに計上されるため ECE を静かに下振れさせる．オフライン iter44 予測ファイルで
  +0.05 加算後 education 確率 > 1.0 の行は 1 件（p_edu > 0.95 の行）．実験チェックリストに
  「confidence > 1.0 の行が 0 件（または 1 件以内で journal に明記）」を含めること．

#### オフライン再検証（レバー note 手順 (a)）— **重大な新発見**

- 保存済み予測ファイル（embedding 再計算不要）で再計算:
  - `results/iter44_boundary_tuning_calibrated_predictions.jsonl`（現行本番モデル，threshold=0）:
    education_recall = **89/170 = 0.5235**（公式 multi-label 定義）．medical = 89/178 = 0.5000．
    top1 = 0.6044．
  - Iter55 実走 `results/20260808_194131/results.jsonl`: education_recall = **87/170 = 0.5118**，
    medical = 89/178 = 0.5000，top1 = 0.6031，fallback/dispatch_failed 0 件．
  - **オフライン 0.5235 と実機 0.5118 は一致しない**（2 行差）．レバー note の成功条件 (a)
    「threshold=0.0 のオフライン再評価で実機 0.5118 と一致」は，既存オフライン成果物では**不成立**．
- 乖離の行単位分析（iter44 オフライン vs Iter55 実走，1600 行）: argmax 不一致 **96/1600 (6%)**，
  確率ベクトルの RMSE 0.0574，|diff| > 0.5 のセル 58 個，最大 diff 0.8855
  （例: education-012 の education 確率がオフライン 0.736 vs 実走 0.0018）．
- 乖離原因の切り分け:
  - 分類器は同一: `models/domain_classifier.joblib` と `models/domain_classifier_iter44.joblib` の
    5 fold 全係数・intercept を数値比較し，coef 最大差 0.0，intercept 最大差 1.1e-16（完全同一）．
  - 質問本文は同一: 実走 results.jsonl と `data/dataset.jsonl` の query 不一致 0/1600．
  - ラベルは同一: 両者の expected_domains 不一致 0/1600（dataset.jsonl の mtime が実走直前
    2026-08-08 19:33 だが，内容は同一）．
  - **したがって乖離は embedding 由来**．journal_archive の記録では，オフライン評価
    （iter29〜iter53 の予測ファイル生成）は `--ollama-host 127.0.0.1 --ollama-port 11435`
    （制御ノード側の ollama）で実行されていた一方，実走は各ノードの Docker 内 ollama
    （`docker-compose.yml:18` の `127.0.0.1:11434:11434`，ノードホストの localhost 限定公開）を
    使う．異なる ollama インスタンス（nomic-embed-text のバージョン・ollama バージョン差）が
    異なる embedding を生成したと推定される（**推測**．ノード側 ollama のバージョン直接確認は
    本調査中の SSH 実行環境の不安定さで未了．計画・実験フェーズで確認すること）．
  - 補足: 制御ノードには現在ローカル ollama（~/.ollama ストア・ollama バイナリ）が存在しない．
- オフライン threshold パイプラインの内部整合性は確認済み: iter44 ファイルの確率に +0.05 を
  適用して argmax を再計算すると flip 41/1600 (2.56%)，education_recall 96/170 = 0.5647，
  `iter52_threshold0.05_predictions.jsonl` / `iter53_per_class_threshold_opt_predictions.jsonl`
  と selected_domain の不一致は 41 件（= flip 集合そのもの）で bit 一致．
- **計画フェーズへの影響**: (a) の成功条件は「既存オフラインファイルで 0.5118 と一致」では
  成立しない．再定義の候補: (a1) オフライン再評価を**実走と同じ embedding 経路**
  （ノードの Docker ollama）でやり直して 0.5118 との一致を確認する（ノードへの SSH 必要）．
  (a2) オフライン基準を 0.5235（既存ファイル）と認め，実機 0.5118 を本番の真の基準値として
  (d) の実走を最終判定に委ねる（オフライン・実走の embedding 乖離を既知の制約として journal に
  明記）．いずれにせよ (d) の実機 1600 問が決定打である点は不変．

#### hygiene task の所見（B88 4.）

- **(a) recall の母数不一致（0.5235 vs 0.5588）— 原因確定**: 母数は両者とも 170
  （expected_domains に education を含む全行，複合 20 件含む）で**同一**．journal_archive
  L2043-2046 の「母数の取り方の違いと推測される」は誤り．異なるのは**分子の定義**:
  - 公式（`metrics.py:71-91` `compute_precision_recall_per_domain`，
    `scripts/analyze_iter52.py:34-35` と同一）: TP = `selected_domain == education` かつ
    education ∈ expected_domains．→ iter44 ファイル: 89/170 = 0.5235，
    iter52b/53 ファイル: 96/170 = 0.5647，Iter55 実走: 87/170 = 0.5118．
  - Iter53 analyst の「独立再計算」: education を含む 170 行の**行レベル正解率**
    （`selected_domain ∈ expected_domains`）．→ iter44 ファイル: 95/170 = 0.5588，
    iter53 ファイル: 102/170 = 0.6000．差はちょうど 6 件の複合行（education と共ドメインの
    2 期待ドメインのうち共ドメインが選択された行）が，公式定義では education の TP にならない
    点．
  - **どちらが正しいか**: 基準値 medical_recall = 0.5112 は公式 multi-label recall
    （iter31 ファイルで 91/178 = 0.5112 を確認済み）であるため，**公式定義が比較可能な正式値**．
    「post-hoc 天井 0.6000」（B81/B83/B84）は行レベル値であり，正式定義では
    **0.5647**（iter52b/53，threshold=0.05）である．0.5647 > 0.5112 なので adopted 判定自体は
    不変だが，journal/backlog の「0.6000」「Iter44 = 0.5588」の記述は
    「0.5647（公式）／0.6000（行レベル）」「Iter44 = 0.5235（公式）／0.5588（行レベル）」に
    定義付きで書き直す必要がある．
- **(b) McNemar の chi2 不一致 — 原因確定**: `metrics.py:248` は
  `chi2 = (|a-b| - 1)² / (a+b)`，すなわち **continuity-corrected McNemar**（docstring に
  「Continuity-corrected McNemar test」と明記．Iter15 の commit 7ba6cde 以来不変）．
  標準公式 `(a-b)²/(a+b)` とは a≠b のとき常に値が異なる（補正版の方が小さい）ため，
  「標準公式と一致しない」は**仕様の差でありバグではない**．ただし d0006 L212-213 の具体例
  （a=13, b=7，実装者報告 chi2=0.8000）は，実装（(6-1)²/20 = 1.25）とも標準公式（36/20 = 1.8）とも
  一致せず，**報告側の手計算誤り**．また Iter53 の p=0.0082 は a=0, b=7 に対して**補正なし**
  公式（49/7 = 7.0 → p = 0.0081）で計算した値で，実装（(7-1)²/7 = 5.14 → p = 0.0234）と異なる．
  結論: 実装は正当な保守的変種で維持してよく，問題は「analyst が手計算で別公式を使う」こと．
  **恒久対策**: McNemar は必ず `metrics.py` の関数（単一情報源）で計算し，手計算チェック時も
  continuity-corrected 公式を使うことを checklist 化すること．（補足: a == b > 0 のとき補正版は
  chi2 = 1/(a+b) > 0 となり p < 1 になるが，これは補正の既知の性質）．
- **(c) answer_quality_accuracy の Iter29 以降未計測 — 原因確定（B88 の記述は不正確な部分あり）**:
  `mise run analyze`（`mise.toml` [tasks.analyze]）は
  `scripts/evaluate_response_quality.py`（`evaluation.py` の
  `compute_answer_quality_accuracy` 経路）を実走 results.jsonl に対して実行する．
  実走ディレクトリの `axis23_metrics.json` は存在する: 20260731_162722 (0.5467)，
  20260801_160058 (0.5500)，20260803_010213 (0.5680)，20260803_092107 (0.5553)，
  20260808_194131 (0.5607)．**つまり実走では計測されていた**．「未計測」の真の理由は:
  Iter29 以降のイテレーションの大半が**オフラインの routing-only 評価**
  （`evaluate_classifier_calibration.py` の出力は selected_domain/confidence/probabilities のみで
  `answer_text` を持たない）であり，answer_quality は構造的に計測不能な成果物であったこと，
  と，実走でも analyze が実行されなかったディレクトリ（`iter45_aggregation_majority_vote_20260802_145653`
  は results.jsonl のみで axis23 なし）が混在すること．Iter57 の実走では `mise run analyze` を
  必ず実行すること（`--ollama-host` 無しでも JMMLU 1500 問は採点可能，手作り 100 問のみ ungraded）．

#### tavily 知見（per-class threshold / post-hoc calibration のオフライン-実行時一致性）

1. **Training-serving skew: モデルが byte-identical でも feature が違えば挙動が変わる**
   （https://clearfeature.dev/solutions/training-serving-skew）: 「The model itself can be
   byte-identical in both places — if the features differ, its behavior differs. Offline metrics
   stop predicting online performance, and nobody changed anything.」本リポジトリの
   「分類器は同一（coef 差 0.0）なのにオフライン 0.5235 vs 実走 0.5118」の状況と完全に対応．
   同概念の整理: https://dswok.com/General-ML/Training-serving-skew
   （「offline metrics improving while online metrics regress」が典型症状）．
   **示唆**: オフライン評価と本番の feature（= embedding）生成経路の同一性を保証・検証する
   メカニズム（同一 ollama インスタンスでの生成，または embedding の永続化と再利用）が不可欠．
2. **threshold 最適化の転移性の前提条件**
   （https://metricgate.com/docs/threshold-optimization-classification）: held-out での最適化，
   「predicted probabilities が poorly calibrated の場合，選択した threshold は新規データに
   信頼できなく転移しない」，「デプロイ時のクラス分布が訓練データと一致することを仮定」．
   **示唆**: threshold=0.05 はオフライン（11435 系 embedding）で選ばれた値であり，
   実走 embedding（11434 系）では確率分布が異なるため，(d) の実走で flip rate と
   per-domain 非退行を必ず再確認する（成功条件にある既存チェックがこれに対応）．
3. **multi-class threshold framework は softmax の確率解釈を捨てる**
   （arXiv 2505.11276, https://arxiv.org/html/2505.11276v1）: 標準 argmax を一般化する
   threshold ベース分類は「discarding the probabilistic interpretation of the softmax-based
   output」を明示．**示唆**: 再正規化なし加算（確率和 1.05）は文献的にも正当化されるが，
   加算後の値は「確率」ではなくなるため，confidence を確率として消費する下流
   （ECE，dispatch ゲート，記録値）への影響を設計で明示する必要がある（本件では
   confidence > 1.0 の ECE bin 問題がそれに該当）．
4. **オフライン-オンラインの一致性検証は shadow/canary が標準手法**
   （https://dagshub.com/blog/model-deployment-types-strategies-and-best-practices，
   https://www.qwak.com/post/shadow-deployment-vs-canary-release-of-machine-learning-models）:
   新モデルを本番入力で並行実行し出力を比較してから切り替える．**示唆**: 本件では
   「先頭 20 問の予備実行で education ノードの報告 confidence が生確率より +0.05 高いことを
   確認する（発火証拠）」＋「(d) の実走でオフライン flip 集合（41 件）と実走 flip 集合の
   一致度を比較する」が shadow 検証に相当する．オフライン-実走の embedding 乖離（96 行
   argmax 差）が既知であるため，実走 flip がオフライン 41 件と完全に一致しないことは
   異常ではなく，**education_recall の絶対値（> 0.5112）が判定基準**になる．
5. **較正版モデル + デフォルト閾値の方が，未較正版 + 手動閾値より頑健**
   （https://valeman.medium.com/classifier-calibration-and-the-end-of-roc-based-threshold-selection-d8e52086cb12）:
   固定 threshold は「deployed in a slightly different setting」で suboptimal になりうる．
   **示唆**: threshold=0.05 は embedding 経路が変わると効き方が変わる固定値であり，
   本研究の範囲（採用済み決定の実装漏れ是正）では問題ないが，将来的に embedding モデルや
   ollama バージョンを更新する場合は threshold の再検証が必要になる点を docs に残す価値がある．

#### 計画フェーズへの示唆（rc-planner 向け）

1. **実装は案 (a)（`classifier.py:estimate_confidence_classifier()` に education のみ
   `+0.05` 加算，再正規化なし）を推奨**．案 (b) を選ぶ場合は「加算が
   `select_dispatch_targets` 以前に probe_responses の confidence 値へ適用される」ことを
   計画節に明記すること（最終選択段階でのみ加算するとオフラインと非等価）．
2. **成功条件 (a) の再定義が必要**: 既存オフライン成果物では 0.5235（≠ 0.5118）になる
   （embedding 経路の乖離が原因，上記「オフライン再検証」節）．(a1)（実走と同じ embedding
   経路でオフライン再計算）か (a2)（オフライン基準 0.5235 を認め (d) を最終判定に）を
   計画で選択し，journal に明記すること．(d) の実走が最終判定である点は不変．
3. **記録指標の定義を固定**: 成功条件・非退行チェックはすべて公式 multi-label recall
   （`metrics.py:compute_precision_recall_per_domain`，education の母数 n=170）で計算し，
   行レベル正解率（0.6000 系）を混用しないこと．journal の「天井 0.6000」は
   0.5647（公式）への書き換えを reflector に申し送ること．
4. **McNemar は `metrics.py` の関数のみ使用**（continuity-corrected．手計算チェックも同一公式）．
5. **実験チェックリスト追加**: (i) 先頭 20 問の予備実行で education ノードの報告 confidence が
   +0.05 高いことの確認（発火証拠，d0004 §4 教訓）．(ii) 実走後に confidence > 1.0 の行数を確認
   （ECE の bin 外漏れ，`metrics.py:505-513`）．(iii) `mise run analyze` を実行し
   answer_quality_accuracy を計測（hygiene (c)）．(iv) 実走の education flip 集合をオフライン
   41 件 flip と比較して一致度を journal に記録（shadow 検証）．
6. **deploy 漏れ防止**: `classifier.py` 変更の commit → `mise run deploy`
   （smoke check の hash 確認）→ 本走の順を厳守（Iter12/Iter22 の教訓，d0004 §4）．
   `models/` は gitignore 対象のため分類器成果物のバージョン管理は git 履歴に残らない
   （config.yml Y4 注記）．本件では分類器は変えないためこのリスクは classifier.py コード側のみ．
7. **未了の確認事項（実験フェーズで実施可）**: ノード側 Docker ollama のバージョンと
   nomic-embed-text の digest を確認し，オフライン（11435 系）との差を特定する
   （SSH: `docker compose exec -T ollama ollama list`）．特定できれば
   「オフライン評価の embedding 経路を本番と揃える」恒久対策（hygiene）として backlog 化できる．

---

### 調査 (Iter57)

- **問い**: B88 の指摘（「Iter52/53 で adopted した `education_per_class_threshold=0.05` が実行時経路に実装されていない」）が現行コードで正しいか．正しい場合，rc-planner が単一レバーとして定式化できる実施方法・落とし穴は何か．
- **検証結果: B88 の指摘は一次情報で全て確認できた（実装漏れ確定）**
  - `classifier.py:47-61` `estimate_confidence_classifier()`: `classifier.predict_proba([query_embedding])[0]` の自分のドメイン成分をそのまま float 返すだけ．education threshold 加算ロジックは**存在しない**．実行時唯一の呼び出し元は `http_server.py:367`（`routing_method=supervised_classifier` 分岐内，現行 config.yaml で有効な経路）．
  - `scripts/evaluate_classifier_calibration.py` `predict_calibrated_rows()`: 加算ロジック `if education_threshold > 0.0: probabilities[edu_idx] += education_threshold` は L237-241（fine-tuned embedding 分岐）と L272-276（ollama 分岐）の**両方**に存在．CLI `--education-threshold`（L363-368）経由でオフライン評価専用のみ．
  - `mise.toml`: `education_threshold` 等の該当パラメータの受け渡しは**一切無い**（deploy は `config.yaml` と `models/` の rsync のみ，start は `--node-id/--dataset/--output` のみ）．config.yaml にも該当フィールドは無い．
  - `scripts/train_domain_classifier.py:200-204`: `intercept_delta = 0.7`（Iter44/45）が訓練時に各 fold の base estimator intercept に加算され，joblib アーティファクトに焼き込まれる．一次確認: 本番 `models/domain_classifier.joblib`（Aug 2 23:41）の education intercept は fold 毎 +0.4764〜+0.6219 で，intercept 適用前の `models/domain_classifier_temperature.joblib`（-0.2177〜-0.0762）とほぼ +0.7 の差で一致し，`domain_classifier_iter44.joblib` と値が同一．**本番モデルは intercept_delta=+0.7 を含み，probability 空間の +0.05 加算は含まない**（加算は intercept には焼き込めず，確率空間操作のため）．
  - Iter55 実機結果 `results/20260808_194131/results.jsonl`（1600 問）を直接再計算: education_recall = 87/170 = **0.5118**（expected_domains に education を含む行 170 件，複合 20 件含む．単一ドメインのみでは 83/150 = 0.5533），medical_recall = 89/178 = 0.5000．「実機実測 0.5118 は intercept_delta のみ反映，threshold=0.05 を含まない」という B88 の説明と整合．
- **rc-planner への引き継ぎ要点**
  - **実装の等価性**: `estimate_confidence_classifier()` は自分のドメイン成分の float しか返さないが，全ノードが同一分類器を共有するため，education ノードが `prob + 0.05` を報告することは，オフライン側の「確率ベクトルの education 成分に +0.05 加算して argmax」と数学的に等価（aggregator は報告値の max を取る）．変更は `classifier.py` の 1 関数だけで十分．
  - **パラメータ化の選択肢**: (A) `classifier.py` に定数 `EDUCATION_THRESHOLD = 0.05` をハードコード（**推奨**．`train_domain_classifier.py:200` の `intercept_delta = 0.7` ハードコードと同一の既存パターンで，config.yaml スキーマ変更不要＝ユーザー確認不要）．(B) config.yaml に新フィールド追加（スキーマ変更のため CLAUDE.md 上，ユーザー確認が必要）．
  - **再正規化は行わない**: adopted 決定（Iter52/53）は加算のみ・再正規化なし（確率和が 1.05 になる）．Iter51 の threshold=0.3 失敗は加算量过大が原因（backlog B76）で，0.05 は再正規化なしで adopted されている．オフラインと bit 一致させるため同じ语义を保つこと．
  - **落とし穴 1（confidence > 1.0）**: `prob_edu > 0.95` の行では報告 confidence が 1.0 を超える．Iter55 データでは education 選定行 205 件中 confidence > 0.95 は 0 件（(0.90, 0.95] が 4 件）だが，flip（予期 2.56%）で新たに選定される行には注意．`metrics.py:compute_ece()`（L486-511）は [0,1] の等幅 bin（最終 bin 上限 1.0 包含）で，confidence > 1.0 の行はどの bin にも入らず `total` 分母のみに計上されるため ECE を静かに下振れさせる．実験チェックリストに「confidence > 1.0 の行が 0 件であることを確認」を含めること．
  - **落とし穴 2（recall 母数，B88 hygiene task）**: 実機 0.5118 = 87/170（複合含む）だが，単一のみでは 0.5533．オフライン Iter52b の 0.5647 は 96/170 と整合するが，Iter53 の 0.6000 は 102/170 か 90/150 か一次情報から曖昧（journal_archive.md L2043-2046 の母数不一致注記と同一問題）．(a) の再検証と (d) の実機比較では**母数定義を n=170（expected_domains に education を含む全行）に固定**し，journal に明記すること．medical 基準値 0.5112（Iter31 系）と Iter55 実機 medical 0.5000（89/178）も母数が異なるため，成功条件「education_recall > 0.5112」の比較対象を計画節で明示すること．
  - **d0004 §4 教訓（no-op 6 連発）への配慮**: 計画節に到達コードパスを明記すること．到達条件: `config.yaml` の `routing_method=supervised_classifier`（現行値）＋ `models/domain_classifier.joblib` がデプロイ済み（deploy の models/ rsync）．`http_server.py:367` は現行構成で必ず到達する（Iter55 がその証拠）．rc-experimenter は本走前に先頭 20 問で予備実行し，education ノードの confidence がオフライン生確率より +0.05 高いことを直接確認すること（発火証拠）．Iter12/Iter22 のデプロイ漏れ教訓から，classifier.py 変更の commit → `mise run deploy`（smoke check の hash 確認）→ 本走の順を厳守すること．
  - **成功条件（config.yml レバー note 踏襲）**: (a) threshold=0.0 のオフライン再評価で実機 0.5118（87/170）と一致．(d) 実機 1600 問で education_recall > 0.5112．非退行: 他 9 ドメイン 18 指標の BH 補正後有意退行 0 件，argmax flip rate < 15%（オフライン Iter52b/53 で 2.56% 実績），top1_accuracy 非退行．
  - **コスト見積もり**: 実装 ~15-30 分（classifier.py 1 関数 + tests/test_classifier.py に education 加算の単体テスト）．(a) のオフライン再検証は ollama ノード 1 台で 1600 件の embedding 計算（evaluate_classifier_calibration.py は逐次処理，~10-30 分）．(d) 実機本走 1600 問 ~96-101 分（config timeout 150 分）＋ `mise run analyze`（B88 により answer_quality_accuracy 計測の復活も同時実施）．
  - **先行研究調査は不要**: 新規手法の検証ではなく adopted 済み決定（Iter52/53，backlog B78/B81）の実装漏れ是正のため，tavily-search は実施しなかった．リポジトリ内一次情報（d0004 §4，d0006 §3 表，backlog B76/B78/B81/B84/B88，journal_archive.md Iter44/Iter51/Iter52 節）で計画に必要な知見は揃っている．

