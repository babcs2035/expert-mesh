<!-- data/・models/ は .gitignore 対象のため，再現性を担保する目的で成果物のハッシュと生成コマンドのみここに記録する（docs/d0003 F5）．内容を再生成した場合は本ファイルも更新すること． -->

# data/ MANIFEST — 再現性のためのハッシュと生成コマンド

`data/`・`models/` はローカルディスク容量の都合で `.gitignore` されているため，実験結果の再現性は
このファイルに記録するハッシュと生成コマンドのみに依存する．各ファイルを再生成した際は，
`sha256sum <path>` を再実行し，値とコミットハッシュを更新すること．

## 評価データセット・分類器訓練データ

生成コマンド（`mise run setup` から呼ばれる／単体でも実行可能．build_dataset.py の docstring 参照）:

```
uv run python build_dataset.py \
    --output data/dataset.jsonl \
    --classifier-train-output data/classifier_train.jsonl
```

`_JMMLU_SAMPLE_SEED=20260726`（評価用）と `_CLASSIFIER_TRAIN_SAMPLE_SEED=20260727`（分類器訓練用）で
サンプリングシードを分離し，さらに質問本文単位でも重複排除している（docs/d0002 §2-1・§6-E で
本文重複 0 件を実測確認済み）．

**2026-07-30 更新（research_frontier 項目2 / d0003 X4）**: 複合ドメイン設問（`_COMPOUND_QUESTIONS`）を
20問→100問（43組み合わせ，10ドメイン全体をカバー）へ拡充した．JMMLU由来の単一ドメイン設問（1500問，
サンプリングシード・タスクマップとも無変更）と分類器訓練データは影響を受けず，
`data/classifier_train.jsonl` のハッシュは変更前と完全一致することを確認済み．

**2026-09-26 更新（Iteration 78, `compound_eval_set_expansion=llm_generated_separate_generator`）**:
LLM 生成 + 独立検証済みの複合設問 315 行（45 ドメインペア × 7 件，`compound-101`〜`compound-415`）を
`data/compound_questions_generated.jsonl` として純追加し，評価データセットを 1600 行→1915 行へ拡張した．
既存 100 行の複合設問（`compound-001`〜`compound-100`）と単一ドメイン 1500 行はビット単位で無変更
（`build_dataset.py:_build_rows()` の `_COMPOUND_QUESTIONS` ループの**後**に append するため）．
分類器訓練データ（`data/classifier_train.jsonl`）も無変更（ハッシュ一致，上表参照）．

生成コマンド（`scripts/generate_compound_eval_questions.py` の docstring 参照．生成器は
`judge_model` とは別系統の `qwen3.5:9b`，検証器は `judge_model`＝Swallow 8B．**wafl-ctrl5 限定**
（config.yml 絶対条件 B））:

```
# 第1パス（45ペア x 12件生成 -> 検証 -> 各ペア7件に切り詰め；16ペアが目標未達で270/315行）
uv run python -m scripts.generate_compound_eval_questions \
    --generator-model qwen3.5:9b \
    --verifier-model schroneko/llama-3.1-swallow-8b-instruct-v0.1:q4_k_m \
    --ollama-host 127.0.0.1 --ollama-port 11499 \
    --per-pair 12 --target-per-pair 7 \
    --output data/compound_questions_generated.jsonl \
    --multidomain-dedup-file data/classifier_train_multidomain.jsonl \
    --multidomain-dedup-file data/classifier_train_multidomain_iter61.jsonl \
    --multidomain-dedup-file data/classifier_train_multidomain_iter64.jsonl \
    --multidomain-dedup-file data/classifier_train_multidomain_iter65.jsonl

# 第2パス（不足16ペアだけを対象に、generate_all_rows()の同一実装をペア単体で再呼び出し。
# コード変更なし。近重複参照に第1パスの採択済み315件を含める。journal.md Iter78「実装・実験」節参照）
# -> 45行を追加生成し、第1パスの270行とマージして315行を確定。

uv run python build_dataset.py \
    --output data/dataset.jsonl \
    --classifier-train-output data/classifier_train.jsonl
```

| ファイル | sha256 | 行数 |
|---|---|---|
| `data/dataset.jsonl` | `2d4397542e67d71cec1d447ceb8f9636f67af254e885bd7e0eafd309a81d2f0c` | 1915 |
| `data/compound_questions_generated.jsonl` | `1bfb5add6a5f4028c3ceb52531987aaea595c409bfaca1c98bbf48c80901d3c6` | 315 |
| `data/classifier_train.jsonl` | `eb89bf7b0ad6303d41f2b668549f85362988de1eaee7b4faf98b3d3f5edcd9ef` | 1427（無変更） |

出典: JMMLU（`nlp-waseda/JMMLU`, commit `3637b25e444ccfdcde4d23a783cbe8e674faa01b`）．ライセンス CC BY-NC-ND 4.0．
複合設問の LLM 生成部分（`compound-101` 以降）は本リポジトリの生成・独立検証パイプラインの出力であり，
JMMLU 由来ではない．

## E6 教師あり分類器

生成コマンド（実機の ollama ノードが必要．scripts/train_domain_classifier.py の docstring 参照．
**wafl-ctrl5 限定**（config.yml 絶対条件 B））．

**2026-09-27 更新（Iteration 82 実装フェーズ．現行の生成コマンド）**: `embedding_view_concatenation`
（prefix なし ⊕ prefix あり の 2048 次元連結）採用により，`--embedding-instruction` と
`--embedding-view-concat` を両方渡す（片方だけだと `embed_query_views()` が `ValueError`）:

```
uv run python -m scripts.train_domain_classifier \
    --train-data data/classifier_train.jsonl \
    --embedding-model qwen3-embedding:0.6b \
    --embedding-instruction "Given a user question, identify the single academic or professional domain it belongs to" \
    --embedding-view-concat \
    --ollama-host 127.0.0.1 --ollama-port 11499 \
    --output models/domain_classifier.joblib
```

Iteration 81 までの生成コマンド（prefix なし・1024 次元．参考として残す）:

```
uv run python -m scripts.train_domain_classifier \
    --train-data data/classifier_train.jsonl \
    --embedding-model qwen3-embedding:0.6b \
    --ollama-host 127.0.0.1 --ollama-port 11499 \
    --output models/domain_classifier.joblib
```

**2026-09-27 訂正（Iteration 81 分析フェーズ）**: 直前の版はこの生成コマンドを
`--embedding-model bge-m3` と記載していたが，これは誤りである．Iteration 80（`bge_m3`）は
**rejected** で判定され，基準線（`qwen3-embedding:0.6b`，artifact `21e16ec6...`）への復元まで
実施済みである（journal.md「Iteration 80 実行済み」節・backlog B127）．MANIFEST 側が
Iteration 80 の実装フェーズ時点の記述のまま更新されていなかった．Iteration 81 も rejected で
復元済みのため，Iteration 82 開始時点の実機構成は上記（prefix なし・qwen3-embedding:0.6b，
`21e16ec6...`）だった．

| ファイル | sha256 |
|---|---|
| `models/domain_classifier.joblib`（現行．Iteration 82 実装フェーズで再訓練，qwen3-embedding:0.6b，prefix なし⊕prefix あり連結，2048次元，`n_features_in_`=2048） | `1cfcd3d836c5b5b421b8a47a487c3fb3ba996dd54aadcebae688cdfc8bcd0a48` |
| `models/domain_classifier_pre_iter82_noconcat.joblib`（Iter82 直前の退避＝Iter81 rejected 後の復元状態，qwen3-embedding:0.6b，prefixなし，1024次元） | `21e16ec63db89f4c8435264e98e6a704382aec723548359f6a74f44753f6ca19` |
| `models/domain_classifier_pre_iter81_noprefix.joblib`（Iter81 直前の退避，qwen3-embedding:0.6b，prefixなし，1024次元） | `21e16ec63db89f4c8435264e98e6a704382aec723548359f6a74f44753f6ca19` |
| `models/domain_classifier_pre_iter80_qwen3_0.6b.joblib`（Iter80 直前の退避，qwen3-embedding:0.6b，1024次元） | `21e16ec63db89f4c8435264e98e6a704382aec723548359f6a74f44753f6ca19` |
| `models/domain_classifier_pre_iter79_nomic.joblib`（Iter79 直前の退避，nomic-embed-text，768次元） | `02caf2b8e7a85972ff47867f05c8145e7d985e2000a2eb55662c6285db408905` |

**Iteration 81 の申し送り 1 への回答（2026-09-27 分析フェーズで確認・訂正済み）**:
`bge-m3` は Iteration 80 で **adopted ではなく rejected** であり，実機は基準線
（`qwen3-embedding:0.6b`，artifact `21e16ec6...`）へ復元済みだった．したがって Iteration 81 開始時点の
実機構成（`21e16ec6...`）が正しく，**誤っていたのは MANIFEST の記述の方**である（backlog B127 の
記録は正しい）．上記の生成コマンドとテーブルはこの訂正を反映済み．以下は訂正前の記述の経緯:

（旧記述）Iteration 80 で採用されたはずの `bge-m3`（sha256
`37d71b63...`）は，Iteration 81 開始時点の実機構成が `embedding_model=qwen3-embedding:0.6b`
（`models/domain_classifier_pre_iter81_noprefix.joblib` と sha256 一致）に戻っていた
（journal.md Iteration 81 開始前提の確認事項）．本反復はこの前提を検証対象とせず，与えられた
単一レバー `embedding_input_instruction_prefix` の実施のみを担当したため，`bge-m3` への
切替とその後の巻き戻しの経緯は分析・考察フェーズで確認すること．

オフライン性能（旧・nomic-embed-text，docs/d0002 §6-E）: 訓練 100.00%（1427/1427），評価 59.87%（898/1500，
1500 行版データセット当時の実測）．過学習の傾向が残る．

オフライン性能（qwen3-embedding:0.6b，Iteration 79 実装フェーズ実測）: 訓練 84.09%（1200/1427），
評価（argmax `selected_domain` が `expected_domains` に含まれる比率，dispatch/aggregation を経ない
分類器単体の値，dataset.jsonl 全 1915 行）75.35%（1443/1915）．旧モデルより訓練精度が下がっている
一方，評価精度は大きく上回っており，過学習が緩和された可能性がある（判定は分析・考察フェーズに委ねる）．

オフライン性能（新・bge-m3，Iteration 80 実装フェーズ実測）: 訓練 100.00%（1427/1427，p≫n による
完全適合），評価（同定義，dataset.jsonl 全 1915 行）72.48%（1388/1915）．qwen3-embedding:0.6b の
75.35% を下回る着地点予測（判定には用いない．本走の実測は分析・考察フェーズで確定する）．

G0（Iteration 80 VRAM 実測ゲート，wafl-ctrl5，`ollama ps`／`nvidia-smi`）: `qwen3-embedding:4b` を
`docker exec ollama-ctrl ollama stop qwen3-embedding:0.6b` 等で枠を空けたうえで pull・ロードし，
常駐サイズ 4.4GB・`100% GPU` を実測．依頼者ノード wafl500 のピーク予算（12288 MiB -
`light_model 3.1GB` - `expert_model 5.3GB` ≈ 3.4GB，合格条件 X+8.4GB≤11.5GB）を 1.3GB 超過し
**G0 不合格**．代替候補 `bge-m3` は常駐サイズ 0.664GB・`100% GPU` で合格（9.06GB ≤ 11.5GB）．
詳細と選定の経緯は backlog B126 参照．

G1（Iteration 80 事前スクリーニング，`data/classifier_train.jsonl` のみの 5-fold StratifiedKFold CV，
`scripts/screen_embedding_models.py`，3 候補同時評価）: `qwen3-embedding:0.6b`（基準線）
cv_accuracy=0.7561（macro-F1 0.7562），`qwen3-embedding:4b` cv_accuracy=0.7722（macro-F1 0.7718，
3 候補中最良だが G0 不合格のため選定対象外），`bge-m3` cv_accuracy=0.7120（macro-F1 0.7140）．
選定規則（G0 を通過した非現行候補のうち CV accuracy 最大）により `bge-m3` を選定．

G2（旧 qwen3-embedding:0.6b／新 bge-m3 artifact の argmax replay，dataset.jsonl 全 1915 行，
conformal 無し，`scripts.evaluate_classifier_calibration`）: discordant 行数 n_d=469
（必要偏り率 `1.96/sqrt(469)`=0.0905）．合格ライン n_d≥30 を満たす．

参考（Iteration 79 の G1/G2，埋め込みモデル nomic-embed-text→qwen3-embedding:0.6b）:
G1: `nomic-embed-text` cv_accuracy=0.5711（macro-F1 0.5710），`qwen3-embedding:0.6b`
cv_accuracy=0.7561（macro-F1 0.7562）．G2: discordant 行数 n_d=787（必要偏り率
`1.96/sqrt(787)`=0.0699）。

### Iteration 81（embedding_input_instruction_prefix=qwen3_instruct_classification_prefix）

埋め込みモデルは `qwen3-embedding:0.6b` のまま変えず，`/api/embeddings` へ渡す文字列に
Qwen3-Embedding の instruct 形式 prefix を付与する（`expert_backend.py:OllamaClient.embed()`
の `instruction` 引数，`config.yaml` の新キー `embedding_instruction`）．

選定文言（`config.yaml:embedding_instruction`，10 ドメイン共通の英語 1 文）:
`Given a user question, identify the single academic or professional domain it belongs to`
（node.py・train_domain_classifier.py 双方で `f"Instruct: {embedding_instruction}\nQuery: {text}"`
として付与．journal.md Iteration 81 計画の P1 に一致）．

F1（prefix有無の埋め込みコサイン類似度，wafl-ctrl5，同一日本語質問文 1 件）: cosine=0.7674
（< 0.999 の合格ラインを満たす．Ollama が prefix を実際に反映していることの直接証拠）．

F2（配布，deploy 後 wafl500〜509 全 10 ノード）: `config.yaml` の `embedding_instruction` 行と
`models/domain_classifier.joblib` の sha256 が全ノードで選定文言・新 artifact（`56d5a882...`）と
一致．

F3（実行時経路，先頭 20 問の予備実行 vs オフライン replay）: `selected_domain` が 20/20 一致
（dispatch 失敗 0 件，分母 20）．

G1（`data/classifier_train.jsonl` 1427 行のみの 5-fold StratifiedKFold CV，
`scripts/screen_embedding_models.py`，P0〜P3 の 4 候補同時評価，wafl-ctrl5）:
P0（prefixなし，基準線）cv_accuracy=0.7561231750705435（macro-F1 0.7562199648757597），
**P1（`Instruct: ...\nQuery: {text}`）cv_accuracy=0.7715225125751441（macro-F1
0.7690555039871017，4候補中最良）**，P2（`Instruct: ...\nInput: {text}`）
cv_accuracy=0.7666102318733898（macro-F1 0.7649437025621662），P3（モデル同梱既定 query
prompt）cv_accuracy=0.7547196662986138（macro-F1 0.7537554700089361）．選定規則（CV accuracy
最大）により P1 を採用．

G2（旧 qwen3-embedding:0.6b・prefixなし artifact `21e16ec6...`／新 qwen3-embedding:0.6b・
prefixあり(P1) artifact `56d5a882...` の argmax replay，dataset.jsonl 全 1915 行，自前
replay スクリプト．`predict_proba` argmax は sklearn 標準の `.predict()` を使用）:
discordant 行数 n_d=276（必要偏り率 `1.96/sqrt(276)`=0.1180）．合格ライン n_d≥30 を満たす．
オフライン accuracy（`expected_domains[0]` に対する argmax の一致率，着地点予測，判定には
不使用）: 旧 0.6840731070496083 → 新 0.7232375979112271．

再訓練コマンド（wafl-ctrl5 限定）:

```
uv run python -m scripts.train_domain_classifier \
    --train-data data/classifier_train.jsonl \
    --embedding-model qwen3-embedding:0.6b \
    --embedding-instruction "Given a user question, identify the single academic or professional domain it belongs to" \
    --ollama-host 127.0.0.1 --ollama-port 11499 \
    --output models/domain_classifier.joblib
```

キャッシュファイル（`scripts/screen_embedding_models.py` の G1 用，prefix 識別子つき）:
`data/embcache_qwen3-embedding_0.6b.npy`（P0，Iter79 のキャッシュを再利用），
`data/embcache_qwen3-embedding_0.6b__p1.npy`／`__p2.npy`／`__p3.npy`（P1〜P3，本反復で新規生成）．

本走実測（`results/20260927_024644/`，1,915 問フルスペック．基準線 `results/20260926_221822/`）:
top1_accuracy 全1915行 0.753003→0.789556（+3.655pt，McNemar continuity-corrected
chi2=25.5968，p=4.207e-7，discordant_a_only(基準線正解→新誤り)=58／discordant_b_only(基準線誤り→新正解)=128）．
既存1,600行部分集合（single 1500 + compound-001〜100）top1: 0.751250→0.781250．
複合415行（compound-001〜415）top1: 0.766265→0.843373．single_domain_top1:
0.749333→0.774667．非退行指標: fallback_rate 0.0→0.0，dispatch_failure_rate
0.000522→0.000522，ECE 0.032745→0.030951，mean_duration_ms 2301.4→2200.7．報告のみ:
answer_quality_accuracy 0.569333→0.58，end_to_end_accuracy 0.335770→0.350392，
compound_domain_set_recall 0.548193→0.526506，compound_mean_dispatched_count
1.880→1.494．

per-domain recall/precision 計20指標（BH補正 q=0.05，`metrics.py`の
`compute_domain_recall_mcnemar_test`／`compute_domain_precision_fisher_test`／
`apply_benjamini_hochberg` を使用．算出根拠・出典は各関数のdocstring参照）:
有意差3件—computer_science_recall（0.7273→0.8528，p≈4.93e-7，改善方向），
**medical_recall（0.7842→0.7178，p=0.003264，悪化方向）**，natural_science_recall
（0.5931→0.6623，p=0.004586，改善方向）．他17指標はBH補正後有意差なし．
medical_recallの有意退行は計画時点の非退行条件①（20指標BH補正後有意退行0件）に抵触する
実測結果であり，判定は分析・考察フェーズに委ねる．

**判定（2026-09-27 分析・考察フェーズ）: `rejected`**．事前登録の判定規則
「非退行①で有意退行1件以上なら rejected」に medical_recall が該当したため，主基準
（+3.655pt・McNemar p=4.207e-7）を満たしていても規則どおり rejected とした．
**復元実施済み**: `config.yaml` から `embedding_instruction` 行を削除，
`cp models/domain_classifier_pre_iter81_noprefix.joblib models/domain_classifier.joblib`
（`21e16ec6...`），`mise run deploy` 再実行．全ノードで `embedding_instruction` 不在・
artifact `21e16ec6...` を確認（smoke_check hashes/probe pass）．
**注意**: prefix 版 artifact（`56d5a882...`）はこの復元で上書きされ現存しない．必要なら
上記「再訓練コマンド」で決定論的に再生成できる（P1 文言・キャッシュ
`data/embcache_qwen3-embedding_0.6b__p1.npy` も保存済み）．

### Iteration 82（embedding_view_concatenation=prefix_and_noprefix_concat）

分類器の入力特徴量を「prefix なし埋め込み（1024次元）」から「**prefix なし ⊕ prefix あり（P1文言）の
2048次元**」へ変更．埋め込みモデル・instruction文言（P1，Iter81選定のまま再探索せず）・分類器ハイパラ・
訓練データ・評価集合は無変更．連結順序は `expert_backend.embed_query_views()` 内部に
`[prefix なし, prefix あり]` で固定（訓練側・実行時側とも同じ関数を呼ぶ．journal.md「Iteration 82」
Q3 参照）．連結後のスケーリング（1/√2 単位ノルム化）は不採用（backlog B129 (2)）．

再訓練コマンド: 本セクション冒頭「2026-09-27 更新」の生成コマンドを参照．

キャッシュファイル（ビュー識別子つき．いずれも wafl-ctrl5 で実測，`ssh -fNT -L 11499:localhost:11434 wafl-ctrl5`）:
- 訓練用（`data/classifier_train.jsonl` 1,427行）: `data/embcache_qwen3-embedding_0.6b.npy`（P0，Iter79
  のキャッシュを再利用）・`data/embcache_qwen3-embedding_0.6b__p1.npy`（P1，Iter81 のキャッシュを再利用）．
- 評価用（`data/dataset.jsonl` 1,915行，本反復で新規生成）: `data/embcache_eval_qwen3-embedding_0.6b.npy`（P0）・
  `data/embcache_eval_qwen3-embedding_0.6b__p1.npy`（P1）．

G1（`data/classifier_train.jsonl` 1,427行のみの5-fold StratifiedKFold CV，`scripts/screen_embedding_models.py`
（Iter82 で `{p0, p1, concat}` の3候補へ差し替え），wafl-ctrl5）: p0（基準線）cv_accuracy=0.756123
（macro-F1 0.756220），p1 cv_accuracy=0.771523（macro-F1 0.769056），**concat（p0⊕p1の2048次元）
cv_accuracy=0.784844（macro-F1 0.784022，3候補中最良）**．計画時点の事前実測値（0.756132/0.771549/0.784863）
と±1e-5で一致．

F1（連結の直接証拠，wafl-ctrl5，実クエリ1件）: 戻り値 len=2048，前半1024がprefixなしembedと完全一致，
後半1024がprefixありembedと完全一致，前半/後半のL2ノルムともに1.0±1e-5．4条件すべてPASS．

G2-a（訓練側到達）: 旧artifact（`21e16ec6...`，`n_features_in_`=1024）を
`models/domain_classifier_pre_iter82_noconcat.joblib`へ退避後に再訓練．新artifact
`1cfcd3d8...`，`n_features_in_`=2048．PASS。

G2-b（検出力，`data/dataset.jsonl`全1,915行のargmax replay）: 旧／新artifactのdiscordant行数
n_d=195（必要偏り率`1.96/sqrt(195)`=0.1404）．合格ラインn_d≥30を満たす。

G2'-a（replayの忠実度）: 旧artifactのreplay由来`selected_domain`が基準線`results/20260926_221822/`の
実測`selected_domain`と1914/1915=99.95%一致（合格ライン99%以上）。

G2'（本走前のper-domain非退行予測，`metrics.py`の既存統計関数をそのまま使用）: 新artifactのreplayで
per-domain 20指標のBH補正（q=0.05）を算出．有意退行0件（予測）。medical_recallの予測値は
0.7842→0.7552（p=0.190430，非有意）。合格条件なし（記録のみ，config.yml絶対条件Aにより本走は必ず実施）。

F2（配布，deploy後wafl500〜509全10ノード）: `config.yaml`の`embedding_instruction`・
`embedding_view_concat: true`行とartifact sha256 `1cfcd3d8...`が全ノードで一致。

F3（実行時経路，先頭20問の予備実行 vs オフラインreplay）: `selected_domain`が20/20一致
（dispatch失敗0件，分母20）。

本走実測（`results/20260927_050049/`，1,915問フルスペック，実測176分。基準線`results/20260926_221822/`）:
top1_accuracy 全1915行 0.753003→0.792167（+3.916pt，McNemar chi2=40.5630，p=1.904e-10，
discordant_a_only(基準線正解→新誤り)=30／discordant_b_only(基準線誤り→新正解)=105）。
既存1,600行部分集合（single 1500 + compound-001〜100）top1: 0.751250→0.786875。
複合415行top1: 0.766265→0.816867。single_domain_top1: 0.749333→0.785333。
Cohen's kappa: 0.721502→0.761499。非退行指標: fallback_rate 0.0→0.0，dispatch_failure_rate
0.000522→0.000522，ECE 0.032745→0.025448，mean_duration_ms 2301.4→2217.1（embed呼び出しが1回
増えたが所要時間は悪化しなかった）。報告のみ: answer_quality_accuracy 0.569333→0.572667，
end_to_end_accuracy 0.335770→0.351958，compound_domain_set_recall 0.548193→0.539759，
compound_mean_dispatched_count 1.880→1.610。

per-domain recall/precision計20指標（BH補正q=0.05，`metrics.py`の`compute_domain_recall_mcnemar_test`／
`compute_domain_precision_fisher_test`／`apply_benjamini_hochberg`を使用）: 有意差1件のみ—
computer_science_recall（0.7273→0.8268，p=4.4e-5，改善方向）。**medical_recall（0.7842→0.7510，
p=0.135593，非有意）は本反復の主目的である「有意退行を起こさない」を満たした**。他18指標も
BH補正後有意差なし。

G2'の予測（medical_recall 0.7842→0.7552，p=0.190430）と本走実測（0.7842→0.7510，p=0.135593）は
方向・非有意という結論の両方で一致（点推定差0.0042pt）。

**判定は分析・考察フェーズに委ねる**（本セクションは実装・実験フェーズの機械可読な実測値の記録のみ）。

## E10 ドメイン別 LoRA アダプタ

生成は3段階（各スクリプトの docstring 参照）:

```
# 1. JMMLU から評価データセットと分離した instruction-tuning データを作成
uv run python scripts/prepare_lora_training_data.py \
    --domains <domain> \
    --output-dir data/lora_train \
    --eval-dataset data/dataset.jsonl \
    --jmmlu-zip /path/to/JMMLU.zip

# 2. ドメインごとに LoRA アダプタを訓練（10 ノード並列で wall-clock 2〜4 時間，docs/d0002 §3-3 Iter18）
uv run python scripts/train_domain_lora.py \
    --model schroneko/llama-3.1-swallow-8b-instruct-v0.1 \
    --data data/lora_train/<domain>.jsonl \
    --output models/lora_adapters/<domain>/ \
    --lora-r 16 --lora-alpha 32 --epochs 3 --batch-size 2

# 3. Ollama へアダプタを登録（ノードごとに実行）
uv run python scripts/create_lora_model.py \
    --base schroneko/llama-3.1-swallow-8b-instruct-v0.1:q4_k_m \
    --adapter models/lora_adapters/<domain>/ \
    --name expert-mesh-<domain>-lora \
    --ollama-host <node_ip> --ollama-port 11434
```

`models/lora_adapters/<domain>/adapter.gguf`（Ollama の `ADAPTER` ディレクティブが実際に参照するファイル）の sha256:

| domain | sha256（adapter.gguf） |
|---|---|
| general | `c5f16bc4c4a93cf0ada78b5ba21405e724722cd946b13ca9073453f861cdf9e1` |
| education | `f3f43b93b0f56da95441782fcba38ac69f3853cbcb1468c4087140ceba82fdb5` |
| legal | `e75c92b0313a103e5464871c9d62375f0fc99dae2254c9c50ed8b5e7d8716a33` |
| medical | `6b5184fd08fdf8ea34f264a80e36fc629b58a448146805c2dad67caa8e2797b1` |
| business_economics | `6e94180f5a0b6554e604f7de65bacfcc76543e071aefdb7713d796f70f220bdb` |
| computer_science | `16b79141ece66a21ff8c65ad953c218b790474ff757fd9049ffc4b0d6da73a37` |
| natural_science | `4b8ea3b30ce481b8efedf7c828b7c646f99077e32f747f1e7b6e98366a9eca7d` |
| mathematics | `958e406025dd05537dab390cb6fd44d5f4815e804bac69436674d286f61d46bc` |
| history_culture | `631742d429a153e78d34d11b7e93cb2785f65c7ff5f3b8a6f91f6884b1f622c7` |
| social_science | `cdaffed7515fe4ffcec110d88da95cca2a30b4fc193e9f987cff16c4a8bc0ff1` |

2026-07-29 時点で，wafl500〜509 の Ollama に上記10種が全て登録済みであることを実機（`docker compose exec
ollama ollama list`）で確認済み．

## 記録日

2026-07-29．`git rev-parse HEAD` = `30e3627020c986dfd24a3b0a4c0cdd26d1136b85`（本ファイル作成時点）．
以降にこれらのファイルを再生成した場合は，このセクションと各ハッシュを更新すること．
