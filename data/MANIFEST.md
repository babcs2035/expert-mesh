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
| `data/dataset.jsonl`（Iteration 78 時点） | `2d4397542e67d71cec1d447ceb8f9636f67af254e885bd7e0eafd309a81d2f0c` | 1915 |
| `data/compound_questions_generated.jsonl` | `1bfb5add6a5f4028c3ceb52531987aaea595c409bfaca1c98bbf48c80901d3c6` | 315 |
| `data/classifier_train.jsonl` | `eb89bf7b0ad6303d41f2b668549f85362988de1eaee7b4faf98b3d3f5edcd9ef` | 1427（無変更） |

出典: JMMLU（`nlp-waseda/JMMLU`, commit `3637b25e444ccfdcde4d23a783cbe8e674faa01b`）．ライセンス CC BY-NC-ND 4.0．
複合設問の LLM 生成部分（`compound-101` 以降）は本リポジトリの生成・独立検証パイプラインの出力であり，
JMMLU 由来ではない．

**2026-09-27 更新（Iteration 85, `single_domain_eval_set_expansion=jmmlu_unused_rows_power_targeted`）**:
JMMLU の未使用単一ドメイン行を `data/single_domain_expansion_iter85.jsonl` として純追加し，評価データセットを
1915 行→**3435 行**（単一 3020 ＋ 複合 415）へ拡張した．既存 1915 行はバイト単位で無変更（`head -n 1915`
で確認済み．`build_dataset.py:_build_rows()` の複合行ループの**後**に append するため）．分類器訓練データ
（`data/classifier_train.jsonl`）も無変更（ハッシュ一致，上表参照）．`models/domain_classifier.joblib` は
Iter84 実行前の基準線（`1cfcd3d8...`，下記 E6 節参照）へ復元して使用した．

**計画（journal.md「Iteration 85」計画節・backlog B137）からの実測乖離**: 計画は 1,532 行の追加
（合計 3447 行）を見込んでいたが，実際に生成されたのは **1,520 行**（合計 3435 行）だった．原因は
本リポジトリがキャッシュする JMMLU.zip の sha256 が `3ba7d912...`（ピン留めコミット `3637b25e...` が
期待するものと異なる．journal_archive.md に既記載の既知の無害な差異）で，`computer_science`
（計画 156→実測 151）・`social_science`（148→144）・`general`（28→25）の 3 ドメインで実際のプールが
計画時の推定よりわずかに小さかったため．他 7 ドメイン（medical・education・business_economics・
natural_science・mathematics・history_culture は計画通り 200，legal は計画通り 0）は計画値と一致．
検出力条件（P1: `1.96·√(p_d/N)` ≤ 1.10pt）は N=3435 でも 1.028pt で引き続き満たす（設計値 1.026pt から
0.002pt の劣化のみ）．`scripts/expand_single_domain_eval.py` はこの乖離を実行時に許容範囲内として記録し，
`main()` が計画値と実測値の両方を stderr に出力する．

生成コマンド（`scripts/expand_single_domain_eval.py` の docstring 参照．LLM・埋め込み不要，CPU のみ）:

```
uv run python scripts/expand_single_domain_eval.py \
    --output data/single_domain_expansion_iter85.jsonl \
    --jmmlu-zip /path/to/JMMLU.zip

uv run python build_dataset.py --output data/dataset.jsonl
```

| ファイル | sha256 | 行数 |
|---|---|---|
| `data/dataset.jsonl`（Iteration 85 時点） | `a48fcf17ec1116732c611425eab800a2a92f7921211bd928f99a5d1e9cb77958` | 3435 |
| `data/single_domain_expansion_iter85.jsonl` | `b880b29c26f481fc9e7a24dff0901f34d7d495273acafe949174f515abec94c4` | 1520 |

**2026-09-28 更新（Iteration 90, `compound_eval_set_expansion=llm_generated_separate_generator_scaleup_730`）**:
複合設問（`compound-416`〜`compound-730`）を LLM 生成 + 独立検証で 315 行追加し，
`data/compound_questions_generated.jsonl` を **315 行 → 630 行**（45 ドメインペア × 14 件）へ純追加．
評価データセットは 3,435 行 → **3,750 行**（単一 3,020 ＋ 複合 730）へ拡張した．既存 315 行
（`compound-101`〜`compound-415`）はビット単位で無変更（`head -n 315` の diff 0 バイトで確認済み）．
単一ドメイン 3,020 行・分類器訓練データ（`data/classifier_train.jsonl`）も無変更．

生成器 `qwen3.5:9b` と検証器（`judge_model`＝Swallow 8B）は Iter78 と一字も変えていない．CLI の
`scripts/generate_compound_eval_questions.py --output` は毎回ファイル全体を上書きするため（`--target-per-pair 14`
を直接渡すと 630 行を新規サンプリングし直し，既存 315 行の純追加という B118 の条件を満たせない），
本反復では同スクリプトの `generate_all_rows()` を Iter78 の「追い上げパス」と同じ手法で直接呼び出し，
既存 315 行を近重複参照に含めたうえで **新規 7 件/ペアだけ**を生成し，既存ファイルの末尾へ追記した
（スクリプト本体は無変更）:

```
# 第1パス（45ペア x 12件生成 -> 検証 -> 各ペア7件に切り詰め；14ペアが目標未達で273/315行）
# 第2パス（不足14ペアの計42件だけを対象に generate_all_rows() をペア単体で再呼び出し。
#   近重複参照に第1パスの採択済み273件も追加。コード変更なし）
# どちらのパスも scripts/generate_compound_eval_questions.py の内部関数を直接呼ぶ小さな
# 使い捨てドライバから実行（同スクリプトの --output は全体上書きのため、CLI 直接実行では
# 既存315行の純追加にならない。ドライバの引数は以下と同一）:
#   generator_model=qwen3.5:9b, verifier_model=schroneko/llama-3.1-swallow-8b-instruct-v0.1:q4_k_m,
#   ollama_host=127.0.0.1, ollama_port=11499, per_pair=12,
#   near_duplicate_reference_texts に既存315行 + 既存100件手作成 + classifier_train.jsonl +
#   classifier_train_multidomain*.jsonl（計3,345件参照）を使用
# 新規315行を既存315行の末尾へ追記（cat 済み315行 + 新規315行 = 630行）

uv run python build_dataset.py \
    --output data/dataset.jsonl \
    --classifier-train-output data/classifier_train.jsonl
```

品質確認（3-gram Jaccard，`scripts/generate_compound_eval_questions.py:_char_ngram_jaccard` を再利用）:
新規 315 行と既存 315 行・既存 100 件手作成の全 415 件参照に対する近重複（≥0.6）**0 件**（最大値
0.1484）．新規 315 行どうしの pairwise 近重複も **0 件**（最大値 0.1411）．45 ペアすべてちょうど 14 行．

| ファイル | sha256 | 行数 |
|---|---|---|
| `data/dataset.jsonl`（Iteration 90 時点，現行） | `2114e04880cf94f9ab16d4615aa9abd764166ba122f7615d7a4b54d7fd7fb2ee` | 3750 |
| `data/compound_questions_generated.jsonl`（現行） | `da0ddd915e7ae851ec03479c894c0826655823734883b6cd7867bfbcee252c4f` | 630 |

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

### Iteration 84（cross_domain_training_data_augmentation=hard_negative_mining_all_domains）

`config.yaml`・`node.py`・`aggregator.py`・`classifier.py`・`train_domain_classifier.py`・
`build_dataset.py` は無変更（`git diff --stat` は `scripts/mine_hard_negatives.py`（新規）と
`data/MANIFEST.md`・`.claude/research/*` のみ）。新規 `scripts/mine_hard_negatives.py` が，
`data/classifier_train.jsonl`（1,427行，本反復でも無変更，sha256 `eb89bf7b...`）に対し，
現行 artifact の `predict_proba` による正解ドメイン確率 `p_true` が低い JMMLU 未使用行
（10ドメイン共通規則，各ドメイン上限100件，legalのみプール0件）を各ドメイン100行追加した
`data/classifier_train_iter84_hardneg.jsonl` を生成する（詳細な選択規則は journal.md
「Iteration 84」計画節を参照．本ファイルは生成コマンドと実測値のみを記録する）。

生成コマンド（wafl-ctrl5 限定）:

```
uv run python -m scripts.mine_hard_negatives \
    --train-data data/classifier_train.jsonl \
    --eval-data data/dataset.jsonl \
    --classifier-model models/domain_classifier.joblib \
    --per-domain 100 \
    --jmmlu-zip <local JMMLU.zip> \
    --embedding-model qwen3-embedding:0.6b \
    --embedding-instruction "Given a user question, identify the single academic or professional domain it belongs to" \
    --embedding-view-concat \
    --ollama-host 127.0.0.1 --ollama-port 11499 \
    --output data/classifier_train_iter84_hardneg.jsonl

uv run python -m scripts.train_domain_classifier \
    --train-data data/classifier_train_iter84_hardneg.jsonl \
    --embedding-model qwen3-embedding:0.6b \
    --embedding-instruction "Given a user question, identify the single academic or professional domain it belongs to" \
    --embedding-view-concat \
    --ollama-host 127.0.0.1 --ollama-port 11499 \
    --output models/domain_classifier.joblib
```

プール実測（`--per-domain 100` 適用前，JMMLU 全 CSV 行のうち重複行・評価集合 1,915 行・
既存訓練 1,427 行を除いた件数）: medical 1110 / history_culture 779 / natural_science 776 /
business_economics 705 / mathematics 348 / computer_science 251 / social_science 244 /
education 232 / general 125 / legal 0（計 4,570）。journal.md 計画節の事前登録値（natural_science
783・education 233・合計 4,578）とは各 7 件・1 件だけ少ない。原因は同一 JMMLU タスク CSV 内に
literal に重複した設問行が存在したため（`college_physics` 内 6 件・`conceptual_physics` 内 1 件・
`high_school_psychology` 内 1 件；ドメイン間の重複ではなく同一タスク内の重複であることを実データで
確認済み）。本スクリプトはプール構築時に設問文の重複を除去するため件数がその分減るが，
各ドメインとも選定数 N=100 に対し十分な余裕があり，選定結果（各ドメイン100件，legalのみ0件，
計900件）に影響はない。

| ファイル | sha256 | 行数 |
|---|---|---|
| `data/classifier_train.jsonl`（無変更） | `eb89bf7b0ad6303d41f2b668549f85362988de1eaee7b4faf98b3d3f5edcd9ef` | 1427 |
| `data/classifier_train_iter84_hardneg.jsonl`（新規） | `fc28f1c66a5548df33721aaeefd23879db9592661fb7c08ba5bdfc61a64f73b6` | 2327（既存1427 + 追加900） |
| `models/domain_classifier.joblib`（現行．Iter84 で再訓練，`n_features_in_`=2048） | `34e4d33bcfb0695a48b410cb70fd1d3d38973e048631b1c1ffe516df423b34c9` | — |
| `models/domain_classifier_pre_iter84_baseline.joblib`（Iter84 直前の退避 = Iter83 本番，`21e16ec6...`系統ではなく`1cfcd3d8...`） | `1cfcd3d836c5b5b421b8a47a487c3fb3ba996dd54aadcebae688cdfc8bcd0a48` | — |

キャッシュファイル（wafl-ctrl5，`ssh -fNT -L 11499:localhost:11434 wafl-ctrl5`）:
`data/embcache_pool_qwen3-embedding_0.6b.npy`（プール4,570行のprefixなし view）・
`data/embcache_pool_qwen3-embedding_0.6b__p1.npy`（同 prefixあり view）・
`data/embcache_pool_qwen3-embedding_0.6b.npy.meta.json`（プールの query 列ハッシュとレコード件数）。

F2（データ）: 新ファイル2,327行．先頭1,427行が`data/classifier_train.jsonl`とバイト一致。
追加900行のドメイン別内訳 `{legal: 0, 他9ドメイン: 100}`。追加900行は評価集合1,915行・既存訓練
1,427行の設問文と重複0件，追加900行同士の重複も0件，id重複0件。すべてPASS。

F3（artifact）: 新artifact sha256 `34e4d33b...`（基準線`1cfcd3d8...`と異なる），`n_features_in_`=2048，
`classes_`は10ドメイン。退避ファイルsha256は基準線と一致。PASS。

G5（本走前のreplay，`/tmp/iter84_replay.py`＋新artifact＋キャッシュ済み評価集合埋め込み，
gt=0.36）: top1_accuracy予測=0.806266（事前登録レンジ0.800〜0.815内），compound_domain_set_recall
予測=0.573494（非退行②の下限0.539759を上回る），compound_mean_dispatched_count予測=2.048193
（非退行③の上限2.10以内）。

F4（配布，deploy後wafl500〜509全10ノード）: 全ノードでartifact sha256が`34e4d33b...`に一致，
`config.yaml`の`dispatch_gap_threshold: 0.36`も全ノードで一致。smoke_check（git-status/hashes/probe）
すべてPASS。

F5（実行時経路，`data/dataset_20.jsonl`先頭20問の予備実行 vs オフラインreplay）: `confidence`・
`dispatched_domains`が20/20一致（dispatch失敗0件）。PASS。

本走実測（`results/20260927_110526/`，1,915問フルスペック，実測約63分。基準線は
Iter83本走`results/20260927_070239/`）:
top1_accuracy 全1915行 0.792689→0.804700（+1.201pt，Wilson 95% CI [0.786342, 0.821838]，
McNemar continuity-corrected chi2=2.674033，**p=0.101997（非有意）**，discordant_a_only(基準線正解→
新誤り)=79／discordant_b_only(基準線誤り→新正解)=102，検出限界 `1.96*sqrt(181)/1915`=1.378pt
（実測Δ1.201ptはこの検出限界を下回る）。
既存1,600行部分集合（single 1500 + compound-001〜100）top1: 0.787500→0.808125。
複合415行top1: 0.816867→0.795181。single_domain_top1: 0.786000→0.807333。
Cohen's kappa: 0.761499→0.785973。漏洩72行（B125既知，education46/history_culture26）を除いた
1,843行top1: 0.804700算出用の1915行に対し0.811177（1843行のみ）。

非退行指標: fallback_rate 0.0→0.0（PASS），dispatch_failure_rate 0.000522→0.001567（≤0.005，PASS），
ECE 0.026→0.075591（≤0.08，PASS，ただし基準線比+4.96pt），mean_duration_ms 2286.9→2280.408
（≤2744.3，PASS），rank1以外が選ばれた行数 2→3（≤10，PASS）。
compound_domain_set_recall 0.581928→0.573494（非退行②の下限0.539759以上，PASS）。
compound_mean_dispatched_count 1.889157→2.048193（非退行③の上限2.10以内，PASS，Iter83比+8.4%）。
報告のみ: answer_quality_accuracy 0.570667→0.580667，end_to_end_accuracy 0.350914→0.362924，
Random/BestSingle/Oracle baseline 0.121671/0.126893/1.0（不変）。

per-domain recall/precision 計20指標（BH補正q=0.05，`metrics.py`の
`compute_domain_recall_mcnemar_test`／`compute_domain_precision_fisher_test`／
`apply_benjamini_hochberg`を使用）: **有意差2件**—**education_recall（0.412→0.313，p=0.000427，
悪化方向）**，history_culture_recall（0.801→0.870，p=0.000796，改善方向）。他18指標はBH補正後
有意差なし。per-domain詳細（baseline→new，recall/precision）:
business_economics 0.8182/0.7746→0.8139/0.7611，computer_science 0.8268/0.8884→0.8571/0.9041，
education 0.4120/0.5304→0.3133/0.6759，general 0.4626/0.8750→0.5022/0.8769，
history_culture 0.8009/0.7400→0.8701/0.7701，legal 0.6626/0.8610→0.6420/0.9123，
mathematics 0.6710/0.8564→0.7013/0.8757，medical 0.7510/0.8153→0.7427/0.7553，
natural_science 0.6494/0.8571→0.6580/0.7958，social_science 0.4545/0.7554→0.5108/0.7239。

**判定は分析・考察フェーズに委ねる**（本セクションは実装・実験フェーズの機械可読な実測値の記録のみ）。

### Iteration 86（cross_domain_training_data_augmentation=hard_negative_mining_all_domains_retry_expanded_set）

`config.yaml`・`node.py`・`aggregator.py`・`classifier.py`・`train_domain_classifier.py`・
`build_dataset.py`・`data/dataset.jsonl`・`scripts/mine_hard_negatives.py` は無変更（`git diff --stat` は
`data/classifier_train_iter86_hardneg.jsonl`（新規）・`models/`・`data/MANIFEST.md`・`.claude/research/*`
のみ）。`scripts/mine_hard_negatives.py` を1行も変更せず，`--eval-data data/dataset.jsonl` を
拡充後の3,435行のまま再実行することで，Iter84の900行（追加分）とIter85拡充分1,520行が355行重複していた
漏洩（journal.md「Iteration 86」調査節Q1参照）を解消した，漏洩0件の訓練集合を作成した。

**G0（調達元の同一性）**: 採掘に用いた `/tmp/expert-mesh-cache/JMMLU.zip` の sha256
`3ba7d912943ede44fb7ec06aa1df067ac6bee157e65c5588736b9b983f0e684d` は MANIFEST 記録値と**一致**
（PASS）。したがって件数一致条項を適用する。

生成コマンド（wafl-ctrl5のOllamaへ `ssh -fNT -L 11499:localhost:11434 wafl-ctrl5` トンネル経由，
`--output`/`--train-data` のみIter84コマンドから差し替え，他は1文字も変更せず）:

```
uv run python -m scripts.mine_hard_negatives \
    --train-data data/classifier_train.jsonl \
    --eval-data data/dataset.jsonl \
    --classifier-model models/domain_classifier.joblib \
    --per-domain 100 \
    --jmmlu-zip /tmp/expert-mesh-cache/JMMLU.zip \
    --embedding-model qwen3-embedding:0.6b \
    --embedding-instruction "Given a user question, identify the single academic or professional domain it belongs to" \
    --embedding-view-concat \
    --ollama-host 127.0.0.1 --ollama-port 11499 \
    --output data/classifier_train_iter86_hardneg.jsonl

uv run python -m scripts.train_domain_classifier \
    --train-data data/classifier_train_iter86_hardneg.jsonl \
    --embedding-model qwen3-embedding:0.6b \
    --embedding-instruction "Given a user question, identify the single academic or professional domain it belongs to" \
    --embedding-view-concat \
    --ollama-host 127.0.0.1 --ollama-port 11499 \
    --output models/domain_classifier.joblib
```

プール実測（評価集合3,435行・既存訓練1,427行を除外後，過去の`classifier_train_iter84_hardneg.jsonl`
900行は今回3,435行の評価集合に含まれているため二重排除は不要）: business_economics 505 /
computer_science 100 / education 109 / general 100 / history_culture 579 / mathematics 148 /
medical 910 / natural_science 576 / social_science 100 / legal 0（計3,127）。journal.md計画節の事前登録値
（各ドメインとも一致）と完全一致。選定結果は各ドメイン100件（computer_science・social_science・general
はプール=N=100のため無作為抽出と同値，education は109件からの選定），legalのみ0件，計900件。

| ファイル | sha256 | 行数 |
|---|---|---|
| `data/classifier_train.jsonl`（無変更） | `eb89bf7b0ad6303d41f2b668549f85362988de1eaee7b4faf98b3d3f5edcd9ef` | 1427 |
| `data/classifier_train_iter86_hardneg.jsonl`（新規） | `bbd4163ceaa4368a044194220d9cb20fb72cebf3163bcb844b63cd5af523764f` | 2327（既存1427 + 追加900） |
| `models/domain_classifier.joblib`（現行．Iter86で再訓練，`n_features_in_`=2048） | `fd1ccd7d3e85c3340b5e44ffc116af078fed61bb1c2f63a9ac8f92583bb21ba4` | — |
| `models/domain_classifier_iter86_hardneg.joblib`（上と同一内容の複製．Iter84の教訓で必須保存） | `fd1ccd7d3e85c3340b5e44ffc116af078fed61bb1c2f63a9ac8f92583bb21ba4` | — |
| `models/domain_classifier_pre_iter84_baseline.joblib`（退避．変更なし） | `1cfcd3d836c5b5b421b8a47a487c3fb3ba996dd54aadcebae688cdfc8bcd0a48` | — |

F2（データ）: 新ファイル2,327行。先頭1,427行が`data/classifier_train.jsonl`とバイト一致
（sha256一致，確認済み）。追加900行のドメイン別内訳 `{legal: 0, 他9ドメイン: 100}`。**追加900行は
評価集合3,435行との設問文重複0件，既存1,427行との重複0件，追加行同士の重複0件，id重複0件**（Python で
実測，全項目0件）。すべてPASS。

F3（artifact）: 新artifact sha256 `fd1ccd7d...`（基準線`1cfcd3d8...`・Iter84`34e4d33b...`のいずれとも異なる），
`n_features_in_`=2048，`classes_`は10ドメイン。退避ファイルsha256は基準線と一致。PASS。

本走前の着地点記録（新artifact ＋ 旧1,915行キャッシュ埋め込みでreplay，gt=0.36．3,435行全体分の
キャッシュ埋め込みは存在しないため1,915行部分集合でのみ算出）: top1_accuracy(1,915部分集合)予測=
0.803133，compound_domain_set_recall予測=0.555422，compound_mean_dispatched_count予測=1.925301。

F4（配布，deploy後wafl500〜509全10ノード）: 全ノードでartifact sha256が`fd1ccd7d...`に一致，
`docker compose exec app wc -l data/dataset.jsonl`が全ノードで3435に一致，`config.yaml`の
`dispatch_gap_threshold: 0.36`も全ノードで一致。smoke_check（git-status/hashes/probe）すべてPASS。

F5（実行時経路，`data/dataset_20.jsonl`先頭20問の予備実行 vs オフラインreplay）: `confidence`
（10ドメイン全件）・`dispatched_domains`が20/20一致（不一致0件）。PASS。

本走実測（`results/20260927_152208/`，3,435問フルスペック，実測約100分。基準線はIter85本走
`results/20260927_130237/`）:

主基準: top1_accuracy 全3,435行 0.786317→0.796507（**+1.019pt**，Wilson 95% CI [0.782715, 0.809635]，
`metrics.py`の`compute_mcnemar_test`使用，continuity-corrected chi2=4.056140，**p=0.044011（有意，
α=0.05）**，discordant_a_only(基準線正解→新誤り)=125／discordant_b_only(基準線誤り→新正解)=160。
主基準(i)有意・(ii)Δ≥+1.0pt（0.796507≥0.796317）はいずれも**PASS**（(ii)の余裕は+0.019pt分と僅少）。

報告のみ: single_domain_top1_accuracy 0.782119→0.797020（n=3020），compound_domain_top1_accuracy
0.816867→0.792771（n=415），1,915行部分集合top1 0.802611（本走前replay予測0.803133と乖離0.05pt，
忠実度は良好）。層別Δ（基準線p_true五分位）は分析フェーズで別途算出のこと（本セクションでは未算出）。

非退行①: per-domain recall/precision計20指標（BH補正q=0.05，`metrics.py`の
`compute_domain_recall_mcnemar_test`／`compute_domain_precision_fisher_test`／
`apply_benjamini_hochberg`を使用）: **有意差4件**—**education_recall（0.387991→0.344111，
p=0.003948，悪化方向）**，history_culture_recall（0.823666→0.883991，p=0.000010，改善方向），
natural_science_recall（0.733179→0.774942，p=0.004607，改善方向），natural_science_precision
（0.872928→0.802885，p=0.008897，悪化方向）。他16指標はBH補正後有意差なし。
per-domain詳細（baseline→iter86，recall/precision）:
business_economics 0.8283/0.7951→0.8445/0.7647，computer_science 0.8796/0.8912→0.8691/0.9146，
education 0.3880/0.5638→0.3441/0.6712，general 0.4841/0.7673→0.5040/0.8194，
history_culture 0.8237/0.7100→0.8840/0.7257，legal 0.6626/0.7594→0.6420/0.8041，
mathematics 0.8005/0.8824→0.7912/0.9045，medical 0.7574/0.8087→0.7642/0.7801，
natural_science 0.7332/0.8729→0.7749/0.8029，social_science 0.5520/0.7667→0.5733/0.7847。
**education_recallの有意退行が2反復連続（Iter84 p=0.000427，Iter86 p=0.003948）で再現した**
（判定は分析・考察フェーズに委ねる）。

非退行②（複合被覆）: compound_domain_set_recall 0.581928→0.555422（下限0.539759以上，**PASS**）。
非退行③（複合予算）: compound_mean_dispatched_count 1.889157→1.925301（上限2.10以内，**PASS**）。
非退行④: fallback_rate 0.0→0.0（**PASS**），dispatch_failure_rate 0.001164→0.000291
（≤0.005，**PASS**）。
非退行⑤: rank1以外が選ばれた行数 5→1（≤15，**PASS**）。
非退行⑥: mean_duration_ms 1574.096→1534.531（≤1888.9，**PASS**）。
非退行⑦: ECE 0.018509→0.040154（≤0.08，**PASS**，ただし基準線比+2.16pt）。

報告のみ: answer_quality_accuracy 0.587417→0.571523，end_to_end_accuracy 0.403202→0.397671
（`mise run analyze`実測，`graded_row_count`=3020/`total_row_count`=3435），
Random/BestSingle/Oracle baseline 0.112082/0.128384/1.0（不変，`best_single_domain_baseline`の
ドメイン別最大値はmedical 0.128384）。

不変条件確認: `total_questions`=3435（一致），`compound_domain_question_count`=415（一致），
新artifact sha256は基準線・Iter84のいずれとも不一致。**追加900行と評価集合の重複は0件**（絶対条件）。

**判定は分析・考察フェーズに委ねる**（本セクションは実装・実験フェーズの機械可読な実測値の記録のみ）。

### Iteration 87（cross_domain_training_data_augmentation=hard_random_hybrid_all_domains）

`config.yaml`・`node.py`・`aggregator.py`・`classifier.py`・`train_domain_classifier.py`・
`build_dataset.py`・`data/dataset.jsonl`・`data/classifier_train.jsonl` は無変更（`git diff --stat` は
`scripts/mine_hard_negatives.py`・`data/classifier_train_iter87_hybrid.jsonl`（新規）・`models/`・
`data/MANIFEST.md`・`.claude/research/*` のみ）。`scripts/mine_hard_negatives.py` の `select_hard_negatives()`
を「最難100」から「最難50 ＋ 残プールから無作為50（`numpy.random.default_rng(87)`）」へ変更し，CLI に
`--random-count`（既定0）・`--random-seed`（既定0）を追加した（`build_pool`／`determine_used_tasks`／
`_pool_hash`／`write_output` は1行も変更していない）。

**前提条件の復元（F0）**: `models/domain_classifier.joblib` が Iter86 artifact（`fd1ccd7d...`）のまま
残っていたため，実装フェーズの最初に `cp models/domain_classifier_pre_iter84_baseline.joblib
models/domain_classifier.joblib` で復元し，sha256が`1cfcd3d836c5b5b421b8a47a487c3fb3ba996dd54aadcebae688cdfc8bcd0a48`
であることを確認した．`mise run deploy` 後，全10ノードで同sha256の一致・smoke_check（git-status/hashes/probe）
PASSを確認した．**この復元を経てから採掘した**（採掘のp_trueは復元後のartifactで計算されている）．

**G0（調達元の同一性）**: 採掘に用いた `/tmp/expert-mesh-cache/JMMLU.zip` の sha256
`3ba7d912943ede44fb7ec06aa1df067ac6bee157e65c5588736b9b983f0e684d` はMANIFEST記録値と**一致**（PASS）。
プール埋め込みキャッシュ（`pool_hash=221e45e8...`，3,127行）も現在のプールと一致したため再埋め込みは
発生せず，実質CPUのみで採掘が完了した。

生成コマンド（wafl-ctrl5のOllamaへ `ssh -fNT -L 11499:localhost:11434 wafl-ctrl5` トンネル経由，Iter86
コマンドに `--random-count 50 --random-seed 87` を追加，`--output` のみ差し替え，他は1文字も変更せず）:

```
uv run python -m scripts.mine_hard_negatives \
    --train-data data/classifier_train.jsonl \
    --eval-data data/dataset.jsonl \
    --classifier-model models/domain_classifier.joblib \
    --per-domain 100 \
    --random-count 50 --random-seed 87 \
    --jmmlu-zip /tmp/expert-mesh-cache/JMMLU.zip \
    --embedding-model qwen3-embedding:0.6b \
    --embedding-instruction "Given a user question, identify the single academic or professional domain it belongs to" \
    --embedding-view-concat \
    --ollama-host 127.0.0.1 --ollama-port 11499 \
    --output data/classifier_train_iter87_hybrid.jsonl

uv run python -m scripts.train_domain_classifier \
    --train-data data/classifier_train_iter87_hybrid.jsonl \
    --embedding-model qwen3-embedding:0.6b \
    --embedding-instruction "Given a user question, identify the single academic or professional domain it belongs to" \
    --embedding-view-concat \
    --ollama-host 127.0.0.1 --ollama-port 11499 \
    --output models/domain_classifier.joblib
```

プール実測（Iter86と完全に同一のプール定義であることを確認済み）: business_economics 505 /
computer_science 100 / education 109 / general 100 / history_culture 579 / mathematics 148 /
medical 910 / natural_science 576 / social_science 100 / legal 0（計3,127）。計画節の事前登録値と完全一致。
選定結果は各ドメイン100件（「最難50 ＋ 残プールから無作為50」，computer_science・social_science・general
はプール=N=100のため無作為側が残り全量で確定し**Iter86と集合として完全一致（no-op）**），legalのみ0件，計900件。

| ファイル | sha256 | 行数 |
|---|---|---|
| `data/classifier_train.jsonl`（無変更） | `eb89bf7b0ad6303d41f2b668549f85362988de1eaee7b4faf98b3d3f5edcd9ef` | 1427 |
| `data/classifier_train_iter87_hybrid.jsonl`（新規） | `63e73c201a8bf46a47c9933b33cc53ab28ada6033d0d6f33d63c71cf0359aa7d` | 2327（既存1427 + 追加900） |
| `models/domain_classifier.joblib`（現行．Iter87で再訓練，`n_features_in_`=2048） | `f6c33edb1b80a43dbd4d7153b9088b97d220d4557958d0149fe043f0c47594d2` | — |
| `models/domain_classifier_iter87_hybrid.joblib`（上と同一内容の複製） | `f6c33edb1b80a43dbd4d7153b9088b97d220d4557958d0149fe043f0c47594d2` | — |
| `models/domain_classifier_pre_iter84_baseline.joblib`（退避．変更なし） | `1cfcd3d836c5b5b421b8a47a487c3fb3ba996dd54aadcebae688cdfc8bcd0a48` | — |

F1（変更の最小性）: `git diff --stat` は `scripts/mine_hard_negatives.py`・`data/classifier_train_iter87_hybrid.jsonl`
（新規）・`models/`・`data/MANIFEST.md`・`.claude/research/*` のみ。`config.yaml`・`node.py`・`classifier.py`・
`aggregator.py`・`train_domain_classifier.py`・`build_dataset.py` の diff 0行。`data/dataset.jsonl`・
`data/classifier_train.jsonl` の sha256 不変。`mine_hard_negatives.py` の diff は `select_hard_negatives()`・
`_run()`の呼び出し・CLI引数追加・docstringのみ（`build_pool`／`determine_used_tasks`／`_pool_hash`／
`write_output`は1行も変更なし）。`ruff check`・`py_compile` PASS。PASS。

F2（データ）: 新ファイル2,327行。先頭1,427行が`data/classifier_train.jsonl`とバイト一致（sha256一致，
確認済み）。追加900行のドメイン別内訳 `{legal: 0, 他9ドメイン: 100}`。**追加900行は評価集合3,435行との
設問文重複0件，既存1,427行との重複0件，追加行同士の重複0件，id重複0件**（Pythonで実測，全項目0件）。
**computer_science・social_science・generalの追加100行はIter86の同ドメイン100行と集合として完全一致**
（no-opの確認，実測でPASS）。**Iter86全体900行との入れ替わり実測行数は219行**（事前登録の期待214±20行
の範囲内）。すべてPASS。

F3（artifact）: 新artifact sha256 `f6c33edb...`（基準線`1cfcd3d8...`・Iter86`fd1ccd7d...`のいずれとも異なる），
`n_features_in_`=2048，`classes_`は10ドメイン。退避ファイルsha256は基準線と一致。PASS。

本走前の着地点記録（新artifact ＋ 旧1,915行キャッシュ埋め込みでreplay，`aggregator.select_dispatch_targets()`
をそのまま再利用，gt=0.36．3,435行全体分のキャッシュ埋め込みは存在しないため1,915行部分集合でのみ算出）:
top1_accuracy(1,915部分集合)予測=0.810966，compound_domain_set_recall予測=0.566265，
compound_mean_dispatched_count予測=1.879518。（replay手法の忠実度は，同じ手法を基準線artifactへ適用すると
Iter83のG1記録値0.793211と完全一致することで確認済み。）

F4（配布，deploy後wafl500〜509全10ノード）: 全ノードでartifact sha256が`f6c33edb...`に一致，
`docker compose exec app wc -l data/dataset.jsonl`が全ノードで3435に一致，`config.yaml`の
`dispatch_gap_threshold: 0.36`も全ノードで一致。smoke_check（git-status/hashes/probe）すべてPASS。

F5（実行時経路，`data/dataset_20.jsonl`先頭20問の予備実行 vs オフラインreplay）: `confidence`
（10ドメイン全件）・`dispatched_domains`が20/20一致（不一致0件）。PASS。

本走実測（`results/20260927_174150/`，3,435問フルスペック，実測約90分。基準線はIter85本走
`results/20260927_130237/`）:

主基準: top1_accuracy 全3,435行 0.786317→0.801456（**+1.514pt**，Wilson 95% CI [0.787782, 0.814456]，
`metrics.py`の`compute_mcnemar_test`使用，continuity-corrected chi2=11.508850，**p=0.000693（有意，
α=0.05）**，discordant_a_only(基準線正解→新誤り)=87／discordant_b_only(基準線誤り→新正解)=139，
discordant_pairs=226，有意境界`1.96・√(n_d)/3435`=0.858pt）。
主基準(i)有意・(ii)Δ≥+1.0pt（0.801456≥0.796317）はいずれも**PASS**。

報告のみ: single_domain_top1_accuracy 0.782119→0.799007（n=3020），compound_domain_top1_accuracy
0.816867→0.819277（n=415），1,915行部分集合top1（実測）0.809922（本走前replay予測0.810966と乖離
0.104pt，忠実度は良好，`selected_domain`行一致率1913/1915=99.90%）。

**層別Δ（基準線p_true五分位，`probe_candidates`のうち各行のexpected_domainsに一致するものの最大confidence
で層別。境界値はIter86の記録値と完全一致することを確認済み）**:

| 層 | n | 上限p_true | 基準線 | Iter87 | Δ | 参考: Iter86 |
|---|---|---|---|---|---|---|
| Q1 | 687 | 0.3542 | 0.0451 | 0.1630 | **+11.79pt** | +14.85pt |
| Q2 | 687 | 0.7209 | 0.8923 | 0.8632 | −2.91pt | −6.84pt |
| Q3 | 687 | 0.9022 | 1.0000 | 0.9898 | −1.02pt | −2.33pt |
| Q4 | 687 | 0.9697 | 1.0000 | 0.9942 | −0.58pt | −1.02pt |
| Q5 | 687 | 0.9985 | 0.9942 | 0.9971 | +0.29pt | +0.44pt |

非退行①: per-domain recall/precision計20指標（BH補正q=0.05，`metrics.py`の
`compute_domain_recall_mcnemar_test`／`compute_domain_precision_fisher_test`／
`apply_benjamini_hochberg`を使用）: **有意差1件**—**history_culture_recall（0.823666→0.870070，
p=0.000105，改善方向）**。**education_recall（p=0.072486）・natural_science_precision（p=0.251250）は
いずれもBH補正後有意差なし**（Iter84・Iter86で連続していたeducation_recallの有意退行は本反復では
再現しなかった）。他18指標もBH補正後有意差なし。
per-domain詳細（baseline→iter87，recall/precision）:
business_economics 0.8283/0.7951→0.8492/0.7922，computer_science 0.8796/0.8912→0.8770/0.9128，
education 0.3880/0.5638→0.3626/0.6461，general 0.4841/0.7673→0.5119/0.8113，
history_culture 0.8237/0.7100→0.8701/0.7184，legal 0.6626/0.7594→0.6831/0.8137，
mathematics 0.8005/0.8824→0.7935/0.8976，medical 0.7574/0.8087→0.7710/0.8000，
natural_science 0.7332/0.8729→0.7564/0.8424，social_science 0.5520/0.7667→0.5787/0.7750。

非退行②（複合被覆）: compound_domain_set_recall 0.581928→0.566265（下限0.539759以上，**PASS**）。
非退行③（複合予算）: compound_mean_dispatched_count 1.889157→1.879518（上限2.10以内，**PASS**）。
非退行④: fallback_rate 0.0→0.0（**PASS**），dispatch_failure_rate 0.001164→0.001456（5行，
≤0.005，**PASS**）。
非退行⑤: rank1以外が選ばれた行数（`selected_domain`≠`probe_candidates`最大confidenceドメイン）
5→5（≤15，**PASS**）。
非退行⑥: mean_duration_ms 1574.096→1544.912（≤1888.9，**PASS**）。
非退行⑦: ECE 0.018509→0.045717（≤0.08，**PASS**）。

報告のみ: answer_quality_accuracy 0.587417→0.587417（実質不変），end_to_end_accuracy 0.403202→0.411063
（+0.786pt，`mise run analyze`実測，`graded_row_count`=3020/`total_row_count`=3435，ノイズ床3SD=2.6pt以内），
Random/BestSingle/Oracle baseline 0.112082/0.128384/1.0（不変，`best_single_domain_baseline`の
ドメイン別最大値はmedical 0.128384）。ECE=0.045717・Brier=0.126108・AUROC=0.820982（n=3430）。

不変条件確認: `total_questions`=3435（一致），`compound_domain_question_count`=415（一致），
新artifact sha256は基準線・Iter86のいずれとも不一致。**追加900行と評価集合の重複は0件**（絶対条件）。
本走top1（1,915行部分集合実測0.809922）はオフラインreplay予測（0.810966）から0.104pt乖離のみ
（1.0pt未満）。invalid条件はいずれにも該当しない。

**判定は分析・考察フェーズに委ねる**（本セクションは実装・実験フェーズの機械可読な実測値の記録のみ）。

### Iteration 88（cross_domain_training_data_augmentation=hard_random_hybrid_ratio_25_75_all_domains）

`config.yaml`・`node.py`・`aggregator.py`・`classifier.py`・`scripts/mine_hard_negatives.py`・
`scripts/train_domain_classifier.py`・`build_dataset.py`・`data/dataset.jsonl`・`data/classifier_train.jsonl`
は無変更（`git diff --stat` は `data/MANIFEST.md`・`.claude/research/*` のみで，`scripts/` 配下の diff は 0 行）。
採掘コマンドの `--random-count` を Iter87 の `50` から `75` へ変更し，`--classifier-model` を
`models/domain_classifier_pre_iter84_baseline.joblib`（`1cfcd3d836c5...`）へ明示指定，`--output` のみ差し替え，
他は Iter87 のコマンドから1文字も変更していない（F0: 採掘前に `models/domain_classifier.joblib` が
`f6c33edb...`（Iter87，全10ノード配布済み）であることを確認済み。採掘のための上書き復元は行っていない）:

```
uv run python -m scripts.mine_hard_negatives \
    --train-data data/classifier_train.jsonl \
    --eval-data data/dataset.jsonl \
    --classifier-model models/domain_classifier_pre_iter84_baseline.joblib \
    --per-domain 100 \
    --random-count 75 --random-seed 87 \
    --jmmlu-zip /tmp/expert-mesh-cache/JMMLU.zip \
    --embedding-model qwen3-embedding:0.6b \
    --embedding-instruction "Given a user question, identify the single academic or professional domain it belongs to" \
    --embedding-view-concat \
    --ollama-host 127.0.0.1 --ollama-port 11499 \
    --output data/classifier_train_iter88_hybrid2575.jsonl

uv run python -m scripts.train_domain_classifier \
    --train-data data/classifier_train_iter88_hybrid2575.jsonl \
    --embedding-model qwen3-embedding:0.6b \
    --embedding-instruction "Given a user question, identify the single academic or professional domain it belongs to" \
    --embedding-view-concat \
    --ollama-host 127.0.0.1 --ollama-port 11499 \
    --output models/domain_classifier.joblib
```

プール実測（Iter86/87 と完全に同一，`pool_hash=221e45e8...`，計3,127行）: business_economics 505 /
computer_science 100 / education 109 / general 100 / history_culture 579 / mathematics 148 /
medical 910 / natural_science 576 / social_science 100 / legal 0。選定結果は各ドメイン100件
（「最難25 ＋ 残プールから無作為75」，computer_science・social_science・generalはプール=N=100のため
無作為側が残り全量で確定し**Iter86・Iter87と集合として完全一致（no-op）**），legalのみ0件，計900件。

| ファイル | sha256 | 行数 |
|---|---|---|
| `data/classifier_train.jsonl`（無変更） | `eb89bf7b0ad6303d41f2b668549f85362988de1eaee7b4faf98b3d3f5edcd9ef` | 1427 |
| `data/classifier_train_iter88_hybrid2575.jsonl`（新規） | `c341baef5e7c42c92490ed4ca76a4e726933e4e7e8b8a9049a70056f636fd502` | 2327（既存1427 + 追加900） |
| `models/domain_classifier.joblib`（現行．Iter88で再訓練，`n_features_in_`=2048） | `51b9ced581f275b0726dfc12f709bb82124f1e780969a5c602582d8b1eb40b34` | — |
| `models/domain_classifier_iter88_hybrid2575.joblib`（上と同一内容の複製） | `51b9ced581f275b0726dfc12f709bb82124f1e780969a5c602582d8b1eb40b34` | — |

**F2ゲート実測**: 出力2,327行，先頭1,427行が`data/classifier_train.jsonl`とバイト一致。追加900行と評価集合
3,435行・既存1,427行・追加行同士の重複はいずれも0件，id重複0件。ドメイン別追加数は{legal: 0, 他9ドメイン各100}
（計画表と完全一致）。cs・social_science・generalの追加100行はIter86・Iter87の同ドメイン100行と集合として
完全一致（no-op確認，PASS）。**入れ替わり行数（domain+queryの同一性ベース，idはhard/random比率変更で
番号が振り直されるため raw line 一致では測れないことを確認した上で算出）: Iter87の追加900行との入れ替わり
実測309行，Iter86の追加900行との入れ替わり実測300行と，事前登録の予測値に完全一致（PASS）**。
F2の決定論的検証: 同じ採掘コマンドの`--random-count`を`0`/`50`に替えた一時出力が，それぞれ
`data/classifier_train_iter86_hardneg.jsonl`／`data/classifier_train_iter87_hybrid.jsonl`と
バイト完全一致することを確認した（確認後に一時ファイルは削除済み）。

**F3ゲート**: 新artifactのsha256（`51b9ced5...`）は`1cfcd3d8...`（pre_iter84）・`fd1ccd7d...`（Iter86）・
`f6c33edb...`（Iter87）のいずれとも不一致。`n_features_in_`=2048，`classes_`は10ドメイン。

**F4ゲート**: `mise run deploy`後，全10ノード（wafl500〜509）でartifact sha256=`51b9ced5...`一致，
`docker compose exec app wc -l /app/data/dataset.jsonl`=3435（10/10）を実機確認（`smoke_check`のgit-status/
hashes/probeもPASS）。

**F5ゲート**: 予備20問（`data/dataset_20.jsonl`）の`confidence`・`selected_domain`・`dispatched_domains`が，
新artifact＋既存埋め込みキャッシュのオフライン予測と**20/20完全一致**。

**5b: 種87/88/89 × 比50/50・25/75 の必須report-only計測（B143・B144(C)）**

6構成のうち50/50種87（`models/domain_classifier_iter87_hybrid.joblib`）と25/75種87
（`models/domain_classifier_iter88_hybrid2575.joblib`）は既存artifactを再利用，残り4構成
（50/50種88・89，25/75種88・89）は追加採掘・再訓練した。`data/embcache_eval_qwen3-embedding_0.6b{,__p1}.npy`
（1,915行，`[plain, instructed]`の順で連結し2048次元）を特徴量として`predict_proba`し，argmax（=rank_1，
`aggregation_method=max_confidence`下では`selected_domain`と一致）で1,915行部分集合のtop1を算出した
（本手法は基準線artifactに適用すると50/50種87の値が Iter87 実測の本走前replay予測 0.810966 と完全一致し，
忠実度を確認済み）。

| 構成 | replay top1（1,915行） |
|---|---|
| 50/50 種87 | 0.810966 |
| 50/50 種88 | 0.813055 |
| 50/50 種89 | 0.809922 |
| 25/75 種87 | 0.806266 |
| 25/75 種88 | 0.809399 |
| 25/75 種89 | 0.807833 |

比ごとの3種平均/SD（標本SD）/レンジ: 50/50 = 平均0.811314・SD 0.001595・レンジ0.003133（種間max0.31pt）。
25/75 = 平均0.807833・SD 0.001567・レンジ0.003133（種間max0.31pt）。
**同一比・異種間のΔの絶対値の最大値は両比とも0.003133（0.31pt）**。これは本反復の事前投影（比の効果
+0.2〜+0.5pt，点推定+0.28pt）と同程度の規模であり，種由来のノイズと比の効果が同オーダーであることを
示す（判定はしない。分析フェーズの結論に委ねる）。

**本走前の着地点記録（B136，25/75種87artifactで実施）**: 1,915行部分集合replay top1=0.806266，
compound_domain_set_recall予測0.553012，compound_mean_dispatched_count予測1.783133。
per-domain（predicted，1,915部分集合）: education precision/recall=0.6149/0.3906，
natural_science precision/recall=0.8427/0.6494，medical precision/recall=0.8117/0.8050，
legal precision/recall=0.8953/0.6337，history_culture precision/recall=0.7529/0.8312。

**本走**: `results/20260927_202256/`（3,435問，約91分で完走，timeout 180分以内）。

**主要メトリクス**

全体（3,435問）: top1_accuracy = 0.801456（Iter87基準線）→ **0.801164**（**Δ = −0.03pt**，Wilson 95%CI
[0.787484, 0.814172]），single_domain_top1 = 0.799007→0.798344（n=3,020），
compound_top1 = 0.819277→0.821687（n=415），fallback_rate = 0.0→0.0，
dispatch_failure_rate = 0.001456→0.000291（1行），mean_duration_ms = 1544.912→1546.784。

主基準: `compute_mcnemar_test`でcontinuity-corrected chi2=0.0，**p=1.0（非有意，α=0.05）**，
discordant_a_only（Iter87正解→Iter88誤り）=64／discordant_b_only（Iter87誤り→Iter88正解）=63，
discordant_pairs=127，有意境界`1.96・√(n_d)/3435`=0.612pt。**主基準(i)非有意・(ii)Δ=−0.03pt<+1.0pt**
（判定語の確定は分析フェーズに委ねる）。

1,915行部分集合（実測）top1 = 0.805222（本走前replay予測0.806266と乖離0.104pt，`selected_domain`行一致率
1913/1915 = 99.90%）。

**層別Δ（基準線p_true の五分位，pre_iter84の実測run `results/20260927_130237/` の`probe_candidates`から
p_true を実測して算出．各n=687，境界値がIter86/87の記録値（Q1≤0.3542／Q2≤0.7209／Q3≤0.9022／Q4≤0.9697）と
完全一致することを確認した上で算出）**

| 層 | 上限p_true | pre_iter84 | Iter87基準線 | Iter88 | Δ（vs Iter87基準線） | Δ（vs pre_iter84，用量反応3点目） |
|---|---|---|---|---|---|---|
| Q1 | 0.3542 | 0.0451 | 0.1630 | 0.1354 | **−2.77pt** | +9.02pt |
| Q2 | 0.7209 | 0.8923 | 0.8632 | 0.8748 | +1.16pt | −1.75pt |
| Q3 | 0.9022 | 1.0000 | 0.9898 | 0.9985 | +0.87pt | −0.15pt |
| Q4 | 0.9697 | 1.0000 | 0.9942 | 0.9985 | +0.44pt | −0.15pt |
| Q5 | 0.9985 | 0.9942 | 0.9971 | 0.9985 | +0.15pt | +0.44pt |

参考（用量反応3点，pre_iter84比）: Q1 +14.85pt（Iter86）→+11.79pt（Iter87）→**+9.02pt（Iter88）**，
Q2 −6.84→−2.91→**−1.75pt**，Q3 −2.33→−1.02→**−0.15pt**，Q4 −1.02→−0.58→**−0.15pt**，
Q5 +0.44→+0.29→**+0.44pt**。全体Δtop1（pre_iter84比）は+1.019pt（Iter86）→+1.514pt（Iter87）→
**+1.485pt（Iter88，0.801164−0.786317）**。

非退行①: per-domain recall/precision計20指標（BH補正q=0.05，`metrics.py`の`compute_domain_recall_mcnemar_test`／
`compute_domain_precision_fisher_test`／`apply_benjamini_hochberg`）で**有意退行0件**（legal_recall生p=0.003283
・mathematics_recallの生p=0.013328はBH補正前は小さいが，BH閾値（20件中の順位×0.05/20）を上回らず非有意）。
**education_recall（0.3626→0.3811，+1.85pt，生p=0.098960，非有意）・natural_science_precision（0.8424→0.8541，
+1.17pt，生p=0.687276，非有意）はいずれもBH補正後有意差なし**。他18指標もBH補正後有意差なし。
per-domain詳細（Iter88実測，recall/precisionの順，前セクションと同じ表記）:
business_economics 0.8538/0.7914，computer_science 0.8665/0.9068，education 0.3811/0.6274，
general 0.5040/0.8038，history_culture 0.8585/0.7283，legal 0.6337/0.7979，
mathematics 0.8121/0.8906，medical 0.7846/0.8141，natural_science 0.7471/0.8541，
social_science 0.5840/0.7631。

非退行②（複合被覆）: compound_domain_set_recall 0.566265→0.553012（下限0.539759以上，**PASS**）。
非退行③（複合予算）: compound_mean_dispatched_count 1.879518→1.783133（上限2.10以内，**PASS**）。
非退行④: fallback_rate 0.0→0.0（**PASS**），dispatch_failure_rate 0.001456→0.000291（1行，
≤0.005，**PASS**）。
非退行⑤: rank1以外が選ばれた行数（`selected_domain`≠`probe_candidates`最大confidenceドメイン）
5→1（≤15，**PASS**）。
非退行⑥: mean_duration_ms 1544.912→1546.784（≤1853.9，**PASS**）。
非退行⑦: ECE 0.045717→0.022189（≤0.08，**PASS**）。

報告のみ: answer_quality_accuracy 0.587417→0.585762，end_to_end_accuracy 0.411063→0.407860
（`mise run analyze`実測，`graded_row_count`=3020/`total_row_count`=3435，ノイズ床3SD=2.6pt以内），
Random/BestSingle/Oracle baseline 0.112082/0.128384/1.0（不変，`best_single_domain_baseline`の
ドメイン別最大値はmedical 0.128384）。Brier=0.124825・AUROC=0.822177（n=3434）。

不変条件確認: `total_questions`=3435（一致），`compound_domain_question_count`=415（一致），
新artifact sha256は`1cfcd3d8...`／`fd1ccd7d...`／`f6c33edb...`のいずれとも不一致。**追加900行と評価集合の
重複は0件**（絶対条件）。本走top1（1,915行部分集合実測0.805222）はオフラインreplay予測（0.806266）から
0.104pt乖離のみ（1.0pt未満）。invalid条件はいずれにも該当しない。

**判定は分析・考察フェーズに委ねる**（本セクションは実装・実験フェーズの機械可読な実測値の記録のみ）。

### Iteration 89（embedding_model_replacement=qwen3_embedding_4b）

`config.yaml:4`の`embedding_model`のみ変更（`qwen3-embedding:0.6b`→`qwen3-embedding:4b`，1行）。
`node.py`・`aggregator.py`・`classifier.py`・`train_domain_classifier.py`・`build_dataset.py`・
`expert_backend.py`・`data/dataset.jsonl`・`data/classifier_train_iter87_hybrid.jsonl`は無変更
（`expert_backend.py`のR2補正〈`num_ctx`指定〉はG0-a/G0-bがR1で合格したため未適用）。

**G0（VRAM・所要時間．静的予算式ではなく実機実測で判定）**

- G0-a（wafl-ctrl5）: `bge-m3`・swallow-8Bを`ollama stop`で退避後，`qwen3-embedding:4b`をロードして
  常駐**4.4GB・100% GPU**を確認（**PASS**）。
- G0-b（deploy後，全10ノード`ollama ps`）: 初回計測でwafl500・wafl507の2ノードのみ
  `qwen3-embedding:4b`が`56%/44% CPU/GPU`（`qwen3.5:4b-q4_K_M`との共存でVRAM逼迫）。
  **R1**（全10ノードで`ollama stop qwen3.5:4b-q4_K_M`→embeddingsエンドポイント呼び出しで再ロード）を適用し，
  全10ノードで`qwen3-embedding:4b`が**100% GPU**であることを確認（**PASS**，R1で着地。R2/R3は不要）。
- G0-c（予備20問，`data/dataset.jsonl`先頭20行）: HTTP 500エラー**0件**，平均所要**1288.65ms**
  （≤3090ms＝基準線`mean_duration_ms`の2倍，**PASS**）。

**G1（report-only．`data/classifier_train_iter87_hybrid.jsonl` 2,327行のみの5-fold StratifiedKFold CV．
評価集合3,435行は未参照）**

| 埋め込みモデル | cv_accuracy | cv_macro_f1 | 次元 |
|---|---|---|---|
| `qwen3-embedding:0.6b`（基準線と同一構成） | 0.751615 | 0.747161 | 2048（1024×2ビュー） |
| `qwen3-embedding:4b` | **0.804465** | 0.798825 | 5120（2560×2ビュー） |

+5.29ptのCV差（本走top1の予測値としては扱わない，計画節Q3の解釈規則どおり）。

**G2（検出力の事前登録．評価3,435行のargmax replay，`metrics.compute_mcnemar_test`を流用）**

旧artifact（`f6c33edb...`，0.6b・2048次元）と新artifact（4b・5120次元で新規訓練した候補モデル）を
3,435行で`predict`し，`id`でペアリングしてMcNemar検定を実施:

- discordant_a_only=119, discordant_b_only=220, discordant_pairs=**339**（≥30，**PASS**）
- 旧argmax accuracy 0.802620 → 新argmax accuracy 0.832023（**+2.940pt**）
- 本走のMcNemar有意境界（事前算出）: `1.96·√339/3435` = **1.05pt**

**変更したもの**

1. `config.yaml:4` `embedding_model`: `qwen3-embedding:0.6b`→`qwen3-embedding:4b`（1行．本レバーの本体）。
2. `models/domain_classifier.joblib`: 4b・2ビュー連結（5120次元）で再訓練。
   訓練コマンド（Iter87節から`--embedding-model`のみ差し替え，wafl-ctrl5のOllamaへ
   `ssh -fNT -L 11499:localhost:11434 wafl-ctrl5`トンネル経由）:
   ```
   uv run python -m scripts.train_domain_classifier \
       --train-data data/classifier_train_iter87_hybrid.jsonl \
       --embedding-model qwen3-embedding:4b \
       --embedding-instruction "Given a user question, identify the single academic or professional domain it belongs to" \
       --embedding-view-concat \
       --ollama-host 127.0.0.1 --ollama-port 11499 \
       --output models/domain_classifier.joblib
   ```
   旧版は`models/domain_classifier_pre_iter89_qwen3_0.6b.joblib`（sha256`f6c33edb...`，Iter87と同一）へ退避。
   新artifactは`models/domain_classifier_iter89_4b.joblib`にも複製。
3. `data/embcache_*`: 新規作成・再生成（すべて2ビュー`{plain, __p1}`）。
   - `data/embcache_train_iter89_qwen3-embedding_0.6b{,__p1}.npy`（G1比較用，2,327行×1024次元）
   - `data/embcache_train_iter89_qwen3-embedding_4b{,__p1}.npy`（G1・本番訓練特徴，2,327行×2560次元）
   - `data/embcache_eval_qwen3-embedding_0.6b{,__p1}.npy`（G2用，**3,435行**へ再生成，旧キャッシュは1,915行のみだった）
   - `data/embcache_eval_qwen3-embedding_4b{,__p1}.npy`（G2用，新規，3,435行×2560次元）

| ファイル | sha256 |
|---|---|
| `models/domain_classifier.joblib`（現行．Iter89で再訓練，`n_features_in_`=5120） | `ff8aad9cf824992f0a99d07d5506ea9ebdbc3491914a165959c3496df7f2cfd6` |
| `models/domain_classifier_iter89_4b.joblib`（上と同一内容の複製） | `ff8aad9cf824992f0a99d07d5506ea9ebdbc3491914a165959c3496df7f2cfd6` |
| `models/domain_classifier_pre_iter89_qwen3_0.6b.joblib`（退避．Iter87と同一） | `f6c33edb1b80a43dbd4d7153b9088b97d220d4557958d0149fe043f0c47594d2` |
| `data/classifier_train_iter87_hybrid.jsonl`（無変更） | `63e73c201a8bf46a47c9933b33cc53ab28ada6033d0d6f33d63c71cf0359aa7d` |

到達確認: 全10ノードで`ollama list`に`qwen3-embedding:4b`存在・`config.yaml`の`embedding_model: qwen3-embedding:4b`一致・
`models/domain_classifier.joblib`のsha256が`ff8aad9c...`で一致・`n_features_in_`=5120を確認済み（`mise run deploy`の
smoke_check・本フェーズでのssh実測の双方）。`ruff check`は変更ファイル起因の新規エラー0件（既存のリポジトリ内
未修正エラーのみ）。`uv run pytest`は既存9件FAIL（無関係，リポジトリ既知のB122）＋309件PASSで新規失敗なし。

**本走**（`results/20260927_232950/`，3,435問1回。基準線はIter87本走`results/20260927_174150/`，top1=0.801456）

主要指標（`uv run python metrics.py --results results/20260927_232950/results.jsonl --json`実測）:

| 指標 | 基準線（Iter87） | Iter89 | 差 |
|---|---|---|---|
| **top1_accuracy（全3,435）** | 0.801456（95%CI [0.787782, 0.814456]） | **0.830859**（95%CI [0.817954, 0.843024]） | **+2.940pt** |
| single_domain_top1（3,020） | — | 0.836093 | — |
| compound_domain_top1（415） | — | 0.792771 | — |
| cohens_kappa | — | 0.816651 | — |
| misrouting_rate | — | 0.169141 | — |
| mean_duration_ms | 1544.912 | 1643.249 | +98.34 |
| ECE | 0.045717 | 0.023073 | -0.0226 |
| Brier | — | 0.110203 | — |
| AUROC | — | 0.838008 | — |

McNemar検定（`metrics.compute_mcnemar_test`，id完全対応3,435行）: discordant_a_only=123／
discordant_b_only=224／discordant_pairs=347／p値=**7.949e-08**（有意）。

per-domain 20指標のBH補正（q=0.05，`metrics.compute_domain_recall_mcnemar_test`／
`compute_domain_precision_fisher_test`／`apply_benjamini_hochberg`）: **有意差4件，いずれも改善方向**
（recall:business_economics 0.8492→0.9188 p=2.84e-05／recall:education 0.3626→0.4180 p=5.98e-03／
precision:general 0.8113→0.9214 p=6.71e-03／precision:mathematics 0.8976→0.9514 p=5.75e-03）。
**有意な退行は0件**。他16指標もBH補正後有意差なし。
per-domain詳細（recall/precisionの順）: business_economics 0.9188/0.8182，computer_science 0.9058/0.9377，
education 0.4180/0.6351，general 0.5119/0.9214，history_culture 0.8701/0.7622，legal 0.6255/0.8261，
mathematics 0.8167/0.9514，medical 0.8073/0.8109，natural_science 0.7842/0.8622，social_science 0.6107/0.8297。

非退行②（複合被覆）: compound_domain_set_recall 0.566265→0.562651（下限0.539759以上，**PASS**）。
非退行③（複合予算）: compound_mean_dispatched_count 1.879518→1.903614（上限2.10以内，**PASS**）。
非退行④: fallback_rate 0.0→0.0（**PASS**），dispatch_failure_rate 0.001456→0.001164（**PASS**，≤0.005）。
非退行⑤: rank1以外が選ばれた行数（`selected_domain`≠`probe_candidates`最大confidenceドメイン） 5→4（≤15，**PASS**）。
非退行⑥: mean_duration_ms 1544.912→1643.249（≤1853.9，**PASS**）。
非退行⑦: ECE 0.045717→0.023073（≤0.08，**PASS**）。

報告のみ: answer_quality_accuracy 0.577152，end_to_end_accuracy 0.417467（`mise run analyze`実測，
`graded_row_count`=3020/`total_row_count`=3435）。

不変条件確認: `total_questions`=3435（一致），`compound_domain_question_count`=415（一致），
新artifact sha256（`ff8aad9c...`）は基準線（`f6c33edb...`）・Iter84〜88のいずれとも不一致。
本走top1（0.830859）はG2オフラインreplay予測（argmax accuracy 0.832023）から**0.116pt乖離のみ**
（1.0pt未満）。invalid条件（G0失敗・n_d<30・total_questions不一致・sha256/n_features_in_不一致）は
いずれにも該当しない。

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

**2026-09-28 更新（Iteration 91, `classifier_regularization_strength=l2_C_selected_by_train_only_cv`,
C=10.0）**: `scripts/train_domain_classifier.py` に定数 `_L2_INVERSE_REGULARIZATION = 10.0` を追加し，
`train_classifier()` の `base_estimator` へ `C=_L2_INVERSE_REGULARIZATION` を渡すよう変更した（コード
2 行＋定数定義のみ．他の設定・コードは無変更）．埋め込みは `data/classifier_train_iter87_hybrid.jsonl`
2,327 行から wafl-ctrl5 の Ollama（127.0.0.1:11499 トンネル経由，wafl500〜509 は不使用）で再計算した．

訓練コマンド（Iter89 節から `--output`／内部 `C` のみ差し替え，他は同一構成）:
```
uv run python -m scripts.train_domain_classifier \
    --train-data data/classifier_train_iter87_hybrid.jsonl \
    --embedding-model qwen3-embedding:4b \
    --embedding-instruction "Given a user question, identify the single academic or professional domain it belongs to" \
    --embedding-view-concat \
    --ollama-host 127.0.0.1 --ollama-port 11499 \
    --output models/domain_classifier.joblib
```

旧版は `models/domain_classifier_pre_iter91_c1.joblib`（sha256 `ff8aad9c...`，Iter89 と同一）へ退避．
新 artifact は `models/domain_classifier_iter91_c10.joblib` にも複製．オフライン replay 専用の
C=1.0 対照 artifact（**絶対にデプロイしない**）は `/tmp/iter91/domain_classifier_iter91_c1_control.joblib`
（リポジトリ外，再現用コマンドは journal.md Iteration 91 参照）．

再訓練時に新規計算した 2,327 行×2ビューの埋め込みと `data/embcache_train_iter89_qwen3-embedding_4b{,__p1}.npy`
の最大絶対差は **0.011406**（qwen3-embedding:4b のセッション跨ぎ非決定性，journal.md B149 (d) 参照）．

| ファイル | sha256 |
|---|---|
| `models/domain_classifier_iter91_c10.joblib`（Iter91 本走で使った `C`=10.0 版．**現行ではない**） | `6a5905f025d6dae91ab333968e71c944d77ba286f4f3e3891c1b3a1fe5ca3b30` |
| `models/domain_classifier_pre_iter91_c1.joblib`（Iter89 と同一．`C`=1.0） | `ff8aad9cf824992f0a99d07d5506ea9ebdbc3491914a165959c3496df7f2cfd6` |
| `data/classifier_train_iter87_hybrid.jsonl`（無変更） | `63e73c201a8bf46a47c9933b33cc53ab28ada6033d0d6f33d63c71cf0359aa7d` |

到達確認 E1〜E5・G0 の実測は journal.md Iteration 91「実行済み」節を参照．

**[2026-09-28 Iter91 分析フェーズ．artifact をロールバックした]**
Iter91 の判定は `no_effect`（Δtop1 +0.160pt，McNemar p=0.7383，事前登録の +1.0pt・p<0.05 をいずれも未達）
であり，さらに非退行① で `legal_recall` の有意退行が 1 件出たため，**`models/domain_classifier.joblib` を
`C`=1.0 の `ff8aad9c...`（Iter89 = Iter90 基準線と同一）へ戻した**．`scripts/train_domain_classifier.py` の
`C=_L2_INVERSE_REGULARIZATION` も revert 済み（再訓練すれば sklearn 既定の `C`=1.0 に戻る）．
`C`=10.0 の artifact は `models/domain_classifier_iter91_c10.joblib` に保存してあるので再採用は可逆である．
**次に実験を行う前に `mise run deploy` を必ず実行すること**（wafl500〜509 上には Iter91 本走時点の
`C`=10.0 artifact が載ったままであり，リポジトリ側の現行 artifact と一致していない）．

| ファイル | sha256（ロールバック後の現行） |
|---|---|
| `models/domain_classifier.joblib`（現行＝`C`=1.0，`n_features_in_`=5120） | `ff8aad9cf824992f0a99d07d5506ea9ebdbc3491914a165959c3496df7f2cfd6` |

**2026-09-28 更新（Iteration 92, `classifier_train_label_map_consistency=japanese_civics_realignment_to_eval_map`）**:
コードは無変更．訓練データのみ新規作成: `data/classifier_train_iter92_civics_aligned.jsonl`
（`data/classifier_train_iter87_hybrid.jsonl` 2,327 行から，history_culture ラベルの japanese_civics
訓練行 14 行を education へ付け替え，評価集合と重複する japanese_civics 8 行を削除し，訓練・評価の
どちらにも未使用の japanese_civics 20 問を education 行として追加．合計 2,327→2,339 行，
education 250→284，history_culture 250→228）．追加 20 行の設問整形は `build_dataset.py` の
`_parse_jmmlu_task_csv()` / `_format_jmmlu_query()` をそのまま import して生成した（id 採番:
`education-civics-001`〜`020`，既存 `education-train-*` / `education-hardneg-*` と衝突しない新規プレフィックス）．

再訓練は wafl-ctrl5 の Ollama（127.0.0.1:11499 トンネル経由，wafl500〜509 は不使用）で行い，
**変更のない 2,319 行は `data/embcache_train_iter89_qwen3-embedding_4b{,__p1}.npy` の該当行をそのまま
再利用（インデックス参照でビット一致，スクリプト内でアサーション検証済み）し，新規に埋め込んだのは
追加 20 行のみ**（`data/embcache_train_iter92_qwen3-embedding_4b_new20{,__p1}.npy` に保存）．
再訓練スクリプトは `/tmp/iter92/retrain.py`（リポジトリ外，`scripts/train_domain_classifier.py` の
`_load_training_rows` / `_extract_sample_weights` / `train_classifier` をそのまま import して使用．
モデル定義・`C`=1.0（既定）・`CalibratedClassifierCV(method="temperature", cv=5, ensemble=True)` は無変更）．

旧版（Iter89/90 基準線と同一，`C`=1.0）は `models/domain_classifier_pre_iter92_civics.joblib` へ退避．

| ファイル | sha256 |
|---|---|
| `data/classifier_train_iter92_civics_aligned.jsonl`（2,339 行．新規） | `39c4ca5252011372ed89375981fa2b1ce3c66150106d9fe4048764976599e3ca` |
| `models/domain_classifier.joblib`（現行＝Iter92 候補，`n_features_in_`=5120） | `98da6f2d0d446cca4c67ec9b6cc908881886dad5983ee007e4eedc2876dc9aa3` |
| `models/domain_classifier_pre_iter92_civics.joblib`（Iter89/90/91 ロールバック後と同一，`C`=1.0） | `ff8aad9cf824992f0a99d07d5506ea9ebdbc3491914a165959c3496df7f2cfd6` |

G0-a〜G0-d の実測（構成検証・特徴量同一性・変更の最小性・オフライン replay）は本イテレーションの
実装・実験フェーズ報告（次の journal 追記，分析フェーズ担当）を参照．**G0-e（デプロイ）は本走前に
実施済み**: `mise run deploy` を実行し，wafl500〜509 全 10 台で `models/domain_classifier.joblib` の
sha256 が `98da6f2d...` に一致すること（`ssh <host> sha256sum` で個別確認）と smoke_check PASS を確認した．
本走は `results/20260928_111644/`（3,750 問）．

**2026-09-28 更新（Iteration 94, `train_eval_question_overlap_removal=remove_duplicate_question_rows_and_rebaseline`）**:
コードは無変更．訓練データのみ新規作成: `data/classifier_train_iter94_dedup.jsonl`
（`data/classifier_train_iter92_civics_aligned.jsonl` 2,339 行から，評価集合 `data/dataset.jsonl`
と `query` が完全一致する 64 行を削除．行の並び・残り 2,275 行の内容は完全一致，2,339→2,275 行．
education 284→238，history_culture 228→210，他 8 ドメインは不変）．削除対象 id は
`/tmp/iter94/iter94_dup_train_ids.json`（リポジトリ外，計画フェーズの重複同定スクリプト
`/tmp/iter94/iter94_overlap.py` が出力，完全一致・正規化一致とも同一の 64 行）．

再訓練は特徴量の新規計算 0 件で行った: 変更のない 2,275 行は Iter89 埋め込みキャッシュ
`data/embcache_train_iter89_qwen3-embedding_4b{,__p1}.npy` と Iter92 新規 20 行キャッシュ
`data/embcache_train_iter92_qwen3-embedding_4b_new20{,__p1}.npy` から id 対応で再構成した
（Ollama 呼び出し・wafl500〜509 とも不使用）．再訓練スクリプトは `/tmp/iter94/retrain.py`
（リポジトリ外，`scripts/train_domain_classifier.py` の `_load_training_rows` /
`_extract_sample_weights` / `train_classifier` をそのまま import して使用．モデル定義・
`C`=1.0（既定）・`CalibratedClassifierCV(method="temperature", cv=5, ensemble=True)` は無変更）．
出力 artifact は，計画フェーズの事前 replay `/tmp/iter94/predict_replay.py` が生成した
`/tmp/iter94/domain_classifier_dedup_preview.joblib` と sha256 が完全一致した
（`2f801357...`，同一の特徴量・同一の学習規則であることの独立確認）．

旧版（Iter92 基準線と同一，`C`=1.0）は `models/domain_classifier_pre_iter94_dedup.joblib` へ退避．

| ファイル | sha256 |
|---|---|
| `data/classifier_train_iter94_dedup.jsonl`（2,275 行．新規） | `d6b237358aa5999e1c3101923ab07c805329001655883875bfcbe5aca5f1a6ec` |
| `models/domain_classifier.joblib`（現行＝Iter94 候補，`n_features_in_`=5120） | `2f8013578b7352141aca7c6a830c8cbde6519a7978cde9b6937b335648c1c697` |
| `models/domain_classifier_pre_iter94_dedup.joblib`（Iter92 と同一） | `98da6f2d0d446cca4c67ec9b6cc908881886dad5983ee007e4eedc2876dc9aa3` |

C1（重複 0 行．完全一致・正規化一致とも 0）・C2（差分が 64 id の削除のみ．行の並び・内容は他 1 文字も
不変）は実装フェーズで実測 PASS．G0-d replay（`/tmp/iter94/replay_actual_vs_pre94.py`）は計画フェーズの
事前予測（Δtop1=-0.133pt）を discordant 34/29・p=0.6143 まで完全再現した（C4 PASS）．本走・デプロイ確認・
C3/C5/C6 の集計は次の journal 追記（実装・実験フェーズ報告）を参照．
