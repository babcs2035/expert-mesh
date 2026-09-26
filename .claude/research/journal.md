## Iteration 81: 埋め込み入力への instruction prefix 付与（qwen3-embedding:0.6b）

### 調査 (Iter81)

本反復のレバーは B127（Iter80 分析フェーズ）で `embedding_input_instruction_prefix` = `qwen3_instruct_classification_prefix` に確定済みで，レバー選定の裁量は無い．調査の問いは 3 つ．**(Q1) Qwen3-Embedding の instruction 形式は具体的にどう書くのが正しく，どの程度の効果が報告されているか．(Q2) prefix は「クエリ側だけ」か「訓練側・推論側の両方」か（symmetric / asymmetric タスクでの扱いの違い）．(Q3) Ollama の `/api/embeddings` は prefix を自動付与するのか．**

**Q1: 形式は `Instruct: {task_description}\nQuery: {text}`．効果は「多くの下流タスクで 1〜5%」**

- Qwen 公式（<https://github.com/QwenLM/Qwen3-Embedding>，HF Model Card <https://huggingface.co/Qwen/Qwen3-Embedding-0.6B> / <https://huggingface.co/Qwen/Qwen3-Embedding-8B>，いずれも 2026-09-27 確認）の `get_detailed_instruct()` は `f'Instruct: {task_description}\nQuery:{query}'` を返す．**モデル同梱の `config_sentence_transformers.json` も `query` prompt = `"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "`，`document` prompt = `""`（空）**である（ollama/ollama issue #16076 <https://github.com/ollama/ollama/issues/16076> に原文が引用されている）．
- 効果の公称値: 「**評価の結果，多くの下流タスクで instruct を使うと使わない場合に比べ通常 1〜5% の改善が得られる**．したがってタスク・シナリオに合わせた instruction を作ることを推奨する．**多言語の文脈では，訓練時の instruction の大半が英語で書かれていたため，instruction は英語で書くことを推奨する**」（Qwen3-Embedding GitHub README，同文が HF Model Card・Docker Hub の `ai/qwen3-embedding` にも掲載）．**本研究の質問文は日本語だが，instruction は英語で書くのが公式推奨である**点は設計に直接効く（事実）．
- ただし「1〜5%」の但し書きは「**retrieval シナリオで query 側に instruct を付けない場合に約 1〜5% 低下する**」という retrieval 中心の文言でもある（同 README の Tip）．**分類タスクでの効果量を保証する一次情報は見つからなかった**（推測と事実の区別: 分類での効果は未確認）．
- 日本語での傍証: hotchpotch 氏の JMTEB 計測（<https://secon.dev/entry/2025/06/11/100000-qwen3-embedding-jmteb>，<https://hotchpotch.dev/articles/qwen3-embedding-jmteb>）は Qwen3-Embedding-0.6B の Classification 66.09 を報告しているが，本文に「**Retrieval, Reranking タスクでは Query の prefix に `Instruct: ...\nQuery:` を追加している**」と明記されており，**Classification は prefix なしで測られている**．つまり「日本語分類で prefix を付けた値」は公開情報として存在しない（事実）．Iter79/80 の学び（公称ベンチは採否の根拠にならない）どおり，G1 CV で自前に数値化するほかない．

**Q2: 分類のような single-side（symmetric）タスクでは，全テキストに同じ instruction を付ける．訓練側と推論側で一致していなければならない**

- 公式コード例は **retrieval 前提の asymmetric な使い方**であり，`# No need to add instruction for retrieval documents` と明記して documents 側には何も付けない（上記 GitHub / Model Card）．
- 一方，**分類・クラスタリング・STS のように「クエリ／文書」の区別が無いタスクでは，全入力テキストに instruction を付けるのが標準的な運用**である．arXiv:2606.01074（"When Is 0.1% Enough?"，<https://arxiv.org/html/2606.01074v1> Appendix A，Qwen3-Embedding-8B を含む instruction 系 4 モデルで MTEB 4 タスク族を評価）は「instruction 系モデルでは全タスクで task-specific instruction を付けて符号化する．**classification / clustering / STS では `Instruct: {instruction}\nInput: {text}` を使い，retrieval ではクエリのみ `Instruct: {instruction}\nQuery: {text}` とし，corpus 文書は instruction なしで符号化する（標準的な asymmetric retrieval 設定に従う）**」と手続きを明記している．**`Query:` ではなく `Input:` を使う流儀がある**点は本反復の文言候補に反映する．
- mteb ライブラリでも各タスクの `prompt`（未指定時は抽象クラスの既定文，例: Clustering は `"Identify categories in user passages."`）が入力テキストへ前置される（<https://github.com/embeddings-benchmark/mteb/discussions/3239>，メンテナ回答）．
- **本研究への帰結**: 分類器の訓練特徴量（`data/classifier_train.jsonl` の質問文）と実行時のクエリ文は**同じ種類のテキスト（1 本の日本語質問）**であり，片側だけに prefix を付けると**訓練時と推論時で入力分布がずれる**．Iter36 の失敗（train/eval のタスク不一致で education_recall 0.4588 → 0.0529）と同型の事故になるため，**`scripts/train_domain_classifier.py`（訓練）と `node.py`（実行時クエリ）の双方に同一 prefix を適用することが必須条件**である．一方，`http_server.py:402-405` が計算する `domain_embedding`（ドメイン名そのものの埋め込み）は「文書側」に相当し，かつ `routing_method=supervised_classifier` の現構成では `estimate_embedding_confidence` 経路に到達しない．**ここには prefix を付けない**（asymmetric 慣行に従い，かつ無変更部分を増やさない）．
- **リスク（推測）**: 全入力に同一の prefix を付ける運用は「全ベクトルに共通の変形」を加えるため，線形分類器の性能に与える影響が小さい（discordant が伸びない）可能性がある．一方で Qwen3 は last-token pooling かつ attention 経由なので単なる平行移動ではなく，効果が出るとすれば表現の再配置による．**効果ゼロの可能性を織り込み，G2 の n_d が小さい場合の解釈規則を事前登録する**（後述）．

**Q3: Ollama は prefix を自動付与しない．クライアント側で付ける必要がある**

- ollama/ollama issue #16076（2025〜2026）は「`/api/embed` が `task: "query" | "document"` を受け取り，対応する prefix を Ollama 側で自動前置すべきだ．**今日はこれをクライアント側でやるしかなく壊れやすい**」と現状を述べている（＝現行の Ollama は自動付与しない）．
- 第三者ベンチ（<https://localaimaster.com/blog/best-ollama-embedding-models>）も「**Ollama の `/api/embed` は prefix を一切付けないので自分のコードで前置する必要がある**」と明記．同記事は embeddinggemma では prefix の有無が hit@1 で 33.3% → 87.5% と決定的だった一方，**`qwen3-embedding` では prefix の有無で結果がほぼ同一（1 クエリ差以内）だった**とも報告している（小規模な独自 retrieval 評価，n が小さく一次情報としては弱い）．**本反復の期待値を過大に見積もらない根拠として扱う**．
- 本リポジトリの実装は `expert_backend.py:140-165` の `OllamaClient.embed()` が **旧 `/api/embeddings`（`prompt` フィールド）**を叩いており，`node.py:202` / `scripts/train_domain_classifier.py:140` / `http_server.py:403-405` のいずれも**生の質問文をそのまま渡している**（prefix なし）．学習時分布とのずれは実在する（コードで確認済み）．

**変更対象コードの特定（Read で確認済み．行番号付き）**

| # | ファイル:行 | 現状 | 本反復での扱い |
|---|---|---|---|
| 1 | `config.yaml:4` | `embedding_model: qwen3-embedding:0.6b`（次行から `confidence_threshold`） | 直後に新キー `embedding_instruction:`（英語 1 文．未設定／null で現行動作）を追加 |
| 2 | `expert_backend.py:140-165` | `async def embed(self, model, text, timeout_s)` が `POST /api/embeddings {"model":..,"prompt": text}`（L150-153） | 省略可能引数 `instruction: str | None = None` を追加し，非 None のとき `prompt` を `f"Instruct: {instruction}\nQuery: {text}"` に置換．**既定 None で全既存呼び出しの動作は不変** |
| 3 | `node.py:202` | `await ollama_client.embed(config["embedding_model"], query)` | `instruction=config.get("embedding_instruction")` を渡す（**実行時クエリ側．本レバーの本体**） |
| 4 | `scripts/train_domain_classifier.py:99-142`（実 embed は L140） | `build_training_features()` が `row["query"]` を素で embed | 引数 `instruction: str | None` を追加し L140 へ渡す．CLI に `--embedding-instruction`（`argparse` 定義は L229 付近，`main()` の受け渡しは L197-206・L244-245）を追加（**訓練側．Q2 より必須**） |
| 5 | `scripts/screen_embedding_models.py:64-66, 73-82, 86-104` | `_CANDIDATES`／`_cache_path()`／`_embed_candidate()` がモデル名のみでキャッシュ名を決める | G1 用に「モデル固定・prefix 文言を変える」比較へ書き換える．**`_cache_path()` に prefix 識別子を必ず含める**（`embcache_qwen3-embedding_0.6b.npy` は prefix なしの Iter79 キャッシュであり，識別子を足さないと prefix 版が無言で旧キャッシュを読み，本リポジトリで 6 回起きた「レバー未到達」事故を再現する） |
| 6 | `tools/smoke_check.py:165-170` | `config["embedding_model"]` で `SMOKE_QUERY` を素で embed し分類器へ通す | 同じ `embedding_instruction` を使うよう修正（しないと deploy 後の smoke_check が訓練時と違う分布で確率を見ることになる） |
| 7 | `http_server.py:402-405` | `state.domain_embedding = await ollama_client.embed(state.embedding_model, state.domain)` | **変更しない**（文書側相当・現構成では未到達．Q2 の結論） |
| 8 | `scripts/evaluate_classifier_calibration.py:395,471,612` / `scripts/evaluate_dispatch_candidate_ranking.py:169` / `scripts/fit_embedding_whitening.py:62` / `scripts/run_central_experiment.py:238` | 素の質問文を embed | **本反復では変更しない**（現行の実験経路に含まれない）．ただし今後これらを prefix 前提の artifact に対して使うと無言で分布がずれるため，`data/MANIFEST.md` に注意を明記する |
| 9 | `models/domain_classifier.joblib` | Iter79 版（sha256 `21e16ec6...`，`n_features_in_`=1024） | prefix 付き埋め込みで再訓練して差し替え．旧版を `models/domain_classifier_pre_iter81_noprefix.joblib` へ `cp` 退避 |

**次元は 1024 のまま変わらない点に注意**．Iter80 は次元不一致で 500 エラーが出るため「prefix/モデルの取り違え」が必ず表面化したが，**本反復は訓練側と推論側で prefix が食い違っても例外は起きず，静かに精度だけ落ちる**．この非対称性が本反復最大の実験運用上のリスクであり，後述の F1〜F3 で明示的に潰す．

### 計画 (Iter81)

**単一レバー**

`embedding_input_instruction_prefix` = **`qwen3_instruct_classification_prefix`**．`/api/embed`（実体は `/api/embeddings`）へ渡す文字列に instruction prefix を付けること**だけ**を変える．埋め込みモデルは `qwen3-embedding:0.6b` のまま，分類器のハイパラ・訓練データ・評価集合・ルーティング設定は一切変えない（prefix 変更に構造的に付随する分類器の再訓練は，Iter79/80 で確立したとおり別レバーとは数えない）．

**固定する構成（直近の最良構成 = Iter79 基準線）**

`config.yaml` の `embedding_model=qwen3-embedding:0.6b`・`routing_method=supervised_classifier`・`confidence_threshold=0.0`・`dispatch_candidate_threshold=0.0`・`dispatch_top_k=2`・`dispatch_gap_threshold=0.29`・`dispatch_gap_max_k=4`・`aggregation_method`・`judge_model`・`classifier_model_path`・各ノードの `light_model=qwen3.5:4b-q4_K_M`／`expert_model=expert-mesh-*-lora`・`probe_timeout_s`／`dispatch_timeout_s`，`data/dataset.jsonl`（1,915 行，ビット単位で不変），`data/classifier_train.jsonl`（1,427 行，不変．train/eval 重複 72 行は B125(c) のとおり本反復でも触らない），`scripts/train_domain_classifier.py` のモデル定義部（`LogisticRegression(max_iter=1000, class_weight=None)` + `CalibratedClassifierCV(method='temperature')` + `_extract_sample_weights()`），`classifier.py`・`aggregator.py`・`metrics.py`・`build_dataset.py`，`docker-compose.yml`．**ドメイン固有の文言を含む prefix は 2026-09-23 恒久運用ルールに抵触するため不可．task_description は 10 ドメイン共通の英語 1 文に限る**（B127 要レビュー (1)）．

**事前ゲート G1（文言の確定．評価集合を一切見ない）**

`data/classifier_train.jsonl`（1,427 行）**のみ**で 5-fold StratifiedKFold（`random_state=42`，`LogisticRegression(max_iter=1000, class_weight=None)` + `sample_weight`）の accuracy / macro-F1 を測る．`data/dataset.jsonl` は参照しない．全て wafl-ctrl5（`ssh -fNT -L 11499:localhost:11434 wafl-ctrl5`）で行い，wafl500〜509 は使わない（絶対条件 B）．**比較する文言は結果を見る前に次の 4 条件へ固定する**（B127 要レビュー (1) の「2〜3 文言を G1 でオフライン比較し 1 つへ確定，本走は 1 回」に従う）．

| id | `/api/embeddings` へ渡す文字列 | 由来 |
|---|---|---|
| **P0** | `{text}`（prefix なし＝現行基準線） | Iter79 構成．`data/embcache_qwen3-embedding_0.6b.npy` を再利用（再計算不要） |
| **P1** | `Instruct: Given a user question, identify the single academic or professional domain it belongs to\nQuery: {text}` | Qwen 公式 `get_detailed_instruct()` の形式 ＋ 本タスク向けの英語 1 文（公式は「タスクに合わせて英語で書く」ことを推奨） |
| **P2** | `Instruct: Given a user question, identify the single academic or professional domain it belongs to\nInput: {text}` | arXiv:2606.01074 Appendix A が classification/clustering/STS に用いる `Input:` 形式．P1 との差は末尾ラベルのみ |
| **P3** | `Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: {text}` | モデル同梱 `config_sentence_transformers.json` の既定 query prompt（文言を自作しない対照） |

- **選定規則（事前登録）**: P1/P2/P3 のうち **CV accuracy が最大のもの**を採用する．同点なら macro-F1，なお同点なら P1 > P2 > P3 の順．**P0（Iter79 実測 cv_accuracy 0.7561 / macro-F1 0.7562．同一手順で再計算して一致を確認する）を 3 つとも下回る場合でも，最良の非 P0 候補で本走は実施する**（config.yml 絶対条件 A．Iter79/80 と同じ規則）．その場合は「改善しない見込み」という着地点予測を本走前に journal へ記録する．
- **CV の位置づけ**: G1 CV は**値の選定にのみ用い，本走 top1 の予測値としては扱わない**（Iter79/80 で順位の予測には 2 回連続で成功しているが，方針は据え置く）．
- **キャッシュ命名**: `data/embcache_qwen3-embedding_0.6b__{p1|p2|p3}.npy` のように prefix 識別子を必ず含める（上表 #5）．

**事前ゲート F（レバー発火の直接証拠．本反復では G2 の n_d と分離する）**

本反復は次元が変わらないため，prefix の付け忘れ・食い違いが例外にならない．そこで**発火の証拠を n_d とは独立に 3 つ取る**．

- **F1（埋め込みレベル）**: wafl-ctrl5 で同一の 1 文を prefix 有り／無しで embed し，**コサイン類似度 < 0.999** を確認する（Ollama が prefix を無視していないことの証明．Q3 の裏取り）．満たさなければ以降に進まない．
- **F2（配布レベル）**: deploy 後，全 10 ノードで `grep '^embedding_instruction:' $REMOTE_DIR/config.yaml` が選定文言と一致し，artifact sha256 が新版と一致すること．
- **F3（実行時経路レベル）**: 先頭 20 問の予備実行の `selected_domain` を，**同じ 20 問をオフライン replay（新 artifact × prefix 付き埋め込み）した argmax と 20/20 一致**することを確認する（埋め込みは決定論的なので完全一致を要求する．dispatch 失敗行は除外して分母を明示）．**不一致があれば実行時側の prefix 未適用を疑い，本走に進まない．**

**事前ゲート G2（検出力）**

旧 artifact（prefix なし）と新 artifact（選定 prefix）の `predict_proba` argmax を `data/dataset.jsonl` 1,915 行で replay し，discordant 行数 n_d と必要偏り率 `1.96/sqrt(n_d)` を算出する．**n_d ≥ 30 を合格条件**とする．**解釈規則（事前登録）**: Iter79/80 では n_d 一桁を「config 未到達」の既定解釈としたが，**本反復に限り F1〜F2 が合格しているなら「未到達」ではなく「prefix の効果が検出限界未満」と解釈する**（同一 prefix を全入力へ一律付与する運用では変化が小さくなりうる，という Q2 のリスクに対応）．その場合も絶対条件 A に従い本走は実施し，判定は後述の「no_effect」に落とす．

**変更するファイルと箇所**: 上表 #1〜#6・#9（#7 は変更しない，#8 は MANIFEST への注記のみ）．加えて `data/MANIFEST.md` に新 artifact の sha256・生成コマンド・選定 prefix 文言・G1/G2/F の実測値を追記する（`data/`・`models/` は gitignore 対象で MANIFEST が唯一の再現性の担保．B123）．`.claude/research/config.yml` の `levers` は既に本レバーを含むため追記不要．

**レバーを読むコード行と到達条件**

- 設定 → 各ノードの config: `mise.toml` L67 の `rsync config.yaml`．到達確認は **F2**．
- 設定 → 実行時のクエリ埋め込み: `node.py:202` → `expert_backend.py:140-153`．到達確認は **F3**（次元不一致では落ちないため，予備 20 問の一致で代替する）．
- 設定 → 訓練特徴量: `scripts/train_domain_classifier.py:140`．到達確認は artifact の sha256 が旧版と異なり，かつ G2 の n_d ≥ 30．
- 埋め込みモデルの pull・`models/` rsync は Iter79/80 と同一（`mise.toml` L70-74・L96-104）．**新規に pull するモデルは無い．**
- 実験 → 指標: `metrics.py` 無変更．到達確認は `question_count == 1915` かつ `compound_domain_question_count == 415`．

**実験手順**

1. **前提確認**: `wc -l data/dataset.jsonl` = 1915，`wc -l data/classifier_train.jsonl` = 1427，`models/domain_classifier.joblib` の sha256 が `21e16ec6...`・`n_features_in_`=1024．
2. **F1**（wafl-ctrl5 のみ）．
3. **G1**（wafl-ctrl5 のみ）: P0〜P3 の CV を 1 回の実行で全て記録し，選定規則で 1 文言へ確定する．
4. **再訓練**（wafl-ctrl5 のみ）: 旧 artifact を `models/domain_classifier_pre_iter81_noprefix.joblib` へ退避したうえで `scripts.train_domain_classifier --train-data data/classifier_train.jsonl --embedding-model qwen3-embedding:0.6b --embedding-instruction '<選定文言>' --ollama-host 127.0.0.1 --ollama-port 11499 --output models/domain_classifier.joblib`．`n_features_in_`=1024 と sha256 を記録．
5. **G2**（wafl-ctrl5 のみ）: 1,915 行の argmax replay で n_d と `1.96/sqrt(n_d)`，新 artifact 単体のオフライン accuracy（着地点予測．判定には使わない）を記録．
6. `config.yaml` に `embedding_instruction` を追記し，`uv run ruff check`（既存 23 件は pre-existing）と `uv run pytest tests/`（pre-existing 9 件 FAIL は B122．**新規失敗 0 件**）を確認．
7. `mise run setup`（**直後に `uv sync --extra research` で research extra を復旧．B118 の落とし穴 2**）→ `wc -l data/dataset.jsonl` = 1915 を再確認．
8. `mise run deploy` → **F2** と smoke_check の pass を確認．
9. **先頭 20 問の予備実行 → F3**．不一致なら本走に進まず原因を潰す．
10. **wafl500〜509 で 1,915 問のフルスペック本走を 1 回**（絶対条件 A）．起動直後に `state.json` を `status=waiting_experiment`・`experiment_dir`・`experiment_deadline`（開始時刻 + 150×60 + 600 秒）へ更新．`mise run analyze -- <timestamp>` まで実施（**引数なし実行は `results/iter45_preliminary/` を誤選択する．B118 の落とし穴 1**）．
11. 指標を「全 1,915 行」「既存 1,600 行部分集合」「複合 415 行」の 3 通りで算出し，Iter79 基準線 `results/20260926_221822/` と id ペアリングで McNemar 検定・per-domain 20 指標の BH 補正を行う．

**仮説（事前登録）**

「Qwen3-Embedding は instruction-aware に訓練されており，現行実装は prefix を一切付けずに学習時分布とずれた入力を与えている．訓練側と推論側の双方へ同一の英語 1 文 instruction を付けると，10 ドメインの線形分離度が上がり，1,915 行本走の `top1_accuracy` が Iter79 基準線 0.753003 から **+1.0pt 以上**改善する（McNemar p<0.05）．ただし公称の 1〜5% は retrieval 中心の値であり，分類での効果量は一次情報が無い．さらに**全入力へ一律に同じ prefix を付ける運用は変化が小さくなりうる**ため，**着地点は 0〜+2pt の範囲，discordant n_d は 50〜300 行**と予測する．」

**成功条件（事前登録．結果を見る前に固定する）**

基準線は Iter79 本走 `results/20260926_221822/`（全 1,915 行: top1=0.753003，single_domain_top1=0.749333，compound_domain_top1=0.766265，compound_domain_set_recall=0.548193，kappa=0.721502，misrouting=0.246997，ECE=0.032745，Brier=0.152849，AUROC=0.777614，fallback=0.0，dispatch_failure=0.000522，mean_duration_ms=2301.4，compound_mean_dispatched_count=1.880，answer_quality=0.569333，end_to_end=0.335770／既存 1,600 行部分集合: top1=0.751250）．

| 区分 | 指標 | 現状 | 合格条件 |
|---|---|---|---|
| **F1** | prefix 有／無の埋め込みのコサイン類似度 | — | **< 0.999**（Ollama が prefix を反映していること） |
| **F2** | 全 10 ノードの `embedding_instruction` と artifact sha256 | — | 全ノードで選定文言・新 sha256 に一致 |
| **F3** | 予備 20 問の `selected_domain` とオフライン replay の一致 | — | **20/20 一致**（dispatch 失敗行は除外し分母を明記） |
| **G1** | 5-fold CV accuracy（`classifier_train.jsonl` のみ） | P0 = 0.7561 | P0〜P3 の 4 値を全て記録し，選定規則どおり 1 文言へ確定できること |
| **G2** | 旧／新 replay の discordant n_d | 参考: Iter80 は 469 | **n_d ≥ 30**．未満でも F1・F2 合格なら「検出限界未満」と解釈し本走は実施 |
| **主基準（効果）** | 全 1,915 行の `top1_accuracy` | 0.753003 | **McNemar p < 0.05 かつ 点推定 +1.0pt 以上**（n_d=100 想定の MDE `1.96·sqrt(n_d)/1915` ≈ 1.02pt を上回る水準） |
| **非退行①** | per-domain recall/precision 計 20 指標 | Iter79 実測 | **BH 補正（q=0.05）後の有意退行 0 件** |
| **非退行②** | `fallback_rate` / `dispatch_failure_rate` | 0.0 / 0.000522 | fallback = 0.0，dispatch_failure ≤ 0.005 |
| **非退行③** | ECE | 0.032745 | **≤ 0.08** |
| **非退行④** | `mean_duration_ms` | 2301.4 | **≤ 2761（+20% 以内）**．prefix は入力トークンを 15〜20 トークン増やすだけなので，Iter80 の「2 倍」枠は不要 |
| 報告のみ | 既存 1,600 行部分集合の top1 | 0.751250 | 判定には用いないが毎回併記 |
| 報告のみ | `compound_domain_top1` / `single_domain_top1` / `compound_domain_set_recall` / `compound_mean_dispatched_count` | 0.766265 / 0.749333 / 0.548193 / 1.880 | 複合 415 行は埋め込み品質の変化に対する感度が単一行より高い（Iter80 の学び 4） |
| 報告のみ | `answer_quality_accuracy` / `end_to_end_accuracy` | 0.569333 / 0.335770 | 生成器は不変のため横ばいを期待（3SD=2.6pt のノイズ床を適用） |

**判定規則（事前登録）**

- **adopted**: 主基準（McNemar p<0.05 かつ +1.0pt 以上）と非退行①②③④をすべて満たす．
- **partial**: 点推定は改善だが p ≥ 0.05 または +1.0pt 未満で，非退行に違反なし．この場合 prefix は**採用せず基準線へ復元**し，レバーは「収束」扱いとする（文言をさらに探すチューニングは B127 要レビュー (1) により行わない）．
- **no_effect**: G2 の n_d < 30（ただし F1・F2 合格）かつ本走の top1 差が ±0.5pt 以内．**「実験不成立」ではなく「prefix は本構成で効かない」という結論**として記録し，基準線へ復元する．
- **rejected**: 点推定が低下，または非退行①で有意退行 1 件以上，または非退行②③④のいずれかに違反．
- **invalid（実験不成立）**: F1・F2・F3 のいずれか不合格，`question_count != 1915`，`compound_domain_question_count != 415`．setup/deploy・実装漏れと解釈する（d0004 §4）．
- **復元手順（partial / no_effect / rejected 共通）**: `config.yaml` の `embedding_instruction` 行を削除し，`cp models/domain_classifier_pre_iter81_noprefix.joblib models/domain_classifier.joblib`（sha256 `21e16ec6...`，`n_features_in_`=1024）で復元，`mise run deploy` を再実行して全 10 ノードで sha256 と config を確認する．

**期待効果とリスク**

期待効果は「モデルを替えずに，学習時分布へ入力形式を合わせるだけで分離度を取り戻す」ことに尽きる．VRAM を増やさないため Iter80 の G0 制約に抵触しない．リスクは 2 つで，(1) **効果がゼロに近い可能性**（公称 1〜5% は retrieval 側の値であり，第三者ベンチでも qwen3-embedding は prefix 有無でほぼ同一だったと報告されている），(2) **訓練側と推論側で prefix が食い違っても例外が出ず静かに劣化する**こと．(1) は G1・G2 で本走前に数値化し，(2) は F1〜F3 で潰す．

### 実験 (Iter81)

事前登録した計画どおりに実装・実験を実施した．判定は行っていない（分析・考察フェーズの担当）．

**変更したファイル**: `expert_backend.py`（`OllamaClient.embed()` に `instruction: str | None = None` を追加し，非 None のとき `prompt = f"Instruct: {instruction}\nQuery: {text}"`．既定 None で既存呼び出しは不変），`node.py:202`（`instruction=config.get("embedding_instruction")` を渡す），`scripts/train_domain_classifier.py`（`build_training_features()` / `_train_and_save()` に `instruction` 引数，CLI `--embedding-instruction`），`scripts/screen_embedding_models.py`（prefix 文言比較へ全面書き換え．`_cache_path()` に `__p1`/`__p2`/`__p3` の識別子を含める），`tools/smoke_check.py:169-171`，`config.yaml`（新キー `embedding_instruction`），`data/MANIFEST.md`（Iter81 節）．`http_server.py:402-405` は計画どおり変更していない．`models/domain_classifier.joblib` は prefix 付きで再訓練（sha256 `56d5a882ec8e...`），旧版を `models/domain_classifier_pre_iter81_noprefix.joblib`（sha256 `21e16ec6...`）へ退避した．

**lint / テスト**: `uv run ruff check` 23 件（実装前と同数．pre-existing），`uv run pytest tests/` 9 failed / 306 passed（`test_build_dataset.py` の 9 件は B122 の既知失敗で実装前後同一）．**新規失敗 0 件**．

**ゲート実測（すべて PASS）**

| ゲート | 実測 | 判定 |
|---|---|---|
| F1（prefix 有無のコサイン） | 0.7674 < 0.999 | PASS |
| G1（`classifier_train.jsonl` 1,427 行のみ 5-fold CV） | P0=0.756123 / **P1=0.771523（最良，採用）** / P2=0.766610 / P3=0.754720 | PASS |
| G2（1,915 行 argmax replay の discordant） | n_d=276 ≥ 30 | PASS |
| F2（deploy 後 全 10 ノード） | `embedding_instruction` 文言・artifact sha256 `56d5a882ec8e...` 全ノード一致 | PASS |
| F3（先頭 20 問 予備実行 vs replay） | `selected_domain` 20/20 一致，dispatch 失敗 0 件 | PASS |

採用した instruction 文言（P1）は `Given a user question, identify the single academic or professional domain it belongs to`．

**本走**: `results/20260927_024644/`．wafl500〜509 で 1,915 問フルスペックを 1 回．`mise run analyze -- 20260927_024644` 実施済み．基準線は `results/20260926_221822/`（Iter79）．

| 指標 | 基準線 | Iter81 | 差 |
|---|---|---|---|
| **全 1,915 行 top1_accuracy** | 0.753003 | **0.789556** | **+3.655pt**（McNemar chi2=25.5968, p=4.207e-7, discordant 58/128） |
| 既存 1,600 行部分集合 top1 | 0.751250 | 0.781250 | +3.000pt |
| 複合 415 行 top1 | 0.766265 | 0.843373 | +7.711pt |
| single_domain_top1 | 0.749333 | 0.774667 | +2.533pt |
| compound_domain_set_recall | 0.548193 | 0.526506 | -2.169pt |
| compound_mean_dispatched_count | 1.880 | 1.494 | -0.386 |
| fallback_rate | 0.0 | 0.0 | 0 |
| dispatch_failure_rate | 0.000522 | 0.000522 | 0 |
| ECE | 0.032745 | 0.030951 | -0.001794 |
| mean_duration_ms | 2301.4 | 2200.7 | -100.7 |
| answer_quality_accuracy | 0.569333 | 0.580000 | +1.067pt |
| end_to_end_accuracy | 0.335770 | 0.350392 | +1.462pt |

**per-domain recall/precision 計 20 指標（BH 補正 q=0.05）**: 有意差 3 件．computer_science_recall 0.7273→0.8528（p≈4.93e-7，改善），natural_science_recall 0.5931→0.6623（p=0.004586，改善），**medical_recall 0.7842→0.7178（p=0.003264，悪化）**．他 17 指標は有意差なし．

**申し送り 1（要検証）**: 本反復開始時の実機構成は `embedding_model=qwen3-embedding:0.6b`・artifact sha256 `21e16ec6...` であり，rc-executor は「これが MANIFEST 上 Iter80 で adopted と記録されていた bge-m3 版（sha256 `37d71b63...`）と一致しない」と報告した．**ただし backlog B127 では Iter80 の判定は `rejected` で，基準線（`21e16ec6...`）への復元を実施したと記録されている**ため，MANIFEST 側の記述が judgment と食い違っている可能性が高い．どちらが誤りかは本反復のスコープ外として MANIFEST に注記のみ残してある．分析フェーズで確認すること．

**申し送り 2（分析フェーズで確認済み）**: wafl500 上に F3 検証用の空ディレクトリ `~/workspace/ktakahashi/expert-mesh/results/iter81_preflight`（root 所有）が残存している．空で無害だが削除に sudo が必要なため放置した．

### Iteration 81 実行済み

**変更**: 単一レバー `embedding_input_instruction_prefix` = `qwen3_instruct_classification_prefix`．埋め込みモデル（`qwen3-embedding:0.6b`）・分類器のハイパラ・訓練データ・評価集合・ルーティング設定を固定したまま，`/api/embeddings` へ渡す文字列を `Instruct: Given a user question, identify the single academic or professional domain it belongs to\nQuery: {text}`（G1 で 4 候補から選定した P1）へ変え，訓練側（`scripts/train_domain_classifier.py`）と実行時クエリ側（`node.py:202`）の双方に同一 prefix を適用した．

**結果（ノイズか信号か）**: 全 1,915 行 top1 **0.753003 → 0.789556（+3.655pt）**．ルーティング系は決定論的でノイズ床を適用しない系だが（config.yml success_criteria (5)），それでも McNemar chi2=25.5968・**p=4.207e-7**（discordant 186 行の内訳 58 悪化 / 128 改善）であり，事前登録の MDE（n_d=100 想定で約 1.02pt）を 3.6 倍上回る．**ノイズではなく明確な信号**である．複合 415 行 top1 +7.71pt，single_domain_top1 +2.53pt，ECE -0.0018，`mean_duration_ms` -100.7（prefix によるトークン増は速度に響かなかった），fallback/dispatch_failure 不変．answer_quality +1.07pt・end_to_end +1.46pt は 3SD=2.6pt のノイズ床内で判定しない．ゲート F1/F2/F3・G1・G2 は全 PASS で，実験は成立している（invalid ではない）．

**判定: rejected（事前登録の判定規則どおり）**．規則は「点推定が低下，**または非退行①（per-domain 20 指標の BH 補正後の有意退行 0 件）に 1 件以上抵触**，または非退行②③④に違反」を rejected と定めている．per-domain 20 指標の BH 補正後の有意差は 3 件で，うち **medical_recall 0.7842 → 0.7178（p=0.003264）が悪化方向**であるため，主基準（+1.0pt 以上かつ p<0.05）を大幅に満たしていても規則上 rejected 以外の判定は取れない（partial・no_effect はいずれも「非退行に違反なし」または「n_d<30 かつ ±0.5pt 以内」を要件とし，本結果はどちらにも該当しない）．**結果を見てから規則を読み替えないという事前登録の趣旨に従い，復元を実施した**: `config.yaml` の `embedding_instruction` 行を削除，`cp models/domain_classifier_pre_iter81_noprefix.joblib models/domain_classifier.joblib`（sha256 `21e16ec6...`），`mise run deploy`．全 10 ノードで `config.yaml` のハッシュ一致（smoke_check）・`embedding_instruction` 不在・artifact `21e16ec6...` を確認し，probe も pass．

**学び 1（送出数が減った構造．事前の仮説は支持された）**: `compound_domain_set_recall` が 0.548193 → 0.526506 と下がり `compound_mean_dispatched_count` が 1.880 → 1.494 に減った原因は，**prefix により確信度分布が鋭くなり，`dispatch_gap_threshold=0.29` による 2 件目以降の送出が起きにくくなったこと**である．`results/20260927_024644/` の `probe_candidates` を基準線と id ペアで比較すると，複合 415 行の rank1−rank2 gap は平均 0.4769 → 0.5764（中央値 0.4449 → 0.6075），**gap < 0.29 の行の割合は 33.49% → 20.48%**，rank1 confidence 平均は 0.6510 → 0.7239 だった．送出内訳も「1 件送出で 1 ドメイン的中」が 236 → 296 行へ増え，「4 件送出で 2 ドメイン的中」が 66 → 46 行へ減っている．**送出予算を揃えると逆転する**: 同じ `probe_candidates` に gap 閾値を掃引して再現すると，新構成は gt=0.40 で mean_k=1.933・set_recall **0.5928** となり，基準線の gt=0.29（mean_k=1.880・0.5482）を **+4.46pt 上回る**．つまり set_recall の低下は埋め込み品質の劣化ではなく，**固定閾値 0.29 が新しい確信度スケールに対して相対的にきつくなったことの帰結**である（gap 閾値は確信度分布のスケールに依存するハイパラであり，埋め込みを変えたら再較正が要る，という一般則）．

**学び 2（medical 退行の正体は「rank 1 → rank 2 の入れ替わり」）**: `metrics.py` の per-domain recall は `selected_domain`（= 固定 top-1）で測る定義であり，実測でも新構成の固定 top-1 medical recall は 0.7178 で journal 記載値と一致する．medical を含む 241 行のうち **23 行が medical を取りこぼし，6 行が新たに取れた**．取りこぼした 23 行で新 artifact の medical の順位を見ると **rank 2 が 19 行，rank 3 が 3 行，rank 4 が 1 行**で，**固定 top-2 で測れば medical recall は 0.9046 → 0.8963 とほぼ不変**（top-3 では 0.9461 → 0.9544 と改善）．吸われた先は education 8 行・business_economics 6 行・computer_science 5 行・history_culture 2 行・natural_science 1 行・social_science 1 行で，特定 1 ドメインへの系統的な混同ではない．**medical の信号が失われたのではなく，prefix が「学術/専門ドメインの同定」という instruction に沿って空間を再配置した結果，medical と隣接ドメインの相対順位が僅差で入れ替わった**と解釈するのが妥当である．なお 2026-09-23 恒久運用ルールにより，medical 固有の閾値・intercept・訓練データ調整による是正は**提案しない**（次レバーは 10 ドメイン共通の特徴量設計で対処する）．

**学び 3（記録の一貫性．申し送り 1 への回答）**: **誤っていたのは `data/MANIFEST.md` の方**である．Iteration 80（`bge_m3`）の判定は journal「Iteration 80 実行済み」節・backlog B127 のいずれも **rejected**（top1 -3.03pt，BH 補正後の有意退行 2 件）で，基準線 `21e16ec6...` への復元まで実施済みと記録されている．MANIFEST だけが Iteration 80 の**実装フェーズ時点の記述**（生成コマンドの `--embedding-model bge-m3`，「Iteration 80 で採用されたはずの bge-m3」）のまま更新されず，あたかも adopted であったかのように読めていた．本フェーズで MANIFEST を訂正した（生成コマンドを `qwen3-embedding:0.6b` へ戻し，訂正の経緯を明記）．**教訓: MANIFEST は実装フェーズで先に書かれるため，rejected で復元した反復では分析フェーズが必ず MANIFEST を巻き戻す責任を負う**（journal/backlog は判定を書くが MANIFEST は書かない，という非対称が今回の食い違いを生んだ）．

**学び 4（実験設計への反省．次レバーへ直結）**: 本反復は「per-domain の非退行を本走後にしか確認していない」ために，1 回の 1,915 問本走（約 100 分 + 10 ノード）を rejected で失った．**per-domain 20 指標の BH 補正は分類器 artifact の argmax replay（決定論的・オフライン）で本走前に完全に予測できる**（`metrics.py` の `compute_domain_recall_mcnemar_test` 等をそのまま流用可能）．Iter79/80/81 で G1（CV）→ G2（discordant n_d）までは型化できていたが，**G2 に per-domain 非退行の事前予測を足す（G2'）**のが次反復以降の標準手順である．

**学び 5（要人間判断）**: 「全体 +3.655pt（p=4.2e-7）を，1 ドメインの recall 退行 1 件で棄却する」という非退行条件①の設計自体に，研究方針として再考の余地がある．ただし判定規則の改定は結果を見てからの事後変更であり，かつ研究の結論に関わる不可逆な方針変更のため，**本フェーズでは規則を変えず rejected を確定させ，規則設計の是非は人間判断に委ねる**（backlog B128 要レビュー）．

**次の一手**: 新レバー **`embedding_view_concatenation` = `prefix_and_noprefix_concat`**（`.claude/research/config.yml` の `levers` 末尾へ追記済み）．学び 2 の「prefix 有り／無しは相補的で，medical は prefix 版でも rank 2 に残っている」という実測を根拠に，2 つのビューを連結した 2048 次元を分類器の特徴量とし，学び 4 の G2'（本走前の per-domain 非退行予測）を事前ゲートに組み込む．

---

## Iteration 80: 埋め込みモデルのスケールアップ（qwen3-embedding:0.6b → 4b）

### 調査 (Iter80)

B125(b) により本反復のレバーは `embedding_model_replacement` = `qwen3_embedding_4b`（config.yml の `values` にある未試行の第 2 値）で確定しており，レバー選定の裁量は無い．調査の問いは 3 つ．**(Q1) `qwen3-embedding:4b` は 0.6b に対して本タスク（日本語 10 ドメイン分類）で上積みが見込めるか．(Q2) 10 ノード（RTX 3060 12GB）の VRAM に収まるか．収まらない場合の代替は何か．(Q3) 次元数の増加（1024 → 2560）が 1,427 行の LogisticRegression 訓練に与える影響は何か．**

**Q1: 公称値は一貫して 4b > 0.6b．ただし日本語分類の直接値は無い**

- Ollama 公式ライブラリの `qwen3-embedding:4b` は **2.5GB（Q4_K_M），native 2560 次元，40K context**（<https://ollama.com/library/qwen3-embedding/tags>，2026-09-26 確認）．0.6b は 639MB・1024 次元・32K context．MTEB multilingual は **0.6b 64.33 / 4b 69.45 / 8b 70.58**（morphllm のベンチ一覧 <https://www.morphllm.com/ollama-embedding-models>，および config.yml lever note と同値）．
- Qwen 公式の Model Card（<https://huggingface.co/Qwen/Qwen3-Embedding-4B>，GitHub <https://github.com/QwenLM/Qwen3-Embedding>）では MMTEB 系の平均が **0.6B 70.70 → 4B 74.60 → 8B 75.22**，CMTEB 系の行でも 4B が 0.6B を一貫して上回る．**サイズ方向の単調性は複数のベンチで一致している**（事実）．
- ただし**日本語（JMTEB）の Classification は 4B の公開値が見つからなかった**．hotchpotch の JMTEB 計測（<https://secon.dev/entry/2025/06/11/100000-qwen3-embedding-jmteb>）は 0.6B のみ（Classification 66.09）で，4B 行は無い．**「4b が日本語分類で 0.6b を上回る」は公称値からの外挿であり，未検証の推測である**．Iter79 の学び 2（公称ベンチは候補を絞る道具であって採否の根拠にならない．MTEB 差 +2.05pt に対し実測 +16.34pt だった）をそのまま適用し，本反復も **G1 オフライン CV で本走前に数値化する**．
- 同時比較する `bge-m3` は Ollama 公式・**1024 次元・約 1.2GB・prefix 不要**（同ベンチ一覧）．JMTEB の Classification 値は上記記事に無いが，多言語モデルであり VRAM の退避先として位置づけられる．

**Q2: VRAM は本反復の最大のリスク．実測では余裕が 3.4GB 程度しかない**

wafl500・wafl503・wafl509 で `nvidia-smi` と `ollama ps` を実測した（2026-09-26，read-only）．

| 項目 | 実測 |
|---|---|
| GPU | RTX 3060 **12288 MiB**（全ノード共通） |
| wafl500 現況 | used 7621 MiB / free 4290 MiB．resident は `qwen3-embedding:0.6b` **2.4GB** ＋ `expert-mesh-general-lora` **5.3GB**（light_model は idle で未ロード） |
| wafl503/509 現況 | `expert-mesh-*-lora` 5.3GB ＋ `qwen3.5:4b-q4_K_M` **3.1GB** ＋ `nomic-embed-text` 0.32GB |
| wafl-ctrl5 | used 8131 / free 3779 MiB．judge 用 swallow-8B 5.3GB ＋ 0.6b 2.4GB ＋ nomic 0.32GB が常駐 |

- **重要な実測事実**: `qwen3-embedding:0.6b` の**常駐サイズは重み 639MB ではなく 2.4GB**である（`ollama ps`，context 4096）．差分の約 1.8GB は KV/活性メモリで，Ollama の SIZE はこれを含む．したがって「4b は 0.6b の約 4 倍」という B125 の見立て（重み比）は常駐サイズの比としては過小で，**4b の常駐サイズは重み 2.5GB ＋ より大きい KV（層数・隠れ次元とも増加）で 4.5〜6GB 規模になる可能性が高い**（推測．実測で確定させる）．
- 本走時の wafl500 のピークは `light_model 3.1GB + expert_model 5.3GB + embedding X` である．**12288 MiB から 8.4GB を引くと X の予算は約 3.4GB** しかない．0.6b（2.4GB）はこの予算に収まっているが，**4b は収まらない可能性が高い**．
- 収まらない場合に何が起きるかも実測から分かる．wafl503/509 では `qwen3-embedding:0.6b` が resident に居らず古い `nomic-embed-text` が残っている．**`OLLAMA_KEEP_ALIVE=-1`（docker-compose.yml:23）でも，Ollama は新しいモデルを載せるために既存モデルを退避する**．非依頼者ノードは埋め込みを起動時の `domain_embedding` 計算（`http_server.py:402-404`）にしか使わないため退避されても実害は無いが，**依頼者 wafl500 は 1 問ごとに埋め込みを呼ぶ（`node.py:202`）ため，退避と再ロードが往復すると所要時間が跳ね上がる**．精度ではなく所要時間（`experiment.timeout_min: 150`）のリスクとして扱う．
- wafl-ctrl5 も free 3779 MiB であり，G1 で 4b を載せる前に `ollama stop` で `qwen3-embedding:0.6b`（2.4GB）と `nomic-embed-text` を一時退避して枠を空ける必要がある．**0.6b と nomic の訓練データ埋め込みは `data/embcache_qwen3-embedding_0.6b.npy` / `data/embcache_nomic-embed-text.npy` にキャッシュ済み**（`scripts/screen_embedding_models.py` の仕様）なので，退避しても G1 の基準線は再計算不要である．

**Q3: 2560 次元 × 1,427 行は p ≫ n だが，1024 次元でも既にそうである**

訓練行数 1,427 に対し特徴次元は 1024 → 2560 へ増える．`LogisticRegression(max_iter=1000)`（L2 正則化，既定 C=1.0）は p ≫ n でも解けるが，**訓練データ内 CV と本走の乖離が広がる方向の変化である**．Iter79 は G1 CV 0.7561 に対し本走 0.7530 とほぼ一致したが，次元が 2.5 倍になると「CV は上がったが本走は上がらない」という乖離が起こり得る．**この点は事前登録の解釈規則として明記しておく**（後述）．

### 計画 (Iter80)

**単一レバー**

`embedding_model_replacement` = **`qwen3_embedding_4b`**．`config.yaml:4` の `embedding_model` を `qwen3-embedding:0.6b` → `qwen3-embedding:4b` へ変更し，同じ埋め込みで `models/domain_classifier.joblib` を再訓練する（再訓練は次元変更に構造的に付随する作業であり別レバーではない．Iter79 で確立した型）．**変更する設定キーは `config.yaml:4` の 1 行のみ．**

**固定する構成（単一レバー原則）**

`config.yaml` の `embedding_model` 以外の全項目（`routing_method=supervised_classifier`，`confidence_threshold=0.0`，`dispatch_top_k=2`，`dispatch_gap_threshold=0.29`／`dispatch_gap_max_k=4`，`aggregation_method`，`judge_model`，`classifier_model_path`，各ノードの `light_model=qwen3.5:4b-q4_K_M`／`expert_model=expert-mesh-*-lora`，`probe_timeout_s`／`dispatch_timeout_s`，`experiment.timeout_min=150`），`data/dataset.jsonl`（1,915 行．ビット単位で Iter79 と同一），`data/classifier_train.jsonl`（1,427 行，無変更．**train/eval 重複 72 行は B125(c) のとおり本反復では触らない**），`scripts/train_domain_classifier.py` のハイパラ一式，`classifier.py`／`http_server.py`／`node.py`／`aggregator.py`／`metrics.py`／`build_dataset.py`，`docker-compose.yml`（`OLLAMA_KEEP_ALIVE=-1` を含む）．**instruction prefix は導入しない**（B125(5)．`multilingual-e5-large`／`ruri-v3-310m` は prefix 付与＝2 レバー目になるため本反復の候補外）．

**事前ゲート G0（VRAM 実測．B125(3)．結果を見る前に判定規則を固定する）**

すべて wafl-ctrl5 と read-only の `ollama ps`／`nvidia-smi` で行い，**本走前に wafl500〜509 で生成処理は走らせない**（config.yml 絶対条件 B）．

1. **G0-a（wafl-ctrl5 での常駐サイズ実測）**: `docker exec ollama-ctrl ollama stop qwen3-embedding:0.6b nomic-embed-text` で枠を空け，`ollama pull qwen3-embedding:4b` → 1 文の `/api/embed` を叩いてロードさせ，`ollama ps` の SIZE と PROCESSOR，`nvidia-smi` の used 差分を記録する．**PROCESSOR が `100% GPU` でなければ（CPU 混在なら）その時点で G0 失敗**．
2. **G0-b（wafl500 の予算判定）**: G0-a で得た常駐サイズを X とし，**X + 3.1GB（light_model）+ 5.3GB（expert_model）≤ 11.5GB** を合格条件とする（12288 MiB から表示等の余白 0.8GB を引いた実効上限）．
3. **G0-c（deploy 後の実機確認）**: deploy 後・本走前に，先頭 20 問の予備実行を回したうえで全 10 ノードの `ollama ps` を取り，(i) 3 モデルすべてが `100% GPU`，(ii) 予備 20 問の 1 問あたり平均所要が **Iter79 実測 2301.4ms の 2 倍（4603ms）以下**，を確認する．満たさない場合は所要時間を外挿し，**1,915 問の予測所要が 120 分を超えるなら G0 失敗**とする．
4. **G0 失敗時の代替（事前登録）**: `qwen3-embedding:4b` を断念し，**G1 で VRAM 実行可能と確認できた候補のうち CV accuracy が最大のもの**（現実的には `bge-m3`，重み約 1.2GB）へレバーの値を切り替える．その場合のみ config.yml の `embedding_model_replacement` の `values` へ `bge_m3` を追記し，backlog へ `[auto-decided]` で記録する．**`iteration_name` は経緯の追跡性のため変更しない**（見出しと実際の値がずれる場合は journal の実行記録に明記する）．

**事前ゲート G1（値の選定．評価集合を一切見ない．B125(1)(2)）**

`scripts/screen_embedding_models.py` を拡張し（`_STAGE_1_CANDIDATES` を本反復用に差し替える），`data/classifier_train.jsonl`（1,427 行）**のみ**で 5-fold StratifiedKFold の accuracy / macro-F1 を測る．`data/dataset.jsonl` は参照しない（現行スクリプトが既にその設計であることは docstring と `--train-data` 既定値で担保されている）．

- 候補は 3 つを**同時に**測る: **`qwen3-embedding:0.6b`（現行＝基準線．キャッシュ再利用）**，**`qwen3-embedding:4b`**，**`bge-m3`**（B125(2) の申し送り）．
- **選定規則（結果を見る前に固定）**: G0 を通過した非現行候補のうち **CV accuracy 最大のものをレバーの値とする**．同点なら macro-F1，なお同点なら `qwen3-embedding:4b` を優先する．**現行 0.6b（Iter79 実測 cv_accuracy 0.7561 / macro-F1 0.7562）を下回る候補しか残らない場合でも，最良の非現行候補で本走は実施する**（config.yml 絶対条件 A．Iter79 の規則 4 と同一）．その際は「改善しない見込み」という着地点予測を本走前に journal へ記録する．
- **CV と本走の乖離に関する解釈規則（Q3 への対処）**: 次元が 2.5 倍になるため，G1 の CV は本走 top1 の上振れした推定になり得る．**G1 CV が本走の予測値であるとは扱わず，値の選定にのみ用いる**．

**事前ゲート G2（検出力）**

旧 artifact（0.6b・1024 次元）と新 artifact（選定モデル）の `predict_proba` argmax を 1,915 行で replay し，discordant 行数 n_d と必要偏り率 `1.96/sqrt(n_d)` を算出する．**n_d ≥ 30 を合格条件**とし，一桁なら「効果なし」ではなく **config 未到達**を既定の解釈とする（d0004 §4）．

**変更するファイルと箇所**

1. `config.yaml:4`: `embedding_model: qwen3-embedding:0.6b` → `embedding_model: qwen3-embedding:4b`（G0/G1 の結果が `bge-m3` ならその名前）．**これが本レバーの本体であり，変更はこの 1 行のみ．**
2. `models/domain_classifier.joblib`: 新埋め込みで再訓練して差し替える．**旧版は `models/domain_classifier_pre_iter80_qwen3_0.6b.joblib` へ `cp` で退避**してから上書きする（Iter77/79 と同じ慣行．flip 計測と rejected 時の復元に必要）．
3. `scripts/screen_embedding_models.py`: 候補リストを本反復用（`qwen3-embedding:0.6b` / `qwen3-embedding:4b` / `bge-m3` の 3 候補同時評価）へ更新し，docstring の選定規則を Iter80 のものへ書き換える．**`data/dataset.jsonl` を参照しない性質は維持する．**
4. `data/MANIFEST.md`: 新 artifact の sha256・生成コマンド・埋め込みモデル名・G0/G1/G2 の実測値を追記する（`data/`・`models/` は `.gitignore` 対象のため MANIFEST が唯一の再現性の担保．B123）．
5. `.claude/research/config.yml`: **G0 失敗で `bge-m3` を採った場合のみ** `values` へ `bge_m3` を追記する．

**変更しないが確認だけするファイル**: `tests/test_node.py:170` ほか計 4 つのテストが埋め込みモデル名を文字列リテラルで持つが，いずれもテスト内で組み立てる config 辞書の値であって `config.yaml` を読まない．**テストの修正は不要．**

**レバーを読むコード行と到達条件**

- 設定 → 全ノードへのモデル配布: `tools/node_models.py:get_models()` が `config["embedding_model"]` を返し，`mise.toml` L96-104 の deploy ループが `ollama pull` する．到達確認は**全 10 ノードで `ollama list | grep qwen3-embedding` に `4b` 行があること**．
- 設定 → 各ノードの config: `mise.toml` L67 の `rsync config.yaml`．到達確認は**全 10 ノードで `grep '^embedding_model:' $REMOTE_DIR/config.yaml`**．
- 設定 → 実行時のクエリ埋め込み: `node.py:202`（依頼者）／`http_server.py:402-404`（各ノードの `domain_embedding`）．到達確認は先頭 20 問の予備実行が 500 を返さないこと（**次元不一致なら必ずここで落ちる**）．
- artifact → 全ノード: `mise.toml` L70-74 の `models/` rsync．到達確認は**全 10 ノードで `n_features_in_` が 2560（`bge-m3` なら 1024）**，または sha256 一致．
- 実験 → 指標: `metrics.py` 無変更．到達確認は `question_count == 1915` かつ `compound_domain_question_count == 415`．
- **レバー発火の直接証拠**: G2 の n_d ≥ 30，かつ本走 `selected_domain` の discordant が同程度であること．

**実験手順**

1. **前提確認**: `wc -l data/dataset.jsonl` = 1915，`wc -l data/classifier_train.jsonl` = 1427，`models/domain_classifier.joblib` の sha256 が `data/MANIFEST.md` の記載（`21e16ec6...`）と一致すること．
2. **G0-a（wafl-ctrl5 のみ）**: 上記のとおり 4b と bge-m3 を pull し，常駐サイズ・PROCESSOR・embed レイテンシを実測．**wafl500〜509 は使わない．**
3. **G1（wafl-ctrl5 のみ）**: `ssh -fNT -L 11499:localhost:11434 wafl-ctrl5` のトンネル経由で `uv run python -m scripts.screen_embedding_models --train-data data/classifier_train.jsonl --ollama-host 127.0.0.1 --ollama-port 11499` を実行．3 候補の CV accuracy / macro-F1 を全て journal へ記録し，選定規則に従って 1 モデルへ確定する．
4. **G0-b 判定** → 合格なら 4b，不合格なら代替候補で以降へ進む．
5. **再訓練（wafl-ctrl5 のみ）**: 旧 artifact を退避したうえで `uv run python -m scripts.train_domain_classifier --train-data data/classifier_train.jsonl --embedding-model <選定モデル> --ollama-host 127.0.0.1 --ollama-port 11499 --output models/domain_classifier.joblib`．`n_features_in_` を確認．
6. **G2（wafl-ctrl5 のみ）**: 旧/新 artifact の argmax replay で n_d と `1.96/sqrt(n_d)` を算出し，新 artifact 単体のオフライン accuracy も着地点予測として記録する（判定には使わない）．
7. `config.yaml:4` を書き換え，`uv run ruff check`（既存 23 件は pre-existing）と `uv run pytest tests/`（pre-existing 9 件 FAIL は B122．**新規失敗 0 件**）を確認．
8. `mise run setup`（**直後に `uv sync --extra research` で research extra を復旧する．B118 の落とし穴 2**）→ `wc -l data/dataset.jsonl` = 1915 を再確認．
9. `mise run deploy` → 上記「到達条件」を全 10 ノードで確認．
10. **先頭 20 問の予備実行 → G0-c 判定**．失敗なら代替候補へ切り替えて 5〜9 をやり直す．
11. **wafl500〜509 で 1,915 問のフルスペック本走を 1 回**（絶対条件 A）．起動直後に `state.json` を `status=waiting_experiment`・`experiment_dir`・`experiment_deadline`（開始時刻 + 150×60 + 600 秒）へ更新．`mise run analyze -- <timestamp>` まで実施（**引数なし実行は `results/iter45_preliminary/` を誤選択する．B118 の落とし穴 1**）．
12. 指標を「全 1,915 行」「既存 1,600 行部分集合」「複合 415 行」の 3 通りで算出し，Iter79 基準線 `results/20260926_221822/` と id ペアリングで McNemar 検定・per-domain 20 指標の BH 補正を行う．

**仮説（事前登録）**

「`qwen3-embedding:0.6b` を `qwen3-embedding:4b`（2560 次元）へ替えると，日本語質問文のドメイン分離度がさらに上がり，1,915 行本走の `top1_accuracy` が Iter79 基準線 0.753003 から **+1.5pt 以上**改善する（McNemar p<0.05）．ただし Iter79 のような 2 桁 pt の伸びは期待しない．**0.753 の時点で残る誤りの多くは埋め込み品質ではなくラベル境界の曖昧さ（social_science と history_culture の相互混同など）に由来する可能性があり，効果は 0〜+3pt のどこかに着地する**と見込む．同一モデル族内のスケールアップであるため discordant n_d は Iter79 の 791 より大幅に小さく，**100〜400 行**と予測する．」

**成功条件（事前登録）**

基準線は Iter79 本走 `results/20260926_221822/`（全 1,915 行: top1=0.753003，single_domain_top1=0.749333，compound_domain_top1=0.766265，compound_domain_set_recall=0.548193，kappa=0.721502，misrouting=0.246997，ECE=0.032745，Brier=0.152849，AUROC=0.777614，fallback=0.0，dispatch_failure=0.000522，mean_duration_ms=2301.4，compound_mean_dispatched_count=1.880，answer_quality=0.569333，end_to_end=0.335770／既存 1,600 行部分集合: top1=0.751250）．

| 区分 | 指標 | 現状 | 合格条件 |
|---|---|---|---|
| **G0（VRAM）** | wafl-ctrl5 での常駐サイズ X と PROCESSOR | 0.6b は 2.4GB / 100% GPU | `100% GPU` かつ **X + 8.4GB ≤ 11.5GB**．deploy 後は予備 20 問の平均所要 ≤ 4603ms，または 1,915 問外挿 ≤ 120 分 |
| **G1（値の選定）** | 5-fold CV accuracy（`classifier_train.jsonl` のみ） | 0.6b = 0.7561 | 3 候補の値を全て記録し，選定規則どおり 1 モデルへ確定できること |
| **G2（検出力）** | 旧/新 replay の discordant n_d | 参考: Iter79 は 791 | **n_d ≥ 30**．未満なら config 未到達を疑い本走前に原因を潰す |
| **主基準（効果）** | 全 1,915 行の `top1_accuracy` | 0.753003 | **McNemar p < 0.05 かつ 点推定 +1.5pt 以上**（n_d=200 想定の MDE `1.96·sqrt(n_d)/1915` ≈ 1.45pt を上回る水準） |
| **非退行①** | per-domain recall/precision 計 20 指標 | Iter79 実測 | **BH 補正（q=0.05）後の有意退行 0 件** |
| **非退行②** | `fallback_rate` / `dispatch_failure_rate` | 0.0 / 0.000522 | fallback = 0.0，dispatch_failure ≤ 0.005 |
| **非退行③** | ECE | 0.032745 | **≤ 0.08**（Iter79 の水準から +5pt 以内．再訓練で温度が変わる余地を見込む） |
| **非退行④（新設）** | `mean_duration_ms` | 2301.4 | **≤ 4603（2 倍以内）**．埋め込みモデルの大型化による退避・再ロードの検出用．超過は「実験不成立」ではなく退行として扱う |
| 報告のみ | 既存 1,600 行部分集合の top1 | 0.751250 | 判定には用いないが毎回併記する |
| 報告のみ | `compound_domain_top1` / `single_domain_top1` / `compound_domain_set_recall` / `compound_mean_dispatched_count` | 0.766265 / 0.749333 / 0.548193 / 1.880 | 送出数は `dispatch_gap_threshold=0.29` 固定の二次効果で動きうる（B125 要レビュー (b)） |
| 報告のみ | `answer_quality_accuracy` / `end_to_end_accuracy` | 0.569333 / 0.335770 | 生成器は不変のため横ばいを期待 |

- **adopted**: 主基準（有意改善 かつ +1.5pt 以上）と非退行①②③④をすべて満たす．
- **partial**: 点推定は改善だが p ≥ 0.05 または +1.5pt 未満，かつ非退行に違反なし．この場合は `embedding_model_replacement` の `values` を使い切ったものとして**収束**扱いにし，次レバーは prefix 付与（`embedding_input_instruction_prefix`，B124 要レビュー (a)）または `dispatch_gap_threshold` 再調整（B125 要レビュー (b)）を backlog で比較する．
- **rejected**: 点推定が低下，または非退行①で有意退行が 1 件以上，または非退行④違反．**復元手順**: `config.yaml:4` を `qwen3-embedding:0.6b` へ戻し，`cp models/domain_classifier_pre_iter80_qwen3_0.6b.joblib models/domain_classifier.joblib` で復元し，`mise run deploy` を再実行して全 10 ノードの `n_features_in_`=1024 を確認する．
- **invalid（実験不成立）**: `question_count != 1915`，`compound_domain_question_count != 415`，G2 の n_d が一桁，またはノードの `ollama list` に選定モデルが無い．いずれも setup/deploy の漏れと解釈する（d0004 §4）．

**期待効果とリスク**

期待効果は「同一モデル族のスケールアップで特徴量の分離度をもう一段上げる」ことに尽きる．一方で本反復は Iter79 と異なり，**改善が得られない側に賭けるべき理由（0.753 到達後の残差がラベル境界起因である可能性，JMTEB 4B 値の不在）と，実行自体が危うくなる側のリスク（VRAM 予算 3.4GB に対し 4b の常駐サイズが未知）の双方を抱えている**．前者は G1 で，後者は G0 で本走前に潰す．どちらのゲートでも失敗した場合の分岐（代替候補 `bge-m3` への切り替え，または「改善しない見込み」を記録したうえでの本走実施）を上記のとおり結果を見る前に固定してある．

### Iteration 80 実行済み

**判定: rejected（事前登録の rejected 条件を 2 つとも満たした）**

#### レバーの値が本走前に切り替わった経緯（見出しと実際の値の齟齬）

見出しの `iteration_name`（「0.6b → 4b」）と実際に本走したレバーの値（`bge-m3`）は一致していない．計画節「G0 失敗時の代替（事前登録）」の分岐がそのまま発火したためである（詳細は backlog B126）．

- **G0-a（wafl-ctrl5 実測）**: `qwen3-embedding:4b` の常駐サイズは **4.4GB**（PROCESSOR は `100% GPU`）．
- **G0-b（wafl500 予算判定）**: 合格条件 `X + 8.4GB ≤ 11.5GB` に対し 12.8GB で **1.3GB 超過 → G0 不合格**．
- 事前登録どおり `qwen3-embedding:4b` を断念し，G0 を通過した非現行候補のうち CV 最大の **`bge-m3`**（常駐 **0.664GB**）へ切り替えた．`iteration_name` は追跡性のため据え置いた（計画節 G0-4 の指示どおり）．

**G1（`data/classifier_train.jsonl` 1,427 行のみの 5-fold CV．評価集合を一切見ていない）**

| 候補 | CV accuracy | macro-F1 | 常駐サイズ | G0 |
|---|---|---|---|---|
| `qwen3-embedding:0.6b`（現行＝基準線） | 0.7561 | 0.7562 | 2.4GB | 合格（現行） |
| `qwen3-embedding:4b` | **0.7722** | 0.7718 | 4.4GB | **不合格**（選定対象外） |
| `bge-m3` | 0.7120 | 0.7140 | 0.664GB | 合格 → **選定** |

**CV で最良だった 4b が VRAM だけで排除され，選定規則が残した唯一の候補が基準線を 4.4pt 下回る `bge-m3` だった**．config.yml 絶対条件 A（「改善しない見込みでも最良の非現行候補で本走は実施する」）に従い，着地点が基準線割れになる見込みを本走前に記録したうえで本走した．

**G2（検出力）**: 旧/新 artifact の argmax replay（1,915 行）で discordant **n_d=469 ≥ 30 で合格**．新 artifact 単体のオフライン argmax accuracy は **0.7248**（旧 0.7535）．

#### 変更したもの

1. `config.yaml:4` `embedding_model`: `qwen3-embedding:0.6b` → `bge-m3`（1 行．本レバーの本体）
2. `models/domain_classifier.joblib`: bge-m3 で再訓練（sha256 `37d71b63...`，`n_features_in_`=1024）．旧版は `models/domain_classifier_pre_iter80_qwen3_0.6b.joblib`（sha256 `21e16ec6...`）へ退避
3. `scripts/screen_embedding_models.py`: 3 候補同時評価へ拡張
4. `data/MANIFEST.md`: 新 artifact の sha256・G0/G1/G2 実測値を追記
5. `.claude/research/config.yml`: `embedding_model_replacement` の `values` へ `bge_m3` を追記

到達確認は全て通過（全 10 ノードで `bge-m3:latest` の存在・`embedding_model: bge-m3`・artifact sha256 一致，先頭 20 問の予備実行で 500 エラー 0・平均 2288.9ms ≤ 4603ms）．lint 新規 0 件，test は pre-existing 9 件 FAIL（B122）のみで 306 件 PASS．**実験は成立している**（`question_count=1915`，`compound_domain_question_count=415`）．

#### 結果（本走 `results/20260927_004001/`，1,915 問 1 回．基準線は Iter79 `results/20260926_221822/`）

| 指標 | Iter79 基準線 | Iter80（bge-m3） | 差 | 判定 |
|---|---|---|---|---|
| **top1_accuracy（全 1,915）** | 0.753003 | **0.722715**（95%CI [0.7022, 0.7423]） | **-3.03pt** | **有意な退行** |
| single_domain_top1（1,500） | 0.749333 | 0.744000 | -0.53pt | ノイズ範囲 |
| compound_domain_top1（415） | 0.766265 | 0.645783 | **-12.05pt** | **有意な退行** |
| 既存 1,600 行部分集合 top1 | 0.751250 | 0.735625 | -1.56pt | 有意でない |
| compound_domain_set_recall | 0.548193 | 0.512048 | -3.61pt | 退行 |
| compound_mean_dispatched_count | 1.880 | 2.104 | +0.224 | 増加 |
| cohens_kappa | 0.721502 | 0.715619 | -0.0059 | — |
| misrouting_rate | 0.246997 | 0.277285 | +3.03pt | 退行 |
| fallback_rate | 0.0 | 0.0 | 0 | 非退行②合格 |
| dispatch_failure_rate | 0.000522 | 0.001567 | +0.001 | 非退行②合格（≤0.005） |
| ECE | 0.032745 | 0.033660 | +0.0009 | 非退行③合格（≤0.08） |
| mean_duration_ms | 2301.4 | 2241.04 | -60 | 非退行④合格（≤4603） |
| Brier | 0.152849 | 0.164695 | +0.0118 | 報告のみ |
| AUROC | 0.777614 | 0.767964 | -0.0097 | 報告のみ |
| answer_quality_accuracy | 0.569333 | 0.574667 | +0.53pt | 報告のみ（生成器不変） |
| end_to_end_accuracy | 0.335770 | 0.335248 | -0.05pt | 報告のみ（横ばい） |

**McNemar 検定（id ペアリング，1,915 行完全対応．`scipy.stats.binomtest` の正確二項検定）**

| 区分 | n | 旧のみ正解 b | 新のみ正解 c | n_d | p 値 | 解釈 |
|---|---|---|---|---|---|---|
| 全 1,915 行 | 1915 | 191 | 133 | 324 | **1.50e-03** | **有意な退行** |
| 単一 1,500 行 | 1500 | 127 | 119 | 246 | 0.656 | **差なし（ノイズ範囲）** |
| **複合 415 行** | 415 | 64 | 14 | 78 | **8.58e-09** | **強く有意な退行** |
| 既存 1,600 行部分集合 | 1600 | 150 | 125 | 275 | 0.148 | 有意でない |

**per-domain 20 指標の BH 補正（q=0.05）: 有意退行 2 件（非退行①違反）**

| 指標 | 旧 | 新 | 差 | p |
|---|---|---|---|---|
| **recall: legal** | 0.7621 | 0.5534 | **-20.87pt** | 1.82e-09 |
| **recall: computer_science** | 0.8155 | 0.6915 | **-12.40pt** | 3.88e-04 |

BH 補正で有意にならなかったものの退行側に振れた指標として precision: general（-13.39pt，p=9.4e-03），precision: history_culture（-9.00pt，p=2.6e-02）があり，**20 指標のうち 12 指標が退行側**（改善側 8）に偏っている．

#### 判定と根拠

事前登録の **rejected 条件は「点推定が低下，または非退行①で有意退行が 1 件以上，または非退行④違反」**であり，**前 2 者を独立に満たしている**（-3.03pt の低下，BH 後の有意退行 2 件）．非退行②③④は全て合格だが，rejected 条件は OR であるため判定は **rejected** で確定する．adopted/partial の余地は無い．

#### 分析（ノイズか有意か，および原因）

1. **これはノイズではない．3 つの独立な測定が一致している**．G1 CV -4.41pt（0.7561→0.7120），G2 オフライン argmax -2.87pt（0.7535→0.7248），本走 top1 -3.03pt（0.7530→0.7227）．**オフライン 2 指標が本走の符号と桁をともに正しく予測した**．Iter77〜79 の本走間ばらつき（同一 config の再現差は 1pt 未満）と比べても -3.03pt は明確に外側であり，McNemar p=1.50e-03 がこれを裏づける．
2. **次元数は原因ではない**．`bge-m3` も `qwen3-embedding:0.6b` も **ともに 1024 次元**であり，訓練行数 1,427・LogisticRegression のハイパラ・訓練スクリプトは完全に同一である．計画時に Q3 として構えた「2560 次元による CV と本走の乖離」は，4b が排除された時点で本反復には該当しなくなった．**残る差は埋め込みベクトルの表現そのものだけ**であり，因果の切り分けとしては珍しく綺麗である．
3. **劣化は単一ドメインではほぼ起きず，複合設問に集中している**（単一 -0.53pt / p=0.656 に対し複合 -12.05pt / p=8.58e-09）．しかも複合では `mean_dispatched_count` が 1.880 → 2.104 と**増えているのに** `set_recall` は 0.548 → 0.512 へ**下がった**．`dispatch_gap_threshold=0.29` は固定なので，送出数が増えたのは **bge-m3 の事後確率分布が平坦化し，rank1 と rank2 以降の gap が縮んだ**ことを意味する．つまり **bge-m3 はクラス間マージンが狭く，より多くの候補へ広く送ってなお当てられていない**．AUROC -0.0097・Brier +0.0118 もこの「順位付けの質そのものの低下」と整合する（ECE がほぼ不変なのは，確率の較正は保たれたまま順位の情報量だけが落ちたことを示す）．
4. **モデル特性としての解釈（推測を含む）**: `bge-m3` は multi-vector／検索（retrieval）用途を主眼に対照学習されたモデルで，公開ベンチでも Retrieval 系が高く Classification 系は相対的に低い．本タスクは「クエリ 1 本を 10 クラスへ線形分離する」分類タスクであり，検索向けの埋め込み空間が必ずしも線形分離に有利ではない．**MTEB/JMTEB の総合スコアやモデルサイズは，本タスクの線形分離性を予測しない**（Iter79 の学び 2 の再確認）．
5. **legal（-20.87pt）と computer_science（-12.40pt）の崩れ方**: この 2 ドメインは語彙が特徴的（法令名・条文・API 名・言語名）で，Iter79 の 0.6b 移行で最も伸びた側でもある．検索向けモデルは語の表層一致に引きずられやすく，「法律用語を含むが実質は business_economics の問い」のような行で誤って legal を引き寄せる／逆に一般語で書かれた法律問題を取りこぼす，という双方向の崩れが起きたと見られる（precision: legal は +4.48pt と上がっており，**legal へ入る数が減って残った分は当たっている＝再現率だけが落ちた**形になっている．これは「引き寄せ」ではなく「取りこぼし」が主因であることを示す）．
6. **過剰一般化しない範囲**: 本反復が示したのは「`bge-m3` は本構成では `qwen3-embedding:0.6b` に劣る」ことだけである．`qwen3-embedding:4b` が優れているか否かは**実機で検証していない**（G1 CV 0.7722 は訓練データ内 CV であって本走の予測ではないと計画時に明記してある）．

#### 復元（事前登録した手順を実行済み）

rejected 判定に伴い，計画節の復元手順をそのまま実行した．

1. `config.yaml:4` を `qwen3-embedding:0.6b` へ戻した（`git diff config.yaml` が空＝Iter79 commit の状態に一致）．
2. `cp models/domain_classifier_pre_iter80_qwen3_0.6b.joblib models/domain_classifier.joblib` で復元．sha256 が `21e16ec6...`，`n_features_in_`=1024 であることを確認．
3. `mise run deploy` を再実行．smoke_check（hashes / probe）は全て pass．
4. 全 10 ノードで `embedding_model: qwen3-embedding:0.6b` と artifact sha256 `21e16ec6...` を確認し，コンテナ内 `ollama list` に `qwen3-embedding:0.6b` が存在することを確認した．

**`bge-m3` のイメージは各ノードに残しているが削除していない**（破壊的操作は行わない方針．常駐しない限り VRAM を占有しないため実害は無く，再評価時の再 pull を省ける）．

#### 学び

1. **VRAM 予算はレバーの値空間を実質的に決める**．本反復で最も精度が高かった候補（4b，CV 0.7722）は，精度ではなく **1.3GB の VRAM 超過**だけで排除され，結果として基準線を下回ることが事前に分かっている候補で本走する羽目になった．`ollama ps` の SIZE は重みファイルサイズではなく **KV/活性を含む常駐サイズ**であり，0.6b では 639MB の重みに対し常駐 2.4GB（約 3.8 倍）だった．**「重みサイズ比で VRAM を見積もる」のは誤りで，必ず `ollama ps` で実測すること**．一方 bge-m3 は重み 1.2GB に対し常駐 0.664GB と**重みより小さく**，比率は一定ですらない．
2. **事前登録した分岐は，都合の悪い結果を招くときこそ効いている**．G0 失敗時の代替候補を「結果を見る前に」固定してあったため，「4b が駄目なら基準線を下回ると分かっている bge-m3 を回すのは無駄では」という後付けの判断が入り込む余地が無かった．結果として「検索向けモデルは本分類タスクに向かない」という否定的だが再利用可能な知見が n=1915 の実測で得られた．**事前登録は adopted を守るためではなく rejected を捨てないための装置である**．
3. **G1 オフライン CV は本走 top1 の「順位」を正しく予測した**（bge-m3 < 0.6b）．Iter79 でも CV 0.7561 に対し本走 0.7530 とほぼ一致しており，2 反復連続で整合した．**`embedding_model_replacement` 系のレバーについては，G1 CV で基準線を明確に下回る候補は本走でも下回る**と扱ってよい見込みが立った（ただし計画節で明記したとおり，次元数が大きく変わる候補では乖離が広がる可能性が残るため，CV は選定にのみ用い予測値としては扱わない方針は維持する）．
4. **埋め込みの質の低下は「単一 → 複合」の順に増幅して現れる**．単一 -0.53pt（有意でない）に対し複合 -12.05pt（p=8.6e-09）．複合設問は rank2 以降の候補順位に依存するため，**確率分布のマージンが縮むと真っ先に壊れる**．逆に言えば，**複合 415 行は埋め込み品質の変化に対する感度が単一行より 20 倍以上高い検出器として機能している**（Iter78 の評価集合拡充の投資が，改善だけでなく退行の検出でも回収された）．
5. **`embedding_model_replacement` レバーは使い切った**（`qwen3_embedding_0.6b`=adopted / `qwen3_embedding_4b`=G0 不合格で実機未検証 / `bge_m3`=rejected）．次は**モデルの交換ではなく入力の整形**へ移る（下記 B127）．

---

## Iteration 79: 埋め込みモデルの差し替え（nomic-embed-text → qwen3-embedding:0.6b）

### 調査 (Iter79)

B115(3) の優先順位（複合評価集合拡充 > 埋め込みモデル差し替え > 全ドメイン共通の訓練データ拡充）に従い，Iter78 で最上位項目が完了（partial だが評価集合は常設採用）したため本反復は `embedding_model_replacement` に着手する．B116(3) により本レバーはユーザー事前承認済みで，着手前の追加確認は不要である．調査の問いは 3 つ．**(Q1) `qwen3-embedding:0.6b` は本タスク（日本語質問文の 10 ドメイン分類）で現行 `nomic-embed-text` を上回る根拠があるか．(Q2) Ollama 経由のドロップイン差し替えで，モデル固有の前処理（instruction prefix 等）を欠くことによる性能劣化は起きないか．(Q3) 実装上，差し替えが本走経路まで届く到達条件と，静かに壊れる箇所はどこか．**

**Q1: 多言語 MTEB では上回るが，日本語ベンチマークでは自明ではない**

- `Qwen3-Embedding-0.6B` の公称値は MTEB multilingual Mean(Task) **64.33**（Accenture AI Refinery SDK のモデルカード <https://sdk.airefinery.accenture.com/>，および Qwen3 Embedding 論文 Table <https://arxiv.org/abs/2506.05176> の 0.6B 行．論文表記は評価版差で 70.70 とする箇所もあり，公称値は出典で揺れる）．config.yml の lever note が採った 64.33 と現行 `nomic-embed-text` の 62.28 の差 **+2.05pt** は，この多言語平均での比較である．
- 一方，**日本語ベンチマーク JMTEB での実測は振るわない**．hotchpotch「Qwen3 Embedding 文章ベクトルの日本語性能を JMTEB で測る」（2025-06-11，<https://secon.dev/entry/2025/06/11/100000-qwen3-embedding-jmteb>）によれば，Qwen3-Embedding-0.6B の **Classification 平均は 66.09**．同記事の同一条件の比較対象は `intfloat/multilingual-e5-large` 72.89，`cl-nagoya/ruri-v3-310m` 78.66 で，**日本語特化・多言語専用モデルに明確に劣る**．著者自身が「Qwen3-Embedding-0.6B の性能が低すぎる気もする」と留保を付けている点は割り引く必要があるが，**本研究のタスクは「日本語質問文の埋め込みを特徴量とする多クラス分類」であり，JMTEB Classification が最も近い代理指標である**ことは動かない．
- ただし**現行 `nomic-embed-text` は英語中心のモデル**（Nomic AI，v1/v1.5．多言語版は別系統の `nomic-embed-text-v2-moe`）であり，JMTEB 系のリストにそもそも載っていない．したがって「Qwen3 は日本語が弱い」という所見は，**「現行より弱い」を意味しない**．現行がベースラインとして低い可能性が高く，差し替えによる改善余地自体は残っている．
- **事実と推測の区別**: 「Qwen3-0.6B < multilingual-e5-large（日本語 Classification）」は実測（出典上記）．「Qwen3-0.6B > nomic-embed-text（日本語 Classification）」は**未検証の推測**である．本反復ではこれを推測のまま本走に賭けず，後述のオフライン事前スクリーニングで先に数値化する．

**Q2: instruction prefix と，Ollama 実装の実態（実機で確認済み）**

- Qwen3-Embedding は decoder 型で最終層 `[EOS]` トークンの hidden state を埋め込みとし，**Query 側には `Instruct: {task}\nQuery: {query}` 形式の指示文を付けて学習されている**（Doc 側は素のまま．上記 secon.dev の論文メモ，および Qwen3-Embedding モデルカード <https://huggingface.co/Qwen/Qwen3-Embedding-0.6B>）．prefix を付けないと検索系タスクで性能が落ちるという報告がある（r/LocalLLaMA の議論など．定量値は一次情報として確認できず）．
- **実機確認（wafl-ctrl5，2026-09-26）**: `ollama pull qwen3-embedding:0.6b` は ollama 0.34.4 で成功．`ollama show` は architecture=qwen3，parameters=**595.78M**，context length=**32768**，embedding length=**1024**，quantization=**Q8_0**．`ollama show --template` が返すのは Qwen3 の **chat template** であり，`/api/embed` はこれを適用しない．実際に `/api/embed` へ日本語の医療相談文 1 件を投げて **1024 次元**のベクトルが返ることを確認した．つまり **Ollama 経由のドロップインでは instruction prefix は一切付かない**．
- **本反復では prefix を付けない**．prefix を導入すると `node.py` のクエリ埋め込みと `scripts/train_domain_classifier.py` の訓練側埋め込みの両方にテキスト整形を足すことになり，「埋め込みモデルの差し替え」と「入力整形の追加」の 2 レバーになる．単一レバー原則に反するため，prefix 版は**次反復以降の独立レバー候補**として backlog へ回す．
- 副作用として，`nomic-embed-text`（274MB）→ `qwen3-embedding:0.6b`（Q8_0，約 639MB）で各ノードの常駐 VRAM が約 +365MB 増える．10 ノードとも `light_model`(qwen3.5:4b-q4_K_M，約 2.4GB) + `expert_model` + embedding を常駐させる構成（`OLLAMA_KEEP_ALIVE=-1`）だが，この増分は許容範囲と見込む（要実測）．

**Q3: 到達条件と「静かに壊れる箇所」**

- 本走経路で埋め込みモデル名を読むのは **`config.yaml:4 embedding_model` の 1 箇所のみ**．そこから (a) `node.py:202` `await ollama_client.embed(config["embedding_model"], query)`（依頼者ノードでのクエリ埋め込み）と (b) `http_server.py:402-404`（各ノード起動時の `domain_embedding` 計算）へ届く．`grep` で確認した限り，次元を決め打ちしている箇所は `scripts/run_central_experiment.py`（768 次元のゼロベクトル fallback）と `scripts/fine_tune_embedding_*.py` のみで，**いずれも本走経路の外**（前者は Iter26 の中央集権ベースライン比較専用）．したがって本走は次元非依存で動く．
- 全 10 ノードへのモデル配布は自動である．`tools/node_models.py:get_models()` が `[light_model, expert_model, config["embedding_model"]]` を返し，`mise.toml` の deploy ループ（L96-104）が各ノードで `ollama pull` する．`models/` は L70-74 で `sudo rm -rf` の後 rsync される．
- **静かに壊れない（＝派手に壊れる）箇所**: 分類器 artifact を再訓練せずに埋め込みだけ替えると，1024 次元のベクトルを 768 次元前提の `LogisticRegression` に渡すことになり sklearn が `X has 1024 features, but ... is expecting 768 features` を投げる．`classifier.py:estimate_confidence_classifier` は例外を握りつぶさないので全ノードの `/probe` が 500 になる．**検知は容易だが，デプロイ順序を誤ると本走が丸ごと無駄になる**ため，config.yaml と `models/domain_classifier.joblib` は必ず同一 deploy で配る．
- **本当に静かに壊れる箇所は評価集合の側にある**．`mise run setup` は `build_dataset.py --generated-compound-questions data/compound_questions_generated.jsonl`（既定値）を呼ぶが，このファイルは `.gitignore: data/*` によりローカルにしか存在しない（backlog B123）．**消失すると dataset.jsonl が黙って 1,600 行へ戻り，Iter78 基準線との比較が成立しなくなる．** 本日時点で存在を確認済み（315 行，sha256 `1bfb5add6a5f4028c3ceb52531987aaea595c409bfaca1c98bbf48c80901d3c6`，`data/dataset.jsonl` は 1,915 行）．実験前に再確認する．

**Iter78 の学びの反映**

Iter78 の学び 2「π̂_d はレバー依存の量であり，評価集合の属性ではない．**次レバーの artifact ペアで π̂_d を本走前にオフライン実測し，n_d < 30 ならそこで打ち手を考え直す**」をそのまま実行する．埋め込み空間ごと入れ替える本レバーは Iter77 の education 固有補正撤去（10 ドメイン中 1 つの決定境界しか動かない）とは比較にならない広さのはずで，n_d は数百規模になると見込む（未検証．オフラインで実測する）．

### 計画 (Iter79)

**単一レバー**

`embedding_model_replacement` = **`qwen3_embedding_0.6b`**．`config.yaml:4` の `embedding_model` を `nomic-embed-text` → `qwen3-embedding:0.6b` へ変更し，同じ埋め込みで `models/domain_classifier.joblib` を再訓練する（再訓練は差し替えに構造的に付随する作業であり，別レバーではない）．

**事前登録した値の選定規則（G1 ゲート．Q1 の推測を本走前に潰すため）**

Q1 のとおり「Qwen3-0.6B > nomic（日本語）」は未検証である．そこで**本走の前に，評価集合を一切見ないオフライン事前スクリーニングで値を確定する**（config.yml の「事前シミュレーションは本走 1 点を絞り込むための事前登録手段」に該当）．

1. `data/classifier_train.jsonl`（1,427 行，**訓練データのみ．`data/dataset.jsonl` は使わない**＝評価集合へのリークなし）に対し，候補ごとに埋め込みを計算し，同一の `LogisticRegression(max_iter=1000)` + `sample_weight`（`_extract_sample_weights` と同一）で **5-fold StratifiedKFold の accuracy / macro-F1** を測る．
2. 候補は 2 段階．第 1 段: `nomic-embed-text`（現行基準）と `qwen3-embedding:0.6b`．**qwen3 の CV accuracy が nomic を上回れば，そこで `qwen3-embedding:0.6b` に確定する**（以降の候補は評価しない）．
3. 第 1 段で qwen3 が nomic を**上回らなかった場合に限り**，第 2 段として `bge-m3`（Ollama 公式ライブラリ，1024 次元，567M，多言語，prefix 不要のドロップイン）を追加評価し，**CV accuracy 最大の候補を採る**（同点なら macro-F1，なお同点なら qwen3）．`multilingual-e5-large` は JMTEB Classification 72.89 と有力だが `query: ` prefix が必須で入力整形の追加＝2 レバー目になるため**本反復では候補に含めない**（backlog 送り）．
4. 3 候補すべてが nomic を下回った場合でも，**最良の非 nomic 候補で本走は実施する**（config.yml 絶対条件 A．「施策を適用したら常に本走」）．その際は着地点予測として「改善しない見込み」を事前に journal へ記録してから走らせる．

この規則は結果を見る前に固定してあり，選定に使うのは訓練データの CV のみである．本走の判定指標（評価集合 1,915 行の top1）は選定に一切使わない．

**仮説（事前登録）**

「英語中心の `nomic-embed-text` を多言語対応の `qwen3-embedding:0.6b` へ差し替えると，日本語質問文のドメイン分離度が上がり，1,915 行本走の `top1_accuracy` が Iter78 基準線 0.589556 から **+1.8pt 以上**改善する（McNemar p<0.05）．同時に，埋め込み空間が総入れ替えになるため argmax の discordant 行数 n_d は Iter77 ペアの 19 行から桁違いに増え，200 行以上になる．」

**固定する構成（単一レバー原則）**

`config.yaml` の `embedding_model` 以外の全項目（`routing_method=supervised_classifier`，`confidence_threshold=0.0`，`dispatch_top_k=2`，`dispatch_gap_threshold=0.29`／`dispatch_gap_max_k=4`，`aggregation_method=max_confidence`，`judge_model`，`classifier_model_path`，各ノードの `light_model`/`expert_model`，`probe_timeout_s`/`dispatch_timeout_s`），`data/dataset.jsonl`（1,915 行．Iter78 の成果物をビット単位でそのまま使う），`data/classifier_train.jsonl`（1,427 行，無変更），`scripts/train_domain_classifier.py` のハイパラ（`_CALIBRATION_METHOD="temperature"`，`_CALIBRATION_CV=5`，`_MAX_ITER=1000`，`ensemble=True`，`class_weight=None` + domain-balanced `sample_weight`），`classifier.py` / `http_server.py` / `node.py` / `aggregator.py` / `metrics.py` / `build_dataset.py`，`experiment.timeout_min=150`．**instruction prefix は導入しない．**

**変更するファイルと箇所**

1. `config.yaml:4`: `embedding_model: nomic-embed-text` → `embedding_model: qwen3-embedding:0.6b`（G1 の結果が別候補ならその名前）．**これが本レバーの本体であり，変更はこの 1 行のみ．**
2. `models/domain_classifier.joblib`: 新埋め込みで再訓練して差し替える．**旧 artifact は `models/domain_classifier_pre_iter79_nomic.joblib` へ退避**してから上書きする（Iter77 の `domain_classifier_pre_iter77_edu_corrected.joblib` と同じ慣行．flip 計測に必要）．
3. `scripts/screen_embedding_models.py`（**新規**）: G1 の事前スクリーニング専用．`data/classifier_train.jsonl` を候補モデルごとに埋め込み（`data/embcache_<model>.npy` へキャッシュし，採用モデル分は再訓練でも再利用する），5-fold StratifiedKFold の accuracy/macro-F1 を JSON で標準出力へ出す．**Ollama ホストは wafl-ctrl5 固定**（引数の既定値を `localhost:11499` とし，docstring に SSH トンネル手順を書く）．
4. `data/MANIFEST.md`: 新 artifact の sha256・生成コマンド・使用した埋め込みモデル名を追記する（`data/`・`models/` は `.gitignore` 対象のため MANIFEST が唯一の再現性の担保．B123 で確認した既存規約）．
5. `.claude/research/config.yml`: G1 が第 2 段へ進んで `bge-m3` を採った場合のみ，`embedding_model_replacement` の `values` へ `bge_m3` を追記する（採らなければ変更しない）．

**変更しないが確認だけするファイル**: `tests/test_node.py:170` ほか計 4 つのテストが `"nomic-embed-text"` を文字列リテラルで持つが，いずれもテスト内で組み立てる config 辞書の値であって `config.yaml` を読まない．**テストの修正は不要**（修正するとレバーの外側を触ることになる）．

**レバーを読むコード行と到達条件（d0004 §4 の再発防止．同型の失敗 6 回の教訓）**

- 設定 → 全ノードのモデル配布: `tools/node_models.py:get_models()` が `config["embedding_model"]` を返し，`mise.toml` L96-104 の deploy ループが各ノードで `ollama pull` する．到達確認は **全 10 ノードで `docker compose exec -T ollama ollama list | grep qwen3-embedding` が 1 行返ること**．
- 設定 → 各ノードの config: `mise.toml` L67 の `rsync ... config.yaml` が配布する．到達確認は **全 10 ノードで `grep '^embedding_model:' $REMOTE_DIR/config.yaml`**．
- 設定 → 実行時のクエリ埋め込み: `node.py:202`．到達確認は本走ログでノード起動時の `domain_embedding` 計算（`http_server.py:402-404`）が例外なく通ること，および先頭 20 問の予備実行が 500 を返さないこと（**次元不一致なら必ずここで落ちる**）．
- artifact → 全ノード: `mise.toml` L70-74 の `models/` rsync．到達確認は **全 10 ノードで `docker compose exec app python -c "import joblib;m=joblib.load('/app/models/domain_classifier.joblib');print(m.estimators_[0].estimator.n_features_in_)"` 相当が 1024 を返すこと**（実装が困難なら sha256 一致確認で代替してよい）．
- 実験 → 指標: `metrics.py` は無変更．到達確認は `question_count == 1915` かつ `compound_domain_question_count == 415`．
- **レバー発火の直接証拠**: 旧/新 artifact の argmax replay による flip 行数 n_d が **30 行以上**（後述 G2）．これが一桁なら「効果が無かった」ではなく **config が届いていない**ことを既定の解釈とする．

**実験手順**

1. **前提確認**: `wc -l data/dataset.jsonl` = 1915，`sha256sum data/compound_questions_generated.jsonl` = `1bfb5a...01d3c6` を確認（B123 のリスク．消えていたら本反復は中止して backlog へ差し戻す）．
2. **G1 事前スクリーニング（wafl-ctrl5 のみ．絶対条件 B）**: `ssh -fNT -L 11499:localhost:11434 wafl-ctrl5` のトンネル経由で `scripts/screen_embedding_models.py` を実行．`qwen3-embedding:0.6b` は wafl-ctrl5 に pull 済み（調査 Q2 で実施）．第 2 段へ進む場合のみ `bge-m3` を追加 pull する．**wafl500〜509 は使わない．** 結果の CV accuracy / macro-F1 を journal に全候補分記録し，選定規則に従って 1 モデルを確定する．
3. **分類器の再訓練（wafl-ctrl5 のみ）**: 旧 artifact を `models/domain_classifier_pre_iter79_nomic.joblib` へ `cp` で退避してから，
   `uv run python -m scripts.train_domain_classifier --train-data data/classifier_train.jsonl --embedding-model <選定モデル> --ollama-host 127.0.0.1 --ollama-port 11499 --output models/domain_classifier.joblib`
   を実行する（`--ollama-host` を wafl500 等へ向けないこと．絶対条件 B）．`n_features_in_` が 1024 であることを確認する．
4. **G2 検出力の事前実測（wafl-ctrl5 のみ）**: `scripts/evaluate_classifier_calibration.py` を conformal オプション無し（素の `predict_proba` argmax）で 2 回走らせ，(旧 artifact × `nomic-embed-text`) と (新 artifact × 選定モデル) の `selected_domain` を 1,915 行全体で突き合わせて discordant 行数 n_d と必要偏り率 `1.96/sqrt(n_d)` を算出する．**Iter78 の学び 1 に従い，pt スケールの MDE ではなく n_d を主たる検出力指標として記録する．** 同時に新 artifact 単体のオフライン accuracy も記録する（着地点予測．本走の判定には使わない）．
5. `config.yaml:4` を書き換え，`uv run ruff check` と `uv run pytest tests/`（pre-existing 9 件 FAIL は B122 の既知事項．新規失敗 0 件であることを確認）を通す．
6. `mise run setup`（**直後に `uv sync --extra research` で research extra を復旧する．B118 の落とし穴 2**）→ `wc -l data/dataset.jsonl` = **1915** を再確認．
7. `mise run deploy` → 上記「到達条件」4 点を全 10 ノードで確認．
8. 先頭 20 問の予備実行でノード疎通を確認（**次元不一致はここで必ず出る**）．
9. **wafl500〜509 で 1,915 問のフルスペック本走を 1 回（絶対条件 A）**．起動直後に `state.json` を `status=waiting_experiment`・`experiment_dir`・`experiment_deadline`（開始時刻 + 150×60 + 600 秒）へ更新する．`mise run analyze -- <timestamp>` まで実施（**引数なし実行は `results/iter45_preliminary/` を誤選択する．B118 の落とし穴 1**）．
10. 指標を「全 1,915 行」「既存 1,600 行部分集合」「複合 415 行」の 3 通りで算出し，Iter78 基準線 `results/20260926_195929/` と id ペアリングで McNemar 検定・per-domain 20 指標の BH 補正を行う．

**実験時間の見積りと `timeout_min`**

Iter78 本走は 1,915 問で実測 51 分（`18 分 + n_compound/9.9 分` の経験式に概ね一致）．埋め込みモデルの差し替えは 1 問あたり埋め込み 1 回分のコストしか変えず，`mean_dispatch_gen_time_ms`=1997 が所要時間の大半を占めるため，所要時間はほぼ横ばい（±10 分）と見込む．ただし**ルーティング先の分布が変われば `dispatch_gap` 経由の平均 dispatch 数（Iter78 実測 `compound_mean_dispatched_count`=2.506）が動き，所要時間が変わる可能性がある**（増える向きにも減る向きにもあり得る）．`experiment.timeout_min: 150` は実測の約 3 倍の余裕があるため**変更しない**．130 分を超えた場合のみ次反復で 180 への引き上げを起票する．

**成功条件（事前登録）**

基準線は Iter78 本走 `results/20260926_195929/`（全 1,915 行: top1=0.589556，single_domain_top1=0.628667，compound_domain_top1=0.448193，compound_domain_set_recall=0.393976，kappa=0.587438，misrouting=0.410444，ECE=0.050552，Brier=0.206919，AUROC=0.731903，fallback=0.0，dispatch_failure=0.000522／既存 1,600 行部分集合: top1=0.615000）．

| 区分 | 指標 | 現状 | 合格条件 |
|---|---|---|---|
| **事前ゲート G1（値の選定）** | 5-fold CV accuracy（`classifier_train.jsonl` のみ） | nomic の実測値を基準に取る | 選定規則どおり 1 モデルへ確定できること．全候補の値を journal に記録 |
| **事前ゲート G2（検出力）** | 旧/新 artifact replay の discordant 行数 n_d（1,915 行） | 参考: Iter77 ペアで 19 | **n_d ≥ 30**．必要偏り率 `1.96/sqrt(n_d)` も併記．**n_d < 30 なら config 未到達を疑い，本走前に原因を潰す** |
| **主基準（効果）** | 全 1,915 行の `top1_accuracy` | 0.589556 | **McNemar p < 0.05 の有意改善 かつ 点推定 +1.8pt 以上**（n_d=200 想定の MDE `1.96·sqrt(n_d)/1915` ≈ 1.45pt を上回る水準として設定） |
| **非退行①** | per-domain recall/precision 計 20 指標 | Iter78 実測 | **BH 補正（q=0.05）後の有意退行 0 件** |
| **非退行②** | `fallback_rate` / `dispatch_failure_rate` | 0.0 / 0.000522 | fallback = 0.0，dispatch_failure ≤ 0.005 |
| **非退行③** | ECE | 0.050552 | **≤ 0.10**（+5pt 以内．埋め込み空間が変われば温度スケーリングの最適値も変わるため，ここは悪化余地を見込んで緩く置く） |
| 報告のみ | 既存 1,600 行部分集合の top1 | 0.615000 | 判定に用いないが毎回併記する（B119 要レビュー (c) への回答） |
| 報告のみ | `compound_domain_top1` / `single_domain_top1` / `compound_domain_set_recall` | 0.448193 / 0.628667 / 0.393976 | 複合設問への効果を初めて 415 行で測る参考値．Iter78 の学び 2 のとおり，ここが動くかどうかが評価集合拡充の投資回収の指標になる |
| 報告のみ | `answer_quality_accuracy` / `end_to_end_accuracy` | 0.564 / 0.287728 | 埋め込み差し替えは生成器を変えないため横ばいを期待 |

- **adopted**: 主基準（有意改善 かつ +1.8pt 以上）と非退行①②③をすべて満たす．
- **partial**: 点推定は改善だが p ≥ 0.05 または +1.8pt 未満，かつ非退行に違反なし．この場合，`qwen3-embedding:4b`（config の第 2 値）または prefix 付与版を次反復の候補として backlog へ残す．
- **rejected**: 点推定が低下，または非退行①で有意退行が 1 件以上．rejected の場合は `config.yaml` を `nomic-embed-text` へ戻し，`models/domain_classifier.joblib` を退避した旧 artifact から復元する（復元手順を journal に明記してから本走に入ること）．
- **invalid（実験不成立）**: `question_count != 1915`，`compound_domain_question_count != 415`，G2 の n_d が一桁，またはノードの `ollama list` に選定モデルが無い．いずれも「効果なし」ではなく setup/deploy の漏れと解釈する（d0004 §4）．

**期待効果**

現行の `nomic-embed-text` は英語中心のモデルであり，日本語 1,915 問のルーティングを英語埋め込み空間の上で行っている．ここを多言語モデルへ替えることは，これまでの 78 反復で触れてこなかった**特徴量そのものの品質**への初めての介入である（Iter39〜43 の fine-tuning はベースモデルを保ったまま補正する試みで，argmax flip rate が過大となり単一レバー原則と両立せず rejected だった．本レバーは差し替えなので同じ制約を受けない）．同時に，Iter78 で拡充した複合 415 行に対して初めて意味のある検出力で効果を測る機会でもある．

### Iteration 79 実行済み

**判定: adopted（事前登録した全条件を満たす）**

#### 変更したもの

`config.yaml:4` の `embedding_model` を `nomic-embed-text` → `qwen3-embedding:0.6b` の 1 行のみ．これに構造的に付随する作業として `models/domain_classifier.joblib` を新埋め込み（1024 次元）で再訓練し（旧版は `models/domain_classifier_pre_iter79_nomic.joblib` へ退避），`scripts/screen_embedding_models.py`（G1 専用・新規）を追加，`data/MANIFEST.md` に新 artifact の sha256・生成コマンド・G1/G2 実測値を追記した．`data/dataset.jsonl`（1,915 行）・`data/classifier_train.jsonl`（1,427 行）・訓練ハイパラ・その他 config は計画どおりビット単位で固定した．instruction prefix は付けていない．

#### 事前ゲートの結果

- **G1（値の選定，`classifier_train.jsonl` のみの 5-fold StratifiedKFold CV．評価集合不参照）**: `nomic-embed-text` cv_accuracy 0.5711 / macro-F1 0.5710，`qwen3-embedding:0.6b` cv_accuracy 0.7561 / macro-F1 0.7562．第 1 段で qwen3 が nomic を **+18.50pt** 上回ったため事前登録の規則どおりここで確定し，第 2 段（`bge-m3`）は起動しなかった．`scripts/screen_embedding_models.py` が `data/dataset.jsonl` を一切参照しないことをソースで確認済み（docstring L10 に明記，`--train-data` 既定は `classifier_train.jsonl`）．
- **G2（検出力）**: 旧/新 artifact の argmax replay で discordant n_d=787（閾値 30 を大幅超過）．本走実測でも `selected_domain` の discordant は **n_d=791**（必要偏り率 `1.96/sqrt(791)`=6.97%）で，replay と整合する．計画が予測した「n_d は 200 行以上」を上回り，レバーが実行時経路まで到達したことの直接証拠になっている．

#### 本走の結果（`results/20260926_221822/`，基準線 Iter78 `results/20260926_195929/`）

rc-executor の報告は，`metrics.py` の再実行と `results.jsonl` からの独立再計算で全項目を裏取りした．`expected_domains` と `query` は 1,915 行すべてで両走一致しており，評価集合は同一である（確認済み）．

| 指標 | Iter78 | Iter79 | 差 |
|---|---|---|---|
| top1_accuracy（全 1,915 行） | 0.589556 | **0.753003**（Wilson CI [0.73319, 0.77180]） | **+16.34pt** |
| cohens_kappa | 0.587438 | 0.721502 | +13.41pt |
| misrouting_rate | 0.410444 | 0.246997 | -16.34pt |
| ECE / Brier / AUROC | 0.050552 / 0.206919 / 0.731903 | 0.032745 / 0.152849 / 0.777614 | すべて改善 |
| fallback_rate / dispatch_failure_rate | 0.0 / 0.000522 | 0.0 / 0.000522 | 同一（同じ 1 行） |
| single_domain_top1（1,500 行） | 0.628667 | 0.749333 | +12.07pt |
| compound_domain_top1（415 行） | 0.448193 | 0.766265 | +31.81pt |
| compound_domain_set_recall | 0.393976 | 0.548193 | +15.42pt |
| compound_mean_dispatched_count | 2.506 | 1.880 | **-0.63**（後述） |
| answer_quality / end_to_end | 0.564 / 0.287728 | 0.569333 / 0.335770 | +0.53pt / +4.80pt |
| mean_duration_ms | 約 2300 | 2301.4 | 横ばい |

McNemar（連続性補正あり，自前再計算）: 全 1,915 行で a_only(旧のみ正解)=128，b_only(新のみ正解)=441，discordant=569，chi2=171.08，**p=4.30e-39**．部分集合も同様に全て有意改善（単一 1,500 行: 114/295，p=5.56e-19／複合 415 行: 14/146，p=3.91e-25／既存 1,600 行部分集合 top1 0.615000→0.751250，118/336，p=2.33e-24）．

**非退行①（per-domain recall 10 + precision 10 の計 20 指標，BH 補正 q=0.05）**: 自前で再計算（recall は exact McNemar，precision は Fisher）した結果，**有意 10 件・方向はすべて改善・有意退行 0 件**．点推定で悪化したのは `social_science_recall`（0.4242→0.4156，p=0.888）と `history_culture_recall`（0.7792→0.7706，p=0.896）の 2 件のみで，いずれも有意でない．非退行②（fallback=0.0，dispatch_failure=0.000522 ≤ 0.005）・非退行③（ECE 0.032745 ≤ 0.10．むしろ改善）も充足．

**到達確認**: `question_count`=1915，`compound_domain_question_count`=415，全 10 ノードで `ollama list` に `qwen3-embedding` と `config.yaml` の値を確認，新分類器の `n_features_in_`=1024，先頭 20 問の予備実行で 500 エラーなし．invalid 条件には一つも該当しない．

**検証**: `uv run ruff check` 新規失敗 0 件（既存 23 件は pre-existing），`uv run pytest tests/` 306 passed / 9 failed（9 件すべて B122 の pre-existing，新規失敗 0 件）．

#### 分析（交絡の検討 — +16.3pt は過去のどのレバーよりも大きいため，本物であることを個別に潰す）

1. **分類器の再訓練そのものの寄与と，埋め込みの寄与を分離できているか**．G1 は**同一の訓練データ・同一の訓練手順・同一のハイパラ・同一の CV 分割（`random_state` 固定）で埋め込みだけを差し替えた比較**であり，cv_accuracy 0.5711→0.7561（+18.50pt）を得ている．訓練手順が同一である以上，この差は埋め込み特徴量の品質に帰属する．本走の +16.34pt は G1 の +18.50pt より小さく，オフラインの分離度改善が実機へほぼそのまま（やや目減りして）伝播した，という素直な解釈と整合する．「再訓練したから上がった」だけなら nomic で再訓練した対照でも上がるはずだが，G1 の nomic 側がまさにその対照であり 0.5711 に留まっている．
2. **評価集合への情報漏洩**．G1 も再訓練も `data/classifier_train.jsonl` のみを使い，`data/dataset.jsonl` を参照していない（スクリプトの引数とソースで確認）．値の選定規則は結果を見る前に journal へ事前登録済みで，選定に評価集合の指標を一切使っていない．
   - ただし**副次的に既存の漏洩を 1 件発見した**．`classifier_train.jsonl` の質問本文と `dataset.jsonl` の質問本文が **72 行重複している**（education 54 行，history_culture 18 行）．d0002 §6-E の「重複 0 件」という記述は現状に合致しない（Iter35 の education 手作り問題追加，または Iter36/37 の `japanese_civics` 再割当以降に混入したと見られる）．**この 72 行は旧走・新走の双方に等しく含まれるため今回の比較を歪めないうえ，向きも逆である**: 漏洩 72 行では新モデルの方が**悪く**（0.875→0.653），漏洩 72 行を除いた 1,843 行では改善幅がむしろ広がる（0.578405→0.756918，**+17.85pt**）．したがって漏洩は今回の判定を有利側へ押していない．絶対値の水増しという別問題は残るため backlog B125 に起票した．
3. **Iter78 で拡充した新規 315 行が新モデルに有利に働いていないか**．複合設問を由来で分けると，**旧来の手作り 100 行が 0.410→0.780（+37.0pt，a=4/b=41，p=8.0e-8）**，Iter78 の LLM 生成 315 行が 0.460→0.762（+30.2pt，a=10/b=105，p=1.9e-18）で，**改善幅が大きいのは新規 315 行ではなく旧来の 100 行の側**である．新規行が新モデルに有利という交絡は支持されない．なお単独で最も改善が小さいのは単一ドメイン 1,500 行（+12.07pt）であり，複合設問ほど恩恵が大きいという構図になっている．
4. **想定外の挙動**．言語崩れ・発散・OOM は観測されず，`mean_duration_ms` も横ばい（2301.4）．唯一の想定外は `compound_mean_dispatched_count` の 2.506→1.880 という低下である．これは `dispatch_gap_threshold=0.29` を固定したまま確信度の分布が鋭くなった（`mean_confidence_std` が上がった）結果，rank2 以降が gap 条件で落ちやすくなった二次効果であり，レバーの直接の帰結ではない．**それでも `compound_domain_set_recall` は 0.394→0.548 と改善しており，「送る数を減らしたのに当てる数が増えた」**（送出先の質が上がった）と読める．ただし複合設問で 2 ドメイン中 1 つしか送られない行が増えている可能性があるため，`dispatch_gap_threshold` の再調整は独立レバーとして起票する価値がある（B125）．

#### 考察（採否判定）

事前登録した adopted 条件は「主基準（McNemar p<0.05 かつ点推定 +1.8pt 以上）かつ非退行①②③をすべて満たす」である．主基準は p=4.30e-39・+16.34pt で桁違いに充足し，非退行①（BH 補正後の有意退行 0 件）・②・③もすべて充足した．**adopted で確定する**．`config.yaml` の `qwen3-embedding:0.6b` と再訓練済み `models/domain_classifier.joblib` はこのまま本番構成として維持する（rejected 時の復元手順は不要となった）．

これは Iter15 以降の全反復で最大の単一レバー効果であり，Iter17 の `routing_method=supervised_classifier`（0.2059→0.5651）に次ぐ規模の前進である．top1_accuracy は初めて 0.75 を超えた．

**単一レバー原則の観点で，分類器の再訓練を同一レバーに含めた判断が妥当だったか**: 妥当だったと判断する．埋め込みを 768→1024 次元へ替えると旧 artifact は次元不一致で例外を投げるため，「差し替えだけして再訓練しない」という構成はそもそも実行不能であり，再訓練は選択肢ではなく差し替えの構成要素である．そのうえで G1 が「訓練手順を固定して埋め込みだけを変えた対照」を提供しているため，再訓練の寄与と埋め込みの寄与は事後に分離できている．この形（**付随作業が避けられない場合は，付随作業を固定した対照をオフラインで別に取る**）は今後の単一レバー運用の型として再利用できる．

#### 学び

1. **78 反復のあいだ，日本語タスクを英語中心の埋め込み空間の上で解き続けていた**．`nomic-embed-text` は英語中心モデルで，JMTEB 系のリストにすら載っていない．Iter29〜77 の大半は，その低品質な特徴量を前提にした後段（較正・閾値・intercept・集約方式・conformal）の微調整であり，改善幅は概ね ±1〜3pt に収まっていた．**後段の作り込みを重ねる前に，特徴量そのものの妥当性（入力言語とモデルの学習言語の一致）を疑うべきだった**．G1 相当の CV スクリーニングは数分で終わる．教訓は「安価なオフライン検証で前提そのものを測れる場合は，レバーの優先順位に関わらず早期に測る」である．
2. **公称ベンチマークの選び方が判定を左右する**．config.yml の lever note は MTEB multilingual 平均（nomic 62.28 vs qwen3 64.33，差 +2.05pt）を根拠にしていたが，実測は +18.50pt（CV）/ +16.34pt（本走）だった．一方，調査で見つけた JMTEB Classification は「qwen3-0.6b 66.09 は multilingual-e5-large 72.89 に劣る」と示しており，この数字だけを見ると着手を見送りかねなかった．**どちらの公称値も本タスクの効果量を予測できていない**．B124 で G1 を挟む判断をしたことが，この不確実性をコスト数分で解消した．公称ベンチは候補を絞る道具であって，採否の根拠にはならない．
3. **効果が大きいレバーほど，二次効果が別のハイパラを陳腐化させる**．確信度分布が鋭くなったことで `dispatch_gap_threshold=0.29`（Iter7x 世代に nomic の分布で調整した値）の実効が変わり，複合設問の平均送出数が 2.506→1.880 へ落ちた．**過去に調整したハイパラは，特徴量を入れ替えた時点で「調整済み」ではなくなる**．今回は結果的に set_recall も改善したため無害だったが，次以降は再調整を独立レバーとして検討する．
4. **train/eval の重複 0 件という前提は，データを触るたびに再検証が要る**．d0002 §6-E の確認は Iter17 時点のもので，Iter35/36/37 のデータ変更を経て現在 72 行が重複している．データセットを変更するレバーの計画フェーズに，重複チェックを定型作業として組み込むべきである．

---

