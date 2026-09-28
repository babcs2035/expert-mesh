## Iteration 99: 埋め込みを日本語特化モデル ruri-v3-310m へ差し替える

### 調査 (Iter99)

**実施場所の申告**: 本フェーズで実験ノード wafl500〜509 には一切触れていない（2026-09-23 絶対条件 (B)）．
行ったのは (a) 開発ホストでのリポジトリ読み取りと既存 `results/20260928_160921/results.jsonl` の
`metrics.py` 再計算（CPU のみ），(b) 公開 API（Hugging Face Hub・Ollama registry）への read-only な
HTTP 問い合わせ，(c) **wafl-ctrl5 への read-only な状態確認**（`docker ps` / `/api/version` / `/api/ps`）
だけである．`models/`・`data/`・`config.yaml` への書き込みは 0 件．モデルの pull・ロードは行っていない
（G0 は実装フェーズの作業として下記に手順を明文化した）．

**(1) 問い**（backlog B165 (e) の必須申し送り 5 点を問いに落としたもの）

- Q1: `cl-nagoya/ruri-v3-310m` を**現行の実行経路（Ollama の `/api/embeddings`）でそのまま取得・実行できるか**．
  できない場合の失敗の出方は「明示的なエラー」か「無言の劣化」か．
- Q2: ruri-v3 の入力書式（prefix 規約）は現行のハードコードされた Qwen instruct 書式と両立するか．
  両立しないなら，その差分は「モデル差し替えに構造的に付随する作業」か「2 本目のレバー」か．
- Q3: 次元・p/n 比はどう変わり，それは仮説にどう効くか．
- Q4: 公称ベンチ（JMTEB）から何をどこまで言えるか．事前投影はどうなるか．

**(2) Q1: GGUF は存在するが「patched llama.cpp で変換」と明記されており，Ollama で動く保証は無い**

- **取得元は 1 つだけ確認できた**: `Targoyle/ruri-v3-310m-GGUF`
  （<https://huggingface.co/Targoyle/ruri-v3-310m-GGUF>，HF Hub API で 2026-09-29 に実測．
  ファイルは `ruri-v3-310m-q8_0.gguf` の **1 本のみ**，`x-linked-size` = **336,949,248 バイト（337MB）**，
  tags に `modernbert`・`base_model:cl-nagoya/ruri-v3-310m`，license `apache-2.0`）．
  q8_0 以外の量子化は公開されていない．
- **最大のリスクは同 README の次の 1 文である**（原文）: 「It was converted using a **patched version of
  `llama.cpp`** to support the ModernBERT architecture with SentencePiece tokenizer」．
  すなわち**上流の llama.cpp/Ollama でロードできるとは作者自身が書いていない**．
  使用例も `llama-embedding` / `llama-server` であって Ollama ではない．
- 上流側の状況: ModernBERT アーキテクチャの llama.cpp 本体への対応は **2025-12-22 にマージ**された
  （llama-cpp-python issue #2144「ModernBERT architecture support was added to llama.cpp on Dec 22, 2025」
  <https://github.com/abetlen/llama-cpp-python/issues/2144>）．ただし ruri-v3 のトークナイザは
  **SentencePiece（`vocab_size` 102,400）**であり，上流対応が BPE 前提なら別問題として残る（未確認）．
- **wafl-ctrl5 の Ollama は `0.34.4`**（2026-09-29 に `/api/version` で実測）．常駐は
  `qwen3-embedding:4b` 3.26GB ＋ `qwen3-embedding:0.6b` 2.37GB ＋ swallow-8B 5.27GB ＝ **約 10.9 / 12GB**．
  337MB の追加は算術上は収まるが，実測で確認する（G0-e）．
- **失敗の出方は 2 種類あり，片方は無言である**．(a) 明示的エラー:
  `unknown model architecture` / `model does not support embeddings`（Ollama issue #12757 が後者の実例．
  <https://github.com/ollama/ollama/issues/12757>）．(b) **無言の劣化**: アーキテクチャが allowlist 外だと
  pooling 種別や次元の指定が無視され，**エラーを出さずに誤ったベクトルが返る**
  （LM Studio bug tracker issue #2177「Embedding GGUFs on non-allowlisted archs ... `/v1/embeddings` can then
  return wrong-dimension vectors instead of erroring」<https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/2177>）．
  本リポジトリは「config を正しく変えたのにコードへ到達せず基準線とビット単位一致」という同型事故を
  6 回起こしている（d0004 §4）ので，**(b) を捕まえる数値検査を G0 に必ず含める**（G0-d）．
- なお ruri-v3 は Ollama 公式 library には無い（`https://ollama.com/library/...` は 404）．
  実運用例としては `hf.co/Targoyle/ruri-v3-310m-GGUF` を Ollama から直接引く記述が複数ある
  （例: <https://github.com/haruneko/local-bot/blob/main/docs/DECISIONS.md>「ruri-v3（日本語特化・768 次元・
  `hf.co/Targoyle/ruri-v3-310m-GGUF`）。`/api/embed`」）．これは**第三者の報告であって本環境での検証ではない**．
- **代替（G0 不合格時）の実現性は確認済み**: `multilingual-e5-large` は Ollama registry に
  `zylonai/multilingual-e5-large` として manifest が存在する（2026-09-29 に registry API で 200 応答を実測）．

**(3) Q2: ruri-v3 の prefix 規約は現行のハードコードと非互換．分類用途の規定値は `トピック: ` である**

- ruri-v3 の model card（<https://huggingface.co/cl-nagoya/ruri-v3-310m>，原文を取得して確認）は
  **「1+3 prefix scheme」**を定め，用途別に次を規定する．
  - **`トピック: ` — is used for classification, clustering, and encoding topical information**（＝本タスク）
  - `検索クエリ: ` — retrieval のクエリ側，`検索文書: ` — retrieval の文書側，および prefix なし．
- 一方，本リポジトリの `expert_backend.py:157` は
  `prompt = f"Instruct: {instruction}\nQuery: {text}" if instruction else text` と
  **Qwen3-Embedding の instruct 書式をハードコード**している．ruri にこの書式を与えるのは
  モデルが規定する入力分布から外れ，「ruri へ差し替えた」ことにならない．
- **判断（自律判断．backlog B166 (a) に記録）**: prefix の**組み立て方（テンプレート）はモデル固有の入力書式**で
  あり，次元変更に伴う分類器再訓練と同じく**差し替えに構造的に付随する作業**と扱う．Iter81 のレバー
  （instruction の**文言探索**）の再開ではない．したがって**文言は掃引せず model card の規定値 1 つに固定**する．
- 他の候補（`検索クエリ: `）や，Iter81 の英文 instruction を ruri 用に流用する案は**一切比較しない**．
  比較した時点で単一レバー原則が濁る．

**(4) Q3: 次元は 5120 → 1536．p/n は 2.25 → 0.67 へ下がる**

- 現行 artifact `models/domain_classifier.joblib`（sha256 `2f801357...`）は
  `qwen3-embedding:4b` の 2 ビュー連結＝ **2560 × 2 = 5120 次元**，訓練集合は
  `data/classifier_train_iter94_dedup.jsonl`（**2,275 行**）．p/n = 2.25．
- ruri-v3-310m は `hidden_size` = **768**（`config.json` を実測）なので 2 ビュー連結で **1536 次元**．
  p/n = 0.67 へ下がる．**方向としては過学習側のリスクが減り，表現容量側のリスクが増える**．
  Iter89 の学び（特徴空間の入れ替えは Q1 層＝最難層を大きく動かす）が効くかどうかは事前には決まらない．
- pooling は **mean**（`1_Pooling/config.json` の `pooling_mode_mean_tokens: true`．
  なお `config.json` の `classifier_pooling: "cls"` は分類ヘッド用の別設定であり sentence-transformers の
  埋め込み経路は mean pooling である）．**GGUF 側の pooling 種別がこれと食い違うと (2)(b) の無言の劣化になる**．
  `max_position_embeddings` = 8192 で，本データの最長行（評価 1,811 文字・訓練 1,239 文字）は余裕で収まる．

**(5) Q4: 公称ベンチは「候補を絞る道具」に留める（Iter79 学び 2）．事前投影は幅が広い**

- JMTEB Classification（hotchpotch, 2025-06-11，
  <https://secon.dev/entry/2025/06/11/100000-qwen3-embedding-jmteb>）: **ruri-v3-310m 78.66**，
  **multilingual-e5-large 72.89**，Qwen3-Embedding-0.6B 66.09．
- **ただし現行は `qwen3-embedding:4b` であり，4b の JMTEB 値は今回も見つからなかった**（Iter89 の調査時と同じ状態）．
  したがって「ruri 78.66 > 現行」は **0.6b 比の外挿**にすぎない．Iter79 の学び 2（MTEB 差 +2.05pt に対し
  実測 +16.34pt）と Iter89（MTEB +5.12pt に対し実測 +2.94pt）のいずれも公称差と実測差が一致しておらず，
  **採否は本走でのみ決める**（B165 (4)）．
- **事前投影**: 点推定 **Δtop1 ≒ +1.0pt**，80% 区間 **−2.0 〜 +4.0pt**．
  上振れ根拠は日本語特化（JMTEB Classification で 0.6b 比 +12.57pt）と，Iter89 が示した
  「特徴空間の入れ替えは訓練データ系列より交換比が良い」という構造．
  下振れ根拠は (i) 現行が既に 4b で底上げ済み，(ii) 次元が 1/3.3 に減る，(iii) q8_0 量子化と
  GGUF 変換経路の忠実性が未検証，の 3 点．**符号の不確実性が大きいことを事前に明記しておく**．

### 仮説 (Iter99)

**Iter98 の機序 M6（誤りは決定則ではなく入力表現の性質に由来する．腕間の誤答行の重なり 87.6%/93.9%）が
正しいなら，改善は入力表現を替えたときにのみ起こる．**日本語の学術・専門ドメイン文を，多言語汎用モデル
（`qwen3-embedding:4b`）ではなく日本語専用に事前学習・対照学習された `ruri-v3-310m` で，かつ
**分類・クラスタリング用途として規定された `トピック: ` prefix** で符号化すれば，
現行で誤っている行（とくに Iter89 が特定した最難層 Q1）の一部が線形分離可能な位置へ移り，top1 が上がる．

**対抗仮説（同じ強さで想定する）**: (H2) 768×2 = 1536 次元は 10 ドメインを分けるには容量が足りず，
5120 次元の現行に劣る．(H3) GGUF 変換（patched llama.cpp・q8_0）が参照実装を再現せず，
公称性能が発現しない．H3 は**効果の不在ではなく実験の不成立**なので，G0-d で本走前に切り分ける．

### 単一レバー (Iter99)

- **レバー**: `embedding_model_replacement` = **`japanese_specialized_ruri_v3_310m`**
  （config.yml に Iter98 分析フェーズが事前登録済みの値．新レバーの追加は不要）．
- **何を何から何へ**: `config.yaml:4` の
  `embedding_model: qwen3-embedding:4b` → **`hf.co/Targoyle/ruri-v3-310m-GGUF:Q8_0`**（G0 で確定する正式タグ名）．
  これに構造的に付随する作業は次の 2 つで，**いずれも別レバーではない**（Iter79/89 で確立した型）．
  1. **分類器の再訓練**（`models/domain_classifier.joblib` を同じ訓練集合・同じモデル定義で作り直す．
     次元が 5120 → 1536 に変わるため必須．B165 (3)）．
  2. **prefix テンプレートのモデル固有化**（上記 Q2．`config.yaml` に新キー
     **`embedding_prompt_template`** を足し，既定値を現行と**ビット単位で同一**の
     `"Instruct: {instruction}\nQuery: {text}"` にしたうえで，本反復では `"トピック: {text}"` に設定する）．
     config.yaml のスキーマ変更は B116 (3) で本レバーに限り事前承認済み．
- **固定する構成（1 つも動かさない）**:
  `data/classifier_train_iter94_dedup.jsonl`（2,275 行）・`data/dataset.jsonl`（3,750 行，sha256 不変）・
  `embedding_view_concat: true`（連結順 [prefix なし, prefix あり] も不変）・
  `routing_method=supervised_classifier`・`confidence_threshold=0.0`・`dispatch_candidate_threshold=0.0`・
  `dispatch_top_k=2`・**`dispatch_gap_threshold=0.36`（B165 (5)．埋め込み変更で gap 分布は必ず動くが，
  閾値の再較正は別レバーであり本反復では絶対に触らない）**・`dispatch_gap_max_k=4`・
  `aggregation_method=max_confidence`・`judge_model`・各ノードの `light_model`/`expert_model`・
  `scripts/train_domain_classifier.py` のモデル定義（`LogisticRegression(max_iter=1000, class_weight=None)`
  ＋ `_extract_sample_weights()` ＋ `CalibratedClassifierCV(method='temperature', ensemble=True)`）・
  `classifier.py`・`aggregator.py`・`metrics.py`・`ecoc_head.py`（本走経路へは未配線のまま）．
  **ドメイン固有の後付け補正は追加しない．棄権／エスカレーション系には触れない．**

**レバーを読むコード行と，そこへ到達する条件（d0004 §4．6 回繰り返した同型事故の恒久対策）**

| # | 経路 | 読むコード | 到達確認（1 つでも欠けたら実験不成立） |
|---|---|---|---|
| 1 | 設定 → 10 ノードへのモデル配布 | `tools/node_models.py:13` が `config["embedding_model"]` を返し `mise.toml` の deploy が `ollama pull` | 全 10 ノードの `ollama list` に ruri の行があること |
| 2 | 設定 → 各ノードの config | `mise.toml` の `rsync config.yaml` | 全 10 ノードで `grep '^embedding_model:' $REMOTE_DIR/config.yaml` が新値 |
| 3 | 設定 → 実行時のクエリ埋め込み | `node.py:202-207` の `embed_query_views(..., config["embedding_model"], ...)` | 予備 20 問が HTTP 500 を返さないこと（1536 次元を 5120 次元の旧 artifact に食わせれば `predict_proba` が必ず例外になるので，Iter36 型の無言の不一致はここで必ず落ちる） |
| 4 | artifact → 10 ノード | `mise.toml` の `models/` rsync | 全ノードの `domain_classifier.joblib` の sha256 一致かつ **`n_features_in_ == 1536`** |
| 5 | **新キー `embedding_prompt_template` の到達（本反復の最大の新規リスク）** | `expert_backend.py:157` のハードコードを置換．呼び出し側は `node.py:202`・`scripts/train_domain_classifier.py:158`・`tools/smoke_check.py:170` の **3 箇所すべて** | (a) 単体テストで template 適用を検証，(b) 訓練時に実際に使った template 文字列を標準出力へ印字し journal に転記，(c) **prefix ありビュー（後半 768 次元）が prefix なしビュー（前半 768 次元）とベクトルとして一致しないこと**を訓練特徴の先頭 1 行で実測（一致したら template 未適用） |
| 6 | 実験 → 指標 | `metrics.py` 無変更 | `total_questions == 3750` かつ `compound_domain_question_count == 730` |

### 事前ゲート G0（実現性．wafl-ctrl5 のみ．wafl500〜509 不使用）

**判定規則は結果を見る前にここで固定する．G0 は「効果の有無」ではなく「そもそも実行可能か」だけを見る．**

- **G0-a（取得）**: wafl-ctrl5 の Ollama（`0.34.4`）で `ollama pull hf.co/Targoyle/ruri-v3-310m-GGUF:Q8_0`
  が成功する（337MB）．タグ名が異なる場合は registry の manifest を引いて正式タグを特定してよい
  （これは調達手段の確定であって設計選択ではない）．失敗なら**不合格**．
- **G0-b（ロードと応答）**: `/api/embeddings` に日本語 1 文を POST して 200 が返り `embedding` が得られる．
  `unknown model architecture` / `model does not support embeddings` 等が出たら**不合格**．
- **G0-c（次元）**: 返る次元が **768** であること（`config.json` の `hidden_size` と一致）．
  768 以外なら**不合格**（LM Studio issue #2177 型の事故）．
- **G0-d（数値忠実性．無言の劣化を捕まえる必須検査）**:
  wafl-ctrl5 上で `sentence-transformers` の `cl-nagoya/ruri-v3-310m`（fp32．310M なので RTX 3060 で十分）を
  **参照実装**とし，`data/classifier_train_iter94_dedup.jsonl` から無作為 **32 行（`random_state=99`）**について
  `トピック: ` 付きの埋め込みを両経路で計算する．合格条件は
  **行ごとコサイン類似度の中央値 ≥ 0.99 かつ最小 ≥ 0.97**．
  下回る場合は pooling 種別・正規化・トークナイザの不一致が疑われるので**不合格**とする
  （「精度が出なかった」ではなく「実験不成立」として扱う根拠になる）．
- **G0-e（VRAM・常駐）**: `ollama ps` で ruri が **`100% GPU`** であること．deploy 後は全 10 ノードで
  ruri と当該ノードの `expert-mesh-*-lora` がともに `100% GPU`（`light_model` の退避は合格を妨げない．
  Iter89 で確立済みの読み）．
- **G0 不合格時の分岐（B165 (2)．Iter89 で確立した型）**: 値を **`multilingual_e5_large`**
  （`zylonai/multilingual-e5-large`．registry に存在することを本日確認．**1024 次元 → 連結 2048 次元**，
  `embedding_prompt_template` は **`"query: {text}"`**，参照実装は `intfloat/multilingual-e5-large`）へ
  切り替え，G0-a〜e を同じ基準で再適用する．**`iteration_name` は追跡性のため変更せず**，
  切り替えた事実と理由を journal の実行節に明記する．
  **E5 も不合格なら `invalid`（実現性）**として本走を行わず分析フェーズへ渡し，
  backlog B165 要レビュー (C)（GGUF 変換や sentence-transformers 経路の導入は 10 ノードの実行基盤の
  構成変更にあたる）として**人間判断を仰ぐ**．本反復内で第 3 の埋め込みモデルを探しに行くことはしない
  （Iter80 が `bge-m3` へ差し替えて「イテレーション名と実際の値がずれる」状態を作った失敗を繰り返さない）．

### 事前ゲート G1・G2（いずれも report-only．本走を省略する材料にはしない）

- **G1（CV．B165 (4)「本走前に wafl-ctrl5 の CV で必ず数値化する」への回答）**:
  wafl-ctrl5 で `data/classifier_train_iter94_dedup.jsonl`（2,275 行）**のみ**を使い，
  ruri の 2 ビュー 1536 次元で 5-fold StratifiedKFold（`random_state=42`）の accuracy / macro-F1 を測り，
  現行 `qwen3-embedding:4b` の同条件の値（**0.802637**）と並べて記録する．評価集合は一切見ない．
  **値の選定には使わない**（レバーは確定済み）．p/n が 2.25 → 0.67 へ下がるため，CV と本走の乖離の向きは
  Iter89（CV +5.29pt に対し本走 +2.94pt）とは変わりうる．**本走の予測値として扱わない．**
- **G2（replay．検出力の事前確定）**: 評価 3,750 行について ruri の 2 ビュー埋め込みを wafl-ctrl5 で計算し
  （`data/embcache_eval_ruri-v3-310m{,__p1}.npy` を新規作成），旧 artifact（`2f801357...`，5120 次元）と
  新 artifact（1536 次元）の `predict_proba` argmax を replay する．
  **discordant 行数 n_d を出し，McNemar の有意境界 `1.96·√n_d / 3750` を本走前に確定して記録する**
  （Iter89 は n_d = 347 で境界 1.063pt．今回も特徴空間を丸ごと入れ替えるので n_d は数百規模になる見込み）．
  **n_d ≥ 30 を最低条件**とし，一桁なら「効果なし」ではなく**設定未到達**を既定の解釈とする（d0004 §4）．
  replay が予測した Δ も記録するが，**この値を見て成功条件を書き換えてはならない**（B131 以来の運用）．
- **本走は G0 合格なら必ず実施する**（2026-09-23 絶対条件 (A)）．G1/G2 の数値が悪いことを理由に
  本走を省略しない．**唯一の安全弁**として，G1 の CV Δ が **−5.0pt 以下**（CV の Δ の SE ≒ 0.22pt に対し
  約 23 SE．施策の効果ではなく実装・書式の破綻としか解釈できない水準）の場合に限り，
  本走前に原因究明へ戻る（効果判定のための足切りではなく**不具合検知**である．B166 (b)）．

### 事前登録する予測 P1〜P6（Iter99）

- **P1（主予測）**: 本走で **Δtop1 ≥ +0.5pt かつ McNemar p < 0.05**（＝ `adopted`）．
  点推定 +1.0pt，80% 区間 −2.0 〜 +4.0pt．**符号の確信は低い**と明記しておく．
- **P2**: G1 の CV top1 が現行 **0.802637** を上回る（Δ ≥ 0）．
- **P3**: G2 replay の予測 Δ と本走実測 Δ の乖離が **≤ 0.5pt**（Iter89 は 0.12pt．測定系の再現性検査）．
- **P4**: 新 artifact の `n_features_in_` == **1536**，全 10 ノードで sha256 一致．
- **P5**: G0-d のコサイン類似度の**中央値 ≥ 0.99**（GGUF 経路が参照実装を再現している）．
- **P6**: rank1−rank2 の confidence gap 分布が動き，`gap < 0.36` の行割合が基準線から **±3pt 以上**ずれる
  （Iter82→83 で同種の移動を実測済み）．**これは報告のみで，`dispatch_gap_threshold` は 0.36 に固定する**（B165 (5)）．
  P6 が当たった場合，閾値の再較正を**次反復のレバー候補**として backlog へ回す．

### 成功条件・非退行条件（事前登録 / Iter99）

**基準線（本走）**: `results/20260928_160921/`（3,750 問）．本日 `metrics.py` で再計算した実測値は
top1 = **0.833067**（Wilson 95%CI [0.820791, 0.844660]），`fallback_rate` 0.0，
`dispatch_failure_rate` 0.000267，`mean_duration_ms` **2534.762**，
`compound_domain_top1_accuracy` **0.790411**，`compound_domain_set_recall` **0.567123**，
`compound_mean_dispatched_count` **1.950685**，ECE **0.031774**，Brier 0.110943，
`single_domain_top1_accuracy` 0.843377．**再現性の床は ±0.25pt．**

- **判定**: Δ ≥ +0.5pt（top1 ≥ **0.838067**）かつ McNemar p < 0.05 → **`adopted`**．
  Δ ≥ +0.5pt だが p ≥ 0.05 → `adopted_small`（要再現）．|Δ| < 0.25pt → `negligible`．
  Δ ≤ −0.5pt または下記 C1〜C7 のいずれか違反 → `rejected`（artifact と `config.yaml` をロールバック）．
  G0 不合格で本走に至らなかった場合 → `invalid`（実現性）．
- **必須の非退行条件**（1 つでも破れたら `rejected` としてロールバック）:
  - **C1**: per-domain precision/recall 計 20 指標の BH 補正後（q=0.05）の有意退行が **0 件**．
  - **C2**: `fallback_rate` が 0.0 のまま，`dispatch_failure_rate` ≤ **0.005**．
  - **C3**: レバー発火の証拠（上表 #1〜#6 と G0-a〜e）がすべて記録されていること．
  - **C4**: 複合設問 730 行の top1 が **≥ 0.780411**（基準 0.790411 から −1.0pt 以内）．
  - **C5**: `compound_domain_set_recall` **≥ 0.5400**，`compound_mean_dispatched_count` **≤ 2.10**．
  - **C6**: `mean_duration_ms` ≤ **3041.7**（基準 2534.762 の +20%．ruri は 310M と現行 4B より小さいので
    短縮方向を見込むが，条件は退行側にのみ置く）．
  - **C7**: ECE ≤ **0.08**（基準 0.031774）．
- **参考値として併記**: Random 0.119467 / BestSingle / Oracle 1.0（success_criteria (3)）と
  `answer_quality` / `end_to_end`．

### 実行フェーズへの申し送り（Iter99）

- **使うホスト**: 埋め込み計算・分類器訓練・G0〜G2 はすべて **wafl-ctrl5**（絶対条件 (B)）．
  **wafl500〜509 は deploy と本走（および予備 20 問）でのみ触れる**．
- **触ってよいファイル**: `config.yaml`（`embedding_model` と新キー `embedding_prompt_template` の 2 行のみ）・
  `expert_backend.py`（`embed()` の prefix 組み立てのテンプレート化．既定値は現行と同一挙動）・
  `node.py` / `scripts/train_domain_classifier.py` / `tools/smoke_check.py`（新キーの受け渡し 3 箇所）・
  `models/domain_classifier.joblib`（再訓練．旧 artifact は
  `models/domain_classifier_pre_iter99_qwen3_4b.joblib` へ退避）・
  `data/embcache_*`（新規キャッシュ）・`tests/`（template の単体テスト）．
  **`data/dataset.jsonl`・`data/classifier_train_iter94_dedup.jsonl`・`metrics.py`・`classifier.py`・
  `aggregator.py`・`ecoc_head.py` は 1 バイトも変更しない．**
- **prefix は `トピック: {text}` 一択**．`検索クエリ: ` 等との比較・文言の掃引は**禁止**（単一レバー原則）．
- **`dispatch_gap_threshold` は 0.36 のまま**．gap 分布が動いても再較正しない（B165 (5)）．
- キャッシュ名は**モデル名とビュー ID を必ず含める**こと（Iter81 の教訓．旧キャッシュを無言で読む事故の防止）．
- G0-d の参照実装導入（`sentence-transformers` + `cl-nagoya/ruri-v3-310m`）は wafl-ctrl5 のローカル環境
  （`~/expert-mesh-iter95/` の uv 3.12 環境）に閉じること．本番イメージ（`Dockerfile`）には入れない．

### 実装・実験 (Iter99)

オーケストレータによる記録（rc-executor が journal へ未記入のまま引き渡したため，フェーズ 3 で記録が
失われないようフェーズ境界で補記した）．**実験ノード wafl500〜509 は一切未使用**である．

**G0（実現性ゲート，wafl-ctrl5）: 全項目 PASS．** (a) `ollama pull` 成功（337MB），(b) `/api/embeddings`
200 応答，(c) 次元 **768**，(d) 数値忠実性は sentence-transformers `cl-nagoya/ruri-v3-310m` 参照実装に対し
コサイン**中央値 0.99604・最小 0.97278**（基準 0.99 / 0.97 をいずれも充足），(e) `ollama ps` が 100% GPU．
代替 `multilingual_e5_large` へのフォールバックは不要だった．G0 最大のリスクとしていた
「patched llama.cpp 由来の GGUF が Ollama 0.34.4 で動くか」は，(d) の忠実性をもって解消した．

**G1（CV，wafl-ctrl5）で安全弁が発火し，本走を実施していない．** ruri 2 ビュー 1536 次元の CV top1 =
**0.729670** に対し基準 0.802637，**Δ = −7.297pt**．事前登録した安全弁「CV Δ ≤ −5.0pt は効果ではなく
実装破綻としか解釈できない水準」（B166 (b)）を超過したため，wafl500〜509 での 3,750 問本走（約 2 時間）は
起動していない．**CV を足切りに使わないという絶対条件 (A) との関係**: これは効果量による足切りではなく，
事前登録済みの実装破綻検知としての停止であり，(A) の禁じる「CV による採否判定」には当たらない．

**実装バグの切り分け（バグの兆候は見つかっていない）**:

| 確認項目 | 実測 | 読み |
|---|---|---|
| p0 単体ビュー | 0.702857 | 両ビューが同程度に低い |
| p1 単体ビュー | 0.707253 | 同上 |
| 2 ビュー連結 | 0.729670 | 連結で改善＝ビュー結合は正しく効いている |
| G0-d コサイン忠実性 | 中央値 0.99604 | 埋め込み値そのものは参照実装と一致 |

内部整合性（単体 < 連結）と数値忠実性の双方が成立しており，**実装バグの兆候は特定できなかった**．
rc-executor の所見は H2（768×2 = 1536 次元では 5120 次元に対し容量が不足）寄りだが，
Δ = −7.297pt という大きさが H2 だけで説明できるかは**分析フェーズの判断事項**として引き継ぐ．

**変更したファイル**: `expert_backend.py`（`embed()` に `prompt_template` 引数を追加．既定値は旧 Qwen
instruct 書式で**ビット同一**），`node.py` / `scripts/train_domain_classifier.py` /
`tools/smoke_check.py` / `scripts/embed_classifier_train_pool.py`（`config["embedding_prompt_template"]`
の受け渡し 3 箇所＋診断補助 1 箇所），`tests/test_expert_backend.py`（新規 6 テスト．既定値温存・
template 適用・plain view 不変を検証）．ruff / pytest は新規失敗 0（既存 9 件の FAIL は環境要因の
再現済み既知不具合）．

**本番状態は変更していない**: `config.yaml` は diff 0（元値 `qwen3-embedding:4b` のまま），
`models/domain_classifier.joblib` は旧 5120 次元のままで未再訓練．両者が整合しているため production は
一貫している．埋め込みキャッシュ
`data/embcache_{train,eval}_iter99_ruri-v3-310m.npy`（train 2275×1536，eval 3750×1536）は作成済みで，
分析フェーズが追加計算に再利用できる．G2（replay）は旧 qwen4b の評価キャッシュが 3,435 行で
現行 3,750 行と不一致のため未実施．

**事前登録した予測の当落**: P1・P3〜P6 は本走がないため**未判定**．**P2（CV ≥ 0.802637）は落選**
（0.729670）．

### Iteration 99 実行済み

**判定: `closed`（本走なし）．value `japanese_specialized_ruri_v3_310m` は「単独差し替え」としては
反証済み・再試行しない．ただし対抗仮説 H2（次元容量の不足）は本フェーズの追加計算で反証された．**

#### 変更（本フェーズで production 状態は 1 バイトも変えていない）

実装フェーズの変更は `expert_backend.py`（`embed()` の `prompt_template` 引数．既定値は旧 Qwen instruct
書式とビット同一）・`node.py`・`scripts/train_domain_classifier.py`・`tools/smoke_check.py`・
`scripts/embed_classifier_train_pool.py`・`tests/test_expert_backend.py`（新規 6 テスト）のみ．
`config.yaml` は diff 0，`models/domain_classifier.joblib` は旧 5120 次元のままで production は自己整合．
**この分岐は後方互換で既定値が旧挙動と一致するため，`closed` 判定でもロールバックせず残す**
（次反復の異種埋め込み融合がこの `prompt_template` 機構を必要とするため，撤去はむしろ手戻りになる）．

#### 追加計算（すべて開発ホストの CPU．wafl500〜509 も wafl-ctrl5 も不使用）

既存キャッシュだけで完結させた．`data/embcache_train_iter89_qwen3-embedding_4b{,__p1}.npy` と
`data/embcache_train_iter92_qwen3-embedding_4b_new20{,__p1}.npy` を id 対応で再構成して
`classifier_train_iter94_dedup.jsonl`（2,275 行）に揃えた 5120 次元行列を作り（欠損 0 件），
`data/embcache_train_iter99_ruri-v3-310m.npy`（2,275×1,536，id 順は dedup と完全一致を検証済み）と
**同一の 5-fold StratifiedKFold（`random_state=42`）・同一のモデル定義**（`train_classifier()` と
`_extract_sample_weights()` をそのまま import）で比較した．再実装のため rc-executor の値とは
0.2〜0.8pt ずれるが（qwen 0.804396 対 報告 0.802637，ruri 0.721319 対 報告 0.729670），
**Δ の符号と桁は完全に再現している**（本再実装で Δ = −8.31pt，報告 −7.30pt）．
CV accuracy の SE は n=2,275 で **0.83pt**．

#### 結果 1: H2（1536 次元では容量が足りない）は**反証された**

各 fold の訓練部分だけで PCA を学習し（漏れなし），ビューごとに次元を落として同じ CV を回した容量曲線:

| 特徴空間 | 次元 | CV top1 |
|---|---|---|
| qwen3-4b 2 ビュー（現行） | 5120 | **0.804396** |
| qwen3-4b を PCA | 1536 | **0.804396** |
| qwen3-4b を PCA | 768 | 0.799121 |
| qwen3-4b を PCA | 512 | 0.795165 |
| qwen3-4b を PCA | 256 | 0.788571 |
| qwen3-4b を PCA | 128 | 0.775824 |
| **ruri-v3-310m 2 ビュー** | **1536** | **0.721319** |
| ruri-v3-310m を PCA | 768 | 0.719121 |
| ruri-v3-310m を PCA | 512 | 0.716484 |

**ruri と次元を完全に揃えた qwen（PCA 1536）は，5120 次元と小数点以下まで同値の 0.804396 である．**
それどころか **128 次元まで落とした qwen（0.775824）ですら ruri の 1536 次元（0.721319）を +5.45pt
上回る**．ruri 側も 768→1536 で +0.22pt しか動かず（SE 0.83pt 内＝飽和）．
すなわち **Δ = −8.31pt のうち次元容量で説明できる分は 0pt** であり，
rc-executor の所見（H2 寄り）は棄却される．損失は表現そのものの性質に帰属する．

分散の形も容量不足とは逆を指す．participation ratio による有効次元は **ruri 111.5 > qwen 76.5**，
分散 90% に要する主成分数は **ruri 220 < qwen 481**，第 1 主成分の寄与は ruri 0.0458 < qwen 0.0721．
ruri の空間は潰れていない（むしろ等方的）．**分散はあるがドメイン弁別に効かない向きに使われている．**

#### 結果 2: AGREE（Iter98 の学びに従い既定で併記．追加計算ほぼ 0）

| 量 | 値 |
|---|---|
| qwen の CV 誤答 | 445 / 2,275 |
| ruri の CV 誤答 | 634 / 2,275 |
| 共通の誤答 | **390** |
| 共通 / qwen 誤答 | **0.8764** |
| 共通 / ruri 誤答 | 0.6151 |
| ruri のみ誤り b / qwen のみ誤り c | **244 / 55**（McNemar χ² = 118.2） |

**qwen の誤答の 87.6% を ruri も誤る．**この 87.6% は Iter98 で softmax 対 ECOC を比べたときの
AGREE（87.6%）と一致する．決定則を替えても，**埋め込みモデル系統を丸ごと替えても，同じ行が落ちる．**
しかも ruri が新たに救った行は 55 行しかなく，新たに落とした行が 244 行ある．
**Iter99 の仮説（表現を替えれば最難層 Q1 の一部が分離可能な位置へ移る）は，救済 55 行に対し
新規損失 244 行という形で明確に否定された．**

ドメイン別でも**10 ドメイン中 9 つが悪化**し（legal のみ +1.30pt，n=77 で SE 4.2pt ＝ ノイズ内），
悪化幅は business_economics −12.40pt / social_science −11.20pt / computer_science −9.20pt …と広く分布する．
qwen 側 recall と Δ の Spearman は **+0.285（p=0.425）**で有意でない．
**特定の難層への集中ではなく，全ドメイン一律の地盤沈下である．**

#### 結果 3（次の一手の根拠）: ruri は単独では劣るが，qwen と**相補的**である

同じ CV で qwen 5120 次元と ruri 1536 次元を単純連結（6,656 次元）した:

| 特徴空間 | CV top1 | qwen 単独との discordant |
|---|---|---|
| qwen 5120（現行） | 0.804396 | — |
| **qwen 5120 ⊕ ruri 1536** | **0.812308（+0.79pt）** | 融合のみ誤り 16 / qwen のみ誤り 34，**McNemar 正確検定 p = 0.0153** |

+0.79pt は CV の SE 0.83pt と同程度だが，**対応のある比較（McNemar）では p=0.0153 で有意**であり，
方向も救済 34 対損失 16 と一貫している．**単独では 8pt 劣る表現が，連結すると現行を上回る向きに効く**
＝ ruri は qwen が捉えていない弁別情報を少量持っている．結果 2 の「共通 390 行」は依然として硬いが，
「ruri のみが救える 55 行」のうち一部が融合で回収されている読みと整合する．

#### 判定と根拠

- **`closed`（`rejected` ではない）**．B163 (a)・B165 (a) と同一の理由: `rejected` は本走の 2 軸表
  （Δ と非退行 C1〜C7）に基づく判定語であり，本走が無い以上どのセルにも到達していない．
  **判定語集合の拡張（`refuted` の新設等）は記録スキーマの破壊的変更なので自動決定せず，
  backlog B165 要レビュー (A) の継続として人間判断に委ねる**（これで 3 反復連続の同型事例）．
- **value `japanese_specialized_ruri_v3_310m` は「単独差し替え」としては反証済み・再試行しない．**
  根拠は (i) 次元を揃えても qwen が 8.3pt 勝つ（H2 反証＝次元を増やす方向の再試行に見込みが無い），
  (ii) 実装忠実性は G0-d のコサイン中央値 0.99604 で担保済み（H3 も反証），
  (iii) 悪化が 9/10 ドメインに一様で，prefix 文言や訓練構成の微調整で埋まる幅ではない．
  **停止したのは「効果が小さかったから」ではなく「機序まで説明が付いたから」である．**
- **`multilingual_e5_large`（事前登録済みの代替）は優先度を下げる．** G0 は不合格でなかったので
  そもそも発動条件を満たしておらず，かつ (a) H2 反証により「連結 2048 次元だから有利」という
  当初の期待が消え，(b) JMTEB Classification は E5 72.89 < ruri 78.66 で，その ruri が 8.3pt 負けた以上
  **公称ベンチは本データで予測力を持たない**（Iter79 学び 2 の 3 例目）．値は残すが次の第 1 候補にはしない．

#### 学び（次の自分へ）

1. **「次元が減ったから負けた」は，PCA で次元を揃えた対照を取るまで言ってはいけない．**
   本件では qwen を 1536 次元へ落としても値が小数点以下まで不変で，容量説は完全に外れていた．
   この対照は既存キャッシュだけで CPU 数分．**次に埋め込みを替えるときは必ず最初にこれを置く．**
2. **AGREE は埋め込み系統をまたいでも 87.6% で，Iter98 の決定則間 AGREE と一致した．**
   誤答の硬い核（約 390 行 / 2,275）は，決定則にも表現モデルにも依存しない．
   単一モデルの差し替えで動かせる層ではない．機序 M6 は「入力表現の性質」から
   **M7「誤答核はデータ側（設問の多ドメイン性・ラベルの一意性の破れ）に由来する可能性が高い」**へ
   読み替えるべき段階に来ている（ただし本反復のデータだけでは M7 は未検証の仮説である）．
3. **弱いモデルを「捨てる」判断と「混ぜる」判断は別である．** 単独 CV で 8.3pt 劣る表現が，
   連結すると McNemar p=0.0153 で現行を上回った．単独性能でモデルを足切りすると，この相補性は見えない．
4. **安全弁（CV Δ ≤ −5.0pt で本走前に停止）は今回正しく働いた．** 停止しなければ約 2 時間の実機本走を
   −8pt の設定に費やしていた．一方で安全弁は「実装破綻の検知」として設計されたが，
   実測は**実装破綻ではなく真の性能差**だった．**安全弁の発火は「バグがある」ことを意味しない**ので，
   発火時は必ず次元を揃えた対照と AGREE を取って，破綻か実力差かを切り分けること．
5. ruri の GGUF（patched llama.cpp 変換・q8_0）は Ollama 0.34.4 で**問題なく動いた**
   （参照実装とのコサイン中央値 0.99604・最小 0.97278）．調達経路そのものは今後も使える．

#### 再現用（すべて開発ホスト CPU，リポジトリ外）

`/tmp/iter99a/build.py`（特徴の id 対応再構成）・`cv.py`（基準 CV）・`sweep.py`（PCA 容量曲線）・
`agree.py`（AGREE・ドメイン別・有効次元）・`fuse.py`（異種連結）．
入力は `data/embcache_train_iter{89,92,99}_*` と `data/classifier_train_iter{87_hybrid,94_dedup}.jsonl` のみ．

## Iteration 98: 分類器の多クラス分解を誤り訂正出力符号へ変える

### 調査 (Iter98)

本フェーズでも**実験ノード wafl500〜509 には一切触れていない**（2026-09-23 絶対条件 (B)）．
開発ホスト上で行ったのは，リポジトリのコード読み取りと，**合成データによる sklearn の API 実現性の
確認（CPU のみ・数十秒）**だけである．`models/`・`data/`・`config.yaml`・`results/` への書き込みは 0 件．

**(1) 問い**（B163 (f) の申し送りを受けて設定した）

- Q1: ECOC は Iter97 で退行の機序と読んだ 2 点，すなわち **(a) 対ごと分解の標本分割（p≫n の悪化）**と
  **(b) 非適格分類器の票**を，構造的に回避するか．回避するなら OvO の反証は「決定層の分解一般」ではなく
  「OvO 固有の弱点」に帰属でき，軸を閉じる根拠になる．
- Q2: ECOC 固有の落とし穴（符号長というハイパラ，復号方式，離散性）は何か．Iter97 の P4（gap 分布の
  収縮）と同型の問題を持ち込まないか．
- Q3: 現行の較正済みパイプライン（`CalibratedClassifierCV(method="temperature")` ＋
  `sample_weight = n/(K*n_d)`）と，`classifier.py:estimate_confidence_classifier()` の
  「`predict_proba` が 10 ドメインで和 1」という前提を壊さずに実装できるか．

**(2) 文献調査（tvly search / extract）**

- **Dietterich & Bakiri (1995), *Solving Multiclass Learning Problems via Error-Correcting Output
  Codes*, JAIR 2:263–286** —— ECOC の原典．k クラスに長さ L の符号語を割り当て，各列が
  **クラスを 2 つのメタクラスに分ける二値問題**になる．行間の Hamming 距離が d なら
  ⌊(d−1)/2⌋ ビットまでの二値分類器の誤りを復号で訂正できる．
- **Allwein, Schapire & Singer (2000), *Reducing Multiclass to Binary: A Unifying Approach for
  Margin Classifiers*, JMLR 1:113–141**,
  <https://www.jmlr.org/papers/volume1/allwein00a/allwein00a.pdf> ——
  本反復の設計はこの論文に従う．PDF から直接引用して確認した事実は 3 点．
  (i) **dense random code の長さは ⌈10·log₂(k)⌉**（sparse は ⌈15·log₂(k)⌉）で，
  **10,000 本の候補からから行間最小 Hamming 距離が最大のものを選ぶ**（「examining 10,000 random codes
  and choosing the code that had the largest ... and did not have any identical columns」）．
  **k=10 なら L = ⌈33.22⌉ = 34**．符号長を掃引せず単一値を事前登録せよという B163 (f)(1) の要件に，
  **文献由来の既定値**で答えられる．
  (ii) **loss-based decoding は Hamming decoding より優れる**（「These bounds indicate that
  **loss-based decoding is superior to Hamming decoding**」）．さらに
  「the loss-based decoding method for **log-loss is the well known and widely used
  maximum-likelihood decoding**」とあり，log 損失での復号＝最尤復号である．
  (iii) 基底学習器が SVM のとき「it is clear that **the widely used one-against-all code is inferior
  to all the other codes we tested**」．ただし「there is **no clear winner** among the four other
  output codes」であり，AdaBoost では「none of the codes is persistently better than the other」．
  すなわち**符号の種類による差は保証されていない**．
- **反証側: Rifkin & Klautau (2004), *In Defense of One-Vs-All Classification*, JMLR 5:101–141**,
  <https://www.jmlr.org/papers/volume5/rifkin04a/rifkin04a.pdf> ——
  「a simple one-vs-all scheme is **as accurate as any other approach**, assuming that the underlying
  binary classifiers are **well-tuned regularized classifiers**」．同論文は，異なる分解方式が
  同じ点で同じ誤りを犯す（AGREE 列が誤り率より大きい）ことを示し，
  「points become errors **not because of deficiencies in the method of combining binary
  classifiers**」と述べる．**Iter97 の実測はこの側と整合した**ので，Iter98 も同じ側に落ちる確率は高い．
  本反復を「改良案」ではなく**軸を閉じるための確認実験**と位置づける根拠である（B163 (f)）．
- **Ghani (2000), *Using Error-Correcting Codes for Text Classification*, ICML** および
  Berger (1999), *Error-correcting output coding for text classification*, IJCAI ——
  テキスト分類（高次元疎特徴・多クラス）で ECOC が有効だった先行例．ただし両者の基底学習器は
  Naive Bayes であり，**弱い基底学習器のとき ECOC の利得が大きい**という Rifkin らの説明と矛盾しない．
  本研究の基底学習器は正則化つき線形 LR であり，条件は Ghani らより Rifkin ら側に近い．
- **Escalera, Pujol & Radeva (2010), *On the Decoding Process in Ternary ECOC*, IEEE TPAMI
  32(1):120–134** —— 復号方式の比較（Hamming / Euclidean / loss-based）．本反復は 3 値符号を使わないので
  0 要素の扱いは論点にならない．

**(3) ECOC は Iter97 の 2 つの機序を構造的に回避するか（Q1 への回答）**

| Iter97 で読んだ退行の機序 | OvO | ECOC（dense random, L=34） |
|---|---|---|
| (a) 標本分割（各二値問題が約 500 行しか見ない） | 該当．2,275 → 約 500 行（1/4.5） | **非該当**．各列が**全 2,275 行**を使う |
| (b) 非適格分類器の票（真クラス d を知らない 36/45 本が投票） | 該当 | **非該当**．全列が 10 クラス全部を ±1 に割り当てる |
| P4 のスコア離散性（得票数 0〜9 の階段） | 該当（`gap<0.36` が 20.1%→8.3%） | **非該当**．log 尤度の和は連続量（合成データで 600 行すべて相異なる値を実測） |

したがって ECOC は (d) の読み（B163 (d)）の**直接の対照実験**として機能する．
**ただし ECOC 固有のリスクがある**: 各列は「5 クラス対 5 クラス」のようなメタクラス分割であり，
意味的にまとまらない多峰なクラス塊を 1 本の線形境界で分けることになる．
n=2,275・p=5,120（p≫n）では**任意のラベル付けが訓練データ上ほぼ必ず線形分離可能**なので，
訓練誤差は 0 に近づく一方で汎化しない恐れがある（Dietterich & Bakiri が要求する「基底学習器が
任意の二分割を学習できること」は満たすが，「汎化できること」は別問題である）．
この懸念は**列ごとの二値 CV 正解率**として直接測れるので，診断出力に含める（下記 P5）．

**(4) 実現性の実測（開発ホスト，CPU のみ，合成データ．Iter97 の落とし穴に相当するものを事前に潰した）**

- **最大の落とし穴（実測）**: **`sklearn.multiclass.OutputCodeClassifier`（sklearn 1.9.0）は
  `predict` しか持たない**（`predict_proba` も `decision_function` も無い）．そのため
  `CalibratedClassifierCV(OutputCodeClassifier(...))` は
  `InvalidParameterError: The 'estimator' parameter of CalibratedClassifierCV must be an object
  implementing 'fit' and 'predict_proba', an object implementing 'fit' and 'decision_function' or
  None` で**そもそも fit できない**．すなわち Iter97 のような「base estimator を差し替えるだけ」の
  変更は**不可能**である．`classifier.py` が要求する「`predict_proba` が 10 ドメインで和 1」を
  満たすには，**復号スコアを返す薄いラッパを自前で書く**必要がある．
- そこで **`EcocLogLossClassifier`**（dense random code ＋ **log 損失による loss-based decoding
  ＝最尤復号**）を約 60 行で試作した．`decision_function(X)[i, d] = Σ_j log σ(M[d,j]·f_j(x_i))`．
  これを既存の `CalibratedClassifierCV(method="temperature", cv=5, ensemble=True)` にそのまま渡せる．
  実測（合成データ）:
  - `CalibratedClassifierCV(EcocLogLossClassifier(...), method="temperature", cv=5, ensemble=True)`
    は fit でき，**`predict_proba` の行和は全行 1.0**（`classifier.py` の前提を満たす）．
  - **`sample_weight` は警告 0 件で 34 本すべての二値 LR へ到達する**．`fit(self, X, y, sample_weight=None)`
    と明示的な引数を持たせたので，sklearn の `has_fit_parameter` 判定が通り，
    Iter97 で必要だった `enable_metadata_routing` は**不要**である．到達の実証として，
    クラス 0 の重みを 20 倍にした場合と等重みの場合で二値 LR の係数が
    `max|Δcoef| = 0.0935`（較正ラッパ経由でも 0.0738）と**有意に動く**ことを確認した
    （0 なら未到達．Iter97 の学び 2 に対応する機械的な検査）．
  - 符号行列は L=34，**行間最小 Hamming 距離 14**（⌊13/2⌋=6 ビットまで訂正可能），重複列 0．
  - **スコアは連続**（合成 600 行で上位 2 位差の相異なる値が 600/600）．Iter97 の P4（離散性による
    gap 収縮）は原理的に起こらない．
  - 計算量（n=1,820 × p=5,120 の合成データ）: 単一 softmax 0.20 秒，5 対 5 の二値 1 本 0.07 秒，
    ECOC 34 本 **1.01 秒**，temperature 較正込み（cv=5）**5.1 秒**．
    3 seed × 5-fold でも **wafl-ctrl5 の CPU で数分〜十数分**に収まる（再埋め込み 0 回・GPU 不要）．
- **デプロイ側の到達条件（足切り通過時のみ問題になる）**: joblib はクラスを**モジュールパス参照**で
  直列化するため，`EcocLogLossClassifier` を `scripts/train_domain_classifier.py` の中で定義して
  スクリプト実行すると `__module__` が `__main__` になり，ノード側で**復元できない**．
  そのため実装時は**リポジトリ直下に `ecoc_head.py` を新設**して同クラスを置き，
  `scripts/train_domain_classifier.py` から import し，`Dockerfile` の `COPY` 行（現行 L14）へ
  `ecoc_head.py` を追加する．スクリーニング段階では直列化しないのでこの作業は不要である．

### 仮説 (Iter98)

**Iter97 の OvO の退行（Δ=−0.806pt）は「決定層を分解したこと」自体ではなく，OvO 固有の 2 点
（各二値問題が全体の約 1/4.5 の行しか見ない標本分割，および真クラスを知らない 36/45 本の票）に
由来する．** 全行を使い非適格分類器を持たない ECOC（dense random code, L=34, 最尤復号）に替えれば，
少なくとも**近傍クラス（social_science・history_culture）の有意退行は再現しない**はずである．
そのうえで CV top1 が上がるかどうかは両論あり（Allwein et al. は OvA 劣位を報告，
Rifkin & Klautau は同等と主張），**Iter97 と同じく両論のまま検定する**．
対抗仮説は「5,120 次元・2,275 行で任意の 5 対 5 分割を線形に学習させても汎化しない」であり，
これが正しければ列ごとの二値 CV 正解率が低く（偶然水準に近く），ECOC も退行する．

### 単一レバー (Iter98)

- **レバー**: `classifier_multiclass_decomposition` = **`error_correcting_output_codes`**
  （config.yml に事前登録済みの未試行 value．新レバーの追加は不要）
- **何を何から何へ**: `scripts/train_domain_classifier.py:train_classifier()` の **L210**
  `base_estimator = LogisticRegression(max_iter=_MAX_ITER, class_weight=None)` を，
  **`EcocLogLossClassifier(estimator=LogisticRegression(max_iter=_MAX_ITER, class_weight=None),
  code_length=34, random_state=98)`**（新モジュール `ecoc_head.py`）へ差し替える．
  L211-214 の `CalibratedClassifierCV(base_estimator, method=_CALIBRATION_METHOD, cv=cv,
  ensemble=True)` と `fit(..., sample_weight=sample_weight)` は**一字も変えない**．
- **事前登録する設計値（掃引しない．B163 (f)(1)）**:
  - **符号長 L = 34 = ⌈10·log₂(10)⌉**（Allwein et al. 2000 の dense random code の既定）．
  - **符号行列**: 候補 10,000 本から**行間最小 Hamming 距離が最大**のものを選ぶ（同論文の手続き）．
    定数列・重複行は排除する．`random_state = 98`（イテレーション番号．固定）．
  - **復号**: log 損失の loss-based decoding（＝最尤復号）．Hamming 復号は使わない（同論文の
    理論的優位に従う）．**代替値の比較は一切行わない**（行えば単一レバー原則が濁る）．
- **レバーを読むコード行と到達条件（d0004 §4 の恒久対策）**:
  - スクリーニング段階: `scripts/screen_classifier_multiclass_decomposition.py` に**腕 C** を追加する
    （`_train_ecoc_classifier()`）．腕 A は既存の `train_classifier()` を無変更で import して呼び，
    **CV top1 = 0.802637 の再現**をもって測定系の健全性を確認する（Iter97 と同じ手順）．
  - 発火の証拠（1 つでも欠けたら実験不成立として中断）:
    (a) 学習済み腕 C の `code_book_.shape == (10, 34)`，(b) 行間最小 Hamming 距離 ≥ 10 を記録，
    (c) `sample_weight` 到達の実証（等重み版との `max|Δcoef| > 0`），
    (d) `fit` 中の警告 0 件（`warnings.catch_warnings(record=True)` ガード．Iter97 の実装を流用），
    (e) `predict_proba` の行和が 1.0 ± 1e-6．
  - 本走段階（足切り通過時のみ）: `models/domain_classifier.joblib` の差し替えとデプロイが唯一の
    到達条件である．旧 artifact を `models/domain_classifier_pre_iter98_ecoc.joblib` へ退避し，
    新 artifact の sha256 が旧と異なること・`ecoc_head` がコンテナ内で import 可能なこと
    （`docker compose exec ... python -c "import ecoc_head"`）・各ノードの joblib の sha256 一致・
    `predict_proba` の行和 1.0 を確認する．
- **固定する構成（1 つも動かさない）**: 訓練集合 `data/classifier_train_iter94_dedup.jsonl`（2,275 行）・
  `sample_weight = n/(K*n_d)`・較正 `temperature`（`cv=5, ensemble=True`）・`C=1.0`・`max_iter=1000`・
  埋め込みモデル `qwen3-embedding:4b`・`embedding_instruction`（Iter81 の P1 文言）・
  `embedding_view_concat: true`（5,120 次元）・評価集合 `data/dataset.jsonl`（3,750 行，読むだけ）・
  `dispatch_gap_threshold: 0.36`・`dispatch_gap_max_k: 4`・`dispatch_top_k: 2`・
  `aggregation_method: max_confidence`・expert/light モデルとノード割り当て．

### 事前スクリーニング設計（事前登録 / Iter98）

Iter95〜97 の型をそのまま踏襲する．実施場所は **wafl-ctrl5**（絶対条件 (B)．`~/expert-mesh-iter95/` の
uv 3.12 環境と `data/embcache_iter96_scaled.npy` の先頭 2,275 行を流用．**新規の埋め込み計算は 0 回**）．
**足切りを通った場合は本走を省略しない**（2026-09-23 絶対条件 (A)）．

- **主評価**: 既存 2,275 行の層化 5-fold（`StratifiedKFold`，seed = 96 / 196 / 296，併合 n = 6,825）．
  テスト fold は両腕で完全に同一．**腕 A = 現行 softmax**（CV top1 = 0.802637 の再現を確認），
  **腕 C = ECOC（L=34, 最尤復号）**．腕 B（OvO）は再実行しない（反証済み．B163 (a)）が，
  Iter97 の実測値を表に併記して 3 方式を比較できるようにする．
- **統計**: `metrics.py` の `compute_mcnemar_test` / `compute_domain_recall_mcnemar_test` /
  `apply_benjamini_hochberg` を流用．**seed 併合値を主とし，discordant を seed 数で割った保守値を
  必ず併記する**（B163 (b)・Iter97 学び 4）．
- **最初から実装する診断出力（B163 (f)(4)．事後の再実行を避ける）**:
  1. 両腕の **10×10 混同行列**．
  2. **34 列それぞれの二値 CV 正解率**（メタクラス分割が学習できているかの直接の測定）と，
     符号行列の行間最小／平均 Hamming 距離．
  3. 較正後の **ECE・Brier**（両腕）と，**`gap ≥ 0.36` バケットの正解率**（Iter97 学び 5 への対応）．
  4. **gap 分布**と `gap < 0.36` の行割合．
  5. 腕 C の**復号スコアの上位 1 位と 2 位の差**の分布（OvO の票差に対応する量．離散性の確認）．

### 反証可能な予測（事前登録 / Iter98）

- **P1（足切りと同値・主予測）**: CV top1 Δ (C−A) **≥ +1.0pt**．
- **P2（機序の切り分け．本反復の中心）**: Iter97 で BH 有意退行だった 2 クラスの recall Δ の合計が
  **≥ −1.0pt**（Iter97 の OvO は social_science −4.53 ＋ history_culture −2.86 = **−7.39pt**）．
  - P2 が通り P1 が外れる → 退行は **OvO 固有**だったが，分解方式では精度は動かない．
    **決定層の軸は打ち止め**にしてよい（B163 (g)(h)）．
  - P2 も外れる（ECOC でも同じ 2 クラスが大きく退行する） → 標本分割・非適格分類器という
    Iter97 の機序の読み（B163 (d)）は**誤り**であり，「10 クラスを二値へ分解すること自体」が
    この特徴空間で損をしている，という別の読みに差し替える必要がある．
- **P3（確率の健全性）**: `predict_proba` の行和が全行 1.0 ± 1e-6 で，かつ `gap < 0.36` の行割合が
  腕 A（20.1%）から **±5pt 以内**（Iter97 の OvO は −11.8pt 逸脱）．連続スコアなので通る見込み．
- **P4（較正の健全性）**: 腕 C の ECE が腕 A の ECE を**上回らない**．
  外れる場合は「鋭さの改善」を性能の証拠として読まない（Iter97 学び 5）．
- **P5（対抗仮説の直接検定）**: 34 列の二値 CV 正解率の**中央値が 0.70 以上**．
  0.6 を下回るなら「p≫n でメタクラス分割は分離できても汎化しない」が支持され，
  ECOC の失敗はこれで説明される（この場合，符号長を伸ばしても救えない）．

### 成功条件・非退行条件（事前登録 / Iter98）

**基準線（本走）**: `results/20260928_160921/` の top1 = **0.833067**（3,750 問，
Wilson 95%CI [0.8208, 0.8447]）．再現性の床は ±0.25pt．
**基準線（CV）**: 腕 A = 0.802637．CV の Δ の SE ≒ **0.22pt**（Iter95/96 から見積り．
足切り +1.0pt は約 4.5 SE）．

- **スクリーニングの足切り（両方を満たさなければ本走せず `closed`．Iter97 と同一水準に据え置く）**:
  1. **CV top1 Δ (C−A) ≥ +1.0pt**．
  2. **per-domain recall の BH 補正後（q=0.05，10 指標）有意な退行が 0 件**
     （seed 併合値を主，保守値を併記）．
  - 外した場合は `models/`・`config.yaml`・`data/dataset.jsonl`・`scripts/train_domain_classifier.py` を
    **1 バイトも変更せず**，スクリーニングスクリプトと結果 JSON だけを残してフェーズ 3 へ渡す．
- **本走（足切り通過時のみ）**: `models/domain_classifier.joblib` を ECOC 版で再訓練し，
  `mise run deploy` と sha256 確認のうえ **wafl500〜509 で 3,750 問フルスペック**を 1 回実行する．
- **本走の判定**: Δ ≥ +0.5pt かつ McNemar p < 0.05 で `adopted`，Δ ≥ +0.5pt だが p ≥ 0.05 なら
  `adopted_small`（要再現），|Δ| < 0.25pt なら `negligible`，Δ ≤ −0.5pt なら `rejected`．
- **必須の非退行条件（1 つでも破れたら artifact をロールバック）**:
  - C1: per-domain precision/recall 計 20 指標の BH 補正後の有意退行が **0 件**．
  - C2: `used_fallback` 率・`dispatch_failed` 件数が基準線から増えない．
  - C3: レバー発火の証拠（上記 (a)〜(e) と，ノード上の joblib の sha256 一致・`ecoc_head` の import 可否）．
  - C4（参考値）: Random / BestSingle / Oracle を併記し BestSingle 超過を明示（success_criteria (3)）．
  - C5: 複合設問 730 行の top1（基準 0.7904）が −1.0pt を超えて下がらないこと．
  - C6: `mean_duration_ms` が基準線の +20% 以内（推論は `predict_proba` 1 回のままで実質不変のはず）．

### 実行フェーズへの申し送り（Iter98）

- **`OutputCodeClassifier` をそのまま使ってはならない**（`predict_proba` も `decision_function` も
  持たず `CalibratedClassifierCV` に渡せない）．`ecoc_head.py` の `EcocLogLossClassifier` を使うこと．
  試作は `/tmp/ecoc_probe.py` にあるが，本実装は責務コメント・docstring・型注釈を付けて書き直すこと．
- **符号長 34・最尤復号・`random_state=98` は事前登録値であり掃引禁止**．
  「符号長を変えたら上がるかもしれない」という誘惑に乗ると単一レバー原則が壊れる（B163 (f)(1)）．
- **`sample_weight` 到達の機械的な検査を実装に埋め込むこと**（警告ガード＋`max|Δcoef| > 0` の実証）．
  Iter97 の学び 2 の型を踏襲する．
- 診断出力（混同行列・列ごと二値 CV 正解率・ECE/Brier・gap 分布・復号スコア差分布）は
  **最初から**出すこと．足切りを外した場合でも機序が読めるようにするためである．
- `dispatch_gap_threshold` の再較正は別レバーであり，本反復では**絶対に触らない**．
- 実機ノード wafl500〜509 は**本走時のみ**．スクリーニングは wafl-ctrl5 で完結する．

### 実装・実験 (Iter98)

**結論: 事前スクリーニングの足切り 2 条件をいずれも不通過．本走は行っていない．**
実機ノード wafl500〜509 には一切触れておらず，`models/`・`data/dataset.jsonl`・`config.yaml`・
`scripts/train_domain_classifier.py`・`Dockerfile` は 1 バイトも変更していない
（`git diff --stat` で実測確認済み）．

**(1) 実装**

- **`ecoc_head.py`（リポジトリ直下に新規）**: `EcocLogLossClassifier`．dense random code，
  L = ⌈10·log₂(K)⌉ = 34，候補 10,000 本から行間最小 Hamming 距離最大を選択，
  `decision_function` は log 損失の loss-based decoding（＝最尤復号）．リポジトリ直下に置いたのは
  joblib がクラスをモジュールパス参照で直列化するためである（`scripts/` 配下や `__main__` で
  定義するとノード側で復元できない）．**足切り不通過のため本番経路へは配線していない**．
- **`scripts/screen_classifier_multiclass_decomposition.py`（既存 298 行を書き直し）**:
  腕 A（既存 softmax，import は無変更）と腕 C（`_train_ecoc_classifier()`）の 3 seed × 固定 5-fold 比較．
  腕 B（OvO）は再実行せず Iter97 実測値を `_ITER97_ARM_B_RECORDED` 定数として表に併記した．
  診断出力（10×10 混同行列・34 列の二値 CV 正解率・符号行列の Hamming 距離・ECE/Brier・gap 分布・
  較正前の生復号スコアの上位 1-2 位差分布）を最初から実装している（B163 (f)(4)）．
  保守側 McNemar は discordant を seed 数 3 で割ったうえで `metrics._mcnemar_from_correctness` の
  既存 χ²/p 式を再利用して算出した（χ² 式を自前で再導出していない点が重要．B163 (b)）．
- **`results/iter98_screening/screening_result.json`（新規）**: 生の実行結果全件．
- 検証: `ruff check` 全通過，`py_compile` OK，`pytest tests/test_metrics.py` 53 passed
  （`metrics.py` は無変更で回帰なし）．`pytest tests/test_train_domain_classifier.py` は
  4 passed / 1 failed だが，失敗は `ModuleNotFoundError: sentence_transformers` であり
  開発ホストの venv に当該パッケージが無いという環境要因．`train_domain_classifier.py` の diff は 0 で
  本反復の変更とは無関係である．
- 合成データによる単体検証: `predict_proba` 行和 1.0，符号行列 (10, 34)・最小行間距離 14，
  `sample_weight` 到達（等重み比 `max|Δcoef| = 0.123`，警告 0 件）．

**(2) 事前スクリーニングの結果**（wafl-ctrl5，CPU のみ，n = 2,275，3 seed × 固定 5-fold，
併合 n = 6,825，所要 約 48 分）

| 腕 | 分解方式 | CV top1 |
|---|---|---|
| A | 単一 softmax（現行） | **0.802637** |
| B | OvO（Iter97 実測の併記．本反復では再実行せず） | 0.794579 |
| C | ECOC（L=34，最尤復号） | **0.788425** |

- **CV Δ (C−A) = −1.421pt** → 足切り 1（≥ +1.0pt）に**不通過．符号も逆**である．
- 全体 McNemar 併合値: discordant 179/82，χ² = 35.31，**p = 2.81e-9**．
  保守値（discordant ÷ 3）: 60/27，χ² = 11.77，**p = 0.00060**．
  **併合・保守のいずれでも有意に悪化**しており，B163 (b) で懸念した判定の反転は起きていない．
- per-domain recall Δ（併合 p・BH 判定 / 保守 p・保守 BH 判定）:
  - education: **−7.00pt**，p=1.23e-8 有意 / 保守 p=0.00137 有意
  - social_science: **−6.40pt**，p=1.60e-10 有意 / 保守 p=4.07e-4 有意
  - general: −2.67pt，p=5.10e-5 有意 / 保守 p=0.0233 非有意
  - business_economics: **+1.87pt**，p=0.0108 有意（改善方向） / 保守 p=0.182 非有意
  - history_culture: −0.16pt，p=1.0 非有意（**Iter97 で BH 有意退行だった域が今回は動いていない**）
  - legal・medical・mathematics・computer_science・natural_science: |Δ| ≤ 1.3pt，いずれも非有意
- **BH 後の有意退行 = 併合 3 件（education・general・social_science）／保守 2 件（education・
  social_science）** → 足切り 2（0 件）に**不通過**．

**(3) 事前登録した予測 P1〜P5 の当落**

- **P1（CV Δ ≥ +1.0pt）: 落選**（−1.421pt）．
- **P2（Iter97 で BH 有意退行だった 2 クラスの recall Δ 合計 ≥ −1.0pt）: 落選**（合計 −6.559pt）．
  内訳が重要で，**history_culture は −0.16pt でほぼ不変＝Iter97 の当該退行は再現しなかった**一方，
  **social_science は −6.40pt と Iter97 の −4.53pt よりさらに悪化**した．
  事前登録どおり「P2 も外れる」ケースに該当し，B163 (d) の機序の読み（標本分割・非適格分類器）は
  差し替えを要する．
- **P3（`predict_proba` 行和 1.0 かつ gap<0.36 割合が腕 A ±5pt 以内）: 当選**
  （行和 1.0±1e-6，gap<0.36 割合 Δ = +0.396pt）．Iter97 の P4（gap の人為的収縮）は ECOC では
  予告どおり起きておらず，復号スコアの連続性という構造的差異は実測で確認された．
- **P4（腕 C の ECE が腕 A を上回らない）: 当選**（ECE 0.01650 ≤ 0.01971）．
  Brier は腕 C がわずかに悪い（0.12796 vs 0.12337）が，P4 の判定基準は事前登録どおり ECE のみ．
- **P5（34 列の二値 CV 正解率の中央値 ≥ 0.70）: 当選＝対抗仮説は棄却**
  （中央値 0.893，最小 0.845）．**「5 対 5 のメタクラス分割は p≫n で訓練上分離できても汎化しない」
  という対抗仮説は成り立たない**．各列の二値問題は十分に汎化しているのに，それらを最尤復号で
  束ねると全体の top1 が下がる．

**(4) 判定**

足切り 1・2 とも不通過のため**本走（wafl500〜509，3,750 問）は実施していない**．
判定語の付与と機序の解釈は分析・考察フェーズの担当とする．
なお wafl-ctrl5 の `~/expert-mesh-iter95/` に `ecoc_head.py` と更新版スクリーニングスクリプトを
配置済みで，再現実行が必要ならそのまま使える．

### Iteration 98 実行済み —— 判定 `closed`（足切り不通過・本走なし．value は「反証済み」）

**変更（このイテレーションで実際に触ったもの）**

- 新規: `ecoc_head.py`（`EcocLogLossClassifier`，本番経路へは**未配線**），
  `scripts/screen_classifier_multiclass_decomposition.py`（書き直し．腕 A/腕 C の 3 seed × 固定 5-fold），
  `results/iter98_screening/screening_result.json`（生データ全件）．
- **無変更**: `models/`・`data/dataset.jsonl`・`config.yaml`・`scripts/train_domain_classifier.py`・
  `Dockerfile`・実機ノード wafl500〜509（`git diff --stat` で実測確認済み）．

**結果（要点のみ．詳細は上記「実装・実験 (Iter98)」）**

- CV top1: 腕 A 0.802637（Iter95/96/97 と厳密一致＝測定系は健全）／腕 C 0.788425，**Δ = −1.421pt**．
  Δ の SE ≒ 0.22pt に対し **6.5 SE 相当**で，符号も足切りと逆．全体 McNemar は併合 p=2.81e-9，
  保守 p=0.00060 で**いずれも有意に悪化**．ノイズではない．
- BH 後の有意退行は併合 3 件／保守 2 件（education −7.00pt，social_science −6.40pt，general −2.67pt）．
- 予測の当落: P1 落選・P2 落選・P3 当選・P4 当選・**P5 当選（＝対抗仮説の棄却）**．

**判定: `closed`（`rejected` ではない）．ただし value `error_correcting_output_codes` は「反証済み」**

- `rejected` は本走の 2 軸表（Δtop1 × McNemar p）に基づく判定語であり，本走をしていない今回は
  どのセルにも到達していない（B159 (A)・B161 (a)・B163 (a) と同一論点．判定語集合の拡張は
  記録スキーマの破壊的変更なので自動決定しない）．
- 一方で value の扱いは Iter95/96 型（「Δ は正だが足切りに届かない＝未通過・再試行の余地あり」）ではなく，
  **Iter97 型（符号ごと逆・保守側でも有意＝反証済み，再試行しない）**とする．
  さらに今回は下記 M6 により**失敗の機序まで説明が付いている**ので，符号長 L や復号方式を変えた
  再試行にも見込みが無い（M6 の予測 (3)）．

**機序の再定式化 M6: 「分解の失敗」ではなく「分解で新しい情報が増えない」**
（B163 (d) の読み＝標本分割・非適格分類器は **P2 落選により棄却**．以下で差し替える）

事前登録した診断出力だけで 3 手順の検証が完結した（開発ホスト，CPU 数秒．実機不使用）．

1. **独立誤りの仮定を置いた場合の予測値と実測の乖離**．34 列の二値 CV 正解率（平均 0.8903）の誤り率で
   **ビット誤りが列間で独立**に起きるとした Monte Carlo（20 万行，`random_state=98` の符号行列を
   再生成．行間最小 Hamming 距離 14 を再現）では **top1 = 0.9997**，1 行あたり平均ビット誤り 3.73 で
   訂正能力 6 ビットに十分収まる．**実測は 0.7884**．
   → 予測どおりなら ECOC は圧勝するはずで，**列間のビット誤りは独立でない**．
2. **ビット誤りは誤答行に全量集中している（恒等式による検証）**．
   「誤答行では観測符号語が競合クラスの符号語へ**丸ごと**移り，正答行のビット誤りは 0」と仮定すると，
   期待される列誤り率 = (ECOC の誤答率 0.21158) × (平均行間 Hamming 距離 17.378 / 34) = **0.1081**．
   実測の平均列誤り率は **0.1097**（乖離 1.5% 相対，1 行あたり 0.055 ビット分）．
   → 34 本の二値分類器は「独立にときどき間違える弱学習器」ではなく，**同じ行で足並みを揃えて
   丸ごと隣のクラスへ倒れている**．符号の冗長性（6 ビットまで訂正）は名目値で，実際に起きる誤りは
   7 ビット以上（典型 17 ビット）の符号語置換なのでこの符号長では原理的に訂正できない．
3. **腕 A と腕 C の誤答行は 9 割方同じ行である（AGREE の実測）**．併合 n=6,825 で腕 A の誤答 1,347 行・
   腕 C の誤答 1,444 行，共通 1,265 行．**腕 C の誤答の 87.6%・腕 A の誤答の 93.9% が共通**．
   → Rifkin & Klautau (2004) の「points become errors **not because of deficiencies in the method of
   combining binary classifiers**」（AGREE が誤り率を上回る）が本データで**そのまま再現**した．

**M6 の言明**: 5,120 次元の埋め込み上では，誤りは**行（＝入力表現）の性質**であって決定則の性質ではない．
どの分解方式も同じ行で同じ向きに倒れるため，**符号の冗長性に注入できる独立な情報が存在しない**．
その状態で分解すると，softmax が学習していた**クラスごとの重み付き読み出し**が，34 列を等重みで足す
**固定の読み出し**に置き換わる分だけ情報が減る．これが Δ = −1.42pt の中身である．

**M6 の系（既存データでの裏づけ）**

- 損失は**事後分布が平坦なクラス**に集中する．腕 A の recall と Δ の Spearman = **+0.596（p=0.069，n=10）**
  で，高 recall 側（mathematics 0.948 → +0.40pt，computer_science 0.920 → +0.40pt）は動かず，
  低 recall 側（education 0.535 → −7.00pt，social_science 0.788 → −6.40pt，general 0.799 → −2.67pt）が
  落ちる．正側合計 +3.97pt に対し負側合計 −16.36pt で**正味の損失**（零和の再配分ではない）．
- 失った質量の**行き先が意味的隣接クラスでない**．education −50 行の行き先は
  business_economics +17・legal +17・history_culture +6・natural_science +6 と分散し，
  social_science −48 行は legal +18・business_economics +17・general +9・**computer_science +8** と
  分散する．意味的近傍への流出（Iter97 で読んだ像）ではなく，**平坦な事後分布を任意の固定読み出しで
  割り振った結果の拡散**である．
- P4（ECE 0.01650 ≤ 0.01971）が当選しているのに top1 が下がる点も M6 と整合する．復号スコアは連続で
  較正も壊れていない（P3 も当選）．**壊れているのは確率の形ではなく，どの行をどのクラスへ割るかという
  情報そのもの**である．

**M6 が生む反証可能な予測（今後この軸を再訪するなら，これらを外した証拠が必要）**

1. `random_state` を変えて別の dense random code を引くと，**退行するドメインの顔ぶれは入れ替わるが
   全体 Δ は −1.4pt 前後のまま**（本反復で退行した education/social_science と，Iter97 の OvO で
   退行した history_culture が入れ替わった事実の一般化）．
2. 符号長を L=34 → 68 に伸ばしても **top1 の変化は ±0.3pt 以内**（ビット誤りが独立でない以上，
   冗長性を足しても訂正できる誤りが増えないため）．
3. OvA・OvO・ECOC のどれを使っても，softmax との誤答行の AGREE は **85% 以上**のまま．
   （以上は**実行しない**．実行する価値が無いことを示すのが M6 の役割である．）

**P2 の内訳の非対称性（history_culture 復帰・education/social_science 悪化）の読み**

Iter97 と Iter98 で退行するドメインが入れ替わったこと自体が M6 の予測 1 の直接の観測である．
OvO では 45 対の組み方，ECOC では 34 列の符号がそれぞれ「平坦な事後分布の行をどこへ倒すか」を決める
**任意の固定読み出し**であり，どのクラスが割を食うかは分解の実装詳細に依存する．
したがって「history_culture の退行が ECOC で解消した」ことを**改善の証拠として読んではならない**
（同時に education が新規に −7.00pt 落ちている）．**共通しているのは「低 recall クラスが損をする」
という一点だけ**であり，これは M6 の系そのものである．

**決定層という軸の総括: 打ち止め（closed）**

- 事前登録では「P2 通過かつ P1 不通過なら打ち止め」としていたが，実際は **P2 も落選**した．
  事前登録の分岐は「軸を閉じる唯一の経路は『退行は OvO 固有だった』と示すこと」という前提で書かれており，
  この前提が誤っていた．**実測はより強い経路で軸を閉じる**: 分解方式に依らず誤答行が共通（AGREE 87.6%）で，
  ビット誤りに独立成分が無い（検証 1・2）以上，**どの分解方式を持ってきても softmax を上回れない**．
  「OvO 固有か否か」は，軸を閉じる根拠として不要になった．
- Rifkin & Klautau (2004) の主張（基底学習器が正則化されていれば分解方式は等価）との整合:
  **本データは同論文の側に 2 反復連続で落ちた**．ただし厳密には「等価」ではなく **softmax が 0.8〜1.4pt
  優る**（OvO −0.806pt，ECOC −1.421pt）．同論文の主張は「OvA が他に劣らない」であって「分解が単一
  多項モデルに劣らない」ではないので，矛盾ではない．本研究の条件（p=5,120 ≫ n=2,275，L2 正則化 C=1.0）
  では，単一の多項 softmax が持つ**クラス間で結合した正規化**が，分解では復元できない情報を担っている，
  と読むのが実測に最も忠実である．
- 以上より **`classifier_multiclass_decomposition` は closed**．同レバーの残 value は無く，再訪もしない．

**学び**

1. **「部分問題が解けている」ことは「分解が有効である」ことの証拠にならない**．P5（列ごと二値 CV 正解率
   中央値 0.893）は事前には ECOC 成功の前提条件として登録したが，実際には**分解が無意味であることの
   証拠**だった．二値精度 0.89 は，softmax の top1 0.803 から機械的に導かれる値（誤答 19.7% のうち
   競合クラスが分割の反対側に来る確率が約半分 → 約 0.90）とほぼ一致する．つまり 34 本の二値分類器は
   新しい識別面を 1 つも作っておらず，**同じ 10 値決定を 34 ビットに再符号化していただけ**である．
   今後「分解して部分問題の精度を見る」型の診断では，**部分問題精度の期待値を基準線から先に計算して
   おく**こと（これをやっていれば計画時点で ECOC の見込みの薄さを数値で示せた）．
2. **誤り訂正符号の訂正能力は，誤りの独立性の仮定の上でしか意味を持たない**．最小 Hamming 距離 14・
   6 ビット訂正という設計値は，本データでは一度も働いていない（実際の誤りは典型 17 ビットの符号語置換）．
   ECOC を検討する場面では，**符号長や最小距離ではなく「列間の誤り相関」を先に測る**．測り方は本反復の
   検証 2 の恒等式（列誤り率 ≒ 行誤り率 × 平均行間 Hamming 距離 / L）で，**追加実験なしに既存の
   診断出力だけで判定できる**．この恒等式が成り立ってしまったら訂正能力は幻である．
3. **AGREE（腕間の誤答行の重なり）は，軸を閉じる/開くの判断に直接使える指標である**．今回 87.6% /
   93.9% という値が，「決定層をこれ以上いじっても無駄で，残る説明変数は入力表現である」という結論を
   1 つの数字で支えた．**McNemar の discordant 数から追加計算 0 で求まる**（共通誤答 = 腕 C の誤答数
   − discordant_a_only）ので，今後は腕比較のたびに既定で併記する．
4. （運用）事前登録した分岐（「P2 通過かつ P1 不通過なら打ち止め」）が，実測が示した経路と食い違うことが
   起こりうる．そのときは**事前登録の分岐に機械的に従うのではなく，前提が崩れた事実を明記したうえで
   実測に即した結論を書く**．事前登録は結論を縛るためではなく，事後の都合の良い解釈を防ぐためにある．

**次イテレーション（Iter99）の方針**

`classifier_multiclass_decomposition` が closed となり，config.yml の既存 levers に未試行 value は無い．
M6 の結論（誤りは決定則ではなく**入力表現**の性質）と，B115(3) のユーザー指示による優先順位
（複合評価集合の拡充＝Iter78 で完了・730 行へ拡大済み ＞ **埋め込みモデルの差し替え** ＞ 全ドメイン共通
ルールでの訓練データ拡充）が同じ方向を指す．そこで `embedding_model_replacement` に新 value
**`japanese_specialized_ruri_v3_310m`** を追記し，Iter99 のレバーとする（B165）．
`iteration_name` = **「埋め込みを日本語特化モデル ruri-v3-310m へ差し替える」**．
根拠は Iter79 の調査で記録済みの JMTEB Classification（`cl-nagoya/ruri-v3-310m` **78.66** vs
`Qwen3-Embedding-0.6B` 66.09，出典 hotchpotch 2025-06-11）．当時は「instruction prefix が必須で
2 レバー目になる」ため候補外としたが，**Iter81 で prefix 機構が，Iter82 で 2 view 連結が既に本番に
入っている**ので，その除外理由は現在では成立しない．
**必須の申し送り**: (i) 現行の埋め込みは各ノードの Ollama `/api/embeddings` 経由（`config.yaml:4`）であり，
ruri-v3-310m が Ollama で取得可能かは未確認である．Iter79/89 と同型の **G0（実現性ゲート）を計画段階に
必ず置く**こと．(ii) G0 不合格なら事前登録した代替 `multilingual_e5_large`（JMTEB Classification 72.89，
`query: ` prefix 必須だが prefix 機構は既存）へ値を切り替える（この 2 値のみ `values` へ追記済み）．
(iii) 次元が変われば `embedding_view_concat` の連結次元も変わるため，`train_domain_classifier.py` の
再訓練は差し替えに構造的に付随する作業であり別レバーではない（Iter79/89 で確立した型）．

## Iteration 97: 分類器の多クラス分解を softmax から一対一へ変える

### 調査 (Iter97)

本フェーズでも**実験ノード wafl500〜509 には一切触れていない**（2026-09-23 絶対条件 (B)）．
開発ホスト上で行ったのは，リポジトリのコード・JSONL の読み取りと，**合成データによる
sklearn の API 実現性の確認（CPU のみ・数秒）**だけである．`models/`・`data/dataset.jsonl`・
`results/` への書き込みは 0 件．

**(1) 問い**（B161 (g) の申し送りを受けて設定した）

- Q1: 「10 クラスが 1 つの softmax と 1 組の線形境界を共有する」という現在の定式化の**外側**に，
  10 ドメイン一律の規則として実装できる代替の定式化はあるか．
- Q2: その代替は，Iter95/96 で観測された**零和的な再配分**（機序 M5）を構造的に回避できるか．
- Q3: 現行の較正済みパイプライン（`CalibratedClassifierCV(method="temperature")` ＋
  `sample_weight = n/(K*n_d)`）と，推論経路 `classifier.py:estimate_confidence_classifier()` の
  「`predict_proba` が 10 ドメインで和 1」という前提を壊さずに実装できるか．

**(2) 文献調査（tvly search / extract）**

- **Fürnkranz (2002), *Round Robin Classification*, JMLR 2:721–747**,
  <https://www.jmlr.org/papers/volume2/fuernkranz02a/html/node3.html> ——
  c クラス問題を c(c−1)/2 個の対ごと二値問題へ分解する方式．引用すると「in the round robin case,
  the base classifier uses fewer examples and thus has **more freedom for fitting a decision
  boundary** between the two classes」「**pairwise decision boundaries can be considerably
  simpler** than those originating from unordered binarization」であり，実例として
  **Knerr et al. (1992)** の数字認識（クラスは対ごとには線形分離可能だが one-against-all は
  単層ネットで解けなかった）を挙げる．さらに **Hsu & Lin (2002)** が
  「**線形カーネル SVM でこそ非線形カーネルより大きな OvO の優位**を得た」ことを引き，
  その理由を対ごと境界の単純さに帰している．**本研究の分類器は線形ヘッドであり条件が一致する**．
- **Galar, Fernández, Barrenechea, Bustince & Herrera (2011), *An overview of ensemble methods
  for binary classifiers in multi-class problems: Experimental study on one-vs-one and one-vs-all
  schemes*, Pattern Recognition 44(8):1761–1776**, <https://sci2s.ugr.es/ovo-ova> ——
  SVM・決定木・kNN 等の基底学習器を横断した実験比較で，OvO が OvA を上回る傾向を報告する．
  同グループのチュートリアル（<https://sci2s.ugr.es/sites/default/files/files/TutorialsAndPlenaryTalks/
  SSTiC-Trends%20in-Classification-Imbalanced-data-sets.pdf>）は，**多クラス不均衡への対処として
  pairwise learning を明示的に位置づけている**（各二値問題が 2 クラスだけを見るため，
  多数クラス全体を相手にする OvA より不均衡が緩む）．本研究は legal 77 行 対 250 行 ×8 という
  多クラス不均衡を抱えるので，この論点も該当する．
- **反証側: Rifkin & Klautau (2004), *In Defense of One-Vs-All Classification*, JMLR 5:101–141**,
  <https://www.jmlr.org/papers/volume5/rifkin04a/rifkin04a.pdf> ——
  「a simple one-vs-all scheme is **as accurate as any other approach**, assuming that the
  underlying binary classifiers are well-tuned」と主張する．**効果が出ない可能性も文献上
  同程度に支持されている**ことを明記しておく．本レバーは片側に寄った改良案ではなく，
  両論ある仮説の検定である．
- **Wu, Lin & Weng (2004), *Probability Estimates for Multi-class Classification by Pairwise
  Coupling*, JMLR 5:975–1005**（ <https://www.jmlr.org/papers/volume5/wu04a/wu04a.pdf> ，
  書誌は <https://stat.nccu.edu.tw/en/members/journalpaper/T-F-Wu-C-J-Lin-R-C-Weng-2004-Probability-Estimates-for-Multi-class-Classification-by-Pairwise-Coupling-Journal-of-Machine-Learning-Research-Vol-5-pp-975-1005-SCIE-10278537> ）
  —— 対ごと二値出力から多クラス確率を作る標準的な手続き（libsvm の実装根拠）．
  **本反復ではこれを自前実装せず**，sklearn の `OneVsOneClassifier.decision_function` を
  既存の temperature 較正に通す（下記 (3) で和 1 を実測確認済み）．自前実装が必要になるのは
  較正後の確率が退化していた場合だけで，その判断はスクリーニングのデータで行う．
- 本リポジトリでの既往: Iter59 前後で **OvR（binary relevance）を rank_2 以降の並べ替え用
  スコアとして**使った例はある（journal_archive の該当節）が，**argmax を決める多クラス分解
  そのものを softmax 以外にした実験は 1 度も無い**（`OneVsOne` の grep ヒット 0 件）．

**(3) 実現性の実測（開発ホスト，CPU のみ，合成データ）**

- `CalibratedClassifierCV(OneVsOneClassifier(LogisticRegression(max_iter=1000)),
  method="temperature", cv=5, ensemble=True)` は sklearn 1.9.0 で fit でき，
  `predict_proba` は (n, 10) で**各行の和が 1.0**（`classifier.py` の前提を満たす）．
- **重大な落とし穴（実測）**: 上記へ `sample_weight` を渡すと
  `UserWarning: Since OneVsOneClassifier does not appear to accept sample_weight, sample weights
  will only be used for the calibration itself.` が出て，**base estimator への重みが黙って
  捨てられる**．そのまま走らせると『分解方式の変更』と『クラス均衡重み（B60）の喪失』を同時に
  変えることになり，単一レバー原則が壊れる．
  `sklearn.config_context(enable_metadata_routing=True)` の下で
  `LogisticRegression(...).set_fit_request(sample_weight=True)` を付けると，45 本すべての
  対ごと LR へ重みが流れ**警告は出なくなる**ことを確認した．
- 計算量: 二値 LR 1 本（400 行 × 5,120 次元）の fit は **0.06 秒**，単一 softmax
  （1,820 行 × 5,120 次元）は **1.0 秒**．45 本でも同オーダーで，スクリーニング全体が
  wafl-ctrl5 の CPU で完結する（**再埋め込み 0 回・GPU 不要**）．
- なお対ごとの重みは，クラス d の行が `n/(K*n_d)` を持つため**どの対でも両クラスの総重みが
  n/K で等しく**なる．すなわち OvO へ移しても『クラス均衡』の意味は保たれ，変わるのは
  「境界を 1 組の共有ベクトルで張るか，対ごとに独立に張るか」だけである．

### 仮説 (Iter97)

**education（CV recall 0.5350）・medical（0.6733）の誤りが減らず，かつドメイン間で零和的に
再配分されるのは，訓練データの量や重みの問題ではなく，10 クラスが 1 つの softmax 正規化と
1 組の線形境界を共有していることに由来する（M5 の構造的な言い換え）．**
対ごとに独立な二値問題へ分解すれば，education 対 medical のような紛らわしい対の境界は
他 8 クラスの事情から解放され，Fürnkranz (2002)・Hsu & Lin (2002) の言う「線形でこそ効く
対ごと境界の単純さ」が得られるはずである．予測される観測は，**弱いクラス（education・medical）の
recall が上がり，かつ強いクラスの退行が Iter95/96 のようには生じない**ことである
（零和なら total は動かないが，分解が効くなら total が動く）．

### 単一レバー (Iter97)

- **レバー**: `classifier_multiclass_decomposition` = `one_vs_one_pairwise_coupling`
  （config.yml の `levers` 末尾に本フェーズで追記．B162 で `[auto-decided]`）
- **何を何から何へ**: `scripts/train_domain_classifier.py:train_classifier()`（L210 付近）の
  base estimator を
  **`LogisticRegression(max_iter=1000, class_weight=None)`（単一 softmax）から
  `OneVsOneClassifier(LogisticRegression(max_iter=1000, class_weight=None))`（45 本の対ごと二値 LR）**へ．
  `CalibratedClassifierCV(method="temperature", ensemble=True, cv=5)` でラップする点，
  `sample_weight` の式，訓練行はそのまま．**変えるのはこの 1 箇所だけ**．
- **レバーを読むコード行と到達条件（d0004 §4 の恒久対策）**:
  - 変更箇所は `scripts/train_domain_classifier.py:train_classifier()` の base estimator 生成行
    （現行 `base_estimator = LogisticRegression(max_iter=_MAX_ITER, class_weight=None)`）．
    この関数は再訓練時に必ず通る（スクリーニングの
    `scripts/screen_composition_preserving_volume_expansion.py` も同関数を import している）．
  - 実行時経路 `classifier.py:estimate_confidence_classifier()` は
    `models/domain_classifier.joblib` を読んで `predict_proba` を呼ぶだけで，**コード変更は不要**．
    したがって「設定が読まれない no-op」は原理的に起こりえない代わりに，
    **artifact の差し替えとデプロイが唯一の到達条件**である．
    本走前に (a) 新 artifact の sha256 が旧と異なること，(b) `n_features_in_` == 5120，
    (c) 10 ドメインの `predict_proba` の行和が 1.0，(d) 各ノード上の
    `models/domain_classifier.joblib` の sha256 が一致すること，を必ず確認する．
- **固定する構成（1 つも動かさない）**: 訓練集合 `data/classifier_train_iter94_dedup.jsonl`
  （2,275 行）・`sample_weight = n/(K*n_d)`・較正 `temperature`・`C=1.0`・`max_iter=1000`・
  埋め込みモデル `qwen3-embedding:4b`・`embedding_instruction`（Iter81 の P1 文言）・
  `embedding_view_concat: true`（5,120 次元）・評価集合 `data/dataset.jsonl`
  （sha256 `2114e048...`，3,750 行，読むだけ）・`dispatch_gap_threshold: 0.36`・
  `dispatch_gap_max_k: 4`・`dispatch_top_k: 2`・`aggregation_method: max_confidence`・
  expert/light モデルとノード割り当て．

### 事前スクリーニング設計（事前登録 / Iter97）

Iter95/96 で確立した型をそのまま踏襲する．**足切りを通った場合は本走を省略しない**
（2026-09-23 絶対条件 (A)）．実施場所は **wafl-ctrl5**（絶対条件 (B)．
`~/expert-mesh-iter95/` の uv 3.12 環境と `data/embcache_iter96_scaled.npy` の先頭 2,275 行を流用．
**新規の埋め込み計算は 0 回**）．

- **主評価**: 既存 2,275 行の層化 5-fold（`StratifiedKFold`，seed = 96 / 196 / 296 の 3 seed，
  併合 n = 6,825）．テスト fold は両腕で完全に同一．
  **腕 A = 現行 softmax ヘッド（Iter96 実測 CV top1 = 0.802637 と一致することを確認すること）**，
  **腕 B = OvO ヘッド**．訓練行・重み・埋め込みは両腕で同一で，違いは base estimator だけ．
  スクリプトは `scripts/screen_composition_preserving_volume_expansion.py` を雛形にし，
  「追加行」の概念を落として base estimator を切り替える形に書き換える
  （統計は `metrics.py` の既存実装 `compute_mcnemar_test` /
  `compute_domain_recall_mcnemar_test` / `apply_benjamini_hochberg` をそのまま使う）．
- **併記する診断値（判定には使わないが必ず出す）**: 両腕の
  (a) per-domain recall（腕 A の実測値は下表），(b) top-2 確率差 `gap` の分布と
  **`gap < 0.36` の行の割合**（dispatch 候補数が動くかの事前把握），
  (c) `sample_weight` 警告の有無（1 件でも出たら**実験不成立として中断**）．

腕 A の per-domain CV recall（Iter96 スクリーニング実測．`results/iter96_screening/screening_result.json`）:
mathematics 0.9480 / computer_science 0.9200 / natural_science 0.8760 / history_culture 0.8524 /
legal 0.8398 / business_economics 0.8160 / general 0.7987 / social_science 0.7880 /
medical 0.6733 / **education 0.5350**．

### 反証可能な予測（事前登録 / Iter97）

- **P1（仮説の核心）**: 弱い 2 クラス（education・medical）の CV recall Δ の合計が **+3.0pt 以上**．
  これが 0 付近なら「共有境界が律速」という読みは誤りで，分解では触れない領域の問題である．
- **P2（零和性の破れ）**: CV top1 の Δ が **+1.0pt 以上**．Iter95（+0.366pt）・Iter96（+0.322pt）は
  いずれも零和で相殺された結果なので，分解が効くならここが初めて動く．
- **P3（不均衡の緩和）**: legal（77 行）の recall が退行しない（Δ ≥ −1.0pt）．
  Galar らの言う pairwise learning の不均衡緩和が効くなら，むしろ上がる側に出る．
- **P4（確率の健全性）**: 較正後 `predict_proba` の行和が全行 1.0 ± 1e-6 で，
  `gap < 0.36` の行の割合が腕 A から **±5pt 以内**に収まる．
  これを外れる場合は dispatch 候補数が変わるため，本走の複合設問指標の解釈に注釈を付ける．

### 成功条件・非退行条件（事前登録 / Iter97）

**基準線（本走）**: `results/20260928_160921/` の top1 = **0.833067**（3,750 問，
Wilson 95%CI [0.8208, 0.8447]）．再現性の床は ±0.25pt．
**基準線（CV）**: 腕 A = 0.802637．CV の Δ のノイズ幅は，Iter95（Δ=+0.366pt, p=0.1595）・
Iter96（Δ=+0.322pt, p=0.1415）から **SE ≒ 0.22pt** と見積もる（足切り +1.0pt は約 4.5 SE）．

- **スクリーニングの足切り（両方を満たさなければ本走せず `closed`）**:
  1. **CV top1 Δ (B−A) ≥ +1.0pt**（Iter95・Iter96 と同一水準を据え置く．
     水準そのものの一般則は B154 (A3) として未回答のまま）．
  2. **per-domain recall の BH 補正後（q=0.05，10 指標）有意な退行が 0 件**．
  - 外した場合は `models/`・`config.yaml`・`data/dataset.jsonl` を**1 バイトも変更せず**
    結果だけ記録してフェーズ 3 へ渡す（Iter95・Iter96 と同じ運用．絶対条件 (A) 非抵触）．
- **本走（足切り通過時のみ）**: `models/domain_classifier.joblib` を OvO 版で再訓練
  （旧 artifact は `models/domain_classifier_pre_iter97_ovo.joblib` へ退避），`mise run deploy` と
  sha256 確認のうえ，**wafl500〜509 で 3,750 問フルスペック**を 1 回実行する．
- **本走の判定**: 全体 top1 が **Δ ≥ +0.5pt かつ McNemar p < 0.05** で `adopted`，
  Δ ≥ +0.5pt だが p ≥ 0.05 なら `adopted_small`（要再現），|Δ| < 0.25pt なら `negligible`，
  Δ ≤ −0.5pt なら `rejected`．
- **必須の非退行条件（1 つでも破れたら artifact をロールバック）**:
  - C1: per-domain precision/recall 計 20 指標の BH 補正後の有意退行が **0 件**．
  - C2: `used_fallback` 率・`dispatch_failed` 件数が基準線から増えない．
  - C3: レバー発火の証拠（新 artifact の sha256 が旧と異なる，`n_features_in_`=5120，
    ノード上の sha256 一致，`predict_proba` の行和 1.0）．
  - C4（参考値）: Random / BestSingle / Oracle を併記し BestSingle 超過を明示（success_criteria (3)）．
  - C5: 複合設問 730 行の top1（基準 0.7904）が −1.0pt を超えて下がらないこと，
    および `compound_domain_set_recall`・`compound_mean_dispatched_count` の併記．
  - C6: `mean_duration_ms` が基準線の +20% 以内（推論は 1 回の `predict_proba` のままなので
    実質不変のはずだが，確認する）．

### 実行フェーズへの申し送り（Iter97）

- **`sample_weight` の metadata routing を必ず入れること**（上記 (3)）．警告が出た時点で中断．
  これを怠ると「分解方式」と「クラス重み」の 2 レバーを同時に動かした実験になる．
- スクリーニングは wafl-ctrl5 で完結する（再埋め込み 0 回・GPU 不要）．
  wafl500〜509 は**本走以外で一切使わない**（絶対条件 (B)）．
- `dispatch_gap_threshold` の再較正は別レバーであり，本反復では**絶対に触らない**．
  gap 分布が動いた場合も，記録に留めて次イテレーションの候補とする．
- 却下した代替案と理由は backlog B162 に記録した（top-2 だけを対ごとに並べ替える 2 段階案・
  非線形ヘッド・LLM による入力表現の拡張・ラベル定義の変更・複合評価集合の拡充）．

### 実験 (Iter97)

**本走は実施していない**．事前登録した足切り 2 条件をいずれも満たさなかったため，停止規則どおり
本番コード（`scripts/train_domain_classifier.py`・`config.yaml`・`data/dataset.jsonl`・`models/`）を
1 バイトも変更せずフェーズ 3 へ引き渡した（Iter93・Iter95・Iter96 と同一運用）．
実機ノード wafl500〜509 は不使用．スクリーニングは wafl-ctrl5 の `~/expert-mesh-iter95/` 環境で完結し，
再埋め込み 0 回・GPU 不要で実行した．

**単一レバー原則の担保（最重要の事前条件）**: 腕 B は
`sklearn.config_context(enable_metadata_routing=True)` ＋
`LogisticRegression.set_fit_request(sample_weight=True)` を入れたうえで，全ての `fit()` 呼び出しを
warning ガードで包み「警告が 1 件でも出たら `SampleWeightDroppedError` を送出して中断」する実装とした．
**警告は一切発生せず**，45 本すべての対ごと LR に `sample_weight` が到達したことを確認した．
すなわち本実験は「クラス均衡重み（B60）の喪失」との 2 レバー同時変更にはなっていない．

**設計**: 3 seed（96/196/296）× 固定 5-fold，n=2,275（併合 6,825）．
腕 A は `train_classifier()` を無変更のまま import して呼び出し，腕 B は `_train_ovo_classifier()`
（`CalibratedClassifierCV(OneVsOneClassifier(LogisticRegression(...)), method="temperature", cv=5,
ensemble=True)`）とした．生データは `results/iter97_screening/screening_result.json`．

**結果**

| 指標 | 値 | 事前登録した足切り | 判定 |
|---|---|---|---|
| 腕 A CV top1 | 0.802637 | —（Iter96 実測と厳密一致） | 測定系は健全 |
| 腕 B CV top1 | 0.794579 | — | — |
| **CV Δ (B−A)** | **−0.806pt** | ① Δ ≥ +1.0pt | **不通過** |
| 全体 McNemar | discordant 138 / 83，χ²=13.19，**p=0.00028** | — | **B が有意に悪い** |
| BH 後の有意退行 | **2 件** | ② 0 件 | **不通過** |

per-domain recall Δ: history_culture −2.86pt（p=0.000144，BH 有意退行），
social_science −4.53pt（p=3.06e-6，BH 有意退行），education +2.24pt（p=0.0648，非有意），
medical −1.87pt（p=0.0303，BH 補正後は非有意），legal ±0.0pt．

**事前登録した予測の照合**: P1（education+medical の recall Δ 合計 ≥ +3.0pt）は実測 **+0.374pt で反証**．
P3（legal recall Δ ≥ −1.0pt）は ±0.0pt で通過．P4（`gap<0.36` の行割合が ±5pt 以内）は
腕 A 20.1% → 腕 B 8.3%（**Δ = −11.8pt**）で**逸脱**した．なお `predict_proba` の行和は両腕とも 1.0 で，
確率化そのものは健全である（Wu-Lin-Weng の自前実装を避け sklearn の temperature 較正で代替した判断は妥当だった）．

**検証**: `ruff check` 通過，`py_compile` OK，`git diff --stat scripts/train_domain_classifier.py` 差分なし，
`git status --short config.yaml models/ data/dataset.jsonl` 差分なし（無変更を実測確認）．

**分析フェーズへの申し送り**: (i) 文献の反証側 Rifkin & Klautau (2004) *In Defense of One-Vs-All* が
今回の実測と整合した．両論ある仮説の検定として事前に反証側を明記していたため，この結果は解釈可能である．
(ii) history_culture・social_science という**強クラス側**で有意退行が出た機序（対ごと分解が強クラス間の
境界をむしろ不安定化した可能性，および P4 の gap 分布の大幅な収縮との関係）は考察対象として残る．
(iii) 予備値 `error_correcting_output_codes`（B162 (C)）を次点として引くかはフェーズ 3 の判断とする．

### Iteration 97 実行済み

**変更したもの**: 事前スクリーニングのみ．`scripts/screen_classifier_multiclass_decomposition.py`（新規）と
`results/iter97_screening/screening_result.json`（新規）の 2 件だけである．
事前登録した足切り 2 条件をいずれも満たさなかったため停止規則どおり本走を行わず，
`scripts/train_domain_classifier.py`・`config.yaml`・`data/dataset.jsonl`・`models/` は 1 バイトも
変更していない．基準線 top1 = 0.833067 は不変．実機ノード wafl500〜509 は不使用．

**結果（再掲）**: 腕 A（現行 softmax）CV top1 = 0.802637（Iter96 実測と厳密一致＝測定系は健全），
腕 B（OvO）= 0.794579，**Δ = −0.806pt**．全体 McNemar discordant 138/83，χ²=13.19，p=0.00028．
BH 後の有意退行 2 件（history_culture −2.86pt，social_science −4.53pt）．
P1 反証（+0.374pt < +3.0pt），P2 不通過，P3 通過，P4 逸脱（`gap<0.36` の行割合 20.1%→8.3%）．
`sample_weight` の警告 0 件で，45 本全ての対ごと LR に重みが到達した（単一レバー原則は担保）．

#### ノイズか信号か

Δ = −0.806pt は，Iter95（+0.366pt, p=0.1595）・Iter96（+0.322pt, p=0.1415）から見積もった
**CV の Δ の SE ≒ 0.22pt の約 3.7 倍**であり，かつ同一 fold の対比較（McNemar p=0.00028）でも
有意に悪い．**ノイズではなく，負方向の信号である**と判断する．

ただし**検定の楽観性を明記しておく**．3 seed × 5-fold の併合 n=6,825 は，実体としては同じ 2,275 行を
3 回数えたものであり，行の独立性を仮定した McNemar は反保守的である．discordant を seed 数で割った
保守側の再計算では，全体 46/28 で χ²=4.56・**p=0.033**（依然有意），social_science は p=0.0055，
history_culture は p=0.020 となる．すなわち**「腕 B が悪い」という全体の結論は保守側でも保たれる**が，
**足切り 2（BH 後の有意退行 0 件）の当落は保守側では入れ替わりうる**（BH q=0.05・10 指標では
最小 p=0.0055 が閾値 0.005 をわずかに超えるため有意 0 件になる）．
今回は**足切り 1 が符号ごと逆方向に外れている**ため，この揺らぎは判定に影響しない．
なお同じ楽観性は Iter95・Iter96 の足切り 2 の評価にも及ぶので，**次回以降は seed 併合 McNemar を
主，seed 除算した保守値を併記**すること（学び 4）．

#### 判定: `closed`（ただし当該 value は再試行不可）

`rejected` は**本走の 2 軸表（Δtop1 × McNemar p）に基づく判定語**であり，本走が無い以上どのセルにも
到達していない．したがって Iter93・Iter95・Iter96（B159 (A)・B161 (a)）と同じく **`closed`** とする．
新しい判定語は設けない（判定語の集合を増やすことは記録スキーマの破壊的変更であり，1 イテレーションの
都合で自動決定すべきでない．必要なら人間判断で導入する — B163 (A)）．

その代わり，**同じ `closed` でも今回は先行 3 例と性質が異なる**ことを記録に残す．
Iter95・Iter96 は「Δ は正だが足切りに届かない」であり，検出力を上げれば再試行の余地があった．
今回は **Δ が負で，かつ有意**である．したがって value `one_vs_one_pairwise_coupling` は
**単なる未通過ではなく反証されたものとして扱い，再試行しない**（config.yml の当該 note に注記した）．

#### 考察 (i): なぜ強クラス側（history_culture・social_science）が退行したのか

退行は「強いクラスが一律に損をした」のではない．腕 A の recall と Δ の**単調性は完全に消えている**
（Spearman ρ = **+0.018, p = 0.96**．Iter96 は ρ = −0.717, p = 0.020 だった）．
natural_science（0.876）は +0.67pt，mathematics（0.948）は −0.53pt にすぎず，
損失は **social_science −4.53 / history_culture −2.86 / medical −1.87** の 3 クラスに集中している．
これらは**意味的な近傍を多く持つクラス**（social_science は general・education・business_economics と，
history_culture は general・social_science と重なる）であり，mathematics・computer_science のような
孤立したクラスは動いていない．

OvO でこの分布になる機序として，文献上よく知られた次の 2 つが本件の条件と噛み合う．

- **(a) 対ごと二値問題の分散増大（p≫n の悪化）**: 各対ごと LR は 2 クラス分の行（約 250+250=500 行）
  しか見ないのに次元は 5,120 のままである．単一 softmax の `w_d` が 2,275 行すべてから推定されるのに対し，
  対ごと境界は**約 1/4.5 の標本で推定される**．近傍クラス間の境界ほどこの分散の影響が大きい．
- **(b) 非適格分類器（non-competent classifier）の票**: ある行の真クラスが d のとき，
  45 本のうち d を含むのは 9 本だけで，残る 36 本は d を知らないまま票を投じる．
  票数で argmax を決める以上，近傍クラスに 1 本負けただけで順位が入れ替わりうる．
  Galar らが OvO に対して動的分類器選択を併用する動機がこれである（本反復では素の OvO を使った）．

**既存データで検証可能な手順（再埋め込み 0 回・CPU のみ）**:
`scripts/screen_classifier_multiclass_decomposition.py` に診断出力を足して再実行する．
1. 両腕の **10×10 混同行列**を出す．予測: 腕 B で失われた social_science の 42 行は
   general・education・business_economics へ偏って流れる（一様には散らない）．
2. **45 対それぞれの二値 CV 正解率**を出す．予測: social_science–general，
   social_science–education，history_culture–general の 3 対が最下位群に来る．
3. 腕 B の**勝者と次点の票差の分布**を出す．予測: 腕 A で正解し腕 B で誤った 138 行のうち
   多数派が**票差 1**（＝ 1 本の対ごと分類器の誤りで覆った）である．
   これが (b) の直接証拠になり，票差が 2 以上ばかりなら (a) 側の説明が優勢になる．

#### 考察 (ii): P4 の gap 分布の収縮（20.1% → 8.3%）は何を意味するか

`OneVsOneClassifier.decision_function` は**整数の得票数**（0〜9）に微小な決定値を足したものであり，
本質的に**離散で分解能が低い**．勝者はしばしば 9 勝，次点は 8 勝といった値を取るので，
スコア差は連続量ではなく粗い階段状になる．これに **単一スカラーの temperature 較正**を掛けると，
階段差がそのまま確率差へ写り，**上位 2 クラスの差が人為的に広がる**．
実測の `gap < 0.36` の行割合 20.1% → 8.3%（−11.8pt）はこの形で説明できる．

**確信度は上がったのに正解率は下がった**（Δ top1 = −0.806pt）という組み合わせなので，
これは「識別が鋭くなった」のではなく**過信（較正の劣化）**である可能性が高い．
運用上の含意は 2 つある．
- **(1)** 仮に本走していれば，`dispatch_gap_threshold=0.36` の下で複数専門家へ送出される行が
  20.1% → 8.3% へ半減し，`compound_mean_dispatched_count` と `compound_domain_set_recall` が
  CV top1 の −0.81pt を超えて悪化した公算が高い．**足切りで止めた判断は結果的に妥当だった**．
- **(2)** `dispatch_gap_threshold` は**分解方式を跨いで移植できない**．
  archive の学び「特徴量を変えるレバーの後には必ず gap 閾値の未較正が残る」（Iter81→82→83 で 3 回観測）は，
  **特徴量だけでなく決定層の形を変えた場合にも成り立つ**．今回は本走していないので再較正は不要であり，
  `dispatch_gap_threshold_recalibration` は adopted・収束のまま触らない（本反復の禁止事項どおり）．

**既存データで検証可能な手順**: 両腕の**行ごとの max-prob と gap をダンプ**し，
(a) ECE と Brier スコア，(b) `gap ≥ 0.36` の行に限った正解率，を腕ごとに比較する．
過信であれば **腕 B の ECE が腕 A より大きく，かつ `gap ≥ 0.36` バケットの正解率が腕 A より低い**．
逆に腕 B の高 gap バケットの正解率が腕 A と同等以上なら，「鋭くなったが総数を落とした」という
別の読みになる．併せて較正前の `decision_function` のヒストグラムを見れば，離散性（数個のモードへの
集中）は目視で確認できる．

#### 考察 (iii): 機序 M5 との整合／不整合

M5 は「クラス総重み一定（`sample_weight = n/(K·n_d)`）の下では，行追加は情報の追加ではなく
**クラス間の決定境界の再配分**である」という読みで，Iter95/96 の**零和性**と
**腕 A の recall に対する Δ の単調性**を根拠にしていた．Iter97 はこの M5 が
「1 つの softmax 正規化と 1 組の共有境界」に由来するという構造的解釈を検定した実験であり，
結果は**部分的に不整合**である．

- **不整合 1（零和性が再現しない）**: per-domain recall Δ の内訳は
  **正側 +3.04pt 対 負側 −10.72pt，合計 −7.68pt**（クラスがほぼ均等なので非重み平均 −0.768pt は
  全体 Δ −0.806pt とよく一致する）．Iter95/96 のような**差し引きゼロの再配分ではなく，正味の損失**である．
  すなわち OvO への変更は「境界の配り直し」ではなく**情報を失う操作**だった（考察 (i) の (a)(b)）．
- **不整合 2（単調性が再現しない）**: Spearman ρ = +0.018（p=0.96）で，Iter96 の −0.717 は消える．
  **強さに対する単調性は M5 の普遍的な性質ではなく，「行を足す」という操作に固有の署名だった**．
- **整合している点**: 弱いクラス（education）だけが改善し（+2.24pt），強いクラスが損をするという
  **符号の向き自体**は残っている．ただし今回は改善幅が小さく非有意（p=0.065，BH 後も非有意）で，
  education の recall は 0.535 → 0.557 と**依然として壊滅的**である．

したがって M5 の**構造的解釈（共有 softmax 正規化が零和性の原因）は支持されなかった**．
決定層の分解方式を根本から変えても education の誤りはほぼそのまま残ったのだから，
**律速は決定層ではなく，5,120 次元の埋め込み表現そのもの（あるいはラベルの張り方）にある**という
読みが最も素直である．M5 自体は「行追加操作に関する経験則」としては生き残るが，
**その原因を softmax 正規化に帰する説明は取り下げる**．

**既存データで検証可能な手順**: 腕 A のクラス別 `‖w_d‖` と，腕 B の対ごと重みベクトルから合成した
実効的なクラス別ノルムを比較する．M5 の構造的解釈が正しければ腕 B では総重み一定の制約が消えて
ノルムの分散が広がるはずで，広がっていなければ「制約は softmax ではなく標本サイズ由来」という
上の読みが補強される．

#### 学び

1. **両論ある仮説を事前に両論のまま登録しておくと，負の結果が解釈可能な知識になる**．
   本反復は Fürnkranz (2002)・Galar et al. (2011)（OvO 優位）と Rifkin & Klautau (2004)
   *In Defense of One-Vs-All*（OvA で十分）を**事前に**並記して検定と位置づけた．
   実測は後者と整合し，「小標本・高次元・クラス数 10」という本研究の条件では
   **対ごと分解の『境界の単純さ』の利得より，標本分割による分散増大の損失が勝つ**ことが分かった．
   Hsu & Lin (2002) の線形カーネル SVM での OvO 優位が本件に外挿できなかったのは，
   本研究の n/p 比（2,275 行 / 5,120 次元）が文献の設定と桁違いに厳しいためと考える．
2. **sklearn のメタ推定器は，サポートしないメタデータを警告 1 行で黙って捨てる**．
   `CalibratedClassifierCV(OneVsOneClassifier(...))` に `sample_weight` を渡すと
   base estimator 側の重みが落ち，気づかなければ「分解方式」と「クラス均衡重み」の
   2 レバー同時変更になっていた．`enable_metadata_routing=True` ＋ `set_fit_request` で明示配線し，
   **「警告が 1 件でも出たら実験不成立として中断する」ガードをコードに埋め込んだ**のが有効だった．
   この型（単一レバー性をコードで機械的に強制する）は今後のスクリーニングでも踏襲する．
3. **決定層（多クラス分解・較正・重み・訓練行）を一巡した結果，CV は 0.79〜0.82 に張り付いたままである**．
   Iter88〜96（データ量・重み・粒度）に加え Iter97（分解方式）まで動かないのだから，
   残る説明変数は**入力表現とラベルの張り方**である．次の探索軸はここに限定してよい．
4. **seed 併合の McNemar は反保守的である**（同じ行を seed 数だけ重複計上している）．
   本反復では結論は変わらなかったが，足切り 2 の当落は保守側で入れ替わりうることを確認した．
   今後は**保守値（discordant を seed 数で割った再計算）を必ず併記**する．
5. **確信度が鋭くなったことを性能改善の証拠として読んではいけない**．腕 B は
   `gap<0.36` の行を半減させながら top1 を下げた．**gap 分布の変化は，正解率と切り離して
   単独では解釈できない**（ECE と高 gap バケットの正解率を必ず併記する）．

**次の一手**: `classifier_multiclass_decomposition` の未試行 value
**`error_correcting_output_codes`** を引く（backlog B163）．これは決定層の軸を**閉じるための
確認実験**であり，OvO 固有の弱点（対ごとの標本分割・得票の離散性・非適格分類器）と
「決定層の分解一般」を切り分ける．既存のスクリーニングスクリプトへ第 3 腕を足すだけで済み，
追加費用はほぼゼロである．ここも通らなければ**決定層の軸は打ち止め**とし，
Iter99 は調査フェーズから入力表現／ラベルの軸を探索する．

