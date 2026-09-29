## Iteration 101: 未使用の light_model を常駐から外して VRAM を空け，融合構成を運用可能にする

### 調査 (Iter101)

**実施場所の申告**: 本フェーズで行ったのは (a) 開発ホストでのリポジトリ読み取り，(b) tavily による外部調査，
(c) **wafl500 / wafl502 への読み取り専用の SSH 接続**（`ollama ps`・`nvidia-smi --query-compute-apps`・
`docker logs` の参照のみ）である．**モデルの起動・推論・設定変更は一切行っていない**ので，
恒久ルール (B)（wafl500〜509 でサブ実験をしない）には抵触しない．計算を伴う作業は発生していない．

**問い**: (Q1) `light_model` は現行構成の実行時経路で本当に呼ばれないか．(Q2) 呼ばれないなら，
それを常駐から外すと VRAM はどれだけ空き，Iter100 の C6 違反（`mean_duration_ms` 9733.350）の機序に
届くか．(Q3) Ollama が VRAM 不足のときに何をするか（無言の劣化の正体）．

**A1（Q1: コード上の到達条件．`light_model` は実行時に呼ばれない）**
- `http_server.py:364-370` の `_estimate_probe_confidence()` は，`confidence_signal_method` が
  multi_sample / stp / semantic_entropy / p_true の**いずれでもない**とき（現行は `self_report`）に
  `routing_method == supervised_classifier` の分岐へ落ち，`estimate_confidence_classifier()` を
  呼ぶだけで返る．コード中のコメントも `# No LLM call:` と明言している．
  すなわち `/probe` は LLM を一切呼ばない．
- `light_model` の他の参照は 3 か所のみ: (i) `http_server.py:397` の**起動時 warmup**，
  (ii) `node.py:233` の `_fallback_answer()`（`select_dispatch_targets()` が空集合を返したときだけ．
  `confidence_threshold=0.0` の現行構成では発生せず，Iter100 本走 3,750 問で `fallback_rate=0.0` 実測），
  (iii) mesh を使わない `scripts/run_central_experiment.py`（本走では未使用）．
- **B172 (d) の根拠 2 はコード上は裏付けられた**．ただし本フェーズはコード読解までであり，
  実ログでの 0 件確認は G0-a（予備 20 問）に残す．

**A2（Q2: 実機の VRAM 実測．本フェーズの新しい一次データ）**
2026-09-29 現在（Iter100 本走の直後，アイドル状態）の実測値である．

| ノード | `ollama ps` の常駐モデル | `nvidia-smi` 実測 (MiB) |
|---|---|---|
| wafl500（入口・general） | expert-mesh-general-lora / qwen3-embedding:4b / ruri-v3-310m（すべて `100% GPU`） | 5,186 + 4,314 + 486 = **9,986 / 12,288** |
| wafl502（peer・legal） | expert-mesh-legal-lora のみ | **5,186 / 12,288** |

- **`light_model` はどちらのノードにも残っていない**．起動時に warmup で載ったものが，
  その後の容量逼迫で追い出されたまま復帰していない（誰も呼ばないので復帰する契機が無い）と読める．
- 起動直後の想定常駐量は，**入口ノード** = light 2,027 + expert 5,186 + qwen 4,314 + ruri 486
  = **12,013 MiB**，**peer** = light 2,027 + expert 5,186 + qwen 4,314 = **11,527 MiB**．
  物理 12,288 MiB に対する余裕はそれぞれ **275 MiB / 761 MiB** しかない．
  CUDA コンテキスト（runner プロセスごとに数百 MiB）と KV キャッシュ（`ollama ps` の CONTEXT は 4096）を
  加えれば**容易に超過する**．**`light_model` を外せば入口 9,986（余裕 2,302）・peer 9,500（余裕 2,788）**
  となり，超過の余地が消える．

**A3（Q2 の補強: ollama の GIN ログに残る「ロードの量子」）**
wafl500 / wafl502 の `docker logs expert-mesh-ollama-1` を読み，所要時間を集計した（読み取りのみ）．

| ノード | エンドポイント | 件数 | 中央値 | p95 | 最大 |
|---|---|---|---|---|---|
| wafl500 | `/api/embeddings` | 12,652 | 30.2ms | 79.9ms | 3,488ms |
| wafl500 | `/api/chat` | 185 | 204.8ms | 8,380.6ms | 8,462.0ms |
| wafl502 | `/api/chat` | 331 | 279.3ms | 8,429.0ms | 8,500.1ms |

- **分布が明確に二峰である**: 生成呼び出しの中央値は 0.2〜0.3 秒なのに，p95 と最大値が
  **8.4〜8.5 秒に集積している**．8.4 秒という一定値は生成量の分散では説明できず，
  **5.3GB の expert モデルを読み直す固定コスト**とみるのが自然である．
  すなわち Iter100 の裾（単一 p95 9,708ms・複合 p25 32,029ms）は，
  **1 回あたり約 8.4 秒のモデル再ロードが 1〜4 回重なった形**として説明がつく．
- **埋め込み自体は支配項ではない**（中央値 30ms×4 回＝約 120ms）．
  これは「融合の内在コストは小さい」という Iter100 の分解（単一中央値 +299ms）と整合する．
- **限界**: この GIN ログは Iter100 本走の全区間を覆っていない可能性があり（保持行数の制約），
  再ロード回数の絶対値は主張できない．主張できるのは**二峰性と 8.4 秒の量子の存在**までである．

**A4（Q3: 外部調査．Ollama は VRAM 不足を無言で劣化させる）**
- Ollama 公式 FAQ（docs.ollama.com/faq，2026 参照）: `keep_alive` は負値で常駐，`0` で即時アンロード．
  サーバ既定は `OLLAMA_KEEP_ALIVE`．**`/api/generate` の per-request `keep_alive` が既定を上書きする**．
- SSD Nodes「Keep an Ollama model loaded in memory」（www.ssdnodes.com）: スケジューラは新規ロードに
  メモリが足りないとき常駐モデルを**アンロードする．`keep_alive: -1` で載せたモデルも退避対象になる**．
  → **`OLLAMA_KEEP_ALIVE=-1` は「絶対に退避しない」保証ではない**（本リポジトリの
  docker-compose.yml のコメントが暗に前提していた読みは，この点で不正確である）．
- ollama/ollama Issue #14258「GPU-to-CPU fallback happens silently with no user-visible warning」
  （github.com/ollama/ollama/issues/14258）: VRAM に収まらないとき Ollama は**警告なしに CPU 実行へ落ちる**．
  利用者に見えるのは「なぜか遅い」だけで，痕跡は debug ログにしか残らない．
  → Iter100 の `dispatch_failure` 0 件・無言のレイテンシ劣化という観測と一致する．
- ollama/ollama Issue #6008: `server.log` の `offloaded 42/81 layers to GPU` と
  `ollama ps` の `PROCESSOR` 列（`100% GPU` / `48%/52% CPU/GPU`）が部分オフロードの判定材料になる
  （modelfit.io / netray.co の 2026 年の解説も同じ手順を挙げる）．→ **G0 の検査項目に採用する**．
- DEV Community「Ollama keep_alive: My Model Reloaded 214 Times in One Day」（dev.to，2026）:
  **同時に載り切らない 2 モデルに `keep_alive: -1` を付けると事態は悪化し，部分 CPU オフロードで
  生成が 42 tok/s → 6 tok/s（約 7 倍）に落ちた**という実測報告．対処は「GPU あたり常駐 1 モデル，
  埋め込みは別ホストへ」．→ **Iter100 の複合 8.4s → 40.9s（約 4.9 倍）と同じ桁**である．
  （注: 二次情報であり査読を経ていない．機序の傍証として扱い，判定には使わない．）
- 総合すると，**「容量をわずかに超えた状態で常駐を宣言すると，退避・再ロードか部分 CPU オフロードの
  どちらかが無言で起きる」**というのが Iter100 で観測された現象の最も素直な説明である．
  Iter100 の `OLLAMA_MAX_LOADED_MODELS=4` は**スロット数**の上限であって容量の上限ではないため，
  この失敗を防げなかった（Iter100 の学び 1 と同じ）．

### 仮説 (Iter101)

**主仮説 H1**: 現行構成で一度も呼ばれない `light_model`（約 2,027 MiB）を起動時 warmup から外すと，
入口ノードで 12,013 → 9,986 MiB，peer で 11,527 → 9,500 MiB となり物理 12,288 MiB に対する余裕が
2.3〜2.8GB に回復する．その結果，**A3 で観測した 8.4 秒のロード量子が消え**，
`mean_duration_ms` は基準線 2,534.762ms 近傍（≤ 3,041.7ms）へ戻る．
**表現（6,656 次元の融合特徴）と分類器 artifact は 1 ビットも変えない**ので，
top1 は Iter100 実測 0.839467 を再現性の床 ±0.25pt の範囲で再現する．

**対抗仮説（事前登録）**:
- **A1'（既に退避済みで効果なし）**: `light_model` は本走の早い段階で追い出され，
  残りの大半の区間では既に VRAM を占有していなかった．この場合レイテンシは改善せず C6 は再び FAIL する．
  → G0-a/G0-vram の実測と，予備 20 問のレイテンシで本走前に検出できる．
- **A2'（真因は KV キャッシュ／CUDA コンテキスト）**: 2.0GB 空けても
  入口ノードの 3 モデル＋context 4096 が収まらず，部分 CPU オフロードが残る．
  → `ollama ps` の `PROCESSOR` 列が `100% GPU` でない行として現れる（G0-b で検出）．
- **A3'（真因は外部競合）**: 他ユーザーのジョブ（B169 の `namit` 等）の再来．
  → 本走の**前後**に全 10 ノードで `nvidia-smi --query-compute-apps` を記録して切り分ける．
- **A4'（fallback 経路の遅延）**: warmup を外すと，万一 fallback が発火したとき初回に
  モデルロードの 8 秒前後が乗る．→ `fallback_rate` は Iter100 実測 0.0 であり，C2 で監視する．

### 単一レバー (Iter101)

**レバー**: `vram_budget_reallocation` = **`drop_unused_light_model_to_enable_fusion`**．
**何を何から何へ**: 各ノードの起動時に `light_model`（`qwen3.5:4b-q4_K_M`）を warmup して
VRAM に常駐させる挙動を，**warmup しない（常駐させない）**へ変える．

**変更するファイルと設定キー**:
1. `config.yaml` — **新キー `warmup_light_model: false` を 1 つ追加するだけ**
   （省略時・`true` 時は従来と**ビット同一**．Iter99/Iter100 と同じ後方互換の型）．
   **既存行は 1 行も変更しない**．とくに `embedding_fusion_models`（ruri 追加）は
   **Iter100 のまま据え置き，ロールバックしない**．
2. `http_server.py` — `create_app()` の `lifespan`（現 397〜401 行）で，フラグが偽のとき
   `warmup_model(state.ollama_client, state.light_model)` を**呼ばず**，`warmed_models` からも外す．
   構造化ログ `light_model_warmup_skipped` を 1 件出す（発火証拠になる）．
3. `node.py` — `NodeState` 生成箇所（82 行付近）へ
   `warmup_light_model=config.get("warmup_light_model", True)` の配線 1 行と，
   `NodeState.__init__` の引数追加．
4. `tests/test_http_server.py` — 2 件（既定＝`light` と `expert` の 2 回 warmup される／
   `false` のとき `light` が warmup されない）．

**レバーを読むコード行と到達条件（config.yml の必須事項）**: 読むのは
`http_server.py` の `lifespan`（`create_app()` 内）1 か所のみ．到達条件は
**deploy 後の各ノードのアプリ起動**であり，`routing_method` や `confidence_signal_method` の
分岐には依存しない．**したがって Iter16/20/21/22/27 型の「設定を変えたが到達しない」失敗は構造的に起こらない**．
発火証拠は (i) 起動ログの `light_model_warmup_skipped`，(ii) `ollama ps` に `light_model` 行が無いこと．

**運用上の必須手順（これを怠ると変更が反映されない）**: `OLLAMA_KEEP_ALIVE=-1` は期限切れしないため，
**deploy 時に ollama コンテナも再起動**して旧セッションの常駐状態を必ず捨てること．
再起動しないと「前の起動で載った `light_model` が残ったまま」になり得る．

**固定する構成（掃引しない）**: `embedding_fusion_models`（ruri 追加），
`models/domain_classifier.joblib`（Iter100 の 6,656 次元・sha256 `e02c641185c28c…8e72bc1`．**再訓練しない**），
`embedding_instruction`，`embedding_view_concat=true`，`dispatch_gap_threshold=0.36`，`dispatch_gap_max_k=4`，
`dispatch_top_k=2`，`confidence_threshold=0.0`，`aggregation_method=max_confidence`，
`docker-compose.yml` の `OLLAMA_MAX_LOADED_MODELS=4` と `OLLAMA_KEEP_ALIVE=-1`，
評価集合（3,750 問）・訓練集合・`metrics.py`・`classifier.py`・`aggregator.py`・`expert_backend.py`．

**禁止事項（単一レバーの境界）**: 埋め込みモデルの量子化変更，埋め込みの専用ノードへの分離，
`num_ctx` / `OLLAMA_NUM_PARALLEL` / `OLLAMA_MAX_LOADED_MODELS` の掃引，`light_model` の差し替え，
分類器の再訓練・正則化の変更，`dispatch_gap_threshold` の再較正，並列化（`asyncio.gather` 化）．
**`config.yaml` の `light_model` キー自体は消さない**（fallback 経路と
`scripts/run_central_experiment.py` が参照するため．常駐だけを外す）．

### G0（実現性ゲート．本走の前に全項目を満たすこと）

- **G0-a（本レバーの前提の実証．B172 の指定）**: 予備 20 問（単一 15 + 複合 5）を wafl500 入口で実行し，
  **全 10 ノードの ollama ログに `light_model` への `/api/chat`・`/api/generate` が 0 件**であることを示す．
  **偽なら本レバーは成立しない**ので本走へ進まず，次善案（`qwen3-embedding:4b` の量子化縮小）へ切り替える．
- **G0-b**: 全 10 ノードの `ollama ps` に `light_model` 行が無く，必要モデル
  （入口 3 本 = expert-general + qwen3-embedding:4b + ruri，peer 2 本 = expert + qwen3-embedding:4b）が
  **すべて `PROCESSOR = 100% GPU`** であること（部分オフロード＝A2' の検出）．
- **G0-vram-1（B172 (e) の必須項目）**: `nvidia-smi --query-compute-apps` の**実測値**の総和が
  **≤ 12,288 MiB かつ余裕 ≥ 1,500 MiB**（予測: 入口 9,986／余裕 2,302，peer 9,500／余裕 2,788）．
  公称サイズではなく実測値を使う．
- **G0-vram-2（B172 (e) の必須項目）**: **本走の後にも**入口ノードと dispatch 先ノードで
  `ollama ps` の行数が期待本数と一致すること（Iter100 はここで不一致だった）．
- **G0-c**: 予備 20 問の `mean_duration_ms` が **≤ 3,041.7ms**．未達なら本走へ進まず機序を再診断する
  （10 時間の本走を無駄にしないための早期検出．A1'/A2' はここで捕まる）．

### 成功条件・非退行条件（事前登録 / Iter101）

**基準線（精度・レイテンシとも）**: `results/20260928_160921/`（3,750 問，pre-Iter100 構成）．
top1 = **0.833067**，`fallback_rate` 0.0，`dispatch_failure_rate` 0.000267，
`mean_duration_ms` **2534.762**，`compound_domain_top1_accuracy` 0.790411，
`compound_domain_set_recall` 0.567123，`compound_mean_dispatched_count` 1.950685，ECE **0.031774**．
**参照点（同一表現の実測）**: Iter100 本走 `results/20260929_081612/` top1 **0.839467**，ECE 0.039556．
**再現性の床は ±0.25pt．**

- **判定**:
  - **`adopted`**: Δtop1（対 0.833067）**≥ +0.5pt**（top1 ≥ **0.838067**）**かつ** McNemar p < 0.05
    **かつ C1〜C7 をすべて満たす**（とくに **C6: `mean_duration_ms` ≤ 3041.7**）．
  - Δtop1 ≥ +0.5pt だが McNemar p ≥ 0.05 → `adopted_small`（要再現）．
  - **精度条件を満たすが C6 のみ違反** → `rejected`（理由は「運用コストでの不採用」．
    この場合 **A1'/A2' が的中**したとみなし，次レバーは B172 の次善案 (i)
    `qwen3-embedding:4b` の量子化縮小，または (ii) 埋め込みの専用ノード分離へ移す）．
  - |Δtop1| < 0.25pt（＝融合の利得が再現しない）→ `negligible`．この場合は**表現ではなく実行基盤の
    非決定性**を疑い，Iter100 の +0.640pt の再現性そのものを次の論点にする．
  - Δtop1 ≤ −0.5pt または C1〜C5・C7 のいずれか違反 → `rejected`．
  - G0 不合格で本走に至らなかった場合 → `invalid`（実現性）．
- **必須の非退行条件**（1 つでも破れたら `rejected`）:
  - **C1**: per-domain precision/recall 計 20 指標の BH 補正後（q=0.05）の有意退行が **0 件**．
  - **C2**: `fallback_rate` が **0.0** のまま（A4' の監視を兼ねる），`dispatch_failure_rate` ≤ **0.005**．
  - **C3**: レバー発火の証拠（起動ログの `light_model_warmup_skipped` と G0-a〜G0-vram-2）が
    **すべて記録されていること**．
  - **C4**: 複合設問 730 行の top1 が **≥ 0.780411**（基準 0.790411 から −1.0pt 以内）．
  - **C5**: `compound_domain_set_recall` **≥ 0.5400**，`compound_mean_dispatched_count` **≤ 2.10**．
  - **C6（本反復の主眼）**: `mean_duration_ms` **≤ 3041.7**（基準 2534.762 の +20%．B172 のとおり据え置き）．
    **分位点も必ず併記する（B172 (e)）**: 単一層の中央値・p95，複合層の p25・中央値．
    参考値（判定には使わない）: 基準線は単一中央値 838ms / 単一 p95 3,207ms / 複合中央値 9,070ms，
    Iter100 は 1,137 / 9,708 / 43,074ms．**本反復の期待は単一中央値 1,100〜1,200ms・
    単一 p95 3,200〜4,000ms・複合中央値 9,000〜11,000ms**（＝融合の内在コスト +299ms だけが乗った形）．
  - **C7**: ECE ≤ **0.08**．**毎回併記する（B172 (c)）．0.05 を超えたら，C7 を満たしていても
    次イテレーションのレバーを校正側（温度校正の見直し）に立てる**．
    本反復は分類器 artifact を再訓練しないので **ECE は Iter100 の 0.039556 の再現**を見込む．
- **参考値として併記**: Random 0.119467 / BestSingle / Oracle 1.0，`answer_quality`，`end_to_end`，
  実行前後の全 10 ノードの `nvidia-smi` 占有記録（A3' の切り分け）．

### 事前登録する予測 P1〜P4（Iter101）

- **P1（主予測）**: `mean_duration_ms` が **2,600〜3,000ms** に着地する（C6 の内側）．
  点推定 **2,834ms**（基準 2,534.762 + 融合の内在コスト約 299ms）．**80% 区間 2,500〜3,600ms**で，
  区間の上端は C6 をはみ出す．符号の確信は「中程度」．
- **P2**: top1 は **0.8375〜0.8415**（Iter100 実測 0.839467 ± 再現性の床 ±0.25pt）．
  0.833067 近傍まで落ちるなら実行基盤の非決定性を疑う．
- **P3**: G0-vram-1 の実測総和は**入口 9,900〜10,100 MiB・peer 9,400〜9,600 MiB**，
  余裕はいずれも **2,200 MiB 以上**．
- **P4**: 本走中の `/api/chat` の所要時間分布から **8.4 秒付近の第 2 の峰が消える**
  （A3 の観測の裏返し．report-only だが機序の直接証拠になる）．

### 実行フェーズへの申し送り (Iter101)

- **使うホスト**: 本走・予備 20 問・deploy でのみ wafl500〜509 を触る．
  分類器の再訓練は**行わない**ので wafl-ctrl5 での計算も原則不要．必要が生じた場合は wafl-ctrl5 を使う．
- **artifact の保全**: `models/domain_classifier.joblib`（6,656 次元）と
  `models/domain_classifier_pre_iter100_qwen3_4b.joblib` の**両方を保持する**（B172）．
  本反復は前者をそのまま使い，**再訓練も差し替えもしない**．
- **deploy 後・本走前に ollama コンテナを再起動**し，`ollama ps` で常駐状態を確認してから G0 を行う．
- **本走は 3,750 問フルスペック**（恒久ルール (A)）．事前シミュレーションで代替しない．
- 本走の**前後**に全 10 ノードの `nvidia-smi --query-compute-apps` と `ollama ps` を記録する（C3・A3'）．

### Iteration 101 実行済み

**判定: `adopted`**（事前登録の判定規則をそのまま適用．閾値は 1 つも緩めていない）．

#### 変更（単一レバー）

実装 commit `69db633`．`config.yaml` に新キー `warmup_light_model: false` を 1 行追加し，
`http_server.py` の `lifespan` が偽のとき `warmup_model(..., state.light_model)` を呼ばずに
構造化ログ `light_model_warmup_skipped` を出す．`node.py` に配線 1 行，`tests/test_http_server.py` に 2 件．
**`light_model` キー自体・融合表現（6,656 次元）・`models/domain_classifier.joblib`・評価集合は 1 ビットも変えていない**．

#### 結果（本走 `results/20260929_192157/`，3,750 問，約 2 時間 20 分）

| 指標 | 基準線 `20260928_160921` | Iter100 `20260929_081612` | **Iter101** |
|---|---|---|---|
| top1_accuracy | 0.833067 | 0.839467 | **0.840000**（Wilson 95%CI [0.827919, 0.851385]） |
| 単一 / 複合 top1 | 0.843377 / 0.790411 | — | **0.848344 (n=3020) / 0.805479 (n=730)** |
| `mean_duration_ms` | 2534.762 | 9733.350 | **2197.842** |
| `fallback_rate` | 0.0 | 0.0 | **0.0** |
| `dispatch_failure_rate` | 0.000267 | — | **0.000267**（1/3750） |
| ECE / Brier / AUROC | 0.031774 / — / — | 0.039556 | **0.039551 / 0.109451 / 0.831199** |
| `compound_domain_set_recall` | 0.567123 | — | **0.578767** |
| `compound_mean_dispatched_count` | 1.950685 | — | **1.973973** |

- McNemar 対基準線: discordant 110（新のみ正解 68／旧のみ正解 42），chi2 = 5.6818，**p = 0.017142**．
- McNemar 対 Iter100: discordant **わずか 4**（3/1），chi2 = 0.25，p = 0.6171．
- per-domain 20 指標の BH 補正（q=0.05）後の有意退行 **0 件**（生 p で 0.05 を下回るのは
  `education_recall` 0.0303（**改善**方向 25/11）と `business_economics_recall` 0.0442（退行方向 5/15）の
  2 件のみで，いずれも BH 後に有意でない）．

#### 事前登録条件の照合（C1〜C7）

| 条件 | 基準 | 実測 | 判定 |
|---|---|---|---|
| 主基準 | Δtop1 ≥ +0.5pt かつ McNemar p < 0.05 | **+0.6933pt**，p = 0.017142 | **PASS** |
| C1 | BH 後の有意退行 0 件 | 0 件 | PASS |
| C2 | `fallback_rate` = 0.0，`dispatch_failure_rate` ≤ 0.005 | 0.0，0.000267 | PASS |
| C3 | `light_model_warmup_skipped` と G0-a〜G0-vram-2 が全記録 | 全 10 ノードで確認．G0 全項目 PASS | PASS |
| C4 | 複合 top1 ≥ 0.780411 | **0.805479**（基準 +1.507pt） | PASS |
| C5 | set_recall ≥ 0.5400 **かつ** mean_dispatched ≤ 2.10 | **0.578767** / **1.973973** | PASS |
| **C6** | `mean_duration_ms` ≤ 3041.7 | **2197.842**（基準比 **−13.3%**） | **PASS** |
| C7 | ECE ≤ 0.08 | **0.039551**（**0.05 未満**） | PASS |

**C6 の分位点（B172 (e) の必須併記）**．括弧内は基準線 / Iter100．

- 単一層: 中央値 **511ms**（838 / 1,137），p95 **3017.25ms**（3,207 / 9,708）
- 複合層: p25 **7,460ms**（7,888 / 32,029），中央値 **8,732ms**（9,070 / 43,074）
- 全体の裾: p90 8,727 / p95 8,887 / p99 9,003 / **max 9,265ms**（基準線は max 13,097ms，Iter100 は 58,000ms）．
  **20 秒を超える行は 0 件**（Iter100 は 717 件）．

**C7 は 0.05 を超えていない**（0.039551．Iter100 の 0.039556 とビット単位でほぼ同一で，
分類器 artifact を再訓練していないという申告と整合する）．よって **B172 (c) の「0.05 超なら次レバーを
校正側に立てる」条項は発火しない**．

#### 対抗仮説 A1'〜A4' の評価

- **A1'（既に退避済みで効果なし）→ 反証**．レイテンシは Iter100 比で −77.4%，**基準線比でも −13.3%** 改善した．
- **A2'（真因は KV キャッシュ／CUDA コンテキスト）→ 反証**．本走の**後**の `post_run_gpu_status.txt` で
  全 10 ノード・計 21 個の `llama-server` が**すべて `PROCESSOR = 100% GPU`**，入口 5,186+4,314+486 = **9,986 MiB**
  （余裕 2,302），peer 5,186+4,298 = **9,484 MiB**（余裕 2,804）．部分オフロード行は 1 行も無い．
- **A3'（外部競合）→ 反証**．`post_run_gpu_status.txt` の `nvidia-smi --query-compute-apps` に現れた
  プロセスは 21 個すべて `/usr/lib/ollama/llama-server` で，他ユーザーのプロセスは 0 件．
  ただしこれは G0 時点と本走直後の 2 点観測であり，2 時間 20 分の全区間の証明ではない．
  **max が 9,265ms で頭打ちし 20 秒超が 0 件**という分布の締まり方は，間欠的な外部競合とは整合しない．
- **A4'（fallback 経路の遅延）→ 発動せず**．`fallback_rate` = 0.0 で `_fallback_answer()` は一度も通っていない．

#### 事前登録予測 P1〜P4 の的中・外れ

- **P1: 外れ（下振れ）**．予測 2,600〜3,000ms（80% 区間 2,500〜3,600ms）に対し実測 **2,197.8ms** で
  **区間の下側に外れた**．「予測より良かった」で済ませず機序を追った結果が下記 M1・M2 である．
- **P2: 的中**．top1 0.840 は予測区間 0.8375〜0.8415 の内側．
- **P3: 的中**．入口 9,986（予測 9,900〜10,100）・peer 9,484（予測 9,400〜9,600），余裕はいずれも 2,200 MiB 以上．
- **P4: 外れ**．「8.4 秒付近の第 2 の峰が消える」は**起きていない**（下記 M2）．

#### 機序の特定（P1 が下振れした理由・調査フェーズ A3 の誤りの訂正）

**M1: 速くなった分はすべて「生成」ではなく「ルーティングのオーバーヘッド」で説明できる．**
`duration_ms` を `dispatch_gen_time_ms`（生成）とその差（probe＋埋め込み＋集約）に分解した．

| | 生成 mean / median | オーバーヘッド mean / median / p95 |
|---|---|---|
| 基準線 | 1,798.0 / 232ms | **736.8 / 610 / 1,057ms** |
| Iter100 | 8,267.5 / 946ms | 1,467.1 / 305 / 5,791ms |
| **Iter101** | **1,783.2 / 228ms** | **414.6 / 299 / 675ms** |

- 全体平均の差 2,534.8 − 2,197.8 = **336.9ms** に対し，オーバーヘッドの差が **322.2ms** を占め，
  生成側の差は **−14.8ms** にすぎない．`answer_text` の平均長は 142.8 → 142.3 字でほぼ同一なので，
  生成量の変化による交絡ではない．
- すなわち**基準線それ自体が VRAM 逼迫の税を払っていた**．基準線の構成は入口 = light 2,027 + expert 5,186
  + qwen 4,314 = 11,527 MiB で余裕はわずか 761 MiB であり，埋め込み呼び出しが遅く・ばらついていた
  （オーバーヘッド p95 1,057ms → 675ms）．**本レバーは Iter100 の退行を戻しただけでなく，
  Iter100 より前から存在していた潜在的なレイテンシ税も同時に取り除いた**．
- 同時に，**事前登録が使った「融合の内在コスト +299ms」という見積もりが過大だった**ことも確定した．
  この +299ms は Iter100 の単一中央値の増分から取ったものだが，その Iter100 自体が VRAM スラッシングの
  只中にあった．2 本目の埋め込み（ruri，355MB）の真の追加費用は，オーバーヘッドが基準線より 322ms
  **減っている**事実から，数十 ms の桁にとどまる．P1 の点推定 2,834ms は「汚染された内在コスト」と
  「基準線は清潔だという誤った前提」の 2 つの誤りを同じ向きに積んでいた．

**M2: 調査フェーズ A3 の「8.4 秒 = 5.3GB の expert 再ロードの量子」という読みは誤りだった．**

- 8.4 秒付近の峰は Iter101 でも**消えていない**: 生成時間の p95 は k=1 層で 8,494ms，k≥2 層で 8,584ms．
  7 秒以上の行は基準線 642 行 → Iter101 **599 行**でほとんど減っていない．
- 正体は**長文回答の生成**である．生成 ≥7 秒の 517 行の `answer_text` の**中央値は 695 字**，
  対して 7 秒未満の 3,233 行は**中央値 9 字**．回答長の分布が二峰であり，GIN ログの二峰性はこれを
  見ていたにすぎない．`config.yaml` に 8.5 秒級のタイムアウトは無い（`dispatch_timeout_s` は 400.0）．
- **それでもレバーは正しく効いた**．Iter100 で実際に壊れていたのは生成時間の**上位 10%**
  （p90 = 42,431ms，20 秒超 717 行）であり，これは 8.4 秒の量子ではなく**余裕 275 MiB での本物の
  スラッシング**だった．**機序の診断が部分的に誤っていても，レバーの選択（VRAM 予算を 2.0GB 空ける）は
  正しかった**という形になっている．結果が良かったことを機序の正しさの証拠にしてはならない．

#### Iter100 比 p = 0.617 の解釈

3,750 問のうち**予測が食い違ったのは 4 行だけ**である．表現（6,656 次元）も分類器 artifact も 1 ビットも
変えていないのだから，これは事前の想定どおりであり，**むしろ実験の内的妥当性（レバーが表現に触れて
いないこと）の確認**として読むのが正しい．したがって次のように言える．

> **融合表現の精度利得（対基準線 +0.69pt，McNemar p = 0.0171）は実在し，Iter101 はそれを
> 基準線より速い運用条件（2,198ms < 2,535ms）で獲得した．**

Iter100 の +0.640pt と Iter101 の +0.693pt は同一の表現から出た同一の効果であり，
独立な 2 回の再現ではない（差は 4 行）．**利得の大きさの再現性は依然 1 回分の証拠しかない**．
また **Iter100 の「複合設問に効いた」という解釈は B172 (b) で取り下げ済みであり，復活させない**．
今回の層別の見え方（対基準線で単一 0.843377 → 0.848344 = +0.497pt，複合 0.790411 → 0.805479 = +1.507pt）も，B172 (b) が示した
「複合行はもともと決定境界近傍に多く振れやすい」という機序に中立な説明で足りる．

#### 学び

1. **「基準線は清潔である」という前提を検証せずに差分の見積もりを立てると，予測は系統的に外れる．**
   基準線 `20260928_160921` は余裕 761 MiB で走っており，オーバーヘッドに 322ms の税を払っていた．
   P1 の外れはノイズではなく，基準線の汚染と「内在コスト」の汚染が同じ向きに積み上がった結果である．
   **今後，レイテンシを主眼にするレバーでは，基準線側の資源余裕も併記してから区間を引くこと．**
2. **ログの分位点の「峰」を機序に翻訳する前に，その峰の行を直接引いて属性を確かめること．**
   A3 は GIN ログの 8.4 秒の集積を「5.3GB モデルの再ロード」と読んだが，実体は長文回答の生成だった．
   `results.jsonl` の `answer_text` 長と突き合わせれば本走前に棄却できた誤りである．
   **ログの集計だけで機序を確定させず，行レベルの属性と結合する**．
3. **診断が誤っていてもレバーが当たることがある．** 本反復は「VRAM を 2.0GB 空ける」という処置が
   正しく，その正しさの理由（本物のスラッシング）と，事前に信じていた理由（8.4 秒の再ロード量子）が
   異なっていた．**採否判定と機序の確度は別々に記録する**．レバーは `adopted` だが，機序 A3 は反証済みである．
4. **`OLLAMA_MAX_LOADED_MODELS` はスロット数であって容量ではない**（Iter100 の学び 1 の再確認）．
   ノードの VRAM は「常駐モデルの実測合計 + 余裕 ≥ 1,500 MiB」という**会計**で管理する必要がある．
   この運用規則を `docs/d0008` §5.6 に恒久化した．
5. **未使用の常駐物を疑うのは安いレバーである．** 変更は config 1 行 + 実装十数行で，
   再訓練も表現変更も伴わないのに，レイテンシを基準線比 −13.3% にした．
   **表現・分類器を動かす前に，実行基盤に残っている「誰も呼ばない常駐物」を洗うこと．**

#### 次の一手

C6 が解消され ECE も 0.05 未満なので，B172 (c)・B172 の次善案 (i)(ii)（量子化縮小・埋め込みノード分離）は
**いずれも発動しない**．一方 B115 (3) の精度本線は 3 本とも打ち止め（`compound_eval_set_expansion` は
B149 で closed，`cross_domain_training_data_augmentation` は B145 で closed，
`embedding_model_replacement` は `multilingual_e5_large` のみ残るが B167 で期待値が消えている）である．
そこで **Iter81/82/83 で 3 度確認され「本研究の標準工程に組み込む」と宣言した恒常規則**
——**特徴量を変えるレバーの後には必ず閾値較正の反復を 1 回挟む**——を適用する．
Iter99/100 で表現が 2,048 → 6,656 次元へ変わったのに `dispatch_gap_threshold` は 2,048 次元用に
較正した 0.36 のまま凍結されており，**較正の負債が 2 反復ぶん溜まっている**．
本走 `results.jsonl` の `probe_candidates` からの決定論的 replay
（`scripts/replay_dispatch_gap_policy.py`．gt=0.36 で set_recall 0.578767 を**完全再現**し忠実性を確認済み）で，
gt を上げると **set_recall 0.5788 → 0.6747（+9.6pt）**の余地があることを実測した．
詳細と次レバーの選定根拠は backlog **B174**．

## Iteration 100: 異種埋め込み（qwen3-4b と ruri-v3-310m）を連結して融合する

### 調査 (Iter100)

**実施場所の申告**: 本フェーズで**実験ノード wafl500〜509 には一切触れていない**．**wafl-ctrl5 にも触れていない**．
行ったのは (a) 開発ホストでのリポジトリ読み取り，(b) 開発ホスト CPU での既存埋め込みキャッシュの
`numpy` によるノルム計測（読み取りのみ），(c) 公開 Web への read-only な検索・取得 だけである．
モデルの pull・ロード・埋め込み計算・ファイル書き込みは 0 件（`models/`・`data/`・`config.yaml` は無変更）．

**(1) 問い**

- Q1: 異種の埋め込みモデルを**単純連結**して下流分類器に食わせる手法は，先行研究でどう位置づけられるか．
  「単独で劣るモデルが連結では効く」という Iter99 副産物の観測は既知の現象か．
- Q2: 連結時に**スケール調整（正規化・重み付け・PCA）が必須か**．必須なら，それは本レバーに付随する作業か，
  それとも第 2 のレバーか．
- Q3: 埋め込みモデルを 2 本同時に常駐させたとき，**10 ノードの実行基盤に何が起こるか**（B167 要レビュー (C)）．
  失敗の出方は「明示的エラー」か「無言の劣化」か．

**(2) Q1: 連結融合は確立した手法であり，「単独性能では見えない相補性」は先行研究の主張と一致する**

- Akl et al., "Fusion Strategies for Embedding Models: Enhancing Text Representations Across MTEB Tasks
  Through Lightweight and Trainable Ensembles", IEEE, 2025（<https://ieeexplore.ieee.org/document/11167556>）は
  「個々の埋め込みモデルの性能は**タスクによって変動する**ため，複数モデルの融合でより頑健な表現が得られるか」を
  正面から調べた研究である．**融合が有効な条件はモデル間の変動＝相補性**であり，単独スコアの順位ではない．
- Frisoni 系の実務ガイド（Zilliz AI FAQ「How can you combine or ensemble multiple Sentence Transformer
  models or embeddings...」<https://zilliz.com/ai-faq/how-can-you-combine-or-ensemble-multiple-sentence-transformer-models-or-embeddings-to-potentially-improve-performance-on-a-task>）は，
  最も単純な融合として (i) 要素ごと平均，(ii) **連結**を挙げ，
  「**連結は各モデル固有の特徴を保存する**（averaging はしない）が次元が増える」と整理している．
  本件は次元が異なる（2560 vs 768）ので**平均は構造的に取れず，連結が唯一の素朴な選択肢**である．
- "Compressed Concatenation of Small Embedding Models", arXiv:2510.04626, 2025
  （<https://arxiv.org/abs/2510.04626>）は小型埋め込みモデル**複数本の連結**を出発点に据え，
  連結表現を圧縮しても MTEB retrieval の性能の 89% を 48 倍圧縮で保てると報告している．
  同論文は「**連結するモデルを増やすと利得は逓減する**」とも述べており，
  **本反復で 3 本目を足さない**（2 本に留める）判断の根拠になる．
- **ただし，これらはいずれも英語中心の retrieval/MTEB での知見であり，本タスク（日本語 10 ドメインの
  単一ラベル分類，訓練 2,275 行）での外挿は保証されない**．Iter79 の学び 2（公称ベンチは予測力を持たない）を
  ここでも適用し，**採否は本走でのみ決める**．本反復の根拠として一次的に重いのは，
  先行研究ではなく **Iter99 分析フェーズが同一データ・同一モデル定義で実測した CV 0.812308（qwen 単独比
  +0.79pt，McNemar 正確検定 p = 0.0153）**の方である（backlog B167 (e)）．

**(3) Q2: スケール調整は不要．両モデルのビューは既に L2 正規化されていることを実測した**

- 開発ホスト CPU で既存キャッシュの行ノルム中央値を計測した（読み取りのみ）:
  - `data/embcache_train_iter89_qwen3-embedding_4b.npy` (2327, **2560**) → 中央値 **1.0000**（1 ビュー）
  - `data/embcache_train_iter99_ruri-v3-310m.npy` (2275, **1536**) → 中央値 **1.4142 = √2**（2 ビュー連結）
  - `data/embcache_eval_iter99_ruri-v3-310m.npy` (3750, 1536) → 同じく **1.4142**
  - すなわち**両モデルとも 1 ビューあたり L2 ノルム 1.0** に揃っている．連結ブロックの「エネルギー」は
    qwen 側 2.0 / ruri 側 2.0 で**同一**である．StandardScaler も per-block 重みも入れる理由が無い．
- 残る非対称は**次元数**である（qwen 5120 次元にエネルギー 2.0，ruri 1536 次元にエネルギー 2.0．
  1 次元あたりでは ruri が約 3.3 倍濃い）．L2 正則化ロジスティック回帰はブロックの実効的な寄与を
  この比に依存させる（Ng, "Feature selection, L1 vs. L2 regularization, and rotational invariance",
  ICML 2004，<https://icml.cc/Conferences/2004/proceedings/papers/354.pdf> が示すとおり L2 は
  座標のスケールに敏感である）．**しかし Iter99 分析フェーズの CV +0.79pt は，まさにこの素の幾何のままで
  実測された値**である．ここに重み係数を入れて掃引すれば，それは「連結」ではなく「融合重みの探索」という
  **第 2 のレバー**になる．**本反復では一切の再重み付け・PCA・スケーリングを行わない**（B168 (e) に記録）．
  重み付け融合（Zilliz の "Weighted Fusion"）は本反復が `adopted` になった場合の**次反復候補**として
  backlog へ回す．

**(4) Q3: 最大の新規リスクは VRAM ではなく Ollama のモデル常駐上限である（本フェーズで新たに特定）**

- **ノードは既に上限 3 本ぴったりで動いている**．`docker-compose.yml:23` は `OLLAMA_KEEP_ALIVE=-1`
  （アイドル退避の無効化）を設定しているが，`OLLAMA_MAX_LOADED_MODELS` は**設定していない**．
  Ollama 公式 FAQ（<https://docs.ollama.com/faq>）は同変数の既定値を
  「**3 × GPU 数，CPU 推論では 3**」と明記する．各ノードは GPU 1 枚なので**既定 3 本**である．
- `node.py:202` の `embed_query_views()` を呼ぶのは**問い合わせの入口ノードだけ**であり，
  そのノードは現在 `light_model`（/probe ごとに必ず呼ばれる）＋ `expert_model`（自ノードへ送出された場合）
  ＋ `embedding_model` の**ちょうど 3 本**を要求している．**ruri を足すと 4 本目になり，既定上限を超える．**
- **失敗の出方は無言である**．Ollama FAQ は「収まらない場合は**ロードできるまでリクエストを待たせる**
  （queue する）」とし，エラーを返すとは書いていない．さらに `keep_alive=-1` で固定したモデルが
  居座ると，別モデルの待ちが解けない事例が報告されている
  （"Ollama keep_alive -1 hangs every other model: a silent deadlock on a 6 GB GPU", DEV Community,
  <https://dev.to/c1-anderson/ollama-keepalive-1-hangs-every-other-model-a-silent-deadlock-on-a-6-gb-gpu-4ma5>．
  **6GB GPU という条件が本クラスタと一致する**）．
  第三者報告なので断定はしないが，**「精度が出ない」ではなく「タイムアウト／レイテンシ爆発」として出る**
  可能性が高く，非退行条件 C2（`dispatch_failure_rate`）と C6（`mean_duration_ms`）で必ず捕まえる．
- **対処（付随作業．B168 (b) に記録）**: `docker-compose.yml` の ollama サービスに
  `OLLAMA_MAX_LOADED_MODELS=4` を追加する．**旧 config（埋め込み 1 本）では要求モデル数が 3 本のままなので
  この変数は発火せず，挙動はビット同一**である（Iter99 の `embedding_prompt_template` 既定値と同じ型の
  後方互換）．したがって第 2 のレバーではない．
- **VRAM は算術上は収まる**: ruri は q8_0 で **337MB**（Iter99 G0-a で実測済み）．
  ただし 6GB に light 2.4GB ＋ qwen3-embedding:4b 3.26GB ＋ expert-LoRA が既に載らず，
  Iter89 の時点で `light_model` の退避が常態化している（Iter99 G0-e の読み）．
  **337MB の追加は退避の頻度を上げる向きに働く**ので，効くのは C6 である．

### 仮説 (Iter100)

**qwen3-embedding:4b と ruri-v3-310m は，同じ設問に対して異なる誤り方をする（相補的である）．
両者の埋め込みを連結した 6,656 次元空間では，片方だけでは線形分離できない設問の一部が分離可能になり，
top1 が上がる．**根拠は Iter99 分析フェーズの実測で，ruri は**単独では qwen に CV で 8.3pt 劣る**
（0.721319 vs 0.804396）にもかかわらず，**連結すると 0.812308 と qwen 単独を +0.79pt 上回り，
対応のある比較で 救済 34 / 損失 16・McNemar 正確検定 p = 0.0153** だった．
「単独性能で足切りすると見落とす相補性がある」というのが本レバーの主張である．

**対抗仮説（同じ強さで想定する）**:

- **(A1) 冗長性仮説**: CV の +0.79pt は CV の SE 0.83pt と同程度で，**訓練 2,275 行の 5-fold に固有の
  揺らぎ**である．本走 3,750 問では 0 付近に縮む（`negligible`）．
  `Compressed Concatenation of Small Embedding Models` (arXiv:2510.04626) の「連結の利得は逓減する」も
  この向きの示唆である．
- **(A2) 次元増による過学習**: p/n が 2.25 → **2.93** へ上がる．CV は訓練集合内の話なので，
  評価集合 3,750 行への外挿で利得が消える．
- **(A3) 実行基盤仮説**: 精度は上がるが (4) の常駐上限・退避によりレイテンシが退行し，C6 で `rejected` になる．
  これは**効果の不在ではなく運用上の不採用**なので，判定時に区別して記録する．
- **(A4) M7（誤答核はデータ側に由来する．共通誤答 390/2,275 行）が正しい**なら，表現をいくら足しても
  共通誤答核は動かず，利得は核の外側の薄い層に限られる．**M7 は未検証の仮説であり本反復では断定しない**
  （B167 (g)）．本反復は M7 の**間接的な検定**になる: 融合で救済される行が共通誤答 390 行の外側に
  集中するなら M7 と整合する．

### 単一レバー (Iter100)

- **レバー**: `embedding_space_fusion` = **`qwen3_4b_plus_ruri_v3_310m_concat`**
  （config.yml の levers 末尾に Iter99 分析フェーズが追加済み．**新レバーの追加は不要**）．
- **何を何から何へ**: 分類器へ渡す特徴を，**qwen3-embedding:4b の 2 ビュー連結 5,120 次元**から，
  **qwen3-embedding:4b の 2 ビュー（5,120）⊕ ruri-v3-310m の 2 ビュー（1,536）= 6,656 次元**へ変える．
  `config.yaml` の実現方法は下記 B168 (a) の新キー 1 つで行う．
- **連結順序を固定する（train/runtime 不一致の防止．Iter36 型事故の恒久対策）**:
  **`[qwen plain, qwen instructed, ruri plain, ruri instructed]`**．
  順序を引数化せず，`expert_backend.py` の 1 つのヘルパ内にハードコードする
  （`embed_query_views()` が Iter82 以来この方式で train/runtime の一致を保証してきたのと同じ型）．
- **各モデルの prompt template は model card の規定値に固定する（文言は掃引しない）**:
  - qwen3-embedding:4b → 現行の instruct 書式 `"Instruct: {instruction}\nQuery: {text}"`（**変更なし**）．
  - ruri-v3-310m → `"トピック: {text}"`（分類・クラスタリング用途の規定 prefix．
    <https://huggingface.co/cl-nagoya/ruri-v3-310m> の 1+3 prefix scheme．Iter99 で採用済み）．
  - `embedding_instruction`（Iter81 の英文 P1 wording）は**現行値のまま変更しない**．
    ruri 側テンプレートは `{instruction}` を使わないが，`embed()` が instructed ビューを発火させる条件が
    `instruction is not None` なので，キー自体は非 None のまま残す必要がある（Iter99 と同じ機構）．
- **付随作業（いずれも別レバーではない．Iter79/89/99 で確立した型）**:
  1. **分類器の再訓練**（5,120 → 6,656 次元に変わるため必須．訓練集合・モデル定義は 1 バイトも変えない）．
     旧 artifact は `models/domain_classifier_pre_iter100_qwen3_4b.joblib` へ退避する．
  2. **`OLLAMA_MAX_LOADED_MODELS=4`**（上記 (4)．旧 config では発火せずビット同一）．
  3. **deploy 時の ruri の pull**（`tools/node_models.py` が融合モデルも返すよう拡張．
     返さないと `ollama pull` が走らず，本走で入口ノードが 404 を踏む）．
- **固定する構成（1 つも動かさない）**:
  `data/classifier_train_iter94_dedup.jsonl`（2,275 行）・`data/dataset.jsonl`（3,750 行）・
  `embedding_instruction`（Iter81 の P1 wording）・`embedding_view_concat: true`・
  `routing_method=supervised_classifier`・`confidence_threshold=0.0`・`dispatch_candidate_threshold=0.0`・
  `dispatch_top_k=2`・**`dispatch_gap_threshold=0.36`（埋め込み次元が増えれば gap 分布は必ず動くが，
  閾値の再較正は別レバーであり本反復では絶対に触らない）**・`dispatch_gap_max_k=4`・
  `aggregation_method=max_confidence`・`judge_model`・各ノードの `light_model`/`expert_model`・
  `scripts/train_domain_classifier.py` のモデル定義（`LogisticRegression(max_iter=1000, class_weight=None)`
  ＋ `_extract_sample_weights()` ＋ `CalibratedClassifierCV(method='temperature', ensemble=True)`）・
  `classifier.py`・`aggregator.py`・`metrics.py`・`ecoc_head.py`．
  **特徴のスケーリング・per-block 重み・PCA は一切入れない**（上記 Q2）．
  **ドメイン固有の後付け補正は追加しない．棄権／エスカレーション系には触れない．**

**レバーを読むコード行と，そこへ到達する条件（d0004 §4．同型事故 6 回の恒久対策）**

| # | 経路 | 読むコード | 到達確認（1 つでも欠けたら実験不成立） |
|---|---|---|---|
| 1 | 設定 → 10 ノードへのモデル配布 | `tools/node_models.py:13`（融合モデルも返すよう拡張）→ `mise.toml:97-103` の `ollama pull` | 全 10 ノードの `ollama list` に **qwen3-embedding:4b と ruri の両方**の行があること |
| 2 | 設定 → 各ノードの config | `mise.toml` の `rsync config.yaml` | 全 10 ノードで新キーが `grep` で読めること |
| 3 | 設定 → 実行時のクエリ埋め込み | `node.py:202-207` の `embed_query_views(...)`（融合対応ヘルパへ差し替え） | 予備 20 問で `len(query_embedding) == 6656` をログに出して確認．**旧 5,120 次元 artifact に 6,656 次元を食わせれば `predict_proba` が必ず例外を投げる**ので，Iter36 型の無言の不一致はここで落ちる |
| 4 | artifact → 10 ノード | `mise.toml` の `models/` rsync | 全ノードで `domain_classifier.joblib` の sha256 一致かつ **`n_features_in_ == 6656`** |
| 5 | **ブロック順序と template の到達（本反復の最大の新規リスク）** | 融合ヘルパ（順序ハードコード）．呼び出し側は `node.py` ・`scripts/train_domain_classifier.py`・`tools/smoke_check.py` の **3 箇所すべて** | (a) 単体テストで 4 ブロックの順序と各 template を検証，(b) 訓練時に**実際に使ったモデル名と template 文字列を標準出力へ印字**し journal へ転記，(c) 訓練特徴の先頭 1 行で **4 ブロックが互いに一致しないこと**と **各ブロックの L2 ノルムが 4 本とも ≈1.0** であることを実測（一致＝template 未適用，ノルム逸脱＝別モデルを引いている） |
| 6 | **Ollama の常駐上限（本フェーズで新規特定）** | `docker-compose.yml` の ollama `environment` | 入口ノードで `ollama ps` に **4 本すべてが並ぶ**こと．並ばない／`100% GPU` でない行がある場合は C6 の退行要因として明記する |
| 7 | 実験 → 指標 | `metrics.py` 無変更 | `total_questions == 3750` かつ `compound_domain_question_count == 730` |

### 事前ゲート G0〜G2（判定規則は結果を見る前にここで固定する）

**すべて wafl-ctrl5 で行う．wafl500〜509 は deploy・予備 20 問・本走でのみ触れる（絶対条件 (B)）．**

- **G0（実現性．Iter99 で大半が済んでいるので軽い）**
  - **G0-a**: wafl-ctrl5 で qwen3-embedding:4b と `hf.co/Targoyle/ruri-v3-310m-GGUF:Q8_0` の**両方**が
    `/api/embeddings` に 200 を返し，次元がそれぞれ **2560 / 768** であること．いずれか外れたら不合格．
  - **G0-b（ブロック整合）**: 訓練集合の先頭 1 行で 4 ブロックを作り，**各ブロックの L2 ノルム ≈ 1.0
    （許容 ±0.02）**，**連結後 6,656 次元**，**4 ブロックが相互に不一致**であること．
  - **G0-c（常駐上限）**: wafl-ctrl5 で埋め込み 2 本を交互に呼んでも 2 本とも `ollama ps` に残り
    `100% GPU` であること．deploy 後は入口ノードで #6 を確認する．
  - **G0 不合格時**: 原因が (4) の常駐上限なら `OLLAMA_MAX_LOADED_MODELS` を上げて再試行してよい
    （調達手段の確定であって設計選択ではない）．それ以外で不合格なら `invalid`（実現性）として
    本走を行わず分析フェーズへ渡す．**本反復内で第 3 の埋め込みモデルを探しに行かない．**

- **G1（CV．report-only．足切りには使わない）**
  `wafl-ctrl5` で `data/classifier_train_iter94_dedup.jsonl`（2,275 行）**のみ**を使い，
  **本番の `scripts/train_domain_classifier.py` の特徴生成経路**で 5-fold StratifiedKFold
  （`random_state=42`）の accuracy / macro-F1 を，**融合 6,656 次元と qwen 単独 5,120 次元の 2 腕について
  同一 run 内で**測る．評価集合は一切見ない．
  - **G1 の第一の役割は効果量の推定ではなく，Iter99 分析フェーズのアドホック計算の再現検査である**．
    **融合 0.812308・qwen 単独 0.804396 を ±0.002 で再現すること**を期待値として事前登録する．
    再現しない場合，効果の有無ではなく**本番訓練経路とアドホック計算の乖離**を疑い，原因を特定してから進む．
  - **キャッシュを再利用する場合は必ず `id` で join すること．**
    `data/embcache_train_iter89_qwen3-embedding_4b.npy` は 2,327 行で訓練集合 2,275 行と**行数が違い，
    かつ `.meta.json`（id 一覧）が存在しない**．行順の暗黙の一致を仮定してはならない（B168 (d)）．
    安全側として**本番経路で両モデルとも埋め込みを取り直し，`.meta.json` に id を書く**ことを既定とする．

- **G2（replay．検出力の事前確定．report-only）**
  評価 3,750 行について**両モデルの 2 ビュー埋め込みを wafl-ctrl5 で計算**し
  （`data/embcache_eval_iter100_{qwen3-embedding_4b,ruri-v3-310m}__{p0,p1}.npy` ＋ `.meta.json`），
  旧 artifact（5,120 次元）と新 artifact（6,656 次元）の `predict_proba` argmax を replay する．
  **discordant 行数 n_d と McNemar の有意境界 `1.96·√n_d / 3750` を本走前に確定して記録する．**
  - 既存 `embcache_eval_qwen3-embedding_4b.npy` は **3,435 行**で現行評価集合 3,750 行と一致しない
    （Iter99 で G2 が実施できなかった原因）．**今回は 3,750 行ぶんを取り直す**．
  - **n_d ≥ 30 を最低条件**とし，一桁なら「効果なし」ではなく**設定未到達**を既定の解釈とする．
  - replay の予測 Δ も記録するが，**この値を見て成功条件を書き換えてはならない**（B131 以来の運用）．

- **本走は G0 合格なら必ず実施する**（2026-09-23 絶対条件 (A)）．G1/G2 の数値が悪いことを理由に省略しない．

**唯一の安全弁（B167 要レビュー (B) への回答を兼ねる．B168 (c)）**:
G1 の **融合 − qwen 単独の CV Δ が −2.0pt 以下**の場合に限り，本走前に原因究明へ戻る．
**Iter99 の −5.0pt より厳しくした根拠**: 本反復の融合特徴は qwen 単独特徴の**真の上位集合**であり，
L2 正則化下で特徴を足して精度が下がることはあっても，**2.0pt（CV の SE 0.83pt の 2.4 倍，
事前期待 +0.79pt からは 3.4 SE）も下がるのは，ブロック順序の不一致・行 join の誤り・
キャッシュの取り違えといった実装破綻としか解釈できない**．
Iter99 の安全弁は「真の性能差」を拾ってしまったが（B167 (B)），上位集合という構造上，
本反復では同じ取り違えが起こりにくい．
**安全弁が発火した場合は，停止する前に必ず (i) ブロック単位の ablation（qwen のみ / ruri のみ / 融合）と
(ii) 腕間 AGREE の算出を義務付ける**（B167 (B) で提案された既定運用の初適用）．

### 事前登録する予測 P1〜P6（Iter100）

- **P1（主予測）**: 本走で **Δtop1 ≥ +0.5pt かつ McNemar p < 0.05**（＝ `adopted`）．
  **点推定 +0.3pt，80% 区間 −0.5 〜 +1.2pt，符号の確信は「中程度」**．
  **点推定が採用閾値 +0.5pt を下回っていることを明示しておく．最頻の着地は `negligible` である**
  （Iter89 の実績 CV +5.29pt → 本走 +2.94pt から，本走は CV より縮む向きを既定で見込む）．
- **P2**: G1 の CV で **融合 0.812308 ± 0.002・qwen 単独 0.804396 ± 0.002 を再現**する（測定系の再現性検査）．
- **P3**: G2 replay の予測 Δ と本走実測 Δ の乖離が **≤ 0.5pt**（Iter89 は 0.12pt）．
- **P4**: 新 artifact の `n_features_in_` == **6656**，全 10 ノードで sha256 一致．
- **P5**: `mean_duration_ms` が基準 2534.762 から **+50 〜 +400ms** の範囲に収まる
  （ruri は 310M と小さく HTTP 往復 2 回の追加が主．Iter82 が 1 回追加で −84ms だった実績から上振れは
  退避の増加に帰属させて読む）．**範囲外なら (4) の常駐上限・退避を第一の説明として調べる．**
- **P6**: 本走で融合により救済された行のうち，**Iter99 の共通誤答 390 行（訓練側）に相当する構造の行**は
  少数に留まる（A4 / M7 と整合する向き）．**これは report-only の観察であり，判定には使わない．**

### 成功条件・非退行条件（事前登録 / Iter100）

**基準線（本走）**: `results/20260928_160921/`（3,750 問）．
top1 = **0.833067**（Wilson 95%CI [0.820791, 0.844660]），`fallback_rate` 0.0，
`dispatch_failure_rate` 0.000267，`mean_duration_ms` **2534.762**，
`compound_domain_top1_accuracy` **0.790411**，`compound_domain_set_recall` **0.567123**，
`compound_mean_dispatched_count` **1.950685**，ECE **0.031774**，Brier 0.110943，
`single_domain_top1_accuracy` 0.843377．**再現性の床は ±0.25pt．**

- **判定**: Δ ≥ +0.5pt（top1 ≥ **0.838067**）かつ McNemar p < 0.05 → **`adopted`**．
  Δ ≥ +0.5pt だが p ≥ 0.05 → `adopted_small`（要再現）．|Δ| < 0.25pt → `negligible`．
  Δ ≤ −0.5pt または下記 C1〜C7 のいずれか違反 → `rejected`（artifact と `config.yaml` をロールバック）．
  G0 不合格で本走に至らなかった場合 → `invalid`（実現性）．
  **精度は基準内だが C6 のみ違反した場合も `rejected` だが，「表現としては中立以上／運用コストで不採用」と
  判定理由に明記する**（A3 と効果の不在を混同しないため）．
- **必須の非退行条件**（1 つでも破れたら `rejected` としてロールバック）:
  - **C1**: per-domain precision/recall 計 20 指標の BH 補正後（q=0.05）の有意退行が **0 件**．
  - **C2**: `fallback_rate` が 0.0 のまま，`dispatch_failure_rate` ≤ **0.005**．
  - **C3**: レバー発火の証拠（上表 #1〜#7 と G0-a〜c）がすべて記録されていること．
  - **C4**: 複合設問 730 行の top1 が **≥ 0.780411**（基準 0.790411 から −1.0pt 以内）．
  - **C5**: `compound_domain_set_recall` **≥ 0.5400**，`compound_mean_dispatched_count` **≤ 2.10**．
  - **C6**: `mean_duration_ms` ≤ **3041.7**（基準 2534.762 の +20%）．
    **config.yml の note「埋め込みが 2 回になるので計画フェーズで上限を見直すこと」への回答**:
    上限は**据え置く**．理由は (i) Iter82 で埋め込み呼び出しを 1 回増やしたとき実測は
    2301.4 → 2217.1ms と**むしろ短縮**しており，埋め込みは所要時間の支配項ではない，
    (ii) 追加分は 310M モデルの往復 2 回で，予測は +50〜+400ms（P5）＝上限まで 507ms の余裕がある，
    (iii) 上限を緩めると (4) の常駐上限による退避スラッシングという**本反復固有の失敗モードを見逃す**．
    **C6 は本反復では効果の検定ではなく実行基盤の健全性検査として機能する．**
  - **C7**: ECE ≤ **0.08**（基準 0.031774）．
- **参考値として併記**: Random 0.119467 / BestSingle / Oracle 1.0 と `answer_quality` / `end_to_end`．

### 実行フェーズへの申し送り（Iter100）

- **使うホスト**: 埋め込み計算・分類器訓練・G0〜G2 はすべて **wafl-ctrl5**（絶対条件 (B)）．
  **wafl500〜509 は deploy と本走（および予備 20 問）でのみ触れる．**
- **触ってよいファイル**:
  - `config.yaml` — **新キー `embedding_fusion_models` の追加のみ**（既存行は 1 行も変えない．
    とくに `embedding_model` / `embedding_instruction` / `embedding_view_concat` / `dispatch_gap_threshold`）．
    ```yaml
    # Iter100 (embedding_space_fusion): 省略時は空リストとして扱われ，挙動は pre-Iter100 とビット同一．
    embedding_fusion_models:
      - model: hf.co/Targoyle/ruri-v3-310m-GGUF:Q8_0
        prompt_template: "トピック: {text}"
    ```
    config.yaml のスキーマ変更は B116 (3) の枠内（埋め込み系レバーに限る事前承認）で扱う．
  - `expert_backend.py` — 融合対応ヘルパの追加（既存 `embed_query_views()` の**既定挙動は温存**．
    融合リストが空なら現行と**ビット同一**の返り値になること）．
  - `node.py` / `scripts/train_domain_classifier.py` / `tools/smoke_check.py` — 新キーの受け渡し 3 箇所．
  - `tools/node_models.py` — 融合モデル名も返すよう拡張（これを忘れると #1 が落ちる）．
  - `docker-compose.yml` — ollama の `environment` に `OLLAMA_MAX_LOADED_MODELS=4` を 1 行追加．
  - `models/domain_classifier.joblib`（再訓練．旧 artifact は
    `models/domain_classifier_pre_iter100_qwen3_4b.joblib` へ退避）．
  - `data/embcache_*`（新規キャッシュ．**必ず `.meta.json` に id 一覧を書く**）・`tests/`．
  - **`data/dataset.jsonl`・`data/classifier_train_iter94_dedup.jsonl`・`metrics.py`・`classifier.py`・
    `aggregator.py`・`ecoc_head.py` は 1 バイトも変更しない．**
- **禁止事項（単一レバー原則の境界）**:
  - **prompt template の文言を掃引しない**（qwen は現行値，ruri は `トピック: ` の 1 通りだけ）．
  - **per-block の重み・スケーリング・PCA・次元削減を入れない**（Q2．入れたら第 2 のレバーになる）．
  - **3 本目の埋め込みモデルを足さない**（arXiv:2510.04626 の逓減の示唆．2 本に留める）．
  - **`dispatch_gap_threshold` を再較正しない**（0.36 のまま．gap 分布は必ず動くが report-only）．
  - **`LogisticRegression` の `C` を触らない**（p/n が 2.25 → 2.93 に上がるが，正則化強度の探索は別レバー）．
- キャッシュ名には**モデル名とビュー ID を必ず含める**（Iter81 の教訓．旧キャッシュを無言で読む事故の防止）．
- 埋め込み 2 本は `expert_backend` 内で**逐次 await**になる（`embed_query_views()` の現行実装と同じ）．
  **`asyncio.gather` 化などの並列化はしないこと**．レイテンシ最適化は本レバーの検証対象を濁らせる．

### 実装・実験 (Iter100)

**変更したファイル（commit `9faf8fe`，実装フェーズで完了済み）**: `config.yaml`（新キー
`embedding_fusion_models` 追加，既存行は無変更），`docker-compose.yml`（ollama `environment` へ
`OLLAMA_MAX_LOADED_MODELS=4` を 1 行追加），`expert_backend.py`（`embed_query_views()` を拡張し
固定順序 `[qwen plain, qwen instructed, ruri plain, ruri instructed]` で連結．
`concat_views=False` との併用は `ValueError`），`node.py` / `scripts/train_domain_classifier.py` /
`tools/smoke_check.py`（新キー配線 3 箇所），`tools/node_models.py`（融合モデルも `ollama pull` 対象に
含めるよう拡張），`tests/test_expert_backend.py`（新規 6 テスト）．
`uv run pytest tests/test_expert_backend.py` は **14 passed**（既存 8 + 新規 6）．
分類器は `scripts/train_domain_classifier.py` で再訓練し，旧 artifact は
`models/domain_classifier_pre_iter100_qwen3_4b.joblib` へ退避済み．

**レバー発火の証拠（C3．上表 #1〜#7 の到達確認）**:
- #1: 全 10 ノード（wafl500〜509）の `ollama list` に `qwen3-embedding:4b` と
  `hf.co/Targoyle/ruri-v3-310m-GGUF:Q8_0` の両方が存在することを確認．
- #4: `models/domain_classifier.joblib` の `n_features_in_` = **6656**（5120+1536 と一致）．
  全 10 ノードの同ファイルの sha256 が `e02c641185c28cf1e4c5dde2e4197613ab5e67ae9c4b988a7fdb4c3a68e72bc1`
  で完全一致（アプリコンテナ内 `/app/models/domain_classifier.joblib` を直接照合）．
- #5: ブロック順序・template 適用は単体テスト 6 件で検証済み（上記 pytest 結果）．
- #7: `metrics.py` 無変更のまま `total_questions == 3750` かつ
  `compound_domain_question_count == 730` を実測（下記メトリクス参照）．
- #6（Ollama 常駐上限）は **未充足**: 実行後の `docker exec expert-mesh-ollama-1 ollama ps`（wafl500，
  入口ノード）は下記 3 本のみで，**`light_model`（`qwen3.5:4b-q4_K_M`）が常駐していない**．
  ```
  NAME                                     SIZE      PROCESSOR    UNTIL
  expert-mesh-general-lora:latest          5.3 GB    100% GPU     Forever
  qwen3-embedding:4b                       4.4 GB    100% GPU     Forever
  hf.co/Targoyle/ruri-v3-310m-GGUF:Q8_0    355 MB    100% GPU     Forever
  ```
  事前調査 (4) で特定していた「物理 VRAM 12,288 MiB に 4 モデル（実測 5,186+4,308+484=9,978 MiB）が
  収まらず，残り 2,310 MiB に対し `light_model` は約 2,400〜2,500 MiB 必要で常駐から追い出される」
  という懸念が実機で実体化した．これは対抗仮説 **A3（実行基盤仮説）の的中**である．

**実行後の GPU 占有記録（全 10 ノード，`nvidia-smi` / `docker ps`）**: 他ジョブの割り込みは**確認されず**
（`namit` 等の外部プロセス 0 件，各ノードとも `expert-mesh-app-1` と `expert-mesh-ollama-1` の 2 コンテナ
のみ稼働）．
| ノード | VRAM使用 (MiB) / 12288 | GPU util |
|---|---|---|
| wafl500 | 10040 | 0% |
| wafl501 | 2027 | 0% |
| wafl502 | 5260 | 0% |
| wafl503 | 2606 | 0% |
| wafl504 | 3267 | 0% |
| wafl505 | 3399 | 0% |
| wafl506 | 3673 | 0% |
| wafl507 | 3673 | 0% |
| wafl508 | 4387 | 0% |
| wafl509 | 2721 | 0% |

**メトリクス（`results/20260929_081612/results.jsonl`，3,750 問，`metrics.py --json`）**:

| 指標 | 基準線 (`20260928_160921`) | Iter100 実測 | Δ |
|---|---|---|---|
| top1_accuracy | 0.833067 | **0.839467** | **+0.640pt** |
| single_domain_top1_accuracy (n=3020) | 0.843377 | 0.847682 | +0.431pt |
| compound_domain_top1_accuracy (n=730) | 0.790411 | 0.805479 | +1.507pt |
| compound_domain_set_recall | 0.567123 | 0.578767 | +1.164pt |
| compound_mean_dispatched_count | 1.950685 | 1.973973 | +0.023 |
| fallback_rate | 0.0 | 0.0 | 0 |
| dispatch_failure_rate | 0.000267 | 0.0008 | +0.000533 |
| ECE | 0.031774 | 0.039556 | +0.007782 |
| mean_duration_ms（全体） | 2534.762 | **9733.350** | **+7198.588 (+284%)** |
| mean_duration_ms（単一ドメイン, n=3020） | 1111.341 | 2190.824 | +1079.482 (+97%) |
| mean_duration_ms（複合, n=730） | 8423.436 | 40936.679 | +32513.243 (+386%) |

**McNemar 検定（対応のある比較，`metrics.compute_mcnemar_test`，id で完全一致する 3,750 行）**:
discordant_a_only（基準線のみ正解）= 44，discordant_b_only（Iter100 のみ正解）= 68，
discordant_pairs（n_d）= **112**，chi2（連続性補正）= 4.723214，**p = 0.029758**（p < 0.05）．
McNemar は net で Iter100 が 24 問多く正解に転じたことを示す．

**C1（per-domain 20 指標，BH q=0.05）**: `metrics.compute_domain_recall_mcnemar_test`（10 domain）＋
`metrics.compute_domain_precision_fisher_test`（10 domain）の p 値 20 個に
`metrics.apply_benjamini_hochberg(q=0.05)` を適用．**BH 補正後の有意退行は 0 件**（生 p 値最小は
recall/business_economics の p=0.044171 と recall/education の p=0.030260 だが，いずれも BH 補正後は
非有意）．→ **PASS**．

**C1〜C7 合否一覧**:

| 条件 | 基準 | 実測 | 合否 |
|---|---|---|---|
| C1 | per-domain 20指標 BH有意退行 0件 | 0件 | **PASS** |
| C2 | fallback_rate=0.0 かつ dispatch_failure_rate≤0.005 | 0.0 / 0.0008 | **PASS** |
| C3 | レバー発火の証拠（#1〜#7） | #1,#4,#5,#7 確認．#6 は不充足（light_model 非常駐） | **一部不充足（#6）** |
| C4 | 複合730行 top1 ≥ 0.780411 | 0.805479 | **PASS** |
| C5 | set_recall≥0.5400 かつ mean_dispatched≤2.10 | 0.578767 / 1.973973 | **PASS** |
| C6 | mean_duration_ms ≤ 3041.7 | 9733.350 | **FAIL** |
| C7 | ECE ≤ 0.08 | 0.039556 | **PASS** |

**判定材料の要約（判定自体は分析フェーズの担当）**: top1 Δ=+0.640pt（≥+0.5pt 基準を充足）かつ
McNemar p=0.029758（<0.05）．C1〜C5・C7 は PASS だが **C6 は基準の 3.2 倍で明確に FAIL**（事前登録の
判定規則により，精度基準を満たしても C6 単独違反で `rejected` となる分岐が Iter100 の事前登録節に
明記されている）．A3（実行基盤仮説）が的中したことを示す一次証拠として，`ollama ps` の
`light_model` 非常駐を上記に記録した．

**限界（明記）**: 基準線 `results/20260928_160921/` 実行時の GPU 占有状況（同時実行プロセスの有無）は
本フェーズでは遡って確認できていない．基準線側で偶発的に外部競合が無かったことを前提に比較している．

**`state.json`**: `status` を `"running"` に戻した．

### Iteration 100 実行済み

**変更**: 分類器へ渡す特徴を qwen3-embedding:4b の 2 ビュー連結 5,120 次元から，
qwen3-embedding:4b 2 ビュー ⊕ ruri-v3-310m 2 ビュー = **6,656 次元**へ変更（固定順序
`[qwen plain, qwen instructed, ruri plain, ruri instructed]`，スケーリング・重み付け・PCA なし）．
付随作業は分類器の再訓練（`n_features_in_`=6656）・`OLLAMA_MAX_LOADED_MODELS=4`・ruri の pull 拡張のみ．
実装は commit `9faf8fe`．本走は `results/20260929_081612`（3,750 問，外部競合なしを実行前後に確認済み）．

**判定: `rejected`（理由は「運用コストでの不採用」．表現としての効果は肯定される）**

事前登録した判定規則をそのまま機械的に適用した結果である．後知恵で閾値を緩めていない．

- 精度側は **`adopted` の条件を満たしていた**: Δtop1 = **+0.640pt**（0.833067 → 0.839467，
  閾値 +0.5pt 以上）かつ McNemar **p = 0.029758**（< 0.05）．
- しかし **C6（`mean_duration_ms` ≤ 3041.7）が 9733.350 で明確に FAIL**（基準の 3.2 倍）．
  事前登録節に「精度は基準内だが C6 のみ違反した場合も `rejected`．ただし『表現としては中立以上／
  運用コストで不採用』と理由に明記する」と書いてあるので，そのとおり `rejected` とする．
- C1・C2・C4・C5・C7 は PASS．C3 は #6（Ollama 常駐上限）が不充足だが，これは C6 違反と同一事象の
  別表現（`light_model` の追い出し）であり，独立した 2 件目の違反として数えない．

**1. 効果の実質性: 3 経路が同じ方向・同程度で一致しており，ノイズではない**

- 本走 Δ の標準誤差は McNemar の discordant から `√n_d / n = √112 / 3750 = **0.282pt**`．
  95%CI は **[+0.09, +1.19]pt** で，再現性の床 ±0.25pt を下回る部分をほぼ含まない．
- 3 つの独立な推定値 — **G1 CV +1.011pt（p=0.001769，SE 0.83pt）／G2 replay 予測 +0.507pt／
  本走実測 +0.640pt** — は互いに **1 SE 以内**に収まる（G1 は本走から +0.45 SE，G2 は −0.47 SE）．
  真値がおよそ **+0.5 〜 +0.7pt** の共通効果であるという読みと矛盾しない．
  事前登録 P3（replay と本走の乖離 ≤ 0.5pt）は **0.133pt で的中**，P1 の点推定 +0.3pt は
  実測 +0.640pt をやや下振れして外したが 80% 区間 −0.5〜+1.2pt には収まる．
- すなわち **対抗仮説 A1（CV 固有の揺らぎ）・A2（次元増による過学習）はいずれも支持されない**．
  一方 **A3（実行基盤の退行）が的中**した．判定を決めたのは A3 だけである．

**2. 伸びが複合に集中して見えることの機序: 「複合に効く」とは言えない（層別では有意差なし）**

本走のデータで層別 McNemar を計算した（開発ホスト，読み取りのみ）．

| 層 | n | a_only | b_only | n_d | net | 正味 Δ | 正確二項 p | discordant 率 |
|---|---|---|---|---|---|---|---|---|
| 単一 | 3020 | 34 | 47 | 81 | +13 | +0.430pt | 0.1821 | 2.68% |
| 複合 | 730 | 10 | 21 | 31 | +11 | +1.507pt | 0.0708 | 4.25% |

- **どちらの層も単独では p < 0.05 に届かない**（有意なのはプールした全体のみ）．複合の +1.507pt は
  **discordant 31 行・net 11 行**に依存しており，1 行あたり 0.137pt 動く粒度である．
- 「複合の方が効いている」かを直接検定すると，**discordant の向きの構成比の層間差は
  Fisher p = 0.393 で有意でない**（単一 47/81 = 58.0% 対 複合 21/31 = 67.7%）．
  一方 **discordant「率」自体は複合が有意に高い（4.25% 対 2.68%，Fisher p = 0.0293）**．
- したがって **「相補的な 2 空間が複合設問で特に効く」という解釈は，本走のデータでは支持も反証も
  できない**．観測されている複合の大きな Δ は，**複合行がもともと決定境界の近くに多く分布していて
  表現をどう変えても振れ幅が大きい**（＝母数 730 と高い discordant 率の合成）という，機序に中立な
  説明だけで足りてしまう．**この解釈を採らず，「複合で伸びた」と書かないこと**．
  複合 argmax が変わった行は 50/730，dispatch 集合が変わった行は 125/730 で，
  いずれも「振れやすさ」の側の数字と整合する．

**3. ECE の悪化（0.031774 → 0.039556，相対 +24.5%）: C7 内だが申し送る**

C7（≤ 0.08）に対してはまだ 2 倍の余裕があり，単独では何の判断も引き起こさない．
ただし**方向は悪化で，かつ機序に説明がつく**: 特徴次元が 5,120 → 6,656 に増えて p/n が 2.25 → 2.93 に
上がり，`CalibratedClassifierCV(method='temperature')` の温度 1 パラメータでは高次元側の
過信を吸収しきれていない可能性がある．**融合の方向を今後も追う（＝次元をさらに増やす）なら
この劣化は単調に積み上がる**ので，次反復以降は ECE を毎回併記し，**0.05 を超えたら C7 の 0.08 を
待たずに校正側を単一レバーとして立てる**ことを申し送る．今回は判定に使わない．

**4. C6 違反の帰属: レイテンシ退行は融合の内在コストではなく VRAM 常駐の副作用である**

`duration_ms` の分位点を取ると，退行の性格がはっきり分かれる．

| 層 | 統計量 | 基準線 | Iter100 |
|---|---|---|---|
| 単一 | 中央値 | 838 | **1137** |
| 単一 | p95 / max | 3207 / 13097 | **9708 / 42530** |
| 単一 | 平均 | 1111 | 2191 |
| 複合 | p25 / 中央値 | 7888 / 9070 | **32029 / 43074** |
| 複合 | 平均 | 8423 | 40937 |

- **単一ドメインの中央値の増分は +299ms** で，これは事前登録 P5（+50 〜 +400ms ＝ 310M モデルへの
  HTTP 往復 2 回ぶん）の**範囲内に収まっている**．つまり**融合そのものの内在コストは予測どおり
  小さい**．平均を 2191ms へ押し上げているのは p95 以降の裾（9.7 秒・最大 42.5 秒）である．
- 複合は裾ではなく **p25 = 32 秒**で分布全体が移動している．複合経路は入口ノードで
  `light_model` を繰り返し呼ぶため，**embedding 2 本と `light_model` が VRAM を奪い合う
  ロード／退避のスラッシングが毎問発生している**と読める（B171 の実測: 5,186+4,308+484 = 9,978 /
  12,288 MiB，残り 2,310 MiB に対し `light_model` は 2,400〜2,500 MiB 必要）．
- **この分解が次レバーの根拠になる**．C6 違反は ruri の**重みそのもの**ではなく**置き場所**の問題で
  あり，容量を 200〜500 MiB 空けるか埋め込みを別ノードへ寄せれば，
  **表現（＝精度 +0.640pt）を 1 ビットも変えずに C6 を満たせる可能性がある**．

**限界（明記）**: 基準線 `results/20260928_160921/`（Sep 28 16:09 取得）実行時の GPU 占有状況は
遡って確認できていない．C6 の比較は「基準線側に外部競合が無かった」という未検証の前提の上にある．
ただし基準線の複合中央値 9,070ms は Iter89 以降の同構成の実績と整合しており，前提が崩れている兆候は無い．

**学び**

1. **`OLLAMA_MAX_LOADED_MODELS` は「スロット数」の上限であって「VRAM 容量」の上限ではない．**
   4 に上げてもモデルが物理的に載らなければ Ollama は黙ってロードと退避を繰り返す．
   計画フェーズは常駐**本数**の制約を正しく見つけたのに，**容量**の制約を検査しなかった．
2. **G0（実現性ゲート）に穴があった．** G0-c は「埋め込み 2 本が `ollama ps` に `100% GPU` で残る」
   までしか見ておらず，`light_model` と `expert_model` を含めた**実行時に同時に必要な全モデルの
   VRAM 合計が物理容量に収まるか**を見ていない．**今後の G0 には
   「実行時に必要な全モデルの VRAM 実測値の総和 ≤ 物理 VRAM，かつ本走後に `ollama ps` の行数が
   期待本数と一致」を必須項目として入れる**（B172 に申し送り済み）．
3. **無言の劣化は平均値ではなく分位点で診る．** 平均 `mean_duration_ms` だけを見ると「全体が 3.2 倍
   遅い」に見えるが，単一層の中央値は予測どおり +299ms しか増えていない．
   **中央値と p95 を分けて記録していなければ，レバーの内在コストと基盤の副作用を分離できなかった**．
   今後 C6 系の条件には中央値も併記する．
4. **層別の Δ の大小をそのまま機序の証拠に読まないこと．** 複合 +1.507pt は単一 +0.431pt の 3.5 倍
   だが，層間の向きの差は有意でなく（p=0.393），複合の discordant 率が高いこと（p=0.029）だけで
   説明がつく．**n の小さい層ほど Δ は大きく見える**という当たり前の罠に，今回あやうく
   「相補的な空間は複合設問に効く」という物語を付けるところだった．
5. **`rejected` の理由を 2 種類に分けて書くことには実務上の意味がある．** 今回の融合は
   「効かなかった」のではなく「今の VRAM 予算では置けなかった」のであり，
   **同じレバーの値を，置き場所を変えて再挑戦する価値がある**．理由を分けていなければ
   `embedding_space_fusion` は反証済みとして閉じられていた．

**ロールバック**: 事前登録どおり `models/domain_classifier.joblib`（6,656 次元）と `config.yaml` の
`embedding_fusion_models` を pre-Iter100 構成へ戻す．ただし**次イテレーションが同じ融合構成を
VRAM 配置だけ変えて再走する**ため，退避済み artifact
`models/domain_classifier_pre_iter100_qwen3_4b.joblib` と融合 artifact の**両方を保持する**
（実施はロールバック／次反復の実装フェーズが行う）．

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

