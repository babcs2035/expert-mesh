# journal — expert-mesh research-cycle

新しいものを常に先頭へ挿入する（逆時系列）．書式はグローバル skill（~/.claude/skills/research-cycle/SKILL.md）を参照．

---

## Iteration 2: embedding ルーティング(方式A)への切替による複合ドメイン被覆の検証

### 調査 (Iter2)

対象レバー `routing_method`（方式 B `self_report` → 方式 A `embedding`）の切替が複合行被覆・信号の質に
効くかを，コード実装と先行研究の両面から調査した．

**問い**
- Q1: 方式 A(embedding) は confidence をどう算出し，方式 B(self_report) と実装上どう違うか．
- Q2: embedding 類似度ベースのルーティング/confidence は self_report より較正が良いのか（先行研究）．
- Q3: Iter1 の制約（`confidence_threshold=0.5` のゲート・複合行での confidence 飽和）は embedding でも
  起き得るか（cosine 類似度の分布特性）．
- Q4: レイテンシ・コストのトレードオフ．

**分かったこと（コード実装の確認: 最重要）**
- 方式 A の算出経路: requester が **full query** を embed し（node.py:143，nomic-embed-text，prefix なし），
  各 expert ノードは起動時に **ドメイン名の単語そのもの**（"medical"/"legal"/"general"）を embed する
  （http_server.py:184-186，prefix なし）．/probe では `estimate_embedding_confidence` = `cosine_similarity`
  を `(sim+1)/2` で [0,1] に再スケールして返す（router.py:114-144）．**LLM 呼び出しは無い**（cosine のみ）．
- 方式 B との差: B は light_model(9b) がドメイン別プロンプトで自己申告スコアを生成し，general は専用の
  **反転プロンプト**（`_build_general_confidence_prompt`, router.py:24-40）で「専門知識なしで答えられる度合い」を
  測る catch-all 設計になっている．**方式 A にはこの general 反転ロジックが無く**，general ノードも単語
  "general" との cosine を計算するだけ．方式 A に切り替えると general の fallback セマンティクスが変質する
  （重要な非対称性）．
- **(sim+1)/2 の再スケールにより，閾値 0.5 はちょうど cosine=0.0 に対応する**．テキスト埋め込みは異方性
  （anisotropy）で対ペア cosine がほぼ正になるため，実運用ではほぼ全ノードが 0.5 を超える見込み．
  → Iter1 の「ゲートが medical(0.2) の dispatch をブロックする」問題は **逆転し，むしろ全ノード通過・
  over-dispatch 側に振れる**可能性が高い（閾値がほぼ無効化する）．
- **構造的キャップ（Iter1 と同型の no-op リスク）**: 単一レバー原則により今回 `dispatch_top_k=1` は固定．
  top_k=1 では複合行で 1 ノードしか /dispatch しないため，`compound_covered_domain_count` は routing_method を
  変えても **複合行数=4 が上限**で，Iter1 ベースライン(4)から原理的に増えない（実データで cap=4 を確認済み；
  results/20260720_171532）．つまり Iter1 の主基準（compound coverage）は routing_method 単独では動かせない．
- nomic-embed-text は **task instruction prefix 必須**（search_query: / search_document: / classification: /
  clustering:）だが，現行コードは query・domain どちらにも prefix を付けていない → 較正劣化の既知の落とし穴．
- probe レイテンシは実測 **~750ms/ノード**（wafl503/medical，34 問，min733/median752/max1203ms）．config.yaml の
  コメント「20-40s」は VRAM 常時確保(KEEP_ALIVE=-1)・GPU 化(B6)より前の **stale な値**．embedding 化の
  レイテンシ削減効果は ~750ms/node 程度に留まり，しかも query の embed は方式に依らず requester 側で 1 回発生する．

**分かったこと（先行研究，出典付き）**
- 自己申告 confidence は過信で有効性が限定的，一方 embedding-similarity は不正確な出力の識別に強い弁別力を
  示す（"Confidence Scoring for LLM-Generated SQL in Supply Chain Data Extraction", amazon.science PDF, 2024）．
  → 方式 A が信号の弁別力で B に優位という一般傾向を支持する（Iter1 で B の飽和・過信が実証済みなのと整合）．
- ただし埋め込みルーティングの閾値は較正依存で脆い: embedding モデルを差し替えると絶対類似度スケールが変わり，
  以前チューニングした閾値が無効化する（SurePrompts "Semantic Router: Embedding-Based Routing Without Calling
  an LLM"）．→ 現行の 0.5 固定閾値は方式 A 用に較正されておらず，較正し直しが必要という含意．
- Semantic Router のベストプラクティスは，ルートを **複数の代表発話(utterances)集合**で定義し query との類似度を
  測る（Aurelio AI semantic_router docs; "Semantic Routing for ... 5G Core Network", arxiv 2404.15869, 2024）．
  現行実装の「ドメイン名 1 単語」でのルート定義は最小構成で信号が弱いと見込まれる．
- nomic-embed-text は非対称タスク用の prefix を前提に学習されている（"Nomic Embed", arxiv 2402.01613; HF model
  card nomic-ai/nomic-embed-text-v1.5）．prefix 無し＋英単語(domain)対日本語(query)のクロスリンガル比較は
  較正上さらに不利になり得る．

**次フェーズ（rc-planner）への示唆**
- 【最重要】Iter1 と同型の落とし穴回避: top_k=1 固定のままでは `compound_covered_domain_count` は構造的に 4 で
  頭打ちのため，これを主基準にすると routing_method は必ず no-op になる．**主基準は「信号の質」に置き換える**べき．
  候補指標: (a) 単一ドメイン 30 問の `top1_accuracy` が self_report ベースライン(0.9706)以上（非退行），
  (b) 複合行での `selected_domain` の妥当性，(c) `misrouting_rate`，(d) probe confidence 分布の弁別力
  （生 cosine と再スケール後値を probe_candidates に記録して比較），(e) probe レイテンシ実測差．
- 較正の観点: `(sim+1)/2` により閾値 0.5 は事実上ほぼ全通過になる懸念があるため，実験では probe_candidates の
  confidence 分布を必ず観察し「ゲートがブロックする/しない」の挙動反転を確認する．非退行として general の
  over-dispatch（Iter1 の general-008 型の余分 dispatch）が悪化しないかを見る．
- general の扱い: 方式 A には反転プロンプトが無く general が単語比較になるため fallback セマンティクスが変わる点を
  分析で明示する．単一ドメイン general 行の精度低下・複合行での general リークに注意．
- コスト/レイテンシ: probe レイテンシ削減は GPU 化後は ~750ms/node 程度と限定的（config のコメント 20-40s は
  stale）．「レイテンシ大幅削減」を売り文句にせず，精度・較正の質で評価するのが妥当．
- 実装上の落とし穴（人間判断素材・backlog 候補）: nomic-embed-text の task prefix 未付与は既知の較正劣化要因だが，
  prefix 付与はコード変更（embed 経路）になり単一レバー・config-only 原則と衝突する．まず prefix 無しの現状のまま
  方式 A を config-only で評価し，劣化が観測されたら prefix 起因かの切り分けを次段に回すのが妥当．この論点を
  backlog に上げる材料として提示する．

### 計画 (Iter2)

**単一レバー**: `config.yaml` の `routing_method` を `self_report`（方式 B・現行既定）→ `embedding`（方式 A）へ変更．
config-only の 1 値変更のみ（コード確認済み: /probe が `state.routing_method` で B/A を分岐し（http_server.py:220），
どちらも既存実装で動作．query の embed は routing_method に依らず requester 側で常時発生し（node.py:143），
各 expert ノードは起動時に domain 名を embed 済み（http_server.py:184）．コード変更は不要）．
**実装上の注意**: `routing_method` は各 expert ノードが起動時に config から読み込む state 値（http_server.py:220 は
`state.routing_method` を参照）のため，切替の反映には config を配布してノードを再起動する必要がある（`mise run deploy`）．
固定する構成（Iter1 最良＝現行 config.yaml のまま）: `dispatch_top_k=1`，`confidence_threshold=0.5`，
`embedding_model=nomic-embed-text`．レバー以外は一切動かさない．

**仮説**:
- H1（信号の質）: embedding cosine ベースの confidence は，self_report の自己申告より弁別力（期待ドメイン node と
  非期待 node の confidence マージン）が同等以上になる（先行研究の一般傾向）．
- H2（構造的キャップ）: `dispatch_top_k=1` 固定のため複合行は 1 ノードしか dispatch されず，
  `compound_covered_domain_count` は routing_method を変えても 4（＝複合行数）で頭打ち（調査で cap=4 を実データ確認）．
  → 複合被覆は本イテレーションの主基準にしない（構造的に動かせないため観測のみとし，判定には使わない）．
- H3（較正の反転リスク）: `(sim+1)/2` 再スケールで閾値 0.5 が cosine=0.0 相当になり，埋め込みの異方性でほぼ全ノードが
  閾値超になる．self_report で保たれていた「単一ドメイン行は 1 ノードのみ dispatch」が崩れ over-dispatch
  （Iter1 の general-008 型リーク）が悪化する懸念がある．また prefix 未付与＋英単語(domain)対日本語(query)の
  クロスリンガル比較で単一ドメイン精度が退行する懸念がある．

**評価コードの追加**: なし（config-only 単一レバー原則を維持）．判定に用いる指標はすべて既存 `metrics.py` の
`--json` 出力と，Iter1 で追加済みの `results.jsonl` フィールド（`probe_candidates`，`dispatched_domains`）からの
オフライン集計で得られる．
- 弁別マージン: 各行で max(期待ドメイン node の confidence) − max(非期待 node の confidence)．probe_candidates から算出．
- 単一行 over-dispatch: 単一ドメイン 30 問の `dispatched_domains` 長の平均．
- probe レイテンシ: 各ノードの log_event(`probe_done`, `local_inference_ms`) から取得（判定には使わず観測）．
- raw cosine の記録（生 cosine と再スケール後の比較）は protocol/http_server のコード変更が必要なため今回は行わず，
  再スケール後 confidence の分布のみで弁別を評価する．

**成功条件（ベースライン＝self_report k=1: results/20260720_171532，34 問．実測値を併記）**:
- 主基準（信号の質＝embedding 採用可否）: 全 34 行の弁別マージン平均が正，かつ positive-margin 行割合 ≥ 0.971
  （baseline 33/34=0.971），mean margin ≥ 0.60（baseline 0.676，ノイズ相当の低下のみ許容）．
  → embedding が self_report と同等以上の弁別力を持つことの条件．
- 非退行基準（割れば embedding 棄却）: `single_domain_top1_accuracy` ≥ 0.933（baseline 0.967=29/30，30 問中
  misroute 2 問以内）．`top1_accuracy` ≥ 0.91（baseline 0.971），`misrouting_rate` ≤ 0.088（baseline 0.029）．
  embedding は決定的（固定埋め込みの cosine のため run 間ノイズほぼ 0）なので，これらを割れば構造的劣化と判定．
- コスト保護基準（割れば embedding 棄却）: 単一ドメイン 30 問の平均 `dispatched_domains` 数 ≤ 1.2
  （baseline 1.000）．`(sim+1)/2` の閾値崩壊による over-dispatch（general リーク悪化）の監視．
- 観測のみ（判定に使わない）: `compound_covered_domain_count`（構造的に top_k=1 で 4 cap，baseline 4），
  `compound_domain_top1_accuracy`（baseline 1.0），probe レイテンシ実測（~750ms/node → embedding は cosine のみで
  短縮見込み．「レイテンシ削減」は売り文句にせず記録のみ）．
- 採用判定: 主基準を満たし，かつ非退行・コスト保護をすべて満たせば embedding 採用（デフォルト化を検討）．
  いずれか 1 つでも割れば embedding 棄却・self_report 維持．prefix 未付与起因が疑われる劣化なら prefix 切り分けを
  次段（backlog B7）へ引き継ぐ．

**prefix 付与のスコープ判断（今回は含めない）**: nomic-embed-text の task prefix（search_query: / search_document:
等）付与は node.py:143（query embed）と http_server.py:184（domain embed）の両 embed 経路のコード変更が必要で，
config-only 単一レバー原則と衝突する．まず prefix 無しの現状のまま embedding を config-only で評価し，退行
（特に単一ドメイン精度低下）が観測された場合に prefix 起因かの切り分けを次段階の課題（backlog B7）として実施する
（調査提案どおり）．prefix をスコープに含める判断はしていないため，本イテレーションでユーザー確認は不要．

### 実装 (Iter2)

**実行した変更**: `config.yaml` の `routing_method: self_report` を `routing_method: embedding` へ 1 行変更．
それ以外のキー（`dispatch_top_k=1`，`confidence_threshold=0.5`，`embedding_model=nomic-embed-text`，
`nodes.*` 等）は無変更．`git diff -- config.yaml` で単一行差分のみであることを確認済み．コード変更は無し
（計画どおり，http_server.py:220 の `state.routing_method` 分岐は既存実装のまま利用）．

**検証**:
- `uv run pytest tests/ -v`: 78 件全 PASS（`test_router.py` の embedding 関連テスト
  `test_estimate_embedding_confidence_rescales_similarity_to_unit_range` 等を含む，config-only 変更のため
  影響なしを確認）．
- `uv run ruff check .`: All checks passed．
- `uv run ruff format --check .`: 10 ファイル（build_dataset.py, expert_backend.py, http_client.py,
  http_server.py, metrics.py, router.py, tests/test_build_dataset.py, tests/test_metrics.py,
  tests/test_run_experiment.py, tests/test_show_logs.py）で reformat 差分あり．いずれも本イテレーションの
  変更（config.yaml のみ）とは無関係な既存差分であり，今回のスコープ外として手を加えていない．

**反映状態**: `routing_method` は各 expert ノードが起動時に読み込む state 値のため，config.yaml の変更だけ
ではまだ実機ノードへ反映されていない．次フェーズ（実験）で `mise run deploy` を実行し，config 配布・ノード
再起動を行った上で実験を開始する必要がある．

### 実験 (Iter2)

**デプロイ**: `mise run deploy` を実行．3 ノード（wafl500/general，wafl502/legal，wafl503/medical）へ
`config.yaml`（`routing_method: embedding`）を配布し，`docker compose up -d --force-recreate app` で
app コンテナを再起動（ollama コンテナは常時稼働のまま，モデル再 pull 不要でキャッシュヒット）．
healthcheck は 1 回リトライ後（wafl503 が起動直後で応答なし）に全ノード healthy．

**反映確認**（重要）: デプロイ後，3 ノードそれぞれで次の 2 通りの方法により `routing_method: embedding` の
反映を確認した．
- `ssh <host> "grep -E '^routing_method:' config.yaml"`: 3 ノードとも `routing_method: embedding`．
- 手動 `/probe` リクエスト（`request_id=manual-check-1`）を各ノードへ送信し，`docker compose logs app` の
  `probe_done` イベントで `routing_method` フィールドを確認: wafl500/wafl502/wafl503 すべて
  `"routing_method": "embedding"`（実行時に読み込まれた state 値そのものを確認，config ファイルの記述だけ
  でなく実際の挙動で裏取り）．手動 probe は実験用の `request_id` と異なるため，本番実験の confidence
  キャッシュには影響しない．

**実行**: `mise run start`（`--node-id wafl500`, `--dataset data/dataset.jsonl`, 34 問）．コンテナ内で
detached 実行し，`run_experiment.log` をポーリングして進捗を確認．

**結果**:
- 結果ディレクトリ: `results/20260720_181842/results.jsonl`（34 行，全問完走．`used_fallback` / `dispatch_failed`
  はいずれも 0 件）．
- 実行時間: 約 6 分 49 秒（`results/20260720_181842` ディレクトリ作成 18:18:42 → `results.jsonl` 書き込み完了
  18:25:32．前回ベースライン self_report 実行（config.yaml コメント記載，34 問で約 46 分）と比較して大幅に
  短時間．計画（調査フェーズ）で見込んだ「probe あたり ~750ms/node，LLM 呼び出し無し（cosine のみ）」と整合．
- `dispatched_domains` は全 34 行が長さ 1（`dispatch_top_k=1` 固定のため，調査フェーズで見込んだ構造的
  cap どおり．閾値 0.5 通過ノードが複数あっても top_k=1 では 1 ノードのみ dispatch されるため，over-dispatch
  は観測されなかった）．
- `probe_candidates` の confidence 値はサンプル行で概ね 0.70〜0.73 の狭い帯域に集中（例: medical-001 の
  3 ノード confidence は 0.708 / 0.709 / 0.724）．計画で懸念した「`(sim+1)/2` 再スケールによる閾値 0.5 の
  ほぼ全通過」と整合する分布が観測された（解釈・弁別マージンの定量評価は次の分析フェーズで行う）．
- ノードログ確認: 3 ノードとも `docker compose logs app` に error/exception/traceback/OOM の該当行なし．

**メトリクス集計**: 本フェーズでは実施せず（次の分析フェーズで `mise run analyze` および `metrics.py` を実行）．

### 分析(実行) (Iter2)

対象: embedding（`results/20260720_181842/results.jsonl`，34 行）／self_report ベースライン
（`results/20260720_171532/results.jsonl`，34 行）．以下はいずれも実測の生数値であり，判定は行わない．

**1. 弁別マージン**（`probe_candidates` から集計．各行で期待ドメイン node の confidence 最大値 − 非期待
ドメイン node の confidence 最大値）:
- embedding: mean margin = -0.0040，positive-margin 率 = 0.5294（18/34）
- self_report: mean margin = 0.6765，positive-margin 率 = 0.9706（33/34）

**2. `metrics.py --json` 出力**:

| 指標 | embedding | self_report |
|---|---|---|
| top1_accuracy | 0.5294 (18/34相当) | 0.9706 |
| misrouting_rate | 0.4706 | 0.0294 |
| single_domain_question_count | 30 | 30 |
| single_domain_top1_accuracy | 0.5000 | 0.9667 |
| compound_domain_question_count | 4 | 4 |
| compound_domain_top1_accuracy | 0.75 | 1.0 |
| precision_recall_per_domain.general | precision=0.4444, recall=0.4000 | precision=1.0, recall=0.9 |
| precision_recall_per_domain.legal | precision=0.4444, recall=0.2857 | precision=1.0, recall=0.9286 |
| precision_recall_per_domain.medical | precision=0.625, recall=0.7143 | precision=0.9167, recall=0.7857 |
| compound_coverage.compound_covered_domain_count | 3 | 4 |
| compound_coverage.compound_expected_domain_total | 8 | 8 |
| compound_coverage.compound_domain_set_recall | 0.375 | 0.5 |
| compound_coverage.compound_domain_coverage_ratio_mean | 0.375 | 0.5 |
| compound_coverage.compound_domain_jaccard_mean | 0.375 | 0.5 |
| compound_coverage.compound_mean_dispatched_count | 1.0 | 1.0 |
| fallback_rate | 0.0 | 0.0 |
| dispatch_failure_rate | 0.0 | 0.0 |
| mean_duration_ms | 11634.03 | 12681.35 |

**3. 単一ドメイン30問の平均 `dispatched_domains` 長**:
- embedding: 1.0000（30/30，全行 dispatch 数 1）
- self_report: 1.0000（30/30，全行 dispatch 数 1）

**4. `single_domain_top1_accuracy`（単一ドメイン30問限定，selected_domainがexpected_domainsと一致する行の割合）**:
- embedding: 0.5000（15/30）
- self_report: 0.9667（29/30）

### 分析(解釈) (Iter2)

対象: embedding（`results/20260720_181842`）vs self_report ベースライン（`results/20260720_171532`）．
計画 (Iter2) の成功条件と実測値を突き合わせて判定し，why を probe_candidates の生値から検証した．

**1. 基準ごとの判定**

- 主基準（信号の質・embedding 採用可否）: **未達（決定的）**．
  - positive-margin 率 = 0.529（基準 ≥ 0.971）→ 大幅未達．
  - mean margin = -0.0040（基準 ≥ 0.60）→ 実質ゼロ，かつ僅かに負．弁別マージンは存在しないに等しい．
- 非退行基準（割れば棄却）: **3 指標すべて未達（決定的）**．
  - `single_domain_top1_accuracy` = 0.500（基準 ≥ 0.933）．
  - `top1_accuracy` = 0.529（基準 ≥ 0.91）．
  - `misrouting_rate` = 0.471（基準 ≤ 0.088）．
  - baseline（self_report: 0.967 / 0.971 / 0.029）から破滅的に劣化しており，基準値との差は後述のノイズ幅を桁で上回る．
- コスト保護基準（割れば棄却）: **達成（ただし限定的な意味）**．
  - 単一ドメイン30問の平均 dispatch 数 = 1.000（基準 ≤ 1.2）．
  - ただしこれは `dispatch_top_k=1` の構造キャップで dispatch が 1 ノードに固定されるためであり，
    「閾値ゲートが正常に効いた」ことの証拠ではない．実際には後述のとおり閾値 0.5 は 102/102 の probe で
    全通過しており（H3 前半の予測どおりゲートは崩壊），over-dispatch が現れなかったのは top_k=1 が
    覆い隠しているだけである（top_k を上げれば全ノードへ dispatch する over-dispatch が顕在化する）．

→ 主基準・非退行がいずれも決定的に未達．**採用条件（主基準達成かつ非退行・コスト保護すべて達成）を満たさず，
embedding は棄却が妥当**．コスト保護のみ達成だが，1 つでも割れば棄却の設計であり結論は動かない．

**2. ノイズか構造的劣化かの判断: 構造的劣化と断定．追加再実行は不要．**

- embedding の confidence は `(sim+1)/2` の cosine のみで算出され，埋め込み推論はサンプリングを伴わず決定的．
  同一 query・同一 domain 語に対し run 間の値はほぼ完全に再現する（journal 実験フェーズで medical-001 の
  3 ノード値 0.708/0.709/0.724 を実測，本分析でも同値を確認）．よって run 間ノイズはほぼ 0 であり，
  0.529 という top1 は「たまたま悪い run」ではなく方式・設定の性質そのものである．
- 劣化幅の大きさ: baseline との差（top1 で -0.44，misroute で +0.44）は，Iter1 で self_report 2 run 間に
  観測された揺らぎ（selected_domain は 34 行完全一致＝実質ノイズ 0）を桁違いに超える．ノイズでは説明不可能．
- 以上より**再現性確認のための追加 run は価値が乏しく，提案しない**（決定的処理という性質上，同じ数値が出る）．

**3. why（最重要）: 「confidence の弁別力消失」が根本原因．調査フェーズの懸念 (a)(b)(c)(d) が複合して顕在化．**

probe_candidates の生値を全 34 行×3 ノード（n=102）で集計した根拠:
- **全 confidence 値が [0.6677, 0.7370]（幅 0.069，std 0.0138）の極狭帯域に潰れている**（懸念 (d) を定量確認）．
  102/102 が閾値 0.5 を通過＝ゲート無効化も確認．異方性（anisotropy）で対ペア cosine がほぼ正の狭域に
  集まるという調査フェーズの予測どおりの分布．
- **勝者マージン（top1−top2 confidence 差）は median 0.0055・mean 0.0075．34 行中 24 行が < 0.01，33 行が < 0.02**．
  ほぼ全行が「3 ノードほぼ同点で僅差の順位が付いただけ」の状態であり，順位付けが実質的にドメイン信号を
  担っていない．
- 決定的な所見: **誤答行の勝者マージン平均（0.0103）は正答行（0.0051）より大きい**．誤答は「僅差で惜しく負けた」
  のではなく，「無関係な cosine の順位でむしろ自信ありげに別ノードが勝った」ケースを含む．cosine 順位が
  真のドメインに対してほぼ無情報（noise）であることを示す．single_domain top1=0.500 は 3 ドメイン一様ランダム
  (≈0.33) をわずかに上回る程度で，残存信号はごく僅か．
- ドメイン別の崩れ方: general の recall が 0.9→0.40 と特に大きく落ちた．self_report の general は
  `_build_general_confidence_prompt` の反転（catch-all）プロンプトで「専門知識なしで答えられる度合い」を測って
  いたが，embedding の general は単に単語 "general" との cosine を取るだけで catch-all セマンティクスが消失する
  （調査フェーズが指摘した非対称性の実データ確認）．
- 上記帯域圧縮の要因は調査フェーズの (a) task prefix 未付与，(b) ドメイン名 1 単語という弱いルート定義，
  (c) 日本語 query 対英単語 domain のクロスリンガル比較，が複合したものと解釈する．いずれも cosine の
  使える動的レンジを縮め，(d) の分布集中＝弁別力消失に帰結している．

**4. 採否の見立て（最終判定は次フェーズ rc-reflector）**

- 数値が示す結論は明確: **現行 config（prefix 無し・単語ルート・閾値 0.5・top_k=1）での embedding は棄却，
  self_report を維持**．主基準と非退行が決定的に未達であり，ノイズではなく設定・方式の構造的劣化．
- ただしこれは「embedding が原理的に劣る」ことの証明ではなく，「config-only の最小構成では使い物にならない」
  ことの実証である．調査フェーズ提案どおり，劣化が prefix 起因か切り分ける価値はある（backlog B7）．
  ただし prefix 付与・複数 utterance でのルート定義はいずれもコード変更を伴い config-only 単一レバー原則の
  外側になるため，rc-reflector で「棄却して次レバー（confidence_threshold）へ進む」か「B7 を人間判断素材として
  上げる」かを決めるのが妥当．
- レバー収束の観点: Iter1（dispatch_top_k 棄却）に続き，config-only で触れる範囲では信号の質を self_report 以上に
  できないことが 2 例目として示された．真のボトルネックは confidence 信号そのものの較正であり，config 値の
  範囲を出た改良（prefix・多 utterance・ドメイン別プロンプト整備）か，research_frontier のドメイン拡張へ
  重心を移す判断材料になる．

### Iteration 2 実行済み

**判定**: `routing_method` レバー（方式 B `self_report` → 方式 A `embedding`）は **棄却**（現行 config の
最小構成では信号の質が self_report に決定的に劣る）．config.yaml の `routing_method` は交絡回避のため
ベースライン（`self_report`）に戻した（`git diff -- config.yaml` が空であることを確認済み）．

**実行した変更**: 単一レバー `config.yaml` の `routing_method: self_report` → `embedding` を 1 行変更
（config-only，コード変更なし）．3 ノードへ `mise run deploy` で配布・app 再起動し，`probe_done` イベントの
`routing_method` フィールドで実機反映（`"embedding"`）を裏取りした．34 問を実行（`results/20260720_181842`，
全問完走・fallback/dispatch_failed 0 件）．判定後にベースライン（`self_report`）へ復帰させた．

**結果（embedding: results/20260720_181842 ／ self_report ベースライン: results/20260720_171532，各 34 問）**:
- 主基準（信号の質・embedding 採用可否）: **決定的未達**．positive-margin 率 0.529（基準 ≥ 0.971），
  mean margin -0.0040（基準 ≥ 0.60，実質ゼロで僅かに負）．弁別マージンは存在しないに等しい．
- 非退行基準（割れば棄却）: **3 指標すべて決定的未達**．`single_domain_top1_accuracy` 0.500（基準 ≥ 0.933，
  baseline 0.967），`top1_accuracy` 0.529（基準 ≥ 0.91，baseline 0.971），`misrouting_rate` 0.471
  （基準 ≤ 0.088，baseline 0.029）．
- コスト保護基準: 達成（単一ドメイン 30 問の平均 dispatch 数 1.000 ≤ 1.2）だが，これは `dispatch_top_k=1` の
  構造キャップで dispatch が 1 ノードに固定されるためで，「閾値ゲートが正常に効いた」証拠ではない．実際は
  閾値 0.5 が 102/102 probe で全通過しゲートは崩壊しており，limited な意味しか持たない．
- ノイズか構造的劣化か: embedding の confidence は `(sim+1)/2` の cosine のみで決定的（サンプリングなし），
  run 間ノイズはほぼ 0．劣化幅は self_report の run 間揺らぎ（selected_domain 34 行完全一致）を桁で上回る．
  **構造的劣化と断定，追加再実行は不要**．

**学び（非自明）**:
- embedding の confidence 値は全 34 行 ×3 ノード（n=102）で [0.6677, 0.7370]（幅 0.069，std 0.0138）の
  **極狭帯域に潰れ，弁別力が実質消失**していた．勝者マージン（top1−top2）は median 0.0055 で 34 行中 24 行が
  < 0.01．誤答行の勝者マージン平均（0.0103）が正答行（0.0051）より大きく，cosine 順位が真のドメインに対して
  ほぼ無情報（noise）である．single_domain top1=0.500 は 3 ドメイン一様ランダム（≈0.33）を僅かに上回る程度．
- 帯域圧縮の要因は，調査で懸念した (a) nomic-embed-text の task prefix 未付与，(b) ドメイン名 1 単語という
  弱いルート定義，(c) 日本語 query 対英単語 domain のクロスリンガル比較，(d) 方式 A に general の反転
  （catch-all）プロンプトが無い非対称性，が複合して顕在化したものと解釈できる．general の recall が
  0.9→0.40 と特に大きく落ちたのは (d) の実データ確認である．
- config-only で触れる範囲では，Iter1（dispatch_top_k）に続き **2 例連続で信号の質を self_report 以上に
  できなかった**．真のボトルネックは confidence 信号そのものの較正であり，config 値の範囲外の改良
  （prefix・多 utterance・ドメイン別プロンプト整備）か research_frontier のドメイン拡張が次の重心候補になる．
- これは「embedding が原理的に劣る」証明ではなく「config-only の最小構成では使い物にならない」実証である．
  prefix 起因かの切り分けはコード変更を伴い単一レバー原則の外側になるため，B7 に未着手のまま残す．

**次イテレーションの方針**: 残る config-only レバーは優先順位 3 の `confidence_threshold`（values: [0.3, 0.5, 0.7]）
のみ．levers 優先順位どおりこれを次の単一レバーとする（Iter3）．今回 embedding 実験で閾値 0.5 が事実上
無意味化していた新知見（self_report 方式では閾値ゲートは機能している）と，B5 で記録した「confidence_threshold を
下げると general の過信リークが悪化するトレードオフ」を踏まえ，rc-planner は fallback 率・general 過信リークを
非退行基準に組み込んで数値化すること（詳細は backlog B8）．config-only の 3 レバーを試し切った後は，停止条件の
判断か research_frontier（新規専門ドメイン追加）への移行を rc-planner が検討する．

---

## Iteration 1: 複合ドメイン行の被覆率指標追加による dispatch_top_k 検証

### Iteration 1 実行済み

**判定**: `dispatch_top_k` レバーは **棄却**（効果が限定的でボトルネックは別要因）．config.yaml の値は
交絡回避のためベースライン（`1`）に戻した．

**実行した変更**: 単一レバー `dispatch_top_k` を `1`→`2`．計測基盤として `run_experiment.py` に観測用
フィールド（`dispatched_domains`, `probe_candidates`）を追加，`metrics.py` に `compute_compound_coverage_metrics`
を追加（いずれも B2/B3 でユーザー承認済み・集約ロジック本体は不変）．B4 の既存テスト import 崩れも修正．

**結果（ベースライン `k=1`: results/20260720_171532/ ／ `k=2`: results/20260720_172557/，各 34 問）**:
- 主基準 `compound_covered_domain_count>=6`: **未達**．実測 4→5（+1 のみ，目標 +2 に届かず）．
  `compound_domain_set_recall` 0.5→0.625，`compound_domain_jaccard_mean` 0.5→0.625．
- コスト保護: `compound_mean_dispatched_count<=2.0` は達成（1.0→1.25）．ただし「単一ドメイン 30 問の
  dispatch 数が 1 のまま」は**未達**．medical-006, general-008 の 2 件が dispatch 数 2 に増加．両ランで
  probe confidence が完全一致のためノイズではなく確定的な副作用（最終選択は confidence 最大のため
  誤答/正答自体は不変で，増えた dispatch は無駄になっている）．
- 非退行 `top1_accuracy>=0.97`・`misrouting_rate<=0.03`: **達成**．両ラン 0.9706 / 0.0294 で完全同一，
  `selected_domain` は 34 行すべて k=1/k=2 で一致．

**学び（非自明）**:
- 計画時のメカニズム予測（`selected_domain` 不変・非退行）は的中したが，**`confidence_threshold=0.5` という
  ゲートの存在を見落としていた**．複合 4 行のうち 3 行（compound-001,002,004）は medical の自己申告
  confidence が 0.2 と低く閾値を越えられず，`dispatch_top_k` を上げても追加 dispatch が発火しない．
  唯一 medical=0.9 で閾値超だった compound-003 のみ被覆が 1→2 に改善した．つまり被覆改善の +1 は
  「閾値を越えた行だけ」で説明でき，`dispatch_top_k` 単独では複合行被覆を伸ばせない．
- **真のボトルネックは confidence 信号の質と閾値**であり，dispatch の並列数ではない．k=3 は複合行の
  期待ドメインが最大 2 つのため k=2 と同一結果になる見込みで，追加検証の価値は低い（k=3 は棄却）．
- 副作用として，閾値をむやみに下げると general の過信リーク（general-008 のような単一行での余分な
  dispatch）が悪化するトレードオフが実データで確認できた．confidence_threshold を動かす場合はこの
  リーク悪化を非退行基準に組み込む必要がある．

**次イテレーションの方針**: レバーを confidence 信号そのものを変える `routing_method`（config levers 優先
順位 2 番目・方式 A embedding）へ移す．self_report の自己申告 confidence が過信/較正不良で複合行の
弁別に効かないことが本イテレーションで実証されたため，embedding 類似度ベースの confidence 算出に
切り替えて複合行被覆と非退行を比較する（詳細は backlog B5）．

---

**単一レバー**: `dispatch_top_k`（`config.yaml` の `dispatch_top_k`）を `1`（現行既定）→ `2` へ変更．
確認のため `3` も回してよいが，実機ノードは 3 台・複合行の expected は 2 ドメインのため `k=2` と `k=3` は
これらの行で同一結果になる見込み．固定する構成: `routing_method=self_report`，`confidence_threshold=0.5`，
`embedding_model=nomic-embed-text`（直近最良構成のまま）．レバー以外は一切動かさない．

**仮説**: 複合ドメイン行（`expected_domains` が 2 件）では，`dispatch_top_k=1` は confidence 最大の 1 ノード
にしか /dispatch しないため，期待 2 ドメインのうち 1 つしか被覆できない（medical と legal の recall がゼロサム）．
`dispatch_top_k=2` にすると閾値超の両ノードへ並行 dispatch が発火し，複合行の期待ドメイン集合を完全被覆できる．
`selected_domain`（最終採用＝confidence 最大）は不変なので既存の top1_accuracy 等は動かないが，新設する
set-valued 被覆指標では改善が観測できるはずである．

**評価コードの追加（レバーではなく計測基盤）**:
- 前提として発見した制約: 現行 `results.jsonl` は単一の `selected_domain` しか記録せず（`run_experiment.py`
  の `_run_one`, L72-83），dispatch 候補集合が残らない．set-valued 被覆は候補集合が必要なため，`run_experiment.py`
  の出力レコードに追記が要る（routing/集約の挙動は変えない・純粋な観測項目の追加）．
  - 追記フィールド `dispatched_domains: list[str]`: `aggregator.select_dispatch_targets(result.probe_responses,
    confidence_threshold, dispatch_top_k)` を再計算し，その各 target の domain を並べる（フロー本体と同じ関数・
    同じ probe_responses を使うので実際に dispatch された集合を忠実に再現．fallback 時は空リスト）．
  - 追記フィールド `probe_candidates: list[{node_id, domain, confidence}]`: `result.probe_responses` 全件（診断用）．
- `metrics.py` への追加関数 `compute_compound_coverage_metrics(results)`（既存関数は一切変更しない）:
  対象は `len(expected_domains) > 1` かつ `dispatched_domains` キーを持つ行のみ（旧 results は `r.get(...)` で
  スキップし後方互換を保つ）．各行で E=set(expected_domains)，D=set(dispatched_domains) として，
  - 被覆数 |D∩E|，被覆率 |D∩E|/|E|，Jaccard |D∩E|/|D∪E| を算出．
  - 集約して次を返す: `compound_rows_evaluated`(int)，`compound_covered_domain_count`(Σ|D∩E|)，
    `compound_expected_domain_total`(Σ|E|)，`compound_domain_set_recall`(=前者/後者, micro)，
    `compound_domain_coverage_ratio_mean`(macro)，`compound_domain_jaccard_mean`(macro)，
    `compound_mean_dispatched_count`(Σ|D|/行数, コスト代理)，`compound_coverage_available`(bool)．
  - `compute_all_metrics` に `"compound_coverage": compute_compound_coverage_metrics(results)` を追加（既存キー不変）．
    `print_summary` にも available 時のみ表示するセクションを追加．
- 既存指標との共存: top1_accuracy・misrouting_rate・precision_recall_per_domain・compound_domain_top1_accuracy
  等は数式・出力形式ともに不変．過去 results との比較可能性を維持する．

**成功条件（複合 4 行・各 expected 2 件＝Σ|E|=8 の規模で数値化）**:
- ベースライン（`dispatch_top_k=1`, 新スキーマで再実行）は複合行で 1 ドメインずつしか被覆せず
  `compound_covered_domain_count≈4`（`compound_domain_set_recall≈0.5`）になる想定．
- 主基準: `dispatch_top_k=2` で `compound_covered_domain_count ≥ 6`（＝ベースライン +2 以上，
  4 行中 2 行以上が 1→2 被覆に改善）．等価に `compound_domain_set_recall ≥ 0.75`（理想は 8/8=1.0）．
  N=4 のため 1 行の揺らぎ（set_recall で ±0.125）を超える +2 行以上を要件とする．
- コスト保護基準: 単一ドメイン行（30 問）の dispatch 数が 1 のままであること（`k=2` が曖昧/複合行でのみ
  発火する確認）．複合行の `compound_mean_dispatched_count ≤ 2.0`．
- 非退行基準: `top1_accuracy ≥ 0.97`・`misrouting_rate ≤ 0.03`（selected_domain ロジック不変のため probe
  ノイズ以外では動かないはず）．

---

### 調査 (Iter1)

対象レバー `dispatch_top_k`（1→2,3）が medical recall 改善に効くかを，先行研究とコード実装の両面から調査した．

**問い**
- Q1: 複数エキスパートへ並行問い合わせした結果の集約方式（自己申告 confidence 最大値以外）にどんな選択肢とトレードオフがあるか．
- Q2: 複合ドメイン（multi-label）質問でルーティング精度が落ちる現象の一般的知見．
- Q3: top_k を増やすコスト（CPU 推論前提）．

**分かったこと（コード実装の確認: 最重要）**
- 現行実装では `dispatch_top_k>1` にしても最終選択ドメイン（`selected_domain`）は top_k=1 と一致し，metrics.py が測る medical recall は動かない．根拠: `/probe` が confidence を request_id 単位でキャッシュ（http_server.py:249 `cache_probe_confidence`），`/dispatch` はその同じ値をそのまま `DispatchResponse.confidence` として返す（http_server.py:309 `pop_probe_confidence`），`select_dispatch_targets` は confidence 降順で top-k を採り（aggregator.py:18），`select_best_dispatch_response` はその中の最大 confidence を選ぶ（aggregator.py:36）．最大 confidence の top-k 先頭＝top_k=1 の選択と同一になる．
- 実データ（results/20260709_214113，34問）で確認: medical recall=0.786 の欠損は全て 4 件の複合 `['medical','legal']` 行に集中（3件が legal を選択，1件が medical）．legal recall=0.929 の欠損も同じ 4 行由来（medical を選んだ 1 件）．単一ドメイン 30 問は recall=1.0．つまり複合行では「1 回答しか返さない」構造上，medical と legal の recall はゼロサムで，両方 1.0 は原理的に不可能．
- 帰結: top_k=2 は複合行で legal と medical の両方へ dispatch するが，最終採用は再び confidence 最大（＝legal）に戻るため `selected_domain` は不変．しかも top_k=2 の再実験は /probe を再実行するので，run 間の probe スコア揺らぎ（temperature=0.1，router.py:17）が乗り，仮に recall が動いてもレバー効果とノイズが分離できない．

**分かったこと（先行研究，出典付き）**
- 自己申告 confidence は系統的に過信・較正不良で，選択信号として弱い（"Wired for Overconfidence", arxiv 2503系; ADVICE, ACL2026; Self-REF/Apple "Learning to Route LLMs with Confidence Tokens"）．本件では複合行の confidence が 0.9〜0.95 に飽和し弁別力が乏しい点が実データとも整合．
- 集約方式の選択肢: (a) LLM-as-judge / fuser LLM が候補回答＋批評を読んで再選定，(b) entropy-weighted voting，(c) 報酬誘導ルーティング（ZOOTER, IJCAI2024）・confidence-aware routing（CARGO）．ただし LLM-as-judge 自体も過信・自己選好バイアスを持つ（"Overconfidence in LLM-as-a-Judge", arxiv; "Self-Preference Bias in LLM-as-a-Judge", arxiv）．全体像は survey "Harnessing Multiple LLMs: A Survey on LLM Ensemble"（arxiv, Awesome-LLM-Ensemble）．なお多数決は「異なるドメインの 2 専門家が別回答を返す」本構成では成立しない．
- 複合ドメインは set-valued prediction として単一ラベルより本質的に難しく，precision/recall/F1/Jaccard/exact-match など集合レベル指標で評価すべき（"Multi-Agent Routing as Set-Valued Prediction: A WildChat Benchmark and Cost-Aware Evaluation", arxiv）．「複合行では top_k の dispatch 集合が期待集合を被覆したか」で測るのが素直．
- top-k のコストは k にほぼ線形（各 expert F FLOPs なら K×F）．実務標準は k=1 か k=2（Mixtral は 8 中 top-2），k>2 は品質向上が乏しく密モデルに近づく（Fedus et al. 2022; 各 MoE 解説）．本件はドメイン特化なので MoE の「多数 expert」設定とは異なり，候補は最大 3 ノードで k>2 は実質意味を持ちにくい．

**次フェーズ（rc-planner）への示唆**
- 最重要: 現行の config-only レバー `dispatch_top_k` は，集約方式（aggregator.select_best_dispatch_response）または metrics の複合行判定を変えない限り，target 指標（medical recall）に対して no-op になる公算が高い．計画では「何を成功とみなすか」を先に決める必要がある．
- コスト面の朗報: top_k>1 が実際に追加 dispatch を発火するのは「閾値 0.5 超のノードが 2 つ以上」＝曖昧/複合行のみ（単一ドメイン行は 1 ノードしか通らず no-op）．さらに複数 dispatch は別ノードへ `asyncio.gather` で並行（node.py:90）なので待ち時間は max(遅い方)で，メッシュ全体の計算量は増えるが requester のレイテンシ増は限定的．
- 具体的な選択肢（人間判断が要る，backlog 登録推奨）:
  - 案X1: `dispatch_top_k` を config-only のまま k∈{1,2,3} で回し，「recall は不変（no-op）」を実証＋レイテンシ実測を得る．純粋な確認実験で安全だが，予測どおり null 結果になる可能性が高くイテレーションのコスパは低い．
  - 案X2 (Recommended): 複合行の評価を set-valued（top_k dispatch 集合が expected_domains を被覆したか）に変更し，top_k>1 の効果を測れる指標を用意する．metrics.py の変更（コード変更＝config-only レバー原則から外れる）と人間承認が必要．
  - 案X3: top_k>1 と集約方式変更（LLM-as-judge を select_best_dispatch_response に導入）をセットで検証．改善幅は最大だがコード変更＋追加 LLM コスト＋judge 自体のバイアスに注意．単一レバー原則に反するため要人間判断．
- いずれにせよ「config-only の単一レバー原則」と「target 指標を動かすのに必要な変更」が衝突している．この論点を backlog に上げ，rc-planner は案X1〜X3 のどれを Iter1 の実験に落とすかを人間承認のうえ数値基準（例: 複合行被覆率，medical set-recall の閾値，許容レイテンシ増）とともに確定させるのが妥当．

---
