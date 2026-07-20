# journal — expert-mesh research-cycle

新しいものを常に先頭へ挿入する（逆時系列）．書式はグローバル skill（~/.claude/skills/research-cycle/SKILL.md）を参照．

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
