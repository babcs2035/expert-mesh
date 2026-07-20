# backlog — 人間判断待ち事項 / 自動判断の記録

このファイルは research-cycle が「本来は人間の判断が要るが，サイクルを止めないために暫定で自動選択した事項」と，
「不可逆・危険なため停止して人間に委ねた事項」を記録する．新しいものを常に先頭に追記する（逆時系列）．

書式:
```
## B{n} [auto-decided YYYY-MM-DD] 題目
- 状況: なぜ判断が要ったか
- 自動選択: 何を選んだか
- 根拠: なぜそれが最も妥当か
- 要レビュー: 人間が確認・却下する際に何を見るか（却下時はこの項目を編集して差し替える）
```
不可逆な事項は `[needs-human YYYY-MM-DD]` として記録し，Slack で @mention 済みであることを明記する．

---

## B6 [user-approved 2026-07-20] 実機ノード拡張（最大10台）と VRAM 常時確保
- 状況: ユーザーから (1) 192.168.15.100〜109（最大10台，同一スペック）が実機として利用可能になったこと，
  (2) 実機のVRAMが解放されてしまっていたこと，の2点の連絡があった．(2) の確認のため3ノード（wafl500/502/503）
  の ollama `/api/ps` を確認したところ全ノードで `models: []`（アンロード済み）だった．
- ユーザーの選択:
  - VRAM確保: docker-compose.ymlのollamaサービスに`OLLAMA_KEEP_ALIVE=-1`を追加（commit 94d4b50，push済み）．
    `mise run deploy`で3ノードに反映し，`/api/ps`で両モデル（qwen3.5:9B, nomic-embed-text）が
    `expires_at: 2318-10-30`（実質無期限）でロード済みであることを確認．
  - ノード拡張の活用方向: 「新規専門ドメインの追加」（既存ドメインの冗長化・ノード数スケール検証は不採用）．
    config.yml の research_frontier に記録済み．具体的なドメイン候補・データセット拡充は次期rc-planner着手時に
    具体化する．
- 根拠: VRAM対応はユーザーの直接指示（運用上の緊急対応，可逆）．ノード拡張の方向性はユーザーが
  AskUserQuestionで直接選択（新規専門ドメイン追加＝メッシュ本来の目的である専門分野分担の拡充に直結）．
- 要レビュー: 新規専門ドメイン追加は，現在進行中のIteration 2（routing_method）の後，かつ単一レバー原則
  （既存ノードの構成・動作を変えない形での追加）を守って着手すること．具体的なドメイン名（教育・金融等）
  はまだ未決定．git push はグローバルCLAUDE.mdの「git push絶対禁止」規約と衝突するため，このVRAM対応の
  push もAskUserQuestionで個別に確認済み（研究サイクルのreflectorによるpushとは別に確認が必要だった点，
  今後も同様の非イテレーション変更ではpush前に確認すること）．

## B5 [auto-decided 2026-07-20] Iter1 の判定と次イテレーションの単一レバー選定
- 状況: Iter1（dispatch_top_k=1→2）の結果，主基準 compound_covered_domain_count>=6 は未達（実測 4→5,
  +1 のみ）．根本原因は confidence_threshold=0.5 のゲートを複合 3 行の medical confidence(0.2) が越えられず
  追加 dispatch が発火しないこと．真のボトルネックは confidence 信号の質であり dispatch 並列数ではないと判明．
  よって dispatch_top_k レバーは棄却．次に振る単一レバーを決める必要があった（可逆な判断）．
- 自動選択: 次レバーを `routing_method`（config levers 優先順位 2 番目）とし，方式 B(self_report,既定)→
  方式 A(embedding) へ振る．次イテレーション名は「embedding ルーティング(方式A)への切替による複合ドメイン
  被覆の検証」．dispatch_top_k は交絡回避のため config.yaml でベースライン(1)に戻した．
- 根拠: (1) levers 優先順位で routing_method が 2 番目．(2) Iter1 で self_report の自己申告 confidence が
  過信/較正不良で複合行の弁別に効かないことが実証され，confidence 信号そのものを変える routing_method は
  根本原因に直接対応する．(3) 3 番目候補 confidence_threshold を先に下げる案は，general の過信リーク
  （Iter1 で general-008 の余分 dispatch を実測）を悪化させるトレードオフがあり，信号の質を改善する方が先．
- 要レビュー: rc-planner は embedding 方式での成功基準（複合行被覆の目標値・非退行として top1_accuracy と
  general 過信リークの許容範囲）を数値化すること．embedding_model=nomic-embed-text の probe レイテンシと
  精度のトレードオフも観測項目に加えること．k=3 は複合行の期待ドメインが最大 2 のため k=2 と同一結果に
  なる見込みで検証価値が低く，dispatch_top_k の追加反復は不要と判断した（この点への異議があれば差し替え）．

## B4 [auto-decided 2026-07-20] 既存テストの壊れた import 修正（本イテレーションとは無関係）
- 状況: rc-implementer が完了条件（`uv run pytest tests/ -v` が通ること）を確認しようとしたところ，
  `tests/test_metrics.py` と `tests/test_build_dataset.py` が存在しないパッケージ `benchmark`
  （`from benchmark.metrics import ...` 等）を import しており，pytest の collection 自体が
  全滅していた．commit `71ac11a` 由来で，本イテレーションの単一レバー（dispatch_top_k）とは無関係の
  既存バグである．
- 自動選択: 他の全テストファイルが使っている import 形式（`from metrics import ...` 等）に，
  該当 2 行のみ修正した．assert・テストロジック本体は変更していない．
- 根拠: 完了条件を満たすには pytest 自体が動く必要があり，かつ修正は import 文のみで最小限のため，
  今回の実装作業に含めて自動判断した（CLAUDE.md「軽微な不明点は合理的な仮定を明記して前に進めてよい」）．
- 要レビュー: このバグ自体は本来 dispatch_top_k とは無関係の既存不具合であり，なぜ・いつから
  collection エラーになっていたかの経緯確認は未実施．必要であれば別途調査すること．

## B3 [user-approved 2026-07-20] 被覆率計測のため run_experiment.py の出力スキーマ拡張
- 状況: rc-planner が B2（metrics.py のみの変更）の具体化を検討したところ，複合ドメイン行の dispatch 候補
  集合（confidence_threshold 超のノードのうち dispatch_top_k 件）を記録する一次データが，現行の
  results.jsonl（run_experiment.py 出力，selected_domain 単体のみ）には存在しないと判明した．metrics.py
  は事後解析のみのため，欠けている一次データを事後に復元することはできない．B2 で承認された範囲
  （metrics.py のみ）を超えるスコープ拡大のため，再度ユーザーに確認した．
- ユーザーの選択: run_experiment.py の出力にも新規フィールド（dispatched_domains, probe_candidates）を
  追加して進める．既存フィールドは変更せず，新フィールドを持たない旧 results.jsonl は metrics.py 側で
  スキップする後方互換設計とする．aggregator.py 等の集約ロジック本体は変更しない．
- 根拠: dispatch_top_k レバーの効果を測定可能にするための最小限の観測項目追加であり，ルーティング・
  集約の挙動そのものは変えない．これを行わない場合，dispatch_top_k レバーは検証不能（no-op のまま）．
- 要レビュー: rc-implementer が実装した後，既存の results.jsonl 読み込みコード（metrics.py の
  _read_results 等）が新フィールドの有無で分岐し，古い実行結果に対しても実行時エラーなく動作することを
  確認すること．dispatch_top_k=1 のベースラインも新スキーマで再実行が必要（旧 results/20260709_214113
  は新フィールドを持たないため比較に使えない）．

## B2 [user-approved 2026-07-20] dispatch_top_k レバーの no-op 判明への対応方針
- 状況: rc-investigator の調査（journal.md「調査 (Iter1)」参照）により，dispatch_top_k を config-only で
  1→2,3 に振っても，現行の集約ロジック（aggregator.select_best_dispatch_response が confidence 最大値を
  選ぶのみ．probe の confidence が http_server.py 側で request_id 単位キャッシュされ dispatch にそのまま
  引き継がれる）では最終的な selected_domain が変わらず，metrics.py が測る medical recall に対して no-op
  になる可能性が高いと判明した．「config-only の単一レバー原則」と「target 指標を動かすのに必要な変更」が
  衝突する分岐点のため，AskUserQuestion でユーザーに直接確認した．
- ユーザーの選択: 案X2（評価方式を拡張）．metrics.py に，複合ドメイン行を対象とした set-valued 被覆判定の
  指標を既存の recall 等に「追加」する形で実装し，dispatch_top_k>1 が実際に候補集合をどれだけ被覆できて
  いるかを測れるようにする．既存の単一 selected_domain 前提の指標（top1_accuracy 等）は変更せず残す．
- 根拠: この指標拡張なしには dispatch_top_k レバー自体が意味をなさない．小規模な評価コード追加のみで，
  集約ロジック（aggregator.py）自体や実験の起動系（mise タスク）には手を入れないため，影響範囲を絞れる．
- 要レビュー: rc-planner が具体的な指標定義（例: dispatch 候補集合と expected_domains の Jaccard/被覆率）
  と成功基準の数値化を行う．rc-implementer が metrics.py への実装を行う際は，既存指標の出力形式・関数を
  破壊しないこと（既存の journal・過去 results との比較可能性を保つため）．

## B1 [auto-decided 2026-07-20] config.yml 初回セットアップ
- 状況: research-cycle 初回起動（このリポジトリでは config.yml/state.json/journal.md/backlog.md が未作成だった）．
  levers の優先順位・success_criteria・tasks コマンド・timeout_min 等を新規に決める必要があった．
- 自動選択: levers の優先順位（1. dispatch_top_k，2. routing_method，3. confidence_threshold）はユーザーに
  AskUserQuestion で確認済み（dispatch_top_k を最優先として承認）．success_criteria は暫定的な定性表現とし，
  timeout_min（90分）・metrics_cmd（最新の results/*/results.jsonl を動的解決）は直近の完走実験ログから算出．
- 根拠: 直近の完走実験（results/20260709_214113/results.jsonl）で metrics.py を実行したところ top1_accuracy=1.0
  だが medical の recall=0.79（legal 0.93, general 1.0）と判明．dispatch_top_k>1 は既存の
  aggregator.select_best_dispatch_response で対応済みのため，新規実装なしで即検証できる最有力候補と判断．
- 要レビュー: success_criteria の数値基準が未確定（イテレーションを重ねてノイズ幅が分かってから rc-planner が
  数値化する設計）．research_frontier の各項目（ベースライン比較・回答品質評価等）は levers 探索後の着手を
  想定しているが，優先度を変えたい場合は config.yml の該当セクションを直接編集してよい．
