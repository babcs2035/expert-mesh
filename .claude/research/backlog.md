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

## B10 [auto-decided 2026-07-21] Iter5: few-shot 差し替えが router.py 側でしか効かない構造的原因の特定と次レバーの方針
- 状況: Iter5（education ノード few-shot 例の education 固有話題への差し替え）は rejected（主基準 2 件未達，非退行 2 件未達）．
  分析で決定的な構造的要因が特定された: build_dataset.py の _EDUCATION_QUESTIONS はテストクエリであり few-shot 例ではない．
  confidence 自己申告ロジックの few-shot 例は router.py の build_confidence_prompt() でハードコード（「歯の痛み→medical」「賃貸契約→legal」）され，
  全ドメイン共通で使われる．education ノードの評価にも medical/legal の例が使われるため，build_dataset.py の変更は confidence 信号に一切影響しない．
  決定的証拠: education ノードの confidence 値が Iter5 とベースラインで完全に同一．
- 自動選択: 次イテレーションの単一レバーを「router.py の build_confidence_prompt() に education 固有の few-shot 例を追加」へ方向付け．
  ただしこれはコード変更を伴うため単一レバー原則（config-only）の枠を超え，ユーザーの判断を仰ぐべき．
  次 rc-planner は以下の 2 選択肢のいずれかを提示する:
  - A: router.py の few-shot 例に education 関連話題を追加（コード変更，単一レバー原則の再設計が必要）
  - B: confidence_threshold の実質的な再較正（0.9 付近の閾値で education の過信を抑制，config-only 維持可能か検証）
  両方とも「単一レバー原則の再設計」が前提．B7 で記録した nomic-embed-text task prefix 未付与の問題も
  信号較正の文脈で並行検討すべき．
- 根拠: 3 イテレーション連続（Iter1: confidence飽和, Iter2: embedding弁別喪失, Iter3: 閾値no-op）で
  config-only の枠内で改善できないことが確定．Iteration 4-5 で education ドメイン追加および few-shot 差し替えを試したが，
  router.py 側の few-shot 構造が根本原因であることが判明．次は router.py の修正または confidence_threshold の実質的再較正へ．
- 要レビュー: 次 rc-planner が具体的な仮説と成功条件を提示する際，(1) router.py の few-shot 例追加が単一レバーとして成立するか，
  (2) 既存 results との比較に使う baseline は results/20260721_085735（Iter5）か results/20260721_011117（Iter4 ベースライン）か，
  (3) education ドメインの追加評価指標（precision/recall 目標値）をどう再定義するか，を明確化すること．

## B9 [auto-decided 2026-07-20] Iter3=confidence_threshold の no-op 確定と config-only レバー探索の収束・移行方針
- 状況: Iter3 対象レバー `confidence_threshold`（candidates [0.3, 0.5, 0.7]，既定 0.5）について，調査で二峰・
  空帯域分布による構造的 no-op が示唆されていた．ゲートは requester 側 aggregator が記録済み probe_responses に
  適用するだけのため，計画フェーズで**新規実験なしに**ベースライン結果 results/20260720_171532（34 行）の
  probe_candidates から閾値掃引をオフライン再計算し，thr=0.3/0.5/0.7/0.85 で fallback=0・total_dispatch=34・
  selected_domain 全 34 行一致（帯域 (0.3,0.7) に値 0 件，fallback は 0.9 以上でのみ発生かつ品質退行側）を確認．
  no-op が決定的に確定した．これで config.yml levers 3 本（dispatch_top_k=Iter1 棄却，routing_method=Iter2
  棄却，confidence_threshold=Iter3 no-op）を試し切り，config-only レバー探索が収束した．次にどの大きな
  方向へ進むかの判断（可逆だが実装量の大きい方向転換）が必要になった．
- 自動選択: 案C3 を採用．(1) 案C1（no-op を新規 run で実証）は棄却＝ゲートがオフライン再計算可能で新規 run
  （約 46 分）が冗長．(2) 案C2（levers.values を稠密域 ~0.15/~0.9 へ差し替え，config-only 単一レバー維持）は
  棄却＝top_k=1 固定下で ~0.15 は依然 no-op，~0.9 は専門ノードを general へ落とす品質退行で改善余地なし．
  (3) 案C3 を採り，本イテレーションは実験・実装をスキップ（config.yaml 無変更）し，**停止して人間判断を仰ぐ**
  形で移行方針を提示する．
- 根拠: 3 イテレーション連続で「config 値では confidence 信号の質を baseline 以上にできない」ことが示され，
  真のボトルネックが confidence 信号の較正（過信・飽和）という config-only の枠外にあることが確定した
  （Iter1: self_report 複合行 confidence 飽和 / Iter2: embedding cosine が [0.67,0.74] に潰れ弁別喪失・
  top1 0.53 / Iter3: 二峰分布で閾値どの候補値でも無反応）．config.yml research_frontier の順序規約
  「levers を試し切ってから research_frontier へ」の発火条件を満たす．
- 要レビュー（人間が選ぶ方向）: 次サイクルの重心を以下 A/B のどちらに置くか判断すること．
  - A: research_frontier「新規専門ドメイン追加」（B6 で方向性はユーザー承認済み）．ただし具体ドメイン選定・
    build_dataset.py 拡充・新規モデル準備・config.yaml ノード追加・router.py ドメイン別プロンプト整備を伴う
    大きめの変更で，次期 rc-planner での具体化と，どのドメイン/何ノードかの人間入力が要る．
  - B: confidence 信号の較正改良（B7 起点の nomic-embed-text task prefix 付与，複数 utterance ルート定義，
    ドメイン別 few-shot プロンプト整備）．いずれもコード変更を伴い config-only 単一レバー原則の外側（未承認）で，
    計測基盤・比較 baseline の再定義が必要．
  いずれも「単一レバー原則の再設計（config-only の枠を出る）」が前提になる点，および 3 イテレーション一貫の
  知見（ボトルネック=信号較正）が A/B どちらにも通底する点を判断材料とすること．C3 のためこのイテレーションでは
  config.yaml は無変更・deploy 不要．

## B8 [auto-decided 2026-07-20] Iter2 の判定と次イテレーション（Iter3）の単一レバー選定
- 状況: Iter2（routing_method: self_report→embedding）の結果，主基準（信号の質）が決定的未達
  （positive-margin 率 0.529 vs 基準 0.971，mean margin -0.0040 vs 0.60），非退行 3 指標も決定的未達
  （single_domain_top1_accuracy 0.500，top1_accuracy 0.529，misrouting_rate 0.471）．embedding は決定的計算
  （cosine のみ）で run 間ノイズがほぼ 0 のため構造的劣化と断定，追加反復は不要．よって routing_method
  レバーは棄却．config levers 優先順位 1（dispatch_top_k, Iter1 棄却）・2（routing_method, Iter2 棄却）を
  使い切り，残る config-only レバーの選定が必要だった（可逆な判断）．
- 自動選択: 次の単一レバーを `confidence_threshold`（config levers 優先順位 3，values: [0.3, 0.5, 0.7]）とし，
  ベースライン 0.5 を基準に振る．次イテレーション（Iter3）名は「confidence_threshold 掃引による fallback 率と
  general 過信リークのトレードオフ検証」．routing_method は交絡回避のため config.yaml でベースライン
  （self_report）に戻した．
- 根拠: (1) levers 優先順位で confidence_threshold が唯一未試行の config-only レバー（1・2 は棄却済み）．
  (2) Iter2 で「self_report 方式では閾値ゲートが機能している（Iter1 で複合行 medical=0.2 が閾値 0.5 に
  ブロックされ dispatch 未発火）」ことと，「embedding 方式では閾値 0.5 が 102/102 probe で全通過し無意味化」
  という対照的な新知見が得られており，self_report ベースラインで閾値を動かす効果を測る素地がある．
  (3) research_frontier（新規専門ドメイン追加）はコード・データセット・ノード追加を伴う大きめの変更で，
  config-only レバーを試し切ってから着手する順序（config.yml research_frontier に明記）を守る．
- 要レビュー: (1) B5 で記録したトレードオフ「confidence_threshold を下げると general の過信リーク
  （Iter1 で general-008 の余分 dispatch を実測）が悪化する」を rc-planner が非退行基準に組み込み数値化する
  こと．具体的には fallback_rate（直近実験では 0.0 のため閾値を上げた側で初めて動く可能性）・単一ドメイン行の
  over-dispatch・general の precision を監視項目に含める．(2) 直近実験は fallback_rate=0.0 のため
  confidence_threshold の効果自体が薄い可能性がある（config levers の note にも記載済み）．null 結果になった
  場合は config-only レバーを試し切ったと判断し，停止条件（グローバル skill）か research_frontier（新規
  専門ドメイン追加）への移行を rc-planner が検討すること．(3) 閾値を下げる方向（0.3）と上げる方向（0.7）で
  効くメカニズムが異なる（下げる＝over-dispatch/リーク，上げる＝fallback 増）ため，どちらを主眼に置くか
  rc-planner が仮説を明確化すること．

## B7 [auto-decided 2026-07-20] nomic-embed-text の task prefix 未付与（Iter2 スコープ外・劣化時の切り分け課題）
- 状況: Iter2（routing_method: self_report→embedding）の調査で，nomic-embed-text は非対称タスク用の
  task instruction prefix（search_query: / search_document: / classification: 等）を前提に学習されているが，
  現行コードは query（node.py:143）にも domain（http_server.py:184）にも prefix を付けていないと判明．
  prefix 無し＋英単語(domain)対日本語(query)のクロスリンガル比較は較正上不利で，embedding ルーティングの
  単一ドメイン精度が退行する既知の落とし穴になり得る．
- 自動選択: Iter2 では prefix 付与を行わず，prefix 無しの現状のまま routing_method=embedding を config-only で
  評価する．退行（特に single_domain_top1_accuracy の低下）が観測された場合に，prefix 起因かの切り分けを
  次段階の課題としてここに残す．
- 根拠: prefix 付与は両 embed 経路（node.py・http_server.py）のコード変更が必要で，Iter2 の config-only
  単一レバー原則と衝突する．まず現状構成で embedding の素の性能を測り，交絡なく劣化要因を特定するのが妥当
  （調査の提案どおり）．
- 要レビュー: Iter2 で embedding が非退行基準を割った場合，(1) prefix 付与（query に search_query:，domain に
  search_document: 等）を加えた再実験を別イテレーションで行うか，(2) domain 定義をドメイン名 1 単語から
  複数代表発話(utterances)へ拡張するか（Semantic Router ベストプラクティス，調査参照），を rc-planner が
  検討すること．いずれもコード変更を伴うため単一レバー原則との整合を人間判断すること．

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
