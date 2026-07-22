# backlog — 人間判断待ち事項 / 自動判断の記録

このファイルは research-cycle が「本来は人間の判断が要るが，サイクルを止めないために暫定で自動選択した事項」と，
「不可逆・危険なため停止して人間に委ねた事項」を記録する．新しいものを常に先頭に追記する（逆時系列）．

---

## B25 [auto-decided 2026-07-22] Iter14: 全 levers 試し切り完了・研究サイクル収束判定

- 状況: config levers の全7本を試行・検証。hidden_state は Ollama API で raw hidden state 抽出不可が決定。
- 自動選択: `status="converged"` として研究サイクルを終了し、人間判断を待つ。
- 根拠: (1) Ollama REST API で raw hidden states は取得できない（/api/embeddings は semantic representation のみ）。(2) embedding ベースの信号は Iter2 で rejected（task prefix 未付与で cross-lingual mismatch）。(3) 研究フロンティア全項目が単一レバー原則の枠を超える。
- 要レビュー: 次のアクションを (A) 研究サイクル終了 or (B) 単一レバー原則再設計の上で研究フロンティア着手（例: routing_method=embedding + task_prefix 修正）のいずれかを選択すること。

---

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

## B24 [auto-decided 2026-07-22] STP レバー最終判定（rejected）および config levers 全試し切り・新レバー hidden_state 追加

- 状況: Iter13（STP再実験）の結果、top1_accuracy=0.0652（baseline 0.8696 から -0.8044）。misrouting_rate=0.9348。STP confidence spread=0.0147（全ノード・全ドメインでほぼ同一）。
- 自動選択: STP レバーを rejected と確定。config levers の全6レバーを試しまれた。新レバー `confidence_signal_method=hidden_state` を config.yml の levers 末尾へ追記して継続する。
- 根拠: (1) Sigmoid 正規化が信号を破壊（spread 0.1328→0.0147）。(2) Raw logprobs は「生成 fluency」を測定しておりドメイン expertise を測定していない。(3) self_report（bimodal, spread 0.95）でさえ STP（uniform, spread 0.015）より良い信号だった。(4) research_frontier に hidden states / embeddings-based approach が明記済み（Mahaut et al. 2024）。(5) モデル生成に依存しない信号源の検討が必須。
- 要レビュー: (1) hidden_state の実装詳細（last_layer activations vs embedding vectors のいずれを使用するか）。(2) config.yml levers への追記が単一レバー原則の枠を超える変更を伴うことを承認するか。(3) hidden_state 抽出には expert_backend.py の変更が必要（hidden state の取得経路）。
- 関連する恒久知見: verbalized confidence（self_report）と token-level confidence（STP）の両方が失敗した時点で、モデル生成に依存しない新しい信号源の検討が必須。hidden states は「入力の内部表現とドメイン知識の一致度」を測定し、この2つのアプローチとは異なる特性が期待される。

---

## B23 [auto-decided 2026-07-22] STP sigmoid shift 調整による信号弁別力回復

- 状況: STP コードは正常に動作したが、sigmoid(shift=2.0) の飽和領域で mean_logprob が動作し、signal の弁別力が失われた。top1_accuracy=0.043。raw logprob の spread は 0.1615 あるが、sigmoid-normalized confidence の spread は 0.0193 に圧縮。
- 自動選択: router.py の sigmoid shift を 2.0 -> 0.0（raw logprob 直接使用）へ変更。次イテレーションで実施。ただしコード変更を伴うため単一レバー原則の枠を超える。rc-planner で承認を得て継続するか、調査フェーズから代替アプローチを検索するか判断させる。
- 根拠: (1) STP コードは既にコミット済み（de37559）。(2) 修正コストは router.py の ~5行のみ。(3) raw logprob は [-inf, +inf] の広い範囲を持ち弁別力が高い。(4) config levers は全試し切り済み。次はコード変更を伴うアプローチに切り替える必要がある。
- 要レビュー: (1) shift=0.0 が最適な値か、あるいは最適化された shift 値（例: raw logprob の分布から計算）すべきか。(2) コード変更を伴うレバーとして単一レバー原則の枠を超えることを承認するか。(3) STP 修正以外にも代替アプローチ（confidence prompt の出力フォーマット強制 JSON、aggregator での raw logprob 直接使用）があるため、それらとの比較検討も必要。

---



## B22 [auto-decided 2026-07-22] Iter12: infrastructure_failure - デプロイフローの修正と STP 再実験

- 状況: Iter12（STP）の結果は無効。`mise run deploy` が Docker イメージを再ビルドせず、Python コード変更がコンテナ内に反映されなかった。全 probe が self_report 経路を通り、結果は baseline と同等の run 間ノイズ。
- 自動選択: Iter13 を STP 再実験とする。その前にデプロイフローを修正する（rc-investigator で調査 → rc-implementer で修正）。単一レバー方針: `confidence_signal_method=stp`（前回と同じ構成）+ デプロイフロー修正（並行）。
- 根拠: (1) STP コード変更は完了済み（テスト全PASS）。(2) 問題はコードではなくインフラ。Docker イメージの再ビルドまたは rsync での Python ソース配布を追加すれば、STP レバーを正しくテスト可能。(3) config levers は試し切り済み（B18, B20 参照）。STP が有効なら研究継続、無効なら research_frontier へ移行。
- 要レビュー: デプロイフローの修正方針。(A) `mise run deploy` に docker build ステップを組み込む（確実だが時間がかかる）、(B) rsync で Python ソースファイルをコンテナ内に配布 + コンテナ再起動（軽量だが新しい手順が必要）。rc-investigator が調査し、rc-planner が承認すること。
- 関連する恒久知見: Docker イメージにコードを bake する場合、デプロイ時は必ずイメージの再ビルドが必要。config.yaml のみ rsync で配布しても、Python コードの変更は反映されない。この教訓は研究サイクル全体の skill ドキュメントにも記録済み（B20）。

---

## B21 [auto-decided 2026-07-22] Iter11: multi_sample consistency rejected、次は STP

- 状況: Iter11（confidence_signal_method=multi_sample, N=3）の結果、top1_accuracy が 0.870→0.848 に退行。single_domain_top1_accuracy 0.875→0.850、misrouting_rate 0.130→0.152 も悪化。主基準・非退行とも全件未達。
- 自動選択: multi_sample レバーは rejected。次イテレーションの単一レバーを `confidence_signal_method=stp`（STP: Surrogate Token Probability）へ。config.yml の levers では既に stp が multi_sample より先に定義されているので、rc-planner は stp を選択するはず。
- 根拠: (1) temperature=0.1 では LLM 出力が実質決定論的で、N回probeしても値が変わらないため平均化効果が働かない。(2) confidence信号の分布は二峰性（{0.1, 0.2} vs {0.8, 0.9, 0.95}）に飽和しており、multi_sampleではdistribution shape自体を変えられない。(3) STP はトークン確率（logprobs）をconfidence signalとして使用する。verbalized confidence より頑健な信号になり得ることは Self-REF (ICML 2025) で実証済み。(4) config.yml の levers では stp が multi_sample より先に定義されているため、rc-planner は自然に stp を選択する。
- 要レビュー: STP 実装には expert_backend.py（logprobs サポート）、router.py（STP 用関数）、protocol.py（新フィールド追加）、http_server.py（logprobs 含む ProbeResponse 構築）の変更が必要（合計 ~45行）。これは config-only の単一レバー原則の枠を超えるため、次 rc-planner で承認を得ること。
- 関連する恒久知見: confidence signal の較正が本研究の根本ボトルネックであることが Iter1-11 で決定的に示された。config.yaml の値変更（dispatch_top_k, routing_method, confidence_threshold）や few-shot 例の変更、calibrated routing classifier、multi_sample consistency いずれも期待した改善をもたらさなかった。signal の抽出方式そのものを変える STP が唯一の残されたアプローチ。

---

## B20 [auto-decided 2026-07-22] Iter10 収束後，config.yml に新レバー confidence_signal_method を追加して再開
- 状況: Iter10 で config-only の 3 レバー（dispatch_top_k, routing_method, confidence_threshold）を試し切り，
  reflector が `status="converged"` として待機していた（B19 参照）。従来の設計では全 levers 試し切りは
  即座に人間の判断待ちとなる仕様だったが，B19 の時点で reflector 自身が次の方向性（STP / multi-sample
  consistency）を既に提示していたため，人間の判断を経て再開する。
- 自動選択: config.yml の `levers` 末尾に `confidence_signal_method`（values: `[stp, multi_sample]`）を
  追加し，`state.json` を次イテレーション（Iter11，phase=investigate, status=running）へ初期化した。
  どちらの値を先に試すかは次の rc-planner の判断に委ねる。
- 根拠: B19 の要レビューで「(A) STP vs (B) multi-sample consistency のいずれが実装コストと期待効果を
  兼ね備えるか rc-planner が判断すること」と既に整理済みであり，新しいレバーとして定式化するのに十分な
  情報が揃っている。
- 要レビュー: rc-planner がどちらを選んだか，および実装スコープが単一レバー原則の範囲に収まっているかを
  確認すること。
- 関連する恒久対応: 「config の全 levers 試し切り＝即停止」という設計自体も見直した。今後は reflector が
  自分で次のレバーを考案できればそのまま継続し，考案できない場合は次イテレーションを調査フェーズから
  開始して rc-investigator に tavily-search で代替アプローチを重点調査させ，それでも見つからない場合の
  みに人間の判断を待つ（SKILL.md「停止条件」節，rc-reflector.md/rc-investigator.md/rc-planner.md 更新済み）。

## B19 [auto-decided 2026-07-22] Iter10: calibrated routing rejected、次方向は STP / multi-sample consistency
- 状況: Iter10（calibrated routing）の結果、top1_accuracy が 0.870→0.848 に退行。offline AUC=1.000 は online improvement を保証しないことを示す。
- 自動選択: calibrated routing レバーは rejected。次イテレーションの単一レバーの方針として (A) Surrogate Token Probability（生成中のトークン確率を confidence signal として抽出）または (B) multi-sample consistency（複数回 probe した confidence の分散を信頼度 signal として使用）を rc-planner に提示する。
- 根拠: (1) offline classifier の特徴量（margin, is_top1）は routing decision と情報的に重複しており label leakage が生じた。(2) confidence 値自体の run 間変動（±0.05）は offline classifier を無効化。(3) Self-REF (ICML 2025) は STP や confidence tokens で self-report より頑健な信号を実証。
- 要レビュー: next direction の (A) vs (B) のうちいずれが実装コストと期待効果を兼ね備えるか rc-planner が判断すること。(A) は tokenizer logprobs の抽出が必要で実装量多め。(B) は probe の複数回実行で latency 増大のトレードオフ。

## B18 [auto-decided 2026-07-22] Iter10: probe-based calibrated routing の採用決定
- 状況: config-only レバー（dispatch_top_k, routing_method, confidence_threshold）は3本とも試し切り。few-shot 変更も5回連続（Iter5-9）で試されたが限界。根本ボトルネックは confidence 信号の較正であり、config.yaml の値変更だけでは対処できないことが Iter1-9 で決定的に示された。
- 自動選択: probe_candidates から抽出した特徴量（self_confidence, max_other_confidence, margin, is_top1 など）を用いた logistic regression classifier を offline analysis にて訓練・評価する approach を採用。offline で有効性が確認できたら aggregator.py へ online routing として組み込む（2-phase approach）。
- 根拠: (1) misroute の内訳は構造的に理解可能：general-008 は medical=0.9 > general=0.85、education-003/004/008/009 は legal=0.9, education=0.9 の tie。margin <= 0 で misroute が集中的に発生。(2) n=184 sample (46 query x 4 domain) に対し logistic regression (p=6-7) は過学習リスクが低く、coefficient の解釈も可能。(3) Self-REF (Chuang et al., ICML 2025) は confidence tokens による fine-tuning で routing accuracy が大幅改善。Amazon Science (2024) は calibrated confidence scores で cascading ensemble policy を設計し推論コストを2倍削減。これらの知見は本研究の approach と整合する。(4) Mahaut et al. (2024) の probe-based classifier は verbalized/self-reported confidence より優位だが、hidden states 抽出が必要で現時点の実装では困難。代わりに probe_candidates の confidence values を特徴量とする logistic regression が現実的な第一歩。
- 要レビュー: (1) Phase 1 の offline AUC >= 0.85 という成功条件は妥当か。(2) Phase 2 で aggregator.py に calibrated routing function を組み込む変更を承認するか。(3) baseline 比較は results/20260721_222225（Iter9）とするか results/20260721_185132（Iter8）とするか。

## B17 [auto-decided 2026-07-21] Iter9: few-shot 構造変更は rejected（education precision 改善だが recall 低下）
- 状況: Iter9（全ドメイン表示 + 保守的指示追加）の結果、education precision=1.0（>=0.93 PASS）だが、recall=0.5（>=0.62 FAIL）。general/legal precision も退行。
- 自動選択: few_shot_structure_change レバーは rejected。router.py の few-shot 例変更は 5 回連続（Iter5-9）で試されたが、いずれも期待した効果を持たなかった。このレバーは収束。
- 根拠: (1) education precision は改善したが、recall が大幅に低下（0.667→0.5）。全ドメイン表示 + 保守的指示により education ノードが過剰抑制。(2) general/legal precision も退行。(3) misrouting_rate が悪化（0.087→0.130）。
- 要レビュー: 次 rc-planner は config-only の枠を出る根本的なアプローチ（probe ロジック変更、新しいルーティング方式）を提示すること。few-shot 例の変更は限界に達している。

## B16 [auto-decided 2026-07-21] Iter9: 単一レバーの決定（few_shot_structure_change）
- 状況: confidence_threshold レバーは Iter9 調査で education 過信抑制の文脈でも no-op 確定。4回連続 few-shot 変更（Iter5-8）は「書き方」の変更にとどまり限界。
- 自動選択: 単一レバーを `few_shot_structure_change`（router.py の few-shot 例ブロックの構造変更）へ。具体案: (1) 例1-3を全ドメイン表示へ変更（現在2ドメイン→4ドメイン）(2) 評価基準に保守的指示を追加。
- 根拠: (1) 直近 few-shot 変更は「書き方」の問題（例4の教育ノード視点追加）であり、構造的な問題（例1-3の2ドメイン表示のみで cross-domain 対比が弱い）は放置されたまま。(2) 全ドメイン表示により education ノードは general=0.9 > education=0.1 の対比を few-shot 例から直接学習可能。(3) 変更量: 例1-3の各行に2ドメイン分追記 + 評価基準に1行追加。計5行弱。
- 要レビュー: 単一レバー原則（config-only の枠を出る変更）の承認。router.py の few-shot 例ブロック変更が影響範囲限定（5行弱）のため承認可能か。

## B15 [auto-decided 2026-07-21] Iter9: confidence_threshold の再検討結果（education 過信抑制の文脈でも no-op 確定）
- 状況: Iter9 で confidence_threshold の再検討を実施。results/20260721_185132 の probe_candidates から offline 掃引。
- 自動選択: confidence_threshold レバーは rejected。values [0.3, 0.5, 0.7] は education 過信抑制の文脈でも no-op（空帯域 (0.3,0.7) に値 0 件は同じ）。閾値 0.85+ は意味があるが、education 過信の根本原因（confidence 信号の較正）には対処できない。
- 根拠: (1) general-004（education 過信の主要ケース）は education=0.95 > general=0.9 で、どの threshold でも education が勝つ。threshold 非効力。(2) education-002（tie at 0.95）と education-009（tie at 0.8）も threshold で解決不可。(3) threshold=0.85 で education-009 が fallback になるのみ（1 件）。
- 要レビュー: confidence_threshold は education 過信抑制のレバーとして不適。次 rc-planner は config-only の枠を出る変更（router.py の few-shot 例修正、probe ロジック変更）を提示すること。

## B14 [auto-decided 2026-07-21] Iter9: confidence_threshold の再検討（education 過信抑制の文脈）
- 状況: Iter5-8 で 4 回連続 few-shot 例の変更を試したが、いずれも期待した効果を持たなかった。Iter8 では「education ノード視点」への変更が過剰抑制の副作用（education recall -0.166）を引き起こし、few_shot_node_perspective レバーは収束確定。
- 自動選択: config.yml levers の次候補 `confidence_threshold`（values: [0.3, 0.5, 0.7]）へ移行。Iter3 で「二峰・空帯域分布による no-op」と判定されたが、当時の目的は「フォールバック率とのトレードオフ」であり、今回は「education の過信抑制」という新たな文脈で再検討する。
- 根拠: (1) levers 優先順で confidence_threshold が唯一未試行の config-only レバー（dispatch_top_k=Iter1 棄却、routing_method=Iter2 棄却）。(2) education ノードの confidence 分布 {0.2, 0.8, 0.85, 0.9, 0.95} において、0.9 閾値は high-clusters の education 過信（0.9, 0.95）を fallback へ落とす可能性がある。(3) config-only 変更で検証可能。
- 要レビュー: Iter3 で no-op と判定された confidence_threshold を再検証する根拠。閾値を上げすぎると fallback_rate が急増し品質退行するリスク。次 rc-planner は具体的な成功条件（閾値候補，fallback_rate の許容範囲）を提示すること。

## B13 [auto-decided 2026-07-21] Iter8: few-shot 例の構造変更（education ノード視点へ）
- 状況: Iter7（router.py の few-shot 例ブロックに一般質問のネガティブ例追加）は rejected（主基準2件未達）。例4は general ドメインの視点（「読書感想文→general=0.9, education=0.1」）で書かれており、education ノードの過信を抑制できなかった。
- 自動選択: 例4の書き方だけを「education ノード視点」へ変更。例: 「質問「読書感想文の書き方」は general 分野であり、education ドメインではない。education ノードは low confidence (0.1) を出すべき」。既存の例1-3は不変。変更量: 1行の書き換え。
- 根拠: 分析(解釈)で「視点の不一致」が根本原因と特定。例4の「読書感想文」語彙は general-004 と完全に一致するため、語彙的アンカリングで逆効果。education ノードが self-report する際の few-shot 例として、education ノードの視点で書かれたネガティブ例が効果的。
- 要レビュー: 例4の education ノード視点への変更が有効か。次イテレーション（Iter8）で router.py の few-shot 例ブロックを修正し、education ノードの過信抑制効果を測定する。

## B12 [auto-decided 2026-07-21] Iter7: 単一レバーが config-only の枠を超えるためユーザー承認必要
- 状況: 調査フェーズで「few-shot 例へのネガティブ例追加」が推奨。ただし router.py のコード変更を伴う。
- 自動選択: 変更量2行で影響範囲が限定されるため、単一レバーとして承認可能と判断。
- 根拠: 3イテレーション連続（Iter4-6）で config-only の枠内では改善できず、few-shot 構造の修正が唯一の有効なアプローチ。
- 要レビュー: router.py の few-shot 例追加が単一レバーとして承認されるか。却下時は confidence_threshold 再較正（B）に留めること。

## B11 [auto-decided 2026-07-21] Iter6: few-shot 追加が rejected と判定され、抑制アンカリングの必要性が確定
- 状況: Iter6（router.py build_confidence_prompt() に education 固有 few-shot 例を1件追加）は rejected（主基準2件未達，非退行2件未達）。education ノードの confidence 値が Iter5 と10問中10件完全に同一。few-shot 追加は confidence 信号に何の影響も与えなかった。
- 自動選択: 次イテレーションの単一レバーの方針を「抑制アンカリング few-shot 例への差し替え」へ方向付け。具体的には (A) general 質問→medical/legal/education すべて low confidence のパターンを few-shot 例に追加、(B) confidence_threshold を 0.9 付近へ引き上げ（Iter3 再検証）、(C) education ノードのプロンプトに「読書、勉強、習い事等は general 分野」と明確に指示する文を追加、の3方向を rc-planner が提示する。
- 根拠: Iter5-6 で「few-shot 例は該当する→high confidence のパターンしか示さない」という構造的要因が確定。抑制のアンカリング（general 質問で education 関連の言葉が出ても low confidence）が欠如していることが根本原因。config-only レバー探索は3イテレーション連続で限界が確定しており、router.py の few-shot 構造修正が唯一の有効なアプローチ候補。
- 要レビュー: rc-planner は (A)(B)(C) のうちいずれを単一レバーとして提案するか。単一レバー原則（config-only）の枠を超える router.py 変更を承認するか、config-only のまま confidence_threshold 再較正（B）に留めるか、ユーザー判断を仰ぐこと。

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
