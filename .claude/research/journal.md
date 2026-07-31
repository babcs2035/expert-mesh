## Iteration 27: 高度な集約方式（majority_vote / llm_judge）の比較実験 — 実験不成立（no-op）

**背景**: research_frontier 項目5（top-k dispatch の高度な集約方式）の実機比較（backlog B47）．
`aggregator.py` への実装は commit `178960a` で完了しており，`config.yaml` の
`dispatch_top_k` を 1→2（`cde9247`），`aggregation_method` を
`max_confidence`→`majority_vote`（`7f72b1a`）→`llm_judge`（`32af2e0`）と切り替えて
1600 問を 3 回実行した．

**本節は 2026-07-31 に事後整理として記録した**．実験は 07-30 22:45 〜 07-31 03:44 に完走していたが，
分析・記録・コミットが行われないまま約 12 時間停止していた（停止の経緯は本節末尾および
docs/d0004 §6-1 を参照）．

### 実験 (Iter27)

| 実験ディレクトリ | 集約方式 | 実行時の HEAD | 期間 | 完走 |
|---|---|---|---|---|
| `results/20260730_224515/results_topk2_maxconf.jsonl` | max_confidence | `9b7f393` | 07-30 22:45 → 07-31 00:21 | 1600/1600 |
| `results/20260731_002420/results_topk2_majorityvote.jsonl` | majority_vote | `7f72b1a` | 07-31 00:24 → 02:00 | 1600/1600 |
| `results/20260731_020358/results_topk2_llmjudge.jsonl` | llm_judge | `32af2e0` | 07-31 02:03 → 03:44 | 1600/1600 |

3 ディレクトリとも当初 `config.yaml`・`git_head.txt`・`metrics.json` を欠いていた（F5 の provenance は
標準経路 `mise run start` でのみ機能するが，Iter27 は独自の呼び出しで実行されたため．docs/d0004 §6-2）．
**2026-07-31 に事後補完した**（Iter25 で B45 が行ったのと同じ方式．各実行の開始時刻と Iter27 の
コミット時刻が 1 対 1 に対応するため HEAD を一意に確定できた）．補完した `config.yaml` スナップショットは
3 件とも `dispatch_top_k: 2` と意図どおりの `aggregation_method` を持っており，
**設定自体は正しく反映されていて，不成立の原因は閾値ゲートのみであることが独立に裏付けられた**．

### 分析 (実行) (Iter27)

Iter25 基準線（`results/20260730_145356/`，1600 問）との対比．

| 指標 | 基準線 | max_confidence | majority_vote | llm_judge |
|---|---|---|---|---|
| top1_accuracy | 0.555625 | 0.555625 | 0.555625 | 0.555625 |
| single_domain_top1_accuracy | 0.5693333 | 0.5693333 | 0.5693333 | 0.5693333 |
| compound_domain_top1_accuracy | 0.35 | 0.35 | 0.35 | 0.35 |
| compound_domain_set_recall | 0.165 | 0.165 | 0.165 | 0.165 |
| Cohen's κ | 0.5214815 | 0.5214815 | 0.5214815 | 0.5214815 |
| ECE | 0.2040206 | 0.2040206 | 0.2040206 | 0.2040206 |
| fallback_rate | 0.1325 | 0.1325 | 0.1325 | 0.1325 |
| mean_duration_ms | 3626.8 | 3599.3 | 3606.9 | 3751.2 |
| answer_quality（JMMLU1500） | 0.5087 | 0.4960 | 0.4913 | 0.5080 |
| McNemar（対基準線） | — | discordant=0, p=1.0 | discordant=0, p=1.0 | discordant=0, p=1.0 |
| **2 ノードへ dispatch した問題数** | 0 | **0/1600** | **0/1600** | **0/1600** |

ルーティング系の指標が小数点以下すべてで一致し，McNemar の不一致ペアも 3 方式とも 0 件．
`dispatched_domains` の長さが 2 以上の行は 1 件も存在しなかった．

### 分析 (解釈) (Iter27)

**レバー**: `aggregation_method`（`dispatch_top_k=2` を前提）

**判定**: **invalid（実験不成立）**．「集約方式に差が無い」ではなく，
**集約が一度も実行されていない**．

**機序**: `aggregator.select_dispatch_targets()`（`aggregator.py:39`）は
`confidence >= confidence_threshold` で候補を絞ってから top-k を取る．
`routing_method=supervised_classifier` では各ノードが 10 クラス LogisticRegression の
自分のクラスの確率のみを返し，10 ノードの総和は 1 になる．よって 2 ノードが同時に
`>= 0.5` を満たすには p₁ + p₂ ≥ 1.0 が必要で，事実上起こり得ない．

実データでも **2 位 confidence の最大値は 0.4955** であり，閾値 0.5 に一度も到達していない
（mean 0.1407 / median 0.1081 / p99 0.4580）．**この結論はデプロイの成否とは無関係に成立する**．

閾値を下げた場合に 2 ノード目が適格になる件数（同データで逆算）: 0.4→75 件（4.7%），
0.3→230 件（14.4%），0.25→365 件（22.8%），0.2→509 件（31.8%）．
ただし閾値を下げると 1 位側の適格数（＝ fallback しない件数）も同時に動く．
`confidence_threshold` が **fallback ゲートと dispatch 候補ゲートの 2 役を兼ねている**ため，
現行実装では集約方式も fallback 方策も単一レバーとして分離できない．
詳細と対処案は docs/d0004 §3・§5 Y2 を参照．

なお複合設問 100 問はすべて 2 ドメインであり，`dispatch_top_k` が実効 1 である限り
`compound_domain_set_recall` の構造的上限は 0.500（実測 0.165）である．top_k=2 が
実際に効けば上限は 1.000 になる．

**副産物 — 回答品質のノイズ床の確定（d0003 X6 に相当）**:
本イテレーションの 3 実行はルーティングが完全に決定論的で同一だったため，Iter25 基準線と
併せて「生成のランダム性のみが異なる 4 回の反復」になった．これは X6 が計画しながら
未実施だった測定そのものである．

- `answer_quality_accuracy`（JMMLU1500）: 0.5087 / 0.4960 / 0.4913 / 0.5080
- 平均 0.5010，**標準偏差 0.87pt**，範囲 1.73pt，**2SD = ±1.74pt，3SD = ±2.61pt**
- 行単位では **359/1500（23.9%）**が反復間で正誤反転

これまで使ってきた暫定値 1.3pt（n=2，d0002 §6-F）を，n=4 の実測 3SD = 2.6pt へ置き換える．
`.claude/research/config.yml` の `success_criteria` に反映済み．
この基準では Iter26 の回答品質 −1.53pt・End-to-End −1.50pt はノイズと確定し，
E10 の +22.3pt は 3SD の 8 倍以上で堅牢なまま維持される．

### 考察 (Iter27)

**総括**: 約 5 時間の実機実行が no-op に費やされた．これは Iter16・20（E3），Iter21・22（E4），
backlog B35（E7）と**同型の失敗**で，「config を正しく変えて実験も完走したが，その設定を読む
コードに実行が到達しない」というパターンである．のべ 6 イテレーション・10 時間以上が
この型で失われている．恒久対策（計画時のコードパス到達条件の明記，本走前の予備実行による
発火確認，基準線との完全一致を「効果なし」ではなく「不成立」と解釈する既定）を
docs/d0004 §4 に定めた．

**次イテレーションの単一レバー**: **fallback 方策の廃止**（d0003 X5，d0004 Y1）を提案する．
`aggregation_method` の再挑戦（d0004 Y3）は，`confidence_threshold` の二重責務を分離する
コード変更（d0004 Y2，ユーザー確認が必要）を終えるまで着手しない．

Y1 を最優先とする根拠は，**既存データから効果が実測済み**である点にある．
`results/central_iter26/`（閾値なし純 argmax ＝ fallback 廃止相当）と
`results/central_iter26b/`（現行の閾値 0.5 + general への fallback）は，アーキテクチャ・
分類器・データセットが同一で fallback 方策だけが異なる（Iter26 で方策の食い違いに気付いた際の
副産物．backlog B46）．

| 指標 | fallback 廃止 | fallback あり（現行） | 差 |
|---|---|---|---|
| top1_accuracy | 0.5850 | 0.5556 | +2.94pt |
| Cohen's κ | 0.5541 | 0.5215 | +3.26pt |
| answer_quality（JMMLU1500） | 0.5507 | 0.4933 | +5.74pt（3SD の 2.2 倍） |
| mean_duration_ms | 4234.8 | 4558.2 | −323ms |

McNemar（現行 vs 廃止）: discordant 77 件（廃止のみ正解 62・現行のみ正解 15），**p = 1.59e-7**．
fallback が発動した 212 問だけを見ると，general へ送った場合のルーティング正解は **18/212（8.5%）**，
fallback せず argmax のドメインへ送った場合は **65/212（30.7%）**．
現行 fallback は，分類器が迷っている問題を正解率 8.5% の選択肢へ振り替えている．

**iteration_name**: 「fallback 方策の廃止によるルーティング精度・回答品質への影響測定」

**実行上の申し送り**: Y1 の実験では `dispatch_top_k` を 2 から **1 へ戻す**こと
（`confidence_threshold` を下げると候補ゲートも緩むため，top_k=1 に固定して単一レバーを保つ）．

### 停止していた経緯（2026-07-31 に判明）

実験は 07-31 03:44 に 3 本とも完走していたが，`state.json` は `phase="implement", status="running"`
のまま 12 時間放置されていた．watchdog がこれを検知できなかったのは，
**Iter23 の使い捨て heartbeat スクリプト `/tmp/iter23_heartbeat.sh` が 07-30 01:42 から
動き続け，`state.json` の `updated_at` を 120 秒ごとに上書きしていた**ためである．
停止条件のマーカー `/tmp/iter23_start.done` が生成されず，無限ループになっていた．
本セッションで当該プロセス（PID 871683，稼働 1 日 14 時間）を停止した．
詳細と再発防止は docs/d0004 §6-1 を参照．

---

## Iteration 26: 中央集権ルータ比較の再実験（research_frontier 項目4 実施，X2再挑戦）

**背景**: Iter24（X2: 中央集権ルータ比較）はrejected判定だったが，回答生成プロンプト・APIエンドポイント
（`/api/generate`→`/api/chat`，`/api/embed`→`/api/embeddings`）の不一致が原因と判明したため，
`scripts/run_central_experiment.py`を修正し（backlog記録訂正節参照），Iter25の新基準線（1600問）に対して
再実験した．

### 実験 (Iter26, 初回実行)

修正後の初回実行で，分散版（top1=0.5556）に対し中央版がtop1=0.585とむしろ高精度になり，
McNemar p=1.6e-7で有意差が出た．原因調査の結果，**分散版はconfidence_threshold=0.5でargmaxドメインの
確率をフィルタしてからdispatchし，閾値未満ならgeneralのlight_modelへフォールバックする一方，中央版の
初回実装は閾値なしの純粋argmaxで常にdispatchしていた**ことが判明．不一致77件全てが分散版のfallback行
（212件中）と一致しており，「同一classifierなのに結果が違う」という謎は実装バグではなく，2つの実装が
異なる意思決定方策を比較していたことによる単一レバー原則違反だったと分かった（backlog B46）．

**是正**: `run_central_experiment.py`に分散版と同一のconfidence_threshold・fallbackロジック
（`node.py`の`FALLBACK_PROMPT_TEMPLATE`・`FALLBACK_MAX_TOKENS`を再利用）を実装し，アーキテクチャのみを
単一レバーとして分離できるようにした．回帰防止テスト`tests/test_run_central_experiment.py`を追加．

### 実験 (Iter26, 修正後・確定)

**実験ディレクトリ**: `results/central_iter26b/`（1600問，全問完走）
**構成**: `results/20260730_145356/`（Iter25，分散版）と同一データセット・同一classifier・同一
confidence_threshold・同一fallback方策．変更点は「ルーティング判断がどこで実行されるか」（各ノードの
サーバー内 vs 手元のスクリプト内，SSH経由）のみ．

**結果**:

| 指標 | 分散版（Iter25） | 中央版（Iter26修正後） | 判定 |
|---|---|---|---|
| top1_accuracy | 0.555625 | 0.555625 | **完全一致** |
| Cohen's kappa | 0.5214814814814815 | 0.5214814814814815 | **完全一致** |
| McNemar 対比較 | - | discordant_pairs=0, p=1.0 | **完全一致**（不一致0件/1600問） |
| fallback_rate | 13.25% | 13.25% | **完全一致** |
| answer_quality_accuracy | 0.508667 | 0.493333 | -1.53pt（ノイズ床±1.3ptに近い） |
| end_to_end_accuracy | 0.328125 | 0.313125 | -1.50pt（同上） |
| mean_duration_ms | 3626.775 | 4558.229 | **+25.7%（中央版が遅い）** |

**判定**: **adopted**（X2の目的である「アーキテクチャのみを単一レバーとした比較」が今回初めて成立した）．

**仮説1（ルーティング精度の一致）**: **完全に支持**．confidence_threshold・fallback方策を揃えた結果，
1600問すべてでルーティング判定が一致した（McNemar discordant=0）．embedding（wafl500/wafl502間でbit単位
一致，実機確認済み）・classifier（sha256完全一致）が同一であることも直接確認済み．

**仮説2（プローブ速度の優位性）**: **不支持**．中央版は分散版より約25.7%遅い．分散版のprobeオーバーヘッド
は`mean_other_ms`=137.5ms/問と小さく，中央版はSSH経由の2回の往復（embedding・回答生成）にそれぞれ
接続確立コストがかかるため，probeオーバーヘッドの削減分を上回るコストを支払っている．
**これは「6GB VRAM制約下でSSH越しに実装した」今回のcentral_router特有のコストであり，設計書が想定する
「1台にモデル常駐させた理想的な中央集権」とは異なる実装であることに注意．**

**仮説3（VRAM制約）**: 未計測（当初計画通り，実施していない．今回の実装はSSH経由で既存の分散ノードの
Ollamaを間借りしているため，central側自体はVRAMを消費しない構成になっている．真のVRAM制約比較には
1台に10 LoRAを常駐させる実装が別途必要で，今回のスコープ外とする）．

**回答品質・End-to-End**: 中央版がやや低い（-1.5pt程度）．d0003のノイズ床推定（約1.3pt）に近い差であり，
生成の確率的性質（temperature未指定=Ollamaデフォルト，サンプリングによる揺らぎ）に起因する可能性が高い．
非退行の判定基準（3SD等）は未確立のため，これ単体では有意とは言い切れない．

**研究の問い2への回答**: 「分散であることのコストは小さいか」という問いに対し，**ルーティング精度は
アーキテクチャに依存せず完全に一致する一方，レイテンシは分散版の方が速い**という結果が得られた．
これは「中央集権ルータが分散メッシュに対してコスト面で優位」という一般的な想定とは逆であり，
VRAM制約下でSSH越しに実装せざるを得ない中央集権アーキテクチャの構造的な弱点を実証的に示している．

**次イテレーションへの示唆**: research_frontier項目4は完了．次はItem5（集約方式比較，Iter27）へ移行する
（backlog B47参照）．

---

## Iteration 25: 評価データセット拡充後の基準線再取得（research_frontier 項目2 実施）

**背景**: ユーザー指示「research_frontier 項目を全て実装・設定せよ」を受け，項目2（複合ドメイン評価データ
セットの拡充）を実施した．`_COMPOUND_QUESTIONS` を20問（medical/legal/education の3組み合わせに偏重）
から100問（10ドメイン・43組み合わせ）へ拡充し，データセットを1520問→1600問へ再生成した．JMMLU由来の
単一ドメイン設問1500問・分類器訓練データ（1427件，ハッシュ完全一致で確認済み）は無変更．

**単一レバー原則との関係**: これは通常の「レバーを1つ振る」実験ではなく，Iter15・Iter23と同種の基盤整備
（データセット自体の変更）である．データセットが変わったため，Iter23のX1基準線（1520問）はもはや
直接比較できず，新データセットでの基準線再取得が必要となった．

### 実験 (Iter25)

**実験ディレクトリ**: `results/20260730_145356/`
**データセット**: 1600問（単一1500 + 複合100，全問完走）
**構成**: E6+E10（`routing_method=supervised_classifier`, `confidence_signal_method=self_report`,
`expert_model=expert-mesh-{domain}-lora`，Iter23と同一，config.yamlの変更なし）

**結果**:

| 指標 | Iter23（1520問，参考） | Iter25（1600問） | 解釈 |
|---|---|---|---|
| single_domain_top1_accuracy | （未分離計上） | 0.5693 | Iter18以来の既知値と完全一致，単一ドメインJMMLU1500問のルーティングは無変更 |
| top1_accuracy（全体） | 0.5651 | 0.5556 | 低下は複合設問の母数が20→100（1.3%→6.25%）へ増えたことによる合成比率の変化であり，ルーティング自体の劣化ではない（下記参照） |
| Cohen's kappa（単一ドメインのみ） | 0.5215 | 0.5215 | 完全一致．単一ドメイン設問のルーティングは決定論的で不変 |
| compound_domain_top1_accuracy | 0.25〜0.95（n=20，d0003指摘の通り信頼できない） | 0.35（n=100） | 初めて統計的に議論できる規模で測定（d0003 X4の主目的を達成） |
| compound_domain_set_recall | 0.125（n=20） | 0.165（n=100） | dispatch_top_k=1のカバレッジ上限問題（d0003指摘）がn=100でも再確認された |
| answer_quality_accuracy | 0.508667 | 0.508667 | 小数点以下まで完全一致（1500問のJMMLU抽出照合部分は無変更なので当然） |
| end_to_end_accuracy | 0.318421 | 0.328125 | +0.97pt，ノイズ床（約1.3pt，d0003 X6）の範囲内 |
| ECE | 0.192654 | 0.204021 | +0.0114，複合設問が増えたことによる母集団変化．ノイズ判定は要検討（複合設問はconfidence較正が悪い可能性） |
| fallback_rate | 13.16% | 13.25% | ほぼ同一 |
| mean_duration_ms | 3555.6 | 3626.8 | ほぼ同一 |
| graded_row_count（LLM-as-judge対象含む） | 該当なし | 1600/1600（全問採点） | 複合100問すべてLLM-as-judgeで採点済み |

**判定**: **adopted**（新基準線として確定）．single_domain_top1_accuracy・Cohen's kappa（単一ドメイン
限定）・answer_quality_accuracyが小数点以下まで完全一致しており，データセット拡充がルーティング精度・
回答品質の測定に悪影響を与えていないことを確認した．overall top1_accuracyの低下（0.5651→0.5556）は，
統計的に信頼できなかった20問の複合設問（Iter15で0.95という偽高値を生んだのと同じ母集団）から，
43通りの組み合わせを持つ100問の複合設問に切り替わったことによる合成比率の変化であり，回帰ではない．

**次イテレーションへの示唆**: 以後の比較（Iter26中央集権ルータ再実験・Iter27集約方式比較）はこのIter25
（1600問）を基準線として使う．compound_domain_set_recall=0.165（n=100）は，dispatch_top_k>1による
複合設問カバレッジ改善の必要性を裏付けており，Iter27の集約方式比較実験の動機と直接つながる．

---

## 記録訂正・commit 漏れの是正（2026-07-30，`/research-cycle continue` 実行時）

**背景**: Iter24 完了後の `continue` 呼び出し時，`git status` で `scripts/run_central_experiment.py`（未追跡）・
`config.yaml`（`central_router` 節，未commit）・`.claude/research/state.json`（heartbeat のみの軽微な差分）が
working tree に残っていることを発見した．Iter24 完了コミット（`ee1d549`）は `.claude/research/*` のみを
含んでおり，Iter24 の単一レバー（`routing_architecture=central_router`）を実装したコード自体は
一度も git に commit されていなかった．過去の記述は書き換えず，本節を追記として残す．

**発見1（実装内容が journal の記述と乖離）**: 本節より前の「実装 (Iter24)」節は `scripts/run_central_experiment.py`
を「229行，`OllamaClient.embed()`/`generate()` をローカルでそのまま利用」と記述しているが，working tree に
残っていた実際のファイルは 411 行であり，`SshEmbeddingClient`／`HttpOllamaGenerator`（`config.yaml:central_router`
の `ssh_user`/`domain_nodes` を読み，SSH 経由で各ドメインの担当ノードへ curl する方式）に置き換わっていた．
これは「調査 (Iter24)」節が指摘した VRAM 制約（6GB に 10 LoRA を1台で載せられない）に対応するための
実装上のピボットと推測されるが，その変更判断・理由は journal に一度も記録されていない．
Slack の「フェーズ3: 実装」報告は「253行」と述べており，journal（229行）とも実ファイル（411行）とも
一致しない．3者の食い違いは，実装が複数回改訂されたにもかかわらず記録が都度更新されなかったことを示す．

**発見2（ruff warning の不一致）**: journal・Slack・Notion はいずれも「ruff 0 warning」としているが，
現在の working tree のファイルには未使用 import（`os`, `subprocess`）による F401 が 2 件ある．
Iter24 の判定（rejected）自体は主基準（top1/kappa/McNemar）に基づくため，この lint 差分は判定を
覆すものではない．

**是正内容**: 上記のコード一式（`scripts/run_central_experiment.py` 新規追加，`config.yaml` の
`central_router` 節追加）は，実際に Iter24 の実験を生成した実体であるため，lint 警告を含めて
**そのままの内容で** git commit した（実験の再現性を優先し，事後的な整形は行わない）．
`.claude/research/state.json` の heartbeat 差分（`updated_at`）は現在時刻へ更新し，`last_commit` を
本コミットのハッシュへ同期した．

**次回への申し送り**: rc-implementer は，計画からの実装方針の変更（今回でいう local→SSH のピボット）が
発生した場合，その理由を「実装 (IterN)」節に都度追記すること．イテレーション完了時の commit 検証
（SKILL.md 記載）は `.claude/research/` 配下だけでなく，そのイテレーションで変更した実コード・設定ファイルが
実際に commit されているかも対象に含めるべきである．

---

## 敵対的総点検・追加修正（2026-07-30，`/research-cycle continue` 実行前）

**背景**: 直前の「記録訂正・環境修復」節（2026-07-29）で行った F1〜F5 の修正自体が正しいかを，
独立した subagent によるレビューと自己点検で敵対的に総点検した．詳細は
`.claude/research/backlog.md` の B40 を参照．要点のみ記す．

**発見1（重大・修正済み）**: `.claude/research/config.yml` の E4（`self_consistency_semantic`）・
E5（`p_true`）の記述が，真の no-op である E3・E7 と同列に書かれていたため誤解を招く状態だった．
実際には，現在の HEAD（`30e3627`，Iter22 の分岐順序修正が反映済み）でこれらを設定すると，
E6（supervised_classifier）の分類器分岐に到達できなくなり **E6 を丸ごと上書きする**．
「no-op」ではない．時系列の誤り（Iter21/22 は E4 が動いた上で退行したのではなく，分岐順序修正が
未適用/未デプロイだったため E4 が 1 度も実行されなかっただけ）も訂正した．

**発見2（重大・ユーザー承認の上で修正済み）**: `state.json.iteration` が `"Iter22"`（無効判定済み）
のまま，SKILL.md が定める「イテレーション完了時の初期化」が未実施だった．前回は `current_lever`
のみに着目し実害なしと判断していたが，`rc-experimenter.md` が「実験ディレクトリ名を現イテレーション
番号から決める」と明記しており，このまま continue すると次の実験が誤って Iter22 として記録される
リスクを見落としていた．ユーザー承認を得て，`iteration`: `Iter23` へインクリメント，
`current_lever`/`experiment_dir`/`experiment_deadline`/`iteration_thread_ts`: null，
`notion_toggle_created`: false，`iteration_name`: null に更新した．

**発見3・4（軽微・修正済み）**: `tools/smoke_check.py` の `_SIGNAL_FIELD_EXPECTATIONS` に到達不能な
dead entry（`"semantic_entropy"` キー，実際の値は `"self_consistency_semantic"`）を削除．
`run_experiment.py` の `write_text` に `encoding="utf-8"` を追加．

**次期 rc-investigator への申し送り**: `state.json` は既に Iter23 として初期化済みである．
このイテレーションの調査を「### 調査 (Iter23)」として記録すること．

---

## 記録訂正・環境修復（2026-07-29，`/research-cycle continue` 実行前の総括調査）

**背景**: `docs/d0002_research_cycle_findings_2026-07.md`（Iter1〜22 の知見総括）・
`docs/d0003_next_experiments_2026-07.md`（次の実験計画）を新規作成し，journal・backlog・
config.yaml・config.yml・http_server.py の実データ突合を行った．以下は journal 側の記録誤りの訂正，
および continue 実行前に対処した環境修復である．過去の記述は書き換えず，本節を追記として残す．

### 訂正 1: E3（`confidence_elicitation`）の採用判定を取り下げ

Iter20 の「同点タイ 82.83%→0.00%，ECE 0.7388→0.1927 の決定的改善」は，**すべて Iter17 の
E6（supervised_classifier）導入時に既に起きていた変化**である．`http_server.py:_estimate_probe_confidence()`
は排他的な if 連鎖で，`routing_method=supervised_classifier` が先に return するため
`confidence_elicitation` の分岐には到達しない（d0002 §6-B）．E3 の有効な測定は Iter16 の 1 回のみで，
そのときの結果は top1 0.2059（McNemar p=0.0783，有意差なし），ECE は 0.7146→0.7388 と悪化していた．
**「採用」の判定は取り下げ，D1（判定保留．設定自体は害をなさない）として backlog に記録した．**

### 訂正 2: ECE の正しい系列

10-bin・confidence 非 null 行を対象に単一実装で再計算した結果，Iter17 以降は **0.1927 で不変**である
（決定論的ルーティング下での再実行は新しい情報を生まない．d0002 §6-A）．

| 実験 | 正しい値 | journal 上の誤記載 |
|---|---|---|
| Iter17 | 0.1927 | 0.2118（不一致） |
| Iter21 | 0.1927 | 0.1903 / 0.1673（journal 内で 2 通りの誤記載） |

Iter21 の「0.1903 へわずかに改善」という記述は誤りで，実際の変化は 0.0000 である．

### 訂正 3: `top1_accuracy` と `single_domain_top1_accuracy` の取り違え

Iter18 Phase C（domain LoRA 採用）の `top1_accuracy` は **0.5651** であり，journal が Iter19/20 の
計画根拠として使っていた **0.5693 は `single_domain_top1_accuracy`（単一ドメイン 1500 問のみの値）**
である．「E10 で top1 が 0.5651→0.5693 改善した」という記述は誤りで，実際は McNemar 不一致 0/1520 で
**完全に不変**（journal 自身も別箇所で不一致 0/1520 と記録しており内部矛盾していた）．

### 環境修復（`/research-cycle continue` 実行前に対処．d0003 第1段階 F1・F1-b 相当）

1. **`config.yaml`**: Iter19 で棄却された `expert_model=qwen3.5:4b-q4_K_M`（全10ノード）が HEAD に
   残置されていたため，Iter18 で採用された `expert-mesh-{domain}-lora` へ戻した．また Iter21/22 で
   無効と判明した `confidence_signal_method=self_consistency_semantic` を `self_report` へ戻した
   （制約: この値以外だと `routing_method=supervised_classifier` の分岐に到達せず分類器が無効化される．
   d0002 §6-D）．**前提条件を実機で確認済み**: wafl500〜509 の Ollama に対応する
   `expert-mesh-{domain}-lora` モデルが全10ノードとも登録済みであることを確認した（2026-07-29）．
2. **`.claude/research/config.yml`**: `levers` 節の前提コメント（Iter15 時点のまま古くなっていた）を
   Iter22 時点の実態へ全面更新し，E3・E4・E7 の no-op / 排他構造の注記を追記した．
3. **未対応（次イテレーション以降の課題）**: docs/d0003 F2（デプロイ検証ゲート）・F3（metrics.py への
   ECE/AUROC/Brier/同点率/分散統合）・F5（再現性マニフェスト）は本セッションで別途着手中．F3 完了までは
   confidence 信号系レバーの判定に success_criteria (4) の指標を手計算に頼らざるを得ない．

**次期 rc-investigator/rc-planner への申し送り**: 上記により，continue 再開後の実験は最良既知構成
（E6 supervised_classifier + E10 domain_lora）から始まる．次に着手すべき優先順位は
`docs/d0003_next_experiments_2026-07.md` §0 を参照（第2段階 X1 基準線再取得 → 第3段階 X2 中央集権
ルータ比較・X4 複合ドメイン評価・X5 fallback 見直し）．

---

