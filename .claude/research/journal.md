## Iteration 28: fallback 方策の廃止によるルーティング精度・回答品質への影響測定

### 計画 (Iter28)

**仮説**: `confidence_threshold` を `0.5→0.0` に下げ，confidence ベースの fallback
（general ノードの light_model への退避）を実質的に無効化すると，ルーティング精度
（top1_accuracy・Cohen's κ）・回答品質（answer_quality_accuracy）が向上し，
mean_duration_ms も短縮する．`results/central_iter26/`（fallback 廃止相当）vs
`central_iter26b/`（現行）の既存比較（アーキテクチャは異なるが分類器・データセットは同一）で
観測された差分が，分散版で config のみを変えても同じ大きさで再現されるかを検証する．

**単一レバー**: `fallback_policy`（`.claude/research/config.yml` の levers 名．実体は
`config.yaml` の `confidence_threshold`）

- `confidence_threshold: 0.5 → 0.0`（唯一の実験対象レバー）

**直近の最良構成へ固定するための復元（レバーではなく，Iter27 の残骸整理）**:

- `dispatch_top_k: 2 → 1`（`confidence_threshold` を下げると `aggregator.py:39` の
  dispatch 候補ゲートも同時に緩むため，`top_k=1` に固定しない限り単一レバー原則が崩れる．
  調査フェーズの申し送りどおり）
- `aggregation_method: llm_judge → max_confidence`（`dispatch_top_k=1` では no-op だが，
  Iter27 で使われたまま残っている値なので整理する）

**変更ファイル・キー**（他のキーは一切変更しない）:

- `config.yaml:5` `confidence_threshold: 0.5` → `0.0`
- `config.yaml:52` `dispatch_top_k: 2` → `1`
- `config.yaml:63` `aggregation_method: llm_judge` → `max_confidence`

**固定する構成（直近の最良構成，Iter25/26 と同一）**: `routing_method=supervised_classifier`，
`confidence_signal_method=self_report`，`confidence_elicitation=top_k_with_probs`（no-op），
`expert_model=expert-mesh-{domain}-lora`（domain_count=10），`light_model=qwen3.5:4b-q4_K_M`，
評価データセットは Iter25 の 1600 問（変更なし）．

**到達条件（コードパス確認，d0004 §4 対策A）**: `node.py:216` の `run_ask_flow()` から
`aggregator.py:28-40` の `select_dispatch_targets()` が呼ばれ，
`confidence >= confidence_threshold` で候補を絞ってから top-k を取る．閾値を `0.0` にすると
`predict_proba` の出力（値域 `[0,1]`）は常にこの条件を満たすため，毎回全 probe_responses が
適格になり，`dispatch_top_k=1` なら必ず argmax の 1 件が返る．`node.py:219` の
`if not targets:`（fallback 発火条件）は，全ノードの probe 自体が失敗する真の異常系でしか
成立しなくなる．`run_experiment.py:87` も同じ関数を再利用するため，1600 問バッチ実行で
確実にこの経路を通る．`http_server.py:201` の `NodeState.confidence_threshold` は格納のみで
参照されない未使用フィールドであり，到達を阻害しない．**到達を阻む分岐は存在しない**．

**予備実行（d0004 §4 対策B，本走前に必須）**: 先頭 20 問程度を実行し，
`results.jsonl` の全行で `dispatched_domains` の長さが 1 であること，かつ fallback 発生件数が
0 件であることを確認する．1 件でも fallback が発生していれば `confidence_threshold` の反映漏れ
（Iter16/20/21/22/27 と同型のデプロイ失敗）を疑い，本走前に原因を特定してから本走に進むこと．

**評価方法**: 1600 問本走を 1 回実行し（`experiment.timeout_min=150` の範囲内，実測目安 約90分），
`mise run analyze` で Iter25 基準線（`results/20260730_145356/`）との比較を行う．

- 主指標: top1_accuracy・Cohen's κ の McNemar 対比較（α=0.05，Wilson 95% CI 併記．success_criteria (1)）
- 副指標: answer_quality_accuracy・end_to_end_accuracy（3SD=2.61pt 未満はノイズと判定．success_criteria (5)）
- mean_duration_ms（速度の変化）
- fallback 発生件数（0/1600 になっていることの直接確認．到達条件が満たされた証拠でもある）
- per-domain precision/recall の非退行確認（success_criteria (2)）

**期待効果**（`results/central_iter26/` vs `central_iter26b/` の実測を分散版での期待値として使う．
Iter26 で「アーキテクチャを変えてもルーティングは完全一致」が実証済みのため，同じ大きさの差が
出ることが期待値だが，一致しないこと自体を無効判定の理由にはしない）:

- top1_accuracy: 0.5556 → 0.585 相当（+2.94pt）
- Cohen's κ: 0.5215 → 0.5541 相当（+3.26pt）
- answer_quality_accuracy: 0.4933 → 0.5507 相当（+5.74pt，3SD=2.61pt の 2.2 倍）
- mean_duration_ms: 4558 → 4235 相当（−323ms，速くなる）

**成功条件**:

1. top1_accuracy が McNemar 検定で基準線に対し有意に改善（p<0.05）し，Wilson 95% CI が
   重ならないこと．
2. fallback 発生件数が 0/1600 であることを直接確認できること（レバーが実際に発火した証拠）．
3. answer_quality_accuracy の変化が 3SD=2.61pt を超えて改善方向であること（悪化していないこと）．
4. per-domain precision/recall の CI 下限が Iter25 基準線の CI 下限を下回らないこと
   （非退行．success_criteria (2)）．

**注意点**: 観測された効果量が事前実測（central_iter26 vs 26b）と大きく異なる場合
（符号が逆転する，効果量が半分以下になる等）は，それ自体を「分散/中央のアーキテクチャ差が
確率境界付近の結果に影響する」という新知見として記録し，無効判定の理由にしないこと．
`compound_domain_set_recall`（現状 0.165）は `dispatch_top_k=1` のままなので構造的上限 0.500 は
変わらないはずであり，変化があれば fallback 廃止が複合設問ルーティングに与えた副次効果として
別途記録する．

**人間判断が必要な論点（backlog に残す，B48 の既存項目を維持）**: fallback という設計思想自体を
撤廃するかどうかの論文上の位置付けは，本実験の結果だけでは決められない．引き続き B48 の
要レビュー項目として残す．

### 実装 (Iter28)

**変更ファイル**: `config.yaml`（計画どおり3行のみ）

- `confidence_threshold: 0.5 → 0.0`
- `dispatch_top_k: 2 → 1`
- `aggregation_method: llm_judge → max_confidence`

**commit**: `d87c006`（`config.yaml` のみを含む単独コミット）．

**テスト/リンタ**: `uv run pytest -q` 211 passed, 2 skipped（回帰なし）．`ruff check .` の既存
警告2件（`scripts/prepare_lora_training_data.py`）は HEAD 時点から存在する今回変更と無関係な
既知の問題であり，`config.yaml` は YAML のため ruff の対象外．

**デプロイ確認（Iter16/20/21/22/27 のデプロイ漏れ再発防止のため実施）**: `mise run deploy` で
実機10ノード（wafl500〜509）へ配布し `app` コンテナを再起動．`tools/smoke_check.py --check hashes`
で全10ノードの `config.yaml` がデプロイ済みコンテナと一致することを確認．`--check probe` も正常．

**予備実行（本走前の必須確認）**: 先頭20問を実行し，全20行で `dispatched_domains` の長さが1，
`used_fallback=False` であることを確認した．**fallback が実質的に無効化されていることの
直接証拠**．予備実行の一時ファイルは削除済み（`results/` には残していない）．

→ 実験フェーズ（1600問本走）に進める状態．

### 実験 (Iter28)

**実験ディレクトリ**: `results/20260731_162722/`．1600問完走（16:27:22→17:58:05，実測約90.7分，
`timeout_min=150` 範囲内）．10ノード（wafl500〜509）のコンテナログに error/traceback/OOM/killed
該当0件，`dispatch_failed=True` の行も0/1600．

**到達確認**: `dispatched_domains` の長さは全1600行で1（`Counter({1: 1600})`），
`used_fallback=True` の行は0件．config変更（`confidence_threshold=0.0`, `dispatch_top_k=1`）が
実データ経路に発火した直接証拠．

**provenance**: `config.yaml` は現HEAD（`d87c006`）と完全一致．`git_head.txt` は `9b7f393`
（`mise run setup` 実行時点のHEAD．config.yamlはbind mountで都度読み込まれる仕様のため矛盾ではない．
9b7f393〜d87c006間の6コミットはconfig.yaml/journal/docsのみでアプリケーションコード変更なしを
`git show --stat` で確認済み）．**申し送り**: `git_head.txt` は config 変更コミットを反映しない
既知の限界があり，将来の分析で `git_head.txt` の値のみから config 内容を推測しないこと．
`metrics.json`／`axis23_metrics.json` は生成・格納済み．

### 分析 (実行) (Iter28)

Iter25 基準線（`results/20260730_145356/`）との対比．

| 指標 | Iter28（本走） | Iter25 基準線 |
|---|---|---|
| top1_accuracy | 0.585（Wilson 95% CI [0.5607, 0.6089]） | 0.555625（CI [0.5312, 0.5798]） |
| Cohen's κ | 0.554074 | 0.521481 |
| single_domain_top1_accuracy | 0.598667 | 0.569333 |
| compound_domain_top1_accuracy | 0.38 | 0.35 |
| compound_domain_set_recall | 0.19 | 0.165 |
| answer_quality_accuracy | 0.546667 | 0.508667 |
| end_to_end_accuracy | 0.31625 | 0.328125 |
| mean_duration_ms | 3394.894 | 3626.775 |
| fallback発生件数 | **0/1600** | 212/1600 |
| dispatch_failure_rate | 0.0 | — |

McNemar対比較（1600問ペア）: discordant_a_only（新側のみ正解）=62，discordant_b_only（基準線側のみ
正解）=15，discordant_pairs=77，chi2=27.4805，**p_value=1.5868×10⁻⁷**．

ドメイン別precision/recall（Wilson CI付き，全10ドメイン算出済み）: `general` ドメインのrecallのみ
CI下限が基準線を下回った（新 90/164 CI [0.4724, 0.6230] vs 基準線 105/164 CI [0.5644, 0.7097]）．
precisionは逆に大幅改善（新 90/138 CI [0.5696, 0.7265] vs 基準線 105/335 CI [0.2661, 0.3650]）．
他9ドメインはCI下限が基準線以上か同程度．良否判定は次の分析(解釈)フェーズで行う．

**運用上の注意**: `mise run analyze` はタイムスタンプのみを引数に取る仕様（フルパスを渡すと
`results/results/...` の誤ネストが発生する）．実行時に一度誤り，即座に気づいて訂正・削除済み．
実験データ自体への影響なし．

### 分析 (解釈) (Iter28)

成功条件（計画節）1〜4 を順に判定する．

**条件1（top1_accuracy: McNemar p<0.05 かつ Wilson 95% CI 非重複）— 実質的に成立，ただし
CI 非重複のみ字義的に僅かに未達（方法論上の注記あり）**

McNemar は discordant=77（新側のみ正解62・基準線側のみ正解15），chi2=27.4805，
**p=1.5868×10⁻⁷** で α=0.05 を大きく下回り，主基準は極めて強く成立する．

一方 Wilson 95% CI は新 [0.5607, 0.6089]・基準線 [0.5312, 0.5798] で，
再計算した重複区間は [0.5607, 0.5798]（幅 1.91pt，各CI幅約4.8〜4.9ptの4割弱）であり，
字義どおりには「重ならない」を満たさない．

この不一致は，比較対象が**同一1600問に対する対応のある（paired）2条件の正誤**であることに
起因する方法論上の問題だと判断する．Wilson CI は2群を独立標本とみなした周辺分布の区間であり，
paired 設計が持つ「1523/1600問（95.2%）で新旧の正誤が一致している」という強い相関情報を
使わない．そのため独立標本前提の周辺CIは実際より広く出て重なりやすく，paired 検定である
McNemar（一致ペアを除き不一致ペアのみで検定する）の方がこの設計には統計的に正しく，
検出力も高い．p=1.59×10⁻⁷ という極めて小さい値は，1.91pt という僅かな周辺CI重複と矛盾しない
（paired 相関を考慮すれば偶然の重複ではなく効果が実在する）．

**判定**: 条件1の実質的な意図（有意な改善）は強く支持される．ただし計画文の字義（CI 非重複を
必須とする書き方）は将来のpaired比較で同様の食い違いを生みうるため，次回計画時の申し送り事項として
残す（本判定を覆す理由にはしない）．

**条件2（fallback発生件数 0/1600）— 明確に成立**

`used_fallback=True` の行は0件，`dispatched_domains` の長さは全1600行で1．レバーが実データ経路に
発火した直接証拠であり，二値条件として曖昧さなく満たされている．

**条件3（answer_quality_accuracy の変化が3SD=2.61ptを超えて改善方向）— 明確に成立**

実測差は +3.8pt（0.508667→0.546667）で，3SD=2.61ptの約1.46倍，ノイズ床（σ=0.87pt換算で約4.4SD）を
大きく超える改善方向の変化であり，ノイズでは説明できない．

**条件4（per-domain precision/recallのCI下限が基準線を下回らない・非退行）— `general`ドメインの
recallのみ字義上違反．ただし構造的要因によるものと判断し，独立した性能劣化とは区別する**

`general`のrecallのみCI下限が基準線を下回った（新 [0.4724, 0.6230] vs 基準線 [0.5644, 0.7097]，
下限差 約9.2pt）．他9ドメインは違反なし．以下の理由により，これを「新配置が`general`ドメインの
識別に一般的に弱くなった」ことの証拠ではなく，**fallback 廃止という単一レバーが構造的に
生む必然的な副作用**と判断する．

1. **変化の起点は数学的に212行に限定される**．今回のレバーは `confidence_threshold` 未満だった
   行（基準線で212/1600）の dispatch 先だけを変える．confidence≥0.5だった残り1388行は基準線・
   新条件のいずれでも argmax dispatch のままで変化しない．したがって全10ドメインのprecision/recall
   の変化は，数学的に必ずこの212行の部分集合内でのみ生じる（McNemar discordant=77≤212 と整合）．
2. **`general`はfallbackの唯一の送り先であること自体が，このドメインの recall 比較を非対称にする**．
   基準線では，真のドメインが`general`かつ低確信（212行の一部）だった問題は，argmaxの予測に
   関わらず機械的に`general`へ送られるため，ほぼ自動的に正解として recall に計上される．
   新条件ではこの「安全網」が外れ，同じ問題が argmax 予測に委ねられる．真のドメインが`general`の
   低確信問題のうち argmax が`general`を指さない分だけ，recall が下がる．これは基準線側の recall が
   fallback という機構によって`general`のみ人為的に嵩上げされていたことの反映であり，新条件側が
   `general`の識別に劣化したことを意味しない．
3. **同一の212行から生じたprecisionの改善が，この解釈と整合する**．`general`のprecisionは
   0.3134→0.6522（CI下限 0.2661→0.5696）へ大幅改善しており，「確信度に関わらず`general`へ
   誤って送られていた他ドメイン問題」が減ったことを直接裏付ける．recallの低下とprecisionの
   大幅改善が同じ212行内で表裏一体に生じているのは，fallbackの撤廃が引き起こす構造変化として
   一貫している．
4. **経路変化は決定論的で，生成のサンプリング揺らぎ（3SDノイズ床）とは無関係**．ルーティングは
   確率的分類器のargmaxで決まり，同一confidenceに対しては常に同一の出力になるため，この15行
   （105→90）の recall 低下は再現性のある構造効果であり，測定ノイズではない．

**判定**: 条件4は`general`ドメインのrecallについて字義上は違反しているが，違反の原因は
レバー自体が意図する機構変化（fallbackという安全網の撤廃）に完全に内在しており，他9ドメインへの
波及や独立した性能劣化の証拠はない．これは「見過ごしてよい」という意味ではなく，
**fallback廃止のトレードオフとして明示的に記録し，人間判断（backlog B48）に委ねるべき副作用**
として扱う．

**end_to_end_accuracy（0.31625 vs 0.328125，差 −1.19pt）の判定**

3SD=2.61ptの範囲内（|-1.19pt| < 2.61pt）であり，**ノイズと判定する**．軸①（top1_accuracy・κ）は
決定論的なため3SDノイズ床の対象外だが，end_to_end_accuracyはanswer_quality同様に生成の
確率的性質を含む軸②③指標であり，config.yml success_criteria (5) の適用対象である．
唯一悪化していた指標だが，統計的に有意な悪化ではない．

**事前実測（central_iter26 vs central_iter26b）との整合性チェック**

| 指標 | 事前実測（central比較） | 実測（分散版，Iter28） | 差 |
|---|---|---|---|
| top1_accuracy | +2.94pt | +2.9375pt | ほぼ完全一致 |
| Cohen's κ | +3.26pt | +3.2593pt | ほぼ完全一致 |
| answer_quality_accuracy | +5.74pt | +3.80pt | −1.94pt（乖離） |
| mean_duration_ms | −323ms（4558→4235，−7.09%） | −231.9ms（3626.8→3394.9，−6.39%） | 相対変化率はほぼ一致 |

top1・κは事前実測とほぼ完全一致し，Iter26で確認済みの「ルーティング判定はアーキテクチャに
依存しない」という知見をfallback廃止の効果についても裏付ける．mean_durationは絶対値では
central版がSSHオーバーヘッド分だけ常に大きい（Iter26既知）ため単純比較できないが，相対変化率
（-7.09% vs -6.39%）で見ればほぼ一致する．

answer_qualityの乖離（実測+3.8pt が事前推定+5.74ptより1.94pt小さい）は，**3SD=2.61ptの
ノイズ床の範囲内**である．すなわち，この乖離は「分散/中央のアーキテクチャ差が新たに効果へ
影響した」と断定できるほど大きくなく，既知の生成サンプリング由来ノイズで説明可能な範囲に
収まる．計画の注意点（事前実測と大きく異なる場合は新知見として記録）に該当する規模の乖離では
ないため，新知見としては記録せず，「事前実測とおおむね整合」と結論する．

**複合設問系（成功条件外の副次観察）**: `compound_domain_top1_accuracy` 0.35→0.38（+3pt），
`compound_domain_set_recall` 0.165→0.19（+2.5pt）．計画が予告した構造的上限（`dispatch_top_k=1`
なので0.500で不変）は変化していないが，上限内での実測値はわずかに改善方向．ただしn=100と
小標本であり，この差だけで有意性を主張できる規模ではない．参考情報として記録するに留める．

**総合判定：adopted**

根拠：(1) 主基準（top1_accuracy McNemar，p=1.59×10⁻⁷）が極めて強く成立し，Wilson CIの僅かな
周辺重複はpaired設計特有の方法論上の理由で主基準の成立を覆さない．(2) fallback発生0件を直接
確認．(3) answer_quality改善+3.8ptはノイズ床3SD=2.61ptを明確に超える．(4) 唯一の非退行違反
（`general`ドメインrecall）はレバー自体が意図する機構変化に内在する構造的トレードオフであり，
同じ212行から生じたprecisionの大幅改善と表裏一体であって，独立した性能劣化ではないと判断した．
end_to_end_accuracyの悪化（−1.19pt）はノイズ床未満で有意でない．事前実測との差分はκ・top1で
ほぼ完全一致，answer_qualityの乖離もノイズ床の範囲内であり，事前推定を裏付ける結果である．

**次フェーズ（rc-reflector）への申し送り**:
- `general`ドメインrecallのトレードオフをbacklog B48の「fallback設計思想の論文上の位置付け」の
  議論に統合し，「recall低下・precision大幅改善という表裏一体の副作用」として明示すること．
- 条件1の計画文（McNemar有意 かつ Wilson CI非重複の両方を必須とする書き方）が，paired比較では
  今回のように食い違いうるという方法論上の注記を，今後の成功条件の書き方に反映するかどうかを
  検討すること．
- 追加反復は不要と判断する（fallback発生0件という二値条件・McNemarのp値・answer_qualityの
  ノイズ床超過のいずれも確信度が高く，n=1の本走で十分な統計的根拠が得られている）．

### 考察 (Iter28)

**単一レバーの判定: 採用（adopted）**．rc-analyst の「分析 (解釈)」節の総合判定を確定させる．
成功条件4項目のうち，条件1（top1_accuracy の有意改善）・条件2（fallback 0/1600 の直接確認）・
条件3（answer_quality の3SD超過改善）の3項目は疑義なく成立している．条件4（非退行）は
`general` ドメインの recall のみ CI 下限を割ったが，同一212行内で precision が大幅改善しており
（0.3134→0.6522），fallback という安全網の撤廃が構造的に生む必然のトレードオフであって，
新配置が `general` の識別に一般的に劣化したという独立の証拠ではないと判断する．この判定は
覆さない．

**得られた学び（次回以降に活きる非自明な点）**:

1. **paired 比較で McNemar と Wilson CI の周辺重複が食い違いうる**．今回 McNemar は
   p=1.59×10⁻⁷ で極めて強く有意なのに，独立標本前提の Wilson 95% CI は1.91pt重なった．
   同一問題集合に対する対応のある2条件比較では，周辺CIの重複判定は保守的すぎる（paired相関を
   使わないため）ので，**次回以降 success_criteria の書き方を「McNemar 有意 かつ Wilson CI
   非重複」という AND 条件で固定しない**．paired 設計だとあらかじめ分かっている実験では，
   計画時点で「主基準は McNemar，Wilson CI は参考情報」と明記する運用に改める．
   （config.yml success_criteria (1) の見直し候補として記録．次回 rc-planner が判断する.）
2. **fallback は精度指標上「安全網」ではなく「識別困難なケースを低正解率の選択肢へ機械的に
   振り替える処理」だった**（8.5% vs argmax 30.7%）．d0002 §8-3の指摘が今回初めて分散版実機で
   統計的に裏付けられた．
3. **fallback廃止によるドメイン別の非対称性は，fallback の送り先が単一ドメイン（general）に
   固定されていることの必然的な帰結**であり，一般的な「新配置は non-general に強く general に
   弱くなった」という解釈をしないこと．次に fallback関連の指標を見るときは，常に「fallback対象
   だった行の集合」に絞って解釈する視点を保つ．
4. **事前実測（central_iter26 vs 26b，中央集権アーキテクチャ）と実測（分散版）の整合性**:
   top1・κはほぼ完全一致（誤差0.04pt未満），answer_qualityは-1.94ptの乖離があったがノイズ床
   3SD=2.61pt内に収まった．Iter26の「アーキテクチャを変えてもルーティング判定は完全一致する」
   という知見が，fallback廃止という別レバーについても再確認された．

**次に振る単一レバーの選定（Y2 vs Y4）**:

docs/d0004 §5 のロードマップでは，Y1（fallback廃止，完了）の後はY2（`confidence_threshold`の
二重責務分離，Y3の前提）が自然な優先順位だが，Y4（分類器の較正，オフライン・低コスト，Y1と
並行可能）を先に行う選択肢もある．以下の判断基準で **Y4 を次イテレーション（Iter29）の単一
レバーとする**．

- **判断基準1（自律判断の可逆性）**: Y2 は `config.yaml` に `dispatch_candidate_threshold` を
  新設し，`aggregator.select_dispatch_targets()` の関数シグネチャを変更する，**設定ファイル
  形式・関数シグネチャの変更**である．config.yml 自身の note（139-141行）に「着手前にユーザー
  確認が必要」と明記されている．rc-reflector の自律判断権限は可逆な判断（レバー選定）に限られ，
  スキーマ変更を伴う着手そのものを今この場で自律的に開始することはできない．
- **判断基準2（コストと独立性）**: Y4（`CalibratedClassifierCV` による分類器較正）は既存の
  訓練データ（`data/classifier_train.jsonl`，1427件）に対するオフライン処理であり，
  d0004 が明記するとおり「Y1と並行して進めてよい」．ECEの較正前後比較は実機の1600問本走を
  必要とせず，スキーマ変更も不要．
- **判断基準3（Y2設計への波及）**: Y4 の結果（較正でECEがどれだけ下がるか）は，Y2で
  `dispatch_candidate_threshold` をどの値に設定すべきかの判断材料になりうる．較正が効けば
  2位confidenceの分布自体が変わり，Y2のデフォルト値設計が変わる可能性がある．Y4を先に行うことで
  Y2の設計（要ユーザー確認）をより具体的な材料とともに提示できる．
- **結論**: Iter29 の単一レバーは **classifier_calibration（Y4，d0003 X9）**とする．
  Y2 は Y4 完了後，スキーマ変更についてユーザー確認を得てから着手する．config.yml の
  levers 末尾に新規レバーとして追記した（backlog B49参照）．

**iteration_name（Iter29）**: 「分類器の較正（CalibratedClassifierCV）によるECE改善とルーティング
非退行の検証」

**要人間判断として残す論点（backlog B48 を維持，新規追加なし）**: fallback という設計思想自体を
撤廃するかどうかの論文上の位置付けは，今回の実験結果（recall低下・precision改善という表裏一体の
トレードオフの実測）だけでは決められない．これは次レバー選定とは独立した，対外的な研究結論に
関わる要人間判断事項であり，backlog に維持する．

---

### 調査 (Iter28)

**問い**: (1) fallback 廃止の実装は「`confidence_threshold` を 0.0 へ下げる」（config-only）と
「`node.py` の fallback 経路を明示的に無効化する」（コード変更）のどちらが単一レバー原則を保ちやすいか．
(2) `results/central_iter26/`（fallback 廃止相当）は実際どういう仕組みで生成されたデータか，
分散版で config だけを変えて本当に再現できる構成か．(3) confidence ベースの fallback/abstention は
文献上どう位置付けられているか（廃止判断の傍証はあるか）．

#### 分かったこと

**(1) 実装方針の比較 — config-only 案（`confidence_threshold: 0.0`）を推奨する**

コードを直接読んで確認した．ゲートは `aggregator.select_dispatch_targets()`
（`aggregator.py:28-40`）1 箇所のみで，

```python
eligible = [r for r in probe_responses if r.confidence >= confidence_threshold]
return sorted(eligible, key=lambda r: r.confidence, reverse=True)[:top_k]
```

呼び出し元は `node.py:run_ask_flow()`（216-217行，`run_experiment.py:87` もこの関数を再利用して
`dispatched_domains` を再計算しているので **1600 問バッチ実行の実データ経路と同一**）．`confidence`
は `predict_proba` の出力で常に `>= 0.0` なので，`confidence_threshold=0.0` にすると `eligible` は
毎回全 probe_responses になり，`top_k=1` なら必ず argmax の 1 件が返る．`node.py:219` の
`if not targets:` （fallback 発火条件）は，全ノードの probe 自体が失敗した真の異常系でしか
成立しなくなり，**confidence ベースの fallback だけが選択的に消える**．これは 1 行の config 変更で
完結し，`node.py`／`aggregator.py` のコード自体は 1 バイトも変える必要がない．

`http_server.py` 側で `NodeState.confidence_threshold`（`http_server.py:201`）を grep したところ，
格納するだけで他に参照箇所が無い（未使用フィールド）ことも確認した．つまり `confidence_threshold`
は実質的に「fallback 経路の唯一のスイッチ」であり，二重責務（fallback ゲート／dispatch 候補ゲート）は
`dispatch_top_k=1` に固定している限り実害が無い（top_k=1 では「1 位が閾値を超えるか」と
「候補が 1 件以上あるか」が同じ条件に潰れるため，Y2 の分離作業を待たずに Iter28 は成立する）．

対して「`node.py` の fallback 経路を明示的に無効化する」案（例: `if not targets:` 分岐を削除し
常に dispatch する）は，`run_ask_flow` の制御フロー自体を変更するコード変更であり，(a) `_fallback_answer`
を呼ぶ経路が実際に消えたことを別途テストで確認する必要がある，(b) 将来 probe が本当に全滅した
異常系（ネットワーク断等）でもフォールバックしなくなり，設計書が想定する「安全網」自体を壊す，
という 2 点で config-only 案より単一レバー原則から外れやすい．**推奨は config-only 案
（`confidence_threshold: 0.0`）**．

**(2) `central_iter26` の生成経緯（再現性の根拠）**

`results/central_iter26/config.yaml` を実際に読むと `confidence_threshold: 0.5` のままであり，
一見閾値を下げたようには見えない．`scripts/run_central_experiment.py` の該当コミット履歴とコード
コメント（237-260行）を確認したところ，**Iter26 初回実装は `confidence_threshold` の閾値チェック自体を
コードに書いていなかった**（常に argmax を dispatch），という経緯だった．つまり config 値ではなく
コード側の欠落によって「fallback 廃止相当」のデータが生成されていた．現行の分散版コード
（`node.py`/`aggregator.py`）には最初から閾値チェックが存在するため，同じ効果を得るには
`confidence_threshold=0.0` という config 変更が対応する形になる（両者は数学的に等価: 常に argmax を
選ぶ = 閾値 0.0 で argmax を選ぶ．`predict_proba` の値域が `[0,1]` である限り差は生じない）．
**Iter26/Iter26b の比較が示す効果は，分散版で `confidence_threshold=0.0` を設定すれば理論上そのまま
再現されるはずだが，「アーキテクチャが違えば実装のわずかな差異が結果に影響しないか」は Iter26 で
初めて経験した論点（B46）でもあるため，実測による確認自体に意味がある**．

**(3) 文献調査（補助）**: confidence ベースの abstention/reject-option 設計は文献上も広く使われる
一方，直近の研究はまさに「verbalized/self-report confidence は正答率と弱くしか相関しない」ことを
問題視している．
- Jiang et al./関連 (arXiv:2410.13284, "Learning to Route LLMs with Confidence Tokens", 2024/2025):
  self-report・logit ベースの信頼度は正答率との相関が弱いと明記した上で，routing/rejection の
  下流有用性に着目すべきと主張．expert-mesh の ECE=0.204（全ドメイン過信）という実測と整合する．
- MDPI 2025 ("An LLM-Based Multi-Path QA System with XGBoost Routing and Threshold-Based Refusal",
  mdpi.com/2079-9292/15/9/1845): 本研究と同型の「閾値で refuse するかを決める」設計を扱い，
  今後の課題として「閾値そのものではなく，較正・OOD検知で低確信と真に回答不能な入力を切り分けるべき」
  と述べている．Y4（分類器較正，CalibratedClassifierCV）の方向性を支持する外部裏付けになる．
- ACL 2025 uncertainlp workshop ("Confidence-Based Response Abstinence"): 「現実的な応用では
  masking rate 0% は理想に過ぎず，ある程度の許容が必要」と述べており，**fallback/abstention の
  完全撤廃が常に最適ではない**という留保も存在する．この点は backlog B48 の「論文上の位置付けは
  人間判断」という申し送りと整合する．
- Uncertainty-Aware Abstention with Provable Alignment Guarantees (arXiv:2607.04430,
  CIC=confidence-interval calibration): 閾値をヒューリスティックに決めるのではなく，較正セットで
  誤り率を統計的に制御する閾値選択を提案．Y2/Y4 で `confidence_threshold` を再設計する際の
  参考になりうる．

**総合**: 文献は「未較正の confidence で閾値ゲートすることの危うさ」を裏付けており，expert-mesh の
実測（fallback 発動 212 問中，正解率が argmax 30.7% → fallback 8.5% へ悪化）はその具体例と整合する．
一方で「fallback/abstention という設計思想自体を捨ててよいか」は文献でも一枚岩ではなく，
人間判断の対象として backlog に残す価値がある（既存の B48 の要レビュー項目のままでよい）．

#### rc-planner への申し送り

1. **単一レバー**: `confidence_threshold: 0.5 → 0.0`（config.yaml 1 行）．
   **同時に `dispatch_top_k: 2 → 1` へ戻すこと**（Iter27 の残骸．top_k=1 に固定しないと
   confidence_threshold の二重責務が発火し単一レバー原則が崩れる．d0004 §5 Y1 注記のとおり）．
   `aggregation_method` は `dispatch_top_k=1` では no-op になるため値自体は any でよいが，
   config.yml の申し送り（69-71行）どおり `max_confidence` へ戻して Iter27 の残骸を消しておくのが
   紛れがなく望ましい．
2. **到達条件（d0004 §4 対策A）**: `node.py:216` → `aggregator.py:39` が読む．
   `run_experiment.py:87` も同じ関数を再利用するため，1600 問バッチ実行で確実に発火する．
   到達を阻む分岐は存在しない（`http_server.py` の `NodeState.confidence_threshold` は未使用の
   格納のみで，routing_method 等による排他制御を受けない）．
3. **予備実行（対策B）**: 本走前に先頭 20 問程度で，`fallback_answer` が 1 件も生成されないこと
   （＝全行で `dispatched_domains` の長さが 1）を確認すること．もし発生していれば
   `confidence_threshold` が反映されていないデプロイ漏れ（Iter16/20/21/22/27 と同型の失敗）を疑う．
4. **成功条件の目安**: `results/central_iter26/` vs `central_iter26b/` の実測（d0004 §5 Y1 表）を
   分散版での期待値として使ってよい．top1 +2.94pt・κ+3.26pt・answer_quality +5.74pt（3SD=2.61pt
   の 2.2 倍）・mean_duration −323ms．Iter26 で「アーキテクチャを変えてもルーティングは完全一致」が
   実証済みなので，同じ大きさの差が出ることが期待値だが，**一致しない場合はそれ自体が新知見**
   （分散/中央のわずかな実装差が確率境界付近で結果に影響する可能性を示す）なので，一致しないことを
   理由に実験を無効と判定しないこと．
5. **人間判断が必要な論点（backlog に残す）**: fallback を完全撤廃するか，較正後に閾値だけ調整するか
   （Y2/Y4 との関係）は文献上も一枚岩ではない．今回の調査では新たな示唆は無く，B48 の既存の
   要レビュー項目をそのまま維持してよい．

---

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

