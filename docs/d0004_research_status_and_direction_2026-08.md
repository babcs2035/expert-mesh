# d0004: 研究の現況と今後の方針（Iter27 時点，2026-07-31）

**この文書の役割**: research-cycle の各フェーズ（rc-investigator / rc-planner / rc-implementer /
rc-experimenter / rc-analyst / rc-reflector）が，次の一手を決める前に最初に読む現況文書である．
`.claude/research/journal.md` はイテレーション単位の生記録であり，全体像は追えない．本書はそれを
Iter27 時点の到達点として集約し，次に何を，どの順で，なぜやるかを示す．

**先行文書との関係（重要）**:
- `docs/d0002_research_cycle_findings_2026-07.md`（Iter1〜22 の総括）と
  `docs/d0003_next_experiments_2026-07.md`（次の実験計画）は，いずれも **2026-07-29，Iter22 時点で
  執筆されたまま更新されていない**．その後 Iter23〜27 が実施され，d0003 が「未実施」としている
  F1〜F5・X1・X2・X4 の大半は既に完了している．**d0002・d0003 を単独の根拠として計画を立ててはならない．
  本書 §4 の棚卸し表で現況を確認すること．**
- 恒久的な知見（手法上の学び）と文献的根拠については d0002 §5・`docs/d0001_literature_survey_2026-07.md`
  が引き続き有効である．本書はそれらを再掲しない．
- 研究の目的・評価軸の定義は `docs/encounter_expert_mesh_design.md` が一次情報である．

---

## 1. 研究の中心命題と，現時点で言えるようになったこと

設計書 §4.5 が掲げる中心命題は次のとおりである．

> 有線 LAN 常時接続という中央集権方式に有利な条件下であっても，HTTP POST ベースの自律分散
> ルーティングは，中央集権ルータに対して大きなオーバーヘッドなく比較可能な精度を達成できる．
> すなわち分散アーキテクチャを採用すること自体のコストは小さい．

**Iter26 でこの命題に対する直接証拠が得られた**．これは d0002 §8-1 が「主張の核心の空白」と
していた箇所であり，本研究で最も重要な前進である．

| 問い（設計書 §7） | 現時点の回答 | 根拠 |
|---|---|---|
| ① 軽量な HTTP 問い合わせだけで「誰に聞くべきか」を自律決定できるか | **できる．ただし自己申告 confidence では不可能で，教師あり分類器が必須**．top1 0.5556（Random 0.101，Oracle 1.0），κ=0.5215 | Iter17（E6 採用），Iter15〜16（self_report の限界） |
| ② 中央集権方式に有利な条件下でも同等性能を許容コストで出せるか | **出せる．しかも分散版の方が速い**．ルーティング判定は 1600 問すべて一致（McNemar discordant=0, p=1.0），レイテンシは分散 3627ms 対 中央 4558ms で**中央版が 25.7% 遅い** | Iter26 |
| ③ 複合ドメインを含む実生活的質問の評価ベンチマークをどう設計するか | **未達**．複合設問 100 問を整備したが（Iter25），`dispatch_top_k` が構造的に機能せず（§3），set_recall は上限 0.5 に対し 0.165 に留まる | Iter25，Iter27 |

②の結果は一般的な直観（中央集権の方が効率的）と逆であり，実装上の理由がある．中央集権版は
6GB VRAM 制約下で 10 個の LoRA を 1 台に常駐させられないため，SSH 越しに各ノードの Ollama を
間借りする構成にせざるを得ない．その往復コストが，分散版の probe オーバーヘッド（137.5ms/問）を
上回る．**「VRAM 制約下では中央集権アーキテクチャが構造的に不利になる」という主張は，本研究
固有の実証的貢献として書ける．**ただし「1 台にモデルを常駐させた理想的な中央集権」との比較では
ないという限定を必ず付すこと（Iter26 の journal に明記済み）．

---

## 2. 確定した結果の一覧（Iter15 以降）

Iter14 以前は評価集合が 46 問しかなく統計的に判定不能だったため（backlog B26），一次情報としては
扱わない．

| Iter | レバー / 作業 | 判定 | 効果 |
|---|---|---|---|
| 15 | E1 評価集合 46→1520 問（JMMLU） | adopted | 統計基盤の確立．Wilson CI・McNemar・3 ベースラインを整備 |
| 16 | E3 `confidence_elicitation=top_k_with_probs` | 判定保留 | McNemar p=0.0783（有意差なし），ECE 悪化．後に E6 下では no-op と判明 |
| 17 | **E6 `routing_method=supervised_classifier`** | **adopted** | top1 0.2059→0.5651，κ 0.1067→0.5215．研究の主要成果 |
| 18 | **E10 `expert_specialization=domain_lora`** | **adopted** | answer_quality 0.2787→0.5013．ノード間に実能力差が生まれ，top1 が初めて下流に帰結を持つ指標になった |
| 19 | E8 `expert_model_size=qwen3.5-4b` | rejected | 1.85 倍遅化，回答品質も低下 |
| 20 | E3 再試行（1520 問） | 無効 | E6 下で no-op |
| 21・22 | E4 `self_consistency_semantic` | 無効 | 分岐順序の問題でコードパス未到達．デプロイ漏れも重なった |
| 23 | F1〜F3・F5 の確定 + X1 基準線再取得 | adopted | 主基準 4 項目が期待値と完全一致（差 < 0.0001）．測定系の健全性を確認 |
| 24 | X2 中央集権ルータ（初回） | rejected | 実装バグ（few-shot 欠落）で不成立 |
| 25 | X4 複合設問 20→100 問，評価集合 1600 問 | adopted | 新基準線 `results/20260730_145356/` |
| 26 | **X2 中央集権ルータ（再実験）** | **adopted** | §1 のとおり．命題②に対する直接証拠 |
| 27 | 集約方式 `aggregation_method` 比較 | **invalid（実験不成立）** | §3 のとおり完全な no-op |

**現在の最良既知構成**（`config.yaml`）: `routing_method=supervised_classifier`,
`confidence_signal_method=self_report`, `expert_model=expert-mesh-{domain}-lora`（10 ノード）,
`light_model=qwen3.5:4b-q4_K_M`, `domain_count=10`.
**基準線**: `results/20260730_145356/`（Iter25，1600 問）.

---

## 3. Iter27 はなぜ不成立だったか — `dispatch_top_k` は現行設定では数学的に発火しない

Iter27 は `dispatch_top_k=2` に設定したうえで `aggregation_method` を
`max_confidence` / `majority_vote` / `llm_judge` の 3 通りで 1600 問ずつ実行した（計 3 回，約 5 時間）．
結果は **3 方式とも Iter25 基準線とビット単位で同一**であった．

| 指標 | 基準線 | max_confidence | majority_vote | llm_judge |
|---|---|---|---|---|
| top1_accuracy | 0.555625 | 0.555625 | 0.555625 | 0.555625 |
| Cohen's κ | 0.5214815 | 0.5214815 | 0.5214815 | 0.5214815 |
| ECE | 0.2040206 | 0.2040206 | 0.2040206 | 0.2040206 |
| fallback_rate | 0.1325 | 0.1325 | 0.1325 | 0.1325 |
| McNemar（対基準線） | — | discordant=0, p=1.0 | discordant=0, p=1.0 | discordant=0, p=1.0 |
| **2 ノードへ dispatch した問題数** | 0 | **0 / 1600** | **0 / 1600** | **0 / 1600** |

### 機序

`aggregator.select_dispatch_targets()`（`aggregator.py:39`）は，まず
`confidence >= confidence_threshold` で候補を絞り，**そのうえで** top-k を取る．

```python
eligible = [r for r in probe_responses if r.confidence >= confidence_threshold]
return sorted(eligible, key=lambda r: r.confidence, reverse=True)[:top_k]
```

`routing_method=supervised_classifier` では，各ノードは 10 クラス LogisticRegression の
**自分のクラスの確率のみ**を返す．10 ノードの確率の総和は 1 であるから，2 ノードが同時に
`confidence >= 0.5` を満たすには p₁ + p₂ ≥ 1.0 が必要で，これは事実上起こり得ない．

実データがこれを裏付ける（1600 問，`results/20260730_224515/`）:

| 順位 | mean | median | p95 | p99 | **max** |
|---|---|---|---|---|---|
| 1 位 confidence | 0.7770 | 0.8343 | 0.9986 | 0.9999 | 1.0000 |
| 2 位 confidence | 0.1407 | 0.1081 | 0.3949 | 0.4580 | **0.4955** |

**2 位の confidence は最大でも 0.4955 で，閾値 0.5 に一度も到達していない．**
したがって `dispatch_top_k` を 2 以上にしても候補は常に 1 件以下であり，集約方式は
呼ばれる余地がない．これはデプロイの成否とは無関係に成立する結論である．

閾値を下げた場合に 2 ノード目が適格になる問題数（同データで逆算）:

| confidence_threshold | 1 ノード以上適格 | 2 ノード以上適格 |
|---|---|---|
| 0.5（現行） | 1388 / 1600 | **0 (0.0%)** |
| 0.4 | 1529 / 1600 | 75 (4.7%) |
| 0.3 | 1586 / 1600 | 230 (14.4%) |
| 0.25 | 1598 / 1600 | 365 (22.8%) |
| 0.2 | 1600 / 1600 | 509 (31.8%) |

### 帰結: `confidence_threshold` の二重責務

上表は同時に，**`confidence_threshold` を下げると 1 ノード目の適格数（＝ fallback しない問題数）も
一緒に動いてしまう**ことを示している．この 1 つのパラメータが

- (a) **fallback ゲート**: 1 位の confidence が閾値未満なら general の light_model へ退避する
- (b) **dispatch 候補ゲート**: 2 位以降を dispatch 対象に含めるか

という**独立した 2 つの役割を兼ねている**．したがって「集約方式の効果」も「fallback 方策の効果」も，
現行実装のままでは単一レバーとして分離できない．**この分離が，複合ドメイン評価（問い③）へ進む
ための前提条件である**（§5 Y2）．

なお複合設問は 100 問すべてが 2 ドメインであるため，`dispatch_top_k` が実効 1 である限り
`compound_domain_set_recall` の**構造的上限は 0.500**（実測 0.165）であり，2 になれば上限は 1.000 になる．

### 副産物: 回答品質のノイズ床が確定した（d0003 X6 の完了）

Iter27 の 3 実行はルーティングが完全に決定論的で同一だったため，意図せず
**「生成のランダム性のみが異なる 4 回の反復実行」**（Iter25 基準線を含む）になった．
これは d0003 X6 が「3 回反復して標準偏差を求める」として計画し，未実施のまま
n=2 の暫定値 1.3pt を使い続けていた測定そのものである．

`answer_quality_accuracy`（JMMLU 1500 問，抽出照合）:

| 実行 | 値 |
|---|---|
| Iter25 基準線 | 0.5087 |
| Iter27 max_confidence | 0.4960 |
| Iter27 majority_vote | 0.4913 |
| Iter27 llm_judge | 0.5080 |

- 平均 0.5010，**標準偏差 0.87pt**，範囲 1.73pt
- **2SD = ±1.74pt，3SD = ±2.61pt**
- 行単位では **359 / 1500（23.9%）の問題が，同一構成の反復間で正誤が反転した**

**判定基準への反映**: 軸②③（`answer_quality_accuracy`・`end_to_end_accuracy`）の変化は，
**3SD＝2.6pt を超えない限り有意と判定してはならない**．`.claude/research/config.yml` の
`success_criteria` に反映済み．

この基準を過去の判定に遡って当てると，Iter26 の回答品質 −1.53pt・End-to-End −1.50pt は
いずれもノイズと判定され，journal の暫定的な読み（「ノイズ床に近い」）が追認される．
一方で E10 の +22.3pt は 3SD の 8 倍以上であり，堅牢な結論として維持される．

---

## 4. 反復している失敗モード —「設定したレバーがコードパスに届かない」

Iter16・20（E3），Iter21・22（E4），backlog B35（E7），そして今回の Iter27 は，
**いずれも同じ型の失敗**である．config を正しく書き換え，正しくデプロイし，実験も完走したが，
その設定を読むコードに実行が到達しないため，結果が基準線と完全一致した．
これまでに **6 イテレーション，のべ 10 時間以上の実機実行**がこの型で失われている．

原因は 2 種類ある．

1. **排他的な if 連鎖**（d0003 §1 制約 1）: `http_server.py:_estimate_probe_confidence()` は
   `confidence_signal_method` 系 → `routing_method` 系 → `confidence_elicitation` の順で先勝ち
   `return` する．そのため `supervised_classifier` の下では `confidence_elicitation` と
   `embedding_postprocess` が読まれない．
2. **前段のゲートで候補が消える**（今回）: `aggregation_method` は候補が 2 件以上ある場合にのみ
   意味を持つが，その候補数を決めるのは別のパラメータ（`confidence_threshold`）である．

### 恒久対策（次イテレーション以降で必ず実施すること）

**対策 A（rc-planner の責務）**: レバーを選んだら，計画フェーズで必ず
**「そのレバーを読むコード行を特定し，そこへ到達する条件を journal の計画節に明記する」**．
到達条件が現行構成で満たされないなら，そのレバーは実験対象にしない．
今回であれば「`aggregation_method` が読まれるのは `select_best_dispatch_response*` が
2 件以上の `DispatchResponse` を受け取ったときだけ．その条件は `select_dispatch_targets` が
2 件返すことで，それには 2 位の confidence ≥ `confidence_threshold` が必要」という 1 段落を
書いた時点で，実験前に不成立が判明した．

**対策 B（rc-implementer / rc-experimenter の責務）**: 実験を本走させる前に，
**データセットの先頭 20 問程度で予備実行し，「レバーが効いた証拠となるフィールド」を直接確認する**．
今回なら `dispatched_domains` の長さが 2 になる行が 1 件でもあるか，である．
Iter21 では `semantic_entropy` フィールドが 0 件だったことが事後に無効の決め手になっており，
これを事前に見ていれば 1520 問の実行は不要だった．`tools/smoke_check.py` は配布物のハッシュ一致
までは確認するが，**「レバーが意味的に発火したか」は確認していない**．ここを拡張する余地がある．

**対策 C（rc-analyst の責務）**: 分析の最初に **基準線との完全一致を疑う**．
主要指標が小数点 6 桁まで一致した場合，それは「効果がなかった」ではなく
**「実験が成立していない」**と解釈することを既定とする．
`routing_method` 系は決定論的なので一致自体は正常だが，そのとき軸②③まで含めて
McNemar discordant=0 なら，レバーは発火していない．

---

## 5. 次にやるべきこと（優先順位つき）

d0003 の X 番号との対応を残しつつ，Iter27 時点で振り直した優先順位を Y 番号で示す．

### Y1（最優先・低コスト）: fallback 方策の廃止／見直し — d0003 X5

**すでに答えが実測されている．** `results/central_iter26/`（閾値なし純 argmax ＝ fallback 廃止相当）と
`results/central_iter26b/`（現行の閾値 0.5 + general への fallback）は，アーキテクチャ・分類器・
データセットが同一で **fallback 方策だけが異なる**．両者の対比は fallback を単一レバーとした比較に
なっている（この 2 つが揃ったのは Iter26 で方策の食い違いに気付いた際の副産物である．backlog B46）．

| 指標 | fallback 廃止 | fallback あり（現行） | 差 |
|---|---|---|---|
| top1_accuracy | **0.5850** | 0.5556 | **+2.94pt** |
| single_domain_top1_accuracy | 0.5987 | 0.5693 | +2.94pt |
| compound_domain_top1_accuracy | 0.3800 | 0.3500 | +3.00pt |
| Cohen's κ | **0.5541** | 0.5215 | **+3.26pt** |
| answer_quality（JMMLU1500） | **0.5507** | 0.4933 | **+5.74pt**（3SD=2.6pt の 2.2 倍） |
| mean_duration_ms | 4234.8 | 4558.2 | −323ms（速い） |

McNemar（現行 vs 廃止）: discordant 77 件（廃止のみ正解 62，現行のみ正解 15），**p = 1.59e-7**．

さらに **fallback が発動した 212 問だけ**を取り出すと:

- fallback して general へ送った場合のルーティング正解: **18 / 212（8.5%）**
- fallback せず argmax のドメインへ送った場合: **65 / 212（30.7%）**

つまり**現行の fallback は，分類器が迷っている問題をわざわざ正解率 8.5% の選択肢へ振り替えている**．
d0002 §8-3 の「fallback は Random（10.1%）を下回る」という指摘が定量的に裏付けられ，かつ
廃止した場合の効果まで判明した．精度・品質・レイテンシのすべてで廃止が優る．

**やること**: `config.yaml` の `confidence_threshold` を 0.0 に下げる（＝ fallback 廃止）か，
`node.py:run_ask_flow` の fallback 経路を明示的に無効化して，**分散版で** 1600 問を 1 回実行し
確認する．Iter26 が「アーキテクチャを変えてもルーティングは完全一致」を示しているため，
中央版で見えた差は分散版でもそのまま再現すると予測される．これが外れた場合はその不一致自体が
新たな知見になる．コストは約 90 分，コード変更は最小．

**注意**: `confidence_threshold` を 0.0 にすると §3 のとおり dispatch 候補ゲートも同時に緩む．
`dispatch_top_k=1` のままにしておけば top-k 側は動かないので単一レバーは保てる．
**Y1 の実験では `dispatch_top_k` を 1 に戻すこと**（現在 2 のまま）．

### Y2（前提整備）: `confidence_threshold` の二重責務を分離する — Y3 の前提

§3 の帰結への対処．`config.yaml` に `dispatch_candidate_threshold` を新設し，

- 1 位の採否（fallback するか）は従来どおり `confidence_threshold`
- 2 位以降を dispatch 候補に含めるかは `dispatch_candidate_threshold`（既定値は
  `confidence_threshold` と同値にして後方互換を保つ）

とする．`aggregator.select_dispatch_targets()` のシグネチャ変更と，`node.py` の呼び出し側の
修正で済む見込み．これにより **fallback 率を固定したまま `dispatch_top_k` だけを振れる**ようになり，
単一レバー原則を満たした状態で Y3 に進める．

### Y3: 複合ドメイン評価の成立 — d0003 X4 の本体，Iter27 のやり直し

Y2 の完了後，`dispatch_candidate_threshold` を 0.25 前後（§3 の表より 2 ノード適格が約 23%）に設定し，
`dispatch_top_k=2` で集約方式 3 種を比較する．主指標は `compound_domain_set_recall`
（現状 0.165，構造的上限 0.500 → top_k=2 で上限 1.000）とし，副指標として複合設問の
`answer_quality`（LLM-as-judge，n=100）を見る．
**単一ドメイン 1500 問の top1 は非退行の確認にのみ使う**（複合設問は 100/1600 = 6.25% しかないため，
全体 top1 は複合側の改善をほとんど反映しない．Iter25 でこの合成比率の問題は既に確認済み）．

これは研究の問い③に直接答える唯一の実験であり，Y1 に次ぐ価値がある．

### Y4（低コスト・オフライン）: 分類器の較正 — d0003 X9

ECE 0.2040，全ドメインで mean_confidence > accuracy という過信が残る（d0002 §8-5）．
`CalibratedClassifierCV`（Platt / isotonic）を既存の訓練データ（`data/classifier_train.jsonl`，
1427 件）に適用し，**オフラインで**較正前後の ECE を比較できる．実機実行なしで判断できるため
コストが極めて低い．較正が効けば，`confidence_threshold` に依存する Y1・Y2・Y3 のすべてに
波及する．**Y1 と並行して進めてよい．**

### Y5: education / legal のデータ不均衡是正 — d0003 X8

legal は JMMLU に対応タスクがなく訓練 77 件，education は心理学・社会学で代理している
（d0002 §8-4）．両ドメインの recall が構造的に低い．訓練データの拡充が必要でコストは中程度．

### 着手しない項目（理由つき）

- **X3（E4 semantic entropy の再設計）・X7（E5 p_true）**: いずれも現行の排他的 if 連鎖の下では
  E6（supervised_classifier）を無効化してしまう（d0003 §1 制約 1）．同じ目的（confidence の較正）は
  Y4 が桁違いに安く達成しうる．**Y4 の結果を見てから判断する．**
- **E3 `linguistic`・E7 `whitening`**: E6 下で no-op と確定済み．実施不要．
- **E10 `offtheshelf_specialized`**: 日本語の法律特化オープン生成モデルが見つからず実施不能．
- **`hidden_state`**: Ollama では取得不可．vLLM / SGLang への移行が前提となり本フェーズのスコープ外．
- **無線アドホック化**: 設計書 Phase 3．本フェーズのスコープ外．

---

## 6. 運用上の申し送り

### 6-1. Iter23 の heartbeat スクリプトが 38 時間放置されていた

`/tmp/iter23_heartbeat.sh` が 2026-07-30 01:42 から 2026-07-31 15:43 まで動き続け，
`.claude/research/state.json` の `updated_at` を 120 秒ごとに上書きしていた．
停止条件のマーカーファイル `/tmp/iter23_start.done` が生成されなかったためである．

**影響**: `updated_at` は「research-cycle が生きている」ことの唯一の指標であり，watchdog の
ハング検知はこれを見ている．偽の heartbeat により，Iter27 が実験完了後（07-31 03:44）に
12 時間停止していたことが検知されなかった．本セッションで当該プロセスを停止済み．

**申し送り**: 実験の監視に使い捨てスクリプトを `/tmp` へ置く場合は，(a) 停止条件を
タイムアウト付きにする，(b) イテレーション完了時に必ず後始末する，のいずれかを徹底すること．
`state.json` の heartbeat は research-cycle オーケストレータ自身だけが更新すべきである．

### 6-2. Iter27 の 3 実行に provenance が残っていない

`results/20260730_224515/` ほか 2 ディレクトリには `config.yaml`・`git_head.txt`・`metrics.json` が
無く，`results_topk2_*.jsonl` という非標準のファイル名だけが置かれている．
F5（再現性マニフェスト，Iter23 で `run_experiment.py:_record_experiment_provenance()` として実装）は
`mise run start` の標準経路を通った場合にのみ機能するが，Iter27 は独自の呼び出しで実行されたため
provenance が失われた．どの構成で走ったかはコミット履歴（`cde9247` → `7f72b1a` → `32af2e0`）から
辿るしかない．

**申し送り**: 標準経路を外れて実験する場合は，`_record_experiment_provenance()` 相当を明示的に
呼ぶこと．結果ディレクトリのファイル名は `results.jsonl` に統一する（`metrics.py` と
`mise run analyze` が既定でこの名前を探すため）．

### 6-3. journal に記録されたコマンドが実在しないことがある

Iter24 の計画節にある `uv run python metrics.py --results A --compare B` は，`metrics.py` に
`--compare` 引数が存在しないため実行できない（実装されているのは関数 `compute_mcnemar_test()` で，
CLI からは呼べない）．McNemar 対比較を CLI から使えるようにするか，計画に書くコマンドを
実在するものに限るか，いずれかの対処が要る．

### 6-4. 人間判断を要する未解決事項

- **D6（d0003 §6 より）**: `.claude/research/config.yml` の `git.push: true` が，グローバル
  CLAUDE.md の規約と衝突しないかの確認．未解決のまま．
- **Y2 のコード変更**: `aggregator.select_dispatch_targets()` のシグネチャと `config.yaml` の
  スキーマを変更する．CLAUDE.md は「既存の public API・設定ファイル形式の変更は事前にユーザーへ
  確認する」と定めているため，**着手前に確認が必要**．
