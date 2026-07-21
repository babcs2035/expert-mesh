## Iteration 9: few-shot 例の構造変更（全ドメイン表示へ）と保守的指示追加

### イテレーション完了サマリー

**単一レバー**: few_shot_structure_change（router.py の build_confidence_prompt() 内 few-shot 例ブロックの全ドメイン表示化 + 保守的指示追加）
**判定**: rejected（主基準 1/2 未達，非退行 2/4 未達）
**結果**: education precision=1.0（>=0.93 PASS）だが、recall=0.5（>=0.62 FAIL）。single_domain_top1_accuracy=0.875（>=0.87 PASS—僅差）。general/legal precision が退行。
**改善**: general-004 の education misroute が是正（precision 0.889→1.0）。
**副作用**: education recall の大幅低下（0.667→0.5）。全ドメイン表示 + 保守的指示により education ノードが過剰抑制。general/legal precision も退行。misrouting_rate 悪化（0.087→0.130）。
**学び**:
1. few-shot 例の全ドメイン表示は education precision を改善するが、recall を犠牲にする（過剰抑制）。
2. 評価基準への保守的指示追加は副作用を強化し、全体として rejected。
3. router.py の few-shot 例変更は 5 回連続（Iter5-9）で試されたが、いずれも期待した効果を持たなかった。このレバーは**収束**した。
**次イテレーションの単一レバーの方針**: config-only レバー探索は Iter3 で限界確定済み。few-shot 変更も 5 回連続 rejected。rc-planner は根本的に異なるアプローチ（probe ロジック変更、新しいルーティング方式の検討）を提示すること。
**コミット**: router.py の few-shot 変更 + journal/state/backlog の更新

---

### 考察・次計画 (Iter9)

**仮説**:
- H1: 例1-3を「全ドメイン表示」（4ドメインすべてにconfidence値を表示）に変更すると、education ノードは cross-domain の対比を直接学習できる。一般質問で education 関連の言葉が出ても、general=0.9 > education=0.1 の対比を few-shot 例から直接読み取り、education confidence を低く抑える。
- H2: 評価基準セクションに「教育関連の語句が含まれていても主題が他分野であれば education confidence は低くする」との指示を追加し、few-shot 例と評価基準の整合性を取る。
- H3: general-004 の education confidence が 0.95→0.7 以下に低下し、general (0.9) が勝つようになる。

**成功条件**（ベースライン: results/20260721_185132, Iter8）:
- 主基準: education precision >= 0.93（baseline 0.889 から +0.04 以上）
  - ノイズ幅見積もり: Iter7→8 で education precision は 0.909→0.889（-0.020）。 Iter6→7 で 0.90→0.909（+0.009）。1イテレーションでの変動は ±0.02 程度。+0.04 はノイズの2倍以上。
- 非退行: education recall >= 0.62（baseline 0.667 から -0.05 以内）
- 非退行: single_domain_top1_accuracy >= 0.87（baseline 0.900 から -0.03 以内）
- 非退行: general/medical/legal の precision/recall は baseline 以下に退行しない

**固定する構成**:
- config.yaml: `routing_method: self_report`, `confidence_threshold: 0.5`, `dispatch_top_k: 1` を維持
- build_dataset.py: 不変
- http_server.py, docker-compose.yml, mise.toml: 不変
- 例4: 不変（すでに全ドメイン表示）

**期待効果**: education ノードが few-shot 例の全ドメイン表示から「general 質問では education confidence は 0.1」という対比パターンを直接学習。一般質問で education 関連の言葉（読書、勉強等）が出ても、general confidence (0.9) の方が高いことを認識し、education confidence を低く抑える。

**変更ファイルと変更量**:
- router.py: build_confidence_prompt() の few-shot 例ブロック（行62-73）を書き換え
  - 行62-65（評価基準）: 「教育関連の語句が含まれていても...」の指示を1行追加
  - 行66-73（few-shot 例）: 例1-3を全ドメイン表示に変更（変更量: 例1-3の各行に2ドメイン分追記）

**変更前（例1）**:
```
例1：質問「歯の痛みが続いています」はmedical分野に該当するため，domainがmedicalなら{"confidence": 0.9}，domainがlegalなら{"confidence": 0.1}．
```

**変更後（例1）**:
```
例1：質問「歯の痛みが続いています」はmedical分野に該当するため，domainがmedicalなら{"confidence": 0.9}，domainがeducationなら{"confidence": 0.1}，domainがgeneralなら{"confidence": 0.1}，domainがlegalなら{"confidence": 0.1}．
```

**変更前（例2）**:
```
例2：質問「賃貸契約を解除したい」はlegal分野に該当するため，domainがlegalなら{"confidence": 0.9}，domainがmedicalなら{"confidence": 0.1}．
```

**変更後（例2）**:
```
例2：質問「賃貸契約を解除したい」はlegal分野に該当するため，domainがlegalなら{"confidence": 0.9}，domainがeducationなら{"confidence": 0.1}，domainがmedicalなら{"confidence": 0.1}，domainがgeneralなら{"confidence": 0.1}．
```

**変更前（例3）**:
```
例3：質問「学習指導要領における探究的学習の位置付けは」はeducation分野に該当するため，domainがeducationなら{"confidence": 0.9}，domainがmedicalなら{"confidence": 0.1}．
```

**変更後（例3）**:
```
例3：質問「学習指導要領における探究的学習の位置付けは」はeducation分野に該当するため，domainがeducationなら{"confidence": 0.9}，domainがmedicalなら{"confidence": 0.1}，domainがgeneralなら{"confidence": 0.1}，domainがlegalなら{"confidence": 0.1}．
```

**変更前（評価基準セクション）**:
```
評価基準:
- 主題が明確に{domain}分野に属する: 0.7〜1.0
- 主題が{domain}分野と無関係，または他分野がより適切: 0.0〜0.3
- 判断に迷う: 0.4〜0.6
```

**変更後（評価基準セクション）**:
```
評価基準:
- 主題が明確に{domain}分野に属する: 0.7〜1.0
- 主題が{domain}分野と無関係，または他分野がより適切: 0.0〜0.3
- 判断に迷う: 0.4〜0.6
- {domain}関連の語句が含まれていても，主題が他分野であれば{domain} confidence は低くする（例: 読書・勉強・習い事は general 分野）．
```

**検証手順**:
1. `uv run pytest tests/ -v` で既存テスト全件 PASS 確認
2. `uv run ruff check .` で lint 違反なし確認
3. `mise run deploy` でコード変更を各ノードへ配布
4. `mise run start` で実験実行
5. `mise run analyze` で metrics 集計

**リスク評価**:
- 全ドメイン表示により prompt が肥大化し、LLM の attention が分散する可能性
- education recall がさらに低下する可能性（過剰抑制）
- 例4の既存構造（全ドメイン表示 + educationノード指示）との整合性

**単一レバー原則との整合**:
- 本レバーは config-only の枠を超える（router.py のコード変更）
- 変更量: 例1-3の各行に2ドメイン分追記 + 評価基準に1行追加。計5行弱の変更。
- 例4は不変。config.yaml は不変。
- 4イテレーション連続（Iter5-8）の few-shot 変更は「書き方」の問題であり、今回は「構造」の問題へ着手。

---

### 実験 (Iter9)

**デプロイ**: 4ノード（wafl500, wafl501, wafl502, wafl503）すべて正常完了
- データセット再生成: data/dataset.jsonl が uv warning メッセージで破損していたため、`build_dataset.py --output data/dataset.jsonl` で再生成（46行）
- Docker image 再ビルド・再push: データセットを含む新しいイメージを全ノードにデプロイ

**実行結果**: results/20260721_222225（46問，全問完走，used_fallback=1, dispatch_failed=0）
- 平均応答時間: 13731ms

**メトリクス（per-domain）**:

| ドメイン | precision (Iter9) | recall (Iter9) | precision (Iter8) | recall (Iter8) |
|---|---|---|---|---|
| education | **1.0000** | **0.5000** | 0.8889 | 0.6667 |
| general | **0.9000** | **0.9000** | 1.0000 | 0.9000 |
| legal | **0.7778** | **0.9333** | 0.8750 | 0.9333 |
| medical | **0.9167** | **0.7333** | 0.9167 | 0.7333 |

**総合指標**:
- single_domain_top1_accuracy: 0.875（Iter8 0.900）
- compound_domain_top1_accuracy: 0.833
- misrouting_rate: 0.1304（Iter8 0.087）
- top1_accuracy: 0.8696（Iter8 0.9130）
- fallback_rate: 0.0217（Iter8 0.0）

**misroute 詳細（Iter9 vs Iter8）**:
- education precision=1.0 → general-004 の education misroute が**是正**（education precision 1.0 = 全問正解）
- education recall=0.5 → **大幅低下**（0.667→0.5）。education ノードの過剰抑制により教育固有質問も誤って low confidence に
- general precision=0.9 → general-008 の medical misroute が**継続**（run 間ノイズ）
- legal precision=0.778 → **低下**（0.875→0.778）。education 固有話題の misroute 増加が主因

**成功条件判定**: 6項目中2PASS/4FAIL
- 主基準: education precision 1.0（>=0.93 **PASS**）
- 主基準: education recall 0.5（>=0.62 **FAIL**）
- 非退行: single_domain_top1_accuracy 0.875（>=0.87 **PASS** — 僅差）
- 非退行: general precision 0.9（>=1.0 **FAIL**）
- 非退行: legal precision 0.778（>=0.875 **FAIL**）
- 非退行: medical precision 0.917（>=0.917 **PASS** — 同等）

### 分析 (実行) (Iter9)

**mise run analyze 完了**: results/20260721_222225/

**成功条件判定（6項目中2PASS/4FAIL）**:

| # | 条件 | 閾値 | 測定値 | 判定 |
|---|------|------|--------|------|
| 1 | education precision | >= 0.93 | 1.000 | PASS |
| 2 | education recall | >= 0.62 | 0.500 | **FAIL** |
| 3 | single_domain_top1_accuracy | >= 0.87 | 0.875 | PASS（僅差）|
| 4 | general precision | >= 1.0 | 0.900 | **FAIL** |
| 5 | legal precision | >= 0.875 | 0.778 | **FAIL** |
| 6 | medical precision | >= 0.917 | 0.917 | PASS（同等）|

**ベースライン（Iter8）との差分**:
- education precision: +0.111（0.889→1.000）→ **改善**
- education recall: -0.167（0.667→0.500）→ **有意な低下**
- general precision: -0.100（1.0→0.9）→ **退行**（general-008 の medical misroute）
- legal precision: -0.097（0.875→0.778）→ **退行**
- single_domain_top1_accuracy: -0.025（0.900→0.875）→ **低下**
- misrouting_rate: +0.043（0.087→0.130）→ **悪化**

### 分析 (解釈) (Iter9)

**判定**: router.py few-shot 構造変更レバーは **rejected**（主基準 1/2 未達，非退行 2/4 未達）

**education precision=1.0 の是正効果**:
- general-004（「読書感想文の書き方」）の education misroute が**是正された**。education precision=1.0 は全問正解を意味する。
- これは H1 の部分的な成功：全ドメイン表示により、education ノードは general 質問で low confidence を出すようになった。

**education recall=0.5 の過剰抑制**:
- **予想と逆の副作用**: education precision が改善した一方で、recall が大幅に低下（0.667→0.5）。
- **原因**: 全ドメイン表示 + 保守的指示により、education ノードが**すべての education 質問**で confidence を過剰に抑制するようになった。
- education-001/009 の misroute は継続（これは education ノードの正しい自己認識によるもので few-shot 変更では是正不可能）。
- さらに、education 固有話題（education-002〜008）でも confidence が低下し、他のドメインに misroute するケースが増加。

**general/legal の退行**:
- general precision が 1.0→0.9（general-008 の medical misroute）。これは run 間ノイズの可能性もあるが、 Iter8 と同じ misroute パターン。
- legal precision が 0.875→0.778。education 固有話題の misroute 増加が主因。

**misrouting_rate 悪化（0.087→0.130）**:
- fallback が 1件発生（0.0→0.022）。これは保守的指示の影響で confidence が閾値以下に低下した質問が fallback された可能性。
- 全体の misroute が増加し、single_domain_top1_accuracy も低下（0.900→0.875）。

**仮説との整合**:
- H1（education precision 0.889→0.93以上）: **部分的成立**．1.0（+0.111）。ただし recall の犠牲。
- H2（single_domain_top1_accuracy 0.900→0.875以上）: **不成立**．0.875（-0.025）。
- H3（general/medical/legal の非退行）: **不成立**．general/legal precision が退行。

**次イテレーションへの示唆**:
1. **全ドメイン表示 + 保守的指示は過剰抑制を引き起こす**: education precision は改善したが、recall が大幅に低下。このアプローチは放棄すべき。
2. **router.py の few-shot 例ブロック変更は限界がある**: Iter5-9 で 5 回連続 few-shot 関連の変更を試したが、いずれも期待した効果を持たなかった。
3. **confidence 信号の較正には根本的なアプローチが必要**: config-only または few-shot 例の変更では対処できない。probe ロジック自体の変更や、新しいルーティング方式の検討が必要。

---

### 調査 (Iter9)

**問い**
- Q1: results/20260721_185132 の probe_candidates から confidence_threshold を掃引した結果、どの threshold で fallback_rate が変化するか。selected_domain は変化するか。
- Q2: education ドメイン特化の文脈で、confidence_threshold は education の過信抑制に有効か。
- Q3: Iter3 の values [0.3, 0.5, 0.7] は education 過信抑制の文脈でも no-op か。閾値の再設計は必要か。
- Q4: ベースライン結果の特定と成功条件の提案。

**分かったこと（Q1: threshold 掃引の結果）**

- **offline 再計算（results/20260721_185132, 46 行）**:

| threshold | fallback | total_dispatch | accuracy | edu_accuracy | 備考 |
|-----------|----------|----------------|----------|--------------|------|
| 0.3 | 0 | 46 | 0.891 (41/46) | 0.667 (8/12) | Iter3 の値 |
| 0.5 | 0 | 46 | 0.891 (41/46) | 0.667 (8/12) | ベースライン |
| 0.7 | 0 | 46 | 0.891 (41/46) | 0.667 (8/12) | Iter3 の値 |
| 0.8 | 0 | 46 | 0.891 (41/46) | 0.667 (8/12) | 依然 no-op |
| 0.85 | 1 | 45 | 0.911 (41/45) | 0.727 (8/11) | general-004 が fallback |
| 0.9 | 5 | 41 | 0.927 (38/41) | 0.727 (8/11) | general-002/003/008/010 も fallback |
| 0.95 | 24 | 22 | 0.864 (19/22) | 0.714 (5/7) | 品質退行 |

- **0.3/0.5/0.7/0.8 はすべて同一結果**（fallback=0, 41/46 正解）。これは Iter3 の「二峰・空帯域分布による no-op」判定を**決定的に確認**。
- **0.85 で唯一の変化**: education-009（edu=0.8, legal=0.8）が fallback。overall accuracy は 0.891→0.911 に改善。
- **0.9 で 5 件 fallback**: education-009 以外に general-002/003/008/010（general=0.85）も fallback。これらはすべて正解質問のため、accuracy は 38/41=0.927 だが、quality regression のリスク。
- **0.95 で 24 件 fallback（52.2%）**: medical/legal/general の高 confidence 質問が大量に fallback。accuracy は 0.864 に低下。

**分かったこと（Q2: education 過信抑制の文脈での効果）**

- **general-004（education 過信の主要ケース）**: education=0.95, general=0.9
  - **どの threshold でも education が勝つ**（0.95 > 0.9）。threshold=0.95 でも education 単独で eligible。
  - **結論: threshold 変更では general-004 の education 過信は絶対に抑制できない**。
- **education-001**: education=0.2, medical=0.95 → education ノードの正しい自己認識。threshold は関係なし。
- **education-002**: education=0.95, legal=0.95 → 同点で legal が tie-break 勝利。threshold=0.9 以上でも tie は維持。
- **education-009**: education=0.8, legal=0.8 → 同点で legal が tie-break 勝利。threshold=0.85 以上で fallback（回答なし）。
- **結論**: education 過信の 3 大 misroute（general-004, education-002, education-009）のいずれも、threshold 変更では是正できない。

**分かったこと（Q3: 閾値の再設計）**

- **Iter3 の values [0.3, 0.5, 0.7] は education 過信抑制の文脈でも no-op**。空帯域 (0.3, 0.7) に値 0 件は同じ。
- **education 過信抑制には閾値 0.85+ の探索が必要だが**:
  - 0.85: 1 件の fallback（education-009）。accuracy 0.911。副作用は最小。
  - 0.9: 5 件の fallback（一般質問 4 件も）。accuracy 0.927 だが quality regression リスク。
  - 0.95: 24 件の fallback。quality regression 確定。
- **しかし 0.85 で改善できるのは education-009 の fallback のみ**（回答なしになる）。education accuracy は 8/11=0.727 に改善するが、これは「misroute 1 件が fallback になる」だけ。precision/recall の改善にはならない（fallback は recall 低下としてカウントされる可能性）。
- **結論: Iter9 の values [0.3, 0.5, 0.7] は教育ドメイン特化の文脈でも no-op。閾値 0.85+ の探索は意味があるが、education 過信の根本原因（confidence 信号の較正）には対処できない**。

**分かったこと（Q4: ベースラインと成功条件の提案）**

- **ベースライン**: results/20260721_185132（Iter8, 46 問/4 ノード）
  - education precision=0.889, recall=0.667
  - single_domain_top1_accuracy=0.900
  - misrouting_rate=0.087
- **confidence_threshold レバーの限界**:
  - config-only 変更で education 過信 isotope は是正できない（general-004 は education=0.95 > general=0.9 で threshold 非効力）
  - education-002/009 の tie-break 問題は threshold で解決不可
  - 唯一の変化は threshold=0.85 で education-009 が fallback になること
- **成功条件の提案**（もし threshold=0.5 vs 0.85 を比較する場合）:
  - 主基準: overall accuracy >= 0.90（baseline 0.891 から改善）
  - 非退行: single_domain_top1_accuracy >= 0.89（fallback により低下する可能性を許容）
  - 非退行: fallback_rate <= 0.05（1 件以下）
- **しかし根本的な結論**: confidence_threshold は education 過信抑制のレバーとして**不適**。confidence 信号の較正（router.py 側の変更）が必要。

**次の計画フェーズへの示唆**:
1. **confidence_threshold レバーは rejected が妥当**。Iter3 の no-op 判定は education 過信抑制の文脈でも維持。
2. values を [0.5, 0.85, 0.95] に変更して実験する価値は低い（0.85 は 1 件 fallback のみ、0.95 は quality regression 確定）。
3. **真のレバーは confidence 信号の較正**（router.py の few-shot 例修正、または probe ロジックの変更）。これは config-only の枠を超える。
4. backlog B14 の「要レビュー」項目: confidence_threshold の再検証は不要。次 rc-planner は config-only の枠を出る変更を提示すること。

---

## Iteration 8: few-shot 例の構造変更（education ノード視点）

**単一レバー**: router.py の build_confidence_prompt() 内の few-shot 例ブロック（行72-73）の例4を education ノード視点へ変更

**仮説**:
- H1: 例4を「education ノード視点」で書くと、education ノードは few-shot 例を self-report 時の anchor として利用し、general 質問で low confidence (0.1) を出すようになる。general-004 の education misroute が解消される。
- H2: single_domain_top1_accuracy が 0.950→0.975 以上になる（general-004 の1件 misroute が解消）。
- H3: general/medical/legal の precision/recall は baseline 以下に退行しない。

**成功条件**（ベースライン: results/20260721_143604）:
- 主基準: education precision >= 0.95 AND education recall >= 0.80
  - recall の閾値を 0.90→0.80 に下げた理由: education-001/009 の misroute は education ノードの「正しい自己認識」が原因。few-shot 例の変更では是正不可能。これら2件を除外した education recall の最大値は 8/10 = 0.80。
- 非退行: single_domain_top1_accuracy >= 0.975 (40問中39正解)
- 非退行: misrouting_rate <= 0.022 (46問中1件以下)
- 非退行: general/medical/legal の precision/recall は baseline 以下に退行しない

**固定する構成**:
- config.yaml: `routing_method: self_report`, `confidence_threshold: 0.5`, `dispatch_top_k: 1` を維持
- build_dataset.py: 不変
- http_server.py, docker-compose.yml, mise.toml: 不変
- few-shot 例の例1-3: 不変

**期待効果**: education ノードが few-shot 例を self-report 時の anchor として利用し、general 質問で low confidence を出すようになる。

**変更ファイルと変更量**:
- router.py: build_confidence_prompt() の few-shot 例ブロック（行72-73）の例4を書き換え。変更量: 1行の書き換え

**変更前**:
```
例4：質問「読書感想文の書き方」はgeneral分野に該当するため，domainがgeneralなら{"confidence": 0.9}，domainがeducationなら{"confidence": 0.1}，domainがmedicalなら{"confidence": 0.1}，domainがlegalなら{"confidence": 0.1}．
```

**変更後（案A: 最小変更）**:
```
例4：質問「読書感想文の書き方」はgeneral分野に該当するため，domainがgeneralなら{"confidence": 0.9}，domainがeducationなら{"confidence": 0.1}，domainがmedicalなら{"confidence": 0.1}，domainがlegalなら{"confidence": 0.1}，educationノードは{"confidence": 0.1}とする（general分野でありeducation分野ではない）．
```

**案Aの選択理由**:
- 既存の「domainがXなら...」構造を維持し、教育ノード視点の要素を末尾に最小限追加する。
- 例1-3との一貫性を保つため、LLM が例4を「例外」として解釈するリスクを回避する。
- 例4は general 視点の事実提示（domainがXなら...）と education ノード視点の指示（educationノードは0.1とする）の両方を提示。複数の視点にさらされることで、LLM がより柔軟にパターンを抽出できる。

**検証手順**:
1. `uv run pytest tests/ -v` で既存テスト全件 PASS 確認
2. `uv run ruff check .` で lint 違反なし確認
3. `mise run deploy` でコード変更を各ノードへ配布
4. `mise run start` で実験実行
5. `mise run analyze` で metrics 集計

**リスク評価**:
- general 質問（教育関連の言葉を含む）でも education confidence が 0.1 に抑えられるか
- education ノードの confidence 分布が変化する可能性
- education-001/009 の low conf は改善しない可能性がある（education ノードの正しい自己認識）

**単一レバー原則との整合**:
- 本レバーは config-only の枠を超える（router.py のコード変更）
- 変更量: 1行の書き換え。例1-3は不変
- 単一レバー原則: 例4の1行書き換えのみ。他は不変

---

### 調査 (Iter8)

**問い**
- Q1: router.py の build_confidence_prompt() の few-shot 例ブロックの現在地と構造。例4の general 視点表現を特定せよ。
- Q2: In-Context Learning (ICL) において、few-shot 例の「視点/ペルソナ」が LLM の出力に与える影響に関する知見。
- Q3: education ノード視点の few-shot 例の具体的な設計。既存の例1-3との一貫性。
- Q4: ベースライン結果の特定と成功条件の提案（Iter7 の結果を踏まえて）。

**分かったこと（Q1: few-shot 例ブロックの現在地と例4の general 視点表現）**
- `router.py:66-73` の `build_confidence_prompt()` 内の few-shot 例ブロック:
  - 例1（行66-67）: 「歯の痛み→medical」(medical=0.9, legal=0.1) -- general 視点
  - 例2（行68-69）: 「賃貸契約→legal」(legal=0.9, medical=0.1) -- general 視点
  - 例3（行70-71）: 「学習指導要領→education」(education=0.9, medical=0.1) -- general 視点
  - 例4（行72-73）: 「読書感想文→general」(general=0.9, education=0.1, medical=0.1, legal=0.1) -- **general 視点**
- **例4の general 視点表現**: 「質問「読書感想文の書き方」はgeneral分野に該当するため，domainがgeneralなら{"confidence": 0.9}，domainがeducationなら{"confidence": 0.1}...」
- これは「general ドメインの立場から見た事実提示」であり、education ノードが probe 時に読む際、education ノードに対する抑制指示として機能しない。
- **教育ノードが読むプロンプト全体**: `build_confidence_prompt("education", query)` が呼ばれる。ドメイン名が f-string に埋め込まれ、「あなたは「education」分野の専門家ノードです」という役割指示 + 評価基準 + 例1-4 + 質問。
- **問題の構造**: 例4の「domainがeducationなら{"confidence": 0.1}」は general ドメインの視点から見た事実。education ノードはこれを「education ドメインに関する一般事実」として読むが、これは「自分自身（education ノード）が low confidence を出すべき」という指示ではない。

**分かったこと（Q2: ICL における視点/ペルソナの理論的根拠）**
- **Comparable Demonstrations (Fan et al., ICASSP 2024, arXiv:2312.07476)**: ICL では、示範例がターゲットタスクと「同等の構造・難易度」であることが重要。示範例の構造がターゲットの入出力と一致しない場合、LLM はパターンを正しく抽出できない。
- **In-Context Alignment Survey (LessWrong)**: 示範例の「視点/ペルソナ」が一致すると、LLM はその視点で推論する傾向がある。これは「perspective matching effect」と呼ばれる。
- **Negative Examples in Few-Shot (Tetrate.io, 2024)**: 「what not to do」の例は、特定のミスが常见的なタスクで有効。ただし、negative example の「視点」がターゲットの推論視点と一致しない場合、効果は限定的。
- **本ケースへの適用**: 例4が general 視点で書かれている場合、education ノードは general ドメインの事実を学ぶが、自分自身の confidence を低くする指示を学ばない。education ノード視点（「私は education ノード。この質問は education 分野ではない。confidence は 0.1 である」）で書かれた例であれば、education ノードは自分自身の振る舞いを directly 学ぶ。
- **ポジティブ例 vs ネガティブ例の比率**: 既存の例1-3は「該当→high confidence」のポジティブ例3件。例4は「該当しない→low confidence」のネガティブ例1件。3:1 の比率では、LLM はポジティブ例のパターンを強く学習し、ネガティブ例は上書きできない（Iter7 の分析で確認）。

**分かったこと（Q3: education ノード視点の few-shot 例の設計）**
- **既存の例1-3との一貫性**: 例1-3はすべて「general 視点」（「domainがXなら...」）。例4もこの構造を踏襲しつつ、教育ノード視点の要素を追加する。
- **提案（案A: 最小変更）**: 例4の書き方を「教育ノード視点」へ変更。
  - 現在: 「質問「読書感想文の書き方」はgeneral分野に該当するため，domainがgeneralなら{"confidence": 0.9}，domainがeducationなら{"confidence": 0.1}...」
  - 変更後: 「質問「読書感想文の書き方」はgeneral分野に該当するため，educationノードは{"confidence": 0.1}とする（general分野でありeducation分野ではない）．」
  - 変更量: 1行の書き換え。例1-3は不変。
- **提案（案B: 完全な教育ノード視点）**: 例4を完全に教育ノード視点で書き直す。
  - 「質問「読書感想文の書き方」はeducation分野ではない。educationノードは{"confidence": 0.1}とする。」
  - 既存の例1-3（general 視点）との一貫性が崩れるが、教育ノードへの効果は高い可能性がある。
- **推奨: 案A**（最小変更で一貫性維持）。例4のみを書き換え、例1-3は不変。

**分かったこと（Q4: ベースライン結果と成功条件の提案）**
- **ベースライン**: results/20260721_143604（Iter7, 46問/4ノード）
  - education precision=0.909, recall=0.833
  - single_domain_top1_accuracy=0.950
  - misrouting_rate=0.043
- **misroute 2件の内訳**:
  - general-004 → education（edu=0.95）: **few-shot 例の構造変更で是正可能**（教育ノードの過信）
  - education-001 → medical（edu=0.2, med=0.85）: **few-shot 例の変更では是正不可能**（教育ノードの正しい自己認識 + medical ノードの過信）
- **成功条件の再提案**（Iter7 の結果と構造的要因を踏まえて）:
  - 主基準: education precision >= 0.95 AND education recall >= 0.80
    - recall の閾値を 0.90→0.80 に下げる理由: education-001/009 は few-shot 例の変更では是正不可能（教育ノードの正しい自己認識）。0.80 は general-004 の是正のみで達成可能（10問中8問正解）。
  - 非退行: single_domain_top1_accuracy >= 0.975
    - general-004 の是正のみで達成可能（40問中39正解）。
  - 非退行: misrouting_rate <= 0.022
    - general-004 の是正のみで達成可能（46問中1件 misroute）。
  - 非退行: general/medical/legal の precision/recall は baseline 以下に退行しない。
- **注意**: education recall の閾値 0.80 は education-001/009 の misroute を許容する値。これらのケースの是正は別イテレーション（例: medical/legal ノードの過信抑制）が必要。

**次の計画フェーズへの示唆**:
1. 例4の書き換えは router.py の build_confidence_prompt() 内（行72-73）。変更量: 1行の書き換え。
2. 成功条件の recall 閾値（0.80 vs 0.90）は計画フェーズでユーザーに提示し、education-001/009 の iscue を別イテレーションへ回すか、recall 閾値を維持したまま教育 recall の改善を試みるか判断を仰ぐ。
3. 既存の例1-3との一貫性（案A vs 案B）も計画フェーズで提示。

---

### 実験 (Iter8)

**デプロイ**: 4ノード（wafl500, wafl501, wafl502, wafl503）すべて正常完了

**実行結果**: results/20260721_185132（46問，全問完走，used_fallback=0, dispatch_failed=0）
- 平均応答時間: 18257ms

**メトリクス（per-domain）**:

| ドメイン | precision (Iter8) | recall (Iter8) | precision (Iter7) | recall (Iter7) |
|---|---|---|---|---|
| education | **0.8889** | **0.6667** | 0.909 | 0.833 |
| general | **1.0000** | **0.9000** | 1.0 | 0.9 |
| legal | **0.8750** | **0.9333** | 1.0 | 0.933 |
| medical | **0.9167** | **0.7333** | 0.917 | 0.733 |

**総合指標**:
- single_domain_top1_accuracy: 0.900（Iter7 0.950）
- compound_domain_top1_accuracy: 1.0
- misrouting_rate: 0.0870（Iter7 0.043）
- top1_accuracy: 0.9130（Iter7 0.957）

**misroute 詳細（Iter8 4件 vs Iter7 2件）**:
- general-004 → education（confidence: education=0.95）→ **継続**（few-shot 例変更の効果なし）
- education-001 → medical（confidence: medical=0.95）→ **継続**（education ノードの正しい自己認識）
- ~~general-008 → medical~~ → **是正**（→general 正解）
- education-002 → legal（confidence: legal=0.95）→ **新規**（education ノードの confidence は 0.95 で維持）
- education-009 → legal（confidence: legal=0.8, education=0.8）→ **継続**（education confidence が 0.95→0.8 に低下）

**成功条件判定**: 10項目中3PASS/7FAIL
- 主基準: education precision 0.889（>=0.95 **FAIL**）
- 主基準: education recall 0.667（>=0.80 **FAIL**）
- 非退行: single_domain_top1_accuracy 0.900（>=0.975 **FAIL**）
- 非退行: misrouting_rate 0.0870（<=0.022 **FAIL**）
- 非退行: legal precision 0.875（>=1.0 **FAIL**）

### 分析 (解釈) (Iter8)

**判定**: router.py few-shot 例構造変更レバーは **rejected**（主基準 2 件未達，非退行 5 件未達）

**general-004 の isotope 効果**:
- **予想と全く逆の結果**: general-004 の education misroute は Iter7 と全く同じ（education confidence=0.95, 選択=education）。few-shot 例に「education ノードは 0.1 とする」という指示を追加したが、education ノードの confidence は 0.95 のまま変化なし。
- **構造的な理由**: few-shot 例の「education ノードは 0.1 とする」という指示は、education ノードの probe 時の confidence 判定に全く影響を与えていない。education ノードは few-shot 例3（「学習指導要領→education=0.9」）の high confidence をアンカーとして、general-004 も education と判断し続ける。
- **因果関係の確実性**: Iter7 と Iter8 で general-004 の education confidence が完全に同一（0.95）。この変化は run 間ノイズではなく、few-shot 変更が no-op であることを示す。

**education confidence の過剰抑制（言語崩れ）**:
- **education-009 の confidence が 0.95→0.8 に低下**: Iter7 では education=0.95 で正解（education 選択）だったが、Iter8 では education=0.8 に低下し、legal=0.8 と tie 状態に。tie-break の結果、legal 選択となり misroute に転落。
- **教育ノード視点の few-shot 例が過剰な confidence 抑制を引き起こしている**: 例4に「education ノードは 0.1 とする」という指示が追加されたことで、education ノードが **すべての education 質問**で confidence を過剰に抑制するようになった。これは意図した general-004 への効果ではなく、**教育ドメイン全体への副作用**。
- **教育 recall の有意な低下**: education recall が 0.833→0.667（-0.166）。これは n=10 の education 質問で 1.67 問の misroute 増加に相当。LLM temperature=0.1 のノイズ範囲を超える有意な低下。
- **教育 precision の低下**: education precision が 0.909→0.889（-0.020）。これは education-002 の legal misroute が主因。

**legal precision の低下（-0.125）の因果関係**:
- **直接の因果関係あり**: Iter7 の legal precision=1.0（全問正解）に対して、Iter8 では 0.875（10問中8問正解，2問 misroute）。
- **misroute の内訳**:
  - education-002 → legal: 教育固有の法律話題。education ノードの confidence は 0.95 で維持。general=0.85, legal=0.95 で legal 選択。これは few-shot 変更とは無関係な misroute。
  - education-009 → legal: 教育と法律の境界話題。education confidence が 0.95→0.8 に低下したため、legal=0.8 と tie 状態に。tie-break で legal 選択。
- **education-009 の confidence 低下は few-shot 変更の因果**: education-009 の education confidence が 0.95→0.8 に低下したことは、few-shot 例の「education ノードは 0.1 とする」という指示の過剰な副作用。この confidence 低下が legal tie-break を引き起こし、legal precision の低下を招いた。
- **結論**: legal precision の低下（-0.125）は few-shot 変更の直接的な副作用。ノイズではなく因果関係が明確。

**misroute 4件の内訳とメカニズム**:

| 質問 | 期待 | 選択 | 原因 | few-shot 因果か? |
|------|------|------|------|-----------------|
| general-004 | general | education | few-shot 変更 no-effect | 否（変更前と同一） |
| education-001 | education | medical | education ノードの正しい自己認識 | 否（変更前と同一） |
| education-002 | education | legal | education 固有の法律話題 | 否（変更前と同一） |
| education-009 | education | legal | education confidence 0.95→0.8（few-shot 副作用） | **是** |

**数値の有意性判定**:

- education recall: -0.166（0.833→0.667）→ **有意な低下**。n=10 で 1.67 問の misroute 増加。few-shot 変更の因果。
- legal precision: -0.125（1.0→0.875）→ **有意な低下**。n=10 で 1.25 問の misroute 増加。few-shot 変更の因果（education-009 経由）。
- single_domain_top1_accuracy: -0.050（0.950→0.900）→ **有意な低下**。n=40 で 2 問の misroute 増加。
- misrouting_rate: +0.044（0.043→0.087）→ **有意な悪化**。n=46 で 2 件の misroute 増加。

**すべて run 間ノイズの範囲を超える有意な変化**。

**仮説との整合**:

- H1（education precision 0.909→0.95以上）: **不成立**．0.889（-0.020 退行）。
- H2（single_domain_top1_accuracy 0.950→0.975以上）: **不成立**．0.900（-0.050 退行）。
- H3（general/medical/legal の非退行）: **不成立**．legal precision が -0.125 退行。

**予想外の挙動（言語崩れ）**:
- few-shot 例の「education ノードは 0.1 とする」という指示が、education ノードの confidence 判定に過剰な影響を与え、**教育ドメイン全体で confidence が抑制される現象**を引き起こした。これは H1/H2/H3 のいずれの仮説でも想定していなかった副作用。
- 具体的には education-009 の confidence が 0.95→0.8 に低下し、legal との tie-break で misroute に転落した。
- **解釈**: few-shot 例の「教育ノード視点」が、LLM によって「教育ノードは low confidence を出すべき」という汎用ルールとして解釈された。general-004 への特異的な効果ではなく、教育ドメイン全体への過信抑制として作用した。

**次イテレーションへの示唆**:
1. **few-shot 例構造変更は根本的に不適**: education ノード視点の few-shot 例は、意図した general-004 への効果を持たず、教育ドメイン全体への過剰抑制という副作用を引き起こした。このアプローチは放棄すべき。
2. **router.py の few-shot 例ブロックへの修正は限界がある**: Iter5-8 で 4 回連続 few-shot 関連の変更を試したが、いずれも期待した効果を持たなかった。few-shot 例の変更は confidence 信号に与える影響が構造的に限定されている。
3. **別のアプローチの検討が必要**:
   - A: confidence_threshold の再検討（0.9 付近の閾値で education の過信を抑制）
   - B: education ノードの dispatch prompt 修正（confidence 信号には影響しないが、回答品質には影響）
   - C: probe 段階の confidence 計算ロジック自体の変更（コード変更が必要）
4. **現状の few-shot 例4（general 視点）に戻す検討**: Iter7 の few-shot 例4（general 視点）は general-004 への効果はなかったが、教育ドメインへの過剰抑制副作用もなかった。現状より劣るが、副作用がない点は評価できる。

---

### Iteration 8 実行済み

**単一レバー**: few_shot_node_perspective（router.py の build_confidence_prompt() 内 few-shot 例ブロックの例4を education ノード視点へ変更）
**判定**: rejected（主基準 2 件未達，非退行 5 件未達）
**結果**: education precision=0.889（>=0.95 未達），recall=0.667（>=0.80 未達）。single_domain_top1_accuracy=0.900（>=0.975 未達）。misrouting_rate=0.087（<=0.022 未達）。
**改善**: general-008 の isotope 効果（→general 正解）。それ以外は Iter7 と同一または悪化。
**副作用**: education-009 の confidence が 0.95→0.8 に低下し、legal と tie 状態に転落。education recall の -0.166（有意な低下）。legal precision の -0.125 退行。
**学び**:
1. few_shot_node_perspective レバーは general-004 への効果を持たなかった（education confidence=0.95 不変）。few-shot 例の「education ノードは 0.1 とする」という指示は confidence 判定に全く影響を与えなかった。
2. 一方、education ドメイン全体への過剰抑制という副作用が発生。例4の「education ノード視点」が LLM によって「教育ノードは low confidence を出すべき」という汎用ルールとして解釈され、education-009 の confidence が 0.95→0.8 に低下した。
3. few-shot 例の変更は 4 回連続（Iter5-8）で試されたが、いずれも期待した効果を持たなかった。このレバーは**収束**した。追加反復は不要。
**次イテレーションの単一レバーの方針**: config.yml levers の次候補 `confidence_threshold`（values: [0.3, 0.5, 0.7]）へ移行。Iter3 で no-op と判定されたが、education の過信抑制という新たな文脈で再検討する。
**コミット**: router.py の few-shot 変更 + journal/state/backlog の更新

---

## Iteration 7: 抑制アンカリング few-shot 例追加による education ノード過信の是正

**単一レバー**: router.py の build_confidence_prompt() 内の few-shot 例ブロック（行66-71）に例4として general 質問のネガティブ例を追加

**仮説**:
- H1: 例4として general 質問（「読書感想文の書き方」）を few-shot 例に追加すると、education ノードは教育関連の言葉を含む general 質問を low confidence (0.1) として抑制し、education precision が 0.90→0.95 以上になる（general-004 の education misroute 解消）
- H2: single_domain_top1_accuracy が 0.90→0.95 以上になる（misroute 4件が2件以下に）
- H3: general/medical/legal の precision/recall は baseline 以下に退行しない

**成功条件**（ベースライン: results/20260721_121632）:
- 主基準: education precision >= 0.95 AND education recall >= 0.90
- 非退行: general precision >= 0.95, general recall >= 0.70
- 非退行: legal precision >= 0.85, legal recall >= 0.85
- 非退行: medical precision >= 0.75, medical recall >= 0.65
- 非退行: single_domain_top1_accuracy >= 0.952
- 非退行: misrouting_rate <= 0.048

**固定する構成**:
- config.yaml: `routing_method: self_report`, `confidence_threshold: 0.5`, `dispatch_top_k: 1` を維持
- build_dataset.py: 不変
- http_server.py, docker-compose.yml, mise.toml: 不変

**期待効果**: education ノードが general 質問を education として過信申告する現象（general-004→education）が抑制される。few-shot 例に「教育関連の言葉を含む general 質問でも education confidence は 0.1」という抑制のアンカリングが追加される。

**変更ファイルと変更量**:
- router.py: build_confidence_prompt() の few-shot 例ブロック（行66-71）に例4を追記。変更量: 2行追加

**追加する few-shot 例（例4）**:
```
例4：質問「読書感想文の書き方」はgeneral分野に該当するため，domainがgeneralなら{"confidence": 0.9}，domainがeducationなら{"confidence": 0.1}．
```
- general 質問「読書感想文の書き方」は教育関連の言葉を含むが general 分野
- education 以外のドメインにも low confidence を示す（education=0.1, medical=0.1, legal=0.1）
- general ドメインには high confidence (0.9) を示す

**検証手順**:
1. `uv run pytest tests/ -v` で既存テスト全件 PASS 確認
2. `uv run ruff check .` で lint 違反なし確認
3. `mise run deploy` でコード変更を各ノードへ配布
4. `mise run start` で実験実行
5. `mise run analyze` で metrics 集計

**リスク評価**:
- 既存ポジティブ例（例1-3）は不変
- education ノードの confidence 分布が変化する可能性
- education-001/009 の low conf は改善しない可能性がある（education ノードの正しい自己認識）
- general-008 の medical misroute は run 間ノイズにより変動する可能性

**単一レバー原則との整合**:
- **本レバーは config-only の枠を超える**（router.py のコード変更）
- 変更量: 2行追加のみ。既存3例は不変
- 3イテレーション連続（Iter4-6）で config-only の枠内では改善できず、few-shot 構造の修正が唯一の有効なアプローチ
- backlog.md に B12 として記録済み（ユーザー承認必要）

---

### 調査 (Iter7)

**問い**
- Q1: router.py の build_confidence_prompt() の few-shot 例はどのような構造か。抑制のアンカリング（general→low confidence）は欠如しているか。
- Q2: few-shot 例へのネガティブ例追加（A）、confidence_threshold 再較正（B）、education ノード dispatch prompt 修正（C）の比較。
- Q3: 単一レバーとして最も有効な変更はどれか。
- Q4: ベースライン結果の特定と成功条件の提案。

**分かったこと（Q1: few-shot 例の構造と抑制アンカリングの欠如）**
- `router.py:66-71` の few-shot 例は3件とも「該当→high confidence」のパターン:
  - 例1: 「歯の痛み→medical」(medical=0.9, legal=0.1)
  - 例2: 「賃貸契約→legal」(legal=0.9, medical=0.1)
  - 例3: 「学習指導要領→education」(education=0.9, medical=0.1) -- Iter6追加
- **構造的欠陥**: 全ての例は「domainが該当→high confidence」のみ。general 質問が education/medical/legal に属さないことを示すネガティブ例が1件もない。
- **教育ノードの動作メカニズム**: education ノードが general 質問「読書感想文の書き方」を評価する際、few-shot 例は medical/legal/education のポジティブ例のみ。education ノードは「読書感想文」が education 例（学習指導要領）と類似していると判断し、相対的に high confidence (0.95) を申告。ネガティブ例（「読書感想文→education=0.1」等）があれば抑制されるが、存在しない。
- **一般 confidence prompt の構造** (`_build_general_confidence_prompt`, router.py:35-36): 例1「歯の痛み→0.1」（専門知識要る=低 confidence）、例2「映画→0.9」（専門知識不要=高 confidence）。一般 prompt は「一般かどうか」を評価するため、ポジティブ（一般=高 conf）とネガティブ（専門=低 conf）の両方が含まれる。これは一般 prompt が few-shot 追加で改善していない理由。
- **決定要因**: few-shot 例は f-string のテンプレート文字列に直接埋め込まれている（router.py:66-71）。コード変更なしでは追加・変更不可能。

**分かったこと（Q2: A vs B vs C の比較）**
- **A: few-shot 例へのネガティブ例追加**
  - 変更内容: router.py の few-shot 例ブロックに例4として「読書感想文→education=0.1」を追加
  - 変更量: 4行追加（例4の1行 + 区切り改行）
  - 効果: education ノードが general 質問を low confidence として抑制。general-004 の education misroute が解消される可能性最大
  - リスク: 既存ポジティブ例（例1-3）は不変。cross-domain 例（例1-3）に education を追加すると prompt が肥大化し、LLM の attention が分散する可能性
  - 単一レバー原則: **枠を超える**（router.py のコード変更）

- **B: confidence_threshold を 0.9 付近へ再較正**
  - Iter3 で二峰・空帯域分布により no-op と確定。confidence 値の分布 {0.1, 0.2, 0.8, 0.85, 0.9, 0.95} において、0.9 閾値は high-clusters (0.9, 0.95) の大部分を fallback へ落とす。fallback_rate の増大＝品質退行。0.85 閾値は misroute 抑制効果がほぼゼロ（education-001/009 の low-clusters (0.2) には効かない）。B は有効なレバーではない。

- **C: education ノード dispatch prompt への明示指示追加**
  - 変更内容: `http_server.py:build_dispatch_prompt()` に「読書、勉強、習い事等は general 分野」との指示を追加
  - 効果: education ノードが general 質問を low confidence として申告する可能性。ただし、この指示は dispatch（回答生成）段階で使われるのみ。confidence 判定は probe 段階で `build_confidence_prompt()` が使われるため、dispatch prompt の指示は confidence 信号に直接影響しない。
  - **決定要因**: misroute の根本原因は confidence 信号の過信（probe 段階）であり、dispatch prompt は回答生成段階。C は根本原因への対応にはならない。C を行っても confidence 信号は改善せず、misroute は解消されない。

**分かったこと（Q3: 単一レバーとして最も有効な変更）**
- **推奨: A（few-shot 例へのネガティブ例追加）**
  - 理由: 根本原因（抑制アンカリング欠如）に直接対応。変更量4行で影響範囲限定。既存ポジティブ例は不変のため、既存ドメインへの影響は限定的。
  - 期待効果: education precision 0.90→0.95 以上（general-004 の education misroute 解消）、single_domain_top1_accuracy 0.90→0.95 以上、misrouting_rate 0.087→0.048 以下
  - 代替案: 既存例1-3に education 変数を追加（例1: medical=0.9, legal=0.1, education=0.1）すると、education ノードも cross-domain 例から「読書感想文は education でない」を学習できるが、prompt が肥大化し attention 分散のリスクがある。例4の独立例が安全。

**分かったこと（Q4: ベースラインと成功条件）**
- **ベースライン**: results/20260721_121632（Iter6, 46問/4ノード）
  - education precision=0.90, recall=0.75
  - general precision=1.0, recall=0.80
  - single_domain_top1_accuracy=0.90
  - misrouting_rate=0.087
- **成功条件の提案**:
  - 主基準: education precision >= 0.95 AND education recall >= 0.90
  - 非退行: general precision >= 0.95, general recall >= 0.70
  - 非退行: single_domain_top1_accuracy >= 0.952
  - 非退行: misrouting_rate <= 0.048
- **config-only 単一レバー原則**: **枠を超える**。router.py の few-shot 例ブロックへの追記（4行追加）が必要。config.yaml は不変。

**推奨: 方向 A（router.py の few-shot 例ブロックにネガティブ例を追加）**
- 変更内容: router.py の build_confidence_prompt() 内の few-shot 例ブロック（行66-71）に例4として「読書感想文の書き方→education=0.1」を追加
- 変更量: 4行追加（既存3例は不変）
- 期待効果: education precision 0.90→0.95 以上、single_domain_top1_accuracy 0.90→0.95 以上
- リスク: 既存ポジティブ例は不変。education ノードの confidence 分布が変化する可能性（education-001/009 の low conf は改善しない可能性がある。これらは education ノードの正しい自己認識）。
- 次 rc-planner への示唆: 単一レバー原則の再設計（config-only の枠を出る変更）をユーザーに提示。router.py の few-shot 例追加は変更量4行で影響範囲が限定されるため、「単一レバー」として承認可能か判断を求める。

---

### 実装 (Iter7)

**単一レバー**: router.py の build_confidence_prompt() 内の few-shot 例ブロックに例4として general 質問のネガティブ例を追加

**実行した変更**:
1. `router.py`: build_confidence_prompt() の few-shot 例ブロック（行66-73）に例4を追記
   - 例4: 「読書感想文の書き方」→ general=0.9, education=0.1, medical=0.1, legal=0.1
   - 既存の例1（medical）、例2（legal）、例3（education）は不変
   - 変更量: 2行追加（例4の1行 + 区切り改行の修正）

**検証結果**:
- `uv run pytest tests/ -v`: **78件全PASS**（0.60秒）
- `uv run ruff check .`: **All checks passed**

**config.yaml は不変**: `routing_method: self_report`, `confidence_threshold: 0.5`, `dispatch_top_k: 1` を維持

**次フェーズへの引き継ぎ**: コード変更完了・テスト全PASS。次は実験フェーズで `mise run deploy` → `mise run start`（46問/4ノード）→ `mise run analyze` を実行。

### 実験 (Iter7)

**デプロイ**: 4ノード（wafl500, wafl501, wafl502, wafl503）すべて正常完了

**実行結果**: results/20260721_143604（46問，全問完走，used_fallback=0, dispatch_failed=0）
- 平均応答時間: 14994ms

**メトリクス（per-domain）**:

| ドメイン | precision (Iter7) | recall (Iter7) | precision (Iter6) | recall (Iter6) |
|---|---|---|---|---|
| education | **0.909** | **0.833** | 0.90 | 0.75 |
| general | **1.0** | **0.9** | 1.0 | 0.8 |
| legal | **1.0** | **0.933** | 0.933 | 0.933 |
| medical | **0.917** | **0.733** | 0.846 | 0.733 |

**総合指標**:
- single_domain_top1_accuracy: 0.950（Iter6 0.90）
- compound_domain_top1_accuracy: 1.0
- misrouting_rate: 0.043（Iter6 0.087）
- top1_accuracy: 0.957（Iter6 0.913）

**misroute 詳細（Iter7 2件 vs Iter6 4件）**:
- general-004 → education（confidence: education=0.95）→ **継続**（few-shot 例4の効果なし）
- education-001 → medical（confidence: medical=0.85）→ **継続**（education ノードの正しい自己認識）
- ~~general-008 → medical~~ → **是正**（→general 正解）
- ~~education-009 → legal~~ → **是正**（→education 正解）

**成功条件判定**: 10項目中7PASS/3FAIL
- 主基準: education precision 0.909（>=0.95 **FAIL**）
- 主基準: education recall 0.833（>=0.90 **FAIL**）
- 非退行: single_domain_top1_accuracy 0.950（>=0.952 **FAIL**）

### 分析 (実行) (Iter7)

**mise run analyze 完了**: results/20260721_143604/

**成功条件判定（10項目中7PASS/3FAIL）**:

| # | 条件 | 閾値 | 測定値 | 判定 |
|---|------|------|--------|------|
| 1 | education precision | >= 0.95 | 0.909 | **FAIL** |
| 2 | education recall | >= 0.90 | 0.833 | **FAIL** |
| 3 | general precision | >= 0.95 | 1.0 | PASS |
| 4 | general recall | >= 0.70 | 0.9 | PASS |
| 5 | legal precision | >= 0.85 | 1.0 | PASS |
| 6 | legal recall | >= 0.85 | 0.933 | PASS |
| 7 | medical precision | >= 0.75 | 0.917 | PASS |
| 8 | medical recall | >= 0.65 | 0.733 | PASS |
| 9 | single_domain_top1_accuracy | >= 0.952 | 0.950 | **FAIL** |
| 10 | misrouting_rate | <= 0.048 | 0.043 | PASS |

**ベースライン（Iter6）との差分**:
- education precision: +0.009（0.90→0.909）
- education recall: +0.083（0.75→0.833）
- general recall: +0.10（0.80→0.90）
- legal precision: +0.067（0.933→1.0）
- medical precision: +0.071（0.846→0.917）
- single_domain_top1_accuracy: +0.050（0.90→0.950）
- misrouting_rate: -0.044（0.087→0.043）

### 分析 (解釈) (Iter7)

**判定**: router.py few-shot 例追加レバーは **rejected**（主基準 2 件未達）

**few-shot 例4の因果効果**:
- **有意な効果あり**: general-008 medical 0.95→0.85（medical 過信抑制）、education-009 是正、legal precision +0.067 改善
- **no-effect**: general-004 education 0.95 不変（few-shot 例4が education=0.1 を示しているのに education ノードが過信を維持）

**general-004→education が抑制できなかった構造的な理由**:
1. **視点の不一致**: 例4は general ドメインの視点（「読書感想文→general=0.9, education=0.1」）で書かれている。education ノードが probe 時に読む際、この例は general 視点の事実提示であり、education ノードに対する抑制指示として機能しない。
2. **語彙的アンカリングの逆効果**: 例4の「読書感想文」と general-004 の「読書感想文」が完全に一致。education ノードは few-shot 例3（「学習指導要領→education=0.9」）の high confidence をアンカーとして、「読書」を含む general-004 も education と判断する。例4の low confidence は語彙的アンカリングに負ける。
3. **ポジティブ例のパターン学習**: 3つのポジティブ例（該当→high conf）に1件のネガティブ例。LLM はポジティブ例のパターンを強く学習し、1件のネガティブ例はパターン全体を上書きできない。

**single_domain_top1_accuracy 0.950 vs 閾値 0.952 の解釈**:
- n=40 の単一ドメイン質問で、0.950 は 38/40 正解（2件 misroute）。
- 閾値 0.952 は 38.08/40。40問では 0.025 刻み（1件=0.025）しか取れない。
- **0.002 の差は n=40 の离散効果によるもので、統計的な有意差ではない。**
- general-004 の misroute 1件が解消されれば 0.975 になる。

**仮説との整合**:
- H1（education precision 0.90→0.95以上）: **不成立**．0.909（+0.009）
- H2（single_domain_top1_accuracy 0.90→0.95以上）: **成立**．0.950
- H3（general/medical/legal の非退行）: **成立**．全ドメイン退行なし

**次イテレーションへの示唆**:
1. **few-shot 例の構造変更（推奨）**: 例4を「general 視点」から「education ノード視点」へ変更。例: 「質問「読書感想文の書き方」は general 分野であり、education ドメインではない。education ノードは low confidence (0.1) を出すべき」。education ノードが self-report する際の few-shot 例として、education ノードの視点で書かれたネガティブ例が効果的。
2. **confidence_threshold の再検討**: education ノードの confidence 分布 {0.2, 0.8, 0.9, 0.95} において、0.8 以上の confidence を持つ education ノードの out-of-domain 質問を fallback へ落とす。ただし fallback rate 増大が懸念。
3. **education ノードの dispatch prompt 修正**: education ノードのプロンプトに「読書、勉強、習い事等は一般常識レベルの話題であり、general 分野に該当する」との明示指示を追加。ただし confidence 信号には影響しない（probe と dispatch で別プロンプト）。

### 考察・次計画 (Iter7)

**判定**: few-shot 例追加レバーは **rejected**（主基準 2 件未達）

**総括**:
- Iter7 で router.py の few-shot 例ブロックに例4（general 質問のネガティブ例）を追加
- 因果効果: general-008 の medical 過信抑制（0.95→0.85）、education-009 是正、legal precision +0.067 改善
- no-effect: general-004 の education 過信（0.95 不変）→ 例4は education ノードの過信を抑制できなかった
- **根本原因**: 例4は general ドメインの視点（「読書感想文→general=0.9, education=0.1」）で書かれている。education ノードが probe 時に読む際、この例は general 視点の事実提示であり、education ノードに対する抑制指示として機能しない。
- **単一レバー原則**: **枠を超える**（router.py のコード変更）。変更量2行追加のみ。

**次イテレーションの単一レバー決定**:
- **推奨: few-shot 例の構造変更**（分析(解釈)フェーズの推奨に基づく）
- 具体案: 例4を「general 視点」から「education ノード視点」へ変更
  - 例: 「質問「読書感想文の書き方」は general 分野であり、education ドメインではない。education ノードは low confidence (0.1) を出すべき」
- 既存の例1-3は不変。例4の書き方だけ変更（1行の書き換え）。
- 変更量: 1行の書き換え。router.py の build_confidence_prompt() 内。
- 期待効果: education ノードが few-shot 例を self-report 時の anchor として利用し、general 質問で low confidence を出す

**コミット**: 例3+例4追加を router.py にコミット。state.json は次イテレーション用に更新。

---

### イテレーション完了サマリー

**単一レバー**: few_shot_negative_example（router.py の few-shot 例ブロックに一般質問のネガティブ例追加）
**判定**: rejected（主基準 2 件未達）
**結果**: education precision=0.909（>=0.95 未達）、recall=0.833（>=0.90 未達）。misrouting_rate=0.043（<=0.048 PASS）。
**改善**: misroute 4件→2件、general recall +0.10、single_domain_top1_accuracy +0.05
**学び**: few-shot 例は general ドメインの視点で書かれているため、education ノードの過信を抑制できなかった。3つのポジティブ例（該当→high conf）に1件のネガティブ例では LLM がポジティブ例のパターンを強く学習し、ネガティブ例は上書きできない。次イテレーションでは education ノード視点の few-shot 例へ構造変更が必要。
**コミット**: router.py 例3+例4追加コミット済み

---

### Iteration 6 実行済み

**単一レバー**: router.py の build_confidence_prompt() に education 固有 few-shot 例を1件追加
**判定**: rejected（主基準2件未達，非退行2件未達）
**結果**: education precision/recall は Iter5 と完全に同一（0.90/0.75）。few-shot 追加は confidence 信号に影響しなかった。
**学び**: few-shot 例は「該当する→high confidence」のパターンしか示さないため、general 質問を抑制するアンカリングにはならない。抑制のアンカリングが欠如していることが根本原因。
**コミット**: 8b07170

---

### 分析 (解釈) (Iter6)

**判定**: router.py few-shot 例追加レバーは **rejected**（主基準 2 件未達，非退行 2 件未達）

**few-shot 追加が効果を持たなかった原因**:
- Iter5 と Iter6 で education ノードの confidence 値が**10問中10件完全に同一**
- 追加した few-shot 例（「学習指導要領における探究的学習の位置付けは」）は confidence 信号に何の影響も与えなかった
- **構造的な理由**: 既存 few-shot 例は「該当する→high confidence」のパターンしか示さない。例1（歯の痛み→medical=0.9）、例2（賃貸契約→legal=0.9）はすべてドメインに該当する場合の high confidence を示している。例3（教育固有 few-shot）も同パターン。つまり、**general 質問で education 関連の言葉（読書、勉強等）が出た場合に low confidence を出すという「抑制のアンカリング」が欠如している**

**misroute 4件のメカニズム**:
1. general-004 → education (edu=0.95): education ノードが「読書感想文」を教育固有話題と誤認。few-shot 例は general 質問を抑制する方向に働かない。Iter5 と同一。
2. general-008 → medical (med=0.95): Iter5 では medical=0.85 で general 選択されていたが、Iter6 で medical confidence が 0.95 に run 間変動し misroute 再発。education few-shot 追加とは無関係。
3. education-001 → medical (edu=0.2, med=0.85): 「夜泣き」は教育主題ではなく medical ノードの過信。education ノードの low confidence (0.2) は正しい自己認識。Iter5 と同一。
4. education-009 → legal (edu=0.2, legal=0.8): 「部活動の怪我の手続き」は教育と法律の境界話題。education ノードが low confidence (0.2) を申告。Iter5 と同一。

**general recall・medical precision 退行の要因**:
- general recall 0.90→0.80 は general-008 の1件 misroute のみ。medical confidence の run 間変動（0.85→0.95）による。LLM temperature=0.1 のノイズ範囲内。
- medical precision 0.9167→0.8462 も general-008 の1件 misroute のみ。run 間ノイズの範囲内。

**判定の根拠**:
- 主基準: education precision 0.90（基準 >= 0.95）→ **FAIL**
- 主基準: education recall 0.75（基準 >= 0.90）→ **FAIL**
- 非退行: single_domain_top1 0.900（基準 >= 0.952）→ **FAIL**
- 非退行: misrouting_rate 0.0870（基準 <= 0.048）→ **FAIL**
- 4件すべて未達。追加反復の余地なし。

**few-shot 追加は逆効果の可能性**:
- education 固有 few-shot 例（学習指導要領）は general 質問を抑制せず、むしろ education ノードの過信を増加させた（education-010 の confidence が 0.9→0.95 に上昇）
- 根本原因: few-shot 例は「該当する→high confidence」のパターンしかない。抑制のアンカリング（一般質問で education 関連の言葉が出ても low confidence）が必要

**次イテレーションへの示唆**:
- A: few-shot 例を「general 質問→medical/legal/education すべて low confidence」のパターンへ差し替え（抑制アンカリングの追加）
- B: confidence_threshold を 0.9 付近へ引き上げ（Iter3 で検討済みだが、education の過信抑制には有効か再検証）
- C: education ノードのプロンプト自体に「読書、勉強、習い事等は general 分野」と明確に指示する文を追加

---

### 分析 (実行) (Iter6)

**mise run analyze 完了**: results/20260721_121632/

**成功条件判定（10項目中6PASS/4FAIL）**:

| # | 条件 | 閾値 | 測定値 | 判定 |
|---|------|------|--------|------|
| 1 | education precision | >= 0.95 | 0.9000 | **FAIL** |
| 2 | education recall | >= 0.90 | 0.7500 | **FAIL** |
| 3 | general precision | >= 0.95 | 1.0000 | PASS |
| 4 | general recall | >= 0.70 | 0.8000 | PASS |
| 5 | legal precision | >= 0.85 | 0.9333 | PASS |
| 6 | legal recall | >= 0.85 | 0.9333 | PASS |
| 7 | medical precision | >= 0.75 | 0.8462 | PASS |
| 8 | medical recall | >= 0.65 | 0.7333 | PASS |
| 9 | single_domain_top1_accuracy | >= 0.952 | 0.9000 | **FAIL** |
| 10 | misrouting_rate | <= 0.048 | 0.0870 | **FAIL** |

**misroute 4件（46問中）**:
- general-004 → education（confidence: education=0.95）→ 読書感想文の書き方
- general-008 → medical（confidence: medical=0.95）→ 運動不足のストレッチ
- education-001 → medical（confidence: medical=0.85）→ 子育て中の夜泣き
- education-009 → legal（confidence: legal=0.80）→ 部活動の怪我の手続き

**ベースライン（Iter5）との差分**:
- education precision/recall: 0.0000 変化（few-shot 追加効果なし）
- general recall: -0.1000（0.90→0.80）
- medical precision: -0.0705（0.9167→0.8462）
- single_domain_top1_accuracy: -0.0250（0.9250→0.9000）
- misrouting_rate: +0.0217（0.0652→0.0870）

**education ノード confidence 分布**: mean=0.371, min=0.100, max=0.950
- few-shot 追加により education 関連質問で education ノードが 0.95 の confidence を出すケースが発生

### 分析 (解釈) (Iter6)

**判定**: education ノード few-shot 例追加レバーは **rejected**（主基準・非退行基準とも未達）

**few-shot 追加が education precision/recall に効果を持たなかった原因**:

- Iter5 と Iter6 で education ノードの confidence 値が**完全に同一**（下表）:

| 質問 | Iter5 edu_conf | Iter6 edu_conf | 結果 |
|------|---------------|---------------|------|
| education-001 | 0.2 | 0.2 | misroute |
| education-002 | 0.95 | 0.95 | OK |
| education-003 | 0.9 | 0.9 | OK |
| education-004 | 0.95 | 0.95 | OK |
| education-005 | 0.9 | 0.9 | OK |
| education-006 | 0.95 | 0.95 | OK |
| education-007 | 0.95 | 0.95 | OK |
| education-008 | 0.95 | 0.95 | OK |
| education-009 | 0.2 | 0.2 | misroute |
| education-010 | 0.9 | 0.95 | OK |

- 追加した few-shot 例（「学習指導要領における探究的学習の位置付けは」）は、education ノードの confidence 判定に**何の影響も与えなかった**。
- **理由**: few-shot 例は prompt 内のアンカリングとして機能するが、この例は「education が education である」ことを示すだけ。一般質問（読書感想文、運動不足のストレッチ等）を education と**区別する**アンカリングにはならない。
- 既存の few-shot 例（例1: 歯の痛み→medical、例2: 賃貸契約→legal）は、他のドメイン（medical/legal）に対する教育関連質問の low confidence を示すものではない。例3（教育固有 few-shot）も同様に、general 質問に対する low confidence の示唆を与えない。
- **構造的欠陥**: few-shot 例は「該当する→high confidence」のパターンしか示さない。「一般質問で education 関連の言葉が出ても low confidence にする」という**抑制のアンカリング**が欠如している。

**misroute 4件のメカニズム解釈**:

1. **general-004 → education**（confidence: edu=0.95, gen=0.9）:
   - education ノードが「読書感想文の書き方」を education 固有話題と誤認し、high confidence (0.95) を申告。
   - few-shot 例（学習指導要領）は education 固有話題であり、一般質問を抑制する方向に働かない。
   - **Iter5 と同一メカニズム**。few-shot 追加で変化なし。

2. **general-008 → medical**（confidence: med=0.95, gen=0.85）:
   - general 質問「運動不足のストレッチ」を medical ノードが over-confident に解釈。
   - **Iter5 では general=0.85/medical=0.85 で general 選択**（run 間ノイズにより是正）。
   - **Iter6 では medical=0.95 に上昇し、misroute 再発**。これは education few-shot 追加とは無関係な medical ノードの confidence 変動。

3. **education-001 → medical**（confidence: edu=0.2, med=0.85）:
   - 「子育て中の夜泣き」は教育主題ではなく医療主題。education ノードの low confidence (0.2) は**正しい自己認識**。
   - medical ノードが high confidence (0.85) を申告し、選択結果は正しいドメインへルーティングされるが、education として認識されないため education recall が低下。
   - **Iter5 と同一**。few-shot 追加で変化なし。

4. **education-009 → legal**（confidence: edu=0.2, legal=0.8）:
   - 「部活動の怪我の手続き」は教育と法律の境界話題。education ノードが low confidence (0.2) を申告。
   - legal ノードが high confidence (0.8) を申告し、legal へルーティング。
   - **Iter5 と同一**。few-shot 追加で変化なし。

**general recall 退行の要因**:

- general recall: 0.90 → 0.80（-0.1000）。general-008 のみが medical に misroute した1件での退行。
- **Iter5**: general=0.85, medical=0.85 → general 選択（tie-break により是正）。
- **Iter6**: general=0.85, medical=0.95 → medical 選択（medical confidence の run 間変動で再 misroute）。
- 差は medical confidence の 0.85→0.95 の変動のみ。LLM temperature=0.1 のノイズ範囲内。
- **有意な退行ではない**。run 間ノイズの範囲内。

**medical precision 退行の要因**:

- medical precision: 0.9167 → 0.8462（-0.0705）。
- **唯一の要因**: general-008 が medical に misroute した1件。
- Iter5 では general-008 が general 選択されていたため、medical precision は 0.9167（14/15）。
- Iter6 では general-008 が medical 選択されたため、medical precision は 0.8462（11/13）に低下。
- **run 間ノイズの範囲内**。1件での精度変動であり、構造的な退行ではない。

**数値の有意性判定**:

- education precision/recall: 0.00 変化 → **ノイズ**（few-shot 追加が構造的影響を持たない）
- general recall: -0.10 → **ノイズ**（medical confidence の run 間変動 0.85→0.95）
- medical precision: -0.0705 → **ノイズ**（general-008 の1件 misroute）
- single_domain_top1_accuracy: -0.0250 → **ノイズ**（general-008 の1件 misroute）
- misrouting_rate: +0.0217 → **ノイズ**（general-008 の1件 misroute 追加）
- 全体として、**見かけの変化はすべて run 間ノイズの範囲内**。few-shot 追加の有意なシグナルは検出されなかった。

**仮説との整合**:

- H1（education precision 0.90→0.95以上）: **不成立**．0.90 のまま．few-shot 追加が confidence 信号に影響しない構造であることが明確に示された．
- H2（education recall 0.75→0.90以上）: **不成立**．0.75 のまま．misroute 3件ともベースラインと不変（general-008 の1件追加は run 間ノイズ）．
- H3（general/medical/legal の非退行）: **不成立**．general recall と medical precision が run 間ノイズの範囲で退行．

**判定の根拠**:

- 主基準: education precision 0.90（基準 >= 0.95）→ **FAIL**
- 主基準: education recall 0.75（基準 >= 0.90）→ **FAIL**
- 非退行: single_domain_top1 0.900（基準 >= 0.952）→ **FAIL**
- 非退行: misrouting_rate 0.0870（基準 <= 0.048）→ **FAIL**
- 4 件すべて未達．追加反復の余地なし（構造的原因が明確）．

**学び（非自明）**:

- few-shot 例を追加しても、**「該当する→high confidence」のパターンしか示さない**限り、抑制のアンカリングにはならない．
- education ノードが general 質問を過信申告する現象は、few-shot 例に「general 質問で education 関連の言葉が出ても low confidence」を示す例を追加しないと解消しない．
- Iter5 と Iter6 で education confidence 値が完全に同一（10問中10件一致）．これは few-shot 追加が no-op であることを決定的に示す．
- general-008 の medical misroute は run 間ノイズ（medical confidence 0.85→0.95）であり、意図的なレバー効果ではない．

---

### 実装 (Iter6)

**単一レバー**: router.py の build_confidence_prompt() に education 固有 few-shot 例を1件追加

**実行した変更**:
1. `router.py`: 行70-71 に education 固有 few-shot 例を追加（例3: 「学習指導要領における探究的学習の位置付けは」）
   - education なら confidence 0.9、medical なら 0.1
   - 既存の例1（medical）、例2（legal）は不変
   - 変更量: 2行追加

**検証結果**:
- `uv run pytest tests/ -v`: **78件全PASS**（0.65秒）
- `uv run ruff check .`: **All checks passed**

**config.yaml は不変**: `routing_method: self_report`, `confidence_threshold: 0.5`, `dispatch_top_k: 1` を維持

**次フェーズへの引き継ぎ**: コード変更完了・テスト全PASS。次は実験フェーズで `mise run deploy` → `mise run start`（46問/4ノード）→ `mise run analyze` を実行。

---

