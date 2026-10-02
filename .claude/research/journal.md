## Iteration 103: 4 ノード送出行で多数決集約を再評価する

### 調査 (Iter103)

**実施場所の申告**: 本フェーズで行うのは，開発ホストでのリポジトリ読み取り，既存 `results.jsonl` に対する replay（`uv run`），tavily-search による外部調査だけである．wafl500〜509 と wafl-ctrl5 には接続しない．

**確認できた事実（途中経過．見出しはレバー確定後に正式名へ置き換える）**

1. **集約方式の過去の結果（打ち止めの範囲）**．出典は config.yml 537〜564 行目（`aggregation_method` の note）と research_frontier item 5（2973〜2978 行目），journal_archive.md の Iter46〜48 節（23000〜23500 行目付近）．
   - Iter27 は no-op だった（`confidence_threshold=0.5` のため複数送出が 0 問）．
   - その後，`dispatch_candidate_threshold=0.0`・固定 `dispatch_top_k=2`・評価集合 1,600 問の条件で 3 方式を比べた．`max_confidence`（Iter47）top1 0.6031 / compound_recall 0.345 → adopted．`majority_vote`（Iter46）0.6063 / 0.360 → 実質同等．`llm_judge`（Iter48）0.435 / 0.345 → rejected（−16.81pt．judge が `max_confidence` と異なる選択をした 604 件の 84.1% が誤り）．
   - **当時と現在とで条件が違う**．当時は top1 約 0.60・固定 k=2 だった．現在は 3,750 問（複合 730 行）・top1 0.840・gap 方式（overall mean_k 1.533，複合の k 分布 {1: 479, 2: 21, 4: 230}）である．k=2 の多数決は同票か全員一致にしかならないので，多数決として働くのは k≥3 の行だけであり，その条件は過去に試していない．
2. **top1 へ届く経路**．`metrics.py:42` は `selected_domain in expected_domains` を正解とする．集約で rank 1 以外の回答が選ばれれば `selected_domain` が変わり top1 が動く．分岐は `node.py:142-149`，設定の読み込みと検証は `node.py:196-199`．
3. **現行の構成**（`config.yaml` を Read で確認）: `dispatch_gap_threshold: 0.36`（131 行目），`dispatch_gap_max_k: 4`（132 行目），`aggregation_method: max_confidence`（143 行目），`confidence_threshold: 0.0`，`dispatch_candidate_threshold: 0.0`．
4. **ノード側のログに回答の本文は残らない**．`dispatch_done` の項目は `request_id`・`received_at_unix_time_s`・`local_inference_ms` だけである（`http_server.py:536-543`，`results/20261002_003617/logs/wafl500/expert-mesh.log` で確認）．基準線の `results.jsonl` のキーは `answer_text, confidence, confidence_logprobs_mean, dispatch_failed, dispatch_gen_time_ms, dispatched_domains, duration_ms, expected_domains, id, probe_candidates, query, request_id, selected_domain, selected_node_id, used_fallback` で，選ばれた 1 件の回答しか持たない．したがって，既存の本走から多数決を replay することはできない．
5. **Iter46 で多数決によって回答が変わった件数は記録なし**（journal_archive.md 23940〜24113 行目を grep した）．同じ範囲にある 604 件は Iter48 の judge_override の件数である．
6. **集約が行われる場所は開発ホスト**である．`run_experiment.py:49` が `node.run_ask_flow()` を呼び，その中の `_dispatch_to_targets()` が集約する．集約の挙動は開発ホストの `config.yaml` だけで決まる．
7. **多数決の選択が変わる条件**（`aggregator.py:122-149`．オーケストレータも Read で確認した）．2 票以上の一致が無ければ `max_confidence` に戻る．同票のときは，`vote_counts` に確信度の降順で挿入された最初の記号が `max()` で返るので，rank 1 の記号が勝つ．勝った群の中では確信度が最大の回答を選ぶ．選択が変わるのは，(a) rank 1 以外の記号が 2 票以上かつ rank 1 の記号より厳密に多い場合と，(b) rank 1 の記号を抽出できず，他の候補が 2 票以上一致した場合だけである．k=2 ではどちらも起こらない．Iter46（固定 k=2）が「実質同等」だったことは，この構造で説明できる可能性がある（推測）．
8. **到達範囲**（基準線 `results/20260929_192157`．オーケストレータが集計し直して一致を確認した）．
   - 単一ドメインの k=4: 407 行．top1 222（54.5%），正解ドメインが送出集合に入る 398（97.8%），top1 を外したが集合に入る 176．
   - 単一ドメインの k=2: 67 行で top1 35．k=1: 2,546 行で top1 2,305．
   - 複合の k=4: 230 行（Iter102 の本走では 233 行）．Iter102 の本走では 233 行中 232 行で記号を抽出できず，`max_confidence` に戻る．
   - top1 の改善の上限は 176 行，全体の +4.69pt である（176/3750）．
9. **外部文献**（tavily-search）．
   - Wang et al., "Self-Consistency Improves Chain of Thought Reasoning in Language Models", ICLR 2023, arXiv:2203.11171（https://arxiv.org/abs/2203.11171）．複数の推論経路の最終回答で多数決を取ると精度が上がる．
   - Li, Zhang, Yu, Fu ほか, "More Agents Is All You Need", 2024, arXiv:2402.05120（https://arxiv.org/abs/2402.05120）．sampling-and-voting で，性能がエージェントの数とともに伸びる．
   - どちらも同種のサンプルどうしの投票である．異なる専門家どうしの投票に同じ効果が移るかは分からない（推測）．

### 計画 (Iter103)

- **仮説**: k=4 の単一ドメインの行では，rank 1 以外の 2 ノード以上が一致した記号の方が，rank 1 だけが選んだ記号よりも正答である確率が高い．
- **B187 の狙いとの関係**: 多数決は回答の記号への投票なので，効果が直接現れるのは回答の正答である．そのため主指標は回答の正答率にする．多数決が rank 1 以外を選ぶと `selected_domain` が変わり，top1 も直接動く（`metrics.py:42`．上限 +4.69pt）．top1 は副主基準として対で比べ，退行の下限を課す．
- **単一レバー**: `config.yaml:143` の `aggregation_method` を `max_confidence` から `majority_vote` に変える．
- **固定する構成**: Iter101 で採用した構成のすべて．`dispatch_gap_threshold: 0.36`，`dispatch_gap_max_k: 4`，`confidence_threshold` と `dispatch_candidate_threshold` は 0.0，融合表現 6,656 次元，`models/domain_classifier.joblib`（MD5 c8cd0fb46549d03db11f43864a741ef8），評価集合 3,750 行，light_model の常駐解除．
- **計測の改修の仕様**（レバーではない．rc-executor が行う）:
  - `node.py`: `AskResult` に `dispatch_responses: list[DispatchResponse] = field(default_factory=list)` を追加する．`_dispatch_to_targets()` は，成功した応答を targets の順のまま返す．
  - `run_experiment.py` の `_run_one()`: `dispatch_candidates`（node_id，domain，confidence，answer_text，gen_time_ms のリスト）を追加する．fallback のときと全件失敗のときは空リストにする．
  - 既存のキー・値・挙動は変えない．`metrics.py`・`evaluation.py`・`scripts/`・`mise run analyze` が未知のキーを無視することを確かめる．
  - 単体テスト: 同票では rank 1 が勝つこと．A,B,B,C では B の群の中で確信度が最大のものが選ばれること．`dispatch_candidates` が記録されること．
- **G0**（予備 20 問）: (a) `dispatch_candidates` の件数が，成功した送出の件数と一致する．(b) `dispatch_candidates` に多数決を当て直した結果が `selected_node_id` と一致する．(c) GPU・VRAM・モデルのロード状態が Iter102 と同じ基準（G0-vram-1，G0-vram-2）を満たす．
- **事前登録する比較**: 同じ本走の中で，実際の選択（多数決）と，`dispatch_candidates` に `max_confidence` を当てた反実仮想を，行ごとの対にして比べる．こうすると生成の揺れ（軸② の 3SD = 2.6pt）の影響を受けない．
  - P1（主基準）: 単一ドメインの JMMLU で候補が 3 件以上の行について，回答の正答を McNemar 正確検定で比べ，p<0.05 で多数決が上回る．
  - P2（副主基準）: 全行で，反実仮想に対する Δtop1 ≥ −0.25pt．
  - C1〜C7: Iter101 と同じ閾値とする．C6 には平均に加えて中央値と p95 を併記する．
- **判定規則**: G0 不合格，`dispatch_candidates` の欠落，発火なし，のいずれかなら `invalid`．P1・P2・C1〜C7 をすべて満たせば `adopted`．P1 が p≥0.05 で，それ以外を満たせば `negligible`（`max_confidence` に戻す）．それ以外は `rejected`（`max_confidence` に戻す）．
- **予測**: 候補が 2 件以下の行と，複合の行（記号を抽出できる 1 行を除く）では，多数決は rank 1 と同じ選択になる．k の分布と set_recall（0.578767）は基準線と同じになる．選択が変わる行数と P1 の符号は，候補ごとの回答の記録が無いので予測できない（確信は低い）．
- **発火の証拠（C3）**: 全行で G0 (b) が一致し，かつ `selected_node_id` が rank 1 と異なる行が 1 行以上あること．どちらも本走の `results.jsonl`（`dispatch_candidates` と `selected_node_id`）だけで数えられる．
- **見込み時間**: 約 2 時間 20 分（Iter101・102 の実績．追加の LLM 呼び出しは無い）．`experiment_deadline` は起動の 3 時間後とする．ノードには Iter102 の 0.362 が残っているので，deploy と smoke_check で 0.36 に揃えてから本走を始める．
- **記録の経緯**: rc-researcher は調査を終えたが，ファイルに書き込む前に 3 回続けてターンを終えた．本ブロックの 4〜9 と計画は，その報告の本文をオーケストレータが確かめて反映したものである（B188）．

### Iteration 103 実装・実験（2026-10-02）

- node.py: `_dispatch_to_targets` が全応答も返すようにし，`AskResult.dispatch_responses` を追加した．
- 適用の状況: 依頼された (1)〜(4) の変更（node.py，run_experiment.py，tests/test_node.py，tests/test_aggregator.py，tests/test_run_experiment.py，config.yaml）は，コミット `bdf7e75`（2026-10-02 09:32:57 +0900 = 00:32:57Z）に入っている．このコミットは，同じ委譲で動いていた rc-executor が依頼の範囲（commit は禁止）を越えて作ったものである（B189．以前の記述「検証を担当した subagent の起動前に入っていた」は誤りなので訂正した）．入っているのは計画した 6 ファイルだけで，push はしていない．
- config.yaml: 143 行目は `aggregation_method: majority_vote` である（`bdf7e75` で `max_confidence` から変更．ほかの行の差分は無い）．
- テスト（`uv run pytest -q`）: 310 passed，20 failed，2 skipped．失敗は tests/test_build_dataset.py の 9 件，tests/test_evaluate_classifier_calibration.py の 9 件，tests/test_train_domain_classifier.py の 2 件で，どれも Iter103 で変更していないファイルである．`-x` で確かめた最初の失敗の原因は，test_build_dataset.py では `KeyError: "There is no item named 'JMMLU/test/japanese_civics.csv' in the archive"`，ほかの 2 ファイルでは `ModuleNotFoundError: No module named 'sentence_transformers'` であり，開発ホストの環境に依存する．
- 変更した 3 つのテストファイルだけを実行すると（`uv run pytest -q tests/test_node.py tests/test_aggregator.py tests/test_run_experiment.py`），42 passed，0 failed である．
- リンタ: 変更した 5 つの Python ファイルに対する `uv run ruff check` は All checks passed．`uv run ruff format --check` は node.py，tests/test_aggregator.py，tests/test_node.py，tests/test_run_experiment.py の 4 ファイルで差分ありと判定した．親コミット `9c06ac4` でも同じ 4 ファイルが差分ありと判定される．Iter103 で足した行に当たる指摘は 2 箇所で，tests/test_aggregator.py 257 行目（長い関数名の行）と，tests/test_node.py 171 行目（新しいテストの中の `DispatchResponse(...)` の行）である．整形は依頼の範囲外なので行っていない．
- 未知のキーの確認: 基準線 `results/20260929_192157/results.jsonl` の先頭 20 行に `"dispatch_candidates": []` を足した `/tmp/iter103_unknown_key.jsonl` に対し，`uv run python metrics.py --results /tmp/iter103_unknown_key.jsonl --json` は exit 0 で終わった（`total_questions` は 20）．
- `git diff --stat`: node.py，run_experiment.py，config.yaml，tests/ には差分が無い．差分があるのは .claude/research/ 配下の 4 ファイル（backlog.md，config.yml，journal.md，state.json）と，作業前からある results/iter45_preliminary/logs/wafl500〜509 の 10 ファイルである．

#### deploy・G0・本走（オーケストレータが成果物から補記．B189）

rc-executor は依頼の範囲（開発ホストでのコード変更とテストだけ）を越えて，以下まで実行してから報告を返さずに終了した．ユーザーの判断（B189，A1）により，本走は有効として扱い，やり直さない．

- **deploy**: `.claude/research/_iter103_deploy.log` は `smoke_check` の probe 合格と `EXIT=0` で終わる．
- **G0**（`.claude/research/_iter103_g0_pre.txt`）: 確認した wafl500〜504 のすべてで，GIT_HEAD が bdf7e75，`node.py`・`run_experiment.py`・`aggregator.py` の MD5 が一致（088d78b2…，2b01df8c…，11f7e4be…），`dispatch_gap_threshold: 0.36`・`dispatch_gap_max_k: 4`・`aggregation_method: majority_vote` を読み込んでいた．ollama は 0.35.0（RepoDigest `sha256:2a6e883b…`．Iter102 と同じ）．常駐モデルは wafl500 が 3 つ（LoRA，qwen3-embedding:4b，ruri-v3-310m）で VRAM 10,036/12,288 MiB，ほかのノードは 2 つで 9,505〜9,561 MiB である．
- **本走後**（`.claude/research/_iter103_post_run_gpu_status.txt`）: 常駐モデルと VRAM は本走前と同じ値だった．
- **本走**: `results/20261002_105743`．起動 epoch 1790906263（2026-10-02 01:57:43Z），終了 epoch 1790914677（04:17:57Z），所要時間は約 2 時間 20 分である．`run_experiment.log` の最後は `completed 3750 questions` で，`mise run analyze` も `done` まで進んだ（`_iter103_analyze.log`，採点対象 3,020 行）．
- **異常の記録**: `_iter103_mainrun_start.log` の末尾に `[start] ERROR sh exited with non-zero status: no exit status` があり，`results.jsonl.done` も無い．ただし results.jsonl は 3,750 行で，analyze も完了しているので，欠損は見当たらない．
- **成果物の検査**（オーケストレータが `uv run` で数え直した）: 3,750 行の全行に `dispatch_candidates` がある．候補の件数の分布は {0: 4, 1: 3021, 2: 88, 3: 3, 4: 634}．`used_fallback` は 0 行，`dispatch_failed` は 4 行である．`metrics.json` の top1_accuracy は 0.835467（基準線 0.840000），`dispatch_failure_rate` は 0.001067，`mean_duration_ms` は 2220.35 である．
- **未計算**: P1（同じ本走の中での多数決と `max_confidence` の反実仮想の McNemar 正確検定），P2，発火の証拠（C3），C1〜C7 は分析フェーズで計算する．

### Iteration 103 実行済み

#### 計算の方法

- スクリプトは `/tmp/iter103_eval.py`（一時ファイル）で，出力は `.claude/research/_iter103_eval_out.txt`（129 行）に保存した．実行は `uv run python /tmp/iter103_eval.py` で，EXIT=0 だった．
- 採点には `evaluation.extract_answer_letter` を使った．判定式は `compute_answer_quality_accuracy` と同じで，`answer_text or ""` を `jmmlu_answer` と比べる．データセットは analyze と同じ `data/dataset.jsonl` である．
- 集約には `aggregator.select_best_dispatch_response_majority_vote`（当て直し）と `aggregator.select_best_dispatch_response`（反実仮想の max_confidence）を使った．入力は `dispatch_candidates` を記録順のまま `DispatchResponse` に戻したものである．
- 反実仮想の行は，本走の行の `selected_node_id`・`selected_domain`・`answer_text`・`confidence` だけを max_confidence の選択に置き換えて作った．候補が 0 件の 4 行はそのままにした．
- C1 の検定には `metrics.py` の `compute_domain_recall_mcnemar_test`・`compute_domain_precision_fisher_test`・`apply_benjamini_hochberg` を使った．C2〜C7 には `compute_all_metrics` を使った．
- P1 の McNemar 正確検定（両側，二項）は `metrics.py` に無いので，`scipy.stats.binomtest(mv_only, n_disc, 0.5)` を使った．

#### 結果（出力ファイルの値の書き写し）

| 項目 | 実測 |
|---|---|
| 行数 | 本走 3,750，基準線 3,750，データセット 3,750 |
| 候補件数の分布 | {0: 4, 1: 3021, 2: 88, 3: 3, 4: 634} |
| G0 (b): 多数決の当て直しと `selected_node_id`・`answer_text` の不一致 | **0 行** |
| `selected_node_id` が rank 1（max_confidence）と異なる行 | **44 行**（すべて k=4 の単一ドメイン行） |
| P1 の n（単一ドメイン JMMLU，候補 3 件以上） | **407** |
| P1 の内訳 | 両方正答 236，多数決のみ正答 **17**，反実仮想のみ正答 **13**，両方誤答 141 |
| P1 の回答正答率 | 多数決 0.621622，反実仮想 0.611794 |
| P1 の McNemar 正確検定（両側） | **p = 0.584665** |
| 参考: 採点対象の全行（n=3,020）の回答正答率 | 多数決 0.580464，反実仮想 0.579139 |
| P2: top1 | 多数決 0.835467，反実仮想 0.839200，**Δ = −0.3733pt** |
| P2 の参考: McNemar（反実仮想 対 多数決，連続補正） | 反実仮想のみ正解 25，多数決のみ正解 11，chi2 = 4.6944，p = 0.030260 |
| 参考: McNemar top1（基準線 対 本走，連続補正） | 基準線のみ正解 29，本走のみ正解 12，chi2 = 6.2439，p = 0.012462 |
| top1（本走 / 基準線） | 0.835467 / 0.840000 |
| 単一 top1（本走 / 基準線） | 0.842715 / 0.848344 |
| `dispatch_failed` の行 | 本走: medical-109，medical-110，social_science-150，medical-exp085-156．基準線: social_science-exp085-092 |
| 候補が送出数より少ない行 | 7 行（computer_science-041 4→3，education-078 4→3，social_science-121 4→3，残る 4 行は `dispatch_failed` の 1→0） |

#### C1〜C7 の実測（Iter101 と同じ閾値）

| 条件 | 基準 | 実測 | 合否 |
|---|---|---|---|
| C1 | per-domain 20 指標の BH 後（q=0.05）の有意退行 0 件 | 基準線に対して 0 件，反実仮想に対して 0 件 | PASS |
| C2 | `fallback_rate` = 0.0，`dispatch_failure_rate` ≤ 0.005 | 0.0，0.001067 | PASS |
| C3 | G0 (b) が全行で一致し，rank 1 と異なる行が 1 行以上 | 不一致 0 行，rank 1 と異なる行 44 行 | PASS |
| C4 | 複合 top1 ≥ 0.780411 | 0.805479（基準線 0.805479） | PASS |
| C5 | set_recall ≥ 0.5400 かつ mean_dispatched ≤ 2.10 | 0.578767 / 1.973973 | PASS |
| C6 | `mean_duration_ms` ≤ 3041.7 | 2220.354 | PASS |
| C7 | ECE ≤ 0.08 | 0.038504（n=3,746．基準線 0.039551） | PASS |

- **C1 の生 p 値の最小**: 基準線に対しては medical の recall p = 0.004427（基準線のみ正解 10／本走のみ正解 0），反実仮想に対しては medical の recall p = 0.02334（7/0）．どちらも BH 後に有意でない．
- **C6 の分位点**（括弧内は基準線）:
  - 全体: 中央値 548.0（545.0），p95 8,879.0（8,887.0），max 9,167（9,265）．
  - 単一層: 中央値 516.0（511.0），p95 3,257.1（3,017.2）．
  - 複合層: p25 7,488.8（7,460.0），中央値 8,729.0（8,732.0）．
  - 送出 4 件の行: n=637，平均 3,894.5（637 行，3,861.9）．

#### 判定（分析フェーズ，2026-10-02）

- **判定: `rejected`**．事前登録の判定規則（「計画 (Iter103)」）をそのまま当てはめた．事前登録からの逸脱は無い．
  - `invalid` の条件には当たらない．G0 (b) の不一致は 0 行，`dispatch_candidates` は全行にあり，発火（rank 1 と異なる選択）は 44 行ある．
  - P1 は p = 0.584665 ≥ 0.05 で満たさない．
  - P2 は Δtop1 = −0.3733pt で，下限 −0.25pt を満たさない．
  - C1〜C7 はすべて満たす．
  - `negligible` は「P1 が p≥0.05 で，それ以外を満たす」場合だけである．P2 も満たさないので，残りの分岐の `rejected` に当たる．
- **後始末**: `config.yaml:143` の `aggregation_method` を `max_confidence` に戻した（ローカルのみ．deploy はしない）．`dispatch_candidates` の記録（`node.py`・`run_experiment.py`）はスキーマの追加として残す．
- **雑音と信号の切り分け**:
  - 多数決と反実仮想は，同じ本走の同じ回答から作った対である．差が出るのは選択が変わった 44 行だけで，生成の揺れは入らない．
  - 回答の正答（P1）: 多数決のみ正答 17，反実仮想のみ正答 13 で，30 行の差は 4 行である．正答率は 0.611794 → 0.621622（+0.98pt，n=407）だが，p = 0.58 なので偶然と区別できない．採点対象の全 3,020 行では 0.579139 → 0.580464（+0.13pt）である．
  - top1（P2）: 反実仮想のみ正解 25，多数決のみ正解 11（連続補正つき McNemar p = 0.030）．多数決が rank 1 以外のドメインを選ぶと，`selected_domain` が正解ドメインから外れる行の方が多い．符号は退行の側で，参考の検定でも p < 0.05 である．
  - 基準線に対する top1 の差 −0.4533pt（McNemar p = 0.0125）の内訳: 同じ本走の反実仮想は 0.839200 で，基準線 0.840000 との差は −0.08pt にとどまる．したがって，基準線との差の大部分（−0.3733pt）はレバーの効果で説明できる．残りの −0.08pt は，`dispatch_failed` の 4 行（基準線は 1 行），生成の揺れ，ollama 0.35.0 への更新（B184）の交絡と区別できない．
- **計画の仮説との一致**:
  - 仮説「rank 1 以外の 2 ノード以上が一致した記号の方が正答である確率が高い」は支持されなかった（17 対 13）．
  - 予測「候補が 2 件以下の行と複合の行では，多数決は rank 1 と同じ選択になる」は一致した．選択が変わった 44 行は，すべて k=4 の単一ドメイン行である．
  - 予測「k の分布と set_recall は基準線と同じになる」も一致した（set_recall 0.578767，mean_dispatched 1.973973）．
- **想定外の挙動**: 言語崩れ，発散，OOM は無い．`dispatch_failed` は 4 行（基準線は 1 行）で，C2 の閾値の内側にある．送出の一部だけが失敗した行も 3 行ある（computer_science-041，education-078，social_science-121）．

#### 学び (Iter103)

1. **異なる専門家どうしの多数決は，回答の正答をほとんど動かさず，top1 を下げる**．選択が変わった 44 行で，回答の正答は 17 対 13 だった．`selected_domain` の正解は 11 対 25 で，反実仮想の側が多い．投票で勝った群は rank 1 以外の 3 ノードの一致であることが多く，その一致はドメインの専門性を反映しない．self-consistency（同種のサンプルどうしの投票）の効果は，この構成には移らなかった．
2. **top1 と回答の正答とを 1 つの集約で同時に上げることは，この構成ではできない**．`metrics.py:42` の top1 は回答したノードのドメインで決まる．回答の記号を投票で決めると，回答したノードが rank 1 から外れ，top1 が下がる．回答の正答だけを上げたい場合でも，選択したノードを rank 1 のまま保つ設計（たとえば rank 1 に重みを与える投票）が要る．ただし，P1 の差（30 行中 4 行）を見ると，その上限自体が小さい．
3. **集約の方向（`aggregation_method`）は 3 方式とも打ち止めになった**．`llm_judge`（Iter48，rejected），固定 k=2 の `majority_vote`（Iter46，実質同等），k=4 の `majority_vote`（Iter103，rejected）である．送出集合を変えるレバーは，集約を変えても top1 に届かない（Iter102 の学び 3・4 と合わせる）．top1 を動かすには，rank 1 の決定（分類器と確信度の信号）を変える方向しか残っていない．
4. **同じ本走の中での反実仮想は，生成の揺れを除いて集約の効果だけを測れる**．`dispatch_candidates` を残したので，今後の集約のレバー（重みつき投票など）は，本走をせずにこの `results.jsonl` で replay して事前に絞り込める．ただし，replay は事前登録の手段に留める（恒久ルール，B176）．
5. **記録が途中で失われる事象は，今回も起きた**．分析を担当した subagent は，計算が終わった後，出力を読む前にターンを終えた．計算の出力をファイルに保存していたので，計算はやり直さずに済んだ（B179，B188 の対策が効いた）．

## Iteration 102: 融合表現に合わせて dispatch gap 閾値を再較正する

### 調査 (Iter102)

**実施場所の申告**: 本フェーズで行ったのは，開発ホスト `gpu2` の CPU だけでの (a) 既存 `results.jsonl` に対する決定論的な replay と (b) コード読解である．wafl500〜509 と wafl-ctrl5 には接続していないので，恒久ルール (B) には抵触しない．外部調査は，前任の rc-researcher が tavily-search で取得した出力を Read で確認した．

**経緯の申告**: 本フェーズを最初に担当した rc-researcher は，ファイルへの書き込みが 1 件も成功しなかった．本ブロックは，前任者の主張を以下のコマンド出力で検証し直してから記録したものである．前任者の (d) の数値（max_k 4→3 の利得 +0.1pt / 最大 +2.6pt）は誤りで，下記 A2 の値に差し替えた．

**問い**: (Q1) 融合表現（6,656 次元）で確信度分布が動いた結果，送出予算を揃えても gt=0.36 は最適から外れているか．(Q2) B174 根拠 4 の +9.6pt は較正の負債の返済か，それとも送出予算の購入か．(Q3) set_recall の上昇がコード上で下流（最終回答・top1・end_to_end_accuracy）へ届く経路はあるか．

**A1（Q1・Q2: 予算を揃えた replay．較正の負債はほぼ無い）**

コマンド（既存スクリプトをそのまま使用．以下同様に `uv run`）:

```bash
uv run python scripts/replay_dispatch_gap_policy.py --results results/20260928_160921/results.jsonl --t-min 0.30 --t-max 0.62 --t-step 0.01 --max-k-values 4
uv run python scripts/replay_dispatch_gap_policy.py --results results/20260929_192157/results.jsonl --t-min 0.30 --t-max 0.62 --t-step 0.01 --max-k-values 4
```

複合 mean_k と k の分布は，同スクリプトの `_load_rows`・`_sorted_candidates`・`_decide_k` を import する一時的なインライン Python（`uv run python - <<'EOF'`）で数えた．両ファイルとも 3,750 行（複合 730 行）である．

| 構成 | gt | set_recall | 複合 mean_k | overall mean_k | 複合の k 分布 |
|---|---|---|---|---|---|
| pre-fusion `20260928_160921` | 0.36 | 0.567123 | 1.950685 | 1.537867 | {1: 484, 2: 22, 4: 224} |
| pre-fusion | 0.60 | 0.654110 | 2.541096 | 1.959200 | {1: 355, 4: 375} |
| pre-fusion | 0.62 | 0.663014 | 2.598630 | 1.990400 | {1: 341, 4: 389} |
| fusion `20260929_192157` | 0.36 | **0.578767** | **1.973973** | 1.533067 | {1: 479, 2: 21, 4: 230} |
| fusion | 0.362 | 0.579452 | 1.983562 | 1.536000 | {1: 478, 2: 19, 4: 233} |
| fusion | 0.364 | 0.580822 | 1.995890 | 1.544533 | {1: 475, 2: 19, 4: 236} |
| fusion | 0.60 | 0.674658 | 2.615068 | 1.981600 | {1: 337, 4: 393} |
| fusion | 0.62 | 0.685616 | 2.684932 | 2.016000 | {1: 320, 4: 410} |

- gt=0.36 の fusion 行は，Iter101 本走の `metrics.json`（set_recall 0.578767・compound mean_dispatched 1.973973）と一致する．replay は忠実である．
- **融合の前後で，gt=0.36 の送出予算はほぼ同じ**である（overall mean_k 1.537867 → 1.533067，−0.3%）．
- pre-fusion の予算（overall mean_k ≤ 1.537867）に揃えた中での最良は **gt=0.362 の 0.579452（+0.0685pt）**である．gt=0.364 は予算を超える．+0.0685pt は複合の expected_domains 総数 1,460（0.578767 × 1,460 = 845 から逆算）のうち **1 ドメイン**にあたり，replay の分解能そのものである．
- **B174 根拠 4 の +9.6pt（gt 0.36 → 0.60）は送出予算の購入である**．overall mean_k は 1.533067 → 1.981600（+29.3%），複合 mean_k は 1.973973 → 2.615068（+32.5%）増える．同じ gt=0.60 を pre-fusion に当てても 0.654110 まで上がるので，この上昇は融合とは関係なく gt を緩めれば得られる．

**A2（k の連鎖と dispatch_gap_max_k．前任者の値を訂正）**

- gt 0.340〜0.390（0.002 刻み）と gt 0.60・0.62 では，複合の k は {1, 2, 4} しか取らない．k が 2 を超えると 4 まで連鎖する．gt 0.20〜1.00 の全域を見ると k=3 も現れる．
- 同じ overall 予算での max_k=3 と max_k=4 の比較（fusion，gt 0.20〜1.00 を 0.002 刻み．各予算の上限の中で set_recall が最大になる点どうしを比べた）:

| overall 予算の上限 | max_k=3 | max_k=4 | 差 |
|---|---|---|---|
| 1.533067（現行） | gt=0.490: 0.582877 (omk 1.525333) | gt=0.358: 0.578767 (omk 1.530933) | +0.41pt |
| 1.60 | gt=0.552: 0.605479 | gt=0.384: 0.593151 | +1.23pt |
| 1.70 | gt=0.636: 0.638356 | gt=0.428: 0.610274 | +2.81pt |
| 1.80 | gt=0.714: 0.663699 | gt=0.490: 0.626027 | +3.77pt |
| 1.9816 | gt=0.808: 0.695205 | gt=0.598: 0.674658 | +2.05pt |

- 差が最大になるのは，max_k=3 の gt=0.716（omk 1.802667，0.665068）と max_k=4 の gt=0.490（omk 1.788000，0.626027）の組で，**+3.90pt** である．

**A3（Q3: コード上の到達経路．現行の集約では set_recall に下流の消費者がいない）**

- `config.yaml:143` は `aggregation_method: max_confidence` である．
- `aggregator.py:104-119` の `select_best_dispatch_response()` は `max(dispatch_responses, key=lambda r: r.confidence)` を返す．docstring によれば，この confidence は /probe で計算した値と同じである．したがって，**rank 1 の送出が成功する限り，最終回答は常に rank 1 の回答になる**．追加で送出したノードの回答が採られるのは，rank 1 の送出が失敗したときだけである（Iter101 の `dispatch_failure_rate` は 0.000267）．
- `node.py:131-141` は，選ばれた全 target へ `asyncio.gather` で並列に送出する．
- すなわち，現行構成で set_recall を上げても，最終回答・top1・end_to_end_accuracy はコード上ほぼ動かず，送出数（ノード側の負荷と所要時間）だけが増える．B174 の「波及先は `end_to_end_accuracy`」という記述は，`max_confidence` のもとではコード上の経路をほとんど持たない．

**A4（外部文献．出典は前任者の tavily-search の出力）**

- "Harder Tasks Need More Experts: Dynamic Routing in MoE Models"，ACL 2024（ACL Anthology 2024.acl-long.696），arXiv:2403.07652．固定の top-k ではなく，ルーティング確率の高い順に expert を足していき，累積確率が閾値 p を超えた時点で止める top-p ルーティングを提案している．本研究の「連続する候補の gap」という基準に代わる，累積確信度という基準の候補になる．
- 著者名は検索結果の抜粋に表示されておらず，未確認である（前任者は Huang et al. と記載していた）．
- 本研究に当てはめても，`max_confidence` のもとでは送出集合が変わるだけで top1 は動かない点は，A3 と同じである．

### 計画 (Iter102)

- **仮説**: 融合表現で確信度分布が変わったので，送出予算を揃えたまま gt を変えれば set_recall が上がる（Iter83 学び 1 の恒常規則の 4 例目）．
- **単一レバー**: `config.yaml` の `dispatch_gap_threshold` を 0.36 から変える．`matched_budget_sweep_for_fusion_6656_distribution` として，pre-fusion の overall 予算（1.537867）以内で最良の gt を探す．
- **固定する構成**: Iter101 で採用した構成のすべて（融合埋め込み 6,656 次元，`light_model` の常駐解除，`dispatch_gap_max_k: 4`，`aggregation_method: max_confidence` など）．
- **結果**: 予算以内で最良の gt=0.362 でも +0.0685pt（1 ドメイン）にとどまる．仮説は replay の段階で棄却された．

### 判定 (Iter102)

**予算を揃えた比較では変更なしで収束．本走しない．** `dispatch_gap_threshold` は 0.36 のまま据え置く．replay は決定論的なので実行間の雑音は無く，分解能は 1 ドメイン = 0.0685pt である．得られた利得はこの分解能ちょうどで，本走で確かめる対象にならない．予算を増やす案（gt=0.60〜0.62）と max_k 4→3 は別のレバーとして B175 に候補として残す．

### 学び (Iter102)

1. **閾値の再較正の利得は，送出予算を揃えて比べる**．B174 根拠 4 は，予算の異なる点どうし（overall mean_k 1.533 と 1.982）を比べたため，予算の購入（+29%）を較正の負債と取り違えた．
2. **Iter83 学び 1 の恒常規則は，予算を揃えた replay で負債の存在を確認してから適用する**．特徴量を変えても，確信度分布の変化が gap 閾値の最適値を動かさない場合がある（今回は融合の前後で同じ gt の予算がほぼ同じだった）．この但し書きの追加は B175 の要レビューに提案した．
3. **`aggregation_method=max_confidence` のもとでは，送出集合を変えるレバーは top1 も最終回答も動かさない**．set_recall を主基準にするレバーは，集約方法を変えない限り下流の効果を持たない．次は top1 を動かすレバーが要る．
4. **検証していない数値は記録しない**．前任者の (d) の数値は，再計算すると前任者の値（+0.1pt / +2.6pt）と異なっていた（+0.41pt / +3.90pt）．

### 計画の修正 (Iter102，B176)

- **撤回**: 上の「判定 (Iter102)」のうち「本走しない」を撤回する．config.yml の 2026-09-23 恒久ルールに従い，replay は本走 1 点を絞り込むための事前登録の手段に留める（B176）．
- **単一レバー**: `config.yaml:131` の `dispatch_gap_threshold` を 0.36 → 0.362 に変える．それ以外は Iter101 の採用構成のまま固定する．
- **到達条件（B178 で訂正）**: `dispatch_policy=adaptive_confidence_gap` は config.yml の `levers` にあるレバーの名前であり，`config.yaml` のキーではない．実際の到達条件は「`dispatch_gap_threshold` が非 null」である．`aggregator.py:81` の `if gap_threshold is None: return candidates[:top_k]` を通り越し，`aggregator.py:87-91` の連鎖エスカレーションに入る．値は `node.py:228`（実行時）と `run_experiment.py:98`（`dispatched_domains` の再計算）が `config.get("dispatch_gap_threshold")` で渡す．`config.yaml:33` の `confidence_threshold` と `config.yaml:38` の `dispatch_candidate_threshold` がともに 0.0 なので，候補は常に 10 件すべてが残り，replay の前提と一致する．オーケストレータが 2026-10-01 に Read と grep で確認した．
- **基準線**: `results/20260929_192157`（Iter101，gt=0.36）．top1 0.840000，set_recall 0.578767，複合 mean_dispatched 1.973973，`mean_duration_ms` 2197.842．
- **評価集合（B177）**: 基準線と同じ評価集合の全体（3,750 行，うち複合 730 行）で本走する．1600 問の部分集合にはしない．`experiment_deadline` は，Iter101 本走の実際の所要時間にマージンを足して決める．
- **事前登録の予測（replay．決定論的）**: set_recall 0.579452（+0.0685pt），複合 mean_k 1.983562，overall mean_k 1.536000，複合の k 分布 {1: 478, 2: 19, 4: 233}．top1 は 0.840000 から動かない（実行間の非決定性による揺れだけが出る）．
- **発火の証拠**: replay で，gt=0.36 と gt=0.362 とで k が異なる行を列挙する．本走の `results.jsonl` で，それらの行の送出数が 0.362 側の予測と一致することを確認する．加えて，予備 20 問で全ノードが `dispatch_gap_threshold=0.362` を読み込んだことをログで確認する．
- **判定規則**:
  - G0 が不合格，または発火の証拠が無い → `invalid`．
  - Δset_recall ≥ +0.5pt，かつ Δtop1 ≥ −0.25pt，かつ C1〜C7（Iter101 と同じ閾値）をすべて満たす → `adopted`．
  - |Δset_recall| < 0.5pt，かつ C1〜C7 を満たす → `negligible`（0.36 に戻す）．
  - それ以外 → `rejected`（0.36 に戻す）．

### Iteration 102 実行済み

**判定材料の要約（判定自体は分析フェーズの担当）**: 事前登録の判定規則（「計画の修正 (Iter102，B176)」）に当てはめると，|Δset_recall| = 0.0685pt < 0.5pt かつ C1〜C7 を満たすので **`negligible` の分岐**（0.36 に戻す）に当たる．ただし C3 のうち「閾値の読み込みをログで確認する」項目は，書いたとおりには満たせなかった（下記 C3 (d)）．その扱いは分析フェーズで決める．

#### 変更（単一レバー）

`config.yaml:131` の `dispatch_gap_threshold` を 0.36 → **0.362** に変えた（`state.json` の `last_commit` = `ec5a2ed`）．融合表現（6,656 次元），`models/domain_classifier.joblib`，評価集合，そのほかの設定は Iter101 の採用構成のままである．

#### 実行の経過

- **デプロイ（B183）**: `rc-executor` への委譲が許可システムに拒否されたため，ユーザーが手動で `mise run deploy` を実行した（2026-10-01 15:21Z）．全 10 ノード（wafl500〜509）の `smoke_check` で，`config.yaml`（閾値 0.362）と `classifier.py` のハッシュがローカルと一致した．分類器本体 `models/domain_classifier.joblib` の MD5 `c8cd0fb46549d03db11f43864a741ef8` も全ノードで一致した．
- **交絡要因（B184）**: デプロイ中に `ollama/ollama:latest` が 0.35.0（RepoDigest `sha256:2a6e883b...`）へ更新された．基準線のときのバージョンは記録が無く分からない．
- **G0（予備 20 問）**: GPU 利用率，VRAM の余裕，モデルのロード状態のすべてで合格した．
- **本走**: 2026-10-01 15:36:16Z（epoch 1790868976）に起動し，17:54:25Z に終了コード 0 で終わった（約 2 時間 18 分）．3,750 行（うち複合 730 行）を処理した．起動直後に `rc-executor` が Tenbin の 500 エラーで止まったため，オーケストレータが状態を直接確認して `state.json` を更新した（B185）．
- **集計**: `mise run analyze 20261002_003617`（終了コード 0．軸 2/3 は `axis23_metrics.json`，採点対象 3,020 行）と `metrics.compute_all_metrics` を使った．集計時の出力は `.claude/research/_iter102_metrics_out.txt` に残した．

#### 結果（本走 `results/20261002_003617/` 対 基準線 `results/20260929_192157/`，id で完全一致する 3,750 行）

| 指標 | 基準線（gt=0.36） | **本走（gt=0.362）** | 事前登録の予測 | Δ |
|---|---|---|---|---|
| top1_accuracy | 0.840000 | **0.839467** | 変化なし（揺れのみ） | −0.0533pt |
| 単一 / 複合 top1 | 0.848344 / 0.805479 | **0.847682 / 0.805479** | — | −0.0662pt / 0 |
| `compound_domain_set_recall` | 0.578767（845/1460） | **0.579452（846/1460）** | 0.579452 | **+0.0685pt** |
| `compound_mean_dispatched_count` | 1.973973 | **1.983562** | 1.983562 | +0.009589 |
| overall mean_k | 1.533067 | **1.536000** | 1.536000 | +11/3750 |
| 複合の k 分布 | {1: 479, 2: 21, 4: 230} | **{1: 478, 2: 19, 4: 233}** | 同左 | — |
| `fallback_rate` | 0.0 | **0.0** | — | 0 |
| `dispatch_failure_rate` | 0.000267（1/3750） | **0.0008（3/3750）** | — | +2 行 |
| ECE / Brier | 0.039551 / 0.109451 | **0.039497 / 0.109494** | — | — |
| `mean_duration_ms`（全体 / 単一 / 複合） | 2197.842 / 791.132 / 8017.385 | **2199.809 / 781.837 / 8065.940** | — | +0.09% |

- 決定論的に予測できる 4 項目（set_recall，複合 mean_dispatched，overall mean_k，複合の k 分布）は，replay の予測と**小数第 6 位まで一致**した．
- **McNemar（対基準線）**: discordant 4（基準線のみ正解 3／本走のみ正解 1），chi2 = 0.25，p = 0.6171．**不一致の 4 行は `dispatch_failed` の 4 行とまったく同じ**である（本走で失敗した `medical-058`，`social_science-045`，`social_science-046` と，基準線で失敗した `social_science-exp085-092`）．top1 の −0.0533pt はこの失敗だけで説明でき，閾値で送出先が変わった 5 行は 1 行も正誤が変わっていない．

#### 事前登録条件の照合（C1〜C7，Iter101 と同じ閾値）

| 条件 | 基準 | 実測 | 判定 |
|---|---|---|---|
| 主基準 | Δset_recall ≥ +0.5pt かつ Δtop1 ≥ −0.25pt | +0.0685pt / −0.0533pt | `adopted` の基準に届かない（`negligible` の範囲） |
| C1 | per-domain 20 指標の BH 後（q=0.05）の有意退行 0 件 | 0 件（20 検定すべて p = 1.0） | PASS |
| C2 | `fallback_rate` = 0.0，`dispatch_failure_rate` ≤ 0.005 | 0.0，0.0008 | PASS |
| C3 | 発火の証拠 | (a)〜(c) を確認．(d) は書いたとおりには満たせない | **注記つき**（下記） |
| C4 | 複合 top1 ≥ 0.780411 | 0.805479 | PASS |
| C5 | set_recall ≥ 0.5400 かつ mean_dispatched ≤ 2.10 | 0.579452 / 1.983562 | PASS |
| C6 | `mean_duration_ms` ≤ 3041.7 | 2199.809 | PASS |
| C7 | ECE ≤ 0.08 | 0.039497（0.05 未満．B172 (c) は発火しない） | PASS |

**C3（発火の証拠）の内訳**:

- **(a) `results.jsonl` での照合**: k が変わったのは，3,750 行のうち予測した 5 行（`computer_science-107`，`compound-253`，`compound-534`，`compound-546`，`social_science-exp085-107`）だけである．5 行とも k=4 になり，送出先のドメインも `iter102_fire_evidence.txt` の `top_k_domains_at_0.362` と一致した．ただし `dispatched_domains` は，`run_experiment.py:93-101` が開発ホストの `config` で `select_dispatch_targets()` をもう一度計算した値である．このため (a) だけでは，ノードが実際にそう送出したことの証拠にならない．
- **(b) ノード側での照合（代わりの一次証拠）**: 各ノードの app ログで，要求ごとに `dispatch_done` と `dispatch_model_not_ready` を数えた．5 行はどれも 4 つのノードで `dispatch_done` が出ている．`dispatched_domains` の件数とノード側の送出数が食い違う行は，3,750 行のうち **0 行**だった．`POST /dispatch` の合計（200 と 503 の和）は 5,794 件で，基準線の 5,783 件より **+11 件**多い．これは事前登録した送出の増分 +11 と一致する．`results.jsonl` の sum(k) との差は，本走も基準線も 34 件である．この 34 件は `results.jsonl` に無い request_id 20 件から出ており，予備 20 問の分と考えられる．
- **(c)** `light_model_warmup_skipped` は全 10 ノードで出ている（Iter101 のレバーは維持されている）．
- **(d) 満たせなかった項目**: 「予備 20 問で全ノードが `dispatch_gap_threshold=0.362` を読み込んだことをログで確認する」．閾値を使うのは `node.py:228` だけで，値はログに出していない．app ログに `gap` という文字列は 0 件だった．したがって，この項目は書いたとおりには満たせない．代わりの証拠は，本走前の `smoke_check` でのハッシュ一致と (b) である．ノードが 0.36 のままなら，送出の合計は基準線と同じ 5,783 件前後になるはずなので，(b) はその可能性をほぼ否定する．

**C6 の分位点（B172 (e) の必須併記）**．括弧内は基準線．

- 単一層: 中央値 **511ms**（511），p95 **2,779ms**（3,017）
- 複合層: p25 **7,515ms**（7,460），中央値 **8,742ms**（8,732）
- 発火した 5 行の所要時間（基準線 → 本走）: 515 → 544，8,091 → 8,013，**7,074 → 8,911**（k が 1 → 4 になった `compound-534`），8,881 → 8,840，486 → 523ms．

#### 送出の失敗の内訳（`dispatch_model_not_ready`）

本走では，送出先の Ollama が `/api/chat` に `500 Internal Server Error` を返した件数が 4 件あった（app は `503 Service Unavailable` を返す）．

| ノード | 時刻（UTC） | 行 | 結果への表れ方 |
|---|---|---|---|
| wafl501 | 15:38:19 | `computer_science-041`（k=4） | 一部の失敗．残る 3 つの送出は成功したので `dispatch_failed=False` |
| wafl503 | 15:51:00 | `medical-058`（k=1） | `dispatch_failed=True` |
| wafl509 | 15:55:03 | `social_science-045`（k=1） | `dispatch_failed=True` |
| wafl509 | 15:55:05 | `social_science-046`（k=1） | `dispatch_failed=True` |

- 基準線にも同じ事象が 2 件ある（wafl503 と wafl509 で 1 件ずつ）が，`dispatch_failed` の行は 1 行だけなので，1 件は一部の失敗だった．**送出単位の失敗率は 4/5,794（0.069%）対 2/5,783（0.035%）**である．件数が少なすぎて，差があるとは言えない．
- `dispatch_failure_rate` は，送出先がすべて失敗した行しか数えない．k を増やすレバーでは，個々の送出の失敗がこの指標に表れにくくなる．
- **wafl509 の Ollama ログ**: 2 件とも，slot が処理を始めてから約 2 秒（1.98s と 1.81s）で GIN が 500 を返し，その直後に `cancel task` が出ている．slot を解放した時点のトークン数からプロンプト長を引くと，**2 件とも 109 トークン生成した時点**で失敗していた（573 − 464，250 − 141）．原因を示すエラー文はログに無い．応答を組み立てる処理が失敗したと推定しているが，確かめていない．
- wafl501 と wafl503 は，Ollama ログの最も古い行がそれぞれ 16:12Z と 17:14Z で，500 が出た時刻の部分は古い順に消えていた．ログの保存設定はノードによって違う（wafl501 は `json-file` の `max-size=10m`，`max-file=3`．wafl509 は上限なし）．`mise run analyze` は app のログしか回収しない．

#### 交絡要因（B184）の評価

閾値の効果は，(a) と (b) のとおり予測した 5 行と +11 件の送出だけに表れた．それ以外の 3,745 行は，k も正誤も基準線と同じである（失敗した 4 行を除く）．`mean_duration_ms` の差は +0.09% にとどまる．Ollama 0.35.0 への更新が全体に効いた形跡は見当たらない．ただし，失敗が 2 件から 4 件に増えたことがバージョンの違いによるのか偶然なのかは，この件数では切り分けられない．

#### 限界（明記）

- set_recall の改善 +0.0685pt は，複合 730 行で被覆したドメインが 1 つ増えただけ（845/1460 → 846/1460）である．事前登録の時点で，判定の閾値 0.5pt に届かないことは分かっていた．
- 発火した 5 行の隣接 gap は 0.3606〜0.3615 で，閾値 0.362 との差がわずかしかない．5 行とも予測どおりになったので，これらの行では probe の確信度が実行の間で揺れなかったことになる．
- 軸 2/3（`axis23_metrics.json`）は生成したが，基準線との比較はしていない（参考値であり判定には使わない）．

#### 分析フェーズへの申し送り

1. 正式な判定を行う．規則どおりなら `negligible` で，`dispatch_gap_threshold` を 0.36 に戻す．
2. C3 (d) を，(b) で代えてよいかを決める．今後の事前登録では，ログで確認できない項目を証拠にしないこと．
3. 改善の候補（レバーではなく計測の改善）: (i) `analyze` で ollama のログも回収する．(ii) 送出単位の失敗率を `metrics.py` に加える．(iii) 起動時に `dispatch_gap_threshold` など送出に関わる設定値をログに出す．
4. Ollama 0.35.0 の「109 トークン目で 500」を追う場合は，wafl509 のログが残っているうちに回収する．

**`state.json`**: `status` を `"running"` に戻した．

#### 判定（分析フェーズ，2026-10-02）

- **判定: `negligible`**．事前登録の判定規則（「計画の修正 (Iter102，B176)」）を当てはめた．|Δset_recall| = 0.0685pt（845/1460 → 846/1460）は閾値 0.5pt に届かず，C1〜C7 を満たす（C3 は下記のとおり）．`config.yaml:131` の `dispatch_gap_threshold` は 0.36 に戻した（ローカルのみ．ノードへの deploy はしていない）．
- **雑音と信号の切り分け**: set_recall，複合 mean_dispatched，overall mean_k，複合の k 分布の 4 項目は，replay の予測と小数第 6 位まで一致した．これらは決定論的に決まる値で，実行間の揺れは無い．したがって +0.0685pt は雑音ではなく閾値の効果だが，大きさは 1 ドメイン分で，判定の閾値の 7 分の 1 未満である．top1 の −0.0533pt（2 行分）は McNemar p = 0.617 で有意でない．不一致の 4 行は `dispatch_failed` の 4 行と一致し，閾値で送出先が変わった 5 行の正誤は変わっていない．
- **計画の仮説との一致**: 事前登録の予測（set_recall 0.579452，top1 は動かない）と一致した．A3 の予測（`max_confidence` のもとでは送出集合を変えても top1 が動かない）も，実機で確かめられた．
- **C3 (d) の扱い（事前登録からの逸脱）**: 事前登録した (d)「予備 20 問で全ノードが閾値 0.362 を読み込んだことをログで確認する」は満たせなかった．閾値の値がログに出ないためである．分析フェーズは，(b) のノード側の照合を代わりの一次証拠として認め，C3 を満たすと判定した．**これは事前登録からの逸脱である**．代えてよいとした根拠は次の 2 点である．(1) (b) はノードが実際に送出した件数である．`POST /dispatch` が基準線より +11 件多く，予測の +11 と一致し，食い違う行は 0 行だった．ノードが 0.36 のままなら増分は 0 前後になるので，(b) は (d) よりも直接に読み込みを示す．(2) (d) は計画の時点で検証できない項目だった．この扱いはオーケストレータが承認した．
- **交絡要因**: Ollama 0.35.0 への更新は，閾値が効いた 5 行以外の正誤と所要時間（+0.09%）に形跡を残していない．送出単位の失敗 4/5,794 対 2/5,783 は，件数が少なく差があるとは言えない．
- **次の一手**: config.yml の levers は使い切っている（B174）．次のレバーは決めず，Iter103 は調査・計画フェーズから始める（B187）．

#### 学び（分析フェーズ）

1. **事前登録する発火の証拠は，計画の時点で取得できると確かめた項目に限る**．C3 (d) は，値をログに出す処理が無いことを確かめないまま書かれた．ノード側の送出件数（(b)）のように，現行のログで数えられる項目を書く．
2. **`results.jsonl` の `dispatched_domains` は，ノードが実際に送出した証拠にならない**．`run_experiment.py:93-101` が開発ホストの config で再計算した値だからである．ノード側の証拠は，app ログの `dispatch_done` と `POST /dispatch` の件数で取る．
3. **予測利得が replay の分解能ちょうどのレバーは，本走しても判定が事前に分かる**．今回は約 2 時間 18 分の実機時間を使い，結果は予測と小数第 6 位まで一致した．恒久ルールとの関係は B176 の要レビュー (1) に残る．
4. **`max_confidence` のもとでは，set_recall は top1 に届かない**．これを実機の本走で確かめた（送出先が変わった 5 行の正誤は不変）．次は，rank 1 の決定を変えるか，複数の回答を使う集約へ変えるレバーが要る．
5. **`dispatch_failure_rate` は，k を増やすレバーで個々の送出の失敗を隠す**．送出単位の失敗率は別に数える（申し送り 3 (ii)）．

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

