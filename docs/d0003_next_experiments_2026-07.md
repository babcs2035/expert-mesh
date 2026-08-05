<!-- d0002 の知見を受けて，次に行うべき実装修正 (F1-F5) と実験 (X1-X9) を優先順位付きで定義する作業計画． -->

# d0003: 次に行うべき実装修正と実験（2026-07 時点）

> **【状態: 歴史的記録 — Iter22 時点の計画．大半が実施済み】**
> 研究は Iteration 54 で収束した．**収束時点の現況は `docs/d0006_research_summary_iter28-54_2026-08.md`
> を参照すること**．本書が「未実施」としている F1〜F5・X1・X2・X4 は，Iter23〜27 でその大半が
> 完了している．**本書を単独の根拠として計画を立ててはならない．**
> §1「着手前に理解すべき構造的制約」は現在も有効である．

## このファイルについて

- **目的**: `d0002_research_cycle_findings_2026-07.md` で明らかになった課題に対し，次に何を，どの順序で，どういう成功条件で行うかを定義する．
- **役割**: research-cycle の次イテレーション以降の計画の根拠．`.claude/research/config.yml` の `levers` を更新する際もここを参照する．
- **利用方法**: 先に d0002 を読むこと．特に「§1 着手前に理解すべき構造的制約」は，これを知らずに実験を組むと無効な実験を再生産するため必読である．
- **記号**: `F{n}` は実装修正（実験を伴わない），`X{n}` は実験を指す．

---

## 0. 実施順序の要約

```
[第 1 段階: 測定系の修復]  ── 実験を伴わない．これを終えるまで新規実験は無効になりうる
  F1  config.yaml を最良既知構成へ戻す
  F2  デプロイ検証ゲートの導入
  F3  metrics.py へ ECE / AUROC / Brier / 同点率 / ノード間分散 を統合
  F4  journal・backlog の記録訂正
  F5  再現性の担保（データ・成果物のハッシュ記録）

[第 2 段階: 基準線の再確立]
  X1  最良既知構成での基準線再取得   ← F1〜F3 に依存
  X6  回答品質のノイズ床の確定       ← X1 と同時に実施

[第 3 段階: 研究の主張を支える実験]
  X2  中央集権ルータとの比較         ← 最重要．論文の中心命題
  X4  複合ドメイン評価の立て直し
  X5  fallback 方針の見直し

[第 4 段階: 残レバーと較正]
  X3  E4（semantic entropy）の再設計
  X7  E5（p_true）
  X8  education / legal のデータ不均衡是正
  X9  分類器の較正
```

優先度の考え方: **第 1 段階を飛ばして第 3・第 4 段階に進んではならない．** 直近 3 イテレーション（Iter20・21・22）はいずれも測定系の問題により実質的に無効であり，同じ失敗を繰り返す危険が高い．

---

## 1. 着手前に理解すべき構造的制約

新しい実験を設計する前に，次の 3 点を必ず踏まえること．いずれも d0002 §6 で実データ・実コードから確認した事実である．

### 制約 1: `_estimate_probe_confidence()` は排他的な if 連鎖である

`http_server.py:301-388` の分岐は，先にマッチしたものが return する構造になっている（HEAD `30e3627` 時点の順序）．

```
313  confidence_signal_method == multi_sample          → return
324  confidence_signal_method == stp                   → return
334  confidence_signal_method == semantic_entropy      → return
345  confidence_signal_method == p_true                → return
354  routing_method == embedding                       → return
364  routing_method == supervised_classifier           → return
371  confidence_elicitation == top_k_with_probs        → return
380  default: self_report                              → return
```

帰結:

- `routing_method=supervised_classifier` のとき，`confidence_elicitation` と `embedding_postprocess` は**読まれない**（no-op）．
- `confidence_signal_method` に `self_report` 以外を設定すると，`routing_method` の分岐に**到達しない**．つまり分類器が無効化される．
- **`confidence_signal_method` と `routing_method` は「独立した設定」ではなく「排他的な代替」である．** 片方を変えるともう片方が効かなくなるため，「confidence 信号だけを差し替える」実験は現在の実装では組めない．

### 制約 2: ルーティング経路は完全に決定論的である

`nomic-embed-text` の埋め込みと joblib の LogisticRegression には確率的要素がない．Iter20/21/22 の 1520 問すべてで `confidence` がビット単位一致した．

帰結:

- 同一構成の再実行は**新しい情報を一切生まない**．run 間ノイズの推定を目的とした反復は無意味である．
- 逆に，同一構成で数値が変われば，それは実装かデプロイのどちらかが変わった証拠になる（F2 の検証ゲートはこの性質を利用する）．
- 一方，回答生成は確率的であり，軸②③には約 1.3pt のノイズ床がある（d0002 §6-F）．

### 制約 3: 軸①と軸②③の依存関係

`expert_model` を変えても `top1_accuracy` は変わらない（ルーティングは light_model + 分類器で決まる）．逆に `routing_method` を変えると軸②③も連動して変わる．実験を設計する際は，どちらの軸に効くレバーなのかを先に確定させること．

---

## 2. 第 1 段階: 測定系の修復

### F1. `config.yaml` を最良既知構成へ戻す

**背景**: Iter19 で棄却された `expert_model=qwen3.5:4b-q4_K_M` が HEAD まで残置されている．Iter18 で採用された domain LoRA に戻されていない（d0002 §6-C）．過去は棄却レバーを baseline に戻す運用だった（backlog B5・B8）．

**作業**:

1. `config.yaml` の 10 ノードの `expert_model` を `expert-mesh-{domain}-lora` に戻す．対応は次のとおり．

   | ノード | domain | expert_model |
   |---|---|---|
   | wafl500 | general | `expert-mesh-general-lora` |
   | wafl501 | education | `expert-mesh-education-lora` |
   | wafl502 | legal | `expert-mesh-legal-lora` |
   | wafl503 | medical | `expert-mesh-medical-lora` |
   | wafl504 | business_economics | `expert-mesh-business_economics-lora` |
   | wafl505 | computer_science | `expert-mesh-computer_science-lora` |
   | wafl506 | natural_science | `expert-mesh-natural_science-lora` |
   | wafl507 | mathematics | `expert-mesh-mathematics-lora` |
   | wafl508 | history_culture | `expert-mesh-history_culture-lora` |
   | wafl509 | social_science | `expert-mesh-social_science-lora` |

   正確な対応は `git show 709b1ee:config.yaml` で確認できる．

2. `confidence_signal_method` を `self_report` に戻す（制約 1 により，これを `self_consistency_semantic` のままにすると分類器が無効化される）．
3. `probe_timeout_s` は E4 用に 120.0 へ引き上げられているが，`self_report` + 分類器経路では LLM 呼び出しが発生しないため 60.0 に戻してよい．ただし戻すこと自体が別のレバー変更になるため，X1 の基準線取得では **120.0 のまま据え置く**ことを推奨する（タイムアウトが発火していない以上，値の違いは結果に影響しない）．
4. 76〜82 行のコメントと実際の値の整合を確認する．

**前提条件**: LoRA アダプタは `models/lora_adapters/` に 10 ドメイン分（`adapter.gguf`，`adapter_model.safetensors`，`adapter_config.json`）が存在することを確認済み．各ノードの Ollama に `expert-mesh-{domain}-lora` が登録されているかは要確認（`scripts/create_lora_model.py` で再登録できる）．

**コスト**: config 編集数分 + 必要ならモデル再登録．

---

### F2. デプロイ検証ゲートの導入

**背景**: 同種のデプロイ事故が 2 回起きている．

- Iter12: `mise run deploy` が Docker イメージを再ビルドせず，Python コード変更が反映されなかった．
- Iter22: 修正が working tree にしかなく，`mise run deploy` は git HEAD から配布するためコンテナに届かなかった．

いずれも約 2 時間の実験が丸ごと無効になった．どちらも**実験開始前に検出可能**だった．

**作業**: `mise run start` の前段に「意図したコードパスが実行されているか」を確認するスモークタスクを追加する．最低限，次を検証する．

1. **git 状態の確認**: `git status --porcelain` が空でない場合に警告する（未コミットの変更はデプロイされない）．
2. **配布物のハッシュ照合**: ローカルの `http_server.py` / `router.py` / `config.yaml` と，各ノードのコンテナ内の同ファイルの md5 が一致することを確認する．
3. **1 問だけのプローブ実行**: 1 問を投げ，設定に応じて期待されるフィールドが埋まっていることを確認する．

   | 設定 | 期待される観測 |
   |---|---|
   | `confidence_signal_method=self_consistency_semantic` | `semantic_entropy` が非 null，`local_inference_ms` が秒オーダー |
   | `confidence_signal_method=stp` | `confidence_logprobs_mean` が非 null |
   | `confidence_signal_method=p_true` | `p_true` が非 null |
   | `routing_method=supervised_classifier` | `local_inference_ms` が数 ms オーダー |

**成功条件**: Iter12 と Iter22 の状況をそれぞれ再現させたとき，スモーク段階で検出できること．

**コスト**: 実装 1〜2 時間．1 実験あたり数分の追加．約 2 時間の実験を 1 回でも救えば元が取れる．

---

### F3. `metrics.py` への指標統合

**背景**: `.claude/research/config.yml` の success_criteria (4) は「confidence 信号系のレバーでは accuracy に加えて ECE・AUROC・同点率・ノード間 confidence 分散を報告する」と定めているが，`metrics.py` にはいずれも実装されていない．毎回その場限りのスクリプトで計算した結果，journal に誤った ECE が 2 通り記録されている（d0002 §7-1）．

**作業**: `metrics.py` に次を追加する．いずれもオフライン計算で，実験の再実行は不要．

| 追加する関数 | 内容 |
|---|---|
| `compute_ece(results, n_bins=10)` | Expected Calibration Error．`confidence` が非 null の行を対象とし，対象行数も併せて返すこと（フォールバック行の扱いで値が変わるため） |
| `compute_brier_score(results)` | Brier score |
| `compute_auroc(results)` | confidence を score，ルーティング正誤を label とした AUROC．較正（ECE）と弁別（AUROC）は別物であり，順位付け問題である本研究では AUROC の方が本質的である |
| `compute_tie_rate(results)` | `probe_candidates` の上位 2 件の confidence が一致する割合 |
| `compute_confidence_dispersion(results)` | probe 内のノード間 confidence の標準偏差と合計の平均（self_report では合計 ≠ 1，top_k_with_probs では 1 になる） |

いずれも既存の実装方針（scipy / numpy 不使用，`math.erf` による閉形式）に合わせる．`compute_all_metrics()` の戻り値と `print_summary()` にも追加する．

**検証**: 追加後，既存の結果に対して次の値が再現することを確認する（本調査で単一実装により算出した正しい系列）．

| 実験 | ディレクトリ | ECE | 同点タイ率 |
|---|---|---|---|
| Iter15 | `results/20260727_010532` | 0.7146 | 98.29% |
| Iter16 | `results/20260727_100917` | 0.7388 | 82.83% |
| Iter17 | `results/20260727_180824` | 0.1927 | 0.00% |
| Iter18 Phase C | `results/20260729_042712` | 0.1927 | 0.00% |
| Iter20 | `results/20260729_110720` | 0.1927 | 0.00% |
| Iter21 | `results/20260729_151234` | 0.1927 | 0.00% |
| Iter22 | `results/20260729_190824` | 0.1927 | 0.00% |

**コスト**: 実装とテスト 2〜3 時間．実験不要．

---

### F4. journal・backlog の記録訂正

**背景**: d0002 §7 に列挙した記録誤りが残っており，次の rc-planner がこれを前提に計画を立てると誤った実験を組む．

**作業**: 次の 3 点を journal に追記する（過去の記述を書き換えるのではなく，訂正として追記する）．

1. **E3（`confidence_elicitation`）の採用判定を再判定する．** Iter20 の「決定的改善」は Iter17 の E6 導入時に既に起きていた変化であり，E3 の有効な測定は Iter16 の 1 回のみである（top1 0.2059，McNemar p=0.0783 で有意差なし，ECE 0.7146→0.7388 と悪化）．
2. **ECE の正しい系列**（F3 の表）を記録する．Iter21 の「0.1903 へ改善」は誤りで，実際の変化は 0.0000 である．
3. **Iter18 Phase C の `top1_accuracy` は 0.5651 であり 0.5693 ではない**（0.5693 は `single_domain_top1_accuracy`）．「E10 で top1 が +0.0042 改善」という記述は誤りで，実際は McNemar 不一致 0/1520 で完全に不変である．

併せて `.claude/research/config.yml` の levers に，E3 と E7 が `supervised_classifier` 下で no-op である旨を注記する（E7 については backlog B35 に既に記録がある）．

**コスト**: 1 時間以内．

---

### F5. 再現性の担保

**背景**: `.gitignore` により `data/`・`results/`・`models/` が管理外である．評価データセット，全実験結果，LoRA アダプタ，分類器がバージョン管理されておらず，再現性がローカルディスクのみに依存している．

**作業**: 容量の都合で成果物そのものを git に入れないのは妥当なので，代わりに次を記録する．

1. `data/dataset.jsonl`・`data/classifier_train.jsonl`・`models/domain_classifier.joblib`・各 LoRA アダプタの **sha256 ハッシュと生成コマンド**を，リポジトリ内のマニフェストファイル（例: `data/MANIFEST.md`）に記録し，これは git 管理下に置く．
2. 各実験ディレクトリに，そのときの `config.yaml` と `git rev-parse HEAD` の結果を保存する（`run_experiment.py` に追加）．Iter22 のような「どの HEAD がデプロイされたか分からない」状況を防ぐ．

**コスト**: 2〜3 時間．

---

## 3. 第 2 段階: 基準線の再確立

### X1. 最良既知構成での基準線再取得

**依存**: F1・F2・F3 の完了．

**目的**: 以後すべての比較の基準となる数値を，正しい構成・正しい計測基盤で 1 度取り直す．現在，E6 + E10 を同時に有効にした状態の完全な指標セットが存在しない（Iter18 Phase C は取得しているが ECE 等が一時スクリプト計算）．

**構成**:

| 設定 | 値 |
|---|---|
| `routing_method` | `supervised_classifier` |
| `confidence_signal_method` | `self_report`（制約 1 により，これ以外にすると分類器が無効化される） |
| `expert_model` | `expert-mesh-{domain}-lora`（10 ノード） |
| `light_model` | `qwen3.5:4b-q4_K_M` |
| `confidence_threshold` | 0.5 |
| `dispatch_top_k` | 1 |
| データセット | JMMLU 1520 問 |

**手順**:

```
1. F1 の config 修正
2. mise run setup      # Docker イメージ再ビルド（Iter12 の教訓）
3. mise run deploy
4. F2 のスモーク検証     # local_inference_ms が数 ms であることを確認
5. mise run start
6. mise run analyze
7. uv run python metrics.py --results results/<dir>/results.jsonl --json
```

**期待値と成功条件**: ルーティング系は決定論的なので Iter18 Phase C と**完全一致**するはずである．一致しなければデプロイか実装に想定外の差がある．

| 指標 | 期待値 | 判定 |
|---|---|---|
| top1_accuracy | 0.5651 | 完全一致すべき |
| Cohen's kappa | 0.5215 | 完全一致すべき |
| ECE | 0.1927 | 完全一致すべき |
| 同点タイ率 | 0.00% | 完全一致すべき |
| answer_quality_accuracy | 0.5013 ± 0.013 | ノイズ床 1.3pt の範囲内（X6 参照） |
| end_to_end_accuracy | 0.3151 ± 0.013 | 同上 |
| mean_duration_ms | 約 3515 | 4B 汎用の 6498ms から戻ることを確認 |

**コスト**: 約 90 分（Iter18 Phase C は 89 分）．

---

### X6. 回答品質のノイズ床の確定

**依存**: X1 と同時実施．

**目的**: 軸②③の判定基準を経験則から実測に置き換える．現在，答えられていない問い —— 「answer_quality_accuracy が何 pt 動いたら有意と言えるか」．

**背景**: 同一構成の Iter20 と Iter22 で 0.2313 と 0.2180（1.33pt 差）という 1 対のデータしかない．標準偏差の推定には不足する．

**手順**: X1 の構成のまま，同一データセットで **3 回**実行し，`answer_quality_accuracy` と `end_to_end_accuracy` の標準偏差を求める．ルーティングは決定論的なので軸①は 3 回とも同一になるはずで，これ自体が F2 の検証にもなる．

**成功条件**: 標準偏差が算出でき，`.claude/research/config.yml` の success_criteria に「軸②③の有意判定は 3SD を超える変化とする」といった形で反映されること．

**コスト**: 約 90 分 × 3 = 4.5 時間．X1 の 1 回目を含めれば追加は 2 回分（3 時間）．

**代替案**: LLM-as-judge を含めた完全な再実行が高コストなら，`results.jsonl` の `answer_text` を保存したうえで採点だけを複数回行う方法もあるが，生成の確率性こそが主要なノイズ源なので**生成から繰り返すべき**である．

---

## 4. 第 3 段階: 研究の主張を支える実験

### X2. 中央集権ルータとの比較（最重要）

**依存**: X1 の完了．

**目的**: 設計書 §4.2(b) が主要比較対象と定める中央集権ルータと比較し，研究の問い 2（「分散であることのコストは小さいか」）に答える．**これが現在，研究の主張における最大の空白である．**

**構成**: 1 台のノードに全専門家モデルを集約し，同一の 1520 問を処理させる．

- ルーティング: 同一の `models/domain_classifier.joblib` を中央で 1 回実行し，最大確率のドメインを選ぶ（分散版は各ノードが自分の確率のみ返して requester が集約するが，分類器が同一なので**ルーティング結果は理論上一致するはず**である）．
- 回答生成: 選ばれたドメインの `expert-mesh-{domain}-lora` を同一ホスト上の Ollama で実行する．
- 実装は既存資産を再利用できる．新規に大きなコンポーネントを作る必要はない．

**測定すべき指標**:

| 分類 | 指標 | 意味 |
|---|---|---|
| 精度 | top1_accuracy，answer_quality，end_to_end | 分散版と一致するか（一致するはず．一致しなければ分散化に精度上のコストがある） |
| **オーバーヘッド** | `other_ms`（= `duration_ms` − `dispatch_gen_time_ms`） | **分散版で 136ms．中央版との差が「分散であることのコスト」そのもの** |
| オーバーヘッド | probe フェーズの所要時間 | 中央版には存在しない．10 ノードへの並列 probe のコスト |
| オーバーヘッド | 通信バイト数 | 設計書 §4.1 軸③が要求 |
| リソース | ピーク VRAM，モデルのロード／アンロード回数 | 中央集約では 10 モデルを 1 台に載せられず swap が発生する可能性が高い．**これは分散版の優位点として主張できる材料になる** |

**成功条件（主張の形）**: 精度が同等（McNemar で有意差なし）でありながら，分散版のオーバーヘッドが実用上許容できる範囲（例えば `other_ms` が全体の 5% 未満）にとどまること．設計書 §4.5 の想定主張「中央集権ルータに対して大きなオーバーヘッドなく比較可能な精度を達成できる」を数値で裏付ける．

**注意**: 中央集約側は 1 台に 10 個の LoRA モデルを載せることになり，6GB VRAM では確実に収まらない．モデルの swap 時間を含めるか除くかで結論が変わるため，**両方を報告する**こと（「モデル常駐を仮定した理想的な中央集権」と「実際に 1 台で動かした場合」）．これは分散アーキテクチャの優位性を主張する上でむしろ重要な論点である．

**コスト**: 実装 1 日程度 + 実験約 2 時間．

---

### X4. 複合ドメイン評価の立て直し

**依存**: X1 の完了．

**目的**: 研究の問い 3（複合ドメイン設問の評価手法）に答えられる状態にする．

**現状の問題**:

- `dispatch_top_k=1` では 2 ドメインを期待する設問のカバレッジ上限が 0.5．実測 `compound_domain_set_recall=0.125`，`compound_mean_dispatched_count=0.70`．
- 複合設問は 1520 問中 20 問（1.3%）しかなく，統計的な議論ができない．
- Iter15 の複合 top1 0.95 は偽高値だった（19/20 が `['medical','legal']` で legal が宣言順優位でタイに勝っていただけ）．

**手順**:

1. **複合設問の拡充**: 20 問 → 100 問規模へ．ドメインの組み合わせを多様化する（現状は `['medical','legal']` に偏っている）．JMMLU は四択形式のため複合設問は手作りを維持する必要があり，これが主な作業コストになる．
2. **`dispatch_top_k=2` での再評価**: 単一ドメイン設問への副作用（無駄な dispatch の増加，レイテンシ増）も併せて測る．Iter1 では medical の confidence 0.2 が閾値に阻まれて k を上げても発火しなかったが，現在は分類器の softmax 連続値なので状況が異なる．
3. **評価指標の確定**: 複合設問に対して `top1_accuracy` を使うのが妥当か，`compound_domain_set_recall` を主指標にすべきかを決める．単一選択を前提とした指標を複合設問に適用していることが，Iter15→Iter17 の「0.95→0.25 への退行」という誤解を招いた．

**成功条件**: 複合設問について，宣言順のような構造的アーティファクトに依存しない指標で，統計的に議論できる規模（n≥100）の結果が得られること．

**コスト**: 設問作成が主．1〜2 日 + 実験約 2 時間．

---

### X5. fallback 方針の見直し

**依存**: X1 の完了．

**目的**: 現行の fallback が有害であることが判明しているため，方針を決め直す．

**現状**（本調査で `results/20260729_190824/results.jsonl` から実測）: `fallback_rate` 13.16%（200/1520），fallback 先は **200/200 すべて general**，その正解率 **8.00%（16/200）**は Random の 10.13% を下回る．一方，非 fallback 行の正解率は **63.86%（843/1320）**．つまり **fallback は「分からないときに general に投げる」という設計だが，実際には Random 未満の精度しか出しておらず，全体の正解率を下げている**．

**比較すべき選択肢**:

| 案 | 内容 | 期待される効果 |
|---|---|---|
| A | `confidence_threshold` を 0.5 → 0.3〜0.4 に下げる | fallback を減らし，分類器の判断をより信用する |
| B | fallback を廃止し，常に最大確率のドメインへ送る | 閾値ゲート自体を外す．最もシンプル |
| C | fallback 時に `dispatch_top_k=2` で上位 2 ドメインへ送る | 不確実なケースこそ複数に聞く．X4 と連動 |

**注意**: これは単一レバーとして `confidence_threshold` を振る形（案 A）に落とし込める．案 B・C はコード変更を伴う．

**成功条件**: `top1_accuracy` と `end_to_end_accuracy` の両方で現行を上回ること．特に fallback 行 200 件の正解率が Random（10.1%）を上回ること．

**コスト**: 案 A なら config 1 行 + 実験約 90 分．

---

## 5. 第 4 段階: 残レバーと較正

### X3. E4（semantic entropy）の再設計

**背景**: d0002 §6-D のとおり，現行の「bug fix」を適用したまま実験すると，採用済みの E6（top1 を 0.2059 から 0.5651 へ改善した最大の成果）が無効化される．**現行計画のまま実行してはならない．**

**選択肢**:

| 案 | 内容 | 長所 | 短所 |
|---|---|---|---|
| **X3-a（推奨）** | 分類器の確率を主とし，semantic entropy を**較正用の補正項**として併用する．例: `confidence = p_classifier × (1 − normalized_entropy)`．ルーティングの argmax は分類器が決めたまま，confidence の値だけが較正される | 単一レバー原則を守れる．E6 を保持したまま E4 の「較正への寄与」だけを測れる．ECE の改善という E4 本来の目的に合致する | `_estimate_probe_confidence()` の構造変更が必要（排他的 if 連鎖から，routing と confidence の 2 段構成へ） |
| X3-b | `routing_method=self_report` に戻したうえで E4 を評価する | 実装変更が不要 | E6 を捨てた状態での評価になり，現行構成への示唆が限定的 |
| X3-c | E4 を見送り，X2 に資源を回す | 最も低コスト | 文献（Farquhar et al., Nature 2024）に基づく仮説が未検証のまま残る |

**X3-a の成功条件**（採用する場合）:

| 分類 | 指標 | ベースライン | 成功条件 |
|---|---|---|---|
| 主基準 | ECE | 0.1927 | 0.150 以下 |
| 主基準 | AUROC | 未測定（F3 で取得） | 現行を上回ること |
| 非退行 | top1_accuracy | 0.5651 | **完全一致すべき**（argmax は分類器が決めるため理論上不変．変化したら実装が誤っている） |
| 報告 | `semantic_entropy` | 未取得 | 分布と confidence との相関 |
| 報告 | mean_duration_ms | 約 3515 | 1 probe あたり最大 9 LLM 呼び出しで大幅増の見込み |

**注意**: `scripts/measure_semantic_diversity.py` は作成済みで，着手前に多様性（cluster 数 ≥ 2，entropy > 0.5 bits）を確認する手順も定義されている．Iter21 の事前測定では mean cluster count 3.25，mean entropy 1.234 bits と，多様性そのものは十分に出ていた．

**コスト**: X3-a は実装 0.5 日 + 実験 3〜4 時間（LLM 呼び出しが 9 倍になるため）．

---

### X7. E5（p_true）

**前提条件（未了）**: **各ノードの Ollama バージョンが v0.12.11 以降であることを実機で確認する．** これが満たされない場合，`estimate_confidence_p_true` は自動的に numeric_scalar の self_report にフォールバックし，Iter21 と同種の「実験したつもりで何も測っていない」状態になる．d0001 §5.2 の時点でこの確認は未実施のまま残っている．

**注意点**:

- STP（Iter13 で棄却）との違いを journal に明文化してから実施すること．STP は生成系列全体の平均トークン確率で長さバイアスがあり流暢さを測るのに対し，P(True) は単一位置の二値分類分布であり測定対象が異なる．これを明示しないと「Iter13 の焼き直し」と解釈される．
- 反証がある: Tian et al. の Table 1 は gpt-3.5 で "Is True" 方式が verbalized より較正が悪いと報告している．Kadavath の良好な結果は 52B base model + 20-shot 設定である．
- **制約 1 により，E5 も E4 と同じ構造問題を抱える**．X3-a と同じ 2 段構成の実装を先に済ませておけば，E5 もそのまま乗せられる．

**コスト**: バージョン確認 30 分 + 実験約 2 時間（X3-a の実装が済んでいれば）．

---

### X8. education / legal のデータ不均衡是正

**背景**: legal は JMMLU に `professional_law` が存在せずタスクプールが 227 問しかないため，分類器の訓練データが 77 件（他 9 ドメインは各 150 件）．education は直接対応するタスクがなく心理学・社会学で代理している．結果として education の precision/recall は 0.520/0.411 と最下位圏で，Iter17 で唯一の非退行違反を出した．

**手順の候補**:

1. legal の訓練データを他の法律系日本語データセットで補う（LegalBench 等．d0001 §5.2 でライセンス未検証として残っている）．
2. `class_weight="balanced"` は既に設定済みなので，oversampling や追加データの方が効果が見込める．
3. education については，代理タスクによる写像の妥当性そのものを見直す．写像表を成果物として明示し，境界事例を別集計することが d0001 §5.1 で推奨されている．

**成功条件**: education と legal の recall が，他ドメインの下限（現状 business_economics の 0.4533）を上回ること．

**コスト**: データ収集次第．1〜3 日．

---

### X9. 分類器の較正

**背景**: 全ドメインで mean_confidence > accuracy であり，ECE 0.1927 のうち [0.90, 1.00) バケットの gap が 0.1750 を占める．分類器のオフライン性能は訓練 100.00% に対し評価 59.87% で，過学習の傾向が残っている．

**手順**: `scripts/train_domain_classifier.py` に `CalibratedClassifierCV`（Platt scaling または isotonic regression）を導入し，交差検証で較正する．訓練データ 1427 件を分割して較正用に使う．

**成功条件**: ECE が 0.150 以下になり，かつ `top1_accuracy` が非退行（argmax の順位は較正で変わりにくいため，理論上ほぼ不変のはず）．

**注意**: これは X3（semantic entropy による較正）と**同じ目的の代替手段**である．較正だけが目的なら，LLM 呼び出しを 9 倍にする X3 より X9 の方が桁違いに安い（実験不要，分類器の再訓練のみ）．**X3 より先に X9 を試すのが合理的である．**

**コスト**: 実装 2〜3 時間 + 検証実験約 90 分．

---

## 6. 人間の判断が必要な事項

以下は自動サイクルで決めるべきでない事項であり，実施前に確認を要する．

| ID | 論点 | 選択肢 |
|---|---|---|
| **D1** | E3（`confidence_elicitation`）の採用判定を取り下げるか | (a) 取り下げ，Iter16 の結果（有意差なし・ECE 悪化）で「棄却」に改める / (b) 「supervised_classifier 下では no-op のため判定不能」として保留にする（**推奨**．設定自体は害をなさない） |
| **D2** | E4 の扱い | X3-a（2 段構成に実装変更して併用）/ X3-b（`self_report` に戻して評価）/ X3-c（見送り）．**X9 を先に試したうえで X3-a を選ぶことを推奨** |
| **D3** | 中央集権ルータ比較（X2）の実装範囲 | 1 台に 10 モデルを常駐できないため，「モデル常駐を仮定した理想値」と「実際に swap を伴う実測値」のどちらを主として報告するか．**両方を報告し，swap コストを分散版の優位点として位置付けることを推奨** |
| **D4** | 複合設問（X4）の拡充規模 | 20 問 → 100 問は手作業が主体になる．どこまで工数をかけるか．研究の問い 3 に答えるには必須 |
| **D5** | `data/` `models/` のバージョン管理方針 | (a) ハッシュのマニフェストのみ git 管理（**推奨**）/ (b) Git LFS を導入 / (c) 現状維持 |
| **D6** | `.claude/research/config.yml` の `git.push: true` | グローバル CLAUDE.md の「`git push` 絶対禁止」規約と衝突している．どちらを優先するか |

---

## 7. 未着手レバーの扱い

`.claude/research/config.yml` の `levers` に定義されていながら未実施のもの．

| レバー | 状態 | 本ドキュメントでの扱い |
|---|---|---|
| E3 `linguistic` | 未実施 | **実施不要**．制約 1 により `supervised_classifier` 下では no-op．`self_report` に戻さない限り測定できない |
| E4 `self_consistency_semantic` | 2 回とも無効 | X3 として再設計．ただし X9 を先に試す |
| E5 `p_true` | 未実施 | X7．Ollama バージョン確認が前提 |
| E7 `embedding_postprocess=whitening` | スキップ済み | **実施不要**．`supervised_classifier` 下で no-op（backlog B35 で確定） |
| E10 `offtheshelf_specialized` | 実施不能 | 日本語の法律特化オープン生成モデルが調査で発見できず．`domain_lora` が唯一の実行可能アプローチ |
| （B7）nomic-embed-text の task prefix 付与 | 未実施 | 現行の分類器経路でも生の埋め込みを使っているため，prefix 付与で分類精度が変わる可能性は残る．X8 と併せて検討する価値がある |
| （B24）`hidden_state` | 取得不可 | Ollama では実現不能．vLLM / SGLang への移行が前提 |

---

## 8. まとめ: なぜこの順序なのか

1. **測定系が壊れている状態で実験しても結論が出ない．** 直近 3 イテレーションはすべて無効であり，原因はいずれも測定系（config の残置，デプロイ不備，指標の計算方法）にあった．F1〜F3 が最優先である理由はここにある．
2. **基準線が存在しない．** E6 + E10 を同時に有効にした構成の完全な指標セットがない．X1 なしには，以後どの実験も「何と比べているのか」が定まらない．
3. **研究の主張の核心が空白である．** ルーティング精度の改善（0.2059 → 0.5651）は十分な成果だが，それは研究の問い 1 に答えるだけである．問い 2（分散のコスト）に答える X2 と，問い 3（複合ドメインの評価手法）に答える X4 がなければ，設計書が掲げた 3 つの問いのうち 2 つが未回答のまま残る．
4. **残レバーの期待値は相対的に低い．** E4・E5 はいずれも「confidence の較正を改善する」ことが目的だが，同じ目的をより安く達成できる X9（分類器の較正）が未検討である．また Iter16 の教訓（elicitation の変更だけでは ECE > 0.7 が動かなかった）は，confidence の抽出方式をいじる方向の限界を示唆している．
