<!-- research-cycle (Iter1〜22) で得られた実験設定・結果・分析・知見の総括．次に何をすべきかは d0003 を参照． -->

# d0002: research-cycle 知見総括（Iter1〜22 / 2026-07 時点）

## このファイルについて

- **目的**: `.claude/research/journal.md` および `journal_archive.md` に分散している Iteration 1〜22 の実験設定・結果・分析・考察を，1 つの参照可能な形に統合する．
- **役割**: 「どういう設定で何を測り，何が分かったか」の唯一の要約．個々のイテレーションの生の記録は journal 側に残す．
- **利用方法**: 新しくこのプロジェクトに参加した場合は本ファイルを最初に読む．次に行うべき実験・修正は `d0003_next_experiments_2026-07.md` を参照する．
- **記載の原則**: 数値はすべて `results/*/results.jsonl` から再計算または `metrics.py` で再現確認したものを正とする．journal の記載と食い違う箇所は §7 に訂正として明記する．

---

## 0. 要約

- **達成できたこと**: 評価基盤を 46 問から JMMLU 1520 問へ拡張し，統計的判定（Wilson CI / McNemar / Cohen's kappa）を可能にした．ルーティング方式を教師あり分類器へ切り替えることで top1_accuracy を 0.2059 → 0.5651，Cohen's kappa を 0.1067 → 0.5215 に改善し，同点タイ率 82.83% を 0.00% に解消した．ドメイン別 LoRA によりノード間に実際の能力差を作り，回答品質を 0.2787 → 0.5013 に改善した．
- **主張できる範囲**: 自己申告 confidence とトークン確率の双方が専門性信号として機能しないことを，10 ドメイン・1520 問の規模で定量化した．教師あり分類器がこれを解決することも示した．BestSingle ベースライン（0.1092）を大きく上回っている．
- **まだ主張できないこと**: 設計書 §4.2(b) が主要比較対象とする**中央集権ルータとの比較が未実施**であり，研究の中心命題（分散であること自体のコストが小さい）を支持する証拠がない．
- **見つかった重大な問題**: rejected 構成が `config.yaml` に残置されたまま直近 3 実験が実行されたこと，`confidence_elicitation`（E3）の採用判定が構造上の no-op に基づいていたこと，直近 2 回の実験がデプロイ不備で無効になったこと．詳細は §6・§7．

---

## 1. 研究の目的・評価軸・ベースライン

### 1-1. 研究の問い

`docs/encounter_expert_mesh_design.md` §7 が定める 3 つの問い．

1. 有線 LAN 常時接続の環境において，多様な専門知識を持つノード群が HTTP POST ベースの軽量な問い合わせだけで「誰に聞くべきか」を自律的に決められるか（ルーティング精度の評価）．
2. その自律分散的な仕組みは，**最も条件の良い（＝中央集権方式に有利な）ネットワーク環境下でも，許容できるオーバーヘッドで中央集権方式と同等の性能を達成できるか**（実現可能性・コストの評価）．
3. その仕組みを，実生活に即した多様な質問（既存ベンチマークではカバーされない複合ドメインの質問を含む）に対して評価するためのベンチマークをどう設計すべきか（評価手法そのものへの貢献）．

### 1-2. 主張の位置付け

`plans/p0001_research_direction_2026-07.md` §4 は，分散・中央ルータなしという構成自体は既に多数の先行事例（A2A の Agent Card，AgentNet NeurIPS 2025，DMoE 2020 等）があるため，アーキテクチャそのもので新規性を主張することは難しいと判断している．代わりに主張しやすい貢献として，次を挙げている．

> **API-only の commodity GPU 環境において，自己申告 confidence とトークン確率の双方が専門性信号として機能しないことの定量化，および教師ありルータとの対比．**

これは Internet of Agents サーベイ（arXiv:2505.07176）が capability evaluation を「self-reported declarations」と「system-level verification」に分け，前者について「登録は速いが不正確または誇張された主張につながりうる」と述べている点への，定量的な裏付けとなる．

### 1-3. 評価軸

正式な定義は `docs/encounter_expert_mesh_design.md` §4.1 にある．

| 軸 | 内容 | 実装 |
|---|---|---|
| ① ルーティング精度 | Top-1/Top-k 正解率，適合率・再現率，誤ルーティング率 | `metrics.py`（オフライン） |
| ② 回答品質 | ドメイン QA ベンチマークでの正答率／LLM-as-judge スコア | `evaluation.py` + `scripts/evaluate_response_quality.py` |
| ③ システム全体の実効性 | ①×② を統合した End-to-End 正答率，レイテンシ内訳，通信バイト数 | 同上（①∧② の論理積） |
| ④ 動的環境での頑健性 | ネットワーク不安定時の成功率低下率など | **本フェーズではスコープ外**（Phase 3） |

軸②の採点方式: JMMLU 由来行（`jmmlu_answer` を持つ 1500 行）は `extract_answer_letter()` で抽出した A/B/C/D と正解の一致，手作りの複合設問 20 行は LLM-as-judge（1〜5 Likert，合格閾値 3，temperature 0.0）．judge には専用モデルを立てず general ノードの `expert_model` を再利用する．

軸③の `end_to_end_accuracy` は「ルーティング正解 **かつ** 回答品質合格」の割合で，未採点行は不正解として扱う．

### 1-4. 必要とされるベースライン

設計書 §4.2 が定める 4 種．

| ベースライン | 目的 | 状態 |
|---|---|---|
| (a) 単一汎用小型モデル（専門化なし） | 専門分化そのもののメリットを測る | Iter18 Phase A が実質的に相当（answer_quality 0.2787） |
| (b) **中央集権ルータ**（1 台に全専門家を集約） | **分散アーキテクチャであること自体のコストを測る主要比較対象** | **未実施** |
| (c) オラクルルーティング | ルーティング精度の理論的上限 | 実装済み（`oracle_accuracy`） |
| (d) クラウド大規模モデル | 参考上限 | 未実施 |

加えて `docs/d0001_literature_survey_2026-07.md` §3.4 は Random / BestSingle / Oracle の 3 つを必須とし，「expert-mesh が BestSingle を上回っているかはまだ報告されていない」と指摘していた．**本調査でこれは達成済みであることを確認した**（§4-2）．

---

## 2. 実験設定の変遷

研究は大きく 3 期に分かれる．

| | 第 I 期（Iter1〜3） | 第 II 期（Iter4〜14） | 第 III 期（Iter15〜22） |
|---|---|---|---|
| 評価集合 | 手作り 34 問（単一 30 + 複合 4） | 手作り 46 問（単一 40 + 複合 6） | **JMMLU 1520 問**（単一 1500 + 複合 20） |
| ドメイン数 | 3（general / legal / medical） | 4（+ education） | **10** |
| ノード | wafl500 / 502 / 503 | + wafl501 | **wafl500〜509**（192.168.15.100〜109） |
| light_model | qwen3.5:9B 系 | `isotnek/qwen3.5:9B-Unsloth-UD-Q4_K_XL` | **`qwen3.5:4b-q4_K_M`** |
| expert_model | light と同一 | light と同一（差分はプロンプト 1 文） | Swallow-8B → **domain LoRA**（Iter18）→ 4B 汎用（Iter19 以降，§6-C 参照） |
| 統計基盤 | なし | なし | Wilson CI / McNemar / Cohen's kappa / 3 ベースライン |

### 2-1. 第 III 期のデータセット

- 出典: JMMLU（`nlp-waseda/JMMLU`, commit `3637b25e444`，56 タスク・7,536 問）．実データで確認したライセンスは **CC BY-NC-ND 4.0**．
- 10 ドメイン各 150 問（単一 1500 問）+ 手作りの複合ドメイン設問 20 問．
- 既知の偏り:
  - **legal**: JMMLU に `professional_law` が存在せず，タスクプールが 227 問・2 タスクしかない．分類器訓練データは 77 件（他 9 ドメインは各 150 件）．
  - **education**: 直接対応する JMMLU タスクがなく，心理学・社会学で代理している（`build_dataset.py` の docstring に明記）．
- 訓練／評価の分離: 評価用シード `_JMMLU_SAMPLE_SEED=20260726`，分類器訓練用シード `_CLASSIFIER_TRAIN_SAMPLE_SEED=20260727` で分離し，さらに質問単位でも除外している．**本調査で `data/dataset.jsonl`（1520 問）と `data/classifier_train.jsonl`（1427 問）の本文重複が 0 件であることを実測確認した**（§6-E）．

### 2-2. ルーティング方式の変遷

1. **self_report（方式 B）** — Iter1〜16 の基本．各ノードの light_model に「あなたは {domain} 分野の専門家です」とプロンプトし confidence を自己申告させる．general のみ反転プロンプト（専門相談 0.0〜0.3 / 日常質問 0.7〜1.0）．temperature=0.1．
2. **embedding（方式 A）** — Iter2 のみ．nomic-embed-text の cosine 類似度を `(sim+1)/2` で再スケール．LLM 呼び出しなし．棄却．
3. **multi_sample** — Iter11．N=3・T=0.1 の平均．棄却．
4. **stp**（Surrogate Token Probability）— Iter12/13．mean logprob を sigmoid（shift=2.0）で正規化．棄却．
5. **top_k_with_probs**（confidence_elicitation）— Iter16．Verbalized Top-K（K=2）．部分的採用（ただし §6-B の訂正を参照）．
6. **supervised_classifier** — Iter17 で採用．各ノードが同一の多クラス LogisticRegression（`max_iter=1000, class_weight="balanced"`）をロードし，requester が計算済みの `query_embedding`（nomic-embed-text，768 次元）から自ドメインの `predict_proba` のみを返す．追加の LLM 呼び出しは不要で，中央ルータも導入しない．
7. **expert_specialization** — Iter18．ルーティング方式は不変で，`expert_model` 側にドメイン別 LoRA を適用．

全期間を通じて固定されているのは `confidence_threshold: 0.5` と `dispatch_top_k: 1`．

### 2-3. 評価指標の導入時期

| 時期 | 追加された指標・基盤 |
|---|---|
| 当初 | top1_accuracy，single/compound 別 top1，misrouting_rate，ドメイン別 precision/recall，fallback_rate，dispatch_failure_rate，mean_duration_ms |
| Iter1 | 複合ドメイン被覆指標一式（`compound_domain_set_recall`，`compound_domain_jaccard_mean` 等）．併せて `results.jsonl` に `dispatched_domains` と `probe_candidates` を追加 |
| Iter15 | **Wilson score interval，Cohen's kappa，McNemar 検定，Random / BestSingle / Oracle の 3 ベースライン**（scipy 不使用，`math.erf` による閉形式実装）．評価軸②③として `evaluation.py` を新規実装 |
| Iter16 | ECE，parse_failure_rate，同点タイ率（いずれも `metrics.py` 外の一時スクリプト） |
| Iter18 | `mise run analyze` に `evaluate_response_quality.py` を統合し，軸②③を標準フローで自動計算 |

---

## 3. イテレーション別の結果一覧

判定の凡例: **採用** / **棄却** / **無効**（実験自体が成立していない） / **no-op**（変更が効果を持たないことが確定）．

### 3-1. 第 I 期（3 ドメイン・34 問）

| Iter | レバー | 判定 | 主要数値 | 学び |
|---|---|---|---|---|
| 1 | `dispatch_top_k` 1→2 | 棄却 | compound_covered 4→5（目標 6 未達），top1 0.9706（両ラン同一） | 複合行 3 件は medical の自己申告 confidence が 0.2 で `confidence_threshold=0.5` のゲートを越えず，k を上げても dispatch が発火しない．**真のボトルネックは並列数ではなく confidence 信号の質** |
| 2 | `routing_method` self_report→embedding | 棄却 | top1 0.5294 vs 0.9706，cosine 生値が **[0.6677, 0.7370]**（幅 0.069，std 0.0138）に潰れ 102/102 が閾値通過 | 帯域圧縮の要因は (a) nomic-embed-text の task prefix 未付与，(b) ドメイン名 1 単語という弱いルート定義，(c) 日本語クエリ対英単語ドメインのクロスリンガル比較，(d) general の catch-all 反転プロンプトが方式 A にない非対称性 |
| 3 | `confidence_threshold` [0.3, 0.5, 0.7] | no-op | confidence の実現値は {0.1, 0.2, 0.8, 0.85, 0.9, 0.95} のみ．0.3〜0.7 帯域は 0 件 | 記録済み `probe_candidates` へのオフライン掃引で決定的に no-op を確認し，実験自体をスキップ |

### 3-2. 第 II 期（4 ドメイン・46 問）

| Iter | レバー | 判定 | 主要数値 | 学び |
|---|---|---|---|---|
| 4 | education ドメイン追加 | 棄却（主基準は達成，非退行違反） | compound_covered 4→6（達成），single_top1 0.9667→0.9000，misrouting 0.0294→0.0870 | 新規ドメイン追加は複合被覆の絶対数を増やすが質は改善しない（set_recall 0.5→0.5）．education が catch-all として振る舞い general recall を押し下げた |
| 5 | `build_dataset.py` の education 質問差し替え | 棄却 | education confidence が baseline と**完全一致** | **決定的**: `_EDUCATION_QUESTIONS` はテストクエリであって few-shot 例ではない．confidence の few-shot 例は `router.py:build_confidence_prompt()` にハードコードされており，データセット変更は信号に一切影響しない |
| 6 | few-shot 例 3 追加 | 棄却 | education confidence が Iter5 と 10 問中 10 件一致 | 既存 few-shot は「該当する→高 confidence」しか示さず，**抑制のアンカリングが欠如**している |
| 7 | 抑制アンカー例 4 追加 | 棄却（7 PASS / 3 FAIL） | top1 0.957，single_top1 0.950（閾値 0.952 に僅差未達） | 視点の不一致，語彙的アンカリングの逆効果，ポジティブ例 3 対ネガティブ例 1 という比率の 3 点が原因 |
| 8 | 例 4 を education 視点へ書き換え | 棄却（3 PASS / 7 FAIL） | top1 0.9130，education recall 0.6667 | 「education ノードは 0.1 とする」の 1 行が「教育ノードは低 confidence を出すべき」という汎用ルールとして誤って一般化された |
| 9 | 例 1-3 の全ドメイン表示化 + 保守的指示 | 棄却（2 PASS / 4 FAIL） | top1 0.8696，education precision 1.0 / recall 0.5 | 過剰抑制という予想と逆の副作用．**Iter5〜9 の 5 回連続で few-shot 系レバーが無効**と確定 |
| 10 | `calibrated_routing`（probe 特徴量の logistic regression） | 棄却 | offline **AUC 1.000**（n=184，precision 0.975）→ online top1 0.870→0.848 | **offline 指標は online 改善を保証しない**．特徴量（margin, is_top1）が routing decision と情報的に重複し label leakage が生じた |
| 11 | `confidence_signal_method=multi_sample`（N=3, T=0.1） | 棄却 | top1 0.8478（-0.0218），confidence に変化があったのは 46 中 8 件 | **temperature=0.1 では出力が実質決定論的**で平均化が働かない．不確実性を消す設定で不確実性を測っていた |
| 12 | `confidence_signal_method=stp` | **無効** | `confidence_logprobs_mean` が 0/46 行，Iter11 と数値完全一致 | `mise run deploy` は rsync で config のみ配布し **Docker イメージを再ビルドしない**．Python コードはイメージに bake されるため変更が反映されなかった |
| 13 | STP 再実験 | 棄却 | top1 **0.0652**（偶然一致 0.25 を約 2.9 SD 下回る），confidence spread 0.0147（raw logprob spread 0.1328 の 1/9） | (1) sigmoid shift=2.0 の設計ミスで弁別力を 9 倍喪失．(2) **raw logprobs は生成の流暢さを測っており専門性ではない**．education ノードが常に最も流暢なため常に最高 confidence を得る |
| 14 | `confidence_signal_method=hidden_state` | 検証不能 | — | Ollama REST API では raw hidden state を取得できない（`/api/embed` は embedding model の semantic representation のみ）．`converged` を提案し人間判断を仰いだ |

### 3-3. 第 III 期（10 ドメイン・1520 問）

| Iter | レバー | 判定 | 主要数値 | 学び |
|---|---|---|---|---|
| 15 | **E1** `eval_set_size` 46→1520 | **採用** | top1 **0.1836**（CI [0.165, 0.204]），kappa **0.081**，同点タイ **98.29%**，ECE 0.7146，Random 0.101 | self_report は 10 ドメインで実質ランダム．クロスドメイン高 confidence 率は mathematics 91.3%，legal 90.4%，general 38.4% と，**専門ノードほど分野外でも高 confidence を出す**．タイ勝者は宣言順に強く依存（general 42.9%，education 33.3%）．Iter14 の `converged` 判定を撤回 |
| 16 | **E3** `confidence_elicitation` → top_k_with_probs | 部分的採用（**§6-B で要再判定**） | top1 **0.2059**（CI [0.1863, 0.2270]），kappa 0.1067，同点タイ **82.83%**，**ECE 0.7388（Iter15 の 0.7146 から悪化）**，McNemar **p=0.0783（有意差なし）** | Top-K は二峰飽和を部分的にしか壊さない（離散 5 段階のためタイが残る）．**numeric_scalar でも top_k_with_probs でも ECE > 0.7 であり，elicitation の変更だけでは self_report の構造的問題は解消されない** |
| 17 | **E6** `routing_method` → supervised_classifier | **採用** | top1 **0.5651**（CI [0.5401, 0.5899]），kappa **0.5215**，同点タイ **0.00%**，ECE **0.1927**，McNemar **p < 0.000001**，fallback_rate 0.1316 | self_report の構造的問題は routing_method の変更でしか解決できないことが確定．Iter2 の棄却は正当だったが原因は「unsupervised であること」であり「信号がない」証明ではなかった．Random→Oracle 距離の 51.5% を充填 |
| 18 | **E10** `expert_specialization` → domain_lora | **採用** | answer_quality **0.2787→0.5013**（+22.3pt），end_to_end **0.1697→0.3151**（+14.5pt），top1 0.5651（不変，McNemar 不一致 0/1520） | **ノード間に能力差が生まれて初めてルーティング精度が回答品質に直結する**．それ以前の top1_accuracy は下流に帰結を持たない代理指標だった．LoRA 訓練は 10 ノード並列で wall-clock 2〜4 時間（直列比 1/10 以下） |
| 19 | **E8** `expert_model_size` → qwen3.5:4b | **棄却** | mean_duration **3515→6498ms（1.85 倍遅い）**，VRAM 5.67→3.4GB，answer_quality 0.5013→0.2373 | 「モデルサイズ = 推論速度」は誤り．原因候補は量子化形式（Q4_K_M vs Q4_K_XL），`ollama create` 由来か `ollama pull` 由来かの差，アーキテクチャ差．回答品質低下の 84% は LoRA 撤去，16% がモデル縮小 |
| 20 | **E3** 再測定 | 採用と記録（**§6-B で無効**） | top1 0.5651，ECE 0.1927，同点タイ 0.00%（すべて Iter17/18C と同一） | 実装フェーズの記録に「`git diff config.yaml` は差分なし」とあり，**設定を一切変更していない純粋な再実行だった** |
| 21 | **E4** `confidence_signal_method` → self_consistency_semantic | **無効** | Iter20 と全メトリクスが同一，`semantic_entropy` が 0/1520 件，`local_inference_ms` が 1-3ms | `http_server.py:_estimate_probe_confidence()` で `routing_method=supervised_classifier` の early return が `confidence_signal_method` の分岐より先にあり，コードパスに 1 度も到達していなかった |
| 22 | **E4** 再実行 | **無効** | 同上（`semantic_entropy` 0/1520 件） | 分岐順序の修正は working tree に適用されたが，`mise run deploy` は git HEAD から配布するためコンテナに反映されなかった．修正は事後に `30e3627` としてコミットされた |

---

## 4. 現在の到達点

### 4-1. 軸① ルーティング精度（Iter17 以降，1520 問）

すべて本調査で `metrics.py` により再現確認済み．

| 指標 | 値 |
|---|---|
| top1_accuracy | **0.5651**（Wilson 95% CI [0.5401, 0.5899]） |
| Cohen's kappa | **0.5215** |
| ECE（10-bin，非フォールバック 1320 行） | **0.1927** |
| 同点タイ率 | **0.00%**（0/1520） |
| misrouting_rate | 0.4349 |
| fallback_rate | 0.1316（200/1520） |
| dispatch_failure_rate | 0.0 |
| 単一ドメイン（1500 問） | 0.5693 |
| 複合ドメイン（20 問） | 0.2500 |

### 4-2. ベースライン比較（文献調査が「未報告」としていた項目）

| ベースライン | 値 | 対比 |
|---|---|---|
| Random | 0.1013 | +45.4pt |
| BestSingle（最良は legal / medical の 0.1092） | 0.1092 | **+45.6pt（超過を確認）** |
| Oracle | 1.0 | Gap@Oracle = 0.4349 |

Random→Oracle の距離のうち **51.5%** を充填している．

### 4-3. ドメイン別 precision / recall

| ドメイン | precision | recall |
|---|---|---|
| legal | 0.8174 | 0.5663 |
| history_culture | 0.7638 | 0.6467 |
| mathematics | 0.7246 | 0.6667 |
| social_science | 0.6850 | 0.5800 |
| computer_science | 0.6136 | 0.5400 |
| natural_science | 0.5800 | 0.5800 |
| education | 0.5200 | 0.4114 |
| medical | 0.5166 | 0.4699 |
| business_economics | 0.5113 | 0.4533 |
| **general** | **0.3168** | 0.6800 |

general は「過剰に引き受ける」（precision 0.32），education と business_economics は recall が 0.45 を下回る．general がボトルネックである構造は MoDEM（arXiv:2410.07490）が Other クラス 52.94% と報告した傾向と一致する．

### 4-4. 軸②③ 回答品質・End-to-End

| 構成 | answer_quality | end_to_end | mean_duration_ms |
|---|---|---|---|
| Iter18 Phase A（LoRA なし，Swallow-8B） | 0.2787 | 0.1697 | — |
| **Iter18 Phase C（domain LoRA）＝最良既知** | **0.5013** | **0.3151** | 3515 |
| Iter19 以降（4B 汎用．§6-C の残置による） | 0.2373 / 0.2313 / 0.2180 | 0.1434 / 0.1355 / 0.1257 | 6498 / 6451 / 6517 |

レイテンシ内訳（Iter18 Phase C）: `dispatch_gen_time_ms` 2972ms，`other_ms` 136ms．軸③が要求する「通信時間 vs ローカル推論時間」の分離は既に取得できている．

### 4-5. 複合ドメイン設問（20 問）

| 指標 | 値 |
|---|---|
| compound_domain_top1_accuracy | 0.2500 |
| compound_domain_set_recall | 0.1250 |
| compound_domain_jaccard_mean | 0.1250 |
| compound_mean_dispatched_count | 0.70 |

`dispatch_top_k=1` では 2 ドメインを期待する設問のカバレッジ上限が 0.5 であり，**構造的に評価が成立していない**．

---

## 5. 手法上の恒久知見

1. **自己申告 confidence は専門性信号として機能しない．** 10 ドメインで kappa 0.081（実質ランダム）．専門ノードほど分野外でも高い confidence を出す（mathematics 91.3% vs general 38.4%）．
2. **トークン確率（STP）も機能しない．** raw logprob は「生成の流暢さ」を測っており，ドメイン専門性ではない．最も流暢な応答を返すノードが常に勝つ．
3. **confidence の抽出方式（elicitation）を変えるだけでは解決しない．** numeric_scalar でも top_k_with_probs でも ECE > 0.7 のままだった．
4. **教師あり分類器のみが構造的問題を解決した．** Iter2 の embedding 失敗は埋め込み空間の anisotropy という既知の幾何的現象であり，「信号がない」証明ではなかった．
5. **同点タイと宣言順依存が最大のボトルネックだった．** 10 ドメイン期に 98.29% がタイになり，実質的に aggregator の宣言順タイブレークがルーティングを決めていた．softmax の連続値でのみ解消できた．
6. **偽高値（構造上のアーティファクト）に注意が必要．** Iter15 の general recall 0.687 はタイ勝率 42.9% 由来，複合ドメイン top1 0.95 は 19/20 が `['medical','legal']` で legal が宣言順優位だったことに由来する．より良い手法へ移行するとこれらが剥がれ「退行」に見える．
7. **ノード間に能力差がなければ top1_accuracy は代理指標にすぎない．** LoRA 導入で初めて実指標になった（top1 不変のまま answer_quality が +22.3pt）．
8. **「モデルサイズ = 推論速度」は誤り．** 4B 汎用が 9B+LoRA より 1.85 倍遅かった．量子化形式・モデルの作成経路・アーキテクチャが支配的である．
9. **offline 指標は online 改善を保証しない．** Iter10 の offline AUC 1.000 が online では退行した．
10. **プロンプト変更の副作用は予測できない．** 1 行の抑制指示が「教育ノードは低 confidence を出すべき」という汎用ルールとして誤って一般化された．
11. **小標本では判定できない．** Iter3・Iter5〜11 の no-op / 僅差棄却群は n=46（SE ±5.0pt）であり，「0.870→0.848」は 1 問差にすぎなかった．

---

## 6. 本調査で新たに判明した事実

以下は 2026-07-29 の総括調査で，実データ・実コード・git 履歴から新たに確認した事項である．いずれも journal には記載がない．

### 6-A. ルーティング経路は完全に決定論的である

Iter20 / Iter21 / Iter22 の 3 実験について，1520 問すべてで `selected_domain` と `confidence` が**ビット単位で一致**した（1520/1520）．一方 `answer_text` は 3 回とも 100% 異なる．

- 理由: `nomic-embed-text` による埋め込みと joblib の LogisticRegression は，いずれも確率的要素を持たない．生成のみが確率的である．
- 帰結 1: **`routing_method=supervised_classifier` の下では，ルーティング系指標の run 間ノイズは「小さい」のではなく「構造的にゼロ」である．** journal が「run 間ノイズは測定誤差の範囲内」と記述している箇所は，この意味に読み替える必要がある．反復実験は新しい情報を一切生まない．
- 帰結 2: top1_accuracy の Wilson CI [0.5401, 0.5899] は「1520 問という標本の抽出誤差」のみを表し，実行のばらつきは含まない．

### 6-B. `confidence_elicitation`（E3）は supervised_classifier 下で no-op である —— Iter20 の採用判定は無効

`http_server.py:_estimate_probe_confidence()` は排他的な if 連鎖であり，分岐は次の順序で return する（HEAD `30e3627` 時点）．

```
313  confidence_signal_method == multi_sample          → return
324  confidence_signal_method == stp                   → return
334  confidence_signal_method == semantic_entropy      → return
345  confidence_signal_method == p_true                → return
354  routing_method == embedding                       → return
364  routing_method == supervised_classifier           → return   ← ここで確定
371  confidence_elicitation == top_k_with_probs        → return   ← 到達しない
380  default: self_report                              → return
```

`routing_method=supervised_classifier` である限り 364 行で return するため，**371 行の `confidence_elicitation` 分岐には到達しない**．これは `embedding_postprocess` が同じ理由で no-op になる（backlog B35 で既に判明していた）のと同一の構造である．

git 履歴による裏付け:

| コミット | 内容 | `routing_method` | `confidence_elicitation` |
|---|---|---|---|
| `d2ec1f6` | Iter16 考察 | `self_report` | `top_k_with_probs` |
| `b3d4952` | Iter17 考察 | **`supervised_classifier`** | `top_k_with_probs` |
| `709b1ee` | Iter18 考察 | `supervised_classifier` | `top_k_with_probs` |
| HEAD | — | `supervised_classifier` | `top_k_with_probs` |

したがって:

- **E3 の有効な測定は Iter16 の 1 回のみ**である．そのときの結果は top1 0.2059（McNemar **p=0.0783，有意差なし**），**ECE は 0.7146 → 0.7388 と悪化**，同点タイは 98.29% → 82.83%（改善するが依然として高い）．
- Iter20 で報告された「同点タイ 82.83%→0.00%，ECE 0.7388→0.1927 の決定的改善」は，**すべて Iter17 の E6（supervised_classifier）導入時に既に起きていた変化**である．Iter20 は設定を一切変更していない再実行であり（実装フェーズの記録にも「`git diff config.yaml` は差分なし」とある），Iter17 / Iter18 Phase C / Iter20 / Iter21 / Iter22 の 5 実験はすべて同一の結果を返している．
- **結論: E3 の「採用」判定は取り下げ，Iter16 の結果（有意差なし・ECE 悪化）に基づいて再判定すべきである．** ただし現行構成で害をなすものではない（単に読まれない設定である）．

### 6-C. rejected 構成が `config.yaml` に残置されている

| コミット | 内容 | `expert_model`（10 ノード） |
|---|---|---|
| `709b1ee` | Iter18 考察: E10 domain_lora **採用** | `expert-mesh-{domain}-lora`（10 種） |
| `032a85b` | Iter19 考察: E8 **棄却** | `qwen3.5:4b-q4_K_M`（10 件） |
| HEAD | — | **`qwen3.5:4b-q4_K_M`（10 件のまま）** |

過去のイテレーションでは棄却したレバーを baseline に戻す運用が確立していた（backlog B5「dispatch_top_k は config.yaml でベースライン(1) に戻した」，B8「routing_method は交絡回避のため self_report に戻した」）．E8 についてのみ，これが実施されていない．

さらに `config.yaml` の 76〜82 行のコメントは「各ノードはドメイン固有の LoRA 統合モデル（`expert-mesh-{domain}-lora`）を使う」と記述しており，**コメントと実際の値が矛盾している**．

帰結: Iter20 / Iter21 / Iter22 の 3 実験はすべて棄却済み構成の上で実行された．軸①（ルーティング）は `expert_model` に依存しないため無傷だが，軸②③の報告値は最良既知構成より大きく下振れしている（answer_quality 0.218〜0.2313 対 0.5013，end_to_end 0.1257〜0.1355 対 0.3151）．

なお LoRA アダプタは `models/lora_adapters/` に 10 ドメイン分（`adapter.gguf` 含む）残っており，復帰は可能である．

### 6-D. Iter22 の「bug fix」は fix ではなく，レバーを 2 つ同時に動かす変更である

Iter21 で発見された bug（`self_consistency_semantic` に到達しない）に対し，Option A として「`confidence_signal_method` の分岐を `routing_method` より先に移動する」修正が適用され，HEAD（`30e3627`）に含まれている．

しかしこの修正の帰結は，§6-B に示した分岐構造から明らかである．`confidence_signal_method=self_consistency_semantic` を有効にすると 334 行で return するため，**364 行の supervised_classifier 分岐には到達せず，分類器は 1 度も呼ばれない**．

journal の Iter21/22 計画にある「`routing_method` は不変．supervised_classifier は confidence を特徴量の 1 つとして使うため，confidence の分布変化が routing に与える影響は限定的」という記述は事実と異なる．`estimate_confidence_classifier(classifier, domain, query_embedding)` は confidence を入力に取らず，`predict_proba([query_embedding])` の結果を返すだけである．両者は排他的な代替関係にある．

**したがって，現行の config のまま Iter22 の実験を実行すると，採用済みの E6（top1 0.2059→0.5651 をもたらした最大の成果）が無効化され，ルーティングが LLM ベースの semantic entropy に置き換わる．**「単一レバー」の想定に反して 2 つの機構が同時に変わるため，得られる結果は E4 の効果としては解釈できない．対処案は d0003 の X5 を参照．

### 6-E. データリークはない

`data/dataset.jsonl`（評価 1520 問）と `data/classifier_train.jsonl`（訓練 1427 問）の `query` 本文を照合した結果，**重複は 0 件**であった（評価側のユニーク本文 1519，訓練側 1427，共通 0）．

これは文献調査（d0001 §5.1）が「E6 は Iter10 の label leakage を再演する危険が最も高い」と警告していた点に対する反証であり，**top1_accuracy 0.5651 というルーティング精度の妥当性は担保されている**．

なお分類器のオフライン性能は訓練 100.00%（1427/1427）に対し評価 59.87%（898/1500）で，過学習の傾向自体は残っている．

### 6-F. 回答品質のノイズ床は約 1.3pt である

Iter20（`20260729_110720`）と Iter22（`20260729_190824`）は，ルーティングがビット単位で同一・`expert_model` も同一という，完全に同一の構成である．それにもかかわらず `answer_quality_accuracy` は **0.2313 と 0.2180（1.33pt 差）**であった．差の全ては生成の確率性（および judge の確率性）に由来する．

これは軸②③に関する初めての実測ノイズ推定である．含意:

- E10 の採用根拠（+22.3pt）はノイズ床の約 17 倍であり，判定は堅牢である．
- 一方 Iter19 で示された「モデル縮小の寄与 -4.1pt」（LoRA 撤去 -22.3pt との分解）はノイズ床の約 3 倍でしかなく，単一 run での分解は根拠が弱い．

### 6-G. 計測基盤の欠落

- **`metrics.py` に ECE・AUROC・Brier score が実装されていない．** 実装されているのは Wilson CI，McNemar，Cohen's kappa，Random / BestSingle / Oracle，複合ドメイン被覆指標である．`.claude/research/config.yml` の success_criteria (4) は「confidence 信号系のレバーでは accuracy に加えて ECE・AUROC・同点率・ノード間 confidence 分散を報告する」と定めているが，ECE と同点率は毎回その場限りのスクリプトで計算されており，§7 の記録誤りの原因になっている．
- **`data/` `results/` `models/` が `.gitignore` されている．** 評価データセット，全実験結果，LoRA アダプタ，分類器がいずれもバージョン管理外であり，再現性がローカルディスクのみに依存している．

---

## 7. journal の記録誤りと訂正

本調査で実データと突合した結果，以下の食い違いを確認した．いずれも実データ側を正とする．

### 7-1. ECE の系列

単一の実装（10-bin，`confidence` が非 null の行が対象）で全実験を再計算した結果．

| 実験 | 本調査の再計算 | journal の記載 | 備考 |
|---|---|---|---|
| Iter15 | 0.7146 | 0.7146 | 一致 |
| Iter16 | 0.7388 | 0.7388 | 一致 |
| Iter17 | **0.1927** | 0.2118 | **不一致** |
| Iter18 Phase C | **0.1927** | 0.1927 | 一致 |
| Iter20 | **0.1927** | 0.1927 | 一致 |
| Iter21 | **0.1927** | 0.1903 / 0.1673 | **不一致（journal 内でも 2 通り）** |
| Iter22 | **0.1927** | 0.1927 | 一致 |

**ECE は Iter17 以降まったく変化していない**（0.1927 で固定）．これは §6-A（決定論性）および §6-B（elicitation が no-op）から必然である．Iter21 の「0.1903 へわずかに改善」という記述は誤りで，実際は 0.0000 の変化である．

### 7-2. top1_accuracy と single_domain_top1_accuracy の取り違え

journal は Iter18 Phase C のベースラインを「top1_accuracy = 0.5693」として Iter19・Iter20 の計画に使用しているが，`metrics.py` で再現すると:

- `top1_accuracy` = **0.5651**
- `single_domain_top1_accuracy` = 0.5693

つまり 0.5693 は単一ドメイン 1500 問のみの値である．この取り違えにより「E10 で top1 が 0.5651→0.5693（+0.0042）改善した」という記述が生じているが，実際には McNemar 不一致対 0/1520 で**完全に不変**である（journal 自身も別の箇所で不一致 0/1520 と記録しており，内部矛盾している）．

### 7-3. その他，journal / backlog 内に既にある不整合

- backlog B36 は次のレバーを `multi_sample_semantic` と記すが，B37 および `config.yml` の E4 定義は `self_consistency_semantic` である．
- backlog B35 は「Iter16 は n=46 の評価集合」と記すが，B31 および実データは 1520 問である（B31 が正しい）．
- アーカイブ内の Iter13 は top1 が 0.043 / 0.0652 / 0.065，Iter6 は single_top1 が 0.9000 と 0.925 の 2 通りで記載されている（分析タイミングごとの揺れ）．
- Iter17 の次計画に「現在 4 ノードすべてが同一モデル（`isotnek/qwen3.5:9B...`）」という旧構成の記述が残っている．
- README 内で，`evaluation.py` を「評価軸②③として実装済み」と書く箇所（201-209 行）と「評価軸②③は未実装」と書く箇所（507・530 行）が矛盾している．
- JMMLU のライセンス表記が README（CC BY-NC-ND 4.0）と d0001 / p0001（CC BY-SA 4.0 中心）で食い違う．実データで確認された CC BY-NC-ND 4.0 が正しい．

---

## 8. 未解決の論点

1. **中央集権ルータとの比較（設計書 §4.2(b)）が未実施．** 研究の問い 2 の主要比較対象であり，主張の核心が空白である．
2. **複合ドメイン評価が成立していない．** `dispatch_top_k=1` で上限 0.5，実測 set_recall 0.125．設問数も 1520 問中 20 問（1.3%）と少ない．研究の問い 3 に答えられない．
3. **fallback が有害である．** 13.16%（200/1520）が general にフォールバックし，その正解率 8.0%（16/200）は Random の 10.1% を下回る．フォールバック先は 200/200 すべて general．
4. **education / legal のデータ不均衡．** legal は JMMLU に対応タスクがなく訓練 77 件，education は代理タスク．education の precision/recall は 0.520/0.411 で最下位圏，Iter17 で唯一の非退行違反を出した．
5. **softmax 確率の過信が残る．** 全ドメインで mean_confidence > accuracy．ECE 0.1927 のうち [0.90, 1.00) バケットの gap が 0.1750 を占める．`CalibratedClassifierCV` 等による較正は未検討．
6. **LLM-as-judge の mean_score が未取得．** Iter18 でノード busy によりタイムアウトした．複合設問 20 問が軸③で不正解扱いになっている．
7. **未実施のレバー**: E4（`self_consistency_semantic`，§6-D の再設計が前提），E5（`p_true`，Ollama v0.12.11+ の実機確認が未了），E3 の `linguistic`，E10 の `offtheshelf_specialized`（日本語の法律特化オープン生成モデルが調査で発見できず実施不能）．
8. **nomic-embed-text の task prefix 未付与**（backlog B7）．Iter2 の embedding 劣化の主因かどうかの切り分けが未実施．現行の supervised_classifier 経路でも生の埋め込みを使っているため，prefix 付与で分類精度が変わる可能性は残る．
9. **hidden_state 信号**（Iter14）．Ollama では取得不可．vLLM / SGLang への移行が前提となる．

---

## 9. 参照

### 内部文書

- `docs/encounter_expert_mesh_design.md` — 技術設計書 v2．§2.2 ノード構成，§4.1 評価軸，§4.2 ベースライン，§7 研究の問い．
- `docs/d0001_literature_survey_2026-07.md` — 文献調査（2026-07）．F1〜F5 の発見とレバー E1〜E7 の根拠．
- `plans/p0001_research_direction_2026-07.md` — 研究の方向性．新規性の所在と推奨順序．
- `docs/d0003_next_experiments_2026-07.md` — **本ファイルの続き．次に行うべき実験・実装修正**．
- `.claude/research/journal.md` / `journal_archive.md` — イテレーション別の一次記録．
- `.claude/research/backlog.md` — 自動判断の記録（B1〜B38）．
- `.claude/research/config.yml` — レバー定義（E1〜E10）と成功条件．

### 主要な外部文献

| 出典 | 本研究との関係 |
|---|---|
| Tian et al., EMNLP 2023 (arXiv:2305.14975) | Verbalized Top-K．gpt-3.5 の ECE を 0.131→0.047．E3 の根拠．本研究では ECE 改善を再現できなかった |
| Farquhar et al., Nature 630:625-630 (2024) | Discrete Semantic Entropy．E4 の根拠．T=0.7〜1.0 での sampling を要求 |
| Xiong et al., ICLR 2024 (arXiv:2306.13063) | Monte Carlo Temperature．T=0.7 推奨．Iter11 の T=0.1 が設計欠陥だった根拠 |
| Kadavath et al., 2022 (arXiv:2207.05221) | P(True)．E5 の根拠．ただし 52B base + 20-shot 設定での結果 |
| Su et al., 2021 (arXiv:2103.15316) | BERT-whitening．E7 の根拠．supervised_classifier 下では no-op |
| Varangot-Reille et al., JAIR 2025 (arXiv:2502.00409) | similarity-based routing の失敗は unsupervised であることに起因 |
| RouterDC, NeurIPS 2024 (arXiv:2409.19886) | CosineClassifier に全タスクで勝利．E6 の根拠 |
| MoDEM (arXiv:2410.07490) | ルータ精度 81.00%，Other クラス 52.94%．general がボトルネックになる構造の先行例 |
| RouterEval (arXiv:2503.10657) | 候補数 2≤m≤10 でルータ性能の伸びが最も急．E9 の根拠 |
| LLMRouterBench (arXiv:2601.07206) | 複数の最近手法が単純ベースラインを安定して上回れない．BestSingle 比較が必須である根拠 |
| S-LoRA, MLSys 2024 / JMedLoRA (arXiv:2310.10083) | 多数 LoRA アダプタの同時配信．E10 の根拠 |
| Internet of Agents survey (arXiv:2505.07176) | self-reported capability declarations の不正確さ．本研究の主張の位置付け |
