<!-- ユーザーが2026-09-23に恒久指示した研究方針転換（ドメイン固有補正の全廃・人間棄権の完全廃止）を
     受けて，全自動でtop1_accuracyを上げる次の一手をtavily-searchで調査した結果の総括．
     .claude/research/config.yml のlevers/research_frontier追加・backlog B115の根拠文書． -->

# d0008: 研究方針の恒久転換と次の一手の調査（2026-09-23）

## このファイルについて

- **目的**: ユーザーが会話上で次の 3 点を恒久方針として指示した．(1) 苦手分野固有の後付け補正は
  今後加えない，(2) ルーティングにおいて人間に頼る（棄権・エスカレーションする）方針は完全に
  廃止する，(3) その上でトータルの正答率（top1_accuracy）向上を追求する．この指示を受け，
  全自動での正答率向上に使える 2025〜2026 年の最新手法を tavily-search で調査した総括である．
- **役割**: 本書の結論は `.claude/research/config.yml` の `levers` / `research_frontier` と
  `backlog.md` B115 に反映済み．次の research-cycle iteration の調査フェーズは，本書と
  `docs/d0007` を合わせて読むこと．d0007（Iter54 収束後）はルーティング精度そのものではなく
  周辺課題（回答言語一貫性・複合 recall・特化モデル）を対象にしていたのに対し，本書は
  **top1_accuracy 自体を全自動で上げる**ことに焦点を絞っている点が異なる．

---

## 1. 新方針が既存の構成に与える影響

Iteration 1〜76 の間に，education ドメイン固有の後付け補正が複数本線へ採用されていた．

| 補正 | 採用時期 | 内容 |
|---|---|---|
| `intercept_delta=+0.7` | Iter44 adopted | 訓練時，education の decision boundary を調整 |
| `education_threshold=0.05` | Iter52/53 adopted | 推論時，education の確率へ後付け加算（backlog B88 により実行時経路への反映漏れが判明済み，現状は実質未反映） |
| `education_proxy_task_*` 系 | Iter32〜37，ほとんど rejected | 訓練データ側の education 固有調整．大半は不採用のため撤去対象は実質上記 2 点 |

ユーザーは「既存分も撤去して 10 ドメイン均一な扱いへ戻す」ことを選択した．これにより，
Iteration 77 で調査フェーズに入ったまま停止していた `routing_abstention_scorer=learned_deferral_head`
（棄権前提のレバー）は目的を失い着手しないことが確定した．conformal 系列（Iter69〜76）が得た
「被覆保証は厳密に得られる」という成果自体は無効にならないが，実行時経路への配線（棄権用途）は
今後検討しない．

---

## 2. 調査①: LLM ルーティング分類器の精度向上手法（2025〜2026年動向）

### 分かったこと

- **Semantic Router（vLLM 向け，2025）**: クエリの reasoning 要否を分類するルータで，MMLU-Pro 上で
  +10.24pt の改善を報告．ただし**ドメインによって改善幅が大きく異なる**（business/economics で
  +20pt 超，一方 engineering・computer science のような技術領域では改善が乏しい）．これは
  本研究の education ドメインだけが弱いという構造と類似する現象であり，「単一の手法が全ドメインへ
  一様に効くわけではない」ことの傍証になる．
- **アンサンブル手法（ICE・eLLM・LLM-Synergy 等，2025〜2026）**: 複数モデルの合議で 7〜15pt の
  改善例が報告されている．ただし，これは「同じ質問を複数モデルに投げて答えを統合する」話であり，
  本研究では `aggregation_method`（Iter46〜48）としてすでに試し，`compound_domain_set_recall` が
  0.345〜0.360 で頭打ちと確定済み．**まだ試していないのは「ルーティング判定そのもの」を複数の
  分類器（埋め込みモデル違い等）の多数決にすること**で，これは今回 research_frontier 相当として
  保留した（レイテンシとのトレードオフという新しい評価軸が要るため）．
- **RACER（Risk-Aware Calibrated Efficient Routing，2026）**: 信頼度に応じた重み付き集約が単純な
  多数決を安定して上回ると報告．ただし本研究はすでに confidence 較正（Iter31，temperature scaling
  adopted）を導入済みで，この知見の中心部分は反映済みと考えられる．

### 出典

- Semantic Router for vLLM: https://arxiv.org/html/2510.08731v1
- Intelligent LLM Routing（アンサンブル手法まとめ）: https://www.swfte.com/blog/intelligent-llm-routing-multi-model-ai
- RACER: https://arxiv.org/html/2603.06616
- A Massive Benchmark and Unified Framework for LLM Routing: https://arxiv.org/html/2601.07206v1

---

## 3. 調査②: 埋め込みモデルの選定（全ドメイン一律に効くレバー候補）

### 分かったこと

現行の `nomic-embed-text`（768 次元）は 2026 年時点の MTEB multilingual ランキングで
62.28 点．Ollama で配布されている既製の多言語埋め込みモデルには，これを上回る選択肢が複数ある．

| モデル | サイズ | MTEB multilingual | 備考 |
|---|---|---|---|
| `nomic-embed-text`（現行） | 274MB | 62.28 | — |
| `qwen3-embedding:0.6b` | 639MB | **64.33** | 現行比 +2.05pt．CPU でも動作可能 |
| `embeddinggemma` | 622MB | 61.15 | 現行と同水準 |
| `qwen3-embedding:4b` | — | **69.45** | 現行比 +7.17pt．GPU 必須 |
| `qwen3-embedding:8b` | 4.7GB（Q4）/ 15GB（FP16） | **70.58**（release 時点 #1） | GPU 必須，VRAM 要件が重い |

**重要な区別**: Iter39〜43 で rejected と確定した `embedding_adaptation`（SetFit / LoRA による
fine-tuning）は「既存モデルを再学習すること」が原因で argmax flip rate が過大になった．
今回の候補は**再学習ではなく既製モデルへの丸ごと差し替え**であり，別のレバーである．また
「全ドメイン一律に効く」ため，ドメイン固有の後付け補正には該当しない．

### GPU 制約（wafl-ctrl5: RTX 3060 12GB）を踏まえた絞り込み

ユーザーは「12GB で動く小型モデルに絞って調査する」ことを選択した．`qwen3-embedding:8b` は
FP16 で 15GB を要求するため対象外とする．**`qwen3-embedding:0.6b`（リスク最小）と
`qwen3-embedding:4b`（改善幅最大，12GB に収まる見込み）の 2 値を次のレバー候補とする**．
ただし wafl-ctrl5 は 2026-09-19 のユーザー指示により合成データ生成用の Ollama とも同居しているため，
VRAM 競合は実機で確認する必要がある．

### 出典

- Best Ollama Embedding Models 2026（MTEB スコア・VRAM 比較）: https://www.morphllm.com/ollama-embedding-models
- Embedding Models 2026: Benchmark and Comparison: https://app.ailog.fr/en/blog/news/embedding-models-2026
- Best LLM/Embedding for 12GB VRAM（2026）: https://localaimaster.com/vram/best-llm-12gb-vram

---

## 4. 調査③: 全ドメイン共通ルールでの訓練データ拡充

### 分かったこと

- **Hard negative mining**: 分類器が誤りやすい行（クラス境界に近い行）を優先的に訓練データへ
  追加する定石．少量データでの分類精度改善に有効という報告が複数ある．
- **Synthetic feature augmentation**: 少数クラスを対象にした特徴量レベルの合成拡張が，
  不均衡データでの汎化性能を改善するという報告（2025年1月）．

いずれも「特定ドメインだけ」を対象にすると，Iter32 で判明した機序（`LogisticRegression(class_weight=
'balanced')` は `sample_weight` と数式レベルで結合しており，1 ドメインの重みを動かすと
他ドメインへ副作用が波及する）に抵触するリスクがある．**新方針（ドメイン固有補正の禁止）とも
整合させるため，全 10 ドメインへ同一パラメータで同時に適用する設計に限定する**必要がある．

### 出典

- Enhancing Text Datasets With Scaling and Targeting Data Augmentation（2025年5月）: https://www.sciencedirect.com/science/article/pii/S0957417425017713
- Synthetic Feature Augmentation Improves Generalization Performance of Language Models（2025年1月）: https://arxiv.org/abs/2501.06434v1
- Hard Negative Mining（実装ガイド）: https://medium.com/@sundardell955/hard-negative-mining-91b5792259c5

---

## 5. 優先順位とユーザー決定

| 優先度 | 候補 | 新方針との整合性 | ユーザー決定 |
|---|---|---|---|
| **1** | 複合設問評価集合（n=100）の拡充 | 人間棄権とは無関係な測定系の課題．制約を受けない | 承認．具体化は次の調査フェーズへ |
| 2 | 埋め込みモデルの差し替え（`qwen3-embedding:0.6b`/`4b`） | 全ドメイン一律．ドメイン固有補正に該当しない | 承認．12GB で動く小型モデルに絞って調査 |
| 3 | 全ドメイン共通ルールでの訓練データ拡充（hard negative mining） | 全ドメイン同時適用に限定すれば整合 | 承認 |
| 見送り | ルーティング判定のアンサンブル化 | 新方針とは矛盾しないが実装コスト中〜高 | research_frontier 相当として保留 |
| クローズ | `routing_abstention_signal` / `routing_abstention_scorer` 系 | 人間棄権の完全廃止に抵触 | 着手しない（B104 A2 完全クローズ） |
| 実行確定 | `education_specific_correction_removal`（既存 education 固有補正の撤去） | 新方針そのもの | 実行．撤去後の非退行確認が必要 |

---

## 5.5 追記（2026-09-23，backlog B116）— 3 点の確定と 2 つの絶対条件

本書 §5 で「ユーザー未決定」としていた 3 点に，同日中に追加の指示があり確定した．

1. **`education_specific_correction_removal` は撤去後に top1_accuracy が悪化しても維持する**．
   精度悪化を理由に education だけを特別扱いする方向へは戻さない．弱いドメインへの手当てを
   検討する場合は，10 ドメイン共通の枠組み（`cross_domain_training_data_augmentation` 等）に
   限定する．
2. **データセット拡充の調達順位を確定**: (i) まずインターネット上の信頼性のある既存データセットを
   調査する（tavily-search 等），(ii) 見つからない場合に限り LLM 生成へ切り替える，(iii) 人手作成は
   (i)(ii) が両方不成立の場合の最終手段とする．この順位は複合設問評価集合の拡充に限らず，
   今後データセットを拡充する場面全般に適用する．
3. **`embedding_model_replacement` は自律着手してよい**（config.yaml のスキーマ変更を伴うが，
   本レバーに限り着手前の追加ユーザー確認は不要）．

加えて，研究サイクル全体に適用する絶対条件を 2 点指示された．

- **(A) 何らかの施策・変更を適用したら，wafl500〜509 を用いた 1600 問のフルスペック本実験を
  必ず実施し，指標を計測・評価し続けること**．オフラインの事前シミュレーションや一部ノードだけの
  簡易確認のみで判定を確定させ，本実験を省略してはならない．
- **(B) 単一 GPU で完結するサブ実験は，wafl500 を含む実験ドメインノード（wafl500〜509）を
  絶対に使わず，必ず wafl-ctrl5 を用いること**．wafl500 は「依頼者兼任」ノードのため誤って
  使われやすく，明示的に禁止する．

詳細は backlog B116・`.claude/research/config.yml` 冒頭の恒久ルールを参照．

## 6. 関連文書

| 文書 | 対象範囲 | 位置づけ |
|---|---|---|
| `docs/d0007_next_research_directions_2026-08.md` | Iter54 収束後 | 周辺課題（言語一貫性・複合recall・特化モデル）の調査．本書とは焦点が異なる |
| **本書 `d0008`** | **2026-09-23 方針転換後** | **top1_accuracy 全自動向上のための次の一手．`.claude/research/config.yml` の levers/research_frontier 追加・backlog B115 の根拠** |
