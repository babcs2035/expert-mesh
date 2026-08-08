<!-- research-cycle Iteration 54 収束後，education_recall改善以外の次の研究方向性をtavily-searchで調査した結果の総括．config.ymlのレバー追加の根拠文書． -->

# d0007: 次の研究方向性の調査（Iter54 収束後，2026-08-08）

## このファイルについて

- **目的**: `d0006` §7（backlog B84）で research-cycle が `status="converged"` に到達し，
  education ドメインの分類精度改善はこれ以上進めないことが確定した後，**次に何を検討・実験すべきか**
  を tavily-search（`tavily-search` / `tavily-dynamic-search` / `tavily-research` skill）で調査し，
  実行可能性を評価した総括である．
- **役割**: 本書の結論は `.claude/research/config.yml` の `levers` および `research_frontier` に反映済み．
  次の research-cycle iteration（Iter55）の investigate/plan フェーズは，本書を最初に読むこと．
- **調査方法**: README「既知の制約と今後の課題」・`d0004` §5「着手しない項目」・
  `research_frontier`（未完了項目）から，education_recall 以外でまだ手つかずの課題を先に洗い出し
  （§1），それぞれについて 4 件の並列調査を実施した（§2〜§5）．
- **記載の原則**: 本書は「何を調べたか・何が分かったか」までを記録する．実装計画（変更ファイル一覧・
  成功条件の確定）は次の research-cycle の計画フェーズ（rc-planner）の責務であり，本書では扱わない．

---

## 1. 調査に先立つ現状整理 — 何が手つかずで残っているか

`README.md`「既知の制約と今後の課題」・`docs/d0004` §5「着手しない項目」・`.claude/research/config.yml`
の `research_frontier`（5 項目中 4 項目は既に完了済み）を確認し，education_recall 以外で残っている
課題を次のように整理した．

| 課題 | 出典 | 現状 |
|---|---|---|
| 複合ドメイン質問で回答が中国語になる | README §既知の制約 2 | **未着手**．原因は qwen3.5 の多言語特性と推測されるのみ |
| `compound_domain_set_recall` が理論上限 1.0 に対し 0.345 で頭打ち | `aggregation_method` レバー note（Iter48 でクローズ） | 3 方式（max_confidence/majority_vote/llm_judge）を試し切ったが，いずれも 0.345〜0.360 止まり |
| 日本語法律特化の生成モデルが存在しない（`offtheshelf_specialized`） | `expert_specialization` レバー note（2026-07-30 時点の調査） | 2026-07 時点で「見つからない」．2026-08 時点で再調査していない |
| hidden_state ベースの手法 | d0004 §5「着手しない項目」 | Ollama では取得不可．vLLM/SGLang 移行が前提でスコープ外のまま |
| 無線アドホック化（Phase 3） | 設計書・d0004 §5 | 明示的にスコープ外 |
| McNemar 計算の実装者/analyst 不一致 | d0006 §6 | 恒久対策（チェックリスト）未実施．研究テーマではなくインフラ改善 |

このうち上位 3 件（回答言語一貫性・複合ドメイン recall・法律/医療特化モデル）と，education_recall とは
独立した「次の研究の柱」を探る 1 件を，tavily-search による調査対象として選定した．

---

## 2. 調査①: 複合ドメイン回答が中国語になる問題

### 調べたこと

Qwen 系・多言語 LLM が意図しない言語（特に中国語）を出力する既知の原因と，その対策手法．

### 分かったこと

原因は 3 点に整理できる．

1. **学習データの言語比率**: 専門用語について中国語表現の方が学習データ中の出現頻度・トークン効率
   （確信度）が高く，モデルが「その言語の方が答えやすい」と判断しうる．
2. **RLHF/RLVR が言語一貫性を報酬に含めない**: 検証可能報酬による強化学習は「正解率」のみを最適化し，
   thinking 過程での言語混在を抑制する報酬設計になっていない（ACL 2025, "Language Mixing in
   Reasoning Language Models"）．
3. **弱い言語指示は上書きされやすい**: システムプロンプトの言語指示が "Prefer Japanese" のような
   弱い表現だと，注入されたコンテキストやユーザー入力側の言語に上書きされる（QwenLM/qwen-code
   Issue #2003）．

対策として実績があるのは次の 2 つで，いずれもモデルの再学習を要しない．

- **(a) システムプロンプトの強化**: 「必ず日本語で応答し，他言語（特に中国語）を一切含めない」という
  強い指示に書き換える．
- **(b) 出力後の言語検出＋再生成**: `langdetect` 等で生成結果の言語を判定し，日本語でなければ
  再生成する後処理層を追加する．

（c）学習時の言語一貫性報酬や制約付きデコーディングは効果が大きいと推測されるが，モデルの再学習・
差し替えを要し高コストである．なお，言語混在を完全に排除すると推論精度がわずかに下がる場合がある
という報告もあり，過度な制約は避けるべきである．

### 実行可能性評価

| 観点 | 評価 |
|---|---|
| 実行コスト | 低（(a) はプロンプト文字列の変更のみ，(b) はアプリ層の後処理追加のみ．いずれも実機実行は軽い） |
| 単一レバー原則との相性 | 良い．ルーティングロジック（confidence 算出・dispatch 判定）に影響を与えず，
  この課題単独で A/B 検証できる |
| public API / 設定ファイルへの影響 | (a) は `http_server.py` の `build_dispatch_prompt()` 内の
  文言変更のみで `config.yaml` のスキーマ変更を伴わない可能性が高い．(b) を config 化する場合は
  新規フィールド追加を要する |

### 出典

- https://github.com/QwenLM/qwen-code/issues/2003
- https://aclanthology.org/2025.emnlp-main.132.pdf（Language Mixing in Reasoning Language Models）
- https://aclanthology.org/2025.emnlp-main.1654.pdf（The Impact of Language Mixing on Bilingual LLM Reasoning）

---

## 3. 調査②: 複合ドメイン `compound_domain_set_recall` の改善

### 調べたこと

各ノードが 10 クラス softmax の自分のクラスの確率のみを返す現行設計の下で，2 位ノードの confidence
が構造的に 0.5 弱を超えられない問題（`aggregation_method` レバー note 参照）に対する，multi-label
classification・MoE 分野の定石．

### 分かったこと

- **multi-label 分類の定石**は，softmax の単一分布ではなく各クラス独立の sigmoid（binary relevance）
  で confidence を出す設計である．確率総和 1 の制約を外せば構造的な上限は解消するが，**分類器の
  出力層・損失関数の再訓練を要する**ため，単一レバー原則（argmax flip rate <15%）との相性は悪い．
  Iter40〜43（embedding 適応）で確認済みの「retraining は単一レバー原則と両立しない」という知見と
  同じ制約に直面する．
- **MoE 分野の adaptive gating**（Huang et al., EMNLP 2023）と **Expert Choice Routing**
  （Google Research）は，1 位・2 位の confidence 差（gap）が閾値未満のときだけ 2 位以降も採用する，
  可変個の専門家への dispatch を確立済みの手法として持つ．これは**既存の分類器を変えず，dispatch 判定
  ロジックだけを変える後付けの閾値判定**であり，argmax（各ノードの 1 位選択）自体は変えない．
- RouterBench・RouterEval・LLMRouterBench（2026年1月時点の統合版）を確認したが，複合ドメイン
  recall を直接の指標とする定石は見当たらなかった．

### 実行可能性評価

固定 `dispatch_top_k=2` を，confidence の 1 位・2 位の差（gap）が閾値 `T` 未満のときだけ動的に
k を増やす **adaptive-k dispatch** への変更が最有力の低コスト案である．

| 観点 | 評価 |
|---|---|
| 実行コスト | 中．`aggregator.select_dispatch_targets()` の dispatch 判定ロジック変更のみで，
  分類器の再訓練は不要 |
| 単一レバー原則との相性 | 良い．argmax は変えないため，既存の単一レバー原則の判定基準にそのまま適合する |
| public API / 設定ファイルへの影響 | **`dispatch_top_k` を固定値運用から動的ポリシーへ変更するため，
  `config.yaml` のスキーマ変更を伴う可能性が高い．Y2（`dispatch_candidate_threshold` 新設）と同様，
  着手前にユーザー確認が必要**（CLAUDE.md の規約） |

### 出典

- Adaptive Gating in Mixture-of-Experts based Language Models（EMNLP 2023）: https://arxiv.org/html/2310.07188
- Mixture-of-Experts with Expert Choice Routing（Google Research）: https://research.google/blog/mixture-of-experts-with-expert-choice-routing
- RouterBench: https://arxiv.org/html/2403.12031v2
- Multi-label classification（binary relevance の定義）: https://en.wikipedia.org/wiki/Multi-label_classification

---

## 4. 調査③: 日本語法律/医療特化オープンウェイト生成モデルの最新動向

### 調べたこと

`expert_specialization=offtheshelf_specialized`（ノードごとに専門特化モデルを使う構想）が
2026-07 時点で「日本語法律特化オープンウェイト生成モデルが見つからない」として保留されていたため，
2026-08 時点での再調査．

### 分かったこと

- **医療分野**: **Medical-Qwen3-Swallow-8B**（Swallow プロジェクト，Qwen3 8B ベースに医学論文・
  診療ガイドラインで継続事前学習，Apache 2.0，2026年公開）が，既存構成（qwen3.5:9b，CPU 推論）と
  同等以下のサイズで導入可能な候補として見つかった．より大型の Medical-Qwen3-Swallow-32B/30B-A3B
  や，NEDO 支援の Weblab-MedLLM シリーズ（120B〜355B，医師国家試験正答率 95.9%）も存在するが，
  いずれも 9B を超え本構成には重すぎる．
- **法律分野**: `llm-jp/awesome-japanese-llm` のドメイン特化型一覧を含め，日本語法律特化かつ
  生成可能なオープンウェイトモデルは**依然として見つからなかった**．karasu-7B-chat-plus
  （Lightblue, Mistral-7B ベース）が学習コーパスに日本の法律・判例を含むが，汎用モデルの学習
  データの一部であり法律特化モデルではない．2026年時点の商用サービス比較でも，法律業務向けに
  推奨されるのは DeepSeek-R1 や Qwen3-235B-A22B のような汎用大型モデルである．

### 実行可能性評価

| 観点 | 評価 |
|---|---|
| 医療ノードへの `Medical-Qwen3-Swallow-8B` 導入 | 中コスト．既存の `expert_model` を差し替え，
  medical ドメインの recall・回答品質を再評価する実機実験が必要．ライセンス・ベンチマーク数値の
  一次ソース（HuggingFace モデルカード）は未取得のため，着手前に確認が要る |
| 法律ノード | 現状維持が妥当．既存の qwen3.5:9b + ドメイン LoRA 構成を継続する |

### 出典

- https://github.com/llm-jp/awesome-japanese-llm
- https://huggingface.co/weblab-LLM-M/Weblab-MedLLM-GLM-4.7
- https://huggingface.co/weblab-LLM-M/AscleLM-1-10B
- https://huggingface.co/pfnet/Llama3-Preferred-MedSwallow-70B
- https://monoist.itmedia.co.jp/mn/articles/2606/15/news070.html（NEDO 医療特化 LLM，2026年6月）
- https://www.siliconflow.com/articles/ja/best-open-source-LLM-for-legal-industry

---

## 5. 調査④: LLM ルーティング分野の2025〜2026年最新動向（education_recall 改善とは独立な新テーマ）

### 調べたこと

分散/エッジ協調推論・confidence calibration・適応的ルーティング・不安定通信下のルーティングの
4 つの切り口で，education_recall 改善とは独立した次の研究テーマの候補を探索した．

### 分かったこと

| 切り口 | 手法・論文 | 概要 | 実行コスト |
|---|---|---|---|
| (a) 分散/エッジ協調推論 | Decentralized Speculative Decoding（Parallax），Intel+Weizmann のクロスアーキテクチャ投機的デコーディング（ICML 2025） | 通信遅延を並列トークン検証に転用し最大2.8倍高速化．expert-mesh の各ノードを draft/verify 役に分ける協調推論への応用余地 | オフラインシミュレーション中心，実機は軽い |
| (b) confidence calibration | CP-Router（conformal prediction，被覆保証付きルーティング），Conformal Arbitrage（NeurIPS 2025） | post-hoc 閾値調整の「境界平行移動」を，統計的被覆保証付きの意思決定に置き換える | **最低**．既存 confidence 値の再利用でオフライン完結 |
| (c) 適応的ルーティング | OrcaRouter（LinUCB バンディットでのオンライン継続適応），BaRP | dispatch 結果の正誤フィードバックでルーティングポリシーを継続改善 | 中〜高．フィードバック記録の実装＋実運用ログでの実機実験 |
| (d) 不安定通信下の協調推論 | Mesh LLM（iroh gossip protocol，ノード離脱対応），PEFT-CE（NAIRR，切断下での早期終了 PEFT） | 有線 LAN 前提を緩め，タイムアウト・再試行・ローカル代替経路を設計 | 高．ノード切断シミュレーションを含む実機実験＋フォールバック経路の追加実装 |

### 実行可能性評価

(b) の conformal prediction 応用が最も低コストで，既存の `classifier_calibration`（temperature
較正，Iter31 adopted）とは異なる統計的性質（被覆保証）を持つため，較正手法の新しい軸として
検討価値がある．(a) はレイテンシ改善という別の評価軸を持ち込む．(c)(d) は実装変更が大きく，
research_frontier（大規模変更）として位置づける方が適切である．

### 出典

- CP-Router: https://arxiv.org/html/2603.04445v1
- Conformal Arbitrage（NeurIPS 2025）: https://proceedings.neurips.cc/paper_files/paper/2025/file/65a655c5a267f678fd3e897e4137ef53-Paper-Conference.pdf
- OrcaRouter: https://arxiv.org/html/2605.30736v1
- Decentralized Speculative Decoding（Parallax）: https://arxiv.org/html/2604.17227v1
- Mesh LLM（iroh gossip protocol）: https://micrologics.org/blog/architecting-mesh-llms-decentralized-peer-to-peer-ai-inference-with-iroh

---

## 6. 優先順位と推奨する次のレバー

実行コスト（低いほど優先）と単一レバー原則との相性（良いほど優先）で 4 候補を並べると次のようになる．

| 優先度 | 候補 | コスト | 単一レバー適合性 | config 変更 |
|---|---|---|---|---|
| **1** | 回答言語一貫性（§2） | 低 | 良い | 不要（プロンプト変更のみ）〜小 |
| **2** | conformal prediction 較正（§5-b） | 低 | 良い（既存 confidence 再利用） | 小 |
| 3 | adaptive-k dispatch（§3） | 中 | 良い（再訓練不要） | **要ユーザー確認**（Y2 相当） |
| research_frontier | 医療特化モデル導入（§4） | 中 | — | 実機評価が必要 |
| research_frontier | 分散協調推論／適応的ルーティング／不安定通信対応（§5 a,c,d） | 中〜高 | — | 大規模変更 |

**Iter55 で最初に着手するレバーとして「回答言語一貫性」を推奨する**．理由は次の3点である．

1. education_recall のように post-hoc 手法の理論的限界に到達する懸念がなく，「日本語で応答したか」
   という二値の成功条件を設定できるため，判定が単純である．
2. ルーティング精度（confidence・分類器）に一切触れないため，これまでの 54 イテレーションで
   確立した最良構成（`classifier_calibration=temperature` 等）を一切揺らさない．
3. 実機実験なしで（既存の複合ドメイン設問の回答文をオフラインで言語判定するだけで）効果を検証できる
   可能性が高く，着手コストが最も低い．

---

## 7. 関連文書

| 文書 | 対象範囲 | 位置づけ |
|---|---|---|
| `docs/encounter_expert_mesh_design.md` | 全期間 | 技術設計書（v2） |
| `docs/d0006_research_summary_iter28-54_2026-08.md` | Iter28〜54 | 収束時点の総括．本書はその後続 |
| **本書 `d0007`** | **Iter54 収束後** | **次の研究方向性の調査．`.claude/research/config.yml` の levers/research_frontier 追加の根拠** |
