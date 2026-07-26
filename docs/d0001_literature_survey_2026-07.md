# d0001: expert-mesh 文献調査・実測記録（2026-07-26）

Iter1〜14 の結果を再検討するために実施した文献調査（tavily）と，リポジトリに対する実測の**完全な記録**である．
意思決定と提案は `plans/p0001_research_direction_2026-07.md` にまとめてあり，本ファイルはその根拠となる
一次情報を出典付きで残すことを目的とする．

**表記規約**: 「論文は〜と報告している」は出典の主張，「〜と考えられる／推測される」は本調査での解釈である．

---

## 第 1 部: リポジトリに対する実測（文献ではなく本調査の測定結果）

### 1.1 評価データセットの実測: 46 問しかない

`data/dataset.jsonl` を直接集計した．キーは `["expected_domains", "id", "is_compound", "query"]`．

| 区分 | 件数 |
|---|---:|
| **総数** | **46** |
| 単一ドメイン | 40（medical 10 / legal 10 / general 10 / education 10） |
| 複合ドメイン | 6（medical+legal 4 / education+medical 1 / education+legal 1） |

`results/*/results.jsonl` の行数も 46（一部の失敗実行は 17 行）で一致する．

### 1.2 主要な判定を問題数に還元する

| イテレーション | 報告値 | 実際の問題数 | 差 |
|---|---|---|---|
| ベースライン | top1 0.8696 | **40 / 46** | — |
| Iter10（calibrated routing） | 0.870 → 0.848 | 40 → **39** / 46 | **1 問** |
| Iter11（multi_sample） | 0.870 → 0.848 | 40 → **39** / 46 | **1 問** |
| Iter11 misrouting | 0.130 → 0.152 | 6 → **7** / 46 | **1 問** |
| single_domain_top1 | 0.8750 | 35 / 40 | — |
| Iter13（STP） | 0.870 → 0.065 | 40 → **3** / 46 | 37 問 |

**統計量**

- p = 0.87, n = 46 の二項標準誤差: SE = √(0.87×0.13/46) = **0.0496（±5.0pt）**
- Wilson 95% CI（40/46）: **[74.3%, 93.9%]**（幅 約 19.5pt）
- 単一ドメイン（35/40, p=0.875）: SE = **±5.2pt**
- **ドメイン別指標は 1 ドメイン 10 問**しかない．p=0.9 での SE = √(0.9×0.1/10) = **±9.5pt**

2.2pt の変化は 0.44 SE にすぎず，ノイズと区別できない．
Iter7 の「education precision 0.90 → 0.909」や Iter9 の「recall 0.833 → 0.5」は，
**1〜2 問の入れ替わりに相当する量**であり，手法の優劣を論じられる解像度ではない．

**結論: Iter3 および Iter5〜11 の「no-op」「僅差で棄却」という判定群は，レバーが効かなかったことの
証明ではなく，評価集合が小さすぎて差を検出できなかったことの表れである可能性が高い．**
一方，Iter2（0.971 → 0.529）と Iter13（0.870 → 0.065）は効果量が十分大きく実在する効果である．

### 1.3 Iter13 の 0.065 は偶然一致を大きく下回る

4 ドメインでのランダム選択の期待値は 0.25（46 問中 11.5 問）．STP の 0.065（**3/46**）は
sd = √(46×0.25×0.75) = 2.94 として (3 − 11.5)/2.94 = **−2.9 SD**．
**偶然より systematically に悪い**という結果は，信号が無効であることよりも，
順位付けの符号が反転している（またはスケーリングが逆になっている）ことを強く示唆する
（これは文献ではなく数値からの推論である）．

検証コストは極めて低い: 保存済み `results.jsonl` の confidence 列を符号反転して top1 を再計算するだけ．

### 1.4 【最重要】全ノードが同一モデルであり「専門家」の実体がない

`config.yaml` の `nodes` セクション:

| ノード | domain | light_model | expert_model |
|---|---|---|---|
| wafl500 | general | `isotnek/qwen3.5:9B-Unsloth-UD-Q4_K_XL` | 同左 |
| wafl501 | education | 同上 | 同左 |
| wafl502 | legal | 同上 | 同左 |
| wafl503 | medical | 同上 | 同左 |

**4 ノードすべてが light/expert とも完全に同一のモデル**である．ノード間の差分は次の 2 箇所のみ．

- `router.py:56`: `あなたは「{domain}」分野の専門家ノードです．`
- `http_server.py:66`: `あなたは「{domain}」分野の専門家です．次の質問に，あなたの専門知識を活かして具体的に回答してください．`

これは設計書 `docs/encounter_expert_mesh_design.md` §2.2 の規定
「**Step 0（オフザシェルフの分野特化モデルをノードごとに割当）** → Step 1（可能なら LoRA で特化，
学習は別環境で実施）→ Step 2（RAG による専門化，任意）」の **Step 0 が未実施**であることを意味する．

**帰結（本調査の推論）**:

1. ノード間に能力差が存在しない．誤ルーティングしても同一モデルがペルソナ文だけ変えて回答する．
2. したがって top1_accuracy は下流に帰結を持たない代理指標である．
3. 設計書が定める評価軸②（回答品質，LLM-as-judge）と③（End-to-End）が未実装のまま残っているのは，
   実装しても現構成では差が出ないためではないか．

「分散環境で自己申告 confidence によるルーティングが機能するか」までは現構成で主張できるが，
「専門家メッシュが有効である」ことは主張できない．

### 1.5 モデルは GPU に完全に載っていた（CPU オフロードではない）

`results/20260721_222225/logs/*/expert-mesh.log` に次の記録がある（4 ノードすべて同値）:

```json
{"level": "INFO", "node_id": "wafl501", "event": "gpu_status",
 "unix_time_s": 1784640117.2276387,
 "model": "isotnek/qwen3.5:9B-Unsloth-UD-Q4_K_XL",
 "size_vram_bytes": 5666399845, "using_gpu": true}
```

**5,666,399,845 バイト = 5.67 GB を VRAM に確保し `using_gpu: true`**．
`dispatch_timeout_s: 400`（実測生成時間 238〜259 秒）という遅さは CPU オフロードではなく，
RTX 3060 で 9B モデルが `DISPATCH_MAX_TOKENS=512` を生成する時間である．

ただし空き VRAM が 6GB 程度の場合，重み 5.67GB では KV cache の余裕がほとんどない．

### 1.6 実験の wall-clock

現行は 46 問で約 46 分．単純比例なら 400 問で約 7 時間になる．
**評価集合の拡張（E1）とモデル小型化は wall-clock を通じて連動する**（本調査の判断）．

### 1.7 未コミットの作業ツリー変更（journal 未記録）

`git status` に，Iter1〜14 のどのイテレーションにも対応しない変更がある．

- `config.yaml`: `confidence_threshold` 0.5→0.3，`dispatch_top_k` 1→2，`confidence_signal_method` stp→self_report
- `router.py`: few-shot 例 5・6・7 の追加（general と medical の切り分け，education と legal の切り分け）

さらに git HEAD（`d56516c`）の `config.yaml` は Iter13 の実験設定（`stp`）のまま未リバートである．
すなわち「リポジトリの現在の設定」と「journal が記録する最良構成（self_report / threshold 0.5 / top_k 1）」が
一致していない．CLAUDE.md の規約に従い変更は加えていない（backlog B27 に登録済み）．

---

## 第 2 部: LLM の不確実性・信頼度推定

### 2.1 【最重要】Iter11 は手法の棄却ではなく実験設計の欠陥である

**Farquhar, Kossen, Kuhn, Gal, *Detecting hallucinations in large language models using semantic entropy*,
Nature 630:625-630, 2024**（https://www.nature.com/articles/s41586-024-07421-0 ）の Methods:

> "In our implementation, we sample at temperature 1 using nucleus sampling (P = 0.9) and top-K
> sampling (K = 50). **We also sample a single generation at low temperature (0.1)** as an estimate
> of the 'best generation' of the model to the context, **which we use to assess the accuracy of the model**."

すなわち temperature 0.1 は，この文献では**不確実性推定のサンプル生成には使わず，
「点推定としての最良回答」を得るためだけに使う設定**である．

- **Wang et al. 2022（self-consistency）**（https://arxiv.org/pdf/2203.11171 ）: 実験は T=0.7 / k=40 を
  中心に，T=0.3〜0.7 の範囲で報告（Figure 6）．
- **Xiong et al., ICLR 2024**（https://arxiv.org/html/2306.13063v2 ）Appendix E.4:
  > "For the use of Self-Random, we set the temperature hyper-parameter as **0.7 to gather a more
  > diverse answer set, as suggested in Wang et al. (2022)**."
- **Cecere et al., *Monte Carlo Temperature*, TrustNLP 2025**（https://arxiv.org/pdf/2502.18389 ）:
  temperature 自体を主題にした実験研究．探索範囲 τ∈[0.1, 1.0]．モデル・データセットごとに最適温度が
  0.3〜1.0 と大きく動き，固定温度の選択が AUROC を数 % 単位で動かすと示す．
  同論文も Farquhar に倣い**正誤判定用の応答だけを temperature 0.1 で生成**している．

**判定**: Iter11（temperature 0.1, N=3）は self-consistency / multi-sample UQ の標準的前提
（T≈0.7〜1.0, N=5〜10）を満たしていない．サンプル間に多様性が無ければ consistency 系の指標は
定数に退化するため，「効果なし」は予測される帰結であり，**手法の棄却根拠にならない**．

**留保**: Farquhar らは "We do not observe the method to be particularly sensitive to details of the
sampling scheme" とも書いている．MCT 論文はこの主張に反証を与えている（温度は AUROC に有意に効く）が，
いずれにせよ T=0.1 は両者の検討範囲の下端かつ「点推定用」の値である．

### 2.2 手法一覧

| 手法 | 出典 | 中核アイデア | 報告された効果 | Ollama API のみで可能か | 実装コスト |
|---|---|---|---|---|---|
| **Semantic Entropy (SE)** | Farquhar et al., Nature 630:625-630, 2024 | T=1 で M サンプル → NLI で意味クラスタ化 → クラスタ上のエントロピー | 幻覚検出 AUROC で naive entropy を一貫して上回る | **可**（生成 M 回 + クラスタ化）．NLI モデルが必要だが同一 LLM への entailment プロンプトで代用可 | 中 |
| **Discrete SE (DSE)** | 同上 / MCT 論文で定式化 | クラスタ確率を**出現頻度**で近似．logprob 不要 | SE とほぼ同等 AUROC（MCT 論文 Table 1） | **可．完全 black-box で最も軽い** | 低 |
| Number of Semantic Sets | Lin et al., TMLR 2024, arXiv:2305.19187 | 意味クラスタ数を数えるだけ | DSE の簡略版 | 可 | 低 |
| **Monte Carlo Temperature** | Cecere et al., TrustNLP 2025, arXiv:2502.18389 | 各サンプルの温度を {0.1, 0.325, 0.55, 0.775, 1.0} から抽出 | oracle 温度との差 3.77%（固定最良 5.34%，ランダム 5.85%）．固定温度に対し**勝率 63〜72%** | 可（`options.temperature` を変えるだけ） | 低 |
| **P(True)** | Kadavath et al., 2022, arXiv:2207.05221 | 回答後に "Is the proposed answer: (A) True (B) False" を別プロンプトで問い，(A) トークンの確率を confidence とする | 52B で正解/不正解サンプルの P(True) 分布が明確に分離．**自分の T=1 サンプルを 5 個文脈に入れると self-eval がさらに改善** | **可**．`logprobs: true, top_logprobs: N`（Ollama v0.12.11 以降，https://docs.ollama.com/api/chat ） | 低〜中 |
| **Verbalized Top-K** | Tian et al., EMNLP 2023, arXiv:2305.14975 | 「候補を K 個挙げ各々に確率を付けよ」．合計を 1 に制約する効果 | gpt-3.5 で ECE **0.131（top-1）→ 0.047（top-2）/ 0.050（top-4）**．相対 50% 超の削減 | **可**（プロンプト変更のみ） | 極小 |
| Linguistic 1S | Tian et al., 2023 | 数値でなく "highly likely" 等の語彙から選ばせ後段で数値へ写像 | 数値 top-1 より良好（ECE 0.062）．ただし top-k には劣る | 可 | 極小 |
| Self-Probing | Xiong et al., ICLR 2024 | 別セッションで「この回答は正しいか」を問う | 過信を部分的に緩和 | 可 | 小 |
| **Top-K prompt + Self-Random(T=0.7, M=5) + Avg-Conf/Pair-Rank** | Xiong et al., ICLR 2024（実務推奨構成） | 言語化 confidence と consistency を統合 | 平均 AUROC **73.0（M=5）** vs Top-K 単発 65.2，CoT 単発 56.4 | 可 | 中 |
| Mahalanobis / kNN OOD | Podolskiy et al., AAAI 2021, arXiv:2101.03778 | 埋め込み空間で ID クラス重心への Mahalanobis 距離．共分散で異方性を補正 | RoBERTa 埋め込みで MSP・cosine を一貫して上回る | 可（`/api/embed` + numpy．ドメイン代表質問集合が必要） | 中 |
| CISC | arXiv:2502.06233 | confidence 重み付き self-consistency | **3 手法中 P(True) が最良の confidence 抽出法**と報告 | 可 | 中 |
| hidden-state probe | Mahaut et al. 2024 | 中間層活性化の線形プローブ | — | **不可**（Ollama が hidden state を出さない．Iter14 で確認済み） | — |

### 2.3 反証・留保

- **P(True) には直接の反証がある**: Tian et al. Table 1 は gpt-3.5-turbo で "'Is True' prob." が
  verbalized より**悪い**較正だったと報告している．Kadavath の良好な結果は 52B の base model かつ
  20-shot（自分の T=1 サンプル 5 個を文脈に入れる）設定であり，RLHF 済み小型モデル + zero-shot で
  そのまま再現する保証はない．
- **専門知識タスクは全手法が苦戦する**: Xiong et al. は「professional law のような専門知識を要する
  タスクでは全手法が苦戦する（ECE 0.16〜0.40）」と明記．expert-mesh はまさにこの領域であり，
  UQ 系手法の伸び代が一般 QA より小さい可能性がある．
- **評価指標のずれ（本調査の推論）**: expert-mesh の top1_accuracy はノード間の**順位付け**問題で，
  較正（ECE）よりも弁別（AUROC）と**ノード間比較可能性**が効く．verbalized が 0/1 に飽和すると
  同点が多発して順位が崩れる．較正指標だけでなく「同点率」「ノード間 confidence の分散」も
  並行して計測すべきである．
- **verbalized confidence の飽和・過信は既知の一般現象**（arXiv:2412.14737，arXiv:2509.25532）．
  後者は「distractor 集合上の総 confidence で正規化する」ことで飽和を緩和できると主張しており，
  expert-mesh にそのまま移植可能な形をしている．
- temperature が真の答え分布そのものを形成するという議論: arXiv:2502.19830．

---

## 第 3 部: LLM ルーティング

### 3.1 手法一覧

| 手法 | 出典 | 中核アイデア | 報告された精度 | 適合性 | コスト |
|---|---|---|---|---|---|
| **MoDEM** | Simonds+ ALTA 2024, arXiv:2410.07490（[PDF](https://aclanthology.org/2024.alta-1.6.pdf) ） | DeBERTa-v3-large (304M) を**ドメイン分類器**として fine-tune し，ドメイン特化 LLM へルーティング | ルーター分類精度 MMLU 全体 **81.00%**（Math 96.63 / Health 81.18 / Science 83.02 / Coding 77.42 / **Other 52.94**）．学習分布内テストは 97% | **最も近い先行研究**．ドメイン＝ノードの構造が同一 | 中 |
| vLLM Semantic Router | Wang+ 2025, arXiv:2510.08731 | ModernBERT 意図分類器で推論モードを選択 | MMLU-Pro で精度 **+10.2pt**，レイテンシ −47.1% | 高 | 中 |
| Hybrid LLM | Ding+ ICLR 2024, arXiv:2404.14618 | DeBERTa エンコーダを BCE で学習し強/弱モデルを選択 | 大モデル呼出 −40%，品質低下なし | 中（2 値選択） | 中 |
| RouteLLM | Ong+ ICLR 2025, arXiv:2406.18665（[code](https://github.com/lm-sys/routellm) ） | 選好データから matrix factorization / BERT / causal-LLM ルーターを学習 | コスト −85% で GPT-4 性能の 95% | 低〜中（選好データが無い） | 大 |
| **RouterDC** | Chen+ NeurIPS 2024, arXiv:2409.19886（[PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/7a641b8ec86162fc875fb9f6456a542f-Paper-Conference.pdf) ） | エンコーダ + LLM 埋め込みを 2 つの contrastive loss で学習（<100M param） | 平均 58.54%．**CosineClassifier と ZOOTER を全タスクで上回る** | 高（「cosine を教師ありで置き換えると勝つ」の直接証拠） | 大 |
| ZOOTER | Lu+ NAACL 2024, arXiv:2311.08692 | 報酬モデルのランキングを蒸留 | 既存 ensemble 同等をより低計算で | 低（報酬モデルが要る） | 大 |
| Srivatsa+ 実証研究 | Insights 2024, arXiv:2405.00467 | multi-label 分類器 / 個別分類器 / クラスタリングを比較 | MLC+pred 63.85 vs Clustering+RoBERTa 61.76．**訓練ルーターでも推論タスクでは最良単体モデルに並ぶ程度** | 高（過度な期待への歯止め） | — |
| UniRoute | Jitkrittum+ 2025, arXiv:2502.08773 | 各 LLM を「代表プロンプト上の正解ベクトル」で表現しクラスタ単位で選択 | 30+ 未知 LLM に汎化 | 中（ノード追加時の再学習回避） | 大 |

### 3.2 教師なし cosine と教師あり分類器の差（Iter2 の再解釈）

- **Varangot-Reille+, JAIR 2025, arXiv:2502.00409** は similarity-based routing を supervised routing と
  明示的に別カテゴリに置き，こう述べる:
  > "**Similarity-based routing frequently fails on complex tasks due to its unsupervised nature**,
  > particularly when discriminating between similar tasks or when noise levels are substantial"
- **RouterDC** は比較対象に **CosineClassifier** を明示的に含め，全タスクで上回っている．
  同じ埋め込みでも教師ありの写像を学習すると弁別力が回復することが実験的に示されている．
- テキスト分類一般でも fine-tune 済みエンコーダは zero-shot LLM を **10〜25pt** 上回るという報告がある
  （AG News / BANKING77．https://asrjetsjournal.org/American_Scientific_Journal/article/view/12048 ，
  Bucher & Martini 2024 https://ipz.uzh.ch/whp/wordpress/wp-content/uploads/2024/08/BucherMartini_2024_LLMs.pdf ）．

**cosine が潰れた原因の既知の説明**: 埋め込みの **anisotropy**（狭い錐への集中）により無関係な文でも
cosine が高止まりする現象．Ethayarajh 2019 以来繰り返し報告されている
（arXiv:2504.16318，https://arxiv.org/html/2606.29571 ）．後者は「密集した encoder」で STS-B の
cosine 相関 0.479，「よく広がった encoder」で 0.857 と，同じ cosine 演算でも幾何が支配的であることを示す．
処方箋は mean-centering / whitening / 上位主成分の除去（Su+ 2021, arXiv:2103.15316）．

**判定: Iter2 の失敗は「埋め込み信号に情報が無い」ことの証明ではない．**
ただし whitening は教師なしのままなので survey の指摘は解消されず，「安価な切り分け」以上の
位置付けをすべきでない．

### 3.3 「中央ルーター無し」の新規性

- A2A の Agent Card による capability 広告，AgentNet（NeurIPS 2025,
  https://neurips.cc/virtual/2025/poster/115584 ）の中央オーケストレータ排除，
  DMoE（Ryabinin & Gusev 2020,
  https://proceedings.neurips.cc/paper_files/paper/2020/file/25ddc0f8c9d3e22e03d3076f98d83cb2-Paper.pdf ）
  など，**分散・中央ルーター無しの構成自体は既に多数存在する**．
- **Internet of Agents サーベイ**（arXiv:2505.07176）は capability evaluation を
  「self-reported declarations」と「system-level verification」に分け，
  > "**self-reporting は登録が速いが，不正確または誇張された主張につながりうる**"
  と明記している．expert-mesh の 14 イテレーションはまさにこの記述の定量的裏付けになっている．

**主張しやすい貢献（本調査の判断）**: 「API-only の commodity GPU 環境において，自己申告 confidence と
トークン確率の双方が専門性信号として機能しないことの定量化，および教師ありルーターとの対比」．
ただし評価規模が 46 問のままではこの主張自体が成立しない．

### 3.4 評価方法とベンチマーク

- **RouterBench**（arXiv:2403.12031）: 405k 推論結果，8 データセット，11 モデル
- **RouterEval**（arXiv:2503.10657, EMNLP 2025 Findings）: 200M+ レコード，8,500 以上の LLM の性能記録
- **LLMRouterBench**（arXiv:2601.07206）: 23,945 prompts → 391,645 instances，21 データセット，33 モデル．
  **「商用ルーターを含む複数の最近手法が単純なベースラインを安定して上回れない」と報告**

**必須ベースライン**: Random / **BestSingle**（常に general ノードへ送る）/ **Oracle**（正解ドメインへ送る上界）．
3 つとも expert-mesh に自明に定義できる．**expert-mesh の 0.87 が BestSingle を上回っているかは
まだ報告されていない．**

**top1_accuracy 以外に報告すべき指標**: Gap@Oracle，PerfGain vs BestSingle，per-domain の precision/recall，
AUC（閾値非依存），ECE / Brier（confidence 較正そのもの）．

---

## 第 4 部: 実験規模の拡大

### 4.1 評価データセット候補

| 名称 | 言語 | 問題数 | ドメイン数 | ライセンス | 適合性 | 出典 |
|---|---|---:|---:|---|---|---|
| **JMMLU** | 日 | **7,536**（1 タスク 86-150） | 56 タスク | CC BY-SA 4.0（3 タスクは CC BY-NC-ND 4.0） | **◎ 最有力**．タスク→ドメイン写像で 4 分野と 10 分野を同一データ上に構成でき，1 分野 300 問以上を確保できる | [HF](https://huggingface.co/datasets/nlp-waseda/JMMLU) / [llm-jp-eval](https://github.com/llm-jp/llm-jp-eval/blob/dev/DATASET_en.md) / https://arxiv.org/html/2402.14531v1 |
| MMLU | 英 | 15,908 | 57 | MIT | △ 英語．日本語動作の現行系に不整合 | https://huggingface.co/datasets/cais/mmlu |
| MMLU-Pro | 英 | 約 12,000 | 14 カテゴリ（選択肢 10 個） | MIT | ○ 14 カテゴリは 10 ノード設計に転用しやすい．ただし英語 | https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro |
| MMLU-ProX | 多言語（日本語含む） | MMLU-Pro 準拠 | 14 | **要確認** | ○ 有望だがライセンス未確認 | https://mmluprox.github.io |
| JMedBench | 日 | 複数データセット統合 | 医療単一（5 タスク種） | 構成データ依存 | ○ 医療ノードの深掘り用 | [COLING 2025](https://aclanthology.org/2025.coling-main.395.pdf) / [HF](https://huggingface.co/datasets/Coldog2333/JMedBench) |
| JMED-LLM | 日 | 各データセット 100 件 | 医療単一 | CC-BY-4.0 中心 | △ 規模が小さい | https://github.com/sociocom/jmed-llm |
| IgakuQA | 日 | 医師国家試験ベース | 医療単一 | 要確認 | ○ 日本の医療 LLM 評価の de facto 標準 | https://arxiv.org/pdf/2509.17444 |
| JBE-QA（司法試験 QA） | 日 | 2015-2024 年の司法試験 | 法律単一 | HF 側は同意ゲート付き | ○ 法律ノードの実データ源 | https://arxiv.org/abs/2511.22869 |
| LegalBench | 英 | 未確認 | 法律 | 未確認 | △ 数値未検証 | — |

**推奨: JMMLU**．56 タスクを 4 分野（現行互換）と 10 分野（拡張）の 2 通りにマッピングすれば，
同一データ上で 4 と 10 を直接比較できる．

### 4.2 ドメイン数 4 → 10 のトレードオフ

**良くなる側**

- **RouterEval**（arXiv:2503.10657）は 8,500 以上の LLM の性能記録から "model-level scaling up" を報告し，
  候補数 m の増加に伴い有能なルータの性能が急速に伸び，**2 ≤ m ≤ 10 の範囲で伸びが最も速い**と明記．
  同論文は m∈{3,5} を easy，m∈{10,100,1000} を hard と定義．MMLU の m=10「all-weak」群では
  個々の LLM が 0.3 未満でも oracle が 0.95 に達する．
- 偶然一致率が 0.25 → 0.10 に下がるため，同じ絶対精度でも Cohen's κ 等の効果量は上がる（自明な数学）．

**悪くなる側**

- **MoDEM**（arXiv:2410.07490）は 5 クラスで OOD の MMLU 総合 **81.00%**，内訳 Math 96.63 /
  Science 83.02 / Health 81.18 / Coding 77.42 / **Other 52.94%**．誤分類の原因を "domain ambiguity" と分析．
  **「Other（＝ 現行の general）が突出して低い」点は，Iter4 で education 追加時に precision が
  0.967 → 0.900 に落ちた現象と構造が同じ**である．
- RouterEval は「既存ルータには大きな改善余地があり，訓練法が不十分だと分類が特定候補に偏る
  （エントロピーで測定）」とも報告．クラス増は偏りリスクを増やす．
- **Label Space Reduction**（https://arxiv.org/html/2502.08436v1 ）は LLM のゼロショット分類で
  ラベル空間を段階的に絞るほど精度が上がることを示し，**ラベル数の増加自体が LLM 分類器の性能を
  落とす**方向の傍証になる．

**帰結（本調査の推論）**: 10 分野化は「検出力は上がるが絶対精度は下がる」で，MoDEM の 81% 前後が
現実的な着地点．現行 0.870（n=46）と単純比較すると「悪化」に見えるが，分野数が違うので同一指標での
比較にならない．**4 分野と 10 分野を同一データで両方測り，κ か chance-corrected 指標で比較する**設計に
しないと解釈不能になる．これを怠ると Iter13 と同種の解釈事故を繰り返す．

### 4.3 モデル候補（1 台あたり空き VRAM 6GB 前提）

| モデル | 量子化 | 重みサイズ | 6GB に収まるか | 日本語 | 出典 |
|---|---|---:|---|---|---|
| **Qwen3.5-9B UD-Q4_K_XL（現行）** | UD-Q4_K_XL | **6.0 GB**（実測ログでは 5.67GB を VRAM 確保） | **△ 余裕なし**（KV cache 込みで逼迫） | 良 | [ollama](https://ollama.com/isotnek/qwen3.5:9B-Unsloth-UD-Q4_K_XL) / §1.5 の実測 |
| **Qwen3.5-4B / Qwen3-4B** | Q4_K_M | 約 2.4-2.5 GB | **◎ 余裕あり** | 良 | https://apxml.com/models/qwen3-4b / https://willitrunai.com/models/qwen-3-4b |
| Llama-3-ELYZA-JP-8B | Q4_K_M | 約 4.9 GB | △ ぎりぎり（context を絞れば可） | 良（日本語特化） | [awesome-japanese-nlp-resources](https://github.com/taishi-i/awesome-japanese-nlp-resources/blob/main/docs/huggingface.md) |
| Llama-3.1-Swallow-8B | Q4_K_M | 約 4.9 GB | △ 同上 | 良 | 同上 |
| Llama3-ELAINE-medLLM-8B | Q4 相当 | 約 5 GB 想定 | △ | 日英中の医療特化 | https://github.com/aistairc/medLLM_QA_benchmark |
| Llama3-Preferred-MedSwallow-70B | — | 数十 GB | × | 医療特化（IgakuQA 0.868） | https://huggingface.co/pfnet/Llama3-Preferred-MedSwallow-70B |
| Preferred-MedLLM-Qwen-72B | — | 数十 GB | × | 医療特化 | https://arxiv.org/html/2504.18080v1 |

**異種モデルメッシュは文献上むしろ標準**である．RouterEval は「この枠組みは異種 LLM と自然に両立する」と
明記し，[vLLM Semantic Router の Mixture-of-Models](https://vllm.ai/blog/2026-07-21-vllm-sr-new-chapter-mom) や
[RouteBalance](https://arxiv.org/html/2606.17949v1) も異種プールを前提にしている．

### 4.4 専門家の実体化（F5 への対処）

**文献の立場は明確にドメイン特化モデル側である．**

- MoDEM の性能向上は router ではなく専門家モデルに由来する．experts は Palmyra-health-70B，
  Qwen2.5-Math，Qwen2.5-Coder といった実際に特化されたモデルで，同社は
  「ほぼすべてのケースで，同サイズの特化モデルが汎用モデルを大きく上回った」と述べる．
- 日本語でも **Llama3-Swallow-70B の IgakuQA 44.6 に対し，医療継続事前学習を施した
  Llama3-Preferred-MedSwallow-70B は 62.6**（https://openreview.net/pdf?id=zQtNbljfK6 ）．

| 案 | 内容 | コスト | 備考 |
|---|---|---|---|
| **S1** | オフザシェルフの分野特化モデルを配置 | 中 | 設計書 §2.2 Step 0 そのもの．**日本語の法律特化オープン生成モデルは今回の調査では発見できなかった**（見つかったのは検索特化の https://arxiv.org/pdf/2412.13205 のみ） |
| **S2（本命）** | 単一ベースモデル + **ドメイン LoRA アダプタ** | 中 | 6GB 制約下で最も現実的．S-LoRA は多数のアダプタを 1 ベースで同時配信する設計（[MLSys 2024](https://proceedings.mlsys.org/paper_files/paper/2024/file/906419cd502575b617cc489a1a696a67-Paper-Conference.pdf) ，[SGLang LoRA](https://lmsysorg.mintlify.app/docs/advanced_features/lora) ）．日本語医療 LoRA の先行例に JMedLoRA（https://openreview.net/pdf?id=BfHX0hKRSe ） |
| S3 | RAG による専門化（設計書 Step 2） | 中〜大 | ドメイン別コーパスの整備が必要 |

**S2 を推す理由（本調査の判断）**: 同じ 10 台の GPU プール上で LoRA ファインチューニングを行う仕組みは
**WAFL-PEFT 側に既にある**．両プロジェクトを接続し，WAFL-PEFT でドメイン特化 LoRA を学習して
expert-mesh のノードへ配布する構成は，研究上も実装上も自然な発展になる．

### 4.5 規模拡大の 3 案

**案 A（推奨・単一レバー）**: データセットのみ差し替え．4 ドメイン維持，モデル維持，JMMLU から
1 ドメイン 100-200 問で n=400-800．SE は ±5.0pt → ±1.5-2.5pt に縮む．過去 14 イテレーションとの
比較不能化を最小に留めつつ，1 問差問題を根絶できる．STP の符号反転バグ検証もこの規模で初めて可能になる．

**案 B**: ドメイン 10 分野 + モデル 4B 化．VRAM 制約上，10 ノード化とモデル変更は分離できない（＝2 レバー同時）．
したがって案 A の後に，まず「4 分野のまま 9B→4B」で 1 イテレーション挟み，モデル変更単独の影響を
測ってから 10 分野へ進むべきである．MoDEM 準拠なら 81% 前後が予想着地点．

**案 C**: 専門家の実体化．4-6 ノードに異なるモデル（general=ELYZA-JP-8B，medical=ELAINE-medLLM-8B，
legal=JBE-QA LoRA 等）を配置し，**評価指標を routing accuracy から end-to-end 回答正解率**へ移す．
研究としての新規性はここが最も高いが，工数と VRAM リスクが最大．

---

## 第 5 部: 反証・懸念・積み残し

### 5.1 反証と懸念

- **比較可能性の喪失は避けられない．** 46 問の consultation 形式（自由文の相談）と JMMLU の四択試験問題は
  タスクの性質が異なる．「頭痛が続いています」に対する分野判定と，解剖学の四択問題に対する分野判定は
  同じ課題ではない．過去 14 イテレーションの数値は新データ上では**再測定しない限り比較できない**．
  最低限，ベストの `self_report` 構成だけは新旧両データで走らせ橋渡し点を残すべきである．
- **JMMLU のタスク→ドメイン写像は主観的である．** MoDEM も MMLU 分野を手動マッピングしており，
  同じ恣意性を抱えている．写像表を成果物として明示し，境界事例を別集計すべきである．
- **教師あり分類器（E6）は Iter10 の label leakage を再演する危険が最も高い．**
  訓練質問と評価質問を**質問単位で完全分離**し，訓練データを probe 実行結果から作らないことが必須．
- **E6 が現状の 0.87 を上回る保証はない．** MoDEM のルーター精度は 5 クラスで 81.00% であり現状より低い．
- **E4 が Iter11 の焼き直しに見えるリスク．** 違いは (a) temperature 0.1→0.7+，(b) N 3→5，
  (c) 集約が平均→意味クラスタのエントロピー，の 3 点．**ユニーク回答数を先に測らずに実装すると
  同じ失敗を繰り返す．**
- **E5 が Iter13 の再演に見えるリスク．** 反論の根拠は「STP は応答全体の流暢さ，P(True) は単一判定
  トークンの自己評価」という測定対象の違いのみ．ここを明文化しないまま実験すると解釈が混同される．
- **Iter5〜9 の 5 連続 no-op と Iter3 の no-op は，レバーが効かないのではなく評価集合が小さすぎて
  差が観測できないことの表れである可能性が高い．** E1 を先に済ませずに他を回すと，
  15 回目・16 回目の no-op を生むだけになる．

### 5.2 積み残し

- MMLU-ProX のライセンス未確認．
- LegalBench の規模・ライセンス未検証．
- Qwen3.5-4B の日本語ドメイン QA での実性能は未確認．
- 日本語の法律特化オープン生成モデルは発見できなかった（LoRA 自作か JBE-QA での特化が必要）．
- Ollama の実機バージョンが v0.12.11 以降か未確認（現在ノード上で Ollama が停止しており，
  `ollama --version` も `/api/version` も応答しない．deploy 後に確認する必要がある）．
  これは E5（P(True)）の実装可否を左右する．
