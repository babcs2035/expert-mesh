# p0001: expert-mesh 今後の研究方針（2026-07-26 策定）

> **【状態: 歴史的記録 — Iter1〜14 時点の方針】**
> 本書が提案したレバー E1〜E7 はいずれも実施済みである（E3 は Iter22 で「`supervised_classifier`
> 下では no-op」と判明した）．研究はその後 Iter54 まで進み `status="converged"` に到達した．
> **収束時点の現況は `docs/d0006_research_summary_iter28-54_2026-08.md` を参照すること．**
> なお本書の F1〜F5・§2b・§3 は `docs/d0001_literature_survey_2026-07.md` と内容が大きく重複する
> （本書が決定，d0001 がその根拠という関係）．

Iter1〜14 の結果と，先行研究の再調査（tavily）に基づく次フェーズの研究方針である．
Iter14 で `status=converged`（実行可能な新レバーを定義できない）としたが，既存の判定を敵対的に
再検討した結果，**棄却判定の多くが統計的に成立しておらず，一部は実験設計の欠陥に起因する**ことが
判明したため，収束判定を撤回して研究を再開する．

- 対象: `config.yaml`，`router.py`，`build_dataset.py`，`metrics.py`
- 前提: 実機 wafl500〜509（RTX 3060 12GB × 10 台）．Ollama で `Qwen3.5-9B-Unsloth-UD-Q4_K_XL` を提供．
  hidden state は取得不可（Iter14 で確認済み）．**logprobs は Ollama v0.12.11 以降で取得可能**．

---

## 1. 決定的な発見（既存判定の再検討）

### F1. 評価集合が 46 問しかなく，棄却判定の多くが「1 問の差」で下されている

`data/dataset.jsonl` の実測: **全 46 問**（単一ドメイン 40 問 = 4 ドメイン × 10 問，複合 6 問）．

主要な判定を問題数に還元すると次のようになる．

| イテレーション | 報告値 | 実際の問題数 | 差 |
|---|---|---|---|
| ベースライン | top1 0.8696 | **40 / 46** | — |
| Iter10（calibrated routing） | 0.870 → 0.848 | 40 → **39** / 46 | **1 問** |
| Iter11（multi_sample） | 0.870 → 0.848 | 40 → **39** / 46 | **1 問** |
| Iter11 misrouting | 0.130 → 0.152 | 6 → **7** / 46 | **1 問** |

p = 0.87, n = 46 における二項標準誤差は **±5.0pt**，Wilson 95% CI は **[74.3%, 93.9%]**（幅約 20pt）である．
2.2pt の変化は 0.44 SE にすぎず，ノイズと区別できない．

さらに深刻なのはドメイン別指標である．各ドメインは **10 問**しかないため，p=0.9 での SE は **±9.5pt** に達する．
Iter7 の「education precision 0.90 → 0.909」や Iter9 の「recall 0.833 → 0.5」は，
**1〜2 問の入れ替わりに相当する量**であり，手法の優劣を論じられる解像度ではない．

**結論: Iter3 および Iter5〜11 の「no-op」「僅差で棄却」という判定群は，レバーが効かなかったことの
証明ではなく，評価集合が小さすぎて差を検出できなかったことの表れである可能性が高い．**

一方，次の 2 件は効果量が十分大きく，実在する効果と判断してよい．
- Iter2（embedding）: 0.971 → 0.529（約 20 SE 相当）
- Iter13（STP）: 0.870 → 0.065（同上）

### F2. Iter11（multi_sample）は「手法の棄却」ではなく「実験設計の欠陥」である

Iter11 は temperature = 0.1，N = 3 で実施し「効果なし」と判定した．しかし文献の前提はこれと異なる．

Farquhar et al., *Detecting hallucinations in large language models using semantic entropy*,
Nature 630:625-630, 2024 の Methods にはこう書かれている．

> "In our implementation, we sample at temperature 1 using nucleus sampling (P = 0.9) and top-K
> sampling (K = 50). **We also sample a single generation at low temperature (0.1)** as an estimate
> of the 'best generation' of the model to the context, **which we use to assess the accuracy of the model**."

すなわち **temperature 0.1 は不確実性推定のサンプル生成には使わず，「点推定としての最良回答」を
得るためだけに使う設定**である．Iter11 は不確実性推定用のサンプルを，まさにその「不確実性を消すための
温度」で取得していたことになる．

- Wang et al. 2022（self-consistency, arXiv:2203.11171）: T = 0.7 / k = 40 が中心．
- Xiong et al., ICLR 2024（arXiv:2306.13063）Appendix E.4: "we set the temperature hyper-parameter as
  **0.7 to gather a more diverse answer set**, as suggested in Wang et al. (2022)."
- Cecere et al., TrustNLP 2025（arXiv:2502.18389）: 温度の探索範囲 τ∈[0.1, 1.0]．正誤判定用の応答のみ 0.1．

**判定: multi_sample 系の手法は，標準的な前提（T ≈ 0.7〜1.0, N = 5〜10）で再試行する価値がある．
Iter11 は棄却根拠にならない．**

### F3. Iter13（STP）の 0.065 は偶然一致を大きく下回っており，符号反転バグの疑いがある

4 ドメインでのランダム選択の期待値は 0.25（46 問中 11.5 問）である．STP の 0.065（**3 / 46**）は
この期待値を約 2.9 SD 下回る．**偶然より systematically に悪い**という結果は，信号が無効であること
よりも，順位付けの符号が反転している（またはスケーリングが逆になっている）ことを強く示唆する．

これは文献ではなく数値からの推論であるが，検証コストは極めて低い（保存済み `results.jsonl` の
confidence 列を符号反転して top1 を再計算するだけ）．**もし反転で 0.87 前後に戻るなら，Iter13 の
「トークン確率は専門性を測らない」という結論そのものが誤りである．**

### F4. Iter2（embedding）の cosine 潰れは「信号が無い」ことではなく既知の幾何的現象である

cosine 値が [0.667, 0.737] に集中した現象は，埋め込み空間の **anisotropy**（狭い錐への集中）として
Ethayarajh 2019 以来繰り返し報告されている既知の現象である．処方箋は mean-centering / whitening /
上位主成分の除去（Su+ 2021, arXiv:2103.15316）．

さらに重要な区別として，ルーティングの調査論文（Varangot-Reille+, JAIR 2025, arXiv:2502.00409）は
similarity-based routing と supervised routing を明示的に別カテゴリとし，
「Similarity-based routing frequently fails on complex tasks **due to its unsupervised nature**」と述べている．
RouterDC（NeurIPS 2024）は比較対象に **CosineClassifier** を含め，全タスクで上回っている．

**判定: Iter2 の失敗は「埋め込みに情報が無い」ことの証明ではなく，「教師なし cosine では取り出せない」
ことの証明である．教師あり分類器は別の帰結になりうる．**

### F5. 【最重要】全ノードが同一モデルであり，「専門家」の実体はプロンプト 1 文しかない

`config.yaml` の 4 ノードは，`light_model` / `expert_model` の**いずれも
`isotnek/qwen3.5:9B-Unsloth-UD-Q4_K_XL` で完全に同一**である．ノード間の差分は次の 2 箇所の
プロンプト文字列だけである．

- `router.py:56`: `あなたは「{domain}」分野の専門家ノードです．`
- `http_server.py:66`: `あなたは「{domain}」分野の専門家です．次の質問に，あなたの専門知識を活かして…`

これは設計書 `docs/encounter_expert_mesh_design.md` §2.2 の規定
「**Step 0（オフザシェルフの分野特化モデルをノードごとに割当）** → Step 1（可能なら LoRA で特化）
→ Step 2（RAG による専門化）」の **Step 0 が未実施**であることを意味する．

帰結は重い．

1. **ノード間に能力差が存在しない．** medical の質問を legal ノードへ「誤ルーティング」しても，
   同一の 9B モデルがペルソナ文だけ変えて回答する．回答品質の差はプロンプト効果に限られる．
2. **したがって top1_accuracy は下流に帰結を持たない代理指標である．** 14 イテレーションが
   最適化してきたのは「ドメインラベルの一致率」であって，システムの有用性ではない．
3. 設計書が定める評価軸②（回答品質，LLM-as-judge）と③（End-to-End）が未実装のまま残っているのは
   偶然ではない．**実装しても現構成では差が出ない**（これは筆者の推論であり文献の主張ではない）．

文献側もドメイン特化モデルの側に立つ．MoDEM（arXiv:2410.07490）の性能向上は router ではなく
専門家モデル（Palmyra-health-70B，Qwen2.5-Math，Qwen2.5-Coder といった実際に特化されたモデル）に
由来し，同論文は「ほぼすべてのケースで同サイズの特化モデルが汎用モデルを大きく上回った」と述べる．
日本語でも Llama3-Swallow-70B の IgakuQA 44.6 に対し，医療継続事前学習を施した
Llama3-Preferred-MedSwallow-70B は 62.6 である．

**「分散環境で自己申告 confidence によるルーティングが機能するか」までは現構成で主張できるが，
「専門家メッシュが有効である」ことは主張できない．** 後者を主張するには Step 0 の実施が前提となる．

なお，過去ログ（`results/20260721_222225`）に
`{"event": "gpu_status", "size_vram_bytes": 5666399845, "using_gpu": true}` が記録されており，
9B モデルは 5.67GB で **GPU に完全に載っていた**（CPU オフロードではない）．
ただし空き VRAM が 6GB 程度しかない場合は余裕がなく，10 ノード構成では検討を要する．

---

## 2. 提案するレバー（優先順）

**F1 が未解決のまま新しいレバーを振れば，15 回目・16 回目の no-op を生むだけである．**
E1 を最優先で実施し，そのうえで E2 以降に進む．

| # | レバー | 変更内容 | 根拠 | Ollama 制約下の可否 | コスト |
|---|---|---|---|---|---|
| **E1** | `eval_set_size` | 46 → **200 問以上**（4 ドメイン層化，複合問題も比例配分）．併せて **Random / BestSingle（常に general へ）/ Oracle（正解ドメインへ）** の 3 ベースラインと Wilson CI・McNemar 検定を導入 | F1 | 可（LLM 非依存） | 小〜中 |
| **E2** | `stp_sign_check` | 保存済み結果の confidence を符号反転して top1 を再計算するだけのオフライン検証 | F3 | 可（実験不要） | 極小 |
| **E3** | `confidence_elicitation` | `numeric_scalar` → **`top_k_with_probs`**（「候補を 2 つ挙げ各々に確率を付けよ」）．合計が 1 に制約されるため 0/1 飽和が機械的に壊れる | Tian et al., EMNLP 2023（arXiv:2305.14975）: ECE 0.131 → **0.047**（top-2） | 可（プロンプトのみ） | 極小 |
| **E4** | `confidence_signal_method` | → **`self_consistency_semantic`**（T = 0.7〜1.0，N = 5，意味クラスタの出現頻度エントロピー = Discrete Semantic Entropy）．**事前にユニーク回答数を計測**し多様性が出ることを確認してから本実験に進む | F2，Farquhar Nature 2024 | 可（生成 5 回 + クラスタ化） | 中 |
| **E5** | `confidence_signal_method` | → **`p_true`**（回答生成後に "Is the proposed answer true? (A) True (B) False" を投げ `top_logprobs` から P("A") を取得） | Kadavath+ 2022（arXiv:2207.05221），CISC（arXiv:2502.06233）で最良の抽出法 | 可（要 Ollama ≥ 0.12.11 の実機確認） | 小 |
| **E6** | `routing_method` | `self_report` → **`supervised_classifier`**（SetFit / multilingual-e5 + logistic head．各ノードが同一分類器を持ち自分のクラス確率のみ返す＝中央ルーター無しを維持） | F4，MoDEM（arXiv:2410.07490）ルーター精度 81.00%，RouterDC が CosineClassifier に勝利 | 可（Ollama 非依存の別プロセス） | 中 |
| **E7** | `embedding_postprocess` | `none` → **`whitening`**（mean-centering + whitening 後に cosine） | F4．Iter2 が「幾何」由来か「信号不在」由来かを切り分ける最小実験 | 可 | 小 |

---

## 2b. 実験規模・設定の拡大（データセット・ドメイン数・モデル）

### 評価データセットを実在ベンチマークへ置き換える（E1 の具体化）

自前生成 46 問を，実在する日本語ドメイン別ベンチマークへ置き換える．

| 名称 | 言語 | 問題数 | ドメイン数 | ライセンス | 適合性 |
|---|---|---:|---:|---|---|
| **JMMLU** | 日 | **7,536** | 56 タスク | CC BY-SA 4.0（3 タスクのみ CC BY-NC-ND） | **◎ 最有力**．タスク→ドメインの写像で 4 分野・10 分野の**両方を同一データ上に構成でき**，1 分野 300 問以上を確保できる |
| MMLU-ProX | 多言語（日本語含む） | MMLU-Pro 準拠 | 14 | 要確認 | ○ 14 カテゴリは 10 ノード設計に転用しやすい |
| JMedBench / IgakuQA | 日 | 複数 | 医療単一 | 構成データ依存 | ○ 医療ノードの深掘り用 |
| JBE-QA（司法試験） | 日 | 司法試験ベース | 法律単一 | HF 側は同意ゲート | ○ 法律ノードの実データ源 |

**JMMLU を主データセットに推奨する．** 4 分野マッピングなら過去 14 イテレーションとの橋渡しができ，
同じデータで 10 分野マッピングも作れるため，ドメイン数の効果を交絡なく測れる．

**規模と wall-clock のトレードオフ（実務上の制約）**: 現行は 46 問で約 46 分かかっている
（`dispatch_timeout_s: 400`，実測生成時間 238〜259 秒）．単純比例なら 400 問で約 7 時間になり，
イテレーションを回せなくなる．**E1（評価拡張）とモデル小型化は wall-clock を通じて連動する**．

### ドメイン数 4 → 10

- **良くなる側**: RouterEval（arXiv:2503.10657）は候補数 m の増加に伴いルータ性能の伸びが速く，
  **2 ≤ m ≤ 10 の範囲が最も急**と報告する．偶然一致率も 0.25 → 0.10 に下がるため，同じ絶対精度でも
  chance-corrected な効果量は上がり，統計的に検出しやすくなる．
- **悪くなる側**: MoDEM は 5 クラスで総合 81.00%，うち **Other（＝ general 相当）が 52.94%** と突出して
  低く，原因を domain ambiguity と分析している．**これは Iter4 で education を追加した際に
  precision が 0.967 → 0.900 に落ちた現象と構造が同じ**であり，general ノードが共通のボトルネックである．
  Label Space Reduction（arXiv:2502.08436）も，ラベル数の増加自体が LLM 分類の精度を落とすと示す．
- **必須の設計上の注意**: 分野数が変われば偶然一致率が変わるため，**生の accuracy を 4 分野と 10 分野で
  直接比較してはならない**．κ 等の chance-corrected 指標を主指標に据えること．これを怠ると
  Iter13 と同種の解釈事故（偶然一致を下回る値を「手法の失敗」と誤読する）を繰り返す．

### モデルの変更とノード数

- 現行 `Qwen3.5-9B-Unsloth-UD-Q4_K_XL` は実測 **5.67GB を GPU に確保**して動作していた．
  空き VRAM が 6GB 程度なら余裕がなく，KV cache を含めると厳しい．
- 代替: **Qwen3.5-4B / Qwen3-4B（Q4_K_M で約 2.4〜2.5GB）** なら余裕があり，生成も速くなるため
  **E1 の評価拡張を wall-clock 的に可能にする**．10 ノード構成では実質必須となる．
- 日本語特化: Llama-3-ELYZA-JP-8B / Llama-3.1-Swallow-8B（Q4_K_M で約 4.9GB）は 6GB にぎりぎり．
- **異種モデルメッシュは文献上むしろ標準**である（RouterEval は「異種 LLM と自然に両立する」と明記．
  vLLM Semantic Router の Mixture-of-Models，RouteBalance も異種プール前提）．

### F5 への対処案（専門家の実体化）

| 案 | 内容 | コスト | 備考 |
|---|---|---|---|
| **S1** | オフザシェルフの分野特化モデルを配置（general=ELYZA-JP-8B，medical=Llama3-ELAINE-medLLM-8B 等） | 中 | 設計書 §2.2 Step 0 そのもの．**日本語の法律特化オープン生成モデルは今回の調査では見つからなかった** |
| **S2** | 単一ベースモデル + **ドメイン LoRA アダプタ**（S-LoRA / SGLang の複数アダプタ配信） | 中 | 6GB 制約下で最も現実的．日本語医療 LoRA の先行例に JMedLoRA がある |
| **S3** | RAG による専門化（設計書 Step 2） | 中〜大 | ドメイン別コーパスの整備が必要 |

**S2 が本命**である．加えて，同じ 10 台の GPU プール上で LoRA ファインチューニングを行う仕組みは
**WAFL-PEFT 側に既にある**．両プロジェクトを接続し，WAFL-PEFT でドメイン特化 LoRA を学習して
expert-mesh のノードへ配布する構成は，研究上も実装上も自然な発展になる．

### 推奨する順序

1. **E1（JMMLU で 4 分野・200 問以上）+ E2（STP 符号検証）** — 測定系の修復
2. モデル 9B → 4B（4 分野のまま．wall-clock 短縮とモデル変更単独の影響測定）
3. **S2（ドメイン LoRA で専門家を実体化）+ 評価軸②③の実装** — ここで初めてルーティングの価値を測れる
4. ドメイン 4 → 10（κ を主指標に）

## 3. 敵対的検討（各案への反証）

- **E6 は Iter10 の label leakage を再演する危険が最も高い**．Iter10 は offline AUC = 1.000 →
  online 退行だった．分類器の訓練質問と評価質問を**質問単位で完全分離**し，訓練データを probe 実行結果
  からではなく質問文とドメインラベルのみから作ることが必須条件である．
- **E6 が現状の 0.87 を上回る保証はない**．MoDEM のルーター精度は 5 クラスで 81.00% であり，
  現状の self_report より低い．特に MoDEM は「Other」クラスで 52.94% と大きく崩れており，これは
  Iter4 で education 追加時に precision が 0.967 → 0.900 に落ちた現象と構造が同じである．
  **general ノードの扱いが両者に共通するボトルネック**と見るべきである．
- **E5 には直接の反証がある**．Tian et al. Table 1 は gpt-3.5-turbo で "Is True" 確率が verbalized より
  **較正が悪い**と報告している．Kadavath の良好な結果は 52B の base model かつ 20-shot（自分の T=1
  サンプル 5 個を文脈に入れる）設定であり，RLHF 済み小型モデルの zero-shot で再現する保証はない．
  E5 は E4 と組み合わせる（サンプルを文脈に入れる）方が文献に忠実である．
- **E7 は Iter2 の再演になりうる**．whitening は教師なしのままなので，「unsupervised だから失敗する」
  という survey の指摘は解消されない．E7 は「安価な切り分け」以上の位置付けをすべきでない．
- **E4 が Iter11 の焼き直しではないか**という点は正直に検討すべきである．違いは (a) temperature
  0.1 → 0.7+，(b) N 3 → 5，(c) 集約が平均 → 意味クラスタのエントロピー，の 3 点である．
  Iter11 の平均化は「同一回答が 3 つ」と「意味の異なる 3 回答」を区別できなかった．
  **ユニーク回答数を先に測らずに実装すると同じ失敗を繰り返す**ため，事前計測を必須とする．
- **課題領域そのものの難しさ**: Xiong et al. は「professional law のような専門知識を要するタスクでは
  全手法が苦戦する（ECE 0.16〜0.40）」と明記している．expert-mesh はまさにこの領域であり，
  不確実性推定系手法の伸び代が一般 QA より小さい可能性がある．
- **評価指標のずれ（推論であり文献の主張ではない）**: expert-mesh の top1_accuracy はノード間の
  **順位付け**問題であり，較正（ECE）よりも弁別（AUROC）と**ノード間比較可能性**が効く．
  verbalized が 0/1 に飽和すると同点が多発して順位が崩れる．E3 が有効ならこの機序による．
  較正指標だけでなく「同点率」「ノード間 confidence の分散」も併せて計測すべきである．

## 4. 新規性の所在

分散・中央ルーター無しの構成自体は既に多数存在する（A2A の Agent Card，AgentNet NeurIPS 2025，
DMoE 2020 等）ため，アーキテクチャそのもので新規性を主張するのは難しい．

一方，Internet of Agents サーベイ（arXiv:2505.07176）は capability evaluation を「self-reported
declarations」と「system-level verification」に分け，**「self-reporting は登録が速いが，不正確または
誇張された主張につながりうる」**と明記している．expert-mesh の 14 イテレーションは，まさにこの記述の
定量的な裏付けになっている．

したがって主張しやすい貢献は「**API-only の commodity GPU 環境において，自己申告 confidence と
トークン確率の双方が専門性信号として機能しないことの定量化，および教師ありルーターとの対比**」である．
ただし**評価規模が 46 問のままではこの主張自体が成立しない**（F1）．E1 は新規性の前提条件でもある．

## 5. 影響範囲

- `data/dataset.jsonl` / `build_dataset.py`: 評価集合の拡張（E1）
- `metrics.py`: Wilson CI，McNemar 検定，3 ベースライン，同点率の追加（E1）
- `router.py`: elicitation 形式（E3），信号抽出方式（E4/E5），whitening（E7）
- `config.yaml`: 各レバーのスイッチ
- 新規: 教師あり分類器の学習・配布経路（E6．訓練/評価の質問単位分離を必須とする）
- Iter3・Iter5〜11 の判定は E1 完了後に**再評価が必要**．「収束済み」判定は撤回する．
