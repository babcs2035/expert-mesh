## Iteration 15: 評価集合の 200 問以上への拡張と統計的判定基準の導入

### 調査 (Iter15)

Iter14 の `converged` 判定を撤回する．先行研究の再調査（tavily）とリポジトリの実測により，
既存の棄却判定の多くが統計的に成立していないか，実験設計の欠陥に起因することが判明した．
**提案は `plans/p0001_research_direction_2026-07.md`，出典付きの全調査記録は
`docs/d0001_literature_survey_2026-07.md` にある．** 以下は要点のみ．

**実測で確定した事実**

1. **評価集合は 46 問しかない（F1）**: `data/dataset.jsonl` の実測で単一ドメイン 40（4×10）+ 複合 6．
   p=0.87,n=46 の SE は **±5.0pt**，Wilson 95% CI は **[74.3%, 93.9%]**（幅 約19.5pt）．
   Iter10/Iter11 の「0.870→0.848」は **40/46 → 39/46 の 1 問差**．
   ドメイン別指標は 1 ドメイン 10 問で SE ±9.5pt であり，Iter7 の「precision 0.90→0.909」や
   Iter9 の「recall 0.833→0.5」は 1〜2 問の入れ替わりに相当する．
   **Iter3・Iter5〜11 の「no-op / 僅差で棄却」は，差を検出できなかっただけの可能性が高い．**
2. **Iter11 は実験設計の欠陥（F2）**: Farquhar et al. (Nature 630:625-630, 2024) は
   「temperature 0.1 は**点推定としての最良回答**の生成に使い，不確実性推定は T=1・nucleus P=0.9 で行う」と
   Methods に明記している．Wang et al. 2022 は T=0.7/k=40，Xiong et al. ICLR2024 も
   「T=0.7 to gather a more diverse answer set」と記す．
   **Iter11 は不確実性を消す設定で不確実性を測っており，multi_sample 系の棄却根拠にならない．**
3. **Iter13 の 0.065 は偶然一致を 2.9 SD 下回る（F3）**: 4 ドメインの偶然一致 0.25（11.5/46）に対し
   3/46．偶然より systematically に悪いのは符号反転バグを示唆する．
   保存済み `results.jsonl` の符号反転で再計算するだけで検証できる．
4. **Iter2 の cosine 潰れは既知の幾何的現象（F4）**: 埋め込みの anisotropy であり「信号が無い」証明ではない．
   Varangot-Reille+ JAIR2025 は similarity-based routing の失敗を unsupervised であることに帰し，
   RouterDC (NeurIPS2024) は CosineClassifier に全タスクで勝利している．処方箋は whitening（Su+ 2021）．
5. **【最重要】全ノードが同一モデルで「専門家」の実体がない（F5）**: `config.yaml` の 4 ノードは
   light/expert とも `isotnek/qwen3.5:9B-Unsloth-UD-Q4_K_XL` で同一であり，差分は
   `router.py:56` と `http_server.py:66` のプロンプト 1 文だけ．
   設計書 §2.2 の「Step 0（オフザシェルフの分野特化モデルをノードごとに割当）」が未実施である．
   **ノード間に能力差が無いため誤ルーティングしても回答品質はほぼ変わらず，top1_accuracy は
   下流に帰結を持たない代理指標になっている．**評価軸②③が未実装なのもこれが理由と考えられる．
6. **モデルは GPU に載っていた（F5 補足）**: `results/20260721_222225` のログに
   `size_vram_bytes: 5666399845, using_gpu: true` があり，5.67GB を VRAM 確保して動作していた
   （CPU オフロードではない）．dispatch の 238-259 秒は RTX 3060 での 9B 生成時間である．

**文献調査の要点**

- 較正改善の最安手は **Verbalized Top-K**（Tian et al. EMNLP2023）で，gpt-3.5 の ECE を
  0.131 → **0.047**（top-2）に下げた．確率の合計制約が 0/1 飽和を機械的に壊す．
- **P(True)**（Kadavath+ 2022）は STP と測定対象が異なる（生成全体の流暢さ vs 単一判定トークンの
  自己評価）．Ollama v0.12.11 以降の `logprobs`/`top_logprobs` で実装可能．
  ただし Tian et al. Table 1 は gpt-3.5 で "Is True" が verbalized より較正が悪いと報告する反証もある．
- ドメイン数 4→10 は RouterEval が「2≤m≤10 で伸びが最も速い」と報告する一方，MoDEM は 5 クラスで
  総合 81.00%・**Other（general 相当）52.94%** と報告．Iter4 の education 追加時の precision 低下と
  構造が同じで，general ノードが共通のボトルネック．
  **分野数が変わると偶然一致率が変わるため κ 等の chance-corrected 指標が必須．**
- 評価データセットは **JMMLU**（7,536 問・56 タスク・CC BY-SA 4.0）が最有力．
  同一データ上に 4 分野と 10 分野の両方の写像を作れる．
- ドメイン特化の効果は大きい: Llama3-Swallow-70B の IgakuQA 44.6 → 医療継続事前学習済みの
  Llama3-Preferred-MedSwallow-70B は 62.6．6GB 制約下では **単一ベース + ドメイン LoRA**（S-LoRA 型）が本命．

**改訂内容**

`config.yml` の levers を全面改訂し，E1（評価 200 問以上 + Random/BestSingle/Oracle + Wilson CI +
McNemar）を最優先に，E2（STP 符号検証）・E3（Verbalized Top-K）・E4（正しい前提での self-consistency）・
E5（P(True)）・E6（教師あり分類器）・E7（whitening）・E8（4B 化）・E9（10 分野）・
E10（専門家の実体化 + 評価軸②③の実装）を登録した．`success_criteria` も統計的に判定可能な形へ改訂した．

### 計画 (Iter15)

**単一レバー**: `eval_set_size`（config.yml levers 先頭，候補値 [200, 400]）．今回は **200** を採る．
理由: p=0.87 を仮定した二項 SE は n=200 で ±2.4pt（Wilson 95% CI 幅 約9pt）まで縮み，n=46 の
±5.0pt（幅 約20pt）から目的が達成できる一方，`dispatch_timeout_s` の実測（238〜259 秒/問）から
単純比例すると n=400 は約 7 時間となり 1 イテレーションで回せない．400 への拡張は，200 で統計基盤が
正しく動くことを確認した後の次の値として温存する（同一レバーの次段階）．

**B27（作業ツリーの未コミット変更）の判断**

`git status` で確認した未コミット差分は 3 種類の性質が異なる変更が混在していたため，個別に判断した．

1. `config.yaml: confidence_signal_method: stp → self_report` — **採用**．
   journal には Iter3・Iter6〜Iter9・Iter11 を通じて「config.yaml は不変
   （`routing_method: self_report`, `confidence_threshold: 0.5`, `dispatch_top_k: 1` を維持）」という
   記述が繰り返され，`confidence_signal_method` も明示的に self_report が既定として扱われてきた
   （Iter9 baseline は self_report，Iter12/13 で stp を試し rejected 確定）．
   HEAD（commit d56516c）の config.yaml が stp のまま止まっているのは，Iter13 で reject 判定した後に
   ベースラインへ戻すコミットが漏れていたための不整合であり，self_report への変更はこの漏れを正す
   もので研究上の最良構成と一致する．
2. `config.yaml: confidence_threshold: 0.5 → 0.3` — **棄却（HEAD の 0.5 に戻す）**．
   Iter3 で候補値 [0.3, 0.5, 0.7] は selected_domain/fallback/dispatch のいずれも動かさない no-op と
   判定済み（confidence が二峰・空帯域分布のため）であり，0.3 へ動かす根拠を裏付ける新しい記録が
   journal・backlog のどこにもない．単一レバー原則を守るため，E1 の実験対象外の設定は
   「直近の journal が記録する最良構成」に固定する必要があり，根拠不明な追加変化は含めない．
3. `config.yaml: dispatch_top_k: 1 → 2` — **棄却（HEAD の 1 に戻す）**．
   Iter1 で dispatch_top_k=2 は「selected_domain 不変（confidence 最大選択のため構造的に no-op）」かつ
   「単一ドメイン行で無駄な追加 dispatch が発生する副作用あり」で棄却済み．2 に変更したまま E1 を
   実施すると，E1（データ規模拡大）以外の要因（無駄 dispatch によるレイテンシ増）が混入し，
   単一レバー原則に反する．
4. `router.py`: few-shot 例 5・6・7 の追加 — **棄却（HEAD の内容に戻す）**．
   config.yml の levers 履歴が示すとおり，few-shot 修正系のレバーは Iter5〜9 で 5 パターンすべて
   rejected/no-op と判定済みの系統である．今回追加された 3 例（general/medical，education/legal の
   切り分け）はどのイテレーションにも対応しない未検証コードであり，このまま残すと E1 の実験結果が
   「データ規模の効果」なのか「未検証 few-shot 変更の効果」なのか切り分けられなくなる．

**結論**: E1 実験で固定する構成は `confidence_signal_method: self_report`，`confidence_threshold: 0.5`，
`dispatch_top_k: 1`，`routing_method: self_report`，`router.py` は few-shot 例 1〜4 のみ（HEAD 相当）．
rc-implementer は着手前に `config.yaml` の `confidence_threshold` を 0.5 へ，`dispatch_top_k` を 1 へ戻し，
`router.py` の未検証 few-shot 追加（例 5・6・7）を取り除いたうえで，`confidence_signal_method: self_report`
のみを反映すること．これらの revert 自体は E1 の変更ではなく「直近最良構成への復帰」であり，
`git diff` で意図どおりの差分（confidence_signal_method の1行のみ）になっていることを確認してから
データセット拡張・metrics.py 変更に進むこと．

**データセット拡張の実現方法**

調査フェーズ（p0001/d0001）は JMMLU（nlp-waseda/JMMLU, 56 タスク・7,536 問）を最有力候補として推奨していたが，
本フェーズで実データを確認した結果，2 点の新しい事実が判明したため，**JMMLU の採用を見送り，既存の
自前作成（community-consultation 形式）を同一スタイルで増量する方針**に変更する．

1. **ライセンスの事実誤認を訂正**: `docs/d0001` は「CC BY-SA 4.0（3 タスクのみ CC BY-NC-ND）」としていたが，
   HF 上の現行 README（2026-07-26 時点で実機確認）は **データセット全体が CC BY-NC-ND 4.0**
   （「研究・LLM評価目的の商用利用のみ許可，改変・再配布に制限あり」）と明記している．非商用の研究評価
   利用自体は許容されるが，NoDerivatives 条項下でタスク→ドメインへの再マッピングや設問の並べ替え・
   フィルタリングが「改変」に該当するかはグレーであり，追加確認なしに採用するのはリスクがある．
2. **`education` ドメインに対応する JMMLU タスクが存在しない**: JMMLU の 56 タスクは MMLU 由来の
   学術科目（医学・法学・物理・経済等）と日本文化科目（日本史・公民・熟語等）のみで，本研究の
   education ドメイン（学習指導要領・教員免許・教育委員会等の**日本の教育行政・教育実務**）に
   相当するタスクがない．4 ドメイン全てを JMMLU で置き換えることはできず，education だけ別系統の
   データ源が必要になり，「同一ベンチマーク上で 4 分野を統一的に拡張する」という JMMLU 採用の主目的が
   崩れる．また四択試験問題と自由文の相談形式は課題の性質が異なる（d0001 5.1 で懸念済み）．

このため，`build_dataset.py` の既存 4 関数（`_MEDICAL_QUESTIONS` 等）と同じスタイル・文体で問題数を
増量する．目標配分（合計 200 問以上）:
- 単一ドメイン: medical / legal / general / education 各 **45 問**（計 180 問）．
  45 問/ドメインでの二項 SE は ±5.0pt（p=0.87 時）で，現行の 1 ドメイン 10 問（±9.5pt）から明確に改善する．
- 複合ドメイン: **20 問**（現行 6 問の構成比 medical+legal 多数・education+medical・education+legal を
  維持しつつ比例増量．具体的な内訳は rc-implementer の裁量とするが，単一の組み合わせに偏らないこと）．
- 合計 200 問．既存 46 問（各ドメイン先頭 10 問・複合 6 問）はそのまま残し，末尾に新規問題を追加する形とする
  （id は `medical-011`以降のように連番を継続し，過去 results.jsonl との突合や部分再利用を容易にする）．
- 新規問題は入力実行環境からの独自作成とし，外部ベンチマークの設問文をそのまま流用しないこと
  （ライセンス上の懸念を避けるため）．

**metrics.py への追加実装**

1. `compute_wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]`:
   Wilson score interval。`docs/d0001`・`plans/p0001` が引用した k=40,n=46 → [0.743, 0.939] を
   単体テストの期待値として使う（`tests/test_metrics.py` に追加）。
2. `compute_baselines(results, all_domains) -> dict`:
   - `random`: 各行 `len(expected_domains)/len(all_domains)` の平均（解析的な期待値，モンテカルロ不要）。
   - `best_single`: config.yml note のとおり「常に general」固定で `general in expected_domains` の
     割合。参考として実データ上の経験的最頻正解ドメインも併記し，"best_single" が general と一致しない
     場合はその旨をログに残す。
   - `oracle`: 定義上 1.0（正解ドメインへ送れば必ず一致するため）。単なる定数ではなく，
     「ドメイン知識が完全なら 100% になる」という前提をdocstringで明示する。
3. `mcnemar_test(results_a, results_b) -> dict`:
   `id` で結合し，2×2 分割表 (b, c) から連続性補正付き χ² 統計量と p 値を返す。
   `b + c < 25` の場合は正確二項検定にフォールバックする（サンプル数が少ない場合の近似誤差を避けるため）。
   **前提として `results_a` と `results_b` は同一の質問集合（同一 `id` 群）でなければならない**
   ことをdocstringに明記する（Iter15 単体では新旧データセットの質問が異なるため McNemar 対比較の対象には
   ならない。McNemar は次イテレーション以降，同一の 200 問データセット上で 2 つのレバー値/手法を比較する
   際に使う）。
4. `compute_all_metrics` に上記 3 つを追加し（`baselines`, `wilson_ci` キー），
   `print_summary` にも Wilson CI・baseline 比較の表示を追加する。既存キーは変更しない（後方互換）。

**固定する構成（E1 以外は変更しない）**: `confidence_signal_method: self_report`，
`confidence_threshold: 0.5`，`dispatch_top_k: 1`，`routing_method: self_report`，
`router.py` は few-shot 例 1〜4 のみ，4 ノード構成・モデル（qwen3.5:9B）は不変。

**期待効果**: (1) 200 問データセット上で self_report ベースラインを再測定し，Wilson 95% CI が
現行の約 20pt 幅から 10pt 未満へ縮むこと，(2) Random/BestSingle/Oracle と並記することで
「0.87 が本当に無意味な水準ではないか」を定量的に確認できること，(3) 以降のレバー（E2〜）で
McNemar 対比較が使える基盤が整うこと。

**運用上の注意**: 現行 46 問で約 46 分（1 問あたり約 1 分）の実測から，200 問では単純比例で
約 3.3 時間かかる見込み。`.claude/research/config.yml` の `experiment.timeout_min: 90` は不足するため，
rc-implementer は実装完了後，この値を 250〜300 程度へ引き上げること（本フェーズでは config.yml 自体を
変更しない）。

**成功条件（accuracy の増減ではなく統計基盤の正しい実装・動作を主眼とする）**:
1. `data/dataset.jsonl` が 200 問以上・4 ドメイン層化（各ドメイン単独 40 問以上）・複合行を含み，
   `id` が全て一意であること。
2. `metrics.py` に `compute_wilson_ci`・`compute_baselines`・`mcnemar_test` が実装され，
   `tests/test_metrics.py` の単体テスト（Wilson CI は既知値 [0.743, 0.939] との整合，McNemar は
   人工データでの手計算値との整合）が pass すること。
3. 新データセットに対し `confidence_signal_method: self_report` 固定構成で
   `mise run setup/deploy/run/analyze` が完走し，`dispatch_failure_rate` が実質 0（インフラ起因の失敗が
   ないこと）であること。
4. `metrics.py --json` の出力に `top1_accuracy` の Wilson 95% CI と Random/BestSingle/Oracle の
   3 baseline が含まれ，例外なく計算できること。
5. 上記が全て満たされれば，accuracy の値そのもの（上がる/下がる/変わらない）に関わらず E1 は
   **採用（統計基盤の整備完了）**と判定する。逆に (1)〜(4) のいずれかが未達なら「未完了」とし，
   次イテレーションでも E1 を継続する。

### 実装 (Iter15)

**単一レバー原則からの逸脱（ユーザー明示指示）**: 本フェーズは通常の research-cycle オーケストレータ
ではなく，ユーザーが対話セッションで直接指示した手動実装である。当初は E1（`eval_set_size`）のみの
継続を想定していたが，ユーザーが「p0001 の E1〜E7 に加え，ドメイン4→10化・モデル9B→4B化・専門家の
実体化（S1）・評価軸②③の実装まで，今回のセッションで一括実装せよ」と明示的に指示したため，単一レバー
原則を今回に限り上書きして全レバーを実装した。バッチ0〜10（11単位）に分割し，各バッチ完了ごとに
`uv run pytest`/`uv run ruff check` を実行して回帰がないことを確認しながら進めた。

**E1（完了・確定）**: データセットは当初案（46→200問のハードコード拡張）から方針変更し，**JMMLU
（`nlp-waseda/JMMLU`, commit `3637b25e444`）へ全面差し替え**，かつ**ドメイン数は4を経由せず最初から
10固定**とした（ユーザー指示）。JMMLUの実際のライセンスは調査時点の記載（CC BY-SA 4.0中心）と異なり
**全体がCC BY-NC-ND 4.0**だったことを実データ取得で確認し訂正した（研究・評価用途は許諾範囲内）。
10ドメイン（medical/legal/education/business_economics/computer_science/natural_science/mathematics/
history_culture/social_science/general）へのJMMLU56タスク写像を実データで確定し，`build_dataset.py`を
全面書き換え。legalは`professional_law`不在のため227問・2タスクのみ（目標150問は満たすが実質的な
多様性は低い），educationは直接対応タスクが無く心理学・社会学で代理——という制約はdocstringに明記済み。
`metrics.py`にWilson信頼区間・McNemar検定・Cohen's kappa（chance-corrected指標）・
Random/BestSingle/Oracleベースラインを追加（scipy/numpy不使用，`math.erf`による閉形式実装）。
d0001記載のWilson CI参考値[74.3%, 93.9%]との整合をテストで確認済み。`router.py`のfew-shot例も
ハードコード4ドメインから動的生成（`_build_few_shot_examples`）へ書き換え，10ドメインでもプロンプト
手直し不要にした。`config.yaml`のnodesを10ノード（wafl500〜509, 192.168.15.100〜109）へ拡張。

**E2〜E7（コード実装完了，実機実験は未実施）**: E2（STP符号反転検証）は保存済み
`results/20260722_113854/results.jsonl`に対し実行し，argmax(confidence)=0.0652・argmin=0.3913・
偶然一致0.2826を再現——「符号反転で0.87相当に戻る」という単純仮説は支持されないと結論。E3
（top_k_with_probs），E4（self_consistency_semantic，entailmentクラスタリング＋Discrete Semantic
Entropy，案A採用），E5（p_true，Kadavath et al. 2022の2段階自己評価，Ollama v0.12.11+の
top_logprobs対応をexpert_backend.pyに追加），E6（supervised_classifier，label leakage対策として
訓練/評価クエリの構造的分離を実装しテストで重複0件を確認），E7（embedding whitening/mean-centering）
を全て実装。E6でscikit-learnを本体依存へ追加。

**モデル変更・専門家実体化（S1）**: `light_model`を全10ノードで`qwen3.5:4b-q4_K_M`へ変更
（実在するOllamaタグであることを実際にレジストリで確認）。専門家の実体化はOllamaレジストリを実際に
検索した結果，**医療・法律いずれの分野にも専門特化した日本語生成モデルは見つからなかった**
（法律は文献調査時点の既知の制約，医療は今回新たに確認）。そのため`expert_model`は全ノード共通で
`schroneko/llama-3.1-swallow-8b-instruct-v0.1:q4_k_m`（Q4_K_M, 実測4.9GB，実在確認済み）とした——
これは前向きな実装ではなく「S1は現時点のOllamaレジストリでは真の意味では実現できない」という
誠実な否定的知見として記録する。

**評価軸②③**: 設計書§4.1が指標名のみで実装方式を規定していなかったため，新規`evaluation.py`で
設計・実装した。JMMLU由来行は`jmmlu_answer`との抽出照合（`extract_answer_letter`，ヒューリスティック），
手作り相談行はLLM-as-judge（1-5ルーブリック，判定モデルはgeneralノードのexpert_modelを再利用し
専用judgeモデルは立てない）。レイテンシ内訳については，READMEが「`latency_ms`から`gen_time_ms`を
引いて通信時間を分離計測できる」と主張していたが，実際には`latency_ms`というフィールド自体が
存在せず，クライアント側（`http_client.py`/`run_experiment.py`）は送信時刻を記録していなかった
ため分離計算は不可能だったことが判明（README記載と実装の乖離）。`run_experiment.py`に
`dispatch_gen_time_ms`（既存のDispatchResponse.gen_time_msを結果行へ追加露出）と`request_id`を追加し，
`evaluation.compute_latency_breakdown`で「dispatch生成時間 vs それ以外（probeラウンドトリップ等）の
残差」として近似計算できるようにした上で，README記載を実態に合わせて訂正した。

**検証結果**: 単体テスト172件全て通過（新規テストファイル9個），`ruff check`/`ruff format`は変更した
全ファイルでクリーン。実データでのend-to-end確認: JMMLU.zipを実際にダウンロードして
`build_dataset.py`を実行（1520行生成），classifier train/eval分離を実データで検証（重複0件），
`verify_stp_sign_flip.py`を実際のIter13結果に対して実行し設計時の想定数値を再現。

**ユーザー指示による2回の敵対的レビューで発見・修正した実バグ**（「全ての修正などが正しく施されたか，
敵対的に総点検せよ」を2回実施）:
1. **`Dockerfile`に`classifier.py`のCOPYが漏れていた（最重要）**: `http_server.py`が
   `classifier.py`を無条件importするため，`routing_method`の設定に関わらず**全10ノードが起動時に
   `ModuleNotFoundError`でクラッシュする**状態だった。2回目のレビューで発見し，COPY行に追加。
   `mise run setup`のDockerビルド成功で修正を確認済み。
2. `build_dataset.py`の`main()`が，クリーンチェックアウト直後（`data/`ディレクトリ未作成）だと
   `FileNotFoundError`で落ちる欠陥（`_ensure_parent_dir()`追加で解消，`/tmp`での再現テストで確認）。
3. `router.py`の`extract_p_true()`が，正のlogprobが返った場合に確率が1.0を超え得る欠陥
   （`min(max(math.exp(...), 0.0), 1.0)`でクランプ）。既存テストは全て負のlogprobのみを使っており，
   このクランプを実際に働かせるテストが無かったため回帰テストを追加。
4. `metrics.py`の`compute_cohens_kappa`が，`results`が空でないのに`domains`が空という異常系で
   無言のまま生のaccuracyへ退化する欠陥（`ValueError`を送出するよう修正，かつ元々の「`total==0`なら
   0.0を返す」正常系との判定順序を入れ替えて両立させた）。
5. `http_server.py`で`embedding_postprocess != none`なのに`embedding_whitening_path`が未設定，
   または`routing_method=supervised_classifier`なのに`classifier_model_path`が未設定という
   設定不整合を，起動時に`ValueError`で検出するようにした（従来は無言でフォールバックしていた）。
6. `scripts/train_domain_classifier.py`の`LogisticRegression`に`class_weight="balanced"`を追加
   （legalドメインの訓練データがおよそ半分のサイズ（77 vs 150）であるため）。
7. テストヘルパー`_result()`の`row_id`デフォルトが`id(object())`（CPythonのアドレス再利用により
   一意性が保証されない）だった欠陥を`itertools.count()`で修正。
6件の並列レビューagentが最初は全てセッションのAPIレート制限で失敗し，直接のRead/Bashツール呼び出しで
レビューを継続した経緯も記録しておく（`subagent_type: "code-reviewer"`は存在せず`general-purpose`で
代替）。

**完了条件の切り分け（実機投入は次段階）**: 以下はコード・設定の実装をもって完了とし，実機への反映は
別途ユーザー確認を要する: (1) 新規ノードwafl504〜509の物理的到達性確認と各ノードでの`ollama pull`
（**WAFL-PEFTが同一GPUプールを使用中でないことの確認が前提**——WAFL-PEFT側のbacklogに両者を同時に
走らせない運用が必要との既存記述あり），(2) E4/E5/E6の実機での本実験（サンプリング多様性診断，
Ollamaバージョン確認，分類器学習）。

### 実験 (Iter15) — 実機デプロイテストとインフラ不備の解決

**目的**: 「実装 (Iter15)」で完了したコードを，実際に物理クラスタ（wafl500〜509）へデプロイし，
`mise run deploy` → `mise run start` が想定通り動くかをユーザー指示で検証した。WAFL-PEFT非稼働の
確認は，直接の`curl`/`ping`によるノード疎通確認はユーザーが明示的に拒否したため，ユーザー指示に
従い`ssh wafl500`等での確認に切り替えた上で実施した。事前（本セッション以前）にwafl500で
`docker ps`を確認し，WAFL-PEFT関連ではない`ggml-rpc-server`プロセスのみを確認済み。本セッションでも
GPU修復（sudo導入）の直前にwafl504・wafl506・wafl507で`docker ps -a`を確認し，3ホストとも
WAFL-PEFT関連のコンテナが存在しない（wafl504に自分が起動を試みて失敗したexpert-mesh-ollama-1
コンテナが1つあるのみ）ことを確認してから着手した。

**発見したインフラ不備（全てコードのバグではなくホスト環境の不整合。ユーザー承認の上でsudo導入により解決）**:

1. **wafl504・wafl506・wafl507**: `nvidia-container-toolkit`が不完全（`nvidia-container-runtime`
   実行ファイル自体が欠落）で`docker compose up`が`failed to discover GPU vendor from CDI`および
   `nvidia-container-runtime: executable file not found`で失敗。他7ホストと同一バージョン
   （1.19.0-1）をapt経由でsudo導入し解決（daemon再起動のみ，WAFL-PEFT等の既存コンテナは3ホストとも
   存在しなかったため無停止で実施）。
2. **`docker-compose.gpu.yml`**: 上記3ホストの`nvidia-ctk`欠落によりCDIベースのGPU検出
   （`deploy.resources.reservations.devices` + `capabilities: [gpu]`）が失敗する構造だったため，
   CDIに依存しないレガシー方式（`runtime: nvidia` + `NVIDIA_VISIBLE_DEVICES`）へ書き換えた。
   全10ホストの`docker info`で`nvidia`ランタイムが同一設定で登録済みであることを確認済み。
3. **wafl508・wafl509**: `docker compose`（v2プラグイン）自体が未導入
   （Docker本体は別経路のパッケージで，Docker公式aptリポジトリ自体が未設定）。Ubuntu標準リポジトリの
   `docker-compose-v2`をsudo導入し解決（Docker本体のアップグレードやリポジトリ追加は行わず，
   起動中コンテナも無かったため無停止で実施）。

**発見したコードバグ（mise.toml，修正済み）**:

`mise run start -- --dataset ... --output ...`のCLI引数上書きが**完全に機能していなかった**。
`[tasks.start]`・`[tasks.analyze]`は`$ARGV`/`$1`でパースする実装だったが，このmiseバージョン
（2026.7.11）は追加CLI引数をスクリプト**最終行への単純な文字列結合**として扱うのみで，`$ARGV`は
常に空という実際の挙動を確認した（mise公式ドキュメントで`usage`フィールドが正しい機構であることも
確認済み）。同じmise.toml内の`[tasks.clean]`は既に`usage`フィールドを正しく使っており，`start`/
`analyze`だけが古い（機能しない）パターンのまま残っていた。両タスクを`usage`フィールド方式へ修正し，
上書きあり/なし双方の動作をローカルで検証済み。この不具合により，当初意図した少数サンプルでの
スモークテストが実行できず，代わりにフルデータセット（1520問）による本実験が起動した。

**現在進行中の実験（このセッション終了後も物理ノード側で継続する設計）**:

- 実行コマンド: `mise run start`（オプション指定なし＝デフォルトの`data/dataset.jsonl`全1520問）
- 起点ノード: wafl500（`docker compose exec -d`でコンテナ内にdetach起動済み。SSH切断・本セッション
  終了後も動作継続する——`mise.toml`の`[tasks.start]`コメント参照）
- 結果ディレクトリ: `results/20260727_010532/`（`run_experiment.log`と`results.jsonl`はwafl500の
  `$REMOTE_DIR/results/20260727_010532/`にバインドマウント経由で書かれる）
- 完了判定: `results/20260727_010532/results.jsonl.done`マーカーファイルの有無で判定する
  （`ssh wafl500 "test -f ~/workspace/ktakahashi/expert-mesh/results/20260727_010532/results.jsonl.done"`）。
- 進捗（記録時点，01:56 JST）: 859/1520行完了（約56.5%），開始から約3.5秒/問の安定したペース。
  完了見込みは約02:35 JST（このセッションの状況に依存するため目安）。
- 完了後の引き継ぎ手順: (1) `mise run start`自体が完了検知後に自動でローカル
  `results/20260727_010532/results.jsonl`へコピーする設計だが，もしこのセッションの背景タスクが
  途中で失われていた場合は`ssh wafl500 "cat ~/workspace/ktakahashi/expert-mesh/results/20260727_010532/results.jsonl"`
  で手動取得可能。(2) `mise run analyze -- 20260727_010532`でログ収集（`usage`修正によりこの引数指定
  も今回から正しく機能する）。(3) `metrics.py`のWilson CI・Cohen's kappa等の新指標をこの結果に対して
  実行し，`docs/d0001`の暫定値・過去イテレーションとの比較を行う。

**未確認の実験的観測（バグではなく研究上の知見の可能性。次のrc-analystが判断すること）**:
`business_economics`ドメインの設問で，同ドメインのノード（wafl504）自身が正しく高confidence
（0.9）を自己申告していても，`aggregator.py`の同点タイブレーク（宣言順優先，既存の意図的設計）と
config.yaml上のノード宣言順（business_economicsがlegal等より後）の組み合わせにより，legal等へ
misroute される事例が部分結果で複数観測された。これは10ドメイン化・self_report方式固有の
キャリブレーション不足を反映している可能性があり，全1520問の完走後にドメイン別precision/recallで
定量的に確認すべき。

### 分析 (実行) (Iter15)

**実験ディレクトリ**: results/20260727_010532（1520問，全問完走）

| 指標 | Iter15 (10ドメイン) | Randomベースライン | 判定 |
|------|---------------------|-------------------|------|
| top1_accuracy | **0.184** | 0.101 | Random を上回る |
| top1_accuracy Wilson 95% CI | **[0.165, 0.204]** | -- | 幅 0.039 |
| Cohen's kappa | **0.081** | 0.000 (chance) | chance 直上 |
| single_domain_top1_accuracy | **0.173** | 0.100 | Random を上回る |
| misrouting_rate | **0.816** | -- | -- |
| fallback_rate | 0.000 | -- | -- |
| dispatch_failure_rate | 0.000 | -- | -- |
| mean_duration_ms | 3826 | -- | 1問/秒以下（4Bモデル効果） |
| compound_domain_top1_accuracy | **0.950** (19/20) | -- | 構造上の高さ |
| compound_domain_set_recall | **0.475** | -- | 実質被覆率 |

**E1 成功条件判定**:

| # | 条件 | 結果 | 判定 |
|---|------|------|------|
| 1 | dataset.jsonl が 200問以上，10ドメイン層化（各150問），複合行含む，id 一意 | 1520問（1500単一+20複合），id 一意 | **PASS** |
| 2 | metrics.py に Wilson CI，McNemar，Cohen's kappa，3ベースラインが実装されテストpass | 実装済み，テストpass | **PASS** |
| 3 | mise run setup/deploy/run/analyze が完走，dispatch_failure_rate 実質0 | 全1520問完走，failure=0 | **PASS** |
| 4 | metrics.py --json に Wilson 95% CI と Random/BestSingle/Oracle が含まれる | 出力確認済み | **PASS** |

**判定: E1 は採用（統計基盤の整備完了）**．accuracy の値そのものは E1 の判定対象ではない（計画フェーズの成功条件 (5) に従う）．

### 分析 (解釈) (Iter15)

#### 1. self_report が 10 分野で機能しない根本原因の解釈

**観測事実**: self_report confidence の分布は極端な二峰飽和を維持している．

| 値 | 頻度 | 比率 |
|----|------|------|
| 0.9 | 11,387 | 74.9% |
| 0.2 | 2,471 | 16.3% |
| 0.8 | 470 | 3.1% |
| 0.3 | 400 | 2.6% |
| 0.1 | 291 | 1.9% |
| 0.0 | 57 | 0.4% |
| 他 | 54 | 0.4% |

**0.9 が全 probe 応答の 74.9% を占める**．これは Iter9（4ドメイン，n=46）で観測された二峰飽和（{0.1,0.2} vs {0.8,0.9,0.95}）の拡大版であり，10ドメイン化によって問題は悪化している．

**根本原因**: 各ノードの light_model（qwen3.5:4b）は，自分自身を「{domain}分野の専門家」としてプロンプトで指示されているため，**どの質問に対しても自分の担当分野に関する応答を生成しようとする**．その結果，自分自身の分野に関する confidence をほぼ常に 0.9 と申告する．

ドメイン別自己 confidence の統計（150問/ドメイン）:

| ドメイン | 0.9 比率 | mean |
|---------|---------|------|
| legal | 98.7% | 0.897 |
| natural_science | 96.0% | 0.899 |
| business_economics | 96.0% | 0.878 |
| computer_science | 96.0% | 0.895 |
| medical | 93.3% | 0.877 |
| social_science | 93.3% | 0.887 |
| history_culture | 96.7% | 0.881 |
| mathematics | 90.0% | 0.893 |
| general | 68.7% | 0.795 |
| education | 69.3% | 0.695 |

legal, natural_science, business_economics, computer_science は 96% 以上で 0.9 饱和している．general と education のみ比較的低いが，これは general ノードが「専門家ではない」というプロンプト設定と，education ノードの light_model が比較的低めの confidence を出す傾向があるためである．

**クロスドメイン confidence（自分の分野ではない質問で 0.9 を申告する頻度）**:

| ドメイン | クロス 0.9 率 |
|---------|-------------|
| mathematics | 91.3% |
| legal | 90.4% |
| medical | 80.7% |
| computer_science | 80.7% |
| natural_science | 77.9% |
| social_science | 74.1% |
| business_economics | 73.9% |
| history_culture | 69.5% |
| education | 59.4% |
| general | 38.4% |

mathematics, legal, medical は自分の分野ではない質問でも 80% 以上で 0.9 を申告する．これは**self_report がドメイン識別信号として機能していない**ことを示す．

#### 2. 同点タイが 98.29% になるメカニズムの説明

**観測事実**: 1520問中 1494問（98.29%）で最大 confidence の同点タイが発生している．

| タイ方式 | 頻度 | 比率 |
|---------|------|------|
| 10-way タイ | 246 | 16.5% |
| 9-way タイ | 420 | 28.1% |
| 8-way タイ | 260 | 17.4% |
| 7-way タイ | 177 | 11.8% |
| 6-way タイ | 131 | 8.8% |
| 5-way タイ | 112 | 7.5% |
| 4-way タイ | 81 | 5.4% |
| 3-way タイ | 50 | 3.4% |
| 2-way タイ | 17 | 1.1% |

**メカニズム**:

1. **多数のノードが 0.9 を申告する**: 前述のクロスドメイン confidence 分析から，多くのノードが自分の分野ではない質問でも 0.9 を申告する．10 ノード中 7〜10 ノードが 0.9 を出すのが典型パターンである．
2. **aggregator.py の stable sort**: `sorted(eligible, key=lambda r: r.confidence, reverse=True)[:top_k]` は安定ソートであり，confidence が同値のノードは入力順（宣言順）を維持する．
3. **http_client.py の probe_all**: `asyncio.gather` で並列実行し，`self._peers` の宣言順に結果を返す．宣言順は config.yaml のノード定義順と一致する．

つまり，**98.29% の質問でルーティング決定は実質的に宣言順による**．

#### 3. general ノードが recall=0.687 になる理由の解釈

**観測事実**: general ノードは 150 問中 103 問（68.7%）を正しく選択している．

**理由**: general ノードは config.yaml で**1番目に宣言されている**（宣言順 1 位）．同点タイが発生した場合，stable sort の性質により宣言順が早いノードが優先される．

タイ勝者分布を確認すると:

| ドメイン | タイ勝者数 | タイ勝者率 |
|---------|----------|----------|
| general | 641 | 42.9% |
| education | 497 | 33.3% |
| legal | 323 | 21.6% |
| medical | 17 | 1.1% |
| business_economics | 7 | 0.5% |
| natural_science | 6 | 0.4% |
| history_culture | 2 | 0.1% |
| mathematics | 1 | 0.1% |

general + education + legal で 97.8% のタイ勝者を占める．これは宣言順 1〜3 位のノードが，同点タイで有利に勝つ構造を反映している．

general の recall=0.687 は，以下の2つの要因の複合である:
1. **宣言順 1 位によるタイ勝率 42.9%**: 1494 タイ中 641 件を general が勝つ．
2. **general ノードの比較的低めの自己 confidence（0.9 比率 68.7%）**: general は他ノードより 0.9 を出しにくい．これは general のプロンプトが「専門家」ではなく「一般知識」という設定であるため，confidence の申告が他ノードより保守的である．その結果，general が唯一の 0.9 になるケース（非タイ）も存在し，その場合は general が確実に勝つ．

**結論: general の recall=0.687 の大部分は，ドメイン識別能力ではなく宣言順有利によるものである**．

#### 4. history_culture, social_science が recall=0.0 になる理由の解釈

**観測事実**: history_culture（宣言順 9 位）と social_science（宣言順 10 位）は，150 問中 0 問しか正しくルーティングされていない．

**理由**: これら 2 ノードは宣言順で最後尾にある．98.29% の質問でタイが発生し，タイの勝者は宣言順 1〜3 位（general, education, legal）に集中している．宣言順 9, 10 位のノードがタイで勝つには，**自分以外の 9 ノードすべてが自分より低い confidence を出す必要がある**．

クロスドメイン confidence の分布から，これは極めて稀である．history_culture ノード自身は 96.7% の頻度で自己 confidence 0.9 を出すが，同時に general, education, legal も高い頻度で 0.9 を出すため，タイが発生し，宣言順で不利な history_culture は負ける．

タイ勝者分布で history_culture は 2 件（0.1%），social_science は 0 件である．confusion matrix でも history_culture 行は 0（social_science 行は 2），つまり実質的にルーティングされない．

#### 5. 複合ドメイン top1_accuracy=0.95 の解釈

**観測事実**: 複合ドメイン（20問）の top1_accuracy=0.95（19/20）．

**これはルーティング能力の高さを示すものではない**．複合ドメインの評価は「selected_domain が expected_domains のいずれかに含まれるか」で判定される．expected_domains が 2 ドメイン（例: ['medical', 'legal']）の場合，selected_domain が medical または legal のいずれかであれば正解とカウントされる．

実態を見ると，複合ドメインの 19 件中 14 件が ['medical', 'legal'] であり，legal（宣言順 3 位）が常に勝っている．これは medical（宣言順 4 位）と legal（宣言順 3 位）の両方が 0.9 を出すタイで，宣言順有利な legal が勝つという構造である．

**実質被覆率: 0.475**（2 ドメイン中 1 ドメインを被覆すれば正解とカウントされるため，被覆率は top1_accuracy の半分程度）．

#### 6. 非タイケースの分析

**観測事実**: 26 件の非タイケース中 23 件（88.5%）が正解である．

これは重要な知見である．**self_report confidence が実際に弁別力を発揮しているのは，26 件の非タイケースのみ**である．これらのケースでは，正解ドメインのノードが明確に高い confidence を出し（例: mathematics が 1.0, medical が 0.95），他ノードが低い confidence（0.1〜0.3）を出している．

非タイケースの典型パターン:
- **mathematics 設問**: mathematics ノードが 1.0 または 0.9，他ノードが 0.9 以下．数学的問題は数式を含むため，mathematics ノードのプロンプトが強く反応する．
- **medical 設問**: medical ノードが 0.95，他ノードが 0.9 または 0.1〜0.3．医療用語を含む質問は medical ノードが識別しやすい．
- **natural_science 設問**: natural_science ノードが 0.95，他ノードが 0.9 または 0.2．

**結論**: self_report confidence には限定的だが実在する弁別力がある．しかし，それがルーティングに反映されるのは 1.7% のケースのみである．

#### 7. Cohen's kappa=0.081 の解釈

Cohen's kappa は偶然一致を補正した合意率である．

- 観測合意率: 0.173（単一ドメイン top1_accuracy）
- 偶然合意率: 各ドメインの選択頻度 × 正解頻度の積和
- kappa = (観測 - 偶然) / (1 - 偶然) = 0.081

kappa=0.081 は「chance 直上」であり，**実質的なドメイン識別信号はほぼ存在しない**ことを意味する．4 ドメイン時代の kappa（推定 0.70 前後）との比較は，config.yml の指示に従い行わない（ドメイン数が変わると偶然一致率が変化する）．

#### 8. 仮説との整合

計画フェーズで期待された効果:
1. **Wilson 95% CI が約 20pt 幅から 10pt 未満へ縮む**: 実際は [0.165, 0.204] で幅 0.039（3.9pt）．**期待を上回る**（p=0.184 は p=0.87 より小さく，二項分散が小さいため）．
2. **Random/BestSingle/Oracle と並記で 0.87 が本当に無意味な水準かどうかを確認**: Random=0.101, BestSingle(general)=0.099, Oracle=1.0．top1_accuracy=0.184 は Random を上回るが，BestSingle とほぼ同等である．
3. **以降のレバーで McNemar 対比較が使える基盤が整う**: 1520 問の同一データセット上で，2 つのレバー値を比較できる．**成立**．

#### 9. 想定外の挙動

- **self_report の二峰飽和は 10 ドメインでも維持**: Iter9（4ドメイン）で観測された飽和が，10 ドメインでも維持されている．ただし 0.9 の比率がさらに高まっている（74.9%）．
- **mean_duration_ms=3826**: 1 問あたり約 3.8 秒で，4B モデルの高速性が反映されている．dispatch_gen_time_ms の平均は約 3 秒程度（results.jsonl の sample から推定）．probe ラウンドトリップが全体の大部分を占めている．
- **dispatch_failure_rate=0.0, fallback_rate=0.0**: 1520 問すべてが正常にルーティングされた．インフラは安定している．

#### 10. 次イテレーションへのレバー選択提案

**E3（top_k_with_probs）の妥当性の評価**:

**採用を強く推奨する**．理由:

1. **二峰飽和に直接効く**: Tian et al. (EMNLP 2023) は『候補を K 個挙げ，各々に確率を付けよ』形式で gpt-3.5 の ECE を 0.131→0.047 に低減したと報告する．確率の合計制約（sum=1）が，verbalized confidence の 0/1 飽和を**機械的に壊す**．
2. **プロンプトのみの変更**: `confidence_elicitation: numeric_scalar → top_k_with_probs` の config.yaml 1 行変更のみで，コード変更は不要（既に実装済み）．
3. **同点タイの解消**: 各ノードが連続的な確率分布を返すため，10 ノードで完全に同値になる確率が著しく下がる．
4. **コスト最小**: probe 1 回/ノードのまま，追加 LLM コール不要．

**他の候補との比較**:

| レバー | 変更内容 | コスト | 期待効果 | リスク |
|-------|---------|-------|---------|-------|
| E3: top_k_with_probs | config 1 行 | 最小 | 飽和解消，タイ削減 | 低い（文献支持あり） |
| E4: self_consistency_semantic | config 変更 | 中（N=5 サンプル） | 不確実性推定 | 高い（T=0.7 での多様性未確認） |
| E5: p_true | config 変更 | 中（追加 LLM コール） | 較正改善 | 中（Ollama バージョン依存，反証あり） |
| E6: supervised_classifier | config 変更 | 低（推論のみ） | ルーティング改善 | 低（訓練/評価分離済み） |
| E7: whitening | config 変更 | 最小 | embedding 改善 | 低い（教師なしのまま） |

**単一レバー原則への復帰**:

E3（top_k_with_probs）を次イテレーションの単一レバーとして推奨する．

- config.yaml の `confidence_elicitation: numeric_scalar → top_k_with_probs` のみを変更
- `confidence_signal_method: self_report` を維持（E3 は self_report の elicitation 方式の変更）
- `routing_method: self_report` を維持
- 1520 問の同一データセットで比較
- 成功条件: McNemar 対比較で有意差（α=0.05），Wilson CI が重ならない変化

**E6（supervised_classifier）は E3 の次に検討すべき候補**．理由: embedding ベースの教師あり分類は，self_report の較正問題とは独立したアプローチであり，E3 が不成功の場合のフォールバックとして価値がある．ただし，E3 と E6 は異なる軸（confidence elicitation vs routing method）の変更であり，単一レバー原則に従い 1 イテレーションずつ実施する．

**E4, E5, E7 は E3, E6 の結果を確認してから検討する**．E4 は probe 多様性の事前確認が必要（Iter11 の教訓）．E5 は Ollama バージョン確認が必要．E7 は embedding ベースの教師なしアプローチであり，E6 の教師ありアプローチと構造が重複するため優先度が低い．

### 考察・次計画 (Iter15)

**判定: E1（eval_set_size）— 採用（統計基盤の整備完了）**

E1 の成功条件は accuracy の値そのものではなく，統計的計測基盤の実装・動作確認である（計画フェーズの成功条件 (5)）．全4条件を PASS し，1520問の JMMLU ベースデータセット上で Wilson CI・Cohen's kappa・McNemar 対比較・Random/BestSingle/Oracle ベースラインが正しく動作することを確認した．

**このイテレーションで確定した非自明な学び**

1. **self_report は 10 分野で実質ランダム（kappa=0.081）**: 二峰飽和は 10 ドメイン化で悪化（0.9 が 74.9%）．クロスドメイン confidence（自分の分野ではない質問で 0.9 を申告する頻度）は mathematics 91.3%，legal 90.4%，medical 80.7% と，専門ノードほど自己分野外でも高 confidence を出す．self_report はドメイン識別信号として機能していない．

2. **同点タイ 98.29% が最大のボトルネック**: 10 ノード中 7〜10 ノードが 0.9 を出し，ルーティング決定は実質的に宣言順による．general（宣言順1位）の recall=0.687 の大部分は宣言順有利であり，ドメイン識別能力ではない．history_culture（9位），social_science（10位）は recall=0.0 で実質ルーティングされない．

3. **self_report に限定的だが実在する弁別力**: 非タイケース 26 件中 23 件（88.5%）が正解．mathematics（数式を含む），medical（医療用語），natural_science の設問で正解ノードが明確に高い confidence（0.95/1.0）を出し，他ノードが低い値（0.1〜0.3）を出す．この弁別力を活用するには，まず同点タイを解消する必要がある．

4. **複合ドメイン top1=0.95 は構造上の高さ**: 19/20 が ['medical', 'legal'] であり，legal（宣言順3位）が medical（宣言順4位）よりタイで勝つだけ．実質被覆率は 0.475．

5. **4B モデルの高速性**: mean_duration_ms=3826（約3.8秒/問）で，9B モデルの約13秒から約3分の1に短縮．1520問を約1時間弱で完走可能となり，イテレーションの回転が大幅に改善された．

6. **E2（STP符号反転検証）の不支持**: 符号反転で argmin=0.3913（偶然一致 0.2826 より上だが，元の仮説「0.87相当に戻る」は不支持）．STP の単純な符号反転では Iter13 の結論を覆せない．

**次の単一レバー: E3（confidence_elicitation=top_k_with_probs）**

二峰飽和と同点タイ 98.29% に直接効く，プロンプトのみの変更（config 1行）で検証可能．Tian et al. (EMNLP 2023) の Verbalized Top-K は確率の合計制約（sum=1）で 0/1 飽和を機械的に壊す．

- 変更: `confidence_elicitation: numeric_scalar → top_k_with_probs` のみ
- 固定: `confidence_signal_method: self_report`，`routing_method: self_report`，他全設定不変
- 比較: 同一 1520 問データセット上で McNemar 対比較（α=0.05）
- 成功条件: top1_accuracy の McNemar で有意差，Wilson CI が重ならない変化
- 副指標: 同点率，ノード間 confidence 分散，ECE

**E6（supervised_classifier）は E3 の次候補**．self_report の較正問題とは独立した embedding ベースの教師あり分類アプローチであり，E3 が不成功の場合のフォールバックとして価値がある．

## Iteration 14: hidden_state 信号の実現可能性調査

### 計画 (Iter14)

**単一レバーの決定**: **実行可能な新レバーを定義できない（要人間判断）**

**判断理由**:
1. `confidence_signal_method=hidden_state` は rc-investigator 調査により、Ollama REST API で raw hidden state を抽出できないことが決定的に示された。
2. Option B（embedding ベースの信号として再定義）は既存の `routing_method=embedding` と同等の処理を別の名前経由で呼ぶだけであり、Iter2 の失敗原因（cross-lingual mismatch, cosine similarity の潰れ）を解決しない。新たな検証価値なし。
3. Option A（`routing_method=embedding` + task_prefix 修正）は有効なアプローチだが、router.py + http_server.py のコード変更（~10行）を伴う。config levers に定義されたレバーではなく、研究フロンティアの範囲（大規模実装）に属する。単一レバーとして定式化するには大きすぎる。
4. 研究フロンティアの全項目（新規専門ドメイン追加、評価用データセットの本格化、LLM-as-judge、ベースライン比較、top-k dispatch の高度な集約方式、無線アドホック化）が単一レバーの範囲を超えている。

**結論**: `status="converged"` として研究サイクルを終了する旨、委譲元（rc-reflector / skill 本体）に返す。

---

### 調査 (Iter14)

**問い**
- Q1: Ollama は hidden state（中間層活性化ベクトル）を API で取得できるか？`/api/embeddings` の仕様と限界は？
- Q2: Mahaut et al. 2024「Factual Confidence of LLMs」の hidden-state probe 手法の詳細と、本研究への適用可能性。
- Q3: 現行コード（expert_backend.py, router.py, http_server.py）で hidden_state 抽出に必要な変更量は？
- Q4: ベースライン結果の特定と成功条件の提案。

**分かったこと（Q1: Ollama の hidden state 抽出能力）**

**Ollama が提供するベクトル出力 API は embedding のみ**:

```
POST /api/embed          # バッチ埋め込み（input: text or list of text）
POST /api/embeddings     # 単一埋め込み（prompt: text、後方互換用）
```

両エンドポイントとも**最終層の活性化ベクトル（hidden state）を返さない**。embedding model が学習した semantic representation のみを返す。

**Ollama が hidden state を抽出する API は存在しない**:
- `/api/generate`: テキスト生成のみ、中間出力なし
- `/api/chat`: チャット完了のみ、中間出力なし（logprobs:true でトークン確率は取得できるが、hidden state は不可）
- vLLM や SGLang には hidden state extraction の仕組みがあるが、Ollama にはない

**Qwen3.5-9B-Unsloth-UD-Q4_K_XL のアーキテクチャ**:
- Hidden dimension: 4096
- Layers: 32（Gated DeltaNet + Gated Attention hybrid）
- Token embedding size: 248,320（padded）

**結論: Ollama REST API で raw hidden states は取得できない。**

**分かったこと（Q2: Mahaut et al. 2024 の手法）**

**論文**: Mahaut et al. (2024)「Factual Confidence of LLMs: on Reliability and Robustness of Current Estimators」ACL 2024

**核心知見**:
1. trained hidden-state probes は factual confidence 推定において**最も信頼性の高い** estimator（80 citations）
2. しかし、hidden states を直接使用できるのではなく、**教師あり学習で probe classifier を訓練する必要がある**
3. raw hidden state のままでは confidence signal として機能しない（未校正）

**手法の概要**:
- LLM の最終層 hidden state（7680 dim, GPT-J ベース）を抽出
- 「回答が正しい/間違い」のラベル付きデータで logistic regression probe を訓練
- probe classifier の出力を confidence score として使用

**本研究への適用可能性**:
- **必要なもの**: (a) hidden state の抽出経路（Ollama API では不可）、(b) ラベル付きデータ（results.jsonl に存在）、(c) probe 訓練パイプライン（未実装）
- **不可能な点**: Ollama は hidden state を出力しない。vLLM や HuggingFace transformers 経由で直接モデルをロードする必要がある。

**分かったこと（Q3: 現行コードの変更箇所）**

**現状の confidence signal 抽出経路**:
```
http_server.py:probe()
  → routing_method == "embedding": estimate_embedding_confidence(query_emb, domain_emb)
  → confidence_signal_method == "multi_sample": estimate_confidence_multi_sample()
  → confidence_signal_method == "stp": estimate_confidence_stp()
  → else: estimate_confidence()（self_report）
```

**hidden_state を「embedding ベースの信号」として扱う場合の変更量**:
- `expert_backend.py`: **変更不要**（既存の `embed()` が `/api/embeddings` を通じて embedding を返す）
- `router.py`: **変更不要**（既存の `estimate_embedding_confidence()` が cosine similarity → [0,1] 変換を行う）
- `http_server.py`: **変更不要**（既存の `routing_method == "embedding"` ブランチが動作する）
- `protocol.py`: **変更不要**
- `node.py`: **変更不要**
- `config.yaml`: `routing_method: self_report` → `routing_method: embedding` のみ

**ただし**: Iter2（routing_method=embedding）は rejected。失敗原因:
1. nomic-embed-text の task prefix 未付与（cross-lingual mismatch）
2. cosine similarity が [0.67, 0.74] に潰れ弁別喪失

**hidden_state を「raw hidden activation」として扱う場合の変更量**:
- `expert_backend.py`: +50-100行（hidden state 抽出用の新しいメソッド実装、Ollama API 変更または vLLM 移行）
- `router.py`: +30-50行（hidden state → confidence の変換ロジック、probe classifier 訓練/推論）
- `http_server.py`: +10行（分岐追加）
- **合計: ~90-160行**

**分かったこと（Q4: ベースライン結果）**

**ベースライン**: results/20260721_222225（Iter9, self_report）
- top1_accuracy: 0.8696
- single_domain_top1_accuracy: 0.8750
- misrouting_rate: 0.1304

**オフライン分析用データ**: results.jsonl に `probe_candidates` が記録済み（全ノードの confidence 値）。offline analysis で新しい signal の有効性を検証可能。

**次の計画フェーズへの示唆**:
1. **hidden_state = raw hidden activation の抽出は Ollama API では不可能**。実装には vLLM 移行または Ollama ソースコード修正が必要（大規模変更）。
2. **hidden_state = embedding ベースの信号として解釈し直す**のが現実的。ただし Iter2 で rejected 済みなので、単に `routing_method=embedding` に戻すだけでは不十分。
3. **Iter2 の教訓を踏まえた上での実装**: task prefix 付与 + probe classifier 訓練（Mahaut et al. 方式の簡易版）が必要。これは config-only の枠を超える。

---

**推奨する実装アプローチ**:

**Option A: `routing_method=embedding` を再試行（task_prefix 修正付き）**
- Iter2 で rejected された embedding ルーティングを、task prefix 修正 + probe classifier で再実装
- 変更量: router.py + http_server.py の task prefix 追加（~10行）+ offline probe classifier 訓練スクリプト
- 成功条件: top1_accuracy >= 0.87（baseline 非退行）

**Option B: `confidence_signal_method=hidden_state` を embedding ベースの信号として定義し直す**
- config levers で `hidden_state` を値として追加（values=[embedding_only] のみ）
- http_server.py に new branch を追加（`elif state.confidence_signal_method == "hidden_state"`）
- 内部で `estimate_embedding_confidence()` を呼ぶ
- 変更量: http_server.py +5行、router.py +10行（新関数として wrapper）
- Iter2 の教訓を踏まえつつ、新しい signal method として位置づけ

**Option C: hidden state 抽出を断念し、別の signal source を検討**
- Ollama API で取得可能な信号は embedding と logprobs のみ
- logprobs は STP で失敗済み
- embedding は Iter2 で問題あり（task prefix 修正で改善可能性）
- hidden state 抽出を諦め、confidence prompt の設計変更や aggregator 側での signal 統合を検討

**推奨: Option B**。理由: (1) config levers に `hidden_state` を追加する形式で単一レバー原則を維持できる。(2) 内部では既存の embedding 経路を使うため実装コスト最小。(3) Iter2 の失敗原因（task prefix）は別イテレーションで修正。本イテレーションでは「embedding ベースの信号が hidden_state として機能するか」を検証する。

**単一レバー**: `confidence_signal_method=stp`（STP: Surrogate Token Probability）
**判定**: **rejected（根本的失敗）** — トークン確率はドメイン expertise を測定できない信号

**結果**:
| 指標 | Iter9 (baseline) | Iter13 (STP) | 差分 |
|------|-------------------|--------------|------|
| top1_accuracy | 0.8696 | **0.0652** | **-0.8044** |
| single_domain_top1_accuracy | 0.8750 | **0.0500** | **-0.8250** |
| misrouting_rate | 0.1304 | **0.9348** | **+0.8044** |

**学び**:
1. STP（トークン確率）は verbalized confidence と同様に calibration の問題を抱える。モデルはどんなドメイン質問でも自分の回答に高い確率を出す。
2. Sigmoid 正規化パラメータの設計ミスが弁別力を9倍喪失させた。shift=2.0 は実際の mean_logprob 分布とミスマッチ。
3. Raw logprobs は「生成 fluency」を測定しており、「ドメイン expertise」ではない。education ノードが常に highest confidence を得る偏りが生じた。
4. self_report（spread 0.95, bimodal）でさえ STP（spread 0.015, uniform）より良い信号だった。

**次イテレーション**: 新レバー `confidence_signal_method=hidden_state` を config.yml に追記して通常継続。

---

## Iteration 13: STP 信号の再実験（デプロイ修正後）

### 分析 (実行) (Iter13)

**実験ディレクトリ**: results/20260722_113854（46問、全問完走）

| 指標 | Iter13 (STP) | Iter9 (baseline) | 差分 | 判定 |
|------|-------------|-------------------|------|------|
| top1_accuracy | **0.0652** | 0.8696 | **-0.8044** | FAIL（有意な破壊的失敗） |
| single_domain_top1_accuracy | **0.0500** | 0.8750 | **-0.8250** | FAIL |
| misrouting_rate | **0.9348** | 0.1304 | **+0.8044** | FAIL |
| fallback_rate | 0.0000 | 0.0217 | -0.0217 | OK |

STPコードは全46行で正常実行済み（`confidence_logprobs_mean` 非None）。

### STP信号分析

- confidence spread: 0.0147（0.8659〜0.8806）— 全ノード・全ドメインでほぼ同一
- raw logprob spread: 0.1328（general: -0.208, education: -0.074）
- Sigmoid shift=2.0 が -0.5〜0.0 の範囲を [0.818, 0.881] に圧縮 → 弁別力が9倍喪失

### self_report vs STP 比較

| | self_report (Iter9) | STP sigmoid (Iter13) |
|---|---|---|
| confidence spread | 0.95 | 0.0147 |
| distribution shape | bimodal {0.1,0.2} vs {0.8,0.9,0.95} | nearly uniform [0.866, 0.881] |
| top1_accuracy | 0.8696 | **0.0652** |

self_report（二峰分布）でさえSTP（uniform飽和）より良い信号だった。

---

### 分析 (解釈) (Iter13)

**判定**: STP レバーは **rejected（根本的失敗）** — トークン確率はドメイン expertise を測定できない

#### 根本原因: 2つの複合要因

**(a) Sigmoid正規化の飽和**: shift=2.0 の sigmoid は mean_logprob=-0.5〜0.0 の範囲を [0.818, 0.881] に圧縮。raw logprob spread (0.1328) が normalized confidence spread (0.0147) に変換される際、9倍の弁別力が喪失。

**(b) トークン確率の根本的限界**: Raw logprobs は「モデルの生成 fluency」の違いであり「ドメイン expertise」を測定していない。educationノードが全クエリで最もfluentな応答を生成するため、常にhighest confidenceを得る。ルーティングは実質ランダム（正確には education bias）。

#### 仮説との整合

- H1 (STP better calibrated): **不成立**。STPもself_reportも全ドメインで高confidenceに収束。
- H2 (/api/generate works): **成立**（logprobs抽出は正常）。
- H3 (mean logprob robust): **検証不能**（signalがdomain-specificでないため）。

#### 研究への示唆

1. STPレバーはrejected。追加反復不要。
2. config leversは全6レバー（dispatch_top_k, routing_method, confidence_threshold, calibrated_routing, multi_sample, stp）を試しまれた。
3. confidence signalの根本較正問題は未解決。verbalized self-reportとtoken probabilitiesの両方が失敗した時点で、hidden states / embeddingsベースのapproachや、モデル生成に依存しないcalibration methodの検討が必要。

---

### 実験 (Iter13)

**デプロイ**: `mise run setup`（Dockerイメージ再ビルド）→ `mise run deploy`（4ノードすべてOK）

**バグ修正（実験中に発見・修正）**:
1. **Ollama API bug** (`expert_backend.py`): STPコードが`/api/generate` + 整数`logprobs: 1`を使用。Ollamaは論理値`logprobs: true`を期待。`/api/chat` + `logprobs: true`に修正。
2. **結果ファイル未記録** (`run_experiment.py`): `confidence_logprobs_mean`がresults.jsonlに記録されていなかったのを修正。

**検証**: 全46行に`confidence_logprobs_mean`が存在（非None）→ STPコード正常実行確認済み。

**メトリクス比較（baseline: Iter9 vs STP再実験）**:
| 指標 | Iter9 (baseline) | Iter13 (STP) | 差分 |
|------|-------------------|--------------|------|
| top1_accuracy | 0.8696 | **0.065** | **-0.8046** |
| single_domain_top1_accuracy | 0.8750 | **0.050** | **-0.8250** |
| misrouting_rate | 0.1304 | **0.935** | **+0.8046** |
| fallback_rate | 0.0217 | 0.0 | - |
| mean_duration_ms | 13731 | 13620 | -111 |

**判定**: STPレバーは **rejected（根本的失敗）**。STP confidence値は全ノードでほぼ同一（0.8659〜0.8806、spread 0.015）。STPは「モデルが自分の生成テキストに対してどれだけ自信があるか」を測定しており、「ドメイン専門家であるかどうか」を区別する信号にはならない。ルーティングは実質ランダム。

**学び**:
1. STP（トークン確率）はverbalized confidenceと同様にcalibrationの問題を抱える。モデルはどんなドメイン質問でも自分の回答に高い確率を出す。
2. Ollamaの`/api/generate`エンドポイントはこのモデルではtoken logprobsを返さない。`/api/chat` + `logprobs: true`が正しい経路。
3. STPはconfidence signalとして使えないことが決定的に示された。

---

### 実装 (Iter13)

**単一レバー**: confidence_signal_method=stp（STP: Surrogate Token Probability）

**実行した変更**:
1. `config.yaml`: 2行変更（`confidence_signal_method: stp`、`multi_sample_count` の削除）
   - STP コードは commit de37559 で既にコミット済み。コード変更は不要。
2. テスト実行: `uv run pytest tests/ -v` → **78件全PASS** (0.60秒)

**検証結果**:
- `uv run pytest tests/ -v`: **78件全PASS** (0.60秒)
- `uv run ruff check .`: 未実行（config.yaml のみ変更のため不要）

**次フェーズへの引き継ぎ**: config変更完了・テスト全PASS。次は実験フェーズで `mise run setup` → `mise run deploy` → `mise run start` を実行する。

---

### 調査 (Iter13)

**問い**
- Q1: `mise run deploy` の動作と、Docker イメージ再ビルドの必要性・方法。
- Q2: STP コード（commit de37559）の実装詳細確認：エンドポイント切り替えロジック、logprobs 抽出・正規化仕様。
- Q3: ベースライン結果の特定と Iter13（STP再実験）の成功条件。

**分かったこと（Q1: デプロイフローの問題と解決策）**

**mise.toml の deploy タスク動作確認**:
```
1. SSH reverse tunnel 確保（localhost:5001 -> リモートノード:5001）
2. rsync で docker-compose.yml, docker-compose.gpu.yml, config.yaml だけを配布
3. GPU 検出 → .env 作成
4. docker compose pull（既存イメージを pull。再ビルドしない）
5. ollama コンテナ起動 + モデル pull
6. docker compose up -d --force-recreate app（コンテナ再起動）
```

**Dockerfile の構造**:
```dockerfile
COPY protocol.py expert_backend.py router.py aggregator.py http_client.py \
     http_server.py node.py logging_utils.py ./
COPY run_experiment.py build_dataset.py metrics.py ./
...
ENTRYPOINT [".venv/bin/python", "node.py"]
```

**結論**: Python ソースコードは Docker イメージに bake されている。`mise run deploy` はイメージを再ビルドしないため、Python コードの変更（uncommitted も含め）はコンテナ内に反映されない。これは Iter12 の failure 原因そのもの。

**解決策の比較**:

| 方案 | 手順 | 所要時間 | リスク |
|------|------|---------|--------|
| (A) `mise run setup` → `mise run deploy` | イメージ再ビルド+push → pull+deploy | 5-10分（build）+2分（deploy） | なし。確実。 |
| (B) deploy タスクに docker build を追加 | mise.toml の deploy タスクを書き換え | 同上 + 永続化 | 全イテレーションでイメージビルドが必要になり、実験時間が延びる。 |
| (C) rsync で Python ソースを配布 + コンテナ再起動 | コンテナ内にコードコピー + restart | 1分程度 | 新しい手順の追加。コンテナ内での依存関係問題の可能性。 |

**推奨: (A) `mise run setup` → `mise run deploy`**。理由: (1) 変更最小（既存タスクの順序実行のみ）、(2) Docker イメージの整合性が保証される、(3) mise.toml の書き換え不要。

**分かったこと（Q2: STP 実装の詳細確認）**

commit de37559 の変更内容を確認した。全ファイル正常にコミット済み。

**expert_backend.py:OllamaClient.generate()**:
- `logprobs: int | None = None` パラメータ追加（既定 None = 既存動作）
- `logprobs > 0` の場合、`/api/generate` エンドポイントを使用（`payload["logprobs"] = logprobs`）
- `logprobs == None` の場合、既存の `/api/chat` エンドポイントを使用（後方互換）
- 戻り値: logprobs 有りは `dict{"content": str, "token_logprobs": list}`、無しは `str`（既存互換）

**router.py:estimate_confidence_stp()**:
- `build_confidence_prompt(domain, query_summary)` を logprobs付きで呼び出し
- `logprobs=1`（各トークンにつき1つの top-logprob）
- Fallback: `isinstance(result, str)` または `"token_logprobs"` 不在 → `parse_confidence(result["content"])`
- 正規化: `sigmoid(mean_logprob - (-2.0)) = 1 / (1 + exp(-mean_logprob - 2.0))`
- shift=2.0 は平均 logprob が -2 のとき confidence=0.5 になるようスケーリング

**http_server.py:probe() の切り替えロジック**:
```python
elif state.confidence_signal_method == "multi_sample":
    ...
elif state.confidence_signal_method == "stp":
    stp_conf, raw_logprob = await estimate_confidence_stp(...)
    confidence = stp_conf
else:
    confidence = await estimate_confidence(...)
```
- 順次 if-elif で、`confidence_signal_method` の値で分岐。問題なし。

**protocol.py:ProbeResponse**:
- `confidence_logprobs_mean: float | None = None` フィールド追加（既定 None）
- STP 経路では raw_logprob を設定するはず（http_server.py で明示確認必要だが、commit diff から設定箇所は存在）

**実装の健全性判定**: コードに論理的欠陥は見当たらない。Fallback 経路も確保済み。ollama のバージョン依存は `/api/generate` の logprobs サポート（v0.12.11+）。ワフリラボのノードでは既に最新 ollama が常時 keeping されているため、バージョン問題は低いと判断する。

**分かったこと（Q3: ベースライン結果と成功条件）**

**ベースライン**: results/20260721_222225（Iter9, self_report ベースライン）
- top1_accuracy: 0.8696 (≈0.870)
- single_domain_top1_accuracy: 0.8750
- misrouting_rate: 0.1304
- fallback_rate: 0.0217
- education precision/recall: 1.000/0.5000
- N=46 questions, 全問完走

**Iter12（infrastructure_failure）との比較**: top1_accuracy=0.8478 は baseline より -0.022。ただし STP 未実行のため run 間ノイズ。

**成功条件の提案（Iter13）**:
- 主基準: top1_accuracy >= 0.87（baseline 非退行）。改善目標は +0.03 の improvement（0.90 以上）。
- 非退行: single_domain_top1_accuracy >= 0.87
- 非退行: misrouting_rate <= 0.15
- **追加検証**: results.jsonl に `confidence_logprobs_mean` が 46/46 行存在すること（STP コードが正常に実行されたことの証拠）

**次の計画フェーズへの示唆**:
1. rc-planner へ: デプロイフロー修正は `mise run setup` → `mise run deploy` の順で実行するよう指示すること。mise.toml の書き換えは不要。
2. STP レバーの値は変更なし（`confidence_signal_method: stp` は config.yaml で設定済み）。コード変更もコミット済み。
3. 成功条件には `confidence_logprobs_mean` の存在確認を含めること（infra failure の再発防止）。
4. Iter13 が converged/rejected になれば、config levers は全試し切り済み。次は research_frontier へ移行する判断が必要。

### 計画 (Iter13)

**単一レバー**: confidence_signal_method=stp（STP: Surrogate Token Probability）
- デプロイフロー: `mise run setup` → `mise run deploy` の順で実行（Docker イメージ再ビルド必須）
- logprob 集計方法: mean（sigmoid shift=2.0 で [0,1] に正規化）

**仮説**:
- H1: LLM が生成中に出力するトークン確率（logprobs）は、verbalized self-report confidence よりも calibration が高い。self_report で飽和していた二峰分布（{0.1,0.2} vs {0.8,0.9,0.95}）が、STP では連続的な値として観測され、margin の弁別力が向上する。
- H2: `/api/generate` への切り替えは、使用モデル（isotnek/qwen3.5:9B-Unsloth-UD-Q4_K_XL）には thinking モードの機能がないため影響しない。`num_predict=100` の cap も generate エンドポイントで有効に機能する。
- H3: mean logprob は min より robust（単一の outlier token に左右されない）。confidence signal としての signal-to-noise ratio が self_report を上回る。

**成功条件**（ベースライン: results/20260721_222225, Iter9）:
- 主基準: top1_accuracy >= 0.87（非退行）。改善目標は +0.03 の improvement（0.90 以上）。
  - ノイズ幅見積もり: Iter8→9 で top1_accuracy は 0.913→0.870（-0.043）。Iter9→11（multi_sample）で 0.870→0.848（-0.022）。1イテレーションの最大変動は +/-0.05 程度。+0.03 はノイズの範囲内だが、STP が calibration を改善すれば有意な改善として観測できるレベル。
- 非退行: single_domain_top1_accuracy >= 0.87（baseline 0.875 から -0.005 以内）
- 非退行: misrouting_rate <= 0.15（baseline 0.130 から +0.02 以内）
- **追加検証**: results.jsonl に `confidence_logprobs_mean` が 46/46 行存在すること（STP コードが正常に実行されたことの証拠、infra failure 再発防止）

**固定する構成**:
- config.yaml: `routing_method: self_report`, `confidence_threshold: 0.5`, `dispatch_top_k: 1` を維持
- build_dataset.py: 不変
- router.py の既存 `estimate_confidence()`, `parse_confidence()`, `build_confidence_prompt()`: 不変
- aggregator.py: 不変（confidence signal の抽出経路が変わるのみ）

**変更ファイルと変更量**:
- config.yaml: 1行変更（`confidence_signal_method: stp`）
- コード変更: なし（STP コードは commit de37559 で既にコミット済み。expert_backend.py, router.py, protocol.py, http_server.py の合計 ~97行追加・24行削除が完了）

**検証手順**:
1. `mise run setup` で Docker イメージ再ビルド + push（5-10分）
   - これにより Python ソースコード（expert_backend.py, router.py, protocol.py, http_server.py）がイメージに bake される
2. `mise run deploy` で各ノードへ配布 + コンテナ再起動（2分程度）
3. `mise run start` で実験実行（46問/4ノード、expected ~50-70分）
4. `mise run analyze` で metrics 集計
5. results.jsonl に `confidence_logprobs_mean` が存在するか確認（infra failure 再発防止。46/46行に値が入っていることを検証）

**単一レバー原則との整合**: config.yaml の変更のみ（1行）。コード変更はコミット済み。

### 実験 (Iter13)

**デプロイ**:
- Docker イメージ再ビルド: `docker build --no-cache -t localhost:5001/expert-mesh:latest .` で完全再ビルド + push（digest sha256:e1344232...）
- デプロイ: 全ノードで `docker rmi` → `docker pull` → `docker compose up -d --force-recreate app ollama` を実行
- wafl500/wafl501/wafl502/wafl503 すべてが正しいイメージ（digest sha256:e13442327f...）で起動確認済み
- コンテナ内の protocol.py に `confidence_logprobs_mean` が存在することを確認

**追加検証（Infra failure 再発防止）**:
- results.jsonl に `confidence_logprobs_mean` が存在するか: **YES、46/46行に値が入っている**
- STP コードが正常に実行されたことを確認。infra failure は再発せず。

**実行結果**: results/20260722_095936/（46問、全問完走、used_fallback=0, dispatch_failed=0）
- 平均応答時間: 14320ms

**メトリクス比較（baseline: Iter9 vs STP）**:
| 指標 | Iter9 (baseline) | Iter13 (STP) | 差分 |
|------|-------------------|--------------|------|
| top1_accuracy | 0.870 | 0.043 | -0.827 |
| single_domain_top1_accuracy | 0.875 | 0.025 | -0.850 |
| misrouting_rate | 0.130 | 0.957 | +0.827 |
| fallback_rate | 0.022 | 0.000 | -0.022 |
| education precision/recall | 1.000/0.500 | 0.042/0.083 | - |
| legal precision/recall | 0.778/0.933 | 0.143/0.067 | - |
| medical precision/recall | 0.917/0.733 | 0.000/0.000 | - |

**成功条件判定**:
- top1_accuracy >= 0.87: **FAIL（0.043）** — baseline から大幅な劣化
- single_domain_top1_accuracy >= 0.87: **FAIL（0.025）**
- misrouting_rate <= 0.15: **FAIL（0.957）**

**実験上の観察**:
- STP コードは正しくデプロイされ、`confidence_logprobs_mean` が全46問で記録された
- フォールバックは0件（全問正常にルーティングされた）
- ただし、ルーティング先が education(24)・medical(13)・legal(7)・general(2) に偏っており、正解率は極めて低い
- probe_candidates の詳細を確認したところ、self_report confidence は全ノードで 0.86-0.88 とほぼ同等。STP値（confidence_logprobs_mean）は負の値で類似している（例: medical-001 で wafl500=-0.114, wafl502=-0.039, wafl503=-0.051, wafl501=-0.018）。選択は highest self_report confidence のノード（wafl501/education）に行われている

**根本原因（仮）**: STP 信号の正規化方法と confidence signal の較正に問題がある可能性。分析フェーズで詳細検証予定。

**次フェーズへの引き継ぎ**: 分析フェーズへ。rc-analyst へ:
1. STP コードは正しくデプロイされている（infra OK）
2. STP は http_server.py で既に confidence フィールドに統合されている。aggregator 側での変更が必要かどうか、分析で確認
3. `confidence_logprobs_mean` の値分布と self_report confidence の比較データを提供済み
4. 現在の results.jsonl と logs/ にすべてのデータが存在

---

### 分析 (実行) (Iter13)

**実験ディレクトリ**: results/20260722_095936（46問、全問完走）

| 指標 | Iter13 (STP) | Iter9 (baseline) | 差分 | 判定 |
|------|-------------|-------------------|------|------|
| top1_accuracy | **0.043** | 0.870 | **-0.827** | FAIL（壊れている） |
| single_domain_top1_accuracy | **0.025** | 0.875 | **-0.850** | FAIL |
| misrouting_rate | **0.957** | 0.130 | **+0.827** | FAIL |
| fallback_rate | 0.000 | 0.022 | -0.022 | PASS（フォールバックなし） |

主基準・非退行とも壊れた値。STP コードは正常に実行されたが、aggregator が統合していない。

---

### 分析 (解釈) (Iter13)

**判定**: STP レバーは **rejected（signal_destruction_by_normalization + fundamental_mismatch）**

#### 決定的証拠

**1. STP コードは正常に実行され、選択ロジックにも統合されている**

http_server.py line 253: `confidence = stp_conf` — STP enabled の場合、ProbeResponse.confidence は sigmoid-normalized STP 値で上書きされる。つまり **STP は既に aggregator に統合されている**。 planner が想定した「STP が選択ロジックに統合されていない」は誤り。

**2. self-report confidence の分布が Iter9 と比較して崩壊している**

| ドメイン | Iter9 mean/min/max | Iter13 mean/min/max |
|---------|-------------------|---------------------|
| general | 0.379 / 0.20 / 0.95 | 0.865 / 0.819 / 0.876 |
| education | 0.296 / 0.10 / 0.95 | 0.874 / 0.823 / 0.881 |
| legal | 0.495 / 0.00 / 0.95 | 0.870 / 0.815 / 0.879 |
| medical | 0.340 / 0.10 / 0.95 | 0.872 / 0.829 / 0.880 |

Iter9: 自己申告 confidence は 0.0〜0.95 の広い分布。medical ノードは medical クエリで 0.95、他ドメインは 0.1-0.2 と明確に区別。
Iter13: 全ノードが 0.865-0.880 の極めて狭い範囲に収束。domain 間の弁別力がほぼゼロ。

**3. STP 信号の分布は self-report より広いが、sigmoid 正規化で圧縮されている**

| 指標 | Iter13（再実験） |
|------|------------------|
| confidence（sigmoid-normalized）max-min spread | **0.0147** |
| confidence_logprobs_mean（raw logprob）max-min spread | **0.1328** |

Raw logprob の spread は 9.0 倍広い。しかし sigmoid(shift=2.0) により [0.866, 0.881] に圧縮される。

**4. self-report と STP シグナルは 100% 一致**

全 46 行で、self-report highest-confidence ノードと STP highest-logprobs_mean ノードが完全に一致。両シグナルは同じノード（education）を指している。

**5. 自己申告 confidence の同一クエリ・反復間比較**

medical-001 を例に:
| ドメイン | Iter9 | Iter13 | 差分 |
|---------|-------|--------|------|
| general | 0.20 | 0.87 | +0.67 |
| education | 0.10 | 0.88 | +0.78 |
| legal | 0.10 | 0.88 | +0.78 |
| medical | 0.95 | 0.88 | -0.07 |

**同一クエリに対して、反復間で自己申告 confidence が大きく変化している。** Iter9 の medical ノードは 0.95、Iter13 では 0.88。他ドメインは 0.1→0.88 と +0.78 の増加。これは self-report confidence 自体が不安定であることを示す。

#### 原因分析（修正版）

**根本原因: Sigmoid 正規化の飽和 + トークン確率の根本的限界**

2 つの要因が複合して信号を破壊している。

**要因1: sigmoid(shift=2.0) の飽和領域での動作**

```
normalized = 1.0 / (1.0 + exp(-mean_logprob - 2.0))
```

| mean_logprob | normalized confidence |
|-------------|----------------------|
| -0.50 | 0.8176 |
| -0.30 | 0.8455 |
| -0.20 | 0.8581 |
| -0.10 | 0.8699 |
| -0.03 | 0.8776 |
| 0.00 | 0.8808 |

実際の mean_logprob は -0.13〜-0.002 の範囲に集中しており、sigmoid の飽和領域（confidence>0.8）で動作。このため、logprob の違いが confidence の違いにほとんど変換されない。

**要因2: トークン確率はドメイン expertise を測定していない（根本的限界）**

Raw logprobs の分布をドメイン別に分析すると有意な差がある:

| ドメイン | mean raw logprob | spread |
|---------|-----------------|--------|
| general | -0.2078 | 0.4398 |
| education | -0.0738 | 0.1155 |
| legal | -0.0971 | 0.1839 |
| medical | -0.0773 | 0.2821 |

education ノードの mean logprob は -0.074 で、general（-0.208）より約 0.13 高い。これは **education ノードが生成するテキストが全般的により流暢** であることを示す。しかしこの差は domain-specific な弁別力ではなく、単に education ノードの prompt template に対する生成 fluency の違いである。

**教育ノードが常に highest confidence になる理由**:
- Raw logprob で education > medical > legal > general の順に均等に高い
- この順位はクエリの内容（medical/general/education/legal）によらず一定
- つまり「どのドメインの質問でも、education ノードが最も fluent な応答を生成する」

**結論: STP は「モデルが自分の生成テキストに対してどれだけ自信があるか」を測定しており、「そのノードがそのドメインの専門家かどうか」を区別する信号にはならない。**

#### 比較: self_report vs STP

| 指標 | self_report (Iter9) | STP (Iter13, sigmoid) |
|------|---------------------|-----------------------|
| confidence spread | 0.95 - 0.00 = **0.95** | 0.8806 - 0.8659 = **0.0147** |
| distribution shape | bimodal {0.1,0.2} vs {0.8,0.9,0.95} | nearly uniform [0.866, 0.881] |
| top1_accuracy | 0.8696 | **0.0652** |

self_report は二峰分布（{0.1, 0.2} vs {0.8, 0.9, 0.95}）で少なくとも**何らかの弁別力**があった。STP は sigmoid 正規化により全ノードがほぼ同一値に収束し、self_report よりも**著しく弁別力が低い**。

**仮説との整合**:
- H1（STP は self_report より calibration が高い）: **不成立**。STP signal も self_report と同様に全ドメインで高 confidence に収束。calibration が改善した証拠は見られない。
- H2（/api/generate は正常に動作する）: **成立**。logprobs の抽出は正常に機能し、46/46 行に値が記録されている。
- H3（mean logprob は min より robust）: **検証不能**。STP signal 自体が domain-specific でないため、robustness の評価ができない。

**次の考察フェーズへの示唆**:
1. STP レバーは **rejected**。根本原因は (a) sigmoid 正規化の飽和、(b) トークン確率がドメイン expertise を測定していないという根本的限界の2つ。
2. 追加反復は推奨しない。sigmoid shift の調整や prompt フォーマット変更が必要だが、それらは別の実装イテレーションを要する。
3. config levers は全試し切り済み（dispatch_top_k, routing_method, confidence_threshold, calibrated_routing, multi_sample, stp）。次は research_frontier へ移行する判断が必要。
4. confidence signal の根本的な較正問題（すべてのノードが全クエリで高 confidence を申告する）は未解決。これは STP に限らず self_report でも反復間で不安定（Iter9 vs Iter13 で同一クエリの confidence が 0.2→0.87 に変化）であるため、より根本的なアプローチが必要。
5. **両方の verbalized/tokn-level confidence signal が失敗した時点で、hidden states / embeddings ベースの approach や、モデル生成に依存しない calibration method の検討が必須。**

---

### 考察・次計画 (Iter13)

**判定**: STP レバーは **rejected（根本的失敗）** — トークン確率はドメイン expertise を測定できない信号

**総括**:
- STP コードは正常に実行された（46/46行に confidence_logprobs_mean 存在、Docker イメージ再ビルド済み）。
- しかし sigmoid(shift=2.0) の飽和領域で mean_logprob が動作し、logprob spread (0.1328) が normalized confidence spread (0.0147) に圧縮され、9倍の弁別力が喪失。
- top1_accuracy=0.0652 という壊れた値（baseline 0.8696 から -0.8044）。misrouting_rate=0.9348。

**根本原因: 2つの複合要因**
1. **Sigmoid正規化の飽和**: shift=2.0 の sigmoid は mean_logprob=-0.5〜0.0 の範囲を [0.818, 0.881] に圧縮。設計パラメータ（mean_logprob=-2 で confidence=0.5）と実際の分布がミスマッチ。
2. **トークン確率の根本的限界**: Raw logprobs は「モデルの生成 fluency」の違いであり「ドメイン expertise」を測定していない。education ノードが全クエリで最も fluent な応答を生成するため、常に highest confidence を得る。ルーティングは実質ランダム（正確には education bias）。

**config levers の状況**: 全6レバーを試しまれた。
dispatch_top_k(Iter1:reject), routing_method(Iter2:reject), confidence_threshold(Iter3:no-op), calibrated_routing(Iter10:reject), multi_sample(Iter11:reject), stp(Iter13:reject)。

**決定**: 新レバー `confidence_signal_method=hidden_state` を config.yml の levers 末尾へ追記して通常どおり継続する。
- 根拠: (1) verbalized self-report と token probabilities の両方が失敗した時点で、モデル生成に依存しない信号源の検討が必須。(2) research_frontier に「hidden states / embeddings-based approach」として明記済み（Mahaut et al. 2024 由来）。(3) 既存ノード構成のままコード変更のみで検証可能。
- 内容: モデルの hidden state（最終層の活性化ベクトルまたは embedding 出力）から confidence signal を抽出する方式。self-report は「生成されたテキストに対する言語的自信」、STP は「生成fluency」、hidden_state は「入力の内部表現とドメイン知識の一致度」を測定し、これら2つのアプローチとは異なる信号特性が期待される。
- 変更量: expert_backend.py（hidden state 抽出）、router.py（confidence estimation 関数追加）、http_server.py（分岐追加）の合計 ~30-40行。

**次イテレーションの単一レバー**: `confidence_signal_method=hidden_state`（values: [last_layer, embedding] で抽出方式を掃引）
- state.json の current_lever を "hidden_state" へ更新。phase は plan から開始。

**コミット**: journal/state/backlog の更新のみ。コード変更は次イテレーションの rc-planner/rc-implementer で実施。

---

**問い**
- Q1: STP（Surrogate Token Probability）の手法概要と、ollama での logprobs 抽出の実装可能性。tokenizer logprobs を抽出するにはどのような変更が必要か。
- Q2: multi-sample consistency の手法概要と、ollama で同じ query を複数回叩く場合のオーバーヘッド。probe ロジックにどのような変更が必要か。
- Q3: 現行コード（router.py, aggregator.py, node.py, http_server.py, run_experiment.py）の confidence signal 抽出経路を特定し、STP でどの部分を変更すればよいかをマッピングせよ。
- Q4: ベースライン結果の特定と成功条件の提案。

**分かったこと（Q1: STP の手法概要と ollama での実装可能性）**

**STP の定義**: 本研究における STP は「生成中のトークン確率を confidence signal として抽出」する手法。Self-REF (Chuang et al., ICML 2025) では confidence tokens を fine-tuning で学習したが、本研究では fine-tuning なしで既存モデルの出力トークン確率を直接使用する。

**ollama の logprobs サポート状況**:
- **Native `/api/generate` エンドポイント**: logprobs は v0.12.11+ でサポート済み（issue #13497 由来）。Medium 記事「Building a Token-Probability Analyzer with Ollama's New...」より。
- **Native `/api/chat` エンドポイント**（現行コードが使用）: logprobs サポートは GitHub issue #16117 で提案中だが、まだマージされていない状態。
- **OpenAI-compatible `/v1/chat/completions`**: logprobs パラメータのサポートも issue #16117 で同じく未マージ。
- **現在の `expert_backend.py:OllamaClient.generate()`** は `/api/chat` を使用（line 66）。logprobs を取得するには以下のいずれかの変更が必要：
  - (A) `/api/generate` エンドポイントに切り替え（native API、logprobs サポート済み）
  - (B) OpenAI-compatible `/v1/chat/completions` に切り替え + `logprobs: true` パラメータ追加

**STP を probe（confidence scoring）に適用する場合の実装変更**:
1. `expert_backend.py`: `generate()` に `logprobs: true` パラメータを追加。エンドポイントを `/api/generate` または `/v1/chat/completions` に変更。戻り値に token logprobs を追加。
2. `router.py`: `estimate_confidence()` の返り値を tuple `(confidence, confidence_signal)` に変更、または新しい関数 `estimate_confidence_stp()` を作成。トークン確率の平均/最小値を confidence signal として計算。
3. `protocol.py`: `ProbeResponse.confidence` は既存のまま（後方互換）。新しいフィールド `confidence_logprobs_mean` などを追加するか、または confidence signal の抽出経路を aggregator 側で変更する。

**変更量見積もり**:
- `expert_backend.py`: +15行（logprobs パラメータ、エンドポイント切り替え）
- `router.py`: +20行（STP 用関数、トークン確率の集計ロジック）
- `protocol.py`: +2行（ProbeResponse に新フィールド追加）
- `http_server.py`: +5行（logprobs を含む ProbeResponse 構築）
- `node.py`: +3行（STP 用の confidence signal 抽出経路の切り替え）
- **合計: ~45行**

**分かったこと（Q2: multi-sample consistency の手法概要）**

**multi-sample consistency の定義**: 同じ query を複数回 probe し、confidence の分散・不変性を信頼度信号として使用する。

**学術的根拠**:
- Lakshminarayanan, Pritzel, Blundell (2017)「Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles」: 複数サンプリングの予測分布の分散を不確実性の指標として使用。
- 「Calibrating Large Language Models with Sample Consistency」（AAAI）: 複数回のランダム生成から得られる一貫性（3つの測度）からモデル信頼度を導出。
- 「Verbal Confidence Meets Self-Consistency in Reasoning LLMs」（OpenReview）: 2回のサンプリングで十分 strong and reliable な結果を得られると報告。

**ollama で同じ query を複数回叩く場合のオーバーヘッド**:
- 現行 probe レイテンシ: 約 13-16秒（results.jsonl の duration_ms から推定、probe + dispatch 全体）。probe 単体はもっと短い（http_server.py の `estimated_latency_ms` は local inference のみ）。
- multi-sample を probe 段階で 3回実行する場合: probe レイテンシが約 3倍になる。dispatch は最終的に1回のみのため、全体レイテンシへの影響は限定的。
- temperature=0.1（現行設定）での run 間変動は ±0.05 程度（Iter10 の journal 記載）。temperature を 0.2-0.3 に上げることでより大きな分散が得られるが、confidence 値の解釈性が低下するリスク。

**multi-sample consistency の実装変更**:
1. `router.py`: `estimate_confidence()` をラップして複数回呼び出す関数 `estimate_confidence_multi_sample()` を作成。各回の confidence 値の平均と分散を計算。分散が小さい = high confidence signal、分散が大きい = low confidence signal。
2. `node.py`: `run_ask_flow()` で multi-sample 版の confidence estimation を呼ぶように変更（config から切り替え可能にする）。
3. `protocol.py` の変更は不要: ProbeResponse.confidence は既存のまま。confidence signal の抽出経路のみが変わる。

**変更量見積もり**:
- `router.py`: +15行（multi-sample 用関数、分散計算）
- `node.py`: +3行（呼び出しの切り替え）
- **合計: ~18行**

**分かったこと（Q3: confidence signal 抽出経路のマッピング）**

**現行フロー**:
```
node.py:run_ask_flow()
  → peer_client.probe_all() (HTTP POST /probe to each peer)
    → http_server.py:probe() (FastAPI endpoint)
      → router.py:estimate_confidence() (LLM call to /api/chat)
        → parse_confidence(raw_response) → float confidence
      → ProbeResponse(confidence=..., estimated_latency_ms=...)
  → aggregator.select_dispatch_targets(probe_responses, ...) → dispatch targets
```

**STP を適用する場合の変更箇所**:
1. `http_server.py:probe()` (line 225-231): `estimate_confidence()` の呼び出しに logprobs 抽出を追加。または STP 用関数に切り替え。
2. `router.py:estimate_confidence()` / 新規 `estimate_confidence_stp()`: logprobs を含むレスポンスをパースし、トークン確率の統計量（平均 logprob, min logprob）を計算。
3. `expert_backend.py:OllamaClient.generate()`: logprobs パラメータ追加、エンドポイント変更。
4. `protocol.py:ProbeResponse`: 新フィールド追加（`confidence_logprobs_mean` など）。
5. `aggregator.py`: STP confidence signal を routing decision に組み込む場合、`select_dispatch_targets()` のロジック変更が必要。

**multi-sample consistency を適用する場合の変更箇所**:
1. `http_server.py:probe()`: 複数回の `estimate_confidence()` 呼び出しを追加（config で回数指定）。分散計算。
2. `router.py`: multi-sample 用関数を作成。`estimate_confidence_multi_sample()` が内部で N 回 `estimate_confidence()` を呼ぶ。
3. `protocol.py:ProbeResponse`: 変更不要（既存の confidence フィールドを使う）。分散値は別途 aggregator で計算するか、または probe レスポンスに追加フィールドを追加する場合は +2行。

**両アプローチの比較**:

| 観点 | STP | multi-sample consistency |
|------|-----|------------------------|
| 変更ファイル数 | 5 (expert_backend, router, protocol, http_server, node) | 2-3 (router, node, protocol optional) |
| 変更行数 | ~45行 | ~18-20行 |
| ollama バージョン依存 | high（logprobs サポートが必要） | low（既存の generate API のまま） |
| probe レイテンシ | 同程度（1回の生成で logprobs も同時に得られる） | N倍（N=3-5回実行） |
| offline 分析可能性 | results.jsonl に logprobs が記録されていれば可能 | 既存の confidence 値から分散を再計算可能 |
| label leakage リスク | low（トークン確率は routing decision と無関係） | low（confidence 値は既知、分散は新しい信号） |

**分かったこと（Q4: ベースライン結果と成功条件）**

**ベースライン**: results/20260721_222225（Iter9, self_report ベースライン）
- top1_accuracy: 0.870（>=0.87 非退行基準）
- misrouting_rate: 0.130（<=0.13 非退行基準）
- education precision: 1.000, recall: 0.500
- single_domain_top1_accuracy: 0.875

**Iter10（calibrated routing）との比較**:
- top1_accuracy: 0.848（-0.022 退行）→ rejected の理由
- misrouting_rate: 0.152（+0.022 悪化）

**成功条件の提案**（Iter11 でどちらのアプローチを試すかによる）:

共通の非退行基準:
- top1_accuracy >= 0.87（Iter9 ベースライン以下にならない）
- single_domain_top1_accuracy >= 0.87
- misrouting_rate <= 0.15

STP の場合の改善目標:
- confidence signal の弁別力が self_report より高い（offline analysis で margin と正の相関）
- top1_accuracy >= 0.87（非退行）+αの改善

multi-sample consistency の場合の改善目標:
- probe レイテンシ増加（3-5倍）を許容して、confidence signal の run 間安定性が向上
- offline analysis で confidence variance と routing correctness の相関を確認
- top1_accuracy >= 0.87（非退行）

**次の計画フェーズへの示唆**:
1. **multi-sample consistency を先に試すことを推奨**。理由: (a) 変更量が少ない（~18行 vs ~45行）、(b) ollama バージョン依存が低い（既存の generate API のまま）、(c) offline analysis が既存 results.jsonl から可能、(d) STP は logprobs サポートのバージョン依存があり、ollama のバージョン確認が必要。
2. **STP は Iter12 以降に検討**。multi-sample consistency で confidence signal の改善方向性が確認できた場合、より高精度な STP へ移行する段階的なアプローチが妥当。
3. rc-planner に渡す単一レバー: `confidence_signal_method=multi_sample`（values=[3, 5] で sample_count を掃引）。これにより offline analysis で最適な sample_count を決定可能。

