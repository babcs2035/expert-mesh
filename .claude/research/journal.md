## Iteration 78: 複合設問評価集合の拡充（既存公開データセットの調査を起点に n=300〜400 へ）

### 調査 (Iter78)

B116(2)・config.yml の `compound_eval_set_expansion` note が定める調達順位 (i) 既存公開データセット → (ii) LLM 生成（生成器を rank_2 の合成訓練データ生成器と分離）→ (iii) 人手作成，に従って調査した．問いは 3 つ．**(Q1) 「1 問で 2 ドメインを同時に要する日本語の相談文」に相当する公開データセットは存在するか（出典・発行元・ライセンス・本研究の 10 ドメイン定義との適合性）．(Q2) 存在しない場合，LLM 生成で循環的妥当性を避ける構成は何か．(Q3) 拡充後に必要な件数はいくつか（検出力の実測に基づく見積り）．**

**Q1: 既存公開データセットの調査 — 該当なし（(i) は不成立と判断する）**

tavily-search（`tvly search`，depth=advanced）で日本語・英語の両方から 7 クエリを投げ，加えて Hugging Face Hub を `hf datasets list --search` で 5 クエリ検索した．確認できた候補と，本研究の要件（**1 行が 2 つのドメインラベルを同時に持つ，日本語の自然文相談**）に対する適合性は次のとおり．

| 候補 | 発行元・出典 | ライセンス/入手性 | 10 ドメイン定義との適合性 |
|---|---|---|---|
| LegalRikai: Open Benchmark | LegalOn Technologies（2026 公開．<https://legalontech.jp/10193>，HF Hub にデータソース公開） | 公開（詳細条件は要確認） | **不適合**．企業法務タスク（契約書修正等）の単一ドメイン．相談文ではなく指示タスク |
| lawqa_jp（日本の法令に関する多肢選択式 QA） | デジタル庁 <https://github.com/digital-go-jp/lawqa_jp> | GitHub 公開（星 276） | **不適合**．legal 単独・4 択形式．JMMLU と同型で複合性なし |
| JMED-LLM（日本語医療 LLM 評価データセット） | 医療 NLP コミュニティ <https://speakerdeck.com/fta98/jmed-llm-...> | 公開 | **不適合**．medical 単独．「医療安全性を医療・法律・倫理の観点で評価」とあるのは**評価観点**であって設問のドメインラベルではない |
| Yahoo!知恵袋データセット | NII ×ヤフー（2007〜．<https://data.mdsc.hokudai.ac.jp/ne/dataset/mdsc19>） | **申込み・研究利用契約が必要**（オープンライセンスではない） | 自然文相談という点だけは合致するが，**ドメインラベルが付与されていない**．ラベル付けを自前で行うなら結局 (ii)/(iii) と同じ作業量．入手に契約手続きの待ちが入り単一イテレーションで完結しない |
| 国民生活センター 相談事例 / PIO-NET | 国民生活センター <https://www.kokusen.go.jp/category/jirei.html> | オープンデータは**窓口一覧 CSV 等に限られ**，相談事例本体は解説記事（HTML）．PIO-NET 生データは外部提供に制約あり | **不適合**．消費生活相談に偏り，10 ドメインへの写像が business_economics/legal へ集中する |
| RouterArena | Lu et al., arXiv:2510.00202（ICLR 2026）<https://arxiv.org/abs/2510.00202>，HF `RouteWorks/RouterArena` | 公開（CC BY 4.0 の論文，データセットは HF 公開） | **不適合**．8,400 クエリ・9 トップレベルドメイン・44 カテゴリだが **英語**，かつ **1 クエリ 1 ドメインの単一ラベル**（既存 23 データセットからのサンプリングで構成）．ルータ評価という目的は近いが複合設問ではない |
| RouterBench / RouterEval / LLMRouterBench | Hu et al. ICML 2024（<https://openreview.net/forum?id=IVXmV8Uxwh>），MilkThink-Lab/RouterEval 等 | 公開 | **不適合**．いずれも「どのモデルへ送るか」のモデル選択ベンチマークで，英語・単一ラベル・ドメイン横断設問ではない |

**結論**: 「日本語」「自然文の実務相談」「1 問に 2 つの専門ドメインラベル」の 3 条件を同時に満たす公開データセットは見つからなかった．司法試験過去問・医事法制の判例解説の類も，商用予備校サイトの有償教材か，判例データベース（出版社との契約が前提．JED2022 山田「日本語判決書を用いたデータセットの構築」<https://jedworkshop.github.io/jed2022/materials/jed2022_c-4_%E5%B1%B1%E7%94%B0.pdf> が「判例データベースはデータベースの著作物として保護され，使用には出版社との交渉・契約が前提」と明記）であり，本研究でそのまま再配布できるライセンスのものは確認できなかった．**したがって (i) `existing_public_dataset` は不成立とし，(ii) へ落とす．**（断定は避けるべき点として，調査は tavily と HF Hub 検索の範囲に限られる．「存在しない」ことの証明ではなく「この探索範囲では見つからなかった」ことの記録である．）

**Q2: LLM 生成における循環的妥当性の回避（(ii) の構成）**

本リポジトリには既に 2 ドメイン同時の日本語相談文を LLM 生成する実績がある（`scripts/generate_multidomain_training_examples.py`，Iter60〜65，rank_2 の多ラベル訓練信号用）．ただしこれは **`config.yaml` の `judge_model`（`schroneko/llama-3.1-swallow-8b-instruct-v0.1:q4_k_m`）を生成器に使っている**（同スクリプト docstring L10，`--model` の既定運用）．評価集合をこの生成器で作ると，(a) 訓練側の合成データと評価側が同一分布になる，(b) 軸②の LLM-as-judge（`scripts/evaluate_response_quality.py`，同じ judge_model）が自分の生成物を採点する，という二重の循環が生じる．config.yml の note が要求する「生成モデル・プロンプトの分離」はこの 2 点を断つためのものである．

多ラベルデータ拡張における生成方式の先行知見として，Iter60 の計画が引いた arXiv:2312.11276 は「2 つの単一ラベル文を連結する Concat 方式は意味的にも統語的にも一貫しないため LLM 生成に一貫して劣る」と報告している（同スクリプト docstring に記載）．本反復でも連結方式は採らず，1 文で 2 ドメインの知識を要する相談文を生成させる方式を踏襲する．

wafl-ctrl5 の Ollama に現在存在するモデルは `nomic-embed-text:latest` と `schroneko/llama-3.1-swallow-8b-instruct-v0.1:q4_k_m` の 2 つだけであることを実機確認した（`docker exec ... ollama list`）．したがって**別系統の生成モデルを 1 つ pull する必要がある**．

**Q3: 必要件数 — 現行 100 行の検出力を実測してから決める**

config.yml は「discordant 15〜19 行」を前提に n=300〜400 を目安としているが，**Iter75 本走と Iter77 本走（education 固有補正撤去という，全体 top1 を +2.0pt 動かした比較的大きなレバー）の複合 100 行を直接突き合わせたところ，`selected_domain` が変わった行はわずか 8 行だった**（全 1,600 行では 100 行）．つまり複合行の discordant 率は π_d ≈ 0.08 で，note の想定（0.15〜0.19）より低い．正誤の discordant はこれ以下になる．

McNemar の検出力（α=0.05 両側）を π_d と n の関数として計算すると次のとおり（`δ_MDE = (z_{0.975}+z_{0.80})·sqrt(π_d/n)`，有意判定に必要な純増減行数 `≈1.96·sqrt(n·π_d)`）．

| 複合行数 n | π_d=0.08 の期待 discordant | 有意に必要な純差（行 / pt） | π_d=0.17 の期待 discordant | 同（行 / pt） |
|---|---|---|---|---|
| 100（現状） | 8 | 5.5 行 / **5.5pt**（実際には 8 行中 8-0 でも p=0.008 と綱渡り） | 17 | 8.1 行 / **8.1pt** |
| 300 | 24 | 9.6 行 / 3.2pt | 51 | 14.0 行 / 4.7pt |
| **415（本計画）** | 33 | 11.3 行 / **2.7pt** | 71 | 16.5 行 / **4.0pt** |

80% 検出力ベースの MDE も n=100 の 11.6pt（π_d=0.17）/ 7.9pt（π_d=0.08）から，n=415 では 5.7pt / 3.9pt へ縮む．**Iter63〜68 が「判定不能」で潰れた ±3〜4 行（3〜4pt）の変化は，n=415・π_d=0.08 なら 11 行前後の純差が必要なので依然として厳しいが，n=100 では原理的に不可能だった領域（discordant 8 行では ±3 行は絶対に有意にならない）が，少なくとも「一方向に偏れば検出できる」領域に入る．** 過大な期待は置かず，到達点は「複合設問の 4pt 級の差を初めて統計的に語れるようになる」ことと定める．

### 計画 (Iter78)

**単一レバー**

`compound_eval_set_expansion` = **`llm_generated_separate_generator`**（Q1 により (i) は不成立．(iii) 人手作成は 315 件規模では現実的でなく，かつ B116(2) が最終手段と定めている）．評価集合の複合行を **100 行 → 415 行**（45 ドメインペア × 7 件 = 315 行を純追加）へ拡充する．データセット全体は 1,600 行 → **1,915 行**．変更するのは評価データのみで，分類器・埋め込み・`config.yaml`・ルーティングのコードは一切触らない．

**仮説（事前登録）**

「複合行を 415 行へ増やすと，複合サブセットの discordant 行数が現状 8 行（Iter75↔Iter77 の実測）から 30 行以上へ増え，複合設問に関する McNemar 検定の最小有意検出幅が 5.5pt から 3.0pt 前後へ縮む．一方，既存 1,600 行部分集合の指標は Iter77 本走と（dispatch 失敗行を除いて）一致する．」

**固定する構成（単一レバー原則）**

`config.yaml` 全項目（`embedding_model=nomic-embed-text`，`routing_method=supervised_classifier`，`confidence_threshold=0.0`，`dispatch_top_k=2`，`dispatch_gap_threshold=0.29`／`dispatch_gap_max_k=4`，`aggregation_method=max_confidence`，`judge_model`），`models/domain_classifier.joblib`（Iter77 で再生成した artifact をそのまま使う．再訓練しない），`data/classifier_train.jsonl`（1,427 件，無変更），`build_dataset.py` の JMMLU 単一ドメイン行 1,500 件（`_DOMAIN_TASK_MAP`・サンプリング seed とも無変更），既存 `_COMPOUND_QUESTIONS` 100 件（本文・順序・id をビット単位で保持），`classifier.py` / `http_server.py` / `aggregator.py` / `node.py` / `metrics.py`．

**変更するファイルと箇所**

1. `scripts/generate_compound_eval_questions.py`（**新規**）: 45 ペア（10 ドメインから 2 つ選ぶ全組合せ）ごとに日本語相談文を生成する．`scripts/generate_multidomain_training_examples.py` を **import しない・プロンプト文言を流用しない**（独立に書き下ろす）．生成中に `build_dataset._COMPOUND_QUESTIONS` を読まない（近重複監査のモードでのみ局所 import する．Iter60 の leak guard と同じ設計）．
2. `data/compound_questions_generated.jsonl`（**新規・コミット対象**）: 生成・フィルタ後に採択した 315 件（`{query, expected_domains, pair, generator_model, generated_at}`）．本走のたびに再生成しない（`mise run setup` が `build_dataset.py` を毎回呼ぶため，生成結果はファイルとして固定しないと評価集合が非決定になる）．
3. `build_dataset.py`: `_load_generated_compound_questions(path)` を追加し，`_build_rows()` の `_COMPOUND_QUESTIONS` ループ（L1147-1155）の**後**に追加行を append する．id は `compound-101` 以降の連番（既存 `compound-001`〜`compound-100` の id と本文は不変）．ファイルが無い場合は従来どおり 1,600 行を返す（テスト互換）．
4. `tests/test_build_dataset.py`: 「拡充後も既存 100 行の id・本文・`expected_domains` が完全一致する」「追加行がすべて `is_compound=True` かつ `len(expected_domains)==2`」「id の重複が無い」を固定する回帰テストを追加．

**レバーを読むコード行と到達条件（d0004 §4 の再発防止．同型の失敗 6 回の教訓）**

- 生成 → データ化: `data/compound_questions_generated.jsonl` を `build_dataset.py:_build_rows()` の新規ブロックが読む．到達条件は `mise.toml` L31 `uv run python build_dataset.py --output data/dataset.jsonl`（`mise run setup` 内）が実行されること．**`mise run setup` を省略すると dataset.jsonl が 1,600 行のまま本走してしまうため，setup を必ず実行し，`wc -l data/dataset.jsonl` = 1915 を確認する．**
- データ → 実験: `mise.toml` L161・L206 が `--dataset data/dataset.jsonl` を `run_experiment.py` へ渡す．dataset.jsonl は Docker イメージに同梱されるため **`mise run deploy` によるイメージ再配布が必須**（Iter22 のデプロイ漏れと同型の失敗を防ぐ．各ノードで `wc -l` を確認する）．
- 実験 → 指標: `metrics.py` L647-649 の `by_compound[len(r["expected_domains"]) > 1]` が複合行を分離し，`compound_domain_question_count` / `compound_domain_top1_accuracy` を出す．`compute_compound_coverage_metrics()`（L127-194）の `compound_rows` も同じ条件で拾う．**到達確認は `compound_domain_question_count == 415`．**
- 到達条件はすべて現行構成で満たされる（新しい config キーもスキーマ変更も導入しないため，「設定は変えたがコードに届かない」型の失敗は構造的に起こり得ない）．

**実験手順**

1. **生成（wafl-ctrl5 のみ．絶対条件 B）**: wafl-ctrl5 の Ollama へ judge_model と別系統のモデルを 1 つ pull する（第一候補 `qwen3.5:9b`．pull できない場合は `qwen3.5:14b-q4_K_M` → `gemma3:12b-it-q4_K_M` の順に代替．**必須条件は `schroneko/llama-3.1-swallow-8b-instruct-v0.1:q4_k_m` 以外であること**．RTX 3060 12GB に収まる q4 量子化を選ぶ）．`ssh -fNT -L 11499:localhost:11434 wafl-ctrl5` のトンネル経由で 45 ペア × 12 件 = 540 件を生成する．**wafl500〜509 は使わない．**
2. **フィルタ（決定的・自動）**: (a) 20〜200 字，(b) 4 択マーカー（`A.` `B.` 等）を含まない，(c) 生成文同士・既存 100 行・`data/classifier_train.jsonl`・`data/classifier_train_multidomain*.jsonl` に対する近重複除去（文字 3-gram Jaccard ≥ 0.6 を除外），(d) ドメイン名そのもの（「法律」「医療」等のラベル語の列挙）を含む機械的な文を除外．
3. **2 ドメイン性の独立検証**: 生成器とは別のモデル（`judge_model` = Swallow 8B）へ「この相談に回答するために必要な専門分野を 10 個の選択肢から 2 つ挙げよ」と問い，**意図した 2 ドメインを順不同で両方挙げた行だけを採択**する．生成器と検証器が別モデルであることが要件（循環回避）．採択率を journal に記録する．ペアごとに 7 件へ切り詰める（不足ペアは追加生成して補う．最終的に 45 ペア × 7 = 315 件）．
4. **人手スポットレビュー**: ランダム 30 件を実装フェーズが読み，「2 ドメインの知識が本当に両方要るか」を判定．不適合が 6 件（20%）を超えたらプロンプトを見直して再生成する（この閾値は事前登録）．
5. `build_dataset.py` を改修し，`uv run pytest tests/test_build_dataset.py` と `uv run ruff check` を通す．`mise run setup`（**直後に `uv sync --extra research` で research extra を復旧する．B118 の落とし穴 2**）→ `wc -l data/dataset.jsonl` = 1915 を確認．
6. **オフラインの検出力検証（wafl-ctrl5）**: 拡充後の 415 複合行について，Iter77 で退避した旧 artifact `models/domain_classifier_pre_iter77_edu_corrected.joblib` と現行 artifact の argmax を比較し，flip 行数を数える．現行 100 行での実測 8 行に対し，415 行での期待値は 33 行．**これが本レバーの主基準の実測値である．**
7. `mise run deploy` → 先頭 20 問の予備実行（`data/dataset_20.jsonl` 相当）でノード疎通を確認．
8. **wafl500〜509 で 1,915 問のフルスペック本走を 1 回（絶対条件 A）**．`mise run analyze -- <timestamp>` まで実施（引数なし実行は `results/iter45_preliminary/` を誤選択する．B118 の落とし穴 1）．
9. 指標を「全 1,915 行」「既存 1,600 行部分集合」「複合 415 行」「新規 315 行」の 4 通りで算出する（`metrics.py --results` に id フィルタを掛けたサブセットを渡す小スクリプトで足りる）．

**実験時間の見積りと `timeout_min`**

Iter77 本走は 1,600 問で約 24 分．問題数比 1.197 倍で約 29 分．過去の最遅実測（Iter27 の 96〜101 分）を基準にしても 115〜121 分で，現行の `experiment.timeout_min: 150` に約 19% の余裕が残る．**したがって config.yml の `timeout_min` は 150 のまま変更しない**（変更すると単一レバー原則の外側で config を触ることにもなる）．ただし本走のポーリングで 130 分を超えた場合は，次イテレーションで 180 への引き上げを backlog へ起票すること．

**成功条件（事前登録）— 主基準は検出力であり top1_accuracy ではない**

基準線は Iter77 本走 `results/20260926_171953/`（top1=0.615625，single_domain_top1=0.629333，compound_domain_top1=0.41，kappa=0.588209，ECE=0.076640，misrouting=0.384375，fallback=0.0，dispatch_failure=0.00125）．

| 区分 | 指標 | 現状 | 合格条件 |
|---|---|---|---|
| **主基準①（規模）** | 複合行数 / 全行数 | 100 / 1,600 | **415 / 1,915**．`metrics.py` の `compound_domain_question_count == 415` |
| **主基準②（検出力）** | 旧/新 artifact replay による複合行の argmax flip 行数 | 8 | **≥ 30 行**（415 行への比例期待値 33 の 9 割）．同時に実測 π̂_d を報告し，McNemar の最小有意検出幅 `1.96·sqrt(n·π̂_d)/n` が **≤ 3.5pt**（現状 5.5pt）になること |
| **主基準③（純追加）** | 既存 100 行の同一性 | — | id `compound-001`〜`compound-100` の `query`・`expected_domains` が拡充前 dataset.jsonl と**完全一致**（diff 0 バイト）．単一ドメイン 1,500 行も同様に完全一致 |
| **主基準④（非循環）** | 生成器と judge/rank_2 生成器の分離 | — | 生成モデル ≠ `schroneko/llama-3.1-swallow-8b-instruct-v0.1:q4_k_m`．生成スクリプトが `generate_multidomain_training_examples` を import しない．近重複監査で既存 100 行・訓練データとの 3-gram Jaccard ≥ 0.6 の行が **0 件** |
| **主基準⑤（データ品質）** | 独立検証器の一致率／人手スポットレビュー | — | 採択行は検証器一致 100%（フィルタ条件）．人手 30 件のうち不適合 **≤ 6 件（20%）** |
| **非退行①** | 1,600 行サブセットの top1_accuracy | 0.615625 | **0.615625 と一致**（ルーティングは決定論的．許容差は dispatch 失敗行に起因する ±0.2pt 以内．ズレたら「拡充の効果」ではなく実行環境の異常を疑う） |
| **非退行②** | 1,600 行サブセットの single_domain_top1 / 旧 compound_top1 | 0.629333 / 0.41 | 同上（±0.2pt 以内） |
| **非退行③** | ドメインごとの選択分布 | — | 新規 315 行を除いた per-domain recall/precision 20 指標が Iter77 と一致 |
| 報告のみ | 全 1,915 行の top1_accuracy・複合 415 行の compound_top1・compound_domain_set_recall | 0.615625 / 0.41 / — | **判定に用いない**（複合行の比率が 6.25%→21.7% へ上がるため全体 top1 は機械的に下がる見込み．混同しないこと） |
| 報告のみ | 新規 315 行と既存 100 行の compound_top1 の差 | — | 生成由来の系統差の有無を見る参考値．差が 15pt を超える場合は生成分布の偏りとして backlog へ起票 |

- **adopted の定義（本レバー固有）**: 本レバーは精度改善ではなく測定系の整備であるため，**主基準①〜⑤と非退行①〜③をすべて満たせば adopted** とする．top1_accuracy の増減は判定に用いない．
- **invalid（実験不成立）の判別**: 本走の results.jsonl が 1,600 行のまま，または `compound_domain_question_count` が 100 のままなら，「効果なし」ではなく **`mise run setup`/`deploy` の漏れ**を既定の解釈とする（d0004 §4）．
- **partial の扱い**: 主基準②（flip ≥ 30 行）だけが未達の場合は，「拡充は完了したが検出力は想定より低い」として `partial` とし，実測 π̂_d から必要 n を再計算して backlog へ残す（さらなる拡充か，複合設問の評価指標自体の見直しかを次反復で判断する）．

**期待効果**

Iter63〜68 の 6 反復と Iter76 の conformal risk control 見送りは，いずれも「複合 100 行では判定できない」ことが理由だった．複合行を 415 行にすることで，以降の複合ドメイン系レバー（`embedding_model_replacement`，`cross_domain_training_data_augmentation`，`dispatch_policy` 系の再訪）が初めて 3〜4pt 級の効果を統計的に語れる土俵に乗る．副次的に，複合行が全体の 21.7% を占めることで，全体 top1 が単一ドメイン行にほぼ支配されていた現状（93.75%）も緩和される．

### 実装・実験 (Iter78)

**実装（差分の要点）**

1. `scripts/generate_compound_eval_questions.py`（新規）: 45 ドメインペア分の日本語相談文を生成する独立スクリプト．`generate_multidomain_training_examples` は import せず，プロンプト文言・ドメイン説明文（`_DOMAIN_PROMPT_HINTS_JA`）を独立に書き下ろした．生成器 `--generator-model` と検証器 `--verifier-model` が同一なら `SystemExit`（循環回避のハード guard）．フィルタは (a) 20〜200 字，(b) 4 択マーカー無し，(c) ドメインラベル語（`_DOMAIN_LABEL_WORDS_JA`）の直接使用無し，(d) 既存 100 行・`classifier_train*.jsonl`・`classifier_train_multidomain*.jsonl` に対する文字 3-gram Jaccard ≥ 0.6 の近重複除外．独立検証は Swallow 8B（`judge_model`）へ「10 分野から 2 つ選べ」と問い，意図した 2 ドメインが順不同で完全一致した行だけを採択する．`build_dataset._COMPOUND_QUESTIONS` の読み込みは `_load_existing_compound_texts()`（近重複監査専用，local import）のみで，生成そのものには一切影響しない（Iter60 の leak guard と同型）．
2. `data/compound_questions_generated.jsonl`（新規・コミット対象，315 行）: 生成・独立検証・近重複フィルタ後に採択した行．45 ペア × 7 件で固定．
3. `build_dataset.py`: `_load_generated_compound_questions(path)` を追加し，`_build_rows()` に `generated_compound_questions_path: str | None = None` 引数を新設．`_COMPOUND_QUESTIONS` ループの**後**に，`compound-{101,...,415}` として追加行を append する．デフォルト `None`（未指定なら追加無し，既存呼び出し元・既存テストは無変更）．CLI (`main()`) にのみ `--generated-compound-questions`（既定値 `data/compound_questions_generated.jsonl`）を新設し，`_build_rows()` へ明示的に渡す．これにより `mise run setup` が呼ぶ実プロダクションパスだけが拡張分を読み，`tests/test_build_dataset.py` の既存呼び出し（`_build_rows()` / `write_dataset()` を直接叩く 9 件のフィクスチャテスト）は不変のまま保たれる（単一レバー原則．実行時の到達条件はこの `main()` の 1 引数のみ）．
4. `tests/test_build_dataset.py`: 5 件追加（`_load_generated_compound_questions` の None/欠損/正常系，`_build_rows` が `generated_compound_questions_path` 未指定なら既存 100 行のみを返すこと，指定時に既存 100 行がバイト一致のまま `compound-101` 以降が追記されること，id 重複が無いこと）．**申し送り**: `tests/fixtures/jmmlu_sample.zip` は Iter36/37 の `education → japanese_civics` 切替に追随しておらず（fixture に `japanese_civics.csv` が無い），本反復と無関係に 9 件が既存失敗している（`git stash -u` で退避しても同数再現，Iter78 変更前から存在）．新規 5 件は `_FIXTURE_DOMAIN_TASK_MAP` の `education` をこの fixture に実在する `sociology` へ差し替えたローカル変数 `_EDUCATION_TASK_FIXED_DOMAIN_TASK_MAP` を使うことでこの pre-existing 破損から独立させた．

検証: 変更 3 ファイルの `uv run ruff check` は PASS．`uv run pytest tests/test_build_dataset.py` は新規 5 件を含む 13 件 PASS，pre-existing 9 件 FAIL（上記と同一原因，本反復による新規失敗 0 件）．

**絶対条件 B（wafl-ctrl5 限定）の遵守**: wafl-ctrl5（192.168.15.10）の Ollama へ `qwen3.5:9b`（6.6GB, q4量子化，RTX 3060 12GB 制約内）を生成器として新規 pull（`judge_model` の Swallow 8B とは別系統）．`ssh -fNT -L 11499:localhost:11434 wafl-ctrl5` のトンネル経由で生成・検証を実行．wafl500〜509 は生成に一切使用していない．

**生成の実行記録**: 第一パスは 45 ペア × 12 件生成 → 検証 → 各ペア 7 件へ切り詰めで 270/315 行（16 ペアが目標未達，生成試行 433 件）．不足 45 行を対象ペアだけに絞った追い上げパス（同じ `generate_all_rows()` を `domains=[d1,d2]` の 2 要素リストで呼ぶだけで対応，コード変更無し，近重複参照に第一パスの採択済みクエリを追加）で全 16 ペアとも過不足なく充足し，315/315 行を確定．生成モデルが `qwen3.5:9b`（thinking モデルで 1 呼出し約 10〜15 秒）かつ Ollama の keep_alive=-1 でも生成器と検証器の 2 モデル（6.6GB+4.9GB＝約11.5GB／12GB）がほぼ VRAM を使い切るため呼出しごとに再ロードが発生し，生成全体（2 パス合計）で実時間**約 78 分**を要した．

**品質確認**:
- 近重複監査（既存 100 行 + `classifier_train.jsonl` + `classifier_train_multidomain*.jsonl`，計 3,030 件参照）: 315 行すべてで 3-gram Jaccard の最大値 = **0.2288**（≥0.6 の該当 0 件）．生成 315 行どうしの pairwise 近重複も 0 件．
- 人手スポットレビュー（乱数シード 78 で 30 件抽出，実装フェーズが判定）: 「2 ドメインの知識が本当に両方必要か」という基準で弱い/不適合と判定したのは **5/30（16.7%）**，事前登録の閾値 ≤6/30（20%）以内．弱かった行はいずれも `general` ドメインを含むペア（`business_economics+general` の医療危機混入，`general+mathematics` の「数学ノートを捨てるか」など計算不要な設問，`general+history_culture` の無関係な 2 話混在）で，`general` が抽象的すぎるために生成が単一ドメインへ寄るか無関係な 2 話を継ぎ足す傾向がある，という所見を得た（backlog 起票要，本文末尾参照）．

**setup/deploy 手順**: `mise run setup`（イメージビルド・push 含む）実行後に `wc -l data/dataset.jsonl` = **1915** を確認．直後に `uv sync --extra research` で research extra を復旧（B118 落とし穴 2 の再発防止）．独立検証: git HEAD 版 `build_dataset.py`（`git show HEAD:` で復元）の `_build_rows()` を `generated_compound_questions_path` 無しで実行した 1,600 行と，新版 dataset.jsonl の対応 id（`{domain}-NNN`，`compound-001`〜`100`）を突き合わせ，**query・expected_domains ともに mismatch 0 件**（完全一致）を確認．`mise run deploy` でイメージ再配布（10 ノード）→ 全ノード healthy，smoke_check（git-status/hashes/probe）すべて PASS．各ノードで `docker compose exec app wc -l /app/data/dataset.jsonl` = **1915**（10/10 ノード一致）を確認．

**レバー発火の証拠**: (1) `data/dataset.jsonl` の `compound_domain_question_count` 相当の実カウント = 415（1500 単一 + 415 複合）．(2) 先頭 20 問相当の予備実行（`data/dataset_preview_iter78.jsonl` = 単一 15 問 + `compound-101`〜`105` の新規複合 5 問，イメージを当該ファイル込みで再ビルド・push 後に実行）で `compound-101`〜`105` が実際に評価対象としてルーティングされることを直接確認（`compound-101 -> medical`，`compound-102`〜`105 -> business_economics`）．(3) 本走のログでも `compound-101`〜`415` が問題なく処理され，`completed 1915 questions` で正常終了．

**本走（絶対条件 A，wafl500〜509，1 回）**: `results/20260926_195929/`．`mise run start -- --dataset data/dataset.jsonl --output results.jsonl` を検出後即座に `state.json` を `status=waiting_experiment`，`experiment_dir=results/20260926_195929`，`experiment_deadline=1790429956`（開始時刻 1790420356 + timeout_min 150×60 + マージン 600 秒）に設定して起動．所要時間は実測**約 51 分**（19:59:29 起動 → 20:50 ごろ完了，ログの `completed 1915 questions` 時刻から逆算）．**単一ドメイン区間（739 問）は約 9 分（≈82 問/分）に対し，複合区間（415 問）は約 42 分（≈9.9 問/分，8 倍以上遅い）**．これは今回固定した `dispatch_top_k=2`／`dispatch_gap_threshold=0.29`／`dispatch_gap_max_k=4`（複合行は上位候補の confidence 差が小さくなりやすく，gap policy が 3〜4 ノードへの追加ディスパッチをより頻繁に発火させるため）が原因と推測される（`compound_mean_dispatched_count`＝2.506，metrics.py 出力参照）．推測の域を出ないが，複合行の存在そのものが実行時の分岐に測定可能な影響を与えている点は，主基準①（規模）とは独立の追加的な傍証として記録する．`mise run analyze -- 20260926_195929`（timestamp 明示指定，B118 落とし穴 1 の再発防止）まで実施．ログ全 10 ノードを grep した範囲で言語崩れ・OOM は無し．`dispatch_model_not_ready`（HTTP 500，一時的なノード過負荷）が 3 ノードで散発したが，いずれもリトライで解決し，最終的な `dispatch_failure_rate` は 1/1915（`medical-109`，probe 自体は正常でも実際の dispatch 呼出し 1 件が失敗）のみ．fallback は 0 件．

**取得した機械可読メトリクス（4 通り，`metrics.py` の既存実装をそのまま使用）**

| 区分 | n | top1_accuracy | single_domain_top1 | compound_domain_top1 | compound_domain_set_recall |
|---|---|---|---|---|---|
| 全 1,915 行 | 1915 | 0.589556 (Wilson 95%CI [0.567366, 0.611387]) | 0.628667 | 0.448193 | 0.393976 |
| 既存 1,600 行部分集合 | 1600 | 0.615000 | 0.628667 | 0.410000 | 0.420000 |
| 複合 415 行 | 415 | 0.448193 | — | 0.448193 | 0.393976 |
| 新規 315 行のみ | 315 | 0.460317 | — | 0.460317 | 0.385714 |

全 1,915 行の付随指標: kappa=0.587438，misrouting_rate=0.410444，fallback_rate=0.0，dispatch_failure_rate=0.000522，ECE=0.050552（n=1914），Brier=0.206919，AUROC=0.731903，answer_quality_accuracy=0.564（graded n=1500），end_to_end_accuracy=0.287728，mean_duration_ms=2424.68（`mean_dispatch_gen_time_ms`=1997.22 が大半を占める）．

**非退行の統計検証（`metrics.py` 既存関数 `compute_mcnemar_test` / `compute_domain_recall_mcnemar_test` / `compute_domain_precision_fisher_test` / `apply_benjamini_hochberg` をそのまま使用，Iter77 基準線 `results/20260926_171953/` と id ペアリング，id 集合完全一致を確認済み）**:

- 全体 top1（既存 1,600 行部分集合 vs Iter77 基準線）: discordant 2 対 1（計 3 件），chi2=0.0，**p=1.0**．基準線 0.615625 → 本走 0.615000（-0.0625pt，noise 内）．
- per-domain recall/precision 計 20 指標（10 ドメイン × 2）: 生の discordant はいずれも 0〜2 件，p 値は 0.4795（`natural_science_recall`）を除き全て 1.0．**BH 補正（q=0.05）後の有意退行 0/20**．

**主基準②（検出力）の実測 — 旧/新 artifact の argmax replay（wafl-ctrl5，`scripts/evaluate_classifier_calibration.py` の既存実装をそのまま使用，`--set-construction` 等の conformal オプションは未指定＝素の `predict_proba` argmax）**: `models/domain_classifier_pre_iter77_edu_corrected.joblib`（旧）と `models/domain_classifier.joblib`（現行）をそれぞれ拡充後の 1,915 行データセット全体で再埋め込み・再推論し（embedding は wafl-ctrl5 の `nomic-embed-text` 経由，絶対条件 B 準拠），`selected_domain` の argmax を比較した．

| 区分 | n | flip 行数 | π̂_d（flip率） |
|---|---|---|---|
| 複合 415 行全体 | 415 | **19** | 0.045783 |
| 既存 100 行部分集合 | 100 | 6 | 0.06 |
| 新規 315 行部分集合 | 315 | 13 | 0.041270 |

McNemar 最小有意検出幅（事前登録式 `1.96·sqrt(n·π̂_d)/n`，n=415, π̂_d=0.045783 を代入）= **2.0587pt**．

**申し送り（本反復スコープ外の所見，backlog 起票用メモ）**:
1. 既存 100 行部分集合の argmax replay flip 数は本手法（`evaluate_classifier_calibration.py` の素の `predict_proba` argmax，dispatch_top_k・aggregation を経由しない）で 6 行だったのに対し，調査 (Iter78) 節で引用した「Iter75↔Iter77 実測 8 行」は本番 `results.jsonl` の `selected_domain`（Iter77 時点の `dispatch_top_k=1` を経由した実際のルーティング結果）同士の比較であり，測定対象（分類器の argmax 単体 vs パイプライン全体の選択結果）が異なる．今回の 19/415・6/100・13/315 はすべて前者（分類器 argmax 単体）の定義に統一して計測した．
2. `tests/fixtures/jmmlu_sample.zip` と `_FIXTURE_DOMAIN_TASK_MAP["education"]`（`japanese_civics`）の不一致による 9 件の既存テスト失敗は，Iter36/37 の本番切替（`build_dataset.py` の `education` proxy task 変更）にテストフィクスチャが追随していない，本反復と無関係の pre-existing バグ．修正は本反復のスコープ外（単一レバー原則）だが，次回のテスト保守タスクとして残す．
3. 人手レビューで判明した「`general` ドメインを含む複合ペアは弱い/無関係な組合せになりやすい」という所見（5/30 の不適合のうち該当 3 件が `general` 絡み）は，`general` の定義が「特定分野に限定されない一般的な話題」で最も抽象的なことに起因すると考えられる．次回複合設問を追加拡充する際は `general` ペアのプロンプトに具体例を追加する等の改善余地がある．
4. 本走の所要時間は約 51 分（複合行区間が単一行区間の 8 倍以上遅い）であり，`experiment.timeout_min: 150` に対しては十分な余裕（実測は上限の約 34%）があるため，130 分超過時の 180 分への引き上げ提案（config.yml 既存の申し送り）は今回発動しなかった．

### Iteration 78 実行済み

**単一レバー**: `compound_eval_set_expansion` = `llm_generated_separate_generator`．複合評価行を 100 → 415 行（45 ドメインペア × 7 件 = 315 行の純追加），評価集合全体を 1,600 → 1,915 行へ拡充した．分類器・埋め込み・ルーティング実装は一切変更していない（測定系のみの変更）．

**判定: partial**（事前登録の partial 規定「主基準②（flip ≥ 30 行）だけが未達の場合は partial」にそのまま該当する）．**adopted ではない**が，拡充した評価集合そのものは Iter79 以降の標準基準線として常設採用する．

| 事前登録基準 | 結果 | 可否 |
|---|---|---|
| ①規模 415/1,915 | 415 / 1,915 | ✅ |
| ②検出力 flip ≥30 行 | **19 行**（π̂_d=0.045783） | ❌ |
| ②検出力 最小有意検出幅 ≤3.5pt | **2.0587pt** | ✅ |
| ③純追加（既存 1,600 行の完全一致） | query・expected_domains とも mismatch 0 件 | ✅ |
| ④非循環（生成器 ≠ judge_model，近重複 0 件） | 生成器 `qwen3.5:9b`／検証器 Swallow 8B，3-gram Jaccard 最大 0.2288 | ✅ |
| ⑤品質（人手不適合 ≤6/30） | 5/30（16.7%） | ✅ |
| 非退行①②③ | 1,600 行サブセット top1=0.615000（基準線 0.615625，-0.0625pt）．McNemar discordant 2:1・p=1.0．per-domain 20 指標の BH 補正後有意退行 0/20 | ✅ |

非退行①の -0.0625pt は「ちょうど 1 行分」であり，本走で唯一の dispatch 失敗行 `medical-109`（`dispatch_failure_rate`=1/1915）で過不足なく説明がつく．ルーティングが決定論的であるという前提は崩れていない．

#### 主基準②の 2 条件が食い違った件 — 事前登録の設計上の誤りを認める

flip 行数（19 < 30，未達）と最小有意検出幅（2.0587pt ≤ 3.5pt，達成）は，**同じ「検出力」を測る 2 つの代理指標のつもりで登録したが，実際には互いに逆方向へ動く量だった**．検出幅は `MDE = 1.96·sqrt(π̂_d/n)` であり，n を固定すると π̂_d が小さいほど機械的に小さくなる．つまり「discordant が少ないほど検出幅が良く見える」という自己矛盾を，事前登録の時点で見落としていた．**pt スケールの検出幅は検出力の妥当な代理指標ではない．** 理由は，π̂_d が小さいとき，検出可能な効果の上限（discordant が全て一方向に倒れた場合の差＝π̂_d そのもの）も同時に縮むためである．今回の実測では検出可能な窓は `[2.06pt, 4.58pt]` しかなく，事前想定（π_d=0.08）の `[2.72pt, 8.0pt]` より明確に狭い．検出幅が「改善」して見えたのは，窓の下端だけを見て上端の縮小を見なかったからである．

検出力を素直に表す量は，**discordant 行数 n_d と，有意判定に必要な一方向偏り率 `1.96/sqrt(n_d)`** である．

| 条件 | n_d | 必要な偏り率 | 80% 検出力に必要な偏り |
|---|---|---|---|
| 拡充前（既存 100 行，argmax 定義） | 6 | 0.800 | 6-0 の全会一致でようやく p=0.031．実質検出不能 |
| **拡充後（415 行，実測）** | **19** | **0.450** | 約 80/20 の分割 |
| 事前登録の目標（415 行，π_d=0.08 想定） | 33 | 0.341 | 約 74/26 の分割 |

したがって正しい読み方は「flip 行数の条件こそが妥当な代理指標であり，それが未達である」．n_d は 6 → 19 と 3.17 倍に増え，必要偏り率は 0.80 → 0.45 へ下がったので**測定系は実質的に前進した**が，事前登録が目標とした水準には届いていない．**事後に検出幅の側だけを採って adopted とするのは，登録した基準を結果に合わせて選び直す行為なので採らない．** 加えて，仮に目標の 33 行に達していても 80% 検出力には 74/26 の偏りが要る．「n=415 なら 3〜4pt 級を統計的に語れる」という計画フェーズの見通し自体が，pt スケールの検出幅に引きずられた楽観だったと訂正する．

#### π̂_d が想定 0.08 に対し 0.045783 と低かった理由

- **新規 315 行が易しいから，ではない**．新規 315 行の flip 率 0.041270（13/315）と既存 100 行の 0.06（6/100）の差は有意でない（2 標本比率検定 z=0.78, p=0.44）．education を含む複合行の比率も既存 20/100・新規 63/315 と**ともに 20%** で，構成比の偏りも説明にならない．したがって「生成設問はルーティングが安定していて易しい」とは，このデータからは言えない．
- **真因は π̂_d の定義にある**．π̂_d は評価集合の固有の性質ではなく，**比較する artifact ペアに依存する量**である．今回 π̂_d を測るのに使った Iter77 の artifact ペア（education 固有補正の撤去）は 10 ドメイン中 1 ドメインの決定境界しか動かさない狭いレバーであり，これで測った 0.0458 は**下限側の推定値**である．事前想定 0.08 も，同じ狭いレバー（Iter75↔Iter77）のパイプライン出力から採った値であり，母数が 100 行と小さく（8/100，Wilson 95%CI 約 [0.041, 0.150]）今回の 0.0458 はその CI にほぼ収まっている．つまり 0.08 → 0.0458 の低下自体が，そもそもノイズ幅の範囲内である．
- 実務的な含意: `embedding_model_replacement` のようにベクトル空間ごと入れ替えるレバーでは flip 率が桁違いに大きくなることが期待できる（Iter39〜43 では fine-tuning の argmax flip rate が「過大」で単一レバー原則と両立しなかったと記録がある）．**「検出力が足りないから評価集合をさらに拡げる」ではなく，「次レバーの artifact ペアで π̂_d を本走前にオフライン実測し，n_d < 30 ならそこで打ち手を考え直す」** が正しい順序である（Iter79 の計画へ申し送り）．
- 力任せの拡充は割に合わない．π̂_d=0.0458 のまま n_d=30 を得るには複合行 655 行，n_d=100 には 2,184 行が要る．後者は下記の実行時間見積りで約 240 分となり `timeout_min: 150` を超える．

#### 複合行 top1 = 0.448193 の解釈

事前登録どおり全 1,915 行の top1（0.589556）は判定に用いない（複合行比率が 6.25% → 21.7% に上がれば機械的に下がる）．参考値として，新規 315 行の compound_top1=0.460317 は既存 100 行の 0.410000 を 5.03pt 上回るが有意ではなく（z=0.89, p=0.37），事前登録の「差が 15pt を超えたら生成分布の偏りとして起票」にも該当しない．生成設問が既存の手作り設問と系統的に異なる難易度である証拠は得られなかった．なお既存 100 行の compound_top1 が基準線と小数点まで一致（0.41）したことは，純追加が既存行を汚していないことの追加的な裏付けである．

#### 想定外事象: 複合行区間が単一行区間より約 8 倍遅い

単一 739 問区間 ≈82 問/分に対し複合 415 問区間 ≈9.9 問/分．`compound_mean_dispatched_count`=2.506 であり，複合行は上位候補の confidence 差が小さく `dispatch_gap_threshold=0.29`／`dispatch_gap_max_k=4` の gap policy が 3〜4 ノードへの追加 dispatch を頻繁に発火させることが原因と推測される（推測の域は出ない）．**今後の実行時間は概ね `18 分 + n_compound/9.9 分` で見積もれる**：n_compound=415 で 60 分（実測 51 分），655 で約 84 分，1,000 で約 119 分（`timeout_min: 150` の 8 割），2,184 で約 240 分（超過）．**複合行の拡充は 1,000 行あたりが現行タイムアウトでの実用上限**であり，それ以上は `timeout_min` の引き上げか dispatch 並列度の見直しとセットでしか成立しない．言語崩れ・OOM・fallback は 0 件．

#### 学び

1. **検出力の事前登録は，pt スケールの MDE ではなく discordant 行数 n_d と必要偏り率 `1.96/sqrt(n_d)` で書くこと．** pt スケールの MDE は π̂_d が下がると自動的に良く見えるうえ，検出可能な効果の上限が同時に縮むことを隠す．今回はこの隠れた自己矛盾を，2 条件を並べて登録したおかげで検出できた（結果的に，複数の代理指標を併記する登録法には意味があった）．
2. **π̂_d はレバー依存の量であり，評価集合の属性ではない．** 狭いレバー（1 ドメインのみを動かす）で測った π̂_d を，広いレバーの検出力設計に流用してはいけない．評価集合サイズ n の設計は「どのレバーを検出したいか」とセットでのみ意味を持つ．
3. 測定系の整備を目的とする反復では，非退行条件（既存部分集合のバイト一致・指標一致）が最重要の防衛線になる．今回それが 1 行のずれまで dispatch 失敗で説明しきれたことで，「拡充の効果」と「実行環境の異常」を混同せずに済んだ．
4. 実行時間は問題数に比例しない．複合行は gap policy 経由で単一行の 8 倍の実時間を食うため，評価集合の設計時には行数ではなく「行の種別ごとの単価」で見積もる必要がある．

#### 次の一手

複合評価集合の拡充（research_frontier 最上位）は規模・純追加・非循環・品質の 4 条件を満たして完了とし，B115(3) の優先順位に従い次レバーは **`embedding_model_replacement` = `qwen3_embedding_0.6b`**（B116(3) によりユーザー確認不要で自律着手可）．次イテレーション名は「埋め込みモデルの差し替え（nomic-embed-text → qwen3-embedding:0.6b）」．基準線は本走 `results/20260926_195929/`（1,600 行サブセット top1=0.615000／全 1,915 行 top1=0.589556）とし，両方を毎回併記する（B119 要レビュー (c) への回答）．詳細と申し送りは backlog B120〜B123．

---

## Iteration 77: education 固有補正の撤去と 10 ドメイン均一化

### 調査 (Iter77)

backlog B116（2026-09-23，user-decided）が定めた優先順位に従い，本反復は `education_specific_correction_removal` を実施する．調査では「撤去の是非」ではなく（B116 により再考の余地なしと確定済み），**(Q1) 撤去対象が実際にコードのどこに何個あるか，(Q2) 単一クラスだけの decision boundary 補正を外すことが理論的に何を意味するか，(Q3) 撤去後の着地点はどこか**の 3 点を明らかにした．

**Q1: 撤去対象の棚卸し（config.yml の記述は古く，実態と食い違っていた）**

config.yml の `education_specific_correction_removal` note は backlog B88 を引いて「`education_threshold=0.05` は実行時経路 `classifier.py:estimate_confidence_classifier()` に未反映であり，実行時への影響は現状ゼロ」と書いているが，**これは Iter57（`production_deployment_gap` の実施）以前の記述であり，現 HEAD では誤りである**．実機コードを直接確認した結果，education 固有補正は次の 4 箇所に存在する．

1. `scripts/train_domain_classifier.py:194-204` — `intercept_delta = 0.7` を `calibrated_model.calibrated_classifiers_` の各 fold の `estimator.intercept_[edu_idx]` へ加算（Iter44/45 adopted，訓練時）．
2. `classifier.py:56,77-78` — `EDUCATION_THRESHOLD = 0.05`．`estimate_confidence_classifier()` が education ノードのときだけ生確率に加算して返す（Iter57 で実行時経路へ配線済み．**実行時に効いている**）．
3. `scripts/evaluate_classifier_calibration.py:280,484-489,611-618,660-667` — `--education-threshold`（既定 0.0）．conformal 系列 Iter75 が `0.05` を明示指定して実行時分布を再現していた．
4. 同 `--education-logit-bias`（既定 0.0，Iter49/50 で rejected 済みの死にパラメータ）．

2 が実行時に効いている証拠は基準線の実測に残っている: `results/20260923_150540/results.jsonl`（Iter75 本走）の education ノードの confidence は 1,600 行すべてで 0.05 以上（**最小 0.050006**，最大 0.996516）であり，`p + 0.05` の下限がそのまま観測されている．

**Q2: 単一クラスだけの intercept シフトの位置づけ（先行研究）**

クラス事前分布に応じて logit を平行移動する手法の標準は Menon, Jayasumana, Rawat, Jain, Veit, Kumar, "Long-tail learning via logit adjustment", ICLR 2021（arXiv:2007.07314，<https://research.google/pubs/long-tail-learning-via-logit-adjustment/>）である．同論文の post-hoc logit adjustment は **全クラスに対し** `logit_y - τ·log π_y` を適用するもので，balanced error に対して Bayes 一貫性を持つ．本リポジトリの `intercept_delta=+0.7`（education のみ）はその単一クラス版に相当するが，(a) 補正量が事前分布から導かれておらず経験的に掃引された値であること，(b) 他 9 クラスに対応する項が無いため balanced error にも plain accuracy にも一貫性を持たないこと，の 2 点で理論的裏付けを欠く．そもそも本リポジトリの訓練データは `_extract_sample_weights()` により 10 ドメインの実効重みが完全に等しく揃えられている（`train_domain_classifier.py` docstring）ため，原理的な logit adjustment 量は全クラスでほぼ 0 である．**すなわち撤去は「補正を外して裸に戻す」というより「事前分布が均一な設定における Bayes 最適な決定則へ戻す」操作と解釈できる**．

一般論として，単一クラスの閾値・intercept を下げることはそのクラスの recall を上げ precision を下げる（accuracy と balanced accuracy で最適閾値が異なる，という多クラス評価指標の教科書的事実．例: MDPI "Selecting and Interpreting Multiclass Loss and Accuracy Assessment Metrics for Classifications with Class Imbalance", <https://www.mdpi.com/2072-4292/13/13/2591>）．実測もこれと整合しており，基準線の education は **recall 0.5353 に対し precision 0.3745** と 10 ドメイン中もっとも precision が低い．撤去でこの偏りが解消される方向へ動くと予想する．

**Q3: 事前シミュレーション（`+0.05` の部分は基準線ファイルだけで閉じる）**

`results/20260923_150540/results.jsonl` の `probe_candidates` は 10 ドメイン全ノードの confidence を保持しているため，**`EDUCATION_THRESHOLD` 撤去の効果は再計算なしで厳密に求まる**（education の値から 0.05 を引いて argmax を取り直すだけ）．

| 量 | 現行（+0.05 あり） | +0.05 撤去後 | 差 |
|---|---|---|---|
| top1（`probe_candidates` の argmax で再計算．`metrics.json` の 0.595625 とは dispatch 失敗行等で 3 行ぶんずれる） | 0.597500 | **0.603125** | +0.5625pt |
| argmax が変わる行数 | — | **38 / 1,600** | 2.375% |
| education が選ばれる行数 | 243 | **205** | -38 |
| education の単一ドメイン recall（n=150） | 0.580000 (87/150) | **0.553333** (83/150) | -2.67pt |
| education confidence の最小値 | 0.050006 | **0.000006** | 下限が外れる |

一方 `intercept_delta=+0.7` の撤去は temperature 較正の内側（fold ごとの logit）に効くため確率から逆算できず，**修正後の joblib で `evaluate_classifier_calibration.py` を回す必要がある**．Iter44 の採用時実績（education_recall +0.0647，argmax flip rate 8.62%）を逆向きに当てると，education_recall -6〜-7pt・flip 約 8〜9%（130〜140 行）が見込み値となる．両方を合わせた着地点の予想は下表（成功条件節）に事前登録する．

**撤去後の artifact 生成に関する技術的所見（実装フェーズへの申し送り）**

`intercept_delta` は `CalibratedClassifierCV.fit()` の**後**に加算されている（`train_domain_classifier.py:192` → `:203-204`）．したがって撤去後のモデルは，現行 artifact `models/domain_classifier.joblib` の各 fold の `estimator.intercept_[edu_idx]` から 0.7 を引いたものと**数学的に厳密に一致する**．これは「他 9 ドメインの係数・intercept がビット単位で不変」という非退行条件を構成的に保証する手段として使える（Iter32 の `class_weight` 結合バグのような相互依存は，加算が fit 後であるためここには生じない）．本番 artifact は改修後の訓練スクリプトで作り直す（B116(B) により埋め込み計算は **wafl-ctrl5** で行う）が，その結果が上記の減算版と一致することを検証に用いること．

### 計画 (Iter77)

**単一レバー**

`education_specific_correction_removal` = `revert_intercept_delta_and_threshold`．education ドメインだけを狙い撃ちにした後付け補正を **(a) 訓練時 `intercept_delta=+0.7`** と **(b) 実行時 `EDUCATION_THRESHOLD=+0.05`** の 2 つ同時に撤去し，10 ドメイン均一な決定則へ戻す．config.yml の note が定める通り，(a)(b) は分類器の再訓練を共有するため 1 反復でまとめて外す（個別に外すと実機コストが倍になるだけで切り分けの意味がない）．

**仮説（事前登録）**

「education 固有補正の撤去により，education の recall は低下するが precision は上昇し，全体 top1_accuracy は基準線 0.595625 から実質的に変化しない（±1.5pt = 二項 SE 0.0123 の約 1.2 倍以内）」．根拠は，(i) `+0.05` の撤去だけを取れば top1 は **+0.56pt 上昇**する（上表，決定論的に算出済み），(ii) `+0.7` の採用時（Iter44）も top1 は有意変化しなかった，(iii) `+0.05` の採用時（Iter52）も McNemar p=0.2636 だった，の 3 点である．**なお B116 により，top1 が悪化した場合でも撤去は維持する．top1 は判定基準ではなく報告対象である．**

**固定する構成**

`config.yaml`（`embedding_model=nomic-embed-text`，`routing_method=supervised_classifier`，`confidence_threshold=0.0`，`dispatch_top_k=1`，`expert_model=expert-mesh-{domain}-lora`，`judge_model`），較正手法（`temperature`，Iter31 adopted），訓練データ `data/classifier_train.jsonl`（1,427 件，内容不変），`_extract_sample_weights()`，`http_server.py` / `aggregator.py` / `node.py` / `mise.toml`，実機 10 ノード構成をすべて変更しない．conformal 系列（Iter69〜76）の実行時配線は行わない（B115(2) によりクローズ済み）．

**変更するファイルと箇所（4 ファイル＋テスト）**

1. `scripts/train_domain_classifier.py`: L194-204 の `intercept_delta` ブロックを削除．`train_classifier()` の docstring から education 固有補正の記述を削る．
2. `classifier.py`: `EDUCATION_THRESHOLD` 定数（L52-56）と `estimate_confidence_classifier()` の `if domain == "education":` 分岐（L77-78）を削除し，全ドメイン一律に生確率を返す．モジュール docstring L31-34 を削除．
3. `scripts/evaluate_classifier_calibration.py`: `--education-threshold` / `--education-logit-bias` の両 CLI 引数と，`predict_calibrated_rows()` 内の加算箇所（L484-489 / L611-618 / L660-667 と logit bias 相当箇所），stderr 診断の該当文字列を削除．
4. テスト: `tests/test_classifier.py` の education 加算テスト（L65-72 周辺）を「**全 10 ドメインで生確率がそのまま返る**」ことを検証する回帰テストへ置き換える（削除ではなく，撤去仕様を固定するテストへ差し替える）．`tests/test_evaluate_classifier_calibration.py:713-828` の `education_threshold` 4 件は対象機能ごと消えるため削除し，代わりに「`predict_calibrated_rows()` の出力が `predict_proba` の生値と一致する（どのクラスにも加算がない）」テストを 1 件追加する．
5. 本番 artifact `models/domain_classifier.joblib` の再生成（`models/` は gitignore 対象のため履歴に残らない．旧版を `models/domain_classifier_pre_iter77_edu_corrected.joblib` として退避し，sha256 を journal に記録すること）．

**レバーを読むコード行と到達条件（d0004 §4 の再発防止）**

本レバーは「設定値の切り替え」ではなく**コードの削除**であるため，到達しない no-op になり得るのは削除漏れの場合だけである．発火の確認は次の 2 点で行う．

- 訓練側: 再生成した joblib の `calibrated_classifiers_[i].estimator.intercept_[edu_idx]` が，退避した旧 artifact の同値から **ちょうど 0.7 低い**こと（全 5 fold）．他 9 クラスの `intercept_` と全クラスの `coef_` は旧 artifact とビット一致すること．
- 実行時: 実機本走の `results/<ts>/results.jsonl` の `probe_candidates` 中 education ノードの confidence 最小値が **0.05 未満**になること（基準線では 0.050006 が下限で，これは `+0.05` が効いている限り破れない不等式である）．

**実験手順**

1. 上記 4 ファイルを改修し，`uv run pytest` と `uv run ruff check` を通す．
2. **wafl-ctrl5（192.168.15.10，B116(B) の絶対条件）**の Ollama に対して `uv run python scripts/train_domain_classifier.py --train-data data/classifier_train.jsonl --embedding-model nomic-embed-text --ollama-host <wafl-ctrl5 の Ollama ホスト> --output models/domain_classifier.joblib` を実行し，artifact を再生成する．上記「訓練側」の差分検証を行う．
3. オフライン事前確認（同じく wafl-ctrl5）: `scripts/evaluate_classifier_calibration.py --dataset data/dataset.jsonl --classifier models/domain_classifier.joblib --education-threshold なし` で 1,600 行の argmax・per-domain recall/precision を取得し，下表の予測と突き合わせる．**これは本実験の代替ではない（B116(A)）．**
4. `mise run setup` → `mise run deploy`（wafl500〜509 へ新 artifact とコードを配布）．**デプロイ検証として各ノードで `grep -c EDUCATION_THRESHOLD classifier.py` が 0 を返すことを確認する**（Iter22 のデプロイ漏れと同型の失敗を防ぐため）．
5. 先頭 20 問の予備実行で education confidence < 0.05 の行が出ることを確認．
6. **wafl500〜509 で 1,600 問のフルスペック本走を 1 回**（B116(A) の絶対条件）．`mise run analyze` まで実施．

**成功条件（事前登録）**

基準線は Iter75 本走 `results/20260923_150540/`（top1=0.595625，kappa=0.564477，misrouting=0.404375，education recall=0.535294 / precision=0.374486，single_domain_top1=0.608，compound_top1=0.41，ECE=0.055085，fallback_rate=0.0）．

| 区分 | 指標 | 基準線 | 予測 | 合格条件 |
|---|---|---|---|---|
| **主基準（完遂）** | education 固有補正の残存 | 4 箇所 | **0 箇所** | `grep -rn "intercept_delta\|EDUCATION_THRESHOLD\|education_threshold\|education_logit_bias" --include="*.py"` が（履歴コメントを除き）0 件．実機ノード側も 0 件 |
| **主基準（発火）** | education confidence 最小値 | 0.050006 | **< 0.01** | 0.05 未満であること |
| **主基準（発火）** | education intercept 差分 | — | **-0.7（全 5 fold）** | 旧 artifact との差が education のみ -0.7．他 9 クラスの coef/intercept はビット一致 |
| 非退行 | 他 9 ドメインの recall/precision（18 指標） | 各値 | ほぼ不変 | BH 補正後に有意退行 **0 件** |
| 報告のみ | top1_accuracy | 0.595625 | **0.58〜0.61** | 判定に用いない（B116．悪化しても撤去維持）．McNemar と Wilson 95%CI を必ず併記 |
| 報告のみ | education recall | 0.535294 | **0.44〜0.50** | 低下が予想され，それ自体は棄却理由にならない |
| 報告のみ | education precision | 0.374486 | **0.42〜0.52** | 上昇が予想される（Q2 のトレードオフ） |
| 報告のみ | ECE / kappa / answer_quality | 0.055085 / 0.564477 / — | — | 軸②③の変化は 3SD=2.6pt を超えない限り有意としない（success_criteria (5)） |

- **adopted の定義（本レバー固有）**: 本レバーはユーザー指示による方針変更の実装であり，「精度が上がったら採用」という通常のレバーとは判定構造が異なる．**主基準 3 つ（完遂・発火 2 種）と非退行条件を満たせば adopted** とし，top1 の増減は結果として報告する．
- **invalid（実験不成立）の判別**: 実機本走の top1 が 0.595625 と小数点以下まで一致，または education confidence 最小値が 0.05 以上のままなら，**「効果なし」ではなく「デプロイ漏れ」を既定の解釈とする**（d0004 §4，Iter22 と同型）．
- **ノイズ幅**: top1 の二項 SE は n=1600・p=0.596 で 0.0123（±2.5pt が 2SE）．education recall は n=150 で SE=0.041（±8.1pt が 2SE）と広く，**education 単体の増減を有意に語れる標本ではない**点を分析フェーズで明示すること．

**期待効果**

10 ドメイン均一な決定則へ戻すことで，以降のレバー（優先度 2 の `embedding_model_replacement`，優先度 3 の `cross_domain_training_data_augmentation`）を「education だけ下駄を履いた状態」ではない基準線の上で評価できるようになる．副次的に，`evaluate_classifier_calibration.py` から education 固有パラメータが消えることで，conformal 系列が抱えていた「評価分布と実行時分布の不一致」（Iter75 で判明した論点）も構造的に解消する．

---

### 実装・実験 (Iter77)

計画どおり 4 箇所の education 固有補正を撤去した．`classifier.py` の `EDUCATION_THRESHOLD` 定数と実行時分岐，`scripts/train_domain_classifier.py` の fit 後 `intercept_delta = 0.7` 加算，`scripts/evaluate_classifier_calibration.py` の `--education-threshold` と死にパラメータ `--education-logit-bias` を削除し，テスト 2 ファイルを「加算がないこと」を検証する回帰テストへ差し替えた．`config.yaml`・`http_server.py`・`aggregator.py`・`node.py`・`mise.toml`・訓練データ本体は無変更（単一レバー原則）．

検証: 対象 3 テストファイル 43 件 PASS，`ruff check` PASS．全体スイートの 9 件 FAIL と 23 件の ruff エラーは `git stash -u` で退避しても同数再現する既存の失敗であり，本変更とは無関係であることを確認した．

artifact 再生成は B116(B) に従い wafl-ctrl5 で実施（`models/domain_classifier_pre_iter77_edu_corrected.joblib` へ旧版を退避，sha256 `835a10d6...9242c9` → 新版 `02caf2b8...db408905`）．本走は B116(A) に従い wafl500〜509 で 1,600 問フルスペックを 1 回実施（`results/20260926_171953/`，約 24 分）．

**主基準の発火確認（3 つとも成立）**

- ①完遂: 全 10 ノードで `grep -c EDUCATION_THRESHOLD classifier.py` が 0．リポジトリ全体の 4 パラメータ grep も履歴コメントを除き 0 件．
- ②発火・実行時: 本走の education confidence 最小値 4.4628668777636105e-06（< 0.05）．wafl-ctrl5 でのオフライン事前確認と同一値で再現．
- ③発火・訓練: 新旧 5 fold すべてで `estimator.intercept_[education]` の差分がちょうど -0.7，他 9 クラスの `intercept_` と全クラスの `coef_` はビット単位で完全一致（`np.max(np.abs(diff)) == 0.0`）．

**取得したメトリクス**（本走 1,600 問．基準線は Iter75 本走 `results/20260923_150540/`）

| 指標 | 基準線 | Iter77 |
|---|---|---|
| top1_accuracy | 0.595625 | **0.615625**（Wilson 95%CI [0.591540, 0.639157]） |
| cohens_kappa | 0.564477 | 0.588209 |
| education recall | 0.535294 | 0.470588 |
| education precision | 0.374486 | 0.547945 |
| ECE | 0.055085 | 0.076640 |

misrouting_rate=0.384375，fallback_rate=0.0，dispatch_failure_rate=0.00125，brier=0.206158，auroc=0.735098，single_domain_top1=0.629333，compound_domain_top1=0.41．axis2/3 は answer_quality_accuracy=0.56，end_to_end_accuracy=0.345．

統計（`metrics.py` の既存関数をそのまま使用）: 全体 top1 の McNemar は discordant 11（基準線のみ正解）対 43（新のみ正解），chi2=17.796，p=2.4586e-05．**非退行条件（他 9 ドメイン×recall/precision=18 指標）は BH 補正（q=0.05）後の有意退行 0 件**．education は recall が McNemar p=0.002569 で低下，precision が Fisher exact p=0.001054 で上昇（選択行数 243→146）．

**申し送り**: `mise run analyze` を引数なしで実行すると `ls -1d results/*/ | sort` のアルファベット順により `results/iter45_preliminary/` を誤選択する．今回は `-- 20260926_171953` の明示指定で回避した（本イテレーションのスコープ外のため未修正）．また `mise run setup` の素の `uv sync` が research extra を落とすため，以降は `uv sync --extra research` で復旧する必要がある．

---

### Iteration 77 実行済み

**変更（単一レバー）**: `education_specific_correction_removal=revert_intercept_delta_and_threshold`．education 固有の後付け補正 4 箇所（訓練時 `intercept_delta=+0.7`，実行時 `EDUCATION_THRESHOLD=+0.05`，評価スクリプトの `--education-threshold`・`--education-logit-bias`）を削除し，10 ドメイン均一な決定則へ戻した．テスト 2 ファイルを「どのクラスにも加算がない」ことを固定する回帰テストへ差し替え，artifact を wafl-ctrl5 で再生成のうえ wafl500〜509 で 1,600 問本走 1 回（`results/20260926_171953/`）．

**判定: adopted**（事前登録した本レバー固有の定義「主基準 3 つ＋非退行を満たせば adopted」を充足）．

- 主基準①完遂: 4 パラメータの grep が履歴コメントを除きリポジトリ・実機 10 ノードとも 0 件．
- 主基準②発火（実行時）: education confidence 最小値 4.46e-06 < 0.05（基準線の下限 0.050006 が破れた）．
- 主基準③発火（訓練）: 新旧 artifact の差分が education intercept のみ厳密に -0.7（全 5 fold），他 9 クラスの `intercept_`・全クラスの `coef_` は `np.max(np.abs(diff)) == 0.0`．
- 非退行: 他 9 ドメイン×recall/precision=18 指標に BH 補正（q=0.05）後の有意退行 0 件．

**結果と有意性の判定**

| 指標 | 基準線 Iter75 | Iter77 | 判定 |
|---|---|---|---|
| top1_accuracy | 0.595625 | **0.615625**（Wilson 95%CI [0.591540, 0.639157]） | **有意な改善**（McNemar 11 vs 43，chi2=17.796，p=2.46e-05） |
| cohens_kappa | 0.564477 | 0.588209 | top1 と同方向 |
| education recall | 0.535294 | 0.470588 | 有意に低下（McNemar p=0.002569）．事前予測 0.44〜0.50 の範囲内 |
| education precision | 0.374486 | 0.547945 | 有意に上昇（Fisher p=0.001054）．事前予測 0.42〜0.52 を上抜け |
| ECE | 0.055085 | 0.076640 | 判定材料外の所見（後述） |

top1 の +2.0pt は**ノイズではない**．軸①（ルーティング系）は決定論的で反復間ノイズ床を持たない（config success_criteria (5)）ことに加え，対応のある McNemar が p=2.46e-05，discordant の内訳が 11 対 43 と一方向に偏っている．二項 SE=0.0123（2SE=±2.5pt）は独立標本を仮定した保守的な幅であり，同一問題集合の対比較ではこちらが主基準である（success_criteria (1)）．

**論点 1: 事前予測レンジ 0.58〜0.61 の上抜け（+2.0pt）の内訳**

事前シミュレーションは `+0.05` 撤去分のみを決定論的に算出して +0.56pt（0.597500→0.603125，flip 38 行）としていた．実測の差分はその約 3.6 倍で，残り約 +1.4pt は `+0.7` 撤去分である．これは temperature 較正の内側（fold ごとの logit）に効くため基準線の `probe_candidates` からは逆算できず，事前レンジの上限側の不確実性として残っていた部分がそのまま顕在化した形である．機序は per-domain の内訳に明瞭に出ている: education の選択行数が 243→146（-97）へ減り，その大半が他 9 ドメインへ戻って **8/9 ドメインの recall が上昇**（social_science +5.4pt，legal +4.4pt，general +4.3pt，medical +3.9pt，mathematics と natural_science は ±0，precision の低下はいずれも -2.6pt 以内）した．education の真の支持数は約 170 行であるのに 243 行を選んでいた＝**過剰選択が全体 top1 を押し下げていた**という解釈で，Q2 の理論（事前分布が均一な設定では単一クラスの intercept シフトは Bayes 最適から離れる方向）と整合する．過剰な一般化は避けるべきで，本反復が示したのは「本データ・本分類器において，事前分布が均一化済みの訓練設定に単一クラスの経験的 intercept を重ねると全体 accuracy を損なう」までである．

**論点 2: ECE 悪化（0.055085→0.076640）の解釈 — 判定材料ではなく所見**

事前登録の成功条件に ECE は「報告のみ」として置かれており，判定に用いないのが妥当である．その上で中身を確認したところ，**これは較正の質的な劣化ではなく，accuracy だけが上がって confidence 分布が動かなかったことの算術的な帰結**である．

- 平均 confidence は 0.5466→0.5398 とほぼ不変，accuracy は 0.5960→0.6164．平均ギャップ（conf - acc）は -0.0493→-0.0766 で，**符号は一貫して負（過小確信）**．10 ビンすべてで acc > conf（新）となっており，ECE ≈ |平均ギャップ| が成立する単調な過小確信である．過大確信側への崩れ（危険な方向）は起きていない．
- 内訳を分けると，single ドメイン行（n=1498）が conf 0.5419 / acc 0.6302（ギャップ -0.088），compound 行（n=100）が conf 0.5080 / acc 0.4100（ギャップ +0.098）で，compound 側は基準線から一切動いていない（0.41 で同値）．つまり ECE の悪化はすべて single 行の accuracy 上昇に由来する．
- 構造的な原因として，較正（temperature）は単一ラベルの訓練データで当てているのに，評価の正解判定は `selected_domain in expected_domains` という複数正解許容であるため，single 行でも過小確信が出やすい．加えて，基準線の ECE が見かけ上良かったのは **education への +0.7/+0.05 が過大確信を人為的に注入して過小確信を部分的に打ち消していたため**であり，補正を外したことでもともとの過小確信が露出したと読める．「ECE が良い基準線」は補正のアーティファクトだったという点は，今後 ECE を横比較する際の注意点として残す．
- 今後のレバーへの影響: 絶対的な confidence 値に閾値を置く施策（conformal 系列，`confidence_threshold`，`dispatch_policy` の gap 閾値など）は，分類器を触るレバーの後に**必ず temperature を当て直してから**評価する必要がある．単調な過小確信なので順位（AUROC 0.7460→0.7351，Brier 0.20242→0.20616 といずれも僅差で悪化）はほぼ保たれており，再較正 1 段で回復可能な種類の劣化である．なお conformal 系列の実行時配線は B115(2) でクローズ済みのため，本件が直ちに新規作業を要求するわけではない．

**論点 3: education の recall -6.5pt / precision +17.3pt のトレードオフ**

Q2 で事前に予測したとおりの方向であり，B116(1) により**撤去は維持する**．education だけを特別扱いする方向へは戻さない．education 単体は n=170 前後（単一ドメイン評価では n=150，SE=0.041）で ±8pt が 2SE に相当し，そもそも単体の増減を強く語れる標本ではない．一方で precision の上昇は選択行数 243→146 という大きな変化を伴っており Fisher p=0.001054 と有意である．全体としては「education の過剰選択を止めたぶん，全ドメインの割り当てが正常化した」と要約される．

**想定外の挙動**: なし（言語崩れ・発散・OOM なし．fallback_rate=0.0，dispatch_failure_rate=0.00125 は基準線と同水準）．軸②③は answer_quality_accuracy=0.56 / end_to_end_accuracy=0.345 で，success_criteria (5) の 3SD=2.6pt を超える変化とは判定しない．

**学び**

1. **単一クラスだけの経験的 intercept 補正は，その補正で改善した指標（education recall）以上のコストを他 9 ドメインから徴収していた**．Iter44/52 の採用時は education recall を主基準にしていたため全体 top1 への負の寄与が見えず，2 反復ぶん誤った方向へ投資していた．ドメイン固有補正を禁じる B115 の方針は，本反復で定量的にも裏づけられた（+2.0pt の回収）．
2. **fit 後に加算した補正は，旧 artifact からの減算で厳密に再現できる**という性質（調査 Q3）が非退行の証明手段としてそのまま使えた．「他 9 クラスがビット一致」を統計的にではなく構成的に示せたのは，`class_weight` 結合（Iter32）のような相互依存が無いことの直接的な証拠になっている．今後も fit 後 post-hoc 補正を入れる際は，この検証可能性を保つ設計にする価値がある．
3. **較正指標は accuracy の変化に引きずられる**．confidence 分布を動かさずに accuracy だけを上げるレバーは，それ自体が良い変更であっても ECE を悪化させる．ECE 単独を成功条件に据えると，accuracy を上げるレバーを誤って棄却しうる．今後は ECE を見るときに必ず「平均 conf と平均 acc の符号付きギャップ」を併記する．
4. 運用上の落とし穴 2 件（`mise run analyze` の引数なし実行が `results/iter45_preliminary/` を誤選択する，`mise run setup` の素の `uv sync` が research extra を落とす）は backlog B118 に記録した．

**次の一手**: B116 の優先順位に従い，優先度 1 の**複合設問評価集合の拡充**（research_frontier 最上位）へ移る．config.yml の levers 末尾へ `compound_eval_set_expansion` を追加した（詳細と比較可能性の担保方針は backlog B118）．


## Iteration 76: conformal予測集合サイズを棄権信号に使う選択的ルーティングの価値を測る

### 調査 (Iter76)

Iter75 の申し送り（backlog B112・停止条件 2）に従い，config の levers 使い切り後の代替アプローチを tavily-search で広めに調査した．B112 が挙げた 4 候補（多ラベル化，binary relevance，conformal risk control，選択的予測）のうち，**前 2 者は本リポジトリで既に試し切り済みである**ことをまず確認した（`dispatch_candidate_ranking=multilabel_binary_relevance_head` は Iter59 で rejected，`multilabel_training_signal` 以降 Iter60〜68 の 9 反復で合成多ラベル信号の量・質量比・構造・質をすべて探索し Iter68 で打ち止め確定）．したがって残る実行可能な候補は後 2 者である．

**問い**

- Q1: conformal risk control（CRC）で複合設問の 2 ドメイン同時被覆を直接制御できるか．本リポジトリの標本規模で意味のある実験になるか．
- Q2: 予測集合を「棄権・人手エスカレーション」の信号として使う場合，先行研究はどう定式化・評価しているか．基準線は何か．
- Q3: その評価を本リポジトリの既存データで実行したとき，どこへ着地するか（Iter71 以降の慣行に従い本実行前に数値で言語化する）．

**Q1: conformal risk control — 定式化は可能だが本標本では実験にならない**

CRC（Angelopoulos, Bates, Fisch, Lei, Schuster, "Conformal Risk Control", ICLR 2024, arXiv:2208.02814，<https://arxiv.org/abs/2208.02814>）は，単調かつ有界な損失の期待値を有限標本で制御する枠組で，参照実装 `aangelopoulos/conformal-risk` の README が多ラベル分類の例として偽陰性割合 `L_i(λ) = 1 - |Y_i ∩ C_λ(X_i)| / |Y_i|` を挙げている（<https://github.com/aangelopoulos/conformal-risk>）．MAPIE のドキュメントも同じ損失で実装を公開している（<https://mapie.readthedocs.io/>）．本リポジトリの `expected_domains` はそのまま `Y_i` として使えるため定式化上の障害はない．

**しかし標本が足りない．** 損失を 1,600 行全体で取ると，複合行は 100 行（校正/評価半では各 ~50 行）しかなく，単一ドメイン行 1,500 行（`|Y_i|=1`，損失は通常の非被覆と一致）が λ の校正をほぼ完全に支配する．すなわち CRC は現行の周辺被覆 conformal とほぼ同じ λ に落ち，複合の同時被覆はほとんど動かない．損失を複合行に限定すれば校正標本は ~50 行となり，Iter60〜68 で繰り返し臨界に達した検出力の壁（R-H: n=100・discordant 15〜19 行では ±3〜4 行を検出できない）にそのまま突き当たる．**Iter68 と同じく「実施しても『効果なし』ではなく『検出力不足で判定不能』としか結論できない実験」であり，着手しない**と判断した（複合設問データセットの拡充は research_frontier 相当・人間判断）．

**Q2: 選択的予測（棄権）— 定式化と評価指標，および基準線の強さ**

- Tayebati et al., "Learning Conformal Abstention Policies for Adaptive Risk Management in Large Language and Vision-Language Models", arXiv:2502.06884（2025，<https://arxiv.org/abs/2502.06884>）が，conformal の予測集合サイズを棄権判定に使う定式化（集合サイズ >1 なら棄権する，LAC/APS を比較対象に AUARC 等で評価する）を扱っている．本イテレーションの着想はこれに直接対応する．
- 評価枠組は選択的分類の標準である risk-coverage 曲線と AURC（Geifman & El-Yaniv, NeurIPS 2017）．さらに Traub, Bungert, Lüth, Baumgartner, Maier-Hein, Maier-Hein, Jäger, "Overcoming Common Flaws in the Evaluation of Selective Classification Systems", NeurIPS 2024, arXiv:2407.01032（<https://arxiv.org/abs/2407.01032>）が，AURC が低 coverage 側の少数標本に支配されるという欠点を指摘し，generalized risk（誤りかつ非棄権の同時確率）の曲線下面積 **AUGRC** を代替として提案している．本イテレーションはこの勧告に従い **AUGRC を主指標，AURC を副指標**とする．
- **基準線の強さに関する注意**: 選択的予測の文献では，素の最大ソフトマックス確率（MSP / softmax response）が強い基準線であり，凝った不確実性指標が安定して上回れないことが繰り返し報告されている（Hendrycks & Gimpel 2017 以来．例えば選択的分類の post-hoc 手法をまとめた文献レビューでも MSP + 温度較正の組合せが上位に来る）．**本リポジトリの実行時 confidence は既に temperature 較正済み（Iter31 adopted）の MSP そのものであり，基準線は相当に強い**．

**Q3: 事前シミュレーション（本実行前に実施．B109 制約 (2) の慣行）**

Iter75 の出力 `results/20260923_161147/Iter75_variantA_edu005.jsonl`（1,600 行，`probabilities` / `expected_domains` / `set_size` / `split` を持つ）だけで計算が閉じる．評価は conformal の評価半 n=800（校正半は q_hat の当てはめに使われており in-sample のため副次扱い）．棄権スコアは大きいほど「任せてよい」向き．正誤は `argmax(probabilities) ∈ expected_domains`（eval 半で argmax は `selected_domain` と 800/800 一致，top1=0.605000）．

| 棄権スコア | AURC | AUGRC | err@cov50% | err@cov70% | err@cov80% | err@cov90% |
|---|---|---|---|---|---|---|
| **max_probability（基準線）** | **0.218717** | **0.138450** | **0.2200** | **0.2804** | **0.3219** | **0.3611** |
| margin（top1-top2） | 0.225014 | 0.142492 | 0.2275 | 0.2982 | 0.3328 | 0.3681 |
| negative entropy | 0.224522 | 0.140497 | 0.2175 | 0.2857 | 0.3281 | 0.3569 |
| **conformal set size（本レバー）** | **0.252187** | **0.153391** | **0.2700** | **0.3179** | **0.3391** | **0.3667** |
| conformal set size（同点を max_prob で解く） | 0.234332 | 0.145975 | 0.2475 | 0.2964 | 0.3391 | 0.3583 |

対応ありブートストラップ（B=10,000，seed 42，eval 半 n=800）: **ΔAUGRC = +0.014941，95%CI [0.007923, 0.022549]**（正は劣化方向．改善方向に出る確率 0.0001）．ΔAURC = +0.033470，95%CI [0.014948, 0.050655]．方向は校正半（n=800，AUGRC 0.162959 対 0.146444）でも全 1,600 行（0.158015 対 0.142462）でも同じで，分割に依存しない．

集合サイズ閾値が到達する coverage と，そこへ max_probability を揃えた対比較（discordant 行の誤り数と二項検定）:

| 閾値 | coverage | n | 誤り率（set size） | 誤り率（max prob） | only-size 側の誤り | only-maxp 側の誤り | 二項 p |
|---|---|---|---|---|---|---|---|
| size<=1 | 0.0300 | 24 | 0.0000 | 0.0000 | 0/3 | 0/3 | 1.0 |
| size<=2 | 0.1200 | 96 | 0.1146 | 0.0938 | 5/30 | 3/30 | 0.727 |
| size<=3 | 0.3000 | 240 | 0.1917 | 0.1625 | 21/70 | 14/70 | 0.311 |
| **size<=4** | **0.5425** | **434** | **0.2834** | **0.2281** | **52/99** | **28/99** | **0.0097** |
| size<=5 | 0.8013 | 641 | 0.3385 | 0.3214 | 46/72 | 35/72 | 0.266 |
| size<=6 | 0.9450 | 756 | 0.3783 | 0.3783 | 19/26 | 19/26 | 1.0 |

**この調査で分かったことの要約**

1. B112 が挙げた 4 候補のうち，多ラベル化・binary relevance は既に試し切り済み（Iter59・Iter68 で打ち止め），conformal risk control は定式化できるが標本規模から判定不能が確定しているため着手しない．**残る実行可能な候補は選択的予測（棄権）ただ 1 つである**．
2. 選択的予測は本リポジトリで一度も測っていないが，**実行時に既に存在する confidence（temperature 較正済み MSP）だけで，棄権 20% ならルーティング誤り 0.3950→0.3219，棄権 50% なら 0.2200 まで下がる**．これは B104 A2（配線の是非）を人間に諮るための運用点の表になる．
3. 一方 **conformal の集合サイズは棄権信号として MSP より有意に劣る**（ΔAUGRC +0.0149，95%CI が 0 を跨がない）．機序は解像度の欠如にあると解釈できる：集合サイズは 1〜8 の 8 段階しかなく，size<=4 と size<=5 の間で coverage が 0.54 から 0.80 へ飛ぶため中間の運用点が存在しない．同点を max_probability で解くと AUGRC が 0.153391→0.145975 と MSP 側へ寄る（それでもなお MSP に届かない）ことがこの解釈を支持する．

### 計画 (Iter76)

**仮説（反証形で事前登録する．Iter74 と同型）**

「conformal の予測集合サイズは，棄権・人手エスカレーションの判定信号として，実行時に既に存在する confidence（max_probability）より優れる」は**成り立たない**．eval 半 n=800 で AUGRC は 0.153391 対 0.138450（Δ=+0.014941，劣化方向，ブートストラップ 95%CI [0.007923, 0.022549]）となり主基準は不成立となる．

この仮説を採る根拠は，(a) 上記シミュレーションが入力 jsonl から決定論的に閉じており実行時の再現は確実であること，(b) 選択的予測の文献で MSP が強い基準線であることが繰り返し報告されており，本リポジトリの confidence は temperature 較正済み（Iter31 adopted）でさらに強いこと，の 2 点である．**本実行は，この予測を実装で確認して conformal 系列を根拠をもって閉じ，同時に選択的ルーティングの運用点の表を成果物として残すための反証実験である．合格を探して信号を振り直すことはしない**（margin・negative entropy も掃引済みで，いずれも MSP を上回らない）．

**単一レバー**

`routing_abstention_signal`: ルーティングの棄権スコアを `max_probability`（基準線．実行時の confidence そのもの）→ **`conformal_set_size`**（Iter75 adopted 構成の予測集合サイズ）へ変更する．動かすのはこの 1 点のみ．

**固定する構成**

入力は `results/20260923_161147/Iter75_variantA_edu005.jsonl`（Iter75 adopted の出力．randomized_aps / `--randomization-seed 42` / `--calibration-source eval_holdout` / `--holdout-seed 42` / `--qhat-quantile-direction alpha_lower` / `--qhat-source true_class` / `--confidence-level 0.90` / `--education-threshold 0.05`）を**再生成せずそのまま使う**（埋め込み再計算なし＝差分が棄権スコアの選択のみになる）．評価対象は `split == "eval"` の 800 行．正誤の定義・分類器・埋め込み・`config.yaml`・`http_server.py`・`classifier.py`・`aggregator.py`・`mise.toml`・実機構成はすべて変更しない．**実行時経路への配線は行わない**（B104 A2 は人間判断事項のまま維持）．

**変更箇所（新規 1 ファイル＋テスト．既存ファイルは変更しない）**

1. 新規 `scripts/evaluate_selective_routing.py`（ファイル冒頭に責務を 1 行で記す）．
   - CLI: `--predictions`（必須，jsonl），`--split`（既定 `eval`），`--abstention-signal`（`{max_probability, conformal_set_size, margin, negative_entropy}`，複数指定可），`--bootstrap`（既定 10000），`--bootstrap-seed`（既定 42），`--output`（json）．
   - **レバーを読む行**: 棄権スコア関数テーブル（`_SIGNALS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]]`）を `--abstention-signal` で引く 1 箇所．未知の値は `ValueError`（無言で基準線へ落ちないこと．Iter69 の教訓）．
   - risk-coverage 曲線はスコア降順の安定ソート（`kind="mergesort"`）で構成し，AURC = 選択的誤り率の全 coverage 平均，AUGRC = generalized risk（誤りかつ非棄権の割合）の全 coverage 平均とする．
   - 対応ありブートストラップで Δ(AUGRC)・Δ(AURC) の点推定と 95%CI を出す（行インデックスを再標本化し，両信号を同一の再標本上で評価する）．
   - `size<=k` 閾値の到達 coverage へ max_probability を揃えた対比較（discordant 行の誤り数と `scipy.stats.binomtest`）を出す．
   - stderr へ発火証拠（`abstention_signal=` / `n=` / `split=` / `set_size` 分布）を出す．
2. `tests/test_evaluate_selective_routing.py`: (a) 手組みの小標本で AURC・AUGRC が手計算値と一致すること，(b) 定数スコア（全行同値）のとき AURC が全体誤り率と一致すること，(c) 未知の `--abstention-signal` が `ValueError` になること，(d) 完全な信号（正解行が全て誤り行より高スコア）のとき AUGRC が理論下限に一致すること．

**到達コードパス**

`uv run python scripts/evaluate_selective_routing.py --predictions results/20260923_161147/Iter75_variantA_edu005.jsonl --split eval --abstention-signal max_probability --abstention-signal conformal_set_size --abstention-signal margin --abstention-signal negative_entropy --bootstrap 10000 --bootstrap-seed 42 --output results/<ts>/Iter76_selective_routing.json`
→ jsonl 読み込み → `split == "eval"` で 800 行抽出 → `_SIGNALS[name]`（**レバーを読む行**）→ risk-coverage → AURC/AUGRC → 対応ありブートストラップ → json + stderr．既存コードの分岐に依存しないため到達は CLI 実行そのもので保証される．

**成功条件（事前登録）**

評価半 n=800．基準線は同一 800 行上の `max_probability`．

| 指標 | 定義 | 基準線（予測） | 本レバー（予測） | 合格条件 |
|---|---|---|---|---|
| **AUGRC（主基準）** | generalized risk の coverage 平均 | 0.138450 | **0.153391** | **conformal_set_size < max_probability かつ ΔAUGRC のブートストラップ 95%CI が 0 を跨がないこと**（予測: FAIL） |
| ΔAUGRC の 95%CI（発火・整合） | 対応あり B=10,000, seed 42 | — | **+0.014941 [0.007923, 0.022549]** | 点推定が予測の ±0.002 以内 |
| AURC（副基準・報告） | 選択的誤り率の coverage 平均 | 0.218717 | 0.252187 | 報告のみ（±0.002） |
| size<=4 での対比較（副基準） | coverage 0.5425 に揃えた discordant | 誤り 28/99 | 誤り 52/99 | 報告のみ（二項 p=0.0097） |
| risk-coverage 表（成果物） | 棄権率 0/10/20/30/50% の誤り率 | 0.3950 / — / 0.3219 / — / 0.2200 | — | 4 信号すべてについて出力すること |

- **adopted**: 主基準を満たす（AUGRC が有意に低い）こと．解釈は「conformal の集合サイズは実行時 confidence より良い棄権信号であり，B104 A2 の配線に棄権用途という積極的理由がある」．
- **rejected（予測される帰結）**: 主基準が不成立．解釈は「**conformal を実行時経路へ配線する理由は棄権用途にも無い（実行時に既にある confidence で足りる）．conformal 系列 Iter69〜76 は被覆保証を厳密に得たが本線のルーティングには接続しないという結論で閉じる**」．
- **実装不成立の判別（事前登録．数値が近接しているため事後に決めない）**:

| 実測パターン | 判定 |
|---|---|
| AUGRC が 0.153391 / 0.138450（±0.002）で ΔAUGRC≈+0.0149 | **正しく発火**（主基準の判定へ進む） |
| 4 信号の AUGRC が互いに完全一致 | **実装不成立**（`--abstention-signal` がスコア関数テーブルへ届いていない） |
| max_probability の AUGRC が 0.138450 から外れる | **実装不成立**（正誤判定または split 抽出の定義違い．基準線は入力 jsonl だけで決まる量） |
| 上記いずれにも当たらない | **実装不成立**を第一に疑う（既定の解釈） |

- **非退行条件**:
  1. **rank_1（argmax）不変**: 棄権は選択そのものを変えないため定義上不変．eval 半の top1 = **0.605000**，全 1,600 行 = **0.597500**，argmax と `selected_domain` の一致 800/800（eval）・1,600/1,600（全体）を出力に含めること．いずれかが動いたら実装バグ．
  2. 入力 jsonl を一切書き換えないこと（実行前後で md5 不変を確認する）．
  3. eval 半の set_size 分布が {1:24, 2:72, 3:144, 4:194, 5:207, 6:115, 7:43, 8:1} と一致すること（発火・入力同一性の証拠）．
  4. coverage=1.0 での選択的誤り率が 4 信号すべてで 0.3950（eval 半）に一致すること（スコアに依らない恒等式．曲線構成の健全性チェック）．
  5. 既存テスト（33 件）が PASS のまま，新規 4 件も PASS．`uv run ruff check` PASS．
- **ノイズ幅**: ΔAUGRC の対応ありブートストラップ SE は約 0.0037（95%CI 幅 0.0146 から逆算）で，点推定 +0.0149 は約 4.0 SE．全体 top1 の二項 SE は n=800・p=0.605 で 0.0173．risk-coverage 表の各点は coverage×800 行の二項比率として SE を併記すること（例: coverage 0.5 の 0.2200 は n=400 で SE=0.0207）．

**期待効果**

(1) B104 A2（conformal を実行時経路へ配線するか，配線するなら棄権・人手エスカレーション用途としてか）に対し，「棄権用途としても conformal に積極的理由は無い」という数値的な答えを与え，人間判断の選択肢を 1 つ確定的に落とす．(2) 本研究で初めて選択的ルーティングの risk-coverage を数値化し，「何割を人手へ回せばルーティング誤りがどこまで下がるか」という運用上の表を成果物として残す．(3) 反証が成立した場合，conformal 系列（Iter69〜76，8 反復）を根拠をもって閉じ，次の論点（複合設問データセットの拡充＝research_frontier 相当・人間判断）へ進める．

**コスト**: オフライン完結．埋め込み再計算なし・分類器再訓練なし・実機ノード不使用．スクリプト実装 30 分＋実行 1 分未満．`config.yaml` のスキーマ変更なしのため自律着手してよい．

**留保（判定に用いない）**

(i) Iter75 で観測された「オフラインの argmax と実機本走の `selected_domain` が 1,600 行中 3 行で食い違う」（B112 要レビュー (2)）は本イテレーションでも未解決であり，本評価はオフライン argmax を正とする．3/1,600=0.19% は上記の効果量より 1 桁小さいため結論を変えない．(ii) 校正半 800 行は q_hat の当てはめに使われているため in-sample であり，副次報告に留める（方向は eval 半と同じ）．(iii) 複合設問 100 行の扱いは「`argmax ∈ expected_domains` なら正」とする単一の定義に統一し，2 ドメイン同時被覆は本イテレーションでは扱わない（検出力不足．Q1 参照）．

### 実装・実験 (Iter76)

**実装**（計画どおり新規 1 ファイル．既存ファイルは無変更）

- 新規 `scripts/evaluate_selective_routing.py`．棄権スコア関数テーブル `_SIGNALS: dict[str, Callable[[dict], float]]`（`max_probability` / `margin` / `negative_entropy` / `conformal_set_size` の 4 エントリ）を `_get_signal_scorer()` が `--abstention-signal` の値で引く 1 箇所だけがレバーを読むコード（計画どおり）．未知の値は `ValueError`（`_get_signal_scorer("not_a_real_signal")` で確認）．
- risk-coverage 曲線はスコア降順の安定ソート（`np.argsort(-score, kind="mergesort")`）で構成し，AURC = 選択的誤り率（`cum_errors(k)/k`）の全 coverage 平均，AUGRC = generalized risk（`cum_errors(k)/n`）の全 coverage 平均（`compute_aurc_augrc`）．
- 棄権率 {0,10,20,30,50}% の誤り率表（`compute_risk_coverage_table`）は，既存の `metrics.compute_wilson_confidence_interval` を再利用して各点の Wilson 95%CI を付与した（自前で二項区間の式を再導出していない）．
- 対応ありブートストラップ（`bootstrap_delta_aurc_augrc`，percentile 法，同一再標本上で基準線・候補信号の両方を再評価）で ΔAUGRC・ΔAURC の点推定と 95%CI を出す．`scipy.stats.binomtest` を使った size<=k 閾値対比較（`compute_size_threshold_comparison`）も実装した．AURC/AUGRC・対応ありブートストラップ・discordant 二項検定は本リポジトリに既存実装が無いため新規実装とし，出典（Geifman & El-Yaniv 2017／Traub et al. 2024／percentile ブートストラップと二項検定は標準的な教科書的構成）をスクリプル冒頭の docstring に明記した．
- 非退行診断（`compute_diagnostics`）で eval 半・全 1,600 行の top1・argmax 一致率・set_size 分布を出力に含めた．
- stderr に発火証拠（`abstention_signal=` / `split=` / `n=` / `set_size_distribution=` と各信号の AURC/AUGRC）を出力．

**テスト**（計画どおり新規 `tests/test_evaluate_selective_routing.py` に 4 件）

1. 小標本（n=4）での AURC/AUGRC の手計算一致．
2. 定数スコアのとき AURC が全体誤り率と一致すること（数学的に厳密な一致は「全行同一正誤ラベル」の退化ケースでのみ成り立つため，全行不正解の n=5 標本で検証．一般の混合正誤標本では tie-break の行順序に依存し厳密には成り立たないことをコメントに明記した．AUGRC は同じ退化ケースでも一致しない（`cum_errors(k)/n` の平均が誤り率そのものにならない）ため，本テストでは AURC のみ検証する）．
3. 未知の `--abstention-signal`（`_get_signal_scorer`）が `ValueError` を送出すること．
4. 完全な信号（正解行が全て誤り行より高スコア）で AUGRC が理論下限 `sum(1..n_incorrect)/n/n` に一致すること．

実行結果: `uv run pytest tests/test_evaluate_selective_routing.py -q` → **4 件 PASS**．`uv run pytest -q`（全体）→ **299 件 PASS，12 件 FAIL**（`tests/test_build_dataset.py`・`tests/test_train_domain_classifier.py`．いずれも `CalibratedClassifierCV` に `classes_` 属性が無いという sklearn バージョン起因の既存失敗．本イテレーション開始前から発生していることを `git stash -u` で新規ファイルを退避して再実行し確認済み＝本変更と無関係）．`uv run ruff check scripts/evaluate_selective_routing.py tests/test_evaluate_selective_routing.py` → **PASS**．

**実験（オフライン．実機ノード不使用，埋め込み再計算なし，分類器再訓練なし）**

入力 `results/20260923_161147/Iter75_variantA_edu005.jsonl`（md5 `417c34141be6b602bfbccaf8a4c2d2a7`，実行前後で不変を確認）をそのまま再利用．コマンド（計画の「到達コードパス」どおり）:

```
uv run python -m scripts.evaluate_selective_routing \
  --predictions results/20260923_161147/Iter75_variantA_edu005.jsonl \
  --split eval \
  --abstention-signal max_probability --abstention-signal conformal_set_size \
  --abstention-signal margin --abstention-signal negative_entropy \
  --bootstrap 10000 --bootstrap-seed 42 \
  --output results/20260923_164424/Iter76_selective_routing.json
```

出力: `results/20260923_164424/Iter76_selective_routing.json`（実行時間 4.2 秒）．

**発火証拠・非退行チェックの結果（すべて事前登録値と一致）**

1. rank_1 不変: eval 半 top1=**0.605000**，全 1,600 行 top1=**0.597500**，argmax と `selected_domain` の一致率 **1.0**（800/800・1,600/1,600）．事前登録値と完全一致．
2. 入力 jsonl の md5 は実行前後で **不変**（`417c34141be6b602bfbccaf8a4c2d2a7`）．
3. eval 半の set_size 分布 = **{1: 24, 2: 72, 3: 144, 4: 194, 5: 207, 6: 115, 7: 43, 8: 1}**．事前登録値と完全一致．
4. coverage=1.0（棄権率 0%）の誤り率は **4 信号すべて 0.3950**（316/800）で一致．
5. テスト・lint は上記のとおり全 PASS．

判別表と照合すると，4 信号の AUGRC は互いに異なり（下表），かつ max_probability の AUGRC が予測値 0.138450 と一致しているため，**「正しく発火」**（実装不成立ではない）と判定できる．

**実測値（eval 半 n=800）**

| 棄権スコア | AURC | AUGRC | err@cov50% | err@cov70% | err@cov80% | err@cov90% |
|---|---|---|---|---|---|---|
| max_probability（基準線） | 0.218717 | 0.138450 | 0.2200 | 0.2804 | 0.3219 | 0.3611 |
| margin | 0.225014 | 0.142492 | 0.2275 | 0.2982 | 0.3328 | 0.3681 |
| negative_entropy | 0.224522 | 0.140497 | 0.2175 | 0.2857 | 0.3281 | 0.3569 |
| conformal_set_size（本レバー） | 0.252187 | 0.153391 | 0.2700 | 0.3179 | 0.3391 | 0.3667 |

対応ありブートストラップ（B=10,000，seed 42，percentile 法，eval 半 n=800，vs max_probability）:

| 信号 | ΔAURC | ΔAURC 95%CI | ΔAUGRC | ΔAUGRC 95%CI | 改善方向の割合 |
|---|---|---|---|---|---|
| conformal_set_size | +0.033470 | [0.014948, 0.050655] | **+0.014941** | **[0.007923, 0.022549]** | 0.0001 |
| margin | +0.006297 | [0.000945, 0.011763] | +0.004042 | [0.000917, 0.007271] | 0.0042 |
| negative_entropy | +0.005805 | [-0.001281, 0.012902] | +0.002047 | [-0.001520, 0.005544] | 0.1272 |

conformal_set_size の ΔAUGRC 点推定・95%CI は事前登録値（+0.014941，[0.007923, 0.022549]）と最終桁まで一致した．

size<=4 での対比較（副基準）: coverage 0.5425（n=434），誤り率 size=0.283410／max_probability=0.228111，only-size 側の誤り 52/99，only-maxp 側の誤り 28/99，`binomtest` 両側 p=0.009683．事前登録値（52/99・28/99・p=0.0097）と一致．他の size 閾値（1,2,3,5,6,7）も事前登録の表と全行一致した．

risk-coverage 表（成果物，棄権率 0/10/20/30/50%）は上表の err@cov 列に対応し，4 信号すべてについて出力済み（各点の Wilson 95%CI も `Iter76_selective_routing.json` に含む）．

判定（adopted/rejected）は次フェーズ（rc-evaluator）の担当のため，ここでは行わない．

### Iteration 76 実行済み

**単一レバー**: `routing_abstention_signal` = `max_probability`（基準線）→ **`conformal_set_size`**．

**判定: rejected（仮説は正しく反証された．実装不成立ではない）**．

**主基準**: 事前登録は「eval 半 n=800 の AUGRC で conformal_set_size < max_probability，かつ ΔAUGRC の対応ありブートストラップ 95%CI が 0 を跨がないこと」．実測は AUGRC 0.153391（conformal_set_size）対 0.138450（max_probability），**ΔAUGRC = +0.014941，95%CI [0.007923, 0.022549]**（B=10,000，seed 42）．CI は 0 を跨がないが**符号が合格条件と逆**であり，conformal_set_size は基準線より**有意に劣る**．したがって主基準は不成立で **rejected**．

**ノイズか有意か**: ΔAUGRC の対応ありブートストラップ SE は約 0.0037（CI 幅 0.0146 から逆算）で，点推定 +0.014941 は約 **4.0 SE**．改善方向に出た再標本は 10,000 中 1 本（0.0001）．方向は eval 半・校正半（0.162959 対 0.146444）・全 1,600 行（0.158015 対 0.142462）で一致し，分割にも依存しない．**ノイズ幅を明確に超えた有意な劣化**である．副基準（coverage 0.5425 に揃えた size<=4 の対比較）も discordant 52/99 対 28/99，二項検定 両側 p=0.009683 で同じ方向を示した．

**実装不成立ではないことの根拠（事前登録した判別表との照合）**: (1) 4 信号の AUGRC が互いに異なる（テーブルへ届いている），(2) 基準線 max_probability の AUGRC が事前登録値 0.138450 と一致（正誤判定・split 抽出の定義が一致），(3) ΔAUGRC の点推定・95%CI が事前登録値と最終桁まで一致，(4) eval 半 set_size 分布が {1:24,2:72,3:144,4:194,5:207,6:115,7:43,8:1} と一致，(5) 入力 jsonl の md5 が実行前後で不変．判別表の「正しく発火」の行に該当する．**Iter69〜71 で繰り返した「実装不成立」とは異なり，今回は仮説そのものが反証された**．

**非退行**: eval 半 top1=0.605000・全 1,600 行 top1=0.597500・argmax と `selected_domain` の一致 800/800・1,600/1,600 で事前登録値と完全一致（棄権は選択を変えないという定義上の恒等式が成立）．coverage=1.0 の誤り率は 4 信号すべて 0.3950 で一致（曲線構成の健全性）．新規テスト 4 件 PASS，既存 33 件 PASS 維持，`ruff check` PASS．全体スイートの 12 件 FAIL は `CalibratedClassifierCV.classes_` 不在という sklearn バージョン起因の既存失敗で，`git stash -u` による退避後も再現するため本イテレーション由来ではない．

**この反復で言えること（因果として言える範囲）**

1. **conformal 予測集合のサイズは，選択的ルーティングの棄権信号として max_probability に劣る**（eval 半 n=800，ΔAUGRC +0.0149，4.0 SE）．一般化できるのは「本リポジトリの 10 ドメイン・temperature 較正済み分類器・Iter75 adopted の randomized APS 構成・名目 0.90」という条件下の主張までである．
2. **機序の考察（解像度の欠如）**: 棄権の目的は「top1 が誤りである確率」で行を順序付けることだが，集合サイズは 1〜8 の 8 段階しか値を取らず，同値行を区別できない．実際 size<=4 と size<=5 の間で coverage が 0.5425 から 0.8013 へ飛び，その間に運用点が存在しない．同点を max_probability で解くと AUGRC が 0.153391→0.145975 と基準線側へ寄る（それでも届かない）ことがこの解釈を支持する．加えて，集合サイズは「複数クラスにまたがる不確実性の広がり」を表す量であって top1 の正しさの単調な指標ではなく，棄権という目的関数に対しては生の確信度スコアの方が直接的である．
3. **同系列の手組み信号も基準線を上回らない**: margin は ΔAUGRC +0.004042 [0.000917, 0.007271]（有意に劣る），negative_entropy は +0.002047 [-0.001520, 0.005544]（差は判定不能）．**temperature 較正済み MSP という基準線が強いという選択的予測の文献（Hendrycks & Gimpel 2017 以来）の報告が，本リポジトリでもそのまま再現した**．
4. **肯定的な成果（判定とは独立）**: 選択的ルーティングの risk-coverage を本研究で初めて数値化した．実行時に既に存在する confidence だけで，棄権 20% でルーティング誤り 0.3950→0.3219，棄権 50% で 0.2200 まで下がる．B104 A2 を人間へ諮る際の運用点の表として `results/20260923_164424/Iter76_selective_routing.json`（Wilson 95%CI 付き）に残した．

**レバーの扱い**: `routing_abstention_signal` は `values: [conformal_set_size]` の単一値のため本反復でクローズ．これにより **conformal 系列（Iter69〜76，8 反復）は「被覆保証は厳密に得た（Iter72 adopted・Iter75 で実行時分布での外的妥当性も確認）が，集合サイズは dispatch の絞り込みにも棄権判定にも使えず，本線のルーティングには接続しない」という結論に到達した**（この結論の確定と B104 A2 への回答は不可逆のため人間判断を仰ぐ．backlog B114 要レビュー (1)）．

**次の一手（停止条件 1 を適用）**: config の levers は再び使い切りだが，Iter76 は既に停止条件 2（調査フェーズからの再探索）の結果であるため，今回の学びから新レバーを 1 つ考案して config へ追記する．**`routing_abstention_scorer = learned_deferral_head`**（校正半 800 行だけで学習した post-hoc の棄権スコア器を eval 半 800 行で評価する）．根拠は本反復の機序考察で，劣った原因が「conformal であること」ではなく「8 段階という解像度の粗さ」であり，複数の既存特徴（max_probability・margin・negative_entropy・set_size・上位確率の形状）を連続値へ束ねれば基準線を上回る余地があるかを，同じ AUGRC・同じブートストラップ枠組で 1 回で判定できるためである．手組み信号の振り直し（margin・entropy）は掃引済みで再訪しない（本反復で明示的に否定した）．検出力は ΔAUGRC の SE≈0.0037 から，0.008 程度以上の効果なら検出できる．

**次の自分向けの非自明な学び**

1. **「CI が 0 を跨がない」だけでは合格条件ではない．符号まで事前登録しておくこと**．今回は CI が 0 を跨がなかったが符号が逆で，事前登録の文言（`conformal_set_size < max_probability` かつ CI が 0 を跨がない）が無ければ「有意差あり＝採用」と誤読し得た．
2. **反証を事前登録して閉じる実験は，判定が事後の解釈に揺れない**．Iter74 に続き 2 例目で，事前シミュレーションの数値（AUGRC・CI・discordant の分割表）が最終桁まで再現した（Iter71 以降 6 反復連続）．オフラインで閉じる評価では，事前シミュレーションを「実行の代替」ではなく「合否条件の数値化」に使う運用が機能している．
3. **離散段数の少ない量を連続スコアの代わりに使うと，risk-coverage の運用点が飛ぶ**．集合サイズのような整数値信号を選択的予測へ持ち込むときは，同点解消規則（今回は max_probability）を最初から設計に含めないと，指標以前に「欲しい coverage を作れない」という実務上の欠陥が出る．

---

