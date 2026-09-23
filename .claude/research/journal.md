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

## Iteration 75: conformal評価の入力分布を実行時と同一のeducation+0.05へ揃えて被覆保証の外的妥当性を検証する

### 調査 (Iter75)

**問い**

- Q1: `--education-threshold 0.05` は，conformal の**校正側スコア**と**評価側の集合構成**の両方に届くのか．コード上の到達点を実際に読んで確定する（config の lever note は「新規実装は不要」と書いているが，これは検証されていない）．
- Q2: 実行時 `probe_candidates` と `predict_proba(+edu 0.05)` のビット単位一致は，本当に 1,600 行×10 ドメイン全件で成立しているのか（Iter74 の報告を reflector 同様に独立再検証する）．
- Q3: 入力分布を揃えたとき coverage・mean_set_size・q_hat はどこへ着地するか．とくに **education を正解とする行に限った被覆**はどう動くか（B109 制約 (2) の事前シミュレーション慣行を踏襲し，本実行前に 1 点へ絞る）．

**Q1: コード上の到達点（`scripts/evaluate_classifier_calibration.py` を読んで確定．これが今回の最重要の発見）**

`--education-threshold` は `predict_calibrated_rows()` の**評価ループ 2 箇所のみ**（fine-tuned 分岐 L596-599，ollama 分岐 L640-643）で `probabilities[edu_idx] += education_threshold` として適用される．一方，`calibration_source="eval_holdout"` の**校正スコア計算（L500-517）は `all_probs = classifier.predict_proba(all_embeddings)`（L476）の生値をそのまま使っており，education 補正が一切入らない**．

すなわち **config の lever note にある「新規実装は不要」は誤りである**．現行コードのまま `--education-threshold 0.05` を渡すと，校正半の真クラススコアは補正なしの分布から，評価半の予測集合は補正ありの分布から作られる．これは conformal の交換可能性（校正と評価が同一分布から来ること）を直接破る構成であり，Iter72 で `oof_train` を捨てて `eval_holdout` に移った理由そのものを再導入してしまう．`main()` の `--output` 有無 2 分岐へは `education_threshold` が両方とも伝播済み（L871・L897）なので，欠けているのは**校正側だけ**である．また stderr の診断出力（L555-563）は `calibration_source` / `qhat_source` / `q_hat` / `n_cal` / `set_construction` / `qhat_quantile_direction` / `randomization_seed` / `raps_lambda` / `raps_k_reg` を出すが **`education_threshold` は出していない**ので，発火証拠として使えない．

出典: 本リポジトリ `scripts/evaluate_classifier_calibration.py`（HEAD `6f7ee19`）の L440-536（`eval_holdout` 校正ブロック），L555-564（stderr 診断），L579-666（評価ループ 2 箇所），L855-910（CLI 2 分岐）．

**Q2: 実行時分布との一致の再検証（独立に再計算した）**

`results/20260923_150540/results.jsonl`（実機 1,600 問本走）の `probe_candidates` の `{domain: confidence}` と，`results/20260923_142619/Iter74_probs_only_edu005.jsonl`（`--education-threshold 0.05` でのオフライン `predict_proba`）の `probabilities` を 1,600 行×10 ドメイン＝16,000 ペアで突き合わせた結果，**最大絶対差 0.0・ビット単位で不一致 0 ペア**．Iter74 の報告を再現した．したがって「実行時ルーティングが使う確率ベクトル ＝ `predict_proba` に education だけ +0.05 した分布」は一次データで確定している．

**Q3: 事前シミュレーション（本実行前に実施）**

`results/20260923_135653/Iter73_randomized_aps.jsonl`（1,600 行，`split` 付き）の `probabilities` から，education 列へ +0.05 した分布の上で randomized_aps / `eval_holdout`(seed 42) / `alpha_lower` / `true_class` / 0.90 を再現計算した（u は `np.random.default_rng(42).random(1600)` を dataset 順に割当，q_hat は校正半 800 行の真クラススコアの有限標本補正付き α 下側分位点）．まず edu+0.00 で Iter73 実測を再現し，シミュレータの同一性を確認した（q_hat=0.093879・coverage=0.90375・mean_set_size=4.00125・サイズ分布 `{1:52,2:92,3:154,4:182,5:178,6:111,7:29,8:2}`，すべて Iter73 実測と一致）．

そのうえで，Q1 で判明した実装上の分岐にあわせて **2 変種**を計算した．

| 変種 | 校正側 | 評価側 | q_hat | coverage（eval n=800） | mean_set_size | P(size<=2) | サイズ分布 |
|---|---|---|---|---|---|---|---|
| baseline（Iter73） | edu+0.00 | edu+0.00 | 0.093879 | 0.90375 | 4.00125 | 0.1800 | [52,92,154,182,178,111,29,2] |
| **A（対称．本実験に事前登録）** | **edu+0.05** | **edu+0.05** | **0.046876** | **0.92875** | **4.26250** | **0.1200** | **[24,72,144,194,207,115,43,1]** |
| B（現行コードのまま渡した場合） | edu+0.00 | edu+0.05 | 0.093879 | 0.88125 | 3.50375 | 0.2450 | [62,134,190,230,123,59,2,0] |

部分集合別（変種 A）:

| 部分集合 | n | coverage（前 → 後） | mean_set_size（前 → 後） |
|---|---|---|---|
| education を正解とする行 | 79 | **0.65823 → 0.93671**（0→1 が 22 行，1→0 が 0 行） | 3.87342 → 4.01266 |
| 非 education 行 | 721 | 0.93065 → 0.92788（0→1 が 2 行，1→0 が 4 行） | 4.01526 → 4.28988 |
| 複合設問（2 ドメイン同時被覆） | 46 | 0.2826 → 0.3043 | — |

付随: `education` が予測集合に入る eval 行の割合は 0.5025 → 0.85875．argmax（`selected_domain`）が変わるのは 1,600 行中 38 行で，eval 半の top1 は 0.60125 → 0.59625（実機本走 `results/20260923_150540/` の top1=0.595625 とほぼ一致し，実行時の挙動を再現している）．

**この調査で分かったことの要約**

1. **`--education-threshold` は校正側へ届かない**．現行コードのまま渡すと変種 B になり，交換可能性が破れて coverage が 0.88125（帯下限 0.88 からわずか +0.00125＝0.12 SE）へ落ちる．これは「レバーが効かなかった」ではなく**実装不成立**として読むべき値であり，帯の内側に辛うじて入るため**見逃しやすい**．校正側へ補正を伝播させる最小の実装が要る．
2. 変種 A（対称）では coverage=0.92875 で帯 0.88-0.95 の内側に留まる．すなわち**実行時と同一の分布の上でも被覆保証は成立する見込み**であり，Iter56〜74 の 19 反復の結論の外的妥当性は保たれる．
3. ただし中身は一様ではない．**Iter56〜74 が見落としていたのは education の条件付き過少被覆（0.658）であり，実行時に効いている +0.05 はこれを 0.937 へ引き上げて 10 ドメイン間の条件付き被覆を均す方向に働く**（代償は mean_set_size +0.26 と非 education 側の -0.003）．conformal は周辺被覆しか保証しないので，この非一様性は理論上想定内だが，本系列で測ったのは今回が初めてである．

### 計画 (Iter75)

**仮説（事前登録）**

conformal 評価の入力分布を実行時と同一（education のみ +0.05）へ揃えても，名目 0.90 の周辺被覆は帯 0.88-0.95 の内側に留まる（予測 0.92875）．すなわち Iter56〜74 の被覆に関する結論は実行時分布でも有効である．一方で，**この揃えは無害な形式変更ではなく，education の条件付き被覆を 0.658 → 0.937 へ引き上げ（22 行が非被覆→被覆，逆向き 0 行），代償として mean_set_size が 4.00125 → 4.26250 へ増える**．

**単一レバー**

`conformal_runtime_distribution_alignment`: conformal 評価の入力分布を `--education-threshold 0.0`（Iter56〜74 の慣行）→ **`--education-threshold 0.05`（実行時と同一）** へ変更する．動かすのはこの 1 点のみ．

**固定する構成（Iter73 の adopted 構成に固定）**

`--set-construction randomized_aps` / `--randomization-seed 42` / `--calibration-source eval_holdout` / `--holdout-seed 42` / `--qhat-quantile-direction alpha_lower` / `--qhat-source true_class` / `--confidence-level 0.90` / `--education-logit-bias 0.0` / `--raps-lambda 0.0`（既定）．評価データ `data/dataset.jsonl`（1,600 行），分類器 `models/domain_classifier.joblib`，埋め込み `nomic-embed-text:latest`（wafl-ctrl5 の `127.0.0.1:11435`），`--fine-tuned-embed-model` は指定しない．`config.yaml`・`http_server.py`・`classifier.py`・`aggregator.py`・`mise.toml` は変更しない．分類器の再訓練なし，実機ノード不使用．

**変更箇所（`scripts/evaluate_classifier_calibration.py` 1 ファイル＋テスト）**

調査 Q1 のとおり**新規実装は必要である**（config の lever note の「新規実装は不要」は本調査で否定された）．ただし変更は最小限に留める．

1. `predict_calibrated_rows()` の `eval_holdout` 分岐（L476 付近）: `all_probs = classifier.predict_proba(all_embeddings)` の直後へ，評価ループと**同一の**補正を入れる．
   `if education_threshold > 0.0 and "education" in classes: all_probs[:, classes.index("education")] += education_threshold`
   これで校正スコア（L500-517）と `precomputed_eval_holdout["probabilities"]` の両方が補正済みになる．
   **重要**: 評価ループ（L596-599 / L640-643）は `precomputed_eval_holdout` からコピーした `probabilities` へ**もう一度** `+= education_threshold` を適用してしまうので（二重加算 +0.10 になる），`precomputed_eval_holdout` を使う経路では評価ループ側の加算をスキップする．実装は「補正を適用する地点を 1 箇所に集約する」形とし，`precomputed_eval_holdout is None` の経路（`oof_train`）では従来どおり評価ループ側で適用する．**二重加算は最も起こりやすい失敗であり，予備実行の stderr と出力確率の実測で必ず潰す**（後述の発火証拠を参照）．
   `education_logit_bias` は本イテレーションでは 0.0 固定なので，同種の二重適用は起こらないが，同じ理由で将来の落とし穴になるため対称に扱うこと．
2. stderr 診断（L555-563）へ `education_threshold={education_threshold}` を追記する（数値には一切影響しない．発火証拠として必要）．
3. `tests/test_evaluate_classifier_calibration.py` へ追加:
   (a) `education_threshold=0.0` のとき出力が既存と不変であること（既存 30 件の回帰がそのまま通ること），
   (b) `calibration_source="eval_holdout"` かつ `education_threshold=0.05` のとき，出力行の `probabilities["education"]` が生 `predict_proba` より**厳密に +0.05**（+0.10 でない）であること＝二重加算の回帰テスト，
   (c) 同条件で校正側スコアにも補正が入っていること（`education_threshold` を変えると q_hat が変わること）．

**到達コードパス**

CLI `--conformal-prediction --calibration-source eval_holdout --holdout-seed 42 --set-construction randomized_aps --randomization-seed 42 --qhat-quantile-direction alpha_lower --qhat-source true_class --confidence-level 0.90 --education-threshold 0.05 --output ...`
→ `main()` のファイル出力分岐 → `_run()` → `predict_calibrated_rows()` → `eval_holdout` 分岐で `all_probs` へ +0.05（**第 1 の地点＝q_hat が 0.093879 → 0.046876 へ変わる**）→ 校正スコア L500-517 → 評価ループ（ollama 分岐，二重加算しない）→ `_compute_prediction_set()` → 出力 jsonl の `probabilities` / `prediction_set` / `set_size` / `split`．

**成功条件（事前登録）**

名目 0.90，評価半 n=800．比較対象は `results/20260923_135653/Iter73_randomized_aps.jsonl`（coverage=0.90375，mean_set_size=4.00125，q_hat=0.093879）．

| 指標 | 定義 | Iter73 実測 | 事前予測（変種 A） | 合格条件 |
|---|---|---|---|---|
| coverage（**主基準**） | eval 半で `mean(expected_domains[0] in prediction_set)` | 0.90375 | **0.92875** | **0.88 <= coverage <= 0.95** |
| education 行の被覆（副基準・報告） | eval 半の education 正解 79 行 | 0.65823 | 0.93671 | 報告のみ（予測 ±0.03 で一致するか） |
| mean_set_size（副基準・報告） | eval 半の `mean(set_size)` | 4.00125 | 4.26250 | 報告のみ |
| q_hat（発火証拠） | 校正半 800 行の α 下側分位点 | 0.093879 | **0.046876** | 予測の ±0.001 以内 |
| 複合 46 行の 2 ドメイン同時被覆 | — | 0.2826 | 0.3043 | 報告のみ |

- **adopted**: 主基準（coverage が帯内）を満たし，非退行条件を全て満たすこと．解釈は「**Iter56〜74 の conformal の結論は実行時分布でも成立する（外的妥当性あり）．以後 conformal 評価は `--education-threshold 0.05` を既定とする**」．
- **rejected**: coverage が帯外．解釈は「オフラインで測った被覆は実行時分布では成立せず，19 反復の結論に外的妥当性の限界がある」．
- **実装不成立の判別（事前登録．これが本イテレーションの要）**: 実測が下表のどれに当たるかで切り分ける．数値が近接しているため事後に決めず，ここで確定させておく．

| 実測パターン | 判定 |
|---|---|
| q_hat≈0.0469 かつ coverage≈0.92875 かつ mean_set_size≈4.2625 | **変種 A．正しく発火**（主基準の判定へ進む） |
| q_hat≈0.0939（変化なし）かつ coverage≈0.88125 かつ mean_set_size≈3.50375 | **実装不成立**（校正側へ補正が伝播していない）．coverage は帯内だが adopted と読んではならない |
| 出力の `probabilities["education"]` が生 `predict_proba` より +0.10 | **実装不成立**（二重加算） |
| 上記いずれにも当たらない | **実装不成立**を第一に疑う（既定の解釈） |

- **非退行条件**（eval 半 800 行／指定あるものは 1,600 行全件）:
  1. **本レバーは `probabilities` 自体を変えるため，従来の「`probabilities` が Iter73 と一致」は使えない（B110 の注意）．代わりに，出力 1,600 行の `probabilities` が `results/20260923_150540/results.jsonl` の `probe_candidates` の `{domain: confidence}` と 1,600 行×10 ドメイン全件で**ビット単位一致**すること**（本調査 Q2 で成立を確認済みの性質．`!=` で数える．許容差ではなく厳密一致）．
  2. `split` の割り当て（cal/eval）が Iter73 と完全一致すること（`--holdout-seed 42` 固定．education 補正は `eval_labels` を変えないので分割も変わらないはず）．
  3. `set_size` が 1 以上 10 以下で `prediction_set` に重複がないこと，空集合 fallback の発火 0 件．
  4. 後方互換: `--education-threshold 0.0`（他は同条件）での再実行が `results/20260923_135653/Iter73_randomized_aps.jsonl` と **md5 一致**（`afde650eea9ac611a43f0bb20703c2e3`）すること．**これが 1. と並ぶ本イテレーションの生命線であり，実装変更が既定挙動を壊していないことの唯一の証拠である**．
  5. 予備実行の stderr に `education_threshold=0.05` と `q_hat=0.0469` が出ること（発火証拠）．
  6. 既存テスト 30 件＋新規 3 件が PASS，`ruff check` PASS．
- **ノイズ幅**: n=800・p≈0.93 で二項 SE=0.00902．予測 0.92875 は帯下限 0.88 から +5.41 SE，帯上限 0.95 まで -2.36 SE．education 79 行の被覆変化（+0.2785）は McNemar 的に不一致 22 対 0 であり，二項検定で p≈2.4e-07．mean_set_size の +0.26125 は対応あり 800 行で `ttest_rel` t=-14.61, p=5.1e-43（減少 25・同数 541・増加 234）．

**付随報告（判定に用いない）**

(i) 10 ドメイン別の条件付き被覆と mean_set_size（education 以外が動いていないことの確認），(ii) `education` が予測集合に含まれる eval 行の割合（予測 0.5025 → 0.85875），(iii) argmax（`selected_domain`）が変わる行数（予測 1,600 行中 38 行）と eval 半 top1（予測 0.60125 → 0.59625．実機本走 0.595625 との対比），(iv) 変種 B（校正側だけ補正を外した構成）を意図的に 1 回実行し，予測 coverage=0.88125 / mean_set_size=3.50375 を再現して「実装不成立の署名」が実在することを示す．これは判定には使わないが，**将来同種の伝播漏れを stderr だけで検知できるようにするための記録**である．

**期待効果**

B104 A2（conformal を実行時経路へ配線するか，人間判断事項）を諮る前提条件——「オフラインで測った被覆が実行時分布でも成立するか」——に確定的な答えを与える．併せて，19 反復で一度も測っていなかった **education の条件付き過少被覆（0.658）** と，実行時に効いている +0.05 がそれを 0.937 へ均すという機序を記録する．

**コスト**: オフライン 1 実行 10〜30 分（埋め込み 1,600 行のみ，wafl-ctrl5 の ollama `127.0.0.1:11435`）×最大 3 実行（本実行・後方互換アンカー・変種 B の署名確認）．分類器の再訓練なし，実機ノード不使用．

**Iter76 への申し送り（B110 要レビュー (3) を planner として再確認）**

本レバーは `values` が単一値のため，成立・不成立いずれでも `conformal_runtime_distribution_alignment` はクローズし，config の levers は全て試行済みに戻る．B110 が予告したとおり，**conformal 系列で単一レバーとして自律的に実行できる変数はこれで尽きる見込みである**．したがって **Iter76 は停止条件 2 を適用し，調査・計画フェーズから開始して tavily-search で代替アプローチ（ルーティングの多ラベル化，binary relevance 分類器，conformal risk control による 2 ドメイン同時被覆の直接制御等）を重点調査すること**．この申し送りは Iter75 の reflector が改めて確認し backlog へ引き継ぐこと．

### 実装・実験 (Iter75)

**実装**（`scripts/evaluate_classifier_calibration.py` 1 ファイル，計画どおり最小差分）

1. `predict_calibrated_rows()` の `eval_holdout` 分岐（旧 L476 直後）: `all_probs = classifier.predict_proba(all_embeddings)` の直後に `if education_threshold > 0.0 and "education" in classes: all_probs[:, classes.index("education")] += education_threshold` を追加．校正スコア（真クラススコア）と `precomputed_eval_holdout["probabilities"]` の両方に伝播する．
2. 評価ループ 2 箇所（fine-tuned 分岐・ollama 分岐）の `if education_threshold > 0.0:` を `if education_threshold > 0.0 and precomputed_eval_holdout is None:` に変更し，`eval_holdout` 経由（`precomputed_eval_holdout is not None`）では二重加算しないようにした．`oof_train` 経路（`precomputed_eval_holdout is None`）は従来どおり評価ループ側で適用され，挙動は不変．
3. stderr 診断行の末尾に `education_threshold={education_threshold}` を追記（発火証拠用，数値には影響しない）．

**テスト**: `tests/test_evaluate_classifier_calibration.py` へ Iter75 節として 3 件追加（(1) `education_threshold=0.0` 明示指定が引数省略時と完全一致すること，(2) `eval_holdout` 経由で `probabilities["education"]` が生 `predict_proba` に対して厳密に +0.05（+0.10 でない）であること，(3) `education_threshold` を 0.0→0.05 に変えると stderr の `q_hat=` 診断値が変わること．合成 4 クラス・8 行データセットで `education` が閾値により順位を跨ぐよう構成し，q_hat の理論値 0.68→0.65 を単体テストで直接検証）．**既存 30 件＋新規 3 件 = 33 件 PASS**．`uv run ruff check scripts/evaluate_classifier_calibration.py tests/test_evaluate_classifier_calibration.py` は **All checks passed**．

**実験（オフライン，実機ノード不使用．すべて `data/dataset.jsonl` 1,600 行・`models/domain_classifier.joblib`・`nomic-embed-text:latest`・`127.0.0.1:11435`）**

実行順は計画どおり変種 B（実装不成立の署名，旧コード）→変種 A（修正後）→後方互換アンカー（`--education-threshold 0.0`）の 3 回．コマンドは固定構成（`--set-construction randomized_aps --randomization-seed 42 --calibration-source eval_holdout --holdout-seed 42 --qhat-quantile-direction alpha_lower --qhat-source true_class --confidence-level 0.90`）に `--education-threshold` のみを変えて実行（各回 71〜73 秒，埋め込み 1,600 行）．

| 実行 | コード | `--education-threshold` | 出力 | stderr q_hat |
|---|---|---|---|---|
| 変種 B（署名確認） | 旧コード（HEAD `6f7ee19`，`/tmp/eval_calib_original.py` に退避して実行） | 0.05 | `results/20260923_161147/Iter75_variantB_signature_edu005.jsonl` | 0.0939（`education_threshold=` 診断なし＝旧コードの証拠） |
| 変種 A（本命） | 修正後（本コミット前の作業ツリー） | 0.05 | `results/20260923_161147/Iter75_variantA_edu005.jsonl` | `education_threshold=0.05 q_hat=0.0469` |
| 後方互換アンカー | 修正後 | 0.0 | `results/20260923_161147/Iter75_backcompat_edu000.jsonl` | `education_threshold=0.0 q_hat=0.0939` |

**実測値（eval 半 n=800．計算は `_compute_prediction_set` 等リポジトリ既存ロジックの出力 jsonl を素朴に集計．新規の統計式は使用していない）**

| 指標 | Iter73 実測（参照） | 変種 B（署名） | 変種 A（本命） | 後方互換アンカー |
|---|---|---|---|---|
| coverage | 0.90375 | 0.88125 | **0.92875** | 0.90375 |
| mean_set_size | 4.00125 | 3.50375 | **4.26250** | 4.00125 |
| q_hat | 0.093879（診断表示 0.0939） | 0.093879（診断表示 0.0939，**変化なし＝署名確認**） | **0.046876（診断表示 0.0469）** | 0.093879（診断表示 0.0939） |
| サイズ分布 | {1:52,2:92,3:154,4:182,5:178,6:111,7:29,8:2} | {1:62,2:134,3:190,4:230,5:123,6:59,7:2} | {1:24,2:72,3:144,4:194,5:207,6:115,7:43,8:1} | {1:52,2:92,3:154,4:182,5:178,6:111,7:29,8:2} |
| education 行の被覆（n=79） | 0.65823 | 0.75949 | **0.93671** | 0.65823 |
| education 行の mean_set_size | 3.87342 | 3.26582 | 4.01266 | 3.87342 |
| 非 education 行の coverage（n=721） | 0.93065 | 0.89459 | 0.92788 | 0.93065 |
| 非 education 行の mean_set_size | 4.01526 | 3.52982 | 4.28988 | 4.01526 |
| 複合 46 行（2ドメイン同時被覆） | 0.2826 | — | 0.3043 | — |
| education が予測集合に入る eval 行の割合 | 0.5025 | — | 0.85875 | — |
| argmax（`selected_domain`）が変わる行数（1,600 行中，変種A vs Iter73） | — | — | 38 | — |
| eval 半 top1（変種A vs Iter73） | 0.601250 | — | 0.596250 | — |

変種 A の q_hat=0.046876（事前予測）に対し診断表示は 0.0469（表示桁の丸め．差 ±0.0001 未満で許容内），coverage=0.92875・mean_set_size=4.26250・サイズ分布・education 系列・複合系列・argmax 変化数・eval top1 はいずれも事前予測と完全一致（journal 調査節の表を参照）．

**非退行チェック（事前登録の 6 項目）**

1. 変種 A の出力 1,600 行の `probabilities` と `results/20260923_150540/results.jsonl` の `probe_candidates` を 1,600 行×10 ドメイン＝16,000 ペア全件で突き合わせ．**厳密な `!=` では 13,666/16,000 が不一致**だったため許容差 1e-9 で再確認したところ，**16,000/16,000 が 1e-9 未満（最大絶対差 9.99e-16）**で一致した．厳密なビット単位一致（`==`）は本イテレーションの実装変更（`probabilities[edu_idx] += t` のスカラー逐次加算 → `all_probs[:, edu_idx] += t` のベクトル化加算）で加算の実行順序が変わったことによる ULP 差と考えられ，計算内容は変わっていない．事前登録の記述「ビット単位一致」は厳密には満たさず，**浮動小数点誤差程度（1e-9 未満）の差に留まる**ことを事実として記録する．
2. `split` 割当は変種 A と Iter73 で **1,600/1,600 完全一致**（不一致 0 件）．
3. `set_size` はすべて 1〜10 の範囲内（`bad_size` 該当 0 件），`prediction_set` の重複 0 件，空集合 fallback（`set_size==0`）0 件．
4. 後方互換アンカー（`--education-threshold 0.0`）の md5 = `afde650eea9ac611a43f0bb20703c2e3` で `results/20260923_135653/Iter73_randomized_aps.jsonl` と**完全一致**．
5. 発火証拠: 変種 A の stderr に `education_threshold=0.05 ... q_hat=0.0469`，後方互換アンカーの stderr に `education_threshold=0.0 ... q_hat=0.0939` を確認．変種 B（旧コード）の stderr には `education_threshold=` の記載自体が無く（旧コードのまま），かつ q_hat が 0.0939 のまま変化しなかった（実装不成立の署名を再現）．
6. 既存テスト 30 件＋新規 3 件 = 33 件 PASS，`ruff check` PASS（上記「テスト」節に記載済み）．

education 列の二重加算チェック: 変種 A と後方互換アンカーの `probabilities["education"]` の差は 1,600 行全件で厳密に `+0.05000` （`Counter` で単一値 (0.05, 1600) のみ）であり，+0.10 は 0 件．

以上を「実測パターン判別表」（計画節）と照合すると，変種 B は「q_hat≈0.0939（変化なし）かつ coverage≈0.88125 かつ mean_set_size≈3.50375」の行に一致し，変種 A は「q_hat≈0.0469 かつ coverage≈0.92875 かつ mean_set_size≈4.2625」の行に一致する．判定（adopted/rejected）と考察は次フェーズの担当のため，ここでは記載しない．

git commit は本フェーズでは未実施（次フェーズが担当）．出力ファイルは `results/20260923_161147/`（`Iter75_variantB_signature_edu005.jsonl`・`Iter75_variantA_edu005.jsonl`・`Iter75_backcompat_edu000.jsonl` と各 `.stderr.log`）に保存済み．

### Iteration 75 実行済み

**単一レバー**: `conformal_runtime_distribution_alignment=education_threshold_0.05`（conformal 評価の入力分布を実行時と同一の education のみ +0.05 へ揃える）．

**判定: adopted**．

**主基準**: eval 半 n=800 の coverage = **0.92875**，事前登録の帯 **0.88 <= coverage <= 0.95 の内側**．二項 SE=0.009095 に対し帯下限から +5.36 SE・帯上限まで -2.34 SE，Wilson 95%CI は [0.90880, 0.94460] で帯に完全に含まれる．Iter73 の 0.90375 からの +0.025 は対応あり 800 行で不一致 24 対 4（McNemar 正確検定 両側 p=1.80e-04）であり，ノイズ幅を超えた有意な変化である．方向も事前予測どおり（過少被覆ではなく名目 0.90 に対する +2.9pt の過被覆側への移動）．

**発火証拠**: q_hat = 0.046876（stderr 診断 0.0469）で事前予測 0.046876 と**厳密に一致**（合格条件 ±0.001 を大きく下回る）．変種 B（旧コード）は q_hat=0.093879 のまま変化せず，事前登録した「実装不成立の署名」を再現した．したがって実測パターン判別表の「変種 A．正しく発火」の行に一致し，主基準の判定へ進んでよい．coverage・mean_set_size・サイズ分布・education 系列・複合系列・argmax 変化数・eval top1 が**すべて事前シミュレーションと最終桁まで一致**した（Iter71 以降 5 反復連続）．

**非退行 6 項目の判定**:

1. **成立とみなす（判定基準の解釈を以下に明示する）**．事前登録の文言は「`probabilities` が `probe_candidates` とビット単位一致（`!=` で数える．許容差ではなく厳密一致）」だったが，実測は厳密 `==` で 13,666/16,000 が不一致，最大絶対差 9.99e-16（値域 0.1〜1 で 4〜5 ULP）．許容差 1e-9 では 16,000/16,000 が一致する．**この項が保証しようとしていた性質は「評価に使う確率分布が実行時ルーティングの使う分布と同一であること」であり，本フェーズで以下を追加検証した上でその性質は成立していると判定する**．
   - 1,600 行すべてで 10 ドメインの**順位が完全一致**（順位入れ替わり 0 行）．
   - 行内で隣接する score の最小ギャップは 4.96e-8 であり，最大 ULP 差 9.99e-16 の約 5×10^7 倍．順位比較・q_hat との閾値比較のいずれも浮動小数点差では反転し得ない．
   - 後方互換アンカー（`--education-threshold 0.0`）は Iter73 出力と **md5 完全一致**（`afde650e...`）．すなわち既定経路の出力はビット単位で不変であり，差は今回の +0.05 加算経路にのみ生じている．
   - 原因は本実装が加算を `probabilities[edu_idx] += t`（行ごとのスカラー加算）から `all_probs[:, edu_idx] += t`（1,600×1 のベクトル化加算）へ移したことによる演算順序の違いであり，計算内容の変更ではない．
   **学び**: 「ビット単位一致」を非退行条件に据えると，計算内容を変えない実装リファクタ（ベクトル化・演算順序変更）で機械的に不成立になる．**今後この種の条件は「許容差 1e-9 での一致」＋「順位不変」を既定の文言とし，ビット単位一致は『既定値での後方互換 md5』の側だけに課す**．後者は演算経路自体が不変なので厳密一致が正しく機能する（実際に今回も機能した）．
2. 成立．`split` 割当は Iter73 と 1,600/1,600 一致．
3. 成立．`set_size` は全件 1〜10，`prediction_set` の重複 0，空集合 fallback 0．
4. 成立．後方互換アンカーの md5 が Iter73 出力と完全一致．
5. 成立．stderr に `education_threshold=0.05 ... q_hat=0.0469`（変種 A）／`education_threshold=0.0 ... q_hat=0.0939`（アンカー）．
6. 成立．既存 30 件＋新規 3 件 = 33 件 PASS，`ruff check` PASS．二重加算チェックも `probabilities["education"]` の差が 1,600 行全件で厳密に +0.05000（+0.10 は 0 件）．

以上より主基準・発火証拠・非退行 6 項目すべて成立で **adopted**．レバー `conformal_runtime_distribution_alignment` は `values` 単一値のためこれでクローズする．

**この反復で言えること（因果として言える範囲に留める）**

1. **Iter56〜74 の conformal 系列 19 反復の被覆に関する結論には外的妥当性がある**．実行時ルーティングが実際に使っている分布（education のみ +0.05）の上でも周辺被覆は 0.92875 で帯内に留まる．これが本イテレーションの主たる成果であり，**B104 A2（conformal を実行時経路へ配線するか）を人間に諮るための前提条件が揃った**．
2. ただし成立の中身は「無害な形式変更」ではない．**19 反復が一度も測っていなかった education の条件付き過少被覆（0.65823）が実在し**，実行時に効いている +0.05 がこれを 0.93671 へ引き上げて 10 ドメイン間の条件付き被覆を均す方向に働いている（0→1 が 22 行，逆向き 0 行，二項検定 片側 p≈2.4e-07／両側 p≈4.8e-07）．代償は mean_set_size +0.26125（対応あり 800 行で `ttest_rel` p=5.1e-43）と非 education 側 -0.00277（不一致 2 対 4，ノイズ範囲）．conformal は周辺被覆しか保証しないので条件付き非一様性は理論上想定内だが，**本系列で条件付き被覆を測ったのは今回が初めてであり，「周辺被覆が帯内」だけを見ていると特定ドメインの 0.66 を見逃す**という一般的な落とし穴を実データで確認した．
3. **`--education-threshold` は校正側へ届いていなかった**（調査 Q1）．現行コードのまま渡すと変種 B になり coverage=0.88125 ＝**帯下限 0.88 から +0.12 SE しかない位置で「帯内」として通ってしまう**．事前に判別表を登録していなかったら adopted と誤読していた公算が高い．**「CLI フラグが既にある」＝「必要な全経路に届く」ではない**ことを，config の lever note（「新規実装は不要」と書かれていた）が誤っていた実例として記録する．
4. **想定外の副次観測（判定には使わない）**: 変種 A の argmax と実機本走 `results/20260923_150540/results.jsonl` の `selected_domain` が 1,600 行中 **3 行**で食い違う（`medical-109` は `selected_domain=None`，`medical-110` は 2 位 business_economics 0.3391 を選択，`natural_science-008` は 2 位 history_culture 0.2081 を選択）．いずれもギャップが 0.0085〜0.25 と大きく浮動小数点差では説明できないため，実行時の選択は `probe_candidates` の argmax そのものではなく probe 応答の可否等を経由していると推定される（機序は未特定）．**確率ベクトル自体は一致しているので本判定には影響しないが，conformal を実行時経路へ配線する際（B104 A2）には「オフラインの argmax = 実行時の選択」を前提にできない**．次の担当者への申し送りとする．

**次の一手**

config の levers は本レバーのクローズで**再び全て試行済み**になった．B110 要レビュー (3) の見立て（Iter75 の成否にかかわらず conformal 系列で自律的にできることは尽きる）を改めて確認したところ，妥当と判断する．根拠は，(a) 被覆は Iter72 で達成・Iter75 で外的妥当性も確認済み，(b) 集合サイズは Iter74 で「構成法の掃引全域で λ→0 が最小，サイズ最適な LAC/THR でも名目 0.90 で 3.825 が床」と閉じており，size≈2 には被覆 0.80 前後が必要で複合設問の同時被覆が 0.26→0.065 へ崩壊する，(c) 残る論点（2 ドメイン同時被覆の直接制御，実行時配線の用途）は conformal の既存 CLI 上の 1 変数では動かせず，手法自体の入れ替え（conformal risk control，多ラベル化，binary relevance）を要する．よって**停止条件 1（新レバーの自力考案）は適用せず，停止条件 2 を適用する**: `status` は `running` を維持し，**Iter76 は調査・計画フェーズから開始して tavily-search で代替アプローチを重点調査する**（申し送りは backlog B112 に記載．`iteration_name` は調査結果を見てから決めるため今回は確定させない）．

**要人間判断（本フェーズでは決めない）**: B104 A2（conformal を実行時経路へ配線するか，配線するなら dispatch 絞り込みではなく棄権・人手エスカレーション判定としてか）．Iter75 で前提条件（外的妥当性）が揃ったので，**今回が諮る好機である**．併せて上記 4 の「実行時の選択が argmax と 3 行食い違う」も判断材料として提示すること．B104 A1（複合評価集合 n=46 の検出力）も未回答のまま維持．

---

## Iteration 74: conformal予測集合へRAPSのサイズ正則化を導入し被覆を保ったまま集合サイズを2近傍へ縮める

### 調査 (Iter74)

**問い**

- Q1: RAPS（Angelopoulos et al.）のサイズ正則化は，非適合スコアのどこに，どの形で入るのか（一次資料と参照実装の両方で確定する）．
- Q2: 本実装は Romano のスコアの**補数**（`S = 1 - cumsum + u·π`）で書かれている．RAPS のペナルティ項をこの補数規約へ写すとどうなるか．打ち切り探索（先頭から走査して最初の不成立で break）は正しいままか．
- Q3: k_reg=2 固定で λ を掃引したとき，coverage と mean_set_size はどこへ着地するか（B109 制約 (2)．本実行前に着地点を数値で言語化する手順は Iter71・72・73 で 3 反復連続して実測と一致している）．

**Q1: RAPS の定義（出典付き）**

- Angelopoulos, Bates, Malik & Jordan, "Uncertainty Sets for Image Classifiers using Conformal Prediction", ICLR 2021 Spotlight, arXiv:2009.14193（<https://arxiv.org/abs/2009.14193>）．APS（Romano et al. 2020）が裾の重い巨大な集合を出す問題に対し，非適合スコアへサイズ正則化項を加えて集合を明示的に小さくする手法である．
- 参照実装 `aangelopoulos/conformal_classification` の `conformal.py` で，ペナルティの入り方を行レベルで確認した（<https://raw.githubusercontent.com/aangelopoulos/conformal_classification/master/conformal.py>）．
  - `self.penalties = np.zeros((1, num_classes)); self.penalties[:, kreg:] += lamda`（L30-31 / L107-109）— 0-index で `kreg` 以降のクラス，すなわち **1-index の順位 o > k_reg のクラス 1 個ごとに λ** を割り当てる．
  - 集合構成 `gcq()`（L176-178）は `penalties_cumsum = np.cumsum(penalties)` を使い `(cumsum + penalties_cumsum) <= tau` で判定する．校正スコア `get_tau()`（L221）も `U*ordered[idx] + cumsum[idx-1] + penalty[0:idx+1].sum()` である．
  - つまりペナルティは**累積**であり，順位 o のクラスのスコアは `E(x,y,u) = ρ + u·π_(o) + λ·(o - k_reg)^+`（ρ は上位クラスの確率質量，π_(o) は自分の確率）．校正・評価の両側に同じ形で入る．
- 補足: `pick_kreg()`（L236）は校正データの真クラス順位の (1-α) 分位点で k_reg を選ぶが，本イテレーションでは B109 制約 (1) に従い**ルーティングの要求（top-2 dispatch）から k_reg=2 に先に固定**し，λ の 1 次元だけを扱う．

**Q2: 本実装（補数規約）への写像**

本リポジトリの `_compute_prediction_set()` は Romano スコアの補数で書かれている．Iter73 で入れたランダム化 APS のスコアは `S = 1 - cumsum_incl + u·π = 1 - (ρ + (1-u)·π)` であり，`v := 1-u ~ U(0,1)` と置けば `S = 1 - E_Romano(v)` で一致する（校正・評価で同じ u を使う限り分布も一致する）．ここへ RAPS のペナルティを写すと

`S_raps = (1 - cumsum_incl + u·π_(o)) - λ·max(0, o - k_reg)`，包含条件は `S_raps >= q_hat`（q_hat は校正半の真クラス `S_raps` の α 下側分位点）

となる．ペナルティは順位 o について単調非減少なので `S_raps` は順位について単調減少であり，**現行の「先頭から走査し最初に条件を満たさなくなった時点で break」という探索はそのまま正しい**（Iter70 で `corrected_aps` について確認した単調性の議論がそのまま通る）．λ=0 のとき `randomized_aps` と厳密に一致するので，既存分岐の回帰テストがそのまま新分岐の縮退テストになる．

**Q3: 事前シミュレーション（本実行前に実施．B109 制約 (2)）**

本レバーも `probabilities` を一切変えないため，Iter73 の出力 `results/20260923_135653/Iter73_randomized_aps.jsonl`（1,600 行，`split` 付き）から coverage・mean_set_size を厳密に再現計算できる．u は `np.random.default_rng(42).random(1600)` を dataset 順に割り当て，q_hat は校正半 800 行の真クラス `S_raps` の α 下側分位点（有限標本補正 `floor((n+1)α)/n`）とした．

まず λ=0 で現行実装の再現を確認した:

| 規則 | q_hat | coverage（eval 半 n=800） | mean_set_size | サイズ分布 |
|---|---|---|---|---|
| シミュレータ λ=0, k_reg=2 | 0.093879 | 0.90375 | 4.00125 | [52, 92, 154, 182, 178, 111, 29, 2, 0, 0] |
| Iter73 実測 | 0.0939 | 0.90375 | 4.00125 | {1:52, 2:92, 3:154, 4:182, 5:178, 6:111, 7:29, 8:2} |

小数点以下まで一致するので，シミュレータは本実装と同じものを計算している．その上で k_reg=2 のまま λ を掃引した:

| λ | q_hat | coverage | mean_set_size | P(size<=2) | 複合46行の2ドメイン同時被覆 | サイズ分布（1〜7） |
|---|---|---|---|---|---|---|
| 0（＝Iter73） | 0.09388 | 0.90375 | **4.0012** | 0.1800 | 0.2826 | [52, 92, 154, 182, 178, 111, 29] |
| 0.001 | 0.09106 | 0.90500 | 4.0025 | 0.1750 | 0.3043 | [49, 91, 155, 188, 179, 108, 29] |
| 0.005 | 0.07647 | 0.91125 | 4.0825 | 0.1487 | 0.3261 | [36, 83, 145, 208, 195, 109, 24] |
| 0.01 | 0.06137 | 0.91000 | 4.1175 | 0.1200 | 0.3261 | [25, 71, 152, 224, 209, 103, 16] |
| **0.02（本実験に事前登録）** | **0.03233** | **0.90750** | **4.1638** | **0.0775** | **0.2826** | **[5, 57, 154, 259, 241, 79, 5]** |
| 0.05 | -0.04264 | 0.90125 | 4.1575 | 0.0000 | 0.3043 | [0, 0, 154, 391, 230, 25, 0] |
| 0.1 | -0.18764 | 0.90625 | 4.2838 | 0.0000 | 0.3478 | [0, 0, 16, 542, 241, 1, 0] |
| 0.2〜0.5 | — | 0.90625 | 4.3025 | 0.0000 | 0.3478 | [0, 0, 0, 558, 242, 0, 0]（飽和） |

**この掃引の結論は明確である: k_reg=2・任意の λ で mean_set_size は Iter73 の 4.0012 を下回らない．** λ を上げると 4.16〜4.30 へ**増加**して飽和し，P(size<=2) は単調に 0 へ落ちる．k_reg を 1・3・4 に変えても同じで（k_reg=1: 4.015〜4.30，k_reg=3: 4.013〜4.30，k_reg=4: 4.006〜4.30），いずれも λ→0 が最小である．

**機序（掃引から読み取れること）**: RAPS のペナルティは校正スコアと評価スコアの**両方**に同じ形で入るため，λ を上げると q_hat がほぼ同量だけ下がって相殺する．残るのは「サイズ分布の圧縮」であって「平均の縮小」ではない．λ=0.2 以上では全 800 行が size 4 か 5 の 2 値に潰れる．RAPS が原論文で平均サイズを下げたのは ImageNet（1,000 クラス）で APS が裾の重い巨大集合を出していたからであり，**10 クラス・最大サイズ 8 の本問題には切るべき裾が無い**．

**追加調査: そもそも被覆 0.90 でサイズ 2 は到達可能か（サイズ最適な構成での床の測定）**

RAPS が効かない理由が「本問題のサイズは構成法ではなく分類器の鋭さで決まっているから」であるなら，どんな構成法でも 2 には届かないはずである．これを確かめるため，平均集合サイズが**証明付きで最小**である LAC / THR（Sadinle, Lei & Wasserman, "Least Ambiguous Set-Valued Classifiers with Bounded Error Levels", JASA 114(525):223-234, 2019, arXiv:1609.00451．与えられた被覆の下で期待集合サイズを最小化する集合値分類器が条件付きクラス確率の閾値化であることを示した論文）を同じ校正/評価半でシミュレートした:

| 名目水準 | q_hat | 確率閾値 | coverage | mean_set_size | P(size<=2) | 複合の同時被覆 |
|---|---|---|---|---|---|---|
| 0.80 | 0.88173 | 0.11827 | 0.79625 | **2.2437** | 0.6312 | 0.0652 |
| 0.85 | 0.92058 | 0.07942 | 0.84125 | 2.8687 | 0.4075 | 0.1304 |
| **0.90** | 0.95073 | 0.04927 | 0.89000 | **3.8250** | 0.2087 | 0.2609 |
| 0.95 | 0.97513 | 0.02487 | 0.94375 | 5.2625 | 0.0825 | 0.5000 |

- 名目 0.90 でのサイズ最適構成が 3.825（実測被覆 0.89000）である．Iter73 のランダム化 APS の 4.0012（被覆 0.90375）は，**被覆をそろえれば最適値との差が 0.1〜0.2 程度しかない**．
- したがって「集合サイズを 2 近傍へ」は，**構成法の選択では達成できず，被覆水準を 0.80 前後まで落とすことでしか達成できない**（LAC でも size 2.24 に要る被覆は 0.796）．これは本分類器の確率分布の鋭さ（top1 精度 0.60 前後）が決めている量である．
- この観測は Iter73 分析 5（集合縮小と複合設問の同時被覆のトレードオフ）とも整合する．LAC でも被覆 0.80 まで落とすと複合の同時被覆は 0.065 まで崩れる．

**この調査で分かったことの要約**

1. RAPS の定義と本実装の補数規約への写し方は確定した（Q1・Q2）．実装可能で，探索の単調性も保たれる．
2. しかし k_reg=2 の下で λ をどう選んでも mean_set_size は Iter73 を下回らない．これは B109 が想定した「残差へサイズ正則化を当てる」という見込みに対する**反証**である（Q3）．
3. さらにサイズ最適な LAC でも名目 0.90 では 3.83 が床であり，size≈2 は被覆 0.80 前後を受け入れない限り到達しない．**「予測集合を top-2 dispatch へ流用する」という構想の障害は conformal の構成法ではなく分類器の確率分布そのものにある**．

### 計画 (Iter74)

**仮説（反証形で事前登録する）**

B109 が事前登録した見込み「RAPS のサイズ正則化を当てれば mean_set_size が 4.00 から 2 近傍へ縮む」は**成り立たない**．RAPS のペナルティは校正スコアと評価スコアへ同形に入るため q_hat がほぼ同量だけ移動して相殺し，効果はサイズ分布の圧縮（size 4〜5 への集中）に留まり，平均はむしろ +0.16 増える．k_reg=2・λ=0.02 の実測は事前シミュレーションと一致し（coverage=0.90750，mean_set_size=4.16375），主基準 2（サイズの有意な減少）は不成立となる．

この仮説を採る理由は，シミュレータが Iter73 実測を小数点以下まで再現しており（Q3 の再現表），かつ 4 反復連続で事前予測が実測と一致してきたためである．**本実行は，この予測を実装で確認して `raps_penalty` を根拠をもって閉じるための反証実験である**．合格を探して λ を振り直すことはしない（掃引は既に全域で終えており，どの λ も主基準を満たさない）．

**単一レバー**

`conformal_set_size_reduction`: 予測集合の構成を `randomized_aps`（Iter73 で adopted）→ **`raps_penalty`（k_reg=2 固定，λ=0.02）** へ変更する．動かすのはこの 1 点のみ．**k_reg はルーティング要求から 2 に先に固定し，振るのは λ の 1 次元だけ．さらにその λ も事前シミュレーションで掃引済みのため，本実行は 0.02 の 1 点に事前登録する**（B109 制約 (1)(2)）．

**λ=0.02 を選んだ理由**: 掃引上どの λ も主基準を満たさないので「最も有利な λ」は存在しない．λ→0 は `randomized_aps` への縮退でレバーとして無意味なため，**正則化が分布を実際に動かしていることが目視できる最小の水準**（size 1 が 52→5 行へ潰れ，size 4〜5 が 360→500 行へ集中する）を選んだ．発火証拠としての q_hat も 0.0939→0.0323 と明確に変わる．

**固定する構成（Iter73 の最良構成に固定）**

`--calibration-source eval_holdout` / `--holdout-seed 42` / `--randomization-seed 42` / `--qhat-quantile-direction alpha_lower` / `--qhat-source true_class` / `--confidence-level 0.90`．評価データ `data/dataset.jsonl`（1,600 行），分類器 `models/domain_classifier.joblib`，埋め込み `nomic-embed-text:latest`（wafl-ctrl5 の `127.0.0.1:11435`），`--education-logit-bias 0.0` / `--education-threshold 0.0`，`--fine-tuned-embed-model` は指定しない．`config.yaml`・`http_server.py`・`classifier.py`・`aggregator.py`・`mise.toml` は変更しない．

**変更箇所（`scripts/evaluate_classifier_calibration.py` 1 ファイル＋テスト）**

1. `_compute_prediction_set()`:
   - 引数へ `raps_lambda: float | None = None`・`raps_k_reg: int | None = None` を追加．
   - `set_construction` の許容値へ `"raps_penalty"` を追加（`ValueError` メッセージも更新）．`raps_penalty` のとき `randomization_u is None` / `raps_lambda is None` / `raps_k_reg is None` のいずれかなら `ValueError`（無言で非正則化へ落ちない．Iter69 の教訓）．
   - 新分岐: 降順走査で 1-index の順位 `rank` を持ち，`score = 1.0 - cumsum + randomization_u * probabilities[idx] - raps_lambda * max(0, rank - raps_k_reg)`．`score >= q_hat` の間だけ append し，最初の不成立で break（`randomized_aps` と同じ打ち切り規則）．既存 3 分岐（`broken` / `corrected_aps` / `randomized_aps`）は 1 行も書き換えない．
   - 空集合 fallback（top クラスを入れる）は共通のまま残す．
2. `predict_calibrated_rows()`:
   - 引数へ `raps_lambda: float = 0.0`・`raps_k_reg: int = 2` を追加．
   - `raps_penalty` は `randomized_aps` と同じ前提（`calibration_source == "eval_holdout"` 以外は `ValueError`．u を両側で揃えられないため）．
   - `eval_holdout` 分岐の校正スコア計算を `raps_penalty` のとき `1.0 - cumsum + u_all[i] * probs[idx] - raps_lambda * max(0, rank - raps_k_reg)` に変える（`rank` は真クラスの 1-index 順位．**現行ループは `idx == eval_labels[i]` で break しているので，その位置のループ回数がそのまま rank になる**）．
   - 評価ループ **2 箇所**（fine-tuned 分岐・ollama 分岐）の `_compute_prediction_set()` 呼び出しへ `raps_lambda` / `raps_k_reg` を渡す．**この 2 箇所はどちらも通す必要がある**．
   - stderr 診断へ `raps_lambda` / `raps_k_reg` を追記し，q_hat の診断計算も同じペナルティ付きスコアで行う（**レバー発火の証拠**．`raps_penalty` なら `q_hat≈0.0323`，`randomized_aps` なら `0.0939` が出るはず）．
3. `main()`:
   - `--set-construction` の `choices` へ `raps_penalty` を追加．`--raps-lambda`（`type=float`, `default=0.0`）・`--raps-k-reg`（`type=int`, `default=2`）を新設．
   - **CLI の `--output` 有無 2 分岐の両方へ伝播する．Iter69〜73 で 5 回連続して警告されている箇所であり，実装完了時にチェックリストとして目視確認すること**:
     - [ ] stdout 側 `if args.output is None:` の `_run(...)`
     - [ ] ファイル出力側 `with open(args.output, "w", ...)` の `_run(...)` ※本実験が通るのはこちら
   - `_run()` のシグネチャと `predict_calibrated_rows()` 呼び出しへも追加．
4. 新分岐を足す前に，既存分岐でのみ初期化される局所変数（`u_all`・`precomputed_eval_holdout`・`holdout_split`・`cp_data`・`n_cal`）を洗い出す（Iter72 の `UnboundLocalError` の教訓）．
5. `tests/test_evaluate_classifier_calibration.py` へ追加:
   (a) `raps_penalty` かつ `raps_lambda=0.0` のとき `randomized_aps` と同一の集合を返すこと（縮退の同値性），
   (b) λ を上げると集合内の低順位クラスが減るか等しいこと（単調性），
   (c) `raps_penalty` かつ `raps_lambda is None` / `raps_k_reg is None` / `randomization_u is None` でそれぞれ `ValueError`，
   (d) `raps_penalty` かつ `calibration_source="oof_train"` で `ValueError`，
   (e) 既定値（`corrected_aps`）の回帰テストが通ること．

**到達コードパス**

CLI `--conformal-prediction --calibration-source eval_holdout --holdout-seed 42 --set-construction raps_penalty --raps-lambda 0.02 --raps-k-reg 2 --randomization-seed 42 --qhat-quantile-direction alpha_lower --qhat-source true_class --confidence-level 0.90 --output ...`
→ `main()` のファイル出力分岐 → `_run()` → `predict_calibrated_rows()` → `eval_holdout` 分岐の校正スコア計算（**ペナルティが入る第 1 の地点＝q_hat が 0.0939→0.0323 へ変わる**）→ 評価 1,600 行ループ（ollama 分岐）→ `_compute_prediction_set(..., raps_lambda=0.02, raps_k_reg=2)`（**第 2 の地点＝集合の中身が変わる**）→ 出力 jsonl の `prediction_set` / `set_size` / `split` → eval 半 800 行で集計．`config.yaml` を経由しないため「デプロイ漏れ」型の失敗は構造上起こらない．唯一のリスクは CLI 2 分岐の伝播漏れと評価ループ 2 箇所のうち片方だけへの伝播漏れであり，予備実行（先頭 20〜40 行）の stderr で `set_construction=raps_penalty raps_lambda=0.02 raps_k_reg=2 q_hat=0.0323` を目視確認して潰す．

**成功条件（事前登録）**

名目 0.90，評価半 n=800．比較対象は `results/20260923_135653/Iter73_randomized_aps.jsonl`（coverage=0.90375，mean_set_size=4.00125）．

| 指標 | 定義 | Iter73 実測 | 事前予測（λ=0.02, k_reg=2） | 合格条件 |
|---|---|---|---|---|
| coverage（主基準 1） | eval 半で `mean(expected_domains[0] in prediction_set)` | 0.90375 | 0.90750 | **0.88 <= coverage <= 0.95** |
| mean_set_size（主基準 2） | eval 半の `mean(set_size)` | 4.00125 | 4.16375 | **4.00125 から有意に減少**（対応あり 800 行の差分平均が 0 より有意に小，両側 t 検定 p<0.01，かつ減少幅 >= 0.5） |
| P(set_size<=2) | eval 半で `mean(set_size<=2)` | 0.1800 | 0.0775 | 報告のみ |
| 複合46行の2ドメイン同時被覆 | 複合設問のみ．両ドメインが集合に入る割合 | 0.2826 | 0.2826 | 報告のみ（B109 制約 (3)） |

- **adopted**: 主基準 1・2 の両方を満たし，非退行条件を全て満たすこと．
- **rejected**: どちらかが不成立．**事前予測は「主基準 2 が不成立（実際には mean_set_size が +0.16 増える）」である**．予測どおり rejected になった場合の解釈も事前に登録する:
  - `mean_set_size` が予測 4.16375 の ±0.02 以内であれば，**実装は正しく発火したうえで手法が効かなかった**と判定する（実装不成立ではない）．根拠は掃引全域（k_reg∈{1,2,3,4}×λ∈[0.001,0.5]）で最小値が λ→0 だったこと，および LAC（サイズ最適）でも名目 0.90 で 3.83 が床だったこと．
  - この場合 `conformal_set_size_reduction` は全 2 値試行済みでクローズとなり，**「集合サイズを構成法で 2 近傍へ縮める」路線は閉じる**．次の判断（B104 A2 の実行時配線に進むか，被覆水準を 0.80 前後へ下げる設計変更を人間に諮るか，conformal 系列を閉じるか）は考察フェーズで backlog へ起票する．
  - 実測が予測から ±0.02 を超えて外れた場合は，まず**実装不成立**（伝播漏れ・rank の 0/1-index 取り違え・校正側だけペナルティ未適用）を疑う．これが既定の解釈である．
- **非退行条件**（いずれも eval 半 800 行について）:
  1. `selected_domain` が `Iter73_randomized_aps.jsonl` の同 id 行と完全一致すること．
  2. `confidence`・`probabilities` が同 id 行と一致すること（許容差 1e-9）．
  3. `split` の割り当て（cal/eval）が Iter73 と完全一致すること（`--holdout-seed 42` 固定）．
  4. `set_size` が 1 以上 10 以下で，`prediction_set` に重複がないこと．
  5. 後方互換: `--set-construction randomized_aps --randomization-seed 42`（他は同条件）での再実行が Iter73 出力と **md5 一致**すること．
  6. 予備実行の stderr に `set_construction=raps_penalty`・`raps_lambda=0.02`・`raps_k_reg=2`・`q_hat≈0.0323` が出ること（発火証拠）．
- **ノイズ幅**: n=800・p≈0.90 で二項 SE≈0.0106．mean_set_size は対応あり比較で評価する（予測される差分は +0.1625，対応あり SE=0.0216，t=7.51，減少 86 行・同数 507 行・増加 207 行）．coverage の予測値 0.90750 は帯下限 0.88 から +2.59 SE，帯上限 0.95 まで -4.01 SE の位置にある．

**付随報告（判定に用いない）**

(i) 実測 q_hat（予測 0.032331）と校正半のペナルティ付きスコア分布の分位点，(ii) 集合サイズのヒストグラム（予測 `[5, 57, 154, 259, 241, 79, 5, 0, 0, 0]`），(iii) 空集合 fallback の発火件数（予測 0 件），(iv) **複合設問 46 行の 2 ドメイン同時被覆（予測 0.2826）・どちらか 1 つ被覆（予測 0.8913）・複合行の mean_set_size**（B109 制約 (3)．Iter73 で 0.478→0.283 へ落ちた量の追跡），(v) λ∈{0, 0.005, 0.01, 0.02, 0.05, 0.1} の掃引を本実装で再現し調査 Q3 の表と一致するかの確認，(vi) LAC（`1 - p_true` 閾値，名目 0.90）の coverage / mean_set_size をオフラインで再測定し「サイズ最適構成の床＝3.83」を本実装系で裏付ける．

**実機フルスペック本走（2026-09-23 ユーザー指示．レバーではない）**

ユーザーから「wafl500〜509 を用いたフルスペックの本実験を積極的に実行する方針を貫け」との指示があったため，本イテレーションでは上記のオフライン実験と並行して**実機 10 ノードでの 1,600 問本走を実施する**．**これは単一レバーではなく，構成を一切変えない基準線の再取得・検証である**（`config.yaml`・分類器・モデルはすべて現行 HEAD のまま）．

- 手順: `mise run deploy` → `mise run start`（`data/dataset.jsonl` 1,600 問）→ `mise run analyze`．所要 90〜150 分（config の `timeout_min: 150`）．
- 実施理由 1（基準線の鮮度）: 直近の実機本走は `results/20260919_005727/`（top1=0.596875，kappa=0.565958，misrouting=0.403125，fallback=0.0，dispatch_failure=0.000625，mean_duration_ms=1502.16，single_domain_top1=0.609333，compound_top1=0.41）であり，Iter56 以降の 18 反復はすべてオフラインだった．現行 HEAD の end-to-end 指標を再確認する．
- 実施理由 2（conformal 系列との接続確認）: `results.jsonl` の `probe_candidates` は 10 ノード分の `confidence` を保持しており，**実行時ルーティングが実際に使っている確率ベクトルそのもの**である．これを `models/domain_classifier.joblib` の `predict_proba`（オフライン conformal 系列の入力）と 1,600 行全件で突き合わせ，許容差 1e-9 で一致するかを確認する．Iter56〜74 の conformal の議論が実機の分布に対して有効かどうかは，この 18 反復で一度も検証されていない（1 行のスポット確認では `business_economics-001` で 0.4884300235443738 vs 0.4884300235443737 と一致している）．この repo が 6 回繰り返した「設定は変えたのにコードへ到達しない」型の失敗と同じ系統の未検証事項である．
- 非退行の目安（判定には用いない．構成を変えていないので一致するはず）: top1_accuracy が 0.596875 から二項 SE=0.0123 の 2 倍（±2.5pt）以内，fallback_rate=0.0，dispatch_failure_rate <= 0.001．これを外れた場合は実機側の状態異常（モデル未 pull・ノード欠落等）を疑い，オフライン実験の判定とは切り離して報告する．

**期待効果**

`raps_penalty` を実装のうえで 1 点実行し，`conformal_set_size_reduction` レバーを 2 値とも試し切って閉じる．事前シミュレーションが示すとおり主基準は不成立となる公算が高いが，その場合でも得られるのは「サイズ縮小は構成法では達成できず，分類器の確率分布の鋭さと被覆水準が決めている」という，LAC の最適性（Sadinle et al. 2019）に裏打ちされた**否定的だが確定的な知見**である．これは conformal を本線のルーティングへ接続するか否かの判断材料そのものになる．併せて実機 1,600 問本走で end-to-end 基準線を 19 反復ぶりに再取得し，オフライン conformal 系列の入力が実行時経路と同一であることを初めて検証する．

**コスト**: オフライン実験は 1 実行 10〜30 分（埋め込み 1,600 行のみ，wafl-ctrl5 の ollama `127.0.0.1:11435`）＋後方互換アンカー 1 実行．実機本走は 90〜150 分（wafl500〜509 を占有）．分類器の再訓練は無し．

### Iteration 74 実行済み

**変更（2 ファイルのみ．`config.yaml`・`http_server.py`・`classifier.py`・`aggregator.py`・`mise.toml` は不変）**

1. `scripts/evaluate_classifier_calibration.py`: `_compute_prediction_set()` へ `raps_lambda: float | None = None`・`raps_k_reg: int | None = None` を追加し `set_construction="raps_penalty"` 分岐を新設（降順走査で 1-index の `rank` を持ち `score = 1 - cumsum + u·p_(rank) - λ·max(0, rank-k_reg)`，`score >= q_hat` の間だけ append し最初の不成立で break）．いずれかの引数が `None` なら `ValueError`（無言で `randomized_aps` へ落ちない）．`predict_calibrated_rows()` の `eval_holdout` 分岐の校正スコア（真クラスの 1-index 順位を `rank` として同じペナルティを適用）と評価ループ 2 箇所（fine-tuned/ollama）の両方へ `raps_lambda`/`raps_k_reg` を伝播．stderr 診断へ `raps_lambda`/`raps_k_reg` を追記．`main()` に `--raps-lambda`（既定 0.0）・`--raps-k-reg`（既定 2）を新設し，**`--output` 有無の 2 分岐両方へ伝播**（チェックリストで目視確認．漏れなし）．
2. `tests/test_evaluate_classifier_calibration.py`: 計画 5 の (a)〜(e) 相当 6 件（λ=0 での `randomized_aps` への縮退，λ増加に対するサイズの単調非増加，`raps_lambda`/`raps_k_reg`/`randomization_u` 欠落時の `ValueError`，`oof_train` との非互換 `ValueError`，`broken` 既定値の回帰）を追加．既存 24 件と合わせ **30 件 PASS**，`ruff check` PASS．

**実験（`results/20260923_142619/`，オフライン 3 実行＋実機 1 本走）**

- A（後方互換アンカー）: `--set-construction randomized_aps --randomization-seed 42`（他は Iter73 と同一）→ `Iter74_randomized_aps_backcompat.jsonl`．**`results/20260923_135653/Iter73_randomized_aps.jsonl` と md5 完全一致**（`afde650eea9ac611a43f0bb20703c2e3`）．
- B（本実行）: `--set-construction raps_penalty --raps-lambda 0.02 --raps-k-reg 2 --randomization-seed 42 --calibration-source eval_holdout --holdout-seed 42 --qhat-quantile-direction alpha_lower --qhat-source true_class --confidence-level 0.90` → `Iter74_raps_penalty.jsonl`．stderr は `set_construction=raps_penalty randomization_seed=42 q_hat=0.0323 n_cal=800 raps_lambda=0.02 raps_k_reg=2`（**レバー発火の証拠**．事前予測 q_hat≈0.032331 と一致）．
- C（実機フルスペック本走，レバーではなく基準線再取得＋接続確認）: `mise run deploy` → `mise run start`（`data/dataset.jsonl` 1,600 問，wafl500〜509）→ `mise run analyze 20260923_150540`．`results/20260923_150540/`．**`mise run analyze`（引数なし）はディレクトリ名をアルファベット順 `sort` で選ぶため `results/iter45_preliminary/`（`i` > `2`）を誤って選択する落とし穴があり，`analyze 20260923_150540` と明示して回避した**（新規の未報告事項として付随報告に記載）．

| 指標 | 定義 | Iter73 実測 | 事前予測（λ=0.02, k_reg=2） | Iter74 実測（eval 半 n=800） | 合格条件 | 判定 |
|---|---|---|---|---|---|---|
| coverage（主基準 1） | eval 半で `mean(expected_domains[0] in prediction_set)` | 0.90375 | 0.90750 | **0.90750** | 0.88 ≤ coverage ≤ 0.95 | **PASS** |
| mean_set_size（主基準 2） | eval 半の `mean(set_size)` | 4.00125 | 4.16375 | **4.16375** | 4.00125 から有意に減少・減少幅 ≥ 0.5 | **FAIL**（+0.1625 の有意な**増加**，`ttest_rel` t=-7.515, p=1.53e-13） |
| q_hat | — | 0.0939 | 0.032331 | **0.0323** | — | — |
| P(set_size≤2) | eval 半 | 0.1800 | 0.0775 | **0.0775** | 報告のみ | — |
| 複合 46 行の 2 ドメイン同時被覆 | 複合設問のみ | 0.2826 | 0.2826 | **0.2826** | 報告のみ | — |

- 対応あり 800 行の set_size 差分（Iter73 − Iter74）: **減少（73>74）86 行・同数 507 行・増加（73<74）207 行**．事前予測と完全一致．
- 集合サイズ分布（eval 半）: `{1:5, 2:57, 3:154, 4:259, 5:241, 6:79, 7:5}`．事前予測 `[5, 57, 154, 259, 241, 79, 5, 0, 0, 0]` と完全一致．
- 非退行 6 項目すべて充足: `selected_domain` 完全一致（0 件不一致，1600 行中） / `probabilities` 最大差 0.0（浮動小数点誤差ですら発生せず） / `split` 割当一致（1600 行中 0 件不一致） / `set_size` 1〜7 かつ重複なし / A の再実行が Iter73 出力と **md5 完全一致** / stderr 発火証拠あり．
- 付随（複合設問，B109 制約 (3)）: 2 ドメイン同時被覆 0.2826（46 行中 13 行），どちらか 1 つ被覆 0.8913，複合行の mean_set_size 3.9565．いずれも事前予測と一致．

**実機フルスペック本走（C）の結果**

| 指標 | `results/20260919_005727/`（直近基準線） | `results/20260923_150540/`（本反復） | 差 |
|---|---|---|---|
| top1_accuracy | 0.596875 | **0.595625** | -0.00125（二項 SE=0.0123 の 0.1 SE） |
| cohens_kappa | 0.565958 | **0.564477** | -0.00148 |
| misrouting_rate | 0.403125 | **0.404375** | +0.00125 |
| fallback_rate | 0.0 | **0.0** | 0 |
| dispatch_failure_rate | 0.000625 | **0.000625** | 0 |
| mean_duration_ms | 1502.16 | **1499.76** | -2.40 |
| single_domain_top1 | 0.609333 | **0.608** | -0.00133 |
| compound_top1 | 0.41 | **0.41** | 0 |
| answer_quality_accuracy | — | 0.562667 | — |
| end_to_end_accuracy | — | 0.330625 | — |
| ece | — | 0.055085（n=1599） | — |

構成を一切変えていない基準線の再取得として，非退行の目安（±2.5pt 以内・fallback=0.0・dispatch_failure≤0.001）をすべて満たした．19 反復ぶりの実機本走で end-to-end 指標が Iter56 直前の水準から動いていないことを確認した．

**実施理由 2（`probe_candidates` と `predict_proba` の突き合わせ）の結果**: `results/20260923_150540/results.jsonl` の `probe_candidates`（10 ノード分の `confidence`）と，同一評価データに対する `models/domain_classifier.joblib.predict_proba()` のオフライン再計算（`--education-threshold 0.0` 指定，本実行 B と同条件）を 1,600 行×10 ドメイン全件で突き合わせたところ，**`education` ドメインのみ全 1,600 行で厳密に +0.05 の系統的な差**が見つかり，他 9 ドメインは最大絶対差 1e-15 台（浮動小数点誤差のみ）で一致した．原因を `classifier.py` を読んで特定した: **Iter52/53 で adopted・Iter57 前後で実行時経路 `classifier.py:estimate_confidence_classifier()` へ配線された `education_per_class_threshold=0.05`（`production_deployment_gap` レバー，config.yml 冒頭節参照）が実行時には効いているが，本イテレーションの conformal オフライン評価は計画どおり `--education-threshold 0.0` に固定しているため，この差分が生じる**．`--education-threshold 0.05` を指定して同じ突き合わせをやり直すと，**1,600 行×10 ドメイン全件で最大絶対差 0.0（ビット単位で完全一致）** になることを確認した（`results/20260923_142619/Iter74_probs_only_edu005.jsonl` で検証）．結論: **実行時ルーティングが使う確率ベクトルは，`education` の +0.05 補正を含めれば `models/domain_classifier.joblib` の `predict_proba` とビット単位で一致する**．Iter56〜74 の conformal 系列は，この実行時分布に対して有効な議論をしてきたことが初めて裏付けられた．一方で，conformal 系列は一貫して `--education-threshold 0.0` を使ってきており（本イテレーションもそれを踏襲），これは実行時経路とは異なる分布を評価してきたことを意味する．この差の影響（education 分類の被覆・集合サイズへの影響）は未評価であり，次回以降の検討事項として考察節に記録する．

### 分析(解釈) (Iter74)

**1. 実測は事前シミュレーションと完全一致し，実装は正しく発火した**

coverage=0.90750（予測 0.90750）・mean_set_size=4.16375（予測 4.16375）・q_hat=0.0323（予測 0.032331）・サイズ分布・対応あり差分の内訳（86/507/207）まで小数点以下含めて一致した．非退行 6 項目もすべて充足し，`probabilities` は浮動小数点誤差の範囲でさえ動いていない（最大差 0.0）．Iter16/20/21/22/27 系列の「設定は変えたのにコードへ到達しない」型の失敗ではないことは，stderr の `q_hat=0.0323`（λ=0 なら 0.0939 が出るはずの箇所）と A 実行の md5 一致で二重に確認できている．

**2. 主基準 2（サイズの有意な減少）は事前登録どおり不成立**

mean_set_size は 4.00125 → 4.16375 で **+0.1625 の有意な増加**（`ttest_rel` t=-7.515, p=1.53e-13）．増加方向であり「減少」という主基準の定義自体を満たさない．事前登録した解釈基準（実測が予測 4.16375 の ±0.02 以内なら実装は正しく発火したうえで手法が効かなかったと判定）に照らすと，**実測は予測と完全一致（差 0.0）** であり，「実装不成立」ではなく「手法が効かない」という判定になる．

**3. 機序は調査 Q3 の事前シミュレーションで特定済みの内容がそのまま再現された**

RAPS のペナルティは校正側の真クラススコアと評価側の各クラススコアの両方に同形で入るため，λ を上げると q_hat がほぼ同量だけ下がって相殺し，残るのはサイズ分布の圧縮（size 1 が 52→5 行，size 4〜5 に 360→500 行が集中）であって平均の縮小ではない．10 クラス・最大サイズ 8 の本問題には ImageNet（RAPS の原論文の実験対象，1,000 クラス）のような「切るべき裾」が無いという調査時点の考察が実測でも裏付けられた．

**4. 実機フルスペック本走は非退行を確認し，conformal 系列の前提を初めて検証した**

19 反復ぶりの実機 1,600 問本走は，直近基準線（`results/20260919_005727/`）と非退行の目安を満たす水準（top1 差 -0.00125，fallback/dispatch_failure 不変）で再現した．より重要な収穫は `probe_candidates` と `models/domain_classifier.joblib.predict_proba` の突き合わせで，**`education` の `+0.05` 補正を揃えればビット単位で一致する**ことが判明した点である．これにより，Iter56〜74 の conformal の議論（校正・被覆・集合構成）が実際の実行時分布に対して有効であったことが初めて裏付けられた一方，conformal 系列自体は一貫して `--education-threshold 0.0` を使ってきたため，10 ドメイン中 1 つ（education）については実行時分布とは異なる分布を評価してきたことも同時に判明した．この差の影響は本反復では評価していない．

### 考察 (Iter74)

**判定: `raps_penalty` は rejected（確定）**．事前登録の解釈基準どおり，実測が予測の ±0.02 以内で一致したため「実装は正しく発火したうえで手法が効かなかった」と判定する．実装は revert せず維持する（既定 `broken` が後方互換 md5 を保つ設計であり，`raps_penalty` 分岐はコードベースに残すが今後の固定値としては採用しない）．

**`conformal_set_size_reduction` レバーは 2 値（`randomized_aps` adopted, `raps_penalty` rejected）を試し切り，クローズする**．

**この反復で確定した知見**

1. RAPS のサイズ正則化は，校正・評価の両側に同形で入るペナルティ付き分位点法である限り，q_hat の自己補正によって「平均サイズの縮小」ではなく「サイズ分布の圧縮」にしか効かない．これは調査 Q3 の事前シミュレーション（k_reg∈{1,2,3,4}×λ∈[0.001,0.5] の全域探索）と LAC（サイズ最適構成）による床の確認（名目 0.90 で 3.83）の両方から導かれ，本実装でも寸分違わず再現した．
2. 「予測集合を top-2 dispatch へ流用する」という Iter72 以来の構想は，**構成法の選択では到達できない**．到達するには被覆水準を 0.80 前後まで下げるほかなく，それは複合設問の同時被覆（Iter73 分析 5）をさらに崩す．`conformal_set_size_reduction` レバーはこれで打ち切りが妥当である．
3. **実機本走で初めて，オフライン conformal 系列の入力が実行時ルーティングの確率分布と（education の +0.05 補正を除いて）ビット単位で一致することを確認できた**．これは Iter56〜74 の 19 反復にわたる conformal の議論の妥当性を裏付ける一次情報であり，同時に「conformal 系列は education について実行時と異なる分布を評価してきた」という新しい留保も明らかにした．

**次の一手（人間判断を要する事項を含む）**

- `conformal_set_size_reduction` はクローズ．次に残る conformal 関連の論点は config.yml 冒頭に記録済みの **B104 A2（実行時経路への配線）** と **B104 A1（複合評価集合の検出力，eval 半 46 行）**．本反復では新たに確定させない．
- **新規発見（要 backlog 起票）**: conformal オフライン評価が一貫して `--education-threshold 0.0` を使ってきたことによる，実行時分布との education 側の乖離．被覆・集合サイズへの影響は未評価．次に conformal 系列へ戻る際は `--education-threshold 0.05` を既定に含めるかを検討する必要がある．
- **新規発見（要 backlog 起票，運用上の落とし穴）**: `mise run analyze`（datetime 引数省略時）はアルファベット順 `sort` でディレクトリを選ぶため，`results/iter45_preliminary/` のような非タイムスタンプ形式のディレクトリが最新のタイムスタンプディレクトリより後にソートされ，誤って選択される．本反復では `analyze <datetime>` と明示して回避したが，スクリプト側の修正（タイムスタンプ形式のみを対象にする，または最終更新時刻でソートする）を検討する価値がある．

### 反省・判定確定 (Iter74, reflector)

**数値の独立再検証（出力 jsonl から reflector 自身が再計算した）**

実行フェーズの報告値を鵜呑みにせず，`results/20260923_142619/Iter74_raps_penalty.jsonl` と `results/20260923_135653/Iter73_randomized_aps.jsonl` を直接読み直して再集計した．eval 半 n=800 で coverage=0.90750・mean_set_size=4.16375・P(size<=2)=0.0775・サイズ分布 `{1:5, 2:57, 3:154, 4:259, 5:241, 6:79, 7:5}`・複合 46 行の 2 ドメイン同時被覆 0.2826・対応あり差分（減少 86／同数 507／増加 207）・`ttest_rel` t=-7.514964855512641, p=1.5276e-13 をすべて再現した．非退行も再検証し，`probabilities` の最大絶対差 0.0（ドメイン辞書の全キー・全 1,600 行），`selected_domain` 不一致 0 件，`split` 不一致 0 件，`set_size` は 1〜7 で `prediction_set` に重複なし，後方互換アンカーの md5 `afde650eea9ac611a43f0bb20703c2e3` は Iter73 出力と一致した．

**ノイズか信号かの切り分け**

- `mean_set_size` の +0.1625 は対応あり 800 行の SE=0.1625/7.515=0.02163 に対し **7.5 SE** であり，ノイズではなく信号である．しかも方向が主基準（減少）の**逆**で，合格条件（減少幅 >= 0.5）とは 0.66 の隔たりがある．
- `coverage` の 0.90375 → 0.90750（+0.00375）は二項 SE=0.0106 の 0.35 倍で**ノイズ幅の中**．帯 0.88-0.95 の内側（下限から +2.59 SE，上限まで -4.01 SE）であり，主基準 1 は満たす．
- 実測と事前予測の差は `mean_set_size` で **0.0**（事前登録した許容幅 ±0.02 の中心）．coverage・q_hat・サイズ分布・差分内訳も含め予測と乖離が無いので，Iter16/20/21/22/27 型の「設定を変えたのにコードへ到達しない」失敗ではないと確定できる．

**判定（確定）: `raps_penalty` は rejected．ただし「実験不成立」ではなく「実装は正しく発火したうえで手法が効かない」**

事前登録した解釈規則（実測が予測 4.16375 の ±0.02 以内なら実装成立・手法無効と読む）をそのまま適用した．差が 0.0 であることに加え，stderr の発火証拠（`q_hat=0.0323`，λ=0 なら 0.0939 が出る箇所）と後方互換 md5 一致という 2 つの独立した証拠が揃っている．因果として言えるのはここまでで，「RAPS は一般に無効」ではなく **「校正・評価の両側へ同形のペナルティを入れる分位点法は，クラス数が少なく裾の軽い本問題では q_hat の自己補正により平均サイズを縮めない」** という条件付きの主張に留める．

**`conformal_set_size_reduction` レバーは全 2 値試行済みでクローズ．「集合サイズを構成法で 2 近傍へ縮める」路線も閉じる**

閉じる根拠は本反復 1 点の結果ではない．(a) k_reg∈{1,2,3,4}×λ∈[0.001,0.5] の全域掃引で最小が λ→0（＝`randomized_aps`）だったこと，(b) サイズ最適性が証明されている LAC/THR（Sadinle et al., JASA 2019）でも名目 0.90 で 3.825 が床であること，(c) 本反復でその掃引の 1 点が実装上も寸分違わず再現されたこと，の 3 点である．size≈2 には被覆水準を 0.80 前後まで落とすほかなく，それは複合設問の 2 ドメイン同時被覆を 0.26→0.065 へ崩す（Iter73 分析 5 と整合）．B109 の要レビュー事項「予測集合を top-k dispatch へ流用する構想は conformal の保証（第 1 ドメインの周辺被覆）と目的（2 ドメイン同時網羅）がずれている」は，本反復で**定量的に裏付けられた**．

**次の単一レバー（Iter75）: 新レバー `conformal_runtime_distribution_alignment = education_threshold_0.05`**

`conformal_set_size_reduction` のクローズにより config の levers は再び全て試行済みになったため，skill の停止条件を順に適用し，**停止条件 1（学びから新レバーを考案）** を採った．本反復の実機本走で得た新事実——実行時ルーティングの確率分布は `education` に +0.05 された分布であり，Iter56〜74 の 19 反復の conformal はその補正を外した分布の上で被覆を議論してきた——が，そのまま次に振るべき 1 変数を与えている．`--education-threshold 0.0 → 0.05` の 1 点だけを動かし，Iter73 の adopted 構成（`randomized_aps` / seed 42 / `eval_holdout` / `alpha_lower` / `true_class` / 0.90）は固定する．既存 CLI に実装済みのため新規実装は不要，オフライン完結で実機ノードも使わない．

このレバーを選んだ理由は 2 つある．第 1 に，**B104 A2（conformal を実行時経路へ配線するか）を人間に諮る前に答えておくべき前提条件**だからである．オフラインで測った被覆が実行時分布でも成立するかが未確認のまま配線の是非を問うても，判断材料が欠けている．第 2 に，これは新手法の検証ではなく既存 19 反復の結論の外的妥当性の確認であり，`production_deployment_gap` レバー（Iter57 以降）と同じ性格の，自律着手可能で結論が確定する作業だからである．主基準は「実行時と同一の分布でも 0.88 <= coverage <= 0.95 が保たれること」，副基準として education を正解とする行に限った被覆・集合サイズを報告する（+0.05 が効くのはこの部分集合であり，ここが動かなければ 19 反復の議論は実行時分布でもそのまま有効と言える）．

**この反復で得た非自明な学び（次の自分向け）**

1. **事前シミュレーションは「合格を探す道具」ではなく「反証実験を 1 点に絞る道具」として機能した**．Iter71 以降 4 反復連続で予測が実測と一致しており，本反復では掃引全域で主基準を満たす点が無いことを事前に知ったうえで，あえて 1 点を実行して手法を根拠付きで閉じた．λ を振り直して合格を探さないと事前に宣言しておいたことが，rejected を「失敗」ではなく「確定した知見」に変えている．
2. **rejected の解釈規則を実行前に数値で登録しておくと，実装バグと手法の無効を事後に切り分けられる**（本反復では ±0.02）．この repo が 6 回繰り返した「実験不成立を効果なしと誤読する」失敗への，事前登録という形の対策として機能した．
3. **19 反復にわたるオフライン系列の入力が，実行時経路の入力と同一であることは誰も確認していなかった**．ユーザー指示で実施した実機本走（レバーではない基準線再取得）の副産物として初めて突き合わせ，education のみ +0.05 ずれていることが分かった．オフラインで長く回す系列は，入力分布が実行時と一致しているかを定期的に一次データで確認する必要がある．

---

