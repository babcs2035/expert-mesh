## Iteration 83: 連結埋め込みの確信度分布に合わせた送出 gap 閾値の再較正

### 調査 (Iter83)

本反復のレバーは backlog B130（Iter82 分析フェーズ）で `dispatch_gap_threshold_recalibration` = `matched_budget_sweep_for_concat_distribution` に確定済みで，レバー選定の裁量は無い．調査の問いは 3 つ．**(Q1) 「スコアリングモデルを変えたら送出／棄却の閾値を較正し直す」という手続きは先行研究上どう扱われているか（本リポジトリで 2 回続けて観測した現象の一般性）．(Q2) 閾値を上げれば送出数も被覆も単調に増えるのだから，「set_recall が上がった」を成功条件にしてよいのか（＝評価設計の問題）．(Q3) 評価集合そのもので閾値を選ぶことの過適合リスクをどう抑えるか．**

**Q1: 閾値はスコアリングモデルに固有のハイパラであり，モデルを差し替えたら再較正するのが標準である**

- Zellinger & Kim, "Rational Tuning of LLM Cascades via Probabilistic Modeling"（arXiv:2501.09345，2025，<https://arxiv.org/abs/2501.09345>）は，カスケードの confidence 閾値を「カスケードを構成するモデル群に対して連続最適化で決めるハイパラ」として定式化している．閾値はモデルの確信度分布の形に依存するので，**構成モデルが変われば閾値は別の最適点へ移る**というのがこの定式化の前提である．
- TREACLE（"Efficient Contextual LLM Cascades through Budget-Constrained Policy Learning"，NeurIPS 2024，<https://proceedings.neurips.cc/paper_files/paper/2024/file/a6deba3b2408af45b3f9994c2152b862-Paper-Conference.pdf>，2026-09-27 確認）でも，比較対象の "Calibrated cascade"（FrugalGPT 系）の閾値は **validation set で tune する**と明記されており，スコア推定器を変えたら閾値も引き直す扱いである．
- 適合予測（conformal prediction）でも同じ構造である．非適合スコアの分布から分位点として閾値を決めるため，**スコア関数（＝ここでは埋め込み＋分類器）を変えたら較正集合から閾値を取り直す**のが手続きそのものに組み込まれている（例: <https://daniel-bethell.co.uk/posts/conformal-prediction-guide>，2026-09-27 確認）．
- **本リポジトリの観測との接続**: Iter81 学び 1・Iter82 学び 3 で 2 回続けて「特徴量を変えた後に `dispatch_gap_threshold=0.29` の未較正が残る」ことを観測した．これは本研究に固有の不具合ではなく，上記の一般則がそのまま現れたものと解釈できる．**Iter58 で決めた T=0.29 は，当時の埋め込み（prefix なし 1024 次元）の確信度分布に対して選ばれた値**であり，Iter82 で特徴量を 2048 次元連結へ変えた時点で前提が失効している．

**Q2: 「set_recall が上がった」は単独では成功条件にならない．送出予算を揃えた比較（iso-budget）にしなければならない**

- `aggregator.select_dispatch_targets()` の gap escalation（L87-91）は `k` を 1 から始めて隣接 rank の gap が閾値未満の間だけ増やす規則であり，**閾値を上げれば `k` は単調非減少，したがって被覆も単調非減少**である．実際，本フェーズのオフライン掃引でも gt を 0.20→0.65 と上げる間，`compound_domain_set_recall` は 0.4964→0.6916 へ単調に増え，`compound_mean_dispatched_count` は 1.369→2.706 へ単調に増えた．**閾値を上げるだけで被覆は好きなだけ買える**から，被覆の増加そのものには情報量が無い．
- 予算制約下での比較は routing 研究の標準的な枠組みでもある．OmniRouter（arXiv:2502.20576，<https://arxiv.org/html/2502.20576v6>，2026-09-27 確認）は品質制約つきのコスト最小化として定式化し，品質閾値を等号で満たす点で比較する．TREACLE も "budget-constrained" を明示している．
- **したがって本反復の主基準は「送出予算を Iter79 基準線と同じ水準（`compound_mean_dispatched_count` ≈ 1.88）に固定した上で `compound_domain_set_recall` が基準線を上回ること」とする**．これは B130 のレバー名 `matched_budget_sweep_for_concat_distribution` の意図そのものである．予算上限を非退行条件として**事前に固定**し，その枠内での被覆改善だけを成果として数える．

**Q3: 過適合リスクは残るが，(i) 予算整合点を選ぶ（argmax を選ばない）・(ii) split-half の交差確認，で抑える**

- 実務上の注意として FrugalGPT 系の解説（<https://www.komos.studio/blog/frugalgpt-llm-cascades>，2026-09-27 確認）は「**閾値の調整に使っていない hold-out を必ず持て．調整集合で良く見えた閾値が新しい仕事で失敗することがある**」と述べている．本研究には複合設問の hold-out が無く（複合 415 行がそのまま評価集合である），掃引も評価集合上で行っている．**この限界は消せないので明記する．**
- 抑制策 1: **掃引の argmax（gt を上げられるだけ上げた点）を選ばない**．予算整合点は「被覆を最大化する点」ではなく「Iter79 と同じコストで比較するために外から決まる点」なので，評価集合へのフィッティングの自由度は事実上 1 次元ぶん潰れている．
- 抑制策 2: **split-half の交差確認（本フェーズで実測済み）**．複合 415 行を偶奇で A（208 行）/ B（207 行）に分け，片方で予算整合閾値を選んでもう片方へ適用した．A で選んだ gt=0.387 を B へ適用すると set_recall 0.5290→**0.5942**，B で選んだ gt=0.336 を A へ適用すると 0.5505→**0.5649**．**どちらの向きでも改善し，符号は反転しない**．半分ずつでは予算整合点が 0.336〜0.387 とばらつく（単調・平坦な曲線の上で予算目標に最も近い点を拾うため）が，被覆の改善方向は安定している．

### 計画 (Iter83)

**単一レバー**

`dispatch_gap_threshold_recalibration` = **`matched_budget_sweep_for_concat_distribution`**．`config.yaml:90` の **`dispatch_gap_threshold` を `0.29` から `0.36` へ変える**こと**だけ**を行う．再訓練も新規実装も無く，変更は config 1 行（＋説明コメント）である．

**閾値 0.36 を選んだ理由（候補を 1 値に絞る手続き．B130 の事前登録要件）**

本フェーズで `results/20260927_050049/results.jsonl`（Iter82 本走）の `probe_candidates` から `aggregator.select_dispatch_targets()` の gap escalation 規則を再現し（`confidence_threshold=0.0`・`dispatch_candidate_threshold=0.0`・`gap_max_k=4`），gt を 0.20〜0.65 まで 0.01 刻み（0.350〜0.395 は 0.005 刻み）で掃引した．**実機は使っていない**（ローカルの results.jsonl のみ．2026-09-23 恒久ルール）．

| gt | `compound_mean_dispatched_count` | `compound_domain_set_recall` | 全 1,915 行の mean dispatched |
|---|---|---|---|
| 0.29（現行＝Iter82 本走の実測値） | 1.6096 | 0.539759 | 1.4078 |
| 0.34 | 1.8000 | 0.569880 | 1.5384 |
| 0.355 | 1.8627 | 0.578313 | 1.5911 |
| **0.357〜0.364（同一の階段．採用点）** | **1.8892** | **0.581928** | **1.6010**（gt=0.36） |
| 0.37 | 1.9277 | 0.585542 | 1.6287 |
| 0.40 | 2.0217 | 0.598795 | 1.7055 |
| 0.50 | 2.3012 | 0.630120 | 1.8914 |
| （参考）Iter79 基準線 gt=0.29 | 1.8795 | 0.548193 | 1.5608 |

- **予算整合点は 0.357〜0.364 の階段である**．Iter79 基準線の複合送出予算 1.8795 に対し，この階段は 1.8892（+0.5%）で最も近い（下側の隣接階段 gt=0.355 は 1.8627 で -0.9%，距離はやや遠い）．
- **階段の中央に近い 0.36 を採る**．0.357 は階段の左端にあり，実機の浮動小数点表現や将来の微小な確信度変動で隣の階段（1.8627）へ落ちうる．0.36 なら階段の内側に余裕があり，同じ予測値（mean_k=1.8892 / set_recall=0.581928）を安定して与える．B130 が例示した 0.357 と**複合設問上の挙動は完全に同一**である．
- **gt=0.40（set_recall 0.598795）は採らない**．Q2 のとおり被覆は予算を増やせば単調に買えるので，予算を +7.6%（1.8795→2.0217）増やして得た +5.06pt は「較正の効果」と「予算増の効果」の交絡である．**本反復が主張したいのは「同じコストで被覆が増える」ことであり，そのためには予算整合点しか使えない．**

**固定する構成（直近の最良構成 = Iter82 本走 `results/20260927_050049/` の構成そのまま）**

`config.yaml` の `embedding_model=qwen3-embedding:0.6b`・`embedding_instruction`（Iter81 の P1 文言）・`embedding_view_concat=true`・`routing_method=supervised_classifier`・`confidence_threshold=0.0`・`dispatch_candidate_threshold=0.0`・**`dispatch_top_k=2`**・**`dispatch_gap_max_k=4`**・`aggregation_method=max_confidence`・`judge_model`・`classifier_model_path`，分類器 artifact `models/domain_classifier.joblib`（**sha256 `1cfcd3d8...`，`n_features_in_`=2048．再訓練しない**），`data/dataset.jsonl`（1,915 行，ビット単位で不変），`data/classifier_train.jsonl`（1,427 行，不変），各ノードの `light_model` / `expert_model`，`probe_timeout_s` / `dispatch_timeout_s`，`node.py`・`aggregator.py`・`classifier.py`・`metrics.py`・`run_experiment.py`（いずれも**コード変更なし**）．**ドメインごとに閾値を変えること，およびドメイン固有の後付け補正は 2026-09-23 恒久運用ルールにより行わない．閾値は 10 ドメイン共通の単一スカラーである．**

**仮説（事前登録．本走前に数値で記録する）**

「Iter82 で採用した連結埋め込みにより rank1−rank2 gap の分布が右へ移動した（平均 0.4769→0.5499，gap<0.29 の行の割合 33.49%→25.30%）ため，Iter58 に旧分布上で選ばれた `dispatch_gap_threshold=0.29` は現構成では過度に厳しい．これを 0.36 へ戻すと，**Iter79 基準線と同じ送出予算のまま複合設問の被覆が基準線を上回る**．」

**着地点予測（`probe_candidates` からの決定論的 replay．G1 で忠実度を確認済み）**

| 指標 | Iter82 本走実測（直近基準線） | Iter79 本走（旧基準線） | **Iter83 予測値（gt=0.36）** |
|---|---|---|---|
| `compound_domain_set_recall` | 0.539759 | 0.548193 | **0.581928**（Iter82 比 **+4.217pt**，Iter79 比 **+3.373pt**） |
| 複合 415 行の被覆スロット数（分母 830） | 448 | 455 | **483**（Iter82 比 獲得 +35 / 喪失 0，符号検定 p=5.82e-11） |
| `compound_mean_dispatched_count` | 1.6096 | 1.8795 | **1.8892** |
| `compound_domain_jaccard_mean` | 0.458072 | — | **0.456466**（送出数が増える分わずかに低下．報告のみ） |
| 全 1,915 行の mean dispatched | 1.4078 | 1.5608 | **1.6010** |
| 複合 415 行の k 分布（k=1 / 2 / 4） | 310 / 31 / 74 | — | **278 / 21 / 116** |
| `top1_accuracy`（全 1,915 行） | 0.792167 | 0.753003 | **0.792167（不変）**．許容幅 ±0.06pt（下記の構造的理由により最大 1 行） |
| `mean_duration_ms` | 2217.1 | 2301.4 | **≈2320**（Iter82 2217.1@k=1.408 と Iter79 2301.4@k=1.561 の線形内挿 ≒ 551ms/送出．非退行枠 2660.5 に対し余裕あり） |
| `answer_quality_accuracy` / `end_to_end_accuracy` | 0.572667 / 0.351958 | 0.569333 / 0.335770 | **構造的にはほぼ不変**．生成の揺らぎ（3SD=2.6pt）の範囲でのみ動く |

**top1 と下流指標が構造的に動かない理由（B130 の申し送りに対する精密化）**

`aggregation_method=max_confidence` の下で最終回答を選ぶ `aggregator.select_best_dispatch_response()`（`aggregator.py:104-119`）は `DispatchResponse.confidence` の最大値を採るが，この confidence は **`/probe` 時に計算された同一の値**である（同関数 docstring に明記）．したがって送出先を増やしても選ばれるのは常に rank 1 であり，`selected_domain` は gap 閾値と独立に決まる．**例外は rank 1 への `/dispatch` が失敗した行だけ**で，そこでは候補が増えたぶん rank 2 が拾われうる（Iter82 の `dispatch_failure_rate` は 0.000522 ＝ 1,915 行中 1 行）．よって `top1_accuracy`・per-domain recall/precision・kappa・ECE は**最大 1 行（0.052pt）の幅でしか動きえない**．`answer_quality_accuracy` / `end_to_end_accuracy` も採点対象の `answer_text` が rank 1 由来のままなので，動くとすれば生成のランダム性による．**B130 の「`end_to_end_accuracy` は送出候補が増えることで改善しうる」は max_confidence 集約の下では機序が無く，本計画では「ノイズ床 2.6pt の範囲で不変」と予測する**（この予測の当否自体も本走で記録する）．

**レバーを読むコード行と，そこへ到達する条件（「config は変えたが実行パスに到達しない」失敗への恒久対策．省略不可）**

| 経路 | レバーを読む行 | 到達条件 | 到達確認の手段 |
|---|---|---|---|
| **実行時側（本体）** | `node.py:225` `gap_threshold=config.get("dispatch_gap_threshold")` → `aggregator.select_dispatch_targets()` の `aggregator.py:87-91`（`while k < ceiling and (candidates[k-1].confidence - candidates[k].confidence) < gap_threshold`） | 依頼者ノード wafl500 のコンテナが**配布後の** `config.yaml` を読むこと．`config.yaml` は `mise.toml:69` の `rsync` で各ホストへ配られ，コンテナは `docker-compose.yml` 経由でマウントする | **F2**: 全 10 ノードで `grep '^dispatch_gap_threshold:' $REMOTE_DIR/config.yaml` が `0.36`．加えて deploy 内の smoke_check（hashes）で config.yaml のハッシュ一致 |
| **記録側（指標の分母）** | `run_experiment.py:98` `gap_threshold=config.get("dispatch_gap_threshold")`．ここは `dispatched_domains` を再計算する**独立した 2 回目の呼び出し**（Iter58 のコメント参照） | 同上（同じ config を読む） | **F3**: 予備 20 問の `dispatched_domains` が同 20 問のオフライン replay（gt=0.36）と一致すること．ここが未到達なら `dispatched_domains` だけ旧閾値で記録され，実際の送出と乖離する |
| 設定 → 各ノード | `mise.toml:69` の `rsync ... config.yaml` | `mise run deploy` の実行 | **F2** |
| **効いたことの直接証拠** | — | — | **F3'**: 本走完了後の `compound_mean_dispatched_count` が **1.6096（Iter82 実測）と一致しない**こと．一致したら閾値が実行パスに届いていない（下記 invalid 条件） |
| **到達しない経路（確認のみ．変更しない）** | 埋め込み（`expert_backend.embed_query_views()`）・分類器（`classifier.py`）・`train_domain_classifier.py` | 本反復では一切触らない | artifact sha256 が **`1cfcd3d8...` のまま不変**であること，`config.yaml` の diff が `dispatch_gap_threshold` 行（＋コメント）のみであること |

**オフラインで先に確認できること / 実機本走でしか確認できないこと**

- **オフラインで確認できる（ローカルの `results/` のみ．実機不使用）**: G1（掃引の忠実度），G2（着地点の事前登録），G3（split-half 交差確認）．いずれも**本フェーズで実測済み**で，下表に結果を記載する．
- **実機本走でしか確認できない**: `mean_duration_ms`（送出数増加の実コスト），`dispatch_failure_rate`（送出先が増えることでのタイムアウト増），`fallback_rate`，`answer_quality_accuracy` / `end_to_end_accuracy`，および「`config.yaml` の 1 行が本当に実行時挙動を変えたか」（F2/F3/F3'）．
- **config.yml 冒頭の絶対条件 (A) により，変更を適用したら wafl500〜509 での 1,915 問フルスペック本走を必ず 1 回実施する．掃引の予測がどれだけ確かでも本走を省略しない**（2026-09-23 恒久ルール: 事前シミュレーションは本走の代替ではない）．

**事前ゲート（結果を見る前に固定する．G1〜G3 は本フェーズで実測済み）**

| ゲート | 内容 | 実測 | 判定 |
|---|---|---|---|
| **G1（掃引の忠実度）** | 本フェーズの replay を gt=0.29 で走らせ，Iter82 本走の実測 `compound_domain_set_recall`=0.539759 / `compound_mean_dispatched_count`=1.6096 を再現すること | replay: set_recall **0.5397590361**，mean_k **1.6096385542**．Iter79 基準線でも gt=0.29 で set_recall 0.5481927711 / mean_k 1.8795180723 を再現 | **PASS**（小数点以下 10 桁まで一致．掃引は実行時挙動を忠実に再現している） |
| **G2（着地点の事前登録）** | gt=0.36 の予測 `compound_mean_dispatched_count` と `compound_domain_set_recall` を**本走前に**数値で記録すること | **mean_k=1.8892 / set_recall=0.581928**（上表に全項目記載） | **PASS**（記録済み．合格条件は課さない） |
| **G3（評価集合への過適合の確認）** | 複合 415 行を偶奇で二分し，片方で選んだ予算整合閾値をもう片方へ適用して改善の符号が反転しないこと | A で選んだ 0.387 → B: 0.5290→**0.5942**．B で選んだ 0.336 → A: 0.5505→**0.5649**．**両方向とも改善** | **PASS** |
| **F1（変更の最小性）** | `git diff` が `config.yaml` の `dispatch_gap_threshold` 行（＋説明コメント）のみであること．`.py` の変更が 0 件であること | 本走前に確認 | 実装フェーズで判定 |
| **F2（配布）** | deploy 後，全 10 ノードで `grep '^dispatch_gap_threshold:' $REMOTE_DIR/config.yaml` が `0.36`．artifact sha256 が `1cfcd3d8...` で不変 | 本走前に確認 | 実装フェーズで判定 |
| **F3（実行時経路）** | 先頭 20 問の予備実行の `dispatched_domains`（集合・長さの双方）が同 20 問のオフライン replay（gt=0.36）と **20/20 一致**すること | 本走前に確認．**不一致なら本走に進まない** | 実装フェーズで判定 |

**成功条件・非退行条件（事前登録．結果を見る前に固定する）**

直近基準線は **Iter82 本走 `results/20260927_050049/`**（全 1,915 行: top1=0.792167，compound_domain_set_recall=0.539759，compound_mean_dispatched_count=1.6096，fallback=0.0，dispatch_failure=0.000522，ECE=0.025448，mean_duration_ms=2217.1，answer_quality=0.572667，end_to_end=0.351958）．予算整合の参照点として **Iter79 本走 `results/20260926_221822/`**（set_recall=0.548193，compound_mean_dispatched_count=1.8795，全 1,915 行 mean dispatched=1.5608）も併記する．

| 区分 | 指標 | 基準線 | 合格条件 |
|---|---|---|---|
| **主基準（効果）** | `compound_domain_set_recall`（複合 415 行，分母 830 スロット） | Iter82: 0.539759 / Iter79: 0.548193 | **(i) Iter79 の 0.548193 を上回り，かつ (ii) Iter82 の 0.539759 に対し +3.0pt 以上（≥ 0.5698）**．2 条件の AND |
| **主基準の予算制約（必須・これが無いと主基準は自明に満たせる）** | `compound_mean_dispatched_count` | Iter79: 1.8795 | **≤ 1.90**（Iter79 の予算 +1.1% 以内）．超過したら「予算を増やして被覆を買っただけ」として rejected |
| **非退行①（構造的不変の確認）** | 全 1,915 行 `top1_accuracy` | 0.792167 | **≥ 0.791645（-0.052pt = 1 行まで）**．**理論上は完全不変であり，2 行以上動いたら実装漏れ・別要因の混入を疑い invalid 判定の検討へ回す** |
| **非退行②** | per-domain recall/precision 計 20 指標（BH 補正 q=0.05） | Iter82 実測 | **有意退行 0 件**（構造的に不変が期待される） |
| **非退行③** | `fallback_rate` / `dispatch_failure_rate` | 0.0 / 0.000522 | fallback = 0.0，**dispatch_failure ≤ 0.005**（送出数が増えるためここが binding になりうる） |
| **非退行④** | `mean_duration_ms` | 2217.1 | **≤ 2660.5（+20% 以内）**．予測 ≈2320 |
| **非退行⑤** | ECE | 0.025448 | **≤ 0.08**（rank 1 の confidence のみに依存するため不変が期待される） |
| 報告のみ | `compound_domain_jaccard_mean` / 複合 k 分布 / 全 1,915 行 mean dispatched | 0.458072 / (310,31,74) / 1.4078 | 予測 0.456466 / (278,21,116) / 1.6010 と対比して記録 |
| 報告のみ | `answer_quality_accuracy` / `end_to_end_accuracy` | 0.572667 / 0.351958 | **3SD=2.6pt のノイズ床**（success_criteria (5)）を適用．超えない限り有意と判定しない |
| 報告のみ | 全 1,915 行 top1 / 既存 1,600 行部分集合 top1 / 複合 415 行 top1 | 0.792167 / 0.786875 / 0.816867 | 毎回併記 |
| 報告のみ | Random / BestSingle / Oracle | 0.121671 / 0.12689 / 1.0 | success_criteria (3) により毎回併記 |

**主基準を top1 から `compound_domain_set_recall` へ置き換える正当化**

1. 上記「top1 と下流指標が構造的に動かない理由」のとおり，`max_confidence` 集約の下で `selected_domain` は rank 1 のみで決まり gap 閾値と独立である．**top1 を主基準に据えれば，本レバーは定義上「効果なし」としか判定できない**．
2. success_criteria (1) は主基準を top1 の McNemar と定めているが，これは「ルーティング先の選択」を動かすレバーを前提にした規定である．本レバーが動かすのは「何件へ送るか」であり，対応する軸①の指標は `compound_domain_set_recall`（複合設問の被覆）である．**success_criteria (1) の趣旨（同一問題集合での対比較・効果量の併記）は維持し，対象指標だけを置き換える**．検定は 830 スロットの対ペア（喪失 a / 獲得 b）に対する符号検定で行い，検出限界 `1.96·sqrt(n_d)/830` を併記する（予測 n_d=35 なら 1.40pt）．
3. top1 は非退行①として**「動かないこと」を確認する側**に回す．これは success_criteria (6)（主要指標が完全一致したら invalid を疑う）の裏返しで，**本レバーに限っては top1 が完全一致することこそ正常**である．したがって invalid 判定は top1 ではなく `compound_mean_dispatched_count` で行う（下記）．
4. 予算制約を非退行ではなく**主基準の一部**に置いた．被覆は閾値を上げれば単調に買えるため（Q2），予算制約を外した「set_recall 改善」は主張として空である．

**判定規則（事前登録）**

- **adopted**: 主基準 (i)(ii) と予算制約（`compound_mean_dispatched_count` ≤ 1.90）を満たし，非退行①〜⑤をすべて満たす．
- **partial**: 予算制約と非退行①〜⑤は満たすが，set_recall が Iter79 の 0.548193 は上回るものの Iter82 比 +3.0pt に届かない（0.548193 < 実測 < 0.5698）．**予測 0.581928 からの乖離の原因**（`dispatch_failure` の増加，probe confidence の実行時変動など）を必ず特定して記録したうえで，`dispatch_gap_threshold` を 0.29 へ復元し，レバーは収束扱いとする．
- **no_effect**: set_recall が Iter82 実測 0.539759 と ±0.5pt 以内．この場合は「閾値が効いていない」疑いが濃いので，**先に F2/F3/F3' を再検証**してから結論を書く（invalid との切り分け）．
- **rejected**: set_recall が Iter79 の 0.548193 以下，または予算制約違反（`compound_mean_dispatched_count` > 1.90），または非退行①〜⑤のいずれかに違反．
- **invalid（実験不成立）**: F1・F2・F3 のいずれか不合格，`question_count != 1915`，`compound_domain_question_count != 415`，**または `compound_mean_dispatched_count` が Iter82 実測 1.6096 と小数点以下 4 桁まで完全一致**（＝ config の 1 行が実行パスに到達していない．success_criteria (6) の本レバー版），または `top1_accuracy` が 2 行以上（0.11pt 超）動いた場合（構造的に起こりえないため別要因の混入を疑う）．
- **復元手順（partial / no_effect / rejected 共通）**: `config.yaml:90` を `dispatch_gap_threshold: 0.29` へ戻し（Iter58 の説明コメントは残す），`mise run deploy` を再実行して全 10 ノードで値と smoke_check を確認する．**artifact は触らない**（本反復では再訓練していないので `1cfcd3d8...` のまま）．`data/MANIFEST.md` の artifact 行は本反復では変更しない．

**実験手順**

1. **前提確認**: `wc -l data/dataset.jsonl` = 1915，`sha256sum models/domain_classifier.joblib` = `1cfcd3d8...`，`config.yaml` の `embedding_view_concat: true` と `embedding_instruction` が存在すること．
2. **変更**: `config.yaml:90` を `dispatch_gap_threshold: 0.36` へ．直前のコメントブロック（L72-89）に **Iter83 で 0.29 から再較正した経緯・掃引の出典（`results/20260927_050049/`）・予算整合の根拠**を追記する．**`.py` は 1 行も変えない．**
3. **F1**: `git diff --stat` が `config.yaml` のみであることを確認．`uv run ruff check .`（pre-existing 23 件のみ）・`uv run pytest tests/`（pre-existing 9 件 FAIL は B122．新規失敗 0 件）．
4. `mise run setup`（**直後に `uv sync --extra research` で research extra を復旧．B118 落とし穴 2**）→ `wc -l data/dataset.jsonl` = 1915 を再確認．
5. `mise run deploy` → **F2** と smoke_check の pass を確認．
6. **先頭 20 問の予備実行 → F3**．`dispatched_domains` をオフライン replay（gt=0.36）と突き合わせ，**不一致なら本走に進まない**．
7. **wafl500〜509 で 1,915 問のフルスペック本走を 1 回**（絶対条件 A）．起動直後に `state.json` を `status=waiting_experiment`・`experiment_dir`・`experiment_deadline`（開始時刻 + 150×60 + 600 秒）へ更新．**送出数が増えるため所要時間は Iter82 の 176 分より延びうる**点に注意．`mise run analyze -- <timestamp>` まで実施（**引数なし実行は `results/iter45_preliminary/` を誤選択する．B118 落とし穴 1**）．
8. 指標を「全 1,915 行」「既存 1,600 行部分集合」「複合 415 行」の 3 通りで算出し，**Iter82 `results/20260927_050049/` と id ペアリング**して比較する．複合被覆は 830 スロットの対ペア（喪失 / 獲得）と符号検定を報告する．**F3'（`compound_mean_dispatched_count` が 1.6096 と異なること）を必ず確認する．**
9. **G2 の予測値と本走実測の一致／不一致を必ず記録する**（Iter82 の G2' と同じ扱い．予測手続きの妥当性検証として蓄積する）．
10. `data/MANIFEST.md` は artifact を変えないため更新不要．ただし journal に本走ディレクトリと gt 値を記録する．

**期待効果とリスク**

期待効果は「Iter82 で採用した連結埋め込みの真価を，古い前提（Iter58 に旧分布上で決めた T=0.29）から解放して測り直す」ことである．同じ送出コストで複合設問の被覆が Iter79 比 +3.37pt になるという予測が当たれば，「特徴量を変えたら閾値を較正し直す」という手続きを本研究の標準工程に組み込む根拠になる（3 回目の同型観測）．リスクは 3 つ．(1) **送出数が +13.7%（全 1,915 行 1.408→1.601）増えることによる所要時間と `dispatch_failure_rate` の悪化**（非退行③④で監視．予測では余裕があるが実測でしか分からない）．(2) **閾値を評価集合上で選んでいること**（G3 の split-half で符号の安定性は確認したが，独立な hold-out ではない．限界として明記する）．(3) **`k=4` の行が 229→367 行へ増える**ため，複数ノードへの同時送出が増えてノード側の負荷が上がる（`dispatch_timeout_s` に対する余裕が縮む）．

### 実験 (Iter83)

**変更・検証（F1〜F3'，計画で事前登録した内容に全て一致）**

- **F1（変更の最小性）PASS**: `git diff --stat` は `config.yaml`（+18/-1 行．全て `dispatch_gap_threshold: 0.29→0.36` の 1 行変更と直前コメントブロックへの追記）のみ．`.py` の変更 0 件．`uv run ruff check .` は既存 23 件のまま（新規 0 件）．`uv run pytest tests/` は 309 passed / 既存 9 件 FAIL（`tests/test_build_dataset.py`，backlog B122，未変更）のまま，新規失敗 0 件．
- `mise run setup` → `uv sync --extra research`（B118 落とし穴 2 の手順どおり）→ `mise run deploy`．smoke_check は全 10 ノードで `config.yaml matches deployed container` を確認し pass．
- **F2（配布）PASS**: 全 10 ノード（wafl500〜509）で `grep '^dispatch_gap_threshold:' $REMOTE_DIR/config.yaml` が `0.36`．分類器 artifact sha256 は `1cfcd3d836c5b5b421b8a47a487c3fb3ba996dd54aadcebae688cdfc8bcd0a48` でローカル・wafl500 とも一致（不変）．
- **F3（実行時経路）PASS**: `data/dataset.jsonl` 先頭 20 行を `data/f3_20_iter83.jsonl` として作成し `docker cp` でコンテナへ投入，`mise run start -- --dataset data/f3_20_iter83.jsonl --output f3_20_iter83.jsonl`（`results/20260927_070120/`）で予備実行．同 20 行の `probe_candidates` を用いたオフライン replay（T=0.36, max_k=4，`aggregator.select_dispatch_targets()` の gap escalation 規則をそのまま再実装）と `dispatched_domains` を突き合わせ，**20/20 完全一致**（k=1 の 16 行・k=4 の 4 行を含む）．一時ファイルは確認後にローカルから削除済み．
- 本走: `mise run start -- --dataset data/dataset.jsonl --output results.jsonl`．実験ディレクトリ **`results/20260927_070239/`**．起動時に `state.json` を `status=waiting_experiment`・`experiment_dir=results/20260927_070239`・`experiment_deadline=1790469794`（起動 1790460194 + 150×60 + 600 秒）へ更新．所要 **約 34 分**（1790460159 起動 〜 1790462279 完了，`run_experiment.log` の `completed 1915 questions` 到達）．Iter82 の 176 分より大幅に短い（要因は未調査．同一 gt=0.29 ではなく gt=0.36 で送出数が増えているにもかかわらず短縮しており，ノード側の GPU 競合状況・キャッシュ効果等の外的要因が寄与した可能性がある．**分析フェーズでの解釈対象として申し送る**）．`mise run analyze -- 20260927_070239` を実施し全 10 ノードのログ回収・axis2/3 指標算出まで完了．
- **F3'（invalid 検出）PASS**: `compound_mean_dispatched_count` 実測 **1.889156626506024** は Iter82 実測 1.6096385542 と明確に異なる（config の 1 行が実行パスに到達している）．

**取得した主要メトリクス（`metrics.py --json`，`results/20260927_070239/results.jsonl`）**

| 指標 | Iter82 基準線 | Iter79 参照 | **Iter83 事前登録予測** | **Iter83 実測** |
|---|---|---|---|---|
| `compound_domain_set_recall` | 0.539759 | 0.548193 | 0.581928 | **0.5819277108433735** |
| `compound_covered_domain_count`（分母 830） | 448 | 455 | 483 | **483** |
| `compound_mean_dispatched_count` | 1.6096385542 | 1.8795180723 | 1.8892 | **1.889156626506024** |
| `compound_domain_jaccard_mean` | 0.458072 | — | 0.456466 | **0.4564658634538148** |
| `top1_accuracy`（全 1,915 行） | 0.792167 | 0.753003 | 0.792167（±0.06pt=1行） | **0.7926892950391645**（Iter82比 +0.0522pt＝ちょうど 1 行分） |
| `single_domain_top1_accuracy`（`single_domain_question_count`=1,500 行） | 0.785333 | — | 予測同様 | **0.786**（Iter82 実測は 0.7853333333333333，計画節記載の 0.786875 とは定義がやや異なる可能性があり要確認） |
| `compound_domain_top1_accuracy`（415 行） | 0.816867 | — | 予測同様 | **0.8168674698795181** |
| `mean_duration_ms` | 2217.1 | 2301.4 | ≈2320 | **2286.826631853786**（analyze 経由の axis23 集計では 2286.932079414838，四捨五入誤差程度の差） |
| `dispatch_failure_rate` | 0.000522 | — | ≤0.005 | **0.0005221932114882506**（不変） |
| `fallback_rate` | 0.0 | — | 0.0 | **0.0** |
| `ece` | 0.025448 | — | ≤0.08（構造的不変予測） | **0.02599932343980014** |
| `brier_score` | — | — | 報告のみ | **0.13758931835315297** |
| `auroc` | — | — | 報告のみ | **0.7819100091827365** |
| `cohens_kappa` | — | — | 報告のみ | **0.762239834086364** |
| `answer_quality_accuracy` | 0.572667 | 0.569333 | ノイズ床内で不変 | **0.5706666666666667** |
| `end_to_end_accuracy` | 0.351958 | 0.335770 | ノイズ床内で不変 | **0.35091383812010446** |

**G2（着地点の事前登録）との一致**: `compound_domain_set_recall`（0.581928 予測 vs 0.5819277108433735 実測，小数点以下 4 桁一致）・`compound_mean_dispatched_count`（1.8892 予測 vs 1.889156626506024 実測）・`compound_domain_jaccard_mean`（0.456466 予測 vs 0.4564658634538148 実測）・`compound_domain_top1_accuracy`（0.816867 予測 vs 0.8168674698795181 実測）は**いずれも 4 桁以上で一致**．`top1_accuracy` は予測どおり「不変（許容幅 1 行）」の範囲に収まり，実際に動いたのは **ちょうど 1 行**（Iter83 0.7926892950391645 − Iter82 0.7921671018276762 = 0.0005221932114883 ≒ 1/1915）で，計画節に記載した「rank1 dispatch 失敗行のみ動きうる」という機序の予測と整合する．`mean_duration_ms` は予測 ≈2320ms に対し実測 2286.8ms（-33ms，予測の線形内挿誤差の範囲）．

**Iter82 との id ペアリング（分析フェーズ向けの準備）**: `results/20260927_050049/results.jsonl`（Iter82）と `results/20260927_070239/results.jsonl`（Iter83）は行数（1,915）・`id` 列の並び順ともに完全一致することを確認済み（`ids_82 == ids_83` が `True`）．したがって行インデックスでの直接ペアリングにより，830 スロット（複合 415 行 × 2 ドメイン）の獲得/喪失対と符号検定は分析フェーズでそのまま既存実装（`scripts/analyze_iter52.py` 等の `mcnemar_2x2` / `wilson_ci` パターン）を再利用して計算できる状態にある．**符号検定の実施・解釈自体は本フェーズでは行わない**．

**異常の有無**: `ollama ps`（wafl500〜502 で確認）に OOM・異常な CPU フォールバックは無し（`qwen3.5:4b-q4_K_M` の 25%/75% CPU/GPU 分割は Iter79 以降と同一パターンで新規異常ではない）．`mise run deploy` の smoke_check は全項目 pass．本走・予備実行とも exit code 0．

**運用上の申し送り（新規）**: 予備 20 問実行時，`data/` はイメージビルド時に COPY されるため，ビルド後に作成したデータファイルはコンテナ内に存在せず `FileNotFoundError` になった（1 回目の予備実行が失敗）．`docker cp` でコンテナへ直接投入することで解決した．journal archive の Iter69/73 等では「先頭 20 行を一時ファイルとして投入」とだけ記載されており，この「イメージ COPY 後に作った一時データはコンテナに無い」という具体的な失敗モードは今回初めて踏んだため，次回の予備実行担当者向けに明記する．

### Iteration 83 実行済み（分析・考察）

**判定: adopted**（事前登録の判定規則を結果を見る前の定義のまま機械的に適用．条件の緩和は行っていない）

**1. 事前登録条件の機械的な当てはめ**

| 区分 | 指標 | 合格条件（事前登録） | Iter83 実測 | 判定 |
|---|---|---|---|---|
| 主基準 (i) | `compound_domain_set_recall` | > Iter79 の 0.548193 | **0.5819277108** | PASS（+3.373pt） |
| 主基準 (ii) | 同上 | ≥ 0.5698（Iter82 比 +3.0pt） | 同上 | PASS（**+4.217pt**） |
| 主基準の予算制約 | `compound_mean_dispatched_count` | ≤ 1.90 | **1.8891566265** | PASS（Iter79 の 1.8795 比 +0.52%） |
| 非退行① | 全 1,915 行 `top1_accuracy` | ≥ 0.791645 | **0.7926892950**（+0.0522pt＝正味 +1 行） | PASS |
| 非退行② | per-domain recall/precision 計 20 指標（BH q=0.05） | 有意退行 0 件 | **有意 0 件**（そもそも値が動いたのは 4 指標のみ，いずれも discordant ≤2 行，最小 p=0.4795） | PASS |
| 非退行③ | `fallback_rate` / `dispatch_failure_rate` | 0.0 / ≤0.005 | **0.0** / **0.000522**（1 行） | PASS |
| 非退行④ | `mean_duration_ms` | ≤ 2660.5 | **2286.83** | PASS |
| 非退行⑤ | `ece` | ≤ 0.08 | **0.0259993** | PASS |
| invalid 検出 | `compound_mean_dispatched_count` が 1.6096 と 4 桁一致 | 不一致であること | 1.8892（明確に不一致） | invalid でない |
| invalid 検出 | `top1_accuracy` が 2 行以上動く | 1 行以内 | 正味 1 行 | invalid でない |

参照点（success_criteria (3)）: Random 0.121671 / BestSingle 0.126893（legal）/ Oracle 1.0．top1 Wilson 95% CI = [0.773956, 0.810251]．
報告のみ: `compound_domain_jaccard_mean` 0.4564659（予測 0.456466 と一致・微減），複合 415 行の k 分布 (k=1/2/4) = **278 / 21 / 116**（予測と完全一致，Iter82 は 310/31/74），全 1,915 行 mean dispatched 1.601044（予測 1.6010），`answer_quality_accuracy` 0.570667（-0.200pt），`end_to_end_accuracy` 0.350914（-0.104pt）＝いずれも 3SD=2.6pt のノイズ床内で不変，`brier` 0.137589 / `auroc` 0.781910 / `cohens_kappa` 0.762240，既存 1,600 行部分集合 top1 0.787500，複合 415 行 top1 0.816867（完全不変），single 1,500 行 top1 0.786000．

**2. 830 スロットの対ペアと符号検定（本フェーズで実測）**

Iter82 `results/20260927_050049/` と Iter83 `results/20260927_070239/` を行インデックスでペアリング（id 列完全一致）．複合 415 行 × 2 ドメイン = 830 スロットの被覆を対比較した．

- **獲得 35 / 喪失 0**（両方被覆 448，両方未被覆 347）．**喪失が 1 件も無い**のは gap escalation が閾値に対して単調（k が非減少）であることの直接の帰結で，理論と完全に整合する．
- 符号検定（両側，厳密二項）: n_d=35，**p = 5.82e-11**．計画節に事前登録した予測（獲得 +35 / 喪失 0 / p=5.82e-11）と**完全一致**した．
- 効果量: set_recall 差 +0.042169（95% CI [+0.028198, +0.056139]，paired 差の正規近似）．検出限界 1.96·√35/830 = 1.397pt に対し実測 4.217pt で，**ノイズ幅の 3.0 倍**．軸①（ルーティング系）は決定論的なので success_criteria (5) のノイズ床は適用しない．

**3. 予測（G2）と実測の一致 — 事前登録手続きの検証**

`compound_domain_set_recall`・`compound_mean_dispatched_count`・`compound_domain_jaccard_mean`・複合 k 分布・全体 mean dispatched・獲得/喪失対・符号検定 p の**全項目が本走実測と一致**した．加えて，本走の `probe_candidates` を Iter82 のものと全 1,915 行 × 10 ドメインで突き合わせたところ **最大絶対差 0.0（ビット単位で同一）**であった．probe の確信度は埋め込みと分類器が同一である限り実機を経ても完全に決定論的であり，**`probe_candidates` からのオフライン replay は本走の送出挙動を（dispatch 失敗を除いて）完全に予測できる**ことが 2 回目の確認として確立した．ただし恒久ルールどおり本走を省略はしない．

**4. 申し送り 1（本走所要 176 分 → 34 分）への回答: 「34 分」は測定ミスであり，外的要因による高速化は起きていない**

- `duration_ms` の総和は Iter82 **70.8 分** / Iter83 **73.0 分**である．`run_experiment.py:146-160` は 1 行ずつ**逐次**実行し，`duration_ms`（同 48-50 行）は probe+dispatch を含む ask フロー全体の実測値なので，**総和は壁時計時間の下界**になる．したがって Iter83 が 34 分で終わることは構造的にありえない．
- 実際の経過時間は復元できる．Iter83: 起動 1790460194（`state.json` の `experiment_deadline` 1790469794 − 9600）→ ローカル `results.jsonl` の mtime 1790464555 で **72.7 分**．Iter82: 起動 1790452839（journal 記載）→ mtime 1790457103 で **71.1 分**．**どちらも duration_ms 総和と 0.4% 以内で一致する．**
- すなわち両走とも約 71〜73 分であり，Iter82 の「176 分」も Iter83 の「34 分」も**壁時計の測定窓の取り違え**である（前者は deploy/analyze を含み，後者はポーリングログの一部区間しか見ていないと考えられる）．**外的要因（GPU 競合等）による高速化という仮説は棄却する．**
- `mean_duration_ms` への交絡も無いことを直接示せた．k 別の平均所要は k=1: 1968.6→1944.9ms，k=2: 3345.4→3747.5ms（n=50 と少なく振れる），k=4: 3481.3→3483.5ms とほぼ不変で，**Iter83 の所要を Iter82 の k 構成比で標準化すると 2217.4ms（Iter82 実測 2217.1ms との差 0.3ms = 0.01%）**．+3.1% の増加は**全て送出数の構成比シフトで説明され，1 問あたりの実コストは変わっていない**．非退行④の判定は健全である．

**5. 申し送り 2（`single_domain_top1_accuracy` 0.785333 vs 0.786875）への回答: 両方正しく，別の集合を指す**

- 0.785333 = **`single_domain_top1_accuracy`**（複合を除く single 1,500 行．1178/1500）．
- 0.786875 = **「既存 1,600 行部分集合」top1**（single 1,500 行 + `compound-001`〜`compound-100` の計 1,600 行．Iter78 の拡充前の評価集合と同一．1259/1600）．
- 計画節の表がこの 2 つを並記していたため実験フェーズが同一指標と誤認した．**定義の齟齬ではなく表記の紛れ**である．Iter83 実測は前者 0.786000，後者 0.787500．以後は必ず「single 1,500 行」「既存 1,600 行部分集合」と母数を明記する．

**6. top1 が動いた 3 行の精査（計画の構造予測に対する補正）**

top1_accuracy の変化は正味 +1 行だが，**対ペアの不一致は 3 行**あった．内訳は次のとおりで，**いずれも本レバーとは無関係な一過性のノード側失敗**である．

| id | Iter82 | Iter83 | 機序 |
|---|---|---|---|
| `medical-033` | dispatch 失敗（`selected_domain=None`） | 成功 | 一過性のタイムアウト．k は両走とも 1 |
| `medical-007` | 成功 | dispatch 失敗 | 同上（失敗行が 1 行から 1 行へ「移動」しただけ） |
| `natural_science-079` | `medical` を選択（rank2，conf 0.2889） | `natural_science` を選択（rank1，conf 0.4117） | **両走とも同一の 4 ドメインへ送出**．Iter82 では rank1 ノードの `/dispatch` 応答だけが欠落し，`select_best_dispatch_response()` が生き残った rank2 を選んだ．`dispatch_failed` は全滅時にしか立たないため**この部分失敗はフラグに現れない** |

計画の「top1 は最大 1 行しか動きえない」は，`dispatch_failure_rate`（全滅）だけを数えていた点で不完全だった．正しくは**「k≥2 の行では rank1 の部分失敗によって rank2 が選ばれうる」**ので，動きうる行数は k≥2 の行数（Iter83 では 417 行）に一過性失敗率を掛けた分だけある．本走では 1 行（0.052pt）に留まったが，**±1 行という許容幅は原理的には狭すぎた**．なお本レバーは k≥2 の行を 323→417 行へ増やすので，この経路の露出はわずかに増える（今回は実測で増えていない）．

**7. 考察 — 本反復が示したこと，示していないこと**

- **示したこと**: 送出予算を Iter79 と同水準（1.8795 → 1.8892，+0.52%）に固定したまま，複合設問の被覆が **+3.373pt**（Iter79 比）改善した．すなわち Iter82 の連結埋め込みは，旧分布上で決めた `T=0.29` の下では被覆をむしろ下げて見えていたが（0.548193→0.539759），**較正し直せば同じコストでより良い候補集合を出せる**．喪失 0 件という非対称な内訳は，改善が「たまたま入れ替わった」のではなく単調な包含関係の拡大であることを示す．
- **示していないこと（過剰一般化の禁止）**: (a) 本レバーは `selected_domain` を動かさないので，**top1_accuracy の改善には一切寄与しない**（実際 +1 行は一過性失敗に由来）．(b) 閾値は複合 415 行＝評価集合そのもので選んでおり，**独立な hold-out は無い**．G3 の split-half（A→B: 0.5290→0.5942，B→A: 0.5505→0.5649）で符号の安定性は確認したが，効果量の大きさは楽観側に偏りうる．(c) 予算整合点を選んだため「較正の効果」を単離できたが，**より大きい予算でどこまで被覆が伸びるかは別問題**である（gt=0.40 で 0.598795）．
- **レバーの扱い**: `dispatch_gap_threshold_recalibration` は **adopted・収束**とする．`config.yaml` の `dispatch_gap_threshold: 0.36` はそのまま残す（復元しない）．今後の基準線は **`results/20260927_070239/`** である．

**8. 学び（次の自分向け）**

1. **特徴量を変えるレバーの後には必ず閾値較正の反復を 1 回挟む**．Iter81・Iter82・Iter83 と 3 回続けて同型の現象を観測した（確信度分布が動けば固定閾値の意味も動く）．文献上も標準の手続き（Zellinger & Kim 2025，TREACLE 2024，conformal prediction）であり，**本研究の標準工程に組み込む**．次に埋め込み・分類器・訓練データを動かすレバー（例: `cross_domain_training_data_augmentation`）を採用したら，その直後に同じ較正を再度行うこと．
2. **単調なハイパラを動かすレバーでは「効果」は必ず予算を揃えて測る**．gap 閾値は上げれば被覆が単調に買えるので，予算制約を非退行ではなく**主基準の一部**に事前登録した設計は機能した（今回 0.40 を採っていれば +5.06pt と見栄えは良いが交絡していた）．同型のレバー（top_k，候補閾値など）でも同じ枠組みを使う．
3. **`dispatch_failed` フラグは「全送出先が失敗した」ときしか立たない．部分失敗は results.jsonl から直接は分からず，`selected_domain` が rank1 でないことでしか検出できない**（本フェーズで `natural_science-079` として初めて具体的に観測）．k を増やすレバーの非退行条件を「top1 が ±1 行」と書くのは狭すぎる．今後は **「rank1 以外が選ばれた行数」を補助指標として数える**こと．
4. **壁時計の所要時間は `duration_ms` の総和と突き合わせて検算する**．逐次実行なので総和は下界であり，これを下回る報告値は測定窓の取り違えである（Iter82「176 分」/ Iter83「34 分」は両方とも誤り．実際は 71.1 分 / 72.7 分）．`run_experiment.py` が開始・終了の壁時計時刻を results 側に記録していないことが根本原因で，将来の計測のために記録を足す余地がある（backlog B132 に記載）．
5. **所要時間の比較は k 構成比で標準化してから行う**．今回 +3.1% の増加は 100% が k 構成比シフトによるもので，1 問あたりの実コストは 0.01% しか動いていなかった．生の平均だけを見ると「遅くなった」と誤読する．
6. **`probe_candidates` はビット単位で再現する**（Iter82 と Iter83 で全 1,915 行 × 10 ドメインの最大絶対差 0.0）．送出段だけを動かすレバーは実機を使わずに完全に事前予測でき，事前登録の信頼度は高い．本走は恒久ルールにより省略しないが，予測と実測の乖離が出たら**まずノード側の一過性失敗を疑う**のが正しい順序である．

---

## Iteration 82: prefix 有無の埋め込み連結（2048 次元）による medical 退行の回避

### 調査 (Iter82)

本反復のレバーは backlog B128（Iter81 分析フェーズ）で `embedding_view_concatenation` = `prefix_and_noprefix_concat` に確定済みで，レバー選定の裁量は無い．調査の問いは 3 つ．**(Q1) 埋め込みビューの連結は先行研究上どう位置付けられ，効果と次元増加のリスクはどう扱われているか．(Q2) 本リポジトリの 1,427 行 × 2048 次元という条件で，正規化・スケーリング・L2 正則化はどう働くか（定量）．(Q3) 訓練側と実行時側で同一の連結順序・前処理が適用される到達条件はどこか．**

**Q1: 「複数ビューの生ベクトル連結」は retrieval では効果が実証されている．ただし本件は同一モデルの instruction 2 ビューで，直接の先行例は見つからなかった**

- **Compressed Concatenation of Small Embedding Models**（arXiv:2510.04626v1，2025-10-06，<https://arxiv.org/html/2510.04626v1>，CIKM 関連 DOI 10.1145/3746252.3760831）は「**複数の小型埋め込みモデルの生ベクトルを連結すると，単体の大型モデルを上回りうる**」ことを MTEB retrieval 部分集合で示した一次情報である．実測（同論文 Table 2，nDCG@10 平均）: Arctic-m 単体 0.50718 / bge-small 単体 0.49088 に対し **連結 [M1,S5] は 0.52033** と両参加モデルを上回る．別の組 [M1,S3] も 0.51784 で両者を上回る．公開モデル `PaDaS-Lab/arctic-m-bge-small`（142M・1152 次元）は当時の legacy MTEB リーダーボードで 56.5 を記録し，**gte-Qwen2-7B（56.24）を上回った**と報告されている（事実）．
- 同論文は「**素朴な連結は次元が高くなる**」ことを明示的な課題として扱い，Matryoshka Representation Learning 損失で訓練した軽量デコーダで 384/256 次元へ圧縮する（4 モデル連結 + 量子化で 48 倍圧縮・性能の 89% を保持）．**つまり次元増加への対処は「圧縮」であって「正則化」ではない**．ただし同論文の評価はコサイン類似度による retrieval であり，**教師あり線形分類器の入力として使う場合の過学習は扱っていない**（本研究との差）．
- 実務ガイド（Zilliz <https://zilliz.com/ai-faq/how-can-you-combine-or-ensemble-multiple-sentence-transformer-models-or-embeddings-to-potentially-improve-performance-on-a-task> / Milvus <https://milvus.io/ai-quick-reference/how-can-you-combine-or-ensemble-multiple-sentence-transformer-models-or-embeddings-to-potentially-improve-performance-on-a-task>，いずれも 2026-09-27 確認）は同じ設計を **late fusion**（連結ベクトルを logistic regression 等のメタ分類器へ入力する）として整理し，「連結は各モデル固有の特徴を保つが次元が増える．curse of dimensionality への対処として PCA 等の次元削減を使うか，計算資源が許すなら連結ベクトルをそのまま使う」と述べる．**本研究は後者（そのまま使う）を選ぶ立場であり，その妥当性は Q2 の実測で判断する．**
- 表形式データでの傍証: arXiv:2603.17737（<https://arxiv.org/html/2603.17737>）は「**一般に，埋め込みを連結する方が元の列を埋め込みで置き換えるより良い**」と報告している（元表現を捨てずに足す方が良い，という本反復の仮説と同方向．ただしドメインは異なる）．
- **instruction を変えた同一モデルの複数ビューを連結する**という本反復の形式そのものの先行研究は見つからなかった（推測と事実の区別: 未確認）．近縁の概念は 2 つある．(a) **prompt ensembling**（同一入力に複数プロンプトを当てて結果を統合する．頑健性が上がるとの報告，例 <https://arxiv.org/html/2502.00847v1>），(b) instruction-following embedding の研究（**InBedder**，arXiv:2402.09642，<https://arxiv.org/html/2402.09642v1>）は「instruction と入力を連結して埋め込む（prompt 方式）」を最も素朴な方式として定式化し，**同一コーパスに異なる instruction を当てるとクラスタリング結果が変わる**ことを定性的に示している．つまり **instruction を変えると表現が実質的に別ビューになる**という前提自体は支持される．
- **本研究への帰結**: 連結の効果は「2 ビューが相補的であること」に依存する．Iter81 の実測（journal「Iteration 81 実行済み」学び 2: medical を取りこぼした 23 行のうち 19 行で medical は prefix 版の rank 2 に残り，固定 top-2 なら medical recall は 0.9046 → 0.8963 とほぼ不変）は，**prefix 版が medical の信号を失ったのではなく順位を入れ替えただけ**であることを示しており，相補性の直接的な証拠になっている．

**Q2: 2 ビューとも既に L2 正規化済み（ノルム 1.000）で，スケーリングのレバーは実質存在しない．p≫n は既に 1024 次元でも成立しており，連結は CV で +1.33pt（実測）**

本フェーズで `data/embcache_qwen3-embedding_0.6b.npy`（P0）と `data/embcache_qwen3-embedding_0.6b__p1.npy`（P1，Iter81 で選定した文言）を用い，**評価集合 `data/dataset.jsonl` を一切参照せず** `data/classifier_train.jsonl`（1,427 行）のみで実測した（wafl-ctrl5 も不要．キャッシュ済み配列の計算のみ）．

- **正規化**: Ollama `/api/embeddings` が返すベクトルは **両ビューとも L2 ノルム = 1.0000000（std 2.75e-08）** である（実測）．したがって **2 ビューのスケールは既に揃っており，連結前の per-view 正規化・標準化は不要**．「片方のビューが大きさで支配する」という連結の典型的な落とし穴は本件では発生しない（事実）．本リポジトリは `StandardScaler` 等を一切挟まない（`scripts/train_domain_classifier.py:214-218` → `train_classifier()` が生の配列を `LogisticRegression` へ渡す）ため，この性質が効いている．
- **L2 正則化との相互作用（定量）**: 連結後のノルムは √2 ≈ 1.414 になる．`LogisticRegression(max_iter=1000, class_weight=None)` は既定 `C=1.0` の L2 正則化であり，**入力ノルムを √2 倍することは実効的に正則化を弱める（同じ margin をより小さい係数で達成できる）**．そこで「生の連結（ノルム √2）」と「連結後に 1/√2 して単位ノルムへ戻す」を両方 5-fold CV（`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` + `sample_weight`，Iter79〜81 と同一手順）で測った．

| 特徴量 | 次元 | 5-fold CV accuracy | macro-F1 | 訓練適合 accuracy |
|---|---|---|---|---|
| P0（prefix なし・現行基準線） | 1024 | 0.756132 | 0.755724 | 0.8549 |
| P1（prefix あり・Iter81 採用文言） | 1024 | 0.771549 | 0.769284 | 0.8318 |
| **[P0, P1] 生連結** | **2048** | **0.784863** | **0.784167** | 0.8963 |
| [P0, P1] / √2（単位ノルム化） | 2048 | 0.770147 | 0.768377 | — |

（P0 = 0.756132・P1 = 0.771549 は journal の Iter81 実測 0.756123 / 0.771523 と 1e-5 オーダーで一致する．差は `sample_weight` 計算の浮動小数点順序による．）
- **読み取り**: (i) **生連結が最良**で，P1 比 +1.33pt，P0 比 +2.87pt．(ii) **1/√2 で単位ノルム化すると P1 と同水準まで戻る（0.7701）**＝この差は表現ではなく実効正則化強度の差である．したがって **本反復は「生連結（ノルム √2）」を採用し，スケーリングを第 2 のレバーとして導入しない**（正則化強度の掃引は別レバー）．
- **p≫n の定量**: 訓練 1,427 行・10 クラスに対し，多クラス LogisticRegression の係数は 1024 次元で 10,250 個（7.2 params/sample），**2048 次元で 20,490 個（14.4 params/sample）**．臨床統計で広く使われる events-per-variable ≥ 10 の目安（1 クラスあたり約 143 行に対し変数 2048）は **1024 次元の時点で既に大きく破っており，本反復で新たに破るわけではない**．L2 罰則付きロジスティック回帰は p≫n でも数値的に安定に解ける（Ridge の標準的性質．例 <https://mbrenndoerfer.com/writing/ridge-regression-l2-regularization-complete-guide>）．
- **過学習の実測**: 訓練適合 accuracy は P0 0.8549 → 連結 0.8963 で，CV との乖離は **9.9pt → 11.1pt** と 1.2pt 広がるだけである．**Iter80 の bge-m3 で観測された「訓練 100% 適合」型の崩壊は起きていない**（config.yml lever note のリスク記述に対する実測回答）．係数のフロベニウスノルムも 32.98（P0）→ 30.01（連結）とむしろ小さい．
- **per-domain の CV recall（連結が全ドメインで P0 以上）**: business_economics 0.7400→0.8267，computer_science 0.8800→0.9067，education 0.6067→0.6267，general 0.6800→0.7067，history_culture 0.7667→0.7800，legal 0.8442→0.8442，mathematics 0.8867→0.9133，**medical 0.7067→0.7200**，natural_science 0.8400→0.8867，social_science 0.6533→0.6667（いずれも P0 → 連結）．**10 ドメイン全てで P0 以上**であり，P1 単体で P0 を下回っていた education（0.5600）・legal（0.8052）・social_science（0.6267）・history_culture（0.7600）も連結では回復している．
- **重要な留保（事実）**: 上の CV は訓練集合内の値であり，**medical については CV と本走で符号が逆である**．CV 上は P1（0.7467）が P0（0.7067）より良いのに，1,915 行本走では P1 が 0.7178・P0 が 0.7842 だった（Iter81 実測）．**したがって CV の per-domain 値を non-regression の予測に使ってはならない．**per-domain 非退行の予測は評価集合 1,915 行の argmax replay（後述の G2'）でのみ行う．

**Q3: 到達条件（本リポジトリで 6 回繰り返した「config は変えたが実行パスに到達しない」失敗への恒久対策）**

Iter81 で `instruction` 対応コードが既定 None で残置されているため（backlog B128 (a)），本反復の追加実装は「2 回 embed して連結する」経路を訓練側と実行時側の**双方**へ入れることに集約される．片側だけに入れると **Iter36 型（train/eval 不一致で education_recall 0.4588 → 0.0529）**の事故になる．現状のコード（すべて Read で確認済み，行番号付き）:

| # | ファイル:行 | 現状 | 本反復での扱い |
|---|---|---|---|
| 1 | `config.yaml:4` | `embedding_model: qwen3-embedding:0.6b` | 変更しない |
| 2 | `config.yaml:5-13` | `embedding_instruction` は**コメントで説明のみ・キーは意図的に不在**（Iter81 の復元） | P1 文言でキーを復活させ，**新キー `embedding_view_concat: true` を追加**（この 2 キーで 1 つの特徴量ビュー仕様を成す） |
| 3 | `expert_backend.py:140-165` | `OllamaClient.embed(model, text, timeout_s, instruction=None)`．L163 で `prompt = f"Instruct: {instruction}\nQuery: {text}" if instruction else text` | **変更しない**．新設する連結ヘルパがこのメソッドを 2 回呼ぶ |
| 4 | `expert_backend.py`（新設） | — | `async def embed_query_views(client, model, text, instruction=None, concat_views=False) -> list[float]` を追加．`concat_views` が False なら従来どおり 1 回 embed，True なら **`embed(model, text)`（prefix なし）の結果に `embed(model, text, instruction=instruction)` の結果を後ろから連結**する．**連結順序は `[prefix なし, prefix あり]` に固定**（G1 CV の `np.hstack([P0, P1])` と同一順序）．`concat_views=True` かつ `instruction` が None の場合は `ValueError` を送出する（設定の取り違えを静かに通さない） |
| 5 | `node.py:202-204` | `query_embedding = await ollama_client.embed(config["embedding_model"], query, instruction=config.get("embedding_instruction"))` | `embed_query_views(...) `へ差し替え，`concat_views=config.get("embedding_view_concat", False)` を渡す（**実行時側**） |
| 6 | `scripts/train_domain_classifier.py:99-152`（実 embed は L149），`:206-217`（`_train_and_save`），`:252-259`（CLI `--embedding-instruction`），`:262-268`（`main()` の受け渡し） | `build_training_features()` が `ollama_client.embed(embedding_model, row["query"], instruction=instruction)` を呼ぶ | L149 を `embed_query_views(...)` へ差し替え，`concat_views` 引数を `build_training_features` / `_train_and_save` へ通し，CLI に `--embedding-view-concat`（`store_true`）を追加（**訓練側．Q3 の必須条件**） |
| 7 | `tools/smoke_check.py:170-172` | `ollama_client.embed(embedding_model, SMOKE_QUERY, instruction=config.get("embedding_instruction"))` の結果を `/probe` へ送る | 同じ `embed_query_views(...)` へ差し替える．**しないと deploy 後の smoke_check が 1024 次元を 2048 次元の分類器へ送り失敗する**（＝この失敗は検知装置として機能する） |
| 8 | `scripts/screen_embedding_models.py:64-120` | Iter81 の P0〜P3 比較 | G1 用に候補を `{p0, p1, concat}` へ差し替える．**キャッシュ名に候補識別子を含める仕組み（`_cache_path()`，L98-112）は維持**．連結候補はキャッシュを計算せず P0/P1 キャッシュの `np.hstack` で構成する（再 embed 不要） |
| 9 | `http_server.py:364-368` | `estimate_confidence_classifier(state.domain_classifier, state.domain, body.query_embedding)` へ **依頼者が計算した query_embedding をそのまま渡す** | **変更不要**．2048 次元がそのまま分類器へ届く |
| 10 | `http_server.py:403-405` | 起動時に `state.domain_embedding`（ドメイン名の埋め込み・1024 次元）を計算 | **変更しない**．`routing_method=supervised_classifier` では `http_server.py:354-362`（`ROUTING_METHOD_EMBEDDING` 分岐）に到達しないため未使用 |
| 11 | `classifier.py:51-70` | `classifier.predict_proba([query_embedding])` | **変更不要**．`n_features_in_`=2048 の artifact に 1024 次元を渡せば sklearn が例外を送出する＝**次元不一致は静かに劣化せず必ず露見する**（Iter81 の最大の運用リスクだった「静かに劣化する」性質が本反復では解消している） |
| 12 | `models/domain_classifier.joblib` | 現行 sha256 `21e16ec6...`，`n_features_in_`=1024（本フェーズで実測確認済み） | 連結特徴量で再訓練して差し替え．旧版を `models/domain_classifier_pre_iter82_noconcat.joblib` へ `cp` 退避 |

### 計画 (Iter82)

**単一レバー**

`embedding_view_concatenation` = **`prefix_and_noprefix_concat`**．分類器の入力特徴量を「prefix なし埋め込み（1024 次元）」から「**prefix なし ⊕ prefix あり（P1 文言）の 2048 次元**」へ変えること**だけ**を行う．埋め込みモデル（`qwen3-embedding:0.6b`）・instruction 文言（Iter81 の G1 で選定済みの P1，再探索しない）・分類器のハイパラ・訓練データ・評価集合・ルーティング設定は一切変えない．連結に構造的に付随する分類器の再訓練は，Iter79〜81 で確立したとおり別レバーとは数えない．**連結後のスケーリング（1/√2 等）は導入しない**（Q2 の実測で劣るため，かつ第 2 のレバーになるため）．

**固定する構成（直近の最良構成 = Iter79 基準線．Iter81 は rejected で復元済み）**

`config.yaml` の `embedding_model=qwen3-embedding:0.6b`・`routing_method=supervised_classifier`・`confidence_threshold=0.0`・`dispatch_candidate_threshold=0.0`・`dispatch_top_k=2`・`dispatch_gap_threshold=0.29`・`dispatch_gap_max_k=4`・`aggregation_method`・`judge_model`・`classifier_model_path`・各ノードの `light_model=qwen3.5:4b-q4_K_M`／`expert_model=expert-mesh-*-lora`・`probe_timeout_s`／`dispatch_timeout_s`，`data/dataset.jsonl`（1,915 行，ビット単位で不変），`data/classifier_train.jsonl`（1,427 行，不変），`scripts/train_domain_classifier.py:190-203` のモデル定義（`LogisticRegression(max_iter=1000, class_weight=None)` + `_extract_sample_weights()` + `CalibratedClassifierCV(method='temperature', ensemble=True)`），`classifier.py`・`aggregator.py`・`metrics.py`・`build_dataset.py`・`docker-compose.yml`．**ドメイン固有の重み付け・閾値による medical の救済は 2026-09-23 恒久運用ルールにより行わない．連結は 10 ドメイン共通の特徴量設計である．**

**仮説（事前登録）**

「prefix あり／なしの 2 ビューは相補的であり（Iter81 学び 2: medical の取りこぼし 23 行中 19 行で medical は prefix 版の rank 2 に残存），両者を連結した 2048 次元を線形分類器へ与えると，**prefix 単体で得た全体改善（+3.655pt）の大半を保ちつつ medical_recall の退行を解消できる**．G1 CV では連結が P1 を +1.33pt 上回り，per-domain CV recall も 10 ドメイン全てで P0 以上であることを実測済みである．**着地点予測: 全 1,915 行 top1 は 0.775〜0.795（基準線 0.753003 に対し +2.2〜+4.2pt），medical_recall は基準線 0.7842 に対し -2pt 〜 +1pt の範囲で BH 補正後の有意退行なし**．リスクは 2048 次元 × 1,427 行の過学習だが，訓練適合と CV の乖離は 9.9pt → 11.1pt の増加に留まり，Iter80 の bge-m3 型の崩壊は観測されていない．」

**レバーを読むコード行と，そこへ到達する条件（本リポジトリで 6 回繰り返した失敗への恒久対策．省略不可）**

| 経路 | レバーを読む行 | 到達条件 | 到達確認の手段 |
|---|---|---|---|
| **訓練側** | `scripts/train_domain_classifier.py:149`（`build_training_features()` 内の embed 呼び出し）→ 新設 `expert_backend.embed_query_views()` → `expert_backend.OllamaClient.embed()` L163 | CLI に `--embedding-instruction '<P1 文言>' --embedding-view-concat` を**両方**渡したときのみ 2 ビュー連結になる．片方だけなら `ValueError`（新設ヘルパの検証）で落ちる | 生成された artifact の **`n_features_in_` == 2048**．これが 1024 なら訓練側に到達していない（**G2-a**） |
| **実行時側（本レバーの本体）** | `node.py:202-204` → 新設 `expert_backend.embed_query_views()`（`concat_views=config.get("embedding_view_concat", False)`，`instruction=config.get("embedding_instruction")`） | 依頼者ノード wafl500 が `config.yaml` の 2 キーを読めること．`mise.toml` L67 の `rsync config.yaml` が全ノードへ配る | **F2**（全 10 ノードで 2 キーの存在と値を `grep`）と **F3**（先頭 20 問の予備実行）．次元不一致なら `classifier.py:69` の `predict_proba` が例外を送出し `/probe` が 500 を返すため，**実行時側の未適用は必ず露見する**（Iter81 と異なり静かには劣化しない） |
| **訓練側と実行時側の一致（Iter36 型事故の防止）** | 両者が同一の `embed_query_views()` を呼び，**連結順序は実装内で `[prefix なし, prefix あり]` に固定**（呼び出し側から順序を指定できるようにしない） | 順序を引数化しないことが一致の保証である．前処理も両側で同一（どちらも Ollama の生ベクトルをそのまま連結．正規化・標準化は両側とも行わない） | **単体テスト**（`tests/` に追加）で `embed_query_views()` の戻り値が `[plain..., instructed...]` の順であることを検証．加えて **F3**（実行時の `selected_domain` とオフライン replay の argmax が 20/20 一致）．順序が逆なら F3 は必ず不一致になる |
| 設定 → 各ノード | `mise.toml` L67 の `rsync config.yaml` | deploy 実行 | **F2** |
| artifact → 各ノード | `mise.toml` L70-74 の `models/` rsync | deploy 実行 | 全 10 ノードで artifact sha256 一致・`n_features_in_`=2048 |
| 実験 → 指標 | `metrics.py`（無変更） | — | `question_count == 1915` かつ `compound_domain_question_count == 415` |
| **到達しない経路（確認のみ）** | `http_server.py:354-362`（`ROUTING_METHOD_EMBEDDING`）・`http_server.py:403-405`（`domain_embedding`） | `routing_method=supervised_classifier` のため到達しない | 変更しない．`domain_embedding` が 1024 次元のまま残るが未使用 |

**オフラインで先に確認できること / 実機 1,600（1,915）問本走で確認すること（明示的に分離する）**

- **オフラインで確認できる（すべて wafl-ctrl5 のみ．絶対条件 (B) により wafl500〜509 は使わない）**: G1（訓練集合のみの 5-fold CV による特徴量構成の妥当性．本フェーズで既に実測済み），G2-a（artifact の `n_features_in_`），G2-b（1,915 行 argmax replay の discordant n_d），**G2'（1,915 行 replay による per-domain 20 指標の BH 補正の事前予測．本反復の新設ゲート）**，F1（連結ベクトルの次元・順序・ノルム），単体テスト・lint．
- **実機本走でしか確認できない**: `selected_domain` の実分布（replay は argmax の予測であって実行時の dispatch 失敗・タイムアウトを含まない），`mean_duration_ms`（embed 呼び出しが 1 回増える影響），`fallback_rate` / `dispatch_failure_rate`，`compound_domain_set_recall` / `compound_mean_dispatched_count`（`dispatch_gap_threshold=0.29` との相互作用），`answer_quality_accuracy` / `end_to_end_accuracy`，ECE（実測確率分布）．
- **config.yml 冒頭の絶対条件 (A) により，変更を適用したら wafl500〜509 での 1,915 問フルスペック本走を必ず 1 回実施する．G2' がどのような予測を出しても本走を省略しない．**G2' は「本走前に着地点を数値で言語化する事前登録手段」であって本走の代替ではない（2026-09-23 恒久ルール）．G2' が medical_recall の有意退行を予測した場合も本走は実施し，**その一致／不一致自体を G2' の予測妥当性の検証として記録する**（Iter81 学び 4 への回答）．

**事前ゲート（結果を見る前に固定する）**

- **G1（特徴量構成の妥当性．訓練集合のみ・評価集合を参照しない）**: `scripts/screen_embedding_models.py` を `{p0, p1, concat}` の 3 候補へ差し替え，`data/classifier_train.jsonl` のみで 5-fold CV（`random_state=42`）を測る．**合格条件: concat の CV accuracy が P1（0.771549）以上**．本フェーズの事前実測値は **concat 0.784863 / P1 0.771549 / P0 0.756132** であり，rc-executor はこの 3 値を再現できること（±0.001 以内）を確認する．**不一致なら実装が計画と違う**．
- **F1（連結の直接証拠．wafl-ctrl5）**: 同一の 1 文に対し `embed_query_views(..., concat_views=True)` の戻り値が (i) 長さ 2048，(ii) 前半 1024 が prefix なし embed と完全一致，(iii) 後半 1024 が prefix あり embed と完全一致，(iv) 前半・後半それぞれの L2 ノルムが 1.0 ± 1e-5，を満たすこと．**4 つすべて必須．**
- **G2-a（訓練側到達）**: 再訓練後の artifact の `n_features_in_` == 2048，かつ sha256 が旧版 `21e16ec6...` と異なること．
- **G2-b（検出力）**: 旧 artifact（1024・prefix なし）と新 artifact（2048・連結）の `predict_proba` argmax を `data/dataset.jsonl` 1,915 行で replay し，discordant 行数 n_d と必要偏り率 `1.96/sqrt(n_d)` を算出．**n_d ≥ 30 を合格条件**（参考: Iter81 は 276）．
- **G2'（新設・本反復の肝．per-domain 非退行の本走前予測）**: 同じ replay の `selected_domain` 予測から per-domain recall/precision 計 20 指標を計算し，基準線に対し `metrics.py` の検定関数で McNemar + BH 補正（q=0.05）を実施して**本走前に journal へ記録する**．手順は 2 段階．
  - **G2'-a（replay の忠実度）**: **旧** artifact の replay 由来 `selected_domain` が，基準線 `results/20260926_221822/results.jsonl` の実測 `selected_domain` と **99% 以上一致**すること（Iter81 の F3 は 20/20 一致だったので高い一致率が期待される）．**これが満たされなければ G2'-b の予測は信用できない**ため，その旨を明記して予測を「参考値」に格下げする．
  - **G2'-b（予測）**: 新 artifact の replay で medical_recall の予測値と BH 補正後 p 値を記録する．**合格条件は課さない（絶対条件 A により本走は必ず実施する）が，予測値を本走前に journal へ書く**ことを必須とする．
- **F2（配布）**: deploy 後，全 10 ノードで `grep -E '^(embedding_instruction|embedding_view_concat):' $REMOTE_DIR/config.yaml` が選定値と一致し，artifact sha256 が新版と一致すること．
- **F3（実行時経路）**: 先頭 20 問の予備実行の `selected_domain` が，同じ 20 問のオフライン replay（新 artifact × 連結埋め込み）の argmax と **20/20 一致**すること（dispatch 失敗行は除外し分母を明記）．**不一致があれば本走に進まない**（連結順序の取り違え・片側未適用の検出）．

**成功条件・非退行条件（事前登録．結果を見る前に固定する）**

基準線は Iter79 本走 `results/20260926_221822/`（全 1,915 行: top1=0.753003，single_domain_top1=0.749333，compound_domain_top1=0.766265，compound_domain_set_recall=0.548193，kappa=0.721502，ECE=0.032745，fallback=0.0，dispatch_failure=0.000522，mean_duration_ms=2301.4，compound_mean_dispatched_count=1.880，answer_quality=0.569333，end_to_end=0.335770／既存 1,600 行部分集合: top1=0.751250．per-domain は medical_recall=0.7842，computer_science_recall=0.7273，natural_science_recall=0.5931 ほか）．

| 区分 | 指標 | 基準線 | 合格条件 |
|---|---|---|---|
| **F1** | 連結ベクトルの次元・前半／後半の一致・各ノルム | — | 2048 次元，前半＝prefix なし，後半＝prefix あり，各ノルム 1.0±1e-5 |
| **F2** | 全 10 ノードの 2 キーと artifact sha256 | — | 全ノード一致 |
| **F3** | 予備 20 問の `selected_domain` と replay argmax | — | **20/20 一致** |
| **G1** | 5-fold CV accuracy（`classifier_train.jsonl` のみ） | P0=0.756132 / P1=0.771549 | **concat ≥ P1**（事前実測 0.784863 の再現） |
| **G2-a** | artifact の `n_features_in_` | 1024 | **2048** かつ sha256 が `21e16ec6...` と異なる |
| **G2-b** | 旧／新 replay の discordant n_d | 参考: Iter81 は 276 | **n_d ≥ 30** |
| **G2'** | replay による per-domain 20 指標の BH 補正予測 | — | 合格条件なし．**本走前に medical_recall 予測値と BH 後 p 値を journal へ記録すること**（G2'-a の一致率も併記） |
| **主基準（効果）** | 全 1,915 行の `top1_accuracy` | 0.753003 | **McNemar p < 0.05 かつ 点推定 +1.0pt 以上** |
| **非退行①（本反復の主目的）** | per-domain recall/precision 計 20 指標 | Iter79 実測 | **BH 補正（q=0.05）後の有意退行 0 件**．とくに **medical_recall（0.7842）が BH 補正後に有意退行しないこと**が本反復の主目的である |
| **非退行②** | `fallback_rate` / `dispatch_failure_rate` | 0.0 / 0.000522 | fallback = 0.0，dispatch_failure ≤ 0.005 |
| **非退行③** | ECE | 0.032745 | **≤ 0.08** |
| **非退行④** | `mean_duration_ms` | 2301.4 | **≤ 2761（+20% 以内）**．embed 呼び出しが 1 回増えるが，Iter81 の prefix 版は逆に -100.7ms だったので余裕はあると見込む（要実測） |
| 報告のみ | 既存 1,600 行部分集合 top1 | 0.751250 | 判定には用いないが毎回併記 |
| 報告のみ | medical_recall の点推定 | 0.7842 | 判定は BH 補正のみで行う（下記「規則を変えない理由」参照） |
| 報告のみ | `compound_domain_top1` / `single_domain_top1` / `compound_domain_set_recall` / `compound_mean_dispatched_count` | 0.766265 / 0.749333 / 0.548193 / 1.880 | Iter81 では set_recall が -2.17pt 下がったが，学び 1 のとおり原因は `dispatch_gap_threshold=0.29` の相対的なきつさであり埋め込み品質の劣化ではない．本反復でも同型の低下が起こりうるが**判定には用いない** |
| 報告のみ | `answer_quality_accuracy` / `end_to_end_accuracy` | 0.569333 / 0.335770 | 3SD=2.6pt のノイズ床（success_criteria (5)）を適用し，超えない限り有意と判定しない |
| 報告のみ | Random 0.101 / BestSingle 0.109 / Oracle 1.0 | — | success_criteria (3) により毎回併記 |

**主基準と非退行条件の設計の正当化（本反復の主目的は medical_recall の退行解消である）**

1. **非退行①の判定規則を Iter81 から一切変えない**．Iter81 は主基準（+3.655pt，p=4.2e-7）を大きく満たしながら medical_recall の BH 補正後有意退行 1 件で rejected になった．その規則の是非は backlog B128 の「要レビュー」で**人間判断に委ねている未決事項**であり，計画フェーズが自動で緩和すれば「結果を見てから規則を読み替えた」ことになる．**同一の規則で測ってこそ，Iter81 と Iter82 の比較が成立する**．
2. **その上で，本反復は規則を変えずに medical の退行を消しにいく**．これが「非退行条件①を主目的に据える」という本反復の設計思想である．Iter81 の失敗は「prefix なしの情報を捨てた」ことに起因する（学び 2: medical は prefix 版でも rank 2 に残っていた）から，**捨てずに足す**（連結）が直接的な対処になる．
3. **主基準は Iter81 と同一の「+1.0pt 以上かつ p<0.05」に据える**．連結は P1 の改善を保ちつつ medical を救う設計であり，「medical は守れたが全体改善も消えた」という着地は採用に値しない（それは基準線への回帰にすぎない）．なお n_d=100 想定の MDE `1.96·sqrt(n_d)/1915` ≈ 1.02pt であり，+1.0pt はこの検出限界と整合する．
4. **medical_recall に点推定の下限（例 ≥ 0.7542）を追加しない**．追加すれば Iter81 より厳しい規則になり，かつ 1 ドメインだけに固有の判定条件を設けることになって 2026-09-23 恒久運用ルール（ドメイン固有の後付け補正の禁止）の精神に反する．**medical は「20 指標のうちの 1 つ」として同じ BH 補正で扱い，点推定は報告のみとする．**
5. **G2' を非退行①の事前予測として新設する**（Iter81 学び 4）．ただし Q2 の留保のとおり **CV の per-domain 値は予測に使わない**（medical は CV と本走で符号が逆だった）．予測に使えるのは評価集合 1,915 行の replay だけである．

**判定規則（事前登録）**

- **adopted**: 主基準（McNemar p<0.05 かつ +1.0pt 以上）と非退行①②③④をすべて満たす．
- **partial**: 非退行①②③④に違反はないが，点推定が +1.0pt 未満または p ≥ 0.05．採用せず基準線へ復元し，レバーは収束扱いとする．
- **no_effect**: G2-b の n_d < 30（ただし F1・F2・F3 合格）かつ本走の top1 差が ±0.5pt 以内．「実験不成立」ではなく「連結は本構成で効かない」という結論として記録し，基準線へ復元する．
- **rejected**: 点推定が低下，または非退行①で BH 補正後の有意退行 1 件以上，または非退行②③④のいずれかに違反．
- **invalid（実験不成立）**: F1・F2・F3・G2-a のいずれか不合格，`question_count != 1915`，`compound_domain_question_count != 415`，または主要指標が基準線と小数点以下まで完全一致（success_criteria (6)）．
- **復元手順（partial / no_effect / rejected 共通）**: `config.yaml` の `embedding_instruction` 行と `embedding_view_concat` 行を削除（Iter81 と同じ説明コメントは残す），`cp models/domain_classifier_pre_iter82_noconcat.joblib models/domain_classifier.joblib`（sha256 `21e16ec6...`，`n_features_in_`=1024）で復元，`mise run deploy` を再実行して全 10 ノードで sha256・config・smoke_check を確認する．**さらに `data/MANIFEST.md` を基準線の状態へ巻き戻す**（Iter81 学び 3 の教訓: MANIFEST は実装フェーズで先に書かれるため，rejected で復元した反復では分析フェーズが必ず巻き戻す責任を負う）．

**実験手順**

1. **前提確認**: `wc -l data/dataset.jsonl` = 1915，`wc -l data/classifier_train.jsonl` = 1427，`sha256sum models/domain_classifier.joblib` = `21e16ec6...`，`n_features_in_`=1024．
2. **実装**: 上表 #2・#4〜#8 と単体テスト（連結順序の検証）．`uv run ruff check`（既存 23 件は pre-existing）と `uv run pytest tests/`（pre-existing 9 件 FAIL は B122．**新規失敗 0 件**）．
3. **F1 と G1**（wafl-ctrl5 のみ．`ssh -fNT -L 11499:localhost:11434 wafl-ctrl5`）: P0/P1 のキャッシュは既存のものを再利用してよい（`data/embcache_qwen3-embedding_0.6b.npy` / `..._0.6b__p1.npy`）．
4. **再訓練**（wafl-ctrl5 のみ）: 旧 artifact を `models/domain_classifier_pre_iter82_noconcat.joblib` へ退避してから
   `uv run python -m scripts.train_domain_classifier --train-data data/classifier_train.jsonl --embedding-model qwen3-embedding:0.6b --embedding-instruction 'Given a user question, identify the single academic or professional domain it belongs to' --embedding-view-concat --ollama-host 127.0.0.1 --ollama-port 11499 --output models/domain_classifier.joblib`．**G2-a** を確認．
5. **G2-b・G2'**（wafl-ctrl5 のみ）: 1,915 行を 2 ビューで embed してキャッシュ（`data/embcache_eval_qwen3-embedding_0.6b{,__p1}.npy` 等，**キャッシュ名に必ずビュー識別子を含める**）し，旧／新 artifact の argmax replay から n_d・per-domain 20 指標の BH 補正・着地点予測を算出して**本走前に journal へ記録**する．
6. `config.yaml` に 2 キーを追加．
7. `mise run setup`（**直後に `uv sync --extra research` で research extra を復旧．B118 落とし穴 2**）→ `wc -l data/dataset.jsonl` = 1915 を再確認．
8. `mise run deploy` → **F2** と smoke_check の pass を確認．
9. **先頭 20 問の予備実行 → F3**．不一致なら本走に進まない．
10. **wafl500〜509 で 1,915 問のフルスペック本走を 1 回**（絶対条件 A）．起動直後に `state.json` を `status=waiting_experiment`・`experiment_dir`・`experiment_deadline`（開始時刻 + 150×60 + 600 秒）へ更新．`mise run analyze -- <timestamp>` まで実施（**引数なし実行は `results/iter45_preliminary/` を誤選択する．B118 落とし穴 1**）．
11. 指標を「全 1,915 行」「既存 1,600 行部分集合」「複合 415 行」の 3 通りで算出し，基準線 `results/20260926_221822/` と id ペアリングで McNemar 検定・per-domain 20 指標の BH 補正を行う．**G2' の予測値と本走実測の一致／不一致も必ず記録する．**
12. `data/MANIFEST.md` に新 artifact の sha256・生成コマンド・連結順序・G1/G2/G2'/F の実測値を追記する（`data/`・`models/` は gitignore 対象で MANIFEST が唯一の再現性の担保．B123）．

**期待効果とリスク**

期待効果は「Iter81 が示した +3.655pt の改善を，情報を捨てずに足す設計で，medical の退行なしに取り戻す」ことに尽きる．新規モデルの pull は無く VRAM も増えない（同一モデルを 2 回呼ぶだけ）．リスクは 3 つ．(1) **2048 次元 × 1,427 行の過学習**（Q2 で訓練適合 0.8963・CV 乖離 +1.2pt と定量化済み．崩壊型ではない），(2) **embed 呼び出しが 1 回増えることによる所要時間の増加**（非退行④で監視．probe は LLM を呼ばないので影響は小さい見込みだが要実測），(3) **連結順序の取り違え**（F1・F3・単体テストの 3 重で潰す）．

### Iteration 82 実行済み（実装・ゲート・本走起動）

**実装（計画表の #2・#4〜#8・単体テストをすべて実施．単一レバー厳守，config は 2 キー追加のみ）**

- `expert_backend.py`: 新設 `async def embed_query_views(client, model, text, timeout_s=DEFAULT_TIMEOUT_S, instruction=None, concat_views=False)`．`concat_views=False` は既存 `OllamaClient.embed()` を 1 回呼ぶだけ（無変更の呼び出し経路）．`concat_views=True` は `embed(instruction=None)` → `embed(instruction=instruction)` の順で 2 回呼び，`[*plain, *instructed]` を返す．順序は関数内に固定（呼び出し側から指定不可）．`instruction is None` かつ `concat_views=True` は `ValueError`．`OllamaClient.embed()` 自体は無変更．
- `node.py:202-204`（実行時側）: `ollama_client.embed(...)` → `embed_query_views(ollama_client, config["embedding_model"], query, instruction=config.get("embedding_instruction"), concat_views=config.get("embedding_view_concat", False))`．
- `scripts/train_domain_classifier.py`（訓練側）: `build_training_features()` に `concat_views: bool = False` を追加，L149 相当の embed 呼び出しを `embed_query_views(...)` へ差し替え．`_train_and_save()` に `concat_views` を追加して伝播．CLI に `--embedding-view-concat`（`store_true`）を追加し `main()` から渡す．
- `tools/smoke_check.py:170-172`: 同じ `embed_query_views(...)` へ差し替え．
- `scripts/screen_embedding_models.py`: 候補を `{p0, p1, concat}` へ差し替え．`concat` は再 embed せず `np.hstack([p0_embeddings, p1_embeddings])`（順序 `[p0, p1]` で `embed_query_views()` と同一）で構成．モジュール docstring・CLI description も Iter82 向けに更新．P2/P3 候補は削除（Iter81 で不採用のまま，本反復のスコープ外）．
- `tests/test_expert_backend.py`: `embed_query_views()` の単体テストを 3 件追加（(i) `concat_views=False` は単一 `embed()` 呼び出しに委譲，(ii) `concat_views=True` は呼び出し順序に関わらず戻り値が `[plain, instructed]` の順に連結される，(iii) `instruction=None` かつ `concat_views=True` は `ValueError`）．
- `config.yaml`: `embedding_instruction`（Iter81 選定の P1 文言，変更なし）を復活させ，新キー `embedding_view_concat: true` を追加．コメントを Iter82 向けに更新．
- **検証**: `uv run ruff check .` は pre-existing 23 件のみ（新規 0 件）．`uv run pytest tests/` は 306 passed / 9 failed（pre-existing，B122）+ 新規 3 件を含め **新規失敗 0 件**．

**事前ゲート（すべて wafl-ctrl5，`ssh -fNT -L 11499:localhost:11434 wafl-ctrl5` 経由．すべて合格）**

| ゲート | 実測 | 判定 |
|---|---|---|
| **G1** | `scripts/screen_embedding_models.py` 再実行（`data/classifier_train.jsonl` 1,427 行，5-fold CV，random_state=42）: **p0=0.756123 / p1=0.771523 / concat=0.784844**．計画の事前実測値（0.756132 / 0.771549 / 0.784863）と全て **±1e-5** で一致（±0.001 の許容内） | **PASS**（concat ≥ p1） |
| **F1** | `embed_query_views(..., concat_views=True)` を実クエリ（`"What is the treatment for pneumonia?"`）で実行．len=2048，前半 1024 が prefix なし embed と完全一致，後半 1024 が prefix あり embed と完全一致，前半ノルム 0.99999998，後半ノルム 0.99999999 | **PASS**（4 条件すべて満たす） |
| **G2-a** | 旧 artifact を `models/domain_classifier_pre_iter82_noconcat.joblib` へ退避（sha256 `21e16ec6...`）後，`uv run python -m scripts.train_domain_classifier --train-data data/classifier_train.jsonl --embedding-model qwen3-embedding:0.6b --embedding-instruction '<P1 文言>' --embedding-view-concat --ollama-host 127.0.0.1 --ollama-port 11499 --output models/domain_classifier.joblib` で再訓練．新 sha256 **`1cfcd3d8...`**，`n_features_in_` **2048** | **PASS** |
| **G2-b** | `data/dataset.jsonl` 1,915 行を `data/embcache_eval_qwen3-embedding_0.6b.npy`（p0）・`data/embcache_eval_qwen3-embedding_0.6b__p1.npy`（p1）へ実測 embed（wafl-ctrl5，キャッシュ名にビュー識別子を含む）．旧／新 artifact の argmax replay で discordant **n_d=195**（`1.96/sqrt(195)=0.1404`） | **PASS**（n_d≥30，参考 Iter81 の 276 と近い規模） |
| **G2'-a（忠実度）** | 旧 artifact の replay `selected_domain` と基準線 `results/20260926_221822/results.jsonl` の実測 `selected_domain` の一致率 **1914/1915 = 99.95%** | **PASS**（≥99%，予測を「参考値」に格下げする必要なし） |
| **G2'（本走前予測，新設）** | 新 artifact の replay（`selected_domain`）から `metrics.py` の `compute_precision_recall_per_domain` / `compute_domain_recall_mcnemar_test` / `compute_domain_precision_fisher_test` / `apply_benjamini_hochberg(q=0.05)` を**そのまま呼び出して**算出（自前の統計式は書いていない）．20 指標中 **有意退行 0 件（BH 補正後）**．有意な改善が 2 件: computer_science_recall（0.7273→0.8268，p=4.4e-5）・natural_science_recall（0.5931→0.6494，p=0.003609）．**medical_recall の予測値 0.7842→0.7552（p=0.190430，BH 補正後も有意でない）**．参考: argmax replay ベースの粗い top1 プロキシ（実際の dispatch/aggregation を経ない，単純な「replay の selected_domain が expected_domains に含まれるか」の割合）は 0.793211．**これは本走の `top1_accuracy` そのものではなく，あくまで着地点の目安**（dispatch_top_k=2・dispatch_gap_threshold=0.29 等の実行時挙動を含まないため） | 合格条件なし（記録のみ）．**medical_recall の有意退行は予測されない** |
| **F2** | `mise run setup`（直後に `uv sync --extra research`）→ `mise run deploy` 実行．全 10 ノードで `grep -E '^(embedding_instruction|embedding_view_concat):'` と artifact sha256 `1cfcd3d8...` が一致することを確認．smoke_check（git-status/hashes/probe）は deploy 内で自動 PASS | **PASS** |
| **F3** | `data/dataset.jsonl` 先頭 20 行を一時ファイルとしてコンテナへ投入し `mise run start -- --dataset ... --output f3_20.jsonl` で予備実行．`selected_domain` と新 artifact replay の argmax（wafl-ctrl5 embed 済みキャッシュから算出）が **20/20 一致**．一時ファイルは実行後に削除 | **PASS** |

**本走**

- 起動: `mise run start -- --dataset data/dataset.jsonl --output results.jsonl`（実験ディレクトリ `results/20260927_050049/`，wafl500〜509 で 1,915 問フルスペック 1 回，絶対条件 (A) 遵守）．完了まで実測 176 分（起動 1790452839〜完了ポーリング確認まで）．`mise run analyze -- 20260927_050049` 実施済み（全 10 ノードのログ回収・axis2/3 指標算出）．`uv run python metrics.py --results results/20260927_050049/results.jsonl --json` で主要指標を取得．

**本走の実測結果（機械可読値そのまま．採否判定は次フェーズの担当）**

基準線 `results/20260926_221822/`（Iter79）と id ペアリングし，`metrics.py` の `compute_mcnemar_test` / `compute_precision_recall_per_domain` / `compute_domain_recall_mcnemar_test` / `compute_domain_precision_fisher_test` / `apply_benjamini_hochberg(q=0.05)` をそのまま呼び出して算出（自前の統計式は書いていない）．

| 指標 | 基準線 (Iter79) | Iter82 本走 | 差 |
|---|---|---|---|
| **全 1,915 行 top1_accuracy** | 0.753003 | **0.792167** | **+3.916pt**（McNemar chi2=40.5630, p=1.904e-10，discordant: baseline のみ正解 30 / Iter82 のみ正解 105，計 135） |
| 既存 1,600 行部分集合 top1（先頭 1,600 行＝single 1,500 + compound-001〜100） | 0.751250 | 0.786875 | +3.5625pt（報告のみ） |
| 複合 415 行 top1 | 0.766265 | 0.816867 | +5.060pt |
| single_domain_top1 | 0.749333 | 0.785333 | +3.600pt |
| compound_domain_set_recall | 0.548193 | 0.539759 | -0.843pt（報告のみ．判定に用いない） |
| compound_mean_dispatched_count | 1.880 | 1.610 | -0.270（報告のみ） |
| Cohen's kappa | 0.721502 | 0.761499 | +0.040 |
| fallback_rate | 0.0 | 0.0 | 0（非退行②合格） |
| dispatch_failure_rate | 0.000522 | 0.000522 | 0（非退行②合格，≤0.005） |
| ECE | 0.032745 | 0.025448 | -0.007297（非退行③合格，≤0.08） |
| mean_duration_ms | 2301.4 | 2217.1 | -84.3（非退行④合格，≤2761．embed 呼び出しが 1 回増えたにもかかわらず所要時間は悪化しなかった） |
| answer_quality_accuracy | 0.569333 | 0.572667 | +0.333pt（報告のみ） |
| end_to_end_accuracy | 0.335770 | 0.351958 | +1.619pt（報告のみ） |
| Random baseline | 0.101 | 0.121671 | 報告のみ（success_criteria (3)） |
| Best-single-domain baseline | 0.109 | 0.12689（legal 最大） | 報告のみ |
| Oracle | 1.0 | 1.0 | 報告のみ |

**非退行①（per-domain 20 指標，BH 補正 q=0.05）実測結果**: **有意退行 0 件**．有意な改善 2 件（computer_science_recall 0.7273→0.8268，p=4.4e-5；education_recall 0.3734→0.4120 は p=0.038867 で有意ではない点に注意，BH 補正後は non-sig）．**medical_recall（本反復の主目的）は 0.7842 → 0.7510（-3.32pt，p=0.135593，BH 補正後も有意でない）**．20 指標の全内訳（precision/recall × 10 ドメイン）は上記スクリプトの生出力として journal 執筆時点で確認済み（有意フラグはすべて `computer_science_recall` の 1 件のみ True かつ改善方向）．

**G2' 予測との一致／不一致**: G2'（replay ベースの事前予測）は medical_recall を 0.7842→0.7552（p=0.190430，非有意）と予測しており，本走実測 0.7842→0.7510（p=0.135593，非有意）と**方向・非有意という結論の両方が一致**した（点推定差は 0.0042pt）．全体 top1 についても G2' の argmax replay プロキシ 0.793211 と本走実測 0.792167 の差は 0.001pt で，**G2' は着地点予測として機能した**（ただし実際の dispatch/aggregation を経ないプロキシであり，本走の代替にはならないことも実測で裏付けられた——両者は近いが同一ではない）．

**判定に必要な事前登録条件との対比（判定そのものは次フェーズが行う）**:
- invalid 判定条件（`question_count != 1915`／`compound_domain_question_count != 415`／主要指標が基準線と小数点以下まで完全一致）はいずれも該当しない．`total_questions=1915`，`compound_domain_question_count=415`．
- 主基準（McNemar p<0.05 かつ +1.0pt 以上）: 点推定 +3.916pt，p=1.904e-10．
- 非退行①②③④の実測値は上表のとおり．

**MANIFEST**: `data/MANIFEST.md` に新 artifact `1cfcd3d8...`・生成コマンド・連結順序 `[prefix なし, prefix あり]`・G1/G2-a/G2-b/G2'/F1/F2/F3 の実測値を追記済み（下記参照）．

### Iteration 82 実行済み（分析・考察）

**変更**: 単一レバー `embedding_view_concatenation` = `prefix_and_noprefix_concat`．埋め込みモデル（`qwen3-embedding:0.6b`）・instruction 文言（Iter81 の P1）・分類器のハイパラ・訓練データ・評価集合・ルーティング設定を固定したまま，分類器の入力特徴量を「prefix なし 1024 次元」から「**prefix なし ⊕ prefix あり（P1）の 2048 次元**」へ変えた．連結順序は `expert_backend.embed_query_views()` の内部に固定し，訓練側・実行時側・smoke_check が同一関数を呼ぶ．

**判定: adopted（事前登録の判定規則にそのまま従った）**

事前登録の `adopted` 条件は「主基準（McNemar p<0.05 かつ +1.0pt 以上）と非退行①②③④をすべて満たす」である．実測を規則の文言と 1 対 1 で対応づける．

| 規則の文言 | 実測 | 充足 |
|---|---|---|
| 主基準: 全 1,915 行 top1 が McNemar p<0.05 | chi2=40.5630, **p=1.904e-10**（discordant 135 = 悪化 30 / 改善 105） | ○ |
| 主基準: 点推定 +1.0pt 以上 | 0.753003 → 0.792167，**+3.916pt** | ○ |
| 非退行①: per-domain 20 指標の BH 補正（q=0.05）後の有意退行 0 件 | **0 件**．BH 後に有意なのは computer_science_recall（0.7273→0.8268, p=4.4e-5）の 1 件のみで改善方向．**medical_recall 0.7842→0.7510 は p=0.135593 で非有意** | ○ |
| 非退行②: fallback=0.0 かつ dispatch_failure ≤ 0.005 | 0.0 / 0.000522（いずれも基準線と同値） | ○ |
| 非退行③: ECE ≤ 0.08 | 0.025448（基準線 0.032745 から改善） | ○ |
| 非退行④: mean_duration_ms ≤ 2761 | 2217.1（基準線 2301.4 から -84.3．embed 呼び出しが 1 回増えても悪化しない） | ○ |

`rejected`（点推定低下・非退行①の有意退行 1 件以上・非退行②③④違反）はいずれも成立せず，`partial`（+1.0pt 未満または p≥0.05）・`no_effect`（n_d<30 かつ ±0.5pt 以内．実測 n_d=195）・`invalid`（F/G 不合格・`question_count != 1915`・`compound_domain_question_count != 415`・基準線と完全一致）も成立しない．**したがって adopted 以外の判定は取れない**．F1・F2・F3・G1・G2-a・G2-b・G2'-a は全 PASS で実験は成立している．

**ノイズか信号か**: ルーティング系は分類器 artifact が決まれば決定論的であり（G2'-a: 旧 artifact の replay が基準線実測と 1914/1915 = 99.95% 一致），run-to-run のノイズ床は事実上存在しない．したがって不確実性は 1,915 行という評価集合の標本誤差だけである．discordant 135 行に対する検出限界は `1.96·sqrt(135)/1915 = 1.19pt` で，**実測 +3.916pt はその 3.3 倍**．p=1.904e-10 は Iter81（p=4.207e-7）よりさらに 3 桁小さい．明確な信号である．複合 415 行 +5.060pt・single_domain +3.600pt・kappa +0.040 と符号も一貫しており，部分集合のどれを取っても改善方向である．

**学び 1（本反復の主目的・medical の検証: 仮説は「退行を有意にしない」水準で支持された．ただし「同等」が示されたわけではない）**

Iter81 は medical_recall 0.7842 → 0.7178（p=0.003264，BH 後も有意）で rejected になった．Iter82 の連結は 0.7842 → 0.7510（p=0.135593，BH 後も非有意）に留まり，**事前登録の規則の上では仮説「prefix の利得を取りつつ prefix なしビューの情報を捨てない」は支持された**．ただし点推定は依然 -3.32pt であり，これを「ノイズ」と言い切るのは誤りである．切り分けを数値で行う．

- **母数と検出限界**: medical を含む行は 241 行．Iter82 の medical に関する discordant は 22 行（取りこぼし 15 / 新たに取得 7）で，この母数での検出限界は `1.96·sqrt(22)/241 = 3.8pt`．**-3.32pt はこの検出限界の内側にある**．つまり p=0.1356 は「退行が無い」ことの証明ではなく，**241 行では 3.3pt の変化を区別できない**という測定系の限界の表明である．「非有意＝同等」と読み替えてはならない．
- **Iter81 との比較で見ると改善は実在する**: 取りこぼしは Iter81 の 23 行から **15 行**へ，新規取得は 6 行から **7 行**へ．正味の損失は -17 行から **-8 行**へ半減した．
- **Iter81 学び 2 との接続（rank 分布）**: `probe_candidates` の確信度順位で測ると，medical 241 行に対する **top-2 recall は基準線 0.9046 → Iter82 0.9087（Iter81 は 0.8963）**，**top-3 は 0.9461 → 0.9710（Iter81 は 0.9544）**．すなわち Iter81 が「rank 1 を他ドメインに譲った（top-2 はほぼ不変）」のに対し，**連結は rank≤2 / rank≤3 の水準では基準線を上回っている**．取りこぼした 15 行でも medical は 13 行で rank 2・1 行で rank 3 に残る（rank 1 だが `selected_domain=None` の dispatch 失敗行が 1 行）．吸われた先は education 7・legal 2・business_economics 2・computer_science 2・history_culture 1 で，特定 1 ドメインへの系統的混同ではない．
- **解釈**: 連結は medical の信号を回復させており（top-2/top-3 は基準線超え），残る -3.32pt は **固定 top-1 という `metrics.py` の recall 定義の下で，medical と隣接ドメインの僅差の順位入れ替わりが 8 行分残っていること**に帰着する．「prefix なしビューの情報を捨てない」という機序は支持されたが，**完全な解消ではなく，検出限界以下への縮小である**．

**学び 2（G2' 予測の的中: 何が保証され，何が保証されていないか）**

G2'（新 artifact の 1,915 行 argmax replay に `metrics.py` の検定関数をそのまま当てる手続き）の本走前予測は medical_recall 0.7842→**0.7552**（p=0.190430，非有意），有意退行 0 件，top1 プロキシ 0.793211 だった．本走実測は 0.7510（p=0.135593，非有意），有意退行 0 件，top1 0.792167．**点推定差は medical で 0.42pt，top1 で 0.10pt，結論（非有意・退行 0 件）は完全一致**した．

- **保証されていること**: 分類器 artifact と埋め込みが決まれば `selected_domain` は決定論的に定まるので，G2'-a（旧 artifact の replay が基準線実測と 99.95% 一致）が示すとおり **replay は実行時の rank 1 選択をほぼ完全に再現する**．したがって **rank 1 のみに依存する指標**（top1_accuracy・per-domain recall/precision・kappa）は本走前に高精度で予測できる．検定も `metrics.py` の関数をそのまま流用するため統計手続きの差異も入らない．
- **保証されていないこと**: (i) **rank 2 以降と閾値に依存する指標**（`compound_domain_set_recall`・`compound_mean_dispatched_count`）は replay の argmax だけでは決まらない．(ii) **実行時側の未適用や次元不一致**は replay では検出できない（replay はオフラインの正しい特徴量で計算するため，実機が旧コードのままでも同じ予測が出る．これを潰すのは F2/F3 の役目である）．(iii) **dispatch 失敗・タイムアウト・所要時間・ECE・下流の answer_quality / end_to_end** は replay の対象外（実測でも medical の取りこぼし 15 行のうち 1 行は `selected_domain=None` の dispatch 失敗であり，replay では rank 1 = medical だった）．(iv) 今回の一致は n=1 の事例であり，「常に当たる」ことの保証ではない．
- **運用への含意**: G2' は **per-domain 非退行の事前登録手段として今後も標準手順に含める**（Iter81 は per-domain を本走後にしか見ず 1 本走を失った）．ただし **「予測が当たるから本走を省く」という運用は `.claude/research/config.yml` 冒頭のユーザー絶対条件 (A)（変更を適用したら wafl500〜509 での 1,915 問フルスペック本走を必ず 1 回実施する）および 2026-09-23 恒久運用ルールにより禁止されている**．G2' は本走 1 点を絞り込み，着地点を本走前に数値で言語化するための事前登録手段であって，本走の代替ではない．本反復の実測はむしろ「replay と本走は近いが同一ではない」（top1 で 0.10pt 差，medical で 0.42pt 差，dispatch 失敗行の存在）ことを裏付けた．

**学び 3（複合設問の送出低下は埋め込みの劣化ではなく `dispatch_gap_threshold=0.29` の未較正である．Iter81 学び 1 と同型）**

`compound_domain_set_recall` -0.843pt・`compound_mean_dispatched_count` 1.880 → 1.610 の一方で複合 415 行 top1 は +5.060pt という一見矛盾した結果を，`probe_candidates` の掃引で切り分けた（複合 415 行，`dispatch_policy=adaptive_confidence_gap` の規則を再現）．

- **確信度分布が鋭くなっている（実測）**: rank1 confidence 平均 0.6510 → 0.7084，rank1−rank2 gap 平均 0.4769 → 0.5499（中央値 0.4449 → 0.5787），**gap < 0.29 の行の割合 33.49% → 25.30%**．2 件目以降を送出する条件に当たる行が減ったため `mean_dispatched_count` が下がった．
- **送出予算を揃えると逆転する（Iter81 と同じ手法）**: gap 閾値を掃引して **Iter82 の `mean_dispatched_count` を基準線と同じ 1.880 に揃えると（gt=0.357，mean_k=1.889）set_recall は 0.581928** となり，基準線の gt=0.29（mean_k=1.880，set_recall=0.548193）を **+3.37pt 上回る**．gt=0.40 なら mean_k=2.022・set_recall=0.598795（+5.06pt）．
- **結論**: set_recall の -0.84pt は**埋め込みの劣化ではない**．`dispatch_gap_threshold` は確信度分布のスケールに依存するハイパラであり，特徴量を変えたら再較正が要る，という一般則の再確認である（Iter81 学び 1 と同型で 2 回目．**特徴量を変えるレバーの後には必ず gap 閾値の未較正が残る**と型化してよい）．これは事前登録どおり判定には用いていないが，次レバーの根拠になる．

**学び 4（B128 の未決事項＝非退行条件①の設計の是非は，実質的に消滅した）**

B128 要レビューは「全体 +3.655pt を medical 1 件の退行で棄却してよいか」を人間判断に委ね，選択肢 A1（維持）/A2（緩和）/A3（下流指標で置換）を挙げていた．**Iter82 が規則を一切変えずに adopted になったことで，この論点は実質的に消滅した**と判断する．根拠は 2 つ．(1) 連結構成は Iter81 の prefix 構成を top1 で上回る（+3.916pt vs +3.655pt）ため，規則を緩めて Iter81 を救う実益がそもそも無い（緩めても得られたのは劣る構成である）．(2) 「規則を緩める」ではなく「規則を満たす設計を見つける」で解決できることを実証した．したがって **A1（現行どおり維持）を既定として継続し，人間へのエスカレーションは行わない**．

ただし完全な無風ではなく，**別の形で 1 点だけ残る**ので記録しておく: 非退行①は 20 指標の AND 条件だが，学び 1 で示したとおり **1 ドメインあたりの母数（medical で 241 行）では約 3.8pt 未満の退行を検出できない**．つまりこの規則は「小さな退行を見逃す」方向にも非対称である．現時点では実害が無く，評価集合の母数を増やさない限り解消しないため，**規則の改定ではなく測定系の課題として backlog に残す**（対処するなら `compound_eval_set_expansion` と同型の per-domain 行数の拡充が要る）．次に非退行①で rejected が出た時点で，改めて A1/A2/A3 の人間判断を求めることとする．

**本番反映の検証（adopted のため復元は行わない）**

- 全 10 ノード（wafl500〜509）で `config.yaml` の `embedding_view_concat: true` と `embedding_instruction`（P1 文言）が存在し，`models/domain_classifier.joblib` の sha256 が **`1cfcd3d8...`** で一致することを本フェーズで再確認した．
- 退避 artifact `models/domain_classifier_pre_iter82_noconcat.joblib`（sha256 `21e16ec6...`，`n_features_in_`=1024）はローカルに現存する．
- `data/MANIFEST.md` は現行 artifact 行を `1cfcd3d8...`（2048 次元，連結）として正しく指しており，Iter81 で起きた「MANIFEST だけが実態と乖離する」不具合（学び 3 / B128 (c)）は再発していない．

**次の一手**: 新レバー **`dispatch_gap_threshold_recalibration` = `matched_budget_sweep_for_concat_distribution`**（`.claude/research/config.yml` の `levers` 末尾へ追記済み）．学び 3 の掃引実測（gt=0.357 で送出予算を揃えたまま set_recall +3.37pt，gt=0.40 で +5.06pt）を根拠に，連結埋め込みの確信度分布へ `dispatch_gap_threshold` を再較正する．選定理由と対案は backlog B130 を参照．

---

## Iteration 81: 埋め込み入力への instruction prefix 付与（qwen3-embedding:0.6b）

### 調査 (Iter81)

本反復のレバーは B127（Iter80 分析フェーズ）で `embedding_input_instruction_prefix` = `qwen3_instruct_classification_prefix` に確定済みで，レバー選定の裁量は無い．調査の問いは 3 つ．**(Q1) Qwen3-Embedding の instruction 形式は具体的にどう書くのが正しく，どの程度の効果が報告されているか．(Q2) prefix は「クエリ側だけ」か「訓練側・推論側の両方」か（symmetric / asymmetric タスクでの扱いの違い）．(Q3) Ollama の `/api/embeddings` は prefix を自動付与するのか．**

**Q1: 形式は `Instruct: {task_description}\nQuery: {text}`．効果は「多くの下流タスクで 1〜5%」**

- Qwen 公式（<https://github.com/QwenLM/Qwen3-Embedding>，HF Model Card <https://huggingface.co/Qwen/Qwen3-Embedding-0.6B> / <https://huggingface.co/Qwen/Qwen3-Embedding-8B>，いずれも 2026-09-27 確認）の `get_detailed_instruct()` は `f'Instruct: {task_description}\nQuery:{query}'` を返す．**モデル同梱の `config_sentence_transformers.json` も `query` prompt = `"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "`，`document` prompt = `""`（空）**である（ollama/ollama issue #16076 <https://github.com/ollama/ollama/issues/16076> に原文が引用されている）．
- 効果の公称値: 「**評価の結果，多くの下流タスクで instruct を使うと使わない場合に比べ通常 1〜5% の改善が得られる**．したがってタスク・シナリオに合わせた instruction を作ることを推奨する．**多言語の文脈では，訓練時の instruction の大半が英語で書かれていたため，instruction は英語で書くことを推奨する**」（Qwen3-Embedding GitHub README，同文が HF Model Card・Docker Hub の `ai/qwen3-embedding` にも掲載）．**本研究の質問文は日本語だが，instruction は英語で書くのが公式推奨である**点は設計に直接効く（事実）．
- ただし「1〜5%」の但し書きは「**retrieval シナリオで query 側に instruct を付けない場合に約 1〜5% 低下する**」という retrieval 中心の文言でもある（同 README の Tip）．**分類タスクでの効果量を保証する一次情報は見つからなかった**（推測と事実の区別: 分類での効果は未確認）．
- 日本語での傍証: hotchpotch 氏の JMTEB 計測（<https://secon.dev/entry/2025/06/11/100000-qwen3-embedding-jmteb>，<https://hotchpotch.dev/articles/qwen3-embedding-jmteb>）は Qwen3-Embedding-0.6B の Classification 66.09 を報告しているが，本文に「**Retrieval, Reranking タスクでは Query の prefix に `Instruct: ...\nQuery:` を追加している**」と明記されており，**Classification は prefix なしで測られている**．つまり「日本語分類で prefix を付けた値」は公開情報として存在しない（事実）．Iter79/80 の学び（公称ベンチは採否の根拠にならない）どおり，G1 CV で自前に数値化するほかない．

**Q2: 分類のような single-side（symmetric）タスクでは，全テキストに同じ instruction を付ける．訓練側と推論側で一致していなければならない**

- 公式コード例は **retrieval 前提の asymmetric な使い方**であり，`# No need to add instruction for retrieval documents` と明記して documents 側には何も付けない（上記 GitHub / Model Card）．
- 一方，**分類・クラスタリング・STS のように「クエリ／文書」の区別が無いタスクでは，全入力テキストに instruction を付けるのが標準的な運用**である．arXiv:2606.01074（"When Is 0.1% Enough?"，<https://arxiv.org/html/2606.01074v1> Appendix A，Qwen3-Embedding-8B を含む instruction 系 4 モデルで MTEB 4 タスク族を評価）は「instruction 系モデルでは全タスクで task-specific instruction を付けて符号化する．**classification / clustering / STS では `Instruct: {instruction}\nInput: {text}` を使い，retrieval ではクエリのみ `Instruct: {instruction}\nQuery: {text}` とし，corpus 文書は instruction なしで符号化する（標準的な asymmetric retrieval 設定に従う）**」と手続きを明記している．**`Query:` ではなく `Input:` を使う流儀がある**点は本反復の文言候補に反映する．
- mteb ライブラリでも各タスクの `prompt`（未指定時は抽象クラスの既定文，例: Clustering は `"Identify categories in user passages."`）が入力テキストへ前置される（<https://github.com/embeddings-benchmark/mteb/discussions/3239>，メンテナ回答）．
- **本研究への帰結**: 分類器の訓練特徴量（`data/classifier_train.jsonl` の質問文）と実行時のクエリ文は**同じ種類のテキスト（1 本の日本語質問）**であり，片側だけに prefix を付けると**訓練時と推論時で入力分布がずれる**．Iter36 の失敗（train/eval のタスク不一致で education_recall 0.4588 → 0.0529）と同型の事故になるため，**`scripts/train_domain_classifier.py`（訓練）と `node.py`（実行時クエリ）の双方に同一 prefix を適用することが必須条件**である．一方，`http_server.py:402-405` が計算する `domain_embedding`（ドメイン名そのものの埋め込み）は「文書側」に相当し，かつ `routing_method=supervised_classifier` の現構成では `estimate_embedding_confidence` 経路に到達しない．**ここには prefix を付けない**（asymmetric 慣行に従い，かつ無変更部分を増やさない）．
- **リスク（推測）**: 全入力に同一の prefix を付ける運用は「全ベクトルに共通の変形」を加えるため，線形分類器の性能に与える影響が小さい（discordant が伸びない）可能性がある．一方で Qwen3 は last-token pooling かつ attention 経由なので単なる平行移動ではなく，効果が出るとすれば表現の再配置による．**効果ゼロの可能性を織り込み，G2 の n_d が小さい場合の解釈規則を事前登録する**（後述）．

**Q3: Ollama は prefix を自動付与しない．クライアント側で付ける必要がある**

- ollama/ollama issue #16076（2025〜2026）は「`/api/embed` が `task: "query" | "document"` を受け取り，対応する prefix を Ollama 側で自動前置すべきだ．**今日はこれをクライアント側でやるしかなく壊れやすい**」と現状を述べている（＝現行の Ollama は自動付与しない）．
- 第三者ベンチ（<https://localaimaster.com/blog/best-ollama-embedding-models>）も「**Ollama の `/api/embed` は prefix を一切付けないので自分のコードで前置する必要がある**」と明記．同記事は embeddinggemma では prefix の有無が hit@1 で 33.3% → 87.5% と決定的だった一方，**`qwen3-embedding` では prefix の有無で結果がほぼ同一（1 クエリ差以内）だった**とも報告している（小規模な独自 retrieval 評価，n が小さく一次情報としては弱い）．**本反復の期待値を過大に見積もらない根拠として扱う**．
- 本リポジトリの実装は `expert_backend.py:140-165` の `OllamaClient.embed()` が **旧 `/api/embeddings`（`prompt` フィールド）**を叩いており，`node.py:202` / `scripts/train_domain_classifier.py:140` / `http_server.py:403-405` のいずれも**生の質問文をそのまま渡している**（prefix なし）．学習時分布とのずれは実在する（コードで確認済み）．

**変更対象コードの特定（Read で確認済み．行番号付き）**

| # | ファイル:行 | 現状 | 本反復での扱い |
|---|---|---|---|
| 1 | `config.yaml:4` | `embedding_model: qwen3-embedding:0.6b`（次行から `confidence_threshold`） | 直後に新キー `embedding_instruction:`（英語 1 文．未設定／null で現行動作）を追加 |
| 2 | `expert_backend.py:140-165` | `async def embed(self, model, text, timeout_s)` が `POST /api/embeddings {"model":..,"prompt": text}`（L150-153） | 省略可能引数 `instruction: str | None = None` を追加し，非 None のとき `prompt` を `f"Instruct: {instruction}\nQuery: {text}"` に置換．**既定 None で全既存呼び出しの動作は不変** |
| 3 | `node.py:202` | `await ollama_client.embed(config["embedding_model"], query)` | `instruction=config.get("embedding_instruction")` を渡す（**実行時クエリ側．本レバーの本体**） |
| 4 | `scripts/train_domain_classifier.py:99-142`（実 embed は L140） | `build_training_features()` が `row["query"]` を素で embed | 引数 `instruction: str | None` を追加し L140 へ渡す．CLI に `--embedding-instruction`（`argparse` 定義は L229 付近，`main()` の受け渡しは L197-206・L244-245）を追加（**訓練側．Q2 より必須**） |
| 5 | `scripts/screen_embedding_models.py:64-66, 73-82, 86-104` | `_CANDIDATES`／`_cache_path()`／`_embed_candidate()` がモデル名のみでキャッシュ名を決める | G1 用に「モデル固定・prefix 文言を変える」比較へ書き換える．**`_cache_path()` に prefix 識別子を必ず含める**（`embcache_qwen3-embedding_0.6b.npy` は prefix なしの Iter79 キャッシュであり，識別子を足さないと prefix 版が無言で旧キャッシュを読み，本リポジトリで 6 回起きた「レバー未到達」事故を再現する） |
| 6 | `tools/smoke_check.py:165-170` | `config["embedding_model"]` で `SMOKE_QUERY` を素で embed し分類器へ通す | 同じ `embedding_instruction` を使うよう修正（しないと deploy 後の smoke_check が訓練時と違う分布で確率を見ることになる） |
| 7 | `http_server.py:402-405` | `state.domain_embedding = await ollama_client.embed(state.embedding_model, state.domain)` | **変更しない**（文書側相当・現構成では未到達．Q2 の結論） |
| 8 | `scripts/evaluate_classifier_calibration.py:395,471,612` / `scripts/evaluate_dispatch_candidate_ranking.py:169` / `scripts/fit_embedding_whitening.py:62` / `scripts/run_central_experiment.py:238` | 素の質問文を embed | **本反復では変更しない**（現行の実験経路に含まれない）．ただし今後これらを prefix 前提の artifact に対して使うと無言で分布がずれるため，`data/MANIFEST.md` に注意を明記する |
| 9 | `models/domain_classifier.joblib` | Iter79 版（sha256 `21e16ec6...`，`n_features_in_`=1024） | prefix 付き埋め込みで再訓練して差し替え．旧版を `models/domain_classifier_pre_iter81_noprefix.joblib` へ `cp` 退避 |

**次元は 1024 のまま変わらない点に注意**．Iter80 は次元不一致で 500 エラーが出るため「prefix/モデルの取り違え」が必ず表面化したが，**本反復は訓練側と推論側で prefix が食い違っても例外は起きず，静かに精度だけ落ちる**．この非対称性が本反復最大の実験運用上のリスクであり，後述の F1〜F3 で明示的に潰す．

### 計画 (Iter81)

**単一レバー**

`embedding_input_instruction_prefix` = **`qwen3_instruct_classification_prefix`**．`/api/embed`（実体は `/api/embeddings`）へ渡す文字列に instruction prefix を付けること**だけ**を変える．埋め込みモデルは `qwen3-embedding:0.6b` のまま，分類器のハイパラ・訓練データ・評価集合・ルーティング設定は一切変えない（prefix 変更に構造的に付随する分類器の再訓練は，Iter79/80 で確立したとおり別レバーとは数えない）．

**固定する構成（直近の最良構成 = Iter79 基準線）**

`config.yaml` の `embedding_model=qwen3-embedding:0.6b`・`routing_method=supervised_classifier`・`confidence_threshold=0.0`・`dispatch_candidate_threshold=0.0`・`dispatch_top_k=2`・`dispatch_gap_threshold=0.29`・`dispatch_gap_max_k=4`・`aggregation_method`・`judge_model`・`classifier_model_path`・各ノードの `light_model=qwen3.5:4b-q4_K_M`／`expert_model=expert-mesh-*-lora`・`probe_timeout_s`／`dispatch_timeout_s`，`data/dataset.jsonl`（1,915 行，ビット単位で不変），`data/classifier_train.jsonl`（1,427 行，不変．train/eval 重複 72 行は B125(c) のとおり本反復でも触らない），`scripts/train_domain_classifier.py` のモデル定義部（`LogisticRegression(max_iter=1000, class_weight=None)` + `CalibratedClassifierCV(method='temperature')` + `_extract_sample_weights()`），`classifier.py`・`aggregator.py`・`metrics.py`・`build_dataset.py`，`docker-compose.yml`．**ドメイン固有の文言を含む prefix は 2026-09-23 恒久運用ルールに抵触するため不可．task_description は 10 ドメイン共通の英語 1 文に限る**（B127 要レビュー (1)）．

**事前ゲート G1（文言の確定．評価集合を一切見ない）**

`data/classifier_train.jsonl`（1,427 行）**のみ**で 5-fold StratifiedKFold（`random_state=42`，`LogisticRegression(max_iter=1000, class_weight=None)` + `sample_weight`）の accuracy / macro-F1 を測る．`data/dataset.jsonl` は参照しない．全て wafl-ctrl5（`ssh -fNT -L 11499:localhost:11434 wafl-ctrl5`）で行い，wafl500〜509 は使わない（絶対条件 B）．**比較する文言は結果を見る前に次の 4 条件へ固定する**（B127 要レビュー (1) の「2〜3 文言を G1 でオフライン比較し 1 つへ確定，本走は 1 回」に従う）．

| id | `/api/embeddings` へ渡す文字列 | 由来 |
|---|---|---|
| **P0** | `{text}`（prefix なし＝現行基準線） | Iter79 構成．`data/embcache_qwen3-embedding_0.6b.npy` を再利用（再計算不要） |
| **P1** | `Instruct: Given a user question, identify the single academic or professional domain it belongs to\nQuery: {text}` | Qwen 公式 `get_detailed_instruct()` の形式 ＋ 本タスク向けの英語 1 文（公式は「タスクに合わせて英語で書く」ことを推奨） |
| **P2** | `Instruct: Given a user question, identify the single academic or professional domain it belongs to\nInput: {text}` | arXiv:2606.01074 Appendix A が classification/clustering/STS に用いる `Input:` 形式．P1 との差は末尾ラベルのみ |
| **P3** | `Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: {text}` | モデル同梱 `config_sentence_transformers.json` の既定 query prompt（文言を自作しない対照） |

- **選定規則（事前登録）**: P1/P2/P3 のうち **CV accuracy が最大のもの**を採用する．同点なら macro-F1，なお同点なら P1 > P2 > P3 の順．**P0（Iter79 実測 cv_accuracy 0.7561 / macro-F1 0.7562．同一手順で再計算して一致を確認する）を 3 つとも下回る場合でも，最良の非 P0 候補で本走は実施する**（config.yml 絶対条件 A．Iter79/80 と同じ規則）．その場合は「改善しない見込み」という着地点予測を本走前に journal へ記録する．
- **CV の位置づけ**: G1 CV は**値の選定にのみ用い，本走 top1 の予測値としては扱わない**（Iter79/80 で順位の予測には 2 回連続で成功しているが，方針は据え置く）．
- **キャッシュ命名**: `data/embcache_qwen3-embedding_0.6b__{p1|p2|p3}.npy` のように prefix 識別子を必ず含める（上表 #5）．

**事前ゲート F（レバー発火の直接証拠．本反復では G2 の n_d と分離する）**

本反復は次元が変わらないため，prefix の付け忘れ・食い違いが例外にならない．そこで**発火の証拠を n_d とは独立に 3 つ取る**．

- **F1（埋め込みレベル）**: wafl-ctrl5 で同一の 1 文を prefix 有り／無しで embed し，**コサイン類似度 < 0.999** を確認する（Ollama が prefix を無視していないことの証明．Q3 の裏取り）．満たさなければ以降に進まない．
- **F2（配布レベル）**: deploy 後，全 10 ノードで `grep '^embedding_instruction:' $REMOTE_DIR/config.yaml` が選定文言と一致し，artifact sha256 が新版と一致すること．
- **F3（実行時経路レベル）**: 先頭 20 問の予備実行の `selected_domain` を，**同じ 20 問をオフライン replay（新 artifact × prefix 付き埋め込み）した argmax と 20/20 一致**することを確認する（埋め込みは決定論的なので完全一致を要求する．dispatch 失敗行は除外して分母を明示）．**不一致があれば実行時側の prefix 未適用を疑い，本走に進まない．**

**事前ゲート G2（検出力）**

旧 artifact（prefix なし）と新 artifact（選定 prefix）の `predict_proba` argmax を `data/dataset.jsonl` 1,915 行で replay し，discordant 行数 n_d と必要偏り率 `1.96/sqrt(n_d)` を算出する．**n_d ≥ 30 を合格条件**とする．**解釈規則（事前登録）**: Iter79/80 では n_d 一桁を「config 未到達」の既定解釈としたが，**本反復に限り F1〜F2 が合格しているなら「未到達」ではなく「prefix の効果が検出限界未満」と解釈する**（同一 prefix を全入力へ一律付与する運用では変化が小さくなりうる，という Q2 のリスクに対応）．その場合も絶対条件 A に従い本走は実施し，判定は後述の「no_effect」に落とす．

**変更するファイルと箇所**: 上表 #1〜#6・#9（#7 は変更しない，#8 は MANIFEST への注記のみ）．加えて `data/MANIFEST.md` に新 artifact の sha256・生成コマンド・選定 prefix 文言・G1/G2/F の実測値を追記する（`data/`・`models/` は gitignore 対象で MANIFEST が唯一の再現性の担保．B123）．`.claude/research/config.yml` の `levers` は既に本レバーを含むため追記不要．

**レバーを読むコード行と到達条件**

- 設定 → 各ノードの config: `mise.toml` L67 の `rsync config.yaml`．到達確認は **F2**．
- 設定 → 実行時のクエリ埋め込み: `node.py:202` → `expert_backend.py:140-153`．到達確認は **F3**（次元不一致では落ちないため，予備 20 問の一致で代替する）．
- 設定 → 訓練特徴量: `scripts/train_domain_classifier.py:140`．到達確認は artifact の sha256 が旧版と異なり，かつ G2 の n_d ≥ 30．
- 埋め込みモデルの pull・`models/` rsync は Iter79/80 と同一（`mise.toml` L70-74・L96-104）．**新規に pull するモデルは無い．**
- 実験 → 指標: `metrics.py` 無変更．到達確認は `question_count == 1915` かつ `compound_domain_question_count == 415`．

**実験手順**

1. **前提確認**: `wc -l data/dataset.jsonl` = 1915，`wc -l data/classifier_train.jsonl` = 1427，`models/domain_classifier.joblib` の sha256 が `21e16ec6...`・`n_features_in_`=1024．
2. **F1**（wafl-ctrl5 のみ）．
3. **G1**（wafl-ctrl5 のみ）: P0〜P3 の CV を 1 回の実行で全て記録し，選定規則で 1 文言へ確定する．
4. **再訓練**（wafl-ctrl5 のみ）: 旧 artifact を `models/domain_classifier_pre_iter81_noprefix.joblib` へ退避したうえで `scripts.train_domain_classifier --train-data data/classifier_train.jsonl --embedding-model qwen3-embedding:0.6b --embedding-instruction '<選定文言>' --ollama-host 127.0.0.1 --ollama-port 11499 --output models/domain_classifier.joblib`．`n_features_in_`=1024 と sha256 を記録．
5. **G2**（wafl-ctrl5 のみ）: 1,915 行の argmax replay で n_d と `1.96/sqrt(n_d)`，新 artifact 単体のオフライン accuracy（着地点予測．判定には使わない）を記録．
6. `config.yaml` に `embedding_instruction` を追記し，`uv run ruff check`（既存 23 件は pre-existing）と `uv run pytest tests/`（pre-existing 9 件 FAIL は B122．**新規失敗 0 件**）を確認．
7. `mise run setup`（**直後に `uv sync --extra research` で research extra を復旧．B118 の落とし穴 2**）→ `wc -l data/dataset.jsonl` = 1915 を再確認．
8. `mise run deploy` → **F2** と smoke_check の pass を確認．
9. **先頭 20 問の予備実行 → F3**．不一致なら本走に進まず原因を潰す．
10. **wafl500〜509 で 1,915 問のフルスペック本走を 1 回**（絶対条件 A）．起動直後に `state.json` を `status=waiting_experiment`・`experiment_dir`・`experiment_deadline`（開始時刻 + 150×60 + 600 秒）へ更新．`mise run analyze -- <timestamp>` まで実施（**引数なし実行は `results/iter45_preliminary/` を誤選択する．B118 の落とし穴 1**）．
11. 指標を「全 1,915 行」「既存 1,600 行部分集合」「複合 415 行」の 3 通りで算出し，Iter79 基準線 `results/20260926_221822/` と id ペアリングで McNemar 検定・per-domain 20 指標の BH 補正を行う．

**仮説（事前登録）**

「Qwen3-Embedding は instruction-aware に訓練されており，現行実装は prefix を一切付けずに学習時分布とずれた入力を与えている．訓練側と推論側の双方へ同一の英語 1 文 instruction を付けると，10 ドメインの線形分離度が上がり，1,915 行本走の `top1_accuracy` が Iter79 基準線 0.753003 から **+1.0pt 以上**改善する（McNemar p<0.05）．ただし公称の 1〜5% は retrieval 中心の値であり，分類での効果量は一次情報が無い．さらに**全入力へ一律に同じ prefix を付ける運用は変化が小さくなりうる**ため，**着地点は 0〜+2pt の範囲，discordant n_d は 50〜300 行**と予測する．」

**成功条件（事前登録．結果を見る前に固定する）**

基準線は Iter79 本走 `results/20260926_221822/`（全 1,915 行: top1=0.753003，single_domain_top1=0.749333，compound_domain_top1=0.766265，compound_domain_set_recall=0.548193，kappa=0.721502，misrouting=0.246997，ECE=0.032745，Brier=0.152849，AUROC=0.777614，fallback=0.0，dispatch_failure=0.000522，mean_duration_ms=2301.4，compound_mean_dispatched_count=1.880，answer_quality=0.569333，end_to_end=0.335770／既存 1,600 行部分集合: top1=0.751250）．

| 区分 | 指標 | 現状 | 合格条件 |
|---|---|---|---|
| **F1** | prefix 有／無の埋め込みのコサイン類似度 | — | **< 0.999**（Ollama が prefix を反映していること） |
| **F2** | 全 10 ノードの `embedding_instruction` と artifact sha256 | — | 全ノードで選定文言・新 sha256 に一致 |
| **F3** | 予備 20 問の `selected_domain` とオフライン replay の一致 | — | **20/20 一致**（dispatch 失敗行は除外し分母を明記） |
| **G1** | 5-fold CV accuracy（`classifier_train.jsonl` のみ） | P0 = 0.7561 | P0〜P3 の 4 値を全て記録し，選定規則どおり 1 文言へ確定できること |
| **G2** | 旧／新 replay の discordant n_d | 参考: Iter80 は 469 | **n_d ≥ 30**．未満でも F1・F2 合格なら「検出限界未満」と解釈し本走は実施 |
| **主基準（効果）** | 全 1,915 行の `top1_accuracy` | 0.753003 | **McNemar p < 0.05 かつ 点推定 +1.0pt 以上**（n_d=100 想定の MDE `1.96·sqrt(n_d)/1915` ≈ 1.02pt を上回る水準） |
| **非退行①** | per-domain recall/precision 計 20 指標 | Iter79 実測 | **BH 補正（q=0.05）後の有意退行 0 件** |
| **非退行②** | `fallback_rate` / `dispatch_failure_rate` | 0.0 / 0.000522 | fallback = 0.0，dispatch_failure ≤ 0.005 |
| **非退行③** | ECE | 0.032745 | **≤ 0.08** |
| **非退行④** | `mean_duration_ms` | 2301.4 | **≤ 2761（+20% 以内）**．prefix は入力トークンを 15〜20 トークン増やすだけなので，Iter80 の「2 倍」枠は不要 |
| 報告のみ | 既存 1,600 行部分集合の top1 | 0.751250 | 判定には用いないが毎回併記 |
| 報告のみ | `compound_domain_top1` / `single_domain_top1` / `compound_domain_set_recall` / `compound_mean_dispatched_count` | 0.766265 / 0.749333 / 0.548193 / 1.880 | 複合 415 行は埋め込み品質の変化に対する感度が単一行より高い（Iter80 の学び 4） |
| 報告のみ | `answer_quality_accuracy` / `end_to_end_accuracy` | 0.569333 / 0.335770 | 生成器は不変のため横ばいを期待（3SD=2.6pt のノイズ床を適用） |

**判定規則（事前登録）**

- **adopted**: 主基準（McNemar p<0.05 かつ +1.0pt 以上）と非退行①②③④をすべて満たす．
- **partial**: 点推定は改善だが p ≥ 0.05 または +1.0pt 未満で，非退行に違反なし．この場合 prefix は**採用せず基準線へ復元**し，レバーは「収束」扱いとする（文言をさらに探すチューニングは B127 要レビュー (1) により行わない）．
- **no_effect**: G2 の n_d < 30（ただし F1・F2 合格）かつ本走の top1 差が ±0.5pt 以内．**「実験不成立」ではなく「prefix は本構成で効かない」という結論**として記録し，基準線へ復元する．
- **rejected**: 点推定が低下，または非退行①で有意退行 1 件以上，または非退行②③④のいずれかに違反．
- **invalid（実験不成立）**: F1・F2・F3 のいずれか不合格，`question_count != 1915`，`compound_domain_question_count != 415`．setup/deploy・実装漏れと解釈する（d0004 §4）．
- **復元手順（partial / no_effect / rejected 共通）**: `config.yaml` の `embedding_instruction` 行を削除し，`cp models/domain_classifier_pre_iter81_noprefix.joblib models/domain_classifier.joblib`（sha256 `21e16ec6...`，`n_features_in_`=1024）で復元，`mise run deploy` を再実行して全 10 ノードで sha256 と config を確認する．

**期待効果とリスク**

期待効果は「モデルを替えずに，学習時分布へ入力形式を合わせるだけで分離度を取り戻す」ことに尽きる．VRAM を増やさないため Iter80 の G0 制約に抵触しない．リスクは 2 つで，(1) **効果がゼロに近い可能性**（公称 1〜5% は retrieval 側の値であり，第三者ベンチでも qwen3-embedding は prefix 有無でほぼ同一だったと報告されている），(2) **訓練側と推論側で prefix が食い違っても例外が出ず静かに劣化する**こと．(1) は G1・G2 で本走前に数値化し，(2) は F1〜F3 で潰す．

### 実験 (Iter81)

事前登録した計画どおりに実装・実験を実施した．判定は行っていない（分析・考察フェーズの担当）．

**変更したファイル**: `expert_backend.py`（`OllamaClient.embed()` に `instruction: str | None = None` を追加し，非 None のとき `prompt = f"Instruct: {instruction}\nQuery: {text}"`．既定 None で既存呼び出しは不変），`node.py:202`（`instruction=config.get("embedding_instruction")` を渡す），`scripts/train_domain_classifier.py`（`build_training_features()` / `_train_and_save()` に `instruction` 引数，CLI `--embedding-instruction`），`scripts/screen_embedding_models.py`（prefix 文言比較へ全面書き換え．`_cache_path()` に `__p1`/`__p2`/`__p3` の識別子を含める），`tools/smoke_check.py:169-171`，`config.yaml`（新キー `embedding_instruction`），`data/MANIFEST.md`（Iter81 節）．`http_server.py:402-405` は計画どおり変更していない．`models/domain_classifier.joblib` は prefix 付きで再訓練（sha256 `56d5a882ec8e...`），旧版を `models/domain_classifier_pre_iter81_noprefix.joblib`（sha256 `21e16ec6...`）へ退避した．

**lint / テスト**: `uv run ruff check` 23 件（実装前と同数．pre-existing），`uv run pytest tests/` 9 failed / 306 passed（`test_build_dataset.py` の 9 件は B122 の既知失敗で実装前後同一）．**新規失敗 0 件**．

**ゲート実測（すべて PASS）**

| ゲート | 実測 | 判定 |
|---|---|---|
| F1（prefix 有無のコサイン） | 0.7674 < 0.999 | PASS |
| G1（`classifier_train.jsonl` 1,427 行のみ 5-fold CV） | P0=0.756123 / **P1=0.771523（最良，採用）** / P2=0.766610 / P3=0.754720 | PASS |
| G2（1,915 行 argmax replay の discordant） | n_d=276 ≥ 30 | PASS |
| F2（deploy 後 全 10 ノード） | `embedding_instruction` 文言・artifact sha256 `56d5a882ec8e...` 全ノード一致 | PASS |
| F3（先頭 20 問 予備実行 vs replay） | `selected_domain` 20/20 一致，dispatch 失敗 0 件 | PASS |

採用した instruction 文言（P1）は `Given a user question, identify the single academic or professional domain it belongs to`．

**本走**: `results/20260927_024644/`．wafl500〜509 で 1,915 問フルスペックを 1 回．`mise run analyze -- 20260927_024644` 実施済み．基準線は `results/20260926_221822/`（Iter79）．

| 指標 | 基準線 | Iter81 | 差 |
|---|---|---|---|
| **全 1,915 行 top1_accuracy** | 0.753003 | **0.789556** | **+3.655pt**（McNemar chi2=25.5968, p=4.207e-7, discordant 58/128） |
| 既存 1,600 行部分集合 top1 | 0.751250 | 0.781250 | +3.000pt |
| 複合 415 行 top1 | 0.766265 | 0.843373 | +7.711pt |
| single_domain_top1 | 0.749333 | 0.774667 | +2.533pt |
| compound_domain_set_recall | 0.548193 | 0.526506 | -2.169pt |
| compound_mean_dispatched_count | 1.880 | 1.494 | -0.386 |
| fallback_rate | 0.0 | 0.0 | 0 |
| dispatch_failure_rate | 0.000522 | 0.000522 | 0 |
| ECE | 0.032745 | 0.030951 | -0.001794 |
| mean_duration_ms | 2301.4 | 2200.7 | -100.7 |
| answer_quality_accuracy | 0.569333 | 0.580000 | +1.067pt |
| end_to_end_accuracy | 0.335770 | 0.350392 | +1.462pt |

**per-domain recall/precision 計 20 指標（BH 補正 q=0.05）**: 有意差 3 件．computer_science_recall 0.7273→0.8528（p≈4.93e-7，改善），natural_science_recall 0.5931→0.6623（p=0.004586，改善），**medical_recall 0.7842→0.7178（p=0.003264，悪化）**．他 17 指標は有意差なし．

**申し送り 1（要検証）**: 本反復開始時の実機構成は `embedding_model=qwen3-embedding:0.6b`・artifact sha256 `21e16ec6...` であり，rc-executor は「これが MANIFEST 上 Iter80 で adopted と記録されていた bge-m3 版（sha256 `37d71b63...`）と一致しない」と報告した．**ただし backlog B127 では Iter80 の判定は `rejected` で，基準線（`21e16ec6...`）への復元を実施したと記録されている**ため，MANIFEST 側の記述が judgment と食い違っている可能性が高い．どちらが誤りかは本反復のスコープ外として MANIFEST に注記のみ残してある．分析フェーズで確認すること．

**申し送り 2（分析フェーズで確認済み）**: wafl500 上に F3 検証用の空ディレクトリ `~/workspace/ktakahashi/expert-mesh/results/iter81_preflight`（root 所有）が残存している．空で無害だが削除に sudo が必要なため放置した．

### Iteration 81 実行済み

**変更**: 単一レバー `embedding_input_instruction_prefix` = `qwen3_instruct_classification_prefix`．埋め込みモデル（`qwen3-embedding:0.6b`）・分類器のハイパラ・訓練データ・評価集合・ルーティング設定を固定したまま，`/api/embeddings` へ渡す文字列を `Instruct: Given a user question, identify the single academic or professional domain it belongs to\nQuery: {text}`（G1 で 4 候補から選定した P1）へ変え，訓練側（`scripts/train_domain_classifier.py`）と実行時クエリ側（`node.py:202`）の双方に同一 prefix を適用した．

**結果（ノイズか信号か）**: 全 1,915 行 top1 **0.753003 → 0.789556（+3.655pt）**．ルーティング系は決定論的でノイズ床を適用しない系だが（config.yml success_criteria (5)），それでも McNemar chi2=25.5968・**p=4.207e-7**（discordant 186 行の内訳 58 悪化 / 128 改善）であり，事前登録の MDE（n_d=100 想定で約 1.02pt）を 3.6 倍上回る．**ノイズではなく明確な信号**である．複合 415 行 top1 +7.71pt，single_domain_top1 +2.53pt，ECE -0.0018，`mean_duration_ms` -100.7（prefix によるトークン増は速度に響かなかった），fallback/dispatch_failure 不変．answer_quality +1.07pt・end_to_end +1.46pt は 3SD=2.6pt のノイズ床内で判定しない．ゲート F1/F2/F3・G1・G2 は全 PASS で，実験は成立している（invalid ではない）．

**判定: rejected（事前登録の判定規則どおり）**．規則は「点推定が低下，**または非退行①（per-domain 20 指標の BH 補正後の有意退行 0 件）に 1 件以上抵触**，または非退行②③④に違反」を rejected と定めている．per-domain 20 指標の BH 補正後の有意差は 3 件で，うち **medical_recall 0.7842 → 0.7178（p=0.003264）が悪化方向**であるため，主基準（+1.0pt 以上かつ p<0.05）を大幅に満たしていても規則上 rejected 以外の判定は取れない（partial・no_effect はいずれも「非退行に違反なし」または「n_d<30 かつ ±0.5pt 以内」を要件とし，本結果はどちらにも該当しない）．**結果を見てから規則を読み替えないという事前登録の趣旨に従い，復元を実施した**: `config.yaml` の `embedding_instruction` 行を削除，`cp models/domain_classifier_pre_iter81_noprefix.joblib models/domain_classifier.joblib`（sha256 `21e16ec6...`），`mise run deploy`．全 10 ノードで `config.yaml` のハッシュ一致（smoke_check）・`embedding_instruction` 不在・artifact `21e16ec6...` を確認し，probe も pass．

**学び 1（送出数が減った構造．事前の仮説は支持された）**: `compound_domain_set_recall` が 0.548193 → 0.526506 と下がり `compound_mean_dispatched_count` が 1.880 → 1.494 に減った原因は，**prefix により確信度分布が鋭くなり，`dispatch_gap_threshold=0.29` による 2 件目以降の送出が起きにくくなったこと**である．`results/20260927_024644/` の `probe_candidates` を基準線と id ペアで比較すると，複合 415 行の rank1−rank2 gap は平均 0.4769 → 0.5764（中央値 0.4449 → 0.6075），**gap < 0.29 の行の割合は 33.49% → 20.48%**，rank1 confidence 平均は 0.6510 → 0.7239 だった．送出内訳も「1 件送出で 1 ドメイン的中」が 236 → 296 行へ増え，「4 件送出で 2 ドメイン的中」が 66 → 46 行へ減っている．**送出予算を揃えると逆転する**: 同じ `probe_candidates` に gap 閾値を掃引して再現すると，新構成は gt=0.40 で mean_k=1.933・set_recall **0.5928** となり，基準線の gt=0.29（mean_k=1.880・0.5482）を **+4.46pt 上回る**．つまり set_recall の低下は埋め込み品質の劣化ではなく，**固定閾値 0.29 が新しい確信度スケールに対して相対的にきつくなったことの帰結**である（gap 閾値は確信度分布のスケールに依存するハイパラであり，埋め込みを変えたら再較正が要る，という一般則）．

**学び 2（medical 退行の正体は「rank 1 → rank 2 の入れ替わり」）**: `metrics.py` の per-domain recall は `selected_domain`（= 固定 top-1）で測る定義であり，実測でも新構成の固定 top-1 medical recall は 0.7178 で journal 記載値と一致する．medical を含む 241 行のうち **23 行が medical を取りこぼし，6 行が新たに取れた**．取りこぼした 23 行で新 artifact の medical の順位を見ると **rank 2 が 19 行，rank 3 が 3 行，rank 4 が 1 行**で，**固定 top-2 で測れば medical recall は 0.9046 → 0.8963 とほぼ不変**（top-3 では 0.9461 → 0.9544 と改善）．吸われた先は education 8 行・business_economics 6 行・computer_science 5 行・history_culture 2 行・natural_science 1 行・social_science 1 行で，特定 1 ドメインへの系統的な混同ではない．**medical の信号が失われたのではなく，prefix が「学術/専門ドメインの同定」という instruction に沿って空間を再配置した結果，medical と隣接ドメインの相対順位が僅差で入れ替わった**と解釈するのが妥当である．なお 2026-09-23 恒久運用ルールにより，medical 固有の閾値・intercept・訓練データ調整による是正は**提案しない**（次レバーは 10 ドメイン共通の特徴量設計で対処する）．

**学び 3（記録の一貫性．申し送り 1 への回答）**: **誤っていたのは `data/MANIFEST.md` の方**である．Iteration 80（`bge_m3`）の判定は journal「Iteration 80 実行済み」節・backlog B127 のいずれも **rejected**（top1 -3.03pt，BH 補正後の有意退行 2 件）で，基準線 `21e16ec6...` への復元まで実施済みと記録されている．MANIFEST だけが Iteration 80 の**実装フェーズ時点の記述**（生成コマンドの `--embedding-model bge-m3`，「Iteration 80 で採用されたはずの bge-m3」）のまま更新されず，あたかも adopted であったかのように読めていた．本フェーズで MANIFEST を訂正した（生成コマンドを `qwen3-embedding:0.6b` へ戻し，訂正の経緯を明記）．**教訓: MANIFEST は実装フェーズで先に書かれるため，rejected で復元した反復では分析フェーズが必ず MANIFEST を巻き戻す責任を負う**（journal/backlog は判定を書くが MANIFEST は書かない，という非対称が今回の食い違いを生んだ）．

**学び 4（実験設計への反省．次レバーへ直結）**: 本反復は「per-domain の非退行を本走後にしか確認していない」ために，1 回の 1,915 問本走（約 100 分 + 10 ノード）を rejected で失った．**per-domain 20 指標の BH 補正は分類器 artifact の argmax replay（決定論的・オフライン）で本走前に完全に予測できる**（`metrics.py` の `compute_domain_recall_mcnemar_test` 等をそのまま流用可能）．Iter79/80/81 で G1（CV）→ G2（discordant n_d）までは型化できていたが，**G2 に per-domain 非退行の事前予測を足す（G2'）**のが次反復以降の標準手順である．

**学び 5（要人間判断）**: 「全体 +3.655pt（p=4.2e-7）を，1 ドメインの recall 退行 1 件で棄却する」という非退行条件①の設計自体に，研究方針として再考の余地がある．ただし判定規則の改定は結果を見てからの事後変更であり，かつ研究の結論に関わる不可逆な方針変更のため，**本フェーズでは規則を変えず rejected を確定させ，規則設計の是非は人間判断に委ねる**（backlog B128 要レビュー）．

**次の一手**: 新レバー **`embedding_view_concatenation` = `prefix_and_noprefix_concat`**（`.claude/research/config.yml` の `levers` 末尾へ追記済み）．学び 2 の「prefix 有り／無しは相補的で，medical は prefix 版でも rank 2 に残っている」という実測を根拠に，2 つのビューを連結した 2048 次元を分類器の特徴量とし，学び 4 の G2'（本走前の per-domain 非退行予測）を事前ゲートに組み込む．

---

