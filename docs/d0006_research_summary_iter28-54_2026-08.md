<!-- research-cycle Iteration 28〜54（収束まで）の実験設定・結果・判定・学びの総括．研究全体の現況を示す最新文書． -->

# d0006: research-cycle 知見総括（Iter28〜54 / 研究収束時点）

## このファイルについて

- **目的**: `d0004`（Iter27 時点で更新停止）以降，研究サイクルが `status="converged"` に到達する
  Iteration 54 までの実験設定・結果・判定・学びを 1 つの参照可能な形に統合する．
- **役割**: **研究全体の現況を示す最新文書**．新しくこのプロジェクトに関わる場合，`d0004` §1〜§2
  （Iter15〜27 の到達点）を読んだあと本書を読めば，収束時点までの全体像が追える．
- **一次情報の所在**:
  - `.claude/research/journal.md` — Iteration 52〜54 の生記録
  - `.claude/research/journal_archive.md` — Iteration 1〜51 の生記録（逆時系列）
  - `.claude/research/experiment_results.json` — 各イテレーションの機械可読な実験結果
    （`e{n}_results`）．旧 `state.json` に蓄積されていたものを 2026-08-05 に分離した
  - `.claude/research/backlog.md` — 自律判断の記録と要人間判断事項
- **記載の原則**: 数値は `experiment_results.json` および journal の rc-analyst 独立計算値を正とする．
  実装者報告値と analyst 再計算値が食い違う箇所は §6 に明記する．

---

## 1. 到達点（収束時点の構成と数値）

### 採用された構成

| 項目 | 値 | 確定したイテレーション |
|---|---|---|
| `routing_method` | `supervised_classifier` | Iter17 |
| `fallback_policy` | `disabled`（`confidence_threshold=0.0`） | Iter28 |
| `classifier_calibration` | `temperature` | Iter31 |
| `classifier_head_adaptation` | `education_boundary_tuning`（intercept_delta=+0.7）<br>＋ `per_class_threshold_optimization`（threshold=0.05） | Iter44 / Iter53 |
| `dispatch_top_k` | 2 | Iter45 |
| `dispatch_candidate_threshold` | 0.0 | Iter45（新設） |
| `aggregation_method` | `max_confidence` | Iter48（llm_judge 棄却により確定） |
| embedding model | `nomic-embed-text`（freeze） | 全期間で固定 |

### 最終的な数値（評価データセット 1600 問）

| メトリクス | Iter31 基準 | 収束時点（Iter53） | 差 |
|---|---|---|---|
| top1_accuracy | 0.6056 | 0.6006 | -0.0050 |
| education_recall | 0.4588 | 0.6000 | **+0.1412** |
| medical_recall | 0.5112 | 0.5056 | -0.0056 |
| ECE | 0.071201 | 0.061493 | -0.009708 |

研究の主目的であった **education ドメインの recall を medical ドメイン基準（0.5112）まで引き上げる**
という目標は達成された（0.6000）．ただし §4 のとおり，これは post-hoc な decision boundary の
平行移動によって得られる上限値でもある．

---

## 2. イテレーション一覧（Iter28〜54）

`flip` は argmax flip rate（単一レバー原則の判定に使う，基準線から予測が反転した割合．**閾値 <15%**）．
`BH退行` は per-domain 20 指標に対する Benjamini-Hochberg 補正後の有意な退行件数．

### 2.1 fallback 廃止と分類器の較正（Iter28〜31）

| Iter | 単一レバー | top1 | edu_recall | flip | 判定 |
|---|---|---|---|---|---|
| 28 | `fallback_policy=disabled` | 0.5850 | — | — | **adopted** |
| 29 | `classifier_calibration=platt` | 0.5956 | — | 11.00% | partial（ECE 0.1675 で目標未達） |
| 30 | `classifier_calibration=isotonic` | 0.5938 | — | 14.31% | partial（medical_recall に BH 退行 1 件） |
| 31 | `classifier_calibration=temperature` | 0.6056 | 0.4588 | — | **adopted**（ECE 0.0712，argmax 不変） |

Iter31 の構成が，以降すべての実験の基準線となる．

### 2.2 訓練データ構成の変更（Iter32〜39）— 全 6 値 rejected

| Iter | 単一レバー | edu_recall | flip | 判定 |
|---|---|---|---|---|
| 32 | `education_proxy_task_revision`（sample_weight 2.0） | 0.4412 | 0.94% | rejected |
| 33 | `education_proxy_task_resampling` 案C（70/40/40） | 0.4412 | — | rejected |
| 34 | `education_proxy_task_resampling` 案A（90/30/30） | 0.4353 | — | rejected |
| 35 | `education_handmade_training_problems`（50 件追加） | 0.4118 | 11.00% | rejected |
| 36 | `education_proxy_task_replacement`（japanese_civics 置換） | 0.0529 | — | rejected（train/eval タスク不一致で崩壊） |
| 37 | `history_culture_japanese_civics_reassignment_to_education` | 0.8824 | 52.50% | **invalid**（label leakage） |
| 38 | `education_hybrid_proxy_and_civics` | 0.4000 | 20.44% | rejected |
| 39 | `class_weight_adjustment`（手動 sample_weight） | 0.4588 | 4.69% | rejected（no-op） |

- **Iter32 の機序**: sklearn の `LogisticRegression(class_weight="balanced")` は `sample_weight` に
  依存して `class_weight` を再計算するため，重み増加とクラスバランス調整が数式レベルで結合し，
  意図と逆に働いた．
- **Iter37 の無効理由**: eval に含まれる `japanese_civics` 150 問が**すべて訓練データに入っており**，
  education recall 100% は暗記の結果であって汎化ではない．`edu_recall=0.8824` という一見劇的な
  改善はこの leakage による．
- **Iter39 が示したこと**: 手動 `sample_weight` は `class_weight="balanced"` の完全な再現であり，
  education_recall・medical_recall とも**数値が一切変わらなかった**．レバーとして no-op である．

### 2.3 embedding 空間の適応（Iter40〜43）— 全 4 値 rejected

| Iter | 単一レバー | edu_recall | medical_recall | flip | BH退行 | 判定 |
|---|---|---|---|---|---|---|
| 40 | `setfit_education_finetune`（full FT） | 0.6529 | 0.3090 | **52.56%** | 13 | rejected |
| 41 | `embedding_adapter_only_lora`（r=16） | 0.5706 | 0.4045 | **35.88%** | 1 | rejected |
| 42 | `embedding_adapter_lora_r8`（r=8） | 0.6235 | 0.4326 | **35.88%** | 2 | rejected |
| 43 | `embedding_adapter_projection_head`（590K） | 0.5529 | 0.3596 | **42.00%** | 15 | rejected |

- **決定的発見（Iter42）**: LoRA r=8 と r=16 は，予測結果も分類器の重みも**ビット単位で同一**だった．
  これは education ドメイン適応に必要な有効自由度が実質 1 主成分（**intrinsic dimensionality <= 8**）
  であることを意味し，rank を下げて flip rate を抑えるという方向が構造的に成立しないことを示す．
- **共通の機序**: 768 次元の embedding 空間を 10 ドメインで共有しているため，education だけを
  分離するには空間の回転が必要になるが，回転は必然的に他ドメインも動かす．表現力を上げた
  Iter43（projection head）でむしろ flip rate が悪化した（42.00%）ことがこれを裏付ける．

### 2.4 分類器ヘッドの後段調整（Iter44, Iter49〜53）— post-hoc 手法

| Iter | 単一レバー | edu_recall | flip | BH退行 | 判定 |
|---|---|---|---|---|---|
| 44 | `education_boundary_tuning`（intercept_delta=+0.7） | 0.5235 | 8.63% | 0 | **adopted** |
| 49 | `education_posthoc_calibration`（logit_bias=+0.3） | 0.5588 | 9.19% | 0 | rejected（改善が有意でない p=0.5443） |
| 50 | `education_posthoc_calibration`（logit_bias=+0.5） | 0.5824 | 11.38% | 0 | rejected（top1 有意悪化 p=0.0014） |
| 51 | `education_per_class_threshold`（threshold=0.3） | 0.8118 | **23.75%** | 8 | rejected |
| 52a | `education_per_class_threshold`（threshold=0.02） | 0.5412 | 0.88% | 0 | **adopted** |
| 52b | `education_per_class_threshold`（threshold=0.05） | 0.5647 | 2.56% | 0 | **adopted** |
| 53 | `per_class_threshold_optimization`（threshold=0.05） | 0.6000 | 2.56% | 0 | **adopted**（正式レバー化） |

- **Iter51 の失敗はスケールの問題**: 確率に +0.3 を再正規化なしで加算すると，確率分布の総和が
  1.0 → 1.3 に変わる．適切な加算量は 0.02〜0.05（2〜5pt の追加質量）である．
- **Iter52/53 は同一の結果**: どちらも Iter44 の予測ファイルに同じ threshold 0.05 を加算しており，
  結果はビット単位で一致する．Iter53 の目的は，この操作を正式なレバー名で config に登録し，
  全レバー試し切り完了を文書化することにあった．

### 2.5 集約方式の比較（Iter45〜48）

| Iter | 内容 | top1 | 判定 |
|---|---|---|---|
| 45 | `dispatch_candidate_threshold` 新設（Y2）＋ majority_vote / top_k=2 へ変更（Y3） | — | 測定系の整備 |
| 46 | `aggregation_method=majority_vote`（top_k=2） | 0.6063 | adopted（条件付き，複合ドメイン recall +19.5pt） |
| 47 | `aggregation_method=max_confidence` clean ベースライン | 0.6031 | max_confidence で十分 |
| 48 | `aggregation_method=llm_judge` | **0.4350** | rejected |

- **Iter45 の意義**: `confidence_threshold` が「fallback 判定」と「dispatch 候補の足切り」という
  二重の責務を持っていたため，Iter27 の集約方式比較は 2 位ノードが一度も適格にならず no-op に
  終わっていた（2 位 confidence の最大値 0.4955 < 閾値 0.5）．責務を分離したことで比較が成立した．
- **Iter46 の単一レバー違反**: `aggregation_method` と `dispatch_top_k` を同時に変更しており，
  厳密には単一レバー原則を満たしていない．Iter47 で clean ベースラインを取り直して是正した．
- **Iter48 の llm_judge**: top1 が -0.1681 と大幅に悪化し，judge が max_confidence の選択を
  上書きしたケースの **84.1% が誤選択**だった．ECE・Brier が改善して見えるのは，judge が
  誤った選択を高い確信度で行うためであり，見かけ上の改善である．

### 2.6 収束（Iter54）

`classifier_training_data_composition=education_soft_label_distillation` が rc-investigator により
提案されたが，計画フェーズで **single_lever_compatibility: 低**と評価され，**実験を実行せずに棄却**
された．全 retraining 実験（Iter32〜38, Iter40〜43）の flip rate が 20〜53% であり，soft label
distillation だけが 15% を下回る根拠がないためである．`status="converged"` へ移行した．

---

## 3. レバー別の最終状態

| レバー | 値の数 | 最終状態 |
|---|---|---|
| `fallback_policy` | 1 | adopted（Iter28） |
| `classifier_calibration` | 3 | 全値試行済み，`temperature` adopted（Iter31） |
| `classifier_training_data_composition` | 6 | **全値 rejected**（Iter32〜38） |
| `class_weight_adjustment` | 1 | rejected（no-op，Iter39） |
| `embedding_adaptation` | 4 | **全値 rejected**（Iter40〜43） |
| `classifier_head_adaptation` | 5 | 3 adopted / 1 exhausted / 1 skip，**クローズ** |
| `aggregation_method` | 3 | 全値試行済み，`max_confidence` adopted（Iter48） |

---

## 4. post-hoc 手法の天井（本研究の中心的な結論）

education_recall の改善は，次の 2 段階の**平行移動**のみで得られている．

| 段階 | 操作 | education_recall | 増分 |
|---|---|---|---|
| 基準 | Iter31（temperature 較正のみ） | 0.4588 | — |
| 第1段 | ＋ intercept_delta = +0.7（Iter44） | 0.5588 | +0.1000 |
| 第2段 | ＋ threshold = 0.05（Iter53） | 0.6000 | +0.0412 |
| | **合計** | **0.6000** | **+0.1412** |

**intercept シフトと threshold 加算は同一の原理である**．確率空間での線形加算は，raw logit 空間での
intercept シフトと同じく decision boundary を**平行移動**させるだけであり，boundary の**方向**
（係数ベクトル）は変えない．したがって boundary を越えない位置にある教育質問の誤分類は，どれだけ
平行移動しても解消しない．これが `education_recall ≈ 0.60` という天井の正体である．

天井を突破するには decision boundary の**回転**，すなわち classifier retraining が必要になるが，
retraining は訓練データの変更を伴い，argmax flip rate <15% という単一レバー原則と構造的に両立しない
（Iter32〜43 で実証済み）．**単一レバー原則を守る限り，この構成での改善余地は尽きている．**

---

## 5. 方法論上の学び（繰り返し観測された失敗の型）

1. **レバーを読むコードに実行が到達しない（no-op 実験）**: Iter16, 20, 21, 22, 27, B35 に加え，
   Iter39 も実質 no-op だった．`d0004` §4 は「6 イテレーション以上この型の失敗が続いている」と
   総括している．計画フェーズで「そのレバーを読むコード行と，そこへ到達する条件」を明記することが
   対策として導入された（Iter54 の計画にもこの記述がある）．
2. **config の 1 フィールドが複数の責務を持つと比較実験が成立しない**: `confidence_threshold` の
   二重責務が Iter27 の不成立を招き，Iter45 の責務分離で解消した．
3. **評価データと訓練データの重なりを機械的に検査する必要がある**: Iter37 の label leakage は，
   eval の 150 問がすべて訓練データに含まれるという明白な重複であり，事前チェックで防げた．
4. **見かけの指標改善を機序で検証する**: Iter48（llm_judge）は ECE・Brier が改善したが，これは
   誤答を高確信度で選ぶことによる見かけ上の改善だった．
5. **単一レバー原則は「変更したフィールド数」ではなく「予測がどれだけ動いたか」で測る**: argmax
   flip rate を判定基準に据えたことで，config 上は 1 値の変更でも実質的に大規模な変更となる
   embedding 適応を，一貫した基準で棄却できた．

---

## 6. 数値の食い違いに関する注記

Iter53 で rc-analyst が予測ファイルから独立に再計算したところ，実装者の報告値と系統的な差が見つかった．

- **母数の違い**: 実装者は Iter44 の education_recall を 0.5235・medical_recall を 0.5000 と報告したが，
  ファイル直接計算では 0.5588・0.5281 である．**差分**（+0.0412 等）は両者で一致するため，recall の
  母数の取り方が異なると推測される．
- **McNemar の chi2 が標準公式と一致しない**: 実装者の chi2 値は標準公式 `(a-b)^2/(a+b)` と合わない
  （例: a=13, b=7 で実装者 chi2=0.8000，公式では 1.8000）．
- **Iter53 以降の記述では analyst の独立計算値を正式値として採用している**．本書の §1 の最終数値も
  analyst 値に従う．§2 の表は `experiment_results.json` の記録値（実装者報告値）であり，Iter44 の
  education_recall がここでは 0.5235 となっている点に注意すること．

この不整合の恒久対策（rc-implementer への McNemar 計算チェックリスト導入）は未実施である．

---

## 7. 要人間判断の最終確定（2026-08-05，backlog B84）

`backlog.md` の B81・B82・B83 が残した 4 件の要人間判断は，ユーザーへ選択肢とメリット・デメリットを
提示のうえ確認し，**全項目「現状維持」で確定した**．研究サイクルはこれをもって完全に終了する．

| 論点 | 決定 |
|---|---|
| **education_recall 基準値の再定義**: `medical_recall=0.5112` を基準にしているが，medical は JMMLU に直接対応するタスク（college_medicine, professional_medicine）を持つ一方，education は代理タスク（sociology, high_school_psychology, moral_disputes）しか持たない．代理タスク側の recall 上限は sociology=0.625 / high_school_psychology=0.438 / moral_disputes=0.435 であり，そもそも同じ土俵に乗せてよいかが疑わしかった． | **現状維持**．medical と同一基準のまま． |
| **classifier retraining への移行可否**: 天井（§4）の突破には boundary の回転が必要であり，単一レバー原則の緩和を承認するか否かの判断が要った．あわせて flip_rate の許容範囲（<15% 厳守か <20% まで許容か）も論点だった． | **移行しない**．education 精度改善はここで打ち止め．flip_rate は `<15%` を厳守．単一レバー原則は緩和しない． |
| **JMMLU 外部の教育固有タスク追加の feasibility**: Iter53/54 の調査では，日本語の教育実務に固有の4択タスクは発見できなかった（EduBench・Pedagogy Benchmark・Dr.Academy はいずれも日本語未対応，全国学力テストは教育行政を含まない）．`japanese_civics` が唯一の候補だが label leakage リスクが高い． | **追加調査しない**． |
| **B27 `[needs-human 2026-07-26]` の棚卸し**: 「作業ツリーに journal 未記録の変更が残っている」件が解消された記録が見当たらなかった． | **陳腐化と判断し解消**．config.yaml は Iter15〜54 で全面刷新済みのため，指摘対象の設定はもう存在しない． |

詳細は `.claude/research/backlog.md` の B84 を参照．今後 `research-cycle continue` が呼ばれても，
新規実験は開始せず，本節と B83・B84 を提示して終了済みである旨を報告すること．

---

## 8. 関連文書

| 文書 | 対象範囲 | 位置づけ |
|---|---|---|
| `docs/encounter_expert_mesh_design.md` | 全期間 | 技術設計書（v2）．評価軸と研究の問いの定義元 |
| `docs/d0001_literature_survey_2026-07.md` | Iter1〜14 | 文献調査と実測の一次記録（歴史的記録） |
| `plans/p0001_research_direction_2026-07.md` | Iter1〜14 | d0001 に基づく方針決定（歴史的記録） |
| `docs/d0002_research_cycle_findings_2026-07.md` | Iter1〜22 | 知見総括（歴史的記録） |
| `docs/d0003_next_experiments_2026-07.md` | Iter22 時点 | 実験計画．大半が実施済み（歴史的記録） |
| `docs/d0004_research_status_and_direction_2026-08.md` | Iter15〜27 | 現況文書．Iter27 で更新停止 |
| `docs/d0005_retraining_analysis_2026-08.md` | Iter53 時点 | retraining 移行分析．本書 §4 の詳細版 |
| **本書 `d0006`** | **Iter28〜54** | **収束時点の最新総括** |
