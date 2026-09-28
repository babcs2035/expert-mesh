## Iteration 94: 訓練集合と評価集合で重複する設問 64 行を除去し基準線を引き直す

### 調査 (Iter94)

backlog B155 (b) が事前登録したレバー `train_eval_question_overlap_removal` を実施するにあたり，
(1) 重複 64 行の同定条件の再現・確定，(2) 汚染（train/test contamination）の扱いに関する文献調査，
(3) 除去後の基準線のオフライン replay による事前予測，の 3 点を行った．
**評価集合 `data/dataset.jsonl` は読むだけで変更していない．実機ノード wafl500〜509 も不使用**
（新規の埋め込み計算は 0 件．すべて既存キャッシュから再構成した）．作業は制御ホスト wafl-ctrl5 上で行い，
中間生成物は `/tmp/iter94/` にのみ書いた（`data/`・`models/`・`results/` は一切変更していない）．

**(1) 重複 64 行の同定条件を確定した —— 「設問文 `query` の完全一致」で十分である**

`data/classifier_train_iter92_civics_aligned.jsonl`（2,339 行）と `data/dataset.jsonl`（3,750 行，
sha256 `2114e048...`）を突き合わせた結果は次のとおり（スクリプト `/tmp/iter94_overlap.py`）．

- **完全一致（`query` の文字列同値）: 64 行．正規化一致（NFKC ＋ 全空白除去 ＋ lowercase）でも 64 行**で，
  両者は同一の集合である．すなわち**表記ゆれによる取りこぼしは無く，同定条件は完全一致で確定してよい**
  （正規化を入れても新たな重複は 1 行も現れない）．これは両者が同じ `build_dataset.py` の
  `_format_jmmlu_query()` を通って生成されているためで，`build_dataset.py:1322` の
  `exclude_queries=eval_queries` による重複排除も同じ完全一致を前提にしている．
  **よって除去規則は本番コードの重複排除規則と同一であり，新しい判定規則を導入しない．**
- 64 行はすべて**ラベルも一致**（訓練側 `domain` が評価側 `expected_domains` に含まれる）．
  **評価側 64 行はすべて単一ドメイン行で，複合設問（`is_compound=true` の 730 行）との重複は 0 行**．
  訓練側 64 行はすべて通常行（`*-train-*`）で，**hard negative 行（`*-hardneg-*`）・
  Iter92 追加の civics 行（`education-civics-*`）は 1 行も含まない**．
- 内訳（訓練ラベル ＝ 評価ラベル）: `education` 46 行，`history_culture` 18 行．他の 8 ドメインは 0 行．
  JMMLU タスク単位では moral_disputes 19 / sociology 14 / high_school_psychology 13 /
  japanese_idiom 5 / world_history 3 / high_school_geography 3 / japanese_history 2 / prehistory 2 /
  high_school_european_history 2 / japanese_geography 1．
- 除去後の訓練行数は **2,339 → 2,275 行**．ドメイン別は `education` 284→238，`history_culture` 228→210 で，
  **他 8 ドメインは不変**（`legal` は 77 行のまま減らない）．config.yml のレバー注記が
  「`legal` など小さいクラスが過度に痩せないか事前に確認せよ」と求めていた点は**問題なし**である
  （最小クラスは `legal` 77 行のまま，`sample_weight` は `n/(K*n_d)` で自動的に再計算される）．

**(2) 基準線 `results/20260928_111644/` における重複 64 行の実測 —— 「最大 1.7pt」は上限であって推定値ではない**

B154 (A1) の「最大 1.7pt の上振れ」は **64/3,750 = 1.707pt**，すなわち「重複 64 行が全問正解でありかつ
その正解が全て記憶によるものだった場合」の**理論上限**である．基準線の実測はこれよりずっと小さい．

| 部分集合 | n | 基準線 top1 |
|---|---|---|
| 全体 | 3,750 | 0.835467 |
| **重複 64 行** | 64 | **0.6875 (44/64)** |
| 残り | 3,686 | 0.838036 |

**重複行の精度は全体平均より低い**（0.6875 < 0.8380）．これは 64 行が recall の低い `education`（全体 0.497）
に偏っているためで，ドメインを揃えて比べると記憶の痕跡が見える: `education` は重複 46 行 0.652 に対し
残り 304 行 0.474（**+17.8pt**），`history_culture` は重複 18 行 0.778 に対し残り 332 行 0.913（−13.5pt，
ただし n=18 で偶然の範囲）．

**(3) 文献調査（tavily-search）—— 「訓練側から除く」か「評価側から除く」か**

- Elangovan, He & Verspoor (2021), *Memorization vs. Generalization: Quantifying Data Leakage in NLP
  Performance Evaluation*, EACL 2021, pp.1325-1335, <https://aclanthology.org/2021.eacl-main.113/>
  （arXiv:2102.01818）—— 訓練集合と評価集合の重複は記憶と汎化を混同させ，報告精度を汎化性能の推定として
  読めなくする．**重複の量と，重複部分／非重複部分それぞれの精度を分けて報告することを推奨している**．
  本反復の (2) はまさにこの分解にあたる．
- Lee et al. (2022), *Deduplicating Training Data Makes Language Models Better*, ACL 2022,
  <https://aclanthology.org/2022.acl-long.577/>（arXiv:2107.06499）—— 標準的なコーパスでは
  **train-test overlap が検証データの 4% 超に及び，重複を訓練側から除くことが基本的な処方**である．
  本研究の重複率は評価集合の 1.71%（64/3,750）でこれより小さいが，除去の向き（訓練側から除く）は同じ．
- Brown et al. (2020), *Language Models are Few-Shot Learners*, NeurIPS 2020, 第 4 節
  *Measuring and Preventing Memorization of Benchmarks*,
  <https://proceedings.neurips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf>
  —— **訓練側の重複排除が事後的に不完全だった場合の対処として，評価集合側から汚染行を除いた
  「clean subset」での精度を併記する**手法を採っている．すなわち文献上，是正の向きは 2 通りある．

**本研究でどちらを採るか（判断と理由）**: **訓練側から除去する**（Lee et al. 側）．

- (i) 評価集合 `data/dataset.jsonl`（sha256 `2114e048...`）は Iter85/90 で拡充したのち，Iter89〜93 の
  全基準線が参照する固定点である．評価側を削ると過去の全 top1 と比較不能になり，
  実機での再取得（1 回約 155 分）が基準線の本数だけ必要になる．
- (ii) 訓練側を除くほうが是正として強い．評価側だけを削っても，分類器は依然として当該設問で学習された
  ままであり（近傍の設問への間接的な影響が残る），Elangovan et al. の言う記憶と汎化の分離は
  完全にはならない．
- (iii) ただし Brown et al. の clean subset 報告は**併記として有用**なので，分析フェーズでは
  全 3,750 行の top1 に加え，**重複 64 行を除いた 3,686 行での top1 も必ず併記する**
  （Iter92 で 8 行について事前登録したのと同じ扱い．B152 要レビュー (b) の踏襲）．

**(4) オフライン replay による事前予測（G0-d 相当．事前登録の予測であり，本走の代替ではない）**

`/tmp/iter94/predict_replay.py` で，除去後 2,275 行の訓練特徴を **Iter92 と同一の規則**
（`embcache_train_iter89_qwen3-embedding_4b{,__p1}.npy` の該当行 ＋
`embcache_train_iter92_qwen3-embedding_4b_new20{,__p1}.npy`，新規埋め込み 0 件）で再構成し，
`scripts/train_domain_classifier.py` の `train_classifier()` をそのまま呼んで分類器を組み，
評価 3,750 行の argmax を現行 artifact（`models/domain_classifier.joblib`，sha256 `98da6f2d...`）と比較した．
Iter91 で replay が分類器 argmax 精度を小数 6 桁一致で再現することは実証済みである．

- **replay top1: 0.836267 → 0.834933（Δ = −0.133pt）．McNemar: discordant 34 / 29，p = 0.614．**
- **重複 64 行: 0.6875 → 0.5781（−10.9pt ＝ 7 行）．残り 3,686 行: 0.8389 → 0.8394（+0.054pt）．**
- per-domain recall（単一ドメイン行）の変化は最大でも ±0.9pt 以内:
  history_culture +0.86pt / social_science +0.68 / computer_science +0.33 / education +0.29 /
  business_economics・legal・mathematics・natural_science ±0.00 / general −0.57 / medical −0.57．

**この分解が本反復の中心的な知見である**: 除去の影響は**ほぼ完全に重複 64 行の中に閉じており**
（7 行分の正解が消える），残り 3,686 行はむしろ微増する．すなわち
**測定系の上振れの実測値は約 0.19pt（7/3,750）であり，B154 が挙げた理論上限 1.7pt の約 1/9 である**．
`education` の recall が上がらない（+0.29pt）点も重要で，**Iter92 で観測した education の低 recall は
リークで嵩上げされていた見かけの値ではなく，除去後も 0.497 のまま残る真の弱点**である．

### 計画 (Iter94)

**単一レバー（事前登録）**: `train_eval_question_overlap_removal` =
**`remove_duplicate_question_rows_and_rebaseline`**（backlog B155 (b) の事前登録どおり）．

**変更点は 1 つだけである**: 訓練集合 `data/classifier_train_iter92_civics_aligned.jsonl` から，
評価集合 `data/dataset.jsonl` と `query` が完全一致する 64 行を除去し（2,339 → 2,275 行），
同一の手順で再訓練して本走 1 回を行う．**本レバーは精度向上を狙うものではなく，測定系の是正である．**
したがって B151 (d)4 の足切り（訓練集合内 CV で +2.3pt）は適用しない（B155 (b)・本タスクの指示）．

**仮説**: 基準線 `results/20260928_111644/`（top1 = 0.835467）は，訓練で見た設問 64 行を評価に含むため
上振れている．その上振れの実体は「重複行のうち記憶によってのみ正解している分」であり，
実測では**約 7 行（0.19pt）**である．重複行を訓練から除けば，(a) 重複 64 行の部分精度が
0.6875 → 0.58 付近へ下がり，(b) 残り 3,686 行の精度は ±0.5pt 以内で変わらず，
(c) 全体 top1 は 0.835 → 0.834 付近（Δ ≈ −0.13pt）へ動く．

**固定する構成（Iter92/93 の最良構成をそのまま維持）**:
埋め込み `qwen3-embedding:4b` の 2 ビュー連結（5,120 次元），instruction prefix 現行値，
`CalibratedClassifierCV(method="temperature", cv=5, ensemble=True)`，`C`=1.0，
`class_weight=None` ＋ドメイン均衡 `sample_weight`（`n/(K*n_d)`，除去後の行数で再計算），
`confidence_threshold=0.0`，`dispatch_top_k=1`，`routing_method=supervised_classifier`，
評価集合 `data/dataset.jsonl`（3,750 行，sha256 `2114e048...`，**変更しない**），
`classifier.py` の推論経路・`config.yaml`・`router.py`・`http_server.py` は無変更．
**基準線は `results/20260928_111644/`（3,750 行，top1 = 0.835467）**．

**成功条件（事前登録．B153 要レビュー (B) / B154 (A4) に従い，効果量と有意性を分けた 2 軸で定義する）**

まず，是正が成立したことを確かめる**必須ガード** C1〜C3（いずれか 1 つでも不成立なら判定は `invalid` とし，
artifact をロールバックして原因を調査する）．

- **C1（重複 0 行）**: 新訓練ファイルと `data/dataset.jsonl` の設問文の一致が，
  **完全一致・正規化一致（NFKC ＋ 全空白除去 ＋ lowercase）のいずれでも 0 行**であること．
- **C2（変更の最小性）**: 新訓練ファイルの行数が **2,275 行**で，旧ファイルとの差分が
  **`/tmp/iter94_dup_train_ids.json` の 64 個の id の削除のみ**であること（行の並び・各行の
  `id`/`query`/`domain` は他に 1 文字も変わらない）．コード（`*.py`）・`config.yaml`・
  `data/dataset.jsonl`・評価条件の変更が 0 件であること．
- **C3（本走の健全性）**: 本走が 3,750 行を完走し，`dispatch_failed` 件数・`used_fallback` 件数が
  基準線と同水準であること（従来の E1〜E5 相当のガード，`mise run deploy` 後の全 10 台での
  artifact sha256 一致確認を含む）．

そのうえで，**判定語は Δtop1（全 3,750 行，新 − 基準線）の効果量と McNemar の有意性の 2 軸で決める**．

| | McNemar p < 0.05 | p ≥ 0.05 |
|---|---|---|
| **\|Δtop1\| < 0.5pt** | `corrected_negligible`（是正完了．基準線の水準は実質不変） | `corrected_negligible`（同左．**事前予測はここ**） |
| **−1.7pt < Δtop1 ≤ −0.5pt** | `corrected_bias_removed`（上振れが実測された） | `corrected_bias_removed_weak`（向きは一致するが有意でない） |
| **Δtop1 ≤ −1.7pt または Δtop1 ≥ +0.5pt** | `unexpected_shift` | `unexpected_shift` |

`unexpected_shift` は「重複 64 行の除去だけでは説明できない変化」を意味し，判定を確定させずに
原因を調査する（C5 の帰属確認と replay との乖離を突き合わせる）．
**下限 −1.7pt は理論上限 64/3,750 に対応し，これを超える低下は除去以外の要因を疑うべき水準である．**

加えて，**是正が意図どおりの場所で起きたことの帰属確認**を副基準として事前登録する
（主基準の判定語は変えないが，不成立なら分析フェーズで必ず論じる）．

- **C4（replay との整合）**: 実測 Δtop1 と replay 予測 Δ = −0.133pt の乖離が **0.5pt 以内**であること
  （過去の反復では replay と本走の差は 3 行程度）．
- **C5（リークの帰属）**: 重複 64 行の部分精度が下がり（予測 0.6875 → 0.58 付近），
  かつ**残り 3,686 行の Δ が ±0.5pt 以内**であること．これが成り立てば，
  「上振れは重複行の記憶に閉じていた」という仮説が支持される．
- **C6（clean subset の併記）**: Brown et al. (2020) 第 4 節に倣い，
  **重複 64 行を除いた 3,686 行での top1** を基準線側・新側の双方について報告する．
  以後の反復はこの 3,686 行基準の値も併せて引き継ぐ．

**非退行の扱い（従来と異なる点を明記する）**: per-domain 20 指標の BH 補正後の有意退行を従来どおり
集計・報告するが，**本レバーは是正であるため，退行が出ても撤回しない**．
退行が出た場合は，それが重複行の分布（`education` 46 行・`history_culture` 18 行の除去）で
説明できるかを必ず論じる．replay 上は最大の低下が general / medical の −0.57pt で，
有意退行は生じない見込みである．

**artifact の扱い（事前登録）**: 新 artifact を採用して `models/domain_classifier.joblib` を更新し，
**新しい基準線とする**（top1 が下がっても是正なので戻さない）．
旧版は `models/domain_classifier_pre_iter94_dedup.joblib` へ退避する（従来どおり可逆にする）．
ただし判定が `invalid` または `unexpected_shift` かつ C5 不成立の場合はロールバックする．

**運用上の遵守事項**: 本体実験（wafl500〜509 でのルーティング本走）以外の処理はすべて制御ホスト
wafl-ctrl5 で行う．ドメイン固有の後付け補正は追加しない（重複判定は設問文の完全一致という
10 ドメイン共通の規則であり，2026-09-23 恒久ルール (1) に適合する．`education` と `history_culture`
にしか行が無いのは実測でそこにしか重複が無いためである）．棄権・エスカレーション系のレバーには
着手しない．**上記 (4) の事前 replay は予測であって本走の代替ではなく，本走はフルスペック
（3,750 行，実機 10 台）で 1 回行う．**

### Iteration 94 実行済み

**変更（単一レバー）**: `train_eval_question_overlap_removal` =
`remove_duplicate_question_rows_and_rebaseline`．訓練集合から評価集合と設問文が完全一致する 64 行を
除去した `data/classifier_train_iter94_dedup.jsonl`（2,275 行，sha256 `d6b23735...`）を新規作成し，
同一手順で再訓練して `models/domain_classifier.joblib` を更新（新 sha256 `2f801357...`，旧版は
`models/domain_classifier_pre_iter94_dedup.joblib` へ退避）．`data/MANIFEST.md` に Iteration 94 節を追記．
**コード変更 0 件**（`config.yaml`・`classifier.py`・`router.py`・`http_server.py`・`data/dataset.jsonl`
（sha256 `2114e048...`）は不変を実測確認）．本走は `results/20260928_160921/`．

**必須ガード**: C1（完全一致・正規化一致とも重複 0 行）PASS ／ C2（差分は指定 64 id の削除のみ，
行順保存，コード変更 0 件）PASS ／ C3（3,750 問完走，wafl500〜509 全 10 台で artifact sha256 一致，
smoke_check PASS，`dispatch_failed` 3→1 件，`used_fallback` 両者 0 件）PASS．
分析フェーズでも `results.jsonl` から top1・McNemar・部分集合精度を独立に再計算し，
rc-executor の報告値と小数 6 桁まで一致することを確認した．

**結果（基準線 `results/20260928_111644/` top1 = 0.835467）**

| 指標 | 基準線 | 新 | Δ |
|---|---|---|---|
| top1（全 3,750 行） | 0.835467 | **0.833067** | **−0.24pt** |
| **重複 64 行** | 0.6875 | **0.59375** | −9.375pt |
| **clean subset 3,686 行（C6）** | 0.838036 | **0.837222** | −0.081pt |

Wilson 95%CI [0.820791, 0.844660]．McNemar（discordant 34/25，両側二項検定）**p = 0.2976**．
参考: single_domain 0.843377（n=3,020），compound_domain 0.790411（n=730），ECE 0.0318，AUROC 0.8337．
per-domain recall は education 0.4556→0.4234（−3.23pt，未補正 p=0.0150），medical +1.59pt（p=0.0614），
他 8 ドメインは ±0.4pt 以内．precision は全 10 ドメイン ±2.2pt 以内で全 p>0.4．
**recall/precision 計 20 指標の BH 補正（q=0.05）後の有意な指標は 0 件**．
副基準は C4（replay 予測 −0.133pt と実測 −0.24pt の乖離 0.107pt ≤ 0.5pt）PASS，
C5（影響が重複 64 行に集中し，残り 3,686 行の Δ は ±0.5pt 以内）PASS，C6 併記済み．

**判定: `corrected_negligible`（採用．是正完了・基準線を新値へ引き直す）**

事前登録の 2 軸表で |Δtop1| = 0.24pt < 0.5pt かつ p = 0.2976 ≥ 0.05 のセルに該当し，
これは計画時に「事前予測はここ」と名指ししたセルそのものである．ノイズと信号の切り分けとして，
Δ = −0.24pt は**再現性の床 ±0.25pt（Iter27 で確定）を下回っており，単独では実行間変動と区別できない**．
すなわち「除去によって top1 が有意に下がった」とは言えず，「基準線の水準は実質不変のまま，
測定系のバイアスだけが除かれた」と読むのが正しい．ただし Δ の符号・大きさは replay 予測（−0.133pt）と
同じ向き・同オーダーであり，かつ C5 により低下分が重複 64 行にほぼ完全に帰属するため，
**ノイズだけで説明されるのではなく「小さな真の効果 ＋ 床以下の変動」の重ね合わせ**と見るのが妥当である．
非退行条件は事前登録どおり集計・報告したうえで，**是正レバーであるため退行が出ても撤回しない**方針に従う
（実際には BH 補正後の有意退行 0 件であり，撤回判断は発生していない）．

**学び**

1. **リークの帰属は重複行の中に閉じていた．** 重複 64 行は −9.375pt（6 行）落ちたのに対し，
   残り 3,686 行は −0.081pt とほぼ不変であった．訓練側から 64 行を抜いても近傍の非重複設問へは
   波及しない，すなわち**この分類器の「記憶」は設問単位で局所的**である．
   言い換えると，上振れの実体は約 6 行（0.16pt）であり，
   **B154 (A1) の「最大 1.7pt の上振れ」は理論上限（64/3,750）にすぎず，実測はその約 1/10 だった**．
   予測（replay 7 行 ＝ 0.19pt）ともよく合う．論文で「汚染により最大 1.7pt 上振れていた」と書くのは
   誤りで，正しくは「汚染分は 0.2pt 未満であり，是正後も結論は変わらない」である．
2. **今後の論文化で用いる基準線の数値を確定した．**
   主表は**新 artifact の全 3,750 行 top1 = 0.833067**（Wilson 95%CI [0.8208, 0.8447]）を用いる．
   Brown et al. (2020) 第 4 節に倣う clean subset の併記値は **3,686 行 top1 = 0.837222**．
   Iter89〜93 の過去反復の値は汚染込みであるが，汚染の寄与が 0.2pt 未満で再現性の床 ±0.25pt 以下である
   ことを本反復が実測したため，**過去反復との Δ 比較はそのまま有効**（再走は不要）である．
   以後の反復は基準線を `results/20260928_160921/`（0.833067）へ切り替える．
3. **`education` recall の −3.23pt は「リーク剥落」ではなく検出力不足の範囲．**
   除去 64 行のうち 46 行が education だったため退行の向きは予想どおりだが，
   未補正 p=0.0150 は 20 指標の BH 補正後に有意ではなくなる（education は n=350 程度で 1 行 ≈ 0.29pt，
   −3.23pt は 11 行に相当し，この規模のドメインでは実行間で普通に動く幅である）．
   Iter94 の事前 replay が education +0.29pt を予測していたのに実測は −3.23pt だった点は
   乖離として記録に残すが，**C4（全体 Δ の乖離 0.107pt）が PASS している以上，
   per-domain レベルの予測精度が全体ほど高くないというだけの話**であり，是正の成否には影響しない．
   重要なのは方向性で，**education の低 recall（0.42〜0.50）はリークで嵩上げされた見かけの値ではなく，
   除去後も残る真の弱点**であることが本走で追認された（Iter92 の学びと B156 の予測を支持）．
4. **測定系の是正レバーには精度レバーの足切り（訓練集合内 CV +2.3pt）を当てはめてはいけない**という
   運用が実地で機能した．本レバーは「上がらないこと」が成功であり，効果量 × 有意性の 2 軸表に
   `corrected_negligible` という**低下も無変化も成功と読める判定語**を事前登録しておいたことで，
   結果を後付けで解釈する余地なく締められた．B153 要レビュー (B) / B154 (A4) の宿題はこれで完了する．
5. **config の `levers` はこれで全て試し切った**（本レバーが末尾）．分類器側の手は Iter79〜93 で一巡し，
   測定系の是正も本反復で完了したため，次イテレーションは調査・計画フェーズからの再探索
   （停止条件 (2)）で始める．詳細と候補は backlog B157 に記録した．

## Iteration 93: 分類器のラベル粒度を JMMLU タスク単位へ細分化しドメインへ写像する

### 調査 (Iter93)

Iter92 の学び 3（`education` は 4 タスクにまたがる多峰クラスで単一の線形境界に収まっていない，
という仮説）を出発点に，backlog B153 (e) が指定したレバー `classifier_label_granularity` を検討した．
本フェーズは (1) 文献調査，(2) 訓練集合内 CV による事前見積り（B151 (d)4 の足切り）の 2 本立てで行った．
**評価集合 `data/dataset.jsonl` には一切触れていない．実機ノード wafl500〜509 も不使用**（特徴量は
既存キャッシュから再構成したため Ollama 呼び出しも発生していない）．

**(1) 文献調査（tavily-search）で分かったこと**

- Silla & Freitas (2011), *A survey of hierarchical classification across different application domains*,
  Data Mining and Knowledge Discovery 22(1-2):31-72,
  <https://www.cs.kent.ac.uk/people/staff/aaf/pub_papers.dir/DMKD-J-2010-Silla.pdf> ——
  「葉クラスで学習して親へ写像する」構成は *flat classification approach* として定式化済みの
  標準的な比較対象である．本レバーはこの flat approach そのものであり，新規手法ではない．
- Chen, Ding & Marculescu (2018), *Understanding the Impact of Label Granularity on CNN-Based
  Image Classification*, IEEE ICDMW 2018, pp.895-904, <https://par.nsf.gov/servlets/purl/10121660>
  —— CIFAR-10 / CIFAR-100 / ImageNet で細粒度ラベルによる学習が粗粒度の訓練精度・汎化精度の
  **双方**を改善したと報告している．本レバーの期待の根拠にあたる肯定的な先行研究．
- Novack, McAuley, Lipton & Garg (2023), *CHiLS: Zero-Shot Image Classification with Hierarchical
  Label Sets*, ICML 2023, arXiv:2302.02551,
  <https://proceedings.mlr.press/v202/novack23a/novack23a.pdf> —— (i) 各クラスのサブクラス集合を作り，
  (ii) サブクラスをラベルとして予測し，(iii) 予測サブクラスを親へ写し戻す，という 3 段構成で
  superclass 精度が改善する．さらに superclass 確率 × subclass 確率の積を採る変種も提示している
  （後述の CV で腕 E として測った）．
- **Pirovano et al. (2025), *The Advantage of Fine-Grained Training*, arXiv:2509.05130
  （Scientific Reports, 2026, doi:10.1038/s41598-026-64362-6），<https://arxiv.org/abs/2509.05130>
  —— 本反復の結果を最もよく説明する文献**．「細粒度ラベルでの学習は**普遍的には**精度を改善しない．
  効果は (a) データの幾何とラベル階層の関係，具体的には細粒度タスクと粗粒度タスクが要求する決定境界の
  重なり具合（著者らの言う *boundary redundancy*），(b) データセット規模，(c) モデルの容量
  （過剰パラメータ化の度合い）に依存し，細粒度学習が有利な領域と粗粒度学習が有利な領域を分ける
  遷移が存在する」．**すなわち「効くか効かないかはデータ側の構造で決まるので測るしかない」**．

この最後の知見から，**本レバーは着手の前に訓練集合内 CV で測るべき典型例**と判断した
（B153 (e) が課した B151 (d)4 の足切りとも整合する）．

**(2) 訓練集合内 CV の設計**

- 特徴量: `data/embcache_train_iter89_qwen3-embedding_4b{,__p1}.npy`（2,319 行）と
  `data/embcache_train_iter92_qwen3-embedding_4b_new20{,__p1}.npy`（20 行）から，Iter92 の
  `/tmp/iter92/retrain.py` と同一の規則で 2,339 行 × 5,120 次元を再構成した（新規の埋め込み計算なし）．
- サブクラスラベル: 設問文をキーに `/tmp/expert-mesh-cache/JMMLU.zip` へ逆引きし，
  **2,339 行すべてが 56 タスクのいずれかに一致（未一致 0 行）**，かつ
  **現行 `build_dataset.py:_DOMAIN_TASK_MAP` との食い違い 0 行**であることを確認した
  （Iter92 の是正で写像ずれが解消されたことの独立な確認にもなっている）．
  hard negative 行（`*-hardneg-*`）も全て JMMLU 由来でタスクを持ち，
  B153 (e) が挙げたリスク (b)「サブクラス割り当て規則の未定義」は発生しなかった．
- クラス構成: 56 タスク，1 タスクあたり **15〜95 行**（最小 `professional_medicine` 15 行）．
  `CalibratedClassifierCV(cv=5)` の内側 fold にも 2 行以上が残るため，リスク (a) の懸念のうち
  「fold が組めない」事象は起きなかった．
- 外側 5-fold StratifiedKFold（層化キーは JMMLU タスク）× seed 0/1/2 の計 15 fold．
  `sample_weight` は本番 `_extract_sample_weights()` と同じ**ドメイン均衡重み** `n/(K*n_d)`（K=10）を
  全腕で共通に使い，変えるのは学習ラベルの粒度と写像規則だけにした（単一レバー原則）．

### 計画 (Iter93)

**単一レバー（事前登録）**: `classifier_label_granularity` =
**`jmmlu_task_level_subclass_then_sum_to_domain`**．

B153 (e) が計画フェーズへ委ねた「argmax 写像のみ／タスク確率のドメイン合算のどちらか 1 つを
事前に決める」という論点は，**合算（sum）**を選ぶ．理由は 3 点である．
(i) CV で合算が argmax 写像を上回った（後述），
(ii) 合算は潜在サブクラスに関する周辺化そのもので，`classifier.py:estimate_confidence_classifier()` が
前提とする「10 次元で総和 1 の確率ベクトル」を構成上そのまま保てる（argmax 写像では 10 次元確率を
別途こしらえる必要があり，較正の意味が変わる），
(iii) 10 ドメインへ同一の規則で適用するため 2026-09-23 恒久ルール (1) に抵触しない．

**仮説**: `education` の recall が 0.65 付近で頭打ちなのは，4 つの JMMLU タスクにまたがる多峰クラスを
単一の線形境界で表現しているためである．サブクラス単位で学習して確率をドメインへ合算すれば，
多峰クラスを複数の線形境界の和で表現でき，`education`・`general`・`social_science` の recall が上がる．

**固定する構成（Iter92 の最良構成）**: 埋め込み `qwen3-embedding:4b` の 2 ビュー連結（5,120 次元），
instruction prefix 現行値，`CalibratedClassifierCV(method="temperature", cv=5, ensemble=True)`，
`C`=1.0，`class_weight=None` ＋ドメイン均衡 `sample_weight`，`confidence_threshold=0.0`，
`dispatch_top_k=1`，訓練 `data/classifier_train_iter92_civics_aligned.jsonl`（2,339 行），
評価 `data/dataset.jsonl`（3,750 行，sha256 `2114e048...`）．
**基準線は `results/20260928_111644/`（3,750 行，top1 = 0.835467）**．

### 着手可否の判定 (Iter93) —— B151 (d)4 の足切りに不合格．本走は行わない

**本番パイプライン（温度較正あり）での訓練集合内 CV．3 seed × 外側 5-fold ＝ 15 fold，n=2,339**

| 腕 | 学習ラベル | ドメイン写像 | CV ドメイン精度 | Δ vs 現行 |
|---|---|---|---|---|
| **A（現行）** | ドメイン 10 クラス | — | **0.7961**（fold 間 sd 0.0174） | — |
| B | JMMLU タスク 56 クラス | argmax タスク → ドメイン | 0.8045（sd 0.0133） | **+0.84pt** |
| **C** | JMMLU タスク 56 クラス | タスク確率をドメインへ合算 | **0.8060**（sd 0.0158） | **+1.00pt** |

**較正ラッパなしの素の `LogisticRegression`（同一分割，3 seed × 5 fold）での写像規則の比較**

| 腕 | 内容 | CV | Δ vs A |
|---|---|---|---|
| A | ドメイン 10 クラス（現行） | 0.7996 | — |
| B | タスク 56 クラス → argmax 写像 | 0.8109 | +1.13pt |
| **C** | タスク 56 クラス → 確率合算 | **0.8147** | **+1.51pt** |
| D | タスク 56 クラス（**タスク**均衡 `sample_weight`）→ 確率合算 | 0.8015 | +0.19pt |
| E | ドメイン確率 × タスク合算確率（CHiLS の積） | 0.8105 | +1.08pt |

**判定: 最良の腕 C でも本番パイプライン CV で +1.00pt であり，B151 (d)4 の足切り
（訓練集合内 CV で +2.3pt 以上）に届かない．よって本走に進まない**（Iter89 の学び
「判定不能と事前に分かるレバーには着手しない」，B153 (e) の着手条件，停止条件 (2)）．
Iter91 で実測した伝達率 43% を当てると end-to-end の予測は **+0.43pt** で，
再現性の床 ±0.25pt をかろうじて超える程度，事前登録し得る効果量閾値 1.0pt の半分未満である．
Iter92（Δ +0.508pt，p = 0.0509）と同じ「境界上で判定語が決まらない」結果を繰り返すことになる．

**さらに，非退行条件①を破る見込みが高い**（本走を行わない第 2 の理由）．
腕 C の per-domain recall（較正あり，15 fold 合算）は次のとおりで，**改善と退行が真っ二つに割れる**．

| ドメイン | タスク数 | A（現行） | C | Δ |
|---|---|---|---|---|
| education | 4 | 0.5411 | 0.6984 | **+15.7pt** |
| social_science | 4 | 0.7773 | 0.8587 | +8.1pt |
| general | 3 | 0.7960 | 0.8613 | +6.5pt |
| legal | 2 | 0.8398 | 0.8701 | +3.0pt |
| mathematics | 5 | 0.9520 | 0.9760 | +2.4pt |
| computer_science | 5 | 0.9200 | 0.9400 | +2.0pt |
| natural_science | 8 | 0.8720 | 0.8560 | −1.6pt |
| business_economics | 8 | 0.8173 | 0.7813 | −3.6pt |
| history_culture | 7 | 0.8494 | 0.7865 | −6.3pt |
| **medical** | **10** | 0.6653 | 0.4893 | **−17.6pt** |

**Δ はドメインあたりの JMMLU タスク数と明確に逆相関している**（タスク数 2〜5 の 6 ドメインは全て改善，
7〜10 の 4 ドメインは全て退行）．機序は，タスク数の多いドメインでは 1 サブクラスあたりの行数が
15〜38 行まで削られ，各サブクラスの境界推定の分散が増えるためと読める
（`medical` は 10 タスク・最小 `professional_medicine` 15 行）．確率を合算しても，
弱い境界を 10 本足したものは 10 倍のデータで引いた 1 本の境界に負ける．
これは Pirovano et al. (2025) の「細粒度学習の利得はサブクラスあたりのデータ量と
boundary redundancy に依存し，普遍的ではない」という主張の**本データでの実例**である．
すなわち **Iter92 の学び 3 の仮説（`education` は多峰クラス）は CV 上は支持された**（education +15.7pt）
一方で，**それを 10 ドメイン一律の規則として適用すると medical で失う分が上回る**．

**代替の粒度も測ったが，やはり足切りに届かない．**
タスク数の偏りを消すため「全ドメインを一律 k 個の潜在サブクラスへ分ける」（訓練 fold 内の埋め込みに
対する k-means．fold 内で fit するのでリークしない）という 10 ドメイン均一な規則を素の
`LogisticRegression`・3 seed × 5 fold で掃引した（基準 A = 0.8019，層化キーはドメイン）．

| k | 2 | 3 | **4** | 5 | 6 | 8 |
|---|---|---|---|---|---|---|
| Δ vs A | +0.61pt | +1.03pt | **+1.68pt** | +1.15pt | +1.17pt | +0.38pt |

k=4 が頂点の上に凸な曲線で，**退行するドメインが無い**点は腕 C より健全である
（medical 0.6813→0.6973，history_culture 0.8494→0.8421，legal 0.8528→0.8398）．
ただし素の LR で +1.68pt であり，較正ラッパを通すと本レバーでは利得が約 2/3 に縮む実測
（腕 C: +1.51pt → +1.00pt）を当てると **+1.1pt 前後**と見込まれ，これも足切りに届かない．

**本イテレーションの結論**: `classifier_label_granularity` は，値 `..._sum_to_domain`
（当初案の `..._argmax_map_to_domain` を含む）・k-means 潜在サブクラス版のいずれも
事前 CV が足切り +2.3pt に届かないため **本走を行わず，レバーを閉じる**．
`config.yml` の当該レバーには本 CV の実測値を注記した（同じ案を再度引く無駄を避けるため）．

**次の一手（rc-planner から人間／オーケストレータへの申し送り．backlog B154）**

分類器側の手（埋め込みモデル・instruction prefix・ビュー連結・較正手法・正則化 `C`・クラス重み・
訓練ラベル写像・ラベル粒度）は Iter79 以降で一巡し，**訓練 2,339 行・特徴 5,120 次元という
現在の構成での訓練集合内 CV はどの腕でも 0.80〜0.82 に張り付いている**．
残る方向は次の 3 つで，いずれも単一レバー原則の外側か人間判断を要する．

- **(A1, 推奨) 測定系の是正**（B153 (f) が「優先度は (e) の次」とした項目）: 訓練と評価で設問文が
  一致する残り 64 行を訓練から除去し，基準線を引き直す．精度レバーではないが，
  絶対水準を最大 1.7pt 上振れさせうるバイアスを，論文化前に取り除ける．オフラインで準備でき，
  新規データ源も不要．ただし基準線の本走 1 回（約 155 分）が要る．
- **(A2) 外部の日本語データ源の調達**（B151 (d)1）: CV で最も弱いのは `education` 0.54 と
  `medical` 0.67 で，hard negative プールの在庫は `education` 9 行・`legal`/`general`/`social_science` 0 行
  （B151 (c)）．行数を増やす以外の手は本反復で尽きた．**ライセンス（NC/ND 条項）と評価集合との
  重複の確認を伴うため，調達前に人間の確認を要する．**
- **(A3) 足切り +2.3pt そのものの見直し**: この値は「伝達率 43% × end-to-end 効果量閾値 1.0pt」から
  導いたものだが，1.0pt という閾値自体が Iter92 の学び 4 で「機序が予測どおり発火しても際どい」と
  判明している．**+0.5pt 級の改善を積み上げる方針へ切り替えるなら**，腕 C（予測 +0.43pt）ではなく
  k-means k=4（予測 +1.1pt・退行ドメインなし）が最有力の候補として残る．
  **これは研究の合否基準の変更であり，人間の判断を仰ぎたい．**

## Iteration 92: 訓練ラベル写像を評価集合へ一致させる（japanese_civics の education 復帰）

### 調査 (Iter92)

backlog B151 (c) の指示により `levers` を使い切った状態からの再探索である．B151 (d) が挙げた 4 観点
（日本語 legal/education/social_science/general の公開データ源／クラス不均衡下の線形分類器／複合クエリの
ルーティング／訓練集合内 CV で +2.3pt 以上を見込めること）を出発点に tavily-search で文献を当たりつつ，
**基準線 `results/20260928_032909/`（3,750 行，top1 = 0.829867）の誤りがどこに集中しているかを
`jmmlu_task` 単位まで分解した**．その結果，候補レバーを机上で比較する前に**測定系の側に
未発見の不整合が 1 つ残っていること**が分かったので，本反復はそれを主題にする．

**(1) 文献調査（tavily-search）で分かったこと**

- Menon et al. (2021), *Long-tail learning via logit adjustment*, ICLR 2021, arXiv:2007.07314,
  <https://arxiv.org/abs/2007.07314> —— クラス事前確率に基づく logit 補正は，重み付けよりも
  長尾クラスの誤りを直接減らす．**本研究への適用可能性の見積り**: 本研究で事前分布が偏っているのは
  `legal`（77 行 / 他 9 ドメイン各 250 行）だけであり，τ=1 の補正が動かすのは訓練行の 3.3%，
  訓練集合内 CV 換算で高々 +0.2pt 程度にしかならない．B151 (d) 4 の足切り（CV +2.3pt）に
  遠く届かないため，**今回は採らない**（Iter89 の学び「判定不能と事前に分かるレバーには着手しない」）．
- Northcutt et al. (2021), *Pervasive Label Errors in Test Sets Destabilize Machine Learning
  Benchmarks*, NeurIPS Datasets & Benchmarks,
  <https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/f2217062e9a397a1dca429e7d70bc6ca-Paper-round1.pdf>
  —— 実務では「補正前のテスト精度」しか見えないため，ラベル定義の誤りはモデル側の差として
  誤読されやすい．**本反復の発見はまさにこの型である**（後述）．
- Nevin et al. (2025), *The effects of mismatched train and test data cleaning*,
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC12190241> —— 訓練側と評価側で前処理・整形が食い違うと，
  モデル選択の判断そのものが変わりうる．**断定は避けるが**，本研究で 3 反復続けて分類器側のレバーが
  no_effect だったことと整合的な説明を与える可能性がある．
- Tunstall et al. (2022) SetFit 系の実務報告（Cloudera FFL, *Few-Shot Text Classification*,
  <https://few-shot-text-classification.fastforwardlabs.com>）—— 凍結埋め込み上の軽量分類器は
  **クラスあたり 8〜30 例**で実用水準に達し，100 例以上は飽和しやすい．
  **本反復への含意**: 後述の欠落サブクラスに対し 34 行しか用意できないが，この範囲は
  「効果が出るなら出る」水準ではある（保証ではない）．
- JMMLU のライセンス: 『日本問題』（japanese_civics・japanese_idiom・japanese_geography）の著作権は
  New Style（VIST 学習塾）にあり **CC BY-NC-ND 4.0**，japanese_history / world_history は STEP 社で
  研究・評価目的以外の商用利用禁止（<https://huggingface.co/datasets/nlp-waseda/JMMLU>，
  <https://github.com/nlp-waseda/JMMLU>）．`build_dataset.py:_RESTRICTED_LICENSE_TASKS` はこの 5 タスクを
  列挙しており，`mise.toml:31` の `build_dataset.py --output data/dataset.jsonl` は
  `--exclude-restricted-license-tasks` を**渡していない**（＝評価集合には japanese_civics が入る）．
  **B151 (d) 1 が求めた「JMMLU 以外の日本語データ源の調達」は今回は不要と判断した**．理由は
  次節のとおり，既存データの中に未使用の正解信号が残っていることが分かったためである．

**(2) 基準線の誤りの所在 —— `education` の不足分はほぼ全量が 1 タスクに集中している**

`data/dataset.jsonl` の `jmmlu_task` で基準線 `results/20260928_032909/` の単一ドメイン行を割ると:

| ドメイン / タスク | n | recall | 主な誤送先 |
|---|---|---|---|
| education / **japanese_civics** | 116 | **0.0086 (1/116)** | history_culture 59・business_economics 36・legal 15 |
| education / high_school_psychology | 73 | 0.685 | medical 14 |
| education / sociology | 85 | 0.659 | history_culture 15 |
| education / moral_disputes | 76 | 0.645 | social_science 12・legal 11 |
| general / miscellaneous | 69 | 0.391 | history_culture 14・natural_science 10 |
| general / formal_logic ・ logical_fallacies | 106 | 0.98 / 0.964 | — |
| social_science / philosophy | 77 | 0.610 | education 21 |

- **education_recall 0.4457 の不足 194 行のうち 115 行（全 3,750 行の 3.07pt）が japanese_civics 1 タスク**
  である．他 3 タスクは 0.645〜0.685 で，10 ドメイン中の中位と同程度でしかない．
- `general` も同型で，0.7429 の不足はほぼ `miscellaneous`（0.391）に集中し，形式論理系 2 タスクは 0.97 前後．

**(3) 原因は分類器ではなく訓練側のタスク→ドメイン写像のずれ（本フェーズで実測）**

`/tmp/expert-mesh-cache/JMMLU.zip` の設問文をキーに，本番訓練ファイル
`data/classifier_train_iter87_hybrid.jsonl`（2,327 行）の各行を JMMLU のタスクへ逆引きした結果:

- **`education` ラベル 250 行の内訳は high_school_psychology 87 / moral_disputes 87 / sociology 76 で，
  japanese_civics は 0 行**．
- 一方 **japanese_civics 22 行が `history_culture` ラベルで訓練集合に入っている**．
  これは誤送先第 1 位（59 行）と一致しており，**現行の訓練データは「公民の設問は history_culture」と
  明示的に教えている**．
- 訓練集合 2,327 行のうち現行 `_DOMAIN_TASK_MAP` と食い違う行は **24 行のみ**で，そのうち 22 行が
  この japanese_civics である（残り 2 行は international_law → business_economics 1 行，
  high_school_microeconomics → mathematics 1 行）．つまり**ずれは 1 タスクに限局している**．
- 経緯: `data/classifier_train.jsonl`（1,427 行，sha256 `eb89bf7b...`）は Iter37/38 の写像更新より前に
  作られたまま再生成されておらず，Iter84 の hard negative 採掘は「現行訓練集合に既出のタスクへプールを
  限定する」方針（backlog B133 (2)）を採ったため，旧写像がそのまま温存された．
  一方 `data/dataset.jsonl` は Iter85（単一ドメイン拡充）・Iter90（複合拡充）で現行
  `_DOMAIN_TASK_MAP` に基づき再構築されており，**評価側だけが新写像に移っていた**．

**Iter36/37/38 の否定的所見との関係（重要．前提が逆である）**: Iter38
（`education_hybrid_proxy_and_civics`）は rejected だが，当時の実装検証項目に
「eval: education 150（旧 proxy のみ），japanese_civics = 0 件 — PASS」と明記されているとおり，
**評価集合から japanese_civics を除いた上で訓練へ 150 行入れた**構成だった（訓練だけに存在する希釈）．
現在は符号が逆で，**評価側に 116 行あり訓練側が 0 行**である．
backlog B133 (2) が civics の流入を避けた判断も，評価側の再構築より前の情報に基づく．
したがって過去 3 回の失敗は本反復の変更を否定しない（断定を避けて言えば，
「同じ変更が別の前提の下で測られたもの」であって再現実験ではない）．

**(4) リーク上限の実測 —— 訓練へ回せる japanese_civics は 34 行が上限**

JMMLU の japanese_civics は全 150 問．内訳は **評価集合に 116 問 / 訓練集合に 22 問（うち 8 問は
評価集合とも重複）/ どちらにも未使用 20 問**．よって**評価集合を汚さずに訓練へ回せるのは
14（非重複の既存訓練行）＋ 20（未使用）＝ 34 行が上限**である．
評価集合に載っている 116 問を訓練へ入れることは絶対にしない．

**(5) 副次的に見つかった測定系の欠陥（本反復では直さない．backlog B152 へ記録）**

訓練 2,327 行と評価 3,750 行の**設問文完全一致が 72 行**ある（うち 64 行はラベルも一致）．
`build_classifier_training_rows()` は `exclude_queries=eval_queries` で重複を排除する設計だが，
本番の訓練ファイルは評価集合の拡充（Iter85/90）より前に作られたため事後的に重複が生じた．
**本反復では (4) の civics 8 行だけを削除し，残り 64 行には触れない**（単一レバー原則）．

### 計画 (Iter92)

**単一レバー**: `classifier_train_label_map_consistency` = **`japanese_civics_realignment_to_eval_map`**
（config.yml の `levers` 末尾に本フェーズで追記．backlog B152 に選定理由を記録）．

**仮説**: `education` の recall が低いのは代理タスクの意味的ギャップでも分類器の容量不足でもなく，
**評価集合の education の 33%（116/350）を占める japanese_civics を，訓練データが
`history_culture` として明示的に教えているため**である．訓練側の写像を評価側と一致させれば，
japanese_civics の recall が 0.0086 から大きく上がり，top1_accuracy が +1.0pt 以上動く．

**変更点（データのみ．コードは 1 行も変えない）**

`data/classifier_train_iter92_civics_aligned.jsonl` を新規に作り，
`data/classifier_train_iter87_hybrid.jsonl` との差分を次の 3 点だけにする．

1. 評価集合に出現しない japanese_civics 訓練行 **14 行**: `domain` を `history_culture` → `education`．
2. 評価集合と設問文が重複する japanese_civics 訓練行 **8 行**: 削除．
3. 訓練にも評価にも未使用の japanese_civics **20 問**: `education` 行として追加．

結果は **education 250→284 行 / history_culture 250→228 行 / 合計 2,327→2,339 行**．
`scripts/train_domain_classifier.py`・`build_dataset.py`・`config.yaml`・`classifier.py`・
`data/dataset.jsonl` は無変更（`C` は Iter91 で revert 済みの既定 1.0 のまま）．

**固定する構成（Iter90 の最良構成）**: 埋め込み `qwen3-embedding:4b` の 2 ビュー連結（5,120 次元），
instruction prefix は現行値のまま，`CalibratedClassifierCV(method="temperature", cv=5, ensemble=True)`，
`class_weight=None` ＋ `_extract_sample_weights()` のドメイン均衡重み，`confidence_threshold=0.0`，
`dispatch_top_k=1`，評価集合 `data/dataset.jsonl` 3,750 行．

**再訓練の手順（絶対条件）**

- 埋め込み計算は **wafl-ctrl5 上で行う**（2026-09-23 恒久運用ルール．wafl500〜509 は本走以外に使わない）．
- **既存キャッシュを再利用し，新規に埋め込むのは追加 20 行だけにする**
  （`data/embcache_train_iter89_qwen3-embedding_4b{,__p1}.npy` の該当行を流用）．
  B149 (d) の `qwen3-embedding:4b` のセッション跨ぎ非決定性（最大絶対差 0.011406）と本レバーの効果が
  交絡するのを避けるための措置であり，**2,319 行の特徴量が基準線 artifact と同一であることを
  G0-b で確認する**．同等の結果が得られる別経路を採ってもよいが，その場合も同一性の確認は必須とする．

**本走前のゲート（G0．すべて本走前に journal へ記録する）**

| # | 内容 | 合格条件 |
|---|---|---|
| G0-a | 新訓練ファイルの構成検証 | 合計 2,339 行 / education 284 / history_culture 228 / japanese_civics 行は全て `education` かつ 34 行 / `history_culture` の japanese_civics 0 行 / 新・改変行と `data/dataset.jsonl` の設問文重複 0 件 |
| G0-b | 特徴量の同一性 | 変更していない 2,319 行の埋め込みがキャッシュとビット一致 |
| G0-c | 変更の最小性 | `git diff --stat` に `scripts/`・`build_dataset.py`・`config.yaml`・`classifier.py`・`node.py`・`aggregator.py` が現れないこと．`data/dataset.jsonl` の sha256 不変 |
| G0-d | オフライン replay による効果量の事前登録 | 新 artifact で `data/dataset.jsonl` を replay し，top1・per-domain・japanese_civics 部分集合 recall を**本走前に**記録する（**予測の事前登録のみ．採否の判断には使わない**．Iter91 で replay が分類器 argmax 精度を小数 6 桁一致で当てることが実証済み） |
| G0-e | **デプロイ** | **wafl500〜509 には Iter91 の `C`=10.0 artifact が載ったままである（B151 (b)）．本走前に必ず `mise run deploy` を実行し，10 台すべてで新 artifact の sha256 一致と smoke_check PASS を確認すること** |

**成功条件（事前登録．基準線は `results/20260928_032909/`，3,750 行，top1 = 0.829867）**

- **主基準（両方を満たすとき `adopted`）**
  - (i) Δtop1_accuracy **≥ +1.0pt**（≥ 0.839867）かつ McNemar 両側 p < 0.05（Wilson 95%CI 併記）．
  - (ii) education 単一 350 行の recall **≥ 0.55**（基準線 0.4457）**かつ** japanese_civics 部分集合
    116 行の recall **≥ 0.30**（基準線 0.0086）．
- **非退行（1 つでも破れば `partial`）**
  - ① per-domain recall/precision 20 指標（単一 3,020 行）の BH 補正 q=0.05 後の有意退行 0 件．
    **history_culture は訓練行が 22 行減るため最も危ない**（基準線 recall 0.9114）．
  - ② 複合 730 行の正解率 **≥ 0.7932**（基準線 0.8082 − 1.5pt）かつ複合行での McNemar 有意退行なし．
  - ③ `fallback_rate` = 0.0，`dispatch_failure_rate` ≤ 0.005，`mean_duration_ms` ≤ 3,040（基準線 2,531 の 1.2 倍）．
- **判定語**: (i)(ii) と非退行①〜③がすべて成立 → `adopted`．
  (i)(ii) のいずれかが成立し非退行が破れる → `partial`．
  (i) が非有意または |Δtop1| < 0.5pt → `no_effect`．到達確認 G0-a〜G0-c が破れる → `invalid`．
  （B149 (c)・B151 要レビュー (a) の「判定語を排他的な決定木へ書き直す」という宿題は未承認のため，
  本反復では上記の順に**上から順に最初に当てはまったものを採る**と明示して曖昧さを消す．）

**期待効果（事前登録の点予測）**: japanese_civics recall 0.0086 → **0.45**（116 行中 52 行．
SetFit 系の実務報告が示す「クラスあたり 30 例前後で実用水準」の下端にあたる 34 行しか用意できないため，
飽和水準には届かないと見込む）．これが overall へ **+1.36pt**，history_culture の減少分 −0.1pt 前後，
差引 **Δtop1 = +1.0〜+1.5pt** と予測する．B151 (d) 4 の「訓練集合内 CV +2.3pt」という足切りは，
**訓練集合に japanese_civics の education 行が 1 行も無い以上，訓練集合内 CV ではこの欠陥が
そもそも観測できない**ため本レバーには適用できない（伝達率 43% は分類器の容量を触るレバーの値であり，
訓練と評価の分布ずれを直すレバーには当てはまらない）．代わりに G0-d の replay を事前登録の予測として使う．

**測定上の注記（分析フェーズへの申し送り）**: 変更点 2 で削除する 8 行は評価集合にも含まれるため，
削除によって汎化と無関係に最大 **0.21pt**（8/3,750）だけ top1 が上振れしうる．
分析フェーズでは**この 8 行を除いた 3,742 行での top1 も必ず併記すること**．

**不成立だった場合の次の一手**: japanese_civics recall が 0.30 に届かない場合，34 行という行数が
不足しているのか（→ 外部の日本語公民データ源の調達．ライセンス確認が要るため人間判断）
それとも `education` というクラス定義自体が 4 タスクをまたいで多峰的すぎるのか（→ クラス定義の再検討）
を切り分けること．

### 実装・実験 (Iter92)

**変更（コード 0 行）**: `data/classifier_train_iter92_civics_aligned.jsonl`（2,339 行，sha256
`39c4ca52...`）を新規作成した．差分は事前登録どおり 3 点のみ — (1) `history_culture` → `education`
への付け替え 14 行（`history_culture-train-002,006,009,024,025,051,052,053,056,063,085,103,110,148`），
(2) 評価集合と設問文が重複する 8 行の削除（同 `-013,014,020,078,099,118,120,135`），
(3) 訓練・評価とも未使用の japanese_civics 20 問を `education` として追加（`education-civics-001`〜`-020`）．
education 250→284 / history_culture 250→228．追加 20 行の設問整形は `build_dataset.py` の
`_parse_jmmlu_task_csv()` / `_format_jmmlu_query()` をそのまま import して生成した（自前整形なし）．
再訓練は wafl-ctrl5 の Ollama（127.0.0.1:11499）で実施し，変更のない 2,319 行は
`data/embcache_train_iter89_qwen3-embedding_4b{,__p1}.npy` を直接インデックス参照で再利用，
新規 embed は追加 20 行のみ（`embed_query_views` と同じ plain→instructed の順）．
新 artifact `models/domain_classifier.joblib` sha256 `98da6f2d0d44...`（n_features_in_=5120），
旧版（基準線と同一 `ff8aad9c...`）は `models/domain_classifier_pre_iter92_civics.joblib` へ退避（可逆）．
`data/MANIFEST.md` 追記済み．再訓練スクリプトは `/tmp/iter92/retrain.py`（リポジトリ外）．

**G0 ゲート**: a（構成検証: 2,339 行／education 284／history_culture 228／japanese_civics 行は全て
education かつ 34 行／history_culture 内の japanese_civics 0 行／評価集合との設問文重複 0 件）**PASS**，
b（特徴量同一性: 2,319 行はキャッシュから直接インデックス参照．スクリプト内 assert で検証）**PASS**，
c（変更の最小性: `git diff --stat` に `scripts/`・`build_dataset.py`・`config.yaml`・`classifier.py`・
`node.py`・`aggregator.py` は一切現れず，`data/dataset.jsonl` sha256 不変 `2114e048...`，
`train_domain_classifier.py` の `C` は既定 1.0 のまま差分 0 行）**PASS**，
d（replay による予測の事前登録: top1 0.831467→0.836267，Δ+0.48pt，McNemar p=0.0636，
japanese_civics recall 0.0086→0.2241，education recall 0.4429→0.4943．**採否判断には使わない**）**記録済**，
e（デプロイ: `mise run deploy` 後，全 10 ノード wafl500〜509 で artifact sha256 が `98da6f2d0d44...` に
一致，smoke_check も全台 PASS）**PASS**．

**本走**: `results/20260928_111644/`（3,750 問，フルスペック 1 回，所要約 155 分 < timeout 180 分）．
基準線は `results/20260928_032909/`（top1 = 0.829867）．

| 指標 | 基準線 | 本反復 | Δ |
|---|---|---|---|
| top1（全 3,750 行） | 0.829867 | 0.835467 | **+0.560pt**（McNemar p = 0.032015，discordant 33/54） |
| top1（3,742 行．重複削除 8 行を除外） | 0.831641 | 0.836718 | +0.508pt（McNemar p = 0.050894，discordant 33/52） |
| education 単一 350 行 recall | 0.445714 | 0.497143 | +5.143pt |
| japanese_civics 部分集合 116 行 recall | 0.008621 | 0.232759 | +22.414pt（1/116 → 27/116） |
| 複合 730 行 top1 | 0.808219 | 0.805479 | −0.274pt（McNemar p = 0.789268） |

per-domain 20 指標の BH 補正（q=0.05）後の**有意退行は 0 件**（生の p 値では `recall:education` の
p = 0.005820 が最小だが，BH のランク 1 閾値 0.0025 に対し不足．他は p > 0.2）．
運用指標は `fallback_rate` 0.0/0.0，`dispatch_failure_rate` 0.000533→0.000800（≤0.005），
`mean_duration_ms` 2531.5→2535.3（≤3040）で全て基準内．

**逸脱・注記**: (1) `mise run deploy` はツールタイムアウト 2 分を超えるためバックグラウンド実行へ
切り替えて完走を待った（コマンド自体は正常終了．Iter89 と同種の運用上の事象）．
(2) G0-b の「ビット一致」はキャッシュからの直接インデックス参照という構成上，独立に再計算して
差分を測る形の検証ではなく，同じ配列を使い回していることによる構造的な保証である．
埋め込みのセッション跨ぎ非決定性（B149 (d)）とは交絡しない設計だが，この違いは明記しておく．
(3) 本走は途中経過のポーリングのみで完走を待ち，強制終了等は行っていない．

### Iteration 92 実行済み

**変更**: コード 0 行．`data/classifier_train_iter92_civics_aligned.jsonl`（2,339 行，sha256
`39c4ca52...`）を新規作成し，訓練側の japanese_civics のラベルを評価側の `_DOMAIN_TASK_MAP` へ
一致させた（`history_culture`→`education` 付け替え 14 行／評価集合と設問文が重複する 8 行を削除／
訓練・評価とも未使用の 20 問を `education` として追加．education 250→284，history_culture 250→228）．
新 artifact `models/domain_classifier.joblib` sha256 `98da6f2d0d44...`．G0-a〜e は全て PASS
（評価集合 sha256 不変，wafl500〜509 の 10 台で artifact sha256 一致，smoke_check PASS）で，
**レバーが発火していることは確認済みである**．

**結果（本走 `results/20260928_111644/` 対 基準線 `results/20260928_032909/`）**

| 指標 | 基準線 | 本反復 | Δ | p |
|---|---|---|---|---|
| top1（3,750 行） | 0.829867 (3112/3750) | 0.835467 (3133/3750) | +0.560pt | McNemar 0.031418（54 改善 / 33 悪化） |
| **top1（3,742 行．訓練削除 8 行を除外．主報告値）** | 0.831641 | 0.836718 | **+0.508pt** | **0.050894** |
| education 単一 350 行 recall | 0.445714 | 0.497143 | +5.143pt | — |
| japanese_civics 116 行 recall | 0.008621 (1/116) | 0.232759 (27/116) | **+22.414pt** | **exact McNemar 2.98e-8（26 改善 / 0 悪化）** |
| **japanese_civics を除く 3,634 行** | — | — | **−5 行（28 改善 / 33 悪化）** | **0.608921** |
| 複合 730 行 top1 | 0.808219 | 0.805479 | −0.274pt | 0.789268 |

per-domain 20 指標の BH 補正（q=0.05）後の有意退行 **0 件**（生 p 最小は `recall:education` の
0.005820 だが BH ランク 1 閾値 0.0025 に不足）．運用指標は全て基準内（`fallback_rate` 0.0，
`dispatch_failure_rate` 0.000800 ≤ 0.005，`mean_duration_ms` 2535.3 ≤ 3040）．

**ノイズか信号か**: 本測定系の再現性の床は top1 ±0.25pt，そこから導かれる McNemar 有意境界は
0.249pt である（B149 恒久申し送り 1，Iter88 実測）．Δ = +0.560pt（3,742 行で +0.508pt）は床の
約 2.0〜2.2 倍であり，**符号は信号として読める**．ただし +1.0pt に対しては約半分で，
主基準 (i) が要求した効果量には届いていない．一方 **japanese_civics 部分集合の 26 改善 / 0 悪化
（exact p = 2.98e-8）はノイズでは説明不能**であり，狙った機序が発火したことは確定した．

**主報告値の選択（論点 3）**: 削除した 8 行は評価集合にも含まれるため事前登録どおり両方を出したうえで，
**3,742 行版（Δ +0.508pt，p = 0.050894）を主報告値とする**．理由は，削除された 8 行の正誤変化は
汎化とは無関係な構成上の産物（基準線ではこの 8 行が `history_culture` ラベルで訓練に入っており
確実な誤りを作っていた）だからである．実際に生じた上振れは **+2 行 = +0.053pt** に留まり，
事前に見積もった最大 0.21pt よりはるかに小さい（基準線 0/8 正解，本反復 2/8 正解）．
2 つの数字の差は 0.05pt で，どちらを採っても効果量の結論は変わらないが，p 値は 0.031 と 0.051 で
有意水準 0.05 をまたぐ．**保守側（3,742 行，p = 0.050894 ＝ 非有意）を主として扱う．**

**判定: `partial`（主たる判定語）**

事前登録の判定条文を literal に当てはめた結果を先に書く．
- `adopted`: (i) Δtop1 ≥ +1.0pt が不成立（+0.508〜+0.560pt）．(ii) education recall 0.497 < 0.55，
  japanese_civics recall 0.2328 < 0.30 も不成立．**不成立**．
- `partial`（「(i)(ii) のいずれかが成立し非退行が破れる」）: (i)(ii) はいずれも不成立，
  非退行①〜③は全て充足．**literal には不成立**．
- `no_effect`（「(i) が非有意 **または** |Δtop1| < 0.5pt」）: 主報告値 3,742 行では p = 0.050894 で
  非有意のため **literal には成立する**（3,750 行を採れば p = 0.031418 で有意，|Δ| = 0.560 ≥ 0.5 となり
  literal には不成立となる．**どちらの行数を採るかで判定語が変わる**）．
- `invalid`: G0-a〜e 全 PASS のため不成立．

すなわち**事前登録の判定木には，「有意性は境界上だが効果量が閾値の半分・非退行は全充足・
狙った機序は明確に発火」という今回の状態を受け止める枝が無い**（B151 要レビュー (a) が指摘した
条文の重なりとは逆向きの，網羅性の欠落である）．そのうえで主たる判定語を **`partial`** とする．
根拠は次の 3 点である．
1. `no_effect` は「レバーが効かなかった」ことを記録する語だが，japanese_civics 26 改善 / 0 悪化
   （p = 2.98e-8）は**レバーが意図した経路で確実に効いたことを直接示す**．この状態を `no_effect` と
   記録すると，次の自分が「訓練／評価のラベル写像ずれは直しても効かない」という誤った事実を
   引き継ぐ．Iter91 の `no_effect`（Δ +0.16pt・床の内側・機序の証拠なし）とは質的に異なる．
2. `partial` の語義は「事前登録した成功条件の一部だけを満たした」であり，今回は
   主基準 (ii) の 2 条件のうち方向・機序は実現したが水準（0.30 / 0.55）に未達，
   (i) は効果量未達という状態で，この語義には合致する．条文の文言（「非退行が破れる」）に
   合致しないのは条文側の欠落である．
3. Iter90 も「設計どおりに動いたが数値基準は未達」で `partial` と記録しており，用語の一貫性を保てる．

**artifact の採否: 維持（ロールバックしない．論点 2）**

`models/domain_classifier.joblib` は新版 `98da6f2d0d44...` のまま，wafl500〜509 も新版配布済みのまま
据え置く（旧版 `ff8aad9c...` は `models/domain_classifier_pre_iter92_civics.joblib` に保持，可逆）．
Iter91 で倹約側の既定（効果が測れず退行がある変更は入れない）を採ったのと条件が異なる:
Iter91 は Δ +0.16pt（床 ±0.25pt の**内側**＝効果が測れない）かつ `legal_recall` に有意退行 1 件だった．
今回は Δ が床の 2 倍以上・退行 0 件・運用指標も基準内である．加えて本変更は精度チューニングではなく，
**訓練ラベルが評価側のラベル定義と矛盾していたという欠陥の是正**であり，仮に Δ が 0 でも
矛盾した訓練データへ戻す積極的な理由が無い．**次イテレーションの基準線は
`results/20260928_111644/`（3,750 行，top1 = 0.835467）へ更新する**（評価集合 `data/dataset.jsonl` は
sha256 不変 `2114e048...` なので行数・母集団は基準線間で一致する）．
再訓練を行う場合の訓練ファイルは `data/classifier_train_iter92_civics_aligned.jsonl` である．

**学び**

1. **`education` の低 recall の主因は分類器側ではなく，訓練ラベルが評価ラベル定義と矛盾していたこと
   だった．**Iter89〜91 の 3 反復にわたって分類器側のレバー（埋め込み・正則化）が no_effect だった
   背景にはこれがあった可能性が高い（断定はしない．Nevin et al. 2025 の「訓練／評価の整形不一致は
   モデル選択の判断自体を変えうる」と整合的な実例である）．**モデル側を触る前に，訓練と評価の
   ラベル定義が同一の写像から生成されているかを毎回確認すること．**
2. **効果は狙った 116 行の中にほぼ完全に閉じており，波及も副作用も無かった．**civics を除く
   3,634 行では 28 改善 / 33 悪化（p = 0.61）で**正味 −5 行**である．すなわち
   「+0.56pt は civics 由来 +26 行と，それ以外での −5 行の差引」であり，
   **このレバーで overall を動かせる上限は評価集合中の該当タスクの行数で決まる**．
   訓練データの局所的な修正が全体へ波及するという期待は，今回のデータでは支持されない．
3. **34 行では civics recall は 0.233 にしか届かなかった（事前予測 0.45 の半分）．**
   計画が事前登録した切り分け（行数不足か，`education` クラス定義の多峰性か）に対しては，
   **行数不足だけでは説明しきれない**と読む．他 3 代理タスク（psychology 0.685・sociology 0.659・
   moral_disputes 0.645）はそれぞれ 76〜87 行で 0.65 前後に留まっており，行数を増やしても
   0.65 付近が `education` クラスの天井である可能性がある．**`education` は 4 タスクにまたがる
   多峰クラスであり，単一の線形境界で表現しきれていない**という仮説が次の検討対象になる
   （→ Iter93 の新レバー `classifier_label_granularity` へ．backlog B153）．
4. **事前登録の効果量閾値は，機序の指標（部分集合 recall）と overall 指標の両方に置いたうえで，
   「部分集合の寄与から overall の上限を先に計算しておく」べきだった．**今回 civics 116 行が
   全問正解になっても overall は +3.07pt が上限であり，実際に得られた recall 0.233 では
   構造的に +0.69pt が上限だった．すなわち **+1.0pt という主基準は，機序が事前予測どおり
   （recall 0.45）に発火しても達成が際どい水準に設定されていた**．次回以降，
   「部分集合の期待改善 × 部分集合の母集団比」を計画時に必ず計算して閾値の実現可能性を検算すること．
5. 判定木の網羅性: `no_effect` の条文（非有意 **または** |Δ| < 0.5pt）は，
   「有意だが効果量が閾値未達」を `no_effect` に落としてしまう幅がある．
   B151 (a) の宿題（判定語を排他的な決定木へ書き直す）は未承認のままだが，
   **今回はその欠落が実際に判定語の選択を左右した**．次の計画フェーズは，
   主基準の効果量条件と有意性条件を分けた 2 軸の表として判定語を定義すること（B153 要レビュー）．

