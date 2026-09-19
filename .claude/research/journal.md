## Iteration 63: rank_1 の選択元を多ラベルヘッドの argmax へ切り替える

### 調査 (Iter63)

**問い**（config.yml が既に詳細な設計を事前登録している＝backlog B96・config.yml:971-1014．
先行研究の新規調査ではなく，一次情報＝コード・実データの確認を優先した）

- Q1: `scripts/evaluate_dispatch_candidate_ranking.py` の `build_new_rows()` は rank_1 をどこで
  基準線からコピーしているか．argmax へ切り替える変更は本当にこの 1 関数のみで完結するか
  （rank_2 選択ロジック `rank2_new = max(...)` との相互作用，rank_1==rank_2 の重複が起きないかを含む）．
- Q2: Iter61 の資産（ヘッド・埋め込みキャッシュ・訓練データ・予測 JSONL）は実在し，再利用可能か．
- Q3: config.yml が事前登録した S1〜S5・N2'/N3/N5/N6 は既存の `compute_iter59_ranking_stats.py`
  でそのまま計算できるか，追加のアドホック集計が要るか．
- Q4: reflector が挙げた実行可能性の参考値（複合100行の rank_1 正解 41→61，
  `compound_domain_set_recall` 0.445→0.49，単一ドメイン argmax 0.610→0.6033）は，既存の予測 JSONL
  から再現できるか．

**分かったこと（全文読了・実行確認による一次情報）**

1. **`build_new_rows()`**（`scripts/evaluate_dispatch_candidate_ranking.py:193-226`，全文読了）:
   `rank_1 = row["dispatched_domains"][0]`（`:208`）が基準線 JSONL からのコピー箇所そのもの．
   `rank2_new = max((d for d in head_scores if d != rank_1), key=...)`（`:211-214`）は
   `rank_1` を除外して選ぶ構造のため，**`rank_1` を `argmax(head_scores)` に差し替えるだけで
   `rank_2` の除外集合も自動的に追随し，`rank_1==rank_2` の重複は構造的に起こり得ない**
   （`rank_2` は常に `rank_1` 以外の 9 ドメインから選ばれるため）．ここまでは reflector の
   「1 関数のみで完結する」という見立てどおりである．
2. **しかし build_new_rows() 単独の変更では `_run()` が壊れる**（重要な訂正）:
   `_assert_rank1_unchanged()`（A1，`:229-240`）が `_run()`（`:380`）で**無条件に**呼ばれており，
   `base["dispatched_domains"][0] != new["dispatched_domains"][0]` を検出すると
   `AssertionError` を送出する．新 rank_1 は基準線と 1600 行中 **380 行（23.75%）** で異なることを
   実測したため（下記 4），build_new_rows() だけを変更すると実行時に必ずこのアサーションで落ちる．
   このアサーションはまさに「rank_1 は再計算しない」という**旧設計の不変条件を保証するための
   意図的な安全装置**（モジュール docstring `:5-13` が同じ不変条件を明記）であり，
   本レバーの目的（rank_1 自体を変える）と正面から衝突する．**変更範囲は最低でも 2 箇所
   （`build_new_rows()` と `_run()` 内の A1 呼び出し／A1 自体の役割変更）になる**．
   config.yml の S4「rank_1 が基準線と異なる行 >0（発火の証拠）」は，A1 を「例外を送出する
   アサーション」から「不一致件数を報告する非致命的なチェック」へ作り替えることでそのまま
   計測できる（precedent: 同ファイルの `_compute_rank2_flip_rate` と同型の設計）．
   モジュール docstring の該当箇所（rank_1 不変性の説明）も実装フェーズで更新が要る
   （ドキュメントのみで実行には影響しないが，古い記述を放置しない）．
3. **`selected_domain` フィールドは本レバーでも再計算されない**（`build_new_rows():219` が
   `"selected_domain": row["selected_domain"]` と基準線から素通しコピーするのは変更後も同じ）．
   `metrics.compute_top1_accuracy()` は `r["selected_domain"] in r["expected_domains"]` を見る
   （`metrics.py:42`）ため，**`selected_domain` を書き換えない限り `compute_top1_accuracy()` が
   返す値は本レバーの効果に関係なく基準線と完全一致し続ける**．実際に確認したところ
   （元の `selected_domain` のまま計算）新旧とも **0.5975 で一致**した．
   一方，**`selected_domain` を新 rank_1 で上書きして** `compute_top1_accuracy()` を計算すると
   **0.60375（基準線比 +0.625pt，むしろ改善）** になった（下記 4 の実測．単一ドメイン
   1500 行 905/1500=0.6033，複合 100 行 61/100=0.61 の加重平均）．
   **config.yml の N2'（top1_accuracy が基準線比 1.0pt を超えて低下しない＝≧0.5875）は，
   `selected_domain` を書き換えない実装では常に自明に PASS する（0.5975=0.5975 で退行しようが
   ないため何も検証していないのと同じ）**．これは CLAUDE.md/config.yml が繰り返し警告している
   「レバーを読むコードに実行が到達しない」型の穴になりうるため，計画フェーズで
   **「N2' の top1_accuracy は `selected_domain` を新 rank_1 で上書きして計算する」と明示的に
   決め，実装しないと N2' が無意味な自明 PASS になる**ことを申し送る．なお実測では上書きした
   場合でも悪化ではなく改善方向なので，判断を誤っても致命的な結果にはならないが，
   「何を検証しているか」を曖昧にしたまま実装するのは避けるべきである．
4. **reflector の参考値を独立に再現した**（`results/iter61_multilabel_ranking_predictions.jsonl`
   の `head_scores` フィールドのみを使い，embedding 呼び出し・実機接続なしで再計算．
   `metrics.compute_compound_coverage_metrics` は無改造で呼んだ）:
   - 複合 100 行の rank_1 正解数: 基準線 **41/100** → 新 rank_1（`argmax(head_scores)`）**61/100**．
   - `compound_domain_set_recall`: 基準線 0.345 → 新 rank_1+新 rank_2 で **0.49**
     （reflector の note にある「0.445→0.49」の 0.445 は Iter61 の値であり，真の基準線 0.345 比では
     +14.5pt に相当）．
   - 単一ドメイン 1500 行の rank_1（argmax head_scores）正解率: **905/1500 = 0.603333**
     （journal Iter62 の N5 実測値 0.603333 と完全一致，同一ヘッドなので当然の整合）．
   - 全 1600 行での rank_1 変化件数: **380/1600（23.75%）**．
   - 以上はすべて既存 JSONL の `head_scores` フィールドのみで再計算可能であり，Ollama 埋め込みも
     実機接続も不要だった（低コストの事前確認として妥当）．
5. **Iter61 資産の実在を再確認した**（`ls -la`）: `data/classifier_train_multidomain_iter61.jsonl`
   （41,551B）・`models/dispatch_multilabel_head_iter61.joblib`（66,134B）・
   `results/iter59_query_embeddings.npz`（9,971,712B）・
   `results/iter61_multilabel_ranking_predictions.jsonl`（982,193B）・
   `results/20260918_202613/results.jsonl`（3,510,699B，基準線）・`data/classifier_train.jsonl`
   （600,281B）が全て実在し，mtime も直近イテレーションの記録と整合する．較正は使わない方針
   （config.yml 既定）なので Iter62 の較正済みヘッドは今回使わない．
6. **`scripts/compute_iter59_ranking_stats.py`**（全文読了）は S1（200 ペア McNemar）・S2（効果量）・
   S3（コスト中立）・S4（Iter59/61/62 では rank_2 flip の意味だが，本レバーでは `_compute_s4` の
   ロジック自体は基準線 `dispatched_domains[1]` 比較のままで rank_2 の変化を測るだけであり，
   本レバー固有の S4「rank_1 が基準線と異なる行 >0」とは別物であることに注意）・N1（`_compute_n1`，
   本レバーでは定義上 FAIL し続けることを config.yml が明記済み，スクリプトは例外を送出せず
   `pass:false` を返すだけなので実行は壊れない）・N3（legal 非退行）は**無改造でそのまま使える**．
   一方，**S5（複合 100 行の rank_1 正解数を対応あり McNemar で基準線 41/100 と比較）はこのスクリプトに
   存在しない新規の集計単位**（既存の `_compute_s1` はドメイン対単位＝200 ペア，S5 は行単位＝100 行の
   `dispatched_domains[0] in expected_domains` 判定）である．ただし `metrics._mcnemar_from_correctness()`
   は `{id: bool}` の辞書を 2 つ受け取る汎用関数（docstring に「per-domain callers が独自の
   correctness 定義を渡せる」と明記）なので，**行 ID をキーにした 100 件の bool 辞書を渡すだけで
   既存の検定ロジックをそのまま再利用でき，新しい統計機構の実装は不要**（Iter59〜62 が繰り返してきた
   「既存関数を薄く呼ぶアドホック集計」と同じパターンで済む）．N6（education/medical 自己被覆が
   Iter62 の 6/20・11/28 から悪化しないこと）も，Iter60〜62 で使ってきた
   `_domain_pair_coverage_maps()` 経由のアドホック集計をそのまま踏襲でき，新規実装は不要．
7. **N2' は上記 3 の理由で公式スクリプトの `compute_top1_accuracy()` を素通しで使うだけでは
   検証にならない**．計画フェーズは「`selected_domain` を新 rank_1 で上書きした行を使って
   `compute_top1_accuracy()` を呼ぶ」ことを明示的な実装項目として登録する必要がある．

**結論**

reflector の見立て（「変更範囲は `build_new_rows()` 1 関数のみ」）は rank_1/rank_2 の選択ロジック
自体については正しいが，**実行を通すには最低でも `_assert_rank1_unchanged()`（A1）の役割変更が
追加で必要**であり，かつ**N2'（top1_accuracy の非退行）を意味のある検証にするには `selected_domain`
フィールドの扱いを計画フェーズで明示的に決める必要がある**（決めずに実装すると自明 PASS になる）．
reflector の実行可能性の参考値（41→61，0.445→0.49，0.610→0.6033）はすべて独立再計算で一致し，
本レバーが no-op でないことが低コストで確認できた．S5 に必要な行単位 McNemar は
`metrics._mcnemar_from_correctness()` を薄く再利用すれば新規実装なしで足りる．Iter61 の資産は
全て実在し再利用可能．

**次フェーズへの示唆**

- 変更対象は `evaluate_dispatch_candidate_ranking.py` 内の **2 箇所**（`build_new_rows()` の
  rank_1 決定ロジック，`_assert_rank1_unchanged()`／`_run()` 呼び出し側の役割変更）と申し送ること．
  「1 関数のみ」と誤って計画すると実装フェーズで確実にアサーションエラーに当たる．
- N2' の計算方法（`selected_domain` を新 rank_1 で上書きするか否か）を計画フェーズで**必ず
  事前登録に明記**すること．上書きする場合，実測では悪化ではなく改善方向（+0.625pt 相当）と
  見込まれるため，N2' の閾値（≧0.5875）は現実的には楽に PASS しうるが，それでも「何を測る指標か」
  を曖昧にしないこと．
- S5 は `metrics._mcnemar_from_correctness()` を行 ID キーの 100 件 bool 辞書で呼ぶアドホックコードを
  1 つ追加するだけで実装できる（新しい統計機構は不要）．N6 は Iter60〜62 のアドホック集計パターンを
  そのまま踏襲する．
- 較正は使わない（Iter61 の未較正ヘッドを再利用．較正済みヘッドと同時に動かすと単一レバー原則が
  破れる）という config.yml の指示を計画フェーズでもそのまま踏襲してよい．
- 実行コストは Q4 の再現で示したとおり極めて低い（既存 JSONL の `head_scores` のみで大半の検証が
  完結し，実機オフライン採点コマンド自体も埋め込みキャッシュ完全ヒットが見込まれる）．

### 計画 (Iter63)

**仮説**

Iter60〜62 で rank_2 側は 0.345→0.445 まで改善したのち較正でも動かず飽和した．残るボトルネックが
**rank_1（基準線ルータの top-1）が複合 100 行で 41/100 しか当たらず，`compound_domain_set_recall` の
構造的上限を 0.705 に固定していること**にあるなら，rank_1 の選択元を多ラベルヘッドの argmax へ
切り替えれば，rank_2 を含む他の全構成を Iter61 のまま固定したままで複合行の被覆が 0.445 を超えて
向上するはずである．逆に向上しないなら，rank_1/rank_2 の双方をヘッド側で決めても現行の埋め込み
表現・訓練データでは上限に到達しており，残るのは合成データ設計・データセット再設計・埋め込み適応に
限られることになる．

**単一レバー（今回変更する唯一の変数）**

`rank1_source`: `baseline_router_selected_domain`（Iter59〜62 の実質値＝基準線 JSONL の
`dispatched_domains[0]` をコピー）→ **`multilabel_head_argmax`**（`head_scores` の最大ドメイン）．
rank_2 は従来どおり「同じ `head_scores` のうち rank_1 を除く最大」であり，**選択ロジックも
ヘッドも訓練データも埋め込みも変更しない**（`build_new_rows():211-214` は `rank_1` を除外して
`max` を取る構造なので，rank_1 を差し替えるだけで除外集合が自動追随し，rank_1==rank_2 の重複は
構造的に起こり得ない＝A2 が保証）．

**確定した実装仕様（本フェーズの決定事項 1・2・3）**

変更は `scripts/evaluate_dispatch_candidate_ranking.py` の **1 ファイル**に閉じる（訓練スクリプトは
無変更，較正済みヘッドは使わない）．変更箇所は次の 4 点で，いずれも「rank_1 の選択元」という
単一の変数に従属する．

1. **rank_1 の選択元を CLI フラグで切り替える（既存挙動は既定値として温存する）**
   - `_parse_args()` に `--rank1-source {baseline,head_argmax}`（**既定値 `baseline`**）を追加し，
     `_run()` 経由で `build_new_rows()` に `rank1_source: str` 引数として渡す．
   - `build_new_rows()`（`:193-226`）の `rank_1 = row["dispatched_domains"][0]`（`:208`）を，
     `head_scores` を先に計算したうえで
     `rank_1 = row["dispatched_domains"][0] if rank1_source == _RANK1_SOURCE_BASELINE
     else max(head_scores, key=lambda d: head_scores[d])` に変更する
     （`_head_scores()` の呼び出しを rank_1 決定より前へ移動する必要がある）．
     `_RANK1_SOURCE_BASELINE = "baseline"` / `_RANK1_SOURCE_HEAD_ARGMAX = "head_argmax"` を
     モジュール定数として置く（マジック文字列を埋め込まない）．
   - **フラグ方式を採る理由**: 既定値 `baseline` を維持することで Iter59〜62 の全実行・全
     アサーションがそのまま再現でき（A10 で実証する），「既存のテストを理由なく削除・弱体化
     させない」という CLAUDE.md 規約を満たしつつ，新旧を同一コードパスで A/B できる．
     ヘッド種別による分岐ではなく明示フラグとするため，どちらのモードで生成された成果物かが
     CLI 履歴から一意に追える．
2. **A1（`_assert_rank1_unchanged()`）の役割を再定義する（削除しない）**
   - `_assert_rank1_unchanged()`（`:229-240`）は**そのまま残し**，`_run()`（`:380`）での呼び出しを
     `if rank1_source == _RANK1_SOURCE_BASELINE:` の条件下に置く．既定モードでは従来どおり
     `AssertionError` を送出する強度を保つ（＝旧設計の不変条件は弱体化しない）．
   - `head_argmax` モードでは，代わりに**新設 A9 `_assert_rank1_matches_head_argmax(new_rows)`**
     を呼ぶ．「全 1600 行で `dispatched_domains[0] == max(head_scores, key=...)` であること」を
     検証し，不一致があれば `AssertionError` を送出する．**A1 が守っていた「rank_1 の由来が
     設計どおりであること」という役割を，新しい由来（ヘッド argmax）に対して同じ強度で
     引き継ぐ**（安全装置を外すのではなく，検査対象の定義を差し替える）．
   - さらに `head_argmax` モードでは**非致命的な報告**として
     `_compute_rank1_change_count(baseline_rows, new_rows)` を新設し，基準線と異なる rank_1 の
     行数・率を `summary["rank1_change_count"]` / `["rank1_change_rate"]` に出力する
     （`_compute_rank2_flip_rate()`（`:266-269`）と同型の設計）．これが S4 の直接の計測値となる．
   - モジュール docstring（`:5-13`，`:15-30`）の「rank_1 は再計算しない」という記述を，
     `--rank1-source` による 2 モード制の説明へ更新する（実行には影響しないが古い記述を残さない）．
3. **`selected_domain` は `head_argmax` モードでのみ新 rank_1 で上書きする（本フェーズの決定）**
   - `build_new_rows():219` の `"selected_domain": row["selected_domain"]` を，
     `head_argmax` モードでは `rank_1`（新 rank_1）を入れるよう変更する．`baseline` モードでは
     従来どおり基準線からの素通しコピー（＝挙動不変）．
   - **決定理由**: (a) 基準線 JSONL では `selected_domain == dispatched_domains[0]` が
     **1600/1600 行で成立している**ことを本フェーズで実測した（不一致 0 行）ため，
     「`selected_domain` ＝ top-1 に選ばれたドメイン」という不変条件を新 rank_1 でも維持する方が
     データの意味論として整合する．(b) 上書きしない場合，`metrics.compute_top1_accuracy()`
     （`metrics.py:42`）は `selected_domain` しか見ないため N2' が 0.5975=0.5975 の**自明 PASS**
     となり，「レバーを読むコードに到達しない」型の穴（Iter58 の教訓）を再生産する．
   - **N2' の基準値は config.yml 暫定案の ≧0.5875 をそのまま確定する**（閾値は変更しない）．
     上書きにより指標の意味は「基準線ルータの top-1 精度」から「ヘッド argmax の top-1 精度」へ
     変わるが，比較対象（基準線 0.5975）と許容幅（-1.0pt）は事前登録どおり据え置く．
     調査フェーズの実測では 0.60375（+0.625pt）と改善方向であり，この閾値で棄却されることは
     考えにくいが，**「rank_1 をヘッドに任せても全 1600 行の top-1 精度を 1.0pt 超は落とさない」
     という仮説を実際に検定している**点に意味がある．
4. **統計スクリプト `scripts/compute_iter59_ranking_stats.py` は無改造で使う**
   - S1・S2・S3・N3 はそのまま利用する．**N1（`_compute_n1`）は本レバーでは定義上 `pass:false` に
     なるため，出力は記録するが判定には用いない**（config.yml が撤回を明記済み．例外は送出せず
     `pass:false` を返すだけなのでパイプラインは壊れない）．**N2（`_compute_n2`）も `exact_match`
     を要求する実装のため `pass:false` になりうるが，判定には `new_top1_accuracy` の数値のみを
     N2' の閾値（≧0.5875）と照合して用いる**．同スクリプトの S4 はドメイン対単位の rank_2 変化を
     測るものであり，本レバーの S4（rank_1 変化行数）とは別物である点を混同しないこと．
   - **S5（複合 100 行の rank_1 正解数の対応あり exact McNemar）**と **N6（education/medical の
     自己被覆）**は，Iter60〜62 と同じ「既存関数を読み取り専用で薄く呼ぶアドホック集計」で行う．
     S5 は `metrics._mcnemar_from_correctness()` に行 ID をキーとする 100 件の bool 辞書
     （`dispatched_domains[0] in expected_domains`）を 2 つ渡すだけで足り，新しい統計機構は
     実装しない．N6 は `compute_iter59_ranking_stats.py:_domain_pair_coverage_maps()` を流用する．

**固定する構成（Iter61 から一切変えない）**

- ヘッド: `models/dispatch_multilabel_head_iter61.joblib`（未較正 `OneVsRestClassifier
  (LogisticRegression(max_iter=1000, class_weight="balanced"))`）を**再訓練せず再利用**．
  **Iter62 の較正済みヘッド（`..._iter62.joblib`）は使わない**（較正の有無を同時に動かすと
  単一レバー原則が破れる．config.yml 指示どおり）．
- 訓練データ `data/classifier_train.jsonl`（1427 行）＋
  `data/classifier_train_multidomain_iter61.jsonl`（135 行）: 再生成しない（実行経路に登場しない）．
- 埋め込みモデル `nomic-embed-text`，キャッシュ `results/iter59_query_embeddings.npz`．
- rank_2 の選択ロジック（`rank_1` 以外の `head_scores` 最大），`_head_scores()`（Iter62 で
  `predict_proba()` に統一済み），`_load_head()`，A2/A3/A6，N5 の計算（`_compute_single_domain_
  argmax_accuracy()` は rank_1 の由来に依存しないヘッド自身の argmax 精度を測る）．
- 基準線 `results/20260918_202613/results.jsonl`（`compound_domain_set_recall` = 0.345，
  `top1_accuracy` = 0.5975），統計スクリプト（**無改造**），`config.yaml`．
- **実行時経路（`node.py` のルータ）への配線は本イテレーションでも行わない**（B94/B95．
  本レバーは rank_1 まで実行時ルータから乖離させるため，配線の人間判断の重要性が増す）．

**出力ファイル命名（Iter61/62 の成果物を上書きしないこと）**

| 種別 | 既存（保護・読み取り専用） | Iter63（新規作成） |
|---|---|---|
| ヘッド | `models/dispatch_multilabel_head_iter61.joblib` | （再利用．新規訓練なし） |
| 予測 | `results/iter61_multilabel_ranking_predictions.jsonl` | `results/iter63_multilabel_ranking_predictions.jsonl` |
| 統計 | `results/iter62_stats.json` | `results/iter63_stats.json` |
| 既定モード回帰検証（A10） | — | `results/iter63_regression_check_baseline_mode.jsonl` |

**実行コマンド（`--ollama-host` は疎通する方を使う．採点時の実績は SSH ローカルフォワード
`127.0.0.1:11435`．埋め込みキャッシュ完全ヒット＝embed 呼び出し 0 件の想定）**

```
# 0) A10 事前チェック（フラグ導入が既定モードで no-op であることの実証）
#    --rank1-source を省略（既定 baseline）して Iter61 のヘッドを再採点し，
#    results/iter61_multilabel_ranking_predictions.jsonl と完全一致することを確認する．
uv run python -m scripts.evaluate_dispatch_candidate_ranking \
    --baseline results/20260918_202613/results.jsonl \
    --head models/dispatch_multilabel_head_iter61.joblib \
    --embedding-model nomic-embed-text --ollama-host 127.0.0.1 --ollama-port 11435 \
    --embedding-cache results/iter59_query_embeddings.npz \
    --output results/iter63_regression_check_baseline_mode.jsonl
diff <(jq -cS . results/iter61_multilabel_ranking_predictions.jsonl) \
     <(jq -cS . results/iter63_regression_check_baseline_mode.jsonl)

# 1) 本走: rank_1 をヘッド argmax にして 1600 問をオフライン採点
#    --iter59-predictions は引数名に反して汎用（_compute_a5_iter59_disagreement()）．
#    対 Iter61 の rank_2 変化を見るため Iter61 の予測を渡す（参考情報．S4 とは別）．
uv run python -m scripts.evaluate_dispatch_candidate_ranking \
    --baseline results/20260918_202613/results.jsonl \
    --head models/dispatch_multilabel_head_iter61.joblib \
    --rank1-source head_argmax \
    --embedding-model nomic-embed-text --ollama-host 127.0.0.1 --ollama-port 11435 \
    --embedding-cache results/iter59_query_embeddings.npz \
    --iter59-predictions results/iter61_multilabel_ranking_predictions.jsonl \
    --output results/iter63_multilabel_ranking_predictions.jsonl

# 2) 指標・検定（Iter59〜62 と同一スクリプト・同一手続き．無改造）
#    N1 は定義上 pass:false，N2 は exact_match:false になりうるが，いずれも記録のみ．
uv run python -m scripts.compute_iter59_ranking_stats \
    --baseline results/20260918_202613/results.jsonl \
    --new results/iter63_multilabel_ranking_predictions.jsonl \
    --output results/iter63_stats.json

# 3) S5（複合 100 行の rank_1 正解数の対応あり exact McNemar）と N6（education/medical 被覆）の
#    アドホック集計．metrics._mcnemar_from_correctness() と
#    compute_iter59_ranking_stats._domain_pair_coverage_maps() を読み取り専用で呼ぶ．
#    公式の採点・統計パスには手を入れない（Iter60〜62 の N6 と同じ手順）．
```

**成功条件（事前登録．事後変更禁止）**

config.yml:994-1005 の暫定案を，上記決定事項 3（`selected_domain` の上書き）を反映したうえで
**閾値は一切変更せずに確定する**．基準線は `results/20260918_202613/results.jsonl`
（`compound_domain_set_recall` = 0.345，複合 100 行の rank_1 正解 41/100，`top1_accuracy` = 0.5975），
第 2 参照点は Iter61/62（0.445，education 6/20，medical 11/28）で，両者を必ず併記する．

- **S1（有意性の維持）**: 全体 200 ペアの exact McNemar（`scipy.stats.binomtest`，α=0.05）で
  基準線比 **p < 0.05**．
- **S2（全体性能の向上）**: `compound_domain_set_recall` が **≧ 0.465**
  （Iter61/62 の 0.445 を点推定で +2.0pt 以上上回る）．
- **S3（コスト中立）**: `mean_dispatch = 2.000000`（完全一致）．
- **S4（発火の証拠）**: `summary["rank1_change_count"] > 0`（rank_1 が基準線と異なる行が存在する）．
- **S5（本レバー固有の主基準）**: 複合 100 行の rank_1 正解数が基準線 41/100 を
  **対応あり exact McNemar で有意に上回る（p < 0.05，かつ正解数が 41 より大きい方向）**．

**判定規則（事前登録）**

- **S1〜S5 全充足 かつ N2'/N3/N5/N6 全充足** → `rank1_source = multilabel_head_argmax` を
  **adopted**．「rank_1 側がボトルネックであり，ヘッド argmax で解消できる」と結論し，
  実行時経路への配線（B95 要レビュー 1）の優先度を最上位へ引き上げる．
- **S1・S3・S4 充足，S5 充足，S2 のみ不成立（ただし 0.445 ≦ recall < 0.465）かつ非退行全充足**
  → **partial**．「rank_1 の改善は複合行の正解を有意に増やすが，集合再現率の押し上げは +2.0pt に
  届かない」と記録する．
- **S1・S3・S4 充足，S5 不成立（p ≧ 0.05）だが複合 100 行の rank_1 正解数 > 41 かつ S2 充足**
  → **partial**．効果量はあるが検出力不足である可能性を記録し，次イテレーションへ引き継ぐ．
- **S2 不成立（recall < 0.445＝Iter61/62 水準割れ）** → **rejected**．
- **S4 不成立（`rank1_change_count == 0`）** → **no-op** として rejected 扱いとし，
  ヘッド argmax が基準線ルータと一致していたことを機序として記録する．
- **非退行 N2'/N3/N5/N6 のいずれかが FAIL** → **adopted にはしない**（S 側が全充足でも最大 partial
  とし，どの指標が退行したかを対外記述の留保として残す）．

**非退行条件（事前登録．N1 は本レバーでは定義上成立しないため撤回済み）**

- **N2'**: 全 1600 行の `top1_accuracy`（**`selected_domain` を新 rank_1 で上書きした行**に対し
  `metrics.compute_top1_accuracy()` で算出）が基準線 0.5975 から 1.0pt を超えて低下しない
  （**≧ 0.5875**）．統計スクリプトの `N2_top1_accuracy_invariance.new_top1_accuracy` を用いる．
- **N3**: legal 自身の被覆 **≧ 8/30**（`_compute_n3()` が自動判定）．
- **N5**: 単一ドメイン 1500 行の argmax 正解率 **≧ 0.590**
  （`_compute_single_domain_argmax_accuracy()`．rank_1 の由来に依存しないヘッド固有の指標）．
- **N6**: education **≧ 6/20** かつ medical **≧ 11/28**（Iter62 の水準から悪化しないこと）．

**アサーション（no-op・交絡対策）**

- **A1（条件付き温存）**: `--rank1-source baseline` のときのみ従来どおり実行し，
  rank_1 の基準線一致を `AssertionError` 強度で検証する（弱体化させない）．
- **A2/A3/A6**: Iter59〜62 の定義のまま無変更で実行する（A2 が rank_1==rank_2 の重複不在を保証）．
- **A5（読み替え・参考）**: 対 Iter61 予測の rank_2 不一致件数を報告する（本レバーでは rank_1 の
  変化に伴い rank_2 の除外集合が変わるため 0 にはならない見込み．判定には用いない）．
- **A9（新設・本レバー固有）**: `--rank1-source head_argmax` のとき，全 1600 行で
  `dispatched_domains[0] == argmax(head_scores)` であること．不一致があれば `AssertionError`．
  A1 が担っていた「rank_1 の由来が設計どおりである」という保証を新しい由来へ引き継ぐ．
- **A10（新設・交絡対策）**: 上記コマンド 0) のとおり，フラグ導入後のスクリプトを**既定モード**で
  Iter61 のヘッドに適用した結果が `results/iter61_multilabel_ranking_predictions.jsonl` と
  **全フィールド完全一致**すること．一致しなければ，フラグ導入のリファクタ自体が効果に混入して
  いることになり，単一レバー原則が破れるため実験を中止して原因を調査する．

**単一レバー原則の確認（混入チェック）**

- ヘッド（Iter61 の未較正 joblib）: 再訓練せず再利用 → **無変更**．較正は導入しない．
- 訓練データ・合成データ生成・埋め込みモデル・埋め込みキャッシュ → 実行経路に登場しない
  → **無変更**．
- rank_2 の選択ロジック・`_head_scores()`・`_load_head()`・A2/A3/A6・N5 の計算 → **無変更**．
- 基準線 JSONL・統計スクリプト・`config.yaml` → **無変更**．
- コード変更は 1 ファイル 4 箇所だが，そのすべてが「rank_1 の選択元」という単一の変数に従属する
  （CLI フラグの追加，rank_1 決定式，A1→A9 の切替，`selected_domain` の由来）．既定モードでは
  全挙動が従来と bit 単位で一致することを A10 で実証するため，実効的な変数は
  **「rank_1 をどこから取るか」1 つだけ**である．
- `selected_domain` の上書きは rank_1 変更に**構造的に従属する**（基準線でも
  `selected_domain == dispatched_domains[0]` が 1600/1600 で成立しており，同じ不変条件を
  維持するだけ）であり，独立した第 2 のレバーではない．
- 出力パス名の変更は測定対象に影響しない（ファイル I/O のみ）．

**既知の制約（申し送り）**

- 本構成は rank_1・rank_2 の双方をオフラインのヘッドが決めるため，**実行時経路（`node.py` の
  ルータによる top-1 選択）との乖離が最大になる**．オフラインで adopted となっても，実機での
  有効性は配線・本走なしには一切主張できない（B94/B95．対外記述に必ず留保を付す）．
- `selected_domain` の上書きは「ヘッド argmax が top-1 として dispatch される」という仮定の下での
  値であり，実機の `aggregator.select_best_dispatch_response()`（実応答の confidence で選ぶ）を
  再現するものではない．`_compute_n2()` の caveat（`compute_iter59_ranking_stats.py:213-223`）は
  本イテレーションでも有効であり，N2' は「オフライン上の field の意味論的整合」に留まる．
- 統計スクリプトの N1 は `pass:false` 固定になる（撤回済みのため判定に用いない）．出力 JSON を
  そのまま「全項目 pass」と読むことはできない点を reflector へ申し送る．
- 低品質行（プロンプトの echo）の混入は Iter60 から続く既知の穴であり，訓練データを据え置く
  本イテレーションでも残る．
- S2/S5 の参考値（recall 0.49，複合 rank_1 正解 61/100）は評価集合上で事前に算出した値であり，
  **効果量の推定値として対外記述に引用してはならない**（config.yml note と同じ制約）．

### 実装 (Iter63)

計画フェーズが確定した仕様どおり，`scripts/evaluate_dispatch_candidate_ranking.py` **1 ファイル
4 箇所**のみを変更した（訓練スクリプト・ヘッド・訓練データ・埋め込み・基準線 JSONL・統計スクリプト
`compute_iter59_ranking_stats.py`・`config.yaml` は無変更）．

1. **CLI フラグ追加**: `_parse_args()` に `--rank1-source {baseline,head_argmax}`
   （既定値 `baseline`，`choices=` で不正値を argparse レベルで拒否）を追加し，`main()` から
   `_run(..., rank1_source=args.rank1_source)` として渡した．マジック文字列を避けるため
   モジュール定数 `_RANK1_SOURCE_BASELINE = "baseline"` / `_RANK1_SOURCE_HEAD_ARGMAX = "head_argmax"`
   を新設（`_N5_SINGLE_DOMAIN_ARGMAX_ACCURACY_FLOOR` の直後）．
2. **`build_new_rows()` の rank_1 決定式を切替**: シグネチャに `rank1_source: str =
   _RANK1_SOURCE_BASELINE` を追加し，`_head_scores()` の呼び出しをループ内で rank_1 決定より
   前へ移動．`baseline` モードは従来どおり `rank_1 = row["dispatched_domains"][0]` と
   `selected_domain = row["selected_domain"]`（素通しコピー，挙動不変）．`head_argmax` モードは
   `rank_1 = max(head_scores, key=lambda domain: head_scores[domain])` とし，
   `selected_domain = rank_1`（計画フェーズ決定事項 3 のとおり新 rank_1 で上書き）．
   `rank2_new = max((d for d in head_scores if d != rank_1), ...)` はどちらのモードでも無変更
   （除外集合が rank_1 の由来変更に自動追随するため，A2 の重複不在保証もそのまま成立）．
3. **A1 を既定モード限定にし，`head_argmax` モードでは新設 A9 へ差し替え**: `_run()` の
   `_assert_rank1_unchanged(baseline_rows, new_rows)` 呼び出しを
   `if rank1_source == _RANK1_SOURCE_BASELINE:` の条件下に置いた（`_assert_rank1_unchanged()`
   自体は削除せず，強度も無変更）．`head_argmax` モードでは新設
   `_assert_rank1_matches_head_argmax(new_rows)`（A9）を呼び，全行で
   `dispatched_domains[0] == argmax(head_scores)` を検証し，不一致があれば
   `AssertionError` を送出する．
4. **S4 の非致命的レポート**: 新設 `_compute_rank1_change_count(baseline_rows, new_rows) -> int`
   （`_compute_rank2_flip_rate()` と同型の単一値返却，読み取り専用の集計のみで例外は送出しない）を
   `head_argmax` モードでのみ `_run()` から呼び，`summary["rank1_change_count"]` /
   `summary["rank1_change_rate"]` として標準エラー出力の JSON サマリに含めた（`baseline` モードでは
   両キーとも出力しない＝既存の JSON 構造に対する非破壊的な追加）．
5. **モジュール docstring 更新**: 冒頭のいわゆる「rank_1 は再計算しない」という説明（旧 `:5-13`）を，
   既定モード（`baseline`）ではその説明が引き続き成立すること，`head_argmax` モードでは A1 の代わりに
   A9 が同じ「由来が設計どおりであること」の保証役を引き継ぐことを追記する形に更新した
   （実行への影響はなし，計画節「決定事項 2」どおり）．

**A10 事前チェック（実機実行・完了，交絡対策）**

計画節記載のコマンド 0）をそのまま実行した（SSH ローカルポートフォワード
`127.0.0.1:11435` 経由，実行前から稼働中のトンネルで疎通確認済み）．

```
uv run python -m scripts.evaluate_dispatch_candidate_ranking \
    --baseline results/20260918_202613/results.jsonl \
    --head models/dispatch_multilabel_head_iter61.joblib \
    --embedding-model nomic-embed-text --ollama-host 127.0.0.1 --ollama-port 11435 \
    --embedding-cache results/iter59_query_embeddings.npz \
    --output results/iter63_regression_check_baseline_mode.jsonl
```

`--rank1-source` を省略（既定 `baseline`）して実行し，`AssertionError` なく完走
（サマリ: `compound_domain_set_recall=0.445`，`n5_single_domain_argmax_accuracy.accuracy=0.6033`，
いずれも Iter61/62 の既知値と一致，埋め込みキャッシュ完全ヒットのため embed 呼び出し 0 件）．
`diff <(jq -cS . results/iter61_multilabel_ranking_predictions.jsonl) <(jq -cS . results/iter63_regression_check_baseline_mode.jsonl)`
は**終了コード 0・差分 0 行**（全 1600 行・全フィールド完全一致）．**A10 PASS**:
フラグ導入のリファクタ自体は既定モードで bit 単位の no-op であり，本レバーの効果測定に混入しない．

**`head_argmax` モードのスモークテスト（実機実行・完了，クラッシュしないことの確認のみ）**

計画節記載のコマンド 1）をそのまま実行した（統計計算コマンド 2)〜3) は実験フェーズの担当のため
今回は実行していない）。

```
uv run python -m scripts.evaluate_dispatch_candidate_ranking \
    --baseline results/20260918_202613/results.jsonl \
    --head models/dispatch_multilabel_head_iter61.joblib \
    --rank1-source head_argmax \
    --embedding-model nomic-embed-text --ollama-host 127.0.0.1 --ollama-port 11435 \
    --embedding-cache results/iter59_query_embeddings.npz \
    --iter59-predictions results/iter61_multilabel_ranking_predictions.jsonl \
    --output results/iter63_multilabel_ranking_predictions.jsonl
```

A9 の `AssertionError` なく完走．サマリ: `compound_domain_set_recall=0.49`
（調査フェーズの独立再計算値と一致），`rank1_change_count=380`／`rank1_change_rate=0.2375`
（調査フェーズの実測「380/1600（23.75%）」と一致，S4 は発火の証拠として非ゼロ確認済み），
`n5_single_domain_argmax_accuracy.accuracy=0.6033`（floor 0.590 を pass），
`a5_iter59_disagreement.mismatches=380`（rank_1 変化に伴い rank_2 の除外集合が変わるため
参考値として非ゼロ，判定には用いない）．
`results/iter63_multilabel_ranking_predictions.jsonl` は決定的（固定の埋め込みキャッシュ・
固定のヘッド）なため，実験フェーズが同一コマンドを再実行しても同一ファイルが得られる見込みだが，
念のため実験フェーズで再生成し直すことを推奨する．

**検証**

- `uv run ruff check scripts/evaluate_dispatch_candidate_ranking.py
  tests/test_evaluate_dispatch_candidate_ranking.py` → `All checks passed!`．
  リポジトリ全体の `uv run ruff check .` は既存の無関係ファイル（`scripts/analyze_iter52.py`・
  `scripts/prepare_lora_training_data.py` 等，計 23 件）で F541/F401 が出るが，いずれも本
  イテレーションの変更対象外であり，本イテレーションで触った 2 ファイルには 1 件も含まれない．
- `mypy` はリポジトリに未導入（`pyproject.toml` に設定なし）のため型チェックは実施していない
  （Iter62 と同じ既知の制約）．
- 新規ユニットテストを 5 件追加した（`tests/test_evaluate_dispatch_candidate_ranking.py`）:
  `build_new_rows()` の baseline/head_argmax 両モードの rank_1・selected_domain の挙動，
  A9（`_assert_rank1_matches_head_argmax`）の pass/fail 両方，S4
  （`_compute_rank1_change_count`）の集計値．いずれも既存の joblib フィクスチャ
  （`_fit_iter59_style_head()`）や辞書直書きの手法を踏襲し，新しい Wrapper やモックは導入していない．
- `uv run pytest tests/test_evaluate_dispatch_candidate_ranking.py -q` → **13 件全て pass**
  （既存 8 件・新規 5 件）．
- 全体テスト (`uv run pytest -q`) では上記ファイル以外に `test_build_dataset.py`・
  `test_train_domain_classifier.py` の計 12 件が失敗するが，`git stash push --
  scripts/evaluate_dispatch_candidate_ranking.py tests/test_evaluate_dispatch_candidate_ranking.py`
  で本イテレーションの変更のみを退避して再実行しても同じ 12 件が同じ原因
  （`train_domain_classifier.py` が `CalibratedClassifierCV.classes_` に依存しており，
  別イテレーションの既存資産で今回の変更対象外）で失敗することを確認済みであり，
  本イテレーションの変更によるものではない．

**実験フェーズへの申し送り**

コードの変更は上記 4 箇所のみで A10 も PASS しているため，計画節のコマンド 2)〜3)
（`results/iter63_multilabel_ranking_predictions.jsonl` に対する統計計算・S5/N6 のアドホック集計）を
そのまま実行してよい状態にある．`results/iter63_multilabel_ranking_predictions.jsonl` は本フェーズの
スモークテストで既に生成済み（上記のとおり `rank1_change_count`/`compound_domain_set_recall` は
調査フェーズの独立再計算値と一致）だが，実験フェーズで以下のコマンドを再実行し直すことを推奨する．

```
# 1) 本走（既に実施済みだが，実験フェーズで独立に再実行して確定させることを推奨）
uv run python -m scripts.evaluate_dispatch_candidate_ranking \
    --baseline results/20260918_202613/results.jsonl \
    --head models/dispatch_multilabel_head_iter61.joblib \
    --rank1-source head_argmax \
    --embedding-model nomic-embed-text --ollama-host 127.0.0.1 --ollama-port 11435 \
    --embedding-cache results/iter59_query_embeddings.npz \
    --iter59-predictions results/iter61_multilabel_ranking_predictions.jsonl \
    --output results/iter63_multilabel_ranking_predictions.jsonl

# 2) 指標・検定（S1・S2・S3・N3 はそのまま利用，N1/N2 は記録のみで判定に用いない）
uv run python -m scripts.compute_iter59_ranking_stats \
    --baseline results/20260918_202613/results.jsonl \
    --new results/iter63_multilabel_ranking_predictions.jsonl \
    --output results/iter63_stats.json

# 3) S5（複合100行のrank_1正解数の対応ありexact McNemar）とN6（education/medical被覆）の
#    アドホック集計。metrics._mcnemar_from_correctness() と
#    compute_iter59_ranking_stats._domain_pair_coverage_maps() を読み取り専用で呼ぶこと
#    （新しい統計機構を実装しないこと。行IDキーの100件bool辞書
#    `dispatched_domains[0] in expected_domains` を2つ渡すだけで足りる）。
```

判定は journal.md 計画節の「判定規則（事前登録）」（adopted / partial / rejected / no-op の 5 分岐）
をそのまま用いること．閾値・判定規則は本フェーズでは一切変更していない．

### 実験 (Iter63)

**本レバーはオフライン完結（backlog B96・config.yml 事前登録どおり）**．埋め込みキャッシュ
`results/iter59_query_embeddings.npz` を再利用し，実機 1600 問本走・生成トラフィックとも不要．
`mise run deploy`／`mise run start` は実行していない（journal 計画節「実行時経路（node.py の
ルータ）への配線は本イテレーションでも行わない」との整合）．

**接続先ホストの再確認（実装フェーズと同じ状況の再確認）**

`curl -m 3 http://192.168.15.100:11434/api/tags` はタイムアウト（exit 28，応答なし）で直接 IP は
不通．一方，実行前から稼働中の SSH ローカルポートフォワード（`127.0.0.1:11435`）経由の
`curl http://127.0.0.1:11435/api/tags` は応答し `nomic-embed-text:latest` を含むモデル一覧を返した
ため，実装フェーズと同じ `--ollama-host 127.0.0.1 --ollama-port 11435` を用いた（レバー以外の
パラメータ変更ではなく，接続先の読み替えのみ）．ただし全コマンドとも埋め込みキャッシュ完全
ヒットのため embed 呼び出しは 0 件だった．

**実行した3コマンド（journal 計画節・実装フェーズ申し送りのコマンドをそのまま独立に再実行した）**

```
# 1) 本走: rank_1 をヘッド argmax にして1600問をオフライン採点
uv run python -m scripts.evaluate_dispatch_candidate_ranking \
    --baseline results/20260918_202613/results.jsonl \
    --head models/dispatch_multilabel_head_iter61.joblib \
    --rank1-source head_argmax \
    --embedding-model nomic-embed-text --ollama-host 127.0.0.1 --ollama-port 11435 \
    --embedding-cache results/iter59_query_embeddings.npz \
    --iter59-predictions results/iter61_multilabel_ranking_predictions.jsonl \
    --output results/iter63_multilabel_ranking_predictions.jsonl

# 2) 指標・検定（compute_iter59_ranking_stats.py，無改造）
uv run python -m scripts.compute_iter59_ranking_stats \
    --baseline results/20260918_202613/results.jsonl \
    --new results/iter63_multilabel_ranking_predictions.jsonl \
    --output results/iter63_stats.json

# 3) S5（複合100行のrank_1正解数の対応ありexact McNemar）とN6（education/medical被覆）の
#    アドホック集計。metrics._mcnemar_from_correctness() と
#    compute_iter59_ranking_stats._domain_pair_coverage_maps() を読み取り専用で呼ぶ
#    （Iter60〜62 と同型のパターン。行IDキーのbool辞書を渡すだけで新規の統計機構は実装していない）。
```

所要時間は3コマンド合計で数秒（`time` 計測: コマンド1が実時間 3.28 秒，ユーザ時間 11.97 秒／
373% CPU＝埋め込み呼び出し 0 件・キャッシュ完全ヒット）．実行中の異常・OOM・タイムアウト・
クラッシュは発生していない．

**A10 の独立再確認（実験フェーズ自身でも交絡対策を再検証）**

実装フェーズが生成した `results/iter63_regression_check_baseline_mode.jsonl` に対し，
`diff <(jq -cS . results/iter61_multilabel_ranking_predictions.jsonl) <(jq -cS . results/iter63_regression_check_baseline_mode.jsonl)`
を実験フェーズでも再実行し，**終了コード 0・差分 0 行**（全 1600 行・全フィールド完全一致）を
再確認した．フラグ導入のリファクタは既定 `baseline` モードで bit 単位の no-op であり続けている．

**1) 本走結果（`evaluate_dispatch_candidate_ranking.py` 標準エラー出力の JSON 全文）**

```json
{
  "n_rows": 1600,
  "mean_dispatch": 2.0,
  "rank2_flip_rate": 0.563125,
  "compound_domain_set_recall": 0.49,
  "compound_rows_evaluated": 100,
  "n5_single_domain_argmax_accuracy": {
    "n_single_domain_rows": 1500,
    "correct": 905,
    "accuracy": 0.6033333333333334,
    "floor": 0.59,
    "pass": true
  },
  "a5_iter59_disagreement": {
    "n_rows": 1600,
    "mismatches": 380,
    "mismatch_rate": 0.2375
  },
  "rank1_change_count": 380,
  "rank1_change_rate": 0.2375
}
```

（A9：例外なし＝全 1600 行で `dispatched_domains[0] == argmax(head_scores)`．`a5_iter59_disagreement`
は `--iter59-predictions` に Iter61 予測を渡しているため対 Iter61 の rank_2 不一致を表す参考値で，
判定には用いない．実装フェーズのスモーク実行値と全項目一致し，`md5sum` も出力ファイル間で一致した
ため，独立再実行による決定性を確認できた．）出力: `results/iter63_multilabel_ranking_predictions.jsonl`
（1600 行，sha256 `4952bff9845d89c0568eb1909ab3422ab70ca07653da208a2bf9a7c4eb2f38af`）。

**2) 指標・検定結果（`results/iter63_stats.json` 全文，無改造スクリプトの出力，sha256
`1cc257f29467da75e88815208c3b5480ff2df5c7e4ea68be43b001a5da17eac9`）**

```json
{
  "baseline_compound_domain_set_recall": 0.345,
  "new_compound_domain_set_recall": 0.49,
  "matches_implementation_phase_diagnostic_recall": false,
  "S1_primary_criterion": {
    "n_pairs": 200,
    "improved_pairs": 43,
    "regressed_pairs": 14,
    "discordant_pairs": 57,
    "chi2_statistic_continuity_corrected": 13.75438596491228,
    "p_value_continuity_corrected": 0.00020833380892648634,
    "p_value_exact_binomtest": 0.0001538890224443423,
    "pass": true
  },
  "S2_effect_size_floor": {
    "baseline_recall": 0.345,
    "new_recall": 0.49,
    "delta_pt": 0.14500000000000002,
    "floor_pt": 0.04,
    "pass": true
  },
  "S3_cost_neutrality": {
    "n_rows": 1600,
    "length_distribution": {"2": 1600},
    "duplicate_rank1_rank2_count": 0,
    "mean_dispatch": 2.0,
    "pass": true
  },
  "S4_flip_rate_evidence_of_firing": {
    "n_rows": 1600,
    "flips": 901,
    "rank2_flip_rate": 0.563125,
    "matches_implementation_phase_value": false,
    "implementation_phase_value": 0.356875,
    "pass": true
  },
  "N1_rank1_invariance": {
    "n_rows": 1600,
    "mismatch_count": 380,
    "mismatch_ids": ["business_economics-001", "business_economics-005", "..."],
    "pass": false
  },
  "N2_top1_accuracy_invariance": {
    "baseline_top1_accuracy": 0.5975,
    "new_top1_accuracy": 0.60375,
    "exact_match": false,
    "pass": false
  },
  "N3_legal_non_regression": {
    "n_legal_involving_pairs": 30,
    "baseline_legal_self_coverage": 8,
    "new_legal_self_coverage": 17,
    "expected_baseline_value": 8,
    "baseline_matches_journal_record": true,
    "pass": true
  },
  "N4_improvement_breakdown_by_domain_category": {
    "legal_involving": {"n_pairs": 60, "improved": 14, "regressed": 3, "unchanged": 43},
    "medical_involving": {"n_pairs": 32, "improved": 3, "regressed": 2, "unchanged": 27},
    "other": {"n_pairs": 108, "improved": 26, "regressed": 9, "unchanged": 73}
  }
}
```

`N1_rank1_invariance.pass=false`・`N2_top1_accuracy_invariance.pass=false`（`exact_match` 判定）は
計画節が事前に「本レバーでは定義上こうなる／判定に用いない」と明記済みの項目であり，出力どおり
記録するに留める（N2' は下記 3) のアドホック計算で別途 `new_top1_accuracy=0.60375` の数値のみを
閾値 ≧0.5875 と照合する）。

**3) S5（複合100行のrank_1正解数の対応ありexact McNemar）と N6（education/medical自己被覆）の
アドホック集計（`metrics._mcnemar_from_correctness()` と
`compute_iter59_ranking_stats._domain_pair_coverage_maps()` を読み取り専用で呼んだ．新規の統計
機構は実装していない）**

```json
{
  "S5_compound_row_rank1_mcnemar": {
    "n_rows": 100,
    "baseline_rank1_correct": 41,
    "new_rank1_correct": 61,
    "discordant_baseline_only_correct": 7,
    "discordant_new_only_correct": 27,
    "discordant_pairs": 34,
    "chi2_continuity_corrected": 10.617647058823529,
    "p_value_continuity_corrected": 0.0011201348827110102,
    "p_value_exact_binomtest": 0.0008213953115046024,
    "direction_favors_new": true
  },
  "N6_education_medical_self_coverage": {
    "education": {"n_pairs": 20, "baseline_covered": 9, "new_covered": 3, "floor": 6},
    "medical": {"n_pairs": 28, "baseline_covered": 13, "new_covered": 18, "floor": 11}
  }
}
```

`baseline_rank1_correct=41`・`new_rank1_correct=61` は調査フェーズ・実装フェーズの独立再計算値
（Q4・スモークテスト）と完全一致した．`education.new_covered=3` は Iter62 の水準（6/20，config.yml
の N6 フロア）を下回っており，`medical.new_covered=18` は Iter62 の水準（11/28）を上回っている
（education 側の実測を裏取りするため，該当 20 行を個別に列挙し baseline/new の `dispatched_domains`
を突き合わせて手計算でも 3/20 と再確認した）。

**成果物・付随確認**

- 新規ファイル: `results/iter63_multilabel_ranking_predictions.jsonl`（実装フェーズのスモーク実行
  結果を独立再実行で再生成．md5sum が実装フェーズ生成時と一致し決定的であることを確認），
  `results/iter63_stats.json`（新規）。`results/iter63_regression_check_baseline_mode.jsonl`
  （実装フェーズ生成物，A10 用）は変更していない。
- Iter61/62 の保護対象資産（`models/dispatch_multilabel_head_iter61.joblib`・
  `results/iter61_multilabel_ranking_predictions.jsonl`・`results/iter61_stats.json`・
  `results/iter62_stats.json`・`results/iter62_multilabel_ranking_predictions.jsonl`・
  `data/classifier_train_multidomain_iter61.jsonl`）の `mtime` を実行前後で確認し，本実験フェーズの
  実行時刻（本日 10:53〜10:58）より前のまま一切変更されていないことを確認した。
- `git status --porcelain` を確認した。実装フェーズが変更した2ファイル
  （`scripts/evaluate_dispatch_candidate_ranking.py`・`tests/test_evaluate_dispatch_candidate_ranking.py`，
  いずれも `M`）と，研究サイクルのメタデータ3ファイル（`.claude/research/config.yml`・
  `.claude/research/journal.md`・`.claude/research/state.json`，各フェーズが継続的に更新するもの）
  以外に，**無関係な作業ツリー変更が2件存在する**（いずれも本イテレーションの変更対象外で，
  実験フェーズの開始前から存在していたもの）:
  (a) `config.yaml`（`central_router.embed_node_host` が `wafl502` から `wafl-ctrl5` へ変更されている。
  本イテレーションのコマンドはいずれも `--ollama-host`/`--ollama-port` を明示指定しており
  `config.yaml` を読まないため実行結果には影響していないが，出所不明の変更であり本フェーズでは
  一切手を触れていない）。
  (b) `results/iter45_preliminary/logs/wafl{500..509}/expert-mesh.log`（10 ファイルとも `M`。
  常時起動中のコンテナ（`OLLAMA_KEEP_ALIVE=-1`）が生成し続けているログの差分と見られ，本
  イテレーションのオフライン採点コマンドはこれらのログファイルに書き込みを行っていない）。
  本フェーズではいずれも変更・復元せず，事実として記録するに留める。
- 実行中の異常・障害は発生していない（3コマンドとも exit code 0，エラー出力なし，OOM・
  タイムアウトなし，A9/A10 とも例外なし）。

**解釈・採否判断はこのフェーズでは行わない**（次の分析(解釈)フェーズに委ねる。上記はすべて
生の実測値であり，`pass: true/false` はスクリプト自身が出力した事前登録済みアサーションの結果を
そのまま転記したものである）。

### 分析 (Iter63)

**数値集計のみ（解釈・採否判定は次フェーズ rc-analyst の担当）**

事前登録の成功条件・非退行条件（journal.md 本イテレーション計画節）に対応する実測値の対応表:

| 条件 | 事前登録の基準 | 実測値 | 基準との機械的な照合 |
|---|---|---|---|
| S1（有意性の維持） | 全体200ペアexact McNemar，p<0.05 | p=0.0001538890224443423（改善43／悪化14／discordant57） | 満たす |
| S2（全体性能の向上） | `compound_domain_set_recall` ≧0.465 | 0.49 | 満たす |
| S3（コスト中立） | mean_dispatch=2.000000（完全一致） | 2.000000（全1600行 len=2，重複0） | 満たす |
| S4（発火の証拠） | `rank1_change_count`>0 | 380（rate 0.2375） | 満たす |
| S5（本レバー固有の主基準） | 複合100行のrank_1正解数41/100を対応ありexact McNemarで有意に上回る（p<0.05かつ正解数>41） | 61/100，discordant34（改善27／悪化7），p=0.0008213953115046024 | 満たす |
| N2'（top1_accuracy非退行） | `selected_domain`上書き後のtop1_accuracy ≧0.5875 | 0.60375 | 満たす |
| N3（legal非退行） | legal自身の被覆 ≧8/30 | 17/30 | 満たす |
| N5（単一ドメインargmax非退行） | ≧0.590 | 0.6033333（905/1500） | 満たす |
| N6（education/medical非退行） | education≧6/20 かつ medical≧11/28 | education 3/20，medical 18/28 | **満たさない**（education単独が未達，AND条件のため全体として不成立） |

**参考（判定には用いない項目，計画節が事前に撤回・限定を明記済み）**

| 項目 | 実測値 |
|---|---|
| N1（rank_1不変，本レバーでは定義上成立しない） | 不一致380/1600（`pass:false`固定） |
| 統計スクリプト自身のN2（`exact_match`要求，判定には`new_top1_accuracy`の数値のみ使用） | `exact_match:false`（`new_top1_accuracy=0.60375`はN2'欄に転記済み） |

**判定不能だった項目**: なし。config.yml/計画節が事前登録した S1〜S5・N2'/N3/N5/N6 の全項目について，
`evaluate_dispatch_candidate_ranking.py`・`compute_iter59_ranking_stats.py` の出力および
`metrics._mcnemar_from_correctness()`／`_domain_pair_coverage_maps()` を用いたアドホック集計から
実測値を取得できた。

### 分析（解釈） (Iter63)

採否の最終確定（backlog 記録・config.yml 更新・次レバー選定・commit）は次の考察フェーズ
（rc-reflector）の担当であり，本節は行わない．本節は一次データ
（`results/20260918_202613/results.jsonl`・`results/iter60/61/62/63_multilabel_ranking_
predictions.jsonl` の各全 1600 行，`results/iter63_stats.json`）を直接集計した独立検算に基づく解釈である．

**0. 独立検算（Iter62 の先例に倣い，公式パスとは別に全項目を再計算した）**

実験節の全実測値が再現した．`compound` 100 行・`single` 1500 行の分離，`mean_dispatch=2.0`
（全 1600 行 `len=2`，rank_1==rank_2 の重複 0），A9 の不一致 0 行，`rank1_change_count=380`，
S1（200 ペア，改善 43／悪化 14／discordant 57，exact p=0.0001538890224443423），
S2（recall 基準線 0.3450 → 0.4900），S5（41/100 → 61/100，discordant 34＝新のみ正解 27／
基準線のみ正解 7，exact p=0.0008213953115046024），N2'（0.59750 → 0.60375），N5（905/1500=0.603333），
N3（legal 8/30 → 17/30），N6（education 3/20，medical 18/28）のすべてが実験節・`stats.json` と
小数以下まで一致した．以下の集計基盤は公式パスと整合している．

自己被覆の時系列（基準線→Iter60→Iter61→Iter62→Iter63，いずれも同一の複合 100 行上で再集計）:
**education 9→4→5→6→3**，**medical 13→19→12→11→18**，**legal 8→20→15→16→17**．
`compound_domain_set_recall` は **0.345→0.480→0.445→0.445→0.490**，複合 100 行の rank_1 正解は
**41→41→41→41→61**（Iter60〜62 は rank_1 を基準線からコピーしていたので 41 固定，本レバーで初めて動いた）．

**1. N6 不成立の機序（本フェーズの主眼）— 決定的な事実は「Iter63 の被覆＝ヘッド top-2 に入るか否か」に尽きる**

本レバー適用後の `dispatched_domains` は **ヘッドの top-2 そのもの**（rank_1＝argmax，rank_2＝
argmax を除く max）である．したがって任意のドメイン d の自己被覆数は
「d が期待ドメインに含まれる複合行のうち，d がヘッド内順位 1 位または 2 位である行数」と
**恒等的に等しい**．該当 20/28 行でヘッド内順位を集計した実測:

| ドメイン | 対象行 | ヘッド 1 位 | ヘッド 2 位 | top-2 計 | 中央順位 | Iter63 実測被覆 |
|---|---|---|---|---|---|---|
| education | 20 | 3 | **0** | **3** | 5 位 | 3/20 |
| medical | 28 | 9 | 9 | 18 | 2 位 | 18/28 |

すなわち **education は 20 行中 1 行もヘッド 2 位に来ておらず**，被覆 3/20 はヘッド 1 位の 3 行
（compound-016・019・095）だけで構成される．medical は 1 位 9 行＋2 位 9 行で 18/28．
**N6 の education／medical の明暗は，rank_1 決定ロジックの副作用ではなく，同一ヘッドの
当該行に対する順位付け能力そのものの差である**．

得点の実測がこれを裏づける（当該ドメインが期待に含まれる複合行のみ，`head_scores` 生値）:

| ドメイン | 自ドメイン得点の中央値 | 行内最大得点の中央値 | 自得点／行内最大 の中央値 |
|---|---|---|---|
| education | **0.0657** | 0.9900 | **0.0899** |
| medical | 0.7364 | 0.9819 | 0.7496 |

education 絡みの複合行において，ヘッドは education にほぼ信号を出していない（行内勝者の 1 割未満）．
medical は勝者の 0.75 倍の得点を出しており，2 枠に入る競争力がある．

**2. Iter61→Iter63 で失われた education 2 行の直接確認（該当 20 行を全件突き合わせた結果）**

education 被覆の遷移（基準線, Iter61, Iter63）の内訳は
`(F,F,F)=11 行`・`(T,T,T)=3 行`・`(T,T,F)=2 行`・`(T,F,F)=4 行` で，**本レバーで新たに失われたのは
2 行のみ**（残る 4 行は Iter60/61 の rank_2 変更時点で既に失われていた）．その 2 行:

| 行 | expected | 基準線 rank_1/rank_2 | Iter61 | Iter63 | Iter63 の head 上位 |
|---|---|---|---|---|---|
| compound-006 | education+legal | **education**/business_economics | **education**/general | general/legal | general 0.9409, legal 0.8461, medical 0.6281, **education 0.2725（4 位）** |
| compound-078 | mathematics+education | **education**/medical | **education**/medical | medical/social_science | medical 0.9996, social_science 0.9804, general 0.9759, **education 0.8531（4 位）** |

**2 行とも「基準線の実行時ルータが rank_1 に education を選んでいたおかげで被覆されていた」行**であり，
rank_1 をヘッド argmax に渡した瞬間に失われた．education が rank_2 経由で失われた行は 0 である
（1 の表のとおり education はヘッド 2 位に 1 行も来ていない）．
逆に medical の改善 6 行（対 Iter61，悪化 0）は compound-004・008・009・025・057・071 で，
いずれも **基準線ルータが business_economics を rank_1 に選んでいた行**をヘッドが legal/medical/
natural_science 等へ置き換え，その結果 medical がヘッド 2 位（実測順位 2 位）として 2 枠目に
滑り込んだものである．

**3. S5 の +20 と N6 の education 退行が両立する理由（見かけの矛盾の解消）**

基準線の実行時ルータは複合 100 行の rank_1 を **business_economics 44 行・education 13 行**に
極端に偏らせていた（ヘッド argmax は最大でも natural_science 19 行で 10 ドメインに分散）．
全 1600 行での dispatch 量・適合率の実測が偏りの性質を示す:

| ドメイン | 基準線 dispatch／適合率 | Iter61 | Iter63 |
|---|---|---|---|
| education | **664 行**／0.173 | 373／0.268 | 322／0.295 |
| medical | 310／0.358 | 349／0.318 | 357／0.322 |
| legal | 190／0.637 | 232／0.591 | 243／0.576 |

基準線は education を全 1600 行の **41.5%** に投げており（真の出現は 170 行），適合率 0.173 の
**過剰出力によって被覆を稼いでいた**．S5 の discordant の内訳もこれと整合する:
新のみ正解 27 行の新 rank_1 は legal 7・computer_science 5・social_science 5・medical 3…と
基準線が過小に選んでいたドメインに集中し，**基準線のみ正解 7 行の基準線 rank_1 は
business_economics 4・education 2・history_culture 1** と，基準線が過剰に選んでいたドメインに集中する．
すなわち S5 の +20（41→61，95% CI [+9.3pt, +30.7pt]）は
**「事前分布に張り付いた過剰出力から，識別に基づく分散した出力への再配分」**であり，
education は構造的にその再配分の負け側に回る．S5 と N6 は矛盾しておらず，同一機序の表裏である．

**4. Iter60〜62 の「rank_2 側の改善が education/medical を犠牲にする」パターンと同型か（判定）**

**損失経路は新規（rank_1 固有）だが，根本原因は同一**と判定する．

- **新規である点**: Iter60〜62 の education 退行（9→4→5→6）は，ヘッド内の rank_2 競争で
  education が他ドメインに押し出された結果だった．今回失われた 2 行はどちらも rank_2 競争と無関係で，
  **基準線ルータが持っていた rank_1 枠（＝ヘッドとは別系統の分類器）の喪失**による．
  複合 100 行で基準線ルータのみ正解 7・ヘッドのみ正解 27・両方正解 34 であり，
  **2 系統の和集合は 68/100 だがヘッド単独では 61/100**．本レバーは
  rank_1（実行時ルータ）と rank_2（ヘッド）という異種 2 分類器のアンサンブルを
  **単一分類器の top-2 へ畳み込む**変更でもあり，多様性由来の 7 行を捨てている．
  その 7 行のうち 2 行が education であった．
- **同一である点**: いずれの場合も education の被覆は「識別信号」ではなく
  「どこかのコンポーネントの過剰出力」に支えられており，選択をヘッド主導＝適合率志向に
  寄せる変更は必ず education を削る．education 被覆は dispatch 量と単調に対応している
  （664→9/20，373→5/20，322→3/20）．Iter62 の学び 4「education の rank_2 出力を増やしても
  被覆は増えない（rank_2 適合率 7.6%）」と，本イテレーションの「education はヘッド 2 位にすら
  1 行も来ない」は同じ事実の別断面である．
- **構造的な帰結（次レバー設計への含意）**: 現行ヘッドの下では，**rank_2 の選び方をどう変えても
  education 被覆は 3/20 から動かせない**（2 位が 0 行のため）．ヘッド内順位 3 位以内まで広げても
  5/20 にしかならず，コストを 2→3 に増やしても N6 フロア 6/20 に届かない（実測）．
  education の改善余地は順位付け層ではなく，**表現（埋め込み）または訓練データ側**にしか残っていない．

**5. ノイズか有意かの判定**

対応あり exact McNemar（同一 20/28 行の固定集合上）:

| 比較 | education | medical |
|---|---|---|
| vs 基準線 | 悪化 6／改善 0，**p=0.0312** | 悪化 1／改善 6，p=0.1250 |
| vs Iter61（同一ヘッド＝直接比較） | 悪化 2／改善 0，p=0.5000 | 悪化 0／改善 6，**p=0.0312** |
| vs Iter62（N6 のフロア基準） | 悪化 3／改善 0，p=0.2500 | 悪化 0／改善 7，**p=0.0156** |

- **medical の改善は有意**（vs Iter61 p=0.0312，vs Iter62 p=0.0156）．方向も一貫しており信号である．
- **education の 3/20 は，N6 のフロア（Iter62 の 6/20）に対しては有意差なし**（p=0.25，discordant 3 件）．
  head ベース 4 反復の値は 4・5・6・3（標本 sd≈1.29）で，3/20 はこのばらつきの下端に位置し，
  **「Iter62 比でさらに退行した」と統計的に主張することはできない**．
  一方 **基準線比 9/20→3/20 は悪化 6／改善 0 で p=0.0312 と有意**であり，
  **R-C（education が基準線を下回る）は本イテレーションで初めて統計的に裏づけられた**．
  対外記述ではこの 2 つを混同しないこと（N6 の不成立はあくまで**事前登録した閾値に対する機械的な未達**であって，
  「Iter62 比の有意な退行」ではない）．
- 主要指標側は明確に信号である．S1 は改善 43／悪化 14（p=1.5e-4，Δ=+14.5pt，95% CI [+7.4pt, +21.6pt]）で，
  Iter61/62（改善 32〜33／悪化 12〜13，p≈0.004）を discordant の絶対数でも上回る．
  **legal 絡み 60 ペアを除いた部分集合（n=140）でも +12.9pt・改善 29／悪化 11・p=0.006427** であり，
  Iter61 で残っていた R-B（legal 絡みを除くと p=0.0501 で有意水準に届かない）は
  **本イテレーションで解消した**．これは本レバーの効果が legal 依存でないことの直接証拠である．

**6. 仮説との整合**

計画節の仮説「rank_1 が複合行で 41/100 しか当たらず recall の上限を 0.705 に固定していることが
残るボトルネックであり，rank_1 をヘッド argmax にすれば 0.445 を超える」は**支持された**．
rank_1 正解は 41→61 に上がり，rank_1 を固定したときの recall 上限は 0.705→**0.805** へ拡大し，
実測 recall も 0.445→0.490 となった．想定外の挙動（発散・言語崩れ・OOM・アサーション違反）はない．

ただし**仮説が想定していなかった副作用が 2 点ある**．
(a) 上記 4 のアンサンブル多様性の喪失（基準線ルータのみ正解 7 行の放棄）．
(b) **rank_1 正解が +20 行増えたのに，2 ドメインとも被覆した行は 12→17 の +5 行にとどまる**
（被覆 1 個の行は 65→64 でほぼ横ばい，被覆 0 個の行が 23→19）．rank_1 が正解した行のうち
rank_2 も正解した割合は Iter61 の 12/41=29.3% から Iter63 の 17/61=27.9% へ**変わっていない**．
すなわち **rank_1 の枠は改善したが，「rank_1 が当たった行で 2 つ目を当てる」能力は Iter62 で
確認された飽和のまま**であり，新しい上限 0.805 に対して実測 0.490 と 0.315 の乖離が残っている．
次のボトルネックはここに移った（ただし Iter62 が示したとおり，これはスコア変換では動かない）．

**7. 事前登録の判定規則との機械的な対応（最終確定は次フェーズ rc-reflector）**

計画節「判定規則（事前登録）」に照らすと:

- 第 1 分岐（S1〜S5 全充足**かつ** N2'/N3/N5/N6 全充足 → adopted）: **該当しない**（N6 が不成立）．
- 第 4 分岐（S2 不成立＝recall<0.445 → rejected）: 該当しない（0.490 ≧ 0.465）．
- 第 5 分岐（S4 不成立＝no-op → rejected）: 該当しない（`rank1_change_count=380`）．
- 第 2・第 3 分岐（S2 のみ／S5 のみの不成立に基づく partial）: いずれも前提が成立しない（S2・S5 とも充足）．
- **最終分岐「非退行 N2'/N3/N5/N6 のいずれかが FAIL → adopted にはしない（S 側が全充足でも最大 partial とし，
  どの指標が退行したかを対外記述の留保として残す）」に該当する**．

したがって機械的な結論は **partial（S1〜S5 全充足・非退行 4 件中 N6 のみ不成立により adopted 不可）**である．
事後の閾値緩和・厳格化は行っていない．Iter62 の partial（実質「効果なし」）とは中身が大きく異なり，
**今回は主基準側に有意かつ大きな効果があるうえで非退行 1 件が未達という型の partial** である点を，
reflector は区別して記録すべきである．

**8. 対外記述で必ず併記すべき留保**

- **R-C（更新・格上げ）**: education 自己被覆は基準線 9/20 → **3/20** で，
  対応あり exact McNemar p=0.0312 の**有意な退行**．本イテレーションで初めて有意になった．
  「複合設問の被覆が全体で改善した」と書く場合，**education 単独では基準線より有意に悪化している**ことを
  必ず併記する．同時に「Iter62 比の追加退行（6→3）は有意ではない（p=0.25）」ことも併記し，
  過大に書かないこと．
- **R-E（新規・アンサンブル多様性の喪失）**: 本構成は rank_1・rank_2 の双方を単一のオフラインヘッドが
  決めるため，実行時ルータが持っていた独立な正解 7 行（複合 100 行中）を捨てている．
  和集合の上限 68/100 に対しヘッド単独 61/100．「ヘッドは実行時ルータより優れている」ではなく
  「ヘッドは実行時ルータより優れているが，両者は部分的に相補的である」が正確な記述である．
- **R-F（実機未検証・B94/B95 の再掲，本イテレーションで最も重い）**: 本構成は rank_1 まで実行時経路
  （`node.py` のルータ）から乖離させる．配線は行っていないため，**オフラインの +14.5pt は実機性能の
  主張には一切使えない**．`selected_domain` の新 rank_1 での上書きも「ヘッド argmax が top-1 として
  dispatch される」という仮定下のオフライン上の値であり，実機の
  `aggregator.select_best_dispatch_response()` を再現しない（N2'=0.60375 の解釈はこの範囲に留まる）．
- **効果量の扱い**: 0.345→0.490（+14.5pt，exact McNemar p=1.5e-4，95% CI [+7.4pt, +21.6pt]，
  mean_dispatch 2.000000 でコスト中立）．legal 絡みを除く部分集合でも +12.9pt・p=0.006427 で
  方向・有意性とも一貫する（R-B は解消）．なお Iter60 の 0.480 は R-A（テスト集合のペア分布を
  参照した配分）を含む値であり，**リーク非依存の構成で 0.48 を超えたのは本イテレーションが初めて**だが，
  0.490 と 0.480 の差自体は検定していない．
- **既知の穴（継続）**: 低品質行（プロンプトの echo）の混入は Iter60 以来未解消．
  目的指標の n が 20/28 と小さく ±2 件程度の真の効果を検出する統計的検出力がないという
  測定基盤の制約も Iter62 から継続している（今回 education の N6 判定がまさにこの制約下にある）．

### 考察 (Iter63)

**判定: partial（部分的成立．ただし Iter62 の partial とは型が異なり，主基準側に大きく有意な効果がある）**

事前登録の判定規則（本イテレーション計画節）への機械的な照合は以下のとおりで，事後の閾値変更は
一切行っていない（Iter29 以降の事前登録運用）．

- S1 p=0.0001538890224443423 / S2 recall 0.490（≧0.465）/ S3 mean_dispatch=2.000000 /
  S4 rank1_change_count=380 / S5 複合 100 行 41→61，exact p=0.0008213953115046024（方向も新側）
  → **S1〜S5 全充足**．
- N2' 0.60375（≧0.5875）/ N3 17/30（≧8/30）/ N5 0.6033333（≧0.590）は充足，
  **N6 のみ不成立**（education 3/20 < フロア 6/20．medical 18/28 は充足だが AND 条件のため全体で不成立）．
- 第 1 分岐（adopted）は N6 不成立により該当せず，第 2・第 3 分岐は前提（S2 または S5 の不成立）が
  成立せず，第 4・第 5 分岐（rejected / no-op）も該当しない．
  **最終分岐「非退行のいずれかが FAIL → adopted にはしない（最大 partial）」に該当する**．

レバー `rank1_source` は config.yml で values 単一値（`multilabel_head_argmax`）のため
**試し切り＝クローズ**する．

**このイテレーションで確定した学び**

1. **rank_1 がボトルネックであるという仮説は支持された**．基準線ルータの top-1 は複合 100 行で
   41/100 しか正解せず recall 上限を 0.705 に固定していたが，ヘッド argmax に替えると 61/100・
   上限 0.805 となり，実測 recall も 0.345→0.490（+14.5pt，95% CI [+7.4pt, +21.6pt]，コスト中立）．
   **legal 絡み 60 ペアを除く n=140 でも +12.9pt・p=0.006427 で有意**となり，Iter61 まで残っていた
   留保 R-B（効果の legal 依存）は本イテレーションで解消した．
2. **しかしボトルネックは消えたのではなく移動した**．rank_1 正解が +20 行増えたのに 2 ドメインとも
   被覆した行は 12→17（+5）にとどまり，「rank_1 が当たった行で 2 つ目も当てる」割合は
   29.3%→27.9% と**不変**である．新上限 0.805 と実測 0.490 の乖離 0.315 がこれに対応する．
   Iter62 が示したとおり，この残差はスコア変換（較正）では動かない．
3. **education の被覆は識別信号ではなく過剰出力に支えられていた（Iter60〜62 と同一の根本原因）**．
   基準線は education を全 1600 行の 41.5%（664 行，適合率 0.173）へ投げており，今回失われた 2 行
   （compound-006・078）はいずれも「基準線ルータが rank_1 に education を選んでいたから被覆されていた」
   行であった．rank_2 競争由来の損失は 0 件である．ヘッドは education 絡みの複合行で自ドメイン得点の
   中央値が行内最大の 0.0899 倍しかなく，**20 行中 1 行もヘッド 2 位に来ない**．したがって
   **順位付け層（rank_1/rank_2 の選び方・スコア変換）をどう変えても education 被覆は 3/20 から
   動かせない**（3 位まで広げても 5/20 で，コストを 2→3 に増やしてもフロア 6/20 に届かないことを実測）．
   改善余地は**表現（埋め込み）または訓練データ側にしか残っていない**．これが次レバー選定の根拠である．
4. **S5 の改善と education の退行は同一機序の表裏であり矛盾しない**．S5 の discordant は，新側のみ
   正解 27 行が legal/computer_science/social_science 等（基準線が過小に選んでいたドメイン）に，
   基準線側のみ正解 7 行が business_economics/education（基準線が過剰に選んでいたドメイン）に集中する．
   本レバーは「事前分布に張り付いた過剰出力から識別ベースの分散出力への再配分」であり，
   education は構造的に負け側に回る．
5. **新しい留保 R-E（アンサンブル多様性の喪失）**: rank_1・rank_2 の双方を単一ヘッドが決める構成は，
   実行時ルータだけが正解していた 7 行（複合 100 行中）を捨てている（和集合 68/100 vs ヘッド単独 61/100）．
   「ヘッドは実行時ルータより優れている」ではなく「優れているが両者は部分的に相補的」が正確な記述である．
   ただし「両系統の和集合を dispatch する」構成は事実上 Iter61（rank_1=ルータ，rank_2=ヘッド argmax）
   そのものであり，recall 0.445 と本構成の 0.490 に劣ることが実測済みである点に注意する
   （多様性の利得は rank_1 の正解数には効くが，2 枠という予算の下では回収できていない）．
6. **統計上の区別を混同しないこと**: education 3/20 は，**基準線 9/20 比では悪化 6／改善 0 で
   p=0.0312 の有意な退行**（留保 R-C が本イテレーションで初めて統計的に裏づけられた）だが，
   **N6 のフロアである Iter62 の 6/20 比では p=0.25 で有意差なし**（head ベース 4 反復の値は
   4・5・6・3 でばらつきの下端）．N6 の不成立は「事前登録した閾値に対する機械的な未達」であって，
   「Iter62 比の有意な追加退行」ではない．

**対外記述の方針（本フェーズの決定）**

- 効果量の正式値を **0.345 → 0.490（Δ=+14.5pt，n=200 のドメイン対 exact McNemar p=1.5e-4，
  95% CI [+7.4pt, +21.6pt]，mean_dispatch=2.000000 でコスト中立）** へ更新する．ただし
  **partial である旨と，これが `rank_1` まで実行時ルータから乖離させたオフライン構成の値である旨
  （R-F）を必ず併記する**．B95 の正式値（Iter61 の +10.0pt）は「rank_1 を実行時ルータのまま保った
  構成での値」として併記し，削除しない（配線可否の人間判断が未決であるため，両構成の値が要る）．
- 留保 **R-B は解消**，**R-C は「基準線比で有意な退行」へ格上げ**，**R-E を新規追加**，
  **R-F（実機未検証）は継続**する．低品質行の混入（Iter60 以来）も未解消のまま継続する．

**本番経路への配線**

**今回も配線しない（4 回目）．かつ，現時点では配線を推奨しない**．理由は 2 つある．
(a) 配線は `config.yaml` のスキーマ変更（＋`node.py:214`・`run_experiment.py:93` の同時変更）を伴い，
rc-reflector の可逆な自律判断の範囲外である（B94/B95/B96 と同一論点．論点は B95 要レビュー 1 に一本化）．
(b) 本イテレーションで R-C が基準線比 p=0.0312 の有意な退行として確定したため，配線は
「全体 +14.5pt と引き換えに education 単独の被覆を基準線より有意に落とす」ことを本番で確定させる
選択になる．この可否は研究の結論に関わる不可逆な判断であり，人間の判断を要する．

**次の一手**

既存 levers は実質試し切り済みであるため skill の停止条件 1 に従い新レバーを考案し，config.yml の
`levers` 末尾へ追記した（backlog B97）．
**`multilabel_synthetic_volume = uniform_nine_per_pair`**（2 ドメイン合成訓練事例を全 45 ペア一律で
3 件／ペア → 9 件／ペアへ増やす．`scripts/generate_multidomain_training_examples.py --per-pair 9
--per-pair-legal 9`）．上記の学び 3（education の改善余地は訓練データ／表現側にしかない）と
学び 2（残差は「2 つ目を当てる」能力にあり，スコア変換では動かない）の両方に同時に効きうる唯一の
低コスト・非リークな軸であるため．配分は一律であり「どのドメインを増やすか」の決定が入らないので
R-A 型リークは構造的に生じない．

### journal ローテーション

本フェーズ実行前の `## Iteration ` 見出しは 4 件（Iter63/62/61/60）で，実イテレーション数と一致
（見出しの欠落なし）．`journal_retention: 3` に従い `rotate_journal.sh` を実行し，Iteration 60 の
ブロックを `journal_archive.md` へ移した．

本イテレーションのコミット: **c1d1116**（`scripts/evaluate_dispatch_candidate_ranking.py`・
`tests/test_evaluate_dispatch_candidate_ranking.py`・`results/iter63_multilabel_ranking_predictions.jsonl`・
`results/iter63_stats.json`・`.claude/research/` 配下 5 ファイル）．作業ツリーに残る
`config.yaml`（`embed_node_host` 変更）と `results/iter45_preliminary/logs/wafl*/expert-mesh.log`
は本イテレーションと無関係のため触れず，コミットに含めていない．

## Iteration 62: 多ラベルヘッド得点のドメイン別較正による rank_2 偏りの是正

### 調査 (Iter62)

**問い**（config.yml が既に詳細な設計を事前登録しているため，先行研究の新規調査ではなく，
一次情報＝コード・実データの確認を優先した．該当節: config.yml:926-965，backlog B95）

- Q1: `scripts/train_multilabel_dispatch_head.py` の現状構造上，`CalibratedClassifierCV` を
  `OneVsRestClassifier` の各ドメインの内部推定器へどう組み込めるか．held-out 分割は訓練データ内で
  完結するか．
- Q2: `scripts/evaluate_dispatch_candidate_ranking.py` の rank_2 選択ロジックは現状どのフィールド・
  関数を使っており，較正後スコアへの切替箇所はどこか．
- Q3: 較正の fit に訓練データのみを使い評価集合の情報を混入させない実装方針は，実データで安全と
  確認できるか（Iter60 の R-A 型リークの再発条件との照合）．
- Q4: Iter61 の資産（合成訓練データ・ヘッド・埋め込みキャッシュ・Iter60 予測 JSONL）は実在し，
  今回もそのまま再利用可能か．
- Q5: 較正対象ドメイン（全 10 ドメイン一律）の各 `n_positive` は，held-out 分割で極端に少数な
  ドメインが出ない水準か．

**分かったこと（全文読了・実行確認による一次情報）**

1. **`scripts/train_multilabel_dispatch_head.py`**（278行，全文読了）: `train_multilabel_ranking_head()`
   （`:145-157`）が唯一の学習箇所で，`model = OneVsRestClassifier(LogisticRegression(max_iter=1000,
   class_weight="balanced")); model.fit(embeddings, Y)` という 2 行のみからなる．`OneVsRestClassifier`
   は多ラベル `Y`（`build_multilabel_targets():87-101` が `MultiLabelBinarizer` で作る 0/1 行列）の
   各列（＝各ドメイン）ごとに base estimator を **独立に clone・fit** する構造なので，base estimator を
   `CalibratedClassifierCV(LogisticRegression(max_iter=1000, class_weight="balanced"),
   method="sigmoid", cv=5)` に差し替えるだけで，較正の held-out 分割は各ドメインの二値問題ごとに
   **`_train_and_save()`（`:191-229`）へ渡された訓練データ（`single_label_rows + synthetic_rows`，
   評価集合を一切含まない）の内部だけで完結する**．CLI 引数・保存形式（`joblib.dump({"model":...,
   "classes":...}, ...)`，`:223`）は無変更で使える．
2. **実機で `CalibratedClassifierCV` を `OneVsRestClassifier` に組み込んで実行確認した**（`uv run
   python3` で sklearn 1.9.0 上に小規模ダミーデータで再現）: `hasattr(CalibratedClassifierCV,
   "decision_function")` は **`False`**（`predict_proba` のみ実装）．そのため
   `OneVsRestClassifier(CalibratedClassifierCV(...)).decision_function(...)` を呼ぶと
   **`AttributeError: This 'OneVsRestClassifier' has no attribute 'decision_function'`** が実際に
   発生することを確認した（`OneVsRestClassifier` は全内部推定器が `decision_function` を持つ場合のみ
   委譲する実装のため）．一方 `predict_proba()` は正常に動作し，多ラベル OvR のため各ドメインの
   確率はドメイン間で正規化されない（実測 `sum per row = 2.906...`，1 にならない．
   `evaluate_dispatch_candidate_ranking.py` のモジュール docstring `:20-30` が既存の未較正ヘッドに
   ついて記述している性質と同じ）．
3. **`scripts/evaluate_dispatch_candidate_ranking.py`**（470行，全文読了）: rank_2 は
   `_head_scores()`（`:166-178`）が `model.decision_function([embedding])[0]` に手動 `_sigmoid()`
   （`:102-104`）を適用してスコア化し，`build_new_rows()`（`:181-214`）の `rank2_new = max(...)`
   （`:199-202`）がそのスコアで rank_1 以外の最大ドメインを選ぶ，という構造である．
   `CalibratedClassifierCV` を導入すると上記 2 の理由で `decision_function()` 自体が呼び出せなく
   なるため，**`_head_scores()` を `model.predict_proba([embedding])[0]` へ切り替える必要がある**
   （config.yml が事前登録した「rank_2 のスコア源を較正後に切り替える」の具体的な実現箇所はここ 1 関数）．
4. **切替の安全性を数値的に検証した**: 未較正の `OneVsRestClassifier(LogisticRegression)` では
   `sigmoid(decision_function(x))` と `predict_proba(x)` が**完全に一致する**（実機再現，
   `np.max(np.abs(sig - p)) == 0.0`，多クラス単一ラベル用途で行われる正規化は多ラベル OvR には
   適用されないため）．したがって `_head_scores()` を `predict_proba()` 方式へ統一する変更は，
   Iter59（`models/dispatch_candidate_ranking_head.joblib`）・Iter60/61（MLB ベースの未較正ヘッド）の
   **既存ヘッドに対してはビット単位の no-op**であり，過去 3 イテレーションの結果の再現性を壊さない．
   これにより「ヘッドの種類で分岐する」実装ではなく，スコア源を一律 `predict_proba()` に統一する
   単純な 1 箇所変更で済み，較正の効果だけを単離できる．
5. **`n_positive` 分布を実データで確認した**（`data/classifier_train.jsonl` 1427行 ＋
   `data/classifier_train_multidomain_iter61.jsonl` 135行の列和）: legal のみ 104（単一 77＋合成 27），
   他 9 ドメインは一律 177（単一 150＋合成 27）．合成 135 行は 45 ペア×3 件均一（Iter61 で確定済み）の
   ため 10 ドメイン全てに合成 27 件ずつが均等配分されている．全ドメインとも `CalibratedClassifierCV`
   の既定 `cv=5`（`StratifiedKFold`）に対し 1 フォールあたり 20 件超の正例が確保でき，極端に少数な
   ドメインは存在しない（config.yml note の「legal 104〜他 150+27」という記述と一致）．
6. **Iter61 資産の実在・再利用可能性を確認した**（`ls -la`）: `data/classifier_train_multidomain_
   iter61.jsonl`（41,551B）・`models/dispatch_multilabel_head_iter61.joblib`（66,134B）・
   `results/iter59_query_embeddings.npz`（9,971,712B）・`results/iter60_multilabel_ranking_
   predictions.jsonl`（981,340B）・`results/iter61_multilabel_ranking_predictions.jsonl`（982,193B）・
   `results/20260918_202613/results.jsonl`（3,510,699B，基準線）が全て実在する．訓練データ・埋め込み
   キャッシュ・基準線は今回そのまま再利用でき，較正の対照（未較正）として Iter61 のヘッド・予測も
   保持されているため，較正前後の直接比較（A5 相当）が可能である．
7. `_load_head()`（`:150-163`）は既に dict payload（`{"model":..., "classes":...}`）とベア推定器の
   両方に対応済みで，`CalibratedClassifierCV` をラップした `OneVsRestClassifier` を保存しても
   ロード側の変更は不要である．`_assert_head_scores_are_domain_names()`（A6，`:260-282`）も
   `head_scores` の**キー**（`classes` リスト由来）のみを検査するため，較正導入後も無変更で使える．
8. `_print_per_domain_cv_diagnostics()`（`:160-188`）は較正とは別に，未較正の診断用モデルで
   `cross_val_predict(method="decision_function")` を呼ぶ独立のコードパスであり，較正導入後も
   `decision_function` を要求され続けるが，こちらは較正対象の本番モデルとは別インスタンスのため
   影響を受けない（変更不要）．

**結論**

config.yml が事前登録した変更範囲（`train_multilabel_dispatch_head.py`・
`evaluate_dispatch_candidate_ranking.py` の 2 箇所のみ）は，実際にそれぞれ 1 箇所ずつの変更
（base estimator の差し替え／スコア取得関数の呼び先変更）で実現できることをコードと実行確認で
検証した．**新たなコード基盤の追加実装は不要**であり，計画フェーズは `cv` の値（`_DIAGNOSTIC_CV=5`
との統一を推奨）・`method`（`sigmoid` 固定，config.yml 指示どおり）・`ensemble`（sklearn 1.9.0 の
既定値 `"auto"` をそのまま使うか明示するか）を確定し，実装フェーズへ直行できる状態にある．

**次フェーズへの示唆**

- レバー名は config.yml の指示どおり `multilabel_rank2_score_calibration`（値
  `per_domain_holdout_calibration`）で確定してよい．アルゴリズム上の選択肢は
  `CalibratedClassifierCV(LogisticRegression(max_iter=1000, class_weight="balanced"),
  method="sigmoid", cv=5)` の一択（`OneVsRestClassifier` 経由で自動的に全 10 ドメイン一律適用され，
  「どのドメインを較正するか」という別の設計判断＝R-A 型リーク再発の入り口が構造的に生じない）．
- `evaluate_dispatch_candidate_ranking.py:_head_scores()` の `predict_proba()` への切替は，
  未較正ヘッド（Iter59/60/61）に対して数値的に no-op であることを検証済みなので，計画フェーズでは
  「較正あり／なし」の A/B を同一スクリプトの同一コードパスで比較でき，スコア取得関数自体の変更が
  交絡にならないことを事前登録に明記するとよい．
- 出力先の命名は Iter60/61 の慣例（`_iter62` サフィックス）を踏襲し，Iter61 側の資産
  （`data/classifier_train_multidomain_iter61.jsonl`・`models/dispatch_multilabel_head_iter61.joblib`・
  `results/iter61_multilabel_ranking_predictions.jsonl`）を上書きしないこと．較正前後の直接比較
  （A5 相当の「対 Iter61 不一致」）には `results/iter61_multilabel_ranking_predictions.jsonl` を
  そのまま渡せる．
- 較正の fit は `_train_and_save()` に渡す訓練データ（1427＋135＝1562行）のみで完結し，評価集合
  （1600問・`_COMPOUND_QUESTIONS`）は一切参照しない設計が実データ・コードの両面で保証できている．
- S1〜S5（暫定，config.yml:950-955）の事前登録・確定は計画フェーズの役割．特に S5（education 9/20
  以上・medical 13/28 以上への回復）は較正が「rank_2 分布の平坦化そのものを弱める」方向に働くかを
  直接見る指標であり，A5（発火の証拠）・N1〜N5（非退行）と併せて確定すること．

### 計画 (Iter62)

**仮説**

Iter60/61 で observed した「rank_2 分布の平坦化に伴う education（9/20→4/20→5/20）・medical
（13/28→12/28）の被覆退行」の原因が，**OvR ヘッドの 10 個の二値問題が独立に学習された結果，
各ドメインの sigmoid 素点がドメイン間で較正されておらず，rank_2 のドメイン間比較（`max` 選択）が
不公平な尺度で行われている**ことにあるなら，訓練データ内の held-out 分割で fit したドメイン別較正を
掛けてから rank_2 を選べば，compound_domain_set_recall 全体（0.445）を落とさずに
education/medical の被覆が回復するはずである．逆に回復しないなら，退行は「素点の較正ずれ」ではなく
埋め込み表現または訓練データの正例分布そのものに起因することになる．

**単一レバー（今回変更する唯一の変数）**

`multilabel_rank2_score_calibration`: `none`（Iter59〜61 の実質値＝未較正の
`OneVsRestClassifier(LogisticRegression(...))` 素点）→ **`per_domain_holdout_calibration`**
（base estimator を `CalibratedClassifierCV(LogisticRegression(max_iter=_MAX_ITER,
class_weight="balanced"), method="sigmoid", cv=5, ensemble=True)` に差し替え，
`OneVsRestClassifier` 経由で全 10 ドメインへ一律適用）．

**確定した実装仕様（本フェーズの決定事項 1・2）**

1. **較正器の引数を確定する**（`scripts/train_multilabel_dispatch_head.py` の
   `train_multilabel_ranking_head()`，`:145-157` の 1 箇所のみ変更）．
   - `method="sigmoid"`: config.yml 事前登録どおり固定（Platt scaling）．sklearn 1.9.0 の既定値と
     同一だが，レバーの本体であるため明示する．
   - `cv=5`: 同ファイルの診断用定数 `_DIAGNOSTIC_CV=5` と数値を揃える．較正の内部分割は
     各ドメインの二値問題に対する `StratifiedKFold(n_splits=5, shuffle=False)` となり，
     最小の legal でも n_positive=104（1 フォールあたり約 20 正例）で十分．**実装時は
     `_DIAGNOSTIC_CV` を流用せず較正専用の新定数（例 `_CALIBRATION_CV = 5`）を置く**
     （診断用 CV と較正用 CV は責務が異なり，将来一方だけ動かせるようにするため）．
   - `ensemble=True`: sklearn 1.9.0 の既定 `"auto"` は非 frozen 推定器に対して `True` に解決される
     （本フェーズで実機確認: `ensemble=True` と `ensemble="auto"` の `predict_proba` の
     max abs diff = 0.0）．将来の既定値変更に左右されないよう**明示指定**する．
   - `n_jobs`: 指定しない（既定 `None`＝逐次）．`OneVsRestClassifier` との二重並列化を避け，
     実行順序に依存する非決定性を持ち込まないため．
   - **決定性**: `shuffle=False` の `StratifiedKFold` と `LogisticRegression`（lbfgs，決定的）の
     組み合わせのため，同一入力に対する再実行で `predict_proba` が完全一致することを実機確認済み
     （max abs diff = 0.0）．`random_state` の追加は不要．
   - 保存形式（`joblib.dump({"model":..., "classes":...})`，`:223`）・CLI 引数・`_assert_a0_...`・
     `_print_per_domain_cv_diagnostics()`（未較正の別インスタンスを使う独立経路）は**無変更**．
2. **スコア源の切替を確定する**（`scripts/evaluate_dispatch_candidate_ranking.py` の
   `_head_scores()`，`:166-178` の 1 箇所のみ変更）．
   - `logits = model.decision_function([embedding])[0]; probabilities = _sigmoid(...)` を
     **`probabilities = np.asarray(model.predict_proba([embedding])[0])`** に置き換える．
     理由は `CalibratedClassifierCV` が `decision_function` を実装しておらず，
     `OneVsRestClassifier.decision_function` が `AttributeError` になるため（調査で実機再現済み）．
   - **この切替は未較正ヘッド（Iter59/60/61）に対して数値的に no-op**（多ラベル OvR では
     `predict_proba` の行方向正規化が行われないため `sigmoid(decision_function(x))` と完全一致，
     実測 max abs diff = 0.0）．したがって**ヘッド種別による分岐を書かず，一律
     `predict_proba()` に統一する**．これにより「スコア取得関数の変更」自体が較正効果の交絡に
     ならないことを保証する（下記 A8 で実証する）．
   - `build_new_rows()` の `rank2_new = max(...)`（`:199-202`）・`_load_head()`（`:150-163`）・
     A6 の `_assert_head_scores_are_domain_names()` は**無変更**．rank_1 は基準線 JSONL から
     コピーされるのみでヘッドに一切依存しないため，rank_1 経路への影響は構造的にない．
   - `_sigmoid()`（`:102-104`）は他に呼び出し元がなくなるが，**削除しない**（無関係な差分を
     増やさないため．実装フェーズで未使用となる場合はその旨だけ報告する）．

**固定する構成（Iter61 から一切変えない）**

- 訓練データ: `data/classifier_train.jsonl`（1427 行）＋
  `data/classifier_train_multidomain_iter61.jsonl`（135 行，45 ペア×3 件均一）．
  **合成データの再生成は行わない**（生成プロンプト・F1〜F4・temperature・生成モデルは
  今回の実行経路に一切登場しない）．
- ヘッド構造の他の全要素（`OneVsRestClassifier`＋`LogisticRegression(max_iter=1000,
  class_weight="balanced")`，`MultiLabelBinarizer` による Y 構築），埋め込みモデル
  `nomic-embed-text`，埋め込みキャッシュ `results/iter59_query_embeddings.npz`．
- rank_1 経路（基準線 JSONL からのコピー），基準線
  `results/20260918_202613/results.jsonl`（compound_domain_set_recall=0.345）．
- 採点スクリプトの他の全ロジック・統計スクリプト `compute_iter59_ranking_stats.py`（**無改造**）．
- `config.yaml` は無変更．**実行時経路への配線は本イテレーションでも行わない**
  （スキーマ変更を伴うためユーザー確認が必要．backlog B94/B95）．

**出力ファイル命名（Iter61 の成果物を上書きしないこと）**

| 種別 | Iter61（保護・読み取り専用） | Iter62（新規作成） |
|---|---|---|
| 訓練データ | `data/classifier_train_multidomain_iter61.jsonl` | （再利用．新規作成なし） |
| ヘッド | `models/dispatch_multilabel_head_iter61.joblib` | `models/dispatch_multilabel_head_iter62.joblib` |
| 予測 | `results/iter61_multilabel_ranking_predictions.jsonl` | `results/iter62_multilabel_ranking_predictions.jsonl` |
| 統計 | `results/iter61_stats.json` | `results/iter62_stats.json` |
| no-op 検証（A8） | — | `results/iter62_noop_check_iter61head.jsonl` |

**実行コマンド（`--ollama-host` は疎通する方を使う．Iter61 実績は訓練時 `192.168.15.100:11434`，
採点時 SSH ローカルフォワード `127.0.0.1:11435`）**

```
# 0) A8 事前チェック（スコア源切替が未較正ヘッドに対して no-op であることの実証）
#    _head_scores() の predict_proba 化を適用した採点スクリプトで Iter61 のヘッドを再採点し，
#    results/iter61_multilabel_ranking_predictions.jsonl と完全一致することを確認する．
uv run python -m scripts.evaluate_dispatch_candidate_ranking \
    --baseline results/20260918_202613/results.jsonl \
    --head models/dispatch_multilabel_head_iter61.joblib \
    --embedding-model nomic-embed-text --ollama-host 127.0.0.1 --ollama-port 11435 \
    --embedding-cache results/iter59_query_embeddings.npz \
    --iter59-predictions results/iter60_multilabel_ranking_predictions.jsonl \
    --output results/iter62_noop_check_iter61head.jsonl
diff <(jq -cS '{id,dispatched_domains,rank2_new}' results/iter61_multilabel_ranking_predictions.jsonl) \
     <(jq -cS '{id,dispatched_domains,rank2_new}' results/iter62_noop_check_iter61head.jsonl)

# 1) 較正付き多ラベルヘッドの訓練（1427 + 135 行を embed．訓練データは Iter61 のものを再利用）
uv run python -m scripts.train_multilabel_dispatch_head \
    --train-data data/classifier_train.jsonl \
    --multilabel-train-data data/classifier_train_multidomain_iter61.jsonl \
    --embedding-model nomic-embed-text --ollama-host 192.168.15.100 \
    --output models/dispatch_multilabel_head_iter62.joblib

# 2) 1600 問のオフライン採点（埋め込みキャッシュ完全ヒットの想定＝embed 呼び出し 0 件）
#    --iter59-predictions は引数名に反して汎用（_compute_a5_iter59_disagreement():316-337）．
#    A5 を「対 Iter61 不一致」に読み替えるため Iter61 の予測を渡す．
uv run python -m scripts.evaluate_dispatch_candidate_ranking \
    --baseline results/20260918_202613/results.jsonl \
    --head models/dispatch_multilabel_head_iter62.joblib \
    --embedding-model nomic-embed-text --ollama-host 127.0.0.1 --ollama-port 11435 \
    --embedding-cache results/iter59_query_embeddings.npz \
    --iter59-predictions results/iter61_multilabel_ranking_predictions.jsonl \
    --output results/iter62_multilabel_ranking_predictions.jsonl

# 3) 指標・検定（Iter59〜61 と同一スクリプト・同一手続き．無改造）
uv run python -m scripts.compute_iter59_ranking_stats \
    --baseline results/20260918_202613/results.jsonl \
    --new results/iter62_multilabel_ranking_predictions.jsonl \
    --output results/iter62_stats.json

# 4) S5 用のドメイン別被覆集計（education / medical / legal）
#    compute_iter59_ranking_stats.py:_domain_pair_coverage_maps() を読み取り専用で呼ぶ
#    アドホック集計．公式の採点・統計パスには手を入れない（Iter61 の N6 と同じ手順）．
```

**成功条件（事前登録．事後変更禁止）**

config.yml:950-955 の暫定案をそのまま確定する（変更点は S5 の測定手順を明文化した点のみで，
閾値は一切変更していない）．基準線は `results/20260918_202613/results.jsonl`（0.345），
第 2 参照点は Iter61（0.445，legal 絡み除外 p=0.0501，education 5/20，medical 12/28）で，
両者を必ず併記する．

- **S1（有意性の維持）**: 全体 200 ペアの exact McNemar（`scipy.stats.binomtest`，α=0.05）で
  基準線比 **p < 0.05**．
- **S2（全体性能の非低下）**: `compound_domain_set_recall` が **Iter61 の 0.445 を下回らない**
  （≧0.445．同値許容）．
- **S3（コスト中立）**: `mean_dispatch = 2.000000`（完全一致）．
- **S4（発火の証拠）**: **対 Iter61 の rank_2 不一致 > 0**（A5．0 件なら較正が no-op）．
- **S5（本レバー固有の主基準）**: **education 自身の被覆 ≧ 9/20 かつ medical ≧ 13/28**
  （いずれも基準線の値まで回復すること）．

**判定規則（事前登録）**

- **S1〜S5 全充足** → `per_domain_holdout_calibration` を **adopted**．Iter60/61 の
  「副次的退行」は較正で解消可能だったと結論し，対外記述から education/medical の退行留保を外す．
- **S1〜S4 充足・S5 のみ不成立（ただし education > 5/20 または medical > 12/28 と Iter61 比で
  改善方向）** → **partial**．「較正は退行を部分的にしか埋め合わせない」と記録し，
  残る余地（config.yml:961-965 の (i)(ii)(iii)）へ引き継ぐ．
- **S2 不成立（0.445 未満へ低下）** → **rejected**．較正は全体性能を犠牲にするため採らない．
- **S4 不成立（対 Iter61 不一致 0）** → **no-op** として rejected 扱いとし，較正が rank_2 の
  順序を変えない（単調変換に留まる）ことを機序として記録する．

**非退行条件（Iter61 から踏襲．いずれか FAIL なら adopted にしない）**

- **N1**: rank_1 が基準線と 1600/1600 で一致（採点スクリプトが assert）．
- **N2**: `top1_accuracy = 0.5975` に完全一致．
- **N3**: legal 自身の被覆 **≧ 8/30**（`_compute_n3():228-250` が自動判定）．
- **N5**: 単一ドメイン 1500 行の argmax 正解率 **≧ 0.590**．
  （N4 は Iter60 時点で欠番．新設しない．）

**アサーション（no-op・交絡対策）**

- **A0**: `(Y.sum(axis=1) >= 2).sum() == n_synthetic_rows`，**期待値 135**（Iter61 と同一．
  訓練データを再利用するため一致しなければ入力取り違え）．`len(mlb.classes_) == 10`，
  被覆ペア数 45/45 も併せて報告．
- **A1/A2/A3/A6/A7**: Iter61 の定義をそのまま使用（A7 は生成を行わないため該当なし＝skip．
  訓練データが Iter61 と同一ファイルであることをもって代替とする）．
- **A5（読み替え）**: 対 **Iter61** 予測の rank_2 不一致 **> 0**（＝S4）．
- **A8（新設・本レバー固有）**: 上記コマンド 0) のとおり，`_head_scores()` の `predict_proba` 化を
  適用した採点スクリプトで **Iter61 のヘッドを再採点した結果が
  `results/iter61_multilabel_ranking_predictions.jsonl` と完全一致**すること．
  一致しなければ「スコア取得関数の変更」自体が効果に混入していることになり，
  単一レバー原則が破れるため実験を中止して原因を調査する．

**単一レバー原則の確認（混入チェック）**

- 訓練データ（1427＋135 行）: 再生成せず Iter61 のファイルをそのまま入力 → **無変更**．
- 生成プロンプト・フィルタ F1〜F4・temperature・生成モデル: 今回の実行経路に登場しない →
  **無変更**．
- rank_1 経路: 基準線 JSONL からのコピーのみでヘッドに非依存 → **無変更**（N1 で実証）．
- 基準線・採点スクリプトの他ロジック・統計スクリプト・埋め込みモデル・埋め込みキャッシュ →
  **無変更**．
- コード変更は 2 ファイル各 1 箇所（base estimator の差し替え／スコア取得の呼び先変更）のみ．
  後者は未較正ヘッドに対し数値的 no-op であることを A8 で実証するため，実効的な変数は
  **「較正の有無」1 つだけ**である．
- 出力パス名の変更は測定対象に影響しない（ファイル I/O のみ）ため単一レバー原則に抵触しない．

**既知の制約（申し送り）**

- 較正は各ドメイン内での単調変換であるため，**rank_1 経路（argmax）と各ドメインの ROC-AUC は
  原理的に不変**であり，効果は「ドメイン間比較」＝rank_2 選択にのみ現れる．S4 が 0 になる
  可能性（較正後もドメイン間の大小関係が変わらない）は事前に想定しておく．
- `CalibratedClassifierCV(ensemble=True)` は各ドメインにつき 5 個の LogisticRegression を
  fit するため，訓練時間が約 5 倍になる（1562 行×10 ドメインの規模では実用上の問題はない見込み）．
- 低品質行（プロンプトの echo）の混入は Iter60 から続く既知の穴であり，訓練データを据え置く
  本イテレーションでもそのまま残る（フィルタは変更しない）．
- 実行時経路への配線は未実施のまま（B94/B95．ユーザー確認待ち）．

### 実装 (Iter62)

計画フェーズが確定した仕様どおり，2 ファイル各 1 箇所のみを変更した（訓練データ・rank_1 経路・
採点/統計スクリプトの他ロジック・`config.yaml` は無変更）．

1. **`scripts/train_multilabel_dispatch_head.py`**
   - import に `from sklearn.calibration import CalibratedClassifierCV` を追加（`:43` 付近）．
   - `_DIAGNOSTIC_CV_RANDOM_STATE = 42` の直後に較正専用の新定数
     `_CALIBRATION_CV = 5`（診断用 `_DIAGNOSTIC_CV` とは別責務のため独立定義，計画フェーズ決定事項 1
     どおり）を追加．
   - `train_multilabel_ranking_head()`（旧 `:145-157`）を，
     `base_estimator = LogisticRegression(max_iter=_MAX_ITER, class_weight="balanced")` →
     `calibrated_estimator = CalibratedClassifierCV(base_estimator, method="sigmoid",
     cv=_CALIBRATION_CV, ensemble=True)` → `OneVsRestClassifier(calibrated_estimator)` の 3 行に
     差し替え．`OneVsRestClassifier` でラップする構造・`model.fit(embeddings, Y)` の呼び出し・
     関数シグネチャ・戻り値の型は無変更．docstring を較正の意図（held-out 分割がドメイン内で
     完結すること，`method`/`ensemble` を明示指定する理由）を説明する内容に更新した．
   - `_train_and_save()`・保存形式（`joblib.dump({"model":..., "classes":...})`）・
     `_print_per_domain_cv_diagnostics()`（未較正の別インスタンスを使う独立経路）・CLI 引数は
     無変更．

2. **`scripts/evaluate_dispatch_candidate_ranking.py`**
   - `_head_scores()`（旧 `:166-178`）の本体を
     `logits = model.decision_function([embedding])[0]; probabilities = _sigmoid(np.asarray(logits))`
     から **`probabilities = np.asarray(model.predict_proba([embedding])[0])`** の 1 行へ置換．
     ヘッド種別による分岐は設けていない（計画どおり一律 `predict_proba()`）．docstring を
     切替理由（`CalibratedClassifierCV` が `decision_function` 未実装のため
     `OneVsRestClassifier.decision_function()` が `AttributeError` になること，未較正ヘッドに対し
     数値的 no-op であること）を説明する内容に更新した．
   - `_sigmoid()`（`:102-104`）は呼び出し元がなくなったが，計画どおり**削除していない**
     （`ruff check` はモジュールレベル未使用関数を検出しないため lint エラーにもならない）．
   - `build_new_rows()` の `rank2_new = max(...)`・`_load_head()`・
     `_assert_head_scores_are_domain_names()`（A6）は無変更．

3. **`tests/test_train_multilabel_dispatch_head.py`**（計画外だが必然的な追随修正．検証節参照）
   - `test_train_multilabel_ranking_head_fits_a_model_that_predicts_seen_multilabel_rows` の
     フィクスチャを各クラスタ 2 行→ 6 行（計 6→18 行）へ拡張．較正導入により
     `CalibratedClassifierCV(cv=5)` が各二値問題の各クラスに 5 例以上を要求するようになった
     ためで，テストの意図（seen row の多ラベル予測復元）は変更していない．

**A8 事前チェック（実機実行・完了）**

計画節記載のコマンドをそのまま実行した．

```
uv run python -m scripts.evaluate_dispatch_candidate_ranking \
    --baseline results/20260918_202613/results.jsonl \
    --head models/dispatch_multilabel_head_iter61.joblib \
    --embedding-model nomic-embed-text --ollama-host 127.0.0.1 --ollama-port 11435 \
    --embedding-cache results/iter59_query_embeddings.npz \
    --iter59-predictions results/iter60_multilabel_ranking_predictions.jsonl \
    --output results/iter62_noop_check_iter61head.jsonl
```

出力サマリ（stderr）: `compound_domain_set_recall=0.445`（Iter61 記録値と一致），
`n5_single_domain_argmax_accuracy.accuracy=0.6033`（pass），埋め込みキャッシュ完全ヒットのため
embed 呼び出し 0 件で完走．

`diff <(jq -cS '{id,dispatched_domains,rank2_new}' results/iter61_multilabel_ranking_predictions.jsonl) \
     <(jq -cS '{id,dispatched_domains,rank2_new}' results/iter62_noop_check_iter61head.jsonl)` は
**0 行**（完全一致）．念のため `head_scores` を含む全フィールドの `jq -cS .` diff も取り，同様に
**0 行**であることを確認した．**A8 PASS**: `_head_scores()` の `predict_proba()` 化は未較正ヘッド
（Iter61）に対してビット単位の no-op であり，スコア取得関数の変更自体は較正効果に混入しない．

**検証**

- `uv run ruff check scripts/train_multilabel_dispatch_head.py
  scripts/evaluate_dispatch_candidate_ranking.py` → `All checks passed!`．
- `mypy` はリポジトリに未導入（`pyproject.toml` に設定なし，`uv run mypy` は
  `error: Failed to spawn: mypy` で未実行．型チェックは実施できず，この旨のみ報告する）．
- `uv run pytest tests/test_train_multilabel_dispatch_head.py
  tests/test_evaluate_dispatch_candidate_ranking.py` を実行．変更直後は
  `test_train_multilabel_ranking_head_fits_a_model_that_predicts_seen_multilabel_rows` が
  `ValueError: Requesting 5-fold cross-validation but provided less than 5 examples for at
  least one class` で失敗した．これは較正導入に伴う必然的な帰結（`CalibratedClassifierCV(cv=5)`
  は各二値問題の各クラスに 5 例以上を要求するが，既存フィクスチャは各クラスタ 2 行しかなかった）
  であり，同テストのフィクスチャを各クラスタ 6 行（計 18 行，legal/medical とも各クラス
  6 件以上）へ拡張して意図（seen row の多ラベル予測復元）を保ったまま解消した．
  再実行で **17 件全て pass**．
- 全体テスト (`uv run pytest -q`) では上記 2 ファイル以外に `test_build_dataset.py`・
  `test_train_domain_classifier.py` の計 12 件が失敗するが，変更前（`git stash` で本イテレーション
  の変更を退避して再実行）でも同じ 12 件が同じ原因（`train_domain_classifier.py` が
  `CalibratedClassifierCV.classes_` に依存しており，これは別イテレーションの既存資産で今回の
  変更対象外）で失敗することを確認済みであり，本イテレーションの変更によるものではない．

**実験フェーズへの申し送り**

コードの変更は上記 2 箇所のみで A8 も PASS しているため，計画節のコマンド 1)〜4)
（較正付きヘッドの訓練 → `models/dispatch_multilabel_head_iter62.joblib` → 1600 問オフライン採点 →
`results/iter62_multilabel_ranking_predictions.jsonl` → 統計 `results/iter62_stats.json`）を
そのまま実行してよい状態にある．訓練は `CalibratedClassifierCV(ensemble=True)` により
Iter61 比で約 5 倍の学習時間が見込まれる（計画節の既知の制約どおり）．

### 実験 (Iter62)

**接続先ホストの読み替え（Iter61 と同じ状況を再確認した上での変更）**

計画書は訓練を `--ollama-host 192.168.15.100`（既定ポート），採点を `--ollama-host 127.0.0.1
--ollama-port 11435` としていた．実行前に到達性を確認したところ，`curl -m 3
http://192.168.15.100:11434/api/tags` は今回もタイムアウト（応答なし）で直接 IP は不通，一方
既存の SSH ローカルポートフォワード（`ssh -fNT -L 11435:localhost:11434 wafl500`，実行前から
2 プロセス稼働中）経由の `curl http://127.0.0.1:11435/api/tags` は応答し，モデル一覧に
`nomic-embed-text:latest` を含むことを確認した．したがって**訓練・採点の両方で
`--ollama-host 127.0.0.1 --ollama-port 11435` に統一した**（Iter61 と同じ読み替え．レバー以外の
パラメータは変更していない）．

**実行した3コマンド（実際のホスト・ポート）**

```
# 1) 較正付き多ラベルヘッドの訓練
uv run python -m scripts.train_multilabel_dispatch_head \
    --train-data data/classifier_train.jsonl \
    --multilabel-train-data data/classifier_train_multidomain_iter61.jsonl \
    --embedding-model nomic-embed-text --ollama-host 127.0.0.1 --ollama-port 11435 \
    --output models/dispatch_multilabel_head_iter62.joblib

# 2) 1600問オフライン採点（A5/S4 は Iter61 予測との不一致に読み替え）
uv run python -m scripts.evaluate_dispatch_candidate_ranking \
    --baseline results/20260918_202613/results.jsonl \
    --head models/dispatch_multilabel_head_iter62.joblib \
    --embedding-model nomic-embed-text --ollama-host 127.0.0.1 --ollama-port 11435 \
    --embedding-cache results/iter59_query_embeddings.npz \
    --iter59-predictions results/iter61_multilabel_ranking_predictions.jsonl \
    --output results/iter62_multilabel_ranking_predictions.jsonl

# 3) 指標・検定（compute_iter59_ranking_stats.py，無改造）
uv run python -m scripts.compute_iter59_ranking_stats \
    --baseline results/20260918_202613/results.jsonl \
    --new results/iter62_multilabel_ranking_predictions.jsonl \
    --output results/iter62_stats.json
```

**1) 訓練結果**

デタッチ実行（バックグラウンド，nohup）で開始し，`poll_interval_sec` に準じて経過確認した．
総所要時間は約 285 秒（実測，`ps -o etimes`）で，事前に見込んだ「Iter61 比で約 5 倍」の学習時間
（＝訓練データ 1562 行×10 ドメインの `CalibratedClassifierCV(cv=5, ensemble=True)` フィット自体の
所要）よりも短く完了した（フィット自体は埋め込み計算より軽量なため）．実行中の異常・OOM・
タイムアウトは発生していない．

`[train_multilabel_dispatch_head] A0 PASS: 135 multi-label rows covering 45 distinct domain pairs`．
診断用 5-fold CV（未較正の別インスタンス経由，較正導入の影響を受けない独立経路であることを
実装フェーズで確認済み）は Iter61 と完全に同一の値だった:

| domain | n_positive | cv_roc_auc | cv_average_precision |
|---|---|---|---|
| business_economics | 177 | 0.8143 | 0.4244 |
| computer_science | 177 | 0.9057 | 0.6116 |
| education | 177 | 0.8531 | 0.4407 |
| general | 177 | 0.8831 | 0.6232 |
| history_culture | 177 | 0.9353 | 0.7634 |
| legal | 104 | 0.9089 | 0.5989 |
| mathematics | 177 | 0.9276 | 0.7411 |
| medical | 177 | 0.7817 | 0.3915 |
| natural_science | 177 | 0.8361 | 0.4512 |
| social_science | 177 | 0.8629 | 0.5815 |

`wrote models/dispatch_multilabel_head_iter62.joblib (n_single_label_rows=1427, n_synthetic_rows=135,
classes=[全10ドメイン名])`．sha256 `cef6c587b342458c9d7adfcf61619ae0c87bc44e126541d39cf36950932b3d83`，
ファイルサイズ 334,699B．

**2) 採点結果（`evaluate_dispatch_candidate_ranking.py` 標準出力の JSON，埋め込みキャッシュ完全
ヒットで embed 呼び出し 0 件）**

```json
{
  "n_rows": 1600,
  "mean_dispatch": 2.0,
  "rank2_flip_rate": 0.436875,
  "compound_domain_set_recall": 0.445,
  "compound_rows_evaluated": 100,
  "n5_single_domain_argmax_accuracy": {
    "n_single_domain_rows": 1500, "correct": 913, "accuracy": 0.6086666666666667,
    "floor": 0.59, "pass": true
  },
  "a5_iter59_disagreement": {
    "n_rows": 1600, "mismatches": 284, "mismatch_rate": 0.1775
  }
}
```

（`a5_iter59_disagreement` は `--iter59-predictions` に `results/iter61_multilabel_ranking_predictions.jsonl`
を渡したため，実際には**対 Iter61 不一致**を表す．A1/A2：例外なし＝rank_1完全一致・
mean_dispatch=2.0．A6：例外なし＝全1600行で `head_scores` のキーが10ドメイン名文字列と完全一致．）
出力: `results/iter62_multilabel_ranking_predictions.jsonl`（1600行，sha256
`829494a8a9ef81525208e8af10392301a19a8e9df91e253c7165f7b4ee71d80f`）。

**3) 指標・検定結果（`results/iter62_stats.json` 全文，sha256
`ee8a023ac9fa37179951652fd9bd6ecabd3e4ce5d03ed45b5f005922d233797d`）**

```json
{
  "baseline_compound_domain_set_recall": 0.345,
  "new_compound_domain_set_recall": 0.445,
  "matches_implementation_phase_diagnostic_recall": false,
  "S1_primary_criterion": {
    "n_pairs": 200, "improved_pairs": 33, "regressed_pairs": 13, "discordant_pairs": 46,
    "chi2_statistic_continuity_corrected": 7.8478260869565215,
    "p_value_continuity_corrected": 0.005088185461451067,
    "p_value_exact_binomtest": 0.004533861582189047,
    "pass": true
  },
  "S2_effect_size_floor": {
    "baseline_recall": 0.345, "new_recall": 0.445, "delta_pt": 0.10000000000000003,
    "floor_pt": 0.04, "pass": true
  },
  "S3_cost_neutrality": {
    "n_rows": 1600, "length_distribution": {"2": 1600}, "duplicate_rank1_rank2_count": 0,
    "mean_dispatch": 2.0, "pass": true
  },
  "S4_flip_rate_evidence_of_firing": {
    "n_rows": 1600, "flips": 699, "rank2_flip_rate": 0.436875,
    "matches_implementation_phase_value": false, "implementation_phase_value": 0.356875,
    "pass": true
  },
  "N1_rank1_invariance": {"n_rows": 1600, "mismatch_count": 0, "mismatch_ids": [], "pass": true},
  "N2_top1_accuracy_invariance": {
    "baseline_top1_accuracy": 0.5975, "new_top1_accuracy": 0.5975, "exact_match": true, "pass": true
  },
  "N3_legal_non_regression": {
    "n_legal_involving_pairs": 30, "baseline_legal_self_coverage": 8, "new_legal_self_coverage": 16,
    "expected_baseline_value": 8, "baseline_matches_journal_record": true, "pass": true
  },
  "N4_improvement_breakdown_by_domain_category": {
    "legal_involving": {"n_pairs": 60, "improved": 9, "regressed": 0, "unchanged": 51},
    "medical_involving": {"n_pairs": 32, "improved": 2, "regressed": 4, "unchanged": 26},
    "other": {"n_pairs": 108, "improved": 22, "regressed": 9, "unchanged": 77}
  }
}
```

**4) S5 用のドメイン別自己被覆（education / medical / legal．`compute_iter59_ranking_stats.py`
の `_domain_pair_coverage_maps()` を読み取り専用で呼ぶアドホック集計，Iter60/61 の N6 と同じ手順．
公式の採点・統計パスには手を入れていない）**

```json
{"domain": "education", "n_pairs": 20, "baseline_self_coverage": 9, "new_self_coverage": 6}
{"domain": "medical", "n_pairs": 28, "baseline_self_coverage": 13, "new_self_coverage": 11}
{"domain": "legal", "n_pairs": 30, "baseline_self_coverage": 8, "new_self_coverage": 16}
```

legal の `new_self_coverage=16` は `stats.json` の `N3_legal_non_regression.new_legal_self_coverage`
（16）と一致し，アドホック集計の実装が公式パスと整合していることを確認した。

**成果物・付随確認**

- 新規ファイル: `models/dispatch_multilabel_head_iter62.joblib`（gitignore 対象，`models/`）・
  `results/iter62_multilabel_ranking_predictions.jsonl`（1600行）・`results/iter62_stats.json`。
  実装フェーズが作成した `results/iter62_noop_check_iter61head.jsonl`（A8 事前チェック，PASS 済み）
  は変更していない。
- Iter61 の4資産（`data/classifier_train_multidomain_iter61.jsonl`・
  `models/dispatch_multilabel_head_iter61.joblib`・
  `results/iter61_multilabel_ranking_predictions.jsonl`・`results/iter61_stats.json`）の `mtime` を
  実行前後で確認し，一切変更されていないことを確認した。
- `git status --short` で確認した差分は `results/iter62_multilabel_ranking_predictions.jsonl`・
  `results/iter62_stats.json`（未追跡の新規2ファイル）のみで，実装フェーズの3ファイル変更
  （`scripts/train_multilabel_dispatch_head.py`・`scripts/evaluate_dispatch_candidate_ranking.py`・
  `tests/test_train_multilabel_dispatch_head.py`）以外の作業ツリーへの変更は無い。
- 実行中の異常・障害は発生していない（ネットワーク接続の読み替えを除き，全3コマンドとも
  exit code 0，エラー出力なし，OOM・タイムアウトなし）。

**解釈・採否判断はこのフェーズでは行わない**（次の分析(解釈)フェーズに委ねる。上記はすべて
生の実測値であり，`pass: true/false` はスクリプト自身が出力した事前登録済みアサーションの結果を
そのまま転記したものである）。

### 分析 (Iter62)

**数値集計のみ（解釈・採否判定は次フェーズ rc-analyst の担当）**

事前登録の成功条件（config.yml，本イテレーション計画節）に対応する実測値の対応表:

| 条件 | 事前登録の基準 | 実測値 | 基準との機械的な照合 |
|---|---|---|---|
| S1（主基準・全体） | n=200 exact McNemar，p<0.05 | p=0.004533861582189047（改善33／悪化13／discordant46） | 満たす |
| S2（全体性能の非低下） | `compound_domain_set_recall` ≧ Iter61 の 0.445 | 0.445（Iter61 と同値） | 満たす（同値） |
| S3（コスト中立） | mean_dispatch=2.000000（完全一致） | 2.000000（全1600行 len=2，重複0） | 満たす |
| S4（発火の証拠） | 対 Iter61 の rank_2 不一致 > 0 | 284/1600（0.1775） | 満たす |
| S5（本レバー固有の主基準） | education 自身の被覆 ≧9/20 かつ medical ≧13/28 | education 6/20，medical 11/28 | **満たさない（両方とも基準未達）** |
| N1（rank_1 不変） | 1600/1600 一致 | 不一致 0 | 満たす |
| N2（top1_accuracy 不変） | 0.5975 完全一致 | 0.5975（完全一致） | 満たす |
| N3（legal 非退行） | legal 自身の被覆 ≧8/30 | 16/30 | 満たす |
| N5（単一ドメイン argmax 非退行） | ≧0.590 | 0.608667（913/1500） | 満たす |

**参考（第2参照点との並置，基準線・Iter61・Iter62）**

| 指標 | 基準線（`20260918_202613`） | Iter61 | Iter62 |
|---|---|---|---|
| compound_domain_set_recall | 0.345 | 0.445 | 0.445 |
| S1 exact p（対基準線） | — | 0.003657766827927844 | 0.004533861582189047 |
| rank2_flip_rate（対基準線） | — | 0.46 | 0.436875 |
| mean_dispatch | — | 2.000000 | 2.000000 |
| top1_accuracy | 0.5975 | 0.5975 | 0.5975 |
| legal 自身の被覆 | 8/30 | 15/30 | 16/30 |
| medical 自身の被覆 | 13/28 | 12/28 | 11/28 |
| education 自身の被覆 | 9/20 | 5/20 | 6/20 |
| N5 単一ドメイン argmax | — | 0.603333 | 0.608667 |
| 対 Iter61 rank_2 不一致（S4/A5） | — | （552，対Iter60） | 284（対Iter61） |

**判定不能だった項目**: なし。config.yml/計画節が事前登録した S1〜S5・N1〜N5 の全項目について，
`evaluate_dispatch_candidate_ranking.py`・`compute_iter59_ranking_stats.py` の出力および
`_domain_pair_coverage_maps()` を用いたアドホック集計から実測値を取得できた。

### 分析（解釈）（Iter62）

採否の最終判定は次の考察フェーズに委ねる．本節は一次データ（`results/iter61_multilabel_ranking_
predictions.jsonl`・`results/iter62_multilabel_ranking_predictions.jsonl` の全 1600 行，
`results/iter62_stats.json`）を直接集計した結果に基づく解釈である．自己被覆の独立再集計は
education 9→5→6（基準線→Iter61→Iter62）・medical 13→12→11・legal 8→15→16 となり，実験節の
アドホック集計および `stats.json` の `N3.new_legal_self_coverage=16` と一致したので，以下の集計基盤は
公式パスと整合している．

**1. S1 成立・S5 不成立の意味づけ（目的変数は動いていない）**

S1（全体 200 ペア，p=0.004534）は基準線に対する全体の改善を示すが，これは Iter61（p=0.003658，
改善 32／悪化 12）とほぼ同一の改善であり，`compound_domain_set_recall` も 0.445 で完全同値である．
すなわち S1 が測っているのは**主として Iter60/61 で既に得られていた legal 側の改善の持続**であって，
本レバーが新たに生んだ効果ではない（N4 内訳: legal_involving 改善 9／悪化 0 は Iter61 の 10／2 と
同水準，medical_involving は改善 2／悪化 4 で Iter61 の 1／3 から改善していない）．

本レバーの目的変数は education/medical 自身の被覆であり，そこでは変化が事実上生じていない．
対 Iter61 のペアごとの不一致（McNemar 的な discordant）を目的行だけで数えると，
**education 20 ペア中 discordant 1 件（改善 1／悪化 0，exact p=1.0），medical 28 ペア中 discordant
1 件（改善 0／悪化 1，exact p=1.0）** にすぎない．全体で 284/1600 行の rank_2 が動いたにもかかわらず，
その発火はほぼ全て目的行の外側で起きている．したがって S4（発火の証拠）は「較正が何かを動かした」
ことしか示さず，「意図した方向に動かした」根拠にはならない．

**2. 較正は目的行の順位をほぼ変えていない**

education/medical が rank_1 でない（＝rank_2 で拾う必要がある）複合行に限り，rank_1 を除く 9 候補の中で
当該ドメインが何位に来るかを集計した．

| ドメイン | 対象行 | 候補内平均順位 Iter61→Iter62 | 1 位になった行数 | 勝者との平均スコア差 |
|---|---|---|---|---|
| education | 16 | 4.62 → 4.62 | 1 → 2 | 0.7130 → 0.2334 |
| medical | 21 | 2.95 → 2.81 | 5 → 4 | 0.3523 → 0.1504 |
| legal | 29 | 3.03 → 3.14 | 14 → 15 | 0.3516 → 0.1016 |

スコア差（マージン）は較正でスケールが縮んだぶん一律に小さくなるが，**順位そのものは平均値で
ほぼ不変**である．特に education は平均 4.62 位のまま動かず，「ドメイン間のスケールずれを直せば
education が上位に来る」という仮説の前提が目的行上で成立していない．

**3. 較正が意図と逆方向に働いた機序（スコア分布の実測）**

全 1600 行の `head_scores` から各ドメインのスコア分布を較正前後で比較した．

| ドメイン | cv_AP | 平均 i61→i62 | 標準偏差 i61→i62 | sd 比 | rank_2 獲得数の純増減 |
|---|---|---|---|---|---|
| medical | 0.3915 | 0.1655→0.1046 | 0.3044→0.0683 | **0.224** | −9 |
| business_economics | 0.4244 | 0.1531→0.0969 | 0.2849→0.0902 | 0.317 | −34 |
| education | 0.4407 | 0.1573→0.0981 | 0.3013→0.0991 | **0.329** | +14 |
| natural_science | 0.4512 | 0.1575→0.1080 | 0.3083→0.1247 | 0.404 | +3 |
| social_science | 0.5815 | 0.1510→0.0989 | 0.2885→0.0994 | 0.344 | −28 |
| legal | 0.5989 | 0.1175→0.0769 | 0.2757→0.1439 | 0.522 | −13 |
| computer_science | 0.6116 | 0.1432→0.1044 | 0.2990→0.1288 | 0.431 | +45 |
| general | 0.6232 | 0.1364→0.0973 | 0.2832→0.1222 | 0.432 | −14 |
| mathematics | 0.7411 | 0.1206→0.1026 | 0.2829→0.1660 | 0.587 | +21 |
| history_culture | 0.7634 | 0.1595→0.1230 | 0.3214→0.1956 | 0.608 | +15 |

**sd の縮み方は各ドメインの分離性能と強く単調に対応する（Spearman ρ(cv_AP, sd 比)=0.952,
p=2.3e-5）**．Platt scaling は分離が悪い二値問題ほど出力を基準率付近へ強く縮めるため，
**最も縮んだのは cv_AP が最下位・下から 3 番目の medical（0.224 倍）と education（0.329 倍）**，
すなわち今回持ち上げたかった当の 2 ドメインである．行方向の合計は 1.461→1.011 へ下がり，
平均値はドメイン間で 0.077〜0.123 とほぼ揃った（Iter61 は 0.117〜0.166）．つまり較正が揃えたのは
スコアの**水準**であり，rank_2 は行内の `max` 比較なので勝敗を決めるのは水準ではなく**可動域
（分散）**である．低 AP ドメインの可動域だけを選択的に潰す変換は，rank_2 競争において
education/medical を構造的に不利にする．レバーの機序仮説とは逆方向である．

なお config.yml が挙げた機序仮説「各ドメインの n_positive が不揃い」は，実データでは
legal=104・他 9 ドメイン=177 であり，education と medical は多数派側の 177 に属する．較正挙動の
ドメイン間差を説明しているのは n_positive ではなく分離性能（cv_AP／cv_ROC_AUC）であり，
**事前登録された機序仮説はデータに支持されない**．

**4. education の増加分はほぼ偽陽性である**

education の rank_2 獲得数は 130→144（+14）だが，そのうち `expected_domains` に education を含む
行は 9→11（+2）にとどまり，複合 20 ペアでの被覆増は +1 である．較正後の education の rank_2 精度は
11/144=7.6% で 10 ドメイン中最低（legal は 37/102=36.3%）．「education の出力を持ち上げる」方向の
効果は多少あるが，行き先が目的行ではなくほぼ無関係な行であり，被覆指標に寄与していない．

**5. ノイズか設計上の問題か**

- 対 Iter61 の目的行変化は education +1・medical −1 であり，discordant がそれぞれ 1 件しかない．
  n=20/28 の固定項目集合に対する対応ありの比較として，これは**ゼロと区別できない**．
  「横ばい」の判定自体は確度が高い（＝改善があったとは言えない）．
- S5 の基準との差（education 6→9 は +3，medical 11→13 は +2）は，二項的なばらつき
  （n=20, p≈0.3 で sd≈2.05／n=28, p≈0.4 で sd≈2.6）と同程度の大きさである．さらに基準線との
  対応あり比較でも education は改善 1／悪化 4（exact p=0.375），medical は改善 2／悪化 4
  （p=0.6875）で，**基準線との差自体が有意ではない**．したがって S5 不成立は
  「回復の証拠が得られなかった」であって「較正が有害であると実証された」ではない．
- 一方で，**較正が効かなかった理由は偶然ではなく構造的**である．較正はドメイン内単調変換で
  あるため rank_1（argmax）と各ドメインの ROC-AUC を原理的に変えず（N1・N2・診断 CV が Iter61 と
  完全同値であることが実証），ドメイン間比較に対する効果は上記 3 のとおり低 AP ドメインの
  可動域圧縮として現れる．これは追加反復で符号が反転する類のばらつきではない．
- 以上より，本イテレーションの結果は「ノイズで埋もれた小さな真の効果」ではなく，
  「介入の作用方向が目的変数に対して中立〜逆」と読むのが，得られたデータに最も整合する．
  ただし目的指標の n が 20/28 と小さく，±2 件程度の真の効果を検出する統計的検出力がない点は
  この系統の測定基盤そのものの制約として残る．

**6. 事前登録の判定規則との機械的な対応（参考，最終判定は次フェーズ）**

S1〜S4 充足・S5 のみ不成立で，かつ education 6/20 > Iter61 の 5/20 であるため，事前登録の
判定規則の文言上は **partial** に該当する．ただしその「改善方向」は discordant 1 件に基づくもので
ノイズと区別できず，medical は −1 で逆方向である点を，採否判断の際に併せて考慮すべきである．

### Iteration 62 実行済み（考察・次計画）

**単一レバー**: `multilabel_rank2_score_calibration = per_domain_holdout_calibration`
（OvR ヘッドの base estimator を `CalibratedClassifierCV(LogisticRegression(max_iter=1000,
class_weight="balanced"), method="sigmoid", cv=5, ensemble=True)` へ差し替え，rank_2 を
ドメイン別較正後のスコアで選ぶ）．

**変更したもの**: `scripts/train_multilabel_dispatch_head.py`（較正器の導入）と
`scripts/evaluate_dispatch_candidate_ranking.py`（スコア源を `decision_function`+sigmoid から
`predict_proba` へ）の各 1 箇所，および対応するテスト．訓練データ（1427＋135 行）・ヘッド構造の
他要素・rank_1 経路・基準線（`results/20260918_202613/results.jsonl`，0.345）・採点/統計スクリプトの
他ロジック・`config.yaml` はすべて Iter61 から固定．スコア源切替が未較正ヘッドに対して数値的 no-op で
あることは A8（Iter61 ヘッドの再採点が `results/iter61_multilabel_ranking_predictions.jsonl` と完全一致）で
実証したので，実効的に動いた変数は「較正の有無」1 つだけである．

**結果**: S1 exact McNemar p=0.004534（改善 33／悪化 13），S2 `compound_domain_set_recall`=0.445
（Iter61 と完全同値），S3 mean_dispatch=2.000000，S4 対 Iter61 の rank_2 不一致 284/1600，
N1（rank_1 1600/1600 不変）・N2（top1_accuracy 0.5975 完全一致）・N3（legal 16/30）・
N5（0.608667）はすべて PASS．**主基準 S5 は不成立**（education 6/20［基準 9/20］，
medical 11/28［基準 13/28］）．

**判定: partial（部分的成立）— ただし実質は「効果なし」．レバーはクローズ（試し切り・収束）**

事前登録の判定規則（本イテレーション計画節，config.yml:950-955 を確定したもの）の**第 2 分岐**
（S1〜S4 充足・S5 のみ不成立，かつ education 6/20 > Iter61 の 5/20 と改善方向）に**文言どおり
機械的に該当する**ため partial とする．事後の緩和・厳格化は行っていない．ただし規定どおり
「較正は退行を部分的にしか埋め合わせない」と記録したうえで，**効果量としては 0 と扱い，
対外記述で『較正により education/medical の退行が緩和された』とは書かない**（下記の留保 R-D）．

- **留保 R-D（partial の中身）**: partial を成立させた「education +1」は対 Iter61 の discordant が
  1 件（改善 1／悪化 0）に基づくものでノイズと区別できず，medical は逆方向（−1，discordant 1 件）である．
  基準線との対応あり比較でも education 改善 1／悪化 4（exact p=0.375），medical 改善 2／悪化 4
  （p=0.6875）で，**目的行では基準線とも Iter61 とも有意差がない**．
- **S1 が測っているものの帰属**: S1（p=0.004534）は Iter60/61 で既に得られていた legal 側の改善の
  持続であって，本レバーの新規効果ではない（`compound_domain_set_recall` は 0.445 で完全同値，
  medical_involving は改善 2／悪化 4 で Iter61 の 1／3 から改善していない）．
- **効果量の対外的な正式値は Iter61 の +10.0pt（p=0.003658）のまま据え置く**（B95 の方針を変更しない）．
  R-C（education・medical が基準線を下回る）も未解消のまま残る．

**学び**

1. **事前登録した機序仮説がデータに支持されなかった**．config.yml が挙げた「各ドメインの n_positive が
   不揃い（legal 104〜他 150+27）だから素点が較正されていない」は，実データでは legal=104・
   他 9 ドメイン=177 であり，**education と medical はいずれも多数派側の 177**に属する．
   較正挙動のドメイン間差を説明しているのは件数ではなく分離性能だった．計画時に n_positive の実値を
   ドメイン別に確認していれば，この仮説は着手前に棄却できた（次回以降，機序仮説の根拠となる数値は
   計画フェーズで必ず実データから引く）．
2. **Platt 較正は rank_2 競争において低分離ドメインを構造的に不利にする（今回の中核的な知見）**．
   較正後の標準偏差比は各ドメインの cv_AP と強く単調に対応し（Spearman ρ=0.952, p=2.3e-5），
   最も強く圧縮されたのは cv_AP 最下位の medical（0.224 倍）と下位の education（0.329 倍），
   すなわち持ち上げたかった当の 2 ドメインだった．**較正が揃えるのはスコアの「水準」だが，
   行内 `max` で決まる rank_2 の勝敗を左右するのは「可動域（分散）」である**．
   水準を揃える変換は，可動域の格差をむしろ拡大しうる．ドメイン間比較を公平にしたいなら，
   較正ではなく順位変換（ドメイン別の経験分位点への写像）など**分散を揃える**方向の変換が要る．
3. **「発火した」ことと「意図した方向に動いた」ことは別**．全体では 284/1600 行の rank_2 が動いたのに，
   目的行（education 20・medical 28）での discordant は各 1 件のみで，発火はほぼ全て目的行の外側だった．
   S4（発火の証拠）型のアサーションは no-op 検出には有効だが，効果の方向の根拠にはならない．
   今後は「目的行に限った発火量」も併せて事前登録するのが妥当である．
4. **education の rank_2 出力を増やしても被覆は増えない**．較正後 education の rank_2 獲得は
   130→144 行（+14）だが `expected_domains` に education を含む行は 9→11（+2）で，
   rank_2 精度 11/144=7.6% は 10 ドメイン中最低（legal は 37/102=36.3%）．
   「出力量を戻せば被覆が戻る」という基準線（421 行の過剰出力）の再現発想は，精度を伴わない限り無効である．
5. **このレバー系統（rank_2 のスコア変換）は収束と判断する**．較正はドメイン内単調変換であるため
   rank_1・各ドメインの ROC-AUC を原理的に変えず，効果はドメイン間比較のみに現れるが，その現れ方が
   上記 2 のとおり目的と逆である．追加反復で符号が反転する種類のばらつきではない．
   なお目的指標の n が 20/28 と小さく ±2 件程度の真の効果を検出する力がない点は，
   この測定基盤自体の制約として残る（S5 のような 2 ドメイン固定の主基準は今後 n の小ささを前提に設計する）．

**次イテレーション（Iter63）の方針**

`multilabel_rank2_score_calibration` は values 単一値のためクローズ（試し切り）．config.yml の既存
levers も実質すべて試し切り済みのため，skill の停止条件 1 に従い，本レバーの note が挙げた
「不成立の場合」の選択肢のうち **(ii) rank_1 側の改善**に対応する新レバー
**`rank1_source = multilabel_head_argmax`** を考案し config.yml の levers 末尾へ追記した
（(i) は R-A 型リーク再発リスクが高く，(iii) は人間判断を要するため見送り．詳細と根拠は backlog B96）．
rank_1 は複合 100 行で 41/100 しか正解せず `compound_domain_set_recall` の上限を 0.705 に固定して
いるのに対し，rank_2 側は今回の結果で飽和が確認されたため，残るボトルネックは rank_1 側である．
イテレーション名は「**rank_1 の選択元を多ラベルヘッドの argmax へ切り替える**」．

**コミット**: `927e363`

## Iteration 61: 2ドメイン合成訓練事例のペア配分の均一化

<!-- 2026-09-19 Iter62 reflector が補填した見出し（Iter61 の各フェーズで追加されておらず，
     journal のローテーションが機能しない状態になっていたため） -->

### 調査 (Iter61)

**問い**（config.yml が事前登録した「実施方法」が，計画フェーズを待たず即座に実験可能かのコードベース
確認．先行研究の新規調査は不要と明示されているため実施しなかった）

- Q1: `scripts/generate_multidomain_training_examples.py` に `--per-pair`/`--per-pair-legal` 相当の
  オプションが既に実装されているか．Iter60 ではどう呼ばれたか．
- Q2: `scripts/train_multilabel_dispatch_head.py`／`scripts/evaluate_dispatch_candidate_ranking.py`／
  `scripts/compute_iter59_ranking_stats.py` が実在し，Iter60 でどう呼ばれたか．
- Q3: `results/iter60_multilabel_ranking_predictions.jsonl` が実在し，A5 の読み替え（対 Iter59 → 対
  Iter60）に必要なスキーマを満たすか．
- Q4: Iter60 で固定するとされるフィルタ F1〜F4・生成モデル・temperature=0.8 が，再生成コマンドで
  意図せず変わらないことをコード上で確認できるか．

**分かったこと（全文読了・grep・実ファイル確認による一次情報）**

1. **`scripts/generate_multidomain_training_examples.py`**（346行，全文読了）: `--per-pair`
   （`_parse_args():311`，デフォルト `_DEFAULT_ROWS_PER_PAIR=3`）・`--per-pair-legal`
   （`:312`，デフォルト `_DEFAULT_ROWS_PER_LEGAL_PAIR=5`）が**既に実装済み**で，`_rows_for_pair()`
   （`:162-166`）が `"legal" in (domain1, domain2)` で振り分ける．Iter60 は実際に
   `--per-pair 3 --per-pair-legal 5` で呼ばれていた（journal Iter60 該当箇所，`git log` の
   `b36cc3f` の親コミット時点のコードと一致）．したがって Iter61 の計画が指示する
   `--per-pair 3 --per-pair-legal 3` は**既存引数への値変更のみ**で実現でき，スクリプト改修は不要．
   F1〜F4 フィルタは `_passes_filters()`（`:128-139`）に定数化されている（F1: 長さ
   `_MIN_QUERY_LENGTH=20`〜`_MAX_QUERY_LENGTH=200`．F2: 四択マーカー `_FOUR_CHOICE_MARKERS`
   （`A.`〜`D.`，全角含む）不在．F3: 完全重複でない（`already_generated` セット）．F4: 改行を含まない
   単一行）．生成 temperature は `_GENERATION_TEMPERATURE=0.8`（`:60`）としてモジュール定数化され，
   `_generate_one()`（`:142-159`）の既定引数として渡る．**引数化されておらず，`--per-pair` 変更では
   一切変わらない**ことを確認した．生成モデル名はハードコードされておらず `--model` 引数
   （`:308`，必須）で渡す設計であり，モジュール docstring（`:29-36`）に Iter60 で使った実行例が
   `--model schroneko/llama-3.1-swallow-8b-instruct-v0.1:q4_k_m` として明記されている．
   `config.yaml:107` の `judge_model` も同じ文字列であり（後述4），**Iter61 実行者がこの docstring の
   コマンドをコピーして `--per-pair`/`--per-pair-legal` の数値だけ変えれば，モデル・temperature は
   自動的に Iter60 と同一のまま**になる．
2. **`scripts/train_multilabel_dispatch_head.py`**（278行，全文読了）: `--train-data`・
   `--multilabel-train-data`・`--embedding-model`・`--ollama-host`／`--ollama-port`・`--output` を
   受け取る CLI が実装済み．`MultiLabelBinarizer` で真の多ラベル目的変数 `Y` を作り，
   `OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight="balanced"))` を fit する
   （`train_multilabel_ranking_head():145-157`，Iter59 と同一の推定器設定）．A0 相当の
   `_assert_a0_true_multilabel_signal()`（`:104-133`）が
   `(Y.sum(axis=1)>=2).sum() == n_synthetic_rows` かつ床 `_A0_MINIMUM_MULTILABEL_ROW_COUNT=120`
   （据え置き，config.yml の指示と一致）を検査する．**Iter61 は `--per-pair 3 --per-pair-legal 3`
   （135件）で呼んでも，このスクリプト自体は無改造でそのまま使える**（合成行数が
   `n_synthetic_rows=135` に変わるだけで，アサーションのロジックは行数に依存しない）．
3. **`scripts/evaluate_dispatch_candidate_ranking.py`／`scripts/compute_iter59_ranking_stats.py`**:
   両方とも実在（`ls -la` で確認，最終更新 2026-09-19 05:40／02:37）．`evaluate_dispatch_candidate_
   ranking.py` の `--iter59-predictions`（`:441-443`）は**引数名が "iter59" だが実装は汎用**で，
   `_compute_a5_iter59_disagreement()`（`:316-337`）は指定した JSONL を `id` でインデックスし
   `row["rank2_new"] != <指定ファイル>[row["id"]]["rank2_new"]` を数えるだけである．
   **config.yml が指示する「A5 を対 Iter60 予測に読み替える」は，このフラグに
   `results/iter60_multilabel_ranking_predictions.jsonl` を渡すだけで実現できる**
   （スクリプト改修は不要）．`compute_iter59_ranking_stats.py` は `scipy.stats.binomtest` による
   exact McNemar（`_exact_mcnemar_binomtest():88-115`）を実装し，Iter60 でも無変更のまま流用された．
4. **`results/iter60_multilabel_ranking_predictions.jsonl`**: 実ファイルとして現存（981,340 bytes，
   1600行，`wc -l` で確認）．1行目を実際に読み，`id`／`expected_domains`／`selected_domain`／
   `dispatched_domains`／`head_scores`／`rank2_baseline`／`rank2_new` の全フィールドを確認した．
   `rank2_new` フィールドがあるため，上記3の A5 読み替えに必要なスキーマを満たしている．
5. **固定パラメータの箇所**: `config.yaml:4` `embedding_model: nomic-embed-text`，`config.yaml:107`
   `judge_model: schroneko/llama-3.1-swallow-8b-instruct-v0.1:q4_k_m`．`git diff config.yaml` で
   現状の未コミット差分を確認したところ，変更は `central_router.embed_node_host`（`wafl502→
   wafl-ctrl5`）の1行のみで，`embedding_model`／`judge_model` の値は Iter60 実行時から変わっていない
   （この diff は research cycle と無関係な既存差分であり，journal Iter60 実装フェーズの記述
   「既存の無関係な未コミット差分1行のみ残存」と一致する）．
6. **journal.md Iter60 該当箇所（実装・実験フェーズ）に記録された実行コマンド一式**（本節末尾の
   「Iteration 60」ブロック，実装フェーズ「実機での動作確認」節および実験フェーズ節）を確認し，
   Iter61 で流用すべきテンプレートとして以下を特定した:
   - 生成: `uv run python -m scripts.generate_multidomain_training_examples --train-data
     data/classifier_train.jsonl --model schroneko/llama-3.1-swallow-8b-instruct-v0.1:q4_k_m
     --ollama-host 127.0.0.1 --ollama-port 11435 --per-pair 3 --per-pair-legal 3 --output
     <Iter61用の新規パス，例: data/classifier_train_multidomain_iter61.jsonl>`
     （Iter60 の出力 `data/classifier_train_multidomain.jsonl` を上書きしないよう別名にすることを
     推奨．A0 の対照値として Iter60 のファイルが必要なため）．
   - A7監査: 同ファイルに対し `--audit-leak` を追加実行．
   - 訓練: `train_multilabel_dispatch_head.py` を `--multilabel-train-data` に上記新規パスを渡し，
     `--output` も新規パス（例: `models/dispatch_multilabel_head_iter61.joblib`）で呼ぶ．
   - 採点: `evaluate_dispatch_candidate_ranking.py` を `--head` に上記新モデル，
     `--embedding-cache results/iter59_query_embeddings.npz`（Iter60 と同一キャッシュ，完全ヒットの
     はず），`--iter59-predictions results/iter60_multilabel_ranking_predictions.jsonl`（A5 読み替え
     の実体），`--output results/iter61_multilabel_ranking_predictions.jsonl` で呼ぶ．
   - 統計: `compute_iter59_ranking_stats.py --baseline results/20260918_202613/results.jsonl --new
     results/iter61_multilabel_ranking_predictions.jsonl --output results/iter61_stats.json`（無変更
     のまま流用）．

**結論**

config.yml が事前登録した「実施方法」1〜4 の全構成要素（`--per-pair`/`--per-pair-legal` 引数・
訓練スクリプト・採点スクリプト・統計スクリプト・A5 の読み替え対象ファイル・F1〜F4/生成モデル/
temperature の固定箇所）が，スクリプトの現物・実ファイル・`git diff` により実在・実装済みであることを
確認した．**新規のコード実装は不要であり，次の「検討・計画」フェーズは値変更（`--per-pair-legal
5→3`）と出力パスの命名だけを決めれば，実装フェーズへ直行できる状態にある**（＝「即座に実験可能」）．
唯一，計画フェーズで明示すべき運用上の注意点は，Iter60 の生成物
（`data/classifier_train_multidomain.jsonl`／`models/dispatch_multilabel_head.joblib`）を
**上書きせず別名で保存すること**（A0 の対照や事後の再現性確認に Iter60 側の実ファイルが必要なため）．

**次フェーズへの示唆**

- レバーは config.yml の指示どおり `multilabel_pair_allocation`（値 `uniform_three_per_pair`）で確定
  でよい．計画フェーズが決めるべきは「出力ファイル名の命名規則」と「Iter60 生成物との共存方法」の
  2点のみで，アルゴリズム的な選択の余地はない（単一レバー原則が実質的にコード上でも保証されている）．
- 生成の乱数性（temperature=0.8）により135件の中身はIter60の153件と一致しないため，A0の
  `n_synthetic_rows`は135に変わる点を計画書に明記すること．
- Iter60 実装フェーズが申し送った「低品質行（プロンプトのテンプレート文言のecho）の混入」は
  F1〜F4のいずれのフィルタにも掛からず通過する既知の穴であり，本イテレーションでも同じ穴が残る
  （単一レバー原則によりフィルタ自体は変更しないため）．次フェーズはこの点を承知の上で進め，
  分析フェーズで低品質行の混入率を確認する申し送りをIter60から継続すること．

### 計画 (Iter61)

**仮説**

Iter60 で観測された compound_domain_set_recall 0.345→0.480（exact p=0.000142）の改善が，
「2 ドメイン合成訓練事例を多ラベル教師として与えること」という**設計一般の効果**であるなら，
評価集合 `_COMPOUND_QUESTIONS` のペア別件数分布（legal×medical が最多）に合わせた
`legal` 絡み 9 ペアのみ 5 件という優遇配分を取り除いても，効果の相当部分が残るはずである．
逆に，優遇配分を外した途端に効果が消失するなら，Iter60 の改善はテスト集合由来の設計情報
（設計レベルの弱いリーク）に相当程度帰属することになる．

**単一レバー（今回変更する唯一の変数）**

`multilabel_pair_allocation`: `legal_weighted_five`（Iter60 の実質的な値，
`--per-pair 3 --per-pair-legal 5` ＝ 45 ペア中 9 ペアのみ 5 件，計 153 件）
→ **`uniform_three_per_pair`（`--per-pair 3 --per-pair-legal 3` ＝ 45 ペア一律 3 件，計 135 件）**．

変更箇所は `scripts/generate_multidomain_training_examples.py` の CLI 引数
`--per-pair-legal` に渡す値のみ（`5` → `3`）で，**スクリプトの改修は一切行わない**．
`_DEFAULT_ROWS_PER_LEGAL_PAIR` 等のモジュール定数もソース上は変更しない（CLI 実引数で上書きする）．

**固定する構成（Iter60 から一切変えない）**

- 生成プロンプト（`generate_multidomain_training_examples.py` 内），フィルタ F1〜F4
  （`_passes_filters():128-139`），生成 temperature（`_GENERATION_TEMPERATURE=0.8`，`:60`，
  モジュール定数のため `--per-pair*` 変更では変化しない），生成モデル
  （`schroneko/llama-3.1-swallow-8b-instruct-v0.1:q4_k_m`，`config.yaml:107` の `judge_model` と同一）．
- ヘッド構造（`OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight="balanced"))`），
  埋め込みモデル（`nomic-embed-text`，`config.yaml:4`），単一ラベル訓練データ
  `data/classifier_train.jsonl`（1427 行）．
- 採点スクリプト `evaluate_dispatch_candidate_ranking.py`・統計スクリプト
  `compute_iter59_ranking_stats.py`（**いずれも無改造で流用**），埋め込みキャッシュ
  `results/iter59_query_embeddings.npz`．
- 基準線 `results/20260918_202613/results.jsonl`（固定 k=2，compound_domain_set_recall 0.345）．
- 実行時経路への配線は本イテレーションでも行わない（`config.yaml` は無変更．スキーマ変更を伴う
  配線は B94 の要レビュー項目としてユーザー確認待ち）．

**出力ファイル命名（Iter60 の生成物を上書きしないこと）**

Iter60 側の実ファイルは A0 の対照・A5 の比較対象・事後の再現性確認に必要なため，
すべて `iter61` サフィックス／プレフィックスの新規パスへ書き出す．

| 種別 | Iter60（保護・読み取り専用） | Iter61（新規作成） |
|---|---|---|
| 合成訓練データ | `data/classifier_train_multidomain.jsonl` | `data/classifier_train_multidomain_iter61.jsonl` |
| ヘッド | `models/dispatch_multilabel_head.joblib` | `models/dispatch_multilabel_head_iter61.joblib` |
| 予測 | `results/iter60_multilabel_ranking_predictions.jsonl` | `results/iter61_multilabel_ranking_predictions.jsonl` |
| 統計 | `results/iter60_stats.json` | `results/iter61_stats.json` |

**実行コマンド（引数名は該当スクリプトの `argparse` を実読して確認済み）**

`--ollama-host` は稼働中ノードに合わせる．Iter60 実績では生成・訓練時が
`--ollama-host 192.168.15.100`（既定ポート 11434），採点時が SSH ローカルフォワード経由の
`--ollama-host 127.0.0.1 --ollama-port 11435` であった．疎通する方を使ってよい
（埋め込み／生成の同一性はモデル名で担保されるため，ホスト指定は単一レバー原則に抵触しない）．

```
# 0) 合成訓練データの再生成（唯一のレバー変更点: --per-pair-legal 5 → 3．135 件目標）
uv run python -m scripts.generate_multidomain_training_examples \
    --train-data data/classifier_train.jsonl \
    --model schroneko/llama-3.1-swallow-8b-instruct-v0.1:q4_k_m \
    --ollama-host 192.168.15.100 \
    --per-pair 3 --per-pair-legal 3 \
    --output data/classifier_train_multidomain_iter61.jsonl

# 0') リーク監査（A7．生成物の選別は行わず，近似重複の検出のみ）
uv run python -m scripts.generate_multidomain_training_examples --audit-leak \
    --output data/classifier_train_multidomain_iter61.jsonl

# 1) 多ラベルヘッドの訓練（1427 + 135 件 embed）
uv run python -m scripts.train_multilabel_dispatch_head \
    --train-data data/classifier_train.jsonl \
    --multilabel-train-data data/classifier_train_multidomain_iter61.jsonl \
    --embedding-model nomic-embed-text --ollama-host 192.168.15.100 \
    --output models/dispatch_multilabel_head_iter61.joblib

# 2) 1600 問のオフライン採点（キャッシュ完全ヒットの想定＝embed 呼び出し 0 件）
#    --iter59-predictions は引数名が "iter59" だが実装は汎用（_compute_a5_iter59_disagreement():316-337）．
#    config.yml の事前登録どおり A5 を「対 Iter60 不一致」に読み替えるため Iter60 の予測を渡す．
uv run python -m scripts.evaluate_dispatch_candidate_ranking \
    --baseline results/20260918_202613/results.jsonl \
    --head models/dispatch_multilabel_head_iter61.joblib \
    --embedding-model nomic-embed-text --ollama-host 127.0.0.1 --ollama-port 11435 \
    --embedding-cache results/iter59_query_embeddings.npz \
    --iter59-predictions results/iter60_multilabel_ranking_predictions.jsonl \
    --output results/iter61_multilabel_ranking_predictions.jsonl

# 3) 指標・検定（Iter59/60 と同一スクリプト・同一手続き．無改造）
uv run python -m scripts.compute_iter59_ranking_stats \
    --baseline results/20260918_202613/results.jsonl \
    --new results/iter61_multilabel_ranking_predictions.jsonl \
    --output results/iter61_stats.json
```

**成功条件（config.yml lever `multilabel_pair_allocation` の事前登録どおり．事後変更禁止）**

- **S1（主基準）**: ドメイン単位 n=200 の exact McNemar（`scipy.stats.binomtest`，α=0.05）で
  **p < 0.05**．
- **S2（効果量下限）**: compound_domain_set_recall が基準線 0.345 に対し **+0.04pt 以上**
  （Iter59/60 と同一に据え置く．Iter60 の +0.135pt を基準にした引き上げは事後的な基準変更に
  当たるため行わない）．
- **S3（コスト中立）**: `mean_dispatch = 2.000000`（完全一致）．
- **S4（発火の証拠）**: `rank2_flip_rate > 0` かつ **対 Iter60 不一致 > 0**．

**非退行条件**

- **N1**: rank_1 が基準線と 1600/1600 で一致．
- **N2**: `top1_accuracy = 0.5975` に完全一致．
- **N3**: legal 自身の被覆 **≧ 8/30**（`_compute_n3():228-250` が自動判定）．
- **N5**: 単一ドメイン 1500 行の argmax 正解率 **≧ 0.590**（採点スクリプトが出力）．
- **N6（新設・報告義務のみ，gate ではない）**: **education 自身の被覆を必ず報告する**
  （Iter60 で 9/20→4/20 の退行が観測されたため）．`compute_iter59_ranking_stats.py` は
  education の内訳を出力しないため，`_domain_pair_coverage_maps()` を読み取り専用で呼び出す
  アドホックな集計で算出してよい（採点・統計の公式パスには手を入れないこと）．

**アサーション（no-op 対策．A0・A5 以外は Iter60 の定義をそのまま使用）**

- **A0（教師信号が真に多ラベル）**: `train_multilabel_dispatch_head.py` の
  `_assert_a0_true_multilabel_signal():104-133` が
  `(Y.sum(axis=1) >= 2).sum() == n_synthetic_rows` を検査する．**今回の期待値は 135**
  （Iter60 は 153）．床 `_A0_MINIMUM_MULTILABEL_ROW_COUNT=120` は据え置き．
  `len(mlb.classes_) == 10` と被覆ペア数（期待 45/45）も併せて報告する．
- **A1**: rank_1 1600/1600 一致（採点スクリプトが assert）．
- **A2**: 全行 k=2・rank_1 ≠ rank_2・mean dispatch = 2.000000．
- **A3**: 対 baseline `rank2_flip_rate > 0`．
- **A5（読み替え）**: 対 **Iter60** 予測（`results/iter60_multilabel_ranking_predictions.jsonl`）の
  rank_2 不一致件数 **> 0**（0 件なら本レバーの no-op を意味するため WARNING）．
- **A6**: `head_scores` のキーが 10 ドメイン名文字列（整数インデックスでない）．
- **A7**: 生成物と `build_dataset._COMPOUND_QUESTIONS` の 3-gram Jaccard 最大値が閾値 0.9 未満
  （Iter60 実績 0.1667）．

**期待効果と事前登録済みの判定規則**

第 2・第 3 の参照点として Iter59（単一ラベル教師，0.350）と Iter60（legal 優遇あり，0.480）を
必ず併記し，「legal 優遇というテスト集合由来の設計情報を取り除いた場合に効果がどれだけ残るか」
として解釈する．

- **S1〜S4 全充足** → Iter60 の adopted を「汎化可能な効果」へ格上げし，効果量の正式値を
  本イテレーションの値に置き換える．
- **S1 不成立だが S2 相当（+0.04pt 以上）は成立** → partial とし，Iter60 の対外記述に
  「legal 絡みの厚い配分に依存する」という注記を恒久的に付す．
- **S1・S2 とも不成立** → Iter60 の効果は設計リーク（配分）に相当程度帰属すると結論し，
  Iter60 の adopted の効力範囲を「legal を含むペアに限定した改善」へ縮小する
  （Iter60 自体の判定は事前登録どおり adopted のまま据え置き，主張の強度だけを落とす）．

**単一レバー原則の確認（混入チェック）**

生成プロンプト・F1〜F4・temperature・生成モデル・ヘッド構造・埋め込みモデル・推論経路・
採点スクリプト・統計スクリプト・基準線・`config.yaml` のいずれにも変更を加えない．
変更は `--per-pair-legal` の実引数値（5→3）と，Iter60 生成物を保護するための出力パス名のみ．
出力パス名の変更は測定対象に影響しない（ファイル I/O のみ）ため単一レバー原則に抵触しない．

**既知の制約（申し送り）**

- temperature=0.8 の生成乱数性により，135 件の本文は Iter60 の 153 件と同一にはならない．
  これは「効果が個々の生成文ではなく配分設計に帰属するか」を見る本イテレーションの目的上，
  むしろ望ましい（Iter60 の該当 135 件を再利用する形は取らない）．
- Iter60 実装フェーズが申し送った「低品質行（プロンプトのテンプレート文言の echo）の混入」は
  F1〜F4 のいずれにも掛からない既知の穴であり，本イテレーションでも同じ穴が残る
  （単一レバー原則によりフィルタは変更しない）．分析フェーズで混入率を報告すること．

### 実装 (Iter61)

**検証内容と結果**

計画フェーズが前提とした4スクリプトの CLI 引数を，`grep -n "add_argument"` と該当行の `Read` で
実物確認した．

1. `scripts/generate_multidomain_training_examples.py`（`_parse_args():295-320`）: `--per-pair`
   （`type=int, default=_DEFAULT_ROWS_PER_PAIR=3`，`:311`）・`--per-pair-legal`
   （`type=int, default=_DEFAULT_ROWS_PER_LEGAL_PAIR=5`，`:312`）・`--output`（`required=True`，`:313`）・
   `--audit-leak`（`action="store_true"`，`:314-319`）・`--train-data`（`default="data/classifier_
   train.jsonl"`）・`--model`／`--ollama-host`／`--ollama-port`（`default=11434`）を計画どおり実装
   済みと確認した．`main():323-341` は `--audit-leak` 指定時は生成せず監査のみ行い，通常時は
   `_generate_and_save(...)` を呼ぶ（`:280` で `open(output_path, "w", ...)`）．出力先の重複チェックは
   実装されていない（無条件に上書き）が，計画どおり Iter61 専用の新規パスを渡すため実害はない．
2. `scripts/train_multilabel_dispatch_head.py`（`_parse_args():230-258`）: `--train-data`
   （`required=True`）・`--multilabel-train-data`（`required=True`）・`--embedding-model`
   （`required=True`）・`--ollama-host`（`required=True`）・`--ollama-port`（`default=11434`）・
   `--output`（`default="models/dispatch_multilabel_head.joblib"`）を確認した．`:222` に
   `os.makedirs(output_dir, exist_ok=True)` があり，出力先ディレクトリ未存在時も自動作成される
   （今回は `models/` が既存のため実際には発火しない）．
3. `scripts/evaluate_dispatch_candidate_ranking.py`（`_parse_args():415-447`）: `--baseline`
   （`required=True`）・`--head`（`required=True`）・`--embedding-model`（`required=True`）・
   `--ollama-host`（`required=True`）・`--ollama-port`（`default=11434`）・`--embedding-cache`
   （`default=None`）・`--output`（`required=True`）・`--iter59-predictions`（`default=None`，引数名は
   "iter59" だが実装は任意の JSONL パスを受け取る汎用実装）を確認した．`main():450-459` は
   `open(args.output, "w", ...)` で出力ファイルを開く．
4. `scripts/compute_iter59_ranking_stats.py`（`_parse_args():356-373`）: `--baseline`
   （`required=True`）・`--new`（`required=True`）・`--output`（`required=True`）を確認した．
   統計計算は `_exact_mcnemar_binomtest()`（`scipy.stats.binomtest` 使用，計画同様に既存実装を
   再利用するのみで手を触れない）．

**結論: コード改修は不要**

4スクリプトとも計画フェーズが前提とした引数名・型・デフォルト値と完全に一致した．出力先の重複
チェックは存在しないが，これは「実装済みコードの欠陥」ではなく単に無条件上書きの仕様であり，
Iter61 専用の新規パスを渡すことで安全に回避できるため，コード改修は行わなかった（生成プロンプト・
F1〜F4・ヘッド構造・推論経路・採点/統計スクリプトのロジック・基準線は無変更）．

**事前準備の確認**

- `data/`・`models/`・`results/` はいずれも既存ディレクトリであり，新規作成は不要だった．
- Iter61 の4出力先（`data/classifier_train_multidomain_iter61.jsonl`・
  `models/dispatch_multilabel_head_iter61.joblib`・
  `results/iter61_multilabel_ranking_predictions.jsonl`・`results/iter61_stats.json`）は，いずれも
  実行前に存在しないことを `ls -e` 相当の存在確認で確認した（重複なし）．
- Iter60 の生成物（`data/classifier_train_multidomain.jsonl`・`models/dispatch_multilabel_head.joblib`・
  `results/iter60_multilabel_ranking_predictions.jsonl`・`results/iter60_stats.json`）の `mtime` を
  実装フェーズ実行前に記録し，本フェーズでは一切変更・削除していないことを確認した（`ls -la` の
  タイムスタンプは journal Iter60 実験フェーズ記録時点から不変）．
- 採点で使う埋め込みキャッシュ `results/iter59_query_embeddings.npz`（9,971,712 bytes）と基準線
  `results/20260918_202613/results.jsonl`（3,510,699 bytes）が実在することを確認した．
- `git status --short` で，本フェーズ着手前からの未コミット差分（`config.yaml` の
  `embed_node_host: wafl502→wafl-ctrl5` 1行，`results/iter45_preliminary/logs/*` のログ更新，
  research-cycle 管理ファイル群）を確認したが，いずれも本イテレーションと無関係であり，
  指示（作業前から存在する未コミット差分は触らない）に従い変更していない．

**実験フェーズへの申し送り**

コード変更なしで実験フェーズにそのまま進める状態である．計画フェーズが `### 計画 (Iter61)` に
記載した4コマンド（生成→A7監査→訓練→採点→統計）をそのまま実行してよい．

### 実験 (Iter61)

**接続先ホストの読み替え（計画書の想定値が現在不通であることを確認した上での変更）**

計画書は生成・訓練を `--ollama-host 192.168.15.100`（既定ポート），採点を
`--ollama-host 127.0.0.1 --ollama-port 11435` で想定していた．実行前に到達性を確認した結果:

- `ping -c 2 -W 2 192.168.15.100` → 100% packet loss．`curl -m 5 http://192.168.15.100:11434/api/tags`
  → タイムアウト（`exit=28`，直接 TCP 到達不可）．
- `ssh wafl500`（同じホストへの SSH 接続）は正常に成功し，`wafl500` 上でのローカル
  `curl http://127.0.0.1:11434/api/tags` も応答した．すなわち ICMP／直接 TCP は不通だが SSH 経由の
  到達は可能という状態．
- 既存の稼働中 SSH ローカルポートフォワード（`ssh -fNT -L 11435:localhost:11434 wafl500`，実行前から
  稼働）経由の `curl http://127.0.0.1:11435/api/tags` は応答し，モデル一覧に
  `schroneko/llama-3.1-swallow-8b-instruct-v0.1:q4_k_m` と `nomic-embed-text:latest` の両方が含まれる
  ことを確認した．

したがって**生成・訓練・採点の3ステップすべてで `--ollama-host 127.0.0.1 --ollama-port 11435` に
統一した**（計画書は生成・訓練を直接 IP，採点を SSH フォワード経由と分けていたが，直接 IP が
不通のため単一の到達可能な経路に揃えた．レバー（`--per-pair-legal`）以外のパラメータは変更しておらず，
接続先ホストは指示どおり環境要因として可到達性を優先した）．

**実行した5コマンド（実際のホスト・ポート）**

```
# 1) 合成訓練データ生成
uv run python -m scripts.generate_multidomain_training_examples \
    --train-data data/classifier_train.jsonl \
    --model schroneko/llama-3.1-swallow-8b-instruct-v0.1:q4_k_m \
    --ollama-host 127.0.0.1 --ollama-port 11435 \
    --per-pair 3 --per-pair-legal 3 \
    --output data/classifier_train_multidomain_iter61.jsonl

# 2) A7リーク監査
uv run python -m scripts.generate_multidomain_training_examples --audit-leak \
    --output data/classifier_train_multidomain_iter61.jsonl

# 3) 多ラベルヘッド訓練
uv run python -m scripts.train_multilabel_dispatch_head \
    --train-data data/classifier_train.jsonl \
    --multilabel-train-data data/classifier_train_multidomain_iter61.jsonl \
    --embedding-model nomic-embed-text --ollama-host 127.0.0.1 --ollama-port 11435 \
    --output models/dispatch_multilabel_head_iter61.joblib

# 4) 1600問オフライン採点
uv run python -m scripts.evaluate_dispatch_candidate_ranking \
    --baseline results/20260918_202613/results.jsonl \
    --head models/dispatch_multilabel_head_iter61.joblib \
    --embedding-model nomic-embed-text --ollama-host 127.0.0.1 --ollama-port 11435 \
    --embedding-cache results/iter59_query_embeddings.npz \
    --iter59-predictions results/iter60_multilabel_ranking_predictions.jsonl \
    --output results/iter61_multilabel_ranking_predictions.jsonl

# 5) 指標・検定
uv run python -m scripts.compute_iter59_ranking_stats \
    --baseline results/20260918_202613/results.jsonl \
    --new results/iter61_multilabel_ranking_predictions.jsonl \
    --output results/iter61_stats.json
```

**1) 生成結果**

`[generate_multidomain_training_examples] wrote 135 rows to data/classifier_train_multidomain_iter61.jsonl
(domains=['business_economics', 'computer_science', 'education', 'general', 'history_culture', 'legal',
'mathematics', 'medical', 'natural_science', 'social_science'])`．135/135 件が欠番なく生成された．
生成物を独立に集計したところ，45 ペア全てが正確に 3 件ずつ（`Counter({3: 45})`）で，計画どおり
「45 ペア一律 3 件」の均一配分になっていることを確認した．

**2) A7（リーク監査）結果**

`max_jaccard=0.12903225806451613`，`median_max_jaccard=0.06542056074766354`．
`[generate_multidomain_training_examples] A7 leak audit PASS (max_jaccard=0.1290 < 0.9)`（閾値0.9未満）．

**3) 訓練結果**

`[train_multilabel_dispatch_head] A0 PASS: 135 multi-label rows covering 45 distinct domain pairs`．
5-fold CV 診断（全ドメイン，`n_positive`・`cv_roc_auc`・`cv_average_precision`）:

| domain | n_positive | cv_roc_auc | cv_average_precision |
|---|---|---|---|
| business_economics | 177 | 0.8143 | 0.4244 |
| computer_science | 177 | 0.9057 | 0.6116 |
| education | 177 | 0.8531 | 0.4407 |
| general | 177 | 0.8831 | 0.6232 |
| history_culture | 177 | 0.9353 | 0.7634 |
| legal | 104 | 0.9089 | 0.5989 |
| mathematics | 177 | 0.9276 | 0.7411 |
| medical | 177 | 0.7817 | 0.3915 |
| natural_science | 177 | 0.8361 | 0.4512 |
| social_science | 177 | 0.8629 | 0.5815 |

`wrote models/dispatch_multilabel_head_iter61.joblib (n_single_label_rows=1427, n_synthetic_rows=135,
classes=[全10ドメイン名])`．legal の `n_positive=104`（単一ラベル77＋legal絡み合成27=9ペア×3件）．

**4) 採点結果（`evaluate_dispatch_candidate_ranking.py` 標準出力の JSON，キャッシュ完全ヒットで
embed 呼び出し0件）**

```json
{
  "n_rows": 1600,
  "mean_dispatch": 2.0,
  "rank2_flip_rate": 0.46,
  "compound_domain_set_recall": 0.445,
  "compound_rows_evaluated": 100,
  "n5_single_domain_argmax_accuracy": {
    "n_single_domain_rows": 1500, "correct": 905, "accuracy": 0.6033333333333334,
    "floor": 0.59, "pass": true
  },
  "a5_iter59_disagreement": {
    "n_rows": 1600, "mismatches": 552, "mismatch_rate": 0.345
  }
}
```

（`a5_iter59_disagreement` は引数名は "iter59" だが実装は汎用であり，`--iter59-predictions` に
`results/iter60_multilabel_ranking_predictions.jsonl` を渡したことで対 **Iter60** 不一致件数を
算出している．A6：例外なし＝全1600行で `head_scores` のキーが10ドメイン名文字列と完全一致．
A1・A2：例外なし＝rank_1完全一致・mean_dispatch=2.0．）

**5) 指標・検定結果（`results/iter61_stats.json` 全文）**

```json
{
  "baseline_compound_domain_set_recall": 0.345,
  "new_compound_domain_set_recall": 0.445,
  "matches_implementation_phase_diagnostic_recall": false,
  "S1_primary_criterion": {
    "n_pairs": 200, "improved_pairs": 32, "regressed_pairs": 12, "discordant_pairs": 44,
    "chi2_statistic_continuity_corrected": 8.204545454545455,
    "p_value_continuity_corrected": 0.004178557568166319,
    "p_value_exact_binomtest": 0.003657766827927844,
    "pass": true
  },
  "S2_effect_size_floor": {
    "baseline_recall": 0.345, "new_recall": 0.445, "delta_pt": 0.10000000000000003,
    "floor_pt": 0.04, "pass": true
  },
  "S3_cost_neutrality": {
    "n_rows": 1600, "length_distribution": {"2": 1600}, "duplicate_rank1_rank2_count": 0,
    "mean_dispatch": 2.0, "pass": true
  },
  "S4_flip_rate_evidence_of_firing": {
    "n_rows": 1600, "flips": 736, "rank2_flip_rate": 0.46,
    "matches_implementation_phase_value": false, "implementation_phase_value": 0.356875,
    "pass": true
  },
  "N1_rank1_invariance": {"n_rows": 1600, "mismatch_count": 0, "mismatch_ids": [], "pass": true},
  "N2_top1_accuracy_invariance": {
    "baseline_top1_accuracy": 0.5975, "new_top1_accuracy": 0.5975, "exact_match": true, "pass": true
  },
  "N3_legal_non_regression": {
    "n_legal_involving_pairs": 30, "baseline_legal_self_coverage": 8, "new_legal_self_coverage": 15,
    "expected_baseline_value": 8, "baseline_matches_journal_record": true, "pass": true
  },
  "N4_improvement_breakdown_by_domain_category": {
    "legal_involving": {"n_pairs": 60, "improved": 10, "regressed": 2, "unchanged": 48},
    "medical_involving": {"n_pairs": 32, "improved": 1, "regressed": 3, "unchanged": 28},
    "other": {"n_pairs": 108, "improved": 21, "regressed": 7, "unchanged": 80}
  }
}
```

`matches_implementation_phase_*` の2フィールドが `false` になっているのは，本イテレーションが
コード改修を伴わない実験（実装フェーズでの速報実測なし）のため，スクリプト内に残る**Iter59自身の
実装フェーズ速報値定数**（`_IMPLEMENTATION_PHASE_COMPOUND_DOMAIN_SET_RECALL=0.35`・
`_IMPLEMENTATION_PHASE_RANK2_FLIP_RATE=0.356875`）と比較されているためであり，Iter60実験フェーズと
同じ想定どおりの挙動である（異常ではない）．

**N6（報告義務のみ．`_domain_pair_coverage_maps()` を読み取り専用で呼び出すアドホック集計，
採点・統計スクリプト本体は変更せず）**

`scripts/compute_iter59_ranking_stats.py` の `_domain_pair_coverage_maps()` を import し，
`baseline_rows`（`results/20260918_202613/results.jsonl`）と `new_rows`
（`results/iter61_multilabel_ranking_predictions.jsonl`）から education 絡みの20ペアのみを
フィルタして集計した結果:

```json
{
  "n_education_involving_pairs": 20,
  "baseline_education_self_coverage": 9,
  "new_education_self_coverage": 5
}
```

education 自身の被覆は基準線 9/20 → 本イテレーション 5/20（Iter60 は 9/20→4/20）．

**成果物・付随確認**

- 新規4ファイル: `data/classifier_train_multidomain_iter61.jsonl`（135行，sha256
  `29d61a60728c5496ff0025a724b65f7546578ca4a98d0f7a20a11783fdaa1f06`）・
  `models/dispatch_multilabel_head_iter61.joblib`（sha256
  `cd0af5dbebee53c5b4c934a508ed58d2bdb088748bc1f5f222127f4bdb330e56`）・
  `results/iter61_multilabel_ranking_predictions.jsonl`（1600行）・`results/iter61_stats.json`。
- Iter60 の4成果物（`data/classifier_train_multidomain.jsonl`・`models/dispatch_multilabel_head.joblib`・
  `results/iter60_multilabel_ranking_predictions.jsonl`・`results/iter60_stats.json`）の `mtime` を
  実行前後で確認し，一切変更されていないことを確認した（実行前後で同一のタイムスタンプ）。
- `git status --short` で，実行前から存在した無関係な未コミット差分（`config.yaml` の
  `embed_node_host` 1行・`results/iter45_preliminary/logs/*`・研究サイクル管理ファイル群）を確認し，
  本フェーズはこれらに一切触れていない（追加されたのは `results/iter61_*` の新規2ファイルのみ）。
- 実行中の異常・障害は発生していない（ネットワーク接続の読み替えを除き，全5コマンドとも exit code 0，
  エラー出力なし）。

**解釈・採否判断はこのフェーズでは行わない**（次の分析・考察フェーズに委ねる。上記はすべて生の
実測値であり，PASS/FAIL の表記はスクリプト自身が出力した事前登録済みアサーションの結果をそのまま
転記したものである）。

### 分析 (Iter61)

**1. 独立検算（`metrics.py` の既存関数と `scipy.stats.binomtest` のみ．不一致 0 件）**

`results/20260918_202613/results.jsonl`（基準線）・`results/iter61_multilabel_ranking_predictions.jsonl`
（新）・`results/iter60_multilabel_ranking_predictions.jsonl`（対照）を読み直し，
`metrics.compute_compound_coverage_metrics()` に準ずる被覆マップ再構成・`compute_top1_accuracy()`・
`binomtest` のみで `results/iter61_stats.json` の全項目を再計算した（作業用スクリプトは `/tmp` に置き，
リポジトリへは追加していない）．**全項目が完全一致し，不一致は 1 件も無い**．

| 項目 | 独立再計算値 | stats.json |
|---|---|---|
| compound_domain_set_recall（基準線 / Iter60 / 新） | 0.345（69/200） / 0.480（96/200） / **0.445（89/200）** | 一致 |
| S1 ペア比較（n=200） | 改善 32 / 悪化 12 / discordant 44 / exact p=0.00365777 | 一致 |
| S2 効果量 | Δ=+0.10pt（floor +0.04pt） | 一致 |
| S3 コスト | 長さ分布 `{2: 1600}`，rank1=rank2 重複 0，mean_dispatch=2.000000 | 一致 |
| S4 rank2_flip_rate | 0.46（736/1600） | 一致 |
| A5（対 Iter60 不一致） | 552/1600（0.345） | 一致 |
| N1 rank_1 不一致 | 0/1600 | 一致 |
| N2 top1_accuracy | 0.5975 → 0.5975 | 一致 |
| N3 legal 自身の被覆 | 8/30 → 15/30 | 一致 |
| N5 単一ドメイン argmax 正解率 | 0.603333（905/1500，floor 0.590） | 一致 |
| N6 education 自身の被覆 | 9/20 → 5/20 | 一致 |

**2. 事前登録の判定規則への該当（機械的確認．事後の緩和・厳格化はしていない）**

| 条件 | 事前登録の閾値 | 実測 | 判定 |
|---|---|---|---|
| S1 主基準 | ドメイン単位 n=200 exact McNemar，p<0.05 | p=0.003658（改善32/悪化12/discordant44） | **PASS** |
| S2 効果量下限 | Δ ≧ +0.04pt | Δ=+0.10pt（0.345→0.445） | **PASS** |
| S3 コスト中立 | mean_dispatch=2.000000 | 2.000000（全1600行 len=2，重複0） | **PASS** |
| S4 発火の証拠 | rank2_flip_rate>0 かつ対 Iter60 不一致>0 | 0.46（736行）・552行 | **PASS** |
| N1 | rank_1 1600/1600 不変 | 不一致 0 | **PASS** |
| N2 | top1_accuracy 0.5975 完全一致 | 0.5975 | **PASS** |
| N3 | legal 自身の被覆 ≧8/30 | 15/30 | **PASS** |
| N5 | 単一ドメイン argmax ≧0.590 | 0.603333 | **PASS** |
| N6 | 報告義務のみ（gate ではない） | education 9/20→5/20 | （報告済・FAIL 判定の対象外） |
| A0/A1/A2/A3/A6/A7 | 各アサーション | 135/135・45ペア×3・rank_1 不一致0・mean 2.0・flip 0.46・キー全ドメイン名・max_jaccard 0.1290 | 全 PASS |

→ **S1〜S4 が全充足**．事前登録の判定規則（config.yml:891-895）の**第 1 分岐「S1〜S4 全充足なら
Iter60 の adopted を『汎化可能な効果』へ格上げし，効果量の正式値を本イテレーションの値に置き換える」
に該当する**．第 2 分岐（S1 不成立・S2 のみ残る＝partial）・第 3 分岐（S1・S2 とも不成立＝設計リーク
帰属）には該当しない．FAIL は 1 件も無い．

**3. 当初の問いへの答え — 「legal 優遇というテスト集合由来の設計情報を除くと効果はどれだけ残るか」**

3 点の並置（基準線・Iter59・Iter60・Iter61）と，legal 絡みでの分解（本フェーズで追加算出）:

| 条件 | compound_domain_set_recall | Δ（対基準線） | S1 exact p |
|---|---|---|---|
| 基準線（`20260918_202613`） | 0.345 | — | — |
| Iter59（単一ラベル教師・OvR） | 0.350 | +0.005pt | 1.0 |
| Iter60（合成153件，legal絡み9ペアのみ5件） | 0.480 | +0.135pt | 0.000142 |
| **Iter61（合成135件，45ペア一律3件）** | **0.445** | **+0.100pt** | **0.003658** |

| 切り口（ペア数） | Iter60 改善/悪化・p | Iter61 改善/悪化・p | Iter60 Δ | Iter61 Δ |
|---|---|---|---|---|
| 全体（200） | 38/11，p=0.000142 | 32/12，p=0.003658 | +13.5pt | +10.0pt |
| legal 絡み（60） | 17/1，p=0.000145 | 10/2，p=0.0386 | +26.7pt | +13.3pt |
| **legal 絡みを全部除く（140）** | 21/10，**p=0.0708** | 22/10，**p=0.0501** | **+7.9pt** | **+8.6pt** |
| legal×medical のみ（24） | 8/0 | 6/0，p=0.031 | — | — |
| legal×medical を除く（176） | 30/11，p=0.00432 | 26/12，p=0.0336 | — | — |

この分解が本イテレーションの中心的な所見である．

- **効果量の縮小 13.5pt → 10.0pt は，ほぼ全量が legal 絡みペアの寄与の縮小で説明される**．
  加重分解すると Iter60 は (60/200)×26.7 + (140/200)×7.9 = 8.0 + 5.5 = 13.5pt，Iter61 は
  (60/200)×13.3 + (140/200)×8.6 = 4.0 + 6.0 = 10.0pt．**legal を厚く配分した分の上積み
  （legal 絡みで +26.7pt→+13.3pt）が消えた一方，legal 非依存部分は +7.9pt→+8.6pt とむしろ微増**
  （差 +0.7pt は改善 21→22・悪化 10→10 の 1 ペア差に相当し，明白にノイズ範囲）．
- すなわち **R-A（設計リークによる水増し）は「存在したが，その影響範囲は legal 絡みペアに限局し，
  効果の本体（legal 非依存の +8pt 前後）は配分設計に依存していない」**と切り分けられた．
  リークの大きさは全体 200 ペアで約 3.5pt と定量できる．
- **R-B（legal 依存）は緩和したが完全には解消していない**．legal 絡みを除いた p は 0.0708→0.0501 で
  改善したものの，α=0.05 をごくわずかに割らない（有意にならない）．ただし n=140・discordant 32 と
  検出力が低い条件であり，効果量（+8.6pt）と方向は Iter60 と一貫している．**「legal 非依存の効果は
  点推定で +8.6pt あり方向も再現しているが，legal を除いた部分集合だけでは α=0.05 の有意性を
  主張できない」**が正確な記述である．主基準 S1 は全体 200 ペアで定義されており，そこでは
  p=0.003658 で成立している（部分集合検定は事前登録の gate ではなく解釈材料である点に注意）．

**4. Iter61 は Iter60 の独立再現でもある（当初想定していなかった副産物）**

生成の乱数（temperature=0.8）により合成 135 件の本文は Iter60 の 153 件と**同一ではない**．
にもかかわらず recall 0.480 / 0.445，legal 非依存部分 +7.9pt / +8.6pt と同水準が出た．
Iter60 予測と Iter61 予測を直接 McNemar にかけると **改善 18 / 悪化 25，exact p=0.360（有意差なし）**
であり，**2 つの独立な生成サンプル間で効果に統計的な差は検出されない**．
これは「効果は個々の生成文（たまたま当たった文）ではなく，2 ドメイン同時ラベルという教師信号の設計に
帰属する」という Iter60 の因果的主張に対する，**独立サンプルによる再現性の証拠（n=2）**である．
同一指標の履歴は Iter46〜59 で 0.345〜0.360 の ±1.5pt 帯に張り付いていたので，
0.445 と 0.480 はいずれもその帯の外にある．

**5. ノイズか信号か — 信号である**

- **(i) 反復間ノイズはゼロ**: 決定論的オフライン採点で embed はキャッシュフルヒット（実呼び出し0件）．
  config.yml success_criteria (5) の 3SD=2.6pt は軸②③（生成のランダム性）に対するノイズ床であり，
  軸①のルーティング指標には適用しない（同項末尾に明記）．
- **(ii) 標本誤差に対して十分大きい**: ペア差の近似 SE = sqrt(32+12)/200 = **3.32pt**，Δ=+10.0pt は
  **3.02 SE**，95% CI は **[+3.5pt, +16.5pt]**．ただし **CI 下端 +3.5pt は事前登録の効果量下限
  +4.0pt をわずかに下回る**（Iter60 は CI [+6.6pt, +20.4pt] で下端が床の上だった）．
  事前登録 S2 は点推定基準なので PASS で正しいが，**「CI 下端まで床を超える」という Iter60 で
  成立していた強い条件は，今回は成立していない**．この 1 点は正直に記録する．
- **(iii) A5=552/1600（34.5%）という規模**: 配分 1 変数（legal 絡み 9 ペアの 5→3 件＝18 件減，
  合成全体でも 153→135 件）の変更で rank_2 の 3 分の 1 が Iter60 と異なる．
  Iter60 の学び 3「rank_2 の順位付けは教師信号の構成に極端に敏感」がここでも再確認された．
  ただし **flip の量が大きいのに結果指標の差は有意でない（p=0.360）**ため，
  「敏感だが，方向性のある効果は配分に依らず安定」という読みになる．
- **(iv) prior シフト説の排除は継続**: content-blind 対照（rank_2 を内容非依存に固定した最良値）は
  **legal 固定 0.350**・medical 固定 0.310・social_science 固定 0.295 で，実測 0.445 はこれを
  9.5pt 上回る（Iter60 は 13pt）．行ごとの内容に反応しているという結論は変わらない．
- **(v) 上限との距離**: rank_1 のみ（k=1 相当）0.205，固定 k=2・rank_1 凍結下のオラクル上限 0.705．
  残余ギャップ 36.0pt のうち **10.0pt（27.8%）を埋めた**（Iter60 は 37.5%，Iter59 は 1.4%）．

**6. 改善の分散と flip 収支（「単一ペアの偶然」ではないことの確認）**

- 改善 32 ペアは **21 ペア種（45 ペア種中）に分散**（Iter60 は 38 改善／26 ペア種）．最大寄与は
  legal×medical の 6 件（Iter60 は 8 件）で，全改善に占める比率は 21%（Iter60 は 21%）と変わらない．
- compound 100 行の rank_2 的中は **基準線 28 → 48**（Iter60 は 55，Iter59 は 29）．
  基準線から flip した compound 63 行に限れば的中 **12 → 32** で，一方向に的中を増やしている
  （Iter59 の「flip するが収支ゼロ」とは質が異なる）．
- N4（stats.json）の内訳では legal 絡み 60 ペア 改善10/悪化2，medical 絡み（legal×medical を除く
  32 ペア）改善1/悪化3，その他 108 ペア 改善21/悪化7．**medical 側が負の収支**である点は下記 7 で扱う．

**7. 副次的退行 — education（N6）と medical（新規観測）**

ドメイン自身の被覆（compound ペアのうち当該ドメインが dispatch に含まれた件数）の 3 点比較:

| domain | 基準線 | Iter60 | Iter61 |
|---|---|---|---|
| legal | 8/30 | 20/30 | 15/30 |
| medical | 13/28 | 19/28 | **12/28** |
| education | 9/20 | **4/20** | **5/20** |
| social_science | 0/18 | 7/18 | 5/18 |
| general | 1/14 | 5/14 | 7/14 |
| computer_science | 3/18 | 5/18 | 8/18 |
| natural_science | 6/18 | 6/18 | 8/18 |
| business_economics | 13/18 | 11/18 | 13/18 |
| history_culture | 15/18 | 15/18 | 14/18 |
| mathematics | 1/18 | 4/18 | 2/18 |

- **education は 2 イテレーション連続で退行**（基準線 9/20 → Iter60 4/20 → Iter61 5/20）．
  education 絡み 40 ペアの収支は Iter60 改善3/悪化6，Iter61 改善3/悪化4 で，**配分レバーを振っても
  退行は解消していない**．機序は Iter60 分析のとおりで，基準線が rank_2=education を 1600 行中
  **421 行（26.3%）**と過剰に出しており，その過剰さが偶然 education compound 行を拾っていた．
  多ラベルヘッドは rank_2 分布を平坦化する（Iter61 では education 130 行，最大は social_science 207 行）
  ため，過剰出力に依存していた被覆が失われる．**compound 行に限れば rank_2=education は
  基準線 23 → Iter60 2 → Iter61 5**．
  退行幅が -5 / -4 と 2 回とも同程度であり，ヘッド重みの乱数や生成文の違いでは説明しにくい
  **構造的（系統的）な退行**と判断する．N6 は gate ではないので本イテレーションの FAIL にはならないが，
  「改善の裏で特定ドメインが一貫して犠牲になっている」という所見として確定した．
- **medical は今回新たに基準線を下回った**（13/28 → Iter61 12/28．Iter60 は 19/28）．
  legal 絡みの合成を 5→3 件に減らしたことで legal×medical ペアの学習圧が下がり，
  Iter60 で得ていた medical 側の上積み（+6）が失われて基準線をわずかに割った形である．
  N3 は legal のみを gate にしており medical の gate は無いため FAIL ではないが，
  **「legal 優遇の除去は legal 自身（20/30→15/30）だけでなく medical（19/28→12/28）にも波及した」**
  という，R-A の影響範囲を示す追加証拠として記録する．

**8. 仮説との整合**

計画の仮説「legal 優遇というテスト集合由来の設計情報を取り除いても，多ラベル教師信号の効果は
有意に残る」は **支持された**（S1 p=0.003658，Δ=+0.10pt）．想定外の挙動（言語崩れ・発散・OOM・
整数キーバグ A6・no-op・A7 リーク）はいずれも観測されていない．想定と異なった点は 2 つ:
- **想定より効果の縮小が小さかった**: 事前の見立て（Iter60 分析の「5 件配分＝改善17/悪化1，
  3 件配分＝改善21/悪化10 と収支が異なる」）からは，均一化でより大きく落ちる可能性も想定されたが，
  実際には 13.5→10.0pt の縮小に留まり，legal 非依存部分は変化しなかった．
- **legal 自身の被覆が想定以上に残った**: legal 優遇を完全に外したにもかかわらず 8/30→15/30
  （N3 の床 8 に対して十分上）．legal は単一ラベル訓練が 77 件と最少（合成込みで n_positive=104）
  だが，均一配分でも基準線の倍近くまで伸びている．

**9. 判定の確信度と追加反復の要否**

- **確信度: 高**．(i) 決定論的で反復間ノイズゼロ，(ii) 独立検算の不一致 0 件，(iii) S1〜S4 全充足で
  FAIL 0 件，(iv) Iter60 という独立生成サンプルでの再現があり両者に有意差なし（p=0.360），
  (v) content-blind 対照を 9.5pt 上回る．
- **同一設計での追加反復は不要**（決定論的なので同じ値が再現するだけ）．
- **確信度が相対的に低い部分**: (a) 効果量の 95% CI 下端 +3.5pt が事前登録床 +4.0pt をわずかに割る，
  (b) legal 絡みを除くと p=0.0501 で有意水準をぎりぎり割らない（Iter60 の 0.0708 からは改善），
  (c) education の 2 連続退行・medical の基準線割れという局所的な負の効果，
  (d) 実行時経路では未検証（オフライン採点のみ．`config.yaml` スキーマ変更＝要ユーザー確認），
  (e) 合成のプロンプト echo 行（Iter60 で 4.6%）は今回未計測．

**次フェーズ（rc-reflector）への申し送り（採否の最終確定は reflector の役割）**

- **事前登録規則の該当分岐**: **第 1 分岐（S1〜S4 全充足 → Iter60 の adopted を「汎化可能な効果」へ
  格上げ，効果量の正式値を本イテレーションの値へ置換）**．partial 分岐・設計リーク帰属分岐には
  該当しない．
- **効果量の正式値の扱い（提案）**: 正式値を **`compound_domain_set_recall` 0.345 → 0.445
  （Δ=+10.0pt，exact McNemar p=0.003658，95% CI [+3.5pt, +16.5pt]，mean_dispatch 2.000000 でコスト中立）**
  とする．Iter60 の +13.5pt は**テスト集合のペア別件数分布を参照した配分を含む値**であり，
  汎化推定値としては引用せず，「legal 絡みを厚く配分した場合の上限値」として位置づけるのが妥当．
  R-A（設計リーク）の定量は「全体で約 3.5pt，影響は legal 絡み 60 ペアに限局」と記述できる．
- **R-A の帰結**: 解消された．設計リークは存在したが効果の本体を作っていない（legal 非依存部分は
  +7.9pt→+8.6pt で不変）．Iter60 の学び 2（A7 は設計レベルのリークを検出できない）は有効なまま残る．
- **R-B の帰結**: 緩和されたが解消していない．legal 絡みを除くと p=0.0501（+8.6pt）．
  対外記述では「主基準は全体 200 ペアで p=0.0037．legal 絡みを除いた部分集合（n=140）では
  +8.6pt・p=0.0501 で方向は一貫するが有意水準には届かない」と併記するのが正確．
  「legal 絡みの厚い配分に依存する」という注記（partial 分岐で要求されていた文言）は，
  配分を均一化しても効果が残った以上**不要**である（依存していたのは配分ではなく legal ペアの存在）．
- **R-C の帰結**: 悪化方向で確定．education の退行は配分と無関係で 2 連続（-5, -4），
  加えて medical も基準線を割った（13/28→12/28）．**次のレバー候補として申し送るべき**と考える．
  具体案（分析フェーズとしての示唆であり採否判断ではない）: 合成データのドメイン別 positive 件数
  ないし OvR の per-domain しきい値／`class_weight` を，rank_2 の出力分布が基準線の偏り
  （education 26.3%）を潰しすぎないよう補正する 1 変数レバー．education は Iter32〜53 で 10 回以上
  レバーを振っても動かなかった問題ドメインであり，ここで初めて**動かせる（悪い方向にだが）**ことが
  判明した点は情報量が大きい．
- **残る最大のボトルネックは依然 rank_1 側**（compound 行で rank_1 正解 41/100，上限 0.705）．

### Iteration 61 実行済み（考察・次計画）

**単一レバー**: `multilabel_pair_allocation = uniform_three_per_pair`
（`scripts/generate_multidomain_training_examples.py --per-pair 3 --per-pair-legal 3`，
45 ペア一律 3 件＝135 件．legal 優遇 +2 件／9 ペアの除去のみが Iter60 との差分）．

**変更したもの**: 合成訓練データの生成コマンド引数 1 つのみ．生成プロンプト・フィルタ F1〜F4・
ヘッド構造・推論経路・採点スクリプト・統計スクリプト・基準線（`results/20260918_202613/results.jsonl`）
はすべて Iter60 から固定．コード改修は 0 行（実装フェーズで 4 スクリプトの CLI を実物確認した結果，
計画の前提と完全一致していたため）．生成物は
`data/classifier_train_multidomain_iter61.jsonl` / `models/dispatch_multilabel_head_iter61.joblib` /
`results/iter61_multilabel_ranking_predictions.jsonl` / `results/iter61_stats.json`．

**結果**: `compound_domain_set_recall` 0.345 → **0.445**（Δ=+10.0pt，ドメイン単位 n=200 の
exact McNemar で改善 32／悪化 12，**p=0.003658**，95% CI [+3.5pt, +16.5pt]）．
mean_dispatch=2.000000（コスト中立），rank2_flip_rate=0.46，対 Iter60 不一致 552/1600．
非退行は N1（rank_1 1600/1600 不変）・N2（top1_accuracy 0.5975 完全一致）・N3（legal 8/30→15/30）・
N5（単一ドメイン argmax 0.603333）すべて PASS．独立検算の不一致 0 件．

**判定: 採用（adopted）— Iter60 の adopted を「汎化可能な効果」へ格上げ．レバーはクローズ（試し切り）**

事前登録の判定規則（config.yml:891-895）の**第 1 分岐**（S1〜S4 全充足）に機械的に該当する．
FAIL は 1 件も無い．事後の緩和・厳格化は行っていない．これに伴い，

- **効果量の正式値を Iter61 の値へ置き換える**:
  `compound_domain_set_recall` **0.345 → 0.445（Δ=+10.0pt，exact McNemar p=0.003658，
  95% CI [+3.5pt, +16.5pt]，mean_dispatch=2.000000）**．
  Iter60 の +13.5pt は，評価集合のペア別件数分布というテスト集合由来の情報を配分決定に含む値であり，
  **汎化推定値として引用しない**（「legal 絡みを厚く配分した場合の上限値」として位置づける）．
- **対外記述に必ず併記する留保（2 点．Iter60 の B94 と同じ運用）**:
  1. **legal 依存（R-B）は緩和したが未解消**．legal 絡み 60 ペアを除いた部分集合（n=140）では
     Δ=+8.6pt・exact p=**0.0501** で，方向は一貫するが α=0.05 に届かない（Iter60 は p=0.0708）．
     主基準 S1 は全体 200 ペアで定義され p=0.003658 で成立しているが，
     「legal を除いた部分集合だけでは有意性を主張できない」ことを明記する．
  2. **効果量 95% CI の下端 +3.5pt が事前登録の効果量床 +4.0pt をわずかに下回る**．
     事前登録 S2 は点推定基準なので PASS で正しいが，Iter60 で成立していた「CI 下端まで床を超える」
     という強い条件は今回は成立していない．
- なお partial 分岐が要求していた注記「legal 絡みの厚い配分に依存する」は，配分を均一化しても
  効果が残った以上**付さない**（依存していたのは配分ではなく legal ペアの存在である）．

**学び**

1. **設計レベルのリークは存在したが，効果の本体を作ってはいなかった**．効果量の縮小
   13.5pt→10.0pt はほぼ全量が legal 絡み 60 ペアの寄与縮小（+26.7pt→+13.3pt）で説明でき，
   legal を除く 140 ペアは +7.9pt→+8.6pt とほぼ不変だった．リークの大きさは全体で約 3.5pt，
   影響範囲は legal 絡みに限局，と定量できた．**本文リーク監査（A7，3-gram Jaccard）では
   検出できない設計レベルのリークでも，配分という 1 変数を均一化して再実験すれば
   その寄与を定量的に切り離せる**．この切り分け手続き自体が再利用可能な方法論である．
2. **独立サンプルによる再現が副産物として得られた**（当初は想定していなかった）．
   生成の乱数（temperature=0.8）により合成 135 件の本文は Iter60 の 153 件と同一ではないのに，
   Iter60 予測と Iter61 予測を直接 McNemar にかけると p=0.360（有意差なし）．
   効果が個々の生成文ではなく「2 ドメイン同時ラベルという教師信号の設計」に帰属する，という
   Iter60 の因果的主張が n=2 の独立再現で裏付けられた．同指標は Iter46〜59 で 0.345〜0.360 の
   ±1.5pt 帯に張り付いていたので，0.445 と 0.480 はいずれもその帯の外にある．
3. **rank_2 の順位付けは教師信号の構成に極端に敏感だが，方向性のある効果は配分に依らず安定**．
   配分 1 変数（153→135 件）の変更で rank_2 の 34.5%（552/1600）が Iter60 と入れ替わるのに，
   結果指標の差は有意でない（p=0.360）．**flip 量の大きさを「不安定さ」と読むのは誤り**で，
   指標側で安定性を確認する必要がある．
4. **改善は特定ドメインの犠牲の上に立っている（新しい所見）**．`education` 自身の被覆は
   基準線 9/20 → Iter60 4/20 → Iter61 5/20 と **2 イテレーション連続で退行**し，退行幅（-5, -4）が
   2 回とも同程度であることから，乱数ではなく**構造的な退行**と判断する．機序は，基準線が
   rank_2=education を 1600 行中 421 行（26.3%）と過剰出力しており，その過剰さが偶然
   education の compound 行を拾っていたこと．多ラベルヘッドは rank_2 分布を平坦化する
   （Iter61 の education は 130 行）ため，過剰出力に依存していた被覆が失われる．
   さらに Iter61 では `medical` も新たに基準線を割った（13/28 → 12/28．Iter60 は 19/28）．
   **education は Iter32〜53 で 10 回以上レバーを振っても 0.4〜0.5 台から動かなかった問題ドメイン
   であり，ここで初めて（悪い方向にだが）動かせることが判明した点は情報量が大きい**．
5. **`models/dispatch_multilabel_head_iter61.joblib` は本番経路へ配線していない**（Iter60 と同じ）．
   配線は `config.yaml` のスキーマ変更（＋`node.py:214` と `run_experiment.py:93` の同時変更）を伴い，
   rc-reflector の自律判断（可逆な判断に限る）の範囲外である．B94 要レビュー 1 と同一論点が
   2 イテレーション続けて未決のため，B95 で 1 本化して人間判断を仰ぐ．

**次イテレーション（Iter62）の方針**

`multilabel_pair_allocation` は values 単一値のためクローズ（試し切り）．config.yml の既存 levers は
実質すべて試し切り済みのため，skill の停止条件 1 に従い，上記の学び 4 に直撃する新レバー
**`multilabel_rank2_score_calibration = per_domain_holdout_calibration`** を考案し config.yml の
levers 末尾へ追記した（詳細と根拠は backlog B95）．
イテレーション名は「**多ラベルヘッド得点のドメイン別較正による rank_2 偏りの是正**」．

**コミット**: `b86eddb`（🎯 Iter61: 配分均一化でも効果は有意に残存，多ラベル教師信号を「汎化可能な効果」へ格上げ）

