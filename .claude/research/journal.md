## Iteration 61: 合成ペア配分の均一化による設計リークの切り分け

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

## Iteration 60: 2ドメイン訓練事例の新規生成による多ラベルヘッドの再訓練

### 調査 (Iter60)

**問い**

- Q1: 少量データでの multi-label 学習データの合成生成（ルールベース結合 vs LLM 生成）の品質・
  分布ギャップ・過学習リスクに関する先行研究．`MultiLabelBinarizer` を使った少量サンプル OvR
  訓練の落とし穴．
- Q2（最優先）: `data/classifier_train.jsonl` の実データ・Iter59 実装（訓練/採点スクリプト・埋め込み
  キャッシュ）・`build_dataset.py` の複合設問実装・`config.yaml` の生成モデル設定を正確に特定する．
- Q3: Q1・Q2 を踏まえた (a)/(b) の実現可能性・推奨デフォルトパラメータの整理．

**分かったこと（Q1: 先行研究）**

- **ルールベース結合（Concat）と LLM 生成の直接比較実験が文献に存在する**．Chen & Zhou (?), arXiv
  2312.11276 "Compositional Generalization for Multi-label Text Classification: A Data-Augmentation
  Approach" は "Concat"（単一ラベル事例をそのまま連結して多ラベル合成例を作る手法，Jia & Liang 2016
  に由来）を GPT-3.5・Flan-T5・VAE 系生成器と同一ベンチマーク（SemEval/AAPD/IMDB）上で比較した．
  **Concat は No-Aug からわずかに改善する（SemEval Jaccard 44.90→45.84，IMDB Jaccard 42.94→46.13，
  Accuracy 4.48→8.71）が，GPT-3.5/Flan-T5 等の生成型手法に一貫して劣る**（IMDB Accuracy: Concat 8.71
  vs GPT-3.5 10.04 vs Flan-T5 11.69）．原因として論文は「連結されたテキストが意味的・統語的に
  一貫していない（neither semantically nor syntactically coherent）」ことを明記している
  （出典: https://arxiv.org/html/2312.11276v3 ）．**本リポジトリの `_COMPOUND_QUESTIONS`（下記 Q2）
  は「2問の連結」ではなく「1つの統合されたシナリオ」であり，この論文の Concat 劣化機序が
  そのまま当てはまる構造的リスクである**ことが確認できた（config.yml note の懸念が文献的にも
  裏付けられた）．
- **低リソース設定での LLM 合成データ拡張は効果があるが，増やしすぎると頭打ちになる**．Empirical
  case study (arXiv 2407.12813, "Data Generation using Large Language Models for Text
  Classification") は，元データ 100 件規模では合成データ拡張で 3〜26% の改善が得られる一方，
  1000 件規模では効果が 5% 未満に縮小し，「合成データ量を増やせば単調に改善するわけではない」
  「生データと合成データを併用するのが望ましい」「合成データ特有のバイアス・パターンに注意」と
  結論している（出典: https://arxiv.org/html/2407.12813v1 ）．本レバーの規模（元1427件に対し
  数十〜百数十件を追加する想定）はこの「低リソース・少量追加」レンジに該当し，効果が出るとすれば
  この規模感が妥当という傍証になる．
- **`MultiLabelBinarizer` は実際に必須**（Iter59 の「不要」という結論はこのイテレーションには
  適用されない）: 実機の sklearn 1.9.0 で実際に検証したところ，`OneVsRestClassifier.fit(X, y)` に
  `y=[["a"],["b"],["a","b"], ...]`（ラベルのリストのリスト）を直接渡すと
  `ValueError: You appear to be using a legacy multi-label data representation. Sequence of
  sequences are no longer supported; use a binary array or sparse matrix instead - the
  MultiLabelBinarizer transformer can convert to this format.` で例外になることを確認した．
  **`MultiLabelBinarizer().fit_transform(y)` で二値インジケータ行列に変換してから
  `OneVsRestClassifier.fit(X, Y)` に渡す必要がある**．
- **`classes_` の意味が変わる点が実装上の落とし穴**: 同じ sklearn 1.9.0 で確認したところ，
  単一ラベル文字列を渡した場合（Iter59の方式）は `model.classes_` がドメイン名文字列の配列になるが，
  `MultiLabelBinarizer` の出力（0/1 行列）を渡した場合は `model.classes_` が単なる列インデックス
  `[0, 1, 2, ...]` になる（ドメイン名の対応は別途保持している `mlb.classes_` を使う必要がある）．
  **Iter59 の `evaluate_dispatch_candidate_ranking.py:_head_scores()`（`zip(head.classes_,
  probabilities)`）をそのまま新ヘッドに使うと，キーがドメイン名ではなく整数になるバグを生む**．
  新スクリプトでは `mlb.classes_`（保存が必要）を使って zip する実装に変える必要がある．
- **`predict_proba()` の挙動も変わる**: 単一ラベル入力時は合計が1になるよう再正規化される
  （Iter59 で確認済み）が，`MultiLabelBinarizer` 由来の真の多ラベル行列を渡すと**再正規化されず**，
  各列が独立 sigmoid のまま返る（同一検証で `predict_proba(X).sum(axis=1)` が1にならないことを
  実機のsklearn 1.9.0で確認）．したがって新ヘッドでは `decision_function()`＋手動sigmoidではなく
  `predict_proba()` を直接使ってよい（Iter59 が `decision_function()` を使った理由＝単一ラベル時の
  再正規化回避，は今回は該当しないが，Iter59との実装対称性を優先するなら decision_function 方式を
  踏襲してもよい．どちらでも数学的に同じランキング結果になる．計画フェーズで選択）．
- **少量データでの binary relevance のクラス不均衡・少数ラベル対策**は Iter59 調査で確認済みの
  Zhang & Zhou (2017) の知見がそのまま今回にも適用される（`class_weight="balanced"` を各二値問題に
  独立適用）．今回新たに追加されるのは「2ドメイン同時ラベルの正例数」という**新しい極少数クラス**
  であり（後述 Q3 のとおり，各ドメインペアに数個ずつしか合成しない前提では，個々のペアの正例数は
  訓練データ全体の1%未満になる），Wikipedia・Zhang & Zhou が指摘する「binary relevance はラベル間
  依存を見ない」という弱点とは別に，**「ラベル共起パターン自体を学習させたいのに，共起の正例が
  極少数」というこのレバー固有のジレンマ**が生じる点に注意が必要．

**分かったこと（Q2: コードベース調査，最優先）**

1. **`data/classifier_train.jsonl`**: 1427行，スキーマは `{"id", "query", "domain"}` の3フィールドの
   み（Iter59調査と同一，再確認済み）．`id` は `"{domain}-train-{NNN:03d}"` 形式（例:
   `business_economics-train-001`）で1427件全てユニーク．ドメイン別件数:
   `business_economics/computer_science/education/general/history_culture/mathematics/medical/
   natural_science/social_science` が各150件，**`legal` のみ77件**．`query` は JMMLU 由来の
   四択問題文（A〜Dの選択肢付き）であり，**自然文の相談文ではない**（実データを実際に読んで確認．
   例: 「かつてマルチブランド政策と呼ばれた...次のどれか? A. 個別ブランド B. ...」）．
2. **`build_dataset.py:617-621`（`_COMPOUND_QUESTIONS` 直前のコメント）**: 「JMMLU の四択問題は
   単一タスクに属し，真のクロスドメイン曖昧性を表現できないため，複合設問は JMMLU 由来ではなく
   手作りとする」と明記されている．実際に `_COMPOUND_QUESTIONS`（`build_dataset.py:621-`，
   Python構造として機械的にパースして確認）は**100件，1〜2文の自然な日本語相談文**（選択肢なし，
   例:「仕事中に転倒して怪我をしました．治療費と休業補償について知りたいです．」）であり，
   `classifier_train.jsonl` の四択問題形式とは**文体・構造が根本的に異なる**．
   ドメインペア別内訳（機械的に集計）: `legal×medical` が12件で最多，`education×medical`・
   `education×legal` が各4件，残り40ペアが各2件（45ペア中43ペアが登場，計100件．legal を含む行は
   30件でIter59調査の記述と一致）．`build_dataset.py:1147-1155`（`_build_rows()`）でこの100件に
   `id=f"compound-{index:03d}"`・`expected_domains`・`is_compound=True` を付与して評価用
   `dataset.jsonl` に組み込む．**IDは `compound-001`〜`compound-100`，`classifier_train.jsonl` の
   `id` 命名（`{domain}-train-NNN`）とは名前空間が別**であり，機械的なID衝突は起きない．
   ただし**リーク回避のために本レバーの実装は `_COMPOUND_QUESTIONS` を一切 import／参照しない**
   ことを構造的に徹底すべき（生成プロンプトの参考にすることも含め避ける．文字列としての漏洩だけで
   なく，シナリオの着想の漏洩も広義のリークとみなす）．
3. **Iter59 の訓練スクリプト `scripts/train_dispatch_candidate_ranking_head.py`**（全文読了）:
   `train_domain_classifier.py` の `_load_training_rows()`（`train_domain_classifier.py:74-77`，
   `{"id","query","domain"}` を JSONL から読むだけ）と `build_training_features()`
   （`train_domain_classifier.py:99-142`）を import して再利用し，`OneVsRestClassifier(
   LogisticRegression(max_iter=1000, class_weight="balanced"))` を1427行で fit する
   （`train_dispatch_candidate_ranking_head.py:83-100`）．**`class_weight="balanced"` かつ
   `_extract_sample_weights()` を使わない設計**（Iter32の乗算結合バグを避けるため）．5-fold CV の
   per-domain ROC-AUC/average precisionを診断出力する（`:103-132`）．
   `build_training_features()`（`train_domain_classifier.py:99-142`）は**両分岐とも
   `labels.append(row["domain"])` で `row["domain"]` をそのまま追加するだけ**であり，型チェックを
   していない．**この関数はコードを変更しなくても `row["domain"]` がリスト（例:
   `["legal","medical"]`）の行を含む JSONL をそのまま渡せば，そのリストを含んだ `labels` を返す**
   （＝新規の合成2ドメイン行を既存関数に流し込むこと自体は無改造で可能．ただし返る `labels` は
   文字列とリストが混在するため，`MultiLabelBinarizer` に渡す前に「文字列は1要素リストへ正規化する」
   前処理が新スクリプト側に必要）．
4. **Iter59 の採点スクリプト `scripts/evaluate_dispatch_candidate_ranking.py`**（全文読了）:
   `--baseline`（固定 `dispatch_top_k=2` の `results/20260918_202613/results.jsonl`）から
   rank_1 をそのまま引き継ぎ（`:148`，既存分類器を再計算しない），embedding キャッシュ
   （`--embedding-cache`，`.npz`，`ids`/`embeddings` の2キー，`:84-101`）を使い回しつつクエリを
   embed し，`_head_scores()`（`:127-131`，`decision_function()`→手動 sigmoid，
   `zip(head.classes_, probabilities)`）で10ドメイン分のスコアを得て `build_new_rows()`
   （`:134-166`）で rank_1 を除く9ドメイン中最大スコアを rank_2 とする．A1（rank_1不変，`:169-180`）・
   A2（コスト中立，`:183-203`）・A3（rank2_flip_rate，`:206-209`）の assert 済み．**この
   `_head_scores()` の `zip(head.classes_, ...)` 部分が，上記Q1で確認した「MLB経由だと `classes_`
   が整数になる」問題の直撃箇所**であり，本レバー実装では改修必須．
5. **`results/iter59_query_embeddings.npz`**: ローカルに実ファイルとして現存（9,971,712 bytes，
   `ls -la` で確認．B93 の記載どおり git 未追跡）．中身は `{ids: array[str], embeddings:
   array[float]}` の2キー（`_load_embedding_cache`/`_save_embedding_cache` の実装から自明，
   `evaluate_dispatch_candidate_ranking.py:84-101`）で，1600問（`compound-*` 100件＋JMMLU由来
   1500件）の embedding をキー=`id` で保持．**この1600件は評価用 `dataset.jsonl` の行であり，
   `classifier_train.jsonl` の1427行とは別集合**（IDの名前空間も異なる）．したがって**この
   キャッシュは評価フェーズ（1600問の再採点）ではそのまま再利用できるが，新設する合成訓練データ
   （`classifier_train.jsonl` 由来ではない新規 `id`）の embedding は含まれておらず，訓練フェーズで
   別途 embed が必要**（訓練コストは新規合成行数分のみ，数十〜百数十件なら数分未満）．
6. **`config.yaml`**: `embedding_model: nomic-embed-text`（`:4`），`judge_model:
   schroneko/llama-3.1-swallow-8b-instruct-v0.1:q4_k_m`（`:107`，回答品質評価用に既存稼働中の
   モデル）．`expert_backend.py:14-34`（`OllamaClient.generate(model, prompt, ...)`）が
   既に汎用の生成API呼び出しとして実装済みで，`scripts/evaluate_response_quality.py` が
   judge_model 呼び出しに使っている実績がある．**(b) LLM 生成を選ぶ場合，この `judge_model` を
   そのまま生成モデルとして転用すればよく，新規のクライアント実装は不要**．

**分かったこと（Q3: 実験設計への示唆）**

1. **(a) ルールベース結合と (b) LLM 生成の比較まとめ**:
   - (a) は文献上の "Concat" 相当だが，本リポジトリの (a) は「2行をそのまま連結」ではなく
     「自然な日本語で1問に結合」という設計（config.ymlのnote）であるため，文献の Concat
     （単純連結）よりは改善されているものの，**実装するには何らかのテンプレート／ルールで
     四択問題文2つを1つの自然文シナリオへ書き換える処理が要る**．これは実質的に軽量な自然言語
     生成であり，「ルールベースで自然文を作る」ことの難度は過小評価すべきでない．また
     `classifier_train.jsonl` の `query` は四択形式（A〜D選択肢付き）で，`_COMPOUND_QUESTIONS` の
     ような選択肢なし自然文とは文体が根本的に異なるため，**(a) を実装しても文体ギャップは
     完全には埋まらない**（選択肢を残すか，除去して文だけ結合するかで設計判断が必要）．
   - (b) は文献（2312.11276）で Concat より一貫して高性能かつ，本プロジェクトには
     `judge_model`（`config.yaml:107`）という生成コストゼロ（追加LoRA不要）で使えるモデルが
     既に存在するため，**実現コストは (a) とほぼ同等（生成トラフィックが加わる程度）で，
     分布ギャップは (a) より小さくなる見込み**．ただし文献（2407.12813）が警告する
     「合成データ固有のバイアス」対策として，生成後のフィルタリング（例: 生成された query が
     実際に2ドメインの語彙を含むか，長さが極端でないか等の軽量チェック）を計画に含めるべき．
   - **推奨**: (b) を第一候補とし，(a) は (b) が実機生成コスト・品質確認の点で見送られた場合の
     フォールバックとして計画書に両論併記する．
2. **推奨デフォルトパラメータ（次フェーズが1つに絞る前提の目安）**:
   - **生成規模**: 文献（2407.12813）の知見（低リソースほど効果大，1000件規模では効果縮小）を
     踏まえ，元の1427件に対し**全45ドメインペア×2〜4件＝90〜180件程度**を追加するのが妥当な
     出発点．test の `_COMPOUND_QUESTIONS` 自身も「ほぼ全ペア×2件＋legal×medicalのみ12件」という
     配分であり（Q2参照），この配分の**「件数」そのもの**（テキスト内容ではなくペアごとの多寡）を
     参考に，legal×medical・education×medical・education×legal を若干厚めにする設計は，
     テキストの漏洩を伴わない範囲で妥当な事前知識の反映といえる．
   - **ドメインペア選定**: legal（訓練77件，最少）を含むペアは，OvR個別二値問題の正例が
     極少数になりやすいため，過学習防止の観点から**生成件数をやや多めに配分**しつつ，
     5-fold CV ROC-AUC の悪化がないかを訓練スクリプトの診断出力（Iter59から継承）で確認する
     運用が要る．
   - **スキーマ**: 新規ファイル（例: `data/classifier_train_multidomain.jsonl`）を新設し，
     `classifier_train.jsonl` 自体は無変更に保つ設計を推奨（既存 `train_domain_classifier.py`
     が rank_1 分類器の訓練に使う唯一の入力ファイルであり，混入・破壊のリスクを避けるため）．
     `{"id": "synth-{domain1}-{domain2}-{NNN}", "query": "...", "domain": [domain1, domain2]}`
     という形式にし，新設の訓練スクリプトが `classifier_train.jsonl`（`domain`は文字列）と
     この新ファイル（`domain`はリスト）の両方を読み込み，`[row["domain"]] if isinstance(...)
     str) else row["domain"]` で正規化してから `MultiLabelBinarizer` に渡す実装にする．
3. **実装上の必須変更点（Q1で判明した落とし穴の反映）**:
   - 訓練: `MultiLabelBinarizer().fit_transform(normalized_labels)` を経由してから
     `OneVsRestClassifier(...).fit(embeddings, Y)` する．`mlb.classes_`（ドメイン名の順序付き配列）
     を**モデルと一緒に保存する**（例: `joblib.dump({"model": model, "classes": mlb.classes_},
     output_path)`，Iter59の「joblib.dump(model, ...)」単体保存から変更が必要）．
   - 採点: `evaluate_dispatch_candidate_ranking.py` の `_head_scores()` を，保存された
     `classes` 配列を使ってドメイン名にマッピングするよう改修する（`head.classes_` を直接使う
     現行実装のままでは整数キーになりバグる）．`decision_function()`＋手動sigmoid／
     `predict_proba()` 直接使用のどちらでも可（Q1参照，数学的に同じ順位になる）．
   - A1（rank_1不変）・A2（コスト中立）・A3（rank2_flip_rate）のassertパターンはIter59のまま
     再利用可能（`evaluate_dispatch_candidate_ranking.py:169-209`のロジックは変更不要，
     `_head_scores()`の内部実装のみ変わる）．

**次の計画フェーズへの示唆**

1. **(a)/(b) の選択を計画フェーズの最初の決定事項とする**．(b) を推奨（文献根拠あり，実装コストは
   同程度，`judge_model` が既に利用可能）．(a) を選ぶ場合は「四択選択肢をどう扱うか（残す/除去）」
   を追加で決める必要がある．
2. **既存の `classifier_train.jsonl` は変更せず，新規ファイルへ合成データを分離する**設計を推奨
   （rank_1分類器の訓練データを汚染しない）．
3. **`MultiLabelBinarizer` 導入に伴う実装変更点（`classes_` の意味変化，`_head_scores()` の
   改修必須）を計画書に明記する**．Iter59のコードをそのまま流用できない箇所として最優先で扱うこと．
4. **リーク防止策**: 生成スクリプト（(a)(b) どちらでも）は `build_dataset.py` の
   `_COMPOUND_QUESTIONS` を一切 import／参照しないことをコードレビュー項目として明記する．
5. **生成件数・ドメインペア配分の初期値**: 45ペア×2〜4件（90〜180件），legal絡みペアをやや厚め，
   を出発点として計画フェーズで具体的な数値に確定する．
6. **評価パイプラインは Iter59 をそのまま踏襲**（基準線 `results/20260918_202613/results.jsonl`，
   成功条件 S1〜S4・非退行 N1〜N4 は config.yml note のとおり完全に揃える）．
   `results/iter59_query_embeddings.npz` はローカルに現存し1600問評価に再利用可能（訓練用の新規
   合成行の embedding だけ追加で必要）．

**出典一覧**
- Chen et al., "Compositional Generalization for Multi-label Text Classification: A
  Data-Augmentation Approach" (arXiv, 2023/2024改訂): https://arxiv.org/html/2312.11276v3
- "Data Generation using Large Language Models for Text Classification: An Empirical Case Study"
  (arXiv 2407.12813): https://arxiv.org/html/2407.12813v1
- scikit-learn 1.9.0 実機検証（本フェーズで `uv run python` により実行し確認，出典は本リポジトリの
  Python環境自体）: `OneVsRestClassifier.fit()` への list-of-lists 直接投入がエラーになること，
  `MultiLabelBinarizer` 経由後は `classes_` が整数配列になること，`predict_proba()` が
  再正規化されなくなること．
- Zhang & Zhou (2017)・Wikipedia "Multi-label classification"・scikit-learn calibration/
  `OneVsRestClassifier` ドキュメント（Iter59調査で確認済み，本イテレーションでも参照）:
  journal.md 旧Iteration 59「調査」節の出典一覧を参照．

### 計画 (Iter60)

**仮説**

Iter58（既存 confidence の gap，compound 判別 AUC 0.576）と Iter59（既存 embedding ＋ 単一ラベル
データの OvR 再分解，2 つ目の正解ドメインの順位比較 Wilcoxon p=0.914）が独立に示したとおり，
2 位枠が改善しない原因は推論側の工夫の不足ではなく，**訓練データに「1 つの設問が 2 ドメインに
またがる」という事例が 1 件も存在しないこと（`data/classifier_train.jsonl` 1427 行が全て単一
ドメイン）**である．2 ドメインにまたがる訓練事例を新規生成して `MultiLabelBinarizer` で真の
多ラベル目的変数を作り，Iter59 と**同一構造**の OvR ヘッドを再訓練すれば，dispatch コストを
一切増やさない（固定 k=2，mean dispatch 2.0）まま `compound_domain_set_recall` が基準線 0.345
から改善する．ヘッド構造・推論経路・評価手続きを Iter59 と完全に揃えるため，**Iter59 の結果
（0.350，McNemar p=1.0）がそのまま対照群（教師信号が単一ラベルの場合）として機能する**．

**単一レバー**

`multilabel_training_signal`: （現状＝訓練事例が単一ドメインのみ）→ `synthetic_two_domain_training_examples`
（**(b) LLM 生成**方式で 2 ドメイン同時ラベルの訓練事例を新規生成し，既存 1427 行に追加して
`MultiLabelBinarizer` 経由で OvR ヘッドを訓練する）．

**生成方式の選択（調査フェーズの (a)/(b) から 1 つに確定）: (b) LLM 生成を採用し，(a) 決定論的
結合は実施しない（同時比較はしない）**．根拠は 3 点:
1. arXiv 2312.11276 が Concat 系（単一ラベル事例の連結）は No-Aug より改善するものの LLM 生成に
   一貫して劣ることを同一ベンチマーク上で示しており（IMDB Accuracy: Concat 8.71 vs GPT-3.5 10.04
   vs Flan-T5 11.69），劣化の原因を「連結テキストが意味的・統語的に一貫しない」と明記している．
2. 評価側の複合設問は「2 問の連結」ではなく「統合された 1 つのシナリオ」であり（調査 Q2），
   (a) の分布ギャップは本リポジトリで特に大きい．さらに `classifier_train.jsonl` の `query` は
   四択問題文（A〜D の選択肢付き）であり，(a) を採ると「選択肢を残す/除去する」という
   **単一レバーに含まれない追加の設計判断**が生じる．
3. `config.yaml:107` の `judge_model`（`schroneko/llama-3.1-swallow-8b-instruct-v0.1:q4_k_m`）と
   `expert_backend.py:24-34` の `OllamaClient.generate()` が既に稼働実績を持ち，**新規のクライアント
   実装なしで (b) を実装できる**（実装コストは (a) とほぼ同等）．

固定する構成（直近最良から変更しない）:
- embedding: `nomic-embed-text`（frozen，再訓練なし）
- rank_1 分類器: `models/domain_classifier.joblib` を**一切変更しない**（md5 不変を確認する）
- ヘッド構造: `OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight="balanced"))`
  ＝ Iter59 と同一．ハイパラ探索は行わない（Iter58 の in-sample 選定問題を繰り返さない）
- 推論・採点手続き: Iter59 の `scripts/evaluate_dispatch_candidate_ranking.py` を踏襲
  （rank_1 は baseline からそのまま引き継ぎ，残り 9 ドメインをヘッドスコア降順で rank_2 とする）
- 基準線: `results/20260918_202613/results.jsonl`（固定 k=2，compound_domain_set_recall 0.345）
- `aggregator.py` / `node.py` / `run_experiment.py` / `classifier.py` / `config.yaml` /
  `data/classifier_train.jsonl` / `build_dataset.py`: 無変更

**【必須の制約】リーク防止**

評価用複合設問 100 問（`build_dataset.py` の `_COMPOUND_QUESTIONS`）はテストセットであり，
訓練データ生成に一切流用しない．具体的な徹底策:
- 生成スクリプト・訓練スクリプトは `build_dataset` を **import しない**（テキストの参照も，
  シナリオ着想の参考にすることも禁止）．
- 生成対象の 10 ドメイン名は `data/classifier_train.jsonl` の `domain` 列の一意集合から
  導出する（`build_dataset.py` の `_DOMAIN_TASKS` を参照しない）．これにより「build_dataset を
  import しない」ことが構造的に成立する．
- 事後のリーク監査（下記 A7）のみ `_COMPOUND_QUESTIONS` を読むが，これは**近似重複の検出器
  であって生成物の選別器ではない**（閾値 0.9 の剽窃ガードのみ．それ未満の類似度で生成物を
  取捨選択することはしない）．

**変更するファイルと箇所**

1. `scripts/generate_multidomain_training_examples.py`（**新規**）
   - `config.yaml` の `judge_model` を `OllamaClient.generate()` で呼び，指定 2 ドメイン双方の
     知識が無いと答えられない 1〜2 文の日本語相談文を 1 件ずつ生成する（選択肢を含めない旨を
     プロンプトで明示）．ドメイン名→日本語説明の対応は本スクリプト内の定数辞書で定義する．
   - `temperature=0.8`（多様性確保），1 リクエスト 1 件，ペア・スロット順は決定論的に走査．
   - **生成規模**: 10 ドメインの全 45 ペア × 3 件 ＝ 135 件をベースとし，`legal` を含む 9 ペア
     のみ +2 件（計 5 件）として **153 件**を目標とする．根拠: 調査 Q3 の推奨レンジ 90〜180 件
     （arXiv 2407.12813 の低リソース域）に収まり，`legal` は訓練 77 件と最少かつ compound 100 行
     の 30 件に登場するため二値問題の正例が特に不足しやすい．**件数配分のみを事前知識として
     反映し，テキストは一切参照しない**．
   - **生成後フィルタ**（各スロット最大 3 回まで再生成，全滅したスロットは欠番として記録）:
     F1 文字数 20〜200，F2 四択マーカー（`A.` `B.` `C.` `D.` 等）を含まない，
     F3 生成済み集合と完全一致しない，F4 改行を含む複数問形式でない．
   - 出力: `data/classifier_train_multidomain.jsonl`（**新規データファイル**，コミットする）．
     形式 `{"id": "synth-{d1}-{d2}-{NNN}", "query": "...", "domain": [d1, d2]}`．
     **`data/classifier_train.jsonl` は無変更**（rank_1 分類器の訓練入力を汚染しないため）．
   - 実採取件数が **120 件未満なら実験を成立させず実装を見直す**（生成品質の下限）．
2. `scripts/train_multilabel_dispatch_head.py`（**新規**．Iter59 の
   `train_dispatch_candidate_ranking_head.py` は**対照群の再現性のため無変更で残す**）
   - `--train-data`（既存 1427 行）と `--multilabel-train-data`（新規合成行）の 2 入力を読み，
     `train_domain_classifier.py` の `_load_training_rows()` / `build_training_features()` を
     再利用して embed する（調査 Q2-3 のとおり，`build_training_features()` は `row["domain"]`
     をそのまま `labels` に積むため，リスト値の行も無改造で通る）．
   - `labels` を `[x] if isinstance(x, str) else x` で正規化 → `MultiLabelBinarizer().fit_transform()`
     → `OneVsRestClassifier(...).fit(embeddings, Y)`．
   - 保存形式を Iter59 から変更: `joblib.dump({"model": model, "classes": list(mlb.classes_)},
     "models/dispatch_multilabel_head.joblib")`（`OneVsRestClassifier.classes_` が MLB 経由では
     整数列になるため，ドメイン名の対応を別途保持する必要がある．調査 Q1 参照）．
   - 5-fold CV の per-domain ROC-AUC / average precision 診断出力は Iter59 から継承する
     （多ラベル化に伴い `StratifiedKFold` が使えないため `KFold(shuffle=True, random_state=42)`
      に変更し，out-of-fold の `decision_function` から列ごとに算出する）．
3. `scripts/evaluate_dispatch_candidate_ranking.py`（**既存を編集**）
   - `_head_scores()` を，保存された `classes`（ドメイン名配列）で zip するよう改修する
     （現行の `zip(head.classes_, probabilities)` は MLB 由来ヘッドでは整数キーになりバグる）．
   - ヘッド読み込みを「dict ペイロード（新）／素の推定器（Iter59 の旧形式）」の両対応にし，
     旧形式では従来どおり `head.classes_` を使う（Iter59 成果物の再採点互換を壊さない）．
   - `--iter59-predictions`（任意）を追加し，Iter59 予測との rank_2 不一致件数を出力に記録する
     （下記 A5 用）．A1 / A2 / A3 のアサーションロジック（`:169-209`）は**変更しない**．
4. `scripts/compute_iter59_ranking_stats.py`（**変更なし・そのまま流用**）
   - `--baseline` / `--new` / `--output` で完全にパラメータ化されており，S1〜S4・N1〜N4 を
     Iter59 と**同一コード・同一手続き**で算出できる．比較可能性を担保するため改変しない．

**実施方法（コマンド手順）**

```
# 0) 合成訓練データの生成（judge_model への生成トラフィックが発生．153 件目標）
uv run python -m scripts.generate_multidomain_training_examples \
    --train-data data/classifier_train.jsonl \
    --model schroneko/llama-3.1-swallow-8b-instruct-v0.1:q4_k_m \
    --ollama-host 192.168.15.100 \
    --per-pair 3 --per-pair-legal 5 \
    --output data/classifier_train_multidomain.jsonl

# 0') リーク監査（A7．生成物を選別せず，近似重複のみ検出）
uv run python -m scripts.generate_multidomain_training_examples --audit-leak \
    --output data/classifier_train_multidomain.jsonl

# 1) 多ラベルヘッドの訓練（1427 + 153 件 embed）
uv run python -m scripts.train_multilabel_dispatch_head \
    --train-data data/classifier_train.jsonl \
    --multilabel-train-data data/classifier_train_multidomain.jsonl \
    --embedding-model nomic-embed-text --ollama-host 192.168.15.100 \
    --output models/dispatch_multilabel_head.joblib

# 2) 1600 問のオフライン採点（embed のみ．キャッシュ再利用）
uv run python -m scripts.evaluate_dispatch_candidate_ranking \
    --baseline results/20260918_202613/results.jsonl \
    --head models/dispatch_multilabel_head.joblib \
    --embedding-model nomic-embed-text --ollama-host 192.168.15.100 \
    --embedding-cache results/iter59_query_embeddings.npz \
    --iter59-predictions results/iter59_ovr_ranking_predictions.jsonl \
    --output results/iter60_multilabel_ranking_predictions.jsonl

# 3) 指標・検定（Iter59 と同一スクリプト・同一手続き）
uv run python -m scripts.compute_iter59_ranking_stats \
    --baseline results/20260918_202613/results.jsonl \
    --new results/iter60_multilabel_ranking_predictions.jsonl \
    --output results/iter60_stats.json
```

**no-op 対策（Iter58/59 で繰り返し発生した「発火しているのに no-op」への機械的アサーション）**

本イテレーションも実行時経路（`node.py:214` / `run_experiment.py:93`）を通らないオフライン検証
であるため，Iter16/20/21/22/27/58 型の「レバーを読む行に到達しない」no-op は構造的に起こらない．
代わりに，**教師信号の多ラベル化が実際にヘッドへ届いていること**を以下で機械的に保証する．

- A0（教師信号が真に多ラベル）: MLB 出力 `Y` について `(Y.sum(axis=1) >= 2).sum()` が合成行数に
  一致し，かつ **≧120** であることを訓練スクリプトで assert．`len(mlb.classes_) == 10` かつ
  全要素が既知のドメイン名文字列であることも assert．合成行が 45 ペア中いくつを被覆したかを出力．
- A1（rank_1 不変）: 1600/1600 一致（Iter59 の実装をそのまま使用）．
- A2（コスト中立）: 全行 k=2・rank_1 ≠ rank_2・mean dispatch = 2.000000．
- A3（発火の証拠・対 baseline）: `rank2_flip_rate > 0`．
- A5（発火の証拠・対 Iter59＝**本レバー固有の no-op 検出**）: 新ヘッドの rank_2 が Iter59 ヘッドの
  rank_2 と**1600 行中 1 行以上で異なる**こと．0 件なら「153 件の合成行がヘッドを一切動かして
  いない」＝本レバーの no-op であり，実験を成立させず実装を見直す．不一致件数を必ず報告する．
- A6（MLB 整数キー・バグの検出）: 採点スクリプトで `head_scores` のキー集合が 10 個のドメイン名
  文字列と完全一致することを assert（整数キーへのすり替わりを機械的に検出する）．
- A7（リーク監査）: 合成 153 件と `_COMPOUND_QUESTIONS` 100 件の全ペアについて文字 3-gram Jaccard
  類似度を算出し，**最大値が 0.9 以上なら実験を無効**とする．最大値・中央値を必ず報告する．

**成功条件（事前登録．Iter59 と完全に同一．判定は事後変更しない）**

- **S1（主基準）**: `compound_domain_set_recall` が基準線 **0.345（69/200）** から上昇し，
  ドメイン単位 n=200 の **exact McNemar（two-sided binomtest，α=0.05）で p < 0.05**．
- **S2（効果量の下限）**: 点推定の上昇が **+0.04pt 以上**（被覆ドメイン数 69 → **77 以上**）．
- **S3（コスト中立）**: 全 1600 行で k=2，mean dispatch = 2.000000（A2 が通ること）．
- **S4（発火の証拠）**: `rank2_flip_rate > 0` かつ **A5 の対 Iter59 不一致件数 > 0**．
- **第 2 の参照点（gate ではなく併記必須）**: Iter59 の OvR ヘッド（教師信号が単一ラベル，
  同一構造）は **0.350（McNemar p=1.0，2 つ目の正解ドメインの順位比較 Wilcoxon p=0.914，
  平均順位 4.201→4.258）**．本イテレーションの結果は基準線 0.345 だけでなく Iter59 の 0.350 とも
  並べて報告し，**「教師信号を多ラベル化した差分」**として解釈する（Iter59 予測との
  exact McNemar も参考値として算出・併記する）．

判定規則: S1〜S4 全て充足なら**採用**（次イテレーションで実行時経路へ配線．ただし `config.yaml`
のスキーマ変更を伴うためユーザー確認が必要），S1 不成立だが S2 相当の上昇（+0.04pt 以上）が
見える場合は **partial**，S1・S2 とも不成立なら**棄却**．

**非退行条件（事前登録）**

- **N1（rank_1 完全不変）**: A1 が 1600/1600 で通ること．破れた結果は単一レバー原則違反として無効．
- **N2（top1_accuracy 不変）**: new 側 rows で再計算した `top1_accuracy` が baseline の **0.5975** と
  小数点以下まで完全一致すること．
- **N3（legal 非退行・過学習チェック）**: legal 自身の被覆が基準線の **8/30 を下回らない**
  （≧8．Iter59 は 9/30）．加えて 1600 行での legal スコアの標準偏差 > 0，5-fold CV の legal
  ROC-AUC を報告する．
- **N4（恩恵の偏りの分解）**: 改善がある場合，legal 絡み / medical 絡み / その他のドメインペア
  単位に分解して報告する．legal 絡みのみに改善が集中する場合は主張の強度を落とす．
- **N5（新規．文体ショートカットの検出）**: 合成行は自然文，既存 1427 行は四択問題文であり，
  評価 1600 問も compound 100 問が自然文・JMMLU 1500 問が四択文であるため，ヘッドが
  「自然文らしさ→多ラベル」という文体ショートカットを学習した可能性が構造的に残る．
  単一ドメイン 1500 行に対するヘッドの argmax 正解率（Iter59 実測 **0.610**）が **0.590 以上**を
  保つことを非退行条件とし，実測値を必ず報告する．これを下回る場合は，compound での改善が
  あっても「文体による識別」の疑いを考察に明記する．

**留保（考察フェーズへの申し送り）**

- R1: OvR のスコアは合計 1 にならない（MLB 経由では `predict_proba()` も再正規化されない）．
  ランキングにのみ使うため決定には影響しないが，**「確率」として対外記述しない**．
- R2: 評価集合上でのハイパラ選択は行わない（推定器の設定は Iter59 から固定，探索しない）．
  生成件数・配分（45 ペア×3，legal 絡みのみ 5）も事前に固定し，結果を見て変更しない．
- R3: 本イテレーションは**オフライン完結・スキーマ変更なし**であり，ユーザー確認なしで自律着手
  してよい．実機の dispatch 挙動・回答品質・レイテンシは測定しない．**採用となった場合の
  実行時経路への配線（`node.py:214` と `run_experiment.py:93` の両方を同時に変更しないと Iter58 と
  同型の no-op を再演する）と `config.yaml` のスキーマ変更は，その時点で初めてユーザー確認が
  必要になる．今回は着手しない．**
- R4: 合成 153 件は元データ 1427 件の約 10% であり，個々のドメインペアの正例は 3〜5 件と極少数
  である（調査 Q1 が指摘した「共起パターンを学習させたいのに共起の正例が極少数」というジレンマ）．
  陰性結果が出た場合，「多ラベル教師信号が無効」なのか「件数が不足」なのかは本イテレーション
  単独では分離できない．次の一手（件数のスケールアップ）の判断材料として，合成行数と 5-fold CV
  診断値の関係を考察で必ず言及する．
- R5: 生成に使う `judge_model` は評価軸②（回答品質の LLM-as-judge）にも使われているモデルである．
  訓練データ生成と回答品質評価が同一モデルであること自体は本イテレーションの指標
  （`compound_domain_set_recall`，dispatch 側の指標）に影響しないが，将来 End-to-End 品質で
  比較する際には交絡要因になりうる点を記録しておく．

### 実装 (Iter60)

**変更・新規ファイル（計画どおり）**

1. `scripts/generate_multidomain_training_examples.py`（新規）: `config.yaml` の `judge_model`
   （`OllamaClient.generate()`，`temperature=0.8`）で 45 ペア×3 件（`legal` 絡み 9 ペアのみ 5 件）＝
   153 件を生成する CLI。ドメイン名は `data/classifier_train.jsonl` から導出（`_load_domain_names()`），
   `build_dataset` は生成コードパスから import しない（下記「リーク防止の確認」参照）。
   生成後フィルタ F1（20〜200 文字）・F2（四択マーカー不使用）・F3（完全一致重複拒否）・F4（複数行拒否）
   をスロットごとに最大 3 回まで再試行．`--audit-leak` モードのみ `build_dataset._COMPOUND_QUESTIONS`
   をローカル import して A7（文字 3-gram Jaccard，閾値 0.9）を計算する。
2. `scripts/train_multilabel_dispatch_head.py`（新規）: `--train-data`（1427 行）と
   `--multilabel-train-data`（153 行）を結合し，`train_domain_classifier.py` の
   `_load_training_rows()`/`build_training_features()` を再利用して embed。`labels` を
   `[x] if isinstance(x,str) else list(x)` で正規化し `MultiLabelBinarizer().fit_transform()` →
   `OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight="balanced")).fit(embeddings, Y)`。
   保存形式は `joblib.dump({"model": model, "classes": list(mlb.classes_)}, output_path)`（Iter59 の
   素の estimator 保存から変更，計画どおり）。5-fold CV 診断は `KFold(shuffle=True, random_state=42)`
   に変更（`StratifiedKFold` は多ラベル `Y` を受け付けないため）。
3. `scripts/evaluate_dispatch_candidate_ranking.py`（既存編集，最小差分）: `_load_head()` を新設し，
   dict ペイロード（新形式）／素の estimator（Iter59 旧形式）の両方に対応。`_head_scores()` は
   常に呼び出し元から渡された `classes`（ドメイン名リスト）で zip するよう変更し，
   `head.classes_` への直接依存を除去した。`--iter59-predictions`（任意）を追加し A5（Iter59 予測との
   rank_2 不一致件数）を算出・報告するようにした。A1〜A3 のロジック（`:169-` 付近）は無変更。
4. `data/classifier_train_multidomain.jsonl`（新規データファイル，実機生成）: 153 行，45 ペア全カバー
   （legal 絡み 9 ペアは各 5 件，他 36 ペアは各 3 件）。
5. `models/dispatch_multilabel_head.joblib`（新規モデル成果物，実機訓練）。
6. `scripts/train_dispatch_candidate_ranking_head.py`・`scripts/compute_iter59_ranking_stats.py`・
   `data/classifier_train.jsonl`・`models/domain_classifier.joblib`・`config.yaml`: **無変更を確認**
   （`git diff` に差分なし，`models/domain_classifier.joblib` の md5 は
   `b360ef827e258256888113a8293625a0`）。

**data/・models/ の扱い（計画からの軽微な逸脱と判断根拠）**: 計画には「`data/classifier_train_multidomain.jsonl`
はコミットする」とあったが，実際の `.gitignore` は `data/*`（`data/MANIFEST.md` 以外）と `models/` を
除外しており，`data/classifier_train.jsonl` 自身や Iter59 の `models/dispatch_candidate_ranking_head.joblib`
も git 追跡外で `data/MANIFEST.md` にも記載がない（イテレーション固有のオフライン成果物は
journal の実施コマンドで再現性を担保する既存運用，docs/d0003 F5）。この既存運用に合わせ，
今回もリポジトリへの force-add や MANIFEST.md への追記はせず，本節に sha256 を記録するに留めた
（Iter59 と同一の扱い）:
`data/classifier_train_multidomain.jsonl` = `ca286dae434ef27d3c92a04cb8e06581c6f85f27a2ad5bdc495ff4d1c90204bd`，
`models/dispatch_multilabel_head.joblib` = `004aaf512bf638cff182d5a4564f778e50b3c0de6be4f3e5805fd45eb1fe9077`。

**単体テスト（新規）**

- `tests/test_generate_multidomain_training_examples.py`: F1〜F4 フィルタの正常系・境界値，
  `_rows_for_pair()` の legal 優遇，`_generate_one()` の再試行打ち切り，`generate_all_rows()` の
  id 命名・重複拒否，`audit_leak()` の近似重複検出（高類似度／低類似度の両方）。
- `tests/test_train_multilabel_dispatch_head.py`: `_normalize_labels()` の str/list 混在正規化，
  `build_multilabel_targets()` の `mlb.classes_` 順序と `Y` の対応，A0 アサーションの成功系・
  行数不一致・120 件未満・ドメイン数不一致の失敗系，`_covered_domain_pairs()`，多ラベル訓練済み
  モデルの実際の予測，sklearn 1.9.0 の「list-of-lists 直接投入はエラー」という前提の回帰ガード。
- `tests/test_evaluate_dispatch_candidate_ranking.py`: `_load_head()` の新旧両形式対応，
  `_head_scores()` が両形式で同じドメイン名キーを返すこと，A6（整数キー・欠落キーの検出），
  N5（複合行を除外した単一ドメイン argmax 精度の計算），A5（Iter59 予測との不一致件数）。
  なお `OneVsRestClassifier.decision_function()` はクラス数がちょうど2のとき1次元配列に退化する
  sklearn の仕様があり（本レバーとは無関係の一般的な挙動），テストのトイデータは 3 ドメイン以上を
  使うことでこの縮退を回避した（4 件目のバグではなく，フィクスチャ設計上の注意点として記録）。
- 追加した 30 テストは全て `uv run pytest` で PASS。

**検証結果**

- `uv run ruff check .`: 新規・変更ファイルはすべて PASS。リポジトリ全体では 23 件のエラー
  （`scripts/analyze_iter52.py` 等の無関係な既存ファイルの f-string 未使用プレースホルダ等）が
  出るが，`git stash` で本イテレーションの変更を退避して再実行しても同じ 23 件が出ることを確認済み
  （本イテレーション由来ではない既存債務）。
- `uv run pytest`（全体）: 269 件中 257 PASS，12 件 FAIL。FAIL 12 件は `tests/test_build_dataset.py`
  （9 件）と `tests/test_train_domain_classifier.py`（3 件）で，いずれも
  `CalibratedClassifierCV` オブジェクトが `.classes_` 属性を持たない（`AttributeError`）という
  sklearn バージョン起因のエラーであり，本イテレーションが触れた
  `scripts/train_domain_classifier.py:201` 付近のコードは無変更．`git stash` で本イテレーションの
  変更を退避して同じ2ファイルを再実行しても同じ 12 件が同じ理由で FAIL することを確認済み
  （本イテレーション由来ではない既存の環境起因の失敗であり，`train_dispatch_candidate_ranking_head.py`
  や `train_multilabel_dispatch_head.py`（`CalibratedClassifierCV` を使わない）には影響しない）。

**実機での動作確認（生成・訓練・A0/A5/A6/A7/N5 の実測）**

wafl500 への SSH ローカルポートフォワード（`ssh -fNT -L 11435:localhost:11434 wafl500`，実行前から
稼働中だったものを流用）経由で `judge_model`／`nomic-embed-text` の双方が生きていることを確認し，
計画の「実施方法」コマンドをそのまま実行した。

1. 生成: `--per-pair 3 --per-pair-legal 5` で実行し，153/153 件が欠番なく生成された
   （スロット再試行での欠落は 0 件）。45 ペア全てを被覆（legal 絡み 9 ペア＝各 5 件，他 36 ペア＝各 3 件）。
2. **A7（リーク監査）**: `max_jaccard=0.1667`，`median_max_jaccard=0.0690`（閾値 0.9 を大きく下回り
   PASS）。
3. 訓練: `models/dispatch_multilabel_head.joblib` を作成。
   **A0（真の多ラベル信号）**: `(Y.sum(axis=1)>=2).sum()=153` が合成行数 153 と完全一致，
   `len(mlb.classes_)==10` かつ全て文字列で PASS。5-fold CV 診断で `legal` の
   `n_positive=122`（単一ラベル 77 ＋ legal 絡み合成 45 と整合），`cv_roc_auc=0.9193`（他ドメインと
   比べ遜色なく，過学習を示唆する明らかな異常なし）。
4. 採点（オフライン，`--embedding-cache results/iter59_query_embeddings.npz` を再利用）:
   出力先は次フェーズの公式ファイル名（`results/iter60_multilabel_ranking_predictions.jsonl`）とは
   別の一時パスに書き出した（**正式な統計検定は本フェーズでは実施しないため**，公式パスへの書き込みは
   次フェーズに委ねる）。
   - **A1**（`_assert_rank1_unchanged`）: 例外なし＝1600/1600 一致。
   - **A2**（`_assert_cost_neutral`）: 例外なし＝`mean_dispatch=2.000000`。
   - **A6**（`_assert_head_scores_are_domain_names`）: 例外なし＝全 1600 行で `head_scores` のキーが
     10 ドメイン名文字列と完全一致。
   - **A3**（`rank2_flip_rate`）: `0.45625`（対 baseline，0 ではないため WARNING 非発火）。
   - **A5**（対 Iter59）: `mismatches=527/1600`（`mismatch_rate=0.329375`，0 ではないため
     WARNING 非発火）＝本レバー固有の no-op ではないことを確認。
   - **N5**（単一ドメイン argmax，文体ショートカット検出）: `accuracy=0.6107`（916/1500），
     floor 0.590 を上回り PASS（Iter59 実測 0.610 とほぼ同水準）。
   - 参考値（**速報，正式な McNemar 検定は未実施**）: `compound_domain_set_recall=0.48`
     （baseline 0.345，Iter59 0.350 から見て大きく上振れ）。この数値は
     `scripts/compute_iter59_ranking_stats.py` を通していない生の速報値であり，S1〜S4・N1〜N4 の
     正式判定は次フェーズが `--output results/iter60_multilabel_ranking_predictions.jsonl` へ書き出した
     上で同スクリプトを実行して行うこと。

**次フェーズへの申し送り**

- 生成データに軽微な品質のばらつきを確認した（例:
  `synth-natural_science-social_science-002` の query が
  「自然科学と社会科学の両方の知識が必要な相談文：」というプロンプトのテンプレート文言の
  ほぼそのままの echo になっており，F1〜F4 のいずれの機械的フィルタにも掛からず通過している）。
  計画で事前登録された F1〜F4 以外のフィルタ（内容の実質性チェック等）は本フェーズの単一レバー
  原則の範囲外として追加しなかったが，次フェーズの考察でこの種の低品質行の混入率と
  `compound_domain_set_recall` への影響を注意深く見ること（R4 の「件数不足 vs 信号無効」の
  切り分けにも関わりうる）。
- 上記の速報 `compound_domain_set_recall=0.48` は実データでの寄り道確認であり，**本フェーズでは
  正式な統計検定（S1 の exact McNemar 等）を意図的に実施していない**。次フェーズは計画の
  「実施方法」手順 2)〜3) を公式パス（`results/iter60_multilabel_ranking_predictions.jsonl`，
  `results/iter60_stats.json`）に対してそのまま再実行し，S1〜S4・N1〜N4 を正式判定すること
  （本フェーズの速報値の再現性は担保されているはず＝同一の入力ファイル・同一コードで再計算するのみ）。
- `data/classifier_train_multidomain.jsonl` と `models/dispatch_multilabel_head.joblib` は
  ローカルディスク上に実ファイルとして現存する（sha256 は本節に記録済み）。`git status` は無関係な
  未コミット変更（`config.yaml` の `embed_node_host: wafl502→wafl-ctrl5`,
  `.claude/research/journal.md`・`state.json` 等）を含んでいたが，本フェーズはこれらに一切触れて
  いない（`config.yaml` は計画どおり無変更）。

### 実験・分析(実行) (Iter60)

本フェーズも新規の実機トラフィック（probe/dispatch/LLM生成）を一切発生させていない。
`results/iter59_query_embeddings.npz`（1600件）が基準線 `results/20260918_202613/results.jsonl`
の全1600 IDを事前に完全カバーしていることを独立に確認した上で（キャッシュ欠落0件），
`scripts/evaluate_dispatch_candidate_ranking.py` を公式パスで実行し，
`results/iter60_multilabel_ranking_predictions.jsonl`（1600行）を生成した
（実行コマンド: `uv run python -m scripts.evaluate_dispatch_candidate_ranking --baseline
results/20260918_202613/results.jsonl --head models/dispatch_multilabel_head.joblib
--embedding-model nomic-embed-text --ollama-host 127.0.0.1 --ollama-port 11435
--embedding-cache results/iter59_query_embeddings.npz --iter59-predictions
results/iter59_ovr_ranking_predictions.jsonl --output
results/iter60_multilabel_ranking_predictions.jsonl`）。キャッシュフルヒットのため実際の
embed 呼び出しは0件（完全オフライン）。続けて `scripts/compute_iter59_ranking_stats.py`
（計画どおり無変更で流用）をそのまま実行し，`results/iter60_stats.json` に S1〜S4・N1〜N4 を
保存した（実行コマンド: `uv run python -m scripts.compute_iter59_ranking_stats --baseline
results/20260918_202613/results.jsonl --new
results/iter60_multilabel_ranking_predictions.jsonl --output results/iter60_stats.json`）。
新規に自作した統計ロジックはなく，`metrics.py` の既存関数（`compute_compound_coverage_metrics`・
`_mcnemar_from_correctness`・`compute_top1_accuracy`）のみを使用（Iter59と同一コード）。
**解釈・採否判断はこのフェーズでは行わない**（次の analyst フェーズに委ねる）。

**A0/A5/A6/A7・N5 の独立再確認（実装フェーズの数値との一致確認）**

- **A0**（真の多ラベル信号）: `data/classifier_train.jsonl`（1427行）と
  `data/classifier_train_multidomain.jsonl`（153行）を再読込し，`train_multilabel_dispatch_head.py`
  の `build_multilabel_targets()`/`_assert_a0_true_multilabel_signal()`/`_covered_domain_pairs()`
  を embed なし（`row["domain"]` のみを使うラベル抽出のみ）で直接呼び出して独立検算した。
  `multilabel_row_count=153`（合成行数と完全一致，かつ床値120以上），`mlb.classes_` は10ドメイン名
  文字列，被覆ペア数=45（45ペア全カバー）。**実装フェーズの報告値と完全一致，PASS**。
- **A5**（対Iter59不一致）: 上記の公式採点実行時に算出。`mismatches=527/1600`
  （`mismatch_rate=0.329375`）。**実装フェーズの速報値（527/1600，0.329375）と完全一致**，
  WARNING非発火（0件ではない）＝本レバー固有のno-opではないことを確認。
- **A6**（`head_scores`のキーがドメイン名文字列）: 同じ採点実行で例外なし＝全1600行で
  `head_scores`のキーが10ドメイン名文字列と完全一致。**PASS**（実装フェーズと一致）。
- **A7**（リーク監査）: `uv run python -m scripts.generate_multidomain_training_examples
  --audit-leak --output data/classifier_train_multidomain.jsonl` を独立に再実行（ネットワーク
  呼び出しなし，ローカルファイル比較のみ）。`max_jaccard=0.1667`，`median_max_jaccard=0.0690`
  （閾値0.9を大きく下回りPASS）。**実装フェーズの報告値と完全一致**。
- **N5**（単一ドメイン argmax，文体ショートカット検出）: 上記の公式採点実行で
  `accuracy=0.6107`（916/1500），floor 0.590を上回りPASS。**実装フェーズの速報値（0.6107）と
  完全一致**。

**S1〜S4・N1〜N4（`results/iter60_stats.json` より，正式判定）**

| 項目 | 値 | 判定 |
|---|---|---|
| S1（主基準，n=200ペア，exact binomtest） | 改善38／悪化11／discordant49，p=0.0001420 | **PASS**（p<0.05） |
| S1（参考，連続性補正chi2） | chi2=13.796，p=0.0002038 | 参考値，同じくPASS方向 |
| S2（効果量下限） | recall 0.345→0.480，Δ=+0.135pt（floor +0.04pt） | **PASS** |
| S3（コスト中立） | 全1600行 len=2，rank1≠rank2重複0，mean_dispatch=2.000000 | **PASS** |
| S4（発火の証拠） | rank2_flip_rate=0.45625（730/1600） | **PASS**（>0） |
| N1（rank_1完全不変） | 不一致0/1600 | **PASS** |
| N2（top1_accuracy不変） | baseline 0.5975 → new 0.5975（完全一致，留保はIter59と同一文言でstats.json内に記録） | **PASS** |
| N3（legal非退行） | legal自身の被覆 8/30 → 20/30 | **PASS**（8を下回らない） |
| N4（内訳，判定なし） | legal絡み60ペア（改善17/悪化1/不変42），medical絡み32ペア（改善4/悪化1/不変27），その他108ペア（改善17/悪化9/不変82） | 報告のみ |

補足: `compute_iter59_ranking_stats.py` は計画どおり無変更のため，`matches_implementation_phase_*`
フィールド（`_IMPLEMENTATION_PHASE_COMPOUND_DOMAIN_SET_RECALL=0.35`,
`_IMPLEMENTATION_PHASE_RANK2_FLIP_RATE=0.356875`）は**Iter59自身の実装フェーズ速報値**であり，
本イテレーション（Iter60）の値と比較する定数ではない（`false`と出るのは想定どおりであり異常ではない）。
Iter60自身の速報値（`compound_domain_set_recall=0.48`，`rank2_flip_rate=0.45625`）とは，本フェーズの
`new_compound_domain_set_recall=0.48`・`rank2_flip_rate=0.45625`が完全一致しており，実装フェーズと
本フェーズの間で不一致はない。

**PASS/FAIL集計**: S1=PASS，S2=PASS，S3=PASS，S4=PASS，N1=PASS，N2=PASS，N3=PASS
（N4は判定なし・報告のみ）。A0/A5/A6/A7・N5 も全てPASSかつ実装フェーズの数値と完全一致。
内容面の解釈・採否判断（採用／partial／棄却の別）は次の analyst フェーズに委ねる。

**成果物**: `results/iter60_multilabel_ranking_predictions.jsonl`（1600行，公式パス），
`results/iter60_stats.json`（S1〜S4・N1〜N4の実測値，機械可読，公式パス）。既存ファイル
（`scripts/evaluate_dispatch_candidate_ranking.py`・`scripts/compute_iter59_ranking_stats.py`・
`models/dispatch_multilabel_head.joblib`・`data/classifier_train_multidomain.jsonl`・
`models/domain_classifier.joblib`（md5 `b360ef827e258256888113a8293625a0`，不変を再確認）・
`config.yaml`（本フェーズでは無変更，既存の無関係な未コミット差分1行のみ残存）は本フェーズでは
変更していない。

### 分析(解釈) (Iter60)

**1. 独立検算（`metrics.py` の既存関数のみ使用．不一致 0 件）**

`results/20260918_202613/results.jsonl`（基準線）・`results/iter60_multilabel_ranking_predictions.jsonl`
（新）・`results/iter59_ovr_ranking_predictions.jsonl`（対照）を読み直し，`metrics.compute_compound_coverage_metrics()`・
`metrics._mcnemar_from_correctness()`・`metrics.compute_top1_accuracy()` と `scipy.stats.binomtest` のみで
`results/iter60_stats.json` の全項目を再計算した（作業用スクリプトは実行後に削除．リポジトリへの
恒久的な追加はしていない）．**全項目が完全一致し，不一致は 1 件も無い**．

| 項目 | 独立再計算値 | stats.json |
|---|---|---|
| compound_domain_set_recall（基準線 / Iter59 / 新） | 0.345（69/200） / 0.350（70/200） / **0.480（96/200）** | 一致 |
| compound_domain_jaccard_mean | 0.2400 → 0.3667（Iter59 は 0.2500） | （未記録，本フェーズで追加算出） |
| S1 ペア比較（n=200） | 改善 38 / 悪化 11 / discordant 49 / chi2=13.7959 / p_cc=0.0002038 | 一致 |
| S1 exact binomtest | p=0.00014197 | 一致 |
| S2 効果量 | Δ=+0.135pt（floor +0.04pt） | 一致 |
| S3 コスト | 長さ分布 `{2: 1600}`，rank1=rank2 重複 0，mean_dispatch=2.000000 | 一致 |
| S4 rank2_flip_rate | 0.45625（730/1600） | 一致 |
| A5（対 Iter59 不一致） | 527/1600（0.329375） | 一致 |
| A6（head_scores キー） | 全 1600 行で 10 ドメイン名文字列の単一キー集合（ユニーク keyset 数 = 1） | 一致 |
| N1 rank_1 不一致 | 0/1600 | 一致 |
| N2 top1_accuracy | 0.5975 → 0.5975 | 一致 |
| N3 legal 自身の被覆 | 8/30 → 20/30 | 一致 |
| N5 単一ドメイン argmax 正解率 | 0.610667（916/1500，floor 0.590） | 一致 |

**2. ノイズか信号か — 信号である（3 つの独立な根拠）**

- **(i) ランダム性が存在しない**: 本イテレーションは決定論的オフライン採点で，embed はキャッシュ
  フルヒット（実呼び出し 0 件）．config.yml success_criteria (5) の 3SD=2.6pt のノイズ床は軸②③
  （生成のランダム性を含む answer_quality / end_to_end）に対するもので，軸①のルーティング指標には
  適用しない（同 (5) 末尾に明記）．反復間ノイズはゼロである．
- **(ii) 標本ノイズに対しても十分大きい**: 残る不確実性は複合設問 100 問（200 ペア）の標本誤差のみ．
  ペア差の近似 SE = sqrt(38+11)/200 = **3.5pt**，Δ=+13.5pt は **3.86 SE**，95% CI は
  **[+6.6pt, +20.4pt]** で，事前登録した効果量下限 +4.0pt は CI 下端よりさらに下にある．
  Iter59 の CI（[-3.8pt, +4.8pt]，+4pt が内側）とは対照的に，今回は下限自体が閾値を超えている．
- **(iii) 過去イテレーションの実測ばらつきと比べても桁が違う**: 同一指標の履歴は Iter47 0.345 /
  Iter48 0.345 / Iter46 0.360 / Iter58 0.345 / Iter59 0.350 で，レバーを振っても ±1.5pt の帯に
  収まり続けていた．今回の +13.5pt はこの帯の 9 倍である．

**3. Iter59 との対比が示すこと — 効いたのは「ヘッド構造」ではなく「教師信号」**

Iter59 と Iter60 は，embedding・ヘッド構造（`OneVsRestClassifier(LogisticRegression(max_iter=1000,
class_weight="balanced"))`）・推論経路・採点スクリプト・基準線・成功条件のすべてが同一で，
**訓練データに 2 ドメインラベル行 153 件が入っているかどうかだけが違う**．結果は
0.350（p=1.0，Wilcoxon p=0.914 で効果ゼロ）→ **0.480**．Iter59 予測を基準にした exact McNemar でも
**改善 36 / 悪化 10，p=0.000156**（本フェーズで追加算出）．

この対比の意味は，Iter58・Iter59 の考察で 2 回連続して立てられた命題
**「単一ラベル訓練データ（1 行 1 ドメイン）の加工・後処理では多ラベル性は生まれない」が正しく，
かつその裏（教師信号を多ラベル化すれば生まれる）も成立した**，ということである．
単一レバーの帰属としては，「OvR という分解の導入」ではなく「2 ドメイン同時ラベルの訓練事例の存在」
に因果を帰すのが妥当で，Iter59 がその切り分けを担保している．ただし厳密には，本イテレーションが
動かしたのは「合成 153 件の追加」という 1 つの操作であり，「合成の質」「件数」「ペア配分」の
どれが効いたかまでは分離していない（R4 の申し送りどおり）．

**4. 発火率 0.45625（730/1600）の意味 — Iter59 より広く動いているが，暴走ではない**

- Iter59 の 0.356875（571 行）から 0.45625（730 行）へ増え，Iter59 予測とも 527 行で異なる．
  訓練データの 10.7%（153/1427+153）を入れ替えただけで rank_2 の 4 割超が動いており，
  **rank_2 の順位付けは教師信号の構成に非常に敏感**である．
- 重要なのは flip の「量」ではなく「収支」である．flip した compound 行 64/100 について，
  rank_2 が正解ドメインを当てた件数は **基準線 11 → 新 38**（Iter59 は flip 31 行で 旧9→新10 と
  ほぼ収支ゼロだった）．compound 100 行全体でも rank_2 の的中は **28 → 55**（Iter59 は 29）．
  **今回の flip は「当たりと外れをほぼ同数入れ替える」のではなく，一方向に的中を増やしている**．
- 単一ドメイン 1500 行でも 666 行（44.4%）が flip しているが，N2（top1 0.5975 不変）・N5（argmax
  正解率 0.6107，Iter59 0.610 と同水準）が保たれており，rank_1 の判別力は損なわれていない．

**5. legal への偏りの検証（N3・N4 の分解）— 偏りは「ある」が，legal だけでは説明できない**

legal 自身の被覆 8/30→20/30 は 200 ペア中の 12 件の改善で，全改善 38 件の 32% を占める．
ドメインペア単位まで分解して確認した（本フェーズで追加算出）．

| 切り口 | 改善 | 悪化 | exact p | recall |
|---|---|---|---|---|
| 全体（200 ペア） | 38 | 11 | 0.000142 | 0.345→0.480 |
| legal 絡み（60 ペア） | 17 | 1 | 0.000145 | — |
| legal×medical のみ（24 ペア，最多 12 行） | 8 | 0 | — | — |
| legal×medical を除く（176 ペア） | 30 | 11 | **0.00432** | — |
| legal 絡みを全部除く（140 ペア） | 21 | 10 | 0.0708 | 0.350→0.4286（+7.9pt） |

- **改善は 26 ペア種（45 ペア種中）に分散しており**，単一ペアに依存していない．最大の寄与源である
  legal×medical（テスト集合で最多の 12 行）を丸ごと除いても **p=0.0043 で有意**であり，
  「legal×medical だけで作られた見かけの改善」ではない．
- ただし **legal 絡みを全部除くと p=0.0708 と有意水準を割る**（効果量は +7.9pt で方向は一貫）．
  n=140・discordant 31 に落ちるため検出力の問題でもあるが，**主基準の有意性は legal 絡みの寄与に
  相当程度依存している**と正直に記述すべきである．
- **設計上の留保（重要）**: 計画は legal 絡み 9 ペアのみ合成件数を 3→5 に増やしており，合成件数別の
  改善収支は **5 件配分＝改善 17/悪化 1，3 件配分＝改善 21/悪化 10** と明確に差がある．
  この配分は「legal は訓練 77 件と最少」という訓練側の理由に加え，**調査 Q3 が評価集合
  `_COMPOUND_QUESTIONS` のペア別件数分布（legal×medical が 12 件で最多）を参照して決めた**もので
  ある（テキストは参照していないが，**テスト集合の分布情報が設計に入っている**）．
  これは A7（Jaccard 監査）では検出できない種類の弱いリークであり，reflector は
  「本文リークは無い（A7 max 0.1667）が，ペア配分という設計レベルの事前知識は入っている」
  と区別して扱うべきである．
- **反面の証拠（偏り説に不利）**: legal を rank_2 に選んだ compound 行の的中率は 19/32=59%
  であり，「compound 行なら無条件に legal を出す」方針の期待値（legal はテスト 100 行中 30 行に
  登場＝30%）の約 2 倍である．さらに下記 6. の content-blind 対照が決定的である．

**6. 過学習・文体ショートカット・prior シフトの可能性を潰す追加検証（本フェーズ独自）**

A7（3-gram Jaccard 最大 0.1667）だけでは「本文の剽窃が無い」ことしか言えないため，
**「内容を見ずに事前分布だけをずらした結果ではないか」**という代替説明を直接検定した．

- **content-blind 対照**: rank_2 を内容に関係なく固定ドメインにした場合の
  compound_domain_set_recall を計算すると，**legal 固定 = 0.350**，medical 固定 = 0.310，
  education 固定 = 0.285，business_economics 固定 = 0.240．すなわち**最良の内容非依存方策でも
  0.350 にしか届かず（偶然にも Iter59 と同値），実測 0.480 はそれを 13pt 上回る**．
  改善は事前分布のシフトでは説明できず，**行ごとの内容に反応している**．
- **文体ショートカット**: N5=0.6107（Iter59 0.610，floor 0.590）で，四択文 1500 行に対する
  判別力は落ちていない．「自然文なら多ラベル」という短絡を学んだなら四択側が崩れるはずだが
  崩れていない．
- **合成データの品質**: 153 件を目視・正規表現で走査したところ，**7 件（4.6%）がプロンプト文言の
  echo**（例: `synth-mathematics-natural_science-003` = 「数学と自然科学の両方の知識が必要な
  相談文：」）で，実装フェーズの申し送りどおり F1〜F4 を素通りしている．**低品質行が 4.6% 混入した
  状態でこの効果量が出ている**ため，効果は品質の良い行が担っていると考えられ，
  フィルタ強化には伸びしろが残っている（悪化方向の交絡ではない）．
- **上限との距離**: rank_1 のみ（k=1 相当）の recall は 0.205，固定 k=2・rank_1 凍結下の
  オラクル上限は 0.705（compound 行で rank_1 が正解しているのは 41/100 で Iter59 と同一）．
  0.345→0.480 は**残余ギャップ 36.0pt のうち 13.5pt（37.5%）を埋めた**ことになる．
  Iter59 は同じ尺度で 1.4% しか埋めていない．

**7. Iter58 の教訓（改善がコスト増で説明できないか）の確認**

Iter58 は mean_dispatch の増加と改善が交絡した．今回は S3 が **全 1600 行 len=2・rank_1≠rank_2 重複 0・
mean_dispatch=2.000000**（独立再計算で一致）であり，**基準線と新方式は同じ 2 ノードを常に叩く**．
dispatch 回数・k・閾値のいずれも変えていないため，コストで説明できる余地は構造的に無い．
なお実機のレイテンシ・回答品質は本イテレーションでは未測定（計画 R3）であり，
「コスト中立」は dispatch 回数についての主張に限定される．

**8. 仮説との整合**

計画の仮説「2 位枠が改善しない原因は推論側の工夫不足ではなく訓練データに多ラベル事例が無いこと．
多ラベル教師信号を作れば，コストを増やさずに compound_domain_set_recall が改善する」は，
**主張・機序ともに支持された**．想定外の挙動（言語崩れ・発散・OOM・整数キーバグ A6・no-op）は
いずれも観測されていない．想定していなかった副次的な観測は 2 点:
- **education の退行**: education 自身の被覆 9/20→**4/20**（-5）で，ドメイン別の悪化 11 件のうち
  5 件が education．compound 行の rank_2 に education が選ばれる回数が 23→2 に激減している
  （基準線は全体で rank_2=education を 421/1600 と過剰に出しており，その過剰さが偶然
  education compound 行を拾っていた）．education は Iter32〜53 で 10 回以上レバーを振っても
  動かなかった問題ドメインであり，**今回の改善の裏で唯一明確に退行している**点は記録に値する
  （事前登録の非退行条件には education の項目が無いため FAIL ではないが，N4 の趣旨に照らして報告する）．
- **social_science の大幅改善**: 0/18→7/18．基準線で唯一の被覆ゼロだったドメインが動いた．

**9. 判定の確信度と追加反復の要否**

- **確信度: 高**．(i) 決定論的で反復間ノイズがゼロ，(ii) 独立検算の不一致 0 件，(iii) 効果量が
  標本 SE の 3.86 倍で CI 下端も事前登録閾値の上，(iv) content-blind 対照（0.350）を 13pt 上回り
  prior シフト説を排除，(v) Iter59 という同一構造の対照群が存在する．
- **同一設計での追加反復は不要**（決定論的なので同じ値が再現するだけ．Iter58/59 と同じ論理）．
- **確信度が相対的に低い部分（追加検証があるとすれば）**: (a) legal 絡みを除くと p=0.0708 で
  有意でない，(b) 合成件数のペア配分にテスト集合の分布情報が入っている，(c) 合成 4.6% が低品質，
  (d) 実行時経路では未検証（オフライン採点のみ）．(a)(b) は
  **「legal 絡みも一律 3 件にした配分での再訓練」**という 1 変数の追試で切り分けられる．

**次フェーズ（rc-reflector）への申し送り**

- **強い所見**: 事前登録した S1〜S4 が全て PASS（S1 p=0.000142，S2 Δ=+0.135pt），N1〜N3・N5 も
  全て PASS．A0/A5/A6/A7 も実装フェーズと完全一致．d0004 §4 型の no-op ではなく，
  コスト中立（mean_dispatch=2.000000）で達成されている．Iter59 という同一構造・教師信号のみ異なる
  対照群があるため，**「多ラベル教師信号そのものが必要だった」という因果的主張が本研究で初めて
  成立する**．Iter58・Iter59 の 2 連続陰性の解釈（単一ラベルの加工では多ラベル性は生まれない）が，
  その対偶の側から裏付けられた．
- **留保点（採否判断の強度に影響する）**:
  1. legal 絡み 60 ペアを除くと exact p=0.0708（効果量 +7.9pt，方向は一貫）．主基準の有意性は
     legal 絡みの寄与に相当程度依存する．ただし最大寄与ペア legal×medical を除いても p=0.0043．
  2. 合成件数のペア配分（legal 絡みのみ 5 件）の決定に，評価集合のペア別件数分布という
     **テスト集合由来の情報**が入っている．本文リークは無い（A7 max_jaccard=0.1667）が，
     設計レベルの弱いリークとして区別して記録すべきである．
  3. education 自身の被覆が 9/20→4/20 と退行（事前登録の非退行条件外）．
  4. 合成 153 件のうち 7 件（4.6%）がプロンプト echo の低品質行．
  5. オフライン採点であり，実行時経路（`node.py:214` / `run_experiment.py:93`）・
     `answer_quality` / `end_to_end` / レイテンシは未検証．
- **リスク**: 実行時経路への配線は `config.yaml` のスキーマ変更を伴い，Iter58 と同型の no-op を
  避けるには 2 箇所を同時に変更する必要がある（計画 R3）．**ユーザー確認が必要**であり，
  reflector が自律的に着手してよい範囲を超える．
- **次の一手の候補（分析フェーズとしての示唆であり，採否判断ではない）**:
  (a) 留保 1・2 を潰す追試（legal 絡みも一律 3 件＝135 件での再訓練．1 変数のみの変更で
      オフライン完結），(b) 合成件数のスケールアップ（R4 の「信号無効 vs 件数不足」は今回
      「信号有効」側に決着したので，残る問いは件数の収穫逓減点），(c) 生成フィルタの強化
      （echo 行 4.6% の除去），(d) 実行時経路への配線（要ユーザー確認）．
  なお本レバーの外に残る最大のボトルネックは依然 rank_1 側（compound 行で 41/100）である．

### 考察・次計画 / イテレーション完了サマリー (Iter60)

**単一レバー**: `multilabel_training_signal = synthetic_two_domain_training_examples`
（`judge_model` による LLM 生成で 2 ドメイン同時ラベルの相談文 153 件を新規作成し，既存 1427 行と
結合して `MultiLabelBinarizer` 経由で OvR ヘッドを再訓練．ヘッド構造・推論経路・採点手続き・基準線は
Iter59 と完全に同一）．

**結果（事前登録項目，`results/iter60_stats.json`）**: S1 exact binomtest p=0.0001420（改善 38／悪化 11，
n=200），S2 `compound_domain_set_recall` 0.345→0.480（Δ=+0.135pt，下限 +0.04pt），
S3 mean_dispatch=2.000000（コスト中立），S4 rank2_flip_rate=0.45625・対 Iter59 不一致 527/1600．
N1（rank_1 不変 0/1600）・N2（top1 0.5975 完全一致）・N3（legal 8/30→20/30）・N5（単一ドメイン
argmax 0.6107，floor 0.590）も全 PASS．A0/A5/A6/A7 は実装フェーズと独立再計算で完全一致
（A7 max_jaccard=0.1667）．**事前登録 7 項目＋補助アサーション 5 項目が全 PASS，FAIL 0 件**．

**判定: adopted（採用．ただし下記 3 点の留保を対外記述に必須で付す）**

判定根拠は 4 点である．
1. **事前登録の判定規則にそのまま該当する**．計画は「S1〜S4 全て充足なら採用」と事前登録し，
   「判定は事後変更しない」と明記していた．全 PASS で FAIL が 1 件も無い以上，留保を理由に
   事後的に判定規則を書き換えて partial へ降格させることは，本研究が Iter29 以降積み上げてきた
   事前登録運用そのものを壊す．留保は**判定の格下げではなく次イテレーションの追試義務**として扱う．
2. **Iter58（partial に留めた事例）とは交絡の質が違う**．Iter58 の partial は「改善の大半が
   dispatch 呼び出し +19.97% の純増で説明でき，gap 信号固有の寄与が有意でない」という
   **主張そのものを無効化しうる交絡**が理由だった．今回は S3（mean_dispatch=2.000000，全 1600 行
   len=2）によりコストでの説明余地が構造的に無く，さらに content-blind 対照（rank_2 を内容に
   関係なく固定ドメインにした場合の最良値 = legal 固定 0.350）を 13pt 上回るため，
   prior シフトでの説明も排除されている．
3. **因果の帰属先が対照群で担保されている**．Iter59 は embedding・ヘッド構造・推論経路・採点
   スクリプト・基準線・成功条件のすべてが同一で，訓練データの多ラベル行 153 件の有無だけが違い，
   0.350（p=1.0）だった．Iter59 予測を基準にした exact McNemar でも改善 36／悪化 10，p=0.000156．
   「効いたのは OvR というヘッド構造ではなく 2 ドメイン同時ラベルという教師信号である」という
   帰属は，本研究で初めて実験的に成立した．
4. **ノイズではない**．決定論的オフライン採点（embed はキャッシュフルヒット，実呼び出し 0 件）で
   反復間ノイズはゼロ．残る標本誤差に対しても Δ=+13.5pt は SE 3.5pt の 3.86 倍，95% CI
   [+6.6pt, +20.4pt] の下端が事前登録閾値 +4.0pt の上にある．同一指標は Iter46〜59 を通じて
   0.345〜0.360 の ±1.5pt 帯に張り付いていた．

**留保（adopted の効力範囲を限定する．対外記述で必ず併記すること）**

- **R-A: 効果量 +13.5pt は汎化推定値として引用しない**．合成件数のペア配分（legal 絡み 9 ペアのみ
  3→5 件）の決定に，評価集合 `_COMPOUND_QUESTIONS` のペア別件数分布（legal×medical が 12 行で最多）
  という**テスト集合由来の情報**が入っている．本文リークは無い（A7 max_jaccard=0.1667）が，
  A7 では検出できない**設計レベルの弱いリーク**である．実際，5 件配分＝改善 17／悪化 1 に対し
  3 件配分＝改善 21／悪化 10 と収支に差がある．
- **R-B: 主基準の有意性は legal 絡みの寄与に相当程度依存する**．legal 絡み 60 ペアを除くと
  exact p=0.0708（+7.9pt，方向は一貫）．ただし最大寄与ペア legal×medical を除いても p=0.0043 で
  有意であり，改善は 45 ペア種中 26 ペア種に分散している．「単一ペアの偶然」ではないが，
  「legal 非依存」とも言えない．
- **R-C: education が唯一明確に退行している**（自身の被覆 9/20→4/20，rank_2=education の選択が
  23→2 に激減）．事前登録の非退行条件に education の項目が無いため FAIL ではないが，education は
  Iter32〜53 で 10 回以上レバーを振っても動かなかった問題ドメインであり，改善の裏で犠牲が出ている
  ことは記録しておく．なお基準線は rank_2=education を 1600 行中 421 行と過剰に出しており，
  その過剰さが偶然 education compound 行を拾っていた側面がある．
- 補足: 合成 153 件のうち 7 件（4.6%）がプロンプト文言の echo（低品質行）である．これは
  悪化方向の交絡であり，効果を水増しする方向ではない（フィルタ強化の伸びしろ）．

**本番経路への配線: 今回は実施しない（保留．要人間判断）**

`models/dispatch_multilabel_head.joblib` は本番経路から参照されない状態のまま保持する
（`config.yaml`・`node.py`・`http_server.py` は本イテレーションで無変更．`models/domain_classifier.joblib`
は md5 `b360ef827e258256888113a8293625a0` で不変）．配線を見送る理由は 2 つ:
(1) 配線は `config.yaml` の**スキーマ変更**（多ラベルヘッドのパス・使用フラグの新設）を伴い，
skill の自律判断ポリシー上ユーザー確認が必要である．(2) Iter58 と同型の no-op を避けるには
`node.py:214` と `run_experiment.py:93` を**同時に**変更する必要があり，これは単一レバー原則の下では
それ自体を 1 イテレーションとして設計すべき作業量である．R-A の追試で効果量の内部妥当性を固めてから
配線する方が，実機 1600 問（約 100 分）の投資に見合う．

**学び（次の自分への申し送り）**

1. **2 イテレーション連続の陰性の「対偶」を狙う設計は情報量が大きい**．Iter58・Iter59 は
   「単一ラベルデータの加工では多ラベル性は生まれない」を独立に 2 度示した．Iter60 はその裏
   （教師信号を多ラベル化すれば生まれる）を，**Iter59 と 1 変数だけ違う構成**で検証した．
   陰性結果を対照群として設計に組み込めたことが，本研究で初めて因果的主張を可能にした．
   今後も陰性が出たら「同じ枠組みで 3 度目」ではなく「その命題の対偶を検証できる最小差分の設計」を
   探すこと．
2. **A7（本文の 3-gram Jaccard 監査）はリーク監査として不十分である**．本文の剽窃は検出できるが，
   「合成件数のペア配分」のような**設計パラメータ経由のテスト集合情報の流入**は素通りする．
   今後リーク監査を設計する際は「生成物のテキスト」だけでなく「生成の設計判断がテスト集合の
   統計を参照していないか」をチェックリストに入れること（今回はこれを見落とし，事後の分析で
   初めて気づいた）．
3. **rank_2 の順位付けは教師信号の構成に極端に敏感である**．訓練データの 10.7%（153/1580）を
   足しただけで rank_2 の 45.6% が動いた．しかも flip の収支は一方向（compound 行の rank_2 的中
   28→55）で，Iter59 の「flip はするが収支ゼロ」とは質が違う．**flip rate は発火の証拠にはなるが
   改善の証拠にはならない**ので，今後も必ず「flip した行の的中収支」まで見ること．
4. **機械的フィルタ F1〜F4（文字数・四択マーカー・完全一致重複・複数行）はプロンプト echo を
   通す**．「〜の両方の知識が必要な相談文：」がそのまま query になった行が 7 件（4.6%）残った．
   LLM 生成データを使う次のイテレーションでは，プロンプト由来の定型句との部分一致チェックを
   フィルタに追加すること．
5. **残るボトルネックは rank_1 側である**．固定 k=2・rank_1 凍結下のオラクル上限は 0.705 で，
   今回はその残余ギャップ 36.0pt のうち 13.5pt（37.5%）を埋めた．上限そのものを上げるには
   compound 行での rank_1 正解率 41/100 を動かす必要があり，これは本レバーの外側の問題である．

**次の一手**

`multilabel_training_signal` は values が単一値（`synthetic_two_domain_training_examples`）のため
**今回でクローズ（試し切り）**．skill の停止条件 1 に従い，本イテレーションの学び（留保 R-A・R-B）
から新レバーを考案し，config.yml の levers 末尾へ追記した．

- **新レバー**: `multilabel_pair_allocation = uniform_three_per_pair`
  （45 ペア一律 3 件＝135 件で再生成・再訓練し，legal 優遇 +2 件を除去する）．
- **選定理由**: 分析フェーズが挙げた候補 (a)〜(d) のうち，(a) が **R-A（設計リーク）と R-B
  （legal 依存）を同時に，1 変数の変更だけで切り分けられる唯一の設計**である．オフライン完結・
  決定論的・実機トラフィックは生成分のみ（135 件）でコストが小さく，Iter60 が対照群として
  そのまま機能する．ここで有意性が残れば adopted の効力範囲を「汎化可能な効果」まで広げられ，
  失われれば「効果は legal 絡みの厚い配分に依存」と主張を正しく弱められる．いずれに転んでも
  結論が確定する．(b) 件数スケールアップと (c) フィルタ強化は，配分という交絡を残したまま
  件数・品質を動かすと帰属が曖昧になるため後回し．(d) 配線は上記のとおり要ユーザー確認．
- **次イテレーション名**: 「合成ペア配分の均一化による設計リークの切り分け」．

**要人間判断**

1. **実行時経路への配線（`config.yaml` のスキーマ変更）**．adopted の成果を実機に反映するには
   `config.yaml` に多ラベルヘッドの設定項目を新設し，`node.py:214` と `run_experiment.py:93` を
   同時に変更する必要がある．スキーマ変更は自律判断の範囲外のため承認を求める．
   推奨は「次イテレーション（配分の均一化）の結果を見てから配線する」．
2. **効果量の対外記述**．R-A のとおり +13.5pt は設計リークを含む値である．論文・報告で引用する
   場合は「legal 絡みを除くと +7.9pt（p=0.0708）」を必ず併記するか，次イテレーションの
   均一配分での値を正式値とするか，方針の確認が要る．

**コミット**: `b36cc3f`（本記録の追記は後続コミット）

---

## Iteration 59: multi-labelヘッドによるdispatch候補順位付けのオフライン検証

### 調査 (Iter59)

**問い**
- Q1: 既存の argmax 単一ラベル分類器の上に，多ラベルランキング（binary relevance / OvR sigmoid ヘッド）を
  後付けする設計の先行研究・実践知見（較正上の注意点，多段設計の先行例，少量データでの過学習対策）。
- Q2（最優先）: 訓練スクリプト・データスキーマ・実行時経路・`results.jsonl` のフィールド・
  `metrics.py` の関連関数・`select_dispatch_targets()` 系の全呼び出し元を正確に特定する。
- Q3: Q1・Q2 を踏まえた，次の計画フェーズが実装計画を書けるレベルの具体的な手順。

**分かったこと（Q1: 先行研究）**

- **OvR/binary relevance の較正が抱える構造的問題は，このリポジトリのコード自身のコメントと
  scikit-learn 公式ドキュメントの両方で一致して確認できた**。`scikit-learn.org/stable/modules/
  calibration.html` §1.16.3.3 は「`CalibratedClassifierCV` はマルチクラスに対して各クラスを
  OvR 方式で個別に較正する．各クラスの較正済み確率は独立に予測されるため合計が 1 にならず，
  事後的な正規化（renormalization）が行われる」と明記している（出典:
  https://scikit-learn.org/stable/modules/calibration.html ，
  補足の Stack Overflow スレッド https://stackoverflow.com/questions/60110209 も同旨）．
  本リポジトリの `scripts/train_domain_classifier.py`（L39-48）・`classifier.py`（L11-19）の
  コメントは，この事実を根拠に「temperature scaling は単一スカラーで logits 全体を再スケールする
  ため softmax 出力が合計 1 を保つが，isotonic/platt は per-class 較正器を持つため合計が 1 に
  ならない」という Iter31 の較正手法選定理由を説明しており，**本レバーで新設する OvR sigmoid ヘッドは
  この「合計が1にならない」問題をまさに抱える方式であることが，過去の自分自身の較正実験
  （Iter29 platt・Iter30 isotonic）の失敗機序（medical_recall の BH 補正後有意悪化）と地続きである**．
  ただし今回はこの非正規化スコアを「確率」としてではなく「rank_2 以降を並べ替えるためのランキング
  スコア」としてのみ使うため（softmax と比較して argmax を取るわけではない），較正の破綻が
  argmax の決定に影響しない設計である点が Iter29/30 との重要な違いである．
  （MetricGate の整理記事 https://metricgate.com/blogs/one-vs-rest-vs-one-vs-one-classifier も
  「OvR スコアは per-binary-problem で較正されているだけで，jointly には較正されていない．
  確率ベクトルが欲しければ softmax か Platt+再正規化が必要」と同旨を述べている）。
- **1位は既存の強いモデルに任せ，2位以降だけ別モデルで並べ替える，という設計そのものの先行例は
  検索範囲内では見つからなかった**。関連が近いのは検索・推薦分野の「retrieve-then-rerank」
  2 段階設計（例: BM25 で候補生成→cross-encoder で re-rank，futureagi.com の LTR 用語集
  https://futureagi.com/glossary/learning-to-rank ，MDPI の MultiLTR
  https://www.mdpi.com/2078-2489/16/4/308）だが，これらは通常**候補集合全体**を再ランキングし，
  「1位だけ固定して残りだけ動かす」という非対称な設計ではない。したがって本レバーの設計
  （rank_1 argmax 固定＋ rank_2+ のみ OvR で並べ替え）は，文献上の直接の先例を持たない独自設計
  であり，その安全性（argmax flip rate 構造的0%）は文献の裏付けではなく本リポジトリのコード
  ロジック自体（`select_dispatch_targets()` を全く変えず，並べ替えは候補リスト構築より前で完結
  させる）から導かれる点を計画書に明記すべきである。
- **少量データでの binary relevance 過学習**: Zhang & Zhou, "Binary Relevance for Multi-Label
  Learning: An Overview" (Frontiers of Computer Science 2017) は，binary relevance の各ラベルの
  二値問題が「クラス不均衡（正例が少数派）」を本質的に抱えることを指摘し，対策として
  クラス重み付け・閾値調整・ラベル依存性の活用（classifier chain）を挙げている
  （出典: http://palm.seu.edu.cn/zhangml/files/FCS'17.pdf ）。
  Wikipedia の Multi-label classification 項目も同旨（binary relevance はラベル間の依存関係を
  一切見ない設計であることが弱点，と明記，https://en.wikipedia.org/wiki/Multi-label_classification ）。
  **本リポジトリのデータでは legal が 77 件（他ドメインの約半分）で最小**であり，かつ
  compound 100 行のうち legal が絡む行が 30 件（全ドメイン中最多，2 番目は medical の 28 件）
  という調査結果（下記 Q2 の実データ確認）と合わせると，**legal の二値分類器の質が本レバーの
  compound recall 改善効果を左右しやすい構造**にあることに注意が必要．
  **対策**: (a) sklearn `LogisticRegression(class_weight="balanced")` を各二値問題に**独立に**
  適用する（OvR の各二値分類器は互いに独立なので，本体分類器で起きた Iter32 の
  `sample_weight × class_weight_` 結合バグは構造的に起こり得ない．各ラベルの `class_weight_`
  はそのラベルの正例/負例比のみに依存する）。(b) `CalibratedClassifierCV` の 5-fold CV を
  ヘッド訓練にもそのまま踏襲し，legal のような少数クラスでの単純な訓練セット丸暗記を避ける。
- **OneVsRestClassifier vs MultiOutputClassifier の実装選択**: scikit-learn 公式ドキュメント
  （https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html）
  によれば，`OneVsRestClassifier` はマルチラベルの場合そのまま binary relevance を実装し，
  `MultiOutputClassifier` は「マルチラベル拡張の別の方法」として同義的に使える。本リポジトリの
  データは各行が単一ドメインラベルのみを持つ（`classifier_train.jsonl` は 1 行 1 ドメイン）ため，
  多ラベル目的変数を作るには `MultiLabelBinarizer` は不要で，**既存の単一ラベル配列をそのまま
  `OneVsRestClassifier(LogisticRegression(...))` に渡せばよい**（各ラベルについて「そのドメイン
  かそれ以外か」の二値問題が自動的に構成される）。

**分かったこと（Q2: コードベース調査，最優先）**

1. **`scripts/train_domain_classifier.py` の構造**（L1-223 通読）:
   - `_load_training_rows()` で `classifier_train.jsonl` を読み，`_extract_sample_weights()` で
     `n_samples / (n_classes * n_domain_samples)` のドメイン別重み（`class_weight='balanced'` と
     数式的に同値，Iter39 で確定した設計）を計算する。
   - `build_training_features()` は各行の `query` を `ollama_client.embed(embedding_model, ...)`
     で埋め込む（**逐次実行，並列化なし**）。**embedding はどこにもキャッシュされない**
     （joblib で保存されるのは学習済み分類器のみで，埋め込みベクトル自体はディスクに残らない）。
     したがって新しい OvR ヘッドを訓練するには，1427 行を**再度 embed し直す必要がある**
     （embedding_model は `nomic-embed-text` で固定，`config.yaml` 参照）。
   - `train_classifier()` は `LogisticRegression(max_iter=1000, class_weight=None)` を
     `CalibratedClassifierCV(method="temperature", cv=5, ensemble=True)` でラップし，
     `education` クラスの `intercept_` に `+0.7`（`intercept_delta`，Iter45 採用）を較正後の
     各 fold の estimator に直接加算する（L200-204）。これは**現行の rank_1 分類器の訓練ロジック**
     であり，本レバーでは一切変更しない（新設する OvR ヘッドは全く別のクラス・別のファイルとして
     並存させる）。
   - **本レバー実装への示唆**: 新スクリプト（例: `scripts/train_dispatch_candidate_ranking_head.py`）
     は `_load_training_rows()`・`build_training_features()` をそのまま import して再利用し，
     `train_classifier()` の代わりに `OneVsRestClassifier(LogisticRegression(max_iter=1000,
     class_weight="balanced"))` を fit する関数を新設するのが最小差分（embed のコードパスは
     完全に共用でき，1427 行の embed 計算だけがコスト＝オフラインで数分）。

2. **`data/classifier_train.jsonl` のスキーマ**: 1427 行，フィールドは `{"id", "query", "domain"}`
   の3つのみ（`sample_weight` フィールドは現状のデータには存在せず，コード側にオプション対応が
   残っているだけ）。ドメイン別行数を実測: `business_economics/computer_science/education/general/
   history_culture/mathematics/medical/natural_science/social_science` が各150件，**`legal` のみ
   77件**（合計 150×9+77=1427 と一致）。

3. **`models/domain_classifier.joblib` の入出力形式**: `classifier.py:load_domain_classifier()`
   で読み込む `CalibratedClassifierCV` オブジェクト。`.classes_`（ドメイン名の配列）と
   `.predict_proba([embedding])[0]`（10 要素の確率配列，合計1）の2つのAPIのみに依存する
   （duck typing，L14-19 のコメント）。**この分類器はレバー実装で一切変更しない**（rank_1 の
   argmax を担保する唯一の情報源として維持する）。

4. **`classifier.py:estimate_confidence_classifier()`（Iter57 で education 加算が追加された実行時
   経路）**: `domain not in classifier.classes_` なら 0.0，そうでなければ
   `predict_proba([query_embedding])[0][domain_index]` を返し，`domain=="education"` のときのみ
   `EDUCATION_THRESHOLD(0.05)` を**正規化なしで**加算する（L59-78）。**呼び出し元は
   `http_server.py:367` の1箇所のみ**（`/probe` エンドポイント，各ノードが自分のドメイン分だけ
   呼ぶ）。`results.jsonl` の `probe_candidates` フィールドは，1行あたり全10ノード分の
   `{"node_id", "domain", "confidence", "confidence_logprobs_mean"}` のリストであり，**この
   `confidence` が `estimate_confidence_classifier()` の返り値そのもの**（実測で確認，
   `results/20260919_005727/results.jsonl` の1行目を確認）。つまり `probe_candidates` は
   「10ドメイン全ての現行分類器 confidence（rank_1〜rank_10 相当）」を既に完全に保持しており，
   **rank_1（argmax）の再現には `results.jsonl` の再生だけで十分**（embedding の再計算は不要）。
   ただし**query_embedding 自体は `results.jsonl` に保存されていない**ため，新設する OvR ヘッドの
   スコアを得るには 1600 問の embedding をあらためて計算し直す必要がある（`node.py:202`
   `ollama_client.embed(config["embedding_model"], query)` が実行時に一度だけ計算し捨てている）。
   この「embedding は非永続」というパターンは `scripts/evaluate_classifier_calibration.py`
   （Iter29 以降，較正手法比較のたびに使われてきたオフライン評価テンプレート）の docstring にも
   明記されている（「query_embedding is not persisted in results.jsonl, so it must be recomputed；
   ただし LLM 生成・probe・dispatch トラフィックは一切発生しない」）。**この既存スクリプトが
   本レバーのオフライン採点スクリプトの直接のテンプレートになる**。

5. **`metrics.py` の関連関数**:
   - `compute_compound_coverage_metrics()`（L127-）: compound 行（`len(expected_domains)>1`）のみを
     対象に，`dispatched_domains` と `expected_domains` の集合演算で
     `compound_domain_set_recall = covered_domain_count / expected_domain_total` を算出する
     （`r.get("dispatched_domains")` で古い形式は自動スキップ，後方互換）。
   - `_mcnemar_from_correctness()`（L228-）: 連続性補正付き McNemar（2値の対応マップ2つを受け取る
     汎用関数）。`compute_mcnemar_test()`（top1 用）・`compute_domain_recall_mcnemar_test()`
     （ドメイン別 recall 用）はいずれもこれを呼ぶだけの薄いラッパー。**Iter58 の主基準検定
     （ドメイン単位 n=200 のペア比較）は，この関数を `(row_id, expected_domain)` ペアの
     `dict[str, bool]`（baseline が被覆したか／new が被覆したか）に直接投入することで実現していた**
     （journal Iter58「実験」節）。本レバーの主基準検定も同一パターンを再利用すればよい。
   - `apply_benjamini_hochberg(p_values, q=0.05)`（L367-）: per-domain 20指標の多重比較補正に使用。

6. **直近の本走結果ファイル**: `ls -1dt results/*/ | head` の結果，最新は
   `results/20260919_005727/`（Iter58 本走，gap方式 T=0.29/max_k=4）。`results.jsonl` の各行は
   `probe_candidates` フィールドを持ち，全10ノードの `{node_id, domain, confidence,
   confidence_logprobs_mean}` を保持していることを実データで確認した（上記4参照）。**オフライン
   採点の rank_1 情報源としてこのファイルをそのまま使える**。ただし Iter58 の gap 方式（k∈{1,2,4}）
   の下で記録された `dispatched_domains` は「固定 k=2」の基準線（compound_domain_set_recall=0.345）
   とは異なる母集団なので，**本レバーの成功条件が参照する 0.345 の基準線には，固定 `dispatch_top_k=2`
   時代の `results/20260918_202613/results.jsonl`（Iter57，gap方式導入前）を使う必要がある**
   （`results/20260919_005727/` を使うと，比較対象が「gap方式 vs OvR方式」になってしまい，
   config.yml note が定義した比較（「固定k=2 vs 固定k=2+OvR並べ替え」）にならない）。
   `results/20260918_202613/results.jsonl` は Iter58 調査・実験フェーズで既に compound_domain_set_recall
   =0.345 と検証済み（`scripts/replay_dispatch_gap_policy.py` の再生値と実機値が完全一致）。

7. **`select_dispatch_targets()` の全呼び出し元**（d0004 §4 の教訓を踏まえ機械的に確認，
   `grep -rn "select_dispatch_targets(" --include="*.py" . | grep -v "/tests/"` を実行）:
   - `aggregator.py:28`（定義そのもの）
   - `node.py:214`（`run_ask_flow()`，実際の dispatch/回答生成に使う唯一の実行時経路）
   - `run_experiment.py:93`（`_run_one()`，metrics 記録用の**2箇所目の独立呼び出し**．
     Iter58 で「gap 引数を渡し忘れる」という第2の no-op バグがここで発生し `ea4f680` で修正済み。
     現在のコードは `gap_threshold`/`gap_max_k` を `node.py` と同一に渡している）。
   - `scripts/replay_dispatch_gap_policy.py`（docstring 内の言及のみ，実際には
     `select_dispatch_targets()` を呼ばず同等ロジックを Python で再実装している独立スクリプト）。
   - **呼び出し元は実質2箇所（`node.py` と `run_experiment.py`）で全数一致**。本レバーは
     第1イテレーションでは**この関数自体を全く呼ばず**（オフライン採点は
     `replay_dispatch_gap_policy.py` と同じ「独立再実装」パターンを踏襲する），実行時経路への
     配線は次々イテレーション（スキーマ変更・ユーザー確認後）に先送りされる設計（config.yml note
     に既に明記済み）なので，**今回は d0004 §4 型の「レバーを読むコードに到達しない」リスク自体が
     存在しない**（そもそも実行時コードを経由しないオフライン検証だから）。ただし，**将来の配線
     段階で `node.py` と `run_experiment.py` の**両方**を同時に変更しないと，Iter58 と全く同じ
     「記録される `dispatched_domains` だけが旧方式のまま」という第2種の no-op を再演するリスクが
     ある**点を，次々イテレーションの計画に申し送る必要がある。
   - `estimate_confidence_classifier()` の呼び出し元は `http_server.py:367` の1箇所のみ
     （grep で確認済み，テスト除く）。本レバーはこの関数も変更しない。

**分かったこと（Q3: 実験設計への示唆）**

1. **OvR ヘッドの具体的な訓練方法**: `sklearn.multiclass.OneVsRestClassifier(
   sklearn.linear_model.LogisticRegression(max_iter=1000, class_weight="balanced"))` を
   `classifier_train.jsonl` の 1427 行（embedding は `scripts/train_domain_classifier.py` の
   `build_training_features()` を再利用して再計算，nomic-embed-text，同一 embedding_model）で fit する。
   `class_weight="balanced"` は**各ラベルの二値問題ごとに独立**に計算されるため（`OneVsRestClassifier`
   は内部で `n_labels` 個の独立な estimator を fit する），Iter32 で判明した
   「`sample_weight` と `class_weight_` の乗算結合」バグは構造的に発生しない（各ラベルの重みが
   他ラベルの行数分布に一切依存しないため）。**embedding は既存の `domain_classifier.joblib` が
   使うものと全く同じ生成経路（同一 `embedding_model`）だが，モデル自体は再利用できない**
   （`domain_classifier.joblib` は `CalibratedClassifierCV(LogisticRegression)` の単一マルチクラス
   分類器であり，`OneVsRestClassifier` の内部構造とは別物のため，学習済みパラメータの流用は不可能．
   **再訓練対象は「新規ヘッドのみ」で，embedding 計算そのものは1427件について1回再実行が必要**）。
2. **オフライン採点の具体的な手順**（新規ファイル2つを想定）:
   - `scripts/train_dispatch_candidate_ranking_head.py`: 上記1のとおり訓練し，
     `models/dispatch_candidate_ranking_head.joblib` として保存する（**`domain_classifier.joblib`
     とは別ファイル**．rank_1 用モデルへの上書き・混同を防ぐため命名を明確に区別する）。
   - `scripts/evaluate_dispatch_candidate_ranking.py`: `scripts/evaluate_classifier_calibration.py`
     と同じ構造（`--dataset data/dataset.jsonl --ollama-host ... --output results/....jsonl`）を
     踏襲し，1600 問について (a) query を embed（1回のみ，OvR ヘッドのスコア計算用），
     (b) 新設 OvR ヘッドの `predict_proba` で10ドメイン分の sigmoid スコアを得る，
     (c) **rank_1 は `results/20260918_202613/results.jsonl` の該当行の `probe_candidates` から
     argmax を取得**（既存分類器を再度呼ばず，確定済みの記録値をそのまま使うことで「rank_1 が
     文字どおり不変である」ことを実装上も保証する），(d) rank_1 以外の9ドメインを OvR sigmoid
     スコア降順に並べ，最上位を rank_2_new とする，(e) `dispatched_domains_new = {rank_1, rank_2_new}`
     を出力に書き込む（固定 k=2，コスト中立）。
3. **統計検定の具体的な手続き**（`metrics.py` 既存関数のみ再利用，Iter58 と同一パターン）:
   - 主基準: `(row_id, expected_domain)` の 200 ペアについて，baseline
     （`results/20260918_202613/results.jsonl` の `dispatched_domains`）と new（上記4の出力）
     それぞれで被覆されたかを `dict[str, bool]` にし，`metrics._mcnemar_from_correctness()` へ
     直接投入する（Iter58 実験フェーズと全く同じ呼び出し方，コード追記不要）。
   - `compute_compound_coverage_metrics()` を new 側の合成 `results` リスト
     （baseline の `expected_domains` 等はそのまま，`dispatched_domains` だけ new に差し替え）に対して
     呼び，`compound_domain_set_recall` を直接得る。
   - 非退行: rank_1 は定義上不変なので `top1_accuracy` の再計算・McNemar は不要（Iter58 の申し送り
     「rank_1 不変が構造的帰結」と同じ論理）。**ただし実装ミスの検出のため，一度だけ
     `sum(new_row["dispatched_domains"][0] == baseline_row["dispatched_domains"][0]
     for id) == 1600`（rank_1 が全行で一致すること）を assert 的に確認するチェックを
     採点スクリプトに含めるべき**（もし不一致があれば「rank_1 を再計算してしまっている」実装バグ）。
4. **レバー発火の証拠フィールド（d0004 §4 対策，過去の no-op 反復への警戒）**: 本レバーは
   argmax（rank_1）を変えないため，「発火した証拠」は**rank_2 の選出ドメインが新旧で何%異なるか**
   （rank2_flip_rate）でしか観測できない。旧方式の rank_2 は「同じ確率分布内で2番目に高い
   softmax 確率のドメイン」，新方式の rank_2 は「OvR ヘッドの独立 sigmoid スコアで最大のドメイン」
   であり，学習アルゴリズムも訓練目的関数も異なるため，**rank2_flip_rate が 0% に近い場合は
   実装が旧ロジックへフォールバックしている（発火していない）ことを疑うべき**というしきい値を
   計画フェーズで明記しておくとよい（目安: Iter58 の「発火の証拠は dispatched_domains 長の分散」
   と同じ発想で，今回は「rank_2 ドメインの分布が変わったか」を確認する）。
   `results/20260918_202613/results.jsonl` の2位 confidence 分布（Iter58 調査節で実測済み，
   平均0.291・中央値0.200の gap）を踏まえると，rank2_flip_rate は 0%〜100% のどこに落ちても
   ありうるため，事前の期待値を置かず，**実測してそのまま報告する**方針が妥当。
5. **legal ドメインへの追加の留保**: compound 100 行のうち legal を含む行が 30 件と最多
   （medical 28，education 20 が続く）である一方，legal の訓練データは 77 件と全ドメイン最小。
   OvR ヘッドの legal 二値分類器の質が本レバーの改善幅を左右しやすいため，考察フェーズでは
   「compound recall の改善が legal 絡みの行に偏っていないか」をドメインペア単位で内訳確認する
   ことを推奨する。

**次の計画フェーズへの示唆**

1. **単一レバー・スキーマ変更なしで着手可能**: 新規ファイル2つ（訓練・採点）の追加のみで，
   既存の `aggregator.py`・`node.py`・`run_experiment.py`・`config.yaml` は一切変更しない。
   `select_dispatch_targets()` の呼び出し元（`node.py`・`run_experiment.py` の2箇所）も触らない
   ため，Iter58 で発生した「2箇所目の呼び出しへの引数追加漏れ」型の no-op は今回は構造的に
   起こり得ない（実行時経路自体を通らないため）。
2. **基準線データは `results/20260918_202613/results.jsonl`（固定 `dispatch_top_k=2`，Iter57）を
   使うこと**。最新の `results/20260919_005727/`（Iter58 gap方式）を誤って基準線に使うと，
   比較対象がずれる（gap方式 vs OvR方式になってしまう）ので要注意。
3. **実施コストは埋め込み計算のみ**（訓練1427件＋評価1600件，合計3027件の embed 呼び出し，
   `scripts/evaluate_classifier_calibration.py` の実績から数分〜十数分で完了見込み，実機の
   probe/dispatch/LLM生成トラフィックは一切発生しない）。
4. **rank_1 が全行で完全一致することの機械的な確認**を採点スクリプトの必須ステップとして計画に
   明記すること（発火の誤検出・過検出どちらも防ぐため）。
5. **legal ドメイン（訓練77件，compound内訳最多30件）の内訳確認**を考察フェーズの必須項目として
   計画時点で申し送ること。

**出典一覧**
- scikit-learn, "1.16. Probability calibration" §1.16.3.3 / §1.16.3.4:
  https://scikit-learn.org/stable/modules/calibration.html
- Stack Overflow, "Multiclass classification: probabilities and calibration":
  https://stackoverflow.com/questions/60110209/multiclass-classification-probabilities-and-calibration
- MetricGate, "One-vs-Rest vs One-vs-One Classifier":
  https://metricgate.com/blogs/one-vs-rest-vs-one-vs-one-classifier
- futureagi.com, "What Is Learning to Rank? Definition & Methods (2026)":
  https://futureagi.com/glossary/learning-to-rank
- MDPI, "MultiLTR: Text Ranking with a Multi-Stage Learning-to-Rank Approach" (2025):
  https://www.mdpi.com/2078-2489/16/4/308
- Zhang & Zhou, "Binary Relevance for Multi-Label Learning: An Overview", Frontiers of Computer
  Science (2017): http://palm.seu.edu.cn/zhangml/files/FCS'17.pdf
- Wikipedia, "Multi-label classification": https://en.wikipedia.org/wiki/Multi-label_classification
- scikit-learn, `OneVsRestClassifier` API reference:
  https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html

### 計画 (Iter59)

**仮説**

compound 設問で2つ目の正解ドメインが上位に来ないのは，各ノードの confidence が
「10クラス softmax の自分のクラスの確率」＝単一ラベル構成の副産物であり，2位以降の順位が
多ラベル的な関連度を表していないためである（Iter58 実測: 2つの正解が上位2位に両方入るのは
3/100 行のみ，2つ目の正解のランク中央値5〜6位）．frozen embedding の上に各ドメイン独立の
sigmoid（OvR / binary relevance）ヘッドを置き，**rank_2 以降の順位付けだけ**をそのスコアで
差し替えれば，dispatch コストを一切増やさない（固定 k=2，mean dispatch 2.0）まま
`compound_domain_set_recall` が基準線 0.345 から改善する．

**単一レバー**

`dispatch_candidate_ranking`: `softmax_second_highest`（現行＝既存分類器の softmax 確率の降順で
2位を選ぶ）→ `multilabel_binary_relevance_head`（既存分類器の argmax を rank_1 に固定したまま，
残り9ドメインを新設 OvR sigmoid ヘッドのスコア降順に並べ，その最上位を rank_2 とする）．

固定する構成（直近最良から変更しない）:
- embedding: `nomic-embed-text`（frozen，再訓練なし）
- rank_1 分類器: `models/domain_classifier.joblib`（`CalibratedClassifierCV(temperature)` ＋
  education `intercept_delta=+0.7`）を**一切変更しない**
- `aggregator.py` / `node.py` / `run_experiment.py` / `classifier.py` / `config.yaml`: 無変更
- dispatch コスト: 固定 k=2（本イテレーションでは gap 方式を使わない．基準線も固定 k=2 の
  `results/20260918_202613/results.jsonl`）

**変更するファイルと箇所（最小差分・新規2ファイルのみ）**

1. `scripts/train_dispatch_candidate_ranking_head.py`（新規）
   - `scripts/train_domain_classifier.py` の `_load_training_rows()` と
     `build_training_features()` を import して再利用（embed 経路を完全共用．
     `_extract_sample_weights()` は**使わない**＝各二値問題に独立な
     `class_weight="balanced"` を使うため．Iter32 の `sample_weight × class_weight_` 結合バグは
     OvR では構造的に起こらない）
   - `OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight="balanced"))` を
     1427 行で fit し `models/dispatch_candidate_ranking_head.joblib` へ保存
     （**`models/domain_classifier.joblib` は上書きしない**．別ファイル名で並存させる）
   - CLI は `train_domain_classifier.py` と同じ体裁（`--train-data`, `--ollama-host`,
     `--ollama-port`, `--embedding-model`, `--output`）
   - 訓練時に 5-fold CV の per-domain ROC-AUC / average precision を stderr に出力する
     （legal 77件の過学習診断用．訓練セット丸暗記の検出）
2. `scripts/evaluate_dispatch_candidate_ranking.py`（新規）
   - テンプレート: `scripts/evaluate_classifier_calibration.py`（Iter29 以降のオフライン採点の定型．
     LLM生成・probe・dispatch トラフィックは発生させず embedding だけ実機ノードへ問い合わせる）
   - 入力: `--baseline results/20260918_202613/results.jsonl`（1600行，全行 k=2 を実測確認済み），
     `--head models/dispatch_candidate_ranking_head.joblib`，`--ollama-host`，
     `--embedding-cache results/iter59_query_embeddings.npz`（あれば再利用，無ければ計算して保存．
     再採点のたびに 1600 回 embed し直さないためのキャッシュ）
   - 手順: (a) 各行の query を embed，(b) ヘッドの `predict_proba` で10ドメインの sigmoid スコア，
     (c) **rank_1 は baseline 行の `dispatched_domains[0]` をそのまま採用（既存分類器を再計算しない）**，
     (d) rank_1 を除く9ドメインをヘッドスコア降順に並べ最上位を rank_2_new，
     (e) `dispatched_domains = [rank_1, rank_2_new]` の新 results リストを出力
   - 出力: `results/iter59_ovr_ranking_predictions.jsonl`（baseline 行の `id` /
     `expected_domains` / `selected_domain` はそのまま引き継ぎ，`dispatched_domains` のみ差し替え，
     加えて `head_scores`（10ドメイン分）と `rank2_baseline` / `rank2_new` を診断用に保持）

**レバーが読まれるコード行と到達条件（d0004 §4 対策）**

本イテレーションは実行時経路（`node.py:214` / `run_experiment.py:93` の
`select_dispatch_targets()` 呼び出し2箇所）を**一切通らない**オフライン検証であるため，
Iter16/20/21/22/27/58 型の「レバーを読むコードに到達しない」no-op は構造的に起こり得ない．
代わりに，採点スクリプトが「正しく発火し，かつ rank_1 を壊していない」ことを機械的に保証する
以下3点を採点スクリプト内の必須アサーションとして実装する:

- A1（rank_1 不変）: `sum(new[i]["dispatched_domains"][0] == base[i]["dispatched_domains"][0]) == 1600`
  を assert する．1件でも不一致なら「rank_1 を再計算してしまっている」実装バグとして中断する．
- A2（コスト中立）: 全1600行で `len(dispatched_domains) == 2` かつ rank_1 ≠ rank_2（重複なし）を
  assert し，mean dispatch = 2.000000 を出力に記録する．
- A3（発火の証拠）: `rank2_flip_rate = mean(rank2_new != rank2_baseline)` を算出して報告する．
  **0.0% の場合はヘッドが旧ロジックへ退化している（例: ヘッドのスコアが softmax と同順）疑いが
  強く，実験を成立させず実装を見直す**．事前の期待値は置かず実測値をそのまま報告する
  （baseline の rank_2 分布は education 421 / business_economics 188 / … / legal 73 と偏っており，
  新旧で分布が変わるかも併記する）．

**実施方法（コマンド手順）**

```
# 1) OvR ヘッドの訓練（1427 件 embed，数分．LLM 生成なし）
uv run python -m scripts.train_dispatch_candidate_ranking_head \
    --train-data data/classifier_train.jsonl \
    --embedding-model nomic-embed-text \
    --ollama-host 192.168.15.100 \
    --output models/dispatch_candidate_ranking_head.joblib

# 2) 1600 問のオフライン採点（embed のみ，probe/dispatch/LLM なし）
uv run python -m scripts.evaluate_dispatch_candidate_ranking \
    --baseline results/20260918_202613/results.jsonl \
    --head models/dispatch_candidate_ranking_head.joblib \
    --embedding-model nomic-embed-text \
    --ollama-host 192.168.15.100 \
    --embedding-cache results/iter59_query_embeddings.npz \
    --output results/iter59_ovr_ranking_predictions.jsonl

# 3) 指標・検定（metrics.py の既存関数のみ使用．コード追加なし）
#    - metrics.compute_compound_coverage_metrics(new_rows) → compound_domain_set_recall
#    - (row_id, expected_domain) の 200 ペアの被覆 bool 2 本を作り
#      scipy.stats.binomtest(b, b+c, 0.5) で exact McNemar（Iter58 主基準と同一手続き）
#      併せて metrics._mcnemar_from_correctness() の連続性補正版も参考値として併記
```

**成功条件（事前登録，判定は全て αレベル固定・事後変更しない）**

S1（主基準）: `compound_domain_set_recall` が基準線 **0.345（69/200，実測で再検算済み）** から
  上昇し，ドメイン単位 n=200 の **exact McNemar（two-sided binomtest，α=0.05）で p < 0.05**．
S2（効果量の下限）: 点推定の上昇が **+0.04 以上（＝被覆ドメイン数 69 → 77 以上）**．
  微小だが有意，という判定を採用に倒さないための下限．
S3（コスト中立）: 全1600行で k=2，mean dispatch = 2.000（A2 の assert が通ること）．
  基準線と同一コストなので Iter58 のような「同コスト参照ポリシー」の追加走行は不要
  （k を伸ばす変種を併せて評価する場合に限り同コスト参照を必須とする，という config.yml の
  条件には該当しない）．
S4（発火の証拠）: `rank2_flip_rate > 0`．0 なら実験不成立として実装を見直す．

S1〜S4 全てを満たせば「採用（次々イテレーションで `config.yaml` へ配線，ただしスキーマ変更のため
ユーザー確認）」，S1 のみ不成立で S2 相当の上昇が見える場合は partial，S1・S2 とも不成立なら棄却．

**非退行条件（事前登録）**

N1（rank_1 完全不変）: A1 の assert（1600/1600 一致）が通ること．これが破れた実験結果は
  単一レバー原則違反として無効とする．
N2（top1_accuracy 不変）: N1 の帰結として定義上不変．念のため new 側 rows に対し
  `metrics.compute_top1_accuracy()` を再計算し，baseline と**完全一致（小数点以下すべて）**する
  ことを確認する（不一致なら実装バグ）．
N3（legal 非退行・過学習チェック）: legal は訓練データ最少（77件，他ドメインは各150件）で，
  compound 100 行のうち legal を含む行が 30 件と最多という不均衡構造にある．基準線の実測値は
  **legal 絡み行の被覆 20/60 ドメイン単位，うち legal 自身の被覆 8/30**．新方式でこの
  「legal 自身の被覆 8/30」が**下回らない**こと（≧8）を非退行条件とする．また，ヘッドの
  legal 二値分類器のスコアが定数的に退化していないこと（1600 行での legal スコアの標準偏差 > 0，
  かつ 5-fold CV の legal ROC-AUC を報告）を確認する．
N4（全体の恩恵の偏り確認）: 改善がある場合，その内訳をドメインペア単位（legal 絡み / medical
  絡み / その他）で分解して報告する．legal 絡みのみに改善が集中している場合は，
  「compound データの 30% が legal であることに依存した結果」として主張の強度を落とす．

**留保（考察フェーズへの申し送り）**

R1: OvR ヘッドのスコアは per-binary で較正されるだけで合計1にならない（scikit-learn 公式・
  Iter29/30 で確認済みの構造）．本設計では argmax 決定に使わずランキングにのみ使うため
  較正の破綻は決定に影響しないが，**得られたスコアを「確率」として対外記述しないこと**．
R2: 1600 問は Iter58 までと同一の in-sample データであり，ヘッドの訓練データ
  （`classifier_train.jsonl` 1427件）とは別集合だが，評価集合上でのハイパラ選択は行わない
  （`LogisticRegression(max_iter=1000, class_weight="balanced")` 固定，探索しない）ことで
  Iter58 の「T を in-sample 選定」問題を繰り返さない．
R3: 本イテレーションはオフライン完結であり，実機の dispatch 挙動・回答品質・レイテンシは
  測定しない．採用となった場合でも，実行時経路への配線（`node.py:214` と
  `run_experiment.py:93` の**両方**を同時に変更しないと Iter58 と同型の「記録される
  `dispatched_domains` だけ旧方式」という no-op を再演する）と `config.yaml` のスキーマ変更
  （`dispatch_ranking_model` 等）はユーザー確認を経て次々イテレーションで行う．
R4: 「1位は既存モデル固定，2位以降だけ別モデルで並べ替える」という非対称設計は，調査フェーズの
  検索範囲では文献上の直接の先例が無い（近いのは retrieve-then-rerank だが候補集合全体を
  再ランキングする）．安全性の根拠は文献ではなく本リポジトリのコードロジック（A1 の assert）に
  ある点を対外記述で明示する．
- Medium (John I. Douglas Marangon), "Multilabel and Multiclass Classification Made Simple with
  scikit-learn": https://medium.com/@johnidouglasmarangon/multilabel-and-multiclass-classification-made-simple-with-scikit-learn-6fa485b55b59

### 実装 (Iter59)

**変更したファイル（新規2件のみ，既存ファイルは無編集）**

1. `scripts/train_dispatch_candidate_ranking_head.py`（新規）: `scripts/train_domain_classifier.py`
   の `_load_training_rows()` / `build_training_features()` を import して再利用し，
   `data/classifier_train.jsonl`（1427件）を embed した上で
   `OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight="balanced"))` を fit する。
   `_extract_sample_weights()`（計画どおり不使用）。5-fold `StratifiedKFold` の out-of-fold
   `decision_function` から per-domain ROC-AUC / average precision を stderr に出力する診断機能を
   実装（`_print_per_domain_cv_diagnostics()`）。保存先 `models/dispatch_candidate_ranking_head.joblib`。
2. `scripts/evaluate_dispatch_candidate_ranking.py`（新規）: `--baseline` の各行の `query` フィールド
   （`results.jsonl` 自体が `query` を保持しているため，計画にあった `data/dataset.jsonl` への
   突合は不要と判明し省略——単一レバー原則を守りつつ計画より単純な実装で同じ入力を得られる差分）
   を embed し，OvR ヘッドの `decision_function()` にシグモイドを適用した非正規化スコア
   （`OneVsRestClassifier.predict_proba()` はマルチクラス単一ラベルの場合合計1に再正規化してしまう
   ため，調査フェーズが指摘した「合計が1にならない独立スコア」を保つには `decision_function` 経由が
   必須と判明——実装中に発見した計画との差分）で rank_2 を再計算する。rank_1 は `--baseline` 行の
   `dispatched_domains[0]` を**計算せずそのままコピー**（A1 を構造的に保証）。埋め込みキャッシュ
   `results/iter59_query_embeddings.npz`（id キー）を実装。`metrics.compute_compound_coverage_metrics()`
   を import して診断用に `compound_domain_set_recall` を stderr へ出力する（計画どおり，最終的な
   McNemar 検定・BH 補正は次の実験・分析フェーズに委ねる）。

**実行環境上の対応（実装作業中に判明，コードには影響なし）**

- このサンドボックスから `192.168.15.100`（wafl500）への直接到達性は無い（`ping`/直接 TCP ともに
  タイムアウト，`ip route` にも当該サブネットへの経路なし）。過去イテレーション（Iter29〜Iter53）
  と同一の確立済み手順（`ssh -fNT -L 11435:localhost:11434 wafl500`，`~/.ssh/config` の
  `ProxyJump wafl` 経由）でローカルポートフォワードを新規に張り直し，`--ollama-host 127.0.0.1
  --ollama-port 11435` で疎通を確認した上で実行した（`curl http://127.0.0.1:11435/api/tags` で
  200 を確認済み）。実機の probe/dispatch/LLM 生成トラフィックは一切発生させていない
  （embedding のみ）。トンネルは次フェーズでの再利用のため起動したまま維持している
  （PID 2086401，`ssh -fNT -L 11435:localhost:11434 wafl500`）。
- `build_training_features()`（`train_domain_classifier.py`，既存・無編集）が
  `fine_tuned_embed_model=None` でも無条件に `from sentence_transformers import SentenceTransformer`
  を実行する既存の実装上，`sentence-transformers` が仮想環境に入っていないと import エラーになる
  ことが判明したため，`uv sync --extra research`（`pyproject.toml` 既定義の optional dependency
  group）を実行して依存関係を追加した。コード変更ではなく，既存の `pyproject.toml` に定義済みの
  extra を有効化しただけである。

**実行結果**

1. 訓練（`uv run python -m scripts.train_dispatch_candidate_ranking_head ...`，1427件 embed）:
   5-fold CV per-domain ROC-AUC/average precision（stderr出力，抜粋）:
   `business_economics 0.8556/0.5191`，`computer_science 0.9130/0.6252`，
   `education 0.8663/0.4445`，`general 0.8981/0.6452`，`history_culture 0.9351/0.7328`，
   **`legal 0.9202/0.6219`（n=77，過学習・退化の兆候なし，むしろ全ドメイン中2番目に高いROC-AUC）**，
   `mathematics 0.9498/0.8053`，`medical 0.7645/0.3406`（全ドメイン中最低），
   `natural_science 0.8902/0.5653`，`social_science 0.8948/0.6161`。
   `models/dispatch_candidate_ranking_head.joblib` を新規保存（`models/domain_classifier.joblib`
   の mtime・md5（`b360ef8...`）は実行前後で不変であることを確認済み，上書きなし）。
2. 採点（`uv run python -m scripts.evaluate_dispatch_candidate_ranking ...`，1600行，
   embedding cache `results/iter59_query_embeddings.npz` 新規生成）:
   `results/iter59_ovr_ranking_predictions.jsonl`（1600行）を生成。

**A1〜A3 実測値（スクリプト内蔵アサーションに加え，独立コードで再検算し二重確認済み）**

- **A1（rank_1不変）**: 1600/1600 行で baseline の `dispatched_domains[0]` と完全一致（不一致0件）。
  スクリプト内の `_assert_rank1_unchanged()` は例外を投げず正常終了し，別途 Python で再検算した
  結果も mismatch=0 で一致。
- **A2（コスト中立）**: 全1600行で `len(dispatched_domains)==2`（重複長は`{2}`のみ），
  rank_1≠rank_2（重複0件），mean dispatch = **2.000000**。
- **A3（発火の証拠）**: `rank2_flip_rate = 0.356875`（1600行中571行で旧方式と新方式のrank_2が
  異なる）。0%ではないため警告は発火せず，実装が発火していることを確認した。

**診断用に得られた副次的な値（正式な統計検定は次フェーズの担当，事前登録された成功条件の判定は
行っていない）**: `compound_domain_set_recall = 0.35`（70/200，基準線0.345=69/200から+0.005，
`metrics.compute_compound_coverage_metrics()` をそのまま呼び出して算出，独自の再計算式は書いていない）。

**型/lint/テスト**

- `uv run ruff check scripts/train_dispatch_candidate_ranking_head.py
  scripts/evaluate_dispatch_candidate_ranking.py` → All checks passed。
- `uv run pytest -q`: 227 passed, 12 failed, 5 warnings（実装前に `git stash` で確認済みの
  **既存の13件の失敗**——`tests/test_build_dataset.py`（JMMLUアーカイブ内に
  `japanese_civics.csv` が無い環境依存のKeyError）と `tests/test_train_domain_classifier.py`
  （`CalibratedClassifierCV` オブジェクトに `classes_` 属性が無い，sklearn バージョン起因の
  既存不具合）——のうち12件がそのまま残存（本イテレーションの変更に起因せず，`uv sync --extra
  research` の副作用で `test_build_training_features_embeds_each_row_in_order` が1件追加で通る
  ようになった）。**新規2ファイルはどちらのテストファイルにも import されておらず，今回の変更が
  これらの失敗の原因でないことを `git stash` での再現テストで確認済み**。新規ファイルに対する
  ユニットテストは追加していない（`scripts/evaluate_classifier_calibration.py` や
  `scripts/replay_dispatch_gap_policy.py` など同種のオフライン採点用テンプレートにも既存テストは
  無く，本リポジトリの慣行と整合的な判断）。

**実験開始可否**: 実験フェーズが開始してよい状態である。`results/iter59_ovr_ranking_predictions.jsonl`
と診断用サマリ（stderr JSON: `{"n_rows": 1600, "mean_dispatch": 2.0, "rank2_flip_rate": 0.356875,
"compound_domain_set_recall": 0.35, "compound_rows_evaluated": 100}`）が揃っており，次フェーズは
`metrics._mcnemar_from_correctness()` を用いた主基準検定（S1）・効果量下限（S2）・legal非退行
（N3）等の事前登録済み判定に直接進める。ただし診断値の compound_domain_set_recall=0.35 は
S2 の効果量下限（+0.04，77/200以上）に遠く届いていない点は，実験・分析フェーズが検定前に
留意すべき事実として申し送る。

### 実験・分析(実行) (Iter59)

本フェーズは新規の実機トラフィック（probe/dispatch/LLM生成）を一切発生させていない。実装フェーズが
既に生成済みの `results/iter59_ovr_ranking_predictions.jsonl`（1600行）と基準線
`results/20260918_202613/results.jsonl`（1600行，固定 `dispatch_top_k=2`）に対し，`metrics.py` の
既存関数（`compute_compound_coverage_metrics`・`_mcnemar_from_correctness`・
`compute_top1_accuracy`）のみを import して事前登録済みの S1〜S4・N1〜N4 を機械的に判定した。
新規に自作した統計ロジックは，`(row_id, expected_domain)` の 200 ペアを作るペアリング処理と，
`scipy.stats.binomtest` による exact McNemar のクロスチェックのみ（Iter58 と同一パターン）。
新規スクリプト `scripts/compute_iter59_ranking_stats.py` を作成・実行し，結果を機械可読な
`results/iter59_stats.json` に保存した（実行コマンド:
`uv run python -m scripts.compute_iter59_ranking_stats --baseline
results/20260918_202613/results.jsonl --new results/iter59_ovr_ranking_predictions.jsonl
--output results/iter59_stats.json`）。**解釈・採否判断はこのフェーズでは行わない**（次の
analyst/reflector フェーズに委ねる）。

**前提の再確認（検定前）**: `compute_compound_coverage_metrics()` を基準線に対して独立に
再実行し，`compound_domain_set_recall = 0.345`（69/200），`compound_rows_evaluated = 100`
（→ 200 ペア）であることを確認した（事前登録値と完全一致，スクリプト内で assert 済み）。

**S1（主基準，ドメイン単位 n=200 のペア比較）**: `metrics._mcnemar_from_correctness()` に
baseline/new 双方の被覆 bool マップを渡した結果，改善（baseline非被覆→new被覆）= **10件**，
悪化（baseline被覆→new非被覆）= **9件**，discordant = 19，連続性補正chi2 = 0.0，
**p = 1.0**（連続性補正版）。`scipy.stats.binomtest(k=9, n=19, p=0.5, alternative="two-sided")`
による exact 版も **p = 1.0** で完全一致。**α=0.05 に対し p<0.05 を満たさず，S1 は FAIL**。

**S2（効果量下限）**: `new_compound_domain_set_recall = 0.35`（70/200，実装フェーズの診断値と
完全一致），基準線 0.345 からの差 **+0.005pt**。事前登録の下限 **+0.04pt（77/200以上）に届かず，
S2 は FAIL**。

**S3（コスト中立）**: 1600行全てで `len(dispatched_domains)==2`（分布 `{2: 1600}`），
rank_1≠rank_2（重複0件），`mean_dispatch = 2.000000`。**PASS**（独立再計算，実装フェーズの
assertion と一致）。

**S4（発火の証拠）**: `rank2_flip_rate = 0.356875`（1600行中571行）。独立再計算した値が
実装フェーズの速報値（0.356875）と完全一致。0%ではないため**PASS**。

**N1（rank_1完全不変）**: 1600/1600行で baseline の `dispatched_domains[0]` と完全一致
（不一致0件）。**PASS**。

**N2（top1_accuracy不変）**: `metrics.compute_top1_accuracy()` を new_rows に対し再計算した結果
`0.5975`（baseline と小数点以下完全一致）。**PASS**。ただし機械的に確認した重要な留保:
`evaluate_dispatch_candidate_ranking.py` の `build_new_rows()` は `selected_domain` フィールドを
baseline行から**そのままコピー**しており，新しい `dispatched_domains` から
`aggregator.select_best_dispatch_response()` を再実行して再導出したものではない。実機では
`select_best_dispatch_response()` は実際に dispatch されたノードの生成結果に依存するため，
この一致は「オフライン採点スクリプトがフィールドを保存しているだけ」であることを示すに留まり，
「rank_2 が変わっても実機の集約結果が不変である」ことまでは示さない（それを確認するには
オンライン再実行が必要で，本イテレーションの設計では意図的に行っていない）。

**N3（legal非退行）**: legal を含む compound ペアは30件（基準線の legal 自身の被覆 = **8/30**，
事前登録値と一致）。new 側の legal 自身の被覆 = **9/30**。8を下回っておらず**PASS**。

**N4（改善・悪化の内訳，ドメイン別分解，判定なし・報告のみ）**: compound 100行の内訳は
legal絡み30行（60ペア，legal∩medical重複12行を含む），medical絡み（legal除く）16行（32ペア），
その他54行（108ペア）。ペア単位の改善/悪化件数:

| カテゴリ | 対象ペア数 | 改善 | 悪化 | 不変 |
|---|---|---|---|---|
| legal絡み | 60 | 5 | 3 | 52 |
| medical絡み（legal除く） | 32 | 1 | 1 | 30 |
| その他 | 108 | 4 | 5 | 99 |
| 合計 | 200 | 10 | 9 | 181 |

（内訳合計は S1 の改善10件・悪化9件と一致，独立検算OK）。改善・悪化とも全カテゴリに分散しており，
legal絡みのみに改善が集中している様子は見られない。

**PASS/FAIL集計**: S1=FAIL，S2=FAIL，S3=PASS，S4=PASS，N1=PASS，N2=PASS，N3=PASS
（N4は判定なし・報告のみ）。**成功条件2/2がFAIL，非退行条件3/3・コスト条件1/1・発火条件1/1が
PASS**。内容面の解釈・採否判断（partial／棄却の別など）は次の analyst/reflector フェーズに委ねる。

**成果物**: `results/iter59_stats.json`（S1〜S4・N1〜N4 の実測値，機械可読），
`scripts/compute_iter59_ranking_stats.py`（新規，`metrics.py` 既存関数のみ再利用，
`uv run ruff check` all pass）。

### 分析(解釈) (Iter59)

**1. 独立検算の結果（`metrics.py` 既存関数のみ使用，不一致0件）**

`results/20260918_202613/results.jsonl`（基準線）と `results/iter59_ovr_ranking_predictions.jsonl`
（新方式）を読み直し，`results/iter59_stats.json` と journal 記載値を独立に再計算した．
**全項目が完全一致し，不一致は1件も無い**．

| 項目 | 独立再計算値 | stats.json |
|---|---|---|
| compound_domain_set_recall（基準線） | 0.345（69/200，rows=100） | 一致 |
| compound_domain_set_recall（新） | 0.350（70/200） | 一致 |
| compound_domain_jaccard_mean | 0.2400 → 0.2500 | （未記録，本フェーズで追加算出） |
| 主基準ペア比較（n=200） | 改善10 / 悪化9 / discordant 19 / chi2=0.0 / p=1.0 | 一致 |
| exact binomtest | p=1.0 | 一致 |
| rank2_flip_rate | 0.356875（571/1600） | 一致 |
| dispatched_domains 長分布 | `{2: 1600}`，rank_1≠rank_2 重複0 | 一致 |
| rank_1 不一致件数 | 0/1600 | 一致 |
| top1_accuracy | 0.5975 → 0.5975 | 一致 |

**2. S1・S2 FAIL は「検出できなかった」のではなく「効果が無い」——より高検出力の直接検証**

S1 の discordant はわずか19件で，これだけでは「弱い効果を検出できなかった（検出力不足）」
可能性を排除できない．実際，ペア差の近似 SE は sqrt(10+9)/200 = **2.18pt**，
95% CI は **+0.005 ± 0.043 = [-3.8pt, +4.8pt]** であり，事前登録の効果量下限 +4.0pt は
CI 上端の内側に入る（＝ +4pt の真の効果があっても本設計では検出しきれない）．
そこで，レバーの機序そのものを直接測る**より高検出力の検定**を追加で行った．

- **指標**: compound 100行それぞれについて，「rank_1 と異なる正解ドメイン」（計 **159 個**）が，
  rank_1 を除く 9 ドメインの中で**何位に来るか**を，旧方式（probe softmax confidence 降順）と
  新方式（OvR sigmoid スコア降順）で対応づけて比較した．
- **結果**: 平均順位 旧 **4.201** vs 新 **4.258**（差 +0.057，**新の方がわずかに悪い方向**），
  中央値はどちらも **4位**．1位的中は 旧28 / 新29，3位以内は 旧74 / 新75．
  順位が動いた対象は 89/159 で，**新が改善46 / 悪化43**．
  **Wilcoxon 符号順位検定 p=0.914**（n=159，非ゼロ差89）．
- **解釈**: 主基準の discordant 19 件よりはるかに多い 89 件の非ゼロ差をもってしても，
  順位の分布は新旧で区別がつかない．すなわち **S1/S2 の FAIL は検出力不足の帰結ではなく，
  「OvR ヘッドは2つ目の正解ドメインについて既存 softmax と同等以上の情報を持たない」という
  実体を反映している**と判断する．

**3. 「発火しているのに効かない」ことの機序 — ヘッドは別物だが同じだけしか知らない**

- **発火は本物**: rank2_flip_rate 35.7%（compound 行に限れば 31/100，単一ドメイン行 36.0%）で，
  rank_2 の**分布そのものが大きく変わっている**（education 421→142，legal 73→115，
  medical 150→213，business_economics 188→223）．実装上のフォールバックではない．
- **しかし精度は同等**: 単一ドメイン1500行で，OvR ヘッドの argmax 正解率は **0.610**，
  既存分類器（probe confidence の argmax）も **0.610** で一致する．
  ヘッドは既存分類器より弱くも強くもなく，**同等の判別力で誤り方だけが脱相関している**．
  その結果，rank_2 の入れ替えは「当たりを外れに変える」と「外れを当たりに変える」を
  ほぼ同数（改善10 / 悪化9，flip した compound 31行でも 旧9正解 / 新10正解）発生させ，
  正味の効果がゼロ付近に収束した．
- **根本原因（訓練データに多ラベル情報が存在しない）**: OvR ヘッドの訓練データは
  `data/classifier_train.jsonl`（1427件）で，**各行は単一ドメインラベルしか持たない**
  （調査フェーズ Q2-2 で確認済み）．binary relevance は「ラベル間の依存を見ない」ことが
  弱点だと文献（Zhang & Zhou 2017）が指摘するが，本件はそれ以前の問題で，
  **「ある質問が2つのドメインに同時に関連する」という事例を1件も学習していない**．
  したがって OvR ヘッドが学べるのは結局「この質問はどのドメインか」という単一ラベル信号の
  別分解にすぎず，2位以降の順位に新しい情報が入る経路が構造的に無い．
  訓練時 CV の ROC-AUC（legal 0.9202 等）が高いことは，この**単一ラベル判別**が
  うまく学習できている証拠であって，**「2つ目の正解を当てられる」ことの証拠ではない**
  （実測の2つ目の正解の中央順位は4位のまま）．Iter58 調査の「gap の compound 判別力は弱い
  （AUC 0.576）」という知見と同じく，**単一ラベル訓練由来の信号はどう加工しても
  多ラベル性を持たない**，というのが両イテレーションを貫く共通の構図である．
- **上限との距離**: 固定 k=2・rank_1 固定という条件下での recall の**オラクル上限は 0.705**
  （rank_1 が正解している compound 行は 41/100 で，残り1枠を必ず当てた場合 141/200）．
  実測 0.345→0.350 はこの上限の半分以下であり，**改善余地は大きいのに本レバーはその
  1/70 しか動かしていない**．なお rank_1 自体が compound 行で正解しているのは 41/100 に
  すぎず，**より大きなボトルネックは本レバーが意図的に凍結した rank_1 側にある**．

**4. N4 内訳にパターンは無い（偏りの証拠なし）**

legal絡み60ペア（改善5/悪化3），medical絡み32ペア（改善1/悪化1），その他108ペア（改善4/悪化5）．
**全カテゴリで改善と悪化がほぼ同数**であり，legal 絡みに改善が集中する懸念（計画 N4，
legal は訓練77件・compound 内訳最多30件）は実測されなかった．
N3（legal 自身の被覆 8/30→9/30）も +1件で，これは discordant 全体が19件しかないことを踏まえれば
ノイズ帯の変動である（legal 単独で有意性を主張できる母数ではない）．
**逆に言えば「legal のデータ不足が効果を潰した」という弁明も成り立たない**——
legal の CV ROC-AUC は 0.9202（全ドメイン中2位）で退化しておらず，
効果が出なかった原因を legal の少数性に帰することはできない．

**5. N2 の留保の評価 — 今回の判定を覆さない．さらに構造的に解消できる**

rc-experimenter の留保（`build_new_rows()` が `selected_domain` を baseline からコピーしており，
`aggregator.select_best_dispatch_response()` を再実行していない）は手続きとしては正しい指摘だが，
**本イテレーションの判定に対する影響は無い**．理由は3点．

1. **判定の論理上**: S1（主基準）と S2（効果量）は `dispatched_domains` のみから計算され，
   `selected_domain` に一切依存しない．N2 は非退行条件であり，仮に N2 が破れても
   FAIL 判定を PASS に転じさせることはできない．
2. **機序上，実機でも top1 は構造的に不変**: `aggregator.select_best_dispatch_response()`
   （aggregator.py:104-119）は `max(dispatch_responses, key=lambda r: r.confidence)` であり，
   ここでの confidence は **/probe 時に確定した既存分類器の confidence**（生成結果に依存しない）．
   rank_1 は定義上その argmax なので，**rank_2 に何を選んでも rank_1 が必ず勝つ**．
   実データでも基準線1600行すべてで `selected_domain == dispatched_domains[0]`（1600/1600），
   `selected_domain == dispatched_domains[1]` は0件であった．
   したがって「実機で再走すれば top1 が変わりうる」という懸念は，
   現行の `aggregation_method=max_confidence`（Iter47/48 で採用・レバークローズ済み）の下では
   成立しない．留保は「max_confidence 以外の集約（majority_vote / llm_judge）へ戻した場合に限り
   有効」と限定できる．
3. **例外条件も実データで空**: rank_1 の dispatch が失敗した場合のみ rank_2 が採用されうるが，
   基準線の `dispatch_failed` は **0/1600**，`used_fallback` も 0/1600 である．
   （ただし Iter58 では dispatch_failed が1件発生しており，将来0である保証は無い．
   実機配線時にはこの経路が唯一の top1 変動源になる．）

**結論: N2 の留保は軽微**．ただし，`answer_quality` / `end_to_end` / レイテンシは
本イテレーションでは一切測定していない（オフライン完結，計画 R3 のとおり）ため，
「実機で何も変わらないことを確認した」とは書けない．正確には
**「rank_2 の入れ替えは，現行の max_confidence 集約の下では実機でも採用回答を変えない」**
という機序上の帰結が示せる，という表現にとどめるべきである．

**6. 仮説との整合 — 仮説は明確に反証された**

計画の仮説は「2位以降の順位が多ラベル的関連度を表していないことが原因であり，
OvR ヘッドで rank_2 を差し替えればコスト中立のまま compound_domain_set_recall が改善する」だった．

- **整合した部分**: 「現行の2位以降の順位が多ラベル関連度を表していない」という**診断**は
  正しい（2つ目の正解の中央順位は4位のまま，オラクル上限 0.705 に対し実測 0.345）．
  また，設計の安全性（rank_1 完全不変・コスト完全中立）は事前の主張どおり構造的に達成された
  （N1 1600/1600，mean dispatch 2.000000）．
- **反証された部分**: 「OvR ヘッドならその多ラベル関連度を表せる」という**処方**は誤りだった．
  同一の単一ラベル訓練データから作った別分解は，順位付けを変えはするが情報を増やさない
  （Wilcoxon p=0.914）．

**7. 判定の確信度と，追加反復の要否**

- **確信度: 高**．理由は，(i) 決定論的なオフライン比較でランダム実行間ノイズが存在せず，
  独立検算で不一致0件，(ii) 主基準（discordant 19）だけでなく，より高検出力な順位レベルの
  検定（非ゼロ差89，Wilcoxon p=0.914）でも効果ゼロが支持され，(iii) 「ヘッドが壊れている／
  弱い」という代替説明が，単一ドメイン argmax 正解率の一致（0.610 vs 0.610）と
  legal ROC-AUC 0.9202 によって排除されるため．
- **同一設計での追加反復は不要**（決定論的なので同じ値が再現するだけ．Iter58 の学び2と同じ）．
- **仮に追加検証するなら**，確認すべきは以下のいずれかであり，いずれも本レバーの
  「同じ訓練データの別分解」という枠組みを**出る**必要がある:
  (a) **多ラベル訓練データの新規作成**（compound 相当の2ドメイン付き訓練事例）．
      これが無い限り，どんなヘッド構造でも2位の情報は増えない（本イテレーションの中心的学び）．
  (b) **rank_1 側の改善**（compound 行での rank_1 正解率 41/100 がより大きなボトルネック）．
  (c) k を増やす方向（Iter58 で検証済み．コスト増に見合う固有寄与は示せていない）．

**次フェーズ（rc-reflector）への申し送り**

- **棄却（rejected）が妥当と考えられる理由**: 事前登録の成功条件 S1・S2 がともに FAIL であり，
  計画文が定めた判定規則（「S1・S2 とも不成立なら棄却」）に機械的に該当する．
  加えて本フェーズの追加検証により，FAIL が検出力不足ではなく実効果ゼロであることが
  裏付けられた（Wilcoxon p=0.914）．partial に倒す材料（方向性のある効果量，特定ドメインでの
  一貫した改善）は見当たらない（+0.005pt，全カテゴリで改善≒悪化）．
- **ただし実験そのものは成立している**: S3・S4・N1〜N3 は全 PASS で，d0004 §4 型の no-op
  （レバーを読むコードに到達しない）ではない．**「実験不成立」ではなく「有効な陰性結果」**
  として記録すべきである．
- **`config.yaml` への配線（スキーマ変更・ユーザー確認）は行わないこと**．
  計画では「オフラインで所定の改善が確認できた場合にのみ次々イテレーションで配線」と
  条件付けており，その条件は満たされていない．
  `models/dispatch_candidate_ranking_head.joblib` は本番経路から参照されていない
  （`models/domain_classifier.joblib` は md5 不変・上書きなしを実装フェーズで確認済み）ため，
  ロールバック作業は不要である．
- **次レバー選定への最大の示唆**: 「単一ラベル訓練データ（1行1ドメイン）の加工では
  多ラベル性は生まれない」という制約が，Iter58（gap 信号，AUC 0.576）と Iter59（OvR ヘッド，
  Wilcoxon p=0.914）の2回連続で確認された．compound_domain_set_recall を本質的に動かすには，
  **多ラベル教師信号そのものを作る**（例: `build_dataset.py` の手作り複合設問と同じ手法で
  2ドメインラベル付き訓練事例を追加し，ヘッドを真の multi-label 問題として訓練する）か，
  **rank_1 側（compound 行で 41/100）を改善する**かのどちらかが必要である．

### 考察 (Iter59)

**判定: 棄却（rejected）．レバー `dispatch_candidate_ranking` はクローズ（収束）**

事前登録した成功条件 S1（主基準・exact McNemar p<0.05）・S2（効果量 +0.04pt 以上）がともに FAIL であり
（p=1.0，実測 +0.005pt），計画文が定めた判定規則「S1・S2 とも不成立なら棄却」に機械的に該当する．
journal の実験・分析節の数値を自分でも突き合わせて確認した（S1: 改善10/悪化9/discordant 19，
exact binomtest p=1.0，S2: 0.345→0.350）．`dispatch_candidate_ranking` は values が
`multilabel_binary_relevance_head` の単一値のみのため，本イテレーションでクローズとする．

**「実験不成立（invalid）」ではなく「有効な陰性結果」である根拠**

config.yml success_criteria (6) は「主要指標が基準線と完全一致し McNemar の不一致ペアが 0 件なら
invalid と判定せよ」と定めるが，本件は discordant 19 件・rank2_flip_rate 35.7%（571/1600）で
レバーは確実に発火している．コスト条件 S3（mean dispatch 2.000000）・発火条件 S4・
非退行条件 N1（rank_1 1600/1600 一致）・N2（top1 0.5975 完全一致）・N3（legal 8/30→9/30）は全 PASS．
d0004 §4 型の no-op ではない．

**棄却の機序（本イテレーションの中心的な学び）**

OvR ヘッドは既存分類器と「同等の判別力で誤り方だけが脱相関している」——単一ドメイン1500行での
argmax 正解率が 0.610 と 0.610 で完全一致し，2つ目の正解ドメインの順位比較（n=159，非ゼロ差89）でも
Wilcoxon p=0.914，平均順位 4.201→4.258 とむしろ微悪化方向だった．discordant 19 件だけでは
検出力不足を疑う余地が残るが，はるかに高検出力のこの順位レベルの検定でも効果ゼロが支持されるため，
**FAIL は検出力不足ではなく実効果ゼロ**と確定できる．
根本原因は `data/classifier_train.jsonl` が 1 行 1 ドメインで複合事例を 1 件も含まないことにあり，
「ある質問が 2 つのドメインに同時に関連する」という事例を学習していない以上，OvR という別分解を
与えても 2 位以降に新しい情報が入る経路が構造的に無い．訓練時 CV の ROC-AUC が高いこと
（legal 0.9202 等）は単一ラベル判別の学習成功を示すだけで，2つ目の正解を当てられる証拠ではない．

**Iter58 との共通構図 — 2回連続で同じ壁に当たった**

- Iter58: 既存 confidence の gap（1位-2位差）で compound を判別しようとした → AUC 0.576．
- Iter59: 既存 embedding ＋ 単一ラベル訓練データの OvR 再分解 → Wilcoxon p=0.914．

**「単一ラベル訓練データ（1行1ドメイン）の加工・後処理では多ラベル性は生まれない」**ことが
異なる 2 つの機序で独立に確認された．同じ枠組み（既存の単一ラベル教師信号を使い回す）での
3 度目の再挑戦は同じ壁に当たる公算が高く，行わない．

**ボトルネックの所在（次の設計の前提）**

固定 k=2・rank_1 固定という条件下での compound_domain_set_recall のオラクル上限は 0.705
（rank_1 が正解している compound 行が 41/100，残り1枠を必ず当てた場合 141/200）．
実測 0.345→0.350 はその半分以下であり，2 位枠の改善余地は大きい．一方で rank_1 自体が
compound 行で正解しているのは 41/100 に過ぎず，**本レバーが意図的に凍結した rank_1 側にも
同規模のボトルネックがある**．

**留保**

- N2（top1_accuracy 一致）は，採点スクリプトが `selected_domain` を基準線からコピーしているために
  生じた一致であり，実機での再導出ではない．ただし現行の `aggregation_method=max_confidence`
  （Iter47/48 でクローズ）の下では `select_best_dispatch_response()` が probe 時点の confidence の
  argmax を取るため rank_1 が構造的に必ず勝ち（基準線 1600/1600 で
  `selected_domain == dispatched_domains[0]`，`dispatch_failed` 0/1600），判定を覆さない．
  この留保は majority_vote / llm_judge へ戻した場合にのみ有効になる．
- 本イテレーションはオフライン完結であり，`answer_quality` / `end_to_end` / レイテンシは未測定．
  「実機で何も変わらないことを確認した」とは書けない．
- `config.yaml` への配線は行わない（計画が条件付けた改善が得られていない）．
  `models/dispatch_candidate_ranking_head.joblib` は本番経路から参照されず，
  `models/domain_classifier.joblib` は md5 不変（上書きなし）なのでロールバック作業は不要．

**次レバー: `multilabel_training_signal = synthetic_two_domain_training_examples`（新設，config.yml 末尾へ追記）**

skill の停止条件 1（journal/backlog の学びから次の有望なレバーを考案できるなら追記して継続）に従う．
既存 levers は実質試し切り（残る `conformal_prediction_true_class_qhat` は B88 で失敗見込み確定，
`post_hoc_langdetect_retry` は Iter55 で langdetect ja=100/100 のため改善余地なし）である．

Iter58・Iter59 が共通して指し示す欠落は「多ラベル教師信号そのものが存在しない」ことなので，
次は**教師信号を作る側**へ移る．評価用の複合設問 100 問（`build_dataset.py` の `_COMPOUND_QUESTIONS`）は
テストセットであり訓練に流用しない（リーク禁止）．代わりに，既存の単一ドメイン訓練行から
2 ドメインにまたがる訓練事例を新規生成し，**真の multi-label 問題として**ヘッドを訓練する．
Iter59 のインフラ（`scripts/train_dispatch_candidate_ranking_head.py`・
`scripts/evaluate_dispatch_candidate_ranking.py`・埋め込みキャッシュ）はそのまま再利用でき，
差分は訓練データの生成部のみ．基準線・成功条件・非退行条件も Iter59 と完全に揃えられるため，
**「ヘッド構造は同じで教師信号だけが違う」という Iter59 との直接対比が成立する**（これが
今回の陰性結果を，次の実験の対照群として活かす最も情報量の多い設計である）．
本レバーが不成立なら，compound 方向の改善余地は「rank_1 側の改善」か
「複合設問データセット自体の再設計（research_frontier）」に限られることになる．

次イテレーション名: 「2ドメイン訓練事例の新規生成による多ラベルヘッドの再訓練」．

