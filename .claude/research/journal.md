## Iteration 65: 2ドメイン合成訓練事例の用量反応（9→18件/ペア）

### 調査 (Iter65)

**問い**（config.yml:1134-1174＝backlog B98 が既に詳細な設計を事前登録している．
新規の先行研究探索ではなく，Iter64 調査と同様に一次情報＝コード・実データの確認を優先した）

- Q1: `--per-pair 18 --per-pair-legal 18` への変更は Iter64（9/9）と同様コード変更なしで成立するか．
  `_MINIMUM_ACCEPTABLE_ROW_COUNT` や `already_generated` の重複率上昇リスクが 18 件/ペアで
  9 件/ペアのときより顕著化しないか．
- Q2: 既存資産（Iter64 の訓練データ・ヘッド・予測 JSONL，Iter61 の対応物）は実在し，
  S5''-b（対 Iter61 12/100 対応あり McNemar）・S5''-c（135/405/810 の 3 点で単調増加）を
  計算するための数値が実データから再現できるか．
- Q3: `evaluate_dispatch_candidate_ranking.py --rank1-source head_argmax` は Iter65 の
  再訓練ヘッドに対してもそのまま使えるか．
- Q4: `compute_iter59_ranking_stats.py` で S5''-a/b/c 相当の集計が既存関数の組み合わせで
  計算できるか，新規実装が要るか．
- Q5: 実行コスト見積り（config.yml「合計40分程度」）は妥当か，wafl-ctrl5 の現状はどうか．

**分かったこと（全文読了・実行確認による一次情報）**

1. **Q1**: `generate_multidomain_training_examples.py` を再確認した．`_rows_for_pair()`（:162-166）
   は `per_pair`／`per_pair_legal` を整数のまま `_generate_all_rows()`（:169-196）の逐次ループへ
   渡すだけで，値の大小（9 か 18 か）に依存する分岐は存在しない．`_MINIMUM_ACCEPTABLE_ROW_COUNT=120`
   （:70）は 810 目標（90% フロアの A8' 相当なら 729）に対して十分小さく，フロアとして機能を保つ．
   `already_generated`（:178，全 45 ペア通しの単一集合）による重複再試行は Iter64（9/9・405 行目標）
   で**実測ゼロ**であることを確認した（`data/classifier_train_multidomain_iter64.jsonl` の
   `query` フィールドを完全一致で数えたところ 405 行中重複 0，ユニーク 405）．18/ペアはこの 2 倍の
   密度で同一ペアの語彙空間から生成するため重複再試行の頻度は構造的に上がりうるが，
   `_MAX_GENERATION_ATTEMPTS_PER_SLOT=3` 回のリトライの範囲内で吸収されるかは 810 行本走でしか
   確定できない．**コード変更を要する懸念は見当たらないが，生成ログの標準エラー出力
   （リトライ失敗の警告）を実装フェーズで監視する必要性は Iter64 より高い**．
2. **Q2**: 既存資産はすべて実在（`ls -la` 確認）：
   `data/classifier_train_multidomain_iter64.jsonl`（127,042B）・
   `models/dispatch_multilabel_head_iter64.joblib`（334,699B）・
   `results/iter64_multilabel_ranking_predictions.jsonl`（966,405B）・
   `results/iter64_stats.json`（S1/S2/S3/S4/N1/N2/N3/N4 を含む，S5 系は含まない）．
   Iter61 側も `data/classifier_train_multidomain_iter61.jsonl`（41,551B）・
   `models/dispatch_multilabel_head_iter61.joblib`（66,134B）・
   `results/iter61_multilabel_ranking_predictions.jsonl`（982,193B）が実在する．
   **S5''-b／S5''-c に必要な「被覆 2 個行数」を，`iter61_multilabel_ranking_predictions.jsonl` と
   `iter64_multilabel_ranking_predictions.jsonl` の `expected_domains`／`dispatched_domains`
   フィールドから直接（新規スクリプト不要のアドホック集計で）独立に再計算し，journal Iter64 の
   数値と完全一致することを確認した**：複合 100 行のうち rank_1 正解・被覆 2 個の内訳は
   Iter61＝(n=100, rank1正解=41, 被覆2個=**12**)，Iter64＝(n=100, rank1正解=72, 被覆2個=**24**)．
   これにより backlog B98 が要求する「対 Iter61（12/100）対応あり McNemar」の起点値 12 と，
   135/405/810 の 3 点で単調性を見る際の 2 点目（135 行時点＝Iter61＝12），3 点目
   （405 行時点＝Iter64＝24）が実データから再現できることを確認した．810 行（Iter65）本走後の
   値と合わせて 3 点（12→24→X）の単調増加を判定すればよい．
3. **Q3**: `evaluate_dispatch_candidate_ranking.py` の `_load_head()`（:173-186）は訓練データの
   由来（Iter64 の 405 行か Iter65 の 810 行か）を一切参照せず，`--head` に渡すパスのみで
   分岐する実装は Iter63/64 から不変であることを再確認した．**Iter65 で再訓練した
   `models/dispatch_multilabel_head_iter65.joblib` を渡すだけでコード変更なしに動作する**．
4. **Q4**: `compute_iter59_ranking_stats.py` に S5 系集計関数は存在しない点は Iter64 と不変．
   一方，上記 2. で実施したとおり「複合 100 行の `expected_domains`／`dispatched_domains` を
   突き合わせて rank_1 正解・被覆 2 個をカウントする」処理は 10 行程度のアドホックコードで足り，
   McNemar は `metrics._mcnemar_from_correctness()`（対 Iter61 用に `{id: bool}` 辞書＝
   「被覆 2 個か否か」を Iter61／Iter65 それぞれで作成し渡す）をそのまま呼べる．
   **S5''-a/b/c のいずれも新規実装は不要**（S5''-a は 2. の割合計算，S5''-b は上記 McNemar 呼び出し，
   S5''-c は 135/405/810 の 3 値の単調性を目視比較するだけ）．
5. **Q5（実行環境）**: wafl-ctrl5 は SSH 接続良好（`docker ps` で `ollama-ctrl` コンテナが
   3 時間稼働中，`ollama list` で `nomic-embed-text` と judge 用 `schroneko/llama-3.1-swallow-8b-instruct-v0.1:q4_k_m`
   の両モデルとも取得済みで pull 不要）．GPU は 12,288MiB 中 5,736MiB 使用・使用率 0%（他プロセスが
   VRAM を確保しているが計算負荷はなく，新規ジョブの実行を妨げない）．ディスクは 435GB 空きで
   十分．ローカルポート 11499 は既に転送済みで `curl http://127.0.0.1:11499/api/tags` が応答した
   （新規に `ssh -L` を張ろうとしたところ `Address already in use` で既存フォワードと衝突したため，
   既存トンネルをそのまま使えばよく，張り直しは不要）．
   コスト見積りについて，config.yml の想定（生成 15 分＋再訓練 20 分＋採点 2 分＝40 分程度）は
   Iter64 の実測（405 行の生成に約 5〜7 分程度で完了したとみられる mtime 差＋再訓練約 14 分＋
   採点約 1 分，各成果物の mtime: 訓練データ 12:06→ヘッド 12:20＝約 14 分→予測 12:21＝約 1 分）
   から見て，生成ステップの所要時間（405→810 行の倍増）が主な不確実性要因である．
   Iter60→61 では 135 行の生成に約 21 分を要した実績（journal Iter64 調査 5）があり，Iter64 では
   405 行を数分〜十数分で終えた可能性もあるため両者の実測に幅があるが，**810 行では生成だけで
   15〜40 分程度，訓練・採点を合わせても config.yml の「40 分程度」という見積りは下振れリスクを
   含む**（実測ベースでは 50〜70 分程度を見ておくのが安全側であり，計画フェーズのタイムアウト
   設定はこの幅を踏まえて余裕を持たせるとよい）．

**結論**

backlog B98 が設計した Iter65 の単一レバー（`--per-pair 18 --per-pair-legal 18`）は，
Iter64 と同型のコード変更ゼロ構成でそのまま実行可能である．既存資産はすべて実在し，
S5''-b／S5''-c が要求する「被覆 2 個行数」の起点値（Iter61＝12，Iter64＝24）を
予測 JSONL から独立に再計算して journal Iter64 の記載と完全一致することを確認できた．
`--rank1-source head_argmax` はコード変更なしに Iter65 の再訓練ヘッドへ流用でき，
S5''-a/b/c はいずれも新規実装なしに既存関数（`metrics._mcnemar_from_correctness()`）と
軽量なアドホック集計で計算できる．唯一の留意点は，18 件/ペアでは `already_generated` の
重複再試行頻度が構造的に上がりうる点（Iter64 では実測ゼロだが 810 行本走でしか確定できない）と，
実行コスト見積り（config.yml の「40 分程度」）がやや楽観的であり得る点である．
wafl-ctrl5 は Ollama コンテナ稼働中・両モデル取得済み・GPU/ディスクとも空きがあり，
ローカルポート 11499 のトンネルも生存していることを確認した．

**次フェーズへの示唆**

- S5''-b の事前登録は「被覆 2 個行を対 Iter61（12/100）と比較する exact McNemar」として，
  上記のアドホック集計コード（`expected_domains`/`dispatched_domains` の完全一致判定，
  約 10 行）をそのまま計画フェーズの実装仕様に転記してよい．新規スクリプトは不要．
- S5''-c（135/405/810 の単調増加）は，135 行時点＝Iter61 の 12，405 行時点＝Iter64 の 24 が
  既に確定しているため，810 行時点（Iter65）の値が **24 を上回れば単調増加が成立する**という
  判定式を計画フェーズで具体的に事前登録すること（「12→24→X で X>24」を明文化する）．
- 実装フェーズでは，18 件/ペアでの生成失敗（リトライ枯渇によるスロット未充足）を検出するため，
  `generate_all_rows()` の標準エラー出力（警告ログ）を必ず確認する運用を申し送る．
  `_MINIMUM_ACCEPTABLE_ROW_COUNT=120` は 810 目標に対し十分低いフロアだが，A8'（config.yml が
  Iter64 で新設した「目標の 90% 未満なら中止」規則）を Iter65 でも踏襲するなら 729 行が下限となる．
- 実行コスト見積りは config.yml の 40 分よりやや長め（50〜70 分程度）を見込んでおくとよい．
  wafl-ctrl5 のセットアップ・トンネルは既に生きているため，計画・実装フェーズで追加の環境構築は
  不要である．

### 計画 (Iter65)

**仮説**

Iter64（3→9 件／ペア，135→405 行）で「複合 100 行のうち 2 ドメインとも被覆した行数」は
12（Iter61・135 行）→24（Iter64・405 行）へ倍増し，rank_1 正解行に占める rank_2 正解割合も
27.9%→33.3% に上がった．一方，1 反復増分（Iter63→Iter64）の対応あり McNemar は discordant=19 で
p=0.169 にとどまり，**評価集合の複合行が 100 行に固定されている以上，1 反復分の増分を単独で
有意にすることは構造的に困難**であると判定された（留保 R-H）．
そこで本イテレーションは**同一変数（2 ドメイン合成訓練事例の件数）の第 2 用量**として
9→18 件／ペア（405→810 行）へ倍増し，**135／405／810 行の 3 点の用量反応**として
「量で 2 ドメイン性が学べるか」を検証する．量が効いているなら被覆 2 個行数は 12→24→X（X>24）と
単調増加し，かつ検出力のある参照点（Iter61 の 12/100）に対して有意差が出るはずである．
逆に 810 行で頭打ちになる，または N5（単一ドメイン argmax 正解率）が 0.590 を割るなら，
「量では 2 ドメイン性を学べない／量は単一ドメイン判別とトレードオフである」と結論できる．

**単一レバー（今回変更する唯一の変数）**

`multilabel_synthetic_volume_dose_response`: `uniform_nine_per_pair`（Iter64 の実質値＝
`--per-pair 9 --per-pair-legal 9`，45 ペア一律 9 件・405 行）→ **`uniform_eighteen_per_pair`**
（`--per-pair 18 --per-pair-legal 18`，45 ペア一律 18 件・**810 行目標**）．
**両引数を同値にして「どのドメインを厚くするか」という配分の決定を存在させない**
（`_rows_for_pair()` は `"legal" in (domain1, domain2)` で分岐するだけなので，同値なら全 45 ペアが
同数になる）．配分を決めると Iter60 の R-A 型リーク（テスト集合由来の設計情報の混入）が再発するため，
**legal だけ多くする等の非一律な値は書かない**．スクリプトの改修は行わない（コード変更ゼロ）．

**確定した実装仕様（本フェーズの決定事項）**

1. **コード変更は一切行わない**．変更は CLI 実引数値（`--per-pair 9 --per-pair-legal 9` →
   `--per-pair 18 --per-pair-legal 18`）と，Iter61/64 の成果物を保護するための出力パス名のみ．
2. **ヘッドは Iter65 の訓練データで再訓練する**（Iter64 の手順を踏襲．`--multilabel-train-data
   data/classifier_train_multidomain_iter65.jsonl`）．較正は導入しない（未較正
   `OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight="balanced"))` のまま）．
3. **採点は `--rank1-source head_argmax` を主系として固定する**（Iter63 で実装済み・コード変更不要．
   調査 Q3 で `_load_head()` が訓練データの由来を参照しないことを再確認済み）．
   `build_new_rows()` は `head_argmax` モードで `selected_domain` を新 rank_1 で上書きするため，
   **N2' はこの上書き後の `selected_domain` に対して算出した値で判定する**．
4. **統計は `scripts/compute_iter59_ranking_stats.py` を無改造で使う**．S1・S2・S3・N3 はそのまま
   利用し，**N1 と N2 の `exact_match` は本構成では定義上 `pass:false` になるため記録のみで判定に
   用いない**（Iter63/64 と同じ）．
5. **S5''-a/b/c と N6'' は「既存関数を読み取り専用で薄く呼ぶアドホック集計」で算出する**
   （調査 Q4 で新規実装不要を確認済み）．具体的には，複合 100 行の `expected_domains` と
   `dispatched_domains` を突き合わせて「rank_1 正解」「2 ドメインとも被覆」を判定する約 10 行の
   集計を書き，S5''-b は `metrics._mcnemar_from_correctness()` に `{行 ID: 被覆 2 個か否か}` の
   bool 辞書（Iter61 側・Iter65 側）を渡して呼ぶ．N6'' は
   `compute_iter59_ranking_stats._domain_pair_coverage_maps()` を流用する．
   **公式の採点・統計スクリプトには手を入れない**．

**固定する構成（Iter64 から一切変えない．config.yml:1134-1174 の note の列挙をそのまま踏襲）**

- 生成側: 生成プロンプト `_build_prompt()`，フィルタ F1〜F4，`_GENERATION_TEMPERATURE=0.8`，
  生成モデル `schroneko/llama-3.1-swallow-8b-instruct-v0.1:q4_k_m`，ペア集合
  （`itertools.combinations` の 45 ペア固定），A7 リーク監査の閾値 0.9，
  `--train-data data/classifier_train.jsonl`．
- 訓練側: ヘッド種別（未較正 OvR ロジスティック回帰）・単一ドメイン訓練データ
  `data/classifier_train.jsonl`（1427 行）・埋め込みモデル `nomic-embed-text`．
  **Iter62 の較正済みヘッドは使わない**．
- 採点側: 基準線 `results/20260918_202613/results.jsonl`（`compound_domain_set_recall`=0.345，
  `top1_accuracy`=0.5975，複合 100 行の rank_1 正解 41/100・2 ドメイン被覆 12/100），
  埋め込みキャッシュ `results/iter59_query_embeddings.npz`（評価クエリ 1600 問は完全ヒットの想定．
  **新規 embed は合成訓練文 810 件のみ**），rank_2 の選択ロジック，`_head_scores()`，A2/A3/A6/A9，
  N5 の計算，`--rank1-source head_argmax`，統計スクリプト（無改造），`config.yaml`．
- **実行時経路（`node.py` のルータ）への配線は本イテレーションでも行わない**（B94/B95．6 回目）．

**出力ファイル命名（Iter61/64 の成果物を上書きしないこと）**

| 種別 | 既存（保護・読み取り専用） | Iter65（新規作成） |
|---|---|---|
| 合成訓練データ | `data/classifier_train_multidomain_iter64.jsonl`（405 行） | `data/classifier_train_multidomain_iter65.jsonl`（810 行目標） |
| ヘッド | `models/dispatch_multilabel_head_iter64.joblib` | `models/dispatch_multilabel_head_iter65.joblib` |
| 予測 | `results/iter64_multilabel_ranking_predictions.jsonl` | `results/iter65_multilabel_ranking_predictions.jsonl` |
| 統計 | `results/iter64_stats.json` | `results/iter65_stats.json` |

**実行計画（実行基盤は config.yml 冒頭の恒久ルールどおり wafl-ctrl5 に一本化．
既存の SSH ローカルフォワード `127.0.0.1:11499` が生存していることを調査 Q5 で確認済みのため，
トンネルの張り直し・再セットアップは不要．`--ollama-host 127.0.0.1 --ollama-port 11499` で統一する）**

```
# 0) 合成訓練データの生成（唯一のレバー変更点: --per-pair 9→18, --per-pair-legal 9→18．810 行目標）
uv run python -m scripts.generate_multidomain_training_examples \
    --train-data data/classifier_train.jsonl \
    --model schroneko/llama-3.1-swallow-8b-instruct-v0.1:q4_k_m \
    --ollama-host 127.0.0.1 --ollama-port 11499 \
    --per-pair 18 --per-pair-legal 18 \
    --output data/classifier_train_multidomain_iter65.jsonl

# 0') リーク監査（A7．生成物の選別は行わず，近似重複の検出のみ）
uv run python -m scripts.generate_multidomain_training_examples --audit-leak \
    --output data/classifier_train_multidomain_iter65.jsonl

# 1) 多ラベルヘッドの再訓練（1427 + 810 件 embed．A0 は実生成行数で検査される）
uv run python -m scripts.train_multilabel_dispatch_head \
    --train-data data/classifier_train.jsonl \
    --multilabel-train-data data/classifier_train_multidomain_iter65.jsonl \
    --embedding-model nomic-embed-text --ollama-host 127.0.0.1 --ollama-port 11499 \
    --output models/dispatch_multilabel_head_iter65.joblib

# 2) 1600 問のオフライン採点（主系＝head_argmax．埋め込みキャッシュ完全ヒットの想定）
#    --iter59-predictions は引数名に反して汎用．S4（対 Iter64 不一致）を測るため Iter64 の予測を渡す．
uv run python -m scripts.evaluate_dispatch_candidate_ranking \
    --baseline results/20260918_202613/results.jsonl \
    --head models/dispatch_multilabel_head_iter65.joblib \
    --rank1-source head_argmax \
    --embedding-model nomic-embed-text --ollama-host 127.0.0.1 --ollama-port 11499 \
    --embedding-cache results/iter59_query_embeddings.npz \
    --iter59-predictions results/iter64_multilabel_ranking_predictions.jsonl \
    --output results/iter65_multilabel_ranking_predictions.jsonl

# 3) 指標・検定（Iter59〜64 と同一スクリプト・同一手続き．無改造）
uv run python -m scripts.compute_iter59_ranking_stats \
    --baseline results/20260918_202613/results.jsonl \
    --new results/iter65_multilabel_ranking_predictions.jsonl \
    --output results/iter65_stats.json

# 4) S5''-a/b/c・N6''・診断値のアドホック集計（読み取り専用．公式パスは無改造）
```

**タイムアウトと実行上の運用（調査 Q5 の実測見積り 50〜70 分を踏まえ余裕を持たせる）**

- 生成ステップ: **タイムアウト 5400 秒（90 分）**・バックグラウンド実行．Iter64 実測は 405 行で
  約 7 分だが，18 件／ペアでは `already_generated`（全 45 ペア通しの単一集合）による重複棄却と
  `_MAX_GENERATION_ATTEMPTS_PER_SLOT=3` のリトライが増える可能性があるため，線形外挿（約 15 分）の
  数倍を許容する．
- 再訓練ステップ: **タイムアウト 3600 秒（60 分）**（Iter64 実測 405 行で約 14 分．
  埋め込み対象は 1427+810 行で約 1.2 倍）．
- 採点・統計・アドホック集計: 各 **タイムアウト 900 秒（15 分）**（Iter64 実測 各 1〜2 分）．
- **生成ログの標準エラー出力を必ず確認する**（`generate_all_rows()` は失敗スロットを警告して
  スキップするだけで例外を送出しない設計．18 件／ペアではリトライ枯渇の頻度が Iter64 より
  構造的に上がりうるため，警告件数と最終行数を journal に記録すること）．

**成功条件（事前登録．事後変更禁止）**

参照点は基準線 `results/20260918_202613/results.jsonl`（recall 0.345，複合 100 行の rank_1 正解
41/100・2 ドメイン被覆 12/100，`top1_accuracy` 0.5975），**Iter61**（135 行．rank_1 正解 41/100・
2 ドメイン被覆 **12/100**），**Iter64**（405 行．recall 0.545，rank_1 正解 72/100・2 ドメイン被覆
**24/100**・割合 24/72=0.3333，`top1_accuracy` 0.599375，N5 0.591333，N6' education 6/20・
medical 20/28）である．Iter61/Iter64 の被覆 2 個行数（12・24）は調査フェーズで予測 JSONL から
独立に再計算し journal 記載と完全一致することを確認済みである．

- **S1**: 全体 200 ペアの対基準線 exact McNemar で **p<0.05**．
- **S2**: `compound_domain_set_recall` **≧ 0.565**（Iter64 実測 0.545 を点推定で +2.0pt 以上上回る）．
- **S3**: `mean_dispatch` = **2.000000**（完全一致．`duplicate_rank1_rank2_count`=0）．
- **S4**: 対 Iter64 予測の不一致行 **> 0**（レバーが予測を実際に動かしたことの確認）．
- **S5''（本レバー固有の主基準．R-H＝1 反復増分の検出力不足を回避する設計）**:
  - **S5''-a（点推定）**: 複合 100 行の rank_1 正解行に占める rank_2 も正解した行の割合が
    **Iter64 の 24/72 = 0.3333 を上回る**（分母は Iter65 実測で可変）．
  - **S5''-b（有意性．検出力のある参照点へ置き換える）**: 複合 100 行の「2 ドメインとも被覆」
    について，**対 Iter61（12/100）対応あり exact McNemar で p<0.05**．
    （対 Iter64 の 1 反復増分は discordant が小さく構造的に有意になり得ないため主基準にしない．
    対 Iter64 の McNemar は**参考値として記録のみ**し，判定には用いない．）
  - **S5''-c（用量反応の単調性）**: 被覆 2 個行数が **135 行→405 行→810 行の 3 点で単調増加**する．
    起点は確定済みで **12（Iter61）→ 24（Iter64）→ X（Iter65）**であり，**X > 24 なら成立**とする．
- **A8'（生成規模の担保．Iter64 で新設した規則を踏襲）**: 生成行数が **729 行（目標 810 の 90%）
  未満**であった場合，「45 ペア一律 18 件」が実現できていないとみなし，**実験を中止して原因を
  調査する**（下方修正した行数のまま採点を進めない）．

**非退行条件（事前登録．事後変更禁止）**

- **N2'**: `new_top1_accuracy` **≧ 0.5875**（**`selected_domain` を新 rank_1 で上書きした後の値**で算出）．
- **N3**: legal 自己被覆 **≧ 8/30**．
- **N5**: 単一ドメイン argmax 正解率 **≧ 0.590**（**留保 R-G に従い据え置き．Iter64 実測 0.591333 で
  余裕は 2 行分しかないが，これを理由に緩めない**）．
- **N6''**: education 自己被覆 **≧ 6/20** かつ medical 自己被覆 **≧ 18/28**．
  **この水準を選んだ理由（config.yml が計画フェーズに委ねた判断）**: (1) Iter64 の計画が採った
  「config.yml の暫定案の閾値を変更せずに確定する」という前例に従う．(2) education は Iter63 で
  3/20 に落ちた後 Iter64 の増量で 6/20 に回復しており，**同一変数の第 2 用量である本レバーで
  さらに倍増して education 被覆が Iter64 水準を下回るなら，それは「量のトレードオフ」を示す
  重要な信号であるため検出できる水準に置くべき**である（3/20 へ戻すと検出力がゼロになる）．
  (3) medical は Iter64 実測 20/28 だが下限は Iter63 水準 18/28 に据え置く（20/28 を下限にすると
  2 行の揺れで落ちるうえ，medical は本レバーの作用機序の主対象ではない）．
  **留保**: education の下限 6/20 は Iter64 実測ちょうどであり余裕がない（1 行の揺れで FAIL しうる）．
  この非対称は承知のうえで採用する．ただし**非退行 FAIL のみの場合は Iter63/64 と同じく最大 partial
  とし，rejected にはしない**（下記の判定基準）．

**採否の判定基準（Iter64 の書きぶりを踏襲して事前に機械的に定める）**

- **adopted**: S1・S2・S3・S4 をすべて充足し，**かつ S5''-a・S5''-b・S5''-c の 3 つすべてを充足**し，
  **かつ非退行 N2'・N3・N5・N6'' の 4 件すべてを充足**した場合．
- **partial**: 上記に達しないが，以下のいずれかに該当する場合．
  1. S1・S3・S4 を充足し，**S5''-a と S5''-c は充足するが S5''-b が p≧0.05 にとどまる**（＝点推定と
     単調性は示せたが有意性が示せない）場合．
  2. S1〜S4 と S5''（a/b/c すべて）を充足するが，**非退行のうち N2'・N3・N6'' のいずれかが FAIL**
     した場合（**N5 の FAIL は partial に含めない．下記 rejected を参照**）．
  3. S2 を充足するが S5'' の充足が a のみにとどまる場合．
- **rejected**: 以下のいずれかに該当する場合．
  1. **S5''-c が不成立**（X ≦ 24＝被覆 2 個行が 405 行で頭打ち）**かつ S5''-a も不成立**
     （＝量で 2 ドメイン性は学べない）．
  2. **S2 が Iter64 実測 0.545 を下回る**（用量反応の逆行）．
  3. **N5 < 0.590**（＝量が単一ドメイン判別とのトレードオフになっている．R-G により
     この水準は緩めないと事前に決めているため，割った時点で rejected とする）．

**探索的な診断値（主基準にしないこと）**

- 低品質行（プロンプト文言の echo）の混入率．Iter60 0.74%（135 行）→ Iter64 1.23%（405 行）と
  率では横ばいだった．810 行で率が上振れするなら「量がノイズを相対的に増やす」証拠になる．
- 対 Iter64 の被覆 2 個行の McNemar（discordant の内訳を含む）．R-H の定量化を続けるための記録．
- rank_1 正解と rank_2 正解の関連（Fisher の正確確率検定．基準線 1e-4 → Iter61 0.0023 →
  Iter63 0.0211 → Iter64 0.2538 と負の関連が消失してきている系列の続き）．
- education 絡み複合 20 行における「自ドメイン得点／行内最大得点」の中央値（Iter63 実績 0.0899）．
- `already_generated` による重複棄却でリトライした回数・スロット未充足の警告件数（18 件／ペアで
  構造的に増える見込みであり，次の用量（36 件／ペア等）を検討する際の上限の手がかりになる）．

**単一レバー原則の確認（混入チェック）**

- 生成プロンプト・F1〜F4・temperature=0.8・生成モデル・ペア集合・A7 閾値 → **無変更**．
- ヘッド種別（未較正）・較正の有無・単一ドメイン訓練データ・埋め込みモデル・埋め込みキャッシュ →
  **無変更**．
- 採点スクリプト・`--rank1-source head_argmax`・rank_2 の選択ロジック・統計スクリプト・基準線・
  `config.yaml` → **無変更**（コード変更ゼロ）．
- **ヘッドの再訓練は本レバーに構造的に従属する**（訓練データを増やす以上，再訓練しなければ
  レバーを読むコードに到達しない）ため，独立した第 2 のレバーではない．
- 実行ホストは Iter64 と同じ wafl-ctrl5（`127.0.0.1:11499`）で統一．ホスト指定はモデル名で
  同一性が担保されるため単一レバー原則に抵触しない．
- 変更は `--per-pair`／`--per-pair-legal` の実引数値（9→18）と出力パス名のみ．

**既知の制約（申し送り）**

- temperature=0.8 の生成乱数性により，810 行は Iter64 の 405 行を包含しない（**再生成であって
  追記ではない**）．「量」の効果と「別の 810 文である」ことは分離できない（本レバーの構造的制約）．
  この点は S5''-c（3 点の単調性）を主基準の一つに含める根拠でもある（単発の差ではなく傾向で見る）．
- 低品質行（プロンプトの echo）の混入は Iter60 から続く既知の穴であり，単一レバー原則により
  フィルタは変更しない．
- 本構成は rank_1・rank_2 の双方をオフラインのヘッドが決めるため，**実行時経路との乖離が最大**
  である（R-F）．オフラインで adopted となっても実機での有効性は配線・本走なしには主張できない．
- S2／S5'' の値を効果量の推定値として対外記述に引用してはならない．

**想定コスト**: 生成 15〜40 分（律速．逐次 await で件数にほぼ線形だが重複再試行で上振れしうる）
＋ヘッド再訓練 約 15〜25 分＋オフライン採点・統計 約 2 分 ＝ **合計 50〜70 分**
（config.yml の「40 分程度」は下振れリスクを含むため，調査フェーズの実測ベース見積りを採用した）．
オフライン完結・`config.yaml` のスキーマ変更なし・実機 1600 問本走なしのため自律着手してよい．

### 実装 (Iter65)

**コード変更**: なし（計画どおり）．`scripts/`・`metrics.py`・`config.yaml` に対する `git status`
差分はゼロであることを確認した（`config.yaml` に既存の未追跡差分 `embed_node_host: wafl502→wafl-ctrl5`
が残っているが，これは本フェーズ開始前から存在した無関係な変更であり，CLAUDE.md の規約
「作業前から存在する未コミットの変更は触らない」に従い放置した．Iter65 の計画は `config.yaml` を
固定構成としているため，この差分は本イテレーションの対象外である）．

**実行コマンド一式の用意と動作確認**（計画節の 4 コマンドをそのまま，実引数・出力パスも計画どおりに
確定した．いずれも `--help` でパース可能なことを確認済み．実際の長時間実行は次フェーズの担当）:

1. 生成（`--per-pair 18 --per-pair-legal 18`，計画の唯一のレバー変更点）：
   ```
   uv run python -m scripts.generate_multidomain_training_examples \
       --train-data data/classifier_train.jsonl \
       --model schroneko/llama-3.1-swallow-8b-instruct-v0.1:q4_k_m \
       --ollama-host 127.0.0.1 --ollama-port 11499 \
       --per-pair 18 --per-pair-legal 18 \
       --output data/classifier_train_multidomain_iter65.jsonl
   ```
   （タイムアウト 5400 秒，バックグラウンド推奨．A8'＝729 行未満で中止．
   標準エラー出力の WARNING 件数を必ず記録すること．）
2. A7 リーク監査：
   ```
   uv run python -m scripts.generate_multidomain_training_examples --audit-leak \
       --output data/classifier_train_multidomain_iter65.jsonl
   ```
3. 多ラベルヘッド再訓練（タイムアウト 3600 秒）：
   ```
   uv run python -m scripts.train_multilabel_dispatch_head \
       --train-data data/classifier_train.jsonl \
       --multilabel-train-data data/classifier_train_multidomain_iter65.jsonl \
       --embedding-model nomic-embed-text --ollama-host 127.0.0.1 --ollama-port 11499 \
       --output models/dispatch_multilabel_head_iter65.joblib
   ```
4. 1600 問オフライン採点（タイムアウト 900 秒．`--iter59-predictions` には S4 用に Iter64 の予測を渡す）：
   ```
   uv run python -m scripts.evaluate_dispatch_candidate_ranking \
       --baseline results/20260918_202613/results.jsonl \
       --head models/dispatch_multilabel_head_iter65.joblib \
       --rank1-source head_argmax \
       --embedding-model nomic-embed-text --ollama-host 127.0.0.1 --ollama-port 11499 \
       --embedding-cache results/iter59_query_embeddings.npz \
       --iter59-predictions results/iter64_multilabel_ranking_predictions.jsonl \
       --output results/iter65_multilabel_ranking_predictions.jsonl
   ```
5. 指標・検定（タイムアウト 900 秒，無改造）：
   ```
   uv run python -m scripts.compute_iter59_ranking_stats \
       --baseline results/20260918_202613/results.jsonl \
       --new results/iter65_multilabel_ranking_predictions.jsonl \
       --output results/iter65_stats.json
   ```
6. S5''-a/b/c・N6'' のアドホック集計（タイムアウト 900 秒．新規スクリプトだが既存関数
   `metrics._mcnemar_from_correctness()`／`compute_iter59_ranking_stats._domain_pair_coverage_maps()`
   を読み取り専用で呼ぶだけの一時スクリプト．Iter64 の `/tmp/iter64_s5_n6.py` と同型で，S5''-b の
   比較対象を対 Iter63 から対 Iter61（計画節が指定する検出力のある起点 12/100）に置き換えた点のみが
   差分）。`/tmp/iter65_s5_n6.py` として用意した：
   ```
   uv run python /tmp/iter65_s5_n6.py
   ```

**動作確認（スモークテスト，実データの一部を使い計算経路のみ検証．判定値そのものは次フェーズの担当）**:
上記 4 コマンドはすべて `--help` でパース成功を確認した．`/tmp/iter65_s5_n6.py` は
`results/iter65_multilabel_ranking_predictions.jsonl` が未生成のため，そのままでは実行できない
（意図的な仕様．Iter65 の予測がまだ存在しない状態で誤った値を出さないため入力パスをハードコードした）。
そこで同じ集計ロジックを一時的に Iter64 の予測ファイル（`results/iter64_multilabel_ranking_predictions.jsonl`）
に差し替えて実行し，`n_covered_iter61=12`・`n_covered_iter64(スタンドイン)=24`・
`education_covered=6/20`・`medical_covered=20/28` が調査フェーズでの一次確認値および
journal Iter64 の記載と完全一致することを確認した．これにより
`_mcnemar_from_correctness()` の呼び出し（対 Iter61 の bool 辞書 100 件）と
`_domain_pair_coverage_maps()` の流用（N3 と同型のペア ID 生成）が Iter65 の実データに対しても
正しく動作することを検証済みである．

**注意点（次フェーズへの申し送り）**

- 生成ステップの標準エラー出力を必ず確認し，`already_generated` によるリトライ枯渇（WARNING）の
  発生回数を journal「実験」節に記録すること．18 件/ペアはこの頻度が構造的に上がりうる（調査 Q1）．
- A8'（729 行未満で中止）に達した場合，そこで停止し原因調査へ切り替えること．後工程（再訓練・採点）
  を進めない．
- `--iter59-predictions` に渡すファイルは Iter64 の予測（`results/iter64_multilabel_ranking_predictions.jsonl`）
  であり，Iter59 の予測ではない（引数名は歴史的経緯で固定，Iter63/64 から不変の仕様）．
- `/tmp/iter65_s5_n6.py` の実行は，先行する採点コマンド（4番）が
  `results/iter65_multilabel_ranking_predictions.jsonl` を出力した後に行うこと．

**実行してよい状態か**: はい．コード変更は不要で，4 本のパイプラインコマンドと 1 本のアドホック
集計スクリプトはすべて用意・動作確認済みである．次フェーズ（rc-experimenter）は上記コマンドを
計画節のタイムアウト設定で順に実行し，A8' の中止基準と標準エラー出力の監視を行った上で結果を
journal「実験」節に記録する．

### 実験 (Iter65)

実行基盤は計画どおり wafl-ctrl5（`127.0.0.1:11499` の生存中トンネル経由）．すべてのコマンドを
バックグラウンド実行＋ポーリング（120 秒間隔）で実行し，タイムアウト超過は発生しなかった．
以下は実測値のみを記録し（判定は次フェーズの担当），機械可読な数値は
`results/iter65_stats.json`・`/tmp/iter65_logs/`（gen.log/audit.log/train.log/score.log/stats.log/s5n6.log）
に残る．

**0) 合成訓練データ生成**
- コマンド: 計画節どおり（`--per-pair 18 --per-pair-legal 18`）．
- 所要時間: 実測 **898 秒（約 15 分 0 秒）**（開始 epoch 1789798849 → 出力ファイル mtime 1789799747）．
  タイムアウト 5400 秒に対し余裕あり．
- 生成行数: **810 行**（目標 810 の 100%．A8' フロア 729 行を超過）．
- 標準エラー出力の `already_generated` 由来リトライ枯渇 WARNING 件数: **0 件**（stdout/stderr 全文
  `/tmp/iter65_logs/gen.log` に "wrote 810 rows" の 1 行のみで WARNING 行なし．
  `grep -ic WARNING` の結果も 0）．

**A8' チェック**: 810 行 ≥ 729 行（目標 810 の 90%）．**続行可**．中止・原因調査には至らなかった．

**0') A7 リーク監査**
- `n_generated_rows`=810，`n_compound_questions`=100．
- **max_jaccard = 0.18947368421052632**（閾値 0.9 未満で `PASS` 判定がスクリプト側で自動出力）．
- median_max_jaccard = 0.06749164895232311．

**1) 多ラベルヘッド再訓練**
- A0 チェック: PASS（810 multi-label rows，45 distinct domain pairs）．
- 所要時間: 実測 **276 秒（約 4 分 36 秒）**（開始 epoch 1789799838 → 出力ファイル mtime 1789800114）．
  タイムアウト 3600 秒に対し余裕あり．
- 出力: `models/dispatch_multilabel_head_iter65.joblib`（334,699 バイト）．
- ドメイン別 CV 指標（cv_roc_auc / cv_average_precision，n_positive はいずれも 312，legal のみ 239）:
  business_economics 0.8795/0.6668，computer_science 0.9250/0.7626，education 0.8975/0.6826，
  general 0.8583/0.5657，history_culture 0.9618/0.8846，legal 0.9353/0.7967，
  mathematics 0.9288/0.7614，medical 0.8070/0.4499，natural_science 0.8406/0.5208，
  social_science 0.8850/0.6887．

**2) 1600 問オフライン採点（`--rank1-source head_argmax`）**
- 所要時間: 実測 **37 秒**（開始 1789800206 → 終了 1789800243）．タイムアウト 900 秒に対し余裕あり．
- `mean_dispatch`=2.0，`rank2_flip_rate`=0.6675，`compound_domain_set_recall`=0.565，
  `compound_rows_evaluated`=100．
- スクリプトが自動出力した WARNING: `N5 single-domain argmax accuracy 0.5633 < floor 0.59`．
- `n5_single_domain_argmax_accuracy`: n=1500，correct=845，accuracy=**0.5633333333333334**．
- `a5_iter59_disagreement`（対 Iter64 予測）: mismatches=889/1600，mismatch_rate=0.555625．
- `rank1_change_count`=532，`rank1_change_rate`=0.3325．

**3) 統計（`compute_iter59_ranking_stats.py`）**
- 所要時間: 1 秒未満（開始 1789800249 → 終了 1789800249）．
- `baseline_compound_domain_set_recall`=0.345，`new_compound_domain_set_recall`=0.565．
- **S1**: n_pairs=200，improved=62，regressed=18，discordant=80，
  chi2(continuity)=23.1125，p(continuity)=1.5279415279678688e-06，
  p(exact_binomtest)=8.142666624995287e-07，`pass`=true（スクリプト内部フラグ．判定は次フェーズ）．
- **S2**: baseline_recall=0.345，new_recall=0.565，delta_pt=0.21999999999999997，floor_pt=0.04．
- **S3**: n_rows=1600，duplicate_rank1_rank2_count=0，mean_dispatch=2.0．
- **S4**: flips=1068，rank2_flip_rate=0.6675（対基準線．implementation_phase_value=0.356875 は
  対 Iter64 の historical 値でスクリプト内部の参考表示）．
- **N1**: mismatch_count=532（`selected_domain` 不変性；本構成では定義上不成立の想定どおり）．
- **N2'**: `new_top1_accuracy`=**0.573125**（`selected_domain` は head_argmax 上書き後の値．
  `evaluate_dispatch_candidate_ranking.py:255` の `selected_domain = rank_1` により，出力
  `results/iter65_multilabel_ranking_predictions.jsonl` の `selected_domain` フィールドは既に
  上書き済みであることをコード読解で確認した上で採用した値）．baseline_top1_accuracy=0.5975．
- **N3**: n_legal_involving_pairs=30，baseline_legal_self_coverage=8，
  **new_legal_self_coverage=15**．
- N4（探索的）: legal_involving n=60（improved17/regressed4/unchanged39），
  medical_involving n=32（improved9/regressed3/unchanged20），
  other n=108（improved36/regressed11/unchanged61）．

**4) S5''-a/b/c・N6'' アドホック集計（`/tmp/iter65_s5_n6.py`，複合 100 行に対して実行）**
- 所要時間: 1 秒未満（開始 1789800269 → 終了 1789800270）．
- **S5''-a（点推定）**: n_rank1_correct=72，n_rank1_and_rank2_correct=**21**，
  ratio=**0.2916666666666667**（Iter64 実測 0.3333333333333333 比較値として記録）．
- **S5''-b（対 Iter61 McNemar）**: n_covered_iter61=12，n_covered_iter65=**21**，
  discordant_a_only=9，discordant_b_only=18，discordant_pairs=27，
  chi2=2.3703703703703702，**p_value=0.12365771040283358**．
- **S5''-c（135/405/810 の 3 点）**: n_covered_iter61_135rows=12，n_covered_iter64_405rows=24，
  **n_covered_iter65_810rows=21**．
- **N6''**: n_education_pairs=20，education_covered=**7**（floor 6）；
  n_medical_pairs=28，medical_covered=**18**（floor 18）．

**実行上の異常の有無**: なし．全 7 コマンドともタイムアウト超過なく完了し，A8' のフロア（729 行）は
810 行で超過した．`already_generated` 由来のリトライ枯渇 WARNING は 0 件だった．

**参照した成果物パス**:
`data/classifier_train_multidomain_iter65.jsonl`（810 行）・
`models/dispatch_multilabel_head_iter65.joblib`・
`results/iter65_multilabel_ranking_predictions.jsonl`・`results/iter65_stats.json`・
`/tmp/iter65_logs/{gen,audit,train,score,stats,s5n6}.log`（実行ログ全文）．

### 分析（解釈） (Iter65)

本節の数値は，`results/iter65_stats.json`・`/tmp/iter65_logs/` の実測値に加え，予測 JSONL
（`results/iter{61,63,64,65}_multilabel_ranking_predictions.jsonl`）と合成訓練データ
（`data/classifier_train_multidomain_iter{61,64,65}.jsonl`）からの独立再計算に基づく．
**採否の確定判定と次レバーの選定は次フェーズ（rc-reflector）の担当であり，本節は事前登録条件の
機械的照合と機序の解釈までに留める．**

**1. 事前登録条件の機械的照合**

「計画 (Iter65)」節の成功条件・非退行条件を，丸めずに実測値と突き合わせた結果は次のとおりである．

| 条件 | 事前登録の閾値 | 実測 | 判定 |
|---|---|---|---|
| S1 | 対基準線 exact McNemar p<0.05 | p=8.142666624995287e-07（improved 62／regressed 18／discordant 80） | **PASS** |
| S2 | `compound_domain_set_recall` ≧ 0.565 | **0.565**（113/200．baseline 0.345，delta_pt=+0.22） | **PASS**（閾値ちょうど） |
| S3 | `mean_dispatch`=2.000000 かつ duplicate=0 | 2.0／0 | **PASS** |
| S4 | 対 Iter64 予測の不一致行 > 0 | `a5_iter59_disagreement` mismatches=889/1600 | **PASS** |
| S5''-a | 21/72 の比が Iter64 の 24/72=0.33333 を**上回る** | 21/72=**0.291667** | **FAIL** |
| S5''-b | 対 Iter61（12/100）exact McNemar p<0.05 | p=**0.123658**（discordant 27＝Iter61のみ 9／Iter65のみ 18） | **FAIL** |
| S5''-c | 12→24→X で **X>24** | X=**21** | **FAIL** |
| A8' | 生成行数 ≧ 729 | 810（目標比 100%） | PASS（中止基準に非該当） |
| A7 | max_jaccard < 0.9 | 0.189474 | PASS |
| N2' | `new_top1_accuracy` ≧ 0.5875 | **0.573125**（917/1600） | **FAIL** |
| N3 | legal 自己被覆 ≧ 8/30 | 15/30 | **PASS** |
| N5 | 単一ドメイン argmax 正解率 ≧ 0.590 | **0.563333**（845/1500） | **FAIL** |
| N6'' | education ≧ 6/20 かつ medical ≧ 18/28 | 7/20・18/28 | **PASS**（medical は下限ちょうど） |

判定基準の各分岐に対する前提の成立状況（分岐の当てはめのみ．**確定判定は reflector**）:

- adopted 分岐（S1〜S4 かつ S5''-a/b/c 全充足 かつ非退行 4 件全充足）: S5'' が 3 つとも FAIL，
  非退行も N2'・N5 が FAIL のため前提が成立しない．
- partial 分岐 1（S5''-a と S5''-c は充足し S5''-b のみ未達）: S5''-a・S5''-c が FAIL のため成立しない．
- partial 分岐 2（S5'' 全充足で非退行が FAIL）: S5'' が全 FAIL のため成立しない．
  なお同分岐は**N5 の FAIL を partial に含めないと明記**している．
- partial 分岐 3（S2 充足で S5'' の充足が a のみ）: S5''-a が FAIL のため成立しない．
- rejected 分岐 1（S5''-c 不成立 **かつ** S5''-a 不成立）: X=21≦24，21/72<24/72 で**両方成立**．
- rejected 分岐 2（S2 が Iter64 の 0.545 を下回る）: 0.565>0.545 のため**不成立**．
- rejected 分岐 3（N5 < 0.590）: 0.563333<0.590 で**成立**．

**2. S5''-c の反転（24→21）はノイズか，構造的な頭打ち／悪化か**

結論を先に書くと，**「24→21 という減少そのものはノイズ範囲である一方，『810 行でも 405 行を
超えない』という頭打ちは 3 点の系列としてノイズでは説明しにくい」**と判定する．根拠は 4 点である．

**2-1. 24→21 の減少は有意でない（対応あり検定）**

被覆 2 個行の対応あり exact McNemar を予測 JSONL から独立に計算した:

| 比較 | Iter 前のみ被覆／Iter 後のみ被覆 | p |
|---|---|---|
| 基準線 3/100 → Iter65 21/100 | 2／20 | **0.000290** |
| Iter61 12/100 → Iter64 24/100 | 4／16 | **0.013906** |
| **Iter64 24/100 → Iter65 21/100** | **7／4** | **0.546494（有意でない）** |
| Iter61 12/100 → Iter65 21/100 | 9／18 | 0.123658（S5''-b の実測値と一致） |

**-3 行の減少は discordant 11 行（7 対 4）に基づくもので，方向はノイズと区別できない．**
Wilson 95% CI も Iter64 [0.1669, 0.3323]・Iter65 [0.1417, 0.2998] と大きく重なる．
したがって「増量が被覆 2 個行を**能動的に壊した**」と読むのは実測を超えている．

**2-2. しかし「頭打ち」の側は，同じデータでもっと強く支持される**

用量反応を 4 点の系列で見ると，増分は明瞭に逓減して 0 を跨いでいる:

| 合成行数 | 被覆 2 個行 | 前点からの増分 | 対応あり p（前点比） |
|---|---|---|---|
| 0（基準線） | 3/100 | — | — |
| 135（Iter61） | 12/100 | +9 | — |
| 405（Iter64） | 24/100 | +12 | 0.0139（対 Iter61） |
| **810（Iter65）** | **21/100** | **−3** | 0.5465（対 Iter64） |

**行数を 2 倍にして増分が 0 を下回った**という事実は，「量を増やすほど 2 ドメイン性が向上する」
という用量反応仮説の予測（単調増加）と直接矛盾する．S5''-b が対 Iter61（12/100，検出力のある
参照点）でも p=0.1237 と有意にならなかった点も同方向である．**Iter64 の partial は
「discordant=19 で検出力が足りない」という R-H 型の未達だったが，Iter65 は
「検出力のある参照点（Iter61）に対しても，405 行時点（p=0.0139）より p が悪化している」**
という点で型が異なる．すなわち**検出力不足では説明できない**．

**2-3. 条件付き割合（S5''-a）は 135 行時点の水準まで戻っている**

| | 合成行数 | S5''-a（rank_1 正解行に占める rank_2 正解） | Wilson 95% CI |
|---|---|---|---|
| Iter61 | 135 | 12/41 = **0.2927** | [0.176, 0.445] |
| Iter64 | 405 | 24/72 = **0.3333** | [0.235, 0.448] |
| **Iter65** | **810** | **21/72 = 0.2917** | [0.199, 0.405] |

**分母（rank_1 正解行数）は Iter64・Iter65 とも 72 で完全に同一**（行の同一性も 62/72 が共通）
であり，分母の変動で説明できない．**6 倍の合成データを投じた Iter65 の条件付き割合が，
135 行時点の Iter61（0.2927）とほぼ同値に戻っている**ことは，この指標に関して
「量」が効いていないことの直接的な表れである（3 点とも CI は互いに重なるため，単点では
どれも有意差ではない．ここで読むべきは有意性ではなく，**6 倍の投入に対して点推定が
前進していない**という系列の形である）．

**2-4. S2（+2.0pt）の中身は「2 ドメイン性」ではなく「全滅行の部分点化」である**

これが本イテレーションで最も重要な機序上の発見である．複合 100 行を被覆数で分解した:

| | 2 個被覆 | 1 個被覆 | 0 個被覆 | 被覆ペア計/200 | recall |
|---|---|---|---|---|---|
| 基準線 | 3 | 63 | 34 | 69 | 0.345 |
| Iter61（135） | 12 | 65 | 23 | 89 | 0.445 |
| Iter63 | 17 | 64 | 19 | 98 | 0.490 |
| Iter64（405） | **24** | 61 | 15 | 109 | 0.545 |
| **Iter65（810）** | **21** | **71** | **8** | **113** | **0.565** |

Iter65 の recall 増（109→113 ペア，+4）は，**0 個被覆行が 15→8（−7 行）へ減って 1 個被覆へ
移った寄与（+10 ペア）から，2 個被覆行が 24→21（−3 行）へ落ちた損失（−6 ペア）を差し引いた
正味**である．**つまり S2 の +2.0pt は「2 つとも当てる能力」ではなく「1 つも当たらない行を
減らす能力」に由来しており，S2 と S5''（本レバーの主基準）が同じ方向を向いていない．**
S2 が PASS したことをもって「2 ドメイン性が改善した」と読んではならない．

**2-5. rank_1 と rank_2 の負の関連が再出現している**

探索的診断値として事前登録された Fisher の正確確率検定（複合 100 行，rank_1 正解 × rank_2 正解）:

| | 分割表 [[両正解, r1のみ],[r2のみ, 両誤り]] | Fisher p | オッズ比 |
|---|---|---|---|
| Iter61 | [[12,29],[36,23]] | 0.0023 | 0.264 |
| Iter63 | [[17,44],[20,19]] | 0.0211 | 0.367 |
| Iter64 | [[24,48],[13,15]] | 0.2538 | 0.577 |
| **Iter65** | **[[21,51],[20,8]]** | **0.0002** | **0.165** |

Iter64 で「負の関連が消失しつつある」と読めた系列は，**Iter65 で系列中もっとも強い負の関連へ
反転した**．内訳を見ると，**rank_1 が外れた 28 行では rank_2 正解が 13/28→20/28 へ大きく増え，
rank_1 が当たった 72 行では rank_2 正解が 24/72→21/72 へ減っている**．
これは「ヘッドが 1 行につき 2 つのドメインを同時に立てられるようになった」のではなく，
**「ヘッドが行ごとに得点質量をどこか 1〜2 個へ強く集中させ，当たり外れが排他的になった」**
挙動と整合する．2-4 の分解（0 個被覆の激減と 2 個被覆の微減）も同じ像を指す．

**3. N5 の退行は，本イテレーションで初めてノイズ範囲を外れた**

N5（単一ドメイン 1500 行の argmax 正解率）の系列と，対応あり McNemar:

| | N5 | Wilson 95% CI | 前反復比 |
|---|---|---|---|
| Iter63 | 0.603333（905/1500） | [0.578, 0.628] | — |
| Iter64 | 0.591333（887/1500） | [0.566, 0.616] | −18 行，p=0.2198（**ノイズ範囲**） |
| **Iter65** | **0.563333（845/1500）** | **[0.5381, 0.5882]** | **−42 行（悪化 129／改善 87），p=0.005276（有意）** |

**判定は明確に「ノイズではない」**．根拠は 3 点である．(a) 対 Iter64 の対応あり McNemar が
p=0.005276 で，Iter63→Iter64 の p=0.2198（同じ実験系列で推定されたノイズ幅）と 1 桁以上異なる．
(b) 減少幅 −2.8pt は，直前反復のドリフト幅 −1.2pt の 2.3 倍である．
(c) **Iter65 単独の Wilson 95% CI の上限 0.5882 が，事前登録の下限 0.590 を下回っている**．
すなわち N5 は「揺れて閾値を割った」のではなく，**閾値が信頼区間の外にある**．
留保 R-G（Iter64 で「N5 は次の増量で最初に割れる指標」と明記された申し送り）は，
本イテレーションで実測として的中した．

**同じことが N2'（`new_top1_accuracy`）にも起きている**: 0.599375（959/1600）→0.573125（917/1600）で
−42 行，対応あり McNemar p=0.007611（悪化 139／改善 97）．Wilson 95% CI は [0.5487, 0.5972] で，
下限 0.5875 は区間内にあるが点推定は 1.44pt 下回る．N2' の FAIL も N5 と同一の実体
（単一ドメイン 1500 行の劣化）で説明でき，独立した 2 つ目の問題ではない
（複合 100 行では rank_1 正解が 72 で Iter64 と同数のため，top1 の −42 行はほぼ全て単一行由来である）．

**3-1. 機序: 訓練信号に占める合成 2 ドメイン文の比率が過半を超えた**

各ドメインの陽性訓練行は「単一ドメイン 150 行（legal のみ 77 行）＋ 合成行（そのドメインを含む
9 ペア × 件数）」である．用量ごとの合成比率は次のとおりで，**Iter65 で初めて 50% を超えた**:

| 用量 | ドメインあたり合成行 | 陽性行計（非 legal） | 合成比率 |
|---|---|---|---|
| Iter61（3/ペア） | 27 | 177 | 15.3% |
| Iter64（9/ペア） | 81 | 231 | 35.1% |
| **Iter65（18/ペア）** | **162** | **312** | **51.9%** |

（Iter65 の再訓練ログの `n_positive`＝312／legal 239 と一致する．）
**各ドメインの決定境界が，過半を「2 つのドメインが混在した文」で学習した状態**になっており，
単一ドメインの短い質問に対する識別が薄まったと解釈するのが自然である．
ドメイン別の内訳もこの解釈と整合する:

| domain | Iter64 | Iter65 | 差 |
|---|---|---|---|
| medical | 70/150 | 51/150 | **−19** |
| natural_science | 92/150 | 79/150 | −13 |
| general | 92/150 | 85/150 | −7 |
| mathematics | 110/150 | 105/150 | −5 |
| legal / social_science | 98 / 83 | 95 / 80 | −3 / −3 |
| history_culture | 117/150 | 115/150 | −2 |
| business_economics / education / computer_science | 70 / 71 / 84 | 72 / 73 / 90 | +2 / +2 / **+6** |

劣化が最も大きい medical は，再訓練ログの CV average_precision が 0.4499 と 10 ドメイン中最下位で
あり，もともと境界が弱いドメインである．誤った先を数えると
**medical→natural_science 9 件，social_science→legal 6 件，natural_science→general 6 件，
medical→education 5 件**と，**いずれも合成ペアで頻繁に共起させた相手ドメインへ流れている**．
これは「合成 2 ドメイン文の増量が，共起ドメイン間の境界を実際に融解させた」という機序の
直接的な証拠である（探索的診断値であり，事前登録された主基準ではない）．

**4. 生成データの質は劣化していない（ノイズ増加仮説は支持されない）**

「量を増やすとノイズも増える」という説明が S5'' の不成立を作っている可能性を検討したが，
**支持されなかった**．3 つの用量の合成データに同一の判定を当てて比較した:

| | 行数 | ペアあたり件数 | 完全重複 | 平均文字数 | echo 疑い（同一パターン） | 短文 <30 字 | ペア内 3-gram Jaccard 平均 | ペア内近似重複 >0.5 |
|---|---|---|---|---|---|---|---|---|
| Iter61 | 135 | 全ペア 3 | 0 | 68.7 | 6（4.44%） | 7（5.19%） | 0.0730 | 0（0.00%） |
| Iter64 | 405 | 全ペア 9 | 0 | 70.8 | 15（3.70%） | 18（4.44%） | 0.0813 | 4（0.25%） |
| **Iter65** | **810** | **全ペア 18** | **0** | **71.6** | **27（3.33%）** | **34（4.20%）** | **0.0821** | **28（0.41%）** |

（echo 判定は journal Iter64 が用いた語句パターンの正確な定義が記録に残っていないため，
本節では 3 ファイルに同一の正規表現を当てて**相対比較のみ**に用いた．絶対値は Iter64 節の
0.74%／1.23% とは定義が異なる．）

**混入率はむしろ微減**（4.44%→3.70%→3.33%），完全重複 0，ペアあたり件数は全 45 ペアで一律 18，
ペア内の多様性（3-gram Jaccard 平均 0.0813→0.0821）も実質横ばいである．近似重複（>0.5）は
0.25%→0.41% と微増したが絶対数 28 件で，810 行の 3.5% にも満たない．
生成時のリトライ枯渇 WARNING も 0 件，A7 の max_jaccard は 0.1895（Iter64 の 0.2344 より低い）．
**つまり 18 件／ペアの時点では生成の飽和・品質劣化はまだ起きていない．**
したがって**「S5'' の不成立はデータ品質の劣化が原因である」という説明は，本データからは
支持されない**．残る説明は 3-1 の「訓練信号の構成比が変わったことによる境界の融解」である．

**5. 仮説との整合**

計画節の仮説は「量が効いているなら被覆 2 個行数は 12→24→X（X>24）と単調増加し，かつ
Iter61（12/100）に対して有意差が出るはず．逆に 810 行で頭打ちになる，または N5 が 0.590 を
割るなら『量では 2 ドメイン性を学べない／量は単一ドメイン判別とトレードオフである』と
結論できる」であった．

**実測は仮説の後半（反証側）に該当する**．しかも**「頭打ち」と「N5 の 0.590 割れ」の
両方が同時に観測された**（X=21≦24，N5=0.5633）．計画節は反証側を明確に事前登録しており，
**この結果は計画が想定した範囲内の反証であって，実験の不成立や測定系の破綻ではない**．
S4（対 Iter64 不一致 889/1600）・A0・A7・A8' はすべて通っており，
config.yml 冒頭が警告する「レバーを読むコードに到達しなかった」型の実験不成立ではない．

仮説が想定していなかった観測は 2 点である．

- (a) **S2 が PASS しながら S5'' が全滅した**（2-4）．計画節は S2 と S5'' が同方向に動くことを
  暗黙に前提していたが，実際には S2 の改善は「0 個被覆行の部分点化」由来で，
  2 個被覆行はむしろ減った．**S2 は本レバーの目的（2 ドメイン性）の代理指標として
  不適切であることが実測で示された**（同じ 200 ペア指標が，行レベルの排他性の変化に対して
  鈍感である）．
- (b) **rank_1 正解行数が 72 で Iter64 と完全に同数**（行の同一性も 62/72 が共通）にもかかわらず，
  その内訳（rank_2 の当たり方）と単一ドメイン側が大きく動いた．
  増量の効果は「複合行の rank_1」ではなく「rank_1 が外れた行の rank_2」と「単一ドメイン行」に
  集中して現れた．

想定外の障害（言語崩れ・発散・OOM・タイムアウト・アサーション違反）は発生していない．

**6. ノイズか信号かの総括**

| 指標 | 変化 | 判定 |
|---|---|---|
| S1（200 ペア被覆，対基準線） | 69→113/200（+22.0pt，improved 62／regressed 18，p=8.14e-07） | **明確な信号** |
| S2（Iter64→Iter65 の増分） | 109→113/200（+2.0pt，CI は大きく重複） | **ノイズ範囲**（閾値ちょうどで PASS） |
| 被覆 2 個行（Iter64→Iter65） | 24→21（discordant 7／4，p=0.5465） | **ノイズ範囲**（減少自体は有意でない） |
| 被覆 2 個行の**増分の系列** | +9 → +12 → **−3**（対 Iter61 の p が 0.0139→0.1237 へ悪化） | **頭打ちの信号**（検出力不足では説明不能） |
| S5''-a（条件付き割合） | 0.2927（135行）→0.3333（405行）→**0.2917（810行）**，分母は 72 で同一 | **前進なし**（6 倍投入で 135 行水準へ回帰） |
| rank_1 × rank_2 の負の関連 | Fisher p 0.2538→**0.0002**，OR 0.577→**0.165** | **構造変化の信号（排他性の強化）** |
| **N5** | 0.5913→**0.5633**（−42 行，p=0.005276，CI 上限 0.5882 < 閾値 0.590） | **有意な退行．ノイズではない** |
| **N2'** | 0.599375→**0.573125**（−42 行，p=0.007611） | **有意な退行**（実体は N5 と同一） |
| education 被覆 | 6/20→7/20 | ノイズ範囲（PASS） |
| medical 被覆 | 20/28→18/28 | ノイズ範囲（下限ちょうどで PASS） |
| 生成データの質（echo 率・多様性） | 3.70%→3.33%，Jaccard 0.0813→0.0821 | **劣化の兆候なし** |

**確信度**: 本イテレーションの主要な判定に関する確信度は**高い**．追加反復を要する種類の
曖昧さ（Iter64 の R-H＝検出力不足）は，本イテレーションでは主要な論点ではない．理由は，
(a) 不成立の主基準 3 つのうち S5''-a・S5''-c は有意性検定を伴わない点推定・単調性の条件であり
検出力の問題を受けない，(b) 唯一の有意性条件 S5''-b は検出力のある参照点（Iter61）で測って
なお p が 405 行時点より**悪化**している，(c) 非退行 N5 の退行は n=1500 で p=0.005276，
かつ閾値が CI の外にある，の 3 点による．
ただし**「3 用量のうち最終点 1 つだけが反転した」という系列の形自体は，1 回の生成乱数
（temperature=0.8 の再生成であり Iter64 の 405 行を包含しない，計画節既知の制約）の実現値に
依存する**．「810 行が 405 行より悪い」という順序関係そのものを確定事実として扱うことには
留保が要る．一方で**「810 行にしても 405 行を上回らなかった」という頭打ちの判定と，
N5 の有意な退行**は，この留保の影響を受けない．

**7. 次フェーズ（rc-reflector）への示唆**

- **機械的照合の結果として，rejected 分岐 1（S5''-c 不成立かつ S5''-a 不成立）と
  rejected 分岐 3（N5<0.590）の前提が文言どおり 2 つとも成立している**一方，
  partial の 3 分岐はいずれも前提が成立しない．**確定判定は reflector の担当であるが，
  事前登録の文言に照らすと選択の余地は小さい**．特に N5 については計画節が
  「R-G により緩めないと事前に決めているため，割った時点で rejected とする」と
  明示的に事後緩和を封じている点に注意が必要である．
- **追加反復（さらなる増量，例: 36 件／ペア）は分析上は推奨しない**．理由は
  (a) 用量反応の増分が 405→810 で既に 0 を跨いだこと，(b) N5 の劣化が 405→810 で
  ノイズ範囲（p=0.22）から有意（p=0.0053）へ移行しており，増量を続ければ
  単一ドメイン判別の劣化だけが確実に進むと予測されること，(c) 生成品質は劣化していない
  （4）ため「もっと綺麗なデータで量を増やせば良い」という逃げ道が本データでは塞がれていること，
  の 3 点である．**`multilabel_synthetic_volume_dose_response` は 2 用量（9・18）で
  用量反応の形が確定したとみなせる**．
- **S2 を本レバーの成功条件に使い続けることの妥当性を再検討する材料がある**（2-4）．
  S2（200 ペアの部分点）と S5''（行レベルの 2 個被覆）は Iter65 で初めて逆方向に動いた．
  対外記述で `compound_domain_set_recall`=0.565 を「2 ドメイン性の改善」として引用すると
  実測と食い違う（2 個被覆行は 24→21 で減っている）．
- **次のレバーの方向に関する分析上の手がかり**: 3-1 の機序（陽性訓練行に占める合成比率が
  51.9% に達し共起ドメイン間の境界が融解した）は，「合成データの量」ではなく
  **「単一ドメイン行と合成行の混合比率そのもの」または「合成行の陽性ラベルの与え方
  （2 ドメイン両方を陽性にするか，重みを下げるか）」がレバーになりうる**ことを示唆する．
  また 2-5 の排他性（1 行内で得点質量が 1 個へ集中する）は，ヘッド側の構造
  （OvR・未較正）に起因する可能性があり，Iter62 で試した較正とは別の切り口になりうる．
  **ただしレバーの選定は reflector の担当であり，ここでは分析から見える方向の列挙に留める．**
- **継続する留保**: R-F（実行時経路への配線を今回も行っていない．7 回目）・R-E（rank_1・rank_2 を
  単一ヘッドが決める構成）・R-G（**本イテレーションで的中．N5 は実際に最初に割れた**）・
  既知の低品質行混入（率としては劣化していないが未解消）．
  **R-H（S5-b 型条件の検出力限界）は，計画節が指示したとおり検出力のある参照点
  （Iter61）へ置き換えることで運用上は解消されており，本イテレーションの不成立を
  検出力不足で説明することはできない**．

### 考察 (Iter65)

**判定: rejected（確定）**

「計画 (Iter65)」節が事前に機械的に定めた判定規則へ実測を照合した結果，
**rejected 分岐 1（S5''-c 不成立 かつ S5''-a 不成立）と rejected 分岐 3（N5<0.590）の
2 つが同時に成立**し，partial の 3 分岐はいずれも前提が成立しない．
したがって `multilabel_synthetic_volume_dose_response=uniform_eighteen_per_pair` は
**rejected** で確定し，本レバー（values 単一値）は試し切り＝クローズする．

- 主基準 S5''：**3 条件すべて FAIL**．S5''-a 21/72=0.2917（閾値 0.3333 超が条件）・
  S5''-b p=0.1237（対 Iter61，閾値 0.05）・S5''-c 12→24→**21**（X>24 が条件）．
- 非退行：N3 15/30・N6'' education 7/20・medical 18/28 は PASS，
  **N2' 0.573125（<0.5875）・N5 0.563333（<0.590）は FAIL**．
- S1・S2・S3・S4 は PASS だが，**S2 の PASS は主基準と同方向ではない**（下記 機序 3）．
- N5 の FAIL については計画節が「R-G により緩めないと事前に決めているため，割った時点で
  rejected とする」と事後緩和を明示的に封じている．**この封じ手が実際に発動した最初の例である．**

**確定した機序**

1. **用量反応は 405 行で頭打ちになっている**．被覆 2 個行の増分は +9（135 行）→+12（405 行）→
   **−3（810 行）**と 0 を跨いだ．24→21 の減少自体は有意でない（discordant 7/4，p=0.5465）ため
   「増量が能動的に壊した」とは言えないが，**検出力のある参照点（Iter61=12/100）に対する p が
   0.0139（405 行）→0.1237（810 行）へ悪化している**ため，「頭打ち」の側は検出力不足では
   説明できない．Iter64 の partial は R-H（1 反復増分の検出力不足）型の未達だったが，
   Iter65 の不成立は型が異なる．
2. **量は単一ドメイン判別とのトレードオフである**．各ドメインの陽性訓練行に占める合成 2 ドメイン文の
   比率は 15.3%(135 行)→35.1%(405 行)→**51.9%(810 行)** と初めて過半を超え（再訓練ログの
   `n_positive`=312 と一致），N5 が −42 行・p=0.005276 で有意に退行した（Wilson 95%CI 上限
   0.5882 が閾値 0.590 の外＝「揺れて割った」のではない）．誤り先は
   medical→natural_science 9 件・social_science→legal 6 件・natural_science→general 6 件・
   medical→education 5 件と，**いずれも合成ペアで頻繁に共起させた相手ドメイン**であり，
   「共起ドメイン間の決定境界が融解した」という機序の直接証拠になっている．
   N2' の FAIL は独立した第 2 の問題ではなく N5 と同一実体である（複合 100 行の rank_1 正解は
   72 で Iter64 と同数のため，top1 の −42 行はほぼ全て単一ドメイン行由来）．
3. **S2（`compound_domain_set_recall`）は本レバーの主基準の代理として不適切である**．
   Iter65 の +2.0pt は，0 個被覆行が 15→8（−7 行，+10 ペア）へ減った寄与から
   2 個被覆行の減少（24→21，−6 ペア）を差し引いた正味であり，**「2 つとも当てる能力」ではなく
   「1 つも当たらない行を減らす能力」に由来する**．S2=0.565 を「2 ドメイン性の改善」として
   対外記述に引用してはならない（2 個被覆行は減っている）．
   また rank_1 正解 × rank_2 正解の Fisher 検定は 0.2538（Iter64）→**0.0002（Iter65，OR=0.165）**と
   系列中もっとも強い負の関連へ反転しており，ヘッドが行ごとに得点質量を 1 箇所へ集中させ
   当たり外れが排他的になった挙動と整合する．

**反証された対立仮説**: 「量を増やすとノイズも増えるから不成立になった」は支持されない．
echo 疑い率 4.44%→3.70%→**3.33%**（同一定義での相対比較）・完全重複 0・ペア内 3-gram Jaccard
0.0813→0.0821（横ばい）・A7 max_jaccard 0.1895（Iter64 の 0.2344 より低い）・生成リトライ枯渇
WARNING 0 件．**18 件／ペアの時点では生成の飽和・品質劣化は起きていない**ため，
「もっと綺麗なデータで量を増やせばよい」という逃げ道は本データで塞がれている．

**学び（次の自分への申し送り）**

- **さらなる増量（36 件／ペア等）は行わない．**増分は既に 0 を跨ぎ，N5 の劣化は
  ノイズ範囲（p=0.22）から有意（p=0.0053）へ移行した．増やせば単一ドメイン判別の劣化だけが
  確実に進む．「量」という変数は 135/405/810 の 3 点で形が確定した．
- **ただし Iter64→65 は 2 つの量を同時に動かしている**：合成文の「本数（＝語彙的多様性）」と
  「陽性訓練行に占める質量比」である．N5 の退行がどちらに由来するかは本イテレーション単独では
  分離できない．**次のレバーはこの交絡の分離に充てる**（下記）．
- **[重要な事実誤認の訂正]** Iter63/64/65 の計画節と config.yml の note は固定構成を
  「未較正 OvR ロジスティック回帰」と書いてきたが，**実際のコード
  `scripts/train_multilabel_dispatch_head.py:172-176` は Iter62（commit 927e363）以降
  `OneVsRestClassifier(CalibratedClassifierCV(LogisticRegression(class_weight='balanced'),
  method='sigmoid', cv=5, ensemble=True))` であり，ヘッドは Platt 較正済みである**．
  Iter63〜65 で同一のため各イテレーション間の比較の妥当性には影響しないが，
  「ヘッド構造（OvR・未較正）を次に疑う」という分析節の示唆は，**較正が既に入っている**という
  前提で読み替える必要がある．この種の「記述と実装の乖離」は config.yml 冒頭が警告する
  実験不成立の温床であり，レバーを選ぶ際は note の記述ではなく必ずコードを読むこと．
- 継続する留保: R-F（実行時経路への配線を今回も行っていない．**7 回目**）・R-E（rank_1/rank_2 を
  単一ヘッドが決める）・既知の低品質行混入（率としては劣化していないが未解消）．
  R-G は本イテレーションで的中し役目を終えた（N5 は実際に最初に割れた）．
  R-H は検出力のある参照点への置換で運用上解消された．

**次のレバー（単一レバー原則）**

config.yml の levers を全文精査した結果，未試行の値が残っているのは
`production_deployment_gap`・`dispatch_policy=adaptive_confidence_gap`（いずれも
`config.yaml` のスキーマ変更または実機本走を要し，前者は education 系の別系統の課題，
後者は着手前のユーザー確認が必要と明記）のみで，**本研究線（多ラベル dispatch）で
オフラインに自律着手できる未試行の値は残っていない**．そこで SKILL.md 「停止条件」の
選択肢 1 に従い，**本イテレーションの学びから新レバーを考案して `levers` 末尾へ追記した**．

- 新レバー: **`multilabel_training_mixture_ratio = single_domain_rows_duplicated_x2`**
  （config.yml 末尾に追記済み）．
- 内容: **合成側を Iter65 の 810 行のまま固定（再生成しない）**し，単一ドメイン訓練行
  （`data/classifier_train.jsonl` 1427 行）を 2 重化した 2854 行を新規ファイルとして
  `--train-data` に渡すことで，**合成文の質量比のみを 51.9%→35.1%（Iter64 水準）へ戻す**．
- 選定理由: 上記のとおり Iter64→65 は「本数」と「質量比」を同時に動かしており，
  N5 の退行の原因を分離できていない．本数を固定して質量比だけを戻せば，
  (a) N5 が 0.590 へ回復するなら退行の原因は**質量比**であり，「本数を増やしつつ質量比を保つ」
  という次の設計（用量反応の再開）が正当化される．(b) 回復しないなら原因は
  **合成文そのものの分布シフト**であり，合成データを単一ドメイン分類の学習から切り離す
  2 ヘッド構成へ移る．**どちらに転んでも次の一手が一意に決まる**設計である．
- `sample_weight` ではなく行の複製を使うのは，Iter32 で実測した
  `class_weight='balanced'` × `sample_weight` の乗算結合（意図した重みが減衰する）を
  構造的に避けるためである．
- 既知の留保（計画フェーズで必ず事前登録すること）: 複製行が
  `CalibratedClassifierCV(cv=5)` の fold を跨ぐため較正値が楽観化しうる．
  N5 は argmax の順序しか使わないため主基準への影響は限定的だが，較正値そのものを引用しないこと．
- コスト: 合成生成不要・ヘッド再訓練約 10 分＋採点／統計 2 分＝**15 分程度**．
  オフライン完結・スキーマ変更なし・実機本走なしのため自律着手してよい．
- 次イテレーション名: **「合成文の質量比のみを Iter64 水準へ戻す（単一ドメイン行の2重化）」**

**要人間判断**: なし（レバーの考案・追記はいずれも可逆な判断の範囲）．
ただし累積した申し送りとして，R-F（実行時経路への配線が 7 イテレーション連続で未実施であり，
本研究線のオフライン成果はいずれも実機での有効性を主張できない）は，
研究の結論を確定させる段階で人間判断を要する．

## Iteration 64: 2ドメイン合成訓練事例の一律増量（3→9件/ペア）

### 調査 (Iter64)

**問い**（config.yml:1044-1089＝backlog B97 が既に詳細な設計を事前登録している．
先行研究の新規調査ではなく，一次情報＝コード・実データの確認を優先した）

- Q1: `scripts/generate_multidomain_training_examples.py` の `--per-pair`／`--per-pair-legal` は，
  Iter61（3/3）→Iter64（9/9）への変更が件数パラメータの変更だけで完結するか（生成プロンプト・
  ペア集合・A7 リーク監査ロジックが件数に依存して壊れないか）．
- Q2: Iter61/Iter63 の既存資産（訓練データ・埋め込みキャッシュ・基準線・予測 JSONL・ヘッド）は
  実在し再利用可能か．
- Q3: `scripts/evaluate_dispatch_candidate_ranking.py` の `--rank1-source head_argmax`（Iter63 で追加）
  は，Iter64 の新訓練データで再訓練したヘッドに対してそのまま使えるか．
- Q4: `scripts/compute_iter59_ranking_stats.py` は config.yml が事前登録した成功条件 S1〜S4・
  非退行 N2'/N3/N5 をそのまま計算できるか．特に S5（複合行の rank_1 正解のうち rank_2 も正解した
  割合が Iter63 の 17/61=27.9% を上回ること）が `metrics._mcnemar_from_correctness()` を薄く呼ぶだけで
  新規実装なしに計算できるか．
- Q5: 405 件の LLM 生成＋ヘッド再訓練＋オフライン採点の実行コストは，Iter61（135 件）からの相似形
  として妥当か．

**分かったこと（全文読了・実行確認による一次情報）**

1. **`generate_multidomain_training_examples.py` は全文読了**（346 行）．`--per-pair`／
   `--per-pair-legal` は `_rows_for_pair()`（:162-166）で `"legal" in (domain1, domain2)` の分岐に
   渡すだけの単純な整数引数であり，`--per-pair 9 --per-pair-legal 9` は両者を同値にする＝**全 45 ペア
   一律 9 件という config.yml の設計を，コード変更なしに CLI 引数だけで実現できる**（config.yml の
   note が「一律増量」と明記する意図とコードの挙動が一致している）．生成プロンプト
   `_build_prompt()`（:113-125）・ペア集合 `itertools.combinations(domains, 2)`（:179，10 ドメインで
   45 ペア固定）・A7 リーク監査 `audit_leak()`（:211-237，`--output` ファイル全体を読むだけで件数に
   依存しない集計）はいずれも件数パラメータを直接参照せず，件数依存で壊れる箇所は見当たらなかった．
   `_MINIMUM_ACCEPTABLE_ROW_COUNT=120`（:70）は 405 目標に対し十分小さく，フロアとして機能する．
   行 ID は `f"synth-{domain1}-{domain2}-{written:03d}"`（:194，ゼロ埋め 3 桁）で 9 件でも桁あふれしない．
   唯一の留意点は `already_generated` が**全 45 ペアを通じた 1 つの集合**（:178）であるため，
   9 件/ペアに増やすと 3 件/ペアのときより「他ペアと文言が重複して弾かれ再試行する」頻度が
   構造的に上がりうる点だが，これは `_MAX_GENERATION_ATTEMPTS_PER_SLOT=3` 回のリトライで吸収する
   既存設計の範囲内であり，コード変更を要する問題ではない．
2. **既存資産は全て実在**（`ls -la` で確認）: `data/classifier_train.jsonl`（600,281B，Jul 30），
   `data/classifier_train_multidomain_iter61.jsonl`（41,551B，Sep19 06:31），
   `models/dispatch_multilabel_head_iter61.joblib`（66,134B，Sep19 06:35），
   `results/iter59_query_embeddings.npz`（9,971,712B，Sep19 02:29），
   `results/20260918_202613/results.jsonl`（3,510,699B，基準線），
   `results/iter63_multilabel_ranking_predictions.jsonl`（982,241B，Sep19 10:58）．いずれも
   mtime が直近イテレーションの記録と整合し，較正済みヘッド（Iter62）は今回使わない方針
   （config.yml 既定）と一致する．
3. **`evaluate_dispatch_candidate_ranking.py` の `--rank1-source head_argmax` は全文読了**（591 行）．
   `_load_head()`（:173-186）は `--head` のパス・アーティファクト形式にのみ依存し，訓練データの
   由来（Iter61 の 135 行か Iter64 の 405 行か）を一切参照しないため，**Iter64 で再訓練した
   `models/dispatch_multilabel_head_iter64.joblib` を `--head` に渡すだけで head_argmax モードは
   コード変更なしにそのまま使える**．`build_new_rows()` の rank_1 決定式（:250-255）・
   `_assert_rank1_matches_head_argmax`（A9，:288-306）・`selected_domain` の上書き（:255）・
   `_compute_rank1_change_count`（S4，:309-322）は Iter63 で確定済みの実装のまま流用でき，
   本レバー固有の追加改修点は見当たらない．
4. **`compute_iter59_ranking_stats.py`**（388 行）を確認したところ，Iter63 の調査で判明した内容
   （journal Iter63 調査 6・7）から変化はなかった：S1（`_compute_s1`，:106-127，200 ペア McNemar）・
   N3（`_compute_n3`，:228-268，legal 非退行）は無改造でそのまま使える．**S5（複合 100 行の
   rank_1 正解のうち rank_2 も正解した割合）に対応する集計関数はこのスクリプトに存在しない**が，
   `metrics._mcnemar_from_correctness()`（`metrics.py:228-258`）は `{id: bool}` の辞書 2 つを
   受け取る汎用関数であり，実際に `results/iter63_multilabel_ranking_predictions.jsonl` の
   `head_scores`／`dispatched_domains` フィールドのみを使い，複合 100 行のうち rank_1 正解 61 行・
   そのうち rank_2 も正解 17 行（＝27.9%，Iter63 note の数値と完全一致）を**独立に再現できた**
   （新規の統計機構は不要，「行 ID をキーにした bool 辞書を渡すだけ」という Iter63 の申し送りが
   本レバーでもそのまま成立する）．N2'／N5 も Iter63 と同じアドホック集計パターン
   （`selected_domain` の上書きと `_compute_single_domain_argmax_accuracy`）をそのまま踏襲できる．
5. **コスト見積り**: Iter61 のコミット時刻列から実測した（`git log` と各成果物の mtime）．
   Iter60 クローズ（06:09:41）→生成完了（`classifier_train_multidomain_iter61.jsonl` 06:31:20，
   135 行で約 21 分）→ヘッド再訓練完了（`dispatch_multilabel_head_iter61.joblib` 06:35:06，約 3.7 分）
   →オフライン採点完了（`iter61_multilabel_ranking_predictions.jsonl` 06:36:18，約 1.2 分，
   埋め込みキャッシュ大部分ヒット）．生成は `generate_all_rows()`（:169-196）がペア・スロットとも
   逐次 `await` するシーケンシャルループであり同時実行しないため，**件数と所要時間はほぼ線形**．
   405 行（135 行の 3 倍）に対しては生成が約 60〜70 分，訓練・採点は行数増加が全体
   （既存 1427＋405 行）に占める比率で小さいため大きく伸びないと見込まれ，**合計 70〜90 分程度**が
   妥当な見積りである．人間の常時監視を要さないオフライン完結の処理であり，実行時間としても
   1 サイクル内で十分許容範囲内である．

**結論**

config.yml（backlog B97）が設計した Iter64 の単一レバーは，計画どおり実行可能である．
`--per-pair 9 --per-pair-legal 9` は生成スクリプトを無改造のまま「全 45 ペア一律 9 件」を
コード上も正しく実現し，Iter61/Iter63 の全既存資産は実在・再利用可能，`--rank1-source head_argmax`
は Iter64 の再訓練ヘッドに対してもコード変更なしに動作する．S5 を含む成功条件・非退行条件は
Iter63 と同じ「既存関数を薄く呼ぶアドホック集計」パターンで新規実装なしに計算でき，
今回の調査で複合 100 行の 61/100・17/61（27.9%）という Iter63 の基準値を独立に再現した．
実行コストは 135 行→405 行の線形スケールから見積もって 70〜90 分程度であり，Iter61 からの
相似形として妥当である．実行前提に齟齬は見当たらなかった．

**次フェーズへの示唆**

- `--per-pair 9 --per-pair-legal 9` は「値を揃える」ことで一律増量を実現する設計であり，
  計画フェーズで別の値（例えば legal だけ多く）を書かないよう注意すること（config.yml の
  「一律増量に限定する理由」＝配分決定によるリーク再発防止と整合させる）．
- S5 の実装は Iter63 と同型（`metrics._mcnemar_from_correctness()` を複合 100 行の bool 辞書で
  呼ぶアドホックコード 1 つ）で足りるが，**分母が可変**（rank_1 正解行数が Iter63 の 61 から
  変わりうる）ため，config.yml が既に指示しているとおり「複合 100 行の被覆 2 個の行数＝17 行との
  対応あり McNemar」も併せて実装すること．
- N2'（top1_accuracy）は Iter63 と同様に `selected_domain` を新 rank_1 で上書きしないと自明 PASS
  になる落とし穴が再発する．Iter64 でも `--rank1-source head_argmax` を使う以上，この上書きロジック
  自体は `build_new_rows()` に既に実装済み（:255）なので，計画フェーズは「N2' はこの上書き後の
  `selected_domain` で計算する」ことを改めて明記するだけでよい．
- N6' の閾値（education ≧3/20・medical ≧18/28）は Iter63 の実測水準を下限とする config.yml の
  設計を計画フェーズでそのまま事前登録すればよく，コード上の追加確認は不要だった．
- 実行コストの見積り（70〜90 分）は生成ステップが律速であり，途中で失敗した場合の再開性
  （`generate_all_rows()` は失敗した個々のスロットを警告してスキップするだけで例外を送出しない
  設計）を踏まえ，実装フェーズは生成ログの標準エラー出力を確認する運用を申し送るとよい．

### 計画 (Iter64)

**仮説**

Iter63 で「rank_1 が当たった行のうち rank_2 も当たる割合」が 29.3%（Iter61）→27.9%（Iter63）と
順位付け層をどう変えても動かないことが確定し，ボトルネックは**多ラベルヘッドが 2 ドメイン性を
学べていないこと＝訓練信号の量**へ移動した．2 ドメイン合成訓練事例を全 45 ペア一律で 3 件／ペア →
9 件／ペア（135→405 行）へ増やせば，ヘッドが 1 行に 2 ドメインが同時に立つ構造をより多くの事例から
学び，**rank_1 正解行での rank_2 正解割合（27.9%）と複合 100 行の集合再現率（0.490）が向上する**はずである．
逆に向上しないなら「量」では 2 ドメイン性を学べないことになり，残るのは生成文の品質・データセット
再設計・埋め込み表現の適応に限られる．

**単一レバー（今回変更する唯一の変数）**

`multilabel_synthetic_volume`: `uniform_three_per_pair`（Iter61 の実質値＝`--per-pair 3
--per-pair-legal 3`，45 ペア一律 3 件・135 行）→ **`uniform_nine_per_pair`**（`--per-pair 9
--per-pair-legal 9`，45 ペア一律 9 件・405 行目標）．
**両引数を同値にすることで「配分の決定」が存在しない形を保つ**（`_rows_for_pair()` は
`"legal" in (domain1, domain2)` で分岐するだけなので，同値なら全ペアが同数になる）．
どのドメインを厚くするかを選ぶと Iter60 の R-A 型リーク（テスト集合由来の設計情報の混入）が
再発するため，**legal だけ多くする等の非一律な値は書かない**．スクリプトの改修は行わない．

**確定した実装仕様（本フェーズの決定事項）**

1. **コード変更は一切行わない**．変更は CLI 実引数値（`--per-pair 3 --per-pair-legal 3` →
   `--per-pair 9 --per-pair-legal 9`）と，Iter61/63 の成果物を保護するための出力パス名のみ．
   出力パス名の変更は測定対象に影響しない（ファイル I/O のみ）．
2. **ヘッドは Iter64 の訓練データで再訓練する**（Iter61 の手順を踏襲．
   `scripts/train_multilabel_dispatch_head.py` に `--multilabel-train-data
   data/classifier_train_multidomain_iter64.jsonl` を渡す）．較正は導入しない（未較正
   `OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight="balanced"))` のまま）．
3. **採点は `--rank1-source head_argmax` を主系として固定する**（Iter63 で実装済み・コード変更不要．
   `_load_head()` は訓練データの由来を参照しないため，Iter64 のヘッドをそのまま `--head` に渡せる）．
   `build_new_rows()` は `head_argmax` モードで `selected_domain` を新 rank_1 で上書きするため，
   **N2' はこの上書き後の `selected_domain` に対して算出した値で判定する**（上書きがないと N2' が
   自明 PASS になる Iter63 の落とし穴）．
4. **統計は `scripts/compute_iter59_ranking_stats.py` を無改造で使う**．S1・S2・S3・N3 はそのまま
   利用し，**N1 は本構成では定義上 `pass:false` になるため記録のみで判定に用いない**
   （Iter63 と同じ．N2 の `exact_match` も同様に記録のみで，数値 `new_top1_accuracy` を N2' と照合する）．
   **S5 と N6' は Iter60〜63 と同じ「既存関数を読み取り専用で薄く呼ぶアドホック集計」で算出する**
   （S5 は `metrics._mcnemar_from_correctness()` に行 ID をキーとする 100 件の bool 辞書を 2 つ渡すだけ．
   N6' は `compute_iter59_ranking_stats._domain_pair_coverage_maps()` を流用．公式の採点・統計パスには
   手を入れない）．新規の統計機構は実装しない．

**固定する構成（Iter61/63 から一切変えない）**

- 生成側: 生成プロンプト `_build_prompt()`，フィルタ F1〜F4，`_GENERATION_TEMPERATURE=0.8`，
  生成モデル `schroneko/llama-3.1-swallow-8b-instruct-v0.1:q4_k_m`，ペア集合
  （`itertools.combinations` の 45 ペア），A7 リーク監査の閾値 0.9，`--train-data
  data/classifier_train.jsonl`．
- 訓練側: ヘッド種別（未較正）・単一ドメイン訓練データ `data/classifier_train.jsonl`（1427 行）・
  埋め込みモデル `nomic-embed-text`．**Iter62 の較正済みヘッドは使わない**．
- 採点側: 基準線 `results/20260918_202613/results.jsonl`（`compound_domain_set_recall`=0.345，
  `top1_accuracy`=0.5975，複合 100 行の rank_1 正解 41/100・2 ドメイン被覆 12/100），
  埋め込みキャッシュ `results/iter59_query_embeddings.npz`（**評価クエリ 1600 問は完全ヒットの想定．
  新規 embed は合成訓練文 405 件のみ**），rank_2 の選択ロジック，`_head_scores()`，A2/A3/A6/A9，
  N5 の計算，`--rank1-source head_argmax`，統計スクリプト（無改造），`config.yaml`．
- **実行時経路（`node.py` のルータ）への配線は本イテレーションでも行わない**（B94/B95．5 回目）．

**出力ファイル命名（Iter61/63 の成果物を上書きしないこと）**

| 種別 | 既存（保護・読み取り専用） | Iter64（新規作成） |
|---|---|---|
| 合成訓練データ | `data/classifier_train_multidomain_iter61.jsonl`（135 行） | `data/classifier_train_multidomain_iter64.jsonl`（405 行目標） |
| ヘッド | `models/dispatch_multilabel_head_iter61.joblib` | `models/dispatch_multilabel_head_iter64.joblib` |
| 予測 | `results/iter63_multilabel_ranking_predictions.jsonl` | `results/iter64_multilabel_ranking_predictions.jsonl` |
| 統計 | `results/iter63_stats.json` | `results/iter64_stats.json` |

**実行コマンド（`--ollama-host` は疎通する方を使う．Iter61 実績では生成・訓練が
`192.168.15.100`（既定ポート 11434），採点が SSH ローカルフォワード `127.0.0.1:11435`．
ホスト指定はモデル名で同一性が担保されるため単一レバー原則に抵触しない）**

```
# 0) 合成訓練データの生成（唯一のレバー変更点: --per-pair 3→9, --per-pair-legal 3→9．405 件目標）
#    generate_all_rows() はペア・スロットとも逐次 await のため約 60〜70 分．
#    スキップされたスロットは stderr に WARNING が出るだけで例外にならないので，
#    実装フェーズは標準エラー出力を必ず確認し，最終行数をログに残すこと．
uv run python -m scripts.generate_multidomain_training_examples \
    --train-data data/classifier_train.jsonl \
    --model schroneko/llama-3.1-swallow-8b-instruct-v0.1:q4_k_m \
    --ollama-host 192.168.15.100 \
    --per-pair 9 --per-pair-legal 9 \
    --output data/classifier_train_multidomain_iter64.jsonl

# 0') リーク監査（A7．生成物の選別は行わず，近似重複の検出のみ）
uv run python -m scripts.generate_multidomain_training_examples --audit-leak \
    --output data/classifier_train_multidomain_iter64.jsonl

# 1) 多ラベルヘッドの再訓練（1427 + 405 件 embed．A0 は実生成行数で検査される）
uv run python -m scripts.train_multilabel_dispatch_head \
    --train-data data/classifier_train.jsonl \
    --multilabel-train-data data/classifier_train_multidomain_iter64.jsonl \
    --embedding-model nomic-embed-text --ollama-host 192.168.15.100 \
    --output models/dispatch_multilabel_head_iter64.joblib

# 2) 1600 問のオフライン採点（主系＝head_argmax．キャッシュ完全ヒットの想定）
#    --iter59-predictions は引数名に反して汎用．S4（対 Iter63 不一致）を測るため Iter63 の予測を渡す．
uv run python -m scripts.evaluate_dispatch_candidate_ranking \
    --baseline results/20260918_202613/results.jsonl \
    --head models/dispatch_multilabel_head_iter64.joblib \
    --rank1-source head_argmax \
    --embedding-model nomic-embed-text --ollama-host 127.0.0.1 --ollama-port 11435 \
    --embedding-cache results/iter59_query_embeddings.npz \
    --iter59-predictions results/iter63_multilabel_ranking_predictions.jsonl \
    --output results/iter64_multilabel_ranking_predictions.jsonl

# 3) 指標・検定（Iter59〜63 と同一スクリプト・同一手続き．無改造）
uv run python -m scripts.compute_iter59_ranking_stats \
    --baseline results/20260918_202613/results.jsonl \
    --new results/iter64_multilabel_ranking_predictions.jsonl \
    --output results/iter64_stats.json

# 4) S5（rank_1 正解行での rank_2 正解割合・被覆 2 個行の対 Iter63 対応あり exact McNemar）と
#    N6'（education/medical 被覆），および探索的診断値（education の自得点／行内最大の中央値）の
#    アドホック集計．metrics._mcnemar_from_correctness() と _domain_pair_coverage_maps() を
#    読み取り専用で呼ぶ（公式の採点・統計パスには手を入れない）．
```

**成功条件（事前登録．事後変更禁止）**

config.yml の `multilabel_synthetic_volume` note の暫定案を，**閾値を一切変更せずに確定する**．
参照点は基準線 `results/20260918_202613/results.jsonl`（recall 0.345，複合 100 行の rank_1 正解
41/100・2 ドメイン被覆 12/100，`top1_accuracy` 0.5975）と **Iter63**（recall 0.490，rank_1 正解
61/100・2 ドメイン被覆 17/100・割合 17/61=27.9%，`top1_accuracy` 0.60375）で，両者を必ず併記する．

- **S1（有意性の維持）**: 全体 200 ペアの exact McNemar（`scipy.stats.binomtest`，α=0.05）で
  基準線比 **p < 0.05**．
- **S2（全体性能の向上）**: `compound_domain_set_recall` **≧ 0.510**
  （Iter63 の 0.490 を点推定で +2.0pt 以上上回る）．
- **S3（コスト中立）**: `mean_dispatch = 2.000000`（完全一致）．
- **S4（発火の証拠）**: Iter63 予測に対する不一致行数 **> 0**
  （`_compute_a5_iter59_disagreement()` の出力．訓練データ増量がヘッドの出力を実際に変えたこと）．
- **S5（本レバー固有の主基準．2 つの副条件をともに事前登録する）**
  - **S5-a（点推定）**: 複合 100 行のうち **rank_1 が正解した行に占める「rank_2 も正解した行」の割合が
    Iter63 の 17/61 = 27.9% を上回る（> 0.2787）**．分母（rank_1 正解行数）は可変であり，
    Iter64 の実測値で計算する．
  - **S5-b（有意性）**: 複合 100 行の **「2 ドメインとも被覆した行」の対 Iter63 対応あり exact McNemar**
    （行 ID をキーとする 100 件の bool 辞書を `metrics._mcnemar_from_correctness()` に渡す）で
    **p < 0.05 かつ被覆行数 > 17**．
  - **分母可変への対処の明記**: S5-a は割合（ボトルネックの直接測定），S5-b は分母固定 100 行での
    有意性検定であり，両者は互いの弱点（S5-a は分母縮小で見かけ上改善しうる／S5-b は割合の改善を
    直接には測らない）を補う関係にあるため，**両方を事前登録し，adopted には両方の充足を要する**．

**判定規則（事前登録）**

- **S1〜S4 かつ S5-a・S5-b 全充足 かつ N2'/N3/N5/N6' 全充足** → `multilabel_synthetic_volume =
  uniform_nine_per_pair` を **adopted**．「2 ドメイン性は訓練信号の量で学習でき，ボトルネックは
  合成データ量にあった」と結論し，次イテレーションは更なる増量（例: 18 件／ペア）の用量反応を検討する．
- **S5-a 充足・S5-b 不成立（p ≧ 0.05）だが被覆行数 > 17 かつ S2 充足** → **partial**
  （効果量はあるが検出力不足の可能性を記録する）．
- **S5-a 不成立（割合 ≦ 0.2787）だが S2 充足（≧0.510）** → **partial**．「量は集合再現率を押し上げるが，
  rank_1 正解行で 2 つ目を当てる割合というボトルネックは解消しない」と記録する．
- **S2 不成立かつ recall ≧ 0.490（Iter63 水準を割らない）** → **partial**（上げ幅が +2.0pt に届かない）．
- **S2 が Iter63 水準割れ（recall < 0.490）** → **rejected**．
- **S4 不成立（対 Iter63 不一致 0 行）** → **no-op** として rejected 扱いとし，増量がヘッドの出力を
  一切変えなかったことを機序として記録する．
- **非退行 N2'/N3/N5/N6' のいずれかが FAIL** → **adopted にはしない**（S 側が全充足でも最大 partial とし，
  どの指標が退行したかを対外記述の留保として残す）．

**非退行条件（事前登録．閾値は実験後に緩めない）**

- **N2'**: 全 1600 行の `top1_accuracy`（**`selected_domain` を新 rank_1 で上書きした行**に対し
  `metrics.compute_top1_accuracy()` で算出．統計スクリプトの
  `N2_top1_accuracy_invariance.new_top1_accuracy` を用いる）が基準線 0.5975 から 1.0pt を超えて
  低下しない（**≧ 0.5875**）．
- **N3**: legal 自身の被覆 **≧ 8/30**（`_compute_n3()` が自動判定．Iter63 実績 17/30）．
- **N5**: 単一ドメイン 1500 行の argmax 正解率 **≧ 0.590**
  （`_compute_single_domain_argmax_accuracy()`．Iter63 実績 0.6033）．
- **N6'（水準を Iter63 の実測へ更新．本フェーズで確定し，実験後に緩めない）**:
  education **≧ 3/20** かつ medical **≧ 18/28**．
  **更新の理由（事前に明記する）**: Iter62 の 6/20 は「基準線ルータが rank_1 に education を
  過剰出力していたから被覆されていた」構成の値であり，`head_argmax` を固定構成とする本レバーでは
  構造的に達成し得ない（Iter63 で機序が確定済み）．達成不能な条件を課すのは事前登録の趣旨に
  反するため水準のみ更新し，**Iter63 の実測値そのものを下限**とする（これ以上の緩和は行わない）．

**アサーション（no-op・交絡対策）**

- **A0（教師信号が真に多ラベル）**: `_assert_a0_true_multilabel_signal()` が
  `(Y.sum(axis=1) >= 2).sum() == n_synthetic_rows` を検査する．**今回の期待値は実生成行数（目標 405）**．
  床 `_A0_MINIMUM_MULTILABEL_ROW_COUNT=120` は据え置き．`len(mlb.classes_)==10` と被覆ペア数
  （期待 45/45）も併せて報告する．
- **A7（リーク監査）**: 生成物と `build_dataset._COMPOUND_QUESTIONS` の 3-gram Jaccard 最大値が
  閾値 0.9 未満（Iter61 実績 0.1290）．
- **A8'（新設・生成規模の担保）**: 生成行数が **360 行（目標 405 の 90%）未満**であった場合，
  「45 ペア一律 9 件」が実現できていない（`already_generated` の重複棄却による取りこぼし）とみなし，
  **実験を中止して原因を調査する**（下方修正した行数で採点を進めない）．
- **A2/A3/A6/A9**: Iter63 の定義のまま無変更で実行する（A9＝全 1600 行で
  `dispatched_domains[0] == argmax(head_scores)`，A2＝k=2・rank_1≠rank_2）．
- **A5（S4 の計測）**: 対 Iter63 予測の不一致件数 > 0．
- **A1 は `head_argmax` モードでは実行されない**（Iter63 の条件分岐どおり．N1 も `pass:false` 固定）．

**探索的な診断値（主基準にしないこと）**

- education 絡み複合 20 行における「自ドメイン得点／行内最大得点」の中央値（Iter63 実績 0.0899）が
  一律増量で押し上がるか．次に訓練データ側と表現側のどちらへ進むかの判断材料として記録する．
- 低品質行（プロンプト文言の echo）の混入率．**量を 3 倍にすればノイズも 3 倍になる**ため，
  Iter60〜63 と同じ手順で混入率を報告する．

**単一レバー原則の確認（混入チェック）**

- 生成プロンプト・F1〜F4・temperature・生成モデル・ペア集合・A7 閾値 → **無変更**．
- ヘッド構造・較正の有無・単一ドメイン訓練データ・埋め込みモデル・埋め込みキャッシュ → **無変更**．
- 採点スクリプト・`--rank1-source head_argmax`・rank_2 の選択ロジック・統計スクリプト・基準線・
  `config.yaml` → **無変更**（コード変更ゼロ）．
- **ヘッドの再訓練は本レバーに構造的に従属する**（訓練データを増やす以上，再訓練しなければ
  レバーを読むコードに到達しない）ため，独立した第 2 のレバーではない．
- 変更は `--per-pair`／`--per-pair-legal` の実引数値（3→9）と出力パス名のみ．

**想定される不成立時の対応（事前に記録する）**

- **S5-a・S5-b ともに不成立**（量では 2 ドメイン性を学べない）→ 次は (i) 生成文の品質改善
  （echo 混入の除去フィルタ．量ではなく質），(ii) 埋め込み表現の適応（`embedding_adaptation` の
  再訪．Iter40〜43 は education_recall が目的であり，多ラベル 2 ドメイン性という別目的での
  再訪は別物），(iii) 複合設問データセット自体の再設計（research_frontier 相当・人間判断）の順で
  backlog に上げる．
- **S2 が Iter63 水準割れ（rejected）** → 増量がノイズ増（低品質行の 3 倍化）で相殺された可能性を
  診断値（混入率）と突き合わせ，(i) の品質改善を最優先候補にする．
- **非退行 FAIL のみ** → Iter63 と同じく最大 partial とし，どの指標が退行したかを対外記述の
  留保に追加する．

**既知の制約（申し送り）**

- temperature=0.8 の生成乱数性により，405 件は Iter61 の 135 件を包含しない（再生成であって
  追記ではない）．「量」の効果と「別の 405 文である」ことは分離できない点は本レバーの構造的制約である．
- 低品質行（プロンプトの echo）の混入は Iter60 から続く既知の穴であり，単一レバー原則により
  フィルタは変更しない．
- 本構成は rank_1・rank_2 の双方をオフラインのヘッドが決めるため，**実行時経路との乖離が最大**である
  （R-F）．オフラインで adopted となっても実機での有効性は配線・本走なしには主張できない．
- S2/S5 の参考値を効果量の推定値として対外記述に引用してはならない．

**想定コスト**: 生成 約 60〜70 分（律速．逐次 await で件数にほぼ線形）＋ヘッド再訓練 約 4〜8 分
＋オフライン採点・統計 約 2 分＝**合計 70〜90 分**．オフライン完結・`config.yaml` のスキーマ変更なし・
実機 1600 問本走なしのため自律着手してよい．

### Iteration 64 実行済み

**(a) 実行基盤の変更（ユーザー明示指示への対応）**

実装フェーズ開始時点で，生成処理が実験ドメインノード `wafl500`（本来は `general` ドメイン担当）の
Ollama を SSH ポートフォワード（ローカル 11435 番）経由で使って走り始めていた．その最中にユーザーから
「**本体の実験（実機ノード本走）以外は wafl-ctrl5 を用いよ**」という明示的な運用方針指示が入ったため，
オーケストレータが当該プロセスを kill 済み（出力ファイル未生成のため中断による破損データなし）．
これを受け，本実装フェーズで以下を実施した．

1. **`wafl-ctrl5`（Hostname 192.168.15.10，User denjo，ProxyJump wafl 経由）の到達性・GPU確認**:
   `nvidia-smi` で RTX 3060（VRAM 12288 MiB，使用前 110 MiB）を確認．`docker info` で
   `nvidia` runtime が既に登録済み（`/etc/docker/daemon.json`，`nvidia-container-runtime`）で
   あることを確認した．これは wafl500〜509 と同じ GPU パススルー方式（`docker-compose.gpu.yml` の
   `runtime: nvidia` 方式，legacy 方式を採用している理由は同ファイルのコメント参照）と一致しており，
   矛盾なく踏襲できると判断した．
2. **Ollama のセットアップ方法の選定**: wafl500〜509 は `docker-compose.yml`＋`docker-compose.gpu.yml`
   の 2 サービス構成（`ollama` + `app`，ローカル registry 5001 番から `expert-mesh` イメージを配布）
   だが，`wafl-ctrl5` は実験メッシュのノードではなく（`app`／`NODE_ID`／`config.yaml` 配線は不要），
   単に Ollama API を呼ぶだけの制御ホストのため，`app` サービス・mise デプロイタスク一式は導入せず，
   `ollama/ollama:latest` イメージ単体を `docker run` で起動する最小構成にした
   （`--runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all
   -e OLLAMA_KEEP_ALIVE=-1 -p 127.0.0.1:11434:11434 --restart unless-stopped`，コンテナ名
   `ollama-ctrl`，volume `ollama_ctrl_data`）．`OLLAMA_KEEP_ALIVE=-1` は他ノードの
   docker-compose.yml の設定意図（VRAM常時確保）をそのまま踏襲．sudo 不要（docker グループ権限で
   起動できた）．
3. **モデル pull**: `ollama pull` 相当を `/api/pull` 経由で実行．
   (a) 合成データ生成用 `schroneko/llama-3.1-swallow-8b-instruct-v0.1:q4_k_m`（成功）．
   (b) 埋め込み用 `nomic-embed-text`（成功）．
   (c) `judge_model`（`config.yaml:107` も同一モデル名だが，本イテレーションはオフライン完結で
   `aggregation_method=llm_judge` 経路を使わないため pull 不要と判断し，実施しなかった）．
   起動確認（generate/embeddings 各 1 回のスモークテスト）で両モデルとも正常応答．
   `/api/generate` 実行後の VRAM 使用量は 5702 MiB／12288 MiB（8B q4 モデル 1 つロード時点）で，
   12GB の RTX 3060 に十分な余裕があることを確認した．
4. SSH ポートフォワードは wafl-ctrl5 向けに新規ローカルポート **11499** で張った
   （`ssh -fNT -L 11499:localhost:11434 wafl-ctrl5`）．**wafl500 向けの既存トンネル
   （ローカル 11435 番，PID 2086401・2177138）には触れていない**（オーケストレータの指示どおり）．
5. **`.claude/research/config.yml` に恒久的な運用ルールとして追記済み**（冒頭のインフラ説明コメント，
   実機ノード構成の説明の直後）．「本体の実験以外は wafl-ctrl5 を使う」という方針を今後のイテレーション
   でも参照できるようにした．

**(b) 生成〜採点の実行結果（実測値．判定は次フェーズの役割のため記録のみ）**

すべて `--ollama-host 127.0.0.1 --ollama-port 11499`（生成・訓練・採点とも wafl-ctrl5 経由に統一．
journal「計画」節が想定していた「生成・訓練は 192.168.15.100，採点は 127.0.0.1:11435」という
Iter61 実績のホスト分離は，ユーザー指示により行わず全工程を wafl-ctrl5 に一本化した．ホスト指定は
モデル名で同一性が担保されるため単一レバー原則には抵触しない，という計画フェーズの申し送りに従った）．

1. **生成**（12:06:09 完了，開始 11:59 頃．**所要時間 約 7 分**．計画時の見積り 60〜70 分より大幅に
   短い．要因は wafl500 と異なり他プロセスと共有しない専有 GPU で逐次実行できたためと考えられる）:
   `data/classifier_train_multidomain_iter64.jsonl` に **405 行**を生成
   （目標 405 行を完全達成．**A8' の中止基準＝360 行未満を大きく上回るため中止不要**）．
   `already_generated` の重複棄却によるリトライ超過は発生しなかった（stderr に WARNING なし）．
2. **A7 リーク監査**: `max_jaccard=0.234375`（閾値 0.9 未満で PASS，`median_max_jaccard=0.0656`）．
3. **多ラベルヘッド再訓練**（12:20:33 完了，**所要時間 約 14 分**，`data/classifier_train.jsonl`
   1427 行 + 新規 405 行の埋め込み計算が律速）: **A0 PASS**（405 行全てが `Y.sum(axis=1)>=2`，
   45/45 ペア被覆，`len(mlb.classes_)==10`）．ドメイン別 CV ROC-AUC は 0.776（medical）〜0.940
   （history_culture）．`models/dispatch_multilabel_head_iter64.joblib` を出力．
4. **オフライン採点**（`--rank1-source head_argmax`，12:21:38 完了，**所要時間 約 1 分**，
   `results/iter59_query_embeddings.npz` のキャッシュヒットで評価クエリ側の埋め込み計算は不要）:
   `n_rows=1600`，`mean_dispatch=2.0`，`rank2_flip_rate=0.616875`，
   `compound_domain_set_recall=0.545`，`rank1_change_count=435`（対基準線）．
5. **統計（`compute_iter59_ranking_stats.py`，無改造）**:
   - **S1**: `p_value_exact_binomtest=1.6525e-06`（< 0.05，pass=true）．
   - **S2**: `baseline_recall=0.345 → new_recall=0.545`（`delta_pt=0.20`，config.yml 事前登録の
     閾値 ≧0.510 を実測 0.545 で上回る）．
   - **S3**: `mean_dispatch=2.0`（完全一致，`duplicate_rank1_rank2_count=0`）．
   - **S4**: 対 Iter63 予測の不一致 987 行（`rank2_flip_rate=0.616875`）／
     `a5_iter59_disagreement.mismatches=853`．いずれも > 0．
   - **N1**: `mismatch_count=435`（`pass=false`．計画どおり定義上 fail で記録のみ，判定に用いない）．
   - **N2'**: `new_top1_accuracy=0.599375`（`selected_domain` は `build_new_rows()` が新 rank_1 で
     上書き済みであることを実データで確認した上で採用した値．基準線 0.5975 比 +0.19pt，
     config.yml の非退行下限 ≧0.5875 を上回る）．
     **注記**: `compute_iter59_ranking_stats.py` 内のコメント（195・214 行）は「`selected_domain` は
     基準線から verbatim にコピーされる」という別モード（旧設計）向けの記述で，`head_argmax` モードの
     実際の挙動（新 rank_1 で上書き）とは食い違っている．今回，全 1600 行で
     `selected_domain == dispatched_domains[0]`（新 rank_1）であることと，基準線との不一致が
     435 行（N1 の `mismatch_count` と一致）であることを実データで検証した上で，
     `new_top1_accuracy=0.599375` を N2' の値として採用した．**このコメントの食い違いはスクリプト
     自体の修正を要する別件**であり，単一レバー原則により本イテレーションではコードを変更していない．
   - **N3**: legal 自己被覆 `new_legal_self_coverage=16/30`（基準線 8/30 を上回り pass=true）．
   - **N5**: 単一ドメイン argmax 正解率 `0.591333`（floor 0.59 を上回り pass=true）．
6. **S5・N6'（アドホック集計．`metrics._mcnemar_from_correctness()` と
   `compute_iter59_ranking_stats._domain_pair_coverage_maps()` を読み取り専用で呼ぶだけの一時
   スクリプトで算出．公式スクリプトは無改造）**:
   - **S5-a（点推定）**: 複合 100 行の rank_1 正解 72 行のうち rank_2 も正解 24 行
     （**24/72 = 0.3333**，Iter63 の 17/61=0.2787 を上回る）．
   - **S5-b（有意性）**: 「2 ドメインとも被覆した行」は Iter63 17 行 → Iter64 24 行．
     対 Iter63 対応あり exact McNemar（`discordant_a_only=6, discordant_b_only=13,
     discordant_pairs=19`）で **p=0.16867**（α=0.05 を下回らず有意ではない．被覆行数
     24 > 17 は満たす）．
   - **N6'**: education 自己被覆 **6/20**（floor 3/20 を上回る），medical 自己被覆 **20/28**
     （floor 18/28 を上回る）．
7. **実行時間の全体感**: 生成 7 分＋ヘッド再訓練 14 分＋採点・統計 2 分 ＝ **合計 約 25 分**
   （計画時の見積り 70〜90 分より大幅に短い．要因は wafl-ctrl5 が専有 GPU である点，および
   `nomic-embed-text` の呼び出しレイテンシが wafl500 経由より低かった可能性がある．次イテレーションの
   コスト見積りの参考にすること）．

**出力ファイル**: `data/classifier_train_multidomain_iter64.jsonl`（405 行），
`models/dispatch_multilabel_head_iter64.joblib`，
`results/iter64_multilabel_ranking_predictions.jsonl`，`results/iter64_stats.json`．
Iter61/63 の既存資産（`*_iter61.*`, `iter63_multilabel_ranking_predictions.jsonl`）は上書きしておらず，
無変更のまま残っている（`ls -la` で mtime 変化なしを確認済み）．

### 分析（解釈） (Iter64)

採否の最終確定（backlog 記録・config.yml 更新・次レバー選定・commit）は次の考察フェーズ
（rc-reflector）の担当であり，本節は行わない．本節は一次データ
（`results/20260918_202613/results.jsonl`，`results/iter61/62/63/64_multilabel_ranking_predictions.jsonl`
の各全 1600 行，`results/iter64_stats.json`，`data/classifier_train_multidomain_iter61/64.jsonl`）を
直接集計した独立検算に基づく解釈である．
なお，オーケストレータからは本節を「### 考察 (Iter64)」として追記するよう指示されたが，
Iter61〜63 の先例では「### 分析（解釈）」＝analyst，「### 考察」＝reflector という書き分けが
確立しており，本フェーズは analyst であるため先例側の見出し名を採用した（内容は指示どおり
分析・解釈に限定し，採否確定・次レバー決定は行っていない）．

**0. 独立検算（実験節の全実測値が再現した）**

`compound_domain_set_recall=0.5450`，`mean_dispatch=2.0`（全 1600 行 `len=2`，rank_1==rank_2 重複 0），
S1（200 ペア，改善 55／悪化 15／discordant 70，exact p=1.6525e-06），S4（対 Iter63 不一致 987 行），
N2'（`selected_domain==dispatched_domains[0]` が全 1600 行で成立することを確認のうえ 0.599375），
N3（legal 16/30），N5（887/1500=0.591333），N6'（education 6/20・medical 20/28），
S5-a（複合 100 行 rank_1 正解 72・うち rank_2 も正解 24＝0.3333），
S5-b（Iter63 のみ被覆 6／Iter64 のみ被覆 13／discordant 19）が小数以下まで一致した．

S5-b の p 値について 1 点補足する．実験節の **p=0.16867 は
`metrics._mcnemar_from_correctness()` の連続性補正 chi2 版**であり，計画節の文言
「exact McNemar」に厳密に対応する `scipy.stats.binomtest(13, 19, 0.5)` は **p=0.16707** である．
**どちらも α=0.05 を大きく上回るため判定は変わらない**（S5-b は不成立）．事後の検定手法の
選び直しは行っていない．

複合 100 行上の時系列（同一 100 行で再集計）:

| 指標 | 基準線 | Iter61 | Iter63 | **Iter64** |
|---|---|---|---|---|
| rank_1 正解 | 41 | 41 | 61 | **72** |
| rank_2 正解（単独） | 28 | 48 | 37 | **37** |
| 2 ドメインとも被覆 | 3 | 12 | 17 | **24** |
| 被覆 1 個／0 個 | 63／34 | 65／23 | 64／19 | **61／15** |
| S5-a（rank_1 正解行での rank_2 正解割合） | 3/41=0.073 | 12/41=0.293 | 17/61=0.279 | **24/72=0.333** |
| `compound_domain_set_recall` | 0.345 | 0.445 | 0.490 | **0.545** |

（注: 計画節が基準線の「2 ドメイン被覆 12/100」と書いているのは Iter61 の値の誤記であり，
真の基準線は 3/100 である．S5-b の事前登録比較先は Iter63 の 17 行なので判定には影響しない．）

**1. 事前登録の判定規則への機械的な当てはめ（第一次判定）**

| 条件 | 事前登録の基準 | 実測 | 照合 |
|---|---|---|---|
| S1 | 200 ペア exact McNemar p<0.05 | p=1.6525e-06（改善 55／悪化 15） | 満たす |
| S2 | `compound_domain_set_recall` ≧0.510 | 0.545 | 満たす |
| S3 | mean_dispatch=2.000000 | 2.000000（重複 0） | 満たす |
| S4 | 対 Iter63 不一致 >0 | 987 行 | 満たす |
| **S5-a** | rank_1 正解行での rank_2 正解割合 >0.2787 | **0.3333** | 満たす |
| **S5-b** | 被覆 2 個行の対 Iter63 McNemar p<0.05 **かつ** 被覆行数 >17 | p=0.169（cc）／0.167（exact），被覆 24>17 | **満たさない**（AND の p 側が未達） |
| N2' | 上書き後 top1_accuracy ≧0.5875 | 0.599375 | 満たす |
| N3 | legal 自己被覆 ≧8/30 | 16/30 | 満たす |
| N5 | 単一 1500 行 argmax ≧0.590 | 0.591333 | 満たす |
| N6' | education ≧3/20 **かつ** medical ≧18/28 | 6/20・20/28 | 満たす |

参考（判定に用いない）: N1 `mismatch_count=435`（`pass:false` 固定），統計スクリプト自身の N2
`exact_match:false`（数値 `new_top1_accuracy` のみ N2' に転記）．判定不能だった項目はない．

計画節「判定規則（事前登録）」の各分岐への照合:

- 第 1 分岐（S1〜S4 かつ S5-a・S5-b 全充足 かつ N2'/N3/N5/N6' 全充足 → **adopted**）:
  **該当しない**（S5-b の p 条件が未達）．
- 第 2 分岐（**S5-a 充足・S5-b 不成立（p≧0.05）だが被覆行数 >17 かつ S2 充足 → partial**）:
  S5-a=0.3333>0.2787 充足，S5-b の p=0.169≧0.05 で不成立，被覆行数 24>17，S2=0.545≧0.510 充足
  → **前提が 4 つとも成立し，本分岐に該当する**．
- 第 3 分岐（S5-a 不成立だが S2 充足 → partial）: 前提（S5-a 不成立）が成立せず該当しない．
- 第 4 分岐（S2 不成立かつ recall≧0.490 → partial）: S2 充足のため該当しない．
- 第 5 分岐（recall<0.490 → rejected）: 0.545 のため該当しない．
- 第 6 分岐（S4 不成立 → no-op/rejected）: 987 行のため該当しない．
- 最終分岐（非退行のいずれか FAIL → 最大 partial）: **非退行 4 件は全て充足**のため発動しない．

**機械的な第一次判定は「partial（第 2 分岐．効果量はあるが検出力不足の可能性を記録する）」である**．
事後の閾値緩和・厳格化・検定手法の差し替えは一切行っていない．
Iter63 の partial（S 側全充足・非退行 1 件 FAIL 型）とは型が異なり，
**今回は非退行が全て通ったうえで主基準の有意性条件 1 つだけが未達という型の partial** である．

**2. S5-b が不成立である機序（本フェーズの主眼）**

**2-1. discordant の内訳は改善方向に偏っているが，n=19 では 0.05 に届かない**

被覆 2 個の行集合の遷移（複合 100 行，対応あり）:

| | Iter64 被覆 2 | Iter64 非被覆 2 | 計 |
|---|---|---|---|
| **Iter63 被覆 2** | 11 | **6** | 17 |
| **Iter63 非被覆 2** | **13** | 70 | 83 |
| 計 | 24 | 76 | 100 |

discordant=19（改善 13／悪化 6，改善比 0.684）．exact 両側二項検定で n=19 が p<0.05 となるには
**多数側が 15 以上（比 0.79）必要**であり，13 では届かない．
**この比 0.684 を維持したまま有意になるには discordant≧30 が必要**で，今回の
discordant 発生率（19/100 行）から逆算すると **複合行が約 158 行必要**である．
評価集合の複合行は 100 行に固定されているため，**本設計では S5-b は「効果量が中規模のとき
構造的に検出できない」条件**であった．

**2-2. 真に変化がないわけではない（参照点を変えると同じ指標が有意になる）**

同一の「被覆 2 個」指標を別の参照点に対して検定すると:

| 比較 | 悪化／改善 | exact p |
|---|---|---|
| 基準線（3/100）→ Iter64（24/100） | 2／23 | **1.94e-05** |
| Iter61（12/100）→ Iter64（24/100） | 4／16 | **0.0118** |
| Iter63（17/100）→ Iter64（24/100） | 6／13 | 0.167 |

すなわち **S5-b の不成立は「Iter64 が Iter63 に対して 1 反復ぶんの増分しか持たない」ことの帰結**であり，
指標そのものが動いていないのではない．事前登録が参照点を Iter63（直前反復）に取った以上，
1 反復ぶんの増分を n=100 行で有意にすることを要求する厳しい条件であったと解釈するのが妥当である．
**ただしこれは事後の言い訳にしてはならず，「検出力不足の可能性を記録する」という
第 2 分岐の文言どおりに扱う**（閾値も参照点も変更していない）．

**2-3. 被覆 2 個が増えた 13 行の由来（何が改善したのか）**

新規に被覆 2 個になった 13 行を Iter63 時点の状態で分解すると:

- **rank_1 は既に正解で rank_2 が新たに当たった行: 8 行**（＝本レバーが直接狙ったボトルネック）
- rank_2 は既に正解で rank_1 が新たに当たった行: 4 行
- 両方が新たに当たった行: 1 行

失われた 6 行は「rank_1 のみ残 5 行・両方喪失 1 行」で，rank_2 側の取りこぼしが主因である．
**13 行中 8 行が「rank_1 正解行で 2 つ目を当てた」由来であることは，S5-a の改善（0.279→0.333）が
rank_1 正解行数の増加（61→72）だけの見かけ上の産物ではないことを示す**．

**2-4. rank_1 正解と rank_2 正解の負の相関が弱まった（本レバーの最も明確な信号）**

複合 100 行での rank_1 正解・rank_2 正解のクロス集計（Fisher 正確検定）:

| 反復 | rank_1 正解 | rank_2 正解 | 両方 | 独立仮定の期待値 | Fisher p |
|---|---|---|---|---|---|
| 基準線 | 41 | 28 | 3 | 11.5 | 0.0001 |
| Iter61 | 41 | 48 | 12 | 19.7 | 0.0023 |
| Iter63 | 61 | 37 | 17 | 22.6 | **0.0211** |
| **Iter64** | **72** | **37** | **24** | 26.6 | **0.2538** |

これまでの構成では「rank_1 が当たる行ほど rank_2 が外れる」という**統計的に有意な負の関連**が
一貫して存在した（基準線 p=1e-4 〜 Iter63 p=0.021）．Iter64 ではこの負の関連が
**有意でなくなり（p=0.254），観測値 24 が独立期待値 26.6 にほぼ並んだ**．
一方 **rank_2 単独の正解行数は 37→37 と完全に横ばい**（内訳は入れ替わり 20/20，p=1.000）である．
すなわち本レバーが動かしたのは「rank_2 の絶対的な当たりやすさ」ではなく，
**「rank_1 が当たった行でも 2 つ目を当てられる」という同時性（＝2 ドメイン性）** の方であり，
これは計画節の仮説が名指ししていた対象そのものである．S5-b の p 値だけでは見えないこの構造変化が，
「量では 2 ドメイン性を学べない」と結論するのを妨げる最大の根拠である．

**2-5. サンプルサイズ起因か真に変化がないかの判定**

以上から，**S5-b の不成立はサンプルサイズ（複合 100 行・discordant 19）に起因する検出力不足で
説明でき，「真に変化がない」ことの証拠ではない**と判定する．根拠は (a) discordant が改善方向に
13:6 と偏り 95% 区間が 0.5 をまたぐ程度であること，(b) 同一指標が基準線比・Iter61 比では
有意であること，(c) 2-3・2-4 の分解が「狙った経路（rank_1 正解行での rank_2 正解）」由来の
改善を示すこと，の 3 点である．
**ただし「効果があった」と断定することもできない**（p=0.167 は Iter64 単独では両立する仮説を
絞り込めていない）．確信度は中程度であり，**同一レバーでの追加反復（例: 18 件／ペアへの用量反応）で
方向の一貫性を確認する価値がある**．

**3. 非退行条件が全て通ったことの意味**

**3-1. N6' は Iter63 の不成立から改善している（本レバーが Iter63 の学びを裏づけた）**

education 自己被覆の時系列は **基準線 9/20 → Iter60 4 → Iter61 5 → Iter62 6 → Iter63 3 → Iter64 6**，
medical は **13 → 19 → 12 → 11 → 18 → 20** である．Iter63 で不成立だった N6（education が
Iter62 の 6/20 を割って 3/20）は，Iter64 で **6/20 に戻り N6' の下限 3/20 を上回った**．

Iter63 の考察 3 は「**順位付け層をどう変えても education 被覆は 3/20 から動かせない．改善余地は
表現または訓練データ側にしかない**」と結論していた．Iter64 は訓練データ側のレバーであり，
探索的診断値がこの予測と整合する:

| 指標（education 絡み複合 20 行） | Iter63 | Iter64 |
|---|---|---|
| 自ドメイン得点／行内最大 の中央値 | 0.0899 | **0.3731** |
| ヘッド内中央順位 | 4.5 位 | 5.0 位 |
| ヘッド 1 位の行数 | 3 | 4 |
| **ヘッド 2 位の行数** | **0** | **2** |
| top-2 内（＝被覆）| 3/20 | **6/20** |

**Iter63 で「20 行中 1 行もヘッド 2 位に来ない」と実測された構造が，一律増量で 2 行に動いた**．
自得点比の中央値も 0.0899→0.3731 と 4 倍余りに上がっている．
**これは探索的診断値であり主基準ではない**（n=20，対 Iter63 の対応あり検定は悪化 0／改善 3，
p=0.25 で有意ではない）が，Iter63 の考察 3 が示した方向が正しかったことの弱い裏づけである．
medical も 18→20（悪化 0／改善 2，p=0.50）で，基準線比では 13→20（悪化 1／改善 8，**p=0.0391**）と有意．

**3-2. 留保 R-C（education が基準線を下回る）は格下げできる**

education 自己被覆の対応あり exact McNemar:

| 比較 | 悪化／改善 | p |
|---|---|---|
| 基準線 9/20 → Iter63 3/20 | 6／0 | **0.0312**（Iter63 で有意な退行） |
| **基準線 9/20 → Iter64 6/20** | **5／2** | **0.4531（有意でない）** |
| Iter62 6/20 → Iter64 6/20 | 2／2 | 1.000 |
| Iter63 3/20 → Iter64 6/20 | 0／3 | 0.250 |

**Iter63 で初めて有意になった R-C は，Iter64 では有意でなくなった**（点推定では依然 6<9 で
基準線を下回るため留保自体は残るが，「有意な退行」という記述は Iter64 構成には当てはまらない）．
reflector は R-C の格付けを Iter63 の「有意な退行」から「点推定で下回るが有意でない」へ
戻すことを検討する材料としてよい（格付けの確定は reflector の判断である）．

**3-3. N5 は通ったが余裕が薄い（次反復への申し送り）**

N5＝単一ドメイン 1500 行の argmax 正解率は **Iter63 0.603333（905/1500）→ Iter64 0.591333（887/1500）**
で **-1.2pt**．対応あり検定では悪化 105／改善 87，**p=0.2198 で有意ではない**（ノイズ範囲）．
しかし **事前登録した下限 0.590 に対する余裕はわずか 0.13pt＝2 行分**である．
複合行の性能（rank_1 正解 61→72）と単一行の性能（905→887）が逆方向に動いており，
**2 ドメイン合成事例の増量が単一ドメイン判別をわずかに希釈している可能性**がある．
現時点では有意ではないので「機序が確定した」とは言えないが，
**更なる増量（18 件／ペア）を検討する場合，N5 は最初に割れる指標である**ことを申し送る．

N2'（top1_accuracy）は 0.599375 で基準線 0.5975 比 +0.19pt（悪化 128／改善 131，p=0.901＝実質不変），
N3（legal 16/30）は Iter63 の 17/30 から -1 行だが下限 8/30 に対して大きな余裕がある．

**4. ノイズか有意かの判定（総括）**

| 指標 | 変化 | 判定 |
|---|---|---|
| S1（200 ペア被覆，対基準線） | 69→109/200（+20.0pt，95%CI [+12.3, +27.7]，改善 55／悪化 15，p=1.65e-06） | **明確な信号** |
| 同上（Iter63→Iter64 の増分） | 98→109/200（+5.5pt，95%CI [-0.1, +11.1]，改善 22／悪化 11，p=0.0801） | 方向は一貫するが**有意でない** |
| legal 絡み 60 ペアを除く n=140 | 49→75（+18.6pt，改善 37／悪化 11，**p=0.000222**） | **信号**（R-B 解消の継続確認） |
| S5-b（被覆 2 個，対 Iter63） | 17→24（改善 13／悪化 6，p=0.167） | **検出力不足．判定不能** |
| S5-a（条件付き割合） | 0.279→0.333（Wilson95%CI [0.235, 0.448] vs [0.182, 0.402]，CI は大きく重なる） | 点推定は上回るが**単独では有意でない** |
| rank_1 と rank_2 の負の関連 | Fisher p 0.0211→0.2538（負の関連の消失） | **構造変化の弱い信号** |
| N5 | 0.6033→0.5913（p=0.220） | **ノイズ範囲**（ただし下限まで 2 行） |
| education 被覆 | 3/20→6/20（p=0.250） | **ノイズ範囲**（診断値の 0.0899→0.3731 は大きい） |
| medical 被覆 | 18/28→20/28（p=0.500） | **ノイズ範囲** |

**recall の絶対水準 0.545 は，Iter59 以降で初めて 0.5 を超えた値**である
（0.345→0.480(Iter60，R-A リーク込み)→0.445→0.445→0.490→**0.545**）．
Iter63 までのリーク非依存構成での最高値 0.490 からの増分 +5.5pt は 200 ペア McNemar で p=0.0801 と
有意ではないが，**対基準線の全体効果（+20.0pt）は Iter61 の +10.0pt・Iter63 の +14.5pt から
単調に拡大しており，方向は 3 反復にわたって一貫している**．

**5. 仮説との整合**

計画節の仮説「2 ドメイン合成訓練事例を 135→405 行に増やせば，rank_1 正解行での rank_2 正解割合
（27.9%）と複合 100 行の集合再現率（0.490）が向上する」は，**集合再現率については明確に支持され
（0.490→0.545，対基準線 p=1.65e-06），条件付き割合については点推定で支持されたが有意性は示せなかった
（0.279→0.333，S5-b p=0.167）**．

仮説が想定していなかった観測は 3 点である．

- (a) **rank_2 単独の正解行数は 37→37 で完全に横ばい**であり，改善は「rank_2 が当たりやすくなった」
  のではなく「rank_1 正解行と rank_2 正解行の重なりが増えた」形で現れた（2-4）．
  これは「2 ドメイン性を学ぶ」という仮説の記述としてはむしろ素直な現れ方だが，
  計画節は rank_2 の絶対性能の向上を暗黙に想定していた．
- (b) **rank_1 正解自体が 61→72 と +11 行増えた**（計画節は rank_1 の変化を主基準に置いていない）．
  対 Iter63 で改善 20／悪化 9（p=0.0614）．合成事例の増量が rank_1（argmax）にも波及している．
- (c) **単一ドメイン側が -1.2pt 低下**（3-3）．

想定外の障害（言語崩れ・発散・OOM・アサーション違反）は発生していない．
A0（405 行全て多ラベル・45/45 ペア被覆）・A7（max_jaccard=0.2344 < 0.9）・A8'（405≧360）・
A9（全 1600 行で rank_1==argmax）はいずれも通っている．

**6. 探索的診断値：増量でノイズも増えたか**

計画節が「量を 3 倍にすればノイズも 3 倍になる」として記録を求めた低品質行（プロンプト文言の echo）
の混入を，Iter61 と同一の語句パターンで数えた:

| | 行数 | ペア数 | 件数分布 | 平均文字数 | 重複文 | echo 疑い |
|---|---|---|---|---|---|---|
| Iter61 | 135 | 45 | 全ペア 3 件 | 68.7 | 0 | 1（**0.74%**） |
| **Iter64** | **405** | **45** | **全ペア 9 件** | 70.8 | 0 | 5（**1.23%**） |

**混入率は 0.74%→1.23% とほぼ横ばい**（絶対数は 1→5 だが，率の差は n=135/405 では区別できない）．
「45 ペア一律 9 件」は完全に実現されており（件数分布が全ペア 9 件で単一），
**増量によるノイズの相対的増加は観測されなかった**．したがって
「S5-b が有意にならなかったのは低品質行の増加で相殺されたため」という説明は，
この診断値からは支持されない．

**7. 対外記述で併記すべき留保（reflector への材料）**

- **効果量**: 0.345 → 0.545（Δ=+20.0pt，200 ペア exact McNemar p=1.65e-06，95%CI [+12.3pt, +27.7pt]，
  mean_dispatch=2.000000 でコスト中立）．legal 絡み 60 ペアを除く n=140 でも +18.6pt・p=0.000222 で
  **R-B（効果の legal 依存）は Iter63 に続き解消状態を維持**．
  ただし **Iter63（0.490）からの増分 +5.5pt 単独は有意ではない（p=0.0801）**．
- **R-C（education）**: 点推定では基準線 9/20 → 6/20 で下回るが，**Iter63 で有意だった退行は
  Iter64 では有意でない（p=0.4531）**．「有意な退行」という Iter63 の記述をそのまま流用しないこと．
- **R-E（アンサンブル多様性の喪失）**: Iter63 と同じ構成（rank_1・rank_2 の双方を単一ヘッドが決める）
  のため継続する．
- **R-F（実機未検証．最も重い）**: 本イテレーションでも `node.py` への配線は行っていない（6 回目）．
  オフラインの +20.0pt は実機性能の主張に一切使えない．
- **新規の watch 項目（N5 の余裕）**: N5 は 0.5913 で下限 0.590 まで 2 行分しかない（3-3）．
- **S5-b の検出力**: 複合 100 行という評価集合では，改善比 0.68 程度の効果は原理的に
  p<0.05 に到達しない（discordant≧30，複合行 158 行相当が必要．2-1）．
  **同型の事前登録条件を今後も使うなら，この検出力限界を計画段階で明記すべきである**．
- **既知の制約（継続）**: temperature=0.8 の再生成であるため「量」の効果と「別の 405 文である」ことは
  分離できない（計画節の申し送りどおり）．低品質行の混入（1.23%）も Iter60 以来未解消．

**8. 次フェーズ（rc-reflector）への示唆**

- **分類は partial（判定規則の第 2 分岐に機械的に該当）**．非退行は 4 件とも充足であり，
  Iter63 の partial（非退行 FAIL 型）とは型が異なる点を区別して記録すること．
- **確信度は中程度**．S5-b の不成立は検出力不足で説明でき，2-3・2-4 は狙った経路での改善を示すが，
  「量で 2 ドメイン性を学べる」と断定するには 1 反復では足りない．
  **同一レバーの用量反応（例: 18 件／ペア）による追加反復で方向の一貫性を確認する選択肢が，
  分析上は最も情報量が大きい**（レバー収束・棄却のいずれよりも）．その際は N5（3-3）を
  最初に割れる指標として注視すること．
- 一方，**更なる増量を選ばない場合の根拠**も分析上は存在する: Iter63→Iter64 の増分は
  200 ペアで p=0.0801，S5-b で p=0.167 と，いずれも「もう 1 反復で有意になるか」が
  不確実な水準にあり，計画節「不成立の場合」が挙げた (i) 生成文の品質改善へ移る判断も
  同じデータと矛盾しない．どちらを採るかは reflector の判断である．

### 考察 (Iter64)

**1. 判定の確定: partial（第 2 分岐）．analyst の第一次分類に同意する**

事前登録の判定規則（本イテレーションの「計画 (Iter64)」節）を実測値へ機械的に当てはめた結果は
次のとおりで，analyst の第一次分類（partial・第 2 分岐）と一致する．

- 第 1 分岐（S1〜S4 かつ **S5-a・S5-b 全充足** かつ非退行全充足 → adopted）: **該当しない**．
  S5-b は「p<0.05 **かつ** 被覆行数 >17」という AND 条件であり，被覆行数 24>17 は満たすが
  p=0.167（exact binomtest）／0.169（連続性補正 chi2）で p 側が未達である．
- 第 2 分岐（**S5-a 充足・S5-b 不成立（p≧0.05）だが被覆行数 >17 かつ S2 充足 → partial**）:
  S5-a=24/72=0.3333 > 0.2787，S5-b の p=0.167≧0.05，被覆行数 24>17，S2=0.545≧0.510 と
  **前提 4 つがすべて成立し，本分岐に文言どおり該当する**．
- 第 3〜6 分岐（S5-a 不成立／S2 不成立／recall<0.490 で rejected／S4 不成立で no-op）:
  いずれも前提が成立せず該当しない．
- 最終分岐（非退行のいずれかが FAIL → 最大 partial）: N2' 0.599375・N3 16/30・N5 0.591333・
  N6' 6/20 と 20/28 で**非退行 4 件すべて充足**のため発動しない．

**したがって `multilabel_synthetic_volume = uniform_nine_per_pair` は partial（部分的成立．
確信度 中）として確定する．** values は単一値のため本レバーは**クローズ（試し切り）**である．

analyst の機序分析（S5-b の不成立はサンプルサイズ起因の検出力不足で説明でき，基準線比 p=1.94e-05・
Iter61 比 p=0.0118 では同一指標が有意，改善 13 行のうち 8 行が狙った経路＝「rank_1 正解行で
2 つ目を当てた」由来，rank_1 正解と rank_2 正解の負の関連が Fisher p 0.0211→0.2538 へ消失）は
一次データの独立検算に基づいており妥当と判断する．**ただしこれを理由に adopted へ格上げすることは
しない**．「検出力不足だったから有意性条件を免除する」という事後の判断は，Iter29 以降の事前登録運用
（B94〜B97 で一貫して守ってきた基準）を壊す．第 2 分岐の文言が「効果量はあるが検出力不足の可能性を
記録する」と定めているとおり，**partial のまま，検出力の限界を記録に残す**のが規則に忠実な扱いである．
逆に，S5-b が不成立であることを理由に rejected へ落とすことも規則にない．

Iter62 の partial（中身が空＝効果量 0）・Iter63 の partial（S 側全充足・非退行 1 件 FAIL）とは型が
異なり，**Iter64 は「非退行が全通したうえで，主基準の 2 副条件のうち有意性条件 1 つだけが
検出力の壁で未達」という型の partial** である．この区別を記録する．

**2. 効果量の対外記述（正式値の更新）**

効果量の正式値を **`compound_domain_set_recall` 0.345 → 0.545（Δ=+20.0pt，ドメイン対 n=200 の
exact McNemar p=1.65e-06，95% CI [+12.3pt, +27.7pt]，mean_dispatch=2.000000 でコスト中立）** へ
更新する（B97 の +14.5pt を置き換える）．**partial である旨**と，**rank_1・rank_2 の双方を
オフラインのヘッドが決める構成の値である旨（R-F）**を必ず併記する．
**B95 の正式値（Iter61 の +10.0pt＝rank_1 を実行時ルータのまま保った構成）は 2 本立ての片方として
維持する**（配線可否が未決である以上，両構成の値が必要なため削除しない）．

書いてよいこと・書いてはならないことを明示する．

- 書いてよい: 「Iter63 までのリーク非依存構成での最高値 0.490 から 0.545 へ上がり，対基準線の
  効果量は Iter61 +10.0pt → Iter63 +14.5pt → Iter64 +20.0pt と 3 反復にわたり単調に拡大した」．
- **書いてはならない**: 「Iter63 からの増分 +5.5pt が有意」（200 ペア McNemar で **p=0.0801**，有意でない）．
  「2 ドメイン性が訓練信号の量で学習できることを示した」（S5-b が未達であり，断定できない）．
  「量を増やせば rank_2 が当たりやすくなる」（rank_2 単独の正解行数は 37→37 で完全に横ばい）．

留保の更新は次のとおり．

- **R-B（効果の legal 依存）**: 解消状態を維持．legal 絡み 60 ペアを除く n=140 でも +18.6pt・
  exact p=0.000222．
- **R-C（education の被覆が基準線を下回る）**: **B97 の「基準線比で有意な退行」という格付けを
  取り下げ，「点推定では基準線 9/20 を下回る（6/20）が，対応あり exact McNemar では有意でない
  （p=0.4531）」へ戻す**．格付けを戻す根拠は事後の基準緩和ではなく，**構成が変わった（Iter64 の
  訓練データ増量後のヘッド）ことによる実測値の変化**である．Iter63 の構成に対する
  「p=0.0312 の有意な退行」という記述は Iter63 構成の事実として残り，Iter64 構成には流用しない．
- **R-E（アンサンブル多様性の喪失）**: 継続（rank_1・rank_2 の双方を単一ヘッドが決める構成のため）．
- **R-F（実機未検証．最も重い）**: 継続．オフラインの +20.0pt は実機性能の主張に一切使えない．
- **R-G（新規・N5 の余裕が薄い）**: 単一ドメイン 1500 行の argmax 正解率は 0.6033→0.5913 で，
  事前登録の下限 0.590 まで **2 行分（0.13pt）** しかない．変化自体は p=0.2198 で有意でないが，
  複合行の性能と単一行の性能が逆方向に動いており，増量が単一ドメイン判別を希釈している可能性を
  否定できない．以後の増量系レバーでは**最初に割れる指標**として注視する．
- **R-H（新規・S5-b 型条件の検出力限界）**: 複合 100 行という評価集合では，改善比 0.68 程度の
  効果は原理的に p<0.05 に到達しない（discordant≧30，複合行 158 行相当が必要）．
  **同型の事前登録条件を今後使う場合は，計画段階でこの検出力限界と，検出力のある参照点
  （基準線または 2 反復前）を併せて事前登録すること．**
- **既知の制約（継続）**: temperature=0.8 の再生成のため「量」の効果と「別の 405 文である」ことは
  分離できない．低品質行（プロンプトの echo）の混入は 0.74%（135 行）→1.23%（405 行）で
  Iter60 以来未解消（ただし率としては横ばいで，**増量がノイズを相対的に増やした証拠はない**）．

**3. 配線（本番経路への接続）: 今回も行わない（5 回目．Iter60・61・62・63・64）**

理由は B95〜B97 と同じく，配線が `config.yaml` のスキーマ変更（＋`node.py:214`・
`run_experiment.py:93` の同時変更）を伴い，rc-reflector の自律判断（可逆な判断に限る）の範囲外で
あることである．論点は **B95 要レビュー 1 に一本化したまま維持**し，本イテレーションで累積させない．

新たな判断材料として次の 2 点を記録する（配線の是非そのものは人間判断）．

- (a) B97 で配線を**推奨しない**根拠としていた「R-C が基準線比で有意に退行する」という事実は，
  Iter64 構成では成立しない（p=0.4531）．**配線に対する反対材料が 1 つ減った**．
- (b) 一方で R-F・R-E は不変であり，オフライン構成と実行時経路の乖離は最大のままである．
  加えて R-G（N5 の余裕 2 行）が新たに加わったため，配線するならば**実機本走で
  単一ドメイン 1500 問の精度を同時に確認すること**が条件になる．

**4. 次の一手: 同一変数の用量反応（9 → 18 件／ペア）へ進む**

analyst が示した 2 択（(i) 同一レバーの用量反応，(ii) 生成文の品質改善）のうち **(i) を採る**．
根拠は次の 4 点である．

1. **(ii) の前提が今回のデータで支持されなかった**．品質改善は「増量でノイズも増えて効果が
   相殺された」という説明を前提とする候補だったが，echo 混入率は 0.74%→1.23% と率としては横ばいで，
   絶対数 5 行／405 行である．この水準のノイズが S5-b の p=0.167 を作っているとは考えにくく，
   品質改善に 1 イテレーションを投じても動く指標が予測できない．
2. **用量反応は現時点で最も情報量が大きい**．135 行（Iter61）・405 行（Iter64）の 2 点に 810 行を
   加えれば，「量」仮説は**単調な用量反応が出るか否か**という 3 点の形で検証でき，
   1 反復ぶんの増分を n=100 行で有意にするという S5-b の構造的な難しさ（R-H）を，
   **検出力のある参照点（135 行構成・基準線）との比較**へ置き換えられる．
3. **反証可能性が高い**．810 行でも被覆 2 個行が頭打ちになる（あるいは N5 が下限 0.590 を割る）なら，
   「量では 2 ドメイン性を学べない／量は単一ドメイン判別とトレードオフである」と**明確に結論でき**，
   計画節「不成立の場合」の (i)(ii)(iii) へ根拠をもって移れる．どちらに転んでも学びが確定する．
4. **コストが小さい**．Iter64 の実測は生成 7 分＋再訓練 14 分＋採点 2 分＝約 25 分であり，
   倍量でも 1 時間以内に収まる見込みである（wafl-ctrl5 の RTX 3060 12GB は 8B q4 モデル 1 つで
   VRAM 5702/12288 MiB，逐次実行のため VRAM 制約には当たらない）．オフライン完結・スキーマ変更なし．

次イテレーションで事前登録すべき点（計画フェーズへの申し送り．**閾値は計画フェーズで確定し，
実験後に緩めない**）:

- **主基準は「対 Iter64 の 1 反復増分」だけに置かないこと**（R-H）．135 行構成（Iter61）または
  基準線を参照点に含め，検出力のある比較を主基準側に必ず 1 つ入れる．
  加えて 135/405/810 の 3 点での**単調性（傾向検定）**を事前登録するとよい．
- **N5 の下限は 0.590 のまま据え置く**（R-G）．余裕が 2 行しかないことを理由に緩めてはならない．
  ここが割れた場合は「量は単一ドメイン判別を犠牲にする」という機序の確定として扱う．
- N2'・N3・N6' は Iter64 の実測水準を下限とするか Iter63 水準を維持するかを計画フェーズで決め，
  一度決めたら動かさない．
- 実行基盤は **wafl-ctrl5 に一本化**する（config.yml 冒頭の恒久ルール）．

**5. 環境・運用上の学び（次の自分への申し送り）**

- **実行基盤を wafl-ctrl5 へ移したことで実行時間が計画見積りの 1/3 になった**（見積り 70〜90 分 →
  実測 25 分）．律速だった LLM 生成が 135 行 21 分（wafl500 経由，Iter61）→405 行 7 分へと，
  行数 3 倍にもかかわらず短縮された．要因は専有 GPU で他プロセスと競合しないことと考えられる．
  **今後のコスト見積りは Iter61 の実績ではなく Iter64 の実績（405 行＝生成 7 分，
  1427+405 行の埋め込み再訓練＝14 分）を基準にすること．**
- **`compute_iter59_ranking_stats.py` のコメント（195・214 行）が `head_argmax` モードの実挙動と
  食い違っている**（「`selected_domain` は基準線から verbatim にコピーされる」と書かれているが，
  実際は新 rank_1 で上書きされる）．今回は実データで挙動を検証して N2' の値を採用したが，
  **これは単一レバー原則のため未修正のまま残した別件のバグ（コメントの誤り）である**．
  次に同スクリプトへ触るイテレーションで是正すること（backlog B98 の要レビューに記載）．
- 計画節に「基準線の 2 ドメイン被覆 12/100」と書いたのは Iter61 の値の誤記で，真の基準線は 3/100
  だった（analyst が検出）．**参照点の数値は計画フェーズで一次データから再計算して書くこと．**

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

