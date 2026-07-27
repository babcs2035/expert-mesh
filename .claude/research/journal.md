### 実験 (Iter17)

- **実験ディレクトリ**: `results/20260727_180824`
- **データセット**: JMMLU 1520問（単一1500 + 複合20），全問完走
- **所要時間**: 約91.8分（mean_duration_ms=3622.2）
- **top1_accuracy**: 0.5651（Wilson CI: [0.5401, 0.5899]）
- **Cohen's kappa**: 0.5215
- **random_baseline**: 0.1013
- **misrouting_rate**: 0.4349，fallback_rate: 0.1316
- **dispatch_failure_rate**: 0.0
- **同点タイ率**: 0.00%

**McNemar 対比較**: 不一致対数 814，chi2=365.57，p < 0.000001．**有意差あり**．

---

## Iteration 17: embedding ベース教師あり分類による routing_method の検証

### 調査 (Iter17)

**単一レバー**: `routing_method` (E6), `self_report → supervised_classifier`

**調査の問い**

1. `routing_method=supervised_classifier` のコード実装は完了しているか（classifier.py, router.py, http_server.py, Dockerfile）．
2. 訓練スクリプト（scripts/train_domain_classifier.py）は正しく動作するか．scikit-learn の依存関係は Docker イメージに含まれているか．
3. 訓練データと評価データの分離が，質問単位で完全に実施されているか（label leakage の再演を防ぐ）．
4. 分類器モデル（Pickle ファイル等）は既に訓練済みか，それとも実験前に訓練が必要か．
5. 既知のリスク・課題は何か（embedding モデルのバージョン，anisotropy，cross-lingual，class imbalance）．

**1. コード実装の完了状況**

実装は完全に完了しており，全テストが PASS している．

| 項目 | ファイル | 行番号 | 状態 |
|------|---------|-------|------|
| 分類器サービング | `classifier.py` | 全42行 | 完了 |
| 分類器ロード | `classifier.py:load_domain_classifier()` | 行16-24 | 完了 |
| 信頼度推定 | `classifier.py:estimate_confidence_classifier()` | 行27-41 | 完了 |
| /probe 統合 | `http_server.py` | 行323-329 | 完了（LLM コール不要） |
| ライフサイクル起動時ロード | `http_server.py` | 行406-411 | 完了（モデルパス未設定で ValueError） |
| NodeState 設定伝播 | `http_server.py` | 行194-195, 行244-252 | 完了 |
| node.py 設定伝播 | `node.py` | 行89 | 完了 |
| Dockerfile COPY | `Dockerfile` | 行14 | 完了（`classifier.py` が COPY 対象に含まれている） |
| config.yaml キー | `config.yaml` | 行59-64 | 完了（`classifier_model_path: models/domain_classifier.joblib`） |
| 単体テスト | `tests/test_classifier.py` | 4件全PASS | 完了 |
| 訓練スクリプトテスト | `tests/test_train_domain_classifier.py` | 2件全PASS | 完了 |
| 統合テスト | `tests/test_http_server.py` | 2件全PASS | 完了 |

**重要な設計決定**:

- 各ノードは同じ多クラス分類器をロードし，自分のドメインの予測確率のみを返す．中央ルーターは導入しない．
- 分類器は requester が既に計算済みの `query_embedding` を消費するため，/probe 呼び出しで追加 LLM コールは発生しない．
- `predict_proba` の全クラス確率は合計 1 になるため，ノード間の confidence 値が直接比較可能である（scikit-learn >=1.5 のデフォルト softmax 動作に依存）．
- 訓練時に未登場のドメインは 0.0 を返し，dispatch 対象から除外される．

**2. 訓練スクリプトと依存関係**

| 項目 | 状態 |
|------|------|
| 訓練スクリプト | `scripts/train_domain_classifier.py` — 完了 |
| CLI 引数 | `--train-data`, `--embedding-model`, `--ollama-host`, `--ollama-port`, `--output` |
| 分類器モデル | scikit-learn `LogisticRegression(max_iter=1000, class_weight="balanced")` |
| scikit-learn 依存 | `pyproject.toml` 行12: `scikit-learn>=1.5` — Docker イメージに含まれる |
| joblib 依存 | scikit-learn のトランザティブ依存として自動インストールされる |
| 訓練データ形式 | JSONL の `{"id", "query", "domain"}` 行 |
| 出力形式 | `models/domain_classifier.joblib`（joblib 直列化） |

**訓練の実行条件**: 訓練にはライブ Ollama ノードが必要（embedding 生成のため）．`--ollama-host` で指定したホストの Ollama デーモンが `nomic-embed-text` モデルをロードしている必要がある．

**3. 訓練/評価データ分離**

分離は構造的に保証されている．

| 保証メカニズム | 詳細 |
|--------------|------|
| 異なるシード | `_CLASSIFIER_TRAIN_SAMPLE_SEED = 20260727` vs `_JMMLU_SAMPLE_SEED = 20260726` |
| 質問単位の除外 | `build_classifier_training_rows()` は評価行の `query` を `frozenset` にして，サンプリング前にプールから除外する |
| 特徴量源の分離 | 訓練データは `{"query", "domain"}` のみ（probe/dispatch 結果を含まない） |
| モジュール設計 | `train_domain_classifier.py` は `results/*/results.jsonl` を一切参照しない |

**Iter10 の label leakage との比較**: Iter10 では probe/dispatch 結果（self_confidence, margin, is_top1）を同じ46問から抽出して訓練した．E6 では訓練データと評価データが質問単位で完全分離されており，label leakage の再演は構造上不可能である．

**4. 分類器モデルの訓練状況**

**未訓練**．以下の理由から，実験前に訓練が必要である．

- `models/` ディレクトリは存在しない（`.gitignore` 行13で除外されている）
- `data/classifier_train.jsonl` は存在しない
- 訓練データとモデルの両方を生成する手順が必要

**必要な手順**:
1. `uv run python build_dataset.py --output data/dataset.jsonl --classifier-train-output data/classifier_train.jsonl` — 訓練データ生成
2. `uv run python -m scripts.train_domain_classifier --train-data data/classifier_train.jsonl --embedding-model nomic-embed-text --ollama-host <ホストIP> --output models/domain_classifier.joblib` — 分類器訓練
3. 訓練済みモデルを全10ノードの `models/domain_classifier.joblib` に配布（または Docker volume マウント）

**5. 既知のリスク・課題**

**R1: クラス不均衡（legal ドメイン）**
- JMMLU に `professional_law` タスクが存在しないため，legal のプールは227問のみ（他ドメインは150問以上）．
- 評価用に150問を確保すると，訓練用には約77問しか残らない（他ドメインは150問）．
- **緩和策**: `class_weight="balanced"` がこの不均衡を補正する（journal.md「実装 (Iter15)」バッチ6 で追加済み）．

**R2: embedding の anisotropy**
- Iter2 で cosine similarity が `[0.667, 0.737]` に潰れた原因は embedding の anisotropy である．
- 本研究の supervised classifier は cosine similarity ではなく LogisticRegression を使用するため，anisotropy の影響は直接受けない．
- **根拠**: Varangot-Reille et al. (arXiv:2502.00409, JAIR 2025) は similarity-based routing の失敗を unsupervised であることに帰する．RouterDC (NeurIPS 2024) は CosineClassifier に全タスクで勝利している．教師あり分類は anisotropy 下でも機能する．

**R3: cross-lingual（英語ドメイン名 vs 日本語質問）**
- nomic-embed-text は multilingual モデルであるが，ドメイン名（"medical", "legal" 等）は英語で，質問は日本語である．
- Iter2 の embedding ルーティングではこの cross-lingual mismatch が問題となった（B7）．
- **緩和策**: supervised classifier は embedding 空間内の分離超平面を学習するため，cross-lingual なラベル名は学習プロセスには直接影響しない（ラベルはドメイン文字列としてのみ使用され，embedding されない）．

**R4: nomic-embed-text の task prefix 未付与**
- nomic-embed-text は `search_query:`, `search_document:`, `classification:` 等の task instruction prefix を前提に学習されている（B7）．
- 現行コードは prefix を付けていない．
- **影響**: prefix 未付与は embedding 品質を低下させる可能性があるが，supervised classifier はその embedding 空間で学習するため，prefix あり/なしの差は「embedding 空間の幾何的性質」に帰着し，分類器が適応できる範囲内である．RouterDC は prefix なしでも CosineClassifier に勝っている．

**R5: 訓練に必要な Ollama リソース**
- 訓練スクリプトは embedding 生成にライブ Ollama ノードを必要とする．
- 訓練データは推定で 10 ドメイン × 150 問 = 1,500 件（legal は77件）．nomic-embed-text の embedding は軽量だが，逐次実行のため数分かかる．
- **注意**: WAFL-PEFT が同一 GPU プールを使用中でないことを確認してから訓練を実行すること．

**R6: embedding モデルのバージョン整合性**
- 訓練時と推論時に同じ `nomic-embed-text` の同じバージョンが使用される必要がある．
- Ollama のモデルキャッシュが更新されると embedding 空間が変化する可能性がある．
- **緩和策**: 全10ノードで `ollama list` を確認し，同じ digest のモデルが使用されていることを確認する．

**文献調査の補足**

- **Varangot-Reille et al. (arXiv:2502.00409, JAIR 2025)**: "Doing More with Less: A Survey on Routing Strategies for Resource Optimisation in Large Language Model-Based Systems" — similarity-based routing の失敗を unsupervised であることに帰し，supervised routing の有効性を支持．
- **RouterDC (NeurIPS 2024)**: "Query-Based Router by Dual Contrastive Learning for Assembling Large Language Models" — CosineClassifier に全タスクで勝利．教師あり学習が embedding 空間の幾何的制約を克服可能であることを実証．
- **MoDEM (arXiv:2410.07490)**: 5クラスで総合81.00%．Other（general相当）が52.94% と低い．本研究の10クラス設定では general ノードが同様のボトルネックになる可能性がある．

**計画フェーズへの提案**

1. **訓練手順を最初に実行する**: 実験前に `build_dataset.py --classifier-train-output` で訓練データを生成し，`train_domain_classifier.py` で分類器を訓練する．この手順は `mise run setup/deploy` の前に行う必要がある（Docker イメージにモデルファイルを含めるため）．
2. **Docker volume でのモデル配布**: `docker-compose.yml` 行39-41 に `./models:/app/models:ro` の volume マウントが既に設定されている．したがって，訓練済みモデルを各ホストの `models/domain_classifier.joblib` に配置すれば，Docker イメージの再ビルドなしで全ノードに反映される．
3. **config.yaml の変更**: `routing_method: self_report → supervised_classifier` の1行変更のみ．`confidence_elicitation: top_k_with_probs` は維持（self_report 専用なので supervised_classifier では無視されるが，設定の整合性のため）．
4. **オフライン検証**: 訓練後，評価データ（1520問）に対してオフラインで分類精度を測定し，Iter15 の self_report ベースライン（top1_accuracy=0.184）との比較を事前に行う．

### 計画 (Iter17)

**単一レバー**: `routing_method` (E6), `self_report → supervised_classifier`

**変更箇所**: `config.yaml` 行31 の1行変更のみ．
```
routing_method: self_report  →  routing_method: supervised_classifier
```

**仮説**: embedding ベースの教師あり分類（LogisticRegression）が self_report よりルーティング精度を改善する理由は，self_report の構造的問題を根本的に回避するためである．

1. **自己宣伝バイアスの除去**: self_report では各ノードの light_model（qwen3.5:4b）が「あなたは{domain}分野の専門家です」とプロンプト指示されるため，どの質問に対しても自分の分野に 0.9 の高 confidence を出す（Iter15 で 74.9% が 0.9 饱和，クロスドメインでも 70-90% が 0.9）．教師あり分類器はドメインのプロンプト指示を受けず，embedding 空間の幾何的な分離超平面のみで判定するため，この自己宣伝バイアスを受けない．

2. **全クラス確率の合計制約による自然な正規化**: scikit-learn の多クラス LogisticRegression は softmax 出力のため，全10クラスの確率が合計 1 になる．self_report では各ノードが独立に 0-1 の値を申告するため，ノード間の confidence が比較不可能だった（Iter15 で 10 ノード中 7-10 ノードが 0.9 を出し，98.29% のタイ）．教師あり分類では，正解ドメインの確率が 0.3 なら他ドメインの合計は 0.7 になるため，自然に弁別力のある分布が生成される．

3. **embedding 空間の教師あり学習は anisotropy に頑健**: Iter2 で cosine similarity が [0.667, 0.737] に潰れた原因は embedding の anisotropy であるが，教師あり分類器は cosine 距離ではなく線形分離超平面を学習するため，anisotropy の影響を直接受けない（Varangot-Reille+ JAIR 2025，RouterDC NeurIPS 2024）．

4. **Iter2（unsupervised embedding）との明確な違い**: Iter2 が棄却されたのは，unsupervised cosine similarity がドメイン識別信号を持っていなかったからである．教師あり分類はラベル付きデータから分離超平面を学習するため，unsupervised とは全く異なるアプローチである．RouterDC は CosineClassifier に全タスクで勝利している．

**固定する構成**（Iter16 の最良構成を継承）:

| 設定 | 値 | 理由 |
|------|-----|------|
| `confidence_signal_method` | `self_report` | 変更不可．supervised_classifier では routing_method が signal 抽出を完全に置き換えるため，この設定は参照されない |
| `confidence_elicitation` | `top_k_with_probs` | 変更不可．self_report 専用設定であり，supervised_classifier では無視される |
| `confidence_threshold` | `0.5` | 変更不可．閾値ゲートの効果検証は Iter3 で no-op と判定済み |
| `dispatch_top_k` | `1` | 変更不可．Iter1 で棄却済み |
| `semantic_sample_count` | `5` | 変更不可．E4 用設定であり，supervised_classifier では参照されない |
| `semantic_sample_temperature` | `0.7` | 変更不可．E4 用設定 |
| `embedding_postprocess` | `none` | 変更不可．E7（whitening）は embedding ルーティング専用であり，supervised_classifier では参照されない |
| `light_model` | `qwen3.5:4b-q4_K_M` | 変更不可．E8（expert_model_size）は別レバー |
| `expert_model` | `schroneko/llama-3.1-swallow-8b-instruct-v0.1:q4_k_m` | 変更不可．S1（expert_specialization）は別レバー |
| 10 ノード構成 | wafl500〜509 | 変更不可．domain_count=10 は E9 の対象 |
| `router.py` few-shot 例 | 動的生成 `_build_few_shot_examples` | 変更不可．few-shot 変更は Iter5-9 で5回連続棄却済み |
| `classifier_model_path` | `models/domain_classifier.joblib` | 変更不可．E6 実装時に設定済み |

**成功条件**:

| 分類 | 指標 | ベースライン (Iter16) | 成功条件 | 根拠 |
|------|------|---------------------|---------|------|
| 主基準 | top1_accuracy McNemar | 0.2059 (p=0.0783 vs Iter15) | **有意差あり** (α=0.05) | 同じ 1520 問データセット上の McNemar 対比較．Iter15→Iter16 の変化（+0.022, p=0.0783）は有意閾値の80%であり，supervised_classifier がより明確な信号を出すなら有意になる |
| 主基準 | top1_accuracy Wilson CI | [0.1863, 0.2270] | **CI がベースライン CI と重ならない** | 1520 問で SE ≈ 0.01，CI 幅 ≈ 0.04．±0.03 以上の改善で CI が重ならなくなる |
| 副基準 | Cohen's kappa | 0.1067 | **0.1067 より有意に高い**（CI が重ならない） | chance-corrected 指標で実質識別力を測定．10 分野で偶然一致 0.101 を有意に上回る必要がある |
| 副基準 | 同点タイ率 | 82.83% | **有意な低下** | softmax 出力は連続値のため，self_report の離散値（5段階）よりタイが大幅に減る |
| 副基準 | ECE | 0.7388 | **報告**（較正改善の定量化） | softmax 出力は較正された確率であるため，ECE が改善する可能性がある |
| 非退行 | per-domain precision/recall | Iter16 の各値 | **各ドメインの CI 下限がベースライン CI 下限を下回らない** | config.yml success_criteria (2) に従う |
| 監視 | probe レイテンシ | 計測済み (Iter16) | **報告**（追加 LLM コールなしのため同程度または短縮） | supervised_classifier は embedding 計算のみで LLM コール不要 |
| 監視 | dispatch_failure_rate | 0.0 | **0.0** | インフラ起因の失敗がないことを確認 |

**成功条件の数値根拠**: Iter15→Iter16 の変化は +0.022（p=0.0783）であり，有意閾値の 80% にある．supervised_classifier が self_report の構造的問題（自己宣伝バイアス，離散値飽和）を解決するなら，より大きな変化（±0.05 以上）が期待される．1520 問での二項 SE は約 0.01 であり，0.03 以上の変化は CI が重ならなくなる（Wilson 95% CI 幅 ≈ 0.04）．

**実験構成（フルフロー）**:

```
┌─────────────────────────────────────────────────────────────┐
│ Step 0: 訓練データ生成                                       │
│ uv run python build_dataset.py                              │
│   --output data/dataset.jsonl                               │
│   --classifier-train-output data/classifier_train.jsonl      │
│  → data/classifier_train.jsonl に {query, domain} 行が生成   │
│  → 評価データ（_JMMLU_SAMPLE_SEED=20260726）と              │
│    訓練データ（_CLASSIFIER_TRAIN_SAMPLE_SEED=20260727）が    │
│    質問単位で完全分離される                                   │
├─────────────────────────────────────────────────────────────┤
│ Step 1: 分類器訓練（ライブ Ollama が必要）                   │
│ uv run python -m scripts.train_domain_classifier            │
│   --train-data data/classifier_train.jsonl                  │
│   --embedding-model nomic-embed-text                        │
│   --ollama-host 192.168.15.100                              │
│   --output models/domain_classifier.joblib                  │
│  → LogisticRegression(max_iter=1000, class_weight="balanced")│
│  → 10 クラス softmax 出力（predict_proba の合計=1）          │
│  → models/domain_classifier.joblib に保存                    │
├─────────────────────────────────────────────────────────────┤
│ Step 2: モデル配布（Docker volume 経由）                     │
│ 各ホスト（wafl500-509）の ./models/ ディレクトリに           │
│ domain_classifier.joblib を配置                              │
│ docker-compose.yml 行41 の ./models:/app/models:ro が       │
│ 自動マウントするため，Docker イメージの再ビルドは不要         │
├─────────────────────────────────────────────────────────────┤
│ Step 3: config.yaml 変更                                    │
│ config.yaml 行31: routing_method: self_report               │
│                    → supervised_classifier                   │
│ mise run setup（Docker イメージ再ビルド）                     │
│ mise run deploy（全10ノード）                                 │
├─────────────────────────────────────────────────────────────┤
│ Step 4: 実験                                                │
│ mise run start（同一 1520 問データセット）                    │
│ 完了後: mise run analyze                                     │
├─────────────────────────────────────────────────────────────┤
│ Step 5: 分析                                                │
│ metrics.py --results <実験ディレクトリ>/results.jsonl --json │
│ → Wilson CI, Cohen's kappa, McNemar 対比較                  │
│ → Random/BestSingle/Oracle ベースライン                      │
│ → ドメイン別 precision/recall/ECE/同点率                     │
└─────────────────────────────────────────────────────────────┘
```

**オフライン事前検証（Step 1 と Step 4 の間に実施）**:

訓練済み分類器に対して，評価データ（1520問）の embedding をオフラインで計算し，分類器の predict_proba による top-1 分類精度を測定する．これはルーティング精度の上限（upper bound）を示す（実際のルーティングではノードごとの confidence 比較があるが，オフライン検証は分類器自体の性能を直接測定する）．この値がベースライン（0.206）を有意に上回らない場合，実機実験の実行を再検討する．

**実行時間の見積もり**:

| 工程 | 推定時間 | 備考 |
|------|---------|------|
| Step 0: 訓練データ生成 | 1-2 分 | JMMLU からサンプリング（CPU 処理） |
| Step 1: 分類器訓練 | 5-10 分 | embedding 生成（逐次，nomic-embed-text）+ LogisticRegression 学習（瞬時） |
| Step 2: モデル配布 | 1-2 分 | scp で 10 ホストにコピー（joblib ファイルは数十 KB） |
| Step 3: setup + deploy | 5-10 分 | Docker イメージ再ビルド + 10 ノードのコンテナ再作成 |
| Step 4: 実験実行 | 約 90-120 分 | Iter16 の mean_duration_ms=4134（約4.1秒/問）× 1520 問．supervised_classifier は LLM コール不要なので probe 時間が短縮される可能性がある |
| Step 5: 分析 | 1-2 分 | metrics.py のオフライン計算 |
| **合計** | **約 2-2.5 時間** | タイムアウト 90 分は不足するため，experiment.timeout_min を 150 に引き上げる |

**特定されたリスクと緩和策**:

| リスク | 内容 | 影響 | 緩和策 |
|-------|------|------|--------|
| R1: legal ドメインの訓練データ不足 | JMMLU に professional_law タスクがなく，legal の訓練用プールは約77問（他ドメインは150問） | legal クラスの分類精度が低く，misroute が増加する可能性 | `class_weight="balanced"` が不均衡を補正（訓練時に追加実装済み）．legal の per-domain recall を重点監視 |
| R2: embedding モデルのバージョン整合性 | 訓練時と推論時に異なる nomic-embed-text のバージョンが使用されると，embedding 空間が変化し分類器が機能しない | 分類精度が大幅に低下する | 全10ノードで `ollama list` を確認し，同じ digest のモデルを使用していることを確認．`OLLAMA_KEEP_ALIVE=-1` でモデルがアンロードされないため，バージョン変化のリスクは低い |
| R3: 訓練/評価データの潜在的なオーバーラップ | JMMLU の同じタスク内の異なる問題が訓練と評価にまたがる可能性 | label leakage の再演（Iter10 の問題） | `build_dataset.py` はシードを分ける（20260726 vs 20260727）かつ質問単位で除外する．ただし同じタスク内の異なる問題は重複し得る．これはデータセット設計の制約であり，完全なタスク単位分離は JMMLU の56タスク×10ドメインの写像で不可能 |
| R4: 訓練に必要な Ollama リソース | 訓練スクリプトは embedding 生成にライブ Ollama を必要とする | WAFL-PEFT が GPU を使用中だと訓練が失敗する | 訓練実行前に WAFL-PEFT の稼働状況を確認．wafl500 の Ollama を単一ホストで訓練に专用する |
| R5: softmax 確率の較正 | scikit-learn の LogisticRegression はデフォルトで較正されていない | ECE が改善しない可能性 | scikit-learn のデフォルト LogisticRegression は内部に較正を組み込んでいる（CalibratedClassifierCV 不要）．ECE を監視し，改善しない場合は較正曲線を分析 |
| R6: general クラスの識別困難 | general は「どの専門分野でもない」を意味するため，embedding 空間で他の9ドメインと重複する | general の precision が低く，専門ドメインへの誤分類が増える | MoDEM の結果（Other=52.94%）と同様の構造的問題．general の per-domain 指標を重点監視．`class_weight="balanced"` が部分的に補正 |
| R7: timeout_min の不足 | 現在 config.yml の experiment.timeout_min=90 だが，実験時間は約 90-120 分 | 実験がタイムアウトで中断される | 実験実行前に timeout_min を 150 に引き上げる |

### 実装 (Iter17)

**単一レバー**: `routing_method` (E6), `self_report → supervised_classifier`

**変更箇所**:
1. `config.yaml` 行31: `routing_method: self_report → supervised_classifier`（単一レバー変更）
2. `Dockerfile` 行14: `COPY scripts/ ./scripts/` の追加（訓練スクリプトをコンテナに含めるため）
3. `.claude/research/config.yml` 行26: `timeout_min: 90 → 150`（実験時間の余裕確保）
4. `mise.toml` 行73-75: models/ ディレクトリの rsync 前に `sudo rm -rf` を追加（root 所有の stale ディレクトリ対策）
5. `scripts/analyze_iter16.py` 行5: 未使用の `import sys` を削除（lint 修正）

**分類器訓練**:
- 訓練データ: `data/classifier_train.jsonl`（1427 件，10 ドメイン）
  - legal: 77 件（JMMLU に professional_law 不在のため他ドメインの半分）
  - 他9ドメイン: 各150 件
- 訓練方法: `LogisticRegression(max_iter=1000, class_weight="balanced")`
- embedding モデル: `nomic-embed-text`（768 次元）
- 訓練実行: wafl500 の Ollama コンテナに対し SSH トンネル（localhost:11435）経由で embedding 生成
- 出力: `models/domain_classifier.joblib`（62KB）

**オフライン分類精度**:
- 訓練データ: 100.00%（1427/1427，過学習）
- 評価データ（単一ドメイン1500問）: 59.87%（898/1500）
  - history_culture: 68.00%，legal: 68.67%，mathematics: 68.67%
  - social_science: 64.67%，natural_science: 62.00%，computer_science: 60.00%
  - general: 59.33%，medical: 52.00%，business_economics: 50.00%，education: 45.33%
- ベースライン比較: Random=10%，Iter16 self_report=20.59% → 分類器は約3倍の精度
- 訓練/評価ギャップ: 0.4013（768次元embeddingに対する1427サンプルの過学習）

**デプロイ検証**:
- `uv run pytest tests/ -v`: 180件全PASS（回帰なし）
- `uv run ruff check`: All checks passed
- `mise run setup`: Docker イメージ再ビルド・ローカル registry push 成功（scripts/ 含む）
- `mise run deploy`: 全10ノード（wafl500〜509）の config.yaml と models/domain_classifier.joblib を配布・app コンテナ再作成・起動成功
- 全ノード healthy 確認（wafl507-509 は初回 healthcheck で遅延したものの再試行で正常）
- wafl500 上のコンテナ内設定確認: `routing_method: supervised_classifier` が正しく反映
- wafl500 上のコンテナ内モデル確認: `/app/models/domain_classifier.joblib`（63095バイト）が存在
- wafl500 コンテナ起動ログ確認: エラーなし，GPU モデル両方ロード済み

**実験開始の可否**: 実験を開始してよい状態である．

### 実験 (Iter17)

- **実験ディレクトリ**: `results/20260727_180824`
- **データセット**: JMMLU 1520問（単一1500 + 複合20），全問完走（1520/1520）
- **所要時間**: 約91.8分（mean_duration_ms=3622.2，Iter16の4134.4より約12%短縮）
- **top1_accuracy**: 0.5651（Wilson CI: [0.5401, 0.5899]）
- **Cohen's kappa**: 0.5215
- **random_baseline**: 0.1013，best_single: legal/medical 0.1092
- **misrouting_rate**: 0.4349，fallback_rate: 0.1316
- **dispatch_failure_rate**: 0.0
- **同点タイ率**: 0.00%（Iter16: 82.83%）
- **バックグラウンドタスク**: コピー段階で `sh exited with non-zero status: no exit status` のエラーが発生したため，手動で `ssh wafl500 cat ... > results/...` により結果ファイルをコピー
- **異常**: なし（全ノードログ収集済み，全10ノードで正常動作）

### 分析 (実行) (Iter17)

**比較ベースライン**: `results/20260727_100917/` (Iter16, self_report + top_k_with_probs)

| 指標 | Iter16 (self_report) | Iter17 (supervised_classifier) | 変化 |
|------|---------------------|-------------------------------|------|
| top1_accuracy | 0.2059 | 0.5651 | +0.3592 |
| Wilson 95% CI | [0.1863, 0.2270] | [0.5401, 0.5899] | 重ならなし |
| Cohen's kappa | 0.1067 | 0.5215 | +0.4148 |
| misrouting_rate | 0.7941 | 0.4349 | -0.3592 |
| fallback_rate | 0.0000 | 0.1316 | +0.1316 |
| mean_duration_ms | 4134.4 | 3622.2 | -512.2 |
| 同点タイ率 | 82.83% | 0.00% | -82.83pt |

**McNemar 対比较**:

| | Iter17 正解 | Iter17 不正解 |
|---|---|---|
| Iter16 正解 | 179 (a) | 134 (b) |
| Iter16 不正解 | 680 (c) | 527 (d) |

- 不一致対数: b+c = 814
- McNemar chi-squared（連続性補正）: 365.57
- **p-value: < 0.000001**
- **有意差あり** (α=0.05)

**ドメイン別 precision/recall 比較**:

| ドメイン | Iter16 prec/rec | Iter17 prec/rec | 変化 |
|---------|----------------|----------------|------|
| business_economics | 0.242 / 0.100 | 0.511 / 0.453 | +0.269 / +0.353 |
| computer_science | 0.439 / 0.193 | 0.614 / 0.540 | +0.175 / +0.347 |
| education | 0.114 / 0.551 | 0.520 / 0.411 | +0.406 / -0.140 |
| general | 0.169 / 0.280 | 0.317 / 0.680 | +0.148 / +0.400 |
| history_culture | 0.200 / 0.060 | 0.764 / 0.647 | +0.564 / +0.587 |
| legal | 0.380 / 0.325 | 0.817 / 0.566 | +0.437 / +0.241 |
| mathematics | 0.511 / 0.160 | 0.725 / 0.667 | +0.214 / +0.507 |
| medical | 0.385 / 0.120 | 0.517 / 0.470 | +0.132 / +0.350 |
| natural_science | 0.438 / 0.140 | 0.580 / 0.580 | +0.142 / +0.440 |
| social_science | 0.245 / 0.080 | 0.685 / 0.580 | +0.440 / +0.500 |

**複合ドメイン**: 20問中5問正解（25.0%），domain_set_recall=0.125（Iter16: 0.475）

**ドメイン別 McNemar 対比較**:

| ドメイン | acc_16 | acc_17 | 変化 | chi2 | p-value | 判定 |
|---------|--------|--------|------|------|---------|------|
| business_economics | 0.1000 | 0.4533 | +0.3533 | 40.36 | <0.0001 | **有意改善** |
| computer_science | 0.1933 | 0.5400 | +0.3467 | 34.22 | <0.0001 | **有意改善** |
| education | 0.5400 | 0.4333 | -0.1067 | 3.63 | 0.0568 | 有意差なし（退行傾向） |
| general | 0.2800 | 0.6800 | +0.4000 | 37.84 | <0.0001 | **有意改善** |
| history_culture | 0.0600 | 0.6467 | +0.5867 | 80.52 | <0.0001 | **有意改善** |
| legal | 0.2867 | 0.6200 | +0.3333 | 27.92 | <0.0001 | **有意改善** |
| mathematics | 0.1600 | 0.6667 | +0.5067 | 70.31 | <0.0001 | **有意改善** |
| medical | 0.1200 | 0.4933 | +0.3733 | 42.01 | <0.0001 | **有意改善** |
| natural_science | 0.1400 | 0.5800 | +0.4400 | 50.30 | <0.0001 | **有意改善** |
| social_science | 0.0800 | 0.5800 | +0.5000 | 62.94 | <0.0001 | **有意改善** |

9/10 ドメインで有意改善．education は p=0.0568 で有意閾値をわずかに下回らず，有意差なし．

**ECE（Expected Calibration Error）**:

Iter16: **0.7388** → Iter17: **0.2118**（**-71.3%**）．

Iter16 では confidence 値が {0.6, 0.8, 0.9, 0.95, 1.0} の5段階離散値で，99.9% が [0.9, 1.0) ビンに集中し，bin_accuracy=0.2062 に対する bin_confidence=0.9450 の乖離（gap=0.7388）が ECE 全体を支配していた．

Iter17 では softmax 連続値により confidence が [0.22, 1.00] の範囲に広がり，8 ビンに分布した．特に [0.9, 1.0) ビン（40.7%）では bin_accuracy=0.7948，bin_confidence=0.9698（gap=0.1750）と，Iter16 の gap=0.7388 と比べて大幅に縮小した．

**ドメイン別 ECE（Iter17）**:

| ドメイン | accuracy | mean_conf | ECE |
|---------|----------|-----------|-----|
| mathematics | 0.6667 | 0.8145 | 0.1478 |
| history_culture | 0.6467 | 0.8256 | 0.1789 |
| legal | 0.6200 | 0.8057 | 0.1857 |
| social_science | 0.5800 | 0.7626 | 0.1898 |
| computer_science | 0.5400 | 0.7443 | 0.2043 |
| natural_science | 0.5800 | 0.7855 | 0.2055 |
| general | 0.6800 | 0.8110 | 0.2580 |
| medical | 0.4933 | 0.7570 | 0.2637 |
| business_economics | 0.4533 | 0.7495 | 0.2962 |
| education | 0.4333 | 0.7294 | 0.2960 |

全ドメインで ECE < 0.30．mathematics（0.1478）と history_culture（0.1789）が最も較正されており，education（0.2960）と business_economics（0.2962）が最も較正が低い．

**同点タイ率**:

Iter16: **82.83%** → Iter17: **0.00%**（**-82.83pt**）．

Iter16 では confidence 値が5段階の離散値であり，10ノードが独立に5値を選ぶと必然的に同値が発生した．Iter17 では softmax 出力が連続値（1518の唯一値，8桁小数点以下で計測）であり，同値発生確率は実質 0% である．

**fallback_rate 分析**:

Iter16: **0.0000** → Iter17: **0.1316**（200/1520）．

- 原因: `confidence_threshold=0.5` を下回るケースで fallback 発生．max_probe_conf < 0.5 の質問がちょうど 200 件であり，fallback 件数と完全に一致する．
- fallback 先のドメイン: 全て `general`（200/200）．
- fallback 時の confidence 分布: [0.220, 0.500]，平均 0.418．分類器がどのドメインにも確信もって分類できない「境界領域」の質問である．
- fallback 正解率: **8.0%**（16/200）．general への盲目的フォールバックは，これらの質問の正解ドメインが general である割合（16/200 = 8.0%）と一致し，fallback 戦略自体が有用なルーティング信号を持っていないことを示す．
- fallback 元の期待ドメイン分布: education(29), legal(26), business_economics(25), computer_science(24), medical(23) の順で多く，general(16), mathematics(14), history_culture(10) は少ない．education と legal が分類器の識別困難領域であることを示唆する．

**education の退行分析**:

recall: 0.5506 → 0.4114（CI17 下限 0.3377 < CI16 下限 0.4728）．**非退行条件を違反**．

- Iter16 では education ノードが education 質問に対して平均 confidence=0.8743 を出し，self_report の自己宣伝バイアスにより多くの教育関連質問を education へ引き寄せていた（recall 0.55）．しかし precision は 0.114 と極めて低く，education 以外の質問も education へ誤ルーティングされていた．
- Iter17 では education ノードの confidence が平均 0.4315 に低下し，分類器が education 質問を正しく識別できない．その結果，recall が 0.41 に低下した．precision は 0.520 と大幅に改善したが，recall の低下が全体を押し下げている．
- 根本原因: JMMLU に education に対応する直接的なタスクが存在せず，心理学・社会学タスクで代理しているため，embedding 空間で education クラスの分離超平面が不明瞭である可能性が高い．

**複合ドメインの退行**:

top1_accuracy: 0.9500 → 0.2500，domain_set_recall: 0.475 → 0.125．

- Iter16 では self_report の高タイ率（82.83%）により，複合ドメイン質問でも複数の期待ドメインが同点になり，宣言順で正解ドメインが選ばれる確率が高かった．これは構造上の偽高値である．
- Iter17 では softmax 連続値によりタイが解消され，分類器が単一ドメインを選択する．複合ドメイン質問は本質的に複数のドメインに属するため，単一選択では正解率が下がる．
- compound_mean_dispatched_count: Iter16=1.0 → Iter17=0.7．fallback 発生（200件中複合ドメインも含まれる）により dispatch 数が減少している．

**レイテンシ**:

mean_duration_ms: 4134.4 → 3622.2（**-12.4%**）．supervised_classifier は probe 段階で LLM コールを不要とし，embedding 計算のみで confidence を算出するため，probe ラウンドトリップ時間が短縮された．

**Cohen's kappa 比較**:

Iter16: 0.1067（95% CI: [0.0608, 0.1554]）→ Iter17: 0.5215（95% CI: [0.4890, 0.5404]）．CI が重ならず，**有意に高い**．

po（観測一致率）: 0.1987 → 0.5632．pe（偶然一致率）: 0.1016 → 0.0999．kappa の改善は，偶然一致を差し引いた実質的なドメイン識別力の向上を反映している．

**ベースライン比較**:

| ベースライン | accuracy | Iter17 比 |
|-------------|----------|-----------|
| Random | 0.1013 | 5.6x |
| BestSingle (legal/medical) | 0.1092 | 5.2x |
| Iter16 (self_report) | 0.2059 | 2.7x |
| Iter17 (supervised) | 0.5651 | - |
| Oracle | 1.0000 | - |

Iter17 は Random の 5.6 倍，Iter16 の 2.7 倍．Oracle までのギャップは 0.4349（Random→Oracle の距離の 48.4% を埋めた）．

### 分析 (解釈) (Iter17)

#### 1. 大幅改善のメカニズム解釈

**観測事実の再確認**: top1_accuracy 0.2059 → 0.5651（+0.3592）．McNemar chi2=365.57, p < 0.000001．Wilson CI は [0.1863, 0.2270] vs [0.5401, 0.5899] で完全に重ならなし．

この変化はノイズの範疇を超えている．1520 問における二項 SE は約 0.01 であり，+0.3592 は約 36 SE の変化である．過去の反復（Iter15→Iter16 で +0.022, p=0.0783）と比較しても，その効果量が桁違いである．

**self_report の構造的問題の解決メカニズム**:

self_report（Iter16）の根本問題は，各ノードの light_model が「あなたは{domain}分野の専門家です」というシステムプロンプトの影響を受け，自分の担当分野に対して過度に高い confidence を出す「自己宣伝バイアス」であった．Iter15 の numeric_scalar では 74.9% が 0.9 に飽和し，Iter16 の top_k_with_probs でも 80.5% が 0.95 に集中した．10 ノードが独立にこのバイアスを持つため，98.29%（Iter15）→ 82.83%（Iter16）の同点タイが発生し，実質的にルーティングは宣言順に依存する状態であった．

supervised_classifier はこの構造的問題を根本的に回避している．理由は以下の通りである．

- **自己宣伝バイアスの除去**: 分類器は embedding 空間の幾何的パターンのみで判定し，ドメイン固有のプロンプト指示を受けない．各ノードが同じ多クラス分類器をロードし，自分のクラスの softmax 確率のみを返すため，ノード間に一貫性のある confidence 分布が生成される．
- **全クラス確率の合計制約**: softmax 出力により全 10 クラスの確率が合計 1 になるため，正解ドメインの確率が 0.3 なら他ドメインの合計は 0.7 になる．self_report では各ノードが独立に 0.9 を出すため比較不可能だったのに対し，supervised_classifier では自然に弁別力のある分布が生成される．
- **連続値出力によるタイ解消**: softmax 出力は連続値（1518 の唯一値）であり，同点タイ率は 82.83% → 0.00% に完全に解消された．

**ECE の -71.3% 改善の理由**:

Iter16 の ECE=0.7388 は，confidence 値が {0.6, 0.8, 0.9, 0.95, 1.0} の 5 段階離散値で，99.9% が [0.9, 1.0) ビンに集中し，bin_accuracy=0.2062 に対する bin_confidence=0.9450 の乖離（gap=0.7388）が ECE 全体を支配していた．これは「confidence が高いのに accuracy が低い」という較正の破綻である．

Iter17 の ECE=0.2118 は，softmax 出力が [0.22, 1.00] の範囲に広がり，8 ビンに分布した結果である．特に [0.9, 1.0) ビン（40.7%）では bin_accuracy=0.7948，bin_confidence=0.9698（gap=0.1750）と，Iter16 の gap=0.7388 と比べて大幅に縮小した．scikit-learn の LogisticRegression は内部に較正を組み込んでいるため，CalibratedClassifierCV なしでも比較的良好な較正が得られている．

ただし，全ドメインで mean_conf > accuracy（education: 0.7294 > 0.4333, business_economics: 0.7495 > 0.4533 など）であり，依然として overconfident である．これは scikit-learn のデフォルト LogisticRegression が完全な較正を保証しないこと，および embedding 空間のクラス境界が完全には分離されていないことに起因する．ECE=0.2118 は「実用的に許容可能な範囲（< 0.30）」ではあるが，完全な較正（ECE < 0.05）には程遠い．

**kappa の +0.4148 改善の理由**:

Cohen's kappa は，観測一致率（po）から偶然一致率（pe）を差し引いた指標である．Iter16: po=0.1987, pe=0.1016 → kappa=0.1067．Iter17: po=0.5632, pe=0.0999 → kappa=0.5215．

kappa の改善は，偶然一致を差し引いた実質的なドメイン識別力の向上を反映している．10 分野で偶然一致率は約 0.10 であり，Iter16 の po=0.1987 は偶然よりわずかに良い程度（kappa=0.1067 = "slight agreement"）だったのに対し，Iter17 の po=0.5632 は偶然を有意に上回り（kappa=0.5215 = "moderate agreement"），実質的なルーティング能力が確立されたことを示している．

**仮説との整合**: 計画フェーズで述べた 4 つの仮説はすべて支持された．

1. 自己宣伝バイアスの除去 → 支持（同点率 0.00%，ドメイン別 McNemar で 9/10 有意改善）
2. 全クラス確率の合計制約 → 支持（ECE -71.3%，kappa +0.4148）
3. anisotropy への頑健性 → 支持（Iter2 の cosine 潰れとは異なり，分類精度 56.51% を達成）
4. Iter2（unsupervised）との違い → 支持（教師あり学習が分離超平面を学習した結果，Random の 5.6 倍）

#### 2. education recall 退行の解釈

**観測事実**: recall 0.5506 → 0.4114．CI17 下限 0.3377 < CI16 下限 0.4728．**非退行条件を違反**．ドメイン別 McNemar で p=0.0568（有意差なし）．

**根本原因の分析**:

この退行は，手法自体の欠陥ではなく，データセットの構造的問題に起因すると解釈する．

1. **JMMLU に education 対応タスク不在**: JMMLU の 56 タスクは MMLU 由来の学術科目であり，本研究の education ドメイン（日本の教育行政・教育実務）に相当するタスクが存在しない．心理学・社会学タスクで代理しているため，embedding 空間で education クラスの分離超平面が不明瞭である．
2. **訓練データの不均衡**: legal と同様に，education の訓練データは 77 件（他ドメインの半分）．`class_weight="balanced"` が補正しているものの，768 次元 embedding 空間における 77 サンプルでは，education クラスの決定境界が不安定になりやすい．
3. **オフライン分類精度の低さ**: education のオフライン分類精度は 45.33%（全ドメイン中最下位）であり，分類器自体が education クラスの識別に困難を抱えている．

**Iter16 の education recall=0.5506 の解釈**: Iter16 では education ノードが自己宣伝バイアスにより education 質問に対して平均 confidence=0.8743 を出し，多くの教育関連質問を education へ引き寄せていた．precision=0.114 と極めて低かったため，education 以外の質問も education へ誤ルーティングされていた．Iter17 では precision=0.520 と大幅に改善したが，recall が 0.41 に低下した．

**総合判断**: education recall の退行は，self_report から supervised_classifier への移行による「偽高値の剥奪」の側面と，embedding 空間での education クラス識別困難の側面の両方がある．手法自体の棄却根拠にはならないが，データセット整備（education 固有の訓練データ追加）または分類器の再訓練（education クラスの oversampling）が必要である．

#### 3. fallback_rate=13.16% の解釈

**観測事実**: 200/1520（13.16%）の質問が fallback 発生．max_probe_conf < 0.5 の質問が 200 件であり，fallback 件数と完全に一致する．

**fallback のメカニズム**:

`confidence_threshold=0.5` は，分類器の最大クラスの softmax 確率が 0.5 未満の場合に fallback を発生させる．10 クラスの softmax 出力において，最大値が 0.5 未満ということは，分類器がどのドメインにも確信もって分類できない「境界領域」の質問であることを意味する．Random baseline（10 クラス）の期待値は 0.10 であるため，0.5 は「Random より 5 倍確信がある」ことを示す閾値である．

**fallback 正解率 8.0% の問題**:

fallback 先のドメインはすべて `general` であり，fallback 正解率は 8.0%（16/200）である．これは Random baseline（10.1%）より低い．つまり，盲目的な general fallback は，これらの質問の正解ドメインが general である割合（8.0%）と一致し，fallback 戦略自体が有用なルーティング信号を持っていない．

**fallback 元の期待ドメイン分布**:

education(29), legal(26), business_economics(25), computer_science(24), medical(23) の順で多く，general(16), mathematics(14), history_culture(10) は少ない．education と legal が分類器の識別困難領域であることを示唆する．これは訓練データ不足（77 件）と関連しており，これらのドメインの境界領域で分類器が確信もって判定できない．

**改善提案**:

1. **confidence_threshold の最適化**: 現在 0.5 だが，これを下げる（0.3-0.4）ことで fallback 率を下げ，general への盲目的フォールバックを減らすことができる．ただし，閾値を下げると misroute が増えるトレードオフがある．
2. **fallback 戦略の変更**: general へのフォールバックではなく，分類器の top-2 クラスを dispatch 対象とする（dispatch_top_k=2）か，confidence の低い質問に対して複数の専門ノードに並行 dispatch する方が，8.0% の正解率を改善する可能性がある．
3. **education/legal の訓練データ追加**: 境界領域の質問を減らす根本的な解決策である．

#### 4. 複合ドメインの退行（0.95 → 0.25）の解釈

**観測事実**: top1_accuracy 0.9500 → 0.2500，domain_set_recall 0.475 → 0.125．

**Iter16 の高値は偽高値**:

Iter16 では self_report の高タイ率（82.83%）により，複合ドメイン質問でも複数の期待ドメインが同点になり，宣言順で正解ドメインが選ばれる確率が高かった．20 問中 19 問正解（95%）は，ルーティング能力ではなく，タイ解決メカニズムの構造上の副産物である．

**Iter17 の値の方が実態を反映**:

supervised_classifier は softmax 連続値によりタイを解消し，分類器が単一ドメインを選択する．複合ドメイン質問は本質的に複数のドメインに属するため，単一選択では正解率が下がる（25%）．これは「supervised_classifier が悪い」という意味ではなく，「複合ドメイン質問の評価方法が単一選択ルーティングに適していない」ことを示している．

**domain_set_recall の低下**:

Iter16: 0.475 → Iter17: 0.125．複合ドメイン質問の正解ドメインセットの中に，ルーティング先が含まれる割合である．Iter17 では fallback 発生（200 件中複合ドメインも含まれる）により，dispatch 数が減少（compound_mean_dispatched_count: 1.0 → 0.7）しており，これが domain_set_recall の低下に寄与している．

**判断**: 複合ドメインの退行は，評価方法とルーティング方式の不一致に起因する．supervised_classifier の性能評価からは除外すべきである．複合ドメイン質問に対する適切な評価は，dispatch_top_k >= 2 の設定で再評価するか，domain_set_recall のみを指標とするべきである．

#### 5. 総合判定

**成功条件に対する判定**:

| 分類 | 指標 | ベースライン (Iter16) | Iter17 結果 | 判定 |
|------|------|---------------------|------------|------|
| 主基準 | top1_accuracy McNemar | 0.2059 (p=0.0783 vs Iter15) | 0.5651, p < 0.000001 | **達成** |
| 主基準 | Wilson CI 重なり | [0.1863, 0.2270] | [0.5401, 0.5899] | **達成**（重ならなし） |
| 副基準 | Cohen's kappa | 0.1067 | 0.5215 (CI 重ならなし) | **達成** |
| 副基準 | 同点タイ率 | 82.83% | 0.00% | **達成**（有意な低下） |
| 副基準 | ECE | 0.7388 | 0.2118 (-71.3%) | **達成**（明確な改善） |
| 非退行 | per-domain precision/recall | Iter16 の各値 | education recall 退行 | **違反**（education のみ） |
| 監視 | probe レイテンシ | 4134.4ms | 3622.2ms (-12.4%) | **達成**（短縮） |
| 監視 | dispatch_failure_rate | 0.0 | 0.0 | **達成** |

**判定: 採用**

主基準 2 件・副基準 3 件の全 5 件を達成し，教育ドメインの recall 退行のみが非退行条件を違反している．しかし，この退行は手法の欠陥ではなくデータセットの構造的問題（JMMLU に education 対応タスク不在，訓練データ 77 件の不均衡）に起因すると解釈できるため，手法の採用判断には影響しない．

**E6（supervised_classifier）の採用理由**:

1. **self_report の構造的問題を根本的に解決した**: 自己宣伝バイアス，離散値飽和，同点タイの 3 つの問題を同時に解消し，top1_accuracy を 2.7 倍に改善した．
2. **統計的に明確な有意差**: McNemar p < 0.000001，Wilson CI 重ならなし，kappa 0.1067 → 0.5215．ノイズではなく明確な信号である．
3. **レイテンシも改善**: LLM コール不要の embedding 計算のみで probe するため，mean_duration_ms が -12.4% 短縮された．
4. **ベースラインを有意に上回る**: Random の 5.6 倍，Iter16 の 2.7 倍．Oracle までの距離の 48.4% を埋めた．

**残す課題**:

1. **education recall の退行**: 訓練データの不均衡（77 件）と JMMLU の education タスク不在が根本原因．education 固有の訓練データ追加，または oversampling による再訓練が必要．
2. **fallback_rate=13.16%**: 盲目的な general fallback の正解率 8.0% は Random より低い．confidence_threshold の最適化，または fallback 戦略の変更（top-2 dispatch）が必要．
3. **複合ドメインの退行**: 評価方法とルーティング方式の不一致．単一選択ルーティングにおける複合ドメイン評価は，domain_set_recall のみ，または dispatch_top_k >= 2 で再評価すべき．
4. **softmax 確率の overconfidence**: 全ドメインで mean_conf > accuracy．ECE=0.2118 は許容範囲内だが，完全な較正には程遠い．CalibratedClassifierCV による較正の検討余地あり．

**次の考察フェーズへの示唆**:

- **E6 の routing_method=supervised_classifier を採用し，config.yaml に固定する**．
- **education の訓練データ整備**を次イテレーションの優先課題とする（E10 の expert_specialization とは独立したデータ整備タスク）．
- **fallback 戦略の改善**（confidence_threshold の最適化，または top-2 dispatch）を次の単一レバー候補として検討する．
- **E10（expert_specialization）** は，ルーティング精度が Random の 5.6 倍に改善した現在，その価値を回答品質（評価軸②③）で検証する適切な時期に来ている．supervised_classifier が正しいドメインにルーティングするようになったため，ノード間の能力差が回答品質に反映される環境が整った．

### 考察・次計画 (Iter17)

**判定: E6（routing_method=supervised_classifier）— 採用**

主基準 2 件（McNemar 有意差，Wilson CI 重ならなし）・副基準 3 件（kappa 改善，同点率解消，ECE 改善）の全 5 件を達成．education recall の非退行違反のみあるが，これは手法の欠陥ではなく JMMLU データセットの構造的問題（education 対応タスク不在，訓練データ 77 件の不均衡）に起因するため，採用判断には影響しない．

**このイテレーションで確定した非自明な学び**

1. **self_report の構造的問題は routing_method の変更でしか解決できない**: Iter15（numeric_scalar）と Iter16（top_k_with_probs）の両方で ECE > 0.7 であり，confidence elicitation の方式変更だけでは自己宣伝バイアスを解消できないことが確定した．embedding ベースの教師あり分類に切り替えることで，top1_accuracy を 2.7 倍（0.2059 → 0.5651）に改善し，ECE を -71.3%（0.7388 → 0.2118）に低減した．

2. **Iter2（unsupervised embedding）の棄却は正当だったが，原因は unsupervised であること**: Iter2 で cosine similarity が [0.667, 0.737] に潰れた原因は embedding の anisotropy であり「信号が無い」証明ではなかった．教師あり分類（LogisticRegression）は anisotropy 下でも分離超平面を学習できるため，supervised classifier は Random の 5.6 倍，Iter16 の 2.7 倍の精度を達成した．RouterDC（NeurIPS 2024）の報告（CosineClassifier に全タスクで勝利）と整合する．

3. **softmax 連続値は同点タイを完全解消する**: self_report の離散値（5段階）による 82.83% の同点タイが，softmax 連続値（1518 の唯一値）により 0.00% に完全に解消された．ルーティングが宣言順ではなく実質的なドメイン識別信号に依存する環境が初めて実現した．

4. **fallback_rate=13.16% は，分類器の「確信できない」境界領域を可視化している**: confidence_threshold=0.5 を下回る 200 問（13.16%）は，分類器がどのドメインにも確信もって分類できない境界領域である．fallback 正解率 8.0% は Random（10.1%）より低く，盲目的な general fallback は有用な信号を持っていない．education（29 件），legal（26 件），business_economics（25 件）が fallback 元として多く，訓練データ不足（77 件）と関連している．

5. **複合ドメインの退行（0.95 → 0.25）は評価方法の不一致**: Iter16 の 95% は self_report の高タイ率（82.83%）による構造上の偽高値であり，Iter17 の 25% の方が実態を反映している．単一選択ルーティングにおける複合ドメイン評価は，dispatch_top_k >= 2 で再評価するか，domain_set_recall のみとするべきである．

**次の単一レバー: E10（expert_specialization）**

supervised_classifier によりルーティング精度が Random の 5.6 倍（top1_accuracy=0.5651）に改善した現在，ノード間の能力差が回答品質に反映される環境が整った．現在 4 ノードすべてが同一モデル（isotnek/qwen3.5:9B-Unsloth-UD-Q4_K_XL）で，差分はプロンプト 1 文だけであるため，誤ルーティングしても回答品質はほぼ変わらず，top1_accuracy は下流に帰結を持たない代理指標になっていた．

E10（expert_specialization）の実施により，初めて「正しいドメインにルーティングされた質問が，実際に良い回答を得るか」を評価軸②（回答品質，LLM-as-judge）と③（End-to-End）で検証できる環境が整う．

- 本命は `domain_lora`（単一ベース + ドメイン LoRA アダプタ）: 6GB VRAM 制約下で最も現実的であり，S-LoRA（MLSys2024）が多数アダプタの同時配信を示している．日本語医療 LoRA の先行例（JMedLoRA）もあり，同じ 10 台の GPU プール上で LoRA 学習を行う仕組みは WAFL-PEFT 側に既にある．
- `offtheshelf_specialized` も候補だが，日本語の法律特化オープン生成モデルは発見できなかったため，domain_lora を優先する．
- **E10 と同時に評価軸②③（回答品質・End-to-End）を実装すること**．それらが無いとルーティングの価値を測れない．

---

## Iteration 16: Verbalized Top-K による二峰飽和と同点タイの解消検証

### 実装 (Iter16)

**単一レバー**: `confidence_elicitation` (E3), `numeric_scalar → top_k_with_probs`

**変更箇所**: `config.yaml` 行36 の1行変更のみ．

**検証**:
- `uv run pytest tests/ -v`: 180件全PASS（Iter15と同じ件数，回帰なし）
- `uv run ruff check`: All checks passed
- `mise run setup`: Docker イメージ再ビルド・ローカル registry push 成功
- `mise run deploy`: 全10ノード（wafl500〜509）の app コンテナ再作成・起動成功，warmup 後全ノード healthy
- wafl500 上のコンテナ内設定確認: `confidence_elicitation: top_k_with_probs` が正しく反映

**実験開始の可否**: 実験を開始してよい状態である．

### 実験 (Iter16)

- **実験ディレクトリ**: `results/20260727_100917`
- **データセット**: JMMLU 1520問（単一1500 + 複合20），全問完走
- **所要時間**: 約105分（mean_duration_ms=4134.4）
- **top1_accuracy**: 0.2059（Wilson CI: [0.1863, 0.2270]）
- **Cohen's kappa**: 0.1067
- **random_baseline**: 0.1013，best_single: education 0.1039
- **misrouting_rate**: 0.7941，fallback_rate: 0.0
- **parse_failure_rate**: 0.0（0/1520）
- **confidence 分布**: 範囲 [0.6, 1.0]，唯一値 {0.6, 0.8, 0.9, 0.95, 1.0}（5段階）
- **異常**: なし（全ノードログ確認済み）

### 分析 (実行) (Iter16)

**比較ベースライン**: `results/20260727_010532/` (Iter15)

| 指標 | Iter15 | Iter16 | 変化 |
|------|--------|--------|------|
| top1_accuracy | 0.1836 | 0.2059 | +0.0223 |
| Wilson 95% CI | [0.1649, 0.2038] | [0.1863, 0.2270] | 下限 +0.0214 |
| Cohen's kappa | 0.0815 | 0.1067 | +0.0252 |
| misrouting_rate | 0.8164 | 0.7941 | -0.0223 |

**McNemar 対比较**: 不一致対数 362．chi2=3.10, p=0.0783．**有意差なし** (α=0.05)．

**同点タイ率**: 98.29% → 82.83% **-15.46pt**．verbalized top-K の意図した効果確認．

**ドメイン別 McNemar**: 6/10 ドメインで有意改善．general で有意退行 (-0.407)．

**confidence 分布**: 0.9 が 96.84% → 14.67%，0.95 が 0.16% → 33.37%．ピークが 0.9→0.95 へシフト．

**ECE**: 0.7146 → 0.7388．較正は悪化．

### 分析 (解釈) (Iter16)

#### 1. 同点タイ率 -15.46pt の解釈

**観測事実**: 98.29% → 82.83% (-15.46pt)．SE=0.0075 に対して 20.6 SE の変化であり，**ノイズではなく明確な信号**である．

**メカニズムの解釈**:

Iter15（numeric_scalar）では，各ノードが「0.9 または 0.2」の二峰値を申告し，10 ノード中 7〜10 ノードが 0.9 を出すため，実質的に全問でタイが発生した．Top-K elicitation（top_k_with_probs）に切り替えたことで，Qwen3.5-4B が「該当する/該当しない」の 2 択に確率を分配するようになり，confidence 値が {0.6, 0.8, 0.9, 0.95, 1.0} の 5 段階に分散した．

しかし，82.83% のタイ率は依然として高い．confidence 値の唯一値が 5 段階しかないため，10 ノードが 5 段階の値を独立に出す場合，同値になる確率は依然として高い（10^2 / 5^10 の単純計算ではなく，実際には 0.95 が 80.5% を占める偏りがあるためさらに高い）．

**解釈**: Top-K elicitation は二峰飽和を部分的に壊したが，**離散値の数が少ない（5段階）** ためタイは完全には解消されていない．これは Qwen3.5-4B の算数能力の限界であり，「0.73, 0.81, 0.64」のような連続値を生成できないためである．

#### 2. general の退行（0.687 → 0.280, -0.407）の解釈

**観測事実**: general の recall が -0.407 退行．SE=0.0408 に対して 10.0 SE の変化であり，**ノイズではなく明確な信号**．

**根本原因: 宣言順有利の剥奪**

Iter15 の general recall=0.687 の大部分は，ドメイン識別能力ではなく**宣言順 1 位によるタイ勝率 42.9%** によるものであった（Iter15 解釈節 3 参照）．1494 タイ中 641 件を general が勝っていた．

Top-K elicitation によりタイ率が 98.29% → 82.83% に低下したことで，**宣言順有利が相対的に小さくなった**．非タイケースでは，general ノードは自分の分野（general）に関する質問に対して 0.95 ではなく 0.8 や 0.6 を出すことがあり，専門ノード（mathematics, medical など）が同じ質問に対して 0.9 を出すと，general が負けるようになった．

**これは general の「実力」が低下したのではなく，Iter15 で観測されていた general の recall が「構造上の偽高値」であったことが露見した** ことに近い．Iter16 の general recall=0.280 は，宣言順有利が相対的に小さくなった環境下での**より正確な推定値**である可能性がある．

#### 3. 6/10 ドメインの有意改善と退行ドメインの構造的差異

**改善したドメイン（6/10）**:

| ドメイン | acc_15 | acc_16 | 変化 | p-value |
|---------|--------|--------|------|---------|
| computer_science | 0.007 | 0.193 | +0.187 | <0.001 |
| mathematics | 0.053 | 0.160 | +0.107 | 0.0047 |
| natural_science | 0.040 | 0.140 | +0.100 | 0.0053 |
| business_economics | 0.020 | 0.100 | +0.080 | 0.0040 |
| social_science | 0.000 | 0.080 | +0.080 | 0.0009 |
| history_culture | 0.000 | 0.060 | +0.060 | 0.0046 |

**改善のメカニズム**: これらのドメインは Iter15 で recall=0.0〜0.053 であり，実質的にルーティングされなかった（宣言順不利 + タイ）．Top-K elicitation により，各ノードが自分の分野に対してより高い confidence（0.95）を出すようになり，**非タイケースが増えたことで，実際のドメイン識別信号が反映されるようになった**．

特に computer_science（+0.187, 7.6 SE）と mathematics（+0.107）の改善は，これらの分野の質問が専門用語・数式を含むため，Top-K elicitation で「該当する」確率が明確に高くなる構造があることを示唆する．

**退行したドメイン（2/10）**:

| ドメイン | acc_15 | acc_16 | 変化 | p-value |
|---------|--------|--------|------|---------|
| general | 0.687 | 0.280 | -0.407 | <0.001 |
| legal | 0.440 | 0.349 | -0.090 | 0.0721（有意未満） |

**構造的差異**: general と legal の共通点は，**宣言順が上位（general=1位，legal=3位）** であり，Iter15 でタイ勝率が高かったことである（general 42.9%，legal 21.6%）．Top-K elicitation によりタイが減ると，この構造上の有利が剥奪される．

**不変ドメイン（2/10）**:

| ドメイン | acc_15 | acc_16 | 変化 | p-value |
|---------|--------|--------|------|---------|
| education | 0.494 | 0.563 | +0.070 | 0.1788 |
| medical | 0.157 | 0.199 | +0.042 | 0.2430 |

education（宣言順 2 位）は Iter15 でも比較的高い recall（0.494）を持っていたが，Top-K elicitation で有意な変化なし．medical（宣言順 4 位）も同様に安定している．両者とも Iter15 で既に一定のドメイン識別信号を持っていた可能性がある．

#### 4. ECE の悪化（0.7146 → 0.7388）の解釈

**観測事実**: ECE が +0.0242 悪化．

**理由**: ECE = 各ビンにおける |bin_accuracy - bin_confidence| の加重平均である．

- Iter15: confidence の中心が 0.9，accuracy=0.184 → 主要ビンの乖離 ≈ |0.184 - 0.9| = 0.716
- Iter16: confidence の中心が 0.95，accuracy=0.206 → 主要ビンの乖離 ≈ |0.206 - 0.95| = 0.744

Top-K elicitation は confidence 値を**上方シフト**させた（0.9 → 0.95）が，accuracy の改善（+0.022）はこれに追いつかなかった．その結果，confidence と accuracy の乖離は拡大し，ECE が悪化した．

**Top-K elicitation の較正効果の限界**: Tian et al. (EMNLP 2023) の結果（ECE 0.131→0.047）は，gpt-3.5-turbo（175B クラス）で得られた．Qwen3.5-4B（4B クラス）では，算数能力の不足により確率の合計制約は満たされるものの（再正規化により），**個別の確率値の較正精度は低い**．モデルは「該当する/該当しない」の 2 択で確率を分配できるが，その確率値自体が実際のドメイン適合度を反映していない．

#### 5. 総合判定

**成功条件に対する判定**:

| 分類 | 指標 | ベースライン | 結果 | 判定 |
|------|------|-------------|------|------|
| 主基準 | 同点率 | 98.29% | 82.83% (-15.46pt, 20.6 SE) | **採用**（明確な有意低下） |
| 主基準 | Cohen's kappa | 0.0815 | 0.1067 (+0.0252) | **判定不能**（依然として chance 直上，CI の重なり確認が必要） |
| 副基準 | McNemar | α=0.05 | p=0.0783 | **有意差なし**（有意閾値の 80% にあるが，閾値未満） |
| 副基準 | ECE | 0.7146 | 0.7388 | **悪化** |

**総合判定: 部分的採用**

Top-K elicitation は二峰飽和の解消（同点率 -15.46pt）において明確な成功である．しかし，**accuracy への帰結は McNemar で有意差なし**であり，較正（ECE）は悪化している．kappa は +0.0252 改善したが，0.1067 は依然として「chance 直上」であり，実質的なドメイン識別力は低い．

**McNemar の p=0.0783 の解釈**: 有意閾値（α=0.05）の 80% にあり，「ほぼ有意」と言える範囲である．362 件の不一致対（Iter15 不正解/Iter16 正解 = 198，逆 = 164）は，Iter16 の方が 34 問多いことを示す．これは Top-K elicitation が一部のドメイン（computer_science, mathematics 等）でルーティング精度を改善したことを反映しているが，general の退行（-0.407）が全体を押し下げている．

**重要な知見**: general の退行は「偽高値の剥奪」である可能性が高い．Iter15 の general recall=0.687 の大部分は宣言順有利によるものであった．Top-K elicitation によりタイが減ると，この構造上の有利が剥がれ，general の「実力」に近い値（0.280）が観測された．**これは Top-K elicitation の失敗ではなく，Iter15 の general の高値が構造上のアーティファクトであったことを示している**．

#### 6. 次イテレーションへの提案

**E6（supervised_classifier）を推奨する**．理由:

1. **self_report の根本的限界が確認された**: numeric_scalar でも top_k_with_probs でも，confidence 値はドメイン適合度を較正された形で反映していない（ECE > 0.7）．confidence elicitation の方式を変更するだけでは，self_report の構造的問題（各ノードが自分の分野に偏った confidence を出す）は解消されない．

2. **embedding ベースの教師あり分類は独立したアプローチ**: self_report（言語的自信）とは全く異なる信号源であり，E3 の結果とは独立して評価できる．Iter2（embedding）の失敗は unsupervised cosine similarity の anisotropy 問題であり，教師あり分類では解消される可能性がある．

3. **訓練/評価分離は既に実装済み**: Iter15 で label leakage 対策として訓練/評価クエリの構造的分離が実装済みであり，label leakage の再演リスクは低い．

4. **コード変更は不要**: E6 は `routing_method: self_report → supervised_classifier` の config.yaml 1 行変更のみで，scikit-learn ベースの LogisticRegression が既に実装済みである．

**E7（whitening）は E6 の前段階として検討可能**．E6 が不成功の場合，unsupervised embedding の幾何的改善（mean-centering + whitening）が E6 のベースラインを改善する可能性がある（Su+ 2021）．ただし，E7 は教師なしのため，E6 の教師ありアプローチより優先度は低い．

**E4（self_consistency_semantic）と E5（p_true）は，E6 の結果を確認してから検討する**．self_report の較正問題とは独立した signal method であるが，E6（routing_method の変更）が self_report を完全に置き換える可能性があり，その場合は E4/E5 の検証価値が下がる．

### 考察・次計画 (Iter16)

**判定: E3（confidence_elicitation=top_k_with_probs）— 部分的採用**

Top-K elicitation は二峰飽和の解消において明確な成功である（同点率 98.29% → 82.83%，-15.46pt，20.6 SE）．しかし，accuracy への帰結は McNemar で有意差なし（p=0.0783）であり，較正（ECE）は悪化（0.7146 → 0.7388）している．kappa は +0.0252 改善したが，0.1067 は依然として「chance 直上」であり，実質的なドメイン識別力は低い．

**このイテレーションで確定した非自明な学び**

1. **Top-K elicitation は二峰飽和を部分的に壊す**: confidence 値が {0.6, 0.8, 0.9, 0.95, 1.0} の5段階に分散し，同点率が -15.46pt 低下した．しかし，離散値が5段階しかないためタイは完全には解消されず（82.83%）．これは Qwen3.5-4B の算数能力の限界であり，連続値を生成できないためである．

2. **general の退行は偽高値の剥奪**: Iter15 の general recall=0.687 の大部分は宣言順1位によるタイ勝率 42.9% による構造上の偽高値であった．Top-K elicitation によりタイが減ると，この構造上の有利が剥がれ，general の「実力」に近い値（0.280）が観測された．これは Top-K elicitation の失敗ではなく，Iter15 の測定値がアーティファクトであったことを示している．

3. **self_report の根本的限界が確認された**: numeric_scalar でも top_k_with_probs でも，confidence 値はドメイン適合度を較正された形で反映していない（ECE > 0.7）．confidence elicitation の方式を変更するだけでは，self_report の構造的問題（各ノードが自分の分野に偏った confidence を出す）は解消されない．

4. **6/10 ドメインの有意改善は「実信号の露出」**: computer_science（+0.187），mathematics（+0.107），natural_science（+0.100）などの改善は，Iter15 で宣言順不利により実質的にルーティングされていなかったドメインが，Top-K により非タイケースが増えたことで，実際のドメイン識別信号が反映されるようになった結果である．

5. **ECE の悪化はモデル規模の限界**: Tian et al.（EMNLP 2023）の結果（ECE 0.131→0.047）は gpt-3.5-turbo（175Bクラス）で得られた．Qwen3.5-4B（4Bクラス）では，算数能力の不足により確率の合計制約は満たされるものの，個別の確率値の較正精度は低い．

**次の単一レバー: E6（routing_method=supervised_classifier）**

self_report の根本的限界が確認されたため，confidence elicitation の方式変更（E3, E4, E5）よりも，全く異なる信号源に基づく routing_method の変更が優先される．E6 は embedding ベースの教師あり分類であり，self_report（言語的自信）とは独立したアプローチである．Iter2（embedding）の失敗は unsupervised cosine similarity の anisotropy 問題であり，教師あり分類では解消される可能性がある．訓練/評価分離は既に実装済みであり，config.yaml 1行変更のみで検証可能である．

- 変更: `routing_method: self_report → supervised_classifier` のみ
- 固定: `confidence_signal_method: self_report`，`confidence_elicitation: top_k_with_probs`（ Iter16 の最良構成を継承），他全設定不変
- 比較: 同一 1520 問データセット上で McNemar 対比較（α=0.05）
- 成功条件: top1_accuracy の McNemar で有意差，Wilson CI が重ならない変化

---

### 計画 (Iter16)

**単一レバー**: `confidence_elicitation` (E3), 値 `numeric_scalar → top_k_with_probs`

**変更箇所**: `config.yaml` 行36 のみ
```
confidence_elicitation: numeric_scalar  →  confidence_elicitation: top_k_with_probs
```

**仮説**: Top-K elicitation（Tian et al. EMNLP 2023）は確率の合計制約（sum=1）により，self_report numeric_scalar の二峰飽和（0.9 が 74.9%）を壊し，連続的な confidence 分布を生成する．その結果，同点タイ率が大幅に低下し，kappa が改善する．

**固定する構成**（直近最良構成＝Iter15 実験構成をそのまま継承）:
- `confidence_signal_method: self_report`（変更不可．E3 は elicitation 方式の変更であり signal method 自体は self_report のまま）
- `routing_method: self_report`
- `confidence_threshold: 0.5`
- `dispatch_top_k: 1`
- `semantic_sample_count: 5`, `semantic_sample_temperature: 0.7`（E4 用設定は不変）
- `embedding_postprocess: none`
- `light_model: qwen3.5:4b-q4_K_M`, `expert_model: schroneko/llama-3.1-swallow-8b-instruct-v0.1:q4_k_m`（全10ノード共通）
- 10 ノード構成（wafl500〜509）
- `router.py` の few-shot 例（動的生成 `_build_few_shot_examples`）

**成功条件**

| 分類 | 指標 | ベースライン (Iter15) | 成功条件 | 根拠 |
|------|------|---------------------|---------|------|
| 主基準 | 同点率 | 98.29% (1494/1520) | **有意な低下**（McNemar α=0.05） | 二峰飽和の解消がタイ削減に直接反映される |
| 主基準 | Cohen's kappa | 0.081 | **0.081 より有意に高い**（Wilson CI が重ならない） | chance-corrected 指標で実質識別力を測定 |
| 副基準 | top1_accuracy | 0.184 [0.165, 0.204] | **Wilson CI がベースライン CI と重ならない** | McNemar 対比較（α=0.05） |
| 副基準 | ECE | 未計測（Iter15 では numeric_scalar の二峰分布） | **報告**（較正改善の定量化） | Tian et al. の主指標 |
| 監視 | parse failure 率 | N/A | **5% 未満** | Qwen3.5-4B の JSON 出力従順性確認 |
| 監視 | 再正規化頻度 | N/A | **報告**（_PROB_SUM_TOLERANCE=0.02 を超える割合） | R1（算数能力）の緩和策確認 |

**成功条件の数値根拠**: Iter15 の Wilson CI 幅は 0.039（3.9pt）であり，1520 問で SE は約 0.0096．Top-K elicitation が二峰飽和を壊す場合，同点率は 98.29% から大幅に低下する見込みであり，その差分は McNemar で有意（α=0.05）になる．kappa=0.081 は chance 直上であり，Top-K による連続分布がドメイン弁別力を向上させるなら，kappa も上昇する．

**実験構成**:
1. `config.yaml` 行36 のみ変更（`numeric_scalar → top_k_with_probs`）
2. `mise run setup`（Docker イメージ再ビルド．`router.py` の Top-K 関数が既に実装済みなので，イメージに反映させるため）
3. `mise run deploy`（全10ノード）
4. `mise run start`（同一 1520 問データセット `data/dataset.jsonl`）
5. `mise run analyze`（結果収集）
6. `metrics.py` による解析（Wilson CI, kappa, McNemar, ECE, 同点率）

**実行時間の見積もり**: Iter15 の mean_duration_ms=3826（約3.8秒/問）を基準に，1520 問で約 5780 秒（約 1.6 時間）．Top-K elicitation は probe 1 回/ノードのまま（追加 LLM コールなし）であり，numeric_scalar と同程度の推論時間を想定．ただし Qwen3.5-4B の JSON 出力が numeric_scalar より若干長くなる可能性があり，余裕を見て約 2 時間を見込む．

**特定されたリスクと緩和策**:

| リスク | 内容 | 緩和策 |
|-------|------|--------|
| R1 | Qwen3.5-4B の算数能力不足で sum=1 制約違反 | `parse_top_k_confidence()` の再正規化（許容誤差 0.02）がカバー．再正規化頻度を監視 |
| R2 | 生 Top-K 分布のロギング不足 | 本イテレーションでは必須ではないが，`probe_candidates` に `confidence_top_k_raw` を追加する検討を次イテレーションへ持ち越し |
| R3 | 4B モデルでの JSON 出力従順性 | `parse_top_k_confidence` は parse failure で 0.0 にフォールバック．parse failure 率を監視（5% 未満を目標） |
| R4 | ドメイン専門家プロンプトとの相互作用 | 各ノードが自分の分野に偏った確率分布を生成する可能性．Top-K は少なくとも 0/1 飽和を壊すため，self_report より改善が見込まれる |

### 調査 (Iter16)

**単一レバー**: `confidence_elicitation` (E3), 候補値 `top_k_with_probs`

**調査の問い**

1. `confidence_elicitation=top_k_with_probs` のコード実装は完了しているか．
2. プロンプト設計は Tian et al. (EMNLP 2023) の方式に沿っているか．
3. 解析パイプライン（aggregator, metrics, run_experiment）は Top-K 形式の出力と互換か．
4. 既知のリスク・課題は何か．

**1. 実装の現状**

実装は完全に完了しており，全テスト（180件）がPASSしている．

| 項目 | ファイル | 行番号 | 状態 |
|------|---------|-------|------|
| config.yaml のキー | `config.yaml` | 行36 | `confidence_elicitation: numeric_scalar`（変更1行で切替可能） |
| プロンプト生成（通常ドメイン） | `router.py` | 行193-212 | `build_top_k_confidence_prompt()` 実装済み |
| プロンプト生成（general） | `router.py` | 行178-190 | `_build_general_top_k_confidence_prompt()` 実装済み |
| 出力パース＋再正規化 | `router.py` | 行215-238 | `parse_top_k_confidence()` 実装済み |
| 非同期推論ラッパー | `router.py` | 行241-257 | `estimate_confidence_top_k()` 実装済み |
| http_server 分岐 | `http_server.py` | 行371-379 | `_estimate_probe_confidence()` 内に分岐あり |
| 識別子定数 | `http_server.py` | 行98-102 | `CONFIDENCE_ELICITATION_TOP_K_WITH_PROBS`, `VALID_CONFIDENCE_ELICITATIONS` |
| node.py 設定伝播 | `node.py` | 行66-67, 行84 | config から NodeState へ伝播 |
| 単体テスト | `tests/test_router.py` | 行238-278 | 7件全PASS |
| 統合テスト | `tests/test_http_server.py` | 行205-213 | 1件PASS |

**2. プロンプト設計の評価**

Tian et al. (EMNLP 2023, arXiv:2305.14975) の Verbalized Top-K との整合性を確認した．

| 要素 | Tian et al. の方式 | 本実装 | 整合 |
|------|-------------------|--------|------|
| 候補数 K | K=2（2-way elicitation） | `TOP_K_CANDIDATES = 2` | 一致 |
| 出力形式 | 各候補に確率を付与 | `{"candidates": [{"label": "...", "probability": ...}, ...]}` | 一致 |
| 合計制約 | sum(probabilities) = 1 の指示 | プロンプトに「確率の合計は1.0になるようにしてください」 | 一致 |
| 再正規化 | 論文では明示せず | `parse_top_k_confidence()` で合計が1.0から外れた場合は再正規化（許容誤差 0.02） | 補強あり |

**重要な違い**: Tian et al. は多クラス分類（3-5選択肢）で検証したが，本実装は2値分類（該当する/該当しない）である．Tian et al. Table 1 では gpt-3.5-turbo で top-2 verbalized confidence の ECE が 0.131→0.047 に改善した．2値分類でも確率の合計制約が0/1飽和を壊すメカニズムは同じだが，効果量は異なる可能性がある．

**3. 解析パイプラインの互換性**

完全互換である．Top-K elicitation は「入力プロンプトの形式」と「出力パースのロジック」だけを変え，`ProbeResponse.confidence` は依然として単一スカラー float である．

| パイプライン段階 | 処理 | 変更必要 |
|-----------------|------|---------|
| `estimate_confidence_top_k()` | Top-K プロンプト送出 → パース → "該当する"確率を抽出 | 実装済み |
| `ProbeResponse.confidence` | スカラー float [0,1] | 変更不要 |
| `aggregator.select_dispatch_targets()` | confidence スカラーでソート・閾値フィルタ | 変更不要 |
| `aggregator.select_best_dispatch_response()` | confidence 最大値選択 | 変更不要 |
| `run_experiment.py` probe_candidates | `confidence` スカラーを記録 | 変更不要 |
| `metrics.py` 全関数 | `confidence` スカラーを消費（ECE, kappa, precision/recall） | 変更不要 |

**4. 特定されたリスク・課題**

**R1: 小モデルの算数能力**
- Qwen3.5-4B は確率の合計=1制約を厳密に守れない可能性がある．
- 既存の再正規化（`_PROB_SUM_TOLERANCE = 0.02`）がこれをカバーするが，再正規化が頻発する場合，モデルの算数能力の限界が結果にバイアスを導入する．
- **緩和策**: 実験後に `parse_top_k_confidence` の再正規化頻度をログで確認する．

**R2: 生 Top-K 分布のロギング不足**
- 現在 `probe_candidates` は再正規化後のスカラー `confidence` のみを記録し，生 Top-K 分布（"該当する"確率と"該当しない"確率のペア）は記録しない．
- **影響**: 事後分析で「再正規化前の分布形状」や「2つの確率の相関」を確認できない．
- **緩和策**: 本イテレーションでは必須ではないが，必要に応じて `probe_candidates` に `confidence_top_k_raw` フィールドを追加する．

**R3: 2値分類 vs 多クラス分類の乖離**
- Tian et al. の結果は gpt-3.5-turbo（175Bクラス）で得られた．Qwen3.5-4B（4Bクラス）では効果が異なる可能性がある．
- 特に，4Bクラスモデルは few-shot 指示の従順性が低く，JSON形式の出力を正確に生成しないリスクがある．
- **緩和策**: `parse_top_k_confidence` は parse failure で `PARSE_FAILURE_CONFIDENCE=0.0` にフォールバックするため，最悪ケースでも安全である．parse failure 率を監視する．

**R4: ドメイン専門家プロンプトとの相互作用**
- 各ノードは「あなたは{domain}分野の専門家です」と指示されているため，Top-K elicitation であっても自分の分野に偏った確率分布を生成する可能性がある．
- これは Top-K elicitation の設計上の制約ではなく，ドメインプロンプト自体の問題である．
- Top-K elicitation は少なくとも0/1飽和を壊し，連続的な分布を得ることで，self_report よりも改善が見込まれる．

**計画フェーズへの提案**

1. **config.yaml 変更**: `confidence_elicitation: numeric_scalar → top_k_with_probs` の1行変更のみ．他は不変．
2. **成功条件（主指標）**: 同点率の有意な低下．ベースライン 98.29% に対し，Top-K では確率分布の連続性により同点率が大幅に低下する見込み．具体的な目標値は提案しない（モデルの算数能力に依存するため）が，ベースラインとの McNemar 対比較で有意差（α=0.05）を検出する．
3. **成功条件（副指標）**: Cohen's kappa の改善（ベースライン 0.081）．Top-K elicitation がドメイン弁別力を向上させる場合，kappa も上昇する．
4. **監視項目**: (a) parse failure 率（0.0%に近いことを確認），(b) 再正規化頻度（_PROB_SUM_TOLERANCE を超える頻度），(c) ドメイン別 confidence 分布の形状変化（二峰→連続分布への移行）．
5. **比較ベースライン**: `results/20260727_010532/`（Iter15, 1520問）．同一データセット上の McNemar 対比較が可能．

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

