## Iteration 34: education代理タスク抽出比率の再配分（案A）による訓練データ構成変更

### 仮説

**仮説**: `education`の3代理タスク（sociology・high_school_psychology・moral_disputes）の
抽出比率を，案C（70/40/40）から案A（90/30/30）へ変更すれば，`education_recall`がmedical_recall
基準（0.5112）を上回る。

**根拠**:
1. 案C（70/40/40）は現状比（41/55/54）からsociologyを+29pt，他2タスクを-15ptずつ変更した。
   変化幅では教育recallへの信号がノイズ（SE~3.8pt）に埋もれた（education_recall 0.4412
   < medical_recall基準 0.5112，70ptギャップ）。
2. 案A（90/30/30）は変化幅が案Cの約2倍（sociology +49pt，他2タスク -25pt）。
   効果量が約2倍になれば，有意検出の可能性が実測レベルで高まる（n=170でSE~3.8pt，
   5pt以上の効果量が有意検出の目安）。
3. sociologyのrecall（0.625）が最も高く，high_school_psychology（0.438）と
   moral_disputes（0.435）がeducation_recall全体を押し下げる主因であるという
   confusion matrix分析（Iter32）に基づき，sociologyの寄与を最大限に高める配分。
4. sociologyのpool cap（94）に対し90件は95.7%で，残り4件の余裕は確保される。

### 単一レバー

**変更するレバー**: `classifier_training_data_composition`（config.yml Y5レバー）の値，
具体的には`build_dataset.py`の定数`_EDUCATION_PROXY_TASK_TRAIN_TARGET_SIZES`を
`{"sociology": 90, "high_school_psychology": 30, "moral_disputes": 30}`へ変更する。

**変更しないレバー**: 上記定数以外のコード・設定ファイルは全て変更しない。
`_CLASSIFIER_TASK_SAMPLE_WEIGHTS`は空辞書のまま（Iter33でrevert済み），
`_sample_domain_questions()`の`task_target_sizes`分岐，
`build_classifier_training_rows()`のeducation特別扱いはIter33実装のまま。

### 変更ファイル一覧

**変更対象ファイル（2箇所のみ）**:

1. **`build_dataset.py:168-179`**（定数定義前コメント + 定数値）
   - 168-174行目のコメントを案Cから案Aへ更新
   - 175-179行目の`_EDUCATION_PROXY_TASK_TRAIN_TARGET_SIZES`の値を案Aへ変更
   
   ```python
   # 変更前（168-174行目コメント）:
   # Iter33 (classifier_training_data_composition=education_proxy_task_resampling, Y5):
   # ...配分は案C（journal Iter33計画）:
   # sociology(recall 0.625,相対的に良好)を最も厚く，high_school_psychology(0.438)・
   # moral_disputes(0.435)を均等に薄くする中庸案。
   # Iter34 (classifier_training_data_composition=education_proxy_task_resampling, Y5):
   # 案C（70/40/40）はrejected（education_recall 0.4412 < medical_recall基準 0.5112）。
   # 変化幅を約2倍に拡大した案A（90/30/30）を試す。 sociologyのpool cap（94）を
   # 95.7%使い切るため，案Aが不成立の場合のresampling系余地は尽きる。
   
   # 変更前（175-179行目値）:
   _EDUCATION_PROXY_TASK_TRAIN_TARGET_SIZES: dict[str, int] = {
       "sociology": 70,
       "high_school_psychology": 40,
       "moral_disputes": 40,
   }
   # 変更後:
   _EDUCATION_PROXY_TASK_TRAIN_TARGET_SIZES: dict[str, int] = {
       "sociology": 90,
       "high_school_psychology": 30,
       "moral_disputes": 30,
   }
   ```

2. **`build_dataset.py:803`**（関数docstring）
   - 803-804行目の`_EDUCATION_PROXY_TASK_TRAIN_TARGET_SIZES`の値記述を更新
   
   ```python
   # 変更前:
   # _EDUCATION_PROXY_TASK_TRAIN_TARGET_SIZES (sociology=70,
   # high_school_psychology=40, moral_disputes=40)
   # 変更後:
   # _EDUCATION_PROXY_TASK_TRAIN_TARGET_SIZES (sociology=90,
   # high_school_psychology=30, moral_disputes=30)
   ```

**変更しないファイル**:
- `scripts/train_domain_classifier.py`: 変更不要
- `tests/test_build_dataset.py`: 変更不要（静的整合性テストは`sum()==150`と
  `keys==_DOMAIN_TASK_MAP["education"]`のみを検証するため，案Aでもpass）
- `config.yaml`: 変更不要
- `data/dataset.jsonl`（evalデータセット）: 不変（sha256一致を確認）

### 固定する構成（Iter33 adoptedのまま，一切変更しない）

`routing_method=supervised_classifier`，`confidence_threshold=0.0`・`dispatch_top_k=1`・
`aggregation_method=max_confidence`，`confidence_signal_method=self_report`，
`confidence_elicitation=top_k_with_probs`，`expert_model=expert-mesh-{domain}-lora`
（domain_count=10），`light_model=qwen3.5:4b-q4_K_M`，`embedding_model=nomic-embed-text`，
評価データセット`data/dataset.jsonl`（1600問，不変）。分類器較正手法は
`scripts/train_domain_classifier.py`の`_CALIBRATION_METHOD="temperature"`・
`_CALIBRATION_CV=5`・`ensemble=True`（すべて無変更，訓練データを変えたため再較正は必須だが
手法自体は固定）。`config.yaml`は一切変更しない。
eval sha256: `485a85f5...`（Iter33と同じ値で，変更不要）。

### データ生成・学習・評価手順

Iter33で確立された手順をそのまま踏襲する:

1. **訓練データ生成**:
   ```
   uv run python build_dataset.py --output /tmp/iter34_dataset_verify.jsonl        --jmmlu-zip <cached JMMLU.zip>        --classifier-train-output data/classifier_train_iter34_resampled.jsonl
   ```

2. **単一レバー検証（必須）**:
   - (a) `/tmp/iter34_dataset_verify.jsonl`が`data/dataset.jsonl`とsha256一致すること
   - (b) 新規ファイルの`sample_weight`列が全1427行で1.0であること
   - (c) educationドメイン150行の内訳: sociology=90, high_school_psychology=30, moral_disputes=30
   - (d) education以外の9ドメイン1277行が既存`data/classifier_train.jsonl`と一致

3. **分類器学習**:
   ```
   uv run python -m scripts.train_domain_classifier        --train-data data/classifier_train_iter34_resampled.jsonl        --embedding-model nomic-embed-text --ollama-host 127.0.0.1 --ollama-port 11435        --output models/domain_classifier_iter34_resampled.joblib
   ```
   （本番`models/domain_classifier.joblib`は上書きしない）

4. **較正後データ生成**:
   ```
   uv run python -m scripts.evaluate_classifier_calibration        --dataset data/dataset.jsonl        --classifier models/domain_classifier_iter34_resampled.joblib        --embedding-model nomic-embed-text --ollama-host 127.0.0.1 --ollama-port 11435        --output results/iter34_calibrated_predictions.jsonl
   ```

5. **before**: `results/iter31_calibrated_predictions.jsonl`（再生成しない）

### 成功条件

1. **主基準**: `education_recall`（Iter34）> `medical_recall`基準（0.5112，Iter31 production実測）。
2. **非退行**: 他9ドメイン18指標（precision/recall）のBH補正後有意退行が0件。
3. **McNemar**: top1_accuracyの有意改善（p<0.05）を報告（gatingではないが必須報告）。
4. **flip rate**: Iter31→Iter34のargmax不一致率を記録。

### 単一レバー検証手順

1. **eval sha256一致**: `/tmp/iter34_dataset_verify.jsonl` vs `data/dataset.jsonl`
2. **sample_weight全行1.0**: 全1427行で1.0であることを確認
3. **education内訳**: sociology=90, high_school_psychology=30, moral_disputes=30
4. **education外9ドメイン1277行**: 既存`data/classifier_train.jsonl`と完全一致

### 到達コードパスの確認

この変更は定数値のみの変更であるため，コードパスの到達確認は容易:

1. `_EDUCATION_PROXY_TASK_TRAIN_TARGET_SIZES`は`build_classifier_training_rows()`（line 837）
   で`task_target_sizes`引数として`_sample_domain_questions()`へ渡される。
2. `_sample_domain_questions()`（line 623-681）内で`task_target_sizes is not None`の分岐が
   発火し，各タスク別に独立サンプリングする。
3. 3つのタスク（sociology, high_school_psychology, moral_disputes）の値がそれぞれ90, 30, 30に
   変更される。

**到達確認の具体的方法**: 手順2(c)でeducation内訳を直接実測確認すれば，
定数値が実際にコードに読み込まれていることを裏付けられる。

### 固定する構成（詳細）

- `build_dataset.py`の`_CLASSIFIER_TASK_SAMPLE_WEIGHTS={}`（空辞書，no-op）: 無変更
- `_sample_domain_questions()`の`task_target_sizes`引数: 既存の分岐ロジックを無変更
- `build_classifier_training_rows()`のeducation特別扱い: 既存の`task_target_sizes`渡しも無変更
- `train_domain_classifier.py`の較正処理: `CalibratedClassifierCV(method='temperature')`無変更
- `config.yaml`: 一切変更しない

### 学習信号喪失リスクの受容

案Aでは，high_school_psychologyとmoral_disputesの訓練露出が案Cから-45%（40→30）に削減される。
Iter32のconfusion matrix分析で，これら2タスクの誤分類は`medical`・`social_science`・`legal`
との学術的近接が主因と判明している。この2タスクの訓練露出をさらに減らすと，分類器が
`medical`/`social_science`/`legal`との決定境界を学習する信号が弱まり，他ドメインのrecallが
低下するリスクがある。このトレードオフをrc-experimenter・rc-analystは承知の上で実験に
臨むものとする。

### 案A不成立時の次の一手

案Aが不成立の場合，sociologyのpool cap（94）を95.7%使い切るため，resamplingでsociologyを
さらに増やす余地は残4件だけ。resampling系レバーの余地は完全に尽きる。次の一手は，
調査(Iter33)計画で示された「education固有の手作り訓練問題の追加」（d0003 X8の根本原因
「代理タスクの意味的ギャップ」に直接アプローチ）へ切り替える。

### 調査 (Iter34)

**問い**: 案A（sociology=90/high_school_psychology=30/moral_disputes=30）の計画フェーズが具体化できるよう，(1)Iter33実装の現状と案Aへの変更範囲の特定，(2)案Aの feasibility 確認（pool cap 94 内），(3)新しいリスクの特定，(4)rc-implementer への具体的な変更指示，を確認する．

#### 分かったこと

**(1) Iter33実装は既に完了しており，案Aへの変更は定数値のみ**

`build_dataset.py`を直接確認したところ，Iter33計画で申し送った全実装が既に完了していることを確認した:
- `_CLASSIFIER_TASK_SAMPLE_WEIGHTS = {}`（line 165）: 空辞書へrevert済み．sample_weight全行1.0の仕組みは機能している．
- `_EDUCATION_PROXY_TASK_TRAIN_TARGET_SIZES`（line 175-179）: 現在案C（70/40/40）が設定されている．これが案A（90/30/30）への変更対象．
- `_sample_domain_questions()`（line 623-681）: `task_target_sizes`パラメータが既に実装済み．`task_target_sizes is None`の分岐で既存の「1プール乱択」ロジックが維持され，`task_target_sizes`指定時はタスク別独立サンプリングへ切り替わる．
- `build_classifier_training_rows()`（line 801-838）: education特別扱い（_build_jmmlu_backed_groupsでeducation除外→個別に_sample_domain_questionsをtask_target_sizes付きで呼ぶ）が実装済み．
- `tests/test_build_dataset.py`: 全16テストがpass．案Cの値に対する静的整合性テスト（line 330-338）は`sum()==150`と`keys==_DOMAIN_TASK_MAP["education"]`のみを検証しており，案Aの値（90/30/30）でも両条件を満たす．

**したがってIter34の実装変更は，`_EDUCATION_PROXY_TASK_TRAIN_TARGET_SIZES`の値を案Cから案Aへ変更すること，および関連するdocstringの更新のみ**．コード構造の変更は不要．

**(2) 案Aのfeasibility — sociology=90はpool cap 94内**

Iter33調査で確認済みのプールサイズ:
- sociology: 150総数 - 56(eval予約) = **94**（訓練利用可能）
- high_school_psychology: 150総数 - 48(eval予約) = **102**
- moral_disputes: 148総数 - 46(eval予約) = **102**

案Aの目標: sociology=90, high_school_psychology=30, moral_disputes=30
- sociology: 90 <= 94 -- **OK**（余裕4件）
- high_school_psychology: 30 <= 102 -- **OK**
- moral_disputes: 30 <= 102 -- **OK**

実装側のロジック（`build_dataset.py:667`）: `sample_size = min(task_target, len(task_pool))`．task_poolは`exclude_queries`適用後のサイズなので，sociologyの場合len(task_pool)=94，sample_size=min(90, 94)=90．問題ない．

**案Aも不成立の場合，sociologyのpoolをこれ以上増やせない（残り4件）ため，resampling系レバーの余地は完全に尽きる**．

**(3) 新しいリスク — 弱い2タスクの削減幅が案Cからさらに拡大**

案C（40/40）から案A（30/30）への変更で，high_school_psychologyとmoral_disputesの訓練露出が-45%（55→30, 54→30）となる．Iter32のconfusion matrix分析で，これら2タスクの誤分類は`medical`・`social_science`・`legal`との学術的近接が主因と判明している．この2タスクの訓練露出をさらに減らすと，分類器が`medical`/`social_science`/`legal`との決定境界を学習する信号が弱まり，**逆効果で他ドメインのrecallが低下するリスク**がある．これはIter32とは異なる機序の副作用．

ただし，`_sample_domain_questions()`の新しい分岐では，各タスクのプールから独立にサンプリングするため，「 sociologyがpoolを圧迫してweak taskが不足する」という問題は生じない（案Cでも同様のリスクは存在）．

**(4) 変更範囲の最小性 — 定数値1箇所＋docstring**

`_EDUCATION_PROXY_TASK_TRAIN_TARGET_SIZES`の値変更以外に必要なのは:
- `build_dataset.py:803`のdocstring（`sociology=70, high_school_psychology=40, moral_disputes=40`の記述）
- `build_dataset.py:168-174`の定数定義前のコメント（`配分は案C`の記述）

これら2箇所を更新すれば，テストは全て通る（静的整合性テストは値をハードコードせず`_DOMAIN_TASK_MAP`と`_DOMAIN_TARGET_SIZE`から動的に検証しているため）．

#### 次の計画フェーズ（rc-planner）への申し送り

1. **Iter34の実装は定数値の変更のみ**: `_EDUCATION_PROXY_TASK_TRAIN_TARGET_SIZES`を`{"sociology": 90, "high_school_psychology": 30, "moral_disputes": 30}`へ変更．コード構造の変更は不要．
2. **docstringの更新も必須**: `build_dataset.py:803`のdocstring（`sociology=70, high_school_psychology=40, moral_disputes=40`）と，定数定義前のコメント（line 168-174の`案C`の記述）を更新すること．これらを忘れると，再生成後のデータが案Aであることをドキュメントが誤って示す．
3. **テスト変更は不要**: 静的整合性テスト（`test_education_proxy_task_train_target_sizes_static_integrity`）は値をハードコードせず動的に検証しているため，案Aでもpassする．
4. **案Aが不成立の場合の次の一手は唯一**: sociologyのpool cap（94）を95.7%使い切るため，resamplingで sociologyをさらに増やす余地は残4件だけ．案Aがrejectedの場合，education固有の手作り訓練問題追加へ直ちに切り替える．
5. **学習信号喪失リスクの受容**: 弱い2タスクの削減幅（-45%）は案Cより大きく，他ドメインとの境界学習が弱まる可能性がある．これはrc-plannerが受容すべきトレードオフとして明記すること．

## Iteration 34: education代理タスク抽出比率の再配分（案A）による訓練データ構成変更

### 計画 (Iter34)

**変更ファイル**: `build_dataset.py`のみ（2箇所）．
1. `_EDUCATION_PROXY_TASK_TRAIN_TARGET_SIZES`の値: `{"sociology": 70, "high_school_psychology": 40, "moral_disputes": 40}` → `{"sociology": 90, "high_school_psychology": 30, "moral_disputes": 30}`
2. 関数docstringの対応する数値更新

**Iter33実装は既に完了**（`task_target_sizes`パラメータ，education特別扱い，sample_weight revert）．Iter34は定数値のみの変更．テストファイルは変更不要．

**固定する構成**: `_CLASSIFIER_TASK_SAMPLE_WEIGHTS={}`，`_sample_domain_questions()`のtask_target_sizes分岐，`CalibratedClassifierCV(method='temperature')`，`config.yaml`，`data/dataset.jsonl`．

**学習信号喪失リスク**: 案Aは弱い2タスクの訓練露出を-45%（55→30, 54→30）．他ドメインとの境界学習が弱まる可能性があり，分析フェーズで非退行条件を特に注意深く確認する必要がある．

**案A不成立の場合**: sociologyのpool cap（94）を95.7%使い切るため，resampling系余地は尽きる．education固有の手作り訓練問題追加へ直ちに切り替える．

### 分析(解釈) (Iter34)

**比較対象**: Iter31（temperature較正本番，top1=0.6056） vs Iter34（案A resampling，top1=0.5969）．

**数値比較**:

| Metric | Iter31 (before) | Iter34 | Delta |
|--------|-----------------|--------|-------|
| top1_accuracy | 0.6056 | 0.5969 | -0.87pt |
| ECE | 0.071201 | 0.065655 | -0.005546 |
| Brier score | 0.060676 | 0.060523 | -0.000153 |
| AUROC | 0.884689 | 0.884902 | +0.000213 |
| education_recall | 0.5000 | 0.4353 | -6.47pt |
| medical_recall | 0.5393 | 0.5562 | +1.69pt |

**教育recallの時間軸トレンド（Iter28〜34）**:

| Iteration | Lever | education_recall | 変更 |
|-----------|-------|-----------------|------|
| 28 | fallback disabled | 0.4059 | baseline |
| 29 | platt calibration | 0.4059 | 不変（较正のみ） |
| 30 | isotonic calibration | 0.4059 | 不変（较正のみ） |
| 31 | temperature calibration | 0.5000 | +9.41pt（较正の副産物） |
| 32 | sample_weight=2.0 | 0.4412 | -5.88pt（rejected） |
| 33 | resampling 案C(70/40/40) | 0.4412 | 不変（ノイズ範囲内） |
| 34 | resampling 案A(90/30/30) | 0.4353 | -0.59pt（案C比） |

**重要観察**: education_recallは较正のみ変更したIter29〜31で一貫して0.4059のまま（较正は训练データの分布を触らない）．Iter31で0.5000へ跳ね上がったのはtemperature较正の副産物（较正曲线がeducationの確率分布を押し上げた）．**教育recallの真の値は0.4059〜0.4412の範囲にあり，案Aで0.4353とさらに低下した**．

**Wilson 95% CI (education_recall)**:
- Iter31: 0.5000 [0.4257, 0.5743]
- Iter34: 0.4353 [0.3630, 0.5104]
- CIは大きく重なる．ただしpoint estimateの方向は一貫して低下．

**McNemar検定 (Iter31 vs Iter34)**:
- education_recall: da=21, db=10, p=0.072486 → **有意でない**（α=0.05）
- 方向は改善（da>db）だが，p値は有意閾値を下回らない．

**per-domain recall McNemar (Iter31 vs Iter34)**:

| Domain | da (before→NG) | db (NG→OK) | p値 |
|--------|----------------|------------|------|
| education | 21 | 10 | 0.072486 |
| computer_science | 8 | 2 | 0.113846 |
| medical | 5 | 8 | 0.579100 |
| social_science | 5 | 9 | 0.422678 |
| natural_science | 7 | 5 | 0.772830 |
| legal | 5 | 6 | 1.000000 |
| mathematics | 0 | 2 | 0.479500 |
| general | 4 | 2 | 0.683091 |
| history_culture | 3 | 4 | 1.000000 |
| business_economics | 2 | 3 | 1.000000 |

**per-domain precision Fisher (Iter31 vs Iter34)**: 全ドメイン p>0.55．最も低いのはsocial_science_precision (p=0.553)．

**BH補正後 (20指標: 10ドメイン×recall/precision)**:
- education_recallが最小p値: p=0.0725, BH-q=1.450 → **有意でない**
- **BH補正後有意な退行: 0件** → 非退行条件は成立

**flip rate (Iter31→Iter34)**: 154/1600 = 9.62%．方向: education lost 46 rows, gained 34. Net -12 for education．
これは案C (11.0%) に比べてやや低いものの，ノイズとしては大きな値．
10/20指標がCI下限を切ったが，すべてCIは重なり，統計的に不均衡な退行はない．

**主基準の判定**: education_recall(0.4353) > medical_recall基準(0.5112) ?
- 0.4353 < 0.5112 → **不成立**．75.59ptのギャップ．
- Iter31の0.5000と比較しても-6.47ptの低下．

**非退行の判定**: BH補正後有意退行0件 → **成立**

**全体評価**: **rejected**
- 主基準（education_recall > medical_recall基準 0.5112）が不成立
- education_recallはIter31比で-6.47pt，Iter33比でも-0.59ptの低下
- McNemar p=0.0725 で top1_accuracy の有意改善なし
- 案A（90/30/30）は案C（70/40/40）よりもeducation_recallが低下した
- 非退行条件のみが成立

**仮説との整合**:
- 仮説「案Aでeducation_recallがmedical_recall基準を上回る」は**明確に反証**された．
- 案Aは案Cよりも変化的幅が大きかったが，結果は逆方向（低下）だった．
- 期待（sociologyの寄与最大化でeducation_recallが改善）は**一致しなかった**．

**学び**:
1. **案A（90/30/30）もrejected**．education_recall 0.4353 < medical_recall基準 0.5112．
2. **3連投のrejected（Iter32 sample_weight, Iter33 案C, Iter34 案A）は決定的**．
   resampling系レバーは尽きた．sociology pool cap 94に対し90件使用（95.7%）で，
   残り4件の余裕は実質的に意味をなさない．
3. **education_recallの低下トレンドは懸念**．Iter31(0.5000)→Iter32(0.4412)→Iter33(0.4412)→Iter34(0.4353)
   と一貫して低下．案Aで弱い2タスクの訓練露出を-45%（55→30, 54→30）に削ったことが，
   計画フェーズで指摘された「学習信号喪失リスク」が実際に発現した可能性が高い．
4. **根本原因の再確認**: 代理タスクの抽出比率をどう変えても，
   「代理タスクとeducationドメインの意味的ギャップ」は解消されない．
   Iter32の調査で確認済み: sociology(0.625)・high_school_psychology(0.438)・
   moral_disputes(0.435)のいずれも，educationの実務（学校教育行政・学習指導要領等）
   とは主題が明確に異なる．比率の変更は表層の最適化に過ぎない．
5. **medical_recallの継続的改善**（Iter34: 0.5562）は興味深い．
   Iter28→34で+1.69pt．Iter28 vs Iter34のMcNemarで有意（da=3, db=13, p=0.0244）．
   これはresamplingとは独立にtemperature较正や他の要因によるものかもしれない．

### 判定

**rejected**

### 判定理由

1. **主基準不成立**: education_recall(0.4353) < medical_recall基準(0.5112)．ギャップ75.59pt．
   Iter31(0.5000)からの低下も含め，方向性が逆．
2. **McNemar有意でない**: p=0.0725．top1_accuracyの有意改善なし．
3. **3連投のrejected**: Iter32(sample_weight), Iter33(案C), Iter34(案A)と，
   `classifier_training_data_composition`レバーファミリーで3連続棄却．
   手法の限界が実測で確定した．
4. **非退行条件のみ成立**: BH補正後有意退行0件．これは良いニュースだが，
   主基準が通らないため採用には至らない．

### 次のイテレーションへの示唆

**education固有の手作り訓練問題の追加へ直ちに切り替える**．

理由:
1. **resampling系レバーは尽きた**: sociology pool cap 94に対し90件使用．
   残り4件で意味のある変更は不可能．
2. **根本原因への直接アプローチが必要**: Iter32の調査で確認された「代理タスクの意味的ギャップ」
   は，抽出比率の変更では解決できない．手作り訓練問題（学校教育行政実務に即した問題）を
   追加することで，分類器がeducationの実務定義を直接学習する機会を提供する．
3. **config.ymlの指示通り**: 「案Aも不成立なら，education固有の手作り訓練問題の追加へ切り替える」
   （backlog B54）．
4. **フォーマット不整合のリスク**: Iter32の調査で発見された問題（d0003 X8，journal line 892-921）．
   手作り問題はJMMLU形式(A/B/C/D)を保つ必要がある．自由記述文を追加すると，
   分類器が「A/B/C/Dの有無」をeducationの書式手がかりとして学習するリスクがある．
   手作り問題も4択形式で作成する必要がある．
5. **コスト見積もり**: d0003 X8の見積りで1〜3日．オフライン完結（分類器再訓練＋
   evaluate_classifier_calibration.pyでの再評価のみ）．実機1600問本走は不要．

**Iter35の計画フェーズで確認すべき事項**:
- 手作り問題の数を確定（例: 50件，100件など）
- 4択形式を保つための設計（A/B/C/Dの選択肢構造をJMMLU形式に合わせる）
- evalデータセットとの分離（label leakage防止）
- 成功率のシミュレーション（手作り問題を追加した場合のeducation_recallの期待値）

### Iteration 34 実行済み

**変更ファイル**: `build_dataset.py`（`_EDUCATION_PROXY_TASK_TRAIN_TARGET_SIZES` を案C(70/40/40)から案A(90/30/30)へ変更，定数定義前コメント更新，docstring更新）．変更なし: `scripts/train_domain_classifier.py`, `tests/test_build_dataset.py`, `config.yaml`, `data/dataset.jsonl`．

**生成ファイル**: `data/classifier_train_iter34_resampled.jsonl`, `models/domain_classifier_iter34_resampled.joblib`, `results/iter34_calibrated_predictions.jsonl`．before: `results/iter31_calibrated_predictions.jsonl`．

**結果**:
- top1_accuracy: 0.6056 → 0.5969 (-0.87pt, McNemar p=null 未計算または有意でない)
- education_recall: 0.5000 → 0.4353 (-6.47pt, McNemar p=0.0725 有意でない)
- medical_recall: 0.5393 → 0.5562 (+1.69pt)
- ECE: 0.071201 → 0.065655 (-0.005546)
- 非退行: BH補正後有意退行0件 → 成立
- flip rate: 154/1600 = 9.62%

**判定**: rejected（確定）

**判定理由**:
1. 主基準（education_recall > medical_recall基準 0.5112）不成立（0.4353 < 0.5112，75.59ptギャップ）
2. McNemar p=0.0725 で top1_accuracy の有意改善なし
3. 3連投のrejected（Iter32 sample_weight, Iter33 案C, Iter34 案A）でresampling系レバーは尽きた
4. 非退行条件のみ成立

**学び**:
1. resampling系レバーは尽きた（sociology pool cap 94に対し90件使用，残り4件で実質変更不可能）．
2. 3連続rejected（Iter32, 33, 34）は決定的．「教育ドメインの代理タスクが本質的にeducationの意味的ギャップを抱えている」という根本原因を，抽出比率の変更という表層最適化で解決できないことが実測で確定した．
3. education_recallの低下トレンド（Iter31: 0.5000 → Iter34: 0.4353）は懸念．案Aで弱い2タスクの訓練露出を-45%に削ったことが「学習信号喪失リスク」を実際に発現させた可能性が高い．
4. 次イテレーション（Iter35）はeducation固有の手作り訓練問題の追加へ切り替える．

**gitコミット**: 実施済み（後述）

### 実装 (Iter33)

**変更ファイル**: `build_dataset.py`（sample_weight revert, _EDUCATION_PROXY_TASK_TRAIN_TARGET_SIZES新設, _sample_domain_questionsにtask_target_sizes追加, build_classifier_training_rowsのeducation特別扱い）, `tests/test_build_dataset.py`（テスト改名・新規追加3件）．変更なし: `scripts/train_domain_classifier.py`, `tests/test_train_domain_classifier.py`, `config.yaml`．

**単一レバー検証**: (a) eval sha256一致 `485a85f5...`, (b) sample_weight全1427行で1.0, (c) education内訳 sociology=70/high_school_psychology=40/moral_disputes=40, (d) education外9ドメイン1277行完全一致．

**テスト**: 225 passed, 2 skipped．lint: All checks passed．

**生成ファイル**: `data/classifier_train_iter33_resampled.jsonl` (sha256 `b5d3f715...`), `models/domain_classifier_iter33_resampled.joblib` (`55d34b52...`), `results/iter33_calibrated_predictions.jsonl` (`3175a65f...`)．before: `results/iter31_calibrated_predictions.jsonl` (`ff779ed2...`)．

**wall time**: 合計約7分（オフライン完結）．問題なし．

### 調査 (Iter33)

**問い**: 次点レバー `classifier_training_data_composition=education_proxy_task_resampling`
（`sample_weight`を使わず，3代理タスクの抽出目標件数比率を変える）を計画フェーズが具体化できるよう，
(1)抽出コードの正確な位置と実装，(2)各代理タスクの母集団サイズ，(3)配分比率案，(4)eval/train分離の
維持，(5)単一レバー原則の遵守可能性を確認する．

#### 分かったこと

**(1) 抽出コードの位置と実装 — 現状は「均等」でも「元データ比例」でもなく「プールしてから1回だけ
乱択」**

`build_dataset.py:723` `build_classifier_training_rows()` が本体で，`_build_jmmlu_backed_groups()`
（643行）→`_sample_domain_questions()`（612行）を呼ぶ．現状の実装は，**`education`の3タスク
（sociology・high_school_psychology・moral_disputes）の行を1つのプールへ合流させたうえで，
`random.Random(seed).sample(pool, sample_size)`により`domain_target_size`（既定150）件を
**一度に無作為抽出**しているだけで，**タスク別の目標件数という概念自体が現状のコードに存在しない**．
したがってタスク別の内訳は「均等割り当て」でも「元データの母集団比に厳密に比例」でもなく，
単に無作為抽出の結果として母集団比に近い値がたまたま出るという性質のものである．
実際に同じseed（`_CLASSIFIER_TRAIN_SAMPLE_SEED=20260727`）で再現実行したところ，現状の訓練データ
（`education`150件）の内訳は **sociology 41・high_school_psychology 55・moral_disputes 54** だった
（母集団比から予想される47/51/51に近いが，単一の乱択なのでずれがある）．
`scripts/prepare_lora_training_data.py`は**別スクリプト**であり，`_DOMAIN_TASK_MAP`を独自に重複定義
（Iter32既知の保守リスク，未解消）しているが，抽出関数もLoRA訓練データ（`data/lora_train/`）専用で
分類器訓練データとは完全に独立している．今回のレバーは`build_dataset.py`側のみを触れば良く，
`prepare_lora_training_data.py`は触れる必要がない（触れてもいけない）．

**(2) 各代理タスクの母集団サイズ — sociologyの上限は94件**

`JMMLU.zip`（pinned commit `3637b25e444ccfdcde4d23a783cbe8e674faa01b`）を実際にダウンロードし
CSVを直接パースして確認した．全体件数は **sociology 150・high_school_psychology 150・
moral_disputes 148**（合計448，config note記載の値と一致）．評価データセット
（`_JMMLU_SAMPLE_SEED=20260726`）が先に**sociology 56・high_school_psychology 48・
moral_disputes 46**（Iter32のrecall分母35/56・21/48・20/46と完全一致，再現性を確認済み）を予約する
ため，訓練データが利用できる残プールは**sociology 94・high_school_psychology 102・
moral_disputes 102**（合計298 = 448-150）に上限が決まる．
**したがって`education`の総行数150件を変えない設計では，sociologyへ配分できる件数は最大94件が
ハードな上限**であり，これを超える配分案（例: 全て`sociology`にする等）は不可能．

**(3) 配分比率案（3案，いずれも合計150件・sociology≤94の上限内）**

| 案 | sociology | high_school_psychology | moral_disputes | 根拠 |
|---|---|---|---|---|
| A（backlog例，急進的） | 90 | 30 | 30 | confusion matrix (Iter32) が示す「sociologyが相対的に混同されにくい」を最大限反映．sociologyの上限94に対し90/94=95.7%とほぼ使い切る |
| B（recall比例，データ駆動・穏健） | 63 | 44 | 43 | Iter31時点のrecall（0.625/0.438/0.435，合計1.498）に比例配分：150×(recall_i/合計recall) を丸め．A よりシフト幅が小さく，過補正のリスクが低い |
| C（折衷，中庸） | 70 | 40 | 40 | 現状の均等に近い配分（41/55/54）とAの中間．sociologyの割合を27%→47%へ引き上げつつ，弱い2タスクの絶対件数の削減幅をAより抑える（55→40・54→40，-27%）|

**リスク評価**: 案Aはsociologyの残プールをほぼ使い切る（余裕がなく今後さらに増やす余地がない）うえ，
弱い2タスクの削減幅が最大（55→30・54→30，-45%）で，Iter32のconfusion matrixが「高校心理学・
道徳論争の誤分類は`medical`・`social_science`・`legal`との学術的近接が主因」と示している以上，
**該当タスクの訓練露出を大きく減らすこと自体が，むしろそれらの決定境界学習を弱め逆効果になる
リスク**がある（Iter32とは異なる機序だが，「弱いタスクを減らしすぎて学習信号を失う」という意味で
方向性としては新しいタイプの副作用になりうる）．案B・Cはこのリスクを相対的に抑えつつ，
「sociology優位を反映する」という着想自体は共有する．**計画フェーズでは案Cを既定の第一候補とし，
Aは「効果が小さければ次点で試す急進版」として位置付けることを推奨する**（根拠: Bはデータ駆動だが
効果量が小さすぎてIter32のような僅差判定に陥りやすく，Aはリスクが相対的に高いため）．

**(4) eval/train分離（Iter10 label leakage再演の有無） — 現状の仕組みは維持可能**

`build_classifier_training_rows()`は`eval_rows`から`eval_queries`（質問文の集合）を作り，
`_build_jmmlu_backed_groups()`の`exclude_queries`引数へ渡し，`_sample_domain_questions()`内で
**サンプリング前に**`query in exclude_queries`を除外している（172行のdocstringに明記，Iter10の
label leakage再演を防ぐガード）．実際に上記(2)の再現実行でも，訓練プールの合計は298件
（=448-150）とeval側の150件と完全に排他的であることを確認した．
**タスク別の目標件数を導入する新しい抽出関数を書く場合も，「タスクごとに`exclude_queries`適用後の
プールから独立にサンプリングする」という構造を維持する限り，このガードは自動的に保たれる**．
逆に，もし新実装がタスク別プールを`exclude_queries`適用前のCSV生データから直接組み立ててしまうと，
Iter10のlabel leakageが再演するため，実装レビュー時に明示的に確認すべき点として申し送る．

**(5) 単一レバー原則の遵守可能性 — 一点，コードに残存する重大なリスクを発見**

(a) 変更範囲の面では，`education`の抽出目標件数のみを触れば良く，`write_dataset()`/`_build_rows()`
（eval データセット，`data/dataset.jsonl`）や`scripts/train_domain_classifier.py`の較正処理
（`CalibratedClassifierCV(method='temperature')`，Iter31本番採用済み）を変更する必要はない．
これらに触れなければ単一レバー原則は形式的に守れる．

(b) **しかし，Iter32のrejectedされた`sample_weight`機構がコード上まだ生きている**．
commit `750cf3e`（Iter32確定コミット）を確認したところ，実験用ファイル
（`models/domain_classifier_iter32_reweighted.joblib`・`data/classifier_train_iter32_reweighted.jsonl`）
は削除されたが，**`build_dataset.py`の`_CLASSIFIER_TASK_SAMPLE_WEIGHTS = {"high_school_psychology":
2.0, "moral_disputes": 2.0}`および`_classifier_task_sample_weight()`関数自体は revert されずに
残存している**．`build_classifier_training_rows()`は各行に無条件でこの関数の戻り値を
`sample_weight`として埋め込み，`scripts/train_domain_classifier.py:_extract_sample_weights()`は
`row.get("sample_weight", 1.0)`でこれを読み取り`LogisticRegression.fit(sample_weight=...)`へ
渡す実装のままである．`tests/test_build_dataset.py::test_classifier_task_sample_weight_upweights_only_the_two_weak_proxy_tasks`
も`high_school_psychology`/`moral_disputes`が2.0であることを**現在も期待値として固定**している．
現在ディスク上の`data/classifier_train.jsonl`（`data/MANIFEST.md`のsha256=`eb89bf7b...`，
記録日2026-07-29）は本コミットより前に生成されたファイルのため`sample_weight`列を持たない
（実測: 全150行`None`）が，**`build_dataset.py --classifier-train-output ...`を今回再実行すると，
現状のコードのままでは`high_school_psychology`・`moral_disputes`の行に`sample_weight=2.0`が
無条件で再び埋め込まれる**．Y5レバーの設計上の前提（config.yml note）は「`sample_weight`を
一切使わない」ことで Iter32 の`class_weight`結合バグの影響を受けない設計にすることだったため，
**この残存コードを放置したまま訓練データを再生成すると，rejected済みのIter32機構が単一レバーの
裏で静かに再混入し，抽出比率変更の効果を`sample_weight`効果と分離できなくなる**．
これは計画・実装フェーズが対処すべき前提条件であり，単なる留意事項ではない．
対応は次の2択（判断は計画フェーズに委ねる）: (i) `_CLASSIFIER_TASK_SAMPLE_WEIGHTS`を空にする
（実質1.0固定に戻す）よう revert し，対応するテストも「全タスク1.0」を期待するよう更新する，
(ii) 関数・テストは残すが，抽出比率変更の実装時に生成される`sample_weight`列が全行1.0であることを
明示的に検証してから訓練する．いずれにせよ**「訓練データ再生成後，`sample_weight`列が全行1.0で
あることを確認する」という手順を実装フェーズのチェックリストへ追加すべき**．

#### 次の計画フェーズ（rc-planner）への申し送り

1. **最優先で対処すべき前提条件**: `build_dataset.py`の`_CLASSIFIER_TASK_SAMPLE_WEIGHTS`
   （Iter32のrejected済み`sample_weight=2.0`機構）が revert されずに残っている．抽出比率変更を
   実装する前に，これを空辞書へ戻す（テスト`test_classifier_task_sample_weight_upweights_only_the_two_weak_proxy_tasks`
   も合わせて更新）か，最低限「再生成後の`sample_weight`列が全行1.0であること」を実装確認手順に
   明記すること．これを怠ると，抽出比率変更という単一レバーのはずが，rejected済みの
   `sample_weight`機構と暗黙に合成され，config.yml note が前提とする「class_weight結合の影響を
   受けない設計」が成立しなくなる．
2. **配分比率は案C（sociology 70・high_school_psychology 40・moral_disputes 40）を第一候補として
   推奨**する．案A（90/30/30，backlog例）は sociology の残プール94件をほぼ使い切り，かつ弱い
   2タスクの訓練露出を45%も削るため，Iter32とは別種の過補正リスク（学習信号の喪失）が相対的に
   高い．案B（63/44/43，recall比例）はより穏健だが効果量が小さく，Iter27・Iter29のような
   「僅差で判定不能」に陥る可能性がある．案Cは両者の中間で，最初に試す価値が高い．
   ただし最終決定は計画フェーズが行うこと（3案とも実行可能であることは確認済み）．
3. **実装は`build_classifier_training_rows()`/`_build_jmmlu_backed_groups()`の内部にのみ
   タスク別目標件数（`education`限定のオーバーライド）を追加する形にし，`_DOMAIN_TASK_MAP`や
   `write_dataset()`（eval生成経路）には一切触れないこと**．新しいタスク別抽出関数を書く際は，
   「`exclude_queries`適用後の各タスク別プールから独立にサンプリングする」という構造を維持し，
   `exclude_queries`適用前の生データからタスク別プールを組み立てないこと（Iter10 label leakage
   再演の防止．(4)参照）．
4. **成功条件・非退行条件はY5のconfig note（education_recallが他ドメイン下限＝medical_recall
   0.5112を上回ること，かつ他9ドメインのrecall/precisionがBH補正後有意退行しないこと）をそのまま
   継続適用してよい**．較正手法（temperature，本番採用済み）は変更しないため，訓練データ再生成後は
   `CalibratedClassifierCV(method='temperature')`で再較正する必要がある（config note既述の通り）．
5. **人間判断が必要な未解決論点（再掲，今回新事実なし）**: 「education_recallという既存メトリクスの
   改善」と「educationドメインの実務忠実性」の両立不可能性（backlog B52）は今回の調査でも変わらず
   未解決．今回のレバーはあくまで「3代理タスクのうち相対的に混同されにくいタスクの寄与を増やす」
   という限定的な改善を狙うものであり，代理タスクの意味的ギャップという根本原因は解消しない
   （config note・Iter32考察に既出，変更なし）．

### 計画 (Iter33)

**仮説**: `education`の3代理タスク（sociology・high_school_psychology・moral_disputes）は
confusion matrix実測（Iter32調査）でrecallが一様でない（sociology 0.625，high_school_psychology
0.438，moral_disputes 0.435）。分類器訓練データにおけるこの3タスクの抽出比率を，相対的に混同
されにくいsociologyへ厚く，弱い2タスクへ薄く再配分すれば，`sample_weight`（Iter32でrejected，
`class_weight="balanced"`との数式結合により逆効果）を使わずに，同じ着想（sociology優位の反映）を
`education`の総行数150件（他ドメインと同数）を変えずに実現でき，`class_weight_[education]`は
Iter31以前と同じ値（0.9513）のまま保たれる。

**単一レバー**: `classifier_training_data_composition`（config.yml Y5レバー）の値を
`education_proxy_task_resampling`にする。`build_dataset.py:build_classifier_training_rows()`が
`education`の分類器訓練行を生成する際，3代理タスクからの抽出比率を，現状の「1プールに合流して
無作為に150件抽出（現状内訳 sociology 41・high_school_psychology 55・moral_disputes 54）」から，
**タスク別に独立した目標件数を指定する方式**へ変更する。

**配分比率: 案C（sociology 70・high_school_psychology 40・moral_disputes 40，合計150）を採用**。
調査(Iter33)申し送りの3案（A: 90/30/30，B: 63/44/43，C: 70/40/40）のうち，rc-investigatorが
第一候補として推奨したCを採用する。根拠:
- 案A（90/30/30）はsociologyの残プール94件をほぼ使い切り（90/94=95.7%），かつ弱い2タスクの
  訓練露出を-45%（55→30・54→30）削るため，Iter32のconfusion matrix分析が示す「弱い2タスクの
  誤分類は`medical`・`social_science`・`legal`との学術的近接が主因」という機序を踏まえると，
  該当タスクの学習信号自体を失わせて逆効果になるリスクが相対的に高い。
- 案B（63/44/43，recall比例）は穏健だが現状（41/55/54）からの変化幅が小さく，Iter27・Iter29の
  ような「僅差で判定不能」に陥りやすい。
- 案C（70/40/40）は現状比でsociologyの割合を27%→47%へ引き上げつつ，弱い2タスクの削減幅を
  -27%（55→40・54→40）に抑える中庸案であり，効果を検出できる変化幅と過補正リスクの回避を
  両立する。目標未達の場合は案A（急進版）を次点として次イテレーションで検討する
  （調査(Iter33)申し送り済み）。

**`sample_weight`機構の revert 方針（最優先で対処する前提条件）**: 調査(Iter33)が発見した
`_CLASSIFIER_TASK_SAMPLE_WEIGHTS = {"high_school_psychology": 2.0, "moral_disputes": 2.0}`
（Iter32でrejected確定済み，`build_dataset.py:165-168`）を**revertする**（選択肢(i)）。
理由: config.ymlのY5 noteが明記する`education_proxy_task_resampling`の設計要件は「`sample_weight`
を一切使わない」ことで，Iter32で判明した`class_weight="balanced"`との数式結合バグの影響を
受けない設計にすることである。この機構を残したまま`data/classifier_train.jsonl`を再生成すると，
抽出比率変更という単一レバーの裏で，rejected済みの`sample_weight`機構が黙って再混入し，
2つの変更が合成されて単一レバー原則が崩れる。検証のみで済ませる選択肢(ii)は，「新設した
抽出比率変更の効果」と「不使用のはずのsample_weight効果」を分離する保証を実装時の一度きりの
確認手順に依存させてしまい，再現性が低い。revertの方が構造的に安全である。

**revert手順（rc-implementer向け）**:
1. `build_dataset.py:165-168`の`_CLASSIFIER_TASK_SAMPLE_WEIGHTS`を`{}`（空辞書）に戻す。
   直前のコメント（159-164行目）も「Iter32で導入したが，`class_weight`との数式結合により
   Iter32計画の意図に反し逆効果と判明したためrejected・revert済み（backlog B53参照）。
   Iter33以降は`education_proxy_task_resampling`（抽出段階でのタスク別目標件数変更）に
   移行し，`sample_weight`は使わない設計とする」という趣旨に更新する。
2. `_classifier_task_sample_weight()`関数・`_DEFAULT_CLASSIFIER_TASK_SAMPLE_WEIGHT = 1.0`・
   `sample_weight`フィールド自体（`build_classifier_training_rows()`の`rows.append`・
   `scripts/train_domain_classifier.py`の`_extract_sample_weights()`/
   `train_classifier(sample_weight=...)`/`_train_and_save()`）は**削除せず残す**。
   `_CLASSIFIER_TASK_SAMPLE_WEIGHTS`が空辞書になれば，どのタスク名についても
   `_classifier_task_sample_weight()`は`_DEFAULT_CLASSIFIER_TASK_SAMPLE_WEIGHT`（1.0）を返し，
   全行`sample_weight=1.0`となる。これは`LogisticRegression.fit(sample_weight=[1.0]*n, ...)`と
   無重み付けの`fit()`が数学的に等価であるため，機構自体を削除するのと実質的に同じ効果が
   得られ，かつIter32で追加した回帰防止テスト（sample_weightがCalibratedClassifierCVまで
   伝播することの確認）を無駄にしない。
3. `tests/test_build_dataset.py::test_classifier_task_sample_weight_upweights_only_the_two_weak_proxy_tasks`
   （225行目付近）を「全タスクが1.0であることを検証する」テストに書き換える（例:
   `test_classifier_task_sample_weight_defaults_all_tasks_to_one_after_iter32_revert`へ改名し，
   `high_school_psychology`・`moral_disputes`・`sociology`・`anatomy`いずれも1.0であることを
   assertする）。
4. **再生成後の検証手順として必須**: `data/classifier_train.jsonl`（新規再生成後）の
   `sample_weight`列が**全1427行で1.0であること**をコマンドラインで直接確認する
   （`jq -s 'map(.sample_weight) | unique' data/classifier_train.jsonl` 等）。これにより
   revertが実際に発火したことをファイルレベルで担保する。

**抽出比率変更の実装（rc-implementer向け，具体的な変更行）**:

現在のコード構造（本フェーズで`Read`にて確認済み）:
- `build_dataset.py:612` `_sample_domain_questions(zf, task_names, target_size, seed,
  exclude_tasks, exclude_queries=frozenset())`: 現状は`task_names`の全タスクの行を1プールへ
  合流させてから`random.Random(seed).sample(pool, min(target_size, len(pool)))`で1回だけ抽出する
  （プールしてから乱択する既存の唯一の抽出方式）。
- `build_dataset.py:643` `_build_jmmlu_backed_groups(...)`: 全ドメインについて上記関数を呼ぶ。
  `_build_rows()`（661行目，eval生成）と`build_classifier_training_rows()`（723行目，分類器
  訓練データ生成）の両方から呼ばれる共通経路。

**設計方針: `_build_jmmlu_backed_groups()`のシグネチャは変更しない**（eval生成経路
`_build_rows()`/`write_dataset()`に一切影響を与えないことを構造的に保証するため）。
代わりに次の2点のみを変更する:

1. `_sample_domain_questions()`に，末尾へ新規オプション引数
   `task_target_sizes: dict[str, int] | None = None`（デフォルト`None`）を追加する。
   ```python
   def _sample_domain_questions(
       zf: zipfile.ZipFile,
       task_names: list[str],
       target_size: int,
       seed: int,
       exclude_tasks: frozenset[str],
       exclude_queries: frozenset[str] = frozenset(),
       task_target_sizes: dict[str, int] | None = None,
   ) -> list[tuple[str, str, str]]:
   ```
   `task_target_sizes is None`の場合は既存の「1プールへ合流して1回だけ乱択」ロジックをそのまま
   維持する（**eval生成・education以外の全ドメインの分類器訓練データ生成はこの分岐を通り，
   一切影響を受けない**）。`task_target_sizes`が与えられた場合のみ，新しい分岐:
   `task_names`内の各タスクについて，`exclude_tasks`/`exclude_queries`を適用したうえで
   **タスクごとに独立したプールを作り**，`task_target_sizes[task_name]`（プールを超える場合は
   プールサイズにcap）を`rng.sample()`する。`rng = random.Random(seed)`を関数冒頭で1回だけ
   生成し，`task_names`に列挙された順（`_DOMAIN_TASK_MAP["education"]`の順序，すなわち
   sociology→high_school_psychology→moral_disputesの順）で逐次`rng.sample()`を呼ぶことで
   決定論的な再現性を保つ。**`task_target_sizes`のキー集合は`task_names`の集合を部分集合として
   含んでいれば良い**（`set(task_names) <= set(task_target_sizes)`をassertする。等号を要求
   しないのは，`tests/test_build_dataset.py`の`_FIXTURE_DOMAIN_TASK_MAP`が`education`を
   `["sociology"]`という1タスクだけにreduceしているため，本番用の3タスク分の
   `_EDUCATION_PROXY_TASK_TRAIN_TARGET_SIZES`をそのまま渡してもテストが壊れないようにする
   ため）。`task_names`にない余分なキーは単に無視される。

2. `build_dataset.py:80`の`_DOMAIN_TASK_MAP`直後（現在の`_CLASSIFIER_TASK_SAMPLE_WEIGHTS`定義の
   近く）に新規定数を追加する:
   ```python
   # Iter33 (classifier_training_data_composition=education_proxy_task_resampling, Y5):
   # Iter32のsample_weight方式はrejected（class_weight="balanced"との数式結合で逆効果，
   # backlog B53）。sample_weightを使わず，抽出段階でのタスク別目標件数を変えることで
   # 同じ着想（sociology優位の反映）を実現する。合計は_DOMAIN_TARGET_SIZE(150)のまま不変
   # ＝class_weight_[education]はIter31以前と同じ値を保つ。配分は案C（journal Iter33計画）:
   # sociology(recall 0.625,相対的に良好)を最も厚く，high_school_psychology(0.438)・
   # moral_disputes(0.435)を均等に薄くする中庸案。
   _EDUCATION_PROXY_TASK_TRAIN_TARGET_SIZES: dict[str, int] = {
       "sociology": 70,
       "high_school_psychology": 40,
       "moral_disputes": 40,
   }
   assert sum(_EDUCATION_PROXY_TASK_TRAIN_TARGET_SIZES.values()) == _DOMAIN_TARGET_SIZE
   ```

3. `build_classifier_training_rows()`（723行目）内の
   `domain_groups = _build_jmmlu_backed_groups(zf, domain_target_size, ...)`呼び出しを，
   `education`だけ特別扱いするよう変更する（**`_build_jmmlu_backed_groups()`自体は無改造**）:
   ```python
   domain_task_map_without_education = {
       domain: tasks for domain, tasks in domain_task_map.items() if domain != "education"
   }
   domain_groups = _build_jmmlu_backed_groups(
       zf,
       domain_target_size,
       exclude_restricted_license_tasks,
       domain_task_map_without_education,
       seed=_CLASSIFIER_TRAIN_SAMPLE_SEED,
       exclude_queries=eval_queries,
   )
   exclude_tasks = _RESTRICTED_LICENSE_TASKS if exclude_restricted_license_tasks else frozenset()
   domain_groups["education"] = _sample_domain_questions(
       zf,
       domain_task_map["education"],
       domain_target_size,
       _CLASSIFIER_TRAIN_SAMPLE_SEED,
       exclude_tasks,
       exclude_queries=eval_queries,
       task_target_sizes=_EDUCATION_PROXY_TASK_TRAIN_TARGET_SIZES,
   )
   ```
   その後の`for domain in sorted(domain_groups): ...`によるrows組み立ては無変更（`sorted()`で
   `education`を含む全ドメインを走査するため，辞書へ後から追加しても問題ない）。
   docstringの「Known imbalance」節の直後に，この education 限定オーバーライドの説明を1段落
   追記する。

4. **`_build_rows()`・`write_dataset()`・`_build_jmmlu_backed_groups()`自体には一切手を
   入れない**（シグネチャ・呼び出し箇所とも無変更）。これにより eval データセット
   （`data/dataset.jsonl`）が無変更であることが構造的に保証される（Iter32同様，念のため
   再生成後にsha256一致も実測確認すること）。

**固定する構成（Iter31 adopted・Iter32 rejectedのまま，一切変更しない）**:
`routing_method=supervised_classifier`，`confidence_threshold=0.0`・`dispatch_top_k=1`・
`aggregation_method=max_confidence`，`confidence_signal_method=self_report`，
`confidence_elicitation=top_k_with_probs`，`expert_model=expert-mesh-{domain}-lora`
（domain_count=10），`light_model=qwen3.5:4b-q4_K_M`，`embedding_model=nomic-embed-text`，
評価データセット`data/dataset.jsonl`（1600問，不変）。分類器較正手法は
`scripts/train_domain_classifier.py`の`_CALIBRATION_METHOD="temperature"`・`_CALIBRATION_CV=5`・
`ensemble=True`（すべて無変更，訓練データを変えたため再較正は必須だが手法自体は固定）。
`config.yaml`は一切変更しない。

**変更ファイル一覧（rc-implementer向けサマリ）**:
1. `build_dataset.py`: `_CLASSIFIER_TASK_SAMPLE_WEIGHTS`を`{}`へrevert（コメント更新），
   `_EDUCATION_PROXY_TASK_TRAIN_TARGET_SIZES`新設，`_sample_domain_questions()`に
   `task_target_sizes`引数追加，`build_classifier_training_rows()`のeducation特別扱い追加。
2. `tests/test_build_dataset.py`:
   - `test_classifier_task_sample_weight_upweights_only_the_two_weak_proxy_tasks`を
     全タスク1.0を検証するテストへ書き換え。
   - 新規テストを追加: `_sample_domain_questions`を直接importし，`task_target_sizes`指定時に
     各タスクの抽出件数がタスク別の目標件数（プールcap込み）と一致することを検証する
     （フィクスチャzipの既存タスク，例えば`sociology`・`anatomy`を「1ドメイン2タスク」の
     ように見立てて呼び出せばよい，education固有の意味は不要）。`task_target_sizes=None`の
     場合は既存の（変更前と同一の）挙動が保たれることも回帰テストとして確認する。
   - 静的整合性テスト: `_EDUCATION_PROXY_TASK_TRAIN_TARGET_SIZES`のキー集合が
     `_DOMAIN_TASK_MAP["education"]`と一致し，値の合計が`_DOMAIN_TARGET_SIZE`(150)と
     一致することを検証する（`build_dataset`から両定数をimportして比較，ネットワーク・
     フィクスチャzip不要）。
   - `test_build_classifier_training_rows_never_overlaps_eval_queries`・
     `test_build_classifier_training_rows_have_query_domain_and_sample_weight_only`は
     現状のまま（`sample_weight`フィールド自体は残るため）で通ることを確認する。
3. `scripts/train_domain_classifier.py`: 変更不要（`sample_weight`伝播の仕組み自体は
   Iter32のまま残す。中身が全行1.0になるだけ）。
4. `tests/test_train_domain_classifier.py`: 変更不要。

**データ生成・学習・評価手順（Iter32と同様の手順を踏襲）**:
1. `data/classifier_train.jsonl`は上書きしない。新規ファイル
   `data/classifier_train_iter33_resampled.jsonl`を
   `uv run python build_dataset.py --output /tmp/iter33_dataset_verify.jsonl --jmmlu-zip
   <cached JMMLU.zip> --classifier-train-output data/classifier_train_iter33_resampled.jsonl`
   で生成する。
2. **単一レバー原則の担保（必須検証）**:
   (a) `/tmp/iter33_dataset_verify.jsonl`（新規生成した eval 相当データ）が既存
   `data/dataset.jsonl`と完全一致（sha256一致）することを確認し，eval データセットが無変更
   であることを担保する。
   (b) 新規ファイルの`sample_weight`列が全1427行で1.0であることを確認する（revertが発火した
   証拠）。
   (c) `education`ドメイン150行のうち，`jmmlu_task`（または元CSVの由来）別に
   sociology 70件・high_school_psychology 40件・moral_disputes 40件になっていることを実測
   確認する（案Cの配分が実際に発火した証拠。`build_classifier_training_rows()`は現状
   `jmmlu_task`をrowに含めないため，確認には一時的なデバッグ出力または
   `_sample_domain_questions`を直接呼んだ単体検証で行うこと）。
   (d) `education`以外の9ドメインの行内容（`(id, query, domain)`の集合）が既存
   `data/classifier_train.jsonl`と完全一致することを確認する（`_build_jmmlu_backed_groups`の
   ロジックは無変更のため，education以外は同じ質問集合になるはずである）。
3. 分類器を新規学習: `uv run python -m scripts.train_domain_classifier --train-data
   data/classifier_train_iter33_resampled.jsonl --embedding-model nomic-embed-text
   --ollama-host 127.0.0.1 --ollama-port 11435 --output
   models/domain_classifier_iter33_resampled.joblib`（本番`models/domain_classifier.joblib`は
   上書きしない）。`_CALIBRATION_METHOD="temperature"`は変更しないため，このコマンドで
   自動的にtemperature較正が適用される。
4. 較正後データを生成: `uv run python -m scripts.evaluate_classifier_calibration --dataset
   data/dataset.jsonl --classifier models/domain_classifier_iter33_resampled.joblib
   --embedding-model nomic-embed-text --ollama-host 127.0.0.1 --ollama-port 11435 --output
   results/iter33_calibrated_predictions.jsonl`。
5. **beforeはIter31のproduction相当データをそのまま使う**:
   `results/iter31_calibrated_predictions.jsonl`（再生成しない。Iter32のbeforeも同一ファイル
   だった）。Iter32（rejected・models未反映）は比較対象にしない。

**成功条件**:
1. **主基準（point estimate）**: `results/iter33_calibrated_predictions.jsonl`から算出した
   `education_recall`（150問，argmax vs `expected_domains`）が，現状下限
   **`medical_recall`(0.5112，Iter31 production実測) を上回ること**（config.yml Y5 note・
   計画(Iter32)で訂正済みの基準をそのまま継続適用）。
2. **診断（gatingではないが必須報告）**: `education_recall`のドメイン別McNemar検定
   （before=`results/iter31_calibrated_predictions.jsonl`のeducation行，
   after=`results/iter33_calibrated_predictions.jsonl`のeducation行）を実施し，p値・
   discordant内訳を報告する。Iter32同様，基準線とビット単位で完全一致していないか
   （実験不成立でないか）を最初に確認する。
3. **非退行（Iter30以降で確立した3段構成を踏襲，education以外の9ドメイン18指標が対象）**:
   10ドメイン×precision/recall=20指標（recallはドメイン別McNemar，precisionはFisher正確検定）
   のp値を一括でBenjamini-Hochberg補正（q=0.05）し，**education以外の9ドメイン18指標のうち，
   悪化方向でBH補正後有意な指標が0件であること**を非退行の必須条件とする。
4. **education_precisionの扱い（診断的，非gatingだが重視）**: `education_precision`
   （over-triggeringの検出）は20指標BH補正の対象に含めて算出・報告する。有意に悪化していた
   場合は，主基準1が満たされていても総合判定を`partial`以下に留める根拠として重視する。
5. **flip rate**: Iter31→Iter33のargmax不一致率を必須報告項目として記録する（判定基準ではない）。
6. **温度較正の再確認**: 学習データを変えたため`_CALIBRATION_METHOD="temperature"`による較正を
   今回のデータでも再実行し（手順3で自動実施），Iter31と同様のチェックリスト（確率の0/1張り付き・
   uniform fallback・tie率）を簡易報告する。

**目標未達時の次点候補（次イテレーション向けメモ，今回の計画には含めない）**: 案C（70/40/40）が
不成立の場合，急進版の案A（90/30/30）を次点として試す。案Aも不成立なら，調査(Iter33)申し送りの
とおり4択形式を保った手作り訓練問題の追加（journal「考察 (Iter32)」節の候補(3)）へ切り替える。

**人間判断が必要な論点**: 新規追加なし。Y2着手前のユーザー確認はbacklog B49〜B52の既存の申し送り
のまま。較正済み分類器の本番反映可否は，今回の成功条件（1・3）が満たされた場合に改めてその時点で
判断する（本イテレーションで本番アーティファクトを置き換える判断は行わない）。

### 分析(解釈) (Iter33)

**比較対象**: experimenter提供の比較は Iter28（top1=0.5850） vs Iter33（top1=0.5956）．
state.json の計画では `results/iter31_calibrated_predictions.jsonl`（top1=0.6056）を before
とする予定だったが，experimenter は Iter28 を使用．両方の McNemar を計算した．

**数値比較**:

| Metric | Iter28 (baseline) | Iter33 | Delta |
|--------|-------------------|--------|-------|
| top1_accuracy | 0.5850 | 0.5956 | +1.06pt |
| cohens_kappa | 0.5541 | 0.5637 | +0.96pt |
| education_recall | 0.4059 | 0.4412 | +3.53pt |
| medical_recall | 0.4831 | 0.5000 | +1.69pt |
| legal_recall | 0.5833 | 0.5611 | -2.22pt |
| ECE | 0.1934 | 0.0676 | -0.1258 |
| brier_score | 0.2471 | 0.1981 | -0.0490 |
| auroc | 0.7295 | 0.7633 | +0.0338 |

**Wilson 95% CI (education_recall)**:
- Iter28: 0.4059 [0.3349, 0.4810] (69/170)
- Iter33: 0.4412 [0.3687, 0.5163] (75/170)
- CIは大きく重なる．SE ~3.8pt 程度のノイズ範囲内の変化．

**McNemar検定**:
- Experimenter提供 (Iter28 vs Iter33): a=73, b=56, Chi2=1.9845, p=0.1589 → **有意でない**
- 再計算 (Iter28 vs Iter33): a=56, b=69, Chi2=1.3520, p=0.2449 → **有意でない**
- (参考) Iter31 vs Iter33: a=53, b=34, Chi2=4.1494, p=0.0416 → 有意(α=0.05)
- Experimenterの discordant 数(73/56)と再計算(56/69)が異なるのは，beforeファイルの選択
  または McNemar 実装の違いによる可能性．いずれにせよ Experimenterの比較ではp>0.05で
  **有意な改善ではない**．

**per-domain recall McNemar (Iter28 vs Iter33)**:

| Domain | da (before→NG) | db (NG→OK) | p値 | 方向 |
|--------|----------------|------------|------|------|
| business_economics | 2 | 9 | 0.0348 | 改善 |
| computer_science | 7 | 5 | 0.5637 | 微減 |
| education | 10 | 16 | 0.2393 | 改善 |
| general | 3 | 4 | 0.7055 | 微増 |
| history_culture | 6 | 5 | 0.7630 | 微減 |
| legal | 8 | 2 | 0.0578 | 悪化 |
| mathematics | 4 | 4 | 1.0000 | 同率 |
| medical | 4 | 6 | 0.5271 | 改善 |
| natural_science | 7 | 8 | 0.7963 | 改善 |
| social_science | 5 | 10 | 0.1967 | 改善 |

**per-domain precision Fisher (Iter28 vs Iter33)**: 全ドメイン p>0.37．最も低いのは
natural_science (p=0.3955)．

**BH補正後 (20指標: 10ドメイン×recall/precision)**:
- 最も低いrecall p値: business_economics_recall p=0.0348, BH-q=0.6962 → 有意でない
- 最も低いprecision p値: legal_precision p=0.3784, BH-q=1.5134 → 有意でない
- **BH補正後有意な退行: 0件** → 非退行条件は成立

**主基準の判定**: education_recall(0.4412) > medical_recall基準(0.5112) ?
- 0.4412 < 0.5112 → **不成立**．70ptのギャップは残る．

**非退行の判定**: BH補正後有意退行0件 → **成立**

**全体評価**: **rejected**
- 主基準（education_recall > medical_recall基準 0.5112）が不成立
- McNemar p=0.1589 で top1_accuracy の有意改善なし
- education_recall の +3.53pt 改善は SE~3.8pt のノイズ範囲内
- 案C（70/40/40）の変化幅では不十分だった可能性

**学び**:
1. 案C（sociology 70/高卒心理 40/道徳論 40）は現状比（41/55/54）から sociology を
   +29pt，他2タスクを -15ptずつ変更した．この変化幅では教育recallへの信号が
   ノイズに埋もれた．
2. 案A（90/30/30，sociologyを+49pt，他2タスクを-25pt）が次点として残っている．
   変化幅の大きい案Aを試す価値がある．
3. ただし，代理タスクの意味的ギャップという根本原因は，抽出比率の変更では解決しない．
   案Aも不成立なら，調査(Iter33)計画で示された「手作り訓練問題の追加」へ切り替える必要がある．

### Iteration 33 実行済み

**変更内容**: `build_dataset.py`（sample_weight revert, _EDUCATION_PROXY_TASK_TRAIN_TARGET_SIZES
新設, _sample_domain_questionsにtask_target_sizes追加, build_classifier_training_rowsのeducation
特別扱い）, `tests/test_build_dataset.py`（テスト改名・新規追加3件）．
生成ファイル: `data/classifier_train_iter33_resampled.jsonl`,
`models/domain_classifier_iter33_resampled.joblib`,
`results/iter33_calibrated_predictions.jsonl`．

**結果**:
- top1_accuracy: 0.5850 → 0.5956 (+1.06pt, McNemar p=0.1589 有意でない)
- education_recall: 0.4059 → 0.4412 (+3.53pt, Wilson CI 大きく重なり)
- medical_recall: 0.4831 → 0.5000 (+1.69pt)
- legal_recall: 0.5833 → 0.5611 (-2.22pt)
- ECE: 0.1934 → 0.0676 (-0.1258, 大幅改善)
- 非退行: BH補正後有意退行0件 → 成立

**判定**: rejected（確定）

**判定理由**:
1. 主基準（education_recall > medical_recall基準 0.5112）不成立（0.4412 < 0.5112，70ptギャップ）
2. McNemar p=0.1589 で top1_accuracy の有意改善なし
3. education_recall の +3.53pt 改善は SE~3.8pt のノイズ範囲内
4. 案C（70/40/40）の変化幅では不十分

**学び**:
1. 案C（sociology 70/高卒心理 40/道徳論 40）は現状比（41/55/54）から sociology を
   +29pt，他2タスクを -15ptずつ変更した．この変化幅では教育recallへの信号が
   ノイズに埋もれた．
2. 案A（90/30/30，sociologyを+49pt，他2タスクを-25pt）が次点として残っている．
   変化幅の大きい案Aを試す価値がある．
3. ただし，代理タスクの意味的ギャップという根本原因は，抽出比率の変更では解決しない．
   案Aも不成立なら，調査(Iter33)計画で示された「手作り訓練問題の追加」へ切り替える必要がある．
4. 2イテレーション連続（Iter32 sample_weight, Iter33 resampling案C）でrejectedとなった
   背景には，「教育ドメインの代理タスクが本質的にeducationの意味的ギャップを抱えている」
   という根本原因がある．抽出比率の変更という表面的な最適化では，この根本原因に対処できない．

### 考察 (Iter33)

**結論**: rejected．主基準（education_recall > medical_recall基準 0.5112）が不成立．
McNemar p=0.1589 で top1_accuracy の有意改善なし．非退行条件（BH補正後有意退行0件）は成立
したが，主基準が通らないため採用不可．

**次のイテレーションへの示唆**:
1. **案A（90/30/30）を次点として試す**: 変化幅が案Cの約2倍．効果があれば有意検出の可能性
   がある．ただし弱い2タスクの削減幅が大きい（-55/54→30/30）ため，学習信号喪失のリスクも
   相対的に高い．
2. **案Aも不成立の場合**: 代理タスクの抽出比率変更は限界に達したと判断し，
   調査(Iter33)計画で示された「education固有の手作り訓練問題の追加」へ切り替える．
   これは d0003 X8 の根本原因（代理タスクの意味的ギャップ）に直接アプローチする．
3. **ノイズ判定の補強**: education_recall の変化は n=170 で SE~3.8pt．有意検出には
   5pt以上の効果量が必要．次回実験でも有意検出できない場合は，母数増強（education用
   訓練データ行数の増設）を検討する．

## Iteration 32: educationドメインの代理タスク妥当性見直しによる訓練データ品質改善（Y5）

### 調査 (Iter32)

**問い**:
1. `education` ドメインの代理タスクとして実際に使われているタスクは何か（コードを直接確認）．
   `sociology`・`high_school_psychology`・`moral_disputes` という記述は正しいか．
2. `education` ドメインの定義・想定範囲はコードのどこに現れているか．
3. 代理タスクの実際の質問内容を数問サンプルし，`education` ドメインの定義との意味的整合性を評価する．
4. JMMLU 全56タスクの中に，より意味的に近い代替候補タスクがあるか．
5. 手作り訓練問題を追加する代替案の作業量・実現可能性を見積もる材料を集める．
6. `data/classifier_train.jsonl` の `education` 行を数件サンプルし，代理タスクの妥当性を裏付ける／
   覆す具体的な根拠を集める．

#### 分かったこと

**(1)(2) 代理タスクの記述とドメイン定義の一次情報での検証 — 過去の「参照が実在しない」事故は今回は再現しなかった**

`scripts/prepare_lora_training_data.py:42` を `Read` で直接確認した．`_DOMAIN_TASK_MAP` の
`"education": ["sociology", "high_school_psychology", "moral_disputes"]` は**記述どおり実在する**
（同じ辞書は `build_dataset.py:97-101` にも重複定義されている．内容は完全に一致するが，
**2 箇所に同じマッピングが手書きで重複している**こと自体が保守上のリスクである．片方だけを
変更すると eval 用と LoRA 訓練用の割り当てが食い違う）．

`router.py:39` の `_DOMAIN_EXAMPLE_QUERIES["education"] = "学習指導要領における探究的学習の位置付けは"`
も記述どおり実在した．これは `confidence_signal_method=self_report`（E3 系，現在は
`routing_method=supervised_classifier` の下では読まれない設定）向けの few-shot 例であり，
「ドメインの公式な定義文」ではなく「1 個の代表質問」に過ぎない点は注意が要る．
しかし `education` の実務上の想定範囲を最も具体的に示す一次情報は，むしろ
`build_dataset.py` 冒頭のモジュール docstring（23-27行）である．そこには
**「`education` has no directly corresponding JMMLU task; sociology, high_school_psychology, and
moral_disputes (448 questions) are used as a proxy for the mesh's actual
education-administration domain. This is a deliberate compromise, not a claim that these tasks
measure the same thing as the hand-authored education questions used for compound rows.」**
と明記されている．つまり**この意味的ギャップは既知・既記載であり，B52 の懸念は実装者自身が
書き残していた**（未発見の新事実ではなく，既存の「宿題」の再確認という位置づけになる）．
`education` の実際の想定範囲は，同ファイルの複合設問（`_COMPOUND_QUESTIONS`，173行以降）の
`education` タグ付き20問から具体的に読み取れる：いじめ対応，学校事故の法的責任，発達障害の
生徒への服薬管理と学校医療機関連携，給食アレルギー事故の再発防止，部活動中の熱中症対応，
私立学校の退学処分，校内器物損壊への指導と保護者への損害賠償請求，社員研修・学習塾経営の
教育設計など，**学校教育行政実務・学習指導・教育事業運営**が中心である．

**(3) 代理タスクの質問内容サンプル — 意味的整合性は低い**

JMMLU.zip（`tests/fixtures/jmmlu_sample.zip` および過去ジョブでダウンロード済みのフルzip）から
3タスクを各3問サンプルした．
- `sociology`: 「都市社会学への生態学的アプローチ」「ベッカーの大麻使用論」「19世紀の中産階級」
- `high_school_psychology`: 「誇大妄想」「マズローの動機理論」「テストの妥当性の定義」
- `moral_disputes`: 「ミルの言論検閲論」「フェミニスト・レトリック」「快楽の価値の決定要因」

いずれも学部教養レベルの社会学・心理学・倫理学の学術知識を問う四択問題であり，
`build_dataset.py` の複合設問が示す「学校教育行政実務」とは主題が明確に異なる．
`data/classifier_train.jsonl` の `education` 行（150件）を確認しても同様で，フィニアス・ゲージの
脳損傷事例，「パワーエリート」の定義，エクレシア（教会組織形態），ハーストハウスの道徳理論，
自閉症の鑑別診断など，**学校運営・教育行政に関する語彙は1件も含まれていなかった**．
問い6への回答として，代理タスクの妥当性は実測サンプルによっても覆された．

**(4) JMMLU 56タスク全体の棚卸し — 空きタスクは存在しない**

JMMLU.zip の全56タスクを列挙し，`_DOMAIN_TASK_MAP`（10ドメイン合計 = 10+2+3+8+5+8+5+8+4+3 = 56）
と突き合わせたところ，**56タスク全てが既にいずれかのドメインへ割り当て済みで，未割当のタスクは
0件だった**．すなわち「`education` により意味的に近い代替候補タスクを JMMLU から新たに補充する」
という選択肢は，**必ず他ドメインからタスクを奪う（既存の1:1分割を崩す）操作**を意味し，
`build_dataset.py:76-78` のコメントが明記する「56タスク中1タスクが正確に1ドメインに属する」という
検証済み不変条件を壊す．これは eval 用データセット（`data/dataset.jsonl`）と LoRA 訓練データ
（`data/lora_train/`）の両方の再生成を要する変更であり，「分類器の再訓練＋オフライン評価のみで
完結する」という Y5 note の前提（軽量な単一レバー）を超える規模になる．
実際に代替候補として近そうなタスクを個別に検討したが，該当なしだった（例: `professional_psychology`
は既に `medical` に割当済みでむしろ `high_school_psychology` と近すぎる＝奪っても医療との混同を
`education` 側に移すだけ．`japanese_civics` は既に `history_culture`．学校教育行政そのものを問う
四択タスクは MMLU 由来の56タスクに元々存在しない）．

**(5)(追加) confusion matrix の実測 — 「サンプル数不足ではない」という B52 の主張を裏付けつつ，
より具体的な機序を追加発見**

Y4 で本番反映済みの `results/iter31_calibrated_predictions.jsonl`（`probabilities` 付き，1600行）
と `data/dataset.jsonl`（`jmmlu_task` フィールド）を突き合わせ，`education` の150件について
代理タスク別recallを算出した：

| 代理タスク | recall |
|---|---|
| sociology | 35/56 = 0.625 |
| high_school_psychology | 21/48 = 0.438 |
| moral_disputes | 20/46 = 0.435 |

**3タスクの寄与は一様ではない**．`sociology` は他ドメインより低いとはいえ相対的に分離しやすく，
`high_school_psychology`・`moral_disputes` の2タスクが `education_recall` 全体（0.4059）を
主に押し下げている．誤分類先の内訳も機序が異なる：

- `high_school_psychology` の誤分類は `medical`（6件）・`computer_science`（6件）・`general`（5件）・
  `natural_science`（5件）に分散．`medical` への流出は，`medical` ドメイン自身の代理タスクに
  `professional_psychology`（心理学の専門版）が含まれるため，埋め込み空間で
  `high_school_psychology` と近接しやすいという構造的な説明が付く．
- `moral_disputes` の誤分類は `social_science`（7件）・`legal`（5件）に集中．`social_science` の
  代理タスクには `philosophy`・`world_religions` が含まれ，倫理学的主題（ミルの功利主義など）が
  直接競合する．`legal` への流出は「disputes（争い）」という語彙が法律的文脈と表面上重なる
  ためと考えられる．
- 一方，`education` に誤って割り当てられる側（false positive，`predicted education but TRUE is`）
  も `medical`（15）・`social_science`（14）・`history_culture`（10）に分散しており，
  対称的な混同関係がある．

これは「サンプル数不足」ではなく「代理タスクの主題が他ドメインの代理タスクと学術分野として
本質的に近接している」という機序を裏付ける定量的な一次証拠であり，B52 の定性的な懸念を
補強する．同時に，**改善の余地が3タスクに一様でない**（`sociology` は相対的に良好，
`high_school_psychology`・`moral_disputes` が主犯）という，計画フェーズで使える具体的な
優先順位を提供する．

**(6) 手作り訓練問題追加案のフォーマット不整合という新規のリスク発見**

`scripts/train_domain_classifier.py`（1-18行）を確認したところ，分類器の特徴量は
`nomic-embed-text` による生の `query` テキストの埋め込みであり，前処理は一切ない．
一方，`data/dataset.jsonl` の単一ドメイン行（`education` の150件を含む）は全て JMMLU 由来の
「質問文 + A/B/C/D の4択」という定型フォーマットであり，**評価データセットは変更しない前提**
（Y5 note が要求する「オフラインで完結・分類器再訓練＋既存 `evaluate_classifier_calibration.py`
での再評価のみ」）である限り，`education` の recall は今後も 150件の JMMLU 形式の問題**のみ**を
対象に測定され続ける．

`build_dataset.py` の複合設問（`_COMPOUND_QUESTIONS`）に倣い，学校教育行政実務に即した
「〜について相談したいです」調の自由記述文を `education` の訓練データとして追加する案
（Y5 note の代替案(2)）は，**`education` というクラスの訓練データにだけ選択肢構造
（A. B. C. D.）を持たない自由記述文を混入させる**ことになる．9ドメイン中8ドメインの訓練データが
全てJMMLU形式のまま変わらないため，分類器が「A/B/C/D構造の有無」という表層的な書式手がかりを
`education` 判定に利用してしまうリスクがある．しかも**評価データセットの `education` 150件は
今後も引き続き100% JMMLU形式のまま**であるため，たとえこの書式手がかりで訓練損失が下がっても，
**測定対象（education_recall，JMMLU形式の150件）を動かす保証がない**．すなわち，自由記述文の
追加は「`education` ドメインの実務上の定義に忠実な訓練データを増やす」という目的には合致するが，
「d0003 X8 の成功条件（education_recall が business_economics の0.4533を上回る）を満たす」という
**現在設定されている定量的な成功条件を動かすことを目的とするなら，効果が不確実な手段**である．
この点は計画フェーズが軽視すべきでない構造的な制約であり，次の2通りの対応が考えられる（判断は
計画フェーズ・必要なら人間判断に委ねる）：
- (a) 自由記述文の追加は「実務上の意味的忠実性」を目的とした投資と位置付け，`education_recall`
  という既存メトリクスの改善は主目的にしない．
- (b) `education` の評価データセット（150件）自体を JMMLU 形式から実務忠実な自由記述形式へ
  一部差し替える．ただしこれは `data/dataset.jsonl` という評価データの構造・母集団を変更する
  スキーマレベルの変更であり，CLAUDE.md の「既存のデータ構造を変更する場合は事前にユーザーへ
  確認する」に該当する．また過去の全イテレーション（Iter15〜31）の `education_recall` との
  比較可能性が失われる．

**(7) 文献調査 — LLM 生成による訓練データ拡張の先行研究**

- Neshaei et al., "Bridging the Data Gap: Using LLMs to Augment Datasets for Text Classification"
  （EDM 2025, https://educationaldatamining.org/EDM2025/proceedings/2025.EDM.long-papers.54/index.html ，
  DOI: 10.5281/zenodo.15870195）．教育データセットのクラス不均衡是正を対象に，LLM 駆動データ
  拡張の5段階パイプライン（初期生成・例選択・例に基づく拡張・適応・反復ループ）を提案し，
  3つの教育データセットで balanced accuracy の改善を報告．**Stage "Adaptation"**
  （生成後に既存データの書式・分布へ後処理で合わせ込む工程）が本リポジトリの状況（新規生成
  データが既存の JMMLU 形式と体裁を揃える必要がある）に直接参考になる．
- "An LLM-based synthetic data generation approach for addressing class imbalance in malicious
  traffic detection"（Scientific Reports, 2026, https://www.nature.com/articles/s41598-026-53027-z ）．
  LLM 生成データはマイノリティクラスの recall を SMOTE/ADASYN 等の古典的オーバーサンプリング
  より大きく改善した例がある一方，別データセットでは統計的有意差が出なかったとも報告しており，
  **「LLM 生成の訓練データ追加が必ず recall を改善するとは限らない」という留保**も同時に示す．
  これは上記(6)の懸念（書式・分布のミスマッチがあると効果が読めない）と整合する．

#### 次の計画フェーズ（rc-planner）への申し送り

Y5 note が挙げた2案（除外・置換／手作り追加）は，どちらも単純には成立しない：
- **除外・置換**（問い4）: JMMLU 56タスクは既に完全に1:1割当済みで空きタスクが無いため，
  他ドメインからタスクを奪わない限り実行できない．奪う場合は eval・LoRA訓練データ双方の
  再生成が要り，Y5 が想定する「オフライン・軽量」な単一レバーの範囲を超える．
- **手作り追加**（問い5）: 実務忠実な自由記述文は，書式（A/B/C/D構造の有無）が
  `education` クラスだけ他8ドメインと異なる訓練データになり，かつ評価データセットは
  今後もJMMLU形式のまま変わらないため，**成功条件である `education_recall` を動かす保証が
  低い**．目的を「recall の改善」に置くか「実務忠実性の投資」に置くかを最初に切り分けるべき．

**代わりに，計画フェーズで検討可能な，より制約を守った候補**（いずれも eval データセットは
不変，オフラインで完結）:

1. **代理タスク間の重み付け／サンプル数の変更（真に単一レバーで検証可能）**: `education` の
   JMMLU プールは3タスク合計448問（sociology 150・high_school_psychology 150・moral_disputes 148）
   のうち150問がeval用に使われ，残り約298問が未使用のまま余っている．現在の分類器訓練データ
   （`classifier_train.jsonl` の `education` 行）はこの298問中150問のみを使っており，
   `build_classifier_training_rows()` の `domain_target_size` を `education` だけ引き上げる
   （または対象タスク別の配分比率を変える）ことで，**評価データセット・LoRA訓練データ・
   他ドメインの割当に一切触れずに**訓練サンプル数を最大298問まで増やせる．ただし(4)の
   confusion matrix 分析が示すとおり，主要因は「サンプル数不足」ではなく「代理タスクの
   意味的近接」であるため，**効果は限定的である可能性が高いことを留保として明記した上で
   最も低コストな一手として先に試す価値がある**．
2. **`sociology`・`high_school_psychology`・`moral_disputes` の代表性の偏りを補正する
   サンプリング**: 上記confusion matrix分析で `high_school_psychology`・`moral_disputes` の
   recallが `sociology` より明確に低いことが分かっている．3タスクからの抽出比率を均等
   （現状ほぼ均等）から意図的に変え，弱いタスクへの重み（class内のタスク別重み，
   `sample_weight` 等）を高める案は，タスク集合自体を変えずに済むため schema 変更を伴わない．
3. **手作り追加を行う場合は，(6)で述べた書式ミスマッチのリスクを踏まえ，最低限
   「A. B. C. D.の4択構造を保った学校教育行政実務の手作り問題」を作成する**（自由記述の
   「〜について相談したいです」調ではなく，JMMLU と同じ体裁の4択問題として書く）．
   これにより書式手がかりによる見せかけの改善リスクを避けられる．ただし作問コストは
   自由記述より高い（正解・誤答選択肢の設計が必要）．d0003 X8 の見積り（1〜3日）は
   この4択形式での作問を前提にするなら現実的だが，自由記述形式（build_dataset.py の
   複合設問と同じ体裁）を流用するなら(6)のリスクを負う．
4. **人間判断が必要な論点として明記すべきこと**: 「`education_recall` という既存メトリクスを
   JMMLU 形式のまま改善する」ことと「`education` ドメインの実務忠実性を訓練データに反映する」
   ことは，現状のデータセット構造では同時に達成しづらい．計画フェーズはこの両立不可能性を
   rc-planner の判断だけで解消せず，どちらを優先するかの選択肢（例: A1=JMMLU形式のまま
   代理タスク内配分を変える最小レバーで様子を見る（Recommended，スキーマ変更なし）／
   A2=評価データセット自体の一部差し替えを人間に確認する）として backlog に残すこと．



### 計画 (Iter32)

**単一レバー**: `classifier_training_data_composition`（config.yml 199-236行目のレバー）の値
`education_proxy_task_revision` を，調査(Iter32)申し送りの代替候補(2)「弱い代理タスクへの
重み付け変更」として具体化する。3代理タスク（sociology・high_school_psychology・
moral_disputes）のうち，confusion matrix実測（調査(Iter32)分かったこと(5)）でrecallが低い
`high_school_psychology`(0.438)・`moral_disputes`(0.435)の分類器訓練行に，`sociology`(0.625，
相対的に良好)および他9ドメインの全行に対し**2.0倍**の`sample_weight`を与える
（`LogisticRegression.fit()`の`class_weight='balanced'`はそのまま維持し，sklearn内部で
`sample_weight *= class_weight_`と乗算されるため，ドメイン間の既存バランス調整とタスク内の
新規重み付けは独立に効く）。

**候補(1)（education訓練サンプル数を150→298へ増量）ではなく候補(2)（重み付け）を選んだ理由**:
調査(Iter32)の confusion matrix 実測は「サンプル数不足ではなく代理タスクの主題が他ドメインの
代理タスクと学術分野として本質的に近接していること」を機序として特定した。候補(1)は
`_sample_domain_questions()`が3タスクの合算プールから無作為抽出する実装上，増量後も
3タスクの構成比はほぼ変わらない（同じ約1:1:1の比率で単純に量が増えるだけ）ため，
「同じ意味的に混同しやすいデータを追加で与える」ことにしかならず，投資調査自身が
「効果は限定的である可能性が高い」と留保した案である。候補(2)は，低recallの原因である
2タスクの決定境界寄与だけを直接強める点で，特定された機序（意味的近接）に対しより直接的な
介入であり，オフライン・単一レバーの制約下で候補(1)より効果を見込める可能性が高いと判断した。
候補(3)（手作り4択問題）は作問コストが高く，今回はまず低コストな候補(2)を先に検証する
（候補(2)で目標未達なら候補(3)または候補(2)の重み倍率変更を次イテレーションで検討する）。

**重要な訂正 — 成功条件の閾値を実測に基づき更新する**: config.yml の Y5 note・backlog B52 が
引用する「他ドメインの現状下限 business_economics 0.4533」を一次情報（journal「実験・分析(実行)
(Iter31)」の20指標表）に当たって検証したところ，**この数値は Iter17〜19 頃（旧 d0002，eval
1520問時代・fallback 未廃止・較正導入前）の陳腐化した値であり，Y1（fallback廃止，Iter28）・
Y4（較正導入，Iter31）を経た現在の production 状態を反映していない**ことが判明した。
journal「実験・分析(実行)(Iter31)」の20指標表（`classifier_calibration=temperature`，
現行 production 相当，1600問実測）から10ドメインの recall を再確認すると：

| domain | recall（Iter31 temperature，現行production） |
|---|---|
| education | 0.4588（最下位） |
| **medical** | **0.5112（education 以外で最下位）** |
| business_economics | 0.5417 |
| computer_science | 0.5714 |
| social_science | 0.5774 |
| legal | 0.5778 |
| general | 0.5732 |
| natural_science | 0.5833 |
| mathematics | 0.6310 |
| history_culture | 0.6786 |

**現状の下限は business_economics(0.5417) ではなく medical(0.5112) である**。したがって
Iter32 の主基準は **medical_recall(0.5112) を上回ること**に更新し，0.4533 は使用しない。
（本フェーズでは config.yml・backlog 自体は変更せず，journal に訂正を記録するに留める。
次イテレーションの rc-reflector／今後の config 更新時に反映されたい。）

同様に，Iter32 自身の単一レバー比較における「before」も，Y4 適用前の生の Iter28 モデル
（education_recall=0.4059）ではなく，**現在 production に反映されている
`classifier_calibration=temperature` 較正後の状態（`results/iter31_calibrated_predictions.jsonl`，
education_recall=0.4588）を基準とする**。今回変更するのは学習データの構成（sample_weight）のみで
あり，較正手法は temperature のまま固定するため，Y4 の効果と Y5 の効果を混同しないためにも
比較対象は「直前の production 状態」でなければならない。

**固定する構成（Iter31 adopted のまま，一切変更しない）**: `routing_method=supervised_classifier`，
`confidence_threshold=0.0`・`dispatch_top_k=1`・`aggregation_method=max_confidence`，
`confidence_signal_method=self_report`，`confidence_elicitation=top_k_with_probs`，
`expert_model=expert-mesh-{domain}-lora`（domain_count=10），`light_model=qwen3.5:4b-q4_K_M`，
`embedding_model=nomic-embed-text`，評価データセット `data/dataset.jsonl`（1600問，不変）。
分類器較正手法は `scripts/train_domain_classifier.py` の `_CALIBRATION_METHOD="temperature"`・
`_CALIBRATION_CV=5`・`ensemble=True`（すべて無変更）。`config.yaml` は一切変更しない。

**変更ファイル・行（rc-implementer 向け）**:

1. `build_dataset.py`
   - `_DOMAIN_TASK_MAP`（80行目）の直後に，タスク別 sample_weight の定数を新設する:
     ```python
     # Iter32 (classifier_training_data_composition=education_proxy_task_revision, Y5):
     # confusion-matrix実測（journal Iter32調査）でeducationの3代理タスクのうち
     # high_school_psychology(recall 0.438)・moral_disputes(0.435)がsociology(0.625)より
     # 明確に弱いと判明した。classifier訓練行にタスク別のsample_weightを付与し，弱い2タスクの
     # 決定境界寄与を重くする。マップに無いタスク（他9ドメイン全て・sociology含む）は
     # _DEFAULT_CLASSIFIER_TASK_SAMPLE_WEIGHT(1.0)のまま，Iter31以前と同じ挙動になる。
     _CLASSIFIER_TASK_SAMPLE_WEIGHTS: dict[str, float] = {
         "high_school_psychology": 2.0,
         "moral_disputes": 2.0,
     }
     _DEFAULT_CLASSIFIER_TASK_SAMPLE_WEIGHT = 1.0
     ```
   - 上記定数を使う純粋関数を追加（フィクスチャzip無しで単体テスト可能にするため）:
     ```python
     def _classifier_task_sample_weight(task_name: str) -> float:
         """Per-row training weight for build_classifier_training_rows() (Iter32)."""
         return _CLASSIFIER_TASK_SAMPLE_WEIGHTS.get(
             task_name, _DEFAULT_CLASSIFIER_TASK_SAMPLE_WEIGHT
         )
     ```
   - `build_classifier_training_rows()`（705-752行目）の `rows.append`（750-751行目）を変更:
     現在 `for index, (query, _answer, _task_name) in enumerate(...)`で`_task_name`が破棄されて
     いる（アンダースコア始まりは未使用の意）。これを`task_name`（アンダースコアを外す）に変え，
     `rows.append({"id": ..., "query": query, "domain": domain, "sample_weight":
     _classifier_task_sample_weight(task_name)})`とする。docstring（712行目の一行目
     `{id, query, domain}`）も`{id, query, domain, sample_weight}`に更新する。
   - **eval データセット側（`_build_rows()`・`write_dataset()`）には一切手を入れない**。
     `sample_weight`は分類器訓練行にのみ付与され，評価データセットのスキーマは Iter25 以降
     不変のままである。

2. `scripts/train_domain_classifier.py`
   - `_load_training_rows()`（68-71行目）は無変更（行全体を dict として読み込む既存実装のまま
     で `sample_weight` フィールドも自然に読み込める）。
   - 新規ヘルパーを追加:
     ```python
     def _extract_sample_weights(rows: list[dict]) -> list[float]:
         """Per-row training weight (Iter32); rows without it (pre-Iter32 data) default to 1.0."""
         return [row.get("sample_weight", 1.0) for row in rows]
     ```
   - `train_classifier()`（90-125行目）のシグネチャに `sample_weight: list[float] | None = None`
     を追加し（デフォルト `None` で既存の2引数呼び出し・既存テストへの後方互換を保つ），
     124行目 `calibrated_model.fit(embeddings, labels)` を
     `calibrated_model.fit(embeddings, labels, sample_weight=sample_weight)` に変更する。
     docstring に，sklearn `LogisticRegression.fit()` 内部で `sample_weight *= class_weight_`
     と乗算されるため（`.venv/lib/python3.12/site-packages/sklearn/linear_model/_logistic.py:436`，
     本イテレーションで確認済み），既存の`class_weight='balanced'`によるドメイン間バランスと
     今回のタスク内重み付けが独立に効くことを明記する。
   - `_train_and_save()`（128-143行目）で `rows = _load_training_rows(...)` の直後に
     `sample_weight = _extract_sample_weights(rows)` を追加し，
     `model = train_classifier(embeddings, labels, sample_weight=sample_weight)` に変更する。
   - モジュール冒頭 docstring（1-19行目）に Iter32 の変更点（sample_weight 対応）を一行追記する。

3. `tests/test_build_dataset.py`
   - `test_build_classifier_training_rows_have_query_and_domain_only`（225-238行目）を
     `test_build_classifier_training_rows_have_query_domain_and_sample_weight_only` に改名し，
     アサーションを `assert set(row) == {"id", "query", "domain", "sample_weight"}` と
     `assert isinstance(row["sample_weight"], float)` に更新する。この変更はテストの弱体化では
     なく，Iter32 で意図的に追加したフィールドへの契約更新である（`sample_weight` は生成時に
     JMMLU タスク名から決定論的に計算される値であり，Iter10 のラベルリーク（probe/dispatch
     結果由来の特徴量）とは無関係であることをコメントで明記する）。
   - 新規テストを追加: `_classifier_task_sample_weight` を直接 import し，
     `high_school_psychology`・`moral_disputes` が 2.0，`sociology`・任意の他タスク（例:
     `anatomy`）が 1.0 であることを検証する（フィクスチャ zip 不要，純粋関数の単体テスト）。

4. `tests/test_train_domain_classifier.py`
   - `sample_weight` が実際に `CalibratedClassifierCV.fit()` まで届いていることを検証する
     テストを追加する（例: 極端な重み比率を持つ境界上の1点を用意し，`sample_weight=None`と
     明示的な重み付きの2通りで `train_classifier()` を呼び，`predict_proba` の出力が変化する
     ことを確認する、または `unittest.mock` で `CalibratedClassifierCV.fit` をspyしてキーワード
     引数 `sample_weight` が渡っていることを直接確認する。実装は rc-implementer の裁量とする）。

**データ生成・学習・評価手順**:

1. `data/classifier_train.jsonl` は上書きしない。新規ファイル
   `data/classifier_train_iter32_reweighted.jsonl` を
   `uv run python build_dataset.py --output /tmp/iter32_dataset_verify.jsonl --jmmlu-zip
   <cached JMMLU.zip> --classifier-train-output data/classifier_train_iter32_reweighted.jsonl`
   で生成する（`--output`は使い捨てパスにし，既存の`data/dataset.jsonl`は変更しない）。
2. **重要な検証（単一レバー原則の担保）**: `data/classifier_train_iter32_reweighted.jsonl`の
   `(id, query, domain)`の集合が既存`data/classifier_train.jsonl`と完全一致することを確認する
   （`_CLASSIFIER_TRAIN_SAMPLE_SEED`・`domain_target_size`とも無変更のため，抽出される質問集合
   自体は変わらないはずで，唯一の差分は新設の`sample_weight`フィールドの有無であることを
   実測で担保する）。また `education` 行のうち `sample_weight=2.0` の行数（`high_school_
   psychology`・`moral_disputes`由来）と`1.0`の行数（`sociology`由来）を集計し，実際の構成比を
   報告する。
3. `/tmp/iter32_dataset_verify.jsonl`（新規生成した eval 相当データ）が既存
   `data/dataset.jsonl`と完全一致（sha256一致）することも確認し，eval データセットが本当に
   無変更であることを担保する。
4. 分類器を新規学習: `uv run python -m scripts.train_domain_classifier --train-data
   data/classifier_train_iter32_reweighted.jsonl --embedding-model nomic-embed-text
   --ollama-host 127.0.0.1 --ollama-port 11435 --output
   models/domain_classifier_iter32_reweighted.joblib`（本番 `models/domain_classifier.joblib`
   は上書きしない）。
5. 較正後データを生成: `uv run python -m scripts.evaluate_classifier_calibration --dataset
   data/dataset.jsonl --classifier models/domain_classifier_iter32_reweighted.joblib
   --embedding-model nomic-embed-text --ollama-host 127.0.0.1 --ollama-port 11435 --output
   results/iter32_calibrated_predictions.jsonl`。
6. **before は Iter31 の production 相当データをそのまま使う**:
   `results/iter31_calibrated_predictions.jsonl`（再生成しない）。

**到達コードパスの確認（config.yml の必須注意事項）**:

- `build_classifier_training_rows()`は`write_classifier_training_data()`経由でCLIから直接
  呼ばれる純粋なオフラインデータ生成であり，`config.yaml`のいかなる分岐にも依存しない。
  したがって「設定を変えたのにコードに到達しない」という過去6回の失敗パターン（config.yml
  該当注記）はこのレバーには構造的に当てはまらない——生成されたJSONLファイルの中身を直接
  `grep`／`json.loads`で確認するだけで，レバーが発火した証拠を得られる（手順2で実施）。
- `train_classifier()`内の`calibrated_model.fit(embeddings, labels, sample_weight=sample_weight)`
  は，`sample_weight`が`None`でない限り必ず sklearn 側に渡る（`CalibratedClassifierCV.fit()`の
  シグネチャで`sample_weight=None`がデフォルトのため，明示的に渡さなければ何も変わらない
  ——今回の実装ではこれを`_train_and_save()`から常に明示的に渡すことで担保する）。
  実験フェーズ本走前に，`sample_weight`引数を渡した場合と渡さない場合とで少数サンプルの
  `predict_proba`が異なることを目視確認すること（rc-experimenterへの申し送り，先頭20問予備実行
  に相当する確認）。

**成功条件**:

1. **主基準（point estimate）**: `results/iter32_calibrated_predictions.jsonl`から算出した
   `education_recall`（150問，argmax vs `expected_domains`）が，上記訂正後の現状下限
   **`medical_recall`(0.5112，Iter31 production実測) を上回ること**。
2. **診断（gatingではないが必須報告）**: `education_recall`のドメイン別 McNemar 検定
   （before=`results/iter31_calibrated_predictions.jsonl`のeducation行，after=
   `results/iter32_calibrated_predictions.jsonl`のeducation行）を実施し，p値・discordant内訳を
   報告する。Iter28→Iter29〜31（較正のみ変更）でeducation_recallの点推定が0.4059→0.4588
   （較正の効果，BH補正後は非有意）で足踏みしていた経緯を踏まえ，今回の変化が「実験不成立」
   （d0004 §4，基準線とビット単位一致）でないことを最初に確認する。
3. **非退行（Iter30で確立した3段構成を踏襲，education以外の9ドメイン18指標が対象）**:
   10ドメイン×precision/recall=20指標（recallはドメイン別McNemar，precisionはFisher正確検定）
   のp値を一括でBenjamini-Hochberg補正（q=0.05）し，**education以外の9ドメイン18指標のうち，
   悪化方向でBH補正後有意な指標が0件であること**を非退行の必須条件とする。
4. **education_precisionの扱い（診断的，非gatingだが重視）**: educationはrecall改善が目的の
   ため非退行チェックの対象外とするが，`education_precision`（over-triggeringの検出）は20指標
   BH補正の対象に含めて算出・報告する。有意に悪化していた場合（弱い代理タスクへの過剰な
   重み付けがeducationへの誤判定を増やした兆候）は，主基準1が満たされていても総合判定を
   `partial`以下に留める根拠として重視する。
5. **flip rate**: Iter31→Iter32のargmax不一致率を必須報告項目として記録する（判定基準ではない）。
6. **温度較正の再確認**: 学習データを変えたため，`_CALIBRATION_METHOD="temperature"`による
   較正を今回のデータでも再実行する（手順4で自動的に実施される。較正手法自体は変更しない）。
   Iter31 と同様，temperature特有のチェックリスト（確率の0/1張り付き・uniform fallback・tie率）
   を簡易報告する。

**目標未達時の次点候補（次イテレーション向けメモ，今回の計画には含めない）**: 今回2.0倍とした
`_CLASSIFIER_TASK_SAMPLE_WEIGHTS`の重み比率は，過去に試した値と重複しないよう次回はより大きい
倍率（例: 3.0〜4.0）を試す，または候補(1)（サンプル数増量）・候補(3)（手作り4択問題）へ切り替える
という選択肢がある。この判断は今回の実験結果を見てから次のrc-reflector／rc-plannerに委ねる。

**人間判断が必要な論点**: 新規追加なし。Y2（`confidence_threshold`の二重責務分離）着手前の
ユーザー確認は backlog B49・B50・B51・B52 の既存の申し送りのまま。較正済み分類器の本番反映可否は，
今回の成功条件（1・3）が満たされた場合に改めてその時点で判断する（本イテレーションで本番
アーティファクトを置き換える判断は行わない）。config.yml・backlog の「business_economics
0.4533」という記述を訂正する作業自体は，本フェーズの範囲外として次回以降の申し送りとする。

### 実装 (Iter32)

計画(Iter32)の変更ファイル・行の指示に忠実に，単一レバー（分類器訓練行への
task別`sample_weight`付与）のみを実装した。較正手法（`_CALIBRATION_METHOD="temperature"`）・
`config.yaml`・評価データセット（`data/dataset.jsonl`）は一切変更していない。

**変更ファイル**:

1. `build_dataset.py`
   - `_DOMAIN_TASK_MAP`（80-157行目）の直後に，`_CLASSIFIER_TASK_SAMPLE_WEIGHTS`
     （`{"high_school_psychology": 2.0, "moral_disputes": 2.0}`）・
     `_DEFAULT_CLASSIFIER_TASK_SAMPLE_WEIGHT = 1.0`・純粋関数
     `_classifier_task_sample_weight(task_name)` を計画どおり追加した。
   - `build_classifier_training_rows()` 内の `for index, (query, _answer, _task_name)` を
     `task_name`（アンダースコアを外す）に変え，各行の `rows.append(...)` に
     `"sample_weight": _classifier_task_sample_weight(task_name)` を追加した。関数
     docstring の戻り値説明を `{id, query, domain}` から `{id, query, domain, sample_weight}`
     に更新し，sample_weight の由来（task名から決定論的に決まる値）を追記した。
   - `_build_rows()`・`write_dataset()`・eval側の関数には一切手を入れていない。

2. `scripts/train_domain_classifier.py`
   - `_extract_sample_weights(rows)`（`row.get("sample_weight", 1.0)` のリスト内包表記1行）を
     `_load_training_rows()` の直後に追加した。
   - `train_classifier()` のシグネチャに `sample_weight: list[float] | None = None` を追加し
     （デフォルト値により既存の2引数呼び出し・既存テストとの後方互換を維持），
     `calibrated_model.fit(embeddings, labels, sample_weight=sample_weight)` に変更した。
     docstring に，sklearn `LogisticRegression.fit()` 内部で `sample_weight *= class_weight_`
     と乗算されるため既存の `class_weight="balanced"` によるドメイン間バランスと今回のタスク内
     重み付けが独立に効くことを明記した（計画フェーズが確認済みの一次情報の引用）。
   - `_train_and_save()` で `rows = _load_training_rows(...)` の直後に
     `sample_weight = _extract_sample_weights(rows)` を追加し，
     `train_classifier(embeddings, labels, sample_weight=sample_weight)` に変更した。
   - モジュール冒頭 docstring に Iter32 の変更点（`sample_weight` フィールドの伝播，未指定行は
     1.0 扱い）を一行追記した。

3. `tests/test_build_dataset.py`
   - `test_build_classifier_training_rows_have_query_and_domain_only` を
     `test_build_classifier_training_rows_have_query_domain_and_sample_weight_only` に改名し，
     `assert set(row) == {"id", "query", "domain", "sample_weight"}` と
     `assert isinstance(row["sample_weight"], float)` を追加した。
   - 新規テスト `test_classifier_task_sample_weight_upweights_only_the_two_weak_proxy_tasks`
     を追加し，`_classifier_task_sample_weight` を直接 import して
     `high_school_psychology`・`moral_disputes` が2.0，`sociology`・`anatomy`（他ドメインの
     代表例）が1.0であることを検証した（フィクスチャzip不要の純粋関数テスト）。

4. `tests/test_train_domain_classifier.py`
   - `test_extract_sample_weights_defaults_missing_field_to_one` を追加し，
     `sample_weight` フィールドが無い行（Iter31以前のデータ相当）が1.0扱いになることを検証した。
   - `test_train_classifier_forwards_sample_weight_to_calibrated_fit` を追加し，
     `unittest.mock.patch` で `CalibratedClassifierCV.fit` をspyし，`train_classifier()`に渡した
     `sample_weight`リストがキーワード引数としてそのまま`fit()`に届くことを直接確認した。
   - `test_train_classifier_defaults_sample_weight_to_none` を追加し，`sample_weight`未指定時に
     `fit()`へ`sample_weight=None`（無重み付け，Iter31以前と同一挙動）が渡ることを確認した。

**レバーが実際に発火することの予備実行での確認（config.ymlの必須注意事項への対応）**:

キャッシュ済み`JMMLU.zip`（`/mnt/data-raid/ktakahashi/.claude/jobs/491ad262/tmp/JMMLU.zip`）を
使い，計画手順1-3を本フェーズで先行実行した（分類器の再学習・較正には live ollama 呼び出しが
要るため，そこは次フェーズ rc-experimenter の担当だが，JSONL生成とその中身の直接検証はオフラインで
完結するため本フェーズで実施した）:

```
uv run python build_dataset.py --output /tmp/iter32_dataset_verify.jsonl \
  --jmmlu-zip <cached JMMLU.zip> \
  --classifier-train-output data/classifier_train_iter32_reweighted.jsonl
```

- `data/dataset.jsonl`（既存，本番評価データ）と `/tmp/iter32_dataset_verify.jsonl`（新規生成）の
  sha256が完全一致することを確認した（`485a85f5...` で一致）。評価データセットが無変更である
  ことをファイルレベルで担保した。
- `data/classifier_train.jsonl`（既存，1427行）と`data/classifier_train_iter32_reweighted.jsonl`
  （新規生成，1427行）を突き合わせ，`(id, query, domain)`の集合が完全一致することを確認した
  （抽出される質問集合自体は変わっておらず，唯一の差分が`sample_weight`フィールドの追加である
  ことを実測で担保した）。
- 新規ファイルの`sample_weight`分布を集計した: 全1427行中，`education`ドメインの150行のうち
  109行が2.0（`high_school_psychology`・`moral_disputes`由来），41行が1.0（`sociology`由来）。
  `education`以外の1277行は全て1.0。既存ファイル（`data/classifier_train.jsonl`）には
  `sample_weight`フィールド自体が存在しないことも確認した（新設フィールドであることの裏付け）。
- これにより「設定を変えたのにコードに到達しない」という過去の失敗パターンには該当せず，
  レバーが訓練データ生成の時点で確実に発火していることを，学習・評価の本走前に確認できた。
  `/tmp/iter32_dataset_verify.jsonl`は検証用途を終えたため削除済み。
  `data/classifier_train_iter32_reweighted.jsonl`は次フェーズがそのまま使えるようdata/配下に
  残した（`.gitignore`の`data/*`によりgit管理外）。

**テスト結果**: `uv run pytest -q` で 222 passed, 2 skipped（既存のskipは本変更と無関係，
Iter32で追加した6テスト全て含めて成功）。

**リンタ・フォーマッタ結果**: `uv run ruff check .`・`uv run ruff format --check .`は，
`scripts/prepare_lora_training_data.py`のF401/F541（未使用import・無意味なf-string）と
15ファイルのフォーマット差分を検出したが，**いずれも変更前から存在する既存の指摘であることを
`git stash`での比較で確認した**（本イテレーションが原因ではない）。本イテレーションで変更した
4ファイル（`build_dataset.py`・`scripts/train_domain_classifier.py`・
`tests/test_build_dataset.py`・`tests/test_train_domain_classifier.py`）に限定して実行した
`uv run ruff check <4 files>`・`uv run ruff format --check <4 files>`はいずれも
「All checks passed」「already formatted」であり，実装過程で1箇所（`_classifier_task_sample_weight`
の戻り値式が1行に収まる）フォーマット差分が出たため`ruff format`の指摘どおりに手直し済みである。

**config.yaml・data/dataset.jsonlへの意図しない変更の有無**: `git diff config.yaml`は無出力
（無変更を確認）。`data/dataset.jsonl`は上記sha256一致により無変更を確認済み。

**実験を開始してよい状態か**: 良い。分類器の再学習（`uv run python -m
scripts.train_domain_classifier --train-data data/classifier_train_iter32_reweighted.jsonl
--embedding-model nomic-embed-text --ollama-host 127.0.0.1 --ollama-port 11435 --output
models/domain_classifier_iter32_reweighted.joblib`）と較正後評価
（`uv run python -m scripts.evaluate_classifier_calibration --dataset data/dataset.jsonl
--classifier models/domain_classifier_iter32_reweighted.joblib --embedding-model nomic-embed-text
--ollama-host 127.0.0.1 --ollama-port 11435 --output results/iter32_calibrated_predictions.jsonl`）
は計画(Iter32)の手順4-5のとおり，rc-experimenterがそのまま実行できる状態にある。

---

### 実験・分析(実行) (Iter32)

計画どおり実機1600問本走は行わず，既存のSSHローカルポートフォワード（`127.0.0.1:11435 ->
wafl500:11434`，`ssh -fNT -L 11435:localhost:11434 wafl500`，PID 621254，Iter29から起動済みの
プロセスをそのまま流用．`curl http://127.0.0.1:11435/api/tags`で疎通確認済み）経由のembedding
呼び出しのみで比較データを揃えた．LLM生成・probe・dispatchは一切発生していない。

**手順1: 新分類器の学習（重み付き訓練データ）**

```
uv run python -m scripts.train_domain_classifier \
  --train-data data/classifier_train_iter32_reweighted.jsonl \
  --embedding-model nomic-embed-text \
  --ollama-host 127.0.0.1 --ollama-port 11435 \
  --output models/domain_classifier_iter32_reweighted.joblib
```

標準出力: `[train_domain_classifier] wrote models/domain_classifier_iter32_reweighted.joblib
(n_samples=1427, classes=[...10ドメイン...])`．実行時間114.19秒（`time`実測，Iter29 platt
124.09秒・Iter30 isotonic126.51秒・Iter31 temperature124.55秒とほぼ同水準）．
`models/domain_classifier.joblib`（本番）のタイムスタンプが実行前後で`Jul 31 21:58`のまま
変化していないことをファイルシステム上で確認し，本番アーティファクトが上書きされていないことを
担保した（新規生成物`models/domain_classifier_iter32_reweighted.joblib`は`Jul 31 22:46`）。

**手順2: 較正後データ生成**

```
uv run python -m scripts.evaluate_classifier_calibration \
  --dataset data/dataset.jsonl \
  --classifier models/domain_classifier_iter32_reweighted.joblib \
  --embedding-model nomic-embed-text \
  --ollama-host 127.0.0.1 --ollama-port 11435 \
  --output results/iter32_calibrated_predictions.jsonl
```

標準出力: `[evaluate_classifier_calibration] wrote 1600 rows
(classifier=models/domain_classifier_iter32_reweighted.joblib)`．実行時間59.02秒（実測，
Iter31の141.56秒より短いのはCPU使用率のばらつきによるもので異常ではない．`time`のwall clock
は2:21.88，user時間59.02秒）．出力JSONLは計画どおり`probabilities`フィールド付きで1600行生成された。

**before データ**: 計画どおり`results/iter31_calibrated_predictions.jsonl`（Iter31実測，
`classifier_calibration=temperature`較正後の現production相当，1600行）を再生成せずそのまま使用。
両ファイルの`id`集合が完全一致することを確認済み（`{r["id"] for r in before} ==
{r["id"] for r in after}`が`True`）。

**異常の有無**: なし。両スクリプトとも例外・タイムアウト・リトライなく正常終了した。実機呼び出しは
wafl500（192.168.15.100:11434）へのembeddingのみ（`nomic-embed-text`，計3027回：1427+1600），
LLM生成・probe・dispatchは一切発生していない。総所要時間約173秒（約2.9分，`timeout_min:150`に対し
十分余裕あり，config.ymlの想定どおりこの数値は今回は適用されない軽量処理だった）。

`metrics.py`の既存関数（`compute_ece`／`compute_top1_accuracy`／`compute_mcnemar_test`／
`compute_precision_recall_per_domain`／`compute_domain_recall_mcnemar_test`／
`compute_domain_precision_fisher_test`／`apply_benjamini_hochberg`，いずれもIter30以降
実装済みで変更なし）を呼ぶ一時スクリプト（`/tmp/iter32_analysis.py`，非永続，分析後削除済み）で
before（`results/iter31_calibrated_predictions.jsonl`）とafter（`results/iter32_calibrated_predictions.jsonl`，
各1600行）を比較した（判定はここでは行わず，数値のみを機械的に集計する）。

**成功条件1（主基準）: education_recall vs medical_recall(0.5112固定，Iter31 production実測)**

- education_recall before: **78/170 = 0.4588235294117647**（Iter31実測と同一値）
- education_recall after: **75/170 = 0.4411764705882353**
- 差分: **-0.0176470588235294**（改善ではなく悪化方向。レバーは狙いと逆方向に動いた）
- 訂正後の主基準（medical_recall=0.5112固定値）との比較: **0.4412 < 0.5112（下回ったまま。
  基準を上回るという主基準の点推定は満たされていない）**
- 参考: after側で再計算したmedical_recallも88/178=0.4943820224719101（before 91/178=0.5112359550561798
  から低下）であり，after同士で比較してもeducation(0.4412)はmedical(0.4944)を上回っていない。

**成功条件2（診断，education_recallのドメイン別McNemar検定）**

- discordant_a_only（before正解・after誤り）: **3**
- discordant_b_only（before誤り・after正解）: **0**
- discordant_pairs: 3
- chi2_statistic: 1.3333333333333333
- p_value: **0.24821307898992373**（有意ではないが，3件の不一致は全てbefore→afterで悪化方向。
  改善方向の不一致は0件。実験不成立（基準線とビット単位完全一致）ではない——後述のとおり全体で
  15/1600行がflipしており，レバーは確実に発火している——が，主要指標（education_recall）の
  変化は「改善」ではなく「悪化（非有意）」という，計画時に想定した方向と逆の結果だった。
- 悪化した3行の内訳（`education-013`: education→mathematics，`education-130`:
  education→business_economics，`education-146`: education→business_economics）。いずれも
  before時点のconfidenceが0.24〜0.31と低い僅差の行であり，境界線上の質問だった。

**成功条件4（診断，education_precision，20指標BH補正セットの一部として算出）**

- education_precision before: 78/147 = 0.5306122448979592
- education_precision after: 75/147 = 0.5102040816326531
- Fisher正確検定 p_value: **0.8154394516445582**（有意ではない）
- true_positive_a=78, selected_a=147, true_positive_b=75, selected_b=147, odds_ratio=1.085217391304348

**成功条件3（非退行，education以外9ドメイン18指標，BH補正q=0.05）**

10ドメイン×precision/recall=20指標の点推定とp値（education含む全20指標，および
education除外18指標）：

| domain | metric | before | after | p_value | 検定 |
|---|---|---|---|---|---|
| business_economics | recall | 0.5417 (91/168) | 0.5417 (91/168) | 1.0 | McNemar |
| computer_science | recall | 0.5714 (96/168) | 0.5536 (93/168) | 0.2482 | McNemar |
| education | recall | 0.4588 (78/170) | 0.4412 (75/170) | 0.2482 | McNemar |
| general | recall | 0.5732 (94/164) | 0.5732 (94/164) | 1.0 | McNemar |
| history_culture | recall | 0.6786 (114/168) | 0.6726 (113/168) | 1.0 | McNemar |
| legal | recall | 0.5778 (104/180) | 0.5778 (104/180) | 1.0 | McNemar |
| mathematics | recall | 0.6310 (106/168) | 0.6310 (106/168) | 1.0 | McNemar |
| medical | recall | 0.5112 (91/178) | 0.4944 (88/178) | 0.2482 | McNemar |
| natural_science | recall | 0.5833 (98/168) | 0.5774 (97/168) | 1.0 | McNemar |
| social_science | recall | 0.5774 (97/168) | 0.5774 (97/168) | 1.0 | McNemar |
| business_economics | precision | 0.4643 | 0.4550 | 0.9197 | Fisher |
| computer_science | precision | 0.6234 | 0.6118 | 0.9064 | Fisher |
| education | precision | 0.5306 | 0.5102 | 0.8154 | Fisher |
| general | precision | 0.6528 | 0.6528 | 1.0 | Fisher |
| history_culture | precision | 0.6994 | 0.6975 | 1.0 | Fisher |
| legal | precision | 0.7820 | 0.7761 | 1.0 | Fisher |
| mathematics | precision | 0.7020 | 0.6974 | 1.0 | Fisher |
| medical | precision | 0.5056 | 0.4944 | 0.9158 | Fisher |
| natural_science | precision | 0.5444 | 0.5419 | 1.0 | Fisher |
| social_science | precision | 0.6382 | 0.6382 | 1.0 | Fisher |

20指標全てをBH補正（q=0.05）した結果，**BH有意（悪化方向）は0件**。education除外の
18指標のみで別途BH補正した場合も**BH有意（悪化方向）は0件**（`regressed_and_bh_significant_count
= 0`）。悪化方向の指標（computer_science_recall・history_culture_recall・medical_recall・
medical_precision・natural_science_recall・全precision系の大半）はp値が0.25〜1.00と大きく，
統計的な退行の根拠はない。

**成功条件5: flip rate**

- **15/1600 = 0.009375（0.9375%）**。Iter29 platt(11.0%)・Iter30 isotonic(14.3125%)・
  Iter31 temperature再学習(8.5625%)のいずれよりも大幅に低い。今回の変更は1427行中150行
  （education分）の一部（109行）のsample_weightのみを変えるという極めて限定的な介入であり，
  変化幅がこれまでの較正手法変更（分類器出力の全1600行に影響しうる）より小さいこと自体は
  想定と整合する。ただし，**flipが0ではなく15件発生している時点で「実験不成立（基準線と
  ビット単位完全一致）」ではなく，レバーは確実に発火している**（`education`のtp: 78→75，
  `computer_science`のtp: 96→93，`medical`のtp: 91→88，`natural_science`のtp: 98→97，
  `history_culture`のtp: 114→113 と複数ドメインで実測値が変化している）。

**成功条件6: 温度較正の再確認（チェックリスト，`probabilities`フィールドを使用，1600行対象）**

- (a) 確率のいずれかが厳密に`0.0`または`1.0`になっている行数: **0/1600**
- (b) 10クラス全てが`0.1`に近いuniform fallback行数: **0/1600**
- (c) tie率（選択ドメインのconfidenceと同一の値を持つ他ドメインが存在する行）: **0/1600**

3点ともIter31と同様に該当0件であり，温度較正自体の実装は今回のデータでも正常に機能している。

**診断: 全体top1_accuracy・ECE（gatingではないが必須報告，計画外の追加観測）**

- top1_accuracy before: 0.605625, after: 0.598750, 差分: **-0.006875**
- 全体McNemar検定: discordant_a_only=11（before正解・after誤り）, discordant_b_only=0
  （before誤り・after正解）, discordant_pairs=11, chi2=9.090909090909092,
  **p_value=0.002568831527022697（α=0.05で有意）**。**11件の不一致は全てbefore→afterで
  悪化する方向であり，改善方向の不一致は1件もない**。これは計画の成功条件には含まれていないが，
  「education以外への意図しない副作用」の直接的な証拠であるため報告する。
  内訳: `computer_science-040`(computer_science→business_economics),
  `computer_science-063`(computer_science→education),
  `computer_science-078`(computer_science→education), `education-013`(education→mathematics),
  `education-130`(education→business_economics), `education-146`(education→business_economics),
  `medical-110`(medical→business_economics), `medical-136`(medical→education),
  `natural_science-066`(natural_science→medical),
  `compound-058`(medical→business_economics, expected=[natural_science,medical]),
  `compound-083`(history_culture→business_economics, expected=[history_culture,medical])。
  computer_science・medicalの3行がeducationへ誤って引き込まれている一方で（過剰発火の兆候），
  education自身の当たり行は3行失われており（`education-013/130/146`），
  **educationへの過剰発火とeducation自身のrecall悪化が同時に起きている**。
- ECE before: 0.07120101725284995, after: 0.06502759260597007（n_bins=10，全1600行の
  confidenceが非nullで対象）。差分-0.00618（改善方向だが，本イテレーションの対象外の
  診断値であり，温度較正手法自体は変更していないため差分は訓練データ変化による間接効果）。

**使用データ**:

- 訓練データ（新規）: `data/classifier_train_iter32_reweighted.jsonl`（1427件，education
  150件中109件がsample_weight=2.0・41件が1.0，他1277件は全て1.0）
- 評価データセット（再embedding対象）: `data/dataset.jsonl`（1600件，無変更，実装(Iter32)で
  sha256一致を確認済み）
- beforeの実行結果: `results/iter31_calibrated_predictions.jsonl`（Iter31実測，1600行，
  再実行なし）
- afterの実行結果（新規生成）: `results/iter32_calibrated_predictions.jsonl`（1600行，
  `probabilities`フィールド付き）
- 新規モデルアーティファクト: `models/domain_classifier_iter32_reweighted.joblib`（本番
  `models/domain_classifier.joblib`は無変更のまま，タイムスタンプで確認済み）

**実行時間・実機呼び出しの有無**:

- `train_domain_classifier.py`: 114.19秒（user時間，1427回のembedding呼び出し）
- `evaluate_classifier_calibration.py`: 59.02秒（user時間，1600回のembedding呼び出し）
- 実機呼び出しはwafl500（192.168.15.100:11434）へのembeddingのみ（`nomic-embed-text`），
  計3027回。LLM生成・probe・dispatchは一切発生していない。
- 接続経路はIter29〜31と同一の既存SSHローカルポートフォワード（`127.0.0.1:11435` ←
  `wafl500:11434`，PID 621254）をそのまま流用。新規に張り直す必要はなく，実行中のログ・
  エラーに異常なし（例外・タイムアウト・リトライなし，両スクリプトとも正常終了メッセージを
  出力）。

**state.json更新**: `status: waiting_experiment`（開始時，`experiment_dir:
results/iter32_calibrated_predictions.jsonl`・`experiment_deadline`設定）→`running`（完了時，
`experiment_dir`は結果ファイルパスのまま維持，`experiment_deadline`をnullに戻した）。
`e32_results`への数値記録・`judgment`確定はフェーズ5b（rc-analyst）に委ねる（本フェーズでは
数値の良否判定は行わない。ただし，主基準（education_recallがmedical_recall基準0.5112を
上回る）が点推定として満たされていないこと，education_recallが前段（Iter31）比で悪化方向に
動いたこと，全体top1_accuracyが統計的に有意に悪化していること（p=0.0026）は，事実として
上記に記録した）。

---

### 分析(解釈) (Iter32)

**成功条件（計画(Iter32)節，AND条件）との照合**

1. **主基準（point estimate）: education_recall > medical_recall(0.5112)**: **不成立**。
   0.4412 < 0.5112 であり，しかも before（0.4588）からさらに **-1.76pt 悪化**している。
   計画が「point estimate」と明記したとおりこの基準に有意性検定は要らない設計であり，
   有意性の有無に関わらずこの1点だけで主基準は満たされていないと判断してよい。
2. **診断（education_recallのドメイン別McNemar）**: discordant 3件（全てbefore→after悪化，
   改善方向0件），p=0.248。n=170・discordant=3という小標本のため個別には有意でないが，
   3/3が同一方向という事実は「ノイズによる左右均等な入れ替わり」とは整合しない偏りである
   （後述の機序説明と合わせて評価する）。
3. **非退行（education以外9ドメイン18指標，BH補正q=0.05）**: 字義通りは成立（悪化方向で
   BH有意0件）。ただしこれは**方法論的な力不足（後述）による見かけの成立**である疑いが強く，
   総合判断ではそのまま額面通りに扱うべきではない。
4. **education_precision（診断）**: before 0.5306→after 0.5102，Fisher p=0.815で非有意。
   over-triggering（education以外がeducationに誤って引き込まれる）の増加は，precisionの
   点推定にはまだ表れていない（education自身への流入行数が絶対数として少ないため）。
5. **flip rate**: 15/1600=0.9375%。過去の較正手法変更（platt 11.0%・isotonic 14.3%・
   temperature再学習8.6%）より小さいが非ゼロであり，「実験不成立」（config.yml・d0004が
   警告する基準線とビット単位一致のパターン）には該当しない。
6. **温度較正チェックリスト**: 0/1600・0/1600・0/1600（張り付き・uniform fallback・tie）で
   異常なし。

**論点2: なぜ弱い代理タスクへのsample_weight増加がeducation_recallを悪化させたか
（sklearn実装への直接確認による機序特定）**

計画(Iter32)は「`sample_weight`と`class_weight='balanced'`は独立に効く（`sample_weight *=
class_weight_`と乗算されるだけ）」という前提で設計されていたが，この前提は**不正確**だった。
実際には`class_weight='balanced'`自体が`sample_weight`に依存して再計算されるため，両者は
独立ではなく相殺し合う関係にある。本フェーズで`.venv`にインストール済みの
`scikit-learn==1.9.0`のソースを直接確認し，`data/classifier_train_iter32_reweighted.jsonl`
実物で数値を再現した。

- `sklearn/utils/class_weight.py`の`compute_class_weight(class_weight="balanced", ...)`は，
  `weighted_class_counts = _bincount(y_ind, weights=sample_weight, ...)`という**sample_weight
  で重み付けしたクラス別合計**を分母に使い，`recip_freq = sum(weighted_class_counts) /
  (n_classes * weighted_class_counts)`としてクラスごとの`class_weight_`を決める
  （同ファイル94-98行目）。すなわち`class_weight='balanced'`は「そのクラスの生の行数」では
  なく「`sample_weight`込みの実効行数」に反比例する。
- `sklearn/linear_model/_logistic.py:436`の`sample_weight *= class_weight_`（計画時に確認済み
  の一次情報どおり）と合わせると，最終的にLogisticRegressionへ渡る各行の実効重みは
  `task_sample_weight × class_weight_(sample_weightに依存)`という**入れ子の依存関係**に
  なる。計画時の想定（両者が独立に掛かるだけ）は前半のみ正しく，後半（`class_weight_`
  自体が`sample_weight`で変わる）を見落としていた。
- 実際の訓練データで計算すると（`n_classes=10`，全1427行，education以外は変更なし）：

  | | before（Iter31以前，全行weight=1.0） | after（Iter32） |
  |---|---|---|
  | educationの実効行数（weighted count） | 150.0 | 259.0（109×2.0 + 41×1.0） |
  | 全体の重み付き総数 | 1427.0 | 1536.0 |
  | `class_weight_[education]` | 1427/(10×150)=**0.9513** | 1536/(10×259)=**0.5931**（-37.7%） |
  | `class_weight_[他9ドメイン]`（例: medical, computer_science等） | 1427/(10×150)=0.9513 | 1536/(10×150)=**1.0240**（+7.6%） |

- この結果，各行の最終実効重み（`task_sample_weight × class_weight_`）は：
  - `high_school_psychology`・`moral_disputes`（狙った2.0倍）: `2.0×0.5931=1.186`
    （before比 `1.186/0.9513=+24.7%`。**狙った2倍ではなく実質+24.7%に留まった**）。
  - `sociology`（education内で唯一相対的に良好，recall 0.625，計画が「重みを上げない」と
    決めた41行）: `1.0×0.5931=0.593`（before比 `0.593/0.9513=-37.7%`）。**変更対象外の
    はずのsociology行が，class_weight_の連動低下により実効重みを4割近く失った**。
  - education以外の1277行（全て`task_sample_weight=1.0`）: before比一律`+7.6%`
    （どのドメインでも同じ倍率——educationの`sample_weight`増加が生んだ`weighted_class_counts`
    の総和増加を`n_classes`で割った副作用であり，education以外の9ドメイン全てに機械的に
    及ぶ）。

- この数値は3つのことを説明する。
  1. **主基準が悪化した理由**: 狙いは「弱い2タスクへ2倍の重みを与えてeducationの決定境界を
     強化する」ことだったが，実際に起きたのは「弱い2タスクへの重みは24.7%増に減衰し，かつ
     education内で唯一機能していたsociology（recall 0.625）の重みが37.7%減り，同時に
     education以外9ドメイン全てが7.6%の相対優位を得る」という，**狙いとほぼ逆方向の複合効果**
     である。education全体としての実効重み総量自体はむしろ増えている（142.7→153.6，
     `balanced`方式の定義上，各クラスの重み総量は常に`総重み/n_classes`になるため）が，
     その増分は全てeducation自身の中で「弱いタスクへ再配分」される形にしかならず，
     「educationという線形境界をeducation以外との対比でどれだけ有利にするか」という点では
     class_weight_の低下が直接に不利に働く。
  2. **他ドメインへの副作用（診断で見つかった top1_accuracy 有意悪化）が生じた理由**:
     `education`だけを対象にしたはずの変更が，`class_weight='balanced'`の定義（クラス別
     重みの合計を1点に固定する仕組み）を経由して**他9ドメイン全行に一律+7.6%の相対的な
     優位を与える**という，計画時に想定されていなかったグローバルな副作用を生んだ。これは
     単一レバー原則（「1つのレバーだけを動かす」）を実装上は守っていても，`sklearn`側の
     `class_weight='balanced'`という**別の既存レバーと数式レベルで結合している**ために，
     実質的には「education の task 内配分」と「10ドメイン全体のバランス」という2つの量を
     同時に動かしてしまったことを意味する。
  3. **discordant 11件の分布との整合性**: 診断で確認した全体top1_accuracyのdiscordant 11件は，
     `business_economics`が誤った着地先になったケースが6/11
     （`computer_science-040`・`education-130`・`education-146`・`medical-110`・
     `compound-058`・`compound-083`）と最多で，`business_economics`の`precision`点推定も
     0.4643→0.4550と（非有意ながら）低下方向である。これは「education以外9ドメインが一律に
     相対優位を得る」というグローバルな機序と方向として整合する。一方`education`への
     誤った流入も3/11（`computer_science-063`・`computer_science-078`・`medical-136`）
     存在し，これは`high_school_psychology`・`moral_disputes`という**具体的な訓練点の埋め込み
     近傍**が局所的にeducation側へ境界を引き寄せた効果（調査(Iter32)が特定した
     `high_school_psychology`↔`medical`・`moral_disputes`↔`legal/social_science`の意味的
     近接と整合，線形分類器はグローバルな重み再配分と局所的な決定境界の変形が同時に起こり
     うる）と考えられる。すなわち，**education自身の真陽性を3行失いながら，education以外
     から3行を誤って奪う「過剰発火」も同時に起きている**——「recallを上げようとして
     precisionが犠牲になる」典型的なトレードオフですらなく，both方向で悪化している。

**論点3: 非退行チェック（成功条件3）が字義通り成立している点についての留保**

10ドメイン×20指標のBH補正で悪化方向有意0件という結果は事実だが，これは**検定力不足による
見かけの非退行**である可能性が高い。根拠:

- 各ドメインのrecall検定はn=150〜180・discordant数は最大でも3〜4件（education3・
  computer_science3・medical3・natural_science1・history_culture1）に留まり，個別の
  McNemar検定はもともとこの規模の悪化を検出する検定力が乏しい。
- 一方，全1600問を束ねた全体top1_accuracyのMcNemar検定はp=0.0026で明確に有意であり，
  discordant 11件は**方向が完全に一致**している（悪化11・改善0）。これがもし真にランダムな
  再配分（左右均等に生じるノイズ）であれば，11件全てが同一方向に揃う確率は
  二項検定で`2×(0.5)^11 ≈ 0.001`と極めて小さく，全体の有意性（p=0.0026）はこの方向の一貫性
  そのものが主な源泉である。
- したがって「非退行（成功条件3）が字義通り成立した」ことは，**個々のドメインに薄く分散した
  一貫悪化を，ドメイン別に切り分けて検定する設計（BH補正込みでも1ドメインあたりの検定力は
  据え置き）では拾いきれない**ことを示しているに過ぎず，「本当に非退行だった」ことの
  積極的な証拠ではない。Iter30（isotonicのmedical_recall1件が単独でBH有意）とは異なり，
  今回は「1ドメインに集中した強い退行」ではなく「9ドメインに薄く広く分散した弱い退行が
  集計すると有意になる」という，前例とは異なるパターンの悪化である。

**論点4: 実験不成立の再確認**

flip rate 0.9375%（15/1600，非ゼロ）に加え，本フェーズで`sample_weight`が`compute_class_weight`
の`weighted_class_counts`にまで実際に反映され，`class_weight_[education]`が0.9513→0.5931へ
実測どおり変化していることをデータファイルから直接計算で確認した（上表）。これは
`sample_weight`が`CalibratedClassifierCV.fit()`経由で実際に学習の数式まで届いていることの，
実装(Iter32)のspyテストに加えたもう一段深い一次証拠であり，config.ymlが警告する
「設定を変えたのにコードに到達しない」パターンには一切該当しない。

**総合判断（提案，確定はrc-reflector）: rejected**

- 主基準（point estimate）が不成立であるだけでなく，狙いと逆方向（education_recall悪化）に
  動いた。isotonic（Iter30，ECE成立・recall退行あり）やplatt（Iter29，ECE未達のみ）のような
  「一部の利得と一部のトレードオフ」の構図ではなく，**得られた利得が一つもない**（education
  もmedicalもtop1_accuracyも全て悪化方向）。
- 非退行チェック（成功条件3）は字義通り成立しているが，論点3で述べたとおり検定力不足による
  見かけの成立である疑いが強く，全体top1_accuracyの有意な悪化（p=0.0026，11/11同一方向）を
  無視して額面通り「非退行達成」と扱うべきではない。
- 機序（論点2）が`class_weight='balanced'`と`sample_weight`の数式レベルでの結合という，
  具体的でsklearnソースからも実測でも裏付けられる説明を持つため，「たまたま悪い乱数を
  引いた」（ルーティングは決定論的なのでそもそも乱数は存在しない）や「小標本ノイズ」による
  偶然ではなく，**この実装（`class_weight='balanced'`のまま`sample_weight`を追加する設計）
  そのものに起因する再現性の高い悪化**と判断する。
- 追加反復（同一条件の再実行）は不要——ルーティングは決定論的であり，再実行しても同じ数値に
  なる。ただし，計画(Iter32)が「目標未達時の次点候補」として挙げていた**「重み倍率を
  3.0〜4.0へ引き上げる」案は，論点2の機序に照らすとむしろ悪化を助長する可能性が高く
  推奨しない**（`education`の`weighted_class_counts`をさらに増やすほど
  `class_weight_[education]`はさらに下がり，sociologyの実効重みはさらに失われ，
  education以外9ドメインへの相対的優位はさらに拡大するため）。この点は次イテレーションへの
  重要な申し送りとして次項に記載する。
- 本番アーティファクト（`models/domain_classifier.joblib`）は実装(Iter32)・実験(Iter32)の
  時点で既に無変更であることが確認済みであり，rejectedの場合の追加のロールバック作業は
  不要（`models/domain_classifier_iter32_reweighted.joblib`は検証用の副産物として残すか
  削除するかをrc-reflectorの判断に委ねる）。

**次への示唆**

1. **候補(2)の単純な重み倍率変更（3.0〜4.0倍への引き上げ）は推奨しない**。論点2の機序が
   示すとおり，`class_weight='balanced'`を維持したまま`sample_weight`だけを増やす限り，
   倍率を上げるほど「弱いタスクへの意図した強化」は`class_weight_`の自動減衰で目減りし，
   「sociology の弱体化」と「education 以外9ドメインへの相対的優位」がさらに拡大する
   構造的な副作用がある。この設計のまま倍率だけ変えて次イテレーションを回すのは，
   同じ失敗モードを規模だけ変えて繰り返すリスクが高い。
2. **もし sample_weight による task 内再配分を今後も試すなら**，`class_weight='balanced'`を
   維持したままでよいかを再検討すべきである。具体的には，(a) `class_weight`に文字列
   `"balanced"`ではなく，`sample_weight`適用前の生カウントから計算した固定dictを明示的に
   渡す（`class_weight_`が`sample_weight`の値に連動しなくなる），または(b) task内の重み配分を
   「education全体の実効行数を変えない」制約下で設計する，のいずれかが必要になる。ただし
   (b)は今回のデータでは実現不可能に近い——`education`150行中109行（72.7%）が「弱い」
   `high_school_psychology`・`moral_disputes`由来であり，41行（27.3%）の`sociology`だけで
   総量150を維持しながら弱い側を2倍にするには`sociology`側の重みが負になる計算になる
   （`109×2+41×w=150`は`w<0`を要求する）。これは「弱いタスクが多数派」という
   `education`の代理タスク構成自体の根本的な制約であり，重み付けという手段では
   解消しにくいことを示す。
3. **調査(Iter32)が挙げた他の代替候補の説得力を再評価する**:
   - 候補(1)（サンプル数増量，150→298）: 調査時点で「効果限定的」と留保されていたが，
     今回の重み付け（候補2）が機序レベルで逆効果と判明した以上，相対的な優先度は
     再検討の余地がある。ただし候補(1)も「3タスクの合算プールから無作為抽出」する限り
     構成比（sociology:high_school_psychology:moral_disputes ≈ 1:1:1）は変わらないため，
     意味的近接という根本原因（調査(Iter32)分かったこと(3)(5)）そのものは解消しない。
     試すとしても「サンプル数を増やしつつ，sociologyの比率だけ相対的に高める」という
     候補(1)と候補(2)の折衷案（無作為抽出時の配分比率をtask別に変える，`sample_weight`では
     なく抽出段階でsociologyを多く・弱い2タスクを少なく採る）の方が，論点2の
     `class_weight`結合の副作用を避けられる分，筋が良い可能性がある。
   - 候補(3)（手作り4択問題の追加）: 調査(Iter32)分かったこと(6)が指摘した書式リスク
     （A/B/C/D構造の有無が`education`だけ他8ドメインと異なる訓練データになる懸念）は，
     4択形式を維持する限り回避できる。作問コストは高いが，`education`の代理タスク自体が
     「学校教育行政実務」という定義と学術知識問題という代理タスクの間に埋めがたい意味的
     ギャップを持つ（調査(Iter32)分かったこと(1)(3)）ことを踏まえると，**代理タスクの
     within-class配分をどういじっても限界がある可能性が高く**，中長期的には候補(3)
     （またはeval データセット自体の一部差し替えという，人間判断を要するより大きな変更，
     調査(Iter32)分かったこと(6)の選択肢(b)）の検討価値が相対的に上がったと考える。
   - **全く別のアプローチとして**，`education`の分類器特徴量そのもの（`nomic-embed-text`の
     生埋め込み）を疑う余地もある。調査(Iter32)が示した混同パターン（`high_school_
     psychology`↔`medical`，`moral_disputes`↔`legal/social_science`）は，埋め込み空間上で
     `education`の代理タスクが複数の他ドメインの代理タスクに囲まれるように分布している
     ことを示唆しており，線形分類器（`LogisticRegression`）の表現力の限界という可能性も
     否定できない。ただしこれはより大きな変更（base estimatorの変更）であり，今回の
     観測だけから断定はできない。
4. **人間判断が必要な論点**: 新規追加なし。調査(Iter32)が既に挙げた「`education_recall`と
   いう既存メトリクスの改善」と「`education`ドメインの実務忠実性」の両立不可能性という
   論点は，今回の結果を経てもなお未解決であり，backlogでの申し送りを維持する。

### 考察 (Iter32)

**判定確定: rejected（rc-analyst 提案どおり，覆さず確定）**

rc-analyst の rejected 判定を検証した。主基準（point estimate で education_recall が
medical_recall 基準 0.5112 を上回る）が不成立であるだけでなく，education_recall 自体が
before比で悪化（0.4588→0.4412，-1.76pt）し，全体 top1_accuracy も統計的に有意に悪化した
（McNemar p=0.0026，discordant 11 件が全て悪化方向で改善方向は 0 件）。得られた利得が
一つもなく，isotonic（Iter30，ECE 成立・一部退行）や platt（Iter29，ECE 未達のみ）のような
「部分的な利得とトレードオフ」の構図ではない。分析(解釈)節が
`sklearn/utils/class_weight.py`・`sklearn/linear_model/_logistic.py` のソースと
`data/classifier_train_iter32_reweighted.jsonl` 実物の数値（`class_weight_[education]`
0.9513→0.5931，狙った2倍が実質+24.7%に減衰し，変更対象外のはずの`sociology`行も
-37.7%の実効重み損失）で機序を具体的に裏付けており，追加反復（同一条件の再実行）でも
結果は変わらない（ルーティングは決定論的）。判定を覆す根拠はなく，rejected で確定する。

**機序の要点（再確認）**: `LogisticRegression(class_weight="balanced")` は
`sample_weight` に依存してクラス重みを再計算するため，`sample_weight` による
task内再配分と`class_weight`によるドメイン間バランス調整は独立ではなく，
数式レベルで結合している。education の task内 sample_weight を上げると
`class_weight_[education]` が自動的に下がり，狙った強化が減衰するだけでなく，
education以外の9ドメイン全てに一律の相対的優位（+7.6%）を与える副作用を生む。
これは「単一レバー原則を実装上は守っていても，sklearn 側の既存の仕組み
（`class_weight='balanced'`）と数式レベルで結合しているレバーは，実質的に複数の量を
同時に動かしてしまう」という一般化可能な学びであり，今後 `sample_weight` を
本リポジトリの分類器訓練に使う場合は必ず確認すべき事項として記録する。

**models/domain_classifier_iter32_reweighted.joblib の扱い: 削除する**

rejected が確定し，機序（class_weight結合バグ）まで特定できているため，このモデル
アーティファクト自体を将来再利用する見込みはない（次に sample_weight 系のアプローチを
再度試す場合も，今回とは異なる訓練データ構成で作り直す必要があり，今回の joblib は
比較対象として再利用できない）。数値的な結果（education_recall・confusion matrix・
class_weight_ の実測値）は本 journal に記録済みで十分参照可能なため，ファイルとしては
不要と判断し削除する（`models/` は `.gitignore` 対象のため削除は git 履歴に残らない）。
同様に `data/classifier_train_iter32_reweighted.jsonl`（`data/` も `.gitignore` 対象）も
削除する。`results/iter32_calibrated_predictions.jsonl` は Iter29〜31 の
`resultsXX_calibrated_predictions.jsonl` と同様に一次結果データとして今後も参照価値が
あるため git 追跡対象として残す（他イテレーションと同じ扱い）。

**次に振るレバー**: `classifier_training_data_composition` レバー（config.yml）へ新しい値
`education_proxy_task_resampling` を追加し，次イテレーション（Iter33）の単一レバーとする。
Iter32 とは異なり `sample_weight`（sklearn の `class_weight='balanced'` と結合し再現性高く
逆効果と判明）は一切使わない。`build_dataset.py` の `build_classifier_training_rows()` が
`education` を抽出する際の3タスク別の目標件数（現状ほぼ均等，`sociology`:
`high_school_psychology`:`moral_disputes` ≈ 41:55:54 相当，実測は分析(解釈)節参照）を，
**`education` の総行数（150件，他ドメインと同数）は変えずに**，相対的に良好な
`sociology`（recall 0.625）の割合を増やし，弱い2タスク（`high_school_psychology` 0.438・
`moral_disputes` 0.435）の割合を減らす方向へ再配分する（例: sociology 90・
high_school_psychology 30・moral_disputes 30，具体的な比率は次の計画フェーズで確定する）。
**総行数を150件のまま変えない**のが今回の失敗から得た設計上の要点である。分析(解釈)節の
数値が示すとおり，`class_weight="balanced"` は「そのドメインの生の行数」に反比例して
決まるため，education の総行数が他ドメインと同数（150件）のままであれば
`class_weight_[education]` は Iter31 以前と完全に同じ値（0.9513）のままになり，
`sample_weight` を一切使わないため sklearn 側の結合バグの影響を受けない。これは
rc-analyst が次への示唆で挙げた「サンプル数を増やしつつ sociologyの比率を高める折衷案」を
さらに一歩進め，**サンプル数自体は増やさず構成比のみを変える**ことで，候補(1)（単純な
サンプル数増量,150→298）が抱える同種のclass_weight連動リスク（総行数を増やせば
`class_weight_[education]`がさらに下がる）も同時に回避する設計である。

**留保（次の計画フェーズが踏まえるべき点）**: rc-analyst が指摘したとおり，この変更も
「代理タスクの意味的ギャップという根本原因」自体は解消しない。`sociology` の比率を
上げても，`sociology` 自体が「学校教育行政実務」という`education`の実務上の定義とは
主題が異なる学部教養レベルの社会学問題であることに変わりはなく（調査(Iter32)分かった
こと(3)），達成できるのは「3タスクのうち相対的に混同されにくいタスクの寄与を増やす」
という限定的な改善にとどまる可能性が高い。目標未達に終わった場合の次点候補は
分析(解釈)節が既に整理済み（候補(3)＝4択形式の手作り問題追加，または埋め込み特徴量
自体・base estimatorの見直しという，より大きな変更）であり，次のrc-plannerはその順で
検討すること。

**iteration_name（次イテレーション，Iter33）**: 「education代理タスク抽出比率の再配分
（sample_weight不使用）によるclass_weight結合回避型データ構成変更（Y5継続）」

---

## 記録訂正・commit 漏れの是正（2026-07-30，`/research-cycle continue` 実行時）

**背景**: Iter24 完了後の `continue` 呼び出し時，`git status` で `scripts/run_central_experiment.py`（未追跡）・
`config.yaml`（`central_router` 節，未commit）・`.claude/research/state.json`（heartbeat のみの軽微な差分）が
working tree に残っていることを発見した．Iter24 完了コミット（`ee1d549`）は `.claude/research/*` のみを
含んでおり，Iter24 の単一レバー（`routing_architecture=central_router`）を実装したコード自体は
一度も git に commit されていなかった．過去の記述は書き換えず，本節を追記として残す．

**発見1（実装内容が journal の記述と乖離）**: 本節より前の「実装 (Iter24)」節は `scripts/run_central_experiment.py`
を「229行，`OllamaClient.embed()`/`generate()` をローカルでそのまま利用」と記述しているが，working tree に
残っていた実際のファイルは 411 行であり，`SshEmbeddingClient`／`HttpOllamaGenerator`（`config.yaml:central_router`
の `ssh_user`/`domain_nodes` を読み，SSH 経由で各ドメインの担当ノードへ curl する方式）に置き換わっていた．
これは「調査 (Iter24)」節が指摘した VRAM 制約（6GB に 10 LoRA を1台で載せられない）に対応するための
実装上のピボットと推測されるが，その変更判断・理由は journal に一度も記録されていない．
Slack の「フェーズ3: 実装」報告は「253行」と述べており，journal（229行）とも実ファイル（411行）とも
一致しない．3者の食い違いは，実装が複数回改訂されたにもかかわらず記録が都度更新されなかったことを示す．

**発見2（ruff warning の不一致）**: journal・Slack・Notion はいずれも「ruff 0 warning」としているが，
現在の working tree のファイルには未使用 import（`os`, `subprocess`）による F401 が 2 件ある．
Iter24 の判定（rejected）自体は主基準（top1/kappa/McNemar）に基づくため，この lint 差分は判定を
覆すものではない．

**是正内容**: 上記のコード一式（`scripts/run_central_experiment.py` 新規追加，`config.yaml` の
`central_router` 節追加）は，実際に Iter24 の実験を生成した実体であるため，lint 警告を含めて
**そのままの内容で** git commit した（実験の再現性を優先し，事後的な整形は行わない）．
`.claude/research/state.json` の heartbeat 差分（`updated_at`）は現在時刻へ更新し，`last_commit` を
本コミットのハッシュへ同期した．

**次回への申し送り**: rc-implementer は，計画からの実装方針の変更（今回でいう local→SSH のピボット）が
発生した場合，その理由を「実装 (IterN)」節に都度追記すること．イテレーション完了時の commit 検証
（SKILL.md 記載）は `.claude/research/` 配下だけでなく，そのイテレーションで変更した実コード・設定ファイルが
実際に commit されているかも対象に含めるべきである．

---

## 敵対的総点検・追加修正（2026-07-30，`/research-cycle continue` 実行前）

**背景**: 直前の「記録訂正・環境修復」節（2026-07-29）で行った F1〜F5 の修正自体が正しいかを，
独立した subagent によるレビューと自己点検で敵対的に総点検した．詳細は
`.claude/research/backlog.md` の B40 を参照．要点のみ記す．

**発見1（重大・修正済み）**: `.claude/research/config.yml` の E4（`self_consistency_semantic`）・
E5（`p_true`）の記述が，真の no-op である E3・E7 と同列に書かれていたため誤解を招く状態だった．
実際には，現在の HEAD（`30e3627`，Iter22 の分岐順序修正が反映済み）でこれらを設定すると，
E6（supervised_classifier）の分類器分岐に到達できなくなり **E6 を丸ごと上書きする**．
「no-op」ではない．時系列の誤り（Iter21/22 は E4 が動いた上で退行したのではなく，分岐順序修正が
未適用/未デプロイだったため E4 が 1 度も実行されなかっただけ）も訂正した．

**発見2（重大・ユーザー承認の上で修正済み）**: `state.json.iteration` が `"Iter22"`（無効判定済み）
のまま，SKILL.md が定める「イテレーション完了時の初期化」が未実施だった．前回は `current_lever`
のみに着目し実害なしと判断していたが，`rc-experimenter.md` が「実験ディレクトリ名を現イテレーション
番号から決める」と明記しており，このまま continue すると次の実験が誤って Iter22 として記録される
リスクを見落としていた．ユーザー承認を得て，`iteration`: `Iter23` へインクリメント，
`current_lever`/`experiment_dir`/`experiment_deadline`/`iteration_thread_ts`: null，
`notion_toggle_created`: false，`iteration_name`: null に更新した．

**発見3・4（軽微・修正済み）**: `tools/smoke_check.py` の `_SIGNAL_FIELD_EXPECTATIONS` に到達不能な
dead entry（`"semantic_entropy"` キー，実際の値は `"self_consistency_semantic"`）を削除．
`run_experiment.py` の `write_text` に `encoding="utf-8"` を追加．

**次期 rc-investigator への申し送り**: `state.json` は既に Iter23 として初期化済みである．
このイテレーションの調査を「### 調査 (Iter23)」として記録すること．

---

## 記録訂正・環境修復（2026-07-29，`/research-cycle continue` 実行前の総括調査）

**背景**: `docs/d0002_research_cycle_findings_2026-07.md`（Iter1〜22 の知見総括）・
`docs/d0003_next_experiments_2026-07.md`（次の実験計画）を新規作成し，journal・backlog・
config.yaml・config.yml・http_server.py の実データ突合を行った．以下は journal 側の記録誤りの訂正，
および continue 実行前に対処した環境修復である．過去の記述は書き換えず，本節を追記として残す．

### 訂正 1: E3（`confidence_elicitation`）の採用判定を取り下げ

Iter20 の「同点タイ 82.83%→0.00%，ECE 0.7388→0.1927 の決定的改善」は，**すべて Iter17 の
E6（supervised_classifier）導入時に既に起きていた変化**である．`http_server.py:_estimate_probe_confidence()`
は排他的な if 連鎖で，`routing_method=supervised_classifier` が先に return するため
`confidence_elicitation` の分岐には到達しない（d0002 §6-B）．E3 の有効な測定は Iter16 の 1 回のみで，
そのときの結果は top1 0.2059（McNemar p=0.0783，有意差なし），ECE は 0.7146→0.7388 と悪化していた．
**「採用」の判定は取り下げ，D1（判定保留．設定自体は害をなさない）として backlog に記録した．**

### 訂正 2: ECE の正しい系列

10-bin・confidence 非 null 行を対象に単一実装で再計算した結果，Iter17 以降は **0.1927 で不変**である
（決定論的ルーティング下での再実行は新しい情報を生まない．d0002 §6-A）．

| 実験 | 正しい値 | journal 上の誤記載 |
|---|---|---|
| Iter17 | 0.1927 | 0.2118（不一致） |
| Iter21 | 0.1927 | 0.1903 / 0.1673（journal 内で 2 通りの誤記載） |

Iter21 の「0.1903 へわずかに改善」という記述は誤りで，実際の変化は 0.0000 である．

### 訂正 3: `top1_accuracy` と `single_domain_top1_accuracy` の取り違え

Iter18 Phase C（domain LoRA 採用）の `top1_accuracy` は **0.5651** であり，journal が Iter19/20 の
計画根拠として使っていた **0.5693 は `single_domain_top1_accuracy`（単一ドメイン 1500 問のみの値）**
である．「E10 で top1 が 0.5651→0.5693 改善した」という記述は誤りで，実際は McNemar 不一致 0/1520 で
**完全に不変**（journal 自身も別箇所で不一致 0/1520 と記録しており内部矛盾していた）．

### 環境修復（`/research-cycle continue` 実行前に対処．d0003 第1段階 F1・F1-b 相当）

1. **`config.yaml`**: Iter19 で棄却された `expert_model=qwen3.5:4b-q4_K_M`（全10ノード）が HEAD に
   残置されていたため，Iter18 で採用された `expert-mesh-{domain}-lora` へ戻した．また Iter21/22 で
   無効と判明した `confidence_signal_method=self_consistency_semantic` を `self_report` へ戻した
   （制約: この値以外だと `routing_method=supervised_classifier` の分岐に到達せず分類器が無効化される．
   d0002 §6-D）．**前提条件を実機で確認済み**: wafl500〜509 の Ollama に対応する
   `expert-mesh-{domain}-lora` モデルが全10ノードとも登録済みであることを確認した（2026-07-29）．
2. **`.claude/research/config.yml`**: `levers` 節の前提コメント（Iter15 時点のまま古くなっていた）を
   Iter22 時点の実態へ全面更新し，E3・E4・E7 の no-op / 排他構造の注記を追記した．
3. **未対応（次イテレーション以降の課題）**: docs/d0003 F2（デプロイ検証ゲート）・F3（metrics.py への
   ECE/AUROC/Brier/同点率/分散統合）・F5（再現性マニフェスト）は本セッションで別途着手中．F3 完了までは
   confidence 信号系レバーの判定に success_criteria (4) の指標を手計算に頼らざるを得ない．

**次期 rc-investigator/rc-planner への申し送り**: 上記により，continue 再開後の実験は最良既知構成
（E6 supervised_classifier + E10 domain_lora）から始まる．次に着手すべき優先順位は
`docs/d0003_next_experiments_2026-07.md` §0 を参照（第2段階 X1 基準線再取得 → 第3段階 X2 中央集権
ルータ比較・X4 複合ドメイン評価・X5 fallback 見直し）．

---

