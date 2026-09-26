## Iteration 80: 埋め込みモデルのスケールアップ（qwen3-embedding:0.6b → 4b）

### 調査 (Iter80)

B125(b) により本反復のレバーは `embedding_model_replacement` = `qwen3_embedding_4b`（config.yml の `values` にある未試行の第 2 値）で確定しており，レバー選定の裁量は無い．調査の問いは 3 つ．**(Q1) `qwen3-embedding:4b` は 0.6b に対して本タスク（日本語 10 ドメイン分類）で上積みが見込めるか．(Q2) 10 ノード（RTX 3060 12GB）の VRAM に収まるか．収まらない場合の代替は何か．(Q3) 次元数の増加（1024 → 2560）が 1,427 行の LogisticRegression 訓練に与える影響は何か．**

**Q1: 公称値は一貫して 4b > 0.6b．ただし日本語分類の直接値は無い**

- Ollama 公式ライブラリの `qwen3-embedding:4b` は **2.5GB（Q4_K_M），native 2560 次元，40K context**（<https://ollama.com/library/qwen3-embedding/tags>，2026-09-26 確認）．0.6b は 639MB・1024 次元・32K context．MTEB multilingual は **0.6b 64.33 / 4b 69.45 / 8b 70.58**（morphllm のベンチ一覧 <https://www.morphllm.com/ollama-embedding-models>，および config.yml lever note と同値）．
- Qwen 公式の Model Card（<https://huggingface.co/Qwen/Qwen3-Embedding-4B>，GitHub <https://github.com/QwenLM/Qwen3-Embedding>）では MMTEB 系の平均が **0.6B 70.70 → 4B 74.60 → 8B 75.22**，CMTEB 系の行でも 4B が 0.6B を一貫して上回る．**サイズ方向の単調性は複数のベンチで一致している**（事実）．
- ただし**日本語（JMTEB）の Classification は 4B の公開値が見つからなかった**．hotchpotch の JMTEB 計測（<https://secon.dev/entry/2025/06/11/100000-qwen3-embedding-jmteb>）は 0.6B のみ（Classification 66.09）で，4B 行は無い．**「4b が日本語分類で 0.6b を上回る」は公称値からの外挿であり，未検証の推測である**．Iter79 の学び 2（公称ベンチは候補を絞る道具であって採否の根拠にならない．MTEB 差 +2.05pt に対し実測 +16.34pt だった）をそのまま適用し，本反復も **G1 オフライン CV で本走前に数値化する**．
- 同時比較する `bge-m3` は Ollama 公式・**1024 次元・約 1.2GB・prefix 不要**（同ベンチ一覧）．JMTEB の Classification 値は上記記事に無いが，多言語モデルであり VRAM の退避先として位置づけられる．

**Q2: VRAM は本反復の最大のリスク．実測では余裕が 3.4GB 程度しかない**

wafl500・wafl503・wafl509 で `nvidia-smi` と `ollama ps` を実測した（2026-09-26，read-only）．

| 項目 | 実測 |
|---|---|
| GPU | RTX 3060 **12288 MiB**（全ノード共通） |
| wafl500 現況 | used 7621 MiB / free 4290 MiB．resident は `qwen3-embedding:0.6b` **2.4GB** ＋ `expert-mesh-general-lora` **5.3GB**（light_model は idle で未ロード） |
| wafl503/509 現況 | `expert-mesh-*-lora` 5.3GB ＋ `qwen3.5:4b-q4_K_M` **3.1GB** ＋ `nomic-embed-text` 0.32GB |
| wafl-ctrl5 | used 8131 / free 3779 MiB．judge 用 swallow-8B 5.3GB ＋ 0.6b 2.4GB ＋ nomic 0.32GB が常駐 |

- **重要な実測事実**: `qwen3-embedding:0.6b` の**常駐サイズは重み 639MB ではなく 2.4GB**である（`ollama ps`，context 4096）．差分の約 1.8GB は KV/活性メモリで，Ollama の SIZE はこれを含む．したがって「4b は 0.6b の約 4 倍」という B125 の見立て（重み比）は常駐サイズの比としては過小で，**4b の常駐サイズは重み 2.5GB ＋ より大きい KV（層数・隠れ次元とも増加）で 4.5〜6GB 規模になる可能性が高い**（推測．実測で確定させる）．
- 本走時の wafl500 のピークは `light_model 3.1GB + expert_model 5.3GB + embedding X` である．**12288 MiB から 8.4GB を引くと X の予算は約 3.4GB** しかない．0.6b（2.4GB）はこの予算に収まっているが，**4b は収まらない可能性が高い**．
- 収まらない場合に何が起きるかも実測から分かる．wafl503/509 では `qwen3-embedding:0.6b` が resident に居らず古い `nomic-embed-text` が残っている．**`OLLAMA_KEEP_ALIVE=-1`（docker-compose.yml:23）でも，Ollama は新しいモデルを載せるために既存モデルを退避する**．非依頼者ノードは埋め込みを起動時の `domain_embedding` 計算（`http_server.py:402-404`）にしか使わないため退避されても実害は無いが，**依頼者 wafl500 は 1 問ごとに埋め込みを呼ぶ（`node.py:202`）ため，退避と再ロードが往復すると所要時間が跳ね上がる**．精度ではなく所要時間（`experiment.timeout_min: 150`）のリスクとして扱う．
- wafl-ctrl5 も free 3779 MiB であり，G1 で 4b を載せる前に `ollama stop` で `qwen3-embedding:0.6b`（2.4GB）と `nomic-embed-text` を一時退避して枠を空ける必要がある．**0.6b と nomic の訓練データ埋め込みは `data/embcache_qwen3-embedding_0.6b.npy` / `data/embcache_nomic-embed-text.npy` にキャッシュ済み**（`scripts/screen_embedding_models.py` の仕様）なので，退避しても G1 の基準線は再計算不要である．

**Q3: 2560 次元 × 1,427 行は p ≫ n だが，1024 次元でも既にそうである**

訓練行数 1,427 に対し特徴次元は 1024 → 2560 へ増える．`LogisticRegression(max_iter=1000)`（L2 正則化，既定 C=1.0）は p ≫ n でも解けるが，**訓練データ内 CV と本走の乖離が広がる方向の変化である**．Iter79 は G1 CV 0.7561 に対し本走 0.7530 とほぼ一致したが，次元が 2.5 倍になると「CV は上がったが本走は上がらない」という乖離が起こり得る．**この点は事前登録の解釈規則として明記しておく**（後述）．

### 計画 (Iter80)

**単一レバー**

`embedding_model_replacement` = **`qwen3_embedding_4b`**．`config.yaml:4` の `embedding_model` を `qwen3-embedding:0.6b` → `qwen3-embedding:4b` へ変更し，同じ埋め込みで `models/domain_classifier.joblib` を再訓練する（再訓練は次元変更に構造的に付随する作業であり別レバーではない．Iter79 で確立した型）．**変更する設定キーは `config.yaml:4` の 1 行のみ．**

**固定する構成（単一レバー原則）**

`config.yaml` の `embedding_model` 以外の全項目（`routing_method=supervised_classifier`，`confidence_threshold=0.0`，`dispatch_top_k=2`，`dispatch_gap_threshold=0.29`／`dispatch_gap_max_k=4`，`aggregation_method`，`judge_model`，`classifier_model_path`，各ノードの `light_model=qwen3.5:4b-q4_K_M`／`expert_model=expert-mesh-*-lora`，`probe_timeout_s`／`dispatch_timeout_s`，`experiment.timeout_min=150`），`data/dataset.jsonl`（1,915 行．ビット単位で Iter79 と同一），`data/classifier_train.jsonl`（1,427 行，無変更．**train/eval 重複 72 行は B125(c) のとおり本反復では触らない**），`scripts/train_domain_classifier.py` のハイパラ一式，`classifier.py`／`http_server.py`／`node.py`／`aggregator.py`／`metrics.py`／`build_dataset.py`，`docker-compose.yml`（`OLLAMA_KEEP_ALIVE=-1` を含む）．**instruction prefix は導入しない**（B125(5)．`multilingual-e5-large`／`ruri-v3-310m` は prefix 付与＝2 レバー目になるため本反復の候補外）．

**事前ゲート G0（VRAM 実測．B125(3)．結果を見る前に判定規則を固定する）**

すべて wafl-ctrl5 と read-only の `ollama ps`／`nvidia-smi` で行い，**本走前に wafl500〜509 で生成処理は走らせない**（config.yml 絶対条件 B）．

1. **G0-a（wafl-ctrl5 での常駐サイズ実測）**: `docker exec ollama-ctrl ollama stop qwen3-embedding:0.6b nomic-embed-text` で枠を空け，`ollama pull qwen3-embedding:4b` → 1 文の `/api/embed` を叩いてロードさせ，`ollama ps` の SIZE と PROCESSOR，`nvidia-smi` の used 差分を記録する．**PROCESSOR が `100% GPU` でなければ（CPU 混在なら）その時点で G0 失敗**．
2. **G0-b（wafl500 の予算判定）**: G0-a で得た常駐サイズを X とし，**X + 3.1GB（light_model）+ 5.3GB（expert_model）≤ 11.5GB** を合格条件とする（12288 MiB から表示等の余白 0.8GB を引いた実効上限）．
3. **G0-c（deploy 後の実機確認）**: deploy 後・本走前に，先頭 20 問の予備実行を回したうえで全 10 ノードの `ollama ps` を取り，(i) 3 モデルすべてが `100% GPU`，(ii) 予備 20 問の 1 問あたり平均所要が **Iter79 実測 2301.4ms の 2 倍（4603ms）以下**，を確認する．満たさない場合は所要時間を外挿し，**1,915 問の予測所要が 120 分を超えるなら G0 失敗**とする．
4. **G0 失敗時の代替（事前登録）**: `qwen3-embedding:4b` を断念し，**G1 で VRAM 実行可能と確認できた候補のうち CV accuracy が最大のもの**（現実的には `bge-m3`，重み約 1.2GB）へレバーの値を切り替える．その場合のみ config.yml の `embedding_model_replacement` の `values` へ `bge_m3` を追記し，backlog へ `[auto-decided]` で記録する．**`iteration_name` は経緯の追跡性のため変更しない**（見出しと実際の値がずれる場合は journal の実行記録に明記する）．

**事前ゲート G1（値の選定．評価集合を一切見ない．B125(1)(2)）**

`scripts/screen_embedding_models.py` を拡張し（`_STAGE_1_CANDIDATES` を本反復用に差し替える），`data/classifier_train.jsonl`（1,427 行）**のみ**で 5-fold StratifiedKFold の accuracy / macro-F1 を測る．`data/dataset.jsonl` は参照しない（現行スクリプトが既にその設計であることは docstring と `--train-data` 既定値で担保されている）．

- 候補は 3 つを**同時に**測る: **`qwen3-embedding:0.6b`（現行＝基準線．キャッシュ再利用）**，**`qwen3-embedding:4b`**，**`bge-m3`**（B125(2) の申し送り）．
- **選定規則（結果を見る前に固定）**: G0 を通過した非現行候補のうち **CV accuracy 最大のものをレバーの値とする**．同点なら macro-F1，なお同点なら `qwen3-embedding:4b` を優先する．**現行 0.6b（Iter79 実測 cv_accuracy 0.7561 / macro-F1 0.7562）を下回る候補しか残らない場合でも，最良の非現行候補で本走は実施する**（config.yml 絶対条件 A．Iter79 の規則 4 と同一）．その際は「改善しない見込み」という着地点予測を本走前に journal へ記録する．
- **CV と本走の乖離に関する解釈規則（Q3 への対処）**: 次元が 2.5 倍になるため，G1 の CV は本走 top1 の上振れした推定になり得る．**G1 CV が本走の予測値であるとは扱わず，値の選定にのみ用いる**．

**事前ゲート G2（検出力）**

旧 artifact（0.6b・1024 次元）と新 artifact（選定モデル）の `predict_proba` argmax を 1,915 行で replay し，discordant 行数 n_d と必要偏り率 `1.96/sqrt(n_d)` を算出する．**n_d ≥ 30 を合格条件**とし，一桁なら「効果なし」ではなく **config 未到達**を既定の解釈とする（d0004 §4）．

**変更するファイルと箇所**

1. `config.yaml:4`: `embedding_model: qwen3-embedding:0.6b` → `embedding_model: qwen3-embedding:4b`（G0/G1 の結果が `bge-m3` ならその名前）．**これが本レバーの本体であり，変更はこの 1 行のみ．**
2. `models/domain_classifier.joblib`: 新埋め込みで再訓練して差し替える．**旧版は `models/domain_classifier_pre_iter80_qwen3_0.6b.joblib` へ `cp` で退避**してから上書きする（Iter77/79 と同じ慣行．flip 計測と rejected 時の復元に必要）．
3. `scripts/screen_embedding_models.py`: 候補リストを本反復用（`qwen3-embedding:0.6b` / `qwen3-embedding:4b` / `bge-m3` の 3 候補同時評価）へ更新し，docstring の選定規則を Iter80 のものへ書き換える．**`data/dataset.jsonl` を参照しない性質は維持する．**
4. `data/MANIFEST.md`: 新 artifact の sha256・生成コマンド・埋め込みモデル名・G0/G1/G2 の実測値を追記する（`data/`・`models/` は `.gitignore` 対象のため MANIFEST が唯一の再現性の担保．B123）．
5. `.claude/research/config.yml`: **G0 失敗で `bge-m3` を採った場合のみ** `values` へ `bge_m3` を追記する．

**変更しないが確認だけするファイル**: `tests/test_node.py:170` ほか計 4 つのテストが埋め込みモデル名を文字列リテラルで持つが，いずれもテスト内で組み立てる config 辞書の値であって `config.yaml` を読まない．**テストの修正は不要．**

**レバーを読むコード行と到達条件**

- 設定 → 全ノードへのモデル配布: `tools/node_models.py:get_models()` が `config["embedding_model"]` を返し，`mise.toml` L96-104 の deploy ループが `ollama pull` する．到達確認は**全 10 ノードで `ollama list | grep qwen3-embedding` に `4b` 行があること**．
- 設定 → 各ノードの config: `mise.toml` L67 の `rsync config.yaml`．到達確認は**全 10 ノードで `grep '^embedding_model:' $REMOTE_DIR/config.yaml`**．
- 設定 → 実行時のクエリ埋め込み: `node.py:202`（依頼者）／`http_server.py:402-404`（各ノードの `domain_embedding`）．到達確認は先頭 20 問の予備実行が 500 を返さないこと（**次元不一致なら必ずここで落ちる**）．
- artifact → 全ノード: `mise.toml` L70-74 の `models/` rsync．到達確認は**全 10 ノードで `n_features_in_` が 2560（`bge-m3` なら 1024）**，または sha256 一致．
- 実験 → 指標: `metrics.py` 無変更．到達確認は `question_count == 1915` かつ `compound_domain_question_count == 415`．
- **レバー発火の直接証拠**: G2 の n_d ≥ 30，かつ本走 `selected_domain` の discordant が同程度であること．

**実験手順**

1. **前提確認**: `wc -l data/dataset.jsonl` = 1915，`wc -l data/classifier_train.jsonl` = 1427，`models/domain_classifier.joblib` の sha256 が `data/MANIFEST.md` の記載（`21e16ec6...`）と一致すること．
2. **G0-a（wafl-ctrl5 のみ）**: 上記のとおり 4b と bge-m3 を pull し，常駐サイズ・PROCESSOR・embed レイテンシを実測．**wafl500〜509 は使わない．**
3. **G1（wafl-ctrl5 のみ）**: `ssh -fNT -L 11499:localhost:11434 wafl-ctrl5` のトンネル経由で `uv run python -m scripts.screen_embedding_models --train-data data/classifier_train.jsonl --ollama-host 127.0.0.1 --ollama-port 11499` を実行．3 候補の CV accuracy / macro-F1 を全て journal へ記録し，選定規則に従って 1 モデルへ確定する．
4. **G0-b 判定** → 合格なら 4b，不合格なら代替候補で以降へ進む．
5. **再訓練（wafl-ctrl5 のみ）**: 旧 artifact を退避したうえで `uv run python -m scripts.train_domain_classifier --train-data data/classifier_train.jsonl --embedding-model <選定モデル> --ollama-host 127.0.0.1 --ollama-port 11499 --output models/domain_classifier.joblib`．`n_features_in_` を確認．
6. **G2（wafl-ctrl5 のみ）**: 旧/新 artifact の argmax replay で n_d と `1.96/sqrt(n_d)` を算出し，新 artifact 単体のオフライン accuracy も着地点予測として記録する（判定には使わない）．
7. `config.yaml:4` を書き換え，`uv run ruff check`（既存 23 件は pre-existing）と `uv run pytest tests/`（pre-existing 9 件 FAIL は B122．**新規失敗 0 件**）を確認．
8. `mise run setup`（**直後に `uv sync --extra research` で research extra を復旧する．B118 の落とし穴 2**）→ `wc -l data/dataset.jsonl` = 1915 を再確認．
9. `mise run deploy` → 上記「到達条件」を全 10 ノードで確認．
10. **先頭 20 問の予備実行 → G0-c 判定**．失敗なら代替候補へ切り替えて 5〜9 をやり直す．
11. **wafl500〜509 で 1,915 問のフルスペック本走を 1 回**（絶対条件 A）．起動直後に `state.json` を `status=waiting_experiment`・`experiment_dir`・`experiment_deadline`（開始時刻 + 150×60 + 600 秒）へ更新．`mise run analyze -- <timestamp>` まで実施（**引数なし実行は `results/iter45_preliminary/` を誤選択する．B118 の落とし穴 1**）．
12. 指標を「全 1,915 行」「既存 1,600 行部分集合」「複合 415 行」の 3 通りで算出し，Iter79 基準線 `results/20260926_221822/` と id ペアリングで McNemar 検定・per-domain 20 指標の BH 補正を行う．

**仮説（事前登録）**

「`qwen3-embedding:0.6b` を `qwen3-embedding:4b`（2560 次元）へ替えると，日本語質問文のドメイン分離度がさらに上がり，1,915 行本走の `top1_accuracy` が Iter79 基準線 0.753003 から **+1.5pt 以上**改善する（McNemar p<0.05）．ただし Iter79 のような 2 桁 pt の伸びは期待しない．**0.753 の時点で残る誤りの多くは埋め込み品質ではなくラベル境界の曖昧さ（social_science と history_culture の相互混同など）に由来する可能性があり，効果は 0〜+3pt のどこかに着地する**と見込む．同一モデル族内のスケールアップであるため discordant n_d は Iter79 の 791 より大幅に小さく，**100〜400 行**と予測する．」

**成功条件（事前登録）**

基準線は Iter79 本走 `results/20260926_221822/`（全 1,915 行: top1=0.753003，single_domain_top1=0.749333，compound_domain_top1=0.766265，compound_domain_set_recall=0.548193，kappa=0.721502，misrouting=0.246997，ECE=0.032745，Brier=0.152849，AUROC=0.777614，fallback=0.0，dispatch_failure=0.000522，mean_duration_ms=2301.4，compound_mean_dispatched_count=1.880，answer_quality=0.569333，end_to_end=0.335770／既存 1,600 行部分集合: top1=0.751250）．

| 区分 | 指標 | 現状 | 合格条件 |
|---|---|---|---|
| **G0（VRAM）** | wafl-ctrl5 での常駐サイズ X と PROCESSOR | 0.6b は 2.4GB / 100% GPU | `100% GPU` かつ **X + 8.4GB ≤ 11.5GB**．deploy 後は予備 20 問の平均所要 ≤ 4603ms，または 1,915 問外挿 ≤ 120 分 |
| **G1（値の選定）** | 5-fold CV accuracy（`classifier_train.jsonl` のみ） | 0.6b = 0.7561 | 3 候補の値を全て記録し，選定規則どおり 1 モデルへ確定できること |
| **G2（検出力）** | 旧/新 replay の discordant n_d | 参考: Iter79 は 791 | **n_d ≥ 30**．未満なら config 未到達を疑い本走前に原因を潰す |
| **主基準（効果）** | 全 1,915 行の `top1_accuracy` | 0.753003 | **McNemar p < 0.05 かつ 点推定 +1.5pt 以上**（n_d=200 想定の MDE `1.96·sqrt(n_d)/1915` ≈ 1.45pt を上回る水準） |
| **非退行①** | per-domain recall/precision 計 20 指標 | Iter79 実測 | **BH 補正（q=0.05）後の有意退行 0 件** |
| **非退行②** | `fallback_rate` / `dispatch_failure_rate` | 0.0 / 0.000522 | fallback = 0.0，dispatch_failure ≤ 0.005 |
| **非退行③** | ECE | 0.032745 | **≤ 0.08**（Iter79 の水準から +5pt 以内．再訓練で温度が変わる余地を見込む） |
| **非退行④（新設）** | `mean_duration_ms` | 2301.4 | **≤ 4603（2 倍以内）**．埋め込みモデルの大型化による退避・再ロードの検出用．超過は「実験不成立」ではなく退行として扱う |
| 報告のみ | 既存 1,600 行部分集合の top1 | 0.751250 | 判定には用いないが毎回併記する |
| 報告のみ | `compound_domain_top1` / `single_domain_top1` / `compound_domain_set_recall` / `compound_mean_dispatched_count` | 0.766265 / 0.749333 / 0.548193 / 1.880 | 送出数は `dispatch_gap_threshold=0.29` 固定の二次効果で動きうる（B125 要レビュー (b)） |
| 報告のみ | `answer_quality_accuracy` / `end_to_end_accuracy` | 0.569333 / 0.335770 | 生成器は不変のため横ばいを期待 |

- **adopted**: 主基準（有意改善 かつ +1.5pt 以上）と非退行①②③④をすべて満たす．
- **partial**: 点推定は改善だが p ≥ 0.05 または +1.5pt 未満，かつ非退行に違反なし．この場合は `embedding_model_replacement` の `values` を使い切ったものとして**収束**扱いにし，次レバーは prefix 付与（`embedding_input_instruction_prefix`，B124 要レビュー (a)）または `dispatch_gap_threshold` 再調整（B125 要レビュー (b)）を backlog で比較する．
- **rejected**: 点推定が低下，または非退行①で有意退行が 1 件以上，または非退行④違反．**復元手順**: `config.yaml:4` を `qwen3-embedding:0.6b` へ戻し，`cp models/domain_classifier_pre_iter80_qwen3_0.6b.joblib models/domain_classifier.joblib` で復元し，`mise run deploy` を再実行して全 10 ノードの `n_features_in_`=1024 を確認する．
- **invalid（実験不成立）**: `question_count != 1915`，`compound_domain_question_count != 415`，G2 の n_d が一桁，またはノードの `ollama list` に選定モデルが無い．いずれも setup/deploy の漏れと解釈する（d0004 §4）．

**期待効果とリスク**

期待効果は「同一モデル族のスケールアップで特徴量の分離度をもう一段上げる」ことに尽きる．一方で本反復は Iter79 と異なり，**改善が得られない側に賭けるべき理由（0.753 到達後の残差がラベル境界起因である可能性，JMTEB 4B 値の不在）と，実行自体が危うくなる側のリスク（VRAM 予算 3.4GB に対し 4b の常駐サイズが未知）の双方を抱えている**．前者は G1 で，後者は G0 で本走前に潰す．どちらのゲートでも失敗した場合の分岐（代替候補 `bge-m3` への切り替え，または「改善しない見込み」を記録したうえでの本走実施）を上記のとおり結果を見る前に固定してある．

### Iteration 80 実行済み

**判定: rejected（事前登録の rejected 条件を 2 つとも満たした）**

#### レバーの値が本走前に切り替わった経緯（見出しと実際の値の齟齬）

見出しの `iteration_name`（「0.6b → 4b」）と実際に本走したレバーの値（`bge-m3`）は一致していない．計画節「G0 失敗時の代替（事前登録）」の分岐がそのまま発火したためである（詳細は backlog B126）．

- **G0-a（wafl-ctrl5 実測）**: `qwen3-embedding:4b` の常駐サイズは **4.4GB**（PROCESSOR は `100% GPU`）．
- **G0-b（wafl500 予算判定）**: 合格条件 `X + 8.4GB ≤ 11.5GB` に対し 12.8GB で **1.3GB 超過 → G0 不合格**．
- 事前登録どおり `qwen3-embedding:4b` を断念し，G0 を通過した非現行候補のうち CV 最大の **`bge-m3`**（常駐 **0.664GB**）へ切り替えた．`iteration_name` は追跡性のため据え置いた（計画節 G0-4 の指示どおり）．

**G1（`data/classifier_train.jsonl` 1,427 行のみの 5-fold CV．評価集合を一切見ていない）**

| 候補 | CV accuracy | macro-F1 | 常駐サイズ | G0 |
|---|---|---|---|---|
| `qwen3-embedding:0.6b`（現行＝基準線） | 0.7561 | 0.7562 | 2.4GB | 合格（現行） |
| `qwen3-embedding:4b` | **0.7722** | 0.7718 | 4.4GB | **不合格**（選定対象外） |
| `bge-m3` | 0.7120 | 0.7140 | 0.664GB | 合格 → **選定** |

**CV で最良だった 4b が VRAM だけで排除され，選定規則が残した唯一の候補が基準線を 4.4pt 下回る `bge-m3` だった**．config.yml 絶対条件 A（「改善しない見込みでも最良の非現行候補で本走は実施する」）に従い，着地点が基準線割れになる見込みを本走前に記録したうえで本走した．

**G2（検出力）**: 旧/新 artifact の argmax replay（1,915 行）で discordant **n_d=469 ≥ 30 で合格**．新 artifact 単体のオフライン argmax accuracy は **0.7248**（旧 0.7535）．

#### 変更したもの

1. `config.yaml:4` `embedding_model`: `qwen3-embedding:0.6b` → `bge-m3`（1 行．本レバーの本体）
2. `models/domain_classifier.joblib`: bge-m3 で再訓練（sha256 `37d71b63...`，`n_features_in_`=1024）．旧版は `models/domain_classifier_pre_iter80_qwen3_0.6b.joblib`（sha256 `21e16ec6...`）へ退避
3. `scripts/screen_embedding_models.py`: 3 候補同時評価へ拡張
4. `data/MANIFEST.md`: 新 artifact の sha256・G0/G1/G2 実測値を追記
5. `.claude/research/config.yml`: `embedding_model_replacement` の `values` へ `bge_m3` を追記

到達確認は全て通過（全 10 ノードで `bge-m3:latest` の存在・`embedding_model: bge-m3`・artifact sha256 一致，先頭 20 問の予備実行で 500 エラー 0・平均 2288.9ms ≤ 4603ms）．lint 新規 0 件，test は pre-existing 9 件 FAIL（B122）のみで 306 件 PASS．**実験は成立している**（`question_count=1915`，`compound_domain_question_count=415`）．

#### 結果（本走 `results/20260927_004001/`，1,915 問 1 回．基準線は Iter79 `results/20260926_221822/`）

| 指標 | Iter79 基準線 | Iter80（bge-m3） | 差 | 判定 |
|---|---|---|---|---|
| **top1_accuracy（全 1,915）** | 0.753003 | **0.722715**（95%CI [0.7022, 0.7423]） | **-3.03pt** | **有意な退行** |
| single_domain_top1（1,500） | 0.749333 | 0.744000 | -0.53pt | ノイズ範囲 |
| compound_domain_top1（415） | 0.766265 | 0.645783 | **-12.05pt** | **有意な退行** |
| 既存 1,600 行部分集合 top1 | 0.751250 | 0.735625 | -1.56pt | 有意でない |
| compound_domain_set_recall | 0.548193 | 0.512048 | -3.61pt | 退行 |
| compound_mean_dispatched_count | 1.880 | 2.104 | +0.224 | 増加 |
| cohens_kappa | 0.721502 | 0.715619 | -0.0059 | — |
| misrouting_rate | 0.246997 | 0.277285 | +3.03pt | 退行 |
| fallback_rate | 0.0 | 0.0 | 0 | 非退行②合格 |
| dispatch_failure_rate | 0.000522 | 0.001567 | +0.001 | 非退行②合格（≤0.005） |
| ECE | 0.032745 | 0.033660 | +0.0009 | 非退行③合格（≤0.08） |
| mean_duration_ms | 2301.4 | 2241.04 | -60 | 非退行④合格（≤4603） |
| Brier | 0.152849 | 0.164695 | +0.0118 | 報告のみ |
| AUROC | 0.777614 | 0.767964 | -0.0097 | 報告のみ |
| answer_quality_accuracy | 0.569333 | 0.574667 | +0.53pt | 報告のみ（生成器不変） |
| end_to_end_accuracy | 0.335770 | 0.335248 | -0.05pt | 報告のみ（横ばい） |

**McNemar 検定（id ペアリング，1,915 行完全対応．`scipy.stats.binomtest` の正確二項検定）**

| 区分 | n | 旧のみ正解 b | 新のみ正解 c | n_d | p 値 | 解釈 |
|---|---|---|---|---|---|---|
| 全 1,915 行 | 1915 | 191 | 133 | 324 | **1.50e-03** | **有意な退行** |
| 単一 1,500 行 | 1500 | 127 | 119 | 246 | 0.656 | **差なし（ノイズ範囲）** |
| **複合 415 行** | 415 | 64 | 14 | 78 | **8.58e-09** | **強く有意な退行** |
| 既存 1,600 行部分集合 | 1600 | 150 | 125 | 275 | 0.148 | 有意でない |

**per-domain 20 指標の BH 補正（q=0.05）: 有意退行 2 件（非退行①違反）**

| 指標 | 旧 | 新 | 差 | p |
|---|---|---|---|---|
| **recall: legal** | 0.7621 | 0.5534 | **-20.87pt** | 1.82e-09 |
| **recall: computer_science** | 0.8155 | 0.6915 | **-12.40pt** | 3.88e-04 |

BH 補正で有意にならなかったものの退行側に振れた指標として precision: general（-13.39pt，p=9.4e-03），precision: history_culture（-9.00pt，p=2.6e-02）があり，**20 指標のうち 12 指標が退行側**（改善側 8）に偏っている．

#### 判定と根拠

事前登録の **rejected 条件は「点推定が低下，または非退行①で有意退行が 1 件以上，または非退行④違反」**であり，**前 2 者を独立に満たしている**（-3.03pt の低下，BH 後の有意退行 2 件）．非退行②③④は全て合格だが，rejected 条件は OR であるため判定は **rejected** で確定する．adopted/partial の余地は無い．

#### 分析（ノイズか有意か，および原因）

1. **これはノイズではない．3 つの独立な測定が一致している**．G1 CV -4.41pt（0.7561→0.7120），G2 オフライン argmax -2.87pt（0.7535→0.7248），本走 top1 -3.03pt（0.7530→0.7227）．**オフライン 2 指標が本走の符号と桁をともに正しく予測した**．Iter77〜79 の本走間ばらつき（同一 config の再現差は 1pt 未満）と比べても -3.03pt は明確に外側であり，McNemar p=1.50e-03 がこれを裏づける．
2. **次元数は原因ではない**．`bge-m3` も `qwen3-embedding:0.6b` も **ともに 1024 次元**であり，訓練行数 1,427・LogisticRegression のハイパラ・訓練スクリプトは完全に同一である．計画時に Q3 として構えた「2560 次元による CV と本走の乖離」は，4b が排除された時点で本反復には該当しなくなった．**残る差は埋め込みベクトルの表現そのものだけ**であり，因果の切り分けとしては珍しく綺麗である．
3. **劣化は単一ドメインではほぼ起きず，複合設問に集中している**（単一 -0.53pt / p=0.656 に対し複合 -12.05pt / p=8.58e-09）．しかも複合では `mean_dispatched_count` が 1.880 → 2.104 と**増えているのに** `set_recall` は 0.548 → 0.512 へ**下がった**．`dispatch_gap_threshold=0.29` は固定なので，送出数が増えたのは **bge-m3 の事後確率分布が平坦化し，rank1 と rank2 以降の gap が縮んだ**ことを意味する．つまり **bge-m3 はクラス間マージンが狭く，より多くの候補へ広く送ってなお当てられていない**．AUROC -0.0097・Brier +0.0118 もこの「順位付けの質そのものの低下」と整合する（ECE がほぼ不変なのは，確率の較正は保たれたまま順位の情報量だけが落ちたことを示す）．
4. **モデル特性としての解釈（推測を含む）**: `bge-m3` は multi-vector／検索（retrieval）用途を主眼に対照学習されたモデルで，公開ベンチでも Retrieval 系が高く Classification 系は相対的に低い．本タスクは「クエリ 1 本を 10 クラスへ線形分離する」分類タスクであり，検索向けの埋め込み空間が必ずしも線形分離に有利ではない．**MTEB/JMTEB の総合スコアやモデルサイズは，本タスクの線形分離性を予測しない**（Iter79 の学び 2 の再確認）．
5. **legal（-20.87pt）と computer_science（-12.40pt）の崩れ方**: この 2 ドメインは語彙が特徴的（法令名・条文・API 名・言語名）で，Iter79 の 0.6b 移行で最も伸びた側でもある．検索向けモデルは語の表層一致に引きずられやすく，「法律用語を含むが実質は business_economics の問い」のような行で誤って legal を引き寄せる／逆に一般語で書かれた法律問題を取りこぼす，という双方向の崩れが起きたと見られる（precision: legal は +4.48pt と上がっており，**legal へ入る数が減って残った分は当たっている＝再現率だけが落ちた**形になっている．これは「引き寄せ」ではなく「取りこぼし」が主因であることを示す）．
6. **過剰一般化しない範囲**: 本反復が示したのは「`bge-m3` は本構成では `qwen3-embedding:0.6b` に劣る」ことだけである．`qwen3-embedding:4b` が優れているか否かは**実機で検証していない**（G1 CV 0.7722 は訓練データ内 CV であって本走の予測ではないと計画時に明記してある）．

#### 復元（事前登録した手順を実行済み）

rejected 判定に伴い，計画節の復元手順をそのまま実行した．

1. `config.yaml:4` を `qwen3-embedding:0.6b` へ戻した（`git diff config.yaml` が空＝Iter79 commit の状態に一致）．
2. `cp models/domain_classifier_pre_iter80_qwen3_0.6b.joblib models/domain_classifier.joblib` で復元．sha256 が `21e16ec6...`，`n_features_in_`=1024 であることを確認．
3. `mise run deploy` を再実行．smoke_check（hashes / probe）は全て pass．
4. 全 10 ノードで `embedding_model: qwen3-embedding:0.6b` と artifact sha256 `21e16ec6...` を確認し，コンテナ内 `ollama list` に `qwen3-embedding:0.6b` が存在することを確認した．

**`bge-m3` のイメージは各ノードに残しているが削除していない**（破壊的操作は行わない方針．常駐しない限り VRAM を占有しないため実害は無く，再評価時の再 pull を省ける）．

#### 学び

1. **VRAM 予算はレバーの値空間を実質的に決める**．本反復で最も精度が高かった候補（4b，CV 0.7722）は，精度ではなく **1.3GB の VRAM 超過**だけで排除され，結果として基準線を下回ることが事前に分かっている候補で本走する羽目になった．`ollama ps` の SIZE は重みファイルサイズではなく **KV/活性を含む常駐サイズ**であり，0.6b では 639MB の重みに対し常駐 2.4GB（約 3.8 倍）だった．**「重みサイズ比で VRAM を見積もる」のは誤りで，必ず `ollama ps` で実測すること**．一方 bge-m3 は重み 1.2GB に対し常駐 0.664GB と**重みより小さく**，比率は一定ですらない．
2. **事前登録した分岐は，都合の悪い結果を招くときこそ効いている**．G0 失敗時の代替候補を「結果を見る前に」固定してあったため，「4b が駄目なら基準線を下回ると分かっている bge-m3 を回すのは無駄では」という後付けの判断が入り込む余地が無かった．結果として「検索向けモデルは本分類タスクに向かない」という否定的だが再利用可能な知見が n=1915 の実測で得られた．**事前登録は adopted を守るためではなく rejected を捨てないための装置である**．
3. **G1 オフライン CV は本走 top1 の「順位」を正しく予測した**（bge-m3 < 0.6b）．Iter79 でも CV 0.7561 に対し本走 0.7530 とほぼ一致しており，2 反復連続で整合した．**`embedding_model_replacement` 系のレバーについては，G1 CV で基準線を明確に下回る候補は本走でも下回る**と扱ってよい見込みが立った（ただし計画節で明記したとおり，次元数が大きく変わる候補では乖離が広がる可能性が残るため，CV は選定にのみ用い予測値としては扱わない方針は維持する）．
4. **埋め込みの質の低下は「単一 → 複合」の順に増幅して現れる**．単一 -0.53pt（有意でない）に対し複合 -12.05pt（p=8.6e-09）．複合設問は rank2 以降の候補順位に依存するため，**確率分布のマージンが縮むと真っ先に壊れる**．逆に言えば，**複合 415 行は埋め込み品質の変化に対する感度が単一行より 20 倍以上高い検出器として機能している**（Iter78 の評価集合拡充の投資が，改善だけでなく退行の検出でも回収された）．
5. **`embedding_model_replacement` レバーは使い切った**（`qwen3_embedding_0.6b`=adopted / `qwen3_embedding_4b`=G0 不合格で実機未検証 / `bge_m3`=rejected）．次は**モデルの交換ではなく入力の整形**へ移る（下記 B127）．

---

## Iteration 79: 埋め込みモデルの差し替え（nomic-embed-text → qwen3-embedding:0.6b）

### 調査 (Iter79)

B115(3) の優先順位（複合評価集合拡充 > 埋め込みモデル差し替え > 全ドメイン共通の訓練データ拡充）に従い，Iter78 で最上位項目が完了（partial だが評価集合は常設採用）したため本反復は `embedding_model_replacement` に着手する．B116(3) により本レバーはユーザー事前承認済みで，着手前の追加確認は不要である．調査の問いは 3 つ．**(Q1) `qwen3-embedding:0.6b` は本タスク（日本語質問文の 10 ドメイン分類）で現行 `nomic-embed-text` を上回る根拠があるか．(Q2) Ollama 経由のドロップイン差し替えで，モデル固有の前処理（instruction prefix 等）を欠くことによる性能劣化は起きないか．(Q3) 実装上，差し替えが本走経路まで届く到達条件と，静かに壊れる箇所はどこか．**

**Q1: 多言語 MTEB では上回るが，日本語ベンチマークでは自明ではない**

- `Qwen3-Embedding-0.6B` の公称値は MTEB multilingual Mean(Task) **64.33**（Accenture AI Refinery SDK のモデルカード <https://sdk.airefinery.accenture.com/>，および Qwen3 Embedding 論文 Table <https://arxiv.org/abs/2506.05176> の 0.6B 行．論文表記は評価版差で 70.70 とする箇所もあり，公称値は出典で揺れる）．config.yml の lever note が採った 64.33 と現行 `nomic-embed-text` の 62.28 の差 **+2.05pt** は，この多言語平均での比較である．
- 一方，**日本語ベンチマーク JMTEB での実測は振るわない**．hotchpotch「Qwen3 Embedding 文章ベクトルの日本語性能を JMTEB で測る」（2025-06-11，<https://secon.dev/entry/2025/06/11/100000-qwen3-embedding-jmteb>）によれば，Qwen3-Embedding-0.6B の **Classification 平均は 66.09**．同記事の同一条件の比較対象は `intfloat/multilingual-e5-large` 72.89，`cl-nagoya/ruri-v3-310m` 78.66 で，**日本語特化・多言語専用モデルに明確に劣る**．著者自身が「Qwen3-Embedding-0.6B の性能が低すぎる気もする」と留保を付けている点は割り引く必要があるが，**本研究のタスクは「日本語質問文の埋め込みを特徴量とする多クラス分類」であり，JMTEB Classification が最も近い代理指標である**ことは動かない．
- ただし**現行 `nomic-embed-text` は英語中心のモデル**（Nomic AI，v1/v1.5．多言語版は別系統の `nomic-embed-text-v2-moe`）であり，JMTEB 系のリストにそもそも載っていない．したがって「Qwen3 は日本語が弱い」という所見は，**「現行より弱い」を意味しない**．現行がベースラインとして低い可能性が高く，差し替えによる改善余地自体は残っている．
- **事実と推測の区別**: 「Qwen3-0.6B < multilingual-e5-large（日本語 Classification）」は実測（出典上記）．「Qwen3-0.6B > nomic-embed-text（日本語 Classification）」は**未検証の推測**である．本反復ではこれを推測のまま本走に賭けず，後述のオフライン事前スクリーニングで先に数値化する．

**Q2: instruction prefix と，Ollama 実装の実態（実機で確認済み）**

- Qwen3-Embedding は decoder 型で最終層 `[EOS]` トークンの hidden state を埋め込みとし，**Query 側には `Instruct: {task}\nQuery: {query}` 形式の指示文を付けて学習されている**（Doc 側は素のまま．上記 secon.dev の論文メモ，および Qwen3-Embedding モデルカード <https://huggingface.co/Qwen/Qwen3-Embedding-0.6B>）．prefix を付けないと検索系タスクで性能が落ちるという報告がある（r/LocalLLaMA の議論など．定量値は一次情報として確認できず）．
- **実機確認（wafl-ctrl5，2026-09-26）**: `ollama pull qwen3-embedding:0.6b` は ollama 0.34.4 で成功．`ollama show` は architecture=qwen3，parameters=**595.78M**，context length=**32768**，embedding length=**1024**，quantization=**Q8_0**．`ollama show --template` が返すのは Qwen3 の **chat template** であり，`/api/embed` はこれを適用しない．実際に `/api/embed` へ日本語の医療相談文 1 件を投げて **1024 次元**のベクトルが返ることを確認した．つまり **Ollama 経由のドロップインでは instruction prefix は一切付かない**．
- **本反復では prefix を付けない**．prefix を導入すると `node.py` のクエリ埋め込みと `scripts/train_domain_classifier.py` の訓練側埋め込みの両方にテキスト整形を足すことになり，「埋め込みモデルの差し替え」と「入力整形の追加」の 2 レバーになる．単一レバー原則に反するため，prefix 版は**次反復以降の独立レバー候補**として backlog へ回す．
- 副作用として，`nomic-embed-text`（274MB）→ `qwen3-embedding:0.6b`（Q8_0，約 639MB）で各ノードの常駐 VRAM が約 +365MB 増える．10 ノードとも `light_model`(qwen3.5:4b-q4_K_M，約 2.4GB) + `expert_model` + embedding を常駐させる構成（`OLLAMA_KEEP_ALIVE=-1`）だが，この増分は許容範囲と見込む（要実測）．

**Q3: 到達条件と「静かに壊れる箇所」**

- 本走経路で埋め込みモデル名を読むのは **`config.yaml:4 embedding_model` の 1 箇所のみ**．そこから (a) `node.py:202` `await ollama_client.embed(config["embedding_model"], query)`（依頼者ノードでのクエリ埋め込み）と (b) `http_server.py:402-404`（各ノード起動時の `domain_embedding` 計算）へ届く．`grep` で確認した限り，次元を決め打ちしている箇所は `scripts/run_central_experiment.py`（768 次元のゼロベクトル fallback）と `scripts/fine_tune_embedding_*.py` のみで，**いずれも本走経路の外**（前者は Iter26 の中央集権ベースライン比較専用）．したがって本走は次元非依存で動く．
- 全 10 ノードへのモデル配布は自動である．`tools/node_models.py:get_models()` が `[light_model, expert_model, config["embedding_model"]]` を返し，`mise.toml` の deploy ループ（L96-104）が各ノードで `ollama pull` する．`models/` は L70-74 で `sudo rm -rf` の後 rsync される．
- **静かに壊れない（＝派手に壊れる）箇所**: 分類器 artifact を再訓練せずに埋め込みだけ替えると，1024 次元のベクトルを 768 次元前提の `LogisticRegression` に渡すことになり sklearn が `X has 1024 features, but ... is expecting 768 features` を投げる．`classifier.py:estimate_confidence_classifier` は例外を握りつぶさないので全ノードの `/probe` が 500 になる．**検知は容易だが，デプロイ順序を誤ると本走が丸ごと無駄になる**ため，config.yaml と `models/domain_classifier.joblib` は必ず同一 deploy で配る．
- **本当に静かに壊れる箇所は評価集合の側にある**．`mise run setup` は `build_dataset.py --generated-compound-questions data/compound_questions_generated.jsonl`（既定値）を呼ぶが，このファイルは `.gitignore: data/*` によりローカルにしか存在しない（backlog B123）．**消失すると dataset.jsonl が黙って 1,600 行へ戻り，Iter78 基準線との比較が成立しなくなる．** 本日時点で存在を確認済み（315 行，sha256 `1bfb5add6a5f4028c3ceb52531987aaea595c409bfaca1c98bbf48c80901d3c6`，`data/dataset.jsonl` は 1,915 行）．実験前に再確認する．

**Iter78 の学びの反映**

Iter78 の学び 2「π̂_d はレバー依存の量であり，評価集合の属性ではない．**次レバーの artifact ペアで π̂_d を本走前にオフライン実測し，n_d < 30 ならそこで打ち手を考え直す**」をそのまま実行する．埋め込み空間ごと入れ替える本レバーは Iter77 の education 固有補正撤去（10 ドメイン中 1 つの決定境界しか動かない）とは比較にならない広さのはずで，n_d は数百規模になると見込む（未検証．オフラインで実測する）．

### 計画 (Iter79)

**単一レバー**

`embedding_model_replacement` = **`qwen3_embedding_0.6b`**．`config.yaml:4` の `embedding_model` を `nomic-embed-text` → `qwen3-embedding:0.6b` へ変更し，同じ埋め込みで `models/domain_classifier.joblib` を再訓練する（再訓練は差し替えに構造的に付随する作業であり，別レバーではない）．

**事前登録した値の選定規則（G1 ゲート．Q1 の推測を本走前に潰すため）**

Q1 のとおり「Qwen3-0.6B > nomic（日本語）」は未検証である．そこで**本走の前に，評価集合を一切見ないオフライン事前スクリーニングで値を確定する**（config.yml の「事前シミュレーションは本走 1 点を絞り込むための事前登録手段」に該当）．

1. `data/classifier_train.jsonl`（1,427 行，**訓練データのみ．`data/dataset.jsonl` は使わない**＝評価集合へのリークなし）に対し，候補ごとに埋め込みを計算し，同一の `LogisticRegression(max_iter=1000)` + `sample_weight`（`_extract_sample_weights` と同一）で **5-fold StratifiedKFold の accuracy / macro-F1** を測る．
2. 候補は 2 段階．第 1 段: `nomic-embed-text`（現行基準）と `qwen3-embedding:0.6b`．**qwen3 の CV accuracy が nomic を上回れば，そこで `qwen3-embedding:0.6b` に確定する**（以降の候補は評価しない）．
3. 第 1 段で qwen3 が nomic を**上回らなかった場合に限り**，第 2 段として `bge-m3`（Ollama 公式ライブラリ，1024 次元，567M，多言語，prefix 不要のドロップイン）を追加評価し，**CV accuracy 最大の候補を採る**（同点なら macro-F1，なお同点なら qwen3）．`multilingual-e5-large` は JMTEB Classification 72.89 と有力だが `query: ` prefix が必須で入力整形の追加＝2 レバー目になるため**本反復では候補に含めない**（backlog 送り）．
4. 3 候補すべてが nomic を下回った場合でも，**最良の非 nomic 候補で本走は実施する**（config.yml 絶対条件 A．「施策を適用したら常に本走」）．その際は着地点予測として「改善しない見込み」を事前に journal へ記録してから走らせる．

この規則は結果を見る前に固定してあり，選定に使うのは訓練データの CV のみである．本走の判定指標（評価集合 1,915 行の top1）は選定に一切使わない．

**仮説（事前登録）**

「英語中心の `nomic-embed-text` を多言語対応の `qwen3-embedding:0.6b` へ差し替えると，日本語質問文のドメイン分離度が上がり，1,915 行本走の `top1_accuracy` が Iter78 基準線 0.589556 から **+1.8pt 以上**改善する（McNemar p<0.05）．同時に，埋め込み空間が総入れ替えになるため argmax の discordant 行数 n_d は Iter77 ペアの 19 行から桁違いに増え，200 行以上になる．」

**固定する構成（単一レバー原則）**

`config.yaml` の `embedding_model` 以外の全項目（`routing_method=supervised_classifier`，`confidence_threshold=0.0`，`dispatch_top_k=2`，`dispatch_gap_threshold=0.29`／`dispatch_gap_max_k=4`，`aggregation_method=max_confidence`，`judge_model`，`classifier_model_path`，各ノードの `light_model`/`expert_model`，`probe_timeout_s`/`dispatch_timeout_s`），`data/dataset.jsonl`（1,915 行．Iter78 の成果物をビット単位でそのまま使う），`data/classifier_train.jsonl`（1,427 行，無変更），`scripts/train_domain_classifier.py` のハイパラ（`_CALIBRATION_METHOD="temperature"`，`_CALIBRATION_CV=5`，`_MAX_ITER=1000`，`ensemble=True`，`class_weight=None` + domain-balanced `sample_weight`），`classifier.py` / `http_server.py` / `node.py` / `aggregator.py` / `metrics.py` / `build_dataset.py`，`experiment.timeout_min=150`．**instruction prefix は導入しない．**

**変更するファイルと箇所**

1. `config.yaml:4`: `embedding_model: nomic-embed-text` → `embedding_model: qwen3-embedding:0.6b`（G1 の結果が別候補ならその名前）．**これが本レバーの本体であり，変更はこの 1 行のみ．**
2. `models/domain_classifier.joblib`: 新埋め込みで再訓練して差し替える．**旧 artifact は `models/domain_classifier_pre_iter79_nomic.joblib` へ退避**してから上書きする（Iter77 の `domain_classifier_pre_iter77_edu_corrected.joblib` と同じ慣行．flip 計測に必要）．
3. `scripts/screen_embedding_models.py`（**新規**）: G1 の事前スクリーニング専用．`data/classifier_train.jsonl` を候補モデルごとに埋め込み（`data/embcache_<model>.npy` へキャッシュし，採用モデル分は再訓練でも再利用する），5-fold StratifiedKFold の accuracy/macro-F1 を JSON で標準出力へ出す．**Ollama ホストは wafl-ctrl5 固定**（引数の既定値を `localhost:11499` とし，docstring に SSH トンネル手順を書く）．
4. `data/MANIFEST.md`: 新 artifact の sha256・生成コマンド・使用した埋め込みモデル名を追記する（`data/`・`models/` は `.gitignore` 対象のため MANIFEST が唯一の再現性の担保．B123 で確認した既存規約）．
5. `.claude/research/config.yml`: G1 が第 2 段へ進んで `bge-m3` を採った場合のみ，`embedding_model_replacement` の `values` へ `bge_m3` を追記する（採らなければ変更しない）．

**変更しないが確認だけするファイル**: `tests/test_node.py:170` ほか計 4 つのテストが `"nomic-embed-text"` を文字列リテラルで持つが，いずれもテスト内で組み立てる config 辞書の値であって `config.yaml` を読まない．**テストの修正は不要**（修正するとレバーの外側を触ることになる）．

**レバーを読むコード行と到達条件（d0004 §4 の再発防止．同型の失敗 6 回の教訓）**

- 設定 → 全ノードのモデル配布: `tools/node_models.py:get_models()` が `config["embedding_model"]` を返し，`mise.toml` L96-104 の deploy ループが各ノードで `ollama pull` する．到達確認は **全 10 ノードで `docker compose exec -T ollama ollama list | grep qwen3-embedding` が 1 行返ること**．
- 設定 → 各ノードの config: `mise.toml` L67 の `rsync ... config.yaml` が配布する．到達確認は **全 10 ノードで `grep '^embedding_model:' $REMOTE_DIR/config.yaml`**．
- 設定 → 実行時のクエリ埋め込み: `node.py:202`．到達確認は本走ログでノード起動時の `domain_embedding` 計算（`http_server.py:402-404`）が例外なく通ること，および先頭 20 問の予備実行が 500 を返さないこと（**次元不一致なら必ずここで落ちる**）．
- artifact → 全ノード: `mise.toml` L70-74 の `models/` rsync．到達確認は **全 10 ノードで `docker compose exec app python -c "import joblib;m=joblib.load('/app/models/domain_classifier.joblib');print(m.estimators_[0].estimator.n_features_in_)"` 相当が 1024 を返すこと**（実装が困難なら sha256 一致確認で代替してよい）．
- 実験 → 指標: `metrics.py` は無変更．到達確認は `question_count == 1915` かつ `compound_domain_question_count == 415`．
- **レバー発火の直接証拠**: 旧/新 artifact の argmax replay による flip 行数 n_d が **30 行以上**（後述 G2）．これが一桁なら「効果が無かった」ではなく **config が届いていない**ことを既定の解釈とする．

**実験手順**

1. **前提確認**: `wc -l data/dataset.jsonl` = 1915，`sha256sum data/compound_questions_generated.jsonl` = `1bfb5a...01d3c6` を確認（B123 のリスク．消えていたら本反復は中止して backlog へ差し戻す）．
2. **G1 事前スクリーニング（wafl-ctrl5 のみ．絶対条件 B）**: `ssh -fNT -L 11499:localhost:11434 wafl-ctrl5` のトンネル経由で `scripts/screen_embedding_models.py` を実行．`qwen3-embedding:0.6b` は wafl-ctrl5 に pull 済み（調査 Q2 で実施）．第 2 段へ進む場合のみ `bge-m3` を追加 pull する．**wafl500〜509 は使わない．** 結果の CV accuracy / macro-F1 を journal に全候補分記録し，選定規則に従って 1 モデルを確定する．
3. **分類器の再訓練（wafl-ctrl5 のみ）**: 旧 artifact を `models/domain_classifier_pre_iter79_nomic.joblib` へ `cp` で退避してから，
   `uv run python -m scripts.train_domain_classifier --train-data data/classifier_train.jsonl --embedding-model <選定モデル> --ollama-host 127.0.0.1 --ollama-port 11499 --output models/domain_classifier.joblib`
   を実行する（`--ollama-host` を wafl500 等へ向けないこと．絶対条件 B）．`n_features_in_` が 1024 であることを確認する．
4. **G2 検出力の事前実測（wafl-ctrl5 のみ）**: `scripts/evaluate_classifier_calibration.py` を conformal オプション無し（素の `predict_proba` argmax）で 2 回走らせ，(旧 artifact × `nomic-embed-text`) と (新 artifact × 選定モデル) の `selected_domain` を 1,915 行全体で突き合わせて discordant 行数 n_d と必要偏り率 `1.96/sqrt(n_d)` を算出する．**Iter78 の学び 1 に従い，pt スケールの MDE ではなく n_d を主たる検出力指標として記録する．** 同時に新 artifact 単体のオフライン accuracy も記録する（着地点予測．本走の判定には使わない）．
5. `config.yaml:4` を書き換え，`uv run ruff check` と `uv run pytest tests/`（pre-existing 9 件 FAIL は B122 の既知事項．新規失敗 0 件であることを確認）を通す．
6. `mise run setup`（**直後に `uv sync --extra research` で research extra を復旧する．B118 の落とし穴 2**）→ `wc -l data/dataset.jsonl` = **1915** を再確認．
7. `mise run deploy` → 上記「到達条件」4 点を全 10 ノードで確認．
8. 先頭 20 問の予備実行でノード疎通を確認（**次元不一致はここで必ず出る**）．
9. **wafl500〜509 で 1,915 問のフルスペック本走を 1 回（絶対条件 A）**．起動直後に `state.json` を `status=waiting_experiment`・`experiment_dir`・`experiment_deadline`（開始時刻 + 150×60 + 600 秒）へ更新する．`mise run analyze -- <timestamp>` まで実施（**引数なし実行は `results/iter45_preliminary/` を誤選択する．B118 の落とし穴 1**）．
10. 指標を「全 1,915 行」「既存 1,600 行部分集合」「複合 415 行」の 3 通りで算出し，Iter78 基準線 `results/20260926_195929/` と id ペアリングで McNemar 検定・per-domain 20 指標の BH 補正を行う．

**実験時間の見積りと `timeout_min`**

Iter78 本走は 1,915 問で実測 51 分（`18 分 + n_compound/9.9 分` の経験式に概ね一致）．埋め込みモデルの差し替えは 1 問あたり埋め込み 1 回分のコストしか変えず，`mean_dispatch_gen_time_ms`=1997 が所要時間の大半を占めるため，所要時間はほぼ横ばい（±10 分）と見込む．ただし**ルーティング先の分布が変われば `dispatch_gap` 経由の平均 dispatch 数（Iter78 実測 `compound_mean_dispatched_count`=2.506）が動き，所要時間が変わる可能性がある**（増える向きにも減る向きにもあり得る）．`experiment.timeout_min: 150` は実測の約 3 倍の余裕があるため**変更しない**．130 分を超えた場合のみ次反復で 180 への引き上げを起票する．

**成功条件（事前登録）**

基準線は Iter78 本走 `results/20260926_195929/`（全 1,915 行: top1=0.589556，single_domain_top1=0.628667，compound_domain_top1=0.448193，compound_domain_set_recall=0.393976，kappa=0.587438，misrouting=0.410444，ECE=0.050552，Brier=0.206919，AUROC=0.731903，fallback=0.0，dispatch_failure=0.000522／既存 1,600 行部分集合: top1=0.615000）．

| 区分 | 指標 | 現状 | 合格条件 |
|---|---|---|---|
| **事前ゲート G1（値の選定）** | 5-fold CV accuracy（`classifier_train.jsonl` のみ） | nomic の実測値を基準に取る | 選定規則どおり 1 モデルへ確定できること．全候補の値を journal に記録 |
| **事前ゲート G2（検出力）** | 旧/新 artifact replay の discordant 行数 n_d（1,915 行） | 参考: Iter77 ペアで 19 | **n_d ≥ 30**．必要偏り率 `1.96/sqrt(n_d)` も併記．**n_d < 30 なら config 未到達を疑い，本走前に原因を潰す** |
| **主基準（効果）** | 全 1,915 行の `top1_accuracy` | 0.589556 | **McNemar p < 0.05 の有意改善 かつ 点推定 +1.8pt 以上**（n_d=200 想定の MDE `1.96·sqrt(n_d)/1915` ≈ 1.45pt を上回る水準として設定） |
| **非退行①** | per-domain recall/precision 計 20 指標 | Iter78 実測 | **BH 補正（q=0.05）後の有意退行 0 件** |
| **非退行②** | `fallback_rate` / `dispatch_failure_rate` | 0.0 / 0.000522 | fallback = 0.0，dispatch_failure ≤ 0.005 |
| **非退行③** | ECE | 0.050552 | **≤ 0.10**（+5pt 以内．埋め込み空間が変われば温度スケーリングの最適値も変わるため，ここは悪化余地を見込んで緩く置く） |
| 報告のみ | 既存 1,600 行部分集合の top1 | 0.615000 | 判定に用いないが毎回併記する（B119 要レビュー (c) への回答） |
| 報告のみ | `compound_domain_top1` / `single_domain_top1` / `compound_domain_set_recall` | 0.448193 / 0.628667 / 0.393976 | 複合設問への効果を初めて 415 行で測る参考値．Iter78 の学び 2 のとおり，ここが動くかどうかが評価集合拡充の投資回収の指標になる |
| 報告のみ | `answer_quality_accuracy` / `end_to_end_accuracy` | 0.564 / 0.287728 | 埋め込み差し替えは生成器を変えないため横ばいを期待 |

- **adopted**: 主基準（有意改善 かつ +1.8pt 以上）と非退行①②③をすべて満たす．
- **partial**: 点推定は改善だが p ≥ 0.05 または +1.8pt 未満，かつ非退行に違反なし．この場合，`qwen3-embedding:4b`（config の第 2 値）または prefix 付与版を次反復の候補として backlog へ残す．
- **rejected**: 点推定が低下，または非退行①で有意退行が 1 件以上．rejected の場合は `config.yaml` を `nomic-embed-text` へ戻し，`models/domain_classifier.joblib` を退避した旧 artifact から復元する（復元手順を journal に明記してから本走に入ること）．
- **invalid（実験不成立）**: `question_count != 1915`，`compound_domain_question_count != 415`，G2 の n_d が一桁，またはノードの `ollama list` に選定モデルが無い．いずれも「効果なし」ではなく setup/deploy の漏れと解釈する（d0004 §4）．

**期待効果**

現行の `nomic-embed-text` は英語中心のモデルであり，日本語 1,915 問のルーティングを英語埋め込み空間の上で行っている．ここを多言語モデルへ替えることは，これまでの 78 反復で触れてこなかった**特徴量そのものの品質**への初めての介入である（Iter39〜43 の fine-tuning はベースモデルを保ったまま補正する試みで，argmax flip rate が過大となり単一レバー原則と両立せず rejected だった．本レバーは差し替えなので同じ制約を受けない）．同時に，Iter78 で拡充した複合 415 行に対して初めて意味のある検出力で効果を測る機会でもある．

### Iteration 79 実行済み

**判定: adopted（事前登録した全条件を満たす）**

#### 変更したもの

`config.yaml:4` の `embedding_model` を `nomic-embed-text` → `qwen3-embedding:0.6b` の 1 行のみ．これに構造的に付随する作業として `models/domain_classifier.joblib` を新埋め込み（1024 次元）で再訓練し（旧版は `models/domain_classifier_pre_iter79_nomic.joblib` へ退避），`scripts/screen_embedding_models.py`（G1 専用・新規）を追加，`data/MANIFEST.md` に新 artifact の sha256・生成コマンド・G1/G2 実測値を追記した．`data/dataset.jsonl`（1,915 行）・`data/classifier_train.jsonl`（1,427 行）・訓練ハイパラ・その他 config は計画どおりビット単位で固定した．instruction prefix は付けていない．

#### 事前ゲートの結果

- **G1（値の選定，`classifier_train.jsonl` のみの 5-fold StratifiedKFold CV．評価集合不参照）**: `nomic-embed-text` cv_accuracy 0.5711 / macro-F1 0.5710，`qwen3-embedding:0.6b` cv_accuracy 0.7561 / macro-F1 0.7562．第 1 段で qwen3 が nomic を **+18.50pt** 上回ったため事前登録の規則どおりここで確定し，第 2 段（`bge-m3`）は起動しなかった．`scripts/screen_embedding_models.py` が `data/dataset.jsonl` を一切参照しないことをソースで確認済み（docstring L10 に明記，`--train-data` 既定は `classifier_train.jsonl`）．
- **G2（検出力）**: 旧/新 artifact の argmax replay で discordant n_d=787（閾値 30 を大幅超過）．本走実測でも `selected_domain` の discordant は **n_d=791**（必要偏り率 `1.96/sqrt(791)`=6.97%）で，replay と整合する．計画が予測した「n_d は 200 行以上」を上回り，レバーが実行時経路まで到達したことの直接証拠になっている．

#### 本走の結果（`results/20260926_221822/`，基準線 Iter78 `results/20260926_195929/`）

rc-executor の報告は，`metrics.py` の再実行と `results.jsonl` からの独立再計算で全項目を裏取りした．`expected_domains` と `query` は 1,915 行すべてで両走一致しており，評価集合は同一である（確認済み）．

| 指標 | Iter78 | Iter79 | 差 |
|---|---|---|---|
| top1_accuracy（全 1,915 行） | 0.589556 | **0.753003**（Wilson CI [0.73319, 0.77180]） | **+16.34pt** |
| cohens_kappa | 0.587438 | 0.721502 | +13.41pt |
| misrouting_rate | 0.410444 | 0.246997 | -16.34pt |
| ECE / Brier / AUROC | 0.050552 / 0.206919 / 0.731903 | 0.032745 / 0.152849 / 0.777614 | すべて改善 |
| fallback_rate / dispatch_failure_rate | 0.0 / 0.000522 | 0.0 / 0.000522 | 同一（同じ 1 行） |
| single_domain_top1（1,500 行） | 0.628667 | 0.749333 | +12.07pt |
| compound_domain_top1（415 行） | 0.448193 | 0.766265 | +31.81pt |
| compound_domain_set_recall | 0.393976 | 0.548193 | +15.42pt |
| compound_mean_dispatched_count | 2.506 | 1.880 | **-0.63**（後述） |
| answer_quality / end_to_end | 0.564 / 0.287728 | 0.569333 / 0.335770 | +0.53pt / +4.80pt |
| mean_duration_ms | 約 2300 | 2301.4 | 横ばい |

McNemar（連続性補正あり，自前再計算）: 全 1,915 行で a_only(旧のみ正解)=128，b_only(新のみ正解)=441，discordant=569，chi2=171.08，**p=4.30e-39**．部分集合も同様に全て有意改善（単一 1,500 行: 114/295，p=5.56e-19／複合 415 行: 14/146，p=3.91e-25／既存 1,600 行部分集合 top1 0.615000→0.751250，118/336，p=2.33e-24）．

**非退行①（per-domain recall 10 + precision 10 の計 20 指標，BH 補正 q=0.05）**: 自前で再計算（recall は exact McNemar，precision は Fisher）した結果，**有意 10 件・方向はすべて改善・有意退行 0 件**．点推定で悪化したのは `social_science_recall`（0.4242→0.4156，p=0.888）と `history_culture_recall`（0.7792→0.7706，p=0.896）の 2 件のみで，いずれも有意でない．非退行②（fallback=0.0，dispatch_failure=0.000522 ≤ 0.005）・非退行③（ECE 0.032745 ≤ 0.10．むしろ改善）も充足．

**到達確認**: `question_count`=1915，`compound_domain_question_count`=415，全 10 ノードで `ollama list` に `qwen3-embedding` と `config.yaml` の値を確認，新分類器の `n_features_in_`=1024，先頭 20 問の予備実行で 500 エラーなし．invalid 条件には一つも該当しない．

**検証**: `uv run ruff check` 新規失敗 0 件（既存 23 件は pre-existing），`uv run pytest tests/` 306 passed / 9 failed（9 件すべて B122 の pre-existing，新規失敗 0 件）．

#### 分析（交絡の検討 — +16.3pt は過去のどのレバーよりも大きいため，本物であることを個別に潰す）

1. **分類器の再訓練そのものの寄与と，埋め込みの寄与を分離できているか**．G1 は**同一の訓練データ・同一の訓練手順・同一のハイパラ・同一の CV 分割（`random_state` 固定）で埋め込みだけを差し替えた比較**であり，cv_accuracy 0.5711→0.7561（+18.50pt）を得ている．訓練手順が同一である以上，この差は埋め込み特徴量の品質に帰属する．本走の +16.34pt は G1 の +18.50pt より小さく，オフラインの分離度改善が実機へほぼそのまま（やや目減りして）伝播した，という素直な解釈と整合する．「再訓練したから上がった」だけなら nomic で再訓練した対照でも上がるはずだが，G1 の nomic 側がまさにその対照であり 0.5711 に留まっている．
2. **評価集合への情報漏洩**．G1 も再訓練も `data/classifier_train.jsonl` のみを使い，`data/dataset.jsonl` を参照していない（スクリプトの引数とソースで確認）．値の選定規則は結果を見る前に journal へ事前登録済みで，選定に評価集合の指標を一切使っていない．
   - ただし**副次的に既存の漏洩を 1 件発見した**．`classifier_train.jsonl` の質問本文と `dataset.jsonl` の質問本文が **72 行重複している**（education 54 行，history_culture 18 行）．d0002 §6-E の「重複 0 件」という記述は現状に合致しない（Iter35 の education 手作り問題追加，または Iter36/37 の `japanese_civics` 再割当以降に混入したと見られる）．**この 72 行は旧走・新走の双方に等しく含まれるため今回の比較を歪めないうえ，向きも逆である**: 漏洩 72 行では新モデルの方が**悪く**（0.875→0.653），漏洩 72 行を除いた 1,843 行では改善幅がむしろ広がる（0.578405→0.756918，**+17.85pt**）．したがって漏洩は今回の判定を有利側へ押していない．絶対値の水増しという別問題は残るため backlog B125 に起票した．
3. **Iter78 で拡充した新規 315 行が新モデルに有利に働いていないか**．複合設問を由来で分けると，**旧来の手作り 100 行が 0.410→0.780（+37.0pt，a=4/b=41，p=8.0e-8）**，Iter78 の LLM 生成 315 行が 0.460→0.762（+30.2pt，a=10/b=105，p=1.9e-18）で，**改善幅が大きいのは新規 315 行ではなく旧来の 100 行の側**である．新規行が新モデルに有利という交絡は支持されない．なお単独で最も改善が小さいのは単一ドメイン 1,500 行（+12.07pt）であり，複合設問ほど恩恵が大きいという構図になっている．
4. **想定外の挙動**．言語崩れ・発散・OOM は観測されず，`mean_duration_ms` も横ばい（2301.4）．唯一の想定外は `compound_mean_dispatched_count` の 2.506→1.880 という低下である．これは `dispatch_gap_threshold=0.29` を固定したまま確信度の分布が鋭くなった（`mean_confidence_std` が上がった）結果，rank2 以降が gap 条件で落ちやすくなった二次効果であり，レバーの直接の帰結ではない．**それでも `compound_domain_set_recall` は 0.394→0.548 と改善しており，「送る数を減らしたのに当てる数が増えた」**（送出先の質が上がった）と読める．ただし複合設問で 2 ドメイン中 1 つしか送られない行が増えている可能性があるため，`dispatch_gap_threshold` の再調整は独立レバーとして起票する価値がある（B125）．

#### 考察（採否判定）

事前登録した adopted 条件は「主基準（McNemar p<0.05 かつ点推定 +1.8pt 以上）かつ非退行①②③をすべて満たす」である．主基準は p=4.30e-39・+16.34pt で桁違いに充足し，非退行①（BH 補正後の有意退行 0 件）・②・③もすべて充足した．**adopted で確定する**．`config.yaml` の `qwen3-embedding:0.6b` と再訓練済み `models/domain_classifier.joblib` はこのまま本番構成として維持する（rejected 時の復元手順は不要となった）．

これは Iter15 以降の全反復で最大の単一レバー効果であり，Iter17 の `routing_method=supervised_classifier`（0.2059→0.5651）に次ぐ規模の前進である．top1_accuracy は初めて 0.75 を超えた．

**単一レバー原則の観点で，分類器の再訓練を同一レバーに含めた判断が妥当だったか**: 妥当だったと判断する．埋め込みを 768→1024 次元へ替えると旧 artifact は次元不一致で例外を投げるため，「差し替えだけして再訓練しない」という構成はそもそも実行不能であり，再訓練は選択肢ではなく差し替えの構成要素である．そのうえで G1 が「訓練手順を固定して埋め込みだけを変えた対照」を提供しているため，再訓練の寄与と埋め込みの寄与は事後に分離できている．この形（**付随作業が避けられない場合は，付随作業を固定した対照をオフラインで別に取る**）は今後の単一レバー運用の型として再利用できる．

#### 学び

1. **78 反復のあいだ，日本語タスクを英語中心の埋め込み空間の上で解き続けていた**．`nomic-embed-text` は英語中心モデルで，JMTEB 系のリストにすら載っていない．Iter29〜77 の大半は，その低品質な特徴量を前提にした後段（較正・閾値・intercept・集約方式・conformal）の微調整であり，改善幅は概ね ±1〜3pt に収まっていた．**後段の作り込みを重ねる前に，特徴量そのものの妥当性（入力言語とモデルの学習言語の一致）を疑うべきだった**．G1 相当の CV スクリーニングは数分で終わる．教訓は「安価なオフライン検証で前提そのものを測れる場合は，レバーの優先順位に関わらず早期に測る」である．
2. **公称ベンチマークの選び方が判定を左右する**．config.yml の lever note は MTEB multilingual 平均（nomic 62.28 vs qwen3 64.33，差 +2.05pt）を根拠にしていたが，実測は +18.50pt（CV）/ +16.34pt（本走）だった．一方，調査で見つけた JMTEB Classification は「qwen3-0.6b 66.09 は multilingual-e5-large 72.89 に劣る」と示しており，この数字だけを見ると着手を見送りかねなかった．**どちらの公称値も本タスクの効果量を予測できていない**．B124 で G1 を挟む判断をしたことが，この不確実性をコスト数分で解消した．公称ベンチは候補を絞る道具であって，採否の根拠にはならない．
3. **効果が大きいレバーほど，二次効果が別のハイパラを陳腐化させる**．確信度分布が鋭くなったことで `dispatch_gap_threshold=0.29`（Iter7x 世代に nomic の分布で調整した値）の実効が変わり，複合設問の平均送出数が 2.506→1.880 へ落ちた．**過去に調整したハイパラは，特徴量を入れ替えた時点で「調整済み」ではなくなる**．今回は結果的に set_recall も改善したため無害だったが，次以降は再調整を独立レバーとして検討する．
4. **train/eval の重複 0 件という前提は，データを触るたびに再検証が要る**．d0002 §6-E の確認は Iter17 時点のもので，Iter35/36/37 のデータ変更を経て現在 72 行が重複している．データセットを変更するレバーの計画フェーズに，重複チェックを定型作業として組み込むべきである．

---

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

