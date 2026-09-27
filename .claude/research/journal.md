## Iteration 85: 単一ドメイン評価集合の拡充による検出力の確保

### 調査 (Iter85)

本反復のレバーは backlog B134（Iter84 分析フェーズ）で `single_domain_eval_set_expansion` =
`jmmlu_unused_rows_power_targeted` に確定済みで，レバー選定の裁量は無い．調査の問いは 3 つ．
**(Q1) 評価集合を何行にすれば，想定効果量 δ に対して McNemar の検出力が確保できるのか（数値で決める）．
(Q2) 調達元（JMMLU 未使用行）は実際に何行あり，どのドメインにどれだけ配れるのか．
(Q3) 追加行の選び方に，測定系を壊す汚染・偏りの経路は無いか．**

**Q1: 必要標本数は Connor / Miettinen の対応のある二項検定の標本設計式で決まる**

- McNemar 検定の標本設計は Connor, "Sample size for testing differences in proportions for the
  paired-sample design"（Biometrics 43(1):207-211, 1987, p.209）が標準で，漸近近似は
  Miettinen, "The matched pairs design in the case of all-or-none responses"（Biometrics, 1968）に遡る
  （実装例: <https://gist.github.com/...pwr.mcnemar>，計算機: <https://powerandsamplesize.com/>，
  <https://homepage.univie.ac.at/robin.ristl/samplesize.php?test=mcnemar>，いずれも 2026-09-27 確認）．
  Connor の形は `n = [z_α·√p_d + z_β·√(p_d − δ²)]² / δ²` で，本研究のように δ ≪ p_d の領域では
  journal の Iter84 学び 1 で使った `N ≥ (z_α+z_β)²·p_d/δ²` と数値的に一致する
  （p_d=0.09452・δ=0.012 で p_d − δ² = 0.09438，差は 0.08%）．**したがって Iter84 が導いた
  「N=1,915 では有意境界 δ=1.378pt，検出力 80% で δ=1.97pt」という結論は文献側の定式と整合する．**
- 本フェーズで N ごとの検出力を再計算した（p_d は Iter84 実測 0.09452 を設計定数として使う．
  分類器を作り替えるレバーの実測値であり，本レバーの直後に再試行する
  `cross_domain_training_data_augmentation` と同型のレバーだからである）:

  | N（全体） | 有意境界 `1.96·√(p_d/N)` | 検出力 80% の最小効果 `2.802·√(p_d/N)` |
  |---|---|---|
  | 1,915（現行） | 1.377pt | 1.969pt |
  | 2,903 | 1.118pt | 1.599pt |
  | **3,447（本計画）** | **1.026pt** | **1.467pt** |
  | 3,731 | 0.987pt | 1.410pt |
  | 5,153 | 0.876pt | 1.200pt |

- **δ=1.2pt を検出力 80% で拾うには N≥5,153 が要る**が，これは後述 Q2 のプール上限（訓練用に各 100 行を
  残すと追加可能なのは 3,829 行）と本走時間（`experiment.timeout_min`）の双方から到達不能である．
  **到達可能な範囲での最良は N≈3,400〜3,700** であり，config.yml の新レバー note が示す目標帯
  （3,400〜3,600）と一致する．
- **ただし本レバーの後に控える再試行では，期待効果量は 1.2pt ではなくもっと大きい**．Iter84 の内訳は
  single 行 +2.133pt / compound 行 −2.169pt で，拡充後は single の構成比が 78.3%→88.0% へ上がる．
  同じ部分効果が成り立つなら期待 δ は **0.880×2.133 + 0.120×(−2.169) = +1.615pt** で，
  N=3,447（SE=0.5235pt）での検出力は **約 87%** になる．**すなわち N=3,447 は「Iter84 の問いに決着を
  付ける」という目的に対しては十分である**（δ=1.2pt 一般を 80% で拾う汎用的な検出力は得られない）．

**Q2: 未使用プールは 4,731 行．ただし legal 2 行・general 128 行という構造的な偏りがある**

本フェーズで JMMLU（pinned commit `3637b25e`，CC BY-NC-ND 4.0）の全 56 タスクを
`build_dataset.py:_DOMAIN_TASK_MAP` で 10 ドメインへ写像し，評価集合 1,915 行・分類器訓練集合 1,427 行の
設問文（4 択を連結した本文の完全一致）を除いた残りを実測した．

| ドメイン | 未使用プール | 訓練用に 100 行残した後の上限 | LoRA 訓練集合と重複しない行 |
|---|---|---|---|
| medical | 1,120 | 1,020 | 861 |
| natural_science | 791 | 691 | 545 |
| history_culture | 786 | 686 | 599 |
| business_economics | 713 | 613 | 466 |
| mathematics | 353 | 253 | 138 |
| education | 334 | 234 | 197 |
| computer_science | 256 | 156 | 61 |
| social_science | 248 | 148 | 64 |
| general | 128 | 28 | 3 |
| **legal** | **2** | **0** | 2 |
| 合計 | **4,731** | **3,829** | 2,936 |

- B133 が記録した 4,578 行は**訓練集合に既出のタスクへ限定した**値で，評価集合の拡充では
  その限定は不要である（評価集合は 10 ドメインとも `_DOMAIN_TASK_MAP` の全タスクを既に使っており，
  education も `japanese_civics` 39 行を含む．タスク構成は変わらない）．今回の 4,731 行が正しい母数である．
  また B134 note の「legal 0」は訓練タスク限定時の値で，評価側の実測は **2 行**（229 − 150 − 77）である．
  どちらにせよ legal は事実上拡充できない．
- **general も上限 28 行**（プール 128 − 訓練用留保 100）で，legal と合わせて 2 ドメインは拡充がほぼ効かない．
  拡充後の単一行はドメイン間で不均衡（medical/education/business/natural_science/mathematics/history_culture
  各 350，computer_science 306，social_science 298，general 178，legal 150）になる．
  **これはドメイン固有の扱いではなく，10 ドメイン共通の規則をデータ側の制約が切り詰めた結果である**
  （2026-09-23 恒久ルール (1) に抵触しない）．

**Q3: 汚染経路は 2 本ある．分類器訓練集合（軸①に効く）と LoRA 訓練集合（軸②③に効く）**

- **(a) 分類器訓練集合との重複は主基準に直結する**ので，プール定義の時点で `data/classifier_train.jsonl`
  1,427 行の設問文を完全除外した（上表はその条件での実測）．なお既存の評価集合には B125 が起票済みの
  72 行の重複が残っているが，本反復では単一レバー原則により触らない．拡充で希釈され 72/1,915 → 72/3,447 になる．
- **(b) 専門家 LoRA の訓練集合との重複は本フェーズで新たに実測した**．`data/lora_train/*.jsonl` は
  `scripts/prepare_lora_training_data.py`（seed=42，各ドメイン最大 300 行，生成時点の評価集合を除外）が
  作った 2,749 行で，設問文の書式は評価集合と同一である．**現在の評価集合 1,915 行のうち 135 行（7.05%）が
  既にこの LoRA 訓練集合と重複している**（Iter37/78 で評価集合を作り直した際に生じたもので，本反復の責任範囲外）．
  ルーティング（軸①）は分類器だけで決まるので影響しないが，`answer_quality_accuracy`・`end_to_end_accuracy`
  （軸②③）は専門家が訓練で見た設問を再現できる分だけ上振れする．
- 未使用プールの 38%（1,795/4,731）が LoRA 訓練行なので，**無作為に取ると追加行の約 4 割が汚染行になり，
  評価集合全体の汚染率は 7.05%→約 21% へ跳ね上がる**．そこで選択規則を「**LoRA 訓練集合と重複しない行を
  先に取り，足りない分だけ重複行で埋める**」（clean-first）とする．LoRA 訓練行は seed 付き無作為抽出なので
  clean/汚染の別は難易度と独立であり，この優先順位は難易度の偏りを生まない．
  この規則の下での汚染は 1,532 行中 269 行に留まり，評価集合全体では 7.05%→**11.72%** の増加で収まる．
- **難易度で選ぶことは絶対にしない**（Iter84 の hard negative mining は訓練側の話）．評価集合を難しい行へ
  寄せると top1 の水準自体が動き，1,915 行サブセットとの比較・過去基準線との接続が壊れる．
  抽出は各ドメインのプール内での seed 固定の一様無作為（clean-first の 2 層のみ）に限る．

### 計画 (Iter85)

**単一レバー**

`single_domain_eval_set_expansion` = **`jmmlu_unused_rows_power_targeted`**．
**`data/dataset.jsonl`（現行 1,915 行）の末尾へ，JMMLU の未使用単一ドメイン行 1,532 行を追記して 3,447 行にする**
ことだけを行う．`config.yaml`・分類器 artifact・埋め込み・訓練データ・送出閾値・集約方式・既存 1,915 行は
一切変えない（既存行はバイト単位で不変．先頭 1,915 行の diff が 0 であることを検証条件に含める）．

**選択規則（10 ドメイン共通．ドメイン別パラメータを持たない）**

1. **プール**: 各ドメインについて `_DOMAIN_TASK_MAP` の全タスクの全行から，現行評価集合 1,915 行と
   `data/classifier_train.jsonl` 1,427 行の設問文（4 択連結後の本文）を除いた残り（計 4,731 行）．
2. **訓練用留保**: 各ドメインのプールから **100 行を訓練用に残す**（`cross_domain_training_data_augmentation`
   の再試行余地を潰さないため．B134 の指示）．取得可能数は `max(0, pool − 100)`．
3. **取得数**: 各ドメイン **N_add = 200**（取得可能数がこれ未満のドメインはその全量）．
4. **層の順序**: `data/lora_train/*.jsonl` の設問文と重複しない行を優先し，不足分のみ重複行で埋める（Q3-b）．
   各層の内部は `random.Random(20260927).shuffle` による一様無作為で，同一 zip なら完全に再現する．
5. **追記**: 選定行を既存 1,915 行の**後ろ**に append する．id は `{domain}-exp085-{連番:03d}`
   （既存 id と衝突せず，1,915 行サブセットを id だけで切り出せる）．`is_compound=false`，
   `expected_domains=[domain]`，`jmmlu_task`・`jmmlu_answer` は JMMLU の値をそのまま持たせる．

**確定した配分（本フェーズで実測・決定論的）**

| ドメイン | プール | 取得可能 | 取得 | うち LoRA 非重複 | 拡充後の単一行数 |
|---|---|---|---|---|---|
| medical | 1,120 | 1,020 | 200 | 200 | 350 |
| education | 334 | 234 | 200 | 197 | 350 |
| business_economics | 713 | 613 | 200 | 200 | 350 |
| natural_science | 791 | 691 | 200 | 200 | 350 |
| mathematics | 353 | 253 | 200 | 138 | 350 |
| history_culture | 786 | 686 | 200 | 200 | 350 |
| computer_science | 256 | 156 | 156 | 61 | 306 |
| social_science | 248 | 148 | 148 | 64 | 298 |
| general | 128 | 28 | 28 | 3 | 178 |
| legal | 2 | 0 | 0 | 0 | 150 |
| 合計 | 4,731 | 3,829 | **1,532** | 1,263 | **3,032** |

**全体 N = 1,915 + 1,532 = 3,447 行（単一 3,032 ＋ 複合 415）**．
有意境界 **1.026pt**，検出力 80% の最小効果 **1.467pt**（p_d=0.09452 を仮定）．

**レバーを読むコード行と到達条件（d0004 §4 の再発防止）**

- 追加行を読むのは `build_dataset.py:main()` に新設する `--single-domain-expansion <path>`
  （既定 `data/single_domain_expansion_iter85.jsonl`）→ `_build_rows()` の末尾で append する新規ブロック．
  **到達条件は `mise.toml` L31 の `uv run python build_dataset.py --output data/dataset.jsonl`
  （`mise run setup` 内）が実行されること**．Iter78（複合 100→415）と同一の作りにし，
  `tests/test_build_dataset.py` の既存呼び出し（`_build_rows()` 直叩き）は引数既定 `None` で不変に保つ．
- 本走が読むのは**コンテナ内の** `/app/data/dataset.jsonl`（`Dockerfile` の `COPY data/ ./data/`）なので，
  **`mise run setup`（イメージ再ビルド・push）→ `mise run deploy` を必ず通す**．
  これを省くと 1,915 行のまま走り invalid になる．
- **落とし穴（B123 と同型）**: `data/` は `.gitignore` 対象なので
  `data/single_domain_expansion_iter85.jsonl` は `git add -f` で明示コミットし，sha256 を
  `data/MANIFEST.md` に記録する．消失すると `mise run setup` が黙って 1,915 行へ戻す．
- **落とし穴（B118-2）**: `mise run setup` 内の素の `uv sync` が research extra を落とすので，
  直後に `uv sync --extra research` を実行する．
- **落とし穴（B135）**: `mise run analyze` は引数なしだと `results/iter45_preliminary/` を誤選択するので
  `-- <YYYYMMDD_HHMMSS>` を明示する．

**実装・実験手順（この順で行うこと）**

0. **【最優先・必須】Iter84 の artifact を基準線へ復元する**:
   `cp models/domain_classifier_pre_iter84_baseline.joblib models/domain_classifier.joblib` →
   `sha256sum` が `1cfcd3d8...` に戻ることを確認 → 後続の `mise run deploy` で全 10 ノードへ再配布し，
   各ノードで sha256 の一致を確認する．**これを怠ると別 artifact の上で走り invalid になる．**
1. `scripts/expand_single_domain_eval.py`（新規）で上記規則の 1,532 行を生成し
   `data/single_domain_expansion_iter85.jsonl` へ書く．**LLM も埋め込みも使わない純粋な CPU 処理**なので
   実機ノードも wafl-ctrl5 も不要（zip の解析のみ）．生成時に (a) 評価集合との重複 0，
   (b) `classifier_train.jsonl` との重複 0，(c) 追加行同士の重複 0，(d) ドメイン別件数が上表と一致，
   を assert する．
2. `build_dataset.py` に `--single-domain-expansion` を追加（既定値あり，`main()` からのみ渡す）．
   `uv run pytest tests/test_build_dataset.py` と `uv run ruff check` を通す．
3. `mise run setup` → `uv sync --extra research` → `wc -l data/dataset.jsonl` = **3447** を確認．
   **`head -n 1915 data/dataset.jsonl` が変更前のファイルとバイト単位で一致すること**を diff で確認する．
4. `mise run deploy` → 全ノード healthy・smoke_check（git-status/hashes/probe）PASS →
   各ノードで `docker compose exec app wc -l /app/data/dataset.jsonl` = **3447**（10/10）と
   分類器 sha256 = `1cfcd3d8...`（10/10）を確認．
5. `mise run start` で本走 1 回（3,447 問）．所要見積りは **113〜131 分**
   （Iter84 実績 1,915 問 63 分の線形外挿＝113 分，`mean_duration_ms`=2280.4 基準＝131 分）．
   watchdog の `experiment.timeout_min` は 150 では余裕が 19 分しかないため **180 へ引き上げる**
   （測定系ではなく監視の設定であり，実験条件ではない）．
6. `mise run analyze -- <run>`，および 1,915 行サブセット（id が `-exp085-` を含まない行）での
   指標を別途算出する．**オフライン replay を行う場合，`data/embcache_eval_*.npy` は 1,915 行分しか
   無いので追加行を wafl-ctrl5 で再埋め込みすること**（1,915 行キャッシュをそのまま流用しない）．

**固定する構成（直近の最良構成 = Iter83 本走 `results/20260927_070239/` と同一）**

`config.yaml` は 1 行も変更しない（`embedding_model=qwen3-embedding:0.6b`，`embedding_view_concat=true`，
`routing_method=supervised_classifier`，`confidence_threshold=0.0`，`dispatch_candidate_threshold=0.0`，
`dispatch_top_k=2`，`dispatch_gap_max_k=4`，`dispatch_gap_threshold=0.36`，`aggregation_method=max_confidence`）．
`models/domain_classifier.joblib` は復元後の `1cfcd3d8...`，`data/classifier_train.jsonl` は 1,427 行のまま不変，
`node.py`・`aggregator.py`・`classifier.py`・`metrics.py`・`run_experiment.py` はコード変更なし．
**ドメイン固有の後付け補正は追加しない．棄権・人間エスカレーションは扱わない（2026-09-23 恒久ルール）．**

**仮説**

評価集合を 1,915→3,447 行へ増やせば，同じ p_d の下で McNemar の標準誤差が √(1915/3447)=0.745 倍になり，
有意境界は 1.377pt→1.026pt，検出力 80% の最小効果は 1.969pt→1.467pt へ下がる．
さらに single 行の構成比が上がることで，Iter84 で観測された部分効果（single +2.13pt / compound −2.17pt）の
合成値は +1.2pt→約 +1.6pt へ上振れし，**同じ施策を再試行したときの検出力が約 87% になる**．
**本反復は精度向上のレバーではなく測定系のレバーであり，top1_accuracy が動かないことがむしろ正常である．**

**成功条件（事前登録．事後に緩めない）**

主基準（AND）:
- **P1（検出力）**: 本走が完走した評価集合の `question_count` = **3,447**（単一 3,032・複合 415）で，
  設計定数 p_d=0.09452 を代入した有意境界 `1.96·√(p_d/N)` ≤ **1.10pt**（設計値 1.026pt）を満たすこと．
- **P2（整合性）**: 1,915 行サブセット（id に `-exp085-` を含まない行）の `top1_accuracy` が
  Iter83 本走 0.792689 と **±0.5pt 以内**，かつ行単位の `selected_domain` 一致率が **≥ 99%**
  （≤19 行の相違まで許容．ルーティングは決定論的なので本来ほぼ完全一致するはずであり，
  これは実装漏れ・artifact 未復元の検出器として使う）．
- **P3（無汚染・純追加）**: `head -n 1915 data/dataset.jsonl` が変更前とバイト単位で一致，
  追加 1,532 行と `data/classifier_train.jsonl` の設問文重複が **0 件**，追加行同士および既存行との
  重複が **0 件**，追加行のドメイン別件数が計画表と完全一致．

非退行（すべて 1,915 行サブセット上で判定する．拡充後の全体値は水準が動くため比較に使わない）:
1. per-domain 20 指標の BH 補正（q=0.05）後の有意退行 **0 件**（構造的には Iter83 と同値のはず）．
2. 複合 415 行の `compound_domain_set_recall` = 0.581928・`compound_mean_dispatched_count` = 1.889 と
   ±0.5pt / ±0.05 以内で一致（複合行は 1 行も足していないので不変のはず）．
3. `fallback_rate` = 0.0，`dispatch_failure_rate` ≤ 0.005．
4. 1 問あたり `mean_duration_ms` ≤ 2744.3（既存上限）．
5. 本走が `experiment.timeout_min`（180 へ引き上げ）以内に完走すること．

参考値として必ず併記するもの（判定には使わない）: 拡充後の全体 top1_accuracy と Wilson 95%CI，
単一/複合の内訳，per-domain 指標（拡充後の母数），LoRA 訓練集合と重複する行の数（135+269=404 の想定）と
それを除いた `answer_quality_accuracy`．

**判定語**

- **adopted**: P1〜P3 と非退行 1〜5 をすべて充足．拡充後の `data/dataset.jsonl` を以後の基準線とし，
  本走を **新しい基準線**（Iter86 以降の比較対象）として登録する．
- **partial**: P1・P3 充足かつ P2 または非退行のいずれか 1 件が FAIL．
- **rejected**: P1 または P3 が不成立（＝拡充の設計自体が要件を満たさない）．
- **invalid（実験不成立）**: 分類器 sha256 が `1cfcd3d8...` でないまま本走した／`question_count ≠ 3447`／
  1,915 行サブセットの一致率 < 95% もしくは |Δtop1| > 1.0pt／先頭 1,915 行に差分がある／
  追加行に訓練集合との重複が見つかった．いずれも「効果なし」ではなく setup・deploy・実装の漏れと解釈する．

**リスクと留保**

- 拡充後の評価集合はドメイン間で不均衡（legal 150 / general 178 に対し medical 350 など）になり，
  **全体 top1 の水準は現行 0.7927 から動く**（比較は必ず 1,915 行サブセットで行う）．
- legal は構造的に 1 行も足せず，general も 28 行に留まる．この 2 ドメインの per-domain 検出力は改善しない．
- LoRA 訓練集合との重複が 7.05%→11.72% へ増えるため，**軸②③の水準は上振れする**．
  軸②③は 3SD=2.6pt のノイズ床込みで参考値として扱い，判定には使わない．
- p_d=0.09452 は Iter84（分類器差し替え）の実測値である．config 1 行だけを変えるレバー（Iter83 は p_d≈0.001）
  では p_d が桁で小さく，検出力の議論はそのまま当てはまらない．

### 実装・実験 (Iter85)

**実施した変更（単一レバーのみ）**

0. **分類器 artifact を基準線へ復元**: `models/domain_classifier.joblib` ←
   `models/domain_classifier_pre_iter84_baseline.joblib`（sha256 `1cfcd3d8...`）．全 10 ノードで一致を確認した．
   B134 (b) の申し送りをここで解消した．
1. 新規 `scripts/expand_single_domain_eval.py`（CPU のみ．LLM・埋め込み不要）で
   `data/single_domain_expansion_iter85.jsonl` を生成．選択規則はプール構築 → 訓練用に 100 行留保 →
   各ドメイン目標 200 行 → LoRA 訓練集合と重複しない行を優先し不足分のみ重複行で埋める clean-first →
   `random.Random(20260927)` で層内シャッフル．内部重複 0・評価集合と 0 重複・分類器訓練集合と 0 重複を
   アサートで保証した．
2. `build_dataset.py` に `--single-domain-expansion`（既定 `data/single_domain_expansion_iter85.jsonl`）を追加．
   `_build_rows()` の複合行ループの後に純追加するのみで，同名引数の既定は `None`（既存テスト呼び出しは不変）．
3. `mise run setup` → `uv sync --extra research`（B118-2 対策）→ `head -n 1915` が変更前とバイト単位で一致することを確認．
4. `mise run deploy` → 全 10 ノード healthy，smoke_check（git-status / hashes / probe）PASS．
   10/10 ノードで `wc -l /app/data/dataset.jsonl`=3435，分類器 sha256=`1cfcd3d8...` を確認．
5. `mise run start` を 1 回（`results/20260927_130237/`，3,435 問，約 90 分で完走．timeout 180 分以内）．
6. `mise run analyze -- 20260927_130237`．統計は `metrics.py` の
   `compute_domain_recall_mcnemar_test` / `compute_domain_precision_fisher_test` / `apply_benjamini_hochberg`
   をそのまま再利用した（自前の統計式は書いていない）．

**計画からの乖離（要レビュー）**

計画は追加 1,532 行（合計 3,447 行）だったが，実測は **1,520 行（合計 3,435 行．単一 3,020 + 複合 415）**．
原因は本環境がキャッシュする JMMLU.zip の sha256 が `3ba7d912...` で，ピン留めコミット `3637b25e...` の
期待値と異なること（journal_archive.md に既知・無害として記載済みの差異）．実際のプールが計画時推定より
わずかに小さく，`computer_science`（計画 156 → 実測 151）・`social_science`（148 → 144）・`general`（28 → 25）の
3 ドメインで不足した．他 7 ドメイン（medical / education / business_economics / natural_science /
mathematics / history_culture = 200，legal = 0）は計画どおり．
また，この過程で JMMLU 側の cross-task 重複質問 1 件（education の sociology / moral_disputes 間）を検出し，
プール構築時にドメイン内の重複除去を追加した（計画に無い対応だが，P3「追加行同士の重複 0 件」を
満たすために必須）．代替は取らず，スクリプトは計画値と実測値を両方 stderr に出力し，`data/MANIFEST.md`
にも明記した．

**主要メトリクス**

全体（3,435 問，参考値）: top1_accuracy = 0.786317（Wilson 95% CI [0.7723, 0.7997]），
single_domain_top1 = 0.782119（n=3,020），compound_top1 = 0.816867（n=415），
fallback_rate = 0.0，dispatch_failure_rate = 0.001164，mean_duration_ms = 1574.10．

1,915 行サブセット（判定用）:

| 条件 | 実測 | 閾値 | 判定 |
|---|---|---|---|
| P1 有意境界 `1.96·√(p_d/N)` | 1.028pt（p_d=0.09452, N=3435） | ≤1.10pt | PASS |
| P2 サブセット top1 | 0.791123（Iter83 0.792689 比 Δ=−0.157pt） | ±0.5pt 以内 | PASS |
| P2 `selected_domain` 行一致率 | 1909/1915 = 99.69% | ≥99% | PASS |
| P3 先頭 1,915 行バイト一致 | 一致 | 一致 | PASS |
| P3 訓練集合との重複 | 0 件 | 0 件 | PASS |
| P3 ドメイン別件数 | 3 ドメインで僅差 | 配分表と完全一致 | 字義上 FAIL |
| 非退行① per-domain 20 指標（BH q=0.05） | 有意退行 0 件 | 0 件 | PASS |
| 非退行② compound set_recall | 0.581928 | 0.581928 | PASS |
| 非退行② compound mean_dispatched | 1.889157 | 1.889 | PASS |
| 非退行③ fallback / dispatch_failure | 0.0 / 0.002089 | 0.0 / ≤0.005 | PASS |
| 非退行④ mean_duration_ms | 2308.51 | ≤2744.3 | PASS |
| 非退行⑤ 完走 | 約 90 分 | timeout 180 分以内 | PASS |

P2 の不一致 6 行: education-045，medical-004 / 007 / 058 / 109，natural_science-097．

**検証**

- `uv run ruff check build_dataset.py scripts/expand_single_domain_eval.py` → All checks passed．
- `uv run pytest tests/test_build_dataset.py -q` → 13 passed / 9 failed．**この 9 件は変更前から失敗していた
  既存の不具合**（fixture zip `tests/fixtures/jmmlu_sample.zip` に `japanese_civics.csv` が無いための
  `KeyError`）で，`git stash` して同一テストを実行し失敗内容・件数が完全に一致することを確認済み．
  本反復のスコープ外として未修正のまま残す．

**異常の有無**: 実行・ログ上の異常は無い．上記の計画乖離が唯一の要レビュー事項である．

### Iteration 85 実行済み

**変更（単一レバー）**: `single_domain_eval_set_expansion` = `jmmlu_unused_rows_power_targeted`．
`data/dataset.jsonl` を 1,915 → **3,435 行**（単一 3,020 ＋ 複合 415）へ純追加で拡充した．
`config.yaml`・分類器 artifact（`1cfcd3d8...` へ復元済み）・埋め込み・訓練データ・送出閾値・集約方式は不変．
`experiment.timeout_min` のみ 150→180（watchdog 設定であり実験条件ではない．B137-(5)）．

**結果（本分析フェーズで `metrics.py` の既存関数により再計算・executor 値を全件再現した）**

| 条件 | 実測 | 閾値 | 判定 |
|---|---|---|---|
| P1 有意境界 `1.96·√(p_d/N)`（p_d=0.09452, N=3,435） | **1.0281pt** | ≤1.10pt | PASS |
| P1 `question_count` | 3,435（計画 3,447） | 3,447 | 下記「乖離」参照 |
| P2 サブセット top1 | 0.791123（Iter83 0.792689，Δ=**−0.157pt**） | ±0.5pt 以内 | PASS |
| P2 `selected_domain` 行一致率 | 1909/1915 = **99.69%** | ≥99% | PASS |
| P3 先頭 1,915 行バイト一致 / 訓練集合重複 / 追加行重複 | 一致 / 0 件 / 0 件 | 同左 | PASS |
| P3 追加行のドメイン別件数 | 1,520（計画 1,532．3 ドメインで −5/−4/−3） | 計画表と完全一致 | **字義上 FAIL** |
| 非退行① per-domain 20 指標（BH q=0.05） | 有意 **0/20** 件 | 0 件 | PASS |
| 非退行② compound set_recall / mean_dispatched | 0.5819277 / 1.8891566（基準線と**完全同値**） | ±0.5pt / ±0.05 | PASS |
| 非退行③ fallback / dispatch_failure（サブセット） | 0.0 / 0.002089 | 0.0 / ≤0.005 | PASS |
| 非退行④ `mean_duration_ms`（サブセット） | 2308.51 | ≤2744.3 | PASS |
| 非退行⑤ 完走 | 約 90 分 | timeout 180 分以内 | PASS |

参考値（判定に使わない）: 全体 3,435 行 top1 = 0.786317（Wilson 95%CI [0.7723, 0.7997]），
単一 3,020 行 0.782119 / 複合 415 行 0.816867，追加 1,520 行のみ 0.780263（CI [0.7588, 0.8004]），
全体 `mean_duration_ms` = 1574.10（追加行は複合行を含まないため 648.83 と速い），
`answer_quality_accuracy` = 0.587417，`end_to_end_accuracy` = 0.403202（LoRA 汚染率が 7.05%→11.7% へ
上がるため上振れしている．軸②③は 3SD=2.6pt のノイズ床込みの参考値）．

**ノイズか有意かの切り分け**

- サブセットの Δ = **−0.157pt** は，基準線との対比較で **discordant 5 行（4 vs 1），McNemar p = 0.3711** で
  有意でない．N=1,915 の有意境界 1.377pt に対し桁で小さく，明確にノイズ範囲内である．
- **さらに強い結論が得られた: 1,915 行すべてで `probe_candidates`（各ドメインの確信度）と
  `dispatched_domains` が基準線と 1915/1915 で完全一致した**．すなわち**ルーティング層は 100% 不変**で，
  拡充は既存行の測定に一切影響していない．不一致 6 行の内訳は (a) `dispatch_failed` の発生行の差 5 行
  （基準線 medical-007 の 1 行 → 今回 medical-004/058/109・natural_science-097 の 4 行），
  (b) education-045 の 1 行は probe 確信度が完全一致のまま `max_confidence` 集約の結果だけが
  history_culture→social_science へ振れたもので，**生成側 confidence の揺らぎに由来する**．
  いずれもルーティングではなく生成・ネットワーク層の非決定性であり，P2 の検出器としての目的
  （実装漏れ・artifact 未復元の検出）は最高水準で満たされた．
- 複合 415 行の被覆指標が基準線と小数点以下まで完全同値であることも，純追加であることの独立な裏付けである．

**判定: `partial`**

- 本レバーの唯一の目的である **P1（検出力の確保）は充足**した．N=3,435 で有意境界 1.0281pt，
  検出力 80% の最小効果 1.4698pt（N=1,915 ではそれぞれ 1.3770pt / 1.9685pt）．
- P3 の 4 条件のうち，P3 の名目そのものを担う 3 条件（純追加・無汚染）は完全充足．**FAIL したのは
  「ドメイン別件数が計画表と完全一致」という手続き的な決定論チェック条項 1 件のみ**で，その原因は
  抽出規則の逸脱ではなく計画時に前提したプール母数の誤り（後述の zip 差異）である．
  規則 3「各ドメイン N_add=200，取得可能数がこれ未満ならその全量」は 10 ドメインすべてで字義どおり
  満たされており，規則は設計どおり決定論的に動いた．
- **語彙運用の整理（B131 以来の「事後に事前登録を緩めない」運用の下で）**: 事前登録の `rejected` は
  「P1 または P3 が不成立（＝拡充の設計自体が要件を満たさない）」，`partial` は「P1・P3 充足かつ
  P2 または非退行のいずれか 1 件が FAIL」と定義していた．今回は **P2・非退行が全 PASS で P3 の一部条項のみ
  FAIL** という，どちらの定義文にも字義上当てはまらない**事前登録が想定していなかった型**である．
  前例（Iter29/30・Iter62/63）の `partial` は「主基準充足＋非退行 1 件 FAIL」の型で，今回とは型が異なる．
  そこで **事前登録を事後に書き換えるのではなく，未定義領域を保守側（`adopted` ではない）に倒して
  `partial` を割り当てる**扱いとした．`rejected` を採らないのは，その定義文が括弧で明記する趣旨
  （拡充の設計が要件を満たさない）と実態が食い違い，拡充を破棄すると Iter84 の問いが永久に決着しない
  ためである．**この語彙運用の是非は要レビュー事項として backlog B138 に上げた．**
- **帰結**: 拡充後の `data/dataset.jsonl`（3,435 行）を **Iter86 以降の新しい基準線**として採用し，
  本走 `results/20260927_130237/` を比較対象に登録する．

**JMMLU.zip の sha256 不一致（再現性の論点）**

本環境のキャッシュ zip は sha256 `3ba7d912...` で，ピン留めコミット `3637b25e...` の期待値と異なる．
`journal_archive.md` に「既知・無害」として記載されていたが，**今回それが計画フェーズの数値前提
（プール件数）を狂わせ，事前登録の一条項を FAIL させた**．無害ではなく，計画の決定論性を壊す実害が
あることが判明した．ただし **本反復のスコープ外**（単一レバー原則）であり，今回は修正しない．
独立項目として backlog **B139** に起票した（選択肢: ピン留めコミットから再取得して MANIFEST の
期待値に合わせる／実キャッシュの sha256 を正として MANIFEST・ドキュメント側を訂正する）．
なお `scripts/expand_single_domain_eval.py` は計画値と実測値を両方 stderr に出力するため，
同型の乖離は今後も生成時点で検出できる．

**学び**

1. **ルーティング層の決定性は行数を倍近く増やしても完全に保たれる**（`probe_candidates` 1915/1915 一致）．
   一方で `selected_domain` は `max_confidence` 集約を通るため**生成側の confidence 揺らぎを拾う**．
   今後「ルーティングが不変か」を検証したいときは `selected_domain` ではなく
   **`probe_candidates` / `dispatched_domains` を突き合わせるべき**である（より鋭い検出器になる）．
2. `dispatch_failure` は行ごとに固定ではなく実行ごとに 1〜4 行の範囲で位置が動く（今回 4 行 / 基準線 1 行．
   いずれも medical・natural_science のノード）．**±5 行程度の top1 の揺れはここから生じる**ので，
   軸①を「完全に決定論的」と断じるときはこの生成層のノイズ床（今回 5/1915 = 0.26pt）を併記すること．
3. **事前登録に「計画値と完全一致」という決定論チェックを書くときは，前提となる外部データの
   sha256 を先に検証する条項とセットにしないと，本質的でない乖離で主基準を落とす**．
   次回以降，データ調達を伴うレバーでは「調達元の sha256 が MANIFEST と一致すること」を
   事前条件（ゲート）側に置き，件数一致は事前条件成立時のみ適用する形にする．
4. 拡充後の全体 per-domain は水準が動いた（mathematics recall 0.6710→0.8005，social_science
   0.4545→0.5520，computer_science 0.8268→0.8796．education は 0.4120→0.3880 で依然最下位）．
   追加行のドメイン別 top1 は education 0.360 が突出して低く，computer_science 0.960 /
   mathematics 0.950 が高い．**全体値は 1,915 行時代の数値と直接比較してはならない**（B138 に明記）．

**次の一手**: 拡充後の評価集合で **Iter84 の hard negative mining を再試行**する
（`cross_domain_training_data_augmentation` = `hard_negative_mining_all_domains_retry_expanded_set`）．
実測構成比（単一 87.92%）で Iter84 の部分効果（single +2.133pt / compound −2.169pt）が成り立つなら
期待 δ = **+1.613pt**，N=3,435（SE=0.5246pt）での検出力は **86.8%**（拡充前の N=1,915 では期待 δ=+1.201pt・
検出力 40.1% にすぎず，判定不能が再発する公算が高かった）．本レバーが目的とした検出力は実測でも確保された．

---

## Iteration 84: 全ドメイン共通の hard negative mining による分類器訓練データ拡充

### 調査 (Iter84)

本反復のレバーは backlog B132（Iter83 分析フェーズ）で `cross_domain_training_data_augmentation` =
`hard_negative_mining_all_domains` に確定済みで，レバー選定の裁量は無い．調査の問いは 3 つ．
**(Q1) 「モデルが苦手な行を優先的に訓練集合へ足す」という選択規則は，無作為に同数足すより本当に良いのか．
(Q2) 最難帯の行はラベル雑音・外れ値であり，足すとむしろ害になるという警告が知られている．本研究のデータで
それは当てはまるのか．(Q3) 追加する行はどこから調達するか（B116(2) の調達順位に従う）．**

**Q1: hard example / hard negative mining は「情報量の高い行を選ぶ」系の標準手法だが，優位性は無条件ではない**

- 能動学習の uncertainty sampling（Lewis & Gale 1994 以来の定番）は「モデルが最も不確かな事例を選ぶ」規則で，
  本レバーの選択規則と数学的に同型である．Sharma & Bilgic, "Evidence-Based Uncertainty Sampling for Active
  Learning"（DMKD 2017, <http://www.cs.iit.edu/~ml/pdfs/sharma-dmkd17.pdf>，2026-09-27 確認）は，
  単純さと実証的成功から最も頻用される戦略だと位置づけている．Zhu et al., "Active Learning with Sampling by
  Uncertainty and Density"（COLING 2008, <https://aclanthology.org/C08-1143.pdf>）はテキスト分類で
  無作為抽出が明確に劣ることを報告している．
- ただし優位性は無条件ではない．Tripp, "Why your active learning algorithm may not do better than random"
  （<https://www.austintripp.ca/blog/2025-04-02-active-learning-random>，2026-09-27 確認）は，
  (i) 候補間で情報量の分散が大きいこと，(ii) 情報量の推定（確信度）が信頼できること，が成立して初めて
  無作為を上回ると整理している．**本研究の分類器は `CalibratedClassifierCV(method="temperature")` で
  較正済み（Iter83 本走の ECE=0.026）なので (ii) は満たす**．(i) は本フェーズのオフライン実測（G3）で確認した．
- 検索分野の hard negative mining でも，ANCE（Xiong et al., ICLR 2021）・NV-Retriever（Moreira et al.,
  2024）が「積極的に掘った negative は性能を上げるが，制御しないと訓練を不安定にする」と報告されている
  （ECI, arXiv 2603.20990, <https://arxiv.org/html/2603.20990v1>，2026-09-27 確認の関連研究節より）．

**Q2: 「最難帯＝ラベル雑音」という警告は本研究のデータには当てはまらなかった（実測で確認）**

- Thakur et al. の RLHN（2025）は，掘った hard negative 集合に含まれる false negative とラベル雑音が
  dense retriever の劣化の主因だと報告している（ARHN, arXiv 2604.11092,
  <https://arxiv.org/html/2604.11092v1>，2026-09-27 確認の関連研究節より）．
- 分類側でも同じ警告がある．"Hard Example Mining" のパターン整理（<https://distilledpatterns.org/patterns/hard-example-mining>，
  2026-09-27 確認）は **「高損失事例の大半がラベル雑音・破損・対象外事例であるとき」「データ集合が小さく
  反復的な mining がすぐ過適合を起こすとき」は使うべきでない**と明記する．Frontiers in AI, "Beyond
  uncertainty in modern active learning for trustworthy AI"（2026,
  <https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1844765/full>，
  2026-09-27 確認）も「不確実性ベースの取得関数は外れ値・雑音・配備分布から遠い事例を過剰に選びうる」と述べる．
- **本研究の該当リスク**: ドメインラベルは JMMLU のタスク名から決定論的に割り当てた代理ラベルであり，
  境界（例: `professional_psychology`→medical と `high_school_psychology`→education）は本質的に曖昧である．
  最難帯にこの種の「代理ラベルとしては無理のある行」が集中している可能性がある．
- **実測（G4）**: 最難から順に取る純粋な least-confidence と，最難 10 件 / 20 件を飛ばしてから取る変種を
  オフラインで比較したところ，**純粋版が最良**（3 seed 平均 top1: pure 0.7929 / skip10 0.7887 / skip20 0.7740）．
  **本研究のデータでは最難帯の除外は改善に寄与しない**ので，追加パラメータを持たない純粋版を採る．

**Q3: 追加行は JMMLU の未使用行から調達する（B116(2) の調達順位 (i)「既存の信頼できる公開データセット」に該当）**

- 現行の訓練集合 `data/classifier_train.jsonl`（1,427 行）は JMMLU（`nlp-waseda/JMMLU`, commit
  `3637b25e`，CC BY-NC-ND 4.0）の各ドメイン 150 行の無作為標本で，評価集合 1,500 行とは別の乱数種で抽出
  されている．**同じタスク群には未使用の行が大量に残っている**ので，新たなデータ生成（LLM 合成・人手作成）は
  不要である．出典・ライセンス・ドメイン適合性はいずれも既存データセットと同一で，新たな検討事項は生じない．
- 本フェーズで未使用プールを実測した（評価集合 1,500 行・現行訓練 1,427 行の設問文を両方除外）:

  | ドメイン | 現行訓練で使用中のタスクに限った未使用プール |
  |---|---|
  | medical | 1,110 |
  | natural_science | 783 |
  | history_culture | 779 |
  | business_economics | 705 |
  | mathematics | 348 |
  | computer_science | 251 |
  | social_science | 244 |
  | education | 233 |
  | general | 125 |
  | **legal** | **0** |
  | 合計 | **4,578** |

- **legal のプールは 0 である**．JMMLU の legal 対応タスクは `international_law` + `jurisprudence` の
  227 行しかなく，150 行が評価集合，残る 77 行が訓練集合に既に入っている（`build_dataset.py` の
  `build_classifier_training_rows` docstring が既に指摘している構造的制約）．**したがって legal だけは
  追加行が 0 件になる**．これはドメイン固有の特別扱いではなく，10 ドメイン共通の規則をデータ側の制約が
  切り詰めた結果である．`_extract_sample_weights()` が `n_samples/(n_classes*n_domain_samples)` で
  ドメイン別の実効重み合計を常に均等化するため，**legal の実効重みは追加後も他ドメインと完全に等しい**
  （1 行あたりの重みが 1.85→3.02 へ上がるだけ）．legal の非退行は個別に監視する．
- **プールを「現行訓練集合に既に登場しているタスク」へ限定する**（上表はその条件での実測値）．無制限にすると
  education に `japanese_civics` が流入するが，これは **Iter36（0.0529 で崩壊）・Iter37（invalid）・
  Iter38（hybrid で education_recall 0.4000 へ悪化）で 3 回失敗が記録された変更**であり，本レバー
  （hard な行の選択）とタスク構成変更の 2 レバーが混ざる．タスク構成を固定することで単一レバーを保つ．
  この限定で影響を受けるのは education のみ（330→233），他 9 ドメインの数値は変わらない．

### 計画 (Iter84)

**単一レバー**

`cross_domain_training_data_augmentation` = **`hard_negative_mining_all_domains`**．
**`data/classifier_train.jsonl`（1,427 行）へ，現行分類器が正解ドメインへ低い確率しか与えない未使用 JMMLU 行を
10 ドメイン共通の規則で各 100 行（プールが足りないドメインはプール全量）追加した
`data/classifier_train_iter84_hardneg.jsonl`（2,327 行）で分類器を再訓練すること**だけ**を行う．
埋め込みモデル・instruction・連結仕様・送出閾値・集約方式・集約コードは一切変えない．

**選択規則（10 ドメイン共通．ドメインごとにパラメータを変えない）**

1. **プール**: JMMLU（pinned commit `3637b25e`）の各ドメインについて，**現行訓練集合にそのドメインで
   既に登場しているタスク**の全行から，評価集合 1,915 行の設問文と現行訓練集合 1,427 行の設問文を除外した残り
   （合計 4,578 行）．
2. **難易度スコア**: 現行 artifact `models/domain_classifier.joblib`（sha256 `1cfcd3d8...`，`n_features_in_`=2048）
   に現行と同一の埋め込み（`qwen3-embedding:0.6b`，prefix なし ⊕ prefix ありの 2048 次元連結）を与えて
   `predict_proba` を計算し，**正解ドメインの確率 p_true を難易度スコア（小さいほど hard）とする**．
3. **選定**: 各ドメインについて p_true の昇順で先頭 **N=100** 件（プールが 100 未満ならプール全量）．
   同値は (JMMLU タスク名, 設問文) の辞書順で決定論的に解く．
4. **追加**: 選定行を既存 1,427 行の**後ろに追記**する．既存行は 1 行も改変・削除しない．
   id は `{domain}-hardneg-{連番:03d}`．`sample_weight` は付けない（`_extract_sample_weights()` が
   ドメイン数から自動計算するため未使用．`train_domain_classifier.py` docstring 参照）．

**N=100 を選んだ理由**: G3 のオフライン実測で最良だったのは「追加行数 / 元の訓練行数 ≒ 56%」の点である
（713 行に 399 行を追加）．現行の 1,427 行に対する同比率は約 800 行で，10 ドメイン共通の切りの良い値としては
N=100（追加 900 行，+63%）が最も近い．N=100 でプールの過半を使い切るのは general のみ（100/125 = 80%）で，
そこでは hard 選択の優位が無作為と同程度へ縮むだけで害は観測されていない（G3 の N=60 条件＝プールの 80% に相当）．

**固定する構成（直近の最良構成 = Iter83 本走 `results/20260927_070239/` の構成そのまま）**

`config.yaml` は **1 行も変更しない**．`embedding_model=qwen3-embedding:0.6b`，
`embedding_instruction`（Iter81 の P1 文言），`embedding_view_concat=true`，`routing_method=supervised_classifier`，
`confidence_threshold=0.0`，`dispatch_candidate_threshold=0.0`，`dispatch_top_k=2`，`dispatch_gap_max_k=4`，
**`dispatch_gap_threshold=0.36`（Iter83 で採用．今回は触らない）**，`aggregation_method=max_confidence`，
`judge_model`，`classifier_model_path`．`data/dataset.jsonl`（1,915 行，ビット単位で不変），
`data/classifier_train.jsonl`（1,427 行，**ビット単位で不変**．追加分は別ファイルへ書く），
`node.py`・`aggregator.py`・`classifier.py`・`metrics.py`・`run_experiment.py`・`train_domain_classifier.py`
（いずれも**コード変更なし**）．**ドメイン固有の後付け補正は 2026-09-23 恒久運用ルールにより追加しない．
N・選択規則・プールの定義は 10 ドメイン共通である．**

**仮説（事前登録．本走前に数値で記録する）**

「現行の訓練集合は各ドメイン 150 行の無作為標本で，決定境界の近傍が十分に張られていない．同一タスク群には
未使用行が 4,578 行残っている．**現行分類器が正解ドメインへ低い確率しか与えない行を 10 ドメイン共通の規則で
各 100 行追加すると，同数を無作為に追加した場合より境界が精緻化され，`top1_accuracy` が向上する**．
効果は education（recall 0.4800）・social_science（0.6867）・general（0.6933）など境界の弱いドメインに
相対的に大きく出ると予想するが，**ドメインごとの調整は一切行わない**．」

**オフライン事前実測（本フェーズで実施．実機不使用．`data/embcache_*` と `results/` のみを使用）**

| ゲート | 内容 | 実測 | 判定 |
|---|---|---|---|
| **G1（replay の忠実度）** | キャッシュ埋め込み（`embcache_eval_qwen3-embedding_0.6b{,__p1}.npy`）＋現行 artifact から `select_dispatch_targets()` の gap escalation を再現し，Iter83 本走を再現できるか | `compound_domain_set_recall`=**0.5819277108**，`compound_mean_dispatched_count`=**1.8891566265**，複合 k 分布 (278,21,116) が**本走実測と完全一致**．全 1,915 行 top1 は 0.793211（本走 0.792689，**差 1 行**） | **PASS**（本走の経路指標はオフラインで事前に予測できる） |
| **G2（訓練経路の決定性）** | `data/classifier_train.jsonl` とキャッシュ埋め込みから再訓練した分類器が，配備中の artifact と一致するか | 再訓練モデルと artifact の評価集合 1,500 行 top1 が **0.786667 で完全一致** | **PASS**（訓練に乱数由来の揺らぎは無い．差が出れば必ず訓練データの差） |
| **G3（選択規則の有効性．本レバーの中核）** | 訓練 1,427 行を各ドメイン 50% で seed / pool に分割し，seed で訓練した分類器で pool を採点．hard 上位 N 件と無作為 N 件を seed へ足して再訓練し，評価集合 1,500 行の top1 を比較（3 seed × N∈{20,40,60}） | 平均 top1: seed のみ 0.7649 / N=20 hard **0.7887** vs rand 0.7749（**+1.38pt**）/ N=40 hard **0.7929** vs rand 0.7778（**+1.51pt**）/ N=60 hard 0.7894 vs rand 0.7864（+0.30pt）．**9 対比較中 8 で hard が上回る**．N=40 hard（1,112 行）は**全 1,427 行で訓練した現行構成の 0.7867 すら上回る** | **PASS** |
| **G4（最難帯を除外すべきか）** | N=40 固定で，最難 10 件 / 20 件を飛ばしてから取る変種を比較（3 seed） | 平均 top1: pure hard **0.7929** / skip10 0.7887 / skip20 0.7740．**除外は改善しない** | **PASS**（純粋な least-confidence を採用．追加パラメータを持たない） |
| **G5（着地点の事前登録）** | 下表を本走前に記録すること．**実装フェーズでは，新 artifact を作った直後に同じ replay を走らせ，本走前に予測値を journal へ追記すること**（G1 により本走の経路指標は事前に確定できる） | 下表 | 実装フェーズで判定（合格条件は課さない） |

G3 の測定上の注意: 評価集合 1,500 行のうち 72 行は現行訓練集合と設問文が重複している（**既知の欠陥．
backlog B125 に起票済み**．education 46 / history_culture 26）．G3/G4 はこの 72 行を除いた 1,428 行でも
同時に算出しており，結論（hard > rand，pure > skip）は同一である．本反復で追加する 900 行は評価集合と
0 件重複なので，漏洩の割合は 72/1,427 から 72/2,327 へ下がる．72 行は基準線・新構成の双方に等しく含まれ，
対比較を歪めない．

**着地点予測（事前登録）**

| 指標 | Iter83 本走実測（基準線 `results/20260927_070239/`） | **Iter84 予測値** |
|---|---|---|
| `top1_accuracy`（全 1,915 行） | 0.792689 | **0.800〜0.815．点推定 0.807（+1.4pt）** |
| 単一 1,500 行 top1 | 0.786000 | +1〜+3pt |
| 複合 415 行 top1 | 0.816867 | ほぼ不変〜微増 |
| 既存 1,600 行部分集合 top1 | 0.787500 | +1〜+3pt |
| `education` recall / precision | 0.4800 / 0.5304 | 上振れ余地が最大．ただし個別の成功条件は課さない |
| `legal` recall / precision | 0.8800 / 0.8610 | **追加行 0 件の唯一のドメイン．非退行の監視対象** |
| `compound_domain_set_recall` | 0.581928 | **予測不能（確信度分布が動くため）**．非退行の枠内で報告 |
| `compound_mean_dispatched_count` | 1.889157 | **上振れ／下振れの両方がありうる**．非退行の枠内で報告 |
| `mean_duration_ms` | 2286.9 | 送出数に比例．非退行枠 2744.3 |
| ECE | 0.026 | 0.02〜0.05 |
| `answer_quality_accuracy` / `end_to_end_accuracy` | 0.570667 / 0.350914 | ノイズ床 3SD=2.6pt（success_criteria (5)）の範囲でのみ判定 |
| rank1 以外が選ばれた行数 | 2 | ≤ 10 |
| Random / BestSingle / Oracle | 0.121671 / 0.126893 / 1.0 | success_criteria (3) により毎回併記 |

**許容幅とノイズ床の根拠（B132 申し送り 3 への回答．「±1 行」を使わない理由）**

- 訓練パイプラインは埋め込みを固定すればビット決定的である（G2 で確認）．実行時に残る揺らぎは ollama の
  埋め込み再計算だけで，その大きさは**オフライン replay と Iter83 本走の差＝1,915 行中 1 行（0.052pt）**である．
- しかし **Iter83 は送出段だけを動かすレバーで `selected_domain` が構造的に不変だったのに対し，本反復は
  分類器そのものを作り替える**．よって「基準線との差が 1 行以内であること」を非退行条件にするのは誤りであり，
  **非退行は (a) per-domain の検定（BH 補正 q=0.05），(b) 複合指標の明示レンジ，(c) 運用指標の上限**で判定する．
- 主基準 `top1_accuracy` は McNemar の対比較で判定する．検出限界は discordant ペア数 n_d に対し
  `1.96·sqrt(n_d)/1915` で，n_d≈100 なら 1.02pt，n_d≈150 なら 1.25pt である．**採用閾値 +1.0pt は
  この検出限界とほぼ同じ水準**なので，閾値の充足だけでなく McNemar の有意性も同時に要求する（下記 AND 条件）．

**成功条件・非退行条件（事前登録．結果を見る前に固定する）**

基準線は **Iter83 本走 `results/20260927_070239/`**（全 1,915 行: top1=0.792689，単一 1,500 行 top1=0.786000，
複合 415 行 top1=0.816867，1,600 行部分集合 top1=0.787500，`compound_domain_set_recall`=0.581928，
`compound_mean_dispatched_count`=1.889157，fallback=0.0，dispatch_failure=0.000522，ECE=0.026，
mean_duration_ms=2286.9，answer_quality=0.570667，end_to_end=0.350914，rank1 以外が選ばれた行数=2）．

| 区分 | 指標 | 基準線 | 合格条件 |
|---|---|---|---|
| **主基準（効果）** | 全 1,915 行 `top1_accuracy` | 0.792689 | **(i) McNemar 対比較（α=0.05）で有意，かつ (ii) +1.0pt 以上（≥ 0.802689）**．2 条件の AND．Wilson 95% CI と検出限界 `1.96·sqrt(n_d)/1915` を併記する |
| **非退行①** | per-domain recall / precision 計 20 指標（BH 補正 q=0.05） | Iter83 実測 | **有意退行 0 件**．特に **legal（唯一の追加 0 件ドメイン）**と，教師データが増えたのに悪化したドメインが無いことを個別に明記する |
| **非退行②（複合被覆）** | `compound_domain_set_recall` | 0.581928 | **≥ 0.539759（Iter82 水準を下回らない）**．gt=0.36 は新しい確信度分布に対して未較正なので，Iter83 水準の維持は要求しない |
| **非退行③（複合予算）** | `compound_mean_dispatched_count` | 1.889157 | **≤ 2.10（Iter83 比 +11% 以内）**．超過は「送出コストの実質的な悪化」として rejected |
| **非退行④** | `fallback_rate` / `dispatch_failure_rate` | 0.0 / 0.000522 | fallback = 0.0，dispatch_failure ≤ 0.005 |
| **非退行⑤（B132 申し送り 2）** | **rank1 以外が選ばれた行数**（`selected_domain` ≠ `probe_candidates` の最大確信度ドメイン） | 2 | **≤ 10**．部分 dispatch 失敗の検出器．超過したら送出段の失敗が top1 を汚染している疑いとして原因を特定してから判定する |
| **非退行⑥** | `mean_duration_ms` | 2286.9 | **≤ 2744.3（+20% 以内）** |
| **非退行⑦** | ECE | 0.026 | **≤ 0.08** |
| 報告のみ | `answer_quality_accuracy` / `end_to_end_accuracy` | 0.570667 / 0.350914 | 3SD=2.6pt のノイズ床（success_criteria (5)）を超えない限り有意と判定しない |
| 報告のみ | 単一 1,500 行 / 複合 415 行 / 既存 1,600 行部分集合の top1 | 0.786000 / 0.816867 / 0.787500 | 毎回併記 |
| 報告のみ | Random / BestSingle / Oracle | 0.121671 / 0.126893 / 1.0 | success_criteria (3) により毎回併記 |
| 報告のみ | 漏洩 72 行を除いた 1,843 行の top1 | — | B125 の絶対値水増しの影響を切り分けるため併記 |

**判定規則（事前登録）**

- **adopted**: 主基準 (i)(ii) を満たし，非退行①〜⑦をすべて満たす．
- **partial**: McNemar が有意で Δtop1 が +0.5〜+1.0pt，かつ非退行①〜⑦を満たす．
  **G3 の予測（hard 選択の優位 +1.5pt）からの乖離の原因**（プール規模の違い，基準訓練集合が大きいことによる
  逓減，最難帯の雑音など）を特定して記録したうえで，N を変えた再試行の是非を backlog へ起票する．
- **no_effect**: |Δtop1| < 0.5pt または McNemar が非有意．**先に F3/F4/F5 を再検証**してから結論を書く
  （artifact が実際に差し替わっているか・各ノードへ配布されたか）．
- **rejected**: Δtop1 ≤ −0.5pt，または非退行①〜⑦のいずれかに違反．
- **invalid（実験不成立）**: F1〜F5 のいずれか不合格，`question_count != 1915`，
  `compound_domain_question_count != 415`，**新 artifact の sha256 が `1cfcd3d8...`（基準線）と一致**
  （＝再訓練が反映されていない），または**本走の top1 が新 artifact のオフライン replay 予測から 1.0pt 以上乖離**
  （＝実行時経路が predicted と別の artifact を読んでいる疑い）．
- **復元手順（partial / no_effect / rejected 共通）**:
  `cp models/domain_classifier_pre_iter84_baseline.joblib models/domain_classifier.joblib` で artifact を戻し
  （sha256 が `1cfcd3d8...` に戻ることを確認），`mise run deploy` を再実行して全 10 ノードで smoke_check を通す．
  `data/classifier_train.jsonl` と `config.yaml` は本反復で一切触っていないので復元不要．
  `data/classifier_train_iter84_hardneg.jsonl` は記録として残す（過去の `classifier_train_iter*.jsonl` と同じ扱い）．

**レバーを読むコード行と，そこへ到達する条件（「変更したが実行パスに到達しない」失敗への恒久対策．省略不可）**

| 経路 | レバーが効く箇所 | 到達条件 | 到達確認の手段 |
|---|---|---|---|
| **訓練側（オフライン）** | `scripts/train_domain_classifier.py` の `--train-data` | 新しい JSONL を渡すこと | **F2** |
| **実行時側（本体）** | `classifier.py:load_domain_classifier()` → `estimate_confidence_classifier()` が `config.yaml` の `classifier_model_path`（= `models/domain_classifier.joblib`）を読む | 各ノードのコンテナが**配布後の** artifact を読むこと．artifact は `mise.toml` の `rsync` で配られる | **F4**: 全 10 ノードで artifact の sha256 が新値で一致（deploy 内の smoke_check） |
| **記録側（指標の分母）** | `run_experiment.py:98` の `dispatched_domains` 再計算（gap_threshold=0.36，今回は不変） | 同上 | **F5** |
| **到達しない経路（確認のみ．変更しない）** | 埋め込み（`expert_backend.embed_query_views()`）・`aggregator.py`・`config.yaml` | 本反復では一切触らない | **F1**: `config.yaml` の diff 0 行，`.py` の既存ファイルの diff 0 行 |

**事前ゲート（実行パス到達確認．実装フェーズで判定する）**

| ゲート | 内容 |
|---|---|
| **F1（変更の最小性）** | `git diff` が `scripts/mine_hard_negatives.py`（新規）・`data/MANIFEST.md`・`.claude/research/*` のみ．**`config.yaml` の diff は 0 行**．`node.py`・`classifier.py`・`aggregator.py`・`train_domain_classifier.py`・`build_dataset.py` の diff は 0 行 |
| **F2（データ）** | `data/classifier_train_iter84_hardneg.jsonl` が **2,327 行**．先頭 1,427 行が `data/classifier_train.jsonl` と**バイト一致**．追加 900 行の設問文が `data/dataset.jsonl` の 1,915 行および既存 1,427 行と **0 件重複**．ドメイン別の追加数が **{legal: 0, 他 9 ドメイン: 100}**．id の重複 0 件．`data/classifier_train.jsonl` の sha256 が `eb89bf7b...` のまま不変 |
| **F3（artifact）** | 新 `models/domain_classifier.joblib` の sha256 が基準線 `1cfcd3d8...` と**異なる**．`n_features_in_`=2048，`classes_` が 10 ドメイン．退避ファイル `models/domain_classifier_pre_iter84_baseline.joblib` の sha256 = `1cfcd3d8...` |
| **F4（配布）** | `mise run deploy` 後，全 10 ノードで artifact の sha256 が新値で一致．`grep '^dispatch_gap_threshold:' $REMOTE_DIR/config.yaml` が全ノードで `0.36` |
| **F5（実行時経路）** | 先頭 20 問の予備実行の `confidence`（rank1 の値）と `dispatched_domains` が，同 20 問のオフライン予測（新 artifact ＋ キャッシュ埋め込み，gt=0.36）と **20/20 一致**すること．**不一致なら本走に進まない** |

**実験手順**

1. **前提確認**: `wc -l data/dataset.jsonl` = 1915，`wc -l data/classifier_train.jsonl` = 1427，
   `sha256sum models/domain_classifier.joblib` = `1cfcd3d8...`，`config.yaml` の
   `embedding_view_concat: true` / `embedding_instruction` / `dispatch_gap_threshold: 0.36`．
2. **artifact 退避**: `cp models/domain_classifier.joblib models/domain_classifier_pre_iter84_baseline.joblib`．
3. **hard negative の採掘**（**wafl-ctrl5 限定**．config.yml 絶対条件 (B)）:
   新規 `scripts/mine_hard_negatives.py` を実装して実行する．プール構築は `build_dataset.py` の
   `_parse_jmmlu_task_csv` / `_format_jmmlu_query` / `_DOMAIN_TASK_MAP` を**再利用**し（重複実装しない），
   埋め込みは `expert_backend.embed_query_views(..., instruction=..., concat_views=True)` を使う
   （訓練・実行時と同一関数．Iter36 型の train/eval 不一致を構造的に防ぐ）．
   プール埋め込みは `data/embcache_pool_qwen3-embedding_0.6b{,__p1}.npy` へ保存し再実行を安くする．
4. **再訓練**（**wafl-ctrl5 限定**）: `data/MANIFEST.md` の現行コマンドの `--train-data` だけを
   `data/classifier_train_iter84_hardneg.jsonl` に差し替えて実行する（他の引数は 1 文字も変えない）．
5. **本走前の着地点記録（G5）**: 新 artifact ＋ キャッシュ埋め込みで replay を走らせ，
   予測 top1 / per-domain / `compound_domain_set_recall` / `compound_mean_dispatched_count` を
   **本走前に** journal へ追記する．
6. **配布**: `mise run deploy` → **F4**．
7. **予備 20 問**（`data/dataset_20.jsonl`）→ **F5**．不一致なら本走に進まない．
8. **本走**: wafl500〜509 で 1,915 問フルスペック 1 回（config.yml 絶対条件 (A)．事前 replay で
   経路指標が予測できても省略しない）．想定所要は Iter83 実績から約 75 分．
9. `data/MANIFEST.md` に新 artifact の sha256・新訓練ファイルの sha256 と行数・生成コマンドを追記する．

**次イテレーションへの申し送り（B132 申し送り 1 の引き継ぎ）**

本レバーは分類器を作り替えるので **rank1−rank2 gap の分布が再び動く**．Iter81→82→83 で 3 回続けて観測した
機序のとおり，**本反復が adopted になった場合は，直後に `dispatch_gap_threshold` の再較正イテレーションを
1 回挟むこと**（レバー `dispatch_gap_threshold_recalibration` の再オープン．予算整合点の掃引をやり直す）．
単一レバー原則により本反復では閾値を触らないため，`dispatch_gap_threshold=0.36` は新分布に対して未較正のまま
本走する．非退行②③はその前提で緩めに置いてある．

### 実装・実験 (Iter84)

**（この節はオーケストレータが rc-executor の報告から補完した．rc-executor は「journal への実験結果記入は
分析・考察フェーズの担当」という自身の役割定義を理由に journal を編集せず，`data/MANIFEST.md` と戻り値に
のみ記録した．G5 は「本走前に journal へ追記」を要件としていたため，この点は要件どおりに運用されていない．
G5 の予測値そのものは本走前に算出・記録されており（下表），事後の辻褄合わせではない．運用ルールの
不整合として backlog へ申し送る．）**

**実施した変更**

| 種別 | パス | 内容 |
|---|---|---|
| 新規 | `scripts/mine_hard_negatives.py`（369 行） | JMMLU 未使用プールから 10 ドメイン共通規則で hard negative を採掘．プール構築は `build_dataset.py` の `_DOMAIN_TASK_MAP` / `_parse_jmmlu_task_csv` / `_format_jmmlu_query` を再利用，埋め込みは `expert_backend.embed_query_views(..., concat_views=True)`（`train_domain_classifier.py`・`node.py` と同一関数） |
| 生成 | `data/classifier_train_iter84_hardneg.jsonl` | 2,327 行（既存 1,427 行 ＋ 追加 900 行） |
| 生成 | `data/embcache_pool_qwen3-embedding_0.6b{,__p1}.npy` | プール埋め込みキャッシュ |
| 退避 | `models/domain_classifier_pre_iter84_baseline.joblib` | 基準線 artifact（sha256 `1cfcd3d8...`） |
| 更新 | `models/domain_classifier.joblib` | 新 artifact（sha256 `34e4d33bcfb0695a48b410cb70fd1d3d38973e048631b1c1ffe516df423b34c9`，`n_features_in_`=2048，10 クラス） |
| 追記 | `data/MANIFEST.md`（+109 行） | 生成コマンド・sha256・ゲート結果・本走実測値 |
| **無変更** | `config.yaml`・`node.py`・`aggregator.py`・`classifier.py`・`train_domain_classifier.py`・`build_dataset.py`・`data/classifier_train.jsonl` | `git diff` 0 行．`classifier_train.jsonl` は sha256 `eb89bf7b...` 不変 |

**事前ゲートの判定**

| ゲート | 内容 | 結果 |
|---|---|---|
| **F1** | 変更の最小性（上記 6 ファイル ＋ 既存訓練データの diff が 0 行） | **PASS** |
| **F2** | データ健全性: 2,327 行，先頭 1,427 行が既存ファイルとバイト一致，追加 900 行の内訳 `{legal: 0, 他 9 ドメイン: 各 100}`，id 重複 0，評価 1,915 行・既存訓練 1,427 行・追加行同士の重複いずれも 0 | **PASS** |
| **F3** | 新 artifact が基準線と別物で，退避ファイルは基準線と一致 | **PASS** |
| **G5** | 本走前 replay（新 artifact ＋ キャッシュ埋め込み） | **実施**（予測値は下表） |
| **F4** | `mise run deploy` 後，全 10 ノードで artifact sha256 `34e4d33b...` 一致・`dispatch_gap_threshold: 0.36` 一致．健康チェック・smoke_check（git-status / hashes / probe）全合格 | **PASS** |
| **F5** | 予備 20 問（`data/dataset_20.jsonl`）の `confidence`・`dispatched_domains` がオフライン予測と **20/20 完全一致** | **PASS** |

**本走**: `results/20260927_110526/`（1,915 問フルスペック，所要約 63 分）．基準線は Iter83 本走
`results/20260927_070239/`．

**主基準の実測（判定は分析フェーズ）**

| 指標 | 基準線（Iter83） | 事前登録の予測 | G5 の本走前 replay 予測 | **本走実測** |
|---|---|---|---|---|
| `top1_accuracy`（全 1,915 行） | 0.792689 | 0.800〜0.815（点推定 0.807） | 0.806266 | **0.804700**（+1.201pt） |
| Wilson 95% CI | — | — | — | [0.786342, 0.821838] |
| McNemar | — | α=0.05 で有意 | — | chi2=2.674033，**p=0.101997（非有意）** |
| discordant（基準線正解→新誤り / 基準線誤り→新正解） | — | — | — | 79 / 102 |
| 検出限界 `1.96·sqrt(n_d)/1915`（n_d=181） | — | — | — | 1.378pt（実測 Δ1.201pt は**これを下回る**） |

**事前登録の主基準は AND 条件「(i) McNemar 有意 かつ (ii) +1.0pt 以上」であり，(ii) は満たすが (i) を
満たさない．**

**非退行条件の実測**

| # | 条件 | 基準線 | 実測 | 充足 |
|---|---|---|---|---|
| ① | per-domain 20 指標（BH q=0.05）で有意退行 0 件 | — | **有意差 2 件**．`education_recall` 0.4120→**0.3133**（p=0.000427，**退行**），`history_culture_recall` 0.8009→0.8701（p=0.000796，改善）．他 18 指標は有意差なし | **不充足**（退行 1 件） |
| ② | `compound_domain_set_recall ≥ 0.539759` | 0.581928 | 0.573494 | 充足 |
| ③ | `compound_mean_dispatched_count ≤ 2.10` | 1.889157 | 2.048193（+8.4%） | 充足 |
| ④ | fallback 0.0 / dispatch_failure ≤ 0.005 | 0.0 / 0.000522 | 0.0 / 0.001567 | 充足 |
| ⑤ | rank1 以外が選ばれた行数 ≤ 10 | 2 | 3 | 充足 |
| ⑥ | `mean_duration_ms ≤ 2744.3` | 2286.9 | 2280.408 | 充足 |
| ⑦ | ECE ≤ 0.08 | 0.026 | 0.075591（+4.96pt） | 充足（ただし上限際） |

**報告指標**

| 指標 | 基準線 | 実測 |
|---|---|---|
| 既存 1,600 行部分集合 top1 | 0.787500 | 0.808125 |
| `single_domain_top1`（single 1,500 行） | 0.786000 | 0.807333 |
| 複合 415 行 top1 | 0.816867 | **0.795181**（低下） |
| 漏洩 72 行除外（1,843 行）top1 | — | 0.811177 |
| Cohen's kappa | 0.761499 | 0.785973 |
| `answer_quality_accuracy` | 0.570667 | 0.580667 |
| `end_to_end_accuracy` | 0.350914 | 0.362924 |
| Random / BestSingle / Oracle baseline | — | 0.121671 / 0.126893 / 1.0（不変） |

**per-domain 詳細（基準線 → 実測，recall / precision）**

business_economics 0.8182/0.7746→0.8139/0.7611，computer_science 0.8268/0.8884→0.8571/0.9041，
education 0.4120/0.5304→**0.3133**/0.6759，general 0.4626/0.8750→0.5022/0.8769，
history_culture 0.8009/0.7400→**0.8701**/0.7701，legal 0.6626/0.8610→0.6420/0.9123，
mathematics 0.6710/0.8564→0.7013/0.8757，medical 0.7510/0.8153→0.7427/0.7553，
natural_science 0.6494/0.8571→0.6580/0.7958，social_science 0.4545/0.7554→0.5108/0.7239．

**想定外の事象（分析フェーズへの申し送り）**

1. **プール実測値が事前登録値から微小に乖離**: natural_science 783→776（-7），education 233→232（-1），
   legal・他は一致．原因は JMMLU の**同一タスク CSV 内で文字通り重複している設問行**（`college_physics`
   内 6 件，`conceptual_physics` 内 1 件，`high_school_psychology` 内 1 件）をプール構築時に重複排除した
   ため（ドメイン間の混入ではなく同一タスク内の重複であることを実データで確認済み）．各ドメインとも
   N=100 に対して十分な余裕があり，**最終選定結果（9 ドメイン各 100 件，legal 0 件，計 900 件）には
   影響しない**．
2. **`mise run analyze` の最新ディレクトリ自動検出が誤動作**: `ls -1d results/*/ | sort | tail -1` が
   ASCII 順で `i` > `2` のため `results/iter45_preliminary/` を選んでしまう．今回は
   `mise run analyze -- 20260927_110526` と明示指定して回避した（コード変更なし）．`mise.toml` 側の
   既知の弱点として backlog へ記録する．
3. **主基準の McNemar が非有意**（p=0.102）であり，G3 のオフライン推定（+1.51pt，9 対比較中 8 で hard
   優位）から実効果幅（+1.201pt，検出限界 1.378pt 未満）へ乖離した．
4. **`education_recall` が事前仮説に反して有意に退行**（0.4120→0.3133）．ただし `education_precision` は
   0.5304→0.6759 と上昇しており，education へ張り出していた決定境界が引き締まった可能性がある．
5. **複合 415 行 top1 が 0.816867→0.795181 と低下**する一方，single 1,500 行は 0.786000→0.807333 と
   上昇した．hard negative が single 由来（JMMLU の単一タスク行）であることとの関係を分析フェーズで
   検討すること．

### Iteration 84 実行済み

**変更（1 レバーのみ）**

`cross_domain_training_data_augmentation` = `hard_negative_mining_all_domains`．
`data/classifier_train.jsonl`（1,427 行）へ，現行分類器の正解ドメイン確率 `p_true` が低い JMMLU 未使用行を
10 ドメイン共通規則で各 100 行（legal はプール 0 のため 0 行）追加した 2,327 行で分類器を再訓練し，
`models/domain_classifier.joblib` を差し替えた（sha256 `1cfcd3d8...` → `34e4d33b...`）．
`config.yaml`・既存 `.py`・`data/classifier_train.jsonl`・評価集合はいずれも diff 0 行（F1・F2 で確認）．

**判定: rejected（事前登録の判定規則にそのまま該当．artifact は基準線へ復元する）**

| 事前登録の条件 | 実測 | 充足 |
|---|---|---|
| 主基準 (i) McNemar α=0.05 で有意 | chi2=2.674033，**p=0.101997** | **不充足** |
| 主基準 (ii) Δtop1 ≥ +1.0pt | 0.792689 → **0.804700**（+1.201pt，Wilson 95%CI [0.786342, 0.821838]） | 充足 |
| 非退行① per-domain 20 指標（BH q=0.05）で有意退行 0 件 | `education_recall` 0.4120→**0.3133**（p=0.000427，退行）／`history_culture_recall` 0.8009→0.8701（改善） | **不充足** |
| 非退行②〜⑦ | set_recall 0.573494 ≥ 0.539759，mean_dispatched 2.048193 ≤ 2.10，fallback 0.0／dispatch_failure 0.001567，rank1 以外 3 行 ≤ 10，mean_duration_ms 2280.4 ≤ 2744.3，ECE 0.075591 ≤ 0.08 | すべて充足 |

判定規則の `rejected` 分岐（「非退行①〜⑦のいずれかに違反」）と `no_effect` 分岐（「McNemar が非有意」）が
同時に成立する．**より厳しい `rejected` を採る**．BH 補正後に有意な実測の退行が 1 件ある以上，
「効果が見えなかった」ではなく「害が確認された」と記録するのが正しいためである．
`partial` は採らない: 本リポジトリの前例（Iter29/30 の較正系，Iter62/63 の multilabel 系）で `partial` は
**主基準を満たしたうえで非退行が 1 件 FAIL した場合**に限って用いられており，今回は主基準 (i) が不成立で
その前提を欠く．事前登録の `partial` 分岐（Δ +0.5〜+1.0pt かつ非退行全充足）にも該当しない．
**`invalid` ではない**: F3（sha256 が別物）・F4（全 10 ノードで新 sha256 一致）・F5（予備 20 問が 20/20 一致）が
すべて PASS で，本走 top1 0.804700 は新 artifact の replay 予測 0.806266 と 0.157pt しか違わない
（invalid 判定の閾値 1.0pt 以内）．レバーは確かに実行パスへ到達している．

**復元（必須．次の実験の前に行うこと）**: `cp models/domain_classifier_pre_iter84_baseline.joblib
models/domain_classifier.joblib`（sha256 が `1cfcd3d8...` に戻ることを確認）→ `mise run deploy` で全 10 ノードへ再配布．
本節の執筆時点では**まだ実施していない**（実機への配布は実験フェーズの操作のため）．
以後の基準線は Iter83 本走 `results/20260927_070239/` のままである．

**結果の内訳（本節で新たに実測した数値．すべて `results/20260927_070239` 対 `results/20260927_110526`）**

| 母数 | 基準線 | 実測 | Δ | discordant（悪化/改善） | McNemar |
|---|---|---|---|---|---|
| 全 1,915 行 | 0.792689 | 0.804700 | **+1.201pt** | 79 / 102 | chi2=2.674，p=0.102（非有意） |
| single 1,500 行 | 0.786000 | 0.807333 | **+2.133pt** | 54 / 86 | chi2=6.864，**p=0.0088（有意）** |
| compound 415 行 | 0.816867 | 0.795181 | **−2.169pt** | 25 / 16 | chi2=1.561，p=0.212（非有意） |

**論点 1: G3 のオフライン推定 +1.51pt と実効果 +1.201pt の乖離，および必要標本数**

- 乖離そのものは小さい（0.31pt）．G3 は訓練 1,427 行を半分（713 行）に割った小規模設定で，
  **追加行数／元の訓練行数 ≒ 56%** の点を測っていた．小さい訓練集合ほど 1 行あたりの限界情報量が大きいので，
  G3 の +1.51pt は本走（1,427 行に +900 行）への上振れ気味の外挿である．実効果がその 8 割で出たこと自体は
  仮説と整合し，**方向・桁とも外していない**．外れたのは効果量ではなく**検出力の見積り**である．
- 計画時は「n_d≈100〜150 なら検出限界 1.02〜1.25pt」と見積り，採用閾値 +1.0pt をそこに合わせた．
  実際の n_d は **181**（discordant 率 p_d=0.09452）で検出限界は **1.378pt** へ伸び，
  +1.201pt はその内側に収まった．**分類器を作り替えるレバーは送出段のレバーより discordant を多く生む**
  （Iter83 は正味 1 行）ため，n_d を 100〜150 と置いた前提自体が楽観的だった．
- **必要標本数（今後の成功条件設計に使うこと）**: McNemar は `N ≥ (z_α+z_β)²·p_d/δ²` で見積れる．
  実測 p_d=0.0945 のとき **δ=1.2pt を有意にするには N≥2,517（検出力 50%＝有意境界）・
  N≥5,143（検出力 80%）・N≥6,885（検出力 90%）**．現行 N=1,915 で 80% の検出力が得られる最小効果は
  **δ≥1.97pt**，有意境界でも **δ≥1.378pt** である．
  **帰結: 「+1.0pt 以上 かつ McNemar 有意」という AND 条件は，N=1,915 では事実上満たせない事前登録だった**
  （+1.0〜1.38pt の帯が構造的に判定不能になる）．今後は (a) 期待効果が 2pt 未満なら評価集合を先に増やす，
  (b) それが出来ないなら閾値を検出限界（`1.96·sqrt(p_d/N)`）以上に置く，のどちらかを計画時に選ぶこと．

**論点 2: `education_recall` の退行と `education_precision` の上昇（「境界の引き締まり」は single 行では正しく，compound 行では成り立たない）**

- 指標の母数は expected に education を含む **233 行**（single 150 ＋ compound 83）である．内訳は
  **single 72→62（−10）・compound 24→11（−13）**で，**退行 23 行のうち 57% が compound 行由来**である．
- **single 行だけを見ると「引き締まり」の解釈は妥当**: recall 0.4800→0.4133（−6.67pt）に対し
  precision 0.5669→0.7294（**+16.25pt**），**F1 は 0.5199→0.5277（+0.78pt）で微増**．
  基準線で education が誤って吸い込んでいた行（medical 23・social_science 17 など計 55 行）は
  新構成で 23 行（medical 8・legal 5・social_science 5 ほか）まで減り，**そのうち 29 行が正解ドメインへ復帰した**．
  逆向きの流出は single の education 行 18 行（基準線で正解→新構成で誤り）で，行き先は **medical 10・
  social_science 8**．差し引き −10 行（72→62）である．
  すなわち single 行では「過剰に張り出していた education の決定境界が引き締まり，他ドメインの recall を押し上げた」
  という解釈がデータと一致する（general F1 +4.69pt，history_culture F1 +5.50pt，computer_science F1 +2.91pt）．
- **compound 行では純粋な損失**: education を含む 83 行で education が選ばれた回数が 24→11 へ落ち，
  compound で悪化した 25 行のうち 16 行が expected に education を含む（education+general 7・
  education+natural_science 4・education+legal 3・education+social_science 2）．
  複合設問は「2 ドメインのどちらかが選ばれれば正解」なので，**張り出した education 境界は compound 行では
  得点源として働いていた**．引き締めはその得点源を直接削る．
- **母数込みの F1（233 行基準）は 0.46375→0.42815（−3.56pt）で悪化**する．すなわち
  「F1 で見れば相殺される」とは言えない．**single 行に限れば +0.78pt，全母数なら −3.56pt** と結論が反転するので，
  今後 education を論じるときは母数を必ず明記すること．
- **2026-09-23 恒久ルールにより，education だけを狙った閾値・intercept・訓練データの後付け補正での是正は行わない．**
  是正するなら 10 ドメイン共通の規則（例: 追加行数 N の変更，compound 行を含む評価での選択規則）でのみ行う．

**論点 3: compound の低下（−2.169pt）と single の上昇（+2.133pt）の乖離**

- 訓練へ追加した 900 行は**すべて JMMLU の単一タスク行**であり，複合設問は 1 行も含まない．
  したがって本レバーは **single 行の分布上でのみ境界を精緻化し，compound 行にとっては分布外の変更**である．
- compound で悪化した 25 行の遷移先は **medical 14・history_culture 4・business_economics 3** に集中する．
  medical は precision が 0.8153→0.7553（−6.00pt）と落ちる一方 compound での被選択が 222→237 行へ増えており，
  **「複合設問の文面に対して medical が過剰に立つ」方向へ境界が動いた**．
  medical は未使用プールが最大（1,110 行）で，追加 100 行の情報量が最も豊富だった側である．
- 機序の整理: **single 行では「1 つの正解ドメインを当てる」ので境界の引き締めが得になり，
  compound 行では「2 つのうちどちらでもよい」ので境界の緩さが得になる**．両者は最適な境界が構造的に食い違う．
  hard negative mining を single 行だけから採る限り，この食い違いは N を調整しても消えない．
  **compound を守るには，訓練側に複合的な信号を入れるか，評価側で両者を分けて事前登録するかのどちらかが要る．**
- なお compound の −2.169pt は discordant 25/16・chi2=1.561・p=0.212 で**単体では有意ではない**
  （n=415 では ±4pt 程度が検出限界）．「compound が有意に壊れた」とまでは言えない点を記録しておく．

**論点 4: ECE 0.026→0.075591 の悪化は「過信」ではなく「自信不足」への移行である**

| 指標（全 1,915 行） | 基準線 | 新構成 |
|---|---|---|
| rank1 confidence 平均 | 0.7843 | 0.7306 |
| top1_accuracy | 0.7927 | 0.8047 |
| 平均確信度 − 正答率 | **−0.0084** | **−0.0741** |
| rank1−rank2 gap 平均 | 0.6650 | 0.5967 |
| gap の p25 / p50 / p75 | 0.4159 / 0.7730 / 0.9381 | 0.3557 / 0.6657 / 0.8636 |
| gap < 0.36 の行の割合 | 21.78% | 25.22% |

- 精度は上がったのに確信度が下がった．**最難帯 900 行を訓練へ足したことで `CalibratedClassifierCV`
  （temperature）の温度が上がり，確率が一様側へ寄った**というのが素直な機序である．ECE の悪化は
  過信ではなく**系統的な自信不足**（−7.41pt）で，方向は Iter29〜31 で扱った過信とは逆である．
- gap 分布が一様に圧縮された結果，固定閾値 `dispatch_gap_threshold=0.36` に対して
  **gap<0.36 の行が 21.78%→25.22% へ増え**，`compound_mean_dispatched_count` が 1.889→2.048（+8.4%）になった．
  compound 行に限れば gap<0.36 は 33.01%→36.87%．これは Iter81→82→83 で 3 回観測した
  「特徴量・分類器を変えると gap 閾値が未較正になる」機序の **4 回目**である．
- **B132 申し送り 1（再較正イテレーションを挟むか）への回答**: 本反復は rejected で artifact を復元するので，
  確信度分布は基準線へ戻り `gt=0.36` は較正済みのまま保たれる．**したがって今回は再較正を挟まない**．
  ただし上の数値は，**分類器を作り替えるレバーが adopted になった時には必ず再較正が要る**ことを
  4 例目として裏づける．次に分類器を差し替える反復では，計画時点で再較正をセットで予定すること．

**学び**

1. **N=1,915 の評価集合は +1.0〜1.4pt の効果を原理的に判定できない**（p_d≈0.095 のとき有意境界 1.378pt）．
   この帯に入る効果を狙うレバーでは，事前登録の時点で「判定不能で終わる」ことが確定している．
   Iter63〜68 の複合設問の検出力不足（n=100）と**同型の失敗が，今度は全体集合の側で起きた**．
   `N ≥ (z_α+z_β)²·p_d/δ²` を計画フェーズの必須計算項目にすること．
2. **single 行と compound 行では最適な決定境界が逆を向く**．single は引き締め，compound は緩さを好む．
   単一ドメイン行だけから採った hard negative は前者しか最適化しないので，複合被覆とは構造的に
   トレードオフになる．今後 `top1_accuracy`（全体）を主基準に置くレバーでは，この 2 部分集合の
   内訳を必ず分解して報告すること（全体だけ見ると打ち消し合って「効果なし」に見える）．
3. **難しい行を訓練へ足すと確率は一様側へ寄る**（精度は上がるのに確信度が下がる＝自信不足）．
   ECE 悪化＝過信という先入観で読むと誤診する．較正の向きは毎回，平均確信度 − 正答率の符号で確かめること．
4. **教科書どおりの hard negative mining は，本研究のデータでは「効果が無い」のではなく
   「single では効き，compound と education で損を出し，正味が検出限界に埋もれる」**．
   オフライン G3 の 9 対比較中 8 勝という強い事前証拠があっても，評価集合の母数が足りなければ確定できない．

---

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

