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

## Iteration 31: 分類器較正のtemperature scaling方式によるargmax不変性の実証とECE目標到達可否の検証

### 調査 (Iter31)

**問い**:
1. 本リポジトリの base estimator（`LogisticRegression(max_iter=1000, class_weight='balanced')`）に
   対して `CalibratedClassifierCV(method='temperature')` を使うと，sklearn は
   `decision_function`（ロジット）を経由するのか，`predict_proba` の対数近似を使うのか．
2. `class_weight='balanced'` は temperature の「argmax 不変」という理論保証を壊す余地があるか．
3. Iter30 で確立した非退行チェック手順（BH補正 q=0.05・recallはドメイン別McNemar・precisionは
   Fisher正確検定・20指標）は temperature でもそのまま再利用可能か．
4. `cv`・`ensemble` パラメータは Platt/isotonic（`cv=5, ensemble=True`）と同じ設定を踏襲すべきか．

#### 分かったこと

**(1) `decision_function` 経路が使われることをソースコードで直接確認 — 一次情報**

本リポジトリの実行環境（`.venv/lib/python3.12/site-packages/sklearn/calibration.py`，
`scikit-learn==1.9.0`）を `Read` で直接確認した。

- `_fit_calibrator()`（687-749行）は `method == "temperature"` の分岐で
  `calibrator = _TemperatureScaling(); calibrator.fit(predictions, y, sample_weight)` を呼ぶ。
  この `predictions` は `CalibratedClassifierCV.fit()` 内で
  `_get_response_values(estimator, X, response_method=["decision_function", "predict_proba"])`
  （`decision_function` を優先するリスト順）から得られており，`LogisticRegression` は
  `decision_function` を実装しているため，**実際に使われるのは
  ロジット（`w^T x + b`，多クラスなので `(n_samples, n_classes)` 形状）そのものであり，
  `predict_proba` の対数近似は使われない**。
- `_TemperatureScaling.fit()`（1077-1181行）の docstring に「If the input appears to be
  probabilities (i.e., values between 0 and 1 that sum to 1 across classes), it will be
  converted to logits using `np.log(p + eps)`」と明記されている。`_convert_to_logits()`
  （954-983行）はこの判定を行うヘルパーだが，`LogisticRegression.decision_function()`
  の出力は確率ではなくロジットそのもの（0-1に収まらず合計も1にならない）ため，この
  変換は発火せず，ロジットがそのまま `raw_prediction = exp(log_beta) * logits` として
  multinomial loss（`HalfMultinomialLoss`）の最小化に使われる（`log_beta` を
  `scipy.optimize.minimize_scalar`で`bounds=(-10.0, 10.0)`の範囲で最適化）。
  `beta_ = exp(log_beta*)` が「逆温度」であり `T = 1/beta_`。

**(2) `class_weight='balanced'` はロジット自体の値には影響するが，temperature の
argmax 不変性を壊す経路にはならない**

`class_weight='balanced'` は `LogisticRegression.fit()` 内の損失関数の重み付けにのみ影響し，
学習後の `decision_function()` は単なる固定の線形写像 `w^T x + b` である。temperature
scaling は，この**固定されたロジットベクトル全体**を単一スカラー `1/T` 倍してから softmax を
取るだけの変換であり，`class_weight` がどうロジットの値そのものを決めたかとは独立に，
「全クラスに同一の正の定数を掛けて softmax を取る操作は argmax を変えない」という数学的事実
（`softmax` は単調変換に対して順序不変）がそのまま成立する。実測でも確認した（下記(3)）。

**(3) 実測検証（`uv run python` での合成データ実験，1427件・legal 77件/他150件という
本リポジトリの訓練データ規模を模した設定）— 新たな重要な留保を発見**

`sklearn.linear_model.LogisticRegression` と `sklearn.calibration.CalibratedClassifierCV` を
実際に合成データ（10ドメイン，legal 77件・他9ドメイン各150件，32次元の埋め込みを模した
乱数特徴量）で fit させ，argmax の一致率を直接計測した。

- **fold内（同一 `(estimator, T)` ペア）での argmax 保持は理論通り厳密に 100%**。
  `ensemble=True` の各 fold について，その fold の base estimator 自身の
  `decision_function` の argmax と，その fold の temperature 較正後 `predict_proba`
  の argmax を比較したところ，5 fold 全てで一致率 1.0（0/1427 不一致）だった。
  sklearn 公式の「T は softmax の argmax の位置に影響しない」という保証は，この
  「単一の (estimator, T) ペア内」という意味で寸分違わず成立している。
- **しかし `ensemble=True` では，本番推論時に使われるのは 5 つの異なる fold
  （80% サブセットで学習した 5 つの異なる `LogisticRegression`，かつ 5 つの異なる T）の
  予測確率の平均であり，全データで学習した単一モデルとの比較では非ゼロの flip が生じる**。
  合成データでの実測: `method='temperature', ensemble=True` で
  全データ学習の単一 base estimator との argmax 不一致 16/1427（1.12%）。
  同条件で `method='sigmoid'` は 55/1427（3.85%），`method='isotonic'` は 60/1427（4.20%）。
  **この非ゼロ flip の原因は temperature の較正曲線の歪みではなく，
  `ensemble=True` 自体が持つバギング的な平均化効果（5 つの異なるサブセットで学習した
  分類器の予測を平均する）であり，sigmoid/isotonic にも共通する構造である**。ただし
  temperature は per-class の曲線歪みという追加の誤差源を持たないため，同じ
  `ensemble=True` 条件でも isotonic/sigmoid よりこの合成データで一貫して小さい。
- **`ensemble=False` にすると，temperature は理論通り厳密に 0% の flip（0/1427）を
  達成した**。この設定では本番推論に使われる base estimator は全データで学習した単一
  モデルであり（サブセット学習の平均化がない），T も CV による out-of-fold 予測から
  1 つだけ学習されて，その単一モデルの固定ロジットに適用される。一方，同じ
  `ensemble=False` でも `sigmoid`（2.87%不一致）・`isotonic`（5.19%不一致）は依然として
  非ゼロだった（OvR 事後正規化由来の歪みは `ensemble` の設定と無関係に残る）。
- sklearn 公式のリリースノート（1.8, `tavily-extract`で直接取得）が示す temperature
  scaling の使用例は `CalibratedClassifierCV(clf, method="temperature", ensemble=False)`
  であり，`ensemble=False` を使うサンプルコードになっている。ただし本リポジトリの
  Iter29（Platt）・Iter30（isotonic）はいずれも `cv=5, ensemble=True` で実施済みであり，
  config.yml のレバー note は「temperatureも同条件を踏襲する」ことを既定の想定として書いている。

**(4) Iter30 の非退行チェック手順（BH補正・McNemar・Fisher）はそのまま再利用可能。
isotonic特有チェックリストは temperature には構造的に該当しない**

- BH補正付き 20 指標非退行チェック（recall=ドメイン別McNemar，precision=Fisher，
  計 20 個の p 値へ BH q=0.05）は手法に依存しない一般的な統計手続きであり，
  temperature でも変更なくそのまま使える。
- Iter30 のisotonic特有チェックリスト（(a) 確率の厳密な0/1張り付き，(b) 全クラス0.1の
  uniform fallback，(c) tie率）は，temperature の実装構造上そもそも発生しない。
  `_CalibratedClassifier.predict_proba()`（833-843行）の `method == "temperature"` 分岐は
  `proba = self.calibrators[0].predict(predictions)` で softmax の出力をそのまま使うため，
  isotonic/sigmoid のような「クラスごとの計算結果を後から正規化し，分母が0ならuniform
  fallbackする」という経路（810-832行）を一切通らない。softmax は有限のロジット入力に対し
  厳密に0や1にはならず（`exp` の値は常に正），tie も理論上は浮動小数点の偶然の一致でしか
  起こらない。したがって temperature ではこの3チェックは「必ず該当なし」になる見込みが高く，
  報告項目として残す価値は低い（形だけ算出して0件であることを確認する程度で十分）。
- Iter30 で判明した `medical_recall` 系統的圧縮（isotonic較正曲線がmedicalクラス固有に
  確率の天井を下げていた）は，temperature が単一スカラーで全ドメイン共通の変換しかしない
  構造上，**再現しないと理論的に予想される**が，逆に言えば temperature は medical だけを
  選択的に補正することもできない。ある特定ドメインの較正だけがずれている場合，temperature
  はそのドメインを狙って直すことはできず，全体の log loss を最小化する単一の T に丸め込む
  （config.yml note の留保どおり）。

**出典**:
- ローカル実行環境の直接確認: `.venv/lib/python3.12/site-packages/sklearn/calibration.py`
  （`_fit_calibrator`687-749行，`_CalibratedClassifier.predict_proba`781-847行，
  `_TemperatureScaling.fit/predict`1068-1230行，`_convert_to_logits`954-983行）を
  `Read`・`grep` で直接確認（2026-07-31実施，一次ソース）。
- `uv run python` での合成データ実測（10ドメイン，legal 77件/他150件を模した規模，
  32次元乱数特徴量，`method in {sigmoid, isotonic, temperature}` × `ensemble in {True, False}`
  の argmax 一致率比較）。本セッションで実施，再現可能。
- https://scikit-learn.org/stable/auto_examples/release_highlights/plot_release_highlights_1_8_0.html
  （`tavily-extract`で直接取得。「Temperature scaling in CalibratedClassifierCV」節，
  `ensemble=False`を使うサンプルコード，「particularly well suited for multiclass problems
  because it provides (better) calibrated probabilities with a single free parameter」の原文）
- https://scikit-learn.org/stable/whats_new/v1.8.html （`tavily-search`。
  「Added temperature scaling method in calibration.CalibratedClassifierCV」の変更履歴，
  Array API対応の追加も1.8で行われたことの確認）
- https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html
  （`tavily-search`。1.9.0時点のAPIリファレンス，`.. versionchanged:: 1.8 Added option
  'temperature'`の再確認）
- journal.md「調査 (Iter30)」節（`method='temperature'`の存在確認・sklearn公式ドキュメントの
  「Tはsoftmaxのargmaxの位置に影響しない」引用は既出のため本イテレーションでは再掲のみ）

#### rc-planner への申し送り

1. **問い1・2は確定的に解消**: `decision_function`（ロジット）経路が使われることをソースコード
   直接確認・実測の両方で確定した。`class_weight='balanced'`はロジットの値を決めるだけで，
   temperatureのargmax不変性の理論保証を壊す経路は存在しない。
2. **問い4（cv/ensemble）に，計画フェーズで判断すべき新しい論点が生じた**。
   `ensemble=True`（Iter29/30と同一設定）を維持すると，temperatureでも
   ensemble平均化に起因する非ゼロのflip（合成データで1.12%，isotonic4.20%・sigmoid3.85%より
   小さいが0ではない）が生じることが判明した。これは「T はsoftmaxのargmaxを変えない」という
   sklearn公式の理論保証が，**個々の(estimator, T)ペア内では厳密に成立するが，
   `ensemble=True`によって生成される最終的な推論モデル（5ペアの平均）全体には及ばない**
   ことを意味する。したがって「temperatureはtop1_accuracy不変が理論的に保証される」という
   config.yml note・backlog B51の記述は，`ensemble=True`を維持する場合は**厳密には
   正確ではなく**，「isotonic/sigmoidより小さいが，ゼロではないflipが生じうる」と
   修正して計画に反映すべきである。
   - 選択肢 A（推奨）: `cv=5, ensemble=True`をIter29/30と完全に同一のまま維持する。
     単一レバー原則を「較正手法のみ」に厳密に限定でき，Iter29/30との直接比較可能性が
     最大になる。ただし成功条件（top1_accuracy非退行・per-domain非退行）の判定基準文言に
     「temperatureの理論的argmax不変性は個々の fold ペア内の話であり，ensemble平均化に
     起因する小さな非ゼロflipは想定内である」旨を明記し，isotonicのような「クラス固有の
     曲線歪み」由来の系統的退行（medical_recallのような）とは区別して解釈する必要がある。
   - 選択肢 B: `ensemble=False`に変更する（sklearn公式のリリースノートのサンプルコードが
     使う設定）。この場合argmax不変性が文字通り厳密に成立する（合成データで実測0%）。
     ただしIter29/30とは`ensemble`パラメータ自体が異なるため，較正手法とensembleの
     2変数が同時に変わることになり，単一レバー原則の厳密な適用としては説明が要る
     （「temperatureは1パラメータのみの学習で過学習リスクが本質的に低いため，
     isotonic/plattで必要だったensemble平均化によるロバスト化が不要」という理屈は立つが，
     計画書で明示的に正当化すること）。
   いずれを選ぶにせよ，計画(Iter31)節に「今回のcv/ensemble設定と，Iter29/30との比較可能性
   への影響」を明記すること。
3. **非退行チェック手順はIter30のBH補正付き20指標チェック（recall=McNemar，precision=Fisher，
   BH q=0.05）をそのまま流用してよい**。isotonic特有の3項目チェックリスト（0/1張り付き・
   uniform fallback・tie率）はtemperatureでは構造的に該当なしと予想されるため，
   簡略化して「該当0件であることの確認」程度に留めてよい（実験時間の節約になる）。
4. **medical_recall問題がtemperatureで再現するかどうかは，Y4全体の結論に関わる重要な
   観察点である**。isotonicのmedical系統的圧縮が「OvR方式のクラス固有曲線歪み」に起因する
   という Iter30 の結論が正しければ，単一Tしか使わないtemperatureではこの種の系統的圧縮は
   原理上起こらないはずである。もしtemperatureでも同様の非退行違反が起きた場合，
   「OvR方式由来ではない別の根本原因（例えばmedicalクラス自体の埋め込み分離の弱さ）」を
   疑う材料になる。
5. **ECE改善幅がplatt（0.16751）・isotonic（0.121424）に届かない可能性は，config.yml note
   どおり留保として残る**。単一Tでは表現力がisotonic/plattより低いため，目標0.150に届かない
   （platt同様partial）シナリオも十分あり得る。この場合は「per-domain非退行のためには
   OvR方式の柔軟性を犠牲にできない」という新知見が得られ，次の一手（isotonicの運用調整，
   例えばmedicalドメインのみ較正を無効化する等）を検討する材料になる（backlog B51要レビュー
   (1)がすでに示唆済み）。

---

### 計画 (Iter31)

**仮説**: `scripts/train_domain_classifier.py:train_classifier()` の較正手法を `method="isotonic"`
（Iter30，partial：ECE目標達成もmedical_recallがBH補正後有意悪化）から `method="temperature"`
へ切り替えると，単一スカラーTでロジット全体を変換する構造上，isotonic/plattのOvR方式由来の
クラス固有曲線歪み（medical_recall悪化の疑わしい原因，Iter30考察）が構造的に排除され，
per-domain非退行が成立する。一方，temperatureは表現力がisotonic/plattより低いため，ECE改善幅が
isotonic（0.121424）はもとよりplatt（0.16751，目標未達実績）にも届かず，目標0.150未達となる
可能性が留保として残る（config.yml note・backlog B51要レビュー(1)）。

**単一レバー**: `classifier_calibration`（`.claude/research/config.yml` のレバー，150行目）。
今回試す値は `values: [platt, isotonic, temperature]` のうち **`temperature` のみ**
（backlog B51の自動選択）。これで config.yml 登録済みの3値をすべて試したことになる。

**cv/ensemble設定の決定（調査(Iter31)申し送り2の選択肢A/Bのいずれかを選ぶ）**:

**選択肢A（`cv=5, ensemble=True`，Iter29/30と完全同一）を採用する**。理由:

1. 単一レバー原則を「較正手法のみ」に厳密に限定できる。`cv`・`ensemble`を較正手法と同時に
   変えると2変数が同時に動き，ECE・flip rate・per-domain結果の変化が較正手法の違い由来か
   ensemble設定の違い由来か切り分けられなくなる。今回のY4の核心的な問いは「isotonic/plattとの
   直接比較の下でtemperatureがmedical_recall問題を回避しつつECE目標に届くか」であり，
   Iter29・Iter30との比較可能性の維持そのものがこのイテレーションの価値の大部分を占める。
2. 調査(Iter31)の実測で明らかになった`ensemble=True`由来の非ゼロflip（合成データで1.12%）は，
   5 fold間のバギング的平均化という手法非依存の一般的機序に起因し，isotonic/plattが抱える
   「OvR方式のクラス固有曲線歪み」（medical_recall悪化の疑わしい原因）とは異なる機序である。
   したがってこの程度のflipは，medical_recallのような系統的・ドメイン固有の退行の温床には
   ならないと考えられ，「temperatureはOvR由来のクラス固有歪みを構造的に持たない」という
   本イテレーションが検証したい理論的主張の意義を損なわない。
3. sklearn公式リリースノートの`ensemble=False`サンプルコードは「temperatureの使い方の一例」に
   過ぎず，本リポジトリがIter29・Iter30で確立した比較条件を犠牲にしてまで踏襲すべき規範とは
   判断しない。
4. 選択肢B（`ensemble=False`）を取らない理由: 較正手法とensembleの2変数が同時に変わり，
   「temperatureが優れているのか，ensemble平均化を止めたことが効いたのか」を切り分けられなく
   なる。仮にtemperatureがper-domain非退行を達成しても，isotonic/plattでも`ensemble=False`に
   すれば同様に改善した可能性を排除できず，Y4全体の結論（較正手法としてのtemperatureの優位性）
   が弱まる。

**成功条件の解釈への反映（調査(Iter31)申し送り2が要求した文言）**: temperatureの理論的argmax
不変性（sklearn公式が保証する「Tはsoftmaxのargmaxの位置に影響しない」）は，個々の
`(estimator, T)` ペア内で厳密に成立する事実であり，`ensemble=True`による5fold平均化に起因する
小さな非ゼロflip（合成データ実測1.12%，isotonic4.20%・sigmoid3.85%より小さいが0ではない）は，
この理論保証が主張する範囲の外側にある，想定内の挙動として扱う。したがって今回の実測で
flip_rateが完全に0%でないこと自体は失敗ではない。成功条件2・3（下記）の判定はあくまで
統計的検定（McNemar／Fisher／BH補正）の結果で行い，「flipが0でないから理論違反」という
短絡的な解釈はしない。

**固定する構成（Iter29/30と完全に同一，`config.yaml`は一切変更しない）**:
`routing_method=supervised_classifier`，`confidence_threshold=0.0`・`dispatch_top_k=1`・
`aggregation_method=max_confidence`（Iter28 adopted構成），`confidence_signal_method=self_report`，
`confidence_elicitation=top_k_with_probs`，`expert_model=expert-mesh-{domain}-lora`
（domain_count=10），`light_model=qwen3.5:4b-q4_K_M`，`embedding_model=nomic-embed-text`，
評価データセットはIter25以降固定の1600問（`data/dataset.jsonl`）。訓練データも
`data/classifier_train.jsonl`（1427件，legal 77件・他9ドメイン各150件）でIter29/30と同一。
`CalibratedClassifierCV`の`cv=5`・`ensemble=True`もIter29/30と同一（上記選択肢A）。
**今回変更するのは`train_classifier()`内の較正手法（`_CALIBRATION_METHOD`定数の値）のみであり，
`config.yaml`のキーは1つも変えない。**

**変更ファイル・行**:

1. `scripts/train_domain_classifier.py`
   - `_CALIBRATION_METHOD = "isotonic"`（64行）→ `"temperature"`に変更。`_CALIBRATION_CV = 5`
     （65行）は無変更。`ensemble=True`（`train_classifier()`117-120行の
     `CalibratedClassifierCV(...)`呼び出しにハードコード）は無変更のまま維持する（上記
     選択肢Aの実装上の反映箇所，これ自体は変更しない）。
   - `_CALIBRATION_METHOD`直上のコメント（45-63行付近，isotonicを選んだ理由の説明）を，
     temperatureを選ぶ理由（Iter30のisotonicがmedical_recallのBH補正後有意悪化でpartial判定・
     backlog B51の自動選択）と，調査(Iter31)が確認した構造上の利点（単一スカラーTでロジット
     全体を変換するためクラスごとの個別較正器を持たず，OvR方式由来のクラス固有曲線歪みを
     構造的に排除する）に更新する。cv/ensembleを維持する理由（比較可能性優先，上記選択肢Aの
     要約）も一言追記する。
   - モジュール冒頭docstring（1-5行）の`Iter30, classifier_calibration=isotonic`を
     `Iter31, classifier_calibration=temperature`に更新する。
   - `train_classifier()`のdocstring（95-116行付近）を`method="temperature"`の説明に更新する。
     isotonic特有の注意点（tie・0/1張り付き・uniform fallback）への言及は，調査(Iter31)
     分かったこと(4)よりtemperatureの実装構造上「該当しない」旨に置き換える。
   - 出力アーティファクト名は`models/domain_classifier_temperature.joblib`（isotonic版
     `models/domain_classifier_isotonic.joblib`・platt版
     `models/domain_classifier_platt.joblib`とは別名で新規生成，本番
     `models/domain_classifier.joblib`は上書きしない）。
   - `tests/test_train_domain_classifier.py`はisotonic特有のアサーションを含んでおらず
     （確認済み，`grep`でisotonic/`_CALIBRATION_METHOD`への直接参照なし），`method`の変更が
     `StratifiedKFold`の分割条件に影響しないため無変更で通る見込み（実装フェーズで実行して
     確認する）。

2. `scripts/evaluate_classifier_calibration.py`: **変更不要**。`probabilities`フィールド
   （10ドメイン全ての確率）はIter30で既に追加済みであり，temperature特有チェックリスト
   （下記手順7）にもそのまま使える。

3. `metrics.py`: **変更不要**。`compute_domain_recall_mcnemar_test`・
   `compute_domain_precision_fisher_test`・`apply_benjamini_hochberg`はIter30で実装済みで，
   手法非依存の統計手続きのためそのまま再利用する。

4. `tests/test_metrics.py`: 変更不要（Iter30で追加したテストは手法非依存のため，そのまま
   有効）。

**評価手順（Iter30の手順1-8をそのまま踏襲し，モデル名・出力ファイル名のみ変更）**:

1. 新分類器の学習: `uv run python -m scripts.train_domain_classifier --train-data
   data/classifier_train.jsonl --embedding-model nomic-embed-text --ollama-host <live node>
   --output models/domain_classifier_temperature.joblib`（Iter29/30と同じくライブなollama
   ノード1台へのembeddingのみ）。
2. 「較正前」データはIter29/30と同一の`results/20260731_162722/results.jsonl`（Iter28実測，
   fallback 0/1600）をそのまま使う。**再実行しない**（3イテレーションを同じ較正前基準で
   揃えて比較可能にするため）。
3. 「較正後」データは`scripts/evaluate_classifier_calibration.py`で1600問を再embeddingし，
   `--classifier models/domain_classifier_temperature.joblib --output
   results/iter31_calibrated_predictions.jsonl`として生成する。
4. `metrics.py:compute_ece(n_bins=10)`を較正前・較正後の両方に同一のbin設定で適用し，ECEを
   比較する（較正前基準0.19336はIter29/30から流用，再計算しない）。
5. top1_accuracyを較正前・較正後で算出し，新旧の正誤ペアで`compute_mcnemar_test`（全体，
   α=0.05）を行う（Iter29/30の手順5と同一）。
6. **per-domain非退行チェック（Iter30で確立した3段構成をそのまま踏襲）**: 全10ドメインに
   ついて，(a) recallは`compute_domain_recall_mcnemar_test`（計10検定），(b) precisionは
   `compute_domain_precision_fisher_test`（計10検定）を実施し，計20個のp値を集めて
   `apply_benjamini_hochberg(p_values, q=0.05)`を一括適用する。adjusted有意かつ方向が悪化
   （較正後の点推定<較正前の点推定）である指標のみを「統計的に有意な退行」と判定する。
7. **temperature特有の実装確認（調査(Iter31)申し送り3により簡略化）**: 較正後の1600行に
   ついて，(a)確率のいずれかが厳密に`0.0`または`1.0`になっている行数，(b)10クラス全てが
   `0.1`に近いuniform fallback行数，(c)tie率の3点を，Iter30と同じ定義で算出するが，
   構造上いずれも「該当0件」になると予想されるため，legalドメイン個別集計などの詳細内訳は
   省略し，3点とも「該当0件であることの確認」に留める簡易報告とする（実験時間の節約）。
   もし予想に反して非ゼロの値が出た場合は，簡易報告に留めず詳細を追加報告すること。
8. 新旧classifierのargmax不一致件数（flip rate）をIter29/30と同じ定義で報告し，
   Iter29（platt，ensemble=True，11.0%）・Iter30（isotonic，ensemble=True，14.3125%）と比較する
   （必須報告項目，判定基準ではない）。

**成功条件（d0003 X9．AND条件．cv/ensemble選択に応じた解釈の但し書きを追加）**:

1. ECE（手順2・4，較正前基準0.19336に対する較正後の値，`n_bins=10`）が**0.150以下**であること。
2. top1_accuracy（手順5）が旧分類器（Iter28実測0.585）に対しMcNemar検定で有意に悪化していない
   （p>=0.05，または新側が改善方向）こと。
3. **per-domain非退行（手順6，Iter30と同一の3段構成）**: 20指標（10ドメイン×precision/recall）
   のp値へBH補正（q=0.05）を適用した結果，adjusted有意かつ悪化方向の指標が**0件**であること。
4. **【但し書き，調査(Iter31)申し送り2】** 条件2・3の判定において，`ensemble=True`に起因する
   合成データ実測1.12%程度の非ゼロargmax flipは，それ自体を理由に条件2・3を不成立とはしない。
   これは個々の`(estimator, T)`ペア内で厳密に成立する理論的argmax不変性が主張する範囲の外側
   （5fold平均化という別の機序）であり，判定はあくまで統計的検定（McNemar／Fisher／BH補正）の
   結果に基づく。もし条件2・3が実際に不成立になった場合，分析(解釈)フェーズでその原因が
   「ensemble平均化由来の偶発的再配分」なのか「temperature特有の別の機序（単一Tへの丸め込み
   によるドメイン固有のトレードオフ）」なのかを切り分けて報告すること。
5. 手順7のtemperature特有チェックリスト（簡易報告）とflip rate（手順8）は，成功・失敗の
   判定基準ではなく必須報告項目として記録する。

**目標未達時の次点候補（次イテレーション向けメモ，今回の計画には含めない）**: config.ymlの
`classifier_calibration`レバーは`platt`・`isotonic`・`temperature`の登録済み3値を今回で
使い切ることになる。仮に3手法いずれもd0003 X9のAND条件を満たせない場合，次の一手は
較正手法そのものの追加候補ではなく，運用的な対処（例：medicalドメインに限定して較正を
無効化する，ドメイン別に異なる較正手法を組み合わせる等）になる可能性が高い（backlog B51
要レビュー(1)がすでに示唆済み）。この判断は本計画の範囲外とし，次イテレーションのrc-reflector
に委ねる。

**人間判断が必要な論点**: 新規追加なし。Y2（`confidence_threshold`の二重責務分離，スキーマ
変更）着手前のユーザー確認はbacklog B49・B50・B51の既存の申し送りのまま。較正済み分類器の
本番反映可否も，temperatureが成功条件（本計画の1-3すべて）を満たした場合に改めてその時点で
判断する（今回のイテレーションで本番アーティファクトを置き換える判断は行わない）。

---

### 実装 (Iter31)

計画どおり単一レバー（`classifier_calibration=temperature`）のみを実装した．`config.yaml` は
変更していない（`git diff --stat -- config.yaml` が空であることを確認済み）。

**変更ファイル**:

1. `scripts/train_domain_classifier.py`
   - `_CALIBRATION_METHOD = "isotonic"` → `"temperature"` に変更。`_CALIBRATION_CV = 5` は無変更。
     `ensemble=True`（`train_classifier()`内`CalibratedClassifierCV(...)`呼び出しにハードコード）
     も計画どおり無変更で維持した（選択肢A，比較可能性優先）。
   - `_CALIBRATION_METHOD`直上のコメントを，temperatureを選ぶ理由（Iter30のisotonicが
     medical_recallのBH補正後有意悪化でpartial判定・backlog B51の自動選択）と，
     調査(Iter31)が確認した構造上の利点（単一スカラーTでロジット全体を変換するため
     クラスごとの個別較正器を持たず，isotonic/PlattのOvR方式由来のクラス固有曲線歪みを
     構造的に排除する）に更新し，cv/ensembleを維持する理由（較正手法のみを単一レバーとして
     切り分けるための比較可能性優先，Iter29/30と同一条件）も追記した。
   - モジュール冒頭docstringの`Iter30, classifier_calibration=isotonic`を
     `Iter31, classifier_calibration=temperature`に更新した。
   - `train_classifier()`のdocstringを`method="temperature"`の説明に更新し，isotonic特有の
     注意点（tie・0/1張り付き・uniform fallback）への言及を，「temperatureの実装構造上
     該当しない」旨に置き換えた。呼び出し側は引き続き手順7でチェックするが，リスクとしてでは
     なく「0件であることの確認」として扱う旨を明記した。
   - 出力アーティファクト名（`--output`の既定値）は変更していない
     （`models/domain_classifier.joblib`のまま）。Iter29/30と同じパターンで，実験フェーズでの
     実行時に`--output models/domain_classifier_temperature.joblib`をCLI引数で明示指定する
     ことで本番アーティファクトを上書きしない運用とする（スクリプト側の既定値変更は不要）。
2. `scripts/evaluate_classifier_calibration.py`: 計画どおり変更不要と確認した。
   `predict_calibrated_rows()`が返す`probabilities`フィールド（Iter30で追加済み，10ドメイン
   全ての確率）はtemperatureの実装確認（手順7）にもそのまま使えることをコード読解で確認した。
3. `metrics.py`: 計画どおり変更不要と確認した。`compute_domain_recall_mcnemar_test`
   （282行）・`compute_domain_precision_fisher_test`（318行）・`apply_benjamini_hochberg`
   （367行）がIter30で実装済みであることを`grep`で確認した。手法非依存の統計手続きのため
   そのまま再利用する。
4. `tests/test_train_domain_classifier.py`: isotonicや`_CALIBRATION_METHOD`への直接参照が
   ないことを`grep`で確認した上で無変更のまま実行し，pass することを確認した。

**テスト結果**: `uv run pytest -q` → 218 passed, 2 skipped（Iter30時点と同数，既存のスキップ
2件は本変更と無関係）。

**lint**: `uv run ruff check .` → 2件のエラー（`scripts/prepare_lora_training_data.py`のF541・
未使用import）が残るが，これはIter29から既知の本変更と無関係な既存差分であることを
`uv run ruff check scripts/train_domain_classifier.py`単体で"All checks passed!"となることで
確認した（単一レバー原則に従い今回も触っていない）。

**config.yaml の確認**: `git diff --stat -- config.yaml`が空であることを確認し，一切変更して
いないことを確認した。

**実験を開始してよい状態か**: はい。コード変更は完了し，テスト・lintとも整合。フェーズ4では，
(1) `scripts/train_domain_classifier.py`で`models/domain_classifier_temperature.joblib`を
1台のライブollamaノードへのembedding呼び出しで新規生成（本番`models/domain_classifier.joblib`
は上書きしない），(2) `scripts/evaluate_classifier_calibration.py`で1600問を再embeddingして
較正後の予測JSONL（`probabilities`フィールド付き）を`results/iter31_calibrated_predictions.jsonl`
として生成，(3) `metrics.py`の既存関数群で較正前（`results/20260731_162722/results.jsonl`，
再実行不要）と較正後を比較し，成功条件1-3（ECE≤0.150・McNemar非退行・per-domain 20指標への
BH補正非退行）と必須報告項目（temperature特有チェックリスト・flip rate）を実測すればよい。

---

### 実験・分析(実行) (Iter31)

計画どおり実機1600問本走は行わず，既存のSSHローカルポートフォワード（`127.0.0.1:11435 ->
wafl500:11434`，`ssh -fNT -L 11435:localhost:11434 wafl500`，Iter29/30から起動済みのプロセスを
そのまま流用．事前に`curl http://127.0.0.1:11435/api/tags`で疎通確認済み）経由のembedding呼び出し
のみで較正前後の比較データを揃えた．LLM生成・probe・dispatchは一切発生していない．

**手順1: 新分類器の学習**

```
uv run python -m scripts.train_domain_classifier \
  --train-data data/classifier_train.jsonl \
  --embedding-model nomic-embed-text \
  --ollama-host 127.0.0.1 --ollama-port 11435 \
  --output models/domain_classifier_temperature.joblib
```

標準出力: `[train_domain_classifier] wrote models/domain_classifier_temperature.joblib
(n_samples=1427, classes=[...10ドメイン...])`．実行時間124.55秒（`time`実測，Iter29のPlatt
124.09秒・Iter30のisotonic126.51秒とほぼ同水準）．`models/domain_classifier_temperature.joblib`
を新規生成し，本番`models/domain_classifier.joblib`のタイムスタンプ（Jul 27 16:08）が今回の実行後
も変化していないこと（＝上書きされていないこと）をファイルシステム上で確認した．

**手順3: 較正後データ生成**

```
uv run python -m scripts.evaluate_classifier_calibration \
  --dataset data/dataset.jsonl \
  --classifier models/domain_classifier_temperature.joblib \
  --embedding-model nomic-embed-text \
  --ollama-host 127.0.0.1 --ollama-port 11435 \
  --output results/iter31_calibrated_predictions.jsonl
```

標準出力: `[evaluate_classifier_calibration] wrote 1600 rows
(classifier=models/domain_classifier_temperature.joblib)`．実行時間141.56秒．出力JSONLは
計画どおり`probabilities`フィールド（10ドメイン全ての確率）付きで1600行生成された．

**手順2**: 較正前データは計画どおり`results/20260731_162722/results.jsonl`（Iter28実測，fallback
0/1600）を再実行せずそのまま使用．新旧2ファイルの`id`集合が完全一致することを確認済み
（`{r["id"] for r in before} == {r["id"] for r in after}`が`True`）．

**異常の有無**: なし．両スクリプトとも例外・タイムアウト・リトライなく正常終了した．実機呼び出し
はwafl500（192.168.15.100:11434）へのembeddingのみ（`nomic-embed-text`，計3027回：1427+1600），
LLM生成・probe・dispatchは一切発生していない．総所要時間は266.11秒（約4.4分，`timeout_min:150`
に対し十分余裕あり）．

`metrics.py`の既存関数（`compute_ece`／`compute_top1_accuracy`／`compute_mcnemar_test`／
`compute_precision_recall_per_domain`）と既存3関数（`compute_domain_recall_mcnemar_test`／
`compute_domain_precision_fisher_test`／`apply_benjamini_hochberg`，いずれもIter30で実装済みで
変更なし）を呼ぶ一時スクリプト（`/tmp/iter31_analysis.py`，非永続）で較正前
（`results/20260731_162722/results.jsonl`）と較正後（`results/iter31_calibrated_predictions.jsonl`，
各1600行）を比較した（判定はここでは行わず，数値のみを機械的に集計する）．

**手順4: ECE（`n_bins=10`で統一）**

- 較正前: **0.193357556998477**（`state.json`の`e29_results.ece_before`／`e30_results.ece_before`と
  同一値，再計算不要のところ実測でも一致することを確認）
- 較正後: **0.07120101725284995**
- 改善幅: 0.122157（較正前→較正後で減少，改善方向）
- 0.150との比較: 較正後0.0712 < 0.150（platt 0.16751・isotonic 0.12142より大幅に低い．目標
  0.150に対し余裕7.88pt，isotonicの2.86ptを大きく上回る）

**手順5: top1_accuracy（1600問，`expected_domains`との一致率）**

- 較正前: 0.585000（Iter28実測と同一値）
- 較正後: 0.605625
- 差分: +0.020625（較正後が高い，Iter29 platt +0.010625・Iter30 isotonic +0.008750より改善幅が大きい）

**手順5: McNemar検定（全体，対応のある2条件比較，較正前=A・較正後=B，連続性補正あり）**

- discordant_a_only（較正前のみ正解）: 30
- discordant_b_only（較正後のみ正解）: 63
- discordant_pairs（合計）: 93
- chi2_statistic: 11.010752688172044
- p_value: **0.0009058485425290641**（α=0.05で有意．較正後が正解に転じた行(63)が誤りに転じた
  行(30)を上回り，方向は改善で統計的に有意．platt(p=0.139)・isotonic(p=0.301)はいずれも有意
  差なしだったのに対し，今回は有意な改善という異なる結果）

**手順8: flip rate（argmaxが変わった行の割合，`id`で対応付け，Iter29/30と同じ定義）**

- **137/1600 = 0.085625（8.5625%）**．Iter29（platt，11.0%）・Iter30（isotonic，14.3125%）
  いずれよりも低い．調査(Iter31)の合成データ実測（`ensemble=True`下でtemperatureは
  isotonic/plattより小さいflipになる，1.12% vs 4.20%/3.85%）と定性的に整合する方向（実データでの
  絶対値は合成データの規模・分離度と異なるため単純比較はできないが，3手法中もっとも低いという
  順序は一致）．

**手順6: per-domain非退行チェック（Iter30で確立した3段構成：recall=ドメイン別McNemar・
precision=Fisher正確検定・20指標へBH補正q=0.05）**

全20指標の点推定（較正前→較正後）と個別検定のp値，BH補正後の有意フラグ：

| domain | metric | before | after | p_value | BH有意 | 方向 |
|---|---|---|---|---|---|---|
| business_economics | recall | 0.5179 | 0.5417 | 0.220671 | 否 | 改善 |
| business_economics | precision | 0.4328 | 0.4643 | 0.546043 | 否 | 改善 |
| computer_science | recall | 0.5417 | 0.5714 | 0.227800 | 否 | 改善 |
| computer_science | precision | 0.5987 | 0.6234 | 0.725154 | 否 | 改善 |
| education | recall | 0.4059 | 0.4588 | 0.015861 | 否 | 改善 |
| education | precision | 0.4631 | 0.5306 | 0.295424 | 否 | 改善 |
| general | recall | 0.5488 | 0.5732 | 0.220671 | 否 | 改善 |
| general | precision | 0.6522 | 0.6528 | 1.000000 | 否 | 改善 |
| history_culture | recall | 0.6667 | 0.6786 | 0.723674 | 否 | 改善 |
| history_culture | precision | 0.7320 | 0.6994 | 0.535412 | 否 | 悪化 |
| legal | recall | 0.5833 | 0.5778 | 1.000000 | 否 | 悪化 |
| legal | precision | 0.7500 | 0.7820 | 0.569458 | 否 | 改善 |
| mathematics | recall | 0.6190 | 0.6310 | 0.723674 | 否 | 改善 |
| mathematics | precision | 0.7075 | 0.7020 | 1.000000 | 否 | 悪化 |
| medical | recall | 0.4831 | 0.5112 | 0.182422 | 否 | 改善 |
| medical | precision | 0.4725 | 0.5056 | 0.599143 | 否 | 改善 |
| natural_science | recall | 0.5655 | 0.5833 | 0.605577 | 否 | 改善 |
| natural_science | precision | 0.5135 | 0.5444 | 0.600359 | 否 | 改善 |
| social_science | recall | 0.5774 | 0.5774 | 0.751830 | 否 | 改善 |
| social_science | precision | 0.6340 | 0.6382 | 1.000000 | 否 | 改善 |

BH（q=0.05）通過（adjusted有意）は20指標中**0件**．悪化方向の指標（history_culture_precision・
legal_recall・mathematics_precision）はいずれもp値が0.53-1.00と大きく，統計的な退行の根拠はない．

**medical_recallの内訳（Iter30でBH補正後有意に悪化していた指標，今回の再現有無を確認）**:
discordant_a_only=2（較正前のみ正解）・discordant_b_only=7（較正後のみ正解）・discordant_pairs=9・
chi2=1.7778・p=0.182422．**Iter30（isotonic，discordant_a_only=19・discordant_b_only=1・
p=0.000144・有意に悪化）とは対照的に，temperatureではmedical_recallはむしろ改善方向
（0.4831→0.5112）であり，統計的に有意な変化もない**．調査(Iter31)の理論的予想（単一Tはクラス
固有のOvR曲線歪みを構造的に持たないため，isotonicのmedical系統的圧縮は再現しないはず）と実測が
一致した．

**手順7: temperature特有の実装確認チェックリスト（`probabilities`フィールドを使用，1600行対象，
調査(Iter31)申し送り3により簡易報告）**

- (a) 確率のいずれかが厳密に`0.0`または`1.0`になっている行数: **0/1600**
- (b) 10クラス全てが`0.1`に近い（`math.isclose(p, 0.1, abs_tol=1e-9)`）uniform fallback行数:
  **0/1600**
- (c) 選択ドメインのconfidenceと同一の値を持つ他ドメインが存在する行の割合（tie率，厳密な
  浮動小数点一致で判定）: **0/1600（0.0000%）**

3点とも予想どおり該当0件だった．softmaxの出力は有限のロジット入力に対し厳密に0や1にはならず，
tieも理論上は浮動小数点の偶然の一致でしか起こらないという調査(Iter31)の実装読解（分かったこと(4)）
と実測が一致した．非ゼロの値は観測されなかったため詳細報告は不要と判断した．

**使用データ**:

- 訓練データ: `data/classifier_train.jsonl`（1427件，legal 77件・他9ドメイン各150件，
  Iter29/30と同一）
- 評価データセット（再embedding対象）: `data/dataset.jsonl`（1600件）
- 較正前の実行結果: `results/20260731_162722/results.jsonl`（Iter28実測，1600行，再実行なし）
- 較正後の実行結果（新規生成）: `results/iter31_calibrated_predictions.jsonl`（1600行，
  `probabilities`フィールド付き）
- 新規モデルアーティファクト: `models/domain_classifier_temperature.joblib`（本番
  `models/domain_classifier.joblib`は無変更のまま，タイムスタンプで確認済み）

**実行時間・実機呼び出しの有無**:

- `train_domain_classifier.py`: 124.55秒（1427回のembedding呼び出し）
- `evaluate_classifier_calibration.py`: 141.56秒（1600回のembedding呼び出し）
- 実機呼び出しはwafl500（192.168.15.100:11434）へのembeddingのみ（`nomic-embed-text`），
  計3027回．LLM生成・probe・dispatchは一切発生していない．
- 接続経路はIter29/30と同一の既存SSHローカルポートフォワード（`127.0.0.1:11435` ←
  `wafl500:11434`）をそのまま流用．新規に張り直す必要はなく，実行中のログ・エラーに異常なし
  （例外・タイムアウト・リトライなし，両スクリプトとも正常終了メッセージを出力）．

**state.json更新**: `status: waiting_experiment`（開始時，`experiment_dir:
results/iter31_calibrated_predictions.jsonl`・`experiment_deadline`設定）→`running`（完了時，
`experiment_dir`/`experiment_deadline`を`null`に戻した）．`e31_results`への数値記録・`judgment`
確定はフェーズ5b（rc-analyst）に委ねる（本フェーズでは数値の良否判定は行わない）．

---

### 分析(解釈) (Iter31)

**成功条件（d0003 X9，計画(Iter31)節，AND条件）との照合**

1. **ECE ≤ 0.150**: 明確に成立。較正後 0.071201 は較正前 0.193358 から −12.22pt（相対63.2%減）
   であり，3手法中もっとも大きい改善幅である。目標0.150に対する余裕は 7.88pt（isotonicの
   2.86pt・plattの未達）を大きく上回り，platt・isotonicいずれと比べても大きな差で目標を
   上回っている。ルーティングは決定論的（config.yml success_criteria (5)）であり，同一1600問・
   同一embeddingモデルに対し較正手法のみを変えた比較のため，この差分はノイズではなく較正手法の
   変更そのものが生んだ実測値と判断してよい（Iter29・Iter30と同じ根拠）。
2. **top1_accuracy 非退行**: 成立するだけでなく，**成功条件が想定した「非退行」の範囲を
   超えて有意な改善**である。計画(Iter31)の条件2は「p>=0.05，または改善方向」を非退行の
   基準としていたが，実測はMcNemar p=0.000906（α=0.05で有意）かつdiscordant_b_only
   （較正後のみ正解，63）がdiscordant_a_only（較正前のみ正解，30）を倍以上上回る改善方向
   であり，**統計的に有意な改善**と言うべき水準にある。top1_accuracyの絶対値も0.585→0.605625
   （+2.06pt）と3手法中もっとも大きい伸びであり，platt（p=0.139，非退行だが有意差なし）・
   isotonic（p=0.301，同）とは質的に異なる結果である。較正が単に「悪化していない」だけでなく
   ルーティング精度そのものを引き上げた可能性を示している。
3. **per-domain 20指標のBH補正後・悪化方向有意指標0件**: 成立。20指標（10ドメイン×
   precision/recall）のうちBH（q=0.05）通過は**0件**であり，isotonicの`medical_recall`
   （p=0.000144，通過1件）やIter29 platt（字義通り基準で9件該当，事後分析で全て非有意と
   判明）と異なり，最初から厳格な多重比較補正の下で悪化方向の有意指標が皆無だった。悪化方向
   の3指標（history_culture_precision, legal_recall, mathematics_precision）もp値は
   0.53〜1.00と大きく，統計的な退行の根拠はない。3手法のうちこの条件を単独で満たしたのは
   temperatureのみである。
4. **【但し書き，調査(Iter31)申し送り2・計画(Iter31)条件4】ensemble由来の非ゼロflip
   （8.5625%＝137/1600）の解釈**: この値は理論保証（sklearn公式の「Tはsoftmaxのargmaxの
   位置に影響しない」）が主張する範囲の**外側**（個々の(estimator, T)ペア内ではなく，
   `ensemble=True`による5fold平均化という別の機序）で生じており，それ自体を理由に条件2・3を
   不成立とはしない，という計画(Iter31)の事前合意どおりに扱った。実際，条件2・3の判定は
   統計的検定（McNemar・Fisher・BH補正）の結果のみに基づいており，flip率が非ゼロであること
   自体はここまでの分析で不利に働いていない。むしろ8.5625%はisotonic（14.3125%）・
   platt（11.0%）より低く，調査(Iter31)の合成データ実測（temperatureのensemble由来flipは
   isotonic/plattより一貫して小さい）と実データでも順序が一致した。この事実は，「ensemble
   平均化由来の偶発的再配分」以上の何かがtop1_accuracyを押し上げているという解釈（条件2）
   を弱めるものではなく，むしろ較正手法自体が生む再配分の総量が少ないままaccuracyが伸びた
   ことを示しており，isotonic/plattのような「大きな再配分の副産物として一部ドメインが
   犠牲になる」構造とは異なる結果である。

**isotonic（Iter30）との対比 — medical_recall問題の再現有無**

Iter30ではmedical_recallがBH補正後も有意に悪化（0.4831→0.3820, p=0.000144，
discordant a_only=19:b_only=1）し，較正後の最大確率0.7062が他9ドメイン（0.7496〜0.8795）
を全て下回るという，isotonic較正曲線のmedicalクラス固有の系統的圧縮が疑われていた。
今回のtemperatureでは，同じmedical_recallがdiscordant a_only=2:b_only=7・p=0.182422
（有意差なし）で，**点推定はむしろ改善方向（0.4831→0.5112）**である。

これは調査(Iter31)の理論的予想——「temperatureは単一スカラーTでロジット全体を変換する
構造上，クラスごとの個別較正器を持たず，isotonic/PlattのOvR方式由来のクラス固有曲線歪みを
構造的に持たない」——を**強く支持する証拠**と評価してよい。理由は次の3点である。

1. 同一の訓練データ（`data/classifier_train.jsonl`，medical 150件）・同一の評価データ
   （1600問）・同一のbase estimator（`LogisticRegression(class_weight='balanced')`）・
   同一の`cv=5, ensemble=True`という条件下で，較正手法だけを変えた比較になっている。
   単一レバー原則が厳密に保たれているため，medical_recallの挙動の違いは較正手法の構造差に
   帰責してよい。
2. isotonicのときに観測された「medicalだけ較正後の最大確率が体系的に低い」という現象は，
   OvR方式（各クラスを独立に二値較正してから正規化する）に固有の自由度の高さ（区分定数
   フィットが特定クラスのheld-outデータで不安定に歪みうる）に起因すると解釈されていた。
   temperatureは全クラスに同一の逆温度を掛けるだけで，どのクラスかによらず変換が対称的
   であるため，「特定の1クラスだけ確率の天井が下がる」という現象が構造的に起こり得ない。
   今回の実測（medical・legal含め全10ドメインで悪化方向の有意指標が0件）はこの構造的な
   予測と整合する。
3. ただし，これは「1回の実験（n=1）による整合」であることに留意が必要である。medical
   クラスの埋め込み分離が本来弱いという可能性自体を否定する証拠ではなく，あくまで
   「OvR方式由来の較正曲線歪みという機序」が今回不在だったことを示すに留まる。それでも
   Iter30が示した唯一の懸念（medical_recall）が，理論から予想された通りの手法変更
   （temperatureへの切り替え）だけで解消したことは，偶然の一致にしては機序の説明が具体的
   （単一スカラー変換とOvR個別較正器という明確な構造差）であり，強い状況証拠と判断する。

**platt/isotonicとの比較でtemperatureが3手法中もっとも成功条件を満たしている理由の考察**

事前の留保（config.yml note・backlog B51・調査(Iter31)分かったこと(4)）は「temperatureは
表現力が低いためECE改善幅がisotonic・plattより小さくなる可能性」を懸念していたが，実測は
その逆で，ECE改善幅は temperature(0.1222) > isotonic(0.0719) > platt(0.0258) という
**もっとも深い**結果になった（数値は較正前0.193358からの絶対改善幅）。この逆転は次のように
解釈できる。

- 較正の訓練データは1427件を10クラスに分割し（medical/legal以外は各150件，legalのみ77件），
  `cv=5`ではさらに1foldあたり数十件規模まで細分される。isotonic・plattはこの少量データ上で
  **クラスごとに個別の較正関数**を学習するため，held-outデータのノイズに対して過学習しやすい
  （isotonicは特に自由度が高い区分定数フィットで，Iter30のmedical系統的圧縮はこの過学習の
  症状と整合する）。temperatureは全クラス共通の**単一スカラーパラメータ**しか学習しないため，
  1427件全体（実質的に多クラスのmultinomial loss全体）から1つのTを推定でき，個々のクラスの
  小標本性に脆弱ではない。つまり，このデータ規模では「表現力の低さ」がむしろ分散を抑え，
  過学習を防いだと考えられる——古典的なバイアス・分散トレードオフで，パラメータ数が少ない
  ほうが小標本の較正タスクでは汎化しやすかった，という説明である。
- もう一つの見立ては，このLogisticRegression分類器の較正誤差が，そもそも「クラスごとに
  異なる歪み方をする」構造ではなく，「全クラス一律に過信（over-confident）している」という
  **大域的な過信バイアス**が支配的だった可能性である。もしそうであれば，単一Tによる大域的
  スケーリングだけで大部分の誤差を解消でき，OvR方式のクラス固有補正は，本来存在しない
  クラス間の歪みの違いを学習データのノイズから読み取ってしまい，かえって較正を悪化させる
  （isotonicのmedical系統的圧縮，plattのECE絶対閾値未達）方向に作用したと考えられる。
- 前者・後者いずれの説明も「表現力が高い手法が必ず良い較正を生むとは限らない」という一般的な
  較正手法選択の知見（少数クラス・小標本条件下での過学習リスク）と整合しており，本リポジトリ
  の訓練データ規模（1427件・10クラス）が，isotonic/plattの柔軟性を活かすには小さすぎた
  可能性を示唆する。ECE改善幅の逆転という結果自体は今回のn=1測定だが，isotonic（Iter30）で
  観測された系統的圧縮という具体的な機序と符合しており，単なる偶然の逆転とは考えにくい。

**本番反映（`models/domain_classifier.joblib`をtemperature版に置き換えるか）についての見解
（提案，確定はrc-reflector）**

**採用（adopted）し，本番反映を進めることを提案する**。判断基準:

- 成功条件1〜3のAND条件をすべて満たした較正手法は，platt・isotonic・temperatureの3手法中
  temperatureのみである。isotonic（Iter30）はmedical_recall悪化で条件3不成立，
  platt（Iter29）はECE絶対閾値未達で条件1不成立と，いずれもpartialで確定している。
  temperatureはこの2つの懸念をいずれも回避しており，「AND条件を字義通り満たす」という
  意味で今回初めて明確なadopted相当の結果が得られている。
- 条件2（top1_accuracy）は非退行を超えて有意な改善（p=0.000906）であり，条件1（ECE）も
  目標に対し7.88ptの余裕がある。isotonic・plattのように「AND条件の一部だけ危うい」形では
  なく，3条件のいずれにも明確な余裕がある。
- 可逆性の観点でも問題は小さい。`models/domain_classifier_temperature.joblib`は既に
  別名で生成済みであり，本番`models/domain_classifier.joblib`との入れ替えはファイルの
  差し替えのみで完結し，何らかの不具合が判明した場合は較正前のjoblibへ即座に戻せる
  （config.yaml自体は一切変更していないため，ロールバックにコード変更は不要）。
- 一方で，確信度を完全な最終確定ではなく「提案」に留めるべき留保点が2つある。(a) 本判定は
  n=1（ルーティングは決定論的だが，1600問という単一の評価セット・単一の訓練データ分割に
  基づく）測定であり，Iter28・Iter29・Iter30と同じ制約を共有している。(b) isotonic
  （Iter30）のmedical_recall悪化が「OvR由来の機序」の実例として片付けられるかどうかは，
  今回の1イテレーションの整合的な結果からの推論であり，直接に反証実験を行ったわけではない
  （例えば，temperatureをさらに複数回・複数の訓練データ分割で再現するような追加検証は
  今回行っていない）。判断の主要な数値（ECE・McNemar・BH補正）自体は確定的であり追加反復を
  要するとは考えないが，「なぜtemperatureがisotonic/plattより優れていたか」という機序の
  説明は今回の考察であり，本番反映後もECE・per-domain指標の定期的なモニタリングを継続する
  ことを勧める。

**確信度と追加反復の要否**: 成功条件1〜3の判定そのものの確信度は高い（決定論的ルーティング・
BH補正済み・3手法全てで同一手順を適用した比較のため）。追加反復（同一実験の再実行）は
不要と考える——ルーティングが決定論的である以上，再実行しても数値は変わらない。ただし
上記のとおり「temperatureが優れていた理由」の機序面の説明はn=1の考察に留まるため，
本番反映後の運用モニタリング（例えば次に大きな訓練データ更新が入った際にECE・per-domain
指標を再確認する）は推奨事項として申し送る。

**総合判断（rc-analyst提案）: adopted（全面採用）**。config.ymlの`classifier_calibration`
レバーは`[platt, isotonic, temperature]`の3値を全て試し終えており，platt=partial・
isotonic=partial・temperature=今回の提案どおりadoptedとなれば，Y4（分類器の較正）は
temperatureの採用をもって完了とすることを提案する。最終的な採否確定と，
`models/domain_classifier.joblib`の実際の置き換え作業はrc-reflectorに委ねる。

---

### 考察 (Iter31)

**判定: adopted（全面採用，rc-analyst提案を覆さず確定）**。d0003 X9 の成功条件（ECE≤0.150・
top1_accuracy非退行・per-domain 20指標のBH補正後悪化方向有意指標0件のAND条件）を，`platt`
（Iter29，ECE絶対閾値未達でpartial）・`isotonic`（Iter30，medical_recallのBH補正後有意悪化で
partial）に続き3手法目の`temperature`が初めて明確に満たした。ECEは0.193358→0.071201（目標に
7.88ptの余裕，3手法中もっとも大きい改善幅），top1_accuracyは0.585→0.605625でMcNemar
p=0.000906の**有意な改善**（非退行を上回る），per-domain 20指標のBH補正後有意指標は0件。
Iter30で唯一の懸念だったmedical_recallも，temperatureでは有意差なし（p=0.182422）でむしろ
改善方向（0.4831→0.5112）と，isotonicの系統的圧縮が再現しなかった。rc-analystの分析（機序：
temperatureは単一スカラーTでロジット全体を変換するためisotonic/plattのOvR方式由来のクラス
固有曲線歪みを構造的に持たない）は，同一訓練データ・同一評価データ・同一cv/ensemble設定という
単一レバー原則が厳密に保たれた比較の下で得られた結果であり，覆す理由を見いだせなかったため
確定させる。

**本番反映: 実施済み**。`models/domain_classifier.joblib`（旧・較正なし，
sha256=`3a5610a...`）を`models/domain_classifier_uncalibrated_pre_iter31.joblib`へ退避のうえ，
`models/domain_classifier_temperature.joblib`（sha256=`04bb9ff...`）で置き換えた。判断根拠:
(1) 成功条件のAND条件を明確な余裕（ECE 7.88pt・top1有意改善・BH補正後有意退行0件）で満たして
いる，(2) `config.yaml`・公開APIの変更を一切伴わない可逆なファイル差し替えである（不具合が
判明すれば`models/domain_classifier_uncalibrated_pre_iter31.joblib`へ即座に戻せる），
(3) これは委譲時の指示で明示的に「rc-reflectorの自律判断範囲内（可逆な判断）として進めて
構わない」とされた操作である。**注意**: `models/`はリポジトリの`.gitignore`（19行目）で除外
されており，この置き換えはgit管理下にない。ロールバック手順と両ファイルのsha256はこの節と
上記に記録した以外に残らないため，次回このモデルに触れる際は本節を参照すること。

**学び**:

1. **isotonicのmedical_recall悪化は「OvR方式由来のクラス固有曲線歪み」という機序で
   説明できることが，temperatureへの切り替えのみで解消したという形で強く裏付けられた**。
   同一データ・同一cv/ensembleの下で較正手法だけを変えた比較が3イテレーション連続で
   積み上がったことで，この機序の特定は単発の考察ではなく再現性のある知見になった。
2. **「表現力が高い較正手法が必ず良い較正を生むとは限らない」という一般的な較正手法選択の
   知見が，本リポジトリの訓練データ規模（1427件・10クラス，legalのみ77件）で実測として
   裏付けられた**。ECE改善幅はtemperature(0.1222) > isotonic(0.0719) > platt(0.0258)と，
   もっとも柔軟性の低い手法がもっとも大きく改善するという事前の留保（config.yml note・
   backlog B51）とは逆の結果になった。小標本条件下ではOvR方式のクラス別自由度がheld-outの
   ノイズを拾って過学習し，かえって較正を悪化させるためと考えられる。次に較正関連のレバーを
   検討する際は，「手法の表現力の高さ＝較正の質」という前提を置かないこと。
3. **`ensemble=True`由来の非ゼロflip（合成データ実測1.12%，実データ8.5625%）は，
   sklearn公式の「Tはsoftmaxのargmaxを変えない」という理論保証の範囲外（個々の
   (estimator, T)ペア内の話であり，5fold平均化という別の機構）であるという整理は，
   今後isotonic/platt/temperatureいずれについても「flipが非ゼロ＝理論違反」という
   短絡的解釈を避けるために有効だった。次回較正手法を検討する際も踏襲すること。
4. **`models/`がgitignore対象であるため，較正済み分類器の本番反映はgit履歴に残らない**。
   D5（backlog未解決事項，`data/`/`models/`のバージョン管理方針）が引き続き未解決であり，
   今回のように本番アーティファクトを差し替える判断が何度も発生する局面では，最低限
   sha256ハッシュのマニフェストをjournal/backlogに記録する運用（今回実施した方式）を
   今後も徹底する必要がある。

**Y4（分類器の較正，d0003 X9）は本イテレーションをもって完了**。config.ymlの
`classifier_calibration`レバーは`[platt, isotonic, temperature]`の3値すべてを試し終えた。

**次イテレーション（Iter32）の単一レバー決定**: d0004 §5の優先順位はY1（完了）→Y4（完了，
本イテレーション）→Y2（前提整備，スキーマ変更を伴い着手前にユーザー確認が必要）→Y3（Y2完了後）
→Y5（education/legalのデータ不均衡是正）である。Y2は`config.yaml`への
`dispatch_candidate_threshold`新設・`aggregator.select_dispatch_targets()`のシグネチャ変更を
伴い，backlog B49・B50・B51で繰り返し「着手前にユーザー確認が必要」と申し送られてきた
不可逆側の判断であり，rc-reflectorの自律判断権限（可逆な判断に限る）では着手を開始できない。
一方，Y3はY2完了が前提のため同様に着手不能。したがって実行可能な登録済みレバーは
`classifier_calibration`（完了）・`fallback_policy`（完了）のみとなり，`aggregation_method`
（Y3）はY2完了までブロックされたまま実質「試せない」状態にある。

これは「config の全 levers を試し切った」場合と実質的に同じ状況（唯一残る登録レバーが
ブロックされていて実行不能）と判断し，SKILL.mdが定める停止条件の優先順1（journal/backlogの
学びから次の有望なレバーを自分で考案し，config.ymlのlevers末尾へ追記して継続する）に従い，
**Y5（education/legalのデータ不均衡是正，d0003 X8）を新規レバーとしてconfig.ymlへ追記し，
Iter32の単一レバーとする**。理由と選定過程はbacklog.md B52に記録する（下記参照）。Y2は
自律着手不能なままのため，backlogの「要レビュー」として引き続き申し送る（新規の追加事項はない）。

---

## Iteration 30: 分類器較正のisotonic方式によるECE目標達成の追試とドメイン別非退行の全数検証

### 調査 (Iter30)

**問い**:
1. 1427件・legal 77件という規模で`CalibratedClassifierCV(method='isotonic')`を使う具体的リスクは何か．
   Iter29が確認した「≪1000件で過学習」という sklearn 公式の目安を，本イテレーションで独立に裏取りできるか．
2. 20指標（10ドメイン×precision/recall）の非退行チェックにおいて，Iter29の学び1（CI下限の単純前後比較は
   多重比較補正なしでは脆弱）を受け，どう改めるべきか．Bonferroni／Benjamini-Hochberg（BH）／区間の
   非交差／ドメイン単位McNemar検定のうち，実装コストと妥当性のバランスが良い方法を1つ推奨する．
3. isotonicはplattより表現力が高い分，過学習時の argmax flip がplattより大きくなりうるか．
   実装上の落とし穴（単調性の破れ・確率の0/1張り付き等）を整理する．
4. `method='temperature'`はsklearnの`CalibratedClassifierCV`に実在するか．Iter29 reflectorの申し送り
   （sklearn>=1.8で利用可能）の前提が正しいかを確認する．

#### 分かったこと

**(1) isotonic較正の技術的妥当性 — Iter29の裏取りに加え，新たな具体的懸念点を確認**

本リポジトリの実行環境（`.venv`，`uv.lock`固定）で `scikit-learn==1.9.0` がインストール済みであることを
`uv run python` から直接確認した．インストール済みパッケージのソース
（`sklearn/calibration.py`，`CalibratedClassifierCV`のdocstring）には
「Isotonic calibration is not recommended when the number of calibration samples is too low
``(≪1000)`` since it then tends to overfit」という文言が verbatim で存在し，Iter29が引用した
sklearn公式ドキュメント（`calibration.html`）の記述と完全に一致することを一次ソース（インストール
済みパッケージそのもの）で再確認した．さらに `tavily-extract` で `calibration.html` を直接取得し，
「Overall, 'isotonic' will perform as well as or better than 'sigmoid' when there is enough data
(**greater than ~ 1000 samples**)」という定量的な閾値の原文を確認した．また同じ文言
（`<<1000`／Platt推奨）が sklearn 0.18（2016年当時）の過去ドキュメントにも既に存在していたことを
web検索で確認しており（`vighneshbirodkar.github.io`のアーカイブ），この目安は最近の変更ではなく
10年近く sklearn が一貫して明記してきた安定した経験則である．

本データでの実測（Iter29既出，本イテレーションで再確認）: `cv=5`・`ensemble=True`の下では
1 fold あたりの較正サンプル数は9ドメインで約30件，legalで約15件．これは「≪1000」を大きく下回るのは
もちろん，isotonic回帰それ自体の性質（ノンパラメトリックで自由度が事実上サンプル数に等しい）から
言えば，1000件どころか一般的な「数百件」規準（emergentmind.comの「200件未満で過学習し得る」という
目安，Iter29既出）にも legal は届かない．**追加確認**: `IsotonicRegression`は`out_of_bounds="clip"`
で運用されており，較正用の held-out データに含まれない極端なスコアはヒストグラムの両端の値へ
クリップされる．該当ドメインの held-out データが少ないほど，この「両端の値」自体が0や1に近い
不安定な推定値になりやすい．

**(2) 多重比較への対処 — Benjamini-Hochberg（BH）法を，指標の対応構造に応じた2種類の検定と
組み合わせて用いることを推奨**

一般的なガイドライン（LaunchDarkly社の実験ドキュメント，2026年時点で確認）は「比較数が3以下なら
Bonferroni，それを超えるとBHの方が検出力とのバランスが良い」と明記している．20指標（10ドメイン×
precision/recall）はこの目安を大きく超えるため，Bonferroni（α=0.05/20=0.0025）は過度に保守的で
真の退行を見逃すリスクが高く，「区間の非交差」を基準にする案（config.ymlの申し送りにある選択肢の
一つ）はBonferroniよりさらに保守的な基準になりがちで感度が低い．

**推奨: BH法（FDR制御，q=0.05）を第一候補とする．ただし適用する検定は，指標ごとの対応構造に応じて
使い分けるべきである**．

- **recall**（分母＝真のドメインがXである行の集合．較正前後で分母の行集合は不変＝対応データ）には，
  既存の`metrics.py:compute_mcnemar_test`をドメイン別にサブセット適用する（=10検定）．これは
  Iter29の学び1が示唆する「ドメイン単位のMcNemar検定」をそのまま使える構造である．
- **precision**（分母＝分類器がXと予測した行の集合．較正で argmax が変われば分母の行集合自体が
  変わる＝非対応データ）は，McNemarの前提（同一対象への対の観測）を満たさないため，2標本比率の
  差の検定（Fisher正確検定または$\chi^2$検定，非対応）を用いる（=10検定）．
- 得られた計20個のp値に対しBH法を一括適用し，adjusted p<0.05のもののみを「統計的に有意な退行」と
  判定する．実装コストは低い（既存の`compute_mcnemar_test`のドメイン別ラッパー関数＋
  `scipy.stats.fisher_exact`または`chi2_contingency`の呼び出し＋BH補正（p値をソートして
  `p_(i) * m / i` を取るだけの数行）で完結し，外部ライブラリの新規追加は不要）．

**(3) isotonic特有の非退行確認の注意点 — sklearn公式ドキュメント・ソースコードで3点を具体的に確認**

- **ties（同値化）による ranking の粗視化**: sklearn公式ドキュメント（`calibration.html`
  1.16.3.3節脚注）が明記：「isotonic regression introduces ties in the predicted probabilities」
  であり，「It is generally expected that calibration does not affect ranking metrics such as
  ROC-AUC. However, these metrics might differ after calibration when using
  `method="isotonic"`」．一方 sigmoid は「a strictly monotonic transformation and thus keeps
  the ranking」と明記されている．本タスクのargmax選択は本質的にランキング操作であるため，
  isotonicはplattよりtie（複数ドメインが同一の較正後確率を持つ状態）を生みやすく，僅差の候補間で
  argmaxが不安定化するリスクがplattより高いと考えられる．cv fold あたりのサンプルが最少のlegal
  （約15件）で最も起きやすい．
- **確率の0/1張り付き（exact zeros）**: sklearn公式ソース（`_CalibratedClassifier.predict_proba`
  のdocstring）に「The predicted probabilities. Can be exact zeros.」と明記されている．
  `IsotonicRegression(out_of_bounds="clip")`は較正用データの範囲外のスコアを最も近い観測値へ
  クリップするため，その観測値自体が0や1（小標本のheld-outデータでは十分あり得る）であれば，
  較正後の確率がそのまま0または1に張り付く．これはIter16で問題視された「verbalized confidence
  の0/1飽和」と同種の病理を，較正という「飽和を直す」はずの処理が別の経路（isotonicの区分定数性）
  で再導入しうることを意味し，ECEの見かけ上の改善と裏腹に個々の予測の信頼性を損なう可能性がある．
- **全クラスが0になった場合のuniform fallback**: sklearn公式ソース（`_fit_calibrator`直後の
  `predict_proba`実装，コメント「In the edge case where for each class calibrator returns a
  zero probability for a given sample, use the uniform distribution instead」）が明記する
  実装上のフォールバック．10クラス全てのOvR較正器が0を返すサンプルが発生すると，較正後確率は
  10クラス均等（各0.1）に置き換わり，argmaxは分類器本来のランキングと無関係な（実装依存の）
  tie-breakで決まる．発生頻度は不明だが，該当した場合は「較正が改善させた」のではなく
  「較正が情報を破壊した」ケースであり，flip rateの数値だけでは区別できない．**実験時は
  `predict_proba`の行和が学習データ内で0.1×10=1.0のuniform行になっていないか（例えば
  `np.allclose`で0.1の一様分布との一致を検出）を追加でチェックすることを推奨する**．

**(4) `method='temperature'`は実在する — Iter29 reflectorの申し送りは正確**

課題文は「sklearnにあるのはsigmoidとisotonicの2値のみのはず」という疑いを提示していたが，
本リポジトリの実行環境で直接確認した結果，**Iter29の申し送りは正確であり，疑いは誤りだった**．

- `uv run python -c "from sklearn.calibration import CalibratedClassifierCV; help(...)"`で，
  `method`パラメータの型注釈が `{'sigmoid', 'isotonic', 'temperature'}` であることを確認．
  docstringに `.. versionchanged:: 1.8 Added option 'temperature'.` と明記されている．
  本リポジトリの`uv.lock`は`scikit-learn==1.9.0`を固定しており，1.8以降のバージョンなので
  `temperature`は現に利用可能である．
- sklearn公式ドキュメント（`calibration.html` 1.16.3.4節，`tavily-extract`で直接取得）は
  temperature scalingについて次のように明記している：「temperature scaling naturally supports
  multiclass predictions by working with logits and finally applying the softmax function」
  （sigmoid/isotonicのようなOvR分解＋事後正規化が不要）．「The parameter T is learned by
  minimizing log_loss ... on a hold-out (calibration) set. Note that T does not affect the
  location of the maximum in the softmax output. Therefore, temperature scaling does not alter
  the accuracy of the calibrating estimator.」——ロジット（`decision_function`の出力，または
  `predict_proba`の対数）全体を単一のスカラーTで割るだけの変換であるため，クラス間の大小関係
  （argmax）が理論的に不変であることが公式に保証されている．sklearnソース
  （`_fit_calibrator`）でも，`method="temperature"`の場合はsigmoid/isotonicのようにクラスごとに
  個別の較正器を作らず，**単一の`_TemperatureScaling`インスタンスのみを fit する**実装になって
  おり，OvR方式に起因するargmax入れ替わりのリスク（Iter29が指摘した多クラス較正の主要懸念）は
  構造的に排除されている．
- 使用中のbase estimator（`LogisticRegression`）は`decision_function`を持つため，temperature
  scalingはロジットを直接使う経路（`predict_proba`の対数を取る近似ではなく）で動作する．

#### rc-planner への申し送り

1. **isotonicの技術的リスクはIter29の想定どおり，むしろ具体化された**．legalドメイン
   （較正fold内約15件）はsklearn公式の「≪1000」「~1000件超で互角以上」のどちらの目安からも
   大きく外れており，`cv=5`のまま実施する場合はplatt以上に慎重な監視が要る．
2. **per-domain非退行チェックの運用を今回から変更することを強く推奨する**：
   `success_criteria (2)`の「CI下限の単純比較」をそのまま使い続けると，Iter29で実際に起きたように
   20指標中9指標が偽陽性で該当してしまう．今回のisotonic実験では**最初から**（事後の穴埋めでなく）
   (a) recallはドメイン別McNemar検定，(b) precisionは2標本比率検定（Fisher正確検定），
   (c) 計20個のp値へBH法（q=0.05）を適用，という3段構成で判定することを計画に含めるべきである．
   これは既存の`compute_mcnemar_test`／`compute_wilson_confidence_interval`の関数群を活かしつつ
   数十行の追加で実装できる．
3. **isotonic特有の実装確認項目を計画・実験段階でチェックリスト化すること**: (a) 較正後
   `predict_proba`の値が厳密に0または1になっている行がないか，(b) 10クラス全て0.1（uniform
   fallback）になっている行がないか，(c) 較正後の同一confidence値を持つ行（tie）の割合，
   の3点をIter29のflip rate報告に加えて算出する．特にlegalドメインの行を優先的に確認する．
4. **isotonicがECE目標（0.150以下）に届かない場合の次点候補は`method='temperature'`で確定できる**．
   Iter29の申し送りは正確であり，本リポジトリの`scikit-learn==1.9.0`で実際に利用可能である．
   temperature scalingはtop1_accuracy不変が理論的・実装的（単一の`_TemperatureScaling`インスタンス
   のみをfitする構造）に保証されるため，「ECE改善とルーティング非退行」というY4の目的に対し，
   sigmoid/isotonicのOvR方式が抱える構造的リスク（argmax入れ替わり，tie，0/1張り付き）を
   そもそも持たない代替である．ただし，temperatureは「多クラス全体で単一のTを学習する」ため，
   ドメインごとの較正の柔軟性はsigmoid/isotonicより低く，legalのように較正のずれ方が
   ドメイン固有の場合には改善幅が小さい可能性がある点は留保として記録する．
5. 今回のisotonic実験の計画では，Iter29の考察で確定した手順（全10ドメインのCIを較正前後で
   同一手順・最初から算出する）に加え，上記2・3の追加チェックを組み込むこと．

**出典**:
- ローカル実行環境の直接確認: `uv run python -c "import sklearn; print(sklearn.__version__)"`
  → `1.9.0`，および`sklearn.calibration.CalibratedClassifierCV`のdocstring・ソース
  （`.venv/lib/python3.12/site-packages/sklearn/calibration.py`）を`help()`・`grep`・`Read`で
  直接確認（2026-07-31実施，一次ソース）．
- https://scikit-learn.org/stable/modules/calibration.html （`tavily-extract`で直接取得，
  1.16.3.3 Multiclass support・1.16.3.4 Temperature Scaling・isotonic過学習閾値・
  ties/ranking注記，2026-07-31時点のstable版）
- http://vighneshbirodkar.github.io/scikit-learn.github.io/dev/modules/generated/sklearn.calibration.CalibratedClassifierCV.html
  （sklearn 0.18時代の同一文言のアーカイブ，`tavily-search`で発見．「≪1000」目安が10年近く
  一貫していることの裏付け）
- https://notes.cs307.org/classifier-calibration.html ，
  https://medium.com/data-science-at-microsoft/model-calibration-for-classification-tasks-using-python-1a7093b57a46
  （isotonicが区分定数関数でsigmoidより過学習しやすいという解説の補強，`tavily-search`）
- https://stats.stackexchange.com/questions/493393/ （isotonicがties経由でROC-AUC等のranking指標に
  影響するというコメント，`tavily-search`）
- https://launchdarkly.com/docs/guides/statistical-methodology/mcc （Bonferroni対BHの使い分け目安
  「3件以下ならBonferroni，それ以上ならBH」，`tavily-search`）
- https://docs.statsig.com/statsig-warehouse-native/features/statistics/methodologies/benjamini-hochberg-procedure
  （BH法の定義，FWER対FDRの違い，`tavily-search`）
- journal.md「調査 (Iter29)」節（本調査の裏取り元，sklearn issue #18709・#34312・
  emergentmind.comの引用は Iter29 で既出のため本イテレーションでは再掲のみ）

---

### 計画 (Iter30)

**仮説**: `scripts/train_domain_classifier.py:train_classifier()` の較正手法を
`method="sigmoid"`（Platt，Iter29 で partial 判定）から `method="isotonic"` へ切り替えると，
isotonic のノンパラメトリックな柔軟性により ECE が Platt（0.16751）よりさらに改善し，
目標の 0.150 以下へ到達する．一方，legal ドメイン（cv fold あたり較正サンプル約 15 件）では
isotonic 特有の過学習・tie・0/1 張り付きにより，per-domain の非退行が Platt 以上に脅かされる
リスクがある．この 2 つのトレードオフを，調査(Iter30) が申し送った多重比較補正済みの統計的
判定手順で最初から検証する．

**単一レバー**: `classifier_calibration`（`.claude/research/config.yml` のレバー名，150-170行）．
今回試す値は `values: [platt, isotonic]` のうち **`isotonic` のみ**（backlog B50 の自動選択）．
`cv=5`・`ensemble=True` は Iter29（Platt）と完全に同一のまま固定し，較正手法のみを変える．
`cv=3` 等の感度分析は，isotonic の主結果（`cv=5`）で per-domain 非退行が崩れた場合にのみ
副次分析として検討し，今回の主比較には含めない（backlog B50 の申し送りどおり）．

**固定する構成（Iter29 と完全に同一，`config.yaml` は一切変更しない）**:
`routing_method=supervised_classifier`，`confidence_threshold=0.0`・`dispatch_top_k=1`・
`aggregation_method=max_confidence`（Iter28 adopted 構成），`confidence_signal_method=self_report`，
`confidence_elicitation=top_k_with_probs`，`expert_model=expert-mesh-{domain}-lora`
（domain_count=10），`light_model=qwen3.5:4b-q4_K_M`，`embedding_model=nomic-embed-text`，
評価データセットは Iter25 以降固定の 1600 問（`data/dataset.jsonl`）．
`CalibratedClassifierCV` の `cv=5`・`ensemble=True` も Iter29 と同一．**今回変更するのは
`train_classifier()` 内の較正手法（`_CALIBRATION_METHOD` 定数の値）のみであり，`config.yaml`
のキーは 1 つも変えない．**

**変更ファイル・行**:

1. `scripts/train_domain_classifier.py`
   - `_CALIBRATION_METHOD = "sigmoid"`（55行）→ `"isotonic"` に変更。`_CALIBRATION_CV = 5`
     （56行）は無変更。
   - 45-54行のコメント（sigmoid を選んだ理由の説明）を，isotonic を選ぶ理由（Iter29 の Platt が
     ECE 絶対閾値未達で partial 判定・backlog B50 の自動選択）と，調査(Iter30) が確認した
     追加リスク（isotonic はノンパラメトリックで自由度が事実上サンプル数に等しく，sigmoid より
     過学習しやすい／`out_of_bounds="clip"` により legal の held-out データが極端値に張り付き
     やすい）に更新する。
   - モジュール冒頭 docstring（1-5行）の `Iter29, classifier_calibration=platt` を
     `Iter30, classifier_calibration=isotonic` に更新。
   - `train_classifier()` の docstring（95-97行）の `method="sigmoid"=Platt` を
     `method="isotonic"` に更新し，isotonic 特有の注意点（ties・0/1 張り付き・uniform
     fallback，調査(Iter30) 分かったこと(3)）を一言追記する。
   - 出力アーティファクト名は `models/domain_classifier_isotonic.joblib`（Platt 版
     `models/domain_classifier_platt.joblib` とは別名で新規生成，本番
     `models/domain_classifier.joblib` は上書きしない）。
   - `tests/test_train_domain_classifier.py` は Iter29 で `cv=5` 実行に必要な最小データ量
     （各クラス5件）へ既に拡張済みであり，`method` を変えても `StratifiedKFold` の分割条件は
     変わらないため無変更で通る見込み（実装フェーズで実行して確認する）。

2. `scripts/evaluate_classifier_calibration.py`
   - `predict_calibrated_rows()`（55-82行）が返す各行の辞書に，`"probabilities"`
     フィールド（`{domain: float(p) for domain, p in zip(classes, probabilities)}`，10 ドメイン
     全ての確率）を追加する。Iter29 では選択ドメインの confidence のみで十分だったが，
     isotonic 特有のチェックリスト（0/1 張り付き・uniform fallback・tie 検出）には全クラスの
     確率ベクトルが必要なため（調査(Iter30) 申し送り3）。既存フィールド（`id`／
     `expected_domains`／`selected_domain`／`confidence`）は変更しない（`metrics.py` の
     既存関数がそのまま読める後方互換を保つ）。
   - モジュール冒頭 docstring（1-26行）に，Iter30 で `probabilities` フィールドを追加した
     理由を追記する。CLI 引数（`--dataset`／`--classifier`／`--output` 等）は変更不要。

3. `metrics.py`（新規関数を追加，既存関数は無変更）
   - `compute_mcnemar_test`（226-261行）と本質的に同じ discordant-pair の χ²／p 値計算を
     `_mcnemar_from_correctness(correct_a: dict[str, bool], correct_b: dict[str, bool]) ->
     dict[str, float]` として切り出す（DRY，重複コード回避が目的の小さな抽出であり目的外の
     大規模リファクタリングではない）。`compute_mcnemar_test` はこのヘルパーを呼ぶよう変更。
   - 新規: `compute_domain_recall_mcnemar_test(results_a: list[dict], results_b: list[dict],
     domain: str) -> dict[str, float]`。`id` が一致する行のうち `domain in
     expected_domains` の行だけをサブセットし，正誤を `selected_domain == domain`
     （recall の定義そのもの）で定義して `_mcnemar_from_correctness` に渡す。id 集合の不一致は
     `compute_mcnemar_test` と同様に `ValueError`。
   - 新規: `compute_domain_precision_fisher_test(results_a: list[dict], results_b: list[dict],
     domain: str) -> dict[str, float]`。precision は分母（`selected_domain == domain` の行集合）
     自体が較正前後で変わる非対応データのため，2×2 分割表
     `[[tp_a, selected_a - tp_a], [tp_b, selected_b - tp_b]]`（`tp` = `selected_domain ==
     domain and domain in expected_domains`）を作り `scipy.stats.fisher_exact`
     （両側検定）で `p_value`・`odds_ratio` を返す。分母 0 件（そのドメインへの選択が
     一方の側で 0 件）の場合は `ValueError`（Wilson CI 同様サイレントに 0 除算しない）。
   - 新規: `apply_benjamini_hochberg(p_values: list[float], q: float = 0.05) ->
     list[bool]`。標準的な BH step-up 手順（p 値を昇順ソートし，最大の `i` で
     `p_(i) <= (i/m)*q` を満たすものを見つけ，それ以下の順位を有意とする）。入力順序を
     保った `bool` のリストを返す。空リストは空リストを返す。
   - `import` 追加: `from scipy.stats import fisher_exact`。
   - `pyproject.toml`: `dependencies`（6-14行付近）に `"scipy>=1.18"` を追加する。現状
     scipy は scikit-learn 経由の間接依存でしか入っておらず（`uv run python -c "import
     scipy"` は通るが `pyproject.toml` に宣言がない），`metrics.py` が直接 import する以上
     明示的な直接依存として宣言すべきである。`uv add scipy` を実行して `uv.lock` を更新する
     （既にインストール済みの 1.18.0 がそのまま解決される見込みで，大きな依存変更は
     発生しないはずだが，実装フェーズで `uv.lock` の diff を確認すること）。

4. `tests/test_metrics.py`
   - `compute_domain_recall_mcnemar_test`：小さなトイデータ（3〜4行，domain 該当行のみ）で
     discordant 件数・p 値が手計算と一致することを確認するテスト，および
     `compute_mcnemar_test` と同様の id 不一致 `ValueError` テストを追加する。
   - `compute_domain_precision_fisher_test`：2×2 のトイデータで `scipy.stats.fisher_exact`
     を直接呼んだ場合と同じ p 値になることを確認するテスト，および分母 0 件時の
     `ValueError` テストを追加する。
   - `apply_benjamini_hochberg`：教科書的な既知の例（例: p値 `[0.01, 0.02, 0.03, 0.04, 0.20]`，
     `q=0.05` で先頭 何件が有意になるか）で結果が一致することを確認するテスト，全て非有意な
     ケース，空リストのテストを追加する。
   - 既存の `test_compute_mcnemar_test_*` 系テストは，`_mcnemar_from_correctness` への
     抽出後も `compute_mcnemar_test` の外部インターフェースは変わらないため無変更で通る見込み
     （実装フェーズで実行して確認する）。

**評価手順**:

1. 新分類器の学習: `uv run python -m scripts.train_domain_classifier
   --train-data data/classifier_train.jsonl --embedding-model nomic-embed-text
   --ollama-host <live node> --output models/domain_classifier_isotonic.joblib`
   （Iter29 と同じくライブな ollama ノード 1 台への embedding のみ）。
2. 「較正前」データは Iter29 と同一の `results/20260731_162722/results.jsonl`
   （Iter28 実測，fallback 0/1600）をそのまま使う。**再実行しない**（Iter29・Iter30 を
   同じ較正前基準で揃えて比較可能にするため）。
3. 「較正後」データは `scripts/evaluate_classifier_calibration.py` で 1600 問を再 embedding し，
   `--classifier models/domain_classifier_isotonic.joblib --output
   results/iter30_calibrated_predictions.jsonl` として生成する。
4. `metrics.py:compute_ece(n_bins=10)` を較正前・較正後の両方に同一の bin 設定で適用し，
   ECE を比較する（Iter29 の較正前基準 0.19336 を流用し，再計算しない）。
5. top1_accuracy を較正前・較正後で算出し，新旧の正誤ペアで `compute_mcnemar_test`
   （全体，α=0.05）を行う（Iter29 の手順5と同一）。
6. **per-domain 非退行チェック（今回から運用変更，調査(Iter30) 申し送り2）**: 全10ドメイン
   について，(a) recall は `compute_domain_recall_mcnemar_test` （計10検定），(b) precision は
   `compute_domain_precision_fisher_test` （計10検定）を実施し，計20個の p 値を集めて
   `apply_benjamini_hochberg(p_values, q=0.05)` を一括適用する。adjusted 有意（BH 通過）かつ
   方向が悪化（較正後の点推定 < 較正前の点推定）である指標のみを「統計的に有意な退行」と
   判定する（有意だが改善方向のものは退行ではない）。全10ドメイン・20指標を**最初から**
   算出し，Iter29 のように事後で穴埋めしない。
7. **isotonic 特有の実装確認チェックリスト（調査(Iter30) 申し送り3，`probabilities`
   フィールドを使って算出）**: 較正後の 1600 行について，(a) `probabilities` の値のいずれかが
   厳密に `0.0` または `1.0` になっている行数，(b) 10 クラス全てが `0.1` に近い
   （`math.isclose(p, 0.1, abs_tol=1e-9)` 相当）uniform fallback 行数，(c) 選択ドメインの
   confidence と同一の値を持つ他ドメインが存在する行の割合（tie 率）。特に legal ドメインの
   行を優先して個別集計する。これらは判定基準ではなく必須報告項目。
8. 新旧 classifier の argmax 不一致件数（flip rate）を Iter29 と同じ定義で報告する（必須報告
   項目，判定基準ではない）。

**成功条件（d0003 X9．AND 条件）**:

1. ECE（手順2・4，較正前基準 0.19336 に対する較正後の値，`n_bins=10`）が **0.150 以下**
   であること。
2. top1_accuracy（手順5）が旧分類器（Iter28 実測 0.585）に対し McNemar 検定で有意に悪化
   していない（p>=0.05，または新側が改善方向）こと。Iter29 と同じく理論的仮定ではなく
   実測比較で判定する。
3. **per-domain 非退行（手順6，3段構成）**: 20指標（10ドメイン×precision/recall）の p 値へ
   BH 補正（q=0.05）を適用した結果，adjusted 有意かつ悪化方向の指標が **0 件**であること。
   （Iter29 で用いた「CI 下限の単純比較」は多重比較補正なしで 20 指標中 9 指標が偽陽性に
   なることが判明済みのため，今回はこの基準を使わない．CI そのものは参考情報として引き続き
   算出・報告する。）
4. 手順7のisotonic特有チェックリスト（0/1張り付き・uniform fallback・tie率）とflip rate
   （手順8）は，成功・失敗の判定基準ではなく必須報告項目として全件記録する。

**目標未達時の次点候補（次イテレーション向けメモ，今回の計画には含めない）**: 調査(Iter30)
申し送り4のとおり，isotonicがECE 0.150以下に届かない場合，`method='temperature'`
（sklearn>=1.8，本リポジトリの`scikit-learn==1.9.0`で利用可能，top1_accuracy不変が理論的に
保証される）を次点候補として検討する。ただしtemperatureは多クラス全体で単一のTを学習するため
ドメインごとの較正の柔軟性はsigmoid/isotonicより低い点は留保として記録しておく。

**人間判断が必要な論点**: 新規追加なし。Y2（`confidence_threshold`の二重責務分離，スキーマ
変更）着手前のユーザー確認は backlog B49・B50 の既存の申し送りのまま。較正済み分類器の本番
反映可否も，isotonicが成功条件（本計画の1-3すべて）を満たした場合に改めてその時点で判断する
（今回のイテレーションで本番アーティファクトを置き換える判断は行わない）。

---

### 実装 (Iter30)

計画どおり単一レバー（`classifier_calibration=isotonic`）のみを実装した．`config.yaml` は
変更していない（`git diff config.yaml` が空であることを確認済み）。

**変更ファイル**:

1. `scripts/train_domain_classifier.py`
   - `_CALIBRATION_METHOD = "sigmoid"` → `"isotonic"` に変更。`_CALIBRATION_CV = 5` は無変更。
   - `_CALIBRATION_METHOD` 直上のコメントを，isotonic を選ぶ理由（Iter29 の Platt が
     ECE 絶対閾値未達で partial 判定・config.yml の `classifier_calibration` レバーが
     isotonic を次点候補として登録済み・backlog B50）と，調査(Iter30) が確認した追加リスク
     （isotonic はノンパラメトリックで自由度が事実上サンプル数に等しく sigmoid より
     過学習しやすい／`out_of_bounds="clip"` により legal の held-out データが極端値に
     張り付きやすい）に更新。
   - モジュール冒頭 docstring の `Iter29, classifier_calibration=platt` を
     `Iter30, classifier_calibration=isotonic` に更新。
   - `train_classifier()` の docstring を `method="isotonic"` の説明に更新し，isotonic
     特有の注意点（tie・0/1 張り付き・uniform fallback，調査(Iter30) 分かったこと(3)）を
     追記。
   - 出力アーティファクト名（`--output` の既定値）は変更していない
     （`models/domain_classifier.joblib` のまま）。計画どおり，実験フェーズでの実行時に
     `--output models/domain_classifier_isotonic.joblib` を明示指定することで本番
     アーティファクトを上書きしない運用とする（CLI 引数のみで対応可能なため，スクリプト
     側の既定値変更は不要と判断）。
   - `tests/test_train_domain_classifier.py` は無変更で実行し，pass することを確認した
     （`method` を変えても `StratifiedKFold` の分割条件は変わらないため）。
2. `scripts/evaluate_classifier_calibration.py`
   - `predict_calibrated_rows()` が返す各行の辞書に `"probabilities"`
     フィールド（`{domain: float(p) for domain, p in zip(classes, probabilities)}`，10
     ドメイン全ての確率）を追加。既存フィールド（`id`／`expected_domains`／
     `selected_domain`／`confidence`）は無変更。
   - モジュール冒頭 docstring に，isotonic 特有のチェックリスト（0/1 張り付き・uniform
     fallback・tie 検出）に全クラスの確率ベクトルが必要なため `probabilities` を追加した，
     という理由を追記。CLI 引数は無変更。
3. `metrics.py`
   - `compute_mcnemar_test`（226-261行相当）から discordant-pair の χ²／p 値計算を
     `_mcnemar_from_correctness(correct_a: dict[str, bool], correct_b: dict[str, bool]) ->
     dict[str, float]` として切り出し，`compute_mcnemar_test` はこのヘルパーを呼ぶよう変更
     （外部インターフェースは無変更）。
   - 新規 `compute_domain_recall_mcnemar_test(results_a, results_b, domain) ->
     dict[str, float]`：`id` が一致する行のうち `domain in expected_domains` の行だけを
     サブセットし，`selected_domain == domain` を正誤として `_mcnemar_from_correctness`
     に渡す。id 集合の不一致は `ValueError`。
   - 新規 `compute_domain_precision_fisher_test(results_a, results_b, domain) ->
     dict[str, float]`：2×2 分割表 `[[tp_a, selected_a - tp_a], [tp_b, selected_b - tp_b]]`
     （`tp` = `selected_domain == domain and domain in expected_domains`）を作り
     `scipy.stats.fisher_exact`（両側）で `p_value`・`odds_ratio` を返す。片側の選択数が
     0 件の場合は `ValueError`。
   - 新規 `apply_benjamini_hochberg(p_values: list[float], q: float = 0.05) ->
     list[bool]`：標準的な BH step-up 手順。入力順序を保った `bool` のリストを返す。
     空リストは空リストを返す。
   - `import` 追加: `from scipy.stats import fisher_exact`。
   - `pyproject.toml` の `dependencies` に `"scipy>=1.18"` を追加し，`uv add "scipy>=1.18"`
     で `uv.lock` を更新した。`uv.lock` の diff を確認したところ，`scipy` パッケージの
     エントリ追加自体は想定どおり小さいが，`lora` extra 配下の nvidia/cuda 系パッケージの
     プラットフォームマーカーが再解決の副作用で一部変化していた（バージョン変更は一切なし，
     `win32`/`AMD64` 条件が一部エントリから外れる形の書き換えのみ）。`git stash` で
     `pyproject.toml` を元に戻した状態で `uv lock --check` を実行し，変更前の `uv.lock` が
     既に最新状態であったこと（＝この差分が scipy 追加以前からの潜在的なズレではなく，
     今回の relock で新たに解決された結果であること）を確認済み。`uv add` でも手動編集＋
     `uv lock` でも同一の差分になることを確認しており，`lora` extra は既定ではインストール
     されない（`uv sync --extra lora` 時のみ関与）ため，本プロジェクトの通常の依存関係
     解決には影響しない。
4. `tests/test_metrics.py`
   - `compute_domain_recall_mcnemar_test`：既存の
     `test_compute_mcnemar_test_matches_known_chi_square_critical_values` と同じ discordant
     カウント（29／15）を `domain="legal"` のサブセットに対して再現するトイデータ（加えて
     サブセット対象外のノイズ行2件が結果に影響しないことも確認），および id 不一致
     `ValueError` テストを追加。
   - `compute_domain_precision_fisher_test`：2×2 トイデータ（`[[6, 4], [2, 6]]`）で
     `scipy.stats.fisher_exact` を直接呼んだ場合と `odds_ratio`／`p_value` が一致することを
     確認するテスト，および分母 0 件（片側でドメインが一度も選択されない）時の `ValueError`
     テストを追加。
   - `apply_benjamini_hochberg`：教科書的な既知の例（p値 `[0.01, 0.02, 0.03, 0.04, 0.20]`，
     `q=0.05` で先頭4件が有意）のテスト，全て非有意なケース，空リストのケースを追加。
   - 既存の `test_compute_mcnemar_test_*` 系テストは無変更で実行し，pass することを確認した。

**テスト結果**: `uv run pytest -q` → 218 passed, 2 skipped（既存のスキップ2件は本変更と
無関係）。新規追加した8件のテストを含め全て pass。

**lint/format**: `uv run ruff check metrics.py scripts/train_domain_classifier.py
scripts/evaluate_classifier_calibration.py tests/test_metrics.py
tests/test_train_domain_classifier.py pyproject.toml` → All checks passed。
`uv run ruff format --check` は `metrics.py`／`tests/test_metrics.py` が未整形と報告されたが，
これは Iter30 の変更前から repository 全体で `ruff format` 規約に沿っていなかった既存差分
であることを `git stash` で変更前の状態に戻して確認済み（単一レバー原則に従い，本イテレーション
の変更範囲外として触っていない）。新規追加した `scripts/evaluate_classifier_calibration.py`
の1行のみ未整形だったため，その箇所だけ手動で1行に整形し直し，整形済みであることを再確認した。
リポジトリ全体の `ruff check .` に残る2件（`scripts/prepare_lora_training_data.py`）は
Iter29 から既知の，本変更と無関係な既存差分であり，単一レバー原則に従い今回も触っていない。

**config.yaml の確認**: `git diff --stat -- config.yaml` が空であることを確認し，一切
変更していないことを確認した。

**実験を開始してよい状態か**: はい。コード変更は完了し，型注釈・テスト・lint とも整合。
フェーズ4では，(1) `scripts/train_domain_classifier.py` で
`models/domain_classifier_isotonic.joblib` を1台のライブ ollama ノードへの embedding
呼び出しで新規生成（本番 `models/domain_classifier.joblib` は上書きしない），(2)
`scripts/evaluate_classifier_calibration.py` で 1600 問を再 embedding して較正後の予測
JSONL（`probabilities` フィールド付き）を生成，(3) `metrics.py` の既存関数群＋新規3関数で
較正前（`results/20260731_162722/results.jsonl`，再実行不要）と較正後を比較し，成功条件
1-4（ECE≤0.150・McNemar 非退行・per-domain 20指標への BH 補正非退行・isotonic 特有チェック
リストと flip rate の報告）を実測すればよい。

---

### 実験 (Iter30)

計画どおり実機 1600 問本走は行わず，Iter29 と同一の SSH ローカルポートフォワード
（`127.0.0.1:11435 -> wafl500:11434`，`ssh -fNT -L 11435:localhost:11434 wafl500`，
既存プロセスが起動済みで新規に張り直す必要はなかった。事前に `curl` で
`http://127.0.0.1:11435/api/tags` が疎通することを確認済み）経由の embedding 呼び出しのみで
較正前後の比較データを揃えた。

1. 新分類器の学習:
   ```
   uv run python -m scripts.train_domain_classifier \
     --train-data data/classifier_train.jsonl \
     --embedding-model nomic-embed-text \
     --ollama-host 127.0.0.1 --ollama-port 11435 \
     --output models/domain_classifier_isotonic.joblib
   ```
   標準出力: `[train_domain_classifier] wrote models/domain_classifier_isotonic.joblib
   (n_samples=1427, classes=[...10ドメイン...])`。実行時間 126.51 秒（実測，Iter29 の Platt
   124.09 秒とほぼ同水準）。`models/domain_classifier_isotonic.joblib` を新規生成し，
   本番 `models/domain_classifier.joblib` のタイムスタンプ（Jul 27 16:08）が今回の実行後も
   変化していないこと（＝上書きされていないこと）をファイルシステム上で確認した。
2. 較正後データ生成:
   ```
   uv run python -m scripts.evaluate_classifier_calibration \
     --dataset data/dataset.jsonl \
     --classifier models/domain_classifier_isotonic.joblib \
     --embedding-model nomic-embed-text \
     --ollama-host 127.0.0.1 --ollama-port 11435 \
     --output results/iter30_calibrated_predictions.jsonl
   ```
   標準出力: `[evaluate_classifier_calibration] wrote 1600 rows
   (classifier=models/domain_classifier_isotonic.joblib)`。実行時間 136.74 秒。出力
   JSONL は計画どおり `probabilities` フィールド（10 ドメイン全ての確率）付きで 1600 行生成された。
3. 較正前データは計画どおり `results/20260731_162722/results.jsonl`（Iter28 実測，fallback
   0/1600）を再実行せずそのまま使用。新旧2ファイルの `id` 集合が完全一致することを確認済み
   （`{r["id"] for r in before} == {r["id"] for r in after}` が `True`）。

**異常の有無**: なし。両スクリプトとも例外・タイムアウト・リトライなく正常終了した。実機呼び出し
は wafl500（192.168.15.100:11434）への embedding のみ（`nomic-embed-text`，計 3027 回：
1427+1600），LLM 生成・probe・dispatch は一切発生していない。

---

### 分析(実行) (Iter30)

`metrics.py` の既存関数（`compute_ece`／`compute_top1_accuracy`／`compute_mcnemar_test`／
`compute_precision_recall_per_domain`）と新規3関数（`compute_domain_recall_mcnemar_test`／
`compute_domain_precision_fisher_test`／`apply_benjamini_hochberg`）を呼ぶ一時スクリプトで
較正前（`results/20260731_162722/results.jsonl`）と較正後（`results/iter30_calibrated_predictions.jsonl`，
各 1600 行）を比較した（判定はここでは行わず，数値のみを機械的に集計する）。

**手順4: ECE（`n_bins=10` で統一）**

- 較正前: **0.193357556998477**（`state.json` の `e29_results.ece_before` と同一値，再計算不要の
  ところ実測でも一致することを確認）
- 較正後: **0.1214241251658703**
- 改善幅: 0.071933（較正前→較正後で減少，改善方向）
- 0.150 との比較: 較正後 0.1214 < 0.150

**手順5: top1_accuracy（1600問，`expected_domains` との一致率）**

- 較正前: 0.585000（Iter28 実測と同一値）
- 較正後: 0.593750
- 差分: +0.008750（較正後が高い）

**手順5: McNemar 検定（全体，対応のある2条件比較，較正前=A・較正後=B，連続性補正あり）**

- discordant_a_only（較正前のみ正解）: 72
- discordant_b_only（較正後のみ正解）: 86
- discordant_pairs（合計）: 158
- chi2_statistic: 1.0696202531645569
- p_value: **0.30103123736220994**（α=0.05 で有意差なし。較正後が正解に転じた行(86)が誤りに
  転じた行(72)を上回り，方向としては改善寄り）

**手順8: flip rate（argmax が変わった行の割合，`id` で対応付け，Iter29 と同じ定義）**

- **229/1600 = 0.143125**（14.3125%）。Iter29（Platt，11.0%）より高い（isotonic の方が
  柔軟な分だけ argmax の入れ替わりが多いという調査(Iter30) の事前予想と整合）。

**手順6: per-domain 非退行チェック（10ドメイン×recall/precision＝20指標，BH補正 q=0.05）**

全20指標の点推定（較正前→較正後）と個別検定のp値，BH補正後の有意フラグ：

| domain | metric | before | after | p_value | BH有意 | 方向 |
|---|---|---|---|---|---|---|
| business_economics | recall | 0.5179 | 0.5357 | 0.546494 | 否 | 改善 |
| business_economics | precision | 0.4328 | 0.4688 | 0.479833 | 否 | 改善 |
| computer_science | recall | 0.5417 | 0.5595 | 0.627626 | 否 | 改善 |
| computer_science | precision | 0.5987 | 0.5529 | 0.430737 | 否 | 悪化 |
| education | recall | 0.4059 | 0.5000 | 0.000796 | **有** | 改善 |
| education | precision | 0.4631 | 0.4315 | 0.585896 | 否 | 悪化 |
| general | recall | 0.5488 | 0.5427 | 1.000000 | 否 | 悪化 |
| general | precision | 0.6522 | 0.6899 | 0.518329 | 否 | 改善 |
| history_culture | recall | 0.6667 | 0.7024 | 0.211300 | 否 | 改善 |
| history_culture | precision | 0.7320 | 0.6705 | 0.231070 | 否 | 悪化 |
| legal | recall | 0.5833 | 0.5889 | 1.000000 | 否 | 改善 |
| legal | precision | 0.7500 | 0.7852 | 0.568393 | 否 | 改善 |
| mathematics | recall | 0.6190 | 0.6786 | 0.009375 | 否 | 改善 |
| mathematics | precision | 0.7075 | 0.6867 | 0.713168 | 否 | 悪化 |
| medical | recall | 0.4831 | 0.3820 | **0.000144** | **有** | **悪化** |
| medical | precision | 0.4725 | 0.5231 | 0.421841 | 否 | 改善 |
| natural_science | recall | 0.5655 | 0.5833 | 0.662521 | 否 | 改善 |
| natural_science | precision | 0.5135 | 0.5475 | 0.530141 | 否 | 改善 |
| social_science | recall | 0.5774 | 0.5238 | 0.052345 | 否 | 悪化 |
| social_science | precision | 0.6340 | 0.6984 | 0.308685 | 否 | 改善 |

BH（q=0.05）通過（adjusted 有意）は20指標中2件: `education_recall`（p=0.000796，改善方向）・
`medical_recall`（p=0.000144，**悪化方向**）。`medical_recall` の内訳:
discordant_a_only=19（較正前のみ正解）・discordant_b_only=1（較正後のみ正解）・
discordant_pairs=20・chi2=14.45。BH 補正後も有意かつ悪化方向の指標は **1件**（`medical_recall`）。

**手順7: isotonic 特有の実装確認チェックリスト（`probabilities` フィールドを使用，1600行対象）**

- (a) 確率のいずれかが厳密に `0.0` または `1.0` になっている行数: **1311/1600**（うち厳密に
  `1.0` を含む行は **0 件**，厳密に `0.0` を含む値の総数は全行合計で **2123 個**）。
- (b) 10クラス全てが `0.1` に近い（`math.isclose(p, 0.1, abs_tol=1e-9)`）uniform fallback 行数:
  **0/1600**。
- (c) 選択ドメインの confidence と同一の値を持つ他ドメインが存在する行の割合（tie率，
  厳密な浮動小数点一致で判定）: **0/1600（0.0000%）**。

legal ドメインの個別集計（優先報告）:
- `legal` が `expected_domains` に含まれる行（180行）のうち，確率に厳密な `0.0`/`1.0` を含む
  行数: **158/180**。
- `legal` が `selected_domain` の行（135行）のうち，同条件: **121/135**。
- legal の uniform fallback 行数: **0/180**。tie 行数: **0/180（0.0000%）**。

**使用データ**:

- 訓練データ: `data/classifier_train.jsonl`（1427件，legal 77件・他9ドメイン各150件，Iter29と同一）
- 評価データセット（再embedding対象）: `data/dataset.jsonl`（1600件）
- 較正前の実行結果: `results/20260731_162722/results.jsonl`（Iter28実測，1600行，再実行なし）
- 較正後の実行結果（新規生成）: `results/iter30_calibrated_predictions.jsonl`（1600行，
  `probabilities`フィールド付き）
- 新規モデルアーティファクト: `models/domain_classifier_isotonic.joblib`（本番
  `models/domain_classifier.joblib`は無変更のまま，タイムスタンプで確認済み）

**実行時間・実機呼び出しの有無**:

- `train_domain_classifier.py`: 126.51秒（1427回のembedding呼び出し）
- `evaluate_classifier_calibration.py`: 136.74秒（1600回のembedding呼び出し）
- 実機呼び出しはwafl500（192.168.15.100:11434）へのembeddingのみ（`nomic-embed-text`），
  計3027回。LLM生成・probe・dispatchは一切発生していない。
- 接続経路はIter29と同一の既存SSHローカルポートフォワード（`127.0.0.1:11435` ←
  `wafl500:11434`）をそのまま流用。新規に張り直す必要はなく，実行中のログ・エラーに異常
  なし（例外・タイムアウト・リトライなし，両スクリプトとも正常終了メッセージを出力）。

**state.json更新**: `status: waiting_experiment`（開始時，`experiment_dir:
results/iter30_calibrated_predictions.jsonl`・`experiment_deadline`設定）→`running`
（完了時，`experiment_dir`/`experiment_deadline`を`null`に戻した）。`e30_results`への数値
記録・`judgment`確定はフェーズ5b（rc-analyst）に委ねる（本フェーズでは数値の良否判定は行わない）。

---

### 分析(解釈) (Iter30)

**成功条件（d0003 X9，AND条件，計画(Iter30)節）との照合**

1. **ECE ≤ 0.150**: 成立。較正後 0.121424 は較正前 0.193358 から −7.19pt（相対37.2%減）であり，
   Iter29（Platt，0.16751）より 4.6pt 深く改善し，目標にも 2.86pt の余裕をもって到達している。
   ルーティングは決定論的（config.yml success_criteria (5)）であり，同一 1600 問・同一
   embedding モデルに対し分類器のみを変えた比較のため，この差分はノイズではなく較正手法の
   変更そのものが生んだ実測値と判断してよい（Iter29 と同じ根拠）。
2. **top1_accuracy 非退行**: 成立。McNemar p=0.301031（α=0.05 で有意差なし）であり，
   discordant_b_only（較正後のみ正解，86）が discordant_a_only（較正前のみ正解，72）を
   上回っているため方向としては改善寄りである。Iter29（p=0.139，b_only=67>a_only=50）と
   同種の非退行パターンが再現している。
3. **per-domain 20指標のBH補正後・悪化方向の有意指標0件**: **不成立**。BH（q=0.05）通過は
   `education_recall`（p=0.000796，改善方向）と`medical_recall`（p=0.000144，悪化方向，
   0.4831→0.3820）の2件で，悪化方向で通過したのは`medical_recall`の1件。discordant内訳は
   a_only（較正前のみ正解）=19・b_only（較正後のみ正解）=1・discordant_pairs=20・chi2=14.45
   であり，19:1という非対称性は補正後もなお際立って大きい。
4. isotonic特有チェックリスト（0/1張り付き1311/1600・uniform fallback 0件・tie率0%）と
   flip rate（229/1600=14.3125%，Plattの11.0%より高い）は報告事項として確認した（詳細は下記）。

**medical_recall悪化（BH通過）の解釈 — Iter28・Iter29との異同**

まず事実確認として`data/classifier_train.jsonl`を実際に確認した結果，**medicalの訓練データは
150件であり，legal（77件）のような少数派ドメインではなく，他8ドメインと同数の多数派ドメインで
ある**（`business_economics`〜`social_science`まで全て150件，`legal`のみ77件）。これは
Iter29の申し送り（「computer_science/mathematicsは150件の多数派ドメインなのに偽陽性で
引っかかった」）が示唆したとおり，訓練データ量の多寡だけではmedical_recallの悪化を説明できない
ことを裏付ける事実である。

次に，Iter29までとの決定的な違いは**検定の厳格さ**にある。Iter29の per-domain 非退行チェックは
「較正前後のCI下限の単純比較」という多重比較補正なしの基準であり，事後の追加分析（B50）で
20指標中9指標が該当したものの全て区間重複で統計的に非有意な偽陽性だったと判明した。今回は
Iter29の教訓を踏まえ，(a) recallはドメイン別McNemar検定，(b) precisionは2標本Fisher正確検定，
(c) 計20個のp値へBH法を**最初から**適用するという，より厳格な手順で臨んだ。その結果として
残った`medical_recall`1件は，Iter29の9件のような「緩い基準でしか引っかからない偽陽性」とは
性質が異なり，**多重比較を補正してもなお統計的に有意な，再現性のある効果**である。BH法は
20検定という規模で偶然生じる誤検出（FDR）を5%以下に抑えるよう設計されており，それでも
生き残った1件は，Iter29の legal recall 低下（追加分析で相対化された）よりも判定上の重みが
大きいと考えるべきである。

precisionは同時に0.4725→0.5231へ改善しており，表面上はIter28のgeneralドメイン
（recall低下・precision大幅改善が同一212行内で表裏一体）と類似する。しかし規模を比較すると
性質が異なる。Iter28のgeneral precisionは0.3134→0.6522（+33.9pt）という recall 低下を
大きく上回る改善であり，かつ「fallbackの送り先が常にgeneralだった」というレバー変更に
数学的に内在する構造（fallback廃止で流入経路が変わるのは必然）が機序として明確だった。
今回のmedicalはprecision改善が+5.06pt（0.4725→0.5231）にとどまり，recall悪化の−10.11pt
（0.4831→0.3820）の半分程度に過ぎない。かつdiscordantの非対称性（a_only=19 : b_only=1）は
Iter28のgeneral（fallback対象212行内の再配分という機構が既知）のような「レバー自体が
生む必然的な流入経路変化」では説明できず，isotonic較正曲線がmedicalクラス固有にどう
振る舞ったかを調べる必要がある。

**追加検証（数値再計算，本フェーズで実施）**: `results/20260731_162722/results.jsonl`
（較正前）と`results/iter30_calibrated_predictions.jsonl`（較正後，`probabilities`
フィールド付き）から，medical_recallが悪化した19行（discordant a_only）を個別に確認した。

- 19行のうち，較正後の`probabilities`でmedicalクラスの値が厳密に`0.0`になっている行は
  **0件**であり，0/1張り付き（isotonic特有チェックリストの(a)）が直接の原因ではない。
  むしろ19行の多くは較正後もmedicalが2位相当の確率（0.21〜0.39）を保持しており，僅差
  （margin 0.003〜0.13）で他ドメイン（`computer_science`5件・`natural_science`4件・
  `education`3件・`history_culture`3件・`business_economics`2件・`mathematics`1件・
  `social_science`0件他）に argmax を奪われている。
- 19行中，較正前の`confidence`（medicalの確信度）が0.87〜0.98という高い値だった行が3件
  含まれており，較正前は明確にmedicalが最有力だったにもかかわらず，較正後は0.35〜0.39まで
  値が圧縮されて argmax を失っている。
- **1600行全体でmedicalクラスの較正後確率の最大値は0.7062であり，他9ドメインの最大値
  （0.7496〜0.8795）を全て下回る**。medicalが較正後に到達しうる確信度の「天井」自体が，
  他ドメインより体系的に低く抑えられている（`business_economics`最大0.8238，
  `history_culture`最大0.8795 など）。selectedとして選ばれた回数も較正前182件→較正後130件
  （−28.6%）へ減少しており，このドメインだけisotonic較正曲線がクラス全体で系統的に
  スコアを下方へ圧縮している疑いが強い。
- 一方，選択された行の`confidence`自体（ECEの算出対象）に厳密な0.0/1.0は1件もなく
  （全1600行で確認済み），isotonic特有チェックリストの「uniform fallback」も0件であるため，
  ECE 0.121424の改善はmedicalの0/1張り付きのような病理によって水増しされたものではない。

以上から，medical_recallの悪化は「isotonicの0/1張り付き」や「legalのような小標本held-out
較正の不安定性」という調査(Iter30)が事前に警戒していた2つの機序のいずれでもなく，
**medicalクラスの isotonic 較正曲線がcv=5較正foldにおいて系統的にスコアを圧縮し，
他ドメインとの僅差の argmax 競争で構造的に不利になる**という，訓練データ量では説明できない
第3の機序である可能性が高い。この機序は事前の投資フェーズでは想定されておらず，**次回
isotonicを継続検討する場合は，legalだけでなくmedicalのように多数派ドメインでも同種の
較正曲線圧縮が起こりうることを踏まえ，cv=3等の感度分析やドメイン単位の較正曲線可視化を
対象ドメインを限定せず行うべき**という新たな示唆を得た。

なお，legalドメイン（Iter29でrecall低下が唯一の懸念だった小標本ドメイン）は今回
recall 0.5833→0.5889へ**改善**しており，isotonicが小標本ドメインで一律に悪化を招くという
調査(Iter30)の事前予想（sigmoidより過学習しやすい，legalが最も影響を受けやすい）はむしろ
反証された。isotonicの実際のリスクは事前に警戒していたlegalではなく，多数派ドメインの
medicalという想定外の箇所に現れており，この点は仮説と実測の不一致として明示しておく。

**isotonic特有チェックリスト（0/1張り付き1311/1600＝82%）の判定への反映**

sklearn公式ドキュメントが警告する「isotonicはties/0-1張り付きを生みやすい」という調査(Iter30)
の申し送りは，非選択クラスの確率に関しては実測でも裏付けられた（1311/1600行で少なくとも
1クラスが厳密0.0，legalは158/180行と特に高率）。ただし上記の追加検証で確認したとおり，
**この0/1張り付きは主に非選択（劣勢）クラスに生じており，ECEの算出対象である選択ドメインの
confidence自体には1件も及んでいない**（厳密0.0/1.0の選択行は0/1600）。したがって，
「ECEの見かけ上の改善が個々の予測の信頼性を代償にしている」という懸念は，少なくとも
ECEの数値そのものについては支持されない。一方で，非選択クラスの0/1張り付きが82%という
高率で生じている事実自体は，isotonic較正曲線の区分定数性・ノンパラメトリックな自由度の高さ
（調査(Iter30)分かったこと(1)）を裏付ける実装上の懸念として記録に値し，medical_recall悪化の
根本原因（較正曲線の系統的圧縮）と同根の現象（held-out較正データが少ない状態でのisotonic
回帰の不安定な区分定数フィット）である可能性が高い。判定上は「ECEの数値を歪める」形では
現れていないが，「特定ドメインの較正曲線が予測不能に歪みうる」という構造的リスクの実例として
medical_recall悪化の解釈に反映させる。

**Iter20（E3）precedentに関する留保**: config.ymlの申し送りが参照する「Iter20 partial運用実績」
は，本journal内の訂正1（環境修復セクション）で，Iter20当時の判定（「効果あり」）自体が
Iter17（supervised_classifier導入）との交絡により事後的に取り下げられ，D1（判定保留）へ
再分類されている経緯がある。したがって「主基準改善・副基準悪化ならpartial」という運用実績の
参照先としては，交絡のないIter29（同一AND条件構造・同一レバー系列）の方がIter30との対称性が
直接的であり，本判定はIter29を主たる比較対象とし，Iter20は参考情報にとどめる。

**総合判断（rc-analyst 提案，確定は rc-reflector）: partial（部分的採用）**

根拠:

1. 成功条件1・2は明確に成立し，特にECEはIter29のPlattを大きく上回る改善で目標に十分な余裕を
   もって到達している。この点はisotonicへの切り替えが「ECE改善」という当初目的に対し
   Platt以上に有効だったことを裏付ける。
2. 成功条件3は字義通り不成立である。BH補正という，Iter29の教訓を踏まえて最初から導入した
   厳格な多重比較補正の下でもなお生き残った`medical_recall`の悪化は，Iter29のlegal recall
   低下（事後分析で多重比較アーティファクトと判明）と同列には扱えない。訓練データ量では
   説明がつかず（medicalは150件の多数派），かつ0/1張り付きという既知のisotonic病理でも
   直接説明できず（19行中0件），較正曲線のクラス固有の系統的圧縮という，事前に想定していな
   かった機序で生じている。discordantの非対称性（19:1）とprecision改善幅（+5.06pt）が
   recall悪化幅（−10.11pt）の半分程度に留まることを踏まえると，Iter28のgeneralドメインの
   ような「レバーに内在する必然的トレードオフ」として判定を覆さない扱いにするのは根拠が
   弱い。
3. 以上を総合すると，「ECE目標達成」という主目的は明確に成立し，「per-domain非退行」という
   副次条件は統計的に確認された1件の悪化により不成立という，Iter29と同型（AND条件の一部が
   未達）だが**逆方向**の未達パターンである。Iter29はECE（主目的側）が未達でtop1・
   per-domain（当時は非有意）が成立していたのに対し，今回はECE・top1（主目的側）が成立し
   per-domain（副次条件）が１件のみ有意に未達という非対称な関係にある。いずれの場合も
   「AND条件の一部未達」を理由に，明確な改善方向にある指標の価値を無視して即rejectedとする
   のは実態を捉えず，かつ未解決の懸念（medical_recall）を残したまま本番へ即時反映する
   adoptedも時期尚早である。**partial（部分的採用）を提案する**。

**本番反映（`models/domain_classifier.joblib`の置き換え）についての見解**

**現時点では見送りを推奨する**（最終決定はrc-reflectorとユーザー確認事項）。判断基準:

- 成功条件のAND条件が字義通り未成立（医療ドメインrecallの統計的に有意な悪化）である以上，
  「採用して本番へ反映する」ための閾値をこの一回の実験だけでは満たしていない。
- 単一レバー原則・可逆性の観点では，本番アーティファクトを据え置く（`models/domain_classifier.joblib`
  は変更しない）方が取り消しコストが低い可逆な選択である。今回のisotonic版は
  `models/domain_classifier_isotonic.joblib`として別名生成済みであり，本番を上書きしていない。
- medical_recallの悪化は，Iter29のlegal recallのように「訓練データ拡充（Y5）で解消しうる」
  という見立てが立ちにくい（medicalは既に150件の多数派ドメインであるため）。原因はisotonic
  較正曲線のクラス固有の圧縮という，追加データではなく較正手法・パラメータ側の対処
  （例: `cv=3`感度分析でmedicalの較正foldサンプル数を増やす，または調査(Iter30)申し送り4の
  `method='temperature'`で全クラス共通の単一スカラー変換に切り替えargmax不変を理論的に
  保証する）が必要と考えられる。
- 一方で，ECE目標達成というY4の主目的自体は今回明確に成立しており，isotonicという手法選択
  そのものを棄却する根拠はない。次イテレーションでmedical_recall悪化の原因を狭く切り分ける
  追加検証（`cv=3`感度分析，またはmethod='temperature'との比較）を行い，その結果を踏まえて
  改めて本番反映を判断することを推奨する。

**確信度と追加反復の要否**: 判定の確信度は中程度以上と考える。ECE・top1_accuracyの2条件は
実測・検定とも明確であり追加反復は不要。medical_recallの悪化はBH補正済みで統計的には
確定的（p=0.000144）だが，**その原因（較正曲線のクラス固有圧縮）を裏付ける機序面の追加検証
（cv=3感度分析，較正曲線そのものの可視化）が次回に要る**という点は明記しておく。1回の本走
（n=1）に基づく判定である点はIter28・Iter29と同じ制約であり，ルーティングが決定論的である
以上，再実行によって数値自体が変わることはない。

---

### 考察 (Iter30)

**単一レバーの判定: 部分的採用（partial）を確定**．rc-analyst の「分析(解釈)」節の総合判定
（partial）をそのまま確定させる（覆さない）．判断基準は3点．

1. `classifier_calibration` レバーの成功条件（d0003 X9，計画(Iter30)節）は「ECE≤0.150 **かつ**
   top1_accuracy 非退行 **かつ** per-domain 20指標の BH 補正後の悪化方向有意指標 0 件」の
   AND 条件である．条件1（ECE 0.121424，目標に2.86pt の余裕）・条件2（McNemar p=0.301031，
   方向は改善寄り）は明確に成立するが，条件3は `medical_recall`（p=0.000144，BH 補正後も有意，
   0.4831→0.3820）が1件残っており字義通り不成立である．3条件AND のうち1条件が不成立である以上，
   無条件の adopted は成立しない．
2. 一方，`medical_recall` 以外の19指標はBH補正を通過しておらず，かつECE・top1_accuracyという
   主目的側の2条件は今回のイテレーションの本来の狙い（Iter29のPlattがECE絶対閾値未達だったため
   isotonicで追試する）に対し明確に達成している．「per-domain 1件の統計的に有意な悪化」のみを
   理由に，2条件の明確な達成を無視して rejected とするのは実態を捉えない．
3. rc-analyst が指摘するとおり，今回の `medical_recall` 悪化は Iter29 の legal recall 低下
   （事後の全ドメイン拡張分析で多重比較アーティファクトと判明，backlog B50）とは性質が異なる．
   Iter30 では調査(Iter30) の申し送りに従い，計画段階から BH 補正・ドメイン別 McNemar／Fisher
   検定を組み込んだ厳格な手順で臨んでおり，その手順を通過してなお残った1件は，Iter29 のような
   「緩い基準でしか引っかからない偽陽性」とは重みが異なる．Iter28（E1，fallback廃止，
   backlog B49）の `general` ドメイン recall 低下がレバーに内在する構造的トレードオフ
   （fallback の送り先が常に general という機構）として明確に説明できたのに対し，今回の
   `medical` は precision 改善（+5.06pt）が recall 悪化（−10.11pt）の半分程度にとどまり，
   `discordant` の非対称性（19:1）も「レバーに内在する必然的再配分」では説明できない．
   したがって Iter28 のような「判定を覆さない扱い」の類推は成り立たず，条件3の不成立を
   額面どおり受け止めて partial とすることが妥当である．

以上，rc-analyst の提案どおり **partial（部分的採用）** で確定する．

**本番反映の判断: 見送り（`models/domain_classifier.joblib` は isotonic 版へ置き換えない）**．
rc-analyst の見解をそのまま採用する．

- 成功条件のAND条件が字義通り未成立（`medical_recall` の統計的に有意な悪化）である以上，
  本番へ反映するための閾値をこの一回の実験だけでは満たしていない．
- 単一レバー原則・可逆性の観点では，本番アーティファクトを据え置く方が取り消しコストの低い
  可逆な選択である．今回のisotonic版は `models/domain_classifier_isotonic.joblib` として
  別名生成済みで，本番（`models/domain_classifier.joblib`，タイムスタンプ Jul 27 16:08 のまま
  変化なしを確認済み）を上書きしていない．
- `medical_recall` の悪化は，Iter29 の legal recall 低下と異なり「訓練データ拡充（Y5）で
  解消しうる」という見立てが立ちにくい（medical は既に150件の多数派ドメイン）．原因は
  訓練データ量ではなく較正手法・パラメータ側にあると考えられ，次回以降の追加検証で切り分ける
  べき問題として残す．

**得られた学び（次回以降に活きる非自明な点）**:

1. **isotonic の実際のリスクは，事前に警戒していた小標本ドメイン（legal）ではなく，多数派
   ドメイン（medical）に現れた**．調査(Iter30) の事前予想（「≪1000件で過学習しやすい」＝
   held-out データが最少の legal が最も影響を受けるはず）は，実測で明確に反証された（legal
   recall はむしろ改善 0.5833→0.5889）．訓練データ量という一次元の指標だけでは isotonic の
   ドメイン別リスクを予測できないことが，Iter29（B50，computer_science・mathematics という
   150件ドメインも偽陽性で該当）に続き2イテレーション連続で確認された．**「小標本ドメインが
   最も脆弱」という直感的な仮説は，較正手法の非退行リスクを評価する際の判断材料として単独では
   信頼できない**．次回以降，較正関連のレバーで事前リスクを予測する際は，訓練データ量だけでなく
   （分析(解釈)で行ったような）較正曲線そのものの形状・到達可能な確信度の天井を確認する必要が
   ある．
2. **BH補正という多重比較への厳格な対処は，Iter29の教訓（9件の偽陽性）を実際に解消した**．
   今回は20指標中2件のみがBH通過（悪化方向1件・改善方向1件）で，Iter29の「20指標中9指標が
   該当（うち1件のみ有意）」という状況から大きく改善した．計画段階から検定手順を組み込む
   （事後の穴埋めをしない）運用が機能したことを確認できた．この運用は今後の per-domain
   非退行チェックの標準手順として定着させてよい．
3. **isotonic の 0/1 張り付き（非選択クラスで82%の行に発生）は，ECE の数値そのものを歪めては
   いなかった**（選択ドメインの confidence に厳密な0/1は0件）が，`medical_recall` 悪化の
   根本原因（較正曲線のクラス固有の系統的圧縮）と同根の現象である可能性が高いと分析(解釈)で
   整理された．isotonic のノンパラメトリックな区分定数フィットが，held-out データの少なさと
   組み合わさると，どのドメインが影響を受けるか事前に予測しにくい形で歪みうるという構造的
   リスクを実証したことは，`method='temperature'`（クラスごとの個別較正器を持たず単一スカラー
   のみで変換するため，この種のクラス固有の歪みが構造的に発生しない）を次に検証する強い動機に
   なる．

**次に振る単一レバーの選定: `classifier_calibration=temperature`**

判断基準（`cv=3` 感度分析 と `method='temperature'` のいずれを優先するか）:

- **`cv=3` 感度分析は今回の `medical_recall` 悪化の根本原因に届きにくいと判断した**．
  分析(解釈)で確認したとおり，`medical` は訓練150件の多数派ドメインであり，`cv=5` でも
  `cv=3` でも1foldあたりの較正サンプル数はおよそ30件→50件程度の違いにとどまり，
  sklearn公式が目安とする「greater than ~1000」からは`cv`を3に変えても依然として大きく
  下回ったままである．かつ，19行の悪化事例のうち0/1張り付きが直接の原因だった行は0件で，
  「較正曲線がクラス全体で系統的にスコアを圧縮する」という機序（分析(解釈)节）は fold
  サンプル数の微調整では解消しない構造的な問題である可能性が高い．`cv=3` は同一手法
  （isotonic の OvR 個別較正）内のハイパラ変更にすぎず，今回発見した「OvR 較正がクラス固有に
  予測不能な歪みを生みうる」という根本の懸念には対処しない．
- **`method='temperature'` は，今回発見した根本原因に構造的に対処する**．調査(Iter30) が
  確認したとおり，temperature scaling はクラスごとに個別の較正器を fit せず，単一の
  `_TemperatureScaling` インスタンスのみでロジット全体を単一スカラー T で割る変換であり，
  argmax（top1_accuracy）が理論的に不変であることが sklearn 公式に保証されている．
  これは isotonic／platt が抱える OvR 方式由来の全リスク（クラス固有の曲線歪み・tie・0/1
  張り付き）を構造的に排除する代替であり，今回 `medical` で顕在化した「事前に予測できない
  ドメイン固有の較正曲線圧縮」という新たな懸念に直接応える．
- 留保（調査(Iter30) 申し送り4，分析(解釈) で既出）: temperature は多クラス全体で単一の T
  しか学習しないため，isotonic（0.121424）は元より Platt（0.16751）と比べても較正の柔軟性は
  低く，ECE 改善幅がより小さい可能性がある．Platt でさえ ECE 絶対閾値（0.150）に届かなかった
  経緯があるため，temperature がECE条件を満たせない可能性は相応にある．しかし，それ自体が
  次回イテレーションで検証すべき有益な情報である．仮に temperature が ECE 目標未達であれば，
  「per-domain 非退行のためには OvR 方式の柔軟性を犠牲にできない」という新しい知見が得られ，
  isotonic の運用（例: medical のみ較正を無効化する，較正曲線を平滑化する等）を再検討する
  材料になる．
- **可逆性・独立性**: `classifier_calibration` は既に config.yml の levers に登録済みだが，
  値は `[platt, isotonic]` のみで `temperature` は未登録のため，本フェーズで
  `values: [platt, isotonic, temperature]` へ末尾追記する（可逆な自動判断，スキーマ変更では
  なく既存レバーへの値追加）．`cv`（既定5）・`ensemble`（既定True）は temperature スケーリング
  自体には適用されない sklearn の実装（分析(解釈)出典の `_fit_calibrator` 参照）だが，同じ
  `CalibratedClassifierCV` API 経由で呼ぶため，実装フェーズで挙動を確認すること．

**iteration_name（Iter31）**: 「分類器較正のtemperature scaling方式によるargmax不変性の実証と
ECE目標到達可否の検証」

**要人間判断として残す論点（新規追加なし）**: Y2（`confidence_threshold` の二重責務分離，
スキーマ変更）の着手前ユーザー確認は backlog B49・B50 の既存の申し送りのまま．fallback
設計思想の論文上の位置付け（backlog B48）も未解決のまま据え置く．較正済み分類器の本番反映
可否も，今回は「見送り」という可逆な既定選択を自律判断で行ったのみで，将来いずれかの較正手法が
成功条件を完全に満たした場合の本番反映という判断（本番運用中のルーティング挙動を変える）自体は，
改めてその時点で検討する．

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

