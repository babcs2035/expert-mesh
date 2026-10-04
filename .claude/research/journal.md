## Iteration 106: kNN 類似度の中心化と ABTT を replay で検証する

**実施場所の申告**: 本フェーズで行ったのは，開発ホストでのリポジトリの読み取り，tavily（`tvly search`）による外部調査，開発ホストの CPU での replay（`uv run`．計算済みの埋め込みキャッシュを読むだけ）である．GPU，LLM，wafl500〜509 は使っていない．本走と deploy は行っていない．

**位置づけ**: config.yml の levers は Iter105 で使い切った（B195）．オーケストレータは方向 (i)（複数の正解を許す確信度．ECE の定義が変わる）を保留し，方向 (ii)（rank 1 の決定を変えて top1 を上げる）を採ると自動で判断した（B196）．調査・計画フェーズは 3 回委譲した．1 回目と 2 回目は，subagent が途中でシェルのコマンドを実行できなくなり，ファイルへの書き込みの前に終わった．3 回目で replay を完了した．

**確かめた事実**:

1. 過去の打ち止めとは重ならない．E7 の whitening（`router.py:670-742`）は `routing_method=embedding` の経路でしか読まれず，kNN 成分には一度も当てられていない（d0004:302，journal_archive.md 36553〜36678 行目）．Iter99 の PCA は LR の入力に対する操作である．k は Iter104 で {1, 2, 3, 5, 7} を掃引済みである．
2. kNN 成分の類似度の計算は `knn_interpolated_head.py:122-123` だけである（L2 正規化と内積）．融合埋め込みは (2275, 6656) で，各ブロックのノルムは 1，全体は 2 である．訓練行どうしの cosine は平均 0.618（p5 0.558，p95 0.705）で，狭い帯に詰まっている．

**外部調査（tavily，2026-10-04）**:

- Wang, Chao, Weinberger, van der Maaten, "SimpleShot: Revisiting Nearest-Neighbor Classification for Few-Shot Learning", 2019, arXiv:1911.04623（https://arxiv.org/abs/1911.04623 ）．CL2N（中心化と L2 正規化）の出典である．論文には，L2 正規化の後の中心化は精度をそれ以上は上げない，という趣旨の記述がある．
- Mu, Viswanath, "All-but-the-Top: Simple and Effective Postprocessing for Word Representations", ICLR 2018（https://openreview.net/ ）．平均と上位の主成分を取り除く後処理（ABTT）である．
- "Centering versus Scaling for Hubness Reduction"（ofai.at），"An Evaluation of Hubness Reduction Methods for Entity Alignment"（dbs.uni-leipzig.de）．中心化を hubness の低減策として扱う．

**replay**（`.claude/research/_iter106_replay_center.py`，結果は `_iter106_replay_result.json`）:

- 固定した構成は LR の artifact，k=2，λ=0.3，T=0.9426，外側 5-fold（seed 104）である．平均と主成分は fold の訓練部分だけから求めた．
- 再現の確認: 現行の OOF CV top1 は 0.826813（Iter105 の 0.8268 と一致）．eval の `predict_proba` は現 artifact との最大の絶対差が 0.0 である．
- 選択規則（事前登録）: CV top1 が現行を上回り，かつ 5 fold のうち 4 fold 以上で上回ること．主候補は cl2n，ほかの 4 つは参考候補である．

| 変種 | CV top1 | 上回った fold 数 | 規則 | eval の N_2 の歪度 / 最大出現回数 | eval 共通 3,744 行の top1 | eval の ECE |
|---|---|---|---|---|---|---|
| none（現行） | 0.826813 | — | — | 3.29 / 39 | 0.850160 | 0.0371 |
| cl2n（主候補） | 0.822857 | 1 | 不適合 | 3.68 / 53 | 0.849092 | 0.0353 |
| block_center | 0.822418 | 1 | 不適合 | 3.48 / 49 | 0.847756 | 0.0351 |
| abtt1 | 0.828571 | 2 | 不適合 | 4.30 / 60 | 0.848825 | 0.0347 |
| abtt3 | 0.829011 | 2 | 不適合 | 4.72 / 64 | 0.849359 | 0.0376 |
| abtt10 | 0.829451 | 3 | 不適合 | 6.13 / 78 | 0.848558 | 0.0452 |

- 基準線の本走 `results/20261004_173104` の共通 3,744 行の top1 は 0.849626 である．none の再現に対する McNemar は，どの変種でも p ≥ 0.29 で，C1（20 指標，BH q=0.05）の有意はどの変種でも 0 件である．
- 中心化で hubness が下がるという前提は成り立たなかった．N_2 の歪度と最大出現回数は，どの変種でも現行より大きい．
- 未解決: none の再現と本走とで selected_domain が共通行で 16 行（全行で 22 行）異なる．`predict_proba` は一致するので，差は分類器の外（同順位の並べ方，送出の失敗など）で生じている．原因は調べていない．

**判定**: 事前登録の規則を満たす候補が無いので，新しいレバーを定義しない（要人間判断: 実行可能な新レバーを定義できない）．skill の早期終了の手順により，フェーズ 2 と 3 は行わない．config.yml の levers は変えていない．採用構成は Iter105（`models/domain_classifier.joblib`，MD5 `c7172ad37c10e1082a42481553ae25b0`，T=0.9426）のままである．次の方向は B197 で人間の判断を仰ぐ．

## Iteration 105: kNN 補間分布の温度を再較正する

### 調査 (Iter105)

**実施場所の申告**: 本フェーズで行ったのは，開発ホストでのリポジトリの読み取り，tavily（`tvly search`）による外部調査，開発ホストの CPU での replay（`uv run`．計算済みの埋め込みキャッシュを読むだけ）である．GPU，LLM，wafl500〜509 は使っていない．

**位置づけ**: B193 と C7 の規定（ECE が 0.05 を超えたら，次の反復を校正側に立てる）による．Iter104 の本走 `results/20261004_132607` の ECE は 0.050104 だった．温度は LR の出力（`CalibratedClassifierCV(method="temperature")`）にしか当たっておらず，kNN 分布を 0.3 混ぜた後の分布には当て直していない（Iter104 の学び 4）．

**確かめた事実**:

1. **送出集合が確信度で決まる経路**（`aggregator.py:28-91` を Read で確認）．
   - `node.py:232-239` が `select_dispatch_targets()` を呼び，`confidence_threshold`（`config.yaml:33`，0.0），`dispatch_candidate_threshold`（`:38`，0.0），`gap_threshold`（`:131`，0.36），`gap_max_k`（`:132`，4）を渡す．
   - `aggregator.py:62` が確信度の降順で安定ソートする．`:71` と `:75-77` の閾値はどちらも 0.0 なので，10 ノードがすべて候補に残る．`:87-91` は，隣り合う順位の差が 0.36 未満の間，k を 4 まで増やす．
   - 各ノードの確信度は `classifier.py:69` の `predict_proba` の自ドメイン列で，実体は `knn_interpolated_head.py:99-107` である．温度を変えると argmax は変わらないが，隣り合う順位の差が変わるので，送出集合は直接動く．
   - `results.jsonl` の `confidence` は rank 1 の probe 確信度で，`metrics.py:486-521` の ECE はこれを読む．判定は `selected_domain in expected_domains`（複数の正解を許す）である．
   - replay の `simulate_rows()` は `:87-91` の連鎖を再現している．違いは同順位の扱いだけで，replay はクラスの順，本番は peers.yaml の順に並べる．
2. **単一ラベルの温度と，複数の正解を許す ECE の食い違い**（journal_archive.md 8973 行目）．複合行では，正解が 2 つあるため構造的に過小確信になる．Iter104 の本走では，単一行の符号付きギャップ（平均確信度 − 平均正答）が −0.031，複合行が −0.129 である．
3. **過去の本走の ECE**（`results/*/metrics.json`）: 20260923_150540 0.0551，20260926_221822 0.0327，20260929_192157（Iter101）0.0396，20261002_105743（Iter103）0.0385，20261004_132607（Iter104）0.0501．同じ LR の artifact で走った Iter101 と Iter103 の差は 0.0011 である．
4. **commit b1f5cb0**（2026-10-04）: 実験の後に `mise run stop` でコンテナを止め，deploy の最後に `docker image prune -a -f` を実行する．このため本走はモデルの再ロードから始まり，先頭の数十問の所要時間が伸びる見込みである（推測）．

**外部調査（tavily，2026-10-04）**:

- Guo, Pleiss, Sun, Weinberger, "On Calibration of Modern Neural Networks", ICML 2017, arXiv:1706.04599（https://arxiv.org/abs/1706.04599 ，https://proceedings.mlr.press/）．温度 T は検証集合の NLL を最小にして当てる．予測の順位を保つので accuracy は変わらない．T>1 で分布が平らになる．
- Minderer ほか, "Revisiting the Calibration of Modern Neural Networks", NeurIPS 2021（proceedings.neurips.cc）．温度スケーリングを，追加の費用が小さい基準の校正法として扱っている．
- Khandelwal ほか, kNN-LM, ICLR 2020, arXiv:1911.00172．補間 \(p = \lambda p_{kNN} + (1-\lambda) p_{LM}\) の出典である（Iter104 と同じ）．補間した後の分布を校正し直す手順は，検索の結果からは確かめられなかった．
- 限定: Guo らは，過信する（T>1 が要る）深層ネットを主な対象にしている．本件の補間分布は，OOF でも eval でも過小確信の向きにある（下記）．そのため T<1（分布を鋭くする向き）になる．温度スケーリングの式はどちらの向きにも使える．

**replay**（`/tmp/iter105/replay_temp.py`，写しは `.claude/research/_iter105_replay_temp.py`．結果は `.claude/research/_iter105_replay_result.json`．開発ホストの CPU で約 53 秒，EXIT=0）:

- 方法: 温度は \(p_T \propto p^{1/T}\)（log p を logit とみなす）で当てる．OOF は訓練集合 2,275 行の外側 5-fold（seed 104）の補間分布である．k=2，λ=0.3 で，kNN の近傍は fold の訓練部分だけから引く．eval の指標は，基準線の各行の `selected_domain`，`dispatched_domains`，`confidence` だけを差し替えて，`metrics.py` の関数で求めた．
- **T の選択（事前登録した唯一の規則．OOF の単一ラベル NLL 最小）**: T* = **0.9426**（探索範囲 [0.05, 20] の端ではない）．fold ごとの T は 0.869 / 0.952 / 0.899 / 1.010 / 0.972 である．OOF の NLL は 0.52657 → 0.52519，OOF の ECE は 0.0241 → 0.0198，符号付きギャップは −0.0195 → −0.0063．OOF の CV top1 は 0.8268 で，Iter104 の値と一致した．
- **T=1 で基準線の本走をどこまで再現できるか**: selected の一致率は 99.573%（不一致 16 行），送出集合の一致率は 98.213%（不一致 67 行）．確信度の差は平均 0.0045，最大 0.226 である．top1 は 0.8504（本走は 0.849867），ECE は 0.050269（本走は 0.050104），`compound_mean_dispatched_count` は 2.068493（本走は 2.064384）．

| 指標 | 本走（基準線） | replay T=1 | replay T*=0.9426 | 条件 |
|---|---|---|---|---|
| top1 | 0.849867 | 0.8504 | **0.8504**（T=1 との不一致 0） | \|Δ\| < 0.25pt |
| McNemar（対 本走） | — | — | 8 対 6，p=0.789 | 参考 |
| 複合行の top1 | 0.805479 | 0.808219 | 0.808219 | C4 ≥ 0.780411 |
| ECE | 0.050104 | 0.050269 | **0.037124** | 主基準 |
| ECE（単一 / 複合） | 0.0326 / 0.1292 | 0.0331 / 0.1316 | 0.0201 / 0.1146 | 参考 |
| 符号付きギャップ（全体） | −0.0501 | −0.0503 | −0.0368 | 参考 |
| `compound_domain_set_recall` | 0.589041 | 0.590411 | **0.574658** | C5 ≥ 0.5400 |
| `compound_mean_dispatched_count` | 2.064384 | 2.068493 | **1.975342** | C5 ≤ 2.10 |
| 全行の平均送出数 | 1.5696 | 1.5765 | 1.5245 | 参考 |
| 複合の k 分布 | {1:451, 2:30, 4:249} | {1:450, 2:30, 4:250} | {1:472, 2:31, 4:227} | 参考 |
| 単一の k 分布 | {1:2507, 2:90, 4:423} | {1:2502, 2:86, 4:432} | {1:2541, 2:91, 4:388} | 参考 |

- **C1（20 指標，BH q=0.05）**: T* と本走の比較で有意は 0 件（最小の p は 0.48）．T* と T=1 の比較でも 0 件．argmax が変わらないので，C1 が動く経路は送出の失敗と埋め込みの微小な差だけである．
- **参考値（選択には使わない）**:
  - eval で ECE が最小になる T は 0.80（ECE 0.0150，mean_dispatched 1.8137，set_recall 0.5521）．
  - eval で C5 の両方を満たす T の範囲は [0.75, 1.02]．
  - OOF で ECE が最小になる T は 0.94．
  - A2（複数の正解を許す判定に合わせた目的関数）: 訓練行はすべて単一ラベルなので，OOF の上では単一ラベルの NLL と一致し，別の値は出ない．eval の上で当てはめた値は選択に使えないので，近いものとして eval の ECE 最小（T=0.80）を併記するに留める．
  - eval の過小確信（ギャップ −0.050）は OOF（−0.020）より大きい．主な原因は複合行の構造的な過小確信（−0.13）であり，単一ラベルの規則ではこれを吸収しきれない（推測）．
- **replay の途中で起きた事故**: 初回のバッチで `replay_temp.py` への Write が「先に Read していない」として失敗した．続く Bash は写しただけの旧スクリプトを実行し，`/tmp/iter104/result_v2.json` を 16:40:15 に上書きした．CV の部分は乱数の種と入力が同じなので元と同じ値である．一方，eval の「基準 artifact の再現」の列は，いまの kNN 補間の artifact で求めた値に変わった．Iter104 の数値は journal の Iter104 節に残っている．このファイルは G0-a の照合元としては使えない（B194 に記録）．

### 計画 (Iter105)

**判断**: T* で ECE は 0.0130 下がる（no-op ではない）．C5 は mean_dispatched 1.975 で上限の内側にある．C1〜C7 を満たす見込みなので，このレバーを採る．代替案の調査（手順 3 の後段）は行わない．

**仮説**: Iter104 の補間分布は，訓練集合の OOF でも eval でも過小確信の向きにある（平均確信度が平均正答を下回る）．OOF の単一ラベル NLL で当てはめた温度 T*=0.9426 で分布を鋭くすると，rank 1 の確信度が上がり，ECE は 0.050 から 0.037 前後へ下がる．argmax は変わらないので top1 は動かない．隣り合う順位の差が広がるので，gap 方式の送出数は減る（mean_dispatched 2.06 → 1.98）．その分 set_recall は約 1.4pt 下がる．

**単一レバー**: `knn_interpolation_temperature`．`KnnInterpolatedClassifier.predict_proba()` の出力を，補間分布 \(p\)（T=1 に相当）から \(p_T \propto p^{1/T}\)，T=0.9426 へ変える．

**固定する構成（Iter104 の採用構成）**: base の LR の artifact（再訓練しない），k=2，λ=0.3，近傍の母集団 `data/classifier_train_iter94_dedup.jsonl` と `data/embcache_train_iter100_fused.npy`，融合表現，`dispatch_gap_threshold` 0.36，`dispatch_gap_max_k` 4，`aggregation_method: max_confidence`，`confidence_threshold` と `dispatch_candidate_threshold` 0.0，評価集合 3,750 行（MD5 `769f2a58dc807e23e1b73e44b97e97b8`），`config.yaml` のすべて．

**レバーを読むコード行と到達条件**:

- 読み込み: `http_server.py:428` → `classifier.py:48`（`joblib.load`）．
- 推定: `http_server.py:379` → `classifier.py:69` → `knn_interpolated_head.py:99-107`（温度を当てる場所）．
- 送出: probe の確信度 → `node.py:232-239` → `aggregator.py:62`（ソート），`:87-91`（gap の連鎖）．rank 1 の確信度は `results.jsonl` の `confidence` に入り，`metrics.py:486-521` の ECE が読む．
- 到達条件: `routing_method: supervised_classifier`（`config.yaml:75`），`confidence_signal_method: self_report`（`:74`），ノードのドメインが `classes_` に含まれること，`dispatch_gap_threshold` が null でないこと．**現行構成は 4 つとも満たす**．

**実装の仕様（フェーズ 2）**:

1. `knn_interpolated_head.py`: `KnnInterpolatedClassifier.__init__` に `temperature: float = 1.0` を加え，T ≤ 0 なら ValueError にする．`predict_proba()` は補間した後に \(p_T \propto \exp(\log(\max(p, 10^{-12}))/T)\) を行ごとに正規化して返す（最大値を引いてから exp を取る）．T=1 のときは現在の出力をそのまま返す．λ=0 の早期 return は，T=1 のときに限る．
   - Iter104 の artifact（`temperature` 属性を持たない pickle）を読めるよう，`__setstate__` で既定値 1.0 を補う．
   - 冒頭の docstring に温度の式と出典（Guo ら，2017）を加える．
2. `scripts/build_knn_interpolated_classifier.py`: `--temperature`（既定 1.0）を加え，ラッパーに渡す．値の導出は `.claude/research/_iter105_replay_temp.py` で再現できる．
3. artifact: 現在の `models/domain_classifier.joblib`（MD5 `c3888f66d90f4172cd7415e5c105328a`）を `models/domain_classifier_pre_iter105_knn.joblib` へ退避する．`--base-classifier models/domain_classifier_pre_iter104_lr.joblib --k 2 --interpolation-lambda 0.3 --temperature 0.9426` で作り直して新しい `models/domain_classifier.joblib` に置き，MD5 を journal に記録する．
4. `knn_interpolated_head.py` はイメージに焼き込まれるので，イメージを再ビルドする（B192 と同じく `mise.toml` の `docker build` と `docker push` だけ．`GIT_HEAD=<HEAD>-dirty-iter105`）．b1f5cb0 により deploy の最後に prune が走るので，前のイメージ `b5b717d1fc7a` には，レジストリの digest `sha256:0a499d0e…` で戻る．
5. テスト（`tests/test_knn_interpolated_head.py` に追加する）: T=1 で従来の出力と一致すること，T≠1 でも行和が 1 になること，T≠1 でも argmax が変わらないこと，T ≤ 0 で ValueError になること，`temperature` 属性を持たない pickle が T=1 で読めること，joblib で往復しても出力が変わらないこと．

**G0（本走の前に，すべての合格を条件とする）**:

- **G0-a**: 新しい artifact で eval キャッシュの `predict_proba` を求める．argmax が現在の artifact と 3,750/3,750 で一致すること．基準線の行の決定を差し替えた ECE が 0.0371 ± 0.0005，`compound_mean_dispatched_count` が 1.9753 ± 0.0030 であること（T を 4 桁に丸めた分の差を幅に含めた）．
- **G0-b**: 全 10 ノードで artifact と `knn_interpolated_head.py` の MD5 がローカルと一致すること．コンテナの中で `joblib.load` した結果が `KnnInterpolatedClassifier` で，`temperature` が 0.9426 であること．
- **G0-c**: 予備 20 問で，`probe_candidates` の確信度の argmax がオフラインの値（T を当てた後）と 20/20 で一致すること．確信度の差は最大 0.02 以内とする（Iter104 の G0-c の最大差は 0.0117）．`mean_duration_ms` は 3,041.7ms 以下であること．

**成功条件と非退行条件（事前登録．以後は変えない）**:

- **基準線**: Iter104 の採用本走 `results/20261004_132607`．top1 0.849867，複合行の top1 0.805479，`compound_domain_set_recall` 0.589041，`compound_mean_dispatched_count` 2.064384，ECE 0.050104，`mean_duration_ms` 2253.224．
- **主基準**: argmax が変わらないレバーなので，ECE を主に見る．
  - `adopted`: ECE ≤ 0.040，かつ |Δtop1| < 0.25pt，かつ C1〜C7 をすべて満たす．
  - `partial`: 0.040 < ECE ≤ 0.045，かつ |Δtop1| < 0.25pt，かつ C1〜C7 を満たす．
  - `negligible`: ECE > 0.045（下げ幅が 0.005 未満）で，C1〜C7 を満たす．
  - `rejected`: Δtop1 ≤ −0.25pt，または C1〜C7 のどれかに違反．
  - `invalid`: G0 に合格しない．または，本走の確信度の分布が基準線と変わらない（ECE の差が 0.001 未満で，かつ複合と単一の k 分布が基準線と同じ）．後者の場合は到達経路を先に調べる（success_criteria (6)）．
- **閾値の根拠**: replay の予測は 0.0371 である．T=1 の replay と本走の ECE の差は 0.00017，同じ artifact で走った本走 2 本（Iter101 と Iter103）の差は 0.0011 である．0.040 は予測から 0.0029 上で，観測した実行間の差の約 2.6 倍の余裕がある．0.040 は Iter101 の水準（0.0396）にあたり，C7 の再較正の基準 0.05 とも十分に離れている．|Δtop1| の帯 0.25pt は，Iter104 の `negligible` の帯と同じ値にした．
- **非退行条件**: Iter104 の C1〜C7 を数値もそのまま使う．
  - **C1**: precision と recall の計 20 指標を BH 補正（q=0.05）し，有意な退行が 0 件．
  - **C2**: `fallback_rate` 0.0，`dispatch_failure_rate` ≤ 0.005．
  - **C3**: G0-a〜G0-c の記録と artifact の MD5 がすべて残っていること．
  - **C4**: 複合行の top1 ≥ 0.780411．
  - **C5**: `compound_domain_set_recall` ≥ 0.5400，`compound_mean_dispatched_count` ≤ 2.10．
  - **C6**: `mean_duration_ms` ≤ 3041.7（絶対値の判定は変えない）．b1f5cb0 により本走がモデルの再ロードから始まるので，参考として，先頭 50 問を除いた平均と，単一層の中央値と p95，複合層の p25 と中央値を併記する．
  - **C7**: ECE ≤ 0.08．

**事前登録する予測**:

- **P1（主予測）**: ECE は 0.035〜0.040（点推定 0.037）で，`adopted` になる．
- **P2**: |Δtop1| < 0.1pt．argmax は変わらず，差は送出の失敗と埋め込みの微小な差だけから生じる（replay では本走と 8 対 6）．
- **P3**: `compound_mean_dispatched_count` は 1.96〜1.99，`compound_domain_set_recall` は 0.570〜0.580．set_recall は約 1.4pt 下がるが，C5 の内側にとどまる．
- **P4**: C1 の有意は 0 件．
- **P5**: 全行の平均送出数が約 3% 減るので，先頭 50 問を除いた `mean_duration_ms` は基準線から −5%〜+2% に入る．再ロードを含めた全体の平均も C6 の内側に入る．
- **P6**: 複合行の ECE は 0.11 前後に残る．単一ラベルで当てた温度では，複数の正解による過小確信は吸収しきれない．

**フェーズ 2 への申し送り**:

- コードの変更は，上の「実装の仕様」の 1，2，5 と，Dockerfile は変えずにイメージを再ビルドすることに限る．`config.yaml` は変えない．
- G0-a は開発ホストの CPU で行える（`uv run`）．`/tmp/iter104/result_v2.json` は上書きされたので照合には使わない．照合元は `.claude/research/_iter105_replay_result.json` である．
- deploy の後，本走の前に G0-b と G0-c を行う．本走は 3,750 問のフルスペックで行う．b1f5cb0 の手順により，analyze の後に `mise run stop` が走る．
- 判定の比較相手は `results/20261004_132607`．C1 は `metrics.py` の関数で計算する．ECE には単一行と複合行の内訳と，符号付きギャップを併記する．
- T を変えた場合の set_recall と所要時間のトレードオフ（参考格子）は `_iter105_replay_result.json` の `reference_not_for_selection` にある．T を選び直すことはしない．

### Iteration 105 実装・実験（2026-10-04）

**再ビルド前のイメージ（ロールバック用．deploy の末尾の prune で手元から消えるため，ビルドの前に記録した）**: `localhost:5001/expert-mesh:latest` の image ID は `sha256:b5b717d1fc7aeac6ff38cf67996d6526304db151a9913fedf6ea3e2a93a99459`（Created 2026-10-04T12:54:34+09:00，`GIT_HEAD=7c0b703-dirty-iter104`）である．RepoDigest とレジストリの v2 manifest の digest は `sha256:0a499d0e172319c7af21a09b64d46203a2872981e2dab9e1a23b1fc17f505868` である．戻すときは `localhost:5001/expert-mesh@sha256:0a499d0e…` を pull する．

**変更したファイル**（`config.yaml` と Dockerfile は変えていない）:

- `knn_interpolated_head.py`（MD5 `76e0d089a512f5b9c0c80be8f9a4a60c`）:
  - `apply_temperature_to_proba()` を加えた．式は \(p_T \propto \exp(\log(\max(p, 10^{-12}))/T)\) で，行の最大値を引いてから exp を取り，行ごとに正規化する．replay の `apply_temperature()` と同じ式である．
  - `KnnInterpolatedClassifier.__init__` に `temperature: float = 1.0` を加えた（T ≤ 0 なら ValueError）．`predict_proba()` は補間の後に温度を当て，T=1 のときは補間の結果をそのまま返す．
  - `__setstate__` で，`temperature` を持たない Iter104 の pickle に 1.0 を補う．冒頭の docstring に式と出典（Guo ら，2017）を加えた．
- `scripts/build_knn_interpolated_classifier.py`: `--temperature`（既定 1.0）を加え，ラッパーに渡す．
- `tests/test_knn_interpolated_head.py`: 6 関数（parametrize を含めて 7 ケース）を加えた．T=1 で従来の出力と一致すること，T≠1 で行和が 1 になること，T≠1 で argmax が変わらないこと，T ≤ 0 で ValueError になること，`temperature` 属性の無い pickle が T=1 で読めること，joblib の往復で出力が変わらないことを確かめる．

**テストと lint**: `uv run pytest -q tests/test_knn_interpolated_head.py tests/test_classifier.py` は 23 passed だった．warning は joblib の NumPy 2.5 の DeprecationWarning だけである．変更した 3 ファイルの `uv run ruff check` は All checks passed だった．

**artifact**:

- 退避: `models/domain_classifier_pre_iter105_knn.joblib`．MD5 `c3888f66d90f4172cd7415e5c105328a` で，元の artifact と一致した．
- 新しい `models/domain_classifier.joblib`: MD5 `c7172ad37c10e1082a42481553ae25b0`．`--base-classifier models/domain_classifier_pre_iter104_lr.joblib --k 2 --interpolation-lambda 0.3 --temperature 0.9426` で作った（n_train=2275，dim=6656）．

**イメージ**: `mise.toml` の `docker build`（`--build-arg GIT_HEAD=b1f5cb0-dirty-iter105`）と `docker push` だけを実行した．`mise run setup` は実行していない．ログは `.claude/research/_iter105_build.log` にある．

- 新イメージの image ID は `sha256:b4a3373f69436b4ab3900d7273cc3adb156b029244fd5750507a966ef52d05c8`，push の digest は `sha256:a328caf4860b925c53cb394004cd6be92879cb325d1095663107c2b441a062e9` である．
- イメージの中の `/app/knn_interpolated_head.py` の MD5 はローカルと一致した．
- `data/dataset.jsonl` の MD5 は，ビルドの前後とも `769f2a58dc807e23e1b73e44b97e97b8` だった．

**deploy**: `mise run deploy` は終了コード 0 で終わった（`.claude/research/_iter105_deploy.log`）．healthcheck と smoke_check（git-status，hashes，probe．probe の latency は 12ms）は通り，末尾の prune は全ノードと手元で実行された．

**G0**:

- **G0-a: 合格**（`.claude/research/_iter105_g0a.json`．スクリプトは `/tmp/iter105/g0a_check.py` で，replay の `simulate_rows()` と `summarize()` を再利用した）．
  - argmax は旧 artifact と 3750/3750 で一致した．
  - ECE は 0.037119（許容 0.0371±0.0005），`compound_mean_dispatched_count` は 1.975342（許容 1.9753±0.0030）だった．
  - replay の式で旧 artifact に T を当てた値との差の最大は 0.0 だった．
- **G0-b: 合格**（`.claude/research/_iter105_g0b.txt`）．
  - wafl500〜509 の全 10 ノードで，`/app/models/domain_classifier.joblib`（`c7172ad3…`）と `/app/knn_interpolated_head.py`（`76e0d089…`）の MD5 がローカルと一致した．
  - コンテナの中の `GIT_HEAD` は `b1f5cb0-dirty-iter105` だった．`joblib.load` の結果は `knn_interpolated_head.KnnInterpolatedClassifier` で，temperature=0.9426，k=2，λ=0.3 だった．
  - `docker inspect` の `Image` の表示は，wafl500〜507 が digest（`a328caf4…`），wafl508〜509 が image ID（`b4a3373f…`）だった．どちらも今回のビルドを指す．
- **G0-c: 合格**（`.claude/research/_iter105_g0c.txt` と `_iter105_g0c.json`．予備実行は `results/20261004_173015/preview20.jsonl`，`data/dataset_20.jsonl`，requester は wafl500）．
  - オンラインとオフライン（T=0.9426 の後）の argmax は 20/20 で一致した．確信度の差は最大 0.0111（business_economics-014）だった．
  - dispatch の失敗と fallback は 0 件，`mean_duration_ms` は 581.1 だった．

**本走**: `results/20261004_173104`（3,750 問，`mise run start -- --dataset data/dataset.jsonl --output results.jsonl`，requester は wafl500）．17:31:04 に起動し，19:50 頃に完了した（EXIT=0．ログは `.claude/research/_iter105_mainrun_start.log`）．

- `results.jsonl` は 3,750 行で，MD5 `d96dad0550d4750e34e39bc9dc63a3bf` はローカルと wafl500 側で一致した．`git_head.txt` は `b1f5cb0-dirty-iter105` である．
- `mise run analyze -- 20261004_173104` は終了コード 0 で終わった（`.claude/research/_iter105_analyze.log`．`answer_quality_accuracy` 0.583113，`end_to_end_accuracy` 0.401067）．
- `uv run python metrics.py --results results/20261004_173104/results.jsonl --json` の出力は `results/20261004_173104/metrics.json` に書いた．
- その後，`mise run stop` で全ノードのコンテナを止めた（終了コード 0．削除はしていない）．

**主要指標**（`metrics.json`，および replay の `summarize()` で同じ関数を両方の本走に当てた値．集計の記録は `.claude/research/_iter105_main_summary.json`）:

| 指標 | Iter105 本走 | 基準線 20261004_132607 |
|---|---|---|
| top1 | 0.848267 | 0.849867 |
| 複合行の top1 | 0.805479 | 0.805479 |
| `compound_domain_set_recall` | 0.573288 | 0.589041 |
| `compound_mean_dispatched_count` | 1.980822 | 2.064384 |
| ECE | 0.037281（n=3,744） | 0.050104（n=3,750） |
| ECE（単一 / 複合） | 0.02095 / 0.11222 | 0.03262 / 0.12915 |
| 符号付きギャップ（全体 / 単一 / 複合） | −0.0365 / −0.0182 / −0.1122 | −0.0501 / −0.0310 / −0.1292 |
| Brier | 0.100591（n=3,744） | 0.102122（n=3,750） |
| `mean_duration_ms` | 2233.403 | 2253.224 |
| 参考: 先頭 50 問を除いた平均 | 2254.678 | 2273.579 |
| 参考: 先頭 50 問の平均 | 659.08 | 746.98 |
| 参考: 単一層の中央値 / p95 | 535.5 / 3089.2 | 539.0 / 3252.4 |
| 参考: 複合層の p25 / 中央値 | 7668.25 / 8785.0 | 7583.5 / 8795.5 |
| `fallback_rate` | 0.0 | 0.0 |
| `dispatch_failure_rate` | 0.0016（6 行） | 0.0 |
| 参考: 全行の平均送出数 | 1.5243 | 1.5696 |
| 参考: 複合の k 分布 | {1:470, 2:32, 4:228} | {1:451, 2:30, 4:249} |
| 参考: 単一の k 分布 | {1:2540, 2:95, 4:385} | {1:2507, 2:90, 4:423} |

- ECE と Brier の n が 3,744 なのは，`confidence` が null の行が 6 行あるためである．この 6 行は `dispatch_failed` の 6 行と同じ件数である．
- C1（BH 補正の 20 指標）と McNemar は，このフェーズでは計算していない．フェーズ 3 で `metrics.py` の関数を使って計算する．
- 実行上の異常: dispatch の失敗が 6 件あった（`medical-110`，`natural_science-002`，`natural_science-079`，`natural_science-109`，`medical-exp085-140`，`education-exp085-044`．いずれも単一行で，送出先は 1 件（`dispatched_domains` は medical / natural_science / education）．原因はこのフェーズでは調べていない）．fallback は 0 件で，ハング，OOM，エラーの終了は無かった．ポーリングでは複合行の区間（17:55〜19:31 頃）で進み方が約 7.5 行/分に落ちたが，wafl500 側の `results.jsonl` は更新され続けていた．

### Iteration 105 実行済み

#### 計算の方法

- スクリプトは `.claude/research/_iter105_post_analysis.py` で，出力は `.claude/research/_iter105_post_analysis.json` に保存した．`uv run` で実行し，EXIT=0 だった．手順は `scripts/_iter101_post_analysis.py` に合わせた．
- 比較の相手は基準線 `results/20261004_132607/results.jsonl` である．id は 3,750 行で完全に一致する．
- top1 の対の比較には `metrics.compute_mcnemar_test`（連続補正つき．a=本走，b=基準線）を使った．C1 には `compute_domain_recall_mcnemar_test`，`compute_domain_precision_fisher_test`，`apply_benjamini_hochberg`（q=0.05）を使った．
- 参考として，送出に失敗した 6 行を両方の run から除いた共通の 3,744 行で，`compute_ece` と `compute_brier_score` を当て直した．主の判定には事前登録どおり `metrics.json` の値を使う．
- **一時スクリプトの退避**: `/tmp/iter105/{g0a_check.py,g0c_check.py,main_summary.py}` は再起動で消えうるので，`.claude/research/_iter105_g0a_check.py`，`_iter105_g0c_check.py`，`_iter105_main_summary.py` へ写した．写しの MD5 は元と一致した（`9ed511ab…`，`33c9a562…`，`873641cc…`）．G0-a の記録と主要指標の表は，これらで再現できる．`.claude/research/_iter10x_*` は過去の反復でも追跡していないので，写した 3 本と分析スクリプトもコミットせず，作業ツリーに残す（backlog B195）．
- テストと lint は分析の時点で再実行した．`uv run pytest -q tests/test_knn_interpolated_head.py tests/test_classifier.py` は 23 passed，変更した 3 ファイルと分析スクリプトの `ruff check` は clean だった．作業ツリーの `knn_interpolated_head.py` の MD5 は `76e0d089a512f5b9c0c80be8f9a4a60c` で，G0-b でイメージの中から取った値と一致する．

#### 結果（本走 `results/20261004_173104` 対 基準線 `results/20261004_132607`）

| 指標 | 基準線 | 本走 | Δ | 事前登録の予測 |
|---|---|---|---|---|
| ECE（`metrics.json`．主基準） | 0.050104（n=3,750） | **0.037281**（n=3,744） | −0.0128 | P1: 0.035〜0.040．的中 |
| 参考: ECE（共通の 3,744 行） | 0.049996 | 0.037281 | −0.0127 | — |
| ECE（単一 / 複合．共通の行） | 0.03245 / 0.12915 | 0.02095 / 0.11222 | −0.0115 / −0.0169 | P6: 複合は 0.11 前後．的中 |
| 符号付きギャップ（全体 / 単一 / 複合） | −0.0501 / −0.0310 / −0.1292 | −0.0365 / −0.0182 / −0.1122 | — | — |
| Brier（`metrics.json`） | 0.102122（n=3,750） | 0.100591（n=3,744） | −0.0015 | — |
| 参考: Brier（共通の 3,744 行） | 0.102235 | 0.100591 | −0.0016 | — |
| top1 | 0.849867 | **0.848267** | **−0.16pt（−6 行）** | P2: \|Δ\| < 0.1pt．**外れ** |
| McNemar（連続補正） | — | 本走のみ正答 0，基準線のみ正答 6，chi2 4.1667，p = 0.0412 | — | — |
| 参考: top1（共通の 3,744 行） | 0.849626 | 0.849626 | 0 | — |
| 複合行の top1 | 0.805479 | 0.805479 | 0 | — |
| `compound_domain_set_recall` | 0.589041 | 0.573288 | −1.58pt | P3: 0.570〜0.580．的中 |
| `compound_mean_dispatched_count` | 2.064384 | 1.980822 | −0.0836 | P3: 1.96〜1.99．的中 |
| 全行の平均送出数 | 1.5696 | 1.5243 | −2.9% | P5 の前提（約 3% 減）と一致 |
| `mean_duration_ms` | 2253.224 | 2233.403 | −0.88% | — |
| 先頭 50 問を除いた平均 | 2273.579 | 2254.678 | −0.83% | P5: −5%〜+2%．的中 |

- **top1 の −6 行の内訳**: 基準線のみ正答の 6 行は，`education-exp085-044`，`medical-110`，`medical-exp085-140`，`natural_science-002`，`natural_science-079`，`natural_science-109` で，本走で `dispatch_failed` になった 6 行と id で完全に一致した．本走のみ正答の行は 0 行である．失敗した 6 行を除いた共通の 3,744 行では，top1 は両方とも 0.849626 で同じ値になる．温度で正誤が変わった行は 1 行も無い．

#### 事前登録条件の照合（C1〜C7，Iter104 と同じ閾値）

| 条件 | 基準 | 実測 | 判定 |
|---|---|---|---|
| 主基準 | ECE ≤ 0.040 かつ \|Δtop1\| < 0.25pt | 0.037281，0.16pt | PASS |
| C1 | 20 指標の BH 補正（q=0.05）後の有意な退行が 0 件 | 0 件（20 指標とも生の p ≥ 0.248） | PASS |
| C2 | `fallback_rate` = 0.0，`dispatch_failure_rate` ≤ 0.005 | 0.0，0.0016 | PASS |
| C3 | G0-a〜G0-c の記録と artifact の MD5 | `_iter105_g0a.json`，`_iter105_g0b.txt`，`_iter105_g0c.{txt,json}` がある．artifact の MD5 は `c7172ad37c10e1082a42481553ae25b0`（全 10 ノードで一致） | PASS |
| C4 | 複合行の top1 ≥ 0.780411 | 0.805479 | PASS |
| C5 | set_recall ≥ 0.5400 かつ mean_dispatched ≤ 2.10 | 0.573288 / 1.980822 | PASS（上限まで残り 0.119） |
| C6 | `mean_duration_ms` ≤ 3041.7 | 2233.403 | PASS |
| C7 | ECE ≤ 0.08 | 0.037281 | PASS．0.05 を下回ったので，校正側に立てる規定は解除される |

- **C1 の内訳**: recall で差が出たのは `natural_science_recall`（本走のみ正答 0 対 基準線のみ正答 3，p = 0.248），`medical_recall`（0 対 2，p = 0.480），`education_recall`（0 対 1，p = 1.0）の 3 指標だけである．この 6 行は，送出に失敗した 6 行と同じである．ほかの 7 ドメインの recall は不一致 0 で，p = 1.0 である．precision の 10 指標はすべて p = 1.0 である（真陽性と選択数の差は，失敗した行の分の 1〜3 行だけ）．
- **C6 の分位点**（括弧内は基準線）: 単一層の中央値 535.5ms（539.0），p95 3,089.2ms（3,252.4）．複合層の p25 7,668.25ms（7,583.5），中央値 8,785.0ms（8,795.5）．先頭 50 問の平均は 659.08ms（746.98）で，再ロードによる遅れは平均を押し上げていない．

#### 事前登録の予測との照合

- **P1（主予測）: 一致**．ECE は 0.037281 で，予測区間 0.035〜0.040 に入り，点推定 0.037 とほぼ同じ値である．replay の 0.037124 との差は 0.00016，G0-a の 0.037119 との差も 0.00016 である．
- **P2: 不一致**．\|Δtop1\| は 0.16pt で，予測の 0.1pt 未満を超えた．argmax が変わらないこと自体は予測どおりで，差はすべて送出の失敗 6 行から生じた．予測は「送出の失敗と埋め込みの微小な差」を差の出どころに挙げていたが，失敗の件数を基準線の水準（0 行）で見込んでいた．このため，区間の幅が足りなかった．
- **P3: 一致**．mean_dispatched は 1.980822，set_recall は 0.573288 で，どちらも区間の内側にある．
- **P4: 一致**．C1 の有意は 0 件である．
- **P5: 一致**．先頭 50 問を除いた平均は −0.83% で，全体の平均も C6 の内側にある．
- **P6: 一致**．複合行の ECE は 0.112 で，0.11 前後に残った．

#### 判定（分析フェーズ，2026-10-04）

- **判定: `adopted`**．事前登録の判定規則（「計画 (Iter105)」）をそのまま当てはめた．閾値は緩めても締めてもいない．ECE は 0.037281 ≤ 0.040，\|Δtop1\| は 0.16pt < 0.25pt で，C1〜C7 をすべて満たす．G0 はすべて合格し，k の分布も基準線から動いているので，`invalid` には当たらない．
- **採用構成**: `models/domain_classifier.joblib`（`KnnInterpolatedClassifier`，k=2，λ=0.3，T=0.9426，MD5 `c7172ad37c10e1082a42481553ae25b0`）を以後の基準とする．T=1 の artifact は `models/domain_classifier_pre_iter105_knn.joblib`（MD5 `c3888f66d90f4172cd7415e5c105328a`）として残す．新しい基準線は本走 `results/20261004_173104` とする．
- **雑音と信号の切り分け**:
  - ECE の −0.0128 は，同じ artifact で走った本走 2 本（Iter101 の 0.0396，Iter103 の 0.0385）の差 0.0011 の約 12 倍ある．replay の予測とは 0.00016 の差で一致した．したがって，雑音ではなく温度の効果とみる．共通の 3,744 行に揃えても −0.0127 で，n の違いは結論に効かない．
  - 送出数の減少（set_recall −1.58pt，mean_dispatched −0.084）も replay の予測（0.5747，1.9753）に近く，温度の効果である．
  - top1 の McNemar は p = 0.041 で，名目上は 0.05 を下回る．ただし不一致の 6 行は，すべて送出に失敗した行である．共通の行では top1 は完全に同じである．したがって，この差はレバーの効果ではない．温度 T は分類器の確率にしか効かないので，送出の失敗はインフラ側の事象と推定している（推定であり，ollama のログが無いので原因は確かめていない）．判定規則は McNemar ではなく \|Δtop1\| の帯で事前登録しているので，判定は変わらない．
- **計画の仮説との一致**: 「補間分布は過小確信の向きにあり，T*=0.9426 で鋭くすると ECE が下がり，argmax は変わらず，送出数が減って set_recall が約 1.4pt 下がる」は，すべて支持された（set_recall の下げ幅は 1.58pt）．ECE の改善は主に単一行で出た（0.0326 → 0.0210）．複合行は 0.129 → 0.112 で，ギャップ −0.112 が残る．
- **想定外の挙動**: 言語崩れ，発散，OOM は無い．送出の失敗が 6 行あった（基準線は 0 行）．送出先のノードの ollama が `/api/chat` に `500 Internal Server Error` を返し，app のログに `dispatch_model_not_ready` が出ていた（wafl503 で 2 件，wafl506 で 3 件，wafl501 で 1 件．ほかに wafl509 で，複合行 `social_science-121` の 4 送出のうち 1 件．これはほかの応答で補われた）．失敗は 17:47〜17:49 と 19:33〜19:35 の 2 つの時間帯に固まっていた．probe は正常で，振り分けも正しかった．
- **後始末**: config.yaml は変えていないので，戻す設定は無い．`mise run stop` でコンテナは止めてある（削除はしていない）．

#### 学び (Iter105)

1. **argmax を変えない校正のレバーは，キャッシュの replay で本走の値まで予測できる**．ECE の差は 0.00016，set_recall と mean_dispatched も予測に近かった．argmax が変わらないので，top1 の差の出どころは送出の失敗だけになる．同じ型のレバーでは，本走は「インフラの揺れの中で予測が保たれるか」の確認になる．
2. **単一ラベルで当てた温度では，複合行の過小確信は消えない**．複合行のギャップは −0.129 → −0.112 で，ほとんど残った．原因は，複数の正解を許す判定（`selected_domain in expected_domains`）と，rank 1 の単一ドメインの確信度との食い違いである．これを ECE から除くには，確信度の定義（たとえば上位の確率の和）か判定の側を変える必要があり，温度の再調整では届かない（推測）．
3. **送出の失敗は，argmax を変えないレバーでも top1 の McNemar を「有意」にする**．今回は 0 対 6 で p = 0.041 だった．判定を \|Δtop1\| の帯で事前登録していたので，判定は揺れなかった．今後の対の比較では，`dispatch_failed` の行を除いた共通の行の値を必ず併記する．
4. **ollama の 500 による送出の失敗は，原因が確かめられたものとしては Iter102 に続いて 2 回目である**（Iter103 の失敗 4 行については，journal に原因の記録が無い）．今回は 7 件が 2 つの時間帯に固まっていた．`mise run analyze` は app のログしか集めないので，ollama のログが残らず，原因を追えなかった．analyze で ollama のログも集める提案を B195 に残した．
5. **levers は使い切った**．Iter106 は `current_lever=null` として調査・計画フェーズから始める（backlog B195．tavily-search で重点調査する）．
6. **一時スクリプトは，結果を記録した時点で `.claude/research/` へ写す**．`/tmp` は再起動で消えうる．今回は 3 本を分析の時点で写した．G0 や集計のスクリプトは，実行フェーズのうちに写しておくと，記録の再現性が切れない．

## Iteration 104: kNN ラベル分布と LR 確率の線形補間

### 調査 (Iter104)

**実施場所の申告**: 本フェーズで行うのは，開発ホストでのリポジトリ読み取り，tavily-search による外部調査，分類器の CPU replay（実施場所は後述）だけである．wafl500〜509 には接続しない．

**確認できた事実（途中経過．見出しはレバー確定後に正式名へ置き換える）**

1. **集約の方向は打ち止め**．llm_judge（Iter48），固定 k=2 の majority_vote（Iter46），k=4 の majority_vote（Iter103）の 3 方式とも結果が出ている（journal.md の Iter103「判定」「学び」節）．max_confidence のもとでは，送出集合を変えるレバーは top1 に届かない（Iter102「判定」「学び」節）．
2. **決定層の方向も打ち止め**．Iter97（OvO）と Iter98（ECOC）は closed．多クラス分解・較正・重み・訓練行を一巡しても CV は 0.79〜0.82 から動かず，記録には「残る説明変数は入力表現とラベルの張り方」とある（journal_archive.md 1352〜1392 行目，1763〜1794 行目）．
3. **表現・データの方向で試した範囲**．埋め込みの差し替え（Iter79，80，89，99），prefix と連結（Iter81，82），融合（Iter100 で効果を確認，Iter101 で採用），hard negative（Iter84〜88），L2 正則化（Iter91），ラベルの写像と粒度（Iter92，93），重複の除去（Iter94），訓練行の増量（Iter95，96）．Iter101 の「次の一手」によれば精度の本線 3 本はいずれも打ち止めである（journal.md の Iter101「次の一手」節）．
4. **rank 1 を決めるコード行**．`classifier.py:69` の `classifier.predict_proba([query_embedding])[0]`．分類器は `CalibratedClassifierCV`（temperature）で `LogisticRegression` を包んだもので，`config.yaml:156` の `classifier_model_path: models/domain_classifier.joblib` が指す．
5. **到達条件**．`confidence_signal_method=self_report` かつ `routing_method: supervised_classifier`（`config.yaml:75`）のとき，`http_server.py` の `_estimate_probe_confidence()` は LLM を呼ばずに `estimate_confidence_classifier()` へ落ちる（Iter101 A1）．現行構成は両方を満たす．
6. **副次的な観察**．融合で次元が 6,656 になり p/n は 2.93．ECE は 0.0318 → 0.0396 へ悪化し，温度 1 パラメータでは高次元側の過信を吸収しきれない可能性が指摘されている（journal_archive.md 442〜449 行目）．

**候補（未検証）**: (c1) PCA などで次元を落としてから LR を学習する．(c2) 訓練集合の埋め込みに対する kNN のラベル分布を LR の確率と混ぜる．どちらも `classifier.py:69` で読まれる artifact の差し替えで単一レバーにできる見込みがある．

**過去に試したかの確認（journal_archive.md・journal.md・config.yml・backlog.md を `kNN|k-NN|k近傍|PCA|次元削減|MLP|SVM|ensemble|アンサンブル|prototype|centroid|LDA|shrinkage` で grep）**

- **c1（PCA）は試し済みで効果なし → 候補から外す**．Iter99 の分析で，fold ごとに訓練部分だけで PCA を学習した容量曲線が取られている（journal_archive.md 833〜847 行目）．qwen3-4b 2 ビューで 5120 次元 0.804396，PCA 1536 次元 0.804396（同値），768 次元 0.799121，512 次元 0.795165，128 次元 0.775824．次元を落としても CV top1 は上がらず，下がる一方である．融合 6,656 次元で同じことが起きない根拠は無いので，c1 は外す．
- **c2（kNN 融合）は試されていない**．Iter97 の計画で「非線形ヘッド（RBF-SVM・MLP・kNN 融合）」として検討され，「5,120 次元 対 2,275 行の p≫n で過学習が濃厚」という理由で，実測なしに却下されている（backlog.md 962〜964 行目）．kNN の確率を LR と補間する形なら，追加のパラメータは近傍数 k と混合重み λ の 2 つで，学習される重みは増えない．過学習の懸念がそのまま当てはまるかは実測で確かめる．
- 「ルーティング判定自体のアンサンブル化（複数分類器/埋め込みの多数決）」は見送りとして記録がある（backlog.md 3075 行目）．kNN と LR の補間はこれと近いので，見送った理由を確かめた．見送りの理由は「実装コスト中〜高，レイテンシとのトレードオフという新しい評価軸が要る」であり，対象は**複数の分類器または複数の埋め込み**の多数決である（backlog.md 3075〜3077 行目）．kNN の補間は，既に計算済みの同じ融合埋め込みを使い，訓練行（約 2,300 行）との内積を 1 回取るだけで，埋め込みや LLM の呼び出しは増えない．したがって見送りの理由はそのままは当てはまらない．
- MLP と SVM を単体のヘッドとして試した記録は見つからなかった（ヒットしたのは LoRA の MLP 層と，文献で引いた SVM だけ）．

**外部調査（tavily-search，2026-10-04）**

- Khandelwal, Levy, Jurafsky, Zettlemoyer, Lewis, "Generalization through Memorization: Nearest Neighbor Language Models", ICLR 2020, arXiv:1911.00172（https://arxiv.org/abs/1911.00172）．事前学習済み LM の次単語分布に，同じ埋め込み空間での kNN 分布を線形補間する（\(p = \lambda p_{kNN} + (1-\lambda) p_{LM}\)）．再学習なしで Wikitext-103 の perplexity を下げた．本件の c2 はこの補間式をクラス分布に当てはめたものである．
- Li, Song, Ma, Qiu, Huang, "KNN-BERT: Fine-Tuning Pre-Trained Models with KNN Classifier", 2021, arXiv:2110.02523（https://arxiv.org/abs/2110.02523）．テキスト分類で，線形分類器の出力に kNN 分類器を組み合わせると精度と頑健性が上がったと報告する．
- "Revisiting k-NN for Fine-tuning Pre-trained Language Models", CCL 2023（https://aclanthology.org/2023.ccl-1.75.pdf）．PLM の分類器と kNN の確率を補間する方式を，テキスト分類で再評価している．
- "Label Distribution Learning-Enhanced Dual-KNN for Text Classification", 2025, arXiv:2503.04869（https://arxiv.org/abs/2503.04869）．kNN をラベル分布の推定に使う近年の拡張．
- 限定: これらはいずれも埋め込みを**微調整した**モデル上の結果である．本件のように凍結した埋め込みと LR の組で同じ利得が出るかは分からない（推測）．また上記の多くは訓練行が数万以上の設定で，約 2,300 行の本件では近傍の質が劣る可能性がある．効果の有無は replay で確かめる．

**replay の実施場所**: config.yml 冒頭の (B) に従い，まず wafl-ctrl5 への SSH を 2 回試した（2026-10-04）．2 回とも許可システムの自動判定に阻止され，接続できなかった．代替として，Iter99〜101 の replay を行った場所を確かめた．Iter99 の分析で行った CV と PCA の容量曲線は「すべて開発ホストの CPU」で行われている（journal_archive.md 819 行目，928 行目）．そこで本反復の replay も**開発ホストの CPU**（`uv run`．埋め込みは計算済みのキャッシュ `data/embcache_{train,eval}_iter100_fused.npy` を読むだけで，再計算はしない．GPU と LLM は使わない）で行う．wafl500〜509 には接続しない．オーケストレータの指示により，wafl-ctrl5 への接続はこれ以上試みない（B191 の要レビューに記録する）．

**artifact を参照するコードの網羅**（`grep -rn --include='*.py' -e 'domain_classifier' -e 'classifier_model_path' -e 'predict_proba' -e 'classes_' -e 'joblib.load'`．`.venv` と `__pycache__` は除外した）

- 本走の経路: `http_server.py:428` が `load_domain_classifier(state.classifier_model_path)` で読み込み（`classifier.py:48` の `joblib.load`），`http_server.py:379` が `estimate_confidence_classifier(state.domain_classifier, state.domain, body.query_embedding)` を呼ぶ．`classifier.py:66` と `:68` が `classes_`，`:69` が `predict_proba([query_embedding])[0]` を読む．
- 補助の経路: `scripts/run_central_experiment.py:243-244`（`predict_proba` と `classes_`），`scripts/mine_hard_negatives.py:344-346`（同じ 2 つ）．
- `calibrated_classifiers_` など，ほかの属性を読む箇所は本走の経路にも補助の経路にも無い．ラッパーが提供すべき属性は `classes_` と `predict_proba(X)` の 2 つだけである．型注釈は `CalibratedClassifierCV`（`http_server.py:195`，`:263`，`classifier.py:40`，`:52`）だが，`classifier.py:11-19` に書かれているとおり duck typing で読んでいる．

**replay 第 1 版（`/tmp/iter104/replay_knn.py`，結果 `/tmp/iter104/result.json`，`/tmp/iter104/near_dup_check.py`）**

- 格子は k ∈ {5, 10, 20, 50} × λ ∈ {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7}．訓練集合 2,275 行の 5-fold CV（seed 104）だけで点を選んだ．kNN の近傍は，CV では fold の訓練部分だけから，eval では訓練集合だけから引く（オーケストレータが漏れの無いことを確認済み）．
- CV で k=5，λ=0.5 を選んだ（CV top1 0.8075 → 0.8237）．eval では top1 が 0.8389 → 0.8547 になった（基準 artifact の再現値との比較）．
- 訓練行との最大 cosine 類似度が 0.90 以上の評価行（278 行）を除いても Δtop1 は +1.56pt で，利得は近重複に依存していない．
- 問題点: k=5 は格子の端だった．ECE は 0.070，mean_k は 1.654 になった．ドメインごとの検定は recall の 10 指標だけを BH 補正しており，C1（precision と recall の計 20 指標）と同じではない．この補正で computer_science の recall が q=0.043 で有意に下がった．

**replay 第 2 版（`/tmp/iter104/replay_knn_v2.py`，結果 `/tmp/iter104/result_v2.json`．開発ホスト CPU，約 50 秒）**

格子の端を解消するため k を {1, 2, 3, 5, 7} に広げた．λ の格子は同じである．選ぶのは訓練集合の 5-fold CV だけで，同値のときは λ の小さい方を，次に k の大きい方を取る．eval の指標は，`metrics.py` の判定用の関数（`compute_top1_accuracy`，`compute_mcnemar_test`，`compute_domain_recall_mcnemar_test`，`compute_domain_precision_fisher_test`，`apply_benjamini_hochberg`，`compute_ece`，`compute_compound_coverage_metrics`）で求めた．基準線 `results/20260929_192157/results.jsonl` の各行について，`selected_domain`，`dispatched_domains`，`confidence` だけを新しい確率分布の決定（gap 0.36，max_k 4）で置き換えた．

- **CV（選択に使う）**: LR 単体 0.8075．kNN 単体は k=1 から 7 の順に 0.7952，0.7635，0.7965，0.7996，0.8035．
  - 最良は **k=2，λ=0.3 で 0.8268（+1.93pt）**．fold ごとの値は 0.835 / 0.820 / 0.842 / 0.826 / 0.811 で，5 fold すべてで同じ fold の LR 単体（0.820 / 0.787 / 0.829 / 0.807 / 0.796）を上回る．
  - 次点は k=2，λ=0.4 の 0.8255，k=1，λ=0.2 の 0.8242，k=3 または 5 で λ=0.4〜0.5 の 0.8237 である．k=2 の周りでは λ=0.2〜0.5 がいずれも 0.822 以上で，選んだ点は孤立した山ではない．
  - k=2 と λ=0.3 は，どちらも格子の内側にある．
- **eval（選んだ 1 点のみ．判定には使わない参考値として，全格子点の最良は k=3，λ=0.4 の 0.8552）**:

| 指標 | 基準線の本走（実測） | 基準 artifact の再現 | k=2，λ=0.3 | 条件 |
|---|---|---|---|---|
| top1 | 0.8400 | 0.8389 | **0.8504** | 主指標 |
| McNemar（対 本走） | — | — | 新方式のみ正答 103，基準線のみ正答 64，p=0.0033 | p<0.05 |
| 単一行の top1 | 0.8483 | 0.8477 | 0.8606 | 参考 |
| 複合行の top1（730 行） | 0.8055 | 0.8027 | 0.8082（19 対 17，p=0.87） | C4 ≥ 0.780411 |
| `compound_domain_set_recall` | 0.5788 | 0.5801 | 0.5904 | C5 ≥ 0.5400 |
| `compound_mean_dispatched_count` | 1.9740 | 1.9699 | **2.0685** | C5 ≤ 2.10 |
| ECE | 0.0396 | 0.0382 | **0.0503** | C7 ≤ 0.08 |
| 全行の平均送出数 | 1.5331 | 1.5333 | 1.5765 | 参考 |

- **C1（20 指標，BH q=0.05，対 本走）**: 有意と判定されたのは 2 指標で，どちらも**改善**の向きである．`business_economics_recall` は新方式のみ正答 17 対 基準線のみ正答 3（p=0.0037），`education_recall` は 38 対 14（p=0.0014）．**有意な退行は 0 件**である．
  - 下がったが有意でない指標: `computer_science_recall` 3 対 10（p=0.096），`medical_recall` 12 対 23（p=0.091），`natural_science_recall` 6 対 10（p=0.45）．
  - precision の 10 指標はすべて p ≥ 0.14 である．
  - 第 1 版で問題になった computer_science の有意な退行は，k=2，λ=0.3 では起きなかった．比較の相手を「基準 artifact の再現」に替えても，有意となるのは `education_recall`（改善）だけである．
- 基準 artifact の再現と本走の差は top1 で 0.11pt である（決定の一致率は 99.49%）．この差は，実行時の埋め込みとキャッシュの微小な差によるとみられる（推測）．

### 計画 (Iter104)

**仮説**: 凍結した融合埋め込み（6,656 次元）の上の線形分類器は，訓練行 2,275 行の局所的な構造を捉えきれていない．同じ埋め込み空間での近傍 2 本のラベル分布を LR の確率に混ぜると，LR の誤りの一部が近傍のラベルで覆る．その結果 rank 1 の決定が変わり，top1 が上がる．根拠は CV で +1.93pt（5 fold すべてで上回る），eval の replay で +1.04pt である．補間の形は kNN-LM（Khandelwal ら，ICLR 2020，arXiv:1911.00172）と KNN-BERT（Li ら，2021，arXiv:2110.02523）に倣う．

**単一レバー**: `knn_label_distribution_interpolation`．分類器 artifact の出力を，LR 単体の \(p_{LR}\)（λ=0 に相当）から，次の補間分布へ変える．

\[ p = \lambda \, p_{kNN,k} + (1-\lambda) \, p_{LR}, \quad k=2,\ \lambda=0.3 \]

\(p_{kNN,k}\) は，L2 正規化した融合埋め込みの cosine 類似度で上位 k 本の訓練行を取り，そのラベルに `_extract_sample_weights()` と同じクラス均衡重みで票を入れて正規化したものである．

**固定する構成（Iter101 の採用構成）**:

- \(p_{LR}\) は既存の `models/domain_classifier.joblib`（MD5 `c8cd0fb46549d03db11f43864a741ef8`）を**再訓練せずに**そのまま包む．
- 近傍の母集団は `data/classifier_train_iter94_dedup.jsonl` の 2,275 行で，埋め込みは `data/embcache_train_iter100_fused.npy` を使う．再埋め込みはしない．
- 融合表現（`embedding_fusion_models`），`dispatch_gap_threshold` 0.36，`dispatch_gap_max_k` 4，`aggregation_method: max_confidence`，`confidence_threshold` と `dispatch_candidate_threshold` 0.0，評価集合 3,750 行，`config.yaml` の `classifier_model_path` はいずれも変えない．

**レバーを読むコード行と到達条件**:

- 読み込み: `http_server.py:428` が `load_domain_classifier(state.classifier_model_path)` を呼び，`classifier.py:48` の `joblib.load` が artifact を読む．
- 推定: `http_server.py:379` が `estimate_confidence_classifier()` を呼び，`classifier.py:66`，`:68` が `classes_` を，`:69` が `predict_proba([query_embedding])[0]` を読む．
- 到達条件: `routing_method: supervised_classifier`（`config.yaml:75`）かつ `confidence_signal_method: self_report` であること．このとき `_estimate_probe_confidence()` は LLM を呼ばずに上の経路へ落ちる（Iter101 A1）．加えて，ノードのドメインが `classes_` に含まれること．**現行構成は 3 つとも満たす**．
- `predict_proba` の出力が変われば `probe_candidates` の確信度が変わり，max_confidence の rank 1 と gap 方式の送出集合がそのまま変わる．no-op になる経路は無い．

**実装の仕様（フェーズ 2）**:

1. リポジトリ直下に `knn_interpolated_head.py` を新設し，`KnnInterpolatedClassifier` を置く（`ecoc_head.py` と同じ理由である．joblib はクラスをモジュールの名前で直列化するので，訓練スクリプトとノードの両方から同じ名前で import できる場所に置く必要がある）．
   - 保持するもの: `base_classifier`（既存の `CalibratedClassifierCV`），`train_embeddings_normalized`（float64，2,275 × 6,656），`train_label_indices`，`train_weights`，`k`，`interpolation_lambda`．
   - `classes_` は `base_classifier.classes_` と同じ配列にする．
   - `predict_proba(X)` は行和が 1 の (n, 10) 配列を返す．近傍の同順位は訓練行の添字が小さい方を取るなど，決定的な規則を決める．
   - 本走の経路が読む属性は `classes_` と `predict_proba` だけである（上の網羅を参照）．`calibrated_classifiers_` などは提供しなくてよい．
2. `scripts/build_knn_interpolated_classifier.py` を新設する．既存の artifact と訓練キャッシュを読み，ラッパーを組み立てて保存する．
   - 引数: `--base-classifier`，`--train-cache`，`--train-data`，`--k 2`，`--interpolation-lambda 0.3`，`--output`．
   - キャッシュのメタデータの `ids` と `domains` が `--train-data` の行と順序まで一致することを検査し，一致しなければ止める．
3. 既存の artifact は `models/domain_classifier_pre_iter104_lr.joblib` へ退避する（MD5 `c8cd0fb46549d03db11f43864a741ef8` のまま）．新しい artifact を `models/domain_classifier.joblib` に置き，MD5 を journal に記録する．大きさは約 121MB の見込みである（float64 の埋め込み）．
4. `Dockerfile:14` の COPY に `knn_interpolated_head.py` を追加する．`models/` をノードへ届ける経路（イメージに焼くかマウントか）は，Dockerfile に `models/` の COPY が見当たらないので，deploy 手順で確かめる．
5. テストを追加する．行和が 1 になること，`classes_` が元の artifact と一致すること，λ=0 で元の artifact と出力が一致すること，joblib で往復しても出力が変わらないこと．

**G0（本走の前に，すべて合格が条件）**:

- **G0-a**: 新しい artifact で eval キャッシュの `predict_proba` を求める．top1（基準線の行の決定だけを差し替えたもの）が 0.8504 ± 0.0005 であり，argmax が `result_v2.json` の replay と 99.9% 以上一致すること．
- **G0-b**: 全 10 ノードで artifact の MD5 が一致し，起動ログで読み込んだクラスが `KnnInterpolatedClassifier` であること．
- **G0-c**: 予備 20 問で `probe_candidates` の確信度が，同じ問いのキャッシュ埋め込みで求めたオフラインの値と argmax で一致すること．`mean_duration_ms` が 3,041.7ms 以下であること．

**成功条件と非退行条件（事前登録）**:

- **基準線**: Iter101 の採用本走 `results/20260929_192157`．top1 0.8400，複合行の top1 0.805479，`compound_domain_set_recall` 0.578767，`compound_mean_dispatched_count` 1.973973，ECE 0.039551，`mean_duration_ms` 2197.84．
- **判定規則**:
  - `adopted`: Δtop1 ≥ +0.5pt（top1 ≥ 0.8450），McNemar p < 0.05，C1〜C7 をすべて満たす．
  - `adopted_small`: Δtop1 ≥ +0.5pt だが p ≥ 0.05（要再現）．
  - `partial`: +0.25pt ≤ Δtop1 < +0.5pt で，C1〜C7 を満たす．
  - `negligible`: |Δtop1| < 0.25pt．
  - `rejected`: Δtop1 ≤ −0.25pt，または C1〜C7 のどれかに違反．
  - `invalid`: G0 に合格せず本走に至らない．
- **非退行条件**: Iter101 の C1〜C7 を数値もそのまま使う（オーケストレータの決定 A1．replay の結果を見てから緩めることも締めることもしない）．
  - **C1**: precision と recall の計 20 指標を BH 補正（q=0.05）し，有意な退行が 0 件．
  - **C2**: `fallback_rate` 0.0，`dispatch_failure_rate` ≤ 0.005．
  - **C3**: レバー発火の証拠（G0-a〜G0-c の記録と artifact の MD5）がすべて残っていること．Iter101 の C3 は light_model の起動ログが対象だったので，本反復のレバーに合わせて対象を置き換えた．
  - **C4**: 複合行の top1 ≥ 0.780411．
  - **C5**: `compound_domain_set_recall` ≥ 0.5400，`compound_mean_dispatched_count` ≤ 2.10．
  - **C6**: `mean_duration_ms` ≤ 3041.7．単一層の中央値と p95，複合層の p25 と中央値を併記する．
  - **C7**: ECE ≤ 0.08 を毎回併記する．0.05 を超えたら，C7 を満たしていても次の反復のレバーを校正側に立てる．

**事前登録する予測**:

- **P1（主予測）**: Δtop1 は +0.7〜+1.3pt（点推定 +1.0pt，top1 0.850 前後）．McNemar p < 0.05．replay では，基準 artifact の再現と本走の間に 0.11pt の差があるので，その幅を区間に含めた．
- **P2**: C1 は満たす見込みが高い．ただし `computer_science_recall`（replay で 3 対 10）と `medical_recall`（12 対 23）が退行の向きにある．本走の揺れで有意に転じ，C1 違反で `rejected` となる見込みは 2 割から 3 割とみる（推測）．オーケストレータは第 1 版の結果から「C1 に抵触して rejected となる可能性が高い」とした．しかし第 2 版の k=2，λ=0.3 では，20 指標の補正で有意な退行は 0 件だった．そのため評価を「中程度の危険」に改める．
- **P3**: `compound_mean_dispatched_count` は 2.03〜2.10 で，C5 の上限 2.10 に近い．超えれば `rejected` になる．C5 違反は C1 と並ぶ主な危険である．
- **P4**: ECE は 0.045〜0.055 で，C7 は満たす．replay の値 0.0503 は 0.05 を超えるので，C7 の規定どおり，**次の反復のレバーを校正側（補間分布に対する温度校正の見直し）に立てる**見込みである．
- **P5**: `mean_duration_ms` は基準線から ±5% 以内（C6 の内側）．kNN の追加の計算は，1 問ごとに 2,275 × 6,656 の内積で，CPU で数 ms とみる．

**フェーズ 2 への申し送り**:

- コードの変更は上の「実装の仕様」の 1〜5 に限る．`config.yaml` は変えない．
- 埋め込みの再計算と LR の再訓練はしない．ラッパーの組み立てと G0-a は開発ホストの CPU で行える（`uv run`）．単一 GPU のサブ実験が必要になった場合は wafl-ctrl5 を使う．ただし本フェーズでは wafl-ctrl5 への SSH が許可システムに阻止された．
- 本走は 3,750 問のフルスペックで行う．deploy の後，本走の前に G0-b と G0-c を行う．
- 判定の比較相手は `results/20260929_192157`．C1 の計算は `metrics.py` の関数で行う（`scripts/_iter101_post_analysis.py` と同じ手順）．

### Iteration 104 実装・実験（2026-10-04）

**変更したファイル**（コミットはフェーズ 3 で行う）:

- `knn_interpolated_head.py`（新設）: `KnnInterpolatedClassifier`．p = λ·p_kNN,k + (1−λ)·p_base を返す．`classes_` は base の値をそのまま返す．近傍の同順位は `argsort(kind="stable")` で添字の小さい訓練行を取る．λ=0 のときは base の出力をそのまま返す．
- `scripts/build_knn_interpolated_classifier.py`（新設）: キャッシュのメタデータの `ids` と `domains` が訓練データと順序まで一致しなければ止める．重みは `scripts/train_domain_classifier.py:_extract_sample_weights()` を再利用した．
- `tests/test_knn_interpolated_head.py`（新設，9 件）．
- `Dockerfile` の 14 行目の COPY に `knn_interpolated_head.py` を追加した．
- `config.yaml` は変えていない．

**artifact とイメージ**:

- 新しい `models/domain_classifier.joblib` は MD5 `c3888f66d90f4172cd7415e5c105328a`，123,846,618 B（k=2，λ=0.3，n_train=2275，dim=6656）である．
- 退避した `models/domain_classifier_pre_iter104_lr.joblib` は MD5 `c8cd0fb46549d03db11f43864a741ef8`（元と一致），2,670,565 B である．
- artifact は `mise run deploy` の rsync で各ノードの `models/` へ届き，`./models:/app/models:ro` でマウントされる．`knn_interpolated_head.py` はイメージに焼き込まれるので，イメージを再ビルドした．
- 再ビルドは B192 の A1 に従い，`mise.toml` の 35 行目（`docker build`）と 38 行目（`docker push`）だけを実行した．`mise run setup` は実行していない．`--build-arg GIT_HEAD=7c0b703-dirty-iter104` を渡した（B192 の B1）．image は `b5b717d1fc7a`，push digest は `sha256:0a499d0e172319c7af21a09b64d46203a2872981e2dab9e1a23b1fc17f505868` である．
- `data/dataset.jsonl` の MD5 は，ビルドの前後とも `769f2a58dc807e23e1b73e44b97e97b8`（3,750 行）で，イメージの中の値も同じである．ログは `.claude/research/_iter104_build.log`．

**テストと lint**: `uv run pytest -q tests/test_knn_interpolated_head.py tests/test_classifier.py` は 16 passed．新設した 3 ファイルの `ruff check` は clean．全体のテストは実行していない（Iter103 の時点で，環境に依存する既知の失敗が 20 件ある）．

**G0**:

- **G0-a: 合格**（`/tmp/iter104/g0a_check.py`）．eval 3,750 行での top1 は 0.8504（`metrics.compute_top1_accuracy`），replay 第 2 版との argmax の一致率は 1.0（不一致 0 件），確率の最大絶対差は 0.0 だった．
- **G0-b: 合格**（`.claude/research/_iter104_g0b.txt`）．wafl500〜509 の全 10 ノードで，`/app/models/domain_classifier.joblib` は `c3888f66…`，`/app/knn_interpolated_head.py` は `3c98c9ba687609a34d485c59eb367db8` で，ローカルと一致した．コンテナの中で artifact を `joblib.load` すると `KnnInterpolatedClassifier`（k=2，λ=0.3）だった．`http_server.py` は読み込んだクラス名をログに出さないので，起動ログの代わりにこの方法で確かめた．起動ログの traceback，ModuleNotFoundError，error の行は 0 件だった．`mise run deploy` は EXIT=0 で，`smoke_check` の hashes と probe は passed（probe の latency は 17ms）だった．
- **G0-c: 合格**（`results/20261004_132357/preview20.jsonl`，`data/dataset_20.jsonl`）．オンラインとオフラインの argmax は 20/20 で一致し，`mean_duration_ms` は 995.95（上限 3,041.7）だった．dispatch の失敗と fallback は 0 件だった．確信度の差は最大 0.0117（business_economics-014）である．

**本走**: `results/20261004_132607`（3,750 問，requester は wafl500，13:26:07 に起動し 15:47 に完了）．

- `results.jsonl` は 3,750 行，MD5 は `11b01b94d04099ae7b34f726754bf00b`（wafl500 側と一致），`git_head.txt` は `7c0b703-dirty-iter104` である．
- `mise run analyze -- 20261004_132607` と `uv run python metrics.py --results results/20261004_132607/results.jsonl --json`（出力は `metrics.json`）は，どちらも EXIT=0 だった．

| 指標 | 値 |
|---|---|
| top1_accuracy | 0.849867（Wilson CI 0.838076〜0.860941） |
| single_domain_top1_accuracy（3,020 行） | 0.860596 |
| compound_domain_top1_accuracy（730 行） | 0.805479 |
| compound_domain_set_recall | 0.589041 |
| compound_mean_dispatched_count | 2.064384 |
| compound_domain_jaccard_mean | 0.444521 |
| ECE | 0.050104 |
| Brier | 0.102122 |
| AUROC | 0.848986 |
| cohens_kappa | 0.844022 |
| mean_duration_ms | 2253.224 |
| fallback_rate | 0.0 |
| dispatch_failure_rate | 0.0 |
| tie_rate | 0.0 |
| answer_quality_accuracy（axis23，3,020 行） | 0.584437 |
| end_to_end_accuracy（axis23） | 0.4016 |

**実行とログの上の異常**:

- 全 10 ノードのログで，OOM，Traceback，ModuleNotFoundError，`dispatch_model_not_ready` は 0 件だった．
- ローカルで `mise run start` の完了を待っていたバックグラウンドのタスクが，2 時間の上限に達して止められた（`compound-721` の表示の時点）．実験はコンテナの中で切り離して動いていたので止まっていない．`mise run start` は再起動せず，wafl500 のログと完了マーカーを直接ポーリングした．完了した後で，`mise run start` の最後の段と同じ `ssh ... cat` で `results.jsonl` と `run_experiment.log` をローカルへ写した．このため `_iter104_mainrun_start.log` の末尾は `[start] ERROR task failed` になっているが，実験の結果には影響していない．

**要人間判断の事項**:

- 実行の途中で「deploy の後に `docker image prune -a -f` を実行する」「実験の後に docker container を止める（image は残す）」という依頼が届いた．イメージの削除は，実装・実験フェーズの禁止事項（モデルやイメージの削除）に当たり，deploy の手順の変更は今回の単一レバーの範囲を超えるので，このフェーズでは実行も手順の変更もしていない．扱いは戻り値の選択肢で確認する．
- `tools/smoke_check.py` の `DEPLOYED_FILES` に `knn_interpolated_head.py` と artifact が入っていない（別の提案として扱う．今回は変更していない）．
- イメージは未コミットの作業ツリーから作った．フェーズ 3 のコミットの後で，コミットの内容とイメージの中身（`knn_interpolated_head.py` の MD5 `3c98c9ba…`）が一致することを確かめる必要がある．

### Iteration 104 実行済み

#### 計算の方法

- スクリプトは `/tmp/iter104_eval.py`（一時ファイル）で，出力は `.claude/research/_iter104_eval_out.txt`（271 行）に保存した．`uv run` で実行し，EXIT=0 だった．
- 比較の相手は基準線 `results/20260929_192157/results.jsonl` である．id は 3,750 行で完全に一致する．
- top1 の対の比較には `metrics.compute_mcnemar_test`（連続補正つき）を使った．参考として，不一致の対に `scipy.stats.binomtest` の正確検定も当てた．
- C1 には `compute_domain_recall_mcnemar_test`，`compute_domain_precision_fisher_test`，`apply_benjamini_hochberg`（q=0.05）を使った（`scripts/_iter101_post_analysis.py` と同じ手順）．C6 の分位点は `numpy.percentile` で求めた．

#### 結果（本走 `results/20261004_132607` 対 基準線 `results/20260929_192157`）

| 指標 | 基準線 | 本走 | Δ | 事前登録の予測 |
|---|---|---|---|---|
| top1 | 0.840000 | **0.849867** | **+0.9867pt（+37 行）** | P1: +0.7〜+1.3pt．的中 |
| McNemar（連続補正） | — | 本走のみ正答 103，基準線のみ正答 66，chi2 7.6686，**p = 0.005619** | — | p < 0.05．的中 |
| 参考: 正確二項検定 | — | p = 0.005461 | — | — |
| 単一行の top1（3,020 行） | 0.848344 | 0.860596 | +1.2252pt（84 対 47） | — |
| 複合行の top1（730 行） | 0.805479 | 0.805479 | 0（19 対 19） | — |
| `compound_domain_set_recall` | 0.578767（845/1460） | 0.589041（860/1460） | +1.03pt | — |
| `compound_mean_dispatched_count` | 1.973973 | 2.064384 | +0.0904 | P3: 2.03〜2.10．的中 |
| overall mean_k | 1.533067 | 1.569600 | +2.4% | — |
| 複合の k 分布 | {1: 479, 2: 21, 4: 230} | {1: 451, 2: 30, 4: 249} | — | — |
| 単一の k 分布 | {1: 2546, 2: 67, 4: 407} | {1: 2507, 2: 90, 4: 423} | — | — |
| ECE | 0.039551 | **0.050104** | +0.0106 | P4: 0.045〜0.055．的中 |
| `mean_duration_ms` | 2197.842 | 2253.224 | +2.52% | P5: ±5% 以内．的中 |
| `fallback_rate` / `dispatch_failure_rate` | 0.0 / 0.000267 | 0.0 / 0.0 | — | — |

#### 事前登録条件の照合（C1〜C7，Iter101 と同じ閾値）

| 条件 | 基準 | 実測 | 判定 |
|---|---|---|---|
| 主基準 | Δtop1 ≥ +0.5pt かつ McNemar p < 0.05 | +0.9867pt，p = 0.005619 | PASS |
| C1 | 20 指標の BH 補正（q=0.05）後の有意な退行が 0 件 | 0 件（有意の 2 件はどちらも改善の向き） | PASS |
| C2 | `fallback_rate` = 0.0，`dispatch_failure_rate` ≤ 0.005 | 0.0，0.0 | PASS |
| C3 | G0-a〜G0-c の記録と artifact の MD5 | G0-a〜c はすべて合格．MD5 `c3888f66d90f4172cd7415e5c105328a` は全 10 ノードで一致 | PASS |
| C4 | 複合 top1 ≥ 0.780411 | 0.805479 | PASS |
| C5 | set_recall ≥ 0.5400 かつ mean_dispatched ≤ 2.10 | 0.589041 / 2.064384 | PASS（上限まで残り 0.036） |
| C6 | `mean_duration_ms` ≤ 3041.7 | 2253.224 | PASS |
| C7 | ECE ≤ 0.08 | 0.050104 | PASS．ただし 0.05 を超えたので，規定により次の反復のレバーは校正側に立てる |

- **C1 の内訳**:
  - BH 補正の後に有意となったのは `business_economics_recall`（本走のみ正答 17 対 基準線のみ正答 3，p = 0.00365）と `education_recall`（37 対 14，p = 0.00207）の 2 件で，どちらも改善の向きである．
  - 改善の向きで補正の後に有意でないもの: `mathematics_recall` 10 対 1（p = 0.0159），`social_science_recall` 19 対 6（p = 0.0164）．
  - 退行の向きのもの: `medical_recall` 12 対 24（p = 0.0668），`computer_science_recall` 3 対 10（p = 0.0961），`natural_science_recall` 6 対 9（p = 0.61），`general_recall` 3 対 5（p = 0.72）．いずれも補正の前でも p ≥ 0.05 である．
  - precision の 10 指標は，すべて p ≥ 0.116 である．
- **C6 の分位点**（括弧内は基準線）:
  - 単一層: 中央値 539ms（511），p95 3,252.4ms（3,017.25）．
  - 複合層: p25 7,583.5ms（7,460），中央値 8,795.5ms（8,732）．
  - 全体: 中央値 569ms（545），p95 8,935ms（8,887），max 9,184ms（9,265）．

#### 判定（分析フェーズ，2026-10-04）

- **判定: `adopted`**．事前登録の判定規則（「計画 (Iter104)」）をそのまま当てはめた．閾値は緩めても締めてもいない．Δtop1 ≥ +0.5pt，McNemar p < 0.05，C1〜C7 をすべて満たす．
- **採用構成**: `models/domain_classifier.joblib`（`KnnInterpolatedClassifier`，k=2，λ=0.3，MD5 `c3888f66d90f4172cd7415e5c105328a`）を以後の基準とする．元の LR の artifact は `models/domain_classifier_pre_iter104_lr.joblib`（MD5 `c8cd0fb46549d03db11f43864a741ef8`）として残す．新しい基準線は本走 `results/20261004_132607` である．
- **雑音と信号の切り分け**:
  - 同じ LR の artifact で走った過去の 4 本（Iter100 0.839467，Iter101 0.840000，Iter102 0.839467，Iter103 の反実仮想 0.839200）の top1 は，標本標準偏差が 0.034pt，幅が 0.08pt である．埋め込みが同じなら分類器の決定は決定的で，実行間の差は主に送出の失敗と埋め込みの微小な差から生じる．今回の +0.99pt はこの幅の 10 倍以上あり，対の検定でも p = 0.0056 である．したがって雑音ではなくレバーの効果とみる．
  - replay との一致: replay は 103 対 64（top1 0.8504），本走は 103 対 66（0.849867）で，差は基準線のみ正答の 2 行（いずれも複合行）である．単一行の top1 は replay の 0.8606 と一致した．set_recall は replay 0.5904 に対し 0.589041（2 ドメイン分），mean_dispatched は 2.0685 に対し 2.064384 である．この差は，調査の時点で測った「基準 artifact の再現と本走の差 0.11pt」の範囲内にある．
  - 雑音で説明できない範囲は「この評価集合の上での +1pt」までである．(k, λ) は CV だけで選んだが，eval の結果を見た後に格子を広げた経緯（B191 (5)）がある．別の評価集合への汎化は確かめていない．
- **計画の仮説との一致**:
  - 「近傍のラベル分布を混ぜると rank 1 の決定が変わり，top1 が上がる」は支持された．利得はすべて単一行で出た（84 対 47，正味 +37 行）．複合行は 19 対 19 で正味 0 である．
  - 「線形分類器が局所的な構造を捉えきれていない」という機序は，直接には確かめていない．確かめたのは補間によって決定が変わり，その差し引きが正になったことまでである．
  - recall の差し引きは education（+23 行），business_economics（+14），social_science（+13），mathematics（+9）が増え，medical（−12），computer_science（−7）が減った．recall が低いドメインが増えているので，kNN の票に入れたクラス均衡重みが小さいクラスへ決定を寄せた可能性がある（推測．重みを外した replay で確かめられる）．
- **想定外の挙動**: 言語崩れ，発散，OOM は無い（全 10 ノードのログで 0 件）．予測していた変化として，確信度の分布が平らになり，gap 方式の送出数が増えた（4 件送出の行が複合 +19 行，単一 +16 行）．この分だけ `mean_duration_ms` が +2.5%，単一層の p95 が +235ms 増えた．
- **後始末**: config.yaml は変えていないので，戻す設定は無い．イメージ `b5b717d1fc7a` は `GIT_HEAD=7c0b703-dirty-iter104` で作った．本走の `git_head.txt` もこの値のままである．フェーズ 3 のコミットに入れた `knn_interpolated_head.py` の MD5 は，イメージの中の値 `3c98c9ba687609a34d485c59eb367db8` と照合した（照合の結果は B193 に記録する）．

#### 学び (Iter104)

1. **実測せずに却下したレバーが効いた**．Iter97 の計画は，kNN 融合を「5,120 次元 対 2,275 行の p≫n で過学習が濃厚」という理由で，実測なしに却下していた．補間の形なら学習される重みは増えず，自由度は k と λ の 2 つだけである．過学習の議論はこの形にはそのまま当てはまらなかった．却下の理由が自由度の見積もりであるときは，その見積もりが候補の実際の形に当てはまるかを確かめる．
2. **artifact を差し替えるレバーは，キャッシュの埋め込みを使った replay でほぼ正確に予測できる**．本走との差は 2 行（0.053pt）だった．分類器の側のレバーは，今後もこの replay で事前に絞り込める．ただし replay は事前登録の手段に留める（恒久ルール，B176）．
3. **rank 1 を変えるレバーは，確信度の gap を通じて送出集合も動かす**．今回は mean_dispatched が 1.974 → 2.064 になり，C5 の上限 2.10 まで残り 0.036 である．次に校正（温度）を変える場合，argmax は変わらないが gap が変わるので，set_recall，mean_dispatched，所要時間は動く．C5 は replay で事前に見積もってから事前登録すること．
4. **補間分布は温度校正の外にある**．温度は LR の出力だけに当てたもので，k=2 の kNN 分布（値は 0，約 0.5，1 に偏る）を 0.3 混ぜた後の分布には当て直していない．ECE が 0.0396 → 0.0501 になったのはこのためとみる（推測）．
5. **待機の上限と実験の寿命を分けておくと，記録が壊れない**．ローカルの待機タスクが 2 時間で止められても，実験はコンテナの中で切り離して走り続け，結果は完了マーカーとノード側の MD5 で確かめられた．`_iter104_mainrun_start.log` の末尾の ERROR は，待機タスクが止められたことだけを示す．

