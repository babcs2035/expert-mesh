<!-- docs/ 配下の文書の索引．各文書がどの Iteration 範囲を対象とし，現行か歴史的記録かを示す． -->

# docs/ 索引

このディレクトリには，expert-mesh の設計・調査・研究サイクルの知見をまとめた文書を置く．

**文書は書かれた時点の到達点を記録したものであり，後続のイテレーションで覆されることがある．**
過去に「d0002 が最新だと思って計画を立てたら，記載内容がすでに実施済みだった」という事故が起きて
いるため，**必ず本索引で対象範囲と状態を確認してから読むこと**．

## 読む順序

1. `encounter_expert_mesh_design.md` — 何を作ろうとしているのか（設計・評価軸の定義）
2. `d0006` — 研究として何が分かったのか（**現況．収束時点の最新総括**）
3. 必要に応じて `d0004`（Iter15〜27 の詳細）や `d0001`〜`d0003`（初期の経緯）を参照

## 文書一覧

| 文書 | 対象 Iteration | 執筆時点 | 状態 |
|---|---|---|---|
| `encounter_expert_mesh_design.md` | 全期間 | v2 | **現行**．技術設計書．ノード構成・4 評価軸・研究の問いの定義元 |
| `d0001_literature_survey_2026-07.md` | Iter1〜14 | 2026-07-26 | 歴史的記録．文献調査（tavily）と実測の一次記録 |
| `d0002_research_cycle_findings_2026-07.md` | Iter1〜22 | 2026-07-29 | 歴史的記録．§4「現在の到達点」は Iter25/26 で更新済み |
| `d0003_next_experiments_2026-07.md` | Iter22 時点の計画 | 2026-07-29 | 歴史的記録．F1〜F5・X1・X2・X4 は実施済み |
| `d0004_research_status_and_direction_2026-08.md` | Iter15〜27 | 2026-07-31 | 部分的に有効．Iter27 で更新停止．§4（no-op 実験の構造的原因）は今も有効 |
| `d0005_retraining_analysis_2026-08.md` | Iter53 時点 | 2026-08-03 | 有効．post-hoc 天井と retraining 移行の分析．§「次の一手」は d0006 §2.6 で覆っている |
| **`d0006_research_summary_iter28-54_2026-08.md`** | **Iter28〜54** | **2026-08-05** | **現況**．研究収束時点の最新総括．未解決の要人間判断もここにある |

関連: `plans/p0001_research_direction_2026-07.md`（Iter1〜14 の方針決定．`d0001` と内容が大きく重複する）

## 研究サイクルの一次記録

docs/ は要約であり，一次記録は `.claude/research/` にある．

| ファイル | 内容 |
|---|---|
| `.claude/research/journal.md` | 直近 3 イテレーションの生記録（逆時系列） |
| `.claude/research/journal_archive.md` | Iteration 1〜51 の生記録（逆時系列） |
| `.claude/research/experiment_results.json` | 各イテレーションの機械可読な実験結果（`e{n}_results`） |
| `.claude/research/backlog.md` | 自律判断の記録と要人間判断事項（`B{n}`） |
| `.claude/research/config.yml` | レバー定義・成功条件・研究フロンティア |
| `.claude/research/state.json` | research-cycle の制御状態（13 キー） |

## 命名規約

- `d{4桁}_{題目}_{年月}.md` — 複数ファイルにまたがる機能や知見の説明．機能変更時に更新する
- 大規模な変更の計画は `plans/p{4桁}_{題目}.md` に置く
- 起動方法やフォルダごとの役割は，リポジトリルートの `README.md` に置く
