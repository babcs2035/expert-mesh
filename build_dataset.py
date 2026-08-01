"""Build the evaluation dataset (design doc 4.3) from JMMLU, fixed at 10 domains.

JMMLU (https://huggingface.co/datasets/nlp-waseda/JMMLU, commit
3637b25e444ccfdcde4d23a783cbe8e674faa01b) is a 56-task, 7,536-question
Japanese four-choice benchmark. This module maps its 56 tasks onto the 10
mesh domains below and samples up to `--domain-target-size` questions per
domain, so the same underlying question pool can support both the routing
(axis 1) and, via the `jmmlu_answer` field, answer-quality (axis 2) metrics.

The dataset is fixed at 10 domains (medical, legal, education,
business_economics, computer_science, natural_science, mathematics,
history_culture, social_science, general); there is no 4-domain mode.

Known mapping limitations (see docs/d0001_literature_survey_2026-07.md and
plans/p0001_research_direction_2026-07.md for the underlying research
rationale):

- `legal` has only 2 constituent tasks (international_law, jurisprudence;
  227 questions total) because JMMLU has no `professional_law` task (unlike
  the English MMLU it derives from). This is a hard ceiling: legal cannot
  reach the 150-question target of the other domains without duplication,
  so it is capped at its actual pool size.
- `education` has no directly corresponding JMMLU task; sociology,
  high_school_psychology, and moral_disputes (448 questions) are used as a
  proxy for the mesh's actual education-administration domain. This is a
  deliberate compromise, not a claim that these tasks measure the same
  thing as the hand-authored education questions used for compound rows.

Licensing: the entire JMMLU dataset is CC BY-NC-ND 4.0 (non-commercial,
no-derivatives; research/evaluation use is explicitly permitted). Five
tasks (japanese_history, world_history, japanese_idiom, japanese_civics,
japanese_geography) additionally carry a named-copyright-holder clause
that separately confirms research/evaluation use is allowed. All five fall
under `history_culture` in this mapping. `--exclude-restricted-license-tasks`
is provided so a future non-research redistribution of this dataset can
opt out of them without needing a code change; it is off by default since
this project's use (routing research) already qualifies as permitted use.

Usage:
    uv run python build_dataset.py --output data/dataset.jsonl
    uv run python build_dataset.py --output data/dataset.jsonl --jmmlu-zip /path/to/JMMLU.zip
    uv run python build_dataset.py --output data/dataset.jsonl --exclude-restricted-license-tasks
"""

import argparse
import csv
import io
import json
import os
import random
import sys
import zipfile
from typing import TextIO

import httpx

# Pinned to a specific commit so the sampled dataset does not silently
# change if upstream revises or re-translates questions after this was written.
_JMMLU_ZIP_SHA = "3637b25e444ccfdcde4d23a783cbe8e674faa01b"
_JMMLU_ZIP_URL = (
    f"https://huggingface.co/datasets/nlp-waseda/JMMLU/resolve/{_JMMLU_ZIP_SHA}/JMMLU.zip"
)
_JMMLU_DOWNLOAD_TIMEOUT_S = 60.0
_JMMLU_CSV_PATH_TEMPLATE = "JMMLU/test/{task_name}.csv"

# Target question count per domain. legal's actual pool (227) is below this;
# _sample_domain_questions caps at the pool size rather than erroring.
_DOMAIN_TARGET_SIZE = 150
# Fixed seed so the same JMMLU.zip always yields the same sampled dataset.
_JMMLU_SAMPLE_SEED = 20260726
# Distinct seed for E6's classifier training pool (build_classifier_training_rows),
# so its sample is independent of (and, via exclude_queries, disjoint from) the
# evaluation dataset's sample — see that function's docstring.
_CLASSIFIER_TRAIN_SAMPLE_SEED = 20260727

# Task -> domain mapping. Every one of JMMLU's 56 tasks appears in exactly
# one domain's list (verified against the task_list in JMMLU.py at the
# pinned commit); see the module docstring for the rationale behind the
# legal/education assignments specifically.
_DOMAIN_TASK_MAP: dict[str, list[str]] = {
    "medical": [
        "virology",
        "nutrition",
        "human_sexuality",
        "clinical_knowledge",
        "human_aging",
        "anatomy",
        "professional_psychology",
        "college_medicine",
        "professional_medicine",
        "medical_genetics",
    ],
    "legal": [
        "international_law",
        "jurisprudence",
    ],
    "education": [
        "sociology",
        "high_school_psychology",
        "moral_disputes",
    ],
    "business_economics": [
        "econometrics",
        "high_school_microeconomics",
        "business_ethics",
        "marketing",
        "high_school_macroeconomics",
        "management",
        "public_relations",
        "professional_accounting",
    ],
    "computer_science": [
        "computer_security",
        "machine_learning",
        "high_school_computer_science",
        "college_computer_science",
        "electrical_engineering",
    ],
    "natural_science": [
        "high_school_chemistry",
        "high_school_physics",
        "college_physics",
        "conceptual_physics",
        "college_biology",
        "high_school_biology",
        "college_chemistry",
        "astronomy",
    ],
    "mathematics": [
        "college_mathematics",
        "high_school_statistics",
        "elementary_mathematics",
        "high_school_mathematics",
        "abstract_algebra",
    ],
    "history_culture": [
        "japanese_history",
        "japanese_civics",
        "high_school_european_history",
        "prehistory",
        "japanese_idiom",
        "japanese_geography",
        "high_school_geography",
        "world_history",
    ],
    "social_science": [
        "security_studies",
        "world_religions",
        "philosophy",
        "global_facts",
    ],
    "general": [
        "miscellaneous",
        "logical_fallacies",
        "formal_logic",
    ],
}

# Iter32 (classifier_training_data_composition=education_proxy_task_revision, Y5) で
# 導入されたsample_weight機構は，`class_weight="balanced"`との数式結合によりIter32計画の
# 意図に反し逆効果と判明したためrejected・revert済み（backlog B53参照）。
# Iter33以降は`education_proxy_task_resampling`（抽出段階でのタスク別目標件数変更）に
# 移行し，`sample_weight`は使わない設計とする。_CLASSIFIER_TASK_SAMPLE_WEIGHTSは空辞書であり，
# _classifier_task_sample_weight()はすべてのタスクで1.0を返す（no-op）。
_CLASSIFIER_TASK_SAMPLE_WEIGHTS: dict[str, float] = {}
_DEFAULT_CLASSIFIER_TASK_SAMPLE_WEIGHT = 1.0

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


def _classifier_task_sample_weight(task_name: str) -> float:
    """Per-row training weight for build_classifier_training_rows() (Iter32)."""
    return _CLASSIFIER_TASK_SAMPLE_WEIGHTS.get(task_name, _DEFAULT_CLASSIFIER_TASK_SAMPLE_WEIGHT)


_RESTRICTED_LICENSE_TASKS: frozenset[str] = frozenset(
    {
        "japanese_history",
        "world_history",
        "japanese_idiom",
        "japanese_civics",
        "japanese_geography",
    }
)

# Hand-authored compound-domain questions (design doc 4.3: "questions
# spanning multiple domains"). JMMLU's four-choice questions each belong to
# a single task and cannot express genuine cross-domain ambiguity, so these
# remain hand-authored rather than JMMLU-derived.
_COMPOUND_QUESTIONS: list[tuple[str, list[str]]] = [
    (
        "仕事中に転倒して怪我をしました．治療費と休業補償について知りたいです．",
        ["medical", "legal"],
    ),
    ("交通事故で怪我をして通院していますが，慰謝料の相場が分かりません．", ["medical", "legal"]),
    ("職場のハラスメントでうつ状態になり，休職を検討しています．", ["medical", "legal"]),
    (
        "ペットが近隣トラブルの原因で怪我をさせてしまいました．治療費と責任について知りたいです．",
        ["medical", "legal"],
    ),
    (
        "学校で子供のアレルギー対応について，給食と保健室の両方の配慮が必要です．",
        ["education", "medical"],
    ),
    ("いじめの問題で，学校への対応と法的なアドバイスが必要です．", ["education", "legal"]),
    (
        "交通事故で後遺障害が残り，後遺障害等級認定の手続きと今後の通院方針の両方について相談したいです．",
        ["medical", "legal"],
    ),
    (
        "職場の化学物質にばく露して体調を崩しました．労災認定と治療方針を教えてください．",
        ["medical", "legal"],
    ),
    (
        "医療事故に遭った可能性があります．診療記録の開示請求と今後の治療についてどう進めればよいですか．",
        ["medical", "legal"],
    ),
    (
        "高齢の親が施設で転倒し骨折しました．施設側の責任追及と治療の両方を検討しています．",
        ["medical", "legal"],
    ),
    (
        "スポーツ中の事故で相手にケガを負わせてしまいました．治療費の負担と損害賠償請求への対応を知りたいです．",
        ["medical", "legal"],
    ),
    (
        "感染症にかかった従業員がいる職場で，就業制限の法的根拠と医学的な対応基準を知りたいです．",
        ["medical", "legal"],
    ),
    (
        "美容医療の施術後に合併症が出ました．治療方針の相談と施術業者への責任追及を同時に進めたいです．",
        ["medical", "legal"],
    ),
    (
        "ペットに噛まれてケガをしました．治療費の請求先と飼い主の法的責任について知りたいです．",
        ["medical", "legal"],
    ),
    (
        "学校での部活動中の熱中症で生徒が搬送されました．今後の予防策と応急対応の指導について知りたいです．",
        ["education", "medical"],
    ),
    (
        "発達障害のある生徒への服薬管理について，学校と医療機関の連携方法を教えてください．",
        ["education", "medical"],
    ),
    (
        "給食のアレルギー事故が発生しました．再発防止策と当日の医学的対応の両方を検証したいです．",
        ["education", "medical"],
    ),
    (
        "校内で発生した器物損壊について，生徒への指導と保護者への損害賠償請求の両方を検討しています．",
        ["education", "legal"],
    ),
    (
        "学校事故で生徒がケガをした場合の学校の法的責任と，学校側の説明責任について知りたいです．",
        ["education", "legal"],
    ),
    (
        "私立学校の退学処分に対して，処分の妥当性と法的な異議申立て手続きを知りたいです．",
        ["education", "legal"],
    ),
    # 2026-07-30 追加分（d0003 X4／research_frontier 項目2: 20問→100問への拡充）．
    # 元の20問が medical/legal/education の組み合わせに偏っていたため（d0003 X4指摘），
    # 10ドメイン全体に広がる40種類の組み合わせを2問ずつ追加し，多様性を確保する．
    (
        "取引先が突然倒産し，売掛金の回収方法と今後の取引先選定の両方について相談したいです．",
        ["business_economics", "legal"],
    ),
    (
        "フランチャイズ契約を結ぶ予定ですが，契約書の法的リスクと収益計画の妥当性を確認したいです．",
        ["business_economics", "legal"],
    ),
    (
        "自社のECサイトの決済システムを刷新したいのですが，コスト試算とセキュリティ要件の両方を検討する必要があります．",
        ["business_economics", "computer_science"],
    ),
    (
        "会計システムをクラウド移行する際の投資回収期間と，データ移行時のシステム設計の両方を相談したいです．",
        ["business_economics", "computer_science"],
    ),
    (
        "従業員の休職者が増えており，人件費への影響と職場の健康管理体制の見直しを同時に検討しています．",
        ["business_economics", "medical"],
    ),
    (
        "健康食品の製造販売を始めたいのですが，事業計画と成分の安全性評価の両方を確認したいです．",
        ["business_economics", "medical"],
    ),
    (
        "再生可能エネルギー事業への投資を検討していますが，発電効率の技術的な見積もりと事業採算性の両方を知りたいです．",
        ["business_economics", "natural_science"],
    ),
    (
        "食品添加物メーカーとの取引を始めるにあたり，化学的な安全性データと契約条件の妥当性を確認したいです．",
        ["business_economics", "natural_science"],
    ),
    (
        "新規事業の需要予測モデルを作りたいのですが，統計的な手法の選び方と投資判断の基準の両方を相談したいです．",
        ["business_economics", "mathematics"],
    ),
    (
        "ローンの借り換えを検討していますが，金利計算の方法と家計への影響を同時に確認したいです．",
        ["business_economics", "mathematics"],
    ),
    (
        "海外進出先の商慣習の歴史的背景と，現地での事業計画の立て方を知りたいです．",
        ["business_economics", "history_culture"],
    ),
    (
        "伝統工芸品の販路拡大を検討していますが，文化的な価値の伝え方とビジネスモデルの両方について相談したいです．",
        ["business_economics", "history_culture"],
    ),
    (
        "地域の人口減少が事業の将来に与える影響と，社会構造の変化を踏まえた事業戦略を知りたいです．",
        ["business_economics", "social_science"],
    ),
    (
        "働き方改革に伴う社内制度の見直しと，従業員満足度への社会心理学的な影響を検討しています．",
        ["business_economics", "social_science"],
    ),
    (
        "社員研修プログラムの費用対効果と，効果的な教育方法の設計の両方を検討しています．",
        ["business_economics", "education"],
    ),
    (
        "学習塾の新規開校を計画していますが，収支計画とカリキュラム設計の両方について相談したいです．",
        ["business_economics", "education"],
    ),
    (
        "個人事業を始めるにあたり，何から手を付けてよいか全体像と，最低限必要な資金計画を知りたいです．",
        ["business_economics", "general"],
    ),
    (
        "副業を始めたいのですが，一般的な注意点と収益化の見込みについて相談したいです．",
        ["business_economics", "general"],
    ),
    (
        "自社アプリの利用者データが漏洩した可能性があり，技術的な原因調査と法的な報告義務の両方に対応する必要があります．",
        ["computer_science", "legal"],
    ),
    (
        "生成AIを使った新サービスを開発していますが，著作権リスクとシステム設計の両方を確認したいです．",
        ["computer_science", "legal"],
    ),
    (
        "遠隔診療システムを導入したいのですが，通信の安全性と医療機器としての運用要件の両方を知りたいです．",
        ["computer_science", "medical"],
    ),
    (
        "ウェアラブル端末の心拍データを解析するアプリを開発中ですが，アルゴリズムの精度と医学的な妥当性を確認したいです．",
        ["computer_science", "medical"],
    ),
    (
        "気象データを使った予測システムを開発していますが，機械学習モデルの設計と気象現象の物理的な妥当性の両方を確認したいです．",
        ["computer_science", "natural_science"],
    ),
    (
        "遺伝子解析ソフトウェアの高速化を検討していますが，計算アルゴリズムと生物学的な解析手法の両方を知りたいです．",
        ["computer_science", "natural_science"],
    ),
    (
        "暗号通信の実装を検討していますが，数論的な安全性の根拠とソフトウェア実装の両方を確認したいです．",
        ["computer_science", "mathematics"],
    ),
    (
        "機械学習モデルの精度評価に使う統計手法と，実装上の計算コストの両方を相談したいです．",
        ["computer_science", "mathematics"],
    ),
    (
        "郷土資料のデジタルアーカイブ化を進めていますが，データベース設計と資料の歴史的な分類方法の両方を知りたいです．",
        ["computer_science", "history_culture"],
    ),
    (
        "伝統芸能の記録映像をAIで自動タグ付けしたいのですが，技術的な実現方法と文化的な分類基準の両方を相談したいです．",
        ["computer_science", "history_culture"],
    ),
    (
        "SNS上の誹謗中傷を検知するシステムを作りたいのですが，自然言語処理の技術と社会的な許容基準の両方を知りたいです．",
        ["computer_science", "social_science"],
    ),
    (
        "地域コミュニティ向けのアプリを開発していますが，システム設計と住民の利用行動の傾向の両方を相談したいです．",
        ["computer_science", "social_science"],
    ),
    (
        "オンライン学習プラットフォームを開発していますが，システムの拡張性と学習効果を高める教材設計の両方を検討しています．",
        ["computer_science", "education"],
    ),
    (
        "プログラミング教育用の教材を作りたいのですが，教育カリキュラムの設計とコードの難易度設定の両方を相談したいです．",
        ["computer_science", "education"],
    ),
    (
        "パソコンの動作が遅く困っていますが，原因の切り分け方と日常的な使い方の改善点を知りたいです．",
        ["computer_science", "general"],
    ),
    (
        "初めてクラウドサービスを契約するのですが，基本的な使い方と選び方のポイントを教えてください．",
        ["computer_science", "general"],
    ),
    (
        "実験動物を扱う研究を始めるにあたり，動物福祉に関する法規制と適切な実験計画の両方を確認したいです．",
        ["natural_science", "legal"],
    ),
    (
        "化学物質を扱う工場の排水基準について，法的な規制値と実際の処理技術の両方を知りたいです．",
        ["natural_science", "legal"],
    ),
    (
        "新しい治療薬の候補化合物について，化学的な性質と臨床応用の可能性の両方を知りたいです．",
        ["natural_science", "medical"],
    ),
    (
        "放射線治療の被ばく線量について，物理的な計算方法と人体への影響評価の両方を相談したいです．",
        ["natural_science", "medical"],
    ),
    (
        "地震の発生確率をモデル化したいのですが，統計的手法と地球科学的な背景の両方を知りたいです．",
        ["natural_science", "mathematics"],
    ),
    (
        "気候変動シミュレーションの精度を検証したいのですが，数値解析の手法と大気科学的な妥当性の両方を確認したいです．",
        ["natural_science", "mathematics"],
    ),
    (
        "遺跡から出土した遺物の年代測定について，物理学的な手法と考古学的な解釈の両方を知りたいです．",
        ["natural_science", "history_culture"],
    ),
    (
        "気候変動が過去の文明の衰退に与えた影響について，科学的なデータと歴史的な記録の両方を調べたいです．",
        ["natural_science", "history_culture"],
    ),
    (
        "気候変動対策への住民意識について，科学的なリスク評価と社会心理学的な要因の両方を知りたいです．",
        ["natural_science", "social_science"],
    ),
    (
        "感染症の流行モデルと，人々の行動変容を促す社会的な仕組みの両方を検討しています．",
        ["natural_science", "social_science"],
    ),
    (
        "理科の実験授業を安全に行うための注意点と，効果的な指導方法の両方を知りたいです．",
        ["natural_science", "education"],
    ),
    (
        "天体観測を使った探究学習を企画していますが，観測手法と授業設計の両方を相談したいです．",
        ["natural_science", "education"],
    ),
    (
        "家庭菜園で野菜がうまく育たず，土壌の性質と基本的な育て方のコツを知りたいです．",
        ["natural_science", "general"],
    ),
    (
        "身近な自然現象について子供に説明したいのですが，分かりやすい伝え方を教えてください．",
        ["natural_science", "general"],
    ),
    (
        "遺産分割の際の相続割合の計算方法と，法的に有効な分割協議の進め方を知りたいです．",
        ["mathematics", "legal"],
    ),
    (
        "保険金の算定方法と，契約上の支払い条件の解釈の両方を確認したいです．",
        ["mathematics", "legal"],
    ),
    (
        "臨床試験の結果を評価するための統計的な有意差の考え方と，治療効果の医学的な解釈の両方を知りたいです．",
        ["mathematics", "medical"],
    ),
    (
        "健康診断の数値の経時変化をどう分析すればよいか，統計的な見方と医学的な意味の両方を相談したいです．",
        ["mathematics", "medical"],
    ),
    (
        "古文書に記された暦の日付を現在の暦に変換する計算方法と，その時代の暦の歴史的背景を知りたいです．",
        ["mathematics", "history_culture"],
    ),
    (
        "人口統計から見る歴史的な人口変動の傾向について，統計的な手法と歴史的解釈の両方を知りたいです．",
        ["mathematics", "history_culture"],
    ),
    (
        "アンケート調査の結果を分析したいのですが，統計的な手法と社会調査としての妥当性の両方を確認したいです．",
        ["mathematics", "social_science"],
    ),
    (
        "選挙の議席配分の計算方法と，その仕組みが社会に与える影響を知りたいです．",
        ["mathematics", "social_science"],
    ),
    (
        "子供が算数でつまずいているのですが，どこでつまずいているかの分析方法と教え方のコツを知りたいです．",
        ["mathematics", "education"],
    ),
    (
        "テストの採点結果の統計的な分析方法と，それを踏まえた指導改善の進め方を相談したいです．",
        ["mathematics", "education"],
    ),
    (
        "住宅ローンの月々の返済額の計算方法を，基本的な考え方から教えてください．",
        ["mathematics", "general"],
    ),
    (
        "家計の支出を分析したいのですが，基本的な集計方法を知りたいです．",
        ["mathematics", "general"],
    ),
    (
        "文化財の保存活用について，歴史的価値の評価方法と関連する法規制の両方を知りたいです．",
        ["history_culture", "legal"],
    ),
    (
        "伝統的な祭礼の運営を巡るトラブルについて，慣習的な背景と法的な解決方法の両方を相談したいです．",
        ["history_culture", "legal"],
    ),
    (
        "感染症の歴史的な流行の記録と，現代の医学的な知見との関連を知りたいです．",
        ["history_culture", "medical"],
    ),
    (
        "伝統医療の歴史的な位置づけと，現代医学から見た有効性の評価の両方を知りたいです．",
        ["history_culture", "medical"],
    ),
    (
        "地域の伝統行事が衰退している背景について，歴史的な経緯と現代の社会構造の変化の両方を知りたいです．",
        ["history_culture", "social_science"],
    ),
    (
        "移民の歴史的な流入とその地域社会への影響について，歴史的事実と社会学的な分析の両方を知りたいです．",
        ["history_culture", "social_science"],
    ),
    (
        "地域の歴史を題材にした授業を企画していますが，史実の正確な調べ方と授業設計の両方を相談したいです．",
        ["history_culture", "education"],
    ),
    (
        "郷土史の教材を作りたいのですが，資料の選び方と子供向けの分かりやすい構成の両方を知りたいです．",
        ["history_culture", "education"],
    ),
    (
        "旅行先の歴史的な背景を簡単に知りたいのですが，どこから調べればよいか教えてください．",
        ["history_culture", "general"],
    ),
    (
        "家系図を作りたいのですが，基本的な調べ方の手順を知りたいです．",
        ["history_culture", "general"],
    ),
    (
        "地域の空き家問題について，社会的な背景と所有者への法的な対応方法の両方を知りたいです．",
        ["social_science", "legal"],
    ),
    (
        "労働組合の活動について，社会的な意義と法的に認められる権利の範囲を知りたいです．",
        ["social_science", "legal"],
    ),
    (
        "高齢化が進む地域の孤立死の問題について，社会的な要因と医療・介護の連携体制の両方を知りたいです．",
        ["social_science", "medical"],
    ),
    (
        "貧困層の健康格差について，社会的な要因と医学的な対策の両方を知りたいです．",
        ["social_science", "medical"],
    ),
    (
        "不登校の生徒が増えている背景について，社会的な要因と学校の対応方法の両方を知りたいです．",
        ["social_science", "education"],
    ),
    (
        "地域格差が教育機会に与える影響と，学校現場での具体的な対応策を知りたいです．",
        ["social_science", "education"],
    ),
    (
        "最近の少子化のニュースについて，基本的な背景を分かりやすく知りたいです．",
        ["social_science", "general"],
    ),
    (
        "地域のコミュニティ活動に参加したいのですが，一般的な始め方を教えてください．",
        ["social_science", "general"],
    ),
    (
        "近所とのちょっとした境界線トラブルについて，一般的な対処法と法的な手続きの両方を知りたいです．",
        ["general", "legal"],
    ),
    (
        "フリマアプリでの取引トラブルについて，一般的な注意点と法的な対応方法を知りたいです．",
        ["general", "legal"],
    ),
]


def _load_jmmlu_zip_bytes(jmmlu_zip_path: str | None) -> bytes:
    """Return the JMMLU.zip contents, from a local path if given or by download."""
    if jmmlu_zip_path is not None:
        with open(jmmlu_zip_path, "rb") as f:
            return f.read()
    response = httpx.get(_JMMLU_ZIP_URL, timeout=_JMMLU_DOWNLOAD_TIMEOUT_S, follow_redirects=True)
    response.raise_for_status()
    return response.content


def _parse_jmmlu_task_csv(zf: zipfile.ZipFile, task_name: str) -> list[dict[str, str]]:
    """Parse one JMMLU task's CSV into rows of {question, A, B, C, D, answer}."""
    raw_bytes = zf.read(_JMMLU_CSV_PATH_TEMPLATE.format(task_name=task_name))
    text = raw_bytes.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    return [{key: value.strip() for key, value in row.items()} for row in reader]


def _format_jmmlu_query(row: dict[str, str]) -> str:
    """Format a JMMLU row as a four-choice question prompt."""
    return f"{row['question']}\nA. {row['A']}\nB. {row['B']}\nC. {row['C']}\nD. {row['D']}"


def _sample_domain_questions(
    zf: zipfile.ZipFile,
    task_names: list[str],
    target_size: int,
    seed: int,
    exclude_tasks: frozenset[str],
    exclude_queries: frozenset[str] = frozenset(),
    task_target_sizes: dict[str, int] | None = None,
) -> list[tuple[str, str, str]]:
    """Sample up to target_size (query, answer, task_name) tuples for one domain's tasks.

    Pools all constituent tasks' rows together before sampling, so the
    domain's questions are not required to be evenly split across tasks.
    Caps at the pool size rather than raising when a domain's tasks
    together hold fewer than target_size questions (true for legal).
    exclude_queries removes specific questions from the pool before
    sampling (used by build_classifier_training_rows to guarantee its
    output never overlaps the evaluation dataset's questions).

    When task_target_sizes is provided, each task is sampled independently
    from its own pool using the task-specific target size (capped at pool
    size). This allows per-task control of representation (e.g., Iter33's
    education proxy task resampling). Uses a single random.Random(seed)
    instance, calling rng.sample() in task_names order for deterministic
    reproducibility.
    """
    rng = random.Random(seed)

    if task_target_sizes is not None:
        assert set(task_names) <= set(task_target_sizes), (
            f"task_target_sizes must cover all task_names: "
            f"{set(task_names) - set(task_target_sizes)} missing"
        )
        result: list[tuple[str, str, str]] = []
        for task_name in task_names:
            if task_name in exclude_tasks:
                continue
            task_pool: list[tuple[str, str, str]] = []
            for row in _parse_jmmlu_task_csv(zf, task_name):
                query = _format_jmmlu_query(row)
                if query in exclude_queries:
                    continue
                task_pool.append((query, row["answer"], task_name))
            task_target = task_target_sizes.get(task_name, target_size)
            sample_size = min(task_target, len(task_pool))
            result.extend(rng.sample(task_pool, sample_size))
        return result

    pool: list[tuple[str, str, str]] = []
    for task_name in task_names:
        if task_name in exclude_tasks:
            continue
        for row in _parse_jmmlu_task_csv(zf, task_name):
            query = _format_jmmlu_query(row)
            if query in exclude_queries:
                continue
            pool.append((query, row["answer"], task_name))
    sample_size = min(target_size, len(pool))
    return rng.sample(pool, sample_size)


def _build_jmmlu_backed_groups(
    zf: zipfile.ZipFile,
    domain_target_size: int,
    exclude_restricted: bool,
    domain_task_map: dict[str, list[str]],
    seed: int = _JMMLU_SAMPLE_SEED,
    exclude_queries: frozenset[str] = frozenset(),
) -> dict[str, list[tuple[str, str, str]]]:
    """Sample every domain's questions from its mapped JMMLU tasks."""
    exclude_tasks = _RESTRICTED_LICENSE_TASKS if exclude_restricted else frozenset()
    return {
        domain: _sample_domain_questions(
            zf, task_names, domain_target_size, seed, exclude_tasks, exclude_queries
        )
        for domain, task_names in domain_task_map.items()
    }


def _build_rows(
    jmmlu_zip_path: str | None,
    domain_target_size: int,
    exclude_restricted_license_tasks: bool,
    domain_task_map: dict[str, list[str]],
) -> list[dict]:
    """Assemble JMMLU-derived single-domain rows and hand-authored compound rows."""
    zip_bytes = _load_jmmlu_zip_bytes(jmmlu_zip_path)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        domain_groups = _build_jmmlu_backed_groups(
            zf, domain_target_size, exclude_restricted_license_tasks, domain_task_map
        )

    rows = []
    for domain in sorted(domain_groups):
        for index, (query, answer, task_name) in enumerate(domain_groups[domain], start=1):
            rows.append(
                {
                    "id": f"{domain}-{index:03d}",
                    "query": query,
                    "expected_domains": [domain],
                    "is_compound": False,
                    "jmmlu_task": task_name,
                    "jmmlu_answer": answer,
                }
            )
    for index, (query, expected_domains) in enumerate(_COMPOUND_QUESTIONS, start=1):
        rows.append(
            {
                "id": f"compound-{index:03d}",
                "query": query,
                "expected_domains": expected_domains,
                "is_compound": True,
            }
        )
    return rows


def write_dataset(
    output: TextIO,
    jmmlu_zip_path: str | None = None,
    domain_target_size: int = _DOMAIN_TARGET_SIZE,
    exclude_restricted_license_tasks: bool = False,
    domain_task_map: dict[str, list[str]] | None = None,
) -> int:
    """Write all dataset rows as JSON Lines to the given stream; return the row count.

    domain_task_map overrides the module-level _DOMAIN_TASK_MAP; tests use
    this to point at a fixture zip containing only one task per domain
    instead of all 56 real JMMLU tasks.
    """
    rows = _build_rows(
        jmmlu_zip_path,
        domain_target_size,
        exclude_restricted_license_tasks,
        domain_task_map if domain_task_map is not None else _DOMAIN_TASK_MAP,
    )
    for row in rows:
        output.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def build_classifier_training_rows(
    jmmlu_zip_path: str | None,
    domain_target_size: int,
    exclude_restricted_license_tasks: bool,
    domain_task_map: dict[str, list[str]],
    eval_rows: list[dict],
) -> list[dict]:
    """Build E6 classifier training rows ({id, query, domain, sample_weight}), disjoint from eval_rows' questions.

    Guards against Iter10's label leakage (the training features there were
    derived from probe/dispatch results on the same 46 questions used for
    evaluation): eval_rows' single-domain questions are excluded from the
    sampling pool *before* sampling (not just tagged afterward), so overlap
    with the evaluation set is structurally impossible rather than merely
    avoided by convention. Uses _CLASSIFIER_TRAIN_SAMPLE_SEED (distinct from
    the eval set's seed) so the two samples are independent draws.

    Passing an explicit --jmmlu-zip (a locally cached JMMLU.zip) avoids
    downloading it a second time when both this and the eval dataset are
    generated in the same run.

    Known imbalance: since eval and training draw from the same
    task-limited pool without overlap, a domain whose pool is close to
    2x domain_target_size ends up with a noticeably smaller training set
    than the rest. At the default domain_target_size=150, legal's pool is
    227 (verified against the real JMMLU.zip): after 150 are reserved for
    eval, only 77 remain for training, versus 150 for every other domain.
    scripts/train_domain_classifier.py does not currently compensate for
    this (e.g. via class_weight), so the classifier may underperform on
    legal specifically for reasons unrelated to the signal itself.

    Each row also carries a per-task sample_weight (Iter32, see
    _classifier_task_sample_weight): rows drawn from a task listed in
    _CLASSIFIER_TASK_SAMPLE_WEIGHTS get that weight, all others default to
    1.0, so pre-Iter32 behavior (uniform weighting) is unchanged unless a
    task is explicitly listed.

    Iter33 education override: the `education` domain is sampled separately
    from other domains, using task-specific target sizes defined by
    _EDUCATION_PROXY_TASK_TRAIN_TARGET_SIZES (sociology=70,
    high_school_psychology=40, moral_disputes=40). This avoids the
    `sample_weight` mechanism that was rejected in Iter32 due to its
    interaction with `class_weight="balanced"`. All other domains continue
    to use the standard pooled sampling via _build_jmmlu_backed_groups().
    """
    eval_queries = frozenset(row["query"] for row in eval_rows if not row["is_compound"])
    zip_bytes = _load_jmmlu_zip_bytes(jmmlu_zip_path)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        domain_task_map_without_education = {
            domain: tasks
            for domain, tasks in domain_task_map.items()
            if domain != "education"
        }
        domain_groups = _build_jmmlu_backed_groups(
            zf,
            domain_target_size,
            exclude_restricted_license_tasks,
            domain_task_map_without_education,
            seed=_CLASSIFIER_TRAIN_SAMPLE_SEED,
            exclude_queries=eval_queries,
        )
        exclude_tasks = (
            _RESTRICTED_LICENSE_TASKS
            if exclude_restricted_license_tasks
            else frozenset()
        )
        domain_groups["education"] = _sample_domain_questions(
            zf,
            domain_task_map["education"],
            domain_target_size,
            _CLASSIFIER_TRAIN_SAMPLE_SEED,
            exclude_tasks,
            exclude_queries=eval_queries,
            task_target_sizes=_EDUCATION_PROXY_TASK_TRAIN_TARGET_SIZES,
        )

    rows = []
    for domain in sorted(domain_groups):
        for index, (query, _answer, task_name) in enumerate(domain_groups[domain], start=1):
            rows.append(
                {
                    "id": f"{domain}-train-{index:03d}",
                    "query": query,
                    "domain": domain,
                    "sample_weight": _classifier_task_sample_weight(task_name),
                }
            )
    return rows


def write_classifier_training_data(
    output: TextIO,
    jmmlu_zip_path: str | None,
    domain_target_size: int,
    exclude_restricted_license_tasks: bool,
    domain_task_map: dict[str, list[str]] | None,
    eval_rows: list[dict],
) -> int:
    """Write classifier training rows as JSON Lines; return the row count."""
    rows = build_classifier_training_rows(
        jmmlu_zip_path,
        domain_target_size,
        exclude_restricted_license_tasks,
        domain_task_map if domain_task_map is not None else _DOMAIN_TASK_MAP,
        eval_rows,
    )
    for row in rows:
        output.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def _ensure_parent_dir(path: str) -> None:
    """Create the parent directory of path if needed (data/ and results/ are gitignored,
    so a clean checkout has neither until something creates them)."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Build the JMMLU-backed, 10-domain evaluation dataset as JSONL"
    )
    parser.add_argument("--output", default=None, help="Output file path; defaults to stdout")
    parser.add_argument(
        "--jmmlu-zip",
        default=None,
        help="Local path to a JMMLU.zip (skips downloading); mainly for tests/offline use",
    )
    parser.add_argument(
        "--domain-target-size",
        type=int,
        default=_DOMAIN_TARGET_SIZE,
        help="Max questions sampled per domain (capped at the domain's actual pool size)",
    )
    parser.add_argument(
        "--exclude-restricted-license-tasks",
        action="store_true",
        help="Exclude the 5 JMMLU tasks with named-copyright-holder clauses (all in history_culture)",
    )
    parser.add_argument(
        "--classifier-train-output",
        default=None,
        help="If set, also write E6 classifier training rows (disjoint from --output's questions) here",
    )
    args = parser.parse_args()

    eval_rows = _build_rows(
        args.jmmlu_zip,
        args.domain_target_size,
        args.exclude_restricted_license_tasks,
        _DOMAIN_TASK_MAP,
    )
    if args.output is None:
        for row in eval_rows:
            sys.stdout.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        _ensure_parent_dir(args.output)
        with open(args.output, "w", encoding="utf-8") as f:
            for row in eval_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[build_dataset] wrote {len(eval_rows)} rows", file=sys.stderr)

    if args.classifier_train_output is not None:
        _ensure_parent_dir(args.classifier_train_output)
        with open(args.classifier_train_output, "w", encoding="utf-8") as f:
            classifier_count = write_classifier_training_data(
                f,
                args.jmmlu_zip,
                args.domain_target_size,
                args.exclude_restricted_license_tasks,
                _DOMAIN_TASK_MAP,
                eval_rows,
            )
        print(f"[build_dataset] wrote {classifier_count} classifier training rows", file=sys.stderr)


if __name__ == "__main__":
    main()
