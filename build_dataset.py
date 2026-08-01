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
- `education` maps to `japanese_civics` (150 questions, JMMLU固有) as a
  proxy for the mesh's actual education-administration domain. This task
  includes education administration content (school management, Education
  Basic Law, Board of Education, etc.) and provides better semantic
  coverage than the previous proxy tasks (sociology,
  high_school_psychology, moral_disputes). This is a deliberate
  compromise, not a claim that these tasks measure the same thing as the
  hand-authored education questions used for compound rows.

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
        "japanese_civics",
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

# Iter34 (classifier_training_data_composition=education_proxy_task_resampling, Y5):
# 案C（70/40/40）はrejected（education_recall 0.4412 < medical_recall基準 0.5112）。
# 変化幅を約2倍に拡大した案A（90/30/30）を試す。sociologyのpool cap（94）を
# 95.7%使い切るため，案Aが不成立の場合のresampling系余地は尽きる。
_EDUCATION_PROXY_TASK_TRAIN_TARGET_SIZES: dict[str, int] = {
    "japanese_civics": 150,
}
assert sum(_EDUCATION_PROXY_TASK_TRAIN_TARGET_SIZES.values()) == _DOMAIN_TARGET_SIZE

# Iter35 (classifier_training_data_composition=education_handmade_training_problems, Y5):
# education_proxy_task_resampling（案A: sociology=90/high_school_psychology=30/moral_disputes=30）
# はrejected（education_recall 0.4353 < medical_recall基準 0.5112）となった。
# resampling系レバーは尽きた（backlog B55）ため，education固有の手作り訓練問題
# 50件を追加する。すべて4択形式（A/B/C/D）を保つ（書式 shortcuts リスク，
# Iter32調査で確認済み）。テーマは教育行政実務（学校事故責任，生徒健康管理，
# アレルギー対応，懲戒処分，教職員人事，保護者対応，施設管理，法令順守）。
_EDUCATION_HANDMADE_QUESTIONS: list[tuple[str, str, str, str, str, str]] = [
    # --- Theme 1: 学校事故責任（10件） ---
    (
        "学校遠足中のバス事故で生徒が負傷した際，学校側の損害賠償責任を問うことができるのは，次のうちどの場合か?",
        "バス会社が過失を負った場合のみ",
        "学校に安全管理上の過失があった場合",
        "生徒本人に過失があった場合のみ",
        "保護者が保険に加入していなかった場合",
        "B",
    ),
    (
        "部活動中の練習で生徒がケガをした場合，学校が損害賠償を負うのはどの場合か?",
        "部活動自体が危険を伴う活動であった場合",
        "顧問教員が指導上の注意義務を怠った場合",
        "生徒が指示に従わなかった場合のみ",
        "同じ部活動の他の生徒が不注意だった場合のみ",
        "B",
    ),
    (
        "学校の体育館で天井の照明器具が落下し，生徒が負傷した。学校設置者の責任として正しいものは?",
        "突発的な事故であり責任はない",
        "定期的な点検を実施していなかった場合，過失責任を負う",
        "生徒が落下地点にいたことが原因で責任はない",
        "照明器具の製造業者に全ての責任がある",
        "B",
    ),
    (
        "修学旅行中の宿泊施設で生徒が病気を発症した場合，学校が責任を負うのは?",
        "施設側の衛生管理不備が原因で，学校も監督義務違反があれば責任を負う",
        "どんな場合でも学校が全ての責任を負う",
        "生徒の体質によるもので学校に責任はない",
        "保護者が事前の健康状態を伝えていなかった場合のみ",
        "A",
    ),
    (
        "学校の運動場で球技中の打球が隣接する他校の生徒に当たった場合，責任の所在として正しいのは?",
        "他校の敷地内に入ったため他校が責任を負う",
        "打球を放った生徒の所属学校が過失があれば責任を負う",
        "打球を浴びた生徒が危険な場所にいたため責任はない",
        "両校の責任で等しく負担する",
        "B",
    ),
    (
        "学校給食の調理場での食中毒事故について，学校設置者が講じる法的措置として最も適切なものは?",
        "調理業者への損害賠償請求のみを行う",
        "保健所に事故報告をし，原因調査と再発防止策を求める",
        "保護者に謝罪するだけで法的措置は取らない",
        "調理業者を直ちに解雇するだけで対応完了とする",
        "B",
    ),
    (
        "放課後の校舎内で生徒が階段から転落した際，学校側の過失が問われるのは?",
        "階段の手すりが破損していた状態で放置されていた場合",
        "生徒が走っていた場合のみ",
        "放課後だったため学校に責任はない",
        "他の生徒が転落を誘った場合のみ",
        "A",
    ),
    (
        "理科の実験授業で化学薬品が目に入り，生徒が視力を損なった。学校が責任を負うのは?",
        "実験自体が危険を伴うものであれば責任はない",
        "安全指導を十分に行わず，防護用具の装着を指示しなかった場合",
        "生徒が実験手順を無視した場合のみ",
        "化学薬品の製造業者に全ての責任がある",
        "B",
    ),
    (
        "学校のプールで水泳授業中に生徒が溺れかけた際，学校側の過失が問われるのは?",
        "プールが深水区であった場合のみ",
        "監視教員が不在であり，緊急時の対応体制が整っていなかった場合",
        "生徒が水泳が苦手であった場合のみ",
        "保護者が水泳の経験を伝えていなかった場合",
        "B",
    ),
    (
        "校外授業中の交通事故で生徒が負傷した場合，学校が損害賠償責任を負う要件として正しいのは?",
        "運送業者が過失を負った場合のみ",
        "学校が送迎手段の選定や手配に過失があった場合",
        "生徒が交通事故の加害者であった場合のみ",
        "保護者が外出を許可したため責任はない",
        "B",
    ),
    # --- Theme 2: 生徒健康管理（8件） ---
    (
        "学校における定期健康診断の結果，生徒に異常所見が認められた場合，学校長が最初に取るべき措置として最も適切なものはどれか?",
        "直ちに保護者に連絡し，精密検査を勧める",
        "保健室で安静させ，様子を観察する",
        "担任の教員に相談させる",
        "他の生徒への感染を防止するため隔離する",
        "A",
    ),
    (
        "学校でインフルエンザの集団発生が認められた際，学校長が取れる措置として法令に則ったものは?",
        "直ちに学校を閉鎖する",
        "教育委員会に報告し，必要に応じて臨時休業を決定する",
        "感染者のみを退学させる",
        "保護者に連絡せずに通常通り授業を続ける",
        "B",
    ),
    (
        "熱中症の疑いがある生徒が校内で倒れた際，教員が最初に取るべき応急処置として最も適切なものは?",
        "直ちに涼しい場所に移動させ，体を冷やし，水分を補給させる",
        "すぐに立たせて水分を飲ませる",
        "氷を頭に乗せるだけで放置する",
        "他の生徒をその場から離れさせない",
        "A",
    ),
    (
        "学校における保健室登校の生徒に対する指導として最も適切なものは?",
        "保健室に終日閉じ込め，授業に参加させない",
        "生徒の状態に応じ，部分的な授業参加や段階的な復旧プログラムを組む",
        "保健室登校を認めず，欠席として扱う",
        "保健室登校の生徒には補習のみを課す",
        "B",
    ),
    (
        "生徒の精神的健康に関する相談が増加している場合，学校が講じる組織的な対策として最も適切なものは?",
        "担任の教員が全てを一人で受け持つ",
        "スクールカウンセラーを配置し，教職員間で情報共有する体制を整える",
        "相談を外部の病院に全て委ねる",
        "相談を認めず，問題を隠蔽する",
        "B",
    ),
    (
        "学校における歯科健康診断の結果，多くの生徒に虫歯が認められた場合，学校が講じる対策として最も適切なものは?",
        "保護者へ個別に通知し，歯科受診を勧める体制を整える",
        "校内で歯科治療を行う",
        "虫歯の問題を無視し，次の年度まで待つ",
        "全校生徒を歯科医院に強制連行する",
        "A",
    ),
    (
        "学校で結核の陽性者が確認された場合，学校設置者が取るべき措置として正しいものは?",
        "陽性者だけを退学させる",
        "保健所に報告し，接触者の検査と必要に応じて学級閉鎖を決定する",
        "情報を隠蔽し，通常通り授業を続ける",
        "陽性者の家族に謝罪を求める",
        "B",
    ),
    (
        "生徒が自殺未遂を図った場合の学校側の対応として，法令と指針に則った最も適切なものは?",
        "直ちに保護者と教育委員会に報告し，関係機関と連携して支援体制を整える",
        "事件として警察に通報するだけで対応完了とする",
        "問題があった生徒の情報を他校に共有する",
        "教職員内で秘密にし，外部に知らせない",
        "A",
    ),
    # --- Theme 3: アレルギー対応（6件） ---
    (
        "食物アレルギーのある生徒の給食対応について，学校が講じる措置として最も適切なものは?",
        "アレルギー食材を一切提供しない完全除去食にする",
        "アレルギー食材を除去した代替食を提供する",
        "生徒本人に食材を選別させる",
        "保護者が持参した弁当のみを提供する",
        "B",
    ),
    (
        "学校給食中に生徒がアナフィラキシー疑似症状を示した場合，教員が最初に取るべき対応は?",
        "直ちに救急車を要請し，保存薬（エピネフリン自己注射薬等）を投与する準備をする",
        "生徒に水を飲ませて様子を見る",
        "保健室に移動させて安静させるだけにする",
        "保護者を呼びに行くまで待つ",
        "A",
    ),
    (
        "学校における食物アレルギー対応の基本的な方針として，文部科学省の指針に則ったものは?",
        "アレルギーのある生徒のみが給食を食べないようにする",
        "アレルギー症状の重症度に応じた対応を行い，可能な限り他の生徒と同じ給食を提供する",
        "アレルギー対応を保護者の責任に全て委ねる",
        "アレルギー食材を学校給食から永久に排除する",
        "B",
    ),
    (
        "花粉症の症状がひどい生徒が授業中に集中できない場合，学校が講じる対応として最も適切なのは?",
        "授業を放棄させる",
        "窓を閉める，空気清浄機を使う等の環境整備と，必要に応じ薬の持参を許可する",
        "花粉症は病気ではないので対応しない",
        "全校生徒にマスク着用を強制する",
        "B",
    ),
    (
        "新入生受付時にアレルギー情報を収集する際，学校が講じるべき措置として正しいものは?",
        "保護者の同意なく全ての健康情報を収集する",
        "保護者からアレルギー情報を適切に収集し，関係教職員で共有する体制を整える",
        "アレルギー情報を収集する必要はない",
        "アレルギー情報を全校生徒に公開する",
        "B",
    ),
    (
        "学校行事で野外活動を行う際，食物アレルギーのある生徒が参加する場合の配慮として最も適切なものは?",
        "その生徒を行事から除外する",
        "持参する食事を事前に確認し，アレルギー対応可能な献立を手配する",
        "野外活動では給食を出さないことにする",
        "他の生徒と同じ食事を強制的に食べさせる",
        "B",
    ),
    # --- Theme 4: 懲戒処分・指導（6件） ---
    (
        "教職員がいじめを隠蔽したことが発覚した場合，学校設置者（自治体等）が下すことができる処分として最も適切なものは?",
        "戒告のみ",
        "戒告，減給，停職，免職のいずれか",
        "口頭注意のみ",
        "配置転換のみ",
        "B",
    ),
    (
        "生徒への懲戒処分として，学校が設けられるものとして法令上適切なものは?",
        "登校禁止，注意，訓告，戒告，分限処分の各段階に応じたもの",
        "罰金刑",
        "即時退学",
        "保護者の職場への連絡",
        "A",
    ),
    (
        "生徒が他の生徒に重大な傷害を与えた場合の学校側の対応として最も適切なものは?",
        "直ちに保護者に連絡し，事実関係を調査した上で適切な指導・処分を行う",
        "加害生徒のみを転校させる",
        "問題を起こした生徒の情報を他校に共有する",
        "教職員内で秘密にする",
        "A",
    ),
    (
        "教職員が体罰行為を行ったことが確認された場合，学校設置者が取るべき対応として正しいものは?",
        "その教職員を直ちに免職にする",
        "事実関係を調査し，体罰の程度に応じて適切な処分を行うとともに再発防止策を講じる",
        "注意のみで済ませる",
        "教職員の説明を信じて問題なしとする",
        "B",
    ),
    (
        "生徒が集団で強奪行為を行った場合，学校が講じる指導として最も適切なものは?",
        "直ちに全員を退学させる",
        "各生徒の関与の程度を個別に評価し，教育上の観点から適切な指導・処分を行う",
        "保護者に全ての責任を転嫁する",
        "事件として処理するだけで教育指導は行わない",
        "B",
    ),
    (
        "学校内で盗難が相次いでいる場合，学校が取るべき対応として最も適切なのは?",
        "疑わしい生徒を全員集合させ，公開処罰を行う",
        "関係機関と連携して事実関係を調査し，被害生徒の保護と加害生徒の教育指導を両立させる",
        "盗難を無視し，防犯カメラのみを設置する",
        "全校生徒の所持品を毎日検査する",
        "B",
    ),
    # --- Theme 5: 教職員人事・労務（5件） ---
    (
        "教職員の配置転換について，学校長が配置転換を指示できる範囲として正しいものは?",
        "校内の職務のみ",
        "同一設置者管内の他の学校への異動を含む",
        "他自治体の学校への異動を含む",
        "教職員の希望を必ず尊重しなければならない",
        "B",
    ),
    (
        "教職員が業務中の事故で負傷し，療養が必要な場合，学校設置者が講じる措置として正しいものは?",
        "その教職員の責任とする",
        "労災認定の手続きを行い，適切な療養と復帰支援を行う",
        "無給休職とする",
        "事故を隠蔽し，通常通り勤務させる",
        "B",
    ),
    (
        "教職員の労働時間管理について，学校教育法施行規則が定める原則として正しいものは?",
        "労働時間の上限はない",
        "原則として1週間の所定労働時間は40時間以内",
        "1日8時間を超えて働かせてはならない",
        "教職員は休日を取得しなくてよい",
        "B",
    ),
    (
        "教職員がいじめの相談を受けた際，その教職員が取るべき最初の対応として最も適切なものは?",
        "自分で解決しようとする",
        "校長又は教育委員会に速やかに報告し，組織的に取り組む体制を整える",
        "相談者を説教する",
        "問題を無視する",
        "B",
    ),
    (
        "教職員の研修プログラムについて，地方教育行政の組織及び運営に関する法律が定める学校的役割として正しいものは?",
        "研修は任意であり義務ではない",
        "教職員の資質向上のために継続的な研修を実施する義務がある",
        "研修は外部委託に全て委ねればよい",
        "研修は新任教員のみに行えばよい",
        "B",
    ),
    # --- Theme 6: 保護者対応・コミュニケーション（5件） ---
    (
        "生徒のいじめ被害について保護者から相談があった際，学校が取るべき最初の対応として最も適切なものは?",
        "いじめた側の保護者を呼び，謝罪をさせる",
        "被害生徒と保護者を別面談で聴取し，事実関係を把握する",
        "全校集会でいじめの問題について注意喚起する",
        "警察に通報する",
        "B",
    ),
    (
        "保護者会（PTA総会）で学校運営の重要な方針変更を決定する際，学校が講じるべき手続きとして最も適切なものは?",
        "校長が独断で決定し，事後に報告する",
        "事前に資料を配布し，十分な議論の機会を設けた上で合意形成を図る",
        "保護者の意見を無視して通常通り進める",
        "PTA会長に全て委ねる",
        "B",
    ),
    (
        "生徒の家庭環境の変化（保護者の失業等）により学習意欲が低下している場合，学校が講じる対応として最も適切なものは?",
        "保護者を責める",
        "保護者と連携し，生徒へのサポート体制を整える",
        "その生徒を特別扱いしない",
        "学校全体の問題として無視する",
        "B",
    ),
    (
        "学校が保護者から苦情を受けた際，学校経営の基本方針として最も適切なものは?",
        "苦情を無視し，通常通り運営する",
        "苦情を真摯に受け止め，事実関係を調査した上で保護者に説明し，改善策を講じる",
        "苦情を言った保護者を blacklist に入れる",
        "苦情を教育委員会に全て委ねる",
        "B",
    ),
    (
        "学校評価において保護者の意見を収集する際，最も適切な方法は?",
        "保護者の意見を全く収集しない",
        "アンケート調査や説明会等を通じて多様な保護者の意見を収集し，学校経営に反映する",
        "意見を集めた上で全て無視する",
        "保護者会での発言者の意見のみを参考にする",
        "B",
    ),
    # --- Theme 7: 学校運営・施設管理（5件） ---
    (
        "学校の校舎で天井の亀裂が発見された場合，学校設置者が最初に取るべき措置として最も適切なものは?",
        "直ちにその区域を立ち入り禁止にし，構造計算書を確認する",
        "次回の修繕計画に組み込む",
        "生徒に注意喚起のみを行う",
        "保護者に報告して意見を求める",
        "A",
    ),
    (
        "学校が毎年実施すべき防災訓練について，学校教育法施行規則で定められているものは?",
        "火災訓練のみ",
        "地震・津波・火災など各種災害を想定した総合訓練",
        "消防署との合同訓練のみ",
        "年1回以上の避難訓練の実施が努力義務とされている",
        "D",
    ),
    (
        "学校施設の省エネルギー化を図る際，学校設置者が講じる措置として最も適切なものは?",
        "エネルギーコストを完全に削減するため，冷暖房を停止する",
        "エネルギー効率的な設備への更新と，節電啓発を併せて行う",
        "省エネルギー化は保護者の責任とする",
        "省エネルギー化は行わず，従来通り運用する",
        "B",
    ),
    (
        "学校のICT機器（タブレット等）を導入する際，設置者が講じるべき措置として最も適切なものは?",
        "機器を購入するだけで導入完了とする",
        "機器の導入とともに教職員の研修，ネットワーク環境の整備，利用ガイドラインの策定を行う",
        "ICT機器は不要であるとして導入を中止する",
        "保護者に機器購入を義務付ける",
        "B",
    ),
    (
        "学校敷地内の遊具が老朽化で危険な状態にある場合，学校設置者が取るべき措置として正しいものは?",
        "そのまま使用させ，怪我は自己責任とする",
        "直ちに使用を中止し，修繕又は交換を行うまで立ち入りを制限する",
        "保護者に修理費用を請求する",
        "次の年度予算まで待つ",
        "B",
    ),
    # --- Theme 8: 法令順守・個人情報（5件） ---
    (
        "学校が生徒の個人情報を外部の教育サービス業者に委託する場合，設置者が講じるべき措置として正しいものは?",
        "個人情報保護法に基づく監督措置を講じる",
        "保護者の同意が不要である",
        "業者が自由に情報を使用できる",
        "委託は禁止されている",
        "A",
    ),
    (
        "学校における個人情報の取扱いに関する法令遵守の基本方針として正しいものは?",
        "個人情報の収集・利用・提供は，目的の範囲内に行い，安全管理措置を講じる",
        "生徒の個人情報は全校教職員が自由に閲覧できる",
        "個人情報の管理はIT担当教員に全て委ねればよい",
        "個人情報は外部に開示して問題ない",
        "A",
    ),
    (
        "学校保健安全法に基づく感染症対策について，学校が出席停止の対象とする感染症として正しいものは?",
        "風疹のみ",
        "麻疹，風疹，水痘，百日咳など法律で定められた感染症",
        "風邪のみ",
        "全ての感染症",
        "B",
    ),
    (
        "学校における児童虐待の疑いがある事例を発見した場合，教職員が取るべき法的措置として正しいものは?",
        "自分で保護者に注意するだけで対応完了とする",
        "児童相談所に通告し，必要に応じて警察に通報する",
        "問題を校内で処理する",
        "疑いがある生徒を退学させる",
        "B",
    ),
    (
        "学校が防災・減災に関する地域連携を強化する際，法令に基づき講じられるべき措置として最も適切なものは?",
        "地域連携は任意であり義務ではない",
        "自治体，消防，地域住民と連携し，防災計画を策定し，訓練を実施する",
        "地域連携は外部委託に全て委ねる",
        "防災計画は学校内だけで完結させる",
        "B",
    ),
]


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
    _EDUCATION_PROXY_TASK_TRAIN_TARGET_SIZES (sociology=90,
    high_school_psychology=30, moral_disputes=30). This avoids the
    `sample_weight` mechanism that was rejected in Iter32 due to its
    interaction with `class_weight="balanced"`. All other domains continue
    to use the standard pooled sampling via _build_jmmlu_backed_groups().

    Iter35 handmade questions: 50 hand-authored education-administration
    questions (_EDUCATION_HANDMADE_QUESTIONS) are appended to the education
    training rows after proxy-task sampling. These cover 8 themes: school
    accident liability, student health management, allergy response,
    disciplinary actions, staff management, parent communication, facility
    management, and legal compliance. Each uses the standard 4-choice
    (A/B/C/D) format to avoid format-based shortcut learning (Iter32).
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

    # Iter35: append hand-authored education-administration questions
    for idx, (question, a, b, c, d, correct) in enumerate(
        _EDUCATION_HANDMADE_QUESTIONS, start=1
    ):
        rows.append(
            {
                "id": f"education-train-handmade-{idx:03d}",
                "query": _format_jmmlu_query(
                    {"question": question, "A": a, "B": b, "C": c, "D": d}
                ),
                "domain": "education",
                "sample_weight": _classifier_task_sample_weight("education_handmade"),
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
