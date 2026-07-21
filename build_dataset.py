"""Build the tier-2 evaluation dataset (design doc 4.3): community-consultation style
questions with ground-truth domain labels, including compound-domain cases.

This is placeholder sample data, not the final research dataset. Each entry's
`expected_domains` lists every domain that would be an acceptable routing target;
a single-domain question has exactly one entry, while a compound-domain question
(e.g. a workplace injury has both a legal and a medical angle) lists both so that
metrics.py can score "any acceptable domain" as well as strict top-1.

Usage:
    uv run python build_dataset.py > data/dataset.jsonl
    uv run python build_dataset.py --output data/dataset.jsonl
"""

import argparse
import json
import sys
from typing import TextIO

# Each row: (query, expected_domains). expected_domains has one entry for a
# single-domain question, or multiple for a compound-domain question where
# either domain is an acceptable routing target (design doc 4.3 tier 2).
_MEDICAL_QUESTIONS: list[tuple[str, list[str]]] = [
    ("3日前から頭痛と38度の発熱が続いています．何科を受診すべきですか．", ["medical"]),
    ("子どもが誤って洗剤を飲んでしまいました．応急処置を教えてください．", ["medical"]),
    ("血圧の薬を飲み忘れた場合，次の服用はどうすればよいですか．", ["medical"]),
    ("最近ずっと不眠が続いていて，日中の集中力も落ちています．", ["medical"]),
    ("膝の痛みが階段の上り下りで悪化します．湿布以外にできることはありますか．", ["medical"]),
    ("インフルエンザの予防接種は何歳から受けられますか．", ["medical"]),
    ("食物アレルギーの検査を受けたいのですが，どこに相談すればよいですか．", ["medical"]),
    ("糖尿病の食事療法について，具体的な注意点を教えてください．", ["medical"]),
    ("腰痛がひどく，仕事に支障が出ています．整形外科と整骨院どちらが良いですか．", ["medical"]),
    ("高齢の親の物忘れが増えてきました．認知症の初期症状でしょうか．", ["medical"]),
]

_LEGAL_QUESTIONS: list[tuple[str, list[str]]] = [
    ("賃貸マンションを退去する際，敷金が全く返還されませんでした．", ["legal"]),
    ("インターネットで購入した商品が届かず，返金にも応じてもらえません．", ["legal"]),
    ("近隣住民との騒音トラブルで，内容証明を送りたいのですが手順を教えてください．", ["legal"]),
    ("交通事故の過失割合に納得できない場合，どこに相談すればよいですか．", ["legal"]),
    ("遺産相続で，兄弟間の取り分について揉めています．", ["legal"]),
    ("会社を突然解雇されました．不当解雇に該当するか知りたいです．", ["legal"]),
    ("離婚時の親権と養育費について，一般的な取り決め方を教えてください．", ["legal"]),
    ("契約書にサインした後で不利な条項に気づきました．解約できますか．", ["legal"]),
    ("アパートの隣室からの水漏れで家財が被害を受けました．賠償請求は可能ですか．", ["legal"]),
    ("フリーランスとして受けた仕事の報酬が支払われません．", ["legal"]),
]

_GENERAL_QUESTIONS: list[tuple[str, list[str]]] = [
    ("おすすめの映画を教えてください．", ["general"]),
    ("簡単に作れる夕食のレシピを知りたいです．", ["general"]),
    ("引っ越し先の地域でおすすめの公園はありますか．", ["general"]),
    ("読書感想文の書き方のコツを教えてください．", ["general"]),
    ("週末の天気に合わせた服装のアドバイスがほしいです．", ["general"]),
    ("観葉植物の育て方で気をつけることは何ですか．", ["general"]),
    ("初めての一人暮らしで揃えるべき家電を教えてください．", ["general"]),
    ("運動不足を解消する簡単なストレッチ方法はありますか．", ["general"]),
    ("旅行の荷造りを効率よく行うコツを教えてください．", ["general"]),
    ("最近話題のカフェについて教えてください．", ["general"]),
]

_EDUCATION_QUESTIONS: list[tuple[str, list[str]]] = [
    ("学習指導要領における探究的学習（PBL）の位置付けと評価方法は？", ["education"]),
    ("特別支援教育における個別教育計画（IEP）の策定プロセスは？", ["education"]),
    ("高校の学校推薦型選抜（推薦入試）の選考基準と審査プロセスは？", ["education"]),
    ("教育委員会の教員配置計画への関与・説明責任の仕組みは？", ["education"]),
    ("算数教育における「活動・評価」の理論的基盤（算数科教育法）は？", ["education"]),
    ("教育課程編成指針に基づく学校独自の教科指導計画の策定方法は？", ["education"]),
    ("高等学校学習指導要領における「総合的な探究の時間」の位置付けは？", ["education"]),
    ("教員免許状更新制における研修プログラムの基準と認定方法は？", ["education"]),
    ("教育基本法第20条（教育の政治的中立性）の具体的な適用事例は？", ["education"]),
    ("小中学校の教育課程における道徳教育の評価基準と方法は？", ["education"]),
]

# Compound-domain questions (design doc 4.3: "questions spanning multiple
# domains") where more than one node's specialty is genuinely relevant.
_COMPOUND_QUESTIONS: list[tuple[str, list[str]]] = [
    (
        "仕事中に転倒して怪我をしました．治療費と休業補償について知りたいです．",
        ["medical", "legal"],
    ),
    (
        "交通事故で怪我をして通院していますが，慰謝料の相場が分かりません．",
        ["medical", "legal"],
    ),
    (
        "職場のハラスメントでうつ状態になり，休職を検討しています．",
        ["medical", "legal"],
    ),
    (
        "ペットが近隣トラブルの原因で怪我をさせてしまいました．治療費と責任について知りたいです．",
        ["medical", "legal"],
    ),
    (
        "学校で子供のアレルギー対応について，給食と保健室の両方の配慮が必要です．",
        ["education", "medical"],
    ),
    (
        "いじめの問題で，学校への対応と法的なアドバイスが必要です．",
        ["education", "legal"],
    ),
]


def _build_rows() -> list[dict]:
    """Assemble all question groups into flat dataset rows with sequential IDs."""
    groups: list[tuple[str, list[tuple[str, list[str]]]]] = [
        ("medical", _MEDICAL_QUESTIONS),
        ("legal", _LEGAL_QUESTIONS),
        ("general", _GENERAL_QUESTIONS),
        ("education", _EDUCATION_QUESTIONS),
        ("compound", _COMPOUND_QUESTIONS),
    ]
    rows = []
    for category, questions in groups:
        for index, (query, expected_domains) in enumerate(questions, start=1):
            rows.append(
                {
                    "id": f"{category}-{index:03d}",
                    "query": query,
                    "expected_domains": expected_domains,
                    "is_compound": len(expected_domains) > 1,
                }
            )
    return rows


def write_dataset(output: TextIO) -> int:
    """Write all dataset rows as JSON Lines to the given stream; return the row count."""
    rows = _build_rows()
    for row in rows:
        output.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Build the tier-2 (community-consultation) evaluation dataset as JSONL"
    )
    parser.add_argument(
        "--output", default=None, help="Output file path; defaults to stdout"
    )
    args = parser.parse_args()

    if args.output is None:
        count = write_dataset(sys.stdout)
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            count = write_dataset(f)
    print(f"[build_dataset] wrote {count} rows", file=sys.stderr)


if __name__ == "__main__":
    main()
