"""方式B（自己申告スコアリング）による担当可否confidenceの算出ロジック．"""

import json
import re

from expert_backend import OllamaClient

_CONFIDENCE_JSON_PATTERN = re.compile(r"\{.*\}", re.DOTALL)
PARSE_FAILURE_CONFIDENCE = 0.0
# {"confidence": 0.87}程度のJSON出力に十分な上限．軽量モデルが指示に従わず長文を
# 生成し続けるのを防ぐガードレール（ollamaはnum_predict未指定だと無制限に生成する）．
CONFIDENCE_MAX_TOKENS = 100
# ollamaのデフォルト温度では同一質問でもconfidenceが0.0〜1.0の間で大きく揺れ，
# 自己申告スコアリングとして機能しないため，決定性を高めるために低く設定する．
CONFIDENCE_TEMPERATURE = 0.1


def build_confidence_prompt(domain: str, query_summary: str) -> str:
    """軽量モデルへ渡す担当可否判定プロンプトを組み立てる．"""
    return (
        f'あなたは「{domain}」分野の専門家ノードです．'
        f"次の質問要約が，あなたの専門分野にどの程度該当するかを判定してください．\n"
        f"質問要約: {query_summary}\n"
        '0.0（担当外）から1.0（完全に担当）までの確信度を，'
        '{"confidence": 0.0} の形式のJSONのみで出力してください．'
    )


def parse_confidence(raw_response: str) -> float:
    """LLM出力からconfidence値を抽出する．

    外部LLM出力のJSON解析に失敗した場合は，誤って高confidenceを名乗り出て
    不要な/dispatchを誘発するより安全な「担当外（0.0）」に倒す．
    """
    match = _CONFIDENCE_JSON_PATTERN.search(raw_response)
    if match is None:
        return PARSE_FAILURE_CONFIDENCE
    try:
        parsed = json.loads(match.group())
        confidence = float(parsed["confidence"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return PARSE_FAILURE_CONFIDENCE
    return min(max(confidence, 0.0), 1.0)


async def estimate_confidence(
    ollama_client: OllamaClient,
    light_model: str,
    domain: str,
    query_summary: str,
    timeout_s: float,
) -> float:
    """軽量モデルに担当可否を問い合わせ，0〜1のconfidenceを返す．"""
    prompt = build_confidence_prompt(domain, query_summary)
    raw_response = await ollama_client.generate(
        light_model,
        prompt,
        timeout_s=timeout_s,
        max_tokens=CONFIDENCE_MAX_TOKENS,
        temperature=CONFIDENCE_TEMPERATURE,
    )
    return parse_confidence(raw_response)
