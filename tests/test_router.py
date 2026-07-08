"""router.pyのconfidence算出ロジック（純粋関数部分）を検証する．"""

from router import PARSE_FAILURE_CONFIDENCE, build_confidence_prompt, parse_confidence


def test_build_confidence_prompt_includes_domain_and_summary() -> None:
    """プロンプトに専門分野名と質問要約が含まれる．"""
    prompt = build_confidence_prompt("medical", "頭痛と発熱についての質問")
    assert "medical" in prompt
    assert "頭痛と発熱についての質問" in prompt


def test_parse_confidence_extracts_value_from_clean_json() -> None:
    """整形されたJSON応答からconfidenceを取り出せる．"""
    assert parse_confidence('{"confidence": 0.87}') == 0.87


def test_parse_confidence_extracts_value_from_json_with_surrounding_text() -> None:
    """LLMがJSON前後に余計な文章を付与しても抽出できる．"""
    raw = '確信度は次の通りです．\n{"confidence": 0.42}\nご参考までに．'
    assert parse_confidence(raw) == 0.42


def test_parse_confidence_falls_back_on_invalid_json() -> None:
    """JSONとして解釈できない応答は担当外（0.0）に倒す．"""
    assert parse_confidence("わかりません") == PARSE_FAILURE_CONFIDENCE


def test_parse_confidence_falls_back_on_missing_key() -> None:
    """confidenceキーを欠いたJSONも担当外に倒す．"""
    assert parse_confidence('{"score": 0.9}') == PARSE_FAILURE_CONFIDENCE


def test_parse_confidence_clamps_value_above_one() -> None:
    """LLMが1.0を超える値を出力しても1.0にクランプする．"""
    assert parse_confidence('{"confidence": 1.5}') == 1.0


def test_parse_confidence_clamps_negative_value() -> None:
    """負の値も0.0にクランプする．"""
    assert parse_confidence('{"confidence": -0.3}') == 0.0
