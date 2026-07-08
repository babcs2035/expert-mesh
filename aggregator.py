"""probe結果の集約・上位k選定を行う純粋ロジック．

dispatch対象が0件だった場合のフォールバック回答生成そのものは，ローカル推論を
要する副作用のある処理のためnode.py側の責務とし，本モジュールは選定結果
（空リストなら「担当者不在」）を返すところまでを扱う．
"""

from protocol import ProbeResponse


def select_dispatch_targets(
    probe_responses: list[ProbeResponse],
    confidence_threshold: float,
    top_k: int = 1,
) -> list[ProbeResponse]:
    """confidenceが閾値以上のノードを上位k件選定する．

    probe_responsesの並び順（=peers.yamlの記載順を反映した呼び出し順）を保つ
    安定ソートで降順に並べるため，confidenceが同点の場合はpeers.yaml記載順で
    先勝ちとなる．該当者がいない場合は空リストを返す．
    """
    eligible = [r for r in probe_responses if r.confidence >= confidence_threshold]
    return sorted(eligible, key=lambda r: r.confidence, reverse=True)[:top_k]
