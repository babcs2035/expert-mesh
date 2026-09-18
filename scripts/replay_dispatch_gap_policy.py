"""Iter58 (dispatch_policy=adaptive_confidence_gap): offline replay of the
chained-escalation gap policy over an existing results.jsonl, to confirm
(T, max_k) before spending any real-machine time.

`aggregator.select_dispatch_targets()`'s gap-escalation branch is a pure
function of each row's already-recorded `probe_candidates` confidences (the
classifier output is deterministic given the embedding, which is unchanged
by this lever). This script re-implements exactly the same decision rule in
Python and re-derives `dispatched_domains` for every row under a grid of
(T, max_k) values, without touching any live node.

Decision rule (must match aggregator.py's gap_threshold branch verbatim):
    candidates = probe_candidates sorted by confidence descending (stable)
    k = 1
    while k < max_k and (candidates[k-1].confidence - candidates[k].confidence) < T:
        k += 1
    dispatched_domains = {c.domain for c in candidates[:k]}

This mirrors the current production gate (confidence_threshold=0.0,
dispatch_candidate_threshold=0.0), under which every node's probe qualifies
and `candidates` is simply the full 10-node list sorted by confidence -- see
aggregator.py:select_dispatch_targets() and journal.md Iter58 "レバーが
読まれるコード行と到達条件". If those thresholds are ever changed, this
script's `candidates` derivation would need to add the same gating before
sorting; it does not attempt to reproduce that gating today because the
source results.jsonl was produced with the gate fully open (see
--source-note below).

Metric definitions replicate metrics.py's
compute_compound_coverage_metrics() exactly (covered/expected_domain_total
over rows with len(expected_domains) > 1), so that the offline replay value
is directly comparable to the real-run metrics.json field of the same name.

Usage:
    uv run python scripts/replay_dispatch_gap_policy.py \\
        --results results/20260918_202613/results.jsonl \\
        --t-min 0.01 --t-max 0.60 --t-step 0.01 \\
        --max-k-values 3,4,5,6 \\
        --single-domain-cost-limit 2.40 --overall-cost-limit 2.45

Prints the full grid (one row per (T, max_k)) as TSV to stdout, and the
selection result (event pre-registered rule: among grid points satisfying
both cost limits, pick the max compound_domain_set_recall; ties broken by
lower mean dispatch cost) as a JSON summary to stderr.
"""

import argparse
import json
import sys
from dataclasses import dataclass

# Source data was produced under confidence_threshold=0.0 /
# dispatch_candidate_threshold=0.0 (config.yaml, Iter57 HEAD), i.e. the gate
# is fully open and every one of the 10 probed nodes qualifies as a
# candidate. This script assumes that gate is still fully open, matching the
# journal.md Iter58 note "現行設定... では rank_1 は常に適格・候補は常に
# 10 件あるため，1600 問すべてでこのループに到達する".
_EXPECTED_CANDIDATE_COUNT = 10


@dataclass(frozen=True)
class _Candidate:
    """A single node's probe confidence for one dataset row, sorted-ready."""

    domain: str
    confidence: float


def _load_rows(results_path: str) -> list[dict]:
    """Read a results.jsonl file into a list of row dicts (one per line)."""
    rows = []
    with open(results_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _sorted_candidates(row: dict) -> list[_Candidate]:
    """Extract and confidence-sort a row's probe_candidates (stable sort,
    matching aggregator.select_dispatch_targets()'s `sorted(..., reverse=True)`
    over the peers.yaml declaration order preserved in probe_candidates)."""
    candidates = [
        _Candidate(domain=c["domain"], confidence=c["confidence"]) for c in row["probe_candidates"]
    ]
    return sorted(candidates, key=lambda c: c.confidence, reverse=True)


def _decide_k(candidates: list[_Candidate], gap_threshold: float, gap_max_k: int) -> int:
    """Chained-escalation k decision, verbatim port of aggregator.py's
    gap_threshold branch (candidates already gate-qualified and sorted)."""
    k = 1
    ceiling = min(gap_max_k, len(candidates))
    while k < ceiling and (candidates[k - 1].confidence - candidates[k].confidence) < gap_threshold:
        k += 1
    return k


def _replay_one_setting(rows: list[dict], gap_threshold: float, gap_max_k: int) -> dict:
    """Compute compound_domain_set_recall and mean-dispatch-count metrics
    for one (T, max_k) setting, replicating metrics.py's
    compute_compound_coverage_metrics() aggregation exactly for the compound
    side, plus the single-domain / overall mean dispatch counts requested by
    the pre-registered selection rule (journal.md Iter58 計画節)."""
    covered_domain_count = 0
    expected_domain_total = 0
    single_domain_dispatch_sum = 0
    single_domain_row_count = 0
    overall_dispatch_sum = 0

    for row in rows:
        candidates = _sorted_candidates(row)
        if len(candidates) != _EXPECTED_CANDIDATE_COUNT:
            raise ValueError(
                f"row {row.get('id')!r} has {len(candidates)} probe_candidates, "
                f"expected {_EXPECTED_CANDIDATE_COUNT}; gate-open assumption may not hold"
            )
        k = _decide_k(candidates, gap_threshold, gap_max_k)
        dispatched = {c.domain for c in candidates[:k]}
        overall_dispatch_sum += len(dispatched)

        expected = set(row["expected_domains"])
        if len(expected) > 1:
            covered_domain_count += len(dispatched & expected)
            expected_domain_total += len(expected)
        else:
            single_domain_dispatch_sum += len(dispatched)
            single_domain_row_count += 1

    return {
        "gap_threshold": round(gap_threshold, 2),
        "gap_max_k": gap_max_k,
        "compound_domain_set_recall": covered_domain_count / expected_domain_total,
        "single_domain_mean_dispatch": single_domain_dispatch_sum / single_domain_row_count,
        "overall_mean_dispatch": overall_dispatch_sum / len(rows),
    }


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the (T, max_k) grid sweep."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", required=True, help="Path to a results.jsonl with probe_candidates.")
    parser.add_argument("--t-min", type=float, default=0.01)
    parser.add_argument("--t-max", type=float, default=0.60)
    parser.add_argument("--t-step", type=float, default=0.01)
    parser.add_argument("--max-k-values", default="3,4,5,6", help="Comma-separated gap_max_k values.")
    parser.add_argument(
        "--single-domain-cost-limit",
        type=float,
        default=2.40,
        help="Pre-registered selection rule cost bound on single-domain-row mean dispatch count.",
    )
    parser.add_argument(
        "--overall-cost-limit",
        type=float,
        default=2.45,
        help="Pre-registered selection rule cost bound on overall mean dispatch count.",
    )
    return parser.parse_args()


def main() -> None:
    """Sweep the (T, max_k) grid, print it as TSV, and print the
    pre-registered selection result as a JSON summary."""
    args = _parse_args()
    rows = _load_rows(args.results)
    max_k_values = [int(v) for v in args.max_k_values.split(",")]

    # T grid built from integer cents to avoid float-step accumulation error
    # (0.01 repeated addition drifts by ~1e-16 per step over 60 steps).
    t_min_cents = round(args.t_min * 100)
    t_max_cents = round(args.t_max * 100)
    t_step_cents = round(args.t_step * 100)
    t_values = [c / 100 for c in range(t_min_cents, t_max_cents + 1, t_step_cents)]

    grid_results = []
    for gap_max_k in max_k_values:
        for t in t_values:
            grid_results.append(_replay_one_setting(rows, t, gap_max_k))

    header = [
        "gap_threshold",
        "gap_max_k",
        "compound_domain_set_recall",
        "single_domain_mean_dispatch",
        "overall_mean_dispatch",
    ]
    print("\t".join(header))
    for r in grid_results:
        print("\t".join(str(r[k]) for k in header))

    # Pre-registered selection rule (journal.md Iter58 計画節): among grid
    # points satisfying both cost limits, pick the max
    # compound_domain_set_recall; ties broken by lower overall_mean_dispatch.
    eligible = [
        r
        for r in grid_results
        if r["single_domain_mean_dispatch"] <= args.single_domain_cost_limit
        and r["overall_mean_dispatch"] <= args.overall_cost_limit
    ]
    selection = None
    if eligible:
        selection = max(
            eligible,
            key=lambda r: (r["compound_domain_set_recall"], -r["overall_mean_dispatch"]),
        )

    print(
        json.dumps(
            {
                "eligible_count": len(eligible),
                "grid_size": len(grid_results),
                "selected": selection,
            },
            ensure_ascii=False,
            indent=2,
        ),
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
