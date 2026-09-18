"""Iter59 (dispatch_candidate_ranking=multilabel_binary_relevance_head): offline
re-ranking of rank_2 dispatch candidates using the OvR head, without running
any of run_experiment.py's probe/dispatch/LLM-generation flow.

rank_1 (the production classifier's argmax) is taken verbatim from a
--baseline results.jsonl (a fixed dispatch_top_k=2 run, i.e. produced
BEFORE Iter58's gap-threshold escalation; see journal Iter59 investigation
finding 6 for why results/20260918_202613/results.jsonl, not the latest
run, is the required baseline) and is never recomputed here -- this
guarantees rank_1 invariance by construction, not just by assertion (A1
below is a redundant safety check on top of this construction, per Iter58's
"レバーを読むコードに到達しない" lesson: catch a reintroduced recomputation
bug even if someone edits build_new_rows() later).

Only rank_2 is replaced: for each row, the trained
models/dispatch_candidate_ranking_head.joblib (an uncalibrated
OneVsRestClassifier of independent per-domain LogisticRegressions, see
scripts/train_dispatch_candidate_ranking_head.py) scores all 10 domains
from the row's query embedding; the domain with the highest score among
the 9 domains other than rank_1 becomes rank_2. The head's
decision_function() output is used (not predict_proba(), which
OneVsRestClassifier renormalizes to sum to 1 across all classes for
single-label multiclass targets -- see
sklearn.multiclass.OneVsRestClassifier.predict_proba's implementation).
Applying a sigmoid to decision_function() recovers each domain's own
independent binary-relevance probability, which by construction does NOT
sum to 1 across domains (Iter59 investigation finding, Q1); sigmoid is
monotonic, so this does not change which domain ranks highest among the
remaining 9 -- it exists only to report an interpretable per-domain score
in the output, per journal.md Iter59 plan step (e).

Iter60 (multilabel_training_signal=synthetic_two_domain_training_examples)
extends this script to also load MultiLabelBinarizer-based heads (see
scripts/train_multilabel_dispatch_head.py), whose fitted
OneVsRestClassifier.classes_ is a plain integer column-index array rather
than domain-name strings (a structural side effect of going through
MultiLabelBinarizer; journal Iter60 investigation Q1). Such heads are
saved as a dict payload ({"model": ..., "classes": [domain names]}); this
script's _load_head() understands both that format and Iter59's bare
estimator (whose classes_ already are domain-name strings), and
_head_scores() always keys its returned dict by domain name (never by the
raw classes_ array) so this distinction cannot leak downstream -- see A6.

Only calls out to a live ollama node for embeddings (query_embedding is
not persisted in results.jsonl, so it must be recomputed for the head's
scoring; the production classifier's rank_1 is NOT recomputed, see above).
No LLM generation, probe, or dispatch traffic is produced. An embedding
cache (--embedding-cache, an .npz keyed by row id) avoids re-embedding all
1600 rows on repeated invocations while iterating on this script.

Usage (module mode; requires a live ollama node reachable for embeddings,
e.g. via the project's standard SSH local port forward,
`ssh -fNT -L 11435:localhost:11434 wafl500`):
    uv run python -m scripts.evaluate_dispatch_candidate_ranking \\
        --baseline results/20260918_202613/results.jsonl \\
        --head models/dispatch_candidate_ranking_head.joblib \\
        --embedding-model nomic-embed-text \\
        --ollama-host 127.0.0.1 --ollama-port 11435 \\
        --embedding-cache results/iter59_query_embeddings.npz \\
        --output results/iter59_ovr_ranking_predictions.jsonl
"""

import argparse
import asyncio
import json
import sys
from typing import TextIO

import joblib
import numpy as np
from sklearn.multiclass import OneVsRestClassifier

from expert_backend import OllamaClient
from metrics import compute_compound_coverage_metrics

# Every baseline row is expected to carry exactly a rank_1 + rank_2 pair
# (the fixed dispatch_top_k=2 configuration that produced
# results/20260918_202613/results.jsonl, before Iter58's gap policy). If a
# row doesn't, the baseline file is not the one this script was designed
# for (see module docstring), and continuing would silently compare against
# the wrong population.
_EXPECTED_BASELINE_DISPATCH_COUNT = 2

# N5 (journal.md Iter60 plan, "文体ショートカットの検出"): the synthetic
# two-domain rows are natural-language sentences while classifier_train.jsonl's
# 1427 existing rows are four-choice JMMLU questions, and the evaluation
# set mirrors this split (100 natural-language compound questions, 1500
# four-choice single-domain questions). A head that merely learned
# "natural-language style -> multi-label" rather than actual cross-domain
# content would still show a compound_domain_set_recall improvement while
# quietly losing single-domain argmax accuracy; this threshold (Iter59's
# measured value was 0.610) is reported for every --head, not just Iter60's.
_N5_SINGLE_DOMAIN_ARGMAX_ACCURACY_FLOOR = 0.590


def _read_jsonl(path: str) -> list[dict]:
    """Load JSON Lines rows from a file."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    """Elementwise logistic sigmoid, 1 / (1 + exp(-x))."""
    return 1.0 / (1.0 + np.exp(-logits))


def _load_embedding_cache(cache_path: str | None) -> dict[str, list[float]]:
    """Load a previously saved {id: embedding} cache, or an empty dict if absent."""
    if cache_path is None:
        return {}
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            ids = data["ids"]
            embeddings = data["embeddings"]
    except FileNotFoundError:
        return {}
    return {row_id: embeddings[i].tolist() for i, row_id in enumerate(ids)}


def _save_embedding_cache(cache_path: str, cache: dict[str, list[float]]) -> None:
    """Persist a {id: embedding} cache to an .npz file (arrays must share one dtype/shape)."""
    ids = list(cache.keys())
    embeddings = np.array([cache[row_id] for row_id in ids])
    np.savez(cache_path, ids=np.array(ids), embeddings=embeddings)


async def _get_query_embeddings(
    ollama_client: OllamaClient,
    embedding_model: str,
    rows: list[dict],
    cache_path: str | None,
) -> dict[str, list[float]]:
    """Return {row id: embedding} for every row, embedding only cache misses.

    Sequential (not concurrent) embed calls, matching
    train_domain_classifier.py's build_training_features and
    evaluate_classifier_calibration.py's existing pattern for single-node
    offline embedding jobs.
    """
    cache = _load_embedding_cache(cache_path)
    missing_rows = [row for row in rows if row["id"] not in cache]
    if missing_rows:
        for row in missing_rows:
            cache[row["id"]] = await ollama_client.embed(embedding_model, row["query"])
        if cache_path is not None:
            _save_embedding_cache(cache_path, cache)
    return cache


def _load_head(head_path: str) -> tuple[OneVsRestClassifier, list[str]]:
    """Load a dispatch-candidate-ranking head, returning (estimator, domain-name-per-column).

    Supports both artifact formats (see module docstring):
    - Iter59 (single-label OvR): a bare OneVsRestClassifier whose .classes_
      already are domain-name strings.
    - Iter60 (MultiLabelBinarizer-based OvR): a dict payload
      {"model": OneVsRestClassifier, "classes": [domain names]}, because
      that estimator's own .classes_ is a plain integer column-index array.
    """
    payload = joblib.load(head_path)
    if isinstance(payload, dict):
        return payload["model"], list(payload["classes"])
    return payload, list(payload.classes_)


def _head_scores(
    model: OneVsRestClassifier, classes: list[str], embedding: list[float]
) -> dict[str, float]:
    """Per-domain independent sigmoid score (not renormalized across domains); see module docstring.

    Always keyed by `classes` (the domain-name list resolved by
    _load_head()), never by `model.classes_` directly -- for an
    MLB-based head, model.classes_ would be integer column indices, not
    domain names (see A6 below, which asserts this never regresses).
    """
    logits = model.decision_function([embedding])[0]
    probabilities = _sigmoid(np.asarray(logits))
    return {domain: float(p) for domain, p in zip(classes, probabilities)}


def build_new_rows(
    baseline_rows: list[dict],
    model: OneVsRestClassifier,
    classes: list[str],
    embeddings_by_id: dict[str, list[float]],
) -> list[dict]:
    """Recompute rank_2 for every baseline row; rank_1 is copied through unchanged.

    Each output row carries `dispatched_domains = [rank_1, rank_2_new]` (for
    direct reuse by metrics.compute_compound_coverage_metrics), plus
    `head_scores` (all 10 domains, diagnostic) and `rank2_baseline` /
    `rank2_new` (diagnostic, for the rank2_flip_rate check below).
    """
    new_rows = []
    for row in baseline_rows:
        rank_1 = row["dispatched_domains"][0]
        rank2_baseline = row["dispatched_domains"][1]
        head_scores = _head_scores(model, classes, embeddings_by_id[row["id"]])
        rank2_new = max(
            (domain for domain in head_scores if domain != rank_1),
            key=lambda domain: head_scores[domain],
        )
        new_rows.append(
            {
                "id": row["id"],
                "expected_domains": row["expected_domains"],
                "selected_domain": row["selected_domain"],
                "dispatched_domains": [rank_1, rank2_new],
                "head_scores": head_scores,
                "rank2_baseline": rank2_baseline,
                "rank2_new": rank2_new,
            }
        )
    return new_rows


def _assert_rank1_unchanged(baseline_rows: list[dict], new_rows: list[dict]) -> None:
    """A1: every row's rank_1 must exactly match the baseline (no re-derivation of argmax)."""
    mismatches = [
        (base["id"], base["dispatched_domains"][0], new["dispatched_domains"][0])
        for base, new in zip(baseline_rows, new_rows)
        if base["dispatched_domains"][0] != new["dispatched_domains"][0]
    ]
    if mismatches:
        raise AssertionError(
            f"A1 (rank_1 invariance) failed on {len(mismatches)}/{len(baseline_rows)} rows, "
            f"e.g. {mismatches[:5]}"
        )


def _assert_cost_neutral(new_rows: list[dict]) -> float:
    """A2: every row keeps exactly 2 distinct dispatched domains; returns the mean dispatch count."""
    for row in new_rows:
        if len(row["dispatched_domains"]) != _EXPECTED_BASELINE_DISPATCH_COUNT:
            raise AssertionError(
                f"A2 (cost neutrality) failed: row {row['id']!r} has "
                f"{len(row['dispatched_domains'])} dispatched domains, "
                f"expected {_EXPECTED_BASELINE_DISPATCH_COUNT}"
            )
        if row["dispatched_domains"][0] == row["dispatched_domains"][1]:
            raise AssertionError(
                f"A2 (cost neutrality) failed: row {row['id']!r} has a duplicate "
                f"rank_1/rank_2 domain ({row['dispatched_domains'][0]!r})"
            )
    mean_dispatch = sum(len(row["dispatched_domains"]) for row in new_rows) / len(new_rows)
    if mean_dispatch != _EXPECTED_BASELINE_DISPATCH_COUNT:
        raise AssertionError(
            f"A2 (cost neutrality) failed: mean dispatch = {mean_dispatch}, "
            f"expected {_EXPECTED_BASELINE_DISPATCH_COUNT:.6f}"
        )
    return mean_dispatch


def _compute_rank2_flip_rate(new_rows: list[dict]) -> float:
    """A3: fraction of rows where the new rank_2 differs from the baseline's rank_2."""
    flips = sum(1 for row in new_rows if row["rank2_new"] != row["rank2_baseline"])
    return flips / len(new_rows)


def _assert_head_scores_are_domain_names(new_rows: list[dict], classes: list[str]) -> None:
    """A6: head_scores' keys must be exactly the domain-name strings, never integer column indices.

    This is the direct machine-checkable guard against the MLB pitfall
    documented in the module docstring: if _load_head()/_head_scores() ever
    regress to keying by a bare OneVsRestClassifier.classes_ on an
    MLB-trained head, this would silently degrade to integer keys (0..9)
    rather than raising -- so this assertion checks every row's key set
    against the expected domain-name set explicitly.
    """
    expected_keys = set(classes)
    for row in new_rows:
        actual_keys = set(row["head_scores"].keys())
        if not all(isinstance(key, str) for key in actual_keys):
            raise AssertionError(
                f"A6 (head_scores domain-name keys) failed on row {row['id']!r}: "
                f"non-string keys found: {actual_keys}"
            )
        if actual_keys != expected_keys:
            raise AssertionError(
                f"A6 (head_scores domain-name keys) failed on row {row['id']!r}: "
                f"keys {actual_keys} != expected domain set {expected_keys}"
            )


def _compute_single_domain_argmax_accuracy(baseline_rows: list[dict], new_rows: list[dict]) -> dict:
    """N5: argmax accuracy of the head's own 10-domain scores on single-domain (non-compound) rows.

    A head that learned "natural-language style implies multi-label" rather
    than genuine cross-domain content could still show a
    compound_domain_set_recall improvement while quietly failing to even
    recover the single, unambiguous domain on the 1500 JMMLU-style rows
    (see module docstring). Iter59's measured value was 0.610; this is
    reported (not raised) for every --head invocation.
    """
    single_domain_pairs = [
        (base, new)
        for base, new in zip(baseline_rows, new_rows)
        if len(base["expected_domains"]) == 1
    ]
    correct = sum(
        1
        for base, new in single_domain_pairs
        if max(new["head_scores"], key=lambda domain: new["head_scores"][domain])
        == base["expected_domains"][0]
    )
    accuracy = correct / len(single_domain_pairs) if single_domain_pairs else 0.0
    return {
        "n_single_domain_rows": len(single_domain_pairs),
        "correct": correct,
        "accuracy": accuracy,
        "floor": _N5_SINGLE_DOMAIN_ARGMAX_ACCURACY_FLOOR,
        "pass": accuracy >= _N5_SINGLE_DOMAIN_ARGMAX_ACCURACY_FLOOR,
    }


def _compute_a5_iter59_disagreement(new_rows: list[dict], iter59_predictions_path: str) -> dict:
    """A5: rank_2 disagreement count between this head and Iter59's single-label OvR head.

    This iteration's single lever is the teacher signal (single-label vs
    true multi-label); if the resulting head's rank_2 choice never differs
    from Iter59's on any of the 1600 rows, the 153 synthetic rows did not
    move the head at all -- i.e. this iteration's own no-op, distinct from
    A3's no-op check against the *production classifier's* baseline rank_2.
    """
    iter59_rows = {row["id"]: row for row in _read_jsonl(iter59_predictions_path)}
    mismatches = sum(
        1 for row in new_rows if row["rank2_new"] != iter59_rows[row["id"]]["rank2_new"]
    )
    rate = mismatches / len(new_rows)
    if mismatches == 0:
        print(
            "[evaluate_dispatch_candidate_ranking] WARNING: A5 mismatches == 0; the "
            "synthetic multi-label rows may not be firing relative to Iter59's "
            "single-label-trained head. See journal.md Iter60 plan, A5.",
            file=sys.stderr,
        )
    return {"n_rows": len(new_rows), "mismatches": mismatches, "mismatch_rate": rate}


async def _run(
    baseline_path: str,
    head_path: str,
    embedding_model: str,
    ollama_host: str,
    ollama_port: int,
    embedding_cache_path: str | None,
    output: TextIO,
    iter59_predictions_path: str | None = None,
) -> None:
    baseline_rows = _read_jsonl(baseline_path)
    for row in baseline_rows:
        if len(row["dispatched_domains"]) != _EXPECTED_BASELINE_DISPATCH_COUNT:
            raise ValueError(
                f"baseline row {row['id']!r} has {len(row['dispatched_domains'])} "
                f"dispatched_domains, expected {_EXPECTED_BASELINE_DISPATCH_COUNT} "
                "-- is --baseline the fixed dispatch_top_k=2 results.jsonl "
                "(results/20260918_202613/results.jsonl), not a gap-policy run?"
            )

    model, classes = _load_head(head_path)
    ollama_client = OllamaClient(host=f"http://{ollama_host}:{ollama_port}")
    embeddings_by_id = await _get_query_embeddings(
        ollama_client, embedding_model, baseline_rows, embedding_cache_path
    )

    new_rows = build_new_rows(baseline_rows, model, classes, embeddings_by_id)

    _assert_rank1_unchanged(baseline_rows, new_rows)
    mean_dispatch = _assert_cost_neutral(new_rows)
    _assert_head_scores_are_domain_names(new_rows, classes)
    rank2_flip_rate = _compute_rank2_flip_rate(new_rows)
    if rank2_flip_rate == 0.0:
        print(
            "[evaluate_dispatch_candidate_ranking] WARNING: rank2_flip_rate == 0.0; "
            "the OvR head may not be firing (e.g. it agrees with the production "
            "classifier's ranking on every row). See journal.md Iter59 plan, A3.",
            file=sys.stderr,
        )
    n5_result = _compute_single_domain_argmax_accuracy(baseline_rows, new_rows)
    if not n5_result["pass"]:
        print(
            f"[evaluate_dispatch_candidate_ranking] WARNING: N5 single-domain argmax "
            f"accuracy {n5_result['accuracy']:.4f} < floor {n5_result['floor']}; see "
            "journal.md Iter60 plan, N5 (style-shortcut check).",
            file=sys.stderr,
        )
    a5_result = None
    if iter59_predictions_path is not None:
        a5_result = _compute_a5_iter59_disagreement(new_rows, iter59_predictions_path)

    compound_metrics = compute_compound_coverage_metrics(new_rows)

    for row in new_rows:
        output.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "n_rows": len(new_rows),
        "mean_dispatch": mean_dispatch,
        "rank2_flip_rate": rank2_flip_rate,
        "compound_domain_set_recall": compound_metrics["compound_domain_set_recall"],
        "compound_rows_evaluated": compound_metrics["compound_rows_evaluated"],
        "n5_single_domain_argmax_accuracy": n5_result,
    }
    if a5_result is not None:
        summary["a5_iter59_disagreement"] = a5_result
    print(json.dumps(summary, ensure_ascii=False, indent=2), file=sys.stderr)
    print(
        f"[evaluate_dispatch_candidate_ranking] wrote {len(new_rows)} rows (head={head_path})",
        file=sys.stderr,
    )


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Re-rank rank_2 dispatch candidates offline using the Iter59 OvR head, "
            "keeping rank_1 identical to a fixed dispatch_top_k=2 baseline results.jsonl"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--baseline",
        required=True,
        help="Fixed dispatch_top_k=2 results.jsonl (e.g. results/20260918_202613/results.jsonl)",
    )
    parser.add_argument(
        "--head", required=True, help="Path to the OneVsRestClassifier joblib artifact"
    )
    parser.add_argument(
        "--embedding-model", required=True, help="Must match config.yaml's embedding_model"
    )
    parser.add_argument("--ollama-host", required=True, help="A live node's ollama daemon host/IP")
    parser.add_argument("--ollama-port", type=int, default=11434)
    parser.add_argument(
        "--embedding-cache",
        default=None,
        help="Optional .npz path to cache/reuse query embeddings across runs (keyed by row id)",
    )
    parser.add_argument("--output", required=True, help="Path to write the new-rows JSONL to")
    parser.add_argument(
        "--iter59-predictions",
        default=None,
        help="Optional: Iter59's predictions JSONL (results/iter59_ovr_ranking_predictions.jsonl), "
        "for A5's rank_2 disagreement report against the single-label-trained head",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = _parse_args()
    with open(args.output, "w", encoding="utf-8") as f:
        asyncio.run(
            _run(
                args.baseline,
                args.head,
                args.embedding_model,
                args.ollama_host,
                args.ollama_port,
                args.embedding_cache,
                f,
                iter59_predictions_path=args.iter59_predictions,
            )
        )


if __name__ == "__main__":
    main()
