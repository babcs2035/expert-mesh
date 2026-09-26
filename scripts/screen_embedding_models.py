"""Iter81 G1 (embedding_input_instruction_prefix): offline 5-fold CV screening
of instruction-prefix wording candidates on data/classifier_train.jsonl ONLY.

The embedding model is fixed to qwen3-embedding:0.6b (Iter79/80's adopted
baseline; unchanged by this lever). What varies across candidates is the
instruction prefix text prepended to every row's query before embedding
(journal.md Iteration 81 plan, table of P0-P3). This module was rewritten
from its Iter80 form (which instead compared embedding *models*); the CV
machinery (StratifiedKFold + LogisticRegression + sample_weight, reusing
scripts/train_domain_classifier.py's helpers) is unchanged.

Selection rule (registered in journal.md Iter81 plan, fixed before results
are seen): P1/P2/P3 are scored with the same 5-fold CV as P0 (no prefix,
the Iter79 baseline, cv_accuracy=0.7561). The candidate with the highest CV
accuracy among {P1, P2, P3} is selected; ties broken by macro-F1, then by
preferring P1 > P2 > P3. If all three score below P0, the best of the three
is still selected (config.yml absolute condition A: a regression here is
not grounds for silently keeping the current value).

Deliberately never reads data/dataset.jsonl (the 1,915-row evaluation set):
doing so here would leak the evaluation set into prefix selection, the same
information-leakage concern train_domain_classifier.py's docstring raises
for probe/dispatch-derived features (Iter10).

Per-candidate embeddings are cached to
data/embcache_qwen3-embedding_0.6b__<candidate_id>.npy (P0 reuses the plain
Iter79 cache with no suffix). The cache path MUST include the prefix
identifier: an earlier repo-wide failure mode (recurring across at least 6
prior iterations, journal.md Iteration 81 change-table row #5) was a config
value being changed correctly but silently not reaching the code that reads
it; here that would look like a prefixed candidate silently loading the
prefix-less Iter79 cache because the cache key did not distinguish them.

Usage (module mode; run against wafl-ctrl5 via the SSH tunnel per the
2026-09-19 operational rule -- this script must NOT be pointed at
wafl500-509):
    ssh -fNT -L 11499:localhost:11434 wafl-ctrl5
    uv run python -m scripts.screen_embedding_models \\
        --train-data data/classifier_train.jsonl \\
        --ollama-host 127.0.0.1 --ollama-port 11499
"""

import argparse
import asyncio
import json
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from expert_backend import OllamaClient
from scripts.train_domain_classifier import (
    _extract_sample_weights,
    _load_training_rows,
    build_training_features,
)

# Fixed embedding model for this lever (unchanged from Iter79/80's adopted value).
_EMBEDDING_MODEL = "qwen3-embedding:0.6b"

# Fixed, pre-registered candidate prefixes (journal.md Iter81 plan). task_description
# is a single English sentence shared across all 10 domains (2026-09-23 operational
# rule against domain-specific post-hoc corrections, backlog B127 review point (1)).
_TASK_DESCRIPTION = (
    "Given a user question, identify the single academic or professional domain "
    "it belongs to"
)
_CANDIDATES: dict[str, str | None] = {
    "p0": None,  # no prefix (Iter79 baseline)
    "p1": f"Instruct: {_TASK_DESCRIPTION}\nQuery: {{text}}",
    "p2": f"Instruct: {_TASK_DESCRIPTION}\nInput: {{text}}",
    "p3": "Instruct: Given a web search query, retrieve relevant passages that "
          "answer the query\nQuery: {text}",
}
_BASELINE_CANDIDATE = "p0"
# Preference order among non-baseline candidates when CV accuracy and macro-F1 tie.
_PREFERENCE_ORDER = ["p1", "p2", "p3"]

_CV_SPLITS = 5
_CV_RANDOM_STATE = 42


def _render_prompt(template: str | None, text: str) -> str:
    """Apply a candidate's prefix template (if any) to a raw query string.

    P0's template is None, meaning the raw text is embedded unchanged. P1-P3's
    templates carry the full literal string (including the `Instruct: ...` prefix)
    with a `{text}` placeholder for the query, matching the exact wording table
    registered in journal.md Iteration 81 plan.
    """
    return text if template is None else template.format(text=text)


def _cache_path(train_data_path: str, candidate_id: str) -> str:
    """Derive data/embcache_qwen3-embedding_0.6b__<candidate_id>.npy from --train-data's directory.

    P0 (no prefix) reuses the plain Iter79 cache name with no candidate suffix,
    since its embeddings are identical to that run's. P1-P3 each get a distinct
    suffix so a prefixed candidate can never silently read another candidate's
    (or P0's prefix-less) cached embeddings -- see module docstring.
    """
    import os

    safe_model = _EMBEDDING_MODEL.replace(":", "_").replace("/", "_")
    data_dir = os.path.dirname(train_data_path) or "."
    if candidate_id == _BASELINE_CANDIDATE:
        return os.path.join(data_dir, f"embcache_{safe_model}.npy")
    return os.path.join(data_dir, f"embcache_{safe_model}__{candidate_id}.npy")


async def _embed_candidate(
    ollama_client: OllamaClient, candidate_id: str, rows: list[dict], cache_path: str
) -> np.ndarray:
    """Return (n_rows, dim) embeddings for one prefix candidate, using/populating a cache file."""
    import os

    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        if cached.shape[0] == len(rows):
            print(f"[screen_embedding_models] using cached embeddings: {cache_path}", file=sys.stderr)
            return cached
        print(
            f"[screen_embedding_models] cache row count mismatch ({cached.shape[0]} != {len(rows)}), "
            f"recomputing: {cache_path}",
            file=sys.stderr,
        )

    template = _CANDIDATES[candidate_id]
    prefixed_rows = [{**row, "query": _render_prompt(template, row["query"])} for row in rows]
    embeddings, _labels = await build_training_features(ollama_client, _EMBEDDING_MODEL, prefixed_rows)
    arr = np.array(embeddings, dtype=np.float64)
    np.save(cache_path, arr)
    print(f"[screen_embedding_models] wrote {cache_path} shape={arr.shape}", file=sys.stderr)
    return arr


def _cross_validate(embeddings: np.ndarray, labels: list[str], sample_weight: list[float]) -> dict[str, float]:
    """5-fold StratifiedKFold CV accuracy / macro-F1 for a single prefix candidate.

    Mirrors train_domain_classifier.train_classifier's base estimator
    (LogisticRegression(max_iter=1000, class_weight=None) + sample_weight)
    but without CalibratedClassifierCV, since candidate selection only needs
    argmax predictions, not calibrated probabilities.
    """
    labels_arr = np.array(labels)
    weights_arr = np.array(sample_weight)
    skf = StratifiedKFold(n_splits=_CV_SPLITS, shuffle=True, random_state=_CV_RANDOM_STATE)

    fold_accuracies = []
    fold_macro_f1s = []
    for train_idx, test_idx in skf.split(embeddings, labels_arr):
        clf = LogisticRegression(max_iter=1000, class_weight=None)
        clf.fit(embeddings[train_idx], labels_arr[train_idx], sample_weight=weights_arr[train_idx])
        preds = clf.predict(embeddings[test_idx])
        fold_accuracies.append(accuracy_score(labels_arr[test_idx], preds))
        fold_macro_f1s.append(f1_score(labels_arr[test_idx], preds, average="macro"))

    return {
        "cv_accuracy_mean": float(np.mean(fold_accuracies)),
        "cv_accuracy_folds": [float(a) for a in fold_accuracies],
        "cv_macro_f1_mean": float(np.mean(fold_macro_f1s)),
        "cv_macro_f1_folds": [float(f) for f in fold_macro_f1s],
    }


async def _run(train_data_path: str, ollama_host: str, ollama_port: int) -> dict:
    """Score P0-P3 with 5-fold CV and apply the Iter81 selection rule."""
    rows = _load_training_rows(train_data_path)
    labels = [row["domain"] for row in rows]
    sample_weight = _extract_sample_weights(rows)
    ollama_client = OllamaClient(host=f"http://{ollama_host}:{ollama_port}")

    results: dict[str, dict] = {}
    for candidate_id in _CANDIDATES:
        cache_path = _cache_path(train_data_path, candidate_id)
        embeddings = await _embed_candidate(ollama_client, candidate_id, rows, cache_path)
        results[candidate_id] = _cross_validate(embeddings, labels, sample_weight)

    # Selection rule (journal.md Iter81 plan): among P1-P3, pick max CV accuracy;
    # ties broken by macro-F1, then by preference order P1 > P2 > P3. P0 is never
    # itself "selected" (selecting it would mean no change, i.e. the lever is a no-op).
    eligible = [name for name in _CANDIDATES if name != _BASELINE_CANDIDATE]

    def _sort_key(name: str) -> tuple[float, float, int]:
        r = results[name]
        preference_rank = len(_PREFERENCE_ORDER) - _PREFERENCE_ORDER.index(name)
        return (r["cv_accuracy_mean"], r["cv_macro_f1_mean"], preference_rank)

    selected_candidate = max(eligible, key=_sort_key)

    return {
        "baseline_cv_accuracy": results[_BASELINE_CANDIDATE]["cv_accuracy_mean"],
        "selected_candidate": selected_candidate,
        "selected_prefix_template": _CANDIDATES[selected_candidate],
        "candidates": results,
        "prefix_templates": _CANDIDATES,
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Iter81 G1: offline CV screening of instruction-prefix wording candidates "
        "on data/classifier_train.jsonl only (no dataset.jsonl access)."
    )
    parser.add_argument("--train-data", default="data/classifier_train.jsonl")
    parser.add_argument(
        "--ollama-host",
        default="127.0.0.1",
        help="wafl-ctrl5 tunnel endpoint (2026-09-19 operational rule: never wafl500-509)",
    )
    parser.add_argument("--ollama-port", type=int, default=11499)
    args = parser.parse_args()

    result = asyncio.run(_run(args.train_data, args.ollama_host, args.ollama_port))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
