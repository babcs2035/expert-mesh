"""Iter80 G1 (embedding_model_replacement): offline 5-fold CV screening of
embedding-model candidates on data/classifier_train.jsonl ONLY.

Selection rule (registered in journal.md Iter80 plan, fixed before results
are seen): all three candidates -- qwen3-embedding:0.6b (current baseline,
reuses the Iter79 embcache), qwen3-embedding:4b, and bge-m3 -- are scored
simultaneously with the same 5-fold CV. The value adopted for
embedding_model_replacement is the candidate with the highest CV accuracy
among those that ALSO pass the G0 VRAM gate (measured separately, offline,
via `ollama ps` / `nvidia-smi` on wafl-ctrl5 -- this script does not measure
VRAM). Ties are broken by macro-F1, then by preferring qwen3-embedding:4b.
If every non-baseline candidate that passes G0 scores below the 0.6b
baseline, the best surviving non-baseline candidate is still selected (the
run proceeds; a regression is not grounds for silently keeping the current
value -- see config.yml Iter80 lever note, condition A).

Deliberately never reads data/dataset.jsonl (the 1,915-row evaluation set):
doing so here would leak the evaluation set into model selection, the same
information-leakage concern train_domain_classifier.py's docstring raises
for probe/dispatch-derived features (Iter10).

Reuses scripts/train_domain_classifier.py's _load_training_rows,
_extract_sample_weights, and train_classifier so the CV here exercises the
exact same LogisticRegression + sample_weight configuration as production
training, differing only in the embedding model and in using bare
StratifiedKFold instead of CalibratedClassifierCV's internal one (the
calibration wrapper is irrelevant to a model-selection signal that only
needs argmax accuracy / macro-F1, and skipping it avoids paying temperature-
scaling cost 5x per candidate).

Per-candidate embeddings are cached to data/embcache_<model>.npy so a
subsequent scripts/train_domain_classifier.py run for the selected model
does not need to re-embed data/classifier_train.jsonl from scratch (the
cache uses the exact row order of --train-data, so it can be reused as long
as that file is unchanged).

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

# Fixed, pre-registered candidate list (journal.md Iter80 plan). All three
# are scored in the same run; "qwen3-embedding:0.6b" is the current baseline
# and its embeddings are read from the Iter79 cache (no re-embedding cost).
_CANDIDATES = ["qwen3-embedding:0.6b", "qwen3-embedding:4b", "bge-m3"]
_BASELINE_CANDIDATE = "qwen3-embedding:0.6b"

_CV_SPLITS = 5
_CV_RANDOM_STATE = 42


def _cache_path(train_data_path: str, embedding_model: str) -> str:
    """Derive data/embcache_<model>.npy from --train-data's directory.

    Sanitizes ':' and '/' (present in model names like "qwen3-embedding:0.6b")
    into '_' so the path is a single valid filename component.
    """
    import os

    safe_model = embedding_model.replace(":", "_").replace("/", "_")
    data_dir = os.path.dirname(train_data_path) or "."
    return os.path.join(data_dir, f"embcache_{safe_model}.npy")


async def _embed_candidate(
    ollama_client: OllamaClient, embedding_model: str, rows: list[dict], cache_path: str
) -> np.ndarray:
    """Return (n_rows, dim) embeddings for embedding_model, using/populating a cache file."""
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

    embeddings, _labels = await build_training_features(ollama_client, embedding_model, rows)
    arr = np.array(embeddings, dtype=np.float64)
    np.save(cache_path, arr)
    print(f"[screen_embedding_models] wrote {cache_path} shape={arr.shape}", file=sys.stderr)
    return arr


def _cross_validate(embeddings: np.ndarray, labels: list[str], sample_weight: list[float]) -> dict[str, float]:
    """5-fold StratifiedKFold CV accuracy / macro-F1 for a single embedding-model candidate.

    Mirrors train_domain_classifier.train_classifier's base estimator
    (LogisticRegression(max_iter=1000, class_weight=None) + sample_weight)
    but without CalibratedClassifierCV, since model selection only needs
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


async def _run(train_data_path: str, ollama_host: str, ollama_port: int, exclude: list[str]) -> dict:
    """Score all _CANDIDATES with 5-fold CV and apply the Iter80 selection rule.

    `exclude` lists candidates that failed the separate, offline G0 VRAM gate
    (measured via `ollama ps` / `nvidia-smi` on wafl-ctrl5, not by this
    script) and are therefore ineligible for selection even if their CV
    accuracy is highest. All candidates are still scored so their CV numbers
    are visible in the output regardless of G0 outcome.
    """
    rows = _load_training_rows(train_data_path)
    labels = [row["domain"] for row in rows]
    sample_weight = _extract_sample_weights(rows)
    ollama_client = OllamaClient(host=f"http://{ollama_host}:{ollama_port}")

    results: dict[str, dict] = {}
    for model_name in _CANDIDATES:
        cache_path = _cache_path(train_data_path, model_name)
        embeddings = await _embed_candidate(ollama_client, model_name, rows, cache_path)
        results[model_name] = _cross_validate(embeddings, labels, sample_weight)

    # Selection rule (journal.md Iter80 plan): among non-baseline candidates
    # that passed G0, pick max CV accuracy; ties broken by macro-F1, then by
    # preferring qwen3-embedding:4b. The baseline is never itself "selected"
    # here (selecting it would mean no change, i.e. the lever is exhausted).
    eligible = [name for name in _CANDIDATES if name != _BASELINE_CANDIDATE and name not in exclude]

    def _sort_key(name: str) -> tuple[float, float, int]:
        r = results[name]
        prefer_4b = 1 if name == "qwen3-embedding:4b" else 0
        return (r["cv_accuracy_mean"], r["cv_macro_f1_mean"], prefer_4b)

    selected_model = max(eligible, key=_sort_key) if eligible else None

    return {
        "baseline_cv_accuracy": results[_BASELINE_CANDIDATE]["cv_accuracy_mean"],
        "excluded_by_g0": exclude,
        "selected_model": selected_model,
        "candidates": results,
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Iter80 G1: offline CV screening of embedding-model candidates "
        "on data/classifier_train.jsonl only (no dataset.jsonl access)."
    )
    parser.add_argument("--train-data", default="data/classifier_train.jsonl")
    parser.add_argument(
        "--ollama-host",
        default="127.0.0.1",
        help="wafl-ctrl5 tunnel endpoint (2026-09-19 operational rule: never wafl500-509)",
    )
    parser.add_argument("--ollama-port", type=int, default=11499)
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Candidate names that failed the offline G0 VRAM gate and must not be selected "
        "even if their CV accuracy is highest (e.g. --exclude qwen3-embedding:4b).",
    )
    args = parser.parse_args()

    result = asyncio.run(_run(args.train_data, args.ollama_host, args.ollama_port, args.exclude))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
