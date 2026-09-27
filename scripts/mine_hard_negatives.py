"""Iter84 (cross_domain_training_data_augmentation=hard_negative_mining_all_domains):
mine JMMLU rows the current deployed classifier finds hardest, for each of the 10
domains, and append them to data/classifier_train.jsonl to produce a new training
file (never mutates classifier_train.jsonl itself).

Selection regime (10 domains share one rule; no per-domain parameters, journal.md
"Iteration 84" 計画節 参照):

1. Pool: for each domain, take only the JMMLU tasks (build_dataset.py's
   `_DOMAIN_TASK_MAP`) that are already represented in the CURRENT training set
   for that domain (determined empirically here by matching each training row's
   query text against every candidate task's CSV rows -- classifier_train.jsonl
   does not itself carry a jmmlu_task field). All rows of those tasks, minus the
   evaluation set's 1,915 question texts and the training set's own 1,427
   question texts, form the domain's pool. This keeps `education`'s pool limited
   to its historically-used 3 proxy tasks (sociology / high_school_psychology /
   moral_disputes), excluding japanese_civics -- mixing that in would reintroduce
   the Iter36/37/38 task-composition change this iteration deliberately holds
   fixed (see module docstring cross-reference in journal.md).
2. Difficulty score: the CURRENT deployed classifier artifact's predict_proba
   for the row's OWN (correct) domain -- the smaller this "p_true", the harder
   the row (least-confidence sampling). Embeddings are produced by the same
   `expert_backend.embed_query_views(..., instruction=..., concat_views=True)`
   call used by both scripts/train_domain_classifier.py (training) and node.py
   (runtime), so this script cannot introduce an Iter36-style train/eval
   embedding mismatch.
3. Selection: per domain, the N hardest rows (ascending p_true; ties broken by
   (jmmlu_task, query) lexicographic order for determinism). Domains whose pool
   is smaller than N (only `legal`, whose pool is empirically 0) contribute
   fewer rows -- this is a structural consequence of JMMLU's task coverage, not
   a per-domain parameter.
4. Output: the selected rows are appended, in sorted-domain order, after an
   exact byte-for-byte copy of --train-data. Each new row's id is
   "{domain}-hardneg-{seq:03d}"; no sample_weight field is added (matching
   train_domain_classifier.py's `_extract_sample_weights()`, which computes
   domain-balanced weights automatically from domain row counts and does not
   require or use a sample_weight column).

Embeddings for the pool are cached to
data/embcache_pool_<model>.npy (the "plain" / no-instruction view) and
data/embcache_pool_<model>__p1.npy (the "instructed" view), mirroring the
existing data/embcache_eval_*.npy / data/embcache_qwen3-embedding_0.6b__p1.npy
naming already used for the evaluation set and CV screening caches. A small
JSON sidecar (same path with .meta.json appended) records a hash of the pool's
query texts in cache order, so a stale or reordered pool can never silently
reuse a mismatched cache. If a cache exists but is shorter than the current
pool (e.g. from an interrupted prior run) it is treated as a valid prefix and
only the remaining rows are embedded.

Usage (module mode; run against wafl-ctrl5 via the SSH tunnel per the
2026-09-19 operational rule -- this script must NOT be pointed at wafl500-509):
    ssh -fNT -L 11499:localhost:11434 wafl-ctrl5
    uv run python -m scripts.mine_hard_negatives \\
        --train-data data/classifier_train.jsonl \\
        --eval-data data/dataset.jsonl \\
        --classifier-model models/domain_classifier.joblib \\
        --per-domain 100 \\
        --embedding-model qwen3-embedding:0.6b \\
        --embedding-instruction "Given a user question, identify the single academic or professional domain it belongs to" \\
        --embedding-view-concat \\
        --ollama-host 127.0.0.1 --ollama-port 11499 \\
        --output data/classifier_train_iter84_hardneg.jsonl
"""

import argparse
import asyncio
import hashlib
import io
import json
import os
import sys
import zipfile
from collections import Counter, defaultdict

import joblib
import numpy as np

from build_dataset import (
    _DOMAIN_TASK_MAP,
    _format_jmmlu_query,
    _load_jmmlu_zip_bytes,
    _parse_jmmlu_task_csv,
)
from expert_backend import OllamaClient, embed_query_views

_CHECKPOINT_EVERY = 200


def _load_jsonl(path: str) -> list[dict]:
    """Read a JSONL file into a list of dicts, skipping blank lines."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _train_queries_by_domain(train_rows: list[dict]) -> dict[str, set[str]]:
    """Group the current training set's question texts by domain."""
    by_domain: dict[str, set[str]] = defaultdict(set)
    for row in train_rows:
        by_domain[row["domain"]].add(row["query"])
    return by_domain


def determine_used_tasks(
    zf: zipfile.ZipFile,
    train_queries_by_domain: dict[str, set[str]],
) -> dict[str, list[str]]:
    """For each domain, return the subset of _DOMAIN_TASK_MAP tasks whose CSV rows
    overlap with that domain's current training-set question texts.

    classifier_train.jsonl carries no jmmlu_task field, so provenance is
    recovered by matching formatted question text against each candidate
    task's CSV rows. Task order is preserved from _DOMAIN_TASK_MAP for
    determinism.
    """
    used: dict[str, list[str]] = {}
    for domain, tasks in _DOMAIN_TASK_MAP.items():
        domain_queries = train_queries_by_domain.get(domain, set())
        used_tasks = []
        for task in tasks:
            task_queries = {
                _format_jmmlu_query(row) for row in _parse_jmmlu_task_csv(zf, task)
            }
            if domain_queries & task_queries:
                used_tasks.append(task)
        used[domain] = used_tasks
    return used


def build_pool(
    zf: zipfile.ZipFile,
    used_tasks: dict[str, list[str]],
    exclude_queries: frozenset[str],
) -> list[dict]:
    """Return {domain, task, query, answer} rows for every domain's pool, excluding
    exclude_queries (the union of the eval set's and the current training set's
    question texts) and any duplicate query text already added to the pool.
    """
    pool: list[dict] = []
    seen: set[str] = set()
    for domain in sorted(used_tasks):
        for task in used_tasks[domain]:
            for row in _parse_jmmlu_task_csv(zf, task):
                query = _format_jmmlu_query(row)
                if query in exclude_queries or query in seen:
                    continue
                seen.add(query)
                pool.append(
                    {"domain": domain, "task": task, "query": query, "answer": row["answer"]}
                )
    return pool


def _pool_hash(pool: list[dict]) -> str:
    """Order-sensitive digest of the pool's query texts, used to validate a cache."""
    hasher = hashlib.sha256()
    for row in pool:
        hasher.update(row["query"].encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _load_cache_prefix(
    plain_path: str, instructed_path: str, meta_path: str, pool_hash: str
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return (plain, instructed) arrays representing a valid prefix of the pool's
    embeddings, or (None, None) if no usable cache exists.
    """
    if not (os.path.exists(plain_path) and os.path.exists(instructed_path) and os.path.exists(meta_path)):
        return None, None
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("pool_hash") != pool_hash:
        print(
            f"[mine_hard_negatives] cache pool_hash mismatch, ignoring stale cache: {meta_path}",
            file=sys.stderr,
        )
        return None, None
    plain = np.load(plain_path)
    instructed = np.load(instructed_path)
    if plain.shape[0] != instructed.shape[0]:
        print(
            "[mine_hard_negatives] cache row-count mismatch between plain/instructed views, ignoring",
            file=sys.stderr,
        )
        return None, None
    return plain, instructed


def _save_cache(
    plain_path: str, instructed_path: str, meta_path: str, pool_hash: str,
    plain: np.ndarray, instructed: np.ndarray, n_total: int,
) -> None:
    """Persist the current (possibly partial) embedding progress to disk."""
    np.save(plain_path, plain)
    np.save(instructed_path, instructed)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"pool_hash": pool_hash, "n_done": int(plain.shape[0]), "n_total": n_total}, f)


async def embed_pool(
    ollama_client: OllamaClient,
    embedding_model: str,
    pool: list[dict],
    instruction: str,
    plain_path: str,
    instructed_path: str,
    meta_path: str,
) -> np.ndarray:
    """Return the pool's (n_pool, 2048) concatenated [plain, instructed] feature matrix.

    Embeds via expert_backend.embed_query_views(concat_views=True) (the exact
    function scripts/train_domain_classifier.py and node.py use), then splits
    the returned 2048-dim vector into its two constituent 1024-dim views so
    each can be cached under the existing embcache_*/embcache_*__p1 naming
    convention. Resumable: an existing cache shorter than len(pool) is treated
    as a valid prefix (validated against pool_hash) and only the remaining
    rows are embedded.
    """
    pool_hash = _pool_hash(pool)
    cached_plain, cached_instructed = _load_cache_prefix(plain_path, instructed_path, meta_path, pool_hash)
    plain_rows: list[list[float]] = cached_plain.tolist() if cached_plain is not None else []
    instructed_rows: list[list[float]] = cached_instructed.tolist() if cached_instructed is not None else []
    start = len(plain_rows)
    if start >= len(pool):
        print(f"[mine_hard_negatives] reusing complete cache ({start} rows)", file=sys.stderr)
    else:
        print(
            f"[mine_hard_negatives] embedding pool rows {start}..{len(pool)} "
            f"(resuming from cache)" if start else f"[mine_hard_negatives] embedding {len(pool)} pool rows",
            file=sys.stderr,
        )
        for i in range(start, len(pool)):
            vec = await embed_query_views(
                ollama_client, embedding_model, pool[i]["query"],
                instruction=instruction, concat_views=True,
            )
            half = len(vec) // 2
            plain_rows.append(vec[:half])
            instructed_rows.append(vec[half:])
            done = i + 1
            if done % _CHECKPOINT_EVERY == 0 or done == len(pool):
                _save_cache(
                    plain_path, instructed_path, meta_path, pool_hash,
                    np.array(plain_rows, dtype=np.float64), np.array(instructed_rows, dtype=np.float64),
                    len(pool),
                )
                print(f"[mine_hard_negatives] embedded {done}/{len(pool)}", file=sys.stderr)
    plain = np.array(plain_rows, dtype=np.float64)
    instructed = np.array(instructed_rows, dtype=np.float64)
    return np.hstack([plain, instructed])


def select_hard_negatives(
    pool: list[dict], p_true: np.ndarray, per_domain: int
) -> list[dict]:
    """Return the per_domain hardest (lowest p_true) rows for each domain, sorted by
    domain name, with deterministic (p_true, task, query) ordering within a domain.
    """
    by_domain: dict[str, list[tuple[float, dict]]] = defaultdict(list)
    for row, p in zip(pool, p_true):
        by_domain[row["domain"]].append((float(p), row))

    selected: list[dict] = []
    for domain in sorted(by_domain):
        ranked = sorted(by_domain[domain], key=lambda t: (t[0], t[1]["task"], t[1]["query"]))
        chosen = ranked[:per_domain]
        for seq, (p, row) in enumerate(chosen, start=1):
            selected.append(
                {
                    "id": f"{domain}-hardneg-{seq:03d}",
                    "query": row["query"],
                    "domain": domain,
                    "_p_true": p,  # dropped before writing; kept here for the summary report
                }
            )
    return selected


def write_output(train_data_path: str, selected: list[dict], output_path: str) -> None:
    """Copy --train-data byte-for-byte, then append the selected rows as JSONL."""
    with open(train_data_path, "rb") as f:
        existing_bytes = f.read()
    with open(output_path, "wb") as out:
        out.write(existing_bytes)
        if existing_bytes and not existing_bytes.endswith(b"\n"):
            out.write(b"\n")
        for row in selected:
            clean_row = {"id": row["id"], "query": row["query"], "domain": row["domain"]}
            out.write(json.dumps(clean_row, ensure_ascii=False).encode("utf-8") + b"\n")


async def _run(args: argparse.Namespace) -> None:
    eval_rows = _load_jsonl(args.eval_data)
    train_rows = _load_jsonl(args.train_data)
    eval_queries = frozenset(row["query"] for row in eval_rows)
    train_queries = frozenset(row["query"] for row in train_rows)
    train_queries_by_domain = _train_queries_by_domain(train_rows)

    zip_bytes = _load_jmmlu_zip_bytes(args.jmmlu_zip)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        used_tasks = determine_used_tasks(zf, train_queries_by_domain)
        pool = build_pool(zf, used_tasks, eval_queries | train_queries)

    pool_counts = Counter(row["domain"] for row in pool)
    print(f"[mine_hard_negatives] used_tasks={used_tasks}", file=sys.stderr)
    print(f"[mine_hard_negatives] pool sizes: {dict(sorted(pool_counts.items()))}", file=sys.stderr)

    safe_model = args.embedding_model.replace(":", "_").replace("/", "_")
    plain_path = os.path.join(os.path.dirname(args.output) or ".", f"embcache_pool_{safe_model}.npy")
    instructed_path = os.path.join(os.path.dirname(args.output) or ".", f"embcache_pool_{safe_model}__p1.npy")
    meta_path = plain_path + ".meta.json"

    ollama_client = OllamaClient(host=f"http://{args.ollama_host}:{args.ollama_port}")
    features = await embed_pool(
        ollama_client, args.embedding_model, pool, args.embedding_instruction,
        plain_path, instructed_path, meta_path,
    )

    model = joblib.load(args.classifier_model)
    classes = list(model.classes_)
    class_index = {c: i for i, c in enumerate(classes)}
    proba = model.predict_proba(features)
    p_true = np.array([proba[i, class_index[row["domain"]]] for i, row in enumerate(pool)])

    selected = select_hard_negatives(pool, p_true, args.per_domain)
    write_output(args.train_data, selected, args.output)

    selected_counts = Counter(row["domain"] for row in selected)
    print(f"[mine_hard_negatives] selected counts: {dict(sorted(selected_counts.items()))}", file=sys.stderr)
    print(
        f"[mine_hard_negatives] wrote {args.output} "
        f"({len(train_rows) + len(selected)} rows = {len(train_rows)} existing + {len(selected)} new)",
        file=sys.stderr,
    )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Mine JMMLU hard negatives (least-confidence) for all 10 domains "
        "and append them to a copy of --train-data"
    )
    parser.add_argument("--train-data", default="data/classifier_train.jsonl")
    parser.add_argument("--eval-data", default="data/dataset.jsonl")
    parser.add_argument("--classifier-model", default="models/domain_classifier.joblib")
    parser.add_argument("--per-domain", type=int, default=100)
    parser.add_argument("--jmmlu-zip", default=None, help="Local JMMLU.zip path; downloads the pinned commit if omitted")
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--embedding-instruction", required=True)
    parser.add_argument(
        "--embedding-view-concat", action="store_true",
        help="Must be set (kept as a flag for symmetry with train_domain_classifier.py's "
        "CLI; this script always concatenates the two views and errors if the flag is absent, "
        "since the deployed 2048-dim artifact requires it)",
    )
    parser.add_argument("--ollama-host", required=True)
    parser.add_argument("--ollama-port", type=int, default=11434)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not args.embedding_view_concat:
        parser.error("--embedding-view-concat is required (the deployed classifier expects 2048-dim input)")

    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
