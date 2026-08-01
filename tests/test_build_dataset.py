"""Tests for the JMMLU-backed, 10-domain evaluation dataset builder.

Uses tests/fixtures/jmmlu_sample.zip (synthetic placeholder content, not
real JMMLU questions) together with a reduced domain_task_map containing
exactly the one task per domain present in that fixture, so these tests
run fully offline with no network access and no copyrighted content.
"""

import io
import json
import zipfile
from pathlib import Path

from build_dataset import (
    _DOMAIN_TASK_MAP,
    _DOMAIN_TARGET_SIZE,
    _EDUCATION_PROXY_TASK_TRAIN_TARGET_SIZES,
    _RESTRICTED_LICENSE_TASKS,
    _build_rows,
    _classifier_task_sample_weight,
    _ensure_parent_dir,
    _sample_domain_questions,
    build_classifier_training_rows,
    write_dataset,
)

_FIXTURE_ZIP = str(Path(__file__).parent / "fixtures" / "jmmlu_sample.zip")

# Mirrors _DOMAIN_TASK_MAP's domain set, but each domain maps to only the
# single representative task actually present in the fixture zip.
_FIXTURE_DOMAIN_TASK_MAP: dict[str, list[str]] = {
    "medical": ["anatomy"],
    "legal": ["international_law"],
    "education": ["sociology"],
    "business_economics": ["marketing"],
    "computer_science": ["computer_security"],
    "natural_science": ["astronomy"],
    "mathematics": ["elementary_mathematics"],
    "history_culture": ["japanese_history"],
    "social_science": ["philosophy"],
    "general": ["miscellaneous"],
}

_TEN_DOMAINS = frozenset(_FIXTURE_DOMAIN_TASK_MAP)


def _write_fixture_dataset(**overrides: object) -> list[dict]:
    """Run write_dataset against the fixture zip/domain map and return parsed rows."""
    buffer = io.StringIO()
    write_dataset(
        buffer,
        jmmlu_zip_path=_FIXTURE_ZIP,
        domain_task_map=_FIXTURE_DOMAIN_TASK_MAP,
        **overrides,
    )
    return [json.loads(line) for line in buffer.getvalue().strip().split("\n")]


def test_write_dataset_produces_one_json_object_per_line() -> None:
    """Every line is valid, independently parsable JSON."""
    buffer = io.StringIO()
    count = write_dataset(
        buffer, jmmlu_zip_path=_FIXTURE_ZIP, domain_task_map=_FIXTURE_DOMAIN_TASK_MAP
    )

    lines = buffer.getvalue().strip().split("\n")
    assert len(lines) == count
    for line in lines:
        json.loads(line)


def test_write_dataset_rows_have_required_fields() -> None:
    """Each row carries a unique id, the query text, and expected_domains."""
    rows = _write_fixture_dataset()

    ids = [row["id"] for row in rows]
    assert len(ids) == len(set(ids)), "dataset ids must be unique"
    for row in rows:
        assert row["query"]
        assert isinstance(row["expected_domains"], list)
        assert len(row["expected_domains"]) >= 1


def test_write_dataset_includes_compound_domain_questions() -> None:
    """At least one row spans more than one domain (design doc 4.3 tier 2)."""
    rows = _write_fixture_dataset()

    compound_rows = [row for row in rows if row["is_compound"]]
    assert len(compound_rows) > 0
    for row in compound_rows:
        assert len(row["expected_domains"]) > 1


def test_write_dataset_covers_all_ten_configured_domains() -> None:
    """The dataset includes single-domain questions for all 10 mesh domains."""
    rows = _write_fixture_dataset()

    single_domain_labels = {row["expected_domains"][0] for row in rows if not row["is_compound"]}
    assert single_domain_labels == _TEN_DOMAINS


def test_write_dataset_maps_jmmlu_tasks_to_expected_domain() -> None:
    """Each single-domain row's jmmlu_task belongs to the domain it was assigned to."""
    rows = _write_fixture_dataset()

    for row in rows:
        if row["is_compound"]:
            continue
        domain = row["expected_domains"][0]
        assert row["jmmlu_task"] in _FIXTURE_DOMAIN_TASK_MAP[domain]
        assert row["jmmlu_answer"] in {"A", "B", "C", "D"}


def test_write_dataset_respects_domain_target_size_cap() -> None:
    """domain_target_size caps sampled rows per domain at the domain's pool size."""
    rows = _write_fixture_dataset(domain_target_size=2)

    per_domain_counts: dict[str, int] = {}
    for row in rows:
        if row["is_compound"]:
            continue
        domain = row["expected_domains"][0]
        per_domain_counts[domain] = per_domain_counts.get(domain, 0) + 1
    for count in per_domain_counts.values():
        assert count <= 2


def test_write_dataset_excludes_restricted_license_tasks_when_flagged() -> None:
    """--exclude-restricted-license-tasks drops japanese_history from history_culture."""
    rows = _write_fixture_dataset(exclude_restricted_license_tasks=True)

    history_rows = [
        row
        for row in rows
        if not row["is_compound"] and row["expected_domains"][0] == "history_culture"
    ]
    assert history_rows == []
    assert "japanese_history" in _RESTRICTED_LICENSE_TASKS


def test_write_dataset_keeps_hand_authored_compound_tier() -> None:
    """Compound rows are hand-authored cross-domain pairs, not JMMLU-derived.

    2026-07-30 (research_frontier item 2 / d0003 X4): expanded from 3 pairs
    (medical/legal/education only) to 43 pairs spanning all 10 domains, so
    this allow-list is now the full set of pairs _COMPOUND_QUESTIONS uses
    rather than just the original medical-heavy trio.
    """
    rows = _write_fixture_dataset()

    compound_domain_pairs = {
        tuple(sorted(row["expected_domains"])) for row in rows if row["is_compound"]
    }
    assert compound_domain_pairs <= {
        ("legal", "medical"),
        ("education", "medical"),
        ("education", "legal"),
        ("business_economics", "legal"),
        ("business_economics", "computer_science"),
        ("business_economics", "medical"),
        ("business_economics", "natural_science"),
        ("business_economics", "mathematics"),
        ("business_economics", "history_culture"),
        ("business_economics", "social_science"),
        ("business_economics", "education"),
        ("business_economics", "general"),
        ("computer_science", "legal"),
        ("computer_science", "medical"),
        ("computer_science", "natural_science"),
        ("computer_science", "mathematics"),
        ("computer_science", "history_culture"),
        ("computer_science", "social_science"),
        ("computer_science", "education"),
        ("computer_science", "general"),
        ("legal", "natural_science"),
        ("medical", "natural_science"),
        ("mathematics", "natural_science"),
        ("history_culture", "natural_science"),
        ("natural_science", "social_science"),
        ("education", "natural_science"),
        ("general", "natural_science"),
        ("legal", "mathematics"),
        ("mathematics", "medical"),
        ("history_culture", "mathematics"),
        ("mathematics", "social_science"),
        ("education", "mathematics"),
        ("general", "mathematics"),
        ("history_culture", "legal"),
        ("history_culture", "medical"),
        ("history_culture", "social_science"),
        ("education", "history_culture"),
        ("general", "history_culture"),
        ("legal", "social_science"),
        ("medical", "social_science"),
        ("education", "social_science"),
        ("general", "social_science"),
        ("general", "legal"),
    }
    for row in rows:
        if row["is_compound"]:
            assert "jmmlu_task" not in row


def test_build_classifier_training_rows_never_overlaps_eval_queries() -> None:
    """E6 label-leakage regression test: training and evaluation questions are always disjoint.

    Guards against Iter10's failure mode (training features derived from
    the same questions used for evaluation) at the level of the dataset
    itself, not just at analysis time.
    """
    eval_rows = _build_rows(
        _FIXTURE_ZIP,
        domain_target_size=1,
        exclude_restricted_license_tasks=False,
        domain_task_map=_FIXTURE_DOMAIN_TASK_MAP,
    )
    train_rows = build_classifier_training_rows(
        _FIXTURE_ZIP,
        domain_target_size=1,
        exclude_restricted_license_tasks=False,
        domain_task_map=_FIXTURE_DOMAIN_TASK_MAP,
        eval_rows=eval_rows,
    )

    eval_queries = {row["query"] for row in eval_rows if not row["is_compound"]}
    train_queries = {row["query"] for row in train_rows}
    assert eval_queries & train_queries == set()
    assert len(train_rows) > 0


def test_build_classifier_training_rows_have_query_domain_and_sample_weight_only() -> None:
    """Training rows carry only {id, query, domain, sample_weight} — no probe/dispatch-derived fields.

    sample_weight (Iter32) is computed deterministically from the JMMLU task
    name at generation time (see _classifier_task_sample_weight); it is
    unrelated to Iter10's label leakage (probe/dispatch-derived features
    such as self_confidence, margin, is_top1 evaluated on the same
    questions used for online testing), which this test guards against by
    construction (the field set below has no room for such fields).
    """
    eval_rows = _write_fixture_dataset(domain_target_size=1)
    train_rows = build_classifier_training_rows(
        _FIXTURE_ZIP,
        domain_target_size=1,
        exclude_restricted_license_tasks=False,
        domain_task_map=_FIXTURE_DOMAIN_TASK_MAP,
        eval_rows=eval_rows,
    )
    for row in train_rows:
        assert set(row) == {"id", "query", "domain", "sample_weight"}
        assert row["domain"] in _TEN_DOMAINS
        assert isinstance(row["sample_weight"], float)


def test_classifier_task_sample_weight_defaults_all_tasks_to_one_after_iter32_revert() -> None:
    """After Iter32 revert, _CLASSIFIER_TASK_SAMPLE_WEIGHTS is empty, so all tasks
    (including the former weak proxy tasks) default to 1.0. Iter33 replaces the
    sample_weight mechanism with education_proxy_task_resampling at the data
    extraction stage."""
    assert _classifier_task_sample_weight("high_school_psychology") == 1.0
    assert _classifier_task_sample_weight("moral_disputes") == 1.0
    assert _classifier_task_sample_weight("sociology") == 1.0
    assert _classifier_task_sample_weight("anatomy") == 1.0


def test_ensure_parent_dir_creates_missing_directory(tmp_path) -> None:
    """A clean checkout has no data/ directory (it is gitignored); main() must create it,
    not crash with FileNotFoundError when opening --output for the first time."""
    target = tmp_path / "data" / "dataset.jsonl"
    assert not target.parent.exists()

    _ensure_parent_dir(str(target))

    assert target.parent.is_dir()


def test_ensure_parent_dir_is_a_no_op_for_a_bare_filename() -> None:
    """A path with no directory component (e.g. writing to the cwd) must not raise."""
    _ensure_parent_dir("dataset.jsonl")


def test_sample_domain_questions_with_task_target_sizes_respects_per_task_targets() -> None:
    """When task_target_sizes is provided, each task is sampled independently with its
    own target size, capped at the pool size."""
    with zipfile.ZipFile(_FIXTURE_ZIP) as zf:
        # Use two tasks from the fixture: sociology (education) and anatomy (medical)
        # as a stand-in for a "2-task domain"
        task_names = ["sociology", "anatomy"]
        target_sizes = {"sociology": 1, "anatomy": 2}
        # Extra keys in target_sizes that are not in task_names should be ignored
        target_sizes["moral_disputes"] = 10

        result = _sample_domain_questions(
            zf,
            task_names=task_names,
            target_size=100,
            seed=42,
            exclude_tasks=frozenset(),
            exclude_queries=frozenset(),
            task_target_sizes=target_sizes,
        )

        soc_count = sum(1 for _q, _a, t in result if t == "sociology")
        anat_count = sum(1 for _q, _a, t in result if t == "anatomy")
        assert soc_count == 1
        assert anat_count == 2
        assert len(result) == 3


def test_sample_domain_questions_without_task_target_sizes_preserves_existing_behavior() -> None:
    """When task_target_sizes is None (default), the function pools all tasks and
    samples once, exactly as before Iter33."""
    with zipfile.ZipFile(_FIXTURE_ZIP) as zf:
        result = _sample_domain_questions(
            zf,
            task_names=["sociology", "anatomy"],
            target_size=5,
            seed=42,
            exclude_tasks=frozenset(),
            exclude_queries=frozenset(),
            task_target_sizes=None,
        )

        assert len(result) == 5
        # All items should be from one of the two tasks
        tasks_seen = {t for _q, _a, t in result}
        assert tasks_seen <= {"sociology", "anatomy"}


def test_education_proxy_task_train_target_sizes_static_integrity() -> None:
    """Static integrity: _EDUCATION_PROXY_TASK_TRAIN_TARGET_SIZES keys must match
    _DOMAIN_TASK_MAP['education'] and values must sum to _DOMAIN_TARGET_SIZE (150)."""
    education_tasks = set(_DOMAIN_TASK_MAP["education"])
    target_keys = set(_EDUCATION_PROXY_TASK_TRAIN_TARGET_SIZES.keys())
    assert target_keys == education_tasks, (
        f"Keys mismatch: {target_keys} vs {education_tasks}"
    )
    assert sum(_EDUCATION_PROXY_TASK_TRAIN_TARGET_SIZES.values()) == _DOMAIN_TARGET_SIZE
