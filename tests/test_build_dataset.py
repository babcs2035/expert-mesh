"""Tests for the JMMLU-backed, 10-domain evaluation dataset builder.

Uses tests/fixtures/jmmlu_sample.zip (synthetic placeholder content, not
real JMMLU questions) together with a reduced domain_task_map containing
exactly the one task per domain present in that fixture, so these tests
run fully offline with no network access and no copyrighted content.
"""

import io
import json
from pathlib import Path

from build_dataset import (
    _RESTRICTED_LICENSE_TASKS,
    _build_rows,
    _ensure_parent_dir,
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
    """Compound rows are the hand-authored medical/legal/education combinations, not JMMLU-derived."""
    rows = _write_fixture_dataset()

    compound_domain_pairs = {
        tuple(sorted(row["expected_domains"])) for row in rows if row["is_compound"]
    }
    assert compound_domain_pairs <= {
        ("legal", "medical"),
        ("education", "medical"),
        ("education", "legal"),
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


def test_build_classifier_training_rows_have_query_and_domain_only() -> None:
    """Training rows carry only {id, query, domain} — no probe/dispatch-derived fields."""
    eval_rows = _write_fixture_dataset(domain_target_size=1)
    train_rows = build_classifier_training_rows(
        _FIXTURE_ZIP,
        domain_target_size=1,
        exclude_restricted_license_tasks=False,
        domain_task_map=_FIXTURE_DOMAIN_TASK_MAP,
        eval_rows=eval_rows,
    )
    for row in train_rows:
        assert set(row) == {"id", "query", "domain"}
        assert row["domain"] in _TEN_DOMAINS


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
