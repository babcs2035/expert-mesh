"""Tests for the tier-2 evaluation dataset builder."""

import io
import json

from build_dataset import write_dataset


def test_write_dataset_produces_one_json_object_per_line() -> None:
    """Every line is valid, independently parsable JSON."""
    buffer = io.StringIO()
    count = write_dataset(buffer)

    lines = buffer.getvalue().strip().split("\n")
    assert len(lines) == count
    for line in lines:
        json.loads(line)


def test_write_dataset_rows_have_required_fields() -> None:
    """Each row carries a unique id, the query text, and expected_domains."""
    buffer = io.StringIO()
    write_dataset(buffer)

    rows = [json.loads(line) for line in buffer.getvalue().strip().split("\n")]
    ids = [row["id"] for row in rows]
    assert len(ids) == len(set(ids)), "dataset ids must be unique"
    for row in rows:
        assert row["query"]
        assert isinstance(row["expected_domains"], list)
        assert len(row["expected_domains"]) >= 1


def test_write_dataset_includes_compound_domain_questions() -> None:
    """At least one row spans more than one domain (design doc 4.3 tier 2)."""
    buffer = io.StringIO()
    write_dataset(buffer)

    rows = [json.loads(line) for line in buffer.getvalue().strip().split("\n")]
    compound_rows = [row for row in rows if row["is_compound"]]
    assert len(compound_rows) > 0
    for row in compound_rows:
        assert len(row["expected_domains"]) > 1


def test_write_dataset_covers_all_configured_domains() -> None:
    """The dataset includes single-domain questions for every node domain in config.yaml."""
    buffer = io.StringIO()
    write_dataset(buffer)

    rows = [json.loads(line) for line in buffer.getvalue().strip().split("\n")]
    single_domain_labels = {
        row["expected_domains"][0] for row in rows if not row["is_compound"]
    }
    assert single_domain_labels == {"medical", "legal", "general", "education"}
