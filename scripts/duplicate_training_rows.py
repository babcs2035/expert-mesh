"""Iter66 (multilabel_training_mixture_ratio=single_domain_rows_duplicated_x2):
duplicate every line of an input JSONL file adjacently (interleave), so that
row occurrence counts double while the underlying content set is unchanged.

Adjacent duplication (line1, line1, line2, line2, ...) is used instead of
block concatenation (all originals, then all copies again) because
CalibratedClassifierCV(cv=5) uses StratifiedKFold(shuffle=False) internally:
under block concatenation, a duplicate pair is almost always split across two
different folds (one side lands in the calibration holdout), which optimistically
biases the calibration; under adjacent duplication, a duplicate pair stays in
the same fold with near certainty. See journal.md Iter66 plan point 1.

`id` fields are NOT rewritten -- duplicated rows keep identical `id`, `query`,
and `domain` values, so a duplicate-collapsed set of output lines equals the
input line set exactly (verifiable via `sort -u`).

Usage:
    uv run python -m scripts.duplicate_training_rows \\
        --input data/classifier_train.jsonl \\
        --output data/classifier_train_iter66_x2.jsonl
"""

import argparse


def duplicate_lines_adjacent(input_path: str, output_path: str) -> int:
    """Read `input_path` line by line and write each line twice, adjacently,
    to `output_path`. Returns the number of output lines written.
    """
    lines_written = 0
    with open(input_path, encoding="utf-8") as infile, open(
        output_path, "w", encoding="utf-8"
    ) as outfile:
        for line in infile:
            stripped = line.rstrip("\n")
            if not stripped:
                continue
            outfile.write(stripped + "\n")
            outfile.write(stripped + "\n")
            lines_written += 2
    return lines_written


def main() -> None:
    """CLI entry point: duplicate --input's rows adjacently into --output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input JSONL path")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    args = parser.parse_args()

    count = duplicate_lines_adjacent(args.input, args.output)
    print(f"Wrote {count} lines (adjacent duplication) to {args.output}")


if __name__ == "__main__":
    main()
