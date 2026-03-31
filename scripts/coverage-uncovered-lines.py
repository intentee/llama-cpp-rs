#!/usr/bin/env python3
"""Find uncovered lines from cargo-llvm-cov JSON export.

Usage:
    python3 scripts/coverage-uncovered-lines.py target/coverage.json

Matches the "Lines" metric used by `cargo llvm-cov --fail-under-lines`.

LLVM segments with has_count=True define code regions with an execution count.
Each such segment applies from (line, col) until the next has_count=True segment.
A line is "executable" if any such segment overlaps it.
A line is "covered" if the maximum count of all overlapping segments is > 0.
"""

import json
import os
import sys

SRC_MARKER = "llama-cpp-bindings/src/"


def compute_line_coverage(segments):
    """Walk segments and compute per-line max execution count."""
    line_max_count = {}

    # Filter to only segments with actual count data
    counted_segments = [
        (line, col, count)
        for line, col, count, has_count, _is_region, is_gap in segments
        if has_count and not is_gap
    ]

    for i, (seg_line, seg_col, count) in enumerate(counted_segments):
        # This segment's count applies from seg_line until the next counted segment
        if i + 1 < len(counted_segments):
            end_line = counted_segments[i + 1][0]
            end_col = counted_segments[i + 1][1]

            # If next segment starts at col 1, it owns that line
            if end_col <= 1:
                end_line -= 1
        else:
            end_line = seg_line

        for line in range(seg_line, max(end_line, seg_line) + 1):
            current_max = line_max_count.get(line, 0)
            line_max_count[line] = max(current_max, count)

    return line_max_count


def find_uncovered_lines(json_path):
    with open(json_path) as json_file:
        data = json.load(json_file)

    totals = data["data"][0]["totals"]["lines"]
    total_lines = totals["count"]
    covered_lines = totals["covered"]
    missed_lines = total_lines - covered_lines
    percent = totals["percent"]

    print(
        f"Lines: {total_lines}  Covered: {covered_lines}  "
        f"Missed: {missed_lines}  ({percent:.2f}%)"
    )
    print()

    uncovered_count = 0

    for file_data in data["data"][0]["files"]:
        filename = file_data["filename"]

        if SRC_MARKER not in filename:
            continue

        short_name = filename.split(SRC_MARKER, 1)[1]
        summary_lines = file_data["summary"]["lines"]

        if summary_lines["count"] == summary_lines["covered"]:
            continue

        segments = file_data.get("segments", [])

        if not segments:
            continue

        line_max_count = compute_line_coverage(segments)

        source_lines = {}

        if os.path.isfile(filename):
            with open(filename) as source_file:
                for line_number, line_text in enumerate(source_file, 1):
                    if line_number in line_max_count and line_max_count[line_number] == 0:
                        source_lines[line_number] = line_text.rstrip()

        for line_number in sorted(source_lines):
            code = source_lines[line_number]
            print(f"{short_name}:{line_number}: {code}")
            uncovered_count += 1

    print()
    print(f"Total uncovered lines: {uncovered_count}")


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python3 scripts/coverage-uncovered-lines.py <coverage.json>",
            file=sys.stderr,
        )
        sys.exit(1)

    find_uncovered_lines(sys.argv[1])


if __name__ == "__main__":
    main()
