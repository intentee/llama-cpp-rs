---
name: coverage
description: Find uncovered lines in the llvm-cov coverage report. Use when checking test coverage, finding gaps, or working toward 100% coverage.
allowed-tools: Bash, Read, Edit, Write, Grep, Glob
---

Find uncovered lines in the project coverage report.

## Steps

1. Run `make test.qwen3.5_0.8B.coverage.json` to collect coverage data and export JSON
2. Run `python3 scripts/coverage-uncovered-lines.py target/coverage.json` to list uncovered lines
3. Fix each uncovered line by adding tests or restructuring code
4. Repeat until `make test.qwen3.5_0.8B.coverage` passes (enforces 100% line coverage)

## Output format

The script prints a summary matching `--fail-under-lines` exactly, then every line with any uncovered region:

```
Lines: 7072  Covered: 7060  Missed: 12  (99.83%)

context/session.rs:201:             check_session_load_length(n_out, max_tokens)?;
model.rs:184:                 c_int::try_from(c_string.as_bytes().len())?,
```

## Key concepts

- **Missed Lines** in the summary is what `--fail-under-lines 100` checks
- **Lines with uncovered regions** (the detailed list) shows every line that has any uncovered code path — fixing all of these is what reaches 100%
- The `?` operator on a covered line creates an uncovered region (the error branch) even though the line itself executed
