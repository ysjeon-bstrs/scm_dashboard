# SCM Dashboard

This repository contains a collection of Streamlit helpers, analytics layers, and
supporting pipelines that model the supply-chain management dashboard.  The
codebase hosts multiple generations of the timeline builder (v4–v8) and a suite
of regression tests that keep the newer pipelines aligned with the historical
behaviour.

## Running the tests

The project uses `pytest` for automated checks.  The end-to-end regression
targets can be executed with:

```bash
pytest tests/e2e
```

## Golden fixtures

The end-to-end regression in `tests/e2e/test_v8_equivalence.py` relies on
precomputed CSV fixtures stored under `tests/golden/`.  The fixtures are derived
from the v5 orchestration layer (which proxies to the v8 application pipeline)
using the deterministic sample dataset shipped with the tests.

Whenever the underlying pipeline logic changes, regenerate the goldens before
committing:

```bash
python -m tests.e2e.update_golden
```

The helper normalises timestamps, sorts rows, and rounds numeric columns so that
the comparisons remain stable.  After regenerating, run `pytest` to ensure the
new outputs are reflected in the equivalence checks.

