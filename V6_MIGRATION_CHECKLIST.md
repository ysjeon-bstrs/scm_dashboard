# V6 Migration Checklist (v5 → v6)

- Branching
  - [ ] Create feature branch `v6/refactor-modularization`
  - [ ] Freeze `v5_main.py` changes except bugfixes

- App entry/UI
  - [ ] New `scm_dashboard_v6/app/main.py` created
  - [ ] Move sidebar controls to `ui/controls.py`
  - [ ] Replace `use_container_width` -> `width='stretch'`

- Features separation
  - [ ] `features/timeline.py`: build + consumption + render step chart
  - [ ] `features/amazon.py`: context + sales/inventory chart
  - [ ] `features/inventory_view.py`: pivot + CSV + lot details

- Charts split
  - [ ] `ui/charts/amazon.py` from v5 `ui/charts.py`
  - [ ] `ui/charts/step.py` from v5 `ui/charts.py`
  - [ ] `ui/charts/utils.py` for colors/safe guards

- Data + domain
  - [ ] `data/loaders.py` (gsheet/excel), `data/cache.py`, `data/config.py`
  - [ ] `domain/centers.py` (normalize), `domain/time.py` (range utils)

- Pipeline
  - [ ] `pipeline/orchestration.py` with single `build_timeline_bundle` API

- Forecast
  - [ ] `forecast/context.py` (AmazonForecastContext)
  - [ ] `forecast/sales_from_inventory.py`
  - [ ] `forecast/consumption.py` left as pure functions

- Adapters
  - [ ] `adapters/v4.py` wrapping v4 calls: kpi, inventory cost, processing, CENTER_COL

- Tests
  - [ ] Replace templates: `tests/test_*_feature_template.py` with real tests
  - [ ] Add unit tests for forecast trimming and inventory-to-sales mapping

- Deprecation cleanup
  - [ ] Remove cross-module private calls (e.g., `_timeline_inventory_matrix` from UI)
  - [ ] Ensure no direct v4 imports outside adapters

- Docs & CI
  - [ ] Update README with v6 structure and run steps
  - [ ] Update CI to run tests for v6 packages
