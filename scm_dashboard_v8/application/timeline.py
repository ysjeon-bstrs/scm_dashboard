"""Timeline application layer delegates to the v5 pipeline implementation."""

from scm_dashboard_v5.pipeline import BuildInputs as _BuildInputs
from scm_dashboard_v5.pipeline import build_timeline_bundle as _build_timeline_bundle

BuildInputs = _BuildInputs
build_timeline_bundle = _build_timeline_bundle

__all__ = ["BuildInputs", "build_timeline_bundle"]
