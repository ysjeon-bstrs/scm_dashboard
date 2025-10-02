"""Restructured SCM dashboard package with modular data pipeline."""

from .pipeline import BuildInputs, build_timeline_bundle

__all__ = ["BuildInputs", "build_timeline_bundle"]
