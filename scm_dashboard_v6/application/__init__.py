"""Application-layer façade for the v6 SCM dashboard."""

from .timeline import BuildInputs, build_timeline_bundle

__all__ = ["BuildInputs", "build_timeline_bundle"]
