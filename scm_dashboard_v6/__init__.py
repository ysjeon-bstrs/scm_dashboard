"""SCM dashboard v6 package exposing the modularised application surface."""

from .application.timeline import build_timeline_bundle, BuildInputs

__all__ = ["BuildInputs", "build_timeline_bundle"]
