"""scm_dashboard_v8 패키지는 Streamlit 의존성을 최소화한 안정형 모듈 구조를 제공합니다."""

from .application.timeline import BuildInputs, build_timeline_bundle

__all__ = ["BuildInputs", "build_timeline_bundle"]
