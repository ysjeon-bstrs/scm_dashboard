"""
v6 앱 엔트리 래퍼 (레포 루트)

- Streamlit Cloud 등에서 레포 루트 기준으로 실행할 때
  패키지 임포트 경로 문제 없이 v6 엔트리를 호출하기 위한 얇은 래퍼.
"""

from __future__ import annotations

from scm_dashboard_v6.app.main import main


if __name__ == "__main__":
    main()


