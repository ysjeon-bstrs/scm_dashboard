"""Alias the v5 consumption module so patches remain effective across namespaces."""

from __future__ import annotations

import sys

from scm_dashboard_v5.forecast import consumption as _v5_consumption

sys.modules[__name__] = _v5_consumption
