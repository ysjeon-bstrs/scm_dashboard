from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import streamlit as st


DATA_SESSION_KEY = "v5_data"
SOURCE_SESSION_KEY = "_v5_source"
SNAPSHOT_RAW_SESSION_KEY = "_snapshot_raw_cache"


@dataclass
class LoadedData:
    """Container for the normalized data stored in the Streamlit session."""

    moves: pd.DataFrame
    snapshot: pd.DataFrame
    snapshot_raw: Optional[pd.DataFrame] = None
    wip: Optional[pd.DataFrame] = None


def get_loaded_data() -> Optional["LoadedData"]:
    """Return the previously cached data bundle if present."""

    data = st.session_state.get(DATA_SESSION_KEY)
    if isinstance(data, LoadedData):
        return data
    return None


def set_loaded_data(data: LoadedData, source_label: str) -> None:
    """Persist the loaded data and metadata in the Streamlit session."""

    st.session_state[DATA_SESSION_KEY] = data
    st.session_state[SOURCE_SESSION_KEY] = source_label
    if data.snapshot_raw is not None:
        st.session_state[SNAPSHOT_RAW_SESSION_KEY] = data.snapshot_raw


def get_source_label() -> Optional[str]:
    """Return the current data source identifier if available."""

    label = st.session_state.get(SOURCE_SESSION_KEY)
    return str(label) if label is not None else None
