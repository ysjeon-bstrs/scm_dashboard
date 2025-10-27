"""KPI 카드 반응형 스타일 모듈.

CSS Grid 기반 반응형 레이아웃 스타일을 제공합니다.
"""

from __future__ import annotations

import streamlit as st


def inject_responsive_styles() -> None:
    """Inject shared CSS styles for KPI cards (re-inject on each run)."""

    st.markdown(
        """
        <style>
        :root {
            --kpi-card-border: rgba(49, 51, 63, 0.2);
            --kpi-card-radius: 0.75rem;
            --kpi-card-padding: 0.85rem 1rem;
        }

        .kpi-card-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(var(--min-card-width, 280px), 1fr));
            gap: 0.75rem;
            align-items: stretch;
        }

        .kpi-sku-card {
            border: 1px solid var(--kpi-card-border);
            border-radius: var(--kpi-card-radius);
            padding: var(--kpi-card-padding);
            background-color: rgba(255, 255, 255, 0.8);
            box-shadow: 0 8px 16px rgba(49, 51, 63, 0.08);
            display: flex;
            flex-direction: column;
            gap: 0.9rem;
        }

        .kpi-sku-title {
            display: flex;
            flex-wrap: wrap;
            gap: 0.35rem 0.75rem;
            align-items: baseline;
            font-size: 1.1rem;
            font-weight: 600;
        }

        .kpi-sku-code {
            font-family: "SFMono-Regular", ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas,
                "Liberation Mono", "Courier New", monospace;
            font-size: 0.85rem;
            padding: 0.1rem 0.45rem;
            border-radius: 0.4rem;
            background-color: rgba(49, 51, 63, 0.08);
        }

        .kpi-section-title {
            font-weight: 600;
            margin-top: 0.25rem;
        }

        .kpi-metric-card {
            border: 1px solid var(--kpi-card-border);
            border-radius: 0.65rem;
            padding: 0.75rem 0.85rem;
            background: rgba(250, 250, 251, 0.9);
            display: flex;
            flex-direction: column;
            gap: 0.3rem;
            min-height: 100%;
            overflow: visible;
        }

        .kpi-metric-card--compact {
            padding: 0.6rem 0.75rem;
        }

        .kpi-metric-label {
            font-size: 0.85rem;
            color: rgba(49, 51, 63, 0.75);
            white-space: normal;
        }

        .kpi-metric-value {
            font-size: 1.25rem;
            font-weight: 700;
            white-space: nowrap;
            word-break: keep-all;
            overflow: visible;
        }

        .kpi-center-card {
            border: 1px solid var(--kpi-card-border);
            border-radius: 0.65rem;
            padding: 0.8rem 0.9rem;
            background-color: rgba(255, 255, 255, 0.85);
            display: flex;
            flex-direction: column;
            gap: 0.55rem;
        }

        .kpi-center-title {
            font-weight: 600;
            font-size: 1rem;
        }

        .kpi-grid--summary {
            --min-card-width: clamp(200px, 24vw, 260px);
        }

        .kpi-grid--centers {
            --min-card-width: clamp(220px, 24vw, 320px);
        }

        .kpi-grid--sku {
            --min-card-width: 320px;
        }

        .kpi-grid--compact {
            --min-card-width: 150px;
        }

        .kpi-grid--center-metrics {
            --min-card-width: clamp(140px, 28vw, 200px);
            align-items: stretch;
        }

        .kpi-grid--centers.kpi-grid--centers-narrow {
            --min-card-width: clamp(260px, 28vw, 340px);
        }

        .kpi-grid--centers.kpi-grid--centers-medium {
            --min-card-width: clamp(240px, 26vw, 320px);
        }

        .kpi-grid--centers.kpi-grid--centers-wide {
            --min-card-width: clamp(220px, 24vw, 300px);
        }

        .kpi-grid--centers.kpi-grid--centers-dense {
            --min-card-width: clamp(200px, 22vw, 280px);
        }

        @media (max-width: 1200px) {
            .kpi-card-grid {
                gap: 0.65rem;
            }

            .kpi-metric-value {
                font-size: 1.2rem;
            }
        }

        @media (max-width: 900px) {
            .kpi-grid--summary,
            .kpi-grid--centers,
            .kpi-grid--sku {
                --min-card-width: clamp(220px, 48vw, 320px);
            }

            .kpi-grid--center-metrics {
                --min-card-width: clamp(150px, 42vw, 200px);
            }

            .kpi-sku-card {
                padding: 0.75rem 0.85rem;
            }
        }

        @media (max-width: 700px) {
            .kpi-grid--summary,
            .kpi-grid--centers,
            .kpi-grid--sku {
                grid-template-columns: repeat(auto-fit, minmax(100%, 1fr));
            }

            .kpi-grid--center-metrics {
                grid-template-columns: repeat(auto-fit, minmax(48%, 1fr));
            }

            .kpi-metric-label {
                font-size: 0.8rem;
            }

            .kpi-metric-value {
                font-size: 1.05rem;
            }
        }

        @media (max-width: 520px) {
            .kpi-grid--center-metrics {
                grid-template-columns: repeat(auto-fit, minmax(100%, 1fr));
            }

            .kpi-sku-title {
                font-size: 1.0rem;
            }

            .kpi-sku-code {
                font-size: 0.8rem;
            }
        }

        @media (prefers-color-scheme: dark) {
            .kpi-sku-card,
            .kpi-center-card,
            .kpi-metric-card {
                background-color: rgba(13, 17, 23, 0.55);
                border-color: rgba(250, 250, 251, 0.15);
            }

            .kpi-metric-label {
                color: rgba(250, 250, 251, 0.75);
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
