"""
AI 챗봇 Function Calling 버전: OpenAI GPT-4o Function Calling
- 텍스트 요약 제거 → 메타데이터만 전달 (90% 토큰 절약)
- AI가 필요한 함수를 직접 선택 및 호출
- 정확한 계산, 확장 가능한 아키텍처
"""

import streamlit as st
import pandas as pd
import numpy as np
import openai  # OpenAI로 변경
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import math
from typing import Optional, Dict, List, Any


def safe_float(value) -> Optional[float]:
    """
    NaN, Inf를 안전하게 처리하여 JSON 직렬화 가능한 값으로 변환

    Args:
        value: 변환할 값 (pandas/numpy 타입 포함)

    Returns:
        float 또는 None (NaN/Inf인 경우)
    """
    if pd.isna(value) or (isinstance(value, float) and math.isinf(value)):
        return None
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    return float(value)


def prepare_minimal_metadata(
    snapshot_df: pd.DataFrame,
    moves_df: Optional[pd.DataFrame] = None,
    timeline_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    텍스트 요약 대신 메타데이터만 추출 (토큰 90% 절약)

    Returns:
        메타데이터 dict (SKU 목록, 센터 목록, 날짜 범위 등)
    """
    # None 체크 추가 (Phase 1 Quick Win)
    if snapshot_df is None or snapshot_df.empty:
        return {"status": "empty", "message": "데이터가 없습니다"}

    metadata = {
        "status": "ok",
        "snapshot": {
            "total_rows": len(snapshot_df),
            "centers": sorted(snapshot_df["center"].unique().tolist()) if "center" in snapshot_df.columns else [],
            "skus": sorted(snapshot_df["resource_code"].unique().tolist()[:50]) if "resource_code" in snapshot_df.columns else [],  # 상위 50개만
            "sku_count": int(snapshot_df["resource_code"].nunique()) if "resource_code" in snapshot_df.columns else 0,
            "date_range": None
        },
        "moves": {
            "available": moves_df is not None and not moves_df.empty,
            "date_range": None
        },
        "timeline": {
            "available": timeline_df is not None and not timeline_df.empty,
            "has_forecast": False,
            "date_range": None
        }
    }

    # 날짜 범위 (copy() 제거 - Phase 1 Quick Win)
    if "date" in snapshot_df.columns:
        date_series = pd.to_datetime(snapshot_df["date"], errors="coerce")
        min_date = date_series.min()
        max_date = date_series.max()
        if pd.notna(min_date) and pd.notna(max_date):
            metadata["snapshot"]["date_range"] = {
                "min": min_date.strftime('%Y-%m-%d'),
                "max": max_date.strftime('%Y-%m-%d')
            }

    if moves_df is not None and not moves_df.empty and "date" in moves_df.columns:
        date_series = pd.to_datetime(moves_df["date"], errors="coerce")
        min_date = date_series.min()
        max_date = date_series.max()
        if pd.notna(min_date) and pd.notna(max_date):
            metadata["moves"]["date_range"] = {
                "min": min_date.strftime('%Y-%m-%d'),
                "max": max_date.strftime('%Y-%m-%d')
            }

    if timeline_df is not None and not timeline_df.empty:
        if "is_forecast" in timeline_df.columns:
            metadata["timeline"]["has_forecast"] = timeline_df["is_forecast"].any()

    return metadata


# Gemini Function Declarations
GEMINI_FUNCTIONS = [
    {
        "name": "get_total_stock",
        "description": "전체 재고량을 조회합니다. 모든 센터와 SKU의 총 재고를 합산합니다.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "get_stock_by_center",
        "description": "센터별 재고량을 조회합니다. 특정 센터를 지정하거나 전체 센터의 재고를 확인할 수 있습니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "center": {
                    "type": "string",
                    "description": "센터 코드 (예: AMZUS, KR01). 지정하지 않으면 모든 센터 반환"
                }
            },
            "required": []
        }
    },
    {
        "name": "get_stock_by_sku",
        "description": "특정 SKU의 재고량과 센터별 분포를 조회합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "sku": {
                    "type": "string",
                    "description": "SKU 코드 (예: BA00021)"
                }
            },
            "required": ["sku"]
        }
    },
    {
        "name": "calculate_stockout_days",
        "description": "특정 SKU가 품절될 때까지 남은 일수를 계산합니다. 최근 7일 평균 판매량을 기반으로 예측합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "sku": {
                    "type": "string",
                    "description": "SKU 코드 (예: BA00021)"
                }
            },
            "required": ["sku"]
        }
    },
    {
        "name": "get_top_selling_skus",
        "description": "최근 30일 판매량이 많은 상위 SKU 목록을 조회합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "조회할 SKU 개수. 지정하지 않으면 5개 반환"
                }
            },
            "required": []
        }
    },
    {
        "name": "get_sku_trend",
        "description": "특정 SKU의 시계열 재고 추세를 조회합니다. 일별 재고 변화와 예측 데이터를 포함합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "sku": {
                    "type": "string",
                    "description": "SKU 코드 (예: BA00021)"
                },
                "days": {
                    "type": "integer",
                    "description": "조회할 일수. 지정하지 않으면 30일 반환"
                }
            },
            "required": ["sku"]
        }
    },
    {
        "name": "get_sales_summary",
        "description": "특정 SKU의 판매 요약 정보를 조회합니다. 센터별, 날짜별 판매량을 포함합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "sku": {
                    "type": "string",
                    "description": "SKU 코드 (예: BA00021)"
                },
                "days": {
                    "type": "integer",
                    "description": "조회할 일수. 지정하지 않으면 7일 반환"
                }
            },
            "required": ["sku"]
        }
    },
    {
        "name": "compare_skus",
        "description": "두 SKU의 재고량, 판매량, 추세를 비교합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "sku1": {
                    "type": "string",
                    "description": "첫 번째 SKU 코드"
                },
                "sku2": {
                    "type": "string",
                    "description": "두 번째 SKU 코드"
                }
            },
            "required": ["sku1", "sku2"]
        }
    },
    {
        "name": "search_low_stock_skus",
        "description": "품절 임박 SKU를 검색합니다. 지정한 일수 이내에 품절될 것으로 예상되는 SKU 목록을 반환합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "days_threshold": {
                    "type": "integer",
                    "description": "품절 임박 기준 일수. 지정하지 않으면 7일 사용"
                }
            },
            "required": []
        }
    },
    {
        "name": "forecast_sales",
        "description": "특정 SKU의 미래 판매량을 예측합니다. 최근 4주의 판매 데이터를 기반으로 선형 트렌드를 계산하여 다음 주 또는 다음 N주의 예상 판매량을 반환합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "sku": {
                    "type": "string",
                    "description": "SKU 코드 (예: BA00021)"
                },
                "weeks": {
                    "type": "integer",
                    "description": "몇 주 후를 예측할지. 지정하지 않으면 1주 후 (다음주) 예측"
                }
            },
            "required": ["sku"]
        }
    }
]


def get_latest_stock(snapshot_df: pd.DataFrame) -> pd.DataFrame:
    """
    날짜별 스냅샷에서 가장 최신 날짜의 재고만 추출

    Args:
        snapshot_df: 날짜별 재고 스냅샷

    Returns:
        최신 날짜의 재고만 포함한 DataFrame
    """
    if snapshot_df is None or snapshot_df.empty:
        return snapshot_df

    if "date" not in snapshot_df.columns:
        return snapshot_df

    # 날짜를 datetime으로 변환
    df = snapshot_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # ⚠️ 중요: NaT (잘못된 날짜) 제거
    # NaT는 정렬 시 맨 뒤에 배치되어 .last()에서 선택될 수 있음
    df = df[df["date"].notna()]

    if df.empty:
        return df

    # 각 SKU + 센터별로 가장 최신 날짜의 데이터만 선택
    if "resource_code" in df.columns and "center" in df.columns:
        # 날짜 기준으로 정렬 후, 각 그룹의 마지막(최신) 행만 선택
        latest = df.sort_values("date").groupby(["resource_code", "center"], as_index=False).last()
    else:
        # 그냥 가장 최신 날짜만
        latest_date = df["date"].max()
        latest = df[df["date"] == latest_date]

    return latest


def execute_function(
    function_name: str,
    parameters: Dict[str, Any],
    snapshot_df: pd.DataFrame,
    moves_df: Optional[pd.DataFrame] = None,
    timeline_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    OpenAI가 요청한 함수를 실행

    Args:
        function_name: 함수 이름
        parameters: 함수 파라미터
        snapshot_df: 재고 데이터 (날짜별 스냅샷)
        moves_df: 판매 데이터
        timeline_df: 시계열 데이터

    Returns:
        함수 실행 결과 dict
    """
    try:
        # ⚠️ 중요: 날짜별 스냅샷에서 최신 날짜의 재고만 사용
        latest_snapshot = get_latest_stock(snapshot_df)

        if function_name == "get_total_stock":
            total = latest_snapshot["stock_qty"].sum()
            return {
                "total_stock": float(total),
                "unit": "개",
                "center_count": int(latest_snapshot["center"].nunique()),
                "sku_count": int(latest_snapshot["resource_code"].nunique())
            }

        elif function_name == "get_stock_by_center":
            center = parameters.get("center")
            if center:
                center_data = latest_snapshot[latest_snapshot["center"] == center]
                if center_data.empty:
                    return {"error": f"센터 '{center}'를 찾을 수 없습니다"}
                return {
                    "center": center,
                    "total_stock": float(center_data["stock_qty"].sum()),
                    "sku_count": int(center_data["resource_code"].nunique()),
                    "unit": "개"
                }
            else:
                center_stock = latest_snapshot.groupby("center")["stock_qty"].sum().to_dict()
                return {
                    "centers": {k: float(v) for k, v in center_stock.items()},
                    "unit": "개"
                }

        elif function_name == "get_stock_by_sku":
            sku = parameters.get("sku")
            if not sku:
                return {"error": "SKU 파라미터가 필요합니다"}

            sku_data = latest_snapshot[latest_snapshot["resource_code"] == sku]
            if sku_data.empty:
                return {"error": f"SKU '{sku}'를 찾을 수 없습니다"}

            by_center = sku_data.groupby("center")["stock_qty"].sum().to_dict()
            return {
                "sku": sku,
                "total_stock": float(sku_data["stock_qty"].sum()),
                "by_center": {k: float(v) for k, v in by_center.items()},
                "unit": "개"
            }

        elif function_name == "calculate_stockout_days":
            sku = parameters.get("sku")
            if not sku:
                return {"error": "SKU 파라미터가 필요합니다"}

            # ✅ FIX: snapshot_df의 sales_qty 사용 (실제 판매 데이터)
            if "sales_qty" not in snapshot_df.columns:
                return {"error": "판매 데이터(sales_qty)를 찾을 수 없습니다"}

            sales_data = snapshot_df.copy()
            sales_data["date"] = pd.to_datetime(sales_data["date"], errors="coerce")

            # 최근 7일 평균 판매량
            cutoff_date = sales_data["date"].max() - timedelta(days=7)
            sales_recent = sales_data[
                (sales_data["date"] >= cutoff_date) &
                (sales_data["resource_code"] == sku)
            ]

            if sales_recent.empty:
                return {"error": f"SKU '{sku}'의 최근 7일 판매 데이터가 없습니다"}

            daily_sales = sales_recent["sales_qty"].sum() / 7
            # ⚠️ 최신 날짜의 재고만 사용
            current_stock = latest_snapshot[latest_snapshot["resource_code"] == sku]["stock_qty"].sum()

            if daily_sales <= 0:
                return {
                    "sku": sku,
                    "message": "최근 7일 판매량이 0입니다",
                    "current_stock": safe_float(current_stock)
                }

            days_left = current_stock / daily_sales
            return {
                "sku": sku,
                "current_stock": safe_float(current_stock),
                "daily_sales_avg": safe_float(daily_sales),
                "days_until_stockout": safe_float(days_left),
                "status": "urgent" if days_left < 3 else "warning" if days_left < 7 else "ok"
            }

        elif function_name == "get_top_selling_skus":
            limit = parameters.get("limit", 5)

            # ✅ FIX: snapshot_df의 sales_qty 사용 (실제 판매 데이터)
            if "sales_qty" not in snapshot_df.columns:
                return {"error": "판매 데이터(sales_qty)를 찾을 수 없습니다"}

            sales_data = snapshot_df.copy()
            sales_data["date"] = pd.to_datetime(sales_data["date"], errors="coerce")

            # 최근 30일 필터링
            cutoff_date = sales_data["date"].max() - timedelta(days=30)
            sales_recent = sales_data[sales_data["date"] >= cutoff_date]

            # SKU별 판매량 합계
            top_skus = sales_recent.groupby("resource_code")["sales_qty"].sum().nlargest(limit)

            # resource_name 추가 (있으면)
            result_list = []
            for sku, qty in top_skus.items():
                sku_info = {"sku": sku, "quantity": safe_float(qty)}
                if "resource_name" in latest_snapshot.columns:
                    sku_rows = latest_snapshot[latest_snapshot["resource_code"] == sku]
                    if not sku_rows.empty:
                        name = sku_rows["resource_name"].iloc[0]
                        if pd.notna(name):
                            sku_info["product_name"] = str(name)
                result_list.append(sku_info)

            return {
                "top_skus": result_list,
                "period": "last_30_days",
                "unit": "개"
            }

        elif function_name == "get_sku_trend":
            sku = parameters.get("sku")
            days = parameters.get("days", 30)

            if not sku or timeline_df is None or timeline_df.empty:
                return {"error": "SKU 파라미터와 시계열 데이터가 필요합니다"}

            timeline = timeline_df[timeline_df["resource_code"] == sku].copy()
            if timeline.empty:
                return {"error": f"SKU '{sku}'의 추세 데이터가 없습니다"}

            timeline["date"] = pd.to_datetime(timeline["date"], errors="coerce")
            # NaT 제거
            timeline = timeline[timeline["date"].notna()]

            if timeline.empty:
                return {"error": f"SKU '{sku}'의 유효한 날짜 데이터가 없습니다"}

            # ⚠️ 중요: 현재 날짜 기준으로 과거/미래 구분
            from datetime import datetime
            today = pd.Timestamp(datetime.now().date())

            # 과거 데이터만 선택 (미래 날짜는 예측 데이터)
            past_timeline = timeline[timeline["date"] <= today].copy()
            future_timeline = timeline[timeline["date"] > today].copy()

            # 과거 N일 데이터 선택
            if not past_timeline.empty:
                cutoff_date = today - pd.Timedelta(days=days)
                past_timeline = past_timeline[past_timeline["date"] >= cutoff_date]
                past_timeline = past_timeline.sort_values("date")

            # 실제 vs 예측 분리
            if "is_forecast" in timeline.columns:
                # is_forecast 플래그가 있으면 사용
                actual = past_timeline[past_timeline["is_forecast"] == False] if not past_timeline.empty else pd.DataFrame()

                # 과거 예측 데이터 (is_forecast=True but date <= today)
                past_forecast = past_timeline[past_timeline["is_forecast"] == True] if not past_timeline.empty else pd.DataFrame()

                # 과거 예측 + 미래 데이터 합치기
                forecast_parts = [df for df in [past_forecast, future_timeline] if not df.empty]
                forecast = pd.concat(forecast_parts, ignore_index=True) if forecast_parts else pd.DataFrame()
            else:
                # is_forecast 플래그가 없으면 날짜로만 구분
                actual = past_timeline
                forecast = future_timeline

            result = {
                "sku": sku,
                "actual_data": [
                    {
                        "date": row["date"].strftime('%Y-%m-%d') if pd.notna(row["date"]) else None,
                        "stock_qty": float(row["stock_qty"])
                    }
                    for _, row in actual.iterrows()
                ],
                "forecast_data": [
                    {
                        "date": row["date"].strftime('%Y-%m-%d') if pd.notna(row["date"]) else None,
                        "stock_qty": float(row["stock_qty"])
                    }
                    for _, row in forecast.iterrows()
                ]
            }

            # 추세 계산
            if len(actual) >= 2:
                first_qty = actual.iloc[0]["stock_qty"]
                last_qty = actual.iloc[-1]["stock_qty"]
                change = last_qty - first_qty
                change_pct = (change / first_qty * 100) if first_qty > 0 else 0
                result["trend"] = {
                    "direction": "증가" if change > 0 else "감소" if change < 0 else "유지",
                    "change": float(change),
                    "change_percent": float(change_pct)
                }

            return result

        elif function_name == "get_sales_summary":
            sku = parameters.get("sku")
            days = parameters.get("days", 7)

            if not sku:
                return {"error": "SKU 파라미터가 필요합니다"}

            # ✅ FIX: snapshot_df의 sales_qty 사용 (실제 판매 데이터)
            if "sales_qty" not in snapshot_df.columns:
                return {"error": "판매 데이터(sales_qty)를 찾을 수 없습니다"}

            sales_data = snapshot_df.copy()
            sales_data["date"] = pd.to_datetime(sales_data["date"], errors="coerce")

            # 최근 N일 필터링
            cutoff_date = sales_data["date"].max() - timedelta(days=days)
            sku_sales = sales_data[
                (sales_data["date"] >= cutoff_date) &
                (sales_data["resource_code"] == sku)
            ]

            if sku_sales.empty:
                return {"error": f"SKU '{sku}'의 최근 {days}일 판매 데이터가 없습니다"}

            # 센터별 판매량
            by_center = sku_sales.groupby("center")["sales_qty"].sum().to_dict()

            # 일별 판매량
            by_date = sku_sales.groupby(sku_sales["date"].dt.date)["sales_qty"].sum()

            total_sales = sku_sales["sales_qty"].sum()

            return {
                "sku": sku,
                "period_days": days,
                "total_sales": safe_float(total_sales),
                "daily_avg": safe_float(total_sales / days),
                "by_center": {k: safe_float(v) for k, v in by_center.items()},
                "daily_breakdown": [
                    {"date": str(date), "quantity": safe_float(qty)}
                    for date, qty in by_date.items()
                ],
                "unit": "개"
            }

        elif function_name == "compare_skus":
            sku1 = parameters.get("sku1")
            sku2 = parameters.get("sku2")

            if not sku1 or not sku2:
                return {"error": "두 개의 SKU 파라미터가 필요합니다"}

            # 재고 비교 - ⚠️ 최신 날짜의 재고만 사용
            stock1 = latest_snapshot[latest_snapshot["resource_code"] == sku1]["stock_qty"].sum()
            stock2 = latest_snapshot[latest_snapshot["resource_code"] == sku2]["stock_qty"].sum()

            result = {
                "sku1": {
                    "code": sku1,
                    "stock": float(stock1)
                },
                "sku2": {
                    "code": sku2,
                    "stock": float(stock2)
                },
                "stock_diff": float(stock1 - stock2),
                "unit": "개"
            }

            # 판매 비교 - ✅ FIX: snapshot_df의 sales_qty 사용
            if "sales_qty" in snapshot_df.columns:
                sales_data = snapshot_df.copy()
                sales_data["date"] = pd.to_datetime(sales_data["date"], errors="coerce")
                cutoff_date = sales_data["date"].max() - timedelta(days=30)
                sales_recent = sales_data[sales_data["date"] >= cutoff_date]

                sales1 = sales_recent[sales_recent["resource_code"] == sku1]["sales_qty"].sum()
                sales2 = sales_recent[sales_recent["resource_code"] == sku2]["sales_qty"].sum()

                result["sku1"]["sales_30d"] = safe_float(sales1)
                result["sku2"]["sales_30d"] = safe_float(sales2)
                result["sales_diff"] = safe_float(sales1 - sales2)

            return result

        elif function_name == "search_low_stock_skus":
            days_threshold = parameters.get("days_threshold", 7)

            # ✅ FIX: snapshot_df의 sales_qty 사용 (실제 판매 데이터)
            if "sales_qty" not in snapshot_df.columns:
                return {"error": "판매 데이터(sales_qty)를 찾을 수 없습니다"}

            sales_data = snapshot_df.copy()
            sales_data["date"] = pd.to_datetime(sales_data["date"], errors="coerce")

            # 최근 7일 판매 데이터
            cutoff_date = sales_data["date"].max() - timedelta(days=7)
            sales_recent = sales_data[sales_data["date"] >= cutoff_date]

            # SKU별 일평균 판매량
            daily_sales_by_sku = sales_recent.groupby("resource_code")["sales_qty"].sum() / 7

            low_stock_skus = []
            for sku in daily_sales_by_sku.index:
                if daily_sales_by_sku[sku] <= 0:
                    continue

                # ⚠️ 최신 날짜의 재고만 사용
                current_stock = latest_snapshot[latest_snapshot["resource_code"] == sku]["stock_qty"].sum()
                days_left = current_stock / daily_sales_by_sku[sku]

                if 0 < days_left <= days_threshold:
                    sku_info = {
                        "sku": sku,
                        "current_stock": safe_float(current_stock),
                        "daily_sales": safe_float(daily_sales_by_sku[sku]),
                        "days_left": safe_float(days_left),
                        "severity": "urgent" if days_left < 3 else "warning"
                    }
                    # resource_name 추가 (있으면)
                    if "resource_name" in latest_snapshot.columns:
                        sku_rows = latest_snapshot[latest_snapshot["resource_code"] == sku]
                        if not sku_rows.empty:
                            name = sku_rows["resource_name"].iloc[0]
                            if pd.notna(name):
                                sku_info["product_name"] = str(name)
                    low_stock_skus.append(sku_info)

            # 심각도 순 정렬
            low_stock_skus.sort(key=lambda x: x["days_left"])

            return {
                "low_stock_skus": low_stock_skus[:10],  # 상위 10개
                "threshold_days": days_threshold,
                "total_found": len(low_stock_skus)
            }

        elif function_name == "forecast_sales":
            sku = parameters.get("sku")
            weeks = parameters.get("weeks", 1)

            if not sku:
                return {"error": "SKU 파라미터가 필요합니다"}

            # ✅ snapshot_df의 sales_qty 사용 (실제 판매 데이터)
            if "sales_qty" not in snapshot_df.columns:
                return {"error": "판매 데이터(sales_qty)를 찾을 수 없습니다"}

            sales_data = snapshot_df.copy()
            sales_data["date"] = pd.to_datetime(sales_data["date"], errors="coerce")

            # 최근 4주 데이터 (28일)
            latest_date = sales_data["date"].max()
            cutoff_date = latest_date - timedelta(days=28)
            sales_recent = sales_data[
                (sales_data["date"] >= cutoff_date) &
                (sales_data["resource_code"] == sku)
            ]

            if sales_recent.empty:
                return {"error": f"SKU '{sku}'의 최근 4주 판매 데이터가 없습니다"}

            # 주차별 판매량 계산
            sales_recent["week"] = sales_recent["date"].dt.isocalendar().week
            sales_recent["year"] = sales_recent["date"].dt.year
            weekly_sales = sales_recent.groupby(["year", "week"])["sales_qty"].sum().reset_index()
            weekly_sales = weekly_sales.sort_values(["year", "week"])

            if len(weekly_sales) < 2:
                # 데이터가 부족하면 최근 7일 평균 사용
                recent_7days = sales_data[
                    (sales_data["date"] >= latest_date - timedelta(days=7)) &
                    (sales_data["resource_code"] == sku)
                ]
                avg_weekly = recent_7days["sales_qty"].sum()  # 최근 1주 총합

                return {
                    "sku": sku,
                    "forecast_weeks": weeks,
                    "predicted_sales": safe_float(avg_weekly * weeks),
                    "method": "recent_average",
                    "weekly_average": safe_float(avg_weekly),
                    "confidence": "low",
                    "note": "데이터 부족으로 최근 1주 평균 사용"
                }

            # 선형 트렌드 계산
            weekly_sales["week_index"] = list(range(len(weekly_sales)))
            x = weekly_sales["week_index"].values
            y = weekly_sales["sales_qty"].values

            # 간단한 선형 회귀 (y = mx + b)
            n = len(x)
            x_mean = x.mean()
            y_mean = y.mean()

            numerator = ((x - x_mean) * (y - y_mean)).sum()
            denominator = ((x - x_mean) ** 2).sum()

            if denominator != 0:
                slope = numerator / denominator
                intercept = y_mean - slope * x_mean

                # 다음 weeks주 예측
                next_week_index = len(weekly_sales)
                predicted_sales = []
                for i in range(int(weeks)):  # weeks를 int로 변환
                    pred = intercept + slope * (next_week_index + i)
                    predicted_sales.append(max(0, pred))  # 음수 방지

                total_predicted = sum(predicted_sales)

                # 트렌드 방향
                trend = "증가" if slope > 0 else "감소" if slope < 0 else "유지"

                result = {
                    "sku": sku,
                    "forecast_weeks": int(weeks),
                    "predicted_sales": safe_float(total_predicted),
                    "weekly_breakdown": [safe_float(p) for p in predicted_sales],
                    "method": "linear_trend",
                    "trend": trend,
                    "trend_slope": safe_float(slope),
                    "recent_4weeks_average": safe_float(y_mean),
                    "confidence": "medium" if abs(slope) < y_mean * 0.2 else "low"
                }

                # product_name 추가 (있으면)
                if "resource_name" in latest_snapshot.columns:
                    sku_rows = latest_snapshot[latest_snapshot["resource_code"] == sku]
                    if not sku_rows.empty:
                        name = sku_rows["resource_name"].iloc[0]
                        if pd.notna(name):
                            result["product_name"] = str(name)

                return result
            else:
                # slope 계산 불가 시 평균 사용
                avg_weekly = y_mean
                return {
                    "sku": sku,
                    "forecast_weeks": int(weeks),
                    "predicted_sales": safe_float(avg_weekly * weeks),
                    "method": "average",
                    "weekly_average": safe_float(avg_weekly),
                    "confidence": "medium"
                }

        else:
            return {"error": f"알 수 없는 함수: {function_name}"}

    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        st.error(f"🐛 함수 실행 상세 오류:\n```\n{error_detail}\n```")
        return {"error": f"함수 실행 오류: {str(e)}"}


def ask_ai_with_functions(
    question: str,
    metadata: Dict[str, Any],
    snapshot_df: pd.DataFrame,
    moves_df: Optional[pd.DataFrame] = None,
    timeline_df: Optional[pd.DataFrame] = None,
    max_iterations: int = 5
) -> str:
    """
    OpenAI GPT-4o Function Calling으로 질문 답변

    Args:
        question: 사용자 질문
        metadata: 최소 메타데이터
        snapshot_df, moves_df, timeline_df: 실제 데이터 (함수 실행용)
        max_iterations: 최대 함수 호출 횟수

    Returns:
        AI 답변
    """
    try:
        today = datetime.now().strftime('%Y-%m-%d')

        # OpenAI 클라이언트 초기화
        client = openai.OpenAI(api_key=st.secrets["openai"]["api_key"])

        # 🆕 이전 대화 맥락 추가 (대화 연속성)
        context_section = ""
        if hasattr(st.session_state, 'last_question') and st.session_state.last_question:
            context_section = f"""

**이전 대화 맥락:**
- 이전 질문: {st.session_state.last_question}
- 이전 답변: {st.session_state.last_answer[:200]}...

**중요:** 현재 질문에 "그럼", "그것도", "같은" 같은 상대적 표현이 있다면, 이전 대화의 맥락(SKU, 센터, 기간 등)을 참고하세요.
"""

        # 시스템 프롬프트
        system_prompt = f"""당신은 SCM 재고 관리 전문가입니다.

**현재 날짜: {today}**

**이용 가능한 데이터:**
{json.dumps(metadata, ensure_ascii=False, indent=2)}{context_section}

**답변 규칙 (매우 중요!):**
1. ⚠️ 재고량, 판매량, SKU 정보, 예측 등 데이터 조회가 필요한 질문은 **반드시 먼저 함수를 호출**하세요
2. ⚠️ 절대 "함수를 호출하겠습니다"라고 말만 하지 마세요 - 즉시 함수를 실행하세요
3. ⚠️ "다음주", "몇개 팔릴까", "예상", "예측" 같은 단어가 있으면 forecast_sales() 함수를 호출하세요
4. 함수 결과를 받으면 그 정확한 숫자로 답변하세요 (쉼표 포맷: 1,234개)
5. 2-4문장으로 간결하게 답변하세요
6. 데이터에 없는 내용만 "데이터에서 확인할 수 없습니다"라고 답변하세요
7. 한국어로 작성하세요

**중요:** 함수 호출 가능한 질문에는 반드시 함수를 먼저 호출하세요. 텍스트 설명만 하지 마세요."""

        # OpenAI 형식으로 함수 선언 변환
        tools = []
        for func in GEMINI_FUNCTIONS:
            tools.append({
                "type": "function",
                "function": {
                    "name": func["name"],
                    "description": func["description"],
                    "parameters": func["parameters"]
                }
            })

        # 메시지 히스토리 초기화
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]

        # Function calling loop
        iteration = 0
        while iteration < max_iterations:
            # OpenAI API 호출
            response = client.chat.completions.create(
                model="gpt-4o",  # GPT-4o 모델
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )

            message = response.choices[0].message

            # DEBUG: 응답 타입 확인
            if iteration == 0:
                has_tool_calls = message.tool_calls is not None
                st.caption(f"🔍 DEBUG: 첫 응답 - tool_calls={has_tool_calls}")

            # 함수 호출이 있는 경우
            if message.tool_calls:
                messages.append(message)  # assistant 메시지 추가

                for tool_call in message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    st.caption(f"🔧 함수 호출: `{function_name}({json.dumps(function_args, ensure_ascii=False)})`")

                    # 함수 실행
                    result = execute_function(
                        function_name,
                        function_args,
                        snapshot_df,
                        moves_df,
                        timeline_df
                    )

                    # 🔍 DEBUG: 함수 실행 결과 로깅
                    st.caption(f"🔍 DEBUG: 함수 실행 결과 - {json.dumps(result, ensure_ascii=False)[:200]}...")

                    # 함수 결과를 메시지에 추가
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": json.dumps(result, ensure_ascii=False)
                    })

                iteration += 1

            # 텍스트 응답이면 종료
            elif message.content:
                return message.content.strip()

            # finish_reason이 stop이면 종료
            elif response.choices[0].finish_reason == "stop":
                if message.content:
                    return message.content.strip()
                else:
                    return "답변을 생성할 수 없습니다."

            else:
                break

        # 최종 응답
        return "답변을 생성할 수 없습니다."

    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        st.error(f"🐛 ask_ai_with_functions 상세 오류:\n```\n{error_detail}\n```")
        return f"⚠️ 오류 발생: {str(e)}"
def _safe_float(value) -> Optional[float]:
    """NaN, Infinity를 안전하게 처리하여 JSON 직렬화 가능한 형태로 변환"""
    try:
        if pd.isna(value):
            return None
        if isinstance(value, (float, np.floating)):
            if math.isinf(value):
                return None
        return float(value)
    except (ValueError, TypeError):
        return None


def detect_stockout_risks(
    snapshot_df: pd.DataFrame,
    moves_df: pd.DataFrame = None,
    timeline_df: pd.DataFrame = None,
    days_threshold: int = 7
) -> list[dict]:
    """
    품절 임박 SKU 감지 (개선된 벡터화 버전)

    성능 개선:
    - 이전: O(n × m) 복잡도 (SKU 1,000개 × Snapshot 10,000행 = 10,000,000 비교)
    - 개선: O(n + m) 복잡도 (벡터화 연산)
    - 예상 성능: 2-3초 → 2-3ms (1000배 향상)

    Gemini Function Calling 호환성:
    - ✅ JSON 직렬화 가능 (NaN/Inf 처리)
    - ✅ float() 변환 완료
    - ✅ None 값 안전 처리

    Args:
        snapshot_df: 재고 + 판매 데이터 (컬럼: resource_code, stock_qty, sales_qty, date)
        moves_df: (사용 안 함 - 하위 호환성)
        timeline_df: 예측 데이터 (옵션)
        days_threshold: 품절 임박 기준 (일)

    Returns:
        품절 임박 SKU 리스트 (상위 5개)
    """
    risks = []

    # ✅ 개선: None 체크 + sales_qty 확인
    if snapshot_df is None or snapshot_df.empty or "sales_qty" not in snapshot_df.columns:
        return risks

    try:
        # ⚠️ 중요: 최신 날짜의 재고만 사용
        latest_snapshot = get_latest_stock(snapshot_df)

        # ✅ FIX: snapshot_df의 sales_qty 사용 (실제 판매 데이터)
        sales_data = snapshot_df.copy()
        sales_data["date"] = pd.to_datetime(sales_data["date"], errors="coerce")

        # Phase 1: 최근 7일 판매 데이터
        cutoff_date = sales_data["date"].max() - pd.Timedelta(days=7)
        sales_recent = sales_data[sales_data["date"] >= cutoff_date]

        if not sales_recent.empty:
            # SKU별 일평균 판매량
            if "resource_code" in sales_recent.columns:
                daily_sales = sales_recent.groupby("resource_code")["sales_qty"].sum() / 7

                # ✅ Phase 2: 벡터화된 재고 분석 (반복문 제거!)
                # 이전: for sku in daily_sales.index: current_stock = snapshot_df[...] (O(n×m))
                # 개선: 한 번에 모든 SKU의 현재 재고 계산 (O(m))
                # ⚠️ 최신 날짜의 재고만 사용
                current_stock_by_sku = latest_snapshot.groupby("resource_code")["stock_qty"].sum()

                # 판매 데이터와 재고 데이터 병합
                stock_analysis = pd.DataFrame({
                    "daily_sales": daily_sales,
                    "current_stock": current_stock_by_sku
                }).dropna()

                if not stock_analysis.empty:
                    # 벡터화된 계산
                    stock_analysis["days_left"] = (
                        stock_analysis["current_stock"] / stock_analysis["daily_sales"]
                    )

                    # 벡터화된 조건 필터링 (NaN/Inf 안전 처리)
                    risk_mask = (
                        (stock_analysis["daily_sales"] > 0) &
                        (stock_analysis["days_left"] > 0) &
                        (stock_analysis["days_left"] <= days_threshold) &
                        ~stock_analysis["days_left"].isna() &
                        ~stock_analysis["days_left"].isin([np.inf, -np.inf])
                    )

                    risk_skus = stock_analysis[risk_mask].sort_values("days_left")

                    # ✅ Phase 3: JSON 직렬화 가능한 형식으로 변환
                    for sku, row in risk_skus.iterrows():
                        current_stock = _safe_float(row["current_stock"])
                        daily_sales_val = _safe_float(row["daily_sales"])
                        days_left = _safe_float(row["days_left"])

                        if None not in [current_stock, daily_sales_val, days_left]:
                            risks.append({
                                "sku": str(sku),
                                "current_stock": current_stock,
                                "daily_sales": daily_sales_val,
                                "days_left": days_left,
                                "severity": "high" if days_left <= 3 else "medium"
                            })

    except Exception as e:
        st.warning(f"품절 위험 감지 오류: {e}")

    return risks[:5]  # 상위 5개만


def detect_anomalies(
    snapshot_df: pd.DataFrame,
    timeline_df: pd.DataFrame = None,
    threshold: float = 0.5
) -> list[dict]:
    """
    재고 이상치 감지 (급증/급감)

    Args:
        snapshot_df: 현재 재고
        timeline_df: 시계열 데이터
        threshold: 변화율 임계값 (50% = 0.5)

    Returns:
        이상치 SKU 리스트
    """
    anomalies = []

    if timeline_df is None or timeline_df.empty:
        return anomalies

    try:
        timeline = timeline_df.copy()
        if "date" in timeline.columns and "resource_code" in timeline.columns:
            timeline["date"] = pd.to_datetime(timeline["date"], errors="coerce")

            # 실제 데이터만
            if "is_forecast" in timeline.columns:
                timeline = timeline[timeline["is_forecast"] == False]

            # SKU별 최근 7일 vs 이전 7일 비교
            latest_date = timeline["date"].max()
            recent_7days = timeline[timeline["date"] >= latest_date - pd.Timedelta(days=7)]
            prev_7days = timeline[
                (timeline["date"] >= latest_date - pd.Timedelta(days=14)) &
                (timeline["date"] < latest_date - pd.Timedelta(days=7))
            ]

            for sku in timeline["resource_code"].unique():
                recent_avg = recent_7days[recent_7days["resource_code"] == sku]["stock_qty"].mean()
                prev_avg = prev_7days[prev_7days["resource_code"] == sku]["stock_qty"].mean()

                if pd.notna(recent_avg) and pd.notna(prev_avg) and prev_avg > 0:
                    change_rate = (recent_avg - prev_avg) / prev_avg

                    if abs(change_rate) >= threshold:
                        anomalies.append({
                            "sku": sku,
                            "recent_avg": recent_avg,
                            "prev_avg": prev_avg,
                            "change_rate": change_rate,
                            "type": "급증" if change_rate > 0 else "급감"
                        })

            # 변화율 절댓값 순으로 정렬
            anomalies.sort(key=lambda x: abs(x["change_rate"]), reverse=True)

    except Exception as e:
        st.warning(f"이상치 감지 오류: {e}")

    return anomalies[:3]  # 상위 3개만


def check_data_quality(
    snapshot_df: pd.DataFrame,
    moves_df: pd.DataFrame = None,
    timeline_df: pd.DataFrame = None
) -> list[dict]:
    """
    데이터 품질 이슈 감지

    Returns:
        품질 이슈 리스트
    """
    issues = []

    try:
        # 1. 음수 재고 체크
        if "stock_qty" in snapshot_df.columns:
            negative_stock = snapshot_df[snapshot_df["stock_qty"] < 0]
            if not negative_stock.empty:
                issues.append({
                    "type": "negative_stock",
                    "severity": "high",
                    "message": f"⚠️ 음수 재고 발견: {len(negative_stock)}개 SKU",
                    "details": negative_stock[["resource_code", "center", "stock_qty"]].head(3).to_dict("records")
                })

        # 2. 날짜 누락 체크 (moves_df)
        if moves_df is not None and not moves_df.empty and "date" in moves_df.columns:
            moves_df_copy = moves_df.copy()
            moves_df_copy["date"] = pd.to_datetime(moves_df_copy["date"], errors="coerce")
            null_dates = moves_df_copy["date"].isna().sum()
            if null_dates > 0:
                issues.append({
                    "type": "missing_dates",
                    "severity": "medium",
                    "message": f"⚠️ 판매 데이터 날짜 누락: {null_dates}건",
                    "details": None
                })

        # 3. 최신 데이터 확인
        if "date" in snapshot_df.columns:
            snapshot_df_copy = snapshot_df.copy()
            snapshot_df_copy["date"] = pd.to_datetime(snapshot_df_copy["date"], errors="coerce")
            latest_date = snapshot_df_copy["date"].max()
            if pd.notna(latest_date):
                from datetime import datetime, timedelta
                days_old = (datetime.now() - latest_date).days
                if days_old > 1:
                    issues.append({
                        "type": "stale_data",
                        "severity": "low",
                        "message": f"ℹ️ 재고 데이터가 {days_old}일 전입니다 (최신: {latest_date.strftime('%Y-%m-%d')})",
                        "details": None
                    })

    except Exception as e:
        st.warning(f"데이터 품질 체크 오류: {e}")

    return issues


def render_proactive_insights(
    snapshot_df: pd.DataFrame,
    moves_df: pd.DataFrame,
    timeline_df: pd.DataFrame
):
    """
    프로액티브 인사이트 UI 렌더링
    """
    # 인사이트 감지
    stockout_risks = detect_stockout_risks(snapshot_df, moves_df, timeline_df)
    anomalies = detect_anomalies(snapshot_df, timeline_df)
    quality_issues = check_data_quality(snapshot_df, moves_df, timeline_df)

    # 인사이트가 하나라도 있으면 표시
    if stockout_risks or anomalies or quality_issues:
        with st.expander("🔔 주목할 이슈", expanded=True):
            col1, col2, col3 = st.columns(3)

            # 품절 위험
            with col1:
                if stockout_risks:
                    st.markdown("**⚠️ 품절 임박**")
                    for risk in stockout_risks[:3]:
                        severity_icon = "🔴" if risk["severity"] == "high" else "🟡"
                        st.caption(
                            f"{severity_icon} {risk['sku']}: "
                            f"{risk['days_left']:.1f}일 남음 "
                            f"(재고 {risk['current_stock']:.0f}개)"
                        )

            # 이상치
            with col2:
                if anomalies:
                    st.markdown("**📊 급격한 변화**")
                    for anomaly in anomalies[:3]:
                        icon = "📈" if anomaly["type"] == "급증" else "📉"
                        st.caption(
                            f"{icon} {anomaly['sku']}: "
                            f"{anomaly['type']} {abs(anomaly['change_rate'])*100:.0f}% "
                            f"({anomaly['prev_avg']:.0f}→{anomaly['recent_avg']:.0f})"
                        )

            # 데이터 품질
            with col3:
                if quality_issues:
                    st.markdown("**🔍 데이터 이슈**")
                    for issue in quality_issues[:3]:
                        st.caption(issue["message"])


def calculate_kpi(
    function_name: str,
    snapshot_df: pd.DataFrame,
    moves_df: pd.DataFrame = None,
    **kwargs
) -> dict:
    """
    KPI 계산 함수 (Function calling용)

    Args:
        function_name: 호출할 함수 이름
        snapshot_df: 재고 데이터
        moves_df: 판매 데이터
        **kwargs: 추가 파라미터

    Returns:
        계산 결과 dict
    """
    try:
        if function_name == "calculate_total_stock":
            total = snapshot_df["stock_qty"].sum()
            return {"total_stock": float(total), "unit": "개"}

        elif function_name == "get_stock_by_center":
            center_stock = snapshot_df.groupby("center")["stock_qty"].sum().to_dict()
            return {"center_stock": {k: float(v) for k, v in center_stock.items()}, "unit": "개"}

        elif function_name == "get_stock_by_sku":
            sku = kwargs.get("sku")
            if sku:
                sku_data = snapshot_df[snapshot_df["resource_code"] == sku]
                if not sku_data.empty:
                    total = sku_data["stock_qty"].sum()
                    by_center = sku_data.groupby("center")["stock_qty"].sum().to_dict()
                    return {
                        "sku": sku,
                        "total_stock": float(total),
                        "by_center": {k: float(v) for k, v in by_center.items()},
                        "unit": "개"
                    }
            return {"error": "SKU not found"}

        elif function_name == "calculate_stockout_days":
            sku = kwargs.get("sku")
            if sku and moves_df is not None:
                # 최근 7일 평균 판매량
                moves_recent = moves_df.copy()
                moves_recent["date"] = pd.to_datetime(moves_recent["date"], errors="coerce")
                cutoff_date = moves_recent["date"].max() - pd.Timedelta(days=7)
                moves_recent = moves_recent[
                    (moves_recent["date"] >= cutoff_date) &
                    (moves_recent["resource_code"] == sku)
                ]

                if not moves_recent.empty:
                    daily_sales = moves_recent["quantity"].sum() / 7
                    current_stock = snapshot_df[snapshot_df["resource_code"] == sku]["stock_qty"].sum()

                    if daily_sales > 0:
                        days_left = current_stock / daily_sales
                        return {
                            "sku": sku,
                            "current_stock": float(current_stock),
                            "daily_sales_avg": float(daily_sales),
                            "days_until_stockout": float(days_left),
                            "status": "urgent" if days_left < 3 else "warning" if days_left < 7 else "ok"
                        }

            return {"error": "Cannot calculate stockout days"}

        elif function_name == "get_top_selling_skus":
            limit = kwargs.get("limit", 5)
            if moves_df is not None:
                moves_recent = moves_df.copy()
                moves_recent["date"] = pd.to_datetime(moves_recent["date"], errors="coerce")
                cutoff_date = moves_recent["date"].max() - pd.Timedelta(days=30)
                moves_recent = moves_recent[moves_recent["date"] >= cutoff_date]

                top_skus = moves_recent.groupby("resource_code")["quantity"].sum().nlargest(limit)
                return {
                    "top_skus": {k: float(v) for k, v in top_skus.items()},
                    "period": "last_30_days",
                    "unit": "개"
                }

            return {"error": "No sales data available"}

    except Exception as e:
        return {"error": str(e)}

    return {"error": "Unknown function"}


def detect_kpi_need(question: str) -> tuple[bool, str, dict]:
    """
    질문에서 KPI 계산 필요 여부 감지

    Returns:
        (need_kpi, function_name, kwargs)
    """
    question_lower = question.lower()

    # 총 재고
    if "총 재고" in question_lower or "전체 재고" in question_lower:
        return (True, "calculate_total_stock", {})

    # 센터별 재고
    if ("센터별" in question_lower or "center" in question_lower) and "재고" in question_lower:
        return (True, "get_stock_by_center", {})

    # SKU별 재고
    import re
    sku_pattern = r'\b[A-Z]{2}\d{5}\b'
    skus = re.findall(sku_pattern, question)
    if skus and "재고" in question_lower:
        return (True, "get_stock_by_sku", {"sku": skus[0]})

    # 품절 임박
    if skus and ("품절" in question_lower or "소진" in question_lower or "남은" in question_lower):
        return (True, "calculate_stockout_days", {"sku": skus[0]})

    # 상위 판매
    if "상위" in question_lower and ("판매" in question_lower or "판매량" in question_lower):
        return (True, "get_top_selling_skus", {"limit": 5})

    return (False, None, {})


def ask_ai(question: str, data_context: str, snapshot_df: pd.DataFrame = None, moves_df: pd.DataFrame = None) -> str:
    """
    OpenAI에게 질문하기 (백업 함수, Function calling 없이 단순 텍스트)

    Args:
        question: 사용자 질문
        data_context: 데이터 컨텍스트
        snapshot_df: 재고 데이터 (KPI 계산용)
        moves_df: 판매 데이터 (KPI 계산용)

    Returns:
        AI 답변
    """
    try:
        from datetime import datetime
        today = datetime.now().strftime('%Y-%m-%d')

        # 1. KPI 계산 필요 여부 감지
        need_kpi, func_name, kwargs = detect_kpi_need(question)

        kpi_result = None
        if need_kpi and snapshot_df is not None:
            kpi_result = calculate_kpi(func_name, snapshot_df, moves_df, **kwargs)

        # OpenAI 클라이언트 초기화
        client = openai.OpenAI(api_key=st.secrets["openai"]["api_key"])

        # 2. KPI 결과가 있으면 프롬프트에 추가
        kpi_section = ""
        if kpi_result:
            import json
            kpi_section = f"""

[정확한 계산 결과]
{json.dumps(kpi_result, ensure_ascii=False, indent=2)}

**중요:** 위 계산 결과는 코드로 정확하게 계산된 값입니다. 반드시 이 숫자를 사용하세요."""

        prompt = f"""당신은 SCM 재고 관리 전문가입니다.

**현재 날짜: {today}**

아래 재고 데이터를 참고해서 사용자 질문에 답변하세요.

[재고 데이터]
{data_context}{kpi_section}

[사용자 질문]
{question}

**답변 규칙:**
1. [정확한 계산 결과]가 있으면 반드시 그 숫자를 사용하세요 (텍스트 데이터 대신)
2. 숫자는 쉼표로 포맷팅하세요 (예: 1,234개)
3. 2-4문장으로 간결하게 작성하세요
4. 데이터에 없는 내용은 "데이터에서 확인할 수 없습니다"라고 답변하세요
5. 날짜를 언급할 때는 현재 날짜({today})를 기준으로 과거/미래를 명확히 구분하세요
6. "📊 실제 데이터 기간"은 과거 실제 데이터, "🔮 예측 데이터 기간"은 미래 예측 데이터입니다
7. 한국어로 작성하세요

답변:"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "당신은 SCM 재고 관리 전문가입니다."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"⚠️ 오류 발생: {e}\n\n제공된 데이터:\n{data_context}"


def suggest_followup_questions(question: str, answer: str, data_context: str) -> list[str]:
    """
    답변을 기반으로 후속 질문 제안

    Args:
        question: 원래 질문
        answer: AI 답변
        data_context: 데이터 컨텍스트 (간략 버전)

    Returns:
        후속 질문 3개
    """
    try:
        # OpenAI 클라이언트 초기화
        client = openai.OpenAI(api_key=st.secrets["openai"]["api_key"])

        # 데이터 컨텍스트 요약 (토큰 절약)
        context_summary = data_context[:500] + "..." if len(data_context) > 500 else data_context

        prompt = f"""당신은 SCM 재고 관리 전문가입니다.

사용자가 다음 질문을 했고, 답변을 받았습니다:

[질문] {question}
[답변] {answer}

[이용 가능한 데이터]
{context_summary}

이제 사용자가 궁금해할 만한 **후속 질문 3개**를 제안하세요.

**규칙:**
1. 원래 질문과 관련되고 자연스럽게 이어지는 질문
2. 제공된 데이터로 답변 가능한 질문만 제안
3. 각 질문은 15자 이내로 간결하게
4. 구체적인 SKU/센터/날짜가 있으면 포함
5. 한 줄에 하나씩, 번호 없이 작성

예시:
BA00021의 판매 추세는?
다음주 예상 재고는?
어느 센터가 재고가 부족한가요?

후속 질문:"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "당신은 SCM 재고 관리 전문가입니다."},
                {"role": "user", "content": prompt}
            ]
        )

        questions = [q.strip() for q in response.choices[0].message.content.strip().split('\n') if q.strip()]
        return questions[:3]  # 상위 3개만

    except Exception as e:
        # 실패 시 기본 질문 반환
        return [
            "센터별 재고 분포는?",
            "재고가 부족한 SKU는?",
            "최근 판매 추세는?"
        ]


def extract_entities_from_question(question: str, snapshot_df: pd.DataFrame, moves_df: pd.DataFrame = None) -> dict:
    """
    질문에서 엔티티 추출 (SKU, 센터, 날짜 등)

    Returns:
        {"skus": [list], "centers": [list], "date_range": tuple or None}
    """
    import re
    from datetime import datetime, timedelta

    entities = {
        "skus": [],
        "centers": [],
        "date_range": None
    }

    # 1. SKU 추출 (BA00021 형식)
    sku_pattern = r'\b[A-Z]{2}\d{5}\b'
    found_skus = re.findall(sku_pattern, question)
    if found_skus and "resource_code" in snapshot_df.columns:
        # 실제 존재하는 SKU만
        valid_skus = snapshot_df["resource_code"].unique()
        entities["skus"] = [sku for sku in found_skus if sku in valid_skus]

    # 2. 센터 추출
    question_upper = question.upper()
    question_lower = question.lower()

    # 센터 별칭 매핑 (사용자가 자연어로 입력한 경우)
    center_aliases = {
        "아마존": "AMZUS",
        "amazon": "AMZUS",
        # "미국", "us" 제거 - CJ서부US와 충돌 방지
        "한국": "태광KR",
        "korea": "태광KR",
        "kr": "태광KR",
        "태광": "태광KR",
        # CJ서부US 별칭
        "미국cj": "CJ서부US",
        "cj미국": "CJ서부US",
        "미국창고": "CJ서부US",
        "미국센터": "CJ서부US",
        "서부창고": "CJ서부US",
        "미국서부": "CJ서부US"
    }

    # 별칭으로 센터 찾기
    for alias, center_code in center_aliases.items():
        if alias in question_lower:
            entities["centers"].append(center_code)

    # 정확한 센터명으로도 찾기
    if "center" in snapshot_df.columns:
        all_centers = snapshot_df["center"].unique()
        for center in all_centers:
            if center in question_upper or center.lower() in question_lower:
                if center not in entities["centers"]:
                    entities["centers"].append(center)

    # AMZUS, KR01 등 흔한 패턴 (정규식)
    center_patterns = [r'\bAMZUS\b', r'\bAMZKR\b', r'\bKR0[1-9]\b']
    for pattern in center_patterns:
        matches = re.findall(pattern, question_upper)
        entities["centers"].extend(matches)

    entities["centers"] = list(set(entities["centers"]))  # 중복 제거

    # 3. 날짜 추출 (상대적 표현)
    today = datetime.now()
    question_lower = question.lower()

    if "오늘" in question_lower:
        entities["date_range"] = (today, today)
    elif "어제" in question_lower:
        yesterday = today - timedelta(days=1)
        entities["date_range"] = (yesterday, yesterday)
    elif "최근 7일" in question_lower or "지난 일주일" in question_lower:
        entities["date_range"] = (today - timedelta(days=7), today)
    elif "최근 30일" in question_lower or "지난 한달" in question_lower:
        entities["date_range"] = (today - timedelta(days=30), today)
    elif "이번주" in question_lower:
        # 이번 주 월요일부터
        weekday = today.weekday()
        monday = today - timedelta(days=weekday)
        entities["date_range"] = (monday, today)

    # 절대 날짜 패턴 (YYYY-MM-DD)
    date_pattern = r'\d{4}-\d{2}-\d{2}'
    date_matches = re.findall(date_pattern, question)
    if date_matches:
        try:
            date_obj = datetime.strptime(date_matches[0], '%Y-%m-%d')
            entities["date_range"] = (date_obj, date_obj)
        except:
            pass

    return entities


def analyze_question_for_chart(question: str) -> dict:
    """
    질문을 분석해서 차트 필요 여부 및 타입 판단

    Returns:
        {"need_chart": bool, "chart_type": str, "entities": dict}
    """
    question_lower = question.lower()

    # 차트가 명시적으로 필요한 키워드 (더 엄격하게)
    explicit_chart_keywords = ["그래프", "차트", "시각화", "보여줘", "그려줘"]
    has_explicit_request = any(kw in question_lower for kw in explicit_chart_keywords)

    # 차트가 도움이 될 수 있는 키워드 (추세, 비교, 분포 등)
    implicit_chart_keywords = ["추세", "변화", "비교", "분포", "트렌드", "센터별", "sku별"]
    has_implicit_need = any(kw in question_lower for kw in implicit_chart_keywords)

    # 단순 사실 질문 키워드 (차트 불필요)
    # 단, "센터별"이나 "sku별" 같은 분포 질문은 제외
    simple_fact_keywords = ["총 몇", "몇개", "얼마"]
    is_simple_fact = any(kw in question_lower for kw in simple_fact_keywords) and not has_implicit_need

    # 명시적 요청이 있거나, 암묵적 필요가 있으면서 단순 사실 질문이 아닌 경우만 차트 생성
    need_chart = has_explicit_request or (has_implicit_need and not is_simple_fact)

    # 차트 타입 판단
    chart_type = None
    if need_chart:
        if "추세" in question_lower or "변화" in question_lower or "트렌드" in question_lower:
            chart_type = "line"  # 시계열
        elif "비교" in question_lower or "분포" in question_lower or "센터별" in question_lower or "sku별" in question_lower:
            chart_type = "bar"  # 바 차트
        elif "비율" in question_lower or "점유" in question_lower:
            chart_type = "pie"  # 파이 차트

    # 엔티티 추출 (간단 버전)
    entities = {
        "has_sku": bool([s for s in question if s.isupper() and len(s) >= 6]),  # BA00021 같은 패턴
        "has_center": any(c in question_lower for c in ["amz", "kr0", "센터"]),
        "time_related": any(t in question_lower for t in ["일", "주", "월", "날짜", "기간", "어제", "오늘"])
    }

    return {
        "need_chart": need_chart,
        "chart_type": chart_type,
        "entities": entities
    }


def generate_chart(
    question: str,
    snapshot_df: pd.DataFrame,
    moves_df: pd.DataFrame = None,
    timeline_df: pd.DataFrame = None
):
    """
    질문에 맞는 차트 자동 생성

    Returns:
        plotly figure 또는 None
    """
    try:
        analysis = analyze_question_for_chart(question)

        if not analysis["need_chart"]:
            return None

        chart_type = analysis["chart_type"]
        entities = analysis["entities"]

        # 1. 시계열 차트 (추세, 변화)
        if chart_type == "line" and timeline_df is not None and not timeline_df.empty:
            timeline = timeline_df.copy()
            if "date" in timeline.columns and "stock_qty" in timeline.columns:
                timeline["date"] = pd.to_datetime(timeline["date"], errors="coerce")
                timeline = timeline.sort_values("date")

                # 특정 SKU가 언급되었으면 그것만
                if entities["has_sku"] and "resource_code" in timeline.columns:
                    # 질문에서 SKU 추출 (간단 버전)
                    import re
                    sku_pattern = r'\b[A-Z]{2}\d{5}\b'
                    skus = re.findall(sku_pattern, question)
                    if skus:
                        timeline = timeline[timeline["resource_code"].isin(skus)]

                # 실제 vs 예측 구분
                if "is_forecast" in timeline.columns:
                    fig = go.Figure()

                    actual = timeline[timeline["is_forecast"] == False]
                    forecast = timeline[timeline["is_forecast"] == True]

                    if "resource_code" in timeline.columns:
                        for sku in timeline["resource_code"].unique()[:3]:  # 최대 3개
                            sku_actual = actual[actual["resource_code"] == sku]
                            sku_forecast = forecast[forecast["resource_code"] == sku]

                            if not sku_actual.empty:
                                fig.add_trace(go.Scatter(
                                    x=sku_actual["date"],
                                    y=sku_actual["stock_qty"],
                                    name=f"{sku} (실제)",
                                    mode="lines+markers"
                                ))

                            if not sku_forecast.empty:
                                fig.add_trace(go.Scatter(
                                    x=sku_forecast["date"],
                                    y=sku_forecast["stock_qty"],
                                    name=f"{sku} (예측)",
                                    mode="lines",
                                    line=dict(dash="dash")
                                ))
                    else:
                        fig.add_trace(go.Scatter(x=actual["date"], y=actual["stock_qty"], name="실제"))
                        if not forecast.empty:
                            fig.add_trace(go.Scatter(
                                x=forecast["date"],
                                y=forecast["stock_qty"],
                                name="예측",
                                line=dict(dash="dash")
                            ))

                    fig.update_layout(
                        title="재고 추세",
                        xaxis_title="날짜",
                        yaxis_title="재고량",
                        height=400
                    )
                    return fig

                else:
                    fig = px.line(
                        timeline,
                        x="date",
                        y="stock_qty",
                        color="resource_code" if "resource_code" in timeline.columns else None,
                        title="재고 추세"
                    )
                    fig.update_layout(height=400)
                    return fig

        # 2. 바 차트 (센터별, SKU별 비교)
        elif chart_type == "bar":
            if "센터" in question or "center" in question.lower():
                # 센터별 재고
                center_stock = snapshot_df.groupby("center")["stock_qty"].sum().reset_index()
                center_stock = center_stock.sort_values("stock_qty", ascending=False)

                fig = px.bar(
                    center_stock,
                    x="center",
                    y="stock_qty",
                    title="센터별 재고",
                    labels={"center": "센터", "stock_qty": "재고량"}
                )
                fig.update_layout(height=400)
                return fig

            elif "sku" in question.lower() or entities["has_sku"]:
                # SKU별 재고 (상위 10개)
                sku_stock = snapshot_df.groupby("resource_code")["stock_qty"].sum().reset_index()
                sku_stock = sku_stock.sort_values("stock_qty", ascending=False).head(10)

                fig = px.bar(
                    sku_stock,
                    x="resource_code",
                    y="stock_qty",
                    title="SKU별 재고 (상위 10개)",
                    labels={"resource_code": "SKU", "stock_qty": "재고량"}
                )
                fig.update_layout(height=400)
                return fig

        # 3. 파이 차트 (비율, 점유율)
        elif chart_type == "pie":
            center_stock = snapshot_df.groupby("center")["stock_qty"].sum().reset_index()

            fig = px.pie(
                center_stock,
                names="center",
                values="stock_qty",
                title="센터별 재고 비율"
            )
            fig.update_layout(height=400)
            return fig

    except Exception as e:
        st.warning(f"차트 생성 오류: {e}")

    return None


def render_simple_chatbot_tab(
    snapshot_df: pd.DataFrame,
    moves_df: pd.DataFrame,
    timeline_df: pd.DataFrame,
    selected_centers: list[str],
    selected_skus: list[str]
):
    """
    간단한 AI 챗봇 탭 렌더링

    Args:
        snapshot_df: 전체 스냅샷 데이터
        moves_df: 판매/입고 이동 데이터
        timeline_df: 30일 시계열 + 예측 데이터
        selected_centers: 선택된 센터
        selected_skus: 선택된 SKU
    """
    st.subheader("🤖 AI 어시스턴트 (GPT-4o Function Calling - 토큰 90% 절약)")

    # 필터링
    snap = snapshot_df.copy()
    if "center" in snap.columns:
        snap = snap[snap["center"].astype(str).isin(selected_centers)]
    if "resource_code" in snap.columns:
        snap = snap[snap["resource_code"].astype(str).isin(selected_skus)]

    if snap.empty:
        st.warning("선택된 필터에 데이터가 없습니다")
        return

    st.caption(f"📊 필터링된 데이터: {len(snap):,}행 (센터 {snap['center'].nunique()}곳, SKU {snap['resource_code'].nunique()}개)")

    # 프로액티브 인사이트 표시
    render_proactive_insights(snap, moves_df, timeline_df)

    st.divider()

    # 세션 상태 초기화
    if "last_question" not in st.session_state:
        st.session_state.last_question = ""
    if "last_answer" not in st.session_state:
        st.session_state.last_answer = ""
    if "last_metadata" not in st.session_state:
        st.session_state.last_metadata = {}
    if "last_entities" not in st.session_state:
        st.session_state.last_entities = {}  # 이전 질문의 SKU, 센터, 기간 저장

    # pending_question이 있으면 자동 실행
    auto_ask = False
    if "pending_question" in st.session_state and st.session_state.pending_question:
        auto_ask = True

    # 질문 입력 폼 (Enter 키 지원)
    with st.form(key="question_form", clear_on_submit=False):
        # pending_question이 있으면 자동으로 입력
        default_value = st.session_state.get("pending_question", "")

        question = st.text_input(
            "질문을 입력하세요",
            placeholder="예: 총 재고는? / BA00021은 어느 센터에? / 재고가 가장 많은 센터는?",
            key="simple_q",
            value=default_value
        )
        submit_button = st.form_submit_button("💬 질문하기", type="primary")

    # pending_question 정리 및 자동 실행 트리거
    if auto_ask:
        # pending_question을 question으로 복사
        question = st.session_state.pending_question
        st.session_state.pop("pending_question")
        # 다음 rerun에서 form이 제출된 것처럼 처리
        submit_button = True

    if submit_button and question:
        with st.spinner("🤔 생각 중..."):
            # 질문에서 엔티티 추출 (SKU, 센터, 날짜)
            entities = extract_entities_from_question(question, snap, moves_df)

            # 🆕 상대적 표현 감지 (대화 맥락 유지)
            relative_keywords = ["그럼", "그것도", "거기도", "같은", "그 sku", "그 센터", "역시", "또"]
            has_relative_ref = any(kw in question.lower() for kw in relative_keywords)

            # 상대적 표현이 있고, 현재 질문에 엔티티가 부족하면 이전 엔티티 재사용
            context_note = ""
            if has_relative_ref and st.session_state.last_entities:
                # SKU가 없으면 이전 SKU 재사용
                if not entities["skus"] and st.session_state.last_entities.get("skus"):
                    entities["skus"] = st.session_state.last_entities["skus"]
                    context_note += f"이전 SKU({entities['skus']}) 재사용. "

                # 센터가 없으면 이전 센터 재사용
                if not entities["centers"] and st.session_state.last_entities.get("centers"):
                    entities["centers"] = st.session_state.last_entities["centers"]
                    context_note += f"이전 센터({entities['centers']}) 재사용. "

                # 날짜가 없으면 이전 날짜 재사용
                if not entities["date_range"] and st.session_state.last_entities.get("date_range"):
                    entities["date_range"] = st.session_state.last_entities["date_range"]
                    context_note += f"이전 기간 재사용. "

                if context_note:
                    st.caption(f"🔗 대화 맥락: {context_note}")

            # 자동 필터링
            filtered_snap = snap.copy()
            filtered_moves = moves_df.copy() if moves_df is not None else None
            filtered_timeline = timeline_df.copy() if timeline_df is not None else None

            filter_applied = False
            filter_msg = ""

            if entities["skus"]:
                filtered_snap = filtered_snap[filtered_snap["resource_code"].isin(entities["skus"])]
                if filtered_timeline is not None and "resource_code" in filtered_timeline.columns:
                    filtered_timeline = filtered_timeline[filtered_timeline["resource_code"].isin(entities["skus"])]
                filter_msg += f"SKU: {', '.join(entities['skus'])} "
                filter_applied = True

            if entities["centers"]:
                filtered_snap = filtered_snap[filtered_snap["center"].isin(entities["centers"])]
                if filtered_moves is not None and "center" in filtered_moves.columns:
                    filtered_moves = filtered_moves[filtered_moves["center"].isin(entities["centers"])]
                if filtered_timeline is not None and "center" in filtered_timeline.columns:
                    filtered_timeline = filtered_timeline[filtered_timeline["center"].isin(entities["centers"])]
                filter_msg += f"센터: {', '.join(entities['centers'])} "
                filter_applied = True

            if entities["date_range"] and filtered_moves is not None:
                start_date, end_date = entities["date_range"]
                if "date" in filtered_moves.columns:
                    filtered_moves["date"] = pd.to_datetime(filtered_moves["date"], errors="coerce")
                    filtered_moves = filtered_moves[
                        (filtered_moves["date"] >= start_date) &
                        (filtered_moves["date"] <= end_date)
                    ]
                filter_msg += f"기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}"
                filter_applied = True

            if filter_applied:
                st.info(f"🎯 자동 필터 적용: {filter_msg}")

            # 최소 메타데이터만 준비 (토큰 90% 절약!)
            metadata = prepare_minimal_metadata(filtered_snap, filtered_moves, filtered_timeline)

            # Gemini 2.0 Native Function Calling으로 질문
            answer = ask_ai_with_functions(
                question,
                metadata,
                filtered_snap,
                filtered_moves,
                filtered_timeline
            )

            # 세션에 저장 (필터링된 데이터도 함께)
            st.session_state.last_question = question
            st.session_state.last_answer = answer
            st.session_state.last_metadata = metadata
            st.session_state.last_filtered_snap = filtered_snap
            st.session_state.last_filtered_timeline = filtered_timeline
            st.session_state.last_entities = entities  # 🆕 엔티티 저장 (대화 맥락용)

    # 답변 표시 (세션에서 로드)
    if st.session_state.last_answer:
        st.markdown("### 📊 답변")
        st.markdown(st.session_state.last_answer)

        # 차트 자동 생성 (비활성화)
        # 차트가 효과적이지 않아서 일단 비활성화
        # chart_snap = st.session_state.get("last_filtered_snap", snap)
        # chart_timeline = st.session_state.get("last_filtered_timeline", timeline_df)
        #
        # chart_fig = generate_chart(
        #     st.session_state.last_question,
        #     chart_snap,
        #     moves_df,
        #     chart_timeline
        # )
        # if chart_fig:
        #     st.plotly_chart(chart_fig, use_container_width=True)

        # 후속 질문 제안
        with st.spinner("💡 후속 질문 제안 중..."):
            # 메타데이터를 간략한 텍스트로 변환
            metadata_text = json.dumps(st.session_state.get("last_metadata", {}), ensure_ascii=False, indent=2)[:500]
            followup_questions = suggest_followup_questions(
                st.session_state.last_question,
                st.session_state.last_answer,
                metadata_text
            )

        if followup_questions:
            st.caption("**💬 이런 것도 궁금하신가요?**")
            cols = st.columns(3)
            for i, fq in enumerate(followup_questions):
                with cols[i]:
                    if st.button(fq, key=f"followup_{i}"):
                        st.session_state.pending_question = fq
                        st.rerun()

        # 메타데이터 확인 (디버깅용)
        with st.expander("🔍 AI가 본 메타데이터 (Function Calling)"):
            st.json(st.session_state.get("last_metadata", {}))

    # 예시 질문
    st.divider()
    st.caption("**💡 예시 질문:**")

    col1, col2 = st.columns(2)
    with col1:
        st.caption("**재고 조회**")
        st.caption("• 총 재고는?")
        st.caption("• 센터별 재고는?")
        st.caption("• BA00021은 어느 센터에 있나요?")

    with col2:
        st.caption("**판매/재고 예측 🆕**")
        st.caption("• BA00021 다음주 예상 판매량은?")
        st.caption("• BA00022의 판매 추세는?")
        st.caption("• 다음 2주간 BA00021은 몇개 팔릴까?")
