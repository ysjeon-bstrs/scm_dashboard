"""
성능 비교 테스트: 원본 vs 개선 버전

실행 방법:
    python test_performance_comparison.py
"""

import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta


# ==========================================
# 원본 함수 (성능 문제 있음)
# ==========================================
def detect_stockout_risks_original(
    snapshot_df: pd.DataFrame,
    moves_df: pd.DataFrame,
    timeline_df: pd.DataFrame = None,
    days_threshold: int = 7
) -> list:
    """원본 함수 (반복문 사용)"""
    risks = []

    if snapshot_df.empty or moves_df is None or moves_df.empty:
        return risks

    try:
        moves_recent = moves_df.copy()
        if "date" in moves_recent.columns:
            moves_recent["date"] = pd.to_datetime(moves_recent["date"], errors="coerce")
            cutoff_date = moves_recent["date"].max() - pd.Timedelta(days=7)
            moves_recent = moves_recent[moves_recent["date"] >= cutoff_date]

            if "move_type" in moves_recent.columns:
                sales_types = ["CustomerShipment", "출고", "판매"]
                moves_recent = moves_recent[moves_recent["move_type"].isin(sales_types)]

            if "resource_code" in moves_recent.columns and "quantity" in moves_recent.columns:
                daily_sales = moves_recent.groupby("resource_code")["quantity"].sum() / 7

                # ❌ 성능 문제: 반복문에서 DataFrame 필터링
                for sku in daily_sales.index:
                    if daily_sales[sku] <= 0:
                        continue

                    current_stock = snapshot_df[snapshot_df["resource_code"] == sku]["stock_qty"].sum()
                    days_left = current_stock / daily_sales[sku]

                    if 0 < days_left <= days_threshold:
                        risks.append({
                            "sku": sku,
                            "current_stock": current_stock,
                            "daily_sales": daily_sales[sku],
                            "days_left": days_left,
                            "severity": "high" if days_left <= 3 else "medium"
                        })

        risks.sort(key=lambda x: x["days_left"])

    except Exception as e:
        print(f"에러: {e}")

    return risks[:5]


# ==========================================
# 개선 함수 (벡터화)
# ==========================================
def detect_stockout_risks_improved(
    snapshot_df: pd.DataFrame,
    moves_df: pd.DataFrame,
    timeline_df: pd.DataFrame = None,
    days_threshold: int = 7
) -> list:
    """개선 함수 (벡터화)"""
    if snapshot_df is None or snapshot_df.empty or moves_df is None or moves_df.empty:
        return []

    required_snapshot_cols = ["resource_code", "stock_qty"]
    required_moves_cols = ["resource_code", "quantity", "date"]

    if not all(col in snapshot_df.columns for col in required_snapshot_cols):
        return []
    if not all(col in moves_df.columns for col in required_moves_cols):
        return []

    try:
        moves_recent = moves_df.copy()
        moves_recent["date"] = pd.to_datetime(moves_recent["date"], errors="coerce")

        max_date = moves_recent["date"].max()
        if pd.isna(max_date):
            return []

        cutoff_date = max_date - pd.Timedelta(days=7)
        moves_recent = moves_recent[moves_recent["date"] >= cutoff_date]

        if "move_type" in moves_recent.columns:
            sales_types = ["CustomerShipment", "출고", "판매"]
            moves_recent = moves_recent[moves_recent["move_type"].isin(sales_types)]

        if moves_recent.empty:
            return []

        # ✅ 벡터화 연산
        daily_sales = moves_recent.groupby("resource_code")["quantity"].sum() / 7
        daily_sales = daily_sales[daily_sales > 0]

        if daily_sales.empty:
            return []

        # ✅ 벡터화: SKU별 현재 재고
        current_stock = snapshot_df.groupby("resource_code")["stock_qty"].sum()

        # 결합
        stock_analysis = pd.DataFrame({
            "current_stock": current_stock,
            "daily_sales": daily_sales
        }).dropna()

        if stock_analysis.empty:
            return []

        # 벡터화: 품절 일수 계산
        stock_analysis["days_left"] = stock_analysis["current_stock"] / stock_analysis["daily_sales"]

        # 조건 필터링
        at_risk = stock_analysis[
            (stock_analysis["days_left"] > 0) &
            (stock_analysis["days_left"] <= days_threshold)
        ].copy()

        at_risk["severity"] = at_risk["days_left"].apply(
            lambda x: "high" if x <= 3 else "medium"
        )

        at_risk = at_risk.sort_values("days_left").head(5)

        # 결과 변환
        risks = []
        for sku, row in at_risk.iterrows():
            risks.append({
                "sku": str(sku),
                "current_stock": float(row["current_stock"]) if pd.notna(row["current_stock"]) and np.isfinite(row["current_stock"]) else 0.0,
                "daily_sales": float(row["daily_sales"]) if pd.notna(row["daily_sales"]) and np.isfinite(row["daily_sales"]) else 0.0,
                "days_left": float(row["days_left"]) if pd.notna(row["days_left"]) and np.isfinite(row["days_left"]) else 0.0,
                "severity": str(row["severity"])
            })

        return risks

    except Exception as e:
        print(f"에러: {e}")
        return []


# ==========================================
# 테스트 데이터 생성
# ==========================================
def generate_test_data(n_skus: int, n_centers: int, n_days: int = 30):
    """
    대규모 테스트 데이터 생성

    Args:
        n_skus: SKU 개수
        n_centers: 센터 개수
        n_days: 판매 기록 일수
    """
    print(f"📊 테스트 데이터 생성 중... (SKU {n_skus:,}개, 센터 {n_centers}개, {n_days}일)")

    # Snapshot 데이터
    snapshot_data = []
    for sku_id in range(n_skus):
        for center_id in range(n_centers):
            snapshot_data.append({
                "resource_code": f"SKU{sku_id:05d}",
                "center": f"CENTER{center_id:02d}",
                "stock_qty": np.random.randint(10, 500),
                "date": datetime.now()
            })

    snapshot_df = pd.DataFrame(snapshot_data)
    print(f"   ✅ Snapshot: {len(snapshot_df):,}행")

    # Moves 데이터 (최근 30일 판매)
    moves_data = []
    for day_offset in range(n_days):
        date = datetime.now() - timedelta(days=n_days - day_offset)
        # 각 날짜마다 일부 SKU만 판매 (50%)
        for sku_id in np.random.choice(n_skus, size=n_skus // 2, replace=False):
            moves_data.append({
                "resource_code": f"SKU{sku_id:05d}",
                "center": f"CENTER{np.random.randint(0, n_centers):02d}",
                "date": date,
                "quantity": np.random.randint(5, 50),
                "move_type": "CustomerShipment"
            })

    moves_df = pd.DataFrame(moves_data)
    print(f"   ✅ Moves: {len(moves_df):,}행")

    return snapshot_df, moves_df


# ==========================================
# 성능 벤치마크
# ==========================================
def benchmark(func, name, snapshot_df, moves_df, iterations=3):
    """
    함수 성능 측정

    Args:
        func: 테스트할 함수
        name: 함수 이름
        snapshot_df: 재고 데이터
        moves_df: 판매 데이터
        iterations: 반복 횟수
    """
    print(f"\n⏱️  {name} 성능 측정 중... (평균 {iterations}회 반복)")

    times = []
    result = None

    for i in range(iterations):
        start_time = time.time()
        result = func(snapshot_df, moves_df)
        elapsed = time.time() - start_time
        times.append(elapsed)
        print(f"   반복 {i+1}: {elapsed:.4f}초")

    avg_time = np.mean(times)
    std_time = np.std(times)

    print(f"   📊 평균: {avg_time:.4f}초 (±{std_time:.4f})")
    print(f"   📋 결과: {len(result)}개 품절 위험 SKU 발견")

    return avg_time, result


# ==========================================
# 메인 테스트
# ==========================================
def main():
    """성능 비교 테스트 실행"""
    print("=" * 60)
    print("🔬 detect_stockout_risks 성능 비교 테스트")
    print("=" * 60)

    # 테스트 시나리오
    test_scenarios = [
        {"n_skus": 100, "n_centers": 3, "name": "소규모"},
        {"n_skus": 500, "n_centers": 5, "name": "중규모"},
        {"n_skus": 1000, "n_centers": 10, "name": "대규모"},
        # {"n_skus": 5000, "n_centers": 10, "name": "초대규모 (원본은 느림!)"},
    ]

    results = []

    for scenario in test_scenarios:
        print("\n" + "=" * 60)
        print(f"📦 시나리오: {scenario['name']}")
        print("=" * 60)

        # 데이터 생성
        snapshot_df, moves_df = generate_test_data(
            n_skus=scenario["n_skus"],
            n_centers=scenario["n_centers"]
        )

        # 원본 함수 테스트
        time_original, result_original = benchmark(
            detect_stockout_risks_original,
            "원본 (반복문)",
            snapshot_df,
            moves_df,
            iterations=3
        )

        # 개선 함수 테스트
        time_improved, result_improved = benchmark(
            detect_stockout_risks_improved,
            "개선 (벡터화)",
            snapshot_df,
            moves_df,
            iterations=3
        )

        # 성능 향상 계산
        speedup = time_original / time_improved if time_improved > 0 else float('inf')

        print("\n" + "-" * 60)
        print(f"🎯 성능 향상: {speedup:.1f}배 빠름!")
        print(f"   원본: {time_original:.4f}초")
        print(f"   개선: {time_improved:.4f}초")
        print(f"   절약: {time_original - time_improved:.4f}초")
        print("-" * 60)

        results.append({
            "scenario": scenario["name"],
            "n_skus": scenario["n_skus"],
            "time_original": time_original,
            "time_improved": time_improved,
            "speedup": speedup
        })

    # 최종 요약
    print("\n" + "=" * 60)
    print("📊 최종 요약")
    print("=" * 60)

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    print("\n" + "=" * 60)
    print(f"✅ 평균 성능 향상: {results_df['speedup'].mean():.1f}배")
    print(f"🏆 최대 성능 향상: {results_df['speedup'].max():.1f}배 ({results_df.loc[results_df['speedup'].idxmax(), 'scenario']})")
    print("=" * 60)

    # 결과 일치 여부 확인
    print("\n🔍 결과 일치 여부 확인...")
    if len(result_original) == len(result_improved):
        print(f"   ✅ 결과 개수 일치: {len(result_original)}개")
    else:
        print(f"   ⚠️  결과 개수 불일치: 원본 {len(result_original)}개 vs 개선 {len(result_improved)}개")


if __name__ == "__main__":
    main()
