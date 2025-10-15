import pandas as pd

from scm_dashboard_v5.forecast import AmazonForecastContext
from scm_dashboard_v5.forecast import consumption
from scm_dashboard_v5.ui import charts


def _capture_plot(monkeypatch):
    captured = {}

    def fake_plotly_chart(fig, *_, **__):
        captured["fig"] = fig

    monkeypatch.setattr(charts.st, "plotly_chart", fake_plotly_chart)
    monkeypatch.setattr(charts.st, "info", lambda *args, **kwargs: None)
    monkeypatch.setattr(charts.st, "warning", lambda *args, **kwargs: None)

    return captured


def _sample_timeline():
    return pd.DataFrame(
        [
            {"date": "2023-01-01", "center": "AMZUS", "resource_code": "SKU1", "stock_qty": 100},
            {"date": "2023-01-02", "center": "AMZUS", "resource_code": "SKU1", "stock_qty": 90},
            {"date": "2023-01-03", "center": "AMZUS", "resource_code": "SKU1", "stock_qty": 80},
            {"date": "2023-01-01", "center": "WIP", "resource_code": "SKU1", "stock_qty": 20},
            {"date": "2023-01-02", "center": "WIP", "resource_code": "SKU1", "stock_qty": 30},
            {"date": "2023-01-03", "center": "WIP", "resource_code": "SKU1", "stock_qty": 10},
        ]
    )


def _sample_forecast():
    return pd.DataFrame(
        [
            {"date": "2023-01-03", "center": "AMZUS", "resource_code": "SKU1", "stock_qty": 80},
            {"date": "2023-01-04", "center": "AMZUS", "resource_code": "SKU1", "stock_qty": 70},
            {"date": "2023-01-05", "center": "AMZUS", "resource_code": "SKU1", "stock_qty": 60},
        ]
    )


def _sample_daily_sales():
    return pd.DataFrame(
        [
            {"date": "2023-01-01", "center": "AMZUS", "resource_code": "SKU1", "sales_ea": 10},
            {"date": "2023-01-02", "center": "AMZUS", "resource_code": "SKU1", "sales_ea": 10},
        ]
    )


def _sample_sales_forecast():
    return pd.DataFrame(
        [
            {"date": "2023-01-03", "center": "AMZUS", "resource_code": "SKU1", "sales_ea": 9},
            {"date": "2023-01-04", "center": "AMZUS", "resource_code": "SKU1", "sales_ea": 8},
            {"date": "2023-01-05", "center": "AMZUS", "resource_code": "SKU1", "sales_ea": 7},
        ]
    )


def test_load_amazon_sales_normalises_center_alias():
    snapshot_raw = pd.DataFrame(
        [
            {
                "snapshot_date": "2023-01-01",
                "center": "어크로스비US",
                "resource_code": "SKU1",
                "fba_output_stock": 4,
            },
            {
                "snapshot_date": "2023-01-02",
                "center": "AcrossBUS",
                "resource_code": "SKU1",
                "fba_output_stock": 5,
            },
        ]
    )

    result = consumption.load_amazon_daily_sales_from_snapshot_raw(
        snapshot_raw,
        centers=("AcrossBUS",),
        skus=("SKU1",),
    )

    assert not result.empty
    assert sorted(result["center"].unique().tolist()) == ["AcrossBUS"]
    assert result["sales_ea"].sum() == 9


def test_render_step_chart_adds_wip_trace(monkeypatch):
    captured = _capture_plot(monkeypatch)

    charts.render_step_chart(
        timeline=_sample_timeline(),
        centers=["AMZUS", "태광KR"],
        skus=["SKU1"],
        start=pd.Timestamp("2023-01-01"),
        end=pd.Timestamp("2023-01-03"),
        today=pd.Timestamp("2023-01-02"),
        show_production=True,
    )

    fig = captured["fig"]
    trace_names = [tr.name for tr in fig.data]
    assert "SKU1 태광KR 생산중" in trace_names

    wip_trace = next(tr for tr in fig.data if tr.name == "SKU1 태광KR 생산중")
    assert [round(v) for v in wip_trace.y] == [20, 30, 10]
    assert all("WIP" not in (name or "") for name in trace_names)


def test_render_step_chart_hides_wip_trace(monkeypatch):
    captured = _capture_plot(monkeypatch)

    charts.render_step_chart(
        timeline=_sample_timeline(),
        centers=["AMZUS", "태광KR"],
        skus=["SKU1"],
        start=pd.Timestamp("2023-01-01"),
        end=pd.Timestamp("2023-01-03"),
        today=pd.Timestamp("2023-01-02"),
        show_wip=False,
    )

    fig = captured["fig"]
    trace_names = [tr.name for tr in fig.data]
    assert "SKU1 태광KR 생산중" not in trace_names
    assert all("WIP" not in (name or "") for name in trace_names)


def test_render_step_chart_accepts_legacy_wip_center(monkeypatch):
    captured = _capture_plot(monkeypatch)

    charts.render_step_chart(
        timeline=_sample_timeline(),
        centers=["AMZUS", "WIP"],
        skus=["SKU1"],
        start=pd.Timestamp("2023-01-01"),
        end=pd.Timestamp("2023-01-03"),
        today=pd.Timestamp("2023-01-02"),
        show_wip=True,
    )

    fig = captured["fig"]
    trace_names = [tr.name for tr in fig.data]
    assert "SKU1 태광KR 생산중" in trace_names
    assert all("WIP" not in (name or "") for name in trace_names)


def test_render_amazon_panel_renders_actual_and_forecast(monkeypatch):
    captured = _capture_plot(monkeypatch)

    ctx = AmazonForecastContext(
        start=pd.Timestamp("2023-01-01"),
        end=pd.Timestamp("2023-01-05"),
        today=pd.Timestamp("2023-01-02"),
        centers=["AMZUS"],
        skus=["SKU1"],
        inv_actual=pd.DataFrame(
            [
                {"date": "2023-01-01", "center": "AMZUS", "resource_code": "SKU1", "stock_qty": 100},
                {"date": "2023-01-02", "center": "AMZUS", "resource_code": "SKU1", "stock_qty": 90},
            ]
        ),
        inv_forecast=_sample_forecast(),
        sales_hist=_sample_daily_sales(),
        sales_ma7=pd.DataFrame(
            [
                {"date": "2023-01-01", "center": "AMZUS", "resource_code": "SKU1", "sales_ea": 10.0},
                {"date": "2023-01-02", "center": "AMZUS", "resource_code": "SKU1", "sales_ea": 10.0},
            ]
        ),
        sales_forecast=_sample_sales_forecast(),
    )

    charts.render_amazon_sales_vs_inventory(ctx)

    fig = captured["fig"]
    trace_names = [tr.name for tr in fig.data]
    assert "SKU1 재고(실측)" in trace_names
    assert "SKU1 재고(예측)" in trace_names

    sales_traces = [tr for tr in fig.data if "판매(" in (tr.name or "")]
    assert sales_traces, "판매 막대가 렌더링되어야 합니다."
    for tr in sales_traces:
        assert all(float(v).is_integer() for v in tr.y), "판매 막대는 정수여야 합니다."


def test_build_amazon_context_skips_forecast_when_disabled(monkeypatch):
    timeline = pd.DataFrame(
        [
            {"date": "2023-01-01", "center": "AMZUS", "resource_code": "SKU1", "stock_qty": 100},
            {"date": "2023-01-02", "center": "AMZUS", "resource_code": "SKU1", "stock_qty": 90},
            {"date": "2023-01-03", "center": "AMZUS", "resource_code": "SKU1", "stock_qty": 80},
            {"date": "2023-01-04", "center": "AMZUS", "resource_code": "SKU1", "stock_qty": 70},
        ]
    )

    monkeypatch.setattr(
        consumption,
        "build_timeline",
        lambda *args, **kwargs: timeline.copy(),
    )

    apply_called = {"flag": False}

    def fake_apply(*args, **kwargs):
        apply_called["flag"] = True
        return pd.DataFrame(columns=["date", "center", "resource_code", "stock_qty"])

    monkeypatch.setattr(consumption, "apply_consumption_with_events", fake_apply)

    snap_long = pd.DataFrame(
        [
            {
                "snapshot_date": "2023-01-02",
                "center": "AMZUS",
                "resource_code": "SKU1",
                "stock_qty": 90,
            }
        ]
    )

    snapshot_raw = pd.DataFrame(
        [
            {
                "snapshot_date": "2023-01-01",
                "center": "AMZUS",
                "resource_code": "SKU1",
                "fba_output_stock": 5,
            },
            {
                "snapshot_date": "2023-01-02",
                "center": "AMZUS",
                "resource_code": "SKU1",
                "fba_output_stock": 6,
            },
        ]
    )

    ctx = consumption.build_amazon_forecast_context(
        snap_long=snap_long,
        moves=pd.DataFrame(),
        snapshot_raw=snapshot_raw,
        centers=["AMZUS"],
        skus=["SKU1"],
        start=pd.Timestamp("2023-01-01"),
        end=pd.Timestamp("2023-01-05"),
        today=pd.Timestamp("2023-01-02"),
        lookback_days=7,
        promotion_events=None,
        use_consumption_forecast=False,
    )

    assert apply_called["flag"] is False
    assert ctx.inv_forecast.empty
    assert ctx.sales_forecast.empty
    assert not ctx.inv_actual.empty
