import pandas as pd

from scm_dashboard_v5.ui import charts


def _capture_plot(monkeypatch):
    captured = {}

    def fake_plotly_chart(fig, *_, **__):
        captured["fig"] = fig

    monkeypatch.setattr(charts.st, "plotly_chart", fake_plotly_chart)
    monkeypatch.setattr(charts.st, "info", lambda *args, **kwargs: None)
    monkeypatch.setattr(charts.st, "warning", lambda *args, **kwargs: None)

    return captured


def _sample_snapshot():
    return pd.DataFrame(
        [
            {"snapshot_date": "2023-01-01", "center": "AMZUS", "resource_code": "SKU1", "stock_qty": 100},
            {"snapshot_date": "2023-01-02", "center": "AMZUS", "resource_code": "SKU1", "stock_qty": 90},
            {"snapshot_date": "2023-01-03", "center": "AMZUS", "resource_code": "SKU1", "stock_qty": 80},
        ]
    )


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


def test_render_amazon_sales_vs_inventory_omits_wip_trace(monkeypatch):
    captured = _capture_plot(monkeypatch)

    charts.render_amazon_sales_vs_inventory(
        snap_long=_sample_snapshot(),
        centers=["AMZUS"],
        skus=["SKU1"],
        start=pd.Timestamp("2023-01-01"),
        end=pd.Timestamp("2023-01-03"),
        today=pd.Timestamp("2023-01-02"),
        color_map={"SKU1": "#123456"},
        show_ma7=False,
        show_inventory_forecast=False,
        timeline=_sample_timeline(),
    )

    fig = captured["fig"]
    assert all("생산중" not in (tr.name or "") for tr in fig.data)
    assert all("WIP" not in (tr.name or "") for tr in fig.data)
