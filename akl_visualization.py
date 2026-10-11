"""Plotly-Abbildungen: Lernkurve, Abweichungstest, Streuung über Seeds. Alle Achsen gesperrt (Touch-Scrollen)."""

import statistics

import plotly.graph_objects as go

import akl_constants as C

_LAYOUT = dict(margin=dict(l=10, r=10, t=40, b=10), height=380, legend=dict(orientation="h", y=-0.2))


def _lock(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _reference_lines(fig, pn: float, pm: float):
    fig.add_hline(y=pn, line_dash="dash", line_color=C.COLOR_NASH, annotation_text="Nash-Preis", annotation_position="bottom right")
    fig.add_hline(y=pm, line_dash="dash", line_color=C.COLOR_MONOPOLY, annotation_text="Monopolpreis", annotation_position="top right")


def build_learning_curve(trace, pn: float, pm: float) -> go.Figure:
    t = [p[0] for p in trace]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=[p[1] for p in trace], mode="lines", name="mittlerer Preis im Fenster", line=dict(color=C.COLOR_A, width=2),
                             hovertemplate="Periode %{x:,.0f}<br>Preis %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=t, y=[p[2] for p in trace], mode="lines", name="Erkundung ε (rechte Achse)", line=dict(color=C.COLOR_NEUTRAL, width=2), yaxis="y2",
                             hovertemplate="Periode %{x:,.0f}<br>ε %{y:.3f}<extra></extra>"))
    _reference_lines(fig, pn, pm)
    fig.update_layout(**_LAYOUT, title="Lernkurve: mittlerer Preis aller Perioden eines Fensters", xaxis_title="Periode",
                      yaxis=dict(title="Preis", fixedrange=True), yaxis2=dict(title="ε", overlaying="y", side="right", range=[0, 1], fixedrange=True, showgrid=False))
    fig.update_xaxes(fixedrange=True)
    return fig


def build_deviation_chart(dev, pn: float, pm: float) -> go.Figure:
    h = len(dev.baseline)
    x = list(range(h))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=[p[1] for p in dev.baseline], mode="lines+markers", name="Rivale ohne Abweichung", line=dict(color=C.COLOR_NEUTRAL, width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=x, y=[p[1] for p in dev.deviated], mode="lines+markers", name="Rivale nach der Abweichung", line=dict(color=C.COLOR_B, width=3)))
    fig.add_trace(go.Scatter(x=x, y=[p[0] for p in dev.deviated], mode="lines+markers", name="Abweichler", line=dict(color=C.COLOR_A, width=2)))
    _reference_lines(fig, pn, pm)
    fig.update_layout(**_LAYOUT, title="Abweichungstest: Preise nach einer erzwungenen Senkung in Periode 0", xaxis_title="Periode nach der Abweichung", yaxis_title="Preis")
    return _lock(fig)


def build_spread(entries: list[dict], xlabel: str, key: str = "delta", title: str = "Kollusionsindex Δ über 20 Zufalls-Seeds", log_x: bool = False) -> go.Figure:
    """Jeder Punkt ein Lauf, Raute = Median. entries: [{'x': Wert, 'runs': [...], 'label': optionaler Text}]."""
    fig = go.Figure()
    xs, ys = [], []
    for e in entries:
        for i, r in enumerate(e["runs"]):
            xs.append(e["x"])
            ys.append(r[key])
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", name="einzelner Lauf", marker=dict(color=C.COLOR_NEUTRAL, size=7),
                             hovertemplate=f"{xlabel} %{{x}}<br>Δ %{{y:.3f}}<extra></extra>"))
    med = [statistics.median(r[key] for r in e["runs"]) for e in entries]
    fig.add_trace(go.Scatter(x=[e["x"] for e in entries], y=med, mode="lines+markers", name="Median", line=dict(color=C.COLOR_A, width=3), marker=dict(symbol="diamond", size=11)))
    fig.add_hline(y=0, line_dash="dash", line_color=C.COLOR_NASH, annotation_text="Nash", annotation_position="bottom right")
    fig.add_hline(y=1, line_dash="dash", line_color=C.COLOR_MONOPOLY, annotation_text="Monopol", annotation_position="top right")
    fig.update_layout(**_LAYOUT, title=title, xaxis_title=xlabel, yaxis_title="Δ")
    if log_x:
        fig.update_xaxes(type="log")
    return _lock(fig)


def build_deviation_gains(runs: list[dict]) -> go.Figure:
    """Diskontierter Gewinn der Abweichung je Lauf, sortiert; unter null = die Strafe trägt."""
    vals = sorted(r["dev_gain"] for r in runs if r["dev_gain"] is not None)
    colors = [C.COLOR_A if v < 0 else C.COLOR_B for v in vals]
    fig = go.Figure(go.Bar(x=list(range(1, len(vals) + 1)), y=vals, marker_color=colors, hovertemplate="Rang %{x}<br>Gewinn %{y:.3f}<extra></extra>"))
    fig.update_layout(**_LAYOUT, title="Gewinn der Abweichung je Lauf (negativ = Strafe trägt)", xaxis_title="Läufe, sortiert", yaxis_title="diskontierter Gewinn")
    return _lock(fig)
