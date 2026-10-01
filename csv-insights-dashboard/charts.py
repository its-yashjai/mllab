"""Plotly figures for the dashboard, styled to read well in light and dark mode."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

# Every chart here shows one series, so it uses a single blue; the dark-mode
# step is a little lighter so it holds contrast on a dark background.
SERIES = {"light": "#2a78d6", "dark": "#3987e5"}
GRID = {"light": "#e1e0d9", "dark": "#2c2c2a"}

# Correlation runs -1..1: red for "move in opposite directions", blue for
# "move together", neutral gray for "no relationship".
DIVERGING_MID = {"light": "#f0efec", "dark": "#383835"}
RED_DARK, RED_LIGHT = "#b8312f", "#f0a3a2"
BLUE_LIGHT, BLUE_DARK = "#86b6ef", "#1c5cab"


def _layout(fig: go.Figure, mode: str, height: int = 360) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        showlegend=False,
        barcornerradius=4,
        hoverlabel=dict(font_size=13),
    )
    fig.update_xaxes(gridcolor=GRID[mode], zeroline=False)
    fig.update_yaxes(gridcolor=GRID[mode], zeroline=False)
    return fig


def histogram(values: pd.Series, mode: str = "light") -> go.Figure:
    fig = go.Figure(
        go.Histogram(
            x=values.dropna(),
            marker_color=SERIES[mode],
            hovertemplate=f"{values.name}: %{{x}}<br>Rows: %{{y:,}}<extra></extra>",
        )
    )
    fig.update_layout(bargap=0.06)
    fig.update_xaxes(title_text=str(values.name))
    fig.update_yaxes(title_text="Rows")
    return _layout(fig, mode)


def ranked_bars(table: pd.DataFrame, label_col: str, value_col: str, mode: str = "light") -> go.Figure:
    """Horizontal bars, largest at the top. Expects `table` sorted largest first."""
    labels = table[label_col].astype(str)
    fig = go.Figure(
        go.Bar(
            x=table[value_col],
            y=labels,
            orientation="h",
            marker_color=SERIES[mode],
            hovertemplate="%{y}<br>" + f"{value_col}: " + "%{x:,.4~g}<extra></extra>",
        )
    )
    fig.update_layout(bargap=0.25)
    fig.update_yaxes(autorange="reversed", title_text=None, automargin=True)
    fig.update_xaxes(title_text=value_col)
    return _layout(fig, mode, height=max(220, 36 * len(table) + 80))


def trend_line(table: pd.DataFrame, date_col: str, value_col: str, mode: str = "light") -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=table[date_col],
            y=table[value_col],
            mode="lines+markers",
            line=dict(color=SERIES[mode], width=2),
            marker=dict(size=8, color=SERIES[mode]),
            hovertemplate="%{x|%d %b %Y}<br>" + f"{value_col}: " + "%{y:,.4~g}<extra></extra>",
        )
    )
    fig.update_layout(hovermode="x")
    fig.update_xaxes(showspikes=True, spikemode="across", spikethickness=1, spikedash="solid", title_text=None)
    fig.update_yaxes(title_text=value_col, rangemode="tozero")
    return _layout(fig, mode)


def correlation_heatmap(corr: pd.DataFrame, mode: str = "light") -> go.Figure:
    labels = [str(c) for c in corr.columns]
    show_numbers = len(labels) <= 10
    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=labels,
            y=labels,
            zmin=-1,
            zmax=1,
            colorscale=[
                [0.0, RED_DARK],
                [0.25, RED_LIGHT],
                [0.5, DIVERGING_MID[mode]],
                [0.75, BLUE_LIGHT],
                [1.0, BLUE_DARK],
            ],
            xgap=2,
            ygap=2,
            texttemplate="%{z:.2f}" if show_numbers else None,
            hovertemplate="%{y} vs %{x}<br>Correlation: %{z:.2f}<extra></extra>",
            colorbar=dict(title=dict(text="Correlation"), tickvals=[-1, -0.5, 0, 0.5, 1], thickness=12),
        )
    )
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, tickangle=-30)
    return _layout(fig, mode, height=max(320, 42 * len(labels) + 120))
