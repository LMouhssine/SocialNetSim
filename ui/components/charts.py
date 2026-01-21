"""Chart components for visualization."""

from typing import Any

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


def create_time_series_chart(
    df: pd.DataFrame,
    x_col: str,
    y_cols: list[str],
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    height: int = 400,
) -> go.Figure:
    """Create a time series line chart.

    Args:
        df: DataFrame with data
        x_col: Column for x-axis
        y_cols: Columns for y-axis (multiple lines)
        title: Chart title
        x_label: X-axis label
        y_label: Y-axis label
        height: Chart height

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    for col in y_cols:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df[x_col],
                y=df[col],
                name=col.replace("_", " ").title(),
                mode="lines",
            ))

    fig.update_layout(
        title=title,
        xaxis_title=x_label or x_col,
        yaxis_title=y_label,
        height=height,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    return fig


def create_distribution_chart(
    values: list[float],
    title: str = "",
    x_label: str = "",
    nbins: int = 30,
    height: int = 350,
    show_mean: bool = True,
) -> go.Figure:
    """Create a histogram/distribution chart.

    Args:
        values: Values to plot
        title: Chart title
        x_label: X-axis label
        nbins: Number of bins
        height: Chart height
        show_mean: Whether to show mean line

    Returns:
        Plotly figure
    """
    fig = px.histogram(
        x=values,
        nbins=nbins,
        title=title,
        labels={"x": x_label, "y": "Count"},
    )

    if show_mean and values:
        import numpy as np
        mean_val = np.mean(values)
        fig.add_vline(
            x=mean_val,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {mean_val:.2f}",
        )

    fig.update_layout(height=height, showlegend=False)
    return fig


def create_comparison_chart(
    data: dict[str, float],
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    height: int = 350,
    orientation: str = "v",
) -> go.Figure:
    """Create a bar chart for comparing values.

    Args:
        data: Dictionary mapping labels to values
        title: Chart title
        x_label: X-axis label
        y_label: Y-axis label
        height: Chart height
        orientation: "v" for vertical, "h" for horizontal

    Returns:
        Plotly figure
    """
    labels = list(data.keys())
    values = list(data.values())

    if orientation == "h":
        fig = px.bar(
            x=values,
            y=labels,
            orientation="h",
            title=title,
            labels={"x": y_label, "y": x_label},
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
    else:
        fig = px.bar(
            x=labels,
            y=values,
            title=title,
            labels={"x": x_label, "y": y_label},
        )

    fig.update_layout(height=height, showlegend=False)
    return fig


def create_multi_metric_chart(
    df: pd.DataFrame,
    x_col: str,
    metrics: list[str],
    title: str = "",
    height: int = 500,
) -> go.Figure:
    """Create a subplot chart with multiple metrics.

    Args:
        df: DataFrame with data
        x_col: Column for x-axis
        metrics: List of metric columns
        title: Overall title
        height: Chart height

    Returns:
        Plotly figure with subplots
    """
    n_metrics = len(metrics)
    rows = (n_metrics + 1) // 2
    cols = min(2, n_metrics)

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[m.replace("_", " ").title() for m in metrics],
    )

    for i, metric in enumerate(metrics):
        row = i // 2 + 1
        col = i % 2 + 1

        if metric in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df[x_col],
                    y=df[metric],
                    name=metric,
                    mode="lines",
                ),
                row=row,
                col=col,
            )

    fig.update_layout(
        title=title,
        height=height,
        showlegend=False,
    )

    return fig


def create_heatmap(
    data: pd.DataFrame,
    title: str = "",
    height: int = 400,
    color_scale: str = "Blues",
) -> go.Figure:
    """Create a heatmap chart.

    Args:
        data: DataFrame where index and columns are categories
        title: Chart title
        height: Chart height
        color_scale: Plotly color scale name

    Returns:
        Plotly figure
    """
    fig = px.imshow(
        data,
        title=title,
        color_continuous_scale=color_scale,
        aspect="auto",
    )

    fig.update_layout(height=height)
    return fig


def create_scatter_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str | None = None,
    size_col: str | None = None,
    title: str = "",
    height: int = 400,
) -> go.Figure:
    """Create a scatter plot.

    Args:
        df: DataFrame with data
        x_col: Column for x-axis
        y_col: Column for y-axis
        color_col: Column for color encoding
        size_col: Column for size encoding
        title: Chart title
        height: Chart height

    Returns:
        Plotly figure
    """
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        size=size_col,
        title=title,
    )

    fig.update_layout(height=height)
    return fig
