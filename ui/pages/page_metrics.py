"""Metrics dashboard page."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ui.state import SessionState


def render():
    """Render metrics page."""
    st.title("📊 Simulation Metrics")

    state = SessionState.get()

    if not state.has_simulation():
        st.warning("Run a simulation first on the Simulation page")
        return

    sim = state.simulation
    results = sim.get_results()
    metrics_df = sim.get_metrics_dataframe()

    # Summary metrics
    st.header("Summary Metrics")

    summary = results.get("metrics_summary", {})

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Steps", summary.get("total_steps", 0))
        st.metric("Total Posts", summary.get("total_posts", 0))

    with col2:
        st.metric("Total Interactions", summary.get("total_interactions", 0))
        st.metric("Peak Interactions", summary.get("peak_interactions", 0))

    with col3:
        st.metric("Engagement Rate", f"{summary.get('engagement_rate', 0):.2%}")
        st.metric("Share Rate", f"{summary.get('share_rate', 0):.2%}")

    with col4:
        st.metric("Misinfo Posts", summary.get("total_misinfo_posts", 0))
        st.metric("Misinfo Share Rate", f"{summary.get('misinfo_share_rate', 0):.2%}")

    st.divider()

    # Time series charts
    st.header("Engagement Over Time")

    # Engagement chart
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Views", "Likes", "Shares", "Comments"),
    )

    fig.add_trace(
        go.Scatter(x=metrics_df["step"], y=metrics_df["views"], name="Views"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=metrics_df["step"], y=metrics_df["likes"], name="Likes"),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(x=metrics_df["step"], y=metrics_df["shares"], name="Shares"),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=metrics_df["step"], y=metrics_df["comments"], name="Comments"),
        row=2, col=2,
    )

    fig.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Active users and posts
    st.header("Activity Metrics")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.line(
            metrics_df,
            x="step",
            y="active_users",
            title="Active Users per Step",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.line(
            metrics_df,
            x="step",
            y="new_posts",
            title="New Posts per Step",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Cascade metrics
    st.header("Cascade Metrics")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.line(
            metrics_df,
            x="step",
            y="active_cascades",
            title="Active Cascades",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.line(
            metrics_df,
            x="step",
            y="total_cascade_reach",
            title="Total Cascade Reach",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Misinformation metrics
    st.header("Misinformation Metrics")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.line(
            metrics_df,
            x="step",
            y="misinfo_posts",
            title="New Misinformation Posts",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.line(
            metrics_df,
            x="step",
            y="misinfo_shares",
            title="Misinformation Shares",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Moderation metrics
    st.header("Moderation Statistics")

    mod_stats = results.get("moderation_stats", {})

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Flagged", mod_stats.get("total_flagged", 0))

    with col2:
        st.metric("Suppressed", mod_stats.get("total_suppressed", 0))

    with col3:
        st.metric("Removed", mod_stats.get("total_removed", 0))

    with col4:
        st.metric("False Positives", mod_stats.get("false_positives", 0))

    # Moderation actions over time
    if "moderation_actions" in metrics_df.columns:
        fig = px.line(
            metrics_df,
            x="step",
            y="moderation_actions",
            title="Moderation Actions per Step",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Events
    st.header("Event Statistics")

    event_stats = results.get("event_stats", {})

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Events", event_stats.get("total_events", 0))
        st.metric("Active Events", event_stats.get("active_events", 0))

    with col2:
        st.metric("Avg Duration", f"{event_stats.get('average_duration', 0):.1f} steps")

    # Events by type
    events_by_type = event_stats.get("events_by_type", {})
    if events_by_type:
        fig = px.bar(
            x=list(events_by_type.keys()),
            y=list(events_by_type.values()),
            title="Events by Type",
            labels={"x": "Event Type", "y": "Count"},
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Raw data table
    st.header("Raw Metrics Data")

    with st.expander("View raw metrics table"):
        st.dataframe(metrics_df, use_container_width=True)

    # Download button
    csv = metrics_df.to_csv(index=False)
    st.download_button(
        "Download Metrics CSV",
        csv,
        "simulation_metrics.csv",
        "text/csv",
    )


if __name__ == "__main__":
    render()
