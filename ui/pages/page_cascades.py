"""Cascade visualization page."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from ui.state import SessionState


def render():
    """Render cascades page."""
    st.title("🌊 Viral Cascades")

    state = SessionState.get()

    if not state.has_simulation():
        st.warning("Run a simulation first on the Simulation page")
        return

    sim = state.simulation
    cascades = list(sim.state.cascades.values())

    if not cascades:
        st.info("No cascades were generated in this simulation")
        return

    # Cascade overview
    st.header("Cascade Overview")

    # Sort by total shares
    cascades_sorted = sorted(cascades, key=lambda c: c.total_shares, reverse=True)

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Cascades", len(cascades))

    with col2:
        active = sum(1 for c in cascades if c.is_active)
        st.metric("Active Cascades", active)

    with col3:
        total_reach = sum(c.total_reach for c in cascades)
        st.metric("Total Reach", total_reach)

    with col4:
        max_shares = max(c.total_shares for c in cascades) if cascades else 0
        st.metric("Max Shares", max_shares)

    st.divider()

    # Top cascades table
    st.header("Top Cascades")

    cascade_data = []
    for c in cascades_sorted[:20]:
        post = sim.state.get_post(c.post_id)
        cascade_data.append({
            "Cascade ID": c.cascade_id,
            "Post ID": c.post_id,
            "Shares": c.total_shares,
            "Reach": c.total_reach,
            "Max Depth": c.max_depth,
            "Peak Velocity": f"{c.peak_velocity:.2f}",
            "Is Misinfo": post.content.is_misinformation if post else False,
            "Status": "Active" if c.is_active else "Inactive",
        })

    df = pd.DataFrame(cascade_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()

    # Cascade analysis
    st.header("Cascade Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Share distribution
        shares = [c.total_shares for c in cascades]
        fig = px.histogram(
            x=shares,
            nbins=30,
            title="Cascade Size Distribution",
            labels={"x": "Total Shares", "y": "Count"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Depth distribution
        depths = [c.max_depth for c in cascades]
        fig = px.histogram(
            x=depths,
            nbins=15,
            title="Cascade Depth Distribution",
            labels={"x": "Max Depth", "y": "Count"},
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Individual cascade explorer
    st.header("Cascade Explorer")

    cascade_options = {
        f"{c.cascade_id} ({c.total_shares} shares)": c.cascade_id
        for c in cascades_sorted[:50]
    }

    selected = st.selectbox(
        "Select a cascade to explore",
        list(cascade_options.keys()),
    )

    if selected:
        cascade_id = cascade_options[selected]
        cascade = sim.state.get_cascade(cascade_id)

        if cascade:
            render_cascade_details(cascade, sim)


def render_cascade_details(cascade, sim):
    """Render detailed view of a cascade."""
    st.subheader(f"Cascade: {cascade.cascade_id}")

    # Basic info
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Shares", cascade.total_shares)
    with col2:
        st.metric("Total Reach", cascade.total_reach)
    with col3:
        st.metric("Max Depth", cascade.max_depth)
    with col4:
        st.metric("Branching Factor", f"{cascade.get_branching_factor():.2f}")

    # Post info
    post = sim.state.get_post(cascade.post_id)
    if post:
        st.markdown("**Original Post:**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.caption(f"Author: {post.author_id}")
        with col2:
            st.caption(f"Quality: {post.content.quality_score:.2f}")
        with col3:
            st.caption(f"Controversy: {post.content.controversy_score:.2f}")
        with col4:
            st.caption(f"Misinfo: {'Yes' if post.content.is_misinformation else 'No'}")

    # Spread over time
    if cascade.shares_by_step:
        st.markdown("**Spread Over Time:**")

        steps = sorted(cascade.shares_by_step.keys())
        shares = [cascade.shares_by_step[s] for s in steps]

        fig = go.Figure()
        fig.add_trace(go.Bar(x=steps, y=shares, name="Shares"))
        fig.update_layout(
            xaxis_title="Step",
            yaxis_title="Shares",
            title="Shares per Step",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Depth distribution
    depth_dist = cascade.get_depth_distribution()
    if depth_dist:
        st.markdown("**Depth Distribution:**")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(depth_dist.keys()),
            y=list(depth_dist.values()),
            name="Nodes at Depth",
        ))
        fig.update_layout(
            xaxis_title="Depth",
            yaxis_title="Nodes",
            title="Nodes by Depth Level",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Cascade tree visualization
    st.markdown("**Cascade Structure:**")

    if cascade.root:
        tree_html = render_cascade_tree(cascade)
        st.markdown(tree_html, unsafe_allow_html=True)
    else:
        st.info("No tree structure available")


def render_cascade_tree(cascade, max_nodes: int = 50) -> str:
    """Render cascade as tree structure.

    Args:
        cascade: Cascade to render
        max_nodes: Maximum nodes to show

    Returns:
        HTML string
    """
    if not cascade.root:
        return "<p>No tree data</p>"

    nodes_shown = [0]  # Use list to allow modification in nested function

    def render_node(node, indent: int = 0) -> str:
        if nodes_shown[0] >= max_nodes:
            return ""

        nodes_shown[0] += 1
        prefix = "&nbsp;" * (indent * 4)

        # Node representation
        html = f"{prefix}├── {node.user_id[:15]} (depth {node.depth}, step {node.step})<br>"

        # Render children
        for child in node.children[:5]:  # Limit children shown
            html += render_node(child, indent + 1)

        if len(node.children) > 5:
            more = len(node.children) - 5
            html += f"{prefix}&nbsp;&nbsp;&nbsp;&nbsp;... and {more} more<br>"

        return html

    html = f"<code style='font-size: 12px;'>"
    html += f"Root: {cascade.root.user_id[:15]} (author)<br>"
    for child in cascade.root.children[:10]:
        html += render_node(child, 1)

    if len(cascade.root.children) > 10:
        more = len(cascade.root.children) - 10
        html += f"... and {more} more direct shares<br>"

    html += "</code>"

    if nodes_shown[0] >= max_nodes:
        html += f"<p><em>Showing first {max_nodes} nodes of {cascade.total_shares + 1}</em></p>"

    return html
