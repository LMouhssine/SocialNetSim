"""Network visualization page."""

import streamlit as st
import streamlit.components.v1 as components
import networkx as nx
import numpy as np

from ui.state import SessionState


def render():
    """Render network page."""
    st.title("🕸️ Network Visualization")

    state = SessionState.get()

    if not state.has_world():
        st.warning("Build a world first on the Simulation page")
        return

    world = state.world
    graph = world.graph

    # Network statistics
    st.header("Network Statistics")

    stats = world.network_generator.get_network_statistics()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Nodes", stats.get("num_nodes", 0))
    with col2:
        st.metric("Edges", stats.get("num_edges", 0))
    with col3:
        st.metric("Density", f"{stats.get('density', 0):.4f}")
    with col4:
        st.metric("Reciprocity", f"{stats.get('reciprocity', 0):.2%}")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Avg In-Degree", f"{stats.get('avg_in_degree', 0):.1f}")
    with col2:
        st.metric("Max In-Degree", stats.get("max_in_degree", 0))
    with col3:
        st.metric("Avg Clustering", f"{stats.get('avg_clustering', 0):.3f}" if stats.get("avg_clustering") else "N/A")
    with col4:
        pass

    st.divider()

    # Visualization settings
    st.header("Network Visualization")

    col1, col2 = st.columns(2)

    with col1:
        sample_size = st.slider(
            "Sample size (nodes)",
            min_value=50,
            max_value=min(500, graph.number_of_nodes()),
            value=min(200, graph.number_of_nodes()),
            help="Larger samples may be slow to render",
        )

    with col2:
        color_by = st.selectbox(
            "Color nodes by",
            ["ideology", "influence", "activity", "followers"],
        )

    # Generate visualization
    if st.button("Generate Visualization"):
        with st.spinner("Generating network visualization..."):
            try:
                html_content = create_network_visualization(
                    world, sample_size, color_by
                )
                components.html(html_content, height=600, scrolling=True)
            except Exception as e:
                st.error(f"Error generating visualization: {e}")

    st.divider()

    # Degree distribution
    st.header("Degree Distribution")

    in_degrees = [d for _, d in graph.in_degree()]
    out_degrees = [d for _, d in graph.out_degree()]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("In-Degree Distribution")
        import plotly.express as px
        fig = px.histogram(
            x=in_degrees,
            nbins=30,
            labels={"x": "In-Degree", "y": "Count"},
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Out-Degree Distribution")
        fig = px.histogram(
            x=out_degrees,
            nbins=30,
            labels={"x": "Out-Degree", "y": "Count"},
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # User trait distributions
    st.divider()
    st.header("User Trait Distributions")

    trait_stats = world.user_generator.get_trait_statistics()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Ideology Distribution")
        ideologies = [u.traits.ideology for u in world.users.values()]
        fig = px.histogram(
            x=ideologies,
            nbins=30,
            labels={"x": "Ideology (-1 to 1)", "y": "Count"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Activity Level Distribution")
        activities = [u.traits.activity_level for u in world.users.values()]
        fig = px.histogram(
            x=activities,
            nbins=30,
            labels={"x": "Activity Level (0-1)", "y": "Count"},
        )
        st.plotly_chart(fig, use_container_width=True)


def create_network_visualization(world, sample_size: int, color_by: str) -> str:
    """Create interactive network visualization using pyvis.

    Args:
        world: World object
        sample_size: Number of nodes to sample
        color_by: Attribute to color nodes by

    Returns:
        HTML string for visualization
    """
    from pyvis.network import Network

    graph = world.graph

    # Sample nodes
    all_nodes = list(graph.nodes())
    if len(all_nodes) > sample_size:
        # Preferentially sample high-degree nodes
        degrees = dict(graph.in_degree())
        sorted_nodes = sorted(all_nodes, key=lambda n: degrees[n], reverse=True)
        sampled_nodes = sorted_nodes[:sample_size]
    else:
        sampled_nodes = all_nodes

    # Create subgraph
    subgraph = graph.subgraph(sampled_nodes)

    # Create pyvis network
    net = Network(
        height="550px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#000000",
    )

    # Configure physics
    net.set_options("""
    {
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "centralGravity": 0.01,
                "springLength": 100,
                "springConstant": 0.08
            },
            "maxVelocity": 50,
            "solver": "forceAtlas2Based",
            "timestep": 0.35,
            "stabilization": {
                "enabled": true,
                "iterations": 100
            }
        },
        "nodes": {
            "scaling": {
                "min": 5,
                "max": 30
            }
        },
        "edges": {
            "color": {"inherit": true},
            "smooth": false
        }
    }
    """)

    # Add nodes with colors
    for node in subgraph.nodes():
        user = world.users.get(node)
        if not user:
            continue

        # Determine color
        if color_by == "ideology":
            # Red to blue gradient based on ideology
            val = (user.traits.ideology + 1) / 2  # Normalize to 0-1
            r = int(255 * (1 - val))
            b = int(255 * val)
            color = f"rgb({r}, 100, {b})"
        elif color_by == "influence":
            intensity = int(user.influence_score * 255)
            color = f"rgb(0, {intensity}, {255 - intensity})"
        elif color_by == "activity":
            intensity = int(user.traits.activity_level * 255)
            color = f"rgb({intensity}, {intensity}, 0)"
        else:  # followers
            followers = len(user.followers)
            intensity = min(255, int(np.log1p(followers) * 50))
            color = f"rgb({intensity}, 0, {255 - intensity})"

        # Size based on followers
        size = 5 + np.log1p(len(user.followers)) * 3

        net.add_node(
            node,
            label=node[:10],
            color=color,
            size=size,
            title=f"{node}\nFollowers: {len(user.followers)}\nIdeology: {user.traits.ideology:.2f}",
        )

    # Add edges
    for source, target in subgraph.edges():
        net.add_edge(source, target)

    # Generate HTML
    html = net.generate_html()
    return html
