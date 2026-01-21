"""Simulation configuration and control page."""

import streamlit as st

from config.schemas import SimulationConfig, load_scenario
from generator import World
from engine import Simulation
from ui.state import SessionState, get_config, set_config, set_world, set_simulation


def render():
    """Render simulation page."""
    st.title("🎮 Simulation Control")

    state = SessionState.get()

    # Configuration section
    st.header("Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Scenario")
        scenario = st.selectbox(
            "Select scenario",
            ["default", "echo_chamber", "misinformation", "custom"],
            help="Pre-configured scenarios for different research questions",
        )

        if scenario != "custom":
            if st.button("Load Scenario"):
                try:
                    config = load_scenario(scenario)
                    set_config(config)
                    st.success(f"Loaded scenario: {scenario}")
                except Exception as e:
                    st.error(f"Error loading scenario: {e}")

    with col2:
        st.subheader("Quick Settings")
        config = get_config()

        num_users = st.slider(
            "Number of users",
            min_value=100,
            max_value=10000,
            value=config.user.num_users,
            step=100,
        )

        num_steps = st.slider(
            "Simulation steps",
            min_value=10,
            max_value=500,
            value=config.num_steps,
            step=10,
        )

        seed = st.number_input(
            "Random seed (empty for random)",
            min_value=0,
            max_value=99999,
            value=config.seed or 42,
        )

    # Advanced configuration
    with st.expander("Advanced Configuration"):
        tab1, tab2, tab3, tab4 = st.tabs(["Network", "Content", "Feed", "Moderation"])

        with tab1:
            st.markdown("**Network Generation (Barabasi-Albert)**")
            edges_per_node = st.slider(
                "Edges per new node",
                min_value=1,
                max_value=10,
                value=config.network.edges_per_new_node,
            )
            weight_degree = st.slider(
                "Degree weight",
                min_value=0.0,
                max_value=1.0,
                value=config.network.weight_degree,
            )
            weight_interest = st.slider(
                "Interest similarity weight",
                min_value=0.0,
                max_value=1.0,
                value=config.network.weight_interest,
            )
            weight_ideology = st.slider(
                "Ideology proximity weight",
                min_value=0.0,
                max_value=1.0,
                value=config.network.weight_ideology,
            )

        with tab2:
            st.markdown("**Content Generation**")
            misinfo_rate = st.slider(
                "Misinformation rate",
                min_value=0.0,
                max_value=0.3,
                value=config.content.misinformation_rate,
                format="%.2f",
            )
            avg_posts = st.slider(
                "Avg posts per user per step",
                min_value=0.01,
                max_value=0.5,
                value=config.content.avg_posts_per_step,
                format="%.2f",
            )

        with tab3:
            st.markdown("**Feed Algorithm**")
            algorithm = st.selectbox(
                "Feed algorithm",
                ["chronological", "engagement", "diverse", "interest"],
                index=["chronological", "engagement", "diverse", "interest"].index(
                    config.feed.algorithm
                ),
            )
            feed_size = st.slider(
                "Feed size",
                min_value=5,
                max_value=50,
                value=config.feed.feed_size,
            )

        with tab4:
            st.markdown("**Content Moderation**")
            mod_enabled = st.checkbox(
                "Enable moderation",
                value=config.moderation.enabled,
            )
            if mod_enabled:
                detection_accuracy = st.slider(
                    "Detection accuracy",
                    min_value=0.0,
                    max_value=1.0,
                    value=config.moderation.detection_accuracy,
                )
                suppression = st.slider(
                    "Suppression factor",
                    min_value=0.0,
                    max_value=1.0,
                    value=config.moderation.suppression_factor,
                )

    # Apply configuration button
    if st.button("Apply Configuration", type="primary"):
        # Create updated config
        new_config = SimulationConfig(
            name=f"custom_{scenario}",
            seed=seed if seed > 0 else None,
            num_steps=num_steps,
        )
        new_config.user.num_users = num_users
        new_config.network.edges_per_new_node = edges_per_node
        new_config.network.weight_degree = weight_degree
        new_config.network.weight_interest = weight_interest
        new_config.network.weight_ideology = weight_ideology
        new_config.content.misinformation_rate = misinfo_rate
        new_config.content.avg_posts_per_step = avg_posts
        new_config.feed.algorithm = algorithm
        new_config.feed.feed_size = feed_size
        new_config.moderation.enabled = mod_enabled

        set_config(new_config)
        st.success("Configuration applied!")

    st.divider()

    # Simulation control
    st.header("Simulation Control")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🏗️ Build World", disabled=state.is_running):
            with st.spinner("Building synthetic world..."):
                try:
                    config = get_config()
                    world = World(config)
                    world.build()
                    set_world(world)
                    st.success(
                        f"World built: {len(world.users)} users, "
                        f"{world.graph.number_of_edges()} connections"
                    )
                except Exception as e:
                    st.error(f"Error building world: {e}")

    with col2:
        can_run = state.has_world() and not state.is_running
        if st.button("▶️ Run Simulation", disabled=not can_run):
            with st.spinner("Running simulation..."):
                try:
                    config = get_config()
                    sim = Simulation(config)
                    sim.initialize(state.world)

                    # Progress tracking
                    progress = st.progress(0)
                    status = st.empty()

                    def update_progress(step, metrics):
                        progress.progress(step / config.num_steps)
                        status.text(f"Step {step}: {metrics.new_interactions} interactions")

                    sim.add_step_callback(update_progress)

                    state.is_running = True
                    results = sim.run(show_progress=False)
                    state.is_running = False

                    set_simulation(sim)
                    state.run_history.append(results)

                    progress.progress(1.0)
                    status.text("Simulation complete!")
                    st.success("Simulation completed successfully!")

                except Exception as e:
                    state.is_running = False
                    st.error(f"Error running simulation: {e}")

    with col3:
        if st.button("🔄 Reset", disabled=state.is_running):
            state.reset()
            st.info("Session reset")
            st.rerun()

    # Status display
    st.divider()
    st.header("Status")

    status_col1, status_col2, status_col3 = st.columns(3)

    with status_col1:
        st.metric(
            "World Status",
            "Built" if state.has_world() else "Not Built",
        )
        if state.has_world():
            st.caption(f"{len(state.world.users)} users")

    with status_col2:
        st.metric(
            "Simulation Status",
            "Complete" if state.has_simulation() else "Not Run",
        )
        if state.has_simulation():
            st.caption(f"Step {state.simulation.state.current_step}")

    with status_col3:
        st.metric(
            "Run History",
            f"{len(state.run_history)} runs",
        )

    # Quick results preview
    if state.has_simulation():
        st.divider()
        st.header("Quick Results")

        results = state.get_results()
        if results:
            metrics = results.get("metrics_summary", {})

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Posts", metrics.get("total_posts", 0))
            with col2:
                st.metric("Total Interactions", metrics.get("total_interactions", 0))
            with col3:
                st.metric(
                    "Engagement Rate",
                    f"{metrics.get('engagement_rate', 0):.2%}",
                )
            with col4:
                st.metric(
                    "Misinfo Share Rate",
                    f"{metrics.get('misinfo_share_rate', 0):.2%}",
                )
