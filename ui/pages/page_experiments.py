"""Experiments page for what-if scenarios."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from config.schemas import SimulationConfig
from scenarios import (
    create_algorithm_comparison,
    create_moderation_impact_study,
    create_echo_chamber_study,
    create_virality_analysis,
)
from scenarios.comparator import ExperimentComparator


def render():
    """Render experiments page."""
    st.title("🧪 What-If Experiments")

    st.markdown("""
    Run controlled experiments to compare different configurations and
    understand how changes affect network dynamics.
    """)

    # Experiment selection
    st.header("Select Experiment")

    experiment_types = {
        "Algorithm Comparison": "Compare feed algorithm performance",
        "Moderation Impact": "Study how moderation affects misinformation",
        "Echo Chamber Formation": "Analyze polarization dynamics",
        "Virality Analysis": "Compare cascade spreading patterns",
    }

    selected_exp = st.selectbox(
        "Experiment type",
        list(experiment_types.keys()),
    )

    st.caption(experiment_types[selected_exp])

    st.divider()

    # Configuration
    st.header("Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        num_users = st.slider(
            "Users",
            min_value=100,
            max_value=2000,
            value=500,
            step=100,
        )

    with col2:
        num_steps = st.slider(
            "Steps",
            min_value=20,
            max_value=200,
            value=50,
            step=10,
        )

    with col3:
        num_runs = st.slider(
            "Runs per variation",
            min_value=1,
            max_value=5,
            value=1,
            help="More runs = better statistics but slower",
        )

    st.divider()

    # Run experiment
    st.header("Run Experiment")

    if "experiment_results" not in st.session_state:
        st.session_state.experiment_results = None

    if st.button("🚀 Run Experiment", type="primary"):
        with st.spinner("Running experiment..."):
            try:
                # Create experiment
                base_config = SimulationConfig(
                    seed=42,
                    num_steps=num_steps,
                )
                base_config.user.num_users = num_users

                if selected_exp == "Algorithm Comparison":
                    experiment = create_algorithm_comparison(
                        base_config, num_steps, num_users, num_runs
                    )
                elif selected_exp == "Moderation Impact":
                    experiment = create_moderation_impact_study(
                        base_config, num_steps, num_users, num_runs
                    )
                elif selected_exp == "Echo Chamber Formation":
                    experiment = create_echo_chamber_study(
                        base_config, num_steps, num_users, num_runs
                    )
                else:
                    experiment = create_virality_analysis(
                        base_config, num_steps, num_users, num_runs
                    )

                # Progress tracking
                progress = st.progress(0)
                status = st.empty()

                def update_progress(var_name, run_idx, total):
                    progress.progress((run_idx + 1) / total)
                    status.text(f"Running: {var_name} (run {run_idx + 1})")

                experiment.set_progress_callback(update_progress)

                # Run
                results = experiment.run(share_world=True)

                progress.progress(1.0)
                status.text("Complete!")

                st.session_state.experiment_results = results
                st.success("Experiment completed!")

            except Exception as e:
                st.error(f"Error running experiment: {e}")
                raise e

    # Display results
    if st.session_state.experiment_results is not None:
        st.divider()
        st.header("Results")

        results = st.session_state.experiment_results
        comparator = ExperimentComparator()

        # Comparison table
        st.subheader("Variation Comparison")

        if results.comparison_summary is not None:
            st.dataframe(results.comparison_summary, use_container_width=True)

        # Generate comparison
        comparison_df = comparator.compare_variations(results)

        # Key metrics visualization
        st.subheader("Key Metrics by Variation")

        key_metrics = ["total_interactions", "engagement_rate", "misinfo_share_rate"]

        for metric in key_metrics:
            value_col = f"{metric}_value"
            if value_col in comparison_df.columns:
                fig = px.bar(
                    comparison_df,
                    x="variation",
                    y=value_col,
                    title=f"{metric.replace('_', ' ').title()}",
                )
                st.plotly_chart(fig, use_container_width=True)

        # Time series comparison
        st.subheader("Engagement Over Time")

        time_series = comparator.get_time_series_comparison(results, "new_interactions")
        if not time_series.empty:
            fig = go.Figure()
            for col in time_series.columns:
                fig.add_trace(go.Scatter(
                    x=time_series.index,
                    y=time_series[col],
                    name=col,
                    mode="lines",
                ))
            fig.update_layout(
                xaxis_title="Step",
                yaxis_title="Interactions",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Summary report
        st.subheader("Summary Report")

        report = comparator.generate_summary_report(results)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Winners by Metric:**")
            for metric, winner in report.get("winners_by_metric", {}).items():
                st.write(f"- {metric}: **{winner}**")

        with col2:
            st.markdown("**Key Findings:**")
            for finding in report.get("key_findings", []):
                st.write(f"- {finding}")

        # Effect sizes
        st.subheader("Effect Sizes vs Baseline")

        effect_df = comparator.calculate_effect_sizes(results)
        if not effect_df.empty:
            st.dataframe(effect_df, use_container_width=True)

        st.divider()

        # Download results
        st.subheader("Export Results")

        if results.comparison_summary is not None:
            csv = results.comparison_summary.to_csv(index=False)
            st.download_button(
                "Download Comparison CSV",
                csv,
                "experiment_comparison.csv",
                "text/csv",
            )


if __name__ == "__main__":
    render()
