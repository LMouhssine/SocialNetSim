"""AI models training and evaluation page."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from ui.state import SessionState
from ai import ViralityPredictor, ChurnPredictor, MisinfoDetector, ModelEvaluator


def render():
    """Render AI models page."""
    st.title("🤖 AI Models")

    state = SessionState.get()

    if not state.has_simulation():
        st.warning("Run a simulation first to generate training data")
        return

    sim = state.simulation

    st.markdown("""
    Train and evaluate machine learning models on simulation data.
    These models can predict virality, user churn, and misinformation.
    """)

    # Model selection
    st.header("Model Training")

    model_type = st.selectbox(
        "Select model to train",
        ["Virality Predictor", "Churn Predictor", "Misinformation Detector"],
    )

    # Initialize session state for models
    if "trained_models" not in st.session_state:
        st.session_state.trained_models = {}

    st.divider()

    # Training configuration
    col1, col2 = st.columns(2)

    with col1:
        algorithm = st.selectbox(
            "Algorithm",
            ["xgboost", "random_forest"],
        )

    with col2:
        n_estimators = st.slider(
            "Number of estimators",
            min_value=50,
            max_value=200,
            value=100,
            step=25,
        )

    # Train button
    if st.button("🎯 Train Model", type="primary"):
        with st.spinner(f"Training {model_type}..."):
            try:
                if model_type == "Virality Predictor":
                    model = train_virality_predictor(sim, algorithm, n_estimators)
                elif model_type == "Churn Predictor":
                    model = train_churn_predictor(sim, algorithm, n_estimators)
                else:
                    model = train_misinfo_detector(sim, algorithm, n_estimators)

                st.session_state.trained_models[model_type] = model
                st.success(f"{model_type} trained successfully!")

            except Exception as e:
                st.error(f"Error training model: {e}")

    # Display trained model results
    if model_type in st.session_state.trained_models:
        st.divider()
        st.header("Model Performance")

        model = st.session_state.trained_models[model_type]
        metrics = model.training_metrics

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)

        if model.task_type == "classification":
            with col1:
                st.metric("Test Accuracy", f"{metrics.get('test_accuracy', 0):.2%}")
            with col2:
                st.metric("Test Precision", f"{metrics.get('test_precision', 0):.2%}")
            with col3:
                st.metric("Test Recall", f"{metrics.get('test_recall', 0):.2%}")
            with col4:
                st.metric("Test F1", f"{metrics.get('test_f1', 0):.2%}")

            # AUC if available
            if "test_auc" in metrics:
                st.metric("Test AUC", f"{metrics['test_auc']:.3f}")
        else:
            with col1:
                st.metric("Test RMSE", f"{metrics.get('test_rmse', 0):.3f}")
            with col2:
                st.metric("Test MAE", f"{metrics.get('test_mae', 0):.3f}")
            with col3:
                st.metric("Test R²", f"{metrics.get('test_r2', 0):.3f}")

        st.divider()

        # Feature importance
        st.header("Feature Importance")

        importance = model.get_feature_importance()
        if importance:
            # Sort by importance
            sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            top_features = sorted_imp[:15]

            fig = px.bar(
                x=[f[1] for f in top_features],
                y=[f[0] for f in top_features],
                orientation="h",
                title="Top 15 Features by Importance",
                labels={"x": "Importance", "y": "Feature"},
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # Model-specific analysis
        st.header("Model Analysis")

        if model_type == "Virality Predictor":
            render_virality_analysis(model, sim)
        elif model_type == "Churn Predictor":
            render_churn_analysis(model, sim)
        else:
            render_misinfo_analysis(model, sim)


def train_virality_predictor(sim, algorithm: str, n_estimators: int):
    """Train virality predictor."""
    posts = list(sim.state.posts.values())

    model = ViralityPredictor(
        viral_threshold=50,
        model_type=algorithm,
        n_estimators=n_estimators,
    )

    model.train_from_simulation(posts, sim.world.users, sim.state)
    return model


def train_churn_predictor(sim, algorithm: str, n_estimators: int):
    """Train churn predictor."""
    model = ChurnPredictor(
        churn_threshold_steps=10,
        model_type=algorithm,
        n_estimators=n_estimators,
    )

    model.train_from_simulation(sim.world.users, sim.state)
    return model


def train_misinfo_detector(sim, algorithm: str, n_estimators: int):
    """Train misinformation detector."""
    posts = list(sim.state.posts.values())

    model = MisinfoDetector(
        model_type=algorithm,
        n_estimators=n_estimators,
    )

    model.train_from_simulation(posts, sim.world.users, sim.state)
    return model


def render_virality_analysis(model, sim):
    """Render virality-specific analysis."""
    st.subheader("Virality Predictions")

    # Get predictions for recent posts
    posts = list(sim.state.posts.values())[-100:]  # Last 100 posts
    predictions_df = model.predict_batch(posts, sim.world.users, sim.state)

    # Show top predicted viral posts
    st.markdown("**Top Predicted Viral Posts:**")
    top_viral = predictions_df.nlargest(10, "viral_probability")
    st.dataframe(top_viral, use_container_width=True, hide_index=True)

    # Prediction distribution
    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            predictions_df,
            x="viral_probability",
            nbins=20,
            title="Virality Probability Distribution",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Actual vs predicted
        viral_counts = predictions_df["viral_prediction"].value_counts().reindex(
            [0, 1], fill_value=0
        )
        fig = px.pie(
            values=viral_counts.values,
            names=["Not Viral", "Viral"],
            title="Predicted Viral vs Not Viral",
        )
        st.plotly_chart(fig, use_container_width=True)


def render_churn_analysis(model, sim):
    """Render churn-specific analysis."""
    st.subheader("Churn Predictions")

    # Get predictions
    predictions_df = model.predict_batch(sim.world.users, sim.state)

    # Show at-risk users
    st.markdown("**Users at High Risk of Churning:**")
    at_risk = predictions_df[predictions_df["churn_probability"] > 0.7]
    st.dataframe(at_risk.head(10), use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            predictions_df,
            x="churn_probability",
            nbins=20,
            title="Churn Probability Distribution",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Risk level breakdown
        def get_risk_level(p):
            if p >= 0.8:
                return "Critical"
            elif p >= 0.6:
                return "High"
            elif p >= 0.4:
                return "Moderate"
            else:
                return "Low"

        predictions_df["risk_level"] = predictions_df["churn_probability"].apply(get_risk_level)
        risk_counts = predictions_df["risk_level"].value_counts()

        fig = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title="User Risk Level Distribution",
        )
        st.plotly_chart(fig, use_container_width=True)


def render_misinfo_analysis(model, sim):
    """Render misinformation detection analysis."""
    st.subheader("Misinformation Detection")

    posts = list(sim.state.posts.values())
    predictions_df = model.detect_batch(posts, sim.world.users, sim.state)

    # Performance evaluation
    perf = model.evaluate_detection_performance(posts, sim.world.users, sim.state)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Accuracy", f"{perf['accuracy']:.2%}")
    with col2:
        st.metric("Precision", f"{perf['precision']:.2%}")
    with col3:
        st.metric("Recall", f"{perf['recall']:.2%}")
    with col4:
        st.metric("F1 Score", f"{perf['f1_score']:.2%}")

    st.markdown("**Confusion Matrix:**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("True Positives", perf["true_positives"])
    with col2:
        st.metric("True Negatives", perf["true_negatives"])
    with col3:
        st.metric("False Positives", perf["false_positives"])
    with col4:
        st.metric("False Negatives", perf["false_negatives"])

    # Flagged posts
    st.markdown("**Flagged Misinformation Posts:**")
    flagged = predictions_df[predictions_df["misinfo_prediction"] == 1]
    st.dataframe(flagged.head(10), use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            predictions_df,
            x="misinfo_probability",
            nbins=20,
            title="Misinformation Probability Distribution",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Actual vs predicted comparison
        data = {
            "Category": ["Actual Misinfo", "Detected Misinfo"],
            "Count": [
                predictions_df["actual_is_misinfo"].sum(),
                predictions_df["misinfo_prediction"].sum(),
            ],
        }
        fig = px.bar(
            data,
            x="Category",
            y="Count",
            title="Actual vs Detected Misinformation",
        )
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    render()
