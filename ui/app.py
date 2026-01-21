"""Main Streamlit application for SocialNetSim."""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st

from ui.state import init_session_state

# Page config must be first Streamlit command
st.set_page_config(
    page_title="SocialNetSim",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    """Main application entry point."""
    # Initialize session state
    init_session_state()

    # Sidebar navigation
    st.sidebar.title("🌐 SocialNetSim")
    st.sidebar.markdown("*AI-Powered Social Network Simulator*")
    st.sidebar.divider()

    # Navigation
    pages = {
        "🎮 Simulation": "simulation",
        "🕸️ Network": "network",
        "📊 Metrics": "metrics",
        "🌊 Cascades": "cascades",
        "🧪 Experiments": "experiments",
        "🤖 AI Models": "ai_models",
    }

    selected_page = st.sidebar.radio(
        "Navigation",
        list(pages.keys()),
        label_visibility="collapsed",
    )

    page_key = pages[selected_page]
    st.session_state.current_page = page_key

    # Render selected page
    if page_key == "simulation":
        from ui.pages import page_simulation
        page_simulation.render()
    elif page_key == "network":
        from ui.pages import page_network
        page_network.render()
    elif page_key == "metrics":
        from ui.pages import page_metrics
        page_metrics.render()
    elif page_key == "cascades":
        from ui.pages import page_cascades
        page_cascades.render()
    elif page_key == "experiments":
        from ui.pages import page_experiments
        page_experiments.render()
    elif page_key == "ai_models":
        from ui.pages import page_ai_models
        page_ai_models.render()

    # Footer
    st.sidebar.divider()
    st.sidebar.caption("SocialNetSim v1.0.0")
    st.sidebar.caption("All data is synthetically generated")


if __name__ == "__main__":
    main()
