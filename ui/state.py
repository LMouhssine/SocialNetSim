"""Session state management for Streamlit app."""

from typing import Any
from dataclasses import dataclass, field

import streamlit as st

from config.schemas import SimulationConfig
from generator import World
from engine import Simulation


@dataclass
class SessionState:
    """Manages simulation session state."""

    config: SimulationConfig | None = None
    world: World | None = None
    simulation: Simulation | None = None
    is_running: bool = False
    current_step: int = 0
    run_history: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def get(cls) -> "SessionState":
        """Get or create session state."""
        if "session_state" not in st.session_state:
            st.session_state.session_state = cls()
        return st.session_state.session_state

    def reset(self) -> None:
        """Reset session state."""
        self.config = None
        self.world = None
        self.simulation = None
        self.is_running = False
        self.current_step = 0
        self.run_history = []

    def has_world(self) -> bool:
        """Check if world is built."""
        return self.world is not None and self.world.is_built()

    def has_simulation(self) -> bool:
        """Check if simulation is initialized."""
        return self.simulation is not None

    def get_results(self) -> dict[str, Any] | None:
        """Get simulation results if available."""
        if self.simulation:
            return self.simulation.get_results()
        return None


def init_session_state() -> None:
    """Initialize session state variables."""
    defaults = {
        "session_state": SessionState(),
        "config_dict": {},
        "selected_scenario": "default",
        "simulation_running": False,
        "current_page": "simulation",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_config() -> SimulationConfig:
    """Get current configuration from session state."""
    state = SessionState.get()
    if state.config is None:
        state.config = SimulationConfig()
    return state.config


def set_config(config: SimulationConfig) -> None:
    """Set configuration in session state."""
    state = SessionState.get()
    state.config = config


def get_world() -> World | None:
    """Get world from session state."""
    return SessionState.get().world


def set_world(world: World) -> None:
    """Set world in session state."""
    SessionState.get().world = world


def get_simulation() -> Simulation | None:
    """Get simulation from session state."""
    return SessionState.get().simulation


def set_simulation(simulation: Simulation) -> None:
    """Set simulation in session state."""
    SessionState.get().simulation = simulation
