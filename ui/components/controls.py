"""Control components for user input."""

from typing import Any

import streamlit as st


def config_slider(
    label: str,
    key: str,
    min_value: float,
    max_value: float,
    default: float,
    step: float | None = None,
    help_text: str = "",
    format_str: str | None = None,
) -> float:
    """Create a configuration slider with consistent styling.

    Args:
        label: Slider label
        key: Session state key
        min_value: Minimum value
        max_value: Maximum value
        default: Default value
        step: Step size
        help_text: Help text
        format_str: Format string for display

    Returns:
        Selected value
    """
    return st.slider(
        label,
        min_value=min_value,
        max_value=max_value,
        value=default,
        step=step,
        help=help_text,
        format=format_str,
        key=key,
    )


def config_selectbox(
    label: str,
    key: str,
    options: list[str],
    default: str | None = None,
    help_text: str = "",
) -> str:
    """Create a configuration selectbox.

    Args:
        label: Selectbox label
        key: Session state key
        options: List of options
        default: Default selected option
        help_text: Help text

    Returns:
        Selected option
    """
    default_idx = 0
    if default and default in options:
        default_idx = options.index(default)

    return st.selectbox(
        label,
        options,
        index=default_idx,
        help=help_text,
        key=key,
    )


def metric_card(
    label: str,
    value: Any,
    delta: float | None = None,
    delta_color: str = "normal",
    help_text: str = "",
) -> None:
    """Display a metric card.

    Args:
        label: Metric label
        value: Metric value
        delta: Change value
        delta_color: Color for delta ("normal", "inverse", "off")
        help_text: Help text
    """
    st.metric(
        label=label,
        value=value,
        delta=delta,
        delta_color=delta_color,
        help=help_text,
    )


def toggle_section(
    label: str,
    key: str,
    default: bool = False,
) -> bool:
    """Create a toggleable section.

    Args:
        label: Section label
        key: Session state key
        default: Default state

    Returns:
        Whether section is expanded
    """
    return st.checkbox(label, value=default, key=key)


def number_input_with_validation(
    label: str,
    key: str,
    min_value: float,
    max_value: float,
    default: float,
    step: float = 1.0,
    help_text: str = "",
) -> float:
    """Create a number input with validation.

    Args:
        label: Input label
        key: Session state key
        min_value: Minimum value
        max_value: Maximum value
        default: Default value
        step: Step size
        help_text: Help text

    Returns:
        Validated value
    """
    value = st.number_input(
        label,
        min_value=min_value,
        max_value=max_value,
        value=default,
        step=step,
        help=help_text,
        key=key,
    )
    return value


def create_config_panel(
    title: str,
    configs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Create a configuration panel with multiple inputs.

    Args:
        title: Panel title
        configs: List of config dicts with keys:
            - type: "slider", "selectbox", "number", "checkbox"
            - label: Display label
            - key: Config key
            - Plus type-specific parameters

    Returns:
        Dictionary of config values
    """
    st.subheader(title)

    values = {}

    for config in configs:
        config_type = config.get("type", "slider")
        label = config["label"]
        key = config["key"]

        if config_type == "slider":
            values[key] = st.slider(
                label,
                min_value=config.get("min", 0),
                max_value=config.get("max", 100),
                value=config.get("default", 50),
                step=config.get("step"),
                help=config.get("help", ""),
            )
        elif config_type == "selectbox":
            values[key] = st.selectbox(
                label,
                config.get("options", []),
                help=config.get("help", ""),
            )
        elif config_type == "number":
            values[key] = st.number_input(
                label,
                min_value=config.get("min", 0),
                max_value=config.get("max", 100),
                value=config.get("default", 0),
                step=config.get("step", 1),
                help=config.get("help", ""),
            )
        elif config_type == "checkbox":
            values[key] = st.checkbox(
                label,
                value=config.get("default", False),
                help=config.get("help", ""),
            )

    return values


def progress_indicator(
    current: int,
    total: int,
    label: str = "Progress",
) -> None:
    """Display a progress indicator.

    Args:
        current: Current value
        total: Total value
        label: Progress label
    """
    progress = current / total if total > 0 else 0
    st.progress(progress, text=f"{label}: {current}/{total}")


def status_indicator(
    status: str,
    message: str = "",
) -> None:
    """Display a status indicator.

    Args:
        status: Status type ("success", "warning", "error", "info")
        message: Status message
    """
    if status == "success":
        st.success(message)
    elif status == "warning":
        st.warning(message)
    elif status == "error":
        st.error(message)
    else:
        st.info(message)
