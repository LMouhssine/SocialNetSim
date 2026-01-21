#!/usr/bin/env python3
"""Run a simulation from command line."""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from config.schemas import SimulationConfig, load_config, load_scenario
from generator import World
from engine import Simulation


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run SocialNetSim simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="default",
        choices=["default", "echo_chamber", "misinformation"],
        help="Pre-defined scenario to use",
    )
    parser.add_argument(
        "--users",
        type=int,
        default=1000,
        help="Number of users",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of simulation steps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/simulations/run",
        help="Output directory for results",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger.remove()
    logger.add(sys.stderr, level=log_level)

    # Load or create configuration
    if args.config:
        logger.info(f"Loading config from {args.config}")
        config = load_config(args.config)
    else:
        logger.info(f"Loading scenario: {args.scenario}")
        config = load_scenario(args.scenario)

    # Override with command line arguments
    config.user.num_users = args.users
    config.num_steps = args.steps
    config.seed = args.seed

    logger.info(f"Configuration: {config.name}")
    logger.info(f"  Users: {config.user.num_users}")
    logger.info(f"  Steps: {config.num_steps}")
    logger.info(f"  Seed: {config.seed}")

    # Build world
    logger.info("Building synthetic world...")
    world = World(config)
    world.build()

    stats = world.get_statistics()
    logger.info(f"World built:")
    logger.info(f"  Topics: {stats['topics']['total']}")
    logger.info(f"  Users: {stats['users']['total']}")
    logger.info(f"  Network edges: {stats['network']['num_edges']}")

    # Run simulation
    logger.info("Running simulation...")
    sim = Simulation(config)
    sim.initialize(world)
    results = sim.run(show_progress=True)

    # Summary
    metrics = results.get("metrics_summary", {})
    logger.info("Simulation complete!")
    logger.info(f"  Total posts: {metrics.get('total_posts', 0)}")
    logger.info(f"  Total interactions: {metrics.get('total_interactions', 0)}")
    logger.info(f"  Engagement rate: {metrics.get('engagement_rate', 0):.2%}")
    logger.info(f"  Misinfo share rate: {metrics.get('misinfo_share_rate', 0):.2%}")

    # Save results
    output_path = Path(args.output)
    logger.info(f"Saving results to {output_path}")
    sim.save(output_path)

    logger.info("Done!")


if __name__ == "__main__":
    main()
