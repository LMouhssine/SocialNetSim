#!/usr/bin/env python3
"""Generate a synthetic world without running simulation."""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from config.schemas import SimulationConfig, load_scenario
from generator import World


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic world data",
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
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/generated/world",
        help="Output directory",
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

    # Load configuration
    logger.info(f"Loading scenario: {args.scenario}")
    config = load_scenario(args.scenario)
    config.user.num_users = args.users
    config.seed = args.seed

    logger.info(f"Configuration:")
    logger.info(f"  Users: {config.user.num_users}")
    logger.info(f"  Topics: {config.content.topics.num_topics}")
    logger.info(f"  Seed: {config.seed}")

    # Build world
    logger.info("Building synthetic world...")
    world = World(config)
    world.build()

    # Display statistics
    stats = world.get_statistics()

    logger.info("World statistics:")
    logger.info(f"  Topics: {stats['topics']['total']}")
    for category, count in stats['topics']['by_category'].items():
        logger.info(f"    {category}: {count}")

    logger.info(f"  Users: {stats['users']['total']}")
    trait_stats = stats['users']['trait_statistics']
    for trait, values in trait_stats.items():
        logger.info(f"    {trait}: mean={values['mean']:.3f}, std={values['std']:.3f}")

    logger.info(f"  Network:")
    net_stats = stats['network']
    logger.info(f"    Nodes: {net_stats['num_nodes']}")
    logger.info(f"    Edges: {net_stats['num_edges']}")
    logger.info(f"    Density: {net_stats['density']:.4f}")
    logger.info(f"    Reciprocity: {net_stats['reciprocity']:.2%}")

    # Save world
    output_path = Path(args.output)
    logger.info(f"Saving world to {output_path}")
    world.save(output_path)

    logger.info("Done!")


if __name__ == "__main__":
    main()
