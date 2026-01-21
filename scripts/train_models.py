#!/usr/bin/env python3
"""Train AI models on simulation data."""

import argparse
import sys
from pathlib import Path
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from engine import Simulation
from ai import ViralityPredictor, ChurnPredictor, MisinfoDetector, ModelEvaluator


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train AI models on simulation data",
    )

    parser.add_argument(
        "--simulation",
        type=str,
        required=True,
        help="Path to saved simulation directory",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "virality", "churn", "misinfo"],
        help="Model to train",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/models",
        help="Output directory for trained models",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="xgboost",
        choices=["xgboost", "random_forest"],
        help="ML algorithm to use",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of estimators for ensemble models",
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

    # Load simulation
    logger.info(f"Loading simulation from {args.simulation}")
    sim = Simulation.load(args.simulation)

    posts = list(sim.state.posts.values())
    users = sim.world.users
    state = sim.state

    logger.info(f"Loaded simulation with {len(posts)} posts and {len(users)} users")

    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {}
    evaluator = ModelEvaluator()

    # Train models
    if args.model in ["all", "virality"]:
        logger.info("Training virality predictor...")
        virality_model = ViralityPredictor(
            viral_threshold=50,
            model_type=args.algorithm,
            n_estimators=args.n_estimators,
        )

        try:
            metrics = virality_model.train_from_simulation(posts, users, state)
            logger.info(f"  Test accuracy: {metrics.get('test_accuracy', 0):.2%}")
            logger.info(f"  Test AUC: {metrics.get('test_auc', 0):.3f}")

            virality_model.save(output_path / "virality_predictor.pkl")
            results["virality"] = metrics
        except Exception as e:
            logger.error(f"  Failed: {e}")

    if args.model in ["all", "churn"]:
        logger.info("Training churn predictor...")
        churn_model = ChurnPredictor(
            churn_threshold_steps=10,
            model_type=args.algorithm,
            n_estimators=args.n_estimators,
        )

        try:
            metrics = churn_model.train_from_simulation(users, state)
            logger.info(f"  Test accuracy: {metrics.get('test_accuracy', 0):.2%}")

            churn_model.save(output_path / "churn_predictor.pkl")
            results["churn"] = metrics
        except Exception as e:
            logger.error(f"  Failed: {e}")

    if args.model in ["all", "misinfo"]:
        logger.info("Training misinformation detector...")
        misinfo_model = MisinfoDetector(
            model_type=args.algorithm,
            n_estimators=args.n_estimators,
        )

        try:
            metrics = misinfo_model.train_from_simulation(posts, users, state)
            logger.info(f"  Test accuracy: {metrics.get('test_accuracy', 0):.2%}")
            logger.info(f"  Test precision: {metrics.get('test_precision', 0):.2%}")
            logger.info(f"  Test recall: {metrics.get('test_recall', 0):.2%}")

            misinfo_model.save(output_path / "misinfo_detector.pkl")
            results["misinfo"] = metrics
        except Exception as e:
            logger.error(f"  Failed: {e}")

    # Save results summary
    with open(output_path / "training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Models saved to {output_path}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
