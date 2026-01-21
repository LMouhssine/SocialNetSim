"""Main simulation loop orchestrating all components."""

from pathlib import Path
from typing import Any, Callable
import json

from loguru import logger
from tqdm import tqdm

from config.schemas import SimulationConfig
from models import User, Post, Interaction, Cascade, Event
from models.enums import InteractionType
from generator import World
from .state import SimulationState
from .feed import FeedRanker
from .engagement import EngagementModel
from .cascade import CascadeEngine
from .events import EventEngine
from .moderation import ModerationEngine
from .metrics import MetricsCollector, StepMetrics


class Simulation:
    """Main simulation orchestrator.

    Coordinates all simulation components:
    - World generation
    - Feed ranking
    - Engagement modeling
    - Cascade spreading
    - Event generation
    - Content moderation
    - Metrics collection
    """

    def __init__(self, config: SimulationConfig):
        """Initialize simulation.

        Args:
            config: Simulation configuration
        """
        self.config = config
        self.seed = config.seed

        # World and state
        self.world: World | None = None
        self.state: SimulationState | None = None

        # Engine components
        self.feed_ranker: FeedRanker | None = None
        self.engagement_model: EngagementModel | None = None
        self.cascade_engine: CascadeEngine | None = None
        self.event_engine: EventEngine | None = None
        self.moderation_engine: ModerationEngine | None = None
        self.metrics_collector: MetricsCollector | None = None

        # Callbacks
        self.step_callbacks: list[Callable[[int, StepMetrics], None]] = []

        self._initialized = False

    def initialize(self, world: World | None = None) -> "Simulation":
        """Initialize simulation with world and components.

        Args:
            world: Optional pre-built world (will build new if None)

        Returns:
            Self for method chaining
        """
        logger.info("Initializing simulation...")

        # Build or use provided world
        if world:
            self.world = world
        else:
            self.world = World(self.config)
            self.world.build()

        # Initialize state
        self.state = SimulationState(self.world.users)

        # Initialize engine components
        self.feed_ranker = FeedRanker(self.config.feed, seed=self.seed)
        self.engagement_model = EngagementModel(self.config.engagement, seed=self.seed)
        self.cascade_engine = CascadeEngine(self.config.cascade, seed=self.seed)
        self.event_engine = EventEngine(
            self.config.events,
            self.world.topics,
            seed=self.seed,
        )
        self.moderation_engine = ModerationEngine(self.config.moderation, seed=self.seed)
        self.metrics_collector = MetricsCollector()

        self._initialized = True
        logger.info("Simulation initialized")

        return self

    def run(
        self,
        num_steps: int | None = None,
        show_progress: bool = True,
    ) -> dict[str, Any]:
        """Run the simulation.

        Args:
            num_steps: Number of steps (defaults to config value)
            show_progress: Whether to show progress bar

        Returns:
            Summary results dictionary
        """
        if not self._initialized:
            raise RuntimeError("Simulation not initialized - call initialize() first")

        num_steps = num_steps or self.config.num_steps
        logger.info(f"Running simulation for {num_steps} steps...")

        iterator = range(num_steps)
        if show_progress:
            iterator = tqdm(iterator, desc="Simulating", unit="step")

        for step in iterator:
            step_metrics = self._run_step()

            # Call callbacks
            for callback in self.step_callbacks:
                callback(step, step_metrics)

        logger.info("Simulation complete")
        return self.get_results()

    def _run_step(self) -> StepMetrics:
        """Run a single simulation step.

        Returns:
            Metrics for this step
        """
        # Advance state
        self.state.advance_step()
        current_step = self.state.current_step

        # Track new items this step
        new_posts: list[Post] = []
        new_interactions: list[Interaction] = []
        new_cascades: list[Cascade] = []
        moderation_actions = 0

        # 1. Process events
        new_event = self.event_engine.process_step(self.state)
        if new_event:
            self.state.add_event(new_event)
            logger.debug(f"Step {current_step}: New event - {new_event.name}")

        # 2. User activity - posting
        for user in self.world.users.values():
            if self.engagement_model.should_user_post(
                user, self.state, self.config.content.avg_posts_per_step
            ):
                post = self._create_post(user)
                new_posts.append(post)

                # Initialize cascade for potentially viral posts
                if post.virality_score > 0.3:
                    cascade = self.cascade_engine.initialize_cascade(
                        post, user, self.state
                    )
                    self.state.add_cascade(cascade)
                    new_cascades.append(cascade)

        # 3. User activity - engagement
        for user in self.world.users.values():
            if not self.engagement_model.should_user_be_active(user, self.state):
                continue

            # Get user's feed
            candidates = self.feed_ranker.get_candidate_posts(user, self.state)
            feed = self.feed_ranker.rank_feed(user, candidates, self.state)

            # Process engagement with feed posts
            for post in feed[:10]:  # Limit feed consumption
                interactions = self.engagement_model.process_engagement(
                    user, post, self.state, self.world.users
                )

                for interaction in interactions:
                    self.state.add_interaction(interaction)
                    new_interactions.append(interaction)

                    # Handle shares for cascades
                    if interaction.interaction_type == InteractionType.SHARE:
                        self._handle_share(user, post, interaction)

        # 4. Cascade spreading
        self._process_cascades(new_interactions)

        # 5. Content moderation
        for post in new_posts:
            decision = self.moderation_engine.moderate_post(post, self.state)
            if decision.action != "none":
                moderation_actions += 1
                author = self.world.users.get(post.author_id)
                if author:
                    self.moderation_engine.update_user_credibility(author, decision)

        # 6. Collect metrics
        step_metrics = self.metrics_collector.collect_step_metrics(
            self.state,
            new_interactions,
            new_posts,
            new_cascades,
            moderation_actions,
        )

        return step_metrics

    def _create_post(self, user: User) -> Post:
        """Create a new post from a user.

        Args:
            user: User creating the post

        Returns:
            Created Post
        """
        # Check for event-boosted topics
        boosted_topics = None
        for event in self.state.active_events:
            if event.affected_topics:
                boosted_topics = event.affected_topics
                break

        post = self.world.generate_post(
            user,
            self.state.current_step,
            forced_topics=boosted_topics if boosted_topics and self.world.rng.random() < 0.3 else None,
        )

        self.state.add_post(post)
        user.record_post(self.state.current_step)

        return post

    def _handle_share(
        self,
        user: User,
        post: Post,
        interaction: Interaction,
    ) -> None:
        """Handle a share interaction.

        Args:
            user: User who shared
            post: Post that was shared
            interaction: Share interaction
        """
        # If post has a cascade, record the share
        if post.cascade_id:
            cascade = self.state.get_cascade(post.cascade_id)
            if cascade:
                # Find source user (who exposed this user to the post)
                source_user_id = interaction.source_user_id or post.author_id
                self.cascade_engine.record_share(
                    cascade, user, source_user_id, self.state
                )

    def _process_cascades(self, new_interactions: list[Interaction]) -> None:
        """Process active cascades for spreading.

        Args:
            new_interactions: New interactions this step
        """
        # Build network followers map
        network_followers = {}
        for user_id in self.world.users:
            network_followers[user_id] = self.world.network_generator.get_followers(user_id)

        for cascade in self.state.get_active_cascades():
            post = self.state.get_post(cascade.post_id)
            if not post:
                continue

            # Get exposures from cascade spread
            exposures = self.cascade_engine.process_cascade_spread(
                cascade, post, self.state, self.world.users, network_followers
            )

            # Process exposed users
            for user, source_user_id in exposures:
                # Calculate share probability
                probs = self.engagement_model.calculate_engagement_probability(
                    user, post, self.state, self.world.users
                )

                # Use cascade engine's threshold model
                if self.cascade_engine.should_user_share(
                    user, post, cascade, self.state, probs.share
                ):
                    # Create share interaction
                    interactions = self.engagement_model.process_engagement(
                        user, post, self.state, self.world.users, source_user_id
                    )

                    for interaction in interactions:
                        self.state.add_interaction(interaction)

                        if interaction.interaction_type == InteractionType.SHARE:
                            self.cascade_engine.record_share(
                                cascade, user, source_user_id, self.state
                            )

    def get_results(self) -> dict[str, Any]:
        """Get simulation results.

        Returns:
            Dictionary of results and metrics
        """
        results = {
            "config": {
                "name": self.config.name,
                "num_steps": self.config.num_steps,
                "num_users": self.config.user.num_users,
            },
            "state_summary": self.state.get_summary_statistics(),
            "metrics_summary": self.metrics_collector.get_summary_metrics(),
            "user_metrics": self.metrics_collector.compute_user_metrics(
                self.state, self.world.users
            ),
            "content_metrics": self.metrics_collector.compute_content_metrics(self.state),
            "polarization_metrics": self.metrics_collector.compute_network_polarization(
                self.state, self.world.users
            ),
            "moderation_stats": self.moderation_engine.get_statistics(),
            "event_stats": self.event_engine.get_event_statistics(self.state),
        }

        return results

    def get_metrics_dataframe(self):
        """Get step metrics as DataFrame.

        Returns:
            pandas DataFrame
        """
        return self.metrics_collector.get_metrics_dataframe()

    def add_step_callback(
        self,
        callback: Callable[[int, StepMetrics], None],
    ) -> None:
        """Add a callback to be called after each step.

        Args:
            callback: Function taking (step, metrics)
        """
        self.step_callbacks.append(callback)

    def save(self, path: str | Path) -> None:
        """Save simulation state.

        Args:
            path: Directory to save to
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(path / "config.json", "w") as f:
            json.dump(self.config.model_dump(), f, indent=2)

        # Save world
        self.world.save(path / "world")

        # Save state
        self.state.save(path / "state.json")

        # Save results
        with open(path / "results.json", "w") as f:
            json.dump(self.get_results(), f, indent=2, default=str)

        # Save metrics
        df = self.get_metrics_dataframe()
        df.to_parquet(path / "metrics.parquet")

        logger.info(f"Simulation saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "Simulation":
        """Load simulation from saved state.

        Args:
            path: Directory to load from

        Returns:
            Loaded Simulation
        """
        path = Path(path)

        # Load config
        with open(path / "config.json") as f:
            config_dict = json.load(f)
        config = SimulationConfig(**config_dict)

        # Create simulation
        sim = cls(config)

        # Load world
        sim.world = World.load(path / "world")

        # Load state
        sim.state = SimulationState.load(path / "state.json", sim.world.users)

        # Initialize components
        sim.feed_ranker = FeedRanker(config.feed, seed=config.seed)
        sim.engagement_model = EngagementModel(config.engagement, seed=config.seed)
        sim.cascade_engine = CascadeEngine(config.cascade, seed=config.seed)
        sim.event_engine = EventEngine(config.events, sim.world.topics, seed=config.seed)
        sim.moderation_engine = ModerationEngine(config.moderation, seed=config.seed)
        sim.metrics_collector = MetricsCollector()

        sim._initialized = True
        logger.info(f"Simulation loaded from {path}")

        return sim
