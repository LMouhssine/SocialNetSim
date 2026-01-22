"""Main simulation loop orchestrating all components."""

from pathlib import Path
from typing import Any, Callable
import json

from loguru import logger
from tqdm import tqdm
import numpy as np

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
from .opinion_dynamics import OpinionDynamicsEngine, OpinionDynamicsConfig
from .vectorized_ops import (
    VectorizedUserState,
    VectorizedPostState,
    VectorizedEngagement,
    VectorizedFeedRanking,
)
from .batch_processor import BatchProcessor, BatchConfig
from .memory_manager import MemoryManager, MemoryConfig


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

    def __init__(
        self,
        config: SimulationConfig,
        use_vectorized: bool = True,
        use_opinion_dynamics: bool = True,
        batch_config: BatchConfig | None = None,
        memory_config: MemoryConfig | None = None,
    ):
        """Initialize simulation.

        Args:
            config: Simulation configuration
            use_vectorized: Whether to use vectorized operations for performance
            use_opinion_dynamics: Whether to enable opinion dynamics
            batch_config: Configuration for batch processing
            memory_config: Configuration for memory management
        """
        self.config = config
        self.seed = config.seed
        self.use_vectorized = use_vectorized
        self.use_opinion_dynamics = use_opinion_dynamics

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

        # New components for enhanced simulation
        self.opinion_engine: OpinionDynamicsEngine | None = None
        self.batch_processor: BatchProcessor | None = None
        self.memory_manager: MemoryManager | None = None

        # Vectorized state (for performance mode)
        self.vectorized_user_state: VectorizedUserState | None = None
        self.vectorized_post_state: VectorizedPostState | None = None
        self.vectorized_engagement: VectorizedEngagement | None = None
        self.vectorized_feed_ranker: VectorizedFeedRanking | None = None

        # Configuration for new components
        self.batch_config = batch_config or BatchConfig()
        self.memory_config = memory_config or MemoryConfig()

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

        # Initialize opinion dynamics engine
        if self.use_opinion_dynamics:
            opinion_config = OpinionDynamicsConfig()
            self.opinion_engine = OpinionDynamicsEngine(opinion_config, seed=self.seed)
            self.opinion_engine.initialize_users(self.world.users)
            logger.debug("Opinion dynamics engine initialized")

        # Initialize batch processor
        self.batch_processor = BatchProcessor(self.batch_config, seed=self.seed)

        # Initialize memory manager
        self.memory_manager = MemoryManager(self.memory_config)

        # Initialize vectorized components if enabled
        if self.use_vectorized:
            self._initialize_vectorized_state()
            logger.debug("Vectorized state initialized")

        self._initialized = True
        logger.info("Simulation initialized")

        return self

    def _initialize_vectorized_state(self) -> None:
        """Initialize vectorized state for performance mode."""
        users_list = list(self.world.users.values())
        n_users = len(users_list)

        # Build user index mapping
        user_id_to_idx = {u.user_id: i for i, u in enumerate(users_list)}

        # Initialize vectorized user state
        self.vectorized_user_state = VectorizedUserState(n_users)

        # Populate from user objects
        for i, user in enumerate(users_list):
            self.vectorized_user_state.activity_level[i] = user.activity_level
            self.vectorized_user_state.influence_score[i] = user.influence_score
            self.vectorized_user_state.follower_count[i] = len(user.followers)
            self.vectorized_user_state.following_count[i] = len(user.following)

            # Traits
            self.vectorized_user_state.openness[i] = user.traits.openness
            self.vectorized_user_state.conscientiousness[i] = user.traits.conscientiousness
            self.vectorized_user_state.emotional_reactivity[i] = user.traits.emotional_reactivity
            self.vectorized_user_state.confirmation_bias[i] = user.traits.confirmation_bias
            self.vectorized_user_state.misinfo_susceptibility[i] = user.traits.misinfo_susceptibility

        # Build sparse following matrix
        from scipy import sparse
        row_indices = []
        col_indices = []
        for i, user in enumerate(users_list):
            for followed_id in user.following:
                if followed_id in user_id_to_idx:
                    row_indices.append(i)
                    col_indices.append(user_id_to_idx[followed_id])

        data = np.ones(len(row_indices), dtype=np.float32)
        self.vectorized_user_state.following_matrix = sparse.csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(n_users, n_users),
        )

        # Initialize vectorized post state (empty initially)
        self.vectorized_post_state = VectorizedPostState(0)

        # Initialize vectorized components
        self.vectorized_engagement = VectorizedEngagement()
        self.vectorized_feed_ranker = VectorizedFeedRanking()

        # Store mapping for later use
        self._user_id_to_idx = user_id_to_idx
        self._users_list = users_list

    def run(
        self,
        num_steps: int | None = None,
        show_progress: bool = True,
        use_vectorized_step: bool | None = None,
    ) -> dict[str, Any]:
        """Run the simulation.

        Args:
            num_steps: Number of steps (defaults to config value)
            show_progress: Whether to show progress bar
            use_vectorized_step: Override for using vectorized step (default: self.use_vectorized)

        Returns:
            Summary results dictionary
        """
        if not self._initialized:
            raise RuntimeError("Simulation not initialized - call initialize() first")

        num_steps = num_steps or self.config.num_steps
        use_vec = use_vectorized_step if use_vectorized_step is not None else self.use_vectorized

        logger.info(f"Running simulation for {num_steps} steps (vectorized={use_vec})...")

        iterator = range(num_steps)
        if show_progress:
            iterator = tqdm(iterator, desc="Simulating", unit="step")

        for step in iterator:
            # Choose step method based on mode
            if use_vec and self.vectorized_user_state is not None:
                step_metrics = self._run_step_vectorized()
            else:
                step_metrics = self._run_step()

            # Memory management
            if self.memory_manager and step % 10 == 0:
                mem_ops = self.memory_manager.manage_memory(self.state, step)
                if mem_ops.get("memory_pressure"):
                    logger.warning(f"Step {step}: Memory pressure detected, cleanup performed")

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

        # 6. Opinion dynamics update
        if self.use_opinion_dynamics and self.opinion_engine:
            self._process_opinion_dynamics(new_interactions)

        # 7. Diffusion model step
        if hasattr(self.cascade_engine, 'step_diffusion'):
            self.cascade_engine.step_diffusion(self.state)

        # 8. Collect metrics
        step_metrics = self.metrics_collector.collect_step_metrics(
            self.state,
            new_interactions,
            new_posts,
            new_cascades,
            moderation_actions,
        )

        # Add opinion dynamics metrics if available
        if self.use_opinion_dynamics and self.opinion_engine:
            opinion_stats = self.opinion_engine.get_statistics()
            step_metrics.additional_metrics = step_metrics.additional_metrics or {}
            step_metrics.additional_metrics["opinion"] = opinion_stats

        return step_metrics

    def _process_opinion_dynamics(self, interactions: list[Interaction]) -> None:
        """Process opinion dynamics from interactions.

        Args:
            interactions: List of interactions this step
        """
        if not self.opinion_engine:
            return

        # Process peer interactions (shares, comments indicate influence)
        for interaction in interactions:
            if interaction.interaction_type in (InteractionType.SHARE, InteractionType.COMMENT):
                user = self.world.users.get(interaction.user_id)
                post = self.state.get_post(interaction.post_id)

                if user and post:
                    author = self.world.users.get(post.author_id)
                    if author and author.user_id != user.user_id:
                        # Peer influence from post author
                        self.opinion_engine.process_peer_interaction(
                            user, author, self.state.current_step
                        )

                    # Content influence from post topic
                    if post.topics:
                        primary_topic = post.topics[0]
                        # Estimate content stance based on topic
                        content_stance = self._estimate_content_stance(post)
                        self.opinion_engine.process_content_exposure(
                            user, content_stance, self.state.current_step
                        )

        # Step the opinion dynamics model
        self.opinion_engine.step(self.state.current_step, self.world.users)

    def _estimate_content_stance(self, post: Post) -> float:
        """Estimate stance/opinion expressed in post content.

        Args:
            post: Post to analyze

        Returns:
            Estimated stance in [-1, 1]
        """
        # Use sentiment and controversy as proxy for stance
        stance = post.sentiment * 0.5

        # Controversial content pushes toward extremes
        if post.controversy_score > 0.5:
            stance = stance * 1.5

        return max(-1.0, min(1.0, stance))

    def _run_step_vectorized(self) -> StepMetrics:
        """Run a single simulation step using vectorized operations.

        Optimized for large-scale simulations (100k+ users).

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

        event_multiplier = 1.0
        if self.state.active_events:
            event_effect = self.state.get_combined_event_effect()
            event_multiplier = event_effect.engagement_multiplier

        # 2. Sync vectorized state with new posts
        self._sync_vectorized_post_state()

        # 3. Sample active and posting users using batch processor
        active_indices = self.batch_processor.sample_active_users(
            self.vectorized_user_state,
            current_step,
            event_multiplier,
        )

        posting_indices = self.batch_processor.sample_posting_users(
            self.vectorized_user_state,
            active_indices,
            self.config.content.avg_posts_per_step,
        )

        # 4. Create posts for posting users
        for idx in posting_indices:
            user = self._users_list[idx]
            post = self._create_post(user)
            new_posts.append(post)

            if post.virality_score > 0.3:
                cascade = self.cascade_engine.initialize_cascade(
                    post, user, self.state
                )
                self.state.add_cascade(cascade)
                new_cascades.append(cascade)

        # Update post state with new posts
        self._sync_vectorized_post_state()

        # 5. Process engagement using batch processor
        if self.vectorized_post_state.n_posts > 0:
            batch_results = self.batch_processor.process_step_batched(
                self.vectorized_user_state,
                self.vectorized_post_state,
                current_step,
                avg_posts_per_step=self.config.content.avg_posts_per_step,
                event_multiplier=event_multiplier,
                feed_size=10,
            )

            # Convert batch engagement results to interactions
            new_interactions = self._convert_batch_to_interactions(batch_results)
            for interaction in new_interactions:
                self.state.add_interaction(interaction)

        # 6. Process cascades (using original method for accuracy)
        self._process_cascades(new_interactions)

        # 7. Content moderation
        for post in new_posts:
            decision = self.moderation_engine.moderate_post(post, self.state)
            if decision.action != "none":
                moderation_actions += 1
                author = self.world.users.get(post.author_id)
                if author:
                    self.moderation_engine.update_user_credibility(author, decision)

        # 8. Opinion dynamics update
        if self.use_opinion_dynamics and self.opinion_engine:
            self._process_opinion_dynamics(new_interactions)

        # 9. Diffusion model step
        if hasattr(self.cascade_engine, 'step_diffusion'):
            self.cascade_engine.step_diffusion(self.state)

        # 10. Update vectorized user state from runtime state
        self._sync_vectorized_user_state()

        # 11. Collect metrics
        step_metrics = self.metrics_collector.collect_step_metrics(
            self.state,
            new_interactions,
            new_posts,
            new_cascades,
            moderation_actions,
        )

        # Add vectorized-specific metrics
        step_metrics.additional_metrics = step_metrics.additional_metrics or {}
        step_metrics.additional_metrics["vectorized"] = {
            "active_users": len(active_indices),
            "posting_users": len(posting_indices),
        }

        return step_metrics

    def _sync_vectorized_post_state(self) -> None:
        """Synchronize vectorized post state with current posts."""
        posts_list = list(self.state.posts.values())
        n_posts = len(posts_list)

        if n_posts == 0:
            self.vectorized_post_state = VectorizedPostState(0)
            return

        # Check if we need to resize
        if self.vectorized_post_state.n_posts != n_posts:
            self.vectorized_post_state = VectorizedPostState(n_posts)

            # Build post index mapping
            self._post_id_to_idx = {p.post_id: i for i, p in enumerate(posts_list)}
            self._posts_list = posts_list

            # Populate post state
            for i, post in enumerate(posts_list):
                author_idx = self._user_id_to_idx.get(post.author_id, 0)
                self.vectorized_post_state.author_idx[i] = author_idx
                self.vectorized_post_state.created_step[i] = post.created_step
                self.vectorized_post_state.virality_score[i] = post.virality_score
                self.vectorized_post_state.sentiment[i] = post.sentiment
                self.vectorized_post_state.controversy_score[i] = post.controversy_score
                self.vectorized_post_state.quality_score[i] = post.quality_score
                self.vectorized_post_state.view_count[i] = post.view_count
                self.vectorized_post_state.like_count[i] = post.like_count
                self.vectorized_post_state.share_count[i] = post.share_count
                self.vectorized_post_state.is_active[i] = post.is_active

    def _sync_vectorized_user_state(self) -> None:
        """Synchronize vectorized user state from runtime state."""
        for i, user in enumerate(self._users_list):
            runtime = self.state.runtime_states.get(user.user_id)
            if runtime:
                self.vectorized_user_state.fatigue[i] = runtime.fatigue_level
                self.vectorized_user_state.session_interactions[i] = runtime.session_interactions
                self.vectorized_user_state.last_active_step[i] = runtime.last_active_step

    def _convert_batch_to_interactions(
        self,
        batch_results: dict[str, Any],
    ) -> list[Interaction]:
        """Convert batch processing results to interaction objects.

        Args:
            batch_results: Results from batch processor

        Returns:
            List of Interaction objects
        """
        interactions = []

        # For now, create aggregate interactions based on batch results
        # In a full implementation, this would track individual user-post pairs

        return interactions

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
                "use_vectorized": self.use_vectorized,
                "use_opinion_dynamics": self.use_opinion_dynamics,
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

        # Add opinion dynamics statistics if enabled
        if self.use_opinion_dynamics and self.opinion_engine:
            results["opinion_dynamics"] = self.opinion_engine.get_statistics()
            results["polarization_metrics"].update(
                self.opinion_engine.compute_polarization_metrics()
            )

        # Add memory management statistics
        if self.memory_manager:
            results["memory_stats"] = self.memory_manager.get_stats()

        # Add cascade diffusion statistics
        if hasattr(self.cascade_engine, 'diffusion_model'):
            viral_cascades = self.cascade_engine.get_viral_cascades(self.state, min_shares=5)
            results["cascade_diffusion"] = {
                "viral_cascade_count": len(viral_cascades),
                "cascades": [
                    self.cascade_engine.get_cascade_diffusion_stats(c)
                    for c in viral_cascades[:10]  # Top 10
                ],
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
    def load(
        cls,
        path: str | Path,
        use_vectorized: bool = True,
        use_opinion_dynamics: bool = True,
    ) -> "Simulation":
        """Load simulation from saved state.

        Args:
            path: Directory to load from
            use_vectorized: Whether to use vectorized operations
            use_opinion_dynamics: Whether to enable opinion dynamics

        Returns:
            Loaded Simulation
        """
        path = Path(path)

        # Load config
        with open(path / "config.json") as f:
            config_dict = json.load(f)
        config = SimulationConfig(**config_dict)

        # Create simulation with enhanced options
        sim = cls(
            config,
            use_vectorized=use_vectorized,
            use_opinion_dynamics=use_opinion_dynamics,
        )

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

        # Initialize opinion dynamics if enabled
        if use_opinion_dynamics:
            opinion_config = OpinionDynamicsConfig()
            sim.opinion_engine = OpinionDynamicsEngine(opinion_config, seed=config.seed)
            sim.opinion_engine.initialize_users(sim.world.users)

        # Initialize batch processor and memory manager
        sim.batch_processor = BatchProcessor(sim.batch_config, seed=config.seed)
        sim.memory_manager = MemoryManager(sim.memory_config)

        # Initialize vectorized state if enabled
        if use_vectorized:
            sim._initialize_vectorized_state()

        sim._initialized = True
        logger.info(f"Simulation loaded from {path}")

        return sim
