"""Network generation using enhanced Barabasi-Albert model."""

from typing import Any

import networkx as nx
import numpy as np
from numpy.random import Generator

from config.schemas import NetworkConfig
from models import User


class NetworkGenerator:
    """Generates social network using enhanced Barabasi-Albert preferential attachment.

    The attachment probability combines three factors:
    - Degree preference (standard BA model)
    - Interest similarity (cosine similarity of interests)
    - Ideology proximity (closer ideology = more likely to connect)
    """

    def __init__(
        self,
        config: NetworkConfig,
        users: dict[str, User],
        seed: int | None = None,
    ):
        """Initialize network generator.

        Args:
            config: Network generation configuration
            users: Dictionary of users to connect
            seed: Random seed for reproducibility
        """
        self.config = config
        self.users = users
        self.rng = np.random.default_rng(seed)
        self.graph: nx.DiGraph = nx.DiGraph()

    def generate_network(self) -> nx.DiGraph:
        """Generate the social network using enhanced BA model.

        Returns:
            Directed graph representing follower relationships
        """
        user_list = list(self.users.values())
        n_users = len(user_list)

        if n_users == 0:
            return self.graph

        # Add all users as nodes
        for user in user_list:
            self.graph.add_node(user.user_id)

        # Start with a small complete graph
        m = self.config.edges_per_new_node
        initial_size = min(m + 1, n_users)

        # Create initial complete graph
        for i in range(initial_size):
            for j in range(initial_size):
                if i != j:
                    self.graph.add_edge(
                        user_list[i].user_id,
                        user_list[j].user_id,
                    )
                    user_list[j].add_follower(user_list[i].user_id)
                    user_list[i].follow(user_list[j].user_id)

        # Add remaining users using enhanced BA model
        for i in range(initial_size, n_users):
            new_user = user_list[i]
            existing_users = user_list[:i]

            # Calculate attachment probabilities
            probs = self._calculate_attachment_probabilities(new_user, existing_users)

            # Select m users to connect to
            n_edges = min(m, len(existing_users))
            selected_indices = self.rng.choice(
                len(existing_users),
                size=n_edges,
                replace=False,
                p=probs,
            )

            for idx in selected_indices:
                target_user = existing_users[idx]

                # New user follows existing user
                self.graph.add_edge(new_user.user_id, target_user.user_id)
                target_user.add_follower(new_user.user_id)
                new_user.follow(target_user.user_id)

                # Probabilistically reciprocate (more likely with similarity)
                similarity = self._calculate_similarity(new_user, target_user)
                if self.rng.random() < similarity * 0.5:
                    self.graph.add_edge(target_user.user_id, new_user.user_id)
                    new_user.add_follower(target_user.user_id)
                    target_user.follow(new_user.user_id)

        # Update user influence scores based on network position
        self._update_influence_scores()

        return self.graph

    def _calculate_attachment_probabilities(
        self,
        new_user: User,
        existing_users: list[User],
    ) -> np.ndarray:
        """Calculate attachment probability for each existing user.

        P = w_degree * (degree_i / total_degree) +
            w_interest * cosine_similarity(interests) +
            w_ideology * (1 - |ideology_diff| / 2)

        Args:
            new_user: User being added
            existing_users: Existing users to potentially connect to

        Returns:
            Normalized probability array
        """
        n = len(existing_users)
        probs = np.zeros(n)

        # Get degrees
        degrees = np.array([
            self.graph.in_degree(u.user_id) + 1  # +1 to avoid zero
            for u in existing_users
        ])
        total_degree = degrees.sum()

        for i, user in enumerate(existing_users):
            # Degree preference (standard BA)
            degree_score = degrees[i] / total_degree

            # Interest similarity
            interest_score = self._calculate_interest_similarity(new_user, user)

            # Ideology proximity
            ideology_diff = abs(new_user.traits.ideology - user.traits.ideology)
            ideology_score = 1 - (ideology_diff / 2)

            # Combine with weights
            prob = (
                self.config.weight_degree * degree_score +
                self.config.weight_interest * interest_score +
                self.config.weight_ideology * ideology_score
            )

            probs[i] = max(prob, 0.001)  # Ensure non-zero

        # Normalize
        probs = probs / probs.sum()
        return probs

    def _calculate_interest_similarity(self, user1: User, user2: User) -> float:
        """Calculate cosine similarity of interests.

        Args:
            user1: First user
            user2: Second user

        Returns:
            Similarity score (0-1)
        """
        if not user1.interests or not user2.interests:
            return 0.0

        common = user1.interests & user2.interests
        if not common:
            return 0.0

        # Weight by interest strength
        score = sum(
            user1.get_interest_weight(t) * user2.get_interest_weight(t)
            for t in common
        )

        # Normalize by geometric mean of interest counts
        norm = np.sqrt(len(user1.interests) * len(user2.interests))
        return min(1.0, score / norm)

    def _calculate_similarity(self, user1: User, user2: User) -> float:
        """Calculate overall similarity between two users.

        Args:
            user1: First user
            user2: Second user

        Returns:
            Similarity score (0-1)
        """
        interest_sim = self._calculate_interest_similarity(user1, user2)
        ideology_sim = 1 - abs(user1.traits.ideology - user2.traits.ideology) / 2

        return 0.5 * interest_sim + 0.5 * ideology_sim

    def _update_influence_scores(self) -> None:
        """Update influence scores based on network position.

        Uses a combination of in-degree and PageRank.
        """
        if self.graph.number_of_nodes() == 0:
            return

        # Calculate PageRank
        try:
            pagerank = nx.pagerank(self.graph, alpha=0.85)
        except nx.PowerIterationFailedConvergence:
            # Fallback to degree-based influence
            pagerank = {
                node: self.graph.in_degree(node) / max(1, self.graph.number_of_nodes())
                for node in self.graph.nodes()
            }

        # Normalize PageRank to 0-1 range
        max_pr = max(pagerank.values()) if pagerank else 1
        min_pr = min(pagerank.values()) if pagerank else 0
        pr_range = max_pr - min_pr if max_pr > min_pr else 1

        for user_id, user in self.users.items():
            pr_score = (pagerank.get(user_id, 0) - min_pr) / pr_range
            in_degree = self.graph.in_degree(user_id)

            # Combine PageRank and follower count
            # Log transform follower count to reduce impact of very popular users
            follower_score = np.log1p(in_degree) / np.log1p(self.graph.number_of_nodes())

            influence = 0.6 * pr_score + 0.4 * follower_score
            user.update_influence(influence)

    def add_edge(self, follower_id: str, followed_id: str) -> bool:
        """Add a follow relationship.

        Args:
            follower_id: User who follows
            followed_id: User being followed

        Returns:
            True if edge was added, False if it already existed
        """
        if self.graph.has_edge(follower_id, followed_id):
            return False

        self.graph.add_edge(follower_id, followed_id)

        if follower_id in self.users and followed_id in self.users:
            self.users[followed_id].add_follower(follower_id)
            self.users[follower_id].follow(followed_id)

        return True

    def remove_edge(self, follower_id: str, followed_id: str) -> bool:
        """Remove a follow relationship.

        Args:
            follower_id: User who follows
            followed_id: User being followed

        Returns:
            True if edge was removed, False if it didn't exist
        """
        if not self.graph.has_edge(follower_id, followed_id):
            return False

        self.graph.remove_edge(follower_id, followed_id)

        if follower_id in self.users and followed_id in self.users:
            self.users[followed_id].remove_follower(follower_id)
            self.users[follower_id].unfollow(followed_id)

        return True

    def get_followers(self, user_id: str) -> list[str]:
        """Get IDs of users following a given user."""
        return list(self.graph.predecessors(user_id))

    def get_following(self, user_id: str) -> list[str]:
        """Get IDs of users a given user follows."""
        return list(self.graph.successors(user_id))

    def get_mutual_followers(self, user_id: str) -> list[str]:
        """Get IDs of mutual connections."""
        followers = set(self.graph.predecessors(user_id))
        following = set(self.graph.successors(user_id))
        return list(followers & following)

    def get_network_statistics(self) -> dict[str, Any]:
        """Get statistics about the network."""
        if self.graph.number_of_nodes() == 0:
            return {}

        in_degrees = [d for _, d in self.graph.in_degree()]
        out_degrees = [d for _, d in self.graph.out_degree()]

        stats = {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "avg_in_degree": np.mean(in_degrees),
            "avg_out_degree": np.mean(out_degrees),
            "max_in_degree": max(in_degrees),
            "max_out_degree": max(out_degrees),
            "reciprocity": nx.reciprocity(self.graph),
        }

        # Try to compute clustering (can be slow for large graphs)
        if self.graph.number_of_nodes() < 10000:
            try:
                stats["avg_clustering"] = nx.average_clustering(self.graph.to_undirected())
            except Exception:
                stats["avg_clustering"] = None

        return stats

    def get_subgraph(self, user_ids: list[str]) -> nx.DiGraph:
        """Get subgraph containing only specified users."""
        return self.graph.subgraph(user_ids).copy()

    def to_dict(self) -> dict[str, Any]:
        """Convert network to dictionary format."""
        return {
            "nodes": list(self.graph.nodes()),
            "edges": list(self.graph.edges()),
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        config: NetworkConfig,
        users: dict[str, User],
        seed: int | None = None,
    ) -> "NetworkGenerator":
        """Create generator from saved data."""
        generator = cls(config, users, seed)

        for node in data.get("nodes", []):
            generator.graph.add_node(node)

        for source, target in data.get("edges", []):
            generator.graph.add_edge(source, target)
            if source in users and target in users:
                users[target].add_follower(source)
                users[source].follow(target)

        generator._update_influence_scores()
        return generator
