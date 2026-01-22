"""User and content embedding models for ML.

Provides embedding generation using:
- Matrix factorization from interaction data
- Node2vec-style random walk embeddings
- Content embeddings from engagement patterns
"""

from dataclasses import dataclass, field
from typing import Any
from collections import defaultdict

import numpy as np
from scipy import sparse
from numpy.typing import NDArray


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models.

    Attributes:
        embedding_dim: Dimension of embeddings
        learning_rate: Learning rate for SGD
        regularization: L2 regularization strength
        n_iterations: Training iterations
        negative_samples: Number of negative samples
        walk_length: Random walk length for node2vec
        num_walks: Number of walks per node
        p: Return parameter for node2vec
        q: In-out parameter for node2vec
    """

    embedding_dim: int = 64
    learning_rate: float = 0.01
    regularization: float = 0.001
    n_iterations: int = 50
    negative_samples: int = 5
    walk_length: int = 20
    num_walks: int = 10
    p: float = 1.0
    q: float = 1.0


class MatrixFactorizationEmbeddings:
    """Generate embeddings via matrix factorization.

    Uses alternating least squares (ALS) or SGD to factorize
    the user-post interaction matrix.
    """

    def __init__(self, config: EmbeddingConfig | None = None):
        """Initialize embedding model.

        Args:
            config: Embedding configuration
        """
        self.config = config or EmbeddingConfig()

        self.user_embeddings: NDArray[np.float64] | None = None
        self.item_embeddings: NDArray[np.float64] | None = None
        self.user_id_to_idx: dict[str, int] = {}
        self.item_id_to_idx: dict[str, int] = {}

    def fit(
        self,
        interactions: list[tuple[str, str, float]],
        method: str = "als",
    ) -> None:
        """Fit embedding model to interaction data.

        Args:
            interactions: List of (user_id, item_id, weight) tuples
            method: Factorization method ('als' or 'sgd')
        """
        # Build index mappings
        users = set()
        items = set()

        for user_id, item_id, _ in interactions:
            users.add(user_id)
            items.add(item_id)

        self.user_id_to_idx = {uid: i for i, uid in enumerate(sorted(users))}
        self.item_id_to_idx = {iid: i for i, iid in enumerate(sorted(items))}

        n_users = len(self.user_id_to_idx)
        n_items = len(self.item_id_to_idx)

        if n_users == 0 or n_items == 0:
            return

        # Build interaction matrix
        row_indices = [self.user_id_to_idx[u] for u, _, _ in interactions]
        col_indices = [self.item_id_to_idx[i] for _, i, _ in interactions]
        data = [w for _, _, w in interactions]

        interaction_matrix = sparse.csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(n_users, n_items),
        )

        # Factorize
        if method == "als":
            self._fit_als(interaction_matrix)
        else:
            self._fit_sgd(interactions)

    def _fit_als(self, interaction_matrix: sparse.csr_matrix) -> None:
        """Fit using alternating least squares.

        Args:
            interaction_matrix: User-item interaction matrix
        """
        n_users, n_items = interaction_matrix.shape
        dim = self.config.embedding_dim
        reg = self.config.regularization

        # Initialize embeddings
        rng = np.random.default_rng(42)
        self.user_embeddings = rng.normal(0, 0.1, (n_users, dim))
        self.item_embeddings = rng.normal(0, 0.1, (n_items, dim))

        # Convert to dense for ALS (not ideal for very large matrices)
        R = interaction_matrix.toarray()
        mask = R > 0

        for iteration in range(self.config.n_iterations):
            # Fix items, update users
            for u in range(n_users):
                rated_items = mask[u]
                if not np.any(rated_items):
                    continue

                V_j = self.item_embeddings[rated_items]
                R_u = R[u, rated_items]

                A = V_j.T @ V_j + reg * np.eye(dim)
                b = V_j.T @ R_u
                self.user_embeddings[u] = np.linalg.solve(A, b)

            # Fix users, update items
            for i in range(n_items):
                rating_users = mask[:, i]
                if not np.any(rating_users):
                    continue

                U_j = self.user_embeddings[rating_users]
                R_i = R[rating_users, i]

                A = U_j.T @ U_j + reg * np.eye(dim)
                b = U_j.T @ R_i
                self.item_embeddings[i] = np.linalg.solve(A, b)

    def _fit_sgd(self, interactions: list[tuple[str, str, float]]) -> None:
        """Fit using stochastic gradient descent.

        Args:
            interactions: List of (user_id, item_id, weight) tuples
        """
        n_users = len(self.user_id_to_idx)
        n_items = len(self.item_id_to_idx)
        dim = self.config.embedding_dim
        lr = self.config.learning_rate
        reg = self.config.regularization

        # Initialize embeddings
        rng = np.random.default_rng(42)
        self.user_embeddings = rng.normal(0, 0.1, (n_users, dim))
        self.item_embeddings = rng.normal(0, 0.1, (n_items, dim))

        for iteration in range(self.config.n_iterations):
            rng.shuffle(interactions)

            for user_id, item_id, weight in interactions:
                u = self.user_id_to_idx[user_id]
                i = self.item_id_to_idx[item_id]

                # Predicted rating
                pred = np.dot(self.user_embeddings[u], self.item_embeddings[i])
                error = weight - pred

                # Gradient update
                user_grad = error * self.item_embeddings[i] - reg * self.user_embeddings[u]
                item_grad = error * self.user_embeddings[u] - reg * self.item_embeddings[i]

                self.user_embeddings[u] += lr * user_grad
                self.item_embeddings[i] += lr * item_grad

    def get_user_embedding(self, user_id: str) -> NDArray[np.float64] | None:
        """Get embedding for a user.

        Args:
            user_id: User identifier

        Returns:
            Embedding vector or None if not found
        """
        if user_id not in self.user_id_to_idx:
            return None
        return self.user_embeddings[self.user_id_to_idx[user_id]]

    def get_item_embedding(self, item_id: str) -> NDArray[np.float64] | None:
        """Get embedding for an item.

        Args:
            item_id: Item identifier

        Returns:
            Embedding vector or None if not found
        """
        if item_id not in self.item_id_to_idx:
            return None
        return self.item_embeddings[self.item_id_to_idx[item_id]]

    def get_similar_users(
        self,
        user_id: str,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Find similar users by embedding similarity.

        Args:
            user_id: Query user ID
            top_k: Number of similar users to return

        Returns:
            List of (user_id, similarity) tuples
        """
        if user_id not in self.user_id_to_idx:
            return []

        user_emb = self.get_user_embedding(user_id)

        # Compute all similarities
        similarities = self.user_embeddings @ user_emb
        norms = np.linalg.norm(self.user_embeddings, axis=1) * np.linalg.norm(user_emb)
        norms[norms == 0] = 1
        similarities = similarities / norms

        # Get top-k (excluding self)
        idx_to_user = {v: k for k, v in self.user_id_to_idx.items()}
        query_idx = self.user_id_to_idx[user_id]

        top_indices = np.argsort(similarities)[::-1]
        results = []

        for idx in top_indices:
            if idx != query_idx and len(results) < top_k:
                results.append((idx_to_user[idx], float(similarities[idx])))

        return results


class Node2VecEmbeddings:
    """Generate embeddings using node2vec-style random walks.

    Simplified implementation using random walks and skip-gram-like training.
    """

    def __init__(self, config: EmbeddingConfig | None = None):
        """Initialize embedding model.

        Args:
            config: Embedding configuration
        """
        self.config = config or EmbeddingConfig()

        self.embeddings: NDArray[np.float64] | None = None
        self.node_id_to_idx: dict[str, int] = {}

    def fit(
        self,
        adj_matrix: sparse.csr_matrix,
        node_ids: list[str],
    ) -> None:
        """Fit embeddings using random walks.

        Args:
            adj_matrix: Adjacency matrix
            node_ids: List of node IDs matching matrix indices
        """
        n_nodes = adj_matrix.shape[0]
        if n_nodes == 0:
            return

        self.node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        dim = self.config.embedding_dim

        # Initialize embeddings
        rng = np.random.default_rng(42)
        self.embeddings = rng.normal(0, 0.1, (n_nodes, dim))
        context_embeddings = rng.normal(0, 0.1, (n_nodes, dim))

        # Generate random walks
        walks = self._generate_walks(adj_matrix, rng)

        # Train using skip-gram style objective
        lr = self.config.learning_rate

        for iteration in range(self.config.n_iterations):
            rng.shuffle(walks)

            for walk in walks:
                # Use window of 2
                for i, center in enumerate(walk):
                    context_indices = walk[max(0, i - 2):i] + walk[i + 1:i + 3]

                    for context in context_indices:
                        # Positive sample
                        score = np.dot(self.embeddings[center], context_embeddings[context])
                        sigmoid = 1 / (1 + np.exp(-score))

                        grad = (1 - sigmoid)
                        self.embeddings[center] += lr * grad * context_embeddings[context]
                        context_embeddings[context] += lr * grad * self.embeddings[center]

                        # Negative samples
                        for _ in range(self.config.negative_samples):
                            neg = rng.integers(n_nodes)
                            if neg == center or neg == context:
                                continue

                            neg_score = np.dot(self.embeddings[center], context_embeddings[neg])
                            neg_sigmoid = 1 / (1 + np.exp(-neg_score))

                            neg_grad = -neg_sigmoid
                            self.embeddings[center] += lr * neg_grad * context_embeddings[neg]
                            context_embeddings[neg] += lr * neg_grad * self.embeddings[center]

    def _generate_walks(
        self,
        adj_matrix: sparse.csr_matrix,
        rng: np.random.Generator,
    ) -> list[list[int]]:
        """Generate random walks from each node.

        Args:
            adj_matrix: Adjacency matrix
            rng: Random number generator

        Returns:
            List of walks (each walk is list of node indices)
        """
        n_nodes = adj_matrix.shape[0]
        walks = []

        for start_node in range(n_nodes):
            for _ in range(self.config.num_walks):
                walk = [start_node]
                current = start_node

                for _ in range(self.config.walk_length - 1):
                    neighbors = adj_matrix[current].indices
                    if len(neighbors) == 0:
                        break

                    # Simple uniform random walk
                    next_node = rng.choice(neighbors)
                    walk.append(next_node)
                    current = next_node

                walks.append(walk)

        return walks

    def get_embedding(self, node_id: str) -> NDArray[np.float64] | None:
        """Get embedding for a node.

        Args:
            node_id: Node identifier

        Returns:
            Embedding vector or None if not found
        """
        if node_id not in self.node_id_to_idx:
            return None
        return self.embeddings[self.node_id_to_idx[node_id]]

    def get_all_embeddings(self) -> dict[str, NDArray[np.float64]]:
        """Get all node embeddings.

        Returns:
            Dictionary mapping node_id to embedding
        """
        return {
            nid: self.embeddings[idx]
            for nid, idx in self.node_id_to_idx.items()
        }


class ContentEmbeddings:
    """Generate content embeddings from engagement patterns."""

    def __init__(self, config: EmbeddingConfig | None = None):
        """Initialize embedding model.

        Args:
            config: Embedding configuration
        """
        self.config = config or EmbeddingConfig()

        self.embeddings: dict[str, NDArray[np.float64]] = {}

    def fit_from_user_embeddings(
        self,
        user_embeddings: dict[str, NDArray[np.float64]],
        content_engagements: dict[str, list[tuple[str, float]]],
    ) -> None:
        """Compute content embeddings as weighted average of engaging users.

        Args:
            user_embeddings: User ID to embedding mapping
            content_engagements: Content ID to list of (user_id, weight) tuples
        """
        for content_id, engagements in content_engagements.items():
            if not engagements:
                continue

            weighted_sum = None
            total_weight = 0.0

            for user_id, weight in engagements:
                if user_id not in user_embeddings:
                    continue

                user_emb = user_embeddings[user_id]
                if weighted_sum is None:
                    weighted_sum = weight * user_emb.copy()
                else:
                    weighted_sum += weight * user_emb
                total_weight += weight

            if weighted_sum is not None and total_weight > 0:
                self.embeddings[content_id] = weighted_sum / total_weight

    def fit_from_interactions(
        self,
        interactions: list[tuple[str, str, str, float]],
        embedding_dim: int | None = None,
    ) -> None:
        """Fit content embeddings from raw interactions.

        Uses matrix factorization internally.

        Args:
            interactions: List of (user_id, content_id, interaction_type, weight) tuples
            embedding_dim: Override embedding dimension
        """
        dim = embedding_dim or self.config.embedding_dim

        # Aggregate by user-content pair
        aggregated = defaultdict(float)
        for user_id, content_id, int_type, weight in interactions:
            aggregated[(user_id, content_id)] += weight

        # Convert to MF format
        mf_interactions = [
            (user_id, content_id, weight)
            for (user_id, content_id), weight in aggregated.items()
        ]

        # Train MF model
        mf = MatrixFactorizationEmbeddings(self.config)
        mf.fit(mf_interactions)

        # Extract content (item) embeddings
        if mf.item_embeddings is not None:
            idx_to_item = {v: k for k, v in mf.item_id_to_idx.items()}
            for idx in range(len(mf.item_embeddings)):
                content_id = idx_to_item[idx]
                self.embeddings[content_id] = mf.item_embeddings[idx]

    def get_embedding(self, content_id: str) -> NDArray[np.float64] | None:
        """Get embedding for content.

        Args:
            content_id: Content identifier

        Returns:
            Embedding vector or None if not found
        """
        return self.embeddings.get(content_id)

    def get_similar_content(
        self,
        content_id: str,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Find similar content by embedding similarity.

        Args:
            content_id: Query content ID
            top_k: Number of similar items to return

        Returns:
            List of (content_id, similarity) tuples
        """
        if content_id not in self.embeddings:
            return []

        query_emb = self.embeddings[content_id]
        query_norm = np.linalg.norm(query_emb)

        if query_norm == 0:
            return []

        results = []
        for cid, emb in self.embeddings.items():
            if cid == content_id:
                continue

            emb_norm = np.linalg.norm(emb)
            if emb_norm == 0:
                continue

            similarity = np.dot(query_emb, emb) / (query_norm * emb_norm)
            results.append((cid, float(similarity)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


class UserEmbeddingModel:
    """Unified user embedding model combining multiple sources."""

    def __init__(self, config: EmbeddingConfig | None = None):
        """Initialize embedding model.

        Args:
            config: Embedding configuration
        """
        self.config = config or EmbeddingConfig()

        self.interaction_embeddings: MatrixFactorizationEmbeddings | None = None
        self.network_embeddings: Node2VecEmbeddings | None = None

        self.combined_embeddings: dict[str, NDArray[np.float64]] = {}

    def fit(
        self,
        interactions: list[tuple[str, str, float]],
        network: sparse.csr_matrix,
        user_ids: list[str],
    ) -> None:
        """Fit user embeddings from interactions and network.

        Args:
            interactions: List of (user_id, item_id, weight) tuples
            network: Social network adjacency matrix
            user_ids: List of user IDs matching network matrix
        """
        # Fit interaction-based embeddings
        self.interaction_embeddings = MatrixFactorizationEmbeddings(self.config)
        self.interaction_embeddings.fit(interactions)

        # Fit network-based embeddings
        self.network_embeddings = Node2VecEmbeddings(self.config)
        self.network_embeddings.fit(network, user_ids)

        # Combine embeddings
        self._combine_embeddings(user_ids)

    def _combine_embeddings(self, user_ids: list[str]) -> None:
        """Combine interaction and network embeddings.

        Args:
            user_ids: List of all user IDs
        """
        dim = self.config.embedding_dim

        for user_id in user_ids:
            embs = []

            # Interaction embedding
            int_emb = self.interaction_embeddings.get_user_embedding(user_id)
            if int_emb is not None:
                embs.append(int_emb)

            # Network embedding
            net_emb = self.network_embeddings.get_embedding(user_id)
            if net_emb is not None:
                embs.append(net_emb)

            if embs:
                # Concatenate and project to original dimension
                combined = np.concatenate(embs)
                # Simple average for same-dimension case
                if len(embs) > 1 and all(e.shape[0] == dim for e in embs):
                    combined = np.mean(embs, axis=0)
                self.combined_embeddings[user_id] = combined

    def get_embedding(self, user_id: str) -> NDArray[np.float64] | None:
        """Get combined embedding for user.

        Args:
            user_id: User identifier

        Returns:
            Combined embedding vector or None
        """
        return self.combined_embeddings.get(user_id)

    def get_all_embeddings(self) -> dict[str, NDArray[np.float64]]:
        """Get all user embeddings.

        Returns:
            Dictionary mapping user_id to embedding
        """
        return self.combined_embeddings.copy()


def create_embeddings_from_simulation(
    users: dict[str, Any],
    interactions: list[Any],
    posts: dict[str, Any],
    embedding_dim: int = 64,
) -> dict[str, Any]:
    """Create user and content embeddings from simulation data.

    Args:
        users: Dictionary of user objects
        interactions: List of interaction objects
        posts: Dictionary of post objects
        embedding_dim: Embedding dimension

    Returns:
        Dictionary with 'user_embeddings' and 'content_embeddings'
    """
    config = EmbeddingConfig(embedding_dim=embedding_dim)

    # Build interaction data
    user_post_interactions = []
    for interaction in interactions:
        weight_map = {
            "view": 0.1,
            "like": 0.5,
            "share": 1.0,
            "comment": 0.8,
        }
        int_type = interaction.interaction_type.value
        weight = weight_map.get(int_type, 0.1)
        user_post_interactions.append((interaction.user_id, interaction.post_id, weight))

    # Build network adjacency
    user_ids = list(users.keys())
    user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    n_users = len(user_ids)

    row_indices = []
    col_indices = []
    for uid, user in users.items():
        from_idx = user_id_to_idx[uid]
        for followed_id in getattr(user, 'following', set()):
            if followed_id in user_id_to_idx:
                to_idx = user_id_to_idx[followed_id]
                row_indices.append(from_idx)
                col_indices.append(to_idx)

    network = sparse.csr_matrix(
        (np.ones(len(row_indices)), (row_indices, col_indices)),
        shape=(n_users, n_users),
    )

    # Train user embedding model
    user_model = UserEmbeddingModel(config)
    user_model.fit(user_post_interactions, network, user_ids)

    # Train content embedding model
    content_engagements = defaultdict(list)
    for interaction in interactions:
        weight_map = {
            "view": 0.1,
            "like": 0.5,
            "share": 1.0,
            "comment": 0.8,
        }
        int_type = interaction.interaction_type.value
        weight = weight_map.get(int_type, 0.1)
        content_engagements[interaction.post_id].append((interaction.user_id, weight))

    content_model = ContentEmbeddings(config)
    content_model.fit_from_user_embeddings(
        user_model.get_all_embeddings(),
        dict(content_engagements),
    )

    return {
        "user_embeddings": user_model.get_all_embeddings(),
        "content_embeddings": content_model.embeddings,
    }
