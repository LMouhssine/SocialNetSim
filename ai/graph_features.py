"""Graph-based feature extraction for ML models.

Provides network structure features including:
- PageRank and centrality measures
- Clustering coefficient
- Cascade structural virality (Witten-Sander)
- Community-based features
"""

from dataclasses import dataclass, field
from typing import Any
from collections import defaultdict

import numpy as np
from scipy import sparse
from numpy.typing import NDArray


@dataclass
class GraphFeatureConfig:
    """Configuration for graph feature extraction.

    Attributes:
        pagerank_damping: PageRank damping factor
        pagerank_iterations: Maximum PageRank iterations
        use_sparse: Use sparse matrix operations
        community_resolution: Resolution for community detection
        max_path_length: Maximum path length for betweenness estimation
    """

    pagerank_damping: float = 0.85
    pagerank_iterations: int = 100
    use_sparse: bool = True
    community_resolution: float = 1.0
    max_path_length: int = 3


@dataclass
class UserGraphFeatures:
    """Graph features for a user.

    Attributes:
        pagerank: PageRank score
        in_degree: Number of followers
        out_degree: Number of following
        betweenness_estimate: Estimated betweenness centrality
        clustering_coefficient: Local clustering coefficient
        eigenvector_centrality: Eigenvector centrality
        community_id: Community membership
        community_size: Size of user's community
        bridging_score: Score for bridging communities
    """

    pagerank: float = 0.0
    in_degree: int = 0
    out_degree: int = 0
    betweenness_estimate: float = 0.0
    clustering_coefficient: float = 0.0
    eigenvector_centrality: float = 0.0
    community_id: int = -1
    community_size: int = 0
    bridging_score: float = 0.0


@dataclass
class CascadeGraphFeatures:
    """Graph features for a cascade.

    Attributes:
        structural_virality: Witten-Sander structural virality measure
        max_depth: Maximum propagation depth
        max_breadth: Maximum breadth at any level
        avg_branching_factor: Average number of children per node
        shape_factor: Breadth/depth ratio
        root_influence: Root node's influence score
        avg_path_length: Average path length from root
    """

    structural_virality: float = 0.0
    max_depth: int = 0
    max_breadth: int = 0
    avg_branching_factor: float = 0.0
    shape_factor: float = 0.0
    root_influence: float = 0.0
    avg_path_length: float = 0.0


class GraphFeatureExtractor:
    """Extracts graph-based features from network data."""

    def __init__(self, config: GraphFeatureConfig | None = None):
        """Initialize extractor.

        Args:
            config: Feature extraction configuration
        """
        self.config = config or GraphFeatureConfig()

        # Cached computations
        self._pagerank_cache: dict[str, float] | None = None
        self._community_cache: dict[str, int] | None = None
        self._cache_valid = False

    def build_adjacency_matrix(
        self,
        users: dict[str, Any],
        user_id_to_idx: dict[str, int] | None = None,
    ) -> tuple[sparse.csr_matrix, dict[str, int]]:
        """Build sparse adjacency matrix from user following relationships.

        Args:
            users: Dictionary of user objects with 'following' attribute
            user_id_to_idx: Optional pre-built index mapping

        Returns:
            Tuple of (adjacency matrix, user_id to index mapping)
        """
        if user_id_to_idx is None:
            user_ids = list(users.keys())
            user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}

        n_users = len(user_id_to_idx)
        row_indices = []
        col_indices = []

        for user_id, user in users.items():
            if user_id not in user_id_to_idx:
                continue
            from_idx = user_id_to_idx[user_id]

            following = getattr(user, 'following', set())
            for followed_id in following:
                if followed_id in user_id_to_idx:
                    to_idx = user_id_to_idx[followed_id]
                    row_indices.append(from_idx)
                    col_indices.append(to_idx)

        data = np.ones(len(row_indices), dtype=np.float32)
        adj_matrix = sparse.csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(n_users, n_users),
        )

        return adj_matrix, user_id_to_idx

    def compute_pagerank(
        self,
        adj_matrix: sparse.csr_matrix,
        damping: float | None = None,
        max_iter: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute PageRank scores.

        Args:
            adj_matrix: Adjacency matrix (row i -> col j means i follows j)
            damping: Damping factor (default from config)
            max_iter: Maximum iterations (default from config)

        Returns:
            Array of PageRank scores
        """
        damping = damping or self.config.pagerank_damping
        max_iter = max_iter or self.config.pagerank_iterations

        n = adj_matrix.shape[0]
        if n == 0:
            return np.array([])

        # Transpose: we want links pointing TO a node
        adj_t = adj_matrix.T.tocsr()

        # Normalize columns (out-degree normalization)
        out_degree = np.array(adj_matrix.sum(axis=1)).flatten()
        out_degree[out_degree == 0] = 1  # Avoid division by zero

        # Build transition matrix
        # P[i,j] = 1/out_degree[j] if j->i, else 0
        col_indices = adj_t.indices
        row_ptr = adj_t.indptr
        data = adj_t.data.copy()

        for i in range(n):
            start, end = row_ptr[i], row_ptr[i + 1]
            for j in range(start, end):
                source = col_indices[j]
                data[j] = 1.0 / out_degree[source]

        P = sparse.csr_matrix((data, adj_t.indices, adj_t.indptr), shape=(n, n))

        # Power iteration
        pr = np.ones(n) / n
        teleport = np.ones(n) / n

        for _ in range(max_iter):
            pr_new = damping * P.dot(pr) + (1 - damping) * teleport
            if np.allclose(pr, pr_new, rtol=1e-6):
                break
            pr = pr_new

        return pr

    def compute_clustering_coefficients(
        self,
        adj_matrix: sparse.csr_matrix,
    ) -> NDArray[np.float64]:
        """Compute local clustering coefficients.

        Args:
            adj_matrix: Adjacency matrix

        Returns:
            Array of clustering coefficients
        """
        n = adj_matrix.shape[0]
        if n == 0:
            return np.array([])

        # Make undirected for clustering
        adj_undirected = adj_matrix + adj_matrix.T
        adj_undirected.data[:] = 1  # Binary

        coefficients = np.zeros(n)

        for i in range(n):
            # Get neighbors
            neighbors = adj_undirected[i].indices
            k = len(neighbors)

            if k < 2:
                coefficients[i] = 0.0
                continue

            # Count edges among neighbors
            neighbor_subgraph = adj_undirected[neighbors][:, neighbors]
            edges_among_neighbors = neighbor_subgraph.nnz / 2  # Undirected

            # Clustering coefficient
            max_edges = k * (k - 1) / 2
            coefficients[i] = edges_among_neighbors / max_edges

        return coefficients

    def estimate_betweenness(
        self,
        adj_matrix: sparse.csr_matrix,
        sample_size: int = 100,
        seed: int | None = None,
    ) -> NDArray[np.float64]:
        """Estimate betweenness centrality using sampling.

        Full betweenness is O(n^3), so we sample source nodes.

        Args:
            adj_matrix: Adjacency matrix
            sample_size: Number of source nodes to sample
            seed: Random seed

        Returns:
            Array of estimated betweenness scores
        """
        n = adj_matrix.shape[0]
        if n == 0:
            return np.array([])

        rng = np.random.default_rng(seed)
        betweenness = np.zeros(n)

        # Sample source nodes
        sources = rng.choice(n, size=min(sample_size, n), replace=False)

        # Make undirected for paths
        adj_undirected = adj_matrix + adj_matrix.T
        adj_undirected.data[:] = 1

        for source in sources:
            # BFS from source
            distances = np.full(n, -1)
            distances[source] = 0
            predecessors = [[] for _ in range(n)]
            num_paths = np.zeros(n)
            num_paths[source] = 1

            queue = [source]
            order = []

            while queue:
                current = queue.pop(0)
                order.append(current)

                neighbors = adj_undirected[current].indices
                for neighbor in neighbors:
                    if distances[neighbor] == -1:
                        distances[neighbor] = distances[current] + 1
                        queue.append(neighbor)

                    if distances[neighbor] == distances[current] + 1:
                        predecessors[neighbor].append(current)
                        num_paths[neighbor] += num_paths[current]

            # Accumulate betweenness
            delta = np.zeros(n)
            for node in reversed(order[1:]):  # Exclude source
                for pred in predecessors[node]:
                    delta[pred] += (num_paths[pred] / num_paths[node]) * (1 + delta[node])
                betweenness[node] += delta[node]

        # Normalize
        if sample_size > 0:
            betweenness *= n / sample_size

        return betweenness

    def detect_communities_louvain_simple(
        self,
        adj_matrix: sparse.csr_matrix,
        resolution: float | None = None,
    ) -> NDArray[np.int32]:
        """Simple community detection using label propagation.

        This is a simplified version; full Louvain would be more accurate.

        Args:
            adj_matrix: Adjacency matrix
            resolution: Resolution parameter (not used in label prop)

        Returns:
            Array of community IDs
        """
        n = adj_matrix.shape[0]
        if n == 0:
            return np.array([], dtype=np.int32)

        # Make undirected
        adj_undirected = adj_matrix + adj_matrix.T

        # Initialize each node in its own community
        labels = np.arange(n, dtype=np.int32)

        # Label propagation
        max_iter = 50
        for _ in range(max_iter):
            changed = False
            order = np.random.permutation(n)

            for node in order:
                neighbors = adj_undirected[node].indices
                if len(neighbors) == 0:
                    continue

                # Count neighbor labels
                label_counts = defaultdict(int)
                for neighbor in neighbors:
                    label_counts[labels[neighbor]] += 1

                # Assign most common label
                best_label = max(label_counts.keys(), key=lambda l: label_counts[l])
                if best_label != labels[node]:
                    labels[node] = best_label
                    changed = True

            if not changed:
                break

        # Relabel to consecutive integers
        unique_labels = np.unique(labels)
        label_map = {old: new for new, old in enumerate(unique_labels)}
        labels = np.array([label_map[l] for l in labels], dtype=np.int32)

        return labels

    def compute_bridging_scores(
        self,
        adj_matrix: sparse.csr_matrix,
        communities: NDArray[np.int32],
    ) -> NDArray[np.float64]:
        """Compute bridging scores for nodes.

        Bridging score = fraction of neighbors in different communities.

        Args:
            adj_matrix: Adjacency matrix
            communities: Community assignments

        Returns:
            Array of bridging scores
        """
        n = adj_matrix.shape[0]
        if n == 0:
            return np.array([])

        # Make undirected
        adj_undirected = adj_matrix + adj_matrix.T

        scores = np.zeros(n)

        for i in range(n):
            neighbors = adj_undirected[i].indices
            if len(neighbors) == 0:
                continue

            my_community = communities[i]
            other_community = sum(
                1 for n in neighbors if communities[n] != my_community
            )
            scores[i] = other_community / len(neighbors)

        return scores

    def extract_user_features(
        self,
        users: dict[str, Any],
    ) -> dict[str, UserGraphFeatures]:
        """Extract graph features for all users.

        Args:
            users: Dictionary of user objects

        Returns:
            Dictionary mapping user_id to features
        """
        # Build adjacency matrix
        adj_matrix, user_id_to_idx = self.build_adjacency_matrix(users)
        idx_to_user_id = {v: k for k, v in user_id_to_idx.items()}
        n = len(user_id_to_idx)

        if n == 0:
            return {}

        # Compute various centrality measures
        pagerank = self.compute_pagerank(adj_matrix)
        clustering = self.compute_clustering_coefficients(adj_matrix)
        betweenness = self.estimate_betweenness(adj_matrix, sample_size=min(100, n))
        communities = self.detect_communities_louvain_simple(adj_matrix)
        bridging = self.compute_bridging_scores(adj_matrix, communities)

        # Compute degrees
        in_degrees = np.array(adj_matrix.sum(axis=0)).flatten()  # Followers
        out_degrees = np.array(adj_matrix.sum(axis=1)).flatten()  # Following

        # Community sizes
        community_sizes = np.bincount(communities)

        # Build feature objects
        features = {}
        for user_id, idx in user_id_to_idx.items():
            features[user_id] = UserGraphFeatures(
                pagerank=float(pagerank[idx]),
                in_degree=int(in_degrees[idx]),
                out_degree=int(out_degrees[idx]),
                betweenness_estimate=float(betweenness[idx]),
                clustering_coefficient=float(clustering[idx]),
                eigenvector_centrality=0.0,  # Would need power iteration
                community_id=int(communities[idx]),
                community_size=int(community_sizes[communities[idx]]),
                bridging_score=float(bridging[idx]),
            )

        return features


class CascadeGraphAnalyzer:
    """Analyzes cascade spreading patterns as graphs."""

    def compute_structural_virality(
        self,
        cascade_tree: dict[str, list[str]],
        root: str,
    ) -> float:
        """Compute Witten-Sander structural virality.

        Structural virality = average path length between all pairs.
        Higher for broadcast-like, lower for chain-like.

        Args:
            cascade_tree: Dictionary mapping parent -> list of children
            root: Root node ID

        Returns:
            Structural virality score
        """
        # Build full node set
        nodes = {root}
        for parent, children in cascade_tree.items():
            nodes.add(parent)
            nodes.update(children)

        if len(nodes) <= 1:
            return 0.0

        # Build parent lookup
        parent_of = {}
        for parent, children in cascade_tree.items():
            for child in children:
                parent_of[child] = parent

        # Compute pairwise distances using BFS from each node
        total_distance = 0
        pair_count = 0

        nodes_list = list(nodes)
        n = len(nodes_list)

        for i, node in enumerate(nodes_list):
            # BFS from this node
            distances = {node: 0}
            queue = [node]

            while queue:
                current = queue.pop(0)

                # Children
                for child in cascade_tree.get(current, []):
                    if child not in distances:
                        distances[child] = distances[current] + 1
                        queue.append(child)

                # Parent
                if current in parent_of:
                    parent = parent_of[current]
                    if parent not in distances:
                        distances[parent] = distances[current] + 1
                        queue.append(parent)

            # Sum distances to other nodes
            for j in range(i + 1, n):
                other = nodes_list[j]
                if other in distances:
                    total_distance += distances[other]
                    pair_count += 1

        if pair_count == 0:
            return 0.0

        avg_distance = total_distance / pair_count
        return avg_distance

    def compute_cascade_features(
        self,
        cascade_tree: dict[str, list[str]],
        root: str,
        user_influence: dict[str, float] | None = None,
    ) -> CascadeGraphFeatures:
        """Compute all graph features for a cascade.

        Args:
            cascade_tree: Dictionary mapping parent -> list of children
            root: Root node ID
            user_influence: Optional user influence scores

        Returns:
            Cascade graph features
        """
        features = CascadeGraphFeatures()

        if not cascade_tree:
            return features

        # Build node set and depth mapping
        nodes = {root}
        depths = {root: 0}
        parent_of = {}

        queue = [root]
        while queue:
            current = queue.pop(0)
            for child in cascade_tree.get(current, []):
                if child not in nodes:
                    nodes.add(child)
                    depths[child] = depths[current] + 1
                    parent_of[child] = current
                    queue.append(child)

        # Structural virality
        features.structural_virality = self.compute_structural_virality(
            cascade_tree, root
        )

        # Max depth
        features.max_depth = max(depths.values()) if depths else 0

        # Breadth at each level
        depth_counts = defaultdict(int)
        for node, depth in depths.items():
            depth_counts[depth] += 1

        features.max_breadth = max(depth_counts.values()) if depth_counts else 0

        # Shape factor
        if features.max_depth > 0:
            features.shape_factor = features.max_breadth / features.max_depth
        else:
            features.shape_factor = 0.0

        # Average branching factor
        if cascade_tree:
            branch_counts = [len(children) for children in cascade_tree.values()]
            features.avg_branching_factor = np.mean(branch_counts)
        else:
            features.avg_branching_factor = 0.0

        # Root influence
        if user_influence and root in user_influence:
            features.root_influence = user_influence[root]

        # Average path length from root
        features.avg_path_length = np.mean(list(depths.values())) if depths else 0.0

        return features


class InteractionGraphBuilder:
    """Builds interaction graphs for feature extraction."""

    def build_interaction_graph(
        self,
        interactions: list[Any],
        interaction_type: str | None = None,
    ) -> tuple[sparse.csr_matrix, dict[str, int]]:
        """Build graph from user interactions.

        Args:
            interactions: List of interaction objects
            interaction_type: Filter by type (None = all)

        Returns:
            Tuple of (adjacency matrix, user_id to index mapping)
        """
        # Collect all user IDs
        user_ids = set()
        edges = []

        for interaction in interactions:
            if interaction_type and interaction.interaction_type.value != interaction_type:
                continue

            user_id = interaction.user_id
            # Get post author as target
            if hasattr(interaction, 'source_user_id') and interaction.source_user_id:
                target_id = interaction.source_user_id
            else:
                continue

            user_ids.add(user_id)
            user_ids.add(target_id)
            edges.append((user_id, target_id))

        if not user_ids:
            return sparse.csr_matrix((0, 0)), {}

        # Build index
        user_id_to_idx = {uid: i for i, uid in enumerate(sorted(user_ids))}
        n = len(user_id_to_idx)

        # Build matrix
        row_indices = [user_id_to_idx[e[0]] for e in edges]
        col_indices = [user_id_to_idx[e[1]] for e in edges]
        data = np.ones(len(edges), dtype=np.float32)

        adj = sparse.csr_matrix((data, (row_indices, col_indices)), shape=(n, n))

        return adj, user_id_to_idx

    def build_weighted_interaction_graph(
        self,
        interactions: list[Any],
        weight_by_type: dict[str, float] | None = None,
    ) -> tuple[sparse.csr_matrix, dict[str, int]]:
        """Build weighted graph from interactions.

        Args:
            interactions: List of interaction objects
            weight_by_type: Weights for each interaction type

        Returns:
            Tuple of (weighted adjacency matrix, user_id to index mapping)
        """
        weight_by_type = weight_by_type or {
            "view": 0.1,
            "like": 0.5,
            "share": 1.0,
            "comment": 0.8,
        }

        # Collect user IDs and weighted edges
        user_ids = set()
        edge_weights = defaultdict(float)

        for interaction in interactions:
            user_id = interaction.user_id
            if hasattr(interaction, 'source_user_id') and interaction.source_user_id:
                target_id = interaction.source_user_id
            else:
                continue

            user_ids.add(user_id)
            user_ids.add(target_id)

            int_type = interaction.interaction_type.value
            weight = weight_by_type.get(int_type, 0.1)
            edge_weights[(user_id, target_id)] += weight

        if not user_ids:
            return sparse.csr_matrix((0, 0)), {}

        # Build matrix
        user_id_to_idx = {uid: i for i, uid in enumerate(sorted(user_ids))}
        n = len(user_id_to_idx)

        row_indices = []
        col_indices = []
        data = []

        for (from_id, to_id), weight in edge_weights.items():
            row_indices.append(user_id_to_idx[from_id])
            col_indices.append(user_id_to_idx[to_id])
            data.append(weight)

        adj = sparse.csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(n, n),
        )

        return adj, user_id_to_idx


def extract_all_graph_features(
    users: dict[str, Any],
    cascades: dict[str, Any],
    interactions: list[Any],
) -> dict[str, Any]:
    """Extract all graph features for users and cascades.

    Args:
        users: Dictionary of user objects
        cascades: Dictionary of cascade objects
        interactions: List of interaction objects

    Returns:
        Dictionary with 'users' and 'cascades' feature dictionaries
    """
    extractor = GraphFeatureExtractor()
    cascade_analyzer = CascadeGraphAnalyzer()

    results = {
        "users": {},
        "cascades": {},
    }

    # Extract user graph features
    results["users"] = extractor.extract_user_features(users)

    # Extract cascade features
    user_influence = {
        uid: f.pagerank for uid, f in results["users"].items()
    }

    for cascade_id, cascade in cascades.items():
        # Build cascade tree from share_tree attribute
        if hasattr(cascade, 'share_tree'):
            tree = cascade.share_tree
            root = cascade.root_user_id if hasattr(cascade, 'root_user_id') else None

            if root and tree:
                results["cascades"][cascade_id] = cascade_analyzer.compute_cascade_features(
                    tree, root, user_influence
                )

    return results
