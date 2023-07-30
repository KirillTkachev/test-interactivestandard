from dataclasses import dataclass


@dataclass
class ClusteringParams:
    embeddings_path: str
    annoy_index_metric: str
    annoy_index_trees: int
    linkage_method: str
    fcluster_criterion: str
    fcluster_threshold: float
