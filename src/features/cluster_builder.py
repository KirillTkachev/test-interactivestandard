from collections import defaultdict

from annoy import AnnoyIndex
from scipy.cluster.hierarchy import linkage, fcluster
import numpy as np

from entities.clustering_params import ClusteringParams


class ClusterBuilder:
    def __init__(self, clustering_params: ClusteringParams):
        self.__annoy_index_metric = clustering_params.annoy_index_metric
        self.__annoy_index_trees = clustering_params.annoy_index_trees
        self.__clustering_criterion = clustering_params.fcluster_criterion
        self.__fcluster_threshold = clustering_params.fcluster_threshold
        self.__linkage_method = clustering_params.linkage_method
        self.__embeddings = np.load(clustering_params.embeddings_path)

    def build_image_clusters(self, all_image_ids: list[str]) -> (np.ndarray, dict):
        annoy_index = self.__build_annoy_index()
        distances = self.__compute_distance_matrix(annoy_index)
        labels = self.__apply_clustering(distances, self.__fcluster_threshold)
        image_id_clusters = defaultdict(set)
        for image_id, cluster_label in zip(all_image_ids, labels):
            image_id_clusters[cluster_label].add(image_id)

        return (labels, image_id_clusters)

    def __build_annoy_index(self) -> AnnoyIndex:
        embeddings = np.array(self.__embeddings)
        n_dimensions = embeddings.shape[1]

        annoy_index = AnnoyIndex(n_dimensions, self.__annoy_index_metric)
        for i, embedding in enumerate(embeddings):
            annoy_index.add_item(i, embedding)

        annoy_index.build(self.__annoy_index_trees)
        return annoy_index

    def __compute_distance_matrix(self, annoy_index: AnnoyIndex) -> list[float]:
        embeddings_length = len(self.__embeddings)
        distances = []

        for i in range(embeddings_length):
            for j in range(i + 1, embeddings_length):
                distance = annoy_index.get_distance(i, j)
                distances.append(distance)

        return distances

    def __apply_clustering(self, distances: list[float], threshold: float) -> np.ndarray:
        condensed_distances = np.array(distances)
        clustering = linkage(condensed_distances, method=self.__linkage_method,
                             optimal_ordering=True)
        return fcluster(clustering, t=threshold, criterion=self.__clustering_criterion)
