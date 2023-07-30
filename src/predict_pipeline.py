import sys
import logging
import json

import click
import pandas as pd

from utils.utils import (
    evaluate_model,
    create_target_dataframe
)

import features.cluster_builder as cluster_builder
import features.embeddings_builder as embeddings_builder
from utils.result_saver import ResultSaver
from entities.inference_params import (
    read_inference_params,
    InferenceParams
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/inference.log"),
        logging.StreamHandler(sys.stdout)
    ]
)


def inference_pipeline(config_path: str):
    inference_pipeline_params = read_inference_params(config_path)
    run_inference_pipeline(inference_pipeline_params)


def run_inference_pipeline(inference_pipeline_params: InferenceParams):
    logging.info(f"start of inference with params {inference_pipeline_params}")

    embedding_params = inference_pipeline_params.embedding_params
    clustering_params = inference_pipeline_params.clustering_params
    result_params = inference_pipeline_params.result_params

    logging.info("start of building embeddings")

    embeddings_builder_ = embeddings_builder.EmbeddingsBuilder(embedding_params)
    if embedding_params.rebuild_embeddings:
        embeddings_builder_.build_embeddings()
    images_to_pathes, all_images_ids = embeddings_builder_.get_images_to_paths()

    logging.info("start of clusterization")

    cluster_builder_ = cluster_builder.ClusterBuilder(clustering_params)
    labels, image_ids_clusters = cluster_builder_.build_image_clusters(all_images_ids)

    logging.info("saving results...")

    result_saver_ = ResultSaver(result_params, image_ids_clusters)
    result_saver_.organize_images(images_to_pathes)

    logging.info("results saved")

    cleaned_dataframe = pd.read_csv(inference_pipeline_params.ground_truth_path)

    metrics = evaluate_model(cleaned_dataframe, create_target_dataframe(labels, images_to_pathes))

    logging.info(f"metrics are: {metrics}")

    with open(inference_pipeline_params.metric_path, "w") as metric_file:
        json.dump(metrics, metric_file)


@click.command(name="inference_pipeline")
@click.argument("config_path")
def inference_pipeline_command(config_path: str):
    inference_pipeline(config_path)


if __name__ == "__main__":
    inference_pipeline_command()
