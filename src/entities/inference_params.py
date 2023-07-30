from dataclasses import dataclass
import yaml
from marshmallow_dataclass import class_schema

from .clustering_params import ClusteringParams
from .embedding_params import EmbeddingParams
from .result_params import ResultParams


@dataclass()
class InferenceParams:
    metric_path: str
    ground_truth_path: str
    clustering_params: ClusteringParams
    embedding_params: EmbeddingParams
    result_params: ResultParams


InferenceParamsSchema = class_schema(InferenceParams)


def read_inference_params(path: str) -> InferenceParams:
    with open(path, "r") as input_stream:
        schema = InferenceParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
