from dataclasses import dataclass

from marshmallow_dataclass import class_schema
import yaml


@dataclass
class DownloadingParams:
    base_url: str
    dataset_url: str
    dataset_output_path: str


DownloadingParamsSchema = class_schema(DownloadingParams)


def read_downloading_params(path: str) -> DownloadingParams:
    with open(path, "r") as input_stream:
        schema = DownloadingParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
