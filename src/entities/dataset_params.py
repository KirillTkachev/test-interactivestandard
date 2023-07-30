from dataclasses import dataclass

from marshmallow_dataclass import class_schema
import yaml


@dataclass
class DatasetParams:
    raw_csv: str
    cleaned_csv: str
    raw_images: str
    cleaned_images: str
    width_to_remove: int
    height_to_remove: int


CleaningParamsSchema = class_schema(DatasetParams)


def read_cleaning_params(path: str) -> DatasetParams:
    with open(path, "r") as input_stream:
        schema = CleaningParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
