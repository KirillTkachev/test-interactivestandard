import os
import logging
import sys

import click
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from entities.dataset_params import(
    DatasetParams,
    read_cleaning_params
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/cleaning.log"),
        logging.StreamHandler(sys.stdout)
    ]
)


def get_params(config_path: str) -> DatasetParams:
    dataset_params = read_cleaning_params(config_path)
    return dataset_params


def clean_dataset(config_path: str):
    params = get_params(config_path)
    logging.info(f"Start dataset cleaning with params: {params}")
    dataframe = pd.read_csv(params.raw_csv)
    dataframe.drop(columns=dataframe.columns[0], axis=1, inplace=True)

    true_ids = []
    images = os.listdir(params.raw_images)
    progress_bar = tqdm(total=len(images), desc="Cleaning dataset")
    for path in images:
        full_path = params.raw_images + '/' + path
        image = Image.open(full_path)
        if image.size != (params.width_to_remove, params.height_to_remove):
            image.save(params.cleaned_images + path)
            true_ids.append(path)
        progress_bar.update(1)

    cleaned_dataframe = dataframe[dataframe['file_name'].isin(true_ids)]
    logging.info(f"The size of cleaned dataset: {cleaned_dataframe.shape}")
    cleaned_grouped_dataframe = cleaned_dataframe.groupby('cluster_id').apply(pd.DataFrame)
    cleaned_grouped_dataframe.to_csv(params.cleaned_csv, index=False)


@click.command(name="clean_dataset")
@click.argument("config_path")
def clean_dataset_command(config_path: str):
    clean_dataset(config_path)


if __name__ == "__main__":
    clean_dataset_command()
