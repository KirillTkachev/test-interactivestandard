import os
import json

import urllib.parse as ul
from zipfile import ZipFile
import click


from entities.downloading_params import(
    DownloadingParams,
    read_downloading_params
)


def get_params(config_path: str) -> DownloadingParams:
    return read_downloading_params(config_path)


def download_data_from_yandex(base_url: str, url: str, output_folder: str) -> str:
    url = ul.quote_plus(url)
    res = os.popen('wget -qO - {}{}'.format(base_url, url)).read()
    json_res = json.loads(res)
    filename = ul.parse_qs(ul.urlparse(json_res['href']).query)['filename'][0]
    os.system("wget '{}' -P '{}' -O '{}'".format(json_res['href'], output_folder, filename))
    return filename


def unzip_file(file_path: str, output_path: str):
    ZipFile.extractall(ZipFile(file_path), output_path)
    os.remove(file_path)


def download_dataset(config_path: str):
    params = get_params(config_path)
    filename = download_data_from_yandex(params.base_url, params.dataset_url, 
                                         params.dataset_output_path)
    unzip_file(filename, params.dataset_output_path)


@click.command(name="download_dataset")
@click.argument("config_path")
def download_dataset_command(config_path: str):
    download_dataset(config_path)


if __name__ == "__main__":
    download_dataset_command()
