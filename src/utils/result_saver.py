import shutil
from pathlib import Path
from entities.result_params import ResultParams


class ResultSaver:
    def __init__(self, result_params: ResultParams, clusters):
        self.__output_directory = result_params.output_path
        self.__image_id_clusters = clusters

    def organize_images(self, images_to_paths: dict[str, Path]):
        for idx, image_id_cluster in enumerate(self.__image_id_clusters.values()):
            if len(image_id_cluster) < 2:
                continue

            self.__move_images_to_directory(f"cluster_{idx}", image_id_cluster, images_to_paths)

        unique_image_ids = set(
            images_to_paths.keys()) - {image_id for cluster in self.__image_id_clusters.values()
                                       for image_id in cluster if len(cluster) >= 2}
        self.__move_images_to_directory("unique", unique_image_ids, images_to_paths)

    def __move_images_to_directory(self, folder_name: str, image_ids: set,
                                   images_to_paths: dict[str, Path]):
        __output_directory = Path(self.__output_directory + "/" + folder_name)
        __output_directory.mkdir(parents=True, exist_ok=True)

        for image_id in image_ids:
            source = images_to_paths[image_id]
            destination = __output_directory / source.name
            shutil.copy(source, destination)
