from pathlib import Path

import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from tqdm.auto import tqdm
from PIL import Image

from entities.embedding_params import EmbeddingParams


class EmbeddingsBuilder:
    def __init__(self, params: EmbeddingParams):
        self.__input_path = Path(params.input_data_path)
        self.__output_path = params.output_path
        self.__batch_size = params.batch_size
        self.__device = params.device
        self.__allowed_extensions = params.allowed_extensions
        self.__model = CLIPModel.from_pretrained(params.clip_model).to(self.__device)
        self.__processor = CLIPProcessor.from_pretrained(params.clip_model)

    def build_embeddings(self) -> (dict[str, Path], list[str]):
        all_embeddings = []

        images_to_pathes, all_image_ids = self.get_images_to_paths()
        progress_bar = tqdm(total=len(all_image_ids), desc="Generating CLIP embeddings")

        for i in range(0, len(all_image_ids), self.__batch_size):
            batch_image_ids, batch_images = self.__process_image_batch(all_image_ids, images_to_pathes,
                                                                       i, self.__batch_size)
            inputs = self.__processor(images=batch_images, return_tensors="pt",
                                      padding=True).to(self.__device)

            with torch.no_grad():
                outputs = self.__model.get_image_features(**inputs)

            all_embeddings.extend(outputs.cpu().numpy())
            progress_bar.update(len(batch_image_ids))
        np.save(self.__output_path, all_embeddings)

        progress_bar.close()


    def get_images_to_paths(self) -> (dict[str, Path], list[str]):
        images_to_paths = {
            image_path.stem: image_path
            for image_path in self.__input_path.iterdir()
            if image_path.suffix.lower() in self.__allowed_extensions
        }
        return images_to_paths, list(images_to_paths.keys())

    def __process_image_batch(self, all_images_ids: list[str], images_to_pathes: dict[str, Path],
                              start_idx: int, batch_size: int) -> (list[str], list[Image.Image]):
        batch_image_ids = all_images_ids[start_idx: start_idx + batch_size]
        batch_images = []

        for image_id in batch_image_ids:
            image = Image.open(images_to_pathes[image_id])
            image.load()
            batch_images.append(image)
        return batch_image_ids, batch_images
