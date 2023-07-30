from dataclasses import dataclass
from typing import List

@dataclass
class EmbeddingParams:
    input_data_path: str
    output_path: str
    clip_model: str
    device: str
    allowed_extensions: List[str]
    batch_size: int
    rebuild_embeddings: bool
