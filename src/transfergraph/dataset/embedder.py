import os

import numpy as np
from transformers import PreTrainedModel

from transfergraph.config import get_root_path_string
from transfergraph.dataset.base_dataset import BaseDataset
from transfergraph.dataset.embed_utils import DatasetEmbeddingMethod
from transfergraph.model_selection.utils import extract_features_without_labels


class DatasetEmbedder:
    def __init__(self, probe_model: PreTrainedModel, dataset: BaseDataset, embedding_method: DatasetEmbeddingMethod, task_type: str):
        self.probe_model = probe_model
        self.dataset = dataset
        self.embedding_method = embedding_method
        self.task_type = task_type

    def embed(self):
        features_tensor, features_dimension = extract_features_without_labels(self.dataset.train_loader, self.probe_model)

        if self.embedding_method == DatasetEmbeddingMethod.DOMAIN_SIMILARITY:
            model_name_sanitized = self.probe_model.name_or_path.replace('/', '_')
            dataset_name_sanitized = self.dataset.name.replace('/', '_')
            directory = os.path.join(
                get_root_path_string(),
                "resources",
                "experiments",
                self.task_type,
                'embedded_dataset/',
                self.embedding_method.value,
                model_name_sanitized
            )

            if not os.path.exists(directory):
                os.makedirs(directory)

            np.save(os.path.join(directory, dataset_name_sanitized + f'_feature.npy'), features_tensor)
        else:
            raise Exception(f"Unexpected DatasetEmbeddingMethod: {self.embedding_method.name}")
