from enum import Enum


class DatasetEmbeddingMethod(Enum):
    DOMAIN_SIMILARITY = "domain_similarity"
    TASK2VEC = "task2vec"
