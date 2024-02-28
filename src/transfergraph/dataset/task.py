from enum import Enum

MAPPING_HUGGINGFACE_TASK_TYPE = {
    "image_classification": "image-classification",
    "sequence_classification": "text-classification",
    "token_classification": "token-classification",
}


class TaskType(Enum):
    IMAGE_CLASSIFICATION = "image_classification"
    SEQUENCE_CLASSIFICATION = "sequence_classification"
    TOKEN_CLASSIFICATION = "token_classification"

    def to_huggingface_task_type(self) -> str:
        return MAPPING_HUGGINGFACE_TASK_TYPE[self.value]
