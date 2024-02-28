import argparse

import torch
from transformers import TrainingArguments

from transfergraph.dataset.base_dataset import BaseDataset
from transfergraph.dataset.task import TaskType


class BaseTrainer:
    def __init__(
            self,
            model: torch.nn.Module,
            dataset: BaseDataset,
            all_training_argument: TrainingArguments,
            task_type: TaskType,
            args: argparse.Namespace
    ):
        self.model = model
        self.dataset = dataset
        self.all_training_argument = all_training_argument
        self.task_type = task_type
        self.args = args
