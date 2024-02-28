import math
import os.path
import time

import evaluate
import pandas as pd
import torch.optim
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from huggingface_hub import create_repo, ModelCard, HfApi
from tqdm.auto import tqdm
from transformers import get_scheduler
from transformers.modelcard import TrainingSummary
from transformers.modeling_utils import unwrap_model
from transformers.training_args import ParallelMode

from transfergraph.config import get_root_path_string
from transfergraph.dataset.hugging_face.dataset import HuggingFaceDatasetText
from transfergraph.trainer.base_trainer import BaseTrainer

logger = get_logger(__name__)


class AccelerateTrainer(BaseTrainer):
    def train(self):
        if self.all_training_argument.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self._prepare()
        self._log_train_start(self.max_training_steps, self.total_batch_size)

        time_start = time.time()
        eval_accuracy = 0
        self._evaluate_epoch()

        for epoch in range(self.all_training_argument.num_train_epochs):
            eval_accuracy = self._train_single_epoch()
            self.epoch += 1

        self.accelerator.end_training()
        self.accelerator.wait_for_everyone()

        self.eval_accuracy = eval_accuracy
        self._save_result_to_csv(time_start)
        self._save_model()

    def _prepare(self):
        accelerator = Accelerator(log_with="wandb", project_dir=get_root_path_string())
        accelerator.init_trackers(os.environ.get("WANDB_PROJECT"), config=vars(self.args))
        set_seed(self.all_training_argument.seed)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.all_training_argument.learning_rate)

        num_update_steps_per_epoch = (math.ceil(len(self.dataset.train_loader) / self.all_training_argument.gradient_accumulation_steps))
        max_training_steps = num_update_steps_per_epoch * self.all_training_argument.num_train_epochs
        self.total_batch_size = self.all_training_argument.per_device_train_batch_size * accelerator.num_processes * self.all_training_argument.gradient_accumulation_steps
        lr_scheduler = get_scheduler(
            self.all_training_argument.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_update_steps_per_epoch * self.all_training_argument.num_train_epochs
        )
        self.train_dataloader, self.eval_dataloader, self.model, self.optimizer, self.lr_scheduler = accelerator.prepare(
            self.dataset.train_loader,
            self.dataset.eval_loader,
            self.model,
            optimizer,
            lr_scheduler,
        )
        self.metric = evaluate.load("accuracy")
        self.accelerator = accelerator
        self.max_training_steps = max_training_steps
        self.epoch = 0
        self.completed_steps = 0
        self.progress_bar = tqdm(range(self.max_training_steps))
        self.train_log = []

    def _evaluate_epoch(self, train_loss=None):
        self.model.eval()

        for batch in self.eval_dataloader:
            with torch.no_grad():
                outputs = self.model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions, targets = self.accelerator.gather_for_metrics((predictions, batch['labels']))
            self.metric.add_batch(predictions=predictions, references=targets)

        eval_metric = self.metric.compute()
        self.accelerator.print(f"epoch {self.epoch}:", eval_metric)

        log_dict = {
            "accuracy": eval_metric["accuracy"],
            "train_loss": train_loss,
            "epoch": self.epoch,
        }
        self.accelerator.log(
            log_dict,
            step=self.completed_steps,
        )
        self.train_log.append(log_dict)

        return eval_metric

    def _train_single_epoch(self):
        total_loss = 0.0

        self.model.train()
        for step, batch in enumerate(self.train_dataloader, start=1):
            loss = self.model(**batch).loss
            loss = loss / self.all_training_argument.gradient_accumulation_steps
            total_loss += loss.detach().float()
            self.accelerator.backward(loss)
            if step % self.all_training_argument.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                self.progress_bar.update(1)
                self.completed_steps += 1

        train_loss_epoch = total_loss.item() / len(self.train_dataloader)
        eval_metric = self._evaluate_epoch(train_loss_epoch)

        return eval_metric["accuracy"]

    def _log_train_start(self, max_training_steps, total_batch_size):
        logger.info("***** Running training *****")
        logger.info(f"  Target dataset name = {self.dataset.name}")
        logger.info(f"  Model name = {self.model.name_or_path}")
        logger.info(f"  Num examples = {len(self.dataset.train_loader)}")
        logger.info(f"  Num Epochs = {self.all_training_argument.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.all_training_argument.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.all_training_argument.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_training_steps}")

    def _read_training_records(self, file_name: str) -> pd.DataFrame:
        file = os.path.join(file_name)

        if not os.path.exists(file):
            all_column = ["model", "finetuned_dataset"] + list(vars(self.args).keys()) + ["train_runtime", "eval_accuracy"]
            return pd.DataFrame(columns=all_column)
        else:
            return pd.read_csv(file, index_col=0)

    def _save_result_to_csv(self, time_start) -> None:
        file_name = os.path.join(get_root_path_string(), 'resources/experiments', self.task_type.value, 'records.csv')
        result_record = self._read_training_records(file_name)

        training_record = vars(self.args)
        training_record["task_type"] = self.task_type.value
        training_record["model"] = self.model.name_or_path
        training_record['finetuned_dataset'] = self.dataset.name
        training_record["train_runtime"] = time.time() - time_start
        training_record["eval_accuracy"] = self.eval_accuracy

        result_record = pd.concat([result_record, pd.DataFrame(training_record, index=[0])], ignore_index=True)
        result_record.to_csv(file_name)

    def _save_model(self):
        if self.all_training_argument.output_dir is not None:
            model_name_sanitized = self.model.name_or_path.replace('/', '_')
            dataset_name_sanitized = self.dataset.name.replace('/', '_')

            if self.args.peft_method is not None:
                model_name = f"{model_name_sanitized}-finetuned-{self.args.peft_method}-{dataset_name_sanitized}"
            else:
                model_name = f"{model_name_sanitized}-finetuned-{dataset_name_sanitized}"
            model_directory = os.path.join(self.all_training_argument.output_dir, model_name)
            os.makedirs(model_directory, exist_ok=True)

            if self.accelerator.is_main_process:
                if self.all_training_argument.push_to_hub:
                    if self.all_training_argument.push_to_hub_organization:
                        repo_name = f"{self.all_training_argument.push_to_hub_organization}/{model_name}"
                    else:
                        repo_name = model_name
                    # Create repo and retrieve repo_id
                    repo_id = create_repo(repo_name, exist_ok=True, private=self.all_training_argument.hub_private_repo).repo_id

            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save_pretrained(
                model_directory,
                is_main_process=self.accelerator.is_main_process,
                save_function=self.accelerator.save
            )

            if self.accelerator.is_main_process:
                if isinstance(self.dataset, HuggingFaceDatasetText):
                    self.dataset.tokenizer.save_pretrained(model_directory)

                if self.all_training_argument.push_to_hub:
                    self.accelerator.print(f"Pushing model to {repo_id}")
                    self._create_model_card(model_name, model_directory)

                    api = HfApi()
                    api.upload_folder(
                        folder_path=model_directory,
                        repo_id=repo_id,
                        commit_message="Finished training.",
                    )

    def _create_model_card(self, model_name: str, model_directory: str):
        model_card_filepath = os.path.join(model_directory, "README.md")
        is_peft_library = False

        if os.path.exists(model_card_filepath):
            library_name = ModelCard.load(model_card_filepath).data.get("library_name")
            is_peft_library = library_name == "peft"

        training_summary = self._create_training_summary(model_name)
        model_card = training_summary.to_model_card()
        with open(model_card_filepath, "w") as f:
            f.write(model_card)

        if is_peft_library:
            unwrap_model(self.model).create_or_update_model_card(model_directory)

    def _create_training_summary(self, model_name: str):
        one_dataset = self.dataset.eval_loader.dataset if self.dataset.train_loader.dataset is not None else self.dataset.train_loader.dataset
        tags = []
        tags.append(one_dataset.builder_name)
        dataset_metadata = [{"config": one_dataset.config_name, "split": str(one_dataset.split)}]
        dataset_args = [one_dataset.config_name]

        finetuned_from = self.model.name_or_path
        tasks = self.task_type.to_huggingface_task_type()
        tags.append(tasks)

        hyperparameters = self._extract_hyperparameters()

        return TrainingSummary(
            model_name=model_name,
            finetuned_from=finetuned_from,
            tasks=tasks,
            tags=tags,
            dataset=self.args.dataset_path,
            dataset_args=dataset_args,
            dataset_tags=[self.args.dataset_path],
            dataset_metadata=dataset_metadata,
            hyperparameters=hyperparameters,
            eval_lines=self.train_log,
            eval_results={"accuracy": self.eval_accuracy}
        )

    def _extract_hyperparameters(self):
        hyperparameters = {k: getattr(self.all_training_argument, k) for k in
                           ["learning_rate", "train_batch_size", "eval_batch_size", "seed"]}

        if self.all_training_argument.parallel_mode not in [ParallelMode.NOT_PARALLEL, ParallelMode.NOT_DISTRIBUTED]:
            hyperparameters["distributed_type"] = (
                "multi-GPU" if self.all_training_argument.parallel_mode == ParallelMode.DISTRIBUTED else self.all_training_argument.parallel_mode.value
            )
        if self.all_training_argument.world_size > 1:
            hyperparameters["num_devices"] = self.all_training_argument.world_size
        if self.all_training_argument.gradient_accumulation_steps > 1:
            hyperparameters["gradient_accumulation_steps"] = self.all_training_argument.gradient_accumulation_steps

        total_train_batch_size = (
                self.all_training_argument.train_batch_size * self.all_training_argument.world_size * self.all_training_argument.gradient_accumulation_steps
        )
        if total_train_batch_size != hyperparameters["train_batch_size"]:
            hyperparameters["total_train_batch_size"] = total_train_batch_size
        total_eval_batch_size = self.all_training_argument.eval_batch_size * self.all_training_argument.world_size
        if total_eval_batch_size != hyperparameters["eval_batch_size"]:
            hyperparameters["total_eval_batch_size"] = total_eval_batch_size

        if self.all_training_argument.adafactor:
            hyperparameters["optimizer"] = "Adafactor"
        else:
            hyperparameters["optimizer"] = (
                f"Adam with betas=({self.all_training_argument.adam_beta1},{self.all_training_argument.adam_beta2}) and"
                f" epsilon={self.all_training_argument.adam_epsilon}"
            )

        hyperparameters["lr_scheduler_type"] = self.all_training_argument.lr_scheduler_type.value
        if self.all_training_argument.warmup_ratio != 0.0:
            hyperparameters["lr_scheduler_warmup_ratio"] = self.all_training_argument.warmup_ratio
        if self.all_training_argument.warmup_steps != 0.0:
            hyperparameters["lr_scheduler_warmup_steps"] = self.all_training_argument.warmup_steps
        if self.all_training_argument.max_steps != -1:
            hyperparameters["training_steps"] = self.all_training_argument.max_steps
        else:
            hyperparameters["num_epochs"] = self.all_training_argument.num_train_epochs

        if self.all_training_argument.fp16:
            hyperparameters["mixed_precision_training"] = "Native AMP"

        if self.all_training_argument.label_smoothing_factor != 0.0:
            hyperparameters["label_smoothing_factor"] = self.all_training_argument.label_smoothing_factor

        return hyperparameters
