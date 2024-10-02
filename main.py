from typing import List, Dict
from finetuning.sft.classes.Experiment import Experiment
from finetuning.sft.classes.FinetuningDataset import FinetuningDataset
from data.DatasetExporter import DatasetExporter
import json


def create_dataset(file_list: List[str]):
    dataset = FinetuningDataset.from_file_list(file_list)
    return dataset


def create_ner_validation_dataset():
    exporter = DatasetExporter()
    exporter.export_validation_dataset(
        headers=False, manual=True, auto=True, timestamp=1726055316
    )
    return True


def queue_experiments(arg_list: List[Dict], dataset_timestamp: int = 1727172426):
    experiments = list()
    for arg_dict in arg_list:
        experiment = Experiment.run_experiment(
            **arg_dict,
            dataset_timestamp=dataset_timestamp,
            base_model_id="google/gemma-2b",
        )
        experiments.append(experiment)
    return experiments


def resume_experiment(
    finetune_timestamp: int,
    dataset_timestamp: int,
    epochs: int,
    model_id: str = "google/gemma-2b",
):
    experiment = Experiment.from_db(
        model_id=model_id,
        model_timestamp=finetune_timestamp,
        dataset_timestamp=dataset_timestamp,
    )
    experiment.finetuned_model.resume_training(epochs=epochs)
    return experiment


if __name__ == "__main__":
    exp1 = {
        "custom_tokens": [
            "<URL>",
            "<ATTACHMENT>",
            "<PHONE>",
            "<DATE>",
            "<EMAIL>",
            "<PER>",
            "<ORG>",
        ],
        "gradient_accumulation_steps": 16,
        "warmup_steps": 500,
    }
    arg_list = [exp1]
    experiments = queue_experiments(arg_list)
