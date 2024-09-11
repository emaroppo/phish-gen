from typing import List, Dict
from finetuning.sft.classes.Experiment import Experiment
from finetuning.sft.classes.FinetuningDataset import FinetuningDataset


def create_dataset(file_list: List[str]):
    dataset = FinetuningDataset.from_file_list(file_list)
    return dataset


def queue_experiments(arg_list: List[Dict], dataset_timestamp: int = 1722274538):
    experiments = list()
    for arg_dict in arg_list:
        experiment = Experiment.run_experiment(
            **arg_dict,
            dataset_timestamp=dataset_timestamp,
            base_model_id="google/gemma-2b",
        )
        experiments.append(experiment)
    return experiments


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
