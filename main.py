from data.processing.labellers.NERMessageLabeller import NERMessageLabeller
from finetuning.sft.classes.Experiment import Experiment
from typing import List, Dict


def queue_experiments(arg_list: List[Dict]):
    experiments = list()
    for arg_dict in arg_list:
        experiment = Experiment.run_experiment(
            **arg_dict,
            dataset_timestamp=1722274538,
            base_model_id="google/gemma-2b",
        )
        experiments.append(experiment)
    return experiments


def entity_validation(
    classifier_id: str = "dslim/bert-base-NER",
    collections: list = ["single_messages", "multipart_messages", "forwarded_messages"],
    file_path: str = "data/",
):
    ner_message_labeller = NERMessageLabeller(classifier_id=classifier_id, task="ner")
    ner_message_labeller.generate_validation_excel(
        collections=collections, file_path=file_path
    )


if __name__ == "__main__":

    exp1 = {
        "related_tokens_dict": {
            "<URL>": ["<URL>"],
            "<ATTACHMENT>": ["<ATTACHMENT>"],
            "<PHONE>": ["<PHONE>"],
            "<DATE>": ["<DATE>"],
            "<EMAIL>": ["<EMAIL>"],
            "<PER>": ["<PER>"],
            "<ORG>": ["<ORG>"],
        }
    }

    exp2 = {
        "related_tokens_dict": {
            "<URL>": ["<URL>", "http://", "url"],
            "<ATTACHMENT>": [
                "<ATTACHMENT>",
                "attachment",
                "file",
                "doc",
                "xls",
                "ppt",
                "pdf",
            ],
            "<PHONE>": ["<PHONE>", "phone", "number"],
            "<DATE>": ["<DATE>", "date", "time"],
            "<EMAIL>": ["<EMAIL>", "email", "mail"],
            "<PER>": ["<PER>", "person", "individual"],
            "<ORG>": ["<ORG>", "organization", "company"],
        }
    }

    exp3 = {
        "related_tokens_dict": {
            "<URL>": ["<URL>", "placeholder", "http://", "url"],
            "<ATTACHMENT>": [
                "<ATTACHMENT>",
                "placeholder",
                "attachment",
                "file",
                "doc",
                "xls",
                "ppt",
                "pdf",
            ],
            "<PHONE>": ["<PHONE>", "placeholder", "phone", "number"],
            "<DATE>": ["<DATE>", "placeholder", "date", "time"],
            "<EMAIL>": ["<EMAIL>", "placeholder", "email", "mail"],
            "<PER>": ["<PER>", "placeholder", "person", "individual"],
            "<ORG>": ["<ORG>", "placeholder", "organization", "company"],
        }
    }

    arg_list = [exp1, exp2, exp3]

    experiments = queue_experiments(arg_list)
