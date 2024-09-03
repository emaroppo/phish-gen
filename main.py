from data.processing.labellers.NERMessageLabeller import NERMessageLabeller
from finetuning.sft.classes.Experiment import Experiment

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

    related_tokens_dict = {
        "<URL>": ["<URL>"],
        "<ATTACHMENT>": ["<ATTACHMENT>"],
        "<PHONE>": ["<PHONE>"],
        "<DATE>": ["<DATE>"],
        "<EMAIL>": ["<EMAIL>"],
        "<PER>": ["<PER>"],
        "<ORG>": ["<ORG>"],
    }

    experiment = Experiment.run_experiment(
        dataset_timestamp=1722274538,
        base_model_id="google/gemma-2b",
        quantization="4bit",
        epochs=2,
        custom_tokens=[
            "<URL>",
            "<ATTACHMENT>",
            "<PHONE>",
            "<DATE>",
            "<EMAIL>",
            "<PER>",
            "<ORG>",
        ],
        related_tokens_dict=related_tokens_dict,
    )

    experiment.evaluate_model_outputs()
