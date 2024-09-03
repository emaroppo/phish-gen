# from offline_finetuning.data_processing.enron.EnronDataImporter import EnronDataImporter
# from offline_finetuning.common_classes.DatasetFactory import DatasetFactory
from inference.MessageGenerator import MessageGenerator
from data.processing.labellers.TopicMessageLabeller import TopicMessageLabeller
from data.QueryManager import query_manager
from data.processing.labellers.NERMessageLabeller import NERMessageLabeller
from finetuning.sft.classes.Experiment import Experiment
from finetuning.sft.classes.FinetunedModel import FinetunedModel
from finetuning.sft.classes.FinetuningDataset import FinetuningDataset


def run_pipeline(
    db_name: str = "enron_emails",
    data_dir: str = "offline_finetuning/data_processing/enron/dataset/maildir",
):

    enron_data_importer = EnronDataImporter(
        db_name=db_name,
        data_dir=data_dir,
    )
    enron_data_importer.pipeline()


def generate_doccamo_dataset(
    databases: dict = {
        "enron_emails_test4": ["step2_single", "step3_placeholders", "step2_multipart"]
    }
):
    dataset_factory = DatasetFactory(databases=databases)
    dataset_factory.generate_doccamo_dataset()


def generate_pytorch_dataset(
    databases: dict = {"enron_emails": ["step2_single", "step2_multipart"]},
    from_files: bool = False,
):
    if from_files:
        run_pipeline(
            db_name="enron_emails",
            data_dir="offline_finetuning/data_processing/enron/dataset/maildir",
        )
    dataset_factory = DatasetFactory(databases=databases)

    dataset = dataset_factory.generate_torch_dataset(
        save_path="offline_finetuning/datasets/pytorch/enron/",
    )

    return dataset


def generate_message(
    base_model_id: str = "google/gemma-2b",
    subject: str = "Budget Review Meeting",
    attachments: bool = False,
    urls: bool = True,
    sentiment: list = ["neutral"],
    guided: str = False,
):
    message_generator = MessageGenerator(
        base_model_id=base_model_id,
        quantized="4bit",
        adapter=f"offline_finetuning/models/gemma-2b/checkpoint-8000",
    )
    message = message_generator.generate_message(
        subject=subject,
        attachments=attachments,
        urls=urls,
        sentiment=sentiment,
        guided=guided,
    )

    return message


# retrieve all message bodies
def retrieve_message_bodies(db_name: str, collections: list):
    bodies = list()
    for collection in collections:
        bodies.extend(
            [
                thread["message"]["body"]
                for thread in query_manager.connection[db_name][collection].aggregate(
                    [{"$project": {"message": {"$arrayElemAt": ["$messages", -1]}}}]
                )
            ]
        )
    return bodies


# retrieve a random message
# print topic
def retrieve_random_message(db_name: str, collections: list):
    message = query_manager.connection[db_name][collections[0]].find_one()
    topic_modelling = TopicMessageLabeller(
        checkpoint_path="offline_finetuning/auto_labelling/topic_modelling/models/topic_model",
    )
    print(topic_modelling.label_message(message["messages"][-1]["body"]))


# generate excels for entity validation


def entity_validation(
    classifier_id: str = "dslim/bert-base-NER",
    collections: list = ["step2_single", "step2_multipart", "step2_forwarded"],
    file_path: str = "offline_finetuning/auto_labelling/validation",
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
