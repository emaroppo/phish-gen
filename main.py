from offline_finetuning.data_processing.enron.EnronDataImporter import EnronDataImporter
from offline_finetuning.common_classes.DatasetFactory import DatasetFactory
from transformers import AutoTokenizer
from inference.MessageGenerator import MessageGenerator


def run_pipeline(
    db_name: str = "enron_emails_test4",
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
    databases: dict = {
        "enron_emails_test4": ["step2_single", "step3_placeholders", "step2_multipart"]
    },
    from_files: bool = False,
):
    if from_files:
        run_pipeline()
    dataset_factory = DatasetFactory(databases=databases)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    dataset = dataset_factory.generate_torch_dataset(
        tokenizer=tokenizer,
        max_length=512,
        save_path="offline_finetuning/datasets/pytorch/enron/",
    )

    return dataset


def generate_message(
    base_model_id: str = "google/gemma-2b",
    subject: str = "Budget Review Meeting",
    attachments: bool = False,
    urls: bool = True,
    guided: str = False,
):
    message_generator = MessageGenerator(
        base_model_id=base_model_id,
        quantized="4bit",
        adapter=f"offline_finetuning/models/gemma-2b/checkpoint-8000",
    )
    message_generator.generate_message(
        subject=subject, attachments=attachments, urls=urls, guided=guided
    )


generate_message()
