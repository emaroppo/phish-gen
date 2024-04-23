from common_classes.DatasetFactory import DatasetFactory
from transformers import BertTokenizer

# Define the databases and collections to query
databases = {
    "enron_emails": ["raw_data_multipart"],
}


def main():
    # Load the dataset
    dataset_factory = DatasetFactory(databases=databases)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset = dataset_factory.generate_dataset(
        tokenizer=tokenizer, save_path="data/datasets/"
    )


main()
