from pydantic import BaseModel
from typing import List, Dict
from common_classes.QueryManager import query_manager
from transformers import PreTrainedTokenizer
from datasets import Dataset
from typing import List


class DatasetFactory(BaseModel):
    databases: Dict[str, List[str]]  # key: db_name, value: list of collections

    def generate_dataset(
        self,
        tokenizer: PreTrainedTokenizer,
        prompt_fields: List[str] = [
            "headers",
        ],
        target_fields: List[str] = [
            "body",
        ],
        max_length: int = 512,
        save_path: str = None,
    ):

        dataset = list()

        projection = {
            "$project": {
                "_id": 0,  # Assuming you don't want to include the MongoDB ID in the results
                "prompt_fields": {
                    field: f"$messages.{field}" for field in prompt_fields
                },
                "target_fields": {
                    field: f"$messages.{field}" for field in target_fields
                },
            }
        }

        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )

        for db_name, collections in self.databases.items():
            for collection in collections:
                # retrieve data from the database
                dataset = query_manager.connection[db_name][collection].aggregate(
                    [projection]
                )
                dataset = [str(doc) for doc in dataset]
                dataset.extend(dataset)

        dataset = Dataset.from_dict({"text": dataset})
        if save_path:
            dataset.save_to_disk(save_path)
        dataset = dataset.map(tokenize_function, batched=True)
        return dataset

    def generate_dataset_for_labelling(
        self, tokenizer: PreTrainedTokenizer, max_length: int = 512
    ):
        dataset = list()
        for db_name, collections in self.databases.items():
            for collection in collections:
                threads = query_manager.connection[db_name][collection].find()
                for thread in threads:
                    for message in thread["messages"]:
                        tokenized_message = tokenizer(
                            message["body"],
                            padding="max_length",
                            truncation=True,
                            max_length=max_length,
                        )
                        dataset.append(tokenized_message)
        return dataset
