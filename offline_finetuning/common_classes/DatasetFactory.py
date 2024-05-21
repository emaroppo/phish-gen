from pydantic import BaseModel
from typing import List, Dict
from offline_finetuning.common_classes.QueryManager import query_manager
from transformers import PreTrainedTokenizer
from datasets import Dataset
from typing import List
import json
from tqdm import tqdm


class DatasetFactory(BaseModel):
    databases: Dict[str, List[str]]  # key: db_name, value: list of collections

    def generate_torch_dataset(
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

    def generate_doccamo_dataset(self):

        messages = list()

        for db_name, collections in self.databases.items():
            for collection in collections:
                pipeline = [
                    {
                        "$match": {
                            "messages.body": {"$exists": True},
                            "messages.headers": {"$exists": True},
                        }
                    },
                    {"$project": {"message": {"$arrayElemAt": ["$messages", -1]}}},
                ]

                messages.extend(
                    list(
                        query_manager.connection[db_name][collection].aggregate(
                            pipeline
                        )
                    )
                )

        entries = list()
        for message in tqdm(messages):

            if "headers" not in message["message"]:
                continue

            if message["message"]["headers"] is None:  # temporary fix
                continue

            labels = list()
            entry_str = str()
            for key, value in message["message"]["headers"].items():
                new_field = f"{key}: {value}"

                labels.append(
                    [len(entry_str), len(entry_str) + len(new_field), "HEADER_FIELD"]
                )
                labels.append([len(entry_str), len(entry_str) + len(key), "HEADER_KEY"])
                labels.append(
                    [
                        len(entry_str) + len(new_field) - len(value),
                        len(entry_str) + len(new_field),
                        "HEADER_VALUE",
                    ]
                )

                entry_str += new_field + "\n"
            labels.append([0, len(entry_str), "HEADER"])
            entry_str += "\n" + message["message"]["body"]
            labels.append(
                [
                    len(entry_str) - len(message["message"]["body"]),
                    len(entry_str),
                    "BODY",
                ]
            )

            entry_str += "\nMessage ID: " + str(message["message"]["_id"])
            entries.append({"text": entry_str, "labels": labels})

        with open("doccano_dataset.jsonl", "w") as f:
            for item in entries:
                f.write(json.dumps(item) + "\n")
