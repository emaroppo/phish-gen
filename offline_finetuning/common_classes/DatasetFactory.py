from pydantic import BaseModel
from typing import List, Dict
from offline_finetuning.common_classes.QueryManager import query_manager
from offline_finetuning.data_processing.enron.regex.placeholder_dict import (
    placeholder_dict,
)
from transformers import PreTrainedTokenizer
from datasets import Dataset
from typing import List
import json
from tqdm import tqdm


class DatasetFactory(BaseModel):
    databases: Dict[str, List[str]]  # key: db_name, value: list of collections

    def doc_to_sample(self, doc: dict):
        prompt = dict()
        return_value = dict()
        prompt["SUBJECT"] = doc["headers"]["Subject"]
        if "special_tokens" in doc:
            if doc["special_tokens"]:
                if "urls" in doc["special_tokens"]:
                    if doc["special_tokens"]["urls"]:
                        prompt["URLS"] = True
                if "attachments" in doc["special_tokens"]:
                    if doc["special_tokens"]["attachments"]:
                        prompt["ATTACHMENTS"] = True

        return_value["BODY"] = doc["body"]
        if "special_tokens" in doc:
            if doc["special_tokens"]:
                if "attachment" in doc["special_tokens"]:
                    return_value["ATTACHMENTS"] = doc["special_tokens"]["attachment"]
                if "url" in doc["special_tokens"]:
                    return_value["URLS"] = doc["special_tokens"]["url"]

        return prompt, return_value

    def generate_torch_dataset(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        save_path: str = None,
    ):
        """
        Target Prompt:
        {
        "Subject": str,
        "Attachments": bool,
        "URLs": bool,
        }
        Target Output:
        {
        "body": str,
        "attachment": Optional[str],
        "url": Optional[str],
        }
        """

        dataset = list()

        projection = {"$project": {"message": {"$arrayElemAt": ["$messages", -1]}}}

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
                threads = [
                    doc["message"]
                    for doc in query_manager.connection[db_name][collection].aggregate(
                        [projection]
                    )
                ]
                for thread in threads:
                    if (
                        thread["is_html"]
                        or thread["word_count"] < 7
                        or thread["word_count"] > 384
                    ):
                        continue
                    if "headers" not in thread:
                        continue
                    if thread["headers"] is None:
                        continue
                    prompt, return_value = self.doc_to_sample(thread)
                    dataset.append(
                        '{"PROMPT": '
                        + str(prompt)
                        + ', "RETURN": '
                        + str(return_value)
                        + "}"
                    )

        print(len(dataset))

        dataset = Dataset.from_dict({"text": dataset})
        # shuffle the dataset
        dataset = dataset.shuffle()
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

    def generate_doccamo_dataset(self, placeholder_dict: dict = placeholder_dict):

        messages = list()

        for db_name, collections in self.databases.items():
            for collection in collections:
                print(collection)
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

            labels = list()
            entry_str = str()

            # add headers labels
            if "headers" not in message["message"]:
                continue

            if message["message"]["headers"] is None:  # temporary fix
                continue

            for key, value in message["message"]["headers"].items():
                key = key.capitalize()
                if key == "Sent":
                    key = "Date"

                if key in ["Date", "From", "To", "Subject", "Cc", "Bcc"]:
                    new_field = f"{key}: {value}"

                    labels.append(
                        [
                            len(entry_str),
                            len(entry_str) + len(new_field),
                            "HEADER_FIELD",
                        ]
                    )
                    """
                    labels.append(
                        [len(entry_str), len(entry_str) + len(key), "HEADER_KEY"]
                    )

                    labels.append(
                        [
                            len(entry_str) + len(new_field) - len(value),
                            len(entry_str) + len(new_field),
                            "HEADER_VALUE",
                        ]
                    )
                    """

                    entry_str += new_field + "\n"
            """
            labels.append([0, len(entry_str), "HEADER"])
            """
            entry_str += "\n" + message["message"]["body"]

            # add placeholder labels
            if "special_tokens" in message["message"]:
                if message["message"]["special_tokens"] is not None:
                    for key, values in message["message"]["special_tokens"].items():
                        for value in values:
                            # find the position of the placeholder in the entry string
                            start = entry_str.find(placeholder_dict[key]["placeholder"])
                            # replace the placeholder with the original value
                            entry_str = entry_str.replace(
                                placeholder_dict[key]["placeholder"], value, 1
                            )
                            end = start + len(value)
                            # add the label to the entry string
                            labels.append([start, end, key.upper()])

                    # add the body label
                    # calculate the length of the body with the original values instead of placeholders
                    body_length = len(message["message"]["body"])
                    for key, values in message["message"]["special_tokens"].items():
                        for value in values:
                            body_length -= len(placeholder_dict[key]["placeholder"])
                            body_length += len(value)
                else:
                    body_length = len(message["message"]["body"])
            else:
                body_length = len(message["message"]["body"])

            labels.append(
                [
                    len(entry_str) - body_length,
                    len(entry_str),
                    "BODY",
                ]
            )

            entry_str += "\nMessage ID: " + str(message["message"]["_id"])
            entries.append({"text": entry_str, "labels": labels})

        with open("doccano_dataset_1.jsonl", "w") as f:
            for item in entries:
                f.write(json.dumps(item) + "\n")
