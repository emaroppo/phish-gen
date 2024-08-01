from pydantic import BaseModel
from typing import List, Dict, Union
from offline_finetuning.common_classes.QueryManager import query_manager
from offline_finetuning.data_processing.enron.regex.placeholder_dict import (
    placeholder_dict,
)
from transformers import PreTrainedTokenizer
from datasets import Dataset
from typing import List
import json
from tqdm import tqdm
from prompt_generation.generate_prompt import generate_prompt_output_pair
from datetime import datetime
from presentation.classes.FinetuningDataset import FinetuningDataset


def get_sentiment(
    sentiment: List[Dict[str, Union[str, int, float]]], max_sentiment: int = 2
) -> List[str]:
    extracted_sentiment = [
        i["label"] for i in sentiment if i["score"] > 1 / (max_sentiment + 1)
    ]

    return extracted_sentiment


def insert_entity_placeholders(
    thread,
    entity_types: List[str] = [
        "URL",
        "ATTACHMENT",
        "PHONE",
        "DATE",
        "PER",
        "ORG",
        "EMAIL",
    ],
    manual=True,
    auto=False,
):
    entity_labels = list()
    for entity_type in entity_types:
        if manual and entity_type in thread["entities"]["manual"]:
            entity_labels.extend(
                [
                    (label, entity_type)
                    for label in thread["entities"]["manual"][entity_type]
                ]
            )
        if auto and entity_type in thread["entities"]["auto"]:
            entity_labels.extend(
                [
                    (label, entity_type)
                    for label in thread["entities"]["auto"][entity_type]
                ]
            )

    entity_labels.sort(key=lambda x: x[0][2], reverse=True)
    body = thread["body"]
    for label in entity_labels:
        body = body[: label[0][1]] + f"<{label[1]}>" + body[label[0][2] :]
    return body


class DatasetFactory(BaseModel):
    databases: Dict[str, List[str]]  # key: db_name, value: list of collections

    def generate_torch_dataset(
        self,
        save_path: str = None,
    ):

        dataset = list()

        finetuning_dataset = FinetuningDataset(
            timestamp=int(datetime.now().timestamp()),
            custom_tokens=[
                "<URL>",
                "<ATTACHMENT>",
                "<PHONE>",
                "<DATE>",
                "<EMAIL>",
                "<PER>",
                "<ORG>",
            ],
        )

        projection = {"$project": {"message": {"$arrayElemAt": ["$messages", -1]}}}

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
                    attachments = False
                    urls = False

                    # realised there were some missing implementations in the handling of forwarded messages, skipping them for now
                    if (
                        thread["is_html"]
                        or thread["word_count"] < 7
                        or thread["word_count"] > 384
                        or thread["forwarded_by"] == True
                    ):
                        continue
                    if "headers" not in thread:
                        continue
                    if thread["headers"] is None:
                        continue

                    if "entities" in thread:
                        if (
                            thread["entities"] is not None
                            and "manual" in thread["entities"]
                        ):
                            if "ATTACHMENT" in thread["entities"]["manual"]:
                                # skip the thread if the attachments_formats contains anything other than pdf, doc, docx, xls, xlsx, ppt, pptx
                                if thread["attachments_format"] and not all(
                                    [
                                        attachment
                                        in [
                                            "pdf",
                                            "doc",
                                            "docx",
                                            "xls",
                                            "xlsx",
                                            "ppt",
                                            "pptx",
                                        ]
                                        for attachment in thread["attachments_format"]
                                    ]
                                ):
                                    continue
                                if (
                                    thread["entities"]["manual"]["ATTACHMENT"]
                                    is not None
                                ):
                                    attachments = True
                            if "URL" in thread["entities"]["manual"]:
                                if thread["entities"]["manual"]["URL"] is not None:
                                    urls = True

                            thread["body"] = insert_entity_placeholders(thread)

                    sentiment = get_sentiment(thread["sentiment"])

                    finetuning_dataset.add_message(
                        urls=urls,
                        attachments=attachments,
                        body=thread["body"],
                        subject=thread["headers"]["Subject"],
                        sentiment=sentiment,
                        attachments_formats=thread["attachments_format"],
                    )
                    dataset.append(
                        generate_prompt_output_pair(
                            body=thread["body"],
                            subject=thread["headers"]["Subject"],
                            attachments=attachments,
                            urls=urls,
                            sentiment=sentiment,
                        )
                    )

        # save dataset to db
        query_manager.connection["datasets"]["summary"].insert_one(
            finetuning_dataset.serialise(include_samples=False)
        )
        query_manager.connection["datasets"][
            f"samples_{finetuning_dataset.timestamp}"
        ].insert_many([message.serialise() for message in finetuning_dataset.messages])

        dataset = Dataset.from_dict({"text": dataset})
        # shuffle the dataset
        dataset = dataset.shuffle()
        if save_path:
            dataset.save_to_disk(save_path)
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

    def generate_doccamo_dataset(
        self,
        placeholder_dict: dict = placeholder_dict,
        output_path: str = "offline_finetuning/datasets/doccano/enron_dataset.jsonl",
    ):

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

        with open(output_path, "w+") as f:
            for item in entries:
                f.write(json.dumps(item) + "\n")
