from pydantic import BaseModel
from typing import List
from data.QueryManager import query_manager
from data.classes.EmailMessage import EmailMessage
from data.classes.DataSample import DataSample
from inference.prompt_generation.generate_prompt import generate_prompt_output_pair
from datasets import Dataset
import os
import json


class DatasetExporter(BaseModel):
    databases: List[str] = ["enron_datasource"]

    def export_training_dataset(self, timestamp: int):
        samples = query_manager.connection["datasets"][f"samples_{timestamp}"].find()
        samples = [DataSample.deserialize(sample) for sample in samples]

        samples = [
            generate_prompt_output_pair(
                body=sample.body,
                subject=sample.subject,
                sentiment=sample.sentiment,
                attachments=sample.attachments,
                urls=sample.urls,
            )
            for sample in samples
        ]

        dataset = Dataset.from_dict({"text": samples})

        if not os.path.exists(f"data/datasets_processed/training/{timestamp}"):
            dataset.save_to_disk(f"data/datasets_processed/training/{timestamp}")

        return dataset

    def export_validation_dataset(
        self, headers: bool, manual: bool, auto: bool, timestamp: int
    ):
        dataset = []
        filters_dicts = []
        project_dict = {
            "_id": 0,
            "message_id": "$messages.message_id",
            "body": "$messages.body",
        }
        if headers:
            filters_dicts.append({"messages.headers": {"$exists": True}})
            project_dict["message_headers"] = "$messages.headers"

        if manual:
            filters_dicts.append({"messages.entities.manual": {"$exists": True}})
            project_dict["message_manual_entities"] = "$messages.entities.manual"

        if auto:
            filters_dicts.append({"messages.entities.auto": {"$exists": True}})
            project_dict["message_auto_entities"] = "$messages.entities.auto"

        match_dict = {"$and": filters_dicts}
        print(match_dict)

        # retrieve all messages matching the filters from all threads
        pipeline = [
            {"$unwind": "$messages"},
            {"$match": match_dict},
            {"$project": project_dict},
        ]

        data_samples = query_manager.connection["enron_datasource"][
            "single_messages"
        ].aggregate(pipeline)

        for message in data_samples:
            sample = {"text": message["body"], "labels": []}

            if headers:
                pass
            if manual:
                for entity_type in message["message_manual_entities"].keys():
                    for entity in message["message_manual_entities"][entity_type]:
                        sample["labels"].append((entity[1], entity[2], entity_type))
            if auto:
                for entity_type in message["message_auto_entities"].keys():
                    for entity in message["message_auto_entities"][entity_type]:
                        sample["labels"].append((entity[1], entity[2], entity_type))
            dataset.append(sample)

        with open(f"data/datasets_processed/validation/{timestamp}.jsonl", "w") as f:
            for sample in dataset:
                f.write(json.dumps(sample) + "\n")

        return dataset

    def export_labelling_dataset(
        self, collections: List[str] = ["single_messages", "multiparts", "forwarderd"]
    ):
        messages = list()
        for database in self.databases:
            for collection in collections:
                # retrieve all elements from messages array from all threads of the collection as a list of message ids and message bodies
                pipeline = [
                    {"$unwind": "$messages"},
                    {
                        "$project": {
                            "_id": 0,
                            "message_id": "$messages.message_id",
                            "message_body": "$messages.message_body",
                        }
                    },
                ]
                new_messages = query_manager.connection[database][collection].aggregate(
                    pipeline
                )
                new_messages = [
                    EmailMessage.deserialize(message) for message in messages
                ]
                messages.extend(new_messages)

        return messages
