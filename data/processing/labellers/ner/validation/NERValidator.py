# read jsonl file
import json
from data.classes.EmailMessage import EmailMessage
from data.classes.EmailThread import EmailThread
from data.processing.ProcessingPipeline import ProcessingPipeline
from data.QueryManager import query_manager
from data.DatasetExporter import DatasetExporter
from typing import List
from pydantic import BaseModel
from tqdm import tqdm
import re


class NERValidator(BaseModel):
    db: str
    collections: List[str]
    validation_dataset: str

    def load_validated_messages(self):
        with open(self.validation_dataset, "r") as f:
            validated_entries = f.readlines()

        messages = list()
        for line in validated_entries:
            entry = json.loads(line)
            message_id = entry["text"].split("\n")[-1].split(":")[-1].strip()
            message = EmailMessage.from_message_id(message_id, "single_messages")
            for entity in entry["label"]:
                message.add_entity(
                    entity[2],
                    message.body[entity[0] : entity[1]],
                    entity[0],
                    entity[1],
                    "manual",
                )
            message.entity_validation_status = True
            message.save(self.db, "single_messages")

        return messages

    def extract_unique_validated_entities(self):
        with open(self.validation_dataset, "r") as f:
            data = f.readlines()

        unique_entities = dict()
        for line in data:
            entry = json.loads(line)
            for entity in entry["label"]:
                if entity[2] not in unique_entities:
                    unique_entities[entity[2]] = set()
                unique_entities[entity[2]].add(entry["text"][entity[0] : entity[1]])

        return unique_entities

    def convert_entities_to_regex(
        self, unique_entities, save_path, entity_types: list = ["PER", "ORG", "LOC"]
    ):
        if not entity_types:
            entity_types = unique_entities.keys()
        unique_entities_regex = dict()
        for entity_type in entity_types:
            entity_list = unique_entities[entity_type]
            regex_list = list()
            for entity in entity_list:
                regex = re.escape(entity)
                regex = rf"\b{regex.replace(" ", "\s+")}\b"
                regex_list.append(regex)
            unique_entities_regex[entity_type] = regex_list

        if save_path:
            with open(save_path, "w") as f:
                json.dump(unique_entities_regex, f)

        return unique_entities_regex

    def update_database_entries(self, unique_entities_regex):
        for entity_type in unique_entities_regex.keys():
            entity_list = unique_entities_regex[entity_type]
            for collection in self.collections:
                for entity in tqdm(entity_list):
                    if entity == "":
                        continue
                    if entity == "\n":
                        continue
                    threads = query_manager.connection[self.db][collection].find(
                        {"messages.body": {"$regex": entity}}
                    )

                    threads = [
                        EmailThread.deserialize(
                            thread, db_name=self.db, collection=collection
                        )
                        for thread in threads
                    ]

                    threads = ProcessingPipeline.extract_entities(
                        threads, entity_type, entity
                    )

                    threads = ProcessingPipeline.remove_overlapping_labels(
                        thread_list=threads, detection_method=["manual"]
                    )

                    for thread in threads:
                        thread.save()

        return True

    def update_validation(self):
        # load validated messages and update corresponding db entries
        self.load_validated_messages()

        # extract validated entities from other messages
        unique_entities = self.extract_unique_validated_entities()
        unique_entities_regex = self.convert_entities_to_regex(
            unique_entities, "data/regex/entities.json"
        )
        self.update_database_entries(unique_entities_regex)

        # create new validation dataset
        exporter = DatasetExporter()
        exporter.export_validation_dataset(
            headers=False, auto=True, manual=True, timestamp=1726055317
        )
        return True
