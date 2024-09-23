from pydantic import BaseModel
from typing import Dict, List
from typing import Optional, List
from data.classes.DataSample import DataSample
from data.QueryManager import query_manager
from datasets import load_from_disk
from data.cleaning.CleaningPipeline import CleaningPipeline
from data.processing.ProcessingPipeline import ProcessingPipeline
import time


class FinetuningDataset(BaseModel):
    timestamp: int
    messages: Optional[List[DataSample]] = list()
    message_count: Optional[int] = 0
    attachment_formats: Optional[Dict[str, int]] = dict()
    sentiments: Optional[Dict[str, int]] = dict()
    entities: Optional[Dict[str, List[str]]] = dict()
    entity_counts: Optional[Dict[str, int]] = dict()
    custom_tokens: List[str] = [
        "<URL>",
        "<ATTACHMENT>",
        "<PHONE>",
        "<DATE>",
        "<EMAIL>",
        "<PER>",
        "<ORG>",
    ]

    @classmethod
    def deserialize(cls, data):
        return cls(
            timestamp=data["timestamp"],
            messages=[DataSample.deserialize(message) for message in data["messages"]],
            message_count=data["message_count"],
            attachment_formats=data["attachment_formats"],
            sentiments=data["sentiments"],
            entities=data["entities"],
            entity_counts=data["entity_counts"],
            custom_tokens=data["custom_tokens"],
        )

    @classmethod
    def from_db(cls, timestamp: int, include_messages=True):
        data = query_manager.connection["datasets"]["summary"].find_one(
            {"timestamp": timestamp}
        )
        if include_messages:
            messages = query_manager.connection["datasets"][
                f"samples_{timestamp}"
            ].find()
        else:
            messages = []
        data["messages"] = messages
        return cls.deserialize(data)

    @classmethod
    def from_file_list(cls, file_list):
        dataset_timestamp = int(time.time())
        dataset = cls(timestamp=dataset_timestamp)

        cleaning_pipeline = CleaningPipeline()
        processing_pipeline = ProcessingPipeline()

        cleaning_pipeline.run_pipeline(path_list=file_list)

        processing_pipeline.run_features_pipeline(collections=["single_messages"])
        processing_pipeline.run_pipeline(
            source_collections=["single_messages"],
            target_collection=f"samples_{dataset_timestamp}",
        )

        # retrieve messages from the database
        messages = query_manager.connection["datasets"][
            f"samples_{dataset_timestamp}"
        ].find()

        messages = [DataSample.deserialize(message) for message in messages]

        dataset = cls(timestamp=dataset_timestamp)
        for message in messages:
            dataset.add_message(message)

        query_manager.connection["datasets"]["summary"].insert_one(dataset.serialise())

        return dataset

    def add_message(self, message: DataSample):

        message.entity_counts = self.custom_tokens

        self.messages.append(message)
        self.message_count += 1
        if message.attachment_formats:
            for attachment_format in message.attachment_formats:
                if attachment_format in self.attachment_formats:
                    self.attachment_formats[attachment_format] += 1
                else:
                    self.attachment_formats[attachment_format] = 1

        for sentiment in message.sentiment:
            if sentiment in self.sentiments:
                self.sentiments[sentiment] += 1
            else:
                self.sentiments[sentiment] = 1

        if message.entity_counts:
            for entity, count in message.entity_counts.items():
                if entity in self.entity_counts:
                    self.entity_counts[entity] += count
                else:
                    self.entity_counts[entity] = count

    def load_finetuning_dataset(self, tokenizer):
        dataset = load_from_disk(f"data/datasets_processed/training/{self.timestamp}")

        dataset = dataset.map(
            lambda samples: tokenizer(
                samples["text"],
                padding="max_length",
                truncation=True,
                max_length=512,
                add_special_tokens=True,
            ),
        ).shuffle()

        # print first sample tokenized
        print(dataset[0])

        return dataset

    def serialise(self, include_samples=False):
        serialised = {
            "timestamp": self.timestamp,
            "message_count": self.message_count,
            "attachment_formats": self.attachment_formats,
            "sentiments": self.sentiments,
            "entities": self.entities,
            "entity_counts": self.entity_counts,
            "custom_tokens": self.custom_tokens,
        }

        if include_samples:
            serialised["messages"] = [message.serialise() for message in self.messages]

        return serialised
