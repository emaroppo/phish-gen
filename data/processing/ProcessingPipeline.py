from pydantic import BaseModel
from typing import List, Dict, Union
from data.classes.EmailThread import EmailThread
from data.classes.DataSample import DataSample
from data.classes.EmailMessage import EmailMessage
import re
import json
from tqdm import tqdm
from data.processing.labellers.NERMessageLabeller import NERMessageLabeller
from data.processing.labellers.SentimentMessageLabeller import (
    SentimentMessageLabeller,
)
from data.processing.labellers.TopicMessageLabeller import TopicMessageLabeller
from data.QueryManager import query_manager


with open("data/regex/placeholders.json", "r") as f:
    placeholders = json.load(f)


def message_data(messages: List[EmailMessage]):
    for message in messages:
        yield message.body


def get_sentiment(
    sentiment: List[Dict[str, Union[str, int, float]]], max_sentiment: int = 2
) -> List[str]:
    extracted_sentiment = [
        i["label"] for i in sentiment if i["score"] > 1 / (max_sentiment + 1)
    ]

    return extracted_sentiment


class ProcessingPipeline(BaseModel):
    db: str = "enron_datasource"

    def check_html(self, thread_list: List[EmailThread]) -> List[EmailThread]:
        for thread in thread_list:
            for message in thread.messages:
                if re.search(r"<html>", message.body):
                    message.is_html = True
        return thread_list

    def get_word_count(self, thread_list: List[EmailThread]) -> List[EmailThread]:
        for thread in thread_list:
            for message in thread.messages:
                message.word_count = len(message.body.split())
        return thread_list

    def get_attachments_formats(
        self, thread_list: List[EmailThread]
    ) -> List[EmailThread]:
        for thread in thread_list:
            for message in thread.messages:
                if message.entities is not None and "manual" in message.entities:
                    if "ATTACHMENT" in message.entities["manual"]:
                        attachments = [
                            entity[0]
                            for entity in message.entities["manual"]["ATTACHMENT"]
                        ]

                        # remove >> and/or ) from the attachment name
                        message.attachments_format = [
                            attachment.split(".")[-1] for attachment in attachments
                        ]
                        message.attachments_format = [
                            attachment.split(")")[0].lower()
                            for attachment in message.attachments_format
                        ]
                        message.attachments_format = [
                            attachment.split(">>")[0].lower()
                            for attachment in message.attachments_format
                        ]
        return thread_list

    def extract_entities(
        self, thread_list: List[EmailThread], placeholder: str, regex: str
    ):
        for thread in thread_list:
            for message in thread.messages:
                matches = re.finditer(regex, message.body)
                extracted_entities = list()

                for match in matches:
                    value = match.group()
                    label = placeholder
                    start = match.start()
                    end = match.end()
                    message.add_entity(
                        start=start,
                        end=end,
                        entity_value=value,
                        entity_type=label,
                        detection_method="manual",
                    )

        return thread_list

    def remove_overlapping_labels(self, thread_list: List[EmailThread]):
        for thread in thread_list:
            for message in thread.messages:
                labels = list()

                if message.entities is not None:
                    for detection_method in message.entities:
                        if message.entities[detection_method] is not None:
                            for entity_type in message.entities[detection_method]:
                                labels.extend(
                                    [
                                        (detection_method, entity_type, label)
                                        for label in message.entities[detection_method][
                                            entity_type
                                        ]
                                    ]
                                )

                labels.sort(key=lambda x: (x[2][1], -len(x[2][0])))

                # Initialize a list to hold non-overlapping labels
                non_overlapping_labels = []

                # Initialize the end index of the last added label to -1
                last_end_idx = -1

                for label in labels:
                    _, start_idx, end_idx = label[2]

                    # If the current label does not overlap with the last added label, add it to the list
                    if start_idx > last_end_idx:
                        non_overlapping_labels.append(label)
                        last_end_idx = end_idx
                # reconstruct the entities dictionary
                message.entities = dict()
                for label in non_overlapping_labels:

                    detection_method, entity_type, entity = label
                    if detection_method not in message.entities:
                        message.entities[detection_method] = dict()
                    if entity_type not in message.entities[detection_method]:
                        message.entities[detection_method][entity_type] = list()

                    message.entities[detection_method][entity_type].append(entity)

        return thread_list

    def predict_sentiment(
        self,
        message_list: List[EmailMessage],
        sentiment_predictor: SentimentMessageLabeller,
        batch_size: int = 128,
    ):

        sentiment = sentiment_predictor.label_message(
            message_data(message_list), batch_size=batch_size
        )
        for message, sentiment in zip(message_list, sentiment):
            message.add_sentiment(sentiment)
        return message_list

    def predict_topic(
        self,
        message_list: List[EmailMessage],
        topic_predictor: TopicMessageLabeller,
        save=True,
    ):
        topic = topic_predictor.label_message(message_data(message_list))
        for message, topic in zip(message_list, topic):
            message.add_topic(topic)
            message.save()

    def extract_named_entities(
        self,
        message_list: List[EmailMessage],
        entity_predictor: NERMessageLabeller,
        batch_size: int = 128,
    ) -> List[EmailMessage]:

        message_bodies = [message.body for message in message_list]
        extracted_entities = entity_predictor.label_message(
            message_bodies, batch_size=batch_size
        )
        for message, extracted_entities in zip(message_list, extracted_entities):
            if extracted_entities:
                for entity in extracted_entities:
                    start, end, label = entity
                    entity_value = message.body[start:end]
                    message.add_entity(
                        entity_type=label,
                        start=start,
                        end=end,
                        entity_value=entity_value,
                        detection_method="auto",
                    )

        return message_list

    def insert_entity_placeholders(
        self,
        message_list: List[EmailMessage],
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

        for message in message_list:
            entity_labels = list()
            for entity_type in entity_types:
                if manual and "manual" in message.entities:
                    if entity_type in message.entities["manual"]:
                        entity_labels.extend(
                            [
                                (label, entity_type)
                                for label in message.entities["manual"][entity_type]
                            ]
                        )

                if auto and entity_type in message.entities["auto"]:
                    entity_labels.extend(
                        [
                            (label, entity_type)
                            for label in message.entities["auto"][entity_type]
                        ]
                    )

            entity_labels.sort(key=lambda x: x[0][2], reverse=True)
            body = message.body
            for label in entity_labels:
                body = body[: label[0][1]] + f"<{label[1]}>" + body[label[0][2] :]
            message.body = body
        return message_list

    def messages_to_datasamples(self, messages: List[EmailMessage]) -> List[DataSample]:
        datasamples = list()
        for message in messages:
            if message.is_html or message.word_count > 1000 or message.headers is None:
                continue
            urls = False
            attachments = False
            sentiment = get_sentiment(message.sentiment)
            attachment_formats = None

            if message.entities is not None:
                if "manual" in message.entities:
                    if "URL" in message.entities["manual"]:
                        urls = True
                    if "ATTACHMENT" in message.entities["manual"]:
                        attachments = True
                        if message.attachments_format and not all(
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
                                for attachment in message.attachments_format
                            ]
                        ):
                            continue

            datasample = DataSample(
                body=message.body,
                attachments=attachments,
                subject=message.headers["Subject"],
                urls=urls,
                sentiment=sentiment,
                attachment_formats=attachment_formats,
            )
            datasamples.append(datasample)
        return datasamples

    def run_features_pipeline(self, collections: List[str]):

        for collection in collections:
            # find all threads
            data = query_manager.connection[self.db][collection].find()
            thread_list = [
                EmailThread.deserialize(thread, db_name=self.db, collection=collection)
                for thread in data
            ]
            thread_list = self.check_html(thread_list)
            thread_list = self.get_word_count(thread_list)

            for thread in thread_list:
                thread.save()

            # retrieve all message bodies and ids
            pipeline = [
                {"$project": {"messages": 1}},
                {"$unwind": "$messages"},
            ]

            message_list = query_manager.connection[self.db][collection].aggregate(
                pipeline
            )
            message_list = [
                EmailMessage.deserialize(message["messages"])
                for message in message_list
            ]

            sentiment_labeller = SentimentMessageLabeller(
                classifier_id="michellejieli/emotion_text_classifier",
                task="sentiment-analysis",
                no_top_k=True,
            )

            ner_labeller = NERMessageLabeller(
                classifier_id="dslim/bert-large-NER",
                task="ner",
            )

            # sentiment analysis
            message_list = self.predict_sentiment(
                message_list=message_list, sentiment_predictor=sentiment_labeller
            )

            # extract entities
            message_list = self.extract_named_entities(
                message_list=message_list, entity_predictor=ner_labeller
            )

            for message in tqdm(message_list):
                message.save(db_name=self.db, target_collection=collection)

            for k, v in tqdm(placeholders.items()):
                for regex in v["regex"]:
                    thread_list = query_manager.connection[self.db][collection].find(
                        {"messages.body": {"$regex": regex}}
                    )

                    thread_list = [
                        EmailThread.deserialize(
                            thread, db_name=self.db, collection=collection
                        )
                        for thread in thread_list
                    ]

                    thread_list = self.extract_entities(
                        thread_list,
                        v["placeholder"],
                        regex,
                    )

                    for thread in thread_list:
                        thread.save()

            # retrieve messages with attachments
            thread_list = query_manager.connection[self.db][collection].find(
                {"messages.entities.manual.ATTACHMENT": {"$exists": True}}
            )
            thread_list = [
                EmailThread.deserialize(thread, db_name=self.db, collection=collection)
                for thread in thread_list
            ]
            thread_list = self.get_attachments_formats(thread_list)
            for thread in thread_list:
                thread.save()

            # remove overlapping labels
            # retrieve all threads
            data = query_manager.connection[self.db][collection].find()
            thread_list = [
                EmailThread.deserialize(thread, db_name=self.db, collection=collection)
                for thread in data
            ]
            thread_list = self.remove_overlapping_labels(thread_list)
            for thread in tqdm(thread_list):
                thread.save()

        return True

    def run_pipeline(self, source_collections: List[str], target_collection: str):
        # self.run_features_pipeline(source_collections)
        for collection in source_collections:
            # retrieve all messages from the collection
            pipeline = [{"$project": {"messages": 1}}, {"$unwind": "$messages"}]
            messages = query_manager.connection[self.db][collection].aggregate(pipeline)
            messages = [
                EmailMessage.deserialize(message["messages"]) for message in messages
            ]

            print(messages[-1])

            # insert entity placeholders
            messages = self.insert_entity_placeholders(messages)

            messages = self.messages_to_datasamples(messages)

            messages = [message.serialise() for message in messages]

            query_manager.connection["datasets"][target_collection].insert_many(
                messages
            )

        return True
