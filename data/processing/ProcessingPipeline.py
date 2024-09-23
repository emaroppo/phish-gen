from pydantic import BaseModel
from typing import List, Dict, Union, Literal, Tuple
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
import copy


with open("data/regex/placeholders.json", "r") as f:
    placeholders = json.load(f)


with open("data/regex/entities_regex.json", "r") as f:
    named_entities = json.load(f)


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

    @staticmethod
    def check_html(thread_list: List[EmailThread]) -> List[EmailThread]:
        for thread in thread_list:
            for message in thread.messages:
                if re.search(r"<html>", message.body):
                    message.is_html = True
        return thread_list

    @staticmethod
    def get_word_count(thread_list: List[EmailThread]) -> List[EmailThread]:
        for thread in thread_list:
            for message in thread.messages:
                message.word_count = len(message.body.split())
        return thread_list

    @staticmethod
    def get_attachments_formats(thread_list: List[EmailThread]) -> List[EmailThread]:
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

    @staticmethod
    def extract_entities(thread_list: List[EmailThread], placeholder: str, regex: str):
        for thread in tqdm(thread_list, desc=f"Extracting {placeholder}"):
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

    @staticmethod
    def retrieve_all_entities(message: EmailMessage) -> EmailMessage:
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
        return message

    @staticmethod
    def remove_overlaps(
        labels: List[Tuple[str, str, Tuple[str, int, int]]]
    ) -> List[Tuple[str, str, Tuple[str, int, int]]]:

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
        new_entities = dict()
        for label in non_overlapping_labels:

            detection_method, entity_type, entity = label
            if detection_method not in new_entities:
                new_entities[detection_method] = dict()
            if entity_type not in new_entities[detection_method]:
                new_entities[detection_method][entity_type] = list()

            new_entities[detection_method][entity_type].append(entity)

        return new_entities

    @classmethod
    def remove_overlapping_labels(
        cls,
        thread_list: List[EmailThread],
        detection_method: Union[
            List[Literal["manual", "auto"]], Literal["all_entities"]
        ] = "all_entities",
    ) -> List[EmailThread]:

        for thread in thread_list:
            for message in thread.messages:
                if detection_method == "all_entities":
                    labels = cls.retrieve_all_entities(message)
                    new_entities = cls.remove_overlaps(labels)

                elif type(detection_method) == list:
                    for method in detection_method:
                        labels = list()

                        if method in message.entities:
                            for entity_type in message.entities[method]:
                                labels.extend(
                                    [
                                        (method, entity_type, label)
                                        for label in message.entities[method][
                                            entity_type
                                        ]
                                    ]
                                )
                    labels.sort(key=lambda x: (x[2][1], -len(x[2][0])))
                    new_entities = cls.remove_overlaps(labels)
                    message.entities.update(new_entities)

        return thread_list

    @staticmethod
    def predict_sentiment(
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

    @classmethod
    def predict_topic(
        cls,
        message_list: List[EmailMessage],
        topic_predictor: TopicMessageLabeller,
        save=True,
    ):
        message_bodies = copy.deepcopy(message_list)
        message_bodies = cls.insert_entity_placeholders(message_bodies)
        message_bodies = [message.body for message in message_bodies]

        for message, message_body in tqdm(zip(message_list, message_bodies)):
            message_label = topic_predictor.label_message(message_body)
            message.add_topic(message_label)

        return message_list

    @staticmethod
    def extract_named_entities(
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

    @staticmethod
    def insert_entity_placeholders(
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
            if message.entities is None:
                continue
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

    @staticmethod
    def messages_to_datasamples(messages: List[EmailMessage]) -> List[DataSample]:
        datasamples = list()
        for message in messages:
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
                        attachment_formats = message.attachments_format

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

    def entity_validation(
        classifier_id: str = "dslim/bert-base-NER",
        collections: list = [
            "single_messages",
        ],
        file_path: str = "data/processing/labellers/ner/validation/",
    ):
        ner_message_labeller = NERMessageLabeller(
            classifier_id=classifier_id, task="ner"
        )
        ner_message_labeller.generate_validation_excel(
            collections=collections, file_path=file_path
        )

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

        for k, v in tqdm(
            placeholders.items(), desc="Extracting URL, ATTACHMENT, PHONE, DATE, EMAIL"
        ):
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

        for k, v in tqdm(
            named_entities.items(), desc="Extracting known named entities"
        ):
            for regex in v:
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
                    k,
                    regex,
                )

                for thread in thread_list:
                    thread.save()

            # retrieve all message bodies and ids
            pipeline = [
                {"$project": {"messages": 1}},
                {"$unwind": "$messages"},
            ]

        message_list = query_manager.connection[self.db][collection].aggregate(pipeline)
        message_list = [
            EmailMessage.deserialize(message["messages"]) for message in message_list
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

        print("Labelling message sentiment")
        # sentiment analysis
        message_list = self.predict_sentiment(
            message_list=message_list, sentiment_predictor=sentiment_labeller
        )

        print("Labelling message topics")
        message_list = self.predict_topic(
            message_list=message_list,
            topic_predictor=TopicMessageLabeller(
                checkpoint_path="data/processing/labellers/topic_modelling/models/topic_model"
            ),
        )

        print("Extracting named entities")
        # extract entities
        message_list = self.extract_named_entities(
            message_list=message_list, entity_predictor=ner_labeller
        )

        for message in tqdm(message_list):
            message.save(db_name=self.db, target_collection=collection)

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
        thread_list = self.remove_overlapping_labels(
            thread_list, detection_method=["manual"]
        )
        for thread in tqdm(thread_list):
            thread.save()

        return True

    def run_pipeline(self, source_collections: List[str], target_collection: str):
        # self.run_features_pipeline(source_collections)
        for collection in source_collections:

            # filter samples
            # check headers exists and is not none

            match_dict = {
                "messages.headers": {"$exists": True, "$ne": None},
                "messages.is_html": {"$ne": True},
                "messages.word_count": {"$lte": 448, "$gte": 16},
            }

            pipeline = [
                {"$project": {"messages": 1}},
                {"$unwind": "$messages"},
                {"$match": match_dict},
            ]
            messages = query_manager.connection[self.db][collection].aggregate(pipeline)
            messages = [
                EmailMessage.deserialize(message["messages"]) for message in messages
            ]

            # remove messages containing attachments formats other than pdf, doc, docx, xls, xlsx, ppt, pptx
            messages = [
                message
                for message in messages
                if not message.attachments_format
                or all(
                    [
                        attachment
                        in ["pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx"]
                        for attachment in message.attachments_format
                    ]
                )
            ]

            # insert entity placeholders
            messages = self.insert_entity_placeholders(messages)

            messages = self.messages_to_datasamples(messages)

            messages = [message.serialise() for message in messages]

            query_manager.connection["datasets"][target_collection].insert_many(
                messages
            )

        return True

    def update_topics(self, collection: str):
        # filter samples
        match_dict = {
            "messages.headers": {"$exists": True},
            "messages.is_html": {"$ne": True},
            "messages.word_count": {"$lte": 448},
        }

        pipeline = [
            {"$project": {"messages": 1}},
            {"$unwind": "$messages"},
            {"$match": match_dict},
        ]
        messages = query_manager.connection[self.db][collection].aggregate(pipeline)
        messages = [
            EmailMessage.deserialize(message["messages"]) for message in messages
        ]

        message_bodies = self.insert_entity_placeholders(copy.deepcopy(messages))
        message_bodies = [message.body for message in message_bodies]
        topic_predictor = TopicMessageLabeller(
            checkpoint_path="data/processing/labellers/topic_modelling/models/topic_model"
        )

        for message, message_body in tqdm(zip(messages, message_bodies)):

            message_label = topic_predictor.label_message(message_body)
            message.add_topic(message_label)
            message.save(db_name=self.db, target_collection=collection)
        return True
