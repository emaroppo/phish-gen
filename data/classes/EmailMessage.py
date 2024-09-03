from bson import ObjectId
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from data.QueryManager import query_manager
import re


class EmailMessageEntry(BaseModel):
    id: ObjectId = Field(alias="_id")
    is_main: bool
    headers: Optional[Dict[str, str]] = None
    body: str
    response: Optional[ObjectId] = None
    forwarded_by: Optional[ObjectId] = None
    entities: Optional[Dict[str, Dict[str, List]]] = None
    sentiment: Optional[List[Dict[str, Union[float, str]]]] = None
    topic: Optional[List[List[Union[str, float]]]] = None
    attachments_format: Optional[List[str]] = None
    disclaimer: Optional[str] = None
    is_html: Optional[bool] = False
    word_count: Optional[int] = None

    class Config:
        arbitrary_types_allowed = True


class EmailMessage(BaseModel):
    id: str
    is_main: bool
    headers: Optional[Dict[str, str]] = None
    body: str
    response: Optional[str] = None
    forwarded_by: Optional[str] = None
    entities: Optional[Dict[str, Dict[str, List]]] = None
    sentiment: Optional[List[Dict[str, Union[float, str]]]] = None
    topic: Optional[List[List[Union[str, float]]]] = None
    attachments_format: Optional[List[str]] = None
    disclaimer: Optional[str] = None
    is_html: Optional[bool] = False
    word_count: Optional[int] = None

    @classmethod
    def deserialize(cls, data: Dict[str, Any]):
        object_id_fields = ["_id", "response", "forwarded_by"]
        for field in object_id_fields:
            if field in data:
                data[field] = str(data[field])

        # update _id key to id
        if "_id" in data:
            data["id"] = data.pop("_id")

        return cls(**data)

    @classmethod
    def from_text(
        cls,
        text: str,
        is_main: bool = False,
        response: ObjectId = None,
        forwarded_by: ObjectId = None,
    ):
        # assuming text holds a string containing the email thread
        message_doc = dict()
        message_doc["id"] = str(ObjectId())
        message_doc["body"] = text
        message_doc["is_main"] = is_main

        if response is not None:
            message_doc["response"] = response
        if forwarded_by is not None:
            message_doc["forwarded_by"] = forwarded_by

        return cls(**message_doc)

    def add_entity(
        self,
        entity_type: str,
        entity_value: str,
        start: int,
        end: int,
        detection_method: str = "manual",
    ):
        if self.entities is None:
            self.entities = dict()
        if detection_method not in self.entities:
            self.entities[detection_method] = dict()
        if entity_type not in self.entities[detection_method]:
            self.entities[detection_method][entity_type] = list()

        self.entities[detection_method][entity_type].append((entity_value, start, end))

    def add_sentiment(self, sentiment: Dict[str, float]):
        self.sentiment = sentiment

    def add_topic(self, topic: List[List[Union[str, float]]]):
        self.topic = topic

    def to_db_entry(self) -> EmailMessageEntry:
        db_entry = {
            "_id": ObjectId(self.id),
            "is_main": self.is_main,
            "headers": self.headers,
            "body": self.body,
        }
        if self.entities is not None:
            db_entry["entities"] = self.entities

        if self.sentiment is not None:
            db_entry["sentiment"] = self.sentiment

        if self.disclaimer is not None:
            db_entry["disclaimer"] = self.disclaimer

        if self.topic is not None:
            db_entry["topic"] = self.topic

        if self.response is not None and self.response != "None":

            db_entry["response"] = ObjectId(self.response)

        if self.forwarded_by is not None and self.forwarded_by != "None":
            db_entry["forwarded_by"] = ObjectId(self.forwarded_by)

        if self.is_html:
            db_entry["is_html"] = self.is_html

        if self.word_count is not None:
            db_entry["word_count"] = self.word_count

        if self.attachments_format is not None:
            db_entry["attachments_format"] = self.attachments_format

        return EmailMessageEntry(**db_entry).model_dump(by_alias=True)

    def save(self, db_name: str, target_collection: str):
        entry = self.to_db_entry()
        query = {"messages._id": ObjectId(self.id)}
        query_manager.connection[db_name][target_collection].update_one(
            query, {"$set": {"messages.$": entry}}
        )
        return True

    def __str__(self):
        return f"EmailMessage({self.id}, {self.body})"
