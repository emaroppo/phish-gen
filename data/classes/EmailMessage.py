from bson import ObjectId
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union, Literal
from data.QueryManager import query_manager


class EmailMessageEntry(BaseModel):
    id: ObjectId = Field(alias="_id")
    is_main: bool
    entity_validation_status: Optional[bool] = False
    headers: Optional[Dict[str, str]] = None
    body: str
    response: Optional[ObjectId] = None
    forwarded_by: Optional[ObjectId] = None
    entities: Optional[Dict[str, Dict[str, List]]] = None
    sentiment: Optional[List[Dict[str, Union[float, str]]]] = None
    topic: Optional[List[str]] = None
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
    entity_validation_status: Optional[bool] = False
    response: Optional[str] = None
    forwarded_by: Optional[str] = None
    entities: Optional[Dict[str, Dict[str, List]]] = None
    sentiment: Optional[List[Dict[str, Union[float, str]]]] = None
    topic: Optional[List[str]] = None
    attachments_format: Optional[List[str]] = None
    disclaimer: Optional[str] = None
    is_html: Optional[bool] = False
    word_count: Optional[int] = None

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "EmailMessage":
        object_id_fields = ["_id", "response", "forwarded_by"]
        for field in object_id_fields:
            if field in data:
                data[field] = str(data[field])

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
        message_doc = dict()
        message_doc["id"] = str(ObjectId())
        message_doc["body"] = text
        message_doc["is_main"] = is_main

        if response is not None:
            message_doc["response"] = response
        if forwarded_by is not None:
            message_doc["forwarded_by"] = forwarded_by

        return cls(**message_doc)

    @classmethod
    def from_message_id(
        cls, message_id: str, collection: str, db: str = "enron_datasource1"
    ) -> "EmailMessage":

        print("message_id", message_id)
        pipeline = [
            {"$unwind": "$messages"},
            {"$match": {"messages._id": ObjectId(message_id)}},
            {"$project": {"messages": 1}},
        ]
        message = query_manager.connection[db][collection].aggregate(pipeline)
        message = list(message)[0]["messages"]

        return cls.deserialize(message)

    def add_entity(
        self,
        entity_type: str,
        entity_value: str,
        start: int,
        end: int,
        detection_method: Literal["auto", "manual"] = "manual",
    ):
        if self.entities is None:
            self.entities = dict()
        if detection_method not in self.entities:
            self.entities[detection_method] = dict()
        if entity_type not in self.entities[detection_method]:
            self.entities[detection_method][entity_type] = list()

        self.entities[detection_method][entity_type].append((entity_value, start, end))

    def add_sentiment(self, sentiment: Dict[str, float]) -> None:
        self.sentiment = sentiment

    def add_topic(self, topic: List[str]) -> None:
        self.topic = topic

    def serialise(self) -> EmailMessageEntry:
        db_entry = {
            "_id": ObjectId(self.id),
            "is_main": self.is_main,
            "headers": self.headers,
            "body": self.body,
            "entity_validation_status": self.entity_validation_status,
        }

        optional_fields = {
            "entities": self.entities,
            "sentiment": self.sentiment,
            "disclaimer": self.disclaimer,
            "topic": self.topic,
            "response": (
                ObjectId(self.response)
                if self.response and self.response != "None"
                else None
            ),
            "forwarded_by": (
                ObjectId(self.forwarded_by)
                if self.forwarded_by and self.forwarded_by != "None"
                else None
            ),
            "is_html": self.is_html if self.is_html else None,
            "word_count": self.word_count,
            "attachments_format": self.attachments_format,
        }

        db_entry.update({k: v for k, v in optional_fields.items() if v is not None})

        return EmailMessageEntry(**db_entry).model_dump(by_alias=True)

    def save(self, db_name: str, target_collection: str):
        entry = self.serialise()
        query = {"messages._id": ObjectId(self.id)}
        query_manager.connection[db_name][target_collection].update_one(
            query, {"$set": {"messages.$": entry}}
        )
        return True

    def __str__(self):
        return f"EmailMessage({self.id}, {self.body})"
