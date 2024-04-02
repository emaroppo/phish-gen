from bson import ObjectId
from pydantic import BaseModel
from typing import Optional, Dict, Any


class EmailMessage(BaseModel):
    id: str
    is_main: bool
    headers: Optional[Dict[str, str]] = None
    body: str
    response: Optional[str] = None
    forwarded_by: Optional[str] = None

    @classmethod
    def deserialize(cls, data: Dict[str, Any]):
        object_id_fields = ["_id", "response", "forwarded_by"]
        for field in object_id_fields:
            if field in data:
                data[field] = str(data[field])

        #update _id key to id
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

    def to_db_entry(self):
        db_entry = {
            "_id": ObjectId(self.id),
            "is_main": self.is_main,
            "headers": self.headers,
            "body": self.body,
        }
        if self.response is not None:
            db_entry["response"] = self.response
        if self.forwarded_by is not None:
            db_entry["forwarded_by"] = self.forwarded_by

        return db_entry

    def __str__(self):
        return f"EmailMessage({self.id}, {self.body})"
