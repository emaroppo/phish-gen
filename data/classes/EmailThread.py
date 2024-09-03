from data.QueryManager import query_manager
from bson import ObjectId
from data.classes.EmailMessage import EmailMessage
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel


class ThreadEntry(BaseModel):
    _id: ObjectId
    file_path: str
    messages: List[Dict[str, Any]]

    class Config:
        arbitrary_types_allowed = True


class EmailThread(BaseModel):
    id: Optional[str]
    messages: List[EmailMessage]
    file_path: str
    db_name: str
    collection: str

    @classmethod
    def deserialize(
        cls, data, db_name: str = None, collection: str = None
    ):  # may need to add db_name and collection fields to properly deserialize?
        object_id_fields = ["_id"]
        for field in object_id_fields:
            if field in data:
                data[field] = str(data[field])

        if "_id" in data:
            data["id"] = data.pop("_id")

        messages = [EmailMessage.deserialize(i) for i in data["messages"]]
        data["messages"] = messages
        data["db_name"] = db_name
        data["collection"] = collection

        return cls(**data)

    @classmethod
    def from_db(cls, _id, db_name, collection):
        # retrieve the thread from the database

        thread_doc = query_manager.retrieve_entry(
            {"_id": _id}, db_name=db_name, collection=collection
        )
        thread_doc["db_name"] = db_name
        thread_doc["collection"] = collection
        thread_doc["id"] = str(thread_doc.pop("_id"))
        thread_doc["id"] = str(thread_doc["id"])
        for i in thread_doc["messages"]:
            i["id"] = str(i["_id"])
            i["id"] = str(i["id"])
            del i["_id"]

        return cls(**thread_doc)

    @classmethod
    def from_text(cls, text, file_path, db_name, collection):
        # assuming text holds a string containing the email thread
        thread_doc = dict()
        thread_doc["_id"] = str(ObjectId())
        thread_doc["file_path"] = file_path
        thread_doc["messages"] = list()

        message = dict()
        message["_id"] = str(ObjectId())
        message["body"] = text
        message["is_main"] = True
        thread_doc["messages"].append(message)

        # rename _id to id and convert to string for each message
        for i in thread_doc["messages"]:
            i["id"] = str(i.pop("_id"))

        messages = [EmailMessage(**i) for i in thread_doc["messages"]]
        thread_doc["messages"] = messages
        thread_doc["db_name"] = db_name
        thread_doc["collection"] = collection
        thread_doc["id"] = str(thread_doc.pop("_id"))

        return cls(**thread_doc)

    def to_db_entry(self) -> ThreadEntry:
        db_entry = {
            "_id": ObjectId(self.id),
            "file_path": self.file_path,
            "messages": [i.to_db_entry() for i in self.messages],
        }
        return ThreadEntry(**db_entry).model_dump()
    
    def save(self, target_collection: Union[str, None] = None):

        if target_collection is None:
            query_manager.connection[self.db_name][self.collection].update_one(
                {"_id": ObjectId(self.id)},
                {"$set": self.to_db_entry()},
                upsert=True,
            )
        else:
            query_manager.connection[self.db_name][target_collection].update_one(
                {"_id": ObjectId(self.id)},
                {"$set": self.to_db_entry()},
                upsert=True,
            )

    def __str__(self):
        return f"EmailThread({self.id}, {self.file_path})"

    


# Path: data/EmailMessage.py
