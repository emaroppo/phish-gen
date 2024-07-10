from offline_finetuning.common_classes.QueryManager import query_manager
from bson import ObjectId
from offline_finetuning.common_classes.EmailMessage import EmailMessage
from typing import List, Optional, Dict, Any
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

    def split_original_messages(
        self, original_message: EmailMessage, split_items: List[str]
    ):
        message_thread = list()
        original_message.body = split_items.pop(0)
        message_thread.append(original_message)
        previous_id = original_message.id

        for i in split_items:
            message = EmailMessage.from_text(i, response=previous_id)
            message_thread.append(message)
            previous_id = message.id

        # save the split thread in the db
        self.messages = message_thread

        self.save()

    def extract_forwarded_messages(
        self,
        thread_id: ObjectId,
        original_message: EmailMessage,
        split_items: List[str],
    ):
        original_message.body = split_items.pop(0)
        # all threads seems to have at most one forwarded message, will update the code to handle multiple forwarded messages if needed
        new_message = EmailMessage.from_text(
            split_items.pop(-1), forwarded_by=original_message.id
        )
        # update the original message body in the db
        # not a fan of this approach if possible would like to change to changing the original EmailMessage object and then saving the changes to db
        query_manager.connection[self.db_name][self.collection].update_one(
            {"_id": thread_id, "messages._id": original_message.id},
            {"$set": {"messages.$[elem].body": original_message.body}},
            array_filters=[{"elem._id": original_message.id}],
        )

        # save the forwarded message
        query_manager.connection[self.db_name][self.collection].update_one(
            {"_id": thread_id},
            {"$push": {"messages": new_message.to_db_entry()}},
        )
        self.messages.append(new_message)
        self.save()

    def extract_disclaimers(self, disclaimer: str, save: bool = True):
        for i in self.messages:
            i.extract_disclaimer(disclaimer)

        if save:
            self.save()

    def remove_footers(self, footer: str, save: bool = True):
        for i in self.messages:
            i.remove_footer(footer)

        if save:
            self.save()

    def extract_entities(
        self,
        placeholder: str = None,
        regex: str = None,
        message_labeller: Any = None,
        detection_mode: str = "manual",
        save=True,
    ):
        if detection_mode == "manual":
            if not placeholder or not regex:
                raise ValueError(
                    "Placeholder and regex must be provided for manual entity extraction"
                )
            for message in self.messages:
                extracted_entities = message.extract_entities(
                    placeholder,
                    regex,
                )
                for entity in extracted_entities:
                    start, end, label, entity_value = entity
                    message.add_entity(
                        start=start,
                        end=end,
                        entity_value=entity_value,
                        entity_type=label,
                        detection_method=detection_mode,
                    )

        elif detection_mode == "auto":
            for message in self.messages:
                extracted_entities = message_labeller.label_message(message.body)
                if extracted_entities:
                    for entity in extracted_entities:
                        start, end, label = entity
                        entity_value = message.body[start:end]
                        message.add_entity(
                            entity_type=label,
                            start=start,
                            end=end,
                            entity_value=entity_value,
                            detection_method=detection_mode,
                        )

        if save:
            self.save()

    def predict_sentiment(self, sentiment_predictor: Any, save=True):
        for message in self.messages:
            sentiment = sentiment_predictor.label_message(message.body)
            message.add_sentiment(sentiment)

        if save:
            self.save()

    def to_db_entry(self) -> ThreadEntry:
        db_entry = {
            "_id": ObjectId(self.id),
            "file_path": self.file_path,
            "messages": [i.to_db_entry() for i in self.messages],
        }
        return ThreadEntry(**db_entry).model_dump()

    def __str__(self):
        return f"EmailThread({self.id}, {self.file_path})"

    def save(self, target_collection: str = None):
        entry = self.to_db_entry()

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


# Path: data/EmailMessage.py
