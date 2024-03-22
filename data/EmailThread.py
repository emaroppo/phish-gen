from data.QueryManager import query_manager
from bson import ObjectId
from data.EmailMessage import EmailMessage
from typing import List


class EmailThread:

    @classmethod
    def from_db(cls, _id, db_name, collection):
        # retrieve the thread from the database
        thread_doc = query_manager.retrieve_entry(
            {"_id": _id}, db_name=db_name, collection=collection
        )
        return cls(**thread_doc)

    @classmethod
    def from_text(cls, text, file_path, db_name, collection):
        # assuming text holds a string containing the email thread
        thread_doc = dict()
        thread_doc["_id"] = ObjectId()
        thread_doc["file_path"] = file_path
        thread_doc["messages"] = list()

        message = dict()
        message["_id"] = ObjectId()
        message["body"] = text
        message["is_main"] = True
        thread_doc["messages"].append(message)

        query_manager.save_entry(thread_doc, db_name, collection)

        return cls(**thread_doc)

    def __init__(self, db_name, collection, _id=None, messages=[], file_path=None):
        self._id = _id
        self.messages = messages
        self.file_path = file_path
        self.db_name = db_name
        self.collection = collection

    def split_original_messages(
        self, original_message: EmailMessage, split_items: List[str]
    ):
        message_thread = list()
        original_message.body = split_items.pop(0)
        message_thread.append(original_message)
        previous_id = original_message._id

        for i in split_items:
            message = EmailMessage.from_text(i, response=previous_id)
            message_thread.append(message)
            previous_id = message._id

        # save the split thread
        message_thread_entries = [i.to_db_entry() for i in message_thread]
        query_manager.connection[self.db_name][self.collection].update_one(
            {"_id": self._id}, {"$set": {"messages": message_thread_entries}}
        )
        self.messages = message_thread

    def extract_forwarded_messages(
        self,
        thread_id: ObjectId,
        original_message: EmailMessage,
        split_items: List[str],
    ):
        original_message.body = split_items.pop(0)
        # all threads seems to have at most one forwarded message, will update the code to handle multiple forwarded messages if needed
        new_message = EmailMessage.from_text(
            split_items.pop(-1), forwarded_by=original_message._id
        )
        # update the original message body in the db
        query_manager.connection[self.db_name][self.collection].update_one(
            {"_id": thread_id, "messages._id": original_message._id},
            {"$set": {"messages.$[elem].body": original_message.body}},
            array_filters=[{"elem._id": original_message._id}],
        )

        # save the forwarded message
        query_manager.connection[self.db_name][self.collection].update_one(
            {"_id": thread_id},
            {"$push": {"messages": new_message.to_db_entry()}},
        )
        self.messages.append(new_message)

    def to_db_entry(self):
        db_entry = {
            "_id": self._id,
            "file_path": self.file_path,
            "messages": [i.to_db_entry() for i in self.messages],
        }
        return db_entry

    def __str__(self):
        return f"EmailThread({self._id}, {self.file_path})"

    def save(self):
        query_manager.save_entry(self.to_db_entry(), self.db_name, self.collection)


# Path: data/EmailMessage.py
