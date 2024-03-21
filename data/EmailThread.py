from data.QueryManager import query_manager
from bson import ObjectId
from data.EmailMessage import EmailMessage
from typing import List


class EmailThread:

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

    def split_thread(self, original_message: EmailMessage, split_items: List[str]):
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

    def extract_forwarded_messages(
        self, original_message: EmailMessage, split_items: List[str]
    ):
        message_thread = list()
        original_message.body = split_items.pop(0)
        # TODO: update the original message body in the db

        previous_id = original_message._id

        # it seems all threads have at most one forwarded message
        forwarded_message = EmailMessage.from_text(
            split_items.pop(-1), forwarded_by=previous_id
        )
        # TODO: append the forwarded message to the messages field of the db entry

    def __str__(self):
        return f"EmailThread({self._id}, {self.file_path})"


# Path: data/EmailMessage.py
