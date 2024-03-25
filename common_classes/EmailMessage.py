from bson import ObjectId


class EmailMessage:
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
        message_doc["_id"] = ObjectId()
        message_doc["body"] = text
        message_doc["is_main"] = is_main

        if response is not None:
            message_doc["response"] = response
        if forwarded_by is not None:
            message_doc["forwarded_by"] = forwarded_by

        return cls(**message_doc)

    def __init__(
        self,
        _id: ObjectId = None,
        is_main: bool = False,
        headers: dict = dict(),
        body: str = "",
        response: ObjectId = None,
        forwarded_by: ObjectId = None,
    ):
        self._id = _id
        self.is_main = is_main
        self.headers = headers
        self.body = body
        self.response = response
        self.forwarded_by = forwarded_by

    def to_db_entry(self):
        db_entry = {
            "_id": self._id,
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
        return f"EmailMessage({self._id}, {self.body})"
