from bson import ObjectId


class EmailMessage:
    @classmethod
    def from_text(
        cls,
        text,
        is_main=False,
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
        _id=None,
        is_main=False,
        headers=dict(),
        body="",
        response=None,
    ):
        self._id = _id
        self.is_main = is_main
        self.headers = headers
        self.body = body
        self.response = response

    def to_db_entry(self):
        db_entry = {
            "_id": self._id,
            "is_main": self.is_main,
            "headers": self.headers,
            "body": self.body,
        }
        if self.response is not None:
            db_entry["response"] = self.response
        return db_entry

    def __str__(self):
        return f"EmailMessage({self.subject}, {self.body}, {self.to})"

    def send(self):
        print(f"Sending email to {self.to}")
