from common_classes.EmailThread import EmailThread
from data.enron.EnronMessage import EnronMessage
import re


class EnronThread(EmailThread):

    def __init__(
        self, _id=None, messages=[], file_path=None, db_name=None, collection=None
    ):
        super().__init__(
            _id=_id,
            messages=messages,
            file_path=file_path,
            db_name=db_name,
            collection=collection,
        )

        if self.messages:
            self.messages = [EnronMessage(**i) for i in messages]

    def extract_forwarded_messages(self, message: EnronMessage):

        fw_regex = re.compile(
            r"(?:-+)\s+Forwarded\s+by\s+(?P<sender>.*)\s+on\s+(?P<datetime>(?:\d){1,2}\/(?:\d){1,2}\/(?:\d){2,4}\s+(?:\d){1,2}:(?:\d){2}\s+(?:(?:AM)|(?:PM)))\s+-+",
            re.MULTILINE | re.DOTALL,
        )
        split_message = fw_regex.split(message.body)

        if len(split_message) > 1:

            super().extract_forwarded_messages(self._id, message, split_message)
            self.messages = [EnronMessage(**i.__dict__) for i in self.messages]
        self.messages = [EnronMessage(**i.__dict__) for i in self.messages]

    def split_original_messages(self, message: EnronMessage):
        og_regex = re.compile(
            r"(?:(?:-+)\s*Original\s*Message\s*-+)",
            re.MULTILINE | re.DOTALL,
        )
        split_message = og_regex.split(message.body)

        if len(split_message) > 1:

            super().split_original_messages(message, split_message)

        self.messages = [EnronMessage(**i.__dict__) for i in self.messages]

    def clean(self):
        for i in self.messages:
            self.split_original_messages(i)

        # make sure self.messages is correctly updated with the new messages

        self.messages = [EnronMessage(**i.__dict__) for i in self.messages]

        for i in self.messages:
            # adjust headers extraction to also works for forwarded messages

            try:  # temporary fix
                i.extract_headers()
            except AttributeError:
                i = EnronMessage(**i.__dict__)
                i.extract_headers()
            self.extract_forwarded_messages(i)
