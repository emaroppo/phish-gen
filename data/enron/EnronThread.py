from pydantic import root_validator
from typing import List, Union
from common_classes.EmailThread import EmailThread
from data.enron.EnronMessage import EnronMessage
from common_classes.EmailMessage import EmailMessage
import re


class EnronThread(EmailThread):
    messages: List[Union[EnronMessage, EmailMessage]]

    @root_validator(pre=True)
    def convert_messages(cls, values):
        messages = values.get('messages', [])
        for i, message in enumerate(messages):
            if isinstance(message, EmailMessage):
                message = EnronMessage(**message.model_dump())
            if isinstance(message, dict):
                if "_id" in message:
                    message["id"] = message.pop("_id")
                message = EnronMessage(**message)
            messages[i] = message
        values['messages'] = messages
        return values

    def extract_forwarded_messages(self, message: EnronMessage):

        fw_regex = re.compile(
            r"(?:-+)\s+Forwarded\s+by\s+(?P<sender>.*)\s+on\s+(?P<datetime>(?:\d){1,2}\/(?:\d){1,2}\/(?:\d){2,4}\s+(?:\d){1,2}:(?:\d){2}\s+(?:(?:AM)|(?:PM)))\s+-+",
            re.MULTILINE | re.DOTALL,
        )
        split_message = fw_regex.split(message.body)

        if len(split_message) > 1:

            super().extract_forwarded_messages(self.id, message, split_message)

        self.messages = [
            EnronMessage(**i.dict()) if isinstance(i, EmailMessage) else i
            for i in self.messages
        ]

    def split_original_messages(self, message: EnronMessage):
        og_regex = re.compile(
            r"(?:(?:-+)\s*Original\s*Message\s*-+)",
            re.MULTILINE | re.DOTALL,
        )
        split_message = og_regex.split(message.body)

        if len(split_message) > 1:

            super().split_original_messages(message, split_message)

        self.messages = [EnronMessage(**i.dict()) for i in self.messages]

    def clean(self):
        for i in self.messages:
            self.split_original_messages(i)

        # make sure self.messages is correctly updated with the new messages

        self.messages = [EnronMessage(**i.dict()) for i in self.messages]

        for i in self.messages:
            # adjust headers extraction to also works for forwarded messages

            try:  # temporary fix
                i.extract_headers()
            except AttributeError:
                i = EnronMessage(**i.dict())
                i.extract_headers()
            self.extract_forwarded_messages(i)
