from data.EmailThread import EmailThread
from data.EmailMessage import EmailMessage
import re


class EnronThread(EmailThread):

    def __init__(self, _id=None, messages=[], file_path=None):
        super().__init__(_id, messages, file_path)

    def extract_forwarded_messages(self, message: EmailMessage):

        fw_regex = re.compile(
            r"(?:-+)\s+Forwarded\s+by\s+(?P<sender>.*)\s+on\s+(?P<datetime>(?:\d){1,2}\/(?:\d){1,2}\/(?:\d){2,4}\s+(?:\d){1,2}:(?:\d){2}\s+(?:(?:AM)|(?:PM)))\s+-+",
            re.MULTILINE | re.DOTALL,
        )
        split_message = fw_regex.split(message.body)

        super().extract_forwarded_messages(self._id, message, split_message)

    def split_original_messages(self, message: EmailMessage):
        og_regex = re.compile(
            r"(?:(?:-+)\s*Original\s*Message\s*-+)",
            re.MULTILINE | re.DOTALL,
        )
        split_message = og_regex.split(message)

        super().split_original_messages(message, split_message)

    def clean(self):
        for i in self.messages:
            self.split_original_messages(i)

        # make sure self.messages is correctly updated with the new messages

        for i in self.messages:
            self.extract_forwarded_messages(i)
