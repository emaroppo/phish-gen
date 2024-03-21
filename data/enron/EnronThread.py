from data.EmailThread import EmailThread
import re


class EnronThread(EmailThread):
    def __init__(self, _id=None, messages=[], file_path=None):
        super().__init__(_id, messages, file_path)

    def split_forwarded_messages(self, message):

        # TODO:rewrite the class to use the EmailMessage class and the parent class
        # This class should mostly deal with extracting the data correctly via regex
        fw_regex = re.compile(
            r"(?:-+)\s+Forwarded\s+by\s+(?P<sender>.*)\s+on\s+(?P<datetime>(?:\d){1,2}\/(?:\d){1,2}\/(?:\d){2,4}\s+(?:\d){1,2}:(?:\d){2}\s+(?:(?:AM)|(?:PM)))\s+-+",
            re.MULTILINE | re.DOTALL,
        )
        split_message = fw_regex.split(message)
        if len(split_message) == 4:
            split_message_doc = {
                "message": split_message[0],
                "sender": split_message[1],
                "datetime": split_message[2],
                "forwarded_message": split_message[3],
            }
            print(split_message_doc)
        elif len(split_message) > 4:
            print(len(split_message))
            split_message_doc = []
            for i in range(len(split_message) % 4):
                split_message_part_doc = {
                    "message": split_message[i * 4],
                    "sender": split_message[i * 4 + 1],
                    "datetime": split_message[i * 4 + 2],
                    "forwarded_message": split_message[i * 4 + 3],
                }
                split_message_doc.append(split_message_part_doc)
            print(split_message_doc)

    def split_original_messages(self, message):
        og_regex = re.compile(
            r"(?:(?:-+)\s*Original\s*Message\s*-+)",
            re.MULTILINE | re.DOTALL,
        )
        split_message = og_regex.split(message)
        split_message_doc = [
            {"message": split_message[i]} for i in range(len(split_message))
        ]
        for i in split_message_doc:
            print(i)
