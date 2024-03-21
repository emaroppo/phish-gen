from tqdm import tqdm
import os
import email
import re
from data.DataImporter import DataImporter
from data.enron.EnronThread import EnronThread


class EnronDataImporter(DataImporter):
    def __init__(self, data_dir, db_name):
        super().__init__(data_dir, db_name)

    def load_raw_data(self):
        for i in tqdm(os.listdir(self.data_dir)):
            if "inbox" in os.listdir(self.data_dir + "/" + i):
                for j in tqdm(os.walk(self.data_dir + "/" + i + "/inbox")):
                    for k in j[2]:
                        with open(j[0] + "/" + k, "r", errors="ignore") as f:
                            try:
                                msg = email.message_from_file(
                                    f, policy=email.policy.default
                                )
                                doc = {
                                    "headers": dict(msg.items()),
                                    "email-threads": msg.get_payload(),
                                    "filepath": j[0] + "/" + k,
                                }
                                self.save_raw(doc)
                            except email.errors.MessageError:
                                print("Error in file: ", j[0] + "/" + k)

    def isolate_multiparts(self, sample=None):
        match_dict = {
            "email-threads": {"$regex": "(-Original Message-)|(- Forwarded by)"}
        }
        if type(sample) == int:
            multithread_docs = self.retrieve_samples(match_dict, n=sample)
        elif sample is None:
            multithread_docs = self.db["raw_data"].find(match_dict)

        self.db["raw_data_multipart"].insert_many(multithread_docs)

    def split_forwarded_messages(self, message):

        # deal with forwarded messages
        fw_regex = re.compile(
            r"(?:-+)\s+Forwarded\s+by\s+(?P<sender>.*)\s+on\s+(?P<datetime>(?:\d){1,2}\/(?:\d){1,2}\/(?:\d){2,4}\s+(?:\d){1,2}:(?:\d){2}\s+(?:(?:AM)|(?:PM)))\s+-+",
            re.MULTILINE | re.DOTALL,
        )

        # split message body with regex, return a list of dictionaries, where each dictionary contains the sender and the datetime of the forwarded message
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

    def split_multipart_data(self, sample=None):
        if sample is not None:
            multithread_docs = self.retrieve_samples(
                {"email-threads": {"$regex": "(- Forwarded by)"}},
                collection="raw_data_multipart",
                n=sample,
            )
        else:
            multithread_docs = self.db["raw_data_multipart"].find(
                {"email-threads": {"$regex": "(- Forwarded by)"}}
            )

        for doc in multithread_docs:
            self.split_original_messages(doc["email-threads"])
            # self.split_forwarded_messages(doc["email-threads"])
