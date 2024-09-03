from pydantic import BaseModel, Field
from typing import List
from data.QueryManager import query_manager
from data.classes.EmailMessage import EmailMessage
from data.classes.EmailThread import EmailThread
from bson import ObjectId
from tqdm import tqdm
import os
import json
import email
import re

with open("data/regex/thread_structure.json", "r") as f:
    thread_structure = json.load(f)

with open("data/regex/noise.json", "r") as f:
    noise = json.load(f)


class CleaningPipeline(BaseModel):
    database_name: str = "enron_datasource"

    def load_raw_data(
        self,
        data_dir: str = "data/datasets_raw/enron",
        out_collection: str = "raw_data",
    ):
        for i in tqdm(os.listdir(data_dir), desc="Loading raw inboxes"):
            if "inbox" in os.listdir(data_dir + "/" + i):
                for j in os.walk(data_dir + "/" + i + "/inbox"):
                    for k in j[2]:
                        with open(j[0] + "/" + k, "r", errors="ignore") as f:
                            text = f.read()
                            thread = EmailThread.from_text(
                                text, j[0] + "/" + k, self.database_name, out_collection
                            )
                            thread.save()
        return True

    def extract_headers(self, thread_list: List[EmailThread]) -> List[EmailThread]:

        for thread in tqdm(thread_list):
            for message in thread.messages:
                if message.headers is None:
                    if message.is_main:
                        message.headers = {
                            k: v
                            for k, v in email.message_from_string(message.body).items()
                        }
                        message.body = email.message_from_string(
                            message.body
                        ).get_payload()
                    else:
                        pattern = thread_structure["headers"]["outer"]
                        match = re.search(pattern, message.body)

                        if match:
                            from_header = match.group("From")
                            subject_header = match.group("Subject")
                            headers_block = match.group("Headers")

                            intermediate_headers = dict(
                                re.findall(
                                    thread_structure["headers"]["inner"],
                                    headers_block,
                                    re.MULTILINE,
                                )
                            )

                            headers = {"From": from_header, "Subject": subject_header}

                            headers.update(intermediate_headers)

                            email_content = re.sub(pattern, "", message.body, count=1)

                            message.headers = headers
                            message.body = email_content

            thread.save()

        return thread_list

    def split_multipart_messages(
        self,
        thread_list: List[EmailThread],
        message_split_regex: re.Pattern,
    ) -> List[EmailThread]:

        # Need to update this loop! using multiple separators would probably overwrite the messages
        for thread in tqdm(thread_list):
            message_thread = list()
            for message in thread.messages:
                split_message = message_split_regex.split(message.body)
                if len(split_message) > 1:
                    message.body = split_message.pop(0)
                    previous_id = message.id
                    message_thread.append(message)

                    for i in split_message:
                        message = EmailMessage.from_text(i, response=previous_id)
                        previous_id = message.id
                        message_thread.append(message)

                else:
                    message_thread.append(message)

            thread.messages = message_thread

        return thread_list

    def split_forwarded_messages(
        self,
        thread_list: List[EmailThread],
        forwarded_regex: re.Pattern,
    ):

        for thread in tqdm(thread_list):
            for message in thread.messages:
                split_message = forwarded_regex.split(message.body)
                # all threads seems to have at most one forwarded message, will update the code to handle multiple forwarded messages if needed
                if len(split_message) > 1:
                    message.body = split_message.pop(0)
                    new_message = EmailMessage.from_text(
                        split_message.pop(-1), forwarded_by=message.id
                    )
                    # find a way to preserve the order of the messages
                    thread.messages.append(new_message)

        return thread_list

    def remove_noise(self, thread_list: List[EmailThread]):
        for disclaimer_pattern in noise["disclaimer"]:
            for thread in tqdm(thread_list):
                for message in thread.messages:
                    if re.search(disclaimer_pattern, message.body):
                        message.disclaimer = re.search(
                            disclaimer_pattern, message.body
                        ).group()
                        message.body = re.sub(disclaimer_pattern, "", message.body)

        for footer_pattern in noise["footer"]:
            for thread in tqdm(thread_list):
                for message in thread.messages:
                    if re.search(footer_pattern, message.body):
                        message.body = re.sub(footer_pattern, "", message.body)

        return thread_list

    def clean_subject(self, thread_list: List[EmailThread]):
        for thread in tqdm(thread_list):
            for message in thread.messages:
                if "Subject" in message.headers:
                    message.headers["Subject"] = message.headers["Subject"].strip()
                    message.headers["Subject"] = re.sub(
                        r"\s+", " ", message.headers["Subject"]
                    )
                    message.headers["Subject"] = re.sub(
                        r"\s*Fwd:\s*",
                        "",
                        message.headers["Subject"],
                        flags=re.IGNORECASE,
                    )
                    message.headers["Subject"] = re.sub(
                        r"\s*Re:\s*",
                        "",
                        message.headers["Subject"],
                        flags=re.IGNORECASE,
                    )

        return thread_list

    def run_pipeline(self):
        self.load_raw_data()
        # retrieve raw data
        # extract headers
        # save to a new collection (clean_data)

        threads = query_manager.connection[self.database_name]["raw_data"].find()
        raw_data = [
            EmailThread.deserialize(
                thread, db_name=self.database_name, collection="raw_data"
            )
            for thread in threads
        ]
        cleaned_data = self.extract_headers(raw_data)
        for i in cleaned_data:
            i.save("single_messages")

        # retrieve multipart messages
        # copy them to a new collection
        # delete them from the original collection
        # split them into individual messages
        # extract headers where is_main is False

        for message_split_regex in thread_structure["reply_separator"]:
            message_split_regex = re.compile(message_split_regex)
            multipart_messages = query_manager.connection[self.database_name][
                "single_messages"
            ].find({"messages.body": {"$regex": message_split_regex}})
            multipart_messages = [
                EmailThread.deserialize(
                    thread, db_name=self.database_name, collection="single_messages"
                )
                for thread in multipart_messages
            ]
            multipart_messages = self.split_multipart_messages(
                multipart_messages, message_split_regex
            )
            multipart_messages = self.extract_headers(multipart_messages)
            for i in multipart_messages:

                i.save(target_collection="multipart_messages")

            multipart_messages_id = [ObjectId(i.id) for i in multipart_messages]
            query_manager.connection[self.database_name]["single_messages"].delete_many(
                {"_id": {"$in": multipart_messages_id}}
            )

        # retrieve forwarded messages
        # delete them from the original collection
        # split them into individual messages
        # extract headers where is_main is False

        for forwarded_regex in thread_structure["forwarded_separator"]:
            forwarded_regex = re.compile(forwarded_regex)
            for i in ["single_messages", "multipart_messages"]:
                forwarded_messages = query_manager.connection[self.database_name][
                    i
                ].find({"messages.body": {"$regex": forwarded_regex}})
                forwarded_messages = [
                    EmailThread.deserialize(
                        thread, db_name=self.database_name, collection=i
                    )
                    for thread in forwarded_messages
                ]
                forwarded_messages = self.split_forwarded_messages(
                    forwarded_messages, forwarded_regex
                )
                forwarded_messages = self.extract_headers(forwarded_messages)

                for j in forwarded_messages:
                    j.save("forwarded_messages")

            forwarded_messages_id = [ObjectId(i.id) for i in forwarded_messages]
            query_manager.connection[self.database_name]["single_messages"].delete_many(
                {"_id": {"$in": forwarded_messages_id}}
            )

        # remove noise
        for i in ["single_messages", "multipart_messages", "forwarded_messages"]:
            threads = query_manager.connection[self.database_name][i].find()
            threads = [
                EmailThread.deserialize(
                    thread, db_name=self.database_name, collection=i
                )
                for thread in threads
            ]
            cleaned_data = self.remove_noise(threads)
            for j in cleaned_data:
                j.save(target_collection=i)
