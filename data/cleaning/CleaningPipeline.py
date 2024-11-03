from pydantic import BaseModel
from typing import List
from data.QueryManager import query_manager
from data.classes.EmailMessage import EmailMessage
from data.classes.EmailThread import EmailThread
from bson import ObjectId
from tqdm import tqdm
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
        self, path_list: List[str], out_collection: str = "single_messages"
    ):
        for path in tqdm(path_list, desc="Loading files"):
            try:
                with open(path, "r", errors="ignore") as f:
                    if path.endswith(".eml"):
                        email_message = email.message_from_file(f)
                    else:
                        email_message = email.message_from_string(f.read())

                if email_message.is_multipart():
                    for part in email_message.walk():
                        if part.get_content_type() == "text/plain":
                            try:
                                thread_body = part.get_payload(decode=True).decode(
                                    part.get_content_charset(failobj="utf-8"),
                                    errors="ignore",
                                )
                            except LookupError:
                                thread_body = part.get_payload(decode=True).decode(
                                    "utf-8", errors="ignore"
                                )
                            break
                else:
                    thread_body = email_message.get_payload(decode=True).decode(
                        email_message.get_content_charset(failobj="utf-8"),
                        errors="ignore",
                    )

                thread = EmailThread.from_text(
                    thread_body, path, self.database_name, out_collection
                )
                thread.messages[0].is_main = True
                thread.messages[0].headers = {k: v for k, v in email_message.items()}
                if not thread.save():
                    print(f"Failed to save thread from {path}")

            except UnicodeDecodeError:
                raise
                continue
            except FileNotFoundError:
                print(f"File not found: {path}")
                continue
            except Exception as e:
                print(f"An error occurred while processing {path}: {e}")
                continue

        return True

    def parse_alt_separator(self, separator_string: str) -> dict:
        alt_separator_parsing = r"On\s+(?P<date>(?:(?:Mon(?:day)?|Tues(?:day)?|Wed(?:nesday)?|Thu(?:rsday)?|Fri(?:day)?|Sat(?:urday)?|Sun(?:day)?),?\s+)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d+,?\s+\d{4}(?:,?\s+(?:at\s+)?\d+:\d+\s+(?:AM|PM))?),?\s+(?P<sender>.+?(\s*<\s*.+)?)\s+wrote:"
        match = re.search(alt_separator_parsing, separator_string)
        if match:
            return {"Sent": match.group("date"), "From": match.group("sender")}
        else:
            return None

    def parse_headers_string(self, header_string: str) -> dict:
        header_parsing_regular_expression = r"^\*?([\w\-]*?):\*?"
        header_parsing_regular_expression = re.compile(
            header_parsing_regular_expression, re.MULTILINE
        )
        headers = header_parsing_regular_expression.split(header_string, maxsplit=0)
        header_fields = headers[1::2]
        header_values = headers[2::2]
        headers = {k: v for k, v in zip(header_fields, header_values)}
        return headers

    def extract_headers_single(self, message_body: str) -> dict:
        header_splitting_regular_expression = r"(?:_+)?\s*(\*?From:\*?.*?[\n\r](?:.*?[\n\r])*?\*?Subj(?:ect)?:\*?([\w\s.,!?;:'\"@\—\-\–\[\]\(\)/$’+&%#|“”]+?)?)[\n\r]{2,}"
        header_splitting_regular_expression = re.compile(
            header_splitting_regular_expression, re.MULTILINE | re.DOTALL
        )
        header_string = header_splitting_regular_expression.search(message_body)

        try:
            header_string = header_string.group(1)
        except AttributeError:
            print(message_body)

        # remove the header string from the thread string
        message_body = header_splitting_regular_expression.sub(
            "", message_body, count=1
        )

        headers = self.parse_headers_string(header_string)
        return headers, message_body

    def extract_headers(self, thread_list: List[EmailThread]) -> List[EmailThread]:
        header_splitting_regular_expression = r"(?:_+)?\s*(\*?From:\*?.*?[\n\r](?:.*?[\n\r])*?\*?Subj(?:ect)?:\*?([\w\s.,!?;:'\"@\—\-\–\[\]\(\)/$’+&%#|“”]+?)?)[\n\r]{2,}"

        header_splitting_regular_expression = re.compile(
            header_splitting_regular_expression
        )
        for thread in tqdm(thread_list, desc="Extracting headers"):
            print(thread.id)
            for message in thread.messages:
                if message.headers is None and re.match(
                    header_splitting_regular_expression, message.body
                ):
                    message.headers, message.body = self.extract_headers_single(
                        message.body + "\n\n"
                    )

            thread.save()

        return thread_list

    def split_multipart_messages(
        self,
        thread_list: List[EmailThread],
        message_split_regex: re.Pattern,
    ) -> List[EmailThread]:

        for thread in tqdm(thread_list, desc="Splitting multipart messages"):
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

        thread_list = self.clean_bodies(thread_list)
        thread_list = self.extract_headers(thread_list)
        return thread_list

    def split_message_alt_single(self, message_body: str, thread_subject: str):
        alt_separator = r"(On\s+(?:(?:(?:Mon(?:day)?|Tues(?:day)?|Wed(?:nesday)?|Thu(?:rsday)?|Fri(?:day)?|Sat(?:urday)?|Sun(?:day)?),?\s+)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d+,\s+\d{4}(?:,?\s+(?:at\s+)?\d+:\d+\s+(?:AM|PM))?),?\s+(?:.+?(?:\s*<\s*.+)?)\s+wrote:)"
        alt_separator = re.compile(alt_separator)
        alt_separator_parsing = r"On\s+(?P<date>(?:(?:Mon(?:day)?|Tues(?:day)?|Wed(?:nesday)?|Thu(?:rsday)?|Fri(?:day)?|Sat(?:urday)?|Sun(?:day)?),?\s+)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d+,?\s+\d{4}(?:,?\s+(?:at\s+)?\d+:\d+\s+(?:AM|PM))?),?\s+(?P<sender>.+?(?:\s*<\s+.+)?)\s+wrote:"
        alt_separator_parsing = re.compile(alt_separator_parsing)

        thread_messages = alt_separator.split(message_body)
        first_message = thread_messages.pop(0)
        alt_separator_strings = thread_messages[::2]
        bodies_list = thread_messages[1::2]

        header_dicts = list()
        for header_string in alt_separator_strings:
            headers = self.parse_alt_separator(header_string)
            headers["Subject"] = thread_subject
            header_dicts.append(headers)

        thread_messages = [
            {"header": header, "body": body}
            for header, body in zip(header_dicts, bodies_list)
        ]

        return first_message, thread_messages

    def split_message_alt(self, thread_list: List[EmailThread]):
        thread_list = self.clean_bodies(thread_list)
        for thread in tqdm(thread_list, desc="Splitting messages on headers"):
            print(thread.id)
            message_thread = list()
            if "Subject" not in thread.messages[0].headers:
                continue
            main_subject = thread.messages[0].headers["Subject"]
            for message in thread.messages:
                if message.headers is not None and "Subject" in message.headers:
                    subject = message.headers["Subject"]
                else:
                    subject = main_subject
                new_message_body, new_messages = self.split_message_alt_single(
                    message.body, subject
                )

                if len(new_messages) > 0:
                    message.body = new_message_body
                    previous_id = message.id
                    message_thread.append(message)

                    for i in new_messages:
                        headers = i["header"]
                        body = i["body"]
                        message = EmailMessage.from_text(body, response=previous_id)
                        message.headers = headers
                        previous_id = message.id
                        message_thread.append(message)

                else:
                    message_thread.append(message)
        thread_list = self.clean_bodies(thread_list)
        return thread_list

    def split_on_headers_single(self, thread_string):
        header_splitting_regular_expression = r"(?:_+)?\s+(\*?From:\*?\s(?:.*)\n(?:[\s\S]*?)\n\*?Subj(?:ect)?:\*?\s(?:.*))"
        header_splitting_regular_expression = re.compile(
            header_splitting_regular_expression
        )

        thread_messages = header_splitting_regular_expression.split(
            thread_string, maxsplit=0
        )
        first_message = thread_messages.pop(0)
        header_list = thread_messages[::2]
        bodies_list = thread_messages[1::2]

        header_dicts = list()
        for header_string in header_list:
            headers = self.parse_headers_string(header_string)
            header_dicts.append(headers)

        thread_messages = [
            {"header": header, "body": body}
            for header, body in zip(header_dicts, bodies_list)
        ]
        thread_messages.insert(0, {"header": None, "body": first_message})
        return first_message, thread_messages

    def split_on_headers(self, thread_list: List[EmailThread]):
        self.clean_bodies(thread_list)
        for thread in tqdm(thread_list, desc="Splitting messages on headers"):
            message_thread = list()
            for message in thread.messages:

                new_message_body, new_messages = self.split_on_headers_single(
                    message.body
                )

                if len(new_messages) > 0:
                    message.body = new_message_body
                    previous_id = message.id
                    message_thread.append(message)
                    # pair body with headers strings from headers capture group
                    for i in new_messages:
                        headers = i["header"]
                        body = i["body"]
                        message = EmailMessage.from_text(body, response=previous_id)
                        message.headers = headers
                        previous_id = message.id
                        message_thread.append(message)

        thread_list = self.clean_bodies(thread_list)
        return thread_list

    def split_forwarded_messages(
        self,
        thread_list: List[EmailThread],
        forwarded_regex: re.Pattern,
    ):

        for thread in tqdm(thread_list, desc="Splitting forwarded messages"):
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

        self.clean_bodies(thread_list)
        self.extract_headers(thread_list)

        return thread_list

    def remove_noise(self, thread_list: List[EmailThread]):
        for disclaimer_pattern in noise["disclaimer"]:
            for thread in tqdm(thread_list, desc="Removing noise"):
                for message in thread.messages:
                    if re.search(disclaimer_pattern, message.body):
                        message.disclaimer = re.search(
                            disclaimer_pattern, message.body
                        ).group()
                        message.body = re.sub(disclaimer_pattern, "", message.body)

        for footer_pattern in noise["footer"]:
            for thread in tqdm(thread_list, desc="Removing noise"):
                for message in thread.messages:
                    if re.search(footer_pattern, message.body):
                        message.body = re.sub(footer_pattern, "", message.body)

        return thread_list

    def clean_bodies(self, thread_list: List[EmailThread]):
        sequences = ["=20", "=09", "=\n", "=3D"]

        for thread in tqdm(thread_list, desc="Basic cleaning"):
            for message in thread.messages:
                for sequence in sequences:
                    message.body = message.body.replace(sequence, "")

                message.body = message.body.replace("[IMAGE]", "")
                cleaned_lines = []
                for line in message.body.split("\n"):
                    # Use regex to remove leading spaces and '>' characters
                    cleaned_line = re.sub(r"^[\t >]+", "", line)
                    cleaned_lines.append(cleaned_line)
                message.body = "\n".join(cleaned_lines)

                message.body = message.body.strip()

        return thread_list

    def clean_subject(self, thread_list: List[EmailThread]):
        for thread in tqdm(thread_list, desc="Cleaning subjects"):
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

    def run_pipeline(self, path_list: List[str] = None):
        if path_list:
            self.load_raw_data(path_list)

        threads = query_manager.connection[self.database_name]["single_messages"].find()
        threads = [
            EmailThread.deserialize(
                thread, db_name=self.database_name, collection="single_messages"
            )
            for thread in threads
        ]

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

            for i in tqdm(multipart_messages):

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

                for j in forwarded_messages:
                    j.save("forwarded_messages")

                forwarded_messages_id = [ObjectId(k.id) for k in forwarded_messages]
                query_manager.connection[self.database_name][i].delete_many(
                    {"_id": {"$in": forwarded_messages_id}}
                )

        # split on headers

        for i in ["single_messages", "multipart_messages", "forwarded_messages"]:

            # TODO: add filter to only retrieve messages that match the pattern
            header_splitting_regular_expression = r"(?:_+)?\s*(\*?From:\*?.*?[\n\r](?:.*?[\n\r])*?\*?Subj(?:ect)?:\*?([\w\s.,!?;:'\"@\—\-\–\[\]\(\)/$’+*&%#\|“”\‘\’]+?)?)[\n\r]{2,}"

            threads = query_manager.connection[self.database_name][i].find(
                {"messages.body": {"$regex": header_splitting_regular_expression}}
            )
            threads = [
                EmailThread.deserialize(
                    thread, db_name=self.database_name, collection=i
                )
                for thread in threads
            ]
            cleaned_data = self.split_on_headers(threads)

            for j in cleaned_data:
                j.save(target_collection="multipart_messages")

            multipart_messages_id = [ObjectId(i.id) for i in cleaned_data]
            query_manager.connection[self.database_name]["single_messages"].delete_many(
                {"_id": {"$in": multipart_messages_id}}
            )

        # split on alt pattern:

        alt_separator = r"(On\s+(?:(?:(?:Mon(?:day)?|Tue(?:sday)?|Wed(?:nesday)?|Thu(?:rsday)?|Fri(?:day)?|Sat(?:urday)?|Sun(?:day)?),?\s+)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d+,?\s+\d{4}(?:,?\s+(?:at\s+)?\d+:\d+\s+(?:AM|PM))?),?\s+(?:.+?(?:\s*<\s+.+)?)\s+wrote:)"
        thread_list = query_manager.connection[self.database_name][
            "single_messages"
        ].find({"messages.body": {"$regex": alt_separator}})
        thread_list = [
            EmailThread.deserialize(
                thread, db_name=self.database_name, collection="single_messages"
            )
            for thread in thread_list
        ]
        cleaned_data = self.split_message_alt(thread_list)

        for i in cleaned_data:
            i.save(target_collection="multipart_messages")

        multipart_messages_id = [ObjectId(i.id) for i in cleaned_data]
        query_manager.connection[self.database_name]["single_messages"].delete_many(
            {"_id": {"$in": multipart_messages_id}}
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
            cleaned_data = self.clean_bodies(cleaned_data)
            for j in cleaned_data:
                j.save(target_collection=i)
