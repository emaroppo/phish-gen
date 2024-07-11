from bson import ObjectId
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
import re


class EmailMessageEntry(BaseModel):
    id: ObjectId = Field(alias="_id")
    is_main: bool
    headers: Optional[Dict[str, str]] = None
    body: str
    response: Optional[ObjectId] = None
    forwarded_by: Optional[ObjectId] = None
    entities: Optional[Dict[str, Dict[str, List]]] = None
    sentiment: Optional[List[Dict[str, Union[float, str]]]] = None
    topic: Optional[List[List[Union[str, float]]]] = None
    attachments_format: Optional[List[str]] = None
    disclaimer: Optional[str] = None
    is_html: Optional[bool] = False
    word_count: Optional[int] = None

    class Config:
        arbitrary_types_allowed = True


class EmailMessage(BaseModel):
    id: str
    is_main: bool
    headers: Optional[Dict[str, str]] = None
    body: str
    response: Optional[str] = None
    forwarded_by: Optional[str] = None
    entities: Optional[Dict[str, Dict[str, List]]] = None
    sentiment: Optional[List[Dict[str, Union[float, str]]]] = None
    topic: Optional[List[List[Union[str, float]]]] = None
    attachments_format: Optional[List[str]] = None
    disclaimer: Optional[str] = None
    is_html: Optional[bool] = False
    word_count: Optional[int] = None

    @classmethod
    def deserialize(cls, data: Dict[str, Any]):
        object_id_fields = ["_id", "response", "forwarded_by"]
        for field in object_id_fields:
            if field in data:
                data[field] = str(data[field])

        # update _id key to id
        if "_id" in data:
            data["id"] = data.pop("_id")

        return cls(**data)

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
        message_doc["id"] = str(ObjectId())
        message_doc["body"] = text
        message_doc["is_main"] = is_main

        if response is not None:
            message_doc["response"] = response
        if forwarded_by is not None:
            message_doc["forwarded_by"] = forwarded_by

        return cls(**message_doc)

    def clean_subject(self):

        if "Subject" in self.headers:

            # remove leading and trailing white spaces
            self.headers["Subject"] = self.headers["Subject"].strip()

            # replace all spaces with a single space
            self.headers["Subject"] = re.sub(r"\s+", " ", self.headers["Subject"])

            # separate fwd and re from the subject
            # TO DO: come up with a way to include re and fwd in entry
            self.headers["Subject"] = re.sub(
                r"\s*Fwd:\s*", "", self.headers["Subject"], flags=re.IGNORECASE
            )
            self.headers["Subject"] = re.sub(
                r"\s*Re:\s*", "", self.headers["Subject"], flags=re.IGNORECASE
            )

    def check_html(self):
        if re.search(r"<html>", self.body, re.IGNORECASE):
            self.is_html = True

    def get_word_count(self):
        self.word_count = len(self.body.split())
        return self.word_count

    def extract_disclaimer(self, disclaimer: str):
        if re.search(disclaimer, self.body):
            self.__dict__["disclaimer"] = re.search(disclaimer, self.body).group()
            self.body = re.sub(disclaimer, "", self.body)

        return self.body

    def extract_entities(self, placeholder: str, regex: str):

        # find values matching regex, save the value, the start and end indices
        matches = re.finditer(regex, self.body)
        extracted_entities = list()

        for match in matches:
            value = match.group()
            label = placeholder
            start = match.start()
            end = match.end()
            extracted_entities.append((start, end, label, value))

        return extracted_entities

    def add_entity(
        self,
        entity_type: str,
        entity_value: str,
        start: int,
        end: int,
        detection_method: str = "manual",
    ):
        if self.entities is None:
            self.entities = dict()
        if detection_method not in self.entities:
            self.entities[detection_method] = dict()
        if entity_type not in self.entities[detection_method]:
            self.entities[detection_method][entity_type] = list()

        self.entities[detection_method][entity_type].append((entity_value, start, end))

    def add_sentiment(self, sentiment: Dict[str, float]):
        self.sentiment = sentiment

    def add_topic(self, topic: List[List[Union[str, float]]]):
        self.topic = topic

    def insert_entity_placeholders(
        self, entity_types: List[str], manual=True, auto=False
    ):
        entity_labels = list()
        for entity_type in entity_types:
            if manual and entity_type in self.special_tokens["manual"]["entities"]:
                entity_labels.extend(
                    self.special_tokens["manual"]["entities"][entity_type]
                )
            if auto and entity_type in self.special_tokens["auto"]["entities"]:
                entity_labels.extend(
                    self.special_tokens["auto"]["entities"][entity_type]
                )

        entity_labels.sort(key=lambda x: x[2], reverse=True)
        for label in entity_labels:
            self.body = self.body[: label[2]] + f"<{label[0]}>" + self.body[label[3] :]

    def remove_footer(self, footer: str):
        self.body = re.sub(footer, "", self.body)
        return self.body

    def get_attachment_format(self):
        if "ATTACHMENT" in self.entities["manual"]:
            attachments = [
                entity[0] for entity in self.entities["manual"]["ATTACHMENT"]
            ]

            self.attachments_formats = [
                attachment.split(".")[-1] for attachment in attachments
            ]

    def to_db_entry(self) -> EmailMessageEntry:
        db_entry = {
            "_id": ObjectId(self.id),
            "is_main": self.is_main,
            "headers": self.headers,
            "body": self.body,
        }
        if self.entities is not None:
            db_entry["entities"] = self.entities

        if self.sentiment is not None:
            db_entry["sentiment"] = self.sentiment

        if self.disclaimer is not None:
            db_entry["disclaimer"] = self.disclaimer

        if self.topic is not None:
            db_entry["topic"] = self.topic

        if self.response is not None and self.response != "None":

            db_entry["response"] = ObjectId(self.response)

        if self.forwarded_by is not None and self.forwarded_by != "None":
            db_entry["forwarded_by"] = ObjectId(self.forwarded_by)

        if self.is_html:
            db_entry["is_html"] = self.is_html

        if self.word_count is not None:
            db_entry["word_count"] = self.word_count

        return EmailMessageEntry(**db_entry).model_dump(by_alias=True)

    def __str__(self):
        return f"EmailMessage({self.id}, {self.body})"
