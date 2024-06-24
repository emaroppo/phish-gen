from bson import ObjectId
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import re


class EmailMessageEntry(BaseModel):
    id: ObjectId = Field(alias="_id")
    is_main: bool
    headers: Optional[Dict[str, str]] = None
    body: str
    response: Optional[ObjectId] = None
    forwarded_by: Optional[ObjectId] = None
    special_tokens: Optional[Dict[str, List[str]]] = None
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
    special_tokens: Optional[Dict[str, List[str]]] = None
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

    def insert_placeholder(self, field_name, placeholder: str, regex: str):
        original_values = re.findall(regex, self.body)
        if original_values:
            if (
                "special_tokens" not in self.__dict__
                or not self.__dict__["special_tokens"]
            ):
                self.__dict__["special_tokens"] = dict()
            if field_name in self.__dict__["special_tokens"]:  # store original values
                self.__dict__["special_tokens"][field_name].extend(original_values)
            else:
                self.__dict__["special_tokens"][field_name] = original_values

        self.body = re.sub(regex, placeholder, self.body)

        return self.body

    def remove_footer(self, footer: str):
        self.body = re.sub(footer, "", self.body)
        return self.body

    def to_db_entry(self) -> EmailMessageEntry:
        db_entry = {
            "_id": ObjectId(self.id),
            "is_main": self.is_main,
            "headers": self.headers,
            "body": self.body,
        }
        if self.special_tokens is not None:
            db_entry["special_tokens"] = self.special_tokens

        if self.disclaimer is not None:
            db_entry["disclaimer"] = self.disclaimer

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
