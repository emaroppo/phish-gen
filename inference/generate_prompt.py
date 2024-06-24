from pydantic import BaseModel
from typing import List, Optional


# TODO: check whether attachments and URLS fields are always present in datataset
class Prompt(BaseModel):
    SUBJECT: str
    ATTACHMENTS: Optional[bool]
    URLS: Optional[bool]


class ReturnMessage(BaseModel):
    BODY: str
    ATTACHMENTS: Optional[List[str]]
    URLS: Optional[List[str]]


class ReturnValue(BaseModel):

    prompt: Prompt
    return_value: ReturnMessage


def generate_prompt(subject: str, attachments: bool, urls: Optional[bool]):
    prompt = dict()
    prompt["SUBJECT"] = subject
    if urls:
        prompt["URLS"] = urls
    if attachments:
        prompt["ATTACHMENTS"] = attachments

    return str(prompt)
