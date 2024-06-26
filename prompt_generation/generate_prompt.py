from pydantic import BaseModel
from typing import List, Optional


# TODO: check whether attachments and URLS fields are always present in datataset
class Prompt(BaseModel):
    subject: str
    attachments: Optional[bool]
    urls: Optional[bool]


class OutputMessage(BaseModel):
    body: str
    # attachments: Optional[List[str]]
    # urls: Optional[List[str]]


class PromptOutputPair(BaseModel):

    prompt: Prompt
    output_message: OutputMessage


def generate_prompt(subject: str, attachments: bool = False, urls: bool = False) -> str:
    prompt = dict()
    prompt["subject"] = subject
    if urls:
        prompt["urls"] = urls
    if attachments:
        prompt["attachments"] = attachments

    return str(prompt)


def generate_target_value(body):
    return str({"body": body})


def generate_prompt_output_pair(
    body: str,
    subject: str,
    attachments: bool = False,
    urls: bool = False,
) -> str:
    prompt = generate_prompt(subject, attachments, urls)
    target_value = generate_target_value(body)
    return f"{prompt}->{target_value}"
