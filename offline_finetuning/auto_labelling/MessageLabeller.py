from pydantic import BaseModel, computed_field
from functools import cached_property
from typing import Any
from transformers import pipeline
import torch
import pandas as pd


class MessageLabeller(BaseModel):
    classifier_id: str
    task: str
    no_top_k: bool = False

    @computed_field
    @cached_property
    def classifier(self) -> Any:
        # TODO: find a way to process the message in batches
        # Check if GPU is available

        device = 0 if torch.cuda.is_available() else -1

        if self.no_top_k:
            classifier = pipeline(
                task=self.task, model=self.classifier_id, top_k=None, device=device
            )
        else:
            classifier = pipeline(
                task=self.task, model=self.classifier_id, device=device
            )
        return classifier

    def label_message(self, message_body: str):
        return self.classifier(message_body)

    def generate_validation_excel(self, unique_values, file_path):
        df = pd.DataFrame(unique_values, columns=["value"])
        df.to_excel(file_path, index=False)
        return file_path

    def validate_custom_tokens(self, urls: bool, attachments: bool, message_body):
        return ("<URL>" in message_body) == urls, (
            "<ATTACHMENT>" in message_body
        ) == attachments
