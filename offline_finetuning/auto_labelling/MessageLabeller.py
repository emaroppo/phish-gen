from pydantic import BaseModel, computed_field
from typing import Any
from transformers import pipeline
import torch


class MessageLabeller(BaseModel):
    classifier_id: str
    task: str
    no_top_k: bool = False

    @computed_field
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
