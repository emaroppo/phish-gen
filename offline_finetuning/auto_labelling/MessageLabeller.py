from pydantic import BaseModel, computed_field
from typing import Any
from transformers import pipeline


class MessageLabeller(BaseModel):
    classifier_id: str
    task: str
    no_top_k: bool = False

    @computed_field
    def classifier(self) -> Any:
        if self.no_top_k:
            classifier = pipeline(task=self.task, model=self.classifier_id, top_k=None)
        else:
            classifier = pipeline(task=self.task, model=self.classifier_id)
        return classifier

    def label_message(self, message_body: str):
        return self.classifier(message_body)
