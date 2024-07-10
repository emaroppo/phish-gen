from typing import Any
from offline_finetuning.auto_labelling.MessageLabeller import MessageLabeller
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
from pydantic import computed_field
import torch
from functools import cached_property


class NERMessageLabeller(MessageLabeller):

    @computed_field
    @cached_property
    def classifier_model(self) -> Any:
        return AutoModelForTokenClassification.from_pretrained(self.classifier_id)

    @computed_field
    @cached_property
    def tokenizer(self) -> Any:
        return AutoTokenizer.from_pretrained(self.classifier_id)

    @computed_field
    @cached_property
    def classifier(self) -> Any:
        device = 0 if torch.cuda.is_available() else -1
        return pipeline(
            task="ner",
            model=self.classifier_model,
            tokenizer=self.tokenizer,
            device=device,
        )

    def label_message(self, message_body: str):
        labels = self.classifier(message_body)

        if labels:
            # compose the tokens back into word
            words = list()
            first_label = labels.pop(0)
            word_start = first_label["start"]
            word_end = first_label["end"]
            word_label = first_label["entity"].split("-")[1]

            for label in labels:

                if label["entity"][0] == "I":
                    word_end = label["end"]
                elif label["entity"][0] == "B":
                    words.append((word_start, word_end, word_label))
                    word_start = label["start"]
                    word_end = label["end"]
                    word_label = label["entity"].split("-")[1]

            return words
