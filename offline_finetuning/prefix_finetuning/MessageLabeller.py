from pydantic import BaseModel
from typing import Tuple, List, Dict
from offline_finetuning.common_classes.QueryManager import query_manager
from common_classes.EmailThread import (
    EmailThread,
)  # replace with dataset factory method once i figure out how to keep track of object ids
from transformers import BertTokenizer, BertForSequenceClassification


class MessageLabeller:
    model: str
    data_sources: Tuple[str, List[str]]
    label_field: str


def label_dataset(self, label_field: str):
    # initialize the model
    self.model = BertForSequenceClassification.from_pretrained(self.model)
    tokenizer = BertTokenizer.from_pretrained(self.model)

    for db_name, collections in self.data_sources:
        for collection in collections:
            threads = query_manager.connection[db_name][collection].find()
            threads = [EmailThread.deserialize(thread) for thread in threads]
            for thread in threads:
                for message in thread.messages:

                    # tokenize the message
                    tokenized_message = tokenizer(
                        message.body,
                        padding="max_length",
                        truncation=True,
                        max_length=512,
                    )

                    # predict the label
                    message.__dict__[label_field] = self.model.predict(
                        tokenized_message
                    )
                thread.save()
