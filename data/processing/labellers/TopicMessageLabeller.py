from data.processing.labellers.topic_modelling.TopicModelling import (
    TopicModelling,
)
from functools import cached_property
from data.processing.labellers.MessageLabeller import MessageLabeller
from pydantic import computed_field
from typing import Any, Optional, List, Union


class TopicMessageLabeller(MessageLabeller):
    classifier_id: str = "bertopic"
    messages: Optional[List[str]] = None
    checkpoint_path: Optional[str] = None
    task: str = "topic_modelling"

    @computed_field
    @cached_property
    def topic_model(self) -> Any:
        if self.checkpoint_path is not None:
            topic_model = TopicModelling(checkpoint_path=self.checkpoint_path)
            return topic_model
        elif self.messages is not None:
            print("No checkpoint path provided, training new model.")
            topic_model = TopicModelling(dataset=self.messages)
            return topic_model
        else:
            raise ValueError("No dataset or checkpoint path provided.")

    def label_message(self, message_body: Union[List[str], str]):
        if type(message_body) == str:
            label = self.topic_model.predict_topic(message_body)
            return label
        if type(messsage_body) == list:
            labels = [
                self.topic_model.predict_topic(message) for message in message_body
            ]
            return labels
