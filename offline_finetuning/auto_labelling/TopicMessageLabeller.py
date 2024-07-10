from offline_finetuning.auto_labelling.topic_modelling.TopicModelling import (
    TopicModelling,
)
from functools import cached_property
from offline_finetuning.auto_labelling.MessageLabeller import MessageLabeller
from pydantic import computed_field
from typing import Any


class TopicMessageLabeller(MessageLabeller):
    classifier_id: str = (
        "offline_finetuning/auto_labelling/topic_modelling/models/topic_model"
    )

    @computed_field
    @cached_property
    def topic_model(self) -> Any:
        return TopicModelling(checkpoint_path=self.classifier_id)

    def label_message(self, message_body: str):
        return self.topic_model.topic_model.get_topic(
            self.topic_model.predict_topic(message_body)[0][0]
        )
