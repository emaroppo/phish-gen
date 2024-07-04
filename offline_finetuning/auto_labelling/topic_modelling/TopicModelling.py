import os
import nltk
from pydantic import BaseModel, computed_field
from typing import Any, List
from bertopic import BERTopic
from nltk.corpus import stopwords


class TopicModelling(BaseModel):
    checkpoint_path: str = None
    dataset: List[str] = None

    @computed_field
    def topic_model(self) -> Any:
        if self.checkpoint_path is not None:
            topic_model = BERTopic.load(self.checkpoint_path)
        elif self.dataset is not None:
            print("No checkpoint path provided, training new model.")

            topic_model = BERTopic(
                language="english",
                calculate_probabilities=False,
                verbose=True,
                checkpoint_path=self.checkpoint_path,
            )
            topic_model.fit_transform(self.dataset)
            topic_model.save(
                "offline_finetuning/auto_labelling/topic_modelling/models/"
            )
        else:
            raise ValueError("No dataset or checkpoint path provided.")

        return topic_model

    def predict_topic(self, text: str):
        return self.topic_model.transform([text])
