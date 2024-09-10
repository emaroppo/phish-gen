import os
import nltk
from pydantic import BaseModel, computed_field
from typing import Any, List
from bertopic import BERTopic
from nltk.corpus import stopwords
from functools import cached_property


class TopicModelling(BaseModel):
    checkpoint_path: str = None
    dataset: List[str] = None
    n_topics: int = 0

    @computed_field
    @cached_property
    def topic_model(self) -> Any:
        if self.checkpoint_path is not None:
            topic_model = BERTopic.load(
                self.checkpoint_path, embedding_model="all-MiniLM-L6-v2"
            )
        elif self.dataset is not None:
            print("No checkpoint path provided, training new model.")

            topic_model = BERTopic(
                language="english",
                calculate_probabilities=False,
                verbose=True,
                embedding_model="all-MiniLM-L6-v2",
            )
            topic_model.fit_transform(self.dataset)
            if self.n_topics > 0:
                topic_model.reduce_topics(self.dataset, n_topics=self.n_topics)
            topic_model.save(
                "data/processing/labellers/topic_modelling/models/topic_model",
                serialization="safetensors",
            )
            topic_model = BERTopic.load(
                "data/processing/labellers/topic_modelling/models/topic_model",
                embedding_model="all-MiniLM-L6-v2",
            )
        else:
            raise ValueError("No dataset or checkpoint path provided.")

        return topic_model

    def predict_topic(self, text: str):
        topic = self.topic_model.find_topics(text, top_n=1)[0]
        topic = [self.topic_model.get_topic(i) for i in topic]
        topic = [i[0] for i in topic[0]]
        return topic
