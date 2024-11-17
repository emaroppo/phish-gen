from pydantic import BaseModel
from typing import List, Dict, Optional
from bson import ObjectId
from finetuning.sft.classes.ModelOutput import ModelOutput
from data.QueryManager import query_manager
import pandas as pd

# import message labelers


class FinetunedCheckpoint(BaseModel):
    timestamp: int
    base_model_id: str
    steps: int
    message_count: Optional[int] = 0
    messages: Optional[List[ModelOutput]] = list()
    sentiment_metrics: Optional[Dict[str, float]] = dict()
    url_metrics: Optional[Dict[str, float]] = dict()
    attachment_metrics: Optional[Dict[str, float]] = dict()
    loss: Optional[float] = None

    @classmethod
    def deserialize(cls, data, include_messages=False):

        if "urls" in data["attachment_metrics"]:
            # drop urls from attachment metrics
            data["attachment_metrics"].pop("urls")
            print(data["attachment_metrics"])
        if "attachments" in data["attachment_metrics"]:
            # drop attachments from attachment metrics
            data["attachment_metrics"].pop("attachments")
        checkpoint = cls(
            timestamp=data["timestamp"],
            base_model_id=data["base_model_id"],
            steps=data["steps"],
            messages=list(),
            message_count=data["message_count"],
            sentiment_metrics=data["sentiment_metrics"],
            url_metrics=data["url_metrics"],
            attachment_metrics=data["attachment_metrics"],
        )

        if include_messages:
            checkpoint.get_messages()
        return checkpoint

    @classmethod
    def from_db(cls, base_model_id, timestamp, steps):
        pipeline = [
            {"$match": {"timestamp": timestamp, "base_model_id": base_model_id}},
            {"$unwind": "$checkpoints"},
            {"$match": {"checkpoints.steps": steps}},
            {"$project": {"checkpoints": 1}},
        ]
        checkpoint = query_manager.connection["models"]["summary"].aggregate(pipeline)
        checkpoint = list(checkpoint)[0]["checkpoints"]
        return cls.deserialize(checkpoint, include_messages=True)

    def get_messages(self):
        messages = query_manager.connection["models"][
            f"outputs_{self.base_model_id.split('/')[-1]}_{self.timestamp}_{self.steps}"
        ].find()

        # count = query_manager.connection["models"][
        #    f"outputs_{self.base_model_id.split('/')[-1]}_{self.timestamp}"

        self.messages = [ModelOutput.deserialize(message) for message in messages]

    def serialise(self, include_messages=False):
        serialised = {
            "timestamp": self.timestamp,
            "base_model_id": self.base_model_id,
            "steps": self.steps,
            "message_count": self.message_count,
            "sentiment_metrics": self.sentiment_metrics,
            "url_metrics": self.url_metrics,
            "attachment_metrics": self.attachment_metrics,
        }
        if include_messages:
            serialised["messages"] = [message.serialise() for message in self.messages]
        return serialised

    def add_message(self, urls, attachments, sentiment, subject, body, save=False):
        message = ModelOutput(
            checkpoint=self.steps,
            body=body,
            prompt={
                "urls": urls,
                "attachments": attachments,
                "sentiment": sentiment,
                "subject": subject,
            },
        )

        if save:
            message.message_id = str(
                query_manager.connection["models"][
                    f"outputs_{self.base_model_id.split('/')[-1]}_{self.timestamp}_{self.steps}"
                ]
                .insert_one(message.serialise())
                .inserted_id
            )

        self.messages.append(message)
        self.message_count += 1

    def update_metrics(self):
        message_results = [
            {
                "prompt_urls": message.prompt["urls"],
                "output_urls": message.output["urls"],
                "prompt_attachments": message.prompt["attachments"],
                "output_attachments": message.output["attachments"],
                "prompt_sentiment": message.prompt["sentiment"][0],
                "output_sentiment": message.output["sentiment"][0],
            }
            for message in self.messages
        ]

        message_results_df = pd.DataFrame(message_results)
        self.sentiment_metrics = {
            "accuracy": (
                message_results_df["prompt_sentiment"]
                == message_results_df["output_sentiment"]
            ).mean()
        }
        self.url_metrics = {
            "accuracy": (
                message_results_df["prompt_urls"] == message_results_df["output_urls"]
            ).mean()
        }
        self.attachment_metrics = {
            "accuracy": (
                message_results_df["prompt_attachments"]
                == message_results_df["output_attachments"]
            ).mean()
        }
        self.attachment_metrics["urls"] = {
            "accuracy": (
                message_results_df["prompt_urls"] == message_results_df["output_urls"]
            ).mean()
        }
        self.attachment_metrics["attachments"] = {
            "accuracy": (
                message_results_df["prompt_attachments"]
                == message_results_df["output_attachments"]
            ).mean()
        }

    def label_output_messages(self, sentiment_labeler, save=True):
        # label output sentiment
        message_bodies = [message.body for message in self.messages]
        print(len(message_bodies))
        labels = sentiment_labeler.label_message(message_bodies)
        print(labels)
        for message, label in zip(self.messages, labels):
            message.add_output_sentiment([label["label"]])

        # label messages with entities
        for message in self.messages:
            message.add_output_entities()
            message.calculate_tasks_results()

        if save:
            for message in self.messages:
                print(message.output)
                query_manager.connection["models"][
                    f"outputs_{self.base_model_id.split('/')[-1]}_{self.timestamp}_{self.steps}"
                ].update_one(
                    {"_id": ObjectId(message.message_id)},
                    {"$set": message.serialise(to_db=True)},
                )
