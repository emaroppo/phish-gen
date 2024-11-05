from pydantic import BaseModel
from typing import List, Optional, Union, Literal
from finetuning.sft.classes.FinetunedCheckpoint import FinetunedCheckpoint
from data.QueryManager import query_manager


class FinetuningArguments(BaseModel):
    rank: int = 16
    quantization: Union[Literal["4bit", "8bit"], None]
    batch_size: int = 2
    gradient_accumulation_steps: int = 16
    learning_rate: float
    warmup_steps: int = 500
    max_seq_length: int = 512
    weight_decay: float = 0.01
    lr_scheduler: Literal["linear", "cosine"] = "linear"

    @classmethod
    def deserialize(cls, data):

        init_arguments = {i: data[i] for i in data if i in cls.model_fields}

        return cls(**init_arguments)


class FinetunedModel(BaseModel):
    timestamp: int
    dataset_timestamp: int
    base_model_id: str
    fine_tuning_arguments: Optional[FinetuningArguments] = None
    checkpoints: Optional[List[FinetunedCheckpoint]] = list()
    loss_history: Optional[List] = list()

    @classmethod
    def deserialize(cls, data, include_messages=False):
        model_arguments = {
            "timestamp": data["timestamp"],
            "dataset_timestamp": data["dataset_timestamp"],
            "base_model_id": data["base_model_id"],
            "checkpoints": [
                FinetunedCheckpoint.deserialize(
                    data=checkpoint, include_messages=include_messages
                )
                for checkpoint in data["checkpoints"]
            ],
        }

        model_arguments["fine_tuning_arguments"] = FinetuningArguments.deserialize(data)

        return cls(
            **model_arguments,
        )

    @classmethod
    def from_db(cls, base_model_id, timestamp, include_messages=False):
        # Load model from database
        finetuned_model = query_manager.connection["models"]["summary"].find_one(
            {"base_model_id": base_model_id, "timestamp": timestamp}
        )
        return cls.deserialize(finetuned_model, include_messages)

    @classmethod
    def initialize_new_embeddings(
        cls, model, tokenizer, related_tokens_dict, custom_tokens=[], unk_token_id=3
    ):
        raise NotImplementedError

    @classmethod
    def train_model(
        cls, dataset, base_model_id, fine_tuning_arguments, epochs, save_steps
    ):
        raise NotImplementedError

    def resume_training(self, epochs):
        raise NotImplementedError

    def add_checkpoint(self, timestamp, base_model_id, steps):
        checkpoint = FinetunedCheckpoint(
            timestamp=timestamp, base_model_id=base_model_id, steps=steps
        )
        self.checkpoints.append(checkpoint)

    def get_checkpoint(self, steps):
        for checkpoint in self.checkpoints:
            if checkpoint.steps == steps:
                return checkpoint
        return None

    def serialise(self):
        return {
            "timestamp": self.timestamp,
            "dataset_timestamp": self.dataset_timestamp,
            "base_model_id": self.base_model_id,
            "rank": self.fine_tuning_arguments.rank,
            "quantization": self.fine_tuning_arguments.quantization,
            "batch_size": self.fine_tuning_arguments.batch_size,
            "gradient_accumulation_steps": self.fine_tuning_arguments.gradient_accumulation_steps,
            "learning_rate": self.fine_tuning_arguments.learning_rate,
            "checkpoints": [checkpoint.serialise() for checkpoint in self.checkpoints],
            "loss_history": self.loss_history,
            "warmup_steps": self.fine_tuning_arguments.warmup_steps,
            "max_seq_length": self.fine_tuning_arguments.max_seq_length,
            "weight_decay": self.fine_tuning_arguments.weight_decay,
            "lr_scheduler": self.fine_tuning_arguments.lr_scheduler,
        }

    def save(self):
        query_manager.connection["models"]["summary"].update_one(
            {"timestamp": self.timestamp, "base_model_id": self.base_model_id},
            {"$set": self.serialise()},
            upsert=True,
        )
        return True
