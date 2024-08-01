from pydantic import BaseModel
from typing import List, Dict, Optional
from presentation.classes.FinetunedCheckpoint import FinetunedCheckpoint
from offline_finetuning.common_classes.QueryManager import query_manager


class FinetunedModel(BaseModel):
    timestamp: int
    dataset_timestamp: int
    base_model_id: str
    rank: int
    quantization: str
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    checkpoints: Optional[List[FinetunedCheckpoint]] = list()

    @classmethod
    def deserialize(cls, data):
        return cls(
            timestamp=data["timestamp"],
            dataset_timestamp=data["dataset_timestamp"],
            base_model_id=data["base_model_id"],
            rank=data["rank"],
            quantization=data["quantization"],
            batch_size=data["batch_size"],
            gradient_accumulation_steps=data["gradient_accumulation_steps"],
            learning_rate=data["learning_rate"],
            checkpoints=[
                FinetunedCheckpoint.deserialize(data=checkpoint, include_messages=False)
                for checkpoint in data["checkpoints"]
            ],
        )

    @classmethod
    def from_db(cls, base_model_id, timestamp):
        # Load model from database
        finetuned_model = query_manager.connection["models"]["summary"].find_one(
            {"base_model_id": base_model_id, "timestamp": timestamp}
        )
        return cls.deserialize(finetuned_model)

    def serialise(self):
        return {
            "timestamp": self.timestamp,
            "dataset_timestamp": self.dataset_timestamp,
            "base_model_id": self.base_model_id,
            "rank": self.rank,
            "quantization": self.quantization,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "checkpoints": [checkpoint.serialise() for checkpoint in self.checkpoints],
        }

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

    def save(self):
        query_manager.connection["models"]["summary"].update_one(
            {"timestamp": self.timestamp, "base_model_id": self.base_model_id},
            {"$set": self.serialise()},
            upsert=True,
        )
        return True
