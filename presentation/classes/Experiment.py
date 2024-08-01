from pydantic import BaseModel
from presentation.classes.FinetunedModel import FinetunedModel
from presentation.classes.FinetuningDataset import FinetuningDataset
import offline_finetuning.training.lora_finetuning as finetune
from datetime import datetime
import os
from bson import ObjectId
from typing import List
from inference.MessageGenerator import MessageGenerator
from offline_finetuning.auto_labelling.SentimentMessageLabeller import (
    SentimentMessageLabeller,
)
from tqdm import tqdm
from offline_finetuning.common_classes.QueryManager import query_manager


class Experiment(BaseModel):
    finetuned_model: FinetunedModel
    finetuning_dataset: FinetuningDataset

    @classmethod
    def from_db(cls, model_id: str, model_timestamp: int, dataset_timestamp: int):
        finetuned_model = FinetunedModel.from_db(
            model_id, model_timestamp, include_messages=False
        )
        finetuning_dataset = FinetuningDataset.from_db(
            dataset_timestamp, include_messages=False
        )
        return cls(
            finetuned_model=finetuned_model, finetuning_dataset=finetuning_dataset
        )

    @classmethod
    def run_experiment(
        cls,
        dataset_timestamp,
        model_id,
        rank,
        quantization,
        batch_size,
        gradient_accumulation_steps,
        learning_rate,
    ):
        model_timestamp = int(datetime.now().timestamp())
        # Load dataset
        dataset_path = f"offline_finetuning/datasets/transformers/{dataset_timestamp}"
        finetuned_model = FinetunedModel(
            timestamp=model_timestamp,
            dataset_timestamp=dataset_timestamp,
            base_model_id=model_id,
            rank=rank,
            quantization=quantization,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
        )

        model = finetune.train_lora(
            dataset_path=dataset_path,
            output_dir=f"offline_finetuning/models/{model_id}/{model_timestamp}",
            model_id=model_id,
            quantized=quantization,
            rank=rank,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
        )

        for checkpoint in os.listdir(
            f"offline_finetuning/models/{model_id.split('/')[-1]}/{model_timestamp}"
        ):
            if os.path.isdir(
                f"offline_finetuning/models/{model_id.split('/')[-1]}/{model_timestamp}/{checkpoint}"
            ):
                n_steps = int(checkpoint.split("-")[1])
                finetuned_model.add_checkpoint(
                    timestamp=model_timestamp,
                    base_model_id=model_id,
                    steps=n_steps,
                )

        dataset = FinetuningDataset.from_db(dataset_timestamp)
        finetuned_model.save()
        return cls(
            finetuned_model=finetuned_model,
            finetuning_dataset=dataset,
        )

    @classmethod
    def generate_evaluation_prompts(
        cls,
        subjects: List[str] = [
            "Payroll Update",
            "Christmas Office Party",
            "Outstanding Invoice",
        ],
        sentiments: List[str] = [
            "sadness",
            "joy",
            "neutral",
            "anger",
            "fear",
            "surprise",
            "disgust",
        ],
    ):
        sample_args = list()
        for subject in subjects:
            for sentiment in sentiments:
                for url in [True, False]:
                    for attachment in [True, False]:
                        sample_arg = {
                            "subject": subject,
                            "attachments": attachment,
                            "urls": url,
                            "sentiment": [sentiment],
                        }
                        sample_args.append(sample_arg)
        return sample_args

    def generate_evaluation_messages(self, checkpoint):
        generator = MessageGenerator(
            finetuned_model=self.finetuned_model,
            checkpoint=checkpoint,
        )
        prompts = self.generate_evaluation_prompts()
        for prompt in tqdm(prompts, desc="Generating messages"):
            generator.generate_message(**prompt)

    def evaluate_model_outputs(self):
        for checkpoint in tqdm(
            self.finetuned_model.checkpoints, desc="Evaluating checkpoints"
        ):
            self.generate_evaluation_messages(checkpoint.steps)

            sentiment_labeller = SentimentMessageLabeller(
                classifier_id="michellejieli/emotion_text_classifier",
                task="sentiment-analysis",
            )
            print("Labelling messages & updating metrics")
            checkpoint.label_output_messages(sentiment_labeler=sentiment_labeller)
            checkpoint.update_metrics()
            self.finetuned_model.save()
            for message in checkpoint.messages:
                print(message.serialise())
                query_manager.connection["models"][
                    f"outputs_{self.finetuned_model.base_model_id.split('/')[-1]}_{self.finetuned_model.timestamp}_{checkpoint.steps}"
                ].update_one(
                    {"_id": ObjectId(message.message_id)},
                    {"$set": message.serialise()},
                    upsert=True,
                )
                print(
                    f"{self.finetuned_model.base_model_id.split('/')[-1]}_{self.finetuned_model.timestamp}_{checkpoint.steps} saved"
                )
