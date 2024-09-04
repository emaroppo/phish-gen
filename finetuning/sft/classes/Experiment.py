from pydantic import BaseModel
from finetuning.sft.classes.FinetunedModel import FinetunedModel
from finetuning.sft.classes.FinetuningDataset import FinetuningDataset
from bson import ObjectId
from typing import List, Dict, Union, Literal
from inference.MessageGenerator import MessageGenerator
from data.processing.labellers.SentimentMessageLabeller import (
    SentimentMessageLabeller,
)
from tqdm import tqdm
from data.QueryManager import query_manager


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
        dataset_timestamp: int,
        base_model_id: str,
        rank: int = 16,
        epochs: int = 2,
        quantization: Union[Literal["4bit", "8bit"], None] = None,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-4,
        custom_tokens: List[str] = [
            "<URL>",
            "<ATTACHMENT>",
            "<PHONE>",
            "<DATE>",
            "<EMAIL>",
            "<PER>",
            "<ORG>",
        ],
        related_tokens_dict: Dict[str, List[str]] = None,
    ):
        dataset = FinetuningDataset.from_db(dataset_timestamp)

        # Load dataset
        finetuned_model = FinetunedModel.train_model(
            dataset=dataset,
            model_id=base_model_id,
            quantized=quantization,
            rank=rank,
            epochs=epochs,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            custom_tokens=custom_tokens,
            related_tokens_dict=related_tokens_dict,
        )

        experiment = cls(
            finetuned_model=finetuned_model,
            finetuning_dataset=dataset,
        )

        experiment.evaluate_model_outputs()

        return experiment

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

    def continue_experiment(self, epochs):
        self.finetuned_model.continue_training(epochs=epochs)
        self.evaluate_model_outputs()

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
