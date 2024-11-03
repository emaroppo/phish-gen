from pydantic import BaseModel
from finetuning.sft.classes.finetuned_models.FinetunedModel import (
    FinetunedModel,
    FinetuningArguments,
)
from finetuning.sft.classes.FinetuningDataset import FinetuningDataset
from bson import ObjectId
from typing import List, Dict, Union, Literal, Type
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
        base_model_class: Type[FinetunedModel],
        fine_tuning_arguments: FinetuningArguments,
        epochs: int = 2,
        custom_tokens: Union[List[str], Dict[str, List[str]]] = [
            "<URL>",
            "<ATTACHMENT>",
            "<PHONE>",
            "<DATE>",
            "<EMAIL>",
            "<PER>",
            "<ORG>",
        ],
    ):
        dataset = FinetuningDataset.from_db(dataset_timestamp)

        finetuned_model = base_model_class.train_model(
            dataset=dataset,
            fine_tuning_arguments=fine_tuning_arguments,
            epochs=epochs,
            save_steps=500,
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
        self.finetuned_model.resume_training(epochs=epochs)
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
