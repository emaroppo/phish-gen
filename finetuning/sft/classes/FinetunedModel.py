from pydantic import BaseModel, computed_field
from functools import cached_property
from typing import List, Dict, Optional, Union, Literal
from finetuning.sft.classes.FinetunedCheckpoint import FinetunedCheckpoint
from data.QueryManager import query_manager
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from transformers.trainer_callback import TrainerCallback
import torch
import json
from finetuning.sft.classes.FinetuningDataset import FinetuningDataset


class LossLoggerCallback(TrainerCallback):
    def __init__(self, finetuned_model, log_file):
        self.finetuned_model = finetuned_model
        self.log_file = log_file
        self.losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            self.losses.append({"step": state.global_step, "loss": logs["loss"]})
            self.finetuned_model.loss_history.append(logs["loss"])
            if state.global_step % 100 == 0:
                self.finetuned_model.save()
            with open(self.log_file, "w") as f:
                json.dump(self.losses, f)


class CheckpointCallback(TrainerCallback):
    def __init__(self, finetuned_model):
        self.finetuned_model = finetuned_model

    def on_save(self, args, state, control, **kwargs):
        timestamp = int(datetime.now().timestamp())
        steps = state.global_step
        self.finetuned_model.add_checkpoint(
            timestamp, self.finetuned_model.base_model_id, steps
        )
        self.finetuned_model.save()


class CastOutputToFloat(torch.nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


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
    custom_tokens: Optional[List[str]] = None
    related_tokens_dict: Optional[Dict[str, List[str]]] = None
    loss_history: Optional[List] = list()

    @computed_field
    @cached_property
    def bnb_config(self) -> Optional[BitsAndBytesConfig]:
        if self.quantization == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
            )
        elif self.quantization == "8bit":
            return BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_quant_type="nf8",
            )
        else:
            return None

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

    @classmethod
    def initialize_new_embeddings(
        model, tokenizer, related_tokens_dict, custom_tokens=[]
    ):
        # Get the embedding layer
        embedding_layer = model.get_input_embeddings()

        # Initialize embeddings for custom tokens
        for token in custom_tokens:
            related_tokens = related_tokens_dict.get(token, [])
            if related_tokens:
                related_token_ids = [
                    tokenizer.convert_tokens_to_ids(rt) for rt in related_tokens
                ]
                related_embeddings = embedding_layer.weight.data[related_token_ids]
                average_embedding = related_embeddings.mean(dim=0)
            else:
                # If no related tokens, initialize with random embeddings
                average_embedding = torch.randn(embedding_layer.embedding_dim)

            # Add the new token to the tokenizer and set its embedding
            tokenizer.add_tokens([token])
            token_id = tokenizer.convert_tokens_to_ids(token)
            embedding_layer.weight.data[token_id] = average_embedding

    @classmethod
    def train_model(
        cls,
        dataset: FinetuningDataset,
        base_model_id: str,
        epochs: int = 2,
        rank: int = 16,
        quantization: Union[Literal["4bit", "8bit"], None] = "4bit",
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-4,
        custom_tokens: Union[List[str], Dict[str, List[str]], None] = None,
        save_steps: int = 500,
    ):
        if type(custom_tokens) == dict:
            custom_tokens_dict = custom_tokens
            custom_tokens = custom_tokens_dict.keys()
        # Create a new finetuned model entry in the database
        timestamp = int(datetime.now().timestamp())
        finetuned_model = FinetunedModel(
            timestamp=timestamp,
            dataset_timestamp=dataset.timestamp,
            base_model_id=base_model_id,
            rank=rank,
            quantization=quantization,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            custom_tokens=custom_tokens,
        )
        finetuned_model.save()

        # Load the base model
        if finetuned_model.quantization:
            bnb_config = finetuned_model.bnb_config
            print(bnb_config)
            model = AutoModelForCausalLM.from_pretrained(
                base_model_id, quantization_config=bnb_config, device_map="auto"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model_id, device_map="auto"
            )

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        tokenizer.pad_token = tokenizer.eos_token

        if custom_tokens:
            if type(custom_tokens) == list:
                tokenizer.add_tokens(custom_tokens)
                model.resize_token_embeddings(len(tokenizer))
            elif type(custom_tokens) == dict:
                FinetunedModel.initialize_new_embeddings(
                    model,
                    tokenizer,
                    custom_tokens=custom_tokens,
                    related_tokens_dict=custom_tokens_dict,
                )

        # Load dataset
        dataset = dataset.load_finetuning_dataset(tokenizer=tokenizer)

        for param in model.parameters():
            param.requires_grad = False
            if param.ndim == 1:
                param.data.to(torch.float32)

        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

        model.lm_head = CastOutputToFloat(model.lm_head)

        config = LoraConfig(
            task_type="CAUSAL_LM",
            r=rank,
        )
        model = get_peft_model(model, config)

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8
        )

        args = TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=epochs,
            save_steps=save_steps,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=1,
            output_dir=f"finetuning/sft/models/{base_model_id.split('/')[-1]}/{timestamp}",
        )

        callbacks = [
            LossLoggerCallback(
                finetuned_model,
                f"finetuning/sft/models/{base_model_id.split('/')[-1]}/{timestamp}/losses.json",
            ),
            CheckpointCallback(finetuned_model),
        ]

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=dataset,
            data_collator=data_collator,
            callbacks=callbacks,
        )

        trainer.train()
        model.save_pretrained(
            f"finetuning/sft/models/{base_model_id.split('/')[-1]}/{timestamp}"
        )
        return finetuned_model

    def resume_training(self, epochs):
        # load model from output_dir
        output_dir = f"finetuning/sft/models/{self.base_model_id.split('/')[-1]}/{self.timestamp}"
        model = AutoModelForCausalLM.from_pretrained(output_dir)
        tokenizer = AutoTokenizer.from_pretrained(output_dir)

        dataset = FinetuningDataset.load_from_timestamp(self.dataset_timestamp)

        dataset = dataset.load_finetuning_dataset(tokenizer)

        # freeze model weights
        for param in model.parameters():
            param.requires_grad = False
            if param.ndim == 1:
                param.data.to(torch.float32)

        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

        class CastOutputToFloat(torch.nn.Sequential):
            def forward(self, x):
                return super().forward(x).to(torch.float32)

        model.lm_head = CastOutputToFloat(model.lm_head)

        config = LoraConfig(task_type="CAUSAL_LM", r=self.rank)
        model = get_peft_model(model, config)

        # model training
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8
        )
        args = TrainingArguments(
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            num_train_epochs=epochs,
            save_steps=500,
            learning_rate=self.learning_rate,
            fp16=True,
            logging_steps=1,
            output_dir=output_dir,
        )

        trainer = Trainer(
            model=model,
            train_dataset=dataset,
            args=args,
            data_collator=data_collator,
            callbacks=[
                LossLoggerCallback(
                    f"finetuning/sft/models/{self.base_model_id.split('/')[-1]}/{self.timestamp}/losses.json"
                ),
                CheckpointCallback(self),
            ],
        )

        trainer.train(
            resume_from_checkpoint=True,
        )
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        self.save()

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
