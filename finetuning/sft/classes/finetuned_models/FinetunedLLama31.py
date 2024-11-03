from finetuning.sft.classes.finetuned_models.FinetunedModel import (
    FinetunedModel,
    FinetuningArguments,
)
from typing import Literal
from finetuning.sft.classes.FinetuningDataset import FinetuningDataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import load_from_disk
from transformers import TrainingArguments
from unsloth import FastLanguageModel, is_bfloat16_supported
from datetime import datetime


class FinetunedLLama31(FinetunedModel):
    base_model_id: Literal["unsloth/Meta-Llama-3.1-8B-bnb-4bit"]

    @classmethod
    def train_model(
        cls,
        dataset: FinetuningDataset,
        fine_tuning_arguments: FinetuningArguments,
        epochs: int = 2,
        save_steps: int = 500,
    ):
        timestamp = int(datetime.now().timestamp())
        base_model_id = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"

        finetuned_model = cls(
            base_model_id=base_model_id,
            dataset_timestamp=dataset.timestamp,
            training_args=fine_tuning_arguments,
            dataset=dataset,
            timestamp=timestamp,
            fine_tuning_arguments=fine_tuning_arguments,
        )

        finetuned_model.save()

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model_id,
            max_seq_length=fine_tuning_arguments.max_seq_length,
            load_in_4bit=True,
            dtype=None,
        )

        # change the padding tokenizer value
        tokenizer.add_special_tokens({"pad_token": "<|reserved_special_token_0|>"})
        model.config.pad_token_id = tokenizer.pad_token_id  # updating model config
        tokenizer.padding_side = (
            "right"  # padding to right (otherwise SFTTrainer shows warning)
        )

        # add eos token at the end of the samples

        def add_eos_token(example):
            example["text"] = example["text"] + tokenizer.eos_token
            return example

        dataset = load_from_disk(
            f"data/datasets_processed/training/{dataset.timestamp}"
        )
        dataset = dataset.map(add_eos_token)
        dataset = dataset.shuffle()

        response_template = "\n->\n"
        collator = DataCollatorForCompletionOnlyLM(
            tokenizer=tokenizer, response_template=response_template
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=fine_tuning_arguments.rank,
            lora_alpha=16,
            lora_dropout=0,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "up_proj",
                "down_proj",
                "o_proj",
                "gate_proj",
            ],
            use_rslora=True,
            use_gradient_checkpointing="unsloth",
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=fine_tuning_arguments.max_seq_length,
            data_collator=collator,
            args=TrainingArguments(
                learning_rate=fine_tuning_arguments.learning_rate,
                lr_scheduler_type=fine_tuning_arguments.lr_scheduler,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=fine_tuning_arguments.gradient_accumulation_steps,
                num_train_epochs=epochs,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=fine_tuning_arguments.weight_decay,
                warmup_steps=fine_tuning_arguments.warmup_steps,
                output_dir=f"fine_tuning/sft/models/{base_model_id.split('/')[-1]}/{timestamp}",
                save_steps=save_steps,
            ),
        )

        trainer.train()
        return finetuned_model

    def resume_training(self, epochs):
        raise NotImplementedError
