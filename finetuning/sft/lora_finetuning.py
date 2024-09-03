# import libraries
import torch
import bitsandbytes as bnb
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model
from typing import Union, Literal
from datasets import load_from_disk
import json
import os


def load_dataset(path: str, tokenizer):

    data = load_from_disk(path)

    data = data.map(
        lambda samples: tokenizer(
            samples["text"], padding="max_length", max_length=512, truncation=True
        ),
    ).shuffle()

    return data


class LossLoggerCallback(TrainerCallback):
    def __init__(self, log_file):
        self.log_file = log_file
        self.losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            self.losses.append({"step": state.global_step, "loss": logs["loss"]})
            with open(self.log_file, "w") as f:
                json.dump(self.losses, f)


def initialize_new_embeddings(model, tokenizer, related_tokens_dict, custom_tokens=[]):
    print(related_tokens_dict)
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


def train_lora(
    dataset_path: str,
    model_id: str,
    quantized: Union[Literal["4bit", "8bit"], None] = None,
    epochs: int = 2,
    output_dir: str = "offline_finetuning/models/gemma-2b/1724075462",
    custom_tokens: list = [
        "<URL>",
        "<ATTACHMENT>",
        "<PHONE>",
        "<DATE>",
        "<EMAIL>",
        "<PER>",
        "<ORG>",
    ],
    related_tokens_dict: dict = {
        "<URL>": ["<URL>"],
        "<ATTACHMENT>": ["<ATTACHMENT>"],
        "<PHONE>": ["<PHONE>"],
        "<DATE>": ["<DATE>"],
        "<EMAIL>": ["<EMAIL>"],
        "<PER>": ["<PER>"],
        "<ORG>": ["<ORG>"],
    },
    rank: int = 16,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    resume_from_checkpoint: bool = False,
):
    log_file = (f"{output_dir}/loss_log.json",)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    if quantized is None:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
        )

    else:
        if quantized == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                selective_precision={"critical_layers": "8bit"},
            )

        elif quantized == "8bit":
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_quant_type="nf8",
                bnb_8bit_compute_dtype=torch.float32,
                selective_precision={"critical_layers": "8bit"},
            )

        else:
            raise ValueError("Invalid quantization type")

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
        )
    if custom_tokens:
        if related_tokens_dict:
            initialize_new_embeddings(model, tokenizer, related_tokens_dict)

        else:
            tokenizer.add_tokens(custom_tokens)
            model.resize_token_embeddings(len(tokenizer))

    data = load_dataset(dataset_path, tokenizer)

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

    config = LoraConfig(task_type="CAUSAL_LM", r=rank)
    model = get_peft_model(model, config)

    # model training
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8
    )
    args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=epochs,
        save_steps=500,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=1,
        output_dir=output_dir,
    )

    trainer = Trainer(
        model=model,
        train_dataset=data,
        args=args,
        data_collator=data_collator,
        callbacks=[LossLoggerCallback(log_file)],
    )

    trainer.train(
        resume_from_checkpoint=resume_from_checkpoint,
    )
    model.save_pretrained(output_dir)


"""
train_lora(
    dataset_path="offline_finetuning/datasets/pytorch/enron",
    model_id="google/gemma-2b",
    quantized="4bit",
    output_dir="offline_finetuning/models/gemma-2b",
)
"""
