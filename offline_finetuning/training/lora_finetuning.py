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
)
from peft import LoraConfig, get_peft_model
from typing import Union, Literal
from datasets import load_from_disk


def load_dataset(path: str, tokenizer):

    data = load_from_disk(path)

    data = data.map(
        lambda samples: tokenizer(
            samples["text"], padding="max_length", max_length=512, truncation=True
        ),
    ).shuffle()

    return data


def train_lora(
    dataset_path: str,
    model_id: str,
    quantized: Union[Literal["4bit", "8bit"], None] = None,
    output_dir: str = "offline_finetuning/models/{model_id}",
):

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
                bnb_4bit_compute_dtype=torch.float32,
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

    data = load_dataset(dataset_path, tokenizer)

    # freeze model weights
    for param in model.parameters():
        param.requires_grad = False
        if param.ndim == 1:
            param.data.to(torch.float32)

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # TO DO: double check if required when not quantizing
    class CastOutputToFloat(torch.nn.Sequential):
        def forward(self, x):
            return super().forward(x).to(torch.float32)

    model.lm_head = CastOutputToFloat(model.lm_head)

    config = LoraConfig(task_type="CAUSAL_LM", r=16)
    model = get_peft_model(model, config)

    # model training
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8
    )
    args = TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=2,
        save_steps=200,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir=output_dir,
    )

    trainer = Trainer(
        model=model, train_dataset=data, args=args, data_collator=data_collator
    )

    trainer.train()
    model.save_pretrained(output_dir)


train_lora(
    dataset_path="offline_finetuning/datasets/pytorch/enron",
    model_id="gpt2",
    quantized=None,
    output_dir="offline_finetuning/models/gpt2",
)
