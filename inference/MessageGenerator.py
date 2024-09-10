from pydantic import BaseModel, computed_field
from typing import Literal, Optional, Union, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch
import outlines
from inference.prompt_generation.generate_prompt import (
    generate_prompt,
    OutputMessage,
    PromptOutputPair,
)
from functools import cached_property
from finetuning.sft.classes.FinetunedModel import FinetunedModel


class MessageGenerator(BaseModel):
    finetuned_model: FinetunedModel
    checkpoint: Optional[int] = None

    @computed_field
    @cached_property
    def tokenizer(self) -> Any:
        tokenizer = AutoTokenizer.from_pretrained(self.finetuned_model.base_model_id)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_tokens(
            [
                "<URL>",
                "<ATTACHMENT>",
                "<PHONE>",
                "<DATE>",
                "<EMAIL>",
                "<PER>",
                "<ORG>",
            ],
        )

        return tokenizer

    @computed_field
    @cached_property
    def gen_model(self) -> Any:

        # Assuming load_model is a function that loads a model and tokenizer based on the given parameters
        if self.finetuned_model.quantization is None:
            model = AutoModelForCausalLM.from_pretrained(
                self.finetuned_model.base_model_id,
                device_map="auto",
            )

        else:
            if self.finetuned_model.quantization == "4bit":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    selective_precision={"critical_layers": "8bit"},
                )

                model = AutoModelForCausalLM.from_pretrained(
                    self.finetuned_model.base_model_id, quantization_config=bnb_config
                )

            elif self.finetuned_model.quantization == "8bit":
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_quant_type="nf8",
                    bnb_8bit_compute_dtype=torch.float16,
                    selective_precision={"critical_layers": "8bit"},
                )

                model = AutoModelForCausalLM.from_pretrained(
                    self.finetuned_model.base_model_id, quantization_config=bnb_config
                )

            else:
                raise ValueError(
                    f"Quantization type {self.finetuned_model.quantization} is not supported"
                )

        model.resize_token_embeddings(len(self.tokenizer))

        if self.checkpoint:

            model = PeftModel.from_pretrained(
                model,
                f"finetuning/sft/models/{self.finetuned_model.base_model_id.split('/')[-1]}/{self.finetuned_model.timestamp}/checkpoint-{self.checkpoint}",
            )

        return model

    def generate_message(
        self,
        subject: str,
        attachments: Optional[bool],
        sentiment: Optional[list],
        urls: Optional[bool],
        guided: Union[Literal["both"], bool] = False,
    ):

        prompt = (
            generate_prompt(
                subject=subject, attachments=attachments, urls=urls, sentiment=sentiment
            )
            + "\n->\n"
        )

        if guided == "both" or not guided:
            input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(
                "cuda"
            )
            attention_mask = torch.ones_like(input_ids).to("cuda")

            output_ids = self.gen_model.generate(
                input_ids,
                max_length=192,
                num_return_sequences=1,
                top_k=50,
                top_p=0.95,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=0.5,
            )

            output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            # get finetuned_model_checkpoint
            checkpoint = self.finetuned_model.get_checkpoint(self.checkpoint)
            try:
                checkpoint.add_message(
                    urls=urls,
                    attachments=attachments,
                    sentiment=sentiment,
                    subject=subject,
                    body=output_text.split("\n->\nbody: ")[1].strip(),
                    save=True,
                )
            except IndexError:
                checkpoint.add_message(
                    urls=urls,
                    attachments=attachments,
                    sentiment=sentiment,
                    subject=subject,
                    body=output_text,
                    save=True,
                )

            return output_text

        if guided:
            model = outlines.models.Transformers(self.gen_model, self.tokenizer)
            generator = outlines.generate.json(model, OutputMessage)
            message = generator([prompt])
            output_text = message
            print(output_text)
