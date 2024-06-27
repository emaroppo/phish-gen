from pydantic import BaseModel, computed_field
from typing import Literal, Optional, Union, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch
import outlines
from prompt_generation.generate_prompt import (
    generate_prompt,
    OutputMessage,
    PromptOutputPair,
)


class MessageGenerator(BaseModel):
    base_model_id: str
    quantized: Optional[Union[Literal["4bit", "8bit"], None]] = None
    adapter: Optional[str] = None

    @computed_field
    def tokenizer(self) -> Any:
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_tokens(["<ATTACHMENT>", "<URL>", "<PHONE>", "<DATE>"])

        return tokenizer

    @computed_field
    def gen_model(self) -> Any:

        # Assuming load_model is a function that loads a model and tokenizer based on the given parameters
        if self.quantized is None:
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model_id,
                device_map="auto",
            )

        else:
            if self.quantized == "4bit":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float32,
                    selective_precision={"critical_layers": "8bit"},
                )

                model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_id, quantization_config=bnb_config
                )

            elif self.quantized == "8bit":
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_quant_type="nf8",
                    bnb_8bit_compute_dtype=torch.float32,
                    selective_precision={"critical_layers": "8bit"},
                )

                model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_id, quantization_config=bnb_config
                )

            else:
                raise ValueError(f"Quantization type {self.quantized} is not supported")

        model.resize_token_embeddings(len(self.tokenizer))

        if self.adapter:

            model = PeftModel.from_pretrained(model, self.adapter)

        return model

    def generate_message(
        self,
        subject: str,
        attachments: bool,
        urls: Optional[bool],
        guided: Union[Literal["both"], bool] = False,
    ):

        prompt = generate_prompt(subject, attachments, urls)

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
            print(output_text)

        if guided:
            model = outlines.models.Transformers(self.gen_model, self.tokenizer)
            generator = outlines.generate.json(model, OutputMessage)
            message = generator([prompt])
            output_text = message
            print(output_text)
