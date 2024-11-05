from pydantic import BaseModel, computed_field
from typing import Literal, Optional, Union, Any, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch
import outlines
from inference.prompt_generation.generate_prompt import (
    generate_prompt,
    OutputMessage,
)
from functools import cached_property
from finetuning.sft.classes.finetuned_models.FinetunedModel import FinetunedModel


class MessageGenerator(BaseModel):
    finetuned_model: FinetunedModel
    checkpoint: Optional[int] = None
    _model_and_tokenizer: Tuple[Any, Any] = None

    def _load_model_and_tokenizer(self) -> Tuple[Any, Any]:
        if self._model_and_tokenizer is None:
            model, tokenizer = self.finetuned_model.load_model(self.checkpoint)
            self._model_and_tokenizer = (model, tokenizer)
        return self._model_and_tokenizer

    @computed_field
    @cached_property
    def tokenizer(self) -> Any:
        _, tokenizer = self._load_model_and_tokenizer()
        return tokenizer

    @computed_field
    @cached_property
    def gen_model(self) -> Any:
        model, _ = self._load_model_and_tokenizer()
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
                max_length=512,
                num_return_sequences=1,
                top_k=50,
                top_p=0.95,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=1.0,
            )

            output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            print(output_text)
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
