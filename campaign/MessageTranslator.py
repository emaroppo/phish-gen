from openai import OpenAI
from typing import List
from pydantic import BaseModel
import pickle as pkl

with open("secrets/api_key.pkl", "rb") as f:
    api_key = pkl.load(f)

client = OpenAI(
    api_key=api_key,
)


class TranslatorResponse(BaseModel):
    translated_text: str
    translated_title: str


class MessageTranslator(BaseModel):
    sys_prompt: str = (
        "You are a proficient translator tasked with translating a list of email messages from English to Italian."
    )
    dos: List[str] = [
        "Prioritize preserving the original meaning, tone and level of formality over a literal translation.",
        "Ensure that the translated messages are grammatically correct and sound natural in Italian.",
    ]
    donts: List[str] = [
        "Avoid unnecessary embellishments or changes to the original messages.",
        "Do not use overly complex, uncommon, or archaic language in the translations.",
        "Do not translate placeholder tokens (e.g. <PER>, <ORG>, <DATE>, etc.)",
    ]

    def generate_prompt(self, original_title, original_text) -> str:
        prompt = f"Translate the following email message from English to Italian:\n\nOriginal Titile:\n\n{original_title} Original Text:\n\n{original_text}\n\n"
        dos = "\n-".join(self.dos)
        donts = "\n-".join(self.donts)
        prompt += f"DOs:\n-{dos}\n\nDON'Ts:\n-{donts}"

        return prompt

    def translate(self, original_title, original_text) -> TranslatorResponse:
        messages = [
            {
                "role": "system",
                "content": self.sys_prompt,
            },
            {
                "role": "user",
                "content": self.generate_prompt(original_title, original_text),
            },
        ]
        return (
            client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=messages,
                response_format=TranslatorResponse,
            )
            .choices[0]
            .message
        ).parsed


translator = MessageTranslator()
