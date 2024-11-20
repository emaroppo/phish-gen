from pydantic import BaseModel, computed_field
from typing import List
import pickle as pkl
from openai import OpenAI

with open("secrets/api_key.pkl", "rb") as f:
    api_key = pkl.load(f)

client = OpenAI(
    api_key=api_key,
)


class TitleList(BaseModel):
    titles: List[str]


class TitleGenerator(BaseModel):
    sys_prompt: str = (
        "You are a helpful assistant tasked with generating subject lines for various emails."
    )
    dos: List[str]
    donts: List[str]

    @computed_field
    def api_key(self) -> str:
        with open("api_key.pkl", "rb") as f:
            return pkl.load(f)

    def generate_prompt(self, context: str) -> str:
        dos = "\n-".join(self.dos)
        donts = "\n-".join(self.donts)
        prompt = "\n\n".join([context, "Do:\n-" + dos, "Don't:\n-" + donts])
        return prompt

    def generate_titles(self, context: str, model="gpt-4o-mini") -> List[str]:

        prompt = self.generate_prompt(context)

        messages = [
            {
                "role": "system",
                "content": self.sys_prompt,
            },
            {"role": "user", "content": prompt},
        ]

        titles = client.beta.chat.completions.parse(
            model="gpt-4o-mini", messages=messages, response_format=TitleList
        )

        titles = titles.choices[0].message.parsed

        return titles


title_generator = TitleGenerator(
    sys_prompt="You are a helpful assistant tasked with generating subject lines for various emails.",
    dos=[
        "Respond with a list of 20 alternatives.",
        "Offer a variety of tones.",
    ],
    donts=[
        "Do not be overly familiar.",
        "Do not use emojis.",
    ],
)
