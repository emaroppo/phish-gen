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
    translation: str


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

    def generate_prompt(self, original_text) -> str:
        prompt = f"Translate the following email message from English to Italian:\n\nOriginal Text:\n\n{original_text}\n\n"
        dos = "\n-".join(self.dos)
        donts = "\n-".join(self.donts)
        prompt += f"DOs:\n-{dos}\n\nDON'Ts:\n-{donts}"

        return prompt

    def translate(self, original_text) -> List[str]:
        messages = [
            {
                "role": "system",
                "content": self.sys_prompt,
            },
            {"role": "user", "content": self.generate_prompt(original_text)},
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
translated_text = translator.translate(
    """Attached is the <ORG> Corp. payroll advisory for <DATE>.

Please remember to review the attached Advisory for updates to your payroll information.  Many of you have personal information that has been updated in SAP HR.  We are continuing to work closely with <ORG> Networks IT group and <ORG> HR on a resolution to correct the issues we have seen for the month of <DATE>.

Our <ORG> IT group and HR IT group has committed to supporting us, we are all working together in an effort to resolve these issues.

We are providing you with updates and we will continue to update you as we have more details and information.

Thank you for your continued efforts."""
)

print(translated_text)
