from pydantic import BaseModel
from inference.TitleGenerator import title_generator
from data.QueryManager import query_manager
from finetuning.sft.classes.finetuned_models import FinetunedLLama31
from inference.MessageGenerator import MessageGenerator
from inference.MessageTranslator import translator
from typing import Optional, List, Tuple
from bson import ObjectId, DBRef
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime
import smtplib
import json
import random

with open("inference/emails.json", "r") as f:
    EMAIL_ADDRESS = json.load(f)

PHISHING_CAMPAIGN_DB = "phishing_campaign"
TEMPLATES_COLLECTION = "templates"
MESSAGES_COLLECTION = "messages"
TITLES_COLLECTION = "titles"

model = FinetunedLLama31.from_db(1730656372)


message_generator = MessageGenerator(model=model, checkpoint=2122)


class TitleEntry(BaseModel):
    id: str
    title: str
    context: str
    target: List[str]
    translation: str

    @classmethod
    def from_dict(cls, data: dict):
        data["id"] = str(data["_id"])
        data.pop("_id")
        return cls(**data)

    def serialise(self):
        return {
            "_id": ObjectId(self.id),
            "title": self.title,
            "context": self.context,
            "target": self.target,
            "translation": self.translation,
        }

    def save(self):
        query_manager.connection[PHISHING_CAMPAIGN_DB][TEMPLATES_COLLECTION].insert_one(
            self.serialise()
        )


class TemplateEntry(BaseModel):
    id: str
    title: TitleEntry
    translation: str
    url: bool = False
    attachment: bool = False
    clicked: int = 0
    not_clicked: int = 0
    body: str

    @classmethod
    def from_dictionary(cls, data):
        data["id"] = str(data["_id"])
        data.pop("_id")
        return cls(**data)

    def serialise(self):
        return {
            "_id": ObjectId(self.id),
            "title": DBRef(TITLES_COLLECTION, self.title.id),
            "translation": self.translation,
            "url": self.url,
            "attachment": self.attachment,
            "clicked": self.clicked,
            "not_clicked": self.not_clicked,
            "body": self.body,
        }

    def save(self):
        query_manager.connection[PHISHING_CAMPAIGN_DB][TEMPLATES_COLLECTION].insert_one(
            self.serialise()
        )


class MessageEntry(BaseModel):
    id: str
    template: TemplateEntry
    date_sent: Optional[int] = None
    clicked: Optional[bool] = None
    message_body: Optional[str] = None

    def serialise(self):
        return {
            "_id": ObjectId(self.id),
            "template": DBRef(TEMPLATES_COLLECTION, self.template.id),
            "date_sent": self.date_sent,
            "clicked": self.clicked,
        }


def generate_titles(context: str, target) -> list:
    titles = title_generator.generate_titles(context)
    titles = [
        TitleEntry(title=title, context=context, target=target) for title in titles
    ]
    return titles


def generate_templates(titles: List[TitleEntry], context: str) -> list:

    templates = list()

    for target, title, context in titles:
        text = message_generator.generate_message(
            subject=title,
            attachments=False,
            sentiment=["neutral"],
            urls=True,
        )
        translation = translator.translate(original_title=title, original_text=text)
        title_translation = translation["translated_title"]
        text_translation = translation["translated_text"]
        title_entry = TitleEntry(
            title=title,
            context=context,
            target=target,
            translation=title_translation,
        )
        template = TemplateEntry(
            body=text,
            title=title_entry,
            translation=text_translation,
            url=True,
            attachment=False,
        )
        templates.append(template)

    return templates


def fill_placeholders(template: TemplateEntry, placeholder_contents: dict) -> str:
    body = template.body
    for placeholder, content in placeholder_contents.items():
        body = body.replace(placeholder, content)
    raise NotImplementedError("fill_placeholders not implemented")


def inject_url(body: str, base_url: str, message_id: str) -> str:
    url = f"{base_url}/{message_id}"
    body = body.replace("<URL>", url)
    return body


def generate_message(template: TemplateEntry, date_sent) -> str:
    message_body = fill_placeholders(template, {})
    message_id = ObjectId()
    message_body = inject_url(message_body, "https://example.com", message_id)
    message_entry = MessageEntry(id=message_id, template=template, date_sent=date_sent)
    message_entry.save()
    return message_body


def send_message(template: TemplateEntry, address: str) -> None:
    date_sent = int(datetime.now().timestamp())
    message = generate_message(template, date_sent)
    msg = MIMEMultipart()
    msg["From"] = ""
    msg["To"] = address
    msg["Subject"] = template.title.title
    msg.attach(MIMEText(message, "plain"))
    server = smtplib.SMTP("localhost", 1025)
    server.send_message(msg)
    server.quit()
    return True


def select_address_template_pair(target_category: str = ""):
    if target_category == "":
        target_category = random.choice(EMAIL_ADDRESS.keys())
    address = random.choice(EMAIL_ADDRESS[target_category])
    # retrieve random template matching target category
    pipeline = [
        {"$match": {"title.target": target_category}},
        {"$sample": {"size": 1}},
    ]
    template = query_manager.connection[PHISHING_CAMPAIGN_DB][
        TEMPLATES_COLLECTION
    ].aggregate(pipeline)
    template = TemplateEntry.from_dictionary(template)
    return address, template
