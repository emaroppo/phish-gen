from campaign.classes.TemplateEntry import TemplateEntry
from campaign.constants import TEMPLATES_COLLECTION
from pydantic import BaseModel
from bson import ObjectId, DBRef
from typing import Optional


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

    def inject_url(body: str, base_url: str, message_id: str) -> str:
        url = f"{base_url}/{message_id}"
        body = body.replace("<URL>", url)
        return body


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
