from pydantic import BaseModel
from bson import ObjectId, DBRef
from typing import List, Dict
from campaign.constants import (
    PHISHING_CAMPAIGN_DB,
    TEMPLATES_COLLECTION,
    TITLES_COLLECTION,
)
from campaign.classes.TitleEntry import TitleEntry
from data.QueryManager import query_manager
from inference.MessageGenerator import MessageGenerator
from campaign.MessageTranslator import translator
from typing import Optional


class TemplateEntry(BaseModel):
    id: Optional[str] = None
    title: TitleEntry
    translation: str
    url: bool = False
    attachment: bool = False
    clicked: int = 0
    not_clicked: int = 0
    body: str

    @classmethod
    def from_dictionary(cls, data: Dict):
        data["id"] = str(data["_id"])
        data.pop("_id")
        print(data["title"])

        data["title"] = query_manager.connection[PHISHING_CAMPAIGN_DB].dereference(
            data["title"]
        )
        return cls(**data)

    def serialise(self):
        return {
            "_id": ObjectId(self.id),
            "title": DBRef(TITLES_COLLECTION, ObjectId(self.title.id)),
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

    @classmethod
    def generate_templates_from_titles(
        cls,
        titles: List[TitleEntry],
        message_generator: MessageGenerator,
    ) -> list:

        templates = list()

        for title in titles:
            text = message_generator.generate_message(
                subject=title.title,
                attachments=False,
                sentiment=["neutral"],
                urls=True,
            ).split("\n->\nbody:")[1]
            translation = translator.translate(original_title=title, original_text=text)
            title.translation = translation.translated_title
            title.save()
            text_translation = translation.translated_text
            title_entry = title
            template = cls(
                body=text,
                title=title_entry,
                translation=text_translation,
                url=True,
                attachment=False,
            )
            templates.append(template)

        return templates

    @classmethod
    def generate_templates_from_context(
        cls, context: str, targets: List[str], message_generator: MessageGenerator
    ) -> list:

        titles = TitleEntry.generate_titles(context=context, targets=targets)
        titles = [title.save() for title in titles]
        return cls.generate_templates_from_titles(
            titles=titles, message_generator=message_generator
        )
