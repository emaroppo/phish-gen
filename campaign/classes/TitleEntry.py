from pydantic import BaseModel
from typing import List, Optional
from bson import ObjectId
from data.QueryManager import query_manager
from campaign.constants import PHISHING_CAMPAIGN_DB, TITLES_COLLECTION
from campaign.TitleGenerator import title_generator


class TitleEntry(BaseModel):
    id: Optional[str] = None
    title: str
    context: str
    targets: List[str]
    translation: Optional[str] = None

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
            "targets": self.targets,
            "translation": self.translation,
        }

    def save(self):
        if self.id is None:
            self.id = str(
                query_manager.connection[PHISHING_CAMPAIGN_DB][TITLES_COLLECTION]
                .insert_one(self.serialise())
                .inserted_id
            )
        else:
            query_manager.connection[PHISHING_CAMPAIGN_DB][
                TITLES_COLLECTION
            ].update_one({"_id": ObjectId(self.id)}, {"$set": self.serialise()})
        return self

    @classmethod
    def generate_titles(cls, context: str, targets) -> list:
        titles = title_generator.generate_titles(context)
        titles = titles.titles
        titles = [
            cls(title=title, context=context, targets=targets) for title in titles
        ]
        return titles
