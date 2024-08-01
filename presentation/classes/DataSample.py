from pydantic import BaseModel, computed_field
from typing import List, Dict, Optional
from functools import cached_property


class DataSample(BaseModel):
    urls: bool
    attachments: bool
    body: str
    subject: str
    attachment_formats: Optional[List[str]] = None
    sentiment: List[str]
    _entity_counts: Optional[Dict[str, int]] = dict()

    @property
    def entity_counts(self):
        return self._entity_counts

    @entity_counts.setter
    def entity_counts(self, custom_tokens: List[str]):
        for token in custom_tokens:
            if token in self.body:
                if token not in self._entity_counts:
                    self._entity_counts[token] = 0
                self._entity_counts[token] += self.body.count(token)

    @classmethod
    def deserialize(cls, data):
        return cls(
            urls=data["urls"],
            attachments=data["attachments"],
            body=data["body"],
            subject=data["subject"],
            attachment_formats=data["attachment_formats"],
            sentiment=data["sentiment"],
            entity_counts=data["entity_counts"],
        )

    def serialise(self):
        return {
            "urls": self.urls,
            "attachments": self.attachments,
            "body": self.body,
            "subject": self.subject,
            "attachment_formats": self.attachment_formats,
            "sentiment": self.sentiment,
            "entity_counts": self.entity_counts,
        }
