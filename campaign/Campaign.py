from finetuning.sft.classes.finetuned_models.FinetunedLLama31 import FinetunedLLama31
from campaign.EmailSender import email_sender
from inference.MessageGenerator import MessageGenerator
from pydantic import BaseModel
from campaign.classes.TitleEntry import TitleEntry
from constants import (
    PHISHING_CAMPAIGN_DB,
    TEMPLATES_COLLECTION,
    TITLES_COLLECTION,
    EMAIL_ADDRESS,
)
from bson import DBRef, ObjectId
from typing import List, Tuple, Dict
from campaign.classes.TemplateEntry import TemplateEntry
from data.QueryManager import query_manager
from pymongo import MongoClient

model = FinetunedLLama31.from_db(1730656372)
message_generator = MessageGenerator(finetuned_model=model, checkpoint=2122)

query_manager.connection = MongoClient(
    "mongodb+srv://emanueleroppo:IeHhiwxboMZNZWGQ@feedback.yduxn.mongodb.net/?retryWrites=true&w=majority&appName=Feedback"
)


class Campaign(BaseModel):
    email_addresses: Dict[str, List[str]]

    def generate_templates_from_context_list(
        self, context_list: List[Tuple[str, List[str]]]
    ):
        for context, targets in context_list:
            print(context)
            templates = TemplateEntry.generate_templates_from_context(
                context=context,
                targets=targets,
                message_generator=message_generator,
            )
            for template in templates:
                template.save()

    def carry_out_campaign(self):
        email_template_pairs = list()
        print(self.email_addresses)

        for target_category in self.email_addresses.keys():
            number_targets = len(self.email_addresses[target_category])

            # add further filters (e.g. time of the year)

            pipeline = [
                {"$match": {"targets": target_category}},
                {"$sample": {"size": number_targets}},
            ]

            titles = [
                TitleEntry.from_dict(title)
                for title in query_manager.connection[PHISHING_CAMPAIGN_DB][
                    TITLES_COLLECTION
                ].aggregate(pipeline)
            ]

            templates = list()
            print(titles)
            for title in titles:
                pipeline = [
                    {"$match": {"title": DBRef(TITLES_COLLECTION, ObjectId(title.id))}},
                    {"$sample": {"size": 1}},
                ]
                template_data = query_manager.connection[PHISHING_CAMPAIGN_DB][
                    TEMPLATES_COLLECTION
                ].aggregate(pipeline)
                for i in template_data:
                    template = i

                template = TemplateEntry.from_dictionary(template)
                templates.append(template)
            email_template_pairs.extend(
                zip(self.email_addresses[target_category], templates)
            )

            print(email_template_pairs)

        for email, template in email_template_pairs:
            # add logic to stagger sending of emails
            email_sender.send_email(email, template)
