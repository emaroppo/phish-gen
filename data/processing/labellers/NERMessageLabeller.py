from typing import Any
from data.processing.labellers.MessageLabeller import MessageLabeller
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
from data.QueryManager import query_manager
from pydantic import computed_field
import torch
from functools import cached_property


class NERMessageLabeller(MessageLabeller):

    @computed_field
    @cached_property
    def classifier_model(self) -> Any:
        return AutoModelForTokenClassification.from_pretrained(self.classifier_id)

    @computed_field
    @cached_property
    def tokenizer(self) -> Any:
        return AutoTokenizer.from_pretrained(self.classifier_id)

    @computed_field
    @cached_property
    def classifier(self) -> Any:
        device = 0 if torch.cuda.is_available() else -1
        return pipeline(
            task="ner",
            model=self.classifier_model,
            tokenizer=self.tokenizer,
            device=device,
        )

    def label_message(self, message_body, batch_size=1):
        labels = self.classifier(message_body, batch_size=batch_size)
        if type(message_body) == str:
            if labels:
                # compose the tokens back into word
                words = list()
                first_label = labels.pop(0)
                word_start = first_label["start"]
                word_end = first_label["end"]
                word_label = first_label["entity"].split("-")[1]

                for label in labels:

                    if label["entity"][0] == "I":
                        word_end = label["end"]
                    elif label["entity"][0] == "B":
                        words.append((word_start, word_end, word_label))
                        word_start = label["start"]
                        word_end = label["end"]
                        word_label = label["entity"].split("-")[1]
                return words
        elif type(message_body) == list:
            predictions = list()
            for prediction in labels:
                if prediction:

                    words = list()
                    first_label = prediction.pop(0)
                    word_start = first_label["start"]
                    word_end = first_label["end"]
                    word_label = first_label["entity"].split("-")[1]

                    for label in prediction:

                        if label["entity"][0] == "I":
                            word_end = label["end"]
                        elif label["entity"][0] == "B":
                            words.append((word_start, word_end, word_label))
                            word_start = label["start"]
                            word_end = label["end"]
                            word_label = label["entity"].split("-")[1]
                    predictions.append(words)
            return predictions

    def generate_validation_excel(
        self,
        collections=["step2_single", "step2_multipart", "step2_forwarded"],
        file_path="offline_finetuning/auto_labelling/validation",
    ):
        people_set = set()
        org_set = set()
        location_set = set()
        misc_set = set()

        for i in collections:
            collection = query_manager.connection["enron_emails"][i]
            # retrieve a list of all entities
            threads = collection.find({"messages.entities.auto": {"$exists": True}})

            for thread in threads:
                for message in thread["messages"]:
                    if "PER" in message["entities"]["auto"]:
                        people = set(
                            [person[0] for person in message["entities"]["auto"]["PER"]]
                        )
                        people_set.update(people)
                    if "ORG" in message["entities"]["auto"]:
                        organizations = [
                            org[0] for org in message["entities"]["auto"]["ORG"]
                        ]
                        org_set.update(organizations)
                    if "LOC" in message["entities"]["auto"]:
                        locations = [
                            loc[0] for loc in message["entities"]["auto"]["LOC"]
                        ]
                        location_set.update(locations)
                    if "MISC" in message["entities"]["auto"]:
                        miscs = [
                            misc[0] for misc in message["entities"]["auto"]["MISC"]
                        ]
                        misc_set.update(miscs)

        super().generate_validation_excel(people_set, f"{file_path}/people.xlsx")
        super().generate_validation_excel(org_set, f"{file_path}/organizations.xlsx")
        super().generate_validation_excel(location_set, f"{file_path}/locations.xlsx")
        super().generate_validation_excel(misc_set, f"{file_path}/misc.xlsx")

        return people_set, org_set, location_set, misc_set
