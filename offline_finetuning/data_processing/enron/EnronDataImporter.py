from tqdm import tqdm
import os
from offline_finetuning.common_classes.DataImporter import DataImporter, query_manager
from bson import ObjectId
from offline_finetuning.data_processing.enron.EnronThread import EnronThread
from offline_finetuning.data_processing.enron.regex.placeholder_dict import (
    placeholder_dict,
)
from offline_finetuning.data_processing.enron.regex.default_footers import (
    footers_collection,
)
from offline_finetuning.data_processing.enron.regex.disclaimers_collection import (
    disclaimers_collection,
)

from offline_finetuning.auto_labelling.NERMessageLabeller import NERMessageLabeller
from offline_finetuning.auto_labelling.topic_modelling.TopicModelling import (
    TopicModelling,
)
from offline_finetuning.auto_labelling.MessageLabeller import MessageLabeller


class EnronDataImporter(DataImporter):

    def step1_load_raw_data(self, collection="step1_raw_data"):
        for i in tqdm(os.listdir(self.data_dir)):
            if "inbox" in os.listdir(self.data_dir + "/" + i):
                for j in tqdm(os.walk(self.data_dir + "/" + i + "/inbox")):
                    for k in j[2]:
                        with open(j[0] + "/" + k, "r", errors="ignore") as f:
                            text = f.read()
                            thread = EnronThread.from_text(
                                text, j[0] + "/" + k, self.db_name, collection
                            )
                            thread.save()

    def step2_split_messages(self):

        # copy collection from step 1
        query_manager.connection[self.db_name]["step2_raw_data"].insert_many(
            query_manager.connection[self.db_name]["step1_raw_data"].find()
        )

        # isolate multipart, forwarded and single messages
        match_dict = {"messages.body": {"$regex": "(-Original Message-)"}}

        self.split_collection("step2_raw_data", "step2_multipart", match_dict)

        match_dict = {"messages.body": {"$regex": "(- Forwarded by)"}}
        self.split_collection("step2_raw_data", "step2_forwarded", match_dict)

        # rename the remaining collection to step2_single
        query_manager.connection[self.db_name]["step2_single"].insert_many(
            query_manager.connection[self.db_name]["step2_raw_data"].find()
        )
        query_manager.connection[self.db_name]["step2_raw_data"].drop()

        # split multipart messages
        for i in tqdm(query_manager.connection[self.db_name]["step2_multipart"].find()):
            thread = EnronThread.deserialize(i, self.db_name, "step2_multipart")
            thread.clean()

        # extract headers from single messages
        for i in tqdm(query_manager.connection[self.db_name]["step2_single"].find()):
            thread = EnronThread.deserialize(
                i, db_name=self.db_name, collection="step2_single"
            )
            for j in thread.messages:
                j.extract_headers_main()
            thread.save()

    def step2_5_clean_messages(self):
        # extract disclaimers
        collections = ("step2_single", "step2_forwarded", "step2_multipart")
        for collection in collections:
            for disclaimer in tqdm(disclaimers_collection):
                threads = [
                    EnronThread.deserialize(thread, self.db_name, collection)
                    for thread in query_manager.connection[self.db_name][
                        collection
                    ].find(
                        {
                            "messages.body": {
                                "$regex": disclaimer,
                            }
                        }
                    )
                ]
                for thread in threads:
                    thread.extract_disclaimers(disclaimer, save=True)

            # get rid of footers
            for footer in tqdm(footers_collection):
                threads = [
                    EnronThread.deserialize(thread, self.db_name, collection)
                    for thread in query_manager.connection[self.db_name][
                        collection
                    ].find(
                        {
                            "messages.body": {
                                "$regex": footer,
                            }
                        }
                    )
                ]
                for thread in threads:
                    thread.remove_footers(footer, save=True)

            # get rid of leading and trailing white spaces

            threads = [
                EnronThread.deserialize(thread, self.db_name, collection)
                for thread in query_manager.connection[self.db_name][collection].find()
            ]
            for thread in threads:
                for message in thread.messages:
                    message.body = message.body.strip()
                    if message.headers:
                        message.clean_subject()
                    message.check_html()
                    message.get_word_count()
                thread.save()

    def step3_label_messages(
        self,
        collections: list,
        placeholder_dict: dict,
        target_collection: str = "step3_placeholders",
    ):

        # Named Entity Recognition

        # Manually Extract Entities

        for key, value in placeholder_dict.items():
            for regex in value["regex"]:
                threads = list()
                for collection in collections:
                    threads.extend(
                        [
                            EnronThread.deserialize(
                                thread, db_name=self.db_name, collection=collection
                            )
                            for thread in query_manager.connection[self.db_name][
                                collection
                            ].find({"messages.body": {"$regex": regex}})
                        ]
                    )
                for thread in tqdm(threads):
                    for message in thread.messages:
                        extracted_entities = message.extract_entities(
                            value["placeholder"],
                            regex,
                        )
                        for entity in extracted_entities:
                            start, end, label, entity_value = entity
                            message.add_entity(
                                start=start,
                                end=end,
                                entity_value=entity_value,
                                entity_type=label,
                                detection_method="manual",
                            )
                    thread.save(target_collection=target_collection)

        message_labeller = NERMessageLabeller(
            classifier_id="dslim/bert-large-NER",
            task="ner",
        )
        for collection in collections:
            threads = [
                EnronThread.deserialize(
                    thread, db_name=self.db_name, collection=collection
                )
                for thread in query_manager.connection[self.db_name][collection].find()
            ]
            for thread in tqdm(threads):
                for message in thread.messages:
                    extracted_entities = message_labeller.label_message(message.body)

                    if extracted_entities:
                        for entity in extracted_entities:

                            start, end, label = entity
                            entity_value = message.body[start:end]
                            message.add_entity(
                                entity_type=label,
                                start=start,
                                end=end,
                                entity_value=entity_value,
                                detection_method="auto",
                            )
                thread.save(target_collection=target_collection)

        # Sentiment Analysis

        message_labeller = MessageLabeller(
            classifier_id="michellejieli/emotion_text_classifier", task="sentiment"
        )
        for collection in collections:
            threads = [
                EnronThread.deserialize(
                    thread, db_name=self.db_name, collection=collection
                )
                for thread in query_manager.connection[self.db_name][collection].find()
            ]
            for thread in tqdm(threads):
                for message in thread.messages:
                    sentiment = message_labeller.label_message(message.body)
                    message.add_sentiment(sentiment)
                thread.save(target_collection=target_collection)

        # perform topic modeling on the messages

        return

    def step3_insert_placeholders(
        self,
        collections: list,
        placeholder_dict: dict = placeholder_dict,
        target_collection: str = "step3_placeholders",
    ):
        for key, value in placeholder_dict.items():
            for regex in value["regex"]:
                threads = list()
                for collection in collections:
                    threads.extend(
                        [
                            EnronThread.deserialize(
                                thread, db_name=self.db_name, collection=collection
                            )
                            for thread in query_manager.connection[self.db_name][
                                collection
                            ].find({"messages.body": {"$regex": regex}})
                        ]
                    )
                for thread in tqdm(threads):
                    thread.extract_entities(
                        key,
                        value["placeholder"],
                        regex,
                        save=True,
                        target_collection=target_collection,
                    )

                    # remove the original messages
                    for collection in collections:
                        query_manager.connection[self.db_name][collection].delete_many(
                            {"_id": ObjectId(thread.id)}
                        )

        for key, value in placeholder_dict.items():
            for regex in value["regex"]:
                threads = [
                    EnronThread.deserialize(
                        thread, db_name=self.db_name, collection="step3_placeholders"
                    )
                    for thread in query_manager.connection[self.db_name][
                        "step3_placeholders"
                    ].find({"messages.body": {"$regex": regex}})
                ]
                for thread in tqdm(threads):
                    thread.insert_placeholder(
                        key,
                        value["placeholder"],
                        regex,
                        save=True,
                    )

    def pipeline(self):
        # self.step1_load_raw_data()
        # self.step2_split_messages()
        # self.step2_5_clean_messages()
        self.step3_label_messages(
            ["step2_single", "step2_forwarded", "step2_multipart"],
            placeholder_dict=placeholder_dict,
            target_collection="step3_placeholders",
        )

        return
