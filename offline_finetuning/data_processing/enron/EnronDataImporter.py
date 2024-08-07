from tqdm import tqdm
import os
import json
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

from offline_finetuning.auto_labelling.SentimentMessageLabeller import (
    SentimentMessageLabeller,
)
from offline_finetuning.auto_labelling.TopicMessageLabeller import TopicMessageLabeller


class EnronDataImporter(DataImporter):

    def step1_load_raw_data(self, collection="step1_raw_data"):
        for i in tqdm(os.listdir(self.data_dir), desc="Loading raw inboxes"):
            if "inbox" in os.listdir(self.data_dir + "/" + i):
                for j in os.walk(self.data_dir + "/" + i + "/inbox"):
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
        for i in tqdm(
            query_manager.connection[self.db_name]["step2_multipart"].find(),
            desc="Splitting multipart messages",
        ):
            thread = EnronThread.deserialize(i, self.db_name, "step2_multipart")
            thread.clean()

        # extract headers from single messages
        for i in tqdm(
            query_manager.connection[self.db_name]["step2_single"].find(),
            desc="Extracting headers",
        ):
            thread = EnronThread.deserialize(
                i, db_name=self.db_name, collection="step2_single"
            )
            for j in thread.messages:
                j.extract_headers_main()
            thread.save()

        # extract forwarded messages
        for i in tqdm(
            query_manager.connection[self.db_name]["step2_forwarded"].find(),
            desc="Separating forwarded messages",
        ):
            thread = EnronThread.deserialize(
                i, db_name=self.db_name, collection="step2_forwarded"
            )
            thread.extract_forwarded_messages(thread.messages[0])

    def step2_5_clean_messages(self):
        # extract disclaimers
        collections = ("step2_single", "step2_forwarded", "step2_multipart")

        for collection in collections:
            print(f"Extracting disclaimers from {collection}...")
            for disclaimer in disclaimers_collection:
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
            print(f"Extracting footers from {collection}...")
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
            print("Tidy up bodies...")
            threads = [
                EnronThread.deserialize(thread, self.db_name, collection)
                for thread in query_manager.connection[self.db_name][collection].find()
            ]
            for thread in threads:
                for message in thread.messages:

                    message.body = message.body.replace("[IMAGE]", "").strip()
                    if message.headers:
                        message.clean_subject()
                    message.check_html()
                    message.get_word_count()
                thread.save()

    def step3_label_messages(
        self,
        collections: list,
        placeholder_dict: dict,
        validated_entities_dict: dict = None,
    ):

        # Named Entity Recognition

        # Manually Extract Entities
        # TODO: move manual entity extraction to NERMessageLabeller class
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

                for thread in tqdm(threads, desc=f"Extracting {key}"):
                    thread.extract_entities(
                        value["placeholder"], regex, detection_mode="manual", save=True
                    )

        for key, value in validated_entities_dict.items():
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

                for thread in tqdm(threads, desc=f"Extracting {key}"):
                    thread.extract_entities(
                        value["placeholder"], regex, detection_mode="manual", save=True
                    )

        ner_message_labeller = NERMessageLabeller(
            classifier_id="dslim/bert-large-NER",
            task="ner",
        )

        sentiment_message_labeller = SentimentMessageLabeller(
            classifier_id="michellejieli/emotion_text_classifier",
            task="sentiment-analysis",
            no_top_k=True,
        )

        """
        topic_message_labeller = TopicMessageLabeller(
            classifier_id="offline_finetuning/auto_labelling/topic_modelling/models/topic_model",
        )
        """

        for collection in collections:
            threads = [
                EnronThread.deserialize(
                    thread, db_name=self.db_name, collection=collection
                )
                for thread in query_manager.connection[self.db_name][collection].find()
            ]

            for thread in tqdm(threads, desc=f"Generating labels for {collection}"):

                thread.extract_entities(
                    message_labeller=ner_message_labeller,
                    detection_mode="auto",
                    save=True,
                )

                thread.predict_sentiment(
                    sentiment_predictor=sentiment_message_labeller, save=True
                )

                # thread.predict_topic(topic_predictor=topic_message_labeller, save=True)

        # extract attachment formats
        print("Extracting attachment formats...")
        for collection in collections:
            threads = [
                EnronThread.deserialize(
                    thread, db_name=self.db_name, collection=collection
                )
                for thread in query_manager.connection[self.db_name][collection].find(
                    {"messages.entities.manual.ATTACHMENT.0": {"$exists": True}}
                )
            ]
            for thread in tqdm(threads):
                thread.get_attachments_formats(save=True)

        # remove overlapping labels
        for collection in collections:
            threads = [
                EnronThread.deserialize(
                    thread, db_name=self.db_name, collection=collection
                )
                for thread in query_manager.connection[self.db_name][collection].find()
            ]
            for thread in tqdm(threads):
                thread.remove_overlapping_labels()

        return

    def pipeline(self):
        self.step1_load_raw_data()
        self.step2_split_messages()
        self.step2_5_clean_messages()

        with open(
            "offline_finetuning/data_processing/enron/regex/entities_reg.json", "r"
        ) as f:
            validated_entities_dict = json.load(f)

        self.step3_label_messages(
            ["step2_single", "step2_forwarded", "step2_multipart"],
            placeholder_dict=placeholder_dict,
            validated_entities_dict=validated_entities_dict,
        )

        return
