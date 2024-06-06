from tqdm import tqdm
import os
from offline_finetuning.common_classes.DataImporter import DataImporter, query_manager
from offline_finetuning.data.enron.EnronThread import EnronThread
from offline_finetuning.data.enron.placeholder_dict import placeholder_dict


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

    def step3_insert_placeholders(
        self,
        collections: list,
        placeholder_dict: dict = placeholder_dict,
        target_collection: str = "step3_placeholders",
    ):
        print(placeholder_dict)
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
                    thread.insert_placeholder(
                        key,
                        value["placeholder"],
                        regex,
                        save=True,
                        target_collection=target_collection,
                    )

    def pipeline(self):
        self.step1_load_raw_data()
        self.step2_split_messages()

        return
