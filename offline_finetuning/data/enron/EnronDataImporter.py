from tqdm import tqdm
import os
from common_classes.DataImporter import DataImporter, query_manager
from data.enron.EnronThread import EnronThread
from offline_finetuning.data.enron.placeholder_dict import placeholder_dict


class EnronDataImporter(DataImporter):
    def __init__(self, data_dir, db_name):
        super().__init__(data_dir, db_name)

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
            thread = EnronThread.from_db(_id=i["_id"], db_name=self.db_name)
            thread.split_messages()
            thread.save()

    def clean_multiparts(self, sample=None):

        if type(sample) == int:
            multithread_docs = self.retrieve_samples(
                collection="raw_data_multipart", n=sample
            )
        elif sample is None:
            multithread_docs = query_manager.connection[self.db_name][
                "raw_data_multipart"
            ].find()

        for i in tqdm(multithread_docs):
            thread = EnronThread.from_db(
                _id=i["_id"], db_name=self.db_name, collection="raw_data_multipart"
            )
            thread.clean()
            thread.save()

    def step3_insert_placeholders(self, collections: list, placeholder_dict:dict = placeholder_dict):
        for key, value in placeholder_dict.items():
            for regex in value[key]["regex"]:
                threads = list()
                for collection in collections:
                    threads.extend([ EnronThread.deserialize(thread) for thread in
                        query_manager.connection[self.db_name][collection].find(
                            {"messages.body": {"$regex": regex}}
                        )
                    ]
                    )
                for thread in tqdm(threads):
                    thread.insert_placeholder(key, value["placeholder"], regex, save=False) #change save to True after testing

    def pipeline(self):
        self.load_raw_data()
        self.isolate_multiparts()
        self.clean_multiparts()
        return
