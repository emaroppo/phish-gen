from tqdm import tqdm
import os
from common_classes.DataImporter import DataImporter, query_manager
from data.enron.EnronThread import EnronThread


class EnronDataImporter(DataImporter):
    def __init__(self, data_dir, db_name):
        super().__init__(data_dir, db_name)

    def load_raw_data(self):
        for i in tqdm(os.listdir(self.data_dir)):
            if "inbox" in os.listdir(self.data_dir + "/" + i):
                for j in tqdm(os.walk(self.data_dir + "/" + i + "/inbox")):
                    for k in j[2]:
                        with open(j[0] + "/" + k, "r", errors="ignore") as f:
                            text = f.read()
                            thread = EnronThread.from_text(
                                text, j[0] + "/" + k, self.db_name, "raw_data"
                            )

    def isolate_multiparts(self, sample=None):
        match_dict = {
            "messages.body": {"$regex": "(-Original Message-)|(- Forwarded by)"}
        }
        if type(sample) == int:
            multithread_docs = self.retrieve_samples(match_dict, n=sample)
        elif sample is None:
            multithread_docs = query_manager.connection[self.db_name]["raw_data"].find(
                match_dict
            )

        query_manager.connection[self.db_name]["raw_data_multipart"].insert_many(
            multithread_docs
        )

    def clean_multiparts(self, sample=None):

        if type(sample) == int:
            multithread_docs = self.retrieve_samples(
                collection="raw_data_multipart", n=sample
            )
        elif sample is None:
            multithread_docs = query_manager.connection["enron_emails"][
                "raw_data_multipart"
            ].find()

        for i in tqdm(multithread_docs):
            thread = EnronThread.from_db(
                _id=i["_id"], db_name=self.db_name, collection="raw_data_multipart"
            )
            thread.clean()
            thread.save()
