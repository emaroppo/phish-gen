from pymongo import MongoClient
from tqdm import tqdm
import os
import email


db = MongoClient().enron_emails


# parse data with email module and store in mongodb
# parse and store headers as well
# separate the email threads into individual emails
def load_raw_data(db, folder_path="data/enron/dataset/maildir"):
    for i in tqdm(os.listdir(folder_path)):
        if "inbox" in os.listdir(folder_path + "/" + i):
            for j in os.walk(folder_path + "/" + i + "/inbox"):

                for k in tqdm(j[2]):
                    with open(j[0] + "/" + k, "r", errors="ignore") as f:
                        try:
                            msg = email.message_from_file(
                                f, policy=email.policy.default
                            )
                            doc = {
                                "headers": dict(msg.items()),
                                "email-threads": msg.get_payload(),
                                "filepath": j[0] + "/" + k,
                            }
                            db["raw_data"].insert_one(doc)
                        except email.errors.MessageError:
                            print("Error in file: ", j[0] + "/" + k)


load_raw_data(db)
