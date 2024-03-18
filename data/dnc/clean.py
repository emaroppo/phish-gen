from bs4 import BeautifulSoup
import os
from pymongo import MongoClient
from tqdm import tqdm

db = MongoClient().dnc_emails


def load_raw_data(db, folder_path="data/WIKILEAKS DNC EMAILS"):
    files = os.listdir(folder_path)

    files = [i for i in files if i.endswith(".htm")]
    files = [i for i in files if not i.startswith(".")]

    for i in tqdm(files):
        with open("data/WIKILEAKS DNC EMAILS/" + i, "r") as f:
            contents = f.read()
            soup = BeautifulSoup(contents, "html.parser")
            # retrieve "content class"
            content = soup.find_all("div", class_="content")
            for thread in content:
                doc = {"email-threads": thread.get_text()}
                db["raw_data"].insert_one(doc)


load_raw_data(db)
