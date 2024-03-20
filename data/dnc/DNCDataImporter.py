from bs4 import BeautifulSoup
import os
from pymongo import MongoClient
from tqdm import tqdm
from data.DataImporter import DataImporter

db = MongoClient().dnc_emails


def load_raw_data(db, folder_path="data/dnc/dataset/WIKILEAKS DNC EMAILS/"):
    files = os.listdir(folder_path)

    files = [i for i in files if i.endswith(".htm")]  # emails are in html format
    files = [
        i for i in files if "-" not in i
    ]  # files containing "-" are not emails or are copy of the raw data of the emails
    files = [i for i in files if not i.startswith(".")]

    for i in tqdm(files):
        with open(folder_path + i, "r") as f:
            contents = f.read()
            soup = BeautifulSoup(contents, "html.parser")
            # retrieve "content class" (stores the email thread)
            content = soup.find_all("div", class_="content")
            for thread in content:
                doc = {"email-threads": thread.prettify()}
                doc["file_path"] = folder_path + "/" + i
                db["raw_data"].insert_one(doc)


load_raw_data(db)
