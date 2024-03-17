from bs4 import BeautifulSoup
import os
from pymongo import MongoClient
from tqdm import tqdm

db = MongoClient().dnc_emails
connection = db.raw_data

files = os.listdir('data/WIKILEAKS DNC EMAILS')
files = [i for i in files if not (i.startswith('.') and i endswith('.htm'))]

for i in tqdm(os.listdir('data/WIKILEAKS DNC EMAILS')):
    with open('data/WIKILEAKS DNC EMAILS/' + i, 'r') as f:
        contents = f.read()
        soup = BeautifulSoup(contents, 'html.parser')
        #retrieve "content class"
        content = soup.find_all("div", class_="content")
        for thread in content:
            doc = { "email-threads": thread.get_text() }
            connection.insert_one(doc)