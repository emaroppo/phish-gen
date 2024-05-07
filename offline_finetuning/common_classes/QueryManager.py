from pymongo import MongoClient
from bson import ObjectId


class QueryManager:

    def __init__(self):
        self.connection = MongoClient()

    def save_entry(self, entry, db_name, collection):
        if "_id" not in entry:
            entry["_id"] = ObjectId()
        self.connection[db_name][collection].update_one(
            {"_id": entry["_id"]},
            {"$set": entry},
            upsert=True,
        )

    def retrieve_entry(self, query, db_name, collection):
        return self.connection[db_name][collection].find_one(query)

    def strip_bodies(self, db_name, collection):
        # replace the body field with the stripped body field
        for entry in self.connection[db_name][collection].find():
            for message in entry["messages"]:
                message["body"] = message["body"].strip()
            self.save_entry(entry, db_name, collection)


query_manager = QueryManager()
