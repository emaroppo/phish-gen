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


query_manager = QueryManager()
