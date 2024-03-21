from pymongo import MongoClient


class QueryManager:

    def __init__(self):
        self.connection = MongoClient()

    def save_entry(self, entry, db_name, collection):
        entry_id = self.connection[db_name][collection].insert_one(entry)
        return entry_id

    def retrieve_entry(self, query, db_name, collection):
        return self.connection[db_name][collection].find_one(query)


query_manager = QueryManager()
