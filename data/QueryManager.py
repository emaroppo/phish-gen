from pymongo import MongoClient

class QueryManager:

    def __init__(self):
        self.connection = MongoClient()


query_manager = QueryManager()
