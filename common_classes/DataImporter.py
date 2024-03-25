from common_classes.QueryManager import query_manager


class DataImporter:
    def __init__(self, data_dir, db_name):
        self.data_dir = data_dir
        self.db_name = db_name

    def save_raw(self, data):
        query_manager[self.db_name]["raw_data"].insert_one(data)
        return

    def retrieve_samples(self, match_dict=dict(), collection="raw_data", n=1):
        # aggregate pipeline to retrieve n samples from the collection
        pipeline = [
            {"$match": match_dict},
            {"$sample": {"size": n}},
        ]
        return list(query_manager[self.db_name][collection].aggregate(pipeline))
