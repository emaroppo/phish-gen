from common_classes.QueryManager import query_manager
import bson.json_util as json_util


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
        return list(
            query_manager.connection[self.db_name][collection].aggregate(pipeline)
        )

    def export_json(
        self, collection="raw_data_multipart", query=dict(), path="data.json", n=500
    ):
        with open(path, "w") as f:
            f.write("[\n")
            for i in self.retrieve_samples(query, collection, n):
                f.write(json_util.dumps(i) + ",\n")
            f.write("]\n")
        return
