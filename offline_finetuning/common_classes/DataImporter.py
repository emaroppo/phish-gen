from offline_finetuning.common_classes.QueryManager import query_manager
from pydantic import BaseModel
import bson.json_util as json_util


class DataImporter(BaseModel):
    db_name: str
    data_dir: str

    def save_raw(self, data):
        query_manager.connection[self.db_name]["step1_raw_data"].insert_one(data)
        return

    def split_collection(self, source_collection, target_collection, match_dict):
        docs = query_manager.connection[self.db_name][source_collection].find(
            match_dict
        )
        query_manager.connection[self.db_name][target_collection].insert_many(docs)
        query_manager.connection[self.db_name][source_collection].delete_many(
            match_dict
        )
        return
