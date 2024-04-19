from pydantic import BaseModel
from typing import List, Dict
from common_classes.QueryManager import query_manager
from torch.utils.data import DataLoader
import torch
from transformers import PreTrainedTokenizer


class DatasetFactory(BaseModel):
    databases: Dict[str, List[str]]  # key: db_name, value: list of collections

    def generate_dataset(self, tokenizer: PreTrainedTokenizer, max_length: int = 512):
        for db_name, collections in self.databases.items():
            for collection in collections:
                # retrieve data from the database
                data = list(
                    query_manager.connection[db_name][collection].find().limit(5000)
                )

                data = [
                    i["messages"][0]
                    for i in data
                    if "messages" in i and "body" in i["messages"][0]
                ]

                tensorized_samples = []
                for sample in data:
                    del sample["_id"]
                    body = sample.pop("body")
                    prompt = f"{str(sample)} -> {str(body)}"
                    encoded_sample = tokenizer(
                        prompt,
                        truncation=True,
                        padding="max_length",
                        max_length=max_length,
                        return_tensors="pt",
                    )
                    tensorized_samples.extend(encoded_sample["input_ids"])

                return DataLoader(tensorized_samples, batch_size=32, shuffle=True)
