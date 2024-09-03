from pydantic import BaseModel
from typing import List, Dict, Union, Optional
from bson import ObjectId


class ModelOutput(BaseModel):
    body: str
    message_id: Optional[str] = None
    prompt: Dict[str, Union[str, List[str], bool]]
    output: Optional[Dict[str, Union[str, List[str], bool]]] = dict()
    tasks: Optional[Dict[str, bool]] = dict()

    @classmethod
    def deserialize(cls, data):
        if "_id" in data:
            data["message_id"] = str(data["_id"])
        else:
            data["message_id"] = None
        return cls(
            message_id=data["message_id"],
            body=data["body"],
            prompt=data["prompt"],
            output=data["output"],
            tasks=data["tasks"],
        )

    def serialise(self, to_db=True):

        return {
            "body": self.body,
            "prompt": self.prompt,
            "output": self.output,
            "tasks": self.tasks,
        }

    def add_output_sentiment(self, sentiment):
        self.output["sentiment"] = sentiment

    def add_output_entities(self):

        if "<URL>" in self.body:
            self.output["urls"] = True
        else:
            self.output["urls"] = False
        if "<ATTACHMENT>" in self.body:
            self.output["attachments"] = True
        else:
            self.output["attachments"] = False

    def calculate_tasks_results(self):
        for task in self.output.keys():
            if self.prompt[task] == self.output[task]:
                self.tasks[task] = True
            else:
                self.tasks[task] = False
