from pydantic import BaseModel
from typing import List, Dict

class MessageGenerator(BaseModel):
    model : str

    def generate(self, prompt: Dict, inject_fields:Dict) -> str:
        message_template = self.model.generate(str(prompt))
        '''
        Assuming inject_fields is a dictionary where each key 
        is a special token and the value is a list of values to inject, in the order in which they should be injected
        Example:
        inject_fields = {
        "[URL]": ["http://malicious.com", "http://malware.org"],
        }
        '''
        for key, value in inject_fields.items():
            for inject_item in inject_fields[key]:
                message_template = message_template.replace(key, inject_item, 1)
        return message_template