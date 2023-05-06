import json
from typing import Union

class AlpacaPromptTemplate:
    def __init__(self):
        file_name = "templates/alpaca_template.json"
        with open(file_name) as fp:
            self.template = json.load(fp)

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        return res
