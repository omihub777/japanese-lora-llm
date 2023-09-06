import json
from typing import Union

class AlpacaPromptTemplate:
    def __init__(self, is_multling:bool=False, file_name:str="templates/alpaca_template.json"):
        """
        Args:
            is_multling: True if a dataset consists of multiple languages.
        """

        with open(file_name) as fp:
            self.template = json.load(fp)
        if is_multling:
            self.template["prompt_input"] = self.template["prompt_input"].replace(" You MUST speak in Japanese.", "")
            self.template["prompt_no_input"] = self.template["prompt_no_input"].replace(" You MUST speak in Japanese.", "")
        

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

class SimplePromptTemplate:
    def __init__(self, file_name:str = "templates/simple_template.json"):
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
