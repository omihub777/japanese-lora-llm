import argparse
import os
import json
from typing import List, Dict, Any

import tqdm
import torch
from peft import PeftModel, PeftConfig
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM
)

from prompts import AlpacaPromptTemplate
from constants import (
    CAUSAL_LM_MODELS,
)


def load_test_model(model_name:str):
    trust_remote_code = "mpt" in model_name

    if "lora" in model_name or "LoRA" in model_name:
        config = PeftConfig.from_pretrained(model_name)
        is_causal = config.base_model_name_or_path in CAUSAL_LM_MODELS
        model_cls = AutoModelForCausalLM if is_causal else AutoModelForSeq2SeqLM

        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        model = model_cls.from_pretrained(
            config.base_model_name_or_path, 
            load_in_8bit=True, 
            device_map={"":0},
            trust_remote_code=trust_remote_code,
        )
        model = PeftModel.from_pretrained(model, model_name, device_map={"":0})
    else:
        is_causal = model_name in CAUSAL_LM_MODELS
        model_cls = AutoModelForCausalLM if is_causal else AutoModelForSeq2SeqLM

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = model_cls.from_pretrained(
            model_name, 
            load_in_8bit=True, 
            device_map={"":0},
            trust_remote_code=trust_remote_code,
            )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token_id
    
    model.eval()
    return model, tokenizer, is_causal


class QualitativeTester:
    def __init__(self, testcase_path:str, model_name:str):
        with open(testcase_path) as f:
            self.testcases = json.load(f)

        self.prompt = AlpacaPromptTemplate()
        self.model, self.tokenizer, self.is_causal = load_test_model(model_name=model_name)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def test(self, gen_hp:Dict[str, Any])->List[Dict]:
        responses = []
        for testcase in tqdm.tqdm(self.testcases):
            test_prompt = self.prompt.generate_prompt(instruction=testcase["instruction"])
            response = self.generate(test_prompt, gen_hp=gen_hp)
            responses.append(
                {
                "instruction":testcase["instruction"],
                "response":response
                }
            )
        return responses

    @torch.no_grad()
    def generate(self, input_text:str, gen_hp:Dict[str, Any])->str:
        with torch.cuda.amp.autocast():
            input_tokens = self.tokenizer(
                input_text,
                truncation=True,
                max_length=1024,
                return_tensors="pt",
            ).to(self.device)
            gen_tokens = self.model.generate(
                input_ids=input_tokens["input_ids"],
                attention_mask=input_tokens["attention_mask"],
                **gen_hp
            )
            gen_text = self.tokenizer.decode(
                gen_tokens[0], skip_special_tokens=True
            )
        gen_text = gen_text[len(input_text):] if self.is_causal else gen_text
        return gen_text



def main(model_name:str, ds_name:str, testcase_path:str, gen_hp:Dict):
    tester = QualitativeTester(testcase_path=testcase_path, model_name=model_name)
    responses = tester.test(gen_hp)

    save_dir = os.path.join("dialogues",ds_name,model_name.split("/")[-1])
    save_path = os.path.join(save_dir, testcase_path.split("/")[-1].split(".")[0]+"_result.json")
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    with open(save_path, "w") as fw:
        fw.write("[\n")
        for response in responses[:-1]:
            json.dump(response, fw, ensure_ascii=False)
            fw.write(",\n")
        json.dump(responses[-1], fw, ensure_ascii=False)
        fw.write(f"\n]")
    print("\n"+save_path)
    print(responses)
            
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds_name", choices=["alpaca", "dolly", "original"], default="alpaca")
    parser.add_argument("--testcase_path", default="testcases/instruction_testcases_ja.jsonl")
    args = parser.parse_args()

    gen_hp = { # LLaMA-Precise preset
        "do_sample": True,
        "top_k": 40,
        "top_p": 0.1,
        "temperature":0.7,
        "repetition_penalty": 1.18,
        "max_new_tokens": 256,
    }

    if args.ds_name=="alpaca":
        model_names = [ # alpaca
            "weights/abeja_gpt_neox_japanese_2.7b_alpaca_cleaned_ja_lora_int8_20230507_183231",
            "weights/EleutherAI_pythia_6.9b_deduped_alpaca_cleaned_ja_lora_int8_20230507_120552",
            "weights/togethercomputer_RedPajama_INCITE_Base_7B_v0.1_alpaca_cleaned_ja_lora_int8_20230507_222908",
            "weights/yahma_llama_7b_hf_alpaca_cleaned_ja_lora_int8_20230508_050313"
        ]
    elif args.ds_name=="dolly":
        model_names = [ # dolly
        ]
    elif args.ds_name=="original":
        model_names = [ # original models (i.e. no FT)
        ]
    else:
        raise ValueError()

    for model_name in model_names:
        main(model_name=model_name, ds_name=args.ds_name, testcase_path=args.testcase_path, gen_hp=gen_hp)