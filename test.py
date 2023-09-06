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

from prompts import AlpacaPromptTemplate, SimplePromptTemplate
from constants import (
    CAUSAL_LM_MODELS,
)


def load_test_model(model_name:str):
    trust_remote_code = "mpt" in model_name

    if "lora" in model_name or "LoRA" in model_name:
        config = PeftConfig.from_pretrained(model_name)
        is_causal = config.base_model_name_or_path in CAUSAL_LM_MODELS
        model_cls = AutoModelForCausalLM if is_causal else AutoModelForSeq2SeqLM

        tokenizer = AutoTokenizer.from_pretrained(
            config.base_model_name_or_path,
            use_fast= "rinna" not in config.base_model_name_or_path,
        )
        model = model_cls.from_pretrained(
            config.base_model_name_or_path, 
            load_in_8bit=True, 
            device_map={"":0},
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(model, model_name, device_map={"":0})

    else:
        is_causal = model_name in CAUSAL_LM_MODELS
        model_cls = AutoModelForCausalLM if is_causal else AutoModelForSeq2SeqLM

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast="rinna" not in model_name,
        )
        model = model_cls.from_pretrained(
            model_name, 
            load_in_8bit=True, 
            device_map={"":0},
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.float16,
            )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token_id
    
    model.eval()
    return model, tokenizer, is_causal


class QualitativeTester:
    def __init__(self, testcase_path:str, model_name:str, prompt_type:str="alpaca"):
        with open(testcase_path) as f:
            self.testcases = json.load(f)

        if prompt_type=="simple":
            if "Llama-2" in model_name:
                self.prompt = SimplePromptTemplate(file_name="templates/simple_llama2_template.json")
            else:
                self.prompt = SimplePromptTemplate()
        elif prompt_type=="alpaca":
            if "Llama-2" in model_name:
                self.prompt = AlpacaPromptTemplate(file_name="templates/alpaca_llama2_template.json")
            else:
                self.prompt = AlpacaPromptTemplate()
        else:
            raise ValueError(f"No such template: {prompt_type}")
        self.model, self.tokenizer, self.is_causal = load_test_model(model_name=model_name)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def test(self, gen_hp:Dict[str, Any])->List[Dict]:
        responses = []
        for testcase in tqdm.tqdm(self.testcases):
            if "input" in list(testcase.keys()):
                test_prompt = self.prompt.generate_prompt(instruction=testcase["instruction"], input=testcase["input"])            
            else:
                test_prompt = self.prompt.generate_prompt(instruction=testcase["instruction"])
            response = self.generate(test_prompt, gen_hp=gen_hp)
            result = {
                "instruction":testcase["instruction"],
                "response":response
                }
            print(result["instruction"])
            print(result["response"])
            responses.append(result)
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
        gen_text = gen_text[len(input_text)-1:] if self.is_causal else gen_text
        return gen_text



def main(model_name:str, ds_name:str, testcase_path:str, gen_hp:Dict, is_save:bool=False, prompt_type:str="alpaca"):
    tester = QualitativeTester(testcase_path=testcase_path, model_name=model_name, prompt_type=prompt_type)
    if "rinna" in model_name:
        gen_hp.update({
        "pad_token_id":tester.tokenizer.pad_token_id,
        "bos_token_id":tester.tokenizer.bos_token_id,
        "eos_token_id":tester.tokenizer.eos_token_id,
        })

    responses = tester.test(gen_hp)
    if is_save:
        save_dir = os.path.join("dialogues",ds_name,model_name.split("/")[-1])
        save_path = os.path.join(save_dir, testcase_path.split("/")[-1].split(".")[0]+"_result.json")
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        # with open(save_path, "w") as fw:
        #     fw.write("[\n")
        #     for response in responses[:-1]:
        #         json.dump(response, fw, ensure_ascii=False)
        #         fw.write(",\n")
        #     json.dump(responses[-1], fw, ensure_ascii=False)
        #     fw.write(f"\n]")
        with open(save_path, "w", encoding="utf-8") as fw:
            json.dump(responses, fw, ensure_ascii=False)
        print("\n"+save_path)
    for res in responses:
        print(res)
            
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds_name", choices=["alpaca", "dolly", "guanaco" ,"original"], required=True)
    parser.add_argument("--prompt_type", choices=["alpaca", "simple"], default="alpaca")
    parser.add_argument("--testcase_path", default="testcases/instruction_testcases_ja.jsonl")
    parser.add_argument("--no_save", action="store_true")
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
        model_names = [ 
            # "weights/abeja_gpt_neox_japanese_2.7b_alpaca_cleaned_ja_lora_int8_20230507_183231",
            # "weights/EleutherAI_pythia_6.9b_deduped_alpaca_cleaned_ja_lora_int8_20230507_120552",
            # "weights/togethercomputer_RedPajama_INCITE_Base_7B_v0.1_alpaca_cleaned_ja_lora_int8_20230507_222908",
            # "weights/yahma_llama_7b_hf_alpaca_cleaned_ja_lora_int8_20230508_050313",
            # "weights/yahma_llama_13b_hf_alpaca_cleaned_ja_lora_int8_20230508_232828",
            # "weights/rinna_japanese_gpt_1b_alpaca_cleaned_ja_lora_int8_20230512_014739",
            # "weights/retrieva_jp_t5_xl_alpaca_cleaned_ja_lora_int8_20230516_000130",
            # "weights/EleutherAI_pythia_12b_deduped_alpaca_cleaned_ja_lora_int8_20230514_180727",
            # "weights/cyberagent_open_calm_7b_alpaca_cleaned_ja_lora_int8_20230517_111546",
            # "weights/rinna_japanese_gpt_neox_3.6b_alpaca_cleaned_ja_lora_int8_20230518_102931",
            # "weights/EleutherAI_pythia_2.8b_deduped_alpaca_cleaned_ja_lora_int8_20230519_022753",
            # "weights/cyberagent_open_calm_3b_alpaca_cleaned_ja_lora_int8_20230524_233925",
        ]
    elif args.ds_name=="dolly":
        model_names = [
            # "weights/abeja_gpt_neox_japanese_2.7b_databricks_dolly_15k_ja_deepl_lora_int8_20230508_143708",
            # "weights/EleutherAI_pythia_6.9b_deduped_databricks_dolly_15k_ja_deepl_lora_int8_20230508_120504",
            # "weights/togethercomputer_RedPajama_INCITE_Base_7B_v0.1_databricks_dolly_15k_ja_deepl_lora_int8_20230508_155710",
            # "weights/yahma_llama_7b_hf_databricks_dolly_15k_ja_deepl_lora_int8_20230508_183230",
            # "weights/yahma_llama_13b_hf_databricks_dolly_15k_ja_deepl_lora_int8_20230509_082716",
            # "weights/rinna_japanese_gpt_1b_databricks_dolly_15k_ja_deepl_lora_int8_20230512_024629",
            # "weights/retrieva_jp_t5_xl_databricks_dolly_15k_ja_deepl_lora_int8_20230516_035259",
            # "weights/EleutherAI_pythia_12b_deduped_databricks_dolly_15k_ja_deepl_lora_int8_20230515_193036",
            # "weights/cyberagent_open_calm_7b_databricks_dolly_15k_ja_deepl_lora_int8_20230517_161809",
            # "weights/rinna_japanese_gpt_neox_3.6b_databricks_dolly_15k_ja_deepl_lora_int8_20230518_141905",
            # "weights/EleutherAI_pythia_2.8b_deduped_databricks_dolly_15k_ja_deepl_lora_int8_20230519_060836",
            # "weights/cyberagent_open_calm_3b_databricks_dolly_15k_ja_deepl_lora_int8_20230525_023049",
        ]
    elif args.ds_name=="guanaco":
        model_names = [
            # "weights/abeja_gpt_neox_japanese_2.7b_guanaco_non_chat_utf8_lora_int8_20230509_115054",
            # "weights/EleutherAI_pythia_6.9b_deduped_guanaco_non_chat_utf8_lora_int8_20230510_010727",
            # "weights/togethercomputer_RedPajama_INCITE_Base_7B_v0.1_guanaco_non_chat_utf8_lora_int8_20230510_120204",
            # "weights/yahma_llama_7b_hf_guanaco_non_chat_utf8_lora_int8_20230510_231027",
            # "weights/yahma_llama_13b_hf_guanaco_non_chat_utf8_lora_int8_20230511_081104",
            # "weights/rinna_japanese_gpt_1b_guanaco_non_chat_utf8_lora_int8_20230512_030943",
            # "weights/retrieva_jp_t5_xl_guanaco_non_chat_utf8_lora_int8_20230516_051522",
            # "weights/EleutherAI_pythia_12b_deduped_guanaco_non_chat_utf8_lora_int8_20230516_124120",
            # "weights/cyberagent_open_calm_7b_guanaco_non_chat_utf8_lora_int8_20230517_181437",
            # "weights/rinna_japanese_gpt_neox_3.6b_guanaco_non_chat_utf8_lora_int8_20230518_154418",
            # "weights/EleutherAI_pythia_2.8b_deduped_guanaco_non_chat_utf8_lora_int8_20230519_073323",
            # "weights/cyberagent_open_calm_3b_guanaco_non_chat_utf8_lora_int8_20230525_033714",
        ]
    elif args.ds_name=="original":
        model_names = [ # original models
            # "EleutherAI/pythia-2.8b-deduped",
            # "EleutherAI/pythia-6.9b-deduped",
            # "EleutherAI/pythia-12b-deduped",
            # "abeja/gpt-neox-japanese-2.7b",
            # "retrieva-jp/t5-xl",
            # "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-2-13b-chat-hf",
            # "cyberagent/open-calm-3b",
            # "cyberagent/open-calm-7b",
            # "rinna/japanese-gpt-neox-3.6b",
            # "togethercomputer/RedPajama-INCITE-Base-7B-v0.1",
            # "yahma/llama-7b-hf",
            # "yahma/llama-13b-hf",
        ]
        args.prompt_type="simple"
    else:
        raise ValueError()

    for model_name in model_names:
        main(
            model_name=model_name, 
            ds_name=args.ds_name, 
            testcase_path=args.testcase_path, 
            gen_hp=gen_hp,
            is_save= False if args.no_save else True,
            prompt_type = args.prompt_type
        )