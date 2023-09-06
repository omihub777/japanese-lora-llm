import json
import os
from typing import List, Union
import datetime


import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM, 
    AutoTokenizer, 
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import (
    prepare_model_for_int8_training, 
    get_peft_model, 
    get_peft_model_state_dict,
    LoraConfig, 
    TaskType,
)
from datasets import load_dataset

from prompts import AlpacaPromptTemplate
from constants import (
    CAUSAL_LM_MODELS,
    LORA_TARGET_MODULES_DICT
)

# When you add a new Base-LLM, you need do the following:
# 1. Add the `model_name` to `CAUSAL_LM_MODELS` if it is a causal language model.
# 2. Add a `target_modules` to `LORA_TARGET_MODULES_DICT` to specify the modules to be decomposed.

# Prepare model
def prepare_model_tokenizer(model_name:str, lora:bool=True, int8:bool=True):
    use_gradient_checkpointing = "mpt" not in model_name
    device_map = {"":0} if "mpt" in model_name else "auto"
    trust_remote_code = "mpt" in model_name
    
    is_causal = model_name in CAUSAL_LM_MODELS

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast= "rinna" not in model_name, #There might be a better way 
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = (0)
        # tokenizer.pad_token_id = (tokenizer.unk_token_id)
        # tokenizer.pad_token = tokenizer.unk_token
    if is_causal:
        tokenizer.padding_side = "left"
    else:
        tokenizer.padding_side = "right"

    model_cls = AutoModelForCausalLM if is_causal else AutoModelForSeq2SeqLM
    model = model_cls.from_pretrained(
        model_name,
        load_in_8bit=int8,
        torch_dtype=torch.float16,
        device_map=device_map,

        trust_remote_code=trust_remote_code
    )
    if int8:
        model = prepare_model_for_int8_training(model, use_gradient_checkpointing=use_gradient_checkpointing)
    if lora:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=LORA_TARGET_MODULES_DICT[model_name],
            lora_dropout=0.05,
            bias="none",
            task_type = TaskType.CAUSAL_LM if is_causal else TaskType.SEQ_2_SEQ_LM,
        )
        model = get_peft_model(model, lora_config)
    return model, tokenizer


def train(
    # model/data params
    model_name:str = "yahma/llama-7b-hf",
    data_path: str = "datasets/alpaca_cleaned_ja.json",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: float = 3.0,
    learning_rate: float = 3e-4,
    cutoff_len: int = 512,
    val_set_size: int = 2000,
    eval_steps: int = 200,
    # llm hyperparams
    add_eos_token: bool = False,
    group_by_length: bool = True,  # faster, but produces an odd training loss curve
    # wandb params
    use_wandb: bool = True,
    wandb_project: str = "japanese-lora-llm",
    wandb_log_model: str = "false",  # options: false | true
):
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    gradient_accumulation_steps = batch_size // micro_batch_size

    is_multling = "guanaco" in data_path
    if "Llama-2" in model_name:
        prompter = AlpacaPromptTemplate(is_multling=is_multling, file_name="templates/alpaca_llama2_template.json")
    else:
        prompter = AlpacaPromptTemplate(is_multling=is_multling)
    wandb_run_name=f'{model_name.replace("/","_").replace("-","_")}_{data_path.split("/")[-1].split(".")[0].replace("-","_")}'

    if use_wandb:
        os.environ["WANDB_PROJECT"] = wandb_project
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    output_dir = "weights/"+wandb_run_name + f"_lora_int8_{timestamp}"
    model, tokenizer = prepare_model_tokenizer(model_name)

    def tokenize(prompt:str, text_target:str=None, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
            text_target=text_target,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        if text_target is None:
            result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        is_causal = model_name in CAUSAL_LM_MODELS

        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            label = data_point["output"] if is_causal else None,
        )
        tokenized_full_prompt = tokenize(
            full_prompt,
            text_target=data_point["output"] if not is_causal else None
        )
        return tokenized_full_prompt

    data = load_dataset("json", data_files=data_path)

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    train_val = data["train"].train_test_split(test_size=val_set_size, shuffle=True, seed=42)
    train_data = (train_val["train"].shuffle().map(generate_and_tokenize_prompt))
    val_data = (train_val["test"].shuffle().map(generate_and_tokenize_prompt))

    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=True,
            logging_steps=1,
            optim="adamw_torch",
            evaluation_strategy="steps",
            save_strategy="no",
            eval_steps=eval_steps,
            output_dir=output_dir,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
        ).__get__(model, type(model))

    model = torch.compile(model)

    if model_name=="rinna/japanese-gpt-1b": # temporary fix for weird bugs regarding device and dtype...
        transformer_mods = model._orig_mod.base_model.model.transformer.h
        for transformer_mod in transformer_mods:
            transformer_mod.attn.bias = transformer_mod.attn.bias.to(torch.bool)
            transformer_mod.attn.c_attn.lora_A.weight.data = transformer_mod.attn.c_attn.lora_A.weight.data.to(model.device)
            transformer_mod.attn.c_attn.lora_B.weight.data = transformer_mod.attn.c_attn.lora_B.weight.data.to(model.device)


    trainer.train()
    model.save_pretrained(output_dir)


if __name__ in "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="datasets/alpaca_cleaned_ja.json")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=float, default=3.0)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--val_set_size", type=int, default=2000)
    parser.add_argument("--micro_batch_size", type=int, default=4)

    args = parser.parse_args()

    train(
        data_path=args.data_path,
        model_name=args.model_name, 
        batch_size=args.batch_size, 
        micro_batch_size=args.micro_batch_size, 
        num_epochs=args.num_epochs,
        eval_steps=args.eval_steps,
        val_set_size=args.val_set_size,
        )