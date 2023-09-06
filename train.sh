#!/bin/bash

python train.py --data_path "datasets/alpaca_cleaned_ja.json" --model_name "meta-llama/Llama-2-7b-chat-hf"
python train.py --data_path "datasets/alpaca_cleaned_ja.json" --model_name "meta-llama/Llama-2-13b-chat-hf"
python test.py --ds_name original
# python train.py --data_path "datasets/msmarco_ja.json" --model_name "abeja/gpt-neox-japanese-2.7b"
# python train.py --data_path "datasets/msmarco_ja.json" --model_name "cyberagent/open-calm-7b"
# python train.py --data_path "datasets/msmarco_ja.json" --model_name "EleutherAI/pythia-12b-deduped"
# python train.py --data_path "datasets/msmarco_ja.json" --model_name "yahma/llama-13b-hf"

# python train.py --data_path "datasets/msmarco_ja.json" --model_name "retrieva-jp/t5-xl"
# python train.py --data_path "datasets/msmarco_ja.json" --model_name "bigscience/mt0-xl"
# python train.py --data_path "datasets/msmarco_ja.json" --model_name "bigscience/mt0-xxl" --micro_batch_size 1
# python train.py --data_path "datasets/msmarco_ja.json" --model_name "rinna/japanese-gpt-neox-3.6b"

# python train.py --model_name cyberagent/open-calm-3b --data_path "datasets/alpaca_cleaned_ja.json" 
# python train.py --model_name cyberagent/open-calm-3b --data_path "datasets/databricks-dolly-15k-ja-deepl.json" --val_set_size 700 --eval_steps 70 
# python train.py --model_name cyberagent/open-calm-3b --data_path "datasets/guanaco_non_chat-utf8.json" --num_epochs 1.0 


# python train.py --model_name "rinna/japanese-gpt-neox-3.6b" --data_path "datasets/alpaca_cleaned_ja.json" 
# python train.py --model_name "rinna/japanese-gpt-neox-3.6b" --data_path "datasets/databricks-dolly-15k-ja-deepl.json" --val_set_size 700 --eval_steps 70 
# python train.py --model_name "rinna/japanese-gpt-neox-3.6b" --data_path "datasets/guanaco_non_chat-utf8.json" --num_epochs 1.0 
# python train.py --model_name "EleutherAI/pythia-2.8b-deduped" --data_path "datasets/alpaca_cleaned_ja.json"
# python train.py --model_name "EleutherAI/pythia-2.8b-deduped" --data_path "datasets/databricks-dolly-15k-ja-deepl.json" --val_set_size 700 --eval_steps 70
# python train.py --model_name "EleutherAI/pythia-2.8b-deduped" --data_path "datasets/guanaco_non_chat-utf8.json" --num_epochs 1.0

# python train.py --model_name cyberagent/open-calm-7b --data_path "datasets/alpaca_cleaned_ja.json" 
# python train.py --model_name cyberagent/open-calm-7b --data_path "datasets/databricks-dolly-15k-ja-deepl.json" --val_set_size 700 --eval_steps 70 
# python train.py --model_name cyberagent/open-calm-7b --data_path "datasets/guanaco_non_chat-utf8.json" --num_epochs 1.0 
# python train.py --model_name "retrieva-jp/t5-xl" --data_path "datasets/alpaca_cleaned_ja.json"
# python train.py --model_name "retrieva-jp/t5-xl" --data_path "datasets/databricks-dolly-15k-ja-deepl.json" --val_set_size 700 --eval_steps 70
# python train.py --model_name "retrieva-jp/t5-xl" --data_path "datasets/guanaco_non_chat-utf8.json" --num_epochs 1.0
# python train.py --model_name "EleutherAI/pythia-12b-deduped" --data_path "datasets/guanaco_non_chat-utf8.json" --num_epochs 1.0
