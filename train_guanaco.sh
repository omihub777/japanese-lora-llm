#!/bin/bash

python train.py --data_path "datasets/guanaco_non_chat-utf8.json" --num_epochs 1.0 --model_name "EleutherAI/pythia-6.9b-deduped"
python train.py --data_path "datasets/guanaco_non_chat-utf8.json" --num_epochs 1.0 --model_name "togethercomputer/RedPajama-INCITE-Base-7B-v0.1"
python train.py --data_path "datasets/guanaco_non_chat-utf8.json" --num_epochs 1.0 --model_name "yahma/llama-7b-hf"
python train.py --data_path "datasets/guanaco_non_chat-utf8.json" --num_epochs 1.0 --model_name "yahma/llama-13b-hf"
