#!/bin/bash

python train.py --model_name "EleutherAI/pythia-6.9b-deduped"
python train.py --model_name "abeja/gpt-neox-japanese-2.7b"
python train.py --model_name "togethercomputer/RedPajama-INCITE-Base-7B-v0.1"
python train.py --model_name "yahma/llama-7b-hf"
