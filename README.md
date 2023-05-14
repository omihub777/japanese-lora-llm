# Japanese LoRA-tuned LLMs :racehorse: :jp:
A collection of Japanese LoRA-tuned LLMs.

## Environment

* Python: 3.10.6
* All models trained on a single NVIDIA GeForce RTX 4090 (24GB)

## Quickstart

```
make setup
python app.py
```

then, access to http://localhost:8080

## How to

* To lora-tune a model, run the following command.
```
python train.py --data_path "datasets/alpaca_cleaned_ja.json" --model_name "abeja/gpt-neox-japanese-2.7b"
```

* To add a new model to lora-tune, in `constants.py`, you need
    * add target modules to lora-tune to `LORA_TARGET_MODULES_DICT`
    * only if the model is a causal model (i.e. decoder only model), add a path (or a model name on HuggingFace Hub) to `CAUSAL_LM_MODELS`


## Progress

* :construction: : WIP
* :white_check_mark: : Evaluation is done

|    |No Tuning|Alpaca-ja|Dolly-ja|Guanaco(*)|
|:--:|:--:|:--:|:--:|:--:|
|LLaMA-7B|:white_check_mark: |:white_check_mark: |:white_check_mark: | :white_check_mark: |
|LLaMA-13B|:white_check_mark: |:white_check_mark: |:white_check_mark: | :white_check_mark: |
|RedPajama-INCITE-7B|:white_check_mark: |:white_check_mark: | :white_check_mark: | :white_check_mark: |
|Pythia-6.9B|:white_check_mark: |:white_check_mark:|:white_check_mark:| :white_check_mark: |
|GPT-NeoX-Japanese-2.7B|:white_check_mark: |:white_check_mark: |:white_check_mark: | :white_check_mark: |
|Japanese-GPT-1B (**)|:construction: |:construction: |:construction: | :construction: |
<!-- |MPT-7B| :construction: | :construction: | :construction: | :construction: | -->

(*) For Guanaco dataset, we use in multilingual in deference to the creator's [intention](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset).
(**) Training script might be something wrong. work in progress.

## TODO

- [x] Train / Eval Guanaco
- [x] Compare results so far
- [ ] Train rinna/japanese-gpt-1b
- [ ] Add a generation interface
- [ ] Add docstrings for each func/class
- [ ] Add a notebook (train/generate) for Colab
- [ ] Upload delta weights(=lora adapters) on Hugging Face Hub
- [ ] Support for MPT-7b
- [ ] Support for LoRA-tuning for emb/head layers

## Results

Please refer to `dialogues/`, to see all evaluation results.

### Alpaca

* "5かける7は？"
    * LLaMA-7B

### Dolly

### Guanaco

## Acknowledgement
* Code
    * [tloen/alpaca-lora](https://github.com/tloen/alpaca-lora)

* Dataset
    * [shi3z/alpaca_ja](https://github.com/shi3z/alpaca_ja)
    * [kunishou/databricks-dolly-15k-ja](https://huggingface.co/datasets/kunishou/databricks-dolly-15k-ja)
    * [JosephusCheung/GuanacoDataset](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset)

* Pretrained Model
    * [mosaicml/mpt-7b](https://huggingface.co/mosaicml/mpt-7b)
    * [yahma/llama-7b-hf](https://huggingface.co/yahma/llama-7b-hf)
    * [yahma/llama-13b-hf](https://huggingface.co/yahma/llama-13b-hf)
    * [EleutherAI/pythia-6.9b-deduped](https://huggingface.co/EleutherAI/pythia-6.9b-deduped)
    * [togethercomputer/RedPajama-INCITE-Base-7B-v0.1](https://huggingface.co/togethercomputer/RedPajama-INCITE-Base-7B-v0.1)
    * [abeja/gpt-neox-japanese-2.7b](https://huggingface.co/abeja/gpt-neox-japanese-2.7b)