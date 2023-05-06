CAUSAL_LM_MODELS = [
    "mosaicml/mpt-7b",
    "yahma/llama-7b-hf",
    "EleutherAI/pythia-6.9b-deduped",
    "togethercomputer/RedPajama-INCITE-Base-7B-v0.1",
    "abeja/gpt-neox-japanese-2.7b",
]

LORA_TARGET_MODULES_DICT = {
    "mosaicml/mpt-7b": ["Wqkv", "out_proj"], 
    "bigscience/mt0-xl": ["q","v"],
    "yahma/llama-7b-hf": ["q_proj","k_proj", "v_proj", "o_proj"],
    "EleutherAI/pythia-6.9b-deduped": ["query_key_value", "dense"],
    "abeja/gpt-neox-japanese-2.7b": ["query_key_value", "dense"],
    "togethercomputer/RedPajama-INCITE-Base-7B-v0.1": ["query_key_value", "dense"]
}
