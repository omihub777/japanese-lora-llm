CAUSAL_LM_MODELS = [
    "mosaicml/mpt-7b",
    "yahma/llama-7b-hf",
    "yahma/llama-13b-hf",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "EleutherAI/pythia-6.9b-deduped",
    "EleutherAI/pythia-12b-deduped",
    "EleutherAI/pythia-2.8b-deduped",
    "togethercomputer/RedPajama-INCITE-Base-7B-v0.1",
    "abeja/gpt-neox-japanese-2.7b",
    "decapoda-research/llama-7b-hf",
    "decapoda-research/llama-13b-hf",
    "rinna/japanese-gpt-1b",
    "cyberagent/open-calm-3b",
    "cyberagent/open-calm-7b",
    "rinna/japanese-gpt-neox-3.6b",
]

LORA_TARGET_MODULES_DICT = {
    "mosaicml/mpt-7b": ["Wqkv", "out_proj"], 
    "bigscience/mt0-xl": ["q","v","SelfAttention.o","EncDecAttention.o","SelfAttention.k", "EncDecAttention.k"], # ["k"] causes NoValueError
    # "bigscience/mt0-xxl": ["q","v","SelfAttention.o","EncDecAttention.o","SelfAttention.k", "EncDecAttention.k"], # ["k"] causes NoValueError
    "bigscience/mt0-xxl": ["q","v"], 
    "retrieva-jp/t5-xl": ["q","v","SelfAttention.o","EncDecAttention.o","SelfAttention.k", "EncDecAttention.k"],
    "yahma/llama-7b-hf": ["q_proj","k_proj", "v_proj", "o_proj"],
    "meta-llama/Llama-2-7b-chat-hf": ["q_proj","k_proj", "v_proj", "o_proj"],
    "meta-llama/Llama-2-13b-chat-hf": ["q_proj","k_proj", "v_proj", "o_proj"],
    "yahma/llama-13b-hf": ["q_proj","k_proj", "v_proj", "o_proj"],
    "EleutherAI/pythia-2.8b-deduped": ["query_key_value", "dense"],
    "EleutherAI/pythia-6.9b-deduped": ["query_key_value", "dense"],
    "EleutherAI/pythia-12b-deduped": ["query_key_value", "dense"],
    "abeja/gpt-neox-japanese-2.7b": ["query_key_value", "dense"],
    "togethercomputer/RedPajama-INCITE-Base-7B-v0.1": ["query_key_value", "dense"],
    "rinna/japanese-gpt-1b":["c_attn"],
    "rinna/japanese-gpt-neox-3.6b":["query_key_value", "dense"],
    "cyberagent/open-calm-3b":["query_key_value", "dense"],
    "cyberagent/open-calm-7b":["query_key_value", "dense"],
}
