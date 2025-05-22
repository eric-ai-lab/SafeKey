MODEL_CONFIG = {
    # Baseline
    "DeepSeek-R1-Distill-Qwen-1.5B": {
        "model_path": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "n_gpu": 1,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    "DeepSeek-R1-Distill-Qwen-7B": {
        "model_path": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "n_gpu": 1,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    "DeepSeek-R1-Distill-Llama-8B": {
        "model_path": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "n_gpu": 1,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    "DeepSeek-R1-Distill-Qwen-14B": {
        "model_path": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "n_gpu": 1,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    "DeepSeek-R1-Distill-Qwen-32B": {
        "model_path": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "n_gpu": 1,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    "Qwen2.5-1.5B-Instruct": {
        "model_path": "Qwen/Qwen2.5-1.5B-Instruct",
        "n_gpu": 1,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    "Qwen2.5-7B-Instruct": {
        "model_path": "Qwen/Qwen2.5-7B-Instruct",
        "n_gpu": 1,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    "Llama-3.1-8B-Instruct": {
        "model_path": "meta-llama/Llama-3.1-8B",
        "n_gpu": 1,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    "Qwen2.5-14B-Instruct": {
        "model_path": "Qwen/Qwen2.5-14B-Instruct",
        "n_gpu": 1,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    "Qwen2.5-32B-Instruct": {
        "model_path": "Qwen/Qwen2.5-32B-Instruct",
        "n_gpu": 1,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    # STAR-1 model
    "DeepSeek-R1-Distill-Qwen-1.5B-STAR1": {
        "model_path": "UCSC-VLAA/STAR1-R1-Distill-1.5B",
        "n_gpu": 1,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    "DeepSeek-R1-Distill-Qwen-7B-STAR1": {
        "model_path": "UCSC-VLAA/STAR1-R1-Distill-7B",
        "n_gpu": 1,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    "DeepSeek-R1-Distill-Llama-8B-STAR1": {
        "model_path": "UCSC-VLAA/STAR1-R1-Distill-8B",
        "n_gpu": 1,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },

    "DeepSeek-R1-Distill-Qwen-14B-STAR1": {
        "model_path": "UCSC-VLAA/STAR1-R1-Distill-14B",
        "n_gpu": 1,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    "DeepSeek-R1-Distill-Qwen-32B-STAR1": {
        "model_path": "UCSC-VLAA/STAR1-R1-Distill-32B",
        "n_gpu": 2,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    # 2k SFT and SafeKey models
    "DeepSeek-R1-Distill-Qwen-7B-STAR_mix2_sft": {
        "model_path": "kzhou35/SFT-7B",
        "n_gpu": 2,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    "DeepSeek-R1-Distill-Qwen-7B-STAR_mix2_safekey": {
        "model_path": "kzhou35/SafeKey-7B",
        "n_gpu": 2,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    "DeepSeek-R1-Distill-Llama-8B-STAR_mix2_sft": {
        "model_path": "kzhou35/SFT-8B",
        "n_gpu": 2,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    
    "DeepSeek-R1-Distill-Llama-8B-STAR_mix2_safekey": {
        "model_path": "kzhou35/SafeKey-8B",
        "n_gpu": 2,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    "DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_sft": {
        "model_path": "kzhou35/SFT-14B",
        "n_gpu": 1,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    "DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_safekey": {
        "model_path": "kzhou35/SafeKey-14B",
        "n_gpu": 1,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
}

DEFAULT_GEN_CONFIG = {
    'system': True,
    'temperature': 0,
    'topp': 1,
    'topk': -1,
    'max_tokens': 8000,
    'repeat_n': 1,
}

RSM_LIST = list(MODEL_CONFIG.keys())

EVAL_DATA = [
    'strongreject',
    'wildjailbreak',
    'jbbbehaviours',
    'jbbbehaviours_benign',
    'wildchat', 
    'xstest',
    'airbench',
    'phtest',
    'prefilling',
    'prefilling_8b',
    "door_multi",
    'star1',
    'sum',
    'key_better_8b',
]


