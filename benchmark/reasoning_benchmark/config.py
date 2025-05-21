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
    
    # add your model here
    "DeepSeek-R1-Distill-Qwen-1.5B-STAR_mix2": {
        "model_path": "/home/ubuntu/mnt/kaiwen/STAR-1/data/models/1.5b_sft_mix_2k/STAR-1/DeepSeek-R1-Distill-Qwen-1.5B/think_flag1/checkpoint-4-595/tfmr",
        "n_gpu": 1,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
     "DeepSeek-R1-Distill-Qwen-1.5B-STAR_mix2_k_g": {
        "model_path": "/home/ubuntu/mnt/kaiwen/STAR-1/data/models/1.5b_sft_mix_2k/STAR-1/DeepSeek-R1-Distill-Qwen-1.5B/think_flag1/checkpoint-4-595/tfmr",
        "n_gpu": 1,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    "DeepSeek-R1-Distill-Qwen-7B-STAR5": {
        "model_path": "/home/ubuntu/mnt/kaiwen/STAR-1/data/models/sft_5k/STAR-1/DeepSeek-R1-Distill-Qwen-7B/think_flag1/checkpoint-4-3120/tfmr",
        "n_gpu": 1,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    "DeepSeek-R1-Distill-Qwen-7B-STAR_mix2": {
        "model_path": "/home/ubuntu/mnt/kaiwen/STAR-1/data/models/sft_mix_2k/STAR-1/DeepSeek-R1-Distill-Qwen-7B/think_flag1/checkpoint-4-595/tfmr",
        "n_gpu": 1,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    "DeepSeek-R1-Distill-Qwen-7B-STAR_mix2_k": {
        "model_path": "/home/ubuntu/mnt/kaiwen/STAR-1/data/models/7b_sft_mix_2k_k_loss_2e/STAR-1/DeepSeek-R1-Distill-Qwen-7B/think_flag1/checkpoint-4-595/tfmr",
        "n_gpu": 1,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    "DeepSeek-R1-Distill-Qwen-7B-STAR_mix2_g": {
        "model_path": "/home/ubuntu/mnt/kaiwen/STAR-1/data/models/7b_sft_mix_2k_gate_loss_2e/STAR-1/DeepSeek-R1-Distill-Qwen-7B/think_flag1/checkpoint-4-595/tfmr_cleaned",
        "n_gpu": 1,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    "DeepSeek-R1-Distill-Qwen-7B-STAR_mix2_gate_theta0": {
        "model_path": "/home/ubuntu/mnt/kaiwen/STAR-1/data/models/7b_sft_mix_2k_10e_g_theta0/STAR-1/DeepSeek-R1-Distill-Qwen-7B/think_flag1/checkpoint-9-1190/tfmr_cleaned",
        "n_gpu": 2,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    "DeepSeek-R1-Distill-Qwen-7B-STAR_mix2_k_gate": {
        "model_path": "/home/ubuntu/mnt/kaiwen/STAR-1/data/models/7b_sft_mix_2k_k_gate_loss_2e/STAR-1/DeepSeek-R1-Distill-Qwen-7B/think_flag1/checkpoint-4-595/tfmr_cleaned",
        "n_gpu": 1,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    "DeepSeek-R1-Distill-Qwen-7B-STAR_mix2_gate_full_ctx": {
        "model_path": "/home/ubuntu/mnt/kaiwen/STAR-1/data/models/7b_sft_mix_2k_10e_g_full_ctx_new/STAR-1/DeepSeek-R1-Distill-Qwen-7B/think_flag1/checkpoint-9-1190/tfmr_cleaned",
        "n_gpu": 2,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    "DeepSeek-R1-Distill-Qwen-7B-STAR_mix6": {
        "model_path": "/home/ubuntu/mnt/kaiwen/STAR-1/data/models/sft_mix_6k/STAR-1/DeepSeek-R1-Distill-Qwen-7B/think_flag1/checkpoint-4-1845/tfmr",
        "n_gpu": 1,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    "DeepSeek-R1-Distill-Llama-8B-STAR_mix2": {
        "model_path": "/home/ubuntu/mnt/kaiwen/STAR-1/data/models/8b_sft_mix_2k/STAR-1/DeepSeek-R1-Distill-Llama-8B/think_flag1/checkpoint-4-595/tfmr",
        "n_gpu": 1,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    "DeepSeek-R1-Distill-Llama-8B-STAR_mix2_k_loss": {
        "model_path": "/home/ubuntu/mnt/kaiwen/STAR-1/data/models/8b_sft_mix_2k_k_new/STAR-1/DeepSeek-R1-Distill-Llama-8B/think_flag1/checkpoint-4-595/tfmr",
        "n_gpu": 2,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    "DeepSeek-R1-Distill-Llama-8B-STAR_mix2_loss_scale": {
        "model_path": "/home/ubuntu/mnt/kaiwen/STAR-1/data/models/8b_sft_mix_2k_loss_scale/STAR-1/DeepSeek-R1-Distill-Llama-8B/think_flag1/checkpoint-4-595/tfmr",
        "n_gpu": 1,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    "DeepSeek-R1-Distill-Llama-8B-STAR_mix2_gate_loss": {
        "model_path": "/home/ubuntu/mnt/kaiwen/STAR-1/data/models/8b_sft_mix_2k_g_new/STAR-1/DeepSeek-R1-Distill-Llama-8B/think_flag1/checkpoint-4-595/tfmr_cleaned",
        "n_gpu": 2,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    "DeepSeek-R1-Distill-Llama-8B-STAR_mix2_gate_v2": {
        "model_path": "/home/ubuntu/mnt/kaiwen/STAR-1/data/models/8b_sft_mix_2k_g_v2_new/STAR-1/DeepSeek-R1-Distill-Llama-8B/think_flag1/checkpoint-4-595/tfmr_cleaned",
        "n_gpu": 2,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    "DeepSeek-R1-Distill-Llama-8B-STAR_mix2_gate_theta1": {
        "model_path": "/home/ubuntu/mnt/kaiwen/STAR-1/data/models/8b_sft_mix_2k_g_theta1_new/STAR-1/DeepSeek-R1-Distill-Llama-8B/think_flag1/checkpoint-4-595/tfmr_cleaned",
        "n_gpu": 2,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    "DeepSeek-R1-Distill-Llama-8B-STAR_mix2_gate_theta0": {
        "model_path": "/home/ubuntu/mnt/kaiwen/STAR-1/data/models/8b_sft_mix_2k_g_theta0_new/STAR-1/DeepSeek-R1-Distill-Llama-8B/think_flag1/checkpoint-4-595/tfmr_cleaned",
        "n_gpu": 2,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    "DeepSeek-R1-Distill-Llama-8B-STAR_mix2_gate_full_ctx": {
        "model_path": "/home/ubuntu/mnt/kaiwen/STAR-1/data/models/8b_sft_mix_2k_g_full_ctx_new/STAR-1/DeepSeek-R1-Distill-Llama-8B/think_flag1/checkpoint-4-595/tfmr_cleaned",
        "n_gpu": 2,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    "DeepSeek-R1-Distill-Llama-8B-STAR_mix2_k_gate": {
        "model_path": "/home/ubuntu/mnt/kaiwen/STAR-1/data/models/8b_sft_mix_2k_k_g_new/STAR-1/DeepSeek-R1-Distill-Llama-8B/think_flag1/checkpoint-4-595/tfmr_cleaned",
        "n_gpu": 2,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    "DeepSeek-R1-Distill-Llama-8B-STAR_mix2_k_gate_01": {
        "model_path": "/home/ubuntu/mnt/kaiwen/STAR-1/data/models/8b_sft_mix_2k_k_g_0.1/STAR-1/DeepSeek-R1-Distill-Llama-8B/think_flag1/checkpoint-4-595/tfmr_cleaned",
        "n_gpu": 2,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    "DeepSeek-R1-Distill-Llama-8B-STAR_mix2_key_loss_scale": {
        "model_path": "/home/ubuntu/mnt/kaiwen/STAR-1/data/models/8b_sft_mix_2k_key_scale_new/STAR-1/DeepSeek-R1-Distill-Llama-8B/think_flag1/checkpoint-4-595/tfmr",
        "n_gpu": 2,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    "DeepSeek-R1-Distill-Llama-8B-STAR_mix2_key_no_mask": {
        "model_path": "/home/ubuntu/mnt/kaiwen/STAR-1/data/models/8b_sft_mix_2k_key_no_mask/STAR-1/DeepSeek-R1-Distill-Llama-8B/think_flag1/checkpoint-4-595/tfmr",
        "n_gpu": 2,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    "DeepSeek-R1-Distill-Llama-8B-STAR_mix2_e7": {
        "model_path": "/home/ubuntu/mnt/kaiwen/STAR-1/data/models/8b_sft_mix_2k_e7/STAR-1/DeepSeek-R1-Distill-Llama-8B/think_flag1/checkpoint-6-833/tfmr",
        "n_gpu": 1,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    "DeepSeek-R1-Distill-Qwen-7B-GRPO_5k": {
        "model_path": "/home/ubuntu/mnt/kaiwen/STAR-1/data/models/Qwen2.5-7B-GRPO_5k",
        "n_gpu": 1,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    "DeepSeek-R1-Distill-Qwen-7B-GRPO_mix2k": {
        "model_path": "/home/ubuntu/mnt/kaiwen/STAR-1/data/models/Qwen2.5-7B-GRPO_mix_2k_2epoch",
        "n_gpu": 1,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    "DeepSeek-R1-Distill-Qwen-7B-GRPO_mix2k_new": {
        "model_path": "/home/ubuntu/mnt/kaiwen/STAR-1/data/models/Qwen-7B-GRPO_mix_2k_3e",
        "n_gpu": 1,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    "DeepSeek-R1-Distill-Qwen-7B-GRPO_mix2k_think_reward": {
        "model_path": "/home/ubuntu/mnt/kaiwen/STAR-1/data/models/Qwen-7B-GRPO_mix_2k_think_reward",
        "n_gpu": 1,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    "DeepSeek-R1-Distill-Qwen-7B-GRPO_5k_cs": {
        "model_path": "/home/ubuntu/mnt/kaiwen/STAR-1/data/models/Qwen2.5-7B-GRPO_5k_cs",
        "n_gpu": 1,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    "DeepSeek-R1-Distill-Qwen-7B-GRPO_mix_2k_cs": {
        "model_path": "/home/ubuntu/mnt/kaiwen/STAR-1/data/models/llama3.3_8B-GRPO_mix_2k_3epoch",
        "n_gpu": 1,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    "DeepSeek-R1-Distill-Qwen-7B-GRPO_mix_2k_cs_4o": {
        "model_path": "/home/ubuntu/mnt/kaiwen/STAR-1/data/models/Qwen2.5-7B-GRPO_mix_2k_cs_gpt4o_judge/checkpoint-79",
        "n_gpu": 1,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    "DeepSeek-R1-Distill-Qwen-7B-GRPO_mix_2k_4o": {
        "model_path": "/home/ubuntu/mnt/kaiwen/STAR-1/data/models/Qwen2.5-7B-GRPO_mix_2k_gpt4o_judge",
        "n_gpu": 1,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    "DeepSeek-R1-Distill-Llama-8B-GRPO_mix2k": {
        "model_path": "/home/ubuntu/mnt/kaiwen/STAR-1/data/models/llama3.3_8B-GRPO_mix_2k_3epoch/checkpoint-237",
        "n_gpu": 1,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    "DeepSeek-R1-Distill-Llama-8B-GRPO_mix2k_e2": {
        "model_path": "/home/ubuntu/mnt/kaiwen/STAR-1/data/models/llama3.3_8B-GRPO_mix_2k_3epoch/checkpoint-79",
        "n_gpu": 1,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    "DeepSeek-R1-Distill-Llama-8B-GRPO_mix2k_4o": {
        "model_path": "/home/ubuntu/mnt/kaiwen/STAR-1/data/models/Llama-8B-GRPO_mix_2k_4o",
        "n_gpu": 1,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    "DeepSeek-R1-Distill-Llama-8B-GRPO_mix2k_cs": {
        "model_path": "/home/ubuntu/mnt/kaiwen/STAR-1/data/models/Llama-8B-GRPO_mix_2k_cs/checkpoint-79",
        "n_gpu": 1,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    "DeepSeek-R1-Distill-Llama-8B-GRPO_mix2k_cs_4o": {
        "model_path": "/home/ubuntu/mnt/kaiwen/STAR-1/data/models/Llama-8B-GRPO_mix_2k_cs_gpt4o_judge/checkpoint-79",
        "n_gpu": 1,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    "DeepSeek-R1-Distill-Llama-8B-GRPO_mix2k_4o_think_reward": {
        "model_path": "/home/ubuntu/mnt/kaiwen/STAR-1/data/models/Llama-8B-GRPO_mix_2k_think_reward_4o",
        "n_gpu": 1,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    "DeepSeek-R1-Distill-Llama-8B-GRPO_mix2k_think_reward": {
        "model_path": "/home/ubuntu/mnt/kaiwen/STAR-1/data/models/Llama-8B-GRPO_mix_2k_think_reward",
        "n_gpu": 1,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    "DeepSeek-R1-Distill-Llama-8B-GRPO_mix2k_think_reward_w1": {
        "model_path": "/home/ubuntu/mnt/kaiwen/STAR-1/data/models/Llama-8B-GRPO_mix_2k_think_reward_w1",
        "n_gpu": 1,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    "DeepSeek-R1-Distill-Qwen-14B-STAR_mix2": {
        "model_path": "/home/ubuntu/mnt/kaiwen/STAR-1/data/models/14b_sft_mix_2k/STAR-1/DeepSeek-R1-Distill-Qwen-14B/think_flag1/checkpoint-4-595/tfmr",
        "n_gpu": 1,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    "DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_k": {
        "model_path": "/home/ubuntu/mnt/kaiwen/STAR-1/data/models/14b_sft_mix_2k_k_loss_2e/STAR-1/DeepSeek-R1-Distill-Qwen-14B/think_flag1/checkpoint-4-595/tfmr",
        "n_gpu": 1,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    "DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g": {
        "model_path": "/home/ubuntu/mnt/kaiwen/STAR-1/data/models/14b_sft_mix_2k_gate_2e/STAR-1/DeepSeek-R1-Distill-Qwen-14B/think_flag1/checkpoint-4-595/tfmr_cleaned",
        "n_gpu": 1,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    "DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1": {
        "model_path": "/home/ubuntu/mnt/kaiwen/STAR-1/data/models/14b_sft_mix_2k_g_theta1_new/STAR-1/DeepSeek-R1-Distill-Qwen-14B/think_flag1/checkpoint-4-595/tfmr_cleaned",
        "n_gpu": 4,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    "DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_k_g": {
        "model_path": "/home/ubuntu/mnt/kaiwen/STAR-1/data/models/14b_sft_mix_2k_k_gate_2e/STAR-1/DeepSeek-R1-Distill-Qwen-14B/think_flag1/checkpoint-4-595/tfmr_cleaned",
        "n_gpu": 1,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    
    "DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_scale": {
        "model_path": "/home/ubuntu/mnt/kaiwen/STAR-1/data/models/14b_sft_mix_2k_loss_scale/STAR-1/DeepSeek-R1-Distill-Qwen-14B/think_flag1/checkpoint-4-595/tfmr",
        "n_gpu": 2,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    "DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_e8": {
        "model_path": "/home/ubuntu/mnt/kaiwen/STAR-1/data/models/14b_sft_mix_2k_e8/STAR-1/DeepSeek-R1-Distill-Qwen-14B/think_flag1/checkpoint-7-952/tfmr",
        "n_gpu": 1,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    
    "DeepSeek-R1-Distill-Qwen-14B-GRPO_mix2k": {
        "model_path": "/home/ubuntu/mnt/kaiwen/STAR-1/data/models/Qwen-14B-GRPO_mix_2k/checkpoint-158",
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
