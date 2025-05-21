
# export WANDB_MODE=disabled

# bash run_sft.sh > output/all_10_1k_llama8b_distill_thinkflag1_03162150.out 2>&1

accelerate launch --config_file ./configs/deepspeed_zero3.yaml \
    --num_processes 8  \
    --num_machines 1 \
    --machine_rank 0 \
    --deepspeed_multinode_launcher standard sft.py \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --data_path /home/ubuntu/mnt/kaiwen/STAR-1/data/train/sft_mix_2k.json \
    --n_epochs 5 \
    --experiment_name STAR-1 \
    --base_model Llama \
    --base_flag 0 \
    --think_flag 1 \
    --output_dir /home/ubuntu/mnt/kaiwen/STAR-1/data/models/8b_sft_mix_2k_g_detach \
    --train_bsz_per_gpu 2 \
    --gradient_accumulation_steps 8 \
    --safety_gate \
    --detach_safety_gate \
    --gate_theta 0.0 \
    --key_sentence_prediction \
    --gate_weight 0.1 \
    --key_sentence_weight 0.1 \
    --key_sentence_prediction_mask_ablation \
    --key_loss_scale_ablation \
    --gate_theta 1.0 \
    --loss_scale_ablation \

# if distill model then base_flag=0 elif instruct model then base_flag=1
# if w/o think then think_flag=1

accelerate launch --config_file ./configs/deepspeed_zero3.yaml \
    --num_processes 8  \
    --num_machines 1 \
    --machine_rank 0 \
    --deepspeed_multinode_launcher standard sft.py \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --data_path /home/ubuntu/mnt/kaiwen/STAR-1/data/train/sft_mix_2k.json \
    --n_epochs 10 \
    --experiment_name STAR-1 \
    --base_model Qwen \
    --base_flag 0 \
    --think_flag 1 \
    --output_dir /home/ubuntu/mnt/kaiwen/STAR-1/data/models/7b_sft_mix_2k_10e_g_theta0 \
    --train_bsz_per_gpu 2 \
    --gradient_accumulation_steps 8 \
    --safety_gate \
    --gate_theta 0.0 \