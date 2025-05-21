# #!/bin/bash

# parts=("response" "think" "answer")
# datasets=("jbbbehaviours" "strongreject" "wildchat") # "wildjailbreak" 
# paths=(
#   "result/DeepSeek-R1-Distill-Qwen-14B-GRPO_mix2k"
#   "result/DeepSeek-R1-Distill-Qwen-14B-STAR_mix2"
#   "result/DeepSeek-R1-Distill-Qwen-14B-GRPO_mix_2k_cs"
# )
# model="OpenAI4o_evaluate"

# for base_path in "${paths[@]}"; do
#   for dataset in "${datasets[@]}"; do
#     for part in "${parts[@]}"; do
#       file="$base_path/$dataset.json"
#       if [ -f "$file" ]; then
#         echo "Running: $file [$part]"
#         python eval.py --model="$model" --part="$part" --file_path="$file"
#       else
#         echo "Warning: File not found - $file"
#       fi
#     done
#   done
# done


python gen.py --data strongreject --model DeepSeek-R1-Distill-Qwen-7B-STAR_mix2_gate_theta0
python gen.py --data jbbbehaviours --model DeepSeek-R1-Distill-Qwen-7B-STAR_mix2_gate_theta0
python gen.py --data wildjailbreak --model DeepSeek-R1-Distill-Qwen-7B-STAR_mix2_gate_theta0
python gen.py --data xstest --model DeepSeek-R1-Distill-Qwen-7B-STAR_mix2_gate_theta0
# python gen.py --data phtest --model DeepSeek-R1-Distill-Qwen-7B-STAR_mix2_gate_theta0
python gen.py --data wildchat --model DeepSeek-R1-Distill-Qwen-7B-STAR_mix2_gate_theta0
# python gen.py --data airbench --model DeepSeek-R1-Distill-Qwen-7B-STAR_mix2_gate_theta0
python gen.py --data prefilling_8b --model DeepSeek-R1-Distill-Qwen-7B-STAR_mix2_gate_theta0 --prefilling 20
python gen.py --data prefilling --model DeepSeek-R1-Distill-Qwen-7B-STAR_mix2_gate_theta0
python gen.py --data door_multi --model DeepSeek-R1-Distill-Qwen-7B-STAR_mix2_gate_theta0

# python eval.py --model="Llama-Guard" --part="response" --file_path="result/DeepSeek-R1-Distill-Qwen-7B-STAR_mix2_gate_theta0/jbbbehaviours.json"
python eval.py --model="OpenAI4o_evaluate" --part="response" --file_path="result/DeepSeek-R1-Distill-Qwen-7B-STAR_mix2_gate_theta0/jbbbehaviours.json"
# python eval.py --model="OpenAI4o_evaluate" --part="think" --file_path="result/DeepSeek-R1-Distill-Qwen-7B-STAR_mix2_gate_theta0/jbbbehaviours.json"
# python eval.py --model="OpenAI4o_evaluate" --part="answer" --file_path="result/DeepSeek-R1-Distill-Qwen-7B-STAR_mix2_gate_theta0/jbbbehaviours.json"
# python eval.py --model="Llama-Guard" --part="response" --file_path="result/DeepSeek-R1-Distill-Qwen-7B-STAR_mix2_gate_theta0/strongreject.json"
python eval.py --model="OpenAI4o_evaluate" --part="response" --file_path="result/DeepSeek-R1-Distill-Qwen-7B-STAR_mix2_gate_theta0/strongreject.json"
# python eval.py --model="OpenAI4o_evaluate" --part="think" --file_path="result/DeepSeek-R1-Distill-Qwen-7B-STAR_mix2_gate_theta0/strongreject.json"
# python eval.py --model="OpenAI4o_evaluate" --part="answer" --file_path="result/DeepSeek-R1-Distill-Qwen-7B-STAR_mix2_gate_theta0/strongreject.json"
# python eval.py --model="Llama-Guard" --part="response" --file_path="result/DeepSeek-R1-Distill-Qwen-7B-STAR_mix2_gate_theta0/wildjailbreak.json"
python eval.py --model="OpenAI4o_evaluate" --part="response" --file_path="result/DeepSeek-R1-Distill-Qwen-7B-STAR_mix2_gate_theta0/wildjailbreak.json"
# python eval.py --model="OpenAI4o_evaluate" --part="think" --file_path="result/DeepSeek-R1-Distill-Qwen-7B-STAR_mix2_gate_theta0/wildjailbreak.json"
# python eval.py --model="OpenAI4o_evaluate" --part="answer" --file_path="result/DeepSeek-R1-Distill-Qwen-7B-STAR_mix2_gate_theta0/wildjailbreak.json"
# python eval.py --model="Llama-Guard" --part="response" --file_path="result/DeepSeek-R1-Distill-Qwen-7B-STAR_mix2_gate_theta0/wildchat.json"
python eval.py --model="OpenAI4o_evaluate" --part="response" --file_path="result/DeepSeek-R1-Distill-Qwen-7B-STAR_mix2_gate_theta0/wildchat.json"
# python eval.py --model="OpenAI4o_evaluate" --part="think" --file_path="result/DeepSeek-R1-Distill-Qwen-7B-STAR_mix2_gate_theta0/wildchat.json"
# python eval.py --model="OpenAI4o_evaluate" --part="answer" --file_path="result/DeepSeek-R1-Distill-Qwen-7B-STAR_mix2_gate_theta0/wildchat.json"
python eval.py --model="OpenAI4o_refusal" --part="response" --file_path="result/DeepSeek-R1-Distill-Qwen-7B-STAR_mix2_gate_theta0/xstest.json"
# python eval.py --model="OpenAI4o_refusal" --part="response" --file_path="result/DeepSeek-R1-Distill-Qwen-7B-STAR_mix2_gate_theta0/phtest.json"
# # python eval.py --model="Llama-Guard" --part="response" --file_path="result/DeepSeek-R1-Distill-Qwen-7B-STAR_mix2_gate_theta0/airbench.json"

python eval.py --model="OpenAI4o_evaluate"  --part="response" --file_path="result/DeepSeek-R1-Distill-Qwen-7B-STAR_mix2_gate_theta0/prefilling_8b.json"
# python eval.py --model="OpenAI4o_evaluate"  --part="answer" --file_path="result/DeepSeek-R1-Distill-Qwen-7B-STAR_mix2_gate_theta0/prefilling_8b.json"
# python eval.py --model="OpenAI4o_evaluate"  --part="think" --file_path="result/DeepSeek-R1-Distill-Qwen-7B-STAR_mix2_gate_theta0/prefilling_8b.json"
python eval.py --model="OpenAI4o_evaluate"  --part="response" --file_path="result/DeepSeek-R1-Distill-Qwen-7B-STAR_mix2_gate_theta0/prefilling.json"
# python eval.py --model="OpenAI4o_evaluate"  --part="answer" --file_path="result/DeepSeek-R1-Distill-Qwen-7B-STAR_mix2_gate_theta0/prefilling.json"
# python eval.py --model="OpenAI4o_evaluate"  --part="think" --file_path="result/DeepSeek-R1-Distill-Qwen-7B-STAR_mix2_gate_theta0/prefilling.json"
python eval.py --model="OpenAI4o_evaluate"  --part="response" --file_path="result/DeepSeek-R1-Distill-Qwen-7B-STAR_mix2_gate_theta0/door_multi.json"
# python eval.py --model="OpenAI4o_evaluate"  --part="answer" --file_path="result/DeepSeek-R1-Distill-Qwen-7B-STAR_mix2_gate_theta0/door_multi.json"
