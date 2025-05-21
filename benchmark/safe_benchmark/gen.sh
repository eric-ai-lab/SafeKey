# export data=$1 # strongreject / jbbbehaviours / wildchat / wildjailbreak / xstest / airbench
# export model=$2 # DeepSeek-R1-Distill-Qwen-1.5B

# python gen.py --data=${data} --model=${model} # > gen_out/${data}/${model}.out 2>&1 &

# wait for 6.5 hours
# sleep 20000

python gen.py --data strongreject --model DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1
python gen.py --data jbbbehaviours --model DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1
python gen.py --data wildjailbreak --model DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1
python gen.py --data xstest --model DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1
# python gen.py --data phtest --model DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1
python gen.py --data wildchat --model DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1
# python gen.py --data airbench --model DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1
python gen.py --data prefilling_8b --model DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1 --prefilling 20
python gen.py --data prefilling --model DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1
python gen.py --data door_multi --model DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1

# python eval.py --model="Llama-Guard" --part="response" --file_path="result/DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1/jbbbehaviours.json"
python eval.py --model="OpenAI4o_evaluate" --part="response" --file_path="result/DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1/jbbbehaviours.json"
# python eval.py --model="OpenAI4o_evaluate" --part="think" --file_path="result/DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1/jbbbehaviours.json"
python eval.py --model="OpenAI4o_evaluate" --part="answer" --file_path="result/DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1/jbbbehaviours.json"
# python eval.py --model="Llama-Guard" --part="response" --file_path="result/DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1/strongreject.json"
python eval.py --model="OpenAI4o_evaluate" --part="response" --file_path="result/DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1/strongreject.json"
# python eval.py --model="OpenAI4o_evaluate" --part="think" --file_path="result/DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1/strongreject.json"
python eval.py --model="OpenAI4o_evaluate" --part="answer" --file_path="result/DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1/strongreject.json"
# python eval.py --model="Llama-Guard" --part="response" --file_path="result/DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1/wildjailbreak.json"
python eval.py --model="OpenAI4o_evaluate" --part="response" --file_path="result/DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1/wildjailbreak.json"
# python eval.py --model="OpenAI4o_evaluate" --part="think" --file_path="result/DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1/wildjailbreak.json"
python eval.py --model="OpenAI4o_evaluate" --part="answer" --file_path="result/DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1/wildjailbreak.json"
# python eval.py --model="Llama-Guard" --part="response" --file_path="result/DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1/wildchat.json"
python eval.py --model="OpenAI4o_evaluate" --part="response" --file_path="result/DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1/wildchat.json"
# python eval.py --model="OpenAI4o_evaluate" --part="think" --file_path="result/DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1/wildchat.json"
python eval.py --model="OpenAI4o_evaluate" --part="answer" --file_path="result/DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1/wildchat.json"
python eval.py --model="OpenAI4o_refusal" --part="response" --file_path="result/DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1/xstest.json"
# python eval.py --model="OpenAI4o_refusal" --part="response" --file_path="result/DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1/phtest.json"
# # python eval.py --model="Llama-Guard" --part="response" --file_path="result/DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1/airbench.json"

python eval.py --model="OpenAI4o_evaluate"  --part="response" --file_path="result/DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1/prefilling_8b.json"
python eval.py --model="OpenAI4o_evaluate"  --part="answer" --file_path="result/DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1/prefilling_8b.json"
# python eval.py --model="OpenAI4o_evaluate"  --part="think" --file_path="result/DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1/prefilling_8b.json"
python eval.py --model="OpenAI4o_evaluate"  --part="response" --file_path="result/DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1/prefilling.json"
python eval.py --model="OpenAI4o_evaluate"  --part="answer" --file_path="result/DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1/prefilling.json"
# python eval.py --model="OpenAI4o_evaluate"  --part="think" --file_path="result/DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1/prefilling.json"
python eval.py --model="OpenAI4o_evaluate"  --part="response" --file_path="result/DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1/door_multi.json"
python eval.py --model="OpenAI4o_evaluate"  --part="answer" --file_path="result/DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1/door_multi.json"
# python eval.py --model="OpenAI4o_evaluate"  --part="think" --file_path="result/DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1/door_multi.json"

# python gen.py --data prefilling_8b --model DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1 --prefilling 20
# python gen.py --data prefilling_8b --model DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1-GRPO_mix2k --prefilling 20
# python gen.py --data prefilling_8b --model DeepSeek-R1-Distill-Qwen-7B-STAR_mix2 --prefilling 20
# python gen.py --data prefilling_8b --model DeepSeek-R1-Distill-Qwen-7B-GRPO_mix2k --prefilling 20
# python gen.py --data prefilling_8b --model DeepSeek-R1-Distill-Qwen-7B-GRPO_mix_2k_cs --prefilling 20
# python gen.py --data prefilling_8b --model DeepSeek-R1-Distill-Qwen-7B-GRPO_mix_2k_cs_4o --prefilling 20
# python gen.py --data prefilling_8b --model DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1-GRPO_cs_4o --prefilling 20
# python gen.py --data star1 --model DeepSeek-R1-Distill-Qwen-7B


# python eval.py --model="OpenAI4o_evaluate" --part="think" --file_path="result/DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1/prefilling_8b.json"
# python eval.py --model="OpenAI4o_evaluate" --part="think" --file_path="result/DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1-GRPO_mix2k/prefilling_8b.json"
# python eval.py --model="OpenAI4o_evaluate" --part="think" --file_path="result/DeepSeek-R1-Distill-Qwen-7B-STAR_mix2/prefilling_8b.json"
# python eval.py --model="OpenAI4o_evaluate" --part="think" --file_path="result/DeepSeek-R1-Distill-Qwen-7B-GRPO_mix2k/prefilling_8b.json"
# python eval.py --model="OpenAI4o_evaluate" --part="think" --file_path="result/DeepSeek-R1-Distill-Qwen-7B-GRPO_mix_2k_cs/prefilling_8b.json"
# python eval.py --model="OpenAI4o_evaluate" --part="think" --file_path="result/DeepSeek-R1-Distill-Qwen-7B-GRPO_mix_2k_cs_4o/prefilling_8b.json"
# python eval.py --model="OpenAI4o_evaluate" --part="think" --file_path="result/DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1-GRPO_cs_4o/prefilling_8b.json"
# python eval.py --model="OpenAI4o_evaluate" --part="think" --file_path="result/DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1-GRPO_mix2k_cs_4o/prefilling_8b.json"
# python eval.py --model="OpenAI4o_evaluate" --part="think" --file_path="result/DeepSeek-R1-Distill-Qwen-7B/star1.json"

# python eval.py --model="OpenAI4o_evaluate" --part="response" --file_path="result/DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1/prefilling_8b.json"
# python eval.py --model="OpenAI4o_evaluate" --part="response" --file_path="result/DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1-GRPO_mix2k/prefilling_8b.json"
# python eval.py --model="OpenAI4o_evaluate" --part="response" --file_path="result/DeepSeek-R1-Distill-Qwen-7B-STAR_mix2/prefilling_8b.json"
# python eval.py --model="OpenAI4o_evaluate" --part="response" --file_path="result/DeepSeek-R1-Distill-Qwen-7B-GRPO_mix2k/prefilling_8b.json"
# python eval.py --model="OpenAI4o_evaluate" --part="response" --file_path="result/DeepSeek-R1-Distill-Qwen-7B-GRPO_mix_2k_cs/prefilling_8b.json"
# python eval.py --model="OpenAI4o_evaluate" --part="response" --file_path="result/DeepSeek-R1-Distill-Qwen-7B-GRPO_mix_2k_cs_4o/prefilling_8b.json"
# python eval.py --model="OpenAI4o_evaluate" --part="response" --file_path="result/DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1-GRPO_cs_4o/prefilling_8b.json"
# python eval.py --model="OpenAI4o_evaluate" --part="response" --file_path="result/DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1-GRPO_mix2k_cs_4o/prefilling_8b.json"
# python eval.py --model="OpenAI4o_evaluate" --part="response" --file_path="result/DeepSeek-R1-Distill-Qwen-7B/star1.json"

# python eval.py --model="OpenAI4o_evaluate" --part="answer" --file_path="result/DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1/prefilling_8b.json"
# python eval.py --model="OpenAI4o_evaluate" --part="answer" --file_path="result/DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1-GRPO_mix2k/prefilling_8b.json"
# python eval.py --model="OpenAI4o_evaluate" --part="answer" --file_path="result/DeepSeek-R1-Distill-Qwen-7B-STAR_mix2/prefilling_8b.json"
# python eval.py --model="OpenAI4o_evaluate" --part="answer" --file_path="result/DeepSeek-R1-Distill-Qwen-7B-GRPO_mix2k/prefilling_8b.json"
# python eval.py --model="OpenAI4o_evaluate" --part="answer" --file_path="result/DeepSeek-R1-Distill-Qwen-7B-GRPO_mix_2k_cs/prefilling_8b.json"
# python eval.py --model="OpenAI4o_evaluate" --part="answer" --file_path="result/DeepSeek-R1-Distill-Qwen-7B-GRPO_mix_2k_cs_4o/prefilling_8b.json"
# python eval.py --model="OpenAI4o_evaluate" --part="answer" --file_path="result/DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1-GRPO_cs_4o/prefilling_8b.json"
# python eval.py --model="OpenAI4o_evaluate" --part="answer" --file_path="result/DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1-GRPO_mix2k_cs_4o/prefilling_8b.json"
# python eval.py --model="OpenAI4o_evaluate" --part="answer" --file_path="result/DeepSeek-R1-Distill-Qwen-7B/star1.json"


# python gen.py --data door_multi --model DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1
# python gen.py --data door_multi --model DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1-GRPO_mix2k
# python gen.py --data door_multi --model DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1-STAR_mix2
# python gen.py --data door_multi --model DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1-GRPO_mix2k_cs
# python gen.py --data door_multi --model DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1-GRPO_mix2k_cs_4o
# python gen.py --data door_multi --model DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1_think_reward
# python gen.py --data door_multi --model DeepSeek-R1-Distill-Qwen-7B-GRPO_mix2k
# python gen.py --data door_multi --model DeepSeek-R1-Distill-Qwen-7B-GRPO_mix_2k_cs
# python gen.py --data door_multi --model DeepSeek-R1-Distill-Qwen-7B-GRPO_mix_2k_cs_4o
# python gen.py --data door_multi --model DeepSeek-R1-Distill-Qwen-7B-STAR_mix2

# python eval.py --model="OpenAI4o_evaluate" --part="think" --file_path="result/DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1/door_multi.json"
# python eval.py --model="OpenAI4o_evaluate" --part="think" --file_path="result/DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1-GRPO_mix2k/door_multi.json"
# python eval.py --model="OpenAI4o_evaluate" --part="think" --file_path="result/DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1-STAR_mix2/door_multi.json"
# python eval.py --model="OpenAI4o_evaluate" --part="think" --file_path="result/DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1-GRPO_mix2k_cs/door_multi.json"
# python eval.py --model="OpenAI4o_evaluate" --part="think" --file_path="result/DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1-GRPO_mix2k_cs_4o/door_multi.json"
# python eval.py --model="OpenAI4o_evaluate" --part="think" --file_path="result/DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1_think_reward/door_multi.json"
# python eval.py --model="OpenAI4o_evaluate" --part="think" --file_path="result/DeepSeek-R1-Distill-Qwen-7B-GRPO_mix2k/door_multi.json"
# python eval.py --model="OpenAI4o_evaluate" --part="think" --file_path="result/DeepSeek-R1-Distill-Qwen-7B-GRPO_mix_2k_cs/door_multi.json"
# python eval.py --model="OpenAI4o_evaluate" --part="think" --file_path="result/DeepSeek-R1-Distill-Qwen-7B-GRPO_mix_2k_cs_4o/door_multi.json"
# python eval.py --model="OpenAI4o_evaluate" --part="think" --file_path="result/DeepSeek-R1-Distill-Qwen-7B-STAR_mix2/door_multi.json"

# python eval.py --model="OpenAI4o_evaluate" --part="response" --file_path="result/DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1/door_multi.json"
# python eval.py --model="OpenAI4o_evaluate" --part="response" --file_path="result/DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1-GRPO_mix2k/door_multi.json"
# python eval.py --model="OpenAI4o_evaluate" --part="response" --file_path="result/DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1-STAR_mix2/door_multi.json"
# python eval.py --model="OpenAI4o_evaluate" --part="response" --file_path="result/DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1-GRPO_mix2k_cs/door_multi.json"
# python eval.py --model="OpenAI4o_evaluate" --part="response" --file_path="result/DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1-GRPO_mix2k_cs_4o/door_multi.json"
# python eval.py --model="OpenAI4o_evaluate" --part="response" --file_path="result/DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1_think_reward/door_multi.json"
# python eval.py --model="OpenAI4o_evaluate" --part="response" --file_path="result/DeepSeek-R1-Distill-Qwen-7B-GRPO_mix2k/door_multi.json"
# python eval.py --model="OpenAI4o_evaluate" --part="response" --file_path="result/DeepSeek-R1-Distill-Qwen-7B-GRPO_mix_2k_cs/door_multi.json"
# python eval.py --model="OpenAI4o_evaluate" --part="response" --file_path="result/DeepSeek-R1-Distill-Qwen-7B-GRPO_mix_2k_cs_4o/door_multi.json"
# python eval.py --model="OpenAI4o_evaluate" --part="response" --file_path="result/DeepSeek-R1-Distill-Qwen-7B-STAR_mix2/door_multi.json"

# python eval.py --model="OpenAI4o_evaluate" --part="answer" --file_path="result/DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1/door_multi.json"
# python eval.py --model="OpenAI4o_evaluate" --part="answer" --file_path="result/DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1-GRPO_mix2k/door_multi.json"
# python eval.py --model="OpenAI4o_evaluate" --part="answer" --file_path="result/DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1-STAR_mix2/door_multi.json"
# python eval.py --model="OpenAI4o_evaluate" --part="answer" --file_path="result/DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1-GRPO_mix2k_cs/door_multi.json"
# python eval.py --model="OpenAI4o_evaluate" --part="answer" --file_path="result/DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1-GRPO_mix2k_cs_4o/door_multi.json"
# python eval.py --model="OpenAI4o_evaluate" --part="answer" --file_path="result/DeepSeek-R1-Distill-Qwen-14B-STAR_mix2_g_theta1_think_reward/door_multi.json"
# python eval.py --model="OpenAI4o_evaluate" --part="answer" --file_path="result/DeepSeek-R1-Distill-Qwen-7B-GRPO_mix2k/door_multi.json"
# python eval.py --model="OpenAI4o_evaluate" --part="answer" --file_path="result/DeepSeek-R1-Distill-Qwen-7B-GRPO_mix_2k_cs/door_multi.json"
# python eval.py --model="OpenAI4o_evaluate" --part="answer" --file_path="result/DeepSeek-R1-Distill-Qwen-7B-GRPO_mix_2k_cs_4o/door_multi.json"
# python eval.py --model="OpenAI4o_evaluate" --part="answer" --file_path="result/DeepSeek-R1-Distill-Qwen-7B-STAR_mix2/door_multi.json"