# python gen.py --data key_better_8b --model DeepSeek-R1-Distill-Llama-8B-STAR_mix2_k_gate --prefilling 500


# python eval.py --model="OpenAI4o_evaluate" --part="response" --file_path="result/DeepSeek-R1-Distill-Llama-8B-STAR_mix2_k_gate/key_better_8b_500.json"

# python gen.py --data xstest --model DeepSeek-R1-Distill-Llama-8B-STAR_mix2_k_gate


# python eval.py --model="OpenAI4o_refusal" --part="response" --file_path="result/DeepSeek-R1-Distill-Llama-8B-STAR_mix2_k_gate/xstest.json"


# python gen.py --data door_multi --model DeepSeek-R1-Distill-Llama-8B-STAR_mix2_gate_loss


# python eval.py --model="OpenAI4o_evaluate" --part="response" --file_path="result/DeepSeek-R1-Distill-Llama-8B-STAR_mix2_gate_loss/door_multi.json"

python eval.py --model="Llama-Guard" --part="response" --file_path="result/DeepSeek-R1-Distill-Llama-8B-STAR_mix2/wildjailbreak.json"