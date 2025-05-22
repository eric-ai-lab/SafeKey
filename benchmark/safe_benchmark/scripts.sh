python gen.py --data strongreject --model DeepSeek-R1-Distill-Llama-8B-STAR_mix2_safekey
python gen.py --data jbbbehaviours --model DeepSeek-R1-Distill-Llama-8B-STAR_mix2_safekey
python gen.py --data wildjailbreak --model DeepSeek-R1-Distill-Llama-8B-STAR_mix2_safekey
python gen.py --data xstest --model DeepSeek-R1-Distill-Llama-8B-STAR_mix2_safekey
python gen.py --data wildchat --model DeepSeek-R1-Distill-Llama-8B-STAR_mix2_safekey
python gen.py --data prefilling_8b --model DeepSeek-R1-Distill-Llama-8B-STAR_mix2_safekey --prefilling 20
python gen.py --data door_multi --model DeepSeek-R1-Distill-Llama-8B-STAR_mix2_safekey


python eval.py --model="OpenAI4o_evaluate" --part="response" --file_path="result/DeepSeek-R1-Distill-Llama-8B-STAR_mix2_safekey/jbbbehaviours.json"
python eval.py --model="OpenAI4o_evaluate" --part="response" --file_path="result/DeepSeek-R1-Distill-Llama-8B-STAR_mix2_safekey/strongreject.json"
python eval.py --model="OpenAI4o_evaluate" --part="response" --file_path="result/DeepSeek-R1-Distill-Llama-8B-STAR_mix2_safekey/wildjailbreak.json"
python eval.py --model="OpenAI4o_evaluate" --part="response" --file_path="result/DeepSeek-R1-Distill-Llama-8B-STAR_mix2_safekey/wildchat.json"
python eval.py --model="OpenAI4o_refusal" --part="response" --file_path="result/DeepSeek-R1-Distill-Llama-8B-STAR_mix2_safekey/xstest.json"

python eval.py --model="OpenAI4o_evaluate"  --part="response" --file_path="result/DeepSeek-R1-Distill-Llama-8B-STAR_mix2_safekey/prefilling_8b.json"
python eval.py --model="OpenAI4o_evaluate"  --part="response" --file_path="result/DeepSeek-R1-Distill-Llama-8B-STAR_mix2_safekey/door_multi.json"
