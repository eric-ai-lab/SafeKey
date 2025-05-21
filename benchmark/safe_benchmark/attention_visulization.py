import torch, matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
import json 
import re

# -------------------------- USER INPUTS --------------------------
# "/home/ubuntu/mnt/kaiwen/STAR-1/data/models/8b_sft_mix_2k/STAR-1/DeepSeek-R1-Distill-Llama-8B/think_flag1/checkpoint-4-595/tfmr"
model_name = "/home/ubuntu/mnt/kaiwen/STAR-1/data/models/8b_sft_mix_2k_k_g_new/STAR-1/DeepSeek-R1-Distill-Llama-8B/think_flag1/checkpoint-4-595/tfmr_cleaned" # "/home/ubuntu/mnt/kaiwen/STAR-1/data/models/8b_sft_mix_2k/STAR-1/DeepSeek-R1-Distill-Llama-8B/think_flag1/checkpoint-4-595/tfmr"   #                               # same checkpoint used at generation time
response_data = json.load(open("/home/ubuntu/mnt/kaiwen/STAR-1/benchmark/safe_benchmark/result/DeepSeek-R1-Distill-Llama-8B-STAR_mix2/strongreject.json" ))["data"]  # "/home/ubuntu/mnt/kaiwen/STAR-1/benchmark/safe_benchmark/result/DeepSeek-R1-Distill-Llama-8B-STAR_mix2/wildjailbreak.json"

device     = "cuda" if torch.cuda.is_available() else "cpu"
# -----------------------------------------------------------------

tok = AutoTokenizer.from_pretrained(model_name)
tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             output_attentions=True).to(device)


def average_key_to_summary_attention(full_text: str,
                                     key_sentence: str,
                                     summaries: str, 
                                     max_tokens: int = 100):
    """
    Return the average last-layer attention (averaged over heads) from every
    token in *key_sentence* to *summaries* inside *full_text*.
    """
    # Tokenise with offset mapping so we can project char spans ⟶ token spans
    enc = tok(full_text,
              return_offsets_mapping=True,
              return_tensors="pt").to(device)
    offset_mapping = enc["offset_mapping"][0].tolist()  # (seq_len, 2)

    # Helper: map character span to inclusive/exclusive token indices
    def to_tok_span(char_span):
        b, e = char_span
        idxs = [i for i, (s, e_) in enumerate(offset_mapping)
                if not (e_ <= b or s >= e)]            # tokens that overlap span
        return idxs[0], idxs[-1] + 1                  # [start, end)

    # Locate spans (first occurrence is fine given construction)
    summ_start = full_text.find(summaries)
    key_start  = full_text.find(key_sentence)
    if summ_start == -1 or key_start == -1:
        return None                                    # something went wrong

    summ_tok_s, summ_tok_e = to_tok_span((summ_start,
                                          summ_start + len(summaries)))
    key_tok_s , key_tok_e  = to_tok_span((key_start ,
                                          key_start  + len(key_sentence)))

    # Forward pass – keep only last-layer attention
    with torch.no_grad():
        out = model(**{k: v for k, v in enc.items() if k != "offset_mapping"})
    # out.attentions[-1] → (1, n_heads, L, L)
    attn_last = out.attentions[-1].mean(dim=1)[0]      # (L, L), avg over heads

    summary_idx = torch.arange(summ_tok_s, summ_tok_e, device=device)

    scores = []
    for ki in range(key_tok_s, min(key_tok_s + max_tokens, key_tok_e)):
        # Sum of attention weights from this key-sentence token → all summary tokens
        scores.append(attn_last[ki, summary_idx].sum().item())

    return sum(scores) / len(scores)

avg_att_summ = 0
avg_att_query = 0
total = 0
for data in response_data:
    if "key_sentence_idx" not in data:
        continue
    total += 1
    # get the sentences
    input_query = data["prompt"] 
    key_sentence_idx = data["key_sentence_idx"]
    sentences = re.split(r'(?<=[.?])(?<!\.\.\.)\s+', data["response"][0])
    
    summaries = " ".join(sentences[:key_sentence_idx-1])
    key_sentence = sentences[key_sentence_idx-1]
    full_input = input_query + data["response"][0][:len(data["response"][0]) // 2]
    
    avg_attn_summ = average_key_to_summary_attention(full_input,
                                                key_sentence,
                                                summaries, max_tokens=500)
    avg_attn_query = average_key_to_summary_attention(full_input,
                                                key_sentence,
                                                input_query, max_tokens=500)
    avg_att_summ += avg_attn_summ
    avg_att_query += avg_attn_query
    
avg_att_summ /= total
avg_att_query /= total
print(f"Average attention from key sentence to input query: {avg_att_query:.4f}")
print(f"Average attention from key sentence to summaries: {avg_att_summ:.4f}")
