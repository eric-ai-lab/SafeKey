#!/usr/bin/env python
import argparse
import json
import torch
import os
import sys
import time
from typing import List, Dict
from tqdm import tqdm
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI, AsyncOpenAI
import asyncio




# ---------------------- KL Eval ----------------------
def compute_kl_divergence(p_probs, q_probs):
    epsilon = 1e-10
    p = p_probs + epsilon
    q = q_probs + epsilon
    return torch.sum(p * torch.log(p / q)).item()

def get_next_token_probs(model, tokenizer, prompt):
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model(**inputs)
        logits = outputs.logits
        next_token_logits = logits[0, -1, :]
        probs = torch.softmax(next_token_logits, dim=-1)
    return probs

def process_kl(args):
    with open(args.file, 'r') as file:
        input_data = json.load(file)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    good_model = AutoModelForCausalLM.from_pretrained(args.model_path).to("cuda:0")
    good_model.eval()
    bad_model = AutoModelForCausalLM.from_pretrained(args.bad_model_path).to("cuda:1")
    bad_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    kl_div_sums = {k: 0.0 for k in args.ks}
    kl_div_counts = {k: 0 for k in args.ks}
    for entry in tqdm(input_data, desc="KL Divergence Calculation"):
        # messages = entry.get("messages", [])
        # if len(messages) < 2:
        #     continue
        # question = messages[0].get("content", "")
        # response = messages[1].get("content", "")
        question = entry["instruction"]
        response = entry["response"][0]
        response_tokens = tokenizer.encode(response, add_special_tokens=False)
        conv = [{"role": "user", "content": question}]
        prompt_base = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        for k in args.ks:
            if k <= len(response_tokens):
                prefill = tokenizer.decode(response_tokens[:k], skip_special_tokens=True)
                full_prompt = prompt_base + " " + prefill
                good_probs = get_next_token_probs(good_model, tokenizer, full_prompt)
                bad_probs = get_next_token_probs(bad_model, tokenizer, full_prompt)
                bad_probs = bad_probs.to(good_probs.device)
                kl_div = compute_kl_divergence(good_probs, bad_probs)
                kl_div_sums[k] += kl_div
                kl_div_counts[k] += 1
    average_kl_div = {k: (kl_div_sums[k] / kl_div_counts[k] if kl_div_counts[k] > 0 else None) for k in args.ks}
    output_file = args.output_prefix + "_kl_eval.json"
    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(average_kl_div, f_out, indent=4)
    print("KL divergence evaluation completed.")
    
    for k in args.ks:
        avg = average_kl_div[k]
        if avg is not None:
            print(f"Token count {k}: Average KL Divergence = {avg:.6f}")
        else:
            print(f"Token count {k}: No data available")
    print(f"Results saved to {output_file}")

# ---------------------- Main ----------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate responses using various methods.")
    parser.add_argument("--file", type=str, default="example.json", help="Path to input JSON (or JSONL for KL).")
    parser.add_argument("--output_prefix", type=str, default="output", help="Prefix for output file.")
    parser.add_argument("--gpu", default="2", help="CUDA device to use.")
    parser.add_argument("--model_path", type=str, help="Path of target model (for KL).")
    parser.add_argument("--bad_model_path", type=str, help="Path of base model (for KL).")
    parser.add_argument("--ks", type=int, nargs='*', default=[4,6,8,10,12,15,18,22,26,30,35,40,50], help="Token counts for KL.")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for OpenAI calls.")
    args = parser.parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    out_dir = os.path.dirname(args.output_prefix)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    
    process_kl(args)
    

if __name__ == "__main__":
    main()