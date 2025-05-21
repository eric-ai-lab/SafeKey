import argparse
from ast import arg
import pytz
from datetime import datetime
import os
from datasets import load_dataset, concatenate_datasets, Dataset
from config import *
import re
import pandas as pd
import random
import json
import os

get_time = lambda: datetime.now(pytz.utc).astimezone(pytz.timezone('US/Pacific')).strftime('%y%m%d-%H%M')

DATA_DIR = "datasets"

def file_exists(file_path):
    return os.path.exists(file_path) and os.path.isfile(file_path)

# dataset load and parsing
def parse_strongreject():

    data= load_dataset(
        "csv",
        data_files="https://raw.githubusercontent.com/alexandrasouly/strongreject/main/strongreject_dataset/strongreject_dataset.csv",
        cache_dir=str(DATA_DIR)+"/StrongREJECT",
    )["train"]

    harmful_inst = data['forbidden_prompt']
    harmful_target = [None for _ in harmful_inst]
    return harmful_inst, harmful_target

def parse_jbbbehaviours():
    data = load_dataset(
        "JailbreakBench/JBB-Behaviors",
        name="behaviors",
        cache_dir=str(DATA_DIR)+"/JBBBehaviours",
    )["harmful"]

    harmful_inst = data['Goal']
    harmful_target = [None for _ in harmful_inst]
    return harmful_inst, harmful_target

def parse_sum():
    harmful_inst = json.load(open('datasets/summarization_prompts.json'))
    harmful_target = [None for _ in harmful_inst]
    
    return harmful_inst, harmful_target

def parse_key_better_8b():
    data = json.load(open('datasets/key_better_samples.json'))
    harmful_inst = [d['instruction'] for d in data]
    harmful_target = [d['response'] for d in data]
    
    return harmful_inst, harmful_target

def parse_jbbbehaviours_benign():
    data = load_dataset(
        "JailbreakBench/JBB-Behaviors",
        name="behaviors",
        cache_dir=str(DATA_DIR)+"/JBBBehaviours",
    )["benign"]

    harmful_inst = data['Goal']
    harmful_target = [None for _ in harmful_inst]
    return harmful_inst, harmful_target

def parse_phtest():
    data = load_dataset(
        "furonghuang-lab/PHTest",
        cache_dir=str(DATA_DIR)+"/PHTest",
    )["train"]

    harmless_data = [d for d in data if d['Harmfulness'] == 'harmless']
    harmless_inst = [d['Request'] for d in harmless_data]
    harmless_target = [None for _ in harmless_inst]
    return harmless_inst, harmless_target

def parse_airbench():
    data = load_dataset("stanford-crfm/air-bench-2024", "default", split="test", cache_dir=str(DATA_DIR)+"/airbench",)
    data = list(data)[::5]
    harmful_inst = [d['prompt'] for d in data]
    harmful_target = [None for _ in harmful_inst]
    return harmful_inst, harmful_target

def parse_wildchat():
    if file_exists(str(DATA_DIR)+"/WildChat/WildChat.csv"):
        data = Dataset.from_pandas(pd.read_csv(str(DATA_DIR)+"/WildChat/WildChat.csv"))
        harmful_inst = data['question']
        harmful_target = [None for _ in harmful_inst]
        return harmful_inst, harmful_target
    data = load_dataset(
        "yuntian-deng/WildChat-1M-Full-with-parameters-internal",
        cache_dir=str(DATA_DIR)+"/WildChat",
    )["train"]
    data = data.filter(lambda x: x["toxic"] == True)
    data = data.filter(lambda x: x["turn"] == 1)
    data = data.filter(lambda x: x["language"] == "English")
    data = data.filter(lambda x: x["openai_moderation"][-1]["flagged"] == True)
    data = data.map(lambda x: {"category": (k:=max(d:=x.pop("openai_moderation")[-1]["category_scores"], key=d.get)), "category_score": d[k], "question": x.pop("conversation")[0]["content"]})
    ds_all = []
    for harmful_type in ["harassment", "harassment/threatening", "hate", "hate/threatening", "self-harm", "self-harm/instructions", "self-harm/intent", "sexual", "sexual/minors", "violence", "violence/graphic"]:
        ds_filter = data.filter(lambda x: x["category"] == harmful_type)
        sorted_ds_filter = ds_filter.sort("category_score", reverse=True)
        top_100_ds_filter = sorted_ds_filter.select(range(min(100, len(sorted_ds_filter))))
        ds_all.append(top_100_ds_filter)
        # print(harmful_type, f": {len(top_100_ds_filter)}") # debug
    data = concatenate_datasets(ds_all)
    df = data.to_pandas()
    df.to_csv(str(DATA_DIR)+"/WildChat/WildChat.csv", index=False)
    harmful_inst = data['question']
    harmful_target = [None for _ in harmful_inst]
    return harmful_inst, harmful_target

def parse_xstest() -> Dataset:
    if file_exists(str(DATA_DIR)+"/XSTest/XSTest.csv"):
        data =  Dataset.from_pandas(pd.read_csv(str(DATA_DIR)+"/XSTest/XSTest.csv"))
        harmful_inst = data['question']
        harmful_target = [None for _ in harmful_inst]
        return harmful_inst, harmful_target
    data = load_dataset(
        "csv",
        data_files="https://raw.githubusercontent.com/paul-rottger/xstest/refs/heads/main/xstest_prompts.csv",
        cache_dir=str(DATA_DIR)+"/XSTest",
    )["train"]
    data = data.filter(lambda x: x["label"] == "safe")
    data = data.map(lambda x: {"question": x.pop("prompt")})
    df = data.to_pandas()
    df.to_csv(str(DATA_DIR)+"/XSTest/XSTest.csv", index=False)
    harmful_inst = data['question']
    harmful_target = [None for _ in harmful_inst]
    return harmful_inst, harmful_target

def parse_wildjailbreak() -> Dataset:
    if file_exists(str(DATA_DIR)+"/wildjailbreak/wildjailbreak.csv"):
        data =  Dataset.from_pandas(pd.read_csv(str(DATA_DIR)+"/wildjailbreak/wildjailbreak.csv"))
        harmful_inst = data['question']
        harmful_target = [None for _ in harmful_inst]
        return harmful_inst, harmful_target
    data = load_dataset(
        "allenai/wildjailbreak",
        name="eval",
        cache_dir=str(DATA_DIR)+"/wildjailbreak",
    )["train"]
    data = data.map(lambda x: {"source": "wildjailbreak", "question": x.pop("adversarial")})
    data = data.filter(lambda x: x["data_type"] == "adversarial_harmful")
    data = data.shuffle(seed=42).select(range(250))
    df = data.to_pandas()
    df.to_csv(str(DATA_DIR)+"/wildjailbreak/wildjailbreak.csv", index=False)
    harmful_inst = data['question']
    harmful_target = [None for _ in harmful_inst]
    return harmful_inst, harmful_target

def parse_prefilling():
    # Load the CSV file
    df = pd.read_csv('datasets/advbench_prefilling/harmful_behaviors_pair.csv')

    # Extract the list of (goal, target) tuples
    goal_list = list(df['goal'])
    target_list = list(df['target'])
    
    return goal_list, target_list

def parse_prefilling_8b():
    # Load the CSV file
    df = pd.read_csv('datasets/advbench_prefilling/harmful_r1_8b_gptjudge.csv')

    # Extract the list of (goal, target) tuples
    goal_list = list(df['goal'])
    target_list = list(df['target'])
    
    return goal_list, target_list

def parse_door_multiturn():
    # Load the CSV file
    data = json.load(open('datasets/door_multiturn/combine.json'))
    
    return data, [None for _ in data]

def parse_star1():
    # Load the CSV file
    data = json.load(open('/home/ubuntu/mnt/kaiwen/STAR-1/data/train/sft_1k.json'))

    # Extract the list of (goal, target) tuples
    goal_list = [d["question"] for d in data]
    goal_list = goal_list[::5]
    return goal_list, [None for _ in goal_list]

def extract_content(tag, text):
    start_idx = text.find(tag)
    # if start_idx == -1:
    #     return None
    start_of_content = start_idx + len(tag)
    if tag == "##the_score: ":
        end_idx = text.find("\n", start_of_content)
    elif tag == "##comparison_result: ":
        while not text[start_of_content].isdigit():
            start_of_content += 1
        # find the first integer
        end_idx = start_of_content 
        while not text[end_idx].isdigit():
            end_idx += 1
        end_idx += 1
    else:
        end_idx = text.find("\n##", start_of_content)
    
    if end_idx == -1:
        content = text[start_of_content:].strip()
    else:
        content = text[start_of_content:end_idx].strip()
    return content

def sample_row(raw_data, sample_num):
    sample_row_list = []
    for l2 in range(1, 17):

        l2_specific_rows = []
        for i,data in enumerate(raw_data):
            cate_idx, l2_name, l3_name, l4_name, prompt = data.values()
            l2_index = int(cate_idx.split('.')[0])
            if l2_index == l2:
                l2_specific_rows.append((cate_idx, l2_name, l3_name, l4_name, prompt))
        
        l2_specific_sampled_row = random.sample(l2_specific_rows, sample_num)
        sample_row_list.extend(l2_specific_sampled_row)
    
    return sample_row_list

def get_args():
    '''for resp_gen and pipeline'''
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model', type=str, default=RSM_LIST[0], choices=RSM_LIST)
    argparser.add_argument('--data', type=str, default='strongreject', choices=EVAL_DATA)
    argparser.add_argument('--system', type=bool, default=DEFAULT_GEN_CONFIG['system'])
    argparser.add_argument('--temperature', type=float, default=DEFAULT_GEN_CONFIG['temperature'])
    argparser.add_argument('--topp', type=float, default=DEFAULT_GEN_CONFIG['topp'])
    argparser.add_argument('--topk', type=int, default=DEFAULT_GEN_CONFIG['topk'])
    argparser.add_argument('--prefilling', type=int, default=99, help='number of tokens to prefill')
    argparser.add_argument('--max_tokens', type=int, default=DEFAULT_GEN_CONFIG['max_tokens'])
    argparser.add_argument('--repeat_n', type=int, default=DEFAULT_GEN_CONFIG['repeat_n'], help='number of sample per prompt input')
    argparser.add_argument('--n', type=int, default=-1, help='number of samples to use, -1 to use all')
    argparser.add_argument('--start_idx', type=int, default=0, help='start index of the samples to use')
    args = argparser.parse_args()

    return args