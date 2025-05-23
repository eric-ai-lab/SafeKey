# SafeKey: Amplifying Aha-Moment Insights for Safety Reasoning

<!-- <p align="center">
📃 <a href="https://arxiv.org/abs/2504.01903" target="_blank">Paper</a> ｜🤗 <a href="https://huggingface.co/datasets/UCSC-VLAA/STAR-1" target="_blank">STAR-1 Data</a> | 🤗 <a href="https://huggingface.co/collections/UCSC-VLAA/star-1-67edda2a042e8ba3e955e522" target="_blank">STAR-1 Model</a> |  📚 <a href="https://ucsc-vlaa.github.io/STAR-1/" target="_blank">Project Page</a>
</p> -->

[Kaiwen Zhou](https://kevinz-01.github.io/), [Xuandong Zhao](https://xuandongzhao.github.io/), [Gaowen Liu](https://scholar.google.com/citations?user=NIv_aeQAAAAJ&hl=en), [Jayanth Srinivasa](https://scholar.google.com/citations?user=HtNfeKYAAAAJ&hl=en), [Aosong Feng](https://scholar.google.com/citations?user=hFhhrmgAAAAJ&hl=en), [Dawn Song](https://dawnsong.io/), [Xin Eric Wang](https://eric-xw.github.io/)

## Introduction

<img src="./figures/fig1.png" alt="main" style="zoom: 33%;" />

- SFT LRMs are vulnerable to jailbreaks.
- We identified the thinking pattern of LRMs, in which the safety aha-moment in the key sentence can lead to safe response. 

<img src="./figures/fig2.png" alt="main" style="zoom: 33%;" />

- We proposed the SafeKey framework to improve LRM safety alignment. 


## Artifacts
<!-- ### Data

| Dataset    | Num. of Sample | URL                                                                 |
|------------|----------------|----------------------------------------------------------------------|
| STAR-1     | 1K             | 🤗 [UCSC-VLAA/STAR-1](https://huggingface.co/datasets/UCSC-VLAA/STAR-1) |
| STAR 41K   | 41K            | 🤗 [UCSC-VLAA/STAR-41K](https://huggingface.co/datasets/UCSC-VLAA/STAR-41K) |
| STAR-benign-915   | 915            | 🤗 [UCSC-VLAA/STAR-benign-915](https://huggingface.co/datasets/UCSC-VLAA/STAR-benign-915) | -->



### Model
| Model                          | URL                               |
|--------------------------------|-------------------------------------------|
| SafeKey-7B          | 🤗 [kzhou35/SafeKey-7B](https://huggingface.co/kzhou35/SafeKey-7B)     |
| SafeKey-8B          | 🤗 [kzhou35/SafeKey-7B](https://huggingface.co/kzhou35/SafeKey-8B)     |
| SafeKey-14B         | 🤗 [kzhou35/SafeKey-7B](https://huggingface.co/kzhou35/SafeKey-14B)   |


## Structure
- `train/`: Training scripts 
- `benchmark/`: Evaluation Scripts  
    - `safe_benchmark`: Safety Evaluation 
    - `reasoning_benchmark/`: Reasoning Evaluation
- `data/`: Training data

## Quick Start
```
git clone https://github.com/eric-ai-lab/SafeKey.git
cd SafeKey
conda env create -f environment.yml
```

## Training
```
cd train
bash run_sft.sh
```
The `run_sft.sh ` looks like:
```
accelerate launch --config_file ./configs/deepspeed_zero3.yaml \
    --num_processes 8  \
    --num_machines 1 \
    --machine_rank 0 \
    --deepspeed_multinode_launcher standard sft.py \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --data_path ../data/train/sft_mix_2k.json \
    --n_epochs 5 \
    --experiment_name safe_lrm \
    --base_model Llama \
    --base_flag 0 \
    --think_flag 1 \
    --output_dir /home/ubuntu/mnt/kaiwen/STAR-1/data/models/8b_safekey \
    --train_bsz_per_gpu 2 \
    --gradient_accumulation_steps 8 \
    --safety_gate \
    --key_sentence_prediction
```
- Change the `model_path` to different model 

## Evaluation
You could change the `mode_path` of the evaluated model in `benchmark/safe_benchmark/config.py`, and `benchmark/reasoning_benchmark/config.py`.
### Safety Benchmark
```
cd benchmark/safe_benchmark
bash scripts.sh
```
Change the model that you want you evaluate in `scripts.sh`.

### Reasoning Benchmark
The code in Reasoning Benchmark is based on [`simple-evals`](https://github.com/openai/simple-evals) and modified.
```
cd benchmark/reasoning_benchmark
bash run_all_evals.sh
```
If you want to change models, change `MODELS` inside the bash scrips `run_all_evals.sh` at Line 7.


## Acknowledgement
This codebase is build upon [STAR-1](https://github.com/UCSC-VLAA/STAR-1/tree/main), thanks to their great work!


## Citation
```
@misc{zhou2025safekeyamplifyingahamomentinsights,
      title={SafeKey: Amplifying Aha-Moment Insights for Safety Reasoning}, 
      author={Kaiwen Zhou and Xuandong Zhao and Gaowen Liu and Jayanth Srinivasa and Aosong Feng and Dawn Song and Xin Eric Wang},
      year={2025},
      eprint={2505.16186},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2505.16186}, 
}
```



