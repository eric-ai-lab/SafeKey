# SafeKey: Amplifying Aha-Moment Insights for Safety Reasoning

<!-- <p align="center">
📃 <a href="https://arxiv.org/abs/2504.01903" target="_blank">Paper</a> ｜🤗 <a href="https://huggingface.co/datasets/UCSC-VLAA/STAR-1" target="_blank">STAR-1 Data</a> | 🤗 <a href="https://huggingface.co/collections/UCSC-VLAA/star-1-67edda2a042e8ba3e955e522" target="_blank">STAR-1 Model</a> |  📚 <a href="https://ucsc-vlaa.github.io/STAR-1/" target="_blank">Project Page</a>
</p> -->

[Kaiwen Zhou](https://kevinz-01.github.io/), [Xuandong Zhao](https://xuandongzhao.github.io/), [Gaowen Liu](https://scholar.google.com/citations?user=NIv_aeQAAAAJ&hl=en), [Jayanth Srinivasa](https://scholar.google.com/citations?user=HtNfeKYAAAAJ&hl=en), [Aosong Feng](https://scholar.google.com/citations?user=hFhhrmgAAAAJ&hl=en), [Dawn Song](https://dawnsong.io/), [Xin Eric Wang](https://eric-xw.github.io/)

## Introduction

<img src="./figures/fig1.pdf" alt="main" style="zoom: 33%;" />
<img src="./figures/fig2.pdf" alt="main" style="zoom: 33%;" />

- 
- 
- 


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
git clone https://github.com/UCSC-VLAA/STAR-1.git
cd STAR-1
pip install -e .
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
    --train_bsz_per_gpu 1 \
    --num_machines 1 \
    --machine_rank 0 \
    --deepspeed_multinode_launcher standard sft.py \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --data_path ../data/STAR-1.json \
    --n_epochs 5 \
    --experiment_name STAR-1 \
    --base_model Qwen \
    --base_flag 0 \
    --think_flag 1
```
- `base_flag`: If distill model then 0 elif instruct model then 1
- `think_flag`: default=1, if `w/o think` then 0 **(Sec 4.2)**
- `train_bsz_per_gpu * num_processes` should be 8 to keep the batchsize as 128
- You change the `model_path` to different model 
- Or change the `data_path` to use different finetune data **(Sec 4.1)**

## Evaluation
You could change the `mode_path` of the evaluated model in `benchmark/config.py`.
### Safety Benchmark
```
cd benchmark/safe_benchmark
bash scripts.sh $model $data
# bash scripts.sh DeepSeek-R1-Distill-Qwen-1.5B strongreject
```

### Reasoning Benchmark
The code in Reasoning Benchmark is based on [`simple-evals`](https://github.com/openai/simple-evals) and modified.
```
cd benchmark/reasoning_benchmark
bash run_all_evals.sh
```
If you want to change models, change `MODELS` inside the bash scrips `run_all_evals.sh` at Line 7.


## Acknowledgement
This codebase is build upon [STAR-1](https://github.com/UCSC-VLAA/STAR-1/tree/main), thanks to their great work!


<!-- ## Citation -->
<!-- ```
@article{wang2025star1saferalignmentreasoning,
    title={STAR-1: Safer Alignment of Reasoning LLMs with 1K Data}, 
    author={Zijun Wang and Haoqin Tu and Yuhan Wang and Juncheng Wu and Jieru Mei and Brian R. Bartoldson and Bhavya Kailkhura and Cihang Xie},
    year={2025},
    journal = {arXiv preprint arXiv:2504.01903}
}
``` -->



