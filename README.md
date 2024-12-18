# Shopping MMLU
This is the repository for 'Shopping MMLU: A Massive Multi-Task Online Shopping Benchmark for Large Language Models', which is accepted by **NeurIPS 2024 Datasets and Benchmarks Track** and used for [Amazon KDD Cup 2024](https://www.aicrowd.com/challenges/amazon-kdd-cup-2024-multi-task-online-shopping-challenge-for-llms). Shopping MMLU is a massive multi-task benchmark for LLMs on online shopping, covering four major shopping skills, **shopping concept understanding**, **shopping knowledge reasoning**, **user behavior alignment**, and **multi-lingual abilities**. 

<img width="1604" alt="image" src="https://github.com/user-attachments/assets/38b8784e-34cb-4add-81f8-538eb91ee1e0">

You can find more detailed information about the dataset in the following links:
- The paper and supplementary materials [here](https://arxiv.org/pdf/2410.20745).
- The workshop of our KDD Cup Challenge and winning solutions [here](https://amazon-kddcup24.github.io).
- The HuggingFace Leaderboard [here](https://huggingface.co/spaces/KL4805/shopping_mmlu_leaderboard). 

## Repo Organization
```
.
├── data:                You will need to create this folder and put the evaluation data in it.
├── skill_wise_eval:     This folder contains code for evaluating a skill as a whole. 
├── task_wise_eval:      This folder contains code for evaluating a single task.
└── README.md
```

## Data
### Where to download? 
The zipfile `data.zip` contains all data in Shopping MMLU. Create a new folder `data`, and unzip the zipfile in it. 

### Data formats
We have five different types of tasks, **multiple choice**, **retrieval**, **ranking**, **named entity recognition**, and **generation**. 

Files for multiple choice questions are organized in `.csv` formats with three columns. 
- `question`: The question of the multiple choice.
- `choices`: The possible choices (4 in total) of this multiple choice.
- `answer`: The answer (within 0, 1, 2, and 3), indicating that the correct answer is `choices[answer]`.

Files for other types of tasks are organized in `.json` formats with two fields, `input_field` and `target_field`. 

## Running evaluations
### Dependencies
Our evaluation code is based on HuggingFace `transformers` with the following dependencies. 
```
transformers==4.37.0
torch==2.1.2+cu121
pandas==2.0.3
evaluate==0.4.1
sentence_transformers==2.2.2
rouge_score
sacrebleu
sacrebleu[jp]
```

### Evaluation on a Single Task
Suppose you want to evaluate `Vicuna-7B-v1.5` model on the `multiple_choice` task of `asin_compatibility`, you can do the following steps. 
```
cd task_wise_eval/
python3 hf_multi_choice.py --test_subject asin_compatibility --model_name vicuna2
# The 'model_name' argument should be set according to 'utils.py'. 
```
Other tasks in other task types involve similar processes. 
### Evaluation on a Skill as a whole
Suppose you want to evaluate `Vicuna-7B-v1.5` model on the skill of `skill1_concept`, you can do the following steps. 
```
cd skill_wise_eval/
python3 hf_skill_inference.py --model_name vicuna2 --filename skill1_concept --output_filename <your_filename>
# After inference, the output file will be saved at `skill_inference_results/skill1_concept/vicuna2_<your_filename>.json`.
python3 skill_evaluation.py --data_filename skill1_concept --output_filename vicuna2_<your_filename>
# After evaluation, the metrics will be saved at `skill_metrics/skill1_concept/vicuna2_<your_filename>_metrics.json`. 
```
Other skills involve similar processes. 

## LeaderBoard
We host a live HuggingFace leaderboard [here](https://huggingface.co/spaces/KL4805/shopping_mmlu_leaderboard). Details of the tested models can be found in the Appendix of our paper. 

We are actively maintaining the leaderboard by adding more models. If you want to test your LLM as a shopping assistant, please leave an issue in this repo. 

## Reference
```
@article{jin2024shopping,
  title={Shopping MMLU: A Massive Multi-Task Online Shopping Benchmark for Large Language Models},
  author={Jin, Yilun and Li, Zheng and Zhang, Chenwei and Cao, Tianyu and Gao, Yifan and Jayarao, Pratik and Li, Mao and Liu, Xin and Sarkhel, Ritesh and Tang, Xianfeng and others},
  journal={arXiv preprint arXiv:2410.20745},
  year={2024}
}
```
