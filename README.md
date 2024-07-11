# ShopBench
This is the repository for 'ShopBench: An Online Shopping Benchmark for LLMs', used for [Amazon KDD Cup 2024](https://www.aicrowd.com/challenges/amazon-kdd-cup-2024-multi-task-online-shopping-challenge-for-llms). ShopBench is a massive multi-task benchmark for LLMs on online shopping, covering four major shopping skills, **shopping concept understanding**, **shopping knowledge reasoning**, **user behavior alignment**, and **multi-lingual abilities**. 

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

### Data formats
We have five different types of tasks, **multiple choice**, **retrieval**, **ranking**, **named entity recognition**, and **generation**. 

Files for multiple choice questions are organized in `.csv` formats with three columns. 
- `question`: The question of the multiple choice.
- `choices`: The possible choices (4 in total) of this multiple choice.
- `answer`: The answer (within 0, 1, 2, and 3), indicating that the correct answer is `choices[answer]`.

Files for other types of tasks are organized in `.json` formats with two fields, `input_field` and `target_field`. 

## Running evaluations
### Evaluation on a Single Task

### Evaluation on a Skill

## Reference
