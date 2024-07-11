from transformers import AutoTokenizer, AutoModelForCausalLM, MistralForCausalLM
import pandas as pd
import numpy as np
import argparse
import time
import sys
import json
import torch
from utils import *




parser = argparse.ArgumentParser()
parser.add_argument('--test_subject', type=str, default='query_named_entity_recognition')
parser.add_argument("--print_interval", type=int, default=10)
parser.add_argument("--max_gen_len", type=int, default=20)
parser.add_argument('--model_name', type=str, default='vicuna2')
parser.add_argument('--use_task_specific_prompt', action='store_true')
args = parser.parse_args()

start_time = time.time()
test_subject = args.test_subject
model_name = args.model_name
tokenizer, model = load_tokenizer_and_model(model_name)
print("Running %s on %s task."%(model_name, test_subject))

filename = f"../data/named_entity_recognition/{test_subject}_dataset.json"
try:
    test_df = pd.read_json(filename, lines=True)
except:
    raise FileNotFoundError(f"{filename} does not exist. Please modify the 'test_subject' argument. ")
all_samples = test_df.shape[0]

true_positive = 0
false_positive = 0
false_negative = 0

for i in range(all_samples):
    train_prompt = gen_system_prompt(args)
    test_prompt = format_example(test_df, i)
    prompt = train_prompt + test_prompt
    if i % args.print_interval == 0:
        print("Sample %d prompt"%i, prompt)
    label = test_df.iloc[i, -1]
    label_lower = []
    for l in label:
        label_lower.append(l.lower())

    inputs = tokenizer(prompt, return_tensors='pt')
    inputs.input_ids = inputs.input_ids.cuda()
    if 'mistral' not in model_name and 'mixtral' not in model_name:
        generate_ids = model.generate(inputs.input_ids, max_new_tokens = args.max_gen_len)
    else:
        generate_ids = model.generate(inputs.input_ids, max_new_tokens = args.max_gen_len, pad_token_id=2)
    result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    answer = result[len(prompt):]# .split('\n')[0]
    answer = answer.lstrip('\n')
    if i % args.print_interval == 0:
        print("Sample %d answer:"%i, answer)
        print("Sample %d ground truth:"%i, label)
    answer = answer.split('\n')[0].lstrip(' ').rstrip(' ')        
    answer = answer.split(',')
    answer = [a for a in answer if a != '']
    answer_lower = []
    for a in answer:
        answer_lower.append(a.lower().lstrip(' ').rstrip(' '))
    if i % args.print_interval == 0:
        print("Sample %d answer" % i, answer_lower)
    true_positive += len(set(answer_lower).intersection(set(label_lower)))
    false_positive += len(answer_lower) - len(set(answer_lower).intersection(set(label_lower)))
    false_negative += len(label_lower) - len(set(answer_lower).intersection(set(label_lower)))
    if i % args.print_interval == 0:
        print(f"Sample {i}, TP {len(set(answer_lower).intersection(set(label_lower)))}, FP {len(answer_lower) - len(set(answer_lower).intersection(set(label_lower)))}, FN {len(label_lower) - len(set(answer_lower).intersection(set(label_lower)))}")
        print()


precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)
f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0

print("Time Cost: %.4fs" % (time.time() - start_time))   

print("The average F1 score of %s on %s task is %.4f, precision %.4f, recall %.4f"%(args.model_name, test_subject, f1, precision, recall))

