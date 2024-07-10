from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import numpy as np
import argparse
import time
import random
import torch
import transformers
import numpy as np
from utils import load_tokenizer_and_model

# This type of tasks ask an LLM to generate a re-ranked list of indices
# the evaluation metric is NDCG
# This applies to a range of re-ranking tasks
# such as recommendation

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        if entry == 'pt':
            entry = 'product type'
        s += " " + entry
    return s

def format_example(df, idx):
    prompt = df.iloc[idx, 0]
    answer = df.iloc[idx, 1]
    return prompt

def ndcg(ranked_list, weight):
    idcg = 0
    dcg = 0
    for i in range(len(ranked_list)):
        position = i+1
        if ranked_list[i]-1 < len(weight):
            relevance = weight[ranked_list[i]-1]
        else:
            relevance = 0
        dcg += (np.power(2, relevance) - 1)/np.log2(position+1)
    weight.sort(reverse=True)
    for i in range(len(weight)):
        position = i+1
        relevance = weight[i]
        idcg += (np.power(2, relevance) - 1)/ np.log2(position+1)

    return dcg/idcg

def is_permutation(arr):
    n = len(arr)
    expected_set = set(range(1, n+1))
    return set(arr) == expected_set

parser = argparse.ArgumentParser()
parser.add_argument('--test_subject', type=str, default='query_product_ranking')
parser.add_argument("--filename", type=str)
parser.add_argument('--model_name', type=str, default='vicuna2')
parser.add_argument('--seed', type=int, default=-1)
parser.add_argument("--print_interval", type=int, default=20)
parser.add_argument('--max_gen_len', type=int, default=20)
parser.add_argument('--use_task_specific_prompt', action='store_true')
args = parser.parse_args()


def gen_system_prompt(subject):
    if args.use_task_specific_prompt:
        prompt = "You are required to perform the task of %s. Please follow the given instructions.\n\n"%format_subject(subject)
    else:
        prompt = 'You are a helpful online shopping assistant. Please answer the following question about online shopping and follow the given instructions.\n\n'
    return prompt

seed = args.seed
if seed > 0:
    random.seed(seed)
    transformers.set_seed(seed)
model_name = args.model_name
test_subject = args.test_subject
print_interval = args.print_interval


start_time = time.time()
print("Running %s model on %s task" % (model_name, test_subject))


tokenizer, model = load_tokenizer_and_model(model_name)

ill_format = 0
total_ndcg = 0
filename = f'../data/ranking/{test_subject}_dataset.json'
try:
    test_df = pd.read_json(filename, lines=True)
except:
    raise FileNotFoundError(f"{filename} does not exist. Please check the 'test_subject' argument.")
all_samples = test_df.shape[0]


for i in range(all_samples):
    train_prompt = gen_system_prompt(test_subject)
    test_prompt = format_example(test_df, i)
    prompt = train_prompt + test_prompt
    if i % args.print_interval == 0:
        print("Sample %d prompt"%i, prompt)
    weight = test_df.iloc[i, -1]
    inputs = tokenizer(prompt, return_tensors = 'pt')
    inputs.input_ids = inputs.input_ids.cuda()
    with torch.no_grad():
        generate_ids = model.generate(inputs.input_ids, max_new_tokens = args.max_gen_len, temperature=0.001)
    result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    generation = result[len(prompt):]
    if i % args.print_interval == 0:
        print("Sample %d generation"%i, generation)
    answer = generation
    if '\n' in answer:
        answer = answer.lstrip('\n').rstrip('\n')
        answer = answer.split('\n')[0]
    ranked_list_str = answer.lstrip().rstrip().split(',')
    ranked_list = []
    for x in ranked_list_str:
        try:
            ranked_list.append(int(x))
        except:
            ill_format += 1
            if i % args.print_interval == 0:
                print("Sample %d ill format"%i)
            continue
    
    if len(ranked_list) != len(weight):
        ill_format += 1
        if i % args.print_interval == 0:
            print("Sample %d ill format"%i)
        # continue
    if not is_permutation(ranked_list):
        ill_format += 1
        if i % args.print_interval == 0:
            print("Sample %d ill format"%i)
        # continue
    # start computing
    if i % args.print_interval == 0:
        print("Sample %d weight"%i, weight)
    ndcg_val = ndcg(ranked_list, weight)
    total_ndcg += ndcg_val
    if i % args.print_interval == 0:
        
        print("Sample %d ranking"%i, ranked_list)
        print("Sample %d ndcg"%i, ndcg_val)
        print()

print("ranking task %s with %s:"%(test_subject, model_name))
print("Average NDCG %.4f, %d ill-formatted generations"%(total_ndcg / all_samples, ill_format))
print("Time cost %.4fs"%(time.time() - start_time))
