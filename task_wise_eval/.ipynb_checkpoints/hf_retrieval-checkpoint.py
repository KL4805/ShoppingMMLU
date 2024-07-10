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
import os

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

parser = argparse.ArgumentParser()
parser.add_argument('--test_subject', type=str, default='product_keyphrase_retrieval')
parser.add_argument('--model_name', type=str, default='vicuna2')
parser.add_argument('--seed', type=int, default=-1)
parser.add_argument("--print_interval", type=int, default=20)
parser.add_argument('--max_gen_len', type=int, default=15)
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

tokenizer, model = load_tokenizer_and_model(model_name)

start_time = time.time()
print("Running %s model on %s task" % (model_name, test_subject))

ill_format = 0
total_hit = 0

filename = f'../data/retrieval/{test_subject}_dataset.json'
try:
    test_df = pd.read_json(filename, lines=True)
except:
    raise FileNotFoundError(f"{filename} does not exist. Please check the 'test_subject' argument. ")
all_samples = test_df.shape[0]

    
for i in range(all_samples):
    system_prompt = gen_system_prompt(test_subject)
    test_prompt = format_example(test_df, i)
    prompt = system_prompt + test_prompt
    if i % args.print_interval == 0:
        print("Sample %d prompt"%i, prompt)
    truth = test_df.iloc[i, -1]
    inputs = tokenizer(prompt, return_tensors = 'pt')
    inputs.input_ids = inputs.input_ids.cuda()
    if 'mistral' in model_name or 'mixtral' in model_name or model_name == 'zephyr' or model_name == 'ecellm-m' :
        generate_ids = model.generate(inputs.input_ids, max_new_tokens = args.max_gen_len, pad_token_id=2)
    elif 'qwen' in model_name:
        generate_ids = model.generate(inputs.input_ids, max_new_tokens = args.max_gen_len, pad_token_id = 151643, temperature=0.0001)
    elif 'phi' in args.model_name or args.model_name == 'ecellm-s':
        generate_ids = model.generate(inputs.input_ids, max_new_tokens = args.max_gen_len, pad_token_id = 50256, temperature=0.0001)
    else:
        generate_ids = model.generate(inputs.input_ids, max_new_tokens = args.max_gen_len)

    result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    generation = result[len(prompt):]
    if i % args.print_interval == 0:
        print("Sample %d generation"%i, generation)

    retrieved_list = generation
    retrieved_list = retrieved_list.lstrip().rstrip()
    if '\n' not in retrieved_list:
        retrieved_list = retrieved_list.split(',') 
    else:
        retrieved_list = retrieved_list.split('\n')[0]
        retrieved_list = retrieved_list.split(',') 
    retrieved_int = []
    for ret in retrieved_list:
        try:
            ret_int = int(ret)
            retrieved_int.append(ret_int)
        except:
            print(f"{ret} cannot be interpreted as int")
            continue
    if len(retrieved_int) > 3:
        retrieved_int = retrieved_int[:3]

    if i % args.print_interval == 0:
        print("Sample %d truth"%i, truth)
    hit = len(set(truth).intersection(set(retrieved_int)))
    hit /= len(truth)
    total_hit += hit
    
    if i % args.print_interval == 0:
        print("Sample %d retrieval"%i, retrieved_list)
        print("Sample %d hit"%i, hit)
        print()

print("retrieval task %s with %s:"%(test_subject, model_name))
print("Average hit rate %.4f, %d ill-formatted generations"%(total_hit / all_samples, ill_format))
print("Time cost %.4fs"%(time.time() - start_time))