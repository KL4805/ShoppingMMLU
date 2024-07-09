from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import numpy as np
import argparse
import time
import random
import torch
import transformers
from utils import load_tokenizer_and_model


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        if entry == 'pt':
            entry='product type'
        s += " " + entry
    return s
    
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)


parser = argparse.ArgumentParser()
parser.add_argument('--test_subject', type=str, default='attribute_selection')
parser.add_argument('--model_name', type=str, default='vicuna2')
parser.add_argument('--seed', type=int, default=-1)
parser.add_argument("--print_interval", type=int, default=20)
parser.add_argument('--use_task_specific_prompt', action='store_true')
parser.add_argument('--use_letter_choices', action='store_true')
args = parser.parse_args()

seed = args.seed
if seed != -1:
    random.seed(seed)
    transformers.set_seed(seed)

model_name = args.model_name
test_subject = args.test_subject
print_interval = args.print_interval
if not args.use_letter_choices:
    if 'review_rating_prediction' not in test_subject:
        choices = ['0', '1', '2', '3']
    else:
        choices = ['1', '2', '3', '4', '5']
else:
    choices = ['A', 'B', 'C', 'D']

def format_example(df, idx):
    prompt = df.iloc[idx, 0]
    k = 4
    if 'review_rating_prediction' not in test_subject:
        candidates = eval(df.iloc[idx, 1])
        for j in range(k):
            if args.use_letter_choices: 
                prompt += "\n({}) {}".format(choices[j], candidates[j])
            else:
                prompt += "\n{}. {}".format(choices[j], candidates[j])
    if args.use_letter_choices:
        prompt += '\n\nPlease answer the question with a single letter indicating the choice. '
    prompt += "\nAnswer: "
    return prompt

def gen_system_prompt(subject):
    if args.use_task_specific_prompt:
        if subject != 'review_rating_prediction':
            prompt = 'The following is a multiple choice question about {}.\n\n'.format(format_subject(subject))
        else:
            prompt = 'The following is a review rating prediction question.\n\n'
    else:
        prompt = 'You are a helpful online shopping assistant. Please answer the following question about online shopping and follow the given instructions.\n\n'
    return prompt

start_time = time.time()
    
print("Running %s model on %s task" % (model_name, test_subject))
tokenizer, model = load_tokenizer_and_model(model_name)

test_df = pd.read_csv("../data/multiple_choice/%s_dataset.csv"%test_subject)
correct = 0
ill_format = 0
all_samples = test_df.shape[0]


for i in range(all_samples):
    few_shot_prompt = gen_system_prompt(test_subject)
    test_prompt = format_example(test_df, i)
    prompt = few_shot_prompt + test_prompt
    

    label = test_df.iloc[i, -1]
    if 'review_rating_prediction' not in test_subject:
        label = choices[int(label)]

    if ('phi' in args.model_name or args.model_name == 'ecellm-s') and not args.use_letter_choices:
        prompt += "\n(Please output a number only) Output: \n"
    if i % print_interval == 0:
        print("Sample %d"%i, prompt)
    inputs = tokenizer(prompt, return_tensors='pt')
    inputs.input_ids = inputs.input_ids.cuda()

    if 'mistral' in args.model_name or 'zephyr' in args.model_name or 'mixtral' in args.model_name or 'ecellm-m' == args.model_name:
        generate_ids = model.generate(inputs.input_ids, max_new_tokens=4, temperature=0, pad_token_id=2)
    elif 'qwen' in args.model_name:
        generate_ids = model.generate(inputs.input_ids, max_new_tokens = 4, pad_token_id = 151643, temperature=0.0001)
    elif 'llama' in args.model_name or args.model_name == 'ecellm-l':
        generate_ids = model.generate(inputs.input_ids, max_new_tokens = 4, pad_token_id = 151643, temperature=0.0001)
    elif 'phi' in args.model_name or args.model_name == 'ecellm-s':
        generate_ids = model.generate(inputs.input_ids, max_new_tokens = 4, pad_token_id = 50256, temperature=0.0001)
    else:
        generate_ids = model.generate(inputs.input_ids, max_new_tokens=4, temperature=0)
    generation = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    answer = generation[len(prompt):]
    if not args.use_letter_choices:
        for k in answer:
            if k.isnumeric():
                answer = k
                break
    else:
        for k in answer:
            if k.isalnum():
                answer = k
                break
    if answer == str(label):
        correct += 1
    elif answer not in choices:
        print(f'ill format generation {answer}')
        ill_format += 1
    if i % print_interval == 0:
        print(f"Sample {i}, pred {answer}, label {label}")
        print()

    
print("%s model's accuracy on %s task is %.4f" % (model_name, test_subject, correct/all_samples))
print("There are %d ill-formatted examples out of %d" % (ill_format, all_samples))
print("Time Cost: %.4fs"%(time.time() - start_time))