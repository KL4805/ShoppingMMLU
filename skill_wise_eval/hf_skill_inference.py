from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import numpy as np
import argparse
import time
import random
import torch
import transformers
import numpy as np
from utils import *
import os

parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, help='the input file name. By default is json')
parser.add_argument('--model_name', type=str, help='the model you want to test.')
parser.add_argument('--print_interval', type=int, default=20)
parser.add_argument('--output_filename', type=str, help='Suffix to output filename')
parser.add_argument('--multi_choice_tokens', type=int, default=1, help='The maximum token length for multiple-choice questions. By default we set to 1. ')
parser.add_argument('--seed', type=int, default=-1, help='random seed, default -1 is not set')
args = parser.parse_args()
if args.seed != -1:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    transformers.set_seed(args.seed)


torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

system_prompt = "You are a helpful online shopping assistant. Please answer the following question about online shopping and follow the given instructions.\n\n"
print(f"Inferencing {args.model_name} model on the skill {args.filename}. \n")


start_time = time.time()
test_df = pd.read_json(f'../data/skills/{args.filename}.json', lines=True)
tokenizer, model = load_tokenizer_and_model(args.model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'
print(f"Skill {args.filename} has {test_df.shape[0]} samples in total. \n")


output_dict = {
    'model_output':[]
}

for i in range(test_df.shape[0]):
    
    prompt = system_prompt + test_df.iloc[i]['input_field']
    task_type = test_df.iloc[i]['task_type']

    if task_type == 'multiple-choice':
        output_len = args.multi_choice_tokens
    else:
        output_len = 100
    inputs = tokenizer(prompt, return_tensors = 'pt')
    inputs.input_ids = inputs.input_ids.cuda()
    if i % args.print_interval == 0:
        print(f"Sample {i} prompt: {prompt}")

    if 'mistral' in args.model_name or 'mixtral' in args.model_name or args.model_name == 'ecellm-m':
        generate_ids = model.generate(inputs.input_ids, max_new_tokens = output_len, pad_token_id = 2, temperature=0)
    elif 'llama2' in args.model_name:
        generate_ids = model.generate(inputs.input_ids, max_new_tokens = output_len, temperature = 0.0001)
    elif 'llama3' in args.model_name:
        generate_ids = model.generate(inputs.input_ids, max_new_tokens = output_len, pad_token_id = 128001, temperature = 0.0001)
    elif 'qwen' in args.model_name:
        generate_ids = model.generate(inputs.input_ids, max_new_tokens = output_len, pad_token_id = 151643, temperature=0.0001)
    elif 'phi' in args.model_name or 'ecellm-s' == args.model_name: 
        if task_type == 'multiple-choice':
            prompt += "\n(Please output a number only) Output: \n"
        inputs = tokenizer(prompt, return_tensors = 'pt')
        inputs.input_ids = inputs.input_ids.cuda()
        generate_ids = model.generate(inputs.input_ids, max_new_tokens = output_len, pad_token_id = 50256, temperature=0)
    else:
        generate_ids = model.generate(inputs.input_ids, max_new_tokens = output_len, temperature=0)    

    result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    result = result[0]
    generation = result[len(prompt):]
    output_dict['model_output'].append(generation)
    if i % args.print_interval == 0:
        print(f"Sample {i} answer: {generation}")
        print()

        



input_filename = args.filename.split('/')[-1].split('.')[0]
if not os.path.exists(f'skill_inference_results/{input_filename}'):
    os.makedirs(f'skill_inference_results/{input_filename}')
output_filename = f"skill_inference_results/{input_filename}/{args.model_name}_{args.output_filename}.json"
output_df = pd.DataFrame(output_dict)
output_df.to_json(output_filename, orient='records', lines=True)
end_time = time.time()
print(f"Inference of {input_filename} with {args.model_name} model takes {end_time-start_time} seconds. ")
    