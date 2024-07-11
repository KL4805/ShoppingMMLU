from transformers import AutoTokenizer, AutoModelForCausalLM, MistralForCausalLM
import pandas as pd
import numpy as np
import argparse
import time
import sys
import json
from sentence_transformers import SentenceTransformer
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import evaluate
from evaluate import load
from rouge_score import rouge_scorer
from utils import *

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)




parser = argparse.ArgumentParser()
parser.add_argument('--test_subject', type=str, default='pt_name_explain')
parser.add_argument('--print_interval', type=int, default=20)
parser.add_argument('--max_gen_len', type=int, default=128)
parser.add_argument('--model_name', type=str, default='vicuna2')
parser.add_argument('--use_task_specific_prompt', action='store_true')
args = parser.parse_args()


start_time = time.time()
test_subject = args.test_subject
model_name = args.model_name
tokenizer, model = load_tokenizer_and_model(model_name)

print("Running %s on %s task."%(model_name, test_subject))

# metric
if 'extraction' in args.test_subject:
    metric = 'rougel'
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    print("Metric is ROUGE-L score")
elif 'translation' in args.test_subject:
    metric = 'bleu'
    sacrebleu = evaluate.load('sacrebleu')
    print("Metric is BLEU score")
elif 'multilingual' in args.test_subject:
    metric = 'multilingual-sent-transformer'
    eval_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2').cuda()
    print("Metric is multilingual sentence transformer similarity")
else:
    metric = 'sent-transformer'
    eval_model = SentenceTransformer('all-MiniLM-L6-v2').cuda()
    print("Metric is sentence transformer similarity.")


filename = f"../data/generation/{test_subject}_dataset.json"
try:
    test_df = pd.read_json(filename, lines=True)
except:
    raise FileNotFoundError(f"{filename} does not exist. Please modify the 'test_subject' argument. ")
total_score = 0
all_samples = test_df.shape[0]
    

for i in range(all_samples):
    train_prompt = gen_system_prompt(args)
    test_prompt = format_example(test_df, i)
    prompt = train_prompt + test_prompt

    if i % args.print_interval == 0:
        print("Sample %d prompt"%i, prompt)
    label = test_df.iloc[i, -1]
    inputs = tokenizer(prompt, return_tensors = 'pt')
    inputs.input_ids = inputs.input_ids.cuda()
    if 'mistral' in args.model_name or 'mixtral' in args.model_name or 'zephyr' in args.model_name or 'ecellm-m' == args.model_name:
        generate_ids = model.generate(inputs.input_ids, max_new_tokens = args.max_gen_len, pad_token_id = 2)
    elif 'qwen' in args.model_name:
        generate_ids = model.generate(inputs.input_ids, max_new_tokens = args.max_gen_len, pad_token_id = 151643, temperature=0.0001)
    elif 'phi' in args.model_name or args.model_name == 'ecellm-s':
        generate_ids = model.generate(inputs.input_ids, max_new_tokens = args.max_gen_len, pad_token_id = 50256, temperature=0.0001)
    else:
        generate_ids = model.generate(inputs.input_ids, max_new_tokens = args.max_gen_len)
    result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    generation = result[len(prompt):]# .split('\n')[0]

    if i % args.print_interval == 0:
        print("Sample %d answer"%i, generation)
        print("Sample %d ground truth"%i, label)
    if metric == 'rougel':
        scores = scorer.score(generation, label)
        total_score += scores['rougeL'].fmeasure
        current_score = scores['rougeL'].fmeasure
    elif metric in ['sent-transformer', 'multilingual-sent-transformer']:
        if isinstance(label, str):
            truth_embedding = eval_model.encode([label])[0]
            generation_embedding = eval_model.encode([generation])[0]
            current_score = ((generation_embedding * truth_embedding).sum()) 
            current_score /= (np.linalg.norm(generation_embedding, ord=2) * np.linalg.norm(truth_embedding, ord=2))
            total_score += current_score
        else:
            scores = []
            generation_embedding = eval_model.encode([generation])[0]
            for label_item in label:
                truth_embedding = eval_model.encode([label_item])[0]
                score_ = (generation_embedding * truth_embedding).sum()
                score_ /= (np.linalg.norm(generation_embedding, ord=2) * np.linalg.norm(truth_embedding, ord=2))
                scores.append(score_)
            current_score = np.mean(scores)
            total_score += current_score
    elif metric == 'bleu':
        # usage: sacrebleu.compute(predictions=xxx, references=yyy)
        # reference can be multiple lists of sentences
        # candidate is a list of sentences
        generation = generation.lstrip('\n').rstrip('\n').split('\n')[0]
        candidate = [generation]
        reference = [[label]]
        if 'JP' not in args.test_subject:
            # japanese needs a different tokenizer
            tokenize='13a'
        else:
            tokenize='ja-mecab'
        current_score = sacrebleu.compute(predictions=candidate, references=reference, lowercase=True,tokenize=tokenize)['score']/100
        total_score += current_score
    else:
        raise NotImplementedError('metric not implemented')
    if i % args.print_interval == 0:
        print("Sample %d score"%i, current_score)
        print()



print("The average score of %s on %s task is %.4f"%(model_name, test_subject, total_score / all_samples))
print("Time Cost: %.4fs" % (time.time() - start_time))   