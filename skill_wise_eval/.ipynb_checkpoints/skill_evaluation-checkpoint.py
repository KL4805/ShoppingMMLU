import pandas as pd
import numpy as np
import argparse
from sentence_transformers import SentenceTransformer
from metrics import *
import os


parser = argparse.ArgumentParser()
parser.add_argument('--data_filename', type=str, help='The data file name that contains answers')
parser.add_argument('--output_filename', type=str, help='The file name that contains model generated answers')
parser.add_argument('--print_interval', type=int, default=20)
args = parser.parse_args()

data_df = pd.read_json(f'../data/skills/{args.data_filename}.json', lines=True)
output_df = pd.read_json(f'skill_inference_results/{args.data_filename}/{args.output_filename}.json', lines=True)
assert data_df.shape[0] == output_df.shape[0]


per_task_metrics = {}
eval_model = SentenceTransformer('all-MiniLM-L6-v2').cuda()
eval_model_multilingual = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2').cuda()


for i in range(data_df.shape[0]):
    task_name = data_df.iloc[i]['task_name']
    generation = output_df.iloc[i]['model_output']
    truth = data_df.iloc[i]['output_field']
    metric = data_df.iloc[i]['metric']

    if task_name not in per_task_metrics:
        per_task_metrics[task_name] = {
            'metric': metric, 
            'sample_score': []
        }

    if metric == 'accuracy':
        acc = accuracy(generation, str(truth))
        per_task_metrics[task_name]['sample_score'].append(acc)
        if i % args.print_interval == 0:
            print(f"Sample {i}, generation: {generation}")
            print(f"Sample {i}, truth: {truth}")
            print(f"Metric ({metric}): {acc}")
            print()
            
    elif metric == 'hit rate@3':
        hit = hit_rate(generation, truth)
        per_task_metrics[task_name]['sample_score'].append(hit)
        if i % args.print_interval == 0:
            print(f"Sample {i}, generation: {generation}")
            print(f"Sample {i}, truth: {truth}")
            print(f"Metric ({metric}): {hit}")
            print()
            
    elif metric == 'rougel':
        rouge_metric = rougel(generation, truth)
        per_task_metrics[task_name]['sample_score'].append(rouge_metric)
        if i % args.print_interval == 0:
            print(f"Sample {i}, generation: {generation}")
            print(f"Sample {i}, truth: {truth}")
            print(f"Metric ({metric}): {rouge_metric}")
            print()
            
    elif metric == 'sent-transformer':
        sent_transformer_score = sent_transformer(generation, truth, eval_model)
        per_task_metrics[task_name]['sample_score'].append(sent_transformer_score)
        if i % args.print_interval == 0:
            print(f"Sample {i}, generation: {generation}")
            print(f"Sample {i}, truth: {truth}")
            print(f"Metric ({metric}): {sent_transformer_score}")
            print()
    elif metric == 'multilingual-sent-transformer':
        sent_transformer_score = sent_transformer(generation, truth, eval_model_multilingual)
        per_task_metrics[task_name]['sample_score'].append(sent_transformer_score)
        if i % args.print_interval == 0:
            print(f"Sample {i}, generation: {generation}")
            print(f"Sample {i}, truth: {truth}")
            print(f"Metric ({metric}): {sent_transformer_score}")
            print()
        
    elif metric == 'micro f1':
        # we need to record tp, fp, fn
        tp, fp, fn = tp_fp_fn(generation, truth)
        per_task_metrics[task_name]['sample_score'].append((tp, fp, fn))
        if i % args.print_interval == 0:
            print(f"Sample {i}, generation: {generation}")
            print(f"Sample {i}, truth: {truth}")
            print(f"Metric ({metric}): tp {tp}, fp {fp}, fn {fn}")
            print() 

    elif metric == 'ndcg':
        ndcg_val = ndcg_eval(generation, truth)
        per_task_metrics[task_name]['sample_score'].append(ndcg_val)
        if i % args.print_interval == 0:
            print(f"Sample {i}, generation: {generation}")
            print(f"Sample {i}, truth: {truth}")
            print(f"Metric ({metric}): {ndcg_val}")
            print() 

    elif metric == 'bleu':
        bleu_val = bleu(generation, truth)
        per_task_metrics[task_name]['sample_score'].append(bleu_val)
        if i % args.print_interval == 0:
            print(f"Sample {i}, generation: {generation}")
            print(f"Sample {i}, truth: {truth}")
            print(f"Metric ({metric}): {bleu_val}")
            print() 
    elif metric == 'jp-bleu':
        bleu_val = bleu(generation, truth, jp=True)
        per_task_metrics[task_name]['sample_score'].append(bleu_val)
        if i % args.print_interval == 0:
            print(f"Sample {i}, generation: {generation}")
            print(f"Sample {i}, truth: {truth}")
            print(f"Metric ({metric}): {bleu_val}")
            print() 

    

# aggregate per_task_metric
for k in per_task_metrics:
    if per_task_metrics[k]['metric'] != 'micro f1':
        print(k, len(per_task_metrics[k]['sample_score']))
        per_task_metrics[k]['overall_metric'] = np.mean(per_task_metrics[k]['sample_score'])
    else:
        per_task_metrics[k]['overall_metric'] = compute_f1_score(per_task_metrics[k]['sample_score'])

overall_metrics = {
    'task_name': [],
    'metric': [],
    'overall_score': []
}
for k in per_task_metrics:
    overall_metrics['task_name'].append(k)
    overall_metrics['metric'].append(per_task_metrics[k]['metric'])
    overall_metrics['overall_score'].append(per_task_metrics[k]['overall_metric'])
track_wise_score = np.mean(overall_metrics['overall_score'])
overall_metrics['task_name'].append('track_wise')
overall_metrics['metric'].append('track_wise')
overall_metrics['overall_score'].append(track_wise_score)
overall_metrics_df = pd.DataFrame(overall_metrics)

if not os.path.exists(f'skill_metrics/{args.data_filename}/'):
    os.makedirs(f'skill_metrics/{args.data_filename}/')

overall_metrics_df.to_json(f"skill_metrics/{args.data_filename}/{args.output_filename}_metrics.json", orient='records', lines=True)
print(f"The overall score of output file '{args.output_filename}' on skill '{args.data_filename}' is {track_wise_score}")

    
        

        