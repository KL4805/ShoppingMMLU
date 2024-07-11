from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
import numpy as np
import evaluate



def accuracy(generation, truth):
    generation = generation.lstrip(' ').rstrip(' ')
    if generation == '':
        return 0
    if generation[0] == truth:
        return 1
    else:
        return 0

def hit_rate(generation, truth):
    generation = generation.lstrip().rstrip()
    if '\n' not in generation:
        retrieved_list = generation.split(',') 
    else:
        retrieved_list = generation.split('\n')[0]
        retrieved_list = generation.split(',') 
    retrieved_int = []
    for ret in retrieved_list:
        try:
            ret_int = int(ret)
            retrieved_int.append(ret_int)
        except:
            continue
    if len(retrieved_int) > 3:
        retrieved_int = retrieved_int[:3]
    hit = len(set(truth).intersection(set(retrieved_int)))
    hit /= len(truth)
    return hit

def rougel(generation, truth):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(generation, truth)
    return scores['rougeL'].fmeasure

def sent_transformer(generation, truth, sent_transformer_model):
    generation_embedding = sent_transformer_model.encode([generation])[0]

    if isinstance(truth, str):
        truth_embedding = sent_transformer_model.encode([truth])[0]
        score = ((generation_embedding * truth_embedding).sum()) 
        score /= (np.linalg.norm(generation_embedding, ord=2) * np.linalg.norm(truth_embedding, ord=2))
        if score > 0:
            return score
        else:
            return 0
    else:
        scores = []
        for label_item in truth:
            truth_embedding = sent_transformer_model.encode([label_item])[0]
            score_ = (generation_embedding * truth_embedding).sum()
            score_ /= (np.linalg.norm(generation_embedding, ord=2) * np.linalg.norm(truth_embedding, ord=2))
            scores.append(score_)
        if np.mean(scores) > 0:
            return np.mean(scores)
        else:
            return 0

def tp_fp_fn(generation, truth):
    # this is used to evaluate named entity recognition
    generation = generation.lstrip('\n')
    generation = generation.split('\n')[0].lstrip(' ').rstrip(' ')
    generation = generation.split(',')
    answer = [a for a in generation if a != '']
    answer_lower = []
    for a in answer:
        answer_lower.append(a.lower().lstrip(' ').rstrip(' '))
    truth_lower = []
    for l in truth:
        truth_lower.append(l.lower())
    true_positive = len(set(answer_lower).intersection(set(truth_lower)))
    false_positive = len(answer_lower) - true_positive
    false_negative = len(truth_lower) - true_positive
    return true_positive, false_positive, false_negative

def compute_f1_score(tp_fp_fn_list):
    total_tp = 0
    total_fp = 0
    total_fn = 0
    for tp, fp, fn in tp_fp_fn_list:
        total_tp += tp
        total_fp += fp
        total_fn += fn
    precision = total_tp / (total_tp + total_fp)
    recall = total_tp / (total_tp + total_fn)
    if precision + recall == 0:
        return 0
    else:
        return 2 * precision * recall / (precision + recall)
        
def ndcg_eval(generation, truth):
    if '\n' in generation: 
        generation = generation.lstrip('\n').rstrip('\n')
        generation = generation.split('\n')[0]
    ranked_list_str = generation.lstrip().rstrip().split(',')
    ranked_list = []
    for x in ranked_list_str:
        try: 
            ranked_list.append(int(x))
        except:
            continue
    if len(ranked_list) > len(truth):
        ranked_list = ranked_list[:len(truth)]
    return ndcg(ranked_list, truth)
        
    
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

def bleu(generation, truth, jp = False):
    sacrebleu = evaluate.load('sacrebleu')
    generation = generation.lstrip('\n').rstrip('\n').split('\n')[0]
    candidate = [generation]
    reference = [[truth]]
    if not jp:
        score = sacrebleu.compute(predictions=candidate, references=reference, lowercase=True)['score']/100
    else:
        score = sacrebleu.compute(predictions=candidate, references=reference, lowercase=True, tokenize='ja-mecab')['score']/100
    return score
    
    
    
    

