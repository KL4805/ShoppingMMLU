from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_tokenizer_and_model(model_name):
    if 'qwen' in model_name or 'gemma' in model_name or 'llama3' in model_name:
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)
    if model_name == 'llama':
        model_path = '../llama1'
    if model_name == 'llama2-7b':
        model_path = 'meta-llama/Llama-2-7b-hf'
    if model_name == 'llama2-13b':
        model_path = 'meta-llama/Llama-2-13b-hf'
    if model_name == 'llama2-70b':
        model_path = 'meta-llama/Llama-2-70b-hf'
    if model_name == 'llama2-7b-chat':
        model_path = 'meta-llama/Llama-2-7b-chat-hf'
    if model_name == 'llama2-13b-chat':
        model_path = 'meta-llama/Llama-2-13b-chat-hf'
    if model_name == 'llama2-70b-chat':
        model_path = 'meta-llama/Llama-2-70b-chat-hf'
    if model_name == 'alpaca':
        model_path = '../alpaca'
    if model_name == 'vicuna1':
        model_path = 'lmsys/vicuna-7b-v1.3'
    if model_name == 'vicuna2':
        model_path = 'lmsys/vicuna-7b-v1.5'
    if model_name == 'vicuna2-13b':
        model_path = 'lmsys/vicuna-13b-v1.5'
    if model_name == 'yi6b':
        model_path = '01-ai/Yi-6B'
    if model_name == 'zephyr':
        model_path = 'HuggingFaceH4/zephyr-7b-beta'
    if model_name == 'mistral':
        model_path = 'mistralai/Mistral-7B-v0.1'
    if model_name == 'falcon7b':
        model_path = 'tiiuae/falcon-7b'
    if model_name == 'qwen0.5b':
        model_path = 'Qwen/Qwen1.5-0.5B'
    if model_name == 'qwen1.8b':
        model_path = 'Qwen/Qwen1.5-1.8B'
    if model_name == 'qwen4b':
        model_path = 'Qwen/Qwen1.5-4B'
    if model_name == 'qwen7b':
        model_path = 'Qwen/Qwen1.5-7B'
    if model_name == 'qwen14b':
        model_path = 'Qwen/Qwen1.5-14B'
    if model_name == 'qwen72b':
        model_path = 'Qwen/Qwen1.5-72B'
    if model_name == 'qwen4b-chat':
        model_path = 'Qwen/Qwen1.5-4B-Chat'
    if model_name == 'qwen7b-chat':
        model_path = 'Qwen/Qwen1.5-7B-Chat'
    if model_name == 'qwen14b-chat':
        model_path = 'Qwen/Qwen1.5-14B-Chat'
    if model_name == 'mistral-instruct':
        model_path = 'mistralai/Mistral-7B-Instruct-v0.2'
    if model_name == 'mixtral-8x7b':
        model_path = "mistralai/Mixtral-8x7B-v0.1"
    if model_name == 'phi2':
        model_path = 'microsoft/phi-2'
    if model_name == 'gemma2b':
        model_path = 'google/gemma-2b'
    if model_name == 'gemma2b-it':
        model_path = 'google/gemma-2b-it'
    if model_name == 'gemma7b':
        model_path = 'google/gemma-7b'
    if model_name == 'gemma7b-it':
        model_path = 'google/gemma-7b-it'
    if model_name == 'ecellm-m':
        model_path = 'NingLab/eCeLLM-M'
    if model_name == 'ecellm-s':
        model_path = 'NingLab/eCeLLM-S'
    if model_name == 'llama3-8b':
        model_path = 'meta-llama/Meta-Llama-3-8B'
    if model_name == 'llama3-8b-instruct':
        model_path = 'meta-llama/Meta-Llama-3-8B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if 'mixtral' in model_name:
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype='auto', trust_remote_code=True, attn_implementation="flash_attention_2")
    elif 'gemma' in model_name:
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.float16, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.float16, trust_remote_code=True)

    return tokenizer, model