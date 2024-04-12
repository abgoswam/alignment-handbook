from numpy import random
import argparse
import re
import json
import os
# import deepspeed
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import time

local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))

if not torch.cuda.is_available():
    raise ValueError("This script requires a GPU to run.")

gpu_device = torch.device(f'cuda:{local_rank}')

def print_once(*args, **kwargs):
    if local_rank == 0:
        print(*args, **kwargs)


def generate_prompt_landmark(max_tokens, seed, prefix_fraction): #(n_garbage, seed, n_garbage_prefix):
    """Generates a text file and inserts an passkey at a random position."""
    n_garbage = (max_tokens - 32 - 26 - 11) // 25
    n_garbage_prefix = int(prefix_fraction * n_garbage)

    rnd_state = random.get_state()
    random.seed(seed)
    n_garbage_suffix = n_garbage - n_garbage_prefix

    task_description = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."
    garbage = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
    garbage_prefix = garbage * n_garbage_prefix
    garbage_suffix = garbage * n_garbage_suffix
    pass_key = random.randint(1, 50000)
    information_line = f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key."
    final_question = "What is the pass key? The pass key is"
    lines = [
        task_description,
        garbage_prefix,
        information_line,
        garbage_suffix,
        final_question,
    ]
    random.set_state(rnd_state)
    return "\n".join(lines), pass_key


def extract_int(response):
    # Search for the first sequence of digits in the response
    match = re.search(r'\d+', response)
    # Return the integer if a match is found, otherwise return None
    return int(match.group()) if match else None


def model_inference(model, tokenizer, prompt_text):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(gpu_device)
    outputs = model.generate(**inputs, max_new_tokens=20)
    prompt_text_token_len = inputs['input_ids'].shape[1]
    generated_text = tokenizer.decode(outputs[0][prompt_text_token_len:], skip_special_tokens=True)

    pass_key = extract_int(generated_text)
    if pass_key is None:
        pass_key = -1
    return pass_key, prompt_text_token_len


def plot_heatmap(results, vmax, max_tokens, output_dir):
    df = pd.DataFrame(results)
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])
    
    pivot_table = pd.pivot_table(df, values='Score', index=['Document Depth', 'Context Length'], aggfunc='mean').reset_index() # This will aggregate
    pivot_table = pivot_table.pivot(index="Document Depth", columns="Context Length", values="Score")
    # Create the heatmap with better aesthetics
    plt.figure(figsize=(17.5, 8))  # Can adjust these dimensions as needed
    sns.heatmap(
        pivot_table,
        # annot=True,
        fmt="g",
        cmap=cmap,
        vmin=0,  # Set the colormap minimum to 0
        vmax=vmax, # Set the colormap maximum to the maximum possible score  
        cbar_kws={'label': 'Score'}
    )

    # More aesthetics
    plt.xlabel('Token Limit')  # X-axis label
    plt.ylabel('Depth Percent')  # Y-axis label
    plt.xticks(rotation=45)  # Rotates the x-axis labels to prevent overlap
    plt.yticks(rotation=0)  # Ensures the y-axis labels are horizontal
    plt.tight_layout()  # Fits everything neatly into the figure area
    # save
    plt.savefig(os.path.join(output_dir, f"heatmap_{max_tokens}.png"))


def estimate_passkey_retrieval_accuracy(model, tokenizer, trials, context_size):
    avg_tokens = None
    results = []
    for prefix_numerator in range(0, 11):
        correct_cnt = 0
        total_tokens = 0
        for k in range(trials):
            prompt_text, pass_key = generate_prompt_landmark(context_size, k, prefix_numerator/10.0)
            pred, length = model_inference(model, tokenizer, prompt_text)
            correct_cnt += 1 if pred == pass_key else 0
            total_tokens += length
        accuracy = correct_cnt/trials
        avg_tokens = total_tokens//trials if avg_tokens is None else avg_tokens
        depth = prefix_numerator/10.0
        print_once(f"token length {avg_tokens}, depth {depth}, accuracy {accuracy}")
        results.append({"Context Length": avg_tokens, "Document Depth": round(depth*100, -1), "Score": correct_cnt})
    return results


def str2bool(v):
    yes = {'yes', 'true', 't', 'y', '1'}
    no = {'no', 'false', 'f', 'n', '0'}
    if v.lower() in yes:
        return True
    elif v.lower() in no:
        return False
    else:
        raise argparse.ArgumentTypeError(f'Expected one of {yes} or {no}, but got {v}.')


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, default=2**17)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--max_position_embeddings", type=int, default=0)
    parser.add_argument("--sliding_window", type=int, default=0)
    parser.add_argument("--seed", type=int, default=98052)
    parser.add_argument("--use_flash_attention", type=str, default="true", help="Whether to use flash attention or not.")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--output_dir", type=str)
    args, unknown_args = parser.parse_known_args()
    print(f"known_args: {args}")
    print(f"unknown_args: {unknown_args}")

    random.seed(args.seed)
    config_kwargs = {}
    if args.max_position_embeddings > 0:
        config_kwargs["max_position_embeddings"] = args.max_position_embeddings
    if args.sliding_window > 0:
        config_kwargs["sliding_window"] = args.sliding_window

    checkpoint = args.model_path
    config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True, **config_kwargs)

    if str2bool(args.use_flash_attention):
        try:
            torch_dtype = config.torch_dtype if config.torch_dtype in [torch.float16, torch.bfloat16] else torch.float16
            model = AutoModelForCausalLM.from_pretrained(checkpoint, config=config, trust_remote_code=True, torch_dtype=torch_dtype, attn_implementation="flash_attention_2")
        except Exception as e:
            print(f"Failed to load model from {checkpoint} using flash_attention_2. Trying to load without flash attention.")
            model = AutoModelForCausalLM.from_pretrained(checkpoint, config=config, trust_remote_code=True, torch_dtype=config.torch_dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(checkpoint, config=config, trust_remote_code=True, torch_dtype=config.torch_dtype)

    # ds_engine = deepspeed.init_inference(model, tensor_parallel={"tp_size": world_size}, dtype=config.torch_dtype, replace_with_kernel_inject=False) #todo: would be nice to have replace_with_kernel_inject=True but it gives cuda compilation errors
    # model = ds_engine.module
    model.to(gpu_device)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    print_once(f"tokenizer.pad_token: {tokenizer.pad_token}")
    if model.config.pad_token_id is None and model.config.eos_token_id is not None:
        model.config.pad_token_id = model.config.eos_token_id
    print_once(f"model.config.pad_token_id: {model.config.pad_token_id}")
    if model.generation_config.pad_token_id is None and model.generation_config.eos_token_id is not None:
        model.generation_config.pad_token_id = model.generation_config.eos_token_id

    inputs = tokenizer("Fun fact:", return_tensors="pt").to(gpu_device)
    outputs = model.generate(**inputs, max_new_tokens=20)
    print_once(tokenizer.decode(outputs[0], skip_special_tokens=True))

    result_list = list()
    # descending order reduces chance of OOM 
    # length_list = [2**i for i in range(17, 9, -1) if 2**i <= args.max_length] 
    length_list = [2**i for i in range(17, 9, -1) if 2**i <= args.max_length] 

    print(f"length_list : {length_list}")

    for context_size in length_list:
        try:
            accuracies = estimate_passkey_retrieval_accuracy(model, tokenizer, args.trials, context_size)
            result_list.extend(accuracies)
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                # Handle out of memory error gracefully
                print(f"CUDA out of memory. context_size: {context_size}")
                # Additional actions like freeing up memory or reducing batch size can be taken here
            else:
                # Re-raise the exception if it's not related to CUDA out of memory
                raise

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        output_file = os.path.join(args.output_dir,"results.jsonl")
        with open(output_file, "w") as out:
            for result in result_list:
                out.write(json.dumps(result) + "\n")
        plot_heatmap(result_list, args.trials, args.max_length, args.output_dir)


if __name__ == "__main__":
    print(f"local_rank: {local_rank}, world_size: {world_size}")
    main()