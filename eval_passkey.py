import random
import argparse
import re
import json
import os
import traceback

#from transformers.utils import logging
from transformers import pipeline as hf_pipeline
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm

if not torch.cuda.is_available():
    raise ValueError("This script requires a GPU to run.")

gpu_device = torch.device('cuda')


def generate_prompt(max_tokens=16384):
    """Generates a text file and inserts an execute line at a random position."""
    # n_garbage_prefix = random.randint(0, n_garbage)
    n_garbage_total = (max_tokens - 32 - 26 - 11) // 25
    n_garbage_prefix = random.randint(0, n_garbage_total)
    n_garbage_suffix = n_garbage_total - n_garbage_prefix

    task_description = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there." # 32 tokens
    garbage = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again." # 25 tokens
    garbage_prefix = garbage * n_garbage_prefix
    garbage_suffix = garbage * n_garbage_suffix
    pass_key = random.randint(1, 50000)
    information_line = f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key." # 26 tokens
    final_question = "What is the pass key? The pass key is" # 11 tokens
    lines = [
        task_description,
        garbage_prefix,
        information_line,
        garbage_suffix,
        final_question
    ]
    return "\n".join(lines), pass_key


def extract_int(response):
    # Search for the first sequence of digits in the response
    match = re.search(r'\d+', response)
    # Return the integer if a match is found, otherwise return None
    return int(match.group()) if match else None


def model_inference(pipeline, prompt_text):
    responses = pipeline(prompt_text)
    assert len(responses) == 1
    response = responses[0]
    assert 'generated_text' in response
    generated_text = response['generated_text']
    pass_key = extract_int(generated_text)
    if pass_key is None:
        pass_key = -1
    return pass_key


def estimate_passkey_retrieval_accuracy(pipeline, trials, context_size, scaled_max_position_embeddings):
    prompt_size = min(context_size, scaled_max_position_embeddings-100)
    correct_cnt = 0
    for i in tqdm(range(trials)):
        prompt_text, pass_key = generate_prompt(prompt_size)
        assert f"The pass key is {pass_key}" in prompt_text
        pred = model_inference(pipeline, prompt_text)
        correct_cnt += 1 if pred == pass_key else 0
    accuracy = correct_cnt/trials
    print(f"context_size: {context_size} accuracy: {correct_cnt/trials}")
    return accuracy

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, default=2**18)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--rope_scaling_factor", type=float, default=1.0)
    parser.add_argument("--rope_scaling_type", type=str, default="linear")
    parser.add_argument("--seed", type=int, default=98052)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()

    random.seed(args.seed)

    checkpoint = args.model_path
    config = AutoConfig.from_pretrained(checkpoint, cache_dir=checkpoint, trust_remote_code=True)
    scaled_max_position_embeddings = config.max_position_embeddings
    if not hasattr(config, 'rope_scaling') or config.rope_scaling is None:
        if args.rope_scaling_type is not None:
            config.rope_scaling={"type": args.rope_scaling_type, "factor": args.rope_scaling_factor}
            scaled_max_position_embeddings=int(config.max_position_embeddings * args.rope_scaling_factor)
            config.max_position_embeddings=scaled_max_position_embeddings

    config.use_cache=False
    try:
        model = AutoModelForCausalLM.from_pretrained(checkpoint, cache_dir=checkpoint, config=config, torch_dtype=torch.float16, trust_remote_code=True, attn_implementation="flash_attention_2")
    except Exception as e:
        print(f"Failed to load model from {checkpoint} using flash_attention_2. Trying to load without flash attention.")
        model = AutoModelForCausalLM.from_pretrained(checkpoint, cache_dir=checkpoint, config=config, torch_dtype=torch.float16, trust_remote_code=True)
    model.to(gpu_device)
    # logger = logging.get_logger("transformers")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding_side="right", trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if model.config.pad_token_id is None and model.config.eos_token_id is not None:
        model.config.pad_token_id = model.config.eos_token_id
    if model.generation_config.pad_token_id is None and model.generation_config.eos_token_id is not None:
        model.generation_config.pad_token_id = model.generation_config.eos_token_id

    my_pipeline = hf_pipeline("text-generation", model=model, tokenizer=tokenizer, batch_size=1, device=gpu_device, trust_remote_code=True, return_full_text=False, max_new_tokens=20)
    outputs = my_pipeline("Fun fact: ")
    print(outputs)

    result_list = list()
    length_list = [131072, 262144]  # [2**i for i in range(11,32) if 2**i <= args.max_length]
    for context_size in length_list:
        accuracy = estimate_passkey_retrieval_accuracy(my_pipeline, args.trials, context_size, scaled_max_position_embeddings)
        result_list.append({"context_size": context_size, "accuracy": accuracy})


    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir,"results.jsonl")
    with open(output_file, "w") as out:
        for result in result_list:
            out.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()