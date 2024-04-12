# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import json
import logging
import math
import os
import random
from typing import Dict, List, Literal, Optional
import warnings

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from datasets.builder import DatasetGenerationError

from .configs_pipeline import DataArguments


DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"

logger = logging.getLogger()

def apply_chat_template(
    example,
    tokenizer,
    task: Literal["sft", "generation", "rm", "dpo"],
):
    if task in ["sft", "generation"]:
        messages = example["messages"]
        # We add an empty system message if there is none
        if messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": ""})
        example["text"] = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True if task == "generation" else False
        )
    elif task == "rm":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            chosen_messages = example["chosen"]
            rejected_messages = example["rejected"]
            # We add an empty system message if there is none
            if chosen_messages[0]["role"] != "system":
                chosen_messages.insert(0, {"role": "system", "content": ""})
            if rejected_messages[0]["role"] != "system":
                rejected_messages.insert(0, {"role": "system", "content": ""})
            example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        else:
            raise ValueError(
                f"Could not format example as dialogue for `rm` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
            )
    elif task == "dpo":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            # For DPO, the inputs are triples of (prompt, chosen, rejected), where `chosen` and `rejected` are the final turn of a dialogue
            # We therefore need to extract the N-1 turns to form the prompt
            prompt_messages = example["chosen"][:-1]
            # Prepend a system message if the first message is not a system message
            if example["chosen"][0]["role"] != "system":
                prompt_messages.insert(0, {"role": "system", "content": ""})
            # Now we extract the final turn to define chosen/rejected responses
            chosen_messages = example["chosen"][-1:]
            rejected_messages = example["rejected"][-1:]
            example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
            example["text_prompt"] = tokenizer.apply_chat_template(prompt_messages, tokenize=False)
        else:
            raise ValueError(
                f"Could not format example as dialogue for `dpo` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
            )
    else:
        raise ValueError(
            f"Task {task} not supported, please ensure that the provided task is one of {['sft', 'generation', 'rm', 'dpo']}"
        )
    return example

def get_datasets(
    data_config: DataArguments | dict,
    splits: List[str] = ["train", "test"],
    shuffle: bool = True
) -> DatasetDict:
    """
    Loads one or more datasets with varying training set proportions.

    Args:
        data_config (`DataArguments` or `dict`):
            Dataset configuration and split proportions.
        splits (`List[str]`, *optional*, defaults to `['train', 'test']`):
            Dataset splits to load and mix. Assumes the splits exist in all datasets and have a `train_` or `test_` prefix.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the training and testing/validation data.

    Returns
        [`DatasetDict`]: The dataset dictionary containing the loaded datasets.
    """

    local_datasets_dir = None
    if type(data_config) is DataArguments:
        # Structure of the config to read the datasets and their mix
        # datasets_mixer:
        #     - 'dataset1': 0.5
        #     - 'dataset2': 0.3
        #     - 'dataset3': 0.2
        dataset_mixer = data_config.dataset_mixer
        local_datasets_dir = data_config.datasets_dir
    elif isinstance(data_config, dict):
        # Structure of the input is:
        #     dataset_mixer = {
        #             "dataset1": 0.5,
        #             "dataset1": 0.3,
        #             "dataset1": 0.2,
        #         }
        dataset_mixer = data_config
        warnings.warn("data_config as dict type, datasets_dir is not supported and hence ignored!")
    else:
        raise ValueError(f"Data config {data_config} not recognized.")

    if local_datasets_dir is not None:
        raw_datasets = mix_datasets_local(
            dataset_mixer,
            splits=splits,
            shuffle=shuffle,
            local_datasets_dir=local_datasets_dir,
            example_concate_rate=data_config.example_concate_rate)
    else:
        raw_datasets = mix_datasets(
            dataset_mixer,
            splits=splits,
            shuffle=shuffle,
            example_concate_rate=data_config.example_concate_rate)

    return raw_datasets


def _load_data_list(jsonl_file):
    with open(jsonl_file, 'r', encoding='utf-8') as fi:
        return [json.loads(line) for line in fi]
    

def _merge_dicts_align_schema(dict_lists: List[List[Dict]]) -> List[Dict]:
    result = []
    features = set([key for dict_list in dict_lists for key in dict_list[0].keys()])
    init_f = {k: None for k in features}
    for dict_list in dict_lists:
        for example in dict_list:
            f = copy.deepcopy(init_f)
            f.update(example)
            result.append(f)

    return result


def _concat_examples(ds_name: str, dataset: Dataset, example_concate_rate: float, shuffle: bool=True) -> Dataset:
    if shuffle:
        random.shuffle(dataset)
    new_examples = []
    for example in dataset:
        if random.random() < example_concate_rate and len(new_examples) > 0:
            new_examples[-1]["messages"] += example["messages"]
        else:
            new_examples.append(example)
    logger.info(f"{ds_name}: concated {len(new_examples)} examples from {len(dataset)} examples")
    dataset = Dataset.from_list(new_examples)
    return dataset


def mix_datasets_local(dataset_mixer: dict, splits: Optional[List[str]] = None, shuffle=True, local_datasets_dir: str = None, example_concate_rate: float = None) -> DatasetDict:
    """
    Loads and mixes datasets according to proportions specified in `dataset_mixer`.

    Args:
        dataset_mixer (`dict`):
            Dictionary containing the dataset names and their training proportions. By default, all test proportions are 1.
        splits (Optional[List[str]], *optional*, defaults to `None`):
            Dataset splits to load and mix. Assumes the splits exist in all datasets and have a `train_` or `test_` prefix.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the training and testing/validation data.
    """
    raw_train_datasets = []
    raw_val_datasets = []
    random.seed(42)
    for ds, frac in dataset_mixer.items():
        if frac < 0:
            raise ValueError("Dataset fractions cannot be negative.")
        
        for split in splits:
            # local dataset dir structure:
            # .
            #   - <dataset name as folder name>
            #     - train.jsonl
            #     - test.jsonl
            # dataset = load_dataset('json', data_files=os.path.join(local_datasets_dir, f"{ds}/{split}.jsonl"))['train']
            if ds.endswith('.jsonl'):
                ds_path = os.path.join(local_datasets_dir, ds)
                if not ds.endswith(f"/{split}.jsonl"):
                    continue
            else:
                ds_path = os.path.join(local_datasets_dir, f"{ds}/{split}.jsonl")
            dataset = _load_data_list(ds_path)
            logger.info(f"{len(dataset)} examples loaded from {ds_path}")
            if 'train' in split and frac < 1 and shuffle:
                random.shuffle(dataset)

            if 'train' in split:
                if frac < 1: # no sampling to test set
                    dataset = dataset[0: int(len(dataset) * frac)]
                    logger.info(f"Dataset {ds_path} is down sampled to {len(dataset)}")
            
                elif frac > 1:
                    dataset = (dataset * math.ceil(frac))[0: int(len(dataset) * frac)]
                    logger.info(f"Dataset {ds_path} is up sampled to {len(dataset)}")
            
            if "train" in split:
                if example_concate_rate is not None:
                    dataset = _concat_examples(ds_path, dataset, example_concate_rate, shuffle=shuffle)
                raw_train_datasets.append(dataset)
            elif "test" in split:
                raw_val_datasets.append(dataset)
            else:
                raise ValueError(f"Split type {split} not recognized as one of test or train.")

    if len(raw_train_datasets) == 0 and len(raw_val_datasets) == 0:
        raise ValueError(
            f"Dataset {dataset_mixer} not recognized and no data loaded. Check the dataset has been correctly formatted."
        )
    
    if len(raw_train_datasets) > 0:
        raw_train_datasets = _merge_dicts_align_schema(raw_train_datasets)

    if len(raw_val_datasets) > 0:
        raw_val_datasets = _merge_dicts_align_schema(raw_val_datasets)

    logger.info(f"All raw_train_datasets size: {len(raw_train_datasets)} examples")
    logger.info(f"All raw_val_datasets size: {len(raw_val_datasets)} examples")

    if shuffle:
        random.shuffle(raw_train_datasets)
        random.shuffle(raw_val_datasets)

    raw_datasets = DatasetDict()
    logger.info(f"Generating hf dataset 'train' split ...")
    raw_datasets['train'] = Dataset.from_list(raw_train_datasets)
    logger.info(f"Generating hf dataset 'test' split ...")
    raw_datasets['test'] = Dataset.from_list(raw_val_datasets)
    
    return raw_datasets


def mix_datasets(dataset_mixer: dict, splits: Optional[List[str]] = None, shuffle=True, example_concate_rate=None) -> DatasetDict:
    """
    Loads and mixes datasets according to proportions specified in `dataset_mixer`.

    Args:
        dataset_mixer (`dict`):
            Dictionary containing the dataset names and their training proportions. By default, all test proportions are 1.
        splits (Optional[List[str]], *optional*, defaults to `None`):
            Dataset splits to load and mix. Assumes the splits exist in all datasets and have a `train_` or `test_` prefix.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the training and testing/validation data.
    """
    raw_datasets = DatasetDict()
    raw_train_datasets = []
    raw_val_datasets = []
    fracs = []
    for ds, frac in dataset_mixer.items():
        fracs.append(frac)
        for split in splits:
            try:
                # Try first if dataset on a Hub repo
                dataset = load_dataset(ds, split=split)
            except DatasetGenerationError:
                # If not, check local dataset
                dataset = load_from_disk(os.path.join(ds, split))

            if "train" in split:
                if example_concate_rate is not None:
                    dataset = _concat_examples(ds, dataset, example_concate_rate, shuffle=shuffle)
                raw_train_datasets.append(dataset)
            elif "test" in split:
                raw_val_datasets.append(dataset)
            else:
                raise ValueError(f"Split type {split} not recognized as one of test or train.")

    if any(frac < 0 for frac in fracs):
        raise ValueError("Dataset fractions cannot be negative.")

    if len(raw_train_datasets) > 0:
        train_subsets = []
        for dataset, frac in zip(raw_train_datasets, fracs):
            train_subset = dataset.select(range(int(frac * len(dataset))))
            train_subsets.append(train_subset)
        if shuffle:
            raw_datasets["train"] = concatenate_datasets(train_subsets).shuffle(seed=42)
        else:
            raw_datasets["train"] = concatenate_datasets(train_subsets)
    # No subsampling for test datasets to enable fair comparison across models
    if len(raw_val_datasets) > 0:
        if shuffle:
            raw_datasets["test"] = concatenate_datasets(raw_val_datasets).shuffle(seed=42)
        else:
            raw_datasets["test"] = concatenate_datasets(raw_val_datasets)

    if len(raw_datasets) == 0:
        raise ValueError(
            f"Dataset {dataset_mixer} not recognized with split {split}. Check the dataset has been correctly formatted."
        )

    return raw_datasets