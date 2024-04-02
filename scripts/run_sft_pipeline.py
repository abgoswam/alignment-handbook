#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""
Supervised fine-tuning script for decoder language models.
"""

import logging
import math
import os
import random
import sys
import copy
from typing import Dict
import time
import itertools
import datasets
import torch
import transformers
import numpy as np
from torch.utils.data import IterableDataset
from transformers import set_seed, AutoModelForCausalLM


from torch.utils.data import Dataset as TorchDataset

from alignment import (
    DataArguments,
    H4ArgumentParser,
    ModelArguments,
    SFTConfig,
    apply_chat_template,
    get_checkpoint,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
)
from trl import SFTTrainer


logger = logging.getLogger(__name__)

import pkg_resources


################################################## monkey patch ###########################
# Bring in run_sft.py in urgency to fix multi-node run, later to see how to beautify
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
)

from transformers.trainer import (
    Trainer,
    TRAINER_STATE_NAME
)

import shutil

def _save_checkpoint_patch(self, model, trial, metrics=None):
    # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
    # want to save except FullyShardedDDP.
    # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

    # Save model checkpoint
    checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

    if self.hp_search_backend is None and trial is None:
        self.store_flos()

    run_dir = self._get_output_dir(trial=trial)
    output_dir = os.path.join(run_dir, checkpoint_folder)

    #############  This is the only change applied by this monkey patch ##############
    staging_output_dir = output_dir

    # if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
    #     logger.warning(
    #         f"Checkpoint destination directory {output_dir} already exists and is non-empty. "
    #         "Saving will proceed but saved results may be invalid."
    #     )
    #     staging_output_dir = output_dir
    # else:
    #     staging_output_dir = os.path.join(run_dir, f"tmp-{checkpoint_folder}")
    ####################################### END ####################################

    self.save_model(staging_output_dir, _internal_call=True)

    if not self.args.save_only_model:
        # Save optimizer and scheduler
        self._save_optimizer_and_scheduler(staging_output_dir)
        # Save RNG state
        self._save_rng_state(staging_output_dir)

    # Determine the new best metric / best model checkpoint
    if metrics is not None and self.args.metric_for_best_model is not None:
        metric_to_check = self.args.metric_for_best_model
        if not metric_to_check.startswith("eval_"):
            metric_to_check = f"eval_{metric_to_check}"
        metric_value = metrics[metric_to_check]

        operator = np.greater if self.args.greater_is_better else np.less
        if (
            self.state.best_metric is None
            or self.state.best_model_checkpoint is None
            or operator(metric_value, self.state.best_metric)
        ):
            self.state.best_metric = metric_value
            self.state.best_model_checkpoint = output_dir

    # Save the Trainer state
    if self.args.should_save:
        self.state.save_to_json(os.path.join(staging_output_dir, TRAINER_STATE_NAME))

    if self.args.push_to_hub:
        self._push_from_checkpoint(staging_output_dir)

    # Place checkpoint in final location after all saving is finished.
    # First wait for everyone to finish writing
    self.args.distributed_state.wait_for_everyone()

    # Then go through the rewriting process, only renaming and rotating from main process(es)
    if self.is_local_process_zero() if self.args.save_on_each_node else self.is_world_process_zero():
        if staging_output_dir != output_dir:
            if os.path.exists(staging_output_dir):
                os.rename(staging_output_dir, output_dir)

                # Ensure rename completed in cases where os.rename is not atomic
                # And can only happen on non-windows based systems
                if os.name != "nt":
                    fd = os.open(output_dir, os.O_RDONLY)
                    os.fsync(fd)
                    os.close(fd)

        # Maybe delete some older checkpoints.
        if self.args.should_save:
            # Solely rely on numerical checkpoint id for rotation.
            # mtime is not reliable especially on some fuse fs in cloud environments.
            self._rotate_checkpoints(use_mtime=False, output_dir=run_dir)
    elif self.is_local_process_zero():
        # Clean up the remaining staging checkpoint folders on other nodes
        if staging_output_dir != output_dir and os.path.exists(staging_output_dir):
            shutil.rmtree(staging_output_dir)

    self.args.distributed_state.wait_for_everyone()

Trainer._save_checkpoint = _save_checkpoint_patch
########################################### END of monkey patch #############################################################


def list_installed_packages():
    for distribution in pkg_resources.working_set:
        print(f"Package: {distribution.project_name}")
        print(f"Version: {distribution.version}")
        print(f"Location: {distribution.location}")
        print("------------------------")

def mask_(labels, tokenizer, data_args, is_eval):
    only_keep_last_assistant_msg = is_eval and data_args.mask_user_and_system_tokens_eval_strategy == "only_keep_last_assistant_msg"

    user_start = tokenizer.convert_tokens_to_ids([data_args.user_start_token])[0]
    user_end = tokenizer.convert_tokens_to_ids([data_args.user_end_token])[0]
    sys_start = tokenizer.convert_tokens_to_ids([data_args.system_start_token])[0]
    sys_end = tokenizer.convert_tokens_to_ids([data_args.system_end_token])[0]
    assistant_start = tokenizer.convert_tokens_to_ids([data_args.assistant_start_token])[0]
    labels = labels.tolist()[0]
    masked_labels = np.asarray(labels)

    start_tokens = {user_start, sys_start}
    end_tokens = {user_end, sys_end}
    s_idx = None
    e_idx = None
    for i, tok in enumerate(labels):
        if tok in start_tokens:
            s_idx = i
        if tok in end_tokens:
            if s_idx is not None:
                masked_labels[s_idx:i+1]= -100
                s_idx = None
            e_idx = i
        if only_keep_last_assistant_msg and tok == assistant_start:
            masked_labels[:i+1] = -100
    
    if only_keep_last_assistant_msg and e_idx is not None:
        masked_labels[e_idx:] = -100

    # Ignore pad tokens in the loss computation
    if tokenizer.pad_token_id != tokenizer.eos_token_id:
        # replace tokenizer.pad_token_id with -100
        masked_labels[masked_labels == tokenizer.pad_token_id] = -100
    else:
        # there are several tokenizer.pad_token_id at the end of masked_labels, replace them with -100 except the first one
        for i in range(len(masked_labels) - 1, 0, -1):
            if masked_labels[i] == tokenizer.pad_token_id and masked_labels[i - 1] == tokenizer.pad_token_id:
                masked_labels[i] = -100
            else:
                break
        # if padding_side = "left"
        for i in range(len(masked_labels)):
            if masked_labels[i] == tokenizer.pad_token_id:
                masked_labels[i] = -100
            else:
                break

    return torch.asarray([masked_labels])

def preprocess(
        lines,
        tokenizer,
        data_args,
        is_eval,
) -> Dict:
    prompts = [line + tokenizer.eos_token for line in lines]
    input_ids = tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length" if (is_eval or not data_args.packing) else "do_not_pad",
        max_length=data_args.tokenizer_lazy_preprocess_max_length,
        truncation=(is_eval or not data_args.packing),
    ).input_ids

    label_ids = input_ids.clone()
    if data_args.mask_user_and_system_tokens or (is_eval and data_args.mask_user_and_system_tokens_eval_strategy is not None):
        label_ids = mask_(label_ids, tokenizer=tokenizer, data_args=data_args, is_eval=is_eval)

    return dict(
        input_ids=input_ids,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
        labels=label_ids,
    )

class LazySupervisedDataset(TorchDataset):
    """lazy loading dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer, data_args, is_eval=False):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.data_args = data_args
        self.is_eval = is_eval

        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        example = self.raw_data[i]
        apply_chat_template(example=example, tokenizer=self.tokenizer, task='sft')
        ret = preprocess([example['text']], self.tokenizer, self.data_args, is_eval=self.is_eval)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret


class ConstantLengthDatasetWithPassinLabel(IterableDataset):

    def __init__(
        self,
        dataset,
        seq_length=1024,
        num_of_sequences=1,
        shuffle=True,
    ):
        self.dataset = dataset
        self.seq_length = seq_length
        self.current_size = 0
        self.max_buffer_size = seq_length * num_of_sequences
        self.shuffle = shuffle

        n_samples = 500

        if len(self.dataset) > n_samples:
            samples = [self.dataset[i] for i in random.sample(range(len(self.dataset)), n_samples)]
        else:
            samples = self.dataset
            n_samples = len(self.dataset)

        self.estimated_length = float(len(self.dataset)) * (sum([len(x["input_ids"]) for x in samples]) / n_samples) / seq_length
        self.estimated_length = int(math.ceil(self.estimated_length * 1.1))
        

    # This is not accurate. See the comments in training loop.
    def __len__(self):
        return self.estimated_length

    def __iter__(self):
        def get_iterator():
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is None:
                iterator = iter(range(len(self.dataset)))
            else:
                worker_id = worker_info.id
                worker_total_num = worker_info.num_workers
                iterator = itertools.islice(iter(range(len(self.dataset))), worker_id, None, worker_total_num)
            return iterator
        iterator = get_iterator()
        more_examples = True
        all_token_ids = []
        all_label_ids = []
        while more_examples:
            while True:
                if len(all_token_ids) >= self.max_buffer_size:
                    break
                try:
                    x = self.dataset[next(iterator)]
                    all_token_ids.extend(x["input_ids"])
                    all_label_ids.extend(x["labels"])
                except StopIteration:
                    iterator = get_iterator()
                    logger.warn("The dataset reached end and the iterator is reset to the start.")
            examples = []
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                labels = all_label_ids [i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    examples.append((input_ids, labels))
                    all_token_ids = all_token_ids[i + self.seq_length:]
                    all_label_ids = all_label_ids[i + self.seq_length:]
            if self.shuffle:
                random.shuffle(examples)
            for example in examples:
                self.current_size += 1
                yield {
                    "input_ids": torch.LongTensor(example[0]),
                    "labels": torch.LongTensor(example[1]),
                }


VALID_CONFIG_TABLE = """
#######################################################################
# | LAZY_PREPROCESS | PACKING | MASK_USER_AND_SYSTEM_TOKENS | Support |
# |-----------------|---------|-----------------------------|---------|
# | False           | False   | False                       | NO      |
# | False           | False   | True                        | NO      |
# | False           | True    | False                       | YES     |
# | False           | True    | True                        | NO      |
# | True            | False   | False                       | YES     |
# | True            | False   | True                        | YES     |
# | True            | True    | False                       | YES     |
# | True            | True    | True                        | YES     |
#######################################################################
"""


def main():
    logger.info(sys.argv)
    
    print("Listing installed packages:")
    list_installed_packages()

    parser = H4ArgumentParser((ModelArguments, DataArguments, SFTConfig))
    model_args, data_args, training_args = parser.parse()

    invalid_message = f"{VALID_CONFIG_TABLE}\n\nThe combination of lazy_preprocess={data_args.lazy_preprocess}, packing={data_args.packing}, and mask_user_and_system_tokens={data_args.mask_user_and_system_tokens} is not supported. Please refer to the table above for valid configurations."
    assert data_args.lazy_preprocess or data_args.packing, invalid_message
    assert data_args.lazy_preprocess or not data_args.mask_user_and_system_tokens, invalid_message

    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    #######################
    # Load pretrained model
    #   Multi-node job is easy to stuck at this step.
    #   So we put this step as early as possible to avoid wasting resources.
    #######################
    logger.info("*** Load pretrained model ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        use_flash_attention_2=model_args.use_flash_attention_2,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    # hack to get past permission issues in AML when multiple processes attempt copying HF files 
    # https://github.com/microsoft/genai/issues/469
    time.sleep(random.uniform(1, 10))  
    for i in range(32,0,-1):
        try:
            model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
            break
        except PermissionError as e:
            if i > 1:
                print(f"Model encounters PermissionError: {e}")
                time.sleep(10)
                continue
            else:
                raise e
    logger.info("*** Model loaded! ***")

    ###############
    # Load datasets
    ###############
    raw_datasets = get_datasets(data_args, splits=data_args.dataset_splits)
    logger.info(
        f"Training on the following datasets and their proportions: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["train"].features)

    ################
    # Load tokenizer
    ################
    # hack to get past permission issues in AML when multiple processes attempt copying HF files 
    # https://github.com/microsoft/genai/issues/469
    for i in range(32,0,-1):
        try:
            tokenizer = get_tokenizer(model_args, data_args)
            break
        except PermissionError as e:
            if i > 1:
                print(f"Tokenizer encounters PermissionError: {e}")
                time.sleep(10)
                continue
            else:
                raise e

    #####################
    # Apply chat template
    #####################
    num_added_toks = 0
    if data_args.mask_user_and_system_tokens:
        assert data_args.system_start_token is not None, "system_start_token must be specificed if mask_user_and_system_tokens=True."
        assert data_args.system_end_token is not None, "system_end_token must be specificed if mask_user_and_system_tokens=True."
        assert data_args.user_start_token is not None, "user_start_token must be specificed if mask_user_and_system_tokens=True."
        assert data_args.user_end_token is not None, "user_end_token must be specificed if mask_user_and_system_tokens=True."


    if not data_args.lazy_preprocess:
        raw_datasets = raw_datasets.map(
            apply_chat_template,
            fn_kwargs={"tokenizer": tokenizer, "task": "sft"},
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            desc="Applying chat template",
        )

        with training_args.main_process_first(desc="Log a few random samples from the processed training set"):
            for index in random.sample(range(len(raw_datasets["train"])), 3):
                logger.info(f"Sample {index} of the processed training set:\n\n{raw_datasets['train'][index]['text']}")

        train_dataset = raw_datasets["train"]
        eval_dataset = raw_datasets["test"]
    else:
        train_dataset = LazySupervisedDataset(raw_datasets["train"], tokenizer, data_args)
        if data_args.packing:
            train_dataset = ConstantLengthDatasetWithPassinLabel(
                dataset=train_dataset,
                seq_length=training_args.max_seq_length,
            )
        # zelin's comment:
        # We don't pack eval dataset. Insteadly, we set padding=True and truncation=True.
        eval_dataset = LazySupervisedDataset(raw_datasets["test"], tokenizer, data_args, is_eval=True)

    def compute_metrics(p):  
        output, labels = p  
        flat_output = output.flatten()  
        non_neg100_mask = flat_output != -100  
        values_for_mean = flat_output[non_neg100_mask].astype(float)  
        if values_for_mean.size > 0:  
            global_mean_value = np.mean(values_for_mean)  
        else:  
            global_mean_value = np.nan  
        return {  
            "token_accuracy": global_mean_value  
        }  
    
    def preprocess_logits_for_metrics(logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Debug code for MMLU_TEST
        # x = shift_logits.argmax(-1)
        # x = torch.where(shift_labels == -100, shift_labels, x).flatten()
        # valid_tokens = [315, 319, 350, 360, -100]
        # invalid_tokens = set([i for i in x if i not in valid_tokens])
        # logger.warning(f"Invalid tokens: {invalid_tokens}")

        return torch.where(
            shift_labels == -100,
            torch.full_like(shift_labels, -100),
            (shift_logits.argmax(-1) == shift_labels).long()
        )


    ########################
    # Initialize the Trainer
    ########################
    # We should always set packing=True, even if data_args.packing is False.
    # This is because SFTTrainer will use DataCollatorForLanguageModeling when packing=False, which overwrites labels in the dataset.
    # The overwritten labels has a bug when padding token is equal to eos token, making eos token mistakenly masked in the loss computation.
    ########################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=training_args.max_seq_length,
        tokenizer=tokenizer,
        packing=True,  # IMPORTANT: DON'T CHANGE THIS!!!
        peft_config=get_peft_config(model_args),
        compute_metrics=compute_metrics if data_args.mask_user_and_system_tokens_eval_strategy else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if data_args.mask_user_and_system_tokens_eval_strategy else None,
    )

    ###############
    # Training loop
    ###############
    # Note that, if data_args.packing is True, the epoch metric is an approximate number.
    # Please refer to ConstantLengthDatasetWithPassinLabel.__len__ for details.
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    ##################################
    # Save model and create model card
    ##################################
    if data_args.save_tokenizer_model_max_length:
        tokenizer.model_max_length = data_args.save_tokenizer_model_max_length

    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    tokenizer_model_src, tokenizer_model_dst= os.path.join(model_args.model_name_or_path, "tokenizer.model"), os.path.join(training_args.output_dir, "tokenizer.model")
    if os.path.exists(tokenizer_model_src) and not os.path.exists(tokenizer_model_dst):
        shutil.copyfile(src=tokenizer_model_src, dst=tokenizer_model_dst)

    # Save everything else on main process
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "dataset": list(data_args.dataset_mixer.keys()),
        "dataset_tags": list(data_args.dataset_mixer.keys()),
        "tags": ["alignment-handbook"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    if training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)

    logger.info("*** Training complete ***")


if __name__ == "__main__":
    main()