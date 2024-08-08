import argparse
import os
from typing import Dict, Tuple
import json

eval_args = argparse.ArgumentParser()
eval_args.add_argument("--model_path", type=str, required=True)
eval_args.add_argument("--task", type=str, required=True)
eval_args.add_argument("--n_shot", type=int, required=True)
eval_args.add_argument("--seed", type=int, required=True)
eval_args.add_argument("--device", type=int, required=False, default=0)
eval_args.add_argument("--batch_size", type=int, required=False, default=4)
eval_args.add_argument("--flash_attn_2",  action='store_true')
eval_args.add_argument("--max_length", type=int, default=8192)
eval_args.add_argument("--downsample", action="store_true")
eval_args.add_argument("--print", action="store_true")
eval_args.add_argument("--num_retrieved_docs", type=int, default=0)
eval_args.add_argument("--flipped_ratio", type=float, default=0)
eval_args = eval_args.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = f"{eval_args.device}"

from eval_utils import (
    eval_generation_em,
    eval_generation_em_answers,
    exact_match_score,
    exact_match_score_with_multiple_candidates
)

import torch
import json
import numpy as np

from tqdm import tqdm
from transformers.models.llama import LlamaTokenizer
from transformers import LlamaForCausalLM
from torch.utils.data import DataLoader, Dataset
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s %(name)s %(lineno)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

TASK_DATA_PATH = {
    "nq": {
        "train": "./eval_data/NQ-open.train-train.jsonl",
        "test": "./eval_data/NQ-open.test.jsonl",
    },
    "tq": {
        "train": "./eval_data/triviaqa.train-train.jsonl",
        "test": "./eval_data/triviaqa.test.jsonl",
    },
    "agnews": {
        "train": "./eval_data/ag_news_train.jsonl",
        "test": "./eval_data/ag_news_test.jsonl",
    },
    "agnews2": {
        "train": "./eval_data/ag_news_train.jsonl",
        "test": "./eval_data/ag_news_test.jsonl",
    },
    "nq_obqa": {
        "validation": "./eval_data/nq-dev-dense-results.json",
        "test": "./eval_data/nq_test.json",
    },
    "tq_obqa": {
        "train": "./eval_data/tq_obqa_train.json",
        "test": "./eval_data/tq_obqa_test.json",
    },
    "hotpotqa": {
        "train": "./eval_data/hotpot_train_v1.1.json",
        "validation": "./eval_data/hotpot_dev_fullwiki_v1.json",
    },
    "amazon": {
        "train": "./eval_data/amazon_train.jsonl",
        "test": "./eval_data/amazon_test.jsonl"
    },
    "amazon2": {
        "train": "./eval_data/amazon_train.jsonl",
        "test": "./eval_data/amazon_test.jsonl"
    },
    "amazon3": {
        "train": "./eval_data/amazon_train.jsonl",
        "test": "./eval_data/amazon_test.jsonl"
    },
    "dbpedia": {
        "train": "./eval_data/dbpedia_train.jsonl",
        "test": "./eval_data/dbpedia_test.jsonl"
    },
    "yelp": {
        "train": "./eval_data/yelp_train.jsonl",
        "test": "./eval_data/yelp_test.jsonl"
    },
    "sst2": {
        "train": "./eval_data/sst2_train.jsonl",
        "test": "./eval_data/sst2_test.jsonl",
    },
    "sst3": {
        "train": "./eval_data/sst2_train.jsonl",
        "test": "./eval_data/sst2_test.jsonl",
    },
    "tweet_hate": {
        "train": "./eval_data/tweet_hate_train.jsonl",
        "test": "./eval_data/tweet_hate_test.jsonl"
    },
    "tweet_offensive": {
        "train": "./eval_data/tweet_offensive_train.jsonl",
        "test": "./eval_data/tweet_offensive_test.jsonl"
    },
    "squad": {
        "train": "./eval_data/squad_train.jsonl",
        "validation": "./eval_data/squad_validation.jsonl"
    },
    "memtrap": {
        "test": "./eval_data/memtrap_proverb_ending_test.jsonl"
    },
}
for task in TASK_DATA_PATH:
    for split in TASK_DATA_PATH[task]:
        if not os.path.exists(TASK_DATA_PATH[task][split]):
            raise FileNotFoundError(f"File {TASK_DATA_PATH[task][split]} not found")

PRINT = eval_args.print
DOWNSAMPLE = eval_args.downsample

if not os.path.exists("./outputs/logs"):
    with open("./outputs/logs", "w") as fn:
        pass


def load_json(path):
    return json.load(open(path, "r"))


def load_jsonl(path, max_line=None):
    with open(path, "r", encoding="utf-8") as fn:
        data = [json.loads(line) for line in fn.readlines()]
        if max_line is not None:
            rng = np.random.RandomState(666)
            data = rng.choice(data, min(max_line, len(data)), replace=False)
    return data


def normalise_pred(pred):
    return pred.strip().split("\n")[0].strip()

def map_pred_to_hate_or_non_hate(pred):
    if pred.startswith("Hate"):
        return "Hate"
    elif pred.startswith("Non-hate"):
        return "Non-hate"
    else:
        return pred

class PromptDataset(Dataset):
    def __init__(self, prompt_list, tokenizer):
        self.data = prompt_list
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def get_dataloader(self, batch_size, max_length):
        def collate_fn(items):
            batch = [item["prompt"] for item in items]
            return self.tokenizer(batch, padding=True, return_tensors="pt", truncation=True, max_length=max_length, add_special_tokens=False)

        return DataLoader(self, batch_size, shuffle=False, collate_fn=collate_fn, num_workers=8)


@torch.no_grad()
def generate(model: LlamaForCausalLM, tokenizer: LlamaTokenizer, prompt_list, generation_kwargs,
             max_length, batch_size, task, n_shot, seed):
    predictions = []
    bar = tqdm(total=len(prompt_list), desc=f"{task}-{n_shot}-{seed}")
    prompt_dataset = PromptDataset(prompt_list, tokenizer)
    dataloader = prompt_dataset.get_dataloader(batch_size, max_length)
    for batch in dataloader:
        model_inputs = batch.to("cuda")
        generate_ids = model.generate(**model_inputs, **generation_kwargs)
        pred_ids = generate_ids[:, model_inputs["input_ids"].shape[1]:]
        pred = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        predictions.extend(pred)
        bar.update(len(batch["input_ids"]))

        if PRINT:
            for cur_pred, cur_input in zip(pred, batch):
                print(cur_input, cur_pred)

    assert len(predictions) == len(prompt_list)
    return predictions


def get_cbqa_prompt(input_example, demonstrations):
    prompt = ""
    for item in demonstrations:
        prompt = prompt + f"Question: {item['question']} Answer: {item['answer'][0]}\n"
    prompt = prompt + f"Question: {input_example['question']} Answer:"
    return prompt


def cbqa_evaluation(model, tokenizer, generation_kwargs, task, n_shot, seed, max_length, batch_size):
    generation_kwargs["max_new_tokens"] = 16
    train_data = load_jsonl(TASK_DATA_PATH[task]["train"], max_line=4096)
    rng = np.random.RandomState(seed)
    rng.shuffle(train_data)
    demonstrations = train_data[:n_shot]
    test_data = load_jsonl(TASK_DATA_PATH[task]["test"])
    prompt_list = []
    for item in test_data:
        prompt_list.append({"prompt": get_cbqa_prompt(item, demonstrations)})
    all_pred_ans = generate(model, tokenizer, prompt_list, generation_kwargs, max_length, batch_size, task, n_shot,
                            seed)
    all_pred_ans = [pred.split("\n")[0] for pred in all_pred_ans]
    em_score = eval_generation_em(test_data, all_pred_ans) * 100
    prompts_and_preds = {"prompts": prompt_list, "preds": all_pred_ans}
    return {"score": em_score}, prompts_and_preds

def memtrap_evaluation(model, tokenizer, generation_kwargs, task, n_shot, seed, max_length, batch_size):
    generation_kwargs["max_new_tokens"] = 16
    assert n_shot == 0, "n_shot must be 0 for memtrap, as it is a zero-shot task"
    test_data = load_jsonl(TASK_DATA_PATH[task]["test"])
    prompt_list = []
    for item in test_data:
        prompt_list.append({"prompt": item["prompt"]})
    all_pred_ans = generate(model, tokenizer, prompt_list, generation_kwargs, max_length, batch_size, task, n_shot,
                            seed)
    all_pred_ans = [pred.split("\n")[0] for pred in all_pred_ans]
    em_score = eval_generation_em(test_data, all_pred_ans) * 100
    prompts_and_preds = {"prompts": prompt_list, "preds": all_pred_ans}
    return {"score": em_score}, prompts_and_preds


def get_sampled_demonstrations(train_data, n_shot, seed):
    rng = np.random.RandomState(seed)
    rng.shuffle(train_data)
    return train_data[:n_shot]


def get_agnews_prompt(input_example, demonstrations, label2str):
    prompt = ""
    for item in demonstrations:
        prompt = prompt + f"Article: {item['text'].strip()} Category: {label2str[item['label']]}\n\n"
    prompt = prompt + f"Article: {input_example['text'].strip()} Category:"
    return prompt

def get_agnewsnoise_prompt(input_example, demonstrations, label2str):
    prompt = ""
    for item in demonstrations:
        prompt = prompt + f"Article: {item['text'].strip()} Label: {label2str[item['label']]}\n\n"
    prompt = prompt + f"Article: {input_example['text'].strip()} Label:"
    return prompt

def agnews2_evaluation(model, tokenizer, generation_kwargs, task, n_shot, seed, max_length, batch_size, flipped_ratio=0)-> Tuple[Dict, Dict]:
    generation_kwargs["max_new_tokens"] = 4
    train_data = load_jsonl(TASK_DATA_PATH[task]["train"])
    test_data = load_jsonl(TASK_DATA_PATH[task]["test"])
    # label2str = {0: "world", 1: "sports", 2: "business", 3: "science"}
    label2str = {0: "A", 1: "B", 2: "C", 3: "D"}
    get_label_call = lambda x: x["label"]
    demonstrations = get_sampled_demonstrations(train_data, n_shot, seed)
    if flipped_ratio > 0:
        demonstrations = choose_wrong_label(demonstrations, seed, flipped_ratio, label2str)
    prompt_list = []
    for idx in range(len(test_data)):
        prompt_list.append({"prompt": get_agnewsnoise_prompt(test_data[idx], demonstrations, label2str)})
    predictions = generate(model, tokenizer, prompt_list, generation_kwargs, max_length, batch_size, task, n_shot, seed)
    score = metric_acc(predictions, test_data, label2str, get_label_call, normalise_pred=normalise_pred)
    return {"score": score}, {"prompts": prompt_list, "preds": predictions}

def agnews_evaluation(model, tokenizer, generation_kwargs, task, n_shot, seed, max_length, batch_size, flipped_ratio=0)-> Tuple[Dict, Dict]:
    generation_kwargs["max_new_tokens"] = 4
    train_data = load_jsonl(TASK_DATA_PATH[task]["train"])
    test_data = load_jsonl(TASK_DATA_PATH[task]["test"])
    label2str = {0: "world", 1: "sports", 2: "business", 3: "science"}
    # label2str = {0: "M", 1: "N", 2: "Q", 3: "P"}
    get_label_call = lambda x: x["label"]
    demonstrations = get_sampled_demonstrations(train_data, n_shot, seed)
    if flipped_ratio > 0:
        demonstrations = choose_wrong_label(demonstrations, seed, flipped_ratio, label2str)
    prompt_list = []
    for idx in range(len(test_data)):
        prompt_list.append({"prompt": get_agnews_prompt(test_data[idx], demonstrations, label2str)})
    predictions = generate(model, tokenizer, prompt_list, generation_kwargs, max_length, batch_size, task, n_shot, seed)
    score = metric_acc(predictions, test_data, label2str, get_label_call, normalise_pred=normalise_pred)
    return {"score": score}, {"prompts": prompt_list, "preds": predictions}

def choose_wrong_label(demonstrations, seed, flipped_ratio, label2str):
    rng = np.random.RandomState(seed)
    flipped_indices = rng.choice(len(demonstrations), int(len(demonstrations) * flipped_ratio), replace=False)
    for idx in flipped_indices:
        demonstrations[idx]["label"] = rng.choice([label for label in label2str if label != demonstrations[idx]["label"]])
    return demonstrations


def get_amazon_prompt(input_example, demonstrations, label2str):
    prompt = ""
    for item in demonstrations:
        prompt = prompt + f"Title: {item['title'].strip()}. Content: {item['content'].strip()} Sentiment: {label2str[item['label']]}\n\n"
    prompt = prompt + f"Title: {input_example['title'].strip()}. Content: {input_example['content'].strip()} Sentiment:"
    return prompt

def get_amazon2_prompt(input_example, demonstrations, label2str):
    prompt = ""
    for item in demonstrations:
        prompt = prompt + f"Title: {item['title'].strip()}. Content: {item['content'].strip()} Label: {label2str[item['label']]}\n\n"
    prompt = prompt + f"Title: {input_example['title'].strip()}. Content: {input_example['content'].strip()} Label:"
    return prompt

def amazon_evaluation(model, tokenizer, generation_kwargs, task, n_shot, seed, max_length, batch_size, flipped_ratio=0) -> Tuple[Dict, Dict]:
    generation_kwargs["max_new_tokens"] = 3
    train_data = load_jsonl(TASK_DATA_PATH["amazon"]["train"])
    test_data = load_jsonl(TASK_DATA_PATH["amazon"]["test"])
    if DOWNSAMPLE:
        sample_test_rng = np.random.RandomState(666)
        test_data = sample_test_rng.choice(test_data, 5000, replace=False)
    label2str = {0: "negative", 1: "positive"}

    # label2str = {0: "foo", 1: "bar"}
    get_label_call = lambda x: x["label"]
    demonstrations = get_sampled_demonstrations(train_data, n_shot, seed)
    if flipped_ratio > 0:
        demonstrations = flip_label(demonstrations, seed, flipped_ratio)
    prompt_list = []
    for idx in range(len(test_data)):
        prompt_list.append({"prompt": get_amazon_prompt(test_data[idx], demonstrations, label2str)})
    predictions = generate(model, tokenizer, prompt_list, generation_kwargs, max_length, batch_size, task, n_shot, seed)
    score = metric_acc(predictions, test_data, label2str, get_label_call, normalise_pred=normalise_pred)
    return {"score": score}, {"prompts": prompt_list, "preds": predictions}

def amazon2_evaluation(model, tokenizer, generation_kwargs, task, n_shot, seed, max_length, batch_size, flipped_ratio=0) -> Tuple[Dict, Dict]:
    generation_kwargs["max_new_tokens"] = 3
    train_data = load_jsonl(TASK_DATA_PATH["amazon"]["train"])
    test_data = load_jsonl(TASK_DATA_PATH["amazon"]["test"])
    if DOWNSAMPLE:
        sample_test_rng = np.random.RandomState(666)
        test_data = sample_test_rng.choice(test_data, 5000, replace=False)
    # label2str = {0: "negative", 1: "positive"}
    label2str = {0: "foo", 1: "bar"}
    get_label_call = lambda x: x["label"]
    demonstrations = get_sampled_demonstrations(train_data, n_shot, seed)
    if flipped_ratio > 0:
        demonstrations = flip_label(demonstrations, seed, flipped_ratio)
    prompt_list = []
    for idx in range(len(test_data)):
        prompt_list.append({"prompt": get_amazon2_prompt(test_data[idx], demonstrations, label2str)})
    predictions = generate(model, tokenizer, prompt_list, generation_kwargs, max_length, batch_size, task, n_shot, seed)
    score = metric_acc(predictions, test_data, label2str, get_label_call, normalise_pred=normalise_pred)
    return {"score": score}, {"prompts": prompt_list, "preds": predictions}

def amazon3_evaluation(model, tokenizer, generation_kwargs, task, n_shot, seed, max_length, batch_size, flipped_ratio=0) -> Tuple[Dict, Dict]:
    generation_kwargs["max_new_tokens"] = 3
    train_data = load_jsonl(TASK_DATA_PATH["amazon"]["train"])
    test_data = load_jsonl(TASK_DATA_PATH["amazon"]["test"])
    if DOWNSAMPLE:
        sample_test_rng = np.random.RandomState(666)
        test_data = sample_test_rng.choice(test_data, 5000, replace=False)
    # label2str = {0: "negative", 1: "positive"}
    label2str = {0: "A", 1: "B"}
    get_label_call = lambda x: x["label"]
    demonstrations = get_sampled_demonstrations(train_data, n_shot, seed)
    if flipped_ratio > 0:
        demonstrations = flip_label(demonstrations, seed, flipped_ratio)
    prompt_list = []
    for idx in range(len(test_data)):
        prompt_list.append({"prompt": get_amazon2_prompt(test_data[idx], demonstrations, label2str)})
    predictions = generate(model, tokenizer, prompt_list, generation_kwargs, max_length, batch_size, task, n_shot, seed)
    score = metric_acc(predictions, test_data, label2str, get_label_call, normalise_pred=normalise_pred)
    return {"score": score}, {"prompts": prompt_list, "preds": predictions}


def get_dbpedia_prompt(input_example, demonstrations, label2str):
    prompt = ""
    for item in demonstrations:
        prompt = prompt + f"Title: {item['title'].strip()}. Content: {item['content'].strip()}\nCategory: {label2str[item['label']]}\n\n"
    prompt = prompt + f"Title: {input_example['title'].strip()}. Content: {input_example['content'].strip()}\nCategory:"
    return prompt


def dbpedia_evaluation(model, tokenizer, generation_kwargs, task, n_shot, seed, max_length, batch_size, flipped_ratio=0) -> Tuple[Dict, Dict]:
    names = ["Company", "EducationalInstitution", "Artist", "Athlete", "OfficeHolder", "MeanOfTransportation",
             "Building", "NaturalPlace", "Village", "Animal", "Plant", "Album", "Film", "WrittenWork", ]
    # names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N"]
    generation_kwargs["max_new_tokens"] = 8
    train_data = load_jsonl(TASK_DATA_PATH["dbpedia"]["train"])
    test_data = load_jsonl(TASK_DATA_PATH["dbpedia"]["test"])
    if DOWNSAMPLE:
        sample_test_rng = np.random.RandomState(666)
        test_data = sample_test_rng.choice(test_data, 10000, replace=False)
    label2str = {idx: name for idx, name in enumerate(names)}
    get_label_call = lambda x: x["label"]
    demonstrations = get_sampled_demonstrations(train_data, n_shot, seed)
    if flipped_ratio > 0:
        demonstrations = choose_wrong_label(demonstrations, seed, flipped_ratio, label2str)
    prompt_list = []
    for idx in range(len(test_data)):
        prompt_list.append({"prompt": get_dbpedia_prompt(test_data[idx], demonstrations, label2str)})
    predictions = generate(model, tokenizer, prompt_list, generation_kwargs, max_length, batch_size, task, n_shot, seed)
    score = metric_acc(predictions, test_data, label2str, get_label_call, normalise_pred=normalise_pred)
    return {"score": score}, {"prompts": prompt_list, "preds": predictions}


def get_yelp_prompt(input_example, demonstrations, label2str):
    prompt = ""
    for item in demonstrations:
        prompt = prompt + f"text: {item['text'].strip()} sentiment: {label2str[item['label']]}\n\n"
    prompt = prompt + f"text: {input_example['text'].strip()} sentiment:"
    return prompt


def yelp_evaluation(model, tokenizer, generation_kwargs, task, n_shot, seed, max_length, batch_size, flipped_ratio=0) -> Tuple[Dict, Dict]:
    generation_kwargs["max_new_tokens"] = 3
    train_data = load_jsonl(TASK_DATA_PATH["yelp"]["train"])
    test_data = load_jsonl(TASK_DATA_PATH["yelp"]["test"])
    if DOWNSAMPLE:
        sample_test_rng = np.random.RandomState(666)
        test_data = sample_test_rng.choice(test_data, 5000, replace=False)
    label2str = {0: "negative", 1: "positive"}
    # label2str = {0: "foo", 1: "bar"}
    get_label_call = lambda x: x["label"]
    demonstrations = get_sampled_demonstrations(train_data, n_shot, seed)
    prompt_list = []
    for idx in range(len(test_data)):
        prompt_list.append({"prompt": get_yelp_prompt(test_data[idx], demonstrations, label2str)})
    predictions = generate(model, tokenizer, prompt_list, generation_kwargs, max_length, batch_size, task, n_shot, seed)
    score = metric_acc(predictions, test_data, label2str, get_label_call, normalise_pred=normalise_pred)
    return {"score": score}, {"prompts": prompt_list, "preds": predictions}


def get_sst2_prompt(input_example, demonstrations, label2str):
    prompt = ""
    for item in demonstrations:
        prompt = prompt + f"text: {item['text'].strip()} sentiment: {label2str[item['label']]}\n\n"
    prompt = prompt + f"text: {input_example['text'].strip()} sentiment:"
    return prompt


def flip_label(demonstrations, seed, flipped_ratio):
    rng = np.random.RandomState(seed)
    flipped_indices = rng.choice(len(demonstrations), int(len(demonstrations) * flipped_ratio), replace=False)
    for idx in flipped_indices:
        demonstrations[idx]["label"] = 1 - demonstrations[idx]["label"]
    return demonstrations

def sst2_evaluation(model, tokenizer, generation_kwargs, task, n_shot, seed, max_length, batch_size, flipped_ratio=0) -> Tuple[Dict, Dict]:
    generation_kwargs["max_new_tokens"] = 3
    train_data = load_jsonl(TASK_DATA_PATH[task]["train"])
    test_data = load_jsonl(TASK_DATA_PATH[task]["test"])
    label2str = {0: "positive", 1: "negative"}
    # label2str = {0: "foo", 1: "bar"}
    get_label_call = lambda x: x["label"]
    demonstrations = get_sampled_demonstrations(train_data, n_shot, seed)
    if flipped_ratio > 0:
        demonstrations = flip_label(demonstrations, seed, flipped_ratio)
    prompt_list = []
    for idx in range(len(test_data)):
        prompt_list.append({"prompt": get_sst2_prompt(test_data[idx], demonstrations, label2str)})
    predictions = generate(model, tokenizer, prompt_list, generation_kwargs, max_length, batch_size, task, n_shot, seed)
    score = metric_acc(predictions, test_data, label2str, get_label_call, normalise_pred=normalise_pred)
    return {"score": score}, {"prompts": prompt_list, "preds": predictions}

def sst3_evaluation(model, tokenizer, generation_kwargs, task, n_shot, seed, max_length, batch_size, flipped_ratio=0) -> Tuple[Dict, Dict]:
    generation_kwargs["max_new_tokens"] = 3
    train_data = load_jsonl(TASK_DATA_PATH[task]["train"])
    test_data = load_jsonl(TASK_DATA_PATH[task]["test"])
    # label2str = {0: "positive", 1: "negative"}
    label2str = {0: "foo", 1: "bar"}
    get_label_call = lambda x: x["label"]
    demonstrations = get_sampled_demonstrations(train_data, n_shot, seed)
    if flipped_ratio > 0:
        demonstrations = flip_label(demonstrations, seed, flipped_ratio)
    prompt_list = []
    for idx in range(len(test_data)):
        prompt_list.append({"prompt": get_sst2_prompt(test_data[idx], demonstrations, label2str)})
    predictions = generate(model, tokenizer, prompt_list, generation_kwargs, max_length, batch_size, task, n_shot, seed)
    score = metric_acc(predictions, test_data, label2str, get_label_call, normalise_pred=normalise_pred)
    return {"score": score}, {"prompts": prompt_list, "preds": predictions}

def get_tweet_hate_prompt(input_example, demonstrations, label2str):
    prompt = ""
    for item in demonstrations:
        prompt = prompt + f"Text: {item['text'].strip()}\nLabel: {label2str[item['label']]}\n"
    prompt = prompt + f"Text: {input_example['text'].strip()}\nLabel:"
    return prompt


def tweet_hate_evaluation(model, tokenizer, generation_kwargs, task, n_shot, seed, max_length, batch_size) -> Tuple[Dict, Dict]:
    generation_kwargs["max_new_tokens"] = 6
    train_data = load_jsonl(TASK_DATA_PATH["tweet_hate"]["train"])
    test_data = load_jsonl(TASK_DATA_PATH["tweet_hate"]["test"])
    label2str = {0: "Non-hate", 1: "Hate"}
    # label2str = {0: "foo", 1: "bar"}
    get_label_call = lambda x: x["label"]
    demonstrations = get_sampled_demonstrations(train_data, n_shot, seed)
    prompt_list = []
    for idx in range(len(test_data)):
        prompt_list.append({"prompt": get_tweet_hate_prompt(test_data[idx], demonstrations, label2str)})
    predictions = generate(model, tokenizer, prompt_list, generation_kwargs, max_length, batch_size, task, n_shot, seed)
    score = metric_acc(predictions, test_data, label2str, get_label_call, normalise_pred=normalise_pred)
    return {"score": score}, {"prompts": prompt_list, "preds": predictions}


def tweet_offensive_evaluation(model, tokenizer, generation_kwargs, task, n_shot, seed, max_length, batch_size) -> Tuple[Dict, Dict]:
    generation_kwargs["max_new_tokens"] = 6
    train_data = load_jsonl(TASK_DATA_PATH["tweet_offensive"]["train"])
    test_data = load_jsonl(TASK_DATA_PATH["tweet_offensive"]["test"])
    label2str = {0: "Non-hate", 1: "Hate"}
    # label2str = {0: "foo", 1: "bar"}
    get_label_call = lambda x: x["label"]
    demonstrations = get_sampled_demonstrations(train_data, n_shot, seed)
    prompt_list = []
    for idx in range(len(test_data)):
        prompt_list.append({"prompt": get_tweet_hate_prompt(test_data[idx], demonstrations, label2str)})
    predictions = generate(model, tokenizer, prompt_list, generation_kwargs, max_length, batch_size, task, n_shot, seed)
    score = metric_acc(predictions, test_data, label2str, get_label_call, normalise_pred=normalise_pred)
    return {"score": score}, {"prompts": prompt_list, "preds": predictions}


def get_obqa_demonstration(train_data, n_shot, seed):
    rng = np.random.RandomState(seed)
    rng.shuffle(train_data)
    demonstrations = train_data[:n_shot]
    return demonstrations


def process_ctx(ctx):
    # ctx = ctx.strip()
    # if ctx[-1] not in [".", "!", "?"]:
    #     ctx = ctx + "."
    # return ctx
    return ctx


def get_nq_obqa_prompt(input_example, demonstrations, num_top_docs=2):
    prompt = ""
    for demon in demonstrations:
        context = ""
        for ctx in demon["ctxs"][:num_top_docs]:
            ctx_text = process_ctx(ctx['text'])
            context += f"{ctx['title']}. {ctx_text}\n"
        prompt += f"Context: {context}Question: {demon['question']}\nAnswer: {demon['answers'][0]}\n\n"

    context = ""
    for ctx in input_example["ctxs"][:num_top_docs]:
        ctx_text = process_ctx(ctx['text'])
        context += f"{ctx['title']}. {ctx_text}\n"
    prompt += f"Context: {context}Question: {input_example['question']}\nAnswer:"
    return prompt


def nq_obqa_evaluation(model, tokenizer, generation_kwargs, task, n_shot, seed, max_length, batch_size, num_retrieved_docs ) -> Tuple[Dict, Dict]:
    generation_kwargs["max_new_tokens"] = 16
    train_data = load_json(TASK_DATA_PATH["nq_obqa"]["validation"])
    data = load_json(TASK_DATA_PATH["nq_obqa"]["test"])
    demonstrations = get_obqa_demonstration(train_data, n_shot, seed)
    prompt_list = []
    for item in data:
        prompt_list.append({"prompt": get_nq_obqa_prompt(item, demonstrations, num_top_docs=num_retrieved_docs)})
    all_pred_ans = generate(model, tokenizer, prompt_list, generation_kwargs, max_length, batch_size, task, n_shot,
                            seed)
    all_pred_ans = [pred.split("\n")[0] for pred in all_pred_ans]
    em_score = eval_generation_em_answers(data, all_pred_ans) * 100
    return {"score": em_score}, {"prompts": prompt_list, "preds": all_pred_ans}


def get_hotpotqa_prompt(input_example, demonstrations):
    prompt = ""
    for demon in demonstrations:
        context = ""
        for title, text in demon["context"]:
            context += f"{title}. {''.join(text)}\n"
        prompt += f"Context: {context}Question: {demon['question']}\nAnswer: {demon['answer']}\n\n"

    context = ""
    for title, text in input_example["context"]:
        context += f"{title}. {''.join(text)}\n"
    prompt += f"Context: {context}Question: {input_example['question']}\nAnswer:"
    return prompt


def hotpotqa_evaluation(model, tokenizer, generation_kwargs, task, n_shot, seed, max_length, batch_size) -> Tuple[Dict, Dict]:
    generation_kwargs["max_new_tokens"] = 20
    train_data = load_json(TASK_DATA_PATH["hotpotqa"]["train"])
    data = load_json(TASK_DATA_PATH["hotpotqa"]["validation"])
    demonstrations = get_obqa_demonstration(train_data, n_shot, seed)
    prompt_list = []
    for item in data:
        prompt_list.append({"prompt": get_hotpotqa_prompt(item, demonstrations)})
    all_pred = generate(model, tokenizer, prompt_list, generation_kwargs, max_length, batch_size, task, n_shot, seed)
    all_pred = [pred.split("\n")[0] for pred in all_pred]
    correct_cnt = 0
    for pred, item in zip(all_pred, data):
        if exact_match_score(pred, item["answer"]):
            correct_cnt += 1
    em_score = correct_cnt / len(data) * 100
    return {"score": em_score}, {"prompts": prompt_list, "preds": all_pred}


def get_squad_prompt(input_example, demonstrations):
    prompt = ""
    for item in demonstrations:
        prompt = prompt + f"Passage: {item['context'].strip()}\nQuestion: {item['question'].strip()}\nAnswer: {item['answers']['text'][0].strip()}\n\n"
    prompt = prompt + f"Passage: {input_example['context'].strip()}\nQuestion: {input_example['question'].strip()}\nAnswer:"
    return prompt


def squad_evaluation(model, tokenizer, generation_kwargs, task, n_shot, seed, max_length, batch_size) -> Tuple[Dict, Dict]:
    generation_kwargs["max_new_tokens"] = 16
    train_data = load_jsonl(TASK_DATA_PATH["squad"]["train"])
    test_data = load_jsonl(TASK_DATA_PATH["squad"]["validation"])
    demonstrations = get_sampled_demonstrations(train_data, n_shot, seed)
    correct_cnt = 0
    prompt_list = []
    for item in test_data:
        prompt_list.append({"prompt": get_squad_prompt(item, demonstrations)})
    predictions = generate(model, tokenizer, prompt_list, generation_kwargs, max_length, batch_size, task, n_shot, seed)
    for pred, item in zip(predictions, test_data):
        pred = normalise_pred(pred)
        targets = item['answers']['text']
        if exact_match_score_with_multiple_candidates(pred, targets):
            correct_cnt += 1
    acc = correct_cnt / len(test_data) * 100
    return {"score": acc}, {"prompts": prompt_list, "preds": predictions}

def nqswap_evaluation(model, tokenizer, generation_kwargs, task, n_shot, seed, max_length, batch_size) -> Tuple[Dict, Dict]:
    generation_kwargs["max_new_tokens"] = 16
    train_data = load_jsonl(TASK_DATA_PATH["nqswap"]["train"])
    test_data = load_jsonl(TASK_DATA_PATH["nqswap"]["validation"])
    demonstrations = get_sampled_demonstrations(train_data, n_shot, seed)
    correct_cnt = 0
    prompt_list = []
    for item in test_data:
        prompt_list.append({"prompt": get_squad_prompt(item, demonstrations)})
    predictions = generate(model, tokenizer, prompt_list, generation_kwargs, max_length, batch_size, task, n_shot, seed)
    for pred, item in zip(predictions, test_data):
        pred = normalise_pred(pred)
        targets = item['answers']['text']
        if exact_match_score_with_multiple_candidates(pred, targets):
            correct_cnt += 1
    acc = correct_cnt / len(test_data) * 100
    return {"score": acc}, {"prompts": prompt_list, "preds": predictions}

def get_tq_obqa_prompt(input_example, demonstrations, num_top_docs=2):
    prompt = ""
    for demon in demonstrations:
        context = ""
        for ctx in demon["ctxs"][:num_top_docs]:
            context += f"{ctx['title']}. {ctx['text']}\n"
        if "target" in demon.keys():
            cur_answer = demon["target"]
        else:
            cur_answer = demon["answers"][0]
        prompt += f"Context: {context}Question: {demon['question']}\nAnswer: {cur_answer}\n\n"

    context = ""
    for ctx in input_example["ctxs"][:num_top_docs]:
        context += f"{ctx['title']}. {ctx['text']}\n"
    prompt += f"Context: {context}Question: {input_example['question']}\nAnswer:"
    return prompt


def tq_obqa_evaluation(model, tokenizer, generation_kwargs, task, n_shot, seed, max_length, batch_size, num_retrieved_docs) -> Tuple[Dict, Dict]:
    generation_kwargs["max_new_tokens"] = 16
    train_data = load_json(TASK_DATA_PATH["tq_obqa"]["train"])
    data = load_json(TASK_DATA_PATH["tq_obqa"]["test"])
    demonstrations = get_obqa_demonstration(train_data, n_shot, seed)
    prompt_list = []
    for item in data:
        prompt_list.append({"prompt": get_tq_obqa_prompt(item, demonstrations, num_top_docs=num_retrieved_docs)})
    predictions = generate(model, tokenizer, prompt_list, generation_kwargs, max_length, batch_size, task, n_shot, seed)
    predictions = [pred.split("\n")[0] for pred in predictions]
    em_score = eval_generation_em_answers(data, predictions) * 100
    return {"score": em_score}, {"prompts": prompt_list, "preds": predictions}


def metric_acc(predictions, test_data, label2str, get_label_call, normalise_pred=None, normalise_target=None):
    correct_cnt = 0
    for pred, item in zip(predictions, test_data):
        if normalise_pred is not None:
            pred = normalise_pred(pred)
        target = label2str[get_label_call(item)]
        if normalise_target is not None:
            target = normalise_target(target)
        if target == pred:
            correct_cnt += 1
    acc = correct_cnt / len(test_data) * 100
    return acc


eval_callables = {
    "nq": cbqa_evaluation,
    "tq": cbqa_evaluation,
    "wq": cbqa_evaluation,
    "sst2": sst2_evaluation,
    "sst3": sst3_evaluation,
    "agnews": agnews_evaluation,
    "agnews2": agnews2_evaluation,
    "nq_obqa": nq_obqa_evaluation,
    "hotpotqa": hotpotqa_evaluation,
    "amazon": amazon_evaluation,
    "amazon2": amazon2_evaluation,
    "amazon3": amazon3_evaluation,
    "dbpedia": dbpedia_evaluation,
    "yelp": yelp_evaluation,
    "tweet_hate": tweet_hate_evaluation,
    "tweet_offensive": tweet_offensive_evaluation,
    "squad": squad_evaluation,
    "tq_obqa": tq_obqa_evaluation,
    "memtrap": memtrap_evaluation
}

def is_obqa(task):
    return "obqa" in task


def main():
    generation_kwargs = {
        "do_sample": False,
        "num_beams": 1,
        "min_length": 1,
        "eos_token_id": 2,
        "use_cache": True,
    }
    results = {
        "task": eval_args.task,
        "n_shot": eval_args.n_shot,
        "seed": eval_args.seed,
        "model": eval_args.model_path
    }

    if "/home/aiops/" in eval_args.model_path:
        # if running on local path
        save_path = "-".join(eval_args.model_path.split("/")[-2:])
    else:
        save_path = eval_args.model_path.split("/")[-1].replace("_iter", "-iter")
    # mke sure the outputs folder exists
    os.makedirs(f"./outputs/{save_path}", exist_ok=True)
    task_save_path = f"./outputs/{save_path}/{eval_args.task}_{eval_args.n_shot}_{eval_args.seed}.json"
    if is_obqa(eval_args.task):
        task_save_path = f"./outputs/{save_path}/{eval_args.task}_{eval_args.n_shot}_{eval_args.seed}_{eval_args.num_retrieved_docs}.json"
    if eval_args.flipped_ratio > 0:
        task_save_path = task_save_path.replace(".json", f"_flipped_{eval_args.flipped_ratio}.json")

    if os.path.exists(task_save_path):
        logger.info(f"{task_save_path} exists, skipping...")
        return

    tokenizer = LlamaTokenizer.from_pretrained('/home/aiops/zhuty/tinyllama/models' , padding_side='left', truncation_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    print(eval_args.flash_attn_2)
    if eval_args.flash_attn_2:
        try:
            print("Trying to use FlashAttention2")
            model = LlamaForCausalLM.from_pretrained(
                eval_args.model_path, torch_dtype=torch.float16, attn_implementation="flash_attention_2",
            )
            task_save_path = task_save_path.replace(".json", "_flash2.json")
        except Exception as err:
            logger.error(err)
            logger.info("cannot use FlashAttention2")
            eval_args.batch_size = 1
            model = LlamaForCausalLM.from_pretrained(eval_args.model_path, torch_dtype=torch.float16)
    else:
        model = LlamaForCausalLM.from_pretrained(eval_args.model_path, torch_dtype=torch.float16, token=json.load(open("/home/aiops/zhuty/hf_token.json")))

    model.eval()
    model = model.cuda()

    evaluation = eval_callables[eval_args.task]

    if not is_obqa(eval_args.task):
        assert eval_args.num_retrieved_docs == 0, "num_retrieved_docs must not be specified for non-obqa tasks"
        score, prompts_and_preds = evaluation(model, tokenizer, generation_kwargs, eval_args.task, eval_args.n_shot,
                       eval_args.seed, eval_args.max_length - 5, eval_args.batch_size, eval_args.flipped_ratio)
    else:
        assert eval_args.num_retrieved_docs != -1, "num_retrieved_docs must be specified for obqa tasks"
        score, prompts_and_preds = evaluation(model, tokenizer, generation_kwargs, eval_args.task, eval_args.n_shot,
                       eval_args.seed, eval_args.max_length - 5, eval_args.batch_size, num_retrieved_docs=eval_args.num_retrieved_docs)

    results.update(score)
    results = json.dumps(results)
    logger.info(results)
    print("Save results to: ", task_save_path)
    with open("./outputs/logs", "a") as fn:
        fn.write(results + "\n")

    with open(task_save_path, "w") as fn:
        fn.write(results)
    with open(task_save_path.replace(".json", "_prompts_and_preds_sample.json"), "w") as fn:
        json.dump({"prompts": prompts_and_preds["prompts"][:10], "preds": prompts_and_preds["preds"][:10]}, fn, indent=4)

if __name__ == '__main__':
    main()
