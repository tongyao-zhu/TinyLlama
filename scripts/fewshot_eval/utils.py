
import re
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Create a function to plot the performance
def plot_performance(data, ):
    tasks = data['task'].unique()
    n_shots = data['n_shot'].unique()

    for task in tasks:
        for n_shot in n_shots:
            for num_docs in data['num_docs'].unique():
                plt.figure(figsize=(10, 6))
                subset = data[(data['task'] == task) & (data['n_shot'] == n_shot) & (data['num_docs'] == num_docs)]
                sns.lineplot(x='steps', y='mean', hue='model_name', data=subset, marker='o')
                plt.title(f'Performance vs Steps for Task: {task}, n_shot: {n_shot}, num_docs: {num_docs}')
                plt.xlabel('Steps')
                plt.ylabel('Mean Performance')
                plt.legend(title='Model')
                plt.grid(True)
                plt.show()
                if not os.path.exists(f'/home/aiops/zhuty/tinyllama/scripts/fewshot_eval/figures/{task}'):
                    os.makedirs(f'/home/aiops/zhuty/tinyllama/scripts/fewshot_eval/figures/{task}')
                plt.savefig(f'/home/aiops/zhuty/tinyllama/scripts/fewshot_eval/figures/{task}/performance_{task}_{n_shot}_{num_docs}.png')

def plot_performance_compare_shots(data):
    tasks = data['task'].unique()

    for task in tasks:
        for num_docs in data['num_docs'].unique():
            plt.figure(figsize=(10, 6))
            subset = data[(data['task'] == task)  & (data['num_docs'] == num_docs)]
            sns.lineplot(x='n_shot', y='mean', hue='model_name', data=subset, marker='o')
            plt.title(f'Performance vs nshot for Task: {task}, num_docs: {num_docs}')
            plt.xlabel('n-shot')
            plt.ylabel('Mean Performance')
            plt.legend(title='Model')
            plt.grid(True)
            plt.show()
            if not os.path.exists(f'/home/aiops/zhuty/tinyllama/scripts/fewshot_eval/figures/{task}'):
                os.makedirs(f'/home/aiops/zhuty/tinyllama/scripts/fewshot_eval/figures/{task}')
            plt.savefig(f'/home/aiops/zhuty/tinyllama/scripts/fewshot_eval/figures/{task}/performance_{task}_{num_docs}.png')


def plot_performance_compare_docs(data):
    tasks = data['task'].unique()
    n_shots = data['n_shot'].unique()

    for task in tasks:
        for n_shot in n_shots:
            # for num_docs in data['num_docs'].unique():
            plt.figure(figsize=(10, 6))
            subset = data[(data['task'] == task) & (data['n_shot'] == n_shot)]
            sns.lineplot(x='num_docs', y='mean', hue='model_name', data=subset, marker='o')
            plt.title(f'Performance vs nshot for Task: {task}, n_shot: {n_shot}')
            plt.xlabel('Num Docs')
            plt.ylabel('Mean Performance')
            plt.legend(title='Model')
            plt.grid(True)
            plt.show()
            if not os.path.exists(f'/home/aiops/zhuty/tinyllama/scripts/fewshot_eval/figures/{task}'):
                os.makedirs(f'/home/aiops/zhuty/tinyllama/scripts/fewshot_eval/figures/{task}')
            plt.savefig(f'/home/aiops/zhuty/tinyllama/scripts/fewshot_eval/figures/{task}/performance_{task}_{n_shot}_doc_comp.png')


def shorten_model_path(model_name):
    model_name = model_name.split('/')[-1]
    model_name = model_name.replace("tiny_LLaMA_", "").replace("yuzhaouoe/", "")
    if 'cc' in model_name:
        return model_name.split('_')[0]
    return model_name

def format_job_name(meta_info_dict, key_importance_order=( 'task_name', 'model_name','iter_name' 'flipped_ratio', 'seed_num', 'shot_num', 'rag_num_docs')):
    job_name = ""
    for key in key_importance_order:
        if key in meta_info_dict:
            if key == 'iter_name':
                # only keep the step number
                job_name += f"{meta_info_dict[key].split('-')[-1]}_"
            else:
                job_name += f"{meta_info_dict[key]}_"
    job_name = job_name.replace("_", "").replace("-", "").replace(".","").replace("/","").lower()[:39]
    return job_name


def extract_model_and_steps(model_name):
    """
    Extracts the model name and step count from the model string.

    Parameters:
    model_name (str): The full model string.

    Returns:
    tuple: A tuple containing the model name and step count.
    """
    model_pattern = re.compile(r'tyzhu/tiny_LLaMA_1b_(.+?)_iter-\d+-ckpt-step-(\d+)_hf')
    match = model_pattern.search(model_name)
    if match:
        return match.group(1), int(match.group(2))
    else:
        raise ValueError(f"Model name {model_name} does not match the expected pattern.")

def extract_step(x):
    # strip the _hf
    x = x[:-3]
    return int(x.split('-')[-1])

def extract_model_name(x):
    """
    Sample input: /home/aiops/zhuty/tinyllama/out/tiny_LLaMA_1b_8k_cc_8k/iter-200000-ckpt-step-25000_hf
    Sample output: tiny_LLaMA_1b_8k
    Args:
        x:
    Returns:

    """
    if 'TinyLlama/TinyLlama-1.1B-step-50K-105b' in x:
        return 'tinyllama-1.1b-50k-105b'
    if '/home/aiops/zhuty' in x:
        folder_name = x.split('/')[-2] # tiny_LLaMA_1b_8k_cc_8k
        return ("_").join(folder_name.split('_')[:4]) # tiny_LLaMA_1b_8k
    else:
        # sample model name: tyzhu/tiny_LLaMA_120M_8k_cc_8k_iter-240000-ckpt-step-60000_hf
        return x.strip("tyzhu/").split("_cc")[0]

def extract_pretrain_dataset(x):
    """
    Sample input: /home/aiops/zhuty/tinyllama/out/tiny_LLaMA_1b_8k_cc_8k/iter-200000-ckpt-step-25000_hf
    Sample output: cc_8k
    Args:
        x:
    Returns:

    """
    if '/home/aiops/zhuty' in x:
        folder_name = x.split('/')[-2] # tiny_LLaMA_1b_8k_cc_8k
        return ("_").join(folder_name.split('_')[4:]) # cc_8k
    else:
        # sample model name: tyzhu/tiny_LLaMA_120M_8k_cc_8k_iter-240000-ckpt-step-60000_hf
        return x.strip("tyzhu/").split("_8k_")[1]

def remove_underline(x):
    return x.replace("_", "")

def normalise_pred(pred):
    return pred.strip().split("\n")[0].strip()

CLASS_TO_TASKS_MAPPING = { "icl":  ['tweet_hate' ,'tweet_offensive', 'agnews' ,'amazon' ,'dbpedia' ,'sst2','yelp',],
                           "icl_short":  ['amazon'  ,'sst2','yelp',],
                           "agnews":  ['agnews'],
                           "agnewsnoise":  ['agnews'],
                           "agnews2noise":  ['agnews2'],
                            "dbpedia": ['dbpedia'],
                            "dbpedianoise": ['dbpedia'],
                          "obqa": [ "squad","hotpotqa"],
                          "obqa_rag": ["nq_obqa", "tq_obqa"],
                          "obqa_rag0": ["nq_obqa", "tq_obqa"],
                          "obqa_rag1": ["nq_obqa", "tq_obqa"],
                          "cbqa": ["tq", "nq"],
                          "memtrap" :["memtrap"],
                           "sst2noise": ["sst2"],
                           "sst3noise": ["sst3"],
                           "amazonnoise": ["amazon"],
                           "amazon2noise": ["amazon2"],
                           "amazon3noise": ["amazon3"],
                          }

CLASS_TO_SEED_MAPPING = {"icl": range(42, 58),
                         "icl_short": range(42, 53),
                         "agnews": range(42, 58),
                         "agnewsnoise": range(42, 50),
                         "agnews2noise": range(42, 50),
                         "obqa": range(42, 46+1),
                         "cbqa": range(42, 46 + 1),
                            "obqa_rag": range(42, 46 + 1),
                            "obqa_rag1": range(42, 46 + 1),
                            "obqa_rag0": range(42, 43),
                         "dbpedia": range(42, 58),
                         "dbpedianoise": range(42, 50),
                         "memtrap":[43],
                            "sst2noise": range(42, 58),
                            "sst3noise": range(42, 58),
                            "amazonnoise": range(42, 58),
                            "amazon2noise": range(42, 50),
                            "amazon3noise": range(42, 50),
                         }

ALL_TASKS = [
    {
        "TASK_CLASS": "obqa",
        "shot_nums": [4],
        "rag_num_docs_lst": [""]
    },
    {
        "TASK_CLASS": "icl",
        "shot_nums": [24, 48],
        "rag_num_docs_lst": [0]
    },
    {
        "TASK_CLASS": "dbpedia",
        "shot_nums": [24, 36 ,48 , 60,72],
        "rag_num_docs_lst": [0]
    },
    {
        "TASK_CLASS": "dbpedianoise",
        "shot_nums": [24, 36, 48, 60, 72],
        "rag_num_docs_lst": [0],
        "flipped_ratios": [0, 0.2, 0.4, 0.6, 0.8, 1]
    },
    {
        "TASK_CLASS": "icl_short",
        "shot_nums": [5],
        "rag_num_docs_lst": [0]
    },
    {
        "TASK_CLASS": "agnews",
        "shot_nums": [5, 10, 24, 36, 48 ],
        "rag_num_docs_lst": [0]
    },
    {
        "TASK_CLASS": "memtrap",
        "shot_nums": [0],
        "rag_num_docs_lst": [""]
    },
    {
        "TASK_CLASS": "cbqa",
        "shot_nums": [12, 24],
        "rag_num_docs_lst": [0]
    },
    {
        "TASK_CLASS": "obqa_rag",
        "shot_nums": [3],
        "rag_num_docs_lst": [1, 3, 5, 10]
    },
    {
        "TASK_CLASS": "obqa_rag1",
        "shot_nums": [2],
        # "rag_num_docs_lst": [1, 3, 5, 7, 10, 15, 20]
        "rag_num_docs_lst": [1, 3, 5, 7, 10, 15,]
    },
    {
        "TASK_CLASS": "obqa_rag0",
        "shot_nums": [0],
        "rag_num_docs_lst": [1, 3, 5, 10, 15, 20, 30, 50]
    },
    {
        "TASK_CLASS": "sst2noise",
        "shot_nums": [10, 20, 40,],
        "flipped_ratios": [0, 0.2, 0.4, 0.6, 0.8, 1]
    },
    {
        "TASK_CLASS": "sst3noise",
        "shot_nums": [10, 20, 40, ],
        "flipped_ratios": [0, 0.2, 0.4, 0.6, 0.8, 1]
    },
    {
        "TASK_CLASS": "amazonnoise",
        "shot_nums": [10, 20, 40, ],
        "flipped_ratios": [0, 0.2, 0.4, 0.6, 0.8, 1]
    },
    {
        "TASK_CLASS": "amazon2noise",
        "shot_nums": [10, 20, 40, ],
        "flipped_ratios": [0, 0.2, 0.4, 0.6, 0.8, 1]
    },
    {
        "TASK_CLASS": "agnewsnoise",
        "shot_nums": [10, 20, 40, ],
        "flipped_ratios": [0, 0.2, 0.4, 0.6, 0.8, 1]
    },
    {
        "TASK_CLASS": "agnews2noise",
        "shot_nums": [10, 20, 40, ],
        "flipped_ratios": [0, 0.2, 0.4, 0.6, 0.8, 1]
    },
    {
        "TASK_CLASS": "amazon3noise",
        "shot_nums": [10, 20, 40, ],
        "flipped_ratios": [0, 0.2, 0.4, 0.6, 0.8, 1]
    },

]

