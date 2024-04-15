from transformers import AutoTokenizer

import os
import argparse
import json
import numpy as np
import pandas as pd
import multiprocessing

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=True)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--text_field", type=str, default='contents')
    args = parser.parse_args()
    return args

def get_doc_length(file):
    doc_lengths = []
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            doc = json.loads(line)
            doc_lengths.append(len(tokenizer.encode(doc['text'])))
    return doc_lengths

def process_file(file):
    doc_lengths = get_doc_length(file)
    length_file = file.replace('.jsonl', '_lengths.csv')
    df = pd.DataFrame(doc_lengths, columns=['length'])
    df.to_csv(length_file, index=False)
    return length_file
def main():
    args = parse_args()
    dataset_name = args.dataset_name
    print(f"Processing {dataset_name}...")
    # list all jsonl files in the directory
    # path = f"/home/aiops/zhuty/ret_pretraining_data/id_added/{dataset_name}/train"
    path = f"/home/aiops/zhuty/ret_pretraining_data/{dataset_name}/valid"
    files = [f for f in os.listdir(path) if f.endswith('.jsonl')]
    print("In total, there are", len(files), "files")
    # get the length of each file with multiprocessing
    with multiprocessing.Pool(64) as p:
        length_files = p.map(process_file, [os.path.join(path, f) for f in files])

    print("Done!")



if __name__ == "__main__":
    main()
