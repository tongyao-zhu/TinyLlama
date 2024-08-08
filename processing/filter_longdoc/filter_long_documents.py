import argparse
import os
import pandas as pd
import json
import nltk
from nltk.tokenize import sent_tokenize
import tqdm
import glob

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='cc')
    parser.add_argument("--threshold", type=int, default=1000, help="The threshold for breaking long documents")
    return parser.parse_args()

def read_length_csv(file_path):
    df = pd.read_csv(file_path)
    return df

def read_all_lengths_csv(file_paths):
    dfs = []
    for file_path in file_paths:
        df = read_length_csv(file_path)
        dfs.append(df)
    total_df = pd.concat(dfs)
    # total df has columns: id, length
    # return a dictionary for
    # key: id
    # value: length
    return total_df.set_index('id').to_dict()['length']
def break_document(data, num_sections):
    # Your document as a string variable
    document = data['contents']

    # Split the document into sentences
    sentences = sent_tokenize(document)

    # Calculate the target number of sentences per section
    target_section_length = len(sentences) // num_sections

    sections = []  # List to hold the sections
    current_section = []  # Current section being filled
    current_length = 0  # Number of sentences in the current section

    for sentence in sentences:
        current_section.append(sentence)
        current_length += 1
        # Check if the current section reaches or exceeds the target length
        if current_length >= target_section_length:
            # Join the sentences to form the section and add to the sections list
            sections.append(' '.join(current_section))
            # Reset for the next section
            current_section = []
            current_length = 0

    # Add the last section if there are any remaining sentences
    if current_section:
        sections.append(' '.join(current_section))

    new_instances = [{"id": f"{data['id']}_{i}", "contents": section} for i, section in enumerate(sections)]
    return new_instances

def copy_with_broken_ids(source_file, target_file, id2section_num):
    with open(source_file, 'r') as source:
        with open(target_file, 'w') as target:
            for i, line in tqdm.tqdm(enumerate(source), desc=f"Processing {source_file}"):
                data = json.loads(line)
                if data['id'] in id2section_num:
                    # break the document
                    new_data_instances = break_document(data, id2section_num[data['id']])
                    for new_data in new_data_instances:
                        target.write(json.dumps(new_data) + '\n')
                else:
                    target.write(line)
    return

def copy_with_length_threshold(source_file, target_file, threshold, lengths_dict):
    """
    Copy the documents from source_file to target_file if the length of the document is less than threshold.
    Args:
        source_file:
        target_file:
        threshold:
        lengths_dict:

    Returns:

    """
    with open(source_file, 'r') as source:
        with open(target_file, 'w') as target:
            for i, line in tqdm.tqdm(enumerate(source), desc=f"Processing {source_file}"):
                data = json.loads(line)
                if lengths_dict[data['id']] < threshold:
                    target.write(line)
    return


def processing(chunk_num):
    args = parse_args()
    # BASE_PATH=f"/home/aiops/zhuty/ret_pretraining_data/id_added/{args.dataset_name}/train"
    BASE_PATH = f'/s3/ret_pretraining_data/{args.dataset_name}/train'
    NEW_PATH=f"/home/aiops/zhuty/ret_pretraining_data/id_added/{args.dataset_name}/train_reduced_{args.threshold}"
    if not os.path.exists(NEW_PATH):
        os.makedirs(NEW_PATH, exist_ok=True)
    # os.mkdir(f"/home/aiops/zhuty/ret_pretraining_data/id_added/{args.dataset_name}/train_reduced_{args.threshold}")
    # "/home/aiops/zhuty/ret_pretraining_data/id_added/cc/train/chunk_*_lengths.csv"
    length_files = glob.glob(f"/home/aiops/zhuty/ret_pretraining_data/id_added/cc/train/chunk_*_lengths.csv")
    lengths_dict = read_all_lengths_csv(length_files)
    # lengths = read_length_csv(f"/home/aiops/zhuty/ret_pretraining_data/id_added/{args.dataset_name}/train/chunk_{chunk_num}_lengths.csv")
    print("Processing chunk: ", chunk_num)
    # broken_ids = {}
    # for i, row in lengths.iterrows():
    #     length = row['length']
    #     if length > args.threshold:
    #         # print(f"Length {length} is greater than threshold {args.threshold}. Breaking document {i}...")
    #         broken_ids[f"{chunk_num}_{i}"] = round(length/args.threshold)
    # print("Total number of documents broken: ", len(broken_ids))
    # print(f"Percentage of documents broken: {len(broken_ids)/lengths.shape[0]*100:.2f}%")
    # print("Broken ids: ", list(broken_ids.keys())[:10])
    copy_with_length_threshold(
        source_file=os.path.join(BASE_PATH, f"chunk_{chunk_num}.jsonl"),
        target_file=os.path.join(NEW_PATH, f"chunk_{chunk_num}.jsonl"),
        threshold=args.threshold,
        lengths_dict=lengths_dict )

    return

def main():
    # Use multiprocessing
    import multiprocessing
    pool = multiprocessing.Pool(64)
    pool.map(processing, range(100))

if __name__ == '__main__':
    main()