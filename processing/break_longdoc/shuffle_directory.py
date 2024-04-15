import os
import random
import argparse
import os
import json
import tqdm
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='cc')
    parser.add_argument('--dir_name', type=str, required=True, help='The directory name')
    return parser.parse_args()


def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]
def main():
    random.seed(42)
    args = parse_args()

    # Directory containing your JSONL files
    folder_path = f'/home/aiops/zhuty/ret_pretraining_data/id_added/{args.dataset_name}/{args.dir_name}'

    # List to store all lines from all files
    all_lines = []
    # count the jsonl files in the directory
    num_chunks = len([file for file in os.listdir(folder_path) if file.endswith('.jsonl')])
    # Read all JSONL files and collect their lines
    for file_name in tqdm.tqdm(os.listdir(folder_path), desc=f"Reading files in {folder_path}"):
        if file_name.endswith('.jsonl'):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as file:
                all_lines.extend(file.readlines())

    print("In total: ", len(all_lines), " lines")
    # Shuffle the collected lines
    random.shuffle(all_lines)
    print("Finished shuffling", len(all_lines), " lines")

    # write the shuffled lines to num_chunks files
    shuffled_folder_path = f'/home/aiops/zhuty/ret_pretraining_data/id_added/{args.dataset_name}/{args.dir_name}_shuffled'
    os.mkdir(shuffled_folder_path)
    chunk_size = len(all_lines) // num_chunks
    for i, chunk in tqdm.tqdm(enumerate(divide_chunks(all_lines, chunk_size)), desc=f"Writing to {shuffled_folder_path}"):
        shuffled_file_path = os.path.join(shuffled_folder_path, f'chunk_{i}.jsonl')
        with open(shuffled_file_path, 'w') as file:
            file.writelines(chunk)

    print(f"Finished writing to {shuffled_folder_path}")

if __name__ == '__main__':
    main()
    