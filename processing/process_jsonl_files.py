import argparse
import json
import os
import tqdm
import csv
import os
import tqdm
from multiprocessing import Pool, cpu_count
import random

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str,required=True, help="jsonl, tsv, fake_title")
parser.add_argument("--dataset_name", type=str, required=True, help="dataset name")
parser.add_argument("--select_type", type=str, required=False, default="last", choices=["last", "first", "keep"], help="select last or first part of the text")
args = parser.parse_args()

def process_jsonl(input_file, output_file):
    # extract chunk number from file, file is of name "chunk_0.jsonl"
    chunk_num = int(input_file.split("_")[-1].split(".")[0])

    # Read the JSONL file
    with open(input_file, 'r') as infile:
        json_lines = infile.readlines()

    # Process each line and add additional keys
    processed_lines = []
    for i, line in enumerate(json_lines):
        data = json.loads(line)

        # Add additional keys to the data
        data['id'] = f"{chunk_num}_{i}"
        data['contents'] = data['text']
        del data['text']
        # Append the processed data to the list
        processed_lines.append(data)

    # Write the processed data back to a new JSONL file
    with open(output_file, 'w') as outfile:
        for processed_line in processed_lines:
            # Write each line as a JSON string in the same order
            outfile.write(json.dumps(processed_line) + '\n')


def add_id_and_break_into_para(input_file, output_file):
    # extract chunk number from file, file is of name "chunk_0.jsonl"
    chunk_num = int(input_file.split("_")[-1].split(".")[0])

    # Read the JSONL file
    with open(input_file, 'r') as infile:
        json_lines = infile.readlines()

    # Process each line and add additional keys
    processed_lines = []
    for i, line in enumerate(json_lines):
        data = json.loads(line)
        paragraphs = data['text'].split("\n")

        for j, para in enumerate(paragraphs):
            new_data = {}
            # Add additional keys to the data
            new_data['id'] = f"{chunk_num}_{i}_{j}"
            new_data['contents'] = para

            # Append the processed data to the list
            processed_lines.append(new_data)

    # Write the processed data back to a new JSONL file
    with open(output_file, 'w') as outfile:
        for processed_line in processed_lines:
            # Write each line as a JSON string in the same order
            outfile.write(json.dumps(processed_line) + '\n')
def make_dup_lines(input_file, output_file):
    # extract chunk number from file, file is of name "chunk_0.jsonl"
    chunk_num = int(input_file.split("_")[-1].split(".")[0])

    # Read the JSONL file
    with open(input_file, 'r') as infile:
        json_lines = infile.readlines()

    # Process each line and add additional keys
    processed_lines = []
    for i, line in enumerate(json_lines):
        data = json.loads(line)
        # Append the processed data to the list
        processed_lines.append(data)
        processed_lines.append(data) # duplicate twice

    # Write the processed data back to a new JSONL file
    with open(output_file, 'w') as outfile:
        for processed_line in processed_lines:
            # Write each line as a JSON string in the same order
            outfile.write(json.dumps(processed_line) + '\n')


def process_jsonl_add_fake_title(input_file, output_file, reduced_length=200, length_max = 2000, select_type="last" ):
    # add "title" field because this is recognized by pyserini.
    # Read the JSONL file
    with open(input_file, 'r') as infile:
        json_lines = infile.readlines()

    # Process each line and add additional keys
    processed_lines = []
    for i, line in enumerate(json_lines):
        data = json.loads(line)

        content = data['contents']

        if select_type=="last":
            splitted_content = content.split()
            if reduced_length and len(splitted_content) > reduced_length:
                content = " ".join(splitted_content[-reduced_length:])
            if len(content) > length_max:
                content = content[-length_max:]
        elif select_type=="first":
            splitted_content = content.split()
            if reduced_length and len(splitted_content) > reduced_length:
                content = " ".join(splitted_content[:reduced_length])
            if len(content) > length_max:
                content = content[:length_max]
        else:
            assert select_type == "keep"
        data['title'] = content
        del data['contents']
        # Append the processed data to the list
        processed_lines.append(data)

    # Write the processed data back to a new JSONL file
    with open(output_file, 'w') as outfile:
        for processed_line in processed_lines:
            # Write each line as a JSON string in the same order
            outfile.write(json.dumps(processed_line) + '\n')

def shuffle_lines(input_folder, output_folder, seed = 42):

    # Read all lines from all files
    all_lines = []
    lines_lengths = []
    for filename in os.listdir(input_folder):
        if filename.endswith('.jsonl'):
            with open(os.path.join(input_folder, filename), 'r') as file:
                curr_lines = file.readlines()
                lines_lengths.append(len(curr_lines))
                all_lines.extend(curr_lines)
    print("Read", len(all_lines), "lines")
    # Shuffle the lines
    random.seed(seed)
    random.shuffle(all_lines)
    curr_chunk = 0
    while len(all_lines) > 0:
        with open(os.path.join(output_folder, f"chunk_{curr_chunk}.jsonl"), 'w') as outfile:
            for line in all_lines[:lines_lengths[curr_chunk]]:
                # Write each line as a JSON string in the same order
                outfile.write(line)
            all_lines = all_lines[lines_lengths[curr_chunk]:]

            curr_chunk += 1

    assert len(all_lines) == 0, "Not all lines are written, remaining lines: {}".format(len(all_lines))

# write as tsv file
def process_tsv(input_file, tsv_file):
    # Read the JSONL file
    with open(input_file, 'r') as infile:
        json_lines = infile.readlines()

    # Process each line and add additional keys
    processed_lines = []
    for i, line in enumerate(json_lines):
        data = json.loads(line)
        # Append the processed data to the list
        processed_lines.append(data)

    # Open the file in write mode with newline=''
    with open(tsv_file, 'w') as file:
        # Create a CSV writer with tab as the delimiter
        writer = csv.writer(file, delimiter='\t')

        # Write the data
        for item in processed_lines:
            writer.writerow([item['id'], item['contents']])
    print("Finished writing to file", tsv_file)

if args.mode == "make_dup":
    # OLD_BASE_DIR = "/home/aiops/zhuty/ret_pretraining_data/redpajama_20b/train"
    # NEW_BASE_DIR = "/home/aiops/zhuty/ret_pretraining_data/redpajama_20b_id_added/train"
    OLD_BASE_DIR = f"/home/aiops/zhuty/ret_pretraining_data/{args.dataset_name}/train"
    NEW_BASE_DIR = f"/home/aiops/zhuty/ret_pretraining_data/{args.dataset_name}_dup/train"

    for file_name in tqdm.tqdm(os.listdir(OLD_BASE_DIR), total=len(os.listdir(OLD_BASE_DIR))):
        # Skip the file if it is not a JSONL file
        if not file_name.endswith('.jsonl'):
            raise RuntimeError(f"File {file_name} is not a JSONL file")
        # Get the full path of the input file
        input_file_path = os.path.join(OLD_BASE_DIR, file_name)
        # Get the full path of the output file
        output_file_path = os.path.join(NEW_BASE_DIR, file_name)
        # Process the JSONL file
        make_dup_lines(input_file_path, output_file_path)
elif args.mode == "add_para_id":
    # OLD_BASE_DIR = "/home/aiops/zhuty/ret_pretraining_data/redpajama_20b/train"
    # NEW_BASE_DIR = "/home/aiops/zhuty/ret_pretraining_data/redpajama_20b_id_added/train"
    OLD_BASE_DIR = f"/home/aiops/zhuty/ret_pretraining_data/{args.dataset_name}/train"
    NEW_BASE_DIR = f"/home/aiops/zhuty/ret_pretraining_data/id_added/{args.dataset_name}/train"

    for file_name in tqdm.tqdm(os.listdir(OLD_BASE_DIR), total=len(os.listdir(OLD_BASE_DIR))):
        # Skip the file if it is not a JSONL file
        if not file_name.endswith('.jsonl'):
            raise RuntimeError(f"File {file_name} is not a JSONL file")
        # Get the full path of the input file
        input_file_path = os.path.join(OLD_BASE_DIR, file_name)
        # Get the full path of the output file
        output_file_path = os.path.join(NEW_BASE_DIR, file_name)
        # Process the JSONL file
        add_id_and_break_into_para(input_file_path, output_file_path)

elif args.mode == "shuffle_lines":
    # OLD_BASE_DIR = "/home/aiops/zhuty/ret_pretraining_data/redpajama_20b/train"
    # NEW_BASE_DIR = "/home/aiops/zhuty/ret_pretraining_data/redpajama_20b_id_added/train"
    OLD_BASE_DIR = f"/home/aiops/zhuty/ret_pretraining_data/id_added/{args.dataset_name}/train"
    NEW_BASE_DIR = f"/home/aiops/zhuty/ret_pretraining_data/id_added/{args.dataset_name}_shuffled/train"
    shuffle_lines(OLD_BASE_DIR, NEW_BASE_DIR)

elif args.mode == "add_id":
    # OLD_BASE_DIR = "/home/aiops/zhuty/ret_pretraining_data/redpajama_20b/train"
    # NEW_BASE_DIR = "/home/aiops/zhuty/ret_pretraining_data/redpajama_20b_id_added/train"
    OLD_BASE_DIR = f"/home/aiops/zhuty/ret_pretraining_data/{args.dataset_name}/train"
    NEW_BASE_DIR = f"/home/aiops/zhuty/ret_pretraining_data/id_added/{args.dataset_name}/train"

    for file_name in tqdm.tqdm(os.listdir(OLD_BASE_DIR), total=len(os.listdir(OLD_BASE_DIR))):
        # Skip the file if it is not a JSONL file
        if not file_name.endswith('.jsonl'):
            raise RuntimeError(f"File {file_name} is not a JSONL file")
        # Get the full path of the input file
        input_file_path = os.path.join(OLD_BASE_DIR, file_name)
        # Get the full path of the output file
        output_file_path = os.path.join(NEW_BASE_DIR, file_name)
        # Process the JSONL file
        process_jsonl(input_file_path, output_file_path)

elif args.mode == "tsv":
    OLD_BASE_DIR = "/home/aiops/zhuty/ret_pretraining_data/redpajama_20b_id_added/train"
    NEW_BASE_DIR = "/home/aiops/zhuty/ret_pretraining_data/redpajama_20b_id_added/queries"

    for file_name in tqdm.tqdm(os.listdir(OLD_BASE_DIR), total=len(os.listdir(OLD_BASE_DIR))):
        # Skip the file if it is not a JSONL file
        if not file_name.endswith('.jsonl'):
            raise RuntimeError(f"File {file_name} is not a JSONL file")
        # Get the full path of the input file
        input_file_path = os.path.join(OLD_BASE_DIR, file_name)
        # Get the full path of the output file
        output_file_path = os.path.join(NEW_BASE_DIR, file_name.replace("jsonl", "tsv"))
        # Process the JSONL file
        process_tsv(input_file_path, output_file_path)

elif args.mode=="fake_title":
    # OLD_BASE_DIR = "/home/aiops/zhuty/ret_pretraining_data/redpajama_2b_id_added/train"
    # NEW_BASE_DIR = "/home/aiops/zhuty/ret_pretraining_data/redpajama_2b_id_added/queries"
    # OLD_BASE_DIR = "/home/aiops/zhuty/ret_pretraining_data/redpajama_20b_id_added/train"
    # NEW_BASE_DIR = "/home/aiops/zhuty/ret_pretraining_data/redpajama_20b_id_added/queries"
    OLD_BASE_DIR = f"/home/aiops/zhuty/ret_pretraining_data/id_added/{args.dataset_name}/train"
    NEW_BASE_DIR = f"/home/aiops/zhuty/ret_pretraining_data/id_added/{args.dataset_name}/queries/{args.select_type}"
    if not os.path.exists(NEW_BASE_DIR):
        os.mkdir(NEW_BASE_DIR)
        print("Created dir", NEW_BASE_DIR)
    def process_file(file_name):
        # Skip the file if it is not a JSONL file
        if not file_name.endswith('.jsonl'):
            raise RuntimeError(f"File {file_name} is not a JSONL file")

        # Get the full path of the input file
        input_file_path = os.path.join(OLD_BASE_DIR, file_name)
        # Get the full path of the output file
        output_file_path = os.path.join(NEW_BASE_DIR, file_name)

        # Process the JSONL file
        process_jsonl_add_fake_title(input_file_path, output_file_path, select_type=args.select_type) # There is a big bug here, the select_type is not passed?

    # Get the list of file names
    file_names = os.listdir(OLD_BASE_DIR)

    # Set the number of processes to the number of available CPU cores
    num_processes = cpu_count()

    # Use a multiprocessing pool to parallelize the processing
    with Pool(num_processes) as pool:
        # Use tqdm for a progress bar
        list(tqdm.tqdm(pool.imap(process_file, file_names), total=len(file_names)))