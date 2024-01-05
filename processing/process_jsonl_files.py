import json
import os
import tqdm
import csv
import os
import tqdm
from multiprocessing import Pool, cpu_count
MODE = "fake_title"
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

def process_jsonl_add_fake_title(input_file, output_file, reduced_length=200, length_max = 2000):
    # add "title" field because this is recognized by pyserini.
    # Read the JSONL file
    with open(input_file, 'r') as infile:
        json_lines = infile.readlines()

    # Process each line and add additional keys
    processed_lines = []
    for i, line in enumerate(json_lines):
        data = json.loads(line)

        content = data['contents']
        splitted_content = content.split()
        if reduced_length and len(splitted_content) > reduced_length:
            content = " ".join(splitted_content[-reduced_length:])
        if len(content) > length_max:
            content = content[-length_max:]
        data['title'] = content
        del data['contents']
        # Append the processed data to the list
        processed_lines.append(data)

    # Write the processed data back to a new JSONL file
    with open(output_file, 'w') as outfile:
        for processed_line in processed_lines:
            # Write each line as a JSON string in the same order
            outfile.write(json.dumps(processed_line) + '\n')


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

if MODE == "jsonl":
    OLD_BASE_DIR = "/home/aiops/zhuty/ret_pretraining_data/redpajama_20b/train"
    NEW_BASE_DIR = "/home/aiops/zhuty/ret_pretraining_data/redpajama_20b_id_added/train"

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

elif MODE == "tsv":
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

elif MODE=="fake_title":
    # OLD_BASE_DIR = "/home/aiops/zhuty/ret_pretraining_data/redpajama_2b_id_added/train"
    # NEW_BASE_DIR = "/home/aiops/zhuty/ret_pretraining_data/redpajama_2b_id_added/queries"
    OLD_BASE_DIR = "/home/aiops/zhuty/ret_pretraining_data/redpajama_20b_id_added/train"
    NEW_BASE_DIR = "/home/aiops/zhuty/ret_pretraining_data/redpajama_20b_id_added/queries"

    def process_file(file_name):
        # Skip the file if it is not a JSONL file
        if not file_name.endswith('.jsonl'):
            raise RuntimeError(f"File {file_name} is not a JSONL file")

        # Get the full path of the input file
        input_file_path = os.path.join(OLD_BASE_DIR, file_name)
        # Get the full path of the output file
        output_file_path = os.path.join(NEW_BASE_DIR, file_name)

        # Process the JSONL file
        process_jsonl_add_fake_title(input_file_path, output_file_path)

    # Get the list of file names
    file_names = os.listdir(OLD_BASE_DIR)

    # Set the number of processes to the number of available CPU cores
    num_processes = cpu_count()

    # Use a multiprocessing pool to parallelize the processing
    with Pool(num_processes) as pool:
        # Use tqdm for a progress bar
        list(tqdm.tqdm(pool.imap(process_file, file_names), total=len(file_names)))