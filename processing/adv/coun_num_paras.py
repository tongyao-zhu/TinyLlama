import os
import json
import tqdm
import argparse
import concurrent.futures
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='cc')
    return parser.parse_args()



def read_jsonl_paths(paths):
    """Read a JSON Lines file and return a list of documents."""
    documents = []
    for file_path in tqdm.tqdm(paths):
        with open(file_path, 'r') as file:
            for line in file:
                document = json.loads(line)
                documents.append(document)
    return documents

def read_jsonl(file_path):
    """Read a JSON Lines file and return a list of documents."""
    documents = []
    with open(file_path, 'r') as file:
        for line in file:
            document = json.loads(line)
            documents.append(document)
    return documents
def count_num_paragrahs(file_path):
    """Count the number of paragraphs in a JSON Lines file."""
    with open(file_path, 'r') as file:
        print("Processing file:", file_path)
        total = 0
        for line in file:
            document = json.loads(line)
            total += len(document['text'].split('\n'))
    return total

def write_jsonl(documents, file_path):
    """Write a list of documents to a JSON Lines file."""
    with open(file_path, 'w') as file:
        for document in documents:
            line = json.dumps(document)
            file.write(line + '\n')
def process_jsonl(file_path, num_chunks=5):
    """read jsonl and break into smaller documents"""
    documents = read_jsonl(file_path)
    # break it into smaller paragraphs
    new_documents = []
    for doc in documents:
        paragraphs = doc['text'].split("\n")
        paragraphs_chunks = [paragraphs[i:i + num_chunks] for i in range(0, len(paragraphs), num_chunks)]
        for i, chunk in enumerate(paragraphs_chunks):
            new_documents.append({"id": doc["id"]+ f"_chunk_{i}"
                                                   , "text": "\n".join(chunk)})
    new_file_path = file_path.replace(".jsonl", f"_chunked_{num_chunks}.jsonl")
    print(f"Writing to {new_file_path}")
    write_jsonl(new_documents, new_file_path)
    print(f"Finished writing to {new_file_path}")
    return


if __name__ == '__main__':
    args = parse_args()
    json_path = '/home/aiops/zhuty/ret_pretraining_data/{}/train'.format(args.dataset_name)
    file_paths = [os.path.join(json_path, file_name) for file_name in os.listdir(json_path)]
    # Use multiprocessing to process the jsonl files
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(process_jsonl, file_paths)



    # # Use multiprocessing to count the total number of paragraphs in the dataset
    # documents = []
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     for curr_docs in executor.map(read_jsonl, file_paths):
    #         documents.extend(curr_docs)
    # num_of_paragraphs = []
    # all_paragraphs = []
    # for doc in documents:
    #     paragraphs = doc['text'].split("\n")
    #     num_of_paragraphs.append(len(paragraphs))
    #     all_paragraphs.extend(paragraphs)
    #
    # # get some stats on the number of paragraphs
    # print("Total number of paragraphs: ", sum(num_of_paragraphs))
    # print("Average number of paragraphs: ", sum(num_of_paragraphs) / len(num_of_paragraphs))
    # print("Max number of paragraphs: ", max(num_of_paragraphs))
    # print("Min number of paragraphs: ", min(num_of_paragraphs))