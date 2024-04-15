import argparse
import json
import os
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='cc')
    return parser.parse_args()

def read_jsonl(file_path):
    """Read a JSON Lines file and return a list of documents."""
    documents = []
    with open(file_path, 'r') as file:
        for line in file:
            document = json.loads(line)
            documents.append(document)
    return documents

def write_jsonl(documents, file_path):
    """Write a list of documents to a JSON Lines file."""
    with open(file_path, 'w') as file:
        for document in documents:
            line = json.dumps(document)
            file.write(line + '\n')

def main():
    args = parse_args()
    path= '/home/aiops/zhuty/ret_pretraining_data/id_added/cc/generated_queries/last'
    for chunk_num in range(0, 100):
        filename = f"tiny_LLaMA_120M_8k_cc_8k_chunk_{chunk_num}_result.jsonl"
        if not os.path.exists(os.path.join(path, filename)):
            print(f"File {filename} does not exist")
            continue
        print(f"Processing {filename}")
        documents = read_jsonl(os.path.join(path, filename))
        input_docs = read_jsonl(os.path.join(path, f"tiny_LLaMA_120M_8k_cc_8k_chunk_{chunk_num}_input.jsonl"))
        new_documents = []
        for input_doc, doc in zip(input_docs, documents):
            gen_query = doc[0]['generated_text']
            new_documents.append({"id": input_doc["id"], "title": gen_query})

        new_file_path = os.path.join("/home/aiops/zhuty/ret_pretraining_data/id_added/cc/", "queries", "last_120m", f"chunk_{chunk_num}.jsonl")
        print(f"Writing to {new_file_path}")
        write_jsonl(new_documents, new_file_path)
    print(f"Finished writing to {new_file_path}")
    return



if __name__ == '__main__':
    main()