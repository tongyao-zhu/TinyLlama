import json
from graphs.utils import get_file_ids
import argparse
import glob
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_dir', type=str, required=True)
                        # default='/home/aiops/zhuty/ret_pretraining_data/redpajama_2b_id_added/train/')
    parser.add_argument('--result_path_file', type=str, required=True)
    parser.add_argument("--reordered_data_dir", type=str, required=True)
    parser.add_argument('--test', action='store_true')
    return parser.parse_args()


def read_jsonl(files_dir):
    """Read a JSON Lines file and return a list of documents."""
    documents = []
    for file_path in glob.glob(os.path.join(files_dir, '*.jsonl')):
        with open(file_path, 'r') as file:
            for line in file:
                document = json.loads(line)
                document['text'] = document['contents']
                del document['contents']
                documents.append(document)
    return documents


def get_num_chunks(files_dir):
    """Get the number of chunks in a directory."""
    return len(glob.glob(os.path.join(files_dir, '*.jsonl')))


def write_jsonl(dir_path, documents, chunk_size=10000):
    """Write a list of documents to chunks of JSON Lines file."""
    num_chunks = len(documents) // chunk_size + 1
    for i in range(num_chunks):
        chunk_path = os.path.join(dir_path, f'chunk_{i}.jsonl')
        start = i * chunk_size
        end = min((i + 1) * chunk_size, len(documents))
        with open(chunk_path, 'w') as file:
            for document in documents[start:end]:
                file.write(json.dumps(document) + '\n')


def main():
    args = parse_args()
    result_path = json.load(open(args.result_path_file, 'r'))
    result_path = [item for sublist in result_path for item in sublist]
    print("First 10 nodes in the result path:", result_path[:10])
    # ensure the correctness of the result path
    # uniqueness
    # no cycle
    assert len(result_path) == len(set(result_path)), 'duplicate node in result path'

    # no duplicate node
    print(len(result_path), len(set(result_path)))
    # get all file ids
    train_file_path = args.train_data_dir
    all_file_ids = get_file_ids(train_file_path)
    print(len(all_file_ids))
    # check if the result path contains all file ids
    assert args.test or set(result_path) == set(all_file_ids), 'result path does not contain all file ids'

    documents = read_jsonl(args.train_data_dir)

    # Create a mapping of doc_id to document
    doc_id_to_document = {doc['id']: doc for doc in documents}

    # Create a list of documents in the specified order
    ordered_documents = [doc_id_to_document[doc_id] for doc_id in result_path]

    # Write the ordered documents to a new JSON Lines file
    write_jsonl(args.reordered_data_dir, ordered_documents,
                chunk_size=len(documents) // get_num_chunks(args.train_data_dir))


if __name__ == '__main__':
    main()
