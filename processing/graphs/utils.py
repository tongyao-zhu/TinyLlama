import json
import os
import tqdm

def get_file_ids(directory_path):
    file_ids = set()

    # Iterate over files in the directory
    for filename in tqdm.tqdm(os.listdir(directory_path)):
        if filename.endswith(".jsonl"):
            file_path = os.path.join(directory_path, filename)

            # Read each JSONL file
            with open(file_path, 'r') as file:
                for line in file:
                    # Parse each line as JSON
                    data = json.loads(line)
                    file_ids.add(data['id'])

    return file_ids

def read_trec_results(file_path):
    results = {}

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()

            query_id = parts[0]
            doc_id = parts[2]
            rank = int(parts[3])
            score = float(parts[4])

            if query_id not in results:
                results[query_id] = []

            results[query_id].append({'doc_id': doc_id, 'score': score, 'rank': rank})

    return results