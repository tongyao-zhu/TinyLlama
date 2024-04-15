from bm25_pt import BM25
import tqdm
import time
import numpy as np


bm25 = BM25(device='cuda')
import os
import json


BASE_DIR = '/home/aiops/zhuty/ret_pretraining_data/id_added/cc/queries/first/'
index_name = "cc_index_first"


def read_chunk(chunk_num):
    """Reads the file through csv.DictReader() and for each row
    """
    all_data = []
    with open(os.path.join(BASE_DIR, f'chunk_{chunk_num}.jsonl'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            data['_id'] = data['id']
            if 'title' in data:
                data['contents'] = data['title']
                del data['title']
            all_data.append(data)
            # client.index(index=f"my_index_{CHUNK_NUM}", id=data['id'], document=data)
    return all_data

UPPER_CHUNK_LIMIT = 100

all_ids = []
all_docs = []
start_time = time.time()
for chunk_num in tqdm.tqdm(range(0, UPPER_CHUNK_LIMIT), desc="Processing chunks"):
    data = read_chunk(chunk_num)
    docs = [d['contents'] for d in data]
    ids = [d['_id'] for d in data]
    all_ids.extend(ids)
    all_docs.extend(docs)

print("Finish reading all chunks, time: ", time.time() - start_time)
print("Total number of documents: ", len(all_docs))
print("Total number of ids: ", len(all_ids))
print("Start adding to index...")
bm25.index(all_docs)
print("Finish adding to index, time: ", time.time() - start_time)
end_time = time.time()

chunk_num = 0
print("Finish adding to index, time: ", end_time - start_time)
data = read_chunk(chunk_num)
print("Start scoring...")
start_time = time.time()
all_queries = [d['contents'] for d in data]

# process the queries in batches
batch_size = 100
queries = []
for i in range(0, len(all_queries), batch_size):
    queries.append(all_queries[i:i+batch_size])

all_scores = []
# score the queries
for i, q in tqdm.tqdm(enumerate(queries), desc='Searching',total=len(queries) ):
    doc_scores = bm25.score_batch(q)
    all_scores.extend(doc_scores)

end_time = time.time()
print("Finish scoring, time: ", end_time - start_time)

all_scores = np.array(all_scores)
# save all_scores as a numpy array
np.save(f'all_scores_chunk_{chunk_num}.npy', all_scores)


