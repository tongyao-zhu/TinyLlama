import faiss
import numpy as np

USE_GPU = False
# Load your existing flat index
flat_index = faiss.read_index('/home/aiops/zhuty/ret_pretraining_data/id_added/c4_news/dense_index/shard_full/flatindex')

# Determine the number of vectors to sample for training
n_sample = 3000000  # for example, 10,000 vectors

# Randomly sample vector IDs
np.random.seed(42)  # for reproducibility
sample_ids = np.random.choice(flat_index.ntotal, n_sample, replace=False).astype(int)
res = faiss.StandardGpuResources()  # use a single GPU
# Retrieve the vectors corresponding to sampled IDs
sampled_vectors = flat_index.reconstruct_n(0, flat_index.ntotal)[sample_ids]
print("sampled_vectors.shape:", sampled_vectors.shape)

index_string = "HNSW64"
# Create an IVFPQ index
ivfpq_index = faiss.index_factory(flat_index.d, index_string,)
if USE_GPU:
    # make it into a gpu index
    ivfpq_index = faiss.index_cpu_to_gpu(res, 0, ivfpq_index)
# Train the IVFPQ index with the sampled vectors
ivfpq_index.train(sampled_vectors)
# Convert to a CPU index (if necessary)
if USE_GPU:
    ivfpq_index = faiss.index_gpu_to_cpu(ivfpq_index)
# Add vectors to the IVFPQ index
ivfpq_index.add(flat_index.reconstruct_n(0, flat_index.ntotal))

# Save the IVFPQ index
faiss.write_index(ivfpq_index, '/home/aiops/zhuty/ret_pretraining_data/id_added/c4_news/dense_index/shard_full/index_{}'.format(index_string.replace(",", "_")))
