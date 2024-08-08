
import pandas as pd
import os

BASE_DIR="/home/aiops/zhuty/ret_pretraining_data/id_added/cc/train"
dfs = []
for i in range(0, 100):
    chunk = pd.read_csv(os.path.join(BASE_DIR, "chunk_{}_lengths.csv".format(i)))
    dfs.append(chunk)
df = pd.concat(dfs, ignore_index=True)

print(df['length'].describe())
print("Total number of tokens (B): ", df['length'].sum()/1e9)

# df = pd.read_csv("chunk_99_lengths.csv")
# get how many lengths are smaller than 1k, 2k, 4k, 8k
for threshold in [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000]:
    print("Number of documents with length smaller than {}: ".format(threshold), df[df['length'] < threshold].shape[0], "({:.2f}%)".format(df[df['length'] < threshold].shape[0]/df.shape[0]*100))
    print("Number of tokens (B) with length smaller than {}: ".format(threshold), df[df['length'] < threshold]['length'].sum()/1e9, "({:.2f}%)".format(df[df['length'] < threshold]['length'].sum()/df['length'].sum()*100))
# get the distribution of the lengths
print(df['length'].describe())
print("Total number of tokens (B): ", df['length'].sum()/1e9)

#
import numpy as np
import matplotlib.pyplot as plt

# Assuming 'lengths' is your numpy array of document lengths
# lengths = np.array([...])
lengths = df['length'].values
# Create logarithmically spaced bins
min_length = lengths.min()
max_length = lengths.max()
bins = np.logspace(np.log10(min_length), np.log10(max_length), 50)

# Plot the histogram of the lengths with logarithmically spaced bins
plt.figure(figsize=(10, 6))
plt.hist(lengths, bins=bins, density=True, alpha=0.6, color='b')

# Set log scale for x and y axis
plt.xscale('log')
plt.yscale('log')

plt.xlabel('Document Length')
plt.ylabel('Frequency')
plt.title('Distribution of Document Lengths (Log-Log Scale)')
plt.grid(True)

plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Sample lengths array
# lengths = np.array([1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000])
lengths = df['length'].values.tolist()
print(lengths[:10])

# Add EOS tokens between documents
eos_token_length = 1
# insert 1 into lengths every time we see a new document
lengths_with_eos = np.insert(lengths, np.arange(1, len(lengths)), eos_token_length)
lengths_with_eos = lengths_with_eos.tolist()
print(lengths_with_eos[:10])

# Constants
context_length = 8192
estimated_number_of_chunks = sum(lengths_with_eos) // context_length

concatenated_chunks = []

def calculate_eos_tokens(lengths, context_length):
    current_length = 0
    curr_chunk = []
    eos_counts = []
    while lengths:
        doc_length = lengths.pop(0)
        if current_length + doc_length == context_length:
            curr_chunk.append(doc_length)
            concatenated_chunks.append(curr_chunk)
            eos_counts.append(curr_chunk.count(eos_token_length))
            current_length = 0
            curr_chunk = []
            if len(concatenated_chunks) % 1000 == 0:
                print("in total you created {} chunks".format(len(concatenated_chunks)),
                      "which is {:.2f}% of the estimated number of chunks".format(len(concatenated_chunks) / estimated_number_of_chunks * 100))
        elif current_length + doc_length > context_length:
            remaining_length = doc_length - (context_length - current_length)
            curr_chunk.append(context_length - current_length)
            lengths.insert(0, remaining_length)
            concatenated_chunks.append(curr_chunk)
            eos_counts.append(curr_chunk.count(eos_token_length))
            current_length = 0
            curr_chunk = []
            if len(concatenated_chunks) % 1000 == 0:
                print("in total you created {} chunks".format(len(concatenated_chunks)),
                      "which is {:.2f}% of the estimated number of chunks".format(len(concatenated_chunks) / estimated_number_of_chunks * 100))
        else:
            current_length += doc_length
            curr_chunk.append(doc_length)
    print("in total you created {} chunks".format(len(concatenated_chunks)))
    return eos_counts

eos_counts = calculate_eos_tokens(lengths_with_eos, context_length)

# Plotting
plt.figure(figsize=(10, 6))
plt.hist(eos_counts, bins=range(min(eos_counts), max(eos_counts) + 2), edgecolor='black', density=True)
plt.title('Distribution of EOS Tokens')
plt.xlabel('Number of EOS Tokens')
plt.ylabel('Frequency')
plt.show()
