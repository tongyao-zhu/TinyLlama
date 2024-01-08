General process to process a new corpus:

We describe the steps to process a new corpus here. 

Suppose that the corpus is named:
dataset_name

### Step 1: Download the corpus
The corpus should have name `dataset_name`, and it should contain two subfolders: `train` and `valid`. 
The `train` folder contains the training documents, and the `valid` folder contains the validation documents.
Each folder contains `chunk_0.jsonl`, `chunk_1.jsonl`, etc. Each chunk is a jsonl file, where each line is a json object.

### Step 2: Preprocess the corpus using tinyllama's processing script
Create a folder named `dataset_name_sample_processed` and run the following command:
```
bash scripts/pajama_processing.sh dataset_name
```
The script is located at `scripts/pajama_processing.sh`. 

### Step 3: Train the baseline
You can now train the baseline model using the following command:
```
bash scripts/pretraining.sh model_name dataset_name
```

### Step 4: Perform search for the training set, for linking documents together
#### Step 4.1: Create the queries
First, we need to add ids to the documents in the training set. The ids are usually not provided in the original corpus.
We can add ids to the documents using the following command:
```
python processing/process_jsonl_files.py --dataset_name dataset_name --mode jsonl
```
This will create a new folder named `dataset_name_id_added` that contains the same documents as the original corpus, but with ids added.

Next, we need to create the queries. We can create the queries using the following command:
```
python processing/create_queries.py --dataset_name dataset_name --mode fake_title
```
It's called `fake_title` because pyserini only supports the `title` field for queries, so we put our queries inside.
Note that we only use the last 200 space-splitted tokens for queries. We also include length truncation to ensure that it's not too long.

#### Step 4.2: Index the documents
We can now index the documents using the following command:
```
bash processing/build_bm25_index.sh dataset_name
```
Please make a folder 'bm25_index' under the `dataset_name_id_added` folder.

#### Step 4.3: Search for the top 100 documents for each query
We can now search for the top 100 documents for each query using the following command:
```
bash processing/search_bm25_general.sh dataset_name chunk_num 
```
where `chunk_num` is the number of chunks in the corpus.
Alternatively, you can run the following command to search for chunks within a range:
```
bash processing/range_search_bm25.sh dataset_name chunk_start chunk_end
```
The results should be saved in the `dataset_name_id_added/bm25_search_results` folder.

### Step 5: Create graph, traverse and create a new corpus for training
#### Step 5.1: Create the graph adjacency list
After obtaining the search result. We can now create the graph adjacency list using the following command:
```
python processing/graphs/analyze_search_path.py --dataset_name dataset_name 
```

#### Step 5.2: Graph traversal
We can now perform graph traversal using the following command:
```
bash processing/graphs/run_traversal.sh dataset_name
```
for undirected, we can run
```
bash processing/graphs/run_traversal_undirected.sh dataset_name
```
The traversed path is saved in the `dataset_name_id_added/traverse_paths` folder. To analyze them, you can look at the notebook
`processing/graphs/analyze_traverse_paths.ipynb`.

#### Step 5.3: Create the new corpus
next, we can create the new corpus using the following command:
```
bash processing/reorder_and_process.sh version_name 
```
where `version_name` is the name of the new corpus. The `version_name` should be defined properly in the `processing/configs/version_to_path.json` file.
A new dataset will have the same structure as the original, except that the documents now has id and are ordered based on the path.

You can rerun Step 2 for this new corpus, and then rerun Step 3 to train the baseline model on this new corpus. 
Then you can compare the performance of the baseline model on the original corpus and the new corpus.
