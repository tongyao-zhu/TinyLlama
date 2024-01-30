import glob

import datasets
from transformers import pipeline, set_seed
from transformers.pipelines.pt_utils import KeyDataset
from datasets import Dataset, load_dataset
from tqdm.auto import tqdm
import argparse
import json
import os
import glob
from transformers import AutoTokenizer, LlamaForCausalLM, AutoModelForCausalLM

def read_jsonl(files_dir):
    """Read a JSON Lines file and return a list of documents."""
    documents = []
    for file_path in glob.glob(os.path.join(files_dir, '*.jsonl')):
        with open(file_path, 'r') as file:
            for line in file:
                document = json.loads(line)
                document['text'] = document['title'] # set to title (which is the query)
                # document['text'] = document['contents']
                # if len(document['contents'])>2000:
                #     document['contents'] = document['contents'][-2000:]
                # del document['contents']
                # documents.append(document)
    print("Finished reading jsonl files", len(documents))
    return documents

class PretrainDataset(Dataset):
    def __init__(self, files_dir):
        super().__init__()
        self.documents = read_jsonl(files_dir)

    def __len__(self):
        return len(self.documents)
    def __getitem__(self, i):
        return self.documents[i]

def remove_prefix(name):
    """
    Remove the prefix. For instance, "google/flan-t5-xl" should be come flan-t5-xl
    :param name:
    :return:
    """
    names =  name.split("/")
    while 'llama' not in  names[-1].lower():
        names = names[:-1]
    return names[-1]
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_dir", required=True)
    parser.add_argument("--chunk_num", required=False, type = str, default="*")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--batch_size", default=128, required=False, type = int)
    parser.add_argument("--save_dir", default = "/home/aiops/zhuty/ret_pretraining_data/redpajama_2b_id_added/generated_queries", required = False)
    parser.add_argument("--first_x", default = -1, type = int)
    return parser.parse_args()

def add_input_text(example, last_tokens=300):
    text = " ".join(example['title'].split()[-last_tokens:])
    if len(text) > 2000:
        print("Warning: text is too long", len(text))
        # avoid lengthy text
        text = text[-1500:]
    example['text'] = text
    return example


set_seed(42)
args = parse_args()

filename = os.path.join(args.save_dir, f"{remove_prefix(args.model_name)}_chunk_{args.chunk_num}_result.jsonl")
input_filename = os.path.join(args.save_dir, f"{remove_prefix(args.model_name)}_chunk_{args.chunk_num}_input.jsonl")

if os.path.exists(filename):
    print("File exists", filename)
    exit(0)
# dataset = datasets.load_dataset(args.dataset_name, split="validation")
# dataset = PretrainDataset(args.train_data_dir)
# Load the dataset
dataset = load_dataset('json', data_files={'train': args.train_data_dir + f'/chunk_{args.chunk_num}.jsonl'})['train']
# if a text is too long, only keep the last 1000 char
dataset = dataset.map(add_input_text, batched=False)

print("Loaded dataset", dataset)

if args.first_x>=0:
    dataset = dataset.select(range(0, args.first_x))

if 't5' in args.model_name:
    pipeline_name = 'text2text-generation'
    pipe = pipeline(pipeline_name, model=args.model_name, device = "cuda", max_new_tokens = 32)
else:
    pipeline_name = "text-generation"
    if "llama" in args.model_name.lower():
        model = LlamaForCausalLM.from_pretrained(args.model_name)
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # print number of parameters
    print("Number of parameters", model.num_parameters())
    model.eval()
    tokenizer.pad_token_id = model.config.eos_token_id
    tokenizer.padding_side = "left"

    pipe = pipeline('text-generation', model=model, device="cuda", tokenizer=tokenizer, return_full_text=False,max_new_tokens=96,
                    num_return_sequences=3, min_new_tokens=16,
                         do_sample=True, temperature=0.6, top_k=5000)



results = []
# KeyDataset (only *pt*) will simply return the item in the dict returned by the dataset item
# as we're not interested in the *target* part of the dataset. For sentence pair use KeyPairDataset
for out in tqdm(pipe(KeyDataset(dataset, "text"), batch_size=args.batch_size), total=len(dataset)):
    results.append(out)

# print(results[:10], dataset[:10])



# write to jsonl, each line is a json
with open(filename, 'w') as outfile:
    for entry in results:
        json.dump(entry, outfile)
        outfile.write('\n')
print("Saved to ", os.path.join(args.save_dir, filename))

# write to jsonl, each line is a json
with open(input_filename, 'w') as outfile:
    for entry in dataset:
        json.dump({'id': entry['id'], 'text': entry['text']}, outfile)
        outfile.write('\n')
print("Saved to ", os.path.join(args.save_dir, input_filename))
# Sample usage"
# python run_batch_inf.py --train_data_dir /home/aiops/zhuty/ret_pretraining_data/redpajama_2b_id_added/queries/ --model_name /home/aiops/zhuty/tinyllama/out/tinyllama_120M/tinyllama_120M_20b/ --batch_size 256 --save_dir /home/aiops/zhuty/ret_pretraining_data/redpajama_2b_id_added/generated_queries --chunk_num '0'