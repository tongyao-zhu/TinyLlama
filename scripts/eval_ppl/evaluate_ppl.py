from transformers import LlamaForCausalLM, LlamaTokenizer
import argparse
import evaluate
import json
from transformers import AutoTokenizer
from tqdm import tqdm

def read_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def split_into_batches(input_texts, batch_size):
    batches = []
    for i in range(0, len(input_texts), batch_size):
        batches.append(input_texts[i:i+batch_size])
    return batches

def get_lengths(texts, tokenizer):
    return [len(tokenizer.encode(text)) for text in texts]

def save_results(results, output_path):
    with open(output_path, 'w') as f:
        f.write(json.dumps(results))
def main():
    args = parse_args()
    # model = LlamaForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # tokenizer = LlamaTokenizer.from_pretrained('/home/aiops/zhuty/tinyllama/models' , padding_side='left', truncation_side="left")
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.push_to_hub(args.model, token = json.load(open("/home/aiops/zhuty/hf_token.json")))
    # tokenizer.pad_token = tokenizer.eos_token
    perplexity = evaluate.load("perplexity", module_type="metric")
    input_texts = read_jsonl(args.dataset)
    input_texts = [text['text'] for text in input_texts]
    lengths = get_lengths(input_texts, tokenizer)
    ppls = []
    # for batch in tqdm(input_texts):
    results = perplexity.compute(model_id=args.model,
                                 add_start_token=False,
                                 predictions=input_texts,
                                 batch_size = 2,
                                 max_length=8192)
    ppls.extend(results['perplexities'])
        # print(list(results.keys()))
        # print(results)
    print("Average PPL: ", sum(ppls)/len(ppls))
    print("Std PPL: ", sum([(ppl - sum(ppls)/len(ppls))**2 for ppl in ppls])/len(ppls))
    length_and_ppl = list(zip(lengths, ppls))

    return
    # model.eval()

def parse_args():
    parser= argparse.ArgumentParser(description='Evaluate PPL')
    parser.add_argument('--model', type=str, required=True, help='model path')
    parser.add_argument('--dataset', type=str, required=True, help='dataset path')
    parser.add_argument('--output', type=str, required=True, help='output path')
    return parser.parse_args()

if __name__ == '__main__':
    main()