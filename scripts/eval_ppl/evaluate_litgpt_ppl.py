from datasets import load_dataset
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import torch
import json
import os
import argparse
import subprocess
from lit_gpt.model import GPT, Block, Config, CausalSelfAttention
from lit_gpt.tokenizer import Tokenizer
from pathlib import Path
import pandas as pd

TOKENIZER_PATH = '/home/aiops/zhuty/tinyllama/models'
MODEL_CONFIG = "tiny_LLaMA_1b_8k"
MODEL_CONFIG = "tiny_LLaMA_1b_16k"
# MODEL_CONFIG = "tiny_LLaMA_360M_8k"


# tinycoder_1M
# tinycoder_1_1b
# MODEL_CONFIG = "tinycoder_1_1b"

def load_lit_gpt_model(mode_path, tokenizer_path=TOKENIZER_PATH):
    config = Config.from_name(MODEL_CONFIG)
    model = GPT(config)
    # resume the state
    state_dict = torch.load(mode_path, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]
    model.load_state_dict(state_dict, strict=True)
    # model's precision is float32
    tokeinzer = Tokenizer(Path(tokenizer_path))
    # model to cuda
    model = model.to(torch.bfloat16)
    model = model.cuda()
    return model, tokeinzer

def load_hf_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path,  attn_implementation="flash_attention_2", torch_dtype=torch.float16, device_map="cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_path,truncation_side='right')
    #model = model.to(torch.bfloat16)
    #model = model.cuda()
    model.eval()
    return model, tokenizer

def get_hash(example):
    """Get hash of content field."""
    return {"hash": hash(example["text"])}  # can use any hashing function here


def check_uniques(example, uniques):
    """Check if current hash is still in set of unique hashes and remove if true."""
    if example["hash"] in uniques:
        uniques.remove(example["hash"])
        return True
    else:
        return False


def check_gpu_memory():
    try:
        # 执行 nvidia-smi 命令并捕获输出
        result = subprocess.check_output("nvidia-smi", shell=True, text=True)

        # 在输出中查找显存信息
        lines = result.split('\n')
        for line in lines:
            if 'Memory' in line and 'MiB / 80' in line:
                return True  # 显卡显存为80GB
    except subprocess.CalledProcessError:
        # 处理命令执行错误
        print("Error executing nvidia-smi command.")

    return False  # 未找到匹配的显卡信息


is_80gb_gpu = check_gpu_memory()

def tokenize(example, tokenizer, model_type):
    if model_type == 'hf':
        inputs = tokenizer.encode(example["text"], add_special_tokens=False) # full text
        inputs = torch.tensor(inputs, dtype=torch.long,) #  device="cuda")
    else:
        inputs = tokenizer.encode(example["text"], ) # device="cuda")
    return {'input_id': inputs}

def split_and_label(dataset, max_length=2048):
    new_input_ids = []
    new_ids = []
    new_labels = []
    print('Using sliding window the create multiple context windows', max_length)

    for input_id, id in zip(dataset['input_id'], dataset['id']):
        all_labels = input_id[1:] + [-100]
        assert len(input_id) == len(all_labels)
        chunks = [input_id[i:i + max_length] for i in range(0, len(input_id), max_length)]
        labels = [all_labels[i:i + max_length] for i in range(0, len(all_labels), max_length)]
        # shift right, then add -100
        for i, (chunk, label) in enumerate(zip(chunks, labels)):
            new_input_ids.append(chunk)
            new_ids.append(f"{id}_{i}")
            new_labels.append(label)

    new_dataset = datasets.Dataset.from_dict({"input_id": new_input_ids, "id": new_ids, "label": new_labels})
    return new_dataset
def process_truncate(example, max_length):
    # truncate the text
    example["input_id"] = example["input_id"][:max_length]
    labels = example["input_id"][1:] + [-100]
    example['label'] = labels[:max_length]
    return example
def process_batch(examples,):
    # batch = dataset[i:i + batch_size]
    # real_batch_size = len(batch["text"])
    # tokenize the batch
    # print("Batch size: ", len(examples))
    batch_size=len(examples)
    # print("0-th example: ", examples[0]['input_id'][:10])
    batch_inputs = [example['input_id'] for example in examples]
    labels = [example['label'] for example in examples]
    # print("Batch input size: ", len(batch_inputs))
    # padding the batch
    batch_max_len = max([len(inputs) for inputs in batch_inputs])
    input_ids = torch.zeros(batch_size, batch_max_len, dtype=torch.long, device="cuda")
    length_mask = torch.zeros(batch_size, batch_max_len, dtype=torch.float, device="cuda")
    shift_labels = torch.zeros(batch_size, batch_max_len, dtype=torch.long, device="cuda")
    for j in range(batch_size):
        input_ids[j, :len(batch_inputs[j])] = torch.tensor(batch_inputs[j])
        length_mask[j, :len(batch_inputs[j])] = 1
        shift_labels[j, :len(labels[j])] = torch.tensor(labels[j])
    return {'input_id': input_ids, 'length_mask': length_mask, 'shift_labels': shift_labels}

def label_model_loss(file_path, model_path,
                     verbose=False,
                     save_file_name=None,
                     average_by_token=False,
                     max_length=8192,
                     batch_size=16,
                     model_type='litgpt',
                     sliding_window=-1,
                     first_n=-1):
    if verbose:
        print("Record example-level loss for the model.")
        #assert save_folder is not None
        #if not os.path.exists(save_folder):
            #os.makedirs(save_folder)
    if model_type == 'hf':
        model, tokenizer = load_hf_model(model_path)

    elif model_type == 'litgpt':
        model, tokenizer = load_lit_gpt_model(model_path)
        tokenizer.bos_id = tokenizer.eos_id
    else:
        raise ValueError("Invalid model type: ", model_type)
    dataset = load_dataset("json",
                           data_dir=file_path,
                           split="train")
    if first_n > 0:
        dataset = dataset.select(range(min(first_n, len(dataset))))

    dataset = dataset.map(tokenize, fn_kwargs={"tokenizer": tokenizer, "model_type": model_type})
    dataset = dataset.add_column('id', range(len(dataset)))    # assign each row an id

    if sliding_window > 0:
        dataset = split_and_label(dataset, sliding_window)
        print("New dataset size: ", len(dataset))
    else:
        # just truncate to max length
        dataset = dataset.map(process_truncate, fn_kwargs={"max_length": max_length})
    print("Dataset size: ", len(dataset))
    # print("0-th data: ", dataset[0])
    # batched_dataset = dataset.map(process_batch, batched=True, batch_size=batch_size)
    print("batch size", batch_size)
    # dataset = dataset.select(range(0, 100))
    # preprocess and tokenize the datase
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    loss_value = []
    lengths_list = []
    id_list = []
    with torch.no_grad():
        # batched dataset
        for i in tqdm(range(0, len(dataset), batch_size)):
            ds_batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
            batch = process_batch(ds_batch)
            curr_batch_size=batch['input_id'].size(0)
            input_ids = batch['input_id'].cuda()
            length_mask = batch['length_mask'].cuda()
            labels = batch['shift_labels'].cuda()
            # forward
            if model_type == 'hf':
                logits = model(input_ids).logits
            else:
                logits = model(input_ids)
            # cacluate loss
            # shift_logits = logits[..., :-1, :].contiguous()
            shift_logits = logits[..., :, :].contiguous()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            # shift_labels = input_ids[..., 1:].contiguous().view(-1)
            shift_labels = labels[..., :].contiguous().view(-1)

            loss = loss_fn(shift_logits, shift_labels)
            # reshape as [batch size x seq length]
            loss = loss.view(curr_batch_size, -1)
            # loss = loss * length_mask[..., :-1]
            loss = loss * length_mask
            # average over the sequence length
            loss_list = []
            length_list = []
            for i in range(curr_batch_size):
                loss_single = loss[i].sum() / length_mask[i].sum()
                if average_by_token:
                    # add the per-token loss on the sequence to the list
                    valid_part = int(length_mask[i].sum().item())
                    temp_list = loss[i].tolist()
                    temp_list = temp_list[:valid_part]
                    loss_list.extend(temp_list)
                else:
                    loss_list.append(loss_single.item())
                length_list.append(int(length_mask[i].sum().item()))
            loss_value.extend(loss_list)
            lengths_list.extend(length_list)
            id_list.extend(ds_batch['id'])
    # print("Min length: ", min_char_len, " Average loss: ", sum(loss_value) / len(loss_value))
    if verbose:
        #with open(save_file_name, "w") as f:
            # save the loss value line by line
            #json.dump(list(zip(lengths_list, loss_value)), f)
        # save as csv
        df = pd.DataFrame({"id": id_list, "length": lengths_list, "loss": loss_value})
        with open(save_file_name, "w") as f:
            df.to_csv(f)

    return sum(loss_value) / len(loss_value)

def parse_args():
    parser= argparse.ArgumentParser(description='Evaluate PPL')
    parser.add_argument('--model', type=str, required=True, help='model path')
    parser.add_argument('--model_type', default='litgpt', required=False, help='model type')
    parser.add_argument('--dataset', type=str, required=True, help='dataset path')
    parser.add_argument('--chunk_n', type=int, required=True, help='chunk number')
    parser.add_argument('--sliding_window', type=int, required=False, default=-1, help='chunk number')
#     parser.add_argument('--output', type=str, required=True, help='output path')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # MODEL_NAME = "tinyllama_1M_model_large_n"
    # all_models = os.listdir("/home/aiops/liuqian/TinyLlama-Data/checkpoints")
    # use prefix to filter
    # all_models = [model for model in all_models if model.startswith(MODEL_NAME)]
    # sort the models
    # all_models.sort()
    # write_f = open("tinyllama1M_large_c4100.tsv", "w")
    # save_folder_name = "tinyllama1M_large_c4100"
    # for model_name in all_models:
        #model_path = f"/home/aiops/liuqian/TinyLlama-Data/checkpoints/{model_name}/iter-001000-ckpt.pth"
        #if not os.path.exists(model_path):
            # continue
        #print(f"Model: {model_name}")
    model_path = args.model
    dataset_path = args.dataset
    model_name = model_path.split("/")[-2]
    # remove the pth
    model_ckpt = model_path.split("/")[-1].split(".")[0]
    ds_name = dataset_path.split("/")[-3]
    assert ds_name in ['cc', 'book', 'arxiv' , 'rpwiki_en'], "Dataset " + ds_name + " is not supported."
    print("Model: ", model_name, " Dataset: ", ds_name, " Chunk: ", args.chunk_n, " Model ckpt: ", model_ckpt, " Sliding window: ", args.sliding_window)
    save_file_name = os.path.join("/home/aiops/zhuty/tinyllama/scripts/eval_ppl/results",
                                  f"{model_name}_chunk_{args.chunk_n}_{ds_name}_{model_ckpt}.csv")
    if args.sliding_window > 0:
        save_file_name = save_file_name.replace(".csv", f"_sw_{args.sliding_window}.csv")
    if args.first_n > 0:
        save_file_name = save_file_name.replace(".csv", f"_first_{args.first_n}.csv")

    if os.path.exists(save_file_name):
        print("Skip model: ", model_path, "because saved file exists.", "File: ", save_file_name)
        exit(0)

    max_length, batch_size = 8192, 8
    if '16k' in model_path:
        print("Extending the model's max length to 16k")
        max_length = 16384
        batch_size = 8
    print("Batch size: ", batch_size, )
    normal_loss = label_model_loss(dataset_path, model_path,
                                       verbose=True, save_file_name=save_file_name,
                                       average_by_token=False, max_length = max_length, batch_size=batch_size, model_type=args.model_type,
                                   sliding_window=args.sliding_window)

    print("Model: ", model_path, " Loss: ", normal_loss)
       # write the model name and loss into file
        #write_f.write(f"{model_name}\t{normal_loss}\n")
        #write_f.flush()
    # write_f.close()