from datasets import load_dataset
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
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
import numpy as np
from scripts.eval_ppl.packing.monkey_patch_packing import monkey_patch_packing_for_model

TOKENIZER_PATH = '/home/aiops/zhuty/tinyllama/models'
# MODEL_CONFIG = "tiny_LLaMA_1b_8k"
# MODEL_CONFIG = "tiny_LLaMA_1b_16k"
# MODEL_CONFIG = "tiny_LLaMA_360M_8k"
DEBUG=False

# tinycoder_1M
# tinycoder_1_1b
# MODEL_CONFIG = "tinycoder_1_1b"

def load_lit_gpt_model(mode_path, tokenizer_path=TOKENIZER_PATH):
    MODEL_CONFIG = model_path.split("/")[-2].split("_cc")[0]
    print("Model config: ", MODEL_CONFIG)
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

def load_hf_model(model_path, attention_type="flash_attention_2"):
    if '2ke8k' in model_path:
        original_path = model_path.replace("2ke8k", "2k")
        config = AutoConfig.from_pretrained(original_path,
                                            token=json.load(open('/home/aiops/zhuty/hf_token.json')))
        config.rope_scaling = {'type': 'dynamic', 'factor': 4}
        model_path = original_path

        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, attn_implementation="flash_attention_2", config=config, device_map="cuda",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, truncation_side='right', token=json.load(open('/home/aiops/zhuty/hf_token.json')))
    elif 'rb5' in model_path:
        config = AutoConfig.from_pretrained(model_path,
                                            token=json.load(open('/home/aiops/zhuty/hf_token.json')))
        print("Extending the rope theta to 100000")
        config.rope_theta = 100000
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, attn_implementation="flash_attention_2", config=config, device_map="cuda",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, truncation_side='right', token=json.load(open('/home/aiops/zhuty/hf_token.json')))
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path,  attn_implementation=attention_type,
                                                     torch_dtype=torch.float16, device_map="cuda",
                                                 token = json.load(open("/home/aiops/zhuty/hf_token.json")))
        tokenizer = AutoTokenizer.from_pretrained("tyzhu/tiny_LLaMA_120M_8k_cc_8k_iter-040000-ckpt-step-10000_hf",truncation_side='right',token = json.load(open("/home/aiops/zhuty/hf_token.json")))

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

def tokenize(example, tokenizer, model_type, text_key='text'):
    if model_type == 'hf':
        inputs = tokenizer.encode(example[text_key], add_special_tokens=False) # full text
        inputs = torch.tensor(inputs, dtype=torch.long,) #  device="cuda")
    else:
        inputs = tokenizer.encode(example[text_key], ) # device="cuda")
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

def add_fixed_length_attention_mask(example, region_length):
    labels = torch.tensor(example["label"])
    # Initialize the attention mask with zeros
    attention_mask = torch.zeros_like(labels)

    # Identify positions with labels not equal to -100
    valid_positions = (labels != -100).nonzero(as_tuple=True)[0]

    # Create a running index to segment the valid positions
    running_index = torch.arange(1, len(valid_positions) // region_length + 2)

    # Assign region values to the attention mask
    for i, start in enumerate(range(0, len(valid_positions), region_length)):
        end = min(start + region_length, len(valid_positions))
        attention_mask[valid_positions[start:end]] = running_index[i]

    example["attention_mask"] = attention_mask
    return example

def add_full_attention_mask(example):
    labels = torch.tensor(example["label"])
    # Initialize the attention mask with zeros
    attention_mask = torch.zeros_like(labels)

    # put 1 to all the valid positions
    valid_positions = (labels != -100).nonzero(as_tuple=True)[0]
    attention_mask[valid_positions] = 1
    example["attention_mask"] = attention_mask
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
    attention_masks = [example['attention_mask'] for example in examples]
    # print("Batch input size: ", len(batch_inputs))
    # padding the batch
    batch_max_len = max([len(inputs) for inputs in batch_inputs])
    input_ids = torch.zeros(batch_size, batch_max_len, dtype=torch.long, device="cuda")
    attention_mask = torch.zeros(batch_size, batch_max_len, dtype=torch.long, device="cuda")
    length_mask = torch.zeros(batch_size, batch_max_len, dtype=torch.long, device="cuda")
    shift_labels = torch.zeros(batch_size, batch_max_len, dtype=torch.long, device="cuda")
    for j in range(batch_size):
        input_ids[j, :len(batch_inputs[j])] = torch.tensor(batch_inputs[j])
        length_mask[j, :len(batch_inputs[j])] = 1
        attention_mask[j, :len(batch_inputs[j])] = torch.tensor(attention_masks[j])
        shift_labels[j, :len(labels[j])] = torch.tensor(labels[j])
    # print("Attention mask", attention_mask)
    return {'input_id': input_ids, 'length_mask': length_mask, 'shift_labels': shift_labels, "attention_mask": attention_mask}

def load_dataset_and_process(file_path, first_n, tokenizer, model_type, model, max_length, sliding_window, fixed_mask_length):
    save_name = ('/home/aiops/zhuty/saved_ppl_datasets/' + file_path.strip("/").split('/')[-2] + f"_first_{first_n}_max_{max_length}_sw_{sliding_window}_mask_{fixed_mask_length}.json")
    if os.path.exists(save_name):
        dataset = datasets.load_from_disk(save_name)
        print("Loaded dataset from ", save_name)
        return dataset

    dataset = load_dataset("json",
                           data_dir=file_path,
                           split="train")
    if first_n > 0:
        dataset = dataset.select(range(min(first_n, len(dataset))))

    dataset = dataset.map(tokenize, fn_kwargs={"tokenizer": tokenizer, "model_type": model_type}, )
    dataset = dataset.add_column('id', range(len(dataset)))    # assign each row an id

    if sliding_window > 0:
        dataset = split_and_label(dataset, sliding_window)
        print("New dataset size: ", len(dataset))
    else:
        # just truncate to max length
        dataset = dataset.map(process_truncate, fn_kwargs={"max_length": max_length})

    if fixed_mask_length > 0:
        print("Adding fixed length attention mask", fixed_mask_length)
        dataset = dataset.map(add_fixed_length_attention_mask, fn_kwargs={"region_length": fixed_mask_length})
        monkey_patch_packing_for_model(model)
    else:
        dataset = dataset.map(add_full_attention_mask)
    print("Saving dataset to ", save_name)
    dataset.save_to_disk(save_name)
    return dataset

def compute_attention_sink(attention_scores, epsilon):
    print("Compute attention sink scores on tensor, shape: ", attention_scores.shape)
    # attention_scores = np.load(score_path)
    num_samples, num_layers, num_heads, num_tokens1, num_tokens2 = attention_scores.shape
    assert num_tokens1 == num_tokens2
    # attention_scores = torch.from_numpy(attention_scores)
    ratios = torch.arange(num_tokens1, 0, -1)[None, None, None, :].expand(num_samples, num_layers, num_heads,
                                                                          num_tokens1, num_tokens2).to(attention_scores)
    importance_scores = (attention_scores / ratios).sum(dim=-2)  # (num_samples, num_layers, num_heads, num_tokens)
    return importance_scores
    # print(importance_scores.shape)
    # metric1 = (importance_scores > epsilon).to(torch.float).mean(dim=(0, 1, 2))
    # return metric1 * 100

def load_model(model_path, model_type, attention_type):
    if model_type == 'hf':
        model, tokenizer = load_hf_model(model_path, attention_type)
    elif model_type == 'litgpt':
        model, tokenizer = load_lit_gpt_model(model_path)
        tokenizer.bos_id = tokenizer.eos_id
    else:
        raise ValueError("Invalid model type: ", model_type)
    return model, tokenizer

def label_model_loss(file_path, model_path,
                     verbose=False,
                     save_file_name=None,
                     average_by_token=False,
                     max_length=8192,
                     batch_size=16,
                     model_type='litgpt',
                     sliding_window=-1,
                     first_n=-1,
                     fixed_mask_length=-1,
                     save_attention=False
                     ):
    if verbose:
        print("Record example-level loss for the model.", "Model: ", model_path, " Dataset: ", file_path, " Model type: ", model_type,
                " Sliding window: ", sliding_window, " Fixed mask length: ", fixed_mask_length, " First n: ", first_n,
                " Max length: ", max_length, " Batch size ", batch_size)
    if save_attention:
        attention_type ='eager'
        batch_size = batch_size // 2
    else:
        attention_type = 'flash_attention_2'

    model, tokenizer = load_model(model_path, model_type, attention_type)
    dataset = load_dataset_and_process(file_path, first_n, tokenizer, model_type, model, max_length, sliding_window, fixed_mask_length)
    print("Dataset size: ", len(dataset))
    if save_attention:
        # filter out the too short examples
        if max_length > 0:
            dataset = dataset.filter(lambda x: len(x['input_id']) >= max_length)
            print(f"After filtering out {max_length} examples, the dataset size is: ", len(dataset))
        elif sliding_window > 0:
            # print(dataset[0])
            dataset = dataset.filter(lambda x: len(x['input_id']) >= sliding_window)
            print(f"After filtering out <{sliding_window} examples, the dataset size is: ", len(dataset))

    # print("0-th data: ", dataset[0])
    # batched_dataset = dataset.map(process_batch, batched=True, batch_size=batch_size)
    print("batch size", batch_size)
    # dataset = dataset.select(range(0, 100))
    # preprocess and tokenize the datase
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    loss_value = []
    lengths_list = []
    id_list = []
    attention_scores_list = []
    # logits_list = []
    with torch.no_grad():
        # batched dataset
        for i in tqdm(range(0, len(dataset), batch_size)):
            ds_batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
            batch = process_batch(ds_batch)
            curr_batch_size=batch['input_id'].size(0)
            input_ids = batch['input_id'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            length_mask = batch['length_mask'].cuda()
            labels = batch['shift_labels'].cuda()
            # forward
            if model_type == 'hf':
                outputs = model(input_ids, attention_mask=attention_mask, output_attentions=save_attention)
                logits = outputs.logits
                if save_attention:
                    # print("saving attention?", save_attention)
                    attentions = outputs.attentions
                    # cpu_attentions = []
                    # for attention in attentions:
                    #     cpu_attentions.append(attention.cpu())  # Move each attention tensor to CPU immediately
                    # Now stack them on the CPU to avoid GPU memory issues
                    attentions = torch.stack(attentions, dim=1)
                    attention_scores_batch = compute_attention_sink(attentions.cpu(), 0.3)
                    attention_scores_list.append(attention_scores_batch)  # Collect the scores
            else:
                logits = model(input_ids,)
            if DEBUG:
                for key in ['input_id', 'attention_mask', 'length_mask', 'shift_labels']:
                    # json.dump(batch[key].tolist(), open(f"{save_file_name.replace('csv', '')}_{key}_batch{i}.npy", "w"))
                    np.save(f"{save_file_name.replace('csv', '')}_{key}_batch{i}.npy", batch[key].cpu().numpy())
                np.save(f"{save_file_name.replace('csv', '')}_logits_batch{i}.npy", logits.cpu().numpy())
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
        df = pd.DataFrame({"id": id_list, "length": lengths_list, "loss": loss_value, })
        with open(save_file_name, "w") as f:
            df.to_csv(f)
        if save_attention:
            with open(save_file_name.replace(".csv", "_attention_scores.npy"), "wb") as f:
                attention_scores = torch.cat(attention_scores_list, dim=0)
                print("Attention scores shape: ", attention_scores.shape)
                print("Saving attention scores to ", save_file_name.replace(".csv", "_attention_scores.npy"))
                np.save(f, attention_scores.cpu().numpy())  # Save the attention sink scores directly

    return sum(loss_value) / len(loss_value)

def parse_args():
    parser= argparse.ArgumentParser(description='Evaluate PPL', allow_abbrev=False)
    parser.add_argument('--model', type=str, required=True, help='model path')
    parser.add_argument('--model_type', default='litgpt', required=False, help='model type')
    parser.add_argument('--dataset', type=str, required=True, help='dataset path')
    parser.add_argument('--chunk_n', type=int, required=True, help='chunk number')
    parser.add_argument('--sliding_window', type=int, required=False, default=-1, help='sliding window size')
    parser.add_argument('--max_seq_length', type=int, required=False, default=-1, help='maximum length of truncation')
    parser.add_argument("--fixed_mask", type=int, default=-1, help="Fixed intradoc mask length")
    parser.add_argument("--first_n", type=int, default=-1, help="First n samples")
    parser.add_argument("--save_attention",  action='store_true', help="Save attention")
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
    if ".pth" in model_path:
        model_name = model_path.split("/")[-2]
        # remove the pth
        model_ckpt = model_path.split("/")[-1].split(".")[0]
    else:
        model_name = model_path.split("/")[-1]
        model_ckpt = ""
    ds_name = dataset_path.split("/")[-3]
    # assert ds_name in ['cc', 'book', 'arxiv' , 'rpwiki_en', "c4", "cc_small"], "Dataset " + ds_name + " is not supported."
    print("Model: ", model_name, " Dataset: ", ds_name, " Chunk: ", args.chunk_n, " Model ckpt: ", model_ckpt, " Sliding window: ", args.sliding_window, " Fixed mask: ", args.fixed_mask, " First n: ", args.first_n, " Max seq length: ", args.max_seq_length)
    save_file_name = os.path.join("/home/aiops/zhuty/tinyllama/scripts/eval_ppl/results",
                                  f"{model_name}_chunk_{args.chunk_n}_{ds_name}_{model_ckpt}_max_{args.max_seq_length}.csv")
    if args.sliding_window > 0:
        save_file_name = save_file_name.replace(".csv", f"_sw_{args.sliding_window}.csv")
    if args.first_n > 0:
        save_file_name = save_file_name.replace(".csv", f"_first_{args.first_n}.csv")
    if args.fixed_mask > 0:
        save_file_name = save_file_name.replace(".csv", f"_mask_{args.fixed_mask}.csv")
    if args.save_attention:
        save_file_name = save_file_name.replace(".csv", f"_saved_attention.csv")

    if os.path.exists(save_file_name):
        print("Skip model: ", model_path, "because saved file exists.", "File: ", save_file_name)
        exit(0)

    max_length, batch_size = args.max_seq_length, int(8192/args.max_seq_length) * 8
    # if '16k' in model_path:
    #     # print("Extending the model's max length to 16k")
    #     # max_length = 16384
    #     batch_size = 8
    if args.sliding_window > 0:
        batch_size = int(8192/args.sliding_window) * 8
    if '7b' in model_path:
        batch_size = batch_size // 8

    print("Batch size: ", batch_size, )
    normal_loss = label_model_loss(dataset_path, model_path,
                                       verbose=True, save_file_name=save_file_name,
                                       average_by_token=False, max_length = max_length, batch_size=batch_size, model_type=args.model_type,
                                   sliding_window=args.sliding_window, fixed_mask_length=args.fixed_mask, first_n=args.first_n,
                                   save_attention=args.save_attention)
    print("Saved result to", save_file_name)
    print("Model: ", model_path, " Loss: ", normal_loss)
       # write the model name and loss into file
        #write_f.write(f"{model_name}\t{normal_loss}\n")
        #write_f.flush()
    # write_f.close()