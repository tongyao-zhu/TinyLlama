from datasets import load_dataset
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

TOKENIZER_PATH = '/home/aiops/zhuty/tinyllama/models'
# MODEL_CONFIG = "tiny_LLaMA_1b_8k"
MODEL_CONFIG = "tiny_LLaMA_360M_8k"


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


def label_model_loss(file_path, model_path,
                     verbose=False,
                     save_file_name=None,
                     average_by_token=False,
                     max_length=8192):
    if verbose:
        print("Record example-level loss for the model.")
        #assert save_folder is not None
        #if not os.path.exists(save_folder):
            #os.makedirs(save_folder)
    model, tokenizer = load_lit_gpt_model(model_path)
    tokenizer.bos_id = tokenizer.eos_id
    dataset = load_dataset("json",
                           data_dir=file_path,
                           split="train")
    # dataset = dataset.select(range(0, 100))
    # preprocess and tokenize the dataset
    batch_size = 16
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    loss_value = []
    lengths_list = []
    with torch.no_grad():
        # batched dataset
        i = 0
        for i in tqdm(range(0, len(dataset), batch_size)):
            batch = dataset[i:i + batch_size]
            real_batch_size = len(batch["text"])
            # tokenize the batch
            batch_inputs = []
            for j in range(real_batch_size):
                inputs_j = tokenizer.encode(batch["text"][j], max_length=max_length, device="cuda")
                batch_inputs.append(inputs_j)
            # padding the batch
            max_len = max([len(inputs) for inputs in batch_inputs])
            input_ids = torch.zeros(real_batch_size, max_len, dtype=torch.long, device="cuda")
            length_mask = torch.zeros(real_batch_size, max_len, dtype=torch.float, device="cuda")
            for j in range(real_batch_size):
                input_ids[j, :len(batch_inputs[j])] = batch_inputs[j]
                length_mask[j, :len(batch_inputs[j])] = 1
            # forward
            logits = model(input_ids)
            # cacluate loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = input_ids[..., 1:].contiguous().view(-1)
            loss = loss_fn(shift_logits, shift_labels)
            # reshape as [batch size x seq length]
            loss = loss.view(real_batch_size, -1)
            loss = loss * length_mask[..., :-1]
            # average over the sequence length
            loss_list = []
            length_list = []
            for i in range(real_batch_size):
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
    # print("Min length: ", min_char_len, " Average loss: ", sum(loss_value) / len(loss_value))
    if verbose:

        with open(save_file_name, "w") as f:
            # save the loss value line by line
            json.dump(list(zip(lengths_list, loss_value)), f)
    return sum(loss_value) / len(loss_value)

def parse_args():
    parser= argparse.ArgumentParser(description='Evaluate PPL')
    parser.add_argument('--model', type=str, required=True, help='model path')
    parser.add_argument('--dataset', type=str, required=True, help='dataset path')
    parser.add_argument('--chunk_n', type=int, required=True, help='chunk number')
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
    print("Model: ", model_name, " Dataset: ", ds_name, " Chunk: ", args.chunk_n, " Model ckpt: ", model_ckpt)
    save_file_name = os.path.join("/home/aiops/zhuty/tinyllama/scripts/eval_ppl/results",
                                  f"{model_name}_chunk_{args.chunk_n}_{ds_name}_{model_ckpt}.json")
    if os.path.exists(save_file_name):
        print("Skip model: ", model_path, "because saved file exists.", "File: ", save_file_name)
        exit(0)

    normal_loss = label_model_loss(dataset_path, model_path,
                                       verbose=True, save_file_name=save_file_name,
                                       average_by_token=False)

    print("Model: ", model_path, " Loss: ", normal_loss)
       # write the model name and loss into file
        #write_f.write(f"{model_name}\t{normal_loss}\n")
        #write_f.flush()
    # write_f.close()