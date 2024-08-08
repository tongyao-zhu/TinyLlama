import glob
import math
import sys
import json
import time
from pathlib import Path
from typing import Optional, Tuple, Union
import math
import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy, XLAStrategy
from torch.utils.data import DataLoader
from functools import partial
import datetime
# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
# from apex.optimizers import FusedAdam #torch optimizer has a cuda backend, which is faster actually
from lit_gpt.model import GPT, Block, Config, CausalSelfAttention
from lit_gpt.packed_dataset import CombinedDataset, PackedDataset
from lit_gpt.speed_monitor import SpeedMonitorFabric as Monitor
from lit_gpt.speed_monitor import estimate_flops, measure_flops
from lit_gpt.utils import chunked_cross_entropy, get_default_supported_precision, num_parameters, step_csv_logger, lazy_load
from pytorch_lightning.loggers import WandbLogger
from lit_gpt import FusedCrossEntropyLoss
import random

import os
# model_name = "tiny_LLaMA_120M"
# name = "tinyllama_120M"
# TODO: What is a better way to pass the model name?
model_name = os.environ['MODEL_NAME']
save_name = os.environ['WANDB_NAME']
gpu_memory = os.environ['GPU_MEMORY']
out_dir = Path("out") / save_name

# Hyperparameters
num_of_devices = int(os.environ['NUMBER_OF_GPU'])
global_batch_size = 512
learning_rate = 4e-4
if "120M" in model_name:
    micro_batch_size = 32
elif '1b' in model_name:
    micro_batch_size = 16
elif '360M' in model_name:
    micro_batch_size = 32
else:
    raise ValueError("Invalid model name")
if '4k' in model_name:
    micro_batch_size = micro_batch_size // 2 # 4k tokens
    global_batch_size = global_batch_size // 2
elif '8k' in model_name:
    micro_batch_size = micro_batch_size // 4 # 8k tokens
    global_batch_size = global_batch_size // 4
elif '16k' in model_name:
    micro_batch_size = micro_batch_size // 8 # 8k tokens
    global_batch_size = global_batch_size // 8
if gpu_memory == '40960':
    micro_batch_size = micro_batch_size // 2

# for evaluation, we can increase the batch size
micro_batch_size = micro_batch_size * 4

max_step=100000

warmup_steps = 2000
log_step_interval = 10
eval_iters = 10000
save_step_interval = min(max_step//1, 2500)
eval_step_interval = 500

weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
min_lr = 4e-5

batch_size = global_batch_size // num_of_devices
gradient_accumulation_steps = batch_size // micro_batch_size
assert gradient_accumulation_steps > 0
warmup_iters = warmup_steps * gradient_accumulation_steps




max_iters = max_step * gradient_accumulation_steps
lr_decay_iters = max_iters
log_iter_interval = log_step_interval * gradient_accumulation_steps


# Treat all dataset equally by their size. If you want to use a different weight for a dataset, add it to the list with the weight.
train_data_config = [
    ("train", 1.0),
    # ("train_slim", 0.693584),
    # ("train_star", 0.306416),
]

val_data_config = [
    ("valid", 1.0),
]

hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}
logger = step_csv_logger("out", save_name, flush_logs_every_n_steps=log_iter_interval)
wandb_logger = WandbLogger()

def setup(
    num_devices: int = 1,
    train_data_dir: Path = Path("data/redpajama_sample"),
    val_data_dir: Optional[Path] = None,
    precision: Optional[str] = None,
    tpu: bool = False,
    resume: Union[bool, Path] = False,
    eval_only: bool = True,
    checkpoint: Optional[Path] = None,
    intradoc_mask: str = "",
    merge_method: str = "",
) -> None:
    precision = precision or get_default_supported_precision(training=True, tpu=tpu)
    print("devices", num_devices, "precision", precision, "resume", resume, "eval_only", eval_only)
    print("train_data_dir", train_data_dir, "val_data_dir", val_data_dir)
    if num_devices > 1:
        if tpu:
            # For multi-host TPU training, the device count for Fabric is limited to the count on a single host.
            num_devices = "auto"
            strategy = XLAStrategy(sync_module_states=False)
        else:
            strategy = FSDPStrategy(
                auto_wrap_policy={Block},
                activation_checkpointing_policy=None,
                state_dict_type="full",
                limit_all_gathers=True,
                cpu_offload=False,
                timeout=datetime.timedelta(seconds=7200),
            )
    else:
        strategy = "auto"
    fabric = L.Fabric(devices=num_devices, strategy=strategy, precision=precision, loggers=[logger, wandb_logger])
    fabric.print(hparams)
    #fabric.launch(main, train_data_dir, val_data_dir, resume)
    resume = out_dir / checkpoint if checkpoint else None

    main(fabric, val_data_dir,resume , eval_only,intradoc_mask,merge_method)

def main(fabric, val_data_dir, resume, eval_only, intradoc_mask, merge_method):
    assert os.path.exists(resume), f"Checkpoint {resume} does not exist"
    if "tiny_LLaMA_1b_8k_intramask" in model_name:
        config = Config.from_name("tiny_LLaMA_1b_8k_intramask")
    else:
        raise ValueError("Invalid model name")
    print("model_name", model_name)

    val_dataloader = create_dataloaders(
        batch_size=micro_batch_size,
        block_size=config.block_size,
        fabric=fabric,
        val_data_dir=val_data_dir,
        seed=3407,
        mask_attn=intradoc_mask,
        merge_method=merge_method,
    )
    if val_dataloader is not None:
        val_dataloader = fabric.setup_dataloaders(val_dataloader)

    fabric.seed_everything(3407)  # same seed for every process to init model (FSDP)

    fabric.print(f"Loading model with {config.__dict__}")
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=False):
        model = GPT(config)
        model.apply(partial(model._init_weights ,n_layer=config.n_layer))
 

    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters {num_parameters(model):,}")

    model = fabric.setup(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2), foreach=False
    )
    # optimizer = FusedAdam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2),adam_w_mode=True)
    # optimizer = fabric.setup_optimizers(optimizer)

    state = {"model": model, "optimizer": optimizer, "hparams": hparams, "iter_num": 0, "step_count": 0}

    if resume is True:
        resume = sorted(out_dir.glob("*.pth"))[-1]
    elif not resume and len(list(out_dir.glob("*.pth"))) > 0:
        fabric.print(f"Found existing checkpoints in {out_dir}. Resuming from the latest one.")
        resume = sorted(out_dir.glob("*.pth"))[-1]
    if resume :
        fabric.print(f"Resuming training from {resume}")
        fabric.load(resume, state, strict=False)

    # train_time = time.perf_counter()
    fabric.print("Running evaluation only mode?", eval_only)
    eval_time= time.perf_counter()

    eval(fabric, state, val_dataloader, )

    fabric.print(f"Evaluation time: {(time.perf_counter()-eval_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


def eval(fabric, state, val_dataloader,):
    model = state["model"]

    if val_dataloader is not None:
        loss = validate(fabric, model, val_dataloader)  # sanity check
        fabric.print(f"Validation loss: {loss:.4f}, PPL: {math.exp(loss):.4f}")
        if os.getenv("LOG_FILE"):
            with open(os.getenv("LOG_FILE"), "w") as f:
                json.dump({"val_loss": loss.item(), "val_ppl": math.exp(loss.item())}, f)
            print("Saved val_loss and val_ppl to {}".format(os.getenv("LOG_FILE")))
        return

@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, val_dataloader: DataLoader) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()

    losses = torch.zeros(eval_iters, device=fabric.device)
    for k, val_data in enumerate(val_dataloader):
        if k >= eval_iters:
            break
        # print("keys", val_data.keys())
        input_ids = val_data['idx'][:, 0 : model.config.block_size].contiguous()
        if "fragment_lens" in val_data:
            print("using fragment_lens and fragment_nums for validation.")
            fragment_lens = val_data["fragment_lens"]
            print("fragment_lens", fragment_lens[0][:10])
            fragment_nums = val_data["fragment_nums"]
            logits = model(input_ids, fragment_lens=fragment_lens, fragment_nums=fragment_nums, force_use_masking=True)
        else:
            logits = model(input_ids)
        targets = val_data['idx'][:, 1 : model.config.block_size + 1].contiguous()
        loss = chunked_cross_entropy(logits, targets, chunk_size=0)

        # loss_func = FusedCrossEntropyLoss()
        # loss = loss_func(logits, targets)
        losses[k] = loss.item()

    # skip the entries with zero
    losses = losses[losses != 0]
    out = losses.mean()

    model.train()
    return out


def create_dataloader(
    batch_size: int, block_size: int, data_dir: Path, fabric, shuffle: bool = True, seed: int = 12345, split="train", mask_attn="", merge_method="no",
) -> DataLoader:
    datasets = []
    data_config = train_data_config if split == "train" else val_data_config
    print("Creating a dataloader, mask_attn:", mask_attn, "merge_method:", merge_method )

    for prefix, _ in data_config:
        filenames = sorted(glob.glob(str(data_dir / f"{prefix}*")))
        print("Found {} files for {}".format(len(filenames), prefix))
        random.seed(seed)
        random.shuffle(filenames)
        dataset = PackedDataset(
            filenames,
            # n_chunks control the buffer size. 
            # Note that the buffer size also impacts the random shuffle
            # (PackedDataset is an IterableDataset. So the shuffle is done by prefetch a buffer and shuffle the buffer)
            n_chunks=512 if split == "train" else 2,
            block_size=block_size,
            shuffle=shuffle,
            seed=seed+fabric.global_rank,
            num_processes=fabric.world_size,
            process_rank=fabric.global_rank,
            mask_attn=mask_attn,
            merge_method=merge_method,
            wrap=False
        )
        datasets.append(dataset)

    if not datasets:
        raise RuntimeError(
            f"No data found at {data_dir}. Make sure you ran prepare_redpajama.py to create the dataset."
        )

    weights = [weight for _, weight in data_config]
    sum_weights = sum(weights)
    weights = [el / sum_weights for el in weights]

    combined_dataset = CombinedDataset(datasets=datasets, seed=seed, weights=weights)

    def collate_fn_with_intradoc_mask(examples: dict, max_num_fragments_in_chunk=512):
        #print(examples[0].keys())
        # a = ([example["idx"] for example in examples])
        #print(len(a), a[0].shape, type(a[0]))
        input_ids = torch.LongTensor(torch.stack([example["idx"] for example in examples]))
        #print("input_ids", input_ids.shape, input_ids.dtype)
        # if "labels" not in examples[0]:
        #     labels = input_ids
        # else:
        #     labels = torch.LongTensor([example["labels"] for example in examples])
        batch_inputs = {"idx": input_ids}
        if "fragment_lens" in examples[0]:
            fragment_lens = [
                torch.tensor(item["fragment_lens"] + (max_num_fragments_in_chunk - len(item["fragment_lens"])) * [-1])
                for item in examples
            ]
            batch_inputs["fragment_lens"] = torch.stack(fragment_lens)
            fragment_nums = torch.tensor([item["fragment_nums"] for item in examples], dtype=torch.int32)
            batch_inputs["fragment_nums"] = fragment_nums
        return batch_inputs

    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, collate_fn=collate_fn_with_intradoc_mask)


def create_dataloaders(
    batch_size: int,
    block_size: int,
    fabric,
    val_data_dir: Optional[Path] = None,
    seed: int = 12345,
    mask_attn="",
    merge_method="none",
) -> Tuple[DataLoader, DataLoader]:
    # Increase by one because we need the next word as well
    effective_block_size = block_size + 1
    val_dataloader = (
        create_dataloader(
            batch_size=batch_size,
            block_size=effective_block_size,
            fabric=fabric,
            data_dir=val_data_dir,
            shuffle=False,
            seed=seed,
            split="validation",
            mask_attn=mask_attn,
            merge_method=merge_method,
        )
        if val_data_dir
        else None
    )
    return val_dataloader


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(setup)
