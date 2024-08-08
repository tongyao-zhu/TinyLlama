#!/bin/bash -l

set -e
set -u

model=$1
task=$2
seed=$3
nshot=$4

# adjust batch size based on model size
if [[ $model == *"120M"* ]]; then
  batch_size=2
elif [[ $model == *"360M"* ]]; then
  batch_size=1
else
  batch_size=1
fi

if [[ $model == *"8k"* ]]; then
  max_len=8192
elif [[ $model == *"16k"* ]]; then
  max_len=16384
else
  max_len=8192
fi

# if task is nq or tq, make batch size to be 8
if [[ $2 == "nq" ]] || [[ $2 == "tq" ]]; then
  batch_size=16
fi

if [[ $2 == "squad" ]]; then
  batch_size=4
elif [[ $2 == "hotpotqa" ]]; then
  batch_size=1
elif [[ $2 == "tq_obqa" ]]; then
  batch_size=1
elif [[ $2 == "agnews" ]]; then
  batch_size=2
elif [[ $2 == "dbpedia" ]]; then
  batch_size=1
elif [[ $2 == "memtrap" ]]; then
  batch_size=16
elif [[ $2 == "sst2" ]]; then
  batch_size=32
elif [[ $2 == *amazon* ]]; then
  # if shot number is 20, set batch to 4
  if [[ $4 == 20 ]]; then
    batch_size=8
  elif [[ $4 == 10 ]]; then
    batch_size=32
  else
    batch_size=1
  fi
fi
export GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0)
# if GPU memory is 81920, double the batch size
if [[ $GPU_MEMORY == 81920 ]]; then
  batch_size=$((batch_size*2))
fi
# if model name is 7b, set batch size to 1 anyways
if [[ $model == *7b* ]]; then
  batch_size=1
fi

echo "Batch size" $batch_size

echo "model: $model"
# for task in  "squad" "nq_obqa" "tq_obqa" "hotpotqa" ; do
# read $5 as num_retrieved_docs, if not provided, set it to 0
num_retrieved_docs=${5:-0}
# read $6 as flipped_ratio, if not provided, set it to 0
flipped_ratio=${6:-0}
use_flash_attn=${7:-0}
if [[ $use_flash_attn == "true" ]]; then
  flash_attn_2="--flash_attn_2"
else
  flash_attn_2=""
fi

echo "do we use flash attn 2" $flash_attn_2


echo "seed: $seed" "nshot: $nshot" "num_retrieved_docs: $num_retrieved_docs" "flipped_ratio: $flipped_ratio"
echo "Evaluating $task..."
python fewshot.py \
      --model_path="${model}" \
      --task="${task}" \
      --n_shot=$nshot \
      --seed="${seed}" \
      --device=0 \
      --batch_size=$batch_size \
      --downsample \
      --num_retrieved_docs=$num_retrieved_docs \
      --flipped_ratio=$flipped_ratio \
      --max_length=$max_len \
      $flash_attn_2