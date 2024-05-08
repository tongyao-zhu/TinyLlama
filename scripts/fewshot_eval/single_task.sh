#!/bin/bash -l

set -e
set -u

model=$1

# adjust batch size based on model size
if [[ $model == *"120M"* ]]; then
  batch_size=2
elif [[ $model == *"360M"* ]]; then
  batch_size=1
else
  batch_size=1
fi

# if task is nq or tq, make batch size to be 8
if [[ $2 == "nq" ]] || [[ $2 == "tq" ]]; then
  batch_size=16
fi

if [[ $2 == "squad" ]]; then
  batch_size=16
elif [[ $2 == "hotpotqa" ]]; then
  batch_size=2
elif [[ $2 == "tq_obqa" ]]; then
  batch_size=2
elif [[ $2 == "agnews" ]]; then
  batch_size=4
elif [[ $2 == "dbpedia" ]]; then
  batch_size=4
fi

echo "Batch size" $batch_size

echo "model: $model"
# for task in  "squad" "nq_obqa" "tq_obqa" "hotpotqa" ; do
task=$2
seed=$3
nshot=$4
# read $5 as num_retrieved_docs, if not provided, set it to 0
num_retrieved_docs=${5:-0}
echo "seed: $seed" "nshot: $nshot" "num_retrieved_docs: $num_retrieved_docs"
echo "Evaluating $task..."
python fewshot.py \
      --model_path="${model}" \
      --task="${task}" \
      --n_shot=$nshot \
      --seed="${seed}" \
      --device=0 \
      --batch_size=$batch_size \
      --downsample \
      --num_retrieved_docs=$num_retrieved_docs