#!/bin/bash -l

set -e
set -u

model=$1
task=$2
seed=$3

# adjust batch size based on model size
if [[ $model == *"120M"* ]]; then
  batch_size=4
elif [[ $model == *"360M"* ]]; then
  batch_size=1
else
  batch_size=1
fi
echo "Batch size" $batch_size

echo "model: $model"
# for task in  "squad" "nq_obqa" "tq_obqa" "hotpotqa" ; do
echo "Evaluating $task..."
    python fewshot.py \
      --model_path="${model}" \
      --task="${task}" \
      --n_shot=0 \
      --seed="${seed}" \
      --device=0 \
      --batch_size=$batch_size

#   for seed in {42..46}; do
