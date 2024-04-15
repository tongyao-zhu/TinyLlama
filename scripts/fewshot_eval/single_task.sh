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
echo "Batch size" $batch_size

echo "model: $model"
# for task in  "squad" "nq_obqa" "tq_obqa" "hotpotqa" ; do
task=$2
seed=$3

echo "seed: $seed"
echo "Evaluating $task..."
python fewshot.py \
      --model_path="${model}" \
      --task="${task}" \
      --n_shot=24 \
      --seed="${seed}" \
      --device=0 \
      --batch_size=$batch_size \
      --downsample