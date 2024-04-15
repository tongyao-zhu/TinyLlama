#!/bin/bash -l

set -e
set -u

model=$1
task=$2
echo "model: $model", "task: $task"

# adjust batch size based on model size
if [[ $model == *"120M"* ]]; then
  batch_size=4
elif [[ $model == *"360M"* ]]; then
  batch_size=1
else
  batch_size=1
fi
echo "Batch size" $batch_size
for seed in {42..57}; do
    echo "seed: $seed"
    echo "Evaluating $task..."
      python fewshot.py \
        --model_path=$model \
        --task="${task}" \
        --n_shot=24 \
        --seed="${seed}" \
        --device=0 \
        --batch_size=$batch_size \
        --downsample
done
