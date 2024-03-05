#!/bin/bash -l

set -e
set -u

model=$1
echo "model: $model"

# adjust batch size based on model size
if [[ $model == *"120M"* ]]; then
  batch_size=8
elif [[ $model == *"360M"* ]]; then
  batch_size=1
else
  batch_size=1
fi
echo "Batch size" $batch_size

for seed in {42..46}; do
for task in "nq" "tq"; do
  python fewshot.py \
    --model_path="${model}" \
    --task="${task}" \
    --n_shot=24 \
    --seed="${seed}" \
    --device=0 \
    --batch_size=$batch_size
done
done