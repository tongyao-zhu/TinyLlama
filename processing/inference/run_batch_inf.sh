#!/bin/bash

# Read the input argument
CHUNK_NUM=$1

# Calculate start and end points for the iteration
let "start = $CHUNK_NUM * 10"
let "end = $start + 9"

# Iterate over the calculated range
for i in $(seq $start $end); do
  python  run_batch_inf.py --train_data_dir /home/aiops/zhuty/ret_pretraining_data/redpajama_2b_id_added/queries/ \
  --model_name /home/aiops/zhuty/tinyllama/out/tinyllama_120M/ \
  --batch_size 32 --save_dir /home/aiops/zhuty/ret_pretraining_data/redpajama_2b_id_added/generated_queries \
  --chunk_num "$1"
done