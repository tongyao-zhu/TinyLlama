#!/bin/bash

MODEL_NAME=$1
DATASET_NAME=$2
QUERY_TYPE=$3
# Read the input argument
start=$4
end=$5

## Calculate start and end points for the iteration
#let "start = $CHUNK_NUM * 10"
#let "end = $start + 9"
export GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0)
# check if memory is 81920
if [[ $GPU_MEMORY == 81920 ]]; then
  BATCH_SIZE=96
else
  BATCH_SIZE=24
fi

if [[ $MODEL_NAME == *"tiny_LLaMA_120M_8k"* ]]; then
  # increse batc size for two times
  BATCH_SIZE=64
fi

echo "Using $BATCH_SIZE batch size"

# Iterate over the calculated range
for i in $(seq $start $end); do
  echo "Running inference for chunk $i"
  python  run_batch_inf.py --train_data_dir /home/aiops/zhuty/ret_pretraining_data/id_added/$DATASET_NAME/queries/$QUERY_TYPE \
  --model_name $MODEL_NAME \
  --batch_size $BATCH_SIZE --save_dir /home/aiops/zhuty/ret_pretraining_data/id_added/$DATASET_NAME/generated_queries/$QUERY_TYPE \
  --chunk_num "$i" ;
done