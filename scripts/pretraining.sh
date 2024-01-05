export WANDB_PROJECT=RetPretrain
export WANDB_API_KEY=5723c6f7e50618fd5dd5a9c2dc2c293f293a25e6
export DATASET_NAME=$2
FULL_DATA_PATH=/home/aiops/zhuty/ret_pretraining_data/$DATASET_NAME\_sample_processed
export MODEL_NAME=$1

# ensure that the model name is valid , either 'tiny_LLaMA_1b' or 'tiny_LLaMA_120M' or 'tiny_LLaMA_120M_4k'
if [ $MODEL_NAME != "tiny_LLaMA_1b" ] && [ $MODEL_NAME != "tiny_LLaMA_120M" ] && [ $MODEL_NAME != "tiny_LLaMA_120M_4k" ]; then
    echo "Error: '$MODEL_NAME' is not a valid model name."
    exit 1
fi


export WANDB_NAME=$MODEL_NAME\_$DATASET_NAME
export NUMBER_OF_GPU=$(python -c "import torch; print(torch.cuda.device_count())")
echo "Using $NUMBER_OF_GPU GPUs"
echo "WANDB_NAME=$WANDB_NAME"

lightning run model \
    --node-rank=0  \
    --main-address=127.0.01 \
    --accelerator=cuda \
    --num-nodes=1 \
    --devices=$NUMBER_OF_GPU \
    pretrain/tinyllama.py --devices $NUMBER_OF_GPU \
    --train_data_dir $FULL_DATA_PATH \
    --val_data_dir $FULL_DATA_PATH

# sample usage
# bash scripts/pretraining.sh redpajama_2b
# bash scripts/pretraining.sh redpajama_2b_reordered_train_top10