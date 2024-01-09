export WANDB_PROJECT=RetPretrain
export WANDB_API_KEY=5723c6f7e50618fd5dd5a9c2dc2c293f293a25e6
export DATASET_NAME=$2
FULL_DATA_PATH=/home/aiops/zhuty/ret_pretraining_data/sample_processed/$DATASET_NAME
export MODEL_NAME=$1


# List of valid model names
valid_models=("tiny_LLaMA_1b" "tiny_LLaMA_120M" "tiny_LLaMA_120M_4k" "tiny_LLaMA_120M_8k" "tiny_LLaMA_1b_4k" "tiny_LLaMA_1b_8k" "tiny_LLaMA_360M") # Add more model names as needed

# Function to check if a model name is valid
is_valid_model() {
    local model_name=$1
    for valid_model in "${valid_models[@]}"; do
        if [ "$model_name" == "$valid_model" ]; then
            return 0 # Model name is valid
        fi
    done
    return 1 # Model name is not valid
}

# Check if the provided MODEL_NAME is valid
if ! is_valid_model "$MODEL_NAME"; then
    echo "Error: '$MODEL_NAME' is not a valid model name."
    exit 1
fi

export WANDB_NAME=$MODEL_NAME\_$DATASET_NAME
export NUMBER_OF_GPU=$(python -c "import torch; print(torch.cuda.device_count())")
export WANDB_TAGS="pretraining,$DATASET_NAME,$MODEL_NAME"
echo "Using $NUMBER_OF_GPU GPUs"
echo "WANDB_NAME=$WANDB_NAME"
echo "WANDB_TAGS=$WANDB_TAGS"

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