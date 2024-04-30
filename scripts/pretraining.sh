export WANDB_PROJECT=RetPretrain
export WANDB_API_KEY=5723c6f7e50618fd5dd5a9c2dc2c293f293a25e6
export DATASET_NAME=$2
export VALID_DATASET_NAME=$3
FULL_DATA_PATH=/home/aiops/zhuty/ret_pretraining_data/sample_processed/$DATASET_NAME
VALID_DATA_PATH=/home/aiops/zhuty/ret_pretraining_data/sample_processed/$VALID_DATASET_NAME/valid
export MODEL_NAME=$1
export resume=$4
export eval_only=$5
export suffix=$6

# check if we need to resume
if [[ $resume == "true" ]]; then
  echo "Resuming from checkpoint"
  export resume=true
else
  echo "Not resuming from checkpoint"
  export resume=false
fi

if [[ $eval_only == "true" ]]; then
  echo "Evaluating only"
  export eval_only=true
else
  echo "Training and evaluating"
  export eval_only=false
fi

# List of valid model names
valid_models=("tiny_LLaMA_1b" "tiny_LLaMA_120M" "tiny_LLaMA_120M_4k" "tiny_LLaMA_120M_8k" "tiny_LLaMA_1b_4k" "tiny_LLaMA_1b_8k" "tiny_LLaMA_360M" "tiny_LLaMA_360M_4k" "tiny_LLaMA_360M_8k" "tiny_LLaMA_1b_8k_intramask" "tiny_LLaMA_1b_8k_adamask" "tiny_LLaMA_1b_8k_intramask_olm" "tiny_LLaMA_360M_8k_intramask" "tiny_LLaMA_360M_8k_intramask_olm" "tiny_LLaMA_360M_8k_adamask") # Add more model names as needed

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

export GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0)
export WANDB_NAME=$MODEL_NAME\_$DATASET_NAME
if [[ $suffix != "" ]]; then
  echo "Adding suffix $suffix"
  export WANDB_NAME=$WANDB_NAME\_$suffix
  echo "WANDB_NAME=$WANDB_NAME"
fi
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
    pretrain/tinyllama.py --num_devices $NUMBER_OF_GPU \
    --train_data_dir $FULL_DATA_PATH \
    --val_data_dir $VALID_DATA_PATH \
    --resume $resume \
    --eval_only $eval_only

# sample usage
# bash scripts/pretraining.sh redpajama_2b
# bash scripts/pretraining.sh redpajama_2b_reordered_train_top10