
export GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0)
export NUMBER_OF_GPU=$(python -c "import torch; print(torch.cuda.device_count())")
echo "Using $NUMBER_OF_GPU GPUs"
echo "WANDB_NAME=$WANDB_NAME"
echo "WANDB_TAGS=$WANDB_TAGS"
lightning run model \
    --node-rank=0  \
    --main-address=127.0.01 \
    --accelerator=cuda \
    --num-nodes=1 \
    --devices="$NUMBER_OF_GPU" \
     test/test_cli.py --num_devices $NUMBER_OF_GPU \
    --train_data_dir /home/aiops/zhuty/ret_pretraining_data/sample_processed/redpajama_2b \
    --val_data_dir /home/aiops/zhuty/ret_pretraining_data/sample_processed/redpajama_2b/valid \
    --resume true
# sample usage
# bash scripts/pretraining.sh redpajama_2b
# bash scripts/pretraining.sh redpajama_2b_reordered_train_top10