# example dataset name: redpajama_2b
DATASET_NAME=$1
export SOURCE_PATH=/home/aiops/zhuty/ret_pretraining_data/$DATASET_NAME
# shellcheck disable=SC1001
export DEST_PATH=/home/aiops/zhuty/ret_pretraining_data/$DATASET_NAME\_sample_processed
export TK_PATH=/home/aiops/zhuty/tinyllama/models

for split in 'train' 'valid' ; do
python scripts/prepare_file.py --source_path $SOURCE_PATH --tokenizer_path $TK_PATH --destination_path $DEST_PATH  --short_name $DATASET_NAME --split $split
done
