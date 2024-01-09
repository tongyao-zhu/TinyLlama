# example dataset name: redpajama_2b
DATASET_NAME=$1
PROCESSING_LENGTH=$2
export SOURCE_PATH=/home/aiops/zhuty/ret_pretraining_data/$DATASET_NAME
# shellcheck disable=SC1001
export DEST_PATH=/home/aiops/zhuty/ret_pretraining_data/sample_processed/$DATASET_NAME
export TK_PATH=/home/aiops/zhuty/tinyllama/models

if [ $PROCESSING_LENGTH == "4k" ]; then
    chunk_size=65552 # (4096 + 1) * 16
    DEST_PATH=$DEST_PATH\_4k
elif [ $PROCESSING_LENGTH == "8k" ]; then
    chunk_size=131088 # (8192 + 1) * 16
    DEST_PATH=$DEST_PATH\_8k
else
   chunk_size=32784 # (2048 + 1) * 16
fi


for split in 'train' 'valid' ; do
python scripts/prepare_file.py --source_path $SOURCE_PATH \
--chunk_size $chunk_size --tokenizer_path $TK_PATH --destination_path $DEST_PATH  --short_name $DATASET_NAME --split $split
done
