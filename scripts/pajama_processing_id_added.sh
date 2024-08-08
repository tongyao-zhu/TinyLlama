# example dataset name: redpajama_2b
DATASET_NAME=$1
PROCESSING_LENGTH=$2
export SOURCE_PATH=/home/aiops/zhuty/ret_pretraining_data/id_added/$DATASET_NAME
# shellcheck disable=SC1001
export DEST_PATH=/home/aiops/zhuty/ret_pretraining_data/sample_processed/$DATASET_NAME
# remove the DEST_PATH if it exists
rm -rf $DEST_PATH
mkdir -p $DEST_PATH
export TK_PATH=/home/aiops/zhuty/tinyllama/models

# if length is 4k, set chunk_size to 65552

if [ $PROCESSING_LENGTH == "4k" ]; then
    chunk_size=65552 # (4096 + 1) * 16
    DEST_PATH=$DEST_PATH\_4k
elif [ $PROCESSING_LENGTH == "8k" ]; then
    chunk_size=131088 # (8192 + 1) * 16
    DEST_PATH=$DEST_PATH\_8k
elif [ $PROCESSING_LENGTH == "16k" ]; then
    chunk_size=262160 # (16384 + 1) * 16
    DEST_PATH=$DEST_PATH\_16k
else
   chunk_size=32784 # (2048 + 1) * 16
fi

echo $chunk_size

for split in 'train'  ; do
python scripts/prepare_file.py --source_path $SOURCE_PATH \
--chunk_size $chunk_size --tokenizer_path $TK_PATH --destination_path $DEST_PATH  --short_name $DATASET_NAME --split $split --text_key contents
done
