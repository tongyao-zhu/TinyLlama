#!/bin/bash

DATASET_NAME=$1

# Declare an associative array
declare -A result_path_file

JSON_FILE=/home/aiops/zhuty/tinyllama/processing/configs/version_to_path.json
# Iterate over the keys in the JSON file and populate the array
for key in $(jq -r 'keys[]' "$JSON_FILE"); do
    result_path_file["$key"]=$(jq -r ".[\"$key\"]" "$JSON_FILE")
done

# Check if the DATASET_NAME exists in the array
if [[ -v result_path_file["$DATASET_NAME"] ]]; then
    # Access and print a value from the array
    echo "The value for '$DATASET_NAME' is: ${result_path_file[$DATASET_NAME]}"
else
    echo "Error: '$DATASET_NAME' is not a valid dataset name."
    exit 1
fi

# Access and print a value from the array
echo "The value for '$DATASET_NAME' is: ${result_path_file[$DATASET_NAME]}"

mkdir /home/aiops/zhuty/ret_pretraining_data/$DATASET_NAME/
mkdir /home/aiops/zhuty/ret_pretraining_data/$DATASET_NAME/train
echo "Copying data to /home/aiops/zhuty/ret_pretraining_data/$DATASET_NAME/valid"
cp -r /home/aiops/zhuty/ret_pretraining_data/redpajama_2b/valid /home/aiops/zhuty/ret_pretraining_data/$DATASET_NAME/valid
echo "finished copying data to /home/aiops/zhuty/ret_pretraining_data/$DATASET_NAME/valid"
# Reorder the data
python processing/create_new_chunk_files.py --result_path_file  ${result_path_file[$DATASET_NAME]} \
--reordered_data_dir /home/aiops/zhuty/ret_pretraining_data/$DATASET_NAME/train/
# run processing
# please run this by yourself, the environment is not set up

# bash scripts/pajama_processing.sh $DATASET_NAME