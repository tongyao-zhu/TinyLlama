#!/bin/bash

# Ensure a path is provided as an argument
path=$1
if [ -z "$path" ]; then
  echo "Usage: $0 <path_to_checkpoint_files>"
  exit 1
fi

source /home/aiops/zhuty/start.sh
cd /home/aiops/zhuty/tinyllama/out
while true
do
    # Your script's main operations go here
    echo "Upload the model to hub"
    # (Replace the above line with your actual task, e.g., a push to hub command)
    python /home/aiops/zhuty/tinyllama/scripts/checkpoint_management/upload_folder.py $path model true
    # Sleep for 1 hour
    sleep 3600
done
