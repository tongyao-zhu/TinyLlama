#!/bin/bash

# Ensure a path is provided as an argument
if [ -z "$1" ]; then
  echo "Usage: $0 <path_to_checkpoint_files>"
  exit 1
fi

path=$1

# Ensure the provided path exists
if [ ! -e "$path" ]; then
  echo "Error: Directory $path does not exist."
  exit 1
fi

# Iterate through the steps in the specified range
# for step in $(seq 5000 5000 75000); do
for step in 77500 ; do
  # Calculate the iteration count
  # iter=$(printf "%06d" $((8 * step)))
  # Generate the file name
  # filename="iter-${iter}-ckpt-step-${step}.pth"
  # get the checkpoint with the step number
  # filename='iter-*-ckpt-step-'"${step}"'.pth'
  # filepath="${path}/${filename}"
  filepath=$path
  # get the last file name
  filename=$(ls -t $filepath | head -n 1)
  path=$(dirname "$filepath")
  # Check if the file exists
  if [ ! -e "${filepath}" ]; then
    echo "Error: ${filename} does not exist in ${path}."
    # continue to the next step
    continue
    exit 1
  fi

  ckpt_prefix=$(basename "$filename" .pth)
  echo "Processing checkpoint ${ckpt_prefix}"

  # Skip if already converted
  if [ -f "${path}/${ckpt_prefix}_hf/config.json" ]; then
    echo "Checkpoint ${ckpt_prefix} already converted, skipping."
    continue
  else
    echo "Converting checkpoint ${ckpt_prefix}"
    bash /home/aiops/zhuty/tinyllama/scripts/convert_to_hf_general.sh "$path" "$ckpt_prefix"
  fi

  python /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/upload_to_hf.py "${path}/${ckpt_prefix}_hf"
done
