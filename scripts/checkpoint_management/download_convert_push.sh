#!/bin/bash

# Sample usage:
# bash checkpoint_management/download_convert_push.sh tyzhu/_home_aiops_zhuty_tinyllama_out_tiny_LLaMA_1b_2k_intramask_cc_2k    ~/temp_models/tiny_LLaMA_1b_2k_intramask_cc_2k
# bash checkpoint_management/download_convert_push.sh tyzhu/_home_aiops_zhuty_tinyllama_out_tiny_LLaMA_1b_2k_cc_2k    ~/temp_models/tiny_LLaMA_1b_2k_cc_2k
# bash checkpoint_management/download_convert_push.sh tyzhu/_home_aiops_zhuty_tinyllama_out_tiny_LLaMA_1b_2k_intramask_cc_me   ~/temp_models/tiny_LLaMA_1b_2k_intramask_cc_merged_v2_2k



# Ensure a path is provided as an argument
if [ -z "$1" ]; then
  echo "Usage: $0 <path_to_checkpoint_files>"
  exit 1
fi

hf_path=$1
path=$2

# Ensure the provided path exists
if [ ! -d "$path" ]; then
  echo "Error: Directory $path does not exist."
  exit 1
fi

# source /home/aiops/zhuty/start.sh
# cd /home/aiops/zhuty/tinyllama/

# Iterate through the steps in the specified range
for step in $(seq 5000 5000 120000); do
  # Calculate the iteration count
  iter=$(printf "%06d" $((8 * step)))
  filename="iter-${iter}-ckpt-step-${step}.pth"
  echo "Downloading ${filename} from ${hf_path}"
  ckpt_prefix=$(basename "$filename" .pth)


  if python3 /home/aiops/zhuty/tinyllama/scripts/checkpoint_management/check_file_exists.py "${path}/${ckpt_prefix}_hf" ; then
    echo "File $hf_path/$filename exists, skipping."
    continue
  fi

  # download
  huggingface-cli download $hf_path $filename --local-dir $path

  # Generate the file name
  filepath="${path}/${filename}"

  # Check if the file exists
  if [ ! -e "${filepath}" ]; then
    echo "Error: ${filename} does not exist in ${path}."
    # continue to the next step
    continue
  fi

  echo "Processing checkpoint ${ckpt_prefix}"

  # Skip if already converted
  if [ -f "${path}/${ckpt_prefix}_hf/config.json" ]; then
    echo "Checkpoint ${ckpt_prefix} already converted, skipping."
  else
    echo "Converting checkpoint ${ckpt_prefix}"
    bash /home/aiops/zhuty/tinyllama/scripts/convert_to_hf_general.sh "$path" "$ckpt_prefix"
  fi
  python /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/upload_to_hf.py "${path}/${ckpt_prefix}_hf"
  rm -r "${path}/${ckpt_prefix}_hf"
  rm "${path}/${filename}"
  echo "Removed ${path}/${ckpt_prefix}_hf"
  echo "Removed ${path}/${filename}"
done
