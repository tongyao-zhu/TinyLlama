path=$1
# get all files ending with .pth, sorted in alphabetical order
ckpt_files=$(ls $path/*.pth | sort)

echo "Detected checkpoint files: $ckpt_files"
reserved_steps="iter-380000-ckpt-step-47500 iter-480000-ckpt-step-60000 iter-600000-ckpt-step-75000 iter-160000-ckpt-step-40000 iter-110000-ckpt-step-27500 iter-240000-ckpt-step-60000"

for ckpt_file in $ckpt_files; do
  # get the filename without the extension
  ckpt_prefix=$(basename $ckpt_file .pth)
  echo "Converting checkpoint $ckpt_prefix"
  # if the checkpoint is already converted (the dir includes a config file, $path/$ckpt_prefix\_hf/config.json), skip
  if [ -f $path/$ckpt_prefix\_hf/config.json ]; then
    echo "Checkpoint $ckpt_prefix already converted, skip"
  else
    echo "Converting checkpoint $ckpt_prefix"
  bash /home/aiops/zhuty/tinyllama/scripts/convert_to_hf_general.sh $path $ckpt_prefix
  fi
  python /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/upload_to_hf.py $path/$ckpt_prefix\_hf
  if [[ $reserved_steps == *$ckpt_prefix* ]]; then
    echo "Reserved step, skipping deletion"
  else
    # delete the converted HF checkpoint
    echo "Deleting $path/$ckpt_prefix\_hf"
    rm -r $path/$ckpt_prefix\_hf
  fi
done



