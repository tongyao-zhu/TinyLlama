
ckpt_dir=$1
ckpt_prefix=$2
mode=$3

if [[ $ckpt_dir == *"tiny_LLaMA_120M_8k"* ]]; then
  MODEL_NAME=tiny_LLaMA_120M_8k
elif [[ $ckpt_dir == *"tiny_LLaMA_360M_8k"* ]]; then
  MODEL_NAME=tiny_LLaMA_360M_8k
elif [[ $ckpt_dir == *"tiny_LLaMA_1b_8k"* ]]; then
  MODEL_NAME=tiny_LLaMA_1b_8k
else
  echo "MODEL_NAME not found"
  exit 1
fi
echo "MODEL_NAME $MODEL_NAME"
echo "ckpt_dir $ckpt_dir ckpt_prefix $ckpt_prefix"


# first, convert to HF
cd /home/aiops/zhuty/tinyllama/scripts/
bash convert_to_hf_general.sh $ckpt_dir $ckpt_prefix ;
echo "Converted to HF"

# second, run evaluation on the PPL datasets
cd /home/aiops/zhuty/tinyllama
for ds in "rpwiki_en_8k" "arxiv_8k" "book_8k" "cc_8k" ; do
  echo "Evaluating $ds..."
  export LOG_FILE=$ckpt_dir/$ckpt_prefix\_hf/eval_$ds.log
  bash scripts/eval_model.sh $MODEL_NAME cc_8k $ds $ckpt_dir/$ckpt_prefix.pth ;
done

# if mode is "ppl", then we are done
if [[ $mode == "ppl" ]]; then
  echo "Finished evaluating PPL"
  exit 0
fi
# third, run evaluation on the benchmark datasets
source /home/aiops/zhuty/start.sh
cd /home/aiops/zhuty/tinyllama/scripts/evaluation
bash hella_eval.sh $ckpt_dir/$ckpt_prefix\_hf