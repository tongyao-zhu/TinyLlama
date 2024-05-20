export LM_HARNESSS_CACHE_PATH=/home/aiops/zhuty/lm_harness_cache

# --tasks agieval,arithmetic,asdiv,bbh,drop,gsm8k,hellaswag,mathqa,mmlu,piqa,pubmedqa,qasper,race,sciq,social_iqa,lambada_standard,openbookqa,arc_easy,arc_challenge,winogrande,mnli,mrpc,rte,qnli,qqp,sst2,wnli,boolq,copa,multirc,wsc,wikitext,logiqa,eq_bench,code2text_python \

# 'arithmetic'
# 'asdiv'
# 'bbh_fewshot'
# 'drop'
# 'scorll'
# 'eq_bench'
# 'code2text_python'

task=$1
model=$2
numshot=$3

output_path=/home/aiops/zhuty/tinyllama/scripts/lm_harness_eval/out/$model/$task/$numshot
if [ ! -d $output_path ]; then
   echo "Creating output directory: $output_path"
   mkdir -p $output_path
fi
#for task in "${tasks[@]}"; do
# print the task name
echo "Evaluating task: $task"
# if the "result.json" file exists in the target folder, skip
if [ -f "$output_path/result.json" ]; then
    echo "Task $task has been evaluated, skip"
    exit 0
fi

lm_eval --model hf \
    --model_args pretrained=$model,tokenizer="tyzhu/tinyllama_common_tokenizer",add_bos_token=False,trust_remote_code=True,prefix_token_id=2 \
    --tasks $task \
    --batch_size auto:4 \
    --num_fewshot $numshot \
    --output_path $output_path \
    --log_samples

    # --cache_requests true \