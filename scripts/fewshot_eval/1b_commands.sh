# sailctl job create eval1bicl -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash icl.sh /home/aiops/zhuty/tinyllama/out/tiny_LLaMA_1b_8k_cc_8k/iter-380000-ckpt-step-47500_hf '
#
# sailctl job create eval1bmrc -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash mrc.sh /home/aiops/zhuty/tinyllama/out/tiny_LLaMA_1b_8k_cc_8k/iter-380000-ckpt-step-47500_hf '
#
#
# sailctl job create eval1bicl -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash icl.sh /home/aiops/zhuty/tinyllama/out/tiny_LLaMA_1b_8k_cc_merged_v1_8k/iter-380000-ckpt-step-47500_hf '
#
# sailctl job create eval1bmrc -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash mrc.sh /home/aiops/zhuty/tinyllama/out/tiny_LLaMA_1b_8k_cc_merged_v1_8k/iter-380000-ckpt-step-47500_hf '
#
#
# sailctl job create eval1bicl -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash icl.sh /home/aiops/zhuty/tinyllama/out/tiny_LLaMA_1b_8k_cc_merged_v2_8k/iter-380000-ckpt-step-47500_hf '
#
# sailctl job create eval1bmrc -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash mrc.sh /home/aiops/zhuty/tinyllama/out/tiny_LLaMA_1b_8k_cc_merged_v2_8k/iter-380000-ckpt-step-47500_hf '
#
#
#sailctl job create eval1bcbqa -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash cbqa.sh /home/aiops/zhuty/tinyllama/out/tiny_LLaMA_1b_8k_cc_8k/iter-380000-ckpt-step-47500_hf '
# sailctl job create eval1bcbqa -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash cbqa.sh /home/aiops/zhuty/tinyllama/out/tiny_LLaMA_1b_8k_cc_merged_v1_8k/iter-380000-ckpt-step-47500_hf '
# sailctl job create eval1bcbqa -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash cbqa.sh /home/aiops/zhuty/tinyllama/out/tiny_LLaMA_1b_8k_cc_merged_v2_8k/iter-380000-ckpt-step-47500_hf '
#
#
## for task in  "squad" "nq_obqa" "tq_obqa" "hotpotqa" ; do
#for task in    "tq_obqa"  ; do
# taskname=$(echo $task | sed 's/_//g')
# for ds in 'cc' 'cc_merged_v1' 'cc_merged_v2' ; do
# sleep 2;
#  echo $task $ds ;
#
# taskname=$(echo $task | sed 's/_//g')
# dsname=$(echo $ds | sed 's/_//g')
# sailctl job create eval1bmrc$taskname -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash mrc_single.sh /home/aiops/zhuty/tinyllama/out/tiny_LLaMA_1b_8k_'$ds'_8k/iter-380000-ckpt-step-47500_hf '$task' '
#done
#done

#  for task in  'agnews' 'amazon' 'dbpedia' 'sst2' 'tweet_hate' 'tweet_offensive' 'yelp'  ; do
# for ds in 'cc' 'cc_merged_v1' 'cc_merged_v2' ; do
#    sleep 2;
#  echo $task $ds ;
#
# taskname=$(echo $task | sed 's/_//g')
# dsname=$(echo $ds | sed 's/_//g')
# sailctl job create eval1bicl$taskname$dsname -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash icl_single.sh /home/aiops/zhuty/tinyllama/out/tiny_LLaMA_1b_8k_'$ds'_8k/iter-380000-ckpt-step-47500_hf '$task' '
#done
#done
#
#
#
for task in  'agnews' 'amazon' 'dbpedia' 'sst2' 'tweet_hate' 'tweet_offensive' 'yelp'  "squad" "nq_obqa" "tq_obqa" "hotpotqa" 'tq' 'nq' ; do
# for task in  'amazon' 'dbpedia' ; do
# for task in "nq" "tq" ; do
 for ds in 'cc' 'cc_merged_v1' 'cc_merged_v2' ; do
  #for seed in {42..46}; do
 sleep 2;
  echo $task $ds ;
 taskname=$(echo $task | sed 's/_//g')
  dsname=$(echo $ds | sed 's/_//g')

sailctl job create zerosheval1b$taskname$dsname -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash zeroshot.sh /home/aiops/zhuty/lm_indexer_data/tyzhu/flan_max_300_added_tyzhu_tiny_LLaMA_1b_8k_'$ds'_8k_iter-380000-ckpt-step-47500_hf/checkpoint-4129 '$task' '42' '
done
done
