 sailctl job create eval360micl -g 1   --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash icl.sh /home/aiops/zhuty/tinyllama/out/tiny_LLaMA_360M_8k_cc_8k/iter-190000-ckpt-step-47500_hf '
            
 sailctl job create eval360mmrc -g 1   --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash mrc.sh /home/aiops/zhuty/tinyllama/out/tiny_LLaMA_360M_8k_cc_8k/iter-190000-ckpt-step-47500_hf '
            
 sailctl job create eval360mcbqa -g 1    --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash cbqa.sh /home/aiops/zhuty/tinyllama/out/tiny_LLaMA_360M_8k_cc_8k/iter-190000-ckpt-step-47500_hf '
            
 sailctl job create eval360micl -g 1   --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash icl.sh /home/aiops/zhuty/tinyllama/out/tiny_LLaMA_360M_8k_cc_merged_v1_8k/iter-190000-ckpt-step-47500_hf '
            
 sailctl job create eval360mmrc -g 1   --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash mrc.sh /home/aiops/zhuty/tinyllama/out/tiny_LLaMA_360M_8k_cc_merged_v1_8k/iter-190000-ckpt-step-47500_hf '
            
 sailctl job create eval360mcbqa -g 1   --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash cbqa.sh /home/aiops/zhuty/tinyllama/out/tiny_LLaMA_360M_8k_cc_merged_v1_8k/iter-190000-ckpt-step-47500_hf '
            
 sailctl job create eval360micl -g 1   --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash icl.sh /home/aiops/zhuty/tinyllama/out/tiny_LLaMA_360M_8k_cc_merged_v2_8k/iter-190000-ckpt-step-47500_hf '
            
 sailctl job create eval360mmrc -g 1   --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash mrc.sh /home/aiops/zhuty/tinyllama/out/tiny_LLaMA_360M_8k_cc_merged_v2_8k/iter-190000-ckpt-step-47500_hf '
            
 sailctl job create eval360mcbqa -g 1    --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash cbqa.sh /home/aiops/zhuty/tinyllama/out/tiny_LLaMA_360M_8k_cc_merged_v2_8k/iter-190000-ckpt-step-47500_hf '
            

for task in  "squad" "nq_obqa" "tq_obqa" "hotpotqa" ; do
for ds in 'cc' 'cc_merged_v1' 'cc_merged_v2' ; do
 taskname=$(echo $task | sed 's/_//g')
 dsname=$(echo $ds | sed 's/_//g')
 sailctl job create evalmrc$taskname$dsname -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash mrc_single.sh /home/aiops/zhuty/tinyllama/out/tiny_LLaMA_360M_8k_'$ds'_8k/iter-190000-ckpt-step-47500_hf  '$task' '
done
done

  for task in  'agnews' 'amazon' 'dbpedia' 'sst2' 'tweet_hate' 'tweet_offensive' 'yelp'  ; do
for ds in 'cc' 'cc_merged_v1' 'cc_merged_v2' ; do
 taskname=$(echo $task | sed 's/_//g')
 dsname=$(echo $ds | sed 's/_//g')
 sailctl job create eval360micl$taskname$dsname -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash icl_single.sh /home/aiops/zhuty/tinyllama/out/tiny_LLaMA_360M_8k_'$ds'_8k/iter-190000-ckpt-step-47500_hf '$task' '
done
done
