 sailctl job create eval120mtqcc42 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_8k_iter-240000-ckpt-step-60000_hf tq 42 '   ; sleep 2;  sailctl job create eval120mtqcc43 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_8k_iter-240000-ckpt-step-60000_hf tq 43 '   ; sleep 2;  sailctl job create eval120mtqcc44 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_8k_iter-240000-ckpt-step-60000_hf tq 44 '   ; sleep 2;  sailctl job create eval120mtqcc45 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_8k_iter-240000-ckpt-step-60000_hf tq 45 '   ; sleep 2;  sailctl job create eval120mtqcc46 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_8k_iter-240000-ckpt-step-60000_hf tq 46 '   ; sleep 2;  sailctl job create eval120mtqccmergedv142 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v1_8k_iter-240000-ckpt-step-60000_hf tq 42 '   ; sleep 2;  sailctl job create eval120mtqccmergedv143 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v1_8k_iter-240000-ckpt-step-60000_hf tq 43 '   ; sleep 2;  sailctl job create eval120mtqccmergedv144 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v1_8k_iter-240000-ckpt-step-60000_hf tq 44 '   ; sleep 2;  sailctl job create eval120mtqccmergedv145 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v1_8k_iter-240000-ckpt-step-60000_hf tq 45 '   ; sleep 2;  sailctl job create eval120mtqccmergedv146 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v1_8k_iter-240000-ckpt-step-60000_hf tq 46 '   ; sleep 2;  sailctl job create eval120mtqccmergedv242 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v2_8k_iter-240000-ckpt-step-60000_hf tq 42 '   ; sleep 2;  sailctl job create eval120mtqccmergedv243 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v2_8k_iter-240000-ckpt-step-60000_hf tq 43 '   ; sleep 2;  sailctl job create eval120mtqccmergedv244 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v2_8k_iter-240000-ckpt-step-60000_hf tq 44 '   ; sleep 2;  sailctl job create eval120mtqccmergedv245 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v2_8k_iter-240000-ckpt-step-60000_hf tq 45 '   ; sleep 2;  sailctl job create eval120mtqccmergedv246 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v2_8k_iter-240000-ckpt-step-60000_hf tq 46 '   ; sleep 2;  sailctl job create eval120mtqccmergedv342 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v3_8k_iter-240000-ckpt-step-60000_hf tq 42 '   ; sleep 2;  sailctl job create eval120mtqccmergedv343 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v3_8k_iter-240000-ckpt-step-60000_hf tq 43 '   ; sleep 2;  sailctl job create eval120mtqccmergedv344 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v3_8k_iter-240000-ckpt-step-60000_hf tq 44 '   ; sleep 2;  sailctl job create eval120mtqccmergedv345 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v3_8k_iter-240000-ckpt-step-60000_hf tq 45 '   ; sleep 2;  sailctl job create eval120mtqccmergedv346 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v3_8k_iter-240000-ckpt-step-60000_hf tq 46 '   ; sleep 2;  sailctl job create eval120mnqcc42 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_8k_iter-240000-ckpt-step-60000_hf nq 42 '   ; sleep 2;  sailctl job create eval120mnqcc43 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_8k_iter-240000-ckpt-step-60000_hf nq 43 '   ; sleep 2;  sailctl job create eval120mnqcc44 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_8k_iter-240000-ckpt-step-60000_hf nq 44 '   ; sleep 2;  sailctl job create eval120mnqcc45 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_8k_iter-240000-ckpt-step-60000_hf nq 45 '   ; sleep 2;  sailctl job create eval120mnqcc46 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_8k_iter-240000-ckpt-step-60000_hf nq 46 '   ; sleep 2;  sailctl job create eval120mnqccmergedv142 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v1_8k_iter-240000-ckpt-step-60000_hf nq 42 '   ; sleep 2;  sailctl job create eval120mnqccmergedv143 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v1_8k_iter-240000-ckpt-step-60000_hf nq 43 '   ; sleep 2;  sailctl job create eval120mnqccmergedv144 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v1_8k_iter-240000-ckpt-step-60000_hf nq 44 '   ; sleep 2;  sailctl job create eval120mnqccmergedv145 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v1_8k_iter-240000-ckpt-step-60000_hf nq 45 '   ; sleep 2;  sailctl job create eval120mnqccmergedv146 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v1_8k_iter-240000-ckpt-step-60000_hf nq 46 '   ; sleep 2;  sailctl job create eval120mnqccmergedv242 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v2_8k_iter-240000-ckpt-step-60000_hf nq 42 '   ; sleep 2;  sailctl job create eval120mnqccmergedv243 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v2_8k_iter-240000-ckpt-step-60000_hf nq 43 '   ; sleep 2;  sailctl job create eval120mnqccmergedv244 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v2_8k_iter-240000-ckpt-step-60000_hf nq 44 '   ; sleep 2;  sailctl job create eval120mnqccmergedv245 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v2_8k_iter-240000-ckpt-step-60000_hf nq 45 '   ; sleep 2;  sailctl job create eval120mnqccmergedv246 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v2_8k_iter-240000-ckpt-step-60000_hf nq 46 '   ; sleep 2;  sailctl job create eval120mnqccmergedv342 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v3_8k_iter-240000-ckpt-step-60000_hf nq 42 '   ; sleep 2;  sailctl job create eval120mnqccmergedv343 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v3_8k_iter-240000-ckpt-step-60000_hf nq 43 '   ; sleep 2;  sailctl job create eval120mnqccmergedv344 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v3_8k_iter-240000-ckpt-step-60000_hf nq 44 '   ; sleep 2;  sailctl job create eval120mnqccmergedv345 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v3_8k_iter-240000-ckpt-step-60000_hf nq 45 '   ; sleep 2;  sailctl job create eval120mnqccmergedv346 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v3_8k_iter-240000-ckpt-step-60000_hf nq 46 '