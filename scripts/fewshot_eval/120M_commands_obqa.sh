 sailctl job create eval120msquadcc42 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_8k_iter-240000-ckpt-step-60000_hf squad 42 '   ; sleep 2;  sailctl job create eval120msquadcc43 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_8k_iter-240000-ckpt-step-60000_hf squad 43 '   ; sleep 2;  sailctl job create eval120msquadcc44 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_8k_iter-240000-ckpt-step-60000_hf squad 44 '   ; sleep 2;  sailctl job create eval120msquadcc45 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_8k_iter-240000-ckpt-step-60000_hf squad 45 '   ; sleep 2;  sailctl job create eval120msquadcc46 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_8k_iter-240000-ckpt-step-60000_hf squad 46 '   ; sleep 2;  sailctl job create eval120msquadccmergedv142 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v1_8k_iter-240000-ckpt-step-60000_hf squad 42 '   ; sleep 2;  sailctl job create eval120msquadccmergedv143 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v1_8k_iter-240000-ckpt-step-60000_hf squad 43 '   ; sleep 2;  sailctl job create eval120msquadccmergedv144 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v1_8k_iter-240000-ckpt-step-60000_hf squad 44 '   ; sleep 2;  sailctl job create eval120msquadccmergedv145 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v1_8k_iter-240000-ckpt-step-60000_hf squad 45 '   ; sleep 2;  sailctl job create eval120msquadccmergedv146 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v1_8k_iter-240000-ckpt-step-60000_hf squad 46 '   ; sleep 2;  sailctl job create eval120msquadccmergedv242 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v2_8k_iter-240000-ckpt-step-60000_hf squad 42 '   ; sleep 2;  sailctl job create eval120msquadccmergedv243 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v2_8k_iter-240000-ckpt-step-60000_hf squad 43 '   ; sleep 2;  sailctl job create eval120msquadccmergedv244 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v2_8k_iter-240000-ckpt-step-60000_hf squad 44 '   ; sleep 2;  sailctl job create eval120msquadccmergedv245 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v2_8k_iter-240000-ckpt-step-60000_hf squad 45 '   ; sleep 2;  sailctl job create eval120msquadccmergedv246 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v2_8k_iter-240000-ckpt-step-60000_hf squad 46 '   ; sleep 2;  sailctl job create eval120msquadccmergedv342 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v3_8k_iter-240000-ckpt-step-60000_hf squad 42 '   ; sleep 2;  sailctl job create eval120msquadccmergedv343 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v3_8k_iter-240000-ckpt-step-60000_hf squad 43 '   ; sleep 2;  sailctl job create eval120msquadccmergedv344 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v3_8k_iter-240000-ckpt-step-60000_hf squad 44 '   ; sleep 2;  sailctl job create eval120msquadccmergedv345 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v3_8k_iter-240000-ckpt-step-60000_hf squad 45 '   ; sleep 2;  sailctl job create eval120msquadccmergedv346 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v3_8k_iter-240000-ckpt-step-60000_hf squad 46 '   ; sleep 2;  sailctl job create eval120mnqobqacc42 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_8k_iter-240000-ckpt-step-60000_hf nq_obqa 42 '   ; sleep 2;  sailctl job create eval120mnqobqacc43 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_8k_iter-240000-ckpt-step-60000_hf nq_obqa 43 '   ; sleep 2;  sailctl job create eval120mnqobqacc44 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_8k_iter-240000-ckpt-step-60000_hf nq_obqa 44 '   ; sleep 2;  sailctl job create eval120mnqobqacc45 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_8k_iter-240000-ckpt-step-60000_hf nq_obqa 45 '   ; sleep 2;  sailctl job create eval120mnqobqacc46 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_8k_iter-240000-ckpt-step-60000_hf nq_obqa 46 '   ; sleep 2;  sailctl job create eval120mnqobqaccmergedv142 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v1_8k_iter-240000-ckpt-step-60000_hf nq_obqa 42 '   ; sleep 2;  sailctl job create eval120mnqobqaccmergedv143 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v1_8k_iter-240000-ckpt-step-60000_hf nq_obqa 43 '   ; sleep 2;  sailctl job create eval120mnqobqaccmergedv144 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v1_8k_iter-240000-ckpt-step-60000_hf nq_obqa 44 '   ; sleep 2;  sailctl job create eval120mnqobqaccmergedv145 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v1_8k_iter-240000-ckpt-step-60000_hf nq_obqa 45 '   ; sleep 2;  sailctl job create eval120mnqobqaccmergedv146 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v1_8k_iter-240000-ckpt-step-60000_hf nq_obqa 46 '   ; sleep 2;  sailctl job create eval120mnqobqaccmergedv242 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v2_8k_iter-240000-ckpt-step-60000_hf nq_obqa 42 '   ; sleep 2;  sailctl job create eval120mnqobqaccmergedv243 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v2_8k_iter-240000-ckpt-step-60000_hf nq_obqa 43 '   ; sleep 2;  sailctl job create eval120mnqobqaccmergedv244 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v2_8k_iter-240000-ckpt-step-60000_hf nq_obqa 44 '   ; sleep 2;  sailctl job create eval120mnqobqaccmergedv245 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v2_8k_iter-240000-ckpt-step-60000_hf nq_obqa 45 '   ; sleep 2;  sailctl job create eval120mnqobqaccmergedv246 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v2_8k_iter-240000-ckpt-step-60000_hf nq_obqa 46 '   ; sleep 2;  sailctl job create eval120mnqobqaccmergedv342 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v3_8k_iter-240000-ckpt-step-60000_hf nq_obqa 42 '   ; sleep 2;  sailctl job create eval120mnqobqaccmergedv343 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v3_8k_iter-240000-ckpt-step-60000_hf nq_obqa 43 '   ; sleep 2;  sailctl job create eval120mnqobqaccmergedv344 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v3_8k_iter-240000-ckpt-step-60000_hf nq_obqa 44 '   ; sleep 2;  sailctl job create eval120mnqobqaccmergedv345 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v3_8k_iter-240000-ckpt-step-60000_hf nq_obqa 45 '   ; sleep 2;  sailctl job create eval120mnqobqaccmergedv346 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v3_8k_iter-240000-ckpt-step-60000_hf nq_obqa 46 '   ; sleep 2;  sailctl job create eval120mtqobqacc42 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_8k_iter-240000-ckpt-step-60000_hf tq_obqa 42 '   ; sleep 2;  sailctl job create eval120mtqobqacc43 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_8k_iter-240000-ckpt-step-60000_hf tq_obqa 43 '   ; sleep 2;  sailctl job create eval120mtqobqacc44 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_8k_iter-240000-ckpt-step-60000_hf tq_obqa 44 '   ; sleep 2;  sailctl job create eval120mtqobqacc45 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_8k_iter-240000-ckpt-step-60000_hf tq_obqa 45 '   ; sleep 2;  sailctl job create eval120mtqobqacc46 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_8k_iter-240000-ckpt-step-60000_hf tq_obqa 46 '   ; sleep 2;  sailctl job create eval120mtqobqaccmergedv142 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v1_8k_iter-240000-ckpt-step-60000_hf tq_obqa 42 '   ; sleep 2;  sailctl job create eval120mtqobqaccmergedv143 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v1_8k_iter-240000-ckpt-step-60000_hf tq_obqa 43 '   ; sleep 2;  sailctl job create eval120mtqobqaccmergedv144 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v1_8k_iter-240000-ckpt-step-60000_hf tq_obqa 44 '   ; sleep 2;  sailctl job create eval120mtqobqaccmergedv145 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v1_8k_iter-240000-ckpt-step-60000_hf tq_obqa 45 '   ; sleep 2;  sailctl job create eval120mtqobqaccmergedv146 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v1_8k_iter-240000-ckpt-step-60000_hf tq_obqa 46 '   ; sleep 2;  sailctl job create eval120mtqobqaccmergedv242 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v2_8k_iter-240000-ckpt-step-60000_hf tq_obqa 42 '   ; sleep 2;  sailctl job create eval120mtqobqaccmergedv243 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v2_8k_iter-240000-ckpt-step-60000_hf tq_obqa 43 '   ; sleep 2;  sailctl job create eval120mtqobqaccmergedv244 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v2_8k_iter-240000-ckpt-step-60000_hf tq_obqa 44 '   ; sleep 2;  sailctl job create eval120mtqobqaccmergedv245 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v2_8k_iter-240000-ckpt-step-60000_hf tq_obqa 45 '   ; sleep 2;  sailctl job create eval120mtqobqaccmergedv246 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v2_8k_iter-240000-ckpt-step-60000_hf tq_obqa 46 '   ; sleep 2;  sailctl job create eval120mtqobqaccmergedv342 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v3_8k_iter-240000-ckpt-step-60000_hf tq_obqa 42 '   ; sleep 2;  sailctl job create eval120mtqobqaccmergedv343 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v3_8k_iter-240000-ckpt-step-60000_hf tq_obqa 43 '   ; sleep 2;  sailctl job create eval120mtqobqaccmergedv344 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v3_8k_iter-240000-ckpt-step-60000_hf tq_obqa 44 '   ; sleep 2;  sailctl job create eval120mtqobqaccmergedv345 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v3_8k_iter-240000-ckpt-step-60000_hf tq_obqa 45 '   ; sleep 2;  sailctl job create eval120mtqobqaccmergedv346 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v3_8k_iter-240000-ckpt-step-60000_hf tq_obqa 46 '   ; sleep 2;  sailctl job create eval120mhotpotqacc42 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_8k_iter-240000-ckpt-step-60000_hf hotpotqa 42 '   ; sleep 2;  sailctl job create eval120mhotpotqacc43 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_8k_iter-240000-ckpt-step-60000_hf hotpotqa 43 '   ; sleep 2;  sailctl job create eval120mhotpotqacc44 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_8k_iter-240000-ckpt-step-60000_hf hotpotqa 44 '   ; sleep 2;  sailctl job create eval120mhotpotqacc45 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_8k_iter-240000-ckpt-step-60000_hf hotpotqa 45 '   ; sleep 2;  sailctl job create eval120mhotpotqacc46 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_8k_iter-240000-ckpt-step-60000_hf hotpotqa 46 '   ; sleep 2;  sailctl job create eval120mhotpotqaccmergedv142 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v1_8k_iter-240000-ckpt-step-60000_hf hotpotqa 42 '   ; sleep 2;  sailctl job create eval120mhotpotqaccmergedv143 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v1_8k_iter-240000-ckpt-step-60000_hf hotpotqa 43 '   ; sleep 2;  sailctl job create eval120mhotpotqaccmergedv144 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v1_8k_iter-240000-ckpt-step-60000_hf hotpotqa 44 '   ; sleep 2;  sailctl job create eval120mhotpotqaccmergedv145 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v1_8k_iter-240000-ckpt-step-60000_hf hotpotqa 45 '   ; sleep 2;  sailctl job create eval120mhotpotqaccmergedv146 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v1_8k_iter-240000-ckpt-step-60000_hf hotpotqa 46 '   ; sleep 2;  sailctl job create eval120mhotpotqaccmergedv242 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v2_8k_iter-240000-ckpt-step-60000_hf hotpotqa 42 '   ; sleep 2;  sailctl job create eval120mhotpotqaccmergedv243 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v2_8k_iter-240000-ckpt-step-60000_hf hotpotqa 43 '   ; sleep 2;  sailctl job create eval120mhotpotqaccmergedv244 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v2_8k_iter-240000-ckpt-step-60000_hf hotpotqa 44 '   ; sleep 2;  sailctl job create eval120mhotpotqaccmergedv245 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v2_8k_iter-240000-ckpt-step-60000_hf hotpotqa 45 '   ; sleep 2;  sailctl job create eval120mhotpotqaccmergedv246 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v2_8k_iter-240000-ckpt-step-60000_hf hotpotqa 46 '   ; sleep 2;  sailctl job create eval120mhotpotqaccmergedv342 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v3_8k_iter-240000-ckpt-step-60000_hf hotpotqa 42 '   ; sleep 2;  sailctl job create eval120mhotpotqaccmergedv343 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v3_8k_iter-240000-ckpt-step-60000_hf hotpotqa 43 '   ; sleep 2;  sailctl job create eval120mhotpotqaccmergedv344 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v3_8k_iter-240000-ckpt-step-60000_hf hotpotqa 44 '   ; sleep 2;  sailctl job create eval120mhotpotqaccmergedv345 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v3_8k_iter-240000-ckpt-step-60000_hf hotpotqa 45 '   ; sleep 2;  sailctl job create eval120mhotpotqaccmergedv346 -g 1  --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash single_task.sh tyzhu/tiny_LLaMA_120M_8k_cc_merged_v3_8k_iter-240000-ckpt-step-60000_hf hotpotqa 46 '