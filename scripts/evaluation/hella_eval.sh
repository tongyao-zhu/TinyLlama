MODEL_NAME=$1

echo "MODEL_NAME $MODEL_NAME"

# adjust batch size based on model size
if [[ $MODEL_NAME == *"120M"* ]]; then
  batch_size=48
elif [[ $MODEL_NAME == *"360M"* ]]; then
  batch_size=24
else
  batch_size=4
fi


lm_eval --model hf \
    --model_args pretrained=$1,dtype="float",tokenizer="meta-llama/Llama-2-7b-hf" \
    --tasks lambada_openai,hellaswag \
    --device cuda:0 \
    --num_fewshot 10 \
    --batch_size $batch_size \
    --output_path $MODEL_NAME

# sample usage
# bash scripts/hella_eval.sh /home/aiops/zhuty/tinyllama/out/tiny_LLaMA_360M_8k_cc_merged_v1_8k/hf_ckpt
# bash scripts/hella_eval.sh /home/aiops/zhuty/tinyllama/out/tiny_LLaMA_360M_8k_cc_8k/hf_ckpt