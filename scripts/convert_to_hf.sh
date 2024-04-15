#python convert_lit_checkpoint.py \
#--checkpoint_name ckpt_5000_2b.pth \
#--out_dir /home/aiops/zhuty/tinyllama/out/tinyllama_120M \
#--model_name tiny_LLaMA_120M \
#--model_only false

# LIT_CKPT_DIR=/home/aiops/zhuty/tinyllama/out/tinyllaMA_1b
#LIT_CKPT_DIR=/home/aiops/zhuty/tinyllama/out/tiny_LLaMA_120M_8k_rpwiki_en_8k
#CKPT_PREFIX=iter-020000-ckpt

#LIT_CKPT_DIR=/home/aiops/zhuty/tinyllama/out/tiny_LLaMA_360M_8k_c4_news_8k
#CKPT_PREFIX=iter-030000-ckpt
#
#LIT_CKPT_DIR=/home/aiops/zhuty/tinyllama/out/tiny_LLaMA_120M_8k_c4_news_8k
#CKPT_PREFIX=iter-030000-ckpt

#LIT_CKPT_DIR=/home/aiops/zhuty/tinyllama/out/tiny_LLaMA_360M_8k_cc_merged_v1_8k
#CKPT_PREFIX=iter-100000-ckpt-step-30000

#LIT_CKPT_DIR=/home/aiops/zhuty/tinyllama/out/tiny_LLaMA_360M_8k_cc_8k
#CKPT_PREFIX=iter-105000-ckpt-step-30000

LIT_CKPT_DIR=/home/aiops/zhuty/tinyllama/out/tiny_LLaMA_360M_8k_cc_8k
CKPT_PREFIX=iter-200000-ckpt-step-50000

LIT_CKPT_DIR=/home/aiops/zhuty/tinyllama/out/tiny_LLaMA_120M_8k_cc_8k
CKPT_PREFIX=iter-200000-ckpt-step-50000

# if tiny_LLaMA_120M_8k in the model_name, set model name
if [[ $LIT_CKPT_DIR == *"tiny_LLaMA_120M_8k"* ]]; then
  MODEL_NAME=tiny_LLaMA_120M_8k
else
  MODEL_NAME=tiny_LLaMA_360M_8k
fi

python convert_lit_checkpoint.py \
--checkpoint_name $CKPT_PREFIX.pth \
--out_dir $LIT_CKPT_DIR \
--model_name $MODEL_NAME \
--model_only false

mkdir -p $LIT_CKPT_DIR/hf_ckpt
# move config
mv $LIT_CKPT_DIR/config.json $LIT_CKPT_DIR/hf_ckpt/config.json
mv $LIT_CKPT_DIR/$CKPT_PREFIX.bin $LIT_CKPT_DIR/hf_ckpt/pytorch_model.bin