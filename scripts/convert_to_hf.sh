#python convert_lit_checkpoint.py \
#--checkpoint_name ckpt_5000_2b.pth \
#--out_dir /home/aiops/zhuty/tinyllama/out/tinyllama_120M \
#--model_name tiny_LLaMA_120M \
#--model_only false

LIT_CKPT_DIR=/home/aiops/zhuty/tinyllama/out/tinyllaMA_1b
CKPT_PREFIX=iter-020000-ckpt

python convert_lit_checkpoint.py \
--checkpoint_name $CKPT_PREFIX.pth \
--out_dir $LIT_CKPT_DIR \
--model_name tiny_LLaMA_1b \
--model_only false

mkdir $LIT_CKPT_DIR/hf_ckpt
# move config
mv $LIT_CKPT_DIR/config.json $LIT_CKPT_DIR/hf_ckpt/config.json
mv $LIT_CKPT_DIR/$CKPT_PREFIX.bin $LIT_CKPT_DIR/hf_ckpt/pytorch_model.bin