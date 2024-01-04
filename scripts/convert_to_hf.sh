#python convert_lit_checkpoint.py \
#--checkpoint_name ckpt_5000_2b.pth \
#--out_dir /home/aiops/zhuty/tinyllama/out/tinyllama_120M \
#--model_name tiny_LLaMA_120M \
#--model_only false

python convert_lit_checkpoint.py \
--checkpoint_name iter-020000-ckpt.pth \
--out_dir /home/aiops/zhuty/tinyllama/out/tinyllama_120M/tinyllama_120M_20b \
--model_name tiny_LLaMA_120M \
--model_only false