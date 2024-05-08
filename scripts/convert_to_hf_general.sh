
LIT_CKPT_DIR=$1
CKPT_PREFIX=$2

# if tiny_LLaMA_120M_8k in the model_name, set model name
if [[ $LIT_CKPT_DIR == *"tiny_LLaMA_120M_8k"* ]]; then
  MODEL_NAME=tiny_LLaMA_120M_8k
elif [[ $LIT_CKPT_DIR == *"tiny_LLaMA_360M_8k"* ]]; then
  MODEL_NAME=tiny_LLaMA_360M_8k
elif [[ $LIT_CKPT_DIR == *"tiny_LLaMA_1b_8k"* ]]; then
  MODEL_NAME=tiny_LLaMA_1b_8k
else
  echo "MODEL_NAME not found"
  exit 1
fi

OUTPUT_DIR=$LIT_CKPT_DIR/$CKPT_PREFIX\_hf
mkdir -p $OUTPUT_DIR
echo "OUTPUT_DIR $OUTPUT_DIR"

python convert_lit_checkpoint.py \
--checkpoint_name $CKPT_PREFIX.pth \
--out_dir $LIT_CKPT_DIR \
--model_name $MODEL_NAME \
--model_only false


# move config
mv $LIT_CKPT_DIR/config.json $OUTPUT_DIR/config.json
mv $LIT_CKPT_DIR/$CKPT_PREFIX.bin $OUTPUT_DIR/pytorch_model.bin

# sample usage
# bash convert_to_hf_general.sh /s3/tinyllama/out_mar13/out/tiny_LLaMA_1b_8k_cc_merged_v3_8k iter-380000-ckpt-step-47500
# bash convert_to_hf_general.sh /home/aiops/zhuty/tinyllama/out/tiny_LLaMA_1b_8k_cc_merged_v2_8k iter-380000-ckpt-step-47500
# bash convert_to_hf_general.sh /home/aiops/zhuty/tinyllama/out/tiny_LLaMA_1b_8k_cc_merged_v1_8k iter-380000-ckpt-step-47500
# bash convert_to_hf_general.sh /home/aiops/zhuty/tinyllama/out/tiny_LLaMA_1b_8k_cc_8k iter-380000-ckpt-step-47500

# bash convert_to_hf_general.sh /home/aiops/zhuty/tinyllama/out/tiny_LLaMA_360M_8k_cc_8k iter-110000-ckpt-step-27500
# bash convert_to_hf_general.sh /home/aiops/zhuty/tinyllama/out/tiny_LLaMA_360M_8k_intramask_cc_8k iter-110000-ckpt-step-27500
# bash convert_to_hf_general.sh /home/aiops/zhuty/tinyllama/out/tiny_LLaMA_360M_8k_cc_merged_v2_8k iter-110000-ckpt-step-27500
# bash convert_to_hf_general.sh /home/aiops/zhuty/tinyllama/out/tiny_LLaMA_360M_8k_intramask_cc_merged_v2_8k iter-110000-ckpt-step-27500
# bash convert_to_hf_general.sh /home/aiops/zhuty/tinyllama/out/tiny_LLaMA_360M_8k_adamask_cc_merged_v2_8k iter-110000-ckpt-step-27500
# bash convert_to_hf_general.sh /home/aiops/zhuty/tinyllama/out/tiny_LLaMA_360M_8k_intramask_cc_merged_v2_8k iter-160000-ckpt-step-40000

# bash convert_to_hf_general.sh /s3/tinyllama/out_mar13/out/tiny_LLaMA_360M_8k_cc_merged_v1_8k iter-160000-ckpt-step-40000
# bash convert_to_hf_general.sh /s3/tinyllama/out_mar13/out/tiny_LLaMA_360M_8k_cc_merged_v2_8k iter-160000-ckpt-step-40000
# bash convert_to_hf_general.sh /s3/tinyllama/out_mar13/out/tiny_LLaMA_360M_8k_cc_merged_v3_8k iter-160000-ckpt-step-40000
# bash convert_to_hf_general.sh /s3/tinyllama/out_mar13/out/tiny_LLaMA_360M_8k_cc_8k iter-160000-ckpt-step-40000

# bash convert_to_hf_general.sh /s3/tinyllama/out_mar13/out/tiny_LLaMA_120M_8k_cc_merged_v1_8k iter-240000-ckpt-step-60000
# bash convert_to_hf_general.sh /s3/tinyllama/out_mar13/out/tiny_LLaMA_120M_8k_cc_merged_v2_8k iter-240000-ckpt-step-60000
# bash convert_to_hf_general.sh /s3/tinyllama/out_mar13/out/tiny_LLaMA_120M_8k_cc_merged_v3_8k iter-240000-ckpt-step-60000
# bash convert_to_hf_general.sh /s3/tinyllama/out_mar13/out/tiny_LLaMA_120M_8k_cc_8k iter-240000-ckpt-step-60000


#for ds in cc cc_merged_v1 cc_merged_v2 cc_merged_v3 ; do
##  size=120M
##  step=iter-240000-ckpt-step-60000
#  size=360M
#  step=iter-160000-ckpt-step-40000
#  python upload_to_hf.py /s3/tinyllama/out_mar13/out/tiny_LLaMA_$size\_8k_$ds\_8k/$step\_hf
#done