MODEL_NAME=$1

export PYTHONPATH='$PYTHONPATH:/home/aiops/zhuty/tinyllama'
echo "MODEL_NAME: $MODEL_NAME"
CHUNK=1000
MODEL_NAME="/home/aiops/zhuty/tinyllama/out/tiny_LLaMA_1b_8k_$MODEL_NAME\_8k/iter-380000-ckpt-step-47500.pth"
DS_NAME="/home/aiops/zhuty/ret_pretraining_data/cc/valid/"
echo "MODEL_NAME: $MODEL_NAME" "DS_NAME: $DS_NAME"
python scripts/eval_ppl/evaluate_litgpt_ppl.py --model  $MODEL_NAME \
    --chunk_n $CHUNK --dataset $DS_NAME

#for CHUNK in {0..628}; do
#    echo "CHUNK: $CHUNK"
#    MODEL_NAME="/home/aiops/zhuty/tinyllama/out/tiny_LLaMA_1b_8k_$MODEL_NAME\_8k/iter-380000-ckpt-step-47500.pth"
#    DS_NAME="/home/aiops/zhuty/ret_pretraining_data/cc/valid/chunk_$CHUNK.jsonl"
#    echo "MODEL_NAME: $MODEL_NAME" "DS_NAME: $DS_NAME"
#
#    python scripts/eval_ppl/evaluate_litgpt_ppl.py --model  $MODEL_NAME \
#    --chunk_n $CHUNK --dataset $DS_NAME
#done
