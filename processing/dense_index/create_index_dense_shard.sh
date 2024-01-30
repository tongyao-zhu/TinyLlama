VERSION=$1
SHARDNUM=$2
export CUDA_VISIBLE_DEVICES=0
TRAIN_DATA_DIR=/home/aiops/zhuty/ret_pretraining_data/id_added/$VERSION/train
OUT_EMBEDDING_DIR=/home/aiops/zhuty/ret_pretraining_data/id_added/$VERSION/dense_embeddings/shard_$SHARDNUM
python -m pyserini.encode \
  input   --corpus $TRAIN_DATA_DIR \
          --fields text \
          --delimiter "sjmkijnvnszhu31dlxjnvie394uss92cnsnn3&&s3" \
          --shard-id $SHARDNUM \
          --shard-num 8 \
  output  --embeddings $OUT_EMBEDDING_DIR \
  encoder --encoder facebook/contriever \
          --fields text \
          --batch 196