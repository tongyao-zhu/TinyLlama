

export JAVA_HOME=/home/aiops/zhuty/jdk-11.0.20
export PATH=$PATH:$JAVA_HOME/bin


CHUNK_NUM=$1
DATASET_NAME=$2

QUERY_PATH=/home/aiops/zhuty/ret_pretraining_data/id_added/$DATASET_NAME/queries/chunk_$CHUNK_NUM.jsonl
OUT_PATH=/home/aiops/zhuty/ret_pretraining_data/id_added/$DATASET_NAME/dense_queries/chunk_$CHUNK_NUM

python encode_queries.py \
  --topics $QUERY_PATH \
  --output  $OUT_PATH \
  --encoder facebook/contriever \
  --device cuda:0