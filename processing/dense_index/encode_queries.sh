

export JAVA_HOME=/home/aiops/zhuty/jdk-11.0.20
export PATH=$PATH:$JAVA_HOME/bin

DATASET_NAME=$1
QUERY_TYPE=$2
CHUNK_NUM_LOWER=$3
CHUNK_NUM_UPPER=$4

mkdir /home/aiops/zhuty/ret_pretraining_data/id_added/$DATASET_NAME/dense_queries/$QUERY_TYPE

for CHUNK_NUM in $(seq $CHUNK_NUM_LOWER $CHUNK_NUM_UPPER); do
  QUERY_PATH=/home/aiops/zhuty/ret_pretraining_data/id_added/$DATASET_NAME/queries/$QUERY_TYPE/chunk_$CHUNK_NUM.jsonl
  OUT_PATH=/home/aiops/zhuty/ret_pretraining_data/id_added/$DATASET_NAME/dense_queries/$QUERY_TYPE/chunk_$CHUNK_NUM
  # if output path is a directory, skip
  if [ -d "$OUT_PATH" ]; then
    echo "$OUT_PATH exists, skip"
    continue
  fi
  python encode_queries.py \
    --topics $QUERY_PATH \
    --output  $OUT_PATH \
    --encoder facebook/contriever \
    --device cuda:0
done
