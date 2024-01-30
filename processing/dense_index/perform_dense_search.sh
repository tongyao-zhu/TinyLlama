

export JAVA_HOME=/home/aiops/zhuty/jdk-11.0.20
export PATH=$PATH:$JAVA_HOME/bin

DATASET_NAME=$1
QUERY_TYPE=$2
CHUNK_NUM=$3

INDEX_PATH=/home/aiops/zhuty/ret_pretraining_data/id_added/$DATASET_NAME/dense_index/shard_full
QUERY_PATH=/home/aiops/zhuty/ret_pretraining_data/id_added/$DATASET_NAME/queries/$QUERY_TYPE/chunk_$CHUNK_NUM.jsonl
OUT_PATH=/home/aiops/zhuty/ret_pretraining_data/id_added/$DATASET_NAME/dense_search_results/$QUERY_TYPE/chunk_$CHUNK_NUM.result.txt
# if it exists, skip
if [ -f "$OUT_PATH" ]; then
  echo "$OUT_PATH exists, skip"
  exit 0
fi

python -m pyserini.search.faiss \
  --index $INDEX_PATH \
  --topics $QUERY_PATH \
  --output  $OUT_PATH \
  --encoder facebook/contriever \
  --encoded-queries /home/aiops/zhuty/ret_pretraining_data/id_added/$DATASET_NAME/dense_queries/$QUERY_TYPE/chunk_$CHUNK_NUM \
  --batch-size 1024 --threads 96 --hits 100 \
  --device cuda:0


#   --encoded-queries /home/aiops/zhuty/ret_pretraining_data/id_added/$DATASET_NAME/dense_queries/chunk_$CHUNK_NUM \
# --nprobes 64