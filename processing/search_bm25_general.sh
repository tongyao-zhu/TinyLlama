export JAVA_HOME=/home/aiops/zhuty/jdk-11.0.20
export PATH=$PATH:$JAVA_HOME/bin

CHUNK_NUM=$1
echo "chunk num" $CHUNK_NUM
DATASET_NAME=$2
QUERY_VERSION=$3
#QUERY_PATH=/home/aiops/zhuty/ret_pretraining_data/redpajama_20b_id_added/queries/chunk_$CHUNK_NUM.jsonl
#OUT_PATH=/home/aiops/zhuty/ret_pretraining_data/redpajama_20b_id_added/bm25_search_results/chunk_$CHUNK_NUM.result.txt
#INDEX_PATH=/home/aiops/zhuty/ret_pretraining_data/redpajama_20b_id_added/bm25_index

# check if output path exists
if [ ! -d "/home/aiops/zhuty/ret_pretraining_data/id_added/$DATASET_NAME/bm25_search_results/$QUERY_VERSION" ]; then
  mkdir /home/aiops/zhuty/ret_pretraining_data/id_added/$DATASET_NAME/bm25_search_results/$QUERY_VERSION
fi

QUERY_PATH=/home/aiops/zhuty/ret_pretraining_data/id_added/$DATASET_NAME/queries/$QUERY_VERSION/chunk_$CHUNK_NUM.jsonl
OUT_PATH=/home/aiops/zhuty/ret_pretraining_data/id_added/$DATASET_NAME/bm25_search_results/$QUERY_VERSION/chunk_$CHUNK_NUM.result.txt
INDEX_PATH=/home/aiops/zhuty/ret_pretraining_data/id_added/$DATASET_NAME/bm25_index

python -m pyserini.search.lucene \
  --index $INDEX_PATH \
  --topics $QUERY_PATH \
  --output $OUT_PATH \
  --bm25 \
  --batch-size 768 \
  --threads 96 \
  --hits 100
