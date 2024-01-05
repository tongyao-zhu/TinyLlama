export JAVA_HOME=/home/aiops/zhuty/jdk-11.0.20
export PATH=$PATH:$JAVA_HOME/bin

CHUNK_NUM=$1
echo "chunk num" $CHUNK_NUM
QUERY_PATH=/home/aiops/zhuty/ret_pretraining_data/redpajama_20b_id_added/queries/chunk_$CHUNK_NUM.jsonl
OUT_PATH=/home/aiops/zhuty/ret_pretraining_data/redpajama_20b_id_added/bm25_search_results/chunk_$CHUNK_NUM.result.txt
INDEX_PATH=/home/aiops/zhuty/ret_pretraining_data/redpajama_20b_id_added/bm25_index

#QUERY_PATH=/home/aiops/zhuty/ret_pretraining_data/redpajama_2b_id_added/queries/chunk_$CHUNK_NUM.jsonl
#OUT_PATH=/home/aiops/zhuty/ret_pretraining_data/redpajama_2b_id_added/bm25_search_results/chunk_$CHUNK_NUM.result.txt
#INDEX_PATH=/home/aiops/zhuty/ret_pretraining_data/redpajama_2b_id_added/bm25_index

python -m pyserini.search.lucene \
  --index $INDEX_PATH \
  --topics $QUERY_PATH \
  --output $OUT_PATH \
  --bm25 \
  --batch-size 768 \
  --threads 96 \
  --hits 100
