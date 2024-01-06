

export JAVA_HOME=/home/aiops/zhuty/jdk-11.0.20
export PATH=$PATH:$JAVA_HOME/bin


CHUNK_NUM=$1

INDEX_PATH=/home/aiops/zhuty/ret_pretraining_data/redpajama_2b_id_added/dense_index/shard_full
QUERY_PATH=/home/aiops/zhuty/ret_pretraining_data/redpajama_2b_id_added/queries/chunk_$CHUNK_NUM.jsonl
OUT_PATH=/home/aiops/zhuty/ret_pretraining_data/redpajama_2b_id_added/dense_search_results/chunk_$CHUNK_NUM.result.txt

python -m pyserini.search.faiss \
  --index $INDEX_PATH \
  --topics $QUERY_PATH \
  --output  $OUT_PATH \
  --encoder facebook/contriever \
  --batch-size 256 --threads 96 --hits 100 \
  --device cuda:0