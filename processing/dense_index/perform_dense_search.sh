

export JAVA_HOME=/home/aiops/zhuty/jdk-11.0.20
export PATH=$PATH:$JAVA_HOME/bin


CHUNK_NUM=$1
DATASET_NAME=$2

INDEX_PATH=/home/aiops/zhuty/ret_pretraining_data/id_added/$DATASET_NAME/dense_index/shard_full
QUERY_PATH=/home/aiops/zhuty/ret_pretraining_data/id_added/$DATASET_NAME/queries/chunk_$CHUNK_NUM.jsonl
OUT_PATH=/home/aiops/zhuty/ret_pretraining_data/id_added/$DATASET_NAME/dense_search_results/chunk_$CHUNK_NUM.result.txt

python -m pyserini.search.faiss \
  --index $INDEX_PATH \
  --topics $QUERY_PATH \
  --output  $OUT_PATH \
  --encoder facebook/contriever \
  --encoded-queries /home/aiops/zhuty/ret_pretraining_data/id_added/$DATASET_NAME/dense_queries/chunk_$CHUNK_NUM \
  --batch-size 128 --threads 96 --hits 100 \
  --device cuda:0