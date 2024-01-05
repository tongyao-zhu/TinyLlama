export JAVA_HOME=/home/aiops/zhuty/jdk-11.0.20
export PATH=$PATH:$JAVA_HOME/bin

#INPUT_PATH=/home/aiops/zhuty/ret_pretraining_data/redpajama_20b_id_added/train
#INDEX_PATH=/home/aiops/zhuty/ret_pretraining_data/redpajama_20b_id_added/bm25_index

INPUT_PATH=/home/aiops/zhuty/ret_pretraining_data/redpajama_2b_id_added/train
INDEX_PATH=/home/aiops/zhuty/ret_pretraining_data/redpajama_2b_id_added/bm25_index
python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input $INPUT_PATH \
  --index $INDEX_PATH \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions