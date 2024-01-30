DSNAME=cc
export JAVA_HOME=/home/aiops/zhuty/jdk-11.0.20
export PATH=$PATH:$JAVA_HOME/bin


# for chunk_num in {0..7} ; do
for chunk_num in {6..6} ; do
echo "chunk num" $chunk_num
  embedding_path=/s3/ret_pretraining_data/id_added/cc_dense_embeddings/dense_embeddings/shard_$chunk_num
  index_path=/home/aiops/zhuty/ret_pretraining_data/id_added/$DSNAME/dense_index/shard_$chunk_num
  python -m pyserini.index.faiss --input $embedding_path --output $index_path &
done
