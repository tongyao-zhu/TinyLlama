export JAVA_HOME=/home/aiops/zhuty/jdk-11.0.20
export PATH=$PATH:$JAVA_HOME/bin

for chunk_num in {0..88}; do
  echo "chunk num" $chunk_num
   bash processing/search_bm25.sh $chunk_num
done
