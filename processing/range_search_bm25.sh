export JAVA_HOME=/home/aiops/zhuty/jdk-11.0.20
export PATH=$PATH:$JAVA_HOME/bin
left=$1
right=$2
for chunk_num in $(seq $left $right); do
  echo "chunk num" $chunk_num
  bash processing/search_bm25.sh $chunk_num ;
done
