export JAVA_HOME=/home/aiops/zhuty/jdk-11.0.20
export PATH=$PATH:$JAVA_HOME/bin
DATA_VERSION=$1
QUERY_VERSION=$2
left=$3
right=$4
for chunk_num in $(seq $left $right); do
  echo "chunk num" $chunk_num
  bash search_bm25_general.sh $chunk_num $DATA_VERSION $QUERY_VERSION;
done
