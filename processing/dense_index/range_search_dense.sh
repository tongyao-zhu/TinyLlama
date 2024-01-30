export JAVA_HOME=/home/aiops/zhuty/jdk-11.0.20
export PATH=$PATH:$JAVA_HOME/bin
DATASET_NAME=$1
QUERY_TYPE=$2
left=$3
right=$4

for chunk_num in $(seq $left $right); do
  echo "chunk num" $chunk_num
  bash perform_dense_search.sh  $DATASET_NAME $QUERY_TYPE $chunk_num ;
done
