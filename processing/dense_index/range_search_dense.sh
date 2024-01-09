export JAVA_HOME=/home/aiops/zhuty/jdk-11.0.20
export PATH=$PATH:$JAVA_HOME/bin
left=$1
right=$2
DATASET_NAME=$3
for chunk_num in $(seq $left $right); do
  echo "chunk num" $chunk_num
  bash perform_dense_search.sh $chunk_num $DATASET_NAME ;
done
