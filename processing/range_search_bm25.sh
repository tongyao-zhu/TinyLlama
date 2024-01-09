export JAVA_HOME=/home/aiops/zhuty/jdk-11.0.20
export PATH=$PATH:$JAVA_HOME/bin
VERSION=$1
left=$2
right=$3
for chunk_num in $(seq $left $right); do
  echo "chunk num" $chunk_num
  bash search_bm25_general.sh $chunk_num $VERSION ;
done
