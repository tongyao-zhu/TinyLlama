VERSION=$1
search_type=$2
query_type=$3
SAVE_PATH=/home/aiops/zhuty/ret_pretraining_data/id_added/$VERSION/traversal_paths/$search_type\_$query_type

# make dir if not exists
mkdir -p $SAVE_PATH
echo "save path: $SAVE_PATH"
for node_selection in "random" "min_degree" "max_degree" ; do
  for k in  10 100 ; do
    degree_measure="all"
    for max_length in 21 100 1000000  ; do
      echo "k=$k, node_selection=$node_selection, degree_measure=$degree_measure"
      echo "max_length=$max_length"
      python graph_traversal.py \
      --adj_list_file /home/aiops/zhuty/ret_pretraining_data/id_added/$VERSION/adj_lists/$search_type/adj_lst_top_100.json \
      --node_selection $node_selection \
      --degree_measure $degree_measure \
      --train_data_dir /home/aiops/zhuty/ret_pretraining_data/id_added/$VERSION/train \
      --undirected \
      --top_k $k \
      --save_dir $SAVE_PATH \
      --max_path_length $max_length
    done
  done
done

# the scripts takes around 5 * 3 * 3 * 2 = 90 minutes to run