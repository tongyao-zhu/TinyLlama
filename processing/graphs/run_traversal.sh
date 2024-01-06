VERSION=$1
for k in 1 3 5 10 20 100 ; do
  for node_selection in "random" "min_degree" "max_degree" ; do
    for degree_measure in "in" "out" "all" ; do
      echo "k=$k, node_selection=$node_selection, degree_measure=$degree_measure"
      python graph_traversal.py \
      --adj_list_file /home/aiops/zhuty/ret_pretraining_data/$VERSION\_id_added/adj_lists/adj_lst_top_$k.json \
      --node_selection $node_selection \
      --degree_measure $degree_measure \
      --train_data_dir /home/aiops/zhuty/ret_pretraining_data/$VERSION\_id_added/train
    done
  done
done

# the scripts takes around 5 * 3 * 3 * 2 = 90 minutes to run