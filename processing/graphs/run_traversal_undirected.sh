# for k in 1 3 5 10 20 ; do
for k in 100 ; do
  for node_selection in "random" "min_degree" "max_degree" ; do
    for degree_measure in "in" "out" "all" ; do
      echo "k=$k, node_selection=$node_selection, degree_measure=$degree_measure"
      python graph_traversal.py \
      --adj_list_file /home/aiops/zhuty/ret_pretraining_data/redpajama_2b_id_added/adj_lists/adj_lst_top_$k.json \
      --node_selection $node_selection \
      --degree_measure $degree_measure \
      --undirected
    done
  done
done

# the scripts takes around 5 * 3 * 3 * 2 = 90 minutes to run