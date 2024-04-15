
for i in $(seq 0 99) ; do
    echo "Processing $i"
python convert_to_adj_lists.py --version cc --search_type el --query_type last_120m --chunk_num $i &
done