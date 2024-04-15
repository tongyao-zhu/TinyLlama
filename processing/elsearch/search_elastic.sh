#!/bin/bash

# check if there are enough cores
num_cores=$(nproc)
if [ "$num_cores" -lt 63 ]; then
    echo "Not enough cores to run this script. Exiting."
    exit 1
fi

# Assign the first argument passed to the script to curr_split
curr_split=$1
query_type=$2

# Calculate lower by multiplying curr_split by 5
lower=$((curr_split * 5))

# Calculate upper by adding 4 to lower
upper=$((lower + 4))
# upper=$lower
echo "Num_cores: $num_cores"
# Print the results
echo "Curr_split: $curr_split"
echo "Lower: $lower"
echo "Upper: $upper"


# start the index
cd /home/aiops/zhuty/elasticsearch-8.12.1-$curr_split
./bin/elasticsearch &
sleep 180
cd /home/aiops/zhuty/tinyllama/processing/elsearch

for i in $(seq $lower $upper) ; do
    echo "Processing $i"
    tmux new -d -s "r$i" "python search.py --chunk_num $i --index_name cc_index_first --query_type $query_type ; sleep 30 "
done

# Loop to wait for all sessions to finish
while true; do
    # Count the number of sessions that match our naming pattern
    count=$(tmux ls | grep -E '^r[0-9]+:' | wc -l)

    # If count is 0, all sessions are done
    if [ "$count" -eq 0 ]; then
        echo "All tmux sessions have completed."
        break
    else
        echo "$count sessions still running..."
    fi

    # Wait for a short period before checking again
    sleep 10
done