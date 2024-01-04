#!/bin/bash

# Assuming your list of job names is stored in a variable named 'job_list'
job_list=$(sailctl job list | grep bm25 | awk '{print $1}')

# Iterate over each job name
for job_name in $job_list; do
    # Execute the delete command for each job
    sailctl job delete  $job_name
done
