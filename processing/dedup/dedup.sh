DATASET_NAME=$1


python minhash_deduplication.py --dataset /home/aiops/zhuty/ret_pretraining_data/$DATASET_NAME \
    --split train \
    --column text \
    --cache-dir .cache \
    --min-ngram-size 5