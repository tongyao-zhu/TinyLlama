from huggingface_hub import snapshot_download

# snapshot_download(repo_id= "MBZUAI-LLM/SlimPajama-627B-DC",
#                 allow_patterns=["validation/RedPajamaCommonCrawl*"],
#                 repo_type='dataset',
#                 local_dir="/home/aiops/zhuty/ret_pretraining_data/cc",)

for i in range(0,100):
    snapshot_download(repo_id= "MBZUAI-LLM/SlimPajama-627B-DC",
                allow_patterns=["train/RedPajamaCommonCrawl/chunk_{i}.jsonl.zst".format(i=i)],
                repo_type='dataset',
                local_dir="/home/aiops/zhuty/ret_pretraining_data/cc",)