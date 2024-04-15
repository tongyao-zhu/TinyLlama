from huggingface_hub import snapshot_download

# snapshot_download(repo_id= "MBZUAI-LLM/SlimPajama-627B-DC",
#                 allow_patterns=["validation/RedPajamaCommonCrawl*"],
#                 repo_type='dataset',
#                 local_dir="/home/aiops/zhuty/ret_pretraining_data/cc",)

for ds_name in ['ArXiv', 'Book']:
    snapshot_download(repo_id= "MBZUAI-LLM/SlimPajama-627B-DC",
                allow_patterns=[f"validation/RedPajama{ds_name}*"],
                repo_type='dataset',
                local_dir=f"/home/aiops/zhuty/ret_pretraining_data/{ds_name.lower()}",)