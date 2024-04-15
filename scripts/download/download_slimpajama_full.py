from huggingface_hub import snapshot_download

# snapshot_download(repo_id= "MBZUAI-LLM/SlimPajama-627B-DC",
#                 allow_patterns=["validation/RedPajamaCommonCrawl*"],
#                 repo_type='dataset',
#                 local_dir="/home/aiops/zhuty/ret_pretraining_data/cc",)
chunk_num = 1
snapshot_download(repo_id= "cerebras/SlimPajama-627B",
                allow_patterns=[f"train/chunk{chunk_num}*"],
                repo_type='dataset',
                local_dir=f"/home/aiops/zhuty/ret_pretraining_data/spchunk{chunk_num}",)