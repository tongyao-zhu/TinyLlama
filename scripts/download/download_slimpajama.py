from huggingface_hub import snapshot_download

snapshot_download(repo_id= "MBZUAI-LLM/SlimPajama-627B-DC",
                allow_patterns=["*RedPajamaWikipedia*"],
                repo_type='dataset',
                local_dir="/home/aiops/zhuty/ret_pretraining_data/rpwiki",)