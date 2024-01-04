from huggingface_hub import snapshot_download
import  json
import os
from typing import Optional
from urllib.request import urlretrieve
from huggingface_hub import snapshot_download


def download_from_hub(repo_id: Optional[str] = None, local_dir: str = "/home/aiops/zhuty/checkpoints/hf-llama/7B") -> None:
    # download from github
    if repo_id is None:
        raise ValueError("Please pass `--repo_id=...`. You can try googling 'huggingface hub llama' for options.")

    hf_token = json.load(open("/home/aiops/zhuty/hf_token.json"))

    snapshot_download(repo_id, local_dir=local_dir, local_dir_use_symlinks=False, token = hf_token)

if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(download_from_hub)
