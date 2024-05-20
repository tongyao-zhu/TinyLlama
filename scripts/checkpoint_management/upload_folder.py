import os
import json
import sys
from huggingface_hub import HfApi
from datetime import datetime

local_path = sys.argv[1]
# check if the second argument is provided, if not,set it to 'model' as the default
if len(sys.argv) <= 2:
    repo_type = "model"
else:
    repo_type = sys.argv[2]

# replace all "/" with "_"
repo_path = 'tyzhu/' + local_path.replace("/", "_")[:64]
api = HfApi()

api.create_repo(
    repo_id=repo_path,
    token = json.load(open("/home/aiops/zhuty/hf_token.json")),
    private=True,
    repo_type=repo_type,
    exist_ok=True,
)


for file in os.listdir(local_path):
    # if it is a folder, upload the folder
    if os.path.isdir(os.path.join(local_path, file)):
        message = "Uploading folder {} to hf".format(local_path) + "tyzhu/{}".format(repo_path)[:64]
        message += "at time {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("Uploading folder {} to hf".format(file) + "tyzhu/{}".format(repo_path)[:64])
        api.upload_folder(
            folder_path=os.path.join(local_path, file),
            path_in_repo=file,
            repo_id=repo_path,
            repo_type=repo_type,
            token = json.load(open("/home/aiops/zhuty/hf_token.json")),
            commit_message=message,
        )
    else:
        print("Uploading file {} to hf".format(file) + "tyzhu/{}".format(repo_path)[:64])
        message = "Uploading file {} to hf".format(local_path) + "tyzhu/{}".format(repo_path)[:64]
        message += "at time {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        # if it is a file, upload the file
        api.upload_file(
            path_or_fileobj=os.path.join(local_path, file),
            path_in_repo=file,
            repo_id=repo_path,
            repo_type=repo_type,
            token = json.load(open("/home/aiops/zhuty/hf_token.json")),
            commit_message=message,
        )