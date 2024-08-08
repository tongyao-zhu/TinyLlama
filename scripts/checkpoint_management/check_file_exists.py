import sys
import json
from huggingface_hub import file_exists

model_path=sys.argv[1]

model_path = "_".join(model_path.split("/")[-2:])
model_path = f"tyzhu/{model_path}"
print("Checking if model exists in HuggingFace Hub, model_path: ", model_path)

if not file_exists(repo_id=model_path, filename='pytorch_model.bin', token=json.load(open('/home/aiops/zhuty/hf_token.json', 'r'))):
    sys.exit(1)
else:
    sys.exit(0)
