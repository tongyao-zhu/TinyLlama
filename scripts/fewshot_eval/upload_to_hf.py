import sys
import json
from transformers import LlamaForCausalLM, LlamaTokenizer

model_path = sys.argv[1]
# for model_path in [
#     '/home/aiops/zhuty/tinyllama/out/tiny_LLaMA_1b_8k_intramask_cc_8k/iter-380000-ckpt-step-47500_hf',
#     # '/home/aiops/zhuty/tinyllama/out/tiny_LLaMA_1b_8k_cc_8k/iter-380000-ckpt-step-47500_hf',
#     # '/home/aiops/zhuty/tinyllama/out/tiny_LLaMA_1b_8k_cc_merged_v1_8k/iter-380000-ckpt-step-47500_hf',
#     # '/home/aiops/zhuty/tinyllama/out/tiny_LLaMA_1b_8k_cc_merged_v2_8k/iter-380000-ckpt-step-47500_hf',
#     # '/home/aiops/zhuty/tinyllama/out/tiny_LLaMA_1b_8k_cc_merged_v3_8k/iter-380000-ckpt-step-47500_hf',
# ]:
tokenizer = LlamaTokenizer.from_pretrained('/home/aiops/zhuty/tinyllama/models' , padding_side='left', truncation_side="left")
tokenizer.pad_token = tokenizer.eos_token
print("Input model path", model_path)
model = LlamaForCausalLM.from_pretrained(model_path)
new_name = "_".join(model_path.split("/")[-2:])
print("Pushing to hub", new_name)
model.push_to_hub(new_name, token = json.load(open("/home/aiops/zhuty/hf_token.json")), private=True)
tokenizer.push_to_hub(new_name, token = json.load(open("/home/aiops/zhuty/hf_token.json")))

# python upload_to_hf.py /s3/tinyllama/out_mar13/out/tiny_LLaMA_360M_8k_cc_8k/iter-160000-ckpt-step-40000_hf
# python upload_to_hf.py /s3/tinyllama/out_mar13/out/tiny_LLaMA_360M_8k_cc_merged_v1_8k/iter-160000-ckpt-step-40000_hf
# python upload_to_hf.py /s3/tinyllama/out_mar13/out/tiny_LLaMA_360M_8k_cc_merged_v2_8k/iter-160000-ckpt-step-40000_hf
# python upload_to_hf.py /s3/tinyllama/out_mar13/out/tiny_LLaMA_360M_8k_cc_merged_v3_8k/iter-160000-ckpt-step-40000_hf
