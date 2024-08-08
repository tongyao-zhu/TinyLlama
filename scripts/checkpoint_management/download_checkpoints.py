import os


# base_str = f"huggingface-cli download --local_dir /home/aiops/zhuty/upload_temp/tiny_LLaMA_1b_8k_intramask_cc_8k tyzhu/tiny_LLaMA_1b_8k_intramask_cc_8k_from_apr24 "

base_str = ""
for steps in range(55000, 80000, 5000):
    base_str += f"rsync -ar --progress iter-{steps*8:06}-ckpt-step-{steps}.pth ~/upload_temp/tiny_LLaMA_1b_8k_intramask_cc_8k ;" # iter-240000-ckpt-step-30000.pth

print(base_str)