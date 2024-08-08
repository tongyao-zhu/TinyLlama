import os
BASE_PATH = "/home/aiops/zhuty/tinyllama/scripts/fewshot_eval/outputs"
NEW_PATH = "/home/aiops/zhuty/tinyllama/scripts/fewshot_eval/prompts_and_preds"
# move the pred file to another path
for sub_dir in os.listdir(BASE_PATH)[:]:
    # skip if not a directory
    if not os.path.isdir(os.path.join(BASE_PATH, sub_dir)):
        continue
    print("Moving files in ", sub_dir)
    files = os.listdir(os.path.join(BASE_PATH, sub_dir))
    preds_files = [f for f in files if f.endswith("prompts_and_preds.json")]
    new_dir = os.path.join(NEW_PATH, sub_dir)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    for f in preds_files:
        # ensure that new file does not exist
        if not os.path.exists(os.path.join(new_dir, f)):
            os.rename(os.path.join(BASE_PATH, sub_dir, f), os.path.join(new_dir, f))
        else:
            print(f"File {f} already exists in {new_dir}")