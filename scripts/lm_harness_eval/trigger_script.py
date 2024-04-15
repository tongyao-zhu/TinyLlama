# 任务列表
tasks = (
    'agieval',
    'gsm8k',
    'qasper',
    'hellaswag',
    'mathqa',
    'mmlu',
    'piqa',
    'pubmedqa',
    'race',
    'sciq',
    'social_iqa',
    'lambada_standard',
    'openbookqa',
    'arc_easy',
    'arc_challenge',
    'winogrande',
    'mnli',
    'mrpc',
    'rte',
    'qnli',
    'qqp',
    'sst2',
    'wnli',
    'boolq',
    'copa',
    'multirc',
    'wsc',
    'wikitext',
    'logiqa',
    'scrolls'
)


def remove_underline(task_name):
    return task_name.replace("_", "")


for model_name in ['cc', 'cc_merged_v1', 'cc_merged_v2', 'cc_merged_v3']:
    # model_path = f'/home/aiops/zhuty/tinyllama/out/tiny_LLaMA_1b_8k_{model_name}_8k/iter-380000-ckpt-step-47500_hf'
    model_path = f'/home/aiops/zhuty/lm_indexer_data/tyzhu/flan_max_300_added_tyzhu_tiny_LLaMA_1b_8k_{model_name}_8k_iter-380000-ckpt-step-47500_hf/checkpoint-4129'
    for task in tasks:
        print(
            f""" sailctl job create hareval1b{remove_underline(task)}{remove_underline(model_name)} -g 1  --debug --command-line-args  ' source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/lm_harness_eval/ ; bash eval.sh {task} {model_path} ' """)
