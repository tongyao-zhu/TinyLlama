import sys

# 任务列表
COMPLETE_LIST = (
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

tasks = {'group1': ['wikitext', 'lambada_standard'],
'group2': ['triviaqa', 'nq_open', 'webqs', 'hellaswag', 'squadv2'],
         'complete': COMPLETE_LIST,
         }
tasks['all'] = tasks['group1'] + tasks['group2']

def remove_underline(task_name):
    return task_name.replace("_", "")

group = sys.argv[1]
priority = sys.argv[2]
size = sys.argv[3]



size_to_versions = {
    '360M': ['cc_merged_v2_8k', 'intramask_cc_8k', 'intramask_cc_merged_v2_8k', 'adamask_cc_merged_v2_8k',
                   'cc_merged_v2_8k_intrav2cont', 'cc_8k', ],
    '1b': ['cc_merged_v2_8k', 'intramask_cc_8k',  'adamask_cc_merged_v2_8k', 'cc_merged_v2_8k_intracccont','cc_8k'],
           # 'cc_merged_v1_8k'], # 'intramask_cc_merged_v2_8k',
    'baseline' : ['cc_8k'],
    'uoe': ['BM25Chunk', 'IntraDoc', 'UniChunk', 'MixChunk'],
}

steps = range(5000, 80000, 5000)

# for model_name in ['cc', 'cc_merged_v1', 'cc_merged_v2', 'cc_merged_v3']:
for ds_version in size_to_versions[size]:
    for step in steps:
        if size == "360M":
            model_path = f'tyzhu/tiny_LLaMA_{size}_8k_{ds_version}_iter-{step * 4:06}-ckpt-step-{step}_hf'
        elif size == "1b":
            model_path = f'tyzhu/tiny_LLaMA_{size}_8k_{ds_version}_iter-{step * 8:06}-ckpt-step-{step}_hf'
        elif size =='baseline':
            model_path = f'TinyLlama/TinyLlama-1.1B-step-50K-105b'
        elif size == 'uoe':
            model_path = f'yuzhaouoe/{ds_version}'
        # model_path = f'/home/aiops/zhuty/tinyllama/out/tiny_LLaMA_{size}_8k_{ds_version}/iter-160000-ckpt-step-40000_hf'

        nshot=5
        if group == 'group1':
            nshot = 0
        # model_path = f'/home/aiops/zhuty/lm_indexer_data/tyzhu/flan_max_300_added_tyzhu_tiny_LLaMA_1b_8k_{model_name}_8k_iter-380000-ckpt-step-47500_hf/checkpoint-4129'
        for task in tasks[group]:
            jobname = f"har{size.lower()}{remove_underline(task)}{remove_underline(ds_version)}"[:40].lower() #max length 40
            print(
                f""" sailctl job create {jobname} -p {priority}  -g 1  --debug --command-line-args  ' source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/lm_harness_eval/ ; bash eval.sh {task} {model_path} {nshot}' """)