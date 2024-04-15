import pandas as pd
import sys
import json
import os
mode=sys.argv[1]

if mode == '120M':
    ckpts = ['iter-080000-ckpt-step-20000', 'iter-160000-ckpt-step-40000',
             'iter-240000-ckpt-step-60000', 'iter-320000-ckpt-step-80000','iter-400000-ckpt-step-100000']

elif mode == '360M':
    ckpts = ['iter-080000-ckpt-step-20000', 'iter-120000-ckpt-step-30000',
                'iter-160000-ckpt-step-40000', 'iter-190000-ckpt-step-47500']

elif mode == '1b':
    ckpts= ['iter-160000-ckpt-step-20000','iter-200000-ckpt-step-25000', 'iter-240000-ckpt-step-30000',
    'iter-300000-ckpt-step-37500']

result_list = []
for ds in ['cc_8k', 'cc_merged_v1_8k', 'cc_merged_v2_8k']:
    for ckpt in ckpts:
        result_dir = f'/s3/tinyllama/out_feb27_sg/out/tiny_LLaMA_{mode}_8k_{ds}/{ckpt}_hf'
        results = json.load(open(os.path.join(result_dir, 'results.json')))
        for eval_ppl_ds in ("arxiv", "book", "cc", "rpwiki_en") :
          file = os.path.join(result_dir, f"eval_{eval_ppl_ds}_8k.log")
          results[f"eval_loss_{eval_ppl_ds}"] = json.load(open(file))['val_loss']
        results['name'] = f'{mode}_{ds}_{ckpt}'
        results['ds'] = ds
        result_list.append(results)
df = pd.DataFrame(result_list)
print(df.head())
print(len(df))
df.to_csv(f'tiny_LLaMA_{mode}_8k_results.csv')



