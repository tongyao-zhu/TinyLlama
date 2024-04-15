import sys

mode = sys.argv[1]

if mode == '120M':
    ckpts = ['iter-080000-ckpt-step-20000', 'iter-160000-ckpt-step-40000',
             'iter-240000-ckpt-step-60000', 'iter-320000-ckpt-step-80000', 'iter-400000-ckpt-step-100000']

elif mode == '360M':
    ckpts = ['iter-080000-ckpt-step-20000', 'iter-120000-ckpt-step-30000',
             'iter-160000-ckpt-step-40000', 'iter-190000-ckpt-step-47500']

elif mode == '1b':
    ckpts = ['iter-160000-ckpt-step-20000', 'iter-200000-ckpt-step-25000', 'iter-240000-ckpt-step-30000',
             'iter-300000-ckpt-step-37500']

else:
    raise ValueError(f"mode {mode} not supported")

for ds in ['cc_8k', 'cc_merged_v1_8k', 'cc_merged_v2_8k']:
    # for ckpt in ckpts:
    for ckpt in [ckpts[-1]]:
#        print(f"rsync -ar --progress /s3/tinyllama/out_feb27_sg/out/tiny_LLaMA_{mode}_8k_{ds}/{ckpt}_hf /home/aiops/zhuty/tinyllama/out/tiny_LLaMA_{mode}_8k_{ds}")

        for task in ['icl', 'mrc', 'cbqa']:
            # print(f"rsync -ar --progress /s3/tinyllama/out_feb27_sg/out/tiny_LLaMA_{mode}_8k_{ds}/{ckpt}_hf /home/aiops/zhuty/tinyllama/out/tiny_LLaMA_{mode}_8k_{ds}")
            print(
                f""" sailctl job create eval{mode.lower()}{task} -g 1 --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args  ' bash /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/scripts/fewshot_eval/ ; bash {task}.sh /home/aiops/zhuty/tinyllama/out/tiny_LLaMA_{mode}_8k_{ds}/{ckpt}_hf '
            """)