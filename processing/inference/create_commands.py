model_name = "tiny_LLaMA_120M_8k_cc_8k"

lower, upper, step = 0, 99, 2
for i in range(lower, upper, step):
    start = i
    end = i + step -1
    if model_name == "tiny_LLaMA_120M_8k_cc_8k":
        name = '120m'
    elif model_name == "tiny_LLaMA_360M_8k_cc_8k":
        name = '360m'
    string = f"""sleep 5; sailctl job create geninf{name}{start}to{end}  -g 1 --command-line-args 'source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/inference ; bash run_batch_inf.sh /home/aiops/zhuty/tinyllama/out/{model_name}/hf_ckpt cc last {start} {end}' """
    print(string)