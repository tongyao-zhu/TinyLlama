

ds, upper, step = 'cc', 100, 2
for i in range(0, upper, step):
    start = i
    end = i + step -1
    print(f"""sailctl job create elsnr{start}to{end}  -g 1 --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/elsearch ; bash search_elastic.sh {start} {end}" """)