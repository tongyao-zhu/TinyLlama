import multiprocessing
import time
import psutil
import os

def cpu_intensive_task(stop_file):
    while not os.path.exists(stop_file):
        pass  # Continue the infinite loop
    print("Stop file detected. Exiting process.")

if __name__ == '__main__':
    stop_file = "stop.txt"  # File to check to stop the script
    processes = []
    num_cores = multiprocessing.cpu_count()

    for _ in range(num_cores):
        p = multiprocessing.Process(target=cpu_intensive_task, args=(stop_file,))
        p.start()
        processes.append(p)

    try:
        while not os.path.exists(stop_file):
            cpu_percent = psutil.cpu_percent(interval=1)
            print(f"Total CPU Utilization: {cpu_percent}%")
            # You can do something with cpu_percent here
            time.sleep(1)
    finally:
        print("Stopping all processes...")
        for p in processes:
            p.terminate()  # Terminate all processes
        for p in processes:
            p.join()  # Wait for processes to terminate
        os.remove(stop_file)  # Clean up stop file
        print("All processes stopped.")
