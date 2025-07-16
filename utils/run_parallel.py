import os
from multiprocessing import Pool, cpu_count
import subprocess
import sys

from utils.misc import get_tmp_dir, read_file_in_root

class ProcRunner:
    def __init__(self, cmd, args, job_name):
        self.cmd = cmd
        self.args = args
        self.job_name = job_name

    def run_proc(self, element):
        cmd = f"{self.cmd} {element} {self.args}"
        return_object = subprocess.run(cmd.split(), capture_output=True)
        with open(os.path.join("logs", f"{element}_{self.job_name}.out"), "w") as f:
            f.write(return_object.stdout.decode())
            f.write(return_object.stderr.decode())
        
        return return_object.returncode

def summarize_results(result_dir, result_file):
    files = os.listdir(result_dir)
    result_str = ""
    for file in files:
        if file.endswith(".csv"):
            with open(os.path.join(result_dir, file), "r") as f:
                result_str += f.read()

    with open(result_file, "r") as f:
        header = f.readline()
        result_str = header + result_str
    
    with open(result_file, "w") as f:
        f.write(result_str)

def main():
    cmd = sys.argv[1]
    args = sys.argv[2]
    result_dir = sys.argv[3]
    result_file = sys.argv[4]
    job_name = sys.argv[5]
    num_processes = int(sys.argv[6]) if len(sys.argv) > 6 else 64
    
    if os.environ.get("elements_file"):
        elements_file = os.environ.get("elements_file")
        with open(elements_file, "r") as f:
            elements = [line.strip() for line in f.readlines()]
    else:
        elements = read_file_in_root("apis.txt")
    
    if not elements:
        print(f"No elements found")
    else:
        print(f"Found {len(elements)} elementss to process.")
        print(f"Starting parallel processing with {num_processes} workers...")
        runner = ProcRunner(cmd, args, job_name)

        count = 0
        with Pool(processes=num_processes) as pool:
            for result in pool.imap_unordered(runner.run_proc, elements):
                count += 1
                if result_file or result_dir:
                    summarize_results(result_dir, result_file)
                
                if result == 0:
                    print(f"Done with {count}/{len(elements)} {job_name} jobs. Successfully completed.")
                else:
                    print(f"Done with {count}/{len(elements)} {job_name} jobs. Faced error.")

        print("\n All elements have been processed.")

if __name__ == "__main__":
    main()