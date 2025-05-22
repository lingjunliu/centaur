import sys, os, random
from utils.coverage_utils import gen_cov_torch
from utils.process_lcov import analyze_lcov
from utils.misc import map_torch_to_driver

def retain_limited_files(directory, retain_count=300, ext=".pkl"):
    pkl_files = [f for f in os.listdir(directory) if f.endswith(ext)]
    if len(pkl_files) <= retain_count:
        print(f"Less than or equal to {retain_count} {ext} files found ({len(pkl_files)}) in {directory}. No files deleted.")
        return len(pkl_files)
    
    files_to_keep = random.sample(pkl_files, retain_count)
    
    for file in pkl_files:
        if file not in files_to_keep:
            file_path = os.path.join(directory, file)
            os.remove(file_path)
    
    pkl_files = [f for f in os.listdir(directory) if f.endswith(ext)]
    return len(pkl_files)

def main():
    if len(sys.argv) < 4:
        print("Missing argument: dir, api, output file")

    dir = sys.argv[1]
    api = sys.argv[2]
    out_file = sys.argv[3]

    print(f"dir: {dir}, api: {api}, out_file: {out_file}")
    
    limit = 300
    if len(sys.argv) > 4:
        limit = int(sys.argv[4])

    torch_to_driver, driver_to_torch = map_torch_to_driver()
    torch_api = driver_to_torch[api]
    dir = f"{dir}/{torch_api}"
    n_inputs = retain_limited_files(dir, retain_count=limit, ext='.pkl')

    if n_inputs == 0:
        print(f"No input files were generated for {torch_api}")
        return
    
    print(f"{dir}/driver.py")
    return_code, lcov_data, memory_error = gen_cov_torch(driver=api, input_file=f"{dir}/driver.py {dir}", capture_output=True, is_snippet=True)
        
    if memory_error:
        print(f"Faced memory error while running on {torch_api}")

    files_dict = analyze_lcov(lcov_data)
    branch_cov = 0
    line_cov = 0
    branches_covered = {}
    for filename in files_dict:
        branches_covered[filename] = set(files_dict[filename]["branches"])
        branch_cov += len(files_dict[filename]["branches"])
        line_cov += len(files_dict[filename]["lines"])

    with open(out_file, "w") as f:
        f.write(f"{api},{branch_cov},{line_cov},{n_inputs}\n")

if __name__ == "__main__":
    main()