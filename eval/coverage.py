import sys, os
from utils.misc import get_tmp_dir, read_pkl, map_torch_to_driver, create_subdir
from utils.coverage_utils import gen_cov_torch
from utils.process_lcov import analyze_lcov

def main():
    api = sys.argv[1]

    # Directory containing the input files
    tmp = get_tmp_dir()
    cov_results = create_subdir(tmp, "coverage_results")
    out_file = os.path.join(cov_results, f"{api}.csv")

    torch_to_driver, driver_to_torch = map_torch_to_driver()
    torch_api = driver_to_torch[api]

    return_code, lcov_data, memory_error = gen_cov_torch(api, f"-m eval.patched_drivers.{api}_cov_in_loop", cpu=True, capture_output=True, is_snippet=True)

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
        f.write(f"{api},{branch_cov},{line_cov}\n")
    
    print(f"Coverage results for {api}:")
    print(f"Branch coverage: {branch_cov}")
    print(f"Line coverage: {line_cov}")
    print(f"Coverage results saved to {out_file}")

if __name__ == "__main__":
    main()