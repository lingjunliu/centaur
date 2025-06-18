import sys, os
from utils.misc import get_tmp_dir, create_subdir, get_dir_in_root
from utils.new_api_utils import get_lib_version
from utils.coverage_utils import get_cov_torch

def main():
    # To save lcov files for debugging, set this to true:
    save_lcov = False
    # If you also need html and text formats, save gen_html to True
    gen_html = False

    api = sys.argv[1]
    save_lcov = int(sys.argv[2]) == 1 if len(sys.argv) > 2 else False

    lib = "torch"
    api = get_lib_version(api, lib=lib)
    prefix = api.replace(".", "_")

    # Directory containing the input files
    tmp = get_tmp_dir()
    cov_results = create_subdir(tmp, "coverage_results")
    out_file = os.path.join(cov_results, f"{api}.csv")
    eval = get_dir_in_root("eval")

    num_branches, num_lines, return_code, coverage_dict = get_cov_torch(f"python {eval}/patched_drivers/{api}/{api}_cov_in_loop.py", prefix=prefix, capture_output=True, gen_html=gen_html, save_lcov=save_lcov)

    with open(out_file, "w") as f:
        f.write(f"{api},{num_branches},{num_lines},{return_code}\n")
    
    if return_code != 0:
        print(f"ERROR: Execution for {api} failed and returned {return_code}")
    print(f"Coverage results for {api}:")
    print(f"Branch coverage: {num_branches}")
    print(f"Line coverage: {num_lines}")
    print(f"Coverage results saved to {out_file}")

if __name__ == "__main__":
    main()