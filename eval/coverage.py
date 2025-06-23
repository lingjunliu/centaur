import sys, os
from utils.misc import get_tmp_dir, create_subdir, get_dir_in_root
from utils.new_api_utils import get_lib_version
from utils.coverage_utils import get_cov_torch

def compute_coverage(api, cov_results, driver_file, lib="torch", save_lcov=False, gen_html=False):
    prefix = api.replace(".", "_")

    # Directory containing the input files
    out_file = os.path.join(cov_results, f"{api}.csv")
    num_branches, num_lines, return_code, coverage_dict = get_cov_torch(f"python {driver_file}", prefix=prefix, capture_output=True, gen_html=gen_html, save_lcov=save_lcov)

    with open(out_file, "w") as f:
        f.write(f"{api},{num_branches},{num_lines},{return_code}\n")
    
    if return_code != 0:
        print(f"ERROR: Execution for {api} failed and returned {return_code}")
    print(f"Coverage results for {api}:")
    print(f"Branch coverage: {num_branches}")
    print(f"Line coverage: {num_lines}")
    print(f"Coverage results saved to {out_file}")
    

def main():
    api = sys.argv[1]
    # If you also need html and text formats, save gen_html to True
    gen_html = False
    save_lcov = int(sys.argv[2]) == 1 if len(sys.argv) > 2 else False
    lib = "torch"
    
    api = get_lib_version(api, lib=lib)
    
    # Directory containing the input files
    cov_results = create_subdir(get_tmp_dir(), "coverage_results")
    eval = get_dir_in_root("eval")
    driver_file = f"{eval}/patched_drivers/{api}/{api}_cov_in_loop.py"
    
    if not os.path.exists(driver_file):
        print(f"{driver_file} does not exist")
        return
    
    compute_coverage(api, cov_results, driver_file, lib=lib, save_lcov=save_lcov, gen_html=gen_html)

if __name__ == "__main__":
    main()