import sys, os
from utils.misc import get_tmp_dir, create_subdir, get_dir_in_root
from utils.new_api_utils import get_lib_version
from utils.coverage_utils import get_coverage_numbers

def compute_coverage(api, cov_results, driver_file, lib="torch", save_lcov=False, gen_html=False, native_only=False, timeout=None):
    prefix = api.replace(".", "_")

    # Directory containing the input files
    out_file = os.path.join(cov_results, f"{api}_{lib}.csv")
    num_branches, num_lines, return_code, coverage_dict = get_coverage_numbers(f"python {driver_file}", lib=lib, prefix=prefix, capture_output=True, gen_html=gen_html, gen_lcov=not gen_html, save_lcov=save_lcov, native_only=native_only, timeout=timeout)

    if return_code > 0:
        print(f"ERROR: Execution for {api} failed and returned {return_code}")
    elif return_code < 0:
        print(f"[CRASH] Raised a signal: {return_code}")
    else:
        with open(out_file, "w") as f:
            f.write(f"{api},{num_branches},{num_lines}\n")
    
    print(f"Coverage results for {api}:")
    print(f"Branch coverage: {num_branches}")
    print(f"Line coverage: {num_lines}")
    print(f"Coverage results saved to {out_file}")
    

def main():
    api = sys.argv[1]
    lib = sys.argv[2] if len(sys.argv) > 2 else "torch"
    gen_html = sys.argv[3].lower() == "html" if len(sys.argv) > 3 else True
    native_only = sys.argv[4].lower() == "true" if len(sys.argv) > 4 else False
    save_lcov = int(sys.argv[5]) == 1 if len(sys.argv) > 5 else False
    timeout = int(sys.argv[6]) if len(sys.argv) > 6 else None
    
    api = get_lib_version(api, lib=lib)
    
    # Directory containing the input files
    cov_results = create_subdir(get_tmp_dir(), "coverage_results")
    eval = get_dir_in_root("eval")
    driver_file = f"{eval}/patched_drivers/{api}/{api}_cov_in_loop.py"
    
    if not os.path.exists(driver_file):
        print(f"{driver_file} does not exist")
        return
    
    compute_coverage(api, cov_results, driver_file, lib=lib, save_lcov=save_lcov, gen_html=gen_html, native_only=native_only, timeout=timeout)

if __name__ == "__main__":
    main()