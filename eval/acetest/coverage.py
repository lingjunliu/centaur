import sys, os
from utils.misc import get_tmp_dir, create_subdir
from utils.new_api_utils import get_lib_version
from eval.coverage import compute_coverage

def main():
    api = sys.argv[1]
    lib = sys.argv[2] if len(sys.argv) > 2 else "torch"
    skip_merge = sys.argv[3].lower() == 'true' if len(sys.argv) > 3 else False

    # Debugging flags
    gen_html = True
    save_lcov = False
    
    api = get_lib_version(api, lib=lib)
    
    # Directory containing the input files
    cov_results = create_subdir(get_tmp_dir(), "acetest_coverage")
    driver_file = os.path.join(get_tmp_dir(), "acetest_patched", api, "driver.py")
    
    if not os.path.exists(driver_file):
        print(f"{driver_file} does not exist")
        return
    
    compute_coverage(api, cov_results, driver_file, lib=lib, save_lcov=save_lcov, gen_html=gen_html, skip_merge=skip_merge)

if __name__ == "__main__":
    main()