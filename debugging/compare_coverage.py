import os, sys
from utils.process_lcov import analyze_lcov
from utils.misc import get_tmp_dir, create_subdir, get_dir_in_root
from utils.coverage_utils import gen_cov
from utils.new_api_utils import get_lib_version

def get_cov_dict_from_file(lcov_file):
    with open(lcov_file, "r") as f:
        lcov_data = f.read()
    coverage_dict = analyze_lcov(lcov_data)

    return coverage_dict

def compare_cov_dicts(original_dict, new_dict, merge=False):
    additional_branches = 0
    additional_lines = 0
    for filename, coverage_info in new_dict.items():
        if filename not in original_dict:
            print(f"\n{filename} newly added to coverage\n")
            additional_branches += len(coverage_info["branches"])
            additional_lines += len(coverage_info["lines"])

            if merge:
                original_dict[filename] = coverage_info
        else:
            new_branches = coverage_info["branches"] - original_dict[filename]["branches"]
            new_lines = coverage_info["lines"] - original_dict[filename]["lines"]

            additional_branches += len(new_branches)
            additional_lines += len(new_lines)

            if merge:
                original_dict[filename]["branches"] = original_dict[filename]["branches"].union(coverage_info["branches"])
                original_dict[filename]["lines"] = original_dict[filename]["lines"].union(coverage_info["lines"])
    
    return additional_branches, additional_lines, original_dict

def print_dict_info(coverage_dict):
    num_branches = 0
    num_lines = 0
    for filename, coverage_info in coverage_dict.items():
        num_branches += len(coverage_info["branches"])
        num_lines += len(coverage_info["lines"])

    print(f"Branches: {num_branches}")
    print(f"Lines: {num_lines}")

def check_coverage_addition(coverage_dict, driver_file, pkl_dir, filename_new_br, prefix):
    print("Original:")
    print_dict_info(coverage_dict)

    merge = True    # Merge the new dictionaries as you go so that we can ignore branches covered by previous inputs
    files_w_new_br = []

    pkl_files = []
    for file in os.listdir(pkl_dir):
        if not file.endswith(".pkl"):
            continue
        pkl_files.append(file)
    
    total = len(pkl_files)
    print(f"Working on {total} pkl files\n")

    for i, file in enumerate(pkl_files):
        pkl_file = os.path.join(pkl_dir, file)
        print(f"[{i+1}/{total}] Working with {file}")
        cmd_line = f"python {driver_file} {pkl_file}"
        return_code, lcov_data = gen_cov(cmd_line, prefix=prefix)
        new_dict = analyze_lcov(lcov_data)
        additional_branches, additional_lines, coverage_dict = compare_cov_dicts(coverage_dict, new_dict, merge=merge)
        if additional_branches > 0:
            print(f"{file} uncovered {additional_branches} new branches and {additional_lines} new lines")
            files_w_new_br.append(pkl_file)
            with open(filename_new_br, "a") as f:
                f.write(f"{pkl_file},{additional_branches}\n")
            print("New:")
            print_dict_info(coverage_dict)

    return files_w_new_br

def create_driver(api):
    code = f"""
import sys, os, pickle, torch

file = sys.argv[1]
with open(file, 'rb') as f:
    input_dict = pickle.load(f)
    try:
        output = {api}(*input_dict['{api}']['args'], **input_dict['{api}']['kwargs'])
    except Exception as e:
        exception_msg = e.__class__.__name__ + ": " + str(e)
    
"""
    custom_drivers_dir = create_subdir(get_tmp_dir(), "custom_drivers")
    driver_file = os.path.join(custom_drivers_dir, f"{api}_driver.py")
    with open(driver_file, "w") as f:
        f.write(code)
    
    return driver_file


def main():
    api = sys.argv[1]
    lib = sys.argv[2] if len(sys.argv) > 2 else "torch"
    
    api = get_lib_version(api, lib=lib)
    prefix = api.replace('.', '_')
    lcov_dir = os.path.join(get_tmp_dir(), "coverage_raw_files")
    lcov_file = os.path.join(lcov_dir, f"{prefix}.lcov")
    
    if not os.path.exists(lcov_file):
        print(f"{lcov_file} does not exist")
        return

    output_dir = create_subdir(get_tmp_dir(), "debug_coverage")
    filename_new_br = os.path.join(output_dir, f"{api}.csv")

    # Titanfuzz
    prefix = f"{api}_titan"
    pkl_dir = os.path.join(get_dir_in_root("eval"), f"titanfuzz/modified_inputs/{api}")
    if not os.path.exists(pkl_dir):
        print(f"No saved inputs found from titanfuzz execution of {api}")
        return

    coverage_dict = get_cov_dict_from_file(lcov_file)
    driver_file = create_driver(api)
    files_w_new_br = check_coverage_addition(coverage_dict, driver_file, pkl_dir, filename_new_br, prefix)
    print("Files that uncovered new branches:")
    print('\n'.join(files_w_new_br))

if __name__ == "__main__":
    main()