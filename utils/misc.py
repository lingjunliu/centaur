import numpy as np
import os
import sys
import pickle
import time
from utils.defaults import list_of_available_dtypes

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def generate_executible_snippet_from_str(code, library="", print_exception=False):
    to_ret = ""
    if library > "":
        to_ret = f"import {library}\n"
    
    to_ret += "try:\n"
    if code > "":
        for line in code.splitlines():
            to_ret += f"\t{line}\n"
    else:
        to_ret += "\tpass\n"
    
    if print_exception:
        to_ret += f"except Exception as e:\n\tprint(str(e.__class__.__name__), str(e))\n"
    else:
        to_ret += f"except:\n\tpass\n"
    
    return to_ret

def get_tensor_size(ll):
    sz = np.dtype(list_of_available_dtypes[ll[1][0]]).itemsize
    for dim in ll[0]:
        sz = sz * dim
    
    sz = sz * .001 * .001 # MB
    return sz

def get_input_size(input, signature):
    total_size = 0
    for arg, domain in signature.items():
        if domain == "tensor":
            total_size += get_tensor_size(input[arg])
        else:
            try:
                total_size += (np.dtype(list_of_available_dtypes[input[arg][0]]).itemsize * .001 * .001) # MB
            except:
                continue
    return total_size

def get_dir_in_root(subdir):
    dir = os.path.join(CUR_DIR, f"../{subdir}")
    os.makedirs(dir, exist_ok=True)
    
    return dir

def save_file_in_root(filename, content):
    filepath = os.path.join(CUR_DIR, f"../{filename}")
    with open(filepath, "w") as f:
        f.write(content)

def append_file_in_root(filename, content):
    filepath = os.path.join(CUR_DIR, f"../{filename}")
    with open(filepath, "a") as f:
        f.write(content)

def read_file_in_root(filename):
    filepath = os.path.join(CUR_DIR, f"../{filename}")
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return [line.strip() for line in f.readlines()]
    
    return []

def get_tmp_dir():
    return get_dir_in_root(".tmp")

def create_subdir(dir, subfolder):
    subdir = os.path.join(dir, subfolder)
    os.makedirs(subdir, exist_ok=True)
    
    return subdir

def save_to_new_pkl(filepath, obj):
    '''
    Save the object to a pickle file. If the file already exists,
    it will not be overwritten.
    '''
    if not os.path.isfile(filepath):
        with open(filepath, "wb") as f:
            pickle.dump(obj, f)

def save_to_pkl(filepath, obj):
    '''
    Save the object to a pickle file. If the file already exists,
    it will be overwritten.
    '''
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)

def read_pkl(filepath):
    with open(filepath, "rb") as f:
        obj = pickle.load(f)
    return obj

def has_time(start, duration):
    if duration > 0:
        elapsed = time.time() - start
        if elapsed >= duration:
            return False
    return True

def is_inhomogeneous(lst):
    if not isinstance(lst, list):
        return False  # Base case: non-list elements are homogeneous
    
    lengths = [len(sublist) if isinstance(sublist, list) else -1 for sublist in lst]
    
    # Check if all sublists have the same length
    if len(set(lengths)) > 1:
        return True
    
    # Recursively check sublists
    return any(is_inhomogeneous(sublist) for sublist in lst if isinstance(sublist, list))

def flatten(lst):
    """Recursively flatten a nested list."""
    if isinstance(lst, list):
        return [item for sublist in lst for item in flatten(sublist)]
    return [lst]

def api_in_file(api, filename):
    """
    Check if the given API is present in the specified file.
    """
    if not os.path.exists(filename):
        print(f"File {filename} does not exist.")
        return False
    with open(filename, "r") as f:
        for line in f.readlines():
            if f"{api}(" in line:
                return True
    return False

def generate_driver_map():
    driver_dir = get_dir_in_root("drivers")
    drivers = read_file_in_root("apis.txt")
    llm_dir = get_dir_in_root("llm")
    # TODO: Bring this list to a common dir
    all_apis_file = f"{llm_dir}/api_full.txt"
    with open(all_apis_file, "r") as f:
        torch_apis = [line.strip() for line in f.readlines()]
    
    drivers_to_api_str = "Driver,API\n"
    apis_to_exclude = ["torch.use_deterministic_algorithms", "torch.utils.deterministic.fill_uninitialized_memory", "torch.tensor"]
    for driver in drivers:
        filepath = f"{driver_dir}/{driver}.py"
        candidates = set()
        for torch_api in torch_apis:
            if torch_api in apis_to_exclude:
                continue
            if api_in_file(torch_api, filepath):
                candidates.add(torch_api)
        
        # prune candidates
        if len(candidates) > 1:
            to_remove = set()
            for candidate in candidates:
                if not candidate.split('.')[-1] in driver:
                    to_remove.add(candidate)
            
            candidates = candidates - to_remove
        
        if len(candidates) == 1:
            drivers_to_api_str += f"{driver},{candidates.pop()}\n"
        elif len(candidates) == 0:
            print(f"No candidate found for drivers/{driver}.py")
            print("-" * 10)
        else:
            print(f"Driver drivers/{driver}.py has {len(candidates)} matching apis:")
            print("\n".join(candidates))
            for candidate in candidates:
                if candidate.split('.')[-1] == driver:
                    print(f"Selected {candidate}")
                    drivers_to_api_str += f"{driver},{candidate}\n"
                    break
            print("-" * 10)
    
    csv_file = os.path.join(CUR_DIR, "../drivers_to_api.csv")
    with open(csv_file, "w") as f:
        f.write(drivers_to_api_str)
        print(f"Saved the mapping to {csv_file}")
    
    with open(f"{CUR_DIR}/../torch_apis.txt", "w") as f:
        for line in drivers_to_api_str.splitlines():
            if line.startswith("Driver"):
                continue    # header
            f.write(f"{line.split(',')[1]}\n")
            

def map_torch_to_driver():
    """
    return torch_to_driver, driver_to_torch maps
    """
    torch_to_driver = {}
    driver_to_torch = {}
    lines = read_file_in_root("drivers_to_api.csv")
    for line in lines:
        driver, torch_api = line.strip().split(",")
        if driver == "Driver":  # Skip the header
            continue
        driver_to_torch[driver] = torch_api
        torch_to_driver[torch_api] = driver

    return torch_to_driver, driver_to_torch

def parse_cancelled_jobs(lib="torch"):
    cancelled_jobs_file = os.path.join(get_tmp_dir(), "cancelled_jobs.log")
    inference_jobs = []
    model_jobs = []
    with open(cancelled_jobs_file, 'r') as file:
        for line in file.readlines():
            tokens = line.strip().split()
            for token in tokens:
                if token.startswith("inf-"):
                    job_id = token.split('-')[1]
                    inference_jobs.append(int(job_id))
                elif token.startswith("modl-"):
                    job_id = token.split('-')[1]
                    model_jobs.append(int(job_id))

    variations = read_file_in_root(f"{lib}_variations.txt")

    cancelled_inference_variations = []
    cancelled_model_variations = []
    for i, variation in enumerate(variations):
        if i+1 in inference_jobs:
            cancelled_inference_variations.append(variation)
        if i+1 in model_jobs:
            cancelled_model_variations.append(variation)

    inf_file = os.path.join(get_tmp_dir(), f"cancelled_infs_{lib}.txt")
    modl_file = os.path.join(get_tmp_dir(), f"cancelled_modls_{lib}.txt")

    with open(inf_file, 'w') as file:
        file.write("\n".join(cancelled_inference_variations))
    
    with open(modl_file, 'w') as file:
        file.write("\n".join(cancelled_model_variations))

    return cancelled_inference_variations, cancelled_model_variations

def merge_csvs(csv_1, csv_2, csv_3):
    with open(csv_1, "r") as f_1:
        lines_1 = f_1.readlines()
    
    with open(csv_2, "r") as f_2:
        lines_2 = f_2.readlines()
    
    lines = ""
    for l_1 in lines_1:
        tokens_1 = l_1.strip().split(",")
        for l_2 in lines_2:
            tokens_2 = l_2.strip().split(",")
            if tokens_1[0] == tokens_2[0]:
                lines += l_1.strip() + "," + ",".join(tokens_2[1:]) + "\n"
                
    with open(csv_3, "w") as f:
        f.write(lines)
        
if __name__ == "__main__":
    if len(sys.argv) > 3:
        merge_csvs(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        generate_driver_map()