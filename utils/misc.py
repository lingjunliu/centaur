import numpy as np
import os
import sys
import pickle
import time
from utils.defaults import list_of_available_dtypes

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

def map_torch_to_driver():
    torch_to_driver = {}
    driver_to_torch = {}
    csv_file = os.path.join(CUR_DIR, "../llm/drivers_to_api.csv")
    with open(csv_file, "r") as f:
        for line in f.readlines():
            driver, torch_api = line.strip().split(",")
            if driver == "Driver":  # Skip the header
                continue
            driver_to_torch[driver] = torch_api
            torch_to_driver[torch_api] = driver
    
    supported = os.path.join(CUR_DIR, "../llm/supported.csv")
    with open(supported, "r") as f:
        for line in f.readlines():
            driver, torch_api, _, _, _ = line.strip().split(",")
            if driver == "API":  # Skip the header
                continue
            if driver in driver_to_torch:
                continue
            if driver == "":
                continue
            driver_to_torch[driver] = torch_api
            torch_to_driver[torch_api] = driver

    return torch_to_driver, driver_to_torch

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
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    dir = os.path.join(cur_dir, f"../{subdir}")
    if not os.path.isdir(dir):
        os.mkdir(dir)
    
    return dir

def read_file_in_root(filename):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(cur_dir, f"../{filename}")
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return f.readlines()
    
    return []

def get_tmp_dir():
    return get_dir_in_root(".tmp")

def create_subdir(dir, subfolder):
    subdir = os.path.join(dir, subfolder)
    if not os.path.isdir(subdir):
        os.mkdir(subdir)
    
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
                lines += l_1.strip() + "," + l_2.strip() + "\n"
                
    with open(csv_3, "w") as f:
        f.write(lines)
        
if __name__ == "__main__":
    if len(sys.argv) > 3:
        merge_csvs(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        torch_to_driver, driver_to_torch = map_torch_to_driver()
        with open(os.path.join(CUR_DIR, "../apis.txt"), "r") as f:
            apis = [line.strip() for line in f.readlines()]
        for driver in apis:
            print(driver_to_torch[driver])