import numpy as np
import os
import sys
from utils.defaults import list_of_available_dtypes

def get_tensor_size(ll):
    sz = np.dtype(list_of_available_dtypes[ll[1][0]]).itemsize
    for dim in ll[0]:
        sz = sz * dim
    
    sz = sz * .001 * .001 # MB
    return sz

def get_tmp_dir():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    tmp_dir = os.path.join(cur_dir, "../.tmp")
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)
    
    return tmp_dir

def create_subdir(dir, subfolder):
    subdir = os.path.join(dir, subfolder)
    if not os.path.isdir(subdir):
        os.mkdir(subdir)
    
    return subdir

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