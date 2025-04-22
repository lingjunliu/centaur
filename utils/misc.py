import numpy as np
import os
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