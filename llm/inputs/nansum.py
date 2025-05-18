
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
import numpy as np

def nansum_inputs():
    list_of_inputs = []

    # Input 1
    input = np.array([[1.0, 2.0, np.nan], [3.0, np.nan, 4.0]])
    dim = (0,)
    keepdim = False
    dtype = np.float32
    input_dict = {"input": input, "dim": dim, "keepdim": keepdim, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input = np.array([[1.0, 2.0, np.nan], [3.0, np.nan, 4.0]])
    dim = (1,)
    keepdim = True
    dtype = np.float64
    input_dict = {"input": input, "dim": dim, "keepdim": keepdim, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input = np.array([[[1.0, np.nan], [2.0, 3.0]], [[np.nan, 4.0], [5.0, 6.0]]])
    dim = (0, 1)
    keepdim = False
    dtype = np.float32
    input_dict = {"input": input, "dim": dim, "keepdim": keepdim, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input = np.array([[[1.0, np.nan], [2.0, 3.0]], [[np.nan, 4.0], [5.0, 6.0]]])
    dim = (2,)
    keepdim = True
    dtype = np.float64
    input_dict = {"input": input, "dim": dim, "keepdim": keepdim, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input = np.array([np.nan, np.nan, np.nan])
    dim = (0,)
    keepdim = False
    dtype = np.float32
    input_dict = {"input": input, "dim": dim, "keepdim": keepdim, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

list_of_inputs = nansum_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('nansum', list_of_inputs)
