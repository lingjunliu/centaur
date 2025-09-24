
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def nansum_inputs():
    list_of_inputs = []

    input1 = np.array([[1.0, 2.0, np.nan], [3.0, np.nan, 5.0]])
    dim1 = (0,)
    keepdim1 = False
    dtype1 = np.float32
    input_dict1 = {
        "input": input1,
        "dim": dim1,
        "keepdim": keepdim1,
        "dtype": dtype1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[[1.0, 2.0, np.nan], [3.0, np.nan, 5.0]], [[6.0, 7.0, 8.0], [9.0, 10.0, np.nan]]])
    dim2 = (0, 2)
    keepdim2 = True
    dtype2 = np.float64
    input_dict2 = {
        "input": input2,
        "dim": dim2,
        "keepdim": keepdim2,
        "dtype": dtype2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([np.nan, np.nan, np.nan])
    dim3 = (0,)
    keepdim3 = False
    dtype3 = np.float64
    input_dict3 = {
        "input": input3,
        "dim": dim3,
        "keepdim": keepdim3,
        "dtype": dtype3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    dim4 = (1,)
    keepdim4 = True
    dtype4 = np.float32
    input_dict4 = {
        "input": input4,
        "dim": dim4,
        "keepdim": keepdim4,
        "dtype": dtype4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([[[1.0, 2.0], [np.nan, 4.0]], [[5.0, np.nan], [7.0, 8.0]]])
    dim5 = (0, 1, 2)
    keepdim5 = False
    dtype5 = np.float16
    input_dict5 = {
        "input": input5,
        "dim": dim5,
        "keepdim": keepdim5,
        "dtype": dtype5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = nansum_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('nansum', generated_inputs)
