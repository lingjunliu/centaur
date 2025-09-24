
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def set_autocast_ipu_dtype_inputs():
    list_of_inputs = []

    input_dict = {
        "dtype": np.dtype('float32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "dtype": np.dtype('float16')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "dtype": np.dtype('int32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "dtype": np.dtype('int8')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "dtype": np.dtype('uint8')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "dtype": np.dtype('float64')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = set_autocast_ipu_dtype_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('set_autocast_ipu_dtype', generated_inputs)
