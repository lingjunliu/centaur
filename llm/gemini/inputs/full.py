
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy, numpy as np

def full_inputs():
    list_of_inputs = []

    # Input 1: Basic example with integer size and float fill value
    input_dict = {
        "size": (2, 3),
        "fill_value": 3.14,
        "dtype": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Larger size, integer fill value, and int dtype
    input_dict = {
        "size": (5, 5, 5),
        "fill_value": 7,
        "dtype": np.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative fill value, size as a single integer, and float64 dtype
    input_dict = {
        "size": (4,),
        "fill_value": -2.5,
        "dtype": np.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Empty tensor size, should produce an empty tensor
    input_dict = {
        "size": (0,),
        "fill_value": 10,
        "dtype": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Size with different dimensions, complex fill value and dtype
    input_dict = {
        "size": (2, 1, 4),
        "fill_value": complex(1.0, -1.0),
        "dtype": np.complex64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Using a boolean dtype
    input_dict = {
        "size": (3, 2),
        "fill_value": 1,
        "dtype": np.bool_
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: A large size
    input_dict = {
        "size": (100, 100),
        "fill_value": 0.0,
        "dtype": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: unsigned int
    input_dict = {
        "size": (2, 2),
        "fill_value": 255,
        "dtype": np.uint8
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = full_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('full', generated_inputs)
