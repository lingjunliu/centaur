
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def lcm_inputs():
    list_of_inputs = []

    # Input 1: Basic integer tensors
    input1 = np.array([2, 4, 6], dtype=np.int32)
    other1 = np.array([3, 5, 7], dtype=np.int32)
    input_dict1 = {"input": input1, "other": other1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Different shapes, but compatible
    input2 = np.array([[2, 4], [6, 8]], dtype=np.int64)
    other2 = np.array([[3, 5], [7, 9]], dtype=np.int64)
    input_dict2 = {"input": input2, "other": other2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Scalars
    input3 = np.array(5, dtype=np.int32)
    other3 = np.array(7, dtype=np.int32)
    input_dict3 = {"input": input3, "other": other3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Broadcasting
    input4 = np.array([2, 4, 6], dtype=np.int64)
    other4 = np.array(3, dtype=np.int64)
    input_dict4 = {"input": input4, "other": other4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Multi-dimensional arrays
    input5 = np.array([[[2, 4], [6, 8]], [[10, 12], [14, 16]]], dtype=np.int32)
    other5 = np.array([[[3, 5], [7, 9]], [[11, 13], [15, 17]]], dtype=np.int32)
    input_dict5 = {"input": input5, "other": other5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Input 6: Mixed dtypes (int32 and int64)
    input6 = np.array([2, 4, 6], dtype=np.int32)
    other6 = np.array([3, 5, 7], dtype=np.int64)
    input_dict6 = {"input": input6, "other": other6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Zero value
    input7 = np.array([0, 4, 6], dtype=np.int32)
    other7 = np.array([3, 0, 7], dtype=np.int32)
    input_dict7 = {"input": input7, "other": other7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    # Input 8: Large Numbers
    input8 = np.array([2**30, 4], dtype=np.int64)
    other8 = np.array([3, 2**31], dtype=np.int64)
    input_dict8 = {"input": input8, "other": other8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    

    return list_of_inputs

generated_inputs = lcm_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('lcm', generated_inputs)
