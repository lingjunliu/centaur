
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def bitwise_left_shift_inputs():
    list_of_inputs = []

    # Case 1: Basic case with positive integers
    input1 = np.array([1, 2, 3, 4], dtype=np.int32)
    other1 = np.array([1, 2, 0, 3], dtype=np.int32)
    input_dict1 = {"input": input1, "other": other1, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Case 2: Different data types (int64)
    input2 = np.array([5, 6, 7, 8], dtype=np.int64)
    other2 = np.array([2, 1, 3, 0], dtype=np.int64)
    input_dict2 = {"input": input2, "other": other2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Case 3: Multi-dimensional array
    input3 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    other3 = np.array([[2, 1], [0, 3]], dtype=np.int32)
    input_dict3 = {"input": input3, "other": other3, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    # Case 4: Scalar other
    input4 = np.array([1, 2, 3, 4], dtype=np.int32)
    other4 = np.array(2, dtype=np.int32)
    input_dict4 = {"input": input4, "other": other4, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Case 5: Different shapes that are broadcastable
    input5 = np.array([[1, 2, 3]], dtype=np.int32)
    other5 = np.array([1, 2, 0], dtype=np.int32)
    input_dict5 = {"input": input5, "other": other5, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Case 6: Larger values that will cause overflow (still valid for bitwise operations)
    input6 = np.array([2**30, 2**31 -1, 10], dtype=np.int32)
    other6 = np.array([1, 2, 3], dtype=np.int32)
    input_dict6 = {"input": input6, "other": other6, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.bitwise_left_shift"] = bitwise_left_shift_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.bitwise_left_shift' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.bitwise_left_shift'.")

check_valid('torch.bitwise_left_shift', generated_inputs['torch.bitwise_left_shift'], lib="torch")
