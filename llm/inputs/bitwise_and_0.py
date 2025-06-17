
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def bitwise_and_inputs():
    list_of_inputs = []

    # Input 1: Basic case with two integer tensors
    input1 = np.array([1, 2, 3, 4], dtype=np.int32)
    input2 = np.array([5, 6, 7, 8], dtype=np.int32)
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different shapes, but broadcastable
    input1 = np.array([[1, 2], [3, 4]], dtype=np.int64)
    input2 = np.array([1, 0], dtype=np.int64)
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Input with negative values
    input1 = np.array([-1, -2, 3, -4], dtype=np.int8)
    input2 = np.array([5, -6, -7, 8], dtype=np.int8)
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Multi-dimensional input
    input1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.uint8)
    input2 = np.array([[[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.uint8)
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Boolean input
    input1 = np.array([True, False, True, False], dtype=bool)
    input2 = np.array([False, True, False, True], dtype=bool)
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different data types (both integers)
    input1 = np.array([1, 2, 3, 4], dtype=np.int16)
    input2 = np.array([5, 6, 7, 8], dtype=np.int32)
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.bitwise_and"] = bitwise_and_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.bitwise_and' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.bitwise_and'.")

check_valid('torch.bitwise_and', generated_inputs['torch.bitwise_and'], lib="torch")
