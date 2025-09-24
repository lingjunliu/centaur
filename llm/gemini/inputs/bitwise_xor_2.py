
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def bitwise_xor_inputs():
    list_of_inputs = []

    # Input 1: Basic case with positive integers
    input1 = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    other1 = 7
    input_dict1 = {"input": input1, "other": other1, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Input with negative integers
    input2 = np.array([-1, -2, -3, -4, -5], dtype=np.int32)
    other2 = 3
    input_dict2 = {"input": input2, "other": other2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Input with a different dtype (int64)
    input3 = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    other3 = 15
    input_dict3 = {"input": input3, "other": other3, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Input as a multi-dimensional array
    input4 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    other4 = 5
    input_dict4 = {"input": input4, "other": other4, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Larger numbers
    input5 = np.array([255, 65535, 2147483647], dtype=np.int64)
    other5 = 16777215
    input_dict5 = {"input": input5, "other": other5, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Input 6: Unsigned integer
    input6 = np.array([1, 2, 3, 4, 5], dtype=np.uint32)
    other6 = 7
    input_dict6 = {"input": input6, "other": other6, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.bitwise_xor_2"] = bitwise_xor_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.bitwise_xor_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.bitwise_xor_2'.")

check_valid('torch.bitwise_xor', generated_inputs['torch.bitwise_xor_2'], lib="torch")
