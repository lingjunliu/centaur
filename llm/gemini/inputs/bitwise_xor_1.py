
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def bitwise_xor_inputs():
    list_of_inputs = []

    # Input 1: Basic integer tensors
    input1 = np.array([1, 2, 3, 4], dtype=np.int32)
    other1 = np.array([5, 6, 7, 8], dtype=np.int32)
    out1 = np.empty_like(input1)
    input_dict1 = {"input": input1, "other": other1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Different shapes, int64
    input2 = np.array([[1, 0], [0, 1]], dtype=np.int64)
    other2 = np.array([[0, 1], [1, 0]], dtype=np.int64)
    out2 = np.empty_like(input2)
    input_dict2 = {"input": input2, "other": other2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Broadcasting, unsigned int
    input3 = np.array([1, 2, 3], dtype=np.uint8)
    other3 = np.array(2, dtype=np.uint8)
    out3 = np.empty_like(input3)
    input_dict3 = {"input": input3, "other": other3, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Multi-dimensional, signed integers, different datatypes but compatible
    input4 = np.array([[1, 2], [3, 4]], dtype=np.int16)
    other4 = np.array([[5, 6], [7, 8]], dtype=np.int32)
    out4 = np.empty_like(input4, dtype=np.int32)
    input_dict4 = {"input": input4, "other": other4, "out": out4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Boolean Tensors (treated as 0 and 1)
    input5 = np.array([True, False, True], dtype=bool)
    other5 = np.array([False, True, False], dtype=bool)
    out5 = np.empty_like(input5, dtype=bool)
    input_dict5 = {"input": input5, "other": other5, "out": out5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Input 6: Using negative values
    input6 = np.array([-1, -2, 3], dtype=np.int32)
    other6 = np.array([5, -6, -7], dtype=np.int32)
    out6 = np.empty_like(input6)
    input_dict6 = {"input": input6, "other": other6, "out": out6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    return list_of_inputs

generated_inputs["torch.bitwise_xor_1"] = bitwise_xor_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.bitwise_xor_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.bitwise_xor_1'.")

check_valid('torch.bitwise_xor', generated_inputs['torch.bitwise_xor_1'], lib="torch")
