
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def bitwise_or_inputs():
    list_of_inputs = []

    # Test case 1: Basic test with positive integers
    input1 = np.array([1, 2, 3, 4], dtype=np.int32)
    other1 = np.array([4, 3, 2, 1], dtype=np.int32)
    input_dict1 = {"input": input1, "other": other1, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Test case 2: Test with negative integers
    input2 = np.array([-1, -2, -3, -4], dtype=np.int32)
    other2 = np.array([-4, -3, -2, -1], dtype=np.int32)
    input_dict2 = {"input": input2, "other": other2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Test case 3: Test with different shapes
    input3 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    other3 = np.array([[4, 3], [2, 1]], dtype=np.int32)
    input_dict3 = {"input": input3, "other": other3, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Test case 4: Test with unsigned integers
    input4 = np.array([1, 2, 3, 4], dtype=np.uint8)
    other4 = np.array([4, 3, 2, 1], dtype=np.uint8)
    input_dict4 = {"input": input4, "other": other4, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Test case 5: Test with broadcasting
    input5 = np.array([1, 2, 3], dtype=np.int32)
    other5 = np.array(1, dtype=np.int32)
    input_dict5 = {"input": input5, "other": other5, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

generated_inputs["torch.bitwise_or"] = bitwise_or_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.bitwise_or' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.bitwise_or'.")

check_valid('torch.bitwise_or', generated_inputs['torch.bitwise_or'], lib="torch")
