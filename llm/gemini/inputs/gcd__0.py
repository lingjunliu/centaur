
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def gcd__inputs():
    list_of_inputs = []

    input1 = np.array([2, 4, 6, 8], dtype=np.int64)
    other1 = np.array([1, 2, 3, 4], dtype=np.int64)
    input_dict1 = {"input": input1, "other": other1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[2, 4], [6, 8]], dtype=np.int32)
    other2 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    input_dict2 = {"input": input2, "other": other2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[-2, 4], [6, -8]], dtype=np.int16)
    other3 = np.array([[1, -2], [-3, 4]], dtype=np.int16)
    input_dict3 = {"input": input3, "other": other3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([2, 4, 6, 8], dtype=np.int8)
    other4 = np.array([1, 0, 3, 0], dtype=np.int8)
    input_dict4 = {"input": input4, "other": other4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([[[2, 4], [6, 8]], [[10, 12], [14, 16]]], dtype=np.int64)
    other5 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64)
    input_dict5 = {"input": input5, "other": other5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

generated_inputs["torch.gcd_"] = gcd__inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.gcd_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.gcd_'.")

check_valid('torch.gcd_', generated_inputs['torch.gcd_'], lib="torch")
