
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def abs_inputs():
    list_of_inputs = []

    input1 = np.array([-1, -2, 3])
    input_dict1 = {"input": torch.from_numpy(input1)}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[-1.5, -2.5], [3.5, -4.5]])
    input_dict2 = {"input": torch.from_numpy(input2).float()}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([1, 2, 3], dtype=np.int64)
    input_dict3 = {"input": torch.from_numpy(input3).long()}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([[-1, 2], [-3, 4]], dtype=np.int32)
    input_dict4 = {"input": torch.from_numpy(input4).int()}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.array([1.0, -2.0, 3.0, -4.0, 5.0], dtype=np.float32)
    input_dict5 = {"input": torch.from_numpy(input5).float()}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

generated_inputs["torch.abs"] = abs_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.abs' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.abs'.")

check_valid('torch.abs', generated_inputs['torch.abs'], lib="torch")
