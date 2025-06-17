
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def log1p_inputs():
    list_of_inputs = []

    # Input 1: Scalar float
    input1 = np.array(0.5, dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 1D tensor with positive values
    input2 = np.array([0, 0.5, 1.0], dtype=np.float32)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Int Tensor
    input3 = np.array([1, 2, 3], dtype=np.int32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    return list_of_inputs

generated_inputs["torch.log1p"] = log1p_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.log1p' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.log1p'.")

check_valid('torch.log1p', generated_inputs['torch.log1p'], lib="torch")
