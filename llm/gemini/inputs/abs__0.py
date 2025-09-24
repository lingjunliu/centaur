
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def abs_inputs():
    list_of_inputs = []

    # Test case 1: Basic float tensor with negative values
    input_tensor = np.array([-1.0, 2.0, -3.0, 4.0]).astype(np.float32)
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 2: Integer tensor with negative values
    input_tensor = np.array([-1, 2, -3, 4]).astype(np.int32)
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Test case 3: Multi-dimensional float tensor
    input_tensor = np.array([[-1.0, 2.0], [-3.0, 4.0]]).astype(np.float64)
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 4: Tensor with only positive values
    input_tensor = np.array([1, 2, 3, 4]).astype(np.int64)
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 5: Zero-dimensional tensor
    input_tensor = np.array(-5).astype(np.int8)
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Test case 6: 3D tensor with floats
    input_tensor = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.abs_"] = abs_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.abs_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.abs_'.")

check_valid('torch.abs_', generated_inputs['torch.abs_'], lib="torch")
