
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def broadcast_to_inputs():
    list_of_inputs = []

    # Case 1: Simple 1D to 2D broadcast
    input1 = np.array([1, 2, 3])
    size1 = (2, 3)
    input_dict1 = {"input": input1, "size": size1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Case 2: Broadcast a scalar to a 3D tensor
    input2 = np.array(5)
    size2 = (2, 3, 4)
    input_dict2 = {"input": input2, "size": size2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Case 3: Broadcast with higher dimensions, int type - Modified for compatibility
    input3 = np.array([[1], [2]])
    size3 = (2, 2)  # Changed size to be compatible with input3
    input_dict3 = {"input": input3, "size": size3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Case 4: Broadcast with float type
    input4 = np.array([1.0, 2.0])
    size4 = (3, 2)
    input_dict4 = {"input": input4, "size": size4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Case 5: Broadcast with bool type
    input5 = np.array([True, False])
    size5 = (2, 2)
    input_dict5 = {"input": input5, "size": size5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.broadcast_to"] = broadcast_to_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.broadcast_to' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.broadcast_to'.")

check_valid('torch.broadcast_to', generated_inputs['torch.broadcast_to'], lib="torch")
