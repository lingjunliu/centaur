
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def ReplicationPad1d_inputs():
    list_of_inputs = []

    # Test case 1: int padding, 3D input
    input = np.arange(8, dtype=np.float32).reshape(1, 2, 4)
    padding = 2
    input_dict = {"padding": padding, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 2: tuple padding, 3D input
    input = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
    padding = (3, 1)
    input_dict = {"padding": padding, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 3: int padding, 2D input
    input = np.arange(6, dtype=np.float32).reshape(2, 3)
    padding = 1
    input_dict = {"padding": padding, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 4: tuple padding, 2D input
    input = np.arange(10, dtype=np.float32).reshape(2, 5)
    padding = (2, 1)
    input_dict = {"padding": padding, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Test case 5: Different data type (int), int padding, 3D input
    input = np.arange(8, dtype=np.int32).reshape(1, 2, 4)
    padding = 2
    input_dict = {"padding": padding, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.ReplicationPad1d_2"] = ReplicationPad1d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.ReplicationPad1d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ReplicationPad1d_2'.")

check_valid('torch.nn.ReplicationPad1d', generated_inputs['torch.nn.ReplicationPad1d_2'], lib="torch")
