
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def ReplicationPad1d_inputs():
    list_of_inputs = []

    # Test case 1: int padding, 3D input
    input = torch.arange(8, dtype=torch.float).reshape(1, 2, 4).numpy()
    padding = 2
    input_dict = {"input": input, "padding": padding}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 2: tuple padding, 3D input
    input = torch.arange(8, dtype=torch.float).reshape(1, 2, 4).numpy()
    padding = (3, 1)
    input_dict = {"input": input, "padding": padding}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 3: int padding, 2D input
    input = torch.arange(6, dtype=torch.float).reshape(2, 3).numpy()
    padding = 1
    input_dict = {"input": input, "padding": padding}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 4: tuple padding, 2D input
    input = torch.arange(6, dtype=torch.float).reshape(2, 3).numpy()
    padding = (2, 1)
    input_dict = {"input": input, "padding": padding}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 5: int padding, 3D input with negative values
    input = torch.arange(-4, 4, dtype=torch.float).reshape(1, 2, 4).numpy()
    padding = 2
    input_dict = {"input": input, "padding": padding}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 6: tuple padding, 3D input with zeros
    input = torch.zeros(1, 2, 4).numpy()
    padding = (1, 2)
    input_dict = {"input": input, "padding": padding}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Test case 7: large padding
    input = torch.randn(1, 1, 5).numpy()
    padding = 10
    input_dict = {"input": input, "padding": padding}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.ReplicationPad1d_1"] = ReplicationPad1d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.ReplicationPad1d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ReplicationPad1d_1'.")

check_valid('torch.nn.ReplicationPad1d', generated_inputs['torch.nn.ReplicationPad1d_1'], lib="torch")
