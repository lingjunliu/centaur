
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def ReplicationPad3d_inputs():
    list_of_inputs = []

    # Test case 1: int padding
    input = torch.randn(2, 3, 4, 5, 6).numpy()
    padding = 2
    input_dict = {"padding": padding, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 2: tuple padding
    input = torch.randn(1, 1, 3, 3, 3).numpy()
    padding = (1, 2, 3, 4, 5, 6)
    input_dict = {"padding": padding, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 3: different input size
    input = torch.randn(4, 2, 7, 8, 9).numpy()
    padding = (0, 1, 2, 0, 1, 2)
    input_dict = {"padding": padding, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 4: single channel input
    input = torch.randn(1, 1, 5, 5, 5).numpy()
    padding = 1
    input_dict = {"padding": padding, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 5: small input size
    input = torch.randn(1, 3, 1, 1, 1).numpy()
    padding = (1, 1, 1, 1, 1, 1)
    input_dict = {"padding": padding, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Test case 6: another input size
    input = torch.randn(8, 5, 16, 32, 64).numpy()
    padding = (2, 2, 4, 4, 1, 1)
    input_dict = {"padding": padding, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.ReplicationPad3d_2"] = ReplicationPad3d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.ReplicationPad3d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ReplicationPad3d_2'.")

check_valid('torch.nn.ReplicationPad3d', generated_inputs['torch.nn.ReplicationPad3d_2'], lib="torch")
