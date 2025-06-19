
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def ZeroPad1d_inputs():
    list_of_inputs = []

    # Test case 1: Integer padding, 2D input
    input = torch.randn(2, 3).numpy()
    padding = 1
    input_dict = {"padding": padding, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 2: Tuple padding, 2D input
    input = torch.randn(2, 3).numpy()
    padding = (2, 1)
    input_dict = {"padding": padding, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 3: Integer padding, 3D input
    input = torch.randn(1, 2, 4).numpy()
    padding = 2
    input_dict = {"padding": padding, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 4: Tuple padding, 3D input
    input = torch.randn(1, 2, 3).numpy()
    padding = (3, 1)
    input_dict = {"padding": padding, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 5: Larger padding values, 3D input
    input = torch.randn(1, 2, 5).numpy()
    padding = (5, 2)
    input_dict = {"padding": padding, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.ZeroPad1d_1"] = ZeroPad1d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.ZeroPad1d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ZeroPad1d_1'.")

check_valid('torch.nn.ZeroPad1d', generated_inputs['torch.nn.ZeroPad1d_1'], lib="torch")
