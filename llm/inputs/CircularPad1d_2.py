
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def CircularPad1d_inputs():
    list_of_inputs = []

    # Input 1: int padding, 2D input
    input = torch.arange(8, dtype=torch.float).reshape(2, 4).numpy()
    padding = 2
    input_dict = {"padding": padding, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: tuple padding, 3D input
    input = torch.arange(12, dtype=torch.float).reshape(1, 3, 4).numpy()
    padding = (3, 1)
    input_dict = {"padding": padding, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: int padding, 3D input, different shape
    input = torch.randn(2, 5, 7).numpy()
    padding = 1
    input_dict = {"padding": padding, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: tuple padding, 2D input, zero padding
    input = torch.randn(4, 5).numpy()
    padding = (0, 0)
    input_dict = {"padding": padding, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: tuple padding, 2D input, different padding values
    input = torch.randn(3, 6).numpy()
    padding = (1, 2)
    input_dict = {"padding": padding, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.CircularPad1d_2"] = CircularPad1d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.CircularPad1d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.CircularPad1d_2'.")

check_valid('torch.nn.CircularPad1d', generated_inputs['torch.nn.CircularPad1d_2'], lib="torch")
