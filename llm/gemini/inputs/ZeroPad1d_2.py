
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def ZeroPad1d_inputs():
    list_of_inputs = []

    # Input 1: int padding, 3D tensor
    input = torch.randn(1, 2, 4).numpy()
    padding = 2
    input_dict = {"padding": padding, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: tuple padding, 3D tensor
    input = torch.randn(1, 2, 3).numpy()
    padding = (3, 1)
    input_dict = {"padding": padding, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: int padding, 2D tensor
    input = torch.randn(2, 4).numpy()
    padding = 1
    input_dict = {"padding": padding, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: tuple padding, 2D tensor
    input = torch.randn(2, 4).numpy()
    padding = (1, 2)
    input_dict = {"padding": padding, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: int padding, 3D tensor, different values
    input = torch.randn(2, 3, 5).numpy()
    padding = 3
    input_dict = {"padding": padding, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: tuple padding, 3D tensor, different values
    input = torch.randn(2, 3, 5).numpy()
    padding = (2, 4)
    input_dict = {"padding": padding, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.ZeroPad1d_2"] = ZeroPad1d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.ZeroPad1d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ZeroPad1d_2'.")

check_valid('torch.nn.ZeroPad1d', generated_inputs['torch.nn.ZeroPad1d_2'], lib="torch")
