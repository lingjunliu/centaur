
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def CircularPad1d_inputs():
    list_of_inputs = []

    # Input 1: Integer padding, 2D input
    input = np.arange(8, dtype=np.float32).reshape(1, 8)
    padding = 2
    input_dict = {"input": input, "padding": padding}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Tuple padding, 3D input
    input = np.arange(12, dtype=np.float32).reshape(1, 2, 6)
    padding = (3, 1)
    input_dict = {"input": input, "padding": padding}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative padding, 2D input
    input = np.arange(10, dtype=np.float32).reshape(1, 10)
    padding = -1
    input_dict = {"input": input, "padding": padding}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Tuple with negative padding, 3D input
    input = np.arange(15, dtype=np.float32).reshape(1, 3, 5)
    padding = (-1, -1)
    input_dict = {"input": input, "padding": padding}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Integer padding, 3D input with different dimensions
    input = np.arange(20, dtype=np.float32).reshape(2, 2, 5)
    padding = 3
    input_dict = {"input": input, "padding": padding}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.CircularPad1d_1"] = CircularPad1d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.CircularPad1d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.CircularPad1d_1'.")

check_valid('torch.nn.CircularPad1d', generated_inputs['torch.nn.CircularPad1d_1'], lib="torch")
