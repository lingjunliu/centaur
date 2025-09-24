
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def ReflectionPad1d_inputs():
    list_of_inputs = []

    # Input 1: Integer padding, 2D input
    input_1 = torch.arange(6, dtype=torch.float).reshape(1, 6).numpy()
    padding_1 = 2
    input_dict_1 = {"input": input_1, "padding": padding_1}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Tuple padding, 2D input
    input_2 = torch.arange(10, dtype=torch.float).reshape(1, 10).numpy()
    padding_2 = (3, 1)
    input_dict_2 = {"input": input_2, "padding": padding_2}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Integer padding, 3D input
    input_3 = torch.randn(2, 3, 5).numpy()
    padding_3 = 1
    input_dict_3 = {"input": input_3, "padding": padding_3}
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Tuple padding, 3D input
    input_4 = torch.randn(1, 4, 7).numpy()
    padding_4 = (2, 3)
    input_dict_4 = {"input": input_4, "padding": padding_4}
    list_of_inputs.append(copy.deepcopy(input_dict_4))
    
    # Input 5: Integer padding, 2D input, negative values
    input_5 = torch.arange(-3, 3, dtype=torch.float).reshape(1, 6).numpy()
    padding_5 = 2
    input_dict_5 = {"input": input_5, "padding": padding_5}
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    return list_of_inputs

generated_inputs["torch.nn.ReflectionPad1d_1"] = ReflectionPad1d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.ReflectionPad1d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ReflectionPad1d_1'.")

check_valid('torch.nn.ReflectionPad1d', generated_inputs['torch.nn.ReflectionPad1d_1'], lib="torch")
