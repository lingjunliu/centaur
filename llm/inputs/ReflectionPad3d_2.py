
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def ReflectionPad3d_inputs():
    list_of_inputs = []

    # Test case 1: int padding, 5D input
    input1 = torch.randn(1, 1, 2, 2, 2).numpy()
    padding1 = 1
    input_dict1 = {"input": input1, "padding": padding1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Test case 2: tuple padding, 5D input, asymmetric padding
    input2 = torch.randn(1, 3, 3, 4, 5).numpy()
    padding2 = (1, 2, 0, 1, 2, 0)
    input_dict2 = {"input": input2, "padding": padding2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Test case 3: int padding, 4D input
    input3 = torch.randn(3, 3, 3, 3).numpy()
    padding3 = 1
    input_dict3 = {"input": input3, "padding": padding3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Test case 4: tuple padding, 4D input, asymmetric padding
    input4 = torch.randn(3, 4, 5, 6).numpy()
    padding4 = (0, 1, 2, 0, 1, 0)
    input_dict4 = {"input": input4, "padding": padding4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Test case 5: int padding, 5D input, small input size
    input5 = torch.randn(2, 2, 3, 3, 3).numpy()
    padding5 = 1
    input_dict5 = {"input": input5, "padding": padding5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.nn.ReflectionPad3d_2"] = ReflectionPad3d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.ReflectionPad3d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ReflectionPad3d_2'.")

check_valid('torch.nn.ReflectionPad3d', generated_inputs['torch.nn.ReflectionPad3d_2'], lib="torch")
