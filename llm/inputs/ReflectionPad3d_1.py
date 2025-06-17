
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def ReflectionPad3d_inputs():
    list_of_inputs = []

    # Case 1: int padding, NCHW
    input_tensor = torch.randn(2, 3, 4, 5, 6).numpy()
    padding = 1
    input_dict = {"input": input_tensor, "padding": padding}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: tuple padding, NCHW
    input_tensor = torch.randn(1, 1, 3, 3, 3).numpy()
    padding = (1, 1, 1, 1, 1, 1)
    input_dict = {"input": input_tensor, "padding": padding}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: int padding, single channel CHW
    input_tensor = torch.randn(1, 3, 3, 3).numpy()
    padding = 2
    input_dict = {"input": input_tensor, "padding": padding}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 4: tuple padding, CHW - removed because this is 3D tensor which is not the same as the described 3D tensor in the docs.
    #input_tensor = torch.randn(3, 4, 5).numpy()
    #padding = (2, 1, 0, 1, 0, 2)
    #input_dict = {"input": input_tensor, "padding": padding}
    #list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: int padding, larger tensor NCHW
    input_tensor = torch.randn(4, 5, 7, 8, 9).numpy()
    padding = 3
    input_dict = {"input": input_tensor, "padding": padding}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.ReflectionPad3d_1"] = ReflectionPad3d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.ReflectionPad3d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ReflectionPad3d_1'.")

check_valid('torch.nn.ReflectionPad3d', generated_inputs['torch.nn.ReflectionPad3d_1'], lib="torch")
