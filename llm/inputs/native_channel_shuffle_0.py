
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def native_channel_shuffle_inputs():
    list_of_inputs = []

    # Input 1: Basic case with float tensor and valid groups
    input1 = torch.randn(1, 4, 2, 2).numpy()
    groups1 = 2
    input_dict1 = {"input": input1, "groups": groups1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Integer tensor with valid groups
    input2 = torch.randint(0, 10, (1, 6, 3, 3)).numpy()
    groups2 = 3
    input_dict2 = {"input": input2, "groups": groups2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D tensor
    input3 = torch.randn(3, 8, 8).numpy()
    groups3 = 4
    input_dict3 = {"input": input3, "groups": groups3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Different batch size
    input4 = torch.randn(2, 12, 4, 4).numpy()
    groups4 = 4
    input_dict4 = {"input": input4, "groups": groups4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: Larger number of channels and groups
    input5 = torch.randn(1, 32, 1, 1).numpy()
    groups5 = 8
    input_dict5 = {"input": input5, "groups": groups5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: 5D tensor
    input6 = torch.randn(1, 2, 2, 2, 2).numpy()
    groups6 = 1
    input_dict6 = {"input": input6, "groups": groups6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    return list_of_inputs

generated_inputs["torch.native_channel_shuffle"] = native_channel_shuffle_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.native_channel_shuffle' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.native_channel_shuffle'.")

check_valid('torch.native_channel_shuffle', generated_inputs['torch.native_channel_shuffle'], lib="torch")
