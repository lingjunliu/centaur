
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def channel_shuffle_inputs():
    list_of_inputs = []

    # Input 1: Basic 4D tensor with groups = 2
    input1 = torch.randn(1, 4, 5, 5).numpy()
    groups1 = 2
    input_dict1 = {"input": input1, "groups": groups1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 3D tensor with groups = 4
    input2 = torch.randn(2, 8, 6).numpy()
    groups2 = 4
    input_dict2 = {"input": input2, "groups": groups2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 5D tensor with groups = 1
    input3 = torch.randn(1, 3, 2, 3, 4).numpy()
    groups3 = 1
    input_dict3 = {"input": input3, "groups": groups3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    # Input 4: 4D tensor with groups equal to the number of channels
    input4 = torch.randn(1, 8, 4, 4).numpy()
    groups4 = 8
    input_dict4 = {"input": input4, "groups": groups4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 4D tensor with integer data type
    input5 = torch.randint(0, 10, (1, 6, 3, 3)).numpy()
    groups5 = 3
    input_dict5 = {"input": input5, "groups": groups5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: 3D tensor with groups != 1, != C
    input6 = torch.randn(1, 12, 10).numpy()
    groups6 = 3
    input_dict6 = {"input": input6, "groups": groups6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    return list_of_inputs

generated_inputs["torch.channel_shuffle"] = channel_shuffle_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.channel_shuffle' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.channel_shuffle'.")

check_valid('torch.channel_shuffle', generated_inputs['torch.channel_shuffle'], lib="torch")
