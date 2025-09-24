
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def AdaptiveAvgPool3d_inputs():
    list_of_inputs = []

    input = torch.randn(1, 64, 8, 9, 10).numpy()
    output_size = (5, 7, 9)
    input_dict = {"output_size": output_size, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 64, 10, 9, 8).numpy()
    output_size = 7
    input_dict = {"output_size": output_size, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 64, 10, 9, 8).numpy()
    output_size = (7, None, None)
    input_dict = {"output_size": output_size, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(2, 32, 16, 16, 16).numpy()
    output_size = (None, 8, None)
    input_dict = {"output_size": output_size, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 16, 32, 32, 32).numpy()
    output_size = (4, 4, 4)
    input_dict = {"output_size": output_size, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 16, 20, 20, 20).numpy()
    output_size = (None, None, None)
    input_dict = {"output_size": output_size, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(3, 8, 12, 14, 16).numpy()
    output_size = (6, 7, 8)
    input_dict = {"output_size": output_size, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.randn(1, 3, 7, 9, 11).numpy()
    output_size = (7, 9, 11)
    input_dict = {"output_size": output_size, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.AdaptiveAvgPool3d_2"] = AdaptiveAvgPool3d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.AdaptiveAvgPool3d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.AdaptiveAvgPool3d_2'.")

check_valid('torch.nn.AdaptiveAvgPool3d', generated_inputs['torch.nn.AdaptiveAvgPool3d_2'], lib="torch")
