
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def AdaptiveAvgPool3d_inputs():
    list_of_inputs = []

    # Test case 1: Single integer output_size
    input1 = torch.randn(1, 3, 10, 12, 14).numpy()
    output_size1 = 5
    input_dict1 = {"output_size": output_size1, "input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Test case 2: Tuple output_size
    input2 = torch.randn(1, 5, 8, 9, 10).numpy()
    output_size2 = (4, 5, 6)
    input_dict2 = {"output_size": output_size2, "input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Test case 3: None in output_size tuple
    input3 = torch.randn(1, 7, 12, 15, 18).numpy()
    output_size3 = (None, 7, None)
    input_dict3 = {"output_size": output_size3, "input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Test case 4: Different input shape (C, D, H, W)
    input4 = torch.randn(3, 10, 12, 14).numpy()
    output_size4 = 6
    input_dict4 = {"output_size": output_size4, "input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Test case 5: Different input shape (N, C, D, H, W) with small values
    input5 = torch.randn(2, 2, 3, 4, 5).numpy()
    output_size5 = (2, 2, 2)
    input_dict5 = {"output_size": output_size5, "input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.AdaptiveAvgPool3d_1"] = AdaptiveAvgPool3d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.AdaptiveAvgPool3d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.AdaptiveAvgPool3d_1'.")

check_valid('torch.nn.AdaptiveAvgPool3d', generated_inputs['torch.nn.AdaptiveAvgPool3d_1'], lib="torch")
