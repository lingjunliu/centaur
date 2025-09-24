
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def adaptive_avg_pool1d_inputs():
    list_of_inputs = []

    # Input 1: Basic case with float tensor
    input1 = torch.randn(1, 3, 16).numpy()
    output_size1 = (8,)
    input_dict1 = {"input": input1, "output_size": output_size1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Float tensor
    input2 = torch.randn(2, 4, 32).numpy()
    output_size2 = (16,)
    input_dict2 = {"input": input2, "output_size": output_size2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3:  Multiple output sizes
    input3 = torch.randn(1, 1, 64).numpy()
    output_size3 = (4,)
    input_dict3 = {"input": input3, "output_size": output_size3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 3D input
    input4 = torch.randn(2, 5, 20).numpy()
    output_size4 = (10,)
    input_dict4 = {"input": input4, "output_size": output_size4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: Single element output size
    input5 = torch.randn(1, 2, 8).numpy()
    output_size5 = (1,)
    input_dict5 = {"input": input5, "output_size": output_size5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.nn.functional.adaptive_avg_pool1d_2"] = adaptive_avg_pool1d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.adaptive_avg_pool1d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.adaptive_avg_pool1d_2'.")

check_valid('torch.nn.functional.adaptive_avg_pool1d', generated_inputs['torch.nn.functional.adaptive_avg_pool1d_2'], lib="torch")
