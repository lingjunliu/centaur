
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def adaptive_max_pool3d_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = torch.randn(1, 3, 16, 16, 16).numpy()
    output_size = 8
    return_indices = False
    input_dict = {"output_size": output_size, "return_indices": return_indices, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = torch.randn(2, 4, 32, 32, 32).numpy()
    output_size = 16
    return_indices = True
    input_dict = {"output_size": output_size, "return_indices": return_indices, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = torch.randn(1, 1, 64, 64, 64).numpy()
    output_size = 32
    return_indices = False
    input_dict = {"output_size": output_size, "return_indices": return_indices, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = torch.randn(4, 2, 8, 8, 8).numpy()
    output_size = 4
    return_indices = True
    input_dict = {"output_size": output_size, "return_indices": return_indices, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = torch.randn(1, 3, 10, 12, 14).numpy()
    output_size = 5
    return_indices = False
    input_dict = {"output_size": output_size, "return_indices": return_indices, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = torch.randn(2, 1, 20, 24, 28).numpy()
    output_size = 10
    return_indices = True
    input_dict = {"output_size": output_size, "return_indices": return_indices, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = torch.randn(1, 2, 5, 7, 9).numpy()
    output_size = 3
    return_indices = False
    input_dict = {"output_size": output_size, "return_indices": return_indices, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = torch.randn(3, 3, 15, 21, 27).numpy()
    output_size = 7
    return_indices = True
    input_dict = {"output_size": output_size, "return_indices": return_indices, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    input_tensor = torch.randn(1, 64, 10, 9, 8).numpy()
    output_size = 7
    return_indices = False
    input_dict = {"output_size": output_size, "return_indices": return_indices, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = torch.randn(4, 16, 20, 18, 16).numpy()
    output_size = 9
    return_indices = True
    input_dict = {"output_size": output_size, "return_indices": return_indices, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.AdaptiveMaxPool3d_1"] = adaptive_max_pool3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.AdaptiveMaxPool3d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.AdaptiveMaxPool3d_1'.")

check_valid('torch.nn.AdaptiveMaxPool3d', generated_inputs['torch.nn.AdaptiveMaxPool3d_1'], lib="torch", suffix=1)
