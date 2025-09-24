
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def adaptive_max_pool3d_inputs():
    list_of_inputs = []

    # Input 1
    input = np.random.randn(1, 3, 10, 10, 10).astype(np.float32)
    output_size = (5, 5, 5)
    return_indices = False
    input_dict = {"output_size": output_size, "return_indices": return_indices, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input = np.random.randn(2, 4, 12, 12, 12).astype(np.float32)
    output_size = (7, 8, 9)
    return_indices = True
    input_dict = {"output_size": output_size, "return_indices": return_indices, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input = np.random.randn(1, 2, 8, 8, 8).astype(np.float32)
    output_size = (4, 4, 4)
    return_indices = False
    input_dict = {"output_size": output_size, "return_indices": return_indices, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    input = np.random.randn(3, 5, 15, 15, 15).astype(np.float32)
    output_size = (6, 6, 6)
    return_indices = True
    input_dict = {"output_size": output_size, "return_indices": return_indices, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input = np.random.randn(1, 1, 5, 5, 5).astype(np.float32)
    output_size = (3, 3, 3)
    return_indices = False
    input_dict = {"output_size": output_size, "return_indices": return_indices, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input = np.random.randn(2, 3, 10, 11, 12).astype(np.float32)
    output_size = (5, 5, 5)
    return_indices = True
    input_dict = {"output_size": output_size, "return_indices": return_indices, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    input = np.random.randn(1, 1, 7, 8, 9).astype(np.float32)
    output_size = (2, 3, 4)
    return_indices = False
    input_dict = {"output_size": output_size, "return_indices": return_indices, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input = np.random.randn(4, 2, 16, 16, 16).astype(np.float32)
    output_size = (8, 8, 8)
    return_indices = True
    input_dict = {"output_size": output_size, "return_indices": return_indices, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    input = np.random.randn(1, 5, 9, 10, 11).astype(np.float32)
    output_size = (4, 5, 6)
    return_indices = False
    input_dict = {"output_size": output_size, "return_indices": return_indices, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input = np.random.randn(2, 2, 6, 7, 8).astype(np.float32)
    output_size = (3, 4, 4)
    return_indices = True
    input_dict = {"output_size": output_size, "return_indices": return_indices, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11
    input = np.random.randn(1, 4, 11, 12, 13).astype(np.float32)
    output_size = (5, 5, 7)
    return_indices = False
    input_dict = {"output_size": output_size, "return_indices": return_indices, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.AdaptiveMaxPool3d_2"] = adaptive_max_pool3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.AdaptiveMaxPool3d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.AdaptiveMaxPool3d_2'.")

check_valid('torch.nn.AdaptiveMaxPool3d', generated_inputs['torch.nn.AdaptiveMaxPool3d_2'], lib="torch", suffix=2)
