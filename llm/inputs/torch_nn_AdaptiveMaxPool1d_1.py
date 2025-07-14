
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def adaptive_max_pool1d_inputs():
    list_of_inputs = []

    # Input 1
    input = np.random.randn(1, 64, 8).astype(np.float32)
    output_size = 5
    return_indices = False
    input_dict = {"output_size": output_size, "return_indices": return_indices, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input = np.random.randn(2, 32, 16).astype(np.float32)
    output_size = 10
    return_indices = True
    input_dict = {"output_size": output_size, "return_indices": return_indices, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input = np.random.randn(1, 1, 20).astype(np.float32)
    output_size = 1
    return_indices = False
    input_dict = {"output_size": output_size, "return_indices": return_indices, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input = np.random.randn(4, 128, 32).astype(np.float32)
    output_size = 20
    return_indices = True
    input_dict = {"output_size": output_size, "return_indices": return_indices, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input = np.random.randn(1, 3, 5).astype(np.float32)
    output_size = 3
    return_indices = False
    input_dict = {"output_size": output_size, "return_indices": return_indices, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input = np.random.randn(10, 5, 25).astype(np.float32)
    output_size = 15
    return_indices = True
    input_dict = {"output_size": output_size, "return_indices": return_indices, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    input = np.random.randn(1, 64, 1).astype(np.float32)
    output_size = 1
    return_indices = False
    input_dict = {"output_size": output_size, "return_indices": return_indices, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    input = np.random.randn(32, 8).astype(np.float32)
    output_size = 4
    return_indices = True
    input_dict = {"output_size": output_size, "return_indices": return_indices, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    input = np.random.randn(1, 16, 64).astype(np.float32)
    output_size = 32
    return_indices = False
    input_dict = {"output_size": output_size, "return_indices": return_indices, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    input = np.random.randn(2, 1, 128).astype(np.float32)
    output_size = 64
    return_indices = True
    input_dict = {"output_size": output_size, "return_indices": return_indices, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11
    input = np.random.randn(1, 1, 1).astype(np.float32)
    output_size = 1
    return_indices = False
    input_dict = {"output_size": output_size, "return_indices": return_indices, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.AdaptiveMaxPool1d_1"] = adaptive_max_pool1d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.AdaptiveMaxPool1d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.AdaptiveMaxPool1d_1'.")

check_valid('torch.nn.AdaptiveMaxPool1d', generated_inputs['torch.nn.AdaptiveMaxPool1d_1'], lib="torch", suffix=1)
