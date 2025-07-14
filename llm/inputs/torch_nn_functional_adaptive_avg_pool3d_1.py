
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def adaptive_avg_pool3d_inputs():
    list_of_inputs = []

    # Input 1
    input = np.random.rand(1, 3, 10, 10, 10).astype(np.float32)
    output_size = (5, 5, 5)
    input_dict = {"input": input, "output_size": output_size}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input = np.random.rand(2, 4, 8, 12, 16).astype(np.float64)
    output_size = (4, 6, 8)
    input_dict = {"input": input, "output_size": output_size}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input = np.random.rand(1, 1, 5, 5, 5).astype(np.float32)
    output_size = (1, 1, 1)
    input_dict = {"input": input, "output_size": output_size}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    input = np.random.rand(3, 2, 7, 9, 11).astype(np.float32)
    output_size = (2, 3, 4)
    input_dict = {"input": input, "output_size": output_size}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    input = np.random.rand(1, 3, 4, 4, 4).astype(np.float32)
    output_size = (2, 2, 2)
    input_dict = {"input": input, "output_size": output_size}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input = np.random.randn(2, 3, 15, 15, 15).astype(np.float32)
    output_size = (7, 7, 7)
    input_dict = {"input": input, "output_size": output_size}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    input = np.random.rand(1, 2, 3, 4, 5).astype(np.float32)
    output_size = (1, 2, 3)
    input_dict = {"input": input, "output_size": output_size}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input = np.random.rand(4, 5, 6, 7, 8).astype(np.float32)
    output_size = (2, 3, 4)
    input_dict = {"input": input, "output_size": output_size}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input = np.random.rand(1, 3, 16, 16, 16).astype(np.float32)
    output_size = (8, 8, 8)
    input_dict = {"input": input, "output_size": output_size}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input = np.random.rand(1, 1, 32, 32, 32).astype(np.float32)
    output_size = (16, 16, 16)
    input_dict = {"input": input, "output_size": output_size}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.adaptive_avg_pool3d_1"] = adaptive_avg_pool3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.adaptive_avg_pool3d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.adaptive_avg_pool3d_1'.")

check_valid('torch.nn.functional.adaptive_avg_pool3d', generated_inputs['torch.nn.functional.adaptive_avg_pool3d_1'], lib="torch", suffix=1)
