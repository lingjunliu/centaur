
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def adaptive_max_pool1d_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = torch.randn(1, 64, 8).numpy()
    output_size = (5,)
    return_indices = False
    input_dict = {"output_size": output_size, "return_indices": return_indices, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = torch.randn(2, 32, 16).numpy()
    output_size = (10,)
    return_indices = True
    input_dict = {"output_size": output_size, "return_indices": return_indices, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = torch.randn(1, 1, 20).numpy()
    output_size = (1,)
    return_indices = False
    input_dict = {"output_size": output_size, "return_indices": return_indices, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = torch.randn(3, 16).numpy()
    output_size = (7,)
    return_indices = True
    input_dict = {"output_size": output_size, "return_indices": return_indices, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    input_tensor = torch.randn(4, 8, 32).numpy()
    output_size = (20,)
    return_indices = False
    input_dict = {"output_size": output_size, "return_indices": return_indices, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = torch.randn(1, 3, 5).numpy()
    output_size = (3,)
    return_indices = True
    input_dict = {"output_size": output_size, "return_indices": return_indices, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = torch.randn(1, 20).numpy()
    output_size = (15,)
    return_indices = False
    input_dict = {"output_size": output_size, "return_indices": return_indices, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = torch.randn(2, 10, 100).numpy()
    output_size = (50,)
    return_indices = True
    input_dict = {"output_size": output_size, "return_indices": return_indices, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    input_tensor = torch.randn(5, 1, 128).numpy()
    output_size = (64,)
    return_indices = False
    input_dict = {"output_size": output_size, "return_indices": return_indices, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = torch.randn(3, 64).numpy()
    output_size = (32,)
    return_indices = True
    input_dict = {"output_size": output_size, "return_indices": return_indices, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.AdaptiveMaxPool1d_2"] = adaptive_max_pool1d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.AdaptiveMaxPool1d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.AdaptiveMaxPool1d_2'.")

check_valid('torch.nn.AdaptiveMaxPool1d', generated_inputs['torch.nn.AdaptiveMaxPool1d_2'], lib="torch", suffix=2)
