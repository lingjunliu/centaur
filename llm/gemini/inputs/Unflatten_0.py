
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def unflatten_inputs():
    list_of_inputs = []

    # Example 1: Basic unflattening
    input = torch.randn(2, 50).numpy()
    dim = 1
    unflattened_size = (2, 5, 5)
    input_dict = {"input": input, "dim": dim, "unflattened_size": unflattened_size}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2: Unflattening with different dimension
    input = torch.randn(10, 20, 30).numpy()
    dim = 1
    unflattened_size = (4, 5)
    input_dict = {"input": input, "dim": dim, "unflattened_size": unflattened_size}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3: Unflattening a different shaped tensor
    input = torch.randn(3, 128).numpy()
    dim = 1
    unflattened_size = (8, 16)
    input_dict = {"input": input, "dim": dim, "unflattened_size": unflattened_size}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 4: Unflattening with a negative dimension index
    input = torch.randn(5, 25).numpy()
    dim = -1
    unflattened_size = (5, 5)
    input_dict = {"input": input, "dim": dim, "unflattened_size": unflattened_size}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 5: Unflattening a 3D tensor
    input = torch.randn(2, 10, 5).numpy()
    dim = 1
    unflattened_size = (2, 5)
    input_dict = {"input": input, "dim": dim, "unflattened_size": unflattened_size}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.Unflatten"] = unflatten_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.Unflatten' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Unflatten'.")

check_valid('torch.nn.Unflatten', generated_inputs['torch.nn.Unflatten'], lib="torch")
