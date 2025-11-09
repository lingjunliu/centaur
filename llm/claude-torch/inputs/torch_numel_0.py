
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def numel_inputs():
    list_of_inputs = []
    
    # Input 1: 1D tensor
    input_dict = {
        "input": torch.randn(5).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 2D tensor
    input_dict = {
        "input": torch.zeros(4, 4).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 5D tensor
    input_dict = {
        "input": torch.randn(1, 2, 3, 4, 5).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 3D tensor
    input_dict = {
        "input": torch.ones(2, 3, 4).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: scalar (0D tensor)
    input_dict = {
        "input": torch.tensor(42.0).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 1D tensor with single element
    input_dict = {
        "input": torch.tensor([1.0]).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 4D tensor
    input_dict = {
        "input": torch.randn(2, 3, 4, 5).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: tensor with negative values
    input_dict = {
        "input": torch.tensor([-1.0, -2.0, -3.0, -4.0]).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 2D tensor with different shape
    input_dict = {
        "input": torch.randn(10, 20).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 6D tensor
    input_dict = {
        "input": torch.randn(1, 2, 2, 2, 2, 2).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: 3D tensor with larger dimensions
    input_dict = {
        "input": torch.zeros(5, 6, 7).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: 1D tensor with many elements
    input_dict = {
        "input": torch.arange(100).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.numel"] = numel_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.numel' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.numel'.")


check_valid('torch.numel', generated_inputs['torch.numel'], lib="torch", suffix=0)
