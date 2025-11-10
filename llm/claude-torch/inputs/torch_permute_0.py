
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def permute_inputs():
    list_of_inputs = []
    
    input_tensor = torch.randn(2, 3, 5).numpy()
    dims = (2, 0, 1)
    input_dict = {
        "input": input_tensor,
        "dims": dims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(4, 6).numpy()
    dims = (1, 0)
    input_dict = {
        "input": input_tensor,
        "dims": dims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(2, 3, 4, 5).numpy()
    dims = (3, 1, 0, 2)
    input_dict = {
        "input": input_tensor,
        "dims": dims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(10).numpy()
    dims = (0,)
    input_dict = {
        "input": input_tensor,
        "dims": dims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(3, 4, 5).numpy()
    dims = (0, 1, 2)
    input_dict = {
        "input": input_tensor,
        "dims": dims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(2, 3, 4, 5, 6).numpy()
    dims = (4, 2, 0, 3, 1)
    input_dict = {
        "input": input_tensor,
        "dims": dims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(10, 20).numpy()
    dims = (1, 0)
    input_dict = {
        "input": input_tensor,
        "dims": dims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.ones(3, 3, 3).numpy()
    dims = (1, 2, 0)
    input_dict = {
        "input": input_tensor,
        "dims": dims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(1, 2, 3, 4).numpy()
    dims = (0, 2, 3, 1)
    input_dict = {
        "input": input_tensor,
        "dims": dims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.zeros(5, 10, 15).numpy()
    dims = (2, 1, 0)
    input_dict = {
        "input": input_tensor,
        "dims": dims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.permute"] = permute_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.permute' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.permute'.")


check_valid('torch.permute', generated_inputs['torch.permute'], lib="torch", suffix=0)
