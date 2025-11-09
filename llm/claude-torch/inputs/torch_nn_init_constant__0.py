
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def constant_inputs():
    list_of_inputs = []
    
    tensor = torch.zeros(5).numpy()
    val = 3.5
    input_dict = {
        "tensor": tensor,
        "val": val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    tensor = torch.zeros((3, 4)).numpy()
    val = -2.0
    input_dict = {
        "tensor": tensor,
        "val": val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    tensor = torch.ones((2, 3, 4)).numpy()
    val = 0.0
    input_dict = {
        "tensor": tensor,
        "val": val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    tensor = torch.randn((2, 2, 2, 2)).numpy()
    val = 0.001
    input_dict = {
        "tensor": tensor,
        "val": val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    tensor = torch.zeros(1).numpy()
    val = 100.0
    input_dict = {
        "tensor": tensor,
        "val": val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    tensor = torch.zeros((10, 10)).numpy()
    val = -50.5
    input_dict = {
        "tensor": tensor,
        "val": val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    tensor = torch.empty(8).numpy()
    val = 1.234567
    input_dict = {
        "tensor": tensor,
        "val": val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    tensor = torch.zeros((1, 2, 3, 4, 5)).numpy()
    val = -0.0001
    input_dict = {
        "tensor": tensor,
        "val": val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    tensor = torch.zeros((5, 5)).numpy()
    val = 1.0
    input_dict = {
        "tensor": tensor,
        "val": val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    tensor = torch.zeros((4, 3, 2)).numpy()
    val = -1.0
    input_dict = {
        "tensor": tensor,
        "val": val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    tensor = torch.zeros(1000).numpy()
    val = 0.5
    input_dict = {
        "tensor": tensor,
        "val": val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    tensor = torch.zeros((7, 3)).numpy()
    val = 42.0
    input_dict = {
        "tensor": tensor,
        "val": val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.init.constant_"] = constant_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.init.constant_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.init.constant_'.")


check_valid('torch.nn.init.constant_', generated_inputs['torch.nn.init.constant_'], lib="torch", suffix=0)
