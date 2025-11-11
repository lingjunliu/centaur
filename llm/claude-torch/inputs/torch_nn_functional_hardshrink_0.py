
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def hardshrink_inputs():
    list_of_inputs = []
    
    input = torch.tensor([0.2, 0.8, -0.3, -0.7, 1.5]).numpy()
    lambd = 0.5
    input_dict = {
        "input": input,
        "lambd": lambd
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([[1.0, -0.5, 0.3], [2.0, -1.5, 0.0]]).numpy()
    lambd = 0.8
    input_dict = {
        "input": input,
        "lambd": lambd
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.randn(2, 3, 4).numpy()
    lambd = 0.1
    input_dict = {
        "input": input,
        "lambd": lambd
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([0.1, 0.2, -0.1, -0.2, 0.0]).numpy()
    lambd = 0.5
    input_dict = {
        "input": input,
        "lambd": lambd
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([2.0, -3.0, 5.0, -4.0]).numpy()
    lambd = 0.5
    input_dict = {
        "input": input,
        "lambd": lambd
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.randn(2, 3, 8, 8).numpy()
    lambd = 1.0
    input_dict = {
        "input": input,
        "lambd": lambd
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor(0.3).numpy()
    lambd = 0.5
    input_dict = {
        "input": input,
        "lambd": lambd
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([1.0, -1.0, 0.5, -0.5]).numpy()
    lambd = 0.0
    input_dict = {
        "input": input,
        "lambd": lambd
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([5.0, -5.0, 3.0, -3.0, 2.0]).numpy()
    lambd = 4.0
    input_dict = {
        "input": input,
        "lambd": lambd
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([-1.0, -2.0, -3.0, -0.3, -5.0]).numpy()
    lambd = 0.5
    input_dict = {
        "input": input,
        "lambd": lambd
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.ones((5, 5)).numpy()
    lambd = 0.3
    input_dict = {
        "input": input,
        "lambd": lambd
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.hardshrink"] = hardshrink_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.hardshrink' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.hardshrink'.")


check_valid('torch.nn.functional.hardshrink', generated_inputs['torch.nn.functional.hardshrink'], lib="torch", suffix=0)
