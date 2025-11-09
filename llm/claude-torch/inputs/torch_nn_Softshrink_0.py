
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def softshrink_inputs():
    list_of_inputs = []
    
    lambd = 0.5
    input_tensor = np.array([1.0, -2.0, 0.3, -0.3, 0.0])
    input_dict = {
        "lambd": lambd,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    lambd = 1.0
    input_tensor = np.array([[2.5, -3.0, 0.5], [1.5, -1.5, 0.0]])
    input_dict = {
        "lambd": lambd,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    lambd = 0.0
    input_tensor = np.array([0.1, -0.1, 1.0, -1.0])
    input_dict = {
        "lambd": lambd,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    lambd = 2.0
    input_tensor = np.array([[[5.0, -5.0], [1.0, -1.0]], [[3.0, -3.0], [0.5, -0.5]]])
    input_dict = {
        "lambd": lambd,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    lambd = 0.25
    input_tensor = np.array([1.5])
    input_dict = {
        "lambd": lambd,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    lambd = 1.5
    input_tensor = np.array([10.0, -10.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0])
    input_dict = {
        "lambd": lambd,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    lambd = 0.1
    input_tensor = np.random.randn(2, 2, 2, 2)
    input_dict = {
        "lambd": lambd,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    lambd = 3.0
    input_tensor = np.array([3.0, -3.0, 3.1, -3.1, 2.9, -2.9])
    input_dict = {
        "lambd": lambd,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    lambd = 0.5
    input_tensor = np.random.randn(10, 10)
    input_dict = {
        "lambd": lambd,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    lambd = 0.75
    input_tensor = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    input_dict = {
        "lambd": lambd,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.Softshrink"] = softshrink_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.Softshrink' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Softshrink'.")


check_valid('torch.nn.Softshrink', generated_inputs['torch.nn.Softshrink'], lib="torch", suffix=0)
