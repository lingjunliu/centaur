
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def reflectionpad2d_inputs():
    list_of_inputs = []
    
    padding = (2, 2, 2, 2)
    input_tensor = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3).numpy()
    input_dict = {
        "padding": padding,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    padding = (1, 1, 2, 0)
    input_tensor = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3).numpy()
    input_dict = {
        "padding": padding,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    padding = (1, 1, 1, 1)
    input_tensor = torch.randn(2, 3, 4, 4).numpy()
    input_dict = {
        "padding": padding,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    padding = (1, 2, 1, 1)
    input_tensor = torch.randn(2, 5, 5).numpy()
    input_dict = {
        "padding": padding,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    padding = (0, 2, 1, 0)
    input_tensor = torch.randn(1, 1, 4, 4).numpy()
    input_dict = {
        "padding": padding,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    padding = (2, 2, 3, 3)
    input_tensor = torch.randn(1, 3, 8, 8).numpy()
    input_dict = {
        "padding": padding,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    padding = (1, 0, 0, 0)
    input_tensor = torch.randn(1, 1, 5, 5).numpy()
    input_dict = {
        "padding": padding,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    padding = (1, 1, 1, 1)
    input_tensor = torch.randn(1, 1, 2, 2).numpy()
    input_dict = {
        "padding": padding,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    padding = (1, 1, 1, 1)
    input_tensor = torch.randn(1, 10, 6, 6).numpy()
    input_dict = {
        "padding": padding,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    padding = (2, 1, 1, 2)
    input_tensor = torch.randn(1, 1, 3, 7).numpy()
    input_dict = {
        "padding": padding,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.ReflectionPad2d_2"] = reflectionpad2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.ReflectionPad2d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ReflectionPad2d_2'.")


check_valid('torch.nn.ReflectionPad2d', generated_inputs['torch.nn.ReflectionPad2d_2'], lib="torch", suffix=2)
