
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def sin_inputs():
    list_of_inputs = []
    
    input_tensor = torch.tensor([0.0, 1.5708, 3.1416, 4.7124]).numpy()
    input_dict = {
        "input": input_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.tensor([-0.5461, -1.2, -2.7266, -0.2746]).numpy()
    input_dict = {
        "input": input_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]).numpy()
    input_dict = {
        "input": input_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(2, 3, 4).numpy()
    input_dict = {
        "input": input_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.tensor([1.5708]).numpy()
    input_dict = {
        "input": input_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.zeros(5).numpy()
    input_dict = {
        "input": input_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.tensor([[-1.0, 0.0, 1.0], [2.0, -2.0, 3.0]]).numpy()
    input_dict = {
        "input": input_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(2, 2, 2, 2).numpy()
    input_dict = {
        "input": input_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.tensor([10.0, 20.0, 30.0, 40.0]).numpy()
    input_dict = {
        "input": input_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.tensor([0.001, 0.002, 0.003]).numpy()
    input_dict = {
        "input": input_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.tensor([1.0, 2.0, 3.0]).numpy()
    out_tensor = torch.empty(3).numpy()
    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.sin"] = sin_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.sin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sin'.")


check_valid('torch.sin', generated_inputs['torch.sin'], lib="torch", suffix=0)
