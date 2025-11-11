
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def equal_inputs():
    list_of_inputs = []
    
    input_tensor = torch.tensor([1, 2, 3]).numpy()
    other_tensor = torch.tensor([1, 2, 3]).numpy()
    input_dict = {
        "input": input_tensor,
        "other": other_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.tensor([1, 2, 3]).numpy()
    other_tensor = torch.tensor([1, 2, 4]).numpy()
    input_dict = {
        "input": input_tensor,
        "other": other_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.tensor([[1, 2], [3, 4]]).numpy()
    other_tensor = torch.tensor([[1, 2], [3, 4]]).numpy()
    input_dict = {
        "input": input_tensor,
        "other": other_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.tensor([[1, 2, 3]]).numpy()
    other_tensor = torch.tensor([[1], [2], [3]]).numpy()
    input_dict = {
        "input": input_tensor,
        "other": other_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.tensor([1.5, 2.5, 3.5]).numpy()
    other_tensor = torch.tensor([1.5, 2.5, 3.5]).numpy()
    input_dict = {
        "input": input_tensor,
        "other": other_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.tensor([-1, -2, -3]).numpy()
    other_tensor = torch.tensor([-1, -2, -3]).numpy()
    input_dict = {
        "input": input_tensor,
        "other": other_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.zeros((2, 3, 4)).numpy()
    other_tensor = torch.zeros((2, 3, 4)).numpy()
    input_dict = {
        "input": input_tensor,
        "other": other_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.tensor(5).numpy()
    other_tensor = torch.tensor(5).numpy()
    input_dict = {
        "input": input_tensor,
        "other": other_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.tensor([]).numpy()
    other_tensor = torch.tensor([]).numpy()
    input_dict = {
        "input": input_tensor,
        "other": other_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.ones((10, 10)).numpy()
    other_tensor = torch.ones((10, 10)).numpy()
    input_dict = {
        "input": input_tensor,
        "other": other_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.equal"] = equal_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.equal' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.equal'.")


check_valid('torch.equal', generated_inputs['torch.equal'], lib="torch", suffix=0)
