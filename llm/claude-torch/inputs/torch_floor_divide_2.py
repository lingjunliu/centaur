
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def floor_divide_inputs():
    list_of_inputs = []
    
    input = torch.tensor([10.0, 20.0, 30.0]).numpy()
    other = 3
    input_dict = {"input": input, "other": other}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([[12.0, 15.0], [18.0, 21.0]]).numpy()
    other = 5
    input_dict = {"input": input, "other": other}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([-10.0, -20.0, -30.0]).numpy()
    other = 3
    input_dict = {"input": input, "other": other}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([[[8.0, 16.0], [24.0, 32.0]], [[40.0, 48.0], [56.0, 64.0]]]).numpy()
    other = 4
    input_dict = {"input": input, "other": other}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([10.0, -10.0, 20.0, -20.0]).numpy()
    other = 7
    input_dict = {"input": input, "other": other}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([100.0]).numpy()
    other = 9
    input_dict = {"input": input, "other": other}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.randn(5, 5).numpy() * 100
    other = 11
    input_dict = {"input": input, "other": other}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([0.0, 5.0, 10.0, 0.0]).numpy()
    other = 2
    input_dict = {"input": input, "other": other}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([10.0, 20.0, 30.0]).numpy()
    other = -3
    input_dict = {"input": input, "other": other}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.ones(2, 2, 2, 2).numpy() * 15
    other = 4
    input_dict = {"input": input, "other": other}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([100, 200, 300], dtype=torch.float32).numpy()
    other = 13
    input_dict = {"input": input, "other": other}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.floor_divide_2"] = floor_divide_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.floor_divide_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.floor_divide_2'.")


check_valid('torch.floor_divide', generated_inputs['torch.floor_divide_2'], lib="torch", suffix=2)
