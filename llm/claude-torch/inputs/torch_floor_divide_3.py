
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def floor_divide_inputs():
    list_of_inputs = []
    
    input = torch.tensor([10.0, 20.0, 30.0]).numpy()
    other = 3.0
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([[12.5, 25.0], [37.5, 50.0]]).numpy()
    other = 5.0
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([-10.0, -20.0, -30.0]).numpy()
    other = 3.0
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([15.0, -15.0, 22.0, -22.0]).numpy()
    other = 4.0
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([[[8.0, 16.0], [24.0, 32.0]], [[40.0, 48.0], [56.0, 64.0]]]).numpy()
    other = 8.0
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([100.0, 200.0, 300.0]).numpy()
    other = -10.0
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor(42.0).numpy()
    other = 7.0
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([10.0, 20.0, 30.0]).numpy()
    other = 2.5
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([1000.0, 2000.0, 3000.0, 4000.0]).numpy()
    other = 100.0
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([0.5, 1.5, 2.5, 3.5]).numpy()
    other = 0.5
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([-50.0, -100.0, -150.0]).numpy()
    other = -5.0
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.floor_divide_3"] = floor_divide_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.floor_divide_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.floor_divide_3'.")


check_valid('torch.floor_divide', generated_inputs['torch.floor_divide_3'], lib="torch", suffix=3)
