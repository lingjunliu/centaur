
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import copy
import numpy as np

def reshape_inputs():
    list_of_inputs = []
    
    input = torch.arange(4.).numpy()
    shape = (2, 2)
    input_dict = {
        "input": input,
        "shape": shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([[0, 1], [2, 3]]).numpy()
    shape = (-1,)
    input_dict = {
        "input": input,
        "shape": shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.arange(24).reshape(2, 3, 4).numpy()
    shape = (6, 4)
    input_dict = {
        "input": input,
        "shape": shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.arange(12.).numpy()
    shape = (2, 2, 3)
    input_dict = {
        "input": input,
        "shape": shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.ones((4, 5)).numpy()
    shape = (5, 4)
    input_dict = {
        "input": input,
        "shape": shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.arange(60).numpy()
    shape = (3, -1, 5)
    input_dict = {
        "input": input,
        "shape": shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.randn(2, 3, 4, 5).numpy()
    shape = (6, 20)
    input_dict = {
        "input": input,
        "shape": shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([5.0]).numpy()
    shape = (1, 1)
    input_dict = {
        "input": input,
        "shape": shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.randn(3, 4, 5).numpy()
    shape = (-1,)
    input_dict = {
        "input": input,
        "shape": shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.arange(48).numpy()
    shape = (2, 2, 3, 4)
    input_dict = {
        "input": input,
        "shape": shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([-1, -2, -3, -4, -5, -6]).numpy()
    shape = (2, 3)
    input_dict = {
        "input": input,
        "shape": shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.randn(100).numpy()
    shape = (10, 10)
    input_dict = {
        "input": input,
        "shape": shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.reshape"] = reshape_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.reshape' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.reshape'.")


check_valid('torch.reshape', generated_inputs['torch.reshape'], lib="torch", suffix=0)
