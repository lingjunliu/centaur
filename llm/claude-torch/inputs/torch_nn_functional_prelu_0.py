
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def prelu_inputs():
    list_of_inputs = []
    
    input = torch.tensor([1.0, -2.0, 3.0, -4.0]).numpy()
    weight = torch.tensor(0.25).numpy()
    input_dict = {
        "input": input,
        "weight": weight
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([[1.0, -1.0], [2.0, -2.0]]).numpy()
    weight = torch.tensor(0.1).numpy()
    input_dict = {
        "input": input,
        "weight": weight
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.randn(2, 3, 4).numpy()
    weight = torch.tensor([0.1, 0.2, 0.3]).numpy()
    input_dict = {
        "input": input,
        "weight": weight
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.randn(2, 5, 3, 3).numpy()
    weight = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]).numpy()
    input_dict = {
        "input": input,
        "weight": weight
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    weight = torch.tensor(0.5).numpy()
    input_dict = {
        "input": input,
        "weight": weight
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([[-1.0, -2.0], [-3.0, -4.0]]).numpy()
    weight = torch.tensor(0.3).numpy()
    input_dict = {
        "input": input,
        "weight": weight
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.randn(1, 4, 2, 2, 2).numpy()
    weight = torch.tensor([0.1, 0.2, 0.3, 0.4]).numpy()
    input_dict = {
        "input": input,
        "weight": weight
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor(-5.0).numpy()
    weight = torch.tensor(0.15).numpy()
    input_dict = {
        "input": input,
        "weight": weight
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([[1.0, -1.0], [-2.0, 2.0]]).numpy()
    weight = torch.tensor(-0.1).numpy()
    input_dict = {
        "input": input,
        "weight": weight
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.randn(10, 8, 20).numpy()
    weight = torch.tensor([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]).numpy()
    input_dict = {
        "input": input,
        "weight": weight
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.prelu"] = prelu_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.prelu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.prelu'.")


check_valid('torch.nn.functional.prelu', generated_inputs['torch.nn.functional.prelu'], lib="torch", suffix=0)
