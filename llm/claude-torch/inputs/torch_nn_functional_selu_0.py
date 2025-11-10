
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def selu_inputs():
    list_of_inputs = []
    
    input_tensor = torch.tensor([1.0, 2.0, 3.0]).numpy()
    input_dict = {
        "input": input_tensor,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    input_dict = {
        "input": input_tensor,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]).numpy()
    input_dict = {
        "input": input_tensor,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).numpy()
    input_dict = {
        "input": input_tensor,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.tensor([[-1.0, -0.5, 0.0], [0.5, 1.0, 1.5]]).numpy()
    input_dict = {
        "input": input_tensor,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(2, 3, 4).numpy()
    input_dict = {
        "input": input_tensor,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(1, 3, 5, 5).numpy()
    input_dict = {
        "input": input_tensor,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.tensor(2.5).numpy()
    input_dict = {
        "input": input_tensor,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.zeros(3, 3).numpy()
    input_dict = {
        "input": input_tensor,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(10, 10).numpy()
    input_dict = {
        "input": input_tensor,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.tensor([0.001, -0.001, 0.0001, -0.0001]).numpy()
    input_dict = {
        "input": input_tensor,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.selu"] = selu_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.selu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.selu'.")


check_valid('torch.nn.functional.selu', generated_inputs['torch.nn.functional.selu'], lib="torch", suffix=0)
