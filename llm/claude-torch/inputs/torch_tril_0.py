
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import copy
import numpy as np

def tril_inputs():
    list_of_inputs = []
    
    input_tensor = torch.randn(3, 3).numpy()
    diagonal = 0
    out = torch.zeros(3, 3).numpy()
    input_dict = {
        "input": input_tensor,
        "diagonal": diagonal,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(4, 6).numpy()
    diagonal = 1
    out = torch.zeros(4, 6).numpy()
    input_dict = {
        "input": input_tensor,
        "diagonal": diagonal,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(4, 6).numpy()
    diagonal = -1
    out = torch.zeros(4, 6).numpy()
    input_dict = {
        "input": input_tensor,
        "diagonal": diagonal,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(5, 5).numpy()
    diagonal = 2
    out = torch.zeros(5, 5).numpy()
    input_dict = {
        "input": input_tensor,
        "diagonal": diagonal,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(2, 2).numpy()
    diagonal = -1
    out = torch.zeros(2, 2).numpy()
    input_dict = {
        "input": input_tensor,
        "diagonal": diagonal,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(6, 4).numpy()
    diagonal = 0
    out = torch.zeros(6, 4).numpy()
    input_dict = {
        "input": input_tensor,
        "diagonal": diagonal,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(3, 3).numpy()
    diagonal = 5
    out = torch.zeros(3, 3).numpy()
    input_dict = {
        "input": input_tensor,
        "diagonal": diagonal,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(4, 4).numpy()
    diagonal = -3
    out = torch.zeros(4, 4).numpy()
    input_dict = {
        "input": input_tensor,
        "diagonal": diagonal,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(1, 1).numpy()
    diagonal = 0
    out = torch.zeros(1, 1).numpy()
    input_dict = {
        "input": input_tensor,
        "diagonal": diagonal,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(7, 3).numpy()
    diagonal = -2
    out = torch.zeros(7, 3).numpy()
    input_dict = {
        "input": input_tensor,
        "diagonal": diagonal,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.tril"] = tril_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.tril' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.tril'.")


check_valid('torch.tril', generated_inputs['torch.tril'], lib="torch", suffix=0)
