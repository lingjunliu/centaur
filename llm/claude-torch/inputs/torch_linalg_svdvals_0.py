
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def svdvals_inputs():
    list_of_inputs = []
    
    A = torch.randn(5, 3).numpy()
    input_dict = {
        "A": A,
        "driver": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    A = torch.randn(4, 4).numpy()
    input_dict = {
        "A": A,
        "driver": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    A = torch.randn(3, 7).numpy()
    input_dict = {
        "A": A,
        "driver": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    A = torch.randn(2, 5, 3).numpy()
    input_dict = {
        "A": A,
        "driver": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    A = torch.randn(2, 3, 4, 5).numpy()
    input_dict = {
        "A": A,
        "driver": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    A = torch.randn(6, 4, dtype=torch.float64).numpy()
    input_dict = {
        "A": A,
        "driver": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    A = torch.randn(5, 5, dtype=torch.cfloat).numpy()
    input_dict = {
        "A": A,
        "driver": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    A = torch.randn(4, 6, dtype=torch.cdouble).numpy()
    input_dict = {
        "A": A,
        "driver": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    A = torch.randn(1, 5).numpy()
    input_dict = {
        "A": A,
        "driver": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    A = torch.randn(5, 1).numpy()
    input_dict = {
        "A": A,
        "driver": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    A = torch.randn(3, 2, 8, 6).numpy()
    input_dict = {
        "A": A,
        "driver": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.linalg.svdvals"] = svdvals_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.linalg.svdvals' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.svdvals'.")


check_valid('torch.linalg.svdvals', generated_inputs['torch.linalg.svdvals'], lib="torch", suffix=0)
