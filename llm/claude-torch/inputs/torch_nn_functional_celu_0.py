
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def celu_inputs():
    list_of_inputs = []
    
    input_dict = {
        "input": np.array([1.0, 2.0, -1.0, -2.0]),
        "alpha": 1.0,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[1.0, -1.0], [2.0, -2.0]]),
        "alpha": 2.0,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[[1.0, -1.0], [0.5, -0.5]], [[0.0, 1.0], [-0.5, 0.5]]]),
        "alpha": 0.5,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([0.1, 0.5, 1.0, 2.0, 5.0]),
        "alpha": 1.0,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([-0.1, -0.5, -1.0, -2.0, -5.0]),
        "alpha": 1.0,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([0.0, 1.0, -1.0, 0.0, 2.0]),
        "alpha": 1.5,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([1.0, -1.0, 2.0, -2.0]),
        "alpha": 10.0,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([0.001, -0.001, 0.01, -0.01]),
        "alpha": 1.0,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.random.randn(2, 3, 4, 5).astype(np.float32),
        "alpha": 1.0,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([1.5]),
        "alpha": 1.0,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.random.randn(10, 10).astype(np.float32),
        "alpha": 0.1,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.celu"] = celu_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.celu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.celu'.")


check_valid('torch.nn.functional.celu', generated_inputs['torch.nn.functional.celu'], lib="torch", suffix=0)
