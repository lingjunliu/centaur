
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def dropout_inputs():
    list_of_inputs = []
    
    input_dict = {
        "input": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        "p": 0.5,
        "training": True,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.random.randn(3, 4).astype(np.float32),
        "p": 0.8,
        "training": True,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.ones((2, 3, 4), dtype=np.float32),
        "p": 0.1,
        "training": True,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.random.randn(2, 3, 8, 8).astype(np.float32),
        "p": 0.3,
        "training": True,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "p": 0.5,
        "training": False,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.random.randn(5, 5).astype(np.float32),
        "p": 0.0,
        "training": True,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([10.0, 20.0, 30.0], dtype=np.float32),
        "p": 1.0,
        "training": True,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.random.randn(1, 2, 3, 4, 5).astype(np.float32),
        "p": 0.4,
        "training": True,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([-1.0, -2.0, 3.0, -4.0, 5.0], dtype=np.float32),
        "p": 0.6,
        "training": True,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.random.randn(10, 10).astype(np.float64),
        "p": 0.25,
        "training": True,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.dropout"] = dropout_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.dropout' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.dropout'.")


check_valid('torch.nn.functional.dropout', generated_inputs['torch.nn.functional.dropout'], lib="torch", suffix=0)
