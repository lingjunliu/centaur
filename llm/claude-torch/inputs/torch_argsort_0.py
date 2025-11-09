
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def argsort_inputs():
    list_of_inputs = []
    
    input_dict = {
        "input": np.array([3.0, 1.0, 4.0, 2.0]),
        "dim": -1,
        "descending": False,
        "stable": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[0.5, 2.0], [1.5, 0.5], [0.1, 3.0]]),
        "dim": 0,
        "descending": False,
        "stable": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[3.0, 1.0, 4.0], [2.0, 5.0, 1.0]]),
        "dim": 1,
        "descending": True,
        "stable": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[[1.0, 3.0], [2.0, 0.0]], [[4.0, 1.0], [3.0, 2.0]]]),
        "dim": 2,
        "descending": False,
        "stable": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([-1.0, -3.0, 2.0, 0.0, -2.0]),
        "dim": -1,
        "descending": False,
        "stable": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([1.0, 2.0, 1.0, 3.0, 2.0]),
        "dim": 0,
        "descending": False,
        "stable": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[5.0, 2.0, 8.0, 2.0], [1.0, 3.0, 1.0, 4.0]]),
        "dim": 1,
        "descending": True,
        "stable": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[1.0, 4.0], [3.0, 2.0], [5.0, 0.0]]),
        "dim": -2,
        "descending": False,
        "stable": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([42.0]),
        "dim": 0,
        "descending": False,
        "stable": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[[[1.0, 2.0], [3.0, 0.0]]]]),
        "dim": 3,
        "descending": True,
        "stable": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.argsort"] = argsort_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.argsort' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.argsort'.")


check_valid('torch.argsort', generated_inputs['torch.argsort'], lib="torch", suffix=0)
