
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import copy
import numpy as np

def diag_embed_inputs():
    list_of_inputs = []
    
    input_dict = {
        "input": np.array([1.0, 2.0, 3.0]),
        "offset": 0,
        "dim1": -2,
        "dim2": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "offset": 1,
        "dim1": -2,
        "dim2": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[5.0, 6.0], [7.0, 8.0]]),
        "offset": -1,
        "dim1": -2,
        "dim2": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
        "offset": 0,
        "dim1": -2,
        "dim2": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([10.0, 20.0, 30.0, 40.0]),
        "offset": 0,
        "dim1": 0,
        "dim2": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([1, 2, 3, 4, 5]),
        "offset": 2,
        "dim1": -2,
        "dim2": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([7.0, 8.0, 9.0]),
        "offset": 3,
        "dim1": -2,
        "dim2": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([1.5, 2.5, 3.5]),
        "offset": -3,
        "dim1": -2,
        "dim2": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        "offset": 0,
        "dim1": 1,
        "dim2": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([42.0]),
        "offset": 0,
        "dim1": -2,
        "dim2": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.diag_embed"] = diag_embed_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.diag_embed' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.diag_embed'.")


check_valid('torch.diag_embed', generated_inputs['torch.diag_embed'], lib="torch", suffix=0)
