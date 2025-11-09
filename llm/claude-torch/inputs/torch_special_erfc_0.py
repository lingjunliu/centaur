
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def special_erfc_inputs():
    list_of_inputs = []
    
    input_dict = {
        "input": np.array([0.0, 1.0, 2.0], dtype=np.float32),
        "out": np.empty(3, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([-2.0, -1.0, -0.5], dtype=np.float32),
        "out": np.empty(3, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32),
        "out": np.empty((2, 2), dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]], dtype=np.float32),
        "out": np.empty((2, 2, 2), dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array(1.5, dtype=np.float32),
        "out": np.empty((), dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32),
        "out": np.empty(4, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([5.0, 10.0, 15.0], dtype=np.float32),
        "out": np.empty(3, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([0.25, 0.75, 1.25], dtype=np.float64),
        "out": np.empty(3, dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([0.001, 0.01, 0.1], dtype=np.float32),
        "out": np.empty(3, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[[[0.5, 1.0], [1.5, 2.0]]]], dtype=np.float32),
        "out": np.empty((1, 1, 2, 2), dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.special.erfc"] = special_erfc_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.special.erfc' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.erfc'.")


check_valid('torch.special.erfc', generated_inputs['torch.special.erfc'], lib="torch", suffix=0)
