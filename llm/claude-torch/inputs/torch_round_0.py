
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import copy
import numpy as np

def round_inputs():
    list_of_inputs = []
    
    input_dict = {
        "input": np.array([4.7, -2.3, 9.1, -7.7]),
        "decimals": 0,
        "out": np.array([0.0, 0.0, 0.0, 0.0])
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([-0.5, 0.5, 1.5, 2.5]),
        "decimals": 0,
        "out": np.array([0.0, 0.0, 0.0, 0.0])
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([0.1234567]),
        "decimals": 3,
        "out": np.array([0.0])
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([1200.1234567]),
        "decimals": -3,
        "out": np.array([0.0])
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[1.234, 2.567], [3.891, 4.123]]),
        "decimals": 1,
        "out": np.array([[0.0, 0.0], [0.0, 0.0]])
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[[12345.678, 98765.432]]]),
        "decimals": -2,
        "out": np.array([[[0.0, 0.0]]])
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([-3.14159, -2.71828, -1.41421]),
        "decimals": 2,
        "out": np.array([0.0, 0.0, 0.0])
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[10.5, 20.5, 30.5], [40.5, 50.5, 60.5]]),
        "decimals": 0,
        "out": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([3.141592653589793]),
        "decimals": 5,
        "out": np.array([0.0])
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([100.456, 200.789, 300.123]),
        "decimals": -1,
        "out": np.array([0.0, 0.0, 0.0])
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.round"] = round_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.round' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.round'.")


check_valid('torch.round', generated_inputs['torch.round'], lib="torch", suffix=0)
