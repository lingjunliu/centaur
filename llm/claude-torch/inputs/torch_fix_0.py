
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def fix_inputs():
    list_of_inputs = []
    
    # Input 1: 1D positive floats
    input_dict = {
        "input": np.array([1.5, 2.7, 3.2, 4.9]),
        "out": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 1D negative floats
    input_dict = {
        "input": np.array([-1.5, -2.7, -3.2, -4.9]),
        "out": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 1D mixed positive and negative floats
    input_dict = {
        "input": np.array([-1.5, 2.7, -3.2, 4.9]),
        "out": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 2D array
    input_dict = {
        "input": np.array([[1.6, 2.3, 3.8], [4.1, 5.9, 6.2]]),
        "out": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 2D array with negative values
    input_dict = {
        "input": np.array([[-1.6, -2.3, -3.8], [-4.1, -5.9, -6.2]]),
        "out": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 3D array
    input_dict = {
        "input": np.array([[[1.5, 2.5], [3.5, 4.5]], [[5.5, 6.5], [7.5, 8.5]]]),
        "out": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: scalar
    input_dict = {
        "input": np.array(3.14159),
        "out": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: with out parameter
    input_dict = {
        "input": np.array([1.5, 2.7, 3.2, 4.9]),
        "out": np.empty(4),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 2D with out parameter
    input_dict = {
        "input": np.array([[1.6, 2.3], [3.8, 4.1]]),
        "out": np.empty((2, 2)),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: zero values
    input_dict = {
        "input": np.array([0.0, 0.5, -0.5, 0.0]),
        "out": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: large 1D array
    input_dict = {
        "input": np.linspace(-10.5, 10.5, 100),
        "out": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: single element array
    input_dict = {
        "input": np.array([5.678]),
        "out": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.fix"] = fix_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.fix' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.fix'.")


check_valid('torch.fix', generated_inputs['torch.fix'], lib="torch", suffix=0)
