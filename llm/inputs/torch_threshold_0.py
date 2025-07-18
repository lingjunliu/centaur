
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def threshold_inputs():
    list_of_inputs = []

    # Input 1: Basic case with 1D tensor
    input_dict = {
        'input': np.array([-5.0, -2.0, 0.0, 3.0, 6.0], dtype=np.float32),
        'threshold': 0.0,
        'value': 10.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D tensor with negative threshold
    input_dict = {
        'input': np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32),
        'threshold': -2.5,
        'value': 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: All elements are greater than the threshold
    input_dict = {
        'input': np.array([10.0, 20.0, 30.0], dtype=np.float32),
        'threshold': 5.0,
        'value': -1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: All elements are less than or equal to the threshold
    input_dict = {
        'input': np.array([-1.0, 0.0, 1.0], dtype=np.float32),
        'threshold': 1.0,
        'value': 100.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D tensor
    input_dict = {
        'input': np.arange(-12, 12, 1, dtype=np.float32).reshape(2, 3, 4),
        'threshold': 0.0,
        'value': 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Input with float64 dtype
    input_dict = {
        'input': np.array([1.1, 2.2, 3.3, 4.4], dtype=np.float64),
        'threshold': 2.5,
        'value': -9.9
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty input tensor
    input_dict = {
        'input': np.array([], dtype=np.float32),
        'threshold': 10.0,
        'value': 20.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Zeros tensor, threshold at zero
    input_dict = {
        'input': np.zeros((3, 3), dtype=np.float32),
        'threshold': 0.0,
        'value': 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large values
    input_dict = {
        'input': np.array([1e6, 1e7, 1e8], dtype=np.float32),
        'threshold': 5e7,
        'value': 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Small decimal values
    input_dict = {
        'input': np.array([0.001, 0.0005, 0.0001], dtype=np.float32),
        'threshold': 0.0006,
        'value': -1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: Threshold and value are the same
    input_dict = {
        'input': np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=np.float32),
        'threshold': 5.0,
        'value': 5.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: High-dimensional empty tensor
    input_dict = {
        'input': np.empty((2, 0, 3), dtype=np.float32),
        'threshold': 0.0,
        'value': 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

# Assume generated_inputs dictionary is pre-initialized
generated_inputs["torch.threshold"] = threshold_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.threshold' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.threshold'.")

check_valid('torch.threshold', generated_inputs['torch.threshold'], lib="torch", suffix=0)
