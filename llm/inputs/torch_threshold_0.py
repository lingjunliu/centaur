
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def threshold_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D tensor, positive threshold and value
    input_dict = {
        "input": np.array([1.0, 2.0, 0.5, 3.0]),
        "threshold": 1.5,
        "value": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D tensor, negative threshold and value
    input_dict = {
        "input": np.array([[-1.0, -2.0], [0.5, 3.0]]),
        "threshold": -1.5,
        "value": -0.5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D tensor, zero threshold and value
    input_dict = {
        "input": np.array([[[1.0, 2.0], [0.5, 3.0]], [[-1.0, -2.0], [-0.5, -3.0]]]),
        "threshold": 0.0,
        "value": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: All elements below threshold
    input_dict = {
        "input": np.array([0.1, 0.2, 0.3]),
        "threshold": 0.5,
        "value": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: All elements above threshold
    input_dict = {
        "input": np.array([1.1, 1.2, 1.3]),
        "threshold": 0.5,
        "value": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Threshold equals largest value
    input_dict = {
        "input": np.array([1.0, 2.0, 0.5, 3.0]),
        "threshold": 3.0,
        "value": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Threshold equals smallest value
    input_dict = {
        "input": np.array([1.0, 2.0, 0.5, 3.0]),
        "threshold": 0.5,
        "value": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Value equals Threshold
    input_dict = {
        "input": np.array([1.0, 2.0, 0.5, 3.0]),
        "threshold": 1.5,
        "value": 1.5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large threshold and value
    input_dict = {
        "input": np.array([1.0, 2.0, 0.5, 3.0]),
        "threshold": 100.0,
        "value": 50.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Negative threshold and positive value
    input_dict = {
        "input": np.array([-1.0, -2.0, -0.5, -3.0]),
        "threshold": -1.5,
        "value": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.threshold"] = threshold_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.threshold' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.threshold'.")

check_valid('torch.threshold', generated_inputs['torch.threshold'], lib="torch", suffix=0)
