
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def celu_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D array
    input_dict = {
        "input": np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32),
        "alpha": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array
    input_dict = {
        "input": np.array([[-1.0, 0.0], [0.5, 1.0]], dtype=np.float64),
        "alpha": 0.5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different alpha
    input_dict = {
        "input": np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32),
        "alpha": 2.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: All negative values
    input_dict = {
        "input": np.array([-3.0, -2.0, -1.0], dtype=np.float64),
        "alpha": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: All positive values
    input_dict = {
        "input": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "alpha": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Zero alpha
    input_dict = {
        "input": np.array([-1.0, 0.0, 1.0], dtype=np.float64),
        "alpha": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 3D array
    input_dict = {
        "input": np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32),
        "alpha": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Large negative values
    input_dict = {
        "input": np.array([-100.0, -50.0, -10.0], dtype=np.float64),
        "alpha": 5.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Mixed types, float32 and float64
    input_dict = {
        "input": np.array([-1.0, 0.0, 1.0], dtype=np.float32),
        "alpha": 0.5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Edge case with small alpha
    input_dict = {
        "input": np.array([-0.1, 0.0, 0.1], dtype=np.float64),
        "alpha": 0.001
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.celu"] = celu_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.celu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.celu'.")

check_valid('torch.celu', generated_inputs['torch.celu'], lib="torch", suffix=0)
