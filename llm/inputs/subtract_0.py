
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def subtract_inputs():
    list_of_inputs = []

    # Case 1: Basic float tensors
    input1 = np.array([1.0, 2.0, 3.0])
    input2 = np.array([0.5, 1.0, 1.5])
    input_dict = {
        "input": input1,
        "other": input2,
        "alpha": 1.0,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Integer tensors with negative values (alpha=1)
    input1 = np.array([-1, 0, 1, 2])
    input2 = np.array([2, 1, 0, -1])
    input_dict = {
        "input": input1,
        "other": input2,
        "alpha": 1,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: 2D tensors (matrices)
    input1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    input2 = np.array([[0.5, 1.0], [1.5, 2.0]])
    input_dict = {
        "input": input1,
        "other": input2,
        "alpha": 1.0,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: 3D tensors (float)
    input1 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    input2 = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]])
    input_dict = {
        "input": input1,
        "other": input2,
        "alpha": 1.0,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Different alpha value (float tensors)
    input1 = np.array([5.0, 10.0, 15.0])
    input2 = np.array([1.0, 2.0, 3.0])
    input_dict = {
        "input": input1,
        "other": input2,
        "alpha": 0.5,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: Broadcasting (float)
    input1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    input2 = np.array([1.0, 2.0, 3.0])
    input_dict = {
        "input": input1,
        "other": input2,
        "alpha": 1.0,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 7: Integer tensor, alpha != 1
    input1 = np.array([5, 10, 15])
    input2 = np.array([1, 2, 3])
    input_dict = {
        "input": input1,
        "other": input2,
        "alpha": 1,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.subtract"] = subtract_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.subtract' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.subtract'.")

check_valid('torch.subtract', generated_inputs['torch.subtract'], lib="torch")
