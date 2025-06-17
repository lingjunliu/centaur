
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def hstack_inputs():
    list_of_inputs = []

    # Case 1: 1D tensors
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    input_dict = {"tensors": [a, b], "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: 2D tensors
    a = np.array([[1], [2], [3]])
    b = np.array([[4], [5], [6]])
    input_dict = {"tensors": [a, b], "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Multiple tensors with different shapes
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    input_dict = {"tensors": [a, b], "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Negative values
    a = np.array([-1, -2, -3])
    b = np.array([4, 5, 6])
    input_dict = {"tensors": [a, b], "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Float tensors
    a = np.array([1.1, 2.2, 3.3])
    b = np.array([4.4, 5.5, 6.6])
    input_dict = {"tensors": [a, b], "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.hstack"] = hstack_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.hstack' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.hstack'.")

check_valid('torch.hstack', generated_inputs['torch.hstack'], lib="torch")
