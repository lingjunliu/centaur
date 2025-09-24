
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def vstack_inputs():
    list_of_inputs = []

    # Case 1: Basic 1D tensors
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    input_dict = {"tensors": [a, b], "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Basic 2D tensors
    a = np.array([[1], [2], [3]])
    b = np.array([[4], [5], [6]])
    input_dict = {"tensors": [a, b], "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Mixed 1D and 2D tensors
    a = np.array([1, 2, 3])
    b = np.array([[4, 5, 6]])
    input_dict = {"tensors": [a, b], "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Different shapes that are still valid
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    input_dict = {"tensors": [a, b], "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Negative values and different data types (int)
    a = np.array([-1, -2, -3], dtype=int)
    b = np.array([4, 5, 6], dtype=int)
    input_dict = {"tensors": [a, b], "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.vstack"] = vstack_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.vstack' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.vstack'.")

check_valid('torch.vstack', generated_inputs['torch.vstack'], lib="torch")
