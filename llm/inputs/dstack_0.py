
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def dstack_inputs():
    list_of_inputs = []

    # Case 1: Two 1D tensors
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    input_dict = {"tensors": [a, b], "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Two 2D tensors
    a = np.array([[1], [2], [3]])
    b = np.array([[4], [5], [6]])
    input_dict = {"tensors": [a, b], "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Two 3D tensors
    a = np.array([[[1]], [[2]], [[3]]])
    b = np.array([[[4]], [[5]], [[6]]])
    input_dict = {"tensors": [a, b], "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Three 1D tensors
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    c = np.array([7, 8, 9])
    input_dict = {"tensors": [a, b, c], "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Two 2D tensors with compatible shapes
    a = np.array([[1, 2, 3]])
    b = np.array([[4, 5, 6]])
    input_dict = {"tensors": [a, b], "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.dstack"] = dstack_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.dstack' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.dstack'.")

check_valid('torch.dstack', generated_inputs['torch.dstack'], lib="torch")
