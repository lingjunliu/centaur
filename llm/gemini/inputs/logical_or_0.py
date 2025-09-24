
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def logical_or_inputs():
    list_of_inputs = []

    # Case 1: Basic boolean tensors
    input1 = np.array([[True, False], [False, True]])
    input2 = np.array([[False, True], [True, False]])
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Integer tensors (0 is False, non-zero is True)
    input1 = np.array([[0, 1], [-1, 2]], dtype=np.int32)
    input2 = np.array([[2, 0], [0, -3]], dtype=np.int32)
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Float tensors (0.0 is False, non-zero is True)
    input1 = np.array([[0.0, 1.5], [-2.0, 3.0]], dtype=np.float32)
    input2 = np.array([[2.5, 0.0], [0.0, -3.5]], dtype=np.float32)
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Mixed types (int and bool)
    input1 = np.array([[0, 1], [0, 1]], dtype=np.int32)
    input2 = np.array([[True, False], [True, False]])
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Different shapes, but broadcastable
    input1 = np.array([[True, False], [False, True]])
    input2 = np.array([False, True])
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.logical_or"] = logical_or_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.logical_or' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.logical_or'.")

check_valid('torch.logical_or', generated_inputs['torch.logical_or'], lib="torch")
