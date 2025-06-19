
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def logical_and_inputs():
    list_of_inputs = []

    # Case 1: Basic boolean tensors
    input1 = np.array([True, False, True, False])
    input2 = np.array([True, True, False, False])
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Integer tensors (0 is False, non-zero is True)
    input1 = np.array([1, 0, 2, -1], dtype=np.int32)
    input2 = np.array([0, 1, -2, 0], dtype=np.int32)
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Floating-point tensors (0.0 is False, non-zero is True)
    input1 = np.array([1.0, 0.0, 2.5, -1.2], dtype=np.float32)
    input2 = np.array([0.0, 1.1, -2.0, 0.0], dtype=np.float32)
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Multi-dimensional tensors
    input1 = np.array([[True, False], [True, True]])
    input2 = np.array([[False, True], [True, False]])
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Different shapes with broadcasting (input2 is a scalar)
    input1 = np.array([True, False, True])
    input2 = np.array(True)
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.logical_and"] = logical_and_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.logical_and' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.logical_and'.")

check_valid('torch.logical_and', generated_inputs['torch.logical_and'], lib="torch")
