
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def vdot_inputs():
    list_of_inputs = []

    # Example 1: Basic integer tensors
    input1 = np.array([2, 3], dtype=np.int64)
    input2 = np.array([2, 1], dtype=np.int64)
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2: Float tensors
    input1 = np.array([2.5, 3.7], dtype=np.float64)
    input2 = np.array([2.1, 1.2], dtype=np.float64)
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3: Negative values
    input1 = np.array([-2, 3], dtype=np.int64)
    input2 = np.array([2, -1], dtype=np.int64)
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 4: Complex tensors
    input1 = np.array([1 + 2j, 3 - 1j], dtype=np.complex128)
    input2 = np.array([2 + 1j, 4 - 0j], dtype=np.complex128)
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Example 5: Complex tensors with negative real and imaginary parts
    input1 = np.array([-1 - 2j, -3 + 1j], dtype=np.complex128)
    input2 = np.array([-2 + 1j, -4 - 0j], dtype=np.complex128)
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Example 6: Integer tensors with different dtypes that are still compatible
    input1 = np.array([5, 6], dtype=np.int32)
    input2 = np.array([7, 8], dtype=np.int64)
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.vdot"] = vdot_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.vdot' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.vdot'.")

check_valid('torch.vdot', generated_inputs['torch.vdot'], lib="torch")
