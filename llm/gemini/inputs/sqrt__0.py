
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def sqrt__inputs():
    list_of_inputs = []

    # Input 1: 1D float tensor with positive values
    input1 = np.array([1.0, 4.0, 9.0, 16.0], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D float tensor with positive and zero values
    input2 = np.array([[0.0, 1.0], [4.0, 9.0]], dtype=np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D float tensor
    input3 = np.random.rand(2, 3, 4).astype(np.float32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Scalar float
    input4 = np.array(25.0, dtype=np.float32)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 1D complex tensor
    input5 = np.array([1+1j, 4+2j, 9+3j], dtype=np.complex64)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: 2D complex tensor
    input6 = np.array([[1+0j, 0+1j], [0-1j, 1-0j]], dtype=np.complex128)
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: 4D float tensor
    input7 = np.random.rand(2, 2, 2, 2).astype(np.float32)
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.sqrt_"] = sqrt__inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.sqrt_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sqrt_'.")

check_valid('torch.sqrt_', generated_inputs['torch.sqrt_'], lib="torch")
