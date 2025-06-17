
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def adjoint_inputs():
    list_of_inputs = []

    # Example 1: Complex tensor
    x = np.arange(4, dtype=np.float32)
    A = np.complex64(x + 1j * x).reshape(2, 2)
    input_dict = {"input": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2: Real tensor
    x = np.arange(9, dtype=np.float32).reshape(3, 3)
    input_dict = {"input": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3: 3D complex tensor
    x = np.arange(8, dtype=np.float32)
    A = np.complex64(x + 1j * x).reshape(2, 2, 2)
    input_dict = {"input": A}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Example 4: Float64 real tensor
    x = np.arange(4, dtype=np.float64).reshape(2,2)
    input_dict = {"input": x}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs["torch.adjoint"] = adjoint_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.adjoint' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.adjoint'.")

check_valid('torch.adjoint', generated_inputs['torch.adjoint'], lib="torch")
