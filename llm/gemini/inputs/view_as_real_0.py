
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def view_as_real_inputs():
    list_of_inputs = []

    # Input 1: 1D complex float tensor
    input1 = torch.randn(5, dtype=torch.complex64).numpy()
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D complex double tensor
    input2 = torch.randn(3, 4, dtype=torch.complex128).numpy()
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D complex float tensor
    input3 = torch.randn(2, 3, 5, dtype=torch.complex64).numpy()
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 4D complex double tensor
    input4 = torch.randn(1, 2, 3, 4, dtype=torch.complex128).numpy()
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: Empty complex tensor
    input5 = torch.empty(0, dtype=torch.complex64).numpy()
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: complex float tensor with negative values
    input6 = (torch.randn(2, 2, dtype=torch.complex64) - 1).numpy()
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    return list_of_inputs

generated_inputs["torch.view_as_real"] = view_as_real_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.view_as_real' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.view_as_real'.")

check_valid('torch.view_as_real', generated_inputs['torch.view_as_real'], lib="torch")
