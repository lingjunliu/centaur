
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def conj_physical_inputs():
    list_of_inputs = []

    # Input 1: Complex Float 64 Tensor
    input1 = torch.randn(2, 3, dtype=torch.complex64).numpy()
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 3: Multi-dimensional complex Tensor
    input3 = torch.randn(2, 2, 2, 2, dtype=torch.complex128).numpy()
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 1D complex Tensor
    input4 = torch.randn(5, dtype=torch.complex64).numpy()
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Complex Tensor with zero values
    input5 = torch.zeros(2, 3, dtype=torch.complex128).numpy()
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Complex Tensor with a mix of positive and negative real/imaginary parts
    input6 = (torch.randn(2, 3) + 1j * torch.randn(2, 3)).numpy()
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    return list_of_inputs

generated_inputs["torch.conj_physical_"] = conj_physical_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.conj_physical_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.conj_physical_'.")

check_valid('torch.conj_physical_', generated_inputs['torch.conj_physical_'], lib="torch")
