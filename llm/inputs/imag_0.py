
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_imag_inputs():
    list_of_inputs = []

    # Input 1: 1D complex tensor
    x = torch.randn(4, dtype=torch.cfloat).numpy()
    input_dict = {"input": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D complex tensor
    x = torch.randn(2, 3, dtype=torch.complex64).numpy()
    input_dict = {"input": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D complex tensor
    x = torch.randn(2, 3, 4, dtype=torch.complex128).numpy()
    input_dict = {"input": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Scalar complex tensor
    x = torch.tensor(complex(1.0, -2.0)).numpy()
    input_dict = {"input": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 4D complex tensor with negative values
    x = (torch.randn(2, 2, 2, 2, dtype=torch.cfloat) * -1).numpy()
    input_dict = {"input": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Complex tensor with only imaginary part
    x = (1j * torch.randn(3, dtype=torch.cfloat)).numpy()
    input_dict = {"input": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.imag"] = torch_imag_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.imag' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.imag'.")

check_valid('torch.imag', generated_inputs['torch.imag'], lib="torch")
