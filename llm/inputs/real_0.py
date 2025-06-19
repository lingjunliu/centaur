
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_real_inputs():
    list_of_inputs = []

    # Input 1: 1D complex tensor
    input1 = torch.randn(4, dtype=torch.cfloat).numpy()
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D complex tensor
    input2 = torch.randn(2, 3, dtype=torch.cdouble).numpy()
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D complex tensor
    input3 = torch.randn(2, 2, 2, dtype=torch.complex64).numpy()
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 4D complex tensor
    input4 = torch.randn(1, 3, 5, 5, dtype=torch.complex128).numpy()
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 1D complex tensor with negative real and imaginary parts
    input5 = (torch.randn(4, dtype=torch.cfloat) * -1).numpy()
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Input 6: Empty complex tensor
    input6 = torch.empty(0, dtype=torch.cfloat).numpy()
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    return list_of_inputs

generated_inputs["torch.real"] = torch_real_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.real' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.real'.")

check_valid('torch.real', generated_inputs['torch.real'], lib="torch")
