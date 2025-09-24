
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def cumsum_inputs():
    list_of_inputs = []

    # Test case 1: 1D integer tensor
    input1 = torch.randint(1, 20, (10,)).numpy()
    dim1 = 0
    input_dict1 = {"input": input1, "dim": dim1, "dtype": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Test case 2: 2D float tensor with negative values
    input2 = torch.randn(5, 5).numpy()
    dim2 = 1
    input_dict2 = {"input": input2, "dim": dim2, "dtype": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Test case 3: 3D complex tensor
    input3 = (torch.randn(2, 3, 4) + 1j * torch.randn(2, 3, 4)).numpy()
    dim3 = 0
    input_dict3 = {"input": input3, "dim": dim3, "dtype": torch.complex64, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Test case 4: 2D integer tensor along dimension 0
    input4 = torch.randint(-10, 10, (3, 4)).numpy()
    dim4 = 0
    input_dict4 = {"input": input4, "dim": dim4, "dtype": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Test case 5: 4D float tensor
    input5 = torch.randn(2, 2, 2, 2).numpy()
    dim5 = 2
    input_dict5 = {"input": input5, "dim": dim5, "dtype": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.cumsum"] = cumsum_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.cumsum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.cumsum'.")

check_valid('torch.cumsum', generated_inputs['torch.cumsum'], lib="torch")
