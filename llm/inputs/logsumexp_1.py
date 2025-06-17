
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def logsumexp_inputs():
    list_of_inputs = []

    # Test case 1: Basic 2D tensor
    input1 = torch.randn(3, 4).numpy()
    dim1 = 1
    keepdim1 = False
    input_dict1 = {"input": input1, "dim": dim1, "keepdim": keepdim1, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Test case 2: 3D tensor with keepdim=True
    input2 = torch.randn(2, 3, 5).numpy()
    dim2 = 0
    keepdim2 = True
    input_dict2 = {"input": input2, "dim": dim2, "keepdim": keepdim2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Test case 3: Negative values and different dimension
    input3 = torch.randn(4, 4) * -1.0
    input3 = input3.numpy()
    dim3 = 1
    keepdim3 = False
    input_dict3 = {"input": input3, "dim": dim3, "keepdim": keepdim3, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Test case 4: Integer tensor
    input4 = torch.randint(0, 10, (2, 5)).float().numpy()
    dim4 = 0
    keepdim4 = True
    input_dict4 = {"input": input4, "dim": dim4, "keepdim": keepdim4, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Test case 5: 1D tensor
    input5 = torch.randn(5).numpy()
    dim5 = 0
    keepdim5 = False
    input_dict5 = {"input": input5, "dim": dim5, "keepdim": keepdim5, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    

    return list_of_inputs

generated_inputs["torch.logsumexp_1"] = logsumexp_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.logsumexp_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.logsumexp_1'.")

check_valid('torch.logsumexp', generated_inputs['torch.logsumexp_1'], lib="torch")
