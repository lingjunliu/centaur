
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def logsumexp_inputs():
    list_of_inputs = []

    # Case 1: Basic 2D tensor, dim=1, keepdim=False
    input_tensor = torch.randn(3, 4).numpy()
    dim = 1
    keepdim = False
    input_dict = {"input": input_tensor, "dim": dim, "keepdim": keepdim, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: 3D tensor, dim=0, keepdim=True
    input_tensor = torch.randn(2, 3, 4).numpy()
    dim = 0
    keepdim = True
    input_dict = {"input": input_tensor, "dim": dim, "keepdim": keepdim, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: 2D tensor with negative values, dim=0, keepdim=False
    input_tensor = torch.randn(2, 5) * -1.0
    input_tensor = input_tensor.numpy()
    dim = 0
    keepdim = False
    input_dict = {"input": input_tensor, "dim": dim, "keepdim": keepdim, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: 1D tensor, dim=0, keepdim=False
    input_tensor = torch.randn(5).numpy()
    dim = 0
    keepdim = False
    input_dict = {"input": input_tensor, "dim": dim, "keepdim": keepdim, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: 4D tensor, dim=(1, 2), keepdim=True
    input_tensor = torch.randn(2, 3, 4, 5).numpy()
    dim = (1, 2)
    keepdim = True
    input_dict = {"input": input_tensor, "dim": dim, "keepdim": keepdim, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: Input with large values
    input_tensor = torch.randn(2, 3) * 100
    input_tensor = input_tensor.numpy()
    dim = 1
    keepdim = False
    input_dict = {"input": input_tensor, "dim": dim, "keepdim": keepdim, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.logsumexp_3"] = logsumexp_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.logsumexp_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.logsumexp_3'.")

check_valid('torch.logsumexp', generated_inputs['torch.logsumexp_3'], lib="torch")
