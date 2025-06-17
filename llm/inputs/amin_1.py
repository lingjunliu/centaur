
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_amin_inputs():
    list_of_inputs = []

    # Case 1: Basic 2D tensor, dim=1, keepdim=False
    input = torch.randn(4, 4).numpy()
    dim = 1
    keepdim = False
    input_dict = {"input": input, "dim": dim, "keepdim": keepdim, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: 3D tensor, dim=(0, 2), keepdim=True
    input = torch.randn(3, 5, 2).numpy()
    dim = (0, 2)
    keepdim = True
    input_dict = {"input": input, "dim": dim, "keepdim": keepdim, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: 1D tensor, dim=0, keepdim=False
    input = torch.randint(-5, 5, (10,)).float().numpy()
    dim = 0
    keepdim = False
    input_dict = {"input": input, "dim": dim, "keepdim": keepdim, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Tensor with negative values, dim=0, keepdim=True
    input = torch.randint(-10, -1, (2, 3, 4)).float().numpy()
    dim = 0
    keepdim = True
    input_dict = {"input": input, "dim": dim, "keepdim": keepdim, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: 4D tensor, dim=2, keepdim=False
    input = torch.randn(2, 3, 4, 5).numpy()
    dim = 2
    keepdim = False
    input_dict = {"input": input, "dim": dim, "keepdim": keepdim, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 6: Integer tensor
    input = torch.randint(0, 10, (3, 3)).numpy()
    dim = 1
    keepdim = True
    input_dict = {"input": input, "dim": dim, "keepdim": keepdim, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.amin_1"] = torch_amin_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.amin_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.amin_1'.")

check_valid('torch.amin', generated_inputs['torch.amin_1'], lib="torch")
