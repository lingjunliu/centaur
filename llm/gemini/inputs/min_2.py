
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_min_inputs():
    list_of_inputs = []

    # Case 1: Basic case with dim=0, keepdim=False
    input_tensor = torch.randn(3, 4).numpy()
    dim = 0
    keepdim = False
    input_dict = {"input": input_tensor, "dim": dim, "keepdim": keepdim, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: dim=1, keepdim=True
    input_tensor = torch.randn(2, 5).numpy()
    dim = 1
    keepdim = True
    input_dict = {"input": input_tensor, "dim": dim, "keepdim": keepdim, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: 3D tensor, dim=2, keepdim=False
    input_tensor = torch.randn(2, 3, 4).numpy()
    dim = 2
    keepdim = False
    input_dict = {"input": input_tensor, "dim": dim, "keepdim": keepdim, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Integer tensor, dim=0, keepdim=True
    input_tensor = torch.randint(0, 10, (4, 3)).numpy()
    dim = 0
    keepdim = True
    input_dict = {"input": input_tensor, "dim": dim, "keepdim": keepdim, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 5: Tensor with negative values, dim=1, keepdim=False
    input_tensor = torch.randint(-5, 5, (3, 5)).float().numpy()
    dim = 1
    keepdim = False
    input_dict = {"input": input_tensor, "dim": dim, "keepdim": keepdim, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.min_2"] = torch_min_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.min_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.min_2'.")

check_valid('torch.min', generated_inputs['torch.min_2'], lib="torch")
