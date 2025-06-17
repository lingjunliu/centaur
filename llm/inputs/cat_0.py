
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_cat_inputs():
    list_of_inputs = []

    # Case 1: Basic concatenation along dimension 0
    x = torch.randn(2, 3).numpy()
    tensors = [x, x, x]
    dim = 0
    input_dict = {"tensors": tensors, "dim": dim, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Concatenation along dimension 1
    x = torch.randn(2, 3).numpy()
    tensors = [x, x, x]
    dim = 1
    input_dict = {"tensors": tensors, "dim": dim, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Different tensor sizes (except in the concatenating dimension)
    x = torch.randn(2, 3).numpy()
    y = torch.randn(2, 5).numpy()
    tensors = [x, y]
    dim = 1
    input_dict = {"tensors": tensors, "dim": dim, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: 3D tensors
    x = torch.randn(2, 3, 4).numpy()
    y = torch.randn(2, 3, 4).numpy()
    tensors = [x, y]
    dim = 0
    input_dict = {"tensors": tensors, "dim": dim, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Negative values and dim
    x = torch.randn(2, 3).numpy()
    tensors = [x, x]
    dim = -1
    input_dict = {"tensors": tensors, "dim": dim, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.cat"] = torch_cat_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.cat' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.cat'.")

check_valid('torch.cat', generated_inputs['torch.cat'], lib="torch")
