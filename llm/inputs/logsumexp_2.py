
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def logsumexp_inputs():
    list_of_inputs = []

    # Case 1: Basic case with a 2D tensor
    input_tensor = torch.randn(3, 3).numpy()
    dim = (1,)
    keepdim = False
    input_dict = {"input": input_tensor, "dim": dim, "keepdim": keepdim, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: 3D tensor, keepdim=True
    input_tensor = torch.randn(2, 3, 4).numpy()
    dim = (0,)
    keepdim = True
    input_dict = {"input": input_tensor, "dim": dim, "keepdim": keepdim, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Negative values and multiple dimensions to reduce
    input_tensor = torch.randn(2, 3, 4).numpy() * -1
    dim = (0, 1)
    keepdim = False
    input_dict = {"input": input_tensor, "dim": dim, "keepdim": keepdim, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 4: Integer tensor
    input_tensor = torch.randint(-5, 5, (4, 4)).numpy()
    dim = (0,)
    keepdim = True
    input_dict = {"input": input_tensor, "dim": dim, "keepdim": keepdim, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: 1D tensor
    input_tensor = torch.randn(5).numpy()
    dim = (0,)
    keepdim = False
    input_dict = {"input": input_tensor, "dim": dim, "keepdim": keepdim, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: All dimensions
    input_tensor = torch.randn(2, 3, 4).numpy()
    dim = (0, 1, 2)
    keepdim = False
    input_dict = {"input": input_tensor, "dim": dim, "keepdim": keepdim, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 7: Empty tensor
    input_tensor = torch.empty(0).numpy()
    dim = (0,)
    keepdim = False
    input_dict = {"input": input_tensor, "dim": dim, "keepdim": keepdim, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.logsumexp_2"] = logsumexp_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.logsumexp_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.logsumexp_2'.")

check_valid('torch.logsumexp', generated_inputs['torch.logsumexp_2'], lib="torch")
