
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def prod_inputs():
    list_of_inputs = []

    # Test case 1: Basic case with dim
    input_tensor = torch.randn(4, 2).numpy()
    dim = 1
    keepdim = False
    dtype = None
    input_dict = {"input": input_tensor, "dim": dim, "keepdim": keepdim, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 2: keepdim = True
    input_tensor = torch.randn(2, 3, 4).numpy()
    dim = 0
    keepdim = True
    dtype = None
    input_dict = {"input": input_tensor, "dim": dim, "keepdim": keepdim, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 3: Different dtype
    input_tensor = torch.randint(0, 10, (3, 3)).numpy()
    dim = 1
    keepdim = False
    dtype = torch.float32
    input_dict = {"input": input_tensor, "dim": dim, "keepdim": keepdim, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 4: Negative dim
    input_tensor = torch.randn(5, 5).numpy()
    dim = -1
    keepdim = False
    dtype = None
    input_dict = {"input": input_tensor, "dim": dim, "keepdim": keepdim, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 5: 1D tensor
    input_tensor = torch.arange(1, 6).numpy()
    dim = 0
    keepdim = False
    dtype = None
    input_dict = {"input": input_tensor, "dim": dim, "keepdim": keepdim, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 6: Complex tensor
    input_tensor = torch.complex(torch.randn(2, 2), torch.randn(2, 2)).numpy()
    dim = 1
    keepdim = False
    dtype = None
    input_dict = {"input": input_tensor, "dim": dim, "keepdim": keepdim, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 7: Empty tensor
    input_tensor = torch.empty(0).numpy()
    dim = 0
    keepdim = False
    dtype = None
    input_dict = {"input": input_tensor, "dim": dim, "keepdim": keepdim, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 8: Large tensor
    input_tensor = torch.randn(10, 10, 10).numpy()
    dim = 2
    keepdim = False
    dtype = None
    input_dict = {"input": input_tensor, "dim": dim, "keepdim": keepdim, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.prod_2"] = prod_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.prod_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.prod_2'.")

check_valid('torch.prod', generated_inputs['torch.prod_2'], lib="torch")
