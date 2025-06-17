
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def var_mean_inputs():
    list_of_inputs = []

    # Case 1: Basic float tensor, dim=None
    input = torch.randn(3, 4, 5).numpy()
    input_dict = {
        "input": input,
        "dim": None,
        "unbiased": True,
        "keepdim": False,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Int tensor, dim=0, keepdim=True
    input = torch.randint(0, 10, (2, 3, 4)).numpy()
    input_dict = {
        "input": input,
        "dim": (0,),
        "unbiased": False,
        "keepdim": True,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Float tensor, dim=(1, 2), unbiased=False
    input = torch.randn(2, 3, 4, 5).numpy()
    input_dict = {
        "input": input,
        "dim": (1, 2),
        "unbiased": False,
        "keepdim": False,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: 1D Float Tensor with negative values
    input = torch.randn(10).numpy() - 2  # Shift to have negative values
    input_dict = {
        "input": input,
        "dim": (0,),
        "unbiased": True,
        "keepdim": False,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 5: Complex Tensor
    input = (torch.randn(3, 3) + 1j * torch.randn(3, 3)).numpy()
    input_dict = {
        "input": input,
        "dim": None,
        "unbiased": False,
        "keepdim": False,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 6: Unbiased True, No dim - remove dim and other keywords
    input = torch.randn(5, 5).numpy()
    input_dict = {
        "input": input,
        "unbiased": True,
        "dim": None,
        "keepdim": False,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.var_mean_3"] = var_mean_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.var_mean_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.var_mean_3'.")

check_valid('torch.var_mean', generated_inputs['torch.var_mean_3'], lib="torch")
