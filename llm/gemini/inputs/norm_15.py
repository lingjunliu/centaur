
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_norm_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor
    input_tensor = torch.randn(3, 4).numpy()
    input_dict = {
        "input": input_tensor,
        "p": 2.0,
        "dim": None,
        "keepdim": False,
        "out": None,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Int tensor with dim specified, cast to float
    input_tensor = torch.randint(-5, 5, (2, 5), dtype=torch.int32).float().numpy()
    input_dict = {
        "input": input_tensor,
        "p": 1.0,
        "dim": [0],
        "keepdim": True,
        "out": None,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Complex tensor
    input_tensor = torch.randn(2, 3, dtype=torch.complex64).numpy()
    input_dict = {
        "input": input_tensor,
        "p": 'fro',
        "dim": None,
        "keepdim": False,
        "out": None,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Tensor with inf norm
    input_tensor = torch.randn(4, 4).numpy()
    input_dict = {
        "input": input_tensor,
        "p": float('inf'),
        "dim": [1],
        "keepdim": True,
        "out": None,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Tensor with negative inf norm
    input_tensor = torch.randn(4, 4).numpy()
    input_dict = {
        "input": input_tensor,
        "p": float('-inf'),
        "dim": [1],
        "keepdim": True,
        "out": None,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D Tensor with multiple dims
    input_tensor = torch.randn(2, 3, 2).numpy()
    input_dict = {
        "input": input_tensor,
        "p": 2.0,
        "dim": [1,2],
        "keepdim": False,
        "out": None,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7:  Specify dtype
    input_tensor = torch.randn(2, 5).numpy()
    input_dict = {
        "input": input_tensor,
        "p": 1.0,
        "dim": [0],
        "keepdim": True,
        "out": None,
        "dtype": torch.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.norm_15"] = torch_norm_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.norm_15' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.norm_15'.")

check_valid('torch.norm', generated_inputs['torch.norm_15'], lib="torch")
