
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_norm_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor
    input_1 = torch.randn(3, 4).numpy()
    input_dict_1 = {
        "input": input_1,
        "p": 2,
        "dim": None,
        "keepdim": False,
        "out": None,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Integer tensor with dim - needs float dtype
    input_2 = torch.randint(-5, 5, (2, 3, 4)).float().numpy()
    input_dict_2 = {
        "input": input_2,
        "p": 1,
        "dim": (0, 1),
        "keepdim": True,
        "out": None,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Complex tensor with 'fro'
    input_3 = torch.complex(torch.randn(2, 2), torch.randn(2, 2)).numpy()
    input_dict_3 = {
        "input": input_3,
        "p": 'fro',
        "dim": None,
        "keepdim": False,
        "out": None,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Tensor with inf norm
    input_4 = torch.randn(5).numpy()
    input_dict_4 = {
        "input": input_4,
        "p": float('inf'),
        "dim": None,
        "keepdim": False,
        "out": None,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Negative tensor with specified dtype
    input_5 = torch.randn(2, 3) - 2
    input_5 = input_5.numpy()
    input_dict_5 = {
        "input": input_5,
        "p": 2,
        "dim": 1,
        "keepdim": True,
        "out": None,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    return list_of_inputs

generated_inputs["torch.norm_3"] = torch_norm_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.norm_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.norm_3'.")

check_valid('torch.norm', generated_inputs['torch.norm_3'], lib="torch")
