
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_norm_inputs():
    list_of_inputs = []

    # Input 1: Float tensor, default p, dim=None
    input1 = torch.randn(3, 4).numpy()
    input_dict1 = {
        "input": input1,
        "p": 'fro',
        "dim": None,
        "keepdim": False,
        "out": None,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 3: Complex tensor, p=2, dim=(0,1)
    input3 = torch.complex(torch.randn(2, 2), torch.randn(2, 2)).numpy()
    input_dict3 = {
        "input": input3,
        "p": 2.0,
        "dim": (0, 1),
        "keepdim": False,
        "out": None,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Float tensor, p=inf, dim=1
    input4 = torch.randn(4, 5).numpy()
    input_dict4 = {
        "input": input4,
        "p": float('inf'),
        "dim": [1],
        "keepdim": False,
        "out": None,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Float tensor, p=-inf, dim=None, keepdim=True
    input5 = torch.randn(2, 3, 4).numpy()
    input_dict5 = {
        "input": input5,
        "p": float('-inf'),
        "dim": None,
        "keepdim": True,
        "out": None,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Input 6: Float tensor, p=2, dim=(0,2)  - REMOVED since matrix_norm needs a 2-tuple, vector_norm flattens.

    # Input 7: Float tensor, p='fro', dim=(0,1,2) - should be valid though deprecated - REMOVED due to RuntimeError: linalg.matrix_norm: dim must be a 2-tuple. Got 0 1 2.

    # Input 8: Float tensor, p=1, dim = None
    input8 = torch.randn(2,3).numpy()
    input_dict8 = {
        "input": input8,
        "p": 1.0,
        "dim": None,
        "keepdim": False,
        "out": None,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Float tensor, p='fro', dim=(0,1)
    input9 = torch.randn(3,4).numpy()
    input_dict9 = {
        "input": input9,
        "p": 'fro',
        "dim": (0,1),
        "keepdim": False,
        "out": None,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))
    

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.norm_12"] = torch_norm_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.norm_12' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.norm_12'.")

check_valid('torch.norm', generated_inputs['torch.norm_12'], lib="torch")
