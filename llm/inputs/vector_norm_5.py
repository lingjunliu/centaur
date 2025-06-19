
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def vector_norm_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor, default norm
    input1 = torch.randn(3, 4).numpy()
    input_dict1 = {"input": input1, "ord": 2, "dim": None, "keepdim": False, "dtype": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Float tensor, L1 norm, dim=0
    input2 = torch.randn(2, 3).numpy()
    input_dict2 = {"input": input2, "ord": 1, "dim": (0,), "keepdim": True, "dtype": torch.float32, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Complex tensor, L2 norm, dim=(0,1)
    input3 = torch.randn(2, 3, dtype=torch.complex64).numpy()
    input_dict3 = {"input": input3, "ord": 2, "dim": (0, 1), "keepdim": False, "dtype": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Float tensor, inf norm, dim=1
    input4 = torch.randn(4, 5).numpy()
    input_dict4 = {"input": input4, "ord": float('inf'), "dim": (1,), "keepdim": True, "dtype": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Float tensor, -inf norm, dim=None
    input5 = torch.randn(2, 2).numpy()
    input_dict5 = {"input": input5, "ord": float('-inf'), "dim": None, "keepdim": False, "dtype": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.linalg.vector_norm_5"] = vector_norm_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.linalg.vector_norm_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.vector_norm_5'.")

check_valid('torch.linalg.vector_norm', generated_inputs['torch.linalg.vector_norm_5'], lib="torch")
