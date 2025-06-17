
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def lu_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor
    A = torch.randn(3, 2).numpy()
    input_dict = {"A": A, "pivot": True, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Batch of matrices
    A = torch.randn(2, 3, 3).numpy()
    input_dict = {"A": A, "pivot": True, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Complex tensor
    A = (torch.randn(2, 2) + 1j * torch.randn(2, 2)).numpy()
    input_dict = {"A": A, "pivot": True, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Double tensor
    A = torch.randn(4, 3, dtype=torch.float64).numpy()
    input_dict = {"A": A, "pivot": True, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Tensor with negative values
    A = torch.randn(3, 4) * -1.0
    A = A.numpy()
    input_dict = {"A": A, "pivot": True, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.linalg.lu"] = lu_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.linalg.lu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.lu'.")

check_valid('torch.linalg.lu', generated_inputs['torch.linalg.lu'], lib="torch")
