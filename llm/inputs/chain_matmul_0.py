
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def chain_matmul_inputs():
    list_of_inputs = []

    # Input 1: Basic case with float tensors
    matrices = [torch.randn(2, 3),
                torch.randn(3, 4),
                torch.randn(4, 2)]
    input_dict = {"matrices": matrices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Two matrices
    matrices = [torch.randn(4, 5),
                torch.randn(5, 4)]
    input_dict = {"matrices": matrices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative values
    matrices = [torch.randn(2, 3) * -1,
                torch.randn(3, 4)]
    input_dict = {"matrices": matrices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different dtypes
    matrices = [torch.randn(2, 3, dtype=torch.float64),
                torch.randn(3, 4, dtype=torch.float64)]
    input_dict = {"matrices": matrices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.chain_matmul"] = chain_matmul_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.chain_matmul' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.chain_matmul'.")

check_valid('torch.chain_matmul', generated_inputs['torch.chain_matmul'], lib="torch")
