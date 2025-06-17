
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_tril_inputs():
    list_of_inputs = []

    # Example 1: Basic case with a square matrix and default diagonal
    input_tensor = torch.randn(3, 3).numpy()
    input_dict = {"input": input_tensor, "diagonal": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2: Rectangular matrix with a positive diagonal
    input_tensor = torch.randn(4, 6).numpy()
    input_dict = {"input": input_tensor, "diagonal": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3: Rectangular matrix with a negative diagonal
    input_tensor = torch.randn(6, 4).numpy()
    input_dict = {"input": input_tensor, "diagonal": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 4: Integer tensor
    input_tensor = torch.randint(0, 10, (5, 5)).numpy()
    input_dict = {"input": input_tensor, "diagonal": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 5: Float64 tensor
    input_tensor = torch.randn(2, 2, dtype=torch.float64).numpy()
    input_dict = {"input": input_tensor, "diagonal": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.tril"] = torch_tril_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.tril' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.tril'.")

check_valid('torch.tril', generated_inputs['torch.tril'], lib="torch")
