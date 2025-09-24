
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_sum_inputs():
    list_of_inputs = []

    # Case 1: Basic float tensor
    input_tensor = torch.randn(2, 3).numpy()
    input_dict = {"input": input_tensor, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Int tensor with specified dtype
    input_tensor = torch.randint(0, 10, (3, 4)).numpy()
    input_dict = {"input": input_tensor, "dtype": torch.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Tensor with negative values
    input_tensor = torch.randint(-10, 10, (5,)).float().numpy()
    input_dict = {"input": input_tensor, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: 3D tensor
    input_tensor = torch.randn(2, 3, 4).numpy()
    input_dict = {"input": input_tensor, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Empty tensor
    input_tensor = torch.empty(0).numpy()
    input_dict = {"input": input_tensor, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.sum_1"] = torch_sum_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.sum_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sum_1'.")

check_valid('torch.sum', generated_inputs['torch.sum_1'], lib="torch")
