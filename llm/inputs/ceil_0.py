
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_ceil_inputs():
    list_of_inputs = []

    # Example 1: 1D tensor with positive and negative floats
    input_tensor = torch.randn(5).numpy()
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2: 2D tensor with positive and negative floats
    input_tensor = torch.randn(2, 3).numpy()
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3: 3D tensor with positive and negative floats
    input_tensor = torch.randn(2, 3, 4).numpy()
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 4: 1D tensor with integers
    input_tensor = torch.randint(-5, 5, (5,)).numpy()
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 5: Scalar float
    input_tensor = np.random.randn()
    input_dict = {"input": np.array(input_tensor)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.ceil"] = torch_ceil_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.ceil' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.ceil'.")

check_valid('torch.ceil', generated_inputs['torch.ceil'], lib="torch")
