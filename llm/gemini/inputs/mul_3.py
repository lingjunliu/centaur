
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_mul_inputs():
    list_of_inputs = []

    # Case 1: Basic float tensor and integer
    input_tensor = torch.randn(3, 4).numpy()
    other_int = 2
    input_dict = {"input": input_tensor, "other": other_int, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Negative values
    input_tensor = torch.randn(2, 2).numpy()
    other_int = -3
    input_dict = {"input": input_tensor, "other": other_int, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Integer tensor
    input_tensor = torch.randint(-5, 5, (5, 5)).numpy()
    other_int = 4
    input_dict = {"input": input_tensor, "other": other_int, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: 1D tensor
    input_tensor = torch.randn(10).numpy()
    other_int = 5
    input_dict = {"input": input_tensor, "other": other_int, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: 3D tensor
    input_tensor = torch.randn(2, 3, 4).numpy()
    other_int = -2
    input_dict = {"input": input_tensor, "other": other_int, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: zero value
    input_tensor = torch.randn(3, 4).numpy()
    other_int = 0
    input_dict = {"input": input_tensor, "other": other_int, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.mul_3"] = torch_mul_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.mul_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.mul_3'.")

check_valid('torch.mul', generated_inputs['torch.mul_3'], lib="torch")
