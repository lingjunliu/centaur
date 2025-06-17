
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_pow_inputs():
    list_of_inputs = []

    # Case 1: Basic float exponent, positive input
    input_tensor = torch.randn(2, 3).numpy()
    exponent = 2.0
    input_dict = {"input": input_tensor, "exponent": exponent, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Float exponent, negative input (expecting complex result if exponent isn't an integer)
    input_tensor = torch.randn(2, 3).numpy() - 1
    exponent = 3.0
    input_dict = {"input": input_tensor, "exponent": exponent, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Integer exponent, negative input
    input_tensor = torch.randn(2, 3).numpy() - 1
    exponent = 3
    input_dict = {"input": input_tensor, "exponent": exponent, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Different shaped input
    input_tensor = torch.randn(5).numpy()
    exponent = 2.0
    input_dict = {"input": input_tensor, "exponent": exponent, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 5: Different shaped input
    input_tensor = torch.randn(2, 2, 2).numpy()
    exponent = 2.0
    input_dict = {"input": input_tensor, "exponent": exponent, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.pow_1"] = torch_pow_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.pow_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.pow_1'.")

check_valid('torch.pow', generated_inputs['torch.pow_1'], lib="torch")
