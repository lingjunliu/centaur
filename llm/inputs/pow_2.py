
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_pow_inputs():
    list_of_inputs = []

    # Case 1: float input, float exponent
    input_tensor = torch.randn(4).numpy()
    exponent = 2.0
    input_dict = {
        "input": input_tensor,
        "exponent": exponent
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: float input, tensor exponent
    input_tensor = torch.randn(4).numpy()
    exponent = torch.arange(1., 5.).numpy()
    input_dict = {
        "input": input_tensor,
        "exponent": exponent
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: int input, int exponent
    input_tensor = torch.randint(1, 5, (4,)).numpy()
    exponent = torch.randint(1, 5, (4,)).numpy()
    input_dict = {
        "input": input_tensor,
        "exponent": exponent
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 4: int input, float exponent
    input_tensor = torch.randint(1, 5, (4,)).numpy()
    exponent = 2.5
    input_dict = {
        "input": input_tensor,
        "exponent": exponent
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: float input, float exponent, different shape (broadcastable)
    input_tensor = torch.randn(2, 3).numpy()
    exponent = 2.0
    input_dict = {
        "input": input_tensor,
        "exponent": exponent
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.pow_2"] = torch_pow_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.pow_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.pow_2'.")

check_valid('torch.pow', generated_inputs['torch.pow_2'], lib="torch")
