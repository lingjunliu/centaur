
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_add_inputs():
    list_of_inputs = []

    # Case 1: Basic addition of two tensors
    input1 = torch.randn(4).numpy()
    other1 = torch.randn(4).numpy()
    alpha1 = 1.0
    out1 = None

    input_dict1 = {
        "input": input1,
        "other": other1,
        "alpha": alpha1,
        "out": out1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Case 2: Addition with a scalar
    input2 = torch.randn(4).numpy()
    other2 = 2.0
    alpha2 = 1.0
    out2 = None
    input_dict2 = {
        "input": input2,
        "other": other2,
        "alpha": alpha2,
        "out": out2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Case 3: Broadcasting example
    input3 = torch.randn(4, 1).numpy()
    other3 = torch.randn(4).numpy()
    alpha3 = 1.0
    out3 = None
    input_dict3 = {
        "input": input3,
        "other": other3,
        "alpha": alpha3,
        "out": out3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Case 4: Negative values and alpha
    input4 = torch.randn(2, 2).numpy()
    other4 = torch.randn(2, 2).numpy()
    alpha4 = -0.5
    out4 = None
    input_dict4 = {
        "input": input4,
        "other": other4,
        "alpha": alpha4,
        "out": out4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Case 5: Integer tensors - Changed alpha to int
    input5 = torch.randint(0, 10, (3, 3)).numpy()
    other5 = torch.randint(0, 10, (3, 3)).numpy()
    alpha5 = 1
    out5 = None
    input_dict5 = {
        "input": input5,
        "other": other5,
        "alpha": alpha5,
        "out": out5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

generated_inputs["torch.add_1"] = torch_add_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.add_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.add_1'.")

check_valid('torch.add', generated_inputs['torch.add_1'], lib="torch")
