
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def leaky_relu_inputs():
    list_of_inputs = []

    # Test case 1: Basic 2D float tensor with positive negative_slope
    input1 = torch.randn(2, 3).numpy()
    input_dict1 = {
        "input": input1,
        "negative_slope": 0.01,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Test case 2: 1D integer tensor with negative values and negative negative_slope
    input2 = torch.randint(-5, 5, (5,)).numpy().astype(np.float32)
    input_dict2 = {
        "input": input2,
        "negative_slope": -0.1,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Test case 3: 3D float tensor with zero negative_slope
    input3 = torch.randn(1, 4, 4).numpy()
    input_dict3 = {
        "input": input3,
        "negative_slope": 0.0,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Test case 4: 4D float tensor
    input4 = torch.randn(2, 2, 2, 2).numpy()
    input_dict4 = {
        "input": input4,
        "negative_slope": 0.2,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Test case 5: Scalar input
    input5 = torch.randn(1).item()
    input_dict5 = {
        "input": np.array(input5),
        "negative_slope": 0.3,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.leaky_relu"] = leaky_relu_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.leaky_relu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.leaky_relu'.")

check_valid('torch.nn.functional.leaky_relu', generated_inputs['torch.nn.functional.leaky_relu'], lib="torch")
