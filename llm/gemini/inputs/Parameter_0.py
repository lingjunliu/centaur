
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def parameter_inputs():
    list_of_inputs = []

    # Input 1: Float tensor, requires_grad=True
    data = torch.randn(3, 4).numpy()
    input_dict = {
        "data": data,
        "requires_grad": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Int tensor, requires_grad=False
    data = torch.randint(0, 10, (2, 2), dtype=torch.int32).numpy()
    input_dict = {
        "data": data,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Complex tensor, requires_grad=True
    data = torch.randn(2, 3, dtype=torch.complex64).numpy()
    input_dict = {
        "data": data,
        "requires_grad": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative float tensor, requires_grad=False
    data = torch.randn(5).numpy() * -1
    input_dict = {
        "data": data,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multi-dimensional float tensor, requires_grad=True
    data = torch.randn(1, 2, 3, 4, 5).numpy()
    input_dict = {
        "data": data,
        "requires_grad": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.Parameter"] = parameter_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.Parameter' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Parameter'.")

check_valid('torch.nn.Parameter', generated_inputs['torch.nn.Parameter'], lib="torch")
