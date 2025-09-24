
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def sgn_inputs():
    list_of_inputs = []

    # Input 1: Complex tensor, 1D
    input_tensor = torch.tensor([3+4j, 7-24j, 0, 1+2j])
    out_tensor = torch.zeros_like(input_tensor)

    input_dict = {
        "input": input_tensor.numpy(),
        "out": out_tensor.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Real tensor, 1D, with negative values
    input_tensor = torch.tensor([-1.0, 2.0, -3.0, 0.0, 5.0])
    out_tensor = torch.zeros_like(input_tensor)
    input_dict = {
        "input": input_tensor.numpy(),
        "out": out_tensor.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Complex tensor, 2D
    input_tensor = torch.tensor([[1+1j, 2-2j], [3+0j, 0-4j]])
    out_tensor = torch.zeros_like(input_tensor)

    input_dict = {
        "input": input_tensor.numpy(),
        "out": out_tensor.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Real tensor, 2D, all zeros
    input_tensor = torch.zeros((2, 2))
    out_tensor = torch.zeros_like(input_tensor)

    input_dict = {
        "input": input_tensor.numpy(),
        "out": out_tensor.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Real tensor, 3D
    input_tensor = torch.randn(2, 3, 4)
    out_tensor = torch.zeros_like(input_tensor)
    input_dict = {
        "input": input_tensor.numpy(),
        "out": out_tensor.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.sgn"] = sgn_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.sgn' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sgn'.")

check_valid('torch.sgn', generated_inputs['torch.sgn'], lib="torch", suffix=0)
