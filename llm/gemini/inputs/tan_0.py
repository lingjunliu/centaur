
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def tan_inputs():
    list_of_inputs = []

    # Input 1: 1D tensor with positive values
    input_tensor = torch.tensor([0.0, 0.5, 1.0]).numpy()
    out_tensor = torch.tensor([0.0, 0.0, 0.0]).numpy()

    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D tensor with negative values
    input_tensor = torch.tensor([-0.5, -1.0, -1.5]).numpy()
    out_tensor = torch.tensor([0.0, 0.0, 0.0]).numpy()
    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D tensor
    input_tensor = torch.tensor([[0.0, 0.2], [0.4, 0.6]]).numpy()
    out_tensor = torch.tensor([[0.0, 0.0], [0.0, 0.0]]).numpy()
    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D tensor
    input_tensor = torch.randn(2, 3, 4).numpy()
    out_tensor = torch.randn(2, 3, 4).numpy()
    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Tensor with zeros
    input_tensor = torch.zeros(3, 3).numpy()
    out_tensor = torch.zeros(3, 3).numpy()
    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs["torch.tan"] = tan_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.tan' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.tan'.")

check_valid('torch.tan', generated_inputs['torch.tan'], lib="torch", suffix=0)
