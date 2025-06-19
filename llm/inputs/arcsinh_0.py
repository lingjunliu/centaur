
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def arcsinh_inputs():
    list_of_inputs = []

    # Input 1: 1D tensor
    input_tensor = torch.tensor([0.0, 1.0, 2.0, -1.0, -2.0]).numpy()
    out_tensor = torch.tensor([]).numpy()

    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D tensor
    input_tensor = torch.tensor([[0.5, 1.5], [-0.5, -1.5]]).numpy()
    out_tensor = torch.tensor([]).numpy()
    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D tensor
    input_tensor = torch.randn(2, 3, 4).numpy()
    out_tensor = torch.tensor([]).numpy()
    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Tensor with zeros
    input_tensor = torch.zeros(5).numpy()
    out_tensor = torch.tensor([]).numpy()
    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Large values
    input_tensor = torch.tensor([100.0, -100.0, 1000.0, -1000.0]).numpy()
    out_tensor = torch.tensor([]).numpy()
    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Tensor with mixed values
    input_tensor = torch.tensor([-0.7, 0.3, 1.2, -2.5, 0.0]).numpy()
    out_tensor = torch.tensor([]).numpy()
    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.arcsinh"] = arcsinh_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.arcsinh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.arcsinh'.")

check_valid('torch.arcsinh', generated_inputs['torch.arcsinh'], lib="torch", suffix=0)
