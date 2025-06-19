
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def frexp_inputs():
    list_of_inputs = []

    # Input 1: 1D tensor
    input_tensor = torch.arange(1.0, 6.0).numpy()
    out_tuple = (torch.randn(5).numpy(), torch.zeros(5, dtype=torch.int32).numpy())
    input_dict = {"input": input_tensor, "out": out_tuple}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D tensor
    input_tensor = torch.randn(2, 3).numpy()
    out_tuple = (torch.randn(2, 3).numpy(), torch.zeros((2, 3), dtype=torch.int32).numpy())
    input_dict = {"input": input_tensor, "out": out_tuple}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D tensor
    input_tensor = torch.randn(2, 2, 2).numpy()
    out_tuple = (torch.randn(2, 2, 2).numpy(), torch.zeros((2, 2, 2), dtype=torch.int32).numpy())
    input_dict = {"input": input_tensor, "out": out_tuple}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Tensor with negative values
    input_tensor = torch.linspace(-5.0, 5.0, 10).numpy()
    out_tuple = (torch.randn(10).numpy(), torch.zeros(10, dtype=torch.int32).numpy())
    input_dict = {"input": input_tensor, "out": out_tuple}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Tensor with zeros
    input_tensor = torch.tensor([0.0, 1.0, 2.0, 0.0, -1.0]).numpy()
    out_tuple = (torch.randn(5).numpy(), torch.zeros(5, dtype=torch.int32).numpy())
    input_dict = {"input": input_tensor, "out": out_tuple}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.frexp"] = frexp_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.frexp' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.frexp'.")

check_valid('torch.frexp', generated_inputs['torch.frexp'], lib="torch", suffix=0)
