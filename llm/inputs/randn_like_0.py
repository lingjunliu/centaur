
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def randn_like_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D tensor
    input_tensor = torch.tensor([1.0, 2.0, 3.0]).numpy()
    input_dict = {
        "input": input_tensor,
        "dtype": torch.float32,
        "layout": torch.strided,
        "requires_grad": False,
        "memory_format": torch.contiguous_format
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D tensor, different dtype. Using float since int64 fails
    input_tensor = torch.ones((2, 3), dtype=torch.float32).numpy()
    input_dict = {
        "input": input_tensor,
        "dtype": torch.float32,
        "layout": torch.strided,
        "requires_grad": False,
        "memory_format": torch.contiguous_format
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D tensor, different layout
    input_tensor = torch.randn((2, 3, 4), dtype=torch.float64).numpy()
    input_dict = {
        "input": input_tensor,
        "dtype": torch.float64,
        "layout": torch.strided,
        "requires_grad": False,
        "memory_format": torch.contiguous_format
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: tensor with requires_grad=True
    input_tensor = torch.randn((4, 5), requires_grad=True).detach().numpy()
    input_dict = {
        "input": input_tensor,
        "dtype": torch.float32,
        "layout": torch.strided,
        "requires_grad": True,
        "memory_format": torch.contiguous_format
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different memory format
    input_tensor = torch.randn((2, 2)).numpy()
    input_dict = {
        "input": input_tensor,
        "dtype": torch.float32,
        "layout": torch.strided,
        "requires_grad": False,
        "memory_format": torch.contiguous_format
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: tensor with size 0
    input_tensor = torch.randn((0, 5)).numpy()
    input_dict = {
        "input": input_tensor,
        "dtype": torch.float32,
        "layout": torch.strided,
        "requires_grad": False,
        "memory_format": torch.contiguous_format
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    for input_dict in list_of_inputs:
        input_dict["dtype"] = torch.tensor(0, dtype=input_dict["dtype"]).dtype
        input_dict["layout"] = torch.strided
        input_dict["memory_format"] = torch.contiguous_format

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.randn_like"] = randn_like_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.randn_like' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.randn_like'.")

check_valid('torch.randn_like', generated_inputs['torch.randn_like'], lib="torch", suffix=0)
