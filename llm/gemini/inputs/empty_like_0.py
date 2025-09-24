
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def empty_like_inputs():
    list_of_inputs = []

    # Input 1: Float tensor, default options
    input1 = torch.randn(2, 3, 4).numpy()
    input_dict1 = {
        "input": input1,
        "dtype": None,
        "layout": None,
        "requires_grad": None,
        "memory_format": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Int tensor, specifying dtype
    input2 = torch.randint(0, 10, (5, 5)).numpy()
    input_dict2 = {
        "input": input2,
        "dtype": torch.float32,
        "layout": None,
        "requires_grad": None,
        "memory_format": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Bool tensor, specifying layout
    input3 = torch.randint(0, 2, (10,)).bool().numpy()
    input_dict3 = {
        "input": input3,
        "dtype": None,
        "layout": torch.strided,
        "requires_grad": None,
        "memory_format": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Complex tensor, specifying requires_grad
    input4 = torch.randn(2, 2, dtype=torch.complex64).numpy()
    input_dict4 = {
        "input": input4,
        "dtype": None,
        "layout": None,
        "requires_grad": True,
        "memory_format": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Long tensor, specifying memory_format
    input5 = torch.randint(0, 100, (3, 1, 5, 5), dtype=torch.int64).numpy()
    input_dict5 = {
        "input": input5,
        "dtype": None,
        "layout": None,
        "requires_grad": None,
        "memory_format": torch.contiguous_format
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Input 6: Float16 tensor
    input6 = torch.randn(4, 4, dtype=torch.float16).numpy()
    input_dict6 = {
        "input": input6,
        "dtype": None,
        "layout": None,
        "requires_grad": False,
        "memory_format": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Empty tensor
    input7 = torch.empty(0).numpy()
    input_dict7 = {
        "input": input7,
        "dtype": None,
        "layout": None,
        "requires_grad": None,
        "memory_format": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs["torch.empty_like"] = empty_like_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.empty_like' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.empty_like'.")

check_valid('torch.empty_like', generated_inputs['torch.empty_like'], lib="torch")
