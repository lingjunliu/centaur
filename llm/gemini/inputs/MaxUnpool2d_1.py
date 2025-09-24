
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def MaxUnpool2d_inputs():
    list_of_inputs = []

    # Case 1: Basic case with specified kernel_size and stride
    input = torch.randn(1, 1, 2, 2)
    indices = torch.randint(0, 4, (1, 1, 2, 2)).long()
    kernel_size = 2
    stride = 2
    padding = 0
    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "input": input.numpy(),
        "indices": indices.numpy(),
        "output_size": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Different kernel_size and stride values
    input = torch.randn(1, 3, 3, 3)
    indices = torch.randint(0, 9, (1, 3, 3, 3)).long()
    kernel_size = 3
    stride = 1
    padding = 0
    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "input": input.numpy(),
        "indices": indices.numpy(),
        "output_size": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: With padding
    input = torch.randn(1, 1, 2, 2)
    indices = torch.randint(0, 4, (1, 1, 2, 2)).long()
    kernel_size = 2
    stride = 2
    padding = 1
    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "input": input.numpy(),
        "indices": indices.numpy(),
        "output_size": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Using output_size
    input = torch.randn(1, 1, 2, 2)
    indices = torch.randint(0, 4, (1, 1, 2, 2)).long()
    kernel_size = 2
    stride = 2
    padding = 0
    output_size = (1, 1, 5, 5)
    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "input": input.numpy(),
        "indices": indices.numpy(),
        "output_size": output_size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Multiple channels
    input = torch.randn(1, 4, 2, 2)
    indices = torch.randint(0, 4, (1, 4, 2, 2)).long()
    kernel_size = 2
    stride = 2
    padding = 0
    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "input": input.numpy(),
        "indices": indices.numpy(),
        "output_size": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.MaxUnpool2d_1"] = MaxUnpool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

check_valid('torch.nn.MaxUnpool2d', generated_inputs['torch.nn.MaxUnpool2d_1'], lib="torch")
