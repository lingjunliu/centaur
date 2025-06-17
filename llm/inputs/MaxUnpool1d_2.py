
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def MaxUnpool1d_inputs():
    list_of_inputs = []

    # Case 1: Basic case with N, C, Hin
    input = torch.randn(1, 1, 4)
    indices = torch.tensor([[ [0, 2, 4, 6] ]])
    kernel_size = (2,)
    stride = (2,)
    padding = 0
    output_size = None

    input_dict = {
        "input": input.numpy(),
        "indices": indices.numpy(),
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "output_size": output_size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: C, Hin, different kernel_size and stride
    input = torch.randn(3, 5)
    indices = torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])
    kernel_size = (3,)
    stride = (1,)
    padding = 0
    output_size = None

    input_dict = {
        "input": input.numpy(),
        "indices": indices.numpy(),
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "output_size": output_size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: With output_size specified, different padding
    input = torch.randn(2, 3, 3)
    indices = torch.tensor([[[0, 1, 2]], [[0, 1, 2]]])
    kernel_size = (2,)
    stride = (1,)
    padding = 1
    output_size = (2, 3, 5)

    input_dict = {
        "input": input.numpy(),
        "indices": indices.numpy(),
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "output_size": (2, 3, 5) if output_size is not None else None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: With larger input
    input = torch.randn(4, 2, 7)
    indices = torch.randint(0, 7, (4, 2, 3))
    kernel_size = (3,)
    stride = (2,)
    padding = 0
    output_size = None

    input_dict = {
        "input": input.numpy(),
        "indices": indices.numpy(),
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "output_size": output_size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 5: float input
    input = torch.randn(1, 1, 4).float()
    indices = torch.tensor([[[0, 2, 4, 6]]])
    kernel_size = (2,)
    stride = (2,)
    padding = 0
    output_size = None

    input_dict = {
        "input": input.numpy(),
        "indices": indices.numpy(),
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "output_size": output_size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.MaxUnpool1d_2"] = MaxUnpool1d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.MaxUnpool1d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.MaxUnpool1d_2'.")

check_valid('torch.nn.MaxUnpool1d', generated_inputs['torch.nn.MaxUnpool1d_2'], lib="torch")
