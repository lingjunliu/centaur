
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def MaxUnpool2d_inputs():
    list_of_inputs = []

    # Case 1: Basic case with N, C, H_in, W_in
    input_tensor = torch.randn(1, 1, 2, 2)
    indices_tensor = torch.tensor([[[[0, 1], [2, 3]]]])
    input_dict = {
        "kernel_size": (2, 2),
        "stride": (2, 2),
        "padding": (0, 0),
        "input": input_tensor.numpy(),
        "indices": indices_tensor.numpy(),
        "output_size": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: C, H_in, W_in
    input_tensor = torch.randn(1, 2, 2)
    indices_tensor = torch.tensor([[[0, 1], [2, 3]]])
    input_dict = {
        "kernel_size": (2, 2),
        "stride": (2, 2),
        "padding": (0, 0),
        "input": input_tensor.numpy(),
        "indices": indices_tensor.numpy(),
        "output_size": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Different kernel_size, stride, and padding
    input_tensor = torch.randn(1, 1, 3, 3)
    indices_tensor = torch.tensor([[[[0, 1, 2], [3, 4, 5], [6, 7, 8]]]])
    input_dict = {
        "kernel_size": (3, 3),
        "stride": (1, 1),
        "padding": (1, 1),
        "input": input_tensor.numpy(),
        "indices": indices_tensor.numpy(),
        "output_size": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: With output_size specified
    input_tensor = torch.randn(1, 1, 2, 2)
    indices_tensor = torch.tensor([[[[0, 1], [2, 3]]]])
    input_dict = {
        "kernel_size": (2, 2),
        "stride": (2, 2),
        "padding": (0, 0),
        "input": input_tensor.numpy(),
        "indices": indices_tensor.numpy(),
        "output_size": (1, 1, 5, 5)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Multiple channels
    input_tensor = torch.randn(1, 3, 2, 2)
    indices_tensor = torch.tensor([[[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]]] )
    input_dict = {
        "kernel_size": (2, 2),
        "stride": (2, 2),
        "padding": (0, 0),
        "input": input_tensor.numpy(),
        "indices": indices_tensor.numpy(),
        "output_size": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.MaxUnpool2d_2"] = MaxUnpool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

check_valid('torch.nn.MaxUnpool2d', generated_inputs['torch.nn.MaxUnpool2d_2'], lib="torch")
