
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def MaxUnpool1d_inputs():
    list_of_inputs = []

    # Test case 1
    kernel_size = (2,)
    stride = 2
    padding = 0
    input_tensor = torch.randn(1, 1, 4)
    pool = torch.nn.MaxPool1d(2, stride=2, return_indices=True)
    output, indices = pool(input_tensor)
    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "input": output.numpy(),
        "indices": indices.numpy(),
        "output_size": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 2: with output_size specified
    kernel_size = (2,)
    stride = 2
    padding = 0
    input_tensor = torch.randn(1, 1, 5)
    pool = torch.nn.MaxPool1d(2, stride=2, return_indices=True)
    output, indices = pool(input_tensor)
    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "input": output.numpy(),
        "indices": indices.numpy(),
        "output_size": tuple(input_tensor.size())
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 3: different kernel size and stride
    kernel_size = (3,)
    stride = 1
    padding = 0
    input_tensor = torch.randn(1, 1, 7)
    pool = torch.nn.MaxPool1d(3, stride=1, return_indices=True)
    output, indices = pool(input_tensor)
    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "input": output.numpy(),
        "indices": indices.numpy(),
        "output_size": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 4: with padding
    kernel_size = (2,)
    stride = 2
    padding = 1
    input_tensor = torch.randn(1, 1, 6)
    pool = torch.nn.MaxPool1d(2, stride=2, padding = 1, return_indices=True)
    output, indices = pool(input_tensor)
    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "input": output.numpy(),
        "indices": indices.numpy(),
        "output_size": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Test case 5: multiple channels
    kernel_size = (2,)
    stride = 2
    padding = 0
    input_tensor = torch.randn(1, 3, 4)
    pool = torch.nn.MaxPool1d(2, stride=2, return_indices=True)
    output, indices = pool(input_tensor)
    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "input": output.numpy(),
        "indices": indices.numpy(),
        "output_size": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.MaxUnpool1d_4"] = MaxUnpool1d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.MaxUnpool1d_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.MaxUnpool1d_4'.")

check_valid('torch.nn.MaxUnpool1d', generated_inputs['torch.nn.MaxUnpool1d_4'], lib="torch")
