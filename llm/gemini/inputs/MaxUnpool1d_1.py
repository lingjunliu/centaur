
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def MaxUnpool1d_inputs():
    list_of_inputs = []

    pool = torch.nn.MaxPool1d(2, stride=2, return_indices=True)
    unpool = torch.nn.MaxUnpool1d(2, stride=2)
    input_tensor = torch.tensor([[[1., 2, 3, 4, 5, 6, 7, 8]]])
    output, indices = pool(input_tensor)

    input_dict = {
        "kernel_size": 2,
        "stride": 2,
        "padding": 0,
        "input": output.numpy(),
        "indices": indices.numpy(),
        "output_size": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    pool = torch.nn.MaxPool1d(2, stride=2, return_indices=True)
    unpool = torch.nn.MaxUnpool1d(2, stride=2)
    input_tensor = torch.tensor([[[1., 2, 3, 4, 5, 6, 7, 8, 9]]])
    output, indices = pool(input_tensor)
    input_dict = {
        "kernel_size": 2,
        "stride": 2,
        "padding": 0,
        "input": output.numpy(),
        "indices": indices.numpy(),
        "output_size": tuple(input_tensor.size())
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    pool = torch.nn.MaxPool1d(kernel_size=3, stride=1, padding=1, return_indices=True)
    input_tensor = torch.randn(2, 3, 10)
    output, indices = pool(input_tensor)

    input_dict = {
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "input": output.numpy(),
        "indices": indices.numpy(),
        "output_size": tuple(input_tensor.size())
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    pool = torch.nn.MaxPool1d(kernel_size=3, stride=1, padding=1, return_indices=True)
    input_tensor = torch.randn(1, 1, 5)
    output, indices = pool(input_tensor)
    input_dict = {
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "input": output.numpy(),
        "indices": indices.numpy(),
        "output_size": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    pool = torch.nn.MaxPool1d(kernel_size=2, stride=1, padding=0, return_indices=True)
    input_tensor = torch.randn(1, 1, 7)
    output, indices = pool(input_tensor)
    input_dict = {
        "kernel_size": 2,
        "stride": 1,
        "padding": 0,
        "input": output.numpy(),
        "indices": indices.numpy(),
        "output_size": tuple(input_tensor.size())
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.MaxUnpool1d_1"] = MaxUnpool1d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.MaxUnpool1d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.MaxUnpool1d_1'.")

check_valid('torch.nn.MaxUnpool1d', generated_inputs['torch.nn.MaxUnpool1d_1'], lib="torch")
