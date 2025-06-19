
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def MaxUnpool3d_inputs():
    list_of_inputs = []

    # Example 1: Basic usage with specified output size
    input1 = torch.randn(2, 3, 4, 5, 6)
    pool1 = torch.nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
    output1, indices1 = pool1(input1)

    input_dict1 = {
        "kernel_size": (2, 2, 2),
        "stride": (2, 2, 2),
        "padding": (0, 0, 0),
        "input": output1.numpy(),
        "indices": indices1.numpy(),
        "output_size": input1.shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    return list_of_inputs

generated_inputs["torch.nn.MaxUnpool3d_2"] = MaxUnpool3d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.MaxUnpool3d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.MaxUnpool3d_2'.")

check_valid('torch.nn.MaxUnpool3d', generated_inputs['torch.nn.MaxUnpool3d_2'], lib="torch")
